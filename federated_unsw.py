from __future__ import annotations

import argparse
import copy
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset


@dataclass
class Config:
    # knobs for one file; CLI can override most of these
    features_csv: str = "Dataset.csv"  # flow feats
    labels_csv: str = "Label.csv"  # class col, same row order as feats

    num_clients: int = 5  # fake "sites" / shards
    global_rounds: int = 8  # how many merge cycles (srv <-> clients)
    local_epochs: int = 2  # full passes over local data per round
    batch_size: int = 512  # mini-batch for SGD
    lr: float = 1e-3  # Adam step size

    test_size: float = 0.2  # frac held out for eval only
    random_seed: int = 42  # repro splits + init
    device: str = "cpu"  # or cuda if you set it

    # label poisoning on ONE client's train shard (-1 = off)
    poison_client: int = -1  # 0 .. num_clients-1
    poison_fraction: float = 0.0  # frac of that client's labels touched
    poison_mode: str = "target"  # target | random_wrong | cycle
    poison_target: int = 0  # for mode=target: wrong label to write


def set_seed(seed: int) -> None:
    # lock rngs so runs match (debug/compare)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_features_and_labels(cfg: Config) -> tuple[np.ndarray, np.ndarray]:
    # resolve paths next to this script
    base = os.path.dirname(os.path.abspath(__file__))
    feat_path = os.path.join(base, cfg.features_csv)
    lab_path = os.path.join(base, cfg.labels_csv)

    X = pd.read_csv(feat_path, dtype=np.float32).values  # (N, 76) approx
    y = pd.read_csv(lab_path)["Label"].values.astype(np.int64)  # int labels

    if len(X) != len(y):
        raise ValueError(f"Row count mismatch: features {len(X)} vs labels {len(y)}.")

    # nn hates nan/inf -> zero them out (simple fix)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    return X, y


def split_train_clients(
    X: np.ndarray,
    y: np.ndarray,
    cfg: Config,
) -> tuple[StandardScaler, list[tuple[np.ndarray, np.ndarray]], np.ndarray, np.ndarray]:
    # stratify: keep class mix similar in train vs test (imbalanced ds)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=cfg.test_size,
        random_state=cfg.random_seed,
        stratify=y,
    )

    scaler = StandardScaler()  # zero mean unit var per col
    X_train = scaler.fit_transform(X_train).astype(np.float32)  # fit on train only
    X_test = scaler.transform(X_test).astype(np.float32)  # same transform for test

    rng = np.random.default_rng(cfg.random_seed)
    perm = rng.permutation(len(X_train))
    X_train, y_train = X_train[perm], y_train[perm]  # shuffle before shard cut

    n = len(X_train)
    cuts = np.array_split(np.arange(n), cfg.num_clients)  # ~equal sized shards
    clients: list[tuple[np.ndarray, np.ndarray]] = []
    for idx in cuts:
        clients.append((X_train[idx], y_train[idx]))  # one (X,y) per client

    return scaler, clients, X_test, y_test


def poison_client_labels(
    y: np.ndarray,
    fraction: float,
    num_classes: int,
    mode: str,
    target_class: int,
    rng: np.random.Generator,
) -> np.ndarray:
    # flip a subset of labels on one client (untargeted / targeted toy attack)
    y = y.copy()
    n = len(y)
    n_bad = int(np.floor(float(fraction) * n))
    if n_bad <= 0:
        return y
    idx = rng.choice(n, size=n_bad, replace=False)
    if mode == "target":
        y[idx] = int(target_class) % num_classes
    elif mode == "random_wrong":
        for i in idx:
            true = int(y[i])
            choices = [c for c in range(num_classes) if c != true]
            y[i] = int(rng.choice(choices))
    elif mode == "cycle":
        y[idx] = (y[idx].astype(np.int64) + 1) % num_classes
    else:
        raise ValueError(f"Unknown poison_mode: {mode}")
    return y


def maybe_poison_one_client(
    clients: list[tuple[np.ndarray, np.ndarray]],
    cfg: Config,
    num_classes: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    if cfg.poison_client < 0 or cfg.poison_fraction <= 0:
        return clients
    k = cfg.poison_client
    if k < 0 or k >= len(clients):
        raise ValueError(f"poison_client must be in [0, {len(clients)-1}], got {k}")
    rng = np.random.default_rng(cfg.random_seed)
    out = list(clients)
    Xk, yk = out[k]
    y_new = poison_client_labels(
        yk,
        cfg.poison_fraction,
        num_classes,
        cfg.poison_mode,
        cfg.poison_target,
        rng,
    )
    n_bad = int(np.floor(cfg.poison_fraction * len(yk)))
    print(
        f"[poison] client_index={k} (of {len(clients)}): "
        f"n_poisoned={n_bad}/{len(yk)} ({100.0 * n_bad / max(len(yk), 1):.2f}%) "
        f"mode={cfg.poison_mode} target={cfg.poison_target}"
    )
    out[k] = (Xk, y_new)
    return out


class MLPClassifier(nn.Module):
    # tiny MLP: all clients share this arch so we can avg weights layer-wise

    def __init__(self, input_dim: int, num_classes: int) -> None:
        super().__init__()
        hidden = 128
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),  # nonlin
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, num_classes),  # logits out (no softmax)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, feats) -> logits (batch, n_classes); CE loss wants raw logits
        return self.net(x)


def get_model_copy(model: nn.Module, device: str) -> nn.Module:
    # deepcopy so each client has own tensors (no shared mem bugs)
    clone = copy.deepcopy(model)
    clone.to(device)
    return clone


def train_one_client(
    model: nn.Module,
    X_local: np.ndarray,
    y_local: np.ndarray,
    cfg: Config,
) -> None:
    # local only: one client's shard, simulates on-device train
    device = torch.device(cfg.device)
    model.train()  # dropout/bn train mode (we have neither but habit)

    ds = TensorDataset(
        torch.from_numpy(X_local),
        torch.from_numpy(y_local),
    )
    loader = DataLoader(ds, batch_size=cfg.batch_size, shuffle=True)  # shuffle each epoch

    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)  # fresh opt each round (FedAvg usual)
    loss_fn = nn.CrossEntropyLoss()  # multiclass CE on logits

    for _ in range(cfg.local_epochs):
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            opt.zero_grad()
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()  # backprop
            opt.step()  # wt update


@torch.no_grad()
def predict_labels(model: nn.Module, X: np.ndarray, y: np.ndarray, device: str, batch_size: int) -> tuple[np.ndarray, np.ndarray]:
    # eval pass -> return y_true + y_pred
    model.eval()
    dev = torch.device(device)
    ds = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
    all_preds: list[np.ndarray] = []
    all_true: list[np.ndarray] = []
    for xb, yb in loader:
        xb = xb.to(dev)
        pred = model(xb).argmax(dim=1).cpu().numpy()  # class id = max logit
        all_preds.append(pred)
        all_true.append(yb.numpy())
    return np.concatenate(all_true), np.concatenate(all_preds)


def compute_tp_fp_fn_tn(cm: np.ndarray) -> dict[int, dict[str, int]]:
    # one-vs-rest stats from full confusion matrix
    totals = cm.sum()
    metrics: dict[int, dict[str, int]] = {}
    for cls_idx in range(cm.shape[0]):
        tp = int(cm[cls_idx, cls_idx])
        fp = int(cm[:, cls_idx].sum() - tp)
        fn = int(cm[cls_idx, :].sum() - tp)
        tn = int(totals - tp - fp - fn)
        metrics[cls_idx] = {"TP": tp, "FP": fp, "FN": fn, "TN": tn}
    return metrics


def evaluate_model(
    model: nn.Module,
    X: np.ndarray,
    y: np.ndarray,
    device: str,
    batch_size: int,
    num_classes: int,
    print_prefix: str = "",
    print_report: bool = False,
) -> dict[str, object]:
    # central eval helper (acc + cm + per-class TP/FP/FN/TN)
    y_true, y_pred = predict_labels(model, X, y, device, batch_size)
    labels = np.arange(num_classes)
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    class_stats = compute_tp_fp_fn_tn(cm)
    report_dict = classification_report(
        y_true,
        y_pred,
        labels=labels,
        digits=4,
        zero_division=0,
        output_dict=True,
    )
    acc = float(report_dict["accuracy"])

    if print_prefix:
        print(f"{print_prefix} accuracy: {acc:.4f}")
    else:
        print(f"Accuracy: {acc:.4f}")
    print("Confusion matrix:")
    print(cm)
    print("Per-class TP/FP/FN/TN:")
    for cls_idx in labels:
        s = class_stats[int(cls_idx)]
        print(f"Class {int(cls_idx)} -> TP:{s['TP']} FP:{s['FP']} FN:{s['FN']} TN:{s['TN']}")
    if print_report:
        print("\nClassification report (test set):")
        print(classification_report(y_true, y_pred, labels=labels, digits=4, zero_division=0))

    return {
        "accuracy": acc,
        "confusion_matrix": cm,
        "per_class_stats": class_stats,
        "report_dict": report_dict,
    }


def weighted_average_state_dicts(
    state_dicts: list[dict[str, torch.Tensor]],
    weights: list[float],
) -> dict[str, torch.Tensor]:
    # FedAvg: theta = sum_k w_k * theta_k; w_k = n_k/sum(n), caller normalizes
    if len(state_dicts) != len(weights):
        raise ValueError("Need one weight per client state dict.")
    if abs(sum(weights) - 1.0) > 1e-5:
        raise ValueError("Weights should sum to 1 for FedAvg.")

    keys = state_dicts[0].keys()
    out: dict[str, torch.Tensor] = {}
    for key in keys:
        acc = torch.zeros_like(state_dicts[0][key], dtype=torch.float32)
        for w, sd in zip(weights, state_dicts):
            acc += float(w) * sd[key].float()  # wt sum of this layer across clients
        out[key] = acc.to(dtype=state_dicts[0][key].dtype)  # back to orig dtype (fp32/fp64)
    return out


def run_federated_loop(cfg: Config, *, verbose: bool = True, return_eval: bool = False) -> dict[str, float | object]:
    set_seed(cfg.random_seed)
    device = cfg.device

    X, y = load_features_and_labels(cfg)
    num_classes = int(y.max()) + 1  # assumes labels 0..C-1
    input_dim = X.shape[1]

    _, clients, X_test, y_test = split_train_clients(X, y, cfg)
    clients = maybe_poison_one_client(clients, cfg, num_classes)

    global_model = MLPClassifier(input_dim, num_classes).to(device)  # srv-side model
    sample_counts = np.array([len(c[0]) for c in clients], dtype=np.float64)
    total_train = sample_counts.sum()
    mix_weights = (sample_counts / total_train).tolist()  # w_k for avg

    if verbose:
        print(f"Samples per client: {sample_counts.astype(int).tolist()}")
        print(f"FedAvg mixing weights (sum={sum(mix_weights):.6f}): {[round(w, 4) for w in mix_weights]}")
        print(f"Test set size: {len(X_test)}, global rounds: {cfg.global_rounds}, local epochs: {cfg.local_epochs}")

    for rnd in range(1, cfg.global_rounds + 1):
        client_states: list[dict[str, torch.Tensor]] = []  # each client's sd after local train

        for client_X, client_y in clients:
            local_model = get_model_copy(global_model, device)  # start from curr global
            train_one_client(local_model, client_X, client_y, cfg)
            # stash sd on CPU; cheap if many clients / big model
            client_states.append({k: v.detach().cpu().clone() for k, v in local_model.state_dict().items()})

        aggregated = weighted_average_state_dicts(client_states, mix_weights)
        global_model.load_state_dict(aggregated)  # srv update

        if verbose:
            print(f"\nRound {rnd:02d}/{cfg.global_rounds}")
            evaluate_model(
                model=global_model,
                X=X_test,
                y=y_test,
                device=device,
                batch_size=cfg.batch_size,
                num_classes=num_classes,
                print_prefix="Global test",
                print_report=False,
            )

    if verbose:
        print("\nFinal evaluation")
    final_eval = evaluate_model(
        model=global_model,
        X=X_test,
        y=y_test,
        device=device,
        batch_size=cfg.batch_size,
        num_classes=num_classes,
        print_prefix="Final global test" if verbose else "",
        print_report=verbose,
    )
    weighted = final_eval["report_dict"]["weighted avg"]
    out: dict[str, float | object] = {
        "accuracy": float(final_eval["accuracy"]),
        "precision": float(weighted["precision"]),
        "recall": float(weighted["recall"]),
        "f1-score": float(weighted["f1-score"]),
    }
    if return_eval:
        out["confusion_matrix"] = final_eval["confusion_matrix"]
        out["per_class_stats"] = final_eval["per_class_stats"]
        out["num_classes"] = num_classes
    return out


def run_experiments(cfg: Config, clients_list: list[int], rounds_list: list[int], results_csv: str) -> pd.DataFrame:
    # grid search over (clients, rounds) and save compact metrics
    rows: list[dict[str, float | int]] = []
    for clients in clients_list:
        for rounds in rounds_list:
            print("\n" + "=" * 70)
            print(f"Running experiment: clients={clients}, rounds={rounds}")
            exp_cfg = copy.deepcopy(cfg)
            exp_cfg.num_clients = int(clients)
            exp_cfg.global_rounds = int(rounds)
            metrics = run_federated_loop(exp_cfg)
            rows.append(
                {
                    "clients": int(clients),
                    "rounds": int(rounds),
                    "accuracy": metrics["accuracy"],
                    "precision": metrics["precision"],
                    "recall": metrics["recall"],
                    "f1-score": metrics["f1-score"],
                }
            )

    df = pd.DataFrame(rows)
    # append by default if file exists; keep old rows unless user deletes file manually
    if os.path.exists(results_csv):
        prev_df = pd.read_csv(results_csv)
        out_df = pd.concat([prev_df, df], ignore_index=True)
    else:
        out_df = df
    out_df.to_csv(results_csv, index=False)
    print(f"\nSaved experiment results (appended) -> {results_csv}")
    print(df.to_string(index=False))
    return df


def parse_int_list(raw: str) -> list[int]:
    # "3,5,10" -> [3, 5, 10]
    vals = [int(x.strip()) for x in raw.split(",") if x.strip()]
    if not vals:
        raise ValueError("List args cannot be empty.")
    if any(v <= 0 for v in vals):
        raise ValueError("List args must contain positive ints.")
    return vals


def parse_args() -> Config:
    p = argparse.ArgumentParser(description="FedAvg FL on UNSW/CIC CSVs.")
    p.add_argument("--clients", type=int, default=5, help="N clients.")
    p.add_argument("--rounds", type=int, default=8, help="Global rounds.")
    p.add_argument("--local-epochs", type=int, default=2, help="Local epochs per client per round.")
    p.add_argument("--batch-size", type=int, default=512, help="Batch size.")
    p.add_argument("--lr", type=float, default=1e-3, help="Adam lr.")
    p.add_argument("--device", type=str, default=None, help='cpu or cuda; default auto.')
    p.add_argument("--clients-list", type=str, default="", help='CSV ints, eg "3,5,10".')
    p.add_argument("--rounds-list", type=str, default="", help='CSV ints, eg "5,10".')
    p.add_argument("--results-csv", type=str, default="experiment_results.csv", help="Where to save grid results.")
    p.add_argument("--poison-client", type=int, default=-1, help="0-based client index to poison; -1 disables.")
    p.add_argument("--poison-fraction", type=float, default=0.0, help="Fraction of that client's labels corrupted.")
    p.add_argument(
        "--poison-mode",
        type=str,
        default="target",
        choices=["target", "random_wrong", "cycle"],
        help="target=fixed class; random_wrong!=y; cycle=(y+1)%C.",
    )
    p.add_argument("--poison-target", type=int, default=0, help="Label used when poison_mode=target.")
    args = p.parse_args()

    cfg = Config(
        num_clients=args.clients,
        global_rounds=args.rounds,
        local_epochs=args.local_epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        poison_client=args.poison_client,
        poison_fraction=args.poison_fraction,
        poison_mode=args.poison_mode,
        poison_target=args.poison_target,
    )
    if args.device:
        cfg.device = args.device
    else:
        cfg.device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg._clients_list = parse_int_list(args.clients_list) if args.clients_list else []  # type: ignore[attr-defined]
    cfg._rounds_list = parse_int_list(args.rounds_list) if args.rounds_list else []  # type: ignore[attr-defined]
    cfg._results_csv = args.results_csv  # type: ignore[attr-defined]
    return cfg


if __name__ == "__main__":
    cfg = parse_args()
    clients_list = getattr(cfg, "_clients_list", [])
    rounds_list = getattr(cfg, "_rounds_list", [])
    # default exp set: same clients, more rounds (1 and 5) -> 4 runs total
    if not clients_list and not rounds_list:
        clients_list = [3, 5]
        rounds_list = [1, 5]
    if clients_list and rounds_list:
        run_experiments(cfg, clients_list, rounds_list, getattr(cfg, "_results_csv", "experiment_results.csv"))
    else:
        run_federated_loop(cfg)
