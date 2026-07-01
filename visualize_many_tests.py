#!/usr/bin/env python3
"""Charts from many_tests_ranked_all.csv -> figures/"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

CSV = Path(__file__).parent / "many_tests_ranked_all.csv"
OUT = Path(__file__).parent / "figures"
OUT.mkdir(exist_ok=True)

sns.set_theme(style="whitegrid", font_scale=1.05)
df = pd.read_csv(CSV)
df = df.drop(columns=["conclusion"], errors="ignore")

df["poison_pct"] = (df["poison_fraction"] * 100).astype(int)
df["scenario"] = df.apply(
    lambda r: "clean" if r["poison_fraction"] == 0 else f"{int(r['poison_fraction']*100)}% {r['poison_mode']}",
    axis=1,
)
scenario_order = ["clean", "25% target", "40% target", "40% random_wrong"]
df["scenario"] = pd.Categorical(df["scenario"], categories=scenario_order, ordered=True)

# best row per (clients, scenario) for cleaner plots
best = (
    df.sort_values(["accuracy", "f1-score"], ascending=False)
    .groupby(["clients", "scenario"], observed=True)
    .first()
    .reset_index()
)


def save(fig: plt.Figure, name: str) -> None:
    path = OUT / name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("wrote", path)


# 1) Accuracy by clients x scenario (grouped bars)
fig, ax = plt.subplots(figsize=(10, 5.5))
sns.barplot(
    data=best,
    x="clients",
    y="accuracy",
    hue="scenario",
    hue_order=scenario_order,
    ax=ax,
    palette="Set2",
)
ax.set_xlabel("Number of federated clients")
ax.set_ylabel("Test accuracy (best per setting)")
ax.set_title("Best accuracy: clients vs poisoning scenario")
ax.set_ylim(0.84, 0.94)
ax.legend(title="Scenario", bbox_to_anchor=(1.02, 1), loc="upper left")
save(fig, "01_accuracy_by_clients_scenario.png")


# 2) Heatmap: mean accuracy over rounds/local_epochs
pivot = (
    df.groupby(["clients", "poison_pct"], observed=True)["accuracy"]
    .max()
    .unstack(fill_value=np.nan)
)
fig, ax = plt.subplots(figsize=(7, 4.5))
sns.heatmap(pivot, annot=True, fmt=".3f", cmap="YlGnBu", ax=ax, vmin=0.85, vmax=0.93)
ax.set_title("Max accuracy: clients vs poison % (all runs)")
ax.set_xlabel("Poisoned label fraction (%)")
ax.set_ylabel("Clients")
save(fig, "02_heatmap_clients_poison_pct.png")


# 3) Effect of global rounds (clean only)
clean = df[df["poison_fraction"] == 0].copy()
fig, ax = plt.subplots(figsize=(8, 5))
for c in sorted(clean["clients"].unique()):
    sub = clean[clean["clients"] == c]
    agg = sub.groupby("rounds")["accuracy"].max().reset_index()
    ax.plot(agg["rounds"], agg["accuracy"], marker="o", linewidth=2, label=f"{c} clients")
ax.set_xlabel("Global rounds")
ax.set_ylabel("Max test accuracy")
ax.set_title("Training length vs accuracy (no poisoning)")
ax.legend()
ax.set_ylim(0.90, 0.93)
save(fig, "03_rounds_vs_accuracy_clean.png")


# 4) Poison drop: clean vs 40% target (best per config)
def poison_drop(poison_frac: float, mode: str = "target") -> pd.DataFrame:
    rows = []
    for c in sorted(df["clients"].unique()):
        clean_acc = df[(df["clients"] == c) & (df["poison_fraction"] == 0)]["accuracy"].max()
        bad = df[
            (df["clients"] == c)
            & (df["poison_fraction"] == poison_frac)
            & (df["poison_mode"] == mode)
        ]["accuracy"].max()
        rows.append({"clients": c, "clean": clean_acc, "poisoned": bad, "drop": clean_acc - bad})
    return pd.DataFrame(rows)


drop_t = poison_drop(0.4, "target")
x = np.arange(len(drop_t))
w = 0.35
fig, ax = plt.subplots(figsize=(8, 5))
ax.bar(x - w / 2, drop_t["clean"], w, label="Clean (best)", color="#2ecc71")
ax.bar(x + w / 2, drop_t["poisoned"], w, label="40% target poison (best)", color="#e74c3c")
ax.set_xticks(x)
ax.set_xticklabels(drop_t["clients"].astype(str))
ax.set_xlabel("Number of clients")
ax.set_ylabel("Accuracy")
ax.set_title("Impact of 40% label poisoning (client 0)")
ax.legend()
ax.set_ylim(0.84, 0.94)
save(fig, "04_clean_vs_40pct_target.png")


# 5) Metrics comparison (top 15 runs)
top = df.nlargest(15, "accuracy").copy()
top["label"] = top.apply(
    lambda r: f"c={int(r['clients'])} r={int(r['rounds'])} e={int(r['local_epochs'])} {r['scenario']}",
    axis=1,
)
fig, ax = plt.subplots(figsize=(10, 7))
y = np.arange(len(top))
ax.barh(y, top["accuracy"], color="#3498db", label="accuracy")
ax.barh(y - 0.25, top["f1-score"], height=0.2, color="#9b59b6", label="f1")
ax.set_yticks(y)
ax.set_yticklabels(top["label"], fontsize=8)
ax.set_xlabel("Score")
ax.set_title("Top 15 runs by accuracy")
ax.legend(loc="lower right")
ax.set_xlim(0.84, 0.94)
ax.invert_yaxis()
save(fig, "05_top15_runs.png")


# 6) Local epochs effect (facet by clients, clean)
fig, axes = plt.subplots(1, 4, figsize=(14, 4), sharey=True)
for ax, c in zip(axes, sorted(clean["clients"].unique())):
    sub = clean[clean["clients"] == c]
    sns.barplot(data=sub, x="rounds", y="accuracy", hue="local_epochs", ax=ax, palette="muted")
    ax.set_title(f"{c} clients")
    ax.set_ylim(0.90, 0.93)
axes[0].set_ylabel("Accuracy")
fig.suptitle("Rounds and local epochs (clean runs only)", y=1.02)
fig.tight_layout()
save(fig, "06_local_epochs_clean_facets.png")

print("\nAll figures in:", OUT.resolve())
