#!/usr/bin/env python3
"""Train key runs, cache confusion matrix + TP/FP/FN/TN, build PNG + interactive HTML."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.io import to_html
from plotly.subplots import make_subplots

from federated_unsw import Config, run_federated_loop

ROOT = Path(__file__).parent
FIG = ROOT / "figures"
CACHE = FIG / "confusion_cache.json"
OUT_HTML = FIG / "confusion_dashboard.html"

# representative configs (10 rounds, 2 local epochs — strong baseline from sweep)
KEY_SCENARIOS: list[tuple[str, dict]] = [
    ("5 clients · clean", {"num_clients": 5, "poison_fraction": 0.0}),
    ("5 clients · 40% target poison", {"num_clients": 5, "poison_fraction": 0.4, "poison_mode": "target"}),
    ("3 clients · 25% target poison", {"num_clients": 3, "poison_fraction": 0.25, "poison_mode": "target"}),
    ("1 client · clean", {"num_clients": 1, "poison_fraction": 0.0}),
    ("1 client · 40% target poison", {"num_clients": 1, "poison_fraction": 0.4, "poison_mode": "target"}),
]


def run_scenario(label: str, overrides: dict) -> dict:
    cfg = Config(
        num_clients=overrides.get("num_clients", 5),
        global_rounds=10,
        local_epochs=2,
        batch_size=1024,
        device="cpu",
        poison_client=0 if overrides.get("poison_fraction", 0) > 0 else -1,
        poison_fraction=overrides.get("poison_fraction", 0.0),
        poison_mode=overrides.get("poison_mode", "target"),
        poison_target=0,
    )
    print(f"\n>>> {label}")
    result = run_federated_loop(cfg, verbose=False, return_eval=True)
    cm = np.asarray(result["confusion_matrix"])
    stats = {int(k): v for k, v in result["per_class_stats"].items()}  # type: ignore
    return {
        "label": label,
        "accuracy": float(result["accuracy"]),
        "confusion_matrix": cm.tolist(),
        "per_class_stats": stats,
    }


def load_or_build_cache(force: bool = False) -> list[dict]:
    if CACHE.exists() and not force:
        print("Loading cached confusion data:", CACHE)
        return json.loads(CACHE.read_text())

    FIG.mkdir(exist_ok=True)
    records = [run_scenario(label, kw) for label, kw in KEY_SCENARIOS]
    CACHE.write_text(json.dumps(records, indent=2))
    print("Saved cache:", CACHE)
    return records


def stats_to_df(stats: dict) -> pd.DataFrame:
    rows = []
    for cls_idx in sorted(stats.keys()):
        s = stats[cls_idx]
        rows.append(
            {
                "class": f"Class {cls_idx}",
                "TP": s["TP"],
                "FP": s["FP"],
                "FN": s["FN"],
                "TN": s["TN"],
            }
        )
    return pd.DataFrame(rows)


def save_png_charts(records: list[dict]) -> None:
    for rec in records:
        slug = rec["label"].replace(" ", "_").replace("·", "").replace("%", "pct")
        cm = np.array(rec["confusion_matrix"])
        n = cm.shape[0]
        labels = [str(i) for i in range(n)]

        # confusion matrix heatmap
        fig, ax = plt.subplots(figsize=(8, 7))
        im = ax.imshow(cm, cmap="Blues")
        ax.set_xticks(range(n), labels)
        ax.set_yticks(range(n), labels)
        ax.set_xlabel("Predicted label")
        ax.set_ylabel("True label")
        ax.set_title(f"Confusion matrix — {rec['label']}\naccuracy={rec['accuracy']:.4f}")
        for i in range(n):
            for j in range(n):
                val = cm[i, j]
                if val > 0:
                    ax.text(j, i, str(val), ha="center", va="center", fontsize=7, color="black")
        fig.colorbar(im, ax=ax, fraction=0.046)
        fig.tight_layout()
        p = FIG / f"cm_{slug}.png"
        fig.savefig(p, dpi=150)
        plt.close(fig)

        # TP / FP / FN / TN per class
        df = stats_to_df(rec["per_class_stats"])
        x = np.arange(len(df))
        w = 0.2
        fig, ax = plt.subplots(figsize=(12, 5))
        for i, col in enumerate(["TP", "FP", "FN", "TN"]):
            ax.bar(x + (i - 1.5) * w, df[col], width=w, label=col)
        ax.set_xticks(x, df["class"], rotation=45, ha="right")
        ax.set_ylabel("Count")
        ax.set_title(f"Per-class TP / FP / FN / TN — {rec['label']}")
        ax.legend()
        fig.tight_layout()
        p = FIG / f"tpfn_{slug}.png"
        fig.savefig(p, dpi=150)
        plt.close(fig)
        print("wrote", p.name)


def build_confusion_sections(records: list[dict], *, include_plotly_js: bool = False) -> str:
    """Return HTML sections for confusion matrix page (no full document wrapper)."""
    parts: list[str] = []
    plotly_config = {"displayModeBar": True, "responsive": True}

    for i, rec in enumerate(records):
        cm = np.array(rec["confusion_matrix"])
        n = cm.shape[0]
        labels = [str(c) for c in range(n)]
        df_stats = stats_to_df(rec["per_class_stats"])

        fig_cm = px.imshow(
            cm,
            x=labels,
            y=labels,
            text_auto=True,
            color_continuous_scale="Blues",
            labels=dict(x="Predicted", y="True", color="Count"),
            title=f"Confusion matrix — {rec['label']} (acc={rec['accuracy']:.4f})",
            aspect="auto",
        )
        fig_cm.update_layout(yaxis=dict(autorange="reversed"), height=420, autosize=True)

        fig_stats = go.Figure()
        colors = {"TP": "#27ae60", "FP": "#e74c3c", "FN": "#f39c12", "TN": "#3498db"}
        for metric in ["TP", "FP", "FN", "TN"]:
            fig_stats.add_trace(
                go.Bar(name=metric, x=df_stats["class"], y=df_stats[metric], marker_color=colors[metric])
            )
        fig_stats.update_layout(
            barmode="group",
            title=f"Per-class TP / FP / FN / TN — {rec['label']}",
            xaxis_title="Class",
            yaxis_title="Count",
            height=420,
            autosize=True,
            legend=dict(orientation="h", y=1.12),
        )

        use_js = include_plotly_js and i == 0
        cm_div = to_html(fig_cm, full_html=False, include_plotlyjs="cdn" if use_js else False, config=plotly_config)
        stats_div = to_html(fig_stats, full_html=False, include_plotlyjs=False, config=plotly_config)
        parts.append(
            f'<section class="chart-block confusion-block"><h2>{rec["label"]}</h2>'
            f'<div class="chart">{cm_div}</div>'
            f'<div class="chart">{stats_div}</div></section>'
        )

    intro = """
    <div class="info-box">
      <strong>Confusion matrix page.</strong>
      Rows = true label, columns = predicted label (diagonal = correct).
      TP / FP / FN / TN are one-vs-rest counts per class.
      Scenarios: 10 global rounds, 2 local epochs; poison on client 0 training labels only.
    </div>
    """
    return intro + "".join(parts)


def load_confusion_records() -> list[dict]:
    if not CACHE.exists():
        raise FileNotFoundError(
            f"Missing {CACHE}. Run: python3 visualize_confusion.py"
        )
    return json.loads(CACHE.read_text())


def build_interactive_html(records: list[dict]) -> None:
    body = build_confusion_sections(records, include_plotly_js=True)
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>Confusion matrix dashboard</title>
  <style>
    body {{ font-family: system-ui, sans-serif; margin: 0; background: #eef1f5; color: #1a1a2e; }}
    header {{ background: #2c3e50; color: #fff; padding: 20px 28px; }}
    header h1 {{ margin: 0 0 6px; font-size: 1.4rem; }}
    header p {{ margin: 0; opacity: 0.9; font-size: 0.9rem; max-width: 720px; line-height: 1.5; }}
    main {{ max-width: 1000px; margin: 0 auto; padding: 20px 16px 40px; }}
    .chart-block {{
      background: #fff; border-radius: 10px; padding: 16px; margin-bottom: 24px;
      box-shadow: 0 2px 8px rgba(0,0,0,0.06);
    }}
    .chart-block h2 {{
      margin: 0 0 12px; font-size: 1.05rem; color: #34495e;
      border-bottom: 2px solid #9b59b6; padding-bottom: 6px;
    }}
    .info-box {{
      background: #fff; border-radius: 8px; padding: 12px 16px; margin-bottom: 20px;
      font-size: 0.9rem; line-height: 1.6;
    }}
  </style>
</head>
<body>
  <header>
    <h1>Confusion matrix &amp; TP / FP / FN / TN</h1>
    <p>Standalone confusion dashboard (also embedded in many_tests_dashboard.html).</p>
  </header>
  <main>{body}</main>
</body>
</html>
"""
    OUT_HTML.write_text(html, encoding="utf-8")
    print("Wrote", OUT_HTML)


def main() -> None:
    FIG.mkdir(exist_ok=True)
    records = load_or_build_cache(force=False)
    save_png_charts(records)
    build_interactive_html(records)


if __name__ == "__main__":
    main()
