#!/usr/bin/env python3
"""Interactive Plotly dashboard — one chart per section, responsive layout."""

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.io import to_html

CSV = Path(__file__).parent / "many_tests_ranked_all.csv"
OUT_HTML = Path(__file__).parent / "figures" / "many_tests_dashboard.html"
OUT_HTML_ROOT = Path(__file__).parent / "many_tests_dashboard.html"
OUT_HTML_PUBLIC = Path(__file__).parent / "public" / "index.html"

SCENARIO_ORDER = ["clean", "25% target", "40% target", "40% random_wrong"]
PLOTLY_CONFIG = {"displayModeBar": True, "responsive": True}
LAYOUT = dict(
    template="plotly_white",
    autosize=True,
    margin=dict(l=60, r=40, t=60, b=60),
    font=dict(size=13),
)


def load_df() -> pd.DataFrame:
    df = pd.read_csv(CSV)
    df = df.drop(columns=["conclusion"], errors="ignore")
    df["scenario"] = df.apply(
        lambda r: "clean"
        if r["poison_fraction"] == 0
        else f"{int(r['poison_fraction'] * 100)}% {r['poison_mode']}",
        axis=1,
    )
    df["scenario"] = pd.Categorical(df["scenario"], categories=SCENARIO_ORDER, ordered=True)
    df["config"] = df.apply(
        lambda r: (
            f"run {int(r['run_id'])} · {int(r['clients'])} clients · "
            f"{int(r['rounds'])} rounds · {int(r['local_epochs'])} local epochs · {r['scenario']}"
        ),
        axis=1,
    )
    return df


def fig_to_div(fig: go.Figure, include_js: bool) -> str:
    fig.update_layout(**LAYOUT, height=420)
    return to_html(
        fig,
        full_html=False,
        include_plotlyjs="cdn" if include_js else False,
        config=PLOTLY_CONFIG,
        div_id=None,
    )


def chart_accuracy_by_scenario(best: pd.DataFrame) -> go.Figure:
    fig = px.bar(
        best,
        x="clients",
        y="accuracy",
        color="scenario",
        barmode="group",
        category_orders={"scenario": SCENARIO_ORDER},
        labels={"clients": "Federated clients", "accuracy": "Test accuracy", "scenario": "Scenario"},
        title="Best accuracy by client count and poisoning scenario",
        text=best["accuracy"].map(lambda v: f"{v:.3f}"),
    )
    fig.update_traces(textposition="outside")
    fig.update_yaxes(range=[0.84, 0.94])
    return fig


def chart_heatmap(df: pd.DataFrame) -> go.Figure:
    pivot = df.groupby(["clients", "poison_fraction"], observed=False)["accuracy"].max().reset_index()
    pivot["poison_pct"] = (pivot["poison_fraction"] * 100).astype(int).astype(str) + "%"
    pivot = pivot.pivot(index="clients", columns="poison_pct", values="accuracy")
    col_order = sorted(pivot.columns, key=lambda x: int(x.replace("%", "")))
    pivot = pivot[col_order]
    fig = px.imshow(
        pivot,
        text_auto=".3f",
        color_continuous_scale="YlGnBu",
        zmin=0.85,
        zmax=0.94,
        labels=dict(x="Poison % (client 0)", y="Clients", color="Max accuracy"),
        title="Max accuracy: clients vs poison level",
        aspect="auto",
    )
    return fig


def chart_scatter_all(df: pd.DataFrame) -> go.Figure:
    fig = px.scatter(
        df,
        x="f1-score",
        y="accuracy",
        color="clients",
        symbol="scenario",
        hover_name="config",
        labels={"f1-score": "F1-score", "accuracy": "Test accuracy", "clients": "Clients"},
        title="All 64 runs (hover for full configuration)",
        color_discrete_sequence=px.colors.qualitative.Set1,
    )
    fig.update_traces(marker=dict(size=11, opacity=0.85))
    fig.update_layout(legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0))
    return fig


def chart_rounds_clean(clean: pd.DataFrame) -> go.Figure:
    agg = clean.groupby(["clients", "rounds"], observed=False)["accuracy"].max().reset_index()
    fig = px.line(
        agg,
        x="rounds",
        y="accuracy",
        color="clients",
        markers=True,
        labels={"rounds": "Global rounds", "accuracy": "Max test accuracy", "clients": "Clients"},
        title="Effect of global rounds (clean runs only)",
    )
    fig.update_yaxes(range=[0.90, 0.93])
    return fig


def chart_clean_vs_poison(df: pd.DataFrame) -> go.Figure:
    rows = []
    for c in sorted(df["clients"].unique()):
        clean_acc = df[(df["clients"] == c) & (df["poison_fraction"] == 0)]["accuracy"].max()
        poison_acc = df[
            (df["clients"] == c)
            & (df["poison_fraction"] == 0.4)
            & (df["poison_mode"] == "target")
        ]["accuracy"].max()
        rows.append({"clients": c, "type": "Clean (best)", "accuracy": clean_acc})
        rows.append({"clients": c, "type": "40% target poison (best)", "accuracy": poison_acc})
    long = pd.DataFrame(rows)
    fig = px.bar(
        long,
        x="clients",
        y="accuracy",
        color="type",
        barmode="group",
        labels={"clients": "Federated clients", "accuracy": "Test accuracy", "type": "Setting"},
        title="Clean vs 40% label poisoning on client 0",
        text=long["accuracy"].map(lambda v: f"{v:.3f}"),
        color_discrete_map={"Clean (best)": "#27ae60", "40% target poison (best)": "#e74c3c"},
    )
    fig.update_traces(textposition="outside")
    fig.update_yaxes(range=[0.84, 0.94])
    return fig


def chart_box_by_scenario(df: pd.DataFrame) -> go.Figure:
    fig = px.box(
        df,
        x="scenario",
        y="accuracy",
        color="scenario",
        category_orders={"scenario": SCENARIO_ORDER},
        labels={"scenario": "Scenario", "accuracy": "Test accuracy"},
        title="Accuracy distribution by scenario (all runs)",
        points="outliers",
    )
    fig.update_layout(showlegend=False)
    fig.update_yaxes(range=[0.84, 0.94])
    return fig


def html_table(df: pd.DataFrame) -> str:
    t = df.sort_values("accuracy", ascending=False)[
        [
            "run_id",
            "clients",
            "rounds",
            "local_epochs",
            "scenario",
            "accuracy",
            "f1-score",
            "precision",
            "recall",
        ]
    ].round(4)
    return t.to_html(index=False, classes="data-table", border=0)


def build_experiments_page(df: pd.DataFrame) -> str:
    best = (
        df.sort_values(["accuracy", "f1-score"], ascending=False)
        .groupby(["clients", "scenario"], observed=False)
        .first()
        .reset_index()
    )
    clean = df[df["poison_fraction"] == 0]

    charts = [
        ("1. Clients & scenarios", chart_accuracy_by_scenario(best)),
        ("2. Heatmap", chart_heatmap(df)),
        ("3. All runs", chart_scatter_all(df)),
        ("4. Training rounds", chart_rounds_clean(clean)),
        ("5. Poison impact", chart_clean_vs_poison(df)),
        ("6. Scenario spread", chart_box_by_scenario(df)),
    ]

    sections = []
    for i, (title, fig) in enumerate(charts):
        div = fig_to_div(fig, include_js=(i == 0))
        sections.append(f'<section class="chart-block"><h2>{title}</h2>{div}</section>')

    table_html = html_table(df)
    return (
        "".join(sections)
        + f"""
    <section class="table-block">
      <h2>7. Full results table</h2>
      <div class="table-wrap">{table_html}</div>
    </section>"""
    )


def build_combined_html(experiments_body: str, confusion_body: str) -> str:
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>Federated experiments dashboard</title>
  <style>
    * {{ box-sizing: border-box; }}
    body {{
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
      margin: 0; padding: 0; background: #eef1f5; color: #1a1a2e;
    }}
    header {{
      background: #2c3e50; color: #fff; padding: 24px 32px 0;
    }}
    header h1 {{ margin: 0 0 8px; font-size: 1.5rem; font-weight: 600; }}
    header p {{ margin: 0 0 16px; opacity: 0.9; font-size: 0.95rem; max-width: 720px; line-height: 1.5; }}
    nav.page-nav {{
      display: flex; gap: 4px; padding-bottom: 0;
    }}
    nav.page-nav button {{
      background: rgba(255,255,255,0.12); color: #fff; border: none;
      padding: 10px 20px; font-size: 0.95rem; cursor: pointer;
      border-radius: 8px 8px 0 0; font-family: inherit;
    }}
    nav.page-nav button:hover {{ background: rgba(255,255,255,0.22); }}
    nav.page-nav button.active {{
      background: #eef1f5; color: #2c3e50; font-weight: 600;
    }}
    main {{ max-width: 1100px; margin: 0 auto; padding: 24px 20px 48px; }}
    .page {{ display: none; }}
    .page.active {{ display: block; }}
    .chart-block {{
      background: #fff; border-radius: 10px; padding: 20px 16px 8px;
      margin-bottom: 28px; box-shadow: 0 2px 8px rgba(0,0,0,0.06);
    }}
    .chart-block h2 {{
      margin: 0 0 12px; font-size: 1.1rem; font-weight: 600; color: #34495e;
      border-bottom: 2px solid #3498db; padding-bottom: 8px;
    }}
    .confusion-block h2 {{ border-bottom-color: #9b59b6; }}
    .chart-block .plotly-graph-div {{ width: 100% !important; }}
    .info-box {{
      background: #fff; border-radius: 8px; padding: 12px 16px; margin-bottom: 20px;
      font-size: 0.9rem; line-height: 1.6; box-shadow: 0 2px 8px rgba(0,0,0,0.06);
    }}
    .table-block {{
      background: #fff; border-radius: 10px; padding: 20px;
      box-shadow: 0 2px 8px rgba(0,0,0,0.06);
    }}
    .table-block h2 {{ margin: 0 0 12px; font-size: 1.1rem; color: #34495e; }}
    .table-wrap {{ overflow-x: auto; max-height: 480px; overflow-y: auto; }}
    table.data-table {{
      width: 100%; border-collapse: collapse; font-size: 0.85rem;
    }}
    table.data-table th {{
      position: sticky; top: 0; background: #3498db; color: #fff;
      text-align: left; padding: 10px 12px; white-space: nowrap;
    }}
    table.data-table td {{
      padding: 8px 12px; border-bottom: 1px solid #ecf0f1;
    }}
    table.data-table tbody tr:nth-child(even) {{ background: #f8f9fa; }}
    table.data-table tbody tr:hover {{ background: #e8f4fc; }}
  </style>
</head>
<body>
  <header>
    <h1>UNSW/CIC federated learning — experiment results</h1>
    <p id="header-desc">
      Interactive charts from <code>many_tests_ranked_all.csv</code> (64 runs).
      Hover for details · zoom/pan with toolbar · double-click chart to reset.
    </p>
    <nav class="page-nav">
      <button type="button" class="active" data-page="experiments">Experiments</button>
      <button type="button" data-page="confusion">Confusion matrices</button>
    </nav>
  </header>
  <main>
    <div id="page-experiments" class="page active">{experiments_body}</div>
    <div id="page-confusion" class="page">{confusion_body}</div>
  </main>
  <script>
    const descriptions = {{
      experiments: "Interactive charts from many_tests_ranked_all.csv (64 runs). Hover for details · zoom/pan with toolbar · double-click chart to reset.",
      confusion: "Confusion matrices and per-class TP / FP / FN / TN for key federated scenarios (10 rounds, 2 local epochs)."
    }};
    document.querySelectorAll(".page-nav button").forEach(btn => {{
      btn.addEventListener("click", () => {{
        const page = btn.dataset.page;
        document.querySelectorAll(".page-nav button").forEach(b => b.classList.remove("active"));
        btn.classList.add("active");
        document.querySelectorAll(".page").forEach(p => p.classList.remove("active"));
        document.getElementById("page-" + page).classList.add("active");
        document.getElementById("header-desc").textContent = descriptions[page];
        window.dispatchEvent(new Event("resize"));
      }});
    }});
  </script>
</body>
</html>
"""


def main() -> None:
    from visualize_confusion import build_confusion_sections, load_confusion_records

    df = load_df()
    experiments_body = build_experiments_page(df)

    try:
        confusion_records = load_confusion_records()
        confusion_body = build_confusion_sections(confusion_records, include_plotly_js=False)
    except FileNotFoundError as exc:
        print(exc)
        confusion_body = (
            '<div class="info-box">'
            "Confusion data not cached yet. Run: <code>python3 visualize_confusion.py</code>"
            "</div>"
        )

    html = build_combined_html(experiments_body, confusion_body)

    OUT_HTML.parent.mkdir(exist_ok=True)
    OUT_HTML_PUBLIC.parent.mkdir(exist_ok=True)
    OUT_HTML.write_text(html, encoding="utf-8")
    OUT_HTML_ROOT.write_text(html, encoding="utf-8")
    OUT_HTML_PUBLIC.write_text(html, encoding="utf-8")
    print("Wrote", OUT_HTML.resolve())
    print("Wrote", OUT_HTML_ROOT.resolve())
    print("Wrote", OUT_HTML_PUBLIC.resolve())


if __name__ == "__main__":
    main()
