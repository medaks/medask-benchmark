"""
Plotly-based interactive visualization for triage benchmark results.

build_*_fig() functions return go.Figure objects (used by Streamlit app).
chart_*() functions are thin wrappers that save HTML to charts_output/.
"""
import math
from collections import defaultdict
from pathlib import Path
from typing import Optional

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from research.db.schema import init_db
from research.db.store import ResultStore

OUTPUT_DIR = Path(__file__).parent.parent / "charts_output"


def _ensure_output_dir():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ── Figure builders (return go.Figure) ──────────────────────────────


def build_overall_accuracy_fig(
    vignette_set: str = "semigran",
    include_baselines: bool = True,
    source_filter: Optional[str] = None,
) -> go.Figure:
    conn = init_db()
    store = ResultStore(conn)

    summaries = store.get_model_summary(vignette_set, source=source_filter)

    fig = go.Figure()

    if summaries:
        labels = []
        means = []
        sds = []
        hover_texts = []

        for s in summaries:
            label = s["model_name"]
            if s["quantization"]:
                label += f" ({s['quantization']})"
            labels.append(label)
            means.append(s["mean_overall"] * 100)
            sds.append(s["sd_overall"] * 100)
            hover_texts.append(
                f"Runs: {s['num_runs']}<br>"
                f"Temp: {s['temperature']}<br>"
                f"Mean: {s['mean_overall']:.1%} +/- {s['sd_overall']:.1%}"
            )

        fig.add_trace(go.Bar(
            name="Benchmark Results",
            x=labels,
            y=means,
            error_y=dict(type="data", array=sds, visible=True),
            text=[f"{m:.1f}%" for m in means],
            textposition="outside",
            hovertext=hover_texts,
            hoverinfo="text",
            marker_color="steelblue",
        ))

    if include_baselines:
        baselines = store.get_baselines(vignette_set)
        if baselines:
            bl_labels = [b["model_name"] for b in baselines]
            bl_means = [b["overall_accuracy"] * 100 for b in baselines]

            fig.add_trace(go.Bar(
                name="Published Baseline (Table 1)",
                x=bl_labels,
                y=bl_means,
                text=[f"{m:.1f}%" for m in bl_means],
                textposition="outside",
                marker_color="coral",
                opacity=0.7,
            ))

    fig.update_layout(
        title=f"Triage Accuracy Comparison - {vignette_set.title()} Dataset",
        yaxis_title="Overall Accuracy (%)",
        yaxis_range=[0, 105],
        barmode="group",
        template="plotly_white",
        legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99),
    )

    conn.close()
    return fig


def build_per_category_fig(
    vignette_set: str = "semigran",
    include_baselines: bool = True,
) -> go.Figure:
    conn = init_db()
    store = ResultStore(conn)

    summaries = store.get_model_summary(vignette_set)

    fig = go.Figure()

    categories = ["em", "ne", "sc"]
    cat_labels = {"em": "Emergency", "ne": "Non-Emergency", "sc": "Self-Care"}
    colors = {"em": "#e74c3c", "ne": "#f39c12", "sc": "#27ae60"}

    models = []
    for s in summaries:
        label = s["model_name"]
        if s["quantization"]:
            label += f" ({s['quantization']})"
        models.append((label, s))

    for cat in categories:
        fig.add_trace(go.Bar(
            name=cat_labels[cat],
            x=[m[0] for m in models],
            y=[m[1][f"mean_{cat}"] * 100 for m in models],
            marker_color=colors[cat],
            text=[f"{m[1][f'mean_{cat}'] * 100:.1f}%" for m in models],
            textposition="outside",
        ))

    if include_baselines:
        baselines = store.get_baselines(vignette_set)
        if baselines:
            bl_labels = [b["model_name"] + " (baseline)" for b in baselines]
            for cat in categories:
                fig.add_trace(go.Bar(
                    name=f"{cat_labels[cat]} (baseline)",
                    x=bl_labels,
                    y=[(b[f"{cat}_accuracy"] or 0) * 100 for b in baselines],
                    marker_color=colors[cat],
                    opacity=0.5,
                    showlegend=False,
                ))

    fig.update_layout(
        title=f"Per-Category Triage Accuracy - {vignette_set.title()}",
        yaxis_title="Accuracy (%)",
        yaxis_range=[0, 105],
        barmode="group",
        template="plotly_white",
    )

    conn.close()
    return fig


def build_settings_comparison_fig(
    model_name: str,
    vary_by: str = "temperature",
    vignette_set: str = "semigran",
) -> Optional[go.Figure]:
    conn = init_db()
    store = ResultStore(conn)

    runs = store.get_runs(model_name=model_name, vignette_set=vignette_set)

    if not runs:
        conn.close()
        return None

    groups = defaultdict(list)
    for r in runs:
        key = r.get(vary_by, "N/A")
        groups[key].append(r)

    fig = go.Figure()

    x_labels = []
    y_means = []
    y_sds = []

    for key in sorted(groups.keys(), key=str):
        x_labels.append(str(key))
        accs = [r["overall_accuracy"] for r in groups[key] if r["overall_accuracy"] is not None]
        if accs:
            mean_acc = sum(accs) / len(accs) * 100
            if len(accs) > 1:
                variance = sum((a * 100 - mean_acc) ** 2 for a in accs) / (len(accs) - 1)
                sd = math.sqrt(variance)
            else:
                sd = 0
        else:
            mean_acc = 0
            sd = 0
        y_means.append(mean_acc)
        y_sds.append(sd)

    fig.add_trace(go.Bar(
        x=x_labels,
        y=y_means,
        error_y=dict(type="data", array=y_sds, visible=True),
        text=[f"{m:.1f}%" for m in y_means],
        textposition="outside",
        marker_color="steelblue",
    ))

    vary_label = vary_by.replace("_", " ").title()
    fig.update_layout(
        title=f"{model_name}: Accuracy by {vary_label} - {vignette_set.title()}",
        xaxis_title=vary_label,
        yaxis_title="Overall Accuracy (%)",
        yaxis_range=[0, 105],
        template="plotly_white",
    )

    conn.close()
    return fig


def build_safety_analysis_fig(
    vignette_set: str = "semigran",
) -> go.Figure:
    conn = init_db()
    store = ResultStore(conn)

    summaries = store.get_model_summary(vignette_set)

    models = []
    for s in summaries:
        label = s["model_name"]
        if s["quantization"]:
            label += f" ({s['quantization']})"
        models.append(label)

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(
        go.Bar(
            name="Safety Rate",
            x=models,
            y=[s["mean_safety"] * 100 for s in summaries],
            marker_color="#27ae60",
        ),
        secondary_y=False,
    )

    fig.add_trace(
        go.Scatter(
            name="Over-triage Rate",
            x=models,
            y=[s["mean_overtriage"] * 100 for s in summaries],
            mode="lines+markers",
            marker=dict(size=10, color="#e74c3c"),
            line=dict(width=2, color="#e74c3c"),
        ),
        secondary_y=True,
    )

    fig.update_layout(
        title=f"Safety Analysis - {vignette_set.title()}",
        template="plotly_white",
    )
    fig.update_yaxes(title_text="Safety Rate (%)", secondary_y=False, range=[0, 105])
    fig.update_yaxes(title_text="Over-triage Rate (%)", secondary_y=True, range=[0, 105])

    conn.close()
    return fig


# ── HTML writers (thin wrappers for CLI) ────────────────────────────


def chart_overall_accuracy(
    vignette_set: str = "semigran",
    include_baselines: bool = True,
    source_filter: Optional[str] = None,
    output_name: str = "overall_accuracy.html",
):
    _ensure_output_dir()
    fig = build_overall_accuracy_fig(vignette_set, include_baselines, source_filter)
    out_path = OUTPUT_DIR / output_name
    fig.write_html(str(out_path), include_plotlyjs=True)
    print(f"Chart saved: {out_path}")
    return str(out_path)


def chart_per_category_breakdown(
    vignette_set: str = "semigran",
    include_baselines: bool = True,
    output_name: str = "per_category_breakdown.html",
):
    _ensure_output_dir()
    fig = build_per_category_fig(vignette_set, include_baselines)
    out_path = OUTPUT_DIR / output_name
    fig.write_html(str(out_path), include_plotlyjs=True)
    print(f"Chart saved: {out_path}")


def chart_settings_comparison(
    model_name: str,
    vary_by: str = "temperature",
    vignette_set: str = "semigran",
    output_name: Optional[str] = None,
):
    _ensure_output_dir()
    fig = build_settings_comparison_fig(model_name, vary_by, vignette_set)
    if fig is None:
        print(f"No runs found for model={model_name}, vignette_set={vignette_set}")
        return
    if output_name is None:
        safe_model = model_name.replace("/", "_").replace(" ", "_")
        output_name = f"settings_{safe_model}_{vary_by}.html"
    out_path = OUTPUT_DIR / output_name
    fig.write_html(str(out_path), include_plotlyjs=True)
    print(f"Chart saved: {out_path}")


def chart_safety_analysis(
    vignette_set: str = "semigran",
    output_name: str = "safety_analysis.html",
):
    _ensure_output_dir()
    fig = build_safety_analysis_fig(vignette_set)
    out_path = OUTPUT_DIR / output_name
    fig.write_html(str(out_path), include_plotlyjs=True)
    print(f"Chart saved: {out_path}")
