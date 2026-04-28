from __future__ import annotations

import shutil
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle


PROJECT_ROOT = Path(__file__).resolve().parent.parent
PAPER_DIR = PROJECT_ROOT / "paper"
FIGURES_DIR = PAPER_DIR / "figures"
REPORT_FIGURES_DIR = PROJECT_ROOT / "reports" / "figures" / "acs"


def ensure_dirs() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def wrap_label(text: str, width: int = 18) -> str:
    return "\n".join(textwrap.wrap(text, width=width))


def draw_box(
    ax,
    x,
    y,
    w,
    h,
    text,
    fc="#f7fafc",
    ec="#2d3748",
    fontsize=9.2,
    weight="normal",
):
    box = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.012,rounding_size=0.018",
        linewidth=1.25,
        edgecolor=ec,
        facecolor=fc,
    )
    ax.add_patch(box)
    ax.text(
        x + w / 2,
        y + h / 2,
        wrap_label(text),
        ha="center",
        va="center",
        fontsize=fontsize,
        fontweight=weight,
        color="#1a202c",
        linespacing=1.15,
    )


def draw_arrow(ax, start, end, color="#4a5568", rad=0.0):
    arrow = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=14,
        linewidth=1.35,
        color=color,
        connectionstyle=f"arc3,rad={rad}",
    )
    ax.add_patch(arrow)


def generate_architecture_figure() -> None:
    fig, ax = plt.subplots(figsize=(14.2, 5.2))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    panels = [
        {
            "x": 0.045,
            "title": "Data Construction",
            "color": "#e8f3ff",
            "edge": "#2b6cb0",
            "items": [
                "ACS PUMS extraction",
                "State-year shards",
                "Temporal split",
                "Train: 2016",
                "Eval: 2017-2018",
            ],
        },
        {
            "x": 0.285,
            "title": "Model Lifecycle",
            "color": "#f2f7ed",
            "edge": "#2f855a",
            "items": [
                "Ordinal encoding",
                "XGBoost baseline",
                "Versioned artifacts",
                "Encoder refit",
                "GPU-aware training",
            ],
        },
        {
            "x": 0.525,
            "title": "Drift Intelligence",
            "color": "#fff7e8",
            "edge": "#c05621",
            "items": [
                "KS / chi-square tests",
                "Severity scoring",
                "Concept-drift signal",
                "2-batch persistence",
                "Rolling retraining window",
            ],
        },
        {
            "x": 0.765,
            "title": "Operations",
            "color": "#f3eefc",
            "edge": "#6b46c1",
            "items": [
                "FastAPI inference",
                "MLflow tracking",
                "Docker services",
                "Prometheus metrics",
                "Grafana dashboards",
            ],
        },
    ]

    panel_w, panel_h, y0 = 0.19, 0.69, 0.17
    for panel in panels:
        x = panel["x"]
        ax.add_patch(
            FancyBboxPatch(
                (x, y0),
                panel_w,
                panel_h,
                boxstyle="round,pad=0.016,rounding_size=0.026",
                facecolor=panel["color"],
                edgecolor=panel["edge"],
                linewidth=1.6,
            )
        )
        ax.text(
            x + panel_w / 2,
            y0 + panel_h - 0.075,
            panel["title"],
            ha="center",
            va="center",
            fontsize=11.5,
            fontweight="bold",
            color="#1a202c",
        )
        ax.plot([x + 0.025, x + panel_w - 0.025], [y0 + panel_h - 0.13] * 2, color=panel["edge"], lw=1.0)
        for idx, item in enumerate(panel["items"]):
            yy = y0 + panel_h - 0.205 - idx * 0.085
            ax.scatter(x + 0.04, yy, s=11, color=panel["edge"], zorder=4)
            ax.text(x + 0.058, yy, item, ha="left", va="center", fontsize=9.2, color="#2d3748")

    for left, right in zip(panels[:-1], panels[1:]):
        draw_arrow(
            ax,
            (left["x"] + panel_w + 0.006, y0 + panel_h / 2),
            (right["x"] - 0.006, y0 + panel_h / 2),
            "#4a5568",
        )

    draw_arrow(
        ax,
        (panels[2]["x"] + panel_w / 2, y0 - 0.002),
        (panels[1]["x"] + panel_w / 2, y0 - 0.002),
        "#2f855a",
        rad=0.22,
    )
    ax.text(
        0.5,
        y0 - 0.052,
        "policy-approved retraining feedback",
        ha="center",
        va="center",
        fontsize=8.8,
        color="#2f855a",
        fontweight="bold",
    )

    ax.text(0.045, 0.935, "research and deployment lifecycle", ha="left", va="center", fontsize=10, color="#4a5568")
    ax.plot([0.245, 0.955], [0.935, 0.935], color="#cbd5e0", lw=1.0)

    fig.tight_layout(pad=0.5)
    fig.savefig(FIGURES_DIR / "architecture_pipeline.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIGURES_DIR / "architecture_pipeline.pdf", bbox_inches="tight")
    plt.close(fig)


def generate_timeline_figure() -> None:
    fig, ax = plt.subplots(figsize=(14.2, 4.2))
    ax.set_xlim(0.5, 12.5)
    ax.set_ylim(-0.34, 0.72)
    ax.set_ylabel("Drift severity", fontsize=10)
    ax.set_xticks(range(1, 13))
    ax.set_xticklabels([f"B{i}" for i in range(1, 13)], fontsize=9)
    ax.set_title(
        "Multi-Year ACS Evaluation Timeline",
        fontsize=12,
        pad=14,
        fontweight="bold",
    )

    year_map = {i: "2017" for i in range(1, 7)}
    year_map.update({i: "2018" for i in range(7, 13)})
    severities = {
        1: 0.0000, 2: 0.4482, 3: 0.3778, 4: 0.5521,
        5: 0.3847, 6: 0.5519, 7: 0.5130, 8: 0.4139,
        9: 0.5062, 10: 0.4997, 11: 0.5067, 12: 0.5169,
    }
    concept_batches = {3, 6, 8, 11, 12}
    retrain_batches = {2, 4, 6, 8, 10, 12}

    ax.axhline(0.10, color="#718096", linewidth=1.0, linestyle="--", alpha=0.7)
    ax.text(12.35, 0.106, "severity trigger", ha="right", va="bottom", fontsize=8, color="#4a5568")

    ax.add_patch(Rectangle((0.5, -0.25), 6.0, 0.11, facecolor="#d6e8ff", edgecolor="none"))
    ax.add_patch(Rectangle((6.5, -0.25), 6.0, 0.11, facecolor="#ffe1c5", edgecolor="none"))
    ax.text(3.5, -0.195, "2017 evaluation source", ha="center", va="center", fontsize=9, color="#1f3552")
    ax.text(9.5, -0.195, "2018 evaluation source", ha="center", va="center", fontsize=9, color="#7b341e")
    ax.axvline(6.5, color="#4a5568", linestyle=":", linewidth=1.1)

    bar_colors = ["#5b8fd6" if year_map[b] == "2017" else "#d98945" for b in range(1, 13)]
    ax.bar(
        range(1, 13),
        [severities[b] for b in range(1, 13)],
        width=0.58,
        color=bar_colors,
        edgecolor="#2d3748",
        linewidth=0.6,
        zorder=2,
    )

    for batch in concept_batches:
        ax.scatter(batch, severities[batch] + 0.045, marker="D", s=54, color="#b83280", zorder=4)

    for batch in retrain_batches:
        ax.scatter(batch, -0.055, marker="^", s=62, color="#2f855a", zorder=4)

    handles = [
        Rectangle((0, 0), 1, 1, facecolor="#5b8fd6", label="2017 source severity"),
        Rectangle((0, 0), 1, 1, facecolor="#d98945", label="2018 source severity"),
        Line2D([0], [0], marker="D", color="w", markerfacecolor="#b83280", markersize=7, label="Concept drift"),
        Line2D([0], [0], marker="^", color="w", markerfacecolor="#2f855a", markersize=8, label="Policy retrain"),
    ]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=4, fontsize=8, frameon=False)
    ax.grid(axis="y", color="#e2e8f0", linewidth=0.8)

    for spine in ax.spines.values():
        spine.set_visible(False)

    fig.tight_layout(pad=0.7)
    fig.savefig(FIGURES_DIR / "acs_temporal_timeline.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIGURES_DIR / "acs_temporal_timeline.pdf", bbox_inches="tight")
    plt.close(fig)


def copy_existing_performance_figure() -> None:
    src = REPORT_FIGURES_DIR / "metrics_comparison.png"
    dst = FIGURES_DIR / "metrics_comparison.png"
    if not src.exists():
        raise FileNotFoundError(f"Expected report figure not found: {src}")
    shutil.copy2(src, dst)


def main() -> None:
    ensure_dirs()
    generate_architecture_figure()
    generate_timeline_figure()
    copy_existing_performance_figure()
    print(f"Saved paper figures to {FIGURES_DIR}")


if __name__ == "__main__":
    main()
