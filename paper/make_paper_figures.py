from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "paper" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

COLORS = {
    "Static": "#6B7280",
    "Immediate": "#C2410C",
    "Policy-Standard": "#0F766E",
    "Policy-CurrentAnchor": "#0F766E",
    "Policy-SlidingNoAnchor": "#2563EB",
}


def save(fig: plt.Figure, name: str) -> None:
    fig.tight_layout()
    fig.savefig(FIG_DIR / f"{name}.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIG_DIR / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def make_main_policy_tradeoff() -> None:
    df = pd.DataFrame(
        [
            {
                "policy": "Static",
                "mean_accuracy": 0.7666666667,
                "mean_f1": 0.7155351052,
                "retrain_count": 0,
                "degradation_area_f1": 0.4245890888,
                "total_retrain_time": 0.0,
            },
            {
                "policy": "Immediate",
                "mean_accuracy": 0.7873000000,
                "mean_f1": 0.7224680089,
                "retrain_count": 11,
                "degradation_area_f1": 0.3364758365,
                "total_retrain_time": 32.5915954113,
            },
            {
                "policy": "Policy-Standard",
                "mean_accuracy": 0.7836666667,
                "mean_f1": 0.7203962528,
                "retrain_count": 6,
                "degradation_area_f1": 0.3604386830,
                "total_retrain_time": 19.2075226307,
            },
        ]
    )

    fig, axes = plt.subplots(1, 3, figsize=(10.8, 3.2))
    x = np.arange(len(df))
    colors = [COLORS[p] for p in df["policy"]]

    axes[0].bar(x, df["mean_f1"], color=colors, width=0.62)
    axes[0].set_title("Predictive Performance")
    axes[0].set_ylabel("Mean F1")
    axes[0].set_ylim(0.70, 0.728)
    axes[0].set_xticks(x, df["policy"], rotation=18, ha="right")
    for idx, val in enumerate(df["mean_f1"]):
        axes[0].text(idx, val + 0.0007, f"{val:.4f}", ha="center", fontsize=8)

    axes[1].bar(x, df["degradation_area_f1"], color=colors, width=0.62)
    axes[1].set_title("Cumulative F1 Degradation")
    axes[1].set_ylabel("Area below batch-1 F1")
    axes[1].set_xticks(x, df["policy"], rotation=18, ha="right")
    for idx, val in enumerate(df["degradation_area_f1"]):
        axes[1].text(idx, val + 0.01, f"{val:.3f}", ha="center", fontsize=8)

    axes[2].bar(x, df["retrain_count"], color=colors, width=0.62)
    axes[2].set_title("Adaptation Cost")
    axes[2].set_ylabel("Retraining events")
    axes[2].set_xticks(x, df["policy"], rotation=18, ha="right")
    for idx, val in enumerate(df["retrain_count"]):
        axes[2].text(idx, val + 0.25, str(int(val)), ha="center", fontsize=8)

    for ax in axes:
        ax.grid(axis="y", alpha=0.22)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    save(fig, "main_policy_tradeoff")


def make_noise_ablation() -> None:
    path = ROOT / "reports" / "diagnostics" / "acs_noise_ablation" / "noise_ablation_summary.csv"
    df = pd.read_csv(path)
    order = [
        "clean_temporal",
        "covariate_feature_shift",
        "label_flip_only",
        "full_with_label_flips",
    ]
    label_map = {
        "clean_temporal": "Clean\ntemporal",
        "covariate_feature_shift": "Covariate +\nfeature shift",
        "label_flip_only": "Label-flip\nstress",
        "full_with_label_flips": "Mixed +\nlabel flips",
    }
    strategies = ["Static", "Policy-CurrentAnchor", "Policy-SlidingNoAnchor"]
    offsets = np.linspace(-0.24, 0.24, len(strategies))
    width = 0.22

    fig, ax = plt.subplots(figsize=(8.4, 3.7))
    x = np.arange(len(order))
    for offset, strategy in zip(offsets, strategies):
        vals = [
            float(df[(df["drift_mode"] == mode) & (df["strategy"] == strategy)]["mean_f1"].iloc[0])
            for mode in order
        ]
        ax.bar(
            x + offset,
            vals,
            width=width,
            label=strategy.replace("Policy-", "Policy: "),
            color=COLORS.get(strategy, "#111827"),
        )

    ax.axvspan(1.5, 3.5, color="#FEE2E2", alpha=0.35, zorder=0)
    ax.text(2.5, 0.755, "Synthetic label-noise stress", ha="center", fontsize=9, color="#991B1B")
    ax.set_title("Retraining Helps Under Learnable Drift, Not Label-Flip Noise")
    ax.set_ylabel("Mean F1")
    ax.set_ylim(0.62, 0.765)
    ax.set_xticks(x, [label_map[m] for m in order])
    ax.legend(ncol=3, fontsize=8, frameon=False, loc="lower left")
    ax.grid(axis="y", alpha=0.22)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    save(fig, "noise_ablation_summary")


def make_regime_diagnostics() -> None:
    metrics = pd.read_csv(ROOT / "reports" / "diagnostics" / "acs_regime" / "regime_metrics.csv")
    importance = pd.read_csv(
        ROOT / "reports" / "diagnostics" / "acs_regime" / "importance_diagnostics.csv"
    )
    labels = {
        "2016_only": "2016 only",
        "2017_2018_only": "2017-2018\nonly",
        "2017_2018_plus_10pct_2016_anchor": "2017-2018 +\n10% anchor",
        "2016_2018_expanding": "2016-2018\nexpanding",
    }
    metrics["label"] = metrics["model"].map(labels)
    importance["label"] = importance["model"].map(labels)
    colors = ["#6B7280", "#0F766E", "#14B8A6", "#C2410C"]

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.4))
    x = np.arange(len(metrics))
    axes[0].bar(x, metrics["f1"], color=colors, width=0.62)
    axes[0].set_title("Held-out 2018 Performance")
    axes[0].set_ylabel("F1")
    axes[0].set_ylim(0.70, 0.748)
    axes[0].set_xticks(x, metrics["label"], rotation=12, ha="right")
    for idx, val in enumerate(metrics["f1"]):
        axes[0].text(idx, val + 0.001, f"{val:.4f}", ha="center", fontsize=8)

    axes[1].bar(x, importance["l1_distance_from_2016"], color=colors, width=0.62)
    axes[1].set_title("Feature-Importance Shift")
    axes[1].set_ylabel("L1 distance from 2016 model")
    axes[1].set_xticks(x, importance["label"], rotation=12, ha="right")
    for idx, val in enumerate(importance["l1_distance_from_2016"]):
        axes[1].text(idx, val + 0.002, f"{val:.3f}", ha="center", fontsize=8)

    for ax in axes:
        ax.grid(axis="y", alpha=0.22)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    save(fig, "temporal_regime_diagnostics")


def make_research_story_map() -> None:
    fig, ax = plt.subplots(figsize=(8.8, 3.8))
    ax.axis("off")

    boxes = [
        (0.04, 0.56, 0.42, 0.32, "Learnable Drift", "Temporal ACS and covariate/feature shifts\ncontain stable signal; anchored policy improves mean F1."),
        (0.54, 0.56, 0.42, 0.32, "Label-Flip Stress", "Subgroup label flips inject contradictions;\nstatic deployment outperforms retraining."),
        (0.04, 0.12, 0.42, 0.32, "Policy Gate", "Persistence and cooldown reduce unnecessary\nadaptation compared with immediate retraining."),
        (0.54, 0.12, 0.42, 0.32, "Core Finding", "Drift detection is not enough: retraining helps\nonly when new labels describe a learnable regime."),
    ]
    facecolors = ["#DCFCE7", "#FEE2E2", "#E0F2FE", "#F5F3FF"]
    edgecolors = ["#166534", "#991B1B", "#075985", "#5B21B6"]

    for (x, y, w, h, title, body), fc, ec in zip(boxes, facecolors, edgecolors):
        rect = plt.Rectangle((x, y), w, h, transform=ax.transAxes, fc=fc, ec=ec, lw=1.5)
        ax.add_patch(rect)
        ax.text(x + 0.02, y + h - 0.08, title, transform=ax.transAxes, fontsize=12, weight="bold", color=ec)
        ax.text(x + 0.02, y + h - 0.18, body, transform=ax.transAxes, fontsize=9.5, va="top", color="#111827")

    ax.annotate(
        "",
        xy=(0.54, 0.72),
        xytext=(0.46, 0.72),
        xycoords=ax.transAxes,
        arrowprops=dict(arrowstyle="->", lw=1.2, color="#374151"),
    )
    ax.annotate(
        "",
        xy=(0.54, 0.28),
        xytext=(0.46, 0.28),
        xycoords=ax.transAxes,
        arrowprops=dict(arrowstyle="->", lw=1.2, color="#374151"),
    )
    ax.text(0.5, 0.505, "Ablation separates useful adaptation from noise chasing", ha="center", transform=ax.transAxes, fontsize=10)

    save(fig, "research_story_map")


def main() -> None:
    make_main_policy_tradeoff()
    make_noise_ablation()
    make_regime_diagnostics()
    make_research_story_map()
    print(f"Saved paper figures to {FIG_DIR}")


if __name__ == "__main__":
    main()
