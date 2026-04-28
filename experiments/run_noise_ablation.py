"""
ACS retraining ablation for clean temporal shift versus synthetic label noise.

This experiment is meant to answer a specific methodological question:
does adaptive retraining help when the stream contains learnable temporal drift,
and does it stop helping when subgroup-specific label flips inject contradictory
label noise?
"""

from __future__ import annotations

import argparse
import copy
import time
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from src.drift_detection import DriftDetector
from src.drift_simulation import DriftSimulator
from src.evaluate import compute_metrics, extract_core_metrics
from src.policy_engine import RetrainingPolicyEngine
from src.retrain import ModelVersionManager, prepare_retraining_data, retrain_model
from src.train import run_training_pipeline
from src.utils import (
    ensure_dir,
    get_path,
    get_processed_data_paths,
    load_drift_config,
    load_settings,
    load_thresholds,
    load_yaml,
)


ABLATION_MODES = {
    "clean_temporal": "Temporal 2017-2018 batches without synthetic perturbations.",
    "covariate_feature_shift": "Temporal batches with covariate and feature-noise shifts, but no label flips.",
    "label_flip_only": "Temporal batches with subgroup-specific label flips only.",
    "full_with_label_flips": "Original ACS drift schedule with covariate shifts, feature noise, and label flips.",
}

LABEL_FLIP_MODES = {"label_flip_only", "full_with_label_flips"}


def build_ablation_config(base_config: dict, mode: str) -> dict:
    """Return an in-memory drift config for the requested ablation mode."""
    if mode not in ABLATION_MODES:
        raise ValueError(f"Unknown ablation mode: {mode}")

    config = copy.deepcopy(base_config)
    for batch_cfg in config.get("batches", {}).values():
        perturbations = copy.deepcopy(batch_cfg.get("perturbations", {}))

        if mode == "clean_temporal":
            perturbations = {}
        elif mode == "covariate_feature_shift":
            perturbations.pop("conditional_shift", None)
        elif mode == "label_flip_only":
            perturbations = {
                "conditional_shift": perturbations["conditional_shift"]
            } if "conditional_shift" in perturbations else {}
        elif mode == "full_with_label_flips":
            pass

        batch_cfg["perturbations"] = perturbations

    return config


def make_policy_config(
    base_thresholds: dict,
    *,
    severity_threshold: float | None = None,
    min_new_samples: int | None = None,
    cooldown_batches: int | None = None,
    anchor_fraction: float | None = None,
    window_batches: int | None = None,
) -> dict:
    """Copy thresholds and override only the ablation-specific knobs."""
    config = copy.deepcopy(base_thresholds)
    policy_cfg = config.setdefault("retraining_policy", {})
    data_cfg = config.setdefault("retraining_data", {})

    if severity_threshold is not None:
        policy_cfg["severity_threshold"] = severity_threshold
    if min_new_samples is not None:
        policy_cfg["min_new_samples"] = min_new_samples
    if cooldown_batches is not None:
        policy_cfg["cooldown_batches"] = cooldown_batches
    if anchor_fraction is not None:
        data_cfg["anchor_fraction"] = anchor_fraction
    if window_batches is not None:
        data_cfg["window_batches"] = window_batches

    return config


def generate_batches_for_mode(
    test_df: pd.DataFrame,
    base_drift_config: dict,
    mode: str,
    seed: int,
) -> dict[str, pd.DataFrame]:
    """Generate ablation batches in memory without overwriting data/batches."""
    config = build_ablation_config(base_drift_config, mode)
    simulator = DriftSimulator(test_df=test_df, config=config, seed=seed)
    return simulator.generate_batches()


def simulate_strategy(
    *,
    drift_mode: str,
    strategy_name: str,
    strategy: str,
    batches: dict[str, pd.DataFrame],
    base_model,
    base_encoder,
    original_train_data: pd.DataFrame,
    base_metrics: dict,
    policy_config: dict,
    settings: dict,
) -> tuple[pd.DataFrame, dict]:
    """Run one online stream simulation from the shared baseline model."""
    print(f"[{drift_mode} | {strategy_name}] Starting stream simulation...")

    current_model = base_model
    current_encoder = base_encoder
    current_reference = original_train_data.copy()
    baseline_metrics = dict(base_metrics)

    detector = DriftDetector(current_reference)
    policy_engine = RetrainingPolicyEngine(policy_config)
    version_mgr = ModelVersionManager()

    accumulated_samples = pd.DataFrame()
    recent_batches: list[pd.DataFrame] = []
    history = []
    total_retrain_time = 0.0
    total_inference_time = 0.0
    retrain_count = 0
    target_col = settings["data"]["target_column"]

    for batch_name, batch_df in batches.items():
        inf_start = time.time()
        encoded = current_encoder.transform(batch_df, include_target=True)
        X = encoded[current_encoder.get_feature_names()]
        y = encoded[target_col]
        y_pred = current_model.predict(X)
        inference_time = time.time() - inf_start
        total_inference_time += inference_time

        metrics = compute_metrics(y, y_pred, prefix="")
        current_metrics = extract_core_metrics(metrics)
        drift_result = detector.detect(batch_df, batch_name)

        accumulated_samples = pd.concat([accumulated_samples, batch_df], ignore_index=True)
        recent_batches.append(batch_df.copy())

        decision = policy_engine.evaluate(
            drift_result,
            len(accumulated_samples),
            current_metrics=current_metrics,
            baseline_metrics=baseline_metrics,
        )

        if strategy == "static":
            should_retrain = False
        elif strategy == "immediate":
            should_retrain = drift_result.drift_detected or decision.concept_drift_met
        else:
            should_retrain = decision.should_retrain

        retrain_triggered = False
        train_duration = 0.0
        if should_retrain:
            retrain_triggered = True
            retrain_count += 1
            retraining_df = prepare_retraining_data(
                recent_batches=recent_batches,
                original_train_data=original_train_data,
                thresholds=policy_config,
                settings=settings,
            )

            retrain_start = time.time()
            retrain_result = retrain_model(
                retraining_df,
                current_encoder,
                version_mgr,
                old_train_data=None,
                use_mlflow=False,
                new_data_sample_count=len(accumulated_samples),
            )
            train_duration = time.time() - retrain_start
            total_retrain_time += train_duration

            current_model = retrain_result["model"]
            current_encoder = retrain_result["encoder"]
            current_reference = retrain_result["train_data"]
            detector.update_reference(current_reference)
            baseline_metrics = extract_core_metrics(
                retrain_result["metrics"],
                prefix="retrain_",
            )
            accumulated_samples = pd.DataFrame()

        policy_engine.register_batch_processed(retrain_triggered)

        history.append(
            {
                "drift_mode": drift_mode,
                "strategy": strategy_name,
                "batch": batch_name,
                "model_version": version_mgr.get_current_version_str(),
                "accuracy": metrics["accuracy"],
                "f1": metrics["f1"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "drift_severity": drift_result.severity_score,
                "drift_flags": drift_result.n_features_drifted,
                "concept_drift": decision.concept_drift_met,
                "retrained": retrain_triggered,
                "train_duration": train_duration,
                "inference_time": inference_time,
                "policy_reason": decision.reason,
            }
        )

    history_df = pd.DataFrame(history)
    summary = summarize_stream(
        history_df,
        drift_mode=drift_mode,
        strategy_name=strategy_name,
        retrain_count=retrain_count,
        total_retrain_time=total_retrain_time,
        total_inference_time=total_inference_time,
        policy_config=policy_config,
    )
    return history_df, summary


def summarize_stream(
    history_df: pd.DataFrame,
    *,
    drift_mode: str,
    strategy_name: str,
    retrain_count: int,
    total_retrain_time: float,
    total_inference_time: float,
    policy_config: dict,
) -> dict:
    """Compute compact stream-level metrics for comparison."""
    baseline_f1 = float(history_df.iloc[0]["f1"])
    f1_drops = baseline_f1 - history_df["f1"]

    recovery_gains = []
    for idx, row in history_df.iterrows():
        if row["retrained"] and idx + 1 < len(history_df):
            recovery_gains.append(float(history_df.iloc[idx + 1]["f1"] - row["f1"]))

    retraining_cfg = policy_config.get("retraining_data", {})
    return {
        "drift_mode": drift_mode,
        "strategy": strategy_name,
        "anchor_fraction": retraining_cfg.get("anchor_fraction"),
        "window_batches": retraining_cfg.get("window_batches"),
        "mean_accuracy": float(history_df["accuracy"].mean()),
        "mean_f1": float(history_df["f1"].mean()),
        "worst_f1": float(history_df["f1"].min()),
        "max_f1_drop_from_batch1": float(f1_drops.max()),
        "degradation_area_f1": float(f1_drops.clip(lower=0).sum()),
        "post_retrain_avg_f1": (
            float(history_df.loc[history_df["retrained"], "f1"].mean())
            if history_df["retrained"].any()
            else 0.0
        ),
        "mean_recovery_gain_next_batch_f1": (
            float(sum(recovery_gains) / len(recovery_gains)) if recovery_gains else 0.0
        ),
        "retrain_count": retrain_count,
        "concept_drift_batches": int(history_df["concept_drift"].sum()),
        "drift_alert_batches": int((history_df["drift_severity"] > 0).sum()),
        "total_retrain_time": total_retrain_time,
        "total_inference_time": total_inference_time,
    }


def add_static_gains(summary_df: pd.DataFrame) -> pd.DataFrame:
    """Add gain/loss against the static baseline for each drift mode."""
    summary_df = summary_df.copy()
    static_f1 = (
        summary_df[summary_df["strategy"] == "Static"]
        .set_index("drift_mode")["mean_f1"]
        .to_dict()
    )
    summary_df["mean_f1_gain_vs_static"] = summary_df.apply(
        lambda row: row["mean_f1"] - static_f1.get(row["drift_mode"], row["mean_f1"]),
        axis=1,
    )
    return summary_df


def save_outputs(
    all_history: pd.DataFrame,
    summary_df: pd.DataFrame,
    output_dir: Path,
) -> None:
    """Persist CSVs, figures, and a short Markdown report."""
    output_dir = ensure_dir(output_dir)
    all_history.to_csv(output_dir / "noise_ablation_batch_metrics.csv", index=False)
    summary_df.to_csv(output_dir / "noise_ablation_summary.csv", index=False)

    for drift_mode, mode_df in all_history.groupby("drift_mode"):
        fig, ax = plt.subplots(figsize=(10, 4.8))
        for strategy, strategy_df in mode_df.groupby("strategy"):
            ax.plot(strategy_df["batch"], strategy_df["f1"], marker="o", label=strategy)
        ax.set_title(f"F1 by Batch: {drift_mode}")
        ax.set_xlabel("Batch")
        ax.set_ylabel("F1")
        ax.grid(axis="y", alpha=0.25)
        ax.legend()
        plt.xticks(rotation=35, ha="right")
        fig.tight_layout()
        fig.savefig(output_dir / f"f1_timeline_{drift_mode}.png", dpi=220)
        plt.close(fig)

    pivot = summary_df.pivot(index="drift_mode", columns="strategy", values="mean_f1")
    fig, ax = plt.subplots(figsize=(10, 5))
    pivot.plot(kind="bar", ax=ax)
    ax.set_title("Mean F1 by Drift Mode and Retraining Strategy")
    ax.set_xlabel("Drift mode")
    ax.set_ylabel("Mean F1")
    ax.grid(axis="y", alpha=0.25)
    plt.xticks(rotation=25, ha="right")
    fig.tight_layout()
    fig.savefig(output_dir / "mean_f1_by_ablation.png", dpi=220)
    plt.close(fig)

    report = build_markdown_report(summary_df)
    with open(output_dir / "noise_ablation_report.md", "w") as f:
        f.write(report)


def build_markdown_report(summary_df: pd.DataFrame) -> str:
    """Create a concise, paper-friendly interpretation report."""
    lines = [
        "# ACS Noise Ablation: When Does Retraining Help?",
        "",
        "This diagnostic separates clean temporal shift, covariate/feature shift, subgroup label flips, and the full mixed ACS drift schedule.",
        "",
        "## Summary Table",
        "",
        dataframe_to_markdown(summary_df),
        "",
        "## Interpretation Guide",
        "",
        "- If policy retraining improves over Static on clean temporal or covariate-only streams, retraining is useful when drift is learnable.",
        "- If policy retraining fails or degrades under label-flip streams, the synthetic concept drift is behaving more like contradictory label noise than a stable new rule.",
        "- If Policy-SlidingNoAnchor beats Policy-CurrentAnchor, the historical anchor is too large for online adaptation.",
        "- If both policy variants lose to Static, the retraining trigger/data policy is not yet aligned with the stream.",
        "",
    ]

    for drift_mode in ABLATION_MODES:
        mode_df = summary_df[summary_df["drift_mode"] == drift_mode]
        if mode_df.empty:
            continue
        best = mode_df.sort_values("mean_f1", ascending=False).iloc[0]
        static = mode_df[mode_df["strategy"] == "Static"].iloc[0]
        lines.extend(
            [
                f"## {drift_mode}",
                "",
                f"Best strategy: **{best['strategy']}** with mean F1 `{best['mean_f1']:.4f}`.",
                f"Static mean F1 was `{static['mean_f1']:.4f}`.",
                "",
            ]
        )

    return "\n".join(lines)


def dataframe_to_markdown(df: pd.DataFrame) -> str:
    """Small markdown-table writer to avoid requiring optional tabulate."""
    columns = list(df.columns)
    rows = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for _, row in df.iterrows():
        values = []
        for col in columns:
            value = row[col]
            if isinstance(value, float):
                values.append(f"{value:.4f}")
            else:
                values.append(str(value))
        rows.append("| " + " | ".join(values) + " |")
    return "\n".join(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run ACS clean-vs-label-noise retraining ablation."
    )
    parser.add_argument(
        "--include-immediate",
        action="store_true",
        help="Also run the immediate-retraining baseline. Slower, but comparable to run_experiments.py.",
    )
    parser.add_argument(
        "--severity-threshold",
        type=float,
        default=None,
        help="Override policy severity threshold. Defaults to thresholds.yaml.",
    )
    parser.add_argument(
        "--min-new-samples",
        type=int,
        default=None,
        help="Override minimum new labeled samples. Defaults to thresholds.yaml.",
    )
    parser.add_argument(
        "--cooldown-batches",
        type=int,
        default=None,
        help="Override cooldown batches. Defaults to thresholds.yaml.",
    )
    parser.add_argument(
        "--window-batches",
        type=int,
        default=None,
        help="Override rolling window batch count. Defaults to thresholds.yaml.",
    )
    parser.add_argument(
        "--label-noise-config",
        type=str,
        default="drift_config_acs_label_flip_stress.yaml",
        help=(
            "Config file used for label-flip ablation modes. Relative names are "
            "resolved under configs/."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    settings = load_settings()
    thresholds = load_thresholds()
    seed = settings["project"]["random_seed"]

    _, test_path = get_processed_data_paths(settings)
    test_df = pd.read_csv(test_path)
    train_path, _ = get_processed_data_paths(settings)
    original_train_data = pd.read_csv(train_path)
    learnable_drift_config = load_drift_config()
    label_noise_drift_config = load_yaml(args.label_noise_config)

    baseline = run_training_pipeline(
        run_name="acs_noise_ablation_baseline",
        use_mlflow=False,
    )
    base_model = baseline["model"]
    base_encoder = baseline["encoder"]
    base_metrics = extract_core_metrics(baseline["metrics"], prefix="test_")

    policy_current_anchor = make_policy_config(
        thresholds,
        severity_threshold=args.severity_threshold,
        min_new_samples=args.min_new_samples,
        cooldown_batches=args.cooldown_batches,
        anchor_fraction=thresholds.get("retraining_data", {}).get("anchor_fraction", 0.1),
        window_batches=args.window_batches,
    )
    policy_sliding = make_policy_config(
        thresholds,
        severity_threshold=args.severity_threshold,
        min_new_samples=args.min_new_samples,
        cooldown_batches=args.cooldown_batches,
        anchor_fraction=0.0,
        window_batches=args.window_batches,
    )

    strategy_specs = [
        ("Static", "static", policy_sliding),
        ("Policy-CurrentAnchor", "policy", policy_current_anchor),
        ("Policy-SlidingNoAnchor", "policy", policy_sliding),
    ]
    if args.include_immediate:
        strategy_specs.append(("Immediate-SlidingNoAnchor", "immediate", policy_sliding))

    histories = []
    summaries = []
    for drift_mode in ABLATION_MODES:
        base_drift_config = (
            label_noise_drift_config
            if drift_mode in LABEL_FLIP_MODES
            else learnable_drift_config
        )
        batches = generate_batches_for_mode(
            test_df=test_df,
            base_drift_config=base_drift_config,
            mode=drift_mode,
            seed=seed,
        )
        for strategy_name, strategy, policy_config in strategy_specs:
            history_df, summary = simulate_strategy(
                drift_mode=drift_mode,
                strategy_name=strategy_name,
                strategy=strategy,
                batches=batches,
                base_model=base_model,
                base_encoder=base_encoder,
                original_train_data=original_train_data,
                base_metrics=base_metrics,
                policy_config=policy_config,
                settings=settings,
            )
            histories.append(history_df)
            summaries.append(summary)

    all_history = pd.concat(histories, ignore_index=True)
    summary_df = add_static_gains(pd.DataFrame(summaries))

    profile = settings["data"].get("dataset_profile", "adult").lower()
    output_dir = get_path("reports_dir", settings) / "diagnostics" / f"{profile}_noise_ablation"
    save_outputs(all_history, summary_df, output_dir)

    print("\n=== Noise Ablation Summary ===")
    print(summary_df.to_string(index=False))
    print(f"\nSaved ablation outputs to {output_dir}")


if __name__ == "__main__":
    main()
