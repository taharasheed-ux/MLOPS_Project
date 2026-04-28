import os
import time
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from src.utils import get_path, load_settings
from src.train import run_training_pipeline
from src.retrain import retrain_model, ModelVersionManager, prepare_retraining_data
from src.drift_detection import DriftDetector
from src.policy_engine import RetrainingPolicyEngine
from src.evaluate import compute_metrics, extract_core_metrics

def load_batches():
    batches = {}
    batch_dir = get_path("batch_data_dir")
    batch_paths = sorted(
        batch_dir.glob("batch_*.csv"),
        key=lambda p: int(p.stem.split("_")[-1]),
    )
    for path in batch_paths:
        batches[path.stem] = pd.read_csv(path)
    return batches

def simulate(
    policy_name,
    severity_threshold=2.0,
    min_new_samples=0,
    cooldown_batches=0,
    strategy="policy",
):
    print(f"[{policy_name}] Starting simulation...")
    
    # 1. Base model training
    res = run_training_pipeline(run_name=f"sim_{policy_name}", use_mlflow=False)
    base_model = res["model"]
    base_encoder = res["encoder"]
    from src.utils import get_processed_data_paths
    train_path, _ = get_processed_data_paths()
    train_data = pd.read_csv(train_path)
    original_train_data = train_data.copy()
    baseline_metrics = extract_core_metrics(res["metrics"], prefix="test_")
    
    version_mgr = ModelVersionManager()
    policy_engine = RetrainingPolicyEngine({
        "retraining_policy": {
            "severity_threshold": severity_threshold,
            "min_new_samples": min_new_samples,
            "cooldown_batches": cooldown_batches
        }
    })
    
    detector = DriftDetector(train_data)
    batches = load_batches()
    
    current_model = base_model
    current_encoder = base_encoder
    acc_samples = pd.DataFrame()
    recent_batches = []
    
    metrics_history = []
    
    total_retrain_time = 0.0
    total_inference_time = 0.0
    retrain_count = 0
    settings = load_settings()
    
    for batch_name, batch_df in batches.items():
        # Evaluate performance (Inference)
        inf_start = time.time()
        encoded = current_encoder.transform(batch_df, include_target=True)
        X = encoded[current_encoder.get_feature_names()]
        y = encoded[settings["data"]["target_column"]]
        y_pred = current_model.predict(X)
        inf_time = time.time() - inf_start
        total_inference_time += inf_time
        
        mets = compute_metrics(y, y_pred, prefix="")
        current_metrics = extract_core_metrics(mets)
        
        # Drift Detection
        drift_res = detector.detect(batch_df, batch_name)
        acc_samples = pd.concat([acc_samples, batch_df])
        recent_batches.append(batch_df.copy())
        
        # Policy
        decision = policy_engine.evaluate(
            drift_res,
            len(acc_samples),
            current_metrics=current_metrics,
            baseline_metrics=baseline_metrics,
        )
        
        retrain_triggered = False
        train_duration = 0.0

        if strategy == "static":
            should_retrain = False
        elif strategy == "immediate":
            should_retrain = (
                drift_res.drift_detected
                or decision.concept_drift_met
            )
        else:
            should_retrain = decision.should_retrain
        
        if should_retrain:
            retrain_triggered = True
            retrain_count += 1
            
            # Retrain
            retraining_df = prepare_retraining_data(
                recent_batches=recent_batches,
                original_train_data=original_train_data,
            )
            ret_start = time.time()
            ret_res = retrain_model(
                retraining_df,
                current_encoder,
                version_mgr,
                old_train_data=None,
                use_mlflow=False,
                new_data_sample_count=len(acc_samples),
            )
            train_duration = time.time() - ret_start
            total_retrain_time += train_duration
            
            current_model = ret_res["model"]
            current_encoder = ret_res["encoder"]
            train_data = ret_res["train_data"]
            acc_samples = pd.DataFrame() 
            detector.update_reference(train_data)
            baseline_metrics = extract_core_metrics(ret_res["metrics"], prefix="retrain_")
            
        policy_engine.register_batch_processed(retrain_triggered)
        
        metrics_history.append({
            "batch": batch_name,
            "model_version": version_mgr.get_current_version_str(),
            "accuracy": mets["accuracy"],
            "f1": mets["f1"],
            "precision": mets["precision"],
            "recall": mets["recall"],
            "drift_severity": drift_res.severity_score,
            "drift_flags": drift_res.n_features_drifted,
            "concept_drift": decision.concept_drift_met,
            "retrained": retrain_triggered,
            "train_duration": train_duration,
            "inference_time": inf_time
        })
        
    df_metrics = pd.DataFrame(metrics_history)
    
    summary = {
        "policy_name": policy_name,
        "mean_accuracy": df_metrics["accuracy"].mean(),
        "mean_f1": df_metrics["f1"].mean(),
        "retrain_count": retrain_count,
        "total_retrain_time": total_retrain_time,
        "total_inference_time": total_inference_time
    }
    summary.update(compute_research_summary(df_metrics))
    
    return df_metrics, summary

def generate_markdown_table(df):
    """Fallback markdown table generator without tabulate."""
    headers = df.columns.tolist()
    header_str = "| " + " | ".join(headers) + " |"
    sep_str = "| " + " | ".join(["---"] * len(headers)) + " |"
    rows = []
    for _, row in df.iterrows():
        row_str = "| " + " | ".join([str(x) for x in row.values]) + " |"
        rows.append(row_str)
    return "\n".join([header_str, sep_str] + rows)


def compute_research_summary(df_metrics: pd.DataFrame) -> dict:
    baseline_f1 = float(df_metrics.iloc[0]["f1"])
    baseline_acc = float(df_metrics.iloc[0]["accuracy"])
    f1_drops = baseline_f1 - df_metrics["f1"]
    acc_drops = baseline_acc - df_metrics["accuracy"]

    recovery_gains = []
    for idx, row in df_metrics.iterrows():
        if row["retrained"] and idx + 1 < len(df_metrics):
            recovery_gains.append(df_metrics.iloc[idx + 1]["f1"] - row["f1"])

    return {
        "worst_batch_accuracy": float(df_metrics["accuracy"].min()),
        "worst_batch_f1": float(df_metrics["f1"].min()),
        "mean_f1_drop_from_batch1": float(f1_drops.mean()),
        "max_f1_drop_from_batch1": float(f1_drops.max()),
        "mean_accuracy_drop_from_batch1": float(acc_drops.mean()),
        "degradation_area_f1": float(f1_drops.clip(lower=0).sum()),
        "degradation_area_accuracy": float(acc_drops.clip(lower=0).sum()),
        "concept_drift_batches": int(df_metrics["concept_drift"].sum()),
        "drift_alert_batches": int((df_metrics["drift_severity"] > 0).sum()),
        "post_retrain_avg_f1": (
            float(df_metrics.loc[df_metrics["retrained"], "f1"].mean())
            if df_metrics["retrained"].any()
            else 0.0
        ),
        "mean_recovery_gain_next_batch_f1": (
            float(sum(recovery_gains) / len(recovery_gains)) if recovery_gains else 0.0
        ),
    }


def get_report_paths(settings: dict) -> tuple[Path, Path]:
    profile = settings["data"].get("dataset_profile", "adult").lower()
    reports_dir = get_path("reports_dir", settings)
    figures_dir = get_path("figures_dir", settings) / profile
    os.makedirs(figures_dir, exist_ok=True)
    report_path = reports_dir / f"results_{profile}.md"
    return report_path, figures_dir


def build_conclusion(df_summary: pd.DataFrame) -> str:
    ranking = df_summary.sort_values(["mean_f1", "mean_accuracy"], ascending=False).reset_index(drop=True)
    best = ranking.iloc[0]
    policy_standard = df_summary[df_summary["policy_name"] == "Policy-Standard"].iloc[0]
    static = df_summary[df_summary["policy_name"] == "Static"].iloc[0]
    immediate = df_summary[df_summary["policy_name"] == "Immediate"].iloc[0]

    lines = [
        f"The best mean-F1 policy in this run was **{best['policy_name']}** "
        f"(mean_f1={best['mean_f1']:.4f}, mean_accuracy={best['mean_accuracy']:.4f}).",
        f"**Policy-Standard** retrained {int(policy_standard['retrain_count'])} times "
        f"versus {int(immediate['retrain_count'])} for **Immediate** and "
        f"{int(static['retrain_count'])} for **Static**.",
    ]

    if policy_standard["mean_f1"] > static["mean_f1"]:
        lines.append(
            "Policy-based retraining improved mean F1 over the static baseline "
            f"by {policy_standard['mean_f1'] - static['mean_f1']:.4f}."
        )
    else:
        lines.append(
            "Policy-based retraining did not beat the static baseline on mean F1 in this run; "
            "this suggests the simulated drift was detectable but not strong enough to justify "
            "every retrain under the current policy and perturbation design."
        )

    lines.append(
        f"Policy-Standard saw {int(policy_standard['concept_drift_batches'])} concept-drift batches "
        f"and achieved a mean next-batch F1 recovery gain of "
        f"{policy_standard['mean_recovery_gain_next_batch_f1']:.4f} after retraining events."
    )

    return " ".join(lines)

def main():
    settings = load_settings()
    report_path, figures_dir = get_report_paths(settings)
    
    results = {}
    summaries = []
    
    # Experiment A/B: Baselines and Adaptive Policies
    print("--- Experiment A & B: Static vs Immediate vs Policy ---")
    scenarios = {
        "Static": {"severity_threshold": 999.0, "min_new_samples": 0, "cooldown_batches": 0, "strategy": "static"},
        "Immediate": {"severity_threshold": 0.0, "min_new_samples": 0, "cooldown_batches": 0, "strategy": "immediate"},
        "Policy-Standard": {"severity_threshold": 0.15, "min_new_samples": 1000, "cooldown_batches": 1, "strategy": "policy"}
    }
    
    for name, args in scenarios.items():
        df_m, summ = simulate(name, **args)
        results[name] = df_m
        summaries.append(summ)

    # Experiment C: Threshold Sensitivity
    print("\n--- Experiment C: Threshold Sensitivities ---")
    sensitivities = [0.05, 0.25, 0.35]
    for thresh in sensitivities:
        name = f"Policy-Thresh{thresh}"
        df_m, summ = simulate(
            name,
            severity_threshold=thresh,
            min_new_samples=1000,
            cooldown_batches=1,
            strategy="policy",
        )
        results[name] = df_m
        summaries.append(summ)

    # Plot 1: Batch-wise metric curves (Accuracy & F1)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
    for name in ["Static", "Immediate", "Policy-Standard"]:
        df = results[name]
        ax1.plot(df["batch"], df["accuracy"], marker='o', label=name)
        ax2.plot(df["batch"], df["f1"], marker='s', label=name)
    ax1.set_title("Experiment A/B: Accuracy Recovery")
    ax1.set_ylabel("Accuracy")
    ax1.legend()
    ax1.grid(True)
    
    ax2.set_title("Experiment A/B: F1-Score Recovery")
    ax2.set_ylabel("F1 Score")
    ax2.legend()
    ax2.grid(True)
    plt.tight_layout()
    plt.savefig(figures_dir / "metrics_comparison.png")
    
    # Plot 2: Drift Timeline and Retraining Events
    df_pol = results["Policy-Standard"]
    plt.figure(figsize=(10, 5))
    plt.plot(df_pol["batch"], df_pol["drift_severity"], marker='x', color='red', label='Drift Severity')
    plt.axhline(y=0.15, color='orange', linestyle='--', label='Threshold (0.15)')
    
    # Retrain vlines
    for idx, row in df_pol.iterrows():
        if row["retrained"]:
            plt.axvline(x=idx, color='green', linestyle=':', label='Retrain Triggered' if idx==0 else "")
            # Annotate model version
            plt.text(idx, 0.05, row["model_version"], color='green')

    plt.title("Experiment B: Drift Score Timeline & Retraining Events (Policy-Standard)")
    plt.ylabel("Severity Score")
    # Clean up duplicate labels in legend
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())
    plt.grid(True)
    plt.savefig(figures_dir / "drift_and_retrains.png")
    
    # Plot 3: Cost vs Performance (Experiment D)
    df_summ = pd.DataFrame(summaries)
    plt.figure(figsize=(8, 6))
    plt.scatter(df_summ["total_retrain_time"], df_summ["mean_accuracy"], s=100)
    for i, row in df_summ.iterrows():
        plt.annotate(row["policy_name"], (row["total_retrain_time"], row["mean_accuracy"]), xytext=(5,5), textcoords='offset points')
    plt.title("Experiment D: Cost (Retrain Time) vs Performance (Mean Accuracy)")
    plt.xlabel("Total Retrain Time (seconds)")
    plt.ylabel("Mean Accuracy")
    plt.grid(True)
    plt.savefig(figures_dir / "cost_vs_performance.png")
    
    # Write profile-specific report
    with open(report_path, "w") as f:
        profile = settings["data"].get("dataset_profile", "adult").upper()
        f.write("# Drift-Aware Retraining Project - Final Report\n\n")
        f.write(f"Dataset profile: **{profile}**\n\n")
        
        f.write("## 1) Experiment A: Static vs Adaptive System\n")
        f.write("We compare a statically deployed model against a system capable of adaptive drift-aware retaining.\n")
        f.write("### Static Baseline Metrics\n")
        f.write(generate_markdown_table(results["Static"]) + "\n\n")
        
        f.write("## 2) Experiment B: Immediate vs Policy-Based Retraining\n")
        f.write("Immediate retraining heavily punishes system resources on every drift detected. Policy-based enforces cooldowns and sample accumulators.\n")
        f.write("### Immediate Retraining Metrics\n")
        f.write(generate_markdown_table(results["Immediate"]) + "\n\n")
        f.write("### Policy-Standard Metrics\n")
        f.write(generate_markdown_table(results["Policy-Standard"]) + "\n\n")
        
        f.write("## 3) Experiment C: Threshold Sensitivity\n")
        f.write("Demonstration of how different drift severity thresholds alter retraining frequencies.\n")
        sens_df = df_summ[df_summ["policy_name"].str.contains("Thresh")]
        f.write(generate_markdown_table(sens_df) + "\n\n")
        
        f.write("## 4) Experiment D: Cost vs Performance Trade-offs\n")
        f.write("Comparison of mean classification metrics against system latency and retrain durations across all configured batches.\n")
        f.write(generate_markdown_table(df_summ) + "\n\n")
        
        f.write("## Core Research Conclusion\n")
        f.write(build_conclusion(df_summ))

    print(
        "\nExperiments executed. Generated outputs located at "
        f"{report_path} and {figures_dir}/"
    )

if __name__ == "__main__":
    main()
