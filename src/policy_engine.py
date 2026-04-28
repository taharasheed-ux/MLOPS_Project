"""
Retraining Policy Engine for the Drift-Aware MLOps Pipeline.

Evaluates multiple conditions before triggering a model retrain to avoid
constant retraining and ensure enough labeled data is available.
"""

from dataclasses import dataclass
from src.utils import load_thresholds, setup_logging
from src.drift_detection import BatchDriftResult

logger = setup_logging("policy_engine", log_file="policy_engine.log")


@dataclass
class PolicyDecision:
    """Result of a retraining policy evaluation."""
    should_retrain: bool
    severity_met: bool
    concept_drift_met: bool
    persistence_met: bool
    samples_met: bool
    cooldown_met: bool
    reason: str
    metric_drops: dict
    consecutive_signal_batches: int


class RetrainingPolicyEngine:
    """
    Evaluates whether retraining should occur based on three conditions:
    1. Global drift severity exceeds threshold
    2. Minimum number of new labeled samples accumulated
    3. Cooldown period has passed since the last retrain
    """

    def __init__(self, config: dict | None = None):
        if config is None:
            config = load_thresholds()

        policy_cfg = config.get("retraining_policy", {})
        self.severity_threshold = policy_cfg.get("severity_threshold", 0.3)
        self.min_new_samples = policy_cfg.get("min_new_samples", 500)
        self.cooldown_batches = policy_cfg.get("cooldown_batches", 1)
        self.persistence_batches = policy_cfg.get("persistence_batches", 1)
        concept_cfg = policy_cfg.get("concept_drift", {})
        self.concept_drift_enabled = concept_cfg.get("enabled", True)
        self.min_f1_drop = concept_cfg.get("min_f1_drop", 0.08)
        self.min_recall_drop = concept_cfg.get("min_recall_drop", 0.08)
        self.signal_combination = policy_cfg.get("signal_combination", "either")

        # Internal state
        self.batches_since_retrain = self.cooldown_batches  # Start ready to retrain
        self.consecutive_signal_batches = 0
        self._last_drift_signal_met = False
        
        logger.info(
            f"Policy Engine initialized: severity>={self.severity_threshold}, "
            f"new_samples>={self.min_new_samples}, cooldown>={self.cooldown_batches}, "
            f"concept_enabled={self.concept_drift_enabled}, "
            f"persistence>={self.persistence_batches}"
        )

    def evaluate(
        self,
        drift_result: BatchDriftResult,
        current_new_samples: int,
        current_metrics: dict | None = None,
        baseline_metrics: dict | None = None,
    ) -> PolicyDecision:
        """
        Evaluate if a retrain is needed based on the current state.

        Parameters
        ----------
        drift_result : BatchDriftResult
            The output from the DriftDetector for the current batch.
        current_new_samples : int
            Total number of new labeled samples accumulated since last retrain.

        Returns
        -------
        PolicyDecision
        """
        logger.info(
            f"Evaluating policy for {drift_result.batch_name} - "
            f"severity={drift_result.severity_score:.4f}, accumulated_samples={current_new_samples}, "
            f"batches_since_retrain={self.batches_since_retrain}"
        )

        severity_met = drift_result.severity_score >= self.severity_threshold
        metric_drops = self._compute_metric_drops(current_metrics, baseline_metrics)
        concept_drift_met = self._concept_drift_met(metric_drops)
        samples_met = current_new_samples >= self.min_new_samples
        cooldown_met = self.batches_since_retrain >= self.cooldown_batches

        if self.signal_combination == "all":
            drift_signal_met = severity_met and concept_drift_met
        else:
            drift_signal_met = severity_met or concept_drift_met

        consecutive_signal_batches = (
            self.consecutive_signal_batches + 1 if drift_signal_met else 0
        )
        persistence_met = consecutive_signal_batches >= self.persistence_batches
        self._last_drift_signal_met = drift_signal_met

        should_retrain = drift_signal_met and persistence_met and samples_met and cooldown_met

        reasons = []
        if should_retrain:
            reasons.append("All conditions met. Triggering retrain.")
        else:
            if not severity_met:
                reasons.append(f"Severity ({drift_result.severity_score:.4f}) < {self.severity_threshold}")
            if self.concept_drift_enabled and not concept_drift_met:
                if metric_drops:
                    reasons.append(
                        "Concept drift below thresholds "
                        f"(f1_drop={metric_drops.get('f1_drop', 0.0):.4f}, "
                        f"recall_drop={metric_drops.get('recall_drop', 0.0):.4f})"
                    )
                else:
                    reasons.append("Concept drift unavailable (no labeled performance metrics)")
            if not persistence_met:
                reasons.append(
                    f"Persistence ({consecutive_signal_batches}) < {self.persistence_batches}"
                )
            if not samples_met:
                reasons.append(f"Samples ({current_new_samples}) < {self.min_new_samples}")
            if not cooldown_met:
                reasons.append(f"Cooldown ({self.batches_since_retrain}) < {self.cooldown_batches}")

        reason_str = " | ".join(reasons)
        logger.info(f"Policy decision: RETRAIN={should_retrain} -> {reason_str}")

        return PolicyDecision(
            should_retrain=should_retrain,
            severity_met=severity_met,
            concept_drift_met=concept_drift_met,
            persistence_met=persistence_met,
            samples_met=samples_met,
            cooldown_met=cooldown_met,
            reason=reason_str,
            metric_drops=metric_drops,
            consecutive_signal_batches=consecutive_signal_batches,
        )

    def _compute_metric_drops(
        self,
        current_metrics: dict | None,
        baseline_metrics: dict | None,
    ) -> dict:
        if not current_metrics or not baseline_metrics:
            return {}

        drops = {}
        for metric_name in ("accuracy", "f1", "precision", "recall"):
            if metric_name in current_metrics and metric_name in baseline_metrics:
                drops[f"{metric_name}_drop"] = max(
                    0.0, baseline_metrics[metric_name] - current_metrics[metric_name]
                )
        return drops

    def _concept_drift_met(self, metric_drops: dict) -> bool:
        if not self.concept_drift_enabled:
            return False
        if not metric_drops:
            return False

        return (
            metric_drops.get("f1_drop", 0.0) >= self.min_f1_drop
            or metric_drops.get("recall_drop", 0.0) >= self.min_recall_drop
        )

    def register_batch_processed(self, retrain_triggered: bool) -> None:
        """
        Update the internal state after a batch processing cycle.
        Call this after evaluation and potential retraining.

        Parameters
        ----------
        retrain_triggered : bool
            Whether a retrain actually happened for this batch.
        """
        if self._last_drift_signal_met:
            self.consecutive_signal_batches += 1
        else:
            self.consecutive_signal_batches = 0

        if retrain_triggered:
            self.batches_since_retrain = 0
            self.consecutive_signal_batches = 0
            logger.info("Retrain registered. Resetting cooldown counter.")
        else:
            self.batches_since_retrain += 1
            logger.info(
                f"No retrain. Cooldown counter incremented to {self.batches_since_retrain}; "
                f"consecutive_signal_batches={self.consecutive_signal_batches}"
            )
