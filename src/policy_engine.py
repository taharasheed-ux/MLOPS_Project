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
    samples_met: bool
    cooldown_met: bool
    reason: str


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

        # Internal state
        self.batches_since_retrain = self.cooldown_batches  # Start ready to retrain
        
        logger.info(
            f"Policy Engine initialized: severity>={self.severity_threshold}, "
            f"new_samples>={self.min_new_samples}, cooldown>={self.cooldown_batches}"
        )

    def evaluate(
        self,
        drift_result: BatchDriftResult,
        current_new_samples: int,
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
        samples_met = current_new_samples >= self.min_new_samples
        cooldown_met = self.batches_since_retrain >= self.cooldown_batches

        should_retrain = severity_met and samples_met and cooldown_met

        reasons = []
        if should_retrain:
            reasons.append("All conditions met. Triggering retrain.")
        else:
            if not severity_met:
                reasons.append(f"Severity ({drift_result.severity_score:.4f}) < {self.severity_threshold}")
            if not samples_met:
                reasons.append(f"Samples ({current_new_samples}) < {self.min_new_samples}")
            if not cooldown_met:
                reasons.append(f"Cooldown ({self.batches_since_retrain}) < {self.cooldown_batches}")

        reason_str = " | ".join(reasons)
        logger.info(f"Policy decision: RETRAIN={should_retrain} -> {reason_str}")

        return PolicyDecision(
            should_retrain=should_retrain,
            severity_met=severity_met,
            samples_met=samples_met,
            cooldown_met=cooldown_met,
            reason=reason_str,
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
        if retrain_triggered:
            self.batches_since_retrain = 0
            logger.info("Retrain registered. Resetting cooldown counter.")
        else:
            self.batches_since_retrain += 1
            logger.info(f"No retrain. Cooldown counter incremented to {self.batches_since_retrain}")
