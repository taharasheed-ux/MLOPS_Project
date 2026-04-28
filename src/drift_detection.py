"""
Drift detection module for the Drift-Aware MLOps Pipeline.

Implements two-layer drift detection:
  Layer 1: Feature-level statistical tests (KS for numerical, Chi-square for categorical)
  Layer 2: Global drift severity score aggregation

Includes Bonferroni correction for multiple testing and effect size reporting.
"""

import numpy as np
import pandas as pd
from scipy import stats
from dataclasses import dataclass, field

from src.utils import load_settings, load_thresholds, setup_logging

logger = setup_logging("drift_detection", log_file="drift_detection.log")


# ── Data classes for structured results ─────────────────────────────

@dataclass
class FeatureDriftResult:
    """Result of drift detection for a single feature."""
    feature: str
    feature_type: str              # "numerical" or "categorical"
    test_name: str                 # "ks_test" or "chi_square"
    statistic: float               # Test statistic (KS stat or chi2 stat)
    p_value: float                 # Raw p-value
    p_value_corrected: float       # Corrected p-value (Bonferroni / BH)
    effect_size: float             # KS statistic or Cramér's V
    is_drifted: bool               # Whether drift is significant after correction
    drift_magnitude: str           # "none", "low", "medium", "high"


@dataclass
class BatchDriftResult:
    """Aggregated drift result for an entire batch."""
    batch_name: str
    feature_results: list[FeatureDriftResult]
    n_features_tested: int
    n_features_drifted: int
    severity_score: float
    drift_detected: bool
    drifted_features: list[str] = field(default_factory=list)
    severity_components: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Convert to a flat dictionary for logging/MLflow."""
        return {
            "batch_name": self.batch_name,
            "n_features_tested": self.n_features_tested,
            "n_features_drifted": self.n_features_drifted,
            "severity_score": self.severity_score,
            "drift_detected": self.drift_detected,
            "drifted_features": ", ".join(self.drifted_features),
            **{
                f"severity_{key}": value
                for key, value in self.severity_components.items()
            },
        }

    def to_feature_df(self) -> pd.DataFrame:
        """Convert feature results to a DataFrame."""
        rows = []
        for fr in self.feature_results:
            rows.append({
                "feature": fr.feature,
                "type": fr.feature_type,
                "test": fr.test_name,
                "statistic": fr.statistic,
                "p_value": fr.p_value,
                "p_value_corrected": fr.p_value_corrected,
                "effect_size": fr.effect_size,
                "is_drifted": fr.is_drifted,
                "magnitude": fr.drift_magnitude,
            })
        return pd.DataFrame(rows)


# ── Drift Detector ──────────────────────────────────────────────────

class DriftDetector:
    """
    Two-layer drift detector.

    Layer 1: Feature-level tests (KS / Chi-square)
    Layer 2: Global severity score

    Parameters
    ----------
    reference_data : pd.DataFrame
        Reference distribution (training data).
    numerical_features : list[str]
        Numerical feature names.
    categorical_features : list[str]
        Categorical feature names.
    config : dict, optional
        Threshold config. Loads from thresholds.yaml if None.
    """

    def __init__(
        self,
        reference_data: pd.DataFrame,
        numerical_features: list[str] | None = None,
        categorical_features: list[str] | None = None,
        config: dict | None = None,
    ):
        if config is None:
            config = load_thresholds()

        if numerical_features is None or categorical_features is None:
            settings = load_settings()
            numerical_features = numerical_features or settings["data"]["numerical_features"]
            categorical_features = categorical_features or settings["data"]["categorical_features"]

        self.reference_data = reference_data
        self.numerical_features = [f for f in numerical_features if f in reference_data.columns]
        self.categorical_features = [f for f in categorical_features if f in reference_data.columns]
        self.all_features = self.numerical_features + self.categorical_features

        # Config
        drift_cfg = config.get("drift_detection", {})
        self.alpha = drift_cfg.get("alpha", 0.05)
        self.correction_method = drift_cfg.get("correction_method", "bonferroni")
        self.min_effect_numerical = drift_cfg.get("min_effect_size", {}).get("numerical", 0.1)
        self.min_effect_categorical = drift_cfg.get("min_effect_size", {}).get("categorical", 0.1)

        severity_cfg = config.get("severity", {})
        self.severity_threshold = severity_cfg.get("threshold", 0.3)
        self.weighting = severity_cfg.get("weighting", "uniform")

        logger.info(
            f"DriftDetector initialized: {len(self.numerical_features)} numerical, "
            f"{len(self.categorical_features)} categorical features, "
            f"α={self.alpha}, correction={self.correction_method}"
        )

    def detect(self, current_data: pd.DataFrame, batch_name: str = "unknown") -> BatchDriftResult:
        """
        Run full drift detection on a batch.

        Parameters
        ----------
        current_data : pd.DataFrame
            Current batch to compare against reference.
        batch_name : str
            Name for logging.

        Returns
        -------
        BatchDriftResult
            Aggregated drift results.
        """
        logger.info(f"Running drift detection on {batch_name}...")

        # Layer 1: Feature-level tests
        feature_results = []

        for feature in self.numerical_features:
            if feature in current_data.columns:
                result = self._test_numerical(feature, current_data)
                feature_results.append(result)

        for feature in self.categorical_features:
            if feature in current_data.columns:
                result = self._test_categorical(feature, current_data)
                feature_results.append(result)

        # Apply multiple testing correction
        feature_results = self._apply_correction(feature_results)

        # Determine drift status based on corrected p-values AND effect sizes
        for fr in feature_results:
            min_effect = (
                self.min_effect_numerical
                if fr.feature_type == "numerical"
                else self.min_effect_categorical
            )
            fr.is_drifted = (
                fr.p_value_corrected < self.alpha and fr.effect_size >= min_effect
            )
            fr.drift_magnitude = self._classify_magnitude(fr.effect_size, fr.feature_type)

        # Layer 2: Global severity score
        n_tested = len(feature_results)
        n_drifted = sum(1 for fr in feature_results if fr.is_drifted)
        drifted_names = [fr.feature for fr in feature_results if fr.is_drifted]

        severity, severity_components = self._compute_severity(feature_results)
        drift_detected = severity >= self.severity_threshold

        result = BatchDriftResult(
            batch_name=batch_name,
            feature_results=feature_results,
            n_features_tested=n_tested,
            n_features_drifted=n_drifted,
            severity_score=severity,
            drift_detected=drift_detected,
            drifted_features=drifted_names,
            severity_components=severity_components,
        )

        logger.info(
            f"  {batch_name}: {n_drifted}/{n_tested} features drifted, "
            f"severity={severity:.4f}, drift_detected={drift_detected}"
        )
        if drifted_names:
            logger.info(f"  Drifted features: {drifted_names}")

        return result

    def _test_numerical(self, feature: str, current_data: pd.DataFrame) -> FeatureDriftResult:
        """Kolmogorov-Smirnov test for numerical features."""
        ref_values = self.reference_data[feature].dropna().values
        cur_values = current_data[feature].dropna().values

        statistic, p_value = stats.ks_2samp(ref_values, cur_values)

        return FeatureDriftResult(
            feature=feature,
            feature_type="numerical",
            test_name="ks_test",
            statistic=statistic,
            p_value=p_value,
            p_value_corrected=p_value,  # Will be corrected later
            effect_size=statistic,      # KS statistic IS the effect size
            is_drifted=False,
            drift_magnitude="none",
        )

    def _test_categorical(self, feature: str, current_data: pd.DataFrame) -> FeatureDriftResult:
        """Chi-square test for categorical features."""
        ref_values = self.reference_data[feature].dropna()
        cur_values = current_data[feature].dropna()

        # Get all unique categories from both datasets
        all_categories = set(ref_values.unique()) | set(cur_values.unique())

        # Build frequency tables aligned on the same categories
        ref_counts = ref_values.value_counts()
        cur_counts = cur_values.value_counts()

        ref_freq = np.array([ref_counts.get(cat, 0) for cat in all_categories], dtype=float)
        cur_freq = np.array([cur_counts.get(cat, 0) for cat in all_categories], dtype=float)

        # Normalize to expected frequencies (scale ref to match cur sample size)
        ref_expected = ref_freq * (cur_freq.sum() / ref_freq.sum()) if ref_freq.sum() > 0 else ref_freq

        # Avoid zero expected frequencies
        ref_expected = np.maximum(ref_expected, 1e-10)

        try:
            statistic, p_value = stats.chisquare(cur_freq, f_exp=ref_expected)
        except ValueError:
            statistic, p_value = 0.0, 1.0

        # Cramér's V as effect size
        n = cur_freq.sum()
        k = len(all_categories)
        cramers_v = np.sqrt(statistic / (n * max(k - 1, 1))) if n > 0 and statistic > 0 else 0.0
        cramers_v = min(cramers_v, 1.0)  # Cap at 1.0

        return FeatureDriftResult(
            feature=feature,
            feature_type="categorical",
            test_name="chi_square",
            statistic=statistic,
            p_value=p_value,
            p_value_corrected=p_value,
            effect_size=cramers_v,
            is_drifted=False,
            drift_magnitude="none",
        )

    def _apply_correction(
        self, results: list[FeatureDriftResult]
    ) -> list[FeatureDriftResult]:
        """Apply multiple testing correction to p-values."""
        if not results:
            return results

        p_values = np.array([r.p_value for r in results])
        n_tests = len(p_values)

        if self.correction_method == "bonferroni":
            corrected = np.minimum(p_values * n_tests, 1.0)
        elif self.correction_method == "benjamini-hochberg":
            corrected = self._benjamini_hochberg(p_values)
        else:
            logger.warning(f"Unknown correction: {self.correction_method}, using bonferroni")
            corrected = np.minimum(p_values * n_tests, 1.0)

        for i, result in enumerate(results):
            result.p_value_corrected = corrected[i]

        return results

    @staticmethod
    def _benjamini_hochberg(p_values: np.ndarray) -> np.ndarray:
        """Benjamini-Hochberg FDR correction."""
        n = len(p_values)
        sorted_idx = np.argsort(p_values)
        sorted_p = p_values[sorted_idx]

        # BH correction: p_adj = p * n / rank
        ranks = np.arange(1, n + 1)
        corrected = sorted_p * n / ranks

        # Enforce monotonicity (take cumulative minimum from the right)
        corrected = np.minimum.accumulate(corrected[::-1])[::-1]
        corrected = np.minimum(corrected, 1.0)

        # Restore original order
        result = np.empty(n)
        result[sorted_idx] = corrected
        return result

    def _compute_severity(
        self, results: list[FeatureDriftResult]
    ) -> tuple[float, dict[str, float]]:
        """
        Compute global drift severity score.

        The score blends:
        - drift ratio: how many features are materially drifted
        - mean drifted effect: average effect among flagged features
        - mean overall effect: background feature movement across all features
        """
        if not results:
            return 0.0, {}

        n_total = len(results)
        drifted = [r for r in results if r.is_drifted]

        if not drifted:
            return 0.0, {
                "drift_ratio": 0.0,
                "mean_drifted_effect": 0.0,
                "mean_overall_effect": float(np.mean([r.effect_size for r in results])),
            }

        drift_ratio = len(drifted) / n_total
        mean_drifted_effect = float(np.mean([r.effect_size for r in drifted]))
        mean_overall_effect = float(np.mean([r.effect_size for r in results]))

        if self.weighting == "uniform":
            severity = (
                0.50 * mean_drifted_effect
                + 0.30 * drift_ratio
                + 0.20 * mean_overall_effect
            )
        else:
            severity = (
                0.50 * mean_drifted_effect
                + 0.30 * drift_ratio
                + 0.20 * mean_overall_effect
            )

        return min(severity, 1.0), {
            "drift_ratio": drift_ratio,
            "mean_drifted_effect": mean_drifted_effect,
            "mean_overall_effect": mean_overall_effect,
        }

    @staticmethod
    def _classify_magnitude(effect_size: float, feature_type: str) -> str:
        """Classify drift magnitude based on effect size."""
        if feature_type == "numerical":
            if effect_size < 0.1:
                return "none"
            elif effect_size < 0.2:
                return "low"
            elif effect_size < 0.4:
                return "medium"
            else:
                return "high"
        else:  # categorical (Cramér's V)
            if effect_size < 0.1:
                return "none"
            elif effect_size < 0.2:
                return "low"
            elif effect_size < 0.3:
                return "medium"
            else:
                return "high"

    def update_reference(self, new_reference: pd.DataFrame) -> None:
        """
        Update the reference distribution (after a successful retrain).

        Parameters
        ----------
        new_reference : pd.DataFrame
            New training data to use as reference.
        """
        self.reference_data = new_reference
        logger.info(f"Reference distribution updated: {len(new_reference)} rows")


def run_drift_detection(
    reference_df: pd.DataFrame | None = None,
    batches: dict[str, pd.DataFrame] | None = None,
) -> dict[str, BatchDriftResult]:
    """
    Run drift detection across all batches.

    Parameters
    ----------
    reference_df : pd.DataFrame, optional
        Reference data. If None, loads from data/processed/train.csv.
    batches : dict[str, pd.DataFrame], optional
        Evaluation batches. If None, loads from data/batches/.

    Returns
    -------
    dict[str, BatchDriftResult]
        Results per batch.
    """
    from src.utils import get_path

    logger.info("=" * 60)
    logger.info("Starting drift detection pipeline")
    logger.info("=" * 60)

    if reference_df is None:
        from src.utils import get_processed_data_paths

        ref_path, _ = get_processed_data_paths()
        reference_df = pd.read_csv(ref_path)
        logger.info(f"Loaded reference data: {len(reference_df)} rows")

    if batches is None:
        batch_dir = get_path("batch_data_dir")
        batches = {}
        for i in range(1, 6):
            path = batch_dir / f"batch_{i}.csv"
            if path.exists():
                batches[f"batch_{i}"] = pd.read_csv(path)

    detector = DriftDetector(reference_df)
    results = {}

    for batch_name, batch_df in batches.items():
        result = detector.detect(batch_df, batch_name=batch_name)
        results[batch_name] = result

    # Summary
    logger.info("=" * 60)
    logger.info("Drift Detection Summary")
    logger.info("-" * 60)
    for name, result in results.items():
        logger.info(
            f"  {name}: severity={result.severity_score:.4f}, "
            f"drifted={result.n_features_drifted}/{result.n_features_tested}, "
            f"alert={'YES' if result.drift_detected else 'no'}"
        )
    logger.info("=" * 60)

    return results


if __name__ == "__main__":
    results = run_drift_detection()
    # Print feature-level details for each batch
    for name, result in results.items():
        print(f"\n{'='*60}")
        print(f"Batch: {name}")
        print(result.to_feature_df().to_string(index=False))
