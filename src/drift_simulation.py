"""
Drift simulation module for the Drift-Aware MLOps Pipeline.

Creates 5 time-ordered evaluation batches from the test data, applying
controlled perturbations defined in configs/drift_config.yaml.

Drift types:
  - Covariate shift: modify feature distributions (shift, scale, resample)
  - Conditional shift: flip labels for specific subgroups
  - Feature importance shift: add noise to key predictive features
"""

import numpy as np
import pandas as pd
from pathlib import Path

from src.utils import load_drift_config, load_settings, get_path, ensure_dir, setup_logging

logger = setup_logging("drift_simulation", log_file="drift_simulation.log")


class DriftSimulator:
    """
    Generate evaluation batches with controlled distribution shifts.

    Parameters
    ----------
    test_df : pd.DataFrame
        The test split to sample from and perturb.
    config : dict, optional
        Drift config. If None, loads from drift_config.yaml.
    seed : int, optional
        Random seed. If None, uses settings.yaml.
    """

    def __init__(
        self,
        test_df: pd.DataFrame,
        config: dict | None = None,
        seed: int | None = None,
    ):
        self.test_df = test_df.copy()
        self.config = config or load_drift_config()
        self.settings = load_settings()

        if seed is None:
            seed = self.settings["project"]["random_seed"]
        self.rng = np.random.RandomState(seed)

        self.batch_size = self.config.get("batch_size", 2000)
        self.target_col = self.settings["data"]["target_column"]

    def generate_batches(self) -> dict[str, pd.DataFrame]:
        """
        Generate all evaluation batches defined in the config.

        Returns
        -------
        dict[str, pd.DataFrame]
            Mapping of batch name → perturbed dataframe.
        """
        batches = {}

        for batch_name, batch_cfg in self.config["batches"].items():
            logger.info(f"Generating {batch_name}: {batch_cfg.get('description', '')}")

            # Sample from test data
            batch_df = self.test_df.sample(
                n=min(self.batch_size, len(self.test_df)),
                replace=True,
                random_state=self.rng.randint(0, 100000),
            ).reset_index(drop=True)

            # Apply perturbations
            perturbations = batch_cfg.get("perturbations", {})

            if "covariate_shift" in perturbations:
                batch_df = self._apply_covariate_shift(
                    batch_df, perturbations["covariate_shift"]
                )

            if "conditional_shift" in perturbations:
                batch_df = self._apply_conditional_shift(
                    batch_df, perturbations["conditional_shift"]
                )

            if "feature_importance_shift" in perturbations:
                batch_df = self._apply_feature_importance_shift(
                    batch_df, perturbations["feature_importance_shift"]
                )

            batches[batch_name] = batch_df
            logger.info(f"  {batch_name}: {len(batch_df)} rows generated")

        return batches

    def _apply_covariate_shift(
        self, df: pd.DataFrame, shift_config: dict
    ) -> pd.DataFrame:
        """
        Apply covariate shift: modify feature distributions.

        Supported methods:
            - shift: add a constant value (numerical only)
            - scale: multiply by a factor (numerical only)
            - resample: redistribute categorical feature values
        """
        df = df.copy()

        for feature, params in shift_config.items():
            if feature not in df.columns:
                logger.warning(f"  Feature '{feature}' not found, skipping")
                continue

            method = params.get("method", "shift")

            if method == "shift":
                value = params["value"]
                df[feature] = df[feature] + value
                logger.info(f"  Covariate shift: {feature} += {value}")

            elif method == "scale":
                value = params["value"]
                df[feature] = df[feature] * value
                logger.info(f"  Covariate scale: {feature} *= {value}")

            elif method == "resample":
                df = self._resample_categorical(df, feature, params)

        return df

    def _resample_categorical(
        self, df: pd.DataFrame, feature: str, params: dict
    ) -> pd.DataFrame:
        """Resample a categorical feature to match a target distribution."""
        target_dist = params.get("target_distribution", {})
        if not target_dist:
            return df

        df = df.copy()
        n = len(df)

        # Get all unique values
        all_values = df[feature].unique().tolist()

        # Build probability mapping
        probs = {}
        other_prob = target_dist.get("Other", 0.0)
        specified_values = [v for v in target_dist if v != "Other"]

        for val in all_values:
            if val in specified_values:
                probs[val] = target_dist[val]
            else:
                # Distribute "Other" probability equally among remaining values
                other_values = [v for v in all_values if v not in specified_values]
                if other_values:
                    probs[val] = other_prob / len(other_values)
                else:
                    probs[val] = 0.0

        # Normalize probabilities
        total = sum(probs.values())
        if total > 0:
            probs = {k: v / total for k, v in probs.items()}

        # Resample
        categories = list(probs.keys())
        probabilities = list(probs.values())
        df[feature] = self.rng.choice(categories, size=n, p=probabilities)

        logger.info(f"  Resampled {feature} to target distribution")
        return df

    def _apply_conditional_shift(
        self, df: pd.DataFrame, conditions: list[dict]
    ) -> pd.DataFrame:
        """
        Apply conditional shift: flip labels for specific subgroups.

        Each condition specifies:
            - feature: column to evaluate
            - operator: comparison operator (>=, >, <=, <, ==, !=)
            - value: threshold value
            - flip_probability: fraction of matching rows to flip
        """
        df = df.copy()

        for cond_cfg in conditions:
            condition = cond_cfg["condition"]
            flip_prob = cond_cfg["flip_probability"]

            feature = condition["feature"]
            operator = condition["operator"]
            value = condition["value"]

            # Build boolean mask
            mask = self._evaluate_condition(df, feature, operator, value)
            n_matching = mask.sum()

            if n_matching == 0:
                logger.warning(f"  No rows match condition: {feature} {operator} {value}")
                continue

            # Randomly select rows to flip
            flip_indices = df.index[mask]
            n_flip = int(n_matching * flip_prob)
            flip_idx = self.rng.choice(flip_indices, size=n_flip, replace=False)

            # Flip labels (swap 0↔1 for encoded, or swap text labels)
            target = self.target_col
            if df[target].dtype == "object":
                # Text labels
                label_map = {">50K": "<=50K", "<=50K": ">50K"}
                df.loc[flip_idx, target] = df.loc[flip_idx, target].map(
                    lambda x: label_map.get(x, x)
                )
            else:
                # Numeric labels
                df.loc[flip_idx, target] = 1 - df.loc[flip_idx, target]

            logger.info(
                f"  Conditional shift: flipped {n_flip}/{n_matching} labels "
                f"where {feature} {operator} {value}"
            )

        return df

    def _apply_feature_importance_shift(
        self, df: pd.DataFrame, noise_config: dict
    ) -> pd.DataFrame:
        """
        Apply feature importance shift: add Gaussian noise to key features.

        noise_std = noise_std_factor * feature_std
        """
        df = df.copy()

        for feature, params in noise_config.items():
            if feature not in df.columns:
                logger.warning(f"  Feature '{feature}' not found, skipping")
                continue

            if not np.issubdtype(df[feature].dtype, np.number):
                logger.warning(f"  Feature '{feature}' is not numeric, skipping noise")
                continue

            factor = params["noise_std_factor"]
            feature_std = df[feature].std()
            noise = self.rng.normal(0, factor * feature_std, size=len(df))
            df[feature] = df[feature] + noise

            logger.info(
                f"  Importance shift: {feature} += N(0, {factor:.2f} * {feature_std:.2f})"
            )

        return df

    @staticmethod
    def _evaluate_condition(
        df: pd.DataFrame, feature: str, operator: str, value
    ) -> pd.Series:
        """Evaluate a comparison condition and return a boolean mask."""
        ops = {
            ">=": lambda col, v: col >= v,
            ">": lambda col, v: col > v,
            "<=": lambda col, v: col <= v,
            "<": lambda col, v: col < v,
            "==": lambda col, v: col == v,
            "!=": lambda col, v: col != v,
        }
        if operator not in ops:
            raise ValueError(f"Unsupported operator: {operator}")
        return ops[operator](df[feature], value)

    def save_batches(
        self, batches: dict[str, pd.DataFrame], output_dir: Path | None = None
    ) -> Path:
        """
        Save generated batches to CSV files.

        Parameters
        ----------
        batches : dict[str, pd.DataFrame]
            Output of generate_batches().
        output_dir : Path, optional
            Directory to save into. Defaults to data/batches/.

        Returns
        -------
        Path
            Output directory.
        """
        if output_dir is None:
            output_dir = get_path("batch_data_dir")

        output_dir = ensure_dir(output_dir)

        for name, df in batches.items():
            path = output_dir / f"{name}.csv"
            df.to_csv(path, index=False)
            logger.info(f"Saved {name} → {path}")

        return output_dir


def run_drift_simulation(
    test_df: pd.DataFrame | None = None,
) -> dict[str, pd.DataFrame]:
    """
    Run the full drift simulation pipeline.

    Parameters
    ----------
    test_df : pd.DataFrame, optional
        Test dataframe. If None, loads from data/processed/test.csv.

    Returns
    -------
    dict[str, pd.DataFrame]
        Named evaluation batches.
    """
    logger.info("=" * 60)
    logger.info("Starting drift simulation")
    logger.info("=" * 60)

    if test_df is None:
        processed_dir = get_path("processed_data_dir")
        test_path = processed_dir / "test.csv"
        if not test_path.exists():
            raise FileNotFoundError(
                f"Test data not found at {test_path}. Run data_processing first."
            )
        test_df = pd.read_csv(test_path)
        logger.info(f"Loaded test data: {len(test_df)} rows")

    simulator = DriftSimulator(test_df)
    batches = simulator.generate_batches()
    simulator.save_batches(batches)

    logger.info(f"Generated {len(batches)} evaluation batches")
    return batches


if __name__ == "__main__":
    run_drift_simulation()
