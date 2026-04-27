"""
Retraining module for the Drift-Aware MLOps Pipeline.

Handles retraining the model when the policy engine decides a retrain
is needed. Supports incremental data accumulation and model versioning.
"""

import time
import pickle
from pathlib import Path

import pandas as pd
import numpy as np
import xgboost as xgb
import mlflow
import mlflow.xgboost

from src.utils import (
    load_settings,
    get_path,
    ensure_dir,
    setup_logging,
)
from src.feature_encoding import EncoderPipeline
from src.train import train_model, save_model
from src.evaluate import compute_metrics

logger = setup_logging("retrain", log_file="retrain.log")


class ModelVersionManager:
    """
    Track model versions and manage retraining history.

    Attributes
    ----------
    current_version : int
        Current model version number.
    history : list[dict]
        List of retraining events with metadata.
    """

    def __init__(self):
        self.current_version = 1
        self.history = []

    def next_version(self) -> str:
        """Increment and return the next model version string."""
        self.current_version += 1
        return f"v{self.current_version}"

    def get_current_version_str(self) -> str:
        """Return the current version string."""
        return f"v{self.current_version}"

    def record_retrain(self, metadata: dict) -> None:
        """Record a retraining event."""
        metadata["version"] = self.get_current_version_str()
        self.history.append(metadata)
        logger.info(f"Recorded retrain event: {metadata}")

    def get_history_df(self) -> pd.DataFrame:
        """Return retraining history as a DataFrame."""
        if not self.history:
            return pd.DataFrame()
        return pd.DataFrame(self.history)


def retrain_model(
    new_data: pd.DataFrame,
    encoder: EncoderPipeline,
    version_manager: ModelVersionManager,
    old_train_data: pd.DataFrame | None = None,
    use_mlflow: bool = True,
) -> dict:
    """
    Retrain the model on new (or accumulated) data.

    Parameters
    ----------
    new_data : pd.DataFrame
        New labeled data to train on (raw, pre-encoding).
    encoder : EncoderPipeline
        Fitted encoder pipeline.
    version_manager : ModelVersionManager
        Version tracker.
    old_train_data : pd.DataFrame, optional
        Previous training data to combine with new data.
    use_mlflow : bool
        Whether to log to MLflow.

    Returns
    -------
    dict
        Contains 'model', 'version', 'metrics', 'model_path', 'train_data'.
    """
    settings = load_settings()
    target_col = settings["data"]["target_column"]
    feature_cols = encoder.get_feature_names()

    # Combine old + new data if available
    if old_train_data is not None:
        combined = pd.concat([old_train_data, new_data], ignore_index=True)
        logger.info(
            f"Combined training data: {len(old_train_data)} old + "
            f"{len(new_data)} new = {len(combined)} total"
        )
    else:
        combined = new_data
        logger.info(f"Training on {len(combined)} samples (no old data)")

    # Encode
    encoded = encoder.transform(combined, include_target=True)
    X = encoded[feature_cols]
    y = encoded[target_col]

    # Train/validation split for early stopping (use last 20% as validation)
    split_idx = int(len(X) * 0.8)
    X_train, X_val = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_val = y.iloc[:split_idx], y.iloc[split_idx:]

    # Train
    version = version_manager.next_version()
    logger.info(f"Retraining model {version}...")

    start_time = time.time()
    model = train_model(X_train, y_train, X_val, y_val)
    train_duration = time.time() - start_time

    # Evaluate on validation set
    y_pred = model.predict(X_val)
    metrics = compute_metrics(y_val.values, y_pred, prefix="retrain_")
    metrics["retrain_duration_seconds"] = train_duration
    metrics["retrain_n_samples"] = len(combined)

    # Save model
    model_path = save_model(model, version=version)

    # Record in version manager
    version_manager.record_retrain({
        "train_samples": len(combined),
        "new_samples": len(new_data),
        "train_duration": train_duration,
        **metrics,
    })

    # MLflow logging
    if use_mlflow:
        try:
            settings = load_settings()
            mlflow.set_tracking_uri(settings["mlflow"]["tracking_uri"])
            mlflow.set_experiment(settings["mlflow"]["experiment_name"])

            with mlflow.start_run(run_name=f"retrain_{version}"):
                mlflow.log_param("model_version", version)
                mlflow.log_param("n_train_samples", len(combined))
                mlflow.log_param("n_new_samples", len(new_data))
                mlflow.log_param("retrain_trigger", "policy")

                for key, val in metrics.items():
                    mlflow.log_metric(key, val)

                mlflow.xgboost.log_model(model, "model")
                logger.info(f"Logged retrain {version} to MLflow")

        except Exception as e:
            logger.warning(f"MLflow logging failed: {e}")

    result = {
        "model": model,
        "version": version,
        "metrics": metrics,
        "model_path": model_path,
        "train_data": combined,
    }

    logger.info(f"Retrain complete: {version}, acc={metrics['retrain_accuracy']:.4f}")
    return result
