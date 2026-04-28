"""
Retraining module for the Drift-Aware MLOps Pipeline.

Handles retraining the model when the policy engine decides a retrain
is needed. Supports incremental data accumulation and model versioning.
"""

import time
import pandas as pd
import mlflow
import mlflow.xgboost

from src.utils import (
    load_settings,
    load_thresholds,
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


def _sample_anchor_data(
    original_train_data: pd.DataFrame,
    anchor_fraction: float,
    target_col: str,
    random_seed: int,
) -> pd.DataFrame:
    """Return a small stratified anchor sample from the original training data."""
    if original_train_data is None or original_train_data.empty or anchor_fraction <= 0:
        return pd.DataFrame(columns=original_train_data.columns if original_train_data is not None else None)

    anchor_fraction = min(max(anchor_fraction, 0.0), 1.0)
    samples = []
    for _, group in original_train_data.groupby(target_col):
        sample_n = max(1, int(len(group) * anchor_fraction))
        sample_n = min(sample_n, len(group))
        samples.append(group.sample(n=sample_n, random_state=random_seed))

    anchor_df = pd.concat(samples, ignore_index=True)
    return anchor_df.drop_duplicates().reset_index(drop=True)


def prepare_retraining_data(
    recent_batches: list[pd.DataFrame],
    original_train_data: pd.DataFrame,
    thresholds: dict | None = None,
    settings: dict | None = None,
) -> pd.DataFrame:
    """
    Build retraining data from a rolling window of recent batches plus a
    stratified anchor sample of the original training data.
    """
    if settings is None:
        settings = load_settings()
    if thresholds is None:
        thresholds = load_thresholds()

    retraining_cfg = thresholds.get("retraining_data", {})
    window_batches = retraining_cfg.get("window_batches", 3)
    anchor_fraction = retraining_cfg.get("anchor_fraction", 0.1)
    target_col = settings["data"]["target_column"]
    random_seed = settings["project"]["random_seed"]

    selected_batches = recent_batches[-window_batches:] if window_batches > 0 else recent_batches
    recent_df = pd.concat(selected_batches, ignore_index=True) if selected_batches else pd.DataFrame()
    anchor_df = _sample_anchor_data(
        original_train_data=original_train_data,
        anchor_fraction=anchor_fraction,
        target_col=target_col,
        random_seed=random_seed,
    )

    combined_parts = [df for df in (anchor_df, recent_df) if not df.empty]
    if not combined_parts:
        return pd.DataFrame(columns=original_train_data.columns)

    combined = pd.concat(combined_parts, ignore_index=True)
    combined = combined.drop_duplicates().reset_index(drop=True)
    logger.info(
        f"Prepared rolling retraining dataset: "
        f"anchor={len(anchor_df)}, recent={len(recent_df)}, total={len(combined)}, "
        f"window_batches={window_batches}, anchor_fraction={anchor_fraction}"
    )
    return combined


def retrain_model(
    new_data: pd.DataFrame,
    encoder: EncoderPipeline,
    version_manager: ModelVersionManager,
    old_train_data: pd.DataFrame | None = None,
    use_mlflow: bool = True,
    new_data_sample_count: int | None = None,
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

    if new_data_sample_count is None:
        new_data_sample_count = len(new_data)

    # Refit encoder on the accumulated raw data so new category values are learned.
    refreshed_encoder = EncoderPipeline()
    encoded = refreshed_encoder.fit_transform(combined, include_target=True)
    feature_cols = refreshed_encoder.get_feature_names()
    X = encoded[feature_cols]
    y = encoded[target_col]

    # Use a shuffled stratified split so validation better reflects the mixed stream.
    from sklearn.model_selection import train_test_split

    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=settings["project"]["random_seed"],
        stratify=y,
    )

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
    encoder_path = refreshed_encoder.save()

    # Record in version manager
    version_manager.record_retrain({
        "train_samples": len(combined),
        "new_samples": new_data_sample_count,
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
                mlflow.log_param("n_new_samples", new_data_sample_count)
                mlflow.log_param("retrain_trigger", "policy")

                for key, val in metrics.items():
                    mlflow.log_metric(key, val)

                mlflow.xgboost.log_model(model, "model")
                mlflow.log_artifact(str(encoder_path))
                logger.info(f"Logged retrain {version} to MLflow")

        except Exception as e:
            logger.warning(f"MLflow logging failed: {e}")

    result = {
        "model": model,
        "encoder": refreshed_encoder,
        "version": version,
        "metrics": metrics,
        "model_path": model_path,
        "encoder_path": encoder_path,
        "train_data": combined,
    }

    logger.info(f"Retrain complete: {version}, acc={metrics['retrain_accuracy']:.4f}")
    return result
