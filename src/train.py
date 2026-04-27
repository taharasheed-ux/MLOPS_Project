"""
Training module for the Drift-Aware MLOps Pipeline.

Trains an XGBoost classifier on the processed Adult Income dataset,
logs everything to MLflow (parameters, metrics, model artifact), and
evaluates on drift batches to establish baseline performance.
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
    PROJECT_ROOT,
)
from src.feature_encoding import EncoderPipeline
from src.evaluate import compute_metrics, compute_detailed_report

logger = setup_logging("train", log_file="train.log")


def load_processed_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load processed train/test CSVs from data/processed/."""
    processed_dir = get_path("processed_data_dir")
    train_df = pd.read_csv(processed_dir / "train.csv")
    test_df = pd.read_csv(processed_dir / "test.csv")
    logger.info(f"Loaded processed data: train={len(train_df)}, test={len(test_df)}")
    return train_df, test_df


def train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    params: dict | None = None,
) -> xgb.XGBClassifier:
    """
    Train an XGBoost classifier.

    Parameters
    ----------
    X_train, y_train : training data
    X_test, y_test : validation data (for early stopping)
    params : dict, optional
        XGBoost hyperparameters. Defaults to settings.yaml values.

    Returns
    -------
    xgb.XGBClassifier
        Trained model.
    """
    if params is None:
        settings = load_settings()
        params = settings["model"]["params"]

    logger.info(f"Training XGBoost with params: {params}")

    model = xgb.XGBClassifier(**params)

    start_time = time.time()
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False,
    )
    train_time = time.time() - start_time

    logger.info(f"Training completed in {train_time:.2f}s")
    return model


def save_model(
    model: xgb.XGBClassifier,
    filepath: str | Path | None = None,
    version: str = "v1",
) -> Path:
    """
    Save trained model to disk.

    Parameters
    ----------
    model : XGBClassifier
        Trained model.
    filepath : str or Path, optional
        Save path. Defaults to models/xgb_model_{version}.pkl.
    version : str
        Model version string.

    Returns
    -------
    Path
        Path to saved model.
    """
    if filepath is None:
        models_dir = ensure_dir(get_path("models_dir"))
        filepath = models_dir / f"xgb_model_{version}.pkl"
    else:
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

    with open(filepath, "wb") as f:
        pickle.dump(model, f)

    logger.info(f"Saved model to {filepath}")
    return filepath


def load_model(filepath: str | Path | None = None) -> xgb.XGBClassifier:
    """Load a trained model from disk."""
    if filepath is None:
        filepath = get_path("models_dir") / "xgb_model_v1.pkl"

    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Model not found: {filepath}")

    with open(filepath, "rb") as f:
        model = pickle.load(f)

    logger.info(f"Loaded model from {filepath}")
    return model


def run_training_pipeline(
    experiment_name: str | None = None,
    run_name: str = "baseline_training",
    use_mlflow: bool = True,
) -> dict:
    """
    Execute the full training pipeline:
    1. Load processed data
    2. Encode features
    3. Train XGBoost
    4. Evaluate on test set
    5. Log to MLflow
    6. Save model and encoder

    Parameters
    ----------
    experiment_name : str, optional
        MLflow experiment name. Defaults to settings.yaml value.
    run_name : str
        Name for this MLflow run.
    use_mlflow : bool
        Whether to log to MLflow. Set False for testing without server.

    Returns
    -------
    dict
        Contains 'model', 'encoder', 'metrics', 'model_path'.
    """
    logger.info("=" * 60)
    logger.info(f"Starting training pipeline: {run_name}")
    logger.info("=" * 60)

    settings = load_settings()

    # 1. Load data
    train_df, test_df = load_processed_data()

    # 2. Encode features
    encoder = EncoderPipeline()
    train_encoded = encoder.fit_transform(train_df, include_target=True)
    test_encoded = encoder.transform(test_df, include_target=True)

    target_col = settings["data"]["target_column"]
    feature_cols = encoder.get_feature_names()

    X_train = train_encoded[feature_cols]
    y_train = train_encoded[target_col]
    X_test = test_encoded[feature_cols]
    y_test = test_encoded[target_col]

    logger.info(f"Features: {len(feature_cols)}, Train: {len(X_train)}, Test: {len(X_test)}")

    # 3. Train model
    model_params = settings["model"]["params"]
    start_time = time.time()
    model = train_model(X_train, y_train, X_test, y_test, params=model_params)
    train_duration = time.time() - start_time

    # 4. Evaluate
    y_pred = model.predict(X_test)
    metrics = compute_metrics(y_test.values, y_pred, prefix="test_")
    detailed = compute_detailed_report(y_test.values, y_pred)
    metrics["train_duration_seconds"] = train_duration

    # 5. Save model and encoder
    model_path = save_model(model, version="v1")
    encoder_path = encoder.save()

    # 6. MLflow logging
    if use_mlflow:
        try:
            mlflow_uri = settings["mlflow"]["tracking_uri"]
            mlflow.set_tracking_uri(mlflow_uri)

            if experiment_name is None:
                experiment_name = settings["mlflow"]["experiment_name"]
            mlflow.set_experiment(experiment_name)

            with mlflow.start_run(run_name=run_name) as run:
                # Log parameters
                mlflow.log_params(model_params)
                mlflow.log_param("n_train_samples", len(X_train))
                mlflow.log_param("n_test_samples", len(X_test))
                mlflow.log_param("n_features", len(feature_cols))
                mlflow.log_param("model_version", "v1")

                # Log metrics
                for key, val in metrics.items():
                    mlflow.log_metric(key, val)

                # Log model artifact
                mlflow.xgboost.log_model(model, "model")

                # Log encoder as artifact
                mlflow.log_artifact(str(encoder_path))

                logger.info(f"MLflow run ID: {run.info.run_id}")
                logger.info(f"MLflow experiment: {experiment_name}")

        except Exception as e:
            logger.warning(
                f"MLflow logging failed (server may not be running): {e}. "
                "Training results saved locally."
            )
    else:
        logger.info("MLflow logging disabled for this run")

    result = {
        "model": model,
        "encoder": encoder,
        "metrics": metrics,
        "model_path": model_path,
        "encoder_path": encoder_path,
        "detailed_report": detailed,
    }

    logger.info("=" * 60)
    logger.info("Training pipeline complete!")
    logger.info(f"  Model: {model_path}")
    logger.info(f"  Test Accuracy: {metrics['test_accuracy']:.4f}")
    logger.info(f"  Test F1: {metrics['test_f1']:.4f}")
    logger.info("=" * 60)

    return result


# ── CLI entry point ─────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train XGBoost model")
    parser.add_argument(
        "--no-mlflow",
        action="store_true",
        help="Disable MLflow logging (useful if server is not running)",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default="baseline_training",
        help="Name for the MLflow run",
    )
    args = parser.parse_args()

    result = run_training_pipeline(
        use_mlflow=not args.no_mlflow,
        run_name=args.run_name,
    )
