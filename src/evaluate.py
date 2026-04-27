"""
Evaluation module for the Drift-Aware MLOps Pipeline.

Computes classification metrics for model evaluation across batches.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    classification_report,
)

from src.utils import setup_logging

logger = setup_logging("evaluate", log_file="evaluate.log")


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    prefix: str = "",
) -> dict:
    """
    Compute classification metrics.

    Parameters
    ----------
    y_true : array-like
        Ground truth labels.
    y_pred : array-like
        Predicted labels.
    prefix : str, optional
        Prefix for metric keys (e.g. 'batch_1_').

    Returns
    -------
    dict
        Dictionary of metric_name -> value.
    """
    metrics = {
        f"{prefix}accuracy": accuracy_score(y_true, y_pred),
        f"{prefix}f1": f1_score(y_true, y_pred, average="binary", zero_division=0),
        f"{prefix}precision": precision_score(y_true, y_pred, average="binary", zero_division=0),
        f"{prefix}recall": recall_score(y_true, y_pred, average="binary", zero_division=0),
    }

    logger.info(
        f"Metrics [{prefix.rstrip('_') or 'eval'}]: "
        f"acc={metrics[f'{prefix}accuracy']:.4f}, "
        f"f1={metrics[f'{prefix}f1']:.4f}, "
        f"prec={metrics[f'{prefix}precision']:.4f}, "
        f"rec={metrics[f'{prefix}recall']:.4f}"
    )

    return metrics


def compute_detailed_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    target_names: list[str] | None = None,
) -> dict:
    """
    Compute a detailed classification report + confusion matrix.

    Parameters
    ----------
    y_true : array-like
        Ground truth labels.
    y_pred : array-like
        Predicted labels.
    target_names : list[str], optional
        Class label names for the report.

    Returns
    -------
    dict
        Contains 'report' (str), 'confusion_matrix' (np.ndarray),
        and 'metrics' (dict).
    """
    if target_names is None:
        target_names = ["<=50K", ">50K"]

    report = classification_report(
        y_true, y_pred, target_names=target_names, zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred)
    metrics = compute_metrics(y_true, y_pred)

    logger.info(f"\n{report}")
    logger.info(f"Confusion matrix:\n{cm}")

    return {
        "report": report,
        "confusion_matrix": cm,
        "metrics": metrics,
    }


def evaluate_on_batches(
    model,
    batches: dict[str, pd.DataFrame],
    feature_columns: list[str],
    target_column: str = "income",
) -> dict[str, dict]:
    """
    Evaluate model performance across multiple data batches.

    Parameters
    ----------
    model : trained model
        Must have a .predict() method.
    batches : dict[str, pd.DataFrame]
        Mapping of batch_name -> encoded dataframe.
    feature_columns : list[str]
        Feature column names.
    target_column : str
        Target column name.

    Returns
    -------
    dict[str, dict]
        Mapping of batch_name -> metrics dict.
    """
    all_metrics = {}

    for batch_name, batch_df in batches.items():
        X = batch_df[feature_columns]
        y = batch_df[target_column]
        y_pred = model.predict(X)

        metrics = compute_metrics(y, y_pred, prefix=f"{batch_name}_")
        all_metrics[batch_name] = metrics

    return all_metrics


def metrics_to_dataframe(batch_metrics: dict[str, dict]) -> pd.DataFrame:
    """
    Convert batch metrics dict to a tidy DataFrame for visualization.

    Parameters
    ----------
    batch_metrics : dict[str, dict]
        Output of evaluate_on_batches().

    Returns
    -------
    pd.DataFrame
        Columns: batch, accuracy, f1, precision, recall.
    """
    rows = []
    for batch_name, metrics in batch_metrics.items():
        # Strip batch prefix from metric keys
        clean = {}
        for key, val in metrics.items():
            clean_key = key.replace(f"{batch_name}_", "")
            clean[clean_key] = val
        clean["batch"] = batch_name
        rows.append(clean)

    df = pd.DataFrame(rows)
    # Reorder columns
    cols = ["batch", "accuracy", "f1", "precision", "recall"]
    return df[[c for c in cols if c in df.columns]]
