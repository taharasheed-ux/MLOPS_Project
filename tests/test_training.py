"""
Tests for the training and evaluation pipeline.

Run with: python -m pytest tests/test_training.py -v
"""

import pytest
import pandas as pd
import numpy as np


@pytest.fixture(scope="module")
def train_df():
    """Load processed training data."""
    from src.utils import get_path
    from src.utils import get_processed_data_paths
    train_path, _ = get_processed_data_paths()
    return pd.read_csv(train_path)


@pytest.fixture(scope="module")
def test_df():
    """Load processed test data."""
    from src.utils import get_path
    from src.utils import get_processed_data_paths
    _, test_path = get_processed_data_paths()
    return pd.read_csv(test_path)


@pytest.fixture(scope="module")
def encoder(train_df):
    """Fit encoder on training data."""
    from src.feature_encoding import EncoderPipeline
    enc = EncoderPipeline()
    enc.fit(train_df)
    return enc


@pytest.fixture(scope="module")
def encoded_data(encoder, train_df, test_df):
    """Encode train and test data."""
    from src.utils import load_settings
    settings = load_settings()
    target_col = settings["data"]["target_column"]
    feature_cols = encoder.get_feature_names()

    train_enc = encoder.transform(train_df, include_target=True)
    test_enc = encoder.transform(test_df, include_target=True)

    return {
        "X_train": train_enc[feature_cols],
        "y_train": train_enc[target_col],
        "X_test": test_enc[feature_cols],
        "y_test": test_enc[target_col],
        "feature_cols": feature_cols,
        "target_col": target_col,
    }


@pytest.fixture(scope="module")
def trained_model(encoded_data):
    """Train a model for testing."""
    from src.train import train_model
    return train_model(
        encoded_data["X_train"],
        encoded_data["y_train"],
        encoded_data["X_test"],
        encoded_data["y_test"],
    )


# ── Training Tests ──────────────────────────────────────────────────

class TestTraining:
    """Test model training."""

    def test_model_trains(self, trained_model):
        """Model should be a fitted XGBClassifier."""
        import xgboost as xgb
        assert isinstance(trained_model, xgb.XGBClassifier)

    def test_model_predicts(self, trained_model, encoded_data):
        """Model should produce predictions."""
        preds = trained_model.predict(encoded_data["X_test"])
        assert len(preds) == len(encoded_data["y_test"])
        assert set(np.unique(preds)).issubset({0, 1})

    def test_model_accuracy_reasonable(self, trained_model, encoded_data):
        """Baseline accuracy should be above 80% on Adult dataset."""
        from src.evaluate import compute_metrics
        preds = trained_model.predict(encoded_data["X_test"])
        metrics = compute_metrics(
            encoded_data["y_test"].values, preds, prefix="test_"
        )
        assert metrics["test_accuracy"] > 0.80, (
            f"Accuracy {metrics['test_accuracy']:.4f} is below 80%"
        )

    def test_model_f1_reasonable(self, trained_model, encoded_data):
        """F1 should be above 60% for the positive class."""
        from src.evaluate import compute_metrics
        preds = trained_model.predict(encoded_data["X_test"])
        metrics = compute_metrics(
            encoded_data["y_test"].values, preds, prefix="test_"
        )
        assert metrics["test_f1"] > 0.60, (
            f"F1 {metrics['test_f1']:.4f} is below 60%"
        )


# ── Evaluation Tests ────────────────────────────────────────────────

class TestEvaluation:
    """Test evaluation utilities."""

    def test_compute_metrics_keys(self):
        from src.evaluate import compute_metrics
        y_true = np.array([0, 1, 1, 0, 1])
        y_pred = np.array([0, 1, 0, 0, 1])
        metrics = compute_metrics(y_true, y_pred, prefix="test_")
        assert "test_accuracy" in metrics
        assert "test_f1" in metrics
        assert "test_precision" in metrics
        assert "test_recall" in metrics

    def test_metrics_values_range(self):
        from src.evaluate import compute_metrics
        y_true = np.array([0, 1, 1, 0, 1, 0, 0, 1])
        y_pred = np.array([0, 1, 0, 0, 1, 1, 0, 1])
        metrics = compute_metrics(y_true, y_pred)
        for key, val in metrics.items():
            assert 0.0 <= val <= 1.0, f"{key}={val} out of range"

    def test_metrics_to_dataframe(self):
        from src.evaluate import metrics_to_dataframe
        batch_metrics = {
            "batch_1": {"batch_1_accuracy": 0.9, "batch_1_f1": 0.8,
                        "batch_1_precision": 0.85, "batch_1_recall": 0.75},
            "batch_2": {"batch_2_accuracy": 0.85, "batch_2_f1": 0.7,
                        "batch_2_precision": 0.8, "batch_2_recall": 0.65},
        }
        df = metrics_to_dataframe(batch_metrics)
        assert len(df) == 2
        assert "batch" in df.columns
        assert "accuracy" in df.columns


# ── Model Save/Load Tests ───────────────────────────────────────────

class TestModelPersistence:
    """Test model save and load."""

    def test_save_load_roundtrip(self, trained_model, encoded_data, tmp_path):
        from src.train import save_model, load_model
        path = save_model(trained_model, filepath=tmp_path / "test_model.pkl")
        loaded = load_model(path)

        # Should produce identical predictions
        preds_original = trained_model.predict(encoded_data["X_test"])
        preds_loaded = loaded.predict(encoded_data["X_test"])
        np.testing.assert_array_equal(preds_original, preds_loaded)


# ── Full Pipeline Test ──────────────────────────────────────────────

class TestFullPipeline:
    """Test the end-to-end training pipeline."""

    def test_pipeline_no_mlflow(self):
        """Full pipeline should work without MLflow server."""
        from src.train import run_training_pipeline
        result = run_training_pipeline(use_mlflow=False, run_name="test_run")

        assert "model" in result
        assert "encoder" in result
        assert "metrics" in result
        assert "model_path" in result
        assert result["metrics"]["test_accuracy"] > 0.80
