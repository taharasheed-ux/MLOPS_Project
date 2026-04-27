"""
Tests for the drift detection module.
"""

import pytest
import pandas as pd
import numpy as np

from src.drift_detection import DriftDetector

@pytest.fixture
def reference_data():
    np.random.seed(42)
    return pd.DataFrame({
        "num1": np.random.normal(0, 1, 1000),
        "num2": np.random.normal(5, 2, 1000),
        "cat1": np.random.choice(["A", "B", "C"], 1000, p=[0.5, 0.3, 0.2]),
        "cat2": np.random.choice(["X", "Y"], 1000, p=[0.8, 0.2])
    })

@pytest.fixture
def current_data_no_drift(reference_data):
    # Same distribution
    np.random.seed(43)
    return pd.DataFrame({
        "num1": np.random.normal(0, 1, 500),
        "num2": np.random.normal(5, 2, 500),
        "cat1": np.random.choice(["A", "B", "C"], 500, p=[0.5, 0.3, 0.2]),
        "cat2": np.random.choice(["X", "Y"], 500, p=[0.8, 0.2])
    })

@pytest.fixture
def current_data_with_drift():
    # Drifted distribution
    np.random.seed(44)
    return pd.DataFrame({
        "num1": np.random.normal(2, 1.5, 500),  # Shifted and scaled
        "num2": np.random.normal(5, 2, 500),    # Unchanged
        "cat1": np.random.choice(["A", "B", "C"], 500, p=[0.2, 0.3, 0.5]), # Resampled
        "cat2": np.random.choice(["X", "Y"], 500, p=[0.8, 0.2])            # Unchanged
    })

@pytest.fixture
def config():
    return {
        "drift_detection": {
            "alpha": 0.05,
            "correction_method": "bonferroni",
            "min_effect_size": {
                "numerical": 0.1,
                "categorical": 0.1
            }
        },
        "severity": {
            "threshold": 0.3,
            "weighting": "uniform"
        }
    }


def test_drift_detection_no_drift(reference_data, current_data_no_drift, config):
    detector = DriftDetector(
        reference_data,
        numerical_features=["num1", "num2"],
        categorical_features=["cat1", "cat2"],
        config=config
    )
    result = detector.detect(current_data_no_drift, "batch_no_drift")
    
    assert result.n_features_tested == 4
    assert result.n_features_drifted == 0
    assert result.drift_detected is False
    assert result.severity_score < config["severity"]["threshold"]


def test_drift_detection_with_drift(reference_data, current_data_with_drift, config):
    detector = DriftDetector(
        reference_data,
        numerical_features=["num1", "num2"],
        categorical_features=["cat1", "cat2"],
        config=config
    )
    result = detector.detect(current_data_with_drift, "batch_with_drift")
    
    assert result.n_features_tested == 4
    assert result.n_features_drifted == 2
    assert "num1" in result.drifted_features
    assert "cat1" in result.drifted_features
    assert "num2" not in result.drifted_features
    assert "cat2" not in result.drifted_features

    # Severity should be > 0 and potentially trigger alert depending on threshold
    assert result.severity_score > 0
    # In this case 2/4 features drifted, uniform weighting severity = 0.5 * mean_effect
    # Check that magnitude is not "none" for drifted features
    for fr in result.feature_results:
        if fr.feature in ["num1", "cat1"]:
            assert fr.is_drifted
            assert fr.drift_magnitude != "none"
        else:
            assert not fr.is_drifted
            assert fr.drift_magnitude == "none"

def test_benjamini_hochberg(reference_data, config):
    # Test BH correction explicitly
    config["drift_detection"]["correction_method"] = "benjamini-hochberg"
    detector = DriftDetector(
        reference_data,
        numerical_features=["num1", "num2"],
        categorical_features=["cat1", "cat2"],
        config=config
    )
    
    # Create fake p-values: some very low (drift), one marginal, one high
    p_vals = np.array([0.001, 0.01, 0.04, 0.5])
    
    # Expected ranking: 0.001 (1), 0.01 (2), 0.04 (3), 0.5 (4)
    # BH Adjusted:
    # rank 4: min(1.0, 0.5 * 4/4) = 0.5
    # rank 3: min(0.5, 0.04 * 4/3) = 0.0533
    # rank 2: min(0.0533, 0.01 * 4/2) = 0.02
    # rank 1: min(0.02, 0.001 * 4/1) = 0.004
    
    corrected = detector._benjamini_hochberg(p_vals)
    
    expected = np.array([0.004, 0.02, 0.0533, 0.5])
    np.testing.assert_allclose(corrected, expected, rtol=1e-3)
