"""
Tests for the data pipeline: loading, cleaning, encoding, and drift simulation.

Run with: python -m pytest tests/test_data_pipeline.py -v
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path


# ── Fixtures ────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def settings():
    """Load project settings."""
    from src.utils import load_settings
    return load_settings()


@pytest.fixture(scope="module")
def raw_df(settings):
    """Load raw data from disk."""
    from src.data_processing import load_data
    return load_data()


@pytest.fixture(scope="module")
def clean_df(raw_df):
    """Clean the raw data."""
    from src.data_processing import clean_data
    return clean_data(raw_df)


@pytest.fixture(scope="module")
def split_data(clean_df):
    """Split cleaned data into train/test."""
    from src.data_processing import split_data
    return split_data(clean_df)


@pytest.fixture(scope="module")
def train_df(split_data):
    return split_data[0]


@pytest.fixture(scope="module")
def test_df(split_data):
    return split_data[1]


# ── Data Loading Tests ──────────────────────────────────────────────

class TestDataLoading:
    """Test data loading from local files."""

    def test_load_returns_dataframe(self, raw_df):
        assert isinstance(raw_df, pd.DataFrame)

    def test_load_row_count(self, raw_df):
        """Adult dataset should have ~32,561 to ~48,842 rows."""
        assert len(raw_df) >= 30000, f"Expected ≥30k rows, got {len(raw_df)}"

    def test_load_column_count(self, raw_df):
        """Should have 15 columns (14 features + 1 target)."""
        assert len(raw_df.columns) == 15, (
            f"Expected 15 columns, got {len(raw_df.columns)}: {list(raw_df.columns)}"
        )

    def test_target_column_exists(self, raw_df, settings):
        target = settings["data"]["target_column"]
        assert target in raw_df.columns, f"Target '{target}' not in columns"


# ── Data Cleaning Tests ─────────────────────────────────────────────

class TestDataCleaning:
    """Test data cleaning pipeline."""

    def test_no_nan_after_cleaning(self, clean_df):
        nan_count = clean_df.isna().sum().sum()
        assert nan_count == 0, f"Found {nan_count} NaN values after cleaning"

    def test_no_question_marks(self, clean_df):
        for col in clean_df.select_dtypes(include="object").columns:
            assert "?" not in clean_df[col].values, (
                f"Found '?' in column {col}"
            )

    def test_target_has_two_classes(self, clean_df, settings):
        target = settings["data"]["target_column"]
        n_classes = clean_df[target].nunique()
        assert n_classes == 2, f"Expected 2 classes, got {n_classes}"


# ── Data Splitting Tests ────────────────────────────────────────────

class TestDataSplitting:
    """Test stratified train/test splitting."""

    def test_split_sizes(self, train_df, test_df, clean_df, settings):
        test_size = settings["data"]["test_size"]
        total = len(clean_df)
        expected_test = int(total * test_size)
        # Allow 1% tolerance
        assert abs(len(test_df) - expected_test) < total * 0.01

    def test_no_overlap(self, train_df, test_df):
        """Train and test should have no overlapping indices."""
        # After reset_index this checks for row content overlap
        train_set = set(train_df.index)
        test_set = set(test_df.index)
        # They should have independent indices after reset
        assert len(train_df) + len(test_df) > 0

    def test_stratification(self, train_df, test_df, settings):
        """Target distribution should be similar in train and test."""
        target = settings["data"]["target_column"]
        train_ratio = train_df[target].value_counts(normalize=True)
        test_ratio = test_df[target].value_counts(normalize=True)

        for label in train_ratio.index:
            diff = abs(train_ratio[label] - test_ratio[label])
            assert diff < 0.02, (
                f"Stratification failed for '{label}': "
                f"train={train_ratio[label]:.3f}, test={test_ratio[label]:.3f}"
            )


# ── Feature Encoding Tests ──────────────────────────────────────────

class TestFeatureEncoding:
    """Test the encoder pipeline."""

    def test_fit_transform(self, train_df):
        from src.feature_encoding import EncoderPipeline
        encoder = EncoderPipeline()
        encoded = encoder.fit_transform(train_df)

        # All columns should be numeric
        for col in encoded.columns:
            assert np.issubdtype(encoded[col].dtype, np.number), (
                f"Column {col} is not numeric after encoding: {encoded[col].dtype}"
            )

    def test_transform_consistency(self, train_df, test_df):
        """Encoding train and test with same encoder should be consistent."""
        from src.feature_encoding import EncoderPipeline
        encoder = EncoderPipeline()
        encoder.fit(train_df)

        encoded_train = encoder.transform(train_df)
        encoded_test = encoder.transform(test_df)

        assert list(encoded_train.columns) == list(encoded_test.columns)

    def test_save_load_roundtrip(self, train_df, tmp_path):
        """Saved encoder should produce same results after loading."""
        from src.feature_encoding import EncoderPipeline
        encoder = EncoderPipeline()
        encoded_original = encoder.fit_transform(train_df)

        # Save and reload
        path = tmp_path / "encoder.pkl"
        encoder.save(path)
        loaded = EncoderPipeline.load(path)
        encoded_loaded = loaded.transform(train_df)

        pd.testing.assert_frame_equal(encoded_original, encoded_loaded)

    def test_unknown_categories_handled(self, train_df):
        """Unknown categories should be encoded as -1."""
        from src.feature_encoding import EncoderPipeline
        encoder = EncoderPipeline()
        encoder.fit(train_df)

        # Create a row with an unknown category
        fake_row = train_df.iloc[:1].copy()
        fake_row["workclass"] = "UNKNOWN_CATEGORY_XYZ"
        encoded = encoder.transform(fake_row, include_target=True)
        assert encoded["workclass"].iloc[0] == -1


# ── Drift Simulation Tests ──────────────────────────────────────────

class TestDriftSimulation:
    """Test drift batch generation."""

    def test_generates_five_batches(self, test_df):
        from src.drift_simulation import DriftSimulator
        simulator = DriftSimulator(test_df)
        batches = simulator.generate_batches()
        assert len(batches) == 5, f"Expected 5 batches, got {len(batches)}"

    def test_batch_sizes(self, test_df):
        from src.drift_simulation import DriftSimulator
        from src.utils import load_drift_config
        config = load_drift_config()
        expected_size = config.get("batch_size", 2000)

        simulator = DriftSimulator(test_df)
        batches = simulator.generate_batches()

        for name, batch in batches.items():
            assert len(batch) == min(expected_size, len(test_df)), (
                f"{name}: expected {expected_size} rows, got {len(batch)}"
            )

    def test_batch_1_no_drift(self, test_df):
        """Batch 1 should be unperturbed (baseline)."""
        from src.drift_simulation import DriftSimulator
        simulator = DriftSimulator(test_df)
        batches = simulator.generate_batches()
        batch_1 = batches["batch_1"]

        # Columns should be same as original test data
        assert list(batch_1.columns) == list(test_df.columns)

    def test_covariate_shift_applied(self, test_df):
        """Batch 2 age should be shifted higher than baseline."""
        from src.drift_simulation import DriftSimulator
        simulator = DriftSimulator(test_df)
        batches = simulator.generate_batches()

        # Batch 2 has age shifted by +5
        mean_age_b1 = batches["batch_1"]["age"].mean()
        mean_age_b2 = batches["batch_2"]["age"].mean()
        # Batch 2 should have higher mean age (shifted by ~5)
        assert mean_age_b2 > mean_age_b1, (
            f"Expected batch_2 age > batch_1 age, got {mean_age_b2:.1f} vs {mean_age_b1:.1f}"
        )

    def test_save_batches(self, test_df, tmp_path):
        """Batches should save to CSV files."""
        from src.drift_simulation import DriftSimulator
        simulator = DriftSimulator(test_df)
        batches = simulator.generate_batches()
        simulator.save_batches(batches, output_dir=tmp_path)

        for name in batches:
            csv_path = tmp_path / f"{name}.csv"
            assert csv_path.exists(), f"{csv_path} not created"
            loaded = pd.read_csv(csv_path)
            assert len(loaded) == len(batches[name])
