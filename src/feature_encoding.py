"""
Feature encoding module for the Drift-Aware MLOps Pipeline.

Uses OrdinalEncoder for categorical features and LabelEncoder for the
binary target column. Avoids one-hot encoding to keep dimensionality
manageable for drift detection.
"""

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder

from src.utils import load_settings, get_path, ensure_dir, setup_logging

logger = setup_logging("feature_encoding", log_file="feature_encoding.log")


class EncoderPipeline:
    """
    Encode categorical features (OrdinalEncoder) and binary target (LabelEncoder).

    Attributes
    ----------
    feature_encoder : OrdinalEncoder
        Fitted encoder for categorical columns.
    label_encoder : LabelEncoder
        Fitted encoder for the target column.
    categorical_features : list[str]
        Names of categorical feature columns.
    numerical_features : list[str]
        Names of numerical feature columns.
    target_column : str
        Name of the target column.
    is_fitted : bool
        Whether the encoders have been fitted.
    """

    def __init__(self):
        settings = load_settings()
        self.categorical_features = settings["data"]["categorical_features"]
        self.numerical_features = settings["data"]["numerical_features"]
        self.target_column = settings["data"]["target_column"]

        self.feature_encoder = OrdinalEncoder(
            handle_unknown="use_encoded_value",
            unknown_value=-1,
        )
        self.label_encoder = LabelEncoder()
        self.is_fitted = False

    def fit(self, df: pd.DataFrame) -> "EncoderPipeline":
        """
        Fit encoders on the training data.

        Parameters
        ----------
        df : pd.DataFrame
            Training dataframe (must include categorical features and target).

        Returns
        -------
        EncoderPipeline
            self, for chaining.
        """
        logger.info("Fitting encoders on training data...")

        # Fit categorical encoder
        cat_data = df[self.categorical_features].astype(str)
        self.feature_encoder.fit(cat_data)
        logger.info(
            f"  OrdinalEncoder fitted on {len(self.categorical_features)} "
            f"categorical features"
        )

        # Fit label encoder on target
        self.label_encoder.fit(df[self.target_column].astype(str))
        logger.info(
            f"  LabelEncoder fitted: classes = {list(self.label_encoder.classes_)}"
        )

        self.is_fitted = True
        return self

    def transform(
        self, df: pd.DataFrame, include_target: bool = True
    ) -> pd.DataFrame:
        """
        Transform a dataframe using fitted encoders.

        Parameters
        ----------
        df : pd.DataFrame
            Dataframe to transform.
        include_target : bool
            Whether to encode the target column. Set False for inference.

        Returns
        -------
        pd.DataFrame
            Fully numeric dataframe.
        """
        if not self.is_fitted:
            raise RuntimeError("Encoders not fitted. Call fit() first.")

        result = df.copy()

        # Encode categorical features
        cat_data = result[self.categorical_features].astype(str)
        encoded = self.feature_encoder.transform(cat_data)
        result[self.categorical_features] = encoded

        # Encode target
        if include_target and self.target_column in result.columns:
            result[self.target_column] = self.label_encoder.transform(
                result[self.target_column].astype(str)
            )

        # Ensure all numeric
        for col in self.numerical_features:
            if col in result.columns:
                result[col] = pd.to_numeric(result[col], errors="coerce")

        logger.info(f"Transformed {len(result)} rows to numeric format")
        return result

    def fit_transform(
        self, df: pd.DataFrame, include_target: bool = True
    ) -> pd.DataFrame:
        """Fit and transform in one step."""
        return self.fit(df).transform(df, include_target=include_target)

    def inverse_transform_target(self, encoded: np.ndarray) -> np.ndarray:
        """Convert encoded target values back to original labels."""
        return self.label_encoder.inverse_transform(encoded)

    def save(self, filepath: str | Path | None = None) -> Path:
        """
        Save fitted encoders to disk.

        Parameters
        ----------
        filepath : str or Path, optional
            Where to save. Defaults to models/encoder_pipeline.pkl.

        Returns
        -------
        Path
            Path to saved file.
        """
        if filepath is None:
            models_dir = ensure_dir(get_path("models_dir"))
            filepath = models_dir / "encoder_pipeline.pkl"
        else:
            filepath = Path(filepath)
            filepath.parent.mkdir(parents=True, exist_ok=True)

        with open(filepath, "wb") as f:
            pickle.dump(
                {
                    "feature_encoder": self.feature_encoder,
                    "label_encoder": self.label_encoder,
                    "categorical_features": self.categorical_features,
                    "numerical_features": self.numerical_features,
                    "target_column": self.target_column,
                },
                f,
            )

        logger.info(f"Saved encoder pipeline to {filepath}")
        return filepath

    @classmethod
    def load(cls, filepath: str | Path | None = None) -> "EncoderPipeline":
        """
        Load a previously saved encoder pipeline.

        Parameters
        ----------
        filepath : str or Path, optional
            Path to the pickle file. Defaults to models/encoder_pipeline.pkl.

        Returns
        -------
        EncoderPipeline
            Fitted encoder pipeline.
        """
        if filepath is None:
            from src.utils import PROJECT_ROOT
            filepath = PROJECT_ROOT / "models" / "encoder_pipeline.pkl"

        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"Encoder file not found: {filepath}")

        with open(filepath, "rb") as f:
            data = pickle.load(f)

        instance = cls.__new__(cls)
        instance.feature_encoder = data["feature_encoder"]
        instance.label_encoder = data["label_encoder"]
        instance.categorical_features = data["categorical_features"]
        instance.numerical_features = data["numerical_features"]
        instance.target_column = data["target_column"]
        instance.is_fitted = True

        logger.info(f"Loaded encoder pipeline from {filepath}")
        return instance

    def get_feature_names(self) -> list[str]:
        """Return ordered list of all feature names (numerical + categorical)."""
        return self.numerical_features + self.categorical_features
