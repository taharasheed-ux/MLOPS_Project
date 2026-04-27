"""
Data processing module for the Drift-Aware MLOps Pipeline.

Handles loading, cleaning, and splitting the UCI Adult Income dataset.
The user must download the dataset and place it in data/raw/.
"""

import pandas as pd
import numpy as np
from pathlib import Path

from src.utils import load_settings, get_path, ensure_dir, setup_logging

logger = setup_logging("data_processing", log_file="data_processing.log")


# ── Column metadata ─────────────────────────────────────────────────
# UCI Adult dataset headers (the raw CSV may or may not have headers)
COLUMN_NAMES = [
    "age", "workclass", "fnlwgt", "education", "education-num",
    "marital-status", "occupation", "relationship", "race", "sex",
    "capital-gain", "capital-loss", "hours-per-week", "native-country",
    "income",
]


def load_data(filepath: str | Path | None = None) -> pd.DataFrame:
    """
    Load the Adult Income dataset from the original UCI files.

    Combines adult.data (training) and adult.test (test) into one dataframe.
    Both files are headerless CSVs. adult.test has a metadata first line
    that is automatically skipped.

    Parameters
    ----------
    filepath : str or Path, optional
        Path to a single CSV file. If None, loads and combines
        adult.data + adult.test from data/raw/.

    Returns
    -------
    pd.DataFrame
        Raw dataframe with consistent column names.
    """
    settings = load_settings()

    if filepath is not None:
        # Single file mode
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"Dataset not found at {filepath}")
        logger.info(f"Loading data from {filepath}")
        df = _load_single_file(filepath)
    else:
        # Combine adult.data + adult.test
        raw_dir = get_path("raw_data_dir", settings)
        train_file = raw_dir / settings["data"]["train_filename"]
        test_file = raw_dir / settings["data"]["test_filename"]

        if not train_file.exists():
            raise FileNotFoundError(
                f"Training data not found at {train_file}. "
                "Place adult.data in data/raw/"
            )

        logger.info(f"Loading training data from {train_file}")
        train_df = _load_single_file(train_file)
        logger.info(f"  Loaded {len(train_df)} rows from adult.data")

        if test_file.exists():
            logger.info(f"Loading test data from {test_file}")
            test_df = _load_single_file(test_file, skip_first_line=True)
            logger.info(f"  Loaded {len(test_df)} rows from adult.test")

            df = pd.concat([train_df, test_df], ignore_index=True)
            logger.info(f"Combined dataset: {len(df)} rows")
        else:
            logger.warning("adult.test not found, using adult.data only")
            df = train_df

    logger.info(f"Loaded {len(df)} rows, {len(df.columns)} columns")
    return df


def _load_single_file(
    filepath: Path, skip_first_line: bool = False
) -> pd.DataFrame:
    """
    Load a single UCI Adult data file (headerless CSV).

    Parameters
    ----------
    filepath : Path
        Path to the file.
    skip_first_line : bool
        If True, skip the first line (adult.test has a metadata line).

    Returns
    -------
    pd.DataFrame
    """
    skiprows = 1 if skip_first_line else 0

    df = pd.read_csv(
        filepath,
        names=COLUMN_NAMES,
        skipinitialspace=True,
        skiprows=skiprows,
        na_values=["?"],
    )

    # Drop any completely empty rows
    df.dropna(how="all", inplace=True)

    # Standardize column names
    df.columns = [c.strip().lower() for c in df.columns]

    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the Adult dataset.

    Operations:
    - Strip whitespace from string columns
    - Replace '?' with NaN
    - Impute missing values (mode for categorical, median for numerical)
    - Drop exact duplicate rows
    - Standardize target column values

    Parameters
    ----------
    df : pd.DataFrame
        Raw dataframe.

    Returns
    -------
    pd.DataFrame
        Cleaned dataframe with no missing values.
    """
    settings = load_settings()
    numerical = settings["data"]["numerical_features"]
    categorical = settings["data"]["categorical_features"]

    logger.info("Cleaning data...")
    df = df.copy()

    # Strip whitespace from object columns
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].str.strip()

    # Replace '?' with NaN
    df.replace("?", np.nan, inplace=True)

    # Remove trailing periods from income (test set artifact)
    if "income" in df.columns:
        df["income"] = df["income"].str.rstrip(".")

    # Impute missing values
    for col in categorical:
        if col in df.columns and df[col].isna().any():
            mode_val = df[col].mode()[0]
            df[col] = df[col].fillna(mode_val)
            logger.info(f"  Imputed {col} with mode: {mode_val}")

    for col in numerical:
        if col in df.columns and df[col].isna().any():
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)
            logger.info(f"  Imputed {col} with median: {median_val}")

    # Drop duplicates
    before = len(df)
    df.drop_duplicates(inplace=True)
    dropped = before - len(df)
    if dropped > 0:
        logger.info(f"  Dropped {dropped} duplicate rows")

    # Ensure correct dtypes for numerical columns
    for col in numerical:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    logger.info(f"Cleaning complete: {len(df)} rows, {df.isna().sum().sum()} NaN remaining")
    return df.reset_index(drop=True)


def split_data(
    df: pd.DataFrame,
    test_size: float | None = None,
    random_seed: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Perform a stratified train/test split.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned dataframe.
    test_size : float, optional
        Fraction for test set. Defaults to settings.yaml value.
    random_seed : int, optional
        Random seed. Defaults to settings.yaml value.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        (train_df, test_df) with reset indices.
    """
    from sklearn.model_selection import train_test_split

    settings = load_settings()
    target_col = settings["data"]["target_column"]

    if test_size is None:
        test_size = settings["data"]["test_size"]
    if random_seed is None:
        random_seed = settings["project"]["random_seed"]

    logger.info(f"Splitting data: test_size={test_size}, seed={random_seed}")

    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_seed,
        stratify=df[target_col],
    )

    logger.info(f"Train: {len(train_df)} rows | Test: {len(test_df)} rows")
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)


def save_processed_data(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> tuple[Path, Path]:
    """
    Save processed train/test splits to data/processed/.

    Returns
    -------
    tuple[Path, Path]
        Paths to saved train and test CSV files.
    """
    out_dir = ensure_dir(get_path("processed_data_dir"))
    train_path = out_dir / "train.csv"
    test_path = out_dir / "test.csv"

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    logger.info(f"Saved processed data: {train_path}, {test_path}")
    return train_path, test_path


def run_data_pipeline() -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Execute the full data pipeline: load → clean → split → save.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        (train_df, test_df)
    """
    logger.info("=" * 60)
    logger.info("Starting data processing pipeline")
    logger.info("=" * 60)

    # Ensure output directories exist
    ensure_dir(get_path("raw_data_dir"))
    ensure_dir(get_path("processed_data_dir"))

    # Load and clean
    raw_df = load_data()
    clean_df = clean_data(raw_df)

    # Split
    train_df, test_df = split_data(clean_df)

    # Save
    save_processed_data(train_df, test_df)

    logger.info("Data pipeline complete!")
    return train_df, test_df


# ── CLI entry point ─────────────────────────────────────────────────
if __name__ == "__main__":
    run_data_pipeline()
