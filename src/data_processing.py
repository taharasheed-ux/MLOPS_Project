"""
Data processing module for the Drift-Aware MLOps Pipeline.

Handles loading, cleaning, and splitting the UCI Adult Income dataset.
The user must download the dataset and place it in data/raw/.
"""

import pandas as pd
import numpy as np
import gc
from pathlib import Path

from src.utils import (
    load_settings,
    get_path,
    get_processed_data_paths,
    ensure_dir,
    setup_logging,
)

logger = setup_logging("data_processing", log_file="data_processing.log")


# ── Column metadata ─────────────────────────────────────────────────
# UCI Adult dataset headers (the raw CSV may or may not have headers)
COLUMN_NAMES = [
    "age", "workclass", "fnlwgt", "education", "education-num",
    "marital-status", "occupation", "relationship", "race", "sex",
    "capital-gain", "capital-loss", "hours-per-week", "native-country",
    "income",
]

ACS_TEMPORAL_YEAR_COLUMN = "DATA_YEAR"


def _sort_year_labels(years: list[str]) -> list[str]:
    """Sort year labels numerically when possible, otherwise lexicographically."""
    try:
        return sorted([str(year) for year in years], key=lambda year: int(year))
    except ValueError:
        return sorted([str(year) for year in years])


def _get_acs_years(acs_cfg: dict) -> list[str]:
    """Return the configured ACS survey years, falling back to the legacy single year key."""
    survey_years = acs_cfg.get("survey_years")
    if survey_years:
        return _sort_year_labels([str(year) for year in survey_years])
    return [str(acs_cfg.get("survey_year", "2018"))]


def _resolve_acs_temporal_split(acs_cfg: dict) -> tuple[bool, set[str], set[str]]:
    """
    Resolve whether temporal splitting is enabled and which years belong to the
    baseline-train vs evaluation partitions.
    """
    survey_years = _get_acs_years(acs_cfg)
    temporal_cfg = acs_cfg.get("temporal_split", {})
    enabled = temporal_cfg.get("enabled", False) and len(survey_years) > 1
    if not enabled:
        return False, set(), set()

    initial_train_years = temporal_cfg.get("initial_train_years", 1)
    initial_train_years = max(1, min(int(initial_train_years), len(survey_years) - 1))
    train_years = set(survey_years[:initial_train_years])
    eval_years = set(survey_years[initial_train_years:])
    return True, train_years, eval_years


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
    dataset_profile = settings["data"].get("dataset_profile", "adult").lower()

    if dataset_profile == "acs":
        return load_acs_data(settings)

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


def load_acs_data(settings: dict | None = None) -> pd.DataFrame:
    """
    Load ACS Income dataset using folktables.

    This fetches ACS PUMS data for configured states/year and returns
    a DataFrame aligned with the project's data pipeline contract.
    """
    if settings is None:
        settings = load_settings()

    try:
        from folktables import ACSDataSource, ACSIncome
    except ImportError as e:
        raise ImportError(
            "folktables is required for ACS dataset loading. "
            "Install dependencies from requirements.txt"
        ) from e

    acs_cfg = settings["data"].get("acs", {})
    survey_years = _get_acs_years(acs_cfg)
    horizon = acs_cfg.get("horizon", "1-Year")
    survey = acs_cfg.get("survey", "person")
    target_col = settings["data"]["target_column"]
    year_col = acs_cfg.get("temporal_year_column", ACS_TEMPORAL_YEAR_COLUMN)
    positive_label = settings["data"].get("positive_label", ">50K")
    negative_label = "<=50K" if positive_label != "<=50K" else "not_positive"

    raw_dir = get_path("raw_data_dir", settings)
    cache_dir = ensure_dir(raw_dir / "folktables_cache")
    snapshot_filename = acs_cfg.get("snapshot_filename", "acs_income_full.csv")
    snapshot_path = raw_dir / snapshot_filename

    use_snapshot = acs_cfg.get("use_local_snapshot_if_exists", True)
    if use_snapshot and snapshot_path.exists():
        logger.info(f"Loading ACS snapshot from {snapshot_path}")
        df_snapshot = pd.read_csv(snapshot_path)
        logger.info(f"Loaded ACS snapshot: {len(df_snapshot)} rows")
        return df_snapshot

    states = acs_cfg.get(
        "states",
        [
            "AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DE", "FL", "GA",
            "HI", "ID", "IL", "IN", "IA", "KS", "KY", "LA", "ME", "MD",
            "MA", "MI", "MN", "MS", "MO", "MT", "NE", "NV", "NH", "NJ",
            "NM", "NY", "NC", "ND", "OH", "OK", "OR", "PA", "RI", "SC",
            "SD", "TN", "TX", "UT", "VT", "VA", "WA", "WV", "WI", "WY",
            "PR",
        ],
    )
    states = [s.upper() for s in states]

    logger.info(
        "Fetching ACS via folktables "
        f"(years={survey_years}, horizon={horizon}, survey={survey}, states={len(states)})"
    )
    yearly_frames = []
    for survey_year in survey_years:
        source = ACSDataSource(
            survey_year=survey_year,
            horizon=horizon,
            survey=survey,
            root_dir=str(cache_dir),
        )
        acs_data = source.get_data(states=states, download=True)
        features, labels, _ = ACSIncome.df_to_pandas(acs_data)

        df_year = features.copy()
        # ACSIncome labels are binary {0, 1}; convert to pipeline-compatible strings
        df_year[target_col] = np.where(labels.astype(int) == 1, positive_label, negative_label)
        df_year.columns = [str(c).strip() for c in df_year.columns]
        df_year[year_col] = str(survey_year)
        yearly_frames.append(df_year)

    df = pd.concat(yearly_frames, ignore_index=True)

    if acs_cfg.get("save_snapshot", True):
        df.to_csv(snapshot_path, index=False)
        logger.info(f"Saved ACS snapshot to {snapshot_path}")

    logger.info(f"Loaded ACS dataset: {len(df)} rows, {len(df.columns)} columns")
    return df


def _load_single_acs_state(
    state: str,
    survey_year: str,
    source,
    target_col: str,
    year_col: str,
    positive_label: str,
    negative_label: str,
) -> pd.DataFrame:
    """Load one ACS state and map it into the project dataframe contract."""
    from folktables import ACSIncome

    acs_data = source.get_data(states=[state], download=True)
    features, labels, _ = ACSIncome.df_to_pandas(acs_data)

    df = features.copy()
    df[target_col] = np.where(labels.astype(int) == 1, positive_label, negative_label)
    df.columns = [str(c).strip() for c in df.columns]
    df[year_col] = str(survey_year)
    return df


def _append_df(df: pd.DataFrame, path: Path, write_header: bool) -> None:
    """Append a dataframe to CSV, writing the header only once."""
    df.to_csv(path, mode="a", header=write_header, index=False)


def _write_df_atomic(df: pd.DataFrame, path: Path) -> None:
    """Write a dataframe atomically to avoid partial shard files after crashes."""
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp_path, index=False)
    tmp_path.replace(path)


def _combine_csv_shards(shard_paths: list[Path], output_path: Path) -> None:
    """Combine per-state CSV shards into one final CSV with a single header."""
    if output_path.exists():
        output_path.unlink()

    header_written = False
    with open(output_path, "wb") as out_f:
        for shard_path in shard_paths:
            with open(shard_path, "rb") as in_f:
                if header_written:
                    next(in_f)
                out_f.write(in_f.read())
            header_written = True


def _run_acs_pipeline_memory_efficient(settings: dict) -> tuple[Path, Path]:
    """
    Build ACS processed splits state-by-state to avoid loading the full national
    dataset into memory at once.
    """
    try:
        from folktables import ACSDataSource
    except ImportError as e:
        raise ImportError(
            "folktables is required for ACS dataset loading. "
            "Install dependencies from requirements.txt"
        ) from e

    acs_cfg = settings["data"].get("acs", {})
    raw_dir = get_path("raw_data_dir", settings)
    ensure_dir(raw_dir)
    ensure_dir(get_path("processed_data_dir", settings))

    train_path, test_path = get_processed_data_paths(settings)
    processed_dir = get_path("processed_data_dir", settings)
    shard_dir = ensure_dir(processed_dir / "acs_shards")

    snapshot_filename = acs_cfg.get("snapshot_filename", "acs_income_full.csv")
    snapshot_path = raw_dir / snapshot_filename
    save_snapshot = acs_cfg.get("save_snapshot", True)
    snapshot_shard_dir = ensure_dir(raw_dir / "acs_snapshot_shards") if save_snapshot else None

    survey_years = _get_acs_years(acs_cfg)
    horizon = acs_cfg.get("horizon", "1-Year")
    survey = acs_cfg.get("survey", "person")
    states = [s.upper() for s in acs_cfg.get("states", [])]
    target_col = settings["data"]["target_column"]
    year_col = acs_cfg.get("temporal_year_column", ACS_TEMPORAL_YEAR_COLUMN)
    positive_label = settings["data"].get("positive_label", ">50K")
    negative_label = "<=50K" if positive_label != "<=50K" else "not_positive"
    random_seed = settings["project"]["random_seed"]
    test_size = settings["data"]["test_size"]
    temporal_split_enabled, train_years, eval_years = _resolve_acs_temporal_split(acs_cfg)

    cache_dir = ensure_dir(raw_dir / "folktables_cache")
    logger.info(
        "Running memory-efficient ACS pipeline "
        f"(years={survey_years}, horizon={horizon}, survey={survey}, states={len(states)}, "
        f"temporal_split={temporal_split_enabled})"
    )

    total_rows = 0
    total_train = 0
    total_test = 0

    from sklearn.model_selection import train_test_split

    shard_keys: list[tuple[str, str]] = []
    shard_total = len(survey_years) * len(states)
    shard_index = 0

    for survey_year in survey_years:
        source = ACSDataSource(
            survey_year=survey_year,
            horizon=horizon,
            survey=survey,
            root_dir=str(cache_dir),
        )

        for state in states:
            shard_index += 1
            shard_keys.append((survey_year, state))
            train_shard = shard_dir / f"train_{survey_year}_{state}.csv"
            test_shard = shard_dir / f"test_{survey_year}_{state}.csv"
            snapshot_shard = (
                snapshot_shard_dir / f"snapshot_{survey_year}_{state}.csv"
                if snapshot_shard_dir is not None
                else None
            )

            if train_shard.exists() and test_shard.exists() and (
                snapshot_shard is None or snapshot_shard.exists()
            ):
                logger.info(
                    f"[{shard_index}/{shard_total}] Skipping ACS state {state} year {survey_year} "
                    "(existing shards found)"
                )
                train_rows = max(sum(1 for _ in open(train_shard, "r")) - 1, 0)
                test_rows = max(sum(1 for _ in open(test_shard, "r")) - 1, 0)
                total_train += train_rows
                total_test += test_rows
                total_rows += train_rows + test_rows
                continue

            logger.info(f"[{shard_index}/{shard_total}] Loading ACS state {state} year {survey_year}")
            state_df = _load_single_acs_state(
                state=state,
                survey_year=survey_year,
                source=source,
                target_col=target_col,
                year_col=year_col,
                positive_label=positive_label,
                negative_label=negative_label,
            )
            state_df = clean_data(state_df)

            if temporal_split_enabled:
                if survey_year in train_years:
                    train_chunk = state_df.reset_index(drop=True)
                    test_chunk = state_df.iloc[0:0].copy().reset_index(drop=True)
                elif survey_year in eval_years:
                    train_chunk = state_df.iloc[0:0].copy().reset_index(drop=True)
                    test_chunk = state_df.reset_index(drop=True)
                else:
                    train_chunk = state_df.reset_index(drop=True)
                    test_chunk = state_df.iloc[0:0].copy().reset_index(drop=True)
            else:
                train_chunk, test_chunk = train_test_split(
                    state_df,
                    test_size=test_size,
                    random_state=random_seed,
                    stratify=state_df[target_col],
                )
                train_chunk = train_chunk.reset_index(drop=True)
                test_chunk = test_chunk.reset_index(drop=True)

            if save_snapshot and snapshot_shard is not None:
                _write_df_atomic(state_df, snapshot_shard)

            _write_df_atomic(train_chunk, train_shard)
            _write_df_atomic(test_chunk, test_shard)

            total_rows += len(state_df)
            total_train += len(train_chunk)
            total_test += len(test_chunk)

            logger.info(
                f"  {state} {survey_year}: rows={len(state_df)}, train={len(train_chunk)}, "
                f"test={len(test_chunk)}"
            )

            del state_df, train_chunk, test_chunk
            gc.collect()

    train_shards = [shard_dir / f"train_{year}_{state}.csv" for year, state in shard_keys]
    test_shards = [shard_dir / f"test_{year}_{state}.csv" for year, state in shard_keys]
    _combine_csv_shards(train_shards, train_path)
    _combine_csv_shards(test_shards, test_path)

    if save_snapshot and snapshot_shard_dir is not None:
        snapshot_shards = [
            snapshot_shard_dir / f"snapshot_{year}_{state}.csv" for year, state in shard_keys
        ]
        _combine_csv_shards(snapshot_shards, snapshot_path)

    logger.info(
        "Memory-efficient ACS pipeline complete: "
        f"rows={total_rows}, train={total_train}, test={total_test}"
    )
    logger.info(f"Saved processed ACS data: {train_path}, {test_path}")
    if save_snapshot:
        logger.info(f"Saved ACS snapshot: {snapshot_path}")
    return train_path, test_path


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
    settings = load_settings()
    out_dir = ensure_dir(get_path("processed_data_dir", settings))
    train_path, test_path = get_processed_data_paths(settings)

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

    settings = load_settings()
    dataset_profile = settings["data"].get("dataset_profile", "adult").lower()

    # Ensure output directories exist
    ensure_dir(get_path("raw_data_dir", settings))
    ensure_dir(get_path("processed_data_dir", settings))

    if dataset_profile == "acs":
        acs_cfg = settings["data"].get("acs", {})
        if acs_cfg.get("memory_efficient_processing", True):
            return _run_acs_pipeline_memory_efficient(settings)

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
