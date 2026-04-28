"""
Utility functions for the Drift-Aware MLOps Pipeline.

Provides config loading, logging setup, and path helpers.
"""

import os
import logging
import yaml
from pathlib import Path


# ── Project root detection ──────────────────────────────────────────
def get_project_root() -> Path:
    """Return the absolute path to the project root directory."""
    # Walk up from this file until we find configs/settings.yaml
    current = Path(__file__).resolve().parent  # src/
    root = current.parent  # project root
    if (root / "configs" / "settings.yaml").exists():
        return root
    # Fallback: assume cwd
    return Path.cwd()


PROJECT_ROOT = get_project_root()


# ── YAML config loader ─────────────────────────────────────────────
def load_yaml(filename: str) -> dict:
    """
    Load a YAML config file from the configs/ directory.

    Parameters
    ----------
    filename : str
        Name of the YAML file (e.g. 'settings.yaml').

    Returns
    -------
    dict
        Parsed YAML content.
    """
    config_path = PROJECT_ROOT / "configs" / filename
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_settings() -> dict:
    """
    Load settings config.

    Default file is configs/settings.yaml.
    Override via SETTINGS_FILE env var:
      - "settings_acs.yaml" (resolved under configs/)
      - "/absolute/path/to/file.yaml"
      - "configs/settings_acs.yaml" (relative to project root)
    """
    settings_file = os.environ.get("SETTINGS_FILE", "settings.yaml")
    candidate = Path(settings_file)

    if candidate.is_absolute():
        config_path = candidate
    else:
        config_path = PROJECT_ROOT / candidate
        if not config_path.exists():
            config_path = PROJECT_ROOT / "configs" / settings_file

    if not config_path.exists():
        raise FileNotFoundError(f"Settings file not found: {config_path}")

    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_drift_config() -> dict:
    """
    Load the drift configuration.

    Resolution order:
    1. `DRIFT_CONFIG_FILE` environment variable
    2. `drift_config_acs.yaml` when dataset_profile=acs and the file exists
    3. `drift_config.yaml`
    """
    config_file = os.environ.get("DRIFT_CONFIG_FILE")
    if config_file:
        candidate = Path(config_file)
        if not candidate.is_absolute():
            candidate = PROJECT_ROOT / candidate
            if not candidate.exists():
                candidate = PROJECT_ROOT / "configs" / config_file
        if not candidate.exists():
            raise FileNotFoundError(f"Drift config file not found: {candidate}")
        with open(candidate, "r") as f:
            return yaml.safe_load(f)

    settings = load_settings()
    dataset_profile = settings["data"].get("dataset_profile", "adult").lower()
    if dataset_profile == "acs":
        acs_config = PROJECT_ROOT / "configs" / "drift_config_acs.yaml"
        if acs_config.exists():
            with open(acs_config, "r") as f:
                return yaml.safe_load(f)

    return load_yaml("drift_config.yaml")


def load_thresholds() -> dict:
    """Load the thresholds.yaml config."""
    return load_yaml("thresholds.yaml")


# ── Path helpers ────────────────────────────────────────────────────
def ensure_dir(path: str | Path) -> Path:
    """Create a directory (and parents) if it doesn't exist. Returns the Path."""
    p = PROJECT_ROOT / path if not Path(path).is_absolute() else Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def get_path(settings_key: str, settings: dict | None = None) -> Path:
    """
    Resolve a path from settings.yaml paths section.

    Parameters
    ----------
    settings_key : str
        Key under `paths:` in settings.yaml (e.g. 'raw_data_dir').
    settings : dict, optional
        Pre-loaded settings dict. If None, loads from file.

    Returns
    -------
    Path
        Absolute path.
    """
    if settings is None:
        settings = load_settings()
    rel = settings["paths"][settings_key]
    return PROJECT_ROOT / rel


def get_processed_data_paths(settings: dict | None = None) -> tuple[Path, Path]:
    """
    Resolve processed train/test file paths from settings.

    This allows different dataset profiles to persist distinct processed splits.
    """
    if settings is None:
        settings = load_settings()

    processed_dir = get_path("processed_data_dir", settings)
    train_name = settings["data"].get("processed_train_filename", "train.csv")
    test_name = settings["data"].get("processed_test_filename", "test.csv")
    return processed_dir / train_name, processed_dir / test_name


# ── Logging ─────────────────────────────────────────────────────────
def setup_logging(
    name: str = "mlops",
    level: str | None = None,
    log_file: str | None = None,
) -> logging.Logger:
    """
    Configure and return a logger.

    Parameters
    ----------
    name : str
        Logger name.
    level : str, optional
        Log level (DEBUG/INFO/WARNING/ERROR). Defaults to settings.yaml value.
    log_file : str, optional
        If provided, also log to this file inside logs/.

    Returns
    -------
    logging.Logger
    """
    if level is None:
        try:
            settings = load_settings()
            level = settings["project"].get("log_level", "INFO")
        except FileNotFoundError:
            level = "INFO"

    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    # Avoid duplicate handlers on repeated calls
    if not logger.handlers:
        fmt = logging.Formatter(
            "%(asctime)s | %(name)-12s | %(levelname)-7s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        # Console handler
        ch = logging.StreamHandler()
        ch.setFormatter(fmt)
        logger.addHandler(ch)

        # File handler (optional)
        if log_file:
            log_dir = ensure_dir("logs")
            fh = logging.FileHandler(log_dir / log_file)
            fh.setFormatter(fmt)
            logger.addHandler(fh)

    return logger
