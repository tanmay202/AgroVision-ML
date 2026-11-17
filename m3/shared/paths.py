"""
AgroVision -- Centralized Path Management
=========================================
Dynamically resolves paths for monorepo root, Member 1, and Member 2 directories.
Prevents hardcoded machine-specific paths and current working directory dependencies.
"""

from pathlib import Path


def get_project_root() -> Path:
    """Return the AgroVision monorepo root directory."""
    return Path(__file__).resolve().parent.parent


def get_member1_root() -> Path:
    """Return the Member 1 (yield model) root directory."""
    return get_project_root() / "member1-yield-model"


def get_member2_root() -> Path:
    """Return the Member 2 (price volatility) root directory."""
    return get_project_root() / "member2-price-volatility"


def get_m1_data_dir() -> Path:
    """Return Member 1 data root directory."""
    return get_member1_root() / "data"


def get_m1_raw_dir() -> Path:
    """Return Member 1 raw data directory."""
    return get_m1_data_dir() / "raw"


def get_m1_processed_dir() -> Path:
    """Return Member 1 processed data directory."""
    return get_m1_data_dir() / "processed"


def get_m1_models_dir() -> Path:
    """Return Member 1 models directory."""
    return get_member1_root() / "models"


def get_m1_scripts_dir() -> Path:
    """Return Member 1 scripts directory."""
    return get_member1_root() / "scripts"
