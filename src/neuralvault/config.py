"""NeuralVault configuration and offline enforcement guard.

Handles environment configuration, default paths, collections.yaml loading,
and NEURALVAULT_OFFLINE enforcement.
"""

import os
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


class OfflineError(Exception):
    """Raised when a network operation is attempted while NEURALVAULT_OFFLINE=1."""

    pass


def is_offline() -> bool:
    """Check if NeuralVault offline mode is enabled via NEURALVAULT_OFFLINE env var.

    Returns:
        True if NEURALVAULT_OFFLINE is set to '1', 'true', 'yes', or 'on' (case-insensitive).
    """
    val = os.getenv("NEURALVAULT_OFFLINE", "0").strip().lower()
    return val in ("1", "true", "yes", "on")


def guard_offline(action: str = "network request") -> None:
    """Raise OfflineError if offline mode is active.

    Args:
        action: Descriptive action string for the error message.

    Raises:
        OfflineError: If offline mode is enabled.
    """
    if is_offline():
        raise OfflineError(
            f"Offline mode is active (NEURALVAULT_OFFLINE=1). Blocked action: {action}"
        )


def get_neuralvault_home() -> Path:
    """Get NEURALVAULT_HOME directory path (defaults to ~/.neuralvault).

    Returns:
        Path object for NEURALVAULT_HOME.
    """
    env_home = os.getenv("NEURALVAULT_HOME")
    if env_home:
        path = Path(env_home)
    else:
        path = Path.home() / ".neuralvault"
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_indexes_dir() -> Path:
    """Get the directory where collection SQLite databases are stored.

    Returns:
        Path object for the indexes directory.
    """
    env_indexes = os.getenv("NEURALVAULT_INDEXES_DIR")
    if env_indexes:
        path = Path(env_indexes)
    else:
        path = Path("indexes")
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_collections_config(config_path: Optional[Path] = None) -> Dict[str, Any]:
    """Load collection definitions from collections.yaml.

    Args:
        config_path: Optional explicit Path to collections.yaml file.

    Returns:
        Dictionary mapping collection names to collection settings.
    """
    if config_path is None:
        possible_paths = [
            Path("collections.yaml"),
            get_neuralvault_home() / "collections.yaml",
        ]
        for p in possible_paths:
            if p.exists():
                config_path = p
                break

    if config_path is None or not config_path.exists():
        return {}

    with open(config_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    return data.get("collections", {})
