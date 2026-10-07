"""Tests for configuration and offline mode guard."""

import pytest
import yaml

from neuralvault.config import (
    OfflineError,
    get_indexes_dir,
    get_neuralvault_home,
    guard_offline,
    is_offline,
    load_collections_config,
)


def test_offline_guard(monkeypatch):
    monkeypatch.setenv("NEURALVAULT_OFFLINE", "1")
    assert is_offline() is True
    with pytest.raises(OfflineError, match="Offline mode is active"):
        guard_offline("download model")

    monkeypatch.setenv("NEURALVAULT_OFFLINE", "0")
    assert is_offline() is False
    guard_offline("download model")  # should not raise


def test_paths(monkeypatch, tmp_path):
    monkeypatch.setenv("NEURALVAULT_HOME", str(tmp_path / "nv_home"))
    monkeypatch.setenv("NEURALVAULT_INDEXES_DIR", str(tmp_path / "nv_indexes"))

    home = get_neuralvault_home()
    indexes = get_indexes_dir()

    assert home == tmp_path / "nv_home"
    assert indexes == tmp_path / "nv_indexes"
    assert home.exists()
    assert indexes.exists()


def test_load_collections_config(tmp_path):
    config_file = tmp_path / "collections.yaml"
    data = {
        "collections": {
            "test_col": {
                "description": "A test collection",
                "embedding_model": "bge-small",
            }
        }
    }
    config_file.write_text(yaml.dump(data), encoding="utf-8")

    loaded = load_collections_config(config_file)
    assert "test_col" in loaded
    assert loaded["test_col"]["description"] == "A test collection"
