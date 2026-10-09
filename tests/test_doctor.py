"""Unit tests for neuralvault doctor and models CLI commands."""

from unittest.mock import patch

import pytest

from neuralvault.cli import main
from neuralvault.config import OfflineError


def test_cli_doctor(capsys):
    test_args = ["neuralvault", "doctor"]
    with patch("sys.argv", test_args):
        main()

    captured = capsys.readouterr()
    assert "NeuralVault Doctor Diagnostic Report" in captured.out
    assert "Vector Search:" in captured.out


def test_cli_models_download_offline(monkeypatch):
    monkeypatch.setenv("NEURALVAULT_OFFLINE", "1")
    test_args = ["neuralvault", "models", "download", "--model", "BAAI/bge-small-en-v1.5"]
    with patch("sys.argv", test_args):
        with pytest.raises(OfflineError, match="Offline mode is active"):
            main()
