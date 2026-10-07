"""Unit tests for NeuralVault CLI commands."""

from unittest.mock import patch

from neuralvault.cli import main


def test_cli_ingest(tmp_path, capsys):
    file1 = tmp_path / "note.md"
    file1.write_text("# Test Note\n\nSome note text.")

    index_dir = tmp_path / "indexes"

    test_args = [
        "neuralvault",
        "ingest",
        "test-col",
        "--path",
        str(file1),
        "--index-dir",
        str(index_dir),
    ]

    with patch("sys.argv", test_args):
        main()

    captured = capsys.readouterr()
    assert "Ingesting" in captured.out
    assert "Ingestion complete" in captured.out
    assert "Total Docs: 1" in captured.out

    store_file = index_dir / "test-col" / "store.jsonl"
    assert store_file.exists()
