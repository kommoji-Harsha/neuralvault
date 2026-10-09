"""Unit tests for document loaders and ingestion engine."""

import json

import yaml

from neuralvault.ingest.engine import IngestEngine
from neuralvault.ingest.loaders import (
    CsvTsvLoader,
    HTMLLoader,
    JsonYamlLoader,
    load_file,
)


def test_text_loader(tmp_path):
    f = tmp_path / "doc.txt"
    f.write_text("Hello world text content.")
    doc, content = load_file(f, collection="test")
    assert doc.mime_type == "text/plain"
    assert doc.title == "Doc"
    assert content == "Hello world text content."


def test_markdown_loader(tmp_path):
    f = tmp_path / "guide.md"
    f.write_text("# User Guide\n\nThis is a test guide.")
    doc, content = load_file(f, collection="test")
    assert doc.mime_type == "text/markdown"
    assert doc.title == "User Guide"
    assert "# User Guide" in content


def test_python_loader(tmp_path):
    f = tmp_path / "script.py"
    f.write_text('"""Module title."""\n\ndef main(): pass')
    doc, content = load_file(f, collection="test")
    assert doc.mime_type == "text/x-python"
    assert doc.title == "Module title."


def test_html_loader(tmp_path):
    f = tmp_path / "page.html"
    f.write_text(
        "<html><head><title>Test Page</title></head><body>"
        "<h1>Heading</h1><p>Body</p></body></html>"
    )
    loader = HTMLLoader(collection="test")
    doc, content = loader.load(f)
    assert doc.title == "Test Page"
    assert "Heading" in content
    assert "Body" in content


def test_csv_tsv_loader(tmp_path):
    f_csv = tmp_path / "data.csv"
    f_csv.write_text("name,age\nAlice,30\nBob,25")
    loader = CsvTsvLoader(collection="test")
    doc, content = loader.load(f_csv)
    assert "name | age" in content
    assert "Alice | 30" in content


def test_json_yaml_loader(tmp_path):
    f_json = tmp_path / "data.json"
    f_json.write_text(json.dumps({"key": "value"}))
    loader = JsonYamlLoader(collection="test")
    doc, content = loader.load(f_json)
    assert '"key": "value"' in content

    f_yaml = tmp_path / "config.yaml"
    f_yaml.write_text(yaml.dump({"setting": True}))
    doc_y, content_y = loader.load(f_yaml)
    assert "setting: true" in content_y.lower()


def test_incremental_ingest_and_deletion(tmp_path):
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()
    f1 = corpus_dir / "file1.md"
    f1.write_text("# File 1\n\nContent 1")

    index_dir = tmp_path / "indexes"
    engine = IngestEngine(collection="col1", index_dir=index_dir)

    # Initial ingest
    stats1 = engine.ingest_path(corpus_dir)
    assert stats1["processed"] == 1
    assert stats1["skipped"] == 0
    assert stats1["deleted"] == 0
    assert stats1["total_documents"] == 1

    # Second ingest (unchanged)
    stats2 = engine.ingest_path(corpus_dir)
    assert stats2["processed"] == 0
    assert stats2["skipped"] == 1

    # Modify file
    f1.write_text("# File 1 Updated\n\nContent 1 updated")
    stats3 = engine.ingest_path(corpus_dir)
    assert stats3["processed"] == 1
    assert stats3["skipped"] == 0

    # Delete file
    f1.unlink()
    stats4 = engine.ingest_path(corpus_dir)
    assert stats4["processed"] == 0
    assert stats4["deleted"] == 1
    assert stats4["total_documents"] == 0
    assert stats4["total_chunks"] == 0
