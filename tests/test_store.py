"""Unit tests for SQLite collection store."""

import pytest

from neuralvault.contract import Chunk, Document
from neuralvault.store.sqlite_store import SqliteStore, cosine_similarity


def test_sqlite_store_crud_and_fts(tmp_path):
    store = SqliteStore("test_collection", index_dir=tmp_path)

    doc = Document(
        doc_id="doc1",
        source="src/main.py",
        title="Main Module",
        mime_type="text/x-python",
        hash="h123",
        mtime=100.0,
        word_count=50,
        collection="test_collection",
    )

    chunk = Chunk(
        chunk_id="c1",
        text="def search_memory(): return True",
        source="src/main.py",
        location="src/main.py::search_memory",
        score=0.0,
        embedding_text="src/main.py::search_memory\n\ndef search_memory(): return True",
        metadata={"doc_id": "doc1"},
    )

    store.add_documents_and_chunks([doc], [chunk], [[1.0, 0.0, 0.0] + [0.0] * 381])

    # FTS BM25 search
    bm25_matches = store.search_bm25("search_memory")
    assert len(bm25_matches) == 1
    assert bm25_matches[0].chunk_id == "c1"

    # Vector cosine search
    vec_matches = store.search_vector([1.0, 0.0, 0.0] + [0.0] * 381)
    assert len(vec_matches) == 1
    assert vec_matches[0].chunk_id == "c1"
    assert vec_matches[0].score > 0.99

    # Document deletion
    store.delete_document("src/main.py")
    assert len(store.search_bm25("search_memory")) == 0
    assert len(store.search_vector([1.0, 0.0, 0.0] + [0.0] * 381)) == 0


def test_sqlite_store_model_mismatch_error(tmp_path):
    SqliteStore(
        "model_test",
        index_dir=tmp_path,
        embedding_model="model_1",
        embedding_dim=384,
    )

    with pytest.raises(ValueError, match="model mismatch"):
        SqliteStore(
            "model_test",
            index_dir=tmp_path,
            embedding_model="model_2",
            embedding_dim=384,
        )


def test_cosine_similarity():
    import numpy as np

    a = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    b = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    c = np.array([0.0, 1.0, 0.0], dtype=np.float32)

    assert cosine_similarity(a, b) == pytest.approx(1.0)
    assert cosine_similarity(a, c) == pytest.approx(0.0)
