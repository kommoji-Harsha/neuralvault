"""Unit tests for post-retrieval processing."""

from neuralvault.contract import Chunk, Document
from neuralvault.retrieve.post_process import (
    apply_relevance_threshold,
    mmr_deduplicate,
    parent_expand_chunks,
)
from neuralvault.store.sqlite_store import SqliteStore


def test_mmr_deduplicate():
    c1 = Chunk(
        chunk_id="c1",
        text="duplicate text content",
        source="a.py",
        location="L1",
        score=0.9,
    )
    c2 = Chunk(
        chunk_id="c2",
        text="duplicate text content",
        source="a.py",
        location="L2",
        score=0.85,
    )
    c3 = Chunk(
        chunk_id="c3",
        text="unique content here",
        source="b.py",
        location="L1",
        score=0.7,
    )

    deduped = mmr_deduplicate([c1, c2, c3], max_similarity=0.8)
    assert len(deduped) == 2
    assert deduped[0].chunk_id == "c1"
    assert deduped[1].chunk_id == "c3"


def test_apply_relevance_threshold():
    c1 = Chunk(chunk_id="c1", text="high score", source="a.py", location="L1", score=0.5)
    c2 = Chunk(chunk_id="c2", text="low score", source="a.py", location="L2", score=0.00001)

    filtered = apply_relevance_threshold([c1, c2], min_score=0.001)
    assert len(filtered) == 1
    assert filtered[0].chunk_id == "c1"


def test_parent_expand_chunks(tmp_path):
    store = SqliteStore("test_parent", index_dir=tmp_path)
    doc = Document(doc_id="d1", source="a.py", title="A", hash="h1", collection="test_parent")

    p_chunk = Chunk(
        chunk_id="p1",
        text="class Parent:\n    def method(self): pass",
        source="a.py",
        location="a.py::Parent",
        score=0.5,
        metadata={"doc_id": "d1"},
    )
    sub_chunk = Chunk(
        chunk_id="c1",
        text="def method(self): pass",
        source="a.py",
        location="a.py::Parent.method (part 1)",
        score=0.8,
        metadata={"doc_id": "d1"},
    )

    store.add_documents_and_chunks([doc], [p_chunk, sub_chunk])

    expanded = parent_expand_chunks([sub_chunk], store=store)
    assert len(expanded) == 1
    assert "class Parent:" in expanded[0].text
    assert "(Parent Expanded)" in expanded[0].location
