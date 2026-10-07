"""Tests for NeuralVault contract models."""

from neuralvault.contract import (
    AskRequest,
    AskResponse,
    Chunk,
    Collection,
    Document,
    SearchRequest,
    SearchResponse,
)


def test_chunk_model():
    chunk = Chunk(
        chunk_id="chunk-1",
        text="def search(): pass",
        source="src/search.py",
        location="src/search.py::search",
        score=0.95,
    )
    assert chunk.chunk_id == "chunk-1"
    assert chunk.text == "def search(): pass"
    assert chunk.source == "src/search.py"
    assert chunk.location == "src/search.py::search"
    assert chunk.score == 0.95
    assert chunk.embedding_text is None
    assert chunk.metadata == {}


def test_document_model():
    doc = Document(
        doc_id="doc-1",
        source="README.md",
        title="README",
        mime_type="text/markdown",
        hash="a1b2c3d4",
        mtime=123456789.0,
        word_count=100,
        collection="test-collection",
    )
    assert doc.doc_id == "doc-1"
    assert doc.source == "README.md"
    assert doc.hash == "a1b2c3d4"
    assert doc.collection == "test-collection"


def test_collection_model():
    col = Collection(name="assistant-project", description="Main project docs")
    assert col.name == "assistant-project"
    assert col.embedding_model == "BAAI/bge-small-en-v1.5"
    assert col.embedding_dim == 384
    assert col.document_count == 0


def test_search_request_and_response():
    req = SearchRequest(query="how does rrf work", collection="docs", top_k=5)
    assert req.query == "how does rrf work"
    assert req.top_k == 5
    assert req.profile == "balanced"

    resp = SearchResponse(
        results=[],
        note="no relevant passages found",
        latency_ms=12.5,
    )
    assert len(resp.results) == 0
    assert resp.note == "no relevant passages found"
    assert resp.latency_ms == 12.5


def test_ask_request_and_response():
    req = AskRequest(query="What is NeuralVault?", collection="docs")
    assert req.query == "What is NeuralVault?"

    chunk = Chunk(
        chunk_id="c1",
        text="NeuralVault is an offline retrieval library.",
        source="DESIGN.md",
        location="DESIGN.md > Overview",
        score=0.9,
    )
    resp = AskResponse(
        answer="NeuralVault is an offline retrieval library.",
        citations=[chunk],
        insufficient_context=False,
    )
    assert resp.answer == "NeuralVault is an offline retrieval library."
    assert len(resp.citations) == 1
    assert resp.citations[0].chunk_id == "c1"
    assert resp.insufficient_context is False
