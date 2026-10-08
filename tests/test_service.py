"""Unit tests for NeuralVault RagService."""

from neuralvault.contract import AskRequest, Chunk, Document, SearchRequest
from neuralvault.service.service import RagService
from neuralvault.store.sqlite_store import SqliteStore


def test_rag_service_flow(tmp_path):
    store = SqliteStore("assistant-project", index_dir=tmp_path)

    doc = Document(
        doc_id="d1",
        source="src/neuralvault/retrieve.py",
        title="Retrieve",
        mime_type="text/x-python",
        hash="h1",
        mtime=123.0,
        word_count=40,
        collection="assistant-project",
    )

    chunk = Chunk(
        chunk_id="c1",
        text="def search_memory(): return True",
        source="src/neuralvault/retrieve.py",
        location="src/neuralvault/retrieve.py::search_memory",
        score=0.9,
        metadata={"doc_id": "d1"},
    )

    store.add_documents_and_chunks([doc], [chunk])

    service = RagService(index_dir=tmp_path, provider_type="hash")

    # Test list_collections
    cols = service.list_collections()
    assert len(cols) >= 1
    col_names = [c.name for c in cols]
    assert "assistant-project" in col_names

    # Test get_document
    fetched_doc = service.get_document("assistant-project", "d1")
    assert fetched_doc is not None
    assert fetched_doc.doc_id == "d1"

    # Test search profiles
    for profile in ["fast", "balanced", "best"]:
        req = SearchRequest(
            query="search_memory",
            collection="assistant-project",
            top_k=2,
            profile=profile,
        )
        resp = service.search(req)
        assert len(resp.results) > 0
        assert resp.results[0].source == "src/neuralvault/retrieve.py"
        assert resp.latency_ms > 0

    # Test ask
    ask_req = AskRequest(
        query="What is search_memory?",
        collection="assistant-project",
        top_k=2,
        profile="balanced",
    )
    ask_resp = service.ask(ask_req)
    assert ask_resp.insufficient_context is False
    assert len(ask_resp.citations) > 0
