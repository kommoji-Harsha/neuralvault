"""Unit tests for FastAPI REST API."""

from fastapi.testclient import TestClient

from neuralvault.api.app import app
from neuralvault.contract import Chunk, Document
from neuralvault.store.sqlite_store import SqliteStore

client = TestClient(app)


def test_rest_api_endpoints(tmp_path, monkeypatch):
    monkeypatch.setenv("NEURALVAULT_INDEXES_DIR", str(tmp_path))

    store = SqliteStore("api_test_col", index_dir=tmp_path)
    doc = Document(
        doc_id="doc_api_1",
        source="docs/api.md",
        title="API Doc",
        mime_type="text/markdown",
        hash="hash1",
        mtime=100.0,
        word_count=10,
        collection="api_test_col",
    )
    chunk = Chunk(
        chunk_id="c_api_1",
        text="NeuralVault REST API provides FastAPI endpoints.",
        source="docs/api.md",
        location="docs/api.md > Endpoints",
        score=0.9,
        metadata={"doc_id": "doc_api_1"},
    )
    store.add_documents_and_chunks([doc], [chunk])

    # 1. GET /collections
    res = client.get("/collections")
    assert res.status_code == 200
    col_names = [c["name"] for c in res.json()]
    assert "api_test_col" in col_names

    # 2. POST /search
    res = client.post(
        "/search",
        json={
            "query": "REST API",
            "collection": "api_test_col",
            "top_k": 2,
            "profile": "balanced",
        },
    )
    assert res.status_code == 200
    s_data = res.json()
    assert len(s_data["results"]) == 1
    assert s_data["results"][0]["chunk_id"] == "c_api_1_comp"

    # 3. POST /ask
    res = client.post(
        "/ask",
        json={
            "query": "What does REST API provide?",
            "collection": "api_test_col",
            "top_k": 2,
            "profile": "balanced",
        },
    )
    assert res.status_code == 200
    a_data = res.json()
    assert a_data["insufficient_context"] is False
    assert len(a_data["citations"]) == 1

    # 4. GET /documents/{collection}/{doc_id}
    res = client.get("/documents/api_test_col/doc_api_1")
    assert res.status_code == 200
    assert res.json()["doc_id"] == "doc_api_1"


def test_rest_api_key_auth(monkeypatch):
    monkeypatch.setenv("NEURALVAULT_API_KEY", "secret_key_123")

    # Unauthorized without key
    res = client.get("/collections")
    assert res.status_code == 401

    # Authorized with key
    res = client.get("/collections", headers={"X-API-Key": "secret_key_123"})
    assert res.status_code == 200
