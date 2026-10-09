"""Unit tests for RagClient Python SDK."""

from neuralvault.client.client import RagClient
from neuralvault.contract import Chunk, Document
from neuralvault.store.sqlite_store import SqliteStore


def test_rag_client_local_mode(tmp_path):
    store = SqliteStore("sdk_test_col", index_dir=tmp_path)
    doc = Document(
        doc_id="d_sdk_1",
        source="sdk.py",
        title="SDK",
        hash="h_sdk",
        collection="sdk_test_col",
    )
    chunk = Chunk(
        chunk_id="c_sdk_1",
        text="RagClient connects locally or over HTTP.",
        source="sdk.py",
        location="sdk.py::RagClient",
        score=0.95,
        metadata={"doc_id": "d_sdk_1"},
    )
    store.add_documents_and_chunks([doc], [chunk])

    client = RagClient(
        mode="local",
        collection="sdk_test_col",
        index_dir=tmp_path,
    )

    cols = client.list_collections()
    assert len(cols) >= 1

    fetched_doc = client.get_document("d_sdk_1")
    assert fetched_doc is not None
    assert fetched_doc.doc_id == "d_sdk_1"

    s_res = client.search("RagClient")
    assert len(s_res.results) == 1

    a_res = client.ask("What is RagClient?")
    assert a_res.insufficient_context is False
