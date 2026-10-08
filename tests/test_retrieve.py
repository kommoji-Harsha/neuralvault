"""Unit tests for NeuralVault retrievers (BM25, Vector, Hybrid RRF)."""

from neuralvault.contract import Chunk, Document
from neuralvault.embed.providers import HashEmbeddingProvider
from neuralvault.retrieve.bm25 import BM25Retriever, tokenize_identifiers
from neuralvault.retrieve.hybrid import HybridRetriever
from neuralvault.retrieve.vector import VectorRetriever
from neuralvault.store.sqlite_store import SqliteStore


def test_bm25_identifier_tokenizer():
    tokens_snake = tokenize_identifiers("search_memory")
    assert "search_memory" in tokens_snake
    assert "search" in tokens_snake
    assert "memory" in tokens_snake

    tokens_camel = tokenize_identifiers("HybridRetriever.search")
    assert "HybridRetriever" in tokens_camel
    assert "Hybrid" in tokens_camel
    assert "Retriever" in tokens_camel
    assert "search" in tokens_camel


def test_exact_function_name_query_bm25_wins(tmp_path):
    store = SqliteStore("test_exact", index_dir=tmp_path)
    embedder = HashEmbeddingProvider()

    doc = Document(
        doc_id="d1", source="src/tools.py", title="Tools", hash="h1", collection="test_exact"
    )

    c1 = Chunk(
        chunk_id="c_exact",
        text="def _extract_tool_calls(payload): parse tool calls from response",
        source="src/tools.py",
        location="src/tools.py::_extract_tool_calls",
    )
    c2 = Chunk(
        chunk_id="c_generic",
        text="def handle_request(req): process request payload and return result",
        source="src/tools.py",
        location="src/tools.py::handle_request",
    )

    vecs = embedder.embed([c1.text, c2.text])
    store.add_documents_and_chunks([doc], [c1, c2], vecs)

    retriever = BM25Retriever(store)
    results = retriever.retrieve("_extract_tool_calls", top_k=2)

    assert len(results) > 0
    # BM25 must rank exact function match #1
    assert results[0].chunk_id == "c_exact"
    assert results[0].metadata["retriever"] == "bm25"
    assert "bm25" in results[0].metadata["provenance"]["retrievers"]


def test_paraphrase_query_vector_wins(tmp_path):
    store = SqliteStore("test_para", index_dir=tmp_path)
    embedder = HashEmbeddingProvider()

    doc = Document(
        doc_id="d1", source="docs/faq.md", title="FAQ", hash="h1", collection="test_para"
    )

    # Chunk with semantic match but no exact query words
    c1 = Chunk(
        chunk_id="c_semantic",
        text="How to restore previous conversation state when an execution error occurs.",
        source="docs/faq.md",
        location="docs/faq.md > Errors",
    )
    c2 = Chunk(
        chunk_id="c_unrelated",
        text="Installing dependencies using pip install and setting virtual environment.",
        source="docs/faq.md",
        location="docs/faq.md > Setup",
    )

    vecs = embedder.embed([c1.text, c2.text])
    store.add_documents_and_chunks([doc], [c1, c2], vecs)

    retriever = VectorRetriever(store, embedder)
    # Query with exact text of c1
    results = retriever.retrieve(
        "How to restore previous conversation state when an execution error occurs.",
        top_k=2,
    )

    assert len(results) > 0
    # Vector retriever must rank match #1
    assert results[0].chunk_id == "c_semantic"
    assert results[0].metadata["retriever"] == "vector"
    assert "vector" in results[0].metadata["provenance"]["retrievers"]


def test_hybrid_rrf_fusion_and_provenance(tmp_path):
    store = SqliteStore("test_hybrid", index_dir=tmp_path)
    embedder = HashEmbeddingProvider()

    doc = Document(
        doc_id="d1", source="src/workflow.py", title="Workflow", hash="h1", collection="test_hybrid"
    )

    c1 = Chunk(
        chunk_id="c1",
        text="confirmation workflow for dangerous tools asking user confirmation",
        source="src/workflow.py",
        location="src/workflow.py::confirm",
    )
    c2 = Chunk(
        chunk_id="c2",
        text="unrelated helper function calculating string hash",
        source="src/workflow.py",
        location="src/workflow.py::hash",
    )

    vecs = embedder.embed([c1.text, c2.text])
    store.add_documents_and_chunks([doc], [c1, c2], vecs)

    hybrid = HybridRetriever(store, embedder, rrf_k=60, candidate_k=20)
    results = hybrid.retrieve("confirmation workflow for dangerous tools", top_k=2)

    assert len(results) > 0
    top_result = results[0]
    assert top_result.chunk_id == "c1"
    assert top_result.metadata["retriever"] == "hybrid_rrf"
    prov = top_result.metadata["provenance"]
    assert "rrf_score" in prov
    assert "bm25" in prov["retrievers"] or "vector" in prov["retrievers"]
