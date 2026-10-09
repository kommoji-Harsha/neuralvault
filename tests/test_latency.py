"""Latency regression benchmark test for RagService balanced profile."""

import time

from neuralvault.contract import Chunk, Document, SearchRequest
from neuralvault.service.service import RagService
from neuralvault.store.sqlite_store import SqliteStore


def test_balanced_profile_latency_under_500ms(tmp_path):
    store = SqliteStore("benchmark_col", index_dir=tmp_path)

    # Populate store with a fixture corpus of 50 documents and 100 chunks
    docs = []
    chunks = []
    for i in range(50):
        d_id = f"doc_{i}"
        src = f"src/module_{i}.py"
        docs.append(
            Document(
                doc_id=d_id,
                source=src,
                title=f"Module {i}",
                mime_type="text/x-python",
                hash=f"hash_{i}",
                mtime=1000.0,
                word_count=100,
                collection="benchmark_col",
            )
        )
        chunks.append(
            Chunk(
                chunk_id=f"chunk_{i}_1",
                text=f"def function_{i}():\n    '''Function {i} docstring.'''\n    return {i}",
                source=src,
                location=f"{src}::function_{i}",
                score=0.8,
                metadata={"doc_id": d_id},
            )
        )
        chunks.append(
            Chunk(
                chunk_id=f"chunk_{i}_2",
                text=f"class Class_{i}:\n    '''Class {i} docstring.'''\n    pass",
                source=src,
                location=f"{src}::Class_{i}",
                score=0.7,
                metadata={"doc_id": d_id},
            )
        )

    store.add_documents_and_chunks(docs, chunks)

    service = RagService(index_dir=tmp_path, provider_type="hash")
    req = SearchRequest(
        query="function_25 docstring",
        collection="benchmark_col",
        top_k=3,
        profile="balanced",
    )

    # Warmup
    service.search(req)

    # Benchmark run
    t0 = time.perf_counter()
    res = service.search(req)
    t1 = time.perf_counter()

    latency_ms = (t1 - t0) * 1000.0

    assert len(res.results) > 0
    assert latency_ms < 500.0, f"Balanced profile latency exceeded 500ms limit: {latency_ms:.2f}ms"
