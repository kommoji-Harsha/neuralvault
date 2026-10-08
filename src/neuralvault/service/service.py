"""NeuralVault Service Layer providing list_collections, search, ask, and get_document."""

import time
from typing import List, Optional

from neuralvault.compress.compressor import ExtractiveCompressor
from neuralvault.config import get_indexes_dir, load_collections_config
from neuralvault.contract import (
    AskRequest,
    AskResponse,
    Collection,
    Document,
    SearchRequest,
    SearchResponse,
)
from neuralvault.embed.providers import get_embedding_provider
from neuralvault.generate.generator import get_generator
from neuralvault.rerank.reranker import get_reranker
from neuralvault.retrieve.hybrid import HybridRetriever
from neuralvault.retrieve.post_process import (
    apply_relevance_threshold,
    mmr_deduplicate,
    parent_expand_chunks,
)
from neuralvault.store.sqlite_store import SqliteStore


class RagService:
    """Core Retrieval-Augmented Generation Service layer."""

    def __init__(self, index_dir=None, provider_type: str = "hash") -> None:
        self.index_dir = get_indexes_dir() if index_dir is None else index_dir
        self.provider_type = provider_type

    def list_collections(self) -> List[Collection]:
        """List all available document collections."""
        config_collections = load_collections_config()
        collections_map = {}

        if self.index_dir.exists():
            for p in self.index_dir.iterdir():
                col_name = None
                if p.suffix == ".db":
                    col_name = p.stem
                elif p.is_dir() and (p / "store.jsonl").exists():
                    col_name = p.name

                if col_name:
                    try:
                        store = SqliteStore(col_name, index_dir=self.index_dir)
                        meta = store.get_collection_metadata()
                        if col_name in config_collections:
                            desc = config_collections[col_name].get("description", meta.description)
                            meta.description = desc
                        collections_map[col_name] = meta
                    except Exception:
                        desc = config_collections.get(col_name, {}).get("description", "")
                        collections_map[col_name] = Collection(
                            name=col_name,
                            description=desc,
                        )

        for col_name, cfg in config_collections.items():
            if col_name not in collections_map:
                collections_map[col_name] = Collection(
                    name=col_name,
                    description=cfg.get("description", ""),
                    embedding_model=cfg.get("embedding_model", "BAAI/bge-small-en-v1.5"),
                    embedding_dim=cfg.get("embedding_dim", 384),
                )

        return sorted(list(collections_map.values()), key=lambda c: c.name)

    def get_document(self, collection: str, doc_id_or_source: str) -> Optional[Document]:
        """Retrieve Document metadata by ID or source path."""
        try:
            store = SqliteStore(collection, index_dir=self.index_dir)
            with store._get_connection() as conn:
                cur = conn.cursor()
                cur.execute(
                    "SELECT * FROM documents WHERE doc_id = ? OR source = ?",
                    (doc_id_or_source, doc_id_or_source),
                )
                row = cur.fetchone()
                if row:
                    import json

                    meta = json.loads(row["metadata"]) if row["metadata"] else {}
                    return Document(
                        doc_id=row["doc_id"],
                        source=row["source"],
                        title=row["title"],
                        mime_type=row["mime_type"],
                        hash=row["hash"],
                        mtime=row["mtime"],
                        word_count=row["word_count"],
                        collection=row["collection"],
                        metadata=meta,
                    )
        except Exception:
            pass
        return None

    def search(self, request: SearchRequest) -> SearchResponse:
        """Perform search query across a collection using requested profile."""
        start_time = time.perf_counter()
        collection = request.collection
        profile = request.profile.lower()

        try:
            store = SqliteStore(collection, index_dir=self.index_dir)
            meta = store.get_collection_metadata()
            embedder = get_embedding_provider(
                self.provider_type,
                model_name=meta.embedding_model,
                dimension=meta.embedding_dim,
            )
            hybrid_retriever = HybridRetriever(
                store, embedder, candidate_k=max(20, request.top_k * 5)
            )

            # 1. Base Retrieval
            candidates = hybrid_retriever.retrieve(request.query, top_k=max(20, request.top_k * 5))

            if not candidates:
                elapsed_ms = (time.perf_counter() - start_time) * 1000.0
                return SearchResponse(
                    results=[], note="no relevant passages found", latency_ms=elapsed_ms
                )

            # 2. Profile Execution
            if profile == "fast":
                final_chunks = candidates[: request.top_k]
            elif profile == "balanced":
                # Hybrid + compression
                filtered = apply_relevance_threshold(candidates, min_score=0.0005)
                compressor = ExtractiveCompressor()
                compressed = compressor.compress(request.query, filtered, max_chars=1500)
                final_chunks = compressed[: request.top_k]
            elif profile == "best":
                # Hybrid + parent expansion + MMR + rerank + compression
                expanded = parent_expand_chunks(candidates, store=store)
                deduped = mmr_deduplicate(expanded)
                filtered = apply_relevance_threshold(deduped, min_score=0.0005)
                reranker = get_reranker("passthrough")
                reranked = reranker.rerank(request.query, filtered, top_k=max(5, request.top_k))
                compressor = ExtractiveCompressor()
                compressed = compressor.compress(request.query, reranked, max_chars=1500)
                final_chunks = compressed[: request.top_k]
            else:
                final_chunks = candidates[: request.top_k]

            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            note = None if final_chunks else "no relevant passages found"

            return SearchResponse(results=final_chunks, note=note, latency_ms=elapsed_ms)

        except Exception as e:
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            return SearchResponse(
                results=[], note=f"no relevant passages found ({e})", latency_ms=elapsed_ms
            )

    def ask(self, request: AskRequest) -> AskResponse:
        """Perform RAG Q&A query over a collection."""
        search_req = SearchRequest(
            query=request.query,
            collection=request.collection,
            top_k=request.top_k,
            profile=request.profile,
        )
        search_resp = self.search(search_req)

        generator = get_generator("mock")
        if not search_resp.results:
            return AskResponse(
                answer="Insufficient relevant context found to answer the query.",
                citations=[],
                note="no relevant passages found",
                insufficient_context=True,
            )

        return generator.generate(request.query, search_resp.results)
