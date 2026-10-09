"""Hybrid retriever using Reciprocal Rank Fusion (RRF) combining BM25 and vector search."""

from typing import Dict, List, Set

from neuralvault.contract import Chunk
from neuralvault.interfaces import EmbeddingProvider, Retriever
from neuralvault.retrieve.bm25 import BM25Retriever
from neuralvault.retrieve.vector import VectorRetriever
from neuralvault.store.sqlite_store import SqliteStore


class HybridRetriever(Retriever):
    """Hybrid retriever combining BM25 and vector retrieval via Reciprocal Rank Fusion (RRF)."""

    def __init__(
        self,
        store: SqliteStore,
        embedder: EmbeddingProvider,
        rrf_k: int = 60,
        candidate_k: int = 20,
    ) -> None:
        self.store = store
        self.embedder = embedder
        self.rrf_k = rrf_k
        self.candidate_k = candidate_k
        self.bm25_retriever = BM25Retriever(store)
        self.vector_retriever = VectorRetriever(store, embedder)

    def retrieve(self, query: str, top_k: int = 3) -> List[Chunk]:
        """Retrieve top_k matching chunks using hybrid RRF search.

        Formula: rrf_score = sum(1 / (rrf_k + rank)) across BM25 and Vector candidate lists.
        """
        if not query.strip():
            return []

        bm25_candidates = self.bm25_retriever.retrieve(query, top_k=self.candidate_k)
        vector_candidates = self.vector_retriever.retrieve(query, top_k=self.candidate_k)

        chunk_map: Dict[str, Chunk] = {}
        rrf_scores: Dict[str, float] = {}
        found_by: Dict[str, Set[str]] = {}
        provenance_map: Dict[str, dict] = {}

        # Process BM25 candidates
        for rank_idx, chunk in enumerate(bm25_candidates):
            cid = chunk.chunk_id
            rank = rank_idx + 1
            score = 1.0 / (self.rrf_k + rank)

            chunk_map[cid] = chunk
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + score
            found_by.setdefault(cid, set()).add("bm25")

            prov = provenance_map.setdefault(
                cid,
                {
                    "retrievers": [],
                    "bm25_score": chunk.score,
                    "bm25_rank": rank,
                },
            )
            prov["bm25_score"] = chunk.score
            prov["bm25_rank"] = rank

        # Process Vector candidates
        for rank_idx, chunk in enumerate(vector_candidates):
            cid = chunk.chunk_id
            rank = rank_idx + 1
            score = 1.0 / (self.rrf_k + rank)

            if cid not in chunk_map:
                chunk_map[cid] = chunk
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + score
            found_by.setdefault(cid, set()).add("vector")

            prov = provenance_map.setdefault(
                cid,
                {
                    "retrievers": [],
                    "vector_score": chunk.score,
                    "vector_rank": rank,
                },
            )
            prov["vector_score"] = chunk.score
            prov["vector_rank"] = rank

        # Sort candidate chunk IDs by RRF score
        sorted_cids = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)
        top_cids = sorted_cids[:top_k]

        results: List[Chunk] = []
        for cid in top_cids:
            chunk = chunk_map[cid]
            meta = dict(chunk.metadata)
            prov = provenance_map[cid]
            prov["retrievers"] = sorted(list(found_by[cid]))
            prov["rrf_score"] = rrf_scores[cid]
            meta["retriever"] = "hybrid_rrf"
            meta["provenance"] = prov

            c = Chunk(
                chunk_id=chunk.chunk_id,
                text=chunk.text,
                source=chunk.source,
                location=chunk.location,
                score=rrf_scores[cid],
                embedding_text=chunk.embedding_text,
                metadata=meta,
            )
            results.append(c)

        return results
