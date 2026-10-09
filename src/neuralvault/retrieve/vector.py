"""Vector similarity retriever querying vector embeddings from SQLite store."""

from typing import List

from neuralvault.contract import Chunk
from neuralvault.interfaces import EmbeddingProvider, Retriever
from neuralvault.store.sqlite_store import SqliteStore


class VectorRetriever(Retriever):
    """Vector similarity retriever using an EmbeddingProvider and SqliteStore."""

    def __init__(self, store: SqliteStore, embedder: EmbeddingProvider) -> None:
        self.store = store
        self.embedder = embedder

    def retrieve(self, query: str, top_k: int = 20) -> List[Chunk]:
        """Retrieve top_k matching chunks using vector cosine similarity search."""
        if not query.strip():
            return []

        query_vecs = self.embedder.embed([query])
        if not query_vecs or not query_vecs[0]:
            return []

        query_vec = query_vecs[0]
        raw_chunks = self.store.search_vector(query_vec, top_k=top_k)

        results: List[Chunk] = []
        for rank_idx, chunk in enumerate(raw_chunks):
            meta = dict(chunk.metadata)
            meta["retriever"] = "vector"
            meta["provenance"] = {
                "retrievers": ["vector"],
                "vector_score": chunk.score,
                "vector_rank": rank_idx + 1,
            }
            c = Chunk(
                chunk_id=chunk.chunk_id,
                text=chunk.text,
                source=chunk.source,
                location=chunk.location,
                score=chunk.score,
                embedding_text=chunk.embedding_text,
                metadata=meta,
            )
            results.append(c)

        return results
