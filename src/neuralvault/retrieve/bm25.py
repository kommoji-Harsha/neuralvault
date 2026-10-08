"""BM25 keyword retriever with snake_case and camelCase identifier tokenizer."""

import re
from typing import List, Set

from neuralvault.contract import Chunk
from neuralvault.interfaces import Retriever
from neuralvault.store.sqlite_store import SqliteStore


def tokenize_identifiers(text: str) -> List[str]:
    """Tokenize query text, expanding snake_case and camelCase identifiers into sub-tokens."""
    if not text.strip():
        return []

    tokens: Set[str] = set()
    raw_words = re.findall(r"\w+", text)

    for word in raw_words:
        tokens.add(word)

        # Split snake_case
        if "_" in word:
            parts = [p for p in word.split("_") if p]
            for p in parts:
                tokens.add(p)

        # Split camelCase / PascalCase
        camel_parts = re.findall(r"[A-Z]?[a-z]+|[A-Z]+(?=[A-Z][a-z]|\d|\b)|[0-9]+", word)
        if len(camel_parts) > 1:
            for cp in camel_parts:
                tokens.add(cp)

    return sorted(tokens)


def expand_fts_query(query: str) -> str:
    """Expand search query into OR-connected token query for FTS5."""
    sub_tokens = tokenize_identifiers(query)
    if not sub_tokens:
        return query
    # Join tokens with OR or match as individual terms
    return " OR ".join(f'"{tok}"' for tok in sub_tokens)


class BM25Retriever(Retriever):
    """BM25 keyword retriever querying SQLite FTS5 index."""

    def __init__(self, store: SqliteStore) -> None:
        self.store = store

    def retrieve(self, query: str, top_k: int = 20) -> List[Chunk]:
        """Retrieve top_k matching chunks using BM25 search over FTS5 index."""
        if not query.strip():
            return []

        fts_query = expand_fts_query(query)
        raw_chunks = self.store.search_bm25(fts_query, top_k=top_k)

        # If expanded OR query returned no results, fallback to exact query string
        if not raw_chunks:
            raw_chunks = self.store.search_bm25(query, top_k=top_k)

        results: List[Chunk] = []
        for rank_idx, chunk in enumerate(raw_chunks):
            meta = dict(chunk.metadata)
            meta["retriever"] = "bm25"
            meta["provenance"] = {
                "retrievers": ["bm25"],
                "bm25_score": chunk.score,
                "bm25_rank": rank_idx + 1,
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
