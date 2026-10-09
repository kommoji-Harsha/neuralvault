"""Personal AI Assistant @tool adapter for NeuralVault search_knowledge_base.

This module provides the search_knowledge_base tool function designed
for integration into the Personal AI Assistant's tool registry.
"""

import os
from typing import Any, Dict, List, Optional

# Try importing tool registry decorator from assistant codebase, fallback to no-op
try:
    from tools.registry import tool
except ImportError:
    def tool(func):  # type: ignore[no-redef]
        return func


_CLIENT_CACHE: Dict[str, Any] = {}


def _get_client(collection: Optional[str] = None) -> Any:
    """Lazy initialization of in-process local RagClient."""
    from neuralvault.client.client import RagClient

    col = collection or os.getenv("NEURALVAULT_COLLECTION", "assistant-project")
    if col not in _CLIENT_CACHE:
        _CLIENT_CACHE[col] = RagClient(mode="local", collection=col)
    return _CLIENT_CACHE[col]


@tool
def search_knowledge_base(query: str, top_k: int = 3) -> List[Dict[str, Any]]:
    """Search your personal knowledge base — documents, notes, code, and project files indexed.

    Use this for: questions about files, code, projects, documents in the knowledge base.

    Do NOT use this for: personal facts about user (search_memory) or live web info (search_web).

    Returns relevant passages with source, location, score, or explicit note if none found.
    """
    try:
        client = _get_client()
        resp = client.search(query=query, top_k=top_k, profile="balanced")

        if not resp.results:
            return [
                {
                    "note": resp.note or "no relevant passages found",
                    "results": [],
                }
            ]

        results = []
        total_chars = 0
        max_chars = 1500

        for chunk in resp.results:
            text_snippet = chunk.text.strip()
            if total_chars + len(text_snippet) > max_chars and results:
                break

            results.append(
                {
                    "chunk_id": chunk.chunk_id,
                    "source": chunk.source,
                    "location": chunk.location,
                    "score": round(chunk.score, 4),
                    "text": text_snippet,
                }
            )
            total_chars += len(text_snippet)

        return results if results else [{"note": "no relevant passages found", "results": []}]

    except Exception as e:
        return [{"error": f"Failed to search knowledge base: {e}"}]
