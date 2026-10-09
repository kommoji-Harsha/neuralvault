"""Cross-encoder rerankers for NeuralVault candidate chunks."""

import os
from pathlib import Path
from typing import Any, List, Optional

from neuralvault.config import get_neuralvault_home, guard_offline, is_offline
from neuralvault.contract import Chunk
from neuralvault.interfaces import Reranker


class PassthroughReranker(Reranker):
    """Passthrough reranker sorting candidates by existing score (baseline/no-op)."""

    def rerank(self, query: str, chunks: List[Chunk], top_k: int = 3) -> List[Chunk]:
        if not chunks:
            return []
        sorted_chunks = sorted(chunks, key=lambda c: c.score, reverse=True)
        return sorted_chunks[:top_k]


class FastEmbedReranker(Reranker):
    """Cross-encoder reranker using FastEmbed BAAI/bge-reranker-base."""

    def __init__(
        self,
        model_name: str = "BAAI/bge-reranker-base",
        cache_dir: Optional[Path] = None,
    ) -> None:
        self.model_name = model_name
        if cache_dir is None:
            self.cache_dir = get_neuralvault_home() / "models"
        else:
            self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._model: Any = None

    def _get_model(self) -> Any:
        if self._model is not None:
            return self._model

        try:
            from fastembed import TextCrossEncoder
        except ImportError as e:
            raise RuntimeError(
                "fastembed TextCrossEncoder is not installed or available."
            ) from e

        if is_offline():
            safe_folder = self.model_name.replace("/", "--")
            model_path = self.cache_dir / f"models--{safe_folder}"
            if not model_path.exists() and not list(self.cache_dir.glob(f"*{safe_folder}*")):
                guard_offline(f"download FastEmbed reranker model '{self.model_name}'")

        os.environ["FASTEMBED_CACHE_PATH"] = str(self.cache_dir)
        self._model = TextCrossEncoder(model_name=self.model_name, cache_dir=str(self.cache_dir))
        return self._model

    def rerank(self, query: str, chunks: List[Chunk], top_k: int = 3) -> List[Chunk]:
        if not chunks or not query.strip():
            return []

        try:
            model = self._get_model()
            # FastEmbed TextCrossEncoder.rerank(query, documents)
            docs = [c.text for c in chunks]
            scores = list(model.rerank(query, docs))

            scored_chunks: List[Chunk] = []
            for i, chunk in enumerate(chunks):
                raw_score = float(scores[i]) if i < len(scores) else chunk.score
                meta = dict(chunk.metadata)
                meta["reranked_score"] = raw_score
                c = Chunk(
                    chunk_id=chunk.chunk_id,
                    text=chunk.text,
                    source=chunk.source,
                    location=chunk.location,
                    score=raw_score,
                    embedding_text=chunk.embedding_text,
                    metadata=meta,
                )
                scored_chunks.append(c)

            scored_chunks.sort(key=lambda c: c.score, reverse=True)
            return scored_chunks[:top_k]
        except Exception:
            # Fallback to passthrough if cross encoder is unavailable or fails
            return PassthroughReranker().rerank(query, chunks, top_k=top_k)


def get_reranker(reranker_type: str = "passthrough", **kwargs: Any) -> Reranker:
    """Factory function for reranker implementations."""
    if reranker_type in ("passthrough", "none"):
        return PassthroughReranker()
    elif reranker_type in ("fastembed", "cross_encoder"):
        return FastEmbedReranker(**kwargs)
    else:
        return PassthroughReranker()
