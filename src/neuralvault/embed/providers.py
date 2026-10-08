"""Embedding providers for NeuralVault.

Includes FastEmbed (ONNX, default BAAI/bge-small-en-v1.5), LlamaCpp (GGUF),
OpenAI-compatible HTTP, and HashEmbeddingProvider (for fast, offline testing).
Enforces NEURALVAULT_OFFLINE mode safety.
"""

import hashlib
import os
from pathlib import Path
from typing import Any, List, Optional

import numpy as np

from neuralvault.config import get_neuralvault_home, guard_offline, is_offline
from neuralvault.interfaces import EmbeddingProvider


class HashEmbeddingProvider(EmbeddingProvider):
    """Deterministic hash-based embedding provider for fast offline unit testing."""

    def __init__(self, model_name: str = "hash-embedding-384", dimension: int = 384) -> None:
        self._model_name = model_name
        self._dimension = dimension

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed(self, texts: List[str]) -> List[List[float]]:
        embeddings: List[List[float]] = []
        for text in texts:
            # Hash text to uint32 seed
            digest = hashlib.sha256(text.encode("utf-8")).digest()
            seed = int.from_bytes(digest[:4], "big")
            rng = np.random.RandomState(seed)
            vec = rng.randn(self._dimension)
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec = vec / norm
            embeddings.append(vec.tolist())
        return embeddings


class FastEmbedProvider(EmbeddingProvider):
    """FastEmbed CPU ONNX provider (default model BAAI/bge-small-en-v1.5)."""

    def __init__(
        self,
        model_name: str = "BAAI/bge-small-en-v1.5",
        cache_dir: Optional[Path] = None,
        dimension: int = 384,
    ) -> None:
        self._model_name = model_name
        self._dimension = dimension
        if cache_dir is None:
            self.cache_dir = get_neuralvault_home() / "models"
        else:
            self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._model: Any = None

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def dimension(self) -> int:
        return self._dimension

    def _get_model(self) -> Any:
        if self._model is not None:
            return self._model

        # FastEmbed uses fastembed.TextEmbedding
        try:
            from fastembed import TextEmbedding
        except ImportError as e:
            raise RuntimeError(
                "fastembed package is not installed. Install via 'pip install fastembed'"
            ) from e

        # Check if offline mode is active and model files are missing
        if is_offline():
            safe_folder = self._model_name.replace("/", "--")
            model_path = self.cache_dir / f"models--{safe_folder}"
            if not model_path.exists() and not list(self.cache_dir.glob(f"*{safe_folder}*")):
                guard_offline(
                    f"download FastEmbed model '{self._model_name}'. "
                    "Run 'neuralvault models download' first when online."
                )

        os.environ["FASTEMBED_CACHE_PATH"] = str(self.cache_dir)
        self._model = TextEmbedding(model_name=self._model_name, cache_dir=str(self.cache_dir))
        return self._model

    def embed(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        model = self._get_model()
        # FastEmbed embed() returns a generator of numpy arrays
        embeddings_gen = model.embed(texts)
        return [vec.tolist() for vec in embeddings_gen]


class LlamaCppEmbeddingProvider(EmbeddingProvider):
    """GGUF embeddings via llama-cpp-python."""

    def __init__(
        self,
        model_path: Path | str,
        model_name: str = "llama-cpp-gguf",
        dimension: int = 384,
    ) -> None:
        self._model_path = Path(model_path)
        self._model_name = model_name
        self._dimension = dimension
        self._llm: Any = None

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def dimension(self) -> int:
        return self._dimension

    def _get_llm(self) -> Any:
        if self._llm is not None:
            return self._llm

        if not self._model_path.exists():
            if is_offline():
                guard_offline(f"access missing GGUF model file '{self._model_path}'")
            raise FileNotFoundError(f"GGUF model file not found: {self._model_path}")

        try:
            from llama_cpp import Llama
        except ImportError as e:
            raise RuntimeError(
                "llama-cpp-python is not installed. Install via 'pip install llama-cpp-python'"
            ) from e

        self._llm = Llama(model_path=str(self._model_path), embedding=True, verbose=False)
        return self._llm

    def embed(self, texts: List[str]) -> List[List[float]]:
        llm = self._get_llm()
        results: List[List[float]] = []
        for text in texts:
            res = llm.create_embedding(text)
            embedding_data = res["data"][0]["embedding"]
            results.append(embedding_data)
        return results


class OpenAIEmbeddingProvider(EmbeddingProvider):
    """OpenAI-compatible HTTP embeddings provider."""

    def __init__(
        self,
        base_url: str = "http://localhost:11434/v1",
        api_key: str = "ollama",
        model_name: str = "text-embedding-3-small",
        dimension: int = 1536,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self._model_name = model_name
        self._dimension = dimension

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed(self, texts: List[str]) -> List[List[float]]:
        guard_offline("call OpenAI-compatible HTTP embeddings endpoint")
        if not texts:
            return []

        import requests

        url = f"{self.base_url}/embeddings"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "input": texts,
            "model": self._model_name,
        }

        response = requests.post(url, json=payload, headers=headers, timeout=30)
        response.raise_for_status()
        data = response.json()

        results: List[List[float]] = []
        for item in sorted(data["data"], key=lambda x: x["index"]):
            results.append(item["embedding"])
        return results


def get_embedding_provider(
    provider_type: str = "fastembed",
    model_name: str = "BAAI/bge-small-en-v1.5",
    dimension: int = 384,
    **kwargs: Any,
) -> EmbeddingProvider:
    """Factory function for embedding providers."""
    if provider_type in ("hash", "test"):
        return HashEmbeddingProvider(model_name=model_name, dimension=dimension)
    elif provider_type == "fastembed":
        return FastEmbedProvider(model_name=model_name, dimension=dimension, **kwargs)
    elif provider_type == "llama_cpp":
        return LlamaCppEmbeddingProvider(model_name=model_name, dimension=dimension, **kwargs)
    elif provider_type in ("openai", "http"):
        return OpenAIEmbeddingProvider(model_name=model_name, dimension=dimension, **kwargs)
    else:
        raise ValueError(f"Unknown embedding provider type: '{provider_type}'")
