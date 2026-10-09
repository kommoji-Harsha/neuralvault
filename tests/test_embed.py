"""Unit tests for embedding providers."""

import pytest

from neuralvault.config import OfflineError
from neuralvault.embed.providers import (
    FastEmbedProvider,
    HashEmbeddingProvider,
    OpenAIEmbeddingProvider,
    get_embedding_provider,
)


def test_hash_embedding_provider():
    provider = HashEmbeddingProvider(dimension=128)
    assert provider.model_name == "hash-embedding-384"
    assert provider.dimension == 128

    vecs = provider.embed(["test sentence 1", "test sentence 2"])
    assert len(vecs) == 2
    assert len(vecs[0]) == 128
    assert len(vecs[1]) == 128

    # Deterministic output test
    vecs2 = provider.embed(["test sentence 1"])
    assert vecs[0] == vecs2[0]


def test_get_embedding_provider_factory():
    hash_p = get_embedding_provider("hash", dimension=64)
    assert isinstance(hash_p, HashEmbeddingProvider)
    assert hash_p.dimension == 64

    fast_p = get_embedding_provider("fastembed")
    assert isinstance(fast_p, FastEmbedProvider)

    with pytest.raises(ValueError, match="Unknown embedding provider type"):
        get_embedding_provider("invalid_provider")


def test_openai_embedding_provider_offline_guard(monkeypatch):
    monkeypatch.setenv("NEURALVAULT_OFFLINE", "1")
    provider = OpenAIEmbeddingProvider()
    with pytest.raises(OfflineError, match="Offline mode is active"):
        provider.embed(["test"])
