"""Tests for NeuralVault interfaces."""

from pathlib import Path

import pytest

from neuralvault.contract import Chunk, Document
from neuralvault.interfaces import (
    Chunker,
    EmbeddingProvider,
    Loader,
)


class DummyLoader(Loader):
    def load(self, filepath: Path) -> tuple[Document, str]:
        doc = Document(
            doc_id="d1",
            source=str(filepath),
            title="Dummy",
            hash="123",
            collection="test",
        )
        return doc, "dummy content"


class DummyChunker(Chunker):
    def chunk(self, doc: Document, text: str) -> list[Chunk]:
        return [
            Chunk(
                chunk_id="c1",
                text=text,
                source=doc.source,
                location="0-10",
                score=1.0,
            )
        ]


class DummyEmbeddingProvider(EmbeddingProvider):
    @property
    def model_name(self) -> str:
        return "dummy-model"

    @property
    def dimension(self) -> int:
        return 4

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[0.1, 0.2, 0.3, 0.4] for _ in texts]


def test_interfaces_instantiation():
    loader = DummyLoader()
    doc, content = loader.load(Path("test.txt"))
    assert doc.doc_id == "d1"
    assert content == "dummy content"

    chunker = DummyChunker()
    chunks = chunker.chunk(doc, content)
    assert len(chunks) == 1
    assert chunks[0].chunk_id == "c1"

    embedder = DummyEmbeddingProvider()
    assert embedder.model_name == "dummy-model"
    assert embedder.dimension == 4
    vecs = embedder.embed(["hello"])
    assert len(vecs) == 1
    assert len(vecs[0]) == 4


def test_unimplemented_interface_raises():
    class UnimplementedLoader(Loader):
        pass

    with pytest.raises(TypeError):
        UnimplementedLoader()  # type: ignore[abstract]
