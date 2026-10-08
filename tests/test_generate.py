"""Unit tests for answer generation and citations."""

import pytest

from neuralvault.config import OfflineError
from neuralvault.contract import Chunk
from neuralvault.generate.generator import MockGenerator, OpenAIGenerator, get_generator


def test_mock_generator_citations():
    c1 = Chunk(
        chunk_id="c1",
        text="NeuralVault is an offline library.",
        source="README.md",
        location="Overview",
        score=0.8,
    )
    gen = MockGenerator()
    res = gen.generate("What is NeuralVault?", [c1])

    assert res.insufficient_context is False
    assert len(res.citations) == 1
    assert res.citations[0].chunk_id == "c1"
    assert "[c1]" in res.answer


def test_mock_generator_insufficient_context():
    gen = MockGenerator(min_score_threshold=0.5)
    c_low = Chunk(chunk_id="c1", text="low score", source="a.txt", location="L1", score=0.0001)

    res = gen.generate("Query", [c_low])
    assert res.insufficient_context is True
    assert res.note == "no relevant passages found"
    assert len(res.citations) == 0


def test_openai_generator_offline_guard(monkeypatch):
    monkeypatch.setenv("NEURALVAULT_OFFLINE", "1")
    gen = OpenAIGenerator()
    c = Chunk(chunk_id="c1", text="test", source="a.txt", location="L1", score=0.8)

    with pytest.raises(OfflineError, match="Offline mode is active"):
        gen.generate("query", [c])


def test_get_generator_factory():
    gen = get_generator("mock")
    assert isinstance(gen, MockGenerator)
