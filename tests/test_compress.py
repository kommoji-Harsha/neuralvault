"""Unit tests for extractive context compression."""

from neuralvault.compress.compressor import ExtractiveCompressor
from neuralvault.contract import Chunk


def test_extractive_compressor_budget():
    c1 = Chunk(
        chunk_id="c1",
        text="The quick brown fox jumps over the lazy dog. Explanation sentence.",
        source="a.txt",
        location="L1",
        score=0.9,
    )

    compressor = ExtractiveCompressor()
    compressed = compressor.compress("fox jumps", [c1], max_chars=50)

    assert len(compressed) == 1
    assert len(compressed[0].text) <= 50
    assert "fox" in compressed[0].text or "jumps" in compressed[0].text
    assert compressed[0].source == "a.txt"


def test_extractive_compressor_code_retention():
    code_text = "def calculate_total():\n    return 100"
    c1 = Chunk(
        chunk_id="c2",
        text=code_text,
        source="b.py",
        location="L1",
        score=0.8,
    )

    compressor = ExtractiveCompressor()
    compressed = compressor.compress("calculate_total", [c1], max_chars=100)

    assert len(compressed) == 1
    assert "def calculate_total" in compressed[0].text
