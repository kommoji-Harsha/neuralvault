"""Unit tests for cross-encoder reranking."""

from neuralvault.contract import Chunk
from neuralvault.rerank.reranker import PassthroughReranker, get_reranker


def test_passthrough_reranker():
    c1 = Chunk(chunk_id="c1", text="text 1", source="a.py", location="L1", score=0.3)
    c2 = Chunk(chunk_id="c2", text="text 2", source="a.py", location="L2", score=0.8)
    c3 = Chunk(chunk_id="c3", text="text 3", source="a.py", location="L3", score=0.6)

    reranker = PassthroughReranker()
    reranked = reranker.rerank("query", [c1, c2, c3], top_k=2)

    assert len(reranked) == 2
    assert reranked[0].chunk_id == "c2"
    assert reranked[1].chunk_id == "c3"


def test_get_reranker_factory():
    r_none = get_reranker("none")
    assert isinstance(r_none, PassthroughReranker)

    r_unknown = get_reranker("unknown_type")
    assert isinstance(r_unknown, PassthroughReranker)
