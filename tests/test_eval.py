"""Unit tests for evaluation harness, metrics, and tool disambiguation."""

from neuralvault.contract import Chunk, Document
from neuralvault.eval.generator import generate_synthetic_dataset
from neuralvault.eval.harness import (
    EvaluationHarness,
    compute_mrr,
    compute_ndcg_at_k,
    compute_recall_at_k,
)
from neuralvault.eval.tool_chooser import evaluate_tool_selection_accuracy
from neuralvault.service.service import RagService
from neuralvault.store.sqlite_store import SqliteStore


def test_eval_metric_calculations():
    gold_sources = {"docs/conf.md"}
    gold_chunks = {"c1"}

    c1 = Chunk(chunk_id="c1", text="text1", source="docs/conf.md", location="L1")
    c2 = Chunk(chunk_id="c2", text="text2", source="docs/other.md", location="L2")

    # Match at rank 1
    assert compute_recall_at_k([c1, c2], gold_sources, gold_chunks, k=5) == 1.0
    assert compute_mrr([c1, c2], gold_sources, gold_chunks) == 1.0
    assert compute_ndcg_at_k([c1, c2], gold_sources, gold_chunks, k=5) == 1.0

    # Match at rank 2
    assert compute_recall_at_k([c2, c1], gold_sources, gold_chunks, k=5) == 1.0
    assert compute_mrr([c2, c1], gold_sources, gold_chunks) == 0.5

    # No match
    assert compute_recall_at_k([c2], gold_sources, gold_chunks, k=5) == 0.0
    assert compute_mrr([c2], gold_sources, gold_chunks) == 0.0


def test_tool_selection_accuracy_benchmark():
    res = evaluate_tool_selection_accuracy()
    assert res["total_cases"] == 30
    assert res["accuracy"] >= 0.95
    assert res["passes_threshold_95"] is True


def test_synthetic_qa_generation(tmp_path):
    c = Chunk(chunk_id="c1", text="def test_function(): pass", source="test.py", location="L1")
    out_file = tmp_path / "synthetic.jsonl"
    items = generate_synthetic_dataset([c], out_file)

    assert len(items) == 1
    assert items[0]["synthetic"] is True
    assert out_file.exists()


def test_evaluation_harness_benchmark(tmp_path):
    store = SqliteStore("eval_test_col", index_dir=tmp_path)
    doc = Document(
        doc_id="d1", source="docs/test.md", title="Test", hash="h1", collection="eval_test_col"
    )
    chunk = Chunk(
        chunk_id="c1",
        text="NeuralVault evaluation harness test",
        source="docs/test.md",
        location="L1",
    )
    store.add_documents_and_chunks([doc], [chunk])

    qa_file = tmp_path / "qa.jsonl"
    qa_file.write_text(
        '{"id":"q1","question":"evaluation harness",'
        '"gold_sources":["docs/test.md"],"gold_chunk_ids":["c1"]}\n'
    )

    service = RagService(index_dir=tmp_path, provider_type="hash")
    harness = EvaluationHarness(service)

    out_json = tmp_path / "results.json"
    out_report = tmp_path / "report.md"

    data, report = harness.run_ablation_benchmark(
        "eval_test_col", qa_file, out_json=out_json, out_report=out_report
    )

    assert out_json.exists()
    assert out_report.exists()
    assert "bm25" in [s["strategy"] for s in data["strategies"]]
    assert "# NeuralVault Evaluation Report" in report
