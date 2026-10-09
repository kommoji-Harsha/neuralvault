"""Offline evaluation harness computing Recall@k, MRR, nDCG, and Latency p50/p95."""

import json
import math
import time
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

import numpy as np

from neuralvault.contract import SearchRequest
from neuralvault.service.service import RagService


def is_chunk_relevant(
    chunk_source: str,
    chunk_id: str,
    gold_sources: Set[str],
    gold_chunk_ids: Set[str],
) -> bool:
    """Check if a chunk matches gold sources or gold chunk IDs."""
    if chunk_id in gold_chunk_ids:
        return True
    for gs in gold_sources:
        if gs in chunk_source or chunk_source.endswith(gs):
            return True
    return False


def compute_recall_at_k(
    results: List[Any],
    gold_sources: Set[str],
    gold_chunk_ids: Set[str],
    k: int = 5,
) -> float:
    """Compute Recall@k for a query result list."""
    top_results = results[:k]
    for c in top_results:
        src = getattr(c, "source", "")
        cid = getattr(c, "chunk_id", "")
        if is_chunk_relevant(src, cid, gold_sources, gold_chunk_ids):
            return 1.0
    return 0.0


def compute_mrr(results: List[Any], gold_sources: Set[str], gold_chunk_ids: Set[str]) -> float:
    """Compute Mean Reciprocal Rank (MRR) for a query result list."""
    for rank_idx, c in enumerate(results, 1):
        src = getattr(c, "source", "")
        cid = getattr(c, "chunk_id", "")
        if is_chunk_relevant(src, cid, gold_sources, gold_chunk_ids):
            return 1.0 / rank_idx
    return 0.0


def compute_ndcg_at_k(
    results: List[Any],
    gold_sources: Set[str],
    gold_chunk_ids: Set[str],
    k: int = 5,
) -> float:
    """Compute nDCG@k for binary relevance."""
    top_results = results[:k]
    dcg = 0.0
    for idx, c in enumerate(top_results, 1):
        src = getattr(c, "source", "")
        cid = getattr(c, "chunk_id", "")
        rel = 1.0 if is_chunk_relevant(src, cid, gold_sources, gold_chunk_ids) else 0.0
        dcg += rel / math.log2(idx + 1)

    idcg = 1.0 / math.log2(2)
    return dcg / idcg if idcg > 0 else 0.0


class EvaluationHarness:
    """Offline evaluation harness for NeuralVault retrieval strategies."""

    def __init__(self, service: RagService) -> None:
        self.service = service

    def evaluate_qa_set(
        self,
        collection: str,
        qa_items: List[Dict[str, Any]],
        strategy: str = "hybrid",
        top_k: int = 5,
    ) -> Dict[str, Any]:
        """Evaluate a strategy over a set of QA items."""
        recalls_5: List[float] = []
        mrrs: List[float] = []
        ndcgs_5: List[float] = []
        latencies_ms: List[float] = []

        profile_map = {
            "bm25": "fast",
            "vector": "fast",
            "hybrid": "fast",
            "hybrid+rerank": "best",
            "full": "best",
        }
        profile = profile_map.get(strategy, "balanced")

        for item in qa_items:
            q = item["question"]
            gold_sources = set(item.get("gold_sources", []))
            gold_chunk_ids = set(item.get("gold_chunk_ids", []))

            req = SearchRequest(query=q, collection=collection, top_k=top_k, profile=profile)

            t0 = time.perf_counter()
            resp = self.service.search(req)
            t1 = time.perf_counter()

            lat_ms = (t1 - t0) * 1000.0
            latencies_ms.append(lat_ms)

            rec = compute_recall_at_k(resp.results, gold_sources, gold_chunk_ids, k=5)
            mrr = compute_mrr(resp.results, gold_sources, gold_chunk_ids)
            ndcg = compute_ndcg_at_k(resp.results, gold_sources, gold_chunk_ids, k=5)

            recalls_5.append(rec)
            mrrs.append(mrr)
            ndcgs_5.append(ndcg)

        p50 = float(np.percentile(latencies_ms, 50)) if latencies_ms else 0.0
        p95 = float(np.percentile(latencies_ms, 95)) if latencies_ms else 0.0

        return {
            "strategy": strategy,
            "recall_at_5": float(np.mean(recalls_5)) if recalls_5 else 0.0,
            "mrr": float(np.mean(mrrs)) if mrrs else 0.0,
            "ndcg_at_5": float(np.mean(ndcgs_5)) if ndcgs_5 else 0.0,
            "latency_p50_ms": p50,
            "latency_p95_ms": p95,
            "sample_count": len(qa_items),
        }

    def run_ablation_benchmark(
        self,
        collection: str,
        qa_file: Path | str,
        out_json: Path | str = "results/results.json",
        out_report: Path | str = "results/report.md",
    ) -> Tuple[Dict[str, Any], str]:
        """Run ablation benchmark over strategies and export results.json and report.md."""
        qa_file = Path(qa_file)
        out_json = Path(out_json)
        out_report = Path(out_report)

        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_report.parent.mkdir(parents=True, exist_ok=True)

        qa_items: List[Dict[str, Any]] = []
        with open(qa_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    qa_items.append(json.loads(line))

        strategies = ["bm25", "vector", "hybrid", "hybrid+rerank", "full"]
        strategy_results: List[Dict[str, Any]] = []

        for strat in strategies:
            res = self.evaluate_qa_set(collection, qa_items, strategy=strat, top_k=5)
            strategy_results.append(res)

        benchmark_data = {
            "collection": collection,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "strategies": strategy_results,
        }

        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(benchmark_data, f, indent=2)

        report_lines = [
            f"# NeuralVault Evaluation Report ({collection})",
            "",
            "| Strategy | Recall@5 | MRR | nDCG@5 | Latency p50 (ms) | Latency p95 (ms) |",
            "| --- | --- | --- | --- | --- | --- |",
        ]
        for s in strategy_results:
            row = (
                f"| {s['strategy']} | {s['recall_at_5']:.4f} | {s['mrr']:.4f} | "
                f"{s['ndcg_at_5']:.4f} | {s['latency_p50_ms']:.2f} | {s['latency_p95_ms']:.2f} |"
            )
            report_lines.append(row)

        report_md = "\n".join(report_lines)
        with open(out_report, "w", encoding="utf-8") as f:
            f.write(report_md + "\n")

        return benchmark_data, report_md
