# NeuralVault Evaluation Specification (Task 8)

## Overview

NeuralVault includes an offline evaluation harness that measures retrieval quality across strategies without requiring live LLM calls.

---

## Metrics Tracked

1. **Recall@k**: Fraction of queries where at least one gold chunk/source appears in top-$k$ results (primary quality metric).
2. **MRR (Mean Reciprocal Rank)**: Mean of $1 / \text{rank}$ for the first gold result returned across queries.
3. **nDCG@k**: Normalized Discounted Cumulative Gain over top-$k$ results.
4. **Latency p50 / p95**: 50th and 95th percentile query execution latency in milliseconds.

---

## Strategy Ablation Table

Evaluated strategies:
- `bm25`: Keyword FTS search only.
- `vector`: Cosine similarity vector search only.
- `hybrid`: Reciprocal Rank Fusion (BM25 + vector).
- `hybrid+rerank`: Hybrid RRF + Cross-Encoder reranking.
- `full`: Full pipeline (Hybrid + Parent Expansion + MMR + Rerank + Compression).

Outputs are persisted to `results/results.json` and `results/report.md`.

---

## Tool Selection Disambiguation Benchmark

A 30-query disambiguation suite validates that assistant planners correctly route user queries to:
- `search_knowledge_base`: for documents, code, repositories, and project files.
- `search_memory`: for personal user facts, background, and preferences.
- `search_web`: for live internet info, news, weather, and current data.

Must pass $\ge 95\%$ accuracy threshold.

---

## CLI Execution
```bash
# Run evaluation harness
neuralvault eval --collection assistant-project --qa-file data/qa/handwritten_assistant_project.jsonl
```
