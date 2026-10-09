# NeuralVault

Offline-first, measurable retrieval library in Python 3.11+ for AI applications. Ingest $\rightarrow$ Chunk $\rightarrow$ Embed $\rightarrow$ Store $\rightarrow$ Retrieve $\rightarrow$ Rerank $\rightarrow$ Compress $\rightarrow$ Cite.

Plugs into any project via Python SDK (`RagClient`), assistant `@tool` adapter, MCP server (`neuralvault mcp`), or REST API.

---

## Architecture Overview

```mermaid
graph TD
    A[Ingest & Loader] --> B[Content Chunking]
    B --> C[Embedding Provider]
    C --> D[(SQLite Store: FTS5 + Vectors)]
    D --> E[Retrieval: BM25 + Vector RRF]
    E --> F[Post-Retrieval: Parent Expansion & MMR]
    F --> G[Cross-Encoder Reranker]
    G --> H[Extractive Context Compressor]
    H --> I[Generator & Citation Synthesis]

    I --> J[@tool Adapter]
    I --> K[MCP Server]
    I --> L[REST API & SDK]
```

---

## Quickstart & Reproduction Steps

```bash
# 1. Install NeuralVault in editable mode with development dependencies
make install

# 2. Run unit tests and linting
make lint
make test

# 3. Ingest current repository into 'my-project' collection
neuralvault ingest my-project --path .

# 4. Search and Ask via CLI
neuralvault search my-project "How does hybrid retrieval work?"
neuralvault ask my-project "What embedding model is used by default?"

# 5. Run offline evaluation harness
neuralvault eval --collection my-project --qa-file data/qa/handwritten_my_project.jsonl

# 6. Start MCP server over stdio or HTTP
neuralvault mcp --transport stdio
```

---

## Evaluation Benchmark Results (`results/results.json`)

Metrics below are produced by running `neuralvault eval` on the `my-project` collection using 10 human gold QA queries:

| Strategy | Recall@5 | MRR | nDCG@5 | Latency p50 (ms) | Latency p95 (ms) |
| --- | --- | --- | --- | --- | --- |
| **bm25** | 0.8000 | 0.5417 | 1.1050 | 14.78 | 23.28 |
| **vector** | 0.8000 | 0.5417 | 1.1050 | 15.39 | 17.18 |
| **hybrid** | 0.8000 | 0.5417 | 1.1050 | 15.09 | 17.22 |
| **hybrid+rerank** | 0.8000 | 0.5167 | 1.0588 | 87.55 | 118.43 |
| **full** | 0.8000 | 0.5167 | 1.0588 | 88.78 | 118.42 |

*Note: All metrics come directly from `results/results.json`.*

---

## What Failed / Lessons Learned

1. **Implicit Model Downloads Break Offline Guards**:
   - *Lesson*: Initial implementations of FastEmbed and HuggingFace tokenizers attempt to download model files implicitly upon class initialization. Wrapping initialization inside explicit `is_offline()` checks with `NEURALVAULT_OFFLINE=1` guards prevents unexpected network access in air-gapped environments.

2. **AST Split Boundaries on Syntax Errors**:
   - *Lesson*: Parsing Python code with `ast.parse()` fails when encountering invalid syntax or partial snippets. Implementing a fallback to recursive paragraph/sentence chunking (`RecursiveGenericChunker`) ensures ingestion never crashes on malformed files.

3. **Identifier Tokenization for Code Retrieval**:
   - *Lesson*: Standard word tokenizers treat `search_memory` or `HybridRetriever` as opaque single tokens, missing exact queries for `search` or `memory`. Splitting `snake_case` and `camelCase` identifiers in BM25 tokenization dramatically improves code function retrieval precision.

4. **Synchronizing Ingest Engine with Search Store**:
   - *Lesson*: Storing ingestion outputs solely in JSONL files keeps `IngestEngine` lightweight, but leaves search stores out of sync. Automatically syncing newly processed chunks and vectors into `SqliteStore` during `IngestEngine.ingest_path()` ensures ingested collections are immediately searchable.
