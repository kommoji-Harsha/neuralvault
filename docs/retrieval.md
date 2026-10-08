# NeuralVault Retrieval Specification (Layer 5)

## Overview

NeuralVault provides hybrid retrieval combining keyword search (BM25) and semantic vector search using Reciprocal Rank Fusion (RRF).

---

## 1. BM25 Keyword Search (`BM25Retriever`)
- Executed against SQLite FTS5 virtual table.
- **Identifier Tokenizer**: Splits `snake_case` (e.g. `search_memory` -> `search`, `memory`) and `camelCase` / `PascalCase` (e.g. `HybridRetriever` -> `Hybrid`, `Retriever`) so code identifiers can be retrieved by whole name or component sub-tokens.
- Critical for exact function names, variable names, and error message retrieval.

## 2. Vector Search (`VectorRetriever`)
- Embeds query text using the collection's `EmbeddingProvider`.
- Performs cosine similarity search over stored float32 embeddings (using `sqlite-vec` or numpy fallback).
- Effective for paraphrases, conceptual queries, and natural language questions.

## 3. Hybrid Retrieval with Reciprocal Rank Fusion (`HybridRetriever`)
- Combines top candidate pools (default top-20 candidates each) from BM25 and Vector search.
- **Formula**:
  $$ \text{score}(c) = \sum_{m \in \{\text{bm25}, \text{vector}\}} \frac{1}{k + \text{rank}_m(c)} $$
  where $k = 60$ is the standard RRF constant.
- Merges candidate lists and sorts chunks by `rrf_score` descending.
- **Provenance Preservation**:
  Every retrieved chunk tracks which retrieval method(s) discovered it and their raw scores in `chunk.metadata["provenance"]`:
  ```json
  {
    "retrievers": ["bm25", "vector"],
    "bm25_score": 0.0012,
    "bm25_rank": 1,
    "vector_score": 0.892,
    "vector_rank": 2,
    "rrf_score": 0.0325
  }
  ```
