# NeuralVault Post-Retrieval, Reranking, Compression & Generation (Layers 5-8)

## 1. Post-Retrieval Processing
- **Parent Expansion**: Retrieves parent chunks (e.g. class headers or parent sections) when small fragments match.
- **MMR Deduplication (`mmr_deduplicate`)**: Eliminates near-duplicate chunks based on Jaccard text overlap to prevent context window duplication.
- **Relevance Threshold (`apply_relevance_threshold`)**: Drops candidate chunks below a minimum relevance score (default `min_score = 0.001`).

## 2. Reranking (`FastEmbedReranker`)
- Uses cross-encoder joint scoring (`BAAI/bge-reranker-base` via FastEmbed) to rescore query-chunk pairs.
- Reduces candidate pool (e.g., top-20 candidates down to top-3).
- Gracefully falls back to `PassthroughReranker` if cross-encoder is disabled or unavailable.

## 3. Extractive Context Compression (`ExtractiveCompressor`)
- Compresses retrieved context to fit a configurable character budget (default `1500` characters across all chunks).
- **Code Retention**: Retains function signatures, docstrings, and query-matching lines. Code blocks (` ``` `) are preserved intact without mid-block splits.
- **Prose Retention**: Ranks sentences by keyword overlap with query and selects top sentences within budget.
- Preserves chunk `source` and `location` attribution.

## 4. Citation Generation & Answer Synthesis (`Generator`)
- Generates answers using cited chunk IDs (e.g. `[chunk_id]`).
- Returns explicit `insufficient_context=True` and note `"no relevant passages found"` when context is empty or below relevance threshold.
- `MockGenerator` provides offline unit test generation without network calls.
- `OpenAIGenerator` connects to local Ollama / OpenAI-compatible HTTP endpoints and enforces `NEURALVAULT_OFFLINE` guards.
