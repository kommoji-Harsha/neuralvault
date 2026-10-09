# NeuralVault Service Layer & CLI Specification

## Overview

The Service layer (`RagService`) exposes high-level Python methods and CLI commands for collection discovery, passage search, Q&A generation, and document metadata retrieval.

---

## Service API (`neuralvault.service.service.RagService`)

### `list_collections() -> list[Collection]`
Lists all configured and discovered document collections with document and chunk counts.

### `get_document(collection: str, doc_id_or_source: str) -> Document | None`
Retrieves Document metadata by unique document ID or source file path.

### `search(request: SearchRequest) -> SearchResponse`
Executes retrieval using one of three named profiles:
- **`fast`**: Hybrid retrieval (BM25 + vector RRF) with no cross-encoder reranking or context compression (~0.1s latency).
- **`balanced`**: Hybrid retrieval + extractive context compression (~0.3s latency, recommended default for local models).
- **`best`**: Hybrid retrieval + parent chunk expansion + MMR deduplication + cross-encoder reranking + extractive context compression (~1-2s latency, best retrieval quality).

Returns `SearchResponse` containing retrieved `Chunk` objects, latency in `latency_ms`, and `note` (set to `"no relevant passages found"` when no passages pass threshold).

### `ask(request: AskRequest) -> AskResponse`
Executes search retrieval and synthesizes an answer using cited chunk IDs. Returns `AskResponse` with `answer`, `citations`, and `insufficient_context` flag.

---

## Command Line Interface (CLI)

```bash
# List all collections
neuralvault collections

# Search a collection
neuralvault search assistant-project "confirmation workflow" --profile balanced --top-k 3

# Ask a question over a collection
neuralvault ask assistant-project "How does tool confirmation work?" --profile balanced

# Ingest a directory or file
neuralvault ingest assistant-project --path README.md

# Run system diagnostic report
neuralvault doctor

# Download embedding model for offline use
neuralvault models download --model BAAI/bge-small-en-v1.5
```
