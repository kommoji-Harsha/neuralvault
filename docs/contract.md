# NeuralVault Contract and Interfaces Specification

## Overview

NeuralVault provides a unified data contract and interface hierarchy used across all layers (ingest, chunk, embed, store, retrieve, rerank, compress, generate) and integration adapters (`@tool` registry, MCP server, REST API, Python SDK).

---

## Data Models (`neuralvault.contract`)

### `Chunk`
Represents a passage/segment of content retrieved from a collection.

- **`chunk_id`** (`str`): Unique identifier for the chunk.
- **`text`** (`str`): Display text content shown to users and LLMs.
- **`source`** (`str`): Source path or identifier (e.g., `src/neuralvault/retrieve.py`).
- **`location`** (`str`): Location context (e.g. heading path `docs/setup.md > Installation` or line numbers).
- **`score`** (`float`): Relevance score (default `0.0`).
- **`embedding_text`** (`Optional[str]`): Text representation prepended with context headers used for embedding/indexing.
- **`metadata`** (`dict`): Additional key-value metadata.

### `Document`
Metadata for an ingested file.

- **`doc_id`** (`str`): Unique document identifier.
- **`source`** (`str`): File path or origin identifier.
- **`title`** (`str`): Inferred or explicit document title.
- **`mime_type`** (`str`): Document MIME type (default `text/plain`).
- **`hash`** (`str`): SHA-256 hash of document content for incremental ingestion.
- **`mtime`** (`float`): File modification timestamp.
- **`word_count`** (`int`): Total word count.
- **`collection`** (`str`): Name of the collection.
- **`metadata`** (`dict`): Extra metadata key-value pairs.

### `Collection`
Metadata and configuration for a database collection.

- **`name`** (`str`): Collection name.
- **`description`** (`str`): Collection description.
- **`embedding_model`** (`str`): Model name (default `BAAI/bge-small-en-v1.5`).
- **`embedding_dim`** (`int`): Vector dimension (default `384`).
- **`document_count`** (`int`): Count of ingested documents.
- **`chunk_count`** (`int`): Count of total chunks.
- **`created_at`** (`str`): ISO creation timestamp.
- **`updated_at`** (`str`): ISO update timestamp.

### `SearchRequest` & `SearchResponse`
Contract for search operation.

- **`SearchRequest`**: `query`, `collection`, `top_k` (default 3), `profile` ('fast', 'balanced', 'best'), `filters`.
- **`SearchResponse`**: `results` (`list[Chunk]`), `note` (`Optional[str]`), `latency_ms` (`float`).
  - *Note:* If no relevant passages are found, `results` is empty or below threshold, and `note` explicitly states `"no relevant passages found"`.

### `AskRequest` & `AskResponse`
Contract for RAG question answering.

- **`AskRequest`**: `query`, `collection`, `top_k` (default 3), `profile`.
- **`AskResponse`**: `answer` (`str`), `citations` (`list[Chunk]`), `note` (`Optional[str]`), `insufficient_context` (`bool`).

---

## Abstract Interfaces (`neuralvault.interfaces`)

- **`Loader`**: `load(filepath: Path) -> Tuple[Document, str]`
- **`Chunker`**: `chunk(doc: Document, text: str) -> List[Chunk]`
- **`EmbeddingProvider`**: `embed(texts: List[str]) -> List[List[float]]`, `model_name`, `dimension`
- **`Retriever`**: `retrieve(query: str, top_k: int) -> List[Chunk]`
- **`Reranker`**: `rerank(query: str, chunks: List[Chunk], top_k: int) -> List[Chunk]`
- **`Compressor`**: `compress(query: str, chunks: List[Chunk], max_chars: int) -> List[Chunk]`
- **`Generator`**: `generate(query: str, chunks: List[Chunk]) -> AskResponse`

---

## Configuration & Offline Guard (`neuralvault.config`)

- **`is_offline() -> bool`**: Returns whether `NEURALVAULT_OFFLINE=1` is enabled.
- **`guard_offline(action: str)`**: Raises `OfflineError` if offline mode is enabled when a network call is attempted.
- **`load_collections_config(config_path)`**: Parses `collections.yaml`.
