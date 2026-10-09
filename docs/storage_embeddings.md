# NeuralVault Storage & Embeddings Specification (Layers 3-4)

## Layer 3: Embeddings

NeuralVault supports swappable embedding providers.

### Supported Providers
- **FastEmbed (`FastEmbedProvider`)**: CPU ONNX-based embedding provider using default `BAAI/bge-small-en-v1.5` (384 dimensions). Requires no PyTorch or compiler.
- **GGUF (`LlamaCppEmbeddingProvider`)**: GGUF local model embeddings via `llama-cpp-python`.
- **OpenAI-Compatible (`OpenAIEmbeddingProvider`)**: HTTP embeddings provider for Ollama, LM Studio, llama.cpp server, or OpenAI APIs.
- **Hash Embedding (`HashEmbeddingProvider`)**: Deterministic hash-based mock embedding provider for fast offline unit testing.

### Model Consistency Enforcement
- The model name and embedding dimension are recorded in `collection_metadata` when a collection database is created.
- Attempting to query or write to a collection with a mismatched model or dimension raises an explicit `ValueError`.

### Offline Mode & Model Cache
- Model files are cached under `NEURALVAULT_HOME/models` (default `~/.neuralvault/models`).
- `neuralvault models download [--model <name>]` pre-downloads embedding models for offline use.
- When `NEURALVAULT_OFFLINE=1` is set, any network attempt to download models raises an `OfflineError`.

---

## Layer 4: Storage

NeuralVault stores each collection in a single SQLite file under `indexes/<collection>.db`.

### Database Schema
- `collection_metadata`: Key-value store for `embedding_model`, `embedding_dim`, and collection configurations.
- `documents`: Stores document metadata (`doc_id`, `source`, `title`, `mime_type`, `hash`, `mtime`, `word_count`, `collection`, `metadata`).
- `chunks`: Stores chunk content and metadata (`chunk_id`, `doc_id`, `source`, `location`, `text`, `embedding_text`, `score`, `metadata`).
- `chunks_fts`: SQLite FTS5 virtual table indexing `text`, `embedding_text`, `source`, and `location` for BM25 keyword search.
- `embeddings`: Stores binary float32 vector BLOBs mapped to `chunk_id`.

### Vector Search Fallback
- Uses `sqlite-vec` extension if available.
- Gracefully degrades to exact numpy cosine similarity search over stored vector BLOBs if `sqlite-vec` is not installed.

### Diagnostic Command
```bash
neuralvault doctor
```
Reports active capabilities, vector search status (sqlite-vec vs numpy fallback), offline mode status, and discovered collections.
