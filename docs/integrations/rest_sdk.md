# NeuralVault REST API, Python SDK, and Assistant Adapter Guide

## 1. REST API

Localhost-only FastAPI server providing OpenAPI documentation.

### Starting the REST Server
```bash
uvicorn neuralvault.api.app:app --host 127.0.0.1 --port 8042
```

### Endpoints
- `GET /collections`: List collections.
- `POST /search`: Search passages (`SearchRequest` -> `SearchResponse`).
- `POST /ask`: Q&A query (`AskRequest` -> `AskResponse`).
- `GET /documents/{collection}/{doc_id}`: Get document metadata.

### API Key Authentication
Set `NEURALVAULT_API_KEY=secret_key` in environment to enforce authentication via header `X-API-Key: secret_key`.

---

## 2. Python SDK (`RagClient`)

```python
from neuralvault import RagClient

# In-process local mode (no server required)
client = RagClient(mode="local", collection="assistant-project")
results = client.search("confirmation workflow")

# HTTP mode
client = RagClient(mode="http", base_url="http://localhost:8042")
answer = client.ask("How does confirmation work?")
```

---

## 3. Tool Schema Exporter

Export `search_knowledge_base` tool definitions for LLM providers:
```bash
neuralvault tools export --format openai
neuralvault tools export --format anthropic
neuralvault tools export --format gemini
neuralvault tools export --format json-schema
```

---

## 4. Personal AI Assistant `@tool` Adapter

File: `integrations/personal_assistant/knowledge.py`

Exposes:
```python
@tool
def search_knowledge_base(query: str, top_k: int = 3) -> list[dict]:
    ...
```
- Primitive parameters only (`str`, `int`).
- Compact JSON list output (bounded to max 1500 chars).
- Gracefully returns `{"note": "no relevant passages found"}` or `{"error": ...}` dicts.
