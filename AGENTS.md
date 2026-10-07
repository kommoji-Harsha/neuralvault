# AGENTS.md: NeuralVault

## Project
Standalone, offline-first retrieval library. Ingest -> chunk -> embed ->
store -> retrieve -> rerank -> compress -> cite. Plugs into any project via
a Python SDK, @tool function, MCP server, and REST API. The authoritative
design is docs/DESIGN.md; the task list is docs/JULES_TASKS.md.

## Naming
Repo, package, import and CLI are all `neuralvault`.

## Non-negotiables
1. OFFLINE-FIRST: search works with no internet and no API key. With
   NEURALVAULT_OFFLINE=1 any network call raises. Never download models
   implicitly or at import time.
2. Windows + Linux (CI on both). Python 3.11+ (test 3.11 and 3.12).
   No PyTorch required, no compiler required. Use pathlib.
3. Minimal core deps; heavy ones are optional extras.
4. Never invent numbers. Any metric in docs comes from results/results.json.
5. Adapters (mcp, api, client, tool) are thin: no retrieval logic in them.

## Stack
SQLite (FTS5 + sqlite-vec, numpy fallback), fastembed (ONNX), optional
GGUF embeddings via llama-cpp-python, Pydantic v2, FastAPI, MCP Python
SDK, pytest, ruff.

## Layout
src/neuralvault/{ingest,chunk,embed,store,retrieve,rerank,compress,
generate,service,eval,api,mcp_server,client,cli}
integrations/personal_assistant/  tests/  data/qa/  results/  docs/
One SQLite file per collection: indexes/<collection>.db

## Tool contract
list_collections, search, ask, get_document. Pydantic models in
src/neuralvault/contract.py, reused by all adapters. Every result has
source, location, score. If nothing relevant: explicit "no relevant
passages found" note, never an empty list. Assistant-facing @tool uses
primitive params only, compact output (default 1500 chars), returns
{"error": ...} instead of raising.

## Rules
1. One task per session; stay in scope; no unrelated refactors.
2. Tests never call real LLM, embedding or network services.
3. Never edit data/qa/handwritten_*.jsonl (human gold data).
4. Never commit secrets, models, indexes, or files over 5 MB.
5. Type hints + docstrings on public functions.
6. Any claim that a technique helps must be backed by an eval run.

## Commands (run before finishing)
make install, make lint, make test

## Definition of done
Lint and tests pass on the CI matrix, acceptance criteria met, docs
updated, PR description lists what changed, how to run it, assumptions.
