# Jules tasks (one per session; merge each PR before the next)
Every task: read AGENTS.md and docs/DESIGN.md first; follow the DESIGN
sections named. Done = make lint + make test pass (Ubuntu and Windows).

## Task 0: Scaffold, contract, config
pyproject.toml (src layout, package neuralvault, extras: embed, vec,
gguf, api, mcp, eval, pdf, dev); package skeleton per AGENTS.md layout;
contract.py (Collection, Chunk(chunk_id,text,source,location,score),
Search/Ask request+response with `note`, Document) with LLM-ready
docstrings; abstract interfaces (Loader, Chunker, EmbeddingProvider,
Retriever, Reranker, Compressor, Generator); config.py + collections.yaml
loader + NEURALVAULT_OFFLINE guard; Makefile; .gitignore additions
(indexes/, .neuralvault/, data/raw/); GitHub Action (lint+test, ubuntu +
windows, py3.11/3.12); tests; docs/contract.md.

## Task 1: Ingestion and chunking (DESIGN Layers 1-2)
Loaders: md, txt, py, html, pdf (pymupdf), docx, csv/tsv, json/yaml.
Chunkers: Markdown heading-path, Python AST (never split a function;
qualified-name header), PDF/DOCX paragraph with overlap + tables intact,
recursive generic, fixed-size (benchmark baseline). Each chunk stores
display_text and embedding_text (context header prepended). SHA-256
incremental ingest, removal of deleted files, metadata per DESIGN.
CLI `neuralvault ingest <collection>` (jsonl store for now).

## Task 2: Storage and embeddings (Layers 3-4)
SQLite per collection: documents, chunks, chunks_fts (FTS5), embeddings
(sqlite-vec, numpy exact fallback). Providers: fastembed (default
BAAI/bge-small-en-v1.5), llama-cpp GGUF, OpenAI-compatible, hashing (tests
only). Store model+dimension per collection; refuse mismatches. Model cache
under NEURALVAULT_HOME; `neuralvault models download`; offline mode never
downloads. `neuralvault doctor` reports active capabilities and fallbacks.

## Task 3: Retrieval (Layer 5)
BM25 with snake_case/camelCase-aware tokenizer; vector; hybrid RRF (k=60,
20 candidates each). Keep per-result provenance (which retriever, raw
scores). Tests: exact function-name query (BM25 must win), paraphrase query
(vector must win).

## Task 4: Post-retrieval, rerank, compress, generate
Implement the post-retrieval steps in DESIGN: MMR/dedup, parent expansion,
relevance threshold, optional cross-encoder reranker (fastembed reranker if
available, else optional extra; 20 -> top_k; skippable), extractive
compressor within a character budget (keeps code blocks, source+location).
Generator: OpenAI-compatible HTTP, optional Gemini, or `none`; must cite
chunk ids and return insufficient_context when below threshold. Mocked LLM
tests.

## Task 5: Service layer, profiles, CLI
Wire list_collections/search/ask/get_document to the contract. Profiles:
fast (hybrid), balanced (hybrid + compression), best (hybrid + rerank +
parent expansion + compression). CLI: search, ask, collections, doctor.
Latency regression test (balanced < 500 ms on a fixture corpus).

## Task 6: MCP server
Official MCP SDK; schemas generated from contract.py; stdio and streamable
HTTP via `neuralvault mcp --transport stdio|http`. Test: start server, call
all four tools, assert equality with direct service calls.
docs/integrations/mcp.md (config locations differ per client).

## Task 7: REST, SDK, tool export, assistant adapter
FastAPI (OpenAPI, optional API key, localhost-only default).
`RagClient(mode="local"|"http")` with identical methods.
`neuralvault tools export --format openai|anthropic|gemini|json-schema`.
integrations/personal_assistant/knowledge.py: `@tool` imported from
`tools.registry`; `search_knowledge_base(query: str, top_k: int = 3)`;
primitive types only; docstring says when to use it (docs/code/projects)
and when NOT (personal facts -> search_memory, live info -> search_web);
compact output; explicit "no relevant passages" result; errors returned as
{"error": ...}; lazy local client; collection from env var. Test with
tests/fixtures/assistant_registry.py (user adds a copy of the assistant's
tools/registry.py): register the tool, check schema, execute end to end.

## Task 8: Evaluation harness
QA JSONL: question, gold_chunk_ids or gold_sources, reference_answer.
Metrics: recall@k, MRR, nDCG, latency p50/p95 (offline, no LLM);
optional LLM-judged faithfulness/citation accuracy. Strategies: bm25,
vector, hybrid, hybrid+rerank, full pipeline. Ablations per the DESIGN
table. Outputs results/report.md and results.json. Synthetic question
generator (marked synthetic). Tool-selection test set: 30 disambiguation
queries across search_memory/search_knowledge_base/search_web docstrings
(mocked chooser interface). CI job: small offline eval, fail if recall@5
below configured threshold.

## Task 9: Packaging and docs
Dockerfile + compose (API + MCP), Streamlit dashboard (benchmark table,
playground with highlighted citations), docs/offline.md, README with
mermaid architecture, reproduction steps, and a results table read from
results.json (no invented numbers), plus "what failed / lessons learned".

## Task A (later, in the ASSISTANT repo, only after V1 is tagged)
Read docs/JULES_PROJECT_BRIEF.md, V1_FROZEN_SPECIFICATION.md and
V2_RAG_DESIGN.md. Add neuralvault as an optional dependency
(requirements-rag.txt), copy the adapter to tools/knowledge.py (guarded so
the assistant starts without neuralvault), add one rule to BOTH planners'
system prompts separating search_memory / search_knowledge_base /
search_web. Tests: existing tools unchanged, new tool with mocked client,
assistant starts without neuralvault. Do not change planner architecture.
