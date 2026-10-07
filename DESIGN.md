> **Naming override:** package, CLI and repo are all `neuralvault` (not `ragbench`). Wherever this doc says `ragbench`, use `neuralvault`.

> **Updated: Oct 2026.** This replaces the earlier V2_RAG_DESIGN.md.
> Same rule applies: nothing here gets built until V1 is tagged.

# NeuralVault — A High-Quality RAG System
### Designed for: Personal AI Assistant V2 and any future project

---

## What this is and what it isn't

This is a standalone, offline-first retrieval library called
**NeuralVault**. It is not a chatbot, not an LLM wrapper, and not
tied to any one project. It plugs into any Python project through a
single tool function, an MCP server, or a REST API. The Personal AI
Assistant is the first consumer, but it is designed so that a future
AI interview coach, a document Q&A app, or anything else can use the
same library without modification.

The design philosophy in one sentence: **retrieval quality is
engineering, not magic** — every component is measurable, every
decision is justified by a benchmark, and nothing is assumed to "just
work" without evidence.

---

## Architecture overview

```
┌─────────────────────────────────────────────────────────┐
│                      NeuralVault                         │
│                                                         │
│  Ingest → Chunk → Embed → Store → Retrieve → Rerank    │
│                                  └─→ Compress → Cite   │
└────────────────────────┬────────────────────────────────┘
                         │
         ┌───────────────┼───────────────┐
         ▼               ▼               ▼
    @tool registry    MCP server    REST / SDK
    (assistant)     (any client)   (any app)
```

Every layer has a clean interface, swappable implementations, and its
own test suite. The assistant's planner never calls NeuralVault
directly — it calls a registered `@tool` that delegates to the library.

---

## Layer 1: Ingestion

### What gets ingested

NeuralVault ingests any of: Markdown, plain text, Python source,
HTML, PDF (via PyMuPDF), DOCX, CSV/TSV (as structured text), and
JSON/YAML (pretty-printed). New formats are added by implementing one
method on a `Loader` interface — nothing else changes.

### Incremental ingestion (the part most RAG tutorials skip)

Every file gets a SHA-256 content hash stored at ingest time. On
re-ingest, only changed and new files are processed. Deleted files
have their chunks removed. This means a 10,000-file corpus can be
kept up to date in seconds after a change, not minutes.

### Metadata extraction

Every document carries: source path, title (inferred from first
heading or filename), MIME type, file modification time, word count,
and a human-readable "collection" label. This metadata is stored
alongside chunks and surfaces in every search result.

---

## Layer 2: Chunking

This is the layer most RAG systems get wrong. Wrong chunking is worse
than no RAG — it guarantees that relevant content gets split in the
wrong place and never retrieved together.

### Strategy by content type

**Python source code** — chunked by AST, not by character count. One
chunk per function, one per class (including its docstring and
`__init__`), one per module-level docstring. A function that exceeds
the size limit is split at statement boundaries, never mid-expression.
The chunk header includes the fully-qualified name:
`src/ragbench/retrieve.py::HybridRetriever.search`. This is the
difference between finding `HybridRetriever.search` and finding a
random 500-character window that happens to contain part of it.

**Markdown and documentation** — chunked by heading hierarchy.
Every chunk carries its full heading path:
`docs/setup.md > Installation > Windows > Virtual environment`.
This path is prepended to the text used for embedding and BM25, so a
query about "Windows installation" finds the right section even if
the word "Windows" only appears in the heading, not the body.

**PDFs and DOCX** — chunked by paragraph with overlap. PDFs get
special treatment: detected section headings (by font size/weight)
are used as chunk boundaries where possible. Tables are kept intact
as a single chunk.

**Generic text** — recursive splitter: paragraph → sentence →
character, trying each boundary in order and only falling back to the
next when the current chunk exceeds the size limit.

### Why two representations per chunk

Each chunk stores two versions of its text:

- **Display text** — what gets shown to the user and sent to the LLM.
  Clean, readable, the actual source content.
- **Embedding text** — display text prepended with the heading/file
  path context. This is what gets embedded and indexed. The model
  never sees this; it exists purely to improve retrieval quality.

The difference in retrieval quality between these two approaches is
measurable and significant on code corpora. It is not optional.

---

## Layer 3: Embeddings

### Default: fastembed (ONNX, no PyTorch)

The default embedding model is `BAAI/bge-small-en-v1.5` via
fastembed. It runs on CPU via ONNX, requires no PyTorch, no CUDA
setup, and no compiler. It produces 384-dimensional vectors. This is
the right default for a project that must work on Windows with no
special hardware.

### Optional: GGUF embeddings via llama-cpp-python

If you already have llama-cpp-python installed (the assistant does),
you can use a GGUF embedding model. This reuses the existing stack
with no new dependencies.

### Optional: OpenAI-compatible endpoint

Any local server (Ollama, LM Studio, llama.cpp server) or cloud API
that speaks the OpenAI embeddings protocol works as a drop-in
replacement. One env var change, no code changes.

### Model consistency enforcement

The model name and embedding dimension are stored in the collection's
database at creation time. If you try to query a collection with a
different model, you get a clear error, not silent garbage results.
This is one of the most common failure modes in production RAG systems
and it costs nothing to prevent.

---

## Layer 4: Storage

### One SQLite file per collection

Every collection is one SQLite file. No servers, no services, no
configuration. The file can be copied, backed up, or deleted like any
other file. This fits the project's local-first philosophy.

Three tables per file:

- `documents` — one row per ingested file, with metadata and content
  hash.
- `chunks` — one row per chunk, with display text, embedding text,
  metadata, and the document ID it belongs to.
- `chunks_fts` — FTS5 virtual table over the chunk text, used for
  BM25 keyword search.
- `embeddings` — vector data, stored via `sqlite-vec` if available,
  with a numpy exact-search fallback if not.

### Graceful degradation

If `sqlite-vec` is not installed, NeuralVault falls back to numpy
cosine similarity over all stored vectors. Slower for large corpora,
but correct results. A `doctor` command reports which capabilities
are active and which are falling back, so you always know what you're
running.

---

## Layer 5: Retrieval — the most important layer

This is where the quality gap between good RAG and bad RAG lives.

### BM25 (keyword search)

SQLite FTS5 provides BM25 ranking out of the box. The tokenizer is
extended to split `snake_case` and `camelCase` identifiers — so a
query for `search_memory` also matches `search` and `memory`
independently. This is critical for code retrieval and is missing in
almost every off-the-shelf RAG tutorial.

### Vector search

Cosine similarity over the stored embeddings. Good at paraphrase
matching and conceptual similarity. Bad at exact function names and
error messages. Used as one of two inputs to hybrid retrieval, never
alone.

### Hybrid retrieval with Reciprocal Rank Fusion (RRF)

The real retrieval strategy combines BM25 and vector search using
Reciprocal Rank Fusion. RRF is simple (no weights to tune), robust,
and consistently outperforms either method alone across most corpora.
The formula: for each result, `score = sum(1 / (k + rank_in_each_list))`
where k=60 is the standard default. The combined list is sorted by
this score.

Candidate pool: 20 results from each method, merged to ~30 unique
results, then reranked down to top_k.

### Post-retrieval processing

After the initial retrieval, three optional steps improve quality:

**Parent expansion** — when a small chunk matches (e.g. a single
function signature), retrieve the parent chunk (the whole class or
section) and include it alongside the small match. This gives the
model context it can't get from the fragment alone.

**MMR deduplication** — Maximal Marginal Relevance removes near-duplicate
chunks. If two chunks from the same file section both score highly,
only the more relevant one is kept. This prevents the model's context
from being wasted on repeated information.

**Relevance threshold** — chunks below a minimum score are dropped
entirely. If nothing passes the threshold, the tool returns an explicit
"no relevant passages found" message. This is better than returning
low-quality results and letting the model hallucinate an answer from
them.

---

## Layer 6: Reranking

After hybrid retrieval produces a candidate set of ~20 results,
a cross-encoder reranker scores every (query, chunk) pair jointly.
Cross-encoders are slower than bi-encoders but significantly more
accurate because they see both texts together.

Default: fastembed's reranker support using
`BAAI/bge-reranker-base`. Small, fast on CPU, no PyTorch.

The reranker reduces the candidate set to the final top_k
(default: 3) that gets sent to the LLM. This is where most of the
quality improvement comes from — getting retrieval right up to ~20
candidates is easier than getting it right at top-3 directly.

---

## Layer 7: Context compression

Sending entire chunks to a small local model wastes context window.
Extractive compression selects the most query-relevant sentences from
each chunk:

- For code: keeps the function signature, docstring, and the lines
  most similar to the query. Code blocks are never split mid-block.
- For prose: scores each sentence by keyword overlap and semantic
  similarity to the query, keeps the top N sentences within a total
  character budget.
- Source and location are always preserved in the compressed output.

The total output budget is configurable (default: 1500 characters
across all results). This keeps the model's context clean for the
actual reasoning task.

---

## Layer 8: Citations

Every result carries: source file path, location (heading path or
line range), a relevance score, and which retrieval method(s) found
it. The LLM is instructed to cite chunk IDs in its answer.

Citations are not optional and are not retrofitted later. They are
part of the data contract from day one. This is cheap to build in
and expensive to add later.

---

## The three retrieval profiles

Three named profiles configure the pipeline for different speed/quality
tradeoffs:

```
fast:     hybrid retrieval, no rerank, no compression
          ~0.1s, good for interactive use with a fast GPU

balanced: hybrid + compression (no rerank)
          ~0.3s, best for small context windows (local models)

best:     hybrid + parent expansion + MMR + rerank + compression
          ~1-2s, best quality, suitable for batch or async use
```

The default profile is configurable per collection. For the assistant's
local model, `balanced` is the right default — the reranker adds latency
that isn't worth it at 3B-4B parameter model quality.

---

## Integration interfaces

### 1. @tool registration (for the Personal AI Assistant)

```python
@tool
def search_knowledge_base(query: str, top_k: int = 3) -> list[dict]:
    """
    Search your personal knowledge base — documents, notes, code,
    and project files you have indexed.

    Use this for: questions about your own files, code, projects,
    documents, or anything you have explicitly added to the knowledge
    base.

    Do NOT use this for: personal facts about the user (use
    search_memory), live internet information (use search_web), or
    general knowledge questions the model can answer directly.

    Returns a list of relevant passages with source, location, and
    a relevance score. If nothing relevant is found, returns an
    explicit note saying so — do not invent an answer.
    """
```

This is the only change needed to the assistant. One new tool,
registered exactly like every other tool, with a clear docstring that
tells the model exactly when to use it and when not to.

### 2. MCP server

NeuralVault runs as an MCP server with four tools: `list_collections`,
`search`, `ask`, and `get_document`. Any MCP client (Claude Desktop,
other assistants) can connect to it.

```
ragbench mcp --transport stdio   # for Claude Desktop
ragbench mcp --transport http    # for HTTP clients
```

### 3. REST API

FastAPI-based REST API with OpenAPI documentation. Localhost-only by
default (no accidental public exposure). Optional API key auth.

### 4. Python SDK

```python
from neuralvault import RagClient

# In-process (no server needed)
client = RagClient(mode="local", collection="assistant-project")
results = client.search("how does the confirmation workflow work")

# HTTP (connects to running server)
client = RagClient(mode="http", base_url="http://localhost:8042")
```

---

## Collections for this project

Three collections to start with:

**assistant-project** — the Personal AI Assistant repository itself.
Index: all Python source, all Markdown docs, README. 20 handwritten
evaluation questions such as:
- "how does the confirmation workflow work for dangerous tools"
- "what does _extract_tool_calls do"
- "how is conversation history rolled back on failure"
- "what is the difference between structured memory and RAG"
- "how are filesystem tools registered"

**personal-notes** — any Markdown notes, PDFs, books, or documents
you want the assistant to be able to answer questions about. Grows
over time as you add files.

**code-references** — technical documentation and references you
frequently consult (language specs, library docs, framework guides).

---

## Evaluation — the part that makes it trustworthy

A RAG system without an eval suite is a demo, not a system. This is
the difference between something that works once and something you
can actually rely on.

### Metrics tracked per retrieval strategy

- **Recall@k** — what fraction of gold chunks appear in the top-k
  results. The primary quality metric.
- **MRR** (Mean Reciprocal Rank) — how high the first correct result
  ranks. Matters for interactive use.
- **nDCG** — accounts for position of all relevant results.
- **Latency p50/p95** — measured on a fixed query set.

### Ablation table

Every technique is measured against the baseline (BM25 only) with all
other techniques held constant. This means you know exactly what each
piece contributes:

```
Strategy                    Recall@5    MRR     p50 latency
───────────────────────────────────────────────────────────
BM25 only (baseline)         0.XX       0.XX      XXms
Vector only                  0.XX       0.XX      XXms
Hybrid (BM25 + vector)       0.XX       0.XX      XXms
+ heading path headers       0.XX       0.XX      XXms
+ identifier tokenizer       0.XX       0.XX      XXms
+ parent expansion           0.XX       0.XX      XXms
+ MMR dedup                  0.XX       0.XX      XXms
+ reranker                   0.XX       0.XX      XXms
+ compression                0.XX       0.XX      XXms
Full pipeline (best)         0.XX       0.XX      XXms
```

The XX values get filled in by actual runs, not by claims. A CI job
runs the offline eval on every PR and fails if recall@5 drops below
the configured threshold.

### Gold data discipline

Human-written evaluation questions are in `data/qa/handwritten_*.jsonl`
and are never modified by code. Synthetic questions generated from
chunks go in `data/qa/synthetic_*.jsonl` and are clearly labeled.
Human-written ground truth is always preferred for quality gates.

---

## What comes after the baseline

These are real, valuable techniques but they belong after a working
baseline is measured, not before:

**Query expansion** — rewrite the user's query into multiple
variations before retrieval. Helps when the user's phrasing doesn't
match the document's vocabulary. Costs one LLM call per query.

**Hypothetical Document Embeddings (HyDE)** — generate a hypothetical
answer to the query, embed that, and use it for retrieval. Surprisingly
effective for questions whose answers look very different from the
questions. Costs one LLM call.

**Semantic caching** — cache retrieval results for semantically
similar queries. Measurably speeds up repeated queries on the same
corpus. Implement only after the corpus is stable.

**Feedback loop** — record which retrieved chunks the model actually
used in its answers. Use this signal to adjust retrieval weights over
time. This is the path toward a system that improves with use rather
than staying static.

**Multi-hop retrieval** — for questions that require synthesizing
across multiple documents, retrieve iteratively: first retrieval
informs a refined query for the second. Adds latency but handles
genuinely complex questions.

None of these get built until the baseline eval numbers are known.
Building optimizations before you have a baseline is how you end up
with a complicated system that isn't measurably better than the
simple one.

---

## Storage decision: why not Qdrant, Chroma, or pgvector

This comes up every time someone designs a RAG system, so it's worth
stating directly:

**Qdrant/Chroma** — excellent systems, but they're separate services.
This project is local-first and runs on a laptop. Adding a service
dependency means: remembering to start it, handling its failure modes
separately, backing up a separate data store, and explaining to anyone
running your project why they need to run Docker first. `sqlite-vec`
gives you good-enough vector search in the same file as everything
else. Switch to Qdrant if the corpus genuinely outgrows SQLite's
performance — and you'll know when that is because you'll have the
benchmark numbers to prove it.

**pgvector** — same argument, plus Postgres. Correct choice for a
multi-user production system. Wrong choice for a personal assistant
on a laptop.

The principle: don't add infrastructure to solve a problem that
doesn't exist yet at your current scale.

---

## Definition of done

NeuralVault is ready to integrate into the assistant when:

1. All three retrieval modes (BM25, vector, hybrid) are implemented
   and benchmarked on the `assistant-project` collection.
2. Hybrid + reranker demonstrably outperforms BM25 alone on the
   handwritten eval set — confirmed by numbers, not claims.
3. Every search result carries source, location, and score.
4. The `balanced` profile runs in under 500ms on the assistant's
   hardware for a typical query.
5. A regression test proves that adding NeuralVault did not change
   any existing tool's behavior.
6. The assistant starts normally if NeuralVault is not installed
   (optional dependency, graceful degradation).
7. `search_memory`, `search_knowledge_base`, and `search_web` have
   clearly non-overlapping docstrings and the eval confirms the model
   picks the right tool at least 95% of the time on a set of
   disambiguation test cases.

Until all seven are true, this file stays a design document.
