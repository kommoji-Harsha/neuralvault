# NeuralVault Ingestion & Chunking Specification (Layers 1-2)

## Layer 1: Ingestion

NeuralVault supports ingestion of multi-format document corpora into collections.

### Supported File Formats
- **Markdown (`.md`, `.markdown`)**: MIME `text/markdown`, title inferred from first H1 heading (`# Title`).
- **Plain Text (`.txt`)**: MIME `text/plain`, title inferred from filename.
- **Python Source (`.py`)**: MIME `text/x-python`, title inferred from module docstring or filepath.
- **HTML (`.html`, `.htm`)**: MIME `text/html`, title inferred from `<title>` or `<h1>` tag. Strips script/style tags and extracts clean text.
- **PDF (`.pdf`)**: MIME `application/pdf`, extracted via PyMuPDF (`fitz`).
- **DOCX (`.docx`)**: MIME `application/vnd.openxmlformats-officedocument.wordprocessingml.document`, extracted via `docx2txt`.
- **CSV/TSV (`.csv`, `.tsv`)**: MIME `text/csv`, formatted as pipe-separated table rows.
- **JSON/YAML (`.json`, `.yaml`, `.yml`)**: MIME `application/json`, formatted as pretty-printed text.

### Incremental Ingestion
1. SHA-256 binary hash is calculated for every ingested file.
2. Re-ingesting a collection checks existing hashes:
   - **Unchanged files** (matching hash) are skipped.
   - **New and modified files** (different hash) are re-parsed and chunked.
   - **Deleted files** (previously ingested files no longer present) have their document entries and chunks purged from storage.

---

## Layer 2: Chunking

NeuralVault applies content-specific chunking strategies. Each chunk produces **two text representations**:
1. `text`: clean display text shown to users and LLMs.
2. `embedding_text`: display text prepended with heading/file path context headers.

### Chunking Strategies
- **Markdown (`MarkdownChunker`)**: Split by heading hierarchy. Prepends full heading path (e.g. `docs/setup.md > Installation > Windows`) to `embedding_text`.
- **Python AST (`PythonAstChunker`)**: Parsed via `ast`. Generates distinct chunks for:
  - Module docstrings (`path::module_docstring`)
  - Class definitions and `__init__` (`path::ClassName`)
  - Methods and functions (`path::ClassName.method_name` or `path::function_name`)
  - Functions exceeding size limits are split at AST top-level statement boundaries without splitting mid-expression.
- **PDF/DOCX (`PdfDocxChunker`)**: Split by paragraph with character overlap; detects page markers and keeps tables intact as single chunks.
- **Generic Fallback (`RecursiveGenericChunker`)**: Recursive splitting across paragraph (`\n\n`), sentence, and character boundaries.
- **Fixed-Size (`FixedSizeChunker`)**: Character window with overlap for benchmark baselines.

---

## JSONL Persistence & CLI Usage

Ingested documents and chunks are stored in `indexes/<collection>/store.jsonl` (along with `documents.jsonl` and `chunks.jsonl`).

### Command Line Interface
```bash
# Ingest current directory into assistant-project collection
neuralvault ingest assistant-project

# Ingest specific file or directory
neuralvault ingest assistant-project --path README.md
```
