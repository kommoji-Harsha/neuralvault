"""NeuralVault Command Line Interface (CLI)."""

import argparse
import json
import os
import sqlite3
import sys
from pathlib import Path

from neuralvault.config import (
    get_indexes_dir,
    get_neuralvault_home,
    guard_offline,
    is_offline,
)
from neuralvault.contract import AskRequest, SearchRequest
from neuralvault.ingest.engine import IngestEngine
from neuralvault.service.service import RagService

TOOL_DOCSTRING = (
    "Search your personal knowledge base — documents, notes, code, "
    "and project files you have indexed.\n\n"
    "Use this for: questions about your own files, code, projects, "
    "documents, or anything you have explicitly added to the knowledge base.\n\n"
    "Do NOT use this for: personal facts about the user (use search_memory), "
    "live internet information (use search_web), or general knowledge questions "
    "the model can answer directly.\n\n"
    "Returns a list of relevant passages with source, location, and a relevance score. "
    "If nothing relevant is found, returns an explicit note saying so — do not invent an answer."
)


def export_tool_schema(fmt: str = "json-schema") -> str:
    """Export search_knowledge_base tool schema for LLM providers."""
    param_schema = {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Search query text."},
            "top_k": {
                "type": "integer",
                "default": 3,
                "description": "Number of relevant passages to retrieve.",
            },
        },
        "required": ["query"],
    }

    if fmt == "openai":
        schema = {
            "type": "function",
            "function": {
                "name": "search_knowledge_base",
                "description": TOOL_DOCSTRING,
                "parameters": param_schema,
            },
        }
    elif fmt == "anthropic":
        schema = {
            "name": "search_knowledge_base",
            "description": TOOL_DOCSTRING,
            "input_schema": param_schema,
        }
    elif fmt == "gemini":
        schema = {
            "function_declarations": [
                {
                    "name": "search_knowledge_base",
                    "description": TOOL_DOCSTRING,
                    "parameters": param_schema,
                }
            ]
        }
    else:  # json-schema
        schema = {
            "title": "search_knowledge_base",
            "description": TOOL_DOCSTRING,
            "type": "object",
            "properties": param_schema["properties"],
            "required": param_schema["required"],
        }

    return json.dumps(schema, indent=2)


def build_parser() -> argparse.ArgumentParser:
    """Build NeuralVault CLI argument parser."""
    parser = argparse.ArgumentParser(
        prog="neuralvault",
        description="NeuralVault CLI: Offline-first retrieval library.",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Ingest command
    ingest_parser = subparsers.add_parser(
        "ingest",
        help="Ingest a file or directory into a collection.",
    )
    ingest_parser.add_argument(
        "collection",
        type=str,
        help="Target collection name.",
    )
    ingest_parser.add_argument(
        "--path",
        "-p",
        type=str,
        default=".",
        help="Path to file or directory to ingest (defaults to current directory).",
    )
    ingest_parser.add_argument(
        "--index-dir",
        type=str,
        default=None,
        help="Custom indexes directory.",
    )

    # Search command
    search_parser = subparsers.add_parser(
        "search",
        help="Search a collection for relevant passages.",
    )
    search_parser.add_argument("collection", type=str, help="Collection name.")
    search_parser.add_argument("query", type=str, help="Search query text.")
    search_parser.add_argument("--top-k", "-k", type=int, default=3, help="Number of results.")
    search_parser.add_argument(
        "--profile",
        type=str,
        default="balanced",
        choices=["fast", "balanced", "best"],
        help="Retrieval profile.",
    )

    # Ask command
    ask_parser = subparsers.add_parser(
        "ask",
        help="Ask a question over a collection.",
    )
    ask_parser.add_argument("collection", type=str, help="Collection name.")
    ask_parser.add_argument("query", type=str, help="Question text.")
    ask_parser.add_argument("--top-k", "-k", type=int, default=3, help="Context chunk count.")
    ask_parser.add_argument(
        "--profile",
        type=str,
        default="balanced",
        choices=["fast", "balanced", "best"],
        help="Retrieval profile.",
    )

    # Collections command
    subparsers.add_parser(
        "collections",
        help="List available document collections.",
    )

    # Tools subparser
    tools_parser = subparsers.add_parser(
        "tools",
        help="Manage and export tool schemas for assistant planners.",
    )
    tools_sub = tools_parser.add_subparsers(dest="tools_command", help="Tools subcommands")
    export_parser = tools_sub.add_parser("export", help="Export search_knowledge_base schema.")
    export_parser.add_argument(
        "--format",
        type=str,
        default="json-schema",
        choices=["openai", "anthropic", "gemini", "json-schema"],
        help="Schema export format.",
    )

    # MCP command
    mcp_parser = subparsers.add_parser(
        "mcp",
        help="Start NeuralVault Model Context Protocol (MCP) server.",
    )
    mcp_parser.add_argument(
        "--transport",
        type=str,
        default="stdio",
        choices=["stdio", "http"],
        help="MCP transport mode (stdio or http).",
    )
    mcp_parser.add_argument(
        "--port",
        type=int,
        default=8042,
        help="HTTP port for MCP server (default: 8042).",
    )

    # Doctor command
    subparsers.add_parser(
        "doctor",
        help="Report active NeuralVault capabilities, fallbacks, and offline status.",
    )

    # Models subparser
    models_parser = subparsers.add_parser(
        "models",
        help="Manage local embedding models.",
    )
    models_sub = models_parser.add_subparsers(dest="models_command", help="Model subcommands")
    download_parser = models_sub.add_parser(
        "download", help="Download embedding model for offline use."
    )
    download_parser.add_argument(
        "--model",
        type=str,
        default="BAAI/bge-small-en-v1.5",
        help="Model name to download (default: BAAI/bge-small-en-v1.5).",
    )

    return parser


def run_doctor() -> None:
    """Run system diagnostic report."""
    print("=== NeuralVault Doctor Diagnostic Report ===")
    nv_home = get_neuralvault_home()
    indexes_dir = get_indexes_dir()
    offline = is_offline()

    print(f"NEURALVAULT_HOME:    {nv_home}")
    print(f"Indexes Directory:   {indexes_dir}")
    print(f"Offline Mode:        {'ENABLED (NEURALVAULT_OFFLINE=1)' if offline else 'Disabled'}")

    # Check sqlite-vec capability
    sqlite_vec_active = False
    try:
        import sqlite_vec

        db = sqlite3.connect(":memory:")
        db.enable_load_extension(True)
        sqlite_vec.load(db)
        sqlite_vec_active = True
    except Exception:
        sqlite_vec_active = False

    vec_msg = (
        "sqlite-vec ACTIVE"
        if sqlite_vec_active
        else "Numpy Cosine Fallback (sqlite-vec not installed)"
    )
    print(f"Vector Search:       {vec_msg}")

    # Check fastembed
    try:
        import fastembed  # noqa: F401

        fastembed_status = "Available"
    except ImportError:
        fastembed_status = "Not installed"
    print(f"FastEmbed Support:   {fastembed_status}")

    # Check PDF/DOCX dependencies
    try:
        import pymupdf  # noqa: F401

        pdf_status = "Available (pymupdf)"
    except ImportError:
        pdf_status = "Not installed"

    try:
        import docx2txt  # noqa: F401

        docx_status = "Available (docx2txt)"
    except ImportError:
        docx_status = "Not installed"

    print(f"PDF Support:         {pdf_status}")
    print(f"DOCX Support:        {docx_status}")

    # List collections in indexes directory
    collections = []
    if indexes_dir.exists():
        for p in indexes_dir.iterdir():
            if p.suffix == ".db":
                collections.append(p.stem)
            elif p.is_dir() and (p / "store.jsonl").exists():
                collections.append(p.name)

    collections = sorted(set(collections))
    print(f"Collections Found:   {', '.join(collections) if collections else 'None'}")
    print("============================================")


def download_model(model_name: str) -> None:
    """Download embedding model files into NEURALVAULT_HOME/models."""
    guard_offline(f"download embedding model '{model_name}'")
    print(f"Downloading FastEmbed model '{model_name}'...")
    cache_dir = get_neuralvault_home() / "models"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ["FASTEMBED_CACHE_PATH"] = str(cache_dir)

    from fastembed import TextEmbedding

    TextEmbedding(model_name=model_name, cache_dir=str(cache_dir))
    print(f"Model '{model_name}' downloaded successfully to {cache_dir}.")


def main() -> None:
    """CLI entrypoint."""
    parser = build_parser()
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(0)

    if args.command == "ingest":
        target = Path(args.path)
        index_dir = Path(args.index_dir) if args.index_dir else None
        engine = IngestEngine(collection=args.collection, index_dir=index_dir)

        print(f"Ingesting '{target}' into collection '{args.collection}'...")
        stats = engine.ingest_path(target)

        print(
            f"Ingestion complete:\n"
            f"  Collection: {stats['collection']}\n"
            f"  Processed:  {stats['processed']} file(s)\n"
            f"  Skipped:    {stats['skipped']} unchanged file(s)\n"
            f"  Deleted:    {stats['deleted']} removed file(s)\n"
            f"  Total Docs: {stats['total_documents']}\n"
            f"  Total Chunks: {stats['total_chunks']}"
        )

    elif args.command == "search":
        service = RagService()
        req = SearchRequest(
            query=args.query,
            collection=args.collection,
            top_k=args.top_k,
            profile=args.profile,
        )
        res = service.search(req)

        print(f"Search results for '{args.query}' in collection '{args.collection}':")
        if res.note:
            print(f"Note: {res.note}")

        for i, chunk in enumerate(res.results, 1):
            print(
                f"\n[{i}] ID: {chunk.chunk_id} | Score: {chunk.score:.4f}\n"
                f"    Source: {chunk.source}\n"
                f"    Location: {chunk.location}\n"
                f"    Text: {chunk.text.strip()}"
            )
        print(f"\nLatency: {res.latency_ms:.2f} ms")

    elif args.command == "ask":
        service = RagService()
        req = AskRequest(
            query=args.query,
            collection=args.collection,
            top_k=args.top_k,
            profile=args.profile,
        )
        res = service.ask(req)

        print(f"Q&A Answer for '{args.query}':")
        if res.note:
            print(f"Note: {res.note}")
        print(f"\n{res.answer}\n")
        print(f"Citations count: {len(res.citations)}")

    elif args.command == "collections":
        service = RagService()
        cols = service.list_collections()
        print("Available Collections:")
        for c in cols:
            print(
                f"- {c.name}: {c.description or 'No description'} "
                f"({c.document_count} docs, {c.chunk_count} chunks, model: {c.embedding_model})"
            )

    elif args.command == "tools":
        if args.tools_command == "export":
            print(export_tool_schema(fmt=args.format))
        else:
            parser.print_help()

    elif args.command == "mcp":
        from neuralvault.mcp_server.server import run_http_server, run_stdio_server

        if args.transport == "stdio":
            run_stdio_server()
        else:
            run_http_server(port=args.port)

    elif args.command == "doctor":
        run_doctor()

    elif args.command == "models":
        if args.models_command == "download":
            download_model(args.model)
        else:
            parser.print_help()


if __name__ == "__main__":
    main()
