"""NeuralVault Command Line Interface (CLI)."""

import argparse
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
from neuralvault.ingest.engine import IngestEngine


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

    # Initializing downloads the model files into cache_dir
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

    elif args.command == "doctor":
        run_doctor()

    elif args.command == "models":
        if args.models_command == "download":
            download_model(args.model)
        else:
            parser.print_help()


if __name__ == "__main__":
    main()
