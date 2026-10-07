"""NeuralVault Command Line Interface (CLI)."""

import argparse
import sys
from pathlib import Path

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

    return parser


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


if __name__ == "__main__":
    main()
