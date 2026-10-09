"""Incremental ingestion engine and JSONL store for NeuralVault.

Handles SHA-256 hash comparison for incremental ingestion,
deleted file chunk removal, and JSONL/SQLite persistence under indexes/<collection>/.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

from neuralvault.chunk.chunkers import get_chunker_for_document
from neuralvault.config import get_indexes_dir, is_offline
from neuralvault.contract import Chunk, Document
from neuralvault.ingest.loaders import LOADER_MAPPING, compute_sha256, load_file


class JsonlStore:
    """JSONL store for storing Documents and Chunks under indexes/<collection>/."""

    def __init__(self, collection: str, index_dir: Path | None = None) -> None:
        self.collection = collection
        if index_dir is None:
            base_dir = get_indexes_dir()
        else:
            base_dir = Path(index_dir)

        self.collection_dir = base_dir / collection
        self.collection_dir.mkdir(parents=True, exist_ok=True)
        self.store_file = self.collection_dir / "store.jsonl"
        self.documents_file = self.collection_dir / "documents.jsonl"
        self.chunks_file = self.collection_dir / "chunks.jsonl"

    def load_all(self) -> Tuple[Dict[str, Document], List[Chunk]]:
        """Load all stored documents (by source path) and chunks."""
        documents: Dict[str, Document] = {}
        chunks: List[Chunk] = []

        if self.store_file.exists():
            with open(self.store_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    item = json.loads(line)
                    kind = item.get("_kind")
                    if kind == "document":
                        item_data = {k: v for k, v in item.items() if k != "_kind"}
                        doc = Document(**item_data)
                        documents[doc.source] = doc
                    elif kind == "chunk":
                        item_data = {k: v for k, v in item.items() if k != "_kind"}
                        chunk = Chunk(**item_data)
                        chunks.append(chunk)

        return documents, chunks

    def save_all(self, documents: Dict[str, Document], chunks: List[Chunk]) -> None:
        """Atomically persist documents and chunks to collection JSONL files."""
        tmp_store = self.collection_dir / "store.jsonl.tmp"
        tmp_docs = self.collection_dir / "documents.jsonl.tmp"
        tmp_chunks = self.collection_dir / "chunks.jsonl.tmp"

        with (
            open(tmp_store, "w", encoding="utf-8") as f_store,
            open(tmp_docs, "w", encoding="utf-8") as f_docs,
            open(tmp_chunks, "w", encoding="utf-8") as f_chunks,
        ):
            for doc in documents.values():
                doc_dict = doc.model_dump()
                doc_dict["_kind"] = "document"
                line = json.dumps(doc_dict)
                f_store.write(line + "\n")
                f_docs.write(json.dumps(doc.model_dump()) + "\n")

            for chunk in chunks:
                chunk_dict = chunk.model_dump()
                chunk_dict["_kind"] = "chunk"
                line = json.dumps(chunk_dict)
                f_store.write(line + "\n")
                f_chunks.write(json.dumps(chunk.model_dump()) + "\n")

        tmp_store.replace(self.store_file)
        tmp_docs.replace(self.documents_file)
        tmp_chunks.replace(self.chunks_file)


class IngestEngine:
    """Incremental ingestion engine for NeuralVault collections."""

    def __init__(self, collection: str, index_dir: Path | None = None) -> None:
        self.collection = collection
        self.store = JsonlStore(collection, index_dir=index_dir)

    def _discover_files(self, target_path: Path) -> List[Path]:
        target_path = Path(target_path)
        if target_path.is_file():
            if target_path.suffix.lower() in LOADER_MAPPING:
                return [target_path]
            return []

        discovered: List[Path] = []
        for p in target_path.rglob("*"):
            if p.is_file() and p.suffix.lower() in LOADER_MAPPING:
                # Ignore hidden files or directories
                if any(part.startswith(".") for part in p.parts):
                    continue
                discovered.append(p)
        return sorted(discovered)

    def ingest_path(self, target_path: Path | str) -> Dict[str, Any]:
        """Ingest target file or directory incrementally into collection.

        Args:
            target_path: Path to a file or directory.

        Returns:
            Dictionary with metrics (processed, skipped, deleted, total_documents, total_chunks).
        """
        target_path = Path(target_path).resolve()
        existing_docs, existing_chunks = self.store.load_all()

        discovered_files = (
            self.find_files_to_process(target_path) if target_path.exists() else []
        )
        discovered_sources: Set[str] = {str(f.resolve()) for f in discovered_files}

        # Track unchanged, modified/new, and deleted sources
        unchanged_sources: Set[str] = set()
        to_process: List[Path] = []

        for f in discovered_files:
            src_str = str(f.resolve())
            if src_str in existing_docs:
                curr_hash = compute_sha256(f)
                if existing_docs[src_str].hash == curr_hash:
                    unchanged_sources.add(src_str)
                else:
                    to_process.append(f)
            else:
                to_process.append(f)

        # Identify deleted sources (previously ingested under target_path that no longer exist)
        deleted_sources: Set[str] = set()
        for src in existing_docs.keys():
            try:
                src_path = Path(src)
                is_sub = src_path == target_path or src_path.is_relative_to(target_path)
                if is_sub and src not in discovered_sources:
                    deleted_sources.add(src)
            except ValueError:
                pass

        # Preserve unchanged files and non-deleted files outside target_path
        new_docs: Dict[str, Document] = {}
        new_chunks: List[Chunk] = []

        for src, doc in existing_docs.items():
            if src in unchanged_sources or (
                src not in discovered_sources and src not in deleted_sources
            ):
                new_docs[src] = doc

        for chunk in existing_chunks:
            if chunk.source in unchanged_sources or (
                chunk.source not in discovered_sources and chunk.source not in deleted_sources
            ):
                new_chunks.append(chunk)

        # Process new and changed files
        processed_count = 0
        newly_processed_sources: Set[str] = set()
        for f in to_process:
            src_str = str(f.resolve())
            doc, text = load_file(f, collection=self.collection)
            doc.source = src_str
            chunker = get_chunker_for_document(doc)
            file_chunks = chunker.chunk(doc, text)

            for c in file_chunks:
                c.source = src_str

            new_docs[src_str] = doc
            new_chunks.extend(file_chunks)
            newly_processed_sources.add(src_str)
            processed_count += 1

        # Save state to JSONL store
        self.store.save_all(new_docs, new_chunks)

        # Sync to SqliteStore with embeddings
        try:
            from neuralvault.embed.providers import get_embedding_provider
            from neuralvault.store.sqlite_store import SqliteStore

            sqlite_store = SqliteStore(
                self.collection, index_dir=self.store.collection_dir.parent
            )
            meta = sqlite_store.get_collection_metadata()
            embed_provider_type = "hash" if is_offline() else "fastembed"
            embedder = get_embedding_provider(
                embed_provider_type,
                model_name=meta.embedding_model,
                dimension=meta.embedding_dim,
            )

            for src in deleted_sources:
                sqlite_store.delete_document(src)

            if newly_processed_sources:
                docs_to_sync = [new_docs[s] for s in newly_processed_sources if s in new_docs]
                chunks_to_sync = [
                    c for c in new_chunks if c.source in newly_processed_sources
                ]
                if chunks_to_sync:
                    chunk_texts = [c.embedding_text or c.text for c in chunks_to_sync]
                    vecs = embedder.embed(chunk_texts)
                    sqlite_store.add_documents_and_chunks(
                        docs_to_sync, chunks_to_sync, vecs
                    )
        except Exception:
            pass

        return {
            "collection": self.collection,
            "processed": processed_count,
            "skipped": len(unchanged_sources),
            "deleted": len(deleted_sources),
            "total_documents": len(new_docs),
            "total_chunks": len(new_chunks),
        }

    def find_files_to_process(self, target_path: Path) -> List[Path]:
        """Find supported files under target path."""
        return self._discover_files(target_path)
