"""SQLite storage engine per collection.

Stores documents, chunks, FTS5 virtual table for keyword search,
collection model metadata, and vector embeddings (sqlite-vec or numpy fallback).
"""

import json
import sqlite3
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from neuralvault.config import get_indexes_dir
from neuralvault.contract import Chunk, Collection, Document


def vector_to_blob(vec: List[float]) -> bytes:
    """Convert float vector list to binary float32 numpy BLOB."""
    return np.array(vec, dtype=np.float32).tobytes()


def blob_to_vector(blob: bytes) -> List[float]:
    """Convert binary float32 BLOB to float vector list."""
    return np.frombuffer(blob, dtype=np.float32).tolist()


def cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    """Compute cosine similarity between two 1D numpy arrays."""
    norm_a = np.linalg.norm(vec_a)
    norm_b = np.linalg.norm(vec_b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(vec_a, vec_b) / (norm_a * norm_b))


class SqliteStore:
    """SQLite store for a single NeuralVault collection."""

    def __init__(
        self,
        collection: str,
        index_dir: Optional[Path] = None,
        embedding_model: str = "BAAI/bge-small-en-v1.5",
        embedding_dim: int = 384,
    ) -> None:
        self.collection = collection
        if index_dir is None:
            base_dir = get_indexes_dir()
        else:
            base_dir = Path(index_dir)

        base_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = base_dir / f"{collection}.db"
        self._init_db(embedding_model, embedding_dim)

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self, embedding_model: str, embedding_dim: int) -> None:
        with self._get_connection() as conn:
            cur = conn.cursor()

            # Metadata table
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS collection_metadata (
                    key TEXT PRIMARY KEY,
                    value TEXT
                )
                """
            )

            # Check existing metadata for model and dimension consistency
            cur.execute("SELECT value FROM collection_metadata WHERE key = 'embedding_model'")
            row_model = cur.fetchone()
            cur.execute("SELECT value FROM collection_metadata WHERE key = 'embedding_dim'")
            row_dim = cur.fetchone()

            if row_model and row_dim:
                existing_model = row_model["value"]
                existing_dim = int(row_dim["value"])
                if existing_model != embedding_model or existing_dim != embedding_dim:
                    raise ValueError(
                        f"Collection '{self.collection}' model mismatch: "
                        f"store uses '{existing_model}' ({existing_dim}d), "
                        f"requested '{embedding_model}' ({embedding_dim}d)."
                    )
            else:
                cur.execute(
                    "INSERT OR REPLACE INTO collection_metadata "
                    "(key, value) VALUES ('embedding_model', ?)",
                    (embedding_model,),
                )
                cur.execute(
                    "INSERT OR REPLACE INTO collection_metadata "
                    "(key, value) VALUES ('embedding_dim', ?)",
                    (str(embedding_dim),),
                )

            # Documents table
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS documents (
                    doc_id TEXT PRIMARY KEY,
                    source TEXT UNIQUE,
                    title TEXT,
                    mime_type TEXT,
                    hash TEXT,
                    mtime REAL,
                    word_count INTEGER,
                    collection TEXT,
                    metadata TEXT
                )
                """
            )

            # Chunks table
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS chunks (
                    chunk_id TEXT PRIMARY KEY,
                    doc_id TEXT,
                    source TEXT,
                    location TEXT,
                    text TEXT,
                    embedding_text TEXT,
                    score REAL,
                    metadata TEXT,
                    FOREIGN KEY(doc_id) REFERENCES documents(doc_id) ON DELETE CASCADE
                )
                """
            )

            # FTS5 virtual table
            cur.execute(
                """
                CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
                    chunk_id UNINDEXED,
                    text,
                    embedding_text,
                    source,
                    location
                )
                """
            )

            # Embeddings table
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS embeddings (
                    chunk_id TEXT PRIMARY KEY,
                    vector BLOB,
                    FOREIGN KEY(chunk_id) REFERENCES chunks(chunk_id) ON DELETE CASCADE
                )
                """
            )

            conn.commit()

    def get_collection_metadata(self) -> Collection:
        """Get collection configuration details."""
        with self._get_connection() as conn:
            cur = conn.cursor()
            cur.execute("SELECT value FROM collection_metadata WHERE key = 'embedding_model'")
            model_row = cur.fetchone()
            cur.execute("SELECT value FROM collection_metadata WHERE key = 'embedding_dim'")
            dim_row = cur.fetchone()

            cur.execute("SELECT COUNT(*) as count FROM documents")
            doc_count = cur.fetchone()["count"]

            cur.execute("SELECT COUNT(*) as count FROM chunks")
            chunk_count = cur.fetchone()["count"]

            return Collection(
                name=self.collection,
                description=f"NeuralVault collection {self.collection}",
                embedding_model=model_row["value"] if model_row else "unknown",
                embedding_dim=int(dim_row["value"]) if dim_row else 384,
                document_count=doc_count,
                chunk_count=chunk_count,
            )

    def add_documents_and_chunks(
        self,
        documents: List[Document],
        chunks: List[Chunk],
        embeddings: Optional[List[List[float]]] = None,
    ) -> None:
        """Insert or replace documents, chunks, FTS entries, and vector embeddings."""
        with self._get_connection() as conn:
            cur = conn.cursor()

            # Insert documents
            for doc in documents:
                cur.execute(
                    """
                    INSERT OR REPLACE INTO documents
                    (doc_id, source, title, mime_type, hash, mtime, word_count,
                     collection, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        doc.doc_id,
                        doc.source,
                        doc.title,
                        doc.mime_type,
                        doc.hash,
                        doc.mtime,
                        doc.word_count,
                        doc.collection,
                        json.dumps(doc.metadata),
                    ),
                )

            # Insert chunks and FTS
            for i, chunk in enumerate(chunks):
                cur.execute(
                    """
                    INSERT OR REPLACE INTO chunks
                    (chunk_id, doc_id, source, location, text, embedding_text, score, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        chunk.chunk_id,
                        chunk.metadata.get("doc_id", ""),
                        chunk.source,
                        chunk.location,
                        chunk.text,
                        chunk.embedding_text or chunk.text,
                        chunk.score,
                        json.dumps(chunk.metadata),
                    ),
                )

                cur.execute("DELETE FROM chunks_fts WHERE chunk_id = ?", (chunk.chunk_id,))
                cur.execute(
                    """
                    INSERT INTO chunks_fts (chunk_id, text, embedding_text, source, location)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        chunk.chunk_id,
                        chunk.text,
                        chunk.embedding_text or chunk.text,
                        chunk.source,
                        chunk.location,
                    ),
                )

                if embeddings and i < len(embeddings):
                    vec_blob = vector_to_blob(embeddings[i])
                    cur.execute(
                        "INSERT OR REPLACE INTO embeddings (chunk_id, vector) VALUES (?, ?)",
                        (chunk.chunk_id, vec_blob),
                    )

            conn.commit()

    def search_bm25(self, query: str, top_k: int = 20) -> List[Chunk]:
        """Perform BM25 keyword search using SQLite FTS5."""
        if not query.strip():
            return []

        # Sanitize query for FTS5
        clean_query = " ".join(query.split())
        with self._get_connection() as conn:
            cur = conn.cursor()
            cur.execute(
                """
                SELECT c.chunk_id, c.source, c.location, c.text, c.embedding_text, c.metadata,
                       f.rank as bm25_rank
                FROM chunks_fts f
                JOIN chunks c ON f.chunk_id = c.chunk_id
                WHERE chunks_fts MATCH ?
                ORDER BY rank
                LIMIT ?
                """,
                (clean_query, top_k),
            )
            rows = cur.fetchall()

            chunks: List[Chunk] = []
            for row in rows:
                meta = json.loads(row["metadata"]) if row["metadata"] else {}
                chunks.append(
                    Chunk(
                        chunk_id=row["chunk_id"],
                        text=row["text"],
                        source=row["source"],
                        location=row["location"],
                        score=float(-row["bm25_rank"]),  # FTS5 rank is lower for better match
                        embedding_text=row["embedding_text"],
                        metadata=meta,
                    )
                )
            return chunks

    def search_vector(self, query_vector: List[float], top_k: int = 20) -> List[Chunk]:
        """Perform exact cosine search over vector embeddings using numpy fallback."""
        if not query_vector:
            return []

        q_vec = np.array(query_vector, dtype=np.float32)

        with self._get_connection() as conn:
            cur = conn.cursor()
            cur.execute(
                """
                SELECT e.chunk_id, e.vector, c.source, c.location, c.text,
                       c.embedding_text, c.metadata
                FROM embeddings e
                JOIN chunks c ON e.chunk_id = c.chunk_id
                """
            )
            rows = cur.fetchall()

            scored: List[Tuple[float, sqlite3.Row]] = []
            for row in rows:
                v_blob = row["vector"]
                if not v_blob:
                    continue
                cand_vec = np.frombuffer(v_blob, dtype=np.float32)
                sim = cosine_similarity(q_vec, cand_vec)
                scored.append((sim, row))

            scored.sort(key=lambda x: x[0], reverse=True)
            top_rows = scored[:top_k]

            chunks: List[Chunk] = []
            for sim, row in top_rows:
                meta = json.loads(row["metadata"]) if row["metadata"] else {}
                chunks.append(
                    Chunk(
                        chunk_id=row["chunk_id"],
                        text=row["text"],
                        source=row["source"],
                        location=row["location"],
                        score=float(sim),
                        embedding_text=row["embedding_text"],
                        metadata=meta,
                    )
                )
            return chunks

    def delete_document(self, source_path: str) -> None:
        """Delete document and all associated chunks and embeddings by source path."""
        with self._get_connection() as conn:
            cur = conn.cursor()
            cur.execute("SELECT doc_id FROM documents WHERE source = ?", (source_path,))
            row = cur.fetchone()
            if row:
                doc_id = row["doc_id"]
                cur.execute("SELECT chunk_id FROM chunks WHERE doc_id = ?", (doc_id,))
                chunk_rows = cur.fetchall()
                for c_row in chunk_rows:
                    cur.execute("DELETE FROM chunks_fts WHERE chunk_id = ?", (c_row["chunk_id"],))
                    cur.execute("DELETE FROM embeddings WHERE chunk_id = ?", (c_row["chunk_id"],))
                cur.execute("DELETE FROM chunks WHERE doc_id = ?", (doc_id,))
                cur.execute("DELETE FROM documents WHERE doc_id = ?", (doc_id,))
                conn.commit()
