"""NeuralVault data contracts and models.

These Pydantic models define the data structures exchanged across NeuralVault
components, adapters (@tool, MCP, REST, SDK), and external consumers.
"""

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class Chunk(BaseModel):
    """A retrieved chunk of content with source attribution and relevance score.

    Attributes:
        chunk_id: Unique identifier for the chunk.
        text: Display text content of the chunk shown to users and LLMs.
        source: File path or origin identifier of the chunk.
        location: Human-readable location within source (heading path or line numbers).
        score: Relevance score assigned during retrieval/reranking.
        embedding_text: Optional text representation used for embedding.
        metadata: Key-value metadata associated with the chunk.
    """

    chunk_id: str = Field(description="Unique identifier for the chunk")
    text: str = Field(description="Display text content of the chunk")
    source: str = Field(description="Source path or file origin")
    location: str = Field(
        description="Location context within file (e.g. heading path or line range)"
    )
    score: float = Field(default=0.0, description="Relevance score for this chunk")
    embedding_text: Optional[str] = Field(
        default=None, description="Embedding text representation prepended with context headers"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Additional arbitrary metadata"
    )


class Document(BaseModel):
    """A document ingested into NeuralVault.

    Attributes:
        doc_id: Unique identifier for the document.
        source: File path or origin identifier.
        title: Inferred or explicit document title.
        mime_type: MIME type of the document.
        hash: SHA-256 hash of the content for incremental ingestion.
        mtime: File modification timestamp.
        word_count: Total word count in document.
        collection: Name of the collection this document belongs to.
        metadata: Extra metadata key-value pairs.
    """

    doc_id: str = Field(description="Unique identifier for the document")
    source: str = Field(description="Source file path or identifier")
    title: str = Field(description="Inferred or explicit title")
    mime_type: str = Field(default="text/plain", description="MIME type")
    hash: str = Field(description="SHA-256 content hash")
    mtime: float = Field(default=0.0, description="File modification time timestamp")
    word_count: int = Field(default=0, description="Word count of document")
    collection: str = Field(description="Collection name")
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Additional document metadata"
    )


class Collection(BaseModel):
    """Metadata and configuration for a document collection.

    Attributes:
        name: Name of the collection.
        description: Description of what content is in this collection.
        embedding_model: Embedding model name used to index this collection.
        embedding_dim: Dimension of vector embeddings for this collection.
        document_count: Number of ingested documents in the collection.
        chunk_count: Total number of chunks in the collection.
        created_at: ISO timestamp when collection was created.
        updated_at: ISO timestamp when collection was last updated.
    """

    name: str = Field(description="Unique collection name")
    description: str = Field(default="", description="Description of collection content")
    embedding_model: str = Field(
        default="BAAI/bge-small-en-v1.5", description="Embedding model name"
    )
    embedding_dim: int = Field(default=384, description="Embedding vector dimension")
    document_count: int = Field(default=0, description="Number of documents in collection")
    chunk_count: int = Field(default=0, description="Total number of chunks in collection")
    created_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
        description="Creation timestamp",
    )
    updated_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
        description="Last updated timestamp",
    )


class SearchRequest(BaseModel):
    """Request payload for retrieving passages.

    Attributes:
        query: Search query text.
        collection: Target collection name.
        top_k: Number of relevant chunks to retrieve.
        profile: Retrieval profile name ('fast', 'balanced', 'best').
        filters: Optional key-value filter conditions.
    """

    query: str = Field(description="Search query text")
    collection: str = Field(description="Target collection name")
    top_k: int = Field(default=3, ge=1, le=100, description="Number of results to return")
    profile: str = Field(
        default="balanced", description="Retrieval profile ('fast', 'balanced', 'best')"
    )
    filters: Dict[str, Any] = Field(default_factory=dict, description="Metadata key-value filters")


class SearchResponse(BaseModel):
    """Response payload containing search results.

    Attributes:
        results: List of retrieved chunks matching the query.
        note: Explicit note or warning message (e.g. 'no relevant passages found').
        latency_ms: Time taken to process search in milliseconds.
    """

    results: List[Chunk] = Field(default_factory=list, description="Retrieved chunks")
    note: Optional[str] = Field(
        default=None, description="Explicit note, e.g. 'no relevant passages found'"
    )
    latency_ms: float = Field(default=0.0, description="Latency in milliseconds")


class AskRequest(BaseModel):
    """Request payload for Q&A over retrieved passages.

    Attributes:
        query: User question or prompt.
        collection: Target collection name.
        top_k: Number of relevant chunks to use for context.
        profile: Retrieval profile name ('fast', 'balanced', 'best').
    """

    query: str = Field(description="User question or prompt")
    collection: str = Field(description="Target collection name")
    top_k: int = Field(
        default=3, ge=1, le=100, description="Number of passages to retrieve for context"
    )
    profile: str = Field(
        default="balanced", description="Retrieval profile ('fast', 'balanced', 'best')"
    )


class AskResponse(BaseModel):
    """Response payload for Q&A query.

    Attributes:
        answer: Generated answer text.
        citations: Chunks cited in generating the answer.
        note: Explicit note or warning message.
        insufficient_context: Boolean flag indicating if retrieved context was insufficient.
    """

    answer: str = Field(description="Generated answer text")
    citations: List[Chunk] = Field(default_factory=list, description="Cited chunks used for answer")
    note: Optional[str] = Field(
        default=None, description="Explicit note, e.g. 'insufficient context to answer question'"
    )
    insufficient_context: bool = Field(
        default=False, description="True if retrieval did not pass threshold to answer safely"
    )
