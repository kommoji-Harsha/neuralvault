"""NeuralVault abstract component interfaces.

Defines abstract base classes for Loader, Chunker, EmbeddingProvider,
Retriever, Reranker, Compressor, and Generator.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Tuple

from neuralvault.contract import AskResponse, Chunk, Document


class Loader(ABC):
    """Abstract interface for document loaders."""

    @abstractmethod
    def load(self, filepath: Path) -> Tuple[Document, str]:
        """Load document metadata and full text content from a file path.

        Args:
            filepath: Path to the document file.

        Returns:
            Tuple of (Document metadata, document full text content).
        """
        pass


class Chunker(ABC):
    """Abstract interface for document chunkers."""

    @abstractmethod
    def chunk(self, doc: Document, text: str) -> List[Chunk]:
        """Split a document's full text content into Chunks.

        Args:
            doc: Document metadata object.
            text: Full raw text of the document.

        Returns:
            List of Chunk objects.
        """
        pass


class EmbeddingProvider(ABC):
    """Abstract interface for embedding generators."""

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Return the model name."""
        pass

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Return the embedding vector dimension."""
        pass

    @abstractmethod
    def embed(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of text strings into vector representations.

        Args:
            texts: List of text strings to embed.

        Returns:
            List of float vector embeddings.
        """
        pass


class Retriever(ABC):
    """Abstract interface for search retrievers."""

    @abstractmethod
    def retrieve(self, query: str, top_k: int = 10) -> List[Chunk]:
        """Retrieve top_k matching chunks for a query.

        Args:
            query: User search query.
            top_k: Number of candidate chunks to retrieve.

        Returns:
            List of retrieved Chunk objects with relevance scores.
        """
        pass


class Reranker(ABC):
    """Abstract interface for candidate rerankers."""

    @abstractmethod
    def rerank(self, query: str, chunks: List[Chunk], top_k: int = 3) -> List[Chunk]:
        """Rerank candidate chunks using joint query-chunk scoring.

        Args:
            query: User query.
            chunks: List of candidate Chunk objects.
            top_k: Final number of reranked chunks to return.

        Returns:
            List of top_k reranked Chunk objects.
        """
        pass


class Compressor(ABC):
    """Abstract interface for context compressors."""

    @abstractmethod
    def compress(self, query: str, chunks: List[Chunk], max_chars: int = 1500) -> List[Chunk]:
        """Compress candidate chunks to fit within a character budget.

        Args:
            query: User query for context relevance.
            chunks: List of input Chunk objects.
            max_chars: Character length budget for output context.

        Returns:
            List of compressed Chunk objects preserving source/location.
        """
        pass


class Generator(ABC):
    """Abstract interface for Q&A generators."""

    @abstractmethod
    def generate(self, query: str, chunks: List[Chunk]) -> AskResponse:
        """Generate an answer with citations given a query and context chunks.

        Args:
            query: User question.
            chunks: Context Chunk objects retrieved for query.

        Returns:
            AskResponse containing answer, citations, and status notes.
        """
        pass
