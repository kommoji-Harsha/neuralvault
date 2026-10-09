"""NeuralVault Python SDK client supporting local (in-process) and HTTP modes."""

from typing import List, Optional

import requests

from neuralvault.contract import (
    AskRequest,
    AskResponse,
    Collection,
    Document,
    SearchRequest,
    SearchResponse,
)
from neuralvault.service.service import RagService


class RagClient:
    """Unified client for NeuralVault retrieval and Q&A."""

    def __init__(
        self,
        mode: str = "local",
        collection: str = "my-project",
        base_url: str = "http://localhost:8042",
        api_key: Optional[str] = None,
        index_dir=None,
    ) -> None:
        self.mode = mode.lower()
        self.default_collection = collection
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self._service: Optional[RagService] = None

        if self.mode == "local":
            self._service = RagService(index_dir=index_dir)

    def list_collections(self) -> List[Collection]:
        """List available document collections."""
        if self.mode == "local":
            assert self._service is not None
            return self._service.list_collections()
        else:
            headers = {}
            if self.api_key:
                headers["X-API-Key"] = self.api_key
            res = requests.get(f"{self.base_url}/collections", headers=headers, timeout=10)
            res.raise_for_status()
            return [Collection(**c) for c in res.json()]

    def get_document(
        self, doc_id_or_source: str, collection: Optional[str] = None
    ) -> Optional[Document]:
        """Get document metadata by ID or source file path."""
        target_col = collection or self.default_collection
        if self.mode == "local":
            assert self._service is not None
            return self._service.get_document(target_col, doc_id_or_source)
        else:
            headers = {}
            if self.api_key:
                headers["X-API-Key"] = self.api_key
            res = requests.get(
                f"{self.base_url}/documents/{target_col}/{doc_id_or_source}",
                headers=headers,
                timeout=10,
            )
            if res.status_code == 404:
                return None
            res.raise_for_status()
            return Document(**res.json())

    def search(
        self,
        query: str,
        collection: Optional[str] = None,
        top_k: int = 3,
        profile: str = "balanced",
    ) -> SearchResponse:
        """Search a collection for relevant passages."""
        target_col = collection or self.default_collection
        req = SearchRequest(
            query=query,
            collection=target_col,
            top_k=top_k,
            profile=profile,
        )

        if self.mode == "local":
            assert self._service is not None
            return self._service.search(req)
        else:
            headers = {}
            if self.api_key:
                headers["X-API-Key"] = self.api_key
            res = requests.post(
                f"{self.base_url}/search",
                json=req.model_dump(),
                headers=headers,
                timeout=30,
            )
            res.raise_for_status()
            return SearchResponse(**res.json())

    def ask(
        self,
        query: str,
        collection: Optional[str] = None,
        top_k: int = 3,
        profile: str = "balanced",
    ) -> AskResponse:
        """Ask a question and generate an answer over a collection."""
        target_col = collection or self.default_collection
        req = AskRequest(
            query=query,
            collection=target_col,
            top_k=top_k,
            profile=profile,
        )

        if self.mode == "local":
            assert self._service is not None
            return self._service.ask(req)
        else:
            headers = {}
            if self.api_key:
                headers["X-API-Key"] = self.api_key
            res = requests.post(
                f"{self.base_url}/ask",
                json=req.model_dump(),
                headers=headers,
                timeout=60,
            )
            res.raise_for_status()
            return AskResponse(**res.json())
