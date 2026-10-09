"""FastAPI REST API server for NeuralVault."""

import os
from typing import List, Optional

from fastapi import Depends, FastAPI, HTTPException, Security, status
from fastapi.security import APIKeyHeader

from neuralvault.contract import (
    AskRequest,
    AskResponse,
    Collection,
    Document,
    SearchRequest,
    SearchResponse,
)
from neuralvault.service.service import RagService

app = FastAPI(
    title="NeuralVault REST API",
    description="Offline-first retrieval library REST API.",
    version="0.1.0",
)

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def verify_api_key(api_key: Optional[str] = Security(api_key_header)) -> None:
    """Validate API key if NEURALVAULT_API_KEY env var is set."""
    expected_key = os.getenv("NEURALVAULT_API_KEY")
    if expected_key:
        if not api_key or api_key != expected_key:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid or missing API key",
            )


@app.get("/collections", response_model=List[Collection], dependencies=[Depends(verify_api_key)])
def list_collections() -> List[Collection]:
    """List available document collections."""
    service = RagService()
    return service.list_collections()


@app.post("/search", response_model=SearchResponse, dependencies=[Depends(verify_api_key)])
def search(request: SearchRequest) -> SearchResponse:
    """Search a collection for relevant passages."""
    service = RagService()
    return service.search(request)


@app.post("/ask", response_model=AskResponse, dependencies=[Depends(verify_api_key)])
def ask(request: AskRequest) -> AskResponse:
    """Ask a question and generate an answer over a collection."""
    service = RagService()
    return service.ask(request)


@app.get(
    "/documents/{collection}/{doc_id_or_source:path}",
    response_model=Document,
    dependencies=[Depends(verify_api_key)],
)
def get_document(collection: str, doc_id_or_source: str) -> Document:
    """Get document metadata by doc_id or source file path."""
    service = RagService()
    doc = service.get_document(collection, doc_id_or_source)
    if not doc:
        raise HTTPException(status_code=404, detail=f"Document '{doc_id_or_source}' not found")
    return doc
