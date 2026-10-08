"""NeuralVault MCP (Model Context Protocol) Server.

Exposes list_collections, search, ask, and get_document tools.
"""

import asyncio
from typing import Any, Dict, Optional

from neuralvault.contract import AskRequest, SearchRequest
from neuralvault.service.service import RagService


def create_mcp_server(service: Optional[RagService] = None) -> Any:
    """Create and configure NeuralVault MCPServer instance with four core tools."""
    try:
        from mcp.server.mcpserver import MCPServer
    except ImportError:
        try:
            from mcp.server.fastmcp import FastMCP as MCPServer
        except ImportError as e:
            raise RuntimeError(
                "MCP SDK is not installed. Install via 'pip install mcp'"
            ) from e

    srv = service or RagService()
    mcp_app = MCPServer("NeuralVault")

    @mcp_app.tool()
    def list_collections() -> Dict[str, Any]:
        """List all available NeuralVault document collections with metadata."""
        cols = srv.list_collections()
        return {"collections": [c.model_dump() for c in cols]}

    @mcp_app.tool()
    def search(
        query: str,
        collection: str,
        top_k: int = 3,
        profile: str = "balanced",
    ) -> Dict[str, Any]:
        """Search a NeuralVault collection for relevant passages with citations and scores."""
        req = SearchRequest(
            query=query,
            collection=collection,
            top_k=top_k,
            profile=profile,
        )
        resp = srv.search(req)
        return resp.model_dump()

    @mcp_app.tool()
    def ask(
        query: str,
        collection: str,
        top_k: int = 3,
        profile: str = "balanced",
    ) -> Dict[str, Any]:
        """Ask a question and generate an answer with citations over a collection."""
        req = AskRequest(
            query=query,
            collection=collection,
            top_k=top_k,
            profile=profile,
        )
        resp = srv.ask(req)
        return resp.model_dump()

    @mcp_app.tool()
    def get_document(collection: str, doc_id_or_source: str) -> Dict[str, Any]:
        """Get document metadata by doc_id or source file path."""
        doc = srv.get_document(collection, doc_id_or_source)
        if doc:
            return doc.model_dump()
        return {"error": f"Document '{doc_id_or_source}' not found in collection '{collection}'"}

    return mcp_app


def run_stdio_server(service: Optional[RagService] = None) -> None:
    """Run MCP server over stdio transport."""
    app = create_mcp_server(service=service)
    if hasattr(app, "run"):
        app.run(transport="stdio")
    else:
        asyncio.run(app.run_stdio_async())


def run_http_server(
    host: str = "127.0.0.1", port: int = 8042, service: Optional[RagService] = None
) -> None:
    """Run MCP server over HTTP transport."""
    app = create_mcp_server(service=service)
    if hasattr(app, "run"):
        app.run(transport="sse", host=host, port=port)
    else:
        asyncio.run(app.run_sse_async(host=host, port=port))
