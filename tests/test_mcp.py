"""Unit and integration equivalence tests for NeuralVault MCP server."""

import asyncio
import json

from neuralvault.contract import AskRequest, Chunk, Document, SearchRequest
from neuralvault.mcp_server.server import create_mcp_server
from neuralvault.service.service import RagService
from neuralvault.store.sqlite_store import SqliteStore


def test_mcp_tools_equivalence_with_service(tmp_path):
    store = SqliteStore("mcp_test_col", index_dir=tmp_path)

    doc = Document(
        doc_id="d100",
        source="docs/mcp.md",
        title="MCP Doc",
        mime_type="text/markdown",
        hash="mcp_hash",
        mtime=500.0,
        word_count=20,
        collection="mcp_test_col",
    )
    chunk = Chunk(
        chunk_id="c100",
        text="MCP server exposes list_collections, search, ask, and get_document tools.",
        source="docs/mcp.md",
        location="docs/mcp.md > Tools",
        score=0.95,
        metadata={"doc_id": "d100"},
    )
    store.add_documents_and_chunks([doc], [chunk])

    service = RagService(index_dir=tmp_path, provider_type="hash")
    mcp_app = create_mcp_server(service=service)

    async def _run_test():
        # 1. Test list_collections tool
        direct_cols = service.list_collections()
        tools = await mcp_app.list_tools()
        tool_map = {t.name: t for t in tools}

        assert "list_collections" in tool_map
        assert "search" in tool_map
        assert "ask" in tool_map
        assert "get_document" in tool_map

        # Execute tool calls via call_tool
        res_cols = await mcp_app.call_tool("list_collections", {})
        mcp_cols_data = json.loads(res_cols.content[0].text)
        mcp_cols = mcp_cols_data["collections"]
        assert len(mcp_cols) == len(direct_cols)
        assert mcp_cols[0]["name"] == direct_cols[0].name

        # 2. Test search tool equivalence
        s_req = SearchRequest(query="MCP server", collection="mcp_test_col", top_k=2)
        direct_search = service.search(s_req).model_dump()
        res_search = await mcp_app.call_tool(
            "search", {"query": "MCP server", "collection": "mcp_test_col", "top_k": 2}
        )
        mcp_search = json.loads(res_search.content[0].text)

        assert mcp_search["results"][0]["chunk_id"] == direct_search["results"][0]["chunk_id"]
        assert mcp_search["results"][0]["text"] == direct_search["results"][0]["text"]

        # 3. Test ask tool equivalence
        a_req = AskRequest(query="What tools are exposed?", collection="mcp_test_col")
        direct_ask = service.ask(a_req).model_dump()
        res_ask = await mcp_app.call_tool(
            "ask", {"query": "What tools are exposed?", "collection": "mcp_test_col"}
        )
        mcp_ask = json.loads(res_ask.content[0].text)

        assert mcp_ask["answer"] == direct_ask["answer"]
        assert len(mcp_ask["citations"]) == len(direct_ask["citations"])

        # 4. Test get_document tool equivalence
        direct_doc = service.get_document("mcp_test_col", "d100")
        assert direct_doc is not None
        res_doc = await mcp_app.call_tool(
            "get_document", {"collection": "mcp_test_col", "doc_id_or_source": "d100"}
        )
        mcp_doc = json.loads(res_doc.content[0].text)

        assert mcp_doc["doc_id"] == direct_doc.doc_id
        assert mcp_doc["source"] == direct_doc.source

    asyncio.run(_run_test())
