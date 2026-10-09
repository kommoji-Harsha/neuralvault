"""Unit tests for Personal AI Assistant tool adapter."""


from integrations.personal_assistant.knowledge import search_knowledge_base
from neuralvault.contract import Chunk, Document
from neuralvault.store.sqlite_store import SqliteStore
from tests.fixtures.assistant_registry import get_registered_tools, tool


def test_assistant_tool_registration():
    @tool
    def registered_tool(query: str, top_k: int = 3):
        return search_knowledge_base(query, top_k=top_k)

    registered = get_registered_tools()
    assert "registered_tool" in registered
    assert registered["registered_tool"]["parameters"]["query"]["type"] == "string"


def test_assistant_tool_execution(tmp_path, monkeypatch):
    store = SqliteStore("assistant-project", index_dir=tmp_path)
    doc = Document(
        doc_id="d_ast_1",
        source="docs/confirmation.md",
        title="Confirmation",
        hash="h_ast",
        collection="assistant-project",
    )
    chunk = Chunk(
        chunk_id="c_ast_1",
        text="The confirmation workflow asks user approval before running dangerous tools.",
        source="docs/confirmation.md",
        location="docs/confirmation.md > Confirmation",
        score=0.9,
        metadata={"doc_id": "d_ast_1"},
    )
    store.add_documents_and_chunks([doc], [chunk])

    # Clear client cache
    from integrations.personal_assistant.knowledge import _CLIENT_CACHE

    _CLIENT_CACHE.clear()

    monkeypatch.setenv("NEURALVAULT_INDEXES_DIR", str(tmp_path))

    # Test tool call with results
    results = search_knowledge_base("confirmation workflow")
    assert len(results) == 1
    assert results[0]["chunk_id"] == "c_ast_1_comp"
    assert "confirmation workflow" in results[0]["text"]

    # Test tool call with no results
    no_results = search_knowledge_base("nonexistent query text 12345")
    assert len(no_results) == 1
    assert "note" in no_results[0]
    assert no_results[0]["note"] == "no relevant passages found"
