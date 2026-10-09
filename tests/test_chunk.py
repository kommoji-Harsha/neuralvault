"""Unit tests for chunker strategies."""

from neuralvault.chunk.chunkers import (
    FixedSizeChunker,
    MarkdownChunker,
    PdfDocxChunker,
    PythonAstChunker,
    RecursiveGenericChunker,
)
from neuralvault.contract import Document


def test_markdown_chunker():
    doc = Document(
        doc_id="d1", source="docs/setup.md", title="Setup", hash="h1", collection="col"
    )
    text = (
        "# Installation\n\nTo install, run pip.\n\n"
        "## Windows\n\nWindows instructions.\n\n"
        "## Linux\n\nLinux instructions."
    )
    chunker = MarkdownChunker()
    chunks = chunker.chunk(doc, text)

    assert len(chunks) == 3
    assert "Installation" in chunks[0].location
    assert "Windows" in chunks[1].location
    assert "docs/setup.md > Installation > Windows" in chunks[1].embedding_text
    assert chunks[1].text == "## Windows\n\nWindows instructions."


def test_python_ast_chunker():
    doc = Document(
        doc_id="d2", source="src/math.py", title="Math", hash="h2", collection="col"
    )
    code = (
        '"""Math module."""\n\n'
        "def add(a, b):\n"
        '    """Add numbers."""\n'
        "    return a + b\n\n"
        "class Calculator:\n"
        '    """Calc class."""\n'
        "    def __init__(self):\n"
        "        pass\n\n"
        "    def multiply(self, a, b):\n"
        "        return a * b\n"
    )
    chunker = PythonAstChunker()
    chunks = chunker.chunk(doc, code)

    # 1 module docstring + 1 top function + 1 class (with __init__) + 1 method
    assert len(chunks) == 4
    locations = [c.location for c in chunks]
    assert "src/math.py::module_docstring" in locations
    assert "src/math.py::add" in locations
    assert "src/math.py::Calculator" in locations
    assert "src/math.py::Calculator.multiply" in locations


def test_pdf_docx_chunker():
    doc = Document(
        doc_id="d3", source="paper.pdf", title="Paper", hash="h3", collection="col"
    )
    text = (
        "--- Page 1 ---\n\nFirst paragraph content.\n\n"
        "| Col1 | Col2 |\n| Val1 | Val2 |\n\n"
        "Paragraph after table."
    )
    chunker = PdfDocxChunker()
    chunks = chunker.chunk(doc, text)

    assert len(chunks) >= 2
    # Ensure table chunk was created
    table_chunks = [c for c in chunks if "| Col1 | Col2 |" in c.text]
    assert len(table_chunks) == 1


def test_recursive_generic_chunker():
    doc = Document(
        doc_id="d4", source="notes.txt", title="Notes", hash="h4", collection="col"
    )
    text = "Paragraph 1 text.\n\nParagraph 2 text.\n\nParagraph 3 text."
    chunker = RecursiveGenericChunker(max_chars=30)
    chunks = chunker.chunk(doc, text)

    assert len(chunks) >= 3
    for c in chunks:
        assert len(c.text) <= 30


def test_fixed_size_chunker():
    doc = Document(
        doc_id="d5", source="raw.txt", title="Raw", hash="h5", collection="col"
    )
    text = "A" * 1200
    chunker = FixedSizeChunker(chunk_size=500, overlap=50)
    chunks = chunker.chunk(doc, text)

    assert len(chunks) == 3
    assert "chars 0-500" in chunks[0].location
