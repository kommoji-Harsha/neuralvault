"""Document loaders for various file formats.

Supports Markdown, plain text, Python, HTML, PDF (via PyMuPDF),
DOCX (via docx2txt), CSV/TSV, and JSON/YAML.
"""

import hashlib
import json
import re
from pathlib import Path
from typing import Tuple, Type

import yaml

from neuralvault.contract import Document
from neuralvault.interfaces import Loader


def compute_sha256(filepath: Path) -> str:
    """Compute SHA-256 hex digest of a file's binary content."""
    sha = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            sha.update(chunk)
    return sha.hexdigest()


def read_text_safe(filepath: Path) -> str:
    """Read file content as text using UTF-8 with fallback encoding error handling."""
    filepath = Path(filepath)
    try:
        return filepath.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return filepath.read_text(encoding="latin-1", errors="replace")


class BaseFileLoader(Loader):
    """Base document loader handling file hashing and Document model construction."""

    mime_type = "text/plain"

    def __init__(self, collection: str = "default") -> None:
        self.collection = collection

    def infer_title(self, filepath: Path, content: str) -> str:
        """Infer document title from content or filename."""
        return filepath.stem.replace("_", " ").replace("-", " ").title()

    def extract_text(self, filepath: Path) -> str:
        """Extract clean text content from file path."""
        return read_text_safe(filepath)

    def load(self, filepath: Path) -> Tuple[Document, str]:
        """Load document metadata and full text from file path."""
        filepath = Path(filepath)
        content = self.extract_text(filepath)
        file_hash = compute_sha256(filepath)
        st = filepath.stat()
        title = self.infer_title(filepath, content)
        word_count = len(content.split())

        doc = Document(
            doc_id=f"doc_{file_hash[:16]}",
            source=str(filepath),
            title=title,
            mime_type=self.mime_type,
            hash=file_hash,
            mtime=st.st_mtime,
            word_count=word_count,
            collection=self.collection,
        )
        return doc, content


class TextLoader(BaseFileLoader):
    """Plain text loader (.txt)."""

    mime_type = "text/plain"


class MarkdownLoader(BaseFileLoader):
    """Markdown document loader (.md)."""

    mime_type = "text/markdown"

    def infer_title(self, filepath: Path, content: str) -> str:
        # Match first H1 heading (# Title)
        match = re.search(r"^#\s+(.+)$", content, re.MULTILINE)
        if match:
            return match.group(1).strip()
        return super().infer_title(filepath, content)


class PythonLoader(BaseFileLoader):
    """Python source code loader (.py)."""

    mime_type = "text/x-python"

    def infer_title(self, filepath: Path, content: str) -> str:
        # Check module docstring or fallback to filename
        match = re.match(r'^\s*"""(.*?)"""', content, re.DOTALL)
        if match:
            first_line = match.group(1).strip().split("\n")[0].strip()
            if first_line:
                return first_line
        return str(filepath)


class HTMLLoader(BaseFileLoader):
    """HTML document loader (.html, .htm)."""

    mime_type = "text/html"

    def infer_title(self, filepath: Path, content: str) -> str:
        raw_html = read_text_safe(filepath)
        title_match = re.search(
            r"<title>(.*?)</title>", raw_html, re.IGNORECASE | re.DOTALL
        )
        if title_match:
            return title_match.group(1).strip()
        h1_match = re.search(
            r"<h1[^>]*>(.*?)</h1>", raw_html, re.IGNORECASE | re.DOTALL
        )
        if h1_match:
            clean_h1 = re.sub(r"<[^>]+>", "", h1_match.group(1)).strip()
            if clean_h1:
                return clean_h1
        return super().infer_title(filepath, content)

    def extract_text(self, filepath: Path) -> str:
        raw_html = read_text_safe(filepath)
        # Remove script and style tags
        cleaned = re.sub(
            r"<(script|style)[^>]*>.*?</\1>", "", raw_html, flags=re.IGNORECASE | re.DOTALL
        )
        # Convert break and block tags to newlines
        cleaned = re.sub(
            r"<(br|p|div|h[1-6]|li|tr)[^>]*>", "\n", cleaned, flags=re.IGNORECASE
        )
        # Strip remaining HTML tags
        text = re.sub(r"<[^>]+>", "", cleaned)
        # Unescape HTML entities
        import html

        text = html.unescape(text)
        # Collapse excessive newlines
        lines = [line.strip() for line in text.splitlines()]
        return "\n".join(line for line in lines if line)


class PDFLoader(BaseFileLoader):
    """PDF loader using PyMuPDF (.pdf)."""

    mime_type = "application/pdf"

    def extract_text(self, filepath: Path) -> str:
        try:
            import pymupdf as fitz
        except ImportError:
            import fitz  # type: ignore[no-redef]

        doc = fitz.open(filepath)
        page_texts = []
        for i, page in enumerate(doc):
            text = page.get_text("text")
            if text.strip():
                page_texts.append(f"--- Page {i + 1} ---\n{text.strip()}")
        doc.close()
        return "\n\n".join(page_texts)


class DocxLoader(BaseFileLoader):
    """DOCX loader using docx2txt (.docx)."""

    mime_type = (
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    )

    def extract_text(self, filepath: Path) -> str:
        import docx2txt

        text = docx2txt.process(str(filepath))
        return text or ""


class CsvTsvLoader(BaseFileLoader):
    """CSV/TSV structured text loader (.csv, .tsv)."""

    mime_type = "text/csv"

    def extract_text(self, filepath: Path) -> str:
        import csv

        filepath = Path(filepath)
        delimiter = "\t" if filepath.suffix.lower() == ".tsv" else ","
        rows = []
        with open(filepath, "r", encoding="utf-8", errors="replace") as f:
            reader = csv.reader(f, delimiter=delimiter)
            for row in reader:
                rows.append(" | ".join(row))
        return "\n".join(rows)


class JsonYamlLoader(BaseFileLoader):
    """Pretty-printed JSON/YAML loader (.json, .yaml, .yml)."""

    mime_type = "application/json"

    def extract_text(self, filepath: Path) -> str:
        filepath = Path(filepath)
        raw = read_text_safe(filepath)
        ext = filepath.suffix.lower()
        if ext == ".json":
            try:
                data = json.loads(raw)
                return json.dumps(data, indent=2)
            except Exception:
                return raw
        else:
            try:
                data = yaml.safe_load(raw)
                return yaml.dump(data, sort_keys=False)
            except Exception:
                return raw


LOADER_MAPPING: dict[str, Type[BaseFileLoader]] = {
    ".md": MarkdownLoader,
    ".markdown": MarkdownLoader,
    ".txt": TextLoader,
    ".py": PythonLoader,
    ".html": HTMLLoader,
    ".htm": HTMLLoader,
    ".pdf": PDFLoader,
    ".docx": DocxLoader,
    ".csv": CsvTsvLoader,
    ".tsv": CsvTsvLoader,
    ".json": JsonYamlLoader,
    ".yaml": JsonYamlLoader,
    ".yml": JsonYamlLoader,
}


def get_loader_for_file(filepath: Path | str, collection: str = "default") -> BaseFileLoader:
    """Get appropriate document loader instance based on file extension."""
    filepath = Path(filepath)
    ext = filepath.suffix.lower()
    loader_cls = LOADER_MAPPING.get(ext, TextLoader)
    return loader_cls(collection=collection)


def load_file(filepath: Path | str, collection: str = "default") -> Tuple[Document, str]:
    """Convenience function to load document metadata and text for any supported file."""
    filepath = Path(filepath)
    loader = get_loader_for_file(filepath, collection=collection)
    return loader.load(filepath)
