"""Chunkers for various content types (Markdown, Python AST, PDF/DOCX, Generic, Fixed-size).

Every chunk produces two text representations:
1. text: display text shown to users and LLMs.
2. embedding_text: display text prepended with heading/file path context headers.
"""

import ast
import re
from pathlib import Path
from typing import List

from neuralvault.contract import Chunk, Document
from neuralvault.interfaces import Chunker


class FixedSizeChunker(Chunker):
    """Fixed-size character window chunker with overlap (benchmark baseline)."""

    def __init__(self, chunk_size: int = 500, overlap: int = 50) -> None:
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self, doc: Document, text: str) -> List[Chunk]:
        if not text.strip():
            return []

        chunks: List[Chunk] = []
        step = max(1, self.chunk_size - self.overlap)
        idx = 0
        i = 0

        while i < len(text):
            chunk_str = text[i : i + self.chunk_size].strip()
            if chunk_str:
                chunk_id = f"{doc.doc_id}_fixed_{idx}"
                location = f"chars {i}-{i + len(chunk_str)}"
                header = f"{doc.source} ({location})"
                embedding_text = f"{header}\n\n{chunk_str}"

                chunks.append(
                    Chunk(
                        chunk_id=chunk_id,
                        text=chunk_str,
                        source=doc.source,
                        location=location,
                        score=0.0,
                        embedding_text=embedding_text,
                        metadata={"type": "fixed_size", "chunk_index": idx},
                    )
                )
                idx += 1
            i += step

        return chunks


class MarkdownChunker(Chunker):
    """Markdown chunker based on heading path hierarchy."""

    def __init__(self, max_chunk_chars: int = 1500) -> None:
        self.max_chunk_chars = max_chunk_chars

    def chunk(self, doc: Document, text: str) -> List[Chunk]:
        if not text.strip():
            return []

        lines = text.splitlines()
        sections: List[dict] = []
        heading_stack: List[tuple[int, str]] = []  # (level, heading_title)

        current_lines: List[str] = []
        current_heading_path = doc.title or doc.source

        for line in lines:
            m = re.match(r"^(#{1,6})\s+(.+)$", line)
            if m:
                # Save previous accumulated section if non-empty
                if current_lines:
                    sec_text = "\n".join(current_lines).strip()
                    if sec_text:
                        sections.append(
                            {"heading_path": current_heading_path, "text": sec_text}
                        )
                    current_lines = []

                level = len(m.group(1))
                heading_title = m.group(2).strip()

                # Pop headings of equal or deeper level from stack
                while heading_stack and heading_stack[-1][0] >= level:
                    heading_stack.pop()
                heading_stack.append((level, heading_title))

                # Build full heading path: doc.source > H1 > H2 > ...
                path_parts = [doc.source] + [h for _, h in heading_stack]
                current_heading_path = " > ".join(path_parts)
                current_lines.append(line)
            else:
                current_lines.append(line)

        if current_lines:
            sec_text = "\n".join(current_lines).strip()
            if sec_text:
                sections.append({"heading_path": current_heading_path, "text": sec_text})

        chunks: List[Chunk] = []
        idx = 0

        for sec in sections:
            path = sec["heading_path"]
            body = sec["text"]

            # If section body exceeds max limit, recursively sub-chunk
            sub_bodies = self._split_text(body, self.max_chunk_chars)
            for sub_i, sub_body in enumerate(sub_bodies):
                chunk_id = f"{doc.doc_id}_md_{idx}"
                location = path if len(sub_bodies) == 1 else f"{path} (part {sub_i + 1})"
                embedding_text = f"{location}\n\n{sub_body}"

                chunks.append(
                    Chunk(
                        chunk_id=chunk_id,
                        text=sub_body,
                        source=doc.source,
                        location=location,
                        score=0.0,
                        embedding_text=embedding_text,
                        metadata={"type": "markdown", "heading_path": path},
                    )
                )
                idx += 1

        return chunks

    def _split_text(self, text: str, max_chars: int) -> List[str]:
        if len(text) <= max_chars:
            return [text]

        paras = text.split("\n\n")
        chunks = []
        curr = []
        curr_len = 0

        for p in paras:
            if curr_len + len(p) + 2 > max_chars and curr:
                chunks.append("\n\n".join(curr))
                curr = [p]
                curr_len = len(p)
            else:
                curr.append(p)
                curr_len += len(p) + 2

        if curr:
            chunks.append("\n\n".join(curr))
        return chunks


class PythonAstChunker(Chunker):
    """Python AST chunker.

    Creates one chunk per module docstring, class, and top-level or method function.
    Preserves qualified names (e.g. src/neuralvault/retrieve.py::HybridRetriever.search)
    and splits long functions at statement boundaries without splitting mid-expression.
    """

    def __init__(self, max_chunk_chars: int = 2000) -> None:
        self.max_chunk_chars = max_chunk_chars

    def chunk(self, doc: Document, text: str) -> List[Chunk]:
        if not text.strip():
            return []

        try:
            tree = ast.parse(text)
        except Exception:
            # Fallback to generic chunker if AST parsing fails
            return RecursiveGenericChunker(max_chars=self.max_chunk_chars).chunk(doc, text)

        lines = text.splitlines()
        chunks: List[Chunk] = []
        idx = 0

        # 1. Module-level docstring
        docstring = ast.get_docstring(tree)
        if docstring:
            mod_chunk_id = f"{doc.doc_id}_py_{idx}"
            qual_name = f"{doc.source}::module_docstring"
            disp_text = f'"""{docstring}"""'
            emb_text = f"{qual_name}\n\n{disp_text}"
            chunks.append(
                Chunk(
                    chunk_id=mod_chunk_id,
                    text=disp_text,
                    source=doc.source,
                    location=qual_name,
                    score=0.0,
                    embedding_text=emb_text,
                    metadata={"type": "python_module_docstring"},
                )
            )
            idx += 1

        # 2. Extract classes and functions
        for stmt in tree.body:
            if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
                fn_chunks = self._chunk_function(doc, lines, stmt, prefix="", start_idx=idx)
                chunks.extend(fn_chunks)
                idx += len(fn_chunks)
            elif isinstance(stmt, ast.ClassDef):
                class_chunks = self._chunk_class(doc, lines, stmt, start_idx=idx)
                chunks.extend(class_chunks)
                idx += len(class_chunks)

        if not chunks:
            # Fallback if python file has no classes or top-level functions
            return RecursiveGenericChunker(max_chars=self.max_chunk_chars).chunk(doc, text)

        return chunks

    def _get_node_text(self, lines: List[str], node: ast.AST) -> str:
        start = getattr(node, "lineno", 1) - 1
        end = getattr(node, "end_lineno", len(lines))
        return "\n".join(lines[start:end])

    def _chunk_function(
        self,
        doc: Document,
        lines: List[str],
        fn_node: ast.AST,
        prefix: str,
        start_idx: int,
    ) -> List[Chunk]:
        fn_name = getattr(fn_node, "name", "function")
        qual_name = f"{doc.source}::{prefix}{fn_name}" if prefix else f"{doc.source}::{fn_name}"
        full_text = self._get_node_text(lines, fn_node)

        if len(full_text) <= self.max_chunk_chars:
            emb_text = f"{qual_name}\n\n{full_text}"
            return [
                Chunk(
                    chunk_id=f"{doc.doc_id}_py_{start_idx}",
                    text=full_text,
                    source=doc.source,
                    location=qual_name,
                    score=0.0,
                    embedding_text=emb_text,
                    metadata={"type": "python_function", "qualified_name": qual_name},
                )
            ]

        # Function is too long -> split at top-level AST statement boundaries
        body_stmts = getattr(fn_node, "body", [])
        chunks: List[Chunk] = []
        curr_lines: List[str] = []
        curr_len = 0
        part = 1

        for stmt in body_stmts:
            stmt_text = self._get_node_text(lines, stmt)
            if curr_len + len(stmt_text) > self.max_chunk_chars and curr_lines:
                chunk_body = "\n".join(curr_lines)
                loc = f"{qual_name} (part {part})"
                emb = f"{loc}\n\n{chunk_body}"
                chunks.append(
                    Chunk(
                        chunk_id=f"{doc.doc_id}_py_{start_idx + len(chunks)}",
                        text=chunk_body,
                        source=doc.source,
                        location=loc,
                        score=0.0,
                        embedding_text=emb,
                        metadata={"type": "python_function_statement_split"},
                    )
                )
                part += 1
                curr_lines = [stmt_text]
                curr_len = len(stmt_text)
            else:
                curr_lines.append(stmt_text)
                curr_len += len(stmt_text)

        if curr_lines:
            chunk_body = "\n".join(curr_lines)
            loc = f"{qual_name} (part {part})" if part > 1 else qual_name
            emb = f"{loc}\n\n{chunk_body}"
            chunks.append(
                Chunk(
                    chunk_id=f"{doc.doc_id}_py_{start_idx + len(chunks)}",
                    text=chunk_body,
                    source=doc.source,
                    location=loc,
                    score=0.0,
                    embedding_text=emb,
                    metadata={"type": "python_function_statement_split"},
                )
            )

        return chunks

    def _chunk_class(
        self,
        doc: Document,
        lines: List[str],
        cls_node: ast.ClassDef,
        start_idx: int,
    ) -> List[Chunk]:
        cls_name = cls_node.name
        qual_name = f"{doc.source}::{cls_name}"
        chunks: List[Chunk] = []

        # Class header + docstring + __init__
        init_node = None
        other_methods = []

        for item in cls_node.body:
            if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if item.name == "__init__":
                    init_node = item
                else:
                    other_methods.append(item)

        cls_header_lines = [f"class {cls_name}:"]
        cls_docstring = ast.get_docstring(cls_node)
        if cls_docstring:
            cls_header_lines.append(f'    """{cls_docstring}"""')

        if init_node:
            init_text = self._get_node_text(lines, init_node)
            cls_header_lines.append(init_text)

        cls_header_text = "\n\n".join(cls_header_lines)
        emb_text = f"{qual_name}\n\n{cls_header_text}"

        chunks.append(
            Chunk(
                chunk_id=f"{doc.doc_id}_py_{start_idx}",
                text=cls_header_text,
                source=doc.source,
                location=qual_name,
                score=0.0,
                embedding_text=emb_text,
                metadata={"type": "python_class"},
            )
        )

        # Other methods
        for method in other_methods:
            m_chunks = self._chunk_function(
                doc, lines, method, prefix=f"{cls_name}.", start_idx=start_idx + len(chunks)
            )
            chunks.extend(m_chunks)

        return chunks


class PdfDocxChunker(Chunker):
    """Paragraph chunker for PDF/DOCX documents with overlap and table preservation."""

    def __init__(self, max_chars: int = 1500, overlap_chars: int = 200) -> None:
        self.max_chars = max_chars
        self.overlap_chars = overlap_chars

    def chunk(self, doc: Document, text: str) -> List[Chunk]:
        if not text.strip():
            return []

        # Separate blocks (paragraphs, page markers, tables)
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        chunks: List[Chunk] = []

        curr_paras: List[str] = []
        curr_len = 0
        idx = 0

        for p in paragraphs:
            # Check if paragraph is a table or heading
            is_table = "|" in p and "\n" in p
            is_page_marker = p.startswith("--- Page ")

            if is_table:
                # Flush pending paragraphs
                if curr_paras:
                    c_text = "\n\n".join(curr_paras)
                    chunks.append(self._make_chunk(doc, c_text, idx, "paragraph"))
                    idx += 1
                    curr_paras = []
                    curr_len = 0
                # Table chunk preserved intact
                chunks.append(self._make_chunk(doc, p, idx, "table"))
                idx += 1
                continue

            if is_page_marker:
                if curr_paras:
                    c_text = "\n\n".join(curr_paras)
                    chunks.append(self._make_chunk(doc, c_text, idx, "page"))
                    idx += 1
                    curr_paras = []
                    curr_len = 0

            if curr_len + len(p) > self.max_chars and curr_paras:
                c_text = "\n\n".join(curr_paras)
                chunks.append(self._make_chunk(doc, c_text, idx, "paragraph"))
                idx += 1

                # Keep overlap paragraphs
                overlap_acc: List[str] = []
                overlap_len = 0
                for prev in reversed(curr_paras):
                    if overlap_len + len(prev) <= self.overlap_chars:
                        overlap_acc.insert(0, prev)
                        overlap_len += len(prev)
                    else:
                        break
                curr_paras = overlap_acc + [p]
                curr_len = sum(len(x) for x in curr_paras)
            else:
                curr_paras.append(p)
                curr_len += len(p)

        if curr_paras:
            c_text = "\n\n".join(curr_paras)
            chunks.append(self._make_chunk(doc, c_text, idx, "paragraph"))

        return chunks

    def _make_chunk(self, doc: Document, text: str, idx: int, chunk_type: str) -> Chunk:
        loc = f"{doc.source} ({chunk_type} {idx + 1})"
        emb = f"{loc}\n\n{text}"
        return Chunk(
            chunk_id=f"{doc.doc_id}_doc_{idx}",
            text=text,
            source=doc.source,
            location=loc,
            score=0.0,
            embedding_text=emb,
            metadata={"type": f"pdf_docx_{chunk_type}"},
        )


class RecursiveGenericChunker(Chunker):
    """Recursive fallback chunker splitting by paragraph -> sentence -> character boundaries."""

    def __init__(self, max_chars: int = 1500) -> None:
        self.max_chars = max_chars

    def chunk(self, doc: Document, text: str) -> List[Chunk]:
        if not text.strip():
            return []

        segments = self._split_recursive(text, self.max_chars)
        chunks: List[Chunk] = []

        for idx, seg in enumerate(segments):
            loc = f"{doc.source} (part {idx + 1})"
            emb = f"{doc.source} > {doc.title}\n\n{seg}"
            chunks.append(
                Chunk(
                    chunk_id=f"{doc.doc_id}_gen_{idx}",
                    text=seg,
                    source=doc.source,
                    location=loc,
                    score=0.0,
                    embedding_text=emb,
                    metadata={"type": "recursive_generic"},
                )
            )

        return chunks

    def _split_recursive(self, text: str, max_chars: int) -> List[str]:
        if len(text) <= max_chars:
            return [text]

        # Paragraph split
        paras = text.split("\n\n")
        if len(paras) > 1:
            return self._combine_units(paras, "\n\n", max_chars)

        # Sentence split
        sentences = re.split(r"(?<=[.!?])\s+", text)
        if len(sentences) > 1:
            return self._combine_units(sentences, " ", max_chars)

        # Hard character boundary split
        return [text[i : i + max_chars] for i in range(0, len(text), max_chars)]

    def _combine_units(self, units: List[str], joiner: str, max_chars: int) -> List[str]:
        result = []
        curr: List[str] = []
        curr_len = 0

        for u in units:
            if not u.strip():
                continue
            if len(u) > max_chars:
                # Recursively sub-split unit
                if curr:
                    result.append(joiner.join(curr))
                    curr = []
                    curr_len = 0
                result.extend(self._split_recursive(u, max_chars))
            elif curr_len + len(u) + len(joiner) > max_chars and curr:
                result.append(joiner.join(curr))
                curr = [u]
                curr_len = len(u)
            else:
                curr.append(u)
                curr_len += len(u) + len(joiner)

        if curr:
            result.append(joiner.join(curr))
        return result


def get_chunker_for_document(doc: Document) -> Chunker:
    """Select appropriate chunker implementation based on document mime type or file extension."""
    ext = Path(doc.source).suffix.lower()
    mime = doc.mime_type.lower()

    if ext in (".md", ".markdown") or "markdown" in mime:
        return MarkdownChunker()
    elif ext == ".py" or "python" in mime:
        return PythonAstChunker()
    elif ext in (".pdf", ".docx") or "pdf" in mime or "wordprocessingml" in mime:
        return PdfDocxChunker()
    else:
        return RecursiveGenericChunker()
