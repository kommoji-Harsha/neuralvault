"""Extractive context compressor within character length budget."""

import re
from typing import List

from neuralvault.contract import Chunk
from neuralvault.interfaces import Compressor


class ExtractiveCompressor(Compressor):
    """Extractive context compressor selecting query-relevant lines/sentences within budget."""

    def __init__(self, default_max_chars: int = 1500) -> None:
        self.default_max_chars = default_max_chars

    def compress(self, query: str, chunks: List[Chunk], max_chars: int = 1500) -> List[Chunk]:
        """Compress candidate chunks to fit within total character budget.

        Preserves source, location, and code blocks without mid-block splitting.
        """
        if not chunks:
            return []

        budget = max_chars if max_chars > 0 else self.default_max_chars
        query_words = set(re.findall(r"\w+", query.lower()))

        compressed_chunks: List[Chunk] = []
        allocated_chars = 0

        # Budget allocated across chunks proportional to score or count
        chunk_budget = max(200, budget // max(1, len(chunks)))

        for chunk in chunks:
            if allocated_chars >= budget:
                break

            remaining_budget = budget - allocated_chars
            target_budget = min(chunk_budget, remaining_budget)

            if len(chunk.text) <= target_budget:
                comp_text = chunk.text
            elif "```" in chunk.text or "def " in chunk.text or "class " in chunk.text:
                comp_text = self._compress_code(chunk.text, query_words, target_budget)
            else:
                comp_text = self._compress_prose(chunk.text, query_words, target_budget)

            if comp_text:
                comp_chunk = Chunk(
                    chunk_id=f"{chunk.chunk_id}_comp",
                    text=comp_text,
                    source=chunk.source,
                    location=chunk.location,
                    score=chunk.score,
                    embedding_text=chunk.embedding_text,
                    metadata=dict(chunk.metadata),
                )
                compressed_chunks.append(comp_chunk)
                allocated_chars += len(comp_text)

        return compressed_chunks

    def _compress_code(self, text: str, query_words: set[str], budget: int) -> str:
        lines = text.splitlines()
        kept_lines = []
        curr_len = 0

        # Always keep function signatures, docstrings, and matching lines
        in_code_block = False
        code_block_acc = []

        for line in lines:
            if line.strip().startswith("```"):
                in_code_block = not in_code_block
                code_block_acc.append(line)
                if not in_code_block:
                    block_str = "\n".join(code_block_acc)
                    if curr_len + len(block_str) <= budget:
                        kept_lines.append(block_str)
                        curr_len += len(block_str)
                    code_block_acc = []
                continue

            if in_code_block:
                code_block_acc.append(line)
                continue

            line_words = set(re.findall(r"\w+", line.lower()))
            is_header = line.strip().startswith(("def ", "class ", "import ", '"""', "#"))
            is_match = bool(query_words.intersection(line_words))

            if is_header or is_match:
                if curr_len + len(line) + 1 <= budget:
                    kept_lines.append(line)
                    curr_len += len(line) + 1

        if not kept_lines:
            return text[:budget]

        return "\n".join(kept_lines)

    def _compress_prose(self, text: str, query_words: set[str], budget: int) -> str:
        sentences = re.split(r"(?<=[.!?])\s+", text)
        scored_sentences = []

        for idx, sent in enumerate(sentences):
            words = set(re.findall(r"\w+", sent.lower()))
            overlap = len(query_words.intersection(words))
            scored_sentences.append((overlap, idx, sent))

        # Sort by overlap descending, then sentence order ascending
        scored_sentences.sort(key=lambda x: (x[0], -x[1]), reverse=True)

        selected_indices = []
        curr_len = 0

        for overlap, idx, sent in scored_sentences:
            if curr_len + len(sent) + 1 <= budget:
                selected_indices.append(idx)
                curr_len += len(sent) + 1

        selected_indices.sort()
        selected_sentences = [sentences[i] for i in selected_indices]

        if not selected_sentences:
            return text[:budget]

        return " ".join(selected_sentences)
