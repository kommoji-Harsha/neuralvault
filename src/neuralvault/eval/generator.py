"""Synthetic QA question generator from collection chunks."""

import json
import re
from pathlib import Path
from typing import Any, Dict, List

from neuralvault.contract import Chunk


def generate_synthetic_qa_item(chunk: Chunk, idx: int) -> Dict[str, Any]:
    """Generate a single synthetic QA dict from a Chunk."""
    lines = [line.strip() for line in chunk.text.splitlines() if line.strip()]
    first_line = lines[0] if lines else chunk.location

    # Clean first line if heading or function signature
    first_line_clean = re.sub(r"^[#\s\*]+", "", first_line).strip()
    if first_line_clean.startswith("def "):
        fn_name = first_line_clean.split("(")[0].replace("def ", "").strip()
        question = f"What does the function {fn_name} do?"
    elif first_line_clean.startswith("class "):
        cls_name = first_line_clean.split(":")[0].replace("class ", "").strip()
        question = f"What is the purpose of class {cls_name}?"
    else:
        topic = first_line_clean[:60]
        question = f"What details are provided about {topic}?"

    return {
        "id": f"synthetic_qa_{idx:03d}",
        "question": question,
        "gold_sources": [chunk.source],
        "gold_chunk_ids": [chunk.chunk_id],
        "reference_answer": chunk.text[:200],
        "synthetic": True,
    }


def generate_synthetic_dataset(
    chunks: List[Chunk],
    output_path: Path | str,
    max_questions: int = 20,
) -> List[Dict[str, Any]]:
    """Generate synthetic QA dataset file from chunks."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    qa_items: List[Dict[str, Any]] = []
    step = max(1, len(chunks) // max_questions) if chunks else 1

    idx = 1
    for i in range(0, len(chunks), step):
        if idx > max_questions:
            break
        chunk = chunks[i]
        item = generate_synthetic_qa_item(chunk, idx)
        qa_items.append(item)
        idx += 1

    with open(output_path, "w", encoding="utf-8") as f:
        for item in qa_items:
            f.write(json.dumps(item) + "\n")

    return qa_items
