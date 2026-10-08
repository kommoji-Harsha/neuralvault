"""Post-retrieval processing (Parent Expansion, MMR, Thresholds)."""

from typing import List, Optional

from neuralvault.contract import Chunk
from neuralvault.store.sqlite_store import SqliteStore


def text_jaccard_similarity(text1: str, text2: str) -> float:
    """Compute word-level Jaccard similarity between two texts."""
    set1 = set(text1.lower().split())
    set2 = set(text2.lower().split())
    if not set1 or not set2:
        return 0.0
    union = set1.union(set2)
    intersection = set1.intersection(set2)
    return len(intersection) / len(union)


def mmr_deduplicate(
    chunks: List[Chunk],
    lambda_mult: float = 0.7,
    max_similarity: float = 0.8,
) -> List[Chunk]:
    """Maximal Marginal Relevance (MMR) deduplication to eliminate near-duplicate chunks."""
    if not chunks:
        return []

    selected: List[Chunk] = []
    unselected = list(chunks)

    # Always select highest-scoring initial chunk
    unselected.sort(key=lambda c: c.score, reverse=True)
    selected.append(unselected.pop(0))

    while unselected:
        best_candidate: Optional[Chunk] = None
        best_mmr_score = float("-inf")

        for cand in unselected:
            # Maximum similarity between candidate and already selected chunks
            max_sim = max(text_jaccard_similarity(cand.text, sel.text) for sel in selected)
            if max_sim >= max_similarity:
                continue

            # MMR equation: lambda * relevance - (1 - lambda) * max_similarity
            mmr_score = lambda_mult * cand.score - (1.0 - lambda_mult) * max_sim
            if mmr_score > best_mmr_score:
                best_mmr_score = mmr_score
                best_candidate = cand

        if best_candidate is None:
            break

        selected.append(best_candidate)
        unselected.remove(best_candidate)

    return selected


def apply_relevance_threshold(chunks: List[Chunk], min_score: float = 0.001) -> List[Chunk]:
    """Filter out chunks scoring below minimum relevance threshold."""
    return [c for c in chunks if c.score >= min_score]


def parent_expand_chunks(
    chunks: List[Chunk], store: Optional[SqliteStore] = None
) -> List[Chunk]:
    """Expand matching chunks with parent chunk context when available."""
    if not store or not chunks:
        return chunks

    expanded: List[Chunk] = []
    for chunk in chunks:
        doc_id = chunk.metadata.get("doc_id")

        if doc_id and "part" in chunk.location:
            with store._get_connection() as conn:
                cur = conn.cursor()
                cur.execute(
                    "SELECT text FROM chunks WHERE doc_id = ? "
                    "AND location NOT LIKE '%part%' LIMIT 1",
                    (doc_id,),
                )
                row = cur.fetchone()
                if row and row["text"]:
                    parent_text = row["text"]
                    new_disp_text = (
                        f"{parent_text}\n\n... [Matching Sub-block] ...\n\n{chunk.text}"
                    )
                    new_emb_text = f"{chunk.location}\n\n{new_disp_text}"
                    expanded.append(
                        Chunk(
                            chunk_id=chunk.chunk_id,
                            text=new_disp_text,
                            source=chunk.source,
                            location=f"{chunk.location} (Parent Expanded)",
                            score=chunk.score,
                            embedding_text=new_emb_text,
                            metadata=dict(chunk.metadata),
                        )
                    )
                    continue

        expanded.append(chunk)

    return expanded
