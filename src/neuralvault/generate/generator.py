"""Answer generators with citation tracking and insufficient context handling."""

from typing import List, Optional

from neuralvault.config import guard_offline
from neuralvault.contract import AskResponse, Chunk
from neuralvault.interfaces import Generator


class MockGenerator(Generator):
    """Deterministic offline mock generator for unit tests."""

    def __init__(self, min_score_threshold: float = 0.001) -> None:
        self.min_score_threshold = min_score_threshold

    def generate(self, query: str, chunks: List[Chunk]) -> AskResponse:
        valid_chunks = [c for c in chunks if c.score >= self.min_score_threshold]

        if not valid_chunks:
            return AskResponse(
                answer="Insufficient relevant context found to answer the query.",
                citations=[],
                note="no relevant passages found",
                insufficient_context=True,
            )

        citations_text = " ".join(f"[{c.chunk_id}]" for c in valid_chunks)
        answer = f"Based on retrieved context {citations_text}:\n"
        for c in valid_chunks:
            answer += f"- {c.text[:100]}...\n"

        return AskResponse(
            answer=answer.strip(),
            citations=valid_chunks,
            note=None,
            insufficient_context=False,
        )


class OpenAIGenerator(Generator):
    """OpenAI-compatible HTTP generator with citation instructions."""

    def __init__(
        self,
        base_url: str = "http://localhost:11434/v1",
        api_key: str = "ollama",
        model: str = "llama3",
        min_score_threshold: float = 0.001,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.min_score_threshold = min_score_threshold

    def generate(self, query: str, chunks: List[Chunk]) -> AskResponse:
        guard_offline("OpenAI-compatible HTTP text generation call")
        valid_chunks = [c for c in chunks if c.score >= self.min_score_threshold]

        if not valid_chunks:
            return AskResponse(
                answer="Insufficient relevant context found to answer the query.",
                citations=[],
                note="no relevant passages found",
                insufficient_context=True,
            )

        import requests

        context_lines = []
        for c in valid_chunks:
            context_lines.append(f"[{c.chunk_id}] ({c.source} > {c.location}):\n{c.text}\n")

        context_str = "\n".join(context_lines)
        system_prompt = (
            "You are a helpful assistant. Answer the user's question using ONLY the context.\n"
            "Cite chunk IDs (e.g. [chunk_id]) whenever referencing facts from the context.\n"
            "If the context does not contain sufficient information, state that clearly."
        )
        user_prompt = f"Context:\n{context_str}\n\nQuestion: {query}"

        url = f"{self.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0.2,
        }

        response = requests.post(url, json=payload, headers=headers, timeout=60)
        response.raise_for_status()
        data = response.json()
        ans_text = data["choices"][0]["message"]["content"]

        return AskResponse(
            answer=ans_text,
            citations=valid_chunks,
            note=None,
            insufficient_context=False,
        )


def get_generator(generator_type: str = "mock", **kwargs: Optional[dict]) -> Generator:
    """Factory function for answer generators."""
    if generator_type in ("mock", "test"):
        return MockGenerator(**kwargs)  # type: ignore[arg-type]
    elif generator_type in ("openai", "http"):
        return OpenAIGenerator(**kwargs)  # type: ignore[arg-type]
    elif generator_type in ("none", "disabled"):
        return MockGenerator(min_score_threshold=1.0)
    else:
        return MockGenerator()
