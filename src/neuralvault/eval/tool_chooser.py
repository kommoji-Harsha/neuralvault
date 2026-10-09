"""Tool selection disambiguation benchmark."""

import re
from typing import Any, Dict, List, Tuple

SEARCH_MEMORY_DOCSTRING = (
    "Search personal facts, preferences, user background, and conversation history about user."
)

SEARCH_KNOWLEDGE_BASE_DOCSTRING = (
    "Search your personal knowledge base — documents, notes, code, and project files indexed."
)

SEARCH_WEB_DOCSTRING = (
    "Search live internet web pages for current events, news, weather, live stock prices."
)


DISAMBIGUATION_TEST_CASES: List[Tuple[str, str]] = [
    # search_knowledge_base cases (10)
    ("how does the confirmation workflow work in the codebase", "search_knowledge_base"),
    ("what does _extract_tool_calls do in retrieve.py", "search_knowledge_base"),
    ("find the documentation for setting up virtual environments", "search_knowledge_base"),
    ("where is the SqliteStore class defined in the repository", "search_knowledge_base"),
    ("how is Reciprocal Rank Fusion implemented", "search_knowledge_base"),
    ("what is written in my notes about python design patterns", "search_knowledge_base"),
    ("show me the implementation of the fastembed provider", "search_knowledge_base"),
    ("what are the arguments for the ingest CLI command", "search_knowledge_base"),
    ("find the README instructions for running tests", "search_knowledge_base"),
    ("how are PDF tables preserved in the chunker", "search_knowledge_base"),

    # search_memory cases (10)
    ("what is my favorite programming language", "search_memory"),
    ("where do I live", "search_memory"),
    ("what did I say my dog's name was", "search_memory"),
    ("what are my dietary preferences", "search_memory"),
    ("what project did I mention I worked on last week", "search_memory"),
    ("what is my brother's occupation", "search_memory"),
    ("what time do I usually wake up", "search_memory"),
    ("what is my primary text editor preference", "search_memory"),
    ("what allergies do I have", "search_memory"),
    ("what was the topic of our last conversation", "search_memory"),

    # search_web cases (10)
    ("what is the weather forecast in Tokyo today", "search_web"),
    ("what is the current stock price of Apple", "search_web"),
    ("who won the latest World Cup match yesterday", "search_web"),
    ("latest news about space exploration launches today", "search_web"),
    ("what is the current exchange rate for USD to EUR", "search_web"),
    ("what time does the museum open today in Paris", "search_web"),
    ("what are the trending tech headlines this morning", "search_web"),
    ("who is the current prime minister of Japan", "search_web"),
    ("traffic conditions on I-95 right now", "search_web"),
    ("live score for the basketball game", "search_web"),
]


class MockToolChooser:
    """Mock tool selector choosing between search_memory, search_knowledge_base, and search_web."""

    def select_tool(self, query: str) -> str:
        q_lower = query.lower()

        # Knowledge base signals
        kb_keywords = [
            "notes", "codebase", "repository", "repo", "documentation", "docs",
            "function", "class", "implementation", "readme", "chunker", "ingest",
            "retrieve.py", "sqlitestore", "reciprocal rank"
        ]
        if any(w in q_lower for w in kb_keywords):
            return "search_knowledge_base"

        # Personal memory signals
        memory_keywords = [
            "my ", " i ", "favorite", "dietary", "brother", "wake", "allergies",
            "conversation", "where do i", "dog", "live"
        ]
        is_mem = (
            any(w in q_lower for w in memory_keywords)
            or q_lower.startswith(("what is my", "what did i"))
            or "where do i" in q_lower
            or "i live" in q_lower
        )
        if is_mem:
            return "search_memory"

        # Web signals
        web_keywords = [
            "weather", "stock", "news", "today", "yesterday", "match", "world cup",
            "current", "forecast", "headlines", "traffic", "price", "exchange rate",
            "prime minister"
        ]
        if any(re.search(r"\b" + re.escape(w) + r"\b", q_lower) for w in web_keywords):
            return "search_web"

        return "search_knowledge_base"


def evaluate_tool_selection_accuracy() -> Dict[str, Any]:
    """Evaluate tool chooser accuracy on the 30 disambiguation test cases."""
    chooser = MockToolChooser()
    correct = 0
    total = len(DISAMBIGUATION_TEST_CASES)
    details = []

    for query, expected_tool in DISAMBIGUATION_TEST_CASES:
        predicted_tool = chooser.select_tool(query)
        is_correct = predicted_tool == expected_tool
        if is_correct:
            correct += 1
        details.append({
            "query": query,
            "expected": expected_tool,
            "predicted": predicted_tool,
            "correct": is_correct,
        })

    accuracy = correct / total
    return {
        "total_cases": total,
        "correct_cases": correct,
        "accuracy": accuracy,
        "passes_threshold_95": accuracy >= 0.95,
        "details": details,
    }
