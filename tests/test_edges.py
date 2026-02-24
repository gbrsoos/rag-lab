"""Tests for graph edge routing functions.

Edge functions are pure Python: they take a state dict and return a string.
No LLM, no Chroma, no external dependencies — straightforward to test.
"""

from rag_lab.graph.edges import route_after_grade


def test_route_after_grade_routes_to_answer_when_sufficient() -> None:
    """When context is graded sufficient, the graph must route to 'answer'.

    This is the happy path: retrieval found relevant chunks, the LLM confirmed
    they are sufficient, so we proceed directly to answer generation without
    entering the rewrite loop.
    """
    state = {
        "query": "What is RAG?",
        "context_grade": "sufficient",
        "retrieval_attempts": 1,
    }

    result = route_after_grade(state)

    assert result == "answer"
