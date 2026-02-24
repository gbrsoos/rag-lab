"""Tests for LangGraph node functions.

Nodes that call the LLM are tested by patching `rag_lab.graph.nodes._llm`.
The real ChatPromptTemplate runs normally (it's pure Python string formatting),
and only the LLM step at the end of the chain is replaced with a mock.

Chain execution flow:
    chain = prompt | _llm.with_structured_output(OutputModel)
    chain.invoke(inputs)
        → prompt.invoke(inputs)          # real, formats the messages
        → mock_structured.invoke(msgs)   # intercepted here, returns our value
"""

from unittest.mock import MagicMock, patch

from langchain_core.documents import Document
from langchain_core.messages import AIMessage

from rag_lab.graph.nodes import rewrite_query, verify_grounding
from rag_lab.prompts.grounding import GroundingOutput


@patch("rag_lab.graph.nodes._llm")
def test_verify_grounding_pass_when_score_above_threshold(mock_llm: MagicMock) -> None:
    """grounding_pass must be True when the LLM returns a score above threshold.

    settings.grounding_threshold defaults to 0.7. A score of 0.85 should
    produce grounding_pass=True and be stored accurately in state.
    """
    mock_structured = MagicMock()
    mock_structured.invoke.return_value = GroundingOutput(
        grounding_score=0.85,
        reasoning="Every claim is directly supported by the cited chunk.",
    )
    mock_llm.with_structured_output.return_value = mock_structured

    state = {
        "query": "What is retrieval-augmented generation?",
        "answer": "RAG combines a retrieval step with a generation step.",
        "retrieved_chunks": [
            Document(
                page_content="RAG combines retrieval with generation.",
                metadata={"source": "rag_paper.pdf"},
            )
        ],
        "cited_chunk_indices": [0],
    }

    result = verify_grounding(state)

    assert result["grounding_score"] == 0.85
    assert result["grounding_pass"] is True
    assert "verify_grounding" in result["node_trace"]


@patch("rag_lab.graph.nodes._llm")
def test_verify_grounding_fail_when_score_below_threshold(mock_llm: MagicMock) -> None:
    """grounding_pass must be False when the LLM returns a score below threshold.

    A score of 0.4 is below the 0.7 threshold. The pass/fail decision is made
    in Python (not by the LLM), so this test verifies that comparison logic.
    """
    mock_structured = MagicMock()
    mock_structured.invoke.return_value = GroundingOutput(
        grounding_score=0.4,
        reasoning="The answer contains claims not found in the cited chunk.",
    )
    mock_llm.with_structured_output.return_value = mock_structured

    state = {
        "query": "What is the capital of France?",
        "answer": "Paris, which has a population of 2 million.",
        "retrieved_chunks": [
            Document(
                page_content="France is a country in Western Europe.",
                metadata={"source": "geography.pdf"},
            )
        ],
        "cited_chunk_indices": [0],
    }

    result = verify_grounding(state)

    assert result["grounding_score"] == 0.4
    assert result["grounding_pass"] is False


@patch("rag_lab.graph.nodes._llm")
def test_rewrite_query_returns_stripped_rewritten_query(mock_llm: MagicMock) -> None:
    """rewrite_query must return the LLM's output as rewritten_query, stripped.

    The node appends .strip() to the LLM output to remove accidental leading/
    trailing whitespace. This test verifies both that the content is captured
    and that stripping happens.
    """
    mock_llm.invoke.return_value = AIMessage(
        content="  What are the main architectural components of a RAG system?  "
    )

    state = {
        "query": "how does rag work?",
        "retrieval_attempts": 1,
        "retrieved_chunks": [],
    }

    result = rewrite_query(state)

    assert (
        result["rewritten_query"]
        == "What are the main architectural components of a RAG system?"
    )
    assert "rewrite_query" in result["node_trace"]
