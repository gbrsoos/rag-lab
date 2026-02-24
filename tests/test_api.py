"""Tests for FastAPI endpoints.

Uses FastAPI's TestClient which runs the ASGI app in-process — no server
needs to be running. External dependencies (compiled_graph, ingest_document)
are patched so tests are fast and require no API keys or local models.
"""

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from langchain_core.documents import Document

pytest.importorskip("multipart")

from rag_lab.api.main import app

client = TestClient(app)


@patch("rag_lab.api.main.compiled_graph")
def test_query_endpoint_returns_expected_fields(mock_graph: MagicMock) -> None:
    """POST /query must invoke the graph and return a well-formed QueryResponse.

    compiled_graph.invoke is mocked to return a realistic state dict so the
    endpoint's serialization logic (Document → ChunkOut, cited flag, etc.) is
    exercised without any LLM call or Chroma connection.
    """
    mock_graph.invoke.return_value = {
        "final_response": "RAG is retrieval-augmented generation.",
        "query_type": "factual",
        "retrieved_chunks": [
            Document(
                page_content="RAG combines retrieval with generation.",
                metadata={"source": "rag_paper.pdf"},
            )
        ],
        "similarity_scores": [0.15],
        "cited_chunk_indices": [0],
        "context_grade": "sufficient",
        "retrieval_attempts": 1,
        "grounding_score": 0.9,
        "grounding_pass": True,
        "node_trace": [
            "classify_query",
            "retrieve",
            "grade_context",
            "answer",
            "verify_grounding",
            "build_final_response",
        ],
    }

    response = client.post("/query", json={"query": "What is RAG?"})

    assert response.status_code == 200
    data = response.json()

    # Verify all QueryResponse fields are present and correctly populated
    assert data["final_response"] == "RAG is retrieval-augmented generation."
    assert data["query_type"] == "factual"
    assert data["context_grade"] == "sufficient"
    assert data["retrieval_attempts"] == 1
    assert data["grounding_score"] == 0.9
    assert data["grounding_pass"] is True
    assert data["node_trace"] == [
        "classify_query",
        "retrieve",
        "grade_context",
        "answer",
        "verify_grounding",
        "build_final_response",
    ]

    # Verify chunk serialization: Document → ChunkOut
    assert len(data["retrieved_chunks"]) == 1
    chunk = data["retrieved_chunks"][0]
    assert chunk["content"] == "RAG combines retrieval with generation."
    assert chunk["score"] == 0.15
    assert chunk["cited"] is True  # index 0 is in cited_chunk_indices


@patch("rag_lab.api.main.ingest_document")
def test_ingest_endpoint_returns_ingest_response(
    mock_ingest: MagicMock, tmp_path: pytest.fixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    """POST /ingest must save the file, call ingest_document, and return IngestResponse.

    ingest_document is mocked to avoid running Docling and Chroma.
    settings.uploads_dir is redirected to tmp_path so the file write succeeds
    without touching the real data/uploads/ directory.
    """
    import rag_lab.api.main as api_module

    monkeypatch.setattr(api_module.settings, "uploads_dir", tmp_path)
    mock_ingest.return_value = {"filename": "test.pdf", "chunk_count": 12}

    response = client.post(
        "/ingest",
        files={"file": ("test.pdf", b"fake pdf content", "application/pdf")},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["filename"] == "test.pdf"
    assert data["chunk_count"] == 12
    assert data["status"] == "ok"

    # Verify ingest_document was called with the saved file path
    mock_ingest.assert_called_once()
    called_path = mock_ingest.call_args[0][0]
    assert called_path.name == "test.pdf"
    assert called_path.parent == tmp_path
