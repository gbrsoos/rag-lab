from langchain_core.documents import Document

from rag_lab.config import settings
from rag_lab.ingestion.pipeline import get_vector_store


def retrieve_chunks(query: str) -> tuple[list[Document], list[float]]:
    """Run dense similarity search against the Chroma collection.

    Uses cosine distance (Chroma's default). The returned scores are
    cosine *distances*, not similarities: lower = more relevant.
        0.0  →  identical vectors
        1.0  →  orthogonal (unrelated)
        2.0  →  opposite direction

    Args:
        query: The search string (original query or rewritten query).

    Returns:
        A tuple of (chunks, scores) where chunks[i] and scores[i] correspond.
        Length is at most settings.retrieval_top_k; may be shorter if the
        collection has fewer documents.
    """
    vector_store = get_vector_store()
    results: list[tuple[Document, float]] = vector_store.similarity_search_with_score(
        query=query,
        k=settings.retrieval_top_k,
    )

    # Unzip list-of-tuples into two parallel lists.
    # The node stores these separately in GraphState so the UI can display
    # scores alongside each chunk without unpacking tuples in template code.
    if not results:
        return [], []

    chunks, scores = zip(*results)
    return list(chunks), list(scores)
