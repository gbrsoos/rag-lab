import operator
from typing import Annotated, TypedDict

from langchain_core.documents import Document


class GraphState(TypedDict, total=False):
    """State contract for the RAG LangGraph DAG.

    Every node receives the full state and returns a *partial* dict containing
    only the keys it modifies. LangGraph merges those partial updates back into
    the state before passing it to the next node.

    Fields annotated with `Annotated[T, operator.add]` use a *reducer*: instead
    of replacing the old value, LangGraph calls operator.add(old, new) to merge.
    This is how node_trace accumulates without any node needing to read the
    current list first.

    `total=False` makes all keys optional at the TypedDict level. In practice,
    `query` is always present (set by the caller before graph.invoke()), and
    every other field is populated progressively as nodes execute.
    """

    # ── Input ─────────────────────────────────────────────────────────────────
    query: str  # The original user query; never modified after entry

    # ── Classification ────────────────────────────────────────────────────────
    # Set by: classify_query
    # Values: "factual" | "comparative" | "summarization" | "out_of_scope"
    query_type: str

    # ── Retrieval ─────────────────────────────────────────────────────────────
    # Set by: retrieve (and rewrite_query sets rewritten_query)
    rewritten_query: str | None  # None until the rewrite loop fires
    retrieval_strategy: str  # "dense" in Phase 1; "bm25" / "hybrid" in Phase 2
    retrieved_chunks: list[Document]  # Top-k chunks from the vector store
    similarity_scores: list[float]  # Parallel list: scores[i] corresponds to chunks[i]

    # ── Grading ───────────────────────────────────────────────────────────────
    # Set by: grade_context
    # Values: "sufficient" | "insufficient"
    context_grade: str

    # Set by: retrieve (incremented each call), initialized to 0 by the caller
    retrieval_attempts: int

    # ── Answer ────────────────────────────────────────────────────────────────
    # Set by: answer
    answer: str
    # Indices into retrieved_chunks that the LLM cited when generating the answer.
    # Using indices (not Document copies) keeps the state lean and avoids
    # duplicating chunk content; the UI reconstructs cited chunks via these indices.
    cited_chunk_indices: list[int]

    # ── Grounding ─────────────────────────────────────────────────────────────
    # Set by: verify_grounding
    grounding_score: float  # 0.0–1.0; how well the answer is supported by cited chunks
    grounding_pass: bool  # True if grounding_score >= settings.grounding_threshold

    # ── Output ────────────────────────────────────────────────────────────────
    # Set by: build_final_response
    # This is the string shown to the user. If grounding failed, a warning is
    # appended here; the raw `answer` field is always preserved unchanged.
    final_response: str

    # ── Metadata / observability ──────────────────────────────────────────────
    # Accumulates the name of each node as it executes, in order.
    # The Annotated reducer (operator.add) means each node returns
    # {"node_trace": ["its_own_name"]} and LangGraph concatenates automatically.
    # The UI uses this list to render the pipeline trace stepper.
    node_trace: Annotated[list[str], operator.add]
