from langchain_anthropic import ChatAnthropic
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate

from rag_lab.config import settings
from rag_lab.graph.state import GraphState
from rag_lab.prompts.answer import AnswerOutput, answer_prompt
from rag_lab.prompts.classify import ClassifyOutput, classify_prompt
from rag_lab.prompts.grade import GradeOutput, grade_prompt
from rag_lab.prompts.grounding import GroundingOutput, grounding_prompt
from rag_lab.retrieval.dense import retrieve_chunks

# ── Shared LLM instance ───────────────────────────────────────────────────────
# Created once at module load. ChatAnthropic is a thin HTTP client wrapper —
# no model weights, no heavy init. All nodes share this instance so they all
# use the same model/temperature and we don't construct multiple clients.

_llm = ChatAnthropic(
    model=settings.llm_model,
    temperature=settings.llm_temperature,
    api_key=settings.anthropic_api_key,
)

# ── Rewrite prompt (defined here — too simple to warrant its own file) ────────
# The rewrite node does not use structured output: we just want a plain string
# back (the new query). For unstructured string output, ChatAnthropic returns
# an AIMessage; we read .content to get the text.

_rewrite_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            (
                "You are a search query optimizer for a document retrieval system. "
                "The current query failed to retrieve sufficient information. "
                "Rewrite the query using different keywords or a different angle "
                "to improve retrieval. Return only the rewritten query, nothing else."
            ),
        ),
        (
            "human",
            (
                "Original query: {query}\n\n"
                "Retrieval attempt: {attempt}\n\n"
                "Chunks retrieved so far (which were insufficient):\n{context}"
            ),
        ),
    ]
)


# ── Private helper ─────────────────────────────────────────────────────────────

def _format_chunks(chunks: list[Document]) -> str:
    """Format a list of chunks as a numbered string for prompt injection.

    Produces:
        [0] chunk text...
        [1] chunk text...

    The indices must stay consistent across grade, rewrite, and answer nodes
    because AnswerOutput.cited_chunk_indices references these same numbers.
    """
    return "\n\n".join(f"[{i}] {doc.page_content}" for i, doc in enumerate(chunks))


# ── Nodes ──────────────────────────────────────────────────────────────────────

def classify_query(state: GraphState) -> dict:
    """Classify the query into factual / comparative / summarization / out_of_scope.

    In Phase 1, the result is stored in state but does not affect routing —
    all queries proceed to retrieval. In Phase 2+, this will drive strategy
    selection (e.g. summarization queries may use a larger top-k).
    """
    chain = classify_prompt | _llm.with_structured_output(ClassifyOutput)
    result: ClassifyOutput = chain.invoke({"query": state["query"]})
    return {
        "query_type": result.query_type,
        "node_trace": ["classify_query"],
    }


def retrieve(state: GraphState) -> dict:
    """Run dense similarity search and increment the attempt counter.

    Uses the rewritten query if one exists (set by rewrite_query on a previous
    loop iteration), otherwise falls back to the original query.

    retrieval_attempts is incremented here rather than in a separate node so
    the edge condition after grade_context can read the updated count.
    """
    query = state.get("rewritten_query") or state["query"]
    chunks, scores = retrieve_chunks(query)
    return {
        "retrieved_chunks": chunks,
        "similarity_scores": scores,
        "retrieval_strategy": "dense",
        "retrieval_attempts": state.get("retrieval_attempts", 0) + 1,
        "node_trace": ["retrieve"],
    }


def grade_context(state: GraphState) -> dict:
    """Evaluate whether the retrieved chunks are sufficient to answer the query.

    The grade ("sufficient" / "insufficient") is used by the conditional edge
    after this node to decide whether to proceed to answer or trigger a rewrite.
    """
    context = _format_chunks(state["retrieved_chunks"])
    chain = grade_prompt | _llm.with_structured_output(GradeOutput)
    result: GradeOutput = chain.invoke(
        {"query": state["query"], "context": context}
    )
    return {
        "context_grade": result.grade,
        "node_trace": ["grade_context"],
    }


def rewrite_query(state: GraphState) -> dict:
    """Rewrite the query to improve retrieval on the next attempt.

    The LLM sees the original query, the chunks that were retrieved (and found
    insufficient), and the attempt number. The attempt number gives the LLM
    context about how many times retrieval has already failed, which can
    encourage more aggressive reformulation on later attempts.

    Returns a plain string (no structured output) — we just need the new query.
    """
    context = _format_chunks(state.get("retrieved_chunks", []))
    chain = _rewrite_prompt | _llm
    result = chain.invoke(
        {
            "query": state["query"],
            "attempt": state.get("retrieval_attempts", 1),
            "context": context,
        }
    )
    return {
        "rewritten_query": result.content.strip(),
        "node_trace": ["rewrite_query"],
    }


def answer(state: GraphState) -> dict:
    """Generate an answer from the retrieved chunks.

    Passes query_type so the LLM can tailor its response style (e.g. a
    comparison for comparative queries, a summary for summarization queries).

    Returns both the answer text and cited_chunk_indices — the zero-based
    positions of the chunks the LLM actually used. These indices are later
    used by verify_grounding to check only the cited evidence.
    """
    context = _format_chunks(state["retrieved_chunks"])
    chain = answer_prompt | _llm.with_structured_output(AnswerOutput)
    result: AnswerOutput = chain.invoke(
        {
            "query": state["query"],
            "query_type": state.get("query_type", "factual"),
            "context": context,
        }
    )
    return {
        "answer": result.answer,
        "cited_chunk_indices": result.cited_chunk_indices,
        "node_trace": ["answer"],
    }


def verify_grounding(state: GraphState) -> dict:
    """Score how well the answer is supported by the chunks it cited.

    Only the cited chunks are passed to this node — not all retrieved chunks.
    This focuses the LLM's attention on the specific evidence the answer
    claimed to use, making the grounding check more precise.

    pass/fail is computed in Python by comparing the score against
    settings.grounding_threshold, not by the LLM. This keeps threshold
    tuning out of the prompt.
    """
    chunks = state["retrieved_chunks"]
    cited_indices = state.get("cited_chunk_indices", [])

    # Guard against out-of-range indices returned by the answer LLM
    cited_context = "\n\n".join(
        f"[{i}] {chunks[i].page_content}"
        for i in cited_indices
        if i < len(chunks)
    ) or "(no chunks cited)"

    chain = grounding_prompt | _llm.with_structured_output(GroundingOutput)
    result: GroundingOutput = chain.invoke(
        {
            "query": state["query"],
            "answer": state["answer"],
            "cited_context": cited_context,
        }
    )
    return {
        "grounding_score": result.grounding_score,
        "grounding_pass": result.grounding_score >= settings.grounding_threshold,
        "node_trace": ["verify_grounding"],
    }


def build_final_response(state: GraphState) -> dict:
    """Package the final response shown to the user.

    If grounding passed, the answer is returned as-is.
    If grounding failed, a warning is appended. The raw `answer` field in
    state is never modified — only `final_response` gets the warning.
    This means the UI can always display both the clean answer and the
    decorated response separately.
    """
    answer_text = state["answer"]
    if state.get("grounding_pass", True):
        final_response = answer_text
    else:
        score = state.get("grounding_score", 0.0)
        final_response = (
            f"{answer_text}\n\n"
            f"⚠️ Low grounding confidence (score: {score:.2f}). "
            "This answer may not be fully supported by the source documents."
        )
    return {
        "final_response": final_response,
        "node_trace": ["build_final_response"],
    }
