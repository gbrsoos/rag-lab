from rag_lab.config import settings
from rag_lab.graph.state import GraphState


def route_after_grade(state: GraphState) -> str:
    """Routing decision after grade_context.

    Decision tree:
        sufficient                          → answer
        insufficient + attempts < max       → rewrite_query  (retry loop)
        insufficient + attempts >= max      → answer         (give up gracefully)

    The "give up gracefully" branch is intentional: after exhausting retries,
    we still generate an answer from whatever chunks we have. The answer node
    is instructed to say so if the context is insufficient, and the grounding
    node will likely flag it — giving the user a response with a low-confidence
    warning rather than a hard failure.
    """
    grade = state.get("context_grade", "insufficient")
    attempts = state.get("retrieval_attempts", 0)

    if grade == "sufficient":
        return "answer"
    if attempts < settings.max_retrieval_attempts:
        return "rewrite_query"
    return "answer"


def route_after_grounding(state: GraphState) -> str:
    """Routing decision after verify_grounding.

    Decision tree:
        grounding passed                    → build_final_response
        grounding failed + attempts < max   → rewrite_query  (targeted re-retrieval)
        grounding failed + attempts >= max  → build_final_response (with warning)

    When grounding fails and retries remain, routing back to rewrite_query
    triggers a new retrieval cycle. The answer node may have cited chunks
    that were weakly relevant — a rewritten query may surface better evidence.

    The attempt counter is shared between the two feedback loops (grade and
    grounding). This is intentional: max_retrieval_attempts is a global budget
    for the entire pipeline run, preventing infinite loops across both loops.
    """
    grounding_pass = state.get("grounding_pass", True)
    attempts = state.get("retrieval_attempts", 0)

    if grounding_pass:
        return "build_final_response"
    if attempts < settings.max_retrieval_attempts:
        return "rewrite_query"
    return "build_final_response"
