from langgraph.graph import END, START, StateGraph

from rag_lab.graph.edges import route_after_grade, route_after_grounding
from rag_lab.graph.nodes import (
    answer,
    build_final_response,
    classify_query,
    grade_context,
    retrieve,
    rewrite_query,
    verify_grounding,
)
from rag_lab.graph.state import GraphState


def build_graph():
    """Assemble and compile the RAG LangGraph DAG.

    Graph structure:
        START
          └─► classify_query          (direct)
                └─► retrieve          (direct)
                      └─► grade_context  (direct)
                            ├─► answer            (if sufficient, or retries exhausted)
                            └─► rewrite_query      (if insufficient + retries remain)
                                  └─► retrieve     (direct, back into the loop)
                            answer
                              └─► verify_grounding  (direct)
                                    ├─► build_final_response  (if grounding passed)
                                    └─► rewrite_query         (if failed + retries remain)
                                          └─► retrieve        (back into the loop)
                            build_final_response
                              └─► END              (direct)

    Returns:
        A compiled LangGraph graph. Call .invoke(initial_state) to run the pipeline.

    Initial state shape expected by the caller:
        {
            "query": str,               # required
            "retrieval_attempts": 0,    # required — edge functions read this counter
        }
        node_trace is handled automatically by its operator.add reducer.
    """
    graph = StateGraph(GraphState)

    # ── Register nodes ────────────────────────────────────────────────────────
    # The string name is what edge functions return and what add_edge references.
    # Using the function's __name__ keeps names consistent with the source code.
    graph.add_node("classify_query", classify_query)
    graph.add_node("retrieve", retrieve)
    graph.add_node("grade_context", grade_context)
    graph.add_node("rewrite_query", rewrite_query)
    graph.add_node("answer", answer)
    graph.add_node("verify_grounding", verify_grounding)
    graph.add_node("build_final_response", build_final_response)

    # ── Direct edges ──────────────────────────────────────────────────────────
    graph.add_edge(START, "classify_query")
    graph.add_edge("classify_query", "retrieve")
    graph.add_edge("retrieve", "grade_context")
    graph.add_edge("answer", "verify_grounding")
    graph.add_edge("rewrite_query", "retrieve")
    graph.add_edge("build_final_response", END)

    # ── Conditional edges ─────────────────────────────────────────────────────
    # The path_map explicitly declares which nodes each return value leads to.
    # Without it, LangGraph cannot resolve the connections at compile time,
    # leaving the target nodes appearing disconnected in the graph.
    graph.add_conditional_edges(
        "grade_context",
        route_after_grade,
        {"answer": "answer", "rewrite_query": "rewrite_query"},
    )
    graph.add_conditional_edges(
        "verify_grounding",
        route_after_grounding,
        {"build_final_response": "build_final_response", "rewrite_query": "rewrite_query"},
    )

    return graph.compile()


# Module-level compiled graph — import this in the API and anywhere else
# that needs to invoke the pipeline. Compiled once at import time.
compiled_graph = build_graph()
