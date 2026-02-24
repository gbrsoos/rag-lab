import httpx
import streamlit as st

# ── Config ────────────────────────────────────────────────────────────────────

API_BASE = "http://localhost:8000"
REQUEST_TIMEOUT = 120.0  # LLM calls + retrieval can take time

# ── Page setup ────────────────────────────────────────────────────────────────

st.set_page_config(page_title="RAG Engineering Lab", layout="wide")
st.title("RAG Engineering Lab")
st.caption("Agentic RAG pipeline · LangGraph · Docling · Chroma · Claude")

# ── Helpers ───────────────────────────────────────────────────────────────────


def format_trace_line(node: str, result: dict) -> str:
    """Annotate a node name with its key output from the query result.

    node_trace is a list of node names in execution order. For each node,
    we know which field in the result it produced, so we can show
    e.g. "grade_context → sufficient" instead of just "grade_context".

    Nodes that may appear multiple times (retrieve, rewrite_query) don't
    have a single result field to show — we just display the name.
    """
    annotations = {
        "classify_query": f"classify_query → {result.get('query_type', '?')}",
        "retrieve": f"retrieve → {len(result.get('retrieved_chunks', []))} chunks",
        "grade_context": f"grade_context → {result.get('context_grade', '?')}",
        "rewrite_query": "rewrite_query → query rewritten",
        "answer": "answer → generated",
        "verify_grounding": (
            f"verify_grounding → {result.get('grounding_score', 0.0):.2f} "
            f"{'✅' if result.get('grounding_pass') else '❌'}"
        ),
        "build_final_response": "build_final_response → done",
    }
    return annotations.get(node, node)


# ── Upload section ────────────────────────────────────────────────────────────

st.header("Document Upload")

uploaded_file = st.file_uploader(
    "Upload a document", type=["pdf", "docx", "md"], label_visibility="collapsed"
)

if st.button("Ingest", disabled=uploaded_file is None):
    with st.spinner(f"Ingesting {uploaded_file.name}…"):
        try:
            response = httpx.post(
                f"{API_BASE}/ingest",
                files={
                    "file": (
                        uploaded_file.name,
                        uploaded_file.getvalue(),
                        uploaded_file.type or "application/octet-stream",
                    )
                },
                timeout=REQUEST_TIMEOUT,
            )
            response.raise_for_status()
            data = response.json()
            st.success(
                f"✅ **{data['filename']}** ingested — {data['chunk_count']} chunks stored."
            )

            # Track ingested docs in session state so the list persists
            if "ingested_docs" not in st.session_state:
                st.session_state.ingested_docs = []
            st.session_state.ingested_docs.append(data["filename"])

        except httpx.ConnectError:
            st.error("Cannot reach the API. Is `uvicorn rag_lab.api.main:app` running?")
        except httpx.HTTPStatusError as e:
            st.error(f"Ingest failed: {e.response.json().get('detail', str(e))}")

# Show previously ingested docs in this session
if st.session_state.get("ingested_docs"):
    st.caption(
        "Ingested this session: " + ", ".join(st.session_state.ingested_docs)
    )

# ── Query section ─────────────────────────────────────────────────────────────

st.header("Query")

query_text = st.text_input("Ask a question about your documents", label_visibility="collapsed", placeholder="Type your question here…")

if st.button("Ask", disabled=not query_text.strip()):
    with st.spinner("Running pipeline…"):
        try:
            response = httpx.post(
                f"{API_BASE}/query",
                json={"query": query_text.strip()},
                timeout=REQUEST_TIMEOUT,
            )
            response.raise_for_status()
            st.session_state.last_result = response.json()
            st.session_state.last_query = query_text.strip()

        except httpx.ConnectError:
            st.error("Cannot reach the API. Is `uvicorn rag_lab.api.main:app` running?")
        except httpx.HTTPStatusError as e:
            st.error(f"Query failed: {e.response.json().get('detail', str(e))}")

# ── Results ───────────────────────────────────────────────────────────────────

if "last_result" in st.session_state:
    result = st.session_state["last_result"]

    st.divider()
    st.subheader(f"Results for: _{st.session_state.get('last_query', '')}_")

    left_col, right_col = st.columns([1, 2])

    # ── Left: Pipeline trace ──────────────────────────────────────────────────
    with left_col:
        st.markdown("**Pipeline Trace**")
        node_trace = result.get("node_trace", [])
        for i, node in enumerate(node_trace):
            prefix = "└─" if i == len(node_trace) - 1 else "├─"
            st.markdown(f"`{prefix}` {format_trace_line(node, result)}")
        if not node_trace:
            st.caption("No trace available.")

    # ── Right: Answer ─────────────────────────────────────────────────────────
    with right_col:
        st.markdown("**Answer**")
        st.write(result.get("final_response", "No response."))

        grounding_score = result.get("grounding_score", 0.0)
        grounding_pass = result.get("grounding_pass", False)
        grounding_label = "✅ passed" if grounding_pass else "❌ failed"
        st.caption(
            f"Grounding score: **{grounding_score:.2f}** — {grounding_label} · "
            f"Retrieval attempts: {result.get('retrieval_attempts', 0)} · "
            f"Query type: {result.get('query_type', '?')}"
        )

        cited = result.get("cited_chunk_indices", [])
        if cited:
            st.caption(f"Cited chunks: {cited}")

    # ── Retrieved chunks ──────────────────────────────────────────────────────
    st.divider()
    st.markdown("**Retrieved Chunks**")
    st.caption(
        "Scores are cosine distances — lower means more similar to the query."
    )

    chunks = result.get("retrieved_chunks", [])
    if not chunks:
        st.caption("No chunks retrieved.")
    else:
        for i, chunk in enumerate(chunks):
            cited_tag = " · ✅ cited" if chunk["cited"] else ""
            source = chunk["metadata"].get("source", "unknown")
            label = f"[{i}] {source} — score: {chunk['score']:.4f}{cited_tag}"

            with st.expander(label, expanded=chunk["cited"]):
                st.write(chunk["content"])
                if chunk["metadata"]:
                    st.json(chunk["metadata"], expanded=False)
