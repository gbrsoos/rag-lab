# Phase 1 Spec — Core Agentic Pipeline

## Goal

A working end-to-end agentic RAG pipeline: upload documents, ask questions, see every intermediate step, with full LangSmith tracing. Dense retrieval only (BM25 and hybrid come in Phase 2).

---

## Project Structure

```
rag-engineering-lab/
├── ARCHITECTURE.md
├── CLAUDE.md
├── pyproject.toml
├── .env                        # API keys (gitignored)
├── .env.example                # Template with placeholder values
├── .gitignore
├── README.md
├── docs/
│   └── phase-1-spec.md
├── data/
│   ├── uploads/                # Raw uploaded documents
│   └── chroma/                 # Persisted Chroma DB
├── src/
│   └── rag_lab/
│       ├── __init__.py
│       ├── config.py           # Settings, env vars, constants
│       ├── graph/
│       │   ├── __init__.py
│       │   ├── state.py        # GraphState TypedDict
│       │   ├── nodes.py        # All node functions
│       │   ├── edges.py        # Conditional edge functions
│       │   └── builder.py      # Assembles and compiles the graph
│       ├── retrieval/
│       │   ├── __init__.py
│       │   └── dense.py        # Chroma setup via langchain-chroma
│       ├── ingestion/
│       │   ├── __init__.py
│       │   └── pipeline.py     # Docling loader + chunker config
│       ├── prompts/
│       │   ├── __init__.py
│       │   ├── classify.py     # Query classification prompt
│       │   ├── grade.py        # Context grading prompt
│       │   ├── answer.py       # Answer generation prompt
│       │   └── grounding.py    # Grounding verification prompt
│       ├── api/
│       │   ├── __init__.py
│       │   └── main.py         # FastAPI app
│       └── ui/
│           ├── __init__.py
│           └── app.py          # Streamlit app
```

---

## Dependencies

Core dependencies (verify latest compatible versions at implementation time):

- `langchain-core`, `langchain-anthropic`, `langchain-huggingface`, `langchain-chroma`
- `langchain-docling`, `docling`
- `langgraph`
- `langsmith`
- `chromadb`
- `fastapi`, `uvicorn`
- `streamlit`
- `pydantic-settings`
- `python-dotenv`
- `httpx` (UI → API communication)

Dev: `ruff`, `pytest`

Use `pyproject.toml` for dependency management. Pin minimum versions, not exact.

**Note:** `docling` has a heavy install footprint. If it causes issues, the fallback is `PyPDFLoader` from `langchain-community` + `RecursiveCharacterTextSplitter` from `langchain-text-splitters`. But try Docling first.

---

## Environment Variables

```
ANTHROPIC_API_KEY=...
LANGCHAIN_API_KEY=...           # LangSmith free tier
LANGCHAIN_TRACING_V2=true
LANGCHAIN_PROJECT=rag-engineering-lab
```

Provide a `.env.example` with placeholders.

---

## Config

Centralized settings in `config.py` using `pydantic-settings`, loaded from `.env`. Should include:

- LLM model name and temperature (use `claude-sonnet-4-20250514`, temperature 0)
- Embedding model name (`all-MiniLM-L6-v2` — local, free, ~80MB, 384-dim)
- Retrieval top-k (start with 5)
- Max retrieval attempts (start with 3)
- Grounding score threshold (start with 0.7)
- File paths for uploads and Chroma persistence
- LangSmith settings

---

## Ecosystem Usage Map

This section clarifies which ecosystem tool to use for each concern. **Do not write custom implementations for any of these.**

| Concern | Use this | From package |
| --- | --- | --- |
| Document loading (PDF, DOCX, MD) | `DoclingLoader` | `langchain-docling` |
| Chunking | `HybridChunker` | `docling.chunking` |
| Embeddings | `HuggingFaceEmbeddings` | `langchain-huggingface` |
| Vector store | `Chroma` | `langchain-chroma` |
| Retriever interface | `.as_retriever()` on Chroma | `langchain-chroma` |
| LLM calls | `ChatAnthropic` | `langchain-anthropic` |
| Structured LLM output | `.with_structured_output()` | `langchain-core` |
| Prompt templates | `ChatPromptTemplate` | `langchain-core` |
| Graph orchestration | `StateGraph` | `langgraph` |
| Tracing | Automatic via env vars | `langsmith` |

**Custom code is only needed for:** LangGraph node functions, edge condition functions, graph assembly, FastAPI routes, Streamlit UI, and config.

---

## Graph State

A `TypedDict` that flows through the entire graph. Should track:

- **Input:** the original query
- **Classification:** query type (factual / comparative / summarization / out_of_scope)
- **Retrieval:** strategy used, retrieved chunks with scores
- **Grading:** whether context was sufficient, how many retrieval attempts so far, any rewritten query
- **Answer:** the generated answer, which chunk indices were cited
- **Grounding:** grounding score (0-1), pass/fail
- **Output:** the final response string
- **Metadata:** ordered list of which nodes executed (for the UI trace display)

Every node returns a **partial** state update — only the keys it changes.

---

## Nodes

Each node is a function that takes `GraphState` and returns a `dict` with the keys it updates. Nodes that call the LLM should use `ChatAnthropic` with `ChatPromptTemplate` for the prompt and `.with_structured_output()` when the response needs to be parsed into a specific shape (e.g., grade, grounding score, cited chunks).

### 1. `classify_query`

Classifies the query into a category: factual, comparative, summarization, or out_of_scope. Uses an LLM call with structured output. The classification is stored in state and informs the answer node's behavior. In Phase 1, classification does not affect routing — all queries proceed to retrieval.

### 2. `retrieve`

Performs dense similarity search against Chroma using the LangChain retriever interface (`.as_retriever()` or `similarity_search_with_score()`). Uses the rewritten query if one exists, otherwise the original query. Returns the top-k chunks and their similarity scores. Increments the retrieval attempt counter. No LLM call.

### 3. `grade_context`

Has the LLM evaluate whether the retrieved chunks contain sufficient information to answer the query. Uses structured output to return a grade ("sufficient" / "insufficient") and reasoning. The prompt receives the query and all retrieved chunk contents.

### 4. `rewrite_query`

Rewrites the query to improve retrieval on the next attempt. The LLM sees the original query, the insufficient chunks, and the attempt count. Returns a rewritten query string.

### 5. `answer`

Generates an answer using the retrieved chunks as context. Uses structured output so the LLM returns both the answer text and a list of cited chunk indices. The prompt instructs the LLM to only use information from the provided chunks and to indicate if the chunks are insufficient.

### 6. `verify_grounding`

Has the LLM evaluate whether the generated answer is well-supported by the cited chunks. Uses structured output to return a grounding score (0.0 to 1.0) and pass/fail based on the configured threshold.

### 7. `build_final_response`

No LLM call. Packages the final response. If grounding passed, returns the answer as-is. If grounding failed, appends a warning about low grounding confidence.

### 8. `handle_out_of_scope` (optional, defer if needed)

Short-circuit node for high-confidence out-of-scope classifications. Returns a canned response. Not required for Phase 1 — can route everything through retrieval initially.

---

## Edge Conditions

### After `classify_query`
Phase 1: always route to `retrieve`. Classification informs behavior, not routing.

### After `grade_context`
- If sufficient → go to `answer`
- If insufficient AND attempts < max → go to `rewrite_query`
- If insufficient AND attempts >= max → go to `answer` anyway (answer with what we have)

### After `rewrite_query`
Always go back to `retrieve`.

### After `verify_grounding`
- If grounding passes → go to `build_final_response`
- If grounding fails AND attempts < max → go to `rewrite_query` (feedback loop)
- If grounding fails AND attempts >= max → go to `build_final_response` (with warning)

### After `build_final_response`
End.

---

## Graph Assembly

Use `StateGraph` from LangGraph. Set `classify_query` as the entry point. Wire all nodes and conditional edges as described above. Compile the graph. The compiled graph is what the API invokes.

---

## Ingestion Pipeline

### Document Loading and Chunking

Use `DoclingLoader` from `langchain-docling` as the primary loader. It handles PDF, DOCX, and Markdown through a single interface. Configure it with Docling's `HybridChunker`, which produces structure-aware chunks that respect document layout, headers, and tables.

Pass `export_type=ExportType.DOC_CHUNKS` to get pre-chunked documents directly from Docling. The `HybridChunker` should be initialized with the embedding model tokenizer (`all-MiniLM-L6-v2`) so chunk sizes align with the embedding model's token limits.

The loader returns LangChain `Document` objects with metadata, so they can be passed directly to Chroma.

### Embedding and Storage

Use `HuggingFaceEmbeddings` from `langchain-huggingface` with `all-MiniLM-L6-v2`. Initialize `Chroma` from `langchain-chroma` with the embedding function and a local persist directory (`data/chroma/`). Use `.add_documents()` to store and `.similarity_search_with_score()` or `.as_retriever()` to search.

---

## FastAPI Backend

### `POST /ingest`
Accepts a file upload (multipart/form-data). Saves the file, runs it through `DoclingLoader` with `HybridChunker`, stores the resulting chunks in Chroma. Returns the filename, number of chunks created, and status.

### `POST /query`
Accepts a JSON body with a `query` string. Constructs the initial graph state, invokes the compiled graph, and returns the full result: final response, query type, retrieved chunks with scores, cited chunk indices, context grade, retrieval attempts, grounding score/pass, and the node trace.

### `GET /documents`
Returns a list of ingested documents with metadata (filename, chunk count, ingestion time).

### `GET /health`
Health check endpoint.

---

## Streamlit UI

### Layout (top to bottom)

**Header:** Project title.

**Document Upload section:** File uploader (accepts .pdf, .docx, and .md), ingest button, status message showing filename and chunk count after ingestion.

**Query section:** Text input, ask button.

**Results section (two columns):**
- **Left column — Pipeline Trace:** Vertical stepper showing each node that executed, in order, with key outputs (e.g., "classify_query → factual", "grade_context → sufficient", "verify_grounding → 0.92 ✅").
- **Right column — Answer:** The generated answer, grounding score with pass/fail indicator, list of cited chunk indices.

**Retrieved Chunks section (full width):** All retrieved chunks displayed as expandable cards. Each card shows: the chunk text, similarity score, source file, chunk index, and a visual indicator if it was cited in the answer.

### Implementation notes
- The UI communicates with the backend via `httpx` — do not import backend modules directly. Keep UI and API decoupled.
- Use `st.session_state` to persist the list of ingested documents across interactions.

---

## LangSmith Integration

Tracing is automatic when the environment variables are set. Every `graph.invoke()` call creates a trace with each node as a child span. No extra code needed beyond the env vars.

### What to verify after Phase 1
- Each query produces a trace with all executed nodes visible
- Conditional edges are visible (can see when grading fails and triggers rewrite)
- LLM calls within nodes show full input/output
- Retrieval nodes show chunks and scores
- Retry loops appear as repeated node executions in the trace

---

## Testing (manual)

Ingest a medium-length PDF (10-30 pages, something technical with clear sections). Then run these five query types:

1. A direct factual question with a clear answer in the document
2. A question requiring synthesis across sections
3. A comparison question
4. A question the document cannot answer (out of scope)
5. A vaguely worded question (should trigger the rewrite loop)

For each, verify in the Streamlit UI that classification, retrieval, grading, answer, and grounding all behave correctly and are displayed. Verify in LangSmith that traces are complete, at least one query triggers the rewrite loop, and all LLM inputs/outputs are logged.

---

## Done Criteria

- [ ] Can upload a PDF, DOCX, or Markdown file via Streamlit
- [ ] Docling parses and chunks the document with structure awareness
- [ ] Chunks are embedded and stored in Chroma
- [ ] Can ask a question and get an answer
- [ ] LangGraph DAG executes all nodes: classify → retrieve → grade → answer → verify → final
- [ ] All LLM nodes use `ChatAnthropic` + `ChatPromptTemplate` + `.with_structured_output()` where applicable
- [ ] Retry loop fires when context is graded insufficient
- [ ] Grounding verification works and flags low-grounding answers
- [ ] Streamlit shows: pipeline trace, answer, grounding score, all retrieved chunks with scores, cited chunk indicators
- [ ] Every query produces a visible trace in LangSmith
- [ ] FastAPI backend serves /ingest, /query, /documents, /health
- [ ] Code is typed, formatted with ruff, and organized per project structure