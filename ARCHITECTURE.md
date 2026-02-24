# RAG Engineering Lab — Architecture

## Overview

A retrieval-augmented generation workbench for uploading documents, asking questions, and getting full transparency into every pipeline stage. The system is agentic — the LLM makes runtime decisions about query routing, context quality, and answer grounding, with multiple feedback loops.

The project demonstrates applied AI/ML engineering: not just building a RAG system, but instrumenting, evaluating, and improving one.

---

## Tech Stack

| Component            | Tool                                      |
| -------------------- | ----------------------------------------- |
| LLM                  | Anthropic Claude (via `langchain-anthropic`) |
| Orchestration        | LangGraph                                 |
| Components           | LangChain (retrievers, prompts, embeddings, output parsers) |
| Document ingestion   | Docling (via `langchain-docling`)         |
| Dense retrieval      | Chroma (local, persisted to disk) via `langchain-chroma` |
| Sparse retrieval     | rank-bm25                                 |
| Reranker             | sentence-transformers cross-encoder       |
| Embeddings           | HuggingFace sentence-transformers (via `langchain-huggingface`) |
| Backend              | FastAPI                                   |
| UI                   | Streamlit                                 |
| Observability & eval | LangSmith (free tier)                     |
| Storage              | Local filesystem                          |
| Containerization     | Docker Compose                            |

### Design principle: use the ecosystem

**Prefer LangChain/LangGraph/Docling built-in classes and methods over custom implementations.** Custom code should only exist for project-specific orchestration logic (the LangGraph nodes and edges). Everything else — loading, chunking, embedding, retrieval, LLM calls, prompt formatting, output parsing, tracing — should use existing ecosystem components.

### Role of each tool

- **Docling** (via `langchain-docling`) = document ingestion. Handles PDF, DOCX, Markdown, and more through a single interface. Provides structure-aware chunking via `HybridChunker` that respects document layout, headers, and tables. Replaces `PyPDFLoader` + `RecursiveCharacterTextSplitter` as the primary ingestion path.
- **LangChain** = component library. `ChatAnthropic` for LLM calls, `HuggingFaceEmbeddings` for embeddings, `Chroma` for vector store, `ChatPromptTemplate` for prompts, structured output parsing via `.with_structured_output()`, retriever interfaces. Not used for chaining — used as a library of parts.
- **LangGraph** = orchestration. Defines the DAG: every node, every edge, conditional routing, feedback loops, state schema. All agentic behavior lives here.
- **LangSmith** = observability and evaluation. Traces every run, logs decisions at each node, supports dataset creation, experiment runs, and metric computation.

---

## Agentic LangGraph DAG

```
Query
  │
  ▼
┌─────────────────┐
│ Classify Query   │  ← Routes: factual / comparative / out-of-scope / ...
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Retrieve         │  ← Strategy chosen based on classification
└────────┬────────┘
         │
         ▼
┌─────────────────┐       ┌──────────────────┐
│ Grade Context    │──────▶│ Rewrite Query     │
│                  │ insuf │                    │
│                  │       └────────┬───────────┘
└────────┬────────┘                │
         │ sufficient              │ (up to N retries, dynamic exit)
         ▼                         │
┌─────────────────┐                │
│ Answer           │◀──────────────┘
│ (+ tool access   │
│  to pull chunks) │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Verify Grounding │
│                  │──── poorly grounded ──▶ targeted re-retrieval ──▶ back to Grade
└────────┬────────┘
         │ well grounded
         ▼
┌─────────────────┐
│ Final Response   │
└─────────────────┘
```

### Agentic behaviors

1. **Query classification and routing** — LLM decides how to handle the query before retrieval begins
2. **Context grading with retry loop** — LLM evaluates retrieval quality and rewrites the query up to N times, deciding dynamically when to give up
3. **Tool use in answer generation** — the answer node can pull additional chunks if it needs more evidence mid-generation
4. **Grounding verification with feedback loop** — if the answer isn't well-supported, the system loops back for targeted re-retrieval

### Graph State Schema (preliminary)

```python
class GraphState(TypedDict):
    query: str                          # Original user query
    query_type: str                     # Classification result
    rewritten_query: str | None         # If query was rewritten
    retrieval_strategy: str             # dense / bm25 / hybrid
    retrieved_chunks: list[Document]    # Retrieved documents
    similarity_scores: list[float]      # Scores per chunk
    reranked_chunks: list[Document] | None  # After reranking
    context_grade: str                  # sufficient / insufficient
    retrieval_attempts: int             # Counter for retry loop
    max_retrieval_attempts: int         # Dynamic exit threshold
    answer: str                         # Generated answer
    cited_chunks: list[Document]        # Chunks actually used
    grounding_score: float              # How well-grounded the answer is
    grounding_pass: bool                # Whether grounding check passed
    final_response: str                 # Final output to user
```

---

## Retrieval Strategies

The system supports three retrieval strategies, selectable per query:

1. **Dense retrieval** — Chroma vector store with cosine similarity, using LangChain's retriever interface
2. **BM25 (sparse)** — rank-bm25 for keyword-based retrieval
3. **Hybrid** — Dense + BM25 with reciprocal rank fusion (RRF) score combination

Optional **cross-encoder reranking** can be applied after any strategy.

### Configurable parameters

- Chunking strategy (Docling HybridChunker vs RecursiveCharacterTextSplitter for comparison)
- Chunk size and overlap (for RecursiveCharacterTextSplitter comparisons in Phase 2)
- Top-k for retrieval
- Reranking on/off
- Score fusion weights (for hybrid)

---

## Phases

### Phase 1: Core agentic pipeline, end-to-end

- Document upload (PDF, DOCX, Markdown) via Docling
- Structure-aware chunking via Docling's HybridChunker
- Dense retrieval via Chroma (using LangChain's retriever interface)
- Full LangGraph DAG: classify → retrieve → grade context → (rewrite + retry if insufficient) → answer → verify grounding → (loop back if poorly grounded)
- LLM calls via `ChatAnthropic` with `.with_structured_output()` for nodes that return structured data
- Streamlit UI showing: query, retrieved chunks with similarity scores, grading decisions, the final answer, which chunks were cited
- LangSmith tracing enabled on every run
- **Deliverable:** A working, demonstrable agentic RAG pipeline

### Phase 2: Retrieval strategy comparison

- Add BM25 retrieval
- Add hybrid search (dense + BM25 with reciprocal rank fusion)
- Add cross-encoder reranking as an optional graph node
- Add RecursiveCharacterTextSplitter as an alternative chunking strategy for comparison with Docling's HybridChunker
- UI allows selecting a strategy and viewing side-by-side results
- **Deliverable:** Same query, different strategies, visible differences

### Phase 3: Evaluation and experimentation

- Build a test dataset in LangSmith (questions + expected answers + relevant chunks)
- Run each retrieval strategy against the dataset as an experiment
- Compute metrics: recall@k, groundedness, answer relevance
- Produce comparison tables or charts
- Screenshot everything for the README
- **Deliverable:** Quantitative proof of system performance and tradeoffs

### Phase 4: Polish and documentation

- Docker Compose setup (retrieval API container, UI container)
- Architecture diagram (LangGraph DAG + system components)
- README with: problem statement, design decisions, evaluation results with real numbers, setup instructions
- Clean, typed, documented code
- **Deliverable:** A portfolio piece ready for public review

---

## Out of Scope

- Raspberry Pi integration
- Pluggable framework / swappable component abstraction layer
- Paid infrastructure (no Pinecone, Weaviate Cloud, etc.)
- Production deployment / cloud hosting