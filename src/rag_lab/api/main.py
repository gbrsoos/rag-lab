from collections import defaultdict

from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from rag_lab.config import settings
from rag_lab.graph.builder import compiled_graph
from rag_lab.ingestion.pipeline import get_vector_store, ingest_document

# ── App setup ─────────────────────────────────────────────────────────────────

app = FastAPI(title="RAG Engineering Lab", version="0.1.0")

# Allow the Streamlit UI (different port) to call this API.
# allow_origins=["*"] is acceptable for local development.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Request / response models ─────────────────────────────────────────────────


class QueryRequest(BaseModel):
    query: str


class ChunkOut(BaseModel):
    """A single retrieved chunk, serialized for JSON output.

    LangChain Document objects are not JSON-serializable directly.
    We flatten them here: content (the chunk text), metadata (source,
    page, etc.), similarity score, and whether the answer node cited it.
    """

    content: str
    metadata: dict
    score: float
    cited: bool


class QueryResponse(BaseModel):
    final_response: str
    query_type: str
    retrieved_chunks: list[ChunkOut]
    context_grade: str
    retrieval_attempts: int
    grounding_score: float
    grounding_pass: bool
    node_trace: list[str]
    cited_chunk_indices: list[int]


class IngestResponse(BaseModel):
    filename: str
    chunk_count: int
    status: str


class DocumentInfo(BaseModel):
    filename: str
    chunk_count: int


class DocumentsResponse(BaseModel):
    documents: list[DocumentInfo]


# ── Endpoints ─────────────────────────────────────────────────────────────────


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/ingest", response_model=IngestResponse)
async def ingest(file: UploadFile) -> IngestResponse:
    """Accept a file upload, chunk it with Docling, and store in Chroma.

    Supported formats: .pdf, .docx, .md
    The file is saved to settings.uploads_dir before ingestion so the
    source path is available for Docling and stored in chunk metadata.
    """
    allowed = {".pdf", ".docx", ".md"}
    suffix = "." + file.filename.split(".")[-1].lower() if "." in file.filename else ""
    if suffix not in allowed:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{suffix}'. Allowed: {allowed}",
        )

    settings.uploads_dir.mkdir(parents=True, exist_ok=True)
    file_path = settings.uploads_dir / file.filename

    with open(file_path, "wb") as f:
        f.write(await file.read())

    result = ingest_document(file_path)

    return IngestResponse(
        filename=result["filename"],
        chunk_count=result["chunk_count"],
        status="ok",
    )


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest) -> QueryResponse:
    """Run the full LangGraph RAG pipeline for a query.

    Constructs the initial state and invokes the compiled graph. The graph
    runs synchronously — all LLM calls and retrieval happen inside .invoke().
    LangSmith tracing is automatic when LANGCHAIN_TRACING_V2=true.

    retrieval_attempts must be initialized to 0 so the edge functions
    have a valid integer to compare against max_retrieval_attempts.
    """
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query must not be empty.")

    initial_state = {
        "query": request.query.strip(),
        "retrieval_attempts": 0,
    }

    result = compiled_graph.invoke(initial_state)

    # Build chunk list, annotating which ones were cited by the answer node
    chunks = result.get("retrieved_chunks", [])
    scores = result.get("similarity_scores", [])
    cited_indices = result.get("cited_chunk_indices", [])

    chunk_out = [
        ChunkOut(
            content=doc.page_content,
            metadata=doc.metadata,
            score=scores[i] if i < len(scores) else 0.0,
            cited=i in cited_indices,
        )
        for i, doc in enumerate(chunks)
    ]

    return QueryResponse(
        final_response=result.get("final_response", ""),
        query_type=result.get("query_type", ""),
        retrieved_chunks=chunk_out,
        context_grade=result.get("context_grade", ""),
        retrieval_attempts=result.get("retrieval_attempts", 0),
        grounding_score=result.get("grounding_score", 0.0),
        grounding_pass=result.get("grounding_pass", False),
        node_trace=result.get("node_trace", []),
        cited_chunk_indices=cited_indices,
    )


@app.get("/documents", response_model=DocumentsResponse)
def list_documents() -> DocumentsResponse:
    """Return the list of ingested documents with chunk counts.

    Uses Chroma's public .get() method to retrieve all chunk metadata,
    then aggregates by the 'source' field that DoclingLoader writes into
    each chunk's metadata. This avoids maintaining a separate document
    registry — Chroma is the single source of truth.
    """
    vector_store = get_vector_store()
    result = vector_store.get(include=["metadatas"])
    metadatas = result.get("metadatas") or []

    counts: dict[str, int] = defaultdict(int)
    for meta in metadatas:
        source = meta.get("source", "unknown") if meta else "unknown"
        counts[source] += 1

    return DocumentsResponse(
        documents=[
            DocumentInfo(filename=source, chunk_count=count)
            for source, count in counts.items()
        ]
    )
