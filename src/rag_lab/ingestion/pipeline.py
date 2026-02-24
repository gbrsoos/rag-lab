import logging
from pathlib import Path

from docling.chunking import HybridChunker
from langchain_chroma import Chroma
from langchain_community.vectorstores.utils import filter_complex_metadata
from langchain_docling import DoclingLoader
from langchain_docling.loader import ExportType
from langchain_huggingface import HuggingFaceEmbeddings
from transformers import AutoTokenizer

from rag_lab.config import settings

logger = logging.getLogger(__name__)

# Maximum token length for all-MiniLM-L6-v2. Chunks exceeding this will be
# silently truncated by the embedding model, producing vectors that don't
# represent the full text.
_MAX_TOKENS = 256


def get_embeddings() -> HuggingFaceEmbeddings:
    """Return the shared embedding model instance.

    HuggingFaceEmbeddings loads ~80 MB of model weights from disk (or
    downloads them on first use to ~/.cache/huggingface). This is called
    once per process — callers should not instantiate their own copies.

    all-MiniLM-L6-v2 produces 384-dimensional vectors and runs entirely
    locally with no API key required.
    """
    return HuggingFaceEmbeddings(model_name=settings.embedding_model)


def get_vector_store() -> Chroma:
    """Return a Chroma vector store connected to the local persist directory.

    All ingested documents share a single collection ("rag_lab") so that
    every query searches across the full document set. Using a fixed
    collection name keeps the API simple; Phase 2 can extend this to
    support per-document or per-strategy collections if needed.

    persist_directory tells Chroma where to write its SQLite + index files.
    On the first call the directory is created; on subsequent calls the
    existing data is loaded. langchain-chroma handles this transparently.
    """
    settings.chroma_dir.mkdir(parents=True, exist_ok=True)
    return Chroma(
        collection_name="rag_lab",
        embedding_function=get_embeddings(),
        persist_directory=str(settings.chroma_dir),
    )


def ingest_document(file_path: Path) -> dict[str, object]:
    """Load, chunk, embed, and store a document in Chroma.

    Args:
        file_path: Absolute or relative path to the source document.
                   Supported formats: PDF, DOCX, Markdown (Docling handles all three).

    Returns:
        A dict with keys:
            filename (str): The source file's name.
            chunk_count (int): Number of chunks stored.

    Flow:
        1. HybridChunker tokenizer is initialized with the same model used for
           embeddings so chunk token counts align with the embedding model's
           context window (256 tokens for all-MiniLM-L6-v2).
        2. DoclingLoader parses the document and emits pre-split Document objects
           (ExportType.DOC_CHUNKS). Each Document has page_content (the chunk
           text) and metadata (source, page, element type, etc.).
        3. get_vector_store() opens (or creates) the Chroma collection.
        4. add_documents() embeds each chunk and writes it to the collection.
    """
    # HybridChunker accepts a HuggingFace model ID as its tokenizer.
    # The "sentence-transformers/" prefix is required by the HuggingFace hub;
    # settings.embedding_model stores the short name ("all-MiniLM-L6-v2").
    chunker = HybridChunker(
        tokenizer=f"sentence-transformers/{settings.embedding_model}",
        max_tokens=_MAX_TOKENS,
    )

    loader = DoclingLoader(
        file_path=str(file_path),
        export_type=ExportType.DOC_CHUNKS,
        chunker=chunker,
    )

    docs = loader.load()

    # ── Token length verification ─────────────────────────────────────────────
    # Use the same tokenizer the embedding model uses to count tokens per chunk.
    # If any chunk exceeds _MAX_TOKENS, the embedding model will truncate it
    # silently — the stored vector won't represent the full chunk text.
    # HybridChunker should prevent this, but we verify rather than assume.
    tokenizer = AutoTokenizer.from_pretrained(
        f"sentence-transformers/{settings.embedding_model}"
    )
    # add_special_tokens=False excludes [CLS] and [SEP] from the count.
    # HybridChunker counts content tokens only — we must do the same or we'll
    # report false positives (a 256-token chunk becomes 258 with special tokens).
    token_counts = [
        len(tokenizer.encode(doc.page_content, add_special_tokens=False))
        for doc in docs
    ]
    if token_counts:
        oversized = [c for c in token_counts if c > _MAX_TOKENS]
        logger.info(
            "Chunk token stats — count: %d, max: %d, mean: %.1f",
            len(docs),
            max(token_counts),
            sum(token_counts) / len(token_counts),
        )
        if oversized:
            logger.warning(
                "%d chunk(s) exceed %d tokens (max seen: %d). "
                "These will be truncated during embedding.",
                len(oversized),
                _MAX_TOKENS,
                max(oversized),
            )

    vector_store = get_vector_store()
    docs = filter_complex_metadata(docs)
    vector_store.add_documents(docs)

    return {"filename": file_path.name, "chunk_count": len(docs)}
