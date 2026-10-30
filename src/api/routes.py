from pathlib import Path
from typing import Optional
from fastapi import APIRouter, HTTPException, Query, status

from src.api.schemas import (
    HealthResponse,
    IndexingResult,
    IngestionPreviewSummary,
    SearchQueryRequest,
    SearchResponse,
    VectorStoreStats,
)
from src.config import settings
from src.ingestion.loaders import DocumentLoader
from src.ingestion.splitter import DocumentSplitter
from src.retrieval.vector_store import ChromaVectorStore
from src.utils.logger import get_logger

logger = get_logger("api_routes")
router = APIRouter()

# Reusable Vector Store singleton for API handlers
_vector_store: Optional[ChromaVectorStore] = None


def get_vector_store() -> ChromaVectorStore:
    """Provides a lazy-loaded singleton ChromaVectorStore instance."""
    global _vector_store
    if _vector_store is None:
        _vector_store = ChromaVectorStore()
    return _vector_store


@router.get(
    "/health",
    response_model=HealthResponse,
    status_code=status.HTTP_200_OK,
    summary="Health Check",
    tags=["System"],
)
async def health_check() -> HealthResponse:
    """Verifies that the API service is alive and healthy."""
    return HealthResponse(
        status="healthy",
        app_name=settings.APP_NAME,
        environment=settings.APP_ENV,
        debug_mode=settings.DEBUG,
    )


@router.get(
    "/info",
    status_code=status.HTTP_200_OK,
    summary="System Information",
    tags=["System"],
)
async def system_info():
    """Returns runtime configuration, ingestion directories, chunking parameters, and vector store details."""
    store = get_vector_store()
    return {
        "app_name": settings.APP_NAME,
        "environment": settings.APP_ENV,
        "documents_directory": str(settings.DOCUMENTS_DIR),
        "chunking_config": {
            "chunk_size": settings.CHUNK_SIZE,
            "chunk_overlap": settings.CHUNK_OVERLAP,
        },
        "vector_store_config": {
            "collection_name": store.collection_name,
            "persist_directory": str(store.persist_directory),
            "total_indexed_chunks": store.collection.count(),
            "embedding_provider": settings.EMBEDDING_PROVIDER,
            "embedding_model": settings.GEMINI_EMBEDDING_MODEL,
        },
        "phase_status": {
            "phase_1_project_setup": "completed",
            "phase_2_ingestion_and_chunking": "completed",
            "phase_3_vector_store_and_embeddings": "completed",
            "phase_4_langgraph_workflow": "pending",
            "phase_5_api_endpoints": "pending",
        },
    }


@router.get(
    "/api/v1/ingest/preview",
    response_model=IngestionPreviewSummary,
    status_code=status.HTTP_200_OK,
    summary="Preview Ingestion & Chunking",
    tags=["Ingestion"],
)
async def preview_ingestion(
    chunk_size: int = Query(
        default=settings.CHUNK_SIZE, ge=100, le=4000, description="Max characters per chunk"
    ),
    chunk_overlap: int = Query(
        default=settings.CHUNK_OVERLAP, ge=0, le=1000, description="Character overlap between chunks"
    ),
    preview_limit: int = Query(
        default=5, ge=1, le=50, description="Number of sample chunks to return in preview"
    ),
) -> IngestionPreviewSummary:
    """Scans the document corpus, loads Markdown/TXT/HTML files, splits them into chunks,
    and returns an ingestion audit summary with metadata.
    """
    if chunk_overlap >= chunk_size:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"chunk_overlap ({chunk_overlap}) must be strictly less than chunk_size ({chunk_size})",
        )

    doc_dir = Path(settings.DOCUMENTS_DIR)
    if not doc_dir.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Documents directory '{doc_dir}' does not exist.",
        )

    loader = DocumentLoader()
    docs, skipped = loader.load_directory(doc_dir)

    splitter = DocumentSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = splitter.split_documents(docs)

    total_files = len(docs) + len(skipped)
    summary = splitter.generate_summary(
        total_files_found=total_files,
        loaded_documents=docs,
        chunks=chunks,
        skipped_count=len(skipped),
        preview_limit=preview_limit,
    )

    return summary


@router.post(
    "/api/v1/ingest/index",
    response_model=IndexingResult,
    status_code=status.HTTP_200_OK,
    summary="Index Corpus into ChromaDB",
    tags=["Vector Store"],
)
async def index_documents(
    chunk_size: int = Query(default=settings.CHUNK_SIZE, ge=100, le=4000),
    chunk_overlap: int = Query(default=settings.CHUNK_OVERLAP, ge=0, le=1000),
) -> IndexingResult:
    """Loads all documents from disk, chunks them, and indexes them into ChromaDB.
    Prevents duplicate re-indexing through deterministic chunk IDs and SHA256 content hashes.
    """
    doc_dir = Path(settings.DOCUMENTS_DIR)
    if not doc_dir.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Documents directory '{doc_dir}' does not exist.",
        )

    loader = DocumentLoader()
    docs, _ = loader.load_directory(doc_dir)

    splitter = DocumentSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = splitter.split_documents(docs)

    store = get_vector_store()
    result = store.index_chunks(chunks)
    return result


@router.post(
    "/api/v1/search",
    response_model=SearchResponse,
    status_code=status.HTTP_200_OK,
    summary="Semantic Similarity Search",
    tags=["Vector Store"],
)
async def semantic_search(request: SearchQueryRequest) -> SearchResponse:
    """Retrieves top-k most semantically relevant document chunks from ChromaDB for a natural language query."""
    store = get_vector_store()
    results = store.similarity_search(
        query=request.query,
        top_k=request.top_k,
        score_threshold=request.score_threshold,
    )
    return SearchResponse(
        query=request.query,
        total_results=len(results),
        results=results,
    )


@router.get(
    "/api/v1/store/stats",
    response_model=VectorStoreStats,
    status_code=status.HTTP_200_OK,
    summary="Vector Store Statistics",
    tags=["Vector Store"],
)
async def store_stats() -> VectorStoreStats:
    """Returns metadata about the active ChromaDB collection and indexed documents."""
    store = get_vector_store()
    indexed_docs = store.get_indexed_documents()
    return VectorStoreStats(
        collection_name=store.collection_name,
        total_chunks=store.collection.count(),
        persist_directory=str(store.persist_directory),
        indexed_documents=indexed_docs,
    )
