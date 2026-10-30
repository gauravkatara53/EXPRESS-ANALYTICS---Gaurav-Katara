from pathlib import Path
from typing import List, Optional
from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile, status

from src.api.schemas import (
    DocumentListResponse,
    FeedbackRequest,
    FeedbackResponse,
    HealthResponse,
    IndexingResult,
    IngestUploadResponse,
    IngestionPreviewSummary,
    QueryRequest,
    QueryResponse,
    SearchQueryRequest,
    SearchResponse,
    VectorStoreStats,
)
from src.config import settings
from src.ingestion.loaders import DocumentLoader
from src.ingestion.splitter import DocumentSplitter
from src.retrieval.vector_store import ChromaVectorStore
from src.services.document_service import DocumentService
from src.services.feedback_service import FeedbackService
from src.services.ingestion_service import IngestionService
from src.services.query_service import QueryService
from src.utils.logger import get_logger

logger = get_logger("api_routes")
router = APIRouter()

# Reusable Vector Store singleton for API handlers
_vector_store: Optional[ChromaVectorStore] = None


def get_vector_store() -> ChromaVectorStore:
    """Dependency provider for ChromaVectorStore singleton."""
    global _vector_store
    if _vector_store is None:
        _vector_store = ChromaVectorStore()
    return _vector_store


def get_query_service(store: ChromaVectorStore = Depends(get_vector_store)) -> QueryService:
    """Dependency provider for QueryService."""
    return QueryService(vector_store=store)


def get_ingestion_service(store: ChromaVectorStore = Depends(get_vector_store)) -> IngestionService:
    """Dependency provider for IngestionService."""
    return IngestionService(vector_store=store)


def get_document_service(store: ChromaVectorStore = Depends(get_vector_store)) -> DocumentService:
    """Dependency provider for DocumentService."""
    return DocumentService(vector_store=store)


def get_feedback_service() -> FeedbackService:
    """Dependency provider for FeedbackService."""
    return FeedbackService()


# ------------------------------------------------------------------------------
# System Endpoints
# ------------------------------------------------------------------------------

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
async def system_info(store: ChromaVectorStore = Depends(get_vector_store)):
    """Returns runtime configuration, ingestion directories, chunking parameters, and vector store details."""
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
        "llm_config": {
            "llm_provider": settings.LLM_PROVIDER,
            "llm_model": settings.GEMINI_LLM_MODEL,
            "max_retries": settings.MAX_RETRIES,
        },
        "phase_status": {
            "phase_1_project_setup": "completed",
            "phase_2_ingestion_and_chunking": "completed",
            "phase_3_vector_store_and_embeddings": "completed",
            "phase_4_langgraph_workflow": "completed",
            "phase_5_api_endpoints": "completed",
        },
    }


# ------------------------------------------------------------------------------
# Core Assignment Endpoints (POST /query, POST /ingest, GET /documents, POST /feedback)
# Supported at both root and /api/v1/ prefixes for full compatibility.
# ------------------------------------------------------------------------------

@router.post(
    "/api/v1/query",
    response_model=QueryResponse,
    status_code=status.HTTP_200_OK,
    summary="Submit Question (RAG Workflow)",
    tags=["RAG Core"],
)
@router.post(
    "/query",
    response_model=QueryResponse,
    status_code=status.HTTP_200_OK,
    include_in_schema=False,
    tags=["RAG Core"],
)
async def query_technical_assistant(
    request: QueryRequest,
    service: QueryService = Depends(get_query_service),
) -> QueryResponse:
    """Executes the Self-Corrective LangGraph workflow to retrieve context and answer questions."""
    try:
        response = service.process_query(request)
        return response
    except Exception as e:
        logger.error(f"Error executing RAG query: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An error occurred while processing the query: {str(e)}",
        )


@router.post(
    "/api/v1/ingest",
    response_model=IngestUploadResponse,
    status_code=status.HTTP_200_OK,
    summary="Ingest New Documents",
    tags=["RAG Core"],
)
@router.post(
    "/ingest",
    response_model=IngestUploadResponse,
    status_code=status.HTTP_200_OK,
    include_in_schema=False,
    tags=["RAG Core"],
)
async def ingest_documents(
    files: List[UploadFile] = File(..., description="Markdown (.md), Text (.txt), or HTML (.html) files"),
    service: IngestionService = Depends(get_ingestion_service),
) -> IngestUploadResponse:
    """Accepts file uploads, validates and saves them, splits into chunks, and indexes them in ChromaDB."""
    if not files:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No files provided for ingestion.",
        )
    result = await service.ingest_uploaded_files(files)
    return result


@router.get(
    "/api/v1/documents",
    response_model=DocumentListResponse,
    status_code=status.HTTP_200_OK,
    summary="List Indexed Documents",
    tags=["RAG Core"],
)
@router.get(
    "/documents",
    response_model=DocumentListResponse,
    status_code=status.HTTP_200_OK,
    include_in_schema=False,
    tags=["RAG Core"],
)
async def list_indexed_documents(
    service: DocumentService = Depends(get_document_service),
) -> DocumentListResponse:
    """Returns the catalog of all technical documents indexed in ChromaDB."""
    return service.list_documents()


@router.post(
    "/api/v1/feedback",
    response_model=FeedbackResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Submit User Feedback",
    tags=["RAG Core"],
)
@router.post(
    "/feedback",
    response_model=FeedbackResponse,
    status_code=status.HTTP_201_CREATED,
    include_in_schema=False,
    tags=["RAG Core"],
)
async def submit_feedback(
    request: FeedbackRequest,
    service: FeedbackService = Depends(get_feedback_service),
) -> FeedbackResponse:
    """Stores thumbs up/down user feedback persistently in SQLite."""
    try:
        return service.record_feedback(request)
    except Exception as e:
        logger.error(f"Error recording feedback: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to record feedback: {str(e)}",
        )


# ------------------------------------------------------------------------------
# Ingestion Preview & Direct Vector Search Endpoints
# ------------------------------------------------------------------------------

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
    """Scans the document corpus on disk, splits into chunks, and returns an audit preview."""
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
    return splitter.generate_summary(
        total_files_found=total_files,
        loaded_documents=docs,
        chunks=chunks,
        skipped_count=len(skipped),
        preview_limit=preview_limit,
    )


@router.post(
    "/api/v1/search",
    response_model=SearchResponse,
    status_code=status.HTTP_200_OK,
    summary="Direct Vector Search",
    tags=["Vector Store"],
)
async def semantic_search(
    request: SearchQueryRequest,
    store: ChromaVectorStore = Depends(get_vector_store),
) -> SearchResponse:
    """Direct semantic similarity search on ChromaDB without running the full LangGraph workflow."""
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
async def store_stats(store: ChromaVectorStore = Depends(get_vector_store)) -> VectorStoreStats:
    """Returns statistics about the active ChromaDB collection and indexed files."""
    return VectorStoreStats(
        collection_name=store.collection_name,
        total_chunks=store.collection.count(),
        persist_directory=str(store.persist_directory),
        indexed_documents=store.get_indexed_documents(),
    )
