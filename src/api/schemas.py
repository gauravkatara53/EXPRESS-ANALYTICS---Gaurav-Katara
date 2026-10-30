from typing import List, Optional
from pydantic import BaseModel, Field, field_validator

from src.database.feedback_db import FeedbackRecord


class ChunkMetadata(BaseModel):
    """Metadata schema attached to every retrieved and indexed document chunk."""

    source_filename: str = Field(description="Name of the source file (e.g., fastapi_overview.md)")
    source_path: str = Field(description="Path to the document relative to the workspace")
    document_title: str = Field(description="Human-readable title extracted from the document")
    file_type: str = Field(description="Normalized file extension (.md, .txt, .html)")
    chunk_index: int = Field(description="0-indexed position of this chunk within the document")
    total_chunks: int = Field(description="Total chunks generated for this source document")
    character_count: int = Field(description="Character length of the chunk content")
    content_hash: Optional[str] = Field(default=None, description="SHA256 hash for deduplication")


class DocumentChunk(BaseModel):
    """Represents an individual text chunk ready for vectorization and RAG retrieval."""

    content: str = Field(description="The textual body of the chunk")
    metadata: ChunkMetadata = Field(description="Source and structural metadata")


class LoadedDocument(BaseModel):
    """Represents a loaded raw document before chunking."""

    source_filename: str
    source_path: str
    document_title: str
    file_type: str
    raw_content: str
    character_count: int
    file_size_bytes: int


class IngestionPreviewSummary(BaseModel):
    """Summary statistics for loaded documents and created chunks."""

    total_files_found: int
    total_files_loaded: int
    total_files_skipped: int
    total_chunks_created: int
    average_chunk_length: float
    documents: List[LoadedDocument]
    chunks_preview: List[DocumentChunk]


class HealthResponse(BaseModel):
    """API health status response model."""

    status: str
    app_name: str
    environment: str
    debug_mode: bool


class SearchResult(BaseModel):
    """Individual search result returned from semantic similarity retrieval."""

    chunk_id: str = Field(description="Unique identifier of the chunk in ChromaDB")
    content: str = Field(description="Retrieved chunk text content")
    metadata: ChunkMetadata = Field(description="Source document metadata and chunk index")
    score: float = Field(description="Cosine similarity score (0.0 to 1.0, higher is better)")


class SearchQueryRequest(BaseModel):
    """Search query request payload."""

    query: str = Field(min_length=1, description="Natural language technical question")
    top_k: int = Field(default=4, ge=1, le=20, description="Maximum number of relevant chunks")
    score_threshold: Optional[float] = Field(
        default=None, ge=0.0, le=1.0, description="Minimum similarity score threshold"
    )


class SearchResponse(BaseModel):
    """Search query response payload."""

    query: str
    total_results: int
    results: List[SearchResult]


class IndexingResult(BaseModel):
    """Result summary of indexing chunks into ChromaDB."""

    total_submitted: int = Field(description="Number of chunks submitted for indexing")
    chunks_indexed: int = Field(description="Number of newly indexed chunks")
    chunks_updated: int = Field(description="Number of existing chunks whose content was updated")
    chunks_skipped_duplicates: int = Field(
        description="Number of existing chunks skipped without re-indexing"
    )
    total_store_chunks: int = Field(description="Total chunks currently in ChromaDB collection")


class VectorStoreStats(BaseModel):
    """Vector database statistics."""

    collection_name: str
    total_chunks: int
    persist_directory: str
    indexed_documents: List[str]


# ------------------------------------------------------------------------------
# Phase 5 Schemas: Query, Ingest, Documents, Feedback
# ------------------------------------------------------------------------------

class QueryRequest(BaseModel):
    """Natural language question request payload for POST /api/v1/query."""

    question: str = Field(min_length=1, max_length=2000, description="User's technical documentation question")
    max_retries: Optional[int] = Field(
        default=2, ge=0, le=5, description="Maximum query reformulation retries allowed"
    )

    @field_validator("question")
    @classmethod
    def validate_non_whitespace(cls, v: str) -> str:
        cleaned = v.strip()
        if not cleaned:
            raise ValueError("Question cannot be empty or whitespace only.")
        return cleaned


class QueryResponse(BaseModel):
    """Response payload for POST /api/v1/query executing the LangGraph workflow."""

    query_id: str = Field(description="Unique identifier for tracing and submitting feedback")
    question: str = Field(description="Original user question")
    answer: str = Field(description="Final generated answer grounded in technical documentation")
    citations: List[str] = Field(description="Source filenames referenced in the answer")
    is_fallback: bool = Field(description="Whether safe fallback response was triggered")
    retry_count: int = Field(description="Number of self-correction retrieval retries executed")
    query_intent: str = Field(description="Classified query intent")
    search_query: str = Field(description="Final search query used for retrieval")
    retrieved_chunks_count: int = Field(description="Total candidate chunks retrieved")
    relevant_chunks_count: int = Field(description="Chunks evaluated as relevant by grader")


class IngestUploadResponse(BaseModel):
    """Response payload for POST /api/v1/ingest file upload."""

    uploaded_files: List[str] = Field(description="Successfully processed and stored files")
    total_chunks_indexed: int = Field(description="New chunks indexed into ChromaDB")
    total_chunks_updated: int = Field(description="Existing chunks updated with new content")
    total_chunks_skipped: int = Field(description="Duplicate chunks skipped without re-embedding")
    total_store_chunks: int = Field(description="Total chunks in ChromaDB collection")
    failed_files: List[str] = Field(description="Files that failed to upload or parse with reasons")


class DocumentInfo(BaseModel):
    """Detailed metadata for an individual document indexed in ChromaDB."""

    source_filename: str = Field(description="File name (e.g. fastapi_overview.md)")
    document_title: str = Field(description="Title extracted from document")
    file_type: str = Field(description="File extension (.md, .txt, .html)")
    source_path: str = Field(description="Relative path on disk")
    chunk_count: int = Field(description="Number of chunks indexed for this document")
    total_characters: int = Field(description="Total characters across all chunks")


class DocumentListResponse(BaseModel):
    """Response payload for GET /api/v1/documents catalog."""

    total_documents: int
    total_chunks: int
    documents: List[DocumentInfo]


class FeedbackRequest(BaseModel):
    """Request payload for POST /api/v1/feedback."""

    query_id: str = Field(min_length=1, description="Unique query ID returned from POST /api/v1/query")
    rating: str = Field(description="Feedback rating: 'thumbs_up' or 'thumbs_down'")
    comment: Optional[str] = Field(default=None, max_length=2000, description="Optional qualitative feedback")

    @field_validator("rating")
    @classmethod
    def validate_rating_value(cls, v: str) -> str:
        normalized = v.strip().lower().replace(" ", "_")
        if normalized in {"thumbs_up", "up", "positive", "1", "+1"}:
            return "thumbs_up"
        elif normalized in {"thumbs_down", "down", "negative", "-1", "0"}:
            return "thumbs_down"
        raise ValueError("Rating must be 'thumbs_up' or 'thumbs_down'.")


class FeedbackResponse(BaseModel):
    """Response payload for POST /api/v1/feedback."""

    status: str = "success"
    message: str = "Feedback recorded successfully."
    feedback: FeedbackRecord
