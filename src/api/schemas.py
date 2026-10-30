from typing import List, Optional
from pydantic import BaseModel, Field


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
