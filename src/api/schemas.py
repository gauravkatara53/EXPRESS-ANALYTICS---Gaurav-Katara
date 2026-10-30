from typing import Any, Dict, List, Optional
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
