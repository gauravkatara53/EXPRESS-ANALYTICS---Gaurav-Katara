from typing import List, Optional, TypedDict
from pydantic import BaseModel, Field

from src.api.schemas import SearchResult


class QueryAnalysisOutput(BaseModel):
    """Pydantic schema for Query Analysis node output."""

    intent: str = Field(
        description="Classified query intent: conceptual, how-to, troubleshooting, or api_reference"
    )
    search_query: str = Field(
        description="Refined, keyword-rich search query optimized for vector store retrieval"
    )


class DocumentGradeOutput(BaseModel):
    """Pydantic schema for Document Grading node output."""

    is_relevant: bool = Field(
        description="Binary grade: True if the document chunk contains context relevant to the question, else False"
    )
    reasoning: str = Field(
        description="Brief explanation of why the document chunk is relevant or irrelevant"
    )


class BatchDocumentGradeOutput(BaseModel):
    """Pydantic schema for high-throughput single-pass multi-document relevance evaluation."""

    relevant_indices: List[int] = Field(
        default_factory=list,
        description="1-based integer indices of the document chunks that contain relevant context for answering the question. Empty list if none are relevant.",
    )
    reasoning: str = Field(
        default="",
        description="Brief explanation of why the selected chunks are relevant, or why none are relevant.",
    )


class GraphState(TypedDict):
    """The central state passed between nodes in the LangGraph Self-Corrective RAG workflow."""

    question: str
    search_query: str
    query_intent: str
    documents: List[SearchResult]
    relevant_documents: List[SearchResult]
    generation: str
    citations: List[str]
    retry_count: int
    max_retries: int
    is_fallback: bool

