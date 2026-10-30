import uuid
from typing import Optional

from src.api.schemas import QueryRequest, QueryResponse
from src.config import settings
from src.graph.workflow import run_rag_pipeline
from src.retrieval.vector_store import ChromaVectorStore
from src.utils.logger import get_logger

logger = get_logger("query_service")


class QueryService:
    """Business logic for executing the LangGraph Self-Corrective RAG workflow."""

    def __init__(self, vector_store: Optional[ChromaVectorStore] = None):
        self.vector_store = vector_store or ChromaVectorStore()

    def process_query(self, request: QueryRequest) -> QueryResponse:
        """Runs the LangGraph workflow for an incoming user question."""
        query_id = str(uuid.uuid4())
        logger.info(f"Processing query [{query_id}]: '{request.question}'")

        state = run_rag_pipeline(
            question=request.question,
            max_retries=request.max_retries if request.max_retries is not None else settings.MAX_RETRIES,
            vector_store=self.vector_store,
        )

        response = QueryResponse(
            query_id=query_id,
            question=request.question,
            answer=state.get("generation", ""),
            citations=state.get("citations", []),
            is_fallback=state.get("is_fallback", False),
            retry_count=state.get("retry_count", 0),
            query_intent=state.get("query_intent", "conceptual"),
            search_query=state.get("search_query", request.question),
            retrieved_chunks_count=len(state.get("documents", [])),
            relevant_chunks_count=len(state.get("relevant_documents", [])),
        )

        logger.info(
            f"Query [{query_id}] completed. Fallback: {response.is_fallback}, "
            f"Retries: {response.retry_count}, Citations: {response.citations}"
        )
        return response
