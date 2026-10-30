from src.config import settings
from src.graph.state import GraphState
from src.utils.logger import get_logger

logger = get_logger("graph_edges")


def decide_to_generate_or_retry(state: GraphState) -> str:
    """Conditional Edge routing logic evaluated after Document Grading.

    Routes:
    - If relevant documents exist -> proceed to 'generate'.
    - If no relevant documents exist and retry limit has not been reached -> proceed to 'rewrite'.
    - If no relevant documents exist and retry limit is reached -> proceed to 'generate' (safe fallback).

    Args:
        state: The current GraphState.

    Returns:
        Routing key: 'generate' or 'rewrite'.
    """
    relevant_docs = state.get("relevant_documents", [])
    retry_count = state.get("retry_count", 0)
    max_retries = state.get("max_retries", settings.MAX_RETRIES)

    if relevant_docs:
        logger.info(
            f"--- [Routing Decision] Found {len(relevant_docs)} relevant document(s). "
            f"Routing to -> 'generate' ---"
        )
        return "generate"

    if retry_count < max_retries:
        logger.info(
            f"--- [Routing Decision] No relevant documents found (Retry {retry_count}/{max_retries}). "
            f"Routing to -> 'rewrite' ---"
        )
        return "rewrite"

    logger.warning(
        f"--- [Routing Decision] No relevant documents found and retry limit reached ({retry_count}/{max_retries}). "
        f"Routing to fallback -> 'generate' ---"
    )
    return "generate"

