from typing import Optional
from langchain_core.language_models.chat_models import BaseChatModel
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from src.config import settings
from src.graph.edges import decide_to_generate_or_retry
from src.graph.nodes import RAGNodes
from src.graph.state import GraphState
from src.retrieval.vector_store import ChromaVectorStore
from src.utils.logger import get_logger

logger = get_logger("workflow")


def create_rag_workflow(
    vector_store: Optional[ChromaVectorStore] = None,
    llm: Optional[BaseChatModel] = None,
) -> CompiledStateGraph:
    """Constructs and compiles the Self-Corrective LangGraph RAG workflow.

    Workflow topology:
      START
        │
        ▼
      [query_analysis]
        │
        ▼
      [retrieve] <──────────────┐
        │                       │
        ▼                       │ (rewrite)
      [grade_documents] ────────┴──> [rewrite_query]
        │ (relevant or retry limit reached)
        ▼
      [generate]
        │
        ▼
       END
    """
    nodes = RAGNodes(vector_store=vector_store, llm=llm)
    workflow = StateGraph(GraphState)

    # Register Nodes
    workflow.add_node("query_analysis", nodes.query_analysis_node)
    workflow.add_node("retrieve", nodes.retrieval_node)
    workflow.add_node("grade_documents", nodes.document_grading_node)
    workflow.add_node("rewrite_query", nodes.query_rewrite_node)
    workflow.add_node("generate", nodes.generation_node)

    # Connect Deterministic Edges
    workflow.add_edge(START, "query_analysis")
    workflow.add_edge("query_analysis", "retrieve")
    workflow.add_edge("retrieve", "grade_documents")

    # Connect Conditional Self-Corrective Edge
    workflow.add_conditional_edges(
        "grade_documents",
        decide_to_generate_or_retry,
        {
            "generate": "generate",
            "rewrite": "rewrite_query",
        },
    )

    # Query rewrite cycles back to retrieval
    workflow.add_edge("rewrite_query", "retrieve")

    # Final generation terminates workflow
    workflow.add_edge("generate", END)

    compiled = workflow.compile()
    logger.info("Compiled LangGraph Self-Corrective RAG StateGraph.")
    return compiled


def run_rag_pipeline(
    question: str,
    max_retries: Optional[int] = None,
    vector_store: Optional[ChromaVectorStore] = None,
    llm: Optional[BaseChatModel] = None,
) -> GraphState:
    """Executes the Self-Corrective RAG workflow for a given question.

    Args:
        question: Raw user question.
        max_retries: Maximum permitted retrieval retries.
        vector_store: Optional injected ChromaVectorStore instance.
        llm: Optional injected ChatModel instance.

    Returns:
        The final GraphState after workflow completion.
    """
    graph = create_rag_workflow(vector_store=vector_store, llm=llm)

    initial_state: GraphState = {
        "question": question,
        "search_query": question,
        "query_intent": "conceptual",
        "documents": [],
        "relevant_documents": [],
        "generation": "",
        "citations": [],
        "retry_count": 0,
        "max_retries": max_retries if max_retries is not None else settings.MAX_RETRIES,
        "is_fallback": False,
    }

    result = graph.invoke(initial_state)
    return result
