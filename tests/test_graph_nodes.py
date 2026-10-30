import pytest
from pathlib import Path

from src.api.schemas import ChunkMetadata, SearchResult
from src.graph.nodes import RAGNodes
from src.graph.state import GraphState
from src.llm.models import MockChatModel
from src.retrieval.vector_store import ChromaVectorStore


@pytest.fixture
def mock_llm() -> MockChatModel:
    return MockChatModel()


@pytest.fixture
def sample_search_result() -> SearchResult:
    return SearchResult(
        chunk_id="test_doc.md#chunk_0",
        content="Pydantic models inherit from BaseModel to provide strict type validation and coercion.",
        metadata=ChunkMetadata(
            source_filename="test_doc.md",
            source_path="data/documents/test_doc.md",
            document_title="Test Pydantic Guide",
            file_type=".md",
            chunk_index=0,
            total_chunks=1,
            character_count=87,
        ),
        score=0.88,
    )


def test_query_analysis_node(mock_llm: MockChatModel):
    nodes = RAGNodes(llm=mock_llm)
    state: GraphState = {
        "question": "How do I validate request models in FastAPI?",
        "search_query": "",
        "query_intent": "",
        "documents": [],
        "relevant_documents": [],
        "generation": "",
        "citations": [],
        "retry_count": 0,
        "max_retries": 2,
        "is_fallback": False,
    }

    result = nodes.query_analysis_node(state)
    assert result["query_intent"] == "how-to"
    assert "validate" in result["search_query"].lower()


def test_retrieval_node(tmp_path: Path, mock_llm: MockChatModel, sample_search_result: SearchResult):
    store = ChromaVectorStore(
        persist_directory=tmp_path / "chroma",
        collection_name="test_retrieval_node",
    )
    # Mock search directly
    store.similarity_search = lambda query, top_k, score_threshold: [sample_search_result]

    nodes = RAGNodes(vector_store=store, llm=mock_llm)
    state: GraphState = {
        "question": "Pydantic models",
        "search_query": "Pydantic models",
        "query_intent": "conceptual",
        "documents": [],
        "relevant_documents": [],
        "generation": "",
        "citations": [],
        "retry_count": 0,
        "max_retries": 2,
        "is_fallback": False,
    }

    result = nodes.retrieval_node(state)
    assert len(result["documents"]) == 1
    assert result["documents"][0].metadata.source_filename == "test_doc.md"


def test_document_grading_node(mock_llm: MockChatModel, sample_search_result: SearchResult):
    nodes = RAGNodes(llm=mock_llm)

    # Relevant document test
    state_relevant: GraphState = {
        "question": "Pydantic BaseModel validation",
        "search_query": "Pydantic BaseModel validation",
        "query_intent": "conceptual",
        "documents": [sample_search_result],
        "relevant_documents": [],
        "generation": "",
        "citations": [],
        "retry_count": 0,
        "max_retries": 2,
        "is_fallback": False,
    }
    res_rel = nodes.document_grading_node(state_relevant)
    assert len(res_rel["relevant_documents"]) == 1

    # Irrelevant document test
    irrelevant_chunk = SearchResult(
        chunk_id="space.md#chunk_0",
        content="Jupiter is the largest planet in our solar system.",
        metadata=sample_search_result.metadata,
        score=0.1,
    )
    state_irrelevant: GraphState = {
        "question": "Pydantic BaseModel validation",
        "search_query": "Pydantic BaseModel validation",
        "query_intent": "conceptual",
        "documents": [irrelevant_chunk],
        "relevant_documents": [],
        "generation": "",
        "citations": [],
        "retry_count": 0,
        "max_retries": 2,
        "is_fallback": False,
    }
    res_irrel = nodes.document_grading_node(state_irrelevant)
    assert len(res_irrel["relevant_documents"]) == 0


def test_query_rewrite_node(mock_llm: MockChatModel):
    nodes = RAGNodes(llm=mock_llm)
    state: GraphState = {
        "question": "FastAPI async endpoints",
        "search_query": "FastAPI async endpoints",
        "query_intent": "how-to",
        "documents": [],
        "relevant_documents": [],
        "generation": "",
        "citations": [],
        "retry_count": 0,
        "max_retries": 2,
        "is_fallback": False,
    }

    res = nodes.query_rewrite_node(state)
    assert res["retry_count"] == 1
    assert len(res["search_query"]) > 0


def test_generation_node_with_context(mock_llm: MockChatModel, sample_search_result: SearchResult):
    nodes = RAGNodes(llm=mock_llm)
    state: GraphState = {
        "question": "How does Pydantic validate models?",
        "search_query": "How does Pydantic validate models?",
        "query_intent": "conceptual",
        "documents": [sample_search_result],
        "relevant_documents": [sample_search_result],
        "generation": "",
        "citations": [],
        "retry_count": 0,
        "max_retries": 2,
        "is_fallback": False,
    }

    res = nodes.generation_node(state)
    assert not res["is_fallback"]
    assert "test_doc.md" in res["citations"]
    assert len(res["generation"]) > 0


def test_generation_node_fallback(mock_llm: MockChatModel):
    nodes = RAGNodes(llm=mock_llm)
    state: GraphState = {
        "question": "How to bake a sourdough bread?",
        "search_query": "How to bake a sourdough bread?",
        "query_intent": "how-to",
        "documents": [],
        "relevant_documents": [],
        "generation": "",
        "citations": [],
        "retry_count": 2,
        "max_retries": 2,
        "is_fallback": False,
    }

    res = nodes.generation_node(state)
    assert res["is_fallback"]
    assert res["citations"] == []
    assert "not contain sufficient information" in res["generation"].lower()
