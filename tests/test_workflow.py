import pytest
from pathlib import Path

from src.api.schemas import ChunkMetadata, DocumentChunk
from src.embeddings.models import DeterministicTestEmbeddings
from src.graph.edges import decide_to_generate_or_retry
from src.graph.state import GraphState
from src.graph.workflow import create_rag_workflow, run_rag_pipeline
from src.llm.models import MockChatModel
from src.retrieval.vector_store import ChromaVectorStore


@pytest.fixture
def populated_vector_store(tmp_path: Path) -> ChromaVectorStore:
    embedder = DeterministicTestEmbeddings(dimension=128)
    store = ChromaVectorStore(
        persist_directory=tmp_path / "chroma_workflow",
        collection_name="workflow_test_col",
        embeddings=embedder,
    )
    chunks = [
        DocumentChunk(
            content="Pydantic models inherit from BaseModel to provide strict data validation in Python.",
            metadata=ChunkMetadata(
                source_filename="pydantic_guide.md",
                source_path="data/documents/pydantic_guide.md",
                document_title="Pydantic Guide",
                file_type=".md",
                chunk_index=0,
                total_chunks=1,
                character_count=85,
            ),
        ),
        DocumentChunk(
            content="FastAPI dependency injection system enables clean context sharing across async handlers.",
            metadata=ChunkMetadata(
                source_filename="fastapi_guide.md",
                source_path="data/documents/fastapi_guide.md",
                document_title="FastAPI Guide",
                file_type=".md",
                chunk_index=0,
                total_chunks=1,
                character_count=89,
            ),
        ),
    ]
    store.index_chunks(chunks)
    return store


def test_decide_to_generate_or_retry():
    # 1. Relevant docs exist -> generate
    state_rel: GraphState = {
        "question": "test",
        "search_query": "test",
        "query_intent": "conceptual",
        "documents": [],
        "relevant_documents": ["dummy_doc"],  # type: ignore
        "generation": "",
        "citations": [],
        "retry_count": 0,
        "max_retries": 2,
        "is_fallback": False,
    }
    assert decide_to_generate_or_retry(state_rel) == "generate"

    # 2. No relevant docs, retries left -> rewrite
    state_rewrite: GraphState = {
        "question": "test",
        "search_query": "test",
        "query_intent": "conceptual",
        "documents": [],
        "relevant_documents": [],
        "generation": "",
        "citations": [],
        "retry_count": 0,
        "max_retries": 2,
        "is_fallback": False,
    }
    assert decide_to_generate_or_retry(state_rewrite) == "rewrite"

    # 3. No relevant docs, retries exhausted -> generate (fallback)
    state_exhausted: GraphState = {
        "question": "test",
        "search_query": "test",
        "query_intent": "conceptual",
        "documents": [],
        "relevant_documents": [],
        "generation": "",
        "citations": [],
        "retry_count": 2,
        "max_retries": 2,
        "is_fallback": False,
    }
    assert decide_to_generate_or_retry(state_exhausted) == "generate"


def test_end_to_end_successful_rag_workflow(populated_vector_store: ChromaVectorStore):
    llm = MockChatModel()
    question = "How do Pydantic models validate data using BaseModel?"

    result = run_rag_pipeline(
        question=question,
        max_retries=2,
        vector_store=populated_vector_store,
        llm=llm,
    )

    assert not result["is_fallback"]
    assert len(result["relevant_documents"]) > 0
    assert len(result["citations"]) > 0
    assert "pydantic_guide.md" in result["citations"]
    assert len(result["generation"]) > 0


def test_end_to_end_irrelevant_query_self_correction(populated_vector_store: ChromaVectorStore):
    llm = MockChatModel()
    question = "How to make a vanilla strawberry birthday cake with chocolate frosting?"

    result = run_rag_pipeline(
        question=question,
        max_retries=2,
        vector_store=populated_vector_store,
        llm=llm,
    )

    # Must trigger self-correction retry loop and safe fallback
    assert result["is_fallback"]
    assert result["retry_count"] >= 2
    assert result["citations"] == []
    assert "not contain sufficient information" in result["generation"].lower()


def test_workflow_configurable_retry_limit(populated_vector_store: ChromaVectorStore):
    llm = MockChatModel()
    question = "Tell me about medieval knights and armor."

    # Test with max_retries = 1
    result = run_rag_pipeline(
        question=question,
        max_retries=1,
        vector_store=populated_vector_store,
        llm=llm,
    )

    assert result["is_fallback"]
    assert result["retry_count"] == 1

