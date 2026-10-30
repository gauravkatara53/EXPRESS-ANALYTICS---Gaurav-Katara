import pytest
from pathlib import Path

from src.api.schemas import ChunkMetadata, DocumentChunk
from src.embeddings.models import DeterministicTestEmbeddings
from src.retrieval.vector_store import ChromaVectorStore


@pytest.fixture
def test_chunks() -> list[DocumentChunk]:
    return [
        DocumentChunk(
            content="FastAPI dependency injection system enables clean context sharing.",
            metadata=ChunkMetadata(
                source_filename="fastapi_doc.md",
                source_path="data/documents/fastapi_doc.md",
                document_title="FastAPI Guide",
                file_type=".md",
                chunk_index=0,
                total_chunks=2,
                character_count=69,
            ),
        ),
        DocumentChunk(
            content="FastAPI supports async def and def handlers with threadpools.",
            metadata=ChunkMetadata(
                source_filename="fastapi_doc.md",
                source_path="data/documents/fastapi_doc.md",
                document_title="FastAPI Guide",
                file_type=".md",
                chunk_index=1,
                total_chunks=2,
                character_count=61,
            ),
        ),
        DocumentChunk(
            content="Pydantic BaseModel validates fields and types using rust core.",
            metadata=ChunkMetadata(
                source_filename="pydantic_doc.md",
                source_path="data/documents/pydantic_doc.md",
                document_title="Pydantic Guide",
                file_type=".md",
                chunk_index=0,
                total_chunks=1,
                character_count=62,
            ),
        ),
    ]


def test_chroma_indexing_and_duplicate_prevention(tmp_path: Path, test_chunks: list[DocumentChunk]):
    embedder = DeterministicTestEmbeddings(dimension=128)
    store = ChromaVectorStore(
        persist_directory=tmp_path / "chroma",
        collection_name="test_collection",
        embeddings=embedder,
    )

    # First indexing run: should index all 3 chunks
    res1 = store.index_chunks(test_chunks)
    assert res1.total_submitted == 3
    assert res1.chunks_indexed == 3
    assert res1.chunks_updated == 0
    assert res1.chunks_skipped_duplicates == 0
    assert res1.total_store_chunks == 3

    # Second indexing run: should skip all 3 chunks as duplicates
    res2 = store.index_chunks(test_chunks)
    assert res2.total_submitted == 3
    assert res2.chunks_indexed == 0
    assert res2.chunks_updated == 0
    assert res2.chunks_skipped_duplicates == 3
    assert res2.total_store_chunks == 3


def test_chroma_update_modified_chunk(tmp_path: Path, test_chunks: list[DocumentChunk]):
    embedder = DeterministicTestEmbeddings(dimension=128)
    store = ChromaVectorStore(
        persist_directory=tmp_path / "chroma",
        collection_name="test_collection",
        embeddings=embedder,
    )

    store.index_chunks(test_chunks)

    # Modify content of the first chunk
    modified_chunk = DocumentChunk(
        content="FastAPI dependency injection is upgraded with new lifespan handlers.",
        metadata=ChunkMetadata(
            source_filename="fastapi_doc.md",
            source_path="data/documents/fastapi_doc.md",
            document_title="FastAPI Guide",
            file_type=".md",
            chunk_index=0,
            total_chunks=2,
            character_count=69,
        ),
    )

    res = store.index_chunks([modified_chunk])
    assert res.chunks_updated == 1
    assert res.chunks_indexed == 0
    assert res.chunks_skipped_duplicates == 0


def test_chroma_disk_persistence(tmp_path: Path, test_chunks: list[DocumentChunk]):
    persist_dir = tmp_path / "chroma_persist"
    embedder = DeterministicTestEmbeddings(dimension=128)

    # Client instance 1: Index chunks
    store1 = ChromaVectorStore(
        persist_directory=persist_dir,
        collection_name="persist_col",
        embeddings=embedder,
    )
    store1.index_chunks(test_chunks)
    assert store1.collection.count() == 3

    # Client instance 2: Open same directory and verify data persists
    store2 = ChromaVectorStore(
        persist_directory=persist_dir,
        collection_name="persist_col",
        embeddings=embedder,
    )
    assert store2.collection.count() == 3
    indexed_docs = store2.get_indexed_documents()
    assert "fastapi_doc.md" in indexed_docs
    assert "pydantic_doc.md" in indexed_docs


def test_chroma_similarity_search_and_metadata(tmp_path: Path, test_chunks: list[DocumentChunk]):
    embedder = DeterministicTestEmbeddings(dimension=128)
    store = ChromaVectorStore(
        persist_directory=tmp_path / "chroma",
        collection_name="search_col",
        embeddings=embedder,
    )
    store.index_chunks(test_chunks)

    # Search for Pydantic
    results = store.similarity_search("Pydantic validation with BaseModel", top_k=2)
    assert len(results) > 0
    top = results[0]
    assert top.metadata.source_filename == "pydantic_doc.md"
    assert top.metadata.document_title == "Pydantic Guide"
    assert top.metadata.source_path == "data/documents/pydantic_doc.md"
    assert top.metadata.chunk_index == 0
    assert top.score > 0.0

    # Search with score threshold
    high_threshold_results = store.similarity_search(
        "Pydantic validation", top_k=5, score_threshold=0.999
    )
    assert len(high_threshold_results) <= len(results)


def test_chroma_delete_document(tmp_path: Path, test_chunks: list[DocumentChunk]):
    embedder = DeterministicTestEmbeddings(dimension=128)
    store = ChromaVectorStore(
        persist_directory=tmp_path / "chroma",
        collection_name="del_col",
        embeddings=embedder,
    )
    store.index_chunks(test_chunks)
    assert store.collection.count() == 3

    deleted = store.delete_document("fastapi_doc.md")
    assert deleted == 2
    assert store.collection.count() == 1
    assert store.get_indexed_documents() == ["pydantic_doc.md"]
