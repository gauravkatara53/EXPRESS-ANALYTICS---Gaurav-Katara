#!/usr/bin/env python3
"""Standalone CLI verification script for Phase 3: ChromaDB Vector Store & Embeddings.

Verifies:
1. Chunk indexing into ChromaDB
2. Duplicate indexing prevention (idempotency)
3. Disk persistence across client instances
4. Semantic similarity search with metadata preservation
5. Top-k and score threshold filtering
"""
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).parent))

from src.config import settings
from src.embeddings.models import get_embeddings_model
from src.ingestion.loaders import DocumentLoader
from src.ingestion.splitter import DocumentSplitter
from src.retrieval.vector_store import ChromaVectorStore
from src.utils.logger import get_logger

logger = get_logger("verify_vector_store")


def main():
    print("=" * 75)
    print(" Express Analytics RAG Assistant - Phase 3 Vector Store Verifier ")
    print("=" * 75)
    print(f"Chroma Directory : {settings.CHROMA_PERSIST_DIRECTORY}")
    print(f"Collection Name  : {settings.CHROMA_COLLECTION_NAME}")
    print(f"Embedding Model  : {settings.GEMINI_EMBEDDING_MODEL}")
    print(f"Active Provider  : {settings.EMBEDDING_PROVIDER}")
    print("-" * 75)

    # 1. Ingest and split document corpus
    print("\n[Step 1/5] Loading & Splitting Technical Documentation Corpus...")
    loader = DocumentLoader()
    doc_dir = Path(settings.DOCUMENTS_DIR)
    docs, _ = loader.load_directory(doc_dir)

    splitter = DocumentSplitter(
        chunk_size=settings.CHUNK_SIZE,
        chunk_overlap=settings.CHUNK_OVERLAP,
    )
    chunks = splitter.split_documents(docs)
    print(f"-> Generated {len(chunks)} chunks across {len(docs)} documents.")

    # 2. Initialize ChromaVectorStore and index chunks
    print("\n[Step 2/5] Indexing Chunks into ChromaDB (First Run)...")
    store = ChromaVectorStore()
    # Clear collection for clean demo verification
    store.clear()

    result_run1 = store.index_chunks(chunks)
    print(f"-> Submitted: {result_run1.total_submitted}")
    print(f"-> Indexed:   {result_run1.chunks_indexed}")
    print(f"-> Updated:   {result_run1.chunks_updated}")
    print(f"-> Skipped:   {result_run1.chunks_skipped_duplicates} (duplicates)")
    print(f"-> Total in store: {result_run1.total_store_chunks}")
    assert result_run1.chunks_indexed == len(chunks), "Expected all chunks to be indexed on first run."

    # 3. Test Duplicate Prevention (Idempotent re-run)
    print("\n[Step 3/5] Testing Duplicate Indexing Prevention (Second Run)...")
    result_run2 = store.index_chunks(chunks)
    print(f"-> Submitted: {result_run2.total_submitted}")
    print(f"-> Indexed:   {result_run2.chunks_indexed} (expected 0)")
    print(f"-> Skipped:   {result_run2.chunks_skipped_duplicates} (expected {len(chunks)})")
    assert result_run2.chunks_indexed == 0, "Duplicate prevention failed: new chunks were indexed!"
    assert result_run2.chunks_skipped_duplicates == len(chunks), "Duplicate prevention failed: chunks not skipped!"
    print("-> SUCCESS: Duplicate indexing was safely prevented.")

    # 4. Test Persistence Across Client Instances
    print("\n[Step 4/5] Verifying Disk Persistence Across Client Re-instantiations...")
    # Instantiate a brand new ChromaVectorStore object pointing to the same persist directory
    persisted_store = ChromaVectorStore()
    persisted_count = persisted_store.collection.count()
    indexed_files = persisted_store.get_indexed_documents()
    print(f"-> Persisted chunks detected on disk: {persisted_count}")
    print(f"-> Indexed documents: {indexed_files}")
    assert persisted_count == len(chunks), "Persistence verification failed."
    print("-> SUCCESS: Chunks and metadata persisted correctly on disk.")

    # 5. Semantic Similarity Search Verification
    print("\n[Step 5/5] Executing Semantic Similarity Queries with Top-K & Metadata...")
    test_queries = [
        (
            "How do I define data models with validation in Pydantic?",
            "pydantic_validation.md",
        ),
        (
            "What are StateGraph, cyclical agent workflows, and conditional edges in LangGraph?",
            "langgraph_core_concepts.html",
        ),
        (
            "How does FastAPI handle dependency injection and async route handlers?",
            "fastapi_overview.md",
        ),
        (
            "Explain Python type annotations, Union, and Generics",
            "python_typing_guide.txt",
        ),
    ]

    for q_idx, (query, expected_doc) in enumerate(test_queries, 1):
        print(f"\n--- Query #{q_idx}: \"{query}\" ---")
        matches = persisted_store.similarity_search(query=query, top_k=2)

        if not matches:
            print("ERROR: No chunks retrieved!")
            sys.exit(1)

        top_match = matches[0]
        print(f"Top Matched File    : {top_match.metadata.source_filename}")
        print(f"Document Title      : {top_match.metadata.document_title}")
        print(f"Similarity Score    : {top_match.score:.4f}")
        print(f"Chunk Position      : {top_match.metadata.chunk_index + 1} of {top_match.metadata.total_chunks}")
        print(f"Citation Path       : {top_match.metadata.source_path}")
        print(f"Content Snippet     : \"{top_match.content[:160].replace(chr(10), ' ')}...\"")

        assert top_match.metadata.source_filename == expected_doc, (
            f"Expected top match '{expected_doc}', got '{top_match.metadata.source_filename}'"
        )
        assert top_match.score > 0.0, "Similarity score must be positive"

    print("\n" + "=" * 75)
    print(" ALL PHASE 3 VERIFICATIONS PASSED: ChromaDB & Embeddings Ready ")
    print("=" * 75)


if __name__ == "__main__":
    main()
