#!/usr/bin/env python3
"""Standalone CLI verification script for Phase 4: LangGraph Self-Corrective RAG Workflow.

Demonstrates:
1. End-to-end question answering with query analysis, retrieval, grading, and cited generation.
2. Self-corrective query rewriting when retrieved context is irrelevant.
3. Safe fallback generation without hallucination when retries are exhausted.
"""
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).parent))

from src.config import settings
from src.graph.workflow import run_rag_pipeline
from src.ingestion.loaders import DocumentLoader
from src.ingestion.splitter import DocumentSplitter
from src.retrieval.vector_store import ChromaVectorStore
from src.utils.logger import get_logger

logger = get_logger("verify_workflow")


def ensure_corpus_indexed(store: ChromaVectorStore):
    """Ensures documents are indexed in ChromaDB before workflow execution."""
    if store.collection.count() == 0:
        print("[Setup] Indexing documentation corpus into ChromaDB...")
        loader = DocumentLoader()
        docs, _ = loader.load_directory(Path(settings.DOCUMENTS_DIR))
        splitter = DocumentSplitter(
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP,
        )
        chunks = splitter.split_documents(docs)
        store.index_chunks(chunks)
        print(f"[Setup] Indexed {len(chunks)} chunks.")
    else:
        print(f"[Setup] Verified {store.collection.count()} chunks in ChromaDB.")


def main():
    print("=" * 80)
    print(" Express Analytics RAG Assistant - Phase 4 LangGraph Workflow Verifier ")
    print("=" * 80)
    print(f"Max Retries      : {settings.MAX_RETRIES}")
    print(f"Vector Store     : {settings.CHROMA_PERSIST_DIRECTORY}")
    print(f"Embedding Model  : {settings.GEMINI_EMBEDDING_MODEL}")
    print(f"LLM Model        : {settings.GEMINI_LLM_MODEL}")
    print("-" * 80)

    store = ChromaVectorStore()
    ensure_corpus_indexed(store)

    # --------------------------------------------------------------------------
    # Scenario 1: Relevant Question (Fast-path: Analysis -> Retrieve -> Grade -> Generate)
    # --------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print(" [SCENARIO 1] Technical Question with Relevant Documentation Context ")
    print("=" * 80)
    question_1 = "How do I validate fields and create schemas using Pydantic BaseModel?"
    print(f"User Question: \"{question_1}\"\n")

    result_1 = run_rag_pipeline(question=question_1, max_retries=2, vector_store=store)

    print("--- Execution Trace ---")
    print(f"Classified Intent  : {result_1['query_intent']}")
    print(f"Refined Search     : {result_1['search_query']}")
    print(f"Retrieved Chunks   : {len(result_1['documents'])}")
    print(f"Relevant Chunks    : {len(result_1['relevant_documents'])}")
    print(f"Retries Triggered  : {result_1['retry_count']}")
    print(f"Fallback Mode      : {result_1['is_fallback']}")
    print(f"Citations          : {result_1['citations']}")
    print("\n--- Generated Answer ---")
    print(result_1["generation"])

    assert not result_1["is_fallback"], "Scenario 1 should succeed without fallback."
    assert len(result_1["citations"]) > 0, "Scenario 1 must cite source documents."
    print("\n-> SUCCESS: Scenario 1 completed with verified context and citations.")

    # --------------------------------------------------------------------------
    # Scenario 2: Off-topic Query (Self-Correction -> Rewrite -> Retry Limit -> Safe Fallback)
    # --------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print(" [SCENARIO 2] Off-Topic Query Triggering Self-Correction & Safe Fallback ")
    print("=" * 80)
    question_2 = "What is the recipe for baking chocolate chip cookies?"
    print(f"User Question: \"{question_2}\"\n")

    result_2 = run_rag_pipeline(question=question_2, max_retries=2, vector_store=store)

    print("--- Execution Trace ---")
    print(f"Initial Search     : {question_2}")
    print(f"Final Search Query : {result_2['search_query']}")
    print(f"Retries Triggered  : {result_2['retry_count']} of {result_2['max_retries']}")
    print(f"Relevant Chunks    : {len(result_2['relevant_documents'])}")
    print(f"Fallback Mode      : {result_2['is_fallback']}")
    print(f"Citations          : {result_2['citations']}")
    print("\n--- Generated Answer ---")
    print(result_2["generation"])

    assert result_2["is_fallback"], "Scenario 2 should trigger safe fallback mode."
    assert result_2["retry_count"] >= 2, "Scenario 2 should exhaust retry limit."
    assert "not contain sufficient information" in result_2["generation"].lower() or "sorry" in result_2["generation"].lower(), "Fallback must not hallucinate."
    print("\n-> SUCCESS: Scenario 2 safely rewrote queries and fell back without hallucination.")

    print("\n" + "=" * 80)
    print(" ALL PHASE 4 WORKFLOW VERIFICATIONS PASSED: LangGraph Self-Corrective RAG Ready ")
    print("=" * 80)


if __name__ == "__main__":
    main()

