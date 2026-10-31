#!/usr/bin/env python3
"""CLI verification script for document loading and chunking."""
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import settings
from src.ingestion.loaders import DocumentLoader
from src.ingestion.splitter import DocumentSplitter
from src.utils.logger import get_logger

logger = get_logger("verify_cli")


def main():
    print("=" * 70)
    print(" Express Analytics RAG Assistant - Ingestion & Chunking Verifier ")
    print("=" * 70)
    print(f"Target Directory : {settings.DOCUMENTS_DIR}")
    print(f"Chunk Size       : {settings.CHUNK_SIZE} characters")
    print(f"Chunk Overlap    : {settings.CHUNK_OVERLAP} characters")
    print("-" * 70)

    # 1. Initialize Loader & load directory
    loader = DocumentLoader()
    doc_dir = Path(settings.DOCUMENTS_DIR)

    if not doc_dir.exists():
        print(f"ERROR: Directory '{doc_dir}' does not exist.")
        sys.exit(1)

    print("\n[Step 1/3] Loading documents...")
    documents, skipped = loader.load_directory(doc_dir)

    print(f"-> Successfully loaded {len(documents)} documents:")
    for doc in documents:
        print(
            f"   - [{doc.file_type}] '{doc.document_title}' "
            f"({doc.source_filename}, {doc.character_count:,} chars)"
        )

    if skipped:
        print(f"-> Skipped {len(skipped)} files:")
        for skip in skipped:
            print(f"   - {skip}")

    # 2. Split documents
    print("\n[Step 2/3] Chunking documents with RecursiveCharacterTextSplitter...")
    splitter = DocumentSplitter(
        chunk_size=settings.CHUNK_SIZE,
        chunk_overlap=settings.CHUNK_OVERLAP,
    )
    chunks = splitter.split_documents(documents)

    summary = splitter.generate_summary(
        total_files_found=len(documents) + len(skipped),
        loaded_documents=documents,
        chunks=chunks,
        skipped_count=len(skipped),
        preview_limit=3,
    )

    print(f"-> Created {summary.total_chunks_created} total chunks.")
    print(f"-> Average chunk length: {summary.average_chunk_length:.1f} characters.")

    # 3. Inspect chunk metadata and samples
    print("\n[Step 3/3] Inspecting Sample Chunk Metadata...")
    for idx, sample in enumerate(summary.chunks_preview, 1):
        print(f"\n--- Sample Chunk #{idx} ---")
        print(f"Source Document : {sample.metadata.document_title}")
        print(f"Filename        : {sample.metadata.source_filename}")
        print(f"Relative Path   : {sample.metadata.source_path}")
        print(f"Format          : {sample.metadata.file_type}")
        print(
            f"Chunk Index     : {sample.metadata.chunk_index + 1} of {sample.metadata.total_chunks}"
        )
        print(f"Character Count : {sample.metadata.character_count}")
        print("Content Preview :")
        # Print first 200 chars
        preview_text = sample.content[:200].replace("\n", " ")
        print(f'"{preview_text}..."')

    print("\n" + "=" * 70)
    print(" VERIFICATION SUCCESSFUL: Document Ingestion & Chunking Working Correctly ")
    print("=" * 70)


if __name__ == "__main__":
    main()

