from pathlib import Path
from typing import List, Optional
from fastapi import UploadFile

from src.api.schemas import DocumentChunk, IngestUploadResponse, LoadedDocument
from src.config import settings
from src.ingestion.loaders import SUPPORTED_EXTENSIONS, DocumentLoader
from src.ingestion.splitter import DocumentSplitter
from src.retrieval.vector_store import ChromaVectorStore
from src.utils.logger import get_logger

logger = get_logger("ingestion_service")


class IngestionService:
    """Business logic for file upload handling, sanitization, chunking, and ChromaDB indexing."""

    def __init__(
        self,
        vector_store: Optional[ChromaVectorStore] = None,
        loader: Optional[DocumentLoader] = None,
        splitter: Optional[DocumentSplitter] = None,
    ):
        self.vector_store = vector_store or ChromaVectorStore()
        self.loader = loader or DocumentLoader()
        self.splitter = splitter or DocumentSplitter(
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP,
        )
        self.docs_dir = Path(settings.DOCUMENTS_DIR).resolve()
        self.docs_dir.mkdir(parents=True, exist_ok=True)

    async def ingest_uploaded_files(self, files: List[UploadFile]) -> IngestUploadResponse:
        """Processes and indexes multiple uploaded documentation files."""
        uploaded_filenames: List[str] = []
        failed_files: List[str] = []
        loaded_documents: List[LoadedDocument] = []

        for upload_file in files:
            raw_name = upload_file.filename or "unnamed_document.txt"
            safe_name = Path(raw_name).name  # Strips directory path traversal

            # Validate non-empty sanitized name
            if not safe_name or safe_name in {".", ".."}:
                failed_files.append(f"{raw_name} (invalid filename)")
                continue

            ext = Path(safe_name).suffix.lower()
            if ext not in SUPPORTED_EXTENSIONS:
                failed_files.append(f"{safe_name} (unsupported format '{ext}')")
                continue

            dest_path = self.docs_dir / safe_name

            try:
                content_bytes = await upload_file.read()
                if len(content_bytes) == 0:
                    failed_files.append(f"{safe_name} (empty file, 0 bytes)")
                    continue

                # Write safely to target document directory
                with open(dest_path, "wb") as f_out:
                    f_out.write(content_bytes)

                # Parse document
                doc = self.loader.load_file(dest_path)
                if doc is not None:
                    loaded_documents.append(doc)
                    uploaded_filenames.append(safe_name)
                    logger.info(f"Uploaded and parsed file '{safe_name}' ({doc.character_count} chars).")
                else:
                    failed_files.append(f"{safe_name} (empty or whitespace-only)")

            except Exception as e:
                logger.error(f"Failed to process file '{safe_name}': {e}")
                failed_files.append(f"{safe_name} (error: {str(e)})")

        # Split and index loaded documents
        if loaded_documents:
            chunks: List[DocumentChunk] = self.splitter.split_documents(loaded_documents)
            index_result = self.vector_store.index_chunks(chunks)
            total_indexed = index_result.chunks_indexed
            total_updated = index_result.chunks_updated
            total_skipped = index_result.chunks_skipped_duplicates
        else:
            total_indexed = 0
            total_updated = 0
            total_skipped = 0

        return IngestUploadResponse(
            uploaded_files=uploaded_filenames,
            total_chunks_indexed=total_indexed,
            total_chunks_updated=total_updated,
            total_chunks_skipped=total_skipped,
            total_store_chunks=self.vector_store.collection.count(),
            failed_files=failed_files,
        )
