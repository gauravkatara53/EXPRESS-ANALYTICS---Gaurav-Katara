from typing import Dict, List, Optional

from src.api.schemas import DocumentInfo, DocumentListResponse
from src.retrieval.vector_store import ChromaVectorStore
from src.utils.logger import get_logger

logger = get_logger("document_service")


class DocumentService:
    """Service for querying and reporting the catalog of indexed documents from ChromaDB."""

    def __init__(self, vector_store: Optional[ChromaVectorStore] = None):
        self.vector_store = vector_store or ChromaVectorStore()

    def list_documents(self) -> DocumentListResponse:
        """Retrieves and aggregates all document metadata from the ChromaDB collection."""
        data = self.vector_store.collection.get(include=["metadatas"])
        metadatas = data.get("metadatas", []) or []

        doc_map: Dict[str, Dict] = {}

        for meta in metadatas:
            filename = meta.get("source_filename")
            if not filename:
                continue

            if filename not in doc_map:
                doc_map[filename] = {
                    "source_filename": filename,
                    "document_title": meta.get("document_title", filename),
                    "file_type": meta.get("file_type", ".txt"),
                    "source_path": meta.get("source_path", ""),
                    "chunk_count": 0,
                    "total_characters": 0,
                }

            doc_map[filename]["chunk_count"] += 1
            doc_map[filename]["total_characters"] += int(meta.get("character_count", 0))

        doc_list = [
            DocumentInfo(**item)
            for item in sorted(doc_map.values(), key=lambda x: x["source_filename"])
        ]

        logger.info(
            f"Catalog listing: {len(doc_list)} unique documents, "
            f"{self.vector_store.collection.count()} total chunks in ChromaDB."
        )

        return DocumentListResponse(
            total_documents=len(doc_list),
            total_chunks=self.vector_store.collection.count(),
            documents=doc_list,
        )
