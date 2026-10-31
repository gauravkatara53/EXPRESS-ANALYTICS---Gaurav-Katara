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
    """Handles file uploads, URL fetching, document parsing, chunking, and ChromaDB indexing."""

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
        """Process and index uploaded documentation files."""
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

    async def ingest_urls(self, urls: List[str]) -> IngestUploadResponse:
        """Fetch documentation from URLs, parse, chunk, and index into ChromaDB."""
        import re
        from urllib.parse import urlparse
        import httpx

        uploaded_filenames: List[str] = []
        failed_files: List[str] = []
        loaded_documents: List[LoadedDocument] = []

        async with httpx.AsyncClient(timeout=15.0, follow_redirects=True) as client:
            for url in urls:
                try:
                    parsed = urlparse(url)
                    if not parsed.scheme or not parsed.netloc:
                        failed_files.append(f"{url} (invalid URL scheme/domain)")
                        continue

                    response = await client.get(url)
                    if response.status_code >= 400:
                        failed_files.append(f"{url} (HTTP error {response.status_code})")
                        continue

                    # Create a sanitized filename based on the URL domain and path
                    url_slug = re.sub(r"[^a-zA-Z0-9]+", "_", f"{parsed.netloc}{parsed.path}").strip("_")
                    if not url_slug:
                        url_slug = "web_document"

                    content_type = response.headers.get("content-type", "").lower()
                    raw_text = response.text

                    if "markdown" in content_type or url.endswith(".md"):
                        file_ext = ".md"
                        title = self.loader._extract_markdown_title(raw_text, fallback=url_slug)
                        clean_content = raw_text
                    elif "plain" in content_type or url.endswith(".txt"):
                        file_ext = ".txt"
                        title = self.loader._extract_text_title(raw_text, fallback=url_slug)
                        clean_content = raw_text
                    else:
                        file_ext = ".html"
                        clean_content, title = self.loader._extract_html_content_and_title(raw_text, fallback=url_slug)

                    filename = f"{url_slug}{file_ext}"
                    dest_path = self.docs_dir / filename
                    dest_path.write_text(raw_text, encoding="utf-8")

                    try:
                        rel_path = str(dest_path.relative_to(Path.cwd()))
                    except ValueError:
                        rel_path = str(dest_path)

                    doc = LoadedDocument(
                        source_filename=filename,
                        source_path=rel_path,
                        document_title=title or url,
                        file_type=file_ext,
                        raw_content=clean_content,
                        character_count=len(clean_content),
                        file_size_bytes=len(raw_text.encode("utf-8")),
                    )
                    loaded_documents.append(doc)
                    uploaded_filenames.append(filename)
                    logger.info(f"Fetched and parsed URL '{url}' as '{filename}' ({doc.character_count:,} chars).")
                except Exception as e:
                    logger.error(f"Failed to fetch/process URL '{url}': {e}")
                    failed_files.append(f"{url} (error: {str(e)})")

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

    async def ingest_documents_and_urls(
        self,
        files: Optional[List[UploadFile]] = None,
        urls: Optional[List[str]] = None,
    ) -> IngestUploadResponse:
        """Unified ingestion handler for both file uploads and URLs."""
        uploaded_all: List[str] = []
        failed_all: List[str] = []
        total_indexed = 0
        total_updated = 0
        total_skipped = 0

        if files:
            file_res = await self.ingest_uploaded_files(files)
            uploaded_all.extend(file_res.uploaded_files)
            failed_all.extend(file_res.failed_files)
            total_indexed += file_res.total_chunks_indexed
            total_updated += file_res.total_chunks_updated
            total_skipped += file_res.total_chunks_skipped

        if urls:
            url_res = await self.ingest_urls(urls)
            uploaded_all.extend(url_res.uploaded_files)
            failed_all.extend(url_res.failed_files)
            total_indexed += url_res.total_chunks_indexed
            total_updated += url_res.total_chunks_updated
            total_skipped += url_res.total_chunks_skipped

        return IngestUploadResponse(
            uploaded_files=uploaded_all,
            total_chunks_indexed=total_indexed,
            total_chunks_updated=total_updated,
            total_chunks_skipped=total_skipped,
            total_store_chunks=self.vector_store.collection.count(),
            failed_files=failed_all,
        )
