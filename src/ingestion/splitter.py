from typing import List, Optional
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.api.schemas import ChunkMetadata, DocumentChunk, LoadedDocument, IngestionPreviewSummary
from src.utils.logger import get_logger

logger = get_logger("document_splitter")


class DocumentSplitter:
    """Splits technical documents into semantically coherent chunks using RecursiveCharacterTextSplitter.

    Design rationale:
    - Recursive splitting inspects double line breaks (paragraphs/sections), single line breaks
      (code blocks / lists), and spaces before breaking words, preserving technical context.
    - Chunk size of 800 chars (~150-200 tokens) provides enough room for technical explanations
      and code snippets without diluting vector retrieval precision.
    - Chunk overlap of 150 chars (~30-40 tokens) ensures boundary continuity between adjacent blocks.
    """

    def __init__(
        self,
        chunk_size: int = 800,
        chunk_overlap: int = 150,
        separators: Optional[List[str]] = None,
    ):
        if chunk_overlap >= chunk_size:
            raise ValueError(
                f"chunk_overlap ({chunk_overlap}) must be strictly less than chunk_size ({chunk_size})"
            )

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        # Hierarchical separators tailored for technical docs
        self.separators = separators or ["\n\n", "\n", " ", ""]

        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=self.separators,
            is_separator_regex=False,
        )

    def split_document(self, document: LoadedDocument) -> List[DocumentChunk]:
        """Splits a single LoadedDocument into chunks with enriched metadata.

        Args:
            document: The loaded document to split.

        Returns:
            List of DocumentChunk instances with preserved metadata.
        """
        raw_texts = self._splitter.split_text(document.raw_content)
        total_chunks = len(raw_texts)

        chunks: List[DocumentChunk] = []
        for idx, text in enumerate(raw_texts):
            clean_text = text.strip()
            if not clean_text:
                continue

            metadata = ChunkMetadata(
                source_filename=document.source_filename,
                source_path=document.source_path,
                document_title=document.document_title,
                file_type=document.file_type,
                chunk_index=idx,
                total_chunks=total_chunks,
                character_count=len(clean_text),
            )
            chunks.append(DocumentChunk(content=clean_text, metadata=metadata))

        logger.debug(
            f"Split '{document.source_filename}' ({document.character_count:,} chars) "
            f"into {len(chunks)} chunks."
        )
        return chunks

    def split_documents(self, documents: List[LoadedDocument]) -> List[DocumentChunk]:
        """Splits multiple LoadedDocuments into a flattened list of DocumentChunks.

        Args:
            documents: List of loaded raw documents.

        Returns:
            Flattened list of all resulting DocumentChunks.
        """
        all_chunks: List[DocumentChunk] = []
        for doc in documents:
            chunks = self.split_document(doc)
            all_chunks.extend(chunks)

        logger.info(
            f"Chunking complete: {len(documents)} documents produced {len(all_chunks)} chunks."
        )
        return all_chunks

    def generate_summary(
        self,
        total_files_found: int,
        loaded_documents: List[LoadedDocument],
        chunks: List[DocumentChunk],
        skipped_count: int = 0,
        preview_limit: int = 5,
    ) -> IngestionPreviewSummary:
        """Constructs an IngestionPreviewSummary with statistics and sample preview chunks."""
        avg_len = (
            sum(len(c.content) for c in chunks) / len(chunks) if chunks else 0.0
        )
        return IngestionPreviewSummary(
            total_files_found=total_files_found,
            total_files_loaded=len(loaded_documents),
            total_files_skipped=skipped_count,
            total_chunks_created=len(chunks),
            average_chunk_length=round(avg_len, 2),
            documents=loaded_documents,
            chunks_preview=chunks[:preview_limit],
        )
