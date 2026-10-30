import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import chromadb
from chromadb.config import Settings as ChromaSettings
from langchain_core.embeddings import Embeddings

from src.api.schemas import ChunkMetadata, DocumentChunk, IndexingResult, SearchResult
from src.config import settings
from src.embeddings.models import get_embeddings_model
from src.utils.logger import get_logger

logger = get_logger("chroma_store")


class ChromaVectorStore:
    """Manages persistent indexing and semantic similarity search using ChromaDB.

    Features:
    - Persistent local on-disk storage across application restarts.
    - Deterministic ID generation and content-hash checking to prevent duplicate indexing.
    - Document addition, updating, and deletion support.
    - Cosine similarity ranking with configurable top-k and score threshold filtering.
    - Full source metadata preservation for downstream citation generation.
    """

    def __init__(
        self,
        persist_directory: Optional[Path] = None,
        collection_name: Optional[str] = None,
        embeddings: Optional[Embeddings] = None,
    ):
        self.persist_directory = Path(persist_directory or settings.CHROMA_PERSIST_DIRECTORY).resolve()
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        self.collection_name = collection_name or settings.CHROMA_COLLECTION_NAME
        self.embeddings = embeddings or get_embeddings_model()

        # Initialize persistent client
        self.client = chromadb.PersistentClient(
            path=str(self.persist_directory),
            settings=ChromaSettings(anonymized_telemetry=False, allow_reset=True),
        )

        # Get or create collection configured for cosine distance
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info(
            f"Initialized ChromaVectorStore (Collection: '{self.collection_name}', "
            f"Path: '{self.persist_directory}', Total Chunks: {self.collection.count()})"
        )

    @staticmethod
    def _compute_chunk_id(chunk: DocumentChunk) -> str:
        """Generates a stable, deterministic ID for a document chunk."""
        return f"{chunk.metadata.source_filename}#chunk_{chunk.metadata.chunk_index}"

    @staticmethod
    def _compute_content_hash(text: str) -> str:
        """Computes a SHA-256 fingerprint of the chunk content."""
        return hashlib.sha256(text.strip().encode("utf-8")).hexdigest()[:16]

    def _metadata_to_dict(self, metadata: ChunkMetadata, content_hash: str) -> Dict[str, str]:
        """Converts ChunkMetadata into a dictionary of primitive types for ChromaDB storage."""
        return {
            "source_filename": metadata.source_filename,
            "source_path": metadata.source_path,
            "document_title": metadata.document_title,
            "file_type": metadata.file_type,
            "chunk_index": int(metadata.chunk_index),
            "total_chunks": int(metadata.total_chunks),
            "character_count": int(metadata.character_count),
            "content_hash": content_hash,
        }

    def _dict_to_metadata(self, meta_dict: Dict) -> ChunkMetadata:
        """Reconstructs ChunkMetadata from ChromaDB metadata dictionary."""
        return ChunkMetadata(
            source_filename=str(meta_dict.get("source_filename", "")),
            source_path=str(meta_dict.get("source_path", "")),
            document_title=str(meta_dict.get("document_title", "")),
            file_type=str(meta_dict.get("file_type", "")),
            chunk_index=int(meta_dict.get("chunk_index", 0)),
            total_chunks=int(meta_dict.get("total_chunks", 1)),
            character_count=int(meta_dict.get("character_count", 0)),
            content_hash=meta_dict.get("content_hash"),
        )

    def index_chunks(self, chunks: List[DocumentChunk]) -> IndexingResult:
        """Indexes document chunks into ChromaDB with duplicate detection and update support.

        - If a chunk already exists with identical content, it is skipped.
        - If a chunk exists with modified content, it is updated.
        - If a chunk does not exist, it is embedded and indexed.

        Args:
            chunks: List of DocumentChunk instances to index.

        Returns:
            IndexingResult with statistics on indexed, updated, and skipped chunks.
        """
        if not chunks:
            return IndexingResult(
                total_submitted=0,
                chunks_indexed=0,
                chunks_updated=0,
                chunks_skipped_duplicates=0,
                total_store_chunks=self.collection.count(),
            )

        # Build candidate IDs and hashes
        candidate_map = {}
        for chunk in chunks:
            cid = self._compute_chunk_id(chunk)
            chash = self._compute_content_hash(chunk.content)
            candidate_map[cid] = (chunk, chash)

        all_candidate_ids = list(candidate_map.keys())

        # Check existing entries in ChromaDB in batches
        existing_lookup: Dict[str, str] = {}
        batch_size = 500
        for i in range(0, len(all_candidate_ids), batch_size):
            batch_ids = all_candidate_ids[i : i + batch_size]
            existing = self.collection.get(ids=batch_ids, include=["metadatas"])
            for eid, emeta in zip(existing["ids"], existing["metadatas"]):
                existing_lookup[eid] = emeta.get("content_hash", "")

        ids_to_upsert: List[str] = []
        documents_to_upsert: List[str] = []
        metadatas_to_upsert: List[Dict] = []
        new_count = 0
        updated_count = 0
        skipped_count = 0

        for cid, (chunk, chash) in candidate_map.items():
            if cid in existing_lookup:
                if existing_lookup[cid] == chash:
                    skipped_count += 1
                    continue
                else:
                    updated_count += 1
            else:
                new_count += 1

            ids_to_upsert.append(cid)
            documents_to_upsert.append(chunk.content)
            metadatas_to_upsert.append(self._metadata_to_dict(chunk.metadata, chash))

        # Compute embeddings only for new or updated chunks
        if ids_to_upsert:
            logger.info(
                f"Generating embeddings for {len(ids_to_upsert)} chunks "
                f"({new_count} new, {updated_count} updated)..."
            )
            embeddings_list = self.embeddings.embed_documents(documents_to_upsert)

            self.collection.upsert(
                ids=ids_to_upsert,
                documents=documents_to_upsert,
                embeddings=embeddings_list,
                metadatas=metadatas_to_upsert,
            )

        total_in_store = self.collection.count()
        logger.info(
            f"Indexing completed: {new_count} indexed, {updated_count} updated, "
            f"{skipped_count} skipped duplicates. Total in store: {total_in_store}"
        )

        return IndexingResult(
            total_submitted=len(chunks),
            chunks_indexed=new_count,
            chunks_updated=updated_count,
            chunks_skipped_duplicates=skipped_count,
            total_store_chunks=total_in_store,
        )

    def similarity_search(
        self,
        query: str,
        top_k: int = 4,
        score_threshold: Optional[float] = None,
        filter_dict: Optional[Dict] = None,
    ) -> List[SearchResult]:
        """Performs semantic similarity search to retrieve the most relevant chunks for a query.

        Args:
            query: Natural language query.
            top_k: Maximum number of chunks to return.
            score_threshold: Minimum cosine similarity score (0.0 to 1.0) to qualify.
            filter_dict: Optional metadata filter for ChromaDB (e.g., {"file_type": ".md"}).

        Returns:
            List of SearchResult objects sorted by similarity score descending.
        """
        clean_query = query.strip()
        if not clean_query:
            return []

        total_items = self.collection.count()
        if total_items == 0:
            logger.warning("Similarity search attempted on an empty ChromaDB collection.")
            return []

        actual_k = min(top_k, total_items)

        # Generate query vector
        query_embedding = self.embeddings.embed_query(clean_query)

        query_kwargs = {
            "query_embeddings": [query_embedding],
            "n_results": actual_k,
            "include": ["documents", "metadatas", "distances"],
        }
        if filter_dict:
            query_kwargs["where"] = filter_dict

        results = self.collection.query(**query_kwargs)

        search_results: List[SearchResult] = []
        if not results or not results["ids"] or not results["ids"][0]:
            return search_results

        matched_ids = results["ids"][0]
        matched_docs = results["documents"][0]
        matched_metas = results["metadatas"][0]
        matched_distances = results["distances"][0]

        for cid, doc, meta, distance in zip(
            matched_ids, matched_docs, matched_metas, matched_distances
        ):
            # Convert cosine distance into cosine similarity score in [0.0, 1.0]
            # ChromaDB cosine distance: distance = 1 - cosine_sim.
            # Thus cosine_sim = 1.0 - distance.
            similarity = max(0.0, min(1.0, 1.0 - distance))
            similarity_rounded = round(similarity, 4)

            # Apply score threshold filtering if configured
            min_score = (
                score_threshold
                if score_threshold is not None
                else settings.SIMILARITY_SCORE_THRESHOLD
            )
            if min_score is not None and similarity_rounded < min_score:
                continue

            metadata_obj = self._dict_to_metadata(meta)
            search_results.append(
                SearchResult(
                    chunk_id=cid,
                    content=doc,
                    metadata=metadata_obj,
                    score=similarity_rounded,
                )
            )

        logger.debug(
            f"Query '{clean_query[:50]}' returned {len(search_results)} chunks (top_k={top_k})."
        )
        return search_results

    def delete_document(self, filename: str) -> int:
        """Deletes all chunks associated with a specific source filename.

        Args:
            filename: Source filename (e.g., 'fastapi_overview.md').

        Returns:
            Number of chunks removed.
        """
        existing = self.collection.get(
            where={"source_filename": filename},
            include=["metadatas"],
        )
        count = len(existing["ids"])
        if count > 0:
            self.collection.delete(ids=existing["ids"])
            logger.info(f"Deleted {count} chunks for document '{filename}'.")
        return count

    def get_indexed_documents(self) -> List[str]:
        """Returns a list of unique source filenames indexed in ChromaDB."""
        data = self.collection.get(include=["metadatas"])
        if not data or not data["metadatas"]:
            return []
        files = {m.get("source_filename") for m in data["metadatas"] if m.get("source_filename")}
        return sorted(list(files))

    def clear(self) -> None:
        """Clears all records in the collection."""
        self.client.delete_collection(self.collection_name)
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info(f"Cleared ChromaDB collection '{self.collection_name}'.")
