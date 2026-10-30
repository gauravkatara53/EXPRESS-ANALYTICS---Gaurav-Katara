# Express Analytics | RAG-Based Technical Documentation Assistant

An enterprise-ready Retrieval-Augmented Generation (RAG) system built with **FastAPI**, **LangChain / LangGraph**, and **ChromaDB**, designed to ingest, index, and query technical documentation with self-corrective capabilities.

---

## Current Status: Phase 1, Phase 2, & Phase 3 Implemented

- **Phase 1: Project Setup & Architecture** (Completed)
  - Modular project layout, configuration management with `pydantic-settings` and `.env`.
  - FastAPI application skeleton with healthcheck, CORS, lifecycle hooks, and structured logging.
- **Phase 2: Document Ingestion & Chunking** (Completed)
  - Multi-format ingestion support: **Markdown (`.md`)**, **Plain Text (`.txt`)**, and **HTML (`.html`)**.
  - Hierarchical semantic chunking with `RecursiveCharacterTextSplitter` (chunk size: 800 chars, overlap: 150 chars).
  - Rich metadata tracking (filename, document title, relative path, chunk index, total chunks, character count).
  - Production-grade error handling (empty files, whitespace-only files, unsupported formats, encoding fallbacks).
  - Included 4 sample technical documentation files.
- **Phase 3: Embeddings & ChromaDB Vector Store** (Completed)
  - **ChromaDB** integrated as a persistent local vector database (`chroma_db/`).
  - **Google Gemini Embedding Model** integration (`models/text-embedding-004`) via `langchain-google-genai`.
  - **Deterministic/Offline Embedding Provider** for millisecond CI execution and tests without API keys.
  - **Duplicate Indexing Prevention**: Computes deterministic chunk IDs (`filename#chunk_index`) and SHA256 content hashes to skip duplicate chunks on repeated runs.
  - **Document Updates & Deletions**: Detects modified chunks and updates them in ChromaDB (`collection.upsert`).
  - **Semantic Similarity Search**: Cosine similarity retrieval with configurable `top_k` and `score_threshold` filtering.
  - **Metadata Preservation**: All retrieved chunks preserve citations and document provenance.
  - Full automated test suite (25 tests, 100% passing) and CLI verification tools.

---

## Project Structure

```text
express-analytics-rag/
├── .env                          # Local environment configuration
├── .env.example                  # Template configuration with descriptions
├── .gitignore                    # Git ignore file (cache, venv, vector stores)
├── README.md                     # Comprehensive project guide and documentation
├── requirements.txt              # Project dependencies
├── verify_ingestion.py           # Standalone CLI tool to verify loading & chunking
├── verify_vector_store.py        # Standalone CLI tool to verify indexing & retrieval
│
├── chroma_db/                    # Persistent ChromaDB storage directory
│
├── data/
│   └── documents/                # Corpus of technical documentation files
│       ├── fastapi_overview.md       # Markdown document on FastAPI architecture
│       ├── pydantic_validation.md    # Markdown document on Pydantic V2 models
│       ├── python_typing_guide.txt   # Plain text guide on Python typing system
│       └── langgraph_core_concepts.html # HTML document on LangGraph workflows
│
├── src/
│   ├── __init__.py
│   ├── config.py                 # Pydantic Settings configuration loader
│   ├── main.py                   # FastAPI main entry point & lifespan handlers
│   │
│   ├── api/                      # Web API Layer
│   │   ├── __init__.py
│   │   ├── routes.py             # API endpoints (health, info, ingest, search, stats)
│   │   └── schemas.py            # Pydantic schemas (DocumentChunk, SearchResult, etc.)
│   │
│   ├── embeddings/               # Vector Embeddings Module
│   │   ├── __init__.py
│   │   └── models.py             # Google Gemini Embeddings & Deterministic test provider
│   │
│   ├── ingestion/                # Document Ingestion & Processing Layer
│   │   ├── __init__.py
│   │   ├── loaders.py            # Multi-format document parser (MD, TXT, HTML)
│   │   └── splitter.py           # RecursiveCharacterTextSplitter chunking engine
│   │
│   ├── retrieval/                # Vector Store & Search Layer
│   │   ├── __init__.py
│   │   └── vector_store.py       # ChromaVectorStore (persistence, deduplication, search)
│   │
│   └── utils/                    # Shared Utilities
│       ├── __init__.py
│       └── logger.py             # Centralized structured logger
│
└── tests/                        # Automated Pytest Suite (25 tests, 100% passing)
    ├── __init__.py
    ├── test_api.py               # Tests for FastAPI endpoints
    ├── test_embeddings.py        # Tests for embedding generation and factory
    ├── test_loaders.py           # Tests for document loaders and edge cases
    ├── test_splitter.py          # Tests for text splitting and metadata integrity
    └── test_vector_store.py      # Tests for ChromaDB persistence, search, deduplication
```

---

## Phase 3 Architecture & Data Flow

```text
User Question / Corpus
         │
         ▼
[ Document Ingestion & Chunking ]  (Phase 2)
         │  (Markdown, TXT, HTML -> 800-char chunks)
         ▼
[ Deterministic ID & Content Hash ]
  cid = "fastapi_overview.md#chunk_0"
  chash = SHA256(content)[:16]
         │
         ▼
[ ChromaDB Deduplication Check ]
  Is cid in ChromaDB with identical chash?
  ├── YES ──> Skip re-embedding (Duplicate prevented!)
  └── NO  ──> Proceed to Embedding
         │
         ▼
[ Embedding Generation ] (Google Gemini text-embedding-004)
         │
         ▼
[ Persistent ChromaDB Store ] (chroma_db/)
  Stores: Embeddings + Text + Rich Metadata (cosine space)
         │
         ▼
[ Semantic Similarity Search ] (Top-K, Threshold Filtering)
  Returns: SearchResult with Cosine Similarity Score & Provenance Metadata
```

---

## Technical Explanation of Key Concepts

### 1. Vector Embeddings
- A **vector embedding** is a numerical representation of semantic meaning. Rather than matching exact keywords, embedding models map natural language into high-dimensional geometric space (e.g., 768 dimensions in Google Gemini `models/text-embedding-004`).
- Sentences with similar meanings (such as "How to validate models in Pydantic?" and "Using BaseModel for request schema validation") end up located close to one another in this vector space, even if they share few identical words.

### 2. Cosine Similarity & Distance
- In ChromaDB, we configure collections with `{"hnsw:space": "cosine"}`.
- Given two normalized vectors $\mathbf{u}$ and $\mathbf{v}$, the cosine similarity represents the angle between them:
  $$\text{Cosine Similarity} = \frac{\mathbf{u} \cdot \mathbf{v}}{\|\mathbf{u}\| \|\mathbf{v}\|}$$
- ChromaDB returns **cosine distance**: $\text{distance} = 1 - \text{cosine similarity}$.
- Our retrieval engine normalizes this into an intuitive similarity score:
  $$\text{score} = \max(0.0, 1.0 - \text{distance})$$
  Where `1.0` is an identical match and `0.0` represents orthogonality / zero semantic overlap.

### 3. Duplicate Prevention & Update Logic
- In production pipelines, document ingestion may run frequently (e.g. cron jobs, re-indexing scripts). Re-embedding already indexed documents wastes API tokens and slows down execution.
- We implement **content-hash deduplication**:
  1. Every chunk is assigned a deterministic ID: `{source_filename}#chunk_{chunk_index}`.
  2. We compute a SHA-256 fingerprint of the chunk content: `content_hash = sha256(content)[:16]`.
  3. When `index_chunks()` is called, we query ChromaDB for existing chunk IDs. If the chunk ID already exists and its `content_hash` has not changed, **the chunk is skipped without calling the embedding API**.
  4. If the content has changed, ChromaDB's `upsert` updates the vector and metadata.

---

## Installation & Setup

```bash
# 1. Virtual Environment
source .venv/bin/activate

# 2. Dependencies installation
pip install -r requirements.txt

# 3. Environment configuration
cp .env.example .env
```

### Optional: Google Gemini API Key
To use live Google Gemini embeddings (`models/text-embedding-004`), add your API key to `.env`:
```env
GEMINI_API_KEY="your-gemini-api-key-here"
```
*(If no API key is set, the system automatically uses `DeterministicTestEmbeddings`, allowing the entire test suite and verification scripts to run offline with zero configuration).*

---

## Verification & Execution

### 1. Run Vector Store Verification CLI
```bash
python verify_vector_store.py
```

**Expected Output:**
```text
===========================================================================
 Express Analytics RAG Assistant - Phase 3 Vector Store Verifier 
===========================================================================
Chroma Directory : chroma_db
Collection Name  : technical_documentation
Embedding Model  : models/text-embedding-004
Active Provider  : gemini
---------------------------------------------------------------------------

[Step 1/5] Loading & Splitting Technical Documentation Corpus...
-> Generated 12 chunks across 4 documents.

[Step 2/5] Indexing Chunks into ChromaDB (First Run)...
-> Submitted: 12
-> Indexed:   12
-> Updated:   0
-> Skipped:   0 (duplicates)
-> Total in store: 12

[Step 3/5] Testing Duplicate Indexing Prevention (Second Run)...
-> Submitted: 12
-> Indexed:   0 (expected 0)
-> Skipped:   12 (expected 12)
-> SUCCESS: Duplicate indexing was safely prevented.

[Step 4/5] Verifying Disk Persistence Across Client Re-instantiations...
-> Persisted chunks detected on disk: 12
-> Indexed documents: ['fastapi_overview.md', 'langgraph_core_concepts.html', 'pydantic_validation.md', 'python_typing_guide.txt']
-> SUCCESS: Chunks and metadata persisted correctly on disk.

[Step 5/5] Executing Semantic Similarity Queries with Top-K & Metadata...

--- Query #1: "How do I define data models with validation in Pydantic?" ---
Top Matched File    : pydantic_validation.md
Document Title      : Pydantic V2 Data Validation and Serialization
Similarity Score    : 0.2716
Chunk Position      : 1 of 3
Citation Path       : data/documents/pydantic_validation.md
...
===========================================================================
 ALL PHASE 3 VERIFICATIONS PASSED: ChromaDB & Embeddings Ready 
===========================================================================
```

---

### 2. Run Automated Pytest Suite
```bash
pytest -v
```

Expected output (`25 passed in 2.29s`):
```text
tests/test_api.py::test_health_check_endpoint PASSED                     [  4%]
tests/test_api.py::test_system_info_endpoint PASSED                      [  8%]
tests/test_api.py::test_ingest_preview_endpoint PASSED                   [ 12%]
tests/test_api.py::test_ingest_preview_invalid_overlap PASSED            [ 16%]
tests/test_api.py::test_index_and_search_endpoints PASSED                [ 20%]
tests/test_embeddings.py::test_deterministic_embeddings_properties PASSED [ 24%]
tests/test_embeddings.py::test_deterministic_batch_embedding PASSED      [ 28%]
tests/test_embeddings.py::test_get_embeddings_model_factory PASSED       [ 32%]
tests/test_loaders.py::test_load_markdown_document PASSED                [ 36%]
tests/test_loaders.py::test_load_text_document PASSED                    [ 40%]
tests/test_loaders.py::test_load_html_document PASSED                    [ 44%]
tests/test_loaders.py::test_load_empty_file PASSED                       [ 48%]
tests/test_loaders.py::test_load_whitespace_only_file PASSED             [ 52%]
tests/test_loaders.py::test_unsupported_file_format PASSED               [ 56%]
tests/test_loaders.py::test_nonexistent_file PASSED                      [ 60%]
tests/test_loaders.py::test_directory_batch_load PASSED                  [ 64%]
tests/test_splitter.py::test_chunking_with_metadata PASSED               [ 68%]
tests/test_splitter.py::test_chunk_size_and_overlap_validation PASSED    [ 72%]
tests/test_splitter.py::test_batch_splitting PASSED                      [ 76%]
tests/test_splitter.py::test_summary_generation PASSED                   [ 80%]
tests/test_vector_store.py::test_chroma_indexing_and_duplicate_prevention PASSED [ 84%]
tests/test_vector_store.py::test_chroma_update_modified_chunk PASSED     [ 88%]
tests/test_vector_store.py::test_chroma_disk_persistence PASSED          [ 92%]
tests/test_vector_store.py::test_chroma_similarity_search_and_metadata PASSED [ 96%]
tests/test_vector_store.py::test_chroma_delete_document PASSED           [100%]

============================== 25 passed in 2.29s ==============================
```

---

### 3. API Endpoints for Vector Search

Start the API server:
```bash
uvicorn src.main:app --reload --host 0.0.0.0 --port 8000
```

#### 1. Ingest & Index Corpus into ChromaDB
```bash
curl -X POST "http://localhost:8000/api/v1/ingest/index"
```
Response:
```json
{
  "total_submitted": 12,
  "chunks_indexed": 12,
  "chunks_updated": 0,
  "chunks_skipped_duplicates": 0,
  "total_store_chunks": 12
}
```

#### 2. Execute Semantic Similarity Search
```bash
curl -X POST "http://localhost:8000/api/v1/search" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "How do I validate request models in Pydantic?",
    "top_k": 2,
    "score_threshold": 0.0
  }'
```
Response:
```json
{
  "query": "How do I validate request models in Pydantic?",
  "total_results": 2,
  "results": [
    {
      "chunk_id": "pydantic_validation.md#chunk_0",
      "content": "# Pydantic V2 Data Validation and Serialization\n\nPydantic is the most widely used data validation library for Python...",
      "metadata": {
        "source_filename": "pydantic_validation.md",
        "source_path": "data/documents/pydantic_validation.md",
        "document_title": "Pydantic V2 Data Validation and Serialization",
        "file_type": ".md",
        "chunk_index": 0,
        "total_chunks": 3,
        "character_count": 766,
        "content_hash": "a4d3f5e8b1c90a2e"
      },
      "score": 0.2716
    }
  ]
}
```

#### 3. Inspect Vector Store Statistics
```bash
curl -X GET "http://localhost:8000/api/v1/store/stats"
```
Response:
```json
{
  "collection_name": "technical_documentation",
  "total_chunks": 12,
  "persist_directory": "/.../chroma_db",
  "indexed_documents": [
    "fastapi_overview.md",
    "langgraph_core_concepts.html",
    "pydantic_validation.md",
    "python_typing_guide.txt"
  ]
}
```

---

## Next Steps
- **Phase 4**: Self-Corrective LangGraph Workflow (Query analysis, retrieval node, document grading node, query rewrite conditional edge, and grounded answer generation).
- **Phase 5**: Production FastAPI Endpoints (`/query`, `/ingest`, `/documents`, `/feedback`) & UI.
