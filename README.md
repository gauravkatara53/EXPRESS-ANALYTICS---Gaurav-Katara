# Express Analytics | RAG-Based Technical Documentation Assistant

An enterprise-ready Retrieval-Augmented Generation (RAG) system built with **LangGraph StateGraph**, **FastAPI**, **ChromaDB**, and **Google Gemini**, designed to ingest, index, grade, and answer questions about technical documentation with self-corrective capabilities.

---

## Current Status: All Core Phases Completed (1 – 5)

- **Phase 1: Project Setup & Architecture** (Completed)
  - Clean modular project layout, configuration management with `pydantic-settings` and `.env`.
  - FastAPI application skeleton with healthcheck, CORS, lifecycle hooks, and structured logging.
- **Phase 2: Document Ingestion & Chunking** (Completed)
  - Multi-format ingestion support: **Markdown (`.md`)**, **Plain Text (`.txt`)**, and **HTML (`.html`)**.
  - Hierarchical semantic chunking with `RecursiveCharacterTextSplitter` (chunk size: 800 chars, overlap: 150 chars).
  - Rich metadata tracking (filename, document title, relative path, chunk index, total chunks, character count).
  - Production-grade error handling (empty files, whitespace-only files, unsupported formats, encoding fallbacks).
  - 4 sample technical documentation files.
- **Phase 3: Embeddings & ChromaDB Vector Store** (Completed)
  - **ChromaDB** integrated as a persistent local vector database (`chroma_db/`).
  - **Google Gemini Embedding Model** integration (`models/text-embedding-004`) via `langchain-google-genai`.
  - **Deterministic/Offline Embedding Provider** for millisecond CI execution and tests without API keys.
  - **Duplicate Indexing Prevention**: Computes deterministic chunk IDs (`filename#chunk_index`) and SHA256 content hashes to skip duplicate chunks on repeated runs.
  - **Document Updates & Deletions**: Detects modified chunks and updates them in ChromaDB (`collection.upsert`).
  - **Semantic Similarity Search**: Cosine similarity retrieval with configurable `top_k` and `score_threshold` filtering.
- **Phase 4: Self-Corrective LangGraph Workflow** (Completed)
  - **LangGraph StateGraph** self-corrective cyclical workflow with 5 dedicated nodes:
    1. `query_analysis`: Intent classification and search query expansion.
    2. `retrieve`: Semantic search over ChromaDB chunks with metadata.
    3. `grade_documents`: Self-corrective CRAG evaluator grading chunks as relevant/irrelevant.
    4. `rewrite_query`: Query optimizer adding technical synonyms when retrieval fails.
    5. `generate`: Grounded answer generation strictly from verified context with citations.
  - **Conditional Routing**:
    - Relevant documents found $\rightarrow$ proceed to `generate`.
    - No relevant documents & retries left $\rightarrow$ route to `rewrite_query` $\rightarrow$ re-retrieve.
    - No relevant documents & retry limit reached $\rightarrow$ proceed to `generate` (safe fallback without hallucination).
- **Phase 5: FastAPI Production Endpoints & Feedback Persistence** (Completed)
  - **POST `/api/v1/query`** (and `/query`): Executes the full LangGraph pipeline; returns grounded answer, citations, retry count, and fallback status.
  - **POST `/api/v1/ingest`** (and `/ingest`): Multi-file upload (`.md`, `.txt`, `.html`) with traversal sanitization, chunking, deduplicated ChromaDB indexing, and summary metrics.
  - **GET `/api/v1/documents`** (and `/documents`): Live catalog of indexed documents retrieved directly from ChromaDB with chunk counts, titles, formats, and characters.
  - **POST `/api/v1/feedback`** (and `/feedback`): Persistent user feedback collection (`thumbs_up` / `thumbs_down` + optional comments) stored in SQLite (`feedback.db`).
  - Decoupled service architecture (`src/services/` and `src/database/`) keeping route controllers thin.
  - **45 Automated Unit & Integration Tests (100% Passing)** with zero-cost mocked LLMs.

---

## Project Structure

```text
express-analytics-rag/
├── .env                          # Local environment configuration
├── .env.example                  # Template configuration with descriptions
├── .gitignore                    # Git ignore file (cache, venv, databases, vector stores)
├── README.md                     # Comprehensive project guide and documentation
├── requirements.txt              # Project dependencies (FastAPI, LangGraph, ChromaDB, etc.)
├── verify_ingestion.py           # CLI tool to verify Phase 1 & 2 loading & chunking
├── verify_vector_store.py        # CLI tool to verify Phase 3 indexing & retrieval
├── verify_workflow.py            # CLI tool to verify Phase 4 LangGraph self-corrective workflow
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
│   ├── api/                      # Web API Routing & Schemas
│   │   ├── __init__.py
│   │   ├── routes.py             # FastAPI routes (/query, /ingest, /documents, /feedback, etc.)
│   │   └── schemas.py            # Pydantic request/response schemas
│   │
│   ├── database/                 # SQLite Persistence
│   │   ├── __init__.py
│   │   └── feedback_db.py        # SQLite feedback database manager
│   │
│   ├── embeddings/               # Vector Embeddings Module
│   │   ├── __init__.py
│   │   └── models.py             # Google Gemini Embeddings & Deterministic test provider
│   │
│   ├── graph/                    # LangGraph Self-Corrective RAG Workflow
│   │   ├── __init__.py
│   │   ├── edges.py              # Conditional routing logic (generate vs rewrite)
│   │   ├── nodes.py              # 5 workflow nodes (analysis, retrieve, grade, rewrite, generate)
│   │   ├── state.py              # TypedDict GraphState & Pydantic structured output schemas
│   │   └── workflow.py           # StateGraph builder and compilation pipeline
│   │
│   ├── ingestion/                # Document Ingestion & Processing Layer
│   │   ├── __init__.py
│   │   ├── loaders.py            # Multi-format document parser (MD, TXT, HTML)
│   │   └── splitter.py           # RecursiveCharacterTextSplitter chunking engine
│   │
│   ├── llm/                      # LLM Integration Module
│   │   ├── __init__.py
│   │   └── models.py             # ChatGoogleGenerativeAI & MockChatModel for offline tests
│   │
│   ├── retrieval/                # Vector Store & Search Layer
│   │   ├── __init__.py
│   │   └── vector_store.py       # ChromaVectorStore (persistence, deduplication, search)
│   │
│   ├── services/                 # Business Logic Service Layer
│   │   ├── __init__.py
│   │   ├── document_service.py   # Aggregates indexed documents from ChromaDB
│   │   ├── feedback_service.py   # Validates and records user feedback
│   │   ├── ingestion_service.py  # Handles file uploads, sanitization, chunking & indexing
│   │   └── query_service.py      # Executes LangGraph pipeline & formats response
│   │
│   └── utils/                    # Shared Utilities
│       ├── __init__.py
│       └── logger.py             # Centralized structured logger
│
└── tests/                        # Automated Pytest Suite (45 tests, 100% passing)
    ├── __init__.py
    ├── test_api.py               # Tests for all FastAPI endpoints (/query, /ingest, /documents, /feedback)
    ├── test_embeddings.py        # Tests for embedding generation and factory
    ├── test_feedback_db.py       # Tests for SQLite feedback schema and queries
    ├── test_graph_nodes.py       # Tests for each LangGraph node independently
    ├── test_loaders.py           # Tests for document loaders and edge cases
    ├── test_splitter.py          # Tests for text splitting and metadata integrity
    ├── test_vector_store.py      # Tests for ChromaDB persistence, search, deduplication
    └── test_workflow.py          # Tests for end-to-end self-corrective LangGraph workflow
```

---

## Phase 4 Self-Corrective LangGraph Workflow

```text
                 START
                   │
                   ▼
         [ 1. query_analysis ]
         - Identifies intent (conceptual, how-to, troubleshooting, api_reference)
         - Expands ambiguous phrasing into domain-specific search query
                   │
                   ▼
   ┌─────────> [ 2. retrieve ]
   │           - Searches ChromaDB using current search_query
   │           - Retrieves top-k chunks with provenance metadata
   │               │
   │               ▼
   │     [ 3. grade_documents ]
   │     - LLM evaluates relevance of each retrieved chunk
   │     - Filters out noise/irrelevant content
   │               │
   │               ▼
   │      < decide_to_generate_or_retry >
   │         │
   │         ├── Relevant Chunks Found OR Retries >= MAX_RETRIES (2)
   │         │     │
   │         │     ▼
   │         │  [ 4. generate ]
   │         │  - If context available: grounded answer citing sources
   │         │  - If fallback: safe non-hallucinating message
   │         │     │
   │         │     ▼
   │         │    END
   │         │
   │         └── No Relevant Chunks AND Retries < MAX_RETRIES
   │               │
   │               ▼
   └────── [ 5. rewrite_query ]
           - Increments retry_count
           - Expands query with technical synonyms & alternative keywords
```

---

## Phase 5 FastAPI API Reference

The server exposes both versioned (`/api/v1/...`) and standard root endpoints (`/...`) to support all client conventions:

### 1. `POST /api/v1/query` (and `/query`)
Executes the LangGraph self-corrective RAG workflow.

**Request Body**:
```json
{
  "question": "How do I validate models with Pydantic BaseModel?",
  "top_k": 4,
  "max_retries": 2
}
```

**Response Body**:
```json
{
  "query_id": "993fe01b-c744-4861-bbd4-539091bca72b",
  "question": "How do I validate models with Pydantic BaseModel?",
  "answer": "Based on the technical documentation:\n\nIn Pydantic V2, you define schemas by inheriting from BaseModel...",
  "citations": [
    "pydantic_validation.md"
  ],
  "is_fallback": false,
  "retry_count": 0,
  "query_intent": "conceptual",
  "execution_details": {
    "search_query": "validate models with Pydantic BaseModel",
    "relevant_chunks_count": 4,
    "max_retries": 2
  }
}
```

*When an off-topic question is asked, the pipeline executes self-correction (query rewriting), detects zero relevant technical context, and returns a safe fallback without hallucination (`is_fallback: true`, `citations: []`).*

---

### 2. `POST /api/v1/ingest` (and `/ingest`)
Uploads Markdown (`.md`), Plain Text (`.txt`), or HTML (`.html`) files to the server, parses them, splits them into semantic chunks, and indexes them in ChromaDB.

**Request**: Multipart form data with one or more files under key `files`.

**Response Body**:
```json
{
  "uploaded_files": ["new_documentation.md"],
  "failed_files": [],
  "chunks_indexed": 3,
  "chunks_updated": 0,
  "chunks_skipped_duplicate": 0,
  "total_store_chunks": 15,
  "message": "Ingestion completed: 1 uploaded, 0 failed, 3 chunks indexed."
}
```

---

### 3. `GET /api/v1/documents` (and `/documents`)
Inspects the persistent ChromaDB collection and returns metadata for all indexed documents.

**Response Body**:
```json
{
  "total_documents": 4,
  "total_chunks": 12,
  "documents": [
    {
      "source_filename": "fastapi_overview.md",
      "document_title": "FastAPI Architecture Overview",
      "file_type": ".md",
      "source_path": "data/documents/fastapi_overview.md",
      "chunk_count": 3,
      "total_characters": 2240
    },
    {
      "source_filename": "pydantic_validation.md",
      "document_title": "Pydantic V2 Data Validation Guide",
      "file_type": ".md",
      "source_path": "data/documents/pydantic_validation.md",
      "chunk_count": 3,
      "total_characters": 2180
    }
  ]
}
```

---

### 4. `POST /api/v1/feedback` (and `/feedback`)
Persists user ratings and qualitative feedback into an SQLite database (`feedback.db`).

**Request Body**:
```json
{
  "query_id": "993fe01b-c744-4861-bbd4-539091bca72b",
  "rating": "thumbs_up",
  "comment": "Accurate response with correct citation of pydantic_validation.md."
}
```

**Response Body**:
```json
{
  "feedback_id": 1,
  "query_id": "993fe01b-c744-4861-bbd4-539091bca72b",
  "rating": "thumbs_up",
  "comment": "Accurate response with correct citation of pydantic_validation.md.",
  "created_at": "2026-10-30T17:55:00.123456",
  "message": "Feedback submitted successfully."
}
```

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
To connect to live Gemini 1.5 Flash and Gemini Embeddings, add your API key to `.env`:
```env
GEMINI_API_KEY="your-gemini-api-key-here"
```
*(If no API key is configured, the system automatically uses `MockChatModel` and `DeterministicTestEmbeddings`, enabling 100% of tests and verification scripts to execute offline in milliseconds).*

---

## Running the Application

### 1. Launch FastAPI Server
```bash
uvicorn src.main:app --reload --port 8000
```
- Interactive OpenAPI / Swagger UI: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- Alternative ReDoc UI: [http://127.0.0.1:8000/redoc](http://127.0.0.1:8000/redoc)

### 2. Sample cURL Commands

**Query the assistant**:
```bash
curl -X POST "http://127.0.0.1:8000/api/v1/query" \
     -H "Content-Type: application/json" \
     -d '{"question": "How does FastAPI handle dependency injection?", "top_k": 3}'
```

**List indexed documents**:
```bash
curl -X GET "http://127.0.0.1:8000/api/v1/documents"
```

**Upload new document**:
```bash
curl -X POST "http://127.0.0.1:8000/api/v1/ingest" \
     -F "files=@data/documents/fastapi_overview.md"
```

**Submit feedback**:
```bash
curl -X POST "http://127.0.0.1:8000/api/v1/feedback" \
     -H "Content-Type: application/json" \
     -d '{"query_id": "test-id-123", "rating": "thumbs_up", "comment": "Great answer!"}'
```

---

## Verification & Execution

### 1. Run LangGraph Self-Corrective Workflow Verifier
```bash
python verify_workflow.py
```

### 2. Run Ingestion and Vector Store Verifiers
```bash
python verify_ingestion.py
python verify_vector_store.py
```

### 3. Run Automated Pytest Suite
```bash
pytest -v
```

Expected output (`45 passed in ~1.3s`):
```text
tests/test_api.py::test_health_check_endpoint PASSED                     [  2%]
tests/test_api.py::test_system_info_endpoint PASSED                      [  4%]
tests/test_api.py::test_ingest_preview_endpoint PASSED                   [  6%]
tests/test_api.py::test_ingest_preview_invalid_overlap PASSED            [  8%]
tests/test_api.py::test_direct_search_and_stats_endpoints PASSED         [ 11%]
tests/test_api.py::test_query_endpoint_success PASSED                    [ 13%]
tests/test_api.py::test_query_endpoint_fallback_for_off_topic PASSED     [ 15%]
tests/test_api.py::test_query_endpoint_empty_question PASSED             [ 17%]
tests/test_api.py::test_upload_file_ingest_endpoint PASSED               [ 20%]
tests/test_api.py::test_upload_file_unsupported_format PASSED            [ 22%]
tests/test_api.py::test_documents_catalog_endpoint PASSED                [ 24%]
tests/test_api.py::test_feedback_endpoint_success PASSED                 [ 26%]
tests/test_api.py::test_feedback_endpoint_invalid_rating PASSED          [ 28%]
tests/test_api.py::test_root_route_aliases PASSED                        [ 31%]
tests/test_embeddings.py::test_deterministic_embeddings_properties PASSED [ 33%]
tests/test_embeddings.py::test_deterministic_batch_embedding PASSED      [ 35%]
tests/test_embeddings.py::test_get_embeddings_model_factory PASSED       [ 37%]
tests/test_feedback_db.py::test_sqlite_feedback_db PASSED                [ 40%]
tests/test_graph_nodes.py::test_query_analysis_node PASSED               [ 42%]
tests/test_graph_nodes.py::test_retrieval_node PASSED                    [ 44%]
tests/test_graph_nodes.py::test_document_grading_node PASSED             [ 46%]
tests/test_graph_nodes.py::test_query_rewrite_node PASSED                [ 48%]
tests/test_graph_nodes.py::test_generation_node_with_context PASSED      [ 51%]
tests/test_graph_nodes.py::test_generation_node_fallback PASSED          [ 53%]
tests/test_loaders.py::test_load_markdown_document PASSED                [ 55%]
tests/test_loaders.py::test_load_text_document PASSED                    [ 57%]
tests/test_loaders.py::test_load_html_document PASSED                    [ 60%]
tests/test_loaders.py::test_load_empty_file PASSED                       [ 62%]
tests/test_loaders.py::test_load_whitespace_only_file PASSED             [ 64%]
tests/test_loaders.py::test_unsupported_file_format PASSED               [ 66%]
tests/test_loaders.py::test_nonexistent_file PASSED                      [ 68%]
tests/test_loaders.py::test_directory_batch_load PASSED                  [ 71%]
tests/test_splitter.py::test_chunking_with_metadata PASSED               [ 73%]
tests/test_splitter.py::test_chunk_size_and_overlap_validation PASSED    [ 75%]
tests/test_splitter.py::test_batch_splitting PASSED                      [ 77%]
tests/test_splitter.py::test_summary_generation PASSED                   [ 80%]
tests/test_vector_store.py::test_chroma_indexing_and_duplicate_prevention PASSED [ 82%]
tests/test_vector_store.py::test_chroma_update_modified_chunk PASSED     [ 84%]
tests/test_vector_store.py::test_chroma_disk_persistence PASSED          [ 86%]
tests/test_vector_store.py::test_chroma_similarity_search_and_metadata PASSED [ 88%]
tests/test_vector_store.py::test_chroma_delete_document PASSED           [ 91%]
tests/test_workflow.py::test_decide_to_generate_or_retry PASSED          [ 93%]
tests/test_workflow.py::test_end_to_end_successful_rag_workflow PASSED   [ 95%]
tests/test_workflow.py::test_end_to_end_irrelevant_query_self_correction PASSED [ 97%]
tests/test_workflow.py::test_workflow_configurable_retry_limit PASSED    [100%]

======================== 45 passed, 1 warning in 1.28s =========================
```

---

## Assignment Requirements Traceability Matrix

| Requirement | Implementation Component | Status |
| :--- | :--- | :--- |
| Corpus of technical docs (3–5 docs) | `data/documents/` (FastAPI, Pydantic, Python Typing, LangGraph) | ✅ Completed |
| Ingestion & Multi-Format Loading | `src/ingestion/loaders.py` (`.md`, `.txt`, `.html`) | ✅ Completed |
| Semantic Text Chunking | `src/ingestion/splitter.py` (`RecursiveCharacterTextSplitter`) | ✅ Completed |
| Vector Store & Embeddings | `src/retrieval/vector_store.py`, `src/embeddings/models.py` | ✅ Completed |
| Persistent Storage & Deduplication | ChromaDB persistence + SHA256 chunk deduplication | ✅ Completed |
| LangGraph StateGraph Workflow | `src/graph/workflow.py`, `src/graph/state.py` | ✅ Completed |
| Query Analysis Node | `src/graph/nodes.py::query_analysis` | ✅ Completed |
| Semantic Retrieval Node | `src/graph/nodes.py::retrieval` | ✅ Completed |
| Document Grading Node (CRAG) | `src/graph/nodes.py::document_grading` | ✅ Completed |
| Query Rewrite Node & Retries | `src/graph/nodes.py::query_rewrite`, `src/graph/edges.py` | ✅ Completed |
| Answer Generation with Citations | `src/graph/nodes.py::generation` | ✅ Completed |
| Fallback without Hallucination | Safe fallback message when context is insufficient | ✅ Completed |
| POST `/query` Endpoint | `src/api/routes.py::post_query` & `src/services/query_service.py` | ✅ Completed |
| POST `/ingest` Endpoint | `src/api/routes.py::post_ingest` & `src/services/ingestion_service.py` | ✅ Completed |
| GET `/documents` Endpoint | `src/api/routes.py::get_documents` & `src/services/document_service.py` | ✅ Completed |
| POST `/feedback` Endpoint | `src/api/routes.py::post_feedback` & `src/database/feedback_db.py` | ✅ Completed |
| Comprehensive Unit & End-to-End Tests | `tests/` (45 automated tests) | ✅ Completed |
