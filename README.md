# Express Analytics | RAG-Based Technical Documentation Assistant

An enterprise-ready Retrieval-Augmented Generation (RAG) system built with **LangGraph StateGraph**, **FastAPI**, **ChromaDB**, **Google Gemini**, and a modern **Streamlit UI**, designed to ingest, index, grade, and answer questions about technical documentation with self-corrective capabilities.

---

## Current Status: All Phases Completed (1 – 6)

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
- **Phase 6: Modern Streamlit Frontend UI** (Completed)
  - **💬 Chat Interface**: ChatGPT-inspired conversational workspace with Markdown rendering, citation chips, fallback warning banners, expandable execution traces, and thumbs up/down feedback.
  - **📁 Document Explorer**: Live catalog of indexed technical documents in ChromaDB with total metrics, format badges, and search filtering.
  - **📤 Document Ingestion**: Multi-file drag-and-drop uploader supporting `.md`, `.txt`, and `.html` with real-time indexing metrics and duplicate protection.
  - **Centralized API Client**: Robust HTTP client with backend health checking, timeouts, and error handling.
  - **52 Automated Unit & Integration Tests (100% Passing)**.

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
├── verify_frontend.py            # CLI tool to verify Phase 6 Streamlit client & API integration
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
├── frontend/                     # Streamlit Frontend Application (Phase 6)
│   ├── app.py                    # Main Streamlit dashboard (Chat, Explorer, Upload)
│   ├── api_client.py             # Centralized HTTP API client for FastAPI
│   └── requirements.txt          # Frontend dependencies (streamlit, requests)
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
└── tests/                        # Automated Pytest Suite (52 tests, 100% passing)
    ├── __init__.py
    ├── test_api.py               # Tests for all FastAPI endpoints (/query, /ingest, /documents, /feedback)
    ├── test_api_client.py        # Tests for frontend API client methods
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
  "search_query": "validate models with Pydantic BaseModel",
  "retrieved_chunks_count": 4,
  "relevant_chunks_count": 4
}
```

---

### 2. `POST /api/v1/ingest` (and `/ingest`)
Uploads Markdown (`.md`), Plain Text (`.txt`), or HTML (`.html`) files to the server, parses them, splits them into semantic chunks, and indexes them in ChromaDB.

**Request**: Multipart form data with one or more files under key `files`.

**Response Body**:
```json
{
  "uploaded_files": ["new_documentation.md"],
  "failed_files": [],
  "total_chunks_indexed": 3,
  "total_chunks_updated": 0,
  "total_chunks_skipped": 0,
  "total_store_chunks": 15
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
  "status": "success",
  "message": "Feedback recorded successfully.",
  "feedback": {
    "id": 1,
    "query_id": "993fe01b-c744-4861-bbd4-539091bca72b",
    "rating": "thumbs_up",
    "comment": "Accurate response with correct citation of pydantic_validation.md.",
    "created_at": "2026-10-30T17:55:00.123456+00:00"
  }
}
```

---

## Phase 6 Streamlit UI Features

The frontend application (`frontend/app.py`) provides:
1. **💬 Conversational Chat Assistant**:
   - ChatGPT-like interface with prompt suggestion chips.
   - Grounded answers with clean Markdown formatting and code blocks.
   - **Source Citations**: Distinctive badges showing exact source files cited (`fastapi_overview.md`, `pydantic_validation.md`, etc.).
   - **Safe Fallback Indicator**: When the question is outside the corpus, triggers safe fallback notification without hallucinating.
   - **Execution Details Expander**: Displays query intent, refined search query, number of retries, and graded relevant chunks.
   - **Answer Feedback**: Integrated 👍 / 👎 buttons with an optional comment dialog persisting directly to SQLite.
2. **📁 Document Explorer**:
   - High-level metric cards: Total Documents, Total Chunks in ChromaDB, Vector Engine.
   - Search filter to query documents by title or filename.
   - Document cards displaying file format badges, chunk counts, character counts, and disk paths.
3. **📤 Document Ingestion**:
   - Drag-and-drop multi-file upload accepting `.md`, `.txt`, and `.html`.
   - Real-time indexing statistics: uploaded count, newly indexed chunks, updated chunks, skipped duplicates.
4. **Backend Health Indicator**:
   - Sidebar badge displaying live connection status to FastAPI (`🟢 Backend Connected`).

---

## Installation & Setup

```bash
# 1. Activate virtual environment
source .venv/bin/activate

# 2. Dependencies installation
pip install -r requirements.txt
pip install -r frontend/requirements.txt

# 3. Environment configuration
cp .env.example .env
```

### Optional: Google Gemini API Key
To connect to live Gemini 1.5 Flash and Gemini Embeddings, add your API key to `.env`:
```env
GEMINI_API_KEY="your-gemini-api-key-here"
```
*(If no API key is configured, the system automatically uses `MockChatModel` and `DeterministicTestEmbeddings`, enabling 100% of tests, UI verifications, and scripts to execute offline in milliseconds).*

---

## How to Run the Applications

### 1. Launch FastAPI Backend
In your first terminal:
```bash
uvicorn src.main:app --reload --port 8000
```
- Interactive OpenAPI / Swagger UI: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

### 2. Launch Streamlit Frontend
In your second terminal:
```bash
streamlit run frontend/app.py --server.port 8501
```
- Streamlit Web Dashboard: [http://localhost:8501](http://localhost:8501)

---

## Verification & Automated Test Suites

### 1. Run Complete Integration Verification Script
```bash
python verify_frontend.py
```
*Automatically validates health check, document listing, technical Q&A, citations, safe fallback, feedback persistence, and file ingestion with cleanup.*

### 2. Run Individual Phase Verifiers
```bash
python verify_workflow.py       # Phase 4 LangGraph workflow verifier
python verify_vector_store.py   # Phase 3 ChromaDB indexing & retrieval verifier
python verify_ingestion.py      # Phase 2 document loader & chunker verifier
```

### 3. Run Automated Pytest Suite
```bash
pytest -v
```

Expected output (`52 passed in ~1.2s`):
```text
tests/test_api.py (14 tests: health, info, preview, search, /query, /ingest, /documents, /feedback)
tests/test_api_client.py (7 tests: health, query, validation error, list docs, upload, feedback)
tests/test_embeddings.py (3 tests: deterministic properties, batching, factory)
tests/test_feedback_db.py (1 test: SQLite schema & feedback operations)
tests/test_graph_nodes.py (6 tests: analysis, retrieval, grading, rewrite, generation)
tests/test_loaders.py (8 tests: markdown, text, html, empty files, traversal, batch)
tests/test_splitter.py (4 tests: recursive chunking, overlap, metadata integrity)
tests/test_vector_store.py (5 tests: deduplication, upsert, persistence, similarity search)
tests/test_workflow.py (4 tests: LangGraph routing, full RAG flow, CRAG self-correction)

======================== 52 passed, 1 warning in 1.12s =========================
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
| Streamlit Chat Interface | `frontend/app.py` (ChatGPT-like UI with citations & fallbacks) | ✅ Completed |
| Streamlit Document Explorer | `frontend/app.py` (Catalog, chunk metrics & search filter) | ✅ Completed |
| Streamlit Document Upload | `frontend/app.py` (Multi-file drag & drop, indexing summary) | ✅ Completed |
| Streamlit Feedback Module | `frontend/app.py` (Thumbs up/down + SQLite persistence) | ✅ Completed |
| Streamlit API Client | `frontend/api_client.py` (Centralized HTTP client with error handling) | ✅ Completed |
| Comprehensive Unit & End-to-End Tests | `tests/` (52 automated tests, 100% passing) | ✅ Completed |
