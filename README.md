# Technical Documentation Assistant (Self-Corrective RAG)

A robust Retrieval-Augmented Generation (RAG) system built with **FastAPI**, **LangGraph**, **ChromaDB**, and **Streamlit**. Designed to ingest technical documentation (Markdown, Text, HTML, URLs), index it in a persistent vector database, grade retrieved chunks for relevance, self-correct query phrasing when retrieval fails, and generate accurate, cited answers with zero-hallucination fallbacks.

---

## Table of Contents

- [Overview](#overview)
- [Key Capabilities](#key-capabilities)
- [System Architecture](#system-architecture)
- [LangGraph Workflow](#langgraph-workflow)
- [Tech Stack](#tech-stack)
- [Quickstart Guide](#quickstart-guide)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Configuration](#configuration)
  - [Running the Services](#running-the-services)
- [API Reference & Examples](#api-reference--examples)
  - [1. Query Assistant (`POST /api/v1/query`)](#1-query-assistant-post-apiv1query)
  - [2. Ingest Files (`POST /api/v1/ingest`)](#2-ingest-files-post-apiv1ingest)
  - [3. Ingest URLs (`POST /api/v1/ingest/url`)](#3-ingest-urls-post-apiv1ingesturl)
  - [4. List Documents (`GET /api/v1/documents`)](#4-list-documents-get-apiv1documents)
  - [5. Submit Feedback (`POST /api/v1/feedback`)](#5-submit-feedback-post-apiv1feedback)
- [Frontend Interface](#frontend-interface)
- [Engineering Decisions & Design Trade-offs](#engineering-decisions--design-trade-offs)
- [Project Structure](#project-structure)
- [Testing & Verification](#testing--verification)

---

## Overview

Standard naive RAG systems pull the top-$k$ nearest chunks from a vector store and blindly feed them into an LLM prompt. In technical domains, this leads to two frequent failure modes:

1. **Retrieval Noise**: Unrelated paragraphs with surface keyword overlap dilute the prompt context, causing the LLM to generate generic or partially incorrect technical advice.
2. **Hallucination on Out-of-Scope Queries**: If the documentation does not contain the answer, naive pipelines fabricate plausible-looking syntax, invalid flags, or non-existent APIs.

This project implements **Self-Corrective RAG (CRAG)** to solve these failure modes:

- Incoming queries are analyzed to classify intent and expand keywords.
- Retrieved chunks are evaluated by a dedicated relevance grader before generation.
- If chunks are irrelevant, a conditional edge loops back to reformulate the query with technical synonyms and re-retrieve.
- If repeated retrieval attempts fail to surface relevant material, the pipeline safely refuses to answer rather than hallucinating.
- Every generated answer explicitly cites the source filenames it drew from.

---

## Key Capabilities

- **Cyclic LangGraph Workflow**: StateGraph architecture coordinating query analysis, retrieval, document grading, query rewriting, and grounded generation with bounded retries.
- **Multi-Format Ingestion**: Supports Markdown (`.md`), plain text (`.txt`), and HTML (`.html`) files with automated script/navigation stripping and title extraction.
- **URL Documentation Scraping**: Direct ingestion of online technical guides and API documentation via URL endpoints.
- **Deduplicated Vector Storage**: Persistent ChromaDB store using SHA-256 chunk content hashing to prevent duplicate embeddings on repeated indexing runs.
- **Dual-Model Support**: Connects to live cloud models (Google Gemini or Groq LPUs) with an automated fallback to offline deterministic models for local runs and CI test suites without API keys.
- **Relational Feedback Loop**: SQLite database tracking user feedback (`thumbs_up` / `thumbs_down`) and comments mapped to unique query IDs.
- **Interactive UI**: Multi-view Streamlit dashboard with a chat interface, execution trace inspection, document catalog explorer, and file uploader.
- **Comprehensive Test Suite**: 55 unit and integration tests passing in Pytest.

---

## System Architecture

```text
                           ┌─────────────────────────────┐
                           │      Streamlit Frontend     │
                           │      (frontend/app.py)      │
                           └──────────────┬──────────────┘
                                          │ HTTP / REST
                                          ▼
                           ┌─────────────────────────────┐
                           │       FastAPI Backend       │
                           │     (src/api/routes.py)     │
                           └──────────────┬──────────────┘
                                          │
          ┌───────────────────────────────┼───────────────────────────────┐
          │                               │                               │
          ▼                               ▼                               ▼
┌───────────────────┐           ┌───────────────────┐           ┌───────────────────┐
│   QueryService    │           │ IngestionService  │           │  FeedbackService  │
│(src/services/...) │           │(src/services/...) │           │  (SQLite Database)│
└─────────┬─────────┘           └─────────┬─────────┘           └─────────┬─────────┘
          │                               │                               │
          ▼                               ▼                               │
┌───────────────────┐           ┌───────────────────┐                     │
│ LangGraph Engine  │           │ Loaders & Splitter│                     │
│  (StateGraph)     │           │(src/ingestion/...)│                     │
└─────────┬─────────┘           └─────────┬─────────┘                     │
          │                               │                               │
          └───────────────────────┬───────┴───────────────────────────────┘
                                  │
                                  ▼
                  ┌───────────────────────────────┐
                  │       ChromaVectorStore       │
                  │   - Persistent local store    │
                  │   - Cosine similarity metric  │
                  │   - SHA-256 chunk dedup       │
                  └───────────────────────────────┘
```

---

## LangGraph Workflow

The core reasoning loop is modeled as a compiled LangGraph `StateGraph`:

```text
                   START
                     │
                     ▼
           [ 1. query_analysis ]
           - Classify intent (conceptual, how-to, troubleshooting, api_reference)
           - Expand ambiguous phrasing into a keyword-rich retrieval query
                     │
                     ▼
     ┌────> [ 2. retrieve ]
     │      - Embed query and retrieve top-k chunks from ChromaDB
     │               │
     │               ▼
     │      [ 3. grade_documents ]
     │      - Grader evaluates candidate chunks for relevance to the question
     │      - Filters out irrelevant noise
     │               │
     │               ▼
     │       < Conditional Edge >
     │        (decide_to_generate_or_retry)
     │          │
     │          ├── [Relevant Chunks >= 1] OR [Retries >= MAX_RETRIES]
     │          │     │
     │          │     ▼
     │          │  [ 4. generate ]
     │          │  - Synthesize answer strictly from verified context
     │          │  - Attach source citations ([filename.md])
     │          │  - Trigger safe fallback message if 0 relevant chunks
     │          │     │
     │          │     ▼
     │          │    END
     │          │
     │          └── [0 Relevant Chunks] AND [Retries < MAX_RETRIES]
     │                │
     │                ▼
     └───── [ 5. rewrite_query ]
            - Reformulate query with technical synonyms
            - Increment retry counter
            - Route back to [retrieve]
```

---

## Tech Stack

| Layer                   | Technology                                                                               | Purpose                                                 |
| :---------------------- | :--------------------------------------------------------------------------------------- | :------------------------------------------------------ |
| **Backend API**         | FastAPI, Uvicorn                                                                         | Async REST endpoints, Pydantic validation, CORS         |
| **Orchestration**       | LangGraph, LangChain Core                                                                | Cyclical StateGraph with typed `GraphState`             |
| **Vector Database**     | ChromaDB                                                                                 | Local persistent vector storage with cosine distance    |
| **LLMs Supported**      | Google Gemini (`gemini-1.5-flash`), Groq (`qwen/qwen3.8-27b`, `llama-3.3-70b-versatile`) | Query analysis, document grading, answer generation     |
| **Embeddings**          | Google Gemini (`models/text-embedding-004`)                                              | 768-dimensional dense vector embeddings                 |
| **Feedback Store**      | SQLite 3 (`feedback.db`)                                                                 | Relational persistence for user thumbs up/down ratings  |
| **Frontend UI**         | Streamlit                                                                                | Chat interface, document explorer, file uploader        |
| **Document Processing** | BeautifulSoup4, RecursiveCharacterTextSplitter                                           | HTML cleanup, title extraction, hierarchical chunking   |
| **Testing**             | Pytest, TestClient                                                                       | Automated test suite with 55 unit and integration tests |

---

## Quickstart Guide

### Prerequisites

- Python 3.9, 3.10, or 3.11 installed.
- Git.

### Installation

1. **Clone the repository:**

   ```bash
   git clone <repository-url>
   cd "untitled folder"
   ```

2. **Create and activate a virtual environment:**

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate       # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   pip install -r frontend/requirements.txt
   ```

### Configuration

Copy the example environment file:

```bash
cp .env.example .env
```

Open `.env` and configure your API key. You can use either Groq (recommended for ultra-low latency) or Google Gemini:

```env
# Application Settings
APP_NAME="Express Analytics RAG Assistant"
APP_ENV="development"
PORT=8000
HOST="0.0.0.0"

# Document & Storage Paths
DOCUMENTS_DIR="data/documents"
CHUNK_SIZE=800
CHUNK_OVERLAP=150
CHROMA_PERSIST_DIRECTORY="chroma_db"
CHROMA_COLLECTION_NAME="technical_documentation"
SQLITE_DB_PATH="feedback.db"

# LLM Provider Configuration
LLM_PROVIDER="groq"                  # "groq", "gemini", or "mock"
GROQ_API_KEY="gsk_..."               # Get from https://console.groq.com
GROQ_MODEL="qwen/qwen3.8-27b"

# Google Gemini Settings (optional if using Groq)
GEMINI_API_KEY=""                    # Get from https://aistudio.google.com
GEMINI_LLM_MODEL="gemini-1.5-flash"
GEMINI_EMBEDDING_MODEL="models/text-embedding-004"

# Retrieval & Workflow Parameters
MAX_RETRIES=2
DEFAULT_TOP_K=4
SIMILARITY_SCORE_THRESHOLD=0.0
```

> **Offline Mode**: If no API keys are provided in `.env`, the system automatically falls back to internal deterministic embeddings and mock chat models. The entire API, frontend, and test suite run without internet connectivity.

### Running the Services

#### 1. Start the FastAPI Backend

```bash
uvicorn src.main:app --reload --port 8000
```

- Swagger API Docs: [http://localhost:8000/docs](http://localhost:8000/docs)
- ReDoc Docs: [http://localhost:8000/redoc](http://localhost:8000/redoc)
- Health Endpoint: [http://localhost:8000/health](http://localhost:8000/health)

#### 2. Start the Streamlit Frontend

In a separate terminal window:

```bash
source .venv/bin/activate
streamlit run frontend/app.py --server.port 8501
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## API Reference & Examples

### 1. Query Assistant (`POST /api/v1/query`)

Submits a technical question to the LangGraph pipeline for retrieval, grading, self-correction, and generation.

**Request:**

```bash
curl -X POST http://localhost:8000/api/v1/query \
  -H "Content-Type: application/json" \
  -d '{
    "question": "How does FastAPI handle dependency injection?",
    "max_retries": 2
  }'
```

**Response (`200 OK`):**

```json
{
  "query_id": "c3e57b98-1d22-48a0-9c10-388cf625801e",
  "question": "How does FastAPI handle dependency injection?",
  "answer": "FastAPI provides a built-in Dependency Injection system using the `Depends` function. Dependencies can be declared in path operations to manage database sessions, authentication, and shared logic...",
  "citations": ["fastapi_overview.md"],
  "is_fallback": false,
  "retry_count": 0,
  "query_intent": "conceptual",
  "search_query": "FastAPI dependency injection mechanism Depends function",
  "retrieved_chunks_count": 4,
  "relevant_chunks_count": 1
}
```

---

### 2. Ingest Files (`POST /api/v1/ingest`)

Uploads Markdown (`.md`), text (`.txt`), or HTML (`.html`) files. Documents are parsed, split into chunks, deduplicated via SHA-256 hashes, and stored in ChromaDB.

**Request:**

```bash
curl -X POST http://localhost:8000/api/v1/ingest \
  -F "files=@data/documents/fastapi_overview.md"
```

**Response (`200 OK`):**

```json
{
  "uploaded_files": ["fastapi_overview.md"],
  "total_chunks_indexed": 0,
  "total_chunks_updated": 0,
  "total_chunks_skipped": 3,
  "total_store_chunks": 12,
  "failed_files": []
}
```

---

### 3. Ingest URLs (`POST /api/v1/ingest/url`)

Fetches documentation directly from HTTP/HTTPS URLs, sanitizes HTML, extracts titles, chunks the content, and indexes it into ChromaDB.

**Request:**

```bash
curl -X POST http://localhost:8000/api/v1/ingest/url \
  -H "Content-Type: application/json" \
  -d '{
    "urls": [
      "https://fastapi.tiangolo.com/tutorial"
    ]
  }'
```

---

### 4. List Documents (`GET /api/v1/documents`)

Returns a catalog of all documents currently indexed in ChromaDB with metadata and chunk counts.

**Request:**

```bash
curl -X GET http://localhost:8000/api/v1/documents
```

**Response (`200 OK`):**

```json
{
  "total_documents": 4,
  "total_chunks": 12,
  "documents": [
    {
      "source_filename": "fastapi_overview.md",
      "document_title": "FastAPI Overview and Core Architecture",
      "file_type": ".md",
      "source_path": "data/documents/fastapi_overview.md",
      "chunk_count": 3,
      "total_characters": 2314
    },
    {
      "source_filename": "langgraph_core_concepts.html",
      "document_title": "LangGraph Core Concepts and State Management",
      "file_type": ".html",
      "source_path": "data/documents/langgraph_core_concepts.html",
      "chunk_count": 3,
      "total_characters": 1942
    },
    {
      "source_filename": "pydantic_validation.md",
      "document_title": "Pydantic V2 Data Validation and Serialization",
      "file_type": ".md",
      "source_path": "data/documents/pydantic_validation.md",
      "chunk_count": 3,
      "total_characters": 1858
    },
    {
      "source_filename": "python_typing_guide.txt",
      "document_title": "Python Typing Guide - Type Annotations and Generics",
      "file_type": ".txt",
      "source_path": "data/documents/python_typing_guide.txt",
      "chunk_count": 3,
      "total_characters": 2185
    }
  ]
}
```

---

### 5. Submit Feedback (`POST /api/v1/feedback`)

Persists user ratings and optional qualitative comments in SQLite.

**Request:**

```bash
curl -X POST http://localhost:8000/api/v1/feedback \
  -H "Content-Type: application/json" \
  -d '{
    "query_id": "c3e57b98-1d22-48a0-9c10-388cf625801e",
    "rating": "thumbs_up",
    "comment": "Accurate response with exact file citation."
  }'
```

**Response (`201 Created`):**

```json
{
  "status": "success",
  "message": "Feedback recorded successfully for query 'c3e57b98-1d22-48a0-9c10-388cf625801e'.",
  "feedback": {
    "id": 1,
    "query_id": "c3e57b98-1d22-48a0-9c10-388cf625801e",
    "rating": "thumbs_up",
    "comment": "Accurate response with exact file citation.",
    "created_at": "2026-10-31T06:10:28.839119+00:00"
  }
}
```

---

## Frontend Interface

The Streamlit interface (`frontend/app.py`) provides three primary workspaces:

1. **Chat Assistant**:
   - Conversational chat input with live LangGraph status notifications.
   - Grounded responses with clickable citation pills.
   - Non-hallucination warning banner when an out-of-scope question triggers the fallback.
   - Expandable workflow trace showing Query ID, classified intent, rewritten query text, retry count, and relevance counts.
   - Inline feedback buttons (👍 Helpful / 👎 Inaccurate) with optional comment submission.

2. **Document Explorer**:
   - Live metrics: Total indexed documents, total chunks, and vector store engine status.
   - Document cards displaying titles, formats (`.md`, `.txt`, `.html`), disk paths, and character counts.
   - Instant search filter for finding indexed files by name or extension.

3. **Upload Documents**:
   - Multi-file drag-and-drop uploader supporting Markdown, text, and HTML.
   - Real-time indexing summary indicating newly indexed chunks, updated chunks, and skipped duplicates.

---

## Engineering Decisions & Design Trade-offs

### 1. Why LangGraph for Self-Corrective RAG?

Standard chain-based pipelines (e.g., standard LangChain `RetrievalQA`) are linear DAGs. They cannot cycle backwards to reformulate a query when retrieval yields irrelevant results. LangGraph allows explicit state management (`GraphState`) and conditional edges, enabling cyclical loops where the system can inspect intermediate grading outputs, decide whether to rewrite and re-retrieve, and cleanly break out when retry bounds are reached.

### 2. Chunking Strategy (800 Chars / 150 Chars Overlap)

- **800 characters (~150-200 tokens)**: Technical documentation contains signatures, code snippets, and explanations. Chunks under 300 characters frequently sever function arguments or class definitions. Chunks over 1,500 characters dilute vector similarity by blending multiple topics together.
- **150 characters overlap**: Guarantees boundary continuity so that introductory explanations and adjacent code blocks remain linked across split boundaries.
- **Recursive Character Splitting**: Splitting on `["\n\n", "\n", " ", ""]` keeps Markdown paragraphs, headings, and code blocks intact whenever possible.

### 3. Duplicate Prevention & Deterministic Chunk IDs

Instead of generating random UUIDs on every upload, chunk IDs follow a deterministic scheme:

```python
chunk_id = f"{source_filename}#chunk_{chunk_index}"
```

Before vectorizing, the store computes a SHA-256 fingerprint of the chunk content. If a chunk with the same ID and hash exists in ChromaDB, embedding generation is skipped. If the hash differs, `collection.upsert` updates the vector and metadata without orphan records.

### 4. Single-Pass Batch Grading vs. Sequential Calls

Grading $k=4$ chunks with sequential LLM calls introduces $4 \times \text{latency}$, easily adding 3-5 seconds to every request. We implemented a batch grading node that sends all candidate chunks in a single structured prompt:

- If the batch call succeeds, all chunks are graded in one LLM roundtrip.
- If JSON parsing fails, the node automatically falls back to per-chunk grading.
- If both fail, it safely retains all chunks rather than crashing the pipeline.

### 5. Grounding & Zero-Hallucination Fallback

The generation node is governed by strict system prompt constraints:

- Must rely solely on the provided context.
- Must cite filenames using bracket notation (`[fastapi_overview.md]`).
- If no chunks pass the grader after the maximum retries, the pipeline returns a clean fallback:
  > _"I am sorry, but the provided technical documentation does not contain sufficient information to answer your question regarding ...."_

---

## Project Structure

```text
├── data/
│   └── documents/              # Canonical technical documentation files
│       ├── fastapi_overview.md
│       ├── langgraph_core_concepts.html
│       ├── pydantic_validation.md
│       └── python_typing_guide.txt
├── frontend/
│   ├── api_client.py           # HTTP client connecting Streamlit to FastAPI
│   ├── app.py                  # Streamlit multi-view UI
│   └── requirements.txt        # Frontend-specific dependencies
├── src/
│   ├── api/
│   │   ├── routes.py           # FastAPI endpoints (/query, /ingest, /documents, /feedback)
│   │   └── schemas.py          # Pydantic V2 request & response schemas
│   ├── database/
│   │   └── feedback_db.py      # SQLite connection & CRUD operations
│   ├── embeddings/
│   │   └── models.py           # Gemini embeddings & offline deterministic fallback
│   ├── graph/
│   │   ├── edges.py            # Conditional routing logic
│   │   ├── nodes.py            # Workflow nodes (analysis, retrieval, grading, rewrite, generation)
│   │   ├── state.py            # TypedDict GraphState definition
│   │   └── workflow.py         # StateGraph builder and compilation
│   ├── ingestion/
│   │   ├── loaders.py          # Markdown, text, HTML document loaders
│   │   └── splitter.py         # Recursive character text chunker
│   ├── llm/
│   │   └── models.py           # LLM factory (Groq, Gemini, deterministic mock)
│   ├── retrieval/
│   │   └── vector_store.py     # ChromaDB persistence, deduplication, and search
│   ├── services/
│   │   ├── document_service.py # Catalog aggregation service
│   │   ├── feedback_service.py # Feedback business logic
│   │   ├── ingestion_service.py# File and URL upload service
│   │   └── query_service.py    # Query pipeline execution service
│   ├── utils/
│   │   └── logger.py           # Structured logging utility
│   ├── config.py               # Pydantic Settings configuration
│   └── main.py                 # FastAPI application entrypoint
├── tests/
│   ├── test_api.py             # Endpoint tests
│   ├── test_api_client.py      # Frontend API client tests
│   ├── test_embeddings.py      # Embedding dimensionality and math tests
│   ├── test_feedback_db.py     # SQLite persistence tests
│   ├── test_graph_nodes.py     # Individual LangGraph node tests
│   ├── test_loaders.py         # Document parsing and sanitization tests
│   ├── test_splitter.py        # Chunking and overlap tests
│   ├── test_vector_store.py    # ChromaDB deduplication and search tests
│   └── test_workflow.py        # End-to-end LangGraph workflow tests
├── scripts/
│   ├── verify_frontend.py          # Automated integration verification script
│   ├── verify_ingestion.py         # CLI ingestion verification script
│   ├── verify_vector_store.py      # CLI vector store verification script
│   └── verify_workflow.py          # CLI LangGraph workflow verification script
├── .env.example                # Environment variable template
├── pyproject.toml              # Project metadata & build tool configuration
├── requirements.txt            # Core backend dependencies
└── README.md                   # Project documentation
```

---

## Testing & Verification

### Running the Test Suite

Execute the complete test suite with verbose output:

```bash
pytest -v
```

**Results:**

- **55 passed tests** across 9 test modules with zero failures.
- Tests cover API routes, schema validation, document parsing, chunking logic, vector store deduplication, LangGraph nodes, conditional routing, and SQLite feedback operations.

### Running Automated Verification Scripts

Stand-alone scripts are provided to verify individual components in isolation:

```bash
# 1. End-to-end integration test (spawns in-process API server and verifies all 8 client interactions)
python scripts/verify_frontend.py

# 2. Verify LangGraph workflow execution, query rewriting, and fallback
python scripts/verify_workflow.py

# 3. Verify ChromaDB indexing, deduplication, and similarity search
python scripts/verify_vector_store.py

# 4. Verify document loading, HTML cleanup, and chunking
python scripts/verify_ingestion.py
```
