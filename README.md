# Express Analytics | RAG-Based Technical Documentation Assistant

An enterprise-ready Retrieval-Augmented Generation (RAG) system built with **FastAPI**, **LangChain / LangGraph**, and **Pydantic**, designed to ingest, index, and query technical documentation with self-corrective capabilities.

---

## Current Status: Phase 1 & Phase 2 Implemented

- **Phase 1: Project Setup & Architecture** (Completed)
  - Modular project layout, configuration management with `pydantic-settings` and `.env`.
  - Modern FastAPI application skeleton with healthcheck, CORS, lifecycle hooks, and structured logging.
- **Phase 2: Document Ingestion & Chunking** (Completed)
  - Multi-format ingestion support: **Markdown (`.md`)**, **Plain Text (`.txt`)**, and **HTML (`.html`)**.
  - Hierarchical semantic chunking with `RecursiveCharacterTextSplitter` (chunk size: 800 chars, overlap: 150 chars).
  - Rich metadata tracking (filename, document title, relative path, chunk index, total chunks, character count).
  - Production-grade error handling (empty files, whitespace-only files, unsupported formats, encoding fallbacks).
  - Included 4 sample technical documentation files.
  - Full automated test suite (16 tests, 100% passing) and CLI verification tool.

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
│   │   ├── routes.py             # API endpoints (/health, /info, /api/v1/ingest/preview)
│   │   └── schemas.py            # Pydantic schemas (DocumentChunk, LoadedDocument, etc.)
│   │
│   ├── ingestion/                # Document Ingestion & Processing Layer
│   │   ├── __init__.py
│   │   ├── loaders.py            # Multi-format document parser (MD, TXT, HTML)
│   │   └── splitter.py           # RecursiveCharacterTextSplitter chunking engine
│   │
│   └── utils/                    # Shared Utilities
│       ├── __init__.py
│       └── logger.py             # Centralized structured logger
│
└── tests/                        # Automated Pytest Suite
    ├── __init__.py
    ├── test_api.py               # Tests for FastAPI endpoints
    ├── test_loaders.py           # Tests for document loaders and edge cases
    └── test_splitter.py          # Tests for text splitting and metadata integrity
```

---

## Technical Design & Chunking Strategy

### 1. Document Ingestion (`src/ingestion/loaders.py`)
- **Markdown (`.md`, `.markdown`)**:
  - Automatically extracts the document title from YAML frontmatter (`title: ...`) or the first `# Heading`.
  - Normalizes line breaks and whitespace.
- **HTML (`.html`, `.htm`)**:
  - Uses `BeautifulSoup` to parse HTML DOM.
  - Extracts the page title from `<title>` or the first `<h1>`.
  - Strips non-content and layout tags (`<script>`, `<style>`, `<nav>`, `<footer>`, `<noscript>`, `<svg>`).
  - Preserves structural breaks (`\n\n` between block elements) so code blocks, sections, and paragraphs remain distinct.
- **Plain Text (`.txt`)**:
  - Extracts the first non-empty line as the document title.
- **Edge Case & Fault Tolerance**:
  - Empty files (0 bytes) or whitespace-only files are safely detected and skipped with a warning log.
  - Unsupported file types (e.g. `.pdf`, `.zip`) raise an `UnsupportedFileTypeError` on direct load or are reported as skipped during batch directory scans.
  - Character encoding issues are handled via UTF-8 parsing with a fallback to Latin-1 and error replacement.

### 2. Chunking Engine (`src/ingestion/splitter.py`)
- **Why `RecursiveCharacterTextSplitter`?**
  - Technical documentation has strong structural boundaries: code snippets, function definitions, headings, and lists.
  - `RecursiveCharacterTextSplitter` attempts splits hierarchically: `["\n\n", "\n", " ", ""]`. This keeps paragraphs and code blocks intact before falling back to individual sentences or words.
- **Chunk Size: 800 characters (~150-200 tokens)**
  - *Rationale*: Small enough for high semantic retrieval precision (avoiding "lost in the middle" dilution during vector search), yet large enough to hold a complete explanation, function signature, or code example.
- **Chunk Overlap: 150 characters (~30-40 tokens)**
  - *Rationale*: A 15-20% overlap prevents context loss across chunk boundaries (such as a paragraph introducing a code block that spans across the cut).
- **Metadata Enrichment**:
  Every chunk is tagged with structured metadata:
  ```json
  {
    "source_filename": "pydantic_validation.md",
    "source_path": "data/documents/pydantic_validation.md",
    "document_title": "Pydantic V2 Data Validation and Serialization",
    "file_type": ".md",
    "chunk_index": 0,
    "total_chunks": 3,
    "character_count": 766
  }
  ```

---

## Installation & Setup

### 1. Create and Activate Virtual Environment
```bash
# Create virtual environment
python3 -m venv .venv

# Activate on macOS/Linux
source .venv/bin/activate

# (On Windows: .venv\Scripts\activate)
```

### 2. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Configure Environment Variables
Copy `.env.example` to `.env`:
```bash
cp .env.example .env
```
Default configuration:
```env
APP_NAME="Express Analytics RAG Assistant"
APP_ENV="development"
DEBUG=true
PORT=8000
HOST="0.0.0.0"

DOCUMENTS_DIR="data/documents"
CHUNK_SIZE=800
CHUNK_OVERLAP=150
```

---

## Verification & Execution

### 1. Run Standalone CLI Ingestion Verifier
To verify document loading, text splitting, and metadata enrichment from the command line:
```bash
python verify_ingestion.py
```

**Expected Output:**
```text
======================================================================
 Express Analytics RAG Assistant - Ingestion & Chunking Verifier 
======================================================================
Target Directory : data/documents
Chunk Size       : 800 characters
Chunk Overlap    : 150 characters
----------------------------------------------------------------------

[Step 1/3] Loading documents...
-> Successfully loaded 4 documents:
   - [.md] 'Pydantic V2 Data Validation and Serialization' (pydantic_validation.md, 1,856 chars)
   - [.txt] 'Python Typing and Type Hints Reference Guide' (python_typing_guide.txt, 1,644 chars)
   - [.html] 'LangGraph Core Concepts and StateGraph Architecture' (langgraph_core_concepts.html, 1,739 chars)
   - [.md] 'FastAPI Overview and Core Architecture' (fastapi_overview.md, 2,312 chars)

[Step 2/3] Chunking documents with RecursiveCharacterTextSplitter...
-> Created 12 total chunks.
-> Average chunk length: 656.6 characters.

[Step 3/3] Inspecting Sample Chunk Metadata...
--- Sample Chunk #1 ---
Source Document : Pydantic V2 Data Validation and Serialization
Filename        : pydantic_validation.md
Relative Path   : data/documents/pydantic_validation.md
Format          : .md
Chunk Index     : 1 of 3
Character Count : 766
...
======================================================================
 VERIFICATION SUCCESSFUL: Document Ingestion & Chunking Working Correctly 
======================================================================
```

---

### 2. Run Automated Pytest Suite
Execute all unit and integration tests:
```bash
pytest -v
```
Expected output:
```text
tests/test_api.py::test_health_check_endpoint PASSED                     [  6%]
tests/test_api.py::test_system_info_endpoint PASSED                      [ 12%]
tests/test_api.py::test_ingest_preview_endpoint PASSED                   [ 18%]
tests/test_api.py::test_ingest_preview_invalid_overlap PASSED            [ 25%]
tests/test_loaders.py::test_load_markdown_document PASSED                [ 31%]
tests/test_loaders.py::test_load_text_document PASSED                    [ 37%]
tests/test_loaders.py::test_load_html_document PASSED                    [ 43%]
tests/test_loaders.py::test_load_empty_file PASSED                       [ 50%]
tests/test_loaders.py::test_load_whitespace_only_file PASSED             [ 56%]
tests/test_loaders.py::test_unsupported_file_format PASSED               [ 62%]
tests/test_loaders.py::test_nonexistent_file PASSED                      [ 68%]
tests/test_loaders.py::test_directory_batch_load PASSED                  [ 75%]
tests/test_splitter.py::test_chunking_with_metadata PASSED               [ 81%]
tests/test_splitter.py::test_chunk_size_and_overlap_validation PASSED    [ 87%]
tests/test_splitter.py::test_batch_splitting PASSED                      [ 93%]
tests/test_splitter.py::test_summary_generation PASSED                   [100%]

============================== 16 passed in 0.99s ==============================
```

---

### 3. Start the FastAPI Server
```bash
uvicorn src.main:app --reload --host 0.0.0.0 --port 8000
```
Open interactive documentation in your browser:
- Swagger UI: [http://localhost:8000/docs](http://localhost:8000/docs)
- ReDoc: [http://localhost:8000/redoc](http://localhost:8000/redoc)

#### Example API Requests & Responses

##### 1. Health Check
```bash
curl -X GET http://localhost:8000/health
```
Response (`200 OK`):
```json
{
  "status": "healthy",
  "app_name": "Express Analytics RAG Assistant",
  "environment": "development",
  "debug_mode": true
}
```

##### 2. System Information
```bash
curl -X GET http://localhost:8000/info
```
Response (`200 OK`):
```json
{
  "app_name": "Express Analytics RAG Assistant",
  "environment": "development",
  "documents_directory": "data/documents",
  "chunking_config": {
    "chunk_size": 800,
    "chunk_overlap": 150
  },
  "phase_status": {
    "phase_1_project_setup": "completed",
    "phase_2_ingestion_and_chunking": "completed",
    "phase_3_vector_store_and_embeddings": "pending",
    "phase_4_langgraph_workflow": "pending",
    "phase_5_api_endpoints": "pending"
  }
}
```

##### 3. Ingestion & Chunking Preview
```bash
curl -X GET "http://localhost:8000/api/v1/ingest/preview?chunk_size=800&chunk_overlap=150&preview_limit=2"
```
Response (`200 OK`):
```json
{
  "total_files_found": 4,
  "total_files_loaded": 4,
  "total_files_skipped": 0,
  "total_chunks_created": 12,
  "average_chunk_length": 656.6,
  "documents": [
    {
      "source_filename": "pydantic_validation.md",
      "source_path": "data/documents/pydantic_validation.md",
      "document_title": "Pydantic V2 Data Validation and Serialization",
      "file_type": ".md",
      "raw_content": "# Pydantic V2 Data Validation...",
      "character_count": 1856,
      "file_size_bytes": 1856
    }
  ],
  "chunks_preview": [
    {
      "content": "# Pydantic V2 Data Validation and Serialization\n\nPydantic is the most widely used data validation library...",
      "metadata": {
        "source_filename": "pydantic_validation.md",
        "source_path": "data/documents/pydantic_validation.md",
        "document_title": "Pydantic V2 Data Validation and Serialization",
        "file_type": ".md",
        "chunk_index": 0,
        "total_chunks": 3,
        "character_count": 766
      }
    }
  ]
}
```

---

## Next Steps (Subsequent Phases)
- **Phase 3**: Vector Store & Embeddings (ChromaDB index creation, persistence, and similarity search).
- **Phase 4**: Self-Corrective LangGraph Workflow (Query analysis, retrieval, document grading, answer generation, conditional retry routing).
- **Phase 5**: Production FastAPI Endpoints (`/query`, `/ingest`, `/documents`, `/feedback`) & Error Handling.
