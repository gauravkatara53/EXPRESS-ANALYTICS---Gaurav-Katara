# Express Analytics | RAG-Based Technical Documentation Assistant

An enterprise-ready Retrieval-Augmented Generation (RAG) system built with **LangGraph StateGraph**, **FastAPI**, **ChromaDB**, and **Google Gemini**, designed to ingest, index, grade, and answer questions about technical documentation with self-corrective capabilities.

---

## Current Status: Phase 1, Phase 2, Phase 3, & Phase 4 Implemented

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
  - **Comprehensive Test Suite**: 35 automated tests (100% passing) with zero-cost mocked LLMs and CLI verification tools.

---

## Project Structure

```text
express-analytics-rag/
├── .env                          # Local environment configuration
├── .env.example                  # Template configuration with descriptions
├── .gitignore                    # Git ignore file (cache, venv, vector stores)
├── README.md                     # Comprehensive project guide and documentation
├── requirements.txt              # Project dependencies
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
│   ├── api/                      # Web API Layer
│   │   ├── __init__.py
│   │   ├── routes.py             # API endpoints (health, info, ingest, search, stats)
│   │   └── schemas.py            # Pydantic schemas (DocumentChunk, SearchResult, etc.)
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
│   └── utils/                    # Shared Utilities
│       ├── __init__.py
│       └── logger.py             # Centralized structured logger
│
└── tests/                        # Automated Pytest Suite (35 tests, 100% passing)
    ├── __init__.py
    ├── test_api.py               # Tests for FastAPI endpoints
    ├── test_embeddings.py        # Tests for embedding generation and factory
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

### Shared State Schema (`src/graph/state.py`)
```python
class GraphState(TypedDict):
    question: str                   # Original user question
    search_query: str               # Active search query (analyzed or rewritten)
    query_intent: str               # conceptual, how-to, troubleshooting, api_reference
    documents: List[SearchResult]   # Chunks retrieved from ChromaDB
    relevant_documents: List[SearchResult] # Chunks graded relevant
    generation: str                 # Final generated answer
    citations: List[str]            # List of cited source filenames
    retry_count: int                # Number of reformulation retries so far
    max_retries: int                # Retry limit before fallback (default: 2)
    is_fallback: bool               # True if answered via safe fallback
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

## Verification & Execution

### 1. Run LangGraph Self-Corrective Workflow Verifier
```bash
python verify_workflow.py
```

**Scenario 1: Technical question with relevant context (Pydantic models)**:
- Trace: `query_analysis` $\rightarrow$ `retrieve` $\rightarrow$ `grade_documents` (4/4 relevant) $\rightarrow$ `generate`
- Output: Answer grounded in context citing `[pydantic_validation.md]`.
- Retries: 0. Fallback: False.

**Scenario 2: Off-topic query triggering self-correction and safe fallback ("chocolate chip cookies")**:
- Trace: `query_analysis` $\rightarrow$ `retrieve` $\rightarrow$ `grade_documents` (0/4 relevant) $\rightarrow$ `rewrite_query` (Retry 1/2) $\rightarrow$ `retrieve` $\rightarrow$ `grade_documents` (0/4 relevant) $\rightarrow$ `rewrite_query` (Retry 2/2) $\rightarrow$ `retrieve` $\rightarrow$ `grade_documents` (0/4 relevant) $\rightarrow$ `generate` (Safe Fallback)
- Output: *"I am sorry, but the provided technical documentation does not contain sufficient information to answer your question regarding 'What is the recipe for baking chocolate chip cookies?'."*
- Retries: 2. Fallback: True. Citations: []. Zero hallucination.

---

### 2. Run Automated Pytest Suite
```bash
pytest -v
```

Expected output (`35 passed in 1.20s`):
```text
tests/test_api.py::test_health_check_endpoint PASSED                     [  2%]
tests/test_api.py::test_system_info_endpoint PASSED                      [  5%]
tests/test_api.py::test_ingest_preview_endpoint PASSED                   [  8%]
tests/test_api.py::test_ingest_preview_invalid_overlap PASSED            [ 11%]
tests/test_api.py::test_index_and_search_endpoints PASSED                [ 14%]
tests/test_embeddings.py::test_deterministic_embeddings_properties PASSED [ 17%]
tests/test_embeddings.py::test_deterministic_batch_embedding PASSED      [ 20%]
tests/test_embeddings.py::test_get_embeddings_model_factory PASSED       [ 22%]
tests/test_graph_nodes.py::test_query_analysis_node PASSED               [ 25%]
tests/test_graph_nodes.py::test_retrieval_node PASSED                    [ 28%]
tests/test_graph_nodes.py::test_document_grading_node PASSED             [ 31%]
tests/test_graph_nodes.py::test_query_rewrite_node PASSED                [ 34%]
tests/test_graph_nodes.py::test_generation_node_with_context PASSED      [ 37%]
tests/test_graph_nodes.py::test_generation_node_fallback PASSED          [ 40%]
tests/test_loaders.py::test_load_markdown_document PASSED                [ 42%]
tests/test_loaders.py::test_load_text_document PASSED                    [ 45%]
tests/test_loaders.py::test_load_html_document PASSED                    [ 48%]
tests/test_loaders.py::test_load_empty_file PASSED                       [ 51%]
tests/test_loaders.py::test_load_whitespace_only_file PASSED             [ 54%]
tests/test_loaders.py::test_unsupported_file_format PASSED               [ 57%]
tests/test_loaders.py::test_nonexistent_file PASSED                      [ 60%]
tests/test_loaders.py::test_directory_batch_load PASSED                  [ 62%]
tests/test_splitter.py::test_chunking_with_metadata PASSED               [ 65%]
tests/test_splitter.py::test_chunk_size_and_overlap_validation PASSED    [ 68%]
tests/test_splitter.py::test_batch_splitting PASSED                      [ 71%]
tests/test_splitter.py::test_summary_generation PASSED                   [ 74%]
tests/test_vector_store.py::test_chroma_indexing_and_duplicate_prevention PASSED [ 77%]
tests/test_vector_store.py::test_chroma_update_modified_chunk PASSED     [ 80%]
tests/test_vector_store.py::test_chroma_disk_persistence PASSED          [ 82%]
tests/test_vector_store.py::test_chroma_similarity_search_and_metadata PASSED [ 85%]
tests/test_vector_store.py::test_chroma_delete_document PASSED           [ 88%]
tests/test_workflow.py::test_decide_to_generate_or_retry PASSED          [ 91%]
tests/test_workflow.py::test_end_to_end_successful_rag_workflow PASSED   [ 94%]
tests/test_workflow.py::test_end_to_end_irrelevant_query_self_correction PASSED [ 97%]
tests/test_workflow.py::test_workflow_configurable_retry_limit PASSED    [100%]

======================== 35 passed, 1 warning in 1.20s =========================
```

---

## Next Steps (Phase 5)
- **Phase 5: Production FastAPI Endpoints & UI**:
  - `POST /query`: Submit question $\rightarrow$ runs LangGraph workflow $\rightarrow$ returns answer + citations + trace metadata.
  - `POST /ingest`: Ingest new documents from file uploads or URLs.
  - `GET /documents`: List all indexed documents in corpus.
  - `POST /feedback`: Thumbs up/down + comments recording.
  - Interactive web UI (Streamlit or minimal frontend).
