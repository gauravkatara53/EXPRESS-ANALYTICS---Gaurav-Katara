import pytest
from src.api.schemas import LoadedDocument
from src.ingestion.splitter import DocumentSplitter


@pytest.fixture
def sample_document() -> LoadedDocument:
    content = (
        "# LangGraph Overview\n\n"
        "LangGraph is a library for building stateful, multi-actor applications with LLMs.\n\n"
        "## Key Features\n\n"
        "1. Cyclical graphs for loops and agent reasoning.\n"
        "2. Built-in persistence for conversation memory and time-travel debugging.\n"
        "3. Human-in-the-loop workflows for approval before critical actions.\n\n"
        "## Code Example\n\n"
        "```python\n"
        "from langgraph.graph import StateGraph, END\n"
        "builder = StateGraph(dict)\n"
        "builder.add_node('step', lambda s: {'result': 'ok'})\n"
        "graph = builder.compile()\n"
        "```\n\n"
        "This wraps up the foundational architecture of LangGraph workflows."
    )
    return LoadedDocument(
        source_filename="langgraph_overview.md",
        source_path="data/documents/langgraph_overview.md",
        document_title="LangGraph Overview",
        file_type=".md",
        raw_content=content,
        character_count=len(content),
        file_size_bytes=len(content.encode("utf-8")),
    )


def test_chunking_with_metadata(sample_document: LoadedDocument):
    splitter = DocumentSplitter(chunk_size=300, chunk_overlap=50)
    chunks = splitter.split_document(sample_document)

    assert len(chunks) > 1
    for idx, chunk in enumerate(chunks):
        assert chunk.content.strip() != ""
        # Metadata checks
        assert chunk.metadata.source_filename == "langgraph_overview.md"
        assert chunk.metadata.document_title == "LangGraph Overview"
        assert chunk.metadata.file_type == ".md"
        assert chunk.metadata.source_path == "data/documents/langgraph_overview.md"
        assert chunk.metadata.chunk_index == idx
        assert chunk.metadata.total_chunks == len(chunks)
        assert chunk.metadata.character_count == len(chunk.content)


def test_chunk_size_and_overlap_validation():
    # overlap must be strictly less than chunk_size
    with pytest.raises(ValueError):
        DocumentSplitter(chunk_size=200, chunk_overlap=200)

    with pytest.raises(ValueError):
        DocumentSplitter(chunk_size=200, chunk_overlap=250)


def test_batch_splitting(sample_document: LoadedDocument):
    splitter = DocumentSplitter(chunk_size=400, chunk_overlap=50)
    docs = [sample_document, sample_document]
    all_chunks = splitter.split_documents(docs)

    single_doc_chunks = splitter.split_document(sample_document)
    assert len(all_chunks) == len(single_doc_chunks) * 2


def test_summary_generation(sample_document: LoadedDocument):
    splitter = DocumentSplitter(chunk_size=300, chunk_overlap=50)
    chunks = splitter.split_document(sample_document)
    summary = splitter.generate_summary(
        total_files_found=1,
        loaded_documents=[sample_document],
        chunks=chunks,
        skipped_count=0,
        preview_limit=2,
    )

    assert summary.total_files_found == 1
    assert summary.total_files_loaded == 1
    assert summary.total_chunks_created == len(chunks)
    assert summary.average_chunk_length > 0
    assert len(summary.chunks_preview) <= 2


def test_splitting_with_trailing_empty_content():
    content = "First meaningful paragraph.\n\nSecond meaningful paragraph.\n\n   \n\n\n   "
    doc = LoadedDocument(
        source_filename="trailing_ws.md",
        source_path="data/documents/trailing_ws.md",
        document_title="Trailing WS Doc",
        file_type=".md",
        raw_content=content,
        character_count=len(content),
        file_size_bytes=len(content.encode("utf-8")),
    )
    splitter = DocumentSplitter(chunk_size=30, chunk_overlap=5)
    chunks = splitter.split_document(doc)

    assert len(chunks) > 0
    for idx, chunk in enumerate(chunks):
        assert chunk.content.strip() != ""
        assert chunk.metadata.chunk_index == idx
        assert chunk.metadata.total_chunks == len(chunks)

