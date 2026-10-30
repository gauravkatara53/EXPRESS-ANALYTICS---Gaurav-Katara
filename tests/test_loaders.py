import pytest
from pathlib import Path

from src.ingestion.loaders import DocumentLoader, UnsupportedFileTypeError


def test_load_markdown_document(tmp_path: Path):
    md_file = tmp_path / "test_doc.md"
    md_file.write_text("# Test Architecture Guide\n\nThis is content for testing markdown ingestion.")

    loader = DocumentLoader()
    doc = loader.load_file(md_file)

    assert doc is not None
    assert doc.document_title == "Test Architecture Guide"
    assert doc.source_filename == "test_doc.md"
    assert doc.file_type == ".md"
    assert "This is content for testing markdown ingestion." in doc.raw_content
    assert doc.character_count > 0


def test_load_text_document(tmp_path: Path):
    txt_file = tmp_path / "test_doc.txt"
    txt_file.write_text("API Reference Manual\n\nThis document details the API endpoints.")

    loader = DocumentLoader()
    doc = loader.load_file(txt_file)

    assert doc is not None
    assert doc.document_title == "API Reference Manual"
    assert doc.source_filename == "test_doc.txt"
    assert doc.file_type == ".txt"
    assert "This document details the API endpoints." in doc.raw_content


def test_load_html_document(tmp_path: Path):
    html_file = tmp_path / "test_page.html"
    html_content = """
    <!DOCTYPE html>
    <html>
    <head><title>HTML Manual Page</title></head>
    <body>
        <script>var x = 10;</script>
        <style>body { color: red; }</style>
        <nav><a href="#">Ignored Navigation</a></nav>
        <h1>Main Technical Content</h1>
        <p>This is the important documentation text that should be retained.</p>
        <footer>Ignored Footer Notice</footer>
    </body>
    </html>
    """
    html_file.write_text(html_content)

    loader = DocumentLoader()
    doc = loader.load_file(html_file)

    assert doc is not None
    assert doc.document_title == "HTML Manual Page"
    assert doc.source_filename == "test_page.html"
    assert doc.file_type == ".html"
    assert "This is the important documentation text that should be retained." in doc.raw_content
    # Non-content tags must be stripped
    assert "var x = 10" not in doc.raw_content
    assert "Ignored Navigation" not in doc.raw_content
    assert "Ignored Footer Notice" not in doc.raw_content


def test_load_empty_file(tmp_path: Path):
    empty_file = tmp_path / "empty.txt"
    empty_file.write_text("")

    loader = DocumentLoader()
    doc = loader.load_file(empty_file)
    assert doc is None


def test_load_whitespace_only_file(tmp_path: Path):
    ws_file = tmp_path / "whitespace.md"
    ws_file.write_text("   \n\n   \t  \n")

    loader = DocumentLoader()
    doc = loader.load_file(ws_file)
    assert doc is None


def test_unsupported_file_format(tmp_path: Path):
    pdf_file = tmp_path / "manual.pdf"
    pdf_file.write_text("fake pdf binary content")

    loader = DocumentLoader()
    with pytest.raises(UnsupportedFileTypeError):
        loader.load_file(pdf_file)


def test_nonexistent_file():
    loader = DocumentLoader()
    with pytest.raises(FileNotFoundError):
        loader.load_file(Path("/non/existent/path/to/file.md"))


def test_directory_batch_load(tmp_path: Path):
    (tmp_path / "doc1.md").write_text("# Doc 1\nContent 1")
    (tmp_path / "doc2.txt").write_text("Doc 2 Title\nContent 2")
    (tmp_path / "doc3.html").write_text("<html><title>Doc 3</title><body>Content 3</body></html>")
    (tmp_path / "ignored.pdf").write_text("pdf content")
    (tmp_path / "empty.md").write_text("")

    loader = DocumentLoader()
    docs, skipped = loader.load_directory(tmp_path)

    assert len(docs) == 3
    assert len(skipped) == 2  # 1 unsupported (.pdf) + 1 empty (.md)
    doc_titles = {d.document_title for d in docs}
    assert "Doc 1" in doc_titles
    assert "Doc 2 Title" in doc_titles
    assert "Doc 3" in doc_titles

