import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
from bs4 import BeautifulSoup

from src.api.schemas import LoadedDocument
from src.utils.logger import get_logger

logger = get_logger("document_loader")

SUPPORTED_EXTENSIONS: Set[str] = {".md", ".markdown", ".txt", ".html", ".htm"}


class UnsupportedFileTypeError(Exception):
    """Raised when an unsupported file format is encountered."""

    pass


class DocumentLoader:
    """Loads and preprocesses Markdown, Text, and HTML technical documents."""

    def __init__(self, supported_extensions: Optional[Set[str]] = None):
        self.supported_extensions = supported_extensions or SUPPORTED_EXTENSIONS

    def _read_file_safe(self, file_path: Path) -> str:
        """Reads a file attempting UTF-8 first, then falls back with error replacement.

        Args:
            file_path: Path to the target file.

        Returns:
            Decoded string content.
        """
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                return f.read()
        except UnicodeDecodeError:
            logger.warning(
                f"UTF-8 decode failed for {file_path}. Falling back to latin-1 with error replacement."
            )
            with open(file_path, "r", encoding="latin-1", errors="replace") as f:
                return f.read()

    def _extract_markdown_title(self, text: str, fallback: str) -> str:
        """Extracts document title from Markdown headers or YAML frontmatter."""
        # Check for YAML frontmatter title
        yaml_match = re.search(r"^---\s*\n.*?title:\s*[\"']?([^\"'\n]+)[\"']?.*?\n---", text, re.DOTALL)
        if yaml_match:
            return yaml_match.group(1).strip()

        # Check for top-level header (# Title)
        for line in text.splitlines():
            stripped = line.strip()
            if stripped.startswith("# "):
                return stripped[2:].strip()
            elif stripped.startswith("## "):
                return stripped[3:].strip()

        return fallback

    def _extract_text_title(self, text: str, fallback: str) -> str:
        """Extracts document title from the first non-empty line of a text document."""
        for line in text.splitlines():
            cleaned = line.strip()
            if cleaned and not cleaned.startswith("---") and not cleaned.startswith("==="):
                # If first line is reasonably short, use it as title
                if len(cleaned) <= 100:
                    return cleaned
                return cleaned[:97] + "..."
        return fallback

    def _extract_html_content_and_title(self, raw_html: str, fallback: str) -> Tuple[str, str]:
        """Parses HTML, removes scripts/styles/navigation, and extracts body text and title."""
        soup = BeautifulSoup(raw_html, "html.parser")

        # Extract title from <title> or <h1>
        title = fallback
        title_tag = soup.find("title")
        if title_tag and title_tag.get_text(strip=True):
            title = title_tag.get_text(strip=True)
        else:
            h1_tag = soup.find("h1")
            if h1_tag and h1_tag.get_text(strip=True):
                title = h1_tag.get_text(strip=True)

        # Decompose non-content elements
        for element in soup(["script", "style", "nav", "footer", "noscript", "svg"]):
            element.decompose()

        # Extract text while preserving structure between blocks
        text = soup.get_text(separator="\n\n", strip=True)
        # Collapse excessive blank lines
        cleaned_text = re.sub(r"\n{3,}", "\n\n", text)
        return cleaned_text, title

    def load_file(self, file_path: Path) -> Optional[LoadedDocument]:
        """Loads and parses a single technical document file.

        Args:
            file_path: Path to the document.

        Returns:
            LoadedDocument instance, or None if the document is empty or should be skipped.

        Raises:
            FileNotFoundError: If the file does not exist.
            UnsupportedFileTypeError: If the file extension is not supported.
        """
        path = Path(file_path).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"File not found: {path}")

        ext = path.suffix.lower()
        if ext not in self.supported_extensions:
            raise UnsupportedFileTypeError(
                f"Unsupported file format '{ext}'. Supported formats: {sorted(list(self.supported_extensions))}"
            )

        file_size = path.stat().st_size
        if file_size == 0:
            logger.warning(f"Skipping empty file (0 bytes): {path.name}")
            return None

        raw_text = self._read_file_safe(path)
        if not raw_text.strip():
            logger.warning(f"Skipping whitespace-only file: {path.name}")
            return None

        fallback_title = path.stem.replace("_", " ").replace("-", " ").title()

        if ext in {".md", ".markdown"}:
            title = self._extract_markdown_title(raw_text, fallback=fallback_title)
            content = raw_text.strip()
        elif ext in {".html", ".htm"}:
            content, title = self._extract_html_content_and_title(raw_text, fallback=fallback_title)
            if not content.strip():
                logger.warning(f"Skipping HTML file with no extractable text content: {path.name}")
                return None
        else:  # .txt
            title = self._extract_text_title(raw_text, fallback=fallback_title)
            content = raw_text.strip()

        # Compute relative source path for clean citations
        try:
            rel_path = str(path.relative_to(Path.cwd()))
        except ValueError:
            rel_path = path.name

        return LoadedDocument(
            source_filename=path.name,
            source_path=rel_path,
            document_title=title,
            file_type=ext,
            raw_content=content,
            character_count=len(content),
            file_size_bytes=file_size,
        )

    def load_directory(
        self, directory_path: Path, recursive: bool = True
    ) -> Tuple[List[LoadedDocument], List[str]]:
        """Loads all supported documents found in a directory.

        Args:
            directory_path: Directory path containing documentation files.
            recursive: Whether to scan subdirectories.

        Returns:
            Tuple of (loaded_documents_list, skipped_or_failed_files_info).

        Raises:
            FileNotFoundError: If the directory does not exist.
        """
        dir_path = Path(directory_path).resolve()
        if not dir_path.is_dir():
            raise FileNotFoundError(f"Directory not found: {dir_path}")

        loaded_documents: List[LoadedDocument] = []
        skipped_files: List[str] = []

        glob_pattern = "**/*" if recursive else "*"
        for item in dir_path.glob(glob_pattern):
            if not item.is_file():
                continue

            # Ignore hidden files or git directories
            if item.name.startswith(".") or "/." in str(item):
                continue

            ext = item.suffix.lower()
            if ext not in self.supported_extensions:
                skipped_files.append(f"{item.name} (unsupported format '{ext}')")
                continue

            try:
                doc = self.load_file(item)
                if doc is not None:
                    loaded_documents.append(doc)
                    logger.info(
                        f"Loaded document '{doc.document_title}' from {item.name} "
                        f"({doc.character_count:,} chars)"
                    )
                else:
                    skipped_files.append(f"{item.name} (empty or whitespace-only)")
            except Exception as e:
                logger.error(f"Error loading {item.name}: {e}")
                skipped_files.append(f"{item.name} (error: {str(e)})")

        logger.info(
            f"Ingestion scan completed: {len(loaded_documents)} documents loaded, "
            f"{len(skipped_files)} files skipped."
        )
        return loaded_documents, skipped_files
