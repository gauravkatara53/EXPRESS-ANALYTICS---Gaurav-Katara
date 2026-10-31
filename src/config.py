import os
from pathlib import Path
from typing import Optional
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application configuration loaded from environment variables and .env file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Application settings
    APP_NAME: str = "Express Analytics RAG Assistant"
    APP_ENV: str = "development"
    DEBUG: bool = True
    PORT: int = 8000
    HOST: str = "0.0.0.0"

    # Corpus & chunking defaults
    DOCUMENTS_DIR: Path = Field(
        default_factory=lambda: Path(os.getenv("DOCUMENTS_DIR", "data/documents"))
    )
    CHUNK_SIZE: int = 800
    CHUNK_OVERLAP: int = 150

    # ChromaDB Vector Store settings
    CHROMA_PERSIST_DIRECTORY: Path = Field(
        default_factory=lambda: Path(os.getenv("CHROMA_PERSIST_DIRECTORY", "chroma_db"))
    )
    CHROMA_COLLECTION_NAME: str = "technical_documentation"

    # SQLite Feedback Database settings
    SQLITE_DB_PATH: Path = Field(
        default_factory=lambda: Path(os.getenv("SQLITE_DB_PATH", "feedback.db"))
    )

    # Embedding Model settings
    EMBEDDING_PROVIDER: str = "gemini"  # "gemini" or "mock"
    GEMINI_EMBEDDING_MODEL: str = "models/text-embedding-004"
    GEMINI_API_KEY: Optional[str] = None
    GOOGLE_API_KEY: Optional[str] = None

    # LLM & Generation settings
    LLM_PROVIDER: str = "groq"  # "groq", "gemini", "openai", or "mock"
    GEMINI_LLM_MODEL: str = "gemini-1.5-flash"
    GROQ_API_KEY: Optional[str] = None
    GROQ_MODEL: str = "qwen/qwen3.8-27b"
    OPENAI_API_KEY: Optional[str] = None
    OPENAI_LLM_MODEL: str = "gpt-4o-mini"
    MAX_RETRIES: int = 2
    TEMPERATURE: float = 0.0

    # Retrieval settings
    DEFAULT_TOP_K: int = 4
    SIMILARITY_SCORE_THRESHOLD: Optional[float] = 0.0


settings = Settings()
