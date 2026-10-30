import os
from pathlib import Path
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

    # Reserved future settings (Phases 3+)
    LLM_PROVIDER: str = "groq"
    VECTOR_STORE: str = "chroma"


settings = Settings()
