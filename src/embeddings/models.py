import hashlib
import math
import os
import re
from typing import List, Optional
from langchain_core.embeddings import Embeddings

from src.config import settings
from src.utils.logger import get_logger

logger = get_logger("embeddings")


class DeterministicTestEmbeddings(Embeddings):
    """Deterministic, normalized dense vector embedding generator for testing and offline environments.

    Produces a fixed-dimension vector (default: 768) using deterministic term-frequency
    and n-gram projection with L2 normalization. Guarantees:
    - Identical texts yield identical vectors (cosine similarity = 1.0).
    - Texts sharing semantic keywords have significantly higher similarity than unrelated texts.
    - Zero network dependencies, zero rate limits, and millisecond execution.
    """

    def __init__(self, dimension: int = 768):
        self.dimension = dimension

    def _text_to_vector(self, text: str) -> List[float]:
        tokens = re.findall(r"\b\w+\b", text.lower())
        if not tokens:
            return [0.0] * self.dimension

        vector = [0.0] * self.dimension

        # Project unigrams and bigrams deterministically
        for i, token in enumerate(tokens):
            # Token hash
            h = int(hashlib.md5(token.encode("utf-8")).hexdigest(), 16)
            idx = h % self.dimension
            sign = 1.0 if ((h >> 4) & 1) else -1.0
            vector[idx] += sign

            # Bigram hash for phrase awareness
            if i > 0:
                bigram = f"{tokens[i-1]}_{token}"
                bh = int(hashlib.sha256(bigram.encode("utf-8")).hexdigest(), 16)
                bidx = bh % self.dimension
                bsign = 1.0 if ((bh >> 4) & 1) else -1.0
                vector[bidx] += bsign * 1.5

        # L2 normalization so cosine similarity equals dot product
        norm = math.sqrt(sum(x * x for x in vector))
        if norm > 0.0:
            vector = [x / norm for x in vector]
        return vector

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embeds a batch of document texts."""
        return [self._text_to_vector(t) for t in texts]

    def embed_query(self, text: str) -> List[float]:
        """Embeds a single query string."""
        return self._text_to_vector(text)


class GoogleGeminiEmbeddings(Embeddings):
    """Wrapper around LangChain's GoogleGenerativeAIEmbeddings for Google Gemini models."""

    def __init__(
        self,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        task_type: Optional[str] = None,
    ):
        from langchain_google_genai import GoogleGenerativeAIEmbeddings

        self.model = model or settings.GEMINI_EMBEDDING_MODEL
        self.api_key = (
            api_key
            or settings.GEMINI_API_KEY
            or settings.GOOGLE_API_KEY
            or os.getenv("GEMINI_API_KEY")
            or os.getenv("GOOGLE_API_KEY")
        )

        if not self.api_key:
            raise ValueError(
                "Google Gemini API Key is required for GoogleGeminiEmbeddings. "
                "Set GEMINI_API_KEY or GOOGLE_API_KEY in .env or environment."
            )

        kwargs = {"model": self.model, "google_api_key": self.api_key}
        if task_type:
            kwargs["task_type"] = task_type

        self._client = GoogleGenerativeAIEmbeddings(**kwargs)
        logger.info(f"Initialized Google Gemini Embeddings with model '{self.model}'.")

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embeds a batch of texts using Google Gemini API."""
        return self._client.embed_documents(texts)

    def embed_query(self, text: str) -> List[float]:
        """Embeds a search query using Google Gemini API."""
        return self._client.embed_query(text)


def get_embeddings_model(
    provider: Optional[str] = None,
    model_name: Optional[str] = None,
    api_key: Optional[str] = None,
) -> Embeddings:
    """Factory to instantiate the appropriate Embeddings provider.

    Args:
        provider: Provider identifier ("gemini", "mock", or None for auto-detection).
        model_name: Override for the embedding model name.
        api_key: Optional explicit API key.

    Returns:
        Instance conforming to LangChain's Embeddings interface.
    """
    selected_provider = (provider or settings.EMBEDDING_PROVIDER or "gemini").lower()
    active_key = (
        api_key
        or settings.GEMINI_API_KEY
        or settings.GOOGLE_API_KEY
        or os.getenv("GEMINI_API_KEY")
        or os.getenv("GOOGLE_API_KEY")
    )

    if selected_provider == "gemini":
        if active_key:
            try:
                return GoogleGeminiEmbeddings(
                    model=model_name or settings.GEMINI_EMBEDDING_MODEL,
                    api_key=active_key,
                )
            except Exception as e:
                logger.warning(
                    f"Failed to initialize live Google Gemini Embeddings: {e}. "
                    "Falling back to DeterministicTestEmbeddings."
                )
                return DeterministicTestEmbeddings(dimension=768)
        else:
            logger.info(
                "No GEMINI_API_KEY / GOOGLE_API_KEY detected in environment. "
                "Using DeterministicTestEmbeddings for local/test operation. "
                "(Set GEMINI_API_KEY in .env to connect to live Google Gemini text-embedding-004)"
            )
            return DeterministicTestEmbeddings(dimension=768)

    elif selected_provider in {"mock", "test", "deterministic"}:
        logger.info("Using DeterministicTestEmbeddings provider.")
        return DeterministicTestEmbeddings(dimension=768)

    else:
        raise ValueError(
            f"Unsupported embedding provider '{selected_provider}'. "
            "Supported providers: 'gemini', 'mock'."
        )
