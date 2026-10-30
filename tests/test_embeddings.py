import math
import pytest
from src.embeddings.models import (
    DeterministicTestEmbeddings,
    get_embeddings_model,
)


def test_deterministic_embeddings_properties():
    embedder = DeterministicTestEmbeddings(dimension=768)

    text1 = "FastAPI uses Starlette and Pydantic for high performance async APIs."
    text2 = "FastAPI uses Starlette and Pydantic for high performance async APIs."
    unrelated = "Quantum entanglement in interstellar particles."

    vec1 = embedder.embed_query(text1)
    vec2 = embedder.embed_query(text2)
    vec_unrelated = embedder.embed_query(unrelated)

    # Dimension check
    assert len(vec1) == 768
    assert len(vec2) == 768
    assert len(vec_unrelated) == 768

    # L2 Normalization check (norm approx 1.0)
    norm1 = math.sqrt(sum(x * x for x in vec1))
    assert pytest.approx(norm1, abs=1e-5) == 1.0

    # Identical texts must produce identical vectors
    assert vec1 == vec2

    # Dot product calculation
    dot_identical = sum(a * b for a, b in zip(vec1, vec2))
    dot_unrelated = sum(a * b for a, b in zip(vec1, vec_unrelated))

    assert pytest.approx(dot_identical, abs=1e-5) == 1.0
    assert dot_unrelated < dot_identical


def test_deterministic_batch_embedding():
    embedder = DeterministicTestEmbeddings(dimension=128)
    texts = ["Document one", "Document two", "Document three"]
    vectors = embedder.embed_documents(texts)

    assert len(vectors) == 3
    for v in vectors:
        assert len(v) == 128
        norm = math.sqrt(sum(x * x for x in v))
        assert pytest.approx(norm, abs=1e-5) == 1.0


def test_get_embeddings_model_factory():
    # Test mock provider
    mock_model = get_embeddings_model(provider="mock")
    assert isinstance(mock_model, DeterministicTestEmbeddings)

    # Test auto fallback when no API key is present
    model = get_embeddings_model(provider="gemini", api_key=None)
    assert model is not None

    with pytest.raises(ValueError):
        get_embeddings_model(provider="invalid_provider")

