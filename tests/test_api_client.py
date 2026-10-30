from unittest.mock import MagicMock, patch
import requests
from frontend.api_client import RAGApiClient


def test_api_client_health_success():
    mock_session = MagicMock()
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {"status": "healthy", "app_name": "Test App"}
    mock_session.get.return_value = mock_resp

    client = RAGApiClient(base_url="http://testserver", session=mock_session)
    res = client.check_health()
    assert res["healthy"] is True
    assert res["data"]["status"] == "healthy"


def test_api_client_health_connection_error():
    mock_session = MagicMock()
    mock_session.get.side_effect = requests.exceptions.ConnectionError("Offline")

    client = RAGApiClient(base_url="http://testserver", session=mock_session)
    res = client.check_health()
    assert res["healthy"] is False
    assert "Cannot connect" in res["error"]


def test_api_client_query_success():
    mock_session = MagicMock()
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "query_id": "test-123",
        "question": "How to validate?",
        "answer": "Use Pydantic.",
        "citations": ["pydantic_validation.md"],
        "is_fallback": False,
        "retry_count": 0,
        "query_intent": "how-to",
        "search_query": "validate",
        "retrieved_chunks_count": 4,
        "relevant_chunks_count": 4,
    }
    mock_session.post.return_value = mock_resp

    client = RAGApiClient(base_url="http://testserver", session=mock_session)
    res = client.query_assistant("How to validate?", max_retries=2)
    assert res["success"] is True
    assert res["data"]["query_id"] == "test-123"
    assert res["data"]["citations"] == ["pydantic_validation.md"]


def test_api_client_query_validation_error():
    mock_session = MagicMock()
    mock_resp = MagicMock()
    mock_resp.status_code = 422
    mock_resp.json.return_value = {"detail": "Question cannot be empty"}
    mock_session.post.return_value = mock_resp

    client = RAGApiClient(base_url="http://testserver", session=mock_session)
    res = client.query_assistant("   ")
    assert res["success"] is False
    assert "Invalid query" in res["error"]


def test_api_client_list_documents():
    mock_session = MagicMock()
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "total_documents": 2,
        "total_chunks": 6,
        "documents": [
            {
                "source_filename": "fastapi_overview.md",
                "document_title": "FastAPI",
                "file_type": ".md",
                "source_path": "data/documents/fastapi_overview.md",
                "chunk_count": 3,
                "total_characters": 1200,
            }
        ],
    }
    mock_session.get.return_value = mock_resp

    client = RAGApiClient(base_url="http://testserver", session=mock_session)
    res = client.list_documents()
    assert res["success"] is True
    assert res["data"]["total_documents"] == 2


def test_api_client_upload_documents():
    mock_session = MagicMock()
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "uploaded_files": ["test.md"],
        "total_chunks_indexed": 2,
        "total_chunks_updated": 0,
        "total_chunks_skipped": 0,
        "total_store_chunks": 14,
        "failed_files": [],
    }
    mock_session.post.return_value = mock_resp

    client = RAGApiClient(base_url="http://testserver", session=mock_session)
    files = [("test.md", b"# Test", "text/markdown")]
    res = client.upload_documents(files)
    assert res["success"] is True
    assert res["data"]["uploaded_files"] == ["test.md"]


def test_api_client_submit_feedback():
    mock_session = MagicMock()
    mock_resp = MagicMock()
    mock_resp.status_code = 201
    mock_resp.json.return_value = {
        "status": "success",
        "message": "Feedback recorded",
        "feedback": {"id": 1, "query_id": "test-123", "rating": "thumbs_up"},
    }
    mock_session.post.return_value = mock_resp

    client = RAGApiClient(base_url="http://testserver", session=mock_session)
    res = client.submit_feedback(
        query_id="test-123", rating="thumbs_up", comment="Great"
    )
    assert res["success"] is True
    assert res["data"]["feedback"]["rating"] == "thumbs_up"

