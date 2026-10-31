import io
from pathlib import Path
from fastapi.testclient import TestClient
from src.main import app

client = TestClient(app)


def test_health_check_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "app_name" in data
    assert "environment" in data


def test_system_info_endpoint():
    response = client.get("/info")
    assert response.status_code == 200
    data = response.json()
    assert "chunking_config" in data
    assert "vector_store_config" in data
    assert "llm_config" in data
    assert data["phase_status"]["phase_1_project_setup"] == "completed"
    assert data["phase_status"]["phase_2_ingestion_and_chunking"] == "completed"
    assert data["phase_status"]["phase_3_vector_store_and_embeddings"] == "completed"
    assert data["phase_status"]["phase_4_langgraph_workflow"] == "completed"
    assert data["phase_status"]["phase_5_api_endpoints"] == "completed"


def test_ingest_preview_endpoint():
    response = client.get("/api/v1/ingest/preview")
    assert response.status_code == 200
    data = response.json()
    assert data["total_files_loaded"] >= 4
    assert data["total_chunks_created"] > 0
    assert len(data["chunks_preview"]) > 0


def test_ingest_preview_invalid_overlap():
    response = client.get("/api/v1/ingest/preview?chunk_size=200&chunk_overlap=300")
    assert response.status_code == 400
    assert "chunk_overlap" in response.json()["detail"]


def test_direct_search_and_stats_endpoints():
    stats_res = client.get("/api/v1/store/stats")
    assert stats_res.status_code == 200
    stats_data = stats_res.json()
    assert "collection_name" in stats_data
    assert "total_chunks" in stats_data

    search_payload = {
        "query": "Pydantic BaseModel validation",
        "top_k": 2,
        "score_threshold": 0.0,
    }
    search_res = client.post("/api/v1/search", json=search_payload)
    assert search_res.status_code == 200
    search_data = search_res.json()
    assert "results" in search_data


# Core endpoint tests: /query, /ingest, /documents, /feedback
def test_query_endpoint_success():
    payload = {
        "question": "How do I validate models with Pydantic BaseModel?",
        "max_retries": 2,
    }
    response = client.post("/api/v1/query", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert "query_id" in data
    assert len(data["query_id"]) > 0
    assert data["question"] == payload["question"]
    assert len(data["answer"]) > 0
    assert "citations" in data
    assert not data["is_fallback"]
    assert data["retry_count"] >= 0


def test_query_endpoint_fallback_for_off_topic():
    payload = {
        "question": "What is the secret recipe for Italian lasagna pasta?",
        "max_retries": 1,
    }
    response = client.post("/api/v1/query", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert data["is_fallback"]
    assert data["citations"] == []
    assert "not contain sufficient information" in data["answer"].lower() or "sorry" in data["answer"].lower()


def test_query_endpoint_empty_question():
    # Blank or whitespace-only questions must return 422 Unprocessable Entity
    response = client.post("/api/v1/query", json={"question": "   "})
    assert response.status_code == 422


def test_upload_file_ingest_endpoint():
    sample_file_content = (
        b"# Docker Deployment Guide\n\n"
        b"FastAPI applications can be containerized using official Python slim images. "
        b"Run uvicorn with host 0.0.0.0 inside the Dockerfile container."
    )
    files = [
        ("files", ("docker_deploy_guide.md", io.BytesIO(sample_file_content), "text/markdown")),
    ]
    try:
        response = client.post("/api/v1/ingest", files=files)
        assert response.status_code == 200
        data = response.json()

        assert "docker_deploy_guide.md" in data["uploaded_files"]
        assert data["total_store_chunks"] > 0
        assert data["failed_files"] == []
    finally:
        uploaded_doc = Path("data/documents/docker_deploy_guide.md")
        if uploaded_doc.exists():
            uploaded_doc.unlink()
        from src.api.routes import get_vector_store
        get_vector_store().delete_document("docker_deploy_guide.md")


def test_upload_file_unsupported_format():
    unsupported_content = b"Binary executable content"
    files = [
        ("files", ("malicious_script.exe", io.BytesIO(unsupported_content), "application/octet-stream")),
    ]
    response = client.post("/api/v1/ingest", files=files)
    assert response.status_code == 200
    data = response.json()

    assert len(data["failed_files"]) == 1
    assert "unsupported format" in data["failed_files"][0]


def test_documents_catalog_endpoint():
    response = client.get("/api/v1/documents")
    assert response.status_code == 200
    data = response.json()

    assert "total_documents" in data
    assert "total_chunks" in data
    assert "documents" in data
    assert data["total_documents"] >= 4

    doc_sample = data["documents"][0]
    assert "source_filename" in doc_sample
    assert "document_title" in doc_sample
    assert "chunk_count" in doc_sample
    assert doc_sample["chunk_count"] > 0


def test_feedback_endpoint_success():
    payload = {
        "query_id": "test-uuid-9876",
        "rating": "thumbs_up",
        "comment": "Accurate response with proper source citations.",
    }
    response = client.post("/api/v1/feedback", json=payload)
    assert response.status_code == 201
    data = response.json()

    assert data["status"] == "success"
    assert data["feedback"]["query_id"] == "test-uuid-9876"
    assert data["feedback"]["rating"] == "thumbs_up"
    assert data["feedback"]["comment"] == "Accurate response with proper source citations."
    assert "created_at" in data["feedback"]


def test_feedback_endpoint_invalid_rating():
    payload = {
        "query_id": "test-uuid-1111",
        "rating": "neutral_average",
    }
    response = client.post("/api/v1/feedback", json=payload)
    assert response.status_code == 422


def test_root_route_aliases():
    # 1. GET /documents
    res_docs = client.get("/documents")
    assert res_docs.status_code == 200
    assert "total_documents" in res_docs.json()

    # 2. POST /feedback
    res_fb = client.post(
        "/feedback",
        json={"query_id": "alias-query-1", "rating": "thumbs_down", "comment": "Too brief."},
    )
    assert res_fb.status_code == 201
    assert res_fb.json()["status"] == "success"


def test_url_ingest_endpoint(monkeypatch):
    import httpx

    class MockResponse:
        status_code = 200
        headers = {"content-type": "text/markdown"}
        text = "# Remote FastAPI Guide\n\nRemote FastAPI documentation fetched from a URL."
        content = text.encode("utf-8")

    async def mock_get(self, url, *args, **kwargs):
        return MockResponse()

    monkeypatch.setattr(httpx.AsyncClient, "get", mock_get)

    try:
        response = client.post("/api/v1/ingest/url", json={"urls": ["https://fastapi.tiangolo.com/tutorial"]})
        assert response.status_code == 200
        data = response.json()
        assert len(data["uploaded_files"]) == 1
        assert "fastapi_tiangolo_com_tutorial.md" in data["uploaded_files"][0]
        assert data["total_chunks_indexed"] >= 1
    finally:
        target = Path("data/documents/fastapi_tiangolo_com_tutorial.md")
        if target.exists():
            target.unlink()
        from src.api.routes import get_vector_store
        get_vector_store().delete_document("fastapi_tiangolo_com_tutorial.md")


def test_url_ingest_invalid_url():
    response = client.post("/api/v1/ingest/url", json={"urls": ["not_a_valid_url"]})
    assert response.status_code == 200
    data = response.json()
    assert len(data["failed_files"]) == 1
    assert "invalid URL" in data["failed_files"][0]
