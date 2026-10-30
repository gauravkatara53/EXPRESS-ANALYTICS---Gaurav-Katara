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
    assert data["chunking_config"]["chunk_size"] > 0
    assert data["chunking_config"]["chunk_overlap"] >= 0
    assert data["phase_status"]["phase_1_project_setup"] == "completed"
    assert data["phase_status"]["phase_2_ingestion_and_chunking"] == "completed"


def test_ingest_preview_endpoint():
    response = client.get("/api/v1/ingest/preview")
    assert response.status_code == 200
    data = response.json()
    assert data["total_files_loaded"] >= 4
    assert data["total_chunks_created"] > 0
    assert data["average_chunk_length"] > 0
    assert len(data["chunks_preview"]) > 0

    first_chunk = data["chunks_preview"][0]
    assert "content" in first_chunk
    assert "metadata" in first_chunk
    assert "source_filename" in first_chunk["metadata"]
    assert "document_title" in first_chunk["metadata"]
    assert "chunk_index" in first_chunk["metadata"]


def test_ingest_preview_invalid_overlap():
    # overlap >= chunk_size should return 400 Bad Request
    response = client.get("/api/v1/ingest/preview?chunk_size=200&chunk_overlap=300")
    assert response.status_code == 400
    assert "chunk_overlap" in response.json()["detail"]
