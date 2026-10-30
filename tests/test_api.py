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
    assert data["chunking_config"]["chunk_size"] > 0
    assert data["chunking_config"]["chunk_overlap"] >= 0
    assert data["phase_status"]["phase_1_project_setup"] == "completed"
    assert data["phase_status"]["phase_2_ingestion_and_chunking"] == "completed"
    assert data["phase_status"]["phase_3_vector_store_and_embeddings"] == "completed"


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
    response = client.get("/api/v1/ingest/preview?chunk_size=200&chunk_overlap=300")
    assert response.status_code == 400
    assert "chunk_overlap" in response.json()["detail"]


def test_index_and_search_endpoints():
    # 1. Index documents into ChromaDB via API
    index_res = client.post("/api/v1/ingest/index")
    assert index_res.status_code == 200
    index_data = index_res.json()
    assert index_data["total_submitted"] > 0
    assert index_data["total_store_chunks"] > 0

    # 2. Check store stats
    stats_res = client.get("/api/v1/store/stats")
    assert stats_res.status_code == 200
    stats_data = stats_res.json()
    assert stats_data["total_chunks"] > 0
    assert len(stats_data["indexed_documents"]) >= 4

    # 3. Perform semantic search
    search_payload = {
        "query": "How does Pydantic validate BaseModel fields?",
        "top_k": 3,
        "score_threshold": 0.0,
    }
    search_res = client.post("/api/v1/search", json=search_payload)
    assert search_res.status_code == 200
    search_data = search_res.json()
    assert search_data["total_results"] > 0
    assert len(search_data["results"]) > 0

    top_result = search_data["results"][0]
    assert "chunk_id" in top_result
    assert "content" in top_result
    assert "metadata" in top_result
    assert "score" in top_result
    assert top_result["metadata"]["document_title"] != ""
