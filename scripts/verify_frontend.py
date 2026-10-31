import os
import sys
import threading
import time
from pathlib import Path

# Add project root and frontend directory to path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(ROOT_DIR / "frontend"))

import uvicorn
from frontend.api_client import RAGApiClient
from src.main import app


class BackgroundServer(threading.Thread):
    """Run an in-process Uvicorn server for local verifications."""

    def __init__(self, host: str = "127.0.0.1", port: int = 8000):
        super().__init__(daemon=True)
        config = uvicorn.Config(app, host=host, port=port, log_level="error")
        self.server = uvicorn.Server(config)

    def run(self):
        self.server.run()

    def stop(self):
        self.server.should_exit = True


def verify_frontend_and_api():
    base_url = "http://127.0.0.1:8000"
    client = RAGApiClient(base_url=base_url)

    # Check if a live server is already running
    spawned_server = None
    health_check = client.check_health()
    if not health_check.get("healthy"):
        print(f"Starting in-process FastAPI test server on {base_url}...")
        spawned_server = BackgroundServer()
        spawned_server.start()
        # Wait up to 5 seconds for health
        for _ in range(25):
            time.sleep(0.2)
            if client.check_health().get("healthy"):
                break

    try:
        print("\n=================================================================")
        print("EXPRESS ANALYTICS | FRONTEND & API CLIENT VERIFICATION")
        print("=================================================================")

        # 1. Health check
        print("1. Testing Health Check (GET /health)...")
        health = client.check_health()
        assert health["healthy"] is True, f"Health check failed: {health}"
        print(f"   [PASS] Backend connected: {health['data']['app_name']}")

        # 2. List documents in explorer
        print("2. Testing Document Explorer (GET /api/v1/documents)...")
        docs_resp = client.list_documents()
        assert docs_resp["success"] is True, f"List documents failed: {docs_resp}"
        data = docs_resp["data"]
        initial_doc_count = data["total_documents"]
        initial_chunk_count = data["total_chunks"]
        print(f"   [PASS] Found {initial_doc_count} documents and {initial_chunk_count} indexed chunks.")

        # 3. Ask technical question with citations
        print("3. Testing Technical Question (POST /api/v1/query)...")
        q_resp = client.query_assistant("How do I validate models with Pydantic BaseModel?")
        assert q_resp["success"] is True, f"Query failed: {q_resp}"
        q_data = q_resp["data"]
        assert q_data["query_id"], "Missing query_id"
        assert len(q_data["citations"]) > 0, "Expected at least one citation"
        assert not q_data["is_fallback"], "Should not trigger safe fallback for on-topic query"
        query_id = q_data["query_id"]
        print(f"   [PASS] Grounded answer generated.")
        print(f"          Citations: {q_data['citations']}")
        print(f"          Query ID: {query_id}")

        # 4. Ask off-topic query and verify safe fallback
        print("4. Testing Off-Topic Question Safe Fallback...")
        fb_q_resp = client.query_assistant("What is the recipe for baking chocolate chip cookies?")
        assert fb_q_resp["success"] is True, f"Query failed: {fb_q_resp}"
        fb_q_data = fb_q_resp["data"]
        assert fb_q_data["is_fallback"] is True, f"Expected is_fallback=True, got: {fb_q_data}"
        assert fb_q_data["citations"] == [], f"Expected empty citations on fallback, got: {fb_q_data['citations']}"
        print("   [PASS] Safe fallback correctly triggered without hallucination.")

        # 5. Submit thumbs-up feedback
        print("5. Testing Feedback Submission (POST /api/v1/feedback)...")
        feedback_resp = client.submit_feedback(
            query_id=query_id,
            rating="thumbs_up",
            comment="Automated verification feedback: highly relevant citation."
        )
        assert feedback_resp["success"] is True, f"Feedback submission failed: {feedback_resp}"
        print(f"   [PASS] Feedback persisted in SQLite: {feedback_resp['data']['feedback']}")

        # 6. Upload new document
        print("6. Testing Document Upload (POST /api/v1/ingest)...")
        new_doc_content = (
            b"# Deployment Architecture Guide\n\n"
            b"FastAPI microservices can be packaged into Docker images for scalable Kubernetes clusters. "
            b"Healthcheck probes connect to the /health endpoint every 30 seconds."
        )
        upload_resp = client.upload_documents([
            ("deployment_architecture.md", new_doc_content, "text/markdown")
        ])
        assert upload_resp["success"] is True, f"Upload failed: {upload_resp}"
        up_data = upload_resp["data"]
        assert "deployment_architecture.md" in up_data["uploaded_files"]
        print(f"   [PASS] Document uploaded and indexed: {up_data['total_chunks_indexed']} new chunks.")

        # 7. Verify Document Explorer reflects newly uploaded document
        print("7. Verifying Document Explorer updates...")
        docs_after = client.list_documents()
        assert docs_after["success"] is True
        filenames = [d["source_filename"] for d in docs_after["data"]["documents"]]
        assert "deployment_architecture.md" in filenames
        print(f"   [PASS] Newly uploaded document visible in catalog ({len(filenames)} total docs).")

        # 8. Query about newly uploaded document
        print("8. Testing query about newly uploaded document...")
        doc_q_resp = client.query_assistant("How are healthcheck probes configured for Kubernetes clusters?")
        assert doc_q_resp["success"] is True
        assert "deployment_architecture.md" in doc_q_resp["data"]["citations"]
        print(f"   [PASS] Answer correctly cited newly uploaded document: {doc_q_resp['data']['citations']}")

        # Clean up uploaded test document from disk and vector store
        test_file_path = ROOT_DIR / "data" / "documents" / "deployment_architecture.md"
        if test_file_path.exists():
            test_file_path.unlink()
        from src.retrieval.vector_store import ChromaVectorStore
        ChromaVectorStore().delete_document("deployment_architecture.md")

        print("\n=================================================================")
        print("SUCCESS: ALL 8 FRONTEND & BACKEND INTEGRATION CHECKS PASSED!")
        print("=================================================================\n")

    finally:
        if spawned_server:
            spawned_server.stop()


if __name__ == "__main__":
    verify_frontend_and_api()
