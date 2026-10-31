import os
from typing import Any, Dict, List, Optional, Tuple
import requests


class RAGApiClient:
    """HTTP client connecting Streamlit to the FastAPI backend."""

    def __init__(
        self,
        base_url: Optional[str] = None,
        timeout: float = 60.0,
        session: Optional[requests.Session] = None,
    ):
        raw_url = base_url or os.getenv("API_BASE_URL", "http://localhost:8000")
        self.base_url = raw_url.rstrip("/")
        self.timeout = timeout
        self.session = session or requests.Session()

    def check_health(self) -> Dict[str, Any]:
        """Check whether the FastAPI backend is running."""
        url = f"{self.base_url}/health"
        try:
            resp = self.session.get(url, timeout=3.0)
            if resp.status_code == 200:
                return {"healthy": True, "data": resp.json()}
            return {"healthy": False, "error": f"HTTP {resp.status_code}: {resp.text}"}
        except requests.exceptions.ConnectionError:
            return {
                "healthy": False,
                "error": f"Cannot connect to backend at {self.base_url}. Ensure the FastAPI server is running.",
            }
        except Exception as e:
            return {"healthy": False, "error": str(e)}

    def query_assistant(self, question: str, max_retries: int = 1) -> Dict[str, Any]:
        """Submit a question to POST /api/v1/query."""
        url = f"{self.base_url}/api/v1/query"
        payload = {"question": question, "max_retries": max_retries}
        try:
            resp = self.session.post(url, json=payload, timeout=self.timeout)
            if resp.status_code == 200:
                return {"success": True, "data": resp.json()}
            elif resp.status_code == 422:
                err_detail = resp.json().get("detail", "Validation error")
                return {"success": False, "error": f"Invalid query: {err_detail}"}
            else:
                return {
                    "success": False,
                    "error": f"Backend Error (HTTP {resp.status_code}): {resp.text}",
                }
        except requests.exceptions.ConnectionError:
            return {
                "success": False,
                "error": f"Connection failed: FastAPI backend is unreachable at {self.base_url}.",
            }
        except requests.exceptions.Timeout:
            return {
                "success": False,
                "error": "Request timed out waiting for LangGraph workflow execution.",
            }
        except Exception as e:
            return {"success": False, "error": f"Unexpected error: {str(e)}"}

    def list_documents(self) -> Dict[str, Any]:
        """Retrieve catalog of indexed documents from GET /api/v1/documents."""
        url = f"{self.base_url}/api/v1/documents"
        try:
            resp = self.session.get(url, timeout=10.0)
            if resp.status_code == 200:
                return {"success": True, "data": resp.json()}
            return {
                "success": False,
                "error": f"Failed to fetch documents (HTTP {resp.status_code}): {resp.text}",
            }
        except requests.exceptions.ConnectionError:
            return {
                "success": False,
                "error": f"Backend unreachable at {self.base_url}.",
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    def upload_documents(
        self, files: List[Tuple[str, bytes, str]]
    ) -> Dict[str, Any]:
        """Upload and index documents via POST /api/v1/ingest."""
        url = f"{self.base_url}/api/v1/ingest"
        multipart_files = [
            ("files", (filename, data, ctype)) for filename, data, ctype in files
        ]
        try:
            resp = self.session.post(url, files=multipart_files, timeout=60.0)
            if resp.status_code in (200, 201):
                return {"success": True, "data": resp.json()}
            return {
                "success": False,
                "error": f"Ingestion failed (HTTP {resp.status_code}): {resp.text}",
            }
        except requests.exceptions.ConnectionError:
            return {
                "success": False,
                "error": f"Backend unreachable at {self.base_url}.",
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    def submit_feedback(
        self, query_id: str, rating: str, comment: Optional[str] = None
    ) -> Dict[str, Any]:
        """Submit user feedback to POST /api/v1/feedback."""
        url = f"{self.base_url}/api/v1/feedback"
        payload = {"query_id": query_id, "rating": rating, "comment": comment}
        try:
            resp = self.session.post(url, json=payload, timeout=10.0)
            if resp.status_code in (200, 201):
                return {"success": True, "data": resp.json()}
            return {
                "success": False,
                "error": f"Feedback failed (HTTP {resp.status_code}): {resp.text}",
            }
        except requests.exceptions.ConnectionError:
            return {
                "success": False,
                "error": f"Backend unreachable at {self.base_url}.",
            }
        except Exception as e:
            return {"success": False, "error": str(e)}
