from typing import List, Optional

from src.api.schemas import FeedbackRequest, FeedbackResponse
from src.database.feedback_db import FeedbackDatabase, FeedbackRecord, get_feedback_db
from src.utils.logger import get_logger

logger = get_logger("feedback_service")


class FeedbackService:
    """Service for validating and recording user feedback in SQLite."""

    def __init__(self, db: Optional[FeedbackDatabase] = None):
        self.db = db or get_feedback_db()

    def record_feedback(self, request: FeedbackRequest) -> FeedbackResponse:
        """Stores a new feedback rating and comment."""
        record = self.db.add_feedback(
            query_id=request.query_id,
            rating=request.rating,
            comment=request.comment,
        )
        return FeedbackResponse(
            status="success",
            message=f"Feedback recorded successfully for query '{request.query_id}'.",
            feedback=record,
        )

    def list_feedback(self, query_id: Optional[str] = None) -> List[FeedbackRecord]:
        """Returns all recorded feedback entries."""
        return self.db.get_feedback(query_id=query_id)
