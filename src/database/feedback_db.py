from datetime import datetime, timezone
from pathlib import Path
import sqlite3
from typing import List, Optional
from pydantic import BaseModel, Field

from src.config import settings
from src.utils.logger import get_logger

logger = get_logger("feedback_db")


class FeedbackRecord(BaseModel):
    """Pydantic schema for stored feedback records."""

    id: int
    query_id: str
    rating: str
    comment: Optional[str] = None
    created_at: str


class FeedbackDatabase:
    """Manages persistent SQLite storage for user answer feedback."""

    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = Path(db_path or settings.SQLITE_DB_PATH).resolve()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        """Initializes the feedback table schema if not already present."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS feedback (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    query_id TEXT NOT NULL,
                    rating TEXT NOT NULL,
                    comment TEXT,
                    created_at TEXT NOT NULL
                )
                """
            )
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_feedback_query_id ON feedback (query_id)"
            )
            conn.commit()
        logger.info(f"Initialized Feedback SQLite database at '{self.db_path}'.")

    def add_feedback(
        self,
        query_id: str,
        rating: str,
        comment: Optional[str] = None,
    ) -> FeedbackRecord:
        """Stores a new feedback entry persistently.

        Args:
            query_id: Unique identifier of the query being rated.
            rating: Rating value ('thumbs_up' or 'thumbs_down').
            comment: Optional qualitative feedback from the user.

        Returns:
            The created FeedbackRecord instance.
        """
        now = datetime.now(timezone.utc).isoformat()
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO feedback (query_id, rating, comment, created_at)
                VALUES (?, ?, ?, ?)
                """,
                (query_id.strip(), rating.strip().lower(), comment.strip() if comment else None, now),
            )
            conn.commit()
            feedback_id = cursor.lastrowid

        logger.info(f"Recorded feedback #{feedback_id} for query '{query_id}': rating={rating}")
        return FeedbackRecord(
            id=feedback_id,
            query_id=query_id,
            rating=rating.strip().lower(),
            comment=comment,
            created_at=now,
        )

    def get_feedback(self, query_id: Optional[str] = None) -> List[FeedbackRecord]:
        """Retrieves feedback records, optionally filtered by query_id."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            if query_id:
                cursor.execute(
                    "SELECT id, query_id, rating, comment, created_at FROM feedback WHERE query_id = ? ORDER BY id DESC",
                    (query_id.strip(),),
                )
            else:
                cursor.execute(
                    "SELECT id, query_id, rating, comment, created_at FROM feedback ORDER BY id DESC"
                )
            rows = cursor.fetchall()

        return [
            FeedbackRecord(
                id=row["id"],
                query_id=row["query_id"],
                rating=row["rating"],
                comment=row["comment"],
                created_at=row["created_at"],
            )
            for row in rows
        ]

    def count(self) -> int:
        """Returns the total number of feedback records stored."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM feedback")
            return cursor.fetchone()[0]


# Singleton instance
_db_instance: Optional[FeedbackDatabase] = None


def get_feedback_db() -> FeedbackDatabase:
    """Dependency provider for FeedbackDatabase."""
    global _db_instance
    if _db_instance is None:
        _db_instance = FeedbackDatabase()
    return _db_instance
