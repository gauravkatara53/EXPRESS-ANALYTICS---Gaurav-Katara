import pytest
from pathlib import Path

from src.database.feedback_db import FeedbackDatabase


def test_sqlite_feedback_db(tmp_path: Path):
    db_file = tmp_path / "test_feedback.db"
    db = FeedbackDatabase(db_path=db_file)

    assert db.count() == 0

    # 1. Add feedback record
    rec1 = db.add_feedback(
        query_id="query-12345",
        rating="thumbs_up",
        comment="Accurate citation and clear explanation.",
    )
    assert rec1.id == 1
    assert rec1.query_id == "query-12345"
    assert rec1.rating == "thumbs_up"
    assert rec1.comment == "Accurate citation and clear explanation."
    assert rec1.created_at is not None
    assert db.count() == 1

    # 2. Add second feedback record
    rec2 = db.add_feedback(
        query_id="query-67890",
        rating="thumbs_down",
        comment="Context was missing edge cases.",
    )
    assert rec2.id == 2
    assert db.count() == 2

    # 3. Retrieve all records
    all_recs = db.get_feedback()
    assert len(all_recs) == 2

    # 4. Filter by query_id
    filtered = db.get_feedback(query_id="query-12345")
    assert len(filtered) == 1
    assert filtered[0].rating == "thumbs_up"
