"""Unit tests for DB repository functions using SQLite in-memory."""
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src.db.base import Base
from src.db.repositories import (
    log_recommendation_request,
    log_ranked_recommendations,
    log_feedback_event,
    get_recommendation_request,
    VALID_EVENT_TYPES,
)


@pytest.fixture
def db_session():
    engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False})
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    yield session
    session.close()
    Base.metadata.drop_all(engine)


def test_log_recommendation_request_success(db_session):
    log_recommendation_request(
        session=db_session,
        request_id="req-001",
        user_id=1,
        k=10,
        model_version="v1",
        scoring_mode="ranker",
        fallback_used=False,
        warnings=[],
    )
    db_session.commit()

    rec = get_recommendation_request(db_session, "req-001")
    assert rec is not None
    assert rec.user_id == 1
    assert rec.k == 10
    assert rec.scoring_mode == "ranker"
    assert rec.fallback_used is False


def test_log_ranked_recommendations_success(db_session):
    items = [
        {"item_id": 10, "rank": 1, "score": 0.9, "retrieval_score": 0.5, "explanation": {"source": "itemknn"}},
        {"item_id": 20, "rank": 2, "score": 0.7, "retrieval_score": 0.4, "explanation": {"source": "itemknn"}},
    ]
    log_ranked_recommendations(
        session=db_session,
        request_id="req-002",
        user_id=1,
        items=items,
        model_version="v1",
        scoring_mode="ranker",
    )
    db_session.commit()

    from src.db.models import RankedRecommendation
    rows = db_session.query(RankedRecommendation).filter_by(request_id="req-002").all()
    assert len(rows) == 2
    assert rows[0].item_id == 10
    assert rows[1].item_id == 20
    assert rows[0].rank == 1


def test_log_feedback_event_success(db_session):
    log_feedback_event(
        session=db_session,
        feedback_id="fb-001",
        request_id="req-001",
        user_id=1,
        item_id=10,
        event_type="click",
        event_value=1.0,
        metadata={"source": "test"},
    )
    db_session.commit()

    from src.db.models import FeedbackLog
    row = db_session.query(FeedbackLog).filter_by(feedback_id="fb-001").first()
    assert row is not None
    assert row.user_id == 1
    assert row.item_id == 10
    assert row.event_type == "click"
    assert row.event_value == 1.0


def test_log_feedback_event_invalid_type(db_session):
    with pytest.raises(ValueError, match="Invalid event_type"):
        log_feedback_event(
            session=db_session,
            feedback_id="fb-002",
            request_id=None,
            user_id=1,
            item_id=10,
            event_type="purchase",  # not in VALID_EVENT_TYPES
        )


def test_get_recommendation_request_success(db_session):
    log_recommendation_request(
        session=db_session,
        request_id="req-003",
        user_id=5,
        k=5,
        model_version="v2",
        scoring_mode="retrieval_only",
        fallback_used=True,
    )
    db_session.commit()

    result = get_recommendation_request(db_session, "req-003")
    assert result is not None
    assert result.user_id == 5
    assert result.fallback_used is True


def test_get_recommendation_request_not_found(db_session):
    result = get_recommendation_request(db_session, "nonexistent-id")
    assert result is None


def test_all_valid_event_types_accepted(db_session):
    for i, et in enumerate(sorted(VALID_EVENT_TYPES)):
        log_feedback_event(
            session=db_session,
            feedback_id=f"fb-{i}",
            request_id=None,
            user_id=1,
            item_id=i + 1,
            event_type=et,
        )
    db_session.commit()

    from src.db.models import FeedbackLog
    rows = db_session.query(FeedbackLog).all()
    assert len(rows) == len(VALID_EVENT_TYPES)
