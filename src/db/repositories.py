"""Repository functions for DB logging — all functions accept an external session."""

import uuid
from typing import Any

from src.db.models import RecommendationRequest, RankedRecommendation, FeedbackLog

VALID_EVENT_TYPES = {"impression", "click", "like", "dislike", "skip"}


def log_recommendation_request(
    session,
    request_id: str,
    user_id: int,
    k: int,
    model_version: str,
    scoring_mode: str,
    fallback_used: bool,
    warnings: list[str] | None = None,
) -> None:
    row = RecommendationRequest(
        request_id=request_id,
        user_id=user_id,
        k=k,
        model_version=model_version,
        scoring_mode=scoring_mode,
        fallback_used=fallback_used,
        warnings_json=warnings or [],
    )
    session.add(row)


def log_ranked_recommendations(
    session,
    request_id: str,
    user_id: int,
    items: list[dict[str, Any]],
    model_version: str,
    scoring_mode: str,
) -> None:
    for item in items:
        row = RankedRecommendation(
            request_id=request_id,
            user_id=user_id,
            item_id=item["item_id"],
            rank=item["rank"],
            score=item.get("score"),
            retrieval_score=item.get("retrieval_score"),
            explanation_json=item.get("explanation"),
            model_version=model_version,
            scoring_mode=scoring_mode,
        )
        session.add(row)


def log_feedback_event(
    session,
    feedback_id: str,
    request_id: str | None,
    user_id: int,
    item_id: int,
    event_type: str,
    event_value: float | None = None,
    metadata: dict | None = None,
) -> None:
    if event_type not in VALID_EVENT_TYPES:
        raise ValueError(
            f"Invalid event_type '{event_type}'. "
            f"Must be one of {sorted(VALID_EVENT_TYPES)}."
        )
    row = FeedbackLog(
        feedback_id=feedback_id,
        request_id=request_id,
        user_id=user_id,
        item_id=item_id,
        event_type=event_type,
        event_value=event_value,
        metadata_json=metadata,
    )
    session.add(row)


def get_recommendation_request(session, request_id: str) -> RecommendationRequest | None:
    return (
        session.query(RecommendationRequest)
        .filter(RecommendationRequest.request_id == request_id)
        .first()
    )
