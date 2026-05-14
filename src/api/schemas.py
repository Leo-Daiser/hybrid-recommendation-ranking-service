from pydantic import BaseModel, ConfigDict, field_validator
from typing import List, Optional

VALID_EVENT_TYPES = {"impression", "click", "like", "dislike", "skip"}


class HealthCheckResponse(BaseModel):
    status: str
    version: str


class RecommendationItem(BaseModel):
    item_id: int
    score: float
    rank: int
    retrieval_score: Optional[float] = None
    explanation: dict


class RecommendationResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    request_id: str
    user_id: int
    items: List[RecommendationItem]
    model_version: str
    fallback_used: bool
    scoring_mode: str
    warnings: List[str] = []


class FeedbackRequest(BaseModel):
    request_id: Optional[str] = None
    user_id: int
    item_id: int
    event_type: str
    event_value: Optional[float] = None
    metadata: Optional[dict] = None

    @field_validator("event_type")
    @classmethod
    def validate_event_type(cls, v: str) -> str:
        if v not in VALID_EVENT_TYPES:
            raise ValueError(
                f"Invalid event_type '{v}'. Must be one of {sorted(VALID_EVENT_TYPES)}."
            )
        return v


class FeedbackResponse(BaseModel):
    feedback_id: str
    status: str
