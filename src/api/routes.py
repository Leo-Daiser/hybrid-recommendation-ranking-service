import uuid
import logging
from contextlib import contextmanager
from fastapi import APIRouter, Request, HTTPException, Query

from src.api.schemas import (
    HealthCheckResponse,
    RecommendationResponse,
    FeedbackRequest,
    FeedbackResponse,
)
from src.core.config import settings
from src.api.recommender_service import recommend_for_user

logger = logging.getLogger(__name__)
router = APIRouter()

# ---------------------------------------------------------------------------
# DB imports — module-level so tests can patch src.api.routes.session_scope etc.
# We use a try/except with no-op stubs so the API starts even without PostgreSQL.
# ---------------------------------------------------------------------------
try:
    from src.db.session import session_scope as _real_session_scope
    from src.db.repositories import (
        log_recommendation_request as _real_log_req,
        log_ranked_recommendations as _real_log_recs,
        log_feedback_event as _real_log_fb,
    )
    _DB_IMPORT_OK = True
except Exception as _db_import_err:
    logger.warning("DB modules could not be imported: %s. DB logging will be disabled.", _db_import_err)
    _DB_IMPORT_OK = False

    @contextmanager
    def _real_session_scope():
        raise RuntimeError("DB not available")
        yield  # noqa: unreachable

    def _real_log_req(*a, **kw):
        raise RuntimeError("DB not available")

    def _real_log_recs(*a, **kw):
        raise RuntimeError("DB not available")

    def _real_log_fb(*a, **kw):
        raise RuntimeError("DB not available")


# These module-level names are the canonical patch targets for tests:
#   patch("src.api.routes.session_scope")
#   patch("src.api.routes.log_recommendation_request")
#   patch("src.api.routes.log_ranked_recommendations")
#   patch("src.api.routes.log_feedback_event")
session_scope = _real_session_scope
log_recommendation_request = _real_log_req
log_ranked_recommendations = _real_log_recs
log_feedback_event = _real_log_fb


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@router.get("/health", response_model=HealthCheckResponse)
def health_check():
    return HealthCheckResponse(status="ok", version=settings.version)


@router.get("/recommend", response_model=RecommendationResponse)
def get_recommendations(
    request: Request,
    user_id: int = Query(..., description="User ID for recommendations"),
    k: int = Query(10, description="Number of items to recommend", ge=1),
):
    app_state = request.app.state
    artifacts = getattr(app_state, "recommender_artifacts", {})
    config = getattr(app_state, "api_config", {})

    if "candidate_cache" not in artifacts and "popularity_cache" not in artifacts:
        raise HTTPException(status_code=503, detail="Retrieval artifacts unavailable")

    try:
        res = recommend_for_user(user_id, k, artifacts, config)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    # --- DB logging (non-blocking: failure adds warning, never breaks serving) ---
    try:
        with session_scope() as db_session:
            log_recommendation_request(
                session=db_session,
                request_id=res["request_id"],
                user_id=res["user_id"],
                k=k,
                model_version=res["model_version"],
                scoring_mode=res["scoring_mode"],
                fallback_used=res["fallback_used"],
                warnings=res.get("warnings", []),
            )
            log_ranked_recommendations(
                session=db_session,
                request_id=res["request_id"],
                user_id=res["user_id"],
                items=res["items"],
                model_version=res["model_version"],
                scoring_mode=res["scoring_mode"],
            )
    except Exception as db_err:
        warn_msg = f"db_logging_failed: {db_err}"
        logger.warning(warn_msg)
        res.setdefault("warnings", []).append(warn_msg)

    return res


@router.post("/feedback", response_model=FeedbackResponse)
def post_feedback(request: Request, feedback: FeedbackRequest):
    feedback_id = str(uuid.uuid4())

    try:
        with session_scope() as db_session:
            log_feedback_event(
                session=db_session,
                feedback_id=feedback_id,
                request_id=feedback.request_id,
                user_id=feedback.user_id,
                item_id=feedback.item_id,
                event_type=feedback.event_type,
                event_value=feedback.event_value,
                metadata=feedback.metadata,
            )
    except ValueError as ve:
        raise HTTPException(status_code=422, detail=str(ve))
    except Exception as db_err:
        logger.error("Feedback DB logging failed: %s", db_err)
        raise HTTPException(status_code=503, detail=f"DB unavailable: {db_err}")

    return FeedbackResponse(feedback_id=feedback_id, status="logged")
