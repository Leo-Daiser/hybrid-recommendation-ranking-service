from fastapi import APIRouter
from src.api.schemas import HealthCheckResponse
from src.core.config import settings

router = APIRouter()

@router.get("/health", response_model=HealthCheckResponse)
def health_check():
    return HealthCheckResponse(
        status="ok",
        version=settings.version
    )
