from fastapi import FastAPI
from contextlib import asynccontextmanager
from src.api.routes import router
from src.core.config import settings
from src.api.recommender_service import load_api_config, load_recommender_artifacts
import logging

logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        config = load_api_config()
        artifacts = load_recommender_artifacts(config)
        app.state.api_config = config
        app.state.recommender_artifacts = artifacts
        logger.info("Recommender artifacts loaded successfully.")
    except Exception as e:
        logger.error(f"Failed to load artifacts: {e}")
        app.state.api_config = {}
        app.state.recommender_artifacts = {}
    yield
    app.state.recommender_artifacts = {}

app = FastAPI(
    title=settings.app_name,
    version=settings.version,
    lifespan=lifespan
)

app.include_router(router)
