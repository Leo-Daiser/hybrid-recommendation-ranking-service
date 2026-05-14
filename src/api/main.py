from fastapi import FastAPI
from src.api.routes import router
from src.core.config import settings

app = FastAPI(
    title=settings.app_name,
    version=settings.version
)

app.include_router(router)
