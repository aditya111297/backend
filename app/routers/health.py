from fastapi import APIRouter
from app.config import settings
from app.database import db
from app.model_loader import registry
from app.schemas import HealthResponse

router = APIRouter(prefix="/api", tags=["Health"])


@router.get("/health", response_model=HealthResponse)
def health_check():
    model_loaded = registry.pipeline is not None
    status = "healthy" if model_loaded else "degraded_missing_artifacts"
    return HealthResponse(
        status=status,
        app_name=settings.APP_NAME,
        model_loaded=model_loaded,
        best_model_name=registry.best_model_name,
        database_enabled=db.enabled,
        artifacts_status=registry.artifacts_status,
        missing_artifacts=registry.missing_artifacts,
        errors=registry.errors,
    )
