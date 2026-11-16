from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.config import settings
from app.model_loader import registry
from app.routers import dashboard, dataset, health, models, prediction, reports, risk


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load existing trained model & evaluation artifacts once on startup
    registry.load_all()
    yield


app = FastAPI(
    title=settings.APP_NAME,
    description="Predictive Analytics for Customer Churn Using Large-Scale Customer Data",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router)
app.include_router(dataset.router)
app.include_router(dashboard.router)
app.include_router(prediction.router)
app.include_router(models.router)
app.include_router(risk.router)
app.include_router(reports.router)


@app.get("/")
def root():
    return {
        "app": settings.APP_NAME,
        "docs": "/docs",
        "health": "/api/health",
    }
