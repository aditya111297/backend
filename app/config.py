import os
from pathlib import Path
from pydantic_settings import BaseSettings

BASE_DIR = Path(__file__).resolve().parent.parent.parent


class Settings(BaseSettings):
    APP_NAME: str = "ChurnAnalytics AI"
    OUTPUTS_DIR: str = str(BASE_DIR / "outputs")
    DATABASE_URL: str = f"sqlite:///{BASE_DIR / 'backend' / 'churn_analytics.db'}"
    ENABLE_DB: bool = True
    CORS_ORIGINS: str = "http://localhost:5173,http://127.0.0.1:5173"
    LOW_RISK_THRESHOLD: float = 0.35
    HIGH_RISK_THRESHOLD: float = 0.70

    class Config:
        env_file = ".env"
        extra = "ignore"

    @property
    def outputs_path(self) -> Path:
        p = Path(self.OUTPUTS_DIR)
        if not p.is_absolute():
            p = (Path(__file__).resolve().parent.parent / p).resolve()
        return p

    @property
    def cors_origins_list(self) -> list[str]:
        return [origin.strip() for origin in self.CORS_ORIGINS.split(",") if origin.strip()]


settings = Settings()
