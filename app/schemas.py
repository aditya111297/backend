from typing import Any, Optional
from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str
    app_name: str
    model_loaded: bool
    best_model_name: Optional[str] = None
    database_enabled: bool
    artifacts_status: dict[str, bool]
    missing_artifacts: list[str]
    errors: list[str]


class PredictionRequest(BaseModel):
    customer_id: Optional[str] = Field(default=None, description="Optional Customer Identifier")
    features: dict[str, Any] = Field(..., description="Customer feature values keyed by feature name")
    low_threshold: Optional[float] = Field(default=None, ge=0.01, le=0.98)
    high_threshold: Optional[float] = Field(default=None, ge=0.02, le=0.99)


class PredictionResponse(BaseModel):
    customer_id: Optional[str] = None
    prediction: int
    prediction_label: str
    churn_probability: float
    retention_probability: float
    risk_category: str
    thresholds_used: dict[str, float]
    threshold_disclaimer: str
    recommended_actions: list[str]
    model_used: str
