import io
from typing import Optional
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
import pandas as pd
from app.database import db
from app.model_loader import registry
from app.schemas import PredictionRequest, PredictionResponse
from app.services.prediction_service import predict_batch_dataframe, predict_single

router = APIRouter(prefix="/api", tags=["Prediction"])


@router.get("/predict/schema")
def get_prediction_form_schema():
    """Returns dynamic feature metadata exported by the notebook so the frontend builds exact form fields."""
    return {
        "model_loaded": registry.pipeline is not None,
        "best_model_name": registry.best_model_name,
        "feature_columns": registry.feature_columns,
        "features": registry.feature_metadata,
    }


@router.post("/predict", response_model=PredictionResponse)
def run_single_prediction(payload: PredictionRequest):
    if registry.pipeline is None:
        raise HTTPException(
            status_code=503,
            detail="Trained model pipeline not loaded. Place 'customer_churn_pipeline.joblib' inside outputs/.",
        )

    try:
        result = predict_single(
            features=payload.features,
            customer_id=payload.customer_id,
            low_th=payload.low_threshold,
            high_th=payload.high_threshold,
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Prediction error: {exc}")

    db.log_prediction(
        prediction_type="individual",
        customer_id=payload.customer_id,
        input_data=payload.features,
        prediction_label=result["prediction_label"],
        churn_probability=result["churn_probability"],
        risk_category=result["risk_category"],
    )
    return PredictionResponse(**result)


@router.post("/predict/batch")
async def run_batch_prediction(
    file: UploadFile = File(...),
    low_threshold: Optional[float] = Form(default=None),
    high_threshold: Optional[float] = Form(default=None),
    download_csv: bool = Form(default=False),
):
    if registry.pipeline is None:
        raise HTTPException(
            status_code=503,
            detail="Trained model pipeline not loaded. Place 'customer_churn_pipeline.joblib' inside outputs/.",
        )

    if not file.filename or not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Please upload a valid .csv file.")

    try:
        raw_bytes = await file.read()
        df = pd.read_csv(io.BytesIO(raw_bytes))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid CSV file: {exc}")

    if df.empty:
        raise HTTPException(status_code=400, detail="Uploaded CSV contains no records.")

    matched_cols = [c for c in registry.feature_columns if c in df.columns]
    if len(matched_cols) == 0:
        raise HTTPException(
            status_code=400,
            detail=f"CSV does not match any expected model features: {registry.feature_columns}",
        )

    try:
        result_df, summary = predict_batch_dataframe(df, low_th=low_threshold, high_th=high_threshold)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {exc}")

    db.log_activity(
        "batch_prediction",
        f"Processed {summary['total_processed']} records from {file.filename} (High risk: {summary['high_risk_customers']})",
    )

    if download_csv:
        stream = io.StringIO()
        result_df.to_csv(stream, index=False)
        stream.seek(0)
        out_name = f"predicted_{file.filename}"
        return StreamingResponse(
            iter([stream.getvalue()]),
            media_type="text/csv",
            headers={"Content-Disposition": f'attachment; filename="{out_name}"'},
        )

    csv_text = result_df.to_csv(index=False)
    preview_records = result_df.head(50).fillna("").to_dict(orient="records")

    return {
        "filename": file.filename,
        "summary": summary,
        "matched_features": matched_cols,
        "missing_features_imputed": [c for c in registry.feature_columns if c not in df.columns],
        "columns": list(result_df.columns),
        "records": preview_records,
        "csv_content": csv_text,
    }


@router.get("/predict/history")
def get_prediction_history(limit: int = 25):
    return {
        "database_enabled": db.enabled,
        "history": db.get_recent_predictions(limit=limit),
    }
