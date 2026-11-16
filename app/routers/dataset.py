import io
from typing import Optional
from fastapi import APIRouter, File, Form, HTTPException, Query, UploadFile
import numpy as np
import pandas as pd
from app.database import db
from app.model_loader import registry

router = APIRouter(prefix="/api/dataset", tags=["Dataset"])


def _normalize_target_series(series: pd.Series) -> pd.Series:
    """Maps target column values into binary 0/1 flags for dashboard analytics."""
    if pd.api.types.is_numeric_dtype(series):
        return (series.fillna(0) > 0).astype(int)
    s_lower = series.astype(str).str.strip().str.lower()
    return s_lower.isin({"1", "yes", "true", "churn", "churned", "exited", "positive"}).astype(int)


@router.post("/upload")
async def upload_dataset(
    file: UploadFile = File(...),
    target_column: Optional[str] = Form(default=None),
):
    if not file.filename or not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only valid .csv files are supported.")

    try:
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Failed to parse CSV file: {exc}")

    if df.empty or len(df.columns) < 2:
        raise HTTPException(status_code=400, detail="Uploaded CSV must contain at least 1 row and 2 columns.")

    detected_target = registry.detect_target_column(df, preferred=target_column)
    registry.set_active_dataset(df, file.filename, detected_target)
    db.log_dataset_upload(file.filename, int(df.shape[0]), int(df.shape[1]), detected_target)
    db.log_activity("dataset_upload", f"Uploaded {file.filename} ({df.shape[0]} rows, {df.shape[1]} cols)")

    return {
        "message": "Dataset uploaded and analyzed successfully.",
        "dataset_name": file.filename,
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "column_names": list(df.columns),
        "detected_target_column": detected_target,
        "requires_target_selection": detected_target is None,
        "feature_columns_matched": [c for c in registry.feature_columns if c in df.columns],
        "missing_model_features": [c for c in registry.feature_columns if c not in df.columns],
    }


@router.get("/preview")
def get_dataset_preview(
    limit: int = Query(default=10, ge=1, le=200),
    search: Optional[str] = Query(default=None),
    columns: Optional[str] = Query(default=None),
):
    if registry.active_dataset_df is None:
        raise HTTPException(
            status_code=404,
            detail="No dataset loaded. Upload a CSV file or provide outputs/customer_churn_dataset.csv.",
        )

    df = registry.active_dataset_df.copy()

    if columns:
        selected = [c.strip() for c in columns.split(",") if c.strip() in df.columns]
        if selected:
            df = df[selected]

    if search and search.strip():
        q = search.strip().lower()
        mask = df.astype(str).apply(lambda col: col.str.lower().str.contains(q, regex=False)).any(axis=1)
        df = df[mask]

    preview_df = df.head(limit).replace({np.nan: None})
    return {
        "dataset_name": registry.active_dataset_name,
        "total_matching_rows": int(len(df)),
        "columns": list(preview_df.columns),
        "records": preview_df.to_dict(orient="records"),
    }


@router.get("/columns")
def get_dataset_columns():
    if registry.active_dataset_df is None:
        raise HTTPException(status_code=404, detail="No dataset loaded.")

    df = registry.active_dataset_df
    col_info = []
    for col in df.columns:
        null_count = int(df[col].isna().sum())
        col_info.append({
            "name": col,
            "dtype": str(df[col].dtype),
            "is_numeric": bool(pd.api.types.is_numeric_dtype(df[col])),
            "unique_values": int(df[col].nunique(dropna=True)),
            "missing_count": null_count,
            "missing_percent": round((null_count / len(df) * 100), 2) if len(df) else 0.0,
            "is_target": col == registry.active_target_column,
            "is_model_feature": col in registry.feature_columns,
        })

    return {
        "dataset_name": registry.active_dataset_name,
        "target_column": registry.active_target_column,
        "columns": col_info,
    }


@router.get("/summary")
def get_dataset_summary():
    if registry.active_dataset_df is None:
        raise HTTPException(status_code=404, detail="No dataset loaded.")

    df = registry.active_dataset_df
    rows, cols = df.shape
    duplicate_rows = int(df.duplicated().sum())
    missing_by_col = {col: int(df[col].isna().sum()) for col in df.columns if df[col].isna().sum() > 0}
    total_missing_cells = int(df.isna().sum().sum())

    # Numerical Summary
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    numerical_summary = []
    for col in num_cols:
        s = df[col].dropna()
        if len(s) == 0:
            continue
        numerical_summary.append({
            "column": col,
            "count": int(len(s)),
            "mean": round(float(s.mean()), 3),
            "std": round(float(s.std()), 3) if len(s) > 1 else 0.0,
            "min": round(float(s.min()), 3),
            "median": round(float(s.median()), 3),
            "max": round(float(s.max()), 3),
        })

    # Categorical Summary
    cat_cols = [c for c in df.columns if c not in num_cols]
    categorical_summary = []
    for col in cat_cols:
        s = df[col].dropna().astype(str)
        top_val = s.mode().iloc[0] if len(s) > 0 else "N/A"
        top_freq = int((s == top_val).sum()) if len(s) > 0 else 0
        value_counts = [
            {"value": str(k), "count": int(v)}
            for k, v in s.value_counts().head(8).items()
        ]
        categorical_summary.append({
            "column": col,
            "unique": int(s.nunique()),
            "top": str(top_val),
            "top_frequency": top_freq,
            "distribution": value_counts,
        })

    # Churn Distribution
    churn_distribution = []
    if registry.active_target_column and registry.active_target_column in df.columns:
        bin_target = _normalize_target_series(df[registry.active_target_column])
        churned = int((bin_target == 1).sum())
        retained = int((bin_target == 0).sum())
        churn_distribution = [
            {"label": "Retained (Active)", "value": retained, "percentage": round(retained / rows * 100, 2) if rows else 0},
            {"label": "Churned", "value": churned, "percentage": round(churned / rows * 100, 2) if rows else 0},
        ]

    return {
        "dataset_name": registry.active_dataset_name,
        "rows": int(rows),
        "columns": int(cols),
        "duplicate_records": duplicate_rows,
        "total_missing_cells": total_missing_cells,
        "missing_by_column": missing_by_col,
        "target_column": registry.active_target_column,
        "churn_distribution": churn_distribution,
        "numerical_summary": numerical_summary,
        "categorical_summary": categorical_summary,
    }
