from fastapi import APIRouter, HTTPException
import numpy as np
import pandas as pd
from app.model_loader import registry
from app.routers.dataset import _normalize_target_series
from app.services.prediction_service import classify_risk, prepare_feature_dataframe

router = APIRouter(prefix="/api", tags=["Dashboard"])


@router.get("/dashboard")
def get_dashboard_metrics():
    if registry.active_dataset_df is None:
        raise HTTPException(
            status_code=404,
            detail="No dataset loaded. Please upload a CSV dataset or place outputs/customer_churn_dataset.csv.",
        )

    df = registry.active_dataset_df
    total_customers = int(len(df))

    # Actual churn counts if target column exists in dataset
    if registry.active_target_column and registry.active_target_column in df.columns:
        target_bin = _normalize_target_series(df[registry.active_target_column])
        churned_customers = int((target_bin == 1).sum())
        active_customers = total_customers - churned_customers
    else:
        target_bin = None
        churned_customers = 0
        active_customers = total_customers

    churn_percentage = round((churned_customers / total_customers * 100), 2) if total_customers > 0 else 0.0

    # Average customer tenure (dynamically detect tenure-like column)
    tenure_col = next((c for c in df.columns if "tenure" in c.lower()), None)
    avg_tenure = None
    if tenure_col and pd.api.types.is_numeric_dtype(df[tenure_col]):
        avg_tenure = round(float(df[tenure_col].dropna().mean()), 2)

    # High-risk customers via loaded ML pipeline
    high_risk_count = 0
    risk_counts = {"Low Risk": 0, "Medium Risk": 0, "High Risk": 0}
    if registry.pipeline is not None and total_customers > 0:
        try:
            X_aligned = prepare_feature_dataframe(df.to_dict(orient="records"))
            if hasattr(registry.pipeline, "predict_proba"):
                probs = registry.pipeline.predict_proba(X_aligned)[:, registry.positive_class_index]
            else:
                probs = registry.pipeline.predict(X_aligned).astype(float)
            for p in probs:
                cat, _ = classify_risk(float(p))
                risk_counts[cat] += 1
            high_risk_count = risk_counts["High Risk"]
        except Exception:
            pass

    # Tenure-based churn analysis (if tenure & target supported)
    tenure_analysis = []
    if tenure_col and pd.api.types.is_numeric_dtype(df[tenure_col]) and target_bin is not None:
        temp = pd.DataFrame({"tenure": df[tenure_col], "churn": target_bin}).dropna()
        if len(temp) > 0:
            bins = [0, 6, 12, 24, 48, 72, float("inf")]
            labels = ["0-6m", "7-12m", "13-24m", "25-48m", "49-72m", "72m+"]
            temp["cohort"] = pd.cut(temp["tenure"], bins=bins, labels=labels, include_lowest=True)
            grouped = temp.groupby("cohort", observed=True)["churn"].agg(["count", "sum"]).reset_index()
            for _, row in grouped.iterrows():
                cnt = int(row["count"])
                ch = int(row["sum"])
                tenure_analysis.append({
                    "cohort": str(row["cohort"]),
                    "total": cnt,
                    "churned": ch,
                    "retained": cnt - ch,
                    "churn_rate": round((ch / cnt * 100), 2) if cnt > 0 else 0.0,
                })

    # Customer segmentation (pick best categorical column with 2-6 unique values)
    segmentation_col = None
    for preferred_name in ["contract", "segment", "geography", "subscription", "plan", "internetservice", "gender"]:
        match = next((c for c in df.columns if preferred_name in c.lower() and c != registry.active_target_column), None)
        if match and 2 <= df[match].nunique(dropna=True) <= 8:
            segmentation_col = match
            break
    if not segmentation_col:
        for c in df.columns:
            if c != registry.active_target_column and not pd.api.types.is_numeric_dtype(df[c]) and 2 <= df[c].nunique(dropna=True) <= 6:
                segmentation_col = c
                break

    segmentation_data = []
    if segmentation_col:
        temp_seg = pd.DataFrame({"segment": df[segmentation_col].fillna("Unknown").astype(str)})
        temp_seg["churn"] = target_bin if target_bin is not None else 0
        grp = temp_seg.groupby("segment")["churn"].agg(["count", "sum"]).reset_index()
        for _, row in grp.iterrows():
            cnt = int(row["count"])
            ch = int(row["sum"])
            segmentation_data.append({
                "segment": str(row["segment"]),
                "total": cnt,
                "churned": ch,
                "retained": cnt - ch,
                "churn_rate": round((ch / cnt * 100), 2) if cnt > 0 else 0.0,
            })

    # Model performance summary from actual CSV artifact
    model_comparison = []
    if registry.model_comparison_df is not None:
        model_comparison = registry.model_comparison_df.replace({np.nan: None}).to_dict(orient="records")

    return {
        "dataset_name": registry.active_dataset_name,
        "is_synthetic_demo": bool(
            "Demo_Data" in df.columns
            and df["Demo_Data"].astype(str).str.contains("SYNTHETIC DEMO", case=False).any()
        ),
        "total_customers": total_customers,
        "churned_customers": churned_customers,
        "active_customers": active_customers,
        "churn_percentage": churn_percentage,
        "average_customer_tenure": avg_tenure,
        "tenure_column": tenure_col,
        "high_risk_customers": high_risk_count,
        "best_performing_model": registry.best_model_name,
        "churn_distribution": [
            {"name": "Active (Retained)", "value": active_customers},
            {"name": "Churned", "value": churned_customers},
        ],
        "risk_distribution": [
            {"name": "Low Risk", "value": risk_counts["Low Risk"]},
            {"name": "Medium Risk", "value": risk_counts["Medium Risk"]},
            {"name": "High Risk", "value": risk_counts["High Risk"]},
        ],
        "tenure_analysis": tenure_analysis,
        "segmentation_column": segmentation_col,
        "customer_segmentation": segmentation_data,
        "model_comparison": model_comparison,
    }
