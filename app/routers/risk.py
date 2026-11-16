from typing import Optional
from fastapi import APIRouter, HTTPException, Query
import numpy as np
from app.model_loader import registry
from app.services.prediction_service import (
    THRESHOLD_DISCLAIMER,
    classify_risk,
    generate_retention_recommendations,
    prepare_feature_dataframe,
)

router = APIRouter(prefix="/api", tags=["Risk Analysis"])


@router.get("/risk-analysis")
def get_risk_analysis(
    risk_filter: Optional[str] = Query(default=None),
    low_threshold: Optional[float] = Query(default=None, ge=0.01, le=0.98),
    high_threshold: Optional[float] = Query(default=None, ge=0.02, le=0.99),
    limit: int = Query(default=100, ge=1, le=500),
):
    if registry.active_dataset_df is None:
        raise HTTPException(status_code=404, detail="No dataset loaded.")
    if registry.pipeline is None:
        raise HTTPException(status_code=503, detail="ML pipeline not loaded.")

    df = registry.active_dataset_df.copy()
    records = df.to_dict(orient="records")
    X_aligned = prepare_feature_dataframe(records)

    if hasattr(registry.pipeline, "predict_proba"):
        probs = registry.pipeline.predict_proba(X_aligned)[:, registry.positive_class_index]
    else:
        probs = registry.pipeline.predict(X_aligned).astype(float)

    id_col = next((c for c in df.columns if "id" in c.lower()), None)

    # Probability histogram bins
    hist_counts, bin_edges = np.histogram(probs, bins=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    prob_distribution = [
        {
            "range": f"{int(bin_edges[i]*100)}-{int(bin_edges[i+1]*100)}%",
            "count": int(hist_counts[i]),
        }
        for i in range(len(hist_counts))
    ]

    counts = {"High Risk": 0, "Medium Risk": 0, "Low Risk": 0}
    enriched_customers = []
    thresholds_used = {}

    for idx, prob in enumerate(probs):
        p_val = float(prob)
        cat, thresholds_used = classify_risk(p_val, low_threshold, high_threshold)
        counts[cat] += 1

        if risk_filter and risk_filter.lower() != "all" and cat.lower() != risk_filter.lower():
            continue

        rec = records[idx]
        cust_id = str(rec[id_col]) if id_col and rec.get(id_col) is not None else f"ROW-{idx + 1}"
        actions = generate_retention_recommendations(rec, p_val, cat)

        # Pick 3 key display attributes from record
        sample_attrs = {
            k: v for k, v in list(rec.items())[:5]
            if k != id_col and k != registry.active_target_column and v is not None
        }

        enriched_customers.append({
            "customer_id": cust_id,
            "churn_probability": round(p_val, 4),
            "risk_category": cat,
            "key_attributes": sample_attrs,
            "recommendations": actions,
        })

    enriched_customers.sort(key=lambda x: x["churn_probability"], reverse=True)

    return {
        "dataset_name": registry.active_dataset_name,
        "total_customers": len(probs),
        "thresholds_used": thresholds_used,
        "threshold_disclaimer": THRESHOLD_DISCLAIMER,
        "causation_note": "Retention recommendations reflect statistical associations observed in historical customer features and do not establish causal proof.",
        "summary": {
            "high_risk": counts["High Risk"],
            "medium_risk": counts["Medium Risk"],
            "low_risk": counts["Low Risk"],
        },
        "risk_segmentation": [
            {"category": "High Risk", "count": counts["High Risk"]},
            {"category": "Medium Risk", "count": counts["Medium Risk"]},
            {"category": "Low Risk", "count": counts["Low Risk"]},
        ],
        "probability_distribution": prob_distribution,
        "customers": enriched_customers[:limit],
    }
