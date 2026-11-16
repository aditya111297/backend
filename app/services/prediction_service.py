from typing import Any, Optional
import numpy as np
import pandas as pd
from app.config import settings
from app.model_loader import registry


THRESHOLD_DISCLAIMER = (
    "Risk categories (Low, Medium, High) are configurable business decision thresholds "
    "based on predicted churn probability and do not represent automatically validated clinical or financial classifications."
)


def classify_risk(prob: float, low_th: Optional[float] = None, high_th: Optional[float] = None) -> tuple[str, dict[str, float]]:
    low = low_th if low_th is not None else settings.LOW_RISK_THRESHOLD
    high = high_th if high_th is not None else settings.HIGH_RISK_THRESHOLD
    if high <= low:
        high = min(0.99, low + 0.15)
    if prob >= high:
        cat = "High Risk"
    elif prob >= low:
        cat = "Medium Risk"
    else:
        cat = "Low Risk"
    return cat, {"low_threshold": round(low, 4), "high_threshold": round(high, 4)}


def generate_retention_recommendations(record: dict[str, Any], prob: float, risk_category: str) -> list[str]:
    """Generates associative retention action suggestions based on available customer attributes.
    Explicitly avoids claiming causal proof."""
    actions: list[str] = []
    lower_keys = {str(k).lower(): (k, v) for k, v in record.items() if v is not None}

    # Check common churn-correlated signals dynamically without hardcoding a single dataset
    for k_low, (orig_k, val) in lower_keys.items():
        val_str = str(val).lower()
        if "contract" in k_low and ("month" in val_str or "short" in val_str):
            actions.append(
                f"Offer an annual or multi-year contract upgrade incentive (observed '{orig_k}: {val}' is associated with higher churn segments)."
            )
        if "tenure" in k_low:
            try:
                if float(val) <= 12:
                    actions.append(
                        f"Enroll customer in early-lifecycle onboarding check-ins (observed '{orig_k}: {val}' falls in early tenure cohort)."
                    )
            except (ValueError, TypeError):
                pass
        if ("monthly" in k_low and "charge" in k_low) or ("fee" in k_low) or ("spend" in k_low):
            try:
                if float(val) >= 75:
                    actions.append(
                        f"Review billing tier and offer a loyalty bundle discount (observed elevated '{orig_k}: {val}')."
                    )
            except (ValueError, TypeError):
                pass
        if ("support" in k_low or "complaint" in k_low or "ticket" in k_low):
            try:
                if float(val) >= 2 or val_str in {"no", "none"}:
                    actions.append(
                        f"Schedule proactive customer success outreach regarding '{orig_k}' ({val})."
                    )
            except (ValueError, TypeError):
                pass

    if not actions:
        if risk_category == "High Risk":
            actions = [
                "Assign dedicated retention specialist for immediate outbound outreach within 24–48 hours.",
                "Present targeted loyalty discount or tailored plan optimization review.",
                "Audit recent service interactions and billing friction points.",
            ]
        elif risk_category == "Medium Risk":
            actions = [
                "Trigger automated engagement campaign highlighting underutilized product benefits.",
                "Offer flexible plan adjustment or periodic satisfaction check-in.",
            ]
        else:
            actions = [
                "Maintain standard lifecycle engagement and periodic value-summary communications.",
                "Consider eligibility for referral or loyalty advocacy programs.",
            ]

    return actions[:4]


def prepare_feature_dataframe(records: list[dict[str, Any]]) -> pd.DataFrame:
    """Aligns incoming records strictly to the exact training feature_columns order,
    safely filling missing columns with NaN so the saved Scikit-learn pipeline imputer handles them."""
    if registry.pipeline is None:
        raise RuntimeError(
            "ML pipeline is not loaded. Verify 'outputs/customer_churn_pipeline.joblib' is present."
        )

    expected_cols = registry.feature_columns
    if not expected_cols:
        raise RuntimeError("Model artifact does not specify 'feature_columns'.")

    raw_df = pd.DataFrame(records)
    aligned = pd.DataFrame(index=raw_df.index)

    for col in expected_cols:
        if col in raw_df.columns:
            series = raw_df[col]
            meta = registry.feature_metadata.get(col, {})
            if meta.get("type") == "number":
                series = pd.to_numeric(series, errors="coerce")
            aligned[col] = series
        else:
            aligned[col] = np.nan

    return aligned[expected_cols]


def predict_single(
    features: dict[str, Any],
    customer_id: Optional[str] = None,
    low_th: Optional[float] = None,
    high_th: Optional[float] = None,
) -> dict[str, Any]:
    df_features = prepare_feature_dataframe([features])
    pred_raw = registry.pipeline.predict(df_features)[0]

    if hasattr(registry.pipeline, "predict_proba"):
        proba_row = registry.pipeline.predict_proba(df_features)[0]
        churn_prob = float(proba_row[registry.positive_class_index])
    else:
        churn_prob = float(pred_raw)

    # Interpret prediction label using target_mapping
    mapped = registry.target_mapping.get(pred_raw, registry.target_mapping.get(str(pred_raw), pred_raw))
    is_churn = int(round(churn_prob >= 0.5)) if isinstance(pred_raw, (float, np.floating)) else (
        1 if str(mapped).lower() in {"1", "yes", "true", "churn", "churned"} or pred_raw == 1 else 0
    )
    pred_label = registry.positive_class_meaning if is_churn == 1 else registry.negative_class_meaning

    risk_cat, thresholds = classify_risk(churn_prob, low_th, high_th)
    actions = generate_retention_recommendations(features, churn_prob, risk_cat)

    return {
        "customer_id": customer_id,
        "prediction": is_churn,
        "prediction_label": pred_label,
        "churn_probability": round(churn_prob, 4),
        "retention_probability": round(1.0 - churn_prob, 4),
        "risk_category": risk_cat,
        "thresholds_used": thresholds,
        "threshold_disclaimer": THRESHOLD_DISCLAIMER,
        "recommended_actions": actions,
        "model_used": registry.best_model_name or "Exported Pipeline",
    }


def predict_batch_dataframe(
    df: pd.DataFrame,
    low_th: Optional[float] = None,
    high_th: Optional[float] = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    records = df.to_dict(orient="records")
    df_features = prepare_feature_dataframe(records)

    preds_raw = registry.pipeline.predict(df_features)
    if hasattr(registry.pipeline, "predict_proba"):
        probas = registry.pipeline.predict_proba(df_features)[:, registry.positive_class_index]
    else:
        probas = preds_raw.astype(float)

    result_df = df.copy()
    labels = []
    risk_cats = []
    rec_actions = []

    for idx, (raw_p, prob) in enumerate(zip(preds_raw, probas)):
        prob_f = float(prob)
        mapped = registry.target_mapping.get(raw_p, registry.target_mapping.get(str(raw_p), raw_p))
        is_churn = 1 if (str(mapped).lower() in {"1", "yes", "true", "churn", "churned"} or raw_p == 1) else 0
        labels.append(registry.positive_class_meaning if is_churn == 1 else registry.negative_class_meaning)
        r_cat, _ = classify_risk(prob_f, low_th, high_th)
        risk_cats.append(r_cat)
        rec_actions.append("; ".join(generate_retention_recommendations(records[idx], prob_f, r_cat)[:2]))

    result_df["Predicted_Churn"] = labels
    result_df["Churn_Probability"] = np.round(probas, 4)
    result_df["Risk_Category"] = risk_cats
    result_df["Recommended_Retention_Action"] = rec_actions

    total = len(result_df)
    churn_count = int((result_df["Predicted_Churn"] == registry.positive_class_meaning).sum())
    non_churn_count = total - churn_count
    high_risk_count = int((result_df["Risk_Category"] == "High Risk").sum())
    medium_risk_count = int((result_df["Risk_Category"] == "Medium Risk").sum())
    low_risk_count = int((result_df["Risk_Category"] == "Low Risk").sum())

    summary = {
        "total_processed": total,
        "predicted_churn": churn_count,
        "predicted_non_churn": non_churn_count,
        "churn_rate_percent": round((churn_count / total * 100), 2) if total > 0 else 0.0,
        "high_risk_customers": high_risk_count,
        "medium_risk_customers": medium_risk_count,
        "low_risk_customers": low_risk_count,
        "average_churn_probability": round(float(np.mean(probas)), 4) if total > 0 else 0.0,
        "threshold_disclaimer": THRESHOLD_DISCLAIMER,
    }

    return result_df, summary
