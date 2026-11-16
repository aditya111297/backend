from fastapi import APIRouter, HTTPException
import numpy as np
from app.model_loader import registry

router = APIRouter(prefix="/api", tags=["Models & Feature Importance"])


@router.get("/models/performance")
def get_models_performance():
    if registry.model_comparison_df is None:
        raise HTTPException(
            status_code=404,
            detail=(
                "No model evaluation results are loaded. Run the evaluation and export steps "
                "in your training notebook, then place the real "
                "'model_comparison_results.csv' file in outputs/. "
                "Dashboard and prediction features do not require this optional file."
            ),
        )

    df = registry.model_comparison_df.copy().replace({np.nan: None})
    records = df.to_dict(orient="records")

    return {
        "best_model_name": registry.best_model_name,
        "models": records,
    }


@router.get("/feature-importance")
def get_feature_importance(top_n: int = 15):
    if registry.feature_importance_df is not None:
        df = registry.feature_importance_df.copy().replace({np.nan: None})
        cols = list(df.columns)
        feat_col = next((c for c in cols if "feat" in c.lower() or "name" in c.lower()), cols[0])
        imp_col = next((c for c in cols if "import" in c.lower() or "score" in c.lower() or "mean" in c.lower()), cols[1] if len(cols) > 1 else cols[0])
        std_col = next((c for c in cols if "std" in c.lower()), None)
        df = df.sort_values(by=imp_col, ascending=False)
        items = [
            {
                "rank": rank,
                "feature": str(row[feat_col]),
                "importance": round(float(row[imp_col]), 6) if row[imp_col] is not None else 0.0,
                "std": round(float(row[std_col]), 6) if (std_col and row[std_col] is not None) else None,
            }
            for rank, (_, row) in enumerate(df.iterrows(), start=1)
        ]
        importance_method = "permutation"
        importance_note = "Permutation importance results exported from the training notebook."
    else:
        named_steps = getattr(registry.pipeline, "named_steps", {})
        estimator = named_steps.get("classifier")
        preprocessor = named_steps.get("preprocessor")
        values = getattr(estimator, "feature_importances_", None)
        if values is None or preprocessor is None:
            raise HTTPException(
                status_code=404,
                detail=(
                    "No feature importance export is available, and the loaded model "
                    "does not provide native feature importances."
                ),
            )

        transformed_features = preprocessor.get_feature_names_out()
        if len(values) != len(transformed_features):
            raise HTTPException(
                status_code=500,
                detail="Saved model feature importances do not match its preprocessor output.",
            )

        grouped_importance = {feature: 0.0 for feature in registry.feature_columns}
        source_features = sorted(registry.feature_columns, key=len, reverse=True)
        for transformed_name, value in zip(transformed_features, values):
            output_name = str(transformed_name).split("__", 1)[-1]
            source_feature = next(
                (
                    feature
                    for feature in source_features
                    if output_name == feature or output_name.startswith(f"{feature}_")
                ),
                None,
            )
            if source_feature is not None:
                grouped_importance[source_feature] += float(value)

        items = [
            {
                "rank": rank,
                "feature": feature,
                "importance": round(value, 6),
                "std": None,
            }
            for rank, (feature, value) in enumerate(
                sorted(grouped_importance.items(), key=lambda item: item[1], reverse=True),
                start=1,
            )
        ]
        importance_method = "native"
        importance_note = (
            "Grouped impurity-based importances from the saved model; "
            "these are not permutation importance scores."
        )

    return {
        "model_name": registry.best_model_name,
        "importance_method": importance_method,
        "importance_note": importance_note,
        "total_features": len(items),
        "top_10": items[:10],
        "top_15": items[:15],
        "all_features": items[:top_n] if top_n > 0 else items,
    }
