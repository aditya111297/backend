import json
from pathlib import Path
from typing import Any, Optional
import joblib
import numpy as np
import pandas as pd
from app.config import settings


REQUIRED_ARTIFACTS = [
    "customer_churn_pipeline.joblib",
]

OPTIONAL_ARTIFACTS = [
    "model_comparison_results.csv",
    "feature_importance.csv",
    "customer_churn_dataset.csv",
    "best_churn_model.pkl",
    "preprocessor.pkl",
    "feature_metadata.json",
]


class ModelRegistry:
    """Loads exported ML pipeline and notebook evaluation artifacts once at startup.
    Never retrains or fits new preprocessing pipelines at runtime."""

    def __init__(self):
        self.outputs_dir: Path = settings.outputs_path
        self.pipeline: Any = None
        self.best_model_name: Optional[str] = None
        self.target_column: Optional[str] = None
        self.target_mapping: dict[Any, Any] = {}
        self.positive_class_index: int = 1
        self.positive_class_meaning: str = "Churn"
        self.negative_class_meaning: str = "No Churn"
        self.feature_columns: list[str] = []
        self.feature_metadata: dict[str, Any] = {}
        self.model_comparison_df: Optional[pd.DataFrame] = None
        self.feature_importance_df: Optional[pd.DataFrame] = None
        self.active_dataset_df: Optional[pd.DataFrame] = None
        self.active_dataset_name: Optional[str] = None
        self.active_target_column: Optional[str] = None
        self.artifacts_status: dict[str, bool] = {}
        self.missing_artifacts: list[str] = []
        self.errors: list[str] = []

    def load_all(self):
        self.errors.clear()
        self.missing_artifacts.clear()
        self.artifacts_status.clear()

        for fname in REQUIRED_ARTIFACTS + OPTIONAL_ARTIFACTS:
            exists = self._find_artifact(fname).is_file()
            self.artifacts_status[fname] = exists
            if not exists:
                self.missing_artifacts.append(fname)

        # 1. Load Main Joblib Pipeline Artifact
        pipeline_path = self._find_artifact("customer_churn_pipeline.joblib")
        if not pipeline_path.is_file():
            self.errors.append(
                f"Required artifact missing: '{self.outputs_dir / 'customer_churn_pipeline.joblib'}'. "
                "Export a trained pipeline from your notebook into the outputs folder."
            )
        else:
            try:
                bundle = joblib.load(pipeline_path)
                if not isinstance(bundle, dict):
                    self.errors.append(
                        "Invalid customer_churn_pipeline.joblib structure: expected a dictionary."
                    )
                else:
                    self.pipeline = bundle.get("model")
                    if self.pipeline is None:
                        self.pipeline = bundle.get("pipeline")
                    if self.pipeline is None:
                        self.errors.append(
                            "Invalid customer_churn_pipeline.joblib structure: expected key 'model' or 'pipeline'."
                        )
                    else:
                        self.best_model_name = bundle.get(
                            "best_model_name",
                            bundle.get("model_name", "Best Trained Model"),
                        )
                        self.target_column = bundle.get("target_column", "Churn")
                        self.target_mapping = bundle.get("target_mapping", {0: "No", 1: "Yes"})
                        self.feature_columns = list(bundle.get("feature_columns", []))
                        self.positive_class_meaning = str(
                            bundle.get("positive_class_meaning")
                            or bundle.get("positive_class")
                            or "Churn"
                        )
                        positive_encoded = self.target_mapping.get(
                            bundle.get("positive_class")
                        )
                        if positive_encoded is not None:
                            for label, encoded in self.target_mapping.items():
                                if label != bundle.get("positive_class") and encoded != positive_encoded:
                                    self.negative_class_meaning = str(label)
                                    break
                        self._resolve_class_indices()
            except Exception as exc:
                self.errors.append(f"Failed to load customer_churn_pipeline.joblib: {exc}")

        # 2. Load Model Comparison Results
        comp_path = self.outputs_dir / "model_comparison_results.csv"
        if comp_path.is_file():
            try:
                self.model_comparison_df = pd.read_csv(comp_path)
            except Exception as exc:
                self.errors.append(f"Failed to read model_comparison_results.csv: {exc}")

        # 3. Load Permutation Feature Importance
        fi_path = self.outputs_dir / "feature_importance.csv"
        if fi_path.is_file():
            try:
                self.feature_importance_df = pd.read_csv(fi_path)
            except Exception as exc:
                self.errors.append(f"Failed to read feature_importance.csv: {exc}")

        # 4. Load Feature Metadata (for dynamic UI form rendering)
        meta_path = self.outputs_dir / "feature_metadata.json"
        if meta_path.is_file():
            try:
                with open(meta_path, "r", encoding="utf-8") as f:
                    self.feature_metadata = json.load(f)
            except Exception as exc:
                self.errors.append(f"Failed to read feature_metadata.json: {exc}")

        # 5. Load Default Exported Dataset if available
        ds_path = self.outputs_dir / "customer_churn_dataset.csv"
        if ds_path.is_file():
            try:
                df = pd.read_csv(ds_path)
                self.set_active_dataset(df, "customer_churn_dataset.csv", self.target_column)
            except Exception as exc:
                self.errors.append(f"Failed to read customer_churn_dataset.csv: {exc}")

        # Infer missing feature metadata from active dataset or pipeline if JSON was omitted
        self._build_fallback_feature_metadata()

    def _find_artifact(self, filename: str) -> Path:
        """Find artifacts in outputs, with a root-level fallback for the main model."""
        output_path = self.outputs_dir / filename
        if output_path.is_file():
            return output_path
        if filename == "customer_churn_pipeline.joblib":
            root_path = self.outputs_dir.parent / filename
            if root_path.is_file():
                return root_path
        return output_path

    def _resolve_class_indices(self):
        """Determine which index in predict_proba corresponds to the positive churn class."""
        classes = getattr(self.pipeline, "classes_", None)
        if classes is not None and len(classes) == 2:
            classes_list = list(classes)
            # Check target_mapping for positive label
            pos_encoded = None
            for k, v in self.target_mapping.items():
                v_str = str(v).strip().lower()
                k_str = str(k).strip().lower()
                if v_str in {"1", "yes", "true", "churn", "churned", "positive"}:
                    pos_encoded = k
                elif k_str in {"1", "yes", "true", "churn", "churned", "positive"} and v in classes_list:
                    pos_encoded = v
            if pos_encoded in classes_list:
                self.positive_class_index = classes_list.index(pos_encoded)
            elif 1 in classes_list:
                self.positive_class_index = classes_list.index(1)
            else:
                self.positive_class_index = 1

    def set_active_dataset(self, df: pd.DataFrame, filename: str, target_col: Optional[str] = None):
        self.active_dataset_df = df.copy()
        self.active_dataset_name = filename
        self.active_target_column = self.detect_target_column(df, preferred=target_col or self.target_column)
        self._build_fallback_feature_metadata()

    def detect_target_column(self, df: pd.DataFrame, preferred: Optional[str] = None) -> Optional[str]:
        if preferred and preferred in df.columns:
            return preferred
        candidates = ["churn", "churned", "is_churn", "exited", "attrition", "target", "label"]
        lower_map = {c.lower().strip(): c for c in df.columns}
        for cand in candidates:
            if cand in lower_map:
                return lower_map[cand]
        return None

    def _build_fallback_feature_metadata(self):
        """Ensure every feature in self.feature_columns has schema metadata (type, options/bounds)
        for dynamic frontend input generation."""
        if not self.feature_columns and self.active_dataset_df is not None:
            self.feature_columns = [
                c for c in self.active_dataset_df.columns
                if c != self.active_target_column and "id" not in c.lower()
            ]

        existing = self.feature_metadata.get("features", self.feature_metadata) if isinstance(self.feature_metadata, dict) else {}
        pipeline_metadata = self._infer_pipeline_feature_metadata()
        enriched: dict[str, Any] = {}

        for col in self.feature_columns:
            if isinstance(existing, dict) and col in existing and isinstance(existing[col], dict):
                enriched[col] = existing[col]
                continue

            if col in pipeline_metadata:
                enriched[col] = pipeline_metadata[col]
                continue

            info: dict[str, Any] = {"name": col, "label": col.replace("_", " ").title()}
            if self.active_dataset_df is not None and col in self.active_dataset_df.columns:
                series = self.active_dataset_df[col].dropna()
                if pd.api.types.is_numeric_dtype(series) and series.nunique() > 10:
                    info["type"] = "number"
                    info["min"] = float(series.min()) if len(series) else 0.0
                    info["max"] = float(series.max()) if len(series) else 1000.0
                    info["default"] = round(float(series.median()), 2) if len(series) else 0.0
                    info["step"] = 0.01 if pd.api.types.is_float_dtype(series) else 1
                elif pd.api.types.is_numeric_dtype(series) and series.nunique() <= 10:
                    unique_vals = sorted(series.unique().tolist())
                    if set(unique_vals).issubset({0, 1}):
                        info["type"] = "select"
                        info["options"] = [0, 1]
                        info["default"] = int(series.mode().iloc[0]) if len(series) else 0
                    else:
                        info["type"] = "number"
                        info["min"] = float(min(unique_vals))
                        info["max"] = float(max(unique_vals))
                        info["default"] = float(series.median())
                        info["step"] = 1
                else:
                    opts = [str(v) for v in series.unique().tolist()[:30]]
                    info["type"] = "select" if len(opts) > 0 else "text"
                    info["options"] = opts
                    info["default"] = str(series.mode().iloc[0]) if len(series) else (opts[0] if opts else "")
            else:
                info["type"] = "text"
                info["default"] = ""
            enriched[col] = info

        self.feature_metadata = enriched

    def _infer_pipeline_feature_metadata(self) -> dict[str, dict[str, Any]]:
        """Infer input control types/defaults from a fitted ColumnTransformer."""
        named_steps = getattr(self.pipeline, "named_steps", {})
        preprocessor = named_steps.get("preprocessor")
        if preprocessor is None:
            return {}

        metadata: dict[str, dict[str, Any]] = {}
        for transformer_name, transformer, columns in getattr(preprocessor, "transformers_", []):
            if transformer_name == "remainder" or isinstance(columns, str):
                continue

            columns = list(columns)
            steps = getattr(transformer, "named_steps", {})
            encoder = steps.get("onehot")
            categories = getattr(encoder, "categories_", None)
            imputer = steps.get("imputer")
            statistics = getattr(imputer, "statistics_", None)

            if categories is not None:
                for col, values in zip(columns, categories):
                    options = [
                        value.item() if isinstance(value, np.generic) else value
                        for value in values
                    ]
                    metadata[col] = {
                        "name": col,
                        "label": col.replace("_", " ").title(),
                        "type": "select",
                        "options": options,
                        "default": options[0] if options else "",
                    }
            elif statistics is not None:
                for index, col in enumerate(columns):
                    default = statistics[index]
                    default = float(default) if pd.notna(default) else 0.0
                    metadata[col] = {
                        "name": col,
                        "label": col.replace("_", " ").title(),
                        "type": "number",
                        "default": round(default, 2),
                        "step": 1 if default.is_integer() else 0.01,
                    }

        return metadata


registry = ModelRegistry()
