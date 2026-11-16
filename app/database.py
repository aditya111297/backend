import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional
from app.config import settings


class DatabaseManager:
    """Optional SQLite persistence for prediction history, dataset metadata, and activity logs.
    All methods degrade gracefully if the database is disabled or unavailable."""

    def __init__(self):
        self.enabled = settings.ENABLE_DB
        db_url = settings.DATABASE_URL
        if db_url.startswith("sqlite:///"):
            self.db_path = Path(db_url.replace("sqlite:///", "", 1))
        else:
            self.db_path = Path("churn_analytics.db")
        if self.enabled:
            try:
                self.db_path.parent.mkdir(parents=True, exist_ok=True)
                self._init_tables()
            except Exception:
                self.enabled = False

    def _get_conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        return conn

    def _init_tables(self):
        with self._get_conn() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS dataset_uploads (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    filename TEXT NOT NULL,
                    rows INTEGER NOT NULL,
                    columns INTEGER NOT NULL,
                    target_column TEXT,
                    uploaded_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS prediction_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    prediction_type TEXT NOT NULL,
                    customer_id TEXT,
                    input_summary TEXT,
                    prediction_label TEXT NOT NULL,
                    churn_probability REAL NOT NULL,
                    risk_category TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS activity_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    action TEXT NOT NULL,
                    details TEXT,
                    created_at TEXT NOT NULL
                );
                """
            )

    def log_dataset_upload(self, filename: str, rows: int, columns: int, target_column: Optional[str]):
        if not self.enabled:
            return
        try:
            with self._get_conn() as conn:
                conn.execute(
                    "INSERT INTO dataset_uploads (filename, rows, columns, target_column, uploaded_at) VALUES (?, ?, ?, ?, ?)",
                    (filename, rows, columns, target_column, datetime.now(timezone.utc).isoformat()),
                )
        except Exception:
            pass

    def log_prediction(
        self,
        prediction_type: str,
        customer_id: Optional[str],
        input_data: dict[str, Any],
        prediction_label: str,
        churn_probability: float,
        risk_category: str,
    ):
        if not self.enabled:
            return
        try:
            with self._get_conn() as conn:
                conn.execute(
                    """INSERT INTO prediction_history
                    (prediction_type, customer_id, input_summary, prediction_label, churn_probability, risk_category, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?)""",
                    (
                        prediction_type,
                        customer_id,
                        json.dumps(input_data, default=str)[:2000],
                        prediction_label,
                        float(churn_probability),
                        risk_category,
                        datetime.now(timezone.utc).isoformat(),
                    ),
                )
        except Exception:
            pass

    def log_activity(self, action: str, details: str = ""):
        if not self.enabled:
            return
        try:
            with self._get_conn() as conn:
                conn.execute(
                    "INSERT INTO activity_logs (action, details, created_at) VALUES (?, ?, ?)",
                    (action, details, datetime.now(timezone.utc).isoformat()),
                )
        except Exception:
            pass

    def get_recent_predictions(self, limit: int = 25) -> list[dict[str, Any]]:
        if not self.enabled:
            return []
        try:
            with self._get_conn() as conn:
                rows = conn.execute(
                    "SELECT * FROM prediction_history ORDER BY id DESC LIMIT ?", (limit,)
                ).fetchall()
                return [dict(r) for r in rows]
        except Exception:
            return []


db = DatabaseManager()
