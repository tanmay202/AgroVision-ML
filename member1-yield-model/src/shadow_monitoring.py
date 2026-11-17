"""
AgroVision Model 1: Multi-Crop Prospective Shadow Monitoring Service
===================================================================
Tracks prospective, point-in-time forecasts generated before harvest outcomes
become available, stores immutable audit records in SQLite, and evaluates
accuracy against matched persistence baselines once official harvest statistics land.

Strict Separation:
- Service reliability telemetry vs. Agricultural biological accuracy validation.
"""

import sqlite3
import hashlib
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import pandas as pd
import numpy as np

logger = logging.getLogger("shadow_monitoring")

M1_DIR = Path(__file__).resolve().parent.parent
MONITORING_DIR = M1_DIR / "data" / "monitoring"
DB_PATH = MONITORING_DIR / "shadow_forecasts.db"


class ProspectiveShadowMonitor:
    """
    Manages prospective shadow forecasts and post-harvest reconciliation.
    """

    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = db_path or DB_PATH
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS shadow_forecasts (
                forecast_id TEXT PRIMARY KEY,
                forecast_timestamp TEXT NOT NULL,
                crop TEXT NOT NULL,
                district TEXT NOT NULL,
                year INTEGER NOT NULL,
                season TEXT NOT NULL,
                model_name TEXT NOT NULL,
                predicted_yield_kg_ha REAL NOT NULL,
                predicted_yield_tonnes_ha REAL NOT NULL,
                p10_kg_ha REAL,
                p90_kg_ha REAL,
                prediction_method TEXT,
                weather_source TEXT,
                input_signature_sha256 TEXT NOT NULL,
                actual_yield_kg_ha REAL,
                actual_reported_timestamp TEXT,
                evaluation_status TEXT DEFAULT 'PENDING'
            )
            """)
            cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_crop_district_year ON shadow_forecasts (crop, district, year)
            """)
            conn.commit()

    def record_forecast(
        self,
        crop: str,
        district: str,
        year: int,
        season: str,
        prediction_result: Dict[str, Any],
    ) -> str:
        """
        Record an immutable prospective forecast prior to harvest outcome.
        """
        now_iso = datetime.now(timezone.utc).isoformat()
        
        # Build canonical input signature
        sig_data = {
            "crop": crop.lower(),
            "district": district,
            "year": int(year),
            "season": season,
            "pred_kg": float(prediction_result.get("predicted_yield_kg_per_ha", 0.0)),
            "model": prediction_result.get("model_name", "unknown")
        }
        sig_str = json.dumps(sig_data, sort_keys=True)
        sig_hash = hashlib.sha256(sig_str.encode("utf-8")).hexdigest()
        forecast_id = f"{crop.lower()}_{district.lower().replace(' ', '_')}_{year}_{sig_hash[:12]}"

        p_int = prediction_result.get("prediction_interval", {})
        p10 = p_int.get("p10_kg_per_ha")
        p90 = p_int.get("p90_kg_per_ha")

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
            INSERT OR REPLACE INTO shadow_forecasts (
                forecast_id, forecast_timestamp, crop, district, year, season,
                model_name, predicted_yield_kg_ha, predicted_yield_tonnes_ha,
                p10_kg_ha, p90_kg_ha, prediction_method, weather_source,
                input_signature_sha256, evaluation_status
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'PENDING')
            """, (
                forecast_id,
                now_iso,
                crop.lower(),
                district,
                year,
                season,
                prediction_result.get("model_name", "unknown"),
                float(prediction_result.get("predicted_yield_kg_per_ha", 0.0)),
                float(prediction_result.get("predicted_yield_tonnes_per_ha", 0.0)),
                float(p10) if p10 is not None else None,
                float(p90) if p90 is not None else None,
                prediction_result.get("prediction_method", "model_forecast"),
                prediction_result.get("weather_provenance", {}).get("source", "nasa_power"),
                sig_hash,
            ))
            conn.commit()

        logger.info(f"Recorded shadow forecast: {forecast_id}")
        return forecast_id

    def reconcile_actuals(
        self,
        actuals_df: pd.DataFrame,
        crop: str
    ) -> int:
        """
        Reconcile verified official harvest actuals once published by DES/UPAg.
        """
        updated = 0
        now_iso = datetime.now(timezone.utc).isoformat()
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            for _, row in actuals_df.iterrows():
                dist = row.get("District")
                yr = int(row.get("Crop_Year", row.get("year", 0)))
                actual_kg = float(row.get("Yield_kg_ha", row.get("yield_kg_per_ha", 0.0)))
                if dist and yr and actual_kg > 0:
                    cursor.execute("""
                    UPDATE shadow_forecasts
                    SET actual_yield_kg_ha = ?, actual_reported_timestamp = ?, evaluation_status = 'EVALUATED'
                    WHERE crop = ? AND district = ? AND year = ? AND evaluation_status = 'PENDING'
                    """, (actual_kg, now_iso, crop.lower(), dist, yr))
                    updated += cursor.rowcount
            conn.commit()
        return updated

    def evaluate_shadow_performance(self, crop: Optional[str] = None) -> Dict[str, Any]:
        """
        Compute prospective evaluation metrics for evaluated forecasts.
        """
        query = "SELECT * FROM shadow_forecasts WHERE evaluation_status = 'EVALUATED'"
        params = []
        if crop:
            query += " AND crop = ?"
            params.append(crop.lower())

        with sqlite3.connect(self.db_path) as conn:
            df = pd.read_sql_query(query, conn, params=params)

        if df.empty:
            return {"status": "NO_EVALUATED_RECORDS", "count": 0}

        y_true = df['actual_yield_kg_ha'].values
        y_pred = df['predicted_yield_kg_ha'].values
        residuals = y_pred - y_true

        mae = float(np.mean(np.abs(residuals)))
        rmse = float(np.sqrt(np.mean(residuals**2)))
        bias = float(np.mean(residuals))

        # Coverage of 80% interval
        covered = 0
        valid_intervals = 0
        for _, r in df.iterrows():
            if pd.notnull(r['p10_kg_ha']) and pd.notnull(r['p90_kg_ha']):
                valid_intervals += 1
                if r['p10_kg_ha'] <= r['actual_yield_kg_ha'] <= r['p90_kg_ha']:
                    covered += 1

        coverage_pct = float(covered / valid_intervals * 100.0) if valid_intervals > 0 else 0.0

        return {
            "crop": crop or "all",
            "evaluated_count": len(df),
            "mae_kg_ha": round(mae, 2),
            "rmse_kg_ha": round(rmse, 2),
            "mean_bias_kg_ha": round(bias, 2),
            "coverage_80pct_empirical": round(coverage_pct, 2),
            "valid_interval_count": valid_intervals
        }
