"""
AgroVision Multi-Source, Multi-District & Multi-Crop Pipeline Adapter
Extracts authentic mandi history + ASOF district weather from TimescaleDB across
Hooghly, Purba Bardhaman, Bankura, and Murshidabad for Rice, Potato, and Wheat,
runs M2 multi-horizon price & volatility models, fuses M1 crop yield and the
Composite Weather+Storage Risk Engine, and atomically persists results.
"""

import json
import logging
import os
import re
import sys
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import pandas as pd
import psycopg
from psycopg.rows import dict_row

from pipeline.risk_engine import (
    compute_composite_risk,
    fetch_latest_district_weather,
    get_district_crop_yield,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("M2Adapter")

LOCAL_ROOT = Path(__file__).resolve().parent.parent
SIBLING_ROOT = Path.home() / "projects/agrovision-ml"
REPO_ROOT = LOCAL_ROOT if (LOCAL_ROOT / "deliverables/member2-ml").exists() else SIBLING_ROOT
PKG_DIR = REPO_ROOT / "deliverables/member2-ml"
if str(PKG_DIR) not in sys.path:
    sys.path.insert(0, str(PKG_DIR))

from src.predictor import InsufficientHistoryError, MultiCommodityPricePredictor

ARTIFACTS_DIR = PKG_DIR / "artifacts"

DB_CONFIG = {
    "host": os.getenv("PGHOST", "127.0.0.1"),
    "port": int(os.getenv("PGPORT", "5432")),
    "dbname": os.getenv("PGDATABASE", "agrovision"),
    "user": os.getenv("PGUSER", "agrovision"),
    "password": os.getenv("PGPASSWORD", "postgres"),
}

SPECIAL_MARKET_NAMES: dict[str, str] = {
    "WB_MND_KALIPUR": "Kalipur",
    "WB_MND_CHAMPADANGA": "Champadanga",
    "WB_MND_SHEORAPHULY": "Sheoraphuly",
    "WB_MND_PANDUA": "Pandua",
    "WB_MND_BISHNUPUR_BANKURA": "Bishnupur(Bankura)",
    "WB_MND_INDUS_BANKURA_SADAR": "Indus(Bankura Sadar)",
    "WB_MND_BANKURA_SADAR": "Bankura Sadar",
}

SPECIAL_VARIETY_NAMES: dict[str, str] = {
    "RICE_COMMON": "Common",
    "RICE_FINE": "Fine",
    "RICE_OTHER": "Other",
    "RICE_SONA_MANSOORI": "Sona Mansoori Non Basmati",
    "RICE_MASURI": "Masuri",
    "RICE_MASOORI": "Masoori-Non Basmti",
    "RICE_RATNACHUDI": "Ratnachudi (718 5-749)",
    "RICE_JAYA": "Jaya",
    "POTATO_JYOTI": "Jyoti",
    "POTATO_CHANDRAMUKHI": "Chandermukhi",
    "POTATO_LOCAL": "Local",
    "POTATO_OTHER": "Other",
    "WHEAT_SONALIKA": "Sonalika",
    "WHEAT_KALYAN": "Kalyan",
    "WHEAT_DESHI": "Deshi",
    "WHEAT_OTHER": "Other",
}


def resolve_series_metadata(
    mandi_id: str,
    crop_id: str,
    mandi_name_db: str,
    canonical_crop_db: str,
    district_id: str,
    district_name_db: str,
) -> dict[str, str]:
    cid_up = crop_id.upper()
    if cid_up.startswith("POTATO"):
        commodity = "potato"
        group = "Vegetables"
    elif cid_up.startswith("WHEAT"):
        commodity = "wheat"
        group = "Cereals"
    else:
        commodity = "rice"
        group = "Cereals"

    if mandi_id in SPECIAL_MARKET_NAMES:
        market_clean = SPECIAL_MARKET_NAMES[mandi_id]
    else:
        market_clean = re.sub(r"\s*APMC\s*$", "", mandi_name_db, flags=re.IGNORECASE).strip()

    if crop_id in SPECIAL_VARIETY_NAMES:
        variety_clean = SPECIAL_VARIETY_NAMES[crop_id]
    else:
        m = re.search(r"\(([^)]+)\)", canonical_crop_db)
        variety_clean = m.group(1).strip() if m else canonical_crop_db

    dist_csv = "Burdwan" if "bardhaman" in district_name_db.lower() else district_name_db

    return {
        "State Name": "West Bengal",
        "District Name": dist_csv,
        "District Label": district_name_db,
        "District ID": district_id,
        "Market Name": market_clean,
        "Variety": variety_clean,
        "Group": group,
        "commodity": commodity,
    }


def lookup_series_meta_from_db(mandi_id: str, crop_id: str) -> dict[str, str]:
    sql = """
    SELECT md.mandi_name, cd.canonical_name, ld.district_id, ld.district_name
    FROM features.mandi_dim md
    JOIN features.location_dim ld ON ld.district_id = md.district_id
    CROSS JOIN features.crop_dim cd
    WHERE md.mandi_id = %s AND cd.crop_id = %s;
    """
    with psycopg.connect(**DB_CONFIG, row_factory=dict_row) as conn, conn.cursor() as cur:
        cur.execute(sql, (mandi_id, crop_id))
        row = cur.fetchone()
    if not row:
        raise KeyError(f"Unknown (mandi_id, crop_id) pair: ({mandi_id}, {crop_id})")
    return resolve_series_metadata(
        mandi_id=mandi_id,
        crop_id=crop_id,
        mandi_name_db=row["mandi_name"],
        canonical_crop_db=row["canonical_name"],
        district_id=row["district_id"],
        district_name_db=row["district_name"],
    )


class M2PipelineAdapter:
    def __init__(self):
        logger.info("Initializing MultiCommodityPricePredictor from artifacts: %s", ARTIFACTS_DIR)
        self.predictor = MultiCommodityPricePredictor(artifacts_dir=ARTIFACTS_DIR)
        self.rice_residuals = self._load_rice_residuals()

    def _load_rice_residuals(self) -> dict[str, dict[str, Any]]:
        residual_files = {
            "short": ARTIFACTS_DIR / "rice_holdout_residuals.json",
            "14d": ARTIFACTS_DIR / "rice_holdout_residuals_14d.json",
            "30d": ARTIFACTS_DIR / "rice_holdout_residuals_30d.json",
        }
        residuals_data = {}
        for horizon, path in residual_files.items():
            if path.exists():
                with open(path, "r") as f:
                    residuals_data[horizon] = json.load(f)
            else:
                residuals_data[horizon] = {
                    "error_quantiles": {"p05": -50.0, "p95": 50.0},
                    "metrics": {},
                }
        return residuals_data

    def _get_storage_index(self, district_id: str) -> float:
        with psycopg.connect(**DB_CONFIG) as conn, conn.cursor() as cur:
            cur.execute(
                "SELECT storage_avail_index FROM features.location_dim WHERE district_id = %s;",
                (district_id,),
            )
            row = cur.fetchone()
            return float(row[0]) if row else 0.72

    def fetch_history_df(
        self, mandi_id: str, crop_id: str, origin_date: date, meta: dict[str, str]
    ) -> pd.DataFrame:
        sql = """
        SELECT (time AT TIME ZONE 'Asia/Kolkata')::date AS reported_date,
               modal_price_inr,
               arrivals_mt
        FROM features.market_features
        WHERE mandi_id = %s
          AND crop_id = %s
          AND (time AT TIME ZONE 'Asia/Kolkata')::date <= %s
        ORDER BY time ASC;
        """
        with psycopg.connect(**DB_CONFIG, row_factory=dict_row) as conn, conn.cursor() as cur:
            cur.execute(sql, (mandi_id, crop_id, origin_date))
            rows = cur.fetchall()

        if not rows:
            return pd.DataFrame()

        raw_df = pd.DataFrame(rows).drop_duplicates(subset=["reported_date"], keep="last").reset_index(drop=True)
        prices = raw_df["modal_price_inr"].astype(float)
        return pd.DataFrame({
            "State Name": meta["State Name"],
            "District Name": meta["District Name"],
            "Market Name": meta["Market Name"],
            "Variety": meta["Variety"],
            "Group": meta["Group"],
            "Reported Date": pd.to_datetime(raw_df["reported_date"]).dt.strftime("%Y-%m-%d"),
            "Modal Price (Rs./Quintal)": prices,
            "Min Price (Rs./Quintal)": prices,
            "Max Price (Rs./Quintal)": prices,
            "Arrivals (Tonnes)": raw_df["arrivals_mt"].astype(float),
        })

    def run_forecast_for_series(
        self,
        mandi_id: str,
        crop_id: str,
        origin_date: date,
        meta: dict[str, str],
    ) -> list[dict[str, Any]]:
        df_history = self.fetch_history_df(mandi_id, crop_id, origin_date, meta)
        if len(df_history) < 30:
            raise InsufficientHistoryError(
                f"Series ({mandi_id}, {crop_id}) has only {len(df_history)} observations; >=30 required."
            )

        comm = meta["commodity"]
        market_str = meta["Market Name"]
        variety_str = meta["Variety"]
        district_id = meta["District ID"]
        district_label = meta["District Label"]
        current_modal_price = float(df_history.iloc[-1]["Modal Price (Rs./Quintal)"])

        weather = fetch_latest_district_weather(district_id, origin_date)
        storage_idx = self._get_storage_index(district_id)

        forecasts: list[dict[str, Any]] = []
        horizon_specs = [
            ("short", "short", 1, False),
            ("14d", 14, 14, False),
            ("30d", 30, 30, True),
        ]

        p_low, p_med, p_high = 0.70, 0.20, 0.10
        vol_class_short = "LOW"

        for h_label, h_arg, h_days, is_exp in horizon_specs:
            kwargs: dict[str, Any] = {}
            if comm == "rice":
                kwargs = {
                    "market": market_str,
                    "variety": variety_str,
                    "prediction_date": str(origin_date),
                }
            res = self.predictor.predict(
                history=df_history,
                commodity=comm,
                horizon=h_arg,
                **kwargs,
            )

            pred_price = float(res["predicted_modal_price"])
            pred_change = float(res.get("predicted_price_change", round(pred_price - current_modal_price, 2)))

            pi = res.get("prediction_interval") or {}
            if "p05" in pi and "p95" in pi:
                lower = round(float(pi["p05"]), 2)
                upper = round(float(pi["p95"]), 2)
            elif "p10" in pi and "p90" in pi:
                lower = round(float(pi["p10"]), 2)
                upper = round(float(pi["p90"]), 2)
            elif comm == "rice":
                q_res = self.rice_residuals[h_label].get("error_quantiles", {})
                lower = round(pred_price + float(q_res.get("p05", -40.0)), 2)
                upper = round(pred_price + float(q_res.get("p95", 40.0)), 2)
            else:
                spread = max(current_modal_price * (0.025 * (h_days ** 0.35)), 15.0)
                lower = round(pred_price - spread, 2)
                upper = round(pred_price + spread, 2)

            if comm == "rice":
                m_res = self.rice_residuals[h_label].get("metrics", {})
                naive_mae = float(m_res.get("naive_baseline_mae", 19.43 * (h_days ** 0.45)))
            else:
                naive_mae = round(current_modal_price * (0.015 * (h_days ** 0.4)), 2)

            if h_label == "short":
                vol_probs = (
                    res.get("volatility_probabilities")
                    or res.get("normalized_volatility_probabilities")
                    or {}
                )
                p_low = float(vol_probs.get("LOW", 0.70))
                p_med = float(vol_probs.get("MEDIUM", 0.20))
                p_high = float(vol_probs.get("HIGH", 0.10))
                vol_class_short = str(
                    res.get("predicted_volatility_class")
                    or res.get("normalized_volatility_class")
                    or "LOW"
                )

            model_ver = str(res.get("model_name") or f"{comm}-{h_label}-v2")

            forecasts.append(
                {
                    "horizon_label": h_label,
                    "target_date": origin_date + timedelta(days=h_days),
                    "current_modal_price": current_modal_price,
                    "predicted_modal_price": round(pred_price, 2),
                    "predicted_price_change": round(pred_change, 2),
                    "predicted_price_lower": lower,
                    "predicted_price_upper": upper,
                    "naive_baseline_price": current_modal_price,
                    "naive_baseline_mae": naive_mae,
                    "volatility_class": vol_class_short if h_label == "short" else None,
                    "volatility_prob_low": p_low if h_label == "short" else None,
                    "volatility_prob_medium": p_med if h_label == "short" else None,
                    "volatility_prob_high": p_high if h_label == "short" else None,
                    "model_version": model_ver,
                    "is_experimental": is_exp,
                    "risk": compute_composite_risk(
                        current_modal_price=current_modal_price,
                        predicted_modal_price=pred_price,
                        predicted_price_change=pred_change,
                        predicted_price_lower=lower,
                        horizon_label=h_label,
                        vol_prob_low=p_low,
                        vol_prob_medium=p_med,
                        vol_prob_high=p_high,
                        forecast_rain_48h_mm=weather["forecast_rain_48h_mm"],
                        trailing_rain_3d_mm=weather["trailing_rain_3d_mm"],
                        rh_mean_pct=weather["rh_mean_pct"],
                        storage_avail_index=storage_idx,
                        district_name=district_label,
                    ),
                }
            )

        return forecasts

    def persist_pipeline_run(
        self,
        crop_id: str,
        mandi_id: str,
        district_id: str,
        district_label: str,
        origin_date: date,
        forecasts: list[dict[str, Any]],
        records_count: int,
    ) -> str:
        yield_forecast = get_district_crop_yield(district_label, crop_id)
        insert_run_sql = """
        INSERT INTO serving.pipeline_runs (crop_id, status, records_ingested, predictions_generated)
        VALUES (%s, 'COMPLETED', %s, %s)
        RETURNING run_id;
        """
        insert_forecast_sql = """
        INSERT INTO serving.price_forecasts (
            run_id, mandi_id, crop_id, horizon_label,
            forecast_origin_date, target_date, current_modal_price,
            predicted_modal_price, predicted_price_change,
            predicted_price_lower, predicted_price_upper,
            naive_baseline_price, naive_baseline_mae,
            volatility_class, volatility_prob_low, volatility_prob_medium, volatility_prob_high,
            model_version, is_experimental
        ) VALUES (
            %s, %s, %s, %s,
            %s, %s, %s,
            %s, %s,
            %s, %s,
            %s, %s,
            %s, %s, %s, %s,
            %s, %s
        );
        """
        insert_risk_sql = """
        INSERT INTO serving.risk_assessments (
            run_id, mandi_id, crop_id, district_id, assessed_origin_date, horizon_label,
            v_norm, h_anomaly, storage_vuln, composite_risk_score, risk_regime,
            forecast_rain_48h_mm, trailing_rain_3d_mm, rh_mean_pct, storage_avail_index,
            advisory_action, advisory_headline, advisory_reason, weather_adjusted_lower,
            predicted_yield_kg_ha, yield_method
        ) VALUES (
            %s, %s, %s, %s, %s, %s,
            %s, %s, %s, %s, %s,
            %s, %s, %s, %s,
            %s, %s, %s, %s,
            %s, %s
        );
        """

        with psycopg.connect(**DB_CONFIG) as conn, conn.cursor() as cur:
            cur.execute(insert_run_sql, (crop_id, records_count, len(forecasts)))
            run_id = cur.fetchone()[0]

            forecast_tuples = [
                (
                    run_id,
                    mandi_id,
                    crop_id,
                    f["horizon_label"],
                    origin_date,
                    f["target_date"],
                    f["current_modal_price"],
                    f["predicted_modal_price"],
                    f["predicted_price_change"],
                    f["predicted_price_lower"],
                    f["predicted_price_upper"],
                    f["naive_baseline_price"],
                    f["naive_baseline_mae"],
                    f["volatility_class"],
                    f["volatility_prob_low"],
                    f["volatility_prob_medium"],
                    f["volatility_prob_high"],
                    f["model_version"],
                    f["is_experimental"],
                )
                for f in forecasts
            ]
            cur.executemany(insert_forecast_sql, forecast_tuples)

            risk_tuples = [
                (
                    run_id,
                    mandi_id,
                    crop_id,
                    district_id,
                    origin_date,
                    f["horizon_label"],
                    f["risk"]["v_norm"],
                    f["risk"]["h_anomaly"],
                    f["risk"]["storage_vuln"],
                    f["risk"]["composite_risk_score"],
                    f["risk"]["risk_regime"],
                    f["risk"]["forecast_rain_48h_mm"],
                    f["risk"]["trailing_rain_3d_mm"],
                    f["risk"]["rh_mean_pct"],
                    f["risk"]["storage_avail_index"],
                    f["risk"]["advisory_action"],
                    f["risk"]["advisory_headline"],
                    f["risk"]["advisory_reason"],
                    f["risk"]["weather_adjusted_lower"],
                    yield_forecast["predicted_yield_kg_ha"],
                    yield_forecast["yield_method"],
                )
                for f in forecasts
            ]
            cur.executemany(insert_risk_sql, risk_tuples)

        return str(run_id)

    def run_all_registered_series(self) -> list[str]:
        sql_series = """
        SELECT mf.mandi_id,
               md.mandi_name,
               mf.crop_id,
               cd.canonical_name,
               mf.district_id,
               ld.district_name,
               COUNT(*)::int AS total_rows,
               MAX((mf.time AT TIME ZONE 'Asia/Kolkata')::date) AS max_date
        FROM features.market_features mf
        JOIN features.mandi_dim md ON md.mandi_id = mf.mandi_id
        JOIN features.crop_dim cd ON cd.crop_id = mf.crop_id
        JOIN features.location_dim ld ON ld.district_id = mf.district_id
        GROUP BY 1, 2, 3, 4, 5, 6
        HAVING COUNT(*) >= 30
        ORDER BY ld.district_name, mf.crop_id, total_rows DESC;
        """
        with psycopg.connect(**DB_CONFIG, row_factory=dict_row) as conn, conn.cursor() as cur:
            cur.execute(sql_series)
            series_rows = cur.fetchall()

        run_ids = []
        for row in series_rows:
            mandi_id = row["mandi_id"]
            crop_id = row["crop_id"]
            meta = resolve_series_metadata(
                mandi_id=mandi_id,
                crop_id=crop_id,
                mandi_name_db=row["mandi_name"],
                canonical_crop_db=row["canonical_name"],
                district_id=row["district_id"],
                district_name_db=row["district_name"],
            )
            try:
                forecasts = self.run_forecast_for_series(
                    mandi_id, crop_id, row["max_date"], meta
                )
                run_id = self.persist_pipeline_run(
                    crop_id=crop_id,
                    mandi_id=mandi_id,
                    district_id=row["district_id"],
                    district_label=row["district_name"],
                    origin_date=row["max_date"],
                    forecasts=forecasts,
                    records_count=row["total_rows"],
                )
                run_ids.append(run_id)
                logger.info(
                    "Committed %-15s | %-20s (%-18s) -> short=₹%.1f, 14d=₹%.1f, 30d=₹%.1f",
                    row["district_name"],
                    meta["Market Name"],
                    crop_id,
                    forecasts[0]["predicted_modal_price"],
                    forecasts[1]["predicted_modal_price"],
                    forecasts[2]["predicted_modal_price"],
                )
            except Exception as exc:
                logger.error("Failed (%s, %s): %s", mandi_id, crop_id, exc)
        return run_ids


if __name__ == "__main__":
    adapter = M2PipelineAdapter()
    print("\n--- Running AgroVision Multi-District & Multi-Crop Batch Pipeline ---")
    completed = adapter.run_all_registered_series()
    print(f"\n[+] Finished. Committed {len(completed)} multi-horizon runs across Rice, Potato, and Wheat.")
