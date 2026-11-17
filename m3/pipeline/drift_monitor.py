"""
AgroVision Covariate Drift Monitor (PSI + Wasserstein-1 Distance)
Compares live TimescaleDB market & environmental feature windows against
frozen historical training distribution summaries in pipeline/drift_reference.json.
"""

from __future__ import annotations

import json
import os
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import psycopg
from psycopg.rows import dict_row
from scipy.stats import wasserstein_distance

DB_CONFIG = {
    "host": os.getenv("PGHOST", "127.0.0.1"),
    "port": int(os.getenv("PGPORT", "5432")),
    "dbname": os.getenv("PGDATABASE", "agrovision"),
    "user": os.getenv("PGUSER", "agrovision"),
    "password": os.getenv("PGPASSWORD", "postgres"),
}

REFERENCE_PATH = Path(__file__).resolve().parent / "drift_reference.json"
DEFAULT_ML_ROOT = Path.home() / "projects" / "agrovision-ml"

PSI_EPSILON = 1e-4
PSI_MODERATE_THRESHOLD = 0.10
PSI_CRITICAL_THRESHOLD = 0.25


def _summarize_series(
    values: np.ndarray,
    *,
    label: str,
    domain: str,
    unit: str,
    n_bins: int = 10,
    max_sample: int = 500,
) -> dict[str, Any]:
    """Builds compact quantile bin edges + downsampled sorted reference array."""
    clean = np.asarray(values, dtype=float)
    clean = clean[np.isfinite(clean)]
    if clean.size < 10:
        raise ValueError(f"Insufficient finite observations ({clean.size}) for {label}")

    clean_sorted = np.sort(clean)

    raw_edges = np.quantile(clean_sorted, np.linspace(0.0, 1.0, n_bins + 1))
    unique_edges = [float(raw_edges[0])]
    for edge in raw_edges[1:]:
        if float(edge) > unique_edges[-1] + 1e-9:
            unique_edges.append(float(edge))
    if len(unique_edges) < 3:
        unique_edges = [float(clean_sorted[0]), float(np.median(clean_sorted)), float(clean_sorted[-1])]

    edges_arr = np.array(unique_edges, dtype=float)
    interior_edges = edges_arr[1:-1]
    bin_indices = np.digitize(clean_sorted, interior_edges, right=False)
    n_actual_bins = len(unique_edges) - 1
    ref_counts = np.bincount(bin_indices, minlength=n_actual_bins).astype(float)
    ref_props = (ref_counts / ref_counts.sum()).tolist()

    if clean_sorted.size <= max_sample:
        ref_sample = [round(float(x), 4) for x in clean_sorted]
    else:
        q_grid = np.linspace(0.0, 1.0, max_sample)
        ref_sample = [round(float(x), 4) for x in np.quantile(clean_sorted, q_grid)]

    return {
        "label": label,
        "domain": domain,
        "unit": unit,
        "n_ref": int(clean_sorted.size),
        "bin_edges": [round(x, 6) for x in unique_edges],
        "ref_props": [round(float(p), 6) for p in ref_props],
        "ref_sample": ref_sample,
        "ref_mean": round(float(np.mean(clean_sorted)), 4),
        "ref_std": round(float(np.std(clean_sorted)), 4),
        "ref_p10": round(float(np.quantile(clean_sorted, 0.10)), 4),
        "ref_p50": round(float(np.quantile(clean_sorted, 0.50)), 4),
        "ref_p90": round(float(np.quantile(clean_sorted, 0.90)), 4),
    }


def build_drift_reference(
    ml_root: Path = DEFAULT_ML_ROOT,
    output_path: Path = REFERENCE_PATH,
) -> dict[str, Any]:
    """
    Extracts compact historical training distribution summaries from agrovision-ml
    (plus TimescaleDB MODIS NDVI baseline) and saves drift_reference.json.
    """
    m2_train_csv = ml_root / "member2-price-volatility/data/processed/rice_train.csv"
    m1_weather_csv = ml_root / "member1-yield-model/data/processed/weather_nasapower_wb.csv"
    m1_yield_ndvi_csv = ml_root / "member1-yield-model/data/raw/ndvi/AgroVision_WB_Rice_NDVI_2000_2019_LONG.csv"

    usecols = [
        "District Name",
        "Modal Price (Rs./Quintal)",
        "Arrivals (Tonnes)",
        "rolling_mean_7",
        "rolling_std_7",
    ]
    hg_chunks: list[pd.DataFrame] = []
    for chunk in pd.read_csv(m2_train_csv, usecols=usecols, chunksize=50_000):
        mask = chunk["District Name"].astype(str).str.strip().str.lower() == "hooghly"
        if mask.any():
            hg_chunks.append(chunk.loc[mask].copy())
    m2_hg = pd.concat(hg_chunks, ignore_index=True)

    # Fetch 10-year daily historical weather for Hooghly (22.90N, 88.39E) so reference
    # grain matches daily environmental_features (avoiding seasonal min/max extreme bias).
    archive_url = (
        "https://archive-api.open-meteo.com/v1/archive?latitude=22.90&longitude=88.39"
        "&start_date=2016-01-01&end_date=2025-12-31"
        "&daily=temperature_2m_max,temperature_2m_min,relative_humidity_2m_mean"
        "&timezone=Asia%2FKolkata"
    )
    try:
        with urllib.request.urlopen(archive_url, timeout=20) as resp:
            arch = json.loads(resp.read().decode("utf-8"))["daily"]
        w_hg = pd.DataFrame({
            "T2M_MAX": arch["temperature_2m_max"],
            "T2M_MIN": arch["temperature_2m_min"],
            "RH2M": arch["relative_humidity_2m_mean"],
        }).dropna()
    except Exception:
        w_df = pd.read_csv(m1_weather_csv)
        w_hg = w_df[w_df["District"].astype(str).str.strip().str.lower() == "hooghly"].copy()

    yn_df = pd.read_csv(m1_yield_ndvi_csv)
    yn_hg = yn_df[yn_df["District"].astype(str).str.strip().str.lower() == "hooghly"].copy()
    ndvi_vals = yn_hg["NDVI"].dropna().to_numpy(dtype=float)

    try:
        with psycopg.connect(**DB_CONFIG) as conn, conn.cursor() as cur:
            cur.execute(
                """
                SELECT column_name FROM information_schema.columns
                WHERE table_schema = 'features' AND table_name = 'ndvi_baseline'
                  AND column_name LIKE '%ndvi%'
                ORDER BY ordinal_position LIMIT 1
                """
            )
            col_row = cur.fetchone()
            if col_row:
                ndvi_col = col_row[0]
                cur.execute(
                    f"SELECT {ndvi_col} FROM features.ndvi_baseline WHERE district_id = %s AND {ndvi_col} IS NOT NULL"
                )
                db_ndvi = np.array([float(r[0]) for r in cur.fetchall()], dtype=float)
                if db_ndvi.size >= 30:
                    ndvi_vals = np.concatenate([ndvi_vals, db_ndvi])
    except Exception:
        pass

    features: dict[str, Any] = {
        "modal_price_inr": _summarize_series(
            m2_hg["Modal Price (Rs./Quintal)"].to_numpy(),
            label="Modal Mandi Price",
            domain="market",
            unit="₹/q",
        ),
        "arrivals_mt": _summarize_series(
            m2_hg["Arrivals (Tonnes)"].to_numpy(),
            label="Daily Mandi Arrivals",
            domain="market",
            unit="MT",
        ),
        "rolling_mean_7": _summarize_series(
            m2_hg["rolling_mean_7"].to_numpy(),
            label="7-Day Rolling Price Mean",
            domain="market",
            unit="₹/q",
        ),
        "rolling_std_7": _summarize_series(
            m2_hg["rolling_std_7"].to_numpy(),
            label="7-Day Price Volatility (Std)",
            domain="market",
            unit="₹/q",
        ),
        "t_max_c": _summarize_series(
            w_hg["T2M_MAX"].to_numpy(),
            label="Max Temperature",
            domain="environment",
            unit="°C",
        ),
        "t_min_c": _summarize_series(
            w_hg["T2M_MIN"].to_numpy(),
            label="Min Temperature",
            domain="environment",
            unit="°C",
        ),
        "rh_mean_pct": _summarize_series(
            w_hg["RH2M"].to_numpy(),
            label="Relative Humidity",
            domain="environment",
            unit="%",
        ),
        "mean_ndvi": _summarize_series(
            ndvi_vals,
            label="Vegetation Index (NDVI)",
            domain="environment",
            unit="NDVI",
        ),
    }

    payload = {
        "schema_version": "1.0.0",
        "district_id": "WB_HG_01",
        "district_name": "Hooghly",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "thresholds": {
            "psi_moderate": PSI_MODERATE_THRESHOLD,
            "psi_critical": PSI_CRITICAL_THRESHOLD,
        },
        "features": features,
    }
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def calculate_psi(bin_edges: list[float], ref_props: list[float], live_values: np.ndarray) -> float:
    """Computes Population Stability Index (PSI) with epsilon smoothing."""
    clean = np.asarray(live_values, dtype=float)
    clean = clean[np.isfinite(clean)]
    if clean.size == 0:
        return 0.0

    interior_edges = np.array(bin_edges[1:-1], dtype=float)
    n_bins = len(ref_props)
    bin_indices = np.digitize(clean, interior_edges, right=False)
    live_counts = np.bincount(bin_indices, minlength=n_bins).astype(float)
    live_props = live_counts / live_counts.sum()

    p = np.clip(np.array(ref_props, dtype=float), PSI_EPSILON, None)
    q = np.clip(live_props, PSI_EPSILON, None)
    p = p / p.sum()
    q = q / q.sum()

    psi = float(np.sum((q - p) * np.log(q / p)))
    return round(max(0.0, psi), 4)


def classify_drift(psi: float, w1: float, ref_std: float, live_samples: int) -> str:
    # For N < 30, empty quantile bins artificially inflate PSI via epsilon smoothing;
    # fall back to standardized Wasserstein-1 distance (W1 / ref_std).
    if live_samples < 30:
        norm_w1 = w1 / max(ref_std, 1e-6)
        if norm_w1 >= 1.0:
            return "SIGNIFICANT_DRIFT"
        if norm_w1 >= 0.5:
            return "MODERATE_SHIFT"
        return "STABLE"
    if psi >= PSI_CRITICAL_THRESHOLD:
        return "SIGNIFICANT_DRIFT"
    if psi >= PSI_MODERATE_THRESHOLD:
        return "MODERATE_SHIFT"
    return "STABLE"


def compute_drift_report(
    mandi_id: Optional[str] = None,
    crop_id: Optional[str] = None,
    window_days: int = 30,
    reference_path: Path = REFERENCE_PATH,
) -> dict[str, Any]:
    """
    Evaluates PSI and Wasserstein-1 Distance across live TimescaleDB windows
    against the frozen training reference distributions.
    """
    if not reference_path.exists():
        raise FileNotFoundError(f"Drift reference file missing: {reference_path}")

    ref_doc = json.loads(reference_path.read_text(encoding="utf-8"))
    ref_features = ref_doc["features"]

    with psycopg.connect(**DB_CONFIG, row_factory=dict_row) as conn, conn.cursor() as cur:
        target_district = "WB_HG_01"
        if mandi_id:
            cur.execute("SELECT district_id FROM features.mandi_dim WHERE mandi_id = %s", (mandi_id,))
            d_row = cur.fetchone()
            if d_row:
                target_district = d_row["district_id"]
        mkt_clauses = [f"district_id = '{target_district}'"]
        params: list[Any] = []
        if mandi_id:
            mkt_clauses.append("mandi_id = %s")
            params.append(mandi_id)
        if crop_id:
            mkt_clauses.append("crop_id = %s")
            params.append(crop_id)

        where_sql = " AND ".join(mkt_clauses)
        cur.execute(
            f"""
            SELECT time, mandi_id, crop_id, modal_price_inr, arrivals_mt
            FROM features.market_features
            WHERE {where_sql}
              AND time >= (
                  SELECT MAX(time) - (%s * INTERVAL '1 day')
                  FROM features.market_features
                  WHERE {where_sql}
              )
            ORDER BY mandi_id, crop_id, time ASC
            """,
            params + [window_days + 10] + params,
        )
        mkt_rows = cur.fetchall()

        cur.execute(
            """
            SELECT time, t_max_c, t_min_c, rh_mean_pct
            FROM features.environmental_features
            WHERE district_id = %s
              AND time <= CURRENT_TIMESTAMP
              AND time >= CURRENT_TIMESTAMP - (%s * INTERVAL '1 day')
            ORDER BY time ASC
            """,
            (target_district, window_days),
        )
        env_rows = cur.fetchall()

        cur.execute(
            """
            SELECT mean_ndvi
            FROM features.environmental_features
            WHERE district_id = %s
              AND mean_ndvi IS NOT NULL
              AND time >= (
                  SELECT MAX(time) - (%s * INTERVAL '1 day')
                  FROM features.environmental_features
                  WHERE district_id = %s AND mean_ndvi IS NOT NULL
              )
            ORDER BY time ASC
            """,
            (target_district, max(window_days, 90), target_district),
        )
        ndvi_rows = cur.fetchall()

    live_series_map: dict[str, np.ndarray] = {}

    if mkt_rows:
        mkt_df = pd.DataFrame(mkt_rows)
        mkt_df["modal_price_inr"] = pd.to_numeric(mkt_df["modal_price_inr"], errors="coerce")
        mkt_df["arrivals_mt"] = pd.to_numeric(mkt_df["arrivals_mt"], errors="coerce")
        mkt_df["rolling_mean_7"] = (
            mkt_df.groupby(["mandi_id", "crop_id"])["modal_price_inr"]
            .transform(lambda s: s.rolling(7, min_periods=3).mean())
        )
        mkt_df["rolling_std_7"] = (
            mkt_df.groupby(["mandi_id", "crop_id"])["modal_price_inr"]
            .transform(lambda s: s.rolling(7, min_periods=3).std().fillna(0.0))
        )
        cutoff_time = mkt_df["time"].max() - pd.Timedelta(days=window_days)
        win_df = mkt_df[mkt_df["time"] >= cutoff_time]

        for col in ("modal_price_inr", "arrivals_mt", "rolling_mean_7", "rolling_std_7"):
            live_series_map[col] = win_df[col].dropna().to_numpy(dtype=float)

    if env_rows:
        env_df = pd.DataFrame(env_rows)
        for col in ("t_max_c", "t_min_c", "rh_mean_pct"):
            live_series_map[col] = pd.to_numeric(env_df[col], errors="coerce").dropna().to_numpy(dtype=float)
    if ndvi_rows:
        live_series_map["mean_ndvi"] = np.array([float(r["mean_ndvi"]) for r in ndvi_rows], dtype=float)

    feature_results: list[dict[str, Any]] = []
    for feat_key, meta in ref_features.items():
        live_arr = live_series_map.get(feat_key, np.array([], dtype=float))
        live_arr = live_arr[np.isfinite(live_arr)]

        if live_arr.size == 0:
            feature_results.append(
                {
                    "feature": feat_key,
                    "label": meta["label"],
                    "domain": meta["domain"],
                    "unit": meta["unit"],
                    "psi": 0.0,
                    "wasserstein_distance": 0.0,
                    "normalized_w1": 0.0,
                    "classification_metric": "NONE",
                    "status": "NO_LIVE_DATA",
                    "ref_mean": meta["ref_mean"],
                    "live_mean": None,
                    "ref_p10": meta["ref_p10"],
                    "ref_p90": meta["ref_p90"],
                    "live_samples": 0,
                }
            )
            continue

        psi_val = calculate_psi(meta["bin_edges"], meta["ref_props"], live_arr)
        w1_val = round(float(wasserstein_distance(np.array(meta["ref_sample"], dtype=float), live_arr)), 4)
        norm_w1_val = round(w1_val / max(float(meta["ref_std"]), 1e-6), 4)
        metric_used = "STANDARDIZED_W1" if int(live_arr.size) < 30 else "PSI"
        status = classify_drift(psi_val, w1_val, float(meta["ref_std"]), int(live_arr.size))

        feature_results.append(
            {
                "feature": feat_key,
                "label": meta["label"],
                "domain": meta["domain"],
                "unit": meta["unit"],
                "psi": psi_val,
                "wasserstein_distance": w1_val,
                "normalized_w1": norm_w1_val,
                "classification_metric": metric_used,
                "status": status,
                "ref_mean": meta["ref_mean"],
                "live_mean": round(float(np.mean(live_arr)), 4),
                "ref_p10": meta["ref_p10"],
                "ref_p90": meta["ref_p90"],
                "live_samples": int(live_arr.size),
            }
        )

    evaluated = [f for f in feature_results if f["status"] != "NO_LIVE_DATA"]
    max_psi = max((f["psi"] for f in evaluated), default=0.0)
    drifted_count = sum(1 for f in evaluated if f["status"] == "SIGNIFICANT_DRIFT")
    moderate_count = sum(1 for f in evaluated if f["status"] == "MODERATE_SHIFT")

    if drifted_count > 0:
        overall_status = "SIGNIFICANT_DRIFT"
    elif moderate_count > 0:
        overall_status = "MODERATE_SHIFT"
    else:
        overall_status = "STABLE"

    return {
        "overall_status": overall_status,
        "max_psi": round(max_psi, 4),
        "drifted_feature_count": drifted_count,
        "moderate_shift_count": moderate_count,
        "window_days": window_days,
        "mandi_id": mandi_id,
        "crop_id": crop_id,
        "reference_generated_at": ref_doc["generated_at"],
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
        "features": feature_results,
    }


if __name__ == "__main__":
    ref = build_drift_reference()
    print(f"Built drift_reference.json ({REFERENCE_PATH.stat().st_size / 1024:.1f} KB) with {len(ref['features'])} features.")
    report = compute_drift_report(window_days=30)
    print(json.dumps(report, indent=2))
