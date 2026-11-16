"""
Tests for Multi-Commodity Shadow Monitoring with Tea & Schema Verification
========================================================================
Validates Phase 5 requirements:
- Isolated shadow databases for Rice, Wheat, Potato, Tea
- Correct ledger schema: prediction_residual, actual_volatility_outcome, probabilities
- Delayed outcome evaluation: pending vs evaluated vs unavailable
- Matched persistence baseline benchmarking
"""

import sys
import gc
from pathlib import Path
import pytest
import pandas as pd

TEST_DIR = Path(__file__).resolve().parent
MEMBER2_DIR = TEST_DIR.parent
SRC_DIR = MEMBER2_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(MEMBER2_DIR) not in sys.path:
    sys.path.insert(0, str(MEMBER2_DIR))

from multi_commodity_shadow import (
    init_commodity_shadow_db,
    get_commodity_db_connection,
    ingest_commodity_mandi_observations,
    generate_commodity_shadow_forecasts,
    evaluate_pending_commodity_shadow_forecasts,
    compute_commodity_shadow_metrics,
    get_commodity_shadow_status,
    is_commodity_shadow_enabled,
    set_commodity_shadow_enabled,
    reset_commodity_shadow_switches,
)


@pytest.fixture(autouse=True)
def clean_switches():
    reset_commodity_shadow_switches()
    yield
    reset_commodity_shadow_switches()
    gc.collect()


def test_init_shadow_db_all_commodities(tmp_path):
    """Verifies that init_commodity_shadow_db supports rice, wheat, potato, and tea."""
    for comm in ["rice", "wheat", "potato", "tea"]:
        db_path = tmp_path / f"{comm}.db"
        res_path = init_commodity_shadow_db(comm, db_path)
        assert res_path.exists()

        conn = get_commodity_db_connection(comm, db_path)
        cursor = conn.cursor()
        cursor.execute("PRAGMA table_info(shadow_forecasts);")
        cols = {row[1] for row in cursor.fetchall()}
        conn.close()

        required_cols = {
            "forecast_id", "commodity", "market", "variety", "origin_date",
            "horizon_days", "origin_modal_price", "predicted_modal_price",
            "matched_actual_price", "prediction_residual", "actual_volatility_outcome",
            "volatility_pred_class", "volatility_probabilities_json", "status"
        }
        missing = required_cols - cols
        assert not missing, f"Missing required columns in {comm} shadow table: {missing}"


def test_tea_shadow_lifecycle(tmp_path):
    """Verifies complete prospective shadow lifecycle for Tea: generation, delayed matching, metrics."""
    db_path = tmp_path / "tea_shadow.db"
    set_commodity_shadow_enabled("tea", True)

    # 1. 35 days of history for Siliguri CTC
    dates = pd.date_range("2024-01-01", periods=35).strftime("%Y-%m-%d")
    history_df = pd.DataFrame([
        {
            "Market Name": "Siliguri",
            "Variety": "CTC",
            "Reported Date": d,
            "Modal Price (Rs./Quintal)": 200.0 + i * 0.4,
            "Arrivals (Tonnes)": 45.0 + (i % 5),
        }
        for i, d in enumerate(dates)
    ])

    # 2. Generate shadow forecast for Tea at origin 2024-01-30
    gen_res = generate_commodity_shadow_forecasts(
        commodity="tea",
        history_df=history_df,
        origin_date="2024-01-30",
        db_path=db_path,
    )
    assert gen_res["status"] == "success"
    assert gen_res["forecasts_generated"] == 1
    assert len(gen_res["forecast_ids"]) == 1

    # Verify status is PENDING
    status_summary = get_commodity_shadow_status("tea", db_path=db_path)
    assert status_summary["forecasts"]["total"] == 1
    assert status_summary["forecasts"]["pending"] == 1
    assert status_summary["forecasts"]["evaluated"] == 0

    # 3. Ingest next-day observation (2024-01-31)
    obs_df = pd.DataFrame([
        {
            "market": "Siliguri",
            "variety": "CTC",
            "reported_date": "2024-01-31",
            "modal_price": 215.0,
            "arrivals": 50.0,
        }
    ])
    ingested = ingest_commodity_mandi_observations("tea", obs_df, db_path=db_path)
    assert ingested == 1

    # 4. Evaluate pending forecasts as of 2024-02-05
    eval_res = evaluate_pending_commodity_shadow_forecasts("tea", as_of_date="2024-02-05", db_path=db_path)
    assert eval_res["status"] == "success"
    assert eval_res["evaluated_count"] == 1
    assert eval_res["still_pending_count"] == 0

    # 5. Check ledger record details
    conn = get_commodity_db_connection("tea", db_path)
    row = conn.execute("SELECT * FROM shadow_forecasts WHERE commodity = 'tea';").fetchone()
    conn.close()

    assert row["status"] == "EVALUATED"
    assert row["matched_observation_date"] == "2024-01-31"
    assert row["matched_actual_price"] == 215.0
    assert row["prediction_residual"] is not None
    assert row["actual_volatility_outcome"] in ("LOW", "MEDIUM", "HIGH")
    assert row["model_name"] == "tea-price-xgb"
    assert row["model_version"] == "1.0.0"

    # 6. Compute metrics
    metrics = compute_commodity_shadow_metrics("tea", db_path=db_path)
    assert "1d" in metrics["horizons"]
    h1 = metrics["horizons"]["1d"]
    assert h1["status"] == "evaluated"
    assert h1["sample_size"] == 1
    assert "mae_rs_qtl" in h1["model_performance"]
    assert "mae_rs_qtl" in h1["persistence_baseline"]
