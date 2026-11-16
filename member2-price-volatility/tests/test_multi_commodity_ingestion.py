"""
AgroVision-ML — Multi-Commodity Live Data Ingestion Test Suite
============================================================
Verifies all Phase 3 requirements:
1. Normalization and alias mapping for all 4 commodities (Rice, Wheat, Potato, Tea).
2. Mixed date format parsing and rejection of invalid/corrupt dates.
3. Modal price validation (> 0 in Rs./quintal) and rejection of invalid/zero/negative prices.
4. Arrivals non-negativity and default filling.
5. Unmapped mandi and variety handling (preserved, reported in audit, not dropped).
6. Anti-fabrication integrity: missing observations are never invented or treated as 0-movement.
7. Idempotent storage: duplicate inserts are rejected without errors or database corruption.
8. Data freshness auditing and staleness flagging (> 14 days lag).
9. Statistical distribution drift detection vs historical training baselines.
10. API security: credentials are never leaked in logs/exceptions; missing key raises DataConnectorAuthError.
"""

import sys
import os
import sqlite3
import tempfile
from pathlib import Path
from datetime import datetime, timezone, timedelta
import pandas as pd
import numpy as np
import pytest

TESTS_DIR = Path(__file__).resolve().parent
MEMBER2_DIR = TESTS_DIR.parent
PROJECT_ROOT = MEMBER2_DIR.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(MEMBER2_DIR / "src") not in sys.path:
    sys.path.insert(0, str(MEMBER2_DIR / "src"))

from ingestion.multi_commodity_ingestion import (
    DataGovInMultiCommodityConnector,
    DataConnectorAuthError,
    DataConnectorNetworkError,
    normalize_and_validate_commodity_records,
    audit_commodity_freshness_and_drift,
    persist_commodity_observations,
    get_ingestion_summary,
    map_commodity_market,
    map_commodity_variety,
    COMMODITY_TRAINING_BASELINES,
)


# ============================================================
# 1. Normalization & Canonical Alias Mapping Tests
# ============================================================

@pytest.mark.parametrize("commodity,raw_market,expected_market,raw_variety,expected_variety", [
    ("rice", "bardhaman", "Burdwan", "paddy(dhan)(common)", "Common"),
    ("rice", "Burdwan", "Burdwan", "Fine", "Fine"),
    ("wheat", "kalna", "Kalna", "sonalika", "Sonalika"),
    ("wheat", "Khanna", "Khanna", "pbw-343", "PBW-343"),
    ("potato", "champadanga", "Champadanga", "jyoti", "Jyoti"),
    ("potato", "Agra", "Agra", "chandramukhi", "Chandermukhi"),
    ("tea", "kolkata", "Kolkata", "ctc", "CTC"),
    ("tea", "Siliguri", "Siliguri", "All Dust", "All Dust"),
])
def test_alias_and_canonical_mapping(commodity, raw_market, expected_market, raw_variety, expected_variety):
    """Verify market and variety aliases resolve to canonical names for all commodities."""
    m_name, m_known = map_commodity_market(commodity, raw_market)
    assert m_known is True
    assert m_name == expected_market

    v_name, v_known = map_commodity_variety(commodity, raw_variety)
    assert v_known is True
    assert v_name == expected_variety


def test_unmapped_market_and_variety_preservation():
    """Unmapped mandis and varieties must be kept with original labels and reported in audit."""
    df_raw = pd.DataFrame([{
        "Market Name": "CompletelyUnknownMandi123",
        "Variety": "ExoticNovelVarietyXYZ",
        "Reported Date": "2024-03-01",
        "Modal Price (Rs./Quintal)": 2600.0,
        "Arrivals (Tonnes)": 50.0,
    }])
    clean_df, audit = normalize_and_validate_commodity_records(df_raw, commodity="wheat")
    assert len(clean_df) == 1
    assert clean_df.iloc[0]["market"] == "CompletelyUnknownMandi123"
    assert clean_df.iloc[0]["variety"] == "ExoticNovelVarietyXYZ"
    assert "CompletelyUnknownMandi123" in audit["unmapped_mandi_labels"]
    assert "ExoticNovelVarietyXYZ" in audit["unmapped_variety_labels"]


# ============================================================
# 2. Date, Price, and Unit Validation Tests
# ============================================================

def test_mixed_date_format_normalization():
    """Verify both ISO YYYY-MM-DD and Indian DD/MM/YYYY dates parse cleanly to YYYY-MM-DD."""
    df_raw = pd.DataFrame([
        {"Market Name": "Burdwan", "Variety": "Common", "Reported Date": "2024-03-15", "Modal Price (Rs./Quintal)": 2500.0},
        {"Market Name": "Burdwan", "Variety": "Common", "Reported Date": "16/03/2024", "Modal Price (Rs./Quintal)": 2510.0},
        {"Market Name": "Burdwan", "Variety": "Common", "Reported Date": "invalid-garbage-date", "Modal Price (Rs./Quintal)": 2520.0},
    ])
    clean_df, audit = normalize_and_validate_commodity_records(df_raw, commodity="rice")
    assert len(clean_df) == 2
    assert audit["rejected_invalid_dates"] == 1
    assert list(clean_df["reported_date"]) == ["2024-03-15", "2024-03-16"]


def test_invalid_price_rejection():
    """Non-positive, NaN, and negative prices must be rejected and logged."""
    df_raw = pd.DataFrame([
        {"Market Name": "Kalna", "Variety": "Other", "Reported Date": "2024-03-01", "Modal Price (Rs./Quintal)": 2200.0},
        {"Market Name": "Kalna", "Variety": "Other", "Reported Date": "2024-03-02", "Modal Price (Rs./Quintal)": 0.0},
        {"Market Name": "Kalna", "Variety": "Other", "Reported Date": "2024-03-03", "Modal Price (Rs./Quintal)": -50.0},
        {"Market Name": "Kalna", "Variety": "Other", "Reported Date": "2024-03-04", "Modal Price (Rs./Quintal)": "N/A"},
    ])
    clean_df, audit = normalize_and_validate_commodity_records(df_raw, commodity="wheat")
    assert len(clean_df) == 1
    assert audit["rejected_invalid_prices"] == 3
    assert clean_df.iloc[0]["modal_price"] == 2200.0


def test_arrivals_non_negativity_and_default():
    """Missing arrivals should default to 0.0; negative arrivals should be clipped to 0.0."""
    df_raw = pd.DataFrame([
        {"Market Name": "Champadanga", "Variety": "Jyoti", "Reported Date": "2024-03-01", "Modal Price": 1200.0},
        {"Market Name": "Champadanga", "Variety": "Jyoti", "Reported Date": "2024-03-02", "Modal Price": 1210.0, "Arrivals": -10.0},
        {"Market Name": "Champadanga", "Variety": "Jyoti", "Reported Date": "2024-03-03", "Modal Price": 1220.0, "Arrivals": 42.5},
    ])
    clean_df, audit = normalize_and_validate_commodity_records(df_raw, commodity="potato")
    assert len(clean_df) == 3
    assert clean_df.iloc[0]["arrivals"] == 0.0
    assert clean_df.iloc[1]["arrivals"] == 0.0
    assert clean_df.iloc[2]["arrivals"] == 42.5


# ============================================================
# 3. Anti-Fabrication & Zero-Movement Policy
# ============================================================

def test_anti_fabrication_never_interpolates_missing_reports():
    """
    Requirement 3: Never fabricate missing observations or silently treat
    a gap in daily reports as zero price movement.
    """
    # 2 observations with a 5-day gap in between
    df_raw = pd.DataFrame([
        {"Market Name": "Burdwan", "Variety": "Common", "Reported Date": "2024-03-01", "Modal Price": 2500.0},
        {"Market Name": "Burdwan", "Variety": "Common", "Reported Date": "2024-03-07", "Modal Price": 2550.0},
    ])
    clean_df, audit = normalize_and_validate_commodity_records(df_raw, commodity="rice")
    
    # Must contain ONLY the 2 genuine dates, with NO synthetic dates in between
    assert len(clean_df) == 2
    dates = list(clean_df["reported_date"])
    assert dates == ["2024-03-01", "2024-03-07"]
    assert "2024-03-02" not in dates
    assert "2024-03-05" not in dates


# ============================================================
# 4. Idempotent Storage & Deduplication Tests
# ============================================================

def test_idempotent_sqlite_persistence(tmp_path):
    """Re-inserting identical batches must insert 0 new rows and increment rows_duplicate."""
    db_file = tmp_path / "test_ingestion_wheat.db"
    df_raw = pd.DataFrame([
        {"Market Name": "Khanna", "Variety": "Dara", "Reported Date": "2024-03-01", "Modal Price": 2400.0, "Arrivals": 50.0},
        {"Market Name": "Khanna", "Variety": "Dara", "Reported Date": "2024-03-02", "Modal Price": 2410.0, "Arrivals": 55.0},
    ])
    clean_df, audit = normalize_and_validate_commodity_records(df_raw, commodity="wheat")

    # Run 1: First insertion
    res1 = persist_commodity_observations(clean_df, commodity="wheat", db_path=db_file, audit_report=audit)
    assert res1["rows_inserted"] == 2
    assert res1["rows_duplicate"] == 0
    assert res1["total_persisted_records"] == 2

    # Run 2: Exact same data inserted again
    res2 = persist_commodity_observations(clean_df, commodity="wheat", db_path=db_file, audit_report=audit)
    assert res2["rows_inserted"] == 0
    assert res2["rows_duplicate"] == 2
    assert res2["total_persisted_records"] == 2

    # Summary telemetry check
    summary = get_ingestion_summary("wheat", db_path=db_file)
    assert summary["total_records"] == 2
    assert summary["active_markets_count"] == 1
    assert summary["latest_observation_date"] == "2024-03-02"
    assert summary["last_rows_duplicate"] == 2


# ============================================================
# 5. Data Freshness & Staleness Auditing
# ============================================================

def test_freshness_audit_staleness_detection():
    """Verify freshness audit correctly flags data older than 14 days as stale."""
    clean_df = pd.DataFrame([
        {"market": "Champadanga", "variety": "Jyoti", "reported_date": "2024-01-01", "modal_price": 950.0},
        {"market": "Champadanga", "variety": "Jyoti", "reported_date": "2024-01-10", "modal_price": 960.0},
    ])

    # Reference date is Jan 12, 2024 (2 days lag -> fresh)
    fresh_audit = audit_commodity_freshness_and_drift(clean_df, commodity="potato", reference_date="2024-01-12")
    assert fresh_audit["is_stale"] is False
    assert fresh_audit["reporting_lag_days"] == 2

    # Reference date is Feb 01, 2024 (22 days lag -> stale)
    stale_audit = audit_commodity_freshness_and_drift(clean_df, commodity="potato", reference_date="2024-02-01")
    assert stale_audit["is_stale"] is True
    assert stale_audit["reporting_lag_days"] == 22


# ============================================================
# 6. Statistical Distribution Drift Detection
# ============================================================

def test_distribution_drift_detection():
    """Verify statistical drift is detected when incoming batch mean deviates significantly from baseline."""
    # Wheat baseline: mean ~1587, std ~435. Extreme price 3500.0 is > 4 std deviations away.
    df_drift = pd.DataFrame([
        {"market": "Khanna", "variety": "Dara", "reported_date": "2024-03-01", "modal_price": 3500.0},
        {"market": "Khanna", "variety": "Dara", "reported_date": "2024-03-02", "modal_price": 3600.0},
    ])
    audit = audit_commodity_freshness_and_drift(df_drift, commodity="wheat", reference_date="2024-03-05")
    assert audit["baseline_comparison"]["drift_detected"] is True
    assert audit["baseline_comparison"]["mean_delta"] > 1800.0

    # Normal prices: ~1600.0 is within 1 std deviation -> no drift
    df_normal = pd.DataFrame([
        {"market": "Khanna", "variety": "Dara", "reported_date": "2024-03-01", "modal_price": 1600.0},
        {"market": "Khanna", "variety": "Dara", "reported_date": "2024-03-02", "modal_price": 1620.0},
    ])
    audit_norm = audit_commodity_freshness_and_drift(df_normal, commodity="wheat", reference_date="2024-03-05")
    assert audit_norm["baseline_comparison"]["drift_detected"] is False


# ============================================================
# 7. Connector Security & Rate Limit Handling
# ============================================================

def test_connector_missing_key_behavior():
    """Connector without API key must return safe diagnostic and raise on fetch."""
    connector = DataGovInMultiCommodityConnector(api_key="")
    diag = connector.check_connection("rice")
    assert diag["accessible"] is False
    assert diag["auth_configured"] is False
    assert diag["error_category"] == "missing_api_key"

    with pytest.raises(DataConnectorAuthError, match="Missing DATA_GOV_IN_API_KEY"):
        connector.fetch_records("rice")


def test_connector_sanitizes_credentials_in_logs():
    """Connector must strictly redact API keys from error messages."""
    fake_secret = "secret_key_abcdef123456789"
    connector = DataGovInMultiCommodityConnector(api_key=fake_secret)
    sanitized = connector._sanitize(f"Connection failed for key {fake_secret} on endpoint")
    assert fake_secret not in sanitized
    assert "[REDACTED_API_KEY]" in sanitized
