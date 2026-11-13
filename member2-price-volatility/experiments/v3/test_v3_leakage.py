"""
AgroVision Member 2 — Volatility V3 Independent Leakage & Target Integrity Tests

Test suite verifying:
1. Strict Temporal Purging / Non-Leakage across split boundaries.
2. Exact Horizon Bounding (max offset strictly <= 7d for 7D, <= 14d for 14D; eliminating V2 gap bug).
3. Feature Point-in-Time Safety (future price mutations do not alter features at time t).
4. Group Boundary Isolation (zero cross-mandi or cross-variety contamination).
5. Insufficient Future History Graceful Handling (NaN on zero forward trades).
6. Threshold Derivation Provenance (strictly training-derived, zero-aware).
"""

import sys
from pathlib import Path
import pytest
import numpy as np
import pandas as pd

V3_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = V3_DIR.parent.parent

sys.path.insert(0, str(V3_DIR))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from volatility_v3_target_definitions import (
    compute_v3_calendar_targets,
    apply_v3_temporal_purge,
    derive_thresholds_from_training,
    classify_volatility_series,
    VOLATILITY_CLASSES,
)
from volatility_v3_features import extract_v3_volatility_features, V3_FEATURE_COLS
from v3_evaluation_utils import compute_market_level_metrics


@pytest.fixture
def multi_mandi_series():
    """Generates synthetic time series across two mandis with irregular weekend gaps."""
    records = []
    # Mandi 1: 50 daily observations with occasional weekend skips
    d_curr = pd.Timestamp("2023-01-01")
    for i in range(60):
        # Skip Sundays
        if d_curr.dayofweek != 6:
            records.append({
                "Market Name": "Burdwan",
                "Variety": "Common",
                "Reported Date": d_curr.strftime("%Y-%m-%d"),
                "Modal Price (Rs./Quintal)": 2500.0 + (i % 5) * 20.0 + (i // 10) * 15.0,
                "Arrivals (Tonnes)": 100.0 + (i % 7) * 5.0,
            })
        d_curr += pd.Timedelta(days=1)

    # Mandi 2: 40 observations starting later, different prices
    d_curr2 = pd.Timestamp("2023-01-10")
    for i in range(50):
        if d_curr2.dayofweek not in (5, 6): # Skip weekends
            records.append({
                "Market Name": "Kalna",
                "Variety": "Common",
                "Reported Date": d_curr2.strftime("%Y-%m-%d"),
                "Modal Price (Rs./Quintal)": 3000.0 + (i % 4) * 25.0,
                "Arrivals (Tonnes)": 80.0 + i * 2.0,
            })
        d_curr2 += pd.Timedelta(days=1)

    df = pd.DataFrame(records)
    return df


def test_calendar_horizon_strict_bounding(multi_mandi_series):
    """
    Asserts that the latest observation date used in the target is STRICTLY
    within the 7-day or 14-day calendar window.
    Directly proves resolution of the V2 shift(-5) variable-horizon flaw (up to 86d/13.8y).
    """
    df = compute_v3_calendar_targets(
        multi_mandi_series,
        group_cols=["Market Name", "Variety"],
        date_col="Reported Date",
        price_col="Modal Price (Rs./Quintal)",
        horizons=[7, 14],
        min_future_obs=1,
    )

    df["Reported Date"] = pd.to_datetime(df["Reported Date"])

    # 7-day horizon checks
    valid_7d = df.dropna(subset=["future_target_date_7d", "future_pct_range_7d"]).copy()
    diffs_7d = (valid_7d["future_target_date_7d"] - valid_7d["Reported Date"]).dt.days
    assert (diffs_7d > 0).all(), "Target dates must be strictly in the future"
    assert (diffs_7d <= 7).all(), f"7-day target horizon exceeded: max offset was {diffs_7d.max()} days"
    assert diffs_7d.max() == 7, "Expected maximum 7d offset to reach 7 days"

    # 14-day horizon checks
    valid_14d = df.dropna(subset=["future_target_date_14d", "future_pct_range_14d"]).copy()
    diffs_14d = (valid_14d["future_target_date_14d"] - valid_14d["Reported Date"]).dt.days
    assert (diffs_14d > 0).all(), "Target dates must be strictly in the future"
    assert (diffs_14d <= 14).all(), f"14-day target horizon exceeded: max offset was {diffs_14d.max()} days"
    assert diffs_14d.max() == 14, "Expected maximum 14d offset to reach 14 days"


def test_strict_temporal_purging_non_leakage(multi_mandi_series):
    """
    Asserts that after temporal purging at split_date:
    1. No row in the cleaned training set has Reported Date >= split_date.
    2. No row in the cleaned training set has future_target_date >= split_date.
    3. Exactly the boundary rows that span across split_date are purged.
    """
    df = compute_v3_calendar_targets(
        multi_mandi_series,
        group_cols=["Market Name", "Variety"],
        date_col="Reported Date",
        price_col="Modal Price (Rs./Quintal)",
        horizons=[7, 14],
    )

    split_date = "2023-01-25"
    split_ts = pd.Timestamp(split_date)

    # 7D purging test
    clean_train_mask_7, is_eval_mask_7, purge_cnt_7 = apply_v3_temporal_purge(
        df, "Reported Date", "future_target_date_7d", split_date
    )

    clean_train_7 = df[clean_train_mask_7]
    assert purge_cnt_7 > 0, "Expected boundary rows to be purged"
    assert (clean_train_7["Reported Date"] < split_ts).all(), "Train date must be before split date"
    assert (clean_train_7["future_target_date_7d"] < split_ts).all(), (
        "Zero leakage invariant failed: train row contains target date >= split date"
    )

    # 14D purging test
    clean_train_mask_14, is_eval_mask_14, purge_cnt_14 = apply_v3_temporal_purge(
        df, "Reported Date", "future_target_date_14d", split_date
    )

    clean_train_14 = df[clean_train_mask_14]
    assert purge_cnt_14 >= purge_cnt_7, "14D purge window must purge at least as many boundary rows as 7D"
    assert (clean_train_14["Reported Date"] < split_ts).all()
    assert (clean_train_14["future_target_date_14d"] < split_ts).all()


def test_feature_point_in_time_safety(multi_mandi_series):
    """
    Verifies that mutating future prices at t+1, t+2 has ZERO effect on features at t.
    Proves point-in-time safety of all 38 features.
    """
    df_base = multi_mandi_series.copy()
    feat_base = extract_v3_volatility_features(
        df_base, ["Market Name", "Variety"], "Reported Date", "Modal Price (Rs./Quintal)"
    )

    # Mutate future rows (last 10 rows of Burdwan)
    df_mutated = multi_mandi_series.copy()
    burdwan_indices = df_mutated[df_mutated["Market Name"] == "Burdwan"].index
    mutate_targets = burdwan_indices[-10:]
    df_mutated.loc[mutate_targets, "Modal Price (Rs./Quintal)"] *= 3.0  # Massive 300% price shock

    feat_mutated = extract_v3_volatility_features(
        df_mutated, ["Market Name", "Variety"], "Reported Date", "Modal Price (Rs./Quintal)"
    )

    # Features at t < mutation_start must be identical
    pre_mutation_indices = burdwan_indices[:-10]
    for col in V3_FEATURE_COLS:
        diff = np.abs(
            feat_base.loc[pre_mutation_indices, col].values
            - feat_mutated.loc[pre_mutation_indices, col].values
        )
        assert np.nanmax(diff) < 1e-6, f"Feature leakage detected in '{col}': pre-mutation values changed!"


def test_group_boundary_isolation():
    """
    Asserts that calculations for Mandi A never mix into Mandi B, even when interleaved.
    """
    dates = pd.date_range("2023-01-01", periods=10, freq="D")
    records = []
    for d in dates:
        records.append({
            "Market Name": "MarketAlpha",
            "Variety": "V1",
            "Reported Date": d,
            "Modal Price (Rs./Quintal)": 1000.0,
            "Arrivals (Tonnes)": 50.0,
        })
        records.append({
            "Market Name": "MarketBeta",
            "Variety": "V1",
            "Reported Date": d,
            "Modal Price (Rs./Quintal)": 5000.0,
            "Arrivals (Tonnes)": 150.0,
        })
    df = pd.DataFrame(records)

    df_res = compute_v3_calendar_targets(
        df, ["Market Name", "Variety"], "Reported Date", "Modal Price (Rs./Quintal)", horizons=[7]
    )

    # Since prices within each market are flat (1000 and 5000 respectively),
    # forward percentage ranges within each group must be exactly 0.0 (not affected by other mandi)
    assert (df_res["future_pct_range_7d"].dropna() == 0.0).all(), (
        "Group isolation failed: Flat prices within markets produced non-zero range due to cross-mandi bleeding!"
    )


def test_insufficient_future_history_handling():
    """
    Asserts that the final observation in a group (which has 0 future observations)
    evaluates to NaN for forward targets.
    """
    dates = pd.date_range("2023-01-01", periods=5, freq="D")
    df = pd.DataFrame({
        "Market Name": "SoloMarket",
        "Variety": "V1",
        "Reported Date": dates,
        "Modal Price (Rs./Quintal)": [2000.0, 2020.0, 2010.0, 2030.0, 2050.0],
    })

    df_res = compute_v3_calendar_targets(
        df, ["Market Name", "Variety"], "Reported Date", "Modal Price (Rs./Quintal)", horizons=[7]
    )

    # Last row has 0 future observations
    last_row = df_res.iloc[-1]
    assert np.isnan(last_row["future_pct_range_7d"]), "Last observation must have NaN forward target"
    assert last_row["future_obs_count_7d"] == 0


def test_threshold_derivation_from_training():
    """
    Verifies zero-aware threshold derivation strictly separates flat prices
    and splits non-zero price percentage movements at median.
    """
    np.random.seed(42)
    # 50 flat observations (0.0)
    zeros = np.zeros(50)
    # 50 non-zero movements with known median
    non_zeros = np.array([0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0] * 5)
    sample = pd.Series(np.concatenate([zeros, non_zeros]))

    th = derive_thresholds_from_training(sample, low_cutoff=0.014286)
    assert th["low_medium_threshold"] == 0.014286
    assert np.isclose(th["medium_high_threshold"], 2.75, atol=0.3)

    classes = classify_volatility_series(sample, th["low_medium_threshold"], th["medium_high_threshold"])
    counts = classes.value_counts()
    assert counts["LOW"] == 50
    assert counts["MEDIUM"] == 25
    assert counts["HIGH"] == 25


def test_flat_streak_and_move_feature_correctness():
    """
    Verifies that streak_zero and nonzero_moves accurately measure
    consecutive flat trading sessions and price change events.
    Guarantees repair of the double-inversion streak calculation bug.
    """
    dates = pd.date_range("2023-01-01", periods=6, freq="D")
    df = pd.DataFrame({
        "Market Name": ["MandiX"] * 6,
        "Variety": ["Common"] * 6,
        "Reported Date": dates,
        "Modal Price (Rs./Quintal)": [100.0, 100.0, 100.0, 105.0, 105.0, 102.0],
    })

    feat = extract_v3_volatility_features(
        df, ["Market Name", "Variety"], "Reported Date", "Modal Price (Rs./Quintal)"
    )

    # Expected streak_zero:
    # Day 0 (100.0): first trade, no prior -> 0
    # Day 1 (100.0): flat -> 1
    # Day 2 (100.0): flat -> 2
    # Day 3 (105.0): moved -> 0
    # Day 4 (105.0): flat -> 1
    # Day 5 (102.0): moved -> 0
    assert feat["streak_zero"].tolist() == [0.0, 1.0, 2.0, 0.0, 1.0, 0.0], (
        f"Incorrect streak_zero values: {feat['streak_zero'].tolist()}"
    )

    # Expected nonzero_moves_5 (rolling 5 with min_periods=5, filled with 0.0 for first 4 obs):
    # Day 0 to 3: < 5 observations -> 0.0
    # Day 4 (5th obs): window has [0, 0, 0, 1, 0] -> 1.0
    # Day 5 (6th obs): window has [0, 0, 1, 0, 1] -> 2.0
    assert feat["nonzero_moves_5"].tolist() == [0.0, 0.0, 0.0, 0.0, 1.0, 2.0], (
        f"Incorrect nonzero_moves_5 values: {feat['nonzero_moves_5'].tolist()}"
    )


def test_temporal_purge_robustness_with_string_and_timestamp_dates():
    """
    Verifies that apply_v3_temporal_purge works seamlessly and identically
    whether dates are passed as strings or pandas Timestamps.
    """
    df_str = pd.DataFrame([
        {"Reported Date": "2023-01-01", "target_date": "2023-01-05"},
        {"Reported Date": "2023-01-20", "target_date": "2023-01-28"},
        {"Reported Date": "2023-01-28", "target_date": "2023-02-02"},
    ])

    clean_train_s, is_eval_s, purged_s = apply_v3_temporal_purge(
        df_str, "Reported Date", "target_date", "2023-01-25"
    )
    assert int(clean_train_s.sum()) == 1
    assert int(is_eval_s.sum()) == 1
    assert purged_s == 1

    # Same test with datetime timestamps
    df_ts = df_str.copy()
    df_ts["Reported Date"] = pd.to_datetime(df_ts["Reported Date"])
    df_ts["target_date"] = pd.to_datetime(df_ts["target_date"])

    clean_train_t, is_eval_t, purged_t = apply_v3_temporal_purge(
        df_ts, "Reported Date", "target_date", "2023-01-25"
    )
    assert int(clean_train_t.sum()) == 1
    assert int(is_eval_t.sum()) == 1
    assert purged_t == 1


def test_single_observation_group_handling():
    """
    Asserts that a market/variety group with only 1 observation evaluates
    gracefully to NaN for all forward and past targets without throwing exceptions.
    """
    df_solo = pd.DataFrame([{
        "Market Name": "LoneMandi",
        "Variety": "RareVariety",
        "Reported Date": "2023-01-01",
        "Modal Price (Rs./Quintal)": 2400.0,
    }])

    res = compute_v3_calendar_targets(
        df_solo, ["Market Name", "Variety"], "Reported Date", "Modal Price (Rs./Quintal)", horizons=[7, 14]
    )
    assert len(res) == 1
    assert np.isnan(res.loc[0, "future_pct_range_7d"])
    assert np.isnan(res.loc[0, "future_pct_range_14d"])
    assert res.loc[0, "future_obs_count_7d"] == 0
    assert res.loc[0, "future_obs_count_14d"] == 0
    assert pd.isna(res.loc[0, "target_7d"])
    assert pd.isna(res.loc[0, "target_14d"])


def test_large_calendar_gaps_do_not_leak_or_bridge():
    """
    Asserts that observations separated by a 30-day hiatus are NOT linked
    by 7-day or 14-day forward windows, preventing the multi-month V2 bridge flaw.
    """
    df_gap = pd.DataFrame([
        {"Market Name": "GapMandi", "Variety": "V1", "Reported Date": "2023-01-01", "Modal Price (Rs./Quintal)": 2000.0},
        {"Market Name": "GapMandi", "Variety": "V1", "Reported Date": "2023-02-01", "Modal Price (Rs./Quintal)": 3000.0},
    ])

    res = compute_v3_calendar_targets(
        df_gap, ["Market Name", "Variety"], "Reported Date", "Modal Price (Rs./Quintal)", horizons=[7, 14]
    )

    # The 2023-01-01 trade has NO trades in (2023-01-01, 2023-01-08] or (2023-01-01, 2023-01-15]
    row0 = res.iloc[0]
    assert np.isnan(row0["future_pct_range_7d"]), "7D window must NOT bridge 31-day gap"
    assert np.isnan(row0["future_pct_range_14d"]), "14D window must NOT bridge 31-day gap"
    assert row0["future_obs_count_7d"] == 0
    assert row0["future_obs_count_14d"] == 0


def test_unsorted_input_determinism():
    """
    Verifies that passing shuffled/unsorted input data produces identical
    results to pre-sorted chronological data.
    """
    dates = pd.date_range("2023-01-01", periods=10, freq="D")
    df_sorted = pd.DataFrame({
        "Market Name": ["StableMandi"] * 10,
        "Variety": ["V1"] * 10,
        "Reported Date": dates,
        "Modal Price (Rs./Quintal)": [2000.0 + i * 10.0 for i in range(10)],
    })

    # Shuffle rows
    np.random.seed(99)
    df_shuffled = df_sorted.sample(frac=1.0, random_state=99).reset_index(drop=True)

    res_sorted = compute_v3_calendar_targets(
        df_sorted, ["Market Name", "Variety"], "Reported Date", "Modal Price (Rs./Quintal)", horizons=[7]
    )
    res_shuffled = compute_v3_calendar_targets(
        df_shuffled, ["Market Name", "Variety"], "Reported Date", "Modal Price (Rs./Quintal)", horizons=[7]
    )

    # Targets must match when aligned by date
    s1 = res_sorted.set_index("Reported Date")["future_pct_range_7d"]
    s2 = res_shuffled.set_index("Reported Date")["future_pct_range_7d"]
    pd.testing.assert_series_equal(s1, s2, check_exact=False, rtol=1e-5)


def test_zero_and_negative_prices_handling():
    """
    Verifies that zero or negative prices do not raise unhandled exceptions
    and produce NaN forward targets.
    """
    df_bad = pd.DataFrame([
        {"Market Name": "BadMandi", "Variety": "V1", "Reported Date": "2023-01-01", "Modal Price (Rs./Quintal)": 0.0},
        {"Market Name": "BadMandi", "Variety": "V1", "Reported Date": "2023-01-02", "Modal Price (Rs./Quintal)": -100.0},
        {"Market Name": "BadMandi", "Variety": "V1", "Reported Date": "2023-01-03", "Modal Price (Rs./Quintal)": 2000.0},
        {"Market Name": "BadMandi", "Variety": "V1", "Reported Date": "2023-01-04", "Modal Price (Rs./Quintal)": 2010.0},
    ])

    res = compute_v3_calendar_targets(
        df_bad, ["Market Name", "Variety"], "Reported Date", "Modal Price (Rs./Quintal)", horizons=[7]
    )
    assert len(res) == 4
    # Rows with 0 or negative price origin should have NaN target
    assert np.isnan(res.loc[0, "future_pct_range_7d"])
    assert np.isnan(res.loc[1, "future_pct_range_7d"])
    # Row 2 (2000.0) has future row 3 (2010.0)
    assert not np.isnan(res.loc[2, "future_pct_range_7d"])


def test_market_level_significance_directional_clarity():
    """
    Verifies that compute_market_level_metrics provides unambiguous directional
    significance flags, ensuring baseline superiority is NEVER misattributed to ML.
    """
    # Create synthetic evaluation where baseline is clearly superior to ML
    n_mandis = 20
    obs_per_mandi = 30
    records = []
    for m in range(n_mandis):
        for o in range(obs_per_mandi):
            records.append({
                "mandi": f"M_{m}",
                "target": "HIGH" if o % 3 == 0 else ("MEDIUM" if o % 3 == 1 else "LOW"),
                # Baseline is always correct
                "base_pred": "HIGH" if o % 3 == 0 else ("MEDIUM" if o % 3 == 1 else "LOW"),
                # ML is random/poor
                "ml_pred": "LOW",
            })
    df_eval = pd.DataFrame(records)

    res = compute_market_level_metrics(
        df_eval,
        mandi_col="mandi",
        y_true_col="target",
        y_pred_col="ml_pred",
        y_base_col="base_pred",
        min_obs_per_mandi=10,
    )

    sig = res["clustered_significance_tests"]
    assert sig["ml_advantage_statistically_significant_at_05"] is False
    assert sig["base_advantage_statistically_significant_at_05"] is True
    assert sig["significance_winner"] == "Persistence"


def test_nan_future_trades_do_not_produce_false_zero_volatility():
    """
    Asserts that when all future calendar observations within [t, t+H] are NaN,
    the target evaluates to NaN rather than being falsely classified as LOW (flat 0% volatility).
    """
    dates = pd.date_range("2023-01-01", periods=4, freq="D")
    df = pd.DataFrame({
        "Market Name": ["MandiNaN"] * 4,
        "Variety": ["Common"] * 4,
        "Reported Date": dates,
        "Modal Price (Rs./Quintal)": [2000.0, np.nan, np.nan, np.nan],
    })

    res = compute_v3_calendar_targets(
        df, ["Market Name", "Variety"], "Reported Date", "Modal Price (Rs./Quintal)", horizons=[7]
    )

    row0 = res.iloc[0]
    assert np.isnan(row0["future_pct_range_7d"]), "Expected NaN forward range when future prices are unobserved (NaN)"
    assert row0["future_obs_count_7d"] == 0, "Valid future observation count must be 0"
    assert pd.isna(row0["target_7d"]), "Target must be NaN, not false LOW"
    assert pd.isna(row0["future_target_date_7d"]), "Target date must be NaT when no valid future observation exists"


def test_timezone_robustness_in_temporal_purging():
    """
    Asserts that apply_v3_temporal_purge operates seamlessly when timestamps
    contain timezone information (e.g., UTC) without raising TypeError.
    """
    df_tz = pd.DataFrame([
        {"Reported Date": pd.Timestamp("2023-01-01 00:00:00", tz="UTC"), "target_date": pd.Timestamp("2023-01-05 00:00:00", tz="UTC")},
        {"Reported Date": pd.Timestamp("2023-01-20 00:00:00", tz="UTC"), "target_date": pd.Timestamp("2023-01-28 00:00:00", tz="UTC")},
        {"Reported Date": pd.Timestamp("2023-01-28 00:00:00", tz="UTC"), "target_date": pd.Timestamp("2023-02-02 00:00:00", tz="UTC")},
    ])

    clean_train, is_eval, purged = apply_v3_temporal_purge(
        df_tz, "Reported Date", "target_date", "2023-01-25"
    )
    assert int(clean_train.sum()) == 1
    assert int(is_eval.sum()) == 1
    assert purged == 1


def test_negative_dirty_prices_do_not_corrupt_forward_or_past_excursions():
    """
    Asserts that non-positive dirty prices (e.g. -50.0) in the future slice
    are filtered out and do not cause artificial massive percentage range spikes.
    """
    df = pd.DataFrame([
        {"Market Name": "DirtyMandi", "Variety": "V1", "Reported Date": "2023-01-01", "Modal Price (Rs./Quintal)": 2000.0},
        {"Market Name": "DirtyMandi", "Variety": "V1", "Reported Date": "2023-01-02", "Modal Price (Rs./Quintal)": -50.0}, # Corrupt price
        {"Market Name": "DirtyMandi", "Variety": "V1", "Reported Date": "2023-01-03", "Modal Price (Rs./Quintal)": 2020.0}, # Real price (1% move)
    ])

    res = compute_v3_calendar_targets(
        df, ["Market Name", "Variety"], "Reported Date", "Modal Price (Rs./Quintal)", horizons=[7]
    )

    row0 = res.iloc[0]
    # Move from 2000 to 2020 is 1.0%; if -50 was included, excursion would be > 103%
    assert not np.isnan(row0["future_pct_range_7d"])
    assert np.isclose(row0["future_pct_range_7d"], 1.0, atol=0.05), (
        f"Corrupt negative price affected forward excursion: range was {row0['future_pct_range_7d']}%"
    )
    assert row0["future_obs_count_7d"] == 1, "Only the 1 valid positive future observation should be counted"


