"""
AgroVision — Leakage & Integrity Tests

Validates:
    1. No target leakage in feature lists
    2. No current-row price columns in price-model features
    3. No price_pct_change in volatility features
    4. No rejected volatility features are configured
    5. No raw current-row arrivals in price features
    6. No arrival_change in price features
    7. Volatility label mappings are consistent
    8. Feature lists contain no duplicates
    9. Commodity paths change dynamically
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(
    0,
    str(Path(__file__).resolve().parent.parent / "src")
)


# ============================================================
# 1. PRICE TARGET LEAKAGE
# ============================================================

def test_no_target_in_price_features():
    """Price target must not appear in price features."""
    from config import PRICE_FEATURES, PRICE_TARGET

    assert PRICE_TARGET not in PRICE_FEATURES, (
        f"TARGET LEAKAGE: {PRICE_TARGET} "
        f"is in PRICE_FEATURES!"
    )


# ============================================================
# 2. VOLATILITY TARGET LEAKAGE
# ============================================================

def test_no_target_in_volatility_features():
    """Volatility target must not appear in volatility features."""
    from config import (
        VOLATILITY_FEATURES,
        VOLATILITY_TARGET,
    )

    assert VOLATILITY_TARGET not in VOLATILITY_FEATURES, (
        f"TARGET LEAKAGE: {VOLATILITY_TARGET} "
        f"is in VOLATILITY_FEATURES!"
    )


# ============================================================
# 3. CURRENT PRICE LEAKAGE
# ============================================================

def test_no_current_price_in_features():
    """
    Current-row Min, Max, and Modal prices must not be
    used as price-model features.

    The model should use historical price information
    such as lag and rolling features instead.
    """

    from config import (
        PRICE_FEATURES,
        PRICE_COLUMN,
        MIN_PRICE_COLUMN,
        MAX_PRICE_COLUMN,
    )

    leaky_columns = {
        PRICE_COLUMN,
        MIN_PRICE_COLUMN,
        MAX_PRICE_COLUMN,
    }

    found = leaky_columns & set(PRICE_FEATURES)

    assert not found, (
        f"CURRENT-ROW PRICE LEAKAGE: {found} "
        f"are in PRICE_FEATURES!\n"
        f"Current-row Min/Max/Modal prices must not "
        f"be used as model features."
    )


# ============================================================
# 4. PRICE-PCT-CHANGE LEAKAGE
# ============================================================

def test_no_pct_change_in_volatility():
    """
    price_pct_change must not be a volatility feature.

    It is derived directly from price movement and is
    therefore excluded from the volatility feature set.
    """

    from config import VOLATILITY_FEATURES

    assert "price_pct_change" not in VOLATILITY_FEATURES, (
        "LEAKAGE: price_pct_change is in "
        "VOLATILITY_FEATURES!"
    )


# ============================================================
# 5. REJECTED VOLATILITY FEATURES
# ============================================================

def test_rejected_volatility_features_are_not_configured():
    """
    Rejected volatility candidates must not enter the
    final volatility model.
    """

    from config import VOLATILITY_FEATURES

    rejected_features = {
        "price_pct_change",
        "rolling_range_7",
        "rolling_range_14",
        "price_change_count_7",
        "days_since_price_change",
        "arrival_change",
        "Arrivals (Tonnes)",
    }

    found = (
        rejected_features
        & set(VOLATILITY_FEATURES)
    )

    assert not found, (
        f"REJECTED FEATURES FOUND: {found}"
    )


# ============================================================
# 6. RAW ARRIVALS LEAKAGE
# ============================================================

def test_no_arrival_column_in_price_features():
    """
    Raw Arrivals column should not be used directly
    because it represents the current-row value.
    """

    from config import (
        PRICE_FEATURES,
        ARRIVAL_COLUMN,
    )

    assert ARRIVAL_COLUMN not in PRICE_FEATURES, (
        f"CURRENT-ROW ARRIVAL LEAKAGE: "
        f"{ARRIVAL_COLUMN} is in PRICE_FEATURES!"
    )


# ============================================================
# 7. ARRIVAL_CHANGE LEAKAGE
# ============================================================

def test_no_arrival_change_uses_current():
    """
    arrival_change uses current-row arrival information
    and therefore must not be a price-model feature.
    """

    from config import PRICE_FEATURES

    assert "arrival_change" not in PRICE_FEATURES, (
        "arrival_change uses current-row data "
        "and must not be a price-model feature."
    )


# ============================================================
# 8. VOLATILITY CLASS MAPPING
# ============================================================

def test_volatility_classes_consistent():
    """
    Verify that volatility label and reverse-label maps
    are internally consistent.
    """

    from config import (
        VOLATILITY_LABEL_MAP,
        VOLATILITY_REVERSE_LABEL_MAP,
        VOLATILITY_CLASSES,
    )

    for cls in VOLATILITY_CLASSES:

        assert cls in VOLATILITY_LABEL_MAP, (
            f"Missing class {cls} in LABEL_MAP"
        )

        code = VOLATILITY_LABEL_MAP[cls]

        assert code in VOLATILITY_REVERSE_LABEL_MAP, (
            f"Missing code {code} in REVERSE_MAP"
        )

        assert (
            VOLATILITY_REVERSE_LABEL_MAP[code]
            == cls
        ), (
            f"Label mapping inconsistent for {cls}"
        )


# ============================================================
# 9. FEATURE LIST DUPLICATES
# ============================================================

def test_feature_lists_have_no_duplicates():
    """Feature lists must not contain duplicate names."""

    from config import (
        PRICE_FEATURES,
        VOLATILITY_FEATURES,
    )

    # Price features
    price_duplicates = [
        feature
        for feature in PRICE_FEATURES
        if PRICE_FEATURES.count(feature) > 1
    ]

    assert len(PRICE_FEATURES) == len(
        set(PRICE_FEATURES)
    ), (
        f"PRICE_FEATURES has duplicates: "
        f"{sorted(set(price_duplicates))}"
    )

    # Volatility features
    volatility_duplicates = [
        feature
        for feature in VOLATILITY_FEATURES
        if VOLATILITY_FEATURES.count(feature) > 1
    ]

    assert len(VOLATILITY_FEATURES) == len(
        set(VOLATILITY_FEATURES)
    ), (
        f"VOLATILITY_FEATURES has duplicates: "
        f"{sorted(set(volatility_duplicates))}"
    )


# ============================================================
# 10. DYNAMIC COMMODITY PATHS
# ============================================================

def test_commodity_paths_dynamic():
    """
    Verify that data/model paths change when the
    commodity is switched.
    """

    from config import (
        set_commodity,
        raw_data_path,
        train_path,
        price_model_path,
    )

    # --------------------------------------------------
    # Rice
    # --------------------------------------------------

    set_commodity("rice")

    rice_raw = str(raw_data_path())
    rice_train = str(train_path())
    rice_model = str(price_model_path())

    # --------------------------------------------------
    # A second arbitrary commodity proves paths are computed dynamically;
    # this test never runs a Tea pipeline or consumes Tea artifacts.
    # --------------------------------------------------

    set_commodity("barley")

    barley_raw = str(raw_data_path())
    barley_train = str(train_path())
    barley_model = str(price_model_path())

    # --------------------------------------------------
    # Verify raw paths
    # --------------------------------------------------

    assert "rice" in rice_raw, (
        "Rice raw path does not contain 'rice'"
    )

    assert "barley" in barley_raw, (
        "Second commodity raw path does not update"
    )

    # --------------------------------------------------
    # Verify train paths
    # --------------------------------------------------

    assert "rice" in rice_train, (
        "Rice train path does not contain 'rice'"
    )

    assert "barley" in barley_train, (
        "Second commodity train path does not update"
    )

    # --------------------------------------------------
    # Verify model paths
    # --------------------------------------------------

    assert "rice" in rice_model, (
        "Rice model path does not contain 'rice'"
    )

    assert "barley" in barley_model, (
        "Second commodity model path does not update"
    )

    # --------------------------------------------------
    # Reset default commodity
    # --------------------------------------------------

    set_commodity("rice")


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    import pytest

    pytest.main([
        __file__,
        "-v",
        "-s",
    ])
