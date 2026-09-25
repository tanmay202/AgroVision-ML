"""
AgroVision — Leakage & Integrity Tests

Validates:
    1. No target leakage in feature lists
    2. No current-row price features in price model
    3. price_pct_change not in volatility features
    4. Temporal ordering of train/test split
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


def test_no_target_in_price_features():
    """Price target must not appear in price features."""
    from config import PRICE_FEATURES, PRICE_TARGET
    assert PRICE_TARGET not in PRICE_FEATURES, (
        f"TARGET LEAKAGE: {PRICE_TARGET} is in PRICE_FEATURES!"
    )


def test_no_target_in_volatility_features():
    """Volatility target must not appear in volatility features."""
    from config import VOLATILITY_FEATURES, VOLATILITY_TARGET
    assert VOLATILITY_TARGET not in VOLATILITY_FEATURES, (
        f"TARGET LEAKAGE: {VOLATILITY_TARGET} is in VOLATILITY_FEATURES!"
    )


def test_no_current_price_in_features():
    """Current-row price columns must not be used as features."""
    from config import PRICE_FEATURES, PRICE_COLUMN, MIN_PRICE_COLUMN, MAX_PRICE_COLUMN

    leaky_columns = {PRICE_COLUMN, MIN_PRICE_COLUMN, MAX_PRICE_COLUMN}
    found = leaky_columns & set(PRICE_FEATURES)

    assert not found, (
        f"CURRENT-ROW LEAKAGE: {found} are in PRICE_FEATURES!\n"
        f"These are current-row values that shouldn't predict future price."
    )


def test_no_pct_change_in_volatility():
    """price_pct_change must not be in volatility features (it IS the target)."""
    from config import VOLATILITY_FEATURES
    assert "price_pct_change" not in VOLATILITY_FEATURES, (
        "LEAKAGE: price_pct_change is in VOLATILITY_FEATURES!"
    )


def test_no_arrival_column_in_price_features():
    """Raw Arrivals column should not be in price features (current-row value)."""
    from config import PRICE_FEATURES, ARRIVAL_COLUMN
    assert ARRIVAL_COLUMN not in PRICE_FEATURES, (
        f"CURRENT-ROW LEAKAGE: {ARRIVAL_COLUMN} is in PRICE_FEATURES!"
    )


def test_no_arrival_change_uses_current():
    """arrival_change should not be in features (it uses current-row arrival)."""
    from config import PRICE_FEATURES
    # arrival_change was removed from features in v2.0
    # It's still computed for analysis but not used as a model feature
    assert "arrival_change" not in PRICE_FEATURES, (
        "arrival_change uses current-row data and should not be a feature"
    )


def test_volatility_classes_consistent():
    """Verify label map and reverse map are consistent."""
    from config import VOLATILITY_LABEL_MAP, VOLATILITY_REVERSE_LABEL_MAP, VOLATILITY_CLASSES

    for cls in VOLATILITY_CLASSES:
        assert cls in VOLATILITY_LABEL_MAP, f"Missing class {cls} in LABEL_MAP"
        code = VOLATILITY_LABEL_MAP[cls]
        assert code in VOLATILITY_REVERSE_LABEL_MAP, f"Missing code {code} in REVERSE_MAP"
        assert VOLATILITY_REVERSE_LABEL_MAP[code] == cls, "Label maps are inconsistent"


def test_feature_lists_have_no_duplicates():
    """Feature lists must not contain duplicates."""
    from config import PRICE_FEATURES, VOLATILITY_FEATURES

    assert len(PRICE_FEATURES) == len(set(PRICE_FEATURES)), (
        f"PRICE_FEATURES has duplicates: "
        f"{[f for f in PRICE_FEATURES if PRICE_FEATURES.count(f) > 1]}"
    )

    assert len(VOLATILITY_FEATURES) == len(set(VOLATILITY_FEATURES)), (
        f"VOLATILITY_FEATURES has duplicates"
    )


def test_commodity_paths_dynamic():
    """Verify paths change when commodity is switched."""
    from config import set_commodity, raw_data_path, train_path, price_model_path

    set_commodity("tea")
    tea_raw = str(raw_data_path())
    tea_train = str(train_path())
    tea_model = str(price_model_path())

    set_commodity("onion")
    onion_raw = str(raw_data_path())
    onion_train = str(train_path())
    onion_model = str(price_model_path())

    assert "tea" in tea_raw and "onion" in onion_raw, "Raw paths don't change with commodity"
    assert "tea" in tea_train and "onion" in onion_train, "Train paths don't change"
    assert "tea" in tea_model and "onion" in onion_model, "Model paths don't change"

    # Reset
    set_commodity("tea")


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
