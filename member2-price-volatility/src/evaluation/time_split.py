"""
AgroVision — Time-Based Train/Test Split (fixed)

Uses ONE global cutoff date for every (Market, Variety) group:
    train = rows dated before the cutoff
    test  = rows dated on/after the cutoff

Why this replaces the per-group 80/20 split:
  - A per-group split puts some groups' test rows in 2003 while the model is
    trained on other groups' 2010-2015 data (cross-group future leakage), and
    the test metrics mix very different market periods.
  - Rows near the cutoff have a target (next price) that lies in the future.
    A train row whose target date falls on/after the cutoff is purged so the
    train labels never peek into the test period.

Also provides walk_forward_splits() for time-ordered CV / tuning.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from config import (
    DATE_COLUMN,
    GROUP_COLUMNS,
    features_final_path,
    train_path,
    test_path,
)

HORIZON_COLUMN = "days_to_next"  # gap (days) between a row and its target row


def _target_dates(df, group_cols):
    """Date on which each row's target (next price) is observed."""
    if HORIZON_COLUMN in df.columns:
        return df[DATE_COLUMN] + pd.to_timedelta(df[HORIZON_COLUMN], unit="D")
    if group_cols:
        return df.groupby(group_cols)[DATE_COLUMN].shift(-1)
    return df[DATE_COLUMN].shift(-1)


def _group_keys(frame, group_cols):
    return set(map(tuple, frame[group_cols].drop_duplicates().to_numpy()))


def split(df=None, test_ratio=0.2, cutoff_date=None, drop_unseen_groups=False):
    """
    Global chronological train/test split with target-leak purge.

    Parameters
    ----------
    df : pd.DataFrame, optional
    test_ratio : float
        Used only when cutoff_date is None: the cutoff is the date quantile
        that leaves ~test_ratio of all rows after it.
    cutoff_date : str | pd.Timestamp, optional
        Explicit cutoff, e.g. "2016-01-01". Overrides test_ratio.
    drop_unseen_groups : bool
        If True, remove test rows of groups that have no training rows.

    Returns
    -------
    train_df, test_df : tuple of pd.DataFrame
    """
    if df is None:
        path = features_final_path()
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")
        df = pd.read_csv(path)

    print("\n" + "=" * 60)
    print("STEP 4: TRAIN/TEST SPLIT (global cutoff)")
    print("=" * 60)

    df = df.copy()
    df[DATE_COLUMN] = pd.to_datetime(df[DATE_COLUMN], errors="coerce")

    n_bad = int(df[DATE_COLUMN].isna().sum())
    if n_bad:
        print(f"   Dropped {n_bad} rows with invalid dates")
        df = df.dropna(subset=[DATE_COLUMN])

    group_cols = [c for c in GROUP_COLUMNS if c in df.columns]
    df = df.sort_values(by=group_cols + [DATE_COLUMN]).reset_index(drop=True)

    # ---- choose cutoff ----
    if cutoff_date is None:
        cutoff = df[DATE_COLUMN].quantile(1 - test_ratio).normalize()
    else:
        cutoff = pd.Timestamp(cutoff_date)
    print(f"   Cutoff date : {cutoff.date()}")

    # ---- rows per year (shows where data actually exists) ----
    per_year = df[DATE_COLUMN].dt.year.value_counts().sort_index()
    print("   Rows per year: " + ", ".join(f"{y}:{n}" for y, n in per_year.items()))

    # ---- split + purge ----
    is_train = df[DATE_COLUMN] < cutoff
    target_dt = _target_dates(df, group_cols)
    purge = is_train & (target_dt >= cutoff)  # NaT compares False -> kept
    print(f"   Purged {int(purge.sum())} train rows whose target falls in test period")

    train_df = df[is_train & ~purge].reset_index(drop=True)
    test_df = df[~is_train].reset_index(drop=True)

    # ---- groups without training history ----
    if group_cols and len(test_df):
        unseen = _group_keys(test_df, group_cols) - _group_keys(train_df, group_cols)
        if unseen:
            n_rows = int(
                test_df[group_cols].apply(tuple, axis=1).isin(unseen).sum()
            )
            print(f"   NOTE: {len(unseen)} groups appear only in test ({n_rows} rows)")
            if drop_unseen_groups:
                keep = ~test_df[group_cols].apply(tuple, axis=1).isin(unseen)
                test_df = test_df[keep].reset_index(drop=True)
                print("         -> dropped from test (drop_unseen_groups=True)")

    total = len(train_df) + len(test_df)
    print(f"\n   Total rows : {total}")
    print(f"   Train rows : {len(train_df)} ({len(train_df) / max(total, 1):.1%})")
    print(f"   Test rows  : {len(test_df)} ({len(test_df) / max(total, 1):.1%})")

    if len(train_df) == 0 or len(test_df) == 0:
        raise ValueError("Empty train or test set - adjust cutoff_date / test_ratio.")

    print(f"\n   Train period: {train_df[DATE_COLUMN].min()} to {train_df[DATE_COLUMN].max()}")
    print(f"   Test period : {test_df[DATE_COLUMN].min()} to {test_df[DATE_COLUMN].max()}")

    # ---- leakage validation (global, not just within group) ----
    if train_df[DATE_COLUMN].max() < test_df[DATE_COLUMN].min():
        print("   PASS: all train dates precede all test dates (global).")
    else:
        raise AssertionError("Temporal overlap between train and test!")

    train_target_max = _target_dates(train_df, group_cols).max()
    if pd.notna(train_target_max) and train_target_max >= test_df[DATE_COLUMN].min():
        print("   WARNING: some train targets extend into the test period.")
    else:
        print("   PASS: no train target reaches into the test period.")

    # ---- save ----
    t_path, te_path = train_path(), test_path()
    t_path.parent.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(t_path, index=False)
    test_df.to_csv(te_path, index=False)
    print(f"\n   Saved: {t_path}")
    print(f"   Saved: {te_path}")

    return train_df, test_df


def walk_forward_splits(df, n_splits=4, min_train_frac=0.5, gap_days=7):
    """
    Expanding-window time-ordered folds for tuning / early stopping.

    Yields (train_idx, val_idx) index arrays. `df` needs a unique index and
    a datetime DATE_COLUMN. `gap_days` should be >= max forecast horizon (7)
    so train targets never reach into the validation window.
    """
    dates = pd.to_datetime(df[DATE_COLUMN])
    edges = pd.date_range(dates.quantile(min_train_frac), dates.max(), periods=n_splits + 1)
    gap = pd.Timedelta(days=gap_days)

    for i in range(n_splits):
        lo, hi = edges[i], edges[i + 1]
        in_val = (dates >= lo) & ((dates < hi) if i < n_splits - 1 else (dates <= hi))
        yield df.index[dates < lo - gap].to_numpy(), df.index[in_val].to_numpy()


def main():
    split()


if __name__ == "__main__":
    main()