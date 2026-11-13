"""
AgroVision Member 2 — Volatility V3 Advanced Evaluation & Statistical Testing Utilities

Provides:
1. Clustered statistical significance testing across mandis (Clustered Bootstrap, Wilcoxon Signed-Rank, Paired Cluster t-test).
2. Multiclass probability calibration metrics:
   - Multiclass Brier score
   - Expected Calibration Error (ECE)
   - Reliability table generation (observed frequency vs mean predicted probability across 10 bins)
3. Market-level disaggregation:
   - Per-mandi Macro F1, Balanced Accuracy, HIGH recall, HIGH precision
   - Small vs Medium vs Large mandi tier breakdown
   - Per-mandi win / loss / tie counters against persistence
"""

from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon, t as student_t, chi2
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_recall_fscore_support,
    confusion_matrix,
    brier_score_loss,
)

VOLATILITY_CLASSES = ["LOW", "MEDIUM", "HIGH"]
VOLATILITY_LABEL_MAP = {"LOW": 0, "MEDIUM": 1, "HIGH": 2}


def compute_comprehensive_metrics(
    y_true: pd.Series,
    y_pred: pd.Series,
    model_name: str = "Model",
    period_name: str = "Evaluation",
    horizon_name: str = "7D",
) -> Dict[str, Any]:
    """Computes all standard pooled classification metrics."""
    y_true = pd.Series(y_true).reset_index(drop=True)
    y_pred = pd.Series(y_pred).reset_index(drop=True)
    n_obs = len(y_true)

    acc = float(accuracy_score(y_true, y_pred))
    bal_acc = float(balanced_accuracy_score(y_true, y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, labels=VOLATILITY_CLASSES, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(y_true, y_pred, labels=VOLATILITY_CLASSES, average="weighted", zero_division=0))

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=VOLATILITY_CLASSES, zero_division=0
    )

    per_class = {}
    for i, c in enumerate(VOLATILITY_CLASSES):
        per_class[c] = {
            "precision": float(precision[i]),
            "recall": float(recall[i]),
            "f1": float(f1[i]),
            "support": int(support[i]),
        }

    cm = confusion_matrix(y_true, y_pred, labels=VOLATILITY_CLASSES).tolist()

    # Detailed confusion analysis for HIGH
    # cm: [[LL, LM, LH], [ML, MM, MH], [HL, HM, HH]]
    high_idx = VOLATILITY_LABEL_MAP["HIGH"]
    low_idx = VOLATILITY_LABEL_MAP["LOW"]
    cm_arr = np.array(cm)

    # False positive alarms for HIGH: predicted HIGH when actual was NOT HIGH
    # False negative alarms for HIGH: actual was HIGH but predicted NOT HIGH
    high_pred_count = int(cm_arr[:, high_idx].sum())
    high_actual_count = int(cm_arr[high_idx, :].sum())
    high_true_pos = int(cm_arr[high_idx, high_idx])
    high_false_pos = high_pred_count - high_true_pos
    high_false_neg = high_actual_count - high_true_pos
    high_fp_rate = float(high_false_pos / max(1, high_pred_count)) # False alarm rate among predicted HIGH
    high_to_low_confusion = int(cm_arr[high_idx, low_idx]) # Actual HIGH misclassified as LOW
    low_to_high_confusion = int(cm_arr[low_idx, high_idx]) # Actual LOW misclassified as HIGH

    return {
        "model": model_name,
        "period": period_name,
        "horizon": horizon_name,
        "n_samples": n_obs,
        "accuracy": round(acc, 4),
        "balanced_accuracy": round(bal_acc, 4),
        "macro_f1": round(macro_f1, 4),
        "weighted_f1": round(weighted_f1, 4),
        "high_recall": round(per_class["HIGH"]["recall"], 4),
        "high_precision": round(per_class["HIGH"]["precision"], 4),
        "high_f1": round(per_class["HIGH"]["f1"], 4),
        "high_false_alarm_rate": round(high_fp_rate, 4),
        "high_to_low_confusion": high_to_low_confusion,
        "low_to_high_confusion": low_to_high_confusion,
        "medium_recall": round(per_class["MEDIUM"]["recall"], 4),
        "medium_precision": round(per_class["MEDIUM"]["precision"], 4),
        "medium_f1": round(per_class["MEDIUM"]["f1"], 4),
        "low_recall": round(per_class["LOW"]["recall"], 4),
        "low_precision": round(per_class["LOW"]["precision"], 4),
        "low_f1": round(per_class["LOW"]["f1"], 4),
        "confusion_matrix": cm,
        "per_class_details": per_class,
    }


def compute_multiclass_brier_and_ece(
    y_true_labels: pd.Series,
    predicted_proba: np.ndarray,
    n_bins: int = 10,
) -> Dict[str, Any]:
    """
    Computes multiclass Brier score and Expected Calibration Error (ECE)
    along with reliability tables for LOW, MEDIUM, and HIGH classes.
    """
    y_true_num = np.array([VOLATILITY_LABEL_MAP[lbl] for lbl in y_true_labels])
    n_classes = len(VOLATILITY_CLASSES)
    n_samples = len(y_true_num)

    # 1. Multiclass Brier Score: 1/N * sum_i sum_c (p_ic - y_ic)^2
    # One-hot true labels
    y_one_hot = np.zeros((n_samples, n_classes))
    for c in range(n_classes):
        y_one_hot[:, c] = (y_true_num == c).astype(float)

    brier_score = float(np.mean(np.sum((predicted_proba - y_one_hot) ** 2, axis=1)))

    # 2. Class-wise Expected Calibration Error (ECE) and reliability tables
    class_ece = {}
    reliability_tables = {}

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)

    for c_idx, c_name in enumerate(VOLATILITY_CLASSES):
        p_c = predicted_proba[:, c_idx]
        y_c = y_one_hot[:, c_idx]

        bin_indices = np.digitize(p_c, bin_edges[1:-1]) # bins 0 to n_bins - 1

        ece_c = 0.0
        bin_records = []

        for b in range(n_bins):
            in_bin = (bin_indices == b)
            bin_count = int(np.sum(in_bin))
            if bin_count > 0:
                bin_acc = float(np.mean(y_c[in_bin]))
                bin_conf = float(np.mean(p_c[in_bin]))
                ece_c += (bin_count / n_samples) * abs(bin_acc - bin_conf)
                bin_records.append({
                    "bin": b,
                    "bin_range": f"[{bin_edges[b]:.1f}, {bin_edges[b+1]:.1f})",
                    "count": bin_count,
                    "mean_predicted_prob": round(bin_conf, 4),
                    "observed_frequency": round(bin_acc, 4),
                    "abs_gap": round(abs(bin_acc - bin_conf), 4),
                })
            else:
                bin_records.append({
                    "bin": b,
                    "bin_range": f"[{bin_edges[b]:.1f}, {bin_edges[b+1]:.1f})",
                    "count": 0,
                    "mean_predicted_prob": 0.0,
                    "observed_frequency": 0.0,
                    "abs_gap": 0.0,
                })

        class_ece[c_name] = round(float(ece_c), 4)
        reliability_tables[c_name] = bin_records

    # Overall macro ECE across classes
    macro_ece = round(float(np.mean(list(class_ece.values()))), 4)

    return {
        "multiclass_brier_score": round(brier_score, 4),
        "macro_ece": macro_ece,
        "class_ece": class_ece,
        "reliability_tables": reliability_tables,
    }


def compute_market_level_metrics(
    eval_df: pd.DataFrame,
    mandi_col: str,
    y_true_col: str,
    y_pred_col: str,
    y_base_col: str,
    min_obs_per_mandi: int = 20,
) -> Dict[str, Any]:
    """
    Computes per-mandi metrics, compares against baseline per mandi,
    and performs cluster-robust statistical tests.
    """
    mandis = eval_df[mandi_col].unique()
    mandi_records = []

    for m in mandis:
        sub = eval_df[eval_df[mandi_col] == m].dropna(subset=[y_true_col, y_pred_col, y_base_col])
        if len(sub) < min_obs_per_mandi:
            continue

        yt = sub[y_true_col]
        yp = sub[y_pred_col]
        yb = sub[y_base_col]

        # ML metrics
        acc_ml = float(accuracy_score(yt, yp))
        bacc_ml = float(balanced_accuracy_score(yt, yp))
        mf1_ml = float(f1_score(yt, yp, labels=VOLATILITY_CLASSES, average="macro", zero_division=0))
        prec_ml, rec_ml, f1_ml, _ = precision_recall_fscore_support(yt, yp, labels=VOLATILITY_CLASSES, zero_division=0)

        # Baseline metrics
        acc_b = float(accuracy_score(yt, yb))
        bacc_b = float(balanced_accuracy_score(yt, yb))
        mf1_b = float(f1_score(yt, yb, labels=VOLATILITY_CLASSES, average="macro", zero_division=0))
        prec_b, rec_b, f1_b, _ = precision_recall_fscore_support(yt, yb, labels=VOLATILITY_CLASSES, zero_division=0)

        high_idx = VOLATILITY_LABEL_MAP["HIGH"]

        mandi_records.append({
            "market_name": m,
            "n_samples": len(sub),
            # ML
            "ml_accuracy": round(acc_ml, 4),
            "ml_balanced_acc": round(bacc_ml, 4),
            "ml_macro_f1": round(mf1_ml, 4),
            "ml_high_recall": round(float(rec_ml[high_idx]), 4),
            "ml_high_precision": round(float(prec_ml[high_idx]), 4),
            # Baseline
            "base_accuracy": round(acc_b, 4),
            "base_balanced_acc": round(bacc_b, 4),
            "base_macro_f1": round(mf1_b, 4),
            "base_high_recall": round(float(rec_b[high_idx]), 4),
            "base_high_precision": round(float(prec_b[high_idx]), 4),
            # Deltas (ML - Base)
            "delta_macro_f1": round(mf1_ml - mf1_b, 4),
            "delta_balanced_acc": round(bacc_ml - bacc_b, 4),
            "delta_high_recall": round(float(rec_ml[high_idx] - rec_b[high_idx]), 4),
            "ml_wins_macro_f1": int(mf1_ml > mf1_b),
            "base_wins_macro_f1": int(mf1_b > mf1_ml),
            "tie_macro_f1": int(mf1_ml == mf1_b),
        })

    mdf = pd.DataFrame(mandi_records)
    if len(mdf) == 0:
        return {"error": "No mandis with sufficient observations"}

    total_mandis = len(mdf)
    ml_wins = int(mdf["ml_wins_macro_f1"].sum())
    base_wins = int(mdf["base_wins_macro_f1"].sum())
    ties = int(mdf["tie_macro_f1"].sum())

    # Disaggregate by mandi volume tiers
    # Tier 1 (Small): <= 25th percentile of mandi size
    # Tier 2 (Medium): 25th to 75th percentile
    # Tier 3 (Large): > 75th percentile
    q25 = mdf["n_samples"].quantile(0.25)
    q75 = mdf["n_samples"].quantile(0.75)

    tier_small = mdf[mdf["n_samples"] <= q25]
    tier_med = mdf[(mdf["n_samples"] > q25) & (mdf["n_samples"] <= q75)]
    tier_large = mdf[mdf["n_samples"] > q75]

    def tier_stats(sub_tier):
        if len(sub_tier) == 0:
            return {}
        return {
            "mandi_count": len(sub_tier),
            "mean_samples": round(float(sub_tier["n_samples"].mean()), 1),
            "mean_ml_macro_f1": round(float(sub_tier["ml_macro_f1"].mean()), 4),
            "mean_base_macro_f1": round(float(sub_tier["base_macro_f1"].mean()), 4),
            "mean_delta_macro_f1": round(float(sub_tier["delta_macro_f1"].mean()), 4),
            "ml_wins": int(sub_tier["ml_wins_macro_f1"].sum()),
            "base_wins": int(sub_tier["base_wins_macro_f1"].sum()),
            "ml_win_rate": round(float(sub_tier["ml_wins_macro_f1"].mean() * 100.0), 1),
        }

    # Clustered Statistical Significance Testing
    # Differences across mandis: d_m = ML_m - Base_m
    diffs = mdf["delta_macro_f1"].to_numpy()

    # 1. Clustered Paired t-test (accounting for mandi cluster)
    mean_diff = float(np.mean(diffs))
    std_diff = float(np.std(diffs, ddof=1))
    se_diff = std_diff / np.sqrt(total_mandis)
    t_stat = mean_diff / (se_diff + 1e-12)
    p_val_t = float(2.0 * (1.0 - student_t.cdf(abs(t_stat), df=total_mandis - 1)))

    # 2. Wilcoxon Signed-Rank Test across mandis (non-parametric)
    # Check if non-zero diffs exist
    non_zero_diffs = diffs[diffs != 0]
    if len(non_zero_diffs) >= 10:
        w_stat, p_val_w = wilcoxon(diffs, zero_method="pratt")
        w_stat, p_val_w = float(w_stat), float(p_val_w)
    else:
        w_stat, p_val_w = None, None

    # 3. Clustered Bootstrap (Resampling entire mandis with replacement, B=2000)
    np.random.seed(42)
    B = 2000
    boot_means = []
    for _ in range(B):
        sample_indices = np.random.choice(len(diffs), size=len(diffs), replace=True)
        boot_means.append(np.mean(diffs[sample_indices]))
    boot_means = np.array(boot_means)
    ci_95_low = float(np.percentile(boot_means, 2.5))
    ci_95_high = float(np.percentile(boot_means, 97.5))
    p_val_boot = float(np.mean(boot_means <= 0.0) * 2.0 if mean_diff > 0 else np.mean(boot_means >= 0.0) * 2.0)
    p_val_boot = min(1.0, p_val_boot)

    # 4. Standard McNemar test over all pooled observations (with dependency caveat noted)
    corr_ml = (eval_df[y_pred_col] == eval_df[y_true_col]).to_numpy()
    corr_base = (eval_df[y_base_col] == eval_df[y_true_col]).to_numpy()
    b = int((corr_ml & ~corr_base).sum())
    c = int((~corr_ml & corr_base).sum())
    if b + c > 0:
        mcnemar_stat = float(((abs(b - c) - 1.0) ** 2) / (b + c))
        p_val_mcnemar = float(1.0 - chi2.cdf(mcnemar_stat, df=1))
    else:
        mcnemar_stat, p_val_mcnemar = 0.0, 1.0

    return {
        "total_mandis_evaluated": total_mandis,
        "mandi_records": mandi_records,
        "win_loss_summary": {
            "ml_wins": ml_wins,
            "base_wins": base_wins,
            "ties": ties,
            "ml_win_rate_pct": round(ml_wins / total_mandis * 100.0, 1),
            "base_win_rate_pct": round(base_wins / total_mandis * 100.0, 1),
        },
        "mandi_volume_tiers": {
            "small_mandis": tier_stats(tier_small),
            "medium_mandis": tier_stats(tier_med),
            "large_mandis": tier_stats(tier_large),
        },
        "clustered_significance_tests": {
            "mean_macro_f1_difference": round(mean_diff, 4),
            "cluster_robust_se": round(se_diff, 4),
            "paired_cluster_t_stat": round(t_stat, 4),
            "paired_cluster_t_pvalue": round(p_val_t, 5),
            "wilcoxon_signed_rank_stat": w_stat,
            "wilcoxon_pvalue": p_val_w,
            "clustered_bootstrap_95_ci": [round(ci_95_low, 4), round(ci_95_high, 4)],
            "clustered_bootstrap_pvalue": round(p_val_boot, 5),
            "ml_advantage_statistically_significant_at_05": bool(mean_diff > 0 and (ci_95_low > 0 or p_val_t < 0.05)),
            "base_advantage_statistically_significant_at_05": bool(mean_diff < 0 and (ci_95_high < 0 or p_val_t < 0.05)),
            "statistically_significant_at_05": bool(p_val_t < 0.05),
            "significance_winner": "ML" if (mean_diff > 0 and p_val_t < 0.05) else ("Persistence" if (mean_diff < 0 and p_val_t < 0.05) else "Neither (Parity)"),
        },
        "pooled_mcnemar_diagnostic": {
            "b_ml_correct_base_wrong": b,
            "c_base_correct_ml_wrong": c,
            "mcnemar_chi2": round(mcnemar_stat, 4),
            "mcnemar_pvalue": p_val_mcnemar,
            "caveat": "Standard McNemar assumes observation independence. Since observations within mandis are temporally correlated, cluster-robust tests above must be preferred.",
        },
    }
