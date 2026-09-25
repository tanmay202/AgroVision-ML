"""
AgroVision - Model Evaluator
Evaluation metrics, visualizations, and feature importance analysis.
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for saving plots
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    mean_absolute_percentage_error,
)

from src.config import OUTPUTS_DIR, TARGET_COLUMN


def calculate_metrics(y_true, y_pred):
    """Calculate regression evaluation metrics."""
    metrics = {
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "MAE": mean_absolute_error(y_true, y_pred),
        "R2": r2_score(y_true, y_pred),
        "MAPE": mean_absolute_percentage_error(y_true, y_pred) * 100,
    }
    return metrics


def print_evaluation_report(metrics, model_name):
    """Print a formatted evaluation report."""
    print(f"\n[STATS] Evaluation Report: {model_name}")
    print("-" * 40)
    print(f"   RMSE  : {metrics['RMSE']:.4f}")
    print(f"   MAE   : {metrics['MAE']:.4f}")
    print(f"   R2    : {metrics['R2']:.4f}")
    print(f"   MAPE  : {metrics['MAPE']:.2f}%")
    print("-" * 40)


def plot_actual_vs_predicted(y_true, y_pred, model_name, save_path=None):
    """Create Actual vs Predicted scatter plot."""
    fig, ax = plt.subplots(figsize=(8, 6))

    ax.scatter(y_true, y_pred, alpha=0.4, s=10, c="steelblue", edgecolors="none")

    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], "r--", linewidth=2, label="Perfect Prediction")

    ax.set_xlabel(f"Actual {TARGET_COLUMN}", fontsize=12)
    ax.set_ylabel(f"Predicted {TARGET_COLUMN}", fontsize=12)
    ax.set_title(f"Actual vs Predicted - {model_name}", fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"   [SAVE] {save_path}")
    plt.close()


def plot_residuals(y_true, y_pred, model_name, save_path=None):
    """Create residual distribution plot."""
    residuals = y_true - y_pred

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].scatter(y_pred, residuals, alpha=0.4, s=10, c="steelblue", edgecolors="none")
    axes[0].axhline(y=0, color="r", linestyle="--", linewidth=2)
    axes[0].set_xlabel(f"Predicted {TARGET_COLUMN}", fontsize=11)
    axes[0].set_ylabel("Residual", fontsize=11)
    axes[0].set_title("Residuals vs Predicted", fontsize=13)
    axes[0].grid(True, alpha=0.3)

    axes[1].hist(residuals, bins=50, color="steelblue", edgecolor="white", alpha=0.8)
    axes[1].axvline(x=0, color="r", linestyle="--", linewidth=2)
    axes[1].set_xlabel("Residual", fontsize=11)
    axes[1].set_ylabel("Frequency", fontsize=11)
    axes[1].set_title("Residual Distribution", fontsize=13)
    axes[1].grid(True, alpha=0.3)

    fig.suptitle(f"Residual Analysis - {model_name}", fontsize=14, y=1.02)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"   [SAVE] {save_path}")
    plt.close()


def plot_feature_importance(model, feature_names, model_name, top_n=15, save_path=None):
    """Plot feature importance from tree-based models."""
    if not hasattr(model, "feature_importances_"):
        print(f"   [WARN] Model {model_name} does not support feature importance")
        return None

    importances = model.feature_importances_
    importance_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": importances,
    }).sort_values("Importance", ascending=True).tail(top_n)

    fig, ax = plt.subplots(figsize=(10, max(6, top_n * 0.4)))
    ax.barh(importance_df["Feature"], importance_df["Importance"], color="steelblue", edgecolor="white")
    ax.set_xlabel("Importance", fontsize=12)
    ax.set_title(f"Top {top_n} Feature Importances - {model_name}", fontsize=14)
    ax.grid(True, axis="x", alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"   [SAVE] {save_path}")
    plt.close()

    return importance_df


def save_evaluation_report(metrics, model_name, save_path=None):
    """Save metrics to a CSV file."""
    report = pd.DataFrame([{"Model": model_name, **metrics}])
    if save_path:
        report.to_csv(save_path, index=False)
        print(f"   [SAVE] {save_path}")
    return report


def evaluate_model(results):
    """
    Full evaluation pipeline.
    Creates metrics report, plots, and feature importance.
    """
    print("\n" + "=" * 60)
    print("STEP 5: MODEL EVALUATION")
    print("=" * 60)

    os.makedirs(OUTPUTS_DIR, exist_ok=True)

    best_name = results["_best_name"]
    best_model = results["_best_model"]
    X_test = results["_X_test"]
    y_test = results["_y_test"]
    feature_names = results["_feature_names"]
    use_log_target = results.get("_use_log_target", False)

    # Predictions
    y_pred = best_model.predict(X_test)

    # Inverse log transform for metrics in original scale
    if use_log_target:
        y_test_orig = np.expm1(y_test)
        y_pred_orig = np.expm1(y_pred)
        print("   [INFO] Inverse log-transforming predictions for evaluation")
    else:
        y_test_orig = y_test
        y_pred_orig = y_pred

    # Metrics (in original scale)
    metrics = calculate_metrics(y_test_orig, y_pred_orig)
    print_evaluation_report(metrics, best_name)

    # Plots (in original scale)
    print("\n[PLOT] Generating evaluation plots...")

    plot_actual_vs_predicted(
        y_test_orig, y_pred_orig, best_name,
        save_path=os.path.join(OUTPUTS_DIR, "actual_vs_predicted.png"),
    )

    plot_residuals(
        y_test_orig, y_pred_orig, best_name,
        save_path=os.path.join(OUTPUTS_DIR, "residual_analysis.png"),
    )

    importance_df = plot_feature_importance(
        best_model, feature_names, best_name,
        save_path=os.path.join(OUTPUTS_DIR, "feature_importance.png"),
    )

    save_evaluation_report(
        metrics, best_name,
        save_path=os.path.join(OUTPUTS_DIR, "evaluation_report.csv"),
    )

    # All models comparison
    comparison = []
    for name in ["Linear Regression", "Random Forest", "Gradient Boosting", "XGBoost"]:
        if name in results:
            model = results[name]["model"]
            preds = model.predict(X_test)
            if use_log_target:
                preds_orig = np.expm1(preds)
            else:
                preds_orig = preds
            m = calculate_metrics(y_test_orig, preds_orig)
            m["Model"] = name
            comparison.append(m)

    if comparison:
        comp_df = pd.DataFrame(comparison)
        comp_path = os.path.join(OUTPUTS_DIR, "model_comparison.csv")
        comp_df.to_csv(comp_path, index=False)
        print(f"\n[SAVE] Saved model comparison to: {comp_path}")
        print("\n[STATS] Model Comparison:")
        print(comp_df.to_string(index=False))

    print(f"\n[OK] All evaluation outputs saved to: {OUTPUTS_DIR}")
