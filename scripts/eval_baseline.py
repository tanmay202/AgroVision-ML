import json
import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import (
    mean_absolute_error,
    root_mean_squared_error,
    r2_score,
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_recall_curve,
    auc,
    classification_report,
    confusion_matrix
)
from pathlib import Path

def eval_baseline(commodity):
    print(f"\n==========================================")
    print(f"=== EVALUATING {commodity.upper()} V1 BASELINE ===")
    print(f"==========================================")
    proc_dir = Path(f"member2-price-volatility/data/processed/{commodity}")
    art_dir = Path(f"member2-price-volatility/artifacts/{commodity}")
    
    # Raw & cleaned counts
    raw_df = pd.read_csv(f"member2-price-volatility/data/raw/{commodity}/{commodity}.csv")
    clean_df = pd.read_csv(proc_dir / f"{commodity}_cleaned.csv")
    print(f"Raw rows: {len(raw_df)}")
    print(f"Clean rows: {len(clean_df)}")
    print(f"Markets: {clean_df['Market Name'].nunique()}")
    print(f"Varieties: {clean_df['Variety'].nunique()}")
    
    # Train & test
    train_df = pd.read_csv(proc_dir / f"{commodity}_train.csv")
    test_df = pd.read_csv(proc_dir / f"{commodity}_test.csv")
    print(f"Train rows: {len(train_df)}")
    print(f"Test rows: {len(test_df)}")
    
    # Price model
    price_model = joblib.load(art_dir / f"{commodity}_price_model.pkl")
    feat_cfg = json.loads((art_dir / f"{commodity}_feature_config.json").read_text())
    if "price_features" in feat_cfg:
        features = feat_cfg["price_features"]
    elif "price_model" in feat_cfg and "features" in feat_cfg["price_model"]:
        features = feat_cfg["price_model"]["features"]
    else:
        features = feat_cfg.get("features", [])
    
    current_price = test_df["Modal Price (Rs./Quintal)"]
    if "future_modal_price" in test_df.columns:
        actual_future_price = test_df["future_modal_price"]
        y_test_change = actual_future_price - current_price
    else:
        y_test_change = test_df["future_price_change"]
        actual_future_price = current_price + y_test_change
    
    pred_change = price_model.predict(test_df[features])
    pred_future_price = current_price + pred_change
    persistence_price = current_price
    
    mae = mean_absolute_error(actual_future_price, pred_future_price)
    rmse = root_mean_squared_error(actual_future_price, pred_future_price)
    r2 = r2_score(actual_future_price, pred_future_price)
    mape = np.mean(np.abs((actual_future_price - pred_future_price) / actual_future_price)) * 100
    
    p_mae = mean_absolute_error(actual_future_price, persistence_price)
    p_rmse = root_mean_squared_error(actual_future_price, persistence_price)
    p_r2 = r2_score(actual_future_price, persistence_price)
    p_mape = np.mean(np.abs((actual_future_price - persistence_price) / actual_future_price)) * 100
    
    print("\n--- Price Model ---")
    print(f"  MAE: Rs.{mae:.2f}")
    print(f"  RMSE: Rs.{rmse:.2f}")
    print(f"  R2: {r2:.4f}")
    print(f"  MAPE: {mape:.2f}%")
    print("\n--- Persistence Baseline ---")
    print(f"  MAE: Rs.{p_mae:.2f}")
    print(f"  RMSE: Rs.{p_rmse:.2f}")
    print(f"  R2: {p_r2:.4f}")
    print(f"  MAPE: {p_mape:.2f}%")
    
    # Movement rows only
    mv_mask = actual_future_price != persistence_price
    print(f"\nMovement rows: {mv_mask.sum()} / {len(test_df)} ({mv_mask.mean()*100:.1f}%)")
    if mv_mask.sum() > 0:
        mv_mae = mean_absolute_error(actual_future_price[mv_mask], pred_future_price[mv_mask])
        mv_rmse = root_mean_squared_error(actual_future_price[mv_mask], pred_future_price[mv_mask])
        mv_p_mae = mean_absolute_error(actual_future_price[mv_mask], persistence_price[mv_mask])
        mv_p_rmse = root_mean_squared_error(actual_future_price[mv_mask], persistence_price[mv_mask])
        print(f"  Movement MAE: Rs.{mv_mae:.2f} (Persistence: Rs.{mv_p_mae:.2f})")
        print(f"  Movement RMSE: Rs.{mv_rmse:.2f} (Persistence: Rs.{mv_p_rmse:.2f})")
    
    # Volatility model
    vol_model = joblib.load(art_dir / f"{commodity}_volatility_model.pkl")
    vol_cfg = json.loads((art_dir / f"{commodity}_volatility_config.json").read_text())
    vol_test = pd.read_csv(proc_dir / f"{commodity}_volatility_dataset_test.csv")
    if "features" in vol_cfg:
        vol_features = vol_cfg["features"]
    elif "volatility_model" in feat_cfg and "features" in feat_cfg["volatility_model"]:
        vol_features = feat_cfg["volatility_model"]["features"]
    elif "volatility_features" in feat_cfg:
        vol_features = feat_cfg["volatility_features"]
    else:
        vol_features = [c for c in vol_test.columns if c not in ["Reported Date", "future_volatility_class", "future_volatility"]]
    y_vol_test = vol_test["volatility"] if "volatility" in vol_test.columns else vol_test["future_volatility_class"]
    
    vol_preds = vol_model.predict(vol_test[vol_features])
    vol_probs = vol_model.predict_proba(vol_test[vol_features])
    
    mapping = {"LOW": 0, "MEDIUM": 1, "HIGH": 2}
    inv_mapping = {0: "LOW", 1: "MEDIUM", 2: "HIGH"}
    
    if isinstance(y_vol_test.iloc[0], str):
        y_vol_test_num = y_vol_test.map(mapping)
    else:
        y_vol_test_num = y_vol_test
        
    if isinstance(vol_preds[0], (int, np.integer)):
        vol_preds_str = [inv_mapping.get(p, str(p)) for p in vol_preds]
        vol_preds_num = vol_preds
    else:
        vol_preds_str = vol_preds
        vol_preds_num = [mapping.get(p, p) for p in vol_preds]
        
    y_vol_test_str = y_vol_test_num.map(inv_mapping)

    acc = accuracy_score(y_vol_test_str, vol_preds_str)
    bal_acc = balanced_accuracy_score(y_vol_test_str, vol_preds_str)
    macro_f1 = f1_score(y_vol_test_str, vol_preds_str, average="macro")
    mcc = matthews_corrcoef(y_vol_test_str, vol_preds_str)
    
    # HIGH PR-AUC
    # high class is 2 or "HIGH"
    high_binary = (y_vol_test_str == "HIGH").astype(int)
    high_idx = 2 if len(vol_probs[0]) > 2 else 1
    high_prob = vol_probs[:, high_idx]
    precision_curve, recall_curve, _ = precision_recall_curve(high_binary, high_prob)
    high_pr_auc = auc(recall_curve, precision_curve)
    
    print("\n--- Volatility Model ---")
    print(f"  Accuracy: {acc*100:.2f}%")
    print(f"  Balanced Accuracy: {bal_acc*100:.2f}%")
    print(f"  Macro F1: {macro_f1:.4f}")
    print(f"  MCC: {mcc:.4f}")
    print(f"  HIGH PR-AUC: {high_pr_auc:.4f}")
    print("\n  Test class distribution:")
    print(y_vol_test_str.value_counts(normalize=True).apply(lambda x: f"{x*100:.2f}%").to_dict())
    print("  Predicted class distribution:")
    print(pd.Series(vol_preds_str).value_counts(normalize=True).apply(lambda x: f"{x*100:.2f}%").to_dict())
    print("\nClassification Report:")
    print(classification_report(y_vol_test_str, vol_preds_str, digits=4, labels=["LOW", "MEDIUM", "HIGH"]))
    print("Confusion Matrix (rows=true, cols=pred):")
    print(confusion_matrix(y_vol_test_str, vol_preds_str, labels=["LOW", "MEDIUM", "HIGH"]))

if __name__ == "__main__":
    eval_baseline("wheat")
    eval_baseline("potato")
