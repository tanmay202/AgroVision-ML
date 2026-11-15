import sys
sys.path.insert(0, "member2-price-volatility/src")

import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score
from catboost import CatBoostRegressor
from xgboost import XGBRegressor

from features_v2 import generate_candidate_price_features

proc_dir = "member2-price-volatility/data/processed/potato"
clean_df = pd.read_csv(f"{proc_dir}/potato_cleaned.csv")
feats_df = generate_candidate_price_features(clean_df)

group_cols = ["Market Name", "Variety"]
date_col = "Reported Date"
price_col = "Modal Price (Rs./Quintal)"

feats_df[date_col] = pd.to_datetime(feats_df[date_col])
feats_df = feats_df.sort_values(group_cols + [date_col]).reset_index(drop=True)

feats_df["future_modal_price"] = feats_df.groupby(group_cols)[price_col].shift(-1)
next_date = feats_df.groupby(group_cols)[date_col].shift(-1)
feats_df["days_to_next"] = (next_date - feats_df[date_col]).dt.days

hist_mask = feats_df.groupby(group_cols).cumcount() >= 30
target_mask = feats_df["future_modal_price"].notna() & (feats_df["days_to_next"] >= 1) & (feats_df["days_to_next"] <= 7)

valid_df = feats_df[hist_mask & target_mask].copy().reset_index(drop=True)
valid_df["future_price_change"] = valid_df["future_modal_price"] - valid_df[price_col]

train_orig = pd.read_csv(f"{proc_dir}/potato_train.csv")
split_date = pd.to_datetime(train_orig[date_col]).max()

train_data = valid_df[valid_df[date_col] <= split_date].copy().reset_index(drop=True)
holdout_data = valid_df[valid_df[date_col] > split_date].copy().reset_index(drop=True)

cand_feats = [c for c in train_data.columns if c not in [
    "State Name", "District Name", "Market Name", "Variety", "Group",
    date_col, price_col, "Min Price (Rs./Quintal)", "Max Price (Rs./Quintal)",
    "future_modal_price", "days_to_next", "future_price_change"
]]

cur_price = holdout_data[price_col].to_numpy()
act_price = holdout_data["future_modal_price"].to_numpy()
p_mae = mean_absolute_error(act_price, cur_price)
p_rmse = root_mean_squared_error(act_price, cur_price)
print(f"Potato Persistence Holdout -> MAE: Rs.{p_mae:.2f}, RMSE: Rs.{p_rmse:.2f}", flush=True)

# Train CatBoost MAE
m_cat = CatBoostRegressor(iterations=400, depth=6, learning_rate=0.04, loss_function="MAE", verbose=0, random_seed=42)
m_cat.fit(train_data[cand_feats], train_data["future_price_change"])
pred_cat = m_cat.predict(holdout_data[cand_feats])
cat_p = cur_price + pred_cat
print(f"CatBoost MAE Holdout -> MAE: Rs.{mean_absolute_error(act_price, cat_p):.2f}, RMSE: Rs.{root_mean_squared_error(act_price, cat_p):.2f}", flush=True)

for eps in [1.0, 2.0, 3.0, 5.0]:
    filt = np.where(np.abs(pred_cat) < eps, 0.0, pred_cat)
    print(f"CatBoost Dead-band eps={eps} -> MAE: Rs.{mean_absolute_error(act_price, cur_price + filt):.2f}, RMSE: Rs.{root_mean_squared_error(act_price, cur_price + filt):.2f}", flush=True)

# Train XGBoost PseudoHuber
m_xgb = XGBRegressor(n_estimators=400, max_depth=5, learning_rate=0.03, objective="reg:pseudohubererror", subsample=0.85, colsample_bytree=0.85, random_state=42, n_jobs=-1)
m_xgb.fit(train_data[cand_feats], train_data["future_price_change"])
pred_xgb = m_xgb.predict(holdout_data[cand_feats])
xgb_p = cur_price + pred_xgb
print(f"XGBoost PseudoHuber Holdout -> MAE: Rs.{mean_absolute_error(act_price, xgb_p):.2f}, RMSE: Rs.{root_mean_squared_error(act_price, xgb_p):.2f}", flush=True)
for eps in [1.0, 2.0, 3.0, 5.0]:
    filt = np.where(np.abs(pred_xgb) < eps, 0.0, pred_xgb)
    print(f"XGBoost Dead-band eps={eps} -> MAE: Rs.{mean_absolute_error(act_price, cur_price + filt):.2f}, RMSE: Rs.{root_mean_squared_error(act_price, cur_price + filt):.2f}", flush=True)
