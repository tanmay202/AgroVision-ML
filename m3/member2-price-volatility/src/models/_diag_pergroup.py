"""Diagnose what per-group XGBoost actually predicts."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor

from config import DATE_COLUMN, GROUP_COLUMNS, PRICE_COLUMN, PRICE_TARGET, MIN_PRICE_COLUMN, MAX_PRICE_COLUMN, train_path, test_path

pc = PRICE_COLUMN

def impute_lags(df):
    df = df.copy()
    group_cols = [c for c in GROUP_COLUMNS if c in df.columns]
    fill_cols = ['lag_1','lag_7','lag_14','lag_30','rolling_mean_7','rolling_mean_14','rolling_mean_30','rolling_std_7','rolling_std_14','arrival_lag_1','arrival_lag_7','arrival_rolling_mean_7','arrival_rolling_mean_14','price_pct_change','arrival_pct_change']
    existing = [c for c in fill_cols if c in df.columns]
    if group_cols:
        df = df.sort_values(group_cols + [DATE_COLUMN])
        df[existing] = df.groupby(group_cols)[existing].transform(lambda s: s.ffill())
        df = df.reset_index(drop=True)
    for col in existing:
        df[col] = df[col].fillna(df[col].median())
    return df

train_df = pd.read_csv(train_path())
test_df = pd.read_csv(test_path())
train_df[DATE_COLUMN] = pd.to_datetime(train_df[DATE_COLUMN], errors='coerce')
test_df[DATE_COLUMN] = pd.to_datetime(test_df[DATE_COLUMN], errors='coerce')
train_df = train_df.dropna(subset=[PRICE_TARGET, pc]).copy()
test_df = test_df.dropna(subset=[PRICE_TARGET, pc]).copy()
train_df = impute_lags(train_df)
test_df = impute_lags(test_df)

if MIN_PRICE_COLUMN in train_df.columns:
    train_df['price_spread'] = train_df[MAX_PRICE_COLUMN] - train_df[MIN_PRICE_COLUMN]
    test_df['price_spread'] = test_df[MAX_PRICE_COLUMN] - test_df[MIN_PRICE_COLUMN]

train_df['future_price_change'] = train_df[PRICE_TARGET] - train_df[pc]
test_df['future_price_change'] = test_df[PRICE_TARGET] - test_df[pc]

features = [pc, MIN_PRICE_COLUMN, MAX_PRICE_COLUMN, 'lag_1','lag_7','lag_14','lag_30','rolling_mean_7','rolling_mean_14','rolling_mean_30','rolling_std_7','rolling_std_14','year','month','day','day_of_week','week_of_year','arrival_lag_1','arrival_lag_7','arrival_rolling_mean_7','arrival_rolling_mean_14','price_pct_change','arrival_pct_change','days_to_next','price_spread']
features = [f for f in features if f in train_df.columns and f in test_df.columns]

# Focus on Parappanangadi (41% zero, highest signal)
mkt, var = 'Parappanangadi', 'Other'
tr = train_df[(train_df['Market Name']==mkt) & (train_df['Variety']==var)].sort_values(DATE_COLUMN)
te = test_df[(test_df['Market Name']==mkt) & (test_df['Variety']==var)].sort_values(DATE_COLUMN)

print(f'Parappanangadi train: {len(tr)} rows, test: {len(te)} rows')
print(f'Train target -- mean: {tr["future_price_change"].mean():.1f}, std: {tr["future_price_change"].std():.1f}')
print(f'Pct zero (train): {(tr["future_price_change"]==0).mean()*100:.1f}%')

# Train on full group
model = XGBRegressor(n_estimators=500, max_depth=4, learning_rate=0.05, min_child_weight=2, subsample=0.8, colsample_bytree=0.8, reg_alpha=0.5, reg_lambda=2.0, objective='reg:pseudohubererror', random_state=42, n_jobs=-1)
model.fit(tr[features], tr['future_price_change'])

pred_change = model.predict(te[features])
print(f'\nPredicted change -- mean: {pred_change.mean():.1f}, std: {pred_change.std():.1f}, min: {pred_change.min():.1f}, max: {pred_change.max():.1f}')
print(f'Actual change    -- mean: {te["future_price_change"].mean():.1f}, std: {te["future_price_change"].std():.1f}')

pred_price = te[pc].values + pred_change
actual_price = te[PRICE_TARGET].values
baseline_price = te[pc].values

mae_model = mean_absolute_error(actual_price, pred_price)
mae_baseline = mean_absolute_error(actual_price, baseline_price)
print(f'\nBaseline MAE: {mae_baseline:.1f}')
print(f'Model MAE:    {mae_model:.1f}')
print(f'Improvement:  {(mae_baseline-mae_model)/mae_baseline*100:+.1f}%')

print('\nTop 10 features:')
imp = pd.DataFrame({'feature': features, 'importance': model.feature_importances_}).sort_values('importance', ascending=False)
print(imp.head(10).to_string(index=False))

print('\nSample predictions:')
df_out = pd.DataFrame({'date': te[DATE_COLUMN].values, 'current': te[pc].values, 'actual_future': actual_price, 'pred_future': pred_price.round(0), 'baseline_future': baseline_price})
print(df_out.head(20).to_string(index=False))
