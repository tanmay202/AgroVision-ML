# Potato Model Artifacts

This folder contains all machine learning model weights, feature configurations, and canonical group mappings for the **Potato** commodity.

## Files

1. `potato_price_model.pkl`: Trained 20-feature XGBoost regressor predicting modal price changes ($\Delta p = p_{t+1} - p_t$).
2. `potato_volatility_model.pkl`: Trained 19-feature class-balanced XGBoost classifier predicting 5-observation forward volatility regime (`LOW`, `MEDIUM`, `HIGH`).
3. `potato_feature_config.json`: Feature specification, price model parameters, and test benchmark metrics (Accuracy: 50.88%, Balanced Accuracy: 44.42%).
4. `potato_volatility_config.json`: Volatility threshold mappings and classification hyper-parameters.
5. `potato_group_mappings.json`: Canonical market (88) and variety (13) strings across 19 West Bengal agricultural districts.
