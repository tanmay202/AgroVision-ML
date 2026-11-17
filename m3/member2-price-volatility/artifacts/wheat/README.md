# Wheat Model Artifacts

This folder contains all machine learning model weights, feature configurations, and canonical group mappings for the **Wheat** commodity.

## Files

1. `wheat_price_model.pkl`: Trained 20-feature XGBoost regressor predicting modal price changes ($\Delta p = p_{t+1} - p_t$).
2. `wheat_volatility_model.pkl`: Trained 19-feature class-balanced XGBoost classifier predicting 5-observation forward volatility regime (`LOW`, `MEDIUM`, `HIGH`).
3. `wheat_feature_config.json`: Feature specification, price model parameters, and test benchmark metrics (Accuracy: 63.49%, Balanced Accuracy: 60.37%, Macro F1: 0.6101, MCC: 0.4151).
4. `wheat_volatility_config.json`: Volatility threshold mappings and classification hyper-parameters.
5. `wheat_group_mappings.json`: Canonical market (38) and variety (16) strings across 15 West Bengal agricultural districts.
