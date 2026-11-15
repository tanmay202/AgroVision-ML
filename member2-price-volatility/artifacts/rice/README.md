# Rice Model Artifacts

This folder contains all machine learning model weights, feature configurations, residual distributions, and canonical group mappings for the **Rice** commodity.

## Files

1. `rice_price_model.pkl`: Production 48-feature XGBoost regressor predicting modal price changes ($\Delta p \in [-7, +7]$ days).
2. `rice_price_final_config.json`: Feature specification and metadata for the frozen price model.
3. `rice_price_model_14d.pkl`: Direct 14-day calendar horizon XGBoost regressor ($[t+14, t+17]$ days).
4. `rice_price_model_14d_config.json`: Configuration for the 14-day horizon model.
5. `rice_price_model_30d.pkl`: Direct 30-day calendar horizon XGBoost regressor ($[t+30, t+33]$ days).
6. `rice_price_model_30d_config.json`: Configuration for the 30-day horizon model.
7. `rice_volatility_normalized_model.pkl`: Champion 35-feature XGBoost classifier for price-normalized volatility (v2).
8. `rice_volatility_normalized_config.json`: Training-derived percentage thresholds ($0.014286\%$, $1.960784\%$) and metadata.
9. `rice_volatility_model.pkl`: Legacy nominal range persistence volatility classifier.
10. `rice_group_mappings.json`: Canonical market (87) and variety (34) strings from West Bengal Agmarknet records.
11. `rice_holdout_residuals*.json`: Empirical holdout residual error distributions (1d, 7d, 14d, 30d) for conformal prediction intervals.
