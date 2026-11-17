# AgroVision-ML — Complete Documentation Index (`docs/`)

> **Master navigation and technical documentation catalog for the AgroVision-ML agricultural intelligence platform.**

---

## 1. System Overview & Core Architecture

- [**Root README (`README.md`)**](../README.md): Master project landing page, executive overview, architecture diagram, 3-crop model registry, quickstart guide, and full technology stack.
- [**Staging & Application Integration Guide (`staging/README.md`)**](../staging/README.md): FastAPI service specifications, request/response formats, staging scripts, and environment configuration.
- [**Operational Deployment Guide (`reports/deployment/README.md`)**](../reports/deployment/README.md): Staging architecture, SQLite persistent databases, telemetry, disaster recovery, and cloud production prerequisites.

---

## 2. Crop-Specific Intelligence Guides

Unified documentation covering pre-harvest biological yield forecasting (Member 1) and post-harvest wholesale mandi market intelligence (Member 2):

- [🌾 **Multi-Crop Directory (`commodities/README.md`)**](../commodities/README.md): Overview of multi-crop agricultural decision support.
- [🌾 **Rice (*Oryza sativa*) Complete Guide (`commodities/rice/README.md`)**](../commodities/rice/README.md): Aus, Aman, Boro biological yield models, 87-mandi price forecasting, 14D/30D horizons, and normalized volatility.
- [🌾 **Wheat (*Triticum aestivum*) Complete Guide (`commodities/wheat/README.md`)**](../commodities/wheat/README.md): Rabi wheat yield modeling, 38-mandi price regressor, and class-balanced volatility classifier.
- [🥔 **Potato (*Solanum tuberosum*) Complete Guide (`commodities/potato/README.md`)**](../commodities/potato/README.md): Rabi potato tuber yield modeling, Huber robustness, tree-saturation resolution, and 88-mandi price forecasting.

---

## 3. Machine Learning Engineering Modules

- [**Member 1: Multi-Crop Yield Intelligence Engine (`member1-yield-model/README.md`)**](../member1-yield-model/README.md): Complete engineering documentation for Rice, Wheat, and Potato yield pipelines, NASA POWER automated weather service, 20-feature/18-feature/19-feature orders, and statistical validation holdouts.
- [**Member 2: Agricultural Market Intelligence Engine (`member2-price-volatility/README.md`)**](../member2-price-volatility/README.md): Point-in-time mandi price forecasting, volatility classification, horizon governance, and Agmarknet data ingestion.
- [**Member 2 Multi-Commodity Guide (`member2-price-volatility/commodities/README.md`)**](../member2-price-volatility/commodities/README.md): Clean commodity separation across Rice, Wheat, and Potato.

---

## 4. Verification & Milestone Audit Reports

- [**Milestone 7 Multi-Crop Production Readiness Report (`reports/member1/M7_Multi_Crop_Production_Readiness.md`)**](../reports/member1/M7_Multi_Crop_Production_Readiness.md): Master release sign-off for Wheat and Potato production models.
- [**Milestone 7 Production Artifact Registry (`reports/member1/M7_Production_Artifact_Registry.md`)**](../reports/member1/M7_Production_Artifact_Registry.md): Cryptographic SHA-256 artifact verification.
- [**Documentation Modernization Report (`reports/documentation/README_Modernization_Report.md`)**](../reports/documentation/README_Modernization_Report.md): Full audit of repository documentation changes, corrections, and scope verification.
