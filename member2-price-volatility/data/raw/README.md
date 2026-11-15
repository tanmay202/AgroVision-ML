# AgroVision-ML — Member 2: Raw Datasets Catalog

This directory contains historical agricultural market price and arrival datasets ingested from **Agmarknet (Directorate of Marketing & Inspection, Ministry of Agriculture & Farmers Welfare, Government of India)**.

---

## 1. Directory Structure

Raw datasets are organized with separated commodity folders while maintaining standard root CSV files:

```text
data/raw/
├── rice/
│   └── rice.csv                  ← Raw Rice records (315,079 observations)
├── wheat/
│   └── wheat.csv                 ← Raw Wheat records (64,364 observations)
├── potato/
│   └── potato.csv                ← Raw Potato records (265,047 observations)
├── tea/
│   └── tea.csv                   ← Raw Tea records (6,432 observations)
├── Potato.csv                    ← Root symlink / copy (backward compatibility)
├── rice.csv                      ← Root file (backward compatibility)
├── Wheat.csv                     ← Root symlink / copy (backward compatibility)
├── tea.csv                       ← Root file (backward compatibility)
└── README.md                     ← This document
```

---

## 2. Ingestion Schema Specification

All commodity datasets conform to the standardized Agmarknet time-series schema:

| Column Name | Data Type | Description | Example |
|:---|:---:|:---|:---|
| `State Name` | String | Indian state | `"West Bengal"` |
| `District Name` | String | Administrative district | `"Burdwan"`, `"Hooghly"`, `"Bankura"` |
| `Market Name` | String | APMC Mandi wholesale market | `"Burdwan"`, `"Champadanga"`, `"Asansol"` |
| `Variety` | String | Commercial variety or grade | `"Common"`, `"Jyoti"`, `"Kalyan"` |
| `Group` | String | Agricultural commodity group | `"Cereals"`, `"Vegetables"` |
| `Arrivals (Tonnes)` | Float | Physical daily mandi trading volume | `120.0` |
| `Min Price (Rs./Quintal)` | Float | Minimum reported daily transaction price | `2400.0` |
| `Max Price (Rs./Quintal)` | Float | Maximum reported daily transaction price | `2550.0` |
| `Modal Price (Rs./Quintal)` | Float | Most frequent daily transaction price (target) | `2500.0` |
| `Reported Date` | String | Observation date (`DD Mon YYYY` or `YYYY-MM-DD`) | `"23 Dec 2015"` |

---

## 3. Commodity Dataset Profiles

### 🌾 1. Rice (`data/raw/rice/rice.csv`)
- **Total Records:** 315,079 valid records (319,092 raw cleaned rows)
- **Date Range:** 2001-03-29 to 2024-02-01
- **Mandis:** 87 unique West Bengal wholesale markets
- **Varieties:** 34 varieties (Common, Fine, Super Fine, Swarna, Minikit, Gobindobhog, etc.)
- **Average Daily Modal Price:** ₹2,450 / Quintal

### 🌾 2. Wheat (`data/raw/wheat/wheat.csv` / `Wheat.csv`)
- **Total Records:** 64,364 valid records
- **Date Range:** 2001-03-16 to 2024-02-01
- **Mandis:** 38 unique West Bengal wholesale markets
- **Varieties:** 16 varieties (Kalyan, Sonalika, Deshi, WL711, Other, Sona, PBW-343, etc.)
- **Average Daily Modal Price:** ₹1,720 / Quintal

### 🥔 3. Potato (`data/raw/potato/potato.csv` / `Potato.csv`)
- **Total Records:** 265,047 valid records
- **Date Range:** 2002-01-18 to 2024-02-02
- **Mandis:** 88 unique West Bengal wholesale markets
- **Varieties:** 13 varieties (Jyoti, Chandermukhi, Desi, Local, F.A.Q., Red, Shimla, etc.)
- **Average Daily Modal Price:** ₹890 / Quintal

### 🍵 4. Tea (`data/raw/tea/tea.csv`)
- **Total Records:** 6,432 valid records
- **Date Range:** 2001-03-20 to 2023-11-15
- **Mandis:** 13 wholesale auction markets in North Bengal
- **Varieties:** 4 commercial grades
- **Average Daily Modal Price:** ₹260 / Quintal
