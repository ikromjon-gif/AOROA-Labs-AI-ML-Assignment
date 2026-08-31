# AOROA Labs — AI/ML Assignment

> **English:** End-to-end e-commerce analytics, customer segmentation, and 7-day sales forecasting using leakage-free rolling backtests.
>
> **한국어:** 이커머스 데이터를 기반으로 EDA, 고객 세분화, 시계열 예측 및 누수 없는 롤링 백테스트를 수행한 AI/ML 분석 프로젝트입니다.

## Project Overview

This repository contains an end-to-end AI/ML workflow for an e-commerce dataset, covering exploratory data analysis (EDA), customer segmentation, sales forecasting, rolling backtesting, robustness analysis, and final forecast generation.

The workflow emphasizes leakage-free validation, interpretable segmentation, reproducible analysis, and business-oriented conclusions.

## Objectives

1. Audit and understand the transaction data.
2. Analyze daily sales trends and weekday seasonality.
3. Segment customers using behavioral and monetary features.
4. Compare baseline and feature-based forecasting approaches.
5. Evaluate models using rolling historical backtests.
6. Select a strong and reasonably simple forecasting approach.
7. Produce a submission-ready 7-day forecast.

## Dataset and Analysis Scope

The project uses the provided e-commerce transaction dataset. Derived datasets and analysis outputs are stored under `results/`.

The daily sales series covers the historical period used for forecasting through **2024-06-02**, with the final forecast covering **2024-06-03 to 2024-06-09**.

## Day 1 — EDA and Sales Analysis

The EDA examined transaction-level distributions and the daily sales series. The analysis identified substantial day-to-day variability and a clear weekly pattern, with stronger expected sales on weekends.

The project established a **56-day weekday-average forecast** as the primary benchmark for subsequent model evaluation.

The repository includes EDA outputs such as:

- `results/eda_statistics.csv`
- `results/missing_values.csv`
- `results/data_audit.csv`
- `results/daily_sales.csv`
- `results/weekday_eda.csv`

## Day 2 — Customer Segmentation

### Feature engineering

Customer-level features include:

- Recency
- Frequency
- Monetary value
- Average transaction value
- Average discount rate
- Average app usage time

The clustering workflow applied appropriate transformations and standardization to behavioral features. Highly skewed monetary/recency variables were transformed using `log1p` before clustering.

### Clustering methodology

K-Means clustering was evaluated for **K = 2 to 8** using multiple complementary criteria, including silhouette, Calinski-Harabasz, Davies-Bouldin, elbow analysis, and stability across random seeds.

The final workflow selected **K = 4**, producing four interpretable customer segments.

### Final segments

| Segment | Customers | Customer Share | Monetary Share | Interpretation |
| --- | ---: | ---: | ---: | --- |
| Cluster 0 | 360 | 45.00% | 44.68% | Regular / valuable customers |
| Cluster 1 | 238 | 29.75% | 20.73% | Discount-driven active customers |
| Cluster 2 | 94 | 11.75% | 27.05% | High-value, highly engaged customers |
| Cluster 3 | 108 | 13.50% | 7.54% | At-risk / dormant customers |

Cluster 2 is strategically important because it represents only **11.75% of customers** while contributing approximately **27.05% of total monetary value**.

Cluster 3 has an average recency of approximately **135 days**, indicating a clear reactivation opportunity.

### PCA visualization

The first two PCA components explain **72.26%** of the standardized feature variance:

- PCA1: **43.14%**
- PCA2: **29.12%**
- Combined: **72.26%**

## Day 3 — Sales Forecasting

### Validation design

Forecasting models were evaluated using rolling historical backtests with a **7-day forecast horizon**.

The final incomplete validation window was excluded so that every evaluated forecast day had observed ground truth. The evaluation therefore contains:

- **7 complete validation windows**
- **49 validation days**

Forecasting features were restricted to information available at each forecast cutoff. Lag and rolling statistics used historical values only, preventing future-data leakage.

### Models compared

#### 1. 56-day Weekday Baseline

For each forecast date, the model uses the average sales for the corresponding weekday over the most recent 56 historical days available at the cutoff.

#### 2. Trend + Weekday + Fourier

A regression-based model combining:

- Time trend
- Day-of-week effects
- Weekly Fourier seasonality terms

#### 3. Lag + Rolling

A Random Forest model using historical features including:

- `lag_1`
- `lag_7`
- `lag_14`
- `rolling_mean_7`
- `rolling_mean_14`
- `rolling_mean_28`
- Weekday
- Trend

Rolling statistics were constructed from shifted historical observations so that the target day's actual sales were not used as an input.

### Model comparison

| Model | MAE | RMSE | MAPE |
| --- | ---: | ---: | ---: |
| 56-day Weekday Baseline | 564,069.73 | 707,357.77 | 8.96% |
| Lag + Rolling | 562,520.32 | **692,877.38** | 8.86% |
| **Trend + Weekday + Fourier** | **551,328.20** | 702,445.75 | **8.53%** |

The **Trend + Weekday + Fourier** model achieved the best MAE and MAPE among the tested models. It improved MAPE by approximately **4.84% relative to the 56-day weekday baseline**.

Lag + Rolling achieved the lowest RMSE and had slightly lower MAPE variability across the validation windows, but its average MAE/MAPE were not as strong as the Fourier model.

### Robustness and model selection

The seven rolling validation windows show that model performance varies over time. The final selection was therefore based on aggregate performance across all validation days rather than a single window.

The selected model was the **Trend + Weekday + Fourier** approach because it provided the strongest overall MAE and MAPE while remaining relatively simple and interpretable.

## Final 7-Day Forecast

Using historical data through **2024-06-02**, the selected forecasting pipeline generated:

| Date | Weekday | Forecast |
| --- | --- | ---: |
| 2024-06-03 | Monday | 5,602,809.43 |
| 2024-06-04 | Tuesday | 5,627,813.93 |
| 2024-06-05 | Wednesday | 5,989,392.88 |
| 2024-06-06 | Thursday | 5,969,582.49 |
| 2024-06-07 | Friday | 6,028,289.22 |
| 2024-06-08 | Saturday | 7,509,605.19 |
| 2024-06-09 | Sunday | 7,528,744.72 |

The forecast retains the observed weekly pattern, with substantially higher predicted sales on Saturday and Sunday.

The final file is:

`results/submission_forecast.csv`

## Repository Structure

```text
AOROA-Labs-AI-ML-Assignment/
├── data/
├── figures/
│   ├── daily_sales_trend.png
│   ├── weekday_sales_pattern.png
│   ├── customer_segments_pca.png
│   ├── segment_revenue_contribution.png
│   ├── forecast_model_comparison.png
│   └── forecast_mape_comparison.png
├── notebooks/
│   ├── AOROA_Day1_Analysis.ipynb
│   ├── AOROA_Day2_Analysis.ipynb
│   └── AOROA_Day3_Analysis.ipynb
├── reports/
│   ├── Day2_Segmentation_Findings.md
│   └── EDA_Findings_Assumptions.md
├── results/
│   ├── baseline_validation_results.csv
│   ├── baseline_weekday_means.csv
│   ├── cluster_profile.csv
│   ├── cluster_size.csv
│   ├── cluster_stability.csv
│   ├── clustering_results_k2_k8.csv
│   ├── customer_feature_correlation.csv
│   ├── customer_features_raw.csv
│   ├── customer_segments.csv
│   ├── daily_sales.csv
│   ├── data_audit.csv
│   ├── eda_statistics.csv
│   ├── missing_values.csv
│   ├── segment_business_profile.csv
│   ├── segment_revenue_contribution.csv
│   ├── submission_forecast.csv
│   └── weekday_eda.csv
├── src/
│   ├── baseline.py
│   ├── forecasting_models.py
│   └── validation.py
└── experiment_log.md
```

## Reproducibility

The notebooks contain the complete analysis workflow, while reusable forecasting and validation logic is separated into the `src/` modules.

The overall pipeline is:

```text
Transaction data
      ↓
Data audit + EDA
      ↓
Customer feature engineering
      ↓
Customer segmentation
      ↓
Daily sales series
      ↓
Rolling backtests
      ↓
Model comparison
      ↓
Robustness check
      ↓
Final forecast
      ↓
submission_forecast.csv
```

## Key Deliverables

- Final 7-day forecast: `results/submission_forecast.csv`
- Customer segmentation outputs: `results/`
- EDA and segmentation reports: `reports/`
- Analysis and model figures: `figures/`
- Reproducible notebooks: `notebooks/`
- Reusable Python modules: `src/`

## Final Takeaways

1. The daily sales series exhibits a clear weekly pattern, with higher weekend demand.
2. Four customer segments provide useful behavioral and monetary differentiation.
3. The high-value segment is disproportionately important to total monetary contribution.
4. Rolling backtesting provides a stronger evaluation framework than relying on a single holdout split.
5. Trend + Weekday + Fourier achieved the best overall MAE and MAPE among the tested forecasting models.
6. The final 7-day forecast passed structural QA, with 7 rows, the expected columns, no missing values, no duplicate dates, and no negative forecasts.
