# AOROA Labs — AI/ML Assignment

## Project Overview

This repository contains an end-to-end analysis and modeling workflow for an e-commerce dataset, covering exploratory data analysis, customer segmentation, sales forecasting, validation, robustness checks, and final forecast generation.

The project was developed as a reproducible AI/ML workflow with a strong emphasis on leakage-free validation, interpretable results, and business-oriented conclusions.

## Objectives

1. Audit and understand the transaction dataset.
2. Analyze historical sales trends and weekday seasonality.
3. Segment customers using behavioral and monetary features.
4. Compare simple and more advanced forecasting approaches.
5. Select a strong but reasonably simple final forecasting model using rolling backtests.
6. Produce a submission-ready 7-day forecast.

## Dataset

The analysis uses the provided e-commerce transaction dataset.

- Transactions: **72,573**
- Unique customers: **800**
- Historical daily period: **2022-07-01 to 2024-06-02**
- Daily observations: **703**

The raw source dataset is not duplicated in this repository when it is not required for submission. Derived analysis outputs are stored under `results/`.

## Day 1 — EDA and Sales Baseline

### Data and sales findings

Daily sales show substantial day-to-day variability together with clear weekly seasonality. Weekend sales are consistently higher than weekday sales, making weekday-aware forecasting an important component of the modeling strategy.

The historical daily-sales statistics were:

- Minimum daily sales: **₩2,656,300**
- Maximum daily sales: **₩10,950,187**
- Mean daily sales: **₩5,647,876.38**
- Median daily sales: **₩5,469,311.00**
- Standard deviation: **₩1,258,926.86**

A 56-day weekday-average baseline was established as the primary benchmark for forecasting.

## Day 2 — Customer Segmentation

### Feature engineering

Customer-level features were built from the transaction data, including:

- Recency
- Frequency
- Monetary value
- Average transaction value
- Average discount rate
- Average app usage time
- Payment-method behavior

`recency_days` and `monetary` were log-transformed with `log1p` because their distributions were strongly right-skewed. Features were then standardized before clustering.

`payment_method_count` was excluded from the primary clustering model because it had very low variance and limited discriminative value.

### Clustering methodology

K-Means models were evaluated for **K = 2 to 8** using:

- Silhouette Score
- Calinski-Harabasz Score
- Davies-Bouldin Score
- Elbow analysis
- Stability across multiple random seeds

### Final segmentation

**K = 4** was selected. It achieved the strongest overall clustering metrics and remained highly stable across random initializations.

| Segment | Customers | Customer Share | Revenue Share | Business Interpretation |
| --- | ---: | ---: | ---: | --- |
| Regular Valuable | 360 | 45.00% | 44.68% | Stable, active, valuable customers | 
| Discount-Driven | 238 | 29.75% | 20.73% | Active customers with high discount usage | 
| High-Value Loyal | 94 | 11.75% | 27.05% | Highest-value and highest-engagement segment |
| At-Risk / Dormant | 108 | 13.50% | 7.54% | Long recency and lower activity |

The High-Value Loyal segment is particularly important: it represents only **11.75% of customers** but contributes approximately **27.05% of total monetary value**.

The At-Risk / Dormant segment has an average recency of approximately **135 days**, making it a clear candidate for win-back and reactivation strategies.

### PCA visualization

The first two PCA components explain **72.26%** of total variance:

- PCA1: 43.14%
- PCA2: 29.12%
- Combined: 72.26%

## Day 3 — Sales Forecasting

### Validation design

Forecasting models were evaluated with rolling historical backtests using a **7-day forecast horizon**.

To avoid incomplete ground truth, the final incomplete validation window was excluded. The final evaluation therefore used:

- **7 complete validation windows**
- **49 validation days**

All model features were restricted to information available at the forecasting cutoff to avoid future leakage.

### Models compared

#### 1. 56-day Weekday Baseline

A simple benchmark using the average sales for each weekday over the most recent 56 historical days available at each cutoff.

#### 2. Trend + Weekday + Fourier

A linear regression model combining:

- Time trend
- Day-of-week effects
- Weekly Fourier seasonality features

#### 3. Lag + Rolling

A Random Forest model using:

- 1-day lag
- 7-day lag
- 14-day lag
- 7-day rolling mean
- 14-day rolling mean
- 28-day rolling mean
- Weekday
- Trend

Rolling features were constructed with `shift(1)` so current-day actual sales were never used as forecasting inputs.

### Model comparison

| Model | MAE | RMSE | MAPE |
| --- | ---: | ---: | ---: |
| 56-day Weekday Baseline | 564,069.73 | 707,357.77 | 8.96% |
| Lag + Rolling | 562,520.32 | **692,877.38** | 8.86% |
| **Trend + Weekday + Fourier** | **551,328.20** | 702,445.75 | **8.53%** |

The **Trend + Weekday + Fourier** model was selected as the final model because it achieved the best overall MAE and MAPE, while maintaining reasonable robustness across rolling validation windows.

The Fourier model improved MAPE by approximately **4.84% relative to the baseline**.

### Robustness

Across the seven validation windows, model performance varied by period. The Lag + Rolling model showed slightly lower MAPE variability, while the Fourier model delivered the strongest average accuracy. The final choice therefore favors the simpler strong overall performer rather than optimizing for a single validation window.

## Final 7-Day Forecast

Using the complete historical dataset through **2024-06-02**, the selected Trend + Weekday + Fourier model produced the following forecast:

| Date | Weekday | Forecast |
| --- | --- | ---: |
| 2024-06-03 | Monday | 5,602,809.43 |
| 2024-06-04 | Tuesday | 5,627,813.93 |
| 2024-06-05 | Wednesday | 5,989,392.88 |
| 2024-06-06 | Thursday | 5,969,582.49 |
| 2024-06-07 | Friday | 6,028,289.22 |
| 2024-06-08 | Saturday | 7,509,605.19 |
| 2024-06-09 | Sunday | 7,528,744.72 |

The forecast preserves the observed weekly pattern, with higher expected sales on Saturday and Sunday.

## Repository Structure

```text
AOROA-Labs-AI-ML-Assignment/
├── data/
├── figures/
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

The notebooks in `notebooks/` contain the complete exploratory and modeling workflow used to generate the reported outputs. The reusable forecasting and validation logic is separated into the `src/` modules.

The workflow can be summarized as:

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

## Outputs

The primary deliverables are:

- `results/submission_forecast.csv`
- Customer segmentation outputs under `results/`
- EDA and segmentation reports under `reports/`
- Model and analysis figures under `figures/`
- Reproducible notebooks under `notebooks/`
- Reusable Python modules under `src/`

## Final Takeaways

1. The sales series contains a strong weekly pattern, especially higher weekend demand.
2. Customer behavior separates naturally into four interpretable segments.
3. The High-Value Loyal segment is disproportionately important for revenue.
4. A leakage-free rolling backtest is materially more informative than a single holdout split.
5. Trend + Weekday + Fourier provided the strongest overall forecasting accuracy among the tested approaches.
6. The final forecast has passed structural QA and is stored in `results/submission_forecast.csv`.
