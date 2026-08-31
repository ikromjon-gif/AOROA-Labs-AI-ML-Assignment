# AOROA Labs — AI/ML Assignment

> End-to-end e-commerce analytics, customer segmentation, leakage-free time-series forecasting, reproducible pipeline, and AOROA-focused business recommendations.

## Overview

This repository contains the completed AOROA Labs AI/ML assignment based on the provided e-commerce transaction dataset.

The project has two major components:

- **Problem 1 — AI/ML Theory & AOROA Private LLM:** variational inference, Transformer attention, diffusion vs GANs, LLM adaptation, and a proposed privacy-preserving AOROA private LLM architecture.
- **Problem 2 — Data Analysis & Modeling:** EDA, customer segmentation, time-series forecasting, rolling validation, robustness checks, final submissions, and a testable business action plan.

The implementation emphasizes reproducibility, chronological validation, leakage prevention, interpretable customer segments, and evidence-based business decisions.

## Dataset

Raw input:

`data/ecommerce_events.csv`

Verified dataset facts:

- 72,573 transactions
- 800 unique users
- 703 historical calendar days
- 2022-07-01 → 2024-06-02
- Columns: `date`, `user_id`, `payment_method`, `discount_rate`, `app_time_min`, `paid_amount`
- 2,314 missing values in `payment_method`
- No missing values in the other listed fields

## Problem 1 — AI/ML Theory & AOROA Private LLM

Final report:

`reports/AOROA_Labs_Problem_1_AI_ML_Theory_Private_LLM.pdf`

Topics:

1. Variational inference, KL divergence, ELBO, reparameterization, and diffusion connection.
2. Transformer self-attention, attention equations, scaling, MHA, KV cache, and GQA.
3. Diffusion vs GAN, consistency models, and flow matching.
4. LoRA/QLoRA, RAG, long-context/hybrid retrieval, SFT, RLHF, DPO, and engineering trade-offs.
5. Proposed AOROA private LLM architecture, RAG + LoRA/QLoRA strategy, evaluation framework, security/privacy, and A/B testing.

AOROA-specific architecture is presented as a recommendation where the supplied assignment material does not establish a current production implementation.

## Problem 2 — Data Analysis & Modeling

Final report:

`reports/AOROA_Labs_Problem_2_Final_Report.md`

### EDA and Data Understanding

The analysis covers data quality, missing values, duplicates, transaction distributions, discount behavior, app usage, daily sales, weekly seasonality, lower-frequency seasonality, assumptions, and forecasting implications.

A strong weekly pattern was identified, with higher average sales on Saturday and Sunday than Monday–Friday.

### Customer Segmentation

Customer-level features:

- `recency_days`
- `frequency`
- `monetary`
- `avg_transaction_value`
- `avg_discount_rate`
- `avg_app_time_min`

Transformations and modeling:

- `log1p` on recency and monetary values
- `StandardScaler`
- K-Means with candidate K = 2–8
- final K = 4
- stability checks across random seeds
- PCA visualization

Verified K=4 metrics:

- Silhouette: **0.3566**
- Calinski-Harabasz: **415.15**
- Davies-Bouldin: **1.0360**
- Customers: 360 / 238 / 94 / 108
- First two PCA components: **72.26%** combined explained variance

| Segment | Customers | Customer Share | Monetary Share | Interpretation |
|---|---:|---:|---:|---|
| Cluster 0 | 360 | 45.00% | 44.68% | Regular / valuable |
| Cluster 1 | 238 | 29.75% | 20.73% | Discount-driven |
| Cluster 2 | 94 | 11.75% | 27.05% | High-value loyal |
| Cluster 3 | 108 | 13.50% | 7.54% | At-risk / dormant |

### Forecasting

The verified workflow compares:

1. **56-day Weekday Baseline**
2. **Trend + Weekday + Fourier**
3. **Lag + Rolling Random Forest**

Validation uses chronological rolling windows and avoids future-target leakage in forecasting features.

Verified rolling-validation results:

| Model | MAE | RMSE | MAPE |
|---|---:|---:|---:|
| 56-day Weekday Baseline | 564,069.73 | 675,100.86 | 8.959% |
| Trend + Weekday + Fourier | **551,328.20** | **654,019.98** | **8.526%** |
| Lag + Rolling | 562,520.32 | 663,496.48 | 8.862% |

The selected forecasting approach is **Trend + Weekday + Fourier (order=3)** based on the verified rolling-validation comparison.

Final forecast horizon:

**2024-06-03 → 2024-06-30 (28 days)**

### Business Action Plan

Primary target: **Cluster 2 — High-Value Loyal**.

- 94 customers
- 11.75% of customers
- 27.05% of monetary value

Proposed intervention: personalized retention/loyalty treatment with randomized control comparison.

Primary metric: revenue per user.

Secondary metrics: repeat-purchase rate, conversion rate, retention rate, incremental revenue.

Guardrails: discount cost, contribution margin, campaign cost, unsubscribe rate.

The recommendation is a testable business hypothesis, not a causal claim.

## Reproducible Pipeline

`main.py` is the one-shot reproducible entry point.

```text
 data/ecommerce_events.csv
            ↓
      Raw data validation
            ↓
       Daily sales
            ↓
    Customer features
            ↓
       K=2...8 tests
            ↓
        Final K=4
            ↓
  Baseline + validation
            ↓
 Trend + Weekday + Fourier
            ↓
       28-day forecast
            ↓
 Generated outputs + QA
            ↓
 Exact comparison with locked submissions
```

The latest clean raw-data run verified:

- 72,573 raw rows
- 703 daily sales records
- 800 customers
- K=4 clustering: Silhouette 0.3566, CH 415.15, DB 1.0360
- 7 complete rolling validation windows
- 28-day forecast
- **Segments exact value match: True**
- **Forecast dates match: True**
- **Forecast values match: True**
- **Forecast equivalent: True**
- **Return code: 0**

`main.py` does not overwrite the locked official submission files during verification.

## Submission Files

Official files:

- `results/submission_segments.csv`
- `results/submission_forecast.csv`

`submission_segments.csv` schema:

- `user_id`
- `segment`

`submission_forecast.csv` schema:

- `date`
- `predicted_sales`

## Repository Structure

```text
AOROA-Labs-AI-ML-Assignment/
├── data/
│   └── ecommerce_events.csv
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
│   ├── AOROA_Labs_Problem_1_AI_ML_Theory_Private_LLM.pdf
│   └── AOROA_Labs_Problem_2_Final_Report.md
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
├── experiment_log.md
├── main.py
└── requirements.txt
```

## Requirements

```bash
pip install -r requirements.txt
```

Current requirements:

- pandas
- numpy
- scikit-learn

## How to Run

From the repository root:

```bash
python main.py
```

The script reads `data/ecommerce_events.csv`, reproduces the verified customer-segmentation and forecasting pipeline, generates support outputs, and compares generated submission outputs against the locked submission files.

## Final Status

- Problem 1 report: **completed**
- Problem 2 report: **completed**
- Raw dataset: **present and verified**
- Source modules: **verified**
- Raw-data reproducibility: **verified**
- Locked submission equivalence: **verified**
- Final submission QA: **passed**
