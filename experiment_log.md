
# AOROA Labs — Experiment Log

## Project
AI/ML Researcher Pre-Assignment — Problem 2

---

# Day 1 — Data Understanding & Baseline

## Experiment 001 — Data Audit

Dataset:
- File: ecommerce_events.csv
- Rows: 72,573
- Columns: 6
- Unique users: 800
- Date range: 2022-07-01 → 2024-06-02
- Unique dates: 703
- Duplicate rows: 0
- Missing calendar days: 0

Missing values:
- payment_method: 2,314
- Other columns: 0

Decision:
- Missing payment_method values are retained.
- They will be treated as an explicit unknown category during segmentation.
- No arbitrary imputation is applied.

Status: COMPLETED

---

## Experiment 002 — Transaction-Level EDA

### paid_amount

- Min: 2,000
- Max: 1,619,435
- Mean: 54,709.84
- Median: 50,130
- Q1: 33,007
- Q3: 71,457
- Std: 31,138.02
- IQR upper fence: 129,132
- Potential high outliers: 1,530

Decision:
Potential high-value transactions are not automatically removed because
they may represent legitimate business transactions.

Status: COMPLETED

---

### discount_rate

- Min: 0
- Max: 0.30
- Mean: approximately 0.0565
- Median: 0
- Approximately 66.5% of transactions have zero discount.

Decision:
The feature is retained for segmentation analysis.

No causal interpretation is made.

Status: COMPLETED

---

### app_time_min

- Min: 1
- Max: 65
- Mean: 16.75
- Median: 16
- Q1: 10
- Q3: 23
- Std: 9.69
- IQR upper fence: 42.5
- Potential high-duration observations: 699

Decision:
Potential outliers are retained unless later analysis indicates invalid data.

Status: COMPLETED

---

## Experiment 003 — Daily Sales EDA

Daily sales:

- Minimum: 2,656,300
- Maximum: 10,950,187
- Mean: 5,647,876.38
- Median: 5,469,311
- Std: 1,258,926.86

Findings:
- High daily volatility
- Significant sales spikes
- Non-monotonic medium-term trend
- Strong weekly seasonality

Status: COMPLETED

---

## Experiment 004 — Weekly Seasonality

Average daily sales:

- Monday: 4,928,258
- Tuesday: 4,953,262
- Wednesday: 5,314,841
- Thursday: 5,295,031
- Friday: 5,347,059
- Saturday: 6,828,375
- Sunday: 6,847,515

Weekend average: approximately 6.84M
Weekday average: approximately 5.17M

Weekend uplift: approximately 32%

Decision:
Weekday seasonality is considered a strong predictive signal.

Status: COMPLETED

---

## Experiment 005 — Official 56-Day Weekday Baseline

Validation design:

- Training: 2022-07-01 → 2024-05-05
- Validation: 2024-05-06 → 2024-06-02
- Validation horizon: 28 days
- Baseline history: final 56 training days

Metrics:

- MAPE: 10.7282%
- MAE: 628,202.54
- RMSE: 734,424.97

Decision:
The official 56-day weekday-average model is frozen as the primary
forecasting benchmark.

Status: FROZEN BASELINE

---

# Key Research Hypothesis

The official baseline captures strong weekday seasonality but does not
fully capture recent sales dynamics and local level changes.

Therefore:

Adding recent historical information while preserving weekly seasonality
should improve forecasting performance relative to the 10.7282% MAPE
baseline.

---

# Data Leakage Rules

1. No future actual sales values may be used as forecasting features.
2. Random train/test splitting is not appropriate.
3. Validation must preserve chronological order.
4. Future observations must not influence historical preprocessing.
5. No causal claims will be made from observational relationships.

---

# Day 1 Status

Data Audit: COMPLETED
Transaction EDA: COMPLETED
Daily Sales Analysis: COMPLETED
Trend Analysis: COMPLETED
Weekly Seasonality: COMPLETED
Official Baseline: COMPLETED
EDA Findings & Assumptions: COMPLETED

Next:
Customer Segmentation
Time-Series Forecasting Experiments
Business Action Plan
