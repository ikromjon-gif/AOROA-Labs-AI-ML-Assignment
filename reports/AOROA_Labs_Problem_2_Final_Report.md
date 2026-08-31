# AOROA Labs — Problem 2 Final Report

This document consolidates the supplied EDA/assumptions and customer-segmentation reports, then adds the verified forecasting and required business-action-plan sections.

---


# Problem 2 — EDA Findings & Assumptions

## 1. Dataset Overview

The dataset contains transaction-level e-commerce records covering the period
from 2022-07-01 to 2024-06-02.

- Transactions: 72,573
- Unique users: 800
- Historical dates: 703
- Date range: 2022-07-01 to 2024-06-02
- Calendar gaps: 0
- Duplicate rows: 0

The dataset contains the following variables:

- `date`
- `user_id`
- `payment_method`
- `discount_rate`
- `app_time_min`
- `paid_amount`

---

## 2. Data Quality Findings

No duplicate transaction rows were identified.

There are 2,314 missing values in `payment_method`, while the remaining
variables contain no missing values.

The missing `payment_method` values are retained during the initial EDA.
For customer segmentation, they will be treated as an explicit unknown
category rather than being assigned an arbitrary payment method.

The date series contains no missing calendar days.

---

## 3. Transaction-Level EDA

### 3.1 Paid Amount

The transaction amount ranges from 2,000 to 1,619,435.

- Mean: 54,709.84
- Median: 50,130
- Q1: 33,007
- Q3: 71,457
- Standard deviation: 31,138.02

The mean is higher than the median and the distribution contains a long
right tail. Using the IQR rule, the upper fence is approximately 129,132,
with 1,530 observations above this threshold.

These observations are considered potential high-value transactions.
They are not automatically removed because they may represent legitimate
customer purchases.

---

### 3.2 Discount Rate

The discount rate ranges from 0 to 0.30.

The median discount rate is 0, indicating that a large proportion of
transactions have no discount.

Observed discount values are:

0, 0.05, 0.10, 0.15, 0.20, 0.25, and 0.30.

Approximately 66.5% of transactions have a zero discount rate.

The variable is therefore potentially useful for customer segmentation.

However, observational relationships between discounts and sales should
not be interpreted as causal effects.

---

### 3.3 App Time

`app_time_min` ranges from 1 to 65 minutes.

- Mean: 16.75 minutes
- Median: 16 minutes
- Q1: 10 minutes
- Q3: 23 minutes
- Standard deviation: 9.69 minutes

The distribution is relatively concentrated around the central range but
contains some high-duration observations.

Using the IQR rule, the upper fence is approximately 42.5 minutes, with
699 observations above this threshold.

These observations are retained unless later analysis provides evidence
that they are invalid.

---

## 4. Daily Sales EDA

Transaction-level records were aggregated by date to create a daily sales
time series.

Daily sales statistics:

- Minimum: 2,656,300
- Maximum: 10,950,187
- Mean: 5,647,876.38
- Median: 5,469,311
- Standard deviation: 1,258,926.86

The daily series shows substantial volatility and several high-sales spikes.

The historical series does not exhibit a simple monotonic linear trend.
Instead, it contains medium-term increases and decreases, indicating
changes in the local sales level over time.

---

## 5. Weekly Seasonality

A strong day-of-week seasonal pattern was identified.

Average daily sales:

| Weekday | Average Sales |
|---|---:|
| Monday | 4,928,258 |
| Tuesday | 4,953,262 |
| Wednesday | 5,314,841 |
| Thursday | 5,295,031 |
| Friday | 5,347,059 |
| Saturday | 6,828,375 |
| Sunday | 6,847,515 |

Average weekend sales are approximately 6.84M, compared with approximately
5.17M for Monday-Friday.

Therefore, weekend sales are approximately 32% higher than weekday sales.

This represents a strong and practically meaningful weekly seasonal signal.

---

## 6. Lower-Frequency Seasonality

Monthly average sales also show noticeable differences across months.

March has one of the highest average daily sales levels, while October is
among the lower months.

However, the available history covers approximately 23 months rather than
multiple complete years. Therefore, the analysis does not claim strong
annual seasonality with high confidence.

Monthly differences are treated as a possible lower-frequency seasonal
component that can be investigated during forecasting experiments.

---

## 7. Forecasting Implications

The EDA suggests that an effective forecasting model should account for:

1. Day-of-week seasonality.
2. Recent sales dynamics.
3. Medium-term changes in the local sales level.
4. Daily volatility.
5. Potential lower-frequency seasonal effects.

The official 56-day weekday-average baseline captures the first component
but does not explicitly model recent dynamics or local level changes.

The official baseline achieved:

- MAPE: 10.7282%
- MAE: 628,202.54
- RMSE: 734,424.97

This baseline is therefore retained as the primary benchmark for subsequent
forecasting experiments.

---

## 8. Assumptions and Decisions

### Missing Values

Missing `payment_method` values are not imputed with an arbitrary payment
method. They will be represented as an unknown category during segmentation
preprocessing.

### Outliers

Potential outliers are not automatically removed. High transaction amounts
and long app sessions may represent legitimate customer behavior.

### Forecast Validation

Random train/test splitting is not appropriate for this time-series task.
Validation must respect chronological order.

### Data Leakage

Future actual sales values must never be used as input features when making
the corresponding future predictions.

### Causal Interpretation

The dataset is observational. Correlations between discounts, app usage,
payment methods, and sales should not be interpreted as causal effects
without an appropriate causal design.

### Baseline

The official 56-day weekday-average method is used as the benchmark and is
not modified.

---

## 9. Key EDA Conclusion

The sales series contains a strong weekly seasonal structure, substantial
daily volatility, and medium-term changes in its local level.

The particularly strong weekend uplift indicates that weekday information
is an important predictive feature. However, the validation results of the
official baseline also show that weekday averages alone cannot fully capture
short-term sales fluctuations.

Therefore, subsequent forecasting experiments will focus on models that
preserve the strong weekly seasonal signal while incorporating recent
historical dynamics without introducing temporal leakage.


---


# AOROA Labs — Day 2 Customer Segmentation

## Dataset

- Customers analyzed: 800
- Clustering method: K-Means
- Candidate K values: 2–8
- Final candidate: K=4
- Scaling: StandardScaler
- Transformation: log1p applied to recency and monetary

## Feature Selection

Primary clustering features:

- recency_days
- frequency
- monetary
- avg_transaction_value
- avg_discount_rate
- avg_app_time_min

payment_method_count was excluded from primary clustering because of very low variance and limited discriminative power.

## Model Selection

K=4 achieved:

- Silhouette Score: 0.3566
- Calinski-Harabasz Score: 415.15
- Davies-Bouldin Score: 1.0360

K=4 was also supported by the elbow analysis.

## Stability

The K=4 solution remained highly stable across multiple random seeds.

Silhouette scores remained approximately 0.3564–0.3566.

## Customer Segments

### Cluster 0 — Regular Valuable
- 360 customers
- 45.00% of customers
- 44.68% of monetary value

Recommended action:
Retention, cross-sell and loyalty engagement.

### Cluster 1 — Discount-Driven
- 238 customers
- 29.75% of customers
- 20.73% of monetary value

Key characteristic:
High discount usage with comparatively lower monetary value.

Recommended action:
Targeted promotions and discount optimization.

### Cluster 2 — High-Value Loyal
- 94 customers
- 11.75% of customers
- 27.05% of monetary value

Key characteristic:
Highest frequency, highest monetary value and highest app engagement.

Recommended action:
VIP retention and premium customer protection.

### Cluster 3 — At-Risk / Dormant
- 108 customers
- 13.50% of customers
- 7.54% of monetary value

Key characteristic:
Very high recency, approximately 135 days since last purchase.

Recommended action:
Win-back and reactivation campaigns.

## PCA Validation

The first two PCA components explain 72.26% of total variance:

- PCA1: 43.14%
- PCA2: 29.12%

The 2D projection shows meaningful separation between the four customer segments.

## Business Conclusion

The segmentation identifies four commercially meaningful customer groups with distinct behavioral and monetary characteristics.

The High-Value Loyal segment represents only 11.75% of customers but contributes approximately 27.05% of total monetary value, indicating strong strategic importance.

The At-Risk / Dormant segment represents 13.50% of customers and approximately 7.54% of monetary value, providing a clear target for reactivation initiatives.


---



# 4. Time-Series Forecasting

## 4.1 Forecasting Objective

The objective is to forecast daily total sales after the historical cutoff of 2024-06-02. The required submission horizon is 28 days, from 2024-06-03 through 2024-06-30.

## 4.2 Official Baseline

The assignment specifies the final 8 weeks (56 days) weekday-average as the official baseline. This baseline captures the strong weekly seasonal pattern identified in EDA but does not explicitly model local trend or smoother periodic structure.

The previously recorded baseline benchmark in the EDA analysis is MAPE 10.7282%, MAE 628,202.54 and RMSE 734,424.97. In the rolling-validation experiment used for model comparison, the verified baseline MAPE was 8.959%. These are retained as separate evaluation contexts rather than being silently combined.

## 4.3 Validation Design

Random train/test splitting is inappropriate for time-series forecasting. Validation therefore respects chronological order and uses rolling windows. The verified pipeline reported seven complete validation windows.

Future actual sales are not used as features for the corresponding forecast, avoiding temporal leakage.

## 4.4 Candidate Models

The verified experiments compared:

1. Official 56-day weekday baseline
2. Trend + Weekday + Fourier
3. Lag + Rolling

## 4.5 Model Comparison

| Model | MAE | RMSE | MAPE |
|---|---:|---:|---:|
| 56-day Weekday Baseline | 564,069.73 | 675,100.86 | 8.959% |
| Trend + Weekday + Fourier | **551,328.20** | **654,019.98** | **8.526%** |
| Lag + Rolling | 562,520.32 | 663,496.48 | 8.862% |

The Trend + Weekday + Fourier model achieved the lowest verified MAE, RMSE and MAPE. Relative to the rolling-validation baseline, MAPE improved by 0.433 percentage points, approximately a 4.84% relative reduction.

Therefore, Trend + Weekday + Fourier is selected as the final forecasting approach.

## 4.6 Final 28-Day Forecast

The final forecast covers 2024-06-03 to 2024-06-30.

| Date | Predicted Sales |
|---|---:|
| 2024-06-03 | 5,602,809 |
| 2024-06-04 | 5,627,814 |
| 2024-06-05 | 5,989,393 |
| 2024-06-06 | 5,969,582 |
| 2024-06-07 | 6,028,289 |
| 2024-06-08 | 7,509,605 |
| 2024-06-09 | 7,528,745 |
| 2024-06-10 | 5,616,167 |
| 2024-06-15 | 7,522,963 |
| 2024-06-20 | 5,996,297 |
| 2024-06-25 | 5,667,886 |
| 2024-06-30 | 7,568,817 |

The complete 28-row forecast is stored in `submission_forecast.csv`.

## 4.7 Forecast QA

The final forecast passed the locked QA:

- 28 rows
- columns: `date`, `predicted_sales`
- no missing values
- no duplicate dates
- numeric predictions
- no negative forecasts
- continuous dates

---

# 5. Business Action Plan

## 5.1 Target Segment

The primary target is **Cluster 2 — High-Value Loyal**:

- 94 customers
- 11.75% of customers
- 27.05% of monetary value
- highest frequency
- highest monetary value
- highest app engagement

This segment is strategically important because its monetary contribution is disproportionately large relative to its customer share.

## 5.2 Treatment

Proposed treatment: a personalized VIP retention and loyalty campaign, potentially including personalized offers, loyalty rewards, premium service and cross-sell recommendations.

The intervention should be precisely defined before testing so that it is reproducible.

## 5.3 Forecast-Based Timing

EDA shows average weekend sales of approximately 6.84M versus approximately 5.17M for Monday-Friday, an uplift of about 32%.

This suggests a testable timing strategy: evaluate whether exposure immediately before or during high-activity periods produces better engagement or revenue outcomes.

Forecasts are used as planning signals, not as causal evidence.

## 5.4 Testable Hypothesis

**H1:** Personalized retention treatment applied to the High-Value Loyal segment increases repeat-purchase activity and revenue per user relative to a randomized control group receiving the standard treatment.

**H0:** The treatment produces no improvement in the primary outcome relative to control.

## 5.5 Experimental Design

High-Value Loyal customers should be randomly split into control and treatment groups.

```text
High-Value Loyal
       |
 Random split
    /       \
Control   Treatment
   |          |
Standard   Personalized
   |          |
   └────┬─────┘
        ↓
 Compare outcomes
```

## 5.6 Success Metrics

Primary metric:

- Revenue per user

Secondary metrics:

- repeat-purchase rate
- conversion rate
- retention rate
- incremental revenue

## 5.7 Risk Guardrails

Monitor:

- discount cost
- gross/contribution margin
- campaign cost
- unsubscribe rate
- excessive incentive use

Revenue growth alone should not define success if profitability deteriorates.

## 5.8 Secondary Opportunity

Cluster 3 — At-Risk/Dormant contains 108 customers (13.50%) and 7.54% of monetary value, with approximately 135 days since last purchase. A secondary win-back experiment is reasonable, but the High-Value Loyal segment should be the first priority because of its higher monetary contribution.

---

# 6. Limitations & Assumptions

- The dataset is observational; associations are not causal effects.
- Potential outliers were retained because they may represent legitimate behavior.
- Approximately 23 months of history are insufficient to claim strong annual seasonality with high confidence.
- Cluster labels are business interpretations, not ground truth.
- Forecasts are estimates rather than guaranteed future sales.
- Business actions are hypotheses that require controlled experimentation.

---

# 7. Final Conclusion

The analysis produces two complementary decision inputs.

Customer segmentation identifies four commercially meaningful groups. High-Value Loyal customers represent 11.75% of customers but approximately 27.05% of monetary value, making them a high-priority retention target.

Forecasting confirms strong weekly seasonality and shows that Trend + Weekday + Fourier outperformed the official weekday baseline in the verified rolling-validation experiment.

The recommended decision process is:

```text
Customer Segmentation
        +
Sales Forecast
        ↓
Target Selection
        ↓
Treatment Timing
        ↓
Controlled Experiment
        ↓
Business Metrics
        ↓
Decision
```

The first proposed experiment is a personalized retention intervention for High-Value Loyal customers, evaluated using revenue per user, repeat purchase, retention and margin guardrails.

**Final modeling decision: K=4 K-Means segmentation + Trend/Weekday/Fourier forecasting + controlled business experiment.**
