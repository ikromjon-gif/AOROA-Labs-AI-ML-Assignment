
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
