
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
