"""
AOROA Labs AI/ML Assignment
Problem 2 — One-shot reproducible pipeline.

Pipeline:
raw data -> daily sales + customer features -> K selection/stability
-> final K=4 segmentation -> baseline/validation -> final 28-day forecast
-> locked submission QA.

The implementation follows the verified Day 1, Day 2 and Day 3 notebook logic.
"""

import os
import sys
import numpy as np
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.metrics import (
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score,
)
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(PROJECT_DIR, "data", "ecommerce_events.csv")
RESULTS_DIR = os.path.join(PROJECT_DIR, "results")

if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

from src.baseline import calculate_weekday_baseline, create_baseline_forecast
from src.forecasting_models import build_trend_weekday_fourier_model
from src.validation import (
    create_validation_windows,
    check_validation_leakage,
    get_complete_windows,
    calculate_metrics,
)


def load_raw_data():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Raw dataset not found: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)

    expected = [
        "date",
        "user_id",
        "payment_method",
        "discount_rate",
        "app_time_min",
        "paid_amount",
    ]

    if list(df.columns) != expected:
        raise ValueError(
            f"Unexpected schema.\nExpected: {expected}\nFound: {list(df.columns)}"
        )

    df["date"] = pd.to_datetime(df["date"])

    if df["date"].isna().any():
        raise ValueError("Invalid/missing dates found.")

    if df["user_id"].isna().any():
        raise ValueError("Missing user_id values found.")

    if df.duplicated().any():
        raise ValueError("Duplicate transaction rows found.")

    return df


def create_daily_sales(df):
    daily_sales = (
        df.groupby("date")["paid_amount"]
        .sum()
        .reset_index(name="total_sales")
        .sort_values("date")
        .reset_index(drop=True)
    )

    if daily_sales["date"].duplicated().any():
        raise ValueError("Duplicate daily dates found.")

    return daily_sales


def create_customer_features(df):
    reference_date = df["date"].max()

    customer_features = (
        df.groupby("user_id")
        .agg(
            last_purchase_date=("date", "max"),
            frequency=("user_id", "size"),
            monetary=("paid_amount", "sum"),
            avg_transaction_value=("paid_amount", "mean"),
            avg_discount_rate=("discount_rate", "mean"),
            avg_app_time_min=("app_time_min", "mean"),
        )
        .reset_index()
    )

    customer_features["recency_days"] = (
        reference_date - customer_features["last_purchase_date"]
    ).dt.days

    # Exact Day 2 treatment for missing payment method.
    df2 = df.copy()
    df2["payment_method_clean"] = df2["payment_method"].fillna("Unknown")

    payment_counts = (
        df2.groupby("user_id")["payment_method_clean"]
        .nunique()
        .reset_index(name="payment_method_count")
    )

    customer_features = customer_features.merge(
        payment_counts,
        on="user_id",
        how="left",
    )

    return customer_features


def run_segmentation(customer_features):
    customer_model = customer_features.copy()

    # Exact verified Day 2 transformation.
    customer_model["recency_log"] = np.log1p(
        customer_model["recency_days"]
    )
    customer_model["monetary_log"] = np.log1p(
        customer_model["monetary"]
    )

    model_features = [
        "recency_log",
        "frequency",
        "monetary_log",
        "avg_transaction_value",
        "avg_discount_rate",
        "avg_app_time_min",
    ]

    scaler = StandardScaler()
    X = customer_model[model_features].copy()
    X_scaled = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(
        X_scaled,
        columns=model_features,
        index=customer_model.index,
    )

    # Candidate K=2...8 — exact Day 2 experiment.
    clustering_results = []
    for k in range(2, 9):
        kmeans = KMeans(
            n_clusters=k,
            random_state=42,
            n_init=20,
        )
        labels = kmeans.fit_predict(X_scaled)

        clustering_results.append(
            {
                "k": k,
                "silhouette": silhouette_score(X_scaled, labels),
                "calinski_harabasz": calinski_harabasz_score(
                    X_scaled, labels
                ),
                "davies_bouldin": davies_bouldin_score(
                    X_scaled, labels
                ),
                "inertia": kmeans.inertia_,
            }
        )

    clustering_results = pd.DataFrame(clustering_results)

    # Exact verified final K=4 model.
    final_kmeans = KMeans(
        n_clusters=4,
        random_state=42,
        n_init=20,
    )
    customer_model["cluster"] = final_kmeans.fit_predict(X_scaled)

    # Exact stability experiment.
    stability_results = []
    for seed in [0, 7, 21, 42, 100, 2026]:
        kmeans = KMeans(
            n_clusters=4,
            random_state=seed,
            n_init=20,
        )
        labels = kmeans.fit_predict(X_scaled)

        stability_results.append(
            {
                "random_state": seed,
                "silhouette": silhouette_score(X_scaled, labels),
                "inertia": kmeans.inertia_,
            }
        )

    stability_results = pd.DataFrame(stability_results)

    # Exact PCA step.
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_scaled)

    customer_model["PCA1"] = X_pca[:, 0]
    customer_model["PCA2"] = X_pca[:, 1]

    # Supporting outputs.
    profile_cols = [
        "recency_days",
        "frequency",
        "monetary",
        "avg_transaction_value",
        "avg_discount_rate",
        "avg_app_time_min",
    ]

    cluster_profile = (
        customer_model.groupby("cluster")[profile_cols]
        .agg(["mean", "median"])
    )

    cluster_size = (
        customer_model["cluster"]
        .value_counts()
        .sort_index()
        .to_frame("customer_count")
    )
    cluster_size["percentage"] = (
        cluster_size["customer_count"]
        / len(customer_model)
        * 100
    )

    revenue_contribution = (
        customer_model.groupby("cluster")
        .agg(
            customers=("user_id", "count"),
            total_monetary=("monetary", "sum"),
            avg_monetary=("monetary", "mean"),
            median_monetary=("monetary", "median"),
        )
    )

    revenue_contribution["customer_share_pct"] = (
        revenue_contribution["customers"]
        / len(customer_model)
        * 100
    )

    total_revenue = revenue_contribution["total_monetary"].sum()
    revenue_contribution["revenue_share_pct"] = (
        revenue_contribution["total_monetary"]
        / total_revenue
        * 100
    )

    corr = customer_features[
        [
            "recency_days",
            "frequency",
            "monetary",
            "avg_transaction_value",
            "avg_discount_rate",
            "avg_app_time_min",
            "payment_method_count",
        ]
    ].corr()

    return (
        customer_model,
        X_scaled,
        clustering_results,
        stability_results,
        cluster_profile,
        cluster_size,
        revenue_contribution,
        corr,
        pca.explained_variance_ratio_,
    )


def run_forecasting(daily_sales):
    # Official 56-day weekday baseline.
    weekday_baseline = calculate_weekday_baseline(
        daily_sales,
        window_days=56,
    )

    baseline_forecast = create_baseline_forecast(
        daily_sales,
        weekday_baseline,
        horizon=7,
    )

    # Rolling validation leakage check.
    validation_windows = create_validation_windows(
        daily_sales,
        horizon=7,
        n_windows=8,
    )

    if not check_validation_leakage(validation_windows):
        raise AssertionError("Validation leakage check failed.")

    complete_windows = get_complete_windows(
        validation_windows,
        daily_sales,
    )

    # Assignment's final 28-day fixed-origin forecast.
    forecast_dates = pd.date_range(
        start=daily_sales["date"].max() + pd.Timedelta(days=1),
        periods=28,
        freq="D",
    )

    predictions, model = build_trend_weekday_fourier_model(
        daily_sales,
        forecast_dates,
        fourier_order=3,
    )

    forecast = pd.DataFrame(
        {
            "date": forecast_dates,
            "predicted_sales": predictions,
        }
    )

    return (
        weekday_baseline,
        baseline_forecast,
        validation_windows,
        complete_windows,
        forecast,
    )


def run_official_baseline_validation(daily_sales):
    """
    Reproduce the Day 1 fixed 28-day baseline evaluation:
    training ends 2024-05-05 and validation is 2024-05-06..2024-06-02.
    """
    validation_start = pd.Timestamp("2024-05-06")
    validation_end = pd.Timestamp("2024-06-02")

    train = daily_sales[daily_sales["date"] < validation_start].copy()
    validation = daily_sales[
        (daily_sales["date"] >= validation_start)
        & (daily_sales["date"] <= validation_end)
    ].copy()

    baseline_window = train.tail(56).copy()
    baseline_window["weekday"] = baseline_window["date"].dt.day_name()

    weekday_means = baseline_window.groupby("weekday")["total_sales"].mean()

    validation["weekday"] = validation["date"].dt.day_name()
    validation["forecast"] = validation["weekday"].map(weekday_means)

    metrics = calculate_metrics(
        validation["total_sales"].values,
        validation["forecast"].values,
    )

    return validation, metrics


def save_outputs(
    df,
    daily_sales,
    customer_features,
    customer_model,
    clustering_results,
    stability_results,
    cluster_profile,
    cluster_size,
    revenue_contribution,
    corr,
    weekday_baseline,
    forecast,
):
    os.makedirs(RESULTS_DIR, exist_ok=True)

    daily_sales.to_csv(
        os.path.join(RESULTS_DIR, "daily_sales.csv"),
        index=False,
    )

    customer_features.to_csv(
        os.path.join(RESULTS_DIR, "customer_features_raw.csv"),
        index=False,
    )

    customer_model.to_csv(
        os.path.join(RESULTS_DIR, "customer_segments.csv"),
        index=False,
    )

    clustering_results.to_csv(
        os.path.join(RESULTS_DIR, "clustering_results_k2_k8.csv"),
        index=False,
    )

    stability_results.to_csv(
        os.path.join(RESULTS_DIR, "cluster_stability.csv"),
        index=False,
    )

    cluster_profile.to_csv(
        os.path.join(RESULTS_DIR, "cluster_profile.csv"),
    )

    cluster_size.to_csv(
        os.path.join(RESULTS_DIR, "cluster_size.csv"),
    )

    revenue_contribution.to_csv(
        os.path.join(RESULTS_DIR, "segment_revenue_contribution.csv"),
    )

    corr.to_csv(
        os.path.join(RESULTS_DIR, "customer_feature_correlation.csv"),
    )

    pd.DataFrame(
        {
            "weekday": weekday_baseline.index,
            "mean_total_sales": weekday_baseline.values,
        }
    ).to_csv(
        os.path.join(RESULTS_DIR, "baseline_weekday_means.csv"),
        index=False,
    )

    forecast.to_csv(
        os.path.join(RESULTS_DIR, "generated_forecast.csv"),
        index=False,
    )

    # Required submission schema.
    submission_segments = customer_model[
        ["user_id", "cluster"]
    ].rename(
        columns={"cluster": "segment"}
    )

    submission_segments.to_csv(
        os.path.join(RESULTS_DIR, "submission_segments_generated.csv"),
        index=False,
    )

    submission_forecast = forecast[
        ["date", "predicted_sales"]
    ].copy()

    submission_forecast.to_csv(
        os.path.join(
            RESULTS_DIR,
            "submission_forecast_generated.csv",
        ),
        index=False,
    )

    return submission_segments, submission_forecast


def verify_against_locked_submissions(
    generated_segments,
    generated_forecast,
):
    """
    Compare newly generated outputs with the verified locked files.

    CSV round-tripping changes dtypes (especially dates), so comparison is
    normalized by key and value before judging equivalence. Existing locked
    submission files are never overwritten.
    """
    locked_segments_path = os.path.join(
        RESULTS_DIR,
        "submission_segments.csv",
    )
    locked_forecast_path = os.path.join(
        RESULTS_DIR,
        "submission_forecast.csv",
    )

    if not (
        os.path.exists(locked_segments_path)
        and os.path.exists(locked_forecast_path)
    ):
        print("⚠️ Locked submission files not found; comparison skipped.")
        return False

    locked_segments = pd.read_csv(locked_segments_path)
    locked_forecast = pd.read_csv(locked_forecast_path)

    # ----------------------------
    # Segmentation normalization
    # ----------------------------
    generated_segments = generated_segments[
        ["user_id", "segment"]
    ].copy()
    locked_segments = locked_segments[
        ["user_id", "segment"]
    ].copy()

    generated_segments["user_id"] = (
        generated_segments["user_id"].astype(str)
    )
    locked_segments["user_id"] = (
        locked_segments["user_id"].astype(str)
    )

    generated_segments["segment"] = pd.to_numeric(
        generated_segments["segment"],
        errors="raise",
    ).astype(int)
    locked_segments["segment"] = pd.to_numeric(
        locked_segments["segment"],
        errors="raise",
    ).astype(int)

    generated_segments = generated_segments.sort_values(
        "user_id"
    ).reset_index(drop=True)
    locked_segments = locked_segments.sort_values(
        "user_id"
    ).reset_index(drop=True)

    segment_exact = generated_segments.equals(locked_segments)

    # ----------------------------
    # Forecast normalization
    # ----------------------------
    generated_forecast = generated_forecast[
        ["date", "predicted_sales"]
    ].copy()
    locked_forecast = locked_forecast.copy()

    # Support the historical locked schema if it is forecast rather than
    # predicted_sales; the submission schema remains predicted_sales.
    if "predicted_sales" not in locked_forecast.columns:
        if "forecast" in locked_forecast.columns:
            locked_forecast = locked_forecast.rename(
                columns={"forecast": "predicted_sales"}
            )
        else:
            raise ValueError(
                "Locked forecast has neither 'predicted_sales' nor 'forecast'."
            )

    generated_forecast["date"] = pd.to_datetime(
        generated_forecast["date"]
    ).dt.normalize()
    locked_forecast["date"] = pd.to_datetime(
        locked_forecast["date"]
    ).dt.normalize()

    generated_forecast["predicted_sales"] = pd.to_numeric(
        generated_forecast["predicted_sales"],
        errors="raise",
    )
    locked_forecast["predicted_sales"] = pd.to_numeric(
        locked_forecast["predicted_sales"],
        errors="raise",
    )

    generated_forecast = generated_forecast.sort_values(
        "date"
    ).reset_index(drop=True)
    locked_forecast = locked_forecast.sort_values(
        "date"
    ).reset_index(drop=True)

    forecast_dates_equal = generated_forecast["date"].equals(
        locked_forecast["date"]
    )

    forecast_values_equal = (
        len(generated_forecast) == len(locked_forecast)
        and np.allclose(
            generated_forecast["predicted_sales"].to_numpy(),
            locked_forecast["predicted_sales"].to_numpy(),
            rtol=1e-10,
            atol=1e-6,
        )
    )

    forecast_match = forecast_dates_equal and forecast_values_equal

    print("\n[LOCKED OUTPUT COMPARISON]")
    print("Segments exact value match:", segment_exact)
    print("Forecast dates match:", forecast_dates_equal)
    print("Forecast values match:", forecast_values_equal)
    print("Forecast equivalent:", forecast_match)

    # If segmentation differs, print the first differences so the cause is
    # visible rather than silently failing.
    if not segment_exact:
        diff = generated_segments.merge(
            locked_segments,
            on="user_id",
            how="outer",
            suffixes=("_generated", "_locked"),
            indicator=True,
        )
        diff = diff[
            (diff["_merge"] != "both")
            | (
                diff["segment_generated"]
                != diff["segment_locked"]
            )
        ]

        print("Segment differences:", len(diff))
        if len(diff):
            print(diff.head(10).to_string(index=False))

    if not forecast_match:
        print(
            "\nForecast comparison summary:"
        )
        print(
            "Generated rows:",
            len(generated_forecast),
            "Locked rows:",
            len(locked_forecast),
        )

        if len(generated_forecast) == len(locked_forecast):
            value_diff = (
                generated_forecast["predicted_sales"].to_numpy()
                - locked_forecast["predicted_sales"].to_numpy()
            )
            print(
                "Max absolute forecast difference:",
                float(np.max(np.abs(value_diff))),
            )

    return segment_exact and forecast_match



def final_qa(submission_segments, submission_forecast):
    assert list(submission_segments.columns) == ["user_id", "segment"]
    assert len(submission_segments) == 800
    assert submission_segments["user_id"].notna().all()
    assert submission_segments["segment"].notna().all()
    assert submission_segments["user_id"].duplicated().sum() == 0
    assert set(submission_segments["segment"].unique()) == {0, 1, 2, 3}

    assert list(submission_forecast.columns) == [
        "date",
        "predicted_sales",
    ]
    assert len(submission_forecast) == 28
    assert submission_forecast["date"].notna().all()
    assert submission_forecast["predicted_sales"].notna().all()
    assert submission_forecast["date"].duplicated().sum() == 0
    assert pd.to_numeric(
        submission_forecast["predicted_sales"],
        errors="coerce",
    ).notna().all()
    assert (submission_forecast["predicted_sales"] >= 0).all()

    dates = pd.DatetimeIndex(
        pd.to_datetime(submission_forecast["date"])
    ).normalize()

    expected_dates = pd.date_range(
        "2024-06-03",
        "2024-06-30",
        freq="D",
    )

    assert dates.equals(expected_dates)

    return True


def main():
    print("=" * 70)
    print("AOROA LABS — RAW-DATA REPRODUCIBLE PIPELINE")
    print("=" * 70)

    print("\n[1] Loading raw dataset")
    df = load_raw_data()
    print("Rows:", len(df))
    print("Columns:", list(df.columns))
    print(
        "Date range:",
        df["date"].min().date(),
        "→",
        df["date"].max().date(),
    )

    print("\n[2] Creating daily sales")
    daily_sales = create_daily_sales(df)
    print("Rows:", len(daily_sales))
    print(
        "Date range:",
        daily_sales["date"].min().date(),
        "→",
        daily_sales["date"].max().date(),
    )

    print("\n[3] Creating customer features")
    customer_features = create_customer_features(df)
    print("Customers:", len(customer_features))

    print("\n[4] Running K=2...8 clustering experiments")
    (
        customer_model,
        X_scaled,
        clustering_results,
        stability_results,
        cluster_profile,
        cluster_size,
        revenue_contribution,
        corr,
        explained_variance,
    ) = run_segmentation(customer_features)

    selected = clustering_results[
        clustering_results["k"] == 4
    ].iloc[0]

    print(
        "K=4:",
        f"Silhouette={selected['silhouette']:.4f},",
        f"CH={selected['calinski_harabasz']:.2f},",
        f"DB={selected['davies_bouldin']:.4f}",
    )
    print(
        "Cluster counts:",
        customer_model["cluster"].value_counts().sort_index().to_dict(),
    )
    print(
        "PCA explained variance:",
        round(float(explained_variance.sum()), 4),
    )

    print("\n[5] Forecasting + validation")
    (
        weekday_baseline,
        baseline_forecast,
        validation_windows,
        complete_windows,
        forecast,
    ) = run_forecasting(daily_sales)

    print("Complete rolling windows:", len(complete_windows))
    print("Forecast rows:", len(forecast))

    print("\n[6] Reproducing official baseline validation")
    _, baseline_metrics = run_official_baseline_validation(
        daily_sales
    )
    print(
        "Official baseline:",
        {
            k: round(float(v), 4)
            for k, v in baseline_metrics.items()
        },
    )

    print("\n[7] Saving generated outputs")
    (
        generated_segments,
        generated_forecast,
    ) = save_outputs(
        df,
        daily_sales,
        customer_features,
        customer_model,
        clustering_results,
        stability_results,
        cluster_profile,
        cluster_size,
        revenue_contribution,
        corr,
        weekday_baseline,
        forecast,
    )

    print(
        "Generated segmentation:",
        len(generated_segments),
    )
    print(
        "Generated forecast:",
        len(generated_forecast),
    )

    print("\n[8] Locked submission comparison")
    exact_match = verify_against_locked_submissions(
        generated_segments,
        generated_forecast,
    )

    print("\n[9] Final QA")
    final_qa(
        generated_segments,
        generated_forecast,
    )

    print("\n" + "=" * 70)

    if exact_match:
        print(
            "🎉 RAW-DATA REPRODUCIBILITY + LOCKED OUTPUT MATCH PASSED"
        )
    else:
        print(
            "⚠️ RAW-DATA PIPELINE + QA PASSED, "
            "but locked-output exact match was not confirmed."
        )

    print("=" * 70)

    # Do not overwrite locked submission files automatically.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
