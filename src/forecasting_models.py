
import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor


def create_fourier_features(
    dates,
    origin_date,
    period=7,
    order=3
):
    """
    Create Fourier features for weekly seasonality.
    """

    dates = pd.to_datetime(dates)

    t = (
        dates - pd.Timestamp(origin_date)
    ).dt.days.values

    features = {}

    for k in range(1, order + 1):
        features[f"sin_{k}"] = np.sin(
            2 * np.pi * k * t / period
        )

        features[f"cos_{k}"] = np.cos(
            2 * np.pi * k * t / period
        )

    return pd.DataFrame(
        features,
        index=dates.index
    )


def build_trend_weekday_fourier_model(
    train_df,
    forecast_dates,
    fourier_order=3
):
    """
    Train a Trend + Weekday + Fourier model
    and generate forecasts.
    """

    train = train_df.copy()

    train["date"] = pd.to_datetime(train["date"])

    forecast_dates = pd.Series(
        pd.to_datetime(forecast_dates)
    )

    origin = train["date"].min()

    # Time trend
    train["t"] = (
        train["date"] - origin
    ).dt.days

    forecast_t = (
        forecast_dates - origin
    ).dt.days
    # Weekday
    train["weekday"] = (
        train["date"].dt.dayofweek
    )

    forecast_weekday = (
        forecast_dates.dt.dayofweek
    )

    # Fourier features
    train_fourier = create_fourier_features(
        train["date"],
        origin_date=origin,
        period=7,
        order=fourier_order
    )

    forecast_fourier = create_fourier_features(
        forecast_dates,
        origin_date=origin,
        period=7,
        order=fourier_order
    )

    # Training features
    X_train = pd.DataFrame({
        "t": train["t"].values
    })

    weekday_train = pd.get_dummies(
        train["weekday"],
        prefix="weekday"
    )

    weekday_forecast = pd.get_dummies(
        forecast_weekday,
        prefix="weekday"
    )

    weekday_forecast = weekday_forecast.reindex(
        columns=weekday_train.columns,
        fill_value=0
    )

    X_train = pd.concat(
        [
            X_train.reset_index(drop=True),
            weekday_train.reset_index(drop=True),
            train_fourier.reset_index(drop=True)
        ],
        axis=1
    )

    X_forecast = pd.DataFrame({
        "t": forecast_t.values
    })

    X_forecast = pd.concat(
        [
            X_forecast.reset_index(drop=True),
            weekday_forecast.reset_index(drop=True),
            forecast_fourier.reset_index(drop=True)
        ],
        axis=1
    )

    # Linear regression
    model = LinearRegression()

    model.fit(
        X_train,
        train["total_sales"]
    )

    predictions = model.predict(
        X_forecast
    )

    return predictions, model


def create_lag_features(df):
    """
    Create lag and rolling features.

    shift(1) ensures that current-day actual sales
    are never used as forecasting features.
    """

    data = df.copy()

    data["date"] = pd.to_datetime(data["date"])

    data = (
        data
        .sort_values("date")
        .reset_index(drop=True)
    )

    data["weekday"] = (
        data["date"].dt.dayofweek
    )

    data["trend"] = np.arange(len(data))

    # Lag features
    data["lag_1"] = (
        data["total_sales"].shift(1)
    )

    data["lag_7"] = (
        data["total_sales"].shift(7)
    )

    data["lag_14"] = (
        data["total_sales"].shift(14)
    )

    # Rolling features
    data["rolling_mean_7"] = (
        data["total_sales"]
        .shift(1)
        .rolling(7)
        .mean()
    )

    data["rolling_mean_14"] = (
        data["total_sales"]
        .shift(1)
        .rolling(14)
        .mean()
    )

    data["rolling_mean_28"] = (
        data["total_sales"]
        .shift(1)
        .rolling(28)
        .mean()
    )

    return data


def train_lag_rolling_model(
    train_df,
    feature_cols=None
):
    """
    Train Random Forest using lag and rolling features.
    """

    if feature_cols is None:
        feature_cols = [
            "weekday",
            "trend",
            "lag_1",
            "lag_7",
            "lag_14",
            "rolling_mean_7",
            "rolling_mean_14",
            "rolling_mean_28"
        ]

    train = train_df.copy()

    train = train.dropna(
        subset=feature_cols + ["total_sales"]
    )

    X_train = train[feature_cols]
    y_train = train["total_sales"]

    model = RandomForestRegressor(
        n_estimators=300,
        max_depth=8,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1
    )

    model.fit(
        X_train,
        y_train
    )

    return model
