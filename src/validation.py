"""Chronological validation utilities for sales forecasting."""

import numpy as np
import pandas as pd

from src.baseline import calculate_weekday_baseline
from src.forecasting_models import build_trend_weekday_fourier_model


def _prepare_daily_sales(daily_sales):
    required = {"date", "total_sales"}
    missing = required.difference(daily_sales.columns)
    if missing:
        raise ValueError(f"daily_sales is missing required columns: {sorted(missing)}")

    data = daily_sales[["date", "total_sales"]].copy()
    data["date"] = pd.to_datetime(data["date"]).dt.normalize()
    data["total_sales"] = pd.to_numeric(data["total_sales"], errors="raise")

    if data[["date", "total_sales"]].isna().any().any():
        raise ValueError("Daily sales contains missing dates or sales values.")
    if not np.isfinite(data["total_sales"].to_numpy(dtype=float)).all():
        raise ValueError("Daily sales contains non-finite sales values.")
    if data["date"].duplicated().any():
        raise ValueError("Daily sales must contain unique dates.")

    data = data.sort_values("date").reset_index(drop=True)
    expected = pd.date_range(data["date"].min(), data["date"].max(), freq="D")
    if not pd.DatetimeIndex(data["date"]).equals(expected):
        raise ValueError("Daily sales dates must be continuous with no missing calendar days.")
    return data


def create_validation_windows(daily_sales, horizon=7, n_windows=8):
    """Create expanding-window, chronological validation folds."""
    data = _prepare_daily_sales(daily_sales)
    if horizon < 1 or n_windows < 1:
        raise ValueError("horizon and n_windows must be positive integers.")
    min_required = 56 + horizon * n_windows
    if len(data) < min_required:
        raise ValueError(
            f"Need at least {min_required} daily rows for {n_windows} windows "
            f"with a 56-day baseline; received {len(data)}."
        )

    rows = []
    for i in range(n_windows, 0, -1):
        cutoff_idx = len(data) - horizon * i
        cutoff = data.loc[cutoff_idx, "date"]
        start = cutoff + pd.Timedelta(days=1)
        end = start + pd.Timedelta(days=horizon - 1)
        rows.append({
            "cutoff_date": cutoff,
            "forecast_start": start,
            "forecast_end": end,
            "horizon": horizon,
        })
    return pd.DataFrame(rows)


def check_validation_leakage(validation_windows):
    """Assert each forecast window starts strictly after its training cutoff."""
    required = {"cutoff_date", "forecast_start", "forecast_end"}
    if not required.issubset(validation_windows.columns):
        raise ValueError(f"validation_windows must contain {sorted(required)}")
    for row in validation_windows.itertuples(index=False):
        if not (row.forecast_start > row.cutoff_date):
            raise AssertionError("Forecast starts on or before the training cutoff.")
        if not (row.forecast_end >= row.forecast_start):
            raise AssertionError("Forecast end precedes forecast start.")
    return True


def get_complete_windows(validation_windows, daily_sales):
    """Keep only windows whose every target date exists in the observed series."""
    data = _prepare_daily_sales(daily_sales)
    available = set(data["date"])
    complete = []
    for row in validation_windows.itertuples(index=False):
        expected = pd.date_range(row.forecast_start, row.forecast_end, freq="D")
        if all(date in available for date in expected):
            complete.append(row.Index if hasattr(row, "Index") else None)
    if len(complete) == len(validation_windows):
        return validation_windows.copy().reset_index(drop=True)

    mask = []
    for row in validation_windows.itertuples(index=False):
        expected = pd.date_range(row.forecast_start, row.forecast_end, freq="D")
        mask.append(all(date in available for date in expected))
    return validation_windows.loc[mask].copy().reset_index(drop=True)


def calculate_metrics(actual, forecast):
    """Calculate MAE, RMSE and MAPE with explicit zero-denominator handling."""
    actual = np.asarray(actual, dtype=float)
    forecast = np.asarray(forecast, dtype=float)
    if actual.shape != forecast.shape or actual.size == 0:
        raise ValueError("actual and forecast must have the same non-empty shape.")
    if not (np.isfinite(actual).all() and np.isfinite(forecast).all()):
        raise ValueError("actual and forecast must contain only finite values.")
    if np.any(actual == 0):
        raise ValueError("MAPE is undefined when actual sales contain zero values.")

    error = actual - forecast
    return {
        "MAE": float(np.mean(np.abs(error))),
        "RMSE": float(np.sqrt(np.mean(error ** 2))),
        "MAPE": float(np.mean(np.abs(error / actual)) * 100),
    }


def calculate_window_metrics(results_df, model_name):
    """Calculate metrics independently for each cutoff."""
    rows = []
    for cutoff, group in results_df.groupby("cutoff_date", sort=True):
        metrics = calculate_metrics(group["actual"], group["forecast"])
        rows.append({"model": model_name, "cutoff_date": cutoff, **metrics})
    return pd.DataFrame(rows)


def create_stability_summary(window_metrics):
    """Summarize mean and population standard deviation across folds."""
    if window_metrics.empty:
        raise ValueError("window_metrics cannot be empty.")
    return (
        window_metrics.groupby("model")
        .agg(
            mean_mae=("MAE", "mean"),
            std_mae=("MAE", lambda x: x.std(ddof=0)),
            mean_rmse=("RMSE", "mean"),
            std_rmse=("RMSE", lambda x: x.std(ddof=0)),
            mean_mape=("MAPE", "mean"),
            std_mape=("MAPE", lambda x: x.std(ddof=0)),
            windows=("cutoff_date", "nunique"),
        )
        .reset_index()
    )


def run_rolling_backtest(daily_sales, validation_windows, fourier_order=3, baseline_days=56):
    """
    Refit baseline and Fourier models independently at every cutoff.

    Only rows dated on or before a fold's cutoff are used for fitting.
    Actual values are joined only after predictions have been generated.
    """
    data = _prepare_daily_sales(daily_sales)
    if not check_validation_leakage(validation_windows):
        raise AssertionError("Validation leakage check failed.")
    if validation_windows.empty:
        raise ValueError("No complete validation windows are available.")

    actual_by_date = data.set_index("date")["total_sales"]
    result_rows = []

    for fold in validation_windows.itertuples(index=False):
        train = data.loc[data["date"] <= fold.cutoff_date].copy()
        forecast_dates = pd.date_range(fold.forecast_start, fold.forecast_end, freq="D")
        if len(train) < baseline_days:
            raise ValueError(f"Insufficient training history at cutoff {fold.cutoff_date}.")
        if not all(date in actual_by_date.index for date in forecast_dates):
            raise ValueError(f"Incomplete actual horizon for cutoff {fold.cutoff_date}.")

        # 56-day weekday mean baseline, fit using training data only.
        weekday_means = calculate_weekday_baseline(train, window_days=baseline_days)
        baseline_predictions = pd.Series(forecast_dates).map(
            lambda date: weekday_means.get(date.day_name(), np.nan)
        ).to_numpy(dtype=float)

        # Refit the trend + weekday + Fourier regression at this fold's cutoff.
        fourier_predictions, _ = build_trend_weekday_fourier_model(
            train, forecast_dates, fourier_order=fourier_order
        )

        actual_values = actual_by_date.loc[forecast_dates].to_numpy(dtype=float)
        for model_name, predictions in (
            ("56-day Weekday Baseline", baseline_predictions),
            ("Trend + Weekday + Fourier", np.asarray(fourier_predictions, dtype=float)),
        ):
            if not np.isfinite(predictions).all():
                raise ValueError(f"{model_name} generated non-finite predictions.")
            for date, actual, prediction in zip(forecast_dates, actual_values, predictions):
                result_rows.append({
                    "model": model_name,
                    "cutoff_date": fold.cutoff_date,
                    "date": date,
                    "actual": actual,
                    "forecast": float(prediction),
                })

    results = pd.DataFrame(result_rows)
    per_window = pd.concat(
        [
            calculate_window_metrics(
                results.loc[results["model"] == model_name],
                model_name,
            )
            for model_name in results["model"].unique()
        ],
        ignore_index=True,
    )
    summary = create_stability_summary(per_window)
    return results, per_window, summary
