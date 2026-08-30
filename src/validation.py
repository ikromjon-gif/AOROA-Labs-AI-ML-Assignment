
import numpy as np
import pandas as pd


def create_validation_windows(
    daily_sales,
    horizon=7,
    n_windows=8
):
    """
    Create rolling historical validation windows.
    """

    daily_sales = (
        daily_sales
        .sort_values("date")
        .reset_index(drop=True)
    )

    validation_windows = []

    for i in range(n_windows, 0, -1):

        cutoff_idx = len(daily_sales) - horizon * i

        cutoff_date = daily_sales.loc[
            cutoff_idx,
            "date"
        ]

        forecast_start = (
            cutoff_date +
            pd.Timedelta(days=1)
        )

        forecast_end = (
            forecast_start +
            pd.Timedelta(days=horizon - 1)
        )

        validation_windows.append({
            "cutoff_date": cutoff_date,
            "forecast_start": forecast_start,
            "forecast_end": forecast_end
        })

    return pd.DataFrame(validation_windows)


def check_validation_leakage(
    validation_windows
):
    """
    Verify that every forecast starts after
    its training cutoff date.
    """

    for _, row in validation_windows.iterrows():

        cutoff = row["cutoff_date"]
        forecast_start = row["forecast_start"]

        assert forecast_start > cutoff

    return True


def get_complete_windows(
    validation_windows,
    daily_sales
):
    """
    Keep only validation windows for which
    the complete forecast horizon exists.
    """

    max_date = daily_sales["date"].max()

    complete_windows = validation_windows[
        validation_windows["forecast_end"] <= max_date
    ].copy()

    return complete_windows


def calculate_metrics(
    actual,
    forecast
):
    """
    Calculate MAE, RMSE and MAPE.
    """

    actual = np.asarray(actual)
    forecast = np.asarray(forecast)

    error = actual - forecast

    mae = np.mean(
        np.abs(error)
    )

    rmse = np.sqrt(
        np.mean(error ** 2)
    )

    mape = np.mean(
        np.abs(error / actual)
    ) * 100

    return {
        "MAE": mae,
        "RMSE": rmse,
        "MAPE": mape
    }


def calculate_window_metrics(
    results_df,
    model_name
):
    """
    Calculate MAE, RMSE and MAPE
    for each validation window.
    """

    rows = []

    for cutoff, group in results_df.groupby(
        "cutoff_date"
    ):

        metrics = calculate_metrics(
            group["actual"].values,
            group["forecast"].values
        )

        rows.append({
            "model": model_name,
            "cutoff_date": cutoff,
            "MAE": metrics["MAE"],
            "RMSE": metrics["RMSE"],
            "MAPE": metrics["MAPE"]
        })

    return pd.DataFrame(rows)


def create_stability_summary(
    window_metrics
):
    """
    Summarize model performance and
    stability across validation windows.
    """

    summary = (
        window_metrics
        .groupby("model")
        .agg(
            mean_mae=("MAE", "mean"),
            std_mae=("MAE", "std"),
            mean_rmse=("RMSE", "mean"),
            std_rmse=("RMSE", "std"),
            mean_mape=("MAPE", "mean"),
            std_mape=("MAPE", "std")
        )
        .reset_index()
    )

    return summary
