
import pandas as pd


def load_daily_sales(project_dir):
    """
    Load daily sales data from the project results directory.
    """
    path = f"{project_dir}/results/daily_sales.csv"

    daily_sales = pd.read_csv(
        path,
        parse_dates=["date"]
    )

    return (
        daily_sales
        .sort_values("date")
        .reset_index(drop=True)
    )


def calculate_weekday_baseline(daily_sales, window_days=56):
    """
    Calculate weekday-average baseline using the
    most recent historical days.
    """

    baseline_window = daily_sales.tail(window_days).copy()

    baseline_window["weekday"] = (
        baseline_window["date"].dt.day_name()
    )

    weekday_baseline = (
        baseline_window
        .groupby("weekday")["total_sales"]
        .mean()
    )

    return weekday_baseline


def create_baseline_forecast(
    daily_sales,
    weekday_baseline,
    horizon=7
):
    """
    Create a future forecast using weekday averages.
    """

    last_date = daily_sales["date"].max()

    forecast_dates = pd.date_range(
        start=last_date + pd.Timedelta(days=1),
        periods=horizon,
        freq="D"
    )

    forecast = pd.DataFrame({
        "date": forecast_dates
    })

    forecast["weekday"] = (
        forecast["date"].dt.day_name()
    )

    forecast["forecast"] = (
        forecast["weekday"]
        .map(weekday_baseline)
    )

    return forecast
