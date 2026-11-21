import numpy as np
import pandas as pd
import pytest

from src.validation import (
    calculate_metrics,
    check_validation_leakage,
    create_validation_windows,
    get_complete_windows,
    run_rolling_backtest,
)


def sample_daily_sales(days=180):
    dates = pd.date_range("2023-01-01", periods=days, freq="D")
    t = np.arange(days)
    sales = 5_000_000 + 250_000 * np.sin(2 * np.pi * t / 7) + 500 * t
    return pd.DataFrame({"date": dates, "total_sales": sales})


def test_rolling_backtest_refits_models_for_each_complete_window():
    daily = sample_daily_sales()
    windows = create_validation_windows(daily, horizon=7, n_windows=8)
    complete = get_complete_windows(windows, daily)

    assert len(windows) == 8
    assert len(complete) == 8
    assert check_validation_leakage(complete)

    predictions, per_window, summary = run_rolling_backtest(daily, complete)

    assert len(predictions) == 8 * 7 * 2
    assert set(predictions["model"]) == {
        "56-day Weekday Baseline",
        "Trend + Weekday + Fourier",
    }
    assert len(per_window) == 8 * 2
    assert set(summary["windows"]) == {8}
    assert np.isfinite(predictions["forecast"]).all()


def test_incomplete_horizon_is_excluded():
    daily = sample_daily_sales()
    windows = create_validation_windows(daily, horizon=7, n_windows=8)
    shortened = daily.iloc[:-3].copy()
    complete = get_complete_windows(windows, shortened)
    assert len(complete) < len(windows)


def test_mape_rejects_zero_actual_values():
    with pytest.raises(ValueError, match="MAPE is undefined"):
        calculate_metrics([0, 10], [1, 9])


def test_daily_sales_rejects_missing_calendar_day():
    daily = sample_daily_sales()
    daily = daily.drop(index=10)
    with pytest.raises(ValueError, match="continuous"):
        create_validation_windows(daily)
