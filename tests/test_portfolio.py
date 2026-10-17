"""Tests for weights validation, valuation, alignment and summaries."""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from analytics.exceptions import (
    InsufficientDataError,
    InvalidInputError,
    InvalidWeightsError,
)
from analytics.portfolio import (
    align_price_frame,
    allocation_values,
    benchmark_comparison,
    build_portfolio_timeseries,
    normalize_weights,
    returns_summary,
    split_currency_totals,
    validate_weights,
)
from config import TRADING_DAYS_PER_YEAR
from services.currency import allocation_by_currency, combine_totals, format_currency, format_percent
from config import FxRate


# ---------------------------------------------------------------------------
# Weight validation
# ---------------------------------------------------------------------------
def test_validate_weights_valid_long_only():
    result = validate_weights({"A": 0.6, "B": 0.4})
    assert result.is_valid
    assert result.total == pytest.approx(1.0)
    assert result.n_assets == 2
    assert result.errors == []


def test_validate_weights_sum_error_reports_total():
    result = validate_weights({"A": 0.6, "B": 0.3})
    assert not result.is_valid
    assert "weights_sum" in result.error_codes
    issue = next(issue for issue in result.errors if issue.code == "weights_sum")
    assert issue.params["total"] == pytest.approx(0.9)


def test_validate_weights_rejects_negative():
    result = validate_weights({"A": 1.2, "B": -0.2})
    assert not result.is_valid
    assert "negative_weights" in result.error_codes


def test_validate_weights_rejects_above_one():
    result = validate_weights({"A": 1.5})
    assert "weights_above_max" in result.error_codes


def test_validate_weights_missing_and_non_numeric():
    result = validate_weights({"A": np.nan, "B": 1.0})
    assert "missing_weights" in result.error_codes or "non_numeric_weights" in result.error_codes


def test_validate_weights_empty():
    result = validate_weights({})
    assert not result.is_valid
    assert "empty_weights" in result.error_codes


def test_validate_weights_single_asset_warning():
    result = validate_weights({"A": 1.0})
    assert result.is_valid
    assert "single_asset_portfolio" in result.warning_codes


def test_validate_weights_dominant_position_warning():
    result = validate_weights({"A": 0.8, "B": 0.2})
    assert "dominant_position" in result.warning_codes


def test_validate_weights_without_full_sum_requirement():
    result = validate_weights({"A": 0.3}, require_full_sum=False)
    assert result.is_valid


def test_normalize_weights_renormalises():
    normalized = normalize_weights({"A": 0.5, "B": 0.25})
    assert normalized["A"] == pytest.approx(2 / 3)
    assert normalized.sum() == pytest.approx(1.0)


def test_normalize_weights_rejects_negative_and_zero_sum():
    with pytest.raises(InvalidWeightsError):
        normalize_weights({"A": -0.5, "B": 1.5})
    with pytest.raises(InvalidWeightsError):
        normalize_weights({"A": 0.0, "B": 0.0})


# ---------------------------------------------------------------------------
# Allocation / valuation
# ---------------------------------------------------------------------------
def test_allocation_values_hand_computed():
    values = allocation_values({"A": 0.6, "B": 0.4}, 1_000.0)
    assert values["A"] == pytest.approx(600.0)
    assert values["B"] == pytest.approx(400.0)
    assert values.sum() == pytest.approx(1_000.0)


def test_allocation_values_rejects_bad_capital():
    with pytest.raises(InvalidInputError):
        allocation_values({"A": 1.0}, float("nan"))
    with pytest.raises(InvalidInputError):
        allocation_values({"A": 1.0}, -100.0)


def test_split_currency_totals_keeps_currencies_separate():
    totals = split_currency_totals(
        {"AAPL": 0.5, "ATW": 0.5}, 10_000.0, {"AAPL": "USD", "ATW": "MAD"}
    )
    assert totals == {"USD": pytest.approx(5_000.0), "MAD": pytest.approx(5_000.0)}


def test_split_currency_totals_requires_metadata():
    with pytest.raises(InvalidInputError) as excinfo:
        split_currency_totals({"AAPL": 1.0}, 1_000.0, {})
    assert excinfo.value.code == "missing_currency"


# ---------------------------------------------------------------------------
# Currency consistency
# ---------------------------------------------------------------------------
def test_allocation_by_currency_single_currency_totals():
    result = allocation_by_currency({"A": 0.5, "B": 0.5}, 1_000.0, {"A": "USD", "B": "USD"})
    assert result.mixed is False
    assert result.combined_total == pytest.approx(1_000.0)
    assert result.combined_currency == "USD"


def test_allocation_by_currency_mixed_without_rate_reports_separately(monkeypatch):
    monkeypatch.delenv("FP_FX_RATE_USDMAD", raising=False)
    result = allocation_by_currency({"A": 0.5, "B": 0.5}, 1_000.0, {"A": "USD", "B": "MAD"})
    assert result.mixed is True
    assert result.combined_total is None
    assert result.currencies == ("MAD", "USD")


def test_allocation_by_currency_mixed_with_explicit_rate(monkeypatch):
    monkeypatch.setenv("FP_FX_RATE_USDMAD", "10")
    monkeypatch.setenv("FP_FX_RATE_DATE", "2024-06-30")
    result = allocation_by_currency({"A": 0.5, "B": 0.5}, 1_000.0, {"A": "USD", "B": "MAD"})
    assert result.mixed is True
    # MAD -> USD at 10 MAD per USD: 500 MAD = 50 USD, plus 500 USD = 550 USD.
    assert result.combined_total == pytest.approx(550.0)
    assert result.combined_currency == "USD"
    assert result.fx_rate is not None and result.fx_rate.rate == pytest.approx(10.0)


def test_combine_totals_without_rate_is_none(monkeypatch):
    monkeypatch.delenv("FP_FX_RATE_USDMAD", raising=False)
    total, currency = combine_totals({"USD": 100.0, "MAD": 900.0})
    assert total is None and currency is None


def test_combine_totals_with_rate_object():
    rate = FxRate(base="USD", quote="MAD", rate=10.0, as_of=date(2024, 1, 1), source="test")
    total, currency = combine_totals({"USD": 100.0, "MAD": 500.0}, rate)
    assert total == pytest.approx(150.0)
    assert currency == "USD"


def test_format_currency_bilingual():
    assert format_currency(1234.5, "USD", "en") == "$1,234.50 USD"
    assert "1\u202f234,50" in format_currency(1234.5, "EUR", "fr")
    assert format_currency(None, "USD", "en") == "—"
    assert format_currency(float("nan"), "MAD", "fr") == "—"


def test_format_percent_options():
    assert format_percent(0.1234, "en") == "12.34%"
    assert format_percent(-0.05, "en") == "-5.00%"
    assert format_percent(0.05, "en", signed=True) == "+5.00%"
    assert format_percent(12.5, "en", from_ratio=False) == "12.50%"


# ---------------------------------------------------------------------------
# Alignment
# ---------------------------------------------------------------------------
def test_align_price_frame_drops_rows_with_partial_missing():
    index = pd.bdate_range("2024-01-01", periods=5)
    prices = pd.DataFrame(
        {"A": [100.0, 101.0, np.nan, 103.0, 104.0], "B": [50.0, 51.0, 52.0, 53.0, 54.0]},
        index=index,
    )
    frame, report = align_price_frame(prices, ["A", "B"])
    assert len(frame) == 4
    assert report.dropped_rows == 1
    assert set(report.retained_symbols) == {"A", "B"}


def test_align_price_frame_excludes_low_coverage_symbol():
    index = pd.bdate_range("2024-01-01", periods=10)
    prices = pd.DataFrame(
        {
            "A": np.linspace(100, 110, 10),
            "B": [50.0] * 2 + [np.nan] * 8,
        },
        index=index,
    )
    frame, report = align_price_frame(prices, ["A", "B"], min_coverage=0.5)
    assert list(frame.columns) == ["A"]
    assert report.excluded_symbols == ("B",)
    assert "alignment.low_coverage_excluded" in report.warnings


def test_align_price_frame_no_overlap_raises():
    index = pd.bdate_range("2024-01-01", periods=3)
    prices = pd.DataFrame({"A": [100.0, np.nan, 102.0], "B": [np.nan, 51.0, np.nan]}, index=index)
    with pytest.raises(InsufficientDataError):
        align_price_frame(prices, ["A", "B"])


def test_align_price_frame_unknown_symbols_raise():
    index = pd.bdate_range("2024-01-01", periods=3)
    prices = pd.DataFrame({"A": [100.0, 101.0, 102.0]}, index=index)
    with pytest.raises(InvalidInputError):
        align_price_frame(prices, ["Z"])


def test_align_price_frame_invalid_coverage_argument():
    index = pd.bdate_range("2024-01-01", periods=3)
    prices = pd.DataFrame({"A": [100.0, 101.0, 102.0]}, index=index)
    with pytest.raises(InvalidInputError):
        align_price_frame(prices, ["A"], min_coverage=2.0)


# ---------------------------------------------------------------------------
# Portfolio valuation
# ---------------------------------------------------------------------------
def test_build_portfolio_timeseries_valuation(two_asset_prices):
    timeseries = build_portfolio_timeseries(
        two_asset_prices, {"A": 0.5, "B": 0.5}, initial_capital=1_000.0
    )
    assert timeseries.current_value == pytest.approx(
        1_000.0 * (1.0 + timeseries.portfolio_returns).prod()
    )
    assert timeseries.allocation.sum() == pytest.approx(1_000.0)
    assert timeseries.weights.sum() == pytest.approx(1.0)
    assert timeseries.n_observations > 200


def test_build_portfolio_timeseries_normalises_weights(two_asset_prices):
    timeseries = build_portfolio_timeseries(two_asset_prices, {"A": 3.0, "B": 1.0}, initial_capital=500.0)
    assert timeseries.weights["A"] == pytest.approx(0.75)
    assert timeseries.weights["B"] == pytest.approx(0.25)


def test_build_portfolio_timeseries_rejects_bad_weights(two_asset_prices):
    with pytest.raises(InvalidWeightsError):
        build_portfolio_timeseries(two_asset_prices, {"A": -1.0, "B": 2.0}, normalize=False)


def test_build_portfolio_timeseries_rejects_bad_capital(two_asset_prices):
    with pytest.raises(InvalidInputError):
        build_portfolio_timeseries(two_asset_prices, {"A": 1.0}, initial_capital=0.0)


def test_build_portfolio_timeseries_empty_weights(two_asset_prices):
    with pytest.raises(InvalidWeightsError):
        build_portfolio_timeseries(two_asset_prices, {})


def test_build_portfolio_timeseries_excludes_asset_without_history(two_asset_prices):
    timeseries = build_portfolio_timeseries(
        two_asset_prices, {"A": 0.5, "B": 0.5, "ZZ": 0.0}, initial_capital=1_000.0
    )
    assert "ZZ" not in timeseries.prices.columns
    assert timeseries.weights.sum() == pytest.approx(1.0)


def test_build_portfolio_timeseries_no_usable_asset():
    index = pd.bdate_range("2024-01-01", periods=3)
    prices = pd.DataFrame({"A": [100.0, 101.0, 102.0]}, index=index)
    with pytest.raises(InvalidInputError):
        build_portfolio_timeseries(prices, {"Z": 1.0})


# ---------------------------------------------------------------------------
# Summaries
# ---------------------------------------------------------------------------
def test_returns_summary_keys_and_values(two_asset_returns):
    returns = (two_asset_returns["A"] + two_asset_returns["B"]) / 2.0
    summary = returns_summary(returns, risk_free_rate=0.02, initial_capital=1_000.0)
    for key in (
        "total_return",
        "annualized_return",
        "annualized_volatility",
        "sharpe_ratio",
        "max_drawdown",
        "value_at_risk_95",
        "current_value",
    ):
        assert key in summary
    assert summary["current_value"] == pytest.approx(1_000.0 * (1.0 + returns).prod())
    assert summary["max_drawdown"] <= 0


def test_returns_summary_nan_for_unusable_metrics():
    returns = pd.Series([0.01] * 10)  # zero volatility -> Sharpe undefined
    summary = returns_summary(returns)
    assert np.isnan(summary["sharpe_ratio"])
    assert summary["annualized_volatility"] == pytest.approx(0.0)


def test_returns_summary_empty_raises():
    with pytest.raises(InsufficientDataError):
        returns_summary(pd.Series(dtype="float64"))


def test_benchmark_comparison_identical_series():
    index = pd.bdate_range("2024-01-01", periods=100)
    series = pd.Series(np.resize([0.01, -0.005], 100), index=index)
    comparison = benchmark_comparison(series, series)
    assert comparison["beta"] == pytest.approx(1.0)
    assert comparison["excess_annualized_return"] == pytest.approx(0.0, abs=1e-12)
    assert comparison["n_observations"] == 100


def test_benchmark_comparison_alignment_drops_missing():
    index = pd.bdate_range("2024-01-01", periods=4)
    portfolio = pd.Series([0.01, np.nan, 0.02, 0.01], index=index)
    benchmark = pd.Series([0.005, 0.001, 0.002, 0.0], index=index)
    comparison = benchmark_comparison(portfolio, benchmark)
    assert comparison["n_observations"] == 3
