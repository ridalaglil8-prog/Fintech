"""Tests for the return calculations (deterministic synthetic data)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from analytics.exceptions import InsufficientDataError, InvalidInputError
from analytics.returns import (
    align_returns_to_weights,
    annualized_return,
    annualized_volatility,
    cumulative_returns,
    cumulative_wealth,
    daily_returns,
    portfolio_returns,
    total_return,
    validate_price_frame,
    wealth_index,
)
from config import TRADING_DAYS_PER_YEAR


# ---------------------------------------------------------------------------
# Daily returns
# ---------------------------------------------------------------------------
def test_daily_returns_simple_exact(constant_growth_prices):
    returns = daily_returns(constant_growth_prices)
    assert len(returns) == 2
    assert returns.iloc[0] == pytest.approx(0.10)
    assert returns.iloc[1] == pytest.approx(0.10)


def test_daily_returns_log_exact(constant_growth_prices):
    returns = daily_returns(constant_growth_prices, method="log")
    assert returns.iloc[0] == pytest.approx(np.log(1.10))
    assert returns.iloc[1] == pytest.approx(np.log(1.10))


def test_daily_returns_keeps_internal_gaps_as_nan():
    index = pd.bdate_range("2024-01-01", periods=3)
    prices = pd.Series([100.0, np.nan, 121.0], index=index)
    returns = daily_returns(prices, drop_leading_na=False)
    assert returns.isna().all()
    assert len(returns) == 3
    # With the leading NaN removed, the two internal observations remain NaN:
    # a missing price is never turned into a fabricated return.
    trimmed = daily_returns(prices, drop_leading_na=True)
    assert trimmed.isna().all()
    assert len(trimmed) == 2


def test_daily_returns_rejects_non_positive_prices():
    prices = pd.Series([100.0, 0.0, 120.0], index=pd.bdate_range("2024-01-01", periods=3))
    with pytest.raises(InvalidInputError) as excinfo:
        daily_returns(prices)
    assert excinfo.value.code == "non_positive_prices"


def test_daily_returns_rejects_non_numeric():
    prices = pd.Series(["100", "110"], index=pd.bdate_range("2024-01-01", periods=2))
    with pytest.raises(InvalidInputError) as excinfo:
        daily_returns(prices)
    assert excinfo.value.code == "non_numeric_input"


def test_daily_returns_rejects_infinite_values():
    prices = pd.Series([100.0, np.inf], index=pd.bdate_range("2024-01-01", periods=2))
    with pytest.raises(InvalidInputError) as excinfo:
        daily_returns(prices)
    assert excinfo.value.code == "non_finite_input"


def test_daily_returns_unknown_method():
    prices = pd.Series([100.0, 110.0], index=pd.bdate_range("2024-01-01", periods=2))
    with pytest.raises(InvalidInputError):
        daily_returns(prices, method="geometric")


def test_validate_price_frame_accepts_missing_values():
    prices = pd.DataFrame(
        {"A": [100.0, np.nan, 120.0], "B": [50.0, 51.0, 52.0]},
        index=pd.bdate_range("2024-01-01", periods=3),
    )
    validate_price_frame(prices)  # must not raise


# ---------------------------------------------------------------------------
# Cumulative / total / annualised
# ---------------------------------------------------------------------------
def test_cumulative_returns_compounds_exactly():
    returns = pd.Series([0.10, 0.10])
    cumulative = cumulative_returns(returns)
    assert cumulative.iloc[0] == pytest.approx(0.10)
    assert cumulative.iloc[1] == pytest.approx(0.21)


def test_cumulative_wealth_and_total_return():
    returns = pd.Series([0.10, -0.10])
    wealth = cumulative_wealth(returns)
    assert wealth.iloc[-1] == pytest.approx(1.10 * 0.90)
    assert total_return(returns) == pytest.approx(1.10 * 0.90 - 1.0)


def test_wealth_index_scales_with_initial_capital():
    returns = pd.Series([0.01, 0.01])
    wealth = wealth_index(returns, initial_value=1_000.0)
    assert wealth.iloc[-1] == pytest.approx(1_000.0 * 1.01 * 1.01)


def test_annualized_return_252_periods_is_geometric():
    daily = 0.0004
    returns = pd.Series([daily] * TRADING_DAYS_PER_YEAR)
    expected = (1.0 + daily) ** TRADING_DAYS_PER_YEAR - 1.0
    assert annualized_return(returns) == pytest.approx(expected, rel=1e-12)


def test_annualized_return_requires_two_observations():
    with pytest.raises(InsufficientDataError) as excinfo:
        annualized_return(pd.Series([0.01]))
    assert excinfo.value.code == "insufficient_observations"
    assert excinfo.value.params["required"] == 2


def test_annualized_return_rejects_bad_annualisation_factor():
    with pytest.raises(InvalidInputError):
        annualized_return(pd.Series([0.01, 0.02]), periods_per_year=0)


def test_annualized_return_on_dataframe_returns_series():
    frame = pd.DataFrame({"A": [0.01, 0.02, 0.03], "B": [0.0, 0.0, 0.0]})
    result = annualized_return(frame)
    assert isinstance(result, pd.Series)
    assert result["B"] == pytest.approx(0.0)
    assert result["A"] > 0


# ---------------------------------------------------------------------------
# Volatility
# ---------------------------------------------------------------------------
def test_annualized_volatility_matches_sample_std():
    values = np.array([0.01, 0.02, 0.03, 0.04])
    returns = pd.Series(values)
    expected = values.std(ddof=1) * np.sqrt(TRADING_DAYS_PER_YEAR)
    assert annualized_volatility(returns) == pytest.approx(expected, rel=1e-12)


def test_annualized_volatility_zero_for_constant_returns():
    returns = pd.Series([0.005] * 10)
    assert annualized_volatility(returns) == pytest.approx(0.0)


def test_annualized_volatility_needs_two_observations():
    with pytest.raises(InsufficientDataError):
        annualized_volatility(pd.Series([0.01]))


# ---------------------------------------------------------------------------
# Portfolio returns
# ---------------------------------------------------------------------------
def test_portfolio_returns_weighted_average():
    index = pd.bdate_range("2024-01-01", periods=2)
    asset_returns = pd.DataFrame({"A": [0.02, 0.00], "B": [0.00, 0.04]}, index=index)
    result = portfolio_returns(asset_returns, {"A": 0.5, "B": 0.5})
    assert result.iloc[0] == pytest.approx(0.01)
    assert result.iloc[1] == pytest.approx(0.02)


def test_portfolio_returns_renormalises_available_weights():
    index = pd.bdate_range("2024-01-01", periods=2)
    asset_returns = pd.DataFrame({"A": [0.02, 0.02]}, index=index)
    # B is absent: the remaining weight is renormalised to 100%.
    result = portfolio_returns(asset_returns, {"A": 0.5, "B": 0.5})
    assert result.iloc[0] == pytest.approx(0.02)


def test_portfolio_returns_without_common_assets_raises():
    index = pd.bdate_range("2024-01-01", periods=2)
    asset_returns = pd.DataFrame({"A": [0.02, 0.02]}, index=index)
    with pytest.raises(InvalidInputError) as excinfo:
        portfolio_returns(asset_returns, {"Z": 1.0})
    assert excinfo.value.code == "no_common_assets"


def test_portfolio_returns_zero_weight_assets_are_ignored():
    index = pd.bdate_range("2024-01-01", periods=2)
    asset_returns = pd.DataFrame({"A": [0.02, 0.02], "B": [np.nan, np.nan]}, index=index)
    result = portfolio_returns(asset_returns, {"A": 1.0, "B": 0.0})
    assert result.tolist() == pytest.approx([0.02, 0.02])


def test_align_returns_drops_rows_with_missing_components():
    index = pd.bdate_range("2024-01-01", periods=3)
    asset_returns = pd.DataFrame({"A": [0.01, np.nan, 0.02], "B": [0.01, 0.01, 0.02]}, index=index)
    frame, weights = align_returns_to_weights(asset_returns, {"A": 0.5, "B": 0.5})
    assert len(frame) == 2  # the row with a missing observation is dropped
    assert weights.sum() == pytest.approx(1.0)


def test_align_returns_empty_frame_raises():
    with pytest.raises(InsufficientDataError):
        align_returns_to_weights(pd.DataFrame(), {"A": 1.0})


def test_align_returns_non_numeric_weights_raise():
    index = pd.bdate_range("2024-01-01", periods=2)
    asset_returns = pd.DataFrame({"A": [0.01, 0.02]}, index=index)
    with pytest.raises(InvalidInputError) as excinfo:
        align_returns_to_weights(asset_returns, {"A": "abc"})
    assert excinfo.value.code in {"non_numeric_weights", "empty_weights"}
