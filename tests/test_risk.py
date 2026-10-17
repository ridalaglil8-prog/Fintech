"""Tests for the risk metrics (deterministic synthetic data)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from analytics.exceptions import (
    InsufficientDataError,
    InvalidInputError,
    InvalidWeightsError,
)
from analytics.risk import (
    beta_alpha,
    concentration_metrics,
    conditional_value_at_risk,
    downside_deviation,
    drawdown_magnitude,
    drawdown_series,
    information_ratio,
    max_drawdown,
    max_drawdown_details,
    portfolio_variance,
    portfolio_volatility,
    risk_contributions,
    sharpe_ratio,
    sortino_ratio,
    tracking_error,
    value_at_risk,
)
from config import TRADING_DAYS_PER_YEAR


# ---------------------------------------------------------------------------
# Drawdowns
# ---------------------------------------------------------------------------
def test_max_drawdown_exact():
    returns = pd.Series([0.10, -0.20, 0.05])
    # wealth: 1.10, 0.88, 0.924 ; running max: 1.10, 1.10, 1.10
    assert max_drawdown(returns) == pytest.approx(0.88 / 1.10 - 1.0)
    assert drawdown_magnitude(returns) == pytest.approx(1 - 0.88 / 1.10)


def test_drawdown_series_is_non_positive():
    returns = pd.Series([0.05, -0.10, 0.02, -0.03])
    series = drawdown_series(returns)
    assert (series <= 1e-12).all()
    assert series.iloc[0] == pytest.approx(0.0)


def test_max_drawdown_details_dates():
    index = pd.bdate_range("2024-01-01", periods=4)
    returns = pd.Series([0.10, -0.20, 0.25, 0.0], index=index)
    details = max_drawdown_details(returns)
    assert details.trough_date == index[1]
    assert details.peak_date == index[0]
    # 0.88 -> recovery above 1.10 requires the 25% day
    assert details.recovery_date == index[2]
    assert details.recovered is True
    assert details.max_drawdown == pytest.approx(0.88 / 1.10 - 1.0)


def test_max_drawdown_not_recovered():
    index = pd.bdate_range("2024-01-01", periods=3)
    returns = pd.Series([0.10, -0.30, 0.01], index=index)
    details = max_drawdown_details(returns)
    assert details.recovery_date is None
    assert details.recovered is False


def test_max_drawdown_requires_data():
    with pytest.raises(InsufficientDataError):
        max_drawdown(pd.Series([0.01]))


def test_max_drawdown_details_rejects_frame():
    frame = pd.DataFrame({"A": [0.01, 0.02]})
    with pytest.raises(InvalidInputError) as excinfo:
        max_drawdown_details(frame)
    assert excinfo.value.code == "expected_series"


# ---------------------------------------------------------------------------
# Variance / volatility from a covariance matrix
# ---------------------------------------------------------------------------
def test_portfolio_variance_hand_computed():
    covariance = pd.DataFrame(
        [[0.04, 0.0], [0.0, 0.01]], index=["A", "B"], columns=["A", "B"]
    )
    variance = portfolio_variance({"A": 0.5, "B": 0.5}, covariance)
    assert variance == pytest.approx(0.25 * 0.04 + 0.25 * 0.01)
    assert portfolio_volatility({"A": 0.5, "B": 0.5}, covariance) == pytest.approx(np.sqrt(0.0125))


def test_portfolio_volatility_annualises():
    covariance = pd.DataFrame([[0.0001, 0.0], [0.0, 0.0001]], index=["A", "B"], columns=["A", "B"])
    daily = portfolio_volatility({"A": 0.5, "B": 0.5}, covariance)
    annual = portfolio_volatility({"A": 0.5, "B": 0.5}, covariance, periods_per_year=TRADING_DAYS_PER_YEAR)
    assert annual == pytest.approx(daily * np.sqrt(TRADING_DAYS_PER_YEAR))


def test_portfolio_variance_requires_aligned_labels():
    covariance = pd.DataFrame([[0.04, 0.0], [0.0, 0.01]], index=["A", "B"], columns=["A", "B"])
    with pytest.raises(InvalidWeightsError) as excinfo:
        portfolio_variance({"A": 1.0, "Z": 0.0}, covariance)
    assert excinfo.value.code == "missing_covariance_assets"


def test_portfolio_variance_rejects_non_square():
    covariance = pd.DataFrame([[0.04, 0.0, 0.0]], index=["A"], columns=["A", "B", "C"])
    with pytest.raises(InvalidInputError):
        portfolio_variance({"A": 1.0}, covariance)


# ---------------------------------------------------------------------------
# Sharpe / Sortino
# ---------------------------------------------------------------------------
def test_sharpe_ratio_closed_form(known_returns):
    rf = 0.02
    expected_return = (1.02**126) - 1.0  # 1% average over 252 days, geometric
    expected_vol = 0.01 * np.sqrt(252 / 251) * np.sqrt(252)
    expected_sharpe = (expected_return - rf) / expected_vol
    assert sharpe_ratio(known_returns, risk_free_rate=rf) == pytest.approx(expected_sharpe, rel=1e-9)


def test_sharpe_ratio_zero_volatility_is_nan():
    returns = pd.Series([0.01] * 20)
    result = sharpe_ratio(returns)
    assert np.isnan(result)


def test_sharpe_ratio_rejects_non_finite_rf():
    with pytest.raises(InvalidInputError):
        sharpe_ratio(pd.Series([0.01, 0.02]), risk_free_rate=float("inf"))


def test_sortino_ratio_uses_downside_deviation():
    returns = pd.Series([0.02, -0.01, 0.02, -0.01] * 20)
    sortino = sortino_ratio(returns, risk_free_rate=0.0)
    sharpe = sharpe_ratio(returns, risk_free_rate=0.0)
    # Downside deviation is smaller than total volatility here, so Sortino > Sharpe.
    assert sortino > sharpe
    assert downside_deviation(returns) < returns.std(ddof=1) * np.sqrt(TRADING_DAYS_PER_YEAR)


# ---------------------------------------------------------------------------
# Tail risk
# ---------------------------------------------------------------------------
def test_value_at_risk_matches_historical_quantile():
    values = np.linspace(-0.04, 0.05, 100)
    returns = pd.Series(values)
    expected = max(0.0, -np.quantile(values, 0.05))
    assert value_at_risk(returns, level=0.95) == pytest.approx(expected)


def test_expected_shortfall_is_at_least_var():
    values = np.linspace(-0.05, 0.05, 200)
    returns = pd.Series(values)
    var = value_at_risk(returns, level=0.95)
    cvar = conditional_value_at_risk(returns, level=0.95)
    assert cvar >= var > 0


def test_value_at_risk_zero_when_all_returns_positive():
    returns = pd.Series([0.01, 0.02, 0.03])
    assert value_at_risk(returns, level=0.95) == pytest.approx(0.0)


def test_value_at_risk_rejects_bad_level():
    with pytest.raises(InvalidInputError):
        value_at_risk(pd.Series([0.01, 0.02]), level=0.2)


# ---------------------------------------------------------------------------
# Concentration
# ---------------------------------------------------------------------------
def test_concentration_equal_weights():
    metrics = concentration_metrics({"A": 0.25, "B": 0.25, "C": 0.25, "D": 0.25})
    assert metrics.hhi == pytest.approx(0.25)
    assert metrics.effective_n == pytest.approx(4.0)
    assert metrics.normalized_hhi == pytest.approx(0.0)
    assert metrics.top1_weight == pytest.approx(0.25)
    assert metrics.top3_weight == pytest.approx(0.75)
    assert metrics.label == "moderate"
    assert metrics.warnings == []


def test_concentration_single_asset():
    metrics = concentration_metrics({"A": 1.0})
    assert metrics.hhi == pytest.approx(1.0)
    assert metrics.effective_n == pytest.approx(1.0)
    assert metrics.label == "concentrated"
    assert "concentration.high_hhi" in metrics.warnings
    assert "concentration.dominant_position" in metrics.warnings


def test_concentration_diversified_threshold():
    weights = {f"A{i}": 1 / 10 for i in range(10)}
    metrics = concentration_metrics(weights)
    assert metrics.hhi == pytest.approx(0.1)
    assert metrics.label == "diversified"


def test_concentration_normalises_weights():
    metrics = concentration_metrics({"A": 50.0, "B": 50.0})
    assert metrics.hhi == pytest.approx(0.5)


def test_concentration_empty_weights_raise():
    with pytest.raises(InvalidWeightsError):
        concentration_metrics({})


# ---------------------------------------------------------------------------
# Risk contributions
# ---------------------------------------------------------------------------
def test_risk_contributions_sum_to_one():
    covariance = pd.DataFrame([[0.04, 0.01], [0.01, 0.09]], index=["A", "B"], columns=["A", "B"])
    contributions = risk_contributions({"A": 0.6, "B": 0.4}, covariance)
    assert contributions.sum() == pytest.approx(1.0)
    assert (contributions >= 0).all()


def test_risk_contributions_absolute_sum_to_volatility():
    covariance = pd.DataFrame([[0.04, 0.0], [0.0, 0.01]], index=["A", "B"], columns=["A", "B"])
    weights = {"A": 0.5, "B": 0.5}
    absolute = risk_contributions(weights, covariance, relative=False)
    assert absolute.sum() == pytest.approx(portfolio_volatility(weights, covariance))


def test_risk_contributions_zero_variance_raises():
    covariance = pd.DataFrame([[0.0, 0.0], [0.0, 0.0]], index=["A", "B"], columns=["A", "B"])
    with pytest.raises(InvalidInputError):
        risk_contributions({"A": 0.5, "B": 0.5}, covariance)


# ---------------------------------------------------------------------------
# Benchmark-relative
# ---------------------------------------------------------------------------
def test_beta_alpha_identical_series():
    index = pd.bdate_range("2024-01-01", periods=100)
    benchmark = pd.Series(np.resize([0.01, -0.005], 100), index=index)
    beta, alpha = beta_alpha(benchmark, benchmark, risk_free_rate=0.0)
    assert beta == pytest.approx(1.0)
    assert alpha == pytest.approx(0.0, abs=1e-12)


def test_beta_of_double_exposure():
    index = pd.bdate_range("2024-01-01", periods=100)
    benchmark = pd.Series(np.resize([0.01, -0.005], 100), index=index)
    portfolio = benchmark * 2.0
    beta, _ = beta_alpha(portfolio, benchmark)
    assert beta == pytest.approx(2.0)


def test_tracking_error_zero_for_identical_series():
    index = pd.bdate_range("2024-01-01", periods=50)
    series = pd.Series(np.resize([0.01, -0.02], 50), index=index)
    assert tracking_error(series, series) == pytest.approx(0.0)


def test_information_ratio_nan_when_no_active_risk():
    index = pd.bdate_range("2024-01-01", periods=50)
    series = pd.Series(np.resize([0.01, -0.02], 50), index=index)
    assert np.isnan(information_ratio(series, series))


def test_beta_alpha_insufficient_overlap():
    index = pd.bdate_range("2024-01-01", periods=5)
    a = pd.Series([0.01, 0.02, 0.01, 0.02, 0.01], index=index)
    b = pd.Series([np.nan] * 4 + [0.01], index=index)
    with pytest.raises(InsufficientDataError):
        beta_alpha(a, b)
