"""Optimisation engine tests: constraints, normalisation, rejections.

Analytic reference values
-------------------------
Three asset return patterns are built so the covariance matrix is **exactly**
diagonal over any 4-day block (every cross product cancels):

* ``A``: ``[+0.01, -0.01]`` repeated         -> daily sigma 1%;
* ``B``: ``[+0.02, +0.02, -0.02, -0.02]``    -> daily sigma 2%;
* ``C``: ``[+0.03, -0.03, -0.03, +0.03]``    -> daily sigma 3%.

All means are exactly zero, so the minimum-variance portfolio has weights
proportional to ``1 / sigma_i^2`` i.e. ``1 : 1/4 : 1/9``, which normalises to
``(0.734694, 0.183673, 0.081633)``. With a 50% cap on every asset the optimum
pins ``A`` at the cap and splits the rest in the same ``1/sigma^2`` ratio:
``(0.5, 0.346154, 0.153846)``.
"""

from __future__ import annotations

import importlib.util

import numpy as np
import pandas as pd
import pytest

from optimization.optimizer import (
    MIN_WEIGHT_EPSILON,
    AvailableOptimizers,
    compare_allocations,
    evaluate_weights,
    optimize_portfolio,
    prepare_returns,
)
from analytics.exceptions import OptimizationError

TRADING_DAYS = 252

MIN_VOL_UNCAPPED = np.array([0.73469388, 0.18367347, 0.08163265])
MIN_VOL_CAPPED = np.array([0.5, 0.34615385, 0.15384615])


def _has(module: str) -> bool:
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ValueError):  # pragma: no cover - defensive
        return False


HAS_PYPFOPT = _has("pypfopt")
HAS_SCIPY = _has("scipy")


@pytest.fixture
def uncorrelated_returns() -> pd.DataFrame:
    """400 business days of exactly uncorrelated returns (vols 1%, 2%, 3%)."""
    n = 400
    index = pd.bdate_range("2021-01-04", periods=n)
    return pd.DataFrame(
        {
            "A": np.resize(np.array([0.01, -0.01]), n),
            "B": np.resize(np.array([0.02, 0.02, -0.02, -0.02]), n),
            "C": np.resize(np.array([0.03, -0.03, -0.03, 0.03]), n),
        },
        index=index,
    )


@pytest.fixture
def growth_returns() -> pd.DataFrame:
    """Two assets with positive geometric growth, zero covariance.

    ``A = [+0.0015, -0.0005]`` (mean 0.0005, sigma 0.001, daily) and
    ``B = [+0.00075, -0.00025, -0.00025, +0.00075]`` (mean 0.00025,
    sigma 0.0005). ``B`` is symmetric around its mean over each 4-day block,
    which makes ``cov(A, B)`` cancel exactly.
    """
    n = 400
    index = pd.bdate_range("2021-01-04", periods=n)
    return pd.DataFrame(
        {
            "A": np.resize(np.array([0.0015, -0.0005]), n),
            "B": np.resize(np.array([0.00075, -0.00025, -0.00025, 0.00075]), n),
        },
        index=index,
    )


def _expected_weights(returns: pd.DataFrame, method: str, rf: float = 0.0, bounds=(0.0, 1.0)):
    """Analytic optimum using the documented estimators (annualised, 252 days).

    ``mu`` follows PyPortfolioOpt's ``mean_historical_return(compounding=True)``:
    ``(1 + r).prod() ** (252 / count) - 1``; ``Sigma`` is the sample covariance
    (ddof=1) times 252. For two assets the solution is
    ``w ~ Sigma^-1 (mu - rf)`` normalised to sum to 1 (all the fixtures here
    have positive ``1' Sigma^-1 (mu - rf)`` and the unconstrained optimum
    already respects the bounds).
    """
    mu = (1.0 + returns).prod() ** (TRADING_DAYS / returns.count()) - 1.0
    covariance = returns.cov() * TRADING_DAYS
    if method == "min_volatility":
        target = np.ones(len(returns.columns))
    else:
        target = mu.to_numpy(dtype="float64") - rf
    solution = np.linalg.solve(covariance.to_numpy(dtype="float64"), target)
    solution = np.clip(solution, bounds[0], bounds[1])
    return solution / solution.sum()


# ---------------------------------------------------------------------------
# prepare_returns
# ---------------------------------------------------------------------------
def test_prepare_returns_keeps_clean_frame(uncorrelated_returns):
    clean, dropped, window = prepare_returns(uncorrelated_returns)
    assert dropped == {}
    assert list(clean.columns) == ["A", "B", "C"]
    assert len(clean) == 400
    assert window == (
        str(uncorrelated_returns.index[0].date()),
        str(uncorrelated_returns.index[-1].date()),
    )


def test_prepare_returns_rejects_empty():
    with pytest.raises(OptimizationError) as info:
        prepare_returns(pd.DataFrame())
    assert info.value.code == "optimization.no_data"


def test_prepare_returns_drops_zero_variance_asset(uncorrelated_returns):
    frame = uncorrelated_returns.assign(Z=0.001)
    clean, dropped, _ = prepare_returns(frame)
    assert "Z" not in clean.columns
    assert dropped["Z"] == "optimization.dropped_zero_variance"


def test_prepare_returns_drops_short_history(uncorrelated_returns):
    """A column with too few observations (but an acceptable gap ratio) is flagged."""
    frame = uncorrelated_returns[["A", "B"]].copy()
    frame.iloc[50:, frame.columns.get_loc("B")] = np.nan  # 50 observations
    with pytest.raises(OptimizationError) as info:
        prepare_returns(frame, max_missing_ratio=0.9)
    assert info.value.code == "optimization.not_enough_assets"
    assert info.value.params["dropped"] == {"B": "optimization.dropped_short_history"}


def test_prepare_returns_requires_enough_aligned_rows(uncorrelated_returns):
    """Both columns pass individually, but their overlap is too short."""
    frame = uncorrelated_returns[["A", "B"]].iloc[:200].copy()
    frame.loc[frame.index[160:], "A"] = np.nan  # A: 160 observations (20% gaps)
    frame.loc[frame.index[:40], "B"] = np.nan  # B: 160 observations (20% gaps)
    with pytest.raises(OptimizationError) as info:
        prepare_returns(frame, min_observations=150)
    assert info.value.code == "optimization.not_enough_observations"
    assert info.value.params["available"] == 120
    assert info.value.params["required"] == 150


def test_prepare_returns_rejects_mostly_missing_column(uncorrelated_returns):
    frame = uncorrelated_returns.copy()
    frame.iloc[:150, frame.columns.get_loc("C")] = np.nan  # 37.5% missing
    clean, dropped, _ = prepare_returns(frame)
    assert dropped["C"] == "optimization.dropped_missing"
    assert "C" not in clean.columns


def test_prepare_returns_drops_non_finite(uncorrelated_returns):
    frame = uncorrelated_returns.copy()
    frame.iloc[10:15, frame.columns.get_loc("B")] = np.inf
    clean, dropped, _ = prepare_returns(frame)
    assert dropped["B"] == "optimization.dropped_non_finite"
    assert "B" not in clean.columns


# ---------------------------------------------------------------------------
# min volatility
# ---------------------------------------------------------------------------
@pytest.mark.skipif(not (HAS_PYPFOPT or HAS_SCIPY), reason="no optimiser installed")
def test_min_volatility_matches_analytic_optimum(uncorrelated_returns):
    result = optimize_portfolio(uncorrelated_returns, method="min_volatility")
    assert result.engine in {"pypfopt", "scipy"}
    np.testing.assert_allclose(
        result.weights.to_numpy(), MIN_VOL_UNCAPPED, atol=1e-3
    )
    assert abs(float(result.weights.sum()) - 1.0) < 1e-9
    assert (result.weights >= -1e-12).all()
    assert list(result.assets) == ["A", "B", "C"]
    assert result.n_observations == 400
    assert "optimization.in_sample_warning" in result.notes


@pytest.mark.skipif(not (HAS_PYPFOPT or HAS_SCIPY), reason="no optimiser installed")
def test_min_volatility_respects_weight_cap(uncorrelated_returns):
    result = optimize_portfolio(
        uncorrelated_returns, method="min_volatility", weight_bounds=(0.0, 0.5)
    )
    np.testing.assert_allclose(result.weights.to_numpy(), MIN_VOL_CAPPED, atol=1e-3)
    assert float(result.weights.max()) <= 0.5 + 1e-6


@pytest.mark.skipif(not HAS_SCIPY, reason="scipy not installed")
def test_scipy_engine_matches_pypfopt(uncorrelated_returns):
    scipy_result = optimize_portfolio(
        uncorrelated_returns, method="min_volatility", prefer_engine="scipy"
    )
    assert scipy_result.engine == "scipy"
    np.testing.assert_allclose(scipy_result.weights.to_numpy(), MIN_VOL_UNCAPPED, atol=1e-3)
    if HAS_PYPFOPT:  # pragma: no cover - depends on the environment
        pypfopt_result = optimize_portfolio(
            uncorrelated_returns, method="min_volatility", prefer_engine="pypfopt"
        )
        np.testing.assert_allclose(
            pypfopt_result.weights.to_numpy(), scipy_result.weights.to_numpy(), atol=1e-3
        )


# ---------------------------------------------------------------------------
# max sharpe
# ---------------------------------------------------------------------------
@pytest.mark.skipif(not (HAS_PYPFOPT or HAS_SCIPY), reason="no optimiser installed")
def test_max_sharpe_matches_analytic_optimum(growth_returns):
    rf = 0.0
    result = optimize_portfolio(growth_returns, method="max_sharpe", risk_free_rate=rf)
    expected = _expected_weights(growth_returns, "max_sharpe", rf=rf)
    np.testing.assert_allclose(result.weights.to_numpy(), expected, atol=1e-3)
    assert result.sharpe > 0
    # The optimised Sharpe must beat a naive equal-weight allocation.
    _, _, equal_weight_sharpe = evaluate_weights(
        pd.Series({"A": 0.5, "B": 0.5}), growth_returns, risk_free_rate=rf
    )
    assert result.sharpe >= equal_weight_sharpe - 1e-9


@pytest.mark.skipif(not (HAS_PYPFOPT or HAS_SCIPY), reason="no optimiser installed")
def test_max_sharpe_prefers_dominating_asset():
    """``GOOD`` has a much higher Sharpe than ``BAD``: it must take all the weight.

    Both 4-day patterns are orthogonal (the cross products cancel), so the
    covariance matrix is diagonal and the long-only optimum is unambiguous.
    ``BAD`` has a negative expected return, so no mix can beat ``GOOD`` alone.
    """
    n = 400
    index = pd.bdate_range("2021-01-04", periods=n)
    good = np.array([0.002, 0.001, -0.001, 0.002])  # mean +0.1%, annualised ~+28%
    bad = np.array([0.002, -0.012, 0.004, 0.006])  # mean 0, annualised ~-0.6%
    returns = pd.DataFrame(
        {"GOOD": np.resize(good, n), "BAD": np.resize(bad, n)}, index=index
    )
    # Sanity check of the construction before trusting the assertion below.
    covariance = returns.cov().to_numpy()
    assert abs(covariance[0, 1]) < 1e-12
    result = optimize_portfolio(returns, method="max_sharpe")
    assert result.weights["GOOD"] > 0.99
    assert result.weights["BAD"] < 0.01


# ---------------------------------------------------------------------------
# validation / error codes
# ---------------------------------------------------------------------------
def test_unknown_method_is_rejected(uncorrelated_returns):
    with pytest.raises(OptimizationError) as info:
        optimize_portfolio(uncorrelated_returns, method="magic")
    assert info.value.code == "optimization.unknown_method"


@pytest.mark.parametrize(
    "bounds",
    [(0.2, 0.1), (-0.1, 0.5), (0.0, 1.5), (0.0,)],
)
def test_invalid_bounds_are_rejected(uncorrelated_returns, bounds):
    with pytest.raises(OptimizationError) as info:
        optimize_portfolio(uncorrelated_returns, weight_bounds=bounds)
    assert info.value.code == "optimization.invalid_bounds"


def test_invalid_risk_free_rate_is_rejected(uncorrelated_returns):
    with pytest.raises(OptimizationError) as info:
        optimize_portfolio(uncorrelated_returns, risk_free_rate=float("nan"))
    assert info.value.code == "optimization.invalid_risk_free_rate"


def test_unknown_engine_is_rejected(uncorrelated_returns):
    with pytest.raises(OptimizationError) as info:
        optimize_portfolio(uncorrelated_returns, prefer_engine="does-not-exist")
    assert info.value.code == "optimization.all_engines_failed"


def test_available_optimizers_describes_environment():
    info = AvailableOptimizers().describe()
    assert set(info) == {"pypfopt", "scipy", "engine"}
    assert info["engine"] in {"pypfopt", "scipy", "none"}


# ---------------------------------------------------------------------------
# evaluation / comparison
# ---------------------------------------------------------------------------
def test_evaluate_weights_reproduces_analytic_risk(uncorrelated_returns):
    weights = pd.Series({"A": 0.5, "B": 0.5})
    frame = uncorrelated_returns[["A", "B"]]
    expected_vol = float(np.sqrt(0.5**2 * 1e-4 + 0.5**2 * 4e-4) * 1.0012528 * np.sqrt(TRADING_DAYS))
    _, volatility, _ = evaluate_weights(weights, frame)
    assert volatility == pytest.approx(expected_vol, rel=1e-3)
    assert volatility == pytest.approx(0.1776, abs=1e-3)


def test_evaluate_weights_normalises_input(uncorrelated_returns):
    half = evaluate_weights(pd.Series({"A": 1.0, "B": 1.0}), uncorrelated_returns[["A", "B"]])
    balanced = evaluate_weights(pd.Series({"A": 0.5, "B": 0.5}), uncorrelated_returns[["A", "B"]])
    assert half == pytest.approx(balanced)


def test_evaluate_weights_rejects_unknown_symbols(uncorrelated_returns):
    from analytics.exceptions import InvalidWeightsError

    with pytest.raises(InvalidWeightsError):
        evaluate_weights(pd.Series({"ZZZ": 1.0}), uncorrelated_returns)


@pytest.mark.skipif(not (HAS_PYPFOPT or HAS_SCIPY), reason="no optimiser installed")
def test_compare_allocations_frame_and_metrics(uncorrelated_returns):
    result = optimize_portfolio(uncorrelated_returns, method="min_volatility")
    current = pd.Series({"A": 0.5, "B": 0.3, "C": 0.2})
    frame = compare_allocations(current, result, uncorrelated_returns)

    assert list(frame.columns) == ["current", "optimized", "delta"]
    assert list(frame.index) == ["A", "B", "C"]
    pd.testing.assert_series_equal(frame["delta"], frame["optimized"] - frame["current"], check_names=False)
    assert float(frame["optimized"].sum()) == pytest.approx(1.0, abs=1e-9)
    assert float(frame["current"].sum()) == pytest.approx(1.0, abs=1e-9)

    metrics = frame.attrs["metrics"]
    assert set(metrics) == {"current", "optimized"}
    assert metrics["optimized"]["volatility"] <= metrics["current"]["volatility"] + 1e-9
    assert metrics["optimized"]["volatility"] == pytest.approx(result.volatility)


def test_weights_are_cleaned_of_numerical_dust(uncorrelated_returns):
    """Weights below MIN_WEIGHT_EPSILON are zeroed and renormalised."""
    from optimization.optimizer import _clean_weights

    cleaned = _clean_weights(np.array([0.5, 0.4999999999, 1e-12]), ["A", "B", "C"], (0.0, 1.0))
    assert cleaned["C"] == 0.0
    assert float(cleaned.sum()) == pytest.approx(1.0, abs=1e-12)
    assert float(cleaned.min()) >= MIN_WEIGHT_EPSILON - 1e-12
