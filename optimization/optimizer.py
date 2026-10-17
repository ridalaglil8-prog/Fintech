"""Portfolio optimisation with PyPortfolioOpt (SciPy fallback).

Methods
-------
* ``min_volatility`` — minimise ``sqrt(w' Sigma w)`` subject to
  ``sum(w) = 1`` and ``lb <= w <= ub`` (long-only by default).
* ``max_sharpe`` — maximise the historical Sharpe ratio
  ``(w' mu - rf) / sqrt(w' Sigma w)`` under the same constraints.

Inputs are **daily returns**: ``mu`` is the annualised historical mean return
(252 periods) and ``Sigma`` the annualised sample covariance matrix.

Honest reporting
----------------
Optimisation is an *in-sample* exercise: the numbers below describe the past,
not the future, and are highly sensitive to the estimation window, the
estimation error of ``mu`` and ``Sigma`` and the constraints chosen. Every
output therefore carries ``n_observations``, the estimation window and the
engine used, and the UI displays the accompanying caveat. Nothing here is a
promise of future performance.

Reliability guards (a result is never produced from unusable data):

* at least two assets and :data:`config.MIN_OBSERVATIONS_FOR_OPTIMISATION`
  aligned observations are required;
* assets with fewer observations or zero variance are dropped (and reported);
* weights are cleaned, clipped to the bounds and renormalised to sum to 1;
* a failed solver raises :class:`~analytics.exceptions.OptimizationError` with a
  translation-ready code instead of returning arbitrary weights.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from analytics.returns import annualized_return, annualized_volatility
from analytics.risk import portfolio_volatility, sharpe_ratio
from analytics.exceptions import InsufficientDataError, InvalidWeightsError, OptimizationError
from config import (
    MIN_OBSERVATIONS_FOR_OPTIMISATION,
    TRADING_DAYS_PER_YEAR,
)

__all__ = [
    "OptimizationResult",
    "AvailableOptimizers",
    "available_methods",
    "optimize_portfolio",
    "evaluate_weights",
    "compare_allocations",
    "MIN_WEIGHT_EPSILON",
]

MIN_WEIGHT_EPSILON = 1e-6

METHODS = ("min_volatility", "max_sharpe")

METHOD_LABEL_KEYS = {
    "min_volatility": "optimization.method.min_volatility",
    "max_sharpe": "optimization.method.max_sharpe",
}


class AvailableOptimizers:
    """Describe which optimisation backends are importable in this environment."""

    def __init__(self) -> None:
        self.pypfopt = self._has("pypfopt")
        self.scipy = self._has("scipy")

    @staticmethod
    def _has(module: str) -> bool:
        import importlib.util

        try:
            return importlib.util.find_spec(module) is not None
        except (ImportError, ValueError):  # pragma: no cover - defensive
            return False

    @property
    def any_available(self) -> bool:
        return self.pypfopt or self.scipy

    def describe(self) -> dict:
        return {
            "pypfopt": self.pypfopt,
            "scipy": self.scipy,
            "engine": "pypfopt" if self.pypfopt else ("scipy" if self.scipy else "none"),
        }


def available_methods() -> tuple[str, ...]:
    """Supported optimisation methods (keys used in the UI and the API)."""
    return METHODS


@dataclass
class OptimizationResult:
    """Outcome of an optimisation run."""

    method: str
    weights: pd.Series
    expected_return: float  #: annualised historical mean return of the portfolio
    volatility: float  #: annualised historical volatility
    sharpe: float  #: historical Sharpe ratio (rf from the settings)
    engine: str  #: "pypfopt" or "scipy"
    n_observations: int
    risk_free_rate: float
    weight_bounds: tuple[float, float]
    assets: tuple[str, ...] = ()
    dropped_assets: dict[str, str] = field(default_factory=dict)  #: symbol -> reason code
    price_range: tuple[str | None, str | None] = (None, None)
    notes: tuple[str, ...] = ()

    @property
    def label_key(self) -> str:
        return METHOD_LABEL_KEYS.get(self.method, "optimization.method.custom")

    def as_dict(self) -> dict:
        return {
            "method": self.method,
            "weights": {str(k): float(v) for k, v in self.weights.items()},
            "expected_return": float(self.expected_return),
            "volatility": float(self.volatility),
            "sharpe": float(self.sharpe),
            "engine": self.engine,
            "n_observations": int(self.n_observations),
            "risk_free_rate": float(self.risk_free_rate),
            "weight_bounds": [float(self.weight_bounds[0]), float(self.weight_bounds[1])],
            "assets": list(self.assets),
            "dropped_assets": dict(self.dropped_assets),
            "notes": list(self.notes),
        }


# ---------------------------------------------------------------------------
# Input preparation
# ---------------------------------------------------------------------------
def prepare_returns(
    returns: pd.DataFrame,
    *,
    min_observations: int = MIN_OBSERVATIONS_FOR_OPTIMISATION,
    max_missing_ratio: float = 0.2,
) -> tuple[pd.DataFrame, dict[str, str], tuple[str | None, str | None]]:
    """Validate and clean the return matrix used for optimisation.

    Returns ``(clean_returns, dropped_assets, price_range)`` where
    ``dropped_assets`` maps a symbol to the reason it was excluded.
    """
    if returns is None or returns.empty:
        raise OptimizationError(
            "No return history available for optimisation.",
            code="optimization.no_data",
        )
    frame = returns.copy()
    if not isinstance(frame.index, pd.DatetimeIndex):
        try:
            frame.index = pd.to_datetime(frame.index)
        except Exception as exc:  # pragma: no cover - defensive
            raise OptimizationError(
                "The return index cannot be interpreted as dates.",
                code="optimization.invalid_dates",
            ) from exc

    dropped: dict[str, str] = {}
    missing_ratio = frame.isna().mean()
    for symbol, ratio in missing_ratio.items():
        if ratio > max_missing_ratio:
            dropped[str(symbol)] = "optimization.dropped_missing"
    frame = frame.drop(columns=[str(column) for column in frame.columns if str(column) in dropped])

    usable: list[str] = []
    for symbol in frame.columns:
        series = frame[symbol].dropna()
        if len(series) < min_observations:
            dropped[str(symbol)] = "optimization.dropped_short_history"
            continue
        if not np.isfinite(series.to_numpy(dtype="float64")).all():
            dropped[str(symbol)] = "optimization.dropped_non_finite"
            continue
        if float(series.var(ddof=1)) <= 0:
            dropped[str(symbol)] = "optimization.dropped_zero_variance"
            continue
        usable.append(str(symbol))

    frame = frame.loc[:, usable]
    if frame.shape[1] < 2:
        raise OptimizationError(
            "At least two assets with a usable history are required.",
            code="optimization.not_enough_assets",
            params={"available": int(frame.shape[1]), "dropped": dropped},
        )

    aligned = frame.dropna(how="any")
    if len(aligned) < min_observations:
        raise OptimizationError(
            f"Only {len(aligned)} aligned observations, {min_observations} required.",
            code="optimization.not_enough_observations",
            params={
                "available": int(len(aligned)),
                "required": int(min_observations),
            },
        )
    price_range = (
        str(aligned.index[0].date()) if len(aligned) else None,
        str(aligned.index[-1].date()) if len(aligned) else None,
    )
    return aligned, dropped, price_range


def _validate_bounds(bounds: Sequence[float] | tuple[float, float]) -> tuple[float, float]:
    if len(bounds) != 2:
        raise OptimizationError(
            "Weight bounds must be a (lower, upper) pair.",
            code="optimization.invalid_bounds",
            params={"bounds": list(bounds)},
        )
    lower, upper = float(bounds[0]), float(bounds[1])
    if not np.isfinite(lower) or not np.isfinite(upper) or lower < 0 or upper > 1 or lower > upper:
        raise OptimizationError(
            "Weight bounds must satisfy 0 <= lower <= upper <= 1 (long-only MVP).",
            code="optimization.invalid_bounds",
            params={"lower": lower, "upper": upper},
        )
    return lower, upper


def _clean_weights(
    weights: np.ndarray, labels: Sequence[str], bounds: tuple[float, float]
) -> pd.Series:
    """Clip to the bounds, zero out numerical dust and renormalise to sum 1."""
    lower, upper = bounds
    values = np.clip(np.asarray(weights, dtype="float64"), lower, upper)
    values[values < MIN_WEIGHT_EPSILON] = 0.0
    total = float(values.sum())
    if not np.isfinite(total) or total <= 0:
        raise OptimizationError(
            "The optimiser returned unusable weights.",
            code="optimization.invalid_result",
        )
    values = values / total
    # A second clip keeps the bounds respected after renormalisation.
    values = np.clip(values, lower, upper)
    total = float(values.sum())
    if total <= 0:  # pragma: no cover - defensive
        raise OptimizationError(
            "The optimiser returned unusable weights.",
            code="optimization.invalid_result",
        )
    values = values / total
    return pd.Series(values, index=list(labels), name="weight")


# ---------------------------------------------------------------------------
# Backends
# ---------------------------------------------------------------------------
def _historical_mean_returns(returns: pd.DataFrame) -> pd.Series:
    """Annualised historical mean return, across PyPortfolioOpt versions.

    The keyword was renamed ``compounded`` -> ``compounding`` in 1.6; both mean
    "compound the daily returns", which is the convention used in this project.
    """
    from pypfopt import expected_returns

    try:
        return expected_returns.mean_historical_return(
            returns,
            returns_data=True,
            compounding=True,
            frequency=TRADING_DAYS_PER_YEAR,
        )
    except TypeError:  # PyPortfolioOpt < 1.6
        return expected_returns.mean_historical_return(
            returns, returns_data=True, frequency=TRADING_DAYS_PER_YEAR
        )


def _optimize_with_pypfopt(
    returns: pd.DataFrame, method: str, bounds: tuple[float, float], risk_free_rate: float
) -> tuple[pd.Series, str]:
    from pypfopt import EfficientFrontier, risk_models

    mu = _historical_mean_returns(returns)
    covariance = risk_models.sample_cov(
        returns, returns_data=True, frequency=TRADING_DAYS_PER_YEAR
    )
    frontier = EfficientFrontier(mu, covariance, weight_bounds=bounds)
    if method == "min_volatility":
        frontier.min_volatility()
    else:
        frontier.max_sharpe(risk_free_rate=risk_free_rate)
    raw = frontier.clean_weights(cutoff=MIN_WEIGHT_EPSILON, rounding=None)
    weights = pd.Series({str(k): float(v) for k, v in raw.items()}, dtype="float64")
    weights = weights.reindex([str(column) for column in returns.columns]).fillna(0.0)
    return _clean_weights(weights.to_numpy(), weights.index, bounds), "pypfopt"


def _optimize_with_scipy(
    returns: pd.DataFrame, method: str, bounds: tuple[float, float], risk_free_rate: float
) -> tuple[pd.Series, str]:
    from scipy.optimize import minimize

    mu = returns.mean() * TRADING_DAYS_PER_YEAR
    covariance = returns.cov() * TRADING_DAYS_PER_YEAR
    matrix = covariance.to_numpy(dtype="float64")
    mean_vector = mu.to_numpy(dtype="float64")
    n_assets = len(mean_vector)
    if not np.isfinite(matrix).all() or not np.isfinite(mean_vector).all():
        raise OptimizationError(
            "The covariance matrix or expected returns contain non-finite values.",
            code="optimization.non_finite_inputs",
        )

    def variance(weights: np.ndarray) -> float:
        return float(weights @ matrix @ weights)

    def neg_sharpe(weights: np.ndarray) -> float:
        volatility = np.sqrt(max(variance(weights), 0.0))
        if volatility <= 0:
            return 1e9
        return -float((weights @ mean_vector - risk_free_rate) / volatility)

    objective = variance if method == "min_volatility" else neg_sharpe
    constraints = ({"type": "eq", "fun": lambda w: float(np.sum(w) - 1.0)},)
    scipy_bounds = tuple((bounds[0], bounds[1]) for _ in range(n_assets))
    initial = np.full(n_assets, 1.0 / n_assets)
    solution = minimize(
        objective,
        initial,
        method="SLSQP",
        bounds=scipy_bounds,
        constraints=constraints,
        options={"maxiter": 500, "ftol": 1e-12},
    )
    if not solution.success or not np.isfinite(solution.x).all():  # pragma: no cover - solver dependent
        raise OptimizationError(
            f"The numerical optimiser did not converge ({solution.message}).",
            code="optimization.solver_failed",
            params={"message": str(solution.message)[:200]},
        )
    return _clean_weights(solution.x, [str(column) for column in returns.columns], bounds), "scipy"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def optimize_portfolio(
    returns: pd.DataFrame,
    *,
    method: str = "min_volatility",
    weight_bounds: tuple[float, float] = (0.0, 1.0),
    risk_free_rate: float = 0.0,
    min_observations: int = MIN_OBSERVATIONS_FOR_OPTIMISATION,
    prefer_engine: str = "auto",
) -> OptimizationResult:
    """Optimise portfolio weights from a daily return matrix.

    Parameters
    ----------
    returns:
        Daily returns (one column per asset), indexed by date.
    method:
        ``"min_volatility"`` or ``"max_sharpe"``.
    weight_bounds:
        Long-only bounds, e.g. ``(0.0, 0.25)`` to cap every asset at 25%.
    risk_free_rate:
        Annualised risk-free rate used by ``max_sharpe`` and by the reported
        Sharpe ratio.
    prefer_engine:
        ``"auto"`` (PyPortfolioOpt then SciPy), ``"pypfopt"`` or ``"scipy"``.
    """
    if method not in METHODS:
        raise OptimizationError(
            f"Unknown optimisation method '{method}'.",
            code="optimization.unknown_method",
            params={"method": method, "available": list(METHODS)},
        )
    bounds = _validate_bounds(weight_bounds)
    if not np.isfinite(risk_free_rate):
        raise OptimizationError(
            "The risk-free rate must be a finite number.",
            code="optimization.invalid_risk_free_rate",
        )
    clean_returns, dropped, window = prepare_returns(returns, min_observations=min_observations)

    engines: list[str]
    if prefer_engine == "auto":
        engines = ["pypfopt", "scipy"]
    else:
        engines = [prefer_engine]

    weights: pd.Series | None = None
    used_engine = "none"
    errors: list[str] = []
    for engine in engines:
        try:
            if engine == "pypfopt":
                weights, used_engine = _optimize_with_pypfopt(clean_returns, method, bounds, risk_free_rate)
            elif engine == "scipy":
                weights, used_engine = _optimize_with_scipy(clean_returns, method, bounds, risk_free_rate)
            else:  # pragma: no cover - guarded by the caller
                raise OptimizationError(
                    f"Unknown optimisation engine '{engine}'.",
                    code="optimization.unknown_engine",
                    params={"engine": engine},
                )
            break
        except ImportError:
            errors.append(f"{engine}:not installed")
            continue
        except OptimizationError as exc:
            errors.append(f"{engine}:{exc.code}")
            continue
        except Exception as exc:  # solver failures, degenerate matrices...
            errors.append(f"{engine}:{type(exc).__name__}")
            continue

    if weights is None:
        raise OptimizationError(
            "Optimisation failed with every available engine: " + "; ".join(errors),
            code="optimization.all_engines_failed",
            params={"errors": errors},
        )

    expected_return, volatility, sharpe = evaluate_weights(
        weights, clean_returns, risk_free_rate=risk_free_rate
    )
    return OptimizationResult(
        method=method,
        weights=weights,
        expected_return=expected_return,
        volatility=volatility,
        sharpe=sharpe,
        engine=used_engine,
        n_observations=int(len(clean_returns)),
        risk_free_rate=float(risk_free_rate),
        weight_bounds=bounds,
        assets=tuple(str(column) for column in clean_returns.columns),
        dropped_assets=dropped,
        price_range=window,
        notes=("optimization.in_sample_warning",),
    )


def evaluate_weights(
    weights: Mapping[str, float] | pd.Series,
    returns: pd.DataFrame,
    *,
    risk_free_rate: float = 0.0,
) -> tuple[float, float, float]:
    """Return ``(annualised return, annualised volatility, Sharpe)`` of a portfolio.

    Metrics are computed on the **asset returns weighted by the given weights**
    (``r_p,t = sum_i w_i r_i,t``) so the number is consistent with the
    dashboard, rather than using the optimiser's own estimates.
    """
    series = weights if isinstance(weights, pd.Series) else pd.Series(dict(weights), dtype="float64")
    available = [str(symbol) for symbol in series.index if str(symbol) in returns.columns]
    if not available:
        raise InvalidWeightsError(
            "None of the weights match the available return columns.",
            code="no_common_assets",
        )
    aligned = series.loc[available].astype("float64")
    if aligned.sum() <= 0:
        raise InvalidWeightsError("Weights sum to zero.", code="zero_total_weight")
    aligned = aligned / aligned.sum()
    frame = returns.loc[:, available].dropna(how="any")
    if len(frame) < 2:
        raise InsufficientDataError(
            "Not enough aligned observations to evaluate the allocation.",
            code="insufficient_observations",
            params={"available": int(len(frame)), "required": 2, "what": "allocation evaluation"},
        )
    portfolio = frame.mul(aligned, axis=1).sum(axis=1)
    expected = float(annualized_return(portfolio, periods_per_year=TRADING_DAYS_PER_YEAR, dropna=False))
    volatility = float(annualized_volatility(portfolio, periods_per_year=TRADING_DAYS_PER_YEAR, dropna=False))
    sharpe = float(sharpe_ratio(portfolio, risk_free_rate=risk_free_rate, dropna=False))
    covariance = frame.cov() * TRADING_DAYS_PER_YEAR
    # portfolio_volatility is the analytic counterpart of the realised volatility
    # above; keeping both is what lets the tests compare weights and risk.
    _ = portfolio_volatility(aligned, covariance)
    return expected, volatility, sharpe


def compare_allocations(
    current_weights: Mapping[str, float] | pd.Series,
    optimized: OptimizationResult,
    returns: pd.DataFrame,
    *,
    risk_free_rate: float = 0.0,
) -> pd.DataFrame:
    """Side-by-side comparison of the current and optimised allocations.

    Returns a frame indexed by asset with ``current``, ``optimized`` and
    ``delta`` weight columns, plus the metrics of both portfolios attached in
    ``DataFrame.attrs``.
    """
    current = current_weights if isinstance(current_weights, pd.Series) else pd.Series(dict(current_weights))
    current = current.astype("float64")
    symbols = sorted(set(current.index) | set(optimized.weights.index))
    frame = pd.DataFrame(
        {
            "current": current.reindex(symbols).fillna(0.0),
            "optimized": optimized.weights.reindex(symbols).fillna(0.0),
        }
    )
    frame["delta"] = frame["optimized"] - frame["current"]
    current_metrics = evaluate_weights(
        current_weights, returns, risk_free_rate=risk_free_rate
    )
    frame.attrs["metrics"] = {
        "current": {
            "expected_return": current_metrics[0],
            "volatility": current_metrics[1],
            "sharpe": current_metrics[2],
        },
        "optimized": {
            "expected_return": optimized.expected_return,
            "volatility": optimized.volatility,
            "sharpe": optimized.sharpe,
        },
    }
    return frame
