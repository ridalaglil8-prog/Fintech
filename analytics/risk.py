"""Risk metrics.

Sign conventions
----------------
* Returns are decimals (``0.01`` = 1%).
* Drawdowns are **non-positive** (``-0.20`` = a 20% peak-to-trough loss).
  Use :func:`drawdown_magnitude` when a positive number is needed for display.
* Value at Risk / Expected Shortfall are returned as **positive loss
  magnitudes** (``0.023`` = "a 2.3% loss on the worst days").

Formulas (daily returns, 252 periods/year by default)
-----------------------------------------------------
* Portfolio variance: ``sigma_p^2 = w' * Sigma * w`` (``Sigma`` = covariance of
  asset returns, annualised outside this function).
* Annualised volatility: ``sigma_p * sqrt(252)``.
* Sharpe ratio: ``(R_ann - rf) / sigma_ann`` where ``R_ann`` is the geometric
  annualised return and ``rf`` the annualised risk-free rate — both annualised
  consistently. If ``sigma_ann == 0`` the ratio is undefined and ``NaN`` is
  returned (never an arbitrary ``inf``).
* Maximum drawdown: from the wealth index ``W_t`` and its running maximum
  ``M_t = max_{s<=t} W_s``, ``DD_t = W_t / M_t - 1`` and
  ``MDD = min_t DD_t`` (<= 0).
* Historical VaR at level ``c``: ``VaR_c = max(0, -quantile(r, 1 - c))``.
* Expected Shortfall: mean loss of the observations beyond the VaR threshold.
* HHI concentration: ``sum_i w_i^2`` (1 is a single-asset portfolio), plus the
  effective number of assets ``1 / HHI`` and a normalised HHI in ``[0, 1]``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

import numpy as np
import pandas as pd

from analytics.exceptions import (
    InsufficientDataError,
    InvalidInputError,
    InvalidWeightsError,
)
from analytics.returns import (
    PriceFrame,
    _check_finite,
    _clean,
    _reduce,
    _require_numeric,
    _require_observations,
    annualized_return,
    annualized_volatility,
    cumulative_wealth,
)
from config import MIN_OBSERVATIONS_FOR_METRICS, TRADING_DAYS_PER_YEAR

__all__ = [
    "DrawdownStats",
    "ConcentrationMetrics",
    "drawdown_series",
    "max_drawdown",
    "max_drawdown_details",
    "drawdown_magnitude",
    "portfolio_variance",
    "portfolio_volatility",
    "sharpe_ratio",
    "sortino_ratio",
    "downside_deviation",
    "value_at_risk",
    "conditional_value_at_risk",
    "concentration_metrics",
    "risk_contributions",
    "beta_alpha",
    "tracking_error",
    "information_ratio",
]


# ---------------------------------------------------------------------------
# Drawdowns
# ---------------------------------------------------------------------------
def drawdown_series(returns: PriceFrame, *, dropna: bool = True) -> PriceFrame:
    """``DD_t = W_t / M_t - 1`` (values <= 0), with ``M`` the running maximum."""
    wealth = cumulative_wealth(returns, dropna=dropna)
    if isinstance(wealth, pd.DataFrame):
        running_max = wealth.cummax()
        return wealth / running_max - 1.0
    running_max = wealth.cummax()
    return wealth / running_max - 1.0


def max_drawdown(returns: PriceFrame, *, dropna: bool = True) -> float | pd.Series:
    """Largest peak-to-trough decline of the wealth index (<= 0)."""

    def _mdd(series: pd.Series) -> float:
        values = series.to_numpy(dtype="float64")
        _require_observations(values.size, minimum=MIN_OBSERVATIONS_FOR_METRICS, what="maximum drawdown")
        wealth = np.cumprod(1.0 + values)
        running_max = np.maximum.accumulate(wealth)
        return float(np.min(wealth / running_max - 1.0))

    return _reduce(returns, _mdd, dropna=dropna)


def drawdown_magnitude(returns: PriceFrame, *, dropna: bool = True) -> float | pd.Series:
    """``abs(max_drawdown)`` — convenient for display ("12.4% drawdown")."""
    result = max_drawdown(returns, dropna=dropna)
    if isinstance(result, pd.Series):
        return result.abs()
    return abs(result)


@dataclass(frozen=True)
class DrawdownStats:
    """Detailed maximum-drawdown information."""

    max_drawdown: float  #: <= 0
    peak_date: pd.Timestamp | None
    trough_date: pd.Timestamp | None
    recovery_date: pd.Timestamp | None
    drawdown_duration_days: int | None  #: peak -> trough (calendar days)
    recovery_days: int | None  #: trough -> recovery (calendar days)

    @property
    def recovered(self) -> bool:
        return self.recovery_date is not None


def max_drawdown_details(returns: pd.Series, *, dropna: bool = True) -> DrawdownStats:
    """Maximum drawdown with peak / trough / recovery dates."""
    if isinstance(returns, pd.DataFrame):
        raise InvalidInputError(
            "max_drawdown_details expects a single return series.",
            code="expected_series",
        )
    series = _clean(returns, dropna=dropna)
    _require_numeric(series, "returns")
    _check_finite(series, "returns")
    values = series.to_numpy(dtype="float64")
    _require_observations(values.size, minimum=MIN_OBSERVATIONS_FOR_METRICS, what="maximum drawdown")

    wealth = np.cumprod(1.0 + values)
    running_max = np.maximum.accumulate(wealth)
    drawdown = wealth / running_max - 1.0
    trough_idx = int(np.argmin(drawdown))
    if trough_idx == 0:
        return DrawdownStats(0.0, None, None, None, None, None)

    peak_idx = int(np.argmax(wealth[: trough_idx + 1]))
    peak_level = wealth[peak_idx]
    recovery_idx: int | None = None
    for idx in range(trough_idx + 1, len(wealth)):
        if wealth[idx] >= peak_level:
            recovery_idx = idx
            break

    index = series.index
    peak_date = index[peak_idx]
    trough_date = index[trough_idx]
    recovery_date = index[recovery_idx] if recovery_idx is not None else None

    def _days(start, end) -> int | None:
        if start is None or end is None:
            return None
        try:
            return int((pd.Timestamp(end) - pd.Timestamp(start)).days)
        except Exception:  # pragma: no cover - non datetime index
            return None

    return DrawdownStats(
        max_drawdown=float(drawdown[trough_idx]),
        peak_date=peak_date,
        trough_date=trough_date,
        recovery_date=recovery_date,
        drawdown_duration_days=_days(peak_date, trough_date),
        recovery_days=_days(trough_date, recovery_date),
    )


# ---------------------------------------------------------------------------
# Variance / volatility at portfolio level
# ---------------------------------------------------------------------------
def _aligned_weight_vector(
    weights: Mapping[str, float] | pd.Series, labels: pd.Index
) -> np.ndarray:
    series = weights if isinstance(weights, pd.Series) else pd.Series(dict(weights), dtype="float64")
    if series.empty:
        raise InvalidWeightsError("No weights provided.", code="empty_weights")
    missing = [symbol for symbol in series.index if symbol not in labels]
    if missing:
        raise InvalidWeightsError(
            f"Assets missing from the covariance matrix: {missing}",
            code="missing_covariance_assets",
            params={"assets": missing},
        )
    vector = series.loc[labels].to_numpy(dtype="float64")
    if not np.isfinite(vector).all():
        raise InvalidWeightsError("Weights contain non-finite values.", code="non_finite_weights")
    return vector


def portfolio_variance(
    weights: Mapping[str, float] | pd.Series, covariance: pd.DataFrame
) -> float:
    """``sigma_p^2 = w' Sigma w`` in the *same units as the covariance input*.

    Pass a daily covariance matrix to obtain a daily variance, or an
    annualised one for an annual variance.
    """
    if covariance is None or covariance.empty:
        raise InvalidInputError("Covariance matrix is empty.", code="empty_covariance")
    labels = covariance.columns
    if list(labels) != list(covariance.index):
        raise InvalidInputError(
            "Covariance matrix must be square with identical labels.",
            code="invalid_covariance_shape",
        )
    vector = _aligned_weight_vector(weights, labels)
    matrix = covariance.to_numpy(dtype="float64")
    if not np.isfinite(matrix).all():
        raise InvalidInputError("Covariance matrix contains non-finite values.", code="non_finite_covariance")
    variance = float(vector @ matrix @ vector)
    if not np.isfinite(variance):
        raise InvalidInputError("Computed variance is not finite.", code="non_finite_result")
    return max(variance, 0.0)  # numerical noise can produce tiny negatives


def portfolio_volatility(
    weights: Mapping[str, float] | pd.Series,
    covariance: pd.DataFrame,
    *,
    periods_per_year: int | None = None,
) -> float:
    """``sqrt(w' Sigma w)``, optionally annualised by ``sqrt(periods_per_year)``."""
    variance = portfolio_variance(weights, covariance)
    volatility = float(np.sqrt(variance))
    if periods_per_year is not None:
        if periods_per_year <= 0:
            raise InvalidInputError(
                "periods_per_year must be positive.",
                code="invalid_periods_per_year",
                params={"value": periods_per_year},
            )
        volatility *= float(np.sqrt(periods_per_year))
    return volatility


# ---------------------------------------------------------------------------
# Risk-adjusted performance
# ---------------------------------------------------------------------------
def sharpe_ratio(
    returns: PriceFrame,
    *,
    risk_free_rate: float = 0.0,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
    dropna: bool = True,
) -> float | pd.Series:
    """``(R_ann - rf) / sigma_ann`` with consistent annualised inputs.

    Returns ``NaN`` when the annualised volatility is zero (undefined ratio)
    and raises :class:`InsufficientDataError` when there is not enough history.
    """
    if not np.isfinite(risk_free_rate):
        raise InvalidInputError(
            "risk_free_rate must be finite.", code="non_finite_input", params={"name": "risk_free_rate"}
        )

    def _sharpe(series: pd.Series) -> float:
        volatility = annualized_volatility(series, periods_per_year=periods_per_year, dropna=False)
        if not np.isfinite(volatility) or volatility <= 0:
            return float("nan")
        growth = annualized_return(series, periods_per_year=periods_per_year, dropna=False)
        return float((growth - risk_free_rate) / volatility)

    return _reduce(returns, _sharpe, dropna=dropna)


def downside_deviation(
    returns: PriceFrame,
    *,
    minimum_acceptable_return: float = 0.0,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
    ddof: int = 1,
    dropna: bool = True,
) -> float | pd.Series:
    """Annualised deviation of returns below the MAR (daily MAR by default)."""
    daily_mar = minimum_acceptable_return / periods_per_year

    def _dd(series: pd.Series) -> float:
        values = series.to_numpy(dtype="float64")
        if values.size < 2:
            raise InsufficientDataError(
                "At least 2 observations are required to compute downside deviation.",
                code="insufficient_observations",
                params={"available": int(values.size), "required": 2, "what": "downside deviation"},
            )
        shortfall = np.minimum(values - daily_mar, 0.0)
        return float(np.sqrt(np.mean(shortfall**2)) * np.sqrt(periods_per_year))

    return _reduce(returns, _dd, dropna=dropna)


def sortino_ratio(
    returns: PriceFrame,
    *,
    risk_free_rate: float = 0.0,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
    dropna: bool = True,
) -> float | pd.Series:
    """``(R_ann - rf) / downside_deviation``; ``NaN`` if downside deviation is 0."""

    def _sortino(series: pd.Series) -> float:
        dd = downside_deviation(
            series,
            minimum_acceptable_return=risk_free_rate,
            periods_per_year=periods_per_year,
            dropna=False,
        )
        if not np.isfinite(dd) or dd <= 0:
            return float("nan")
        growth = annualized_return(series, periods_per_year=periods_per_year, dropna=False)
        return float((growth - risk_free_rate) / dd)

    return _reduce(returns, _sortino, dropna=dropna)


# ---------------------------------------------------------------------------
# Tail risk (historical, non-parametric)
# ---------------------------------------------------------------------------
def value_at_risk(
    returns: PriceFrame,
    *,
    level: float = 0.95,
    dropna: bool = True,
) -> float | pd.Series:
    """Historical VaR as a positive loss magnitude at confidence ``level``."""
    if not 0.5 < level < 1.0:
        raise InvalidInputError(
            "VaR level must be strictly between 0.5 and 1.0.",
            code="invalid_confidence_level",
            params={"value": level},
        )

    def _var(series: pd.Series) -> float:
        values = series.to_numpy(dtype="float64")
        if values.size < MIN_OBSERVATIONS_FOR_METRICS:
            raise InsufficientDataError(
                "Not enough observations for Value at Risk.",
                code="insufficient_observations",
                params={"available": int(values.size), "required": MIN_OBSERVATIONS_FOR_METRICS, "what": "VaR"},
            )
        return float(max(0.0, -np.quantile(values, 1.0 - level)))

    return _reduce(returns, _var, dropna=dropna)


def conditional_value_at_risk(
    returns: PriceFrame,
    *,
    level: float = 0.95,
    dropna: bool = True,
) -> float | pd.Series:
    """Historical Expected Shortfall (mean loss beyond the VaR threshold)."""
    if not 0.5 < level < 1.0:
        raise InvalidInputError(
            "CVaR level must be strictly between 0.5 and 1.0.",
            code="invalid_confidence_level",
            params={"value": level},
        )

    def _cvar(series: pd.Series) -> float:
        values = series.to_numpy(dtype="float64")
        if values.size < MIN_OBSERVATIONS_FOR_METRICS:
            raise InsufficientDataError(
                "Not enough observations for Expected Shortfall.",
                code="insufficient_observations",
                params={"available": int(values.size), "required": MIN_OBSERVATIONS_FOR_METRICS, "what": "CVaR"},
            )
        threshold = float(np.quantile(values, 1.0 - level))
        tail = values[values <= threshold]
        if tail.size == 0:
            return float(max(0.0, -threshold))
        return float(max(0.0, -float(np.mean(tail))))

    return _reduce(returns, _cvar, dropna=dropna)


# ---------------------------------------------------------------------------
# Concentration / diversification
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ConcentrationMetrics:
    """Portfolio concentration indicators (all based on the weights only)."""

    hhi: float  #: sum(w^2); 1.0 = single asset
    normalized_hhi: float  #: (HHI - 1/N) / (1 - 1/N), in [0, 1]
    effective_n: float  #: 1 / HHI
    top1_weight: float
    top3_weight: float
    n_assets: int
    label: str  #: "diversified" | "moderate" | "concentrated"
    warnings: list[str] = field(default_factory=list)


def concentration_metrics(
    weights: Mapping[str, float] | pd.Series,
    *,
    diversified_threshold: float = 0.15,
    concentrated_threshold: float = 0.25,
) -> ConcentrationMetrics:
    """HHI, normalised HHI, effective number of assets and top-weight metrics.

    Thresholds follow the common interpretation of the Herfindahl index applied
    to portfolio weights.
    """
    series = weights if isinstance(weights, pd.Series) else pd.Series(dict(weights), dtype="float64")
    if series.empty:
        raise InvalidWeightsError("No weights provided.", code="empty_weights")
    values = series.to_numpy(dtype="float64")
    if not np.isfinite(values).all():
        raise InvalidWeightsError("Weights contain non-finite values.", code="non_finite_weights")
    values = np.clip(values, 0.0, None)
    total = float(values.sum())
    if total <= 0:
        raise InvalidWeightsError("Weights must sum to a positive value.", code="non_positive_weight_sum")
    normalised = values / total

    hhi = float(np.sum(normalised**2))
    n_assets = int((normalised > 0).sum())
    normalized_hhi = 0.0 if n_assets <= 1 else (hhi - 1.0 / n_assets) / (1.0 - 1.0 / n_assets)
    effective_n = float(1.0 / hhi) if hhi > 0 else 0.0
    ordered = np.sort(normalised)[::-1]

    if hhi <= diversified_threshold:
        label = "diversified"
    elif hhi <= concentrated_threshold:
        label = "moderate"
    else:
        label = "concentrated"

    warnings: list[str] = []
    if hhi > concentrated_threshold:
        warnings.append("concentration.high_hhi")
    if effective_n < 3 and n_assets >= 3:
        warnings.append("concentration.low_effective_n")
    if ordered.size and ordered[0] > 0.5:
        warnings.append("concentration.dominant_position")

    return ConcentrationMetrics(
        hhi=hhi,
        normalized_hhi=float(normalized_hhi),
        effective_n=effective_n,
        top1_weight=float(ordered[0]) if ordered.size else 0.0,
        top3_weight=float(ordered[:3].sum()) if ordered.size else 0.0,
        n_assets=n_assets,
        label=label,
        warnings=warnings,
    )


# ---------------------------------------------------------------------------
# Risk contribution
# ---------------------------------------------------------------------------
def risk_contributions(
    weights: Mapping[str, float] | pd.Series,
    covariance: pd.DataFrame,
    *,
    relative: bool = True,
) -> pd.Series:
    """Marginal contribution of each asset to portfolio volatility.

    ``RC_i = w_i * (Sigma w)_i / sigma_p`` is the *absolute* contribution to
    volatility (the values sum to ``sigma_p``). With ``relative=True`` the
    result is divided by ``sigma_p`` and therefore sums to 1 (100%).
    """
    labels = covariance.columns
    vector = _aligned_weight_vector(weights, labels)
    matrix = covariance.to_numpy(dtype="float64")
    if not np.isfinite(matrix).all():
        raise InvalidInputError("Covariance matrix contains non-finite values.", code="non_finite_covariance")
    variance = float(vector @ matrix @ vector)
    if not np.isfinite(variance) or variance <= 0:
        raise InvalidInputError(
            "Portfolio variance is zero: risk contributions are undefined.",
            code="non_finite_result",
        )
    sigma_p = float(np.sqrt(variance))
    absolute = vector * (matrix @ vector) / sigma_p
    values = absolute / sigma_p if relative else absolute
    return pd.Series(values, index=list(labels), name="risk_contribution")


# ---------------------------------------------------------------------------
# Benchmark-relative metrics
# ---------------------------------------------------------------------------
def beta_alpha(
    returns: pd.Series,
    benchmark_returns: pd.Series,
    *,
    risk_free_rate: float = 0.0,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
) -> tuple[float, float]:
    """CAPM ``beta`` (systematic exposure) and ``alpha`` (annualised excess return).

    ``beta = cov(r, r_b) / var(r_b)`` computed on aligned daily observations.
    ``alpha = R_ann - (rf + beta * (R_b,ann - rf))``.
    """
    aligned = pd.concat([returns.rename("p"), benchmark_returns.rename("b")], axis=1).dropna()
    if len(aligned) < MIN_OBSERVATIONS_FOR_METRICS:
        raise InsufficientDataError(
            "Not enough overlapping observations for beta/alpha.",
            code="insufficient_observations",
            params={
                "available": int(len(aligned)),
                "required": MIN_OBSERVATIONS_FOR_METRICS,
                "what": "beta/alpha",
            },
        )
    variance = float(aligned["b"].var(ddof=1))
    if not np.isfinite(variance) or variance <= 0:
        return float("nan"), float("nan")
    beta = float(aligned["p"].cov(aligned["b"]) / variance)
    r_ann = annualized_return(aligned["p"], periods_per_year=periods_per_year, dropna=False)
    b_ann = annualized_return(aligned["b"], periods_per_year=periods_per_year, dropna=False)
    alpha = float(r_ann - (risk_free_rate + beta * (b_ann - risk_free_rate)))
    return beta, alpha


def tracking_error(
    returns: pd.Series,
    benchmark_returns: pd.Series,
    *,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
) -> float:
    """Annualised standard deviation of the active return ``r - r_b``."""
    aligned = pd.concat([returns.rename("p"), benchmark_returns.rename("b")], axis=1).dropna()
    if len(aligned) < 2:
        raise InsufficientDataError(
            "At least 2 overlapping observations are required for tracking error.",
            code="insufficient_observations",
            params={"available": int(len(aligned)), "required": 2, "what": "tracking error"},
        )
    active = aligned["p"] - aligned["b"]
    return float(active.std(ddof=1) * np.sqrt(periods_per_year))


def information_ratio(
    returns: pd.Series,
    benchmark_returns: pd.Series,
    *,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
) -> float:
    """``active annualised return / tracking error``; ``NaN`` when TE is 0."""
    aligned = pd.concat([returns.rename("p"), benchmark_returns.rename("b")], axis=1).dropna()
    if len(aligned) < MIN_OBSERVATIONS_FOR_METRICS:
        raise InsufficientDataError(
            "Not enough overlapping observations for the information ratio.",
            code="insufficient_observations",
            params={
                "available": int(len(aligned)),
                "required": MIN_OBSERVATIONS_FOR_METRICS,
                "what": "information ratio",
            },
        )
    te = tracking_error(aligned["p"], aligned["b"], periods_per_year=periods_per_year)
    if not np.isfinite(te) or te <= 0:
        return float("nan")
    p_ann = annualized_return(aligned["p"], periods_per_year=periods_per_year, dropna=False)
    b_ann = annualized_return(aligned["b"], periods_per_year=periods_per_year, dropna=False)
    return float((p_ann - b_ann) / te)
