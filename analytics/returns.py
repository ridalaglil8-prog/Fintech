"""Return calculations.

Conventions and formulas
------------------------
* A **price** series is a strictly positive level indexed by date.
* **Simple (arithmetic) daily return**:
      r_t = P_t / P_{t-1} - 1
* **Log (geometric) daily return**:
      r_t = ln(P_t / P_{t-1})
* **Cumulative (compound) return** over the sample:
      R = prod_t (1 + r_t) - 1
* **Annualised return** (geometric, using ``periods_per_year`` periods, 252 by
  default for daily data):
      R_ann = (1 + R) ** (periods_per_year / n) - 1
  where ``n`` is the number of observed returns. The formula is *not*
  extrapolated when fewer than two observations are available: an
  :class:`InsufficientDataError` is raised instead.
* **Annualised volatility** (sample standard deviation, ``ddof=1``):
      sigma_ann = std(r_t) * sqrt(periods_per_year)
* **Portfolio return** with weights ``w`` (long-only, sum to 1):
      r_p,t = sum_i w_i * r_i,t

Missing data policy
-------------------
Prices are **never** forward-filled or interpolated to fabricate a return.
Callers decide explicitly how to treat missing observations:

* ``daily_returns`` keeps internal gaps as ``NaN`` (an explicit signal that a
  price is missing) and only drops the leading ``NaN`` that is structurally
  caused by differencing the first price.
* Every statistical function below drops ``NaN`` observations **loudly**:
  ``dropna=True`` (the default) removes them before computing, and
  ``dropna=False`` lets them propagate. Neither path invents a value.

All functions validate their inputs and raise
:class:`~analytics.exceptions.InvalidInputError` (non-numeric data, non-positive
prices, ``inf`` values) or :class:`~analytics.exceptions.InsufficientDataError`
(not enough observations) rather than returning a silent ``NaN``.
"""

from __future__ import annotations

from typing import Mapping

import numpy as np
import pandas as pd

from analytics.exceptions import InsufficientDataError, InvalidInputError
from config import MIN_OBSERVATIONS_FOR_METRICS, TRADING_DAYS_PER_YEAR

__all__ = [
    "PriceFrame",
    "validate_price_frame",
    "daily_returns",
    "cumulative_returns",
    "cumulative_wealth",
    "total_return",
    "annualized_return",
    "annualized_volatility",
    "wealth_index",
    "portfolio_returns",
    "align_returns_to_weights",
]

PriceFrame = pd.Series | pd.DataFrame


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------
def _require_numeric(obj: PriceFrame, name: str) -> None:
    """Raise if ``obj`` is not a numeric Series/DataFrame."""
    if isinstance(obj, (pd.Series, pd.DataFrame)):
        if not obj.empty and obj.dtypes.apply(
            lambda dtype: pd.api.types.is_numeric_dtype(dtype)
        ).all():
            return
    raise InvalidInputError(
        f"`{name}` must be a non-empty numeric Series or DataFrame.",
        code="non_numeric_input",
        params={"name": name},
    )


def _check_finite(obj: PriceFrame, name: str) -> None:
    """Raise on ``inf``/``-inf`` values (``NaN`` is allowed: it means 'missing')."""
    values = obj.to_numpy(dtype="float64", na_value=np.nan) if isinstance(obj, pd.DataFrame) else obj.to_numpy(dtype="float64")
    if np.isinf(values).any():
        raise InvalidInputError(
            f"`{name}` contains non-finite values (inf/-inf).",
            code="non_finite_input",
            params={"name": name},
        )


def _clean(returns: PriceFrame, *, dropna: bool = True) -> PriceFrame:
    """Explicitly handle missing observations (never fills values)."""
    if not dropna:
        return returns
    if isinstance(returns, pd.DataFrame):
        return returns.dropna(how="any")
    return returns.dropna()


def _require_observations(n: int, *, minimum: int, what: str) -> None:
    if n < minimum:
        raise InsufficientDataError(
            f"Not enough observations to compute {what}: {n} available, "
            f"{minimum} required.",
            code="insufficient_observations",
            params={"available": int(n), "required": int(minimum), "what": what},
        )


# ---------------------------------------------------------------------------
# Prices -> returns
# ---------------------------------------------------------------------------
def validate_price_frame(prices: PriceFrame, *, name: str = "prices") -> None:
    """Validate a price series/frame.

    Rules
    -----
    * numeric dtype, non-empty;
    * strictly positive prices (zero or negative levels are invalid and would
      make returns meaningless);
    * no ``inf`` values (``NaN`` = missing observation, allowed).
    """
    if prices is None:
        raise InvalidInputError(f"`{name}` is missing.", code="missing_input", params={"name": name})
    _require_numeric(prices, name)
    _check_finite(prices, name)
    values = prices.to_numpy(dtype="float64", na_value=np.nan)
    if not np.isnan(values).all() and np.nanmin(values) <= 0:
        raise InvalidInputError(
            f"`{name}` contains zero or negative prices.",
            code="non_positive_prices",
            params={"name": name},
        )


def daily_returns(
    prices: PriceFrame,
    *,
    method: str = "simple",
    drop_leading_na: bool = True,
) -> PriceFrame:
    """Compute daily returns from a price series/frame.

    Parameters
    ----------
    prices:
        Positive price levels indexed by date.
    method:
        ``"simple"`` (default) or ``"log"``.
    drop_leading_na:
        Remove the structural first ``NaN`` created by differencing. Internal
        gaps are preserved as ``NaN`` so callers can align them explicitly.
    """
    if method not in {"simple", "log"}:
        raise InvalidInputError(
            f"Unknown return method '{method}'.",
            code="unknown_return_method",
            params={"method": method},
        )
    validate_price_frame(prices)
    if method == "simple":
        returns = prices.pct_change(fill_method=None)
    else:  # log returns require positive prices, checked above
        returns = np.log(prices / prices.shift(1))
    if drop_leading_na:
        if isinstance(returns, pd.DataFrame):
            returns = returns.iloc[1:] if len(returns) else returns
        else:
            returns = returns.iloc[1:] if len(returns) else returns
    # Drop rows that are *entirely* missing: they carry no information at all
    # and cannot distort any statistic.
    if isinstance(returns, pd.DataFrame) and not returns.empty:
        returns = returns.dropna(how="all")
    return returns


# ---------------------------------------------------------------------------
# Aggregation helpers (float for Series, Series for DataFrame)
# ---------------------------------------------------------------------------
def _reduce(
    returns: PriceFrame,
    fn,
    *,
    dropna: bool = True,
    name: str = "returns",
    validate: bool = True,
):
    if validate:
        _require_numeric(returns, name)
        _check_finite(returns, name)
    cleaned = _clean(returns, dropna=dropna)
    if isinstance(cleaned, pd.DataFrame):
        if cleaned.shape[1] == 0:
            raise InvalidInputError("No columns to aggregate.", code="empty_input", params={"name": name})
        return cleaned.apply(lambda col: fn(col.dropna()) if dropna is False else fn(col))
    return fn(cleaned)


def cumulative_returns(returns: PriceFrame, *, dropna: bool = True) -> PriceFrame:
    """Compound return path: ``prod(1 + r) - 1`` cumulated over time."""
    return _reduce(returns, lambda s: (1.0 + s).cumprod() - 1.0, dropna=dropna)


def cumulative_wealth(returns: PriceFrame, *, dropna: bool = True) -> PriceFrame:
    """Wealth index normalised to 1.0 at the first observation."""
    return _reduce(returns, lambda s: (1.0 + s).cumprod(), dropna=dropna)


def wealth_index(
    returns: PriceFrame, *, initial_value: float = 1.0, dropna: bool = True
) -> PriceFrame:
    """Wealth index starting at ``initial_value`` (e.g. the initial capital)."""
    if not np.isfinite(initial_value):
        raise InvalidInputError("initial_value must be finite.", code="non_finite_input", params={"name": "initial_value"})
    wealth = cumulative_wealth(returns, dropna=dropna)
    return wealth * float(initial_value)


def total_return(returns: PriceFrame, *, dropna: bool = True) -> float | pd.Series:
    """Total compound return over the whole sample."""

    def _total(series: pd.Series) -> float:
        values = series.to_numpy(dtype="float64")
        if values.size == 0:
            raise InsufficientDataError(
                "No observations available.",
                code="insufficient_observations",
                params={"available": 0, "required": MIN_OBSERVATIONS_FOR_METRICS, "what": "total return"},
            )
        return float(np.prod(1.0 + values) - 1.0)

    return _reduce(returns, _total, dropna=dropna)


def annualized_return(
    returns: PriceFrame,
    *,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
    dropna: bool = True,
) -> float | pd.Series:
    """Geometric annualised return ``(1 + R_total) ** (ppy / n) - 1``.

    ``n`` is the number of observed returns. At least
    :data:`config.MIN_OBSERVATIONS_FOR_METRICS` observations are required.
    """
    if periods_per_year <= 0:
        raise InvalidInputError(
            "periods_per_year must be positive.",
            code="invalid_periods_per_year",
            params={"value": periods_per_year},
        )

    def _annualise(series: pd.Series) -> float:
        values = series.to_numpy(dtype="float64")
        _require_observations(values.size, minimum=MIN_OBSERVATIONS_FOR_METRICS, what="annualised return")
        growth = float(np.prod(1.0 + values))
        if growth <= 0:
            # Only reachable with a <= -100% return: capital fully lost.
            return -1.0
        return float(growth ** (periods_per_year / values.size) - 1.0)

    return _reduce(returns, _annualise, dropna=dropna)


def annualized_volatility(
    returns: PriceFrame,
    *,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
    ddof: int = 1,
    dropna: bool = True,
) -> float | pd.Series:
    """Annualised sample standard deviation: ``std(r) * sqrt(ppy)``.

    Returns ``0.0`` for a constant return series (zero volatility is a valid,
    if degenerate, result) — callers such as the Sharpe ratio must handle it.
    """
    if periods_per_year <= 0:
        raise InvalidInputError(
            "periods_per_year must be positive.",
            code="invalid_periods_per_year",
            params={"value": periods_per_year},
        )

    def _vol(series: pd.Series) -> float:
        values = series.to_numpy(dtype="float64")
        if values.size < 2:
            raise InsufficientDataError(
                "At least 2 observations are required to compute volatility.",
                code="insufficient_observations",
                params={
                    "available": int(values.size),
                    "required": 2,
                    "what": "annualised volatility",
                },
            )
        return float(np.std(values, ddof=ddof) * np.sqrt(periods_per_year))

    return _reduce(returns, _vol, dropna=dropna)


# ---------------------------------------------------------------------------
# Portfolio level
# ---------------------------------------------------------------------------
def align_returns_to_weights(
    asset_returns: pd.DataFrame,
    weights: Mapping[str, float] | pd.Series,
    *,
    drop_partial: bool = True,
) -> tuple[pd.DataFrame, pd.Series]:
    """Align asset returns with portfolio weights.

    * only assets present in both the return frame and the weights are kept;
    * zero-weight assets are dropped (they cannot influence the result);
    * when ``drop_partial`` is True, rows where *any* retained asset is missing
      are dropped (a portfolio return is only meaningful when every component
      is observed). Nothing is interpolated.
    """
    if asset_returns is None or asset_returns.empty:
        raise InsufficientDataError(
            "No asset return history available.",
            code="insufficient_observations",
            params={"available": 0, "required": MIN_OBSERVATIONS_FOR_METRICS, "what": "portfolio returns"},
        )
    _require_numeric(asset_returns, "asset_returns")
    _check_finite(asset_returns, "asset_returns")

    weight_series = (
        weights.copy() if isinstance(weights, pd.Series) else pd.Series(dict(weights), dtype="float64")
    )
    if isinstance(weight_series, pd.Series) and weight_series.empty:
        raise InvalidInputError("No portfolio weights provided.", code="empty_weights")

    weight_series = pd.to_numeric(weight_series, errors="coerce")
    if weight_series.isna().any():
        raise InvalidInputError(
            "Weights contain non-numeric values.",
            code="non_numeric_weights",
        )
    weight_series = weight_series[weight_series > 0]

    common = [symbol for symbol in weight_series.index if symbol in asset_returns.columns]
    if not common:
        raise InvalidInputError(
            "None of the portfolio assets have return history.",
            code="no_common_assets",
            params={"assets": list(weight_series.index)},
        )

    frame = asset_returns.loc[:, common]
    weights_aligned = weight_series.loc[common]
    if drop_partial:
        frame = frame.dropna(how="any")
    if frame.empty:
        raise InsufficientDataError(
            "No overlapping observations across the selected assets.",
            code="no_overlapping_observations",
            params={"available": 0, "required": MIN_OBSERVATIONS_FOR_METRICS, "what": "portfolio returns"},
        )

    total = float(weights_aligned.sum())
    if not np.isfinite(total) or total <= 0:
        raise InvalidInputError(
            "The sum of the retained weights must be strictly positive.",
            code="non_positive_weight_sum",
            params={"total": float(total)},
        )
    return frame, weights_aligned / total


def portfolio_returns(
    asset_returns: pd.DataFrame,
    weights: Mapping[str, float] | pd.Series,
    *,
    name: str = "portfolio",
    drop_partial: bool = True,
) -> pd.Series:
    """Weighted daily portfolio returns: ``r_p = sum_i w_i * r_i``.

    The retained weights are renormalised to sum to 1 so that the result stays
    interpretable when an asset is missing from the price history (the
    renormalisation is reported by :func:`align_returns_to_weights`).
    """
    frame, aligned = align_returns_to_weights(asset_returns, weights, drop_partial=drop_partial)
    result = frame.mul(aligned, axis=1).sum(axis=1)
    result.name = name
    return result
