"""Portfolio-level calculations: weight validation, alignment, valuation.

Design principles
-----------------
* **Nothing is fabricated.** Missing prices are never interpolated or
  forward-filled. Alignment is explicit, quantified and reported back to the
  caller through :class:`AlignmentReport` so the interface can explain exactly
  how many observations were used and dropped.
* **Currency-agnostic.** Everything here works on *levels* and *returns*; a
  timeseries is expressed in the currency of its inputs. Mixing currencies is a
  presentation/business decision handled by ``services.currency``.
* **Weights are fractions** (``0.25`` = 25%), long-only in this MVP.

Formulas
--------
* Alignment: assets are intersected on their (date, symbol) coverage; rows with
  any missing retained asset are dropped when ``drop_partial=True``.
* Portfolio return: ``r_p,t = sum_i w_i,renorm * r_i,t``.
* Wealth: ``W_t = V0 * cumprod(1 + r_p,t)``, with ``V0`` the initial capital.
* Allocation value: ``V_i = w_i * V0``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from analytics.exceptions import (
    InsufficientDataError,
    InvalidInputError,
    InvalidWeightsError,
)
from analytics.returns import (
    _check_finite,
    _require_numeric,
    annualized_return,
    annualized_volatility,
    daily_returns,
    portfolio_returns,
    total_return,
)
from analytics.risk import (
    beta_alpha,
    conditional_value_at_risk,
    information_ratio,
    max_drawdown,
    sharpe_ratio,
    sortino_ratio,
    tracking_error,
    value_at_risk,
)
from config import (
    MIN_OBSERVATIONS_FOR_METRICS,
    RECOMMENDED_MIN_OBSERVATIONS,
    TRADING_DAYS_PER_YEAR,
    WEIGHT_SUM_TOLERANCE,
)

__all__ = [
    "WeightIssue",
    "WeightValidation",
    "validate_weights",
    "normalize_weights",
    "allocation_values",
    "AlignmentReport",
    "align_price_frame",
    "split_currency_totals",
    "PortfolioTimeseries",
    "build_portfolio_timeseries",
    "returns_summary",
    "benchmark_comparison",
]


# ---------------------------------------------------------------------------
# Weights
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class WeightIssue:
    """A validation problem, expressed as a code + parameters for i18n."""

    code: str
    params: dict = field(default_factory=dict)

    def __str__(self) -> str:  # pragma: no cover - debugging helper
        return f"{self.code}({self.params})"


@dataclass(frozen=True)
class WeightValidation:
    """Result of :func:`validate_weights`."""

    is_valid: bool
    total: float
    errors: list[WeightIssue]
    warnings: list[WeightIssue]
    n_assets: int

    @property
    def error_codes(self) -> list[str]:
        return [issue.code for issue in self.errors]

    @property
    def warning_codes(self) -> list[str]:
        return [issue.code for issue in self.warnings]


def _as_weight_series(weights: Mapping[str, float] | pd.Series) -> pd.Series:
    if weights is None:
        raise InvalidWeightsError("Weights are missing.", code="empty_weights")
    if isinstance(weights, pd.Series):
        series = weights.astype("float64")
    else:
        try:
            series = pd.Series(dict(weights), dtype="float64")
        except (TypeError, ValueError) as exc:
            raise InvalidWeightsError(
                "Weights cannot be interpreted as a numeric mapping.",
                code="non_numeric_weights",
            ) from exc
    series = series[series.index.notna()]
    return series


def validate_weights(
    weights: Mapping[str, float] | pd.Series,
    *,
    tolerance: float = WEIGHT_SUM_TOLERANCE,
    require_full_sum: bool = True,
    allow_zero: bool = True,
    max_weight: float = 1.0,
) -> WeightValidation:
    """Validate long-only portfolio weights.

    Checks performed
    ----------------
    * the mapping is not empty and has no duplicate/blank symbols;
    * every weight is finite and numeric;
    * ``0 <= w_i <= max_weight`` (no shorting in the MVP);
    * ``abs(sum(w) - 1) <= tolerance`` when ``require_full_sum`` is True.

    Returns
    -------
    :class:`WeightValidation` — never raises, so the UI can display every
    problem at once instead of stopping at the first one.
    """
    errors: list[WeightIssue] = []
    warnings: list[WeightIssue] = []

    try:
        series = _as_weight_series(weights)
    except InvalidWeightsError as exc:
        return WeightValidation(False, 0.0, [WeightIssue(exc.code)], [], 0)

    if series.empty:
        return WeightValidation(False, 0.0, [WeightIssue("empty_weights")], [], 0)

    if series.index.duplicated().any():
        errors.append(WeightIssue("duplicate_symbols"))

    numeric = pd.to_numeric(series, errors="coerce")
    non_numeric = numeric.isna() & series.notna()
    if non_numeric.any():
        errors.append(
            WeightIssue("non_numeric_weights", {"symbols": list(numeric.index[non_numeric])})
        )
    if numeric.isna().any():
        errors.append(
            WeightIssue("missing_weights", {"symbols": list(numeric.index[numeric.isna()])})
        )
        numeric = numeric.fillna(0.0)

    values = numeric.to_numpy(dtype="float64")
    if not np.isfinite(values).all():
        errors.append(WeightIssue("non_finite_weights"))
        values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)

    negative = values < -tolerance
    if negative.any():
        errors.append(
            WeightIssue(
                "negative_weights",
                {
                    "symbols": list(numeric.index[negative]),
                    "min": float(values[negative].min()),
                },
            )
        )
    above_max = values > max_weight + tolerance
    if above_max.any():
        errors.append(
            WeightIssue(
                "weights_above_max",
                {"symbols": list(numeric.index[above_max]), "max_weight": max_weight},
            )
        )

    clipped = np.clip(values, 0.0, None)
    total = float(clipped.sum())
    if require_full_sum and abs(total - 1.0) > tolerance:
        errors.append(WeightIssue("weights_sum", {"total": total, "tolerance": tolerance}))
    if require_full_sum and total == 0:
        errors.append(WeightIssue("zero_total_weight"))

    if not allow_zero and (clipped == 0).any():
        errors.append(WeightIssue("zero_weights", {"symbols": list(numeric.index[clipped == 0])}))

    n_assets = int((clipped > 0).sum())
    if n_assets == 1:
        warnings.append(WeightIssue("single_asset_portfolio"))
    if total > 0 and n_assets > 1 and float(np.max(clipped)) > 0.5:
        warnings.append(
            WeightIssue("dominant_position", {"weight": float(np.max(clipped))})
        )

    return WeightValidation(
        is_valid=not errors,
        total=total,
        errors=errors,
        warnings=warnings,
        n_assets=n_assets,
    )


def normalize_weights(
    weights: Mapping[str, float] | pd.Series, *, drop_zero: bool = True
) -> pd.Series:
    """Renormalise weights so they sum to exactly 1 (>= 0, finite).

    Raises :class:`InvalidWeightsError` when the input contains negative,
    non-finite or zero-sum weights: normalising them would hide a data error.
    """
    series = _as_weight_series(weights)
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.isna().any():
        raise InvalidWeightsError(
            "Weights contain non-numeric or missing values.",
            code="non_numeric_weights",
        )
    values = numeric.to_numpy(dtype="float64")
    if not np.isfinite(values).all():
        raise InvalidWeightsError("Weights contain non-finite values.", code="non_finite_weights")
    if (values < 0).any():
        raise InvalidWeightsError(
            "Long-only portfolios cannot contain negative weights.",
            code="negative_weights",
        )
    total = float(values.sum())
    if total <= 0:
        raise InvalidWeightsError("Weights sum to zero.", code="zero_total_weight")
    normalized = numeric / total
    if drop_zero:
        normalized = normalized[normalized > 0]
    normalized.name = "weight"
    return normalized


def allocation_values(
    weights: Mapping[str, float] | pd.Series, capital: float
) -> pd.Series:
    """Monetary allocation per asset: ``V_i = w_i * capital``."""
    if not np.isfinite(capital):
        raise InvalidInputError("capital must be finite.", code="non_finite_input", params={"name": "capital"})
    if capital < 0:
        raise InvalidInputError("capital must not be negative.", code="negative_capital", params={"capital": capital})
    series = _as_weight_series(weights)
    values = series.to_numpy(dtype="float64") * float(capital)
    return pd.Series(values, index=series.index, name="value")


def split_currency_totals(
    weights: Mapping[str, float] | pd.Series,
    capital: float,
    currency_map: Mapping[str, str],
) -> dict[str, float]:
    """Sum the allocation value per currency (never mixes currencies).

    ``currency_map`` maps a symbol to its ISO currency code. Symbols missing
    from the map raise an error: silently assuming a currency would produce a
    wrong total.
    """
    series = _as_weight_series(weights)
    unknown = [symbol for symbol in series.index if symbol not in currency_map]
    if unknown:
        raise InvalidInputError(
            f"Missing currency information for: {unknown}",
            code="missing_currency",
            params={"symbols": unknown},
        )
    values = allocation_values(series, capital)
    totals: dict[str, float] = {}
    for symbol, value in values.items():
        currency = currency_map[symbol]
        totals[currency] = totals.get(currency, 0.0) + float(value)
    return totals


# ---------------------------------------------------------------------------
# Price alignment
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class AlignmentReport:
    """Diagnostics describing how prices were aligned across assets."""

    requested_symbols: tuple[str, ...]
    retained_symbols: tuple[str, ...]
    excluded_symbols: tuple[str, ...]
    coverage: dict[str, float]  #: ratio of non-missing prices per retained symbol
    observations_before: int
    observations_after: int
    dropped_rows: int
    first_date: pd.Timestamp | None
    last_date: pd.Timestamp | None
    warnings: tuple[str, ...] = ()

    @property
    def dropped_share(self) -> float:
        if self.observations_before <= 0:
            return 0.0
        return self.dropped_rows / self.observations_before


def align_price_frame(
    prices: pd.DataFrame,
    symbols: Sequence[str] | None = None,
    *,
    min_coverage: float = 0.6,
    drop_partial: bool = True,
    dropna_all: bool = True,
) -> tuple[pd.DataFrame, AlignmentReport]:
    """Align a price frame on common dates and validate coverage.

    Parameters
    ----------
    prices:
        Price levels indexed by date (rows) and symbol (columns).
    symbols:
        Requested symbols; others are ignored. ``None`` keeps every column.
    min_coverage:
        Minimum share of non-missing prices required for a symbol to be kept.
    drop_partial:
        Drop rows where any retained symbol has a missing price. A portfolio
        return needs every component, so this defaults to True.
    dropna_all:
        Also drop rows where all prices are missing (holidays, weekends).
    """
    if prices is None or prices.empty:
        raise InsufficientDataError(
            "No price data available.",
            code="insufficient_observations",
            params={"available": 0, "required": MIN_OBSERVATIONS_FOR_METRICS, "what": "prices"},
        )
    _require_numeric(prices, "prices")
    _check_finite(prices, "prices")
    if not 0.0 <= min_coverage <= 1.0:
        raise InvalidInputError(
            "min_coverage must be between 0 and 1.",
            code="invalid_min_coverage",
            params={"value": min_coverage},
        )

    frame = prices.copy()
    if not isinstance(frame.index, pd.DatetimeIndex):
        try:
            frame.index = pd.to_datetime(frame.index)
        except Exception as exc:  # pragma: no cover - defensive
            raise InvalidInputError(
                "Price index cannot be interpreted as dates.",
                code="invalid_date_index",
            ) from exc
    frame = frame.sort_index()
    frame = frame[~frame.index.duplicated(keep="last")]

    requested = tuple(str(symbol) for symbol in (symbols if symbols is not None else frame.columns))
    available = [symbol for symbol in requested if symbol in frame.columns]
    if not available:
        raise InvalidInputError(
            "None of the requested symbols are present in the price data.",
            code="no_common_assets",
            params={"assets": list(requested)},
        )
    frame = frame.loc[:, available]

    if dropna_all:
        frame = frame.dropna(how="all")

    observations_before = int(len(frame))
    coverage_denominator = max(observations_before, 1)
    coverage = {symbol: float(frame[symbol].notna().sum()) / coverage_denominator for symbol in frame.columns}

    warnings: list[str] = []
    excluded = [symbol for symbol, ratio in coverage.items() if ratio < min_coverage]
    if excluded:
        warnings.append("alignment.low_coverage_excluded")
    retained = [symbol for symbol in frame.columns if symbol not in excluded]
    frame = frame.loc[:, retained]
    if not retained:
        raise InsufficientDataError(
            "No asset has enough price history for the selected period.",
            code="insufficient_coverage",
            params={"min_coverage": min_coverage, "assets": list(excluded)},
        )

    if drop_partial:
        frame = frame.dropna(how="any")
    if frame.empty:
        raise InsufficientDataError(
            "No common observation dates across the selected assets.",
            code="no_overlapping_observations",
            params={"available": 0, "required": MIN_OBSERVATIONS_FOR_METRICS, "what": "prices"},
        )
    if len(frame) < MIN_OBSERVATIONS_FOR_METRICS:
        raise InsufficientDataError(
            "Not enough common observations to analyse the portfolio.",
            code="insufficient_observations",
            params={
                "available": int(len(frame)),
                "required": MIN_OBSERVATIONS_FOR_METRICS,
                "what": "prices",
            },
        )

    dropped_rows = observations_before - int(len(frame))
    if observations_before > 0 and dropped_rows / observations_before > 0.25:
        warnings.append("alignment.many_rows_dropped")
    if len(frame) < RECOMMENDED_MIN_OBSERVATIONS:
        warnings.append("alignment.short_history")

    report = AlignmentReport(
        requested_symbols=requested,
        retained_symbols=tuple(retained),
        excluded_symbols=tuple(excluded),
        coverage={symbol: coverage[symbol] for symbol in retained},
        observations_before=observations_before,
        observations_after=int(len(frame)),
        dropped_rows=int(dropped_rows),
        first_date=frame.index[0] if len(frame) else None,
        last_date=frame.index[-1] if len(frame) else None,
        warnings=tuple(warnings),
    )
    return frame, report


# ---------------------------------------------------------------------------
# Portfolio timeseries
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class PortfolioTimeseries:
    """Aligned prices, returns and wealth index for a portfolio."""

    prices: pd.DataFrame
    asset_returns: pd.DataFrame
    portfolio_returns: pd.Series
    wealth: pd.Series
    allocation: pd.Series
    weights: pd.Series
    initial_capital: float
    current_value: float
    total_return: float
    alignment: AlignmentReport

    @property
    def n_observations(self) -> int:
        return int(len(self.portfolio_returns))

    @property
    def first_date(self) -> pd.Timestamp | None:
        return self.alignment.first_date

    @property
    def last_date(self) -> pd.Timestamp | None:
        return self.alignment.last_date

    @property
    def cumulative_return(self) -> float:
        return self.total_return


def build_portfolio_timeseries(
    prices: pd.DataFrame,
    weights: Mapping[str, float] | pd.Series,
    *,
    initial_capital: float = 100_000.0,
    min_coverage: float = 0.6,
    normalize: bool = True,
) -> PortfolioTimeseries:
    """Full portfolio valuation pipeline.

    1. validate weights (long-only, sum to 1 after normalisation);
    2. align prices across assets and report coverage;
    3. compute per-asset and portfolio daily returns;
    4. compound the portfolio return into a wealth index starting at
       ``initial_capital``.
    """
    if not np.isfinite(initial_capital) or initial_capital <= 0:
        raise InvalidInputError(
            "initial_capital must be a strictly positive finite number.",
            code="invalid_capital",
            params={"capital": initial_capital},
        )
    validation = validate_weights(weights)
    if not validation.is_valid and not normalize:
        raise InvalidWeightsError(
            "Portfolio weights are invalid: " + ", ".join(validation.error_codes),
            code="invalid_weights",
            params={"errors": validation.error_codes, "total": validation.total},
        )
    normalized_weights = normalize_weights(weights)

    aligned_prices, report = align_price_frame(
        prices, list(normalized_weights.index), min_coverage=min_coverage
    )
    missing = [symbol for symbol in normalized_weights.index if symbol not in aligned_prices.columns]
    if missing:
        # An asset with no usable history cannot be valued: it is excluded and
        # the remaining weights are renormalised (reported through the report).
        normalized_weights = normalized_weights.drop(index=missing)
        if normalized_weights.empty:
            raise InsufficientDataError(
                "None of the portfolio assets have usable price history.",
                code="no_common_assets",
                params={"assets": missing},
            )
        normalized_weights = normalize_weights(normalized_weights)

    asset_returns = daily_returns(aligned_prices)
    portfolio_series = portfolio_returns(asset_returns, normalized_weights)
    returns_for_wealth = portfolio_series.dropna()
    if len(returns_for_wealth) < MIN_OBSERVATIONS_FOR_METRICS:
        raise InsufficientDataError(
            "Not enough returns to value the portfolio.",
            code="insufficient_observations",
            params={
                "available": int(len(returns_for_wealth)),
                "required": MIN_OBSERVATIONS_FOR_METRICS,
                "what": "portfolio valuation",
            },
        )
    wealth = float(initial_capital) * (1.0 + returns_for_wealth).cumprod()
    wealth.name = "value"

    return PortfolioTimeseries(
        prices=aligned_prices,
        asset_returns=asset_returns,
        portfolio_returns=returns_for_wealth,
        wealth=wealth,
        allocation=allocation_values(normalized_weights, initial_capital),
        weights=normalized_weights,
        initial_capital=float(initial_capital),
        current_value=float(wealth.iloc[-1]),
        total_return=float(wealth.iloc[-1] / float(initial_capital) - 1.0),
        alignment=report,
    )


# ---------------------------------------------------------------------------
# Metrics aggregation
# ---------------------------------------------------------------------------
def _safe(fn, *args, **kwargs) -> float:
    """Evaluate a metric, returning ``NaN`` instead of raising on thin data."""
    try:
        value = fn(*args, **kwargs)
    except (InsufficientDataError, InvalidInputError):
        return float("nan")
    try:
        value = float(value)
    except (TypeError, ValueError):  # pragma: no cover - defensive
        return float("nan")
    return value if np.isfinite(value) else float("nan")


def returns_summary(
    returns: pd.Series,
    *,
    risk_free_rate: float = 0.0,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
    initial_capital: float | None = None,
) -> dict[str, float]:
    """Compute the standard metric block for a return series (or ``NaN``).

    ``NaN`` signals "not computable with the available history" and is rendered
    as an em dash by the interface — never as an invented number.
    """
    if returns is None or len(returns) == 0:
        raise InsufficientDataError("No returns provided.", code="insufficient_observations", params={"available": 0, "required": 1, "what": "summary"})
    clean = returns.dropna()
    if clean.empty:
        raise InsufficientDataError(
            "The return series is empty after removing missing values.",
            code="insufficient_observations",
            params={"available": 0, "required": 1, "what": "summary"},
        )
    values = clean.to_numpy(dtype="float64")
    n = int(values.size)
    metric_kwargs = {"risk_free_rate": risk_free_rate, "periods_per_year": periods_per_year}
    summary = {
        "n_observations": float(n),
        "total_return": _safe(total_return, clean, dropna=False),
        "annualized_return": _safe(annualized_return, clean, periods_per_year=periods_per_year, dropna=False),
        "annualized_volatility": _safe(annualized_volatility, clean, periods_per_year=periods_per_year, dropna=False),
        "sharpe_ratio": _safe(sharpe_ratio, clean, dropna=False, **metric_kwargs),
        "sortino_ratio": _safe(sortino_ratio, clean, dropna=False, **metric_kwargs),
        "max_drawdown": _safe(max_drawdown, clean, dropna=False),
        "value_at_risk_95": _safe(value_at_risk, clean, level=0.95, dropna=False),
        "conditional_value_at_risk_95": _safe(conditional_value_at_risk, clean, level=0.95, dropna=False),
        "best_day": float(np.max(values)),
        "worst_day": float(np.min(values)),
        "positive_days_share": float(np.mean(values > 0)),
        "mean_daily_return": float(np.mean(values)),
        "median_daily_return": float(np.median(values)),
        "skewness": _safe(pd.Series(values).skew),
        "excess_kurtosis": _safe(lambda s: pd.Series(s).kurtosis(), values),
    }
    if initial_capital is not None:
        wealth = float(initial_capital) * np.cumprod(1.0 + values)
        summary["initial_capital"] = float(initial_capital)
        summary["current_value"] = float(wealth[-1])
        summary["gain_loss"] = float(wealth[-1] - float(initial_capital))
    return summary


def benchmark_comparison(
    returns: pd.Series,
    benchmark_returns: pd.Series,
    *,
    risk_free_rate: float = 0.0,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
) -> dict[str, float]:
    """Compare a portfolio return series with a benchmark return series."""
    if returns is None or benchmark_returns is None:
        raise InvalidInputError("Both return series are required.", code="missing_input", params={"name": "returns"})
    aligned = pd.concat([returns.rename("portfolio"), benchmark_returns.rename("benchmark")], axis=1).dropna()
    if aligned.empty:
        raise InsufficientDataError(
            "No overlapping observations between the portfolio and the benchmark.",
            code="no_overlapping_observations",
            params={"available": 0, "required": 1, "what": "benchmark comparison"},
        )
    portfolio = aligned["portfolio"]
    benchmark = aligned["benchmark"]
    beta, alpha = _safe_pair(beta_alpha, portfolio, benchmark, risk_free_rate=risk_free_rate, periods_per_year=periods_per_year)
    p_ann = _safe(annualized_return, portfolio, periods_per_year=periods_per_year, dropna=False)
    b_ann = _safe(annualized_return, benchmark, periods_per_year=periods_per_year, dropna=False)
    p_vol = _safe(annualized_volatility, portfolio, periods_per_year=periods_per_year, dropna=False)
    b_vol = _safe(annualized_volatility, benchmark, periods_per_year=periods_per_year, dropna=False)
    return {
        "n_observations": float(len(aligned)),
        "portfolio_annualized_return": p_ann,
        "benchmark_annualized_return": b_ann,
        "excess_annualized_return": p_ann - b_ann if np.isfinite(p_ann) and np.isfinite(b_ann) else float("nan"),
        "portfolio_annualized_volatility": p_vol,
        "benchmark_annualized_volatility": b_vol,
        "beta": beta,
        "alpha": alpha,
        "tracking_error": _safe(tracking_error, portfolio, benchmark, periods_per_year=periods_per_year),
        "information_ratio": _safe(information_ratio, portfolio, benchmark, periods_per_year=periods_per_year),
        "correlation": _safe(lambda a, b: pd.Series(a).corr(pd.Series(b)), portfolio.to_numpy(), benchmark.to_numpy()),
        "portfolio_total_return": _safe(total_return, portfolio, dropna=False),
        "benchmark_total_return": _safe(total_return, benchmark, dropna=False),
    }


def _safe_pair(fn, first: pd.Series, second: pd.Series, **kwargs) -> tuple[float, float]:
    try:
        pair = fn(first, second, **kwargs)
        values = tuple(float(value) for value in pair)
    except (InsufficientDataError, InvalidInputError):
        return float("nan"), float("nan")
    return tuple(
        value if np.isfinite(value) else float("nan") for value in values
    )  # type: ignore[return-value]
