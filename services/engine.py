"""Analytics engine: one shared bundle of computed results for every page.

``build_analytics_bundle`` runs the whole pipeline for the current portfolio:

    market data (cached) -> alignment -> returns -> portfolio valuation
        -> risk / correlation / concentration / benchmark metrics
        -> per-currency allocation totals

Every failure mode is converted into a structured :class:`Notice` so pages can
render a friendly, translated message instead of a traceback, and no page ever
recomputes a metric with its own formula.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

import numpy as np
import pandas as pd

from analytics.correlation import (
    CorrelationResult,
    average_pairwise_correlation,
    correlation_details,
    covariance_matrix,
    diversification_ratio,
)
from analytics.exceptions import FinPilotError, InsufficientDataError
from analytics.portfolio import (
    AlignmentReport,
    PortfolioTimeseries,
    benchmark_comparison,
    build_portfolio_timeseries,
    returns_summary,
)
from analytics.returns import daily_returns
from analytics.risk import ConcentrationMetrics, concentration_metrics, risk_contributions
from config import RECOMMENDED_MIN_OBSERVATIONS, TRADING_DAYS_PER_YEAR, DataMode, Market
from notices import Notice, error, info, warning
from services.currency import CurrencyTotals, allocation_by_currency
from services.market_data import MarketDataBundle, load_market_data
from services.portfolio import Portfolio
from services.settings import AppSettings

__all__ = ["AnalyticsBundle", "build_analytics_bundle", "get_bundle", "refresh_bundle"]


@dataclass
class AnalyticsBundle:
    """Everything the interface needs to display a portfolio analysis."""

    ok: bool
    portfolio: Portfolio
    settings: AppSettings
    market: Market
    data_mode: DataMode
    window: tuple[date, date]

    # data ---------------------------------------------------------------
    prices: pd.DataFrame = field(default_factory=pd.DataFrame)
    asset_returns: pd.DataFrame = field(default_factory=pd.DataFrame)
    benchmark_prices: pd.Series | None = None
    benchmark_returns: pd.Series | None = None
    currency_map: dict[str, str] = field(default_factory=dict)
    is_simulated: bool = False
    data_source_key: str = "unknown"
    data_source_label: str = "unknown"
    last_observation: pd.Timestamp | None = None
    first_observation: pd.Timestamp | None = None
    coverage: dict[str, float] = field(default_factory=dict)
    dropped_observations: int = 0
    requested_symbols: tuple[str, ...] = ()
    available_symbols: tuple[str, ...] = ()
    missing_symbols: tuple[str, ...] = ()

    # analytics ----------------------------------------------------------
    timeseries: PortfolioTimeseries | None = None
    metrics: dict[str, float] = field(default_factory=dict)
    benchmark_metrics: dict[str, float] = field(default_factory=dict)
    correlation: CorrelationResult | None = None
    average_correlation: float = float("nan")
    covariance_annual: pd.DataFrame | None = None
    concentration: ConcentrationMetrics | None = None
    diversification_ratio: float = float("nan")
    risk_contributions: pd.Series | None = None
    currency_totals: CurrencyTotals | None = None

    notices: list[Notice] = field(default_factory=list)

    # -- convenience ------------------------------------------------------
    @property
    def weights(self) -> pd.Series:
        if self.timeseries is not None:
            return self.timeseries.weights
        return self.portfolio.weights_series()

    @property
    def allocation(self) -> pd.Series:
        if self.timeseries is not None:
            return self.timeseries.allocation
        return self.portfolio.allocation()

    @property
    def capital(self) -> float:
        return float(self.portfolio.capital)

    @property
    def current_value(self) -> float | None:
        if self.timeseries is None:
            return None
        return self.timeseries.current_value

    @property
    def cumulative_return(self) -> float | None:
        if self.timeseries is None:
            return None
        return self.timeseries.total_return

    @property
    def portfolio_returns(self) -> pd.Series | None:
        return self.timeseries.portfolio_returns if self.timeseries is not None else None

    @property
    def wealth(self) -> pd.Series | None:
        return self.timeseries.wealth if self.timeseries is not None else None

    @property
    def n_observations(self) -> int:
        return self.timeseries.n_observations if self.timeseries is not None else 0

    @property
    def short_history(self) -> bool:
        return 0 < self.n_observations < RECOMMENDED_MIN_OBSERVATIONS

    @property
    def errors(self) -> list[Notice]:
        return [notice for notice in self.notices if notice.is_error]

    @property
    def warnings(self) -> list[Notice]:
        return [notice for notice in self.notices if notice.level == "warning"]

    @property
    def currency(self) -> str:
        return self.portfolio.market.currency

    def metric(self, name: str, default: float = float("nan")) -> float:
        value = self.metrics.get(name, default)
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return default
        return numeric

    def display_metric(self, name: str) -> float | None:
        """Return a metric or ``None`` when it is not computable (renders ``—``)."""
        value = self.metric(name)
        return None if not np.isfinite(value) else value


def _empty_bundle(
    portfolio: Portfolio,
    settings: AppSettings,
    data: MarketDataBundle | None,
    notices: list[Notice],
    *,
    coverage: dict[str, float] | None = None,
) -> AnalyticsBundle:
    return AnalyticsBundle(
        ok=False,
        portfolio=portfolio,
        settings=settings,
        market=settings.market,
        data_mode=settings.data_mode,
        window=data.window if data is not None else settings.window(),
        prices=data.prices if data is not None else pd.DataFrame(),
        benchmark_prices=data.benchmark_series if data is not None else None,
        currency_map=data.currency_map if data is not None else dict(portfolio.currency_map),
        is_simulated=bool(data.is_simulated) if data is not None else settings.data_mode is DataMode.DEMO,
        data_source_key=data.data_source_key if data is not None else "unavailable",
        data_source_label=data.data_source_label if data is not None else "unavailable",
        last_observation=data.last_observation if data is not None else None,
        first_observation=data.result.first_observation if data is not None else None,
        coverage=coverage or {},
        requested_symbols=data.requested_symbols if data is not None else tuple(portfolio.symbols),
        available_symbols=data.result.available_symbols if data is not None else (),
        missing_symbols=data.result.missing_symbols if data is not None else (),
        notices=notices,
    )


def build_analytics_bundle(
    portfolio: Portfolio,
    settings: AppSettings,
    *,
    csv_payloads: tuple[tuple[str, bytes], ...] = (),
    today: date | None = None,
) -> AnalyticsBundle:
    """Run the full pipeline for a portfolio (never raises for data problems)."""
    settings = settings.normalised()
    notices: list[Notice] = []

    if portfolio.is_empty:
        notices.append(error("notice.data.no_assets_for_market"))
        return _empty_bundle(portfolio, settings, None, notices)

    data = load_market_data(portfolio, settings, csv_payloads=csv_payloads, today=today)
    notices.extend(data.notices)

    weights = portfolio.weights_series()
    weights = weights[weights > 0]
    if weights.empty:
        notices.append(error("notice.data.no_assets_for_market"))
        return _empty_bundle(portfolio, settings, data, notices)

    # --- alignment & valuation ------------------------------------------
    try:
        timeseries = build_portfolio_timeseries(
            data.prices,
            weights,
            initial_capital=portfolio.capital,
            min_coverage=settings.min_coverage,
        )
    except FinPilotError as exc:
        notices.append(error(exc.code, **exc.params))
        return _empty_bundle(portfolio, settings, data, notices)
    except Exception as exc:  # pragma: no cover - defensive
        notices.append(error("error.unexpected_error", detail=f"{type(exc).__name__}: {exc}"[:200]))
        return _empty_bundle(portfolio, settings, data, notices)

    alignment: AlignmentReport = timeseries.alignment
    coverage = dict(alignment.coverage)
    alignment_params = {
        "symbols": list(alignment.excluded_symbols),
        "required": RECOMMENDED_MIN_OBSERVATIONS,
    }
    for code in alignment.warnings:
        notices.append(warning(code, **alignment_params))

    asset_returns = timeseries.asset_returns
    portfolio_returns = timeseries.portfolio_returns

    # --- headline metrics ------------------------------------------------
    metrics = returns_summary(
        portfolio_returns,
        risk_free_rate=settings.risk_free_rate,
        periods_per_year=TRADING_DAYS_PER_YEAR,
        initial_capital=portfolio.capital,
    )
    metrics["assets_used"] = float(len(timeseries.weights))
    metrics["total_dropped"] = float(alignment.dropped_rows)

    # --- benchmark --------------------------------------------------------
    benchmark_returns: pd.Series | None = None
    benchmark_prices = data.benchmark_series
    benchmark_metrics: dict[str, float] = {}
    if benchmark_prices is not None and len(benchmark_prices.dropna()) > 1:
        try:
            benchmark_returns = daily_returns(benchmark_prices.dropna())
            benchmark_returns.name = data.result.benchmark_symbol or "benchmark"
            benchmark_metrics = benchmark_comparison(
                portfolio_returns,
                benchmark_returns,
                risk_free_rate=settings.risk_free_rate,
                periods_per_year=TRADING_DAYS_PER_YEAR,
            )
            metrics["benchmark_annualized_return"] = benchmark_metrics.get(
                "benchmark_annualized_return", float("nan")
            )
            metrics["excess_annualized_return"] = benchmark_metrics.get(
                "excess_annualized_return", float("nan")
            )
            metrics["beta"] = benchmark_metrics.get("beta", float("nan"))
            metrics["alpha"] = benchmark_metrics.get("alpha", float("nan"))
            metrics["tracking_error"] = benchmark_metrics.get("tracking_error", float("nan"))
            metrics["information_ratio"] = benchmark_metrics.get("information_ratio", float("nan"))
            metrics["benchmark_correlation"] = benchmark_metrics.get("correlation", float("nan"))
        except FinPilotError as exc:
            notices.append(warning(exc.code, **exc.params))
    elif portfolio.benchmark:
        notices.append(warning("us.benchmark_unavailable", symbol=portfolio.benchmark))
    else:
        notices.append(info("data.benchmark_unavailable"))

    # --- correlation / covariance ----------------------------------------
    correlation: CorrelationResult | None = None
    covariance_annual: pd.DataFrame | None = None
    average_correlation = float("nan")
    diversification = float("nan")
    contributions: pd.Series | None = None
    if asset_returns.shape[1] >= 2:
        try:
            correlation = correlation_details(asset_returns)
            average_correlation = correlation.average
            covariance_annual = covariance_matrix(asset_returns, periods_per_year=TRADING_DAYS_PER_YEAR)
            diversification = diversification_ratio(
                timeseries.weights, asset_returns, periods_per_year=TRADING_DAYS_PER_YEAR
            )
            contributions = risk_contributions(timeseries.weights, covariance_annual)
        except FinPilotError as exc:
            notices.append(warning(exc.code, **exc.params))
    else:
        notices.append(info("risk.insufficient"))

    # --- concentration ----------------------------------------------------
    concentration: ConcentrationMetrics | None = None
    try:
        concentration = concentration_metrics(timeseries.weights)
    except FinPilotError as exc:
        notices.append(warning(exc.code, **exc.params))

    # --- currency-aware totals -------------------------------------------
    currency_map = dict(data.currency_map)
    for symbol in timeseries.weights.index:
        currency_map.setdefault(symbol, portfolio.currency_map.get(symbol, portfolio.market.currency))
    try:
        currency_totals = allocation_by_currency(timeseries.weights, portfolio.capital, currency_map)
    except FinPilotError as exc:
        notices.append(error(exc.code, **exc.params))
        currency_totals = None

    if timeseries.n_observations < RECOMMENDED_MIN_OBSERVATIONS:
        notices.append(info("alignment.short_history", required=RECOMMENDED_MIN_OBSERVATIONS))

    bundle = AnalyticsBundle(
        ok=True,
        portfolio=portfolio,
        settings=settings,
        market=settings.market,
        data_mode=settings.data_mode,
        window=data.window,
        prices=timeseries.prices,
        asset_returns=asset_returns,
        benchmark_prices=benchmark_prices,
        benchmark_returns=benchmark_returns,
        currency_map=currency_map,
        is_simulated=data.is_simulated,
        data_source_key=data.data_source_key,
        data_source_label=data.data_source_label,
        last_observation=timeseries.last_date,
        first_observation=timeseries.first_date,
        coverage=coverage,
        dropped_observations=alignment.dropped_rows,
        requested_symbols=data.requested_symbols,
        available_symbols=tuple(timeseries.prices.columns),
        missing_symbols=tuple(
            symbol for symbol in data.requested_symbols if symbol not in timeseries.prices.columns
        ),
        timeseries=timeseries,
        metrics=metrics,
        benchmark_metrics=benchmark_metrics,
        correlation=correlation,
        average_correlation=average_correlation,
        covariance_annual=covariance_annual,
        concentration=concentration,
        diversification_ratio=diversification,
        risk_contributions=contributions,
        currency_totals=currency_totals,
        notices=notices,
    )
    return bundle


# ---------------------------------------------------------------------------
# Session-backed access
# ---------------------------------------------------------------------------
def get_bundle(*, force: bool = False) -> AnalyticsBundle:
    """Return the analytics bundle for the current session.

    The bundle is computed once per (portfolio, settings) state and cached in
    ``st.session_state``, so navigating between pages does not recompute it.
    Market data itself is cached at the service layer (``st.cache_data``).
    """
    from services.state import get_csv_payloads, get_portfolio, get_settings

    portfolio = get_portfolio()
    settings = get_settings()
    key = f"{portfolio.cache_key()}|{settings.snapshot()}"

    try:
        import streamlit as st

        cache = st.session_state.get("fp_bundle_cache", {})
        if not force and key in cache:
            return cache[key]
        bundle = build_analytics_bundle(portfolio, settings, csv_payloads=get_csv_payloads())
        st.session_state["fp_bundle_cache"] = {key: bundle}
        return bundle
    except Exception:  # pragma: no cover - no Streamlit runtime (tests/scripts)
        return build_analytics_bundle(portfolio, settings, csv_payloads=get_csv_payloads())


def refresh_bundle() -> AnalyticsBundle:
    """Force a recomputation (used after import or settings changes)."""
    return get_bundle(force=True)


def bundle_needs(market: Market, data_mode: DataMode) -> bool:
    """Whether the cached bundle matches the requested context."""
    bundle = get_bundle()
    return bundle.market is market and bundle.data_mode is data_mode


def ensure_portfolio_data() -> None:
    """Convenience guard used by pages that require usable data."""
    bundle = get_bundle()
    if not bundle.ok:
        raise InsufficientDataError(
            "The portfolio has no usable market data.",
            code="market_data_unavailable",
        )
