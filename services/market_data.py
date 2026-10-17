"""Market data orchestration: provider selection, caching, diagnostics.

The service layer is the only place where providers are instantiated, so pages
never need to know whether prices come from a simulation, yfinance or an
imported CSV file.

Caching policy
--------------
Only *market data* is cached (``st.cache_data`` with a 30-minute TTL), keyed on
plain hashable arguments — never on session objects and never with a secret in
the payload. Portfolio and settings live in ``st.session_state`` and are always
read fresh.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Iterable

import pandas as pd

from config import (
    CACHE_TTL_SECONDS,
    DataMode,
    Market,
)
from data.providers import PriceRequest, PriceResult, get_provider
from data.universe import load_morocco_universe
from notices import Notice, info, warning
from services.cache import cached_data
from services.portfolio import Portfolio
from services.settings import AppSettings

__all__ = [
    "MarketDataBundle",
    "build_price_request",
    "load_market_data",
    "clear_market_data_cache",
    "request_from_portfolio",
    "cost_of_data_note",
]


@dataclass
class MarketDataBundle:
    """Provider result enriched with the context of the request."""

    result: PriceResult
    window: tuple[date, date]
    market: Market
    data_mode: DataMode
    requested_symbols: tuple[str, ...]
    benchmark: str | None
    headers: dict[str, str]

    # -- convenience passthroughs -----------------------------------------
    @property
    def ok(self) -> bool:
        return self.result.ok

    @property
    def prices(self) -> pd.DataFrame:
        return self.result.prices

    @property
    def benchmark_series(self) -> pd.Series | None:
        return self.result.benchmark

    @property
    def notices(self) -> list[Notice]:
        return self.result.notices

    @property
    def is_simulated(self) -> bool:
        return self.result.is_simulated

    @property
    def currency_map(self) -> dict[str, str]:
        return self.result.currency_map

    @property
    def last_observation(self) -> pd.Timestamp | None:
        return self.result.last_observation

    @property
    def data_source_key(self) -> str:
        return self.result.source_key

    @property
    def data_source_label(self) -> str:
        return self.result.source_label

    @property
    def errors(self) -> list[Notice]:
        return self.result.errors

    def compare_window(self) -> tuple[str | None, str | None]:
        first = self.result.first_observation
        last = self.result.last_observation
        return (
            str(first.date()) if first is not None else None,
            str(last.date()) if last is not None else None,
        )


def build_price_request(
    portfolio: Portfolio,
    settings: AppSettings,
    *,
    csv_payloads: Iterable[tuple[str, bytes]] = (),
    today: date | None = None,
) -> PriceRequest:
    """Translate a portfolio + settings into a provider-agnostic request."""
    start, end = settings.window(today=today)
    symbols = tuple(symbol for symbol in portfolio.weights if portfolio.weights[symbol] > 0)
    if not symbols:
        symbols = portfolio.symbols[:1]
    payloads = tuple((str(name), bytes(payload)) for name, payload in csv_payloads)
    local_files: tuple[str, ...] = ()
    if settings.data_mode is DataMode.HISTORICAL and settings.market is Market.MOROCCO:
        local_files = load_morocco_universe().csv_files
    return PriceRequest(
        symbols=symbols,
        start=start,
        end=end,
        market=settings.market,
        benchmark=portfolio.benchmark,
        csv_payloads=payloads,
        local_csv_files=local_files,
        seed=settings.demo_seed if settings.data_mode is DataMode.DEMO else None,
    )


@cached_data(ttl=CACHE_TTL_SECONDS, show_spinner=False)
def _load_market_data_cached(
    symbols: tuple[str, ...],
    start_iso: str,
    end_iso: str,
    market_value: str,
    data_mode_value: str,
    benchmark: str | None,
    csv_payloads: tuple[tuple[str, bytes], ...],
    local_csv_files: tuple[str, ...],
    seed: int | None,
) -> PriceResult:
    """Cached provider call (hashable arguments only)."""
    request = PriceRequest(
        symbols=symbols,
        start=date.fromisoformat(start_iso),
        end=date.fromisoformat(end_iso),
        market=Market(market_value),
        benchmark=benchmark,
        csv_payloads=csv_payloads,
        local_csv_files=local_csv_files,
        seed=seed,
    )
    provider = get_provider(request.market, data_mode_value)
    try:
        return provider.fetch(request)
    except Exception as exc:  # pragma: no cover - provider contracts forbid raising
        return PriceResult(
            requested_symbols=symbols,
            provider_key=getattr(provider, "key", "unknown"),
            source_key=getattr(provider, "key", "unknown"),
            source_label="unavailable",
        ).failed("error.unexpected_error", detail=f"{type(exc).__name__}: {exc}"[:300])


def load_market_data(
    portfolio: Portfolio,
    settings: AppSettings,
    *,
    csv_payloads: Iterable[tuple[str, bytes]] = (),
    today: date | None = None,
) -> MarketDataBundle:
    """Load (and cache) the price history for a portfolio.

    Never raises for a data problem: failures are reported through error
    notices, and no simulated data is ever substituted for real data.
    """
    request = build_price_request(portfolio, settings, csv_payloads=csv_payloads, today=today)
    result = _load_market_data_cached(
        tuple(request.symbols),
        request.start.isoformat(),
        request.end.isoformat(),
        request.market.value,
        settings.data_mode.value,
        request.benchmark,
        tuple(request.csv_payloads),
        tuple(request.local_csv_files),
        request.seed,
    )
    notices = list(result.notices)
    if not result.ok and not result.errors:
        notices.append(
            warning("notice.us.no_data", symbols=list(request.symbols))
            if settings.market is Market.US
            else warning("notice.data.no_assets_for_market")
        )
    headers = {
        "market": request.market.value,
        "mode": settings.data_mode.value,
        "start": request.start.isoformat(),
        "end": request.end.isoformat(),
        "source": result.source_key,
    }
    return MarketDataBundle(
        result=result,
        window=(request.start, request.end),
        market=request.market,
        data_mode=settings.data_mode,
        requested_symbols=tuple(request.symbols),
        benchmark=request.benchmark,
        headers=headers,
    )


def clear_market_data_cache() -> bool:
    """Clear the cached market data (Settings page action)."""
    from services.cache import clear_caches

    return clear_caches()


def request_from_portfolio(
    portfolio: Portfolio,
    settings: AppSettings,
    *,
    csv_payloads: Iterable[tuple[str, bytes]] = (),
) -> PriceRequest:
    """Alias kept explicit for readability in tests and the AI context builder."""
    return build_price_request(portfolio, settings, csv_payloads=csv_payloads)


def cost_of_data_note(settings: AppSettings) -> Notice:
    """Informational notice describing the active data source."""
    if settings.data_mode is DataMode.DEMO:
        return info("notice.demo.simulated_data", seed=settings.demo_seed, market=settings.market.value)
    return info(
        "notice.us.data_loaded" if settings.market is Market.US else "notice.ma.data_loaded",
        count="—",
        start="—",
        end="—",
    )
