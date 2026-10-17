"""Provider abstraction for market data.

The application only ever consumes :class:`PriceResult` objects, so a new data
source (a verified Casablanca Stock Exchange feed, a paid vendor, a database)
can be plugged in by implementing :class:`MarketDataProvider` and registering it
in :func:`get_provider` — no page or analytics code has to change.

Failure policy
--------------
A provider that cannot deliver data returns a result with ``ok=False`` and an
explicit error notice. It **never** silently substitutes simulated prices: the
user decides whether to switch to demo mode or import a CSV file.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Sequence

import pandas as pd

from config import DataMode, Market
from notices import Notice, error

__all__ = [
    "PriceRequest",
    "PriceResult",
    "MarketDataProvider",
    "get_provider",
    "available_providers",
]


@dataclass(frozen=True)
class PriceRequest:
    """Everything a provider needs to answer a price history request."""

    symbols: tuple[str, ...]
    start: date
    end: date
    market: Market = Market.US
    benchmark: str | None = None
    #: In-memory CSV payloads ``(filename, raw_bytes)`` coming from the UI uploader.
    csv_payloads: tuple[tuple[str, bytes], ...] = ()
    #: Local CSV files declared in the Moroccan universe JSON.
    local_csv_files: tuple[str, ...] = ()
    #: Deterministic seed used by the demo provider only.
    seed: int | None = None

    def __post_init__(self) -> None:
        if not self.symbols:
            raise ValueError("PriceRequest requires at least one symbol.")
        if self.start > self.end:
            raise ValueError("PriceRequest start date must be before the end date.")

    @property
    def is_empty(self) -> bool:
        return not self.symbols

    def with_symbols(self, symbols: Sequence[str]) -> "PriceRequest":
        return PriceRequest(
            symbols=tuple(symbols),
            start=self.start,
            end=self.end,
            market=self.market,
            benchmark=self.benchmark,
            csv_payloads=self.csv_payloads,
            local_csv_files=self.local_csv_files,
            seed=self.seed,
        )


@dataclass
class PriceResult:
    """Normalised provider answer."""

    prices: pd.DataFrame = field(default_factory=lambda: pd.DataFrame())
    benchmark: pd.Series | None = None
    benchmark_symbol: str | None = None
    currency_map: dict[str, str] = field(default_factory=dict)
    names: dict[str, str] = field(default_factory=dict)
    source_key: str = "unknown"
    source_label: str = "unknown"
    is_simulated: bool = False
    requested_symbols: tuple[str, ...] = ()
    available_symbols: tuple[str, ...] = ()
    missing_symbols: tuple[str, ...] = ()
    notices: list[Notice] = field(default_factory=list)
    provider_key: str = "unknown"

    # -- convenience -------------------------------------------------------
    @property
    def ok(self) -> bool:
        return not self.prices.empty and not any(notice.is_error for notice in self.notices)

    @property
    def errors(self) -> list[Notice]:
        return [notice for notice in self.notices if notice.is_error]

    @property
    def warnings(self) -> list[Notice]:
        return [notice for notice in self.notices if notice.level == "warning"]

    @property
    def last_observation(self) -> pd.Timestamp | None:
        if self.prices.empty:
            return None
        return self.prices.index[-1]

    @property
    def first_observation(self) -> pd.Timestamp | None:
        if self.prices.empty:
            return None
        return self.prices.index[0]

    @property
    def currencies(self) -> tuple[str, ...]:
        seen: list[str] = []
        for symbol in self.prices.columns if not self.prices.empty else self.currency_map:
            code = self.currency_map.get(str(symbol), "USD")
            if code not in seen:
                seen.append(code)
        return tuple(seen)

    def currency_of(self, symbol: str) -> str:
        return self.currency_map.get(symbol, self.currency_map.get("*", "USD"))

    def failed(self, code: str, **params) -> "PriceResult":
        self.notices.append(error(code, **params))
        return self


class MarketDataProvider(ABC):
    """Interface implemented by every market data source."""

    key: str = "abstract"
    label_key: str = "provider.abstract"
    market: Market = Market.US
    data_mode: DataMode = DataMode.HISTORICAL
    is_simulated: bool = False

    @abstractmethod
    def fetch(self, request: PriceRequest) -> PriceResult:
        """Return prices for ``request`` or a result carrying an error notice."""

    # -- optional hooks ----------------------------------------------------
    def supports(self, symbol: str) -> bool:  # pragma: no cover - default
        return True

    def describe(self) -> dict:
        return {
            "key": self.key,
            "label_key": self.label_key,
            "market": self.market.value,
            "data_mode": self.data_mode.value,
            "is_simulated": self.is_simulated,
        }


def get_provider(market: Market, data_mode: DataMode | str, **kwargs) -> MarketDataProvider:
    """Factory returning the provider for a (market, data mode) pair."""
    from data.providers.morocco_market import MoroccoMarketProvider
    from data.providers.us_market import UsMarketProvider

    if isinstance(data_mode, str):
        data_mode = DataMode(data_mode)
    if data_mode is DataMode.DEMO:
        from data.providers.demo_provider import DemoMarketProvider

        return DemoMarketProvider(market)
    if market is Market.MOROCCO:
        return MoroccoMarketProvider(**kwargs)
    return UsMarketProvider(**kwargs)


def available_providers() -> dict[str, dict]:
    """Describe every registered provider (used by the Settings page)."""
    from data.providers.demo_provider import DemoMarketProvider
    from data.providers.morocco_market import MoroccoMarketProvider
    from data.providers.us_market import UsMarketProvider

    providers = [
        DemoMarketProvider(Market.US),
        DemoMarketProvider(Market.MOROCCO),
        UsMarketProvider(),
        MoroccoMarketProvider(),
    ]
    return {provider.key: provider.describe() for provider in providers}


def default_window(days: int = 365 * 5, *, today: date | None = None) -> tuple[date, date]:
    today = today or date.today()
    return today - timedelta(days=days), today
