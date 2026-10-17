"""Market data providers (US via yfinance, Morocco via CSV adapters)."""

from __future__ import annotations

from data.providers.base import (
    MarketDataProvider,
    PriceRequest,
    PriceResult,
    get_provider,
    available_providers,
)
from data.providers.morocco_market import MoroccoMarketProvider, MoroccoCsvAdapter, MoroccoDataAdapter
from data.providers.us_market import UsMarketProvider

__all__ = [
    "MarketDataProvider",
    "PriceRequest",
    "PriceResult",
    "get_provider",
    "available_providers",
    "UsMarketProvider",
    "MoroccoMarketProvider",
    "MoroccoCsvAdapter",
    "MoroccoDataAdapter",
]
