"""The portfolio domain object and its reconciliation rules.

A portfolio is a plain, serialisable value object (no database in this MVP):

* ``capital``      — initial investment amount, in the portfolio currency;
* ``weights``      — fractions summing to 1 (long-only);
* ``assets``       — instrument metadata (symbol -> :class:`Instrument`);
* ``benchmark``    — optional reference symbol;
* ``market``/``data_mode`` — the context the portfolio was built for.

Reconciliation is the rule that keeps the app honest: switching market or data
mode never silently keeps incompatible holdings, and a demo portfolio can never
be built out of real tickers (which would amount to publishing invented prices
for real securities).
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from typing import Iterable, Mapping

import pandas as pd

from analytics.exceptions import InvalidWeightsError
from analytics.portfolio import allocation_values, validate_weights
from config import DEFAULT_CAPITAL, AssetClass, DataMode, Market
from data.universe import (
    DEMO_UNIVERSE_MOROCCO,
    DEMO_UNIVERSE_US,
    US_UNIVERSE,
    US_BENCHMARKS,
    Instrument,
    benchmarks_for,
    instrument_lookup,
    load_morocco_universe,
    universe_for,
)
from notices import Notice, info

__all__ = [
    "Portfolio",
    "DEFAULT_WEIGHTS",
    "default_portfolio",
    "reconcile_portfolio",
    "portfolio_from_symbols",
]


#: Default allocations per (market, data mode). Demo allocations only use
#: fictional ``DEMO-*`` instruments; historical US allocations only use real
#: tickers that yfinance can serve.
DEFAULT_WEIGHTS: dict[tuple[Market, DataMode], dict[str, float]] = {
    (Market.US, DataMode.DEMO): {
        "DEMO-US-EQ": 0.35,
        "DEMO-US-TECH": 0.20,
        "DEMO-US-BND": 0.25,
        "DEMO-US-GOLD": 0.10,
        "DEMO-US-REIT": 0.10,
    },
    (Market.MOROCCO, DataMode.DEMO): {
        "DEMO-MA-EQ-A": 0.30,
        "DEMO-MA-BANK": 0.25,
        "DEMO-MA-EQ-B": 0.20,
        "DEMO-MA-BND": 0.20,
        "DEMO-MA-CASH": 0.05,
    },
    (Market.US, DataMode.HISTORICAL): {
        "SPY": 0.40,
        "AGG": 0.25,
        "QQQ": 0.15,
        "GLD": 0.10,
        "VNQ": 0.10,
    },
    # No default Moroccan allocation: the user must provide verified data.
    (Market.MOROCCO, DataMode.HISTORICAL): {},
}

DEFAULT_BENCHMARKS: dict[tuple[Market, DataMode], str | None] = {
    (Market.US, DataMode.DEMO): "DEMO-US-MKT",
    (Market.MOROCCO, DataMode.DEMO): "DEMO-MA-MKT",
    (Market.US, DataMode.HISTORICAL): "^GSPC",
    (Market.MOROCCO, DataMode.HISTORICAL): None,
}


@dataclass
class Portfolio:
    """A long-only allocation, its metadata and its data context."""

    weights: dict[str, float] = field(default_factory=dict)
    capital: float = DEFAULT_CAPITAL
    benchmark: str | None = None
    market: Market = Market.US
    data_mode: DataMode = DataMode.DEMO
    assets: dict[str, Instrument] = field(default_factory=dict)
    name: str = "My portfolio"
    updated_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds")
    )

    # -- basics -----------------------------------------------------------
    def __post_init__(self) -> None:
        self.weights = {str(symbol): float(weight) for symbol, weight in self.weights.items()}

    @property
    def symbols(self) -> tuple[str, ...]:
        return tuple(self.weights.keys())

    @property
    def n_assets(self) -> int:
        return sum(1 for weight in self.weights.values() if weight > 0)

    @property
    def total_weight(self) -> float:
        return float(sum(self.weights.values()))

    @property
    def is_empty(self) -> bool:
        return not self.weights or self.total_weight <= 0

    @property
    def is_demo(self) -> bool:
        return self.data_mode is DataMode.DEMO

    @property
    def currency_map(self) -> dict[str, str]:
        """Currency per symbol, taken from the instrument metadata."""
        default = self.market.currency
        return {
            symbol: (self.assets[symbol].currency if symbol in self.assets else default)
            for symbol in self.weights
        }

    @property
    def currencies(self) -> tuple[str, ...]:
        seen: list[str] = []
        for code in self.currency_map.values():
            if code not in seen:
                seen.append(code)
        return tuple(seen)

    @property
    def is_multi_currency(self) -> bool:
        return len(self.currencies) > 1

    @property
    def benchmark_currency(self) -> str | None:
        if not self.benchmark:
            return None
        instrument = self.assets.get(self.benchmark)
        return instrument.currency if instrument else self.market.currency

    # -- derived ----------------------------------------------------------
    def weights_series(self) -> pd.Series:
        series = pd.Series(self.weights, dtype="float64")
        series.name = "weight"
        return series

    def allocation(self, capital: float | None = None) -> pd.Series:
        """Monetary allocation per asset."""
        return allocation_values(self.weights_series(), self.capital if capital is None else capital)

    def validation(self):
        """Weight validation (never raises)."""
        return validate_weights(self.weights_series())

    def normalized(self) -> "Portfolio":
        """Return a copy whose weights sum to exactly 1 (or raise)."""
        series = self.weights_series()
        total = float(series.sum())
        if total <= 0:
            raise InvalidWeightsError("The portfolio has no positive weight.", code="zero_total_weight")
        if any(weight < 0 for weight in series):
            raise InvalidWeightsError(
                "Long-only portfolios cannot contain negative weights.",
                code="negative_weights",
            )
        return self.with_weights({str(symbol): float(weight / total) for symbol, weight in series.items()})

    def scaled_allocation(self) -> pd.Series:
        """Allocation normalised to the capital actually invested."""
        series = self.weights_series()
        total = float(series.sum())
        if total <= 0:
            return pd.Series(dtype="float64")
        return (series / total) * float(self.capital)

    # -- mutation ---------------------------------------------------------
    def with_weights(self, weights: Mapping[str, float]) -> "Portfolio":
        return replace(
            self,
            weights={str(symbol): float(weight) for symbol, weight in weights.items()},
            updated_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        )

    def with_assets(self, assets: Mapping[str, Instrument]) -> "Portfolio":
        merged = dict(self.assets)
        merged.update({str(symbol): instrument for symbol, instrument in assets.items()})
        return replace(self, assets=merged)

    def with_capital(self, capital: float) -> "Portfolio":
        return replace(self, capital=float(capital))

    def with_benchmark(self, benchmark: str | None) -> "Portfolio":
        return replace(self, benchmark=benchmark)

    def with_context(self, market: Market, data_mode: DataMode) -> "Portfolio":
        return replace(self, market=market, data_mode=data_mode)

    def add_asset(self, instrument: Instrument, weight: float = 0.0) -> "Portfolio":
        assets = dict(self.assets)
        assets[instrument.symbol] = instrument
        weights = dict(self.weights)
        weights.setdefault(instrument.symbol, float(weight))
        return replace(
            self,
            assets=assets,
            weights=weights,
            updated_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        )

    def remove_asset(self, symbol: str) -> "Portfolio":
        weights = {key: value for key, value in self.weights.items() if key != symbol}
        return replace(
            self,
            weights=weights,
            benchmark=None if self.benchmark == symbol else self.benchmark,
            updated_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        )

    # -- serialisation ----------------------------------------------------
    def to_dict(self) -> dict:
        return {
            "weights": dict(self.weights),
            "capital": float(self.capital),
            "benchmark": self.benchmark,
            "market": self.market.value,
            "data_mode": self.data_mode.value,
            "assets": {symbol: instrument.as_dict() for symbol, instrument in self.assets.items()},
            "name": self.name,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, payload: Mapping | None) -> "Portfolio":
        if not payload:
            return default_portfolio()
        weights = {str(symbol): float(weight) for symbol, weight in (payload.get("weights") or {}).items()}
        try:
            market = Market(str(payload.get("market", Market.US.value)).upper())
        except ValueError:
            market = Market.US
        try:
            data_mode = DataMode(str(payload.get("data_mode", DataMode.DEMO.value)).lower())
        except ValueError:
            data_mode = DataMode.DEMO
        assets: dict[str, Instrument] = {}
        for symbol, meta in (payload.get("assets") or {}).items():
            if isinstance(meta, dict) and meta.get("symbol"):
                assets[str(symbol)] = Instrument.from_dict(
                    meta, default_market=market, default_currency=market.currency
                )
            elif isinstance(meta, Instrument):
                assets[str(symbol)] = meta
        benchmark = payload.get("benchmark")
        return cls(
            weights=weights,
            capital=float(payload.get("capital", DEFAULT_CAPITAL) or DEFAULT_CAPITAL),
            benchmark=str(benchmark) if benchmark else None,
            market=market,
            data_mode=data_mode,
            assets=assets,
            name=str(payload.get("name", "My portfolio")),
            updated_at=str(payload.get("updated_at", datetime.now(timezone.utc).isoformat(timespec="seconds"))),
        )

    def cache_key(self) -> str:
        """Stable, secret-free key for caching derived analytics."""
        weights = ",".join(f"{symbol}:{weight:.8f}" for symbol, weight in sorted(self.weights.items()))
        return (
            f"{self.market.value}|{self.data_mode.value}|{self.capital:.4f}|"
            f"{self.benchmark or '-'}|{weights}"
        )


# ---------------------------------------------------------------------------
# Defaults & reconciliation
# ---------------------------------------------------------------------------
def default_portfolio(
    market: Market | None = None,
    data_mode: DataMode | None = None,
    *,
    capital: float = DEFAULT_CAPITAL,
) -> Portfolio:
    """Build the default portfolio for a (market, data mode) pair."""
    market = market or Market.US
    data_mode = data_mode or DataMode.DEMO
    weights = dict(DEFAULT_WEIGHTS.get((market, data_mode), {}))
    benchmark = DEFAULT_BENCHMARKS.get((market, data_mode))
    assets = instrument_lookup(
        [*weights.keys(), *([benchmark] if benchmark else [])],
        market=market,
        data_mode=data_mode.value,
    )
    return Portfolio(
        weights=weights,
        capital=capital,
        benchmark=benchmark,
        market=market,
        data_mode=data_mode,
        assets=assets,
    )


def portfolio_from_symbols(
    symbols: Iterable[str],
    *,
    market: Market,
    data_mode: DataMode,
    capital: float = DEFAULT_CAPITAL,
    benchmark: str | None = None,
    weights: Mapping[str, float] | None = None,
) -> Portfolio:
    """Create a portfolio from a symbol list, resolving their metadata."""
    symbols = [str(symbol) for symbol in symbols]
    assets = instrument_lookup(symbols, market=market, data_mode=data_mode.value)
    if benchmark and benchmark not in assets:
        assets.update(instrument_lookup([benchmark], market=market, data_mode=data_mode.value))
    if weights is None:
        equal = 1.0 / len(symbols) if symbols else 0.0
        resolved = {symbol: equal for symbol in symbols}
    else:
        resolved = {str(symbol): float(weight) for symbol, weight in weights.items()}
    return Portfolio(
        weights=resolved,
        capital=capital,
        benchmark=benchmark,
        market=market,
        data_mode=data_mode,
        assets=assets,
    )


def _is_demo_symbol(symbol: str) -> bool:
    return symbol.startswith("DEMO-")


def reconcile_portfolio(
    portfolio: Portfolio | None,
    market: Market,
    data_mode: DataMode,
    *,
    morocco_universe=None,
    available_symbols: Iterable[str] | None = None,
) -> tuple[Portfolio, list[Notice]]:
    """Adapt a portfolio to a new (market, data mode) context.

    Rules
    -----
    * demo mode may only hold ``DEMO-*`` instruments — anything else is
      replaced by the default demo portfolio (a fabricated price for a real
      ticker would be misleading);
    * historical mode drops ``DEMO-*`` instruments the same way;
    * symbols that do not exist in the new market's universe are removed, and
      the remaining weights are renormalised. If nothing can be kept, the
      default portfolio for the target context is loaded.
    """
    notices: list[Notice] = []
    base = portfolio if portfolio is not None else default_portfolio(market, data_mode)
    effective_mode = DataMode(data_mode)
    effective_market = Market(market)

    if base.market is not effective_market or base.data_mode is not effective_mode:
        notices.append(info("data.portfolio_reconciled"))

    kept: dict[str, float] = {}
    universe = universe_for(effective_market, data_mode=effective_mode.value, morocco=morocco_universe)
    allowed = set(universe) | set(available_symbols or ())
    mismatched = [
        symbol
        for symbol in base.weights
        if (_is_demo_symbol(symbol) if effective_mode is DataMode.HISTORICAL else not _is_demo_symbol(symbol))
    ]
    if mismatched and effective_mode is DataMode.DEMO:
        notices.append(info("demo.assets_required", market=effective_market.value))
        return default_portfolio(effective_market, effective_mode, capital=base.capital), notices

    for symbol, weight in base.weights.items():
        if symbol in mismatched:
            continue
        if allowed and symbol not in allowed:
            continue
        kept[symbol] = float(weight)

    if not kept:
        fallback = default_portfolio(effective_market, effective_mode, capital=base.capital)
        if base.weights:
            notices.append(info("data.no_assets_for_market"))
        return fallback, notices

    total = sum(kept.values())
    if total <= 0:
        equal = 1.0 / len(kept)
        kept = {symbol: equal for symbol in kept}

    candidates = {*kept, *(available_symbols or ())}
    if base.benchmark:
        candidates.add(base.benchmark)
    assets = instrument_lookup(
        candidates,
        market=effective_market,
        data_mode=effective_mode.value,
        extra=base.assets,
        morocco=morocco_universe,
    )
    benchmark = base.benchmark
    valid_benchmarks = benchmarks_for(effective_market, data_mode=effective_mode.value, morocco=morocco_universe)
    if benchmark and valid_benchmarks and benchmark not in valid_benchmarks and benchmark not in (available_symbols or ()):
        benchmark = None
    if benchmark is None:
        available_benchmarks = list(valid_benchmarks)
        benchmark = available_benchmarks[0] if available_benchmarks else None

    reconciled = Portfolio(
        weights=kept,
        capital=base.capital,
        benchmark=benchmark,
        market=effective_market,
        data_mode=effective_mode,
        assets={symbol: instrument for symbol, instrument in assets.items() if symbol in candidates},
        name=base.name,
    )
    return reconciled, notices


def portfolio_universe_demo(market: Market) -> dict[str, Instrument]:
    """Convenience accessor used by tests."""
    return dict(DEMO_UNIVERSE_MOROCCO if market is Market.MOROCCO else DEMO_UNIVERSE_US)


def registered_universes() -> dict[str, dict[str, Instrument]]:
    """Diagnostics helper listing every known universe."""
    morocco = load_morocco_universe()
    return {
        "us": dict(US_UNIVERSE),
        "ma_user": dict(morocco.assets),
        "demo_us": dict(DEMO_UNIVERSE_US),
        "demo_ma": dict(DEMO_UNIVERSE_MOROCCO),
    }


__all__ += ["portfolio_universe_demo", "registered_universes"]
