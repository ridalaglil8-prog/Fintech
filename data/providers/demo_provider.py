"""Demo provider — reproducible simulated prices, no network access.

This provider exists so the whole application (dashboard, analytics,
optimisation, stress testing, AI context) works offline and without any API
key. Every result is flagged ``is_simulated=True`` and every page displays the
label, so a simulated price can never be mistaken for a market observation.
"""

from __future__ import annotations

import pandas as pd

from config import DEMO_SEED, DataMode, Market
from data.demo_data import demo_benchmark, generate_demo_prices
from data.providers.base import MarketDataProvider, PriceRequest, PriceResult
from notices import info, warning

__all__ = ["DemoMarketProvider"]


class DemoMarketProvider(MarketDataProvider):
    """Generate deterministic synthetic series for fictional ``DEMO-*`` symbols."""

    label_key = "provider.demo"
    data_mode = DataMode.DEMO
    is_simulated = True

    def __init__(self, market: Market = Market.US, *, seed: int | None = None) -> None:
        self.market = market
        self.seed = DEMO_SEED if seed is None else int(seed)
        self.key = f"demo_{market.value.lower()}"

    def fetch(self, request: PriceRequest) -> PriceResult:
        seed = self.seed if request.seed is None else int(request.seed)
        benchmark_symbol = request.benchmark or demo_benchmark(self.market).symbol
        symbols = list(dict.fromkeys([*request.symbols, benchmark_symbol]))
        notices = [
            info("demo.simulated_data", seed=seed, market=self.market.value),
            info("demo.business_days_only"),
        ]
        try:
            frame = generate_demo_prices(
                symbols, request.start, request.end, market=self.market, seed=seed
            )
        except Exception as exc:  # pragma: no cover - defensive
            return PriceResult(
                requested_symbols=tuple(request.symbols),
                provider_key=self.key,
                source_key=self.key,
                source_label="demo",
                is_simulated=True,
                notices=[*notices, *[self._error(exc)]],
            )

        benchmark_series: pd.Series | None = None
        if benchmark_symbol in frame.columns:
            benchmark_series = frame[benchmark_symbol].copy()
            benchmark_series.name = benchmark_symbol
            frame = frame.drop(columns=[benchmark_symbol])
        else:  # pragma: no cover - defensive
            notices.append(warning("demo.benchmark_missing", symbol=benchmark_symbol))

        missing = [symbol for symbol in request.symbols if symbol not in frame.columns]
        if missing:
            notices.append(warning("demo.unknown_symbols", symbols=missing))

        requested = tuple(symbol for symbol in request.symbols if symbol in frame.columns)
        currency = self.market.currency
        return PriceResult(
            prices=frame,
            benchmark=benchmark_series,
            benchmark_symbol=benchmark_symbol if benchmark_series is not None else None,
            currency_map={symbol: currency for symbol in frame.columns},
            names={symbol: symbol for symbol in frame.columns},
            source_key=self.key,
            source_label="demo",
            is_simulated=True,
            requested_symbols=tuple(request.symbols),
            available_symbols=tuple(requested),
            missing_symbols=tuple(missing),
            notices=notices,
            provider_key=self.key,
        )

    @staticmethod
    def _error(exc: Exception):
        from notices import error as _error

        return _error("demo.generation_failed", reason=type(exc).__name__, detail=str(exc))
