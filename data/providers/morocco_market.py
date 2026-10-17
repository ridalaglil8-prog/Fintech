"""Moroccan market provider (Casablanca Stock Exchange).

Why this provider is CSV-first
------------------------------
Very few Casablanca Stock Exchange securities are reliably available through
free public APIs, so FinPilot AI **does not invent** any Moroccan ticker,
price or index. Instead it defines a small adapter interface:

* :class:`MoroccoCsvAdapter` — the working implementation: it consumes CSV
  files uploaded in the interface or referenced from the user's own universe
  file (``data/morocco_universe.json``);
* :class:`VerifiedSourceAdapter` — the documented extension point for a future
  licensed/verified data feed (implement ``fetch`` and register it in
  :class:`MoroccoMarketProvider`).

Until a verified feed is configured, the interface tells the user that real
Moroccan data are unavailable and offers the two supported alternatives:
demo mode or a CSV import.
"""

from __future__ import annotations

from dataclasses import field
from datetime import date
from pathlib import Path
from typing import Protocol, Sequence, runtime_checkable

import pandas as pd

from config import DataMode, Market
from data.csv_loader import LoadedPrices, combine_loaded_prices, load_price_csv, validate_price_frame
from data.providers.base import MarketDataProvider, PriceRequest, PriceResult
from notices import Notice, error, info, warning

__all__ = [
    "MoroccoDataAdapter",
    "MoroccoCsvAdapter",
    "VerifiedSourceAdapter",
    "MoroccoMarketProvider",
]


@runtime_checkable
class MoroccoDataAdapter(Protocol):
    """One possible source of Casablanca Stock Exchange price history."""

    key: str
    label_key: str

    def fetch(self, request: PriceRequest) -> PriceResult:  # pragma: no cover - protocol
        ...


class MoroccoCsvAdapter:
    """Read Moroccan price history from validated CSV files."""

    key = "ma_csv"
    label_key = "provider.ma_csv"

    def fetch(self, request: PriceRequest) -> PriceResult:
        base = PriceResult(
            requested_symbols=tuple(request.symbols),
            provider_key=self.key,
            source_key=self.key,
            source_label="csv",
        )
        payloads: list[tuple[str, object]] = [(name, raw) for name, raw in request.csv_payloads]
        for path in request.local_csv_files:
            file_path = Path(path)
            if not file_path.exists():
                base.notices.append(warning("ma.local_file_missing", file=file_path.name))
                continue
            payloads.append((file_path.name, file_path))
        if not payloads:
            return base.failed("ma.no_source")

        loaded: list[LoadedPrices] = []
        failures: list[Notice] = []
        for name, source in payloads:
            try:
                loaded.append(load_price_csv(source, default_currency="MAD", source_label=name))
            except Exception as exc:
                failures.append(
                    error("ma.file_rejected", file=name, reason=getattr(exc, "code", type(exc).__name__), detail=str(exc)[:200])
                )
        base.notices.extend(failures)
        if not loaded:
            base.notices.append(error("ma.no_valid_file"))
            return base

        try:
            merged = combine_loaded_prices(loaded)
        except Exception as exc:  # pragma: no cover - defensive
            base.notices.append(error("ma.merge_failed", detail=str(exc)[:200]))
            return base

        prices = merged.prices
        for index, dataset in enumerate(loaded):
            base.notices.append(
                info(
                    "ma.file_loaded",
                    file=merged.sources[index] if index < len(merged.sources) else "csv",
                    rows=int(len(dataset.prices)),
                    symbols=len(dataset.symbols),
                )
            )
        for issue in merged.issues:
            if issue.level == "error":
                base.notices.append(error(issue.key.split("notice.", 1)[-1], **issue.params))
            elif issue.level == "warning":
                base.notices.append(warning(issue.key.split("notice.", 1)[-1], **issue.params))

        # Restrict to the requested window and symbols -----------------------
        mask = (prices.index >= pd.Timestamp(request.start)) & (prices.index <= pd.Timestamp(request.end))
        windowed = prices.loc[mask]
        if windowed.empty:
            base.notices.append(
                error(
                    "ma.no_data_in_window",
                    start=request.start.isoformat(),
                    end=request.end.isoformat(),
                )
            )
            return base

        available = [symbol for symbol in request.symbols if symbol in windowed.columns]
        missing = [symbol for symbol in request.symbols if symbol not in windowed.columns]
        if missing:
            base.notices.append(warning("ma.symbols_unavailable", symbols=missing))
        if not available:
            base.notices.append(error("ma.no_requested_symbols", symbols=list(request.symbols)))
            return base

        benchmark_symbol = request.benchmark
        benchmark_series: pd.Series | None = None
        if benchmark_symbol and benchmark_symbol in windowed.columns:
            benchmark_series = windowed[benchmark_symbol].dropna().copy()
            benchmark_series.name = benchmark_symbol
            available = [symbol for symbol in available if symbol != benchmark_symbol]
        elif benchmark_symbol:
            base.notices.append(warning("ma.benchmark_unavailable", symbol=benchmark_symbol))

        if not available:
            base.notices.append(error("ma.no_requested_symbols", symbols=list(request.symbols)))
            return base

        asset_prices = windowed.loc[:, available]
        for column in asset_prices.columns:
            if asset_prices[column].notna().sum() < 2:
                base.notices.append(warning("ma.symbols_unavailable", symbols=[str(column)]))
        asset_prices = asset_prices.dropna(how="all")
        asset_prices = asset_prices.loc[:, asset_prices.notna().any()]

        currencies = {
            symbol: merged.currencies.get(symbol, "MAD") for symbol in asset_prices.columns
        }
        base.prices = asset_prices
        base.benchmark = benchmark_series
        base.benchmark_symbol = benchmark_symbol if benchmark_series is not None else None
        base.currency_map = currencies
        base.names = {symbol: merged.names.get(symbol, symbol) for symbol in asset_prices.columns}
        base.available_symbols = tuple(asset_prices.columns)
        base.missing_symbols = tuple(missing)
        base.notices.append(
            info(
                "ma.data_loaded",
                count=len(asset_prices.columns),
                start=str(asset_prices.index[0].date()),
                end=str(asset_prices.index[-1].date()),
                files=len(loaded),
            )
        )
        # Sanity review of the assembled frame (gaps, duplicates, coverage).
        for issue in validate_price_frame(asset_prices, expected_currency="MAD", currencies=currencies):
            if issue.level == "error":
                base.notices.append(error(issue.key.split("notice.", 1)[-1], **issue.params))
            elif issue.level == "warning":
                base.notices.append(warning(issue.key.split("notice.", 1)[-1], **issue.params))
        return base


class VerifiedSourceAdapter:
    """Extension point for a future verified/licensed Moroccan data feed.

    Register an implementation by passing it to
    :class:`MoroccoMarketProvider` (or by setting the ``FP_MOROCCO_ADAPTER``
    environment variable to an importable ``module:attribute`` path). Until a
    real feed is configured, this adapter reports that it is not configured
    instead of returning invented prices.
    """

    key = "ma_verified_feed"
    label_key = "provider.ma_verified_feed"

    def __init__(self, fetcher=None, *, label: str | None = None) -> None:
        self._fetcher = fetcher
        self.label = label or "verified feed"

    @property
    def configured(self) -> bool:
        return callable(self._fetcher)

    def fetch(self, request: PriceRequest) -> PriceResult:
        base = PriceResult(
            requested_symbols=tuple(request.symbols),
            provider_key=self.key,
            source_key=self.key,
            source_label=self.label,
        )
        if not self.configured:
            return base.failed("ma.verified_source_not_configured")
        try:  # pragma: no cover - only executed with a real feed
            result = self._fetcher(request)  # type: ignore[misc]
        except Exception as exc:
            return base.failed("ma.verified_source_failed", reason=type(exc).__name__, detail=str(exc)[:200])
        if not isinstance(result, PriceResult):  # pragma: no cover - defensive
            return base.failed("ma.verified_source_invalid")
        return result


class MoroccoMarketProvider(MarketDataProvider):
    """CSV-first Moroccan market provider with an extensible adapter list."""

    key = "ma_csv"
    label_key = "provider.ma_csv"
    market = Market.MOROCCO
    data_mode = DataMode.HISTORICAL
    is_simulated = False

    def __init__(
        self,
        adapters: Sequence[MoroccoDataAdapter] | None = None,
        *,
        local_csv_files: Sequence[str] | None = None,
        verified_fetcher=None,
    ) -> None:
        self.adapters: list[MoroccoDataAdapter] = list(
            adapters
            if adapters is not None
            else [MoroccoCsvAdapter(), VerifiedSourceAdapter(verified_fetcher)]
        )
        self.local_csv_files = tuple(local_csv_files or ())

    def fetch(self, request: PriceRequest) -> PriceResult:
        enriched = request
        if self.local_csv_files and not request.local_csv_files:
            enriched = PriceRequest(
                symbols=request.symbols,
                start=request.start,
                end=request.end,
                market=request.market,
                benchmark=request.benchmark,
                csv_payloads=request.csv_payloads,
                local_csv_files=self.local_csv_files,
                seed=request.seed,
            )
        attempts: list[PriceResult] = []
        for adapter in self.adapters:
            result = adapter.fetch(enriched)
            attempts.append(result)
            if result.ok:
                result.provider_key = self.key
                return result

        # Nothing worked: return the most informative attempt, keeping every
        # error so the user knows exactly what to fix (upload a CSV, configure a
        # feed, or switch to demo mode).
        merged = attempts[-1]
        all_errors: list[Notice] = []
        for attempt in attempts:
            all_errors.extend(attempt.errors)
        merged.notices = [*merged.notices, *all_errors]
        merged.provider_key = self.key
        merged.source_label = "csv"
        return merged
