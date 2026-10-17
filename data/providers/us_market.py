"""US market provider backed by yfinance.

Scope and limitations
---------------------
* Only **available** US-listed history is requested through yfinance; a symbol
  that returns nothing is reported as unavailable, never replaced by a
  simulated series.
* Network/SSL/rate-limit failures are converted into a friendly error notice
  (and the full exception is kept in the log for local development).
* Prices are adjusted for splits and dividends (``auto_adjust=True``) and must
  be strictly positive.
"""

from __future__ import annotations

import logging
from datetime import timedelta

import numpy as np
import pandas as pd

from config import (
    MARKET_DATA_MAX_RETRIES,
    MARKET_DATA_TIMEOUT_SECONDS,
    DataMode,
    Market,
)
from data.providers.base import MarketDataProvider, PriceRequest, PriceResult
from notices import info, warning

try:  # yfinance is optional: the app must start even if it is missing.
    import yfinance as yf

    YFINANCE_AVAILABLE = True
except Exception:  # pragma: no cover - import guard
    yf = None  # type: ignore[assignment]
    YFINANCE_AVAILABLE = False

logger = logging.getLogger(__name__)

__all__ = ["UsMarketProvider", "YFINANCE_AVAILABLE", "US_BENCHMARK_FALLBACKS"]

#: Benchmarks tried in order when the primary index (S&P 500) is unavailable.
US_BENCHMARK_FALLBACKS: tuple[str, ...] = ("^GSPC", "SPY", "^IXIC", "^DJI")

_CLOSE_FIELDS = ("Close", "Adj Close", "close", "adj close")


class UsMarketProvider(MarketDataProvider):
    """Fetch US historical prices from yfinance."""

    key = "us_yfinance"
    label_key = "provider.us_yfinance"
    market = Market.US
    data_mode = DataMode.HISTORICAL
    is_simulated = False

    def __init__(
        self,
        *,
        timeout: float = MARKET_DATA_TIMEOUT_SECONDS,
        max_retries: int = MARKET_DATA_MAX_RETRIES,
        auto_adjust: bool = True,
    ) -> None:
        self.timeout = float(timeout)
        self.max_retries = max(0, int(max_retries))
        self.auto_adjust = bool(auto_adjust)

    # ------------------------------------------------------------------
    def fetch(self, request: PriceRequest) -> PriceResult:
        benchmark = request.benchmark or "^GSPC"
        symbols = list(dict.fromkeys([*request.symbols, benchmark]))
        result = PriceResult(
            requested_symbols=tuple(request.symbols),
            provider_key=self.key,
            source_key=self.key,
            source_label="yfinance",
            currency_map={symbol: "USD" for symbol in symbols},
        )
        if not YFINANCE_AVAILABLE:
            return result.failed("us.library_missing")

        raw, failure = self._download(symbols, request)
        if failure is not None:
            return result.notices.append(failure) or result

        if raw is None or raw.empty:
            return result.failed("us.no_data", symbols=request.symbols)

        close = self._extract_close(raw)
        if close is None or close.empty:
            return result.failed("us.no_close_field", symbols=symbols)

        close = close.sort_index()
        close.index = pd.DatetimeIndex(close.index).tz_localize(None) if getattr(close.index, "tz", None) is not None else pd.DatetimeIndex(close.index)
        close = close[~close.index.duplicated(keep="last")]
        close = close.dropna(how="all")
        close = close.loc[:, [column for column in close.columns if close[column].notna().any()]]

        non_positive = [str(column) for column in close.columns if (close[column].dropna() <= 0).any()]
        if non_positive:
            close.loc[:, non_positive] = close.loc[:, non_positive].where(close.loc[:, non_positive] > 0)
            result.notices.append(warning("us.non_positive_prices", symbols=non_positive))

        if close.empty:
            return result.failed("us.no_data", symbols=request.symbols)

        benchmark_series: pd.Series | None = None
        if benchmark in close.columns:
            benchmark_series = close[benchmark].dropna().copy()
            benchmark_series.name = benchmark
            close = close.drop(columns=[benchmark])
        else:
            result.notices.append(warning("us.benchmark_unavailable", symbol=benchmark))

        missing = [symbol for symbol in request.symbols if symbol not in close.columns]
        if missing:
            result.notices.append(warning("us.symbols_unavailable", symbols=missing))

        available = [symbol for symbol in request.symbols if symbol in close.columns]
        if not available:
            return result.failed("us.no_requested_symbols")

        result.prices = close.loc[:, available]
        result.benchmark = benchmark_series if benchmark_series is not None and not benchmark_series.empty else None
        result.benchmark_symbol = benchmark if result.benchmark is not None else None
        result.available_symbols = tuple(available)
        result.missing_symbols = tuple(missing)
        result.currency_map = {symbol: "USD" for symbol in available}

        dropped_rows = int(len(close) - close.loc[:, available].dropna(how="all").shape[0])
        if dropped_rows > 0:
            result.notices.append(info("us.empty_rows_ignored", count=dropped_rows))
        result.notices.append(
            info(
                "us.data_loaded",
                count=len(available),
                start=str(close.index[0].date()) if len(close) else "—",
                end=str(close.index[-1].date()) if len(close) else "—",
                adjusted=self.auto_adjust,
            )
        )
        return result

    # ------------------------------------------------------------------
    def _download(
        self, symbols: list[str], request: PriceRequest
    ) -> tuple[pd.DataFrame | None, object | None]:
        """Download raw data with retries; returns ``(frame, failure_notice)``."""
        last_exception: Exception | None = None
        end = request.end + timedelta(days=1)  # yfinance's end date is exclusive
        for attempt in range(self.max_retries + 1):
            try:
                raw = yf.download(
                    tickers=symbols,
                    start=request.start.isoformat(),
                    end=end.isoformat(),
                    auto_adjust=self.auto_adjust,
                    progress=False,
                    threads=False,
                    timeout=self.timeout,
                    group_by="column",
                )
                if raw is not None and not raw.empty:
                    return raw, None
                last_exception = None
            except Exception as exc:  # network, SSL, rate limits, parsing...
                last_exception = exc
                logger.warning("yfinance attempt %s failed: %s", attempt + 1, exc)
        from notices import error as _error

        if last_exception is not None:
            return None, _error(
                "us.request_failed",
                reason=type(last_exception).__name__,
                detail=str(last_exception)[:300],
            )
        return None, None

    # ------------------------------------------------------------------
    @staticmethod
    def _extract_close(raw: pd.DataFrame) -> pd.DataFrame | None:
        """Extract the close-price frame from yfinance's many shapes."""
        if isinstance(raw.columns, pd.MultiIndex):
            level0 = raw.columns.get_level_values(0)
            for field in _CLOSE_FIELDS:
                if field in level0:
                    frame = raw.xs(field, axis=1, level=0)
                    return frame.astype("float64")
            return None
        for field in _CLOSE_FIELDS:
            if field in raw.columns:
                # Single ticker: one column of prices.
                series = raw[field]
                if isinstance(series, pd.DataFrame):  # pragmatic safety
                    series = series.iloc[:, 0]
                frame = series.to_frame(name=str(raw.attrs.get("ticker", series.name or "SYMBOL")))
                return frame.astype("float64")
        # Last resort: a single-column frame of floats (rare yfinance shapes).
        numeric = raw.select_dtypes(include=[np.number])
        if numeric.shape[1] == 1:
            numeric.columns = ["SYMBOL"]
            return numeric.astype("float64")
        return None
