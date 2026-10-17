"""CSV price-history import with strict validation.

This module is the **only** supported way to bring Moroccan (or any external)
market data into FinPilot AI, because the application must never invent or
guess a Casablanca Stock Exchange price.

Accepted layouts (auto-detected)
--------------------------------
1. **Long format** (one row per observation)::

       date,symbol,close[,currency][,name]
       2024-01-02,ATW,412.50,MAD

2. **Wide format** (one column per instrument)::

       date,SYMBOL_A,SYMBOL_B
       2024-01-02,412.50,88.30

Robustness rules
----------------
* delimiter auto-detection (``,`` ``;`` ``\\t`` ``|``) and encoding fallback
  (UTF-8 with BOM, UTF-8, Latin-1) — Casablanca exports are frequently
  semicolon-separated and Latin-1 encoded;
* dates parsed as ISO or day-first (``31/12/2024``);
* prices must be numeric and **strictly positive** (a zero or negative price is
  a data error, not a missing value);
* duplicated ``(date, symbol)`` rows keep the last occurrence and raise a
  warning;
* symbols with too little history are excluded with a warning;
* the file size and row count are capped to protect the server.

Nothing is ever forward-filled or interpolated: gaps stay ``NaN`` and are
reported through the coverage diagnostics.
"""

from __future__ import annotations

import csv
import io
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import IO, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from analytics.exceptions import DataValidationError, InvalidInputError
from config import MIN_OBSERVATIONS_FOR_METRICS

__all__ = [
    "CsvIssue",
    "LoadedPrices",
    "detect_delimiter",
    "read_text_with_fallback",
    "load_price_csv",
    "validate_price_frame",
    "MAX_CSV_BYTES",
    "MAX_CSV_ROWS",
]

MAX_CSV_BYTES = 25 * 1024 * 1024  # 25 MB (aligned with Streamlit's upload limit)
MAX_CSV_ROWS = 500_000

DATE_ALIASES = ("date", "datetime", "time", "timestamp", "jour", "date_", "seance", "séance")
SYMBOL_ALIASES = ("symbol", "ticker", "code", "isin", "valeur", "instrument", "asset")
PRICE_ALIASES = ("close", "closing", "cloture", "clôture", "cours", "cours_cloture", "price", "adj close", "adj_close", "value", "last", "cours de clôture")
CURRENCY_ALIASES = ("currency", "devise", "ccy")
NAME_ALIASES = ("name", "label", "nom", "libelle", "libellé", "description")


@dataclass(frozen=True)
class CsvIssue:
    """A validation problem found while reading a CSV file."""

    level: str  #: "error" | "warning" | "info"
    code: str
    params: dict = field(default_factory=dict)

    @property
    def key(self) -> str:
        return f"notice.csv.{self.code}"


@dataclass(frozen=True)
class LoadedPrices:
    """Validated price history extracted from one or more CSV files."""

    prices: pd.DataFrame
    currencies: dict[str, str]
    names: dict[str, str]
    sources: tuple[str, ...]
    issues: tuple[CsvIssue, ...] = ()

    @property
    def symbols(self) -> list[str]:
        return list(self.prices.columns)

    @property
    def errors(self) -> list[CsvIssue]:
        return [issue for issue in self.issues if issue.level == "error"]

    @property
    def warnings(self) -> list[CsvIssue]:
        return [issue for issue in self.issues if issue.level == "warning"]

    @property
    def last_observation(self) -> pd.Timestamp | None:
        return self.prices.index[-1] if not self.prices.empty else None

    @property
    def first_observation(self) -> pd.Timestamp | None:
        return self.prices.index[0] if not self.prices.empty else None

    def with_prices(self, prices: pd.DataFrame) -> "LoadedPrices":
        return LoadedPrices(
            prices=prices,
            currencies=self.currencies,
            names=self.names,
            sources=self.sources,
            issues=self.issues,
        )


# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------
def read_text_with_fallback(raw: bytes) -> tuple[str, str]:
    """Decode bytes trying UTF-8 (with/without BOM) then Latin-1.

    Returns ``(text, encoding)``.
    """
    for encoding in ("utf-8-sig", "utf-8", "latin-1"):
        try:
            return raw.decode(encoding), encoding
        except UnicodeDecodeError:
            continue
    raise DataValidationError(
        "The file could not be decoded as text.",
        code="undecodable_file",
    )


def detect_delimiter(sample: str, *, default: str = ",") -> str:
    """Guess the column delimiter of a CSV sample."""
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=",;\t|")
        return dialect.delimiter
    except csv.Error:
        counts = {delimiter: sample.count(delimiter) for delimiter in (",", ";", "\t", "|")}
        best = max(counts, key=counts.get)
        return best if counts[best] > 0 else default


def _normalise(text: str) -> str:
    return re.sub(r"\s+", " ", str(text).strip().lower())


def _match_column(columns: Sequence[str], aliases: Iterable[str]) -> str | None:
    lookup = {_normalise(column): column for column in columns}
    for alias in aliases:
        if alias in lookup:
            return lookup[alias]
    for column in columns:
        for alias in aliases:
            if alias and alias in _normalise(column):
                return column
    return None


def _read_bytes(source: str | Path | bytes | IO[bytes] | IO[str]) -> tuple[bytes, str]:
    """Read a source and return ``(bytes, label)``."""
    if isinstance(source, bytes):
        return source, "bytes"
    if isinstance(source, (str, Path)):
        path = Path(source)
        if not path.exists():
            raise DataValidationError(
                f"File not found: {path.name}",
                code="file_not_found",
                params={"file": path.name},
            )
        size = path.stat().st_size
        if size > MAX_CSV_BYTES:
            raise DataValidationError(
                f"File too large ({size / 1e6:.1f} MB), limit is {MAX_CSV_BYTES / 1e6:.0f} MB.",
                code="file_too_large",
                params={"size_mb": round(size / 1e6, 1), "limit_mb": MAX_CSV_BYTES // (1024 * 1024)},
            )
        return path.read_bytes(), path.name
    # file-like (e.g. streamlit UploadedFile)
    if hasattr(source, "read"):
        payload = source.read()  # type: ignore[union-attr]
        if isinstance(payload, str):
            payload = payload.encode("utf-8")
        label = getattr(source, "name", "uploaded.csv")
        if len(payload) > MAX_CSV_BYTES:
            raise DataValidationError(
                f"File too large ({len(payload) / 1e6:.1f} MB).",
                code="file_too_large",
                params={"size_mb": round(len(payload) / 1e6, 1), "limit_mb": MAX_CSV_BYTES // (1024 * 1024)},
            )
        try:  # allow re-reading the same upload in the same session
            source.seek(0)
        except Exception:  # pragma: no cover - not seekable
            pass
        return payload, str(label)
    raise DataValidationError(
        "Unsupported CSV source (expected bytes, a path or a file-like object).",
        code="unsupported_source",
    )


def _parse_dates(series: pd.Series) -> pd.Series:
    """Parse dates as ISO first, then day-first (31/12/2024)."""
    parsed = pd.to_datetime(series, errors="coerce", format="ISO8601")
    missing = parsed.isna()
    if missing.any():
        fallback = pd.to_datetime(series[missing], errors="coerce", dayfirst=True)
        parsed.loc[missing] = fallback
    return parsed


def _to_numeric_prices(series: pd.Series) -> pd.Series:
    """Convert a price column to floats, tolerating thousands separators."""
    cleaned = (
        series.astype("string")
        .str.strip()
        .str.replace("\u00a0", "", regex=False)
        .str.replace(" ", "", regex=False)
        .str.replace(",", ".", regex=False)
    )
    return pd.to_numeric(cleaned, errors="coerce")


# ---------------------------------------------------------------------------
# Main entry points
# ---------------------------------------------------------------------------
def load_price_csv(
    source: str | Path | bytes | IO[bytes],
    *,
    default_currency: str = "MAD",
    source_label: str | None = None,
    min_observations: int = MIN_OBSERVATIONS_FOR_METRICS,
) -> LoadedPrices:
    """Load and validate a price CSV file.

    Raises :class:`~analytics.exceptions.DataValidationError` when the file is
    unusable (not decodable, too large, no date column, no valid price column,
    only non-positive prices). Recoverable problems (dropped rows, duplicates,
    short history, missing currency) are returned as warnings.
    """
    raw, label = _read_bytes(source)
    label = source_label or label
    if not raw.strip():
        raise DataValidationError("The file is empty.", code="empty_file", params={"file": label})

    text, encoding = read_text_with_fallback(raw)
    delimiter = detect_delimiter(text[:8192])
    try:
        frame = pd.read_csv(
            io.StringIO(text),
            sep=delimiter,
            dtype=str,
            keep_default_na=False,
            na_filter=False,
        )
    except Exception as exc:  # pragma: no cover - pandas raises many types
        raise DataValidationError(
            f"The CSV file could not be parsed: {exc}",
            code="unparsable_csv",
            params={"file": label},
        ) from exc

    if frame.empty:
        raise DataValidationError("The file contains no data rows.", code="no_rows", params={"file": label})
    if len(frame) > MAX_CSV_ROWS:
        raise DataValidationError(
            f"The file contains {len(frame):,} rows, above the {MAX_CSV_ROWS:,} row limit.",
            code="too_many_rows",
            params={"rows": len(frame), "limit": MAX_CSV_ROWS},
        )

    issues: list[CsvIssue] = [
        CsvIssue("info", "file_loaded", {"file": label, "rows": len(frame), "delimiter": delimiter, "encoding": encoding})
    ]
    columns = list(frame.columns)
    date_column = _match_column(columns, DATE_ALIASES)
    if date_column is None:
        raise DataValidationError(
            "No date column found (expected 'date', 'jour', 'datetime', ...).",
            code="missing_date_column",
            params={"columns": columns},
        )

    symbol_column = _match_column(columns, SYMBOL_ALIASES)
    price_column = _match_column(columns, PRICE_ALIASES)
    if symbol_column is not None and price_column is not None:
        prices, currencies, names, long_issues = _parse_long_format(
            frame, date_column, symbol_column, price_column, columns, default_currency, label
        )
        issues.extend(long_issues)
    else:
        prices, currencies, names, wide_issues = _parse_wide_format(
            frame, date_column, columns, default_currency, label
        )
        issues.extend(wide_issues)

    prices, post_issues = _finalise_prices(prices, currencies, min_observations=min_observations, label=label)
    issues.extend(post_issues)

    return LoadedPrices(
        prices=prices,
        currencies=currencies,
        names=names,
        sources=(label,),
        issues=tuple(issues),
    )


def _parse_long_format(
    frame: pd.DataFrame,
    date_column: str,
    symbol_column: str,
    price_column: str,
    columns: Sequence[str],
    default_currency: str,
    label: str,
) -> tuple[pd.DataFrame, dict[str, str], dict[str, str], list[CsvIssue]]:
    issues: list[CsvIssue] = []
    dates = _parse_dates(frame[date_column])
    invalid_dates = int(dates.isna().sum())
    if invalid_dates:
        issues.append(CsvIssue("warning", "invalid_dates_dropped", {"count": invalid_dates, "file": label}))
    prices = _to_numeric_prices(frame[price_column])
    invalid_prices = prices.isna()
    if invalid_prices.any():
        issues.append(
            CsvIssue(
                "warning",
                "non_numeric_prices_dropped",
                {"count": int(invalid_prices.sum()), "file": label},
            )
        )
    working = pd.DataFrame(
        {
            "date": dates,
            "symbol": frame[symbol_column].astype("string").str.strip(),
            "price": prices,
        }
    ).dropna(subset=["date", "price"])
    working = working[working["symbol"].notna() & (working["symbol"] != "")]

    currency_column = _match_column(columns, CURRENCY_ALIASES)
    currencies: dict[str, str] = {}
    if currency_column is not None:
        currency_map = (
            pd.DataFrame({"symbol": frame[symbol_column].astype("string").str.strip(), "ccy": frame[currency_column].astype("string").str.upper().str.strip()})
            .dropna()
            .drop_duplicates(subset=["symbol"])
        )
        currencies = {
            str(row.symbol): (str(row.ccy) if row.ccy else default_currency)
            for row in currency_map.itertuples(index=False)
        }
    name_column = _match_column(columns, NAME_ALIASES)
    names: dict[str, str] = {}
    if name_column is not None:
        name_map = (
            pd.DataFrame({"symbol": frame[symbol_column].astype("string").str.strip(), "name": frame[name_column].astype("string").str.strip()})
            .dropna()
            .drop_duplicates(subset=["symbol"])
        )
        names = {str(row.symbol): str(row.name) for row in name_map.itertuples(index=False)}
    else:
        issues.append(CsvIssue("info", "long_format_without_names", {"file": label}))

    duplicates = int(working.duplicated(subset=["date", "symbol"]).sum())
    if duplicates:
        issues.append(CsvIssue("warning", "duplicate_rows_merged", {"count": duplicates, "file": label}))
        working = working.sort_values("date").drop_duplicates(subset=["date", "symbol"], keep="last")

    wide = working.pivot(index="date", columns="symbol", values="price")
    wide.index.name = "date"
    return wide, currencies, names, issues


def _parse_wide_format(
    frame: pd.DataFrame,
    date_column: str,
    columns: Sequence[str],
    default_currency: str,
    label: str,
) -> tuple[pd.DataFrame, dict[str, str], dict[str, str], list[CsvIssue]]:
    issues: list[CsvIssue] = []
    dates = _parse_dates(frame[date_column])
    invalid_dates = int(dates.isna().sum())
    if invalid_dates:
        issues.append(CsvIssue("warning", "invalid_dates_dropped", {"count": invalid_dates, "file": label}))

    data_columns = [column for column in columns if column != date_column]
    numeric_columns: list[str] = []
    dropped: list[str] = []
    for column in data_columns:
        values = _to_numeric_prices(frame[column])
        non_empty = frame[column].astype("string").str.strip() != ""
        if non_empty.any() and values[non_empty].notna().mean() >= 0.5:
            frame[column] = values
            numeric_columns.append(column)
        else:
            dropped.append(column)
    if dropped:
        issues.append(CsvIssue("warning", "non_numeric_columns_ignored", {"columns": dropped, "file": label}))
    if not numeric_columns:
        raise DataValidationError(
            "No numeric price column found in the file.",
            code="no_price_column",
            params={"columns": list(data_columns), "file": label},
        )

    working = frame.loc[:, [date_column, *numeric_columns]].copy()
    working[date_column] = dates
    working = working.dropna(subset=[date_column])
    wide = working.set_index(date_column).astype("float64")
    wide.index.name = "date"
    wide = wide.sort_index()
    duplicates = int(wide.index.duplicated().sum())
    if duplicates:
        issues.append(CsvIssue("warning", "duplicate_dates_merged", {"count": duplicates, "file": label}))
        wide = wide[~wide.index.duplicated(keep="last")]
    currencies = {symbol: (default_currency or "MAD").upper() for symbol in wide.columns}
    return wide, currencies, {}, issues


def _finalise_prices(
    prices: pd.DataFrame,
    currencies: dict[str, str],
    *,
    min_observations: int,
    label: str,
) -> tuple[pd.DataFrame, list[CsvIssue]]:
    issues: list[CsvIssue] = []
    if prices is None or prices.empty:
        raise DataValidationError(
            "No usable price observations were found in the file.",
            code="no_valid_observations",
            params={"file": label},
        )
    prices = prices.sort_index()
    if not isinstance(prices.index, pd.DatetimeIndex):
        raise DataValidationError("Price index is not made of dates.", code="invalid_date_index", params={"file": label})

    days = prices.index.normalize() if getattr(prices.index, "tz", None) is None else prices.index.tz_localize(None).normalize()
    prices.index = pd.DatetimeIndex(days)

    duplicated = int(prices.index.duplicated().sum())
    if duplicated:
        issues.append(CsvIssue("warning", "duplicate_dates_merged", {"count": duplicated, "file": label}))
        prices = prices[~prices.index.duplicated(keep="last")]

    today = pd.Timestamp.now().normalize()
    future = prices.index > today
    if future.any():
        issues.append(
            CsvIssue("warning", "future_dates_dropped", {"count": int(future.sum()), "file": label})
        )
        prices = prices.loc[~future]

    negative = (prices <= 0).to_numpy().any(axis=0)
    if negative.any():
        bad = list(prices.columns[negative])
        # A zero/negative price is a data error: it is only dropped when other
        # valid observations exist for the symbol, otherwise it is fatal.
        remaining = prices.loc[:, bad].where(prices.loc[:, bad] > 0)
        keep = [column for column in bad if remaining[column].notna().sum() >= min_observations]
        if len(keep) < len(bad):
            fatal = [column for column in bad if column not in keep]
            raise DataValidationError(
                f"Non-positive prices found for {fatal}.",
                code="non_positive_prices",
                params={"symbols": fatal, "file": label},
            )
        issues.append(CsvIssue("warning", "non_positive_prices_dropped", {"symbols": bad, "file": label}))
        prices.loc[:, bad] = remaining

    prices = prices.replace([np.inf, -np.inf], np.nan)

    coverage = prices.notna().sum()
    short = [str(symbol) for symbol, count in coverage.items() if count < min_observations]
    if short:
        issues.append(
            CsvIssue(
                "warning",
                "symbols_with_insufficient_history",
                {"symbols": short, "minimum": min_observations, "file": label},
            )
        )
        prices = prices.drop(columns=short)
    if prices.empty or prices.shape[1] == 0:
        raise DataValidationError(
            "No symbol has enough observations after validation.",
            code="no_valid_symbols",
            params={"minimum": min_observations, "file": label},
        )

    empty_rows = prices.isna().all(axis=1)
    if empty_rows.any():
        prices = prices.loc[~empty_rows]
    if len(prices) < min_observations:
        raise DataValidationError(
            f"Only {len(prices)} usable observations, {min_observations} required.",
            code="insufficient_observations",
            params={"available": int(len(prices)), "required": min_observations, "file": label},
        )
    return prices, issues


# ---------------------------------------------------------------------------
# Standalone validation (used by tests and by the market-data service)
# ---------------------------------------------------------------------------
def validate_price_frame(
    prices: pd.DataFrame,
    *,
    min_observations: int = MIN_OBSERVATIONS_FOR_METRICS,
    expected_currency: str | None = None,
    currencies: Mapping[str, str] | None = None,
) -> list[CsvIssue]:
    """Review a price frame and return the list of problems found.

    This function never mutates or fixes the frame: it only reports, so the
    caller decides what to do (mirrors :func:`load_price_csv`'s warnings).
    """
    issues: list[CsvIssue] = []
    if prices is None or prices.empty:
        return [CsvIssue("error", "empty_frame", {})]
    if not isinstance(prices.index, pd.DatetimeIndex):
        issues.append(CsvIssue("error", "invalid_date_index", {}))
    else:
        if prices.index.duplicated().any():
            issues.append(CsvIssue("warning", "duplicate_dates_merged", {"count": int(prices.index.duplicated().sum())}))
        if not prices.index.is_monotonic_increasing:
            issues.append(CsvIssue("warning", "unsorted_dates", {}))
    if prices.isna().all(axis=1).any():
        issues.append(CsvIssue("info", "rows_without_any_price", {"count": int(prices.isna().all(axis=1).sum())}))

    for column in prices.columns:
        series = pd.to_numeric(prices[column], errors="coerce")
        if series.notna().sum() < min_observations:
            issues.append(
                CsvIssue("warning", "symbols_with_insufficient_history", {"symbols": [str(column)], "minimum": min_observations})
            )
        if (series.dropna() <= 0).any():
            issues.append(CsvIssue("error", "non_positive_prices", {"symbols": [str(column)]}))
        if series.notna().sum() and series.isna().any():
            issues.append(
                CsvIssue("info", "symbol_has_gaps", {"symbol": str(column), "missing": int(series.isna().sum())})
            )
    if currencies and expected_currency:
        mismatched = {symbol: code for symbol, code in currencies.items() if code != expected_currency}
        if mismatched:
            issues.append(CsvIssue("info", "currency_mismatch", {"expected": expected_currency, "found": mismatched}))
    return issues


def combine_loaded_prices(datasets: Sequence[LoadedPrices], *, how: str = "outer") -> LoadedPrices:
    """Merge several validated CSV imports into one price history."""
    usable = [dataset for dataset in datasets if dataset is not None and not dataset.prices.empty]
    if not usable:
        raise InvalidInputError("No CSV data to merge.", code="no_csv_data")
    frame = pd.concat([dataset.prices for dataset in usable], axis=1, join=how)
    frame = frame.sort_index()
    # Duplicate columns can appear when the same symbol is exported twice: keep
    # the most recent (last) non-missing observation.
    if frame.columns.duplicated().any():
        frame = frame.T.groupby(level=0).last().T.sort_index()
    currencies: dict[str, str] = {}
    names: dict[str, str] = {}
    issues: list[CsvIssue] = []
    sources: list[str] = []
    for dataset in usable:
        currencies.update(dataset.currencies)
        names.update(dataset.names)
        issues.extend(dataset.issues)
        sources.extend(dataset.sources)
    return LoadedPrices(
        prices=frame,
        currencies=currencies,
        names=names,
        sources=tuple(sources),
        issues=tuple(issues),
    )
