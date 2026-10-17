"""Currency handling: formatting, per-currency totals and explicit conversion.

Policy
------
US instruments are quoted in USD, Moroccan instruments in MAD. FinPilot AI
**never guesses an exchange rate**:

* when every holding shares one currency, a single total is reported;
* with several currencies, allocation values are reported **per currency**, and
  no combined total is produced unless the user (or the environment) declares an
  explicit rate, which is then displayed with its value, date and source.

This module is framework-free so the rules can be unit-tested.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
import pandas as pd

from analytics.exceptions import InvalidInputError
from analytics.portfolio import allocation_values, split_currency_totals
from config import FxRate, load_env_fx_rate

__all__ = [
    "CURRENCY_SYMBOLS",
    "CurrencyTotals",
    "format_currency",
    "format_number",
    "format_percent",
    "format_quantity",
    "allocation_by_currency",
    "is_single_currency",
    "combine_totals",
    "get_configured_fx_rate",
    "currency_label",
]

CURRENCY_SYMBOLS: dict[str, str] = {
    "USD": "$",
    "MAD": "DH",
    "EUR": "€",
    "GBP": "£",
    "CHF": "CHF",
    "JPY": "¥",
}


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------
def _group_digits(value: float, decimals: int, language: str) -> str:
    """Format a number with the conventions of the selected language."""
    text = f"{abs(value):,.{decimals}f}"
    if language == "fr":
        # 1 234 567,89  (narrow no-break space + comma)
        text = text.replace(",", "\u202f").replace(".", ",")
    return text


def format_number(value: float | None, language: str = "en", decimals: int = 2) -> str:
    """Format a plain number (``—`` for missing/NaN values)."""
    if value is None:
        return "—"
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return "—"
    if not np.isfinite(numeric):
        return "—"
    text = _group_digits(numeric, decimals, language)
    return f"-{text}" if numeric < 0 else text


def format_quantity(value: float | None, language: str = "en") -> str:
    """Format a count (no decimals)."""
    return format_number(value, language, decimals=0)


def format_percent(
    value: float | None,
    language: str = "en",
    decimals: int = 2,
    *,
    signed: bool = False,
    from_ratio: bool = True,
) -> str:
    """Format a ratio (``0.1234`` -> ``12.34%``) or a percentage value."""
    if value is None:
        return "—"
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return "—"
    if not np.isfinite(numeric):
        return "—"
    if from_ratio:
        numeric *= 100.0
    text = _group_digits(numeric, decimals, language)
    sign = ""
    if numeric < 0:
        sign = "-"
        text = text.lstrip("-")
    elif signed and numeric > 0:
        sign = "+"
    separator = "\u202f" if language == "fr" else ""
    return f"{sign}{text}{separator}%"


def format_currency(
    value: float | None,
    currency: str = "USD",
    language: str = "en",
    *,
    decimals: int = 2,
    compact: bool = False,
) -> str:
    """Format a monetary amount with its currency code or symbol."""
    if value is None:
        return "—"
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return "—"
    if not np.isfinite(numeric):
        return "—"
    code = (currency or "USD").upper()
    symbol = CURRENCY_SYMBOLS.get(code)
    if compact and abs(numeric) >= 1_000_000:
        scaled = numeric / 1_000_000
        body = f"{_group_digits(scaled, max(0, decimals - 1), language)}M"
    elif compact and abs(numeric) >= 10_000:
        scaled = numeric / 1000
        body = f"{_group_digits(scaled, max(0, decimals - 1), language)}K"
    else:
        body = _group_digits(numeric, decimals, language)
    sign = "-" if numeric < 0 else ""
    body = body.lstrip("-")
    if language == "fr":
        return f"{sign}{body} {symbol or code}"
    return f"{sign}{symbol or ''}{body} {code}".replace("  ", " ").strip()


# ---------------------------------------------------------------------------
# Totals per currency
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class CurrencyTotals:
    """Allocation values grouped by currency, plus combination diagnostics."""

    totals: dict[str, float]
    mixed: bool
    combined_total: float | None
    combined_currency: str | None
    fx_rate: FxRate | None
    base_currency: str

    @property
    def currencies(self) -> tuple[str, ...]:
        return tuple(sorted(self.totals))

    @property
    def single_currency(self) -> str | None:
        return self.currencies[0] if len(self.currencies) == 1 else None

    def get(self, currency: str) -> float:
        return float(self.totals.get(currency, 0.0))


def allocation_by_currency(
    weights: Mapping[str, float] | pd.Series,
    capital: float,
    currency_map: Mapping[str, str],
) -> CurrencyTotals:
    """Allocation value per currency (never mixes currencies silently)."""
    totals = split_currency_totals(weights, capital, currency_map)
    fx_rate = get_configured_fx_rate()
    mixed = len(totals) > 1
    combined: float | None = None
    combined_currency: str | None = None
    if not mixed:
        combined_currency = next(iter(totals)) if totals else None
        combined = float(sum(totals.values())) if totals else None
    elif fx_rate is not None:
        converted, currency = combine_totals(totals, fx_rate)
        combined, combined_currency = converted, currency
    series = weights if isinstance(weights, pd.Series) else pd.Series(dict(weights))
    base = _dominant_currency(series, currency_map) or (combined_currency or "USD")
    return CurrencyTotals(
        totals={str(code): float(value) for code, value in totals.items()},
        mixed=mixed,
        combined_total=combined,
        combined_currency=combined_currency,
        fx_rate=fx_rate,
        base_currency=base,
    )


def _dominant_currency(
    weights: Mapping[str, float] | pd.Series, currency_map: Mapping[str, str]
) -> str | None:
    series = weights if isinstance(weights, pd.Series) else pd.Series(dict(weights))
    if series.empty:
        return None
    grouped = series.groupby(pd.Series(currency_map).reindex(series.index)).sum()
    grouped = grouped.dropna()
    if grouped.empty:
        return None
    return str(grouped.idxmax())


def is_single_currency(currency_map: Mapping[str, str]) -> bool:
    return len({code for code in currency_map.values() if code}) <= 1


def get_configured_fx_rate() -> FxRate | None:
    """The explicitly declared FX rate from the environment (``.env``)."""
    return load_env_fx_rate()


def _convert(amount: float, source: str, target: str, fx_rate: FxRate) -> float:
    if source == target:
        return amount
    if source == fx_rate.base and target == fx_rate.quote:
        return amount * fx_rate.rate
    if source == fx_rate.quote and target == fx_rate.base:
        return amount / fx_rate.rate
    raise InvalidInputError(
        f"No declared exchange rate between {source} and {target}.",
        code="missing_currency",
        params={"symbols": [f"{source}->{target}"]},
    )


def combine_totals(
    totals: Mapping[str, float], fx_rate: FxRate | None = None, target: str | None = None
) -> tuple[float | None, str | None]:
    """Combine per-currency totals into one amount.

    Returns ``(None, None)`` when no explicit rate covers the conversion: the
    application then reports separate currency totals instead of guessing.
    """
    if not totals:
        return None, None
    rate = fx_rate if fx_rate is not None else get_configured_fx_rate()
    currencies = [code for code in totals if code]
    if len(currencies) == 1:
        return float(sum(totals.values())), currencies[0]
    if rate is None:
        return None, None
    destination = target or rate.base
    combined = 0.0
    for code, amount in totals.items():
        try:
            combined += _convert(float(amount), code, destination, rate)
        except InvalidInputError:
            return None, None
    return combined, destination


def currency_label(currency: str, language: str = "en") -> str:
    """Localised currency name, e.g. ``dirham marocain (MAD)``."""
    from utils.i18n import translate

    code = (currency or "").upper()
    key = f"currency.{code}"
    text = translate(key, language)
    return text if text != key else code
