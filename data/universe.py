"""Instrument universes for the US and Moroccan markets.

Design rules that matter for financial correctness
-------------------------------------------------
* The **US universe** lists real, verifiable US-listed tickers. Prices for them
  are only ever loaded from yfinance (historical mode) — never invented.
* The **Moroccan universe** is *not* hard-coded: no Casablanca Stock Exchange
  symbol, price or index is invented here. The user provides their own symbols
  and metadata either through a JSON file (``data/morocco_universe.json``,
  see ``data/morocco_universe.example.json``) or through CSV imports.
* **Demo universes** use explicitly fictional symbols (``DEMO-*``) so a
  simulated price can never be mistaken for a real security quote.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Iterable, Mapping

from config import AssetClass, Market

__all__ = [
    "Instrument",
    "US_UNIVERSE",
    "US_BENCHMARKS",
    "DEMO_UNIVERSE_US",
    "DEMO_UNIVERSE_MOROCCO",
    "DEMO_BENCHMARKS",
    "MOROCCO_UNIVERSE_PATH",
    "load_morocco_universe",
    "morocco_universe_template",
    "universe_for",
    "benchmarks_for",
    "instrument_lookup",
    "DEMO_PREFIX",
]

DEMO_PREFIX = "DEMO-"


@dataclass(frozen=True)
class Instrument:
    """Metadata for a tradable instrument or index."""

    symbol: str
    name: str
    asset_class: AssetClass = AssetClass.UNKNOWN
    currency: str = "USD"
    market: Market = Market.US
    sector: str | None = None
    source: str = "builtin"
    is_demo: bool = False
    name_fr: str | None = None

    @property
    def is_benchmark(self) -> bool:
        return self.asset_class is AssetClass.UNKNOWN and self.sector == "benchmark"

    def display_name(self, language: str = "en") -> str:
        if language == "fr" and self.name_fr:
            return self.name_fr
        return self.name

    def as_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "name": self.name,
            "name_fr": self.name_fr,
            "asset_class": self.asset_class.value,
            "currency": self.currency,
            "market": self.market.value,
            "sector": self.sector,
            "source": self.source,
            "is_demo": self.is_demo,
        }

    @classmethod
    def from_dict(cls, payload: Mapping, *, default_market: Market, default_currency: str) -> "Instrument":
        asset_class_raw = str(payload.get("asset_class", "unknown")).lower()
        try:
            asset_class = AssetClass(asset_class_raw)
        except ValueError:
            asset_class = AssetClass.UNKNOWN
        return cls(
            symbol=str(payload["symbol"]).strip(),
            name=str(payload.get("name", payload["symbol"])).strip(),
            asset_class=asset_class,
            currency=str(payload.get("currency", default_currency)).upper(),
            market=default_market,
            sector=payload.get("sector"),
            source=str(payload.get("source", "user_provided")),
            is_demo=bool(payload.get("is_demo", False)),
            name_fr=payload.get("name_fr"),
        )


def _instrument(symbol: str, name: str, asset_class: AssetClass, sector: str, name_fr: str | None = None) -> Instrument:
    return Instrument(
        symbol=symbol,
        name=name,
        asset_class=asset_class,
        currency="USD",
        market=Market.US,
        sector=sector,
        source="builtin",
        name_fr=name_fr,
    )


# ---------------------------------------------------------------------------
# US universe (real tickers; prices only from yfinance)
# ---------------------------------------------------------------------------
US_UNIVERSE: dict[str, Instrument] = {
    symbol: instrument
    for symbol, instrument in [
        ("AAPL", _instrument("AAPL", "Apple Inc.", AssetClass.TECHNOLOGY, "Technology")),
        ("MSFT", _instrument("MSFT", "Microsoft Corporation", AssetClass.TECHNOLOGY, "Technology")),
        ("NVDA", _instrument("NVDA", "NVIDIA Corporation", AssetClass.TECHNOLOGY, "Semiconductors")),
        ("GOOGL", _instrument("GOOGL", "Alphabet Inc.", AssetClass.TECHNOLOGY, "Communication Services")),
        ("AMZN", _instrument("AMZN", "Amazon.com Inc.", AssetClass.TECHNOLOGY, "Consumer Discretionary")),
        ("META", _instrument("META", "Meta Platforms Inc.", AssetClass.TECHNOLOGY, "Communication Services")),
        ("JPM", _instrument("JPM", "JPMorgan Chase & Co.", AssetClass.FINANCIALS, "Financials")),
        ("BAC", _instrument("BAC", "Bank of America Corp.", AssetClass.FINANCIALS, "Financials")),
        ("JNJ", _instrument("JNJ", "Johnson & Johnson", AssetClass.EQUITY, "Health Care")),
        ("PG", _instrument("PG", "Procter & Gamble Co.", AssetClass.EQUITY, "Consumer Staples")),
        ("XOM", _instrument("XOM", "Exxon Mobil Corporation", AssetClass.EQUITY, "Energy")),
        ("TSLA", _instrument("TSLA", "Tesla Inc.", AssetClass.EQUITY, "Consumer Discretionary")),
        ("SPY", _instrument("SPY", "SPDR S&P 500 ETF Trust", AssetClass.EQUITY, "Broad US equity ETF")),
        ("VOO", _instrument("VOO", "Vanguard S&P 500 ETF", AssetClass.EQUITY, "Broad US equity ETF")),
        ("VTI", _instrument("VTI", "Vanguard Total Stock Market ETF", AssetClass.EQUITY, "Total US market ETF")),
        ("IWM", _instrument("IWM", "iShares Russell 2000 ETF", AssetClass.EQUITY, "US small caps ETF")),
        ("QQQ", _instrument("QQQ", "Invesco QQQ Trust (Nasdaq 100)", AssetClass.TECHNOLOGY, "Growth equity ETF")),
        ("EFA", _instrument("EFA", "iShares MSCI EAFE ETF", AssetClass.EQUITY, "Developed markets ex-US ETF")),
        ("EEM", _instrument("EEM", "iShares MSCI Emerging Markets ETF", AssetClass.EQUITY, "Emerging markets ETF")),
        ("AGG", _instrument("AGG", "iShares Core U.S. Aggregate Bond ETF", AssetClass.FIXED_INCOME, "US bonds ETF")),
        ("BND", _instrument("BND", "Vanguard Total Bond Market ETF", AssetClass.FIXED_INCOME, "US bonds ETF")),
        ("TLT", _instrument("TLT", "iShares 20+ Year Treasury Bond ETF", AssetClass.FIXED_INCOME, "Long US Treasuries ETF")),
        ("SHY", _instrument("SHY", "iShares 1-3 Year Treasury Bond ETF", AssetClass.FIXED_INCOME, "Short US Treasuries ETF")),
        ("LQD", _instrument("LQD", "iShares iBoxx $ Investment Grade Corporate Bond ETF", AssetClass.FIXED_INCOME, "Corporate bonds ETF")),
        ("GLD", _instrument("GLD", "SPDR Gold Shares", AssetClass.COMMODITY, "Gold ETF")),
        ("VNQ", _instrument("VNQ", "Vanguard Real Estate ETF", AssetClass.REAL_ESTATE, "US REITs ETF")),
    ]
}

US_BENCHMARKS: dict[str, Instrument] = {
    symbol: Instrument(
        symbol=symbol,
        name=name,
        asset_class=AssetClass.UNKNOWN,
        currency="USD",
        market=Market.US,
        sector="benchmark",
        source="builtin",
    )
    for symbol, name in [
        ("^GSPC", "S&P 500 Index"),
        ("^IXIC", "Nasdaq Composite Index"),
        ("^DJI", "Dow Jones Industrial Average"),
        ("SPY", "SPDR S&P 500 ETF Trust"),
        ("AGG", "iShares Core U.S. Aggregate Bond ETF"),
    ]
}

# The S&P 500 is not part of the selectable asset universe of the builder.
US_UNIVERSE_WITHOUT_BENCHMARKS = {
    symbol: instrument for symbol, instrument in US_UNIVERSE.items() if symbol not in US_BENCHMARKS
}


# ---------------------------------------------------------------------------
# Demo universes — explicitly fictional symbols
# ---------------------------------------------------------------------------
def _demo(
    symbol: str,
    name: str,
    asset_class: AssetClass,
    sector: str,
    currency: str,
    market: Market,
    name_fr: str,
) -> Instrument:
    return Instrument(
        symbol=symbol,
        name=name,
        asset_class=asset_class,
        currency=currency,
        market=market,
        sector=sector,
        source="demo",
        is_demo=True,
        name_fr=name_fr,
    )


DEMO_UNIVERSE_US: dict[str, Instrument] = {
    item.symbol: item
    for item in [
        _demo("DEMO-US-EQ", "Simulated US Large-Cap Equity", AssetClass.EQUITY, "Simulated equity", "USD", Market.US, "Actions US grandes capitalisations (simulé)"),
        _demo("DEMO-US-TECH", "Simulated US Technology Equity", AssetClass.TECHNOLOGY, "Simulated technology", "USD", Market.US, "Actions technologiques US (simulé)"),
        _demo("DEMO-US-SC", "Simulated US Small-Cap Equity", AssetClass.EQUITY, "Simulated small caps", "USD", Market.US, "Petites capitalisations US (simulé)"),
        _demo("DEMO-US-BND", "Simulated US Government Bonds", AssetClass.FIXED_INCOME, "Simulated government bonds", "USD", Market.US, "Obligations d'État US (simulé)"),
        _demo("DEMO-US-CORP", "Simulated US Corporate Bonds", AssetClass.FIXED_INCOME, "Simulated corporate bonds", "USD", Market.US, "Obligations d'entreprise US (simulé)"),
        _demo("DEMO-US-GOLD", "Simulated Gold", AssetClass.COMMODITY, "Simulated commodity", "USD", Market.US, "Or (simulé)"),
        _demo("DEMO-US-REIT", "Simulated US Real Estate", AssetClass.REAL_ESTATE, "Simulated REITs", "USD", Market.US, "Immobilier coté US (simulé)"),
        _demo("DEMO-US-CASH", "Simulated Cash / Money Market", AssetClass.CASH, "Simulated cash", "USD", Market.US, "Liquidités (simulé)"),
    ]
}

DEMO_UNIVERSE_MOROCCO: dict[str, Instrument] = {
    item.symbol: item
    for item in [
        _demo("DEMO-MA-EQ-A", "Simulated Moroccan Equity A", AssetClass.EQUITY, "Simulated equity", "MAD", Market.MOROCCO, "Action marocaine A (simulé)"),
        _demo("DEMO-MA-EQ-B", "Simulated Moroccan Equity B", AssetClass.EQUITY, "Simulated equity", "MAD", Market.MOROCCO, "Action marocaine B (simulé)"),
        _demo("DEMO-MA-BANK", "Simulated Moroccan Banking Equity", AssetClass.FINANCIALS, "Simulated financials", "MAD", Market.MOROCCO, "Action bancaire marocaine (simulé)"),
        _demo("DEMO-MA-TELCO", "Simulated Moroccan Telecom Equity", AssetClass.EQUITY, "Simulated telecom", "MAD", Market.MOROCCO, "Télécom marocain (simulé)"),
        _demo("DEMO-MA-BND", "Simulated Moroccan Government Bonds", AssetClass.FIXED_INCOME, "Simulated government bonds", "MAD", Market.MOROCCO, "Obligations d'État marocaines (simulé)"),
        _demo("DEMO-MA-CASH", "Simulated Moroccan Cash", AssetClass.CASH, "Simulated cash", "MAD", Market.MOROCCO, "Liquidités MAD (simulé)"),
    ]
}

DEMO_BENCHMARKS: dict[Market, Instrument] = {
    Market.US: _demo(
        "DEMO-US-MKT",
        "Simulated US Broad Market Index",
        AssetClass.UNKNOWN,
        "benchmark",
        "USD",
        Market.US,
        "Indice de marché US large (simulé)",
    ),
    Market.MOROCCO: _demo(
        "DEMO-MA-MKT",
        "Simulated Moroccan Broad Market Index",
        AssetClass.UNKNOWN,
        "benchmark",
        "MAD",
        Market.MOROCCO,
        "Indice de marché marocain large (simulé)",
    ),
}


# ---------------------------------------------------------------------------
# Moroccan universe (user supplied)
# ---------------------------------------------------------------------------
DATA_DIR = Path(__file__).resolve().parent
MOROCCO_UNIVERSE_PATH = DATA_DIR / "morocco_universe.json"
MOROCCO_UNIVERSE_EXAMPLE_PATH = DATA_DIR / "morocco_universe.example.json"


@dataclass(frozen=True)
class MoroccoUniverse:
    """User-provided Casablanca Stock Exchange metadata and CSV locations."""

    assets: dict[str, Instrument]
    benchmark: Instrument | None
    source_label: str
    as_of: str | None = None
    csv_files: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()

    @property
    def is_empty(self) -> bool:
        return not self.assets

    def with_assets(self, extra: Iterable[Instrument]) -> "MoroccoUniverse":
        merged = dict(self.assets)
        for instrument in extra:
            merged[instrument.symbol] = instrument
        return replace(self, assets=merged)


def morocco_universe_template() -> dict:
    """Documented JSON template for a user-provided Moroccan universe."""
    return {
        "_instructions": (
            "FinPilot AI does not ship any Casablanca Stock Exchange symbol or price. "
            "Fill in your own verified instruments here (or import CSV files from the "
            "Portfolio Builder). Every symbol listed below must be present in the "
            "provided CSV files, otherwise it is reported as unavailable."
        ),
        "source": "user-provided (e.g. broker export, exchange factsheet)",
        "as_of": None,
        "currency": "MAD",
        "csv_files": [],
        "benchmark": None,
        "assets": [],
    }


def load_morocco_universe(
    path: Path | str | None = None,
) -> MoroccoUniverse:
    """Load the user-provided Moroccan universe.

    Returns an empty universe (with an informational warning) when the file is
    missing: the application never invents Moroccan tickers.
    """
    universe_path = Path(path) if path is not None else MOROCCO_UNIVERSE_PATH
    if not universe_path.exists():
        return MoroccoUniverse(
            assets={},
            benchmark=None,
            source_label="none",
            warnings=("morocco.no_universe_file",),
        )
    try:
        payload = json.loads(universe_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return MoroccoUniverse(
            assets={},
            benchmark=None,
            source_label="invalid",
            warnings=("morocco.invalid_universe_file",),
        )
    if not isinstance(payload, dict):
        return MoroccoUniverse(
            assets={},
            benchmark=None,
            source_label="invalid",
            warnings=("morocco.invalid_universe_file",),
        )

    default_currency = str(payload.get("currency", "MAD")).upper()
    assets: dict[str, Instrument] = {}
    warnings: list[str] = []
    raw_assets = payload.get("assets") or []
    if isinstance(raw_assets, list):
        for entry in raw_assets:
            if not isinstance(entry, dict) or not entry.get("symbol"):
                warnings.append("morocco.invalid_asset_entry")
                continue
            instrument = Instrument.from_dict(
                entry, default_market=Market.MOROCCO, default_currency=default_currency
            )
            instrument = replace(instrument, source=str(entry.get("source", payload.get("source", "user_provided"))))
            assets[instrument.symbol] = instrument
    else:
        warnings.append("morocco.invalid_assets_block")

    benchmark: Instrument | None = None
    raw_benchmark = payload.get("benchmark")
    if isinstance(raw_benchmark, dict) and raw_benchmark.get("symbol"):
        benchmark = Instrument.from_dict(
            raw_benchmark, default_market=Market.MOROCCO, default_currency=default_currency
        )
        benchmark = replace(benchmark, sector="benchmark", asset_class=AssetClass.UNKNOWN)

    csv_files: tuple[str, ...] = ()
    raw_csv = payload.get("csv_files")
    if isinstance(raw_csv, list):
        csv_files = tuple(str(item) for item in raw_csv if str(item).strip())

    if not assets:
        warnings.append("morocco.empty_universe")

    return MoroccoUniverse(
        assets=assets,
        benchmark=benchmark,
        source_label=str(payload.get("source", "user_provided")),
        as_of=payload.get("as_of"),
        csv_files=csv_files,
        warnings=tuple(warnings),
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def universe_for(market: Market, *, data_mode: str = "demo", morocco: MoroccoUniverse | None = None) -> dict[str, Instrument]:
    """Return the selectable instruments for a market + data mode."""
    if data_mode == "demo":
        return dict(DEMO_UNIVERSE_MOROCCO if market is Market.MOROCCO else DEMO_UNIVERSE_US)
    if market is Market.MOROCCO:
        universe = morocco or load_morocco_universe()
        return dict(universe.assets)
    return dict(US_UNIVERSE)


def benchmarks_for(market: Market, *, data_mode: str = "demo", morocco: MoroccoUniverse | None = None) -> dict[str, Instrument]:
    """Return the available benchmarks for a market + data mode."""
    if data_mode == "demo":
        return {DEMO_BENCHMARKS[market].symbol: DEMO_BENCHMARKS[market]}
    if market is Market.MOROCCO:
        universe = morocco or load_morocco_universe()
        if universe.benchmark is None:
            return {}
        return {universe.benchmark.symbol: universe.benchmark}
    return dict(US_BENCHMARKS)


def instrument_lookup(
    symbols: Iterable[str],
    *,
    market: Market,
    data_mode: str = "demo",
    extra: Mapping[str, Instrument] | None = None,
    morocco: MoroccoUniverse | None = None,
) -> dict[str, Instrument]:
    """Resolve symbol metadata, falling back to a generic placeholder entry.

    A symbol that is not in the known universe is still usable (CSV imports),
    but it is clearly marked as "user provided" in the interface.
    """
    known: dict[str, Instrument] = {}
    known.update(universe_for(market, data_mode=data_mode, morocco=morocco))
    known.update(benchmarks_for(market, data_mode=data_mode, morocco=morocco))
    if extra:
        known.update(dict(extra))

    resolved: dict[str, Instrument] = {}
    for symbol in symbols:
        if symbol in known:
            resolved[symbol] = known[symbol]
            continue
        currency = "MAD" if market is Market.MOROCCO else "USD"
        resolved[symbol] = Instrument(
            symbol=symbol,
            name=symbol,
            asset_class=AssetClass.UNKNOWN,
            currency=currency,
            market=market,
            source="user_provided",
        )
    return resolved
