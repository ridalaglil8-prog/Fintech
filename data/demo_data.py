"""Reproducible synthetic price series for demo mode.

The generator is a **one-factor model**: every simulated asset is driven by a
common market factor plus an idiosyncratic shock, which produces realistic
co-movement (``corr(a, b) = beta_a * beta_b``) without ever encoding a real
security's price history.

Daily simple returns for asset ``i``:
    r_i,t = mu_i / 252 + (sigma_i / sqrt(252)) * (beta_i * f_t + sqrt(1 - beta_i^2) * e_i,t)
with ``f_t``, ``e_i,t`` ~ N(0, 1) and ``mu_i``/``sigma_i`` annualised.

Prices are then ``P_t = P_0 * cumprod(1 + r_t)``, so they are always positive.

Determinism
-----------
The random generator is seeded with :data:`config.DEMO_SEED` and the symbols are
sorted before drawing, so the *same* inputs always produce the *same* series —
which is exactly what the unit tests rely on.

Everything produced here is **simulated data** and is labelled as such in the
interface (``is_simulated = True``). It must never be presented as a real
market observation.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta

import numpy as np
import pandas as pd

from analytics.exceptions import InsufficientDataError, InvalidInputError
from config import DEMO_SEED, Market, TRADING_DAYS_PER_YEAR
from data.universe import (
    DEMO_BENCHMARKS,
    DEMO_UNIVERSE_MOROCCO,
    DEMO_UNIVERSE_US,
    Instrument,
)

__all__ = [
    "DemoProfile",
    "DEMO_PROFILES",
    "demo_instruments",
    "demo_universe",
    "demo_benchmark",
    "generate_demo_prices",
    "demo_analysis_window",
    "SIMULATION_LABEL",
]

SIMULATION_LABEL = "simulated"

#: Starting level of every simulated price series (index 100 = neutral base).
BASE_LEVEL = 100.0


@dataclass(frozen=True)
class DemoProfile:
    """Annualised parameters of a simulated asset."""

    annual_return: float
    annual_volatility: float
    beta: float  #: exposure to the common market factor, in [0, 1]
    start_level: float = BASE_LEVEL


_DEMO_PROFILES_US: dict[str, DemoProfile] = {
    "DEMO-US-EQ": DemoProfile(0.075, 0.170, 0.95),
    "DEMO-US-TECH": DemoProfile(0.105, 0.280, 0.85),
    "DEMO-US-SC": DemoProfile(0.080, 0.230, 0.90),
    "DEMO-US-BND": DemoProfile(0.030, 0.060, 0.10),
    "DEMO-US-CORP": DemoProfile(0.042, 0.085, 0.30),
    "DEMO-US-GOLD": DemoProfile(0.040, 0.150, 0.05),
    "DEMO-US-REIT": DemoProfile(0.065, 0.200, 0.60),
    "DEMO-US-CASH": DemoProfile(0.025, 0.005, 0.00),
}

_DEMO_PROFILES_MOROCCO: dict[str, DemoProfile] = {
    "DEMO-MA-EQ-A": DemoProfile(0.070, 0.190, 0.90),
    "DEMO-MA-EQ-B": DemoProfile(0.055, 0.220, 0.80),
    "DEMO-MA-BANK": DemoProfile(0.080, 0.240, 0.85),
    "DEMO-MA-TELCO": DemoProfile(0.060, 0.160, 0.60),
    "DEMO-MA-BND": DemoProfile(0.035, 0.070, 0.15),
    "DEMO-MA-CASH": DemoProfile(0.022, 0.004, 0.00),
}

DEMO_PROFILES: dict[Market, dict[str, DemoProfile]] = {
    Market.US: _DEMO_PROFILES_US,
    Market.MOROCCO: _DEMO_PROFILES_MOROCCO,
}

_DEFAULT_PROFILE = DemoProfile(0.060, 0.180, 0.70)


def demo_universe(market: Market) -> dict[str, Instrument]:
    """The fictional instrument universe used in demo mode."""
    return dict(DEMO_UNIVERSE_MOROCCO if market is Market.MOROCCO else DEMO_UNIVERSE_US)


def demo_benchmark(market: Market) -> Instrument:
    """The fictional broad-market index used as the demo benchmark."""
    return DEMO_BENCHMARKS[market]


def demo_instruments(market: Market) -> dict[str, Instrument]:
    """Universe + benchmark for demo mode."""
    instruments = demo_universe(market)
    benchmark = demo_benchmark(market)
    instruments[benchmark.symbol] = benchmark
    return instruments


def profile_for(symbol: str, market: Market) -> DemoProfile:
    profiles = DEMO_PROFILES.get(market, {})
    if symbol in profiles:
        return profiles[symbol]
    # Deterministic, neutral fallback for unknown demo symbols.
    seed = sum(ord(char) for char in symbol) % 997
    annual_vol = 0.12 + (seed % 140) / 1000.0  # 12% - 26%
    annual_return = 0.04 + (seed % 40) / 1000.0  # 4% - 8%
    beta = 0.5 + (seed % 50) / 100.0  # 0.5 - 1.0
    return DemoProfile(annual_return, annual_vol, min(beta, 1.0))


def demo_analysis_window(years: int = 5, *, today: date | None = None) -> tuple[date, date]:
    """Calendar window used to generate the demo series."""
    today = today or date.today()
    return today - timedelta(days=int(365.25 * years)), today


def _business_days(start: date, end: date) -> pd.DatetimeIndex:
    if start >= end:
        raise InvalidInputError(
            "The demo period must span at least two different dates.",
            code="invalid_date_range",
            params={"start": str(start), "end": str(end)},
        )
    index = pd.bdate_range(start=pd.Timestamp(start), end=pd.Timestamp(end), freq="B")
    if len(index) < 3:
        raise InsufficientDataError(
            "Not enough business days in the selected demo period.",
            code="insufficient_observations",
            params={"available": int(len(index)), "required": 3, "what": "demo data"},
        )
    return index


def generate_demo_prices(
    symbols: list[str] | tuple[str, ...],
    start: date | pd.Timestamp,
    end: date | pd.Timestamp,
    *,
    market: Market = Market.US,
    seed: int = DEMO_SEED,
) -> pd.DataFrame:
    """Generate a reproducible, correlated price history for fictional assets.

    Parameters
    ----------
    symbols:
        Fictional demo symbols (typically ``DEMO-*``). Unknown symbols get a
        deterministic neutral profile so custom demo universes still work.
    start, end:
        Inclusive calendar window; weekends are excluded (Mon–Fri business
        days). Market holidays are intentionally *not* modelled and this is
        flagged in the interface.
    market:
        Selects the default profile table (USD vs MAD demo universe).
    seed:
        Reproducibility seed. Same seed + same symbols + same window gives
        bit-identical prices.

    Returns
    -------
    pandas.DataFrame
        Price levels indexed by business day, one column per symbol. The
        synthetic nature of the data is preserved by the caller
        (``is_simulated=True``) and displayed in the UI.
    """
    ordered = sorted({str(symbol).strip() for symbol in symbols if str(symbol).strip()})
    if not ordered:
        raise InvalidInputError("No symbol requested for the demo series.", code="empty_symbols")
    if not isinstance(seed, int) or seed < 0:
        raise InvalidInputError("The demo seed must be a non-negative integer.", code="invalid_seed", params={"seed": seed})

    index = _business_days(pd.Timestamp(start).date(), pd.Timestamp(end).date())
    n_periods = len(index)
    rng = np.random.default_rng(seed)
    factor = rng.standard_normal(n_periods)

    data: dict[str, np.ndarray] = {}
    for symbol in ordered:
        profile = profile_for(symbol, market)
        beta = float(np.clip(profile.beta, 0.0, 1.0))
        idiosyncratic = rng.standard_normal(n_periods)
        # Systematic part = beta * factor * sigma_i; the idiosyncratic part
        # completes the variance so that total volatility matches the profile
        # and corr(i, j) = beta_i * beta_j by construction.
        systematic = beta * factor * profile.annual_volatility
        specific = np.sqrt(max(1.0 - beta**2, 0.0)) * idiosyncratic * profile.annual_volatility
        daily = profile.annual_return / TRADING_DAYS_PER_YEAR + (
            systematic + specific
        ) / np.sqrt(TRADING_DAYS_PER_YEAR)
        daily = np.clip(daily, -0.75, 0.75)  # numerical safety, never triggered in practice
        levels = profile.start_level * np.cumprod(1.0 + daily)
        data[symbol] = np.round(levels, 4)

    frame = pd.DataFrame(data, index=index)
    frame.index.name = "date"
    return frame


def demo_price_frame(
    symbols: list[str] | tuple[str, ...],
    start: date | pd.Timestamp,
    end: date | pd.Timestamp,
    *,
    market: Market = Market.US,
    seed: int = DEMO_SEED,
) -> tuple[pd.DataFrame, Instrument]:
    """Convenience helper returning (asset prices, benchmark series)."""
    benchmark = demo_benchmark(market)
    requested = list(dict.fromkeys([*symbols, benchmark.symbol]))
    frame = generate_demo_prices(requested, start, end, market=market, seed=seed)
    return frame.drop(columns=[benchmark.symbol]), frame[benchmark.symbol]
