"""Shared pytest fixtures: deterministic data, project path, clean state.

Every fixture is built from explicit arithmetic so the expected values can be
verified by hand — no random data is used for assertions.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import DataMode, Market  # noqa: E402
from services.portfolio import default_portfolio, portfolio_from_symbols  # noqa: E402
from services.settings import AppSettings  # noqa: E402


@pytest.fixture(scope="session")
def project_root() -> Path:
    return ROOT


@pytest.fixture
def business_days() -> pd.DatetimeIndex:
    """Deterministic 260-day business-day index."""
    return pd.bdate_range("2022-01-03", periods=260)


@pytest.fixture
def constant_growth_prices() -> pd.Series:
    """100 -> 110 -> 121: two +10% days (hand-checkable)."""
    index = pd.bdate_range("2024-01-01", periods=3)
    return pd.Series([100.0, 110.0, 121.0], index=index, name="A")


@pytest.fixture
def two_asset_prices(business_days: pd.DatetimeIndex) -> pd.DataFrame:
    """Two assets with an exactly known, uncorrelated return pattern.

    * ``A``: +1%, -1% repeated  -> mean 0     (round trips to its start level for
      every pair of days, so the level path is easy to reason about);
    * ``B``: +2%, +2%, -2%, -2% -> mean 0, variance 4x that of ``A`` and zero
      covariance with ``A`` (the cross products cancel over each 4-day block).

      cov(A, B) over a 4-day block:
      (0.01*0.02) + (-0.01*0.02) + (0.01*-0.02) + (-0.01*-0.02) = 0
    """
    n = len(business_days)
    pattern_a = np.array([0.01, -0.01])
    pattern_b = np.array([0.02, 0.02, -0.02, -0.02])
    returns_a = np.resize(pattern_a, n)
    returns_b = np.resize(pattern_b, n)
    prices_a = 100.0 * np.cumprod(1.0 + returns_a)
    prices_b = 50.0 * np.cumprod(1.0 + returns_b)
    return pd.DataFrame({"A": prices_a, "B": prices_b}, index=business_days)


@pytest.fixture
def two_asset_returns(two_asset_prices: pd.DataFrame) -> pd.DataFrame:
    return two_asset_prices.pct_change(fill_method=None).dropna()


@pytest.fixture
def known_returns() -> pd.Series:
    """Alternating 0% / +2% returns: mean 1%, sample std = 1% * sqrt(252/251)."""
    values = np.resize(np.array([0.0, 0.02]), 252)
    index = pd.bdate_range("2023-01-02", periods=252)
    return pd.Series(values, index=index, name="portfolio")


@pytest.fixture
def demo_portfolio() -> "object":
    return default_portfolio(Market.US, DataMode.DEMO)


@pytest.fixture
def demo_settings() -> AppSettings:
    from datetime import date

    return AppSettings(
        language="en",
        market=Market.US,
        data_mode=DataMode.DEMO,
        risk_free_rate=0.03,
        start_date=date(2021, 1, 4),
        end_date=date(2023, 12, 29),
    )


@pytest.fixture
def morocco_demo_portfolio() -> "object":
    return default_portfolio(Market.MOROCCO, DataMode.DEMO)


@pytest.fixture
def equal_weight_portfolio():
    """A small, explicitly weighted 3-asset portfolio (weights sum to 1)."""

    def _factory(symbols: list[str], capital: float = 10_000.0, market=Market.US, data_mode=DataMode.DEMO):
        weights = {symbol: 1.0 / len(symbols) for symbol in symbols}
        return portfolio_from_symbols(
            symbols, market=market, data_mode=data_mode, capital=capital, weights=weights
        )

    return _factory


@pytest.fixture(autouse=True)
def _clean_i18n_cache():
    """Keep translation state deterministic across tests."""
    from utils import i18n

    i18n.set_language("en")
    yield
    i18n.set_language("en")


@pytest.fixture
def csv_bytes() -> "callable":
    def _factory(text: str, encoding: str = "utf-8") -> bytes:
        return text.strip().encode(encoding)

    return _factory
