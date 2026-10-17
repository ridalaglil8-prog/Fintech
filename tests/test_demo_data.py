"""Demo data generator tests: determinism, realism and honest labelling."""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from analytics.exceptions import InsufficientDataError, InvalidInputError
from config import DEMO_SEED, TRADING_DAYS_PER_YEAR, Market
from data.demo_data import (
    DEMO_PROFILES,
    demo_analysis_window,
    demo_benchmark,
    demo_instruments,
    demo_price_frame,
    demo_universe,
    generate_demo_prices,
    profile_for,
)

WINDOW = (date(2020, 1, 1), date(2024, 12, 31))
SYMBOLS = ["DEMO-US-EQ", "DEMO-US-TECH", "DEMO-US-BND"]


def test_demo_prices_are_deterministic():
    first = generate_demo_prices(SYMBOLS, *WINDOW)
    second = generate_demo_prices(SYMBOLS, *WINDOW)
    pd.testing.assert_frame_equal(first, second)


def test_demo_seed_changes_the_series():
    base = generate_demo_prices(SYMBOLS, *WINDOW, seed=DEMO_SEED)
    other = generate_demo_prices(SYMBOLS, *WINDOW, seed=DEMO_SEED + 1)
    assert not base.equals(other)


def test_demo_prices_are_positive_and_within_the_window():
    frame = generate_demo_prices(SYMBOLS, *WINDOW)
    assert (frame > 0).all().all()
    assert list(frame.columns) == sorted(SYMBOLS)  # columns are sorted for reproducibility
    assert frame.index[0] >= pd.Timestamp(WINDOW[0])
    assert frame.index[-1] <= pd.Timestamp(WINDOW[1])
    assert frame.index.is_monotonic_increasing


def test_demo_calendar_contains_business_days_only():
    frame = generate_demo_prices(SYMBOLS, *WINDOW)
    assert frame.index.dayofweek.max() <= 4
    assert len(frame) > 1_200  # ~5 years of business days


def test_demo_volatility_is_close_to_the_profile():
    frame = generate_demo_prices(["DEMO-US-EQ"], *WINDOW)
    returns = frame["DEMO-US-EQ"].pct_change(fill_method=None).dropna()
    realised = float(returns.std(ddof=1) * np.sqrt(TRADING_DAYS_PER_YEAR))
    expected = DEMO_PROFILES[Market.US]["DEMO-US-EQ"].annual_volatility
    assert realised == pytest.approx(expected, rel=0.2)


def test_demo_correlations_follow_the_one_factor_model():
    """corr(i, j) ~ beta_i * beta_j by construction."""
    frame = generate_demo_prices(["DEMO-US-EQ", "DEMO-US-TECH", "DEMO-US-CASH"], *WINDOW)
    returns = frame.pct_change(fill_method=None).dropna()
    profiles = DEMO_PROFILES[Market.US]
    expected = profiles["DEMO-US-EQ"].beta * profiles["DEMO-US-TECH"].beta
    assert returns["DEMO-US-EQ"].corr(returns["DEMO-US-TECH"]) == pytest.approx(expected, abs=0.06)
    # The "cash" style asset has a near-zero beta, hence no correlation.
    assert abs(returns["DEMO-US-EQ"].corr(returns["DEMO-US-CASH"])) < 0.15


def test_demo_price_frame_separates_the_benchmark():
    prices, benchmark = demo_price_frame(["DEMO-US-EQ", "DEMO-US-TECH"], *WINDOW)
    assert list(prices.columns) == ["DEMO-US-EQ", "DEMO-US-TECH"]
    assert benchmark.name == demo_benchmark(Market.US).symbol
    assert len(benchmark) == len(prices)


def test_demo_window_helper():
    today = date(2026, 6, 1)
    start, end = demo_analysis_window(3, today=today)
    assert end == today
    assert start == today - timedelta(days=int(365.25 * 3))


def test_unknown_demo_symbol_gets_a_deterministic_neutral_profile():
    profile = profile_for("DEMO-CUSTOM-1", Market.US)
    assert profile == profile_for("DEMO-CUSTOM-1", Market.US)
    assert 0.12 <= profile.annual_volatility <= 0.26
    assert 0.04 <= profile.annual_return <= 0.08
    assert 0.5 <= profile.beta <= 1.0


def test_demo_universes_are_fictional_and_market_specific():
    us = demo_universe(Market.US)
    morocco = demo_universe(Market.MOROCCO)
    assert us and morocco
    assert all(symbol.startswith("DEMO-") for symbol in us)
    assert all(symbol.startswith("DEMO-") for symbol in morocco)
    assert set(us) != set(morocco)
    assert demo_benchmark(Market.US).symbol in demo_instruments(Market.US)


def test_demo_generator_rejects_empty_symbols():
    with pytest.raises(InvalidInputError) as info:
        generate_demo_prices([], *WINDOW)
    assert info.value.code == "empty_symbols"


def test_demo_generator_rejects_negative_seed():
    with pytest.raises(InvalidInputError) as info:
        generate_demo_prices(SYMBOLS, *WINDOW, seed=-1)
    assert info.value.code == "invalid_seed"


@pytest.mark.parametrize(
    ("start", "end"),
    [(date(2024, 1, 10), date(2024, 1, 10)), (date(2024, 1, 10), date(2024, 1, 5))],
)
def test_demo_generator_rejects_invalid_ranges(start, end):
    with pytest.raises(InvalidInputError) as info:
        generate_demo_prices(SYMBOLS, start, end)
    assert info.value.code == "invalid_date_range"


def test_demo_generator_requires_a_week_of_business_days():
    # Saturday -> Sunday: no business day at all.
    with pytest.raises(InsufficientDataError) as info:
        generate_demo_prices(SYMBOLS, date(2024, 1, 6), date(2024, 1, 7))
    assert info.value.code == "insufficient_observations"


def test_morocco_demo_series_are_independent_from_us():
    us = generate_demo_prices(["DEMO-MA-EQ-A"], *WINDOW, market=Market.US)
    morocco = generate_demo_prices(["DEMO-MA-EQ-A"], *WINDOW, market=Market.MOROCCO)
    assert not us.equals(morocco)
