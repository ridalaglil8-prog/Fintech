"""Market data provider tests: honest failures, currency labelling, adapters.

No network call is ever made: yfinance is monkeypatched, and the Moroccan
provider is exercised through in-memory CSV payloads.
"""

from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from config import DataMode, Market
from data.providers.base import (
    PriceRequest,
    PriceResult,
    available_providers,
    get_provider,
)
from data.providers.demo_provider import DemoMarketProvider
from data.providers.morocco_market import (
    MoroccoCsvAdapter,
    MoroccoMarketProvider,
    VerifiedSourceAdapter,
)
from data.providers.us_market import US_BENCHMARK_FALLBACKS, UsMarketProvider

WINDOW = (date(2023, 1, 2), date(2023, 12, 29))


def make_request(symbols, market=Market.US, benchmark=None, **kwargs) -> PriceRequest:
    return PriceRequest(
        symbols=tuple(symbols),
        start=WINDOW[0],
        end=WINDOW[1],
        market=market,
        benchmark=benchmark,
        **kwargs,
    )


# ---------------------------------------------------------------------------
# Request object and factory
# ---------------------------------------------------------------------------
def test_price_request_requires_symbols_and_a_valid_window():
    with pytest.raises(ValueError):
        PriceRequest(symbols=(), start=WINDOW[0], end=WINDOW[1])
    with pytest.raises(ValueError):
        PriceRequest(symbols=("A",), start=WINDOW[1], end=WINDOW[0])


def test_price_request_with_symbols_keeps_the_window():
    request = PriceRequest(symbols=("A",), start=WINDOW[0], end=WINDOW[1], benchmark="^GSPC")
    updated = request.with_symbols(["A", "B"])
    assert updated.symbols == ("A", "B")
    assert (updated.start, updated.end, updated.benchmark) == (request.start, request.end, request.benchmark)


@pytest.mark.parametrize(
    ("market", "data_mode", "expected"),
    [
        (Market.US, DataMode.DEMO, DemoMarketProvider),
        (Market.MOROCCO, DataMode.DEMO, DemoMarketProvider),
        (Market.US, DataMode.HISTORICAL, UsMarketProvider),
        (Market.MOROCCO, DataMode.HISTORICAL, MoroccoMarketProvider),
        (Market.US, "demo", DemoMarketProvider),  # strings are accepted
    ],
)
def test_provider_factory(market, data_mode, expected):
    assert isinstance(get_provider(market, data_mode), expected)


def test_available_providers_lists_every_backend():
    providers = available_providers()
    assert {"demo_us", "us_yfinance", "ma_csv"} <= set(providers)


# ---------------------------------------------------------------------------
# Demo provider
# ---------------------------------------------------------------------------
def test_demo_provider_is_labelled_as_simulated():
    provider = DemoMarketProvider(Market.US)
    result = provider.fetch(make_request(["DEMO-US-EQ", "DEMO-US-BND"]))
    assert result.ok
    assert result.is_simulated is True
    assert result.source_key == "demo_us"
    assert sorted(result.prices.columns) == ["DEMO-US-BND", "DEMO-US-EQ"]
    assert result.benchmark_symbol == "DEMO-US-MKT"
    assert result.benchmark is not None and not result.benchmark.empty
    assert set(result.currency_map.values()) == {"USD"}
    codes = {notice.code for notice in result.notices}
    assert "demo.simulated_data" in codes and "demo.business_days_only" in codes
    assert all(notice.level != "error" for notice in result.notices)
    # The benchmark is *not* mixed into the asset price frame.
    assert "DEMO-US-MKT" not in result.prices.columns


def test_demo_provider_is_reproducible():
    provider = DemoMarketProvider(Market.US)
    first = provider.fetch(make_request(["DEMO-US-EQ"]))
    second = provider.fetch(make_request(["DEMO-US-EQ"]))
    pd.testing.assert_frame_equal(first.prices, second.prices)


def test_demo_provider_uses_mad_for_morocco():
    result = DemoMarketProvider(Market.MOROCCO).fetch(
        make_request(["DEMO-MA-EQ-A"], market=Market.MOROCCO)
    )
    assert result.currency_map == {"DEMO-MA-EQ-A": "MAD"}
    assert result.benchmark_symbol == "DEMO-MA-MKT"


def test_demo_provider_benchmark_can_be_overridden():
    provider = DemoMarketProvider(Market.US)
    result = provider.fetch(make_request(["DEMO-US-EQ"], benchmark="DEMO-US-TECH"))
    assert result.benchmark_symbol == "DEMO-US-TECH"
    assert result.benchmark is not None and result.benchmark.name == "DEMO-US-TECH"
    # The benchmark is returned separately, never as an asset column.
    assert result.prices.columns.tolist() == ["DEMO-US-EQ"]


# ---------------------------------------------------------------------------
# US provider (yfinance monkeypatched)
# ---------------------------------------------------------------------------
def _fake_yfinance_frame(symbols, periods=10):
    index = pd.bdate_range("2023-01-02", periods=periods)
    columns = pd.MultiIndex.from_product([["Close", "High"], symbols])
    values = {}
    for position, symbol in enumerate(symbols):
        level = 100.0 + 10 * position
        values[("Close", symbol)] = [level + day for day in range(periods)]
        values[("High", symbol)] = [level + day + 1 for day in range(periods)]
    return pd.DataFrame(values, index=index, columns=columns)


def test_us_provider_extracts_prices_and_reports_missing_symbols(monkeypatch):
    frame = _fake_yfinance_frame(["AAPL", "^GSPC"])

    def fake_download(**kwargs):
        return frame

    monkeypatch.setattr("data.providers.us_market.yf.download", fake_download)
    provider = UsMarketProvider(max_retries=0)
    result = provider.fetch(make_request(["AAPL", "MSFT"]))
    assert result.is_simulated is False
    assert list(result.prices.columns) == ["AAPL"]
    assert result.benchmark_symbol == "^GSPC"
    assert result.missing_symbols == ("MSFT",)
    codes = {notice.code for notice in result.notices}
    assert "us.symbols_unavailable" in codes
    assert "us.data_loaded" in codes
    assert result.currency_map == {"AAPL": "USD"}


def test_us_provider_converts_download_failures_into_errors(monkeypatch):
    def exploding_download(**kwargs):
        raise RuntimeError("network unreachable")

    monkeypatch.setattr("data.providers.us_market.yf.download", exploding_download)
    provider = UsMarketProvider(max_retries=0)
    result = provider.fetch(make_request(["AAPL"]))
    assert result.ok is False
    assert result.prices.empty
    assert result.is_simulated is False
    assert result.errors[0].code == "us.request_failed"
    # A real-data failure never falls back to simulated prices.
    assert result.benchmark is None


def test_us_provider_reports_empty_downloads(monkeypatch):
    monkeypatch.setattr("data.providers.us_market.yf.download", lambda **kwargs: pd.DataFrame())
    result = UsMarketProvider(max_retries=0).fetch(make_request(["AAPL"]))
    assert result.prices.empty
    assert result.errors[0].code == "us.no_data"


def test_us_provider_never_returns_requested_symbols_it_did_not_find(monkeypatch):
    frame = _fake_yfinance_frame(["^GSPC"])

    monkeypatch.setattr("data.providers.us_market.yf.download", lambda **kwargs: frame)
    result = UsMarketProvider(max_retries=0).fetch(make_request(["AAPL"]))
    assert result.prices.empty
    assert result.errors[0].code == "us.no_requested_symbols"
    assert result.is_simulated is False


def test_us_benchmark_fallbacks_are_ordered_starting_with_the_sp500():
    assert US_BENCHMARK_FALLBACKS[0] == "^GSPC"
    assert len(set(US_BENCHMARK_FALLBACKS)) == len(US_BENCHMARK_FALLBACKS)


def test_us_close_extraction_handles_both_yfinance_shapes():
    multi = _fake_yfinance_frame(["AAPL"])
    extracted = UsMarketProvider._extract_close(multi)
    assert list(extracted.columns) == ["AAPL"]

    single = pd.DataFrame({"Close": [1.0, 2.0], "High": [3.0, 4.0]})
    single.attrs["ticker"] = "ZZZ"
    extracted = UsMarketProvider._extract_close(single)
    assert list(extracted.columns) == ["ZZZ"]

    assert UsMarketProvider._extract_close(pd.DataFrame({"a": [1.0]})) is None


# ---------------------------------------------------------------------------
# Morocco (CSV-first)
# ---------------------------------------------------------------------------
MA_CSV = """
date,symbol,close,currency,name
2023-01-02,ATW,412.50,MAD,Attijariwafa Bank
2023-01-03,ATW,415.00,MAD,Attijariwafa Bank
2023-01-04,ATW,413.00,MAD,Attijariwafa Bank
2023-01-05,ATW,418.00,MAD,Attijariwafa Bank
2023-01-06,IAM,88.30,MAD,Maroc Telecom
2023-01-09,IAM,88.90,MAD,Maroc Telecom
"""


def ma_request(symbols, **kwargs):
    return PriceRequest(
        symbols=tuple(symbols),
        start=date(2023, 1, 1),
        end=date(2023, 1, 31),
        market=Market.MOROCCO,
        csv_payloads=kwargs.pop("csv_payloads", (("casablanca.csv", MA_CSV.strip().encode("utf-8")),)),
        **kwargs,
    )


def test_morocco_csv_adapter_returns_validated_prices():
    result = MoroccoCsvAdapter().fetch(ma_request(["ATW"]))
    assert result.ok
    assert list(result.prices.columns) == ["ATW"]
    assert result.currency_map == {"ATW": "MAD"}
    assert result.names == {"ATW": "Attijariwafa Bank"}
    assert result.is_simulated is False
    assert result.source_label == "csv"
    codes = {notice.code for notice in result.notices}
    assert "ma.data_loaded" in codes


def test_morocco_csv_adapter_warns_when_the_benchmark_is_absent():
    result = MoroccoCsvAdapter().fetch(ma_request(["ATW"], benchmark="MASI"))
    assert result.benchmark is None
    assert "ma.benchmark_unavailable" in {notice.code for notice in result.notices}
    assert result.benchmark_symbol is None


def test_morocco_csv_adapter_uses_a_benchmark_column_when_present():
    text = (
        "date,ATW,MASI\n"
        "2023-01-02,412.50,12000\n"
        "2023-01-03,415.00,12050\n"
        "2023-01-04,413.00,12010\n"
        "2023-01-05,418.00,12100\n"
        "2023-01-06,419.00,12150\n"
    )
    request = PriceRequest(
        symbols=("ATW",),
        start=date(2023, 1, 1),
        end=date(2023, 1, 31),
        market=Market.MOROCCO,
        benchmark="MASI",
        csv_payloads=(("wide.csv", text.encode("utf-8")),),
    )
    result = MoroccoCsvAdapter().fetch(request)
    assert result.benchmark_symbol == "MASI"
    assert list(result.benchmark.index) == list(pd.bdate_range("2023-01-02", "2023-01-06"))
    assert list(result.prices.columns) == ["ATW"]


def test_morocco_csv_adapter_requires_a_source():
    request = PriceRequest(
        symbols=("ATW",), start=date(2023, 1, 1), end=date(2023, 1, 31), market=Market.MOROCCO
    )
    result = MoroccoCsvAdapter().fetch(request)
    assert result.prices.empty
    assert result.errors[0].code == "ma.no_source"


def test_morocco_csv_adapter_handles_an_empty_window():
    request = PriceRequest(
        symbols=("ATW",),
        start=date(2020, 1, 1),
        end=date(2020, 12, 31),
        market=Market.MOROCCO,
        csv_payloads=(("casablanca.csv", MA_CSV.strip().encode("utf-8")),),
    )
    result = MoroccoCsvAdapter().fetch(request)
    assert result.prices.empty
    assert "ma.no_data_in_window" in {notice.code for notice in result.errors}


def test_morocco_csv_adapter_reports_unknown_symbols():
    request = PriceRequest(
        symbols=("UNKNOWN",),
        start=date(2023, 1, 1),
        end=date(2023, 1, 31),
        market=Market.MOROCCO,
        csv_payloads=(("casablanca.csv", MA_CSV.strip().encode("utf-8")),),
    )
    result = MoroccoCsvAdapter().fetch(request)
    assert result.prices.empty
    codes = {notice.code for notice in result.notices}
    assert "ma.symbols_unavailable" in codes
    assert "ma.no_requested_symbols" in codes


def test_morocco_csv_adapter_rejects_a_broken_file():
    request = PriceRequest(
        symbols=("ATW",),
        start=date(2023, 1, 1),
        end=date(2023, 1, 31),
        market=Market.MOROCCO,
        csv_payloads=(("broken.csv", b"not,a,price,file\n1,2,3,4\n"),),
    )
    result = MoroccoCsvAdapter().fetch(request)
    codes = {notice.code for notice in result.notices}
    assert "ma.file_rejected" in codes
    assert "ma.no_valid_file" in codes
    assert result.prices.empty


def test_morocco_provider_never_falls_back_to_simulated_prices():
    provider = MoroccoMarketProvider()
    result = provider.fetch(
        PriceRequest(
            symbols=("ATW",), start=date(2023, 1, 1), end=date(2023, 1, 31), market=Market.MOROCCO
        )
    )
    assert result.is_simulated is False
    assert result.prices.empty
    assert result.provider_key == "ma_csv"
    codes = {notice.code for notice in result.notices}
    assert "ma.no_source" in codes
    assert "ma.verified_source_not_configured" in codes


def test_morocco_provider_warns_about_missing_local_files():
    provider = MoroccoMarketProvider(local_csv_files=[f"definitely-missing-{date.today():%Y%m%d}.csv"])
    result = provider.fetch(
        PriceRequest(
            symbols=("ATW",), start=date(2023, 1, 1), end=date(2023, 1, 31), market=Market.MOROCCO
        )
    )
    assert "ma.local_file_missing" in {notice.code for notice in result.notices}


def test_verified_source_adapter_without_a_fetcher_is_not_configured():
    adapter = VerifiedSourceAdapter()
    assert adapter.configured is False
    result = adapter.fetch(ma_request(["ATW"]))
    assert result.errors[0].code == "ma.verified_source_not_configured"


def test_verified_source_adapter_reports_a_failing_feed():
    def broken_feed(request):
        raise TimeoutError("feed timeout")

    result = VerifiedSourceAdapter(broken_feed).fetch(ma_request(["ATW"]))
    assert result.errors[0].code == "ma.verified_source_failed"


def test_morocco_provider_prefers_the_configured_verified_feed():
    index = pd.bdate_range("2023-01-02", periods=5)
    payload = PriceResult(
        prices=pd.DataFrame({"ATW": [1.0, 2.0, 3.0, 4.0, 5.0]}, index=index),
        benchmark=None,
        currency_map={"ATW": "MAD"},
        source_key="ma_verified_feed",
        source_label="verified feed",
        requested_symbols=("ATW",),
        available_symbols=("ATW",),
    )

    def good_feed(request):
        return payload

    provider = MoroccoMarketProvider(adapters=[VerifiedSourceAdapter(good_feed)])
    result = provider.fetch(ma_request(["ATW"]))
    assert result.ok
    assert result.provider_key == "ma_csv"  # normalised to the provider key
    assert list(result.prices.columns) == ["ATW"]
