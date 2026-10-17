"""CSV import tests: layouts, encodings, validation and failure modes.

All fixtures are tiny, hand-written files so the expected frames are explicit.
"""

from __future__ import annotations

import pandas as pd
import pytest

from analytics.exceptions import DataValidationError
from data.csv_loader import (
    combine_loaded_prices,
    detect_delimiter,
    load_price_csv,
    read_text_with_fallback,
    validate_price_frame,
)

MIN_OBS = 3

LONG_CSV = """
date,symbol,close,currency,name
2023-01-02,ATW,412.50,MAD,Attijariwafa Bank
2023-01-03,ATW,415.00,MAD,Attijariwafa Bank
2023-01-04,ATW,413.00,MAD,Attijariwafa Bank
2023-01-02,IAM,88.30,MAD,Maroc Telecom
2023-01-03,IAM,88.90,MAD,Maroc Telecom
2023-01-04,IAM,89.10,MAD,Maroc Telecom
"""

WIDE_CSV = """
date;ATW;IAM
02/01/2023;412,50;88,30
03/01/2023;415,00;88,90
04/01/2023;413,00;89,10
05/01/2023;418,00;89,40
"""


def test_long_format_is_parsed_with_metadata(csv_bytes):
    loaded = load_price_csv(
        csv_bytes(LONG_CSV), min_observations=MIN_OBS, source_label="casablanca.csv"
    )
    assert sorted(loaded.symbols) == ["ATW", "IAM"]
    assert loaded.currencies == {"ATW": "MAD", "IAM": "MAD"}
    assert loaded.names["ATW"] == "Attijariwafa Bank"
    assert loaded.sources == ("casablanca.csv",)
    assert loaded.first_observation == pd.Timestamp("2023-01-02")
    assert loaded.last_observation == pd.Timestamp("2023-01-04")
    assert loaded.errors == []
    assert loaded.prices.loc[pd.Timestamp("2023-01-03"), "IAM"] == pytest.approx(88.90)


def test_wide_format_dayfirst_dates_and_french_decimals(csv_bytes):
    loaded = load_price_csv(
        csv_bytes(WIDE_CSV), min_observations=4, source_label="wide.csv", default_currency="MAD"
    )
    assert loaded.symbols == ["ATW", "IAM"]
    assert loaded.prices.index[0] == pd.Timestamp("2023-01-02")
    assert loaded.prices.loc[pd.Timestamp("2023-01-02"), "ATW"] == pytest.approx(412.50)
    assert loaded.currencies == {"ATW": "MAD", "IAM": "MAD"}
    # The gradual import confirms the comma decimal separator was understood.
    assert loaded.prices["IAM"].iloc[-1] == pytest.approx(89.40)


def test_wide_format_fills_gaps_with_nan(csv_bytes):
    text = "date,A,B\n2023-01-02,10,20\n2023-01-03,11,\n2023-01-04,12,22\n2023-01-05,13,23"
    loaded = load_price_csv(csv_bytes(text), min_observations=MIN_OBS)
    assert pd.isna(loaded.prices.loc[pd.Timestamp("2023-01-03"), "B"])
    # Missing prices are never filled in.
    assert loaded.prices.isna().any().any()


def test_duplicate_rows_keep_the_last_occurrence(csv_bytes):
    text = (
        "date,symbol,close\n"
        "2023-01-02,A,100\n"
        "2023-01-02,A,110\n"
        "2023-01-03,A,111\n"
        "2023-01-04,A,112\n"
    )
    loaded = load_price_csv(csv_bytes(text), min_observations=MIN_OBS)
    assert loaded.prices.loc[pd.Timestamp("2023-01-02"), "A"] == pytest.approx(110.0)
    assert any(issue.code == "duplicate_rows_merged" for issue in loaded.warnings)


def test_non_numeric_price_rows_are_dropped(csv_bytes):
    text = (
        "date,symbol,close\n"
        "2023-01-02,A,100\n"
        "2023-01-03,A,n/a\n"
        "2023-01-04,A,102\n"
        "2023-01-05,A,103\n"
    )
    loaded = load_price_csv(csv_bytes(text), min_observations=MIN_OBS)
    assert len(loaded.prices) == 3
    assert any(issue.code == "non_numeric_prices_dropped" for issue in loaded.warnings)


def test_symbol_without_enough_history_is_excluded(csv_bytes):
    text = (
        "date,symbol,close\n"
        "2023-01-02,A,100\n"
        "2023-01-03,A,101\n"
        "2023-01-04,A,102\n"
        "2023-01-02,B,50\n"
        "2023-01-03,B,51\n"
    )
    loaded = load_price_csv(csv_bytes(text), min_observations=MIN_OBS)
    assert loaded.symbols == ["A"]
    dropped = [issue for issue in loaded.warnings if issue.code == "symbols_with_insufficient_history"]
    assert dropped and dropped[0].params["symbols"] == ["B"]


def test_no_symbol_survives_validation(csv_bytes):
    text = "date,symbol,close\n2023-01-02,A,100\n2023-01-03,A,101\n"
    with pytest.raises(DataValidationError) as info:
        load_price_csv(csv_bytes(text), min_observations=MIN_OBS)
    assert info.value.code == "no_valid_symbols"


# ---------------------------------------------------------------------------
# Fatal errors
# ---------------------------------------------------------------------------
def test_missing_date_column_is_fatal(csv_bytes):
    with pytest.raises(DataValidationError) as info:
        load_price_csv(csv_bytes("symbol,close\nA,100\n"), min_observations=MIN_OBS)
    assert info.value.code == "missing_date_column"


def test_no_numeric_column_is_fatal(csv_bytes):
    text = "date;A;B\n2023-01-02;abc;def\n2023-01-03;ghi;jkl\n"
    with pytest.raises(DataValidationError) as info:
        load_price_csv(csv_bytes(text), min_observations=MIN_OBS)
    assert info.value.code == "no_price_column"


def test_empty_file_is_fatal(csv_bytes):
    with pytest.raises(DataValidationError) as info:
        load_price_csv(csv_bytes("   \n  "), min_observations=MIN_OBS)
    assert info.value.code == "empty_file"


def test_file_not_found_is_fatal(tmp_path):
    with pytest.raises(DataValidationError) as info:
        load_price_csv(tmp_path / "does-not-exist.csv")
    assert info.value.code == "file_not_found"


def test_unsupported_source_is_fatal():
    with pytest.raises(DataValidationError) as info:
        load_price_csv(12345)  # type: ignore[arg-type]
    assert info.value.code == "unsupported_source"


def test_all_zero_prices_are_fatal(csv_bytes):
    text = "date,symbol,close\n2023-01-02,A,0\n2023-01-03,A,0\n2023-01-04,A,0\n"
    with pytest.raises(DataValidationError) as info:
        load_price_csv(csv_bytes(text), min_observations=MIN_OBS)
    assert info.value.code == "non_positive_prices"


def test_isolated_non_positive_price_is_dropped_with_warning(csv_bytes):
    text = (
        "date,symbol,close\n"
        "2023-01-02,A,100\n"
        "2023-01-03,A,0\n"
        "2023-01-04,A,102\n"
        "2023-01-05,A,103\n"
    )
    loaded = load_price_csv(csv_bytes(text), min_observations=MIN_OBS)
    assert pd.isna(loaded.prices.loc[pd.Timestamp("2023-01-03"), "A"])
    assert any(issue.code == "non_positive_prices_dropped" for issue in loaded.warnings)


def test_future_dates_are_dropped(csv_bytes):
    text = (
        "date,symbol,close\n"
        "2023-01-02,A,100\n"
        "2023-01-03,A,101\n"
        "2023-01-04,A,102\n"
        "2099-01-01,A,999\n"
    )
    loaded = load_price_csv(csv_bytes(text), min_observations=MIN_OBS)
    assert loaded.last_observation == pd.Timestamp("2023-01-04")
    assert any(issue.code == "future_dates_dropped" for issue in loaded.warnings)


# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------
def test_encoding_fallback_decodes_latin1():
    text, encoding = read_text_with_fallback("Cours de clôture".encode("latin-1"))
    assert text == "Cours de clôture"
    assert encoding == "latin-1"


def test_encoding_fallback_prefers_utf8():
    text, encoding = read_text_with_fallback("Cours de clôture".encode("utf-8"))
    assert text == "Cours de clôture"
    assert encoding in {"utf-8-sig", "utf-8"}


@pytest.mark.parametrize("delimiter", [",", ";", "\t", "|"])
def test_delimiter_detection(delimiter):
    sample = f"date{delimiter}symbol{delimiter}close\n2023-01-02{delimiter}A{delimiter}100"
    assert detect_delimiter(sample) == delimiter


# ---------------------------------------------------------------------------
# Validation helper and merging
# ---------------------------------------------------------------------------
def test_validate_price_frame_reports_problems_without_mutating():
    index = pd.bdate_range("2023-01-02", periods=4)
    frame = pd.DataFrame({"A": [100.0, None, 102.0, -5.0], "B": [50.0, 51.0, 52.0, 53.0]}, index=index)
    before = frame.copy(deep=True)
    issues = validate_price_frame(
        frame, min_observations=3, expected_currency="MAD", currencies={"A": "USD", "B": "MAD"}
    )
    codes = {issue.code for issue in issues}
    assert "symbol_has_gaps" in codes
    assert "non_positive_prices" in codes
    assert "currency_mismatch" in codes
    assert frame.equals(before)


def test_validate_price_frame_flags_short_history():
    frame = pd.DataFrame({"A": [1.0, 2.0]}, index=pd.bdate_range("2023-01-02", periods=2))
    issues = validate_price_frame(frame, min_observations=10)
    assert any(issue.code == "symbols_with_insufficient_history" for issue in issues)


def test_combine_loaded_prices_merges_columns(csv_bytes):
    first = load_price_csv(
        csv_bytes("date,A\n2023-01-02,10\n2023-01-03,11\n2023-01-04,12\n"),
        min_observations=MIN_OBS,
        default_currency="MAD",
        source_label="one.csv",
    )
    second = load_price_csv(
        csv_bytes("date,B\n2023-01-03,20\n2023-01-04,21\n2023-01-05,22\n"),
        min_observations=MIN_OBS,
        default_currency="MAD",
        source_label="two.csv",
    )
    merged = combine_loaded_prices([first, second])
    assert sorted(merged.symbols) == ["A", "B"]
    assert merged.sources == ("one.csv", "two.csv")
    assert len(merged.prices) == 4
    assert pd.isna(merged.prices.loc[pd.Timestamp("2023-01-02"), "B"])


def test_combine_loaded_prices_requires_data():
    from analytics.exceptions import InvalidInputError

    with pytest.raises(InvalidInputError):
        combine_loaded_prices([])
