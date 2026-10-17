"""Central configuration and domain vocabulary for FinPilot AI.

Everything that encodes a *financial assumption* lives here so it is easy to
audit and to change in one place:

* annualisation conventions (252 trading days),
* default risk-free rate,
* demo-data seed (reproducible synthetic series),
* analysis windows and validation thresholds,
* LLM provider configuration (read from environment variables only).

The module is framework-agnostic: it never imports Streamlit, so it can be
used from the analytics engine, the tests and the CLI alike.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import date, timedelta
from enum import Enum

try:  # python-dotenv is optional at import time (never a hard requirement).
    from dotenv import load_dotenv

    load_dotenv(override=False)
except Exception:  # pragma: no cover - dotenv missing or unreadable .env
    pass


# ---------------------------------------------------------------------------
# Identity
# ---------------------------------------------------------------------------
APP_NAME = "FinPilot AI"
APP_ICON = "🧭"
APP_VERSION = "1.0.0"
APP_DESCRIPTION = (
    "AI-powered investment portfolio management and analytics platform for "
    "US and Moroccan markets (educational project)."
)

# ---------------------------------------------------------------------------
# Financial conventions
# ---------------------------------------------------------------------------
#: Number of trading periods used to annualise daily statistics.
TRADING_DAYS_PER_YEAR = 252

#: Annualised risk-free rate used by the Sharpe / Sortino ratios (configurable
#: in the Settings page; expressed as a decimal, e.g. 0.03 = 3%).
DEFAULT_RISK_FREE_RATE = 0.03

#: Minimum number of *return* observations required to compute a metric.
MIN_OBSERVATIONS_FOR_METRICS = 2

#: Below this number of observations the app shows a "low confidence" warning.
RECOMMENDED_MIN_OBSERVATIONS = 60

#: Optimisation is refused below this number of aligned observations.
MIN_OBSERVATIONS_FOR_OPTIMISATION = 60

#: Tolerance used when checking that weights sum to 1 (analytics level).
WEIGHT_SUM_TOLERANCE = 1e-6

#: Tolerance accepted by the user interface before offering to normalise
#: weights automatically (1% of the portfolio).
WEIGHT_SUM_UI_TOLERANCE = 0.01

#: Minimum weight that is still considered an intentional allocation.
MIN_MEANINGFUL_WEIGHT = 1e-6

# ---------------------------------------------------------------------------
# Defaults for the user interface (overridable through the Settings page)
# ---------------------------------------------------------------------------
DEFAULT_CAPITAL = 100_000.0
DEFAULT_ANALYSIS_YEARS = 5
MIN_ANALYSIS_YEARS = 1
MAX_ANALYSIS_YEARS = 25

#: Reproducible seed for the simulated (demo) price series.
DEMO_SEED = 20240517

#: Cache lifetime for market-data requests (seconds).
CACHE_TTL_SECONDS = 60 * 30

#: Network timeout (seconds) for market-data providers.
MARKET_DATA_TIMEOUT_SECONDS = 20
MARKET_DATA_MAX_RETRIES = 2

# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------


class Market(str, Enum):
    """Supported market venues."""

    US = "US"
    MOROCCO = "MA"

    @property
    def currency(self) -> str:
        return "USD" if self is Market.US else "MAD"


class DataMode(str, Enum):
    """Where the price history comes from."""

    #: Reproducible synthetic series, generated locally, labelled as simulated.
    DEMO = "demo"
    #: Real historical data (yfinance for the US, CSV import for Morocco).
    HISTORICAL = "historical"


class AssetClass(str, Enum):
    EQUITY = "equity"
    TECHNOLOGY = "technology"
    FINANCIALS = "financials"
    FIXED_INCOME = "fixed_income"
    COMMODITY = "commodity"
    REAL_ESTATE = "real_estate"
    CASH = "cash"
    UNKNOWN = "unknown"


LANGUAGES: dict[str, str] = {"en": "English", "fr": "Français"}


def _env(name: str, default: str) -> str:
    value = os.getenv(name)
    return value.strip() if value and value.strip() else default


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        return float(raw.strip())
    except ValueError:
        return default


# ---------------------------------------------------------------------------
# LLM configuration (secrets stay in the environment, never in the code)
# ---------------------------------------------------------------------------
LLM_ENV_API_KEY = "LLM_API_KEY"
LLM_ENV_BASE_URL = "LLM_BASE_URL"
LLM_ENV_MODEL = "LLM_MODEL"
LLM_ENV_TIMEOUT = "LLM_TIMEOUT_SECONDS"
LLM_ENV_MAX_TOKENS = "LLM_MAX_TOKENS"
LLM_ENV_TEMPERATURE = "LLM_TEMPERATURE"

#: Fallback environment variables accepted for convenience.
LLM_API_KEY_ALIASES = ("LLM_API_KEY", "OPENAI_API_KEY")

DEFAULT_LLM_BASE_URL = "https://api.openai.com/v1"
DEFAULT_LLM_MODEL = "gpt-4o-mini"


@dataclass(frozen=True)
class LLMSettings:
    """Resolved LLM configuration. ``api_key`` is never logged or displayed."""

    api_key: str | None
    base_url: str
    model: str
    timeout_seconds: float
    max_tokens: int
    temperature: float

    @property
    def configured(self) -> bool:
        return bool(self.api_key)

    @property
    def masked_key(self) -> str:
        """Return a masked representation safe to display in the UI."""
        if not self.api_key:
            return "—"
        if len(self.api_key) <= 8:
            return "*" * len(self.api_key)
        return f"{self.api_key[:4]}…{self.api_key[-4:]}"

    @property
    def endpoint(self) -> str:
        return f"{self.base_url.rstrip('/')}/chat/completions"


def _first_env(names: tuple[str, ...]) -> str | None:
    for name in names:
        raw = os.getenv(name)
        if raw and raw.strip():
            return raw.strip()
    return None


def load_llm_settings() -> LLMSettings:
    """Read LLM configuration from the environment (called at request time)."""
    return LLMSettings(
        api_key=_first_env(LLM_API_KEY_ALIASES),
        base_url=_env(LLM_ENV_BASE_URL, DEFAULT_LLM_BASE_URL),
        model=_env(LLM_ENV_MODEL, DEFAULT_LLM_MODEL),
        timeout_seconds=_env_float(LLM_ENV_TIMEOUT, 45.0),
        max_tokens=int(_env_float(LLM_ENV_MAX_TOKENS, 900)),
        temperature=_env_float(LLM_ENV_TEMPERATURE, 0.2),
    )


# ---------------------------------------------------------------------------
# Optional explicit FX rates (never inferred, never guessed)
# ---------------------------------------------------------------------------
FX_RATE_ENV = "FP_FX_RATE_USDMAD"
FX_RATE_DATE_ENV = "FP_FX_RATE_DATE"


@dataclass(frozen=True)
class FxRate:
    """An explicitly declared FX rate: 1 ``base`` = ``rate`` ``quote``."""

    base: str
    quote: str
    rate: float
    as_of: date | None
    source: str

    @property
    def label(self) -> str:
        return f"{self.base}/{self.quote}"


def load_env_fx_rate() -> FxRate | None:
    """Return the explicitly configured USD/MAD rate, if any.

    FinPilot AI never invents exchange rates: without an explicit rate the
    application reports separate per-currency totals instead of a combined one.
    """
    raw = os.getenv(FX_RATE_ENV)
    if not raw or not raw.strip():
        return None
    try:
        rate = float(raw.strip())
    except ValueError:
        return None
    if not (rate > 0):
        return None
    as_of: date | None = None
    raw_date = os.getenv(FX_RATE_DATE_ENV)
    if raw_date and raw_date.strip():
        try:
            as_of = date.fromisoformat(raw_date.strip())
        except ValueError:
            as_of = None
    return FxRate(
        base="USD",
        quote="MAD",
        rate=rate,
        as_of=as_of,
        source=".env (FP_FX_RATE_USDMAD)",
    )


# ---------------------------------------------------------------------------
# Application defaults (overridable from the environment / Settings page)
# ---------------------------------------------------------------------------
DEFAULT_LANGUAGE = _env("FP_DEFAULT_LANGUAGE", "en")
DEFAULT_DATA_MODE = _env("FP_DEFAULT_DATA_MODE", DataMode.DEMO.value)
DEFAULT_MARKET = _env("FP_DEFAULT_MARKET", Market.US.value)


def default_language() -> str:
    lang = DEFAULT_LANGUAGE.lower()
    return lang if lang in LANGUAGES else "en"


def default_market() -> Market:
    try:
        return Market(DEFAULT_MARKET.upper())
    except ValueError:
        return Market.US


def default_data_mode() -> DataMode:
    try:
        return DataMode(DEFAULT_DATA_MODE.lower())
    except ValueError:
        return DataMode.DEMO


def default_start_date(years: int = DEFAULT_ANALYSIS_YEARS, today: date | None = None) -> date:
    today = today or date.today()
    try:
        return today.replace(year=today.year - years)
    except ValueError:  # 29 February
        return today.replace(year=today.year - years, day=28)


def analysis_window(years: int, today: date | None = None) -> tuple[date, date]:
    """Return a (start, end) window of ``years`` years ending today."""
    today = today or date.today()
    years = max(MIN_ANALYSIS_YEARS, min(MAX_ANALYSIS_YEARS, int(years)))
    return default_start_date(years, today), today


def default_end_date(today: date | None = None) -> date:
    return today or date.today()


def lookback_window(days: int, today: date | None = None) -> tuple[date, date]:
    today = today or date.today()
    return today - timedelta(days=days), today
