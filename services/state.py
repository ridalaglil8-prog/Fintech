"""Session state management (no database in this MVP).

Every piece of user-specific state lives here:

``fp_settings``        — :class:`~services.settings.AppSettings`
``fp_portfolio``       — :class:`~services.portfolio.Portfolio`
``fp_language``        — ``"en"`` / ``"fr"`` (shared with ``utils.i18n``)
``fp_csv_payloads``    — raw uploaded Moroccan CSV files (name, bytes)
``fp_csv_data``        — validated CSV imports
``fp_ai_messages``     — AI assistant conversation
``fp_stress_shocks``   — last shock definitions used by the stress-testing page
``fp_optimization``    — last optimisation result (for the assistant's context)
``fp_bundle_cache``    — last computed analytics bundle (invalidated on change)

Changing the portfolio, market, data mode or window invalidates the cached
bundle so every page immediately reflects the new state.
"""

from __future__ import annotations

from typing import Any, Iterable

from config import DataMode, Market
from notices import Notice, info
from services.portfolio import Portfolio, default_portfolio, reconcile_portfolio
from services.settings import AppSettings, default_settings
from utils import i18n

__all__ = [
    "init_state",
    "get_settings",
    "set_settings",
    "update_settings",
    "get_portfolio",
    "set_portfolio",
    "get_language",
    "set_language",
    "set_market",
    "set_data_mode",
    "add_csv_payloads",
    "get_csv_payloads",
    "clear_csv_payloads",
    "get_csv_data",
    "set_csv_data",
    "get_ai_messages",
    "append_ai_message",
    "clear_ai_messages",
    "get_stress_shocks",
    "set_stress_shocks",
    "get_optimization_result",
    "set_optimization_result",
    "invalidate_cache",
    "reset_portfolio",
    "apply_language",
]

_FALLBACK: dict[str, Any] = {}


def _state():
    """Return the Streamlit session state, or a local fallback outside a runtime."""
    try:
        import streamlit as st

        return st.session_state
    except Exception:  # pragma: no cover - scripts/tests
        return _FALLBACK


def init_state() -> None:
    """Create the default state (idempotent)."""
    state = _state()
    if "fp_settings" not in state:
        state["fp_settings"] = default_settings()
    settings: AppSettings = state["fp_settings"]
    if "fp_language" not in state:
        state["fp_language"] = settings.language
        i18n.set_language(settings.language)
    if "fp_portfolio" not in state:
        state["fp_portfolio"] = default_portfolio(settings.market, settings.data_mode, capital=settings.capital)
    state.setdefault("fp_csv_payloads", [])
    state.setdefault("fp_csv_data", None)
    state.setdefault("fp_ai_messages", [])
    state.setdefault("fp_stress_shocks", {})
    state.setdefault("fp_optimization", None)
    state.setdefault("fp_notices", [])
    state.setdefault("fp_bundle_cache", {})


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------
def get_settings() -> AppSettings:
    init_state()
    return _state()["fp_settings"]


def set_settings(settings: AppSettings) -> AppSettings:
    state = _state()
    normalised = settings.normalised()
    state["fp_settings"] = normalised
    i18n.set_language(normalised.language)
    invalidate_cache()
    return normalised


def update_settings(**changes: Any) -> AppSettings:
    return set_settings(get_settings().with_changes(**changes))


# ---------------------------------------------------------------------------
# Portfolio
# ---------------------------------------------------------------------------
def get_portfolio() -> Portfolio:
    init_state()
    return _state()["fp_portfolio"]


def set_portfolio(portfolio: Portfolio) -> Portfolio:
    state = _state()
    state["fp_portfolio"] = portfolio
    invalidate_cache()
    return portfolio


def reset_portfolio() -> Portfolio:
    settings = get_settings()
    return set_portfolio(default_portfolio(settings.market, settings.data_mode, capital=settings.capital))


# ---------------------------------------------------------------------------
# Language, market, data mode
# ---------------------------------------------------------------------------
def get_language() -> str:
    return i18n.get_language()


def set_language(language: str) -> str:
    code = i18n.set_language(language)
    settings = get_settings()
    if settings.language != code:
        set_settings(settings.with_changes(language=code))
    else:
        invalidate_cache()
    return code


def apply_language(language: str) -> str:
    """Session-start helper: ensure the settings and i18n module agree."""
    settings = get_settings()
    if language != settings.language:
        settings = set_settings(settings.with_changes(language=language))
    return i18n.set_language(settings.language)


def set_market(market: Market) -> tuple[Portfolio, list[Notice]]:
    """Switch market and reconcile the portfolio with the new universe."""
    settings = get_settings()
    if settings.market is market:
        return get_portfolio(), []
    settings = set_settings(settings.with_changes(market=market, start_date=None, end_date=None))
    reconciled, notices = reconcile_portfolio(get_portfolio(), market, settings.data_mode)
    if market is Market.MOROCCO and settings.data_mode is DataMode.HISTORICAL and not reconciled.weights:
        notices = [*notices, info("notice.data.no_assets_for_market")]
    set_portfolio(reconciled)
    return reconciled, notices


def set_data_mode(data_mode: DataMode) -> tuple[Portfolio, list[Notice]]:
    """Switch data mode and reconcile the portfolio with the new rules."""
    settings = get_settings()
    if settings.data_mode is data_mode:
        return get_portfolio(), []
    settings = set_settings(settings.with_changes(data_mode=data_mode))
    reconciled, notices = reconcile_portfolio(get_portfolio(), settings.market, data_mode)
    if data_mode is DataMode.HISTORICAL and not reconciled.weights:
        notices = [*notices, info("notice.data.no_assets_for_market")]
    set_portfolio(reconciled)
    return reconciled, notices


# ---------------------------------------------------------------------------
# Moroccan CSV imports
# ---------------------------------------------------------------------------
def add_csv_payloads(payloads: Iterable[tuple[str, bytes]]) -> list[tuple[str, bytes]]:
    state = _state()
    existing = list(state.get("fp_csv_payloads", []))
    known = {name for name, _ in existing}
    for name, payload in payloads:
        if name in known:
            existing = [(n, p) for n, p in existing if n != name]
        existing.append((str(name), bytes(payload)))
    state["fp_csv_payloads"] = existing
    invalidate_cache()
    return existing


def get_csv_payloads() -> tuple[tuple[str, bytes], ...]:
    init_state()
    return tuple(_state().get("fp_csv_payloads", []))


def clear_csv_payloads() -> None:
    state = _state()
    state["fp_csv_payloads"] = []
    state["fp_csv_data"] = None
    invalidate_cache()


def get_csv_data():
    init_state()
    return _state().get("fp_csv_data")


def set_csv_data(data) -> None:
    _state()["fp_csv_data"] = data
    invalidate_cache()


# ---------------------------------------------------------------------------
# AI conversation
# ---------------------------------------------------------------------------
def get_ai_messages() -> list[dict[str, str]]:
    init_state()
    return list(_state().get("fp_ai_messages", []))


def append_ai_message(role: str, content: str, *, meta: dict | None = None) -> None:
    messages = list(_state().get("fp_ai_messages", []))
    entry: dict[str, Any] = {"role": role, "content": content}
    if meta:
        entry["meta"] = meta
    messages.append(entry)
    _state()["fp_ai_messages"] = messages


def clear_ai_messages() -> None:
    _state()["fp_ai_messages"] = []


# ---------------------------------------------------------------------------
# Stress testing / optimisation memory
# ---------------------------------------------------------------------------
def get_stress_shocks() -> dict[str, float]:
    init_state()
    return dict(_state().get("fp_stress_shocks", {}))


def set_stress_shocks(shocks: dict[str, float]) -> dict[str, float]:
    state = _state()
    state["fp_stress_shocks"] = {str(symbol): float(value) for symbol, value in shocks.items()}
    return dict(state["fp_stress_shocks"])


def get_optimization_result():
    init_state()
    return _state().get("fp_optimization")


def set_optimization_result(result) -> None:
    _state()["fp_optimization"] = result


# ---------------------------------------------------------------------------
# Cache invalidation
# ---------------------------------------------------------------------------
def invalidate_cache() -> None:
    """Drop the computed analytics bundle (market-data cache is kept)."""
    try:
        _state()["fp_bundle_cache"] = {}
    except Exception:  # pragma: no cover - defensive
        pass
