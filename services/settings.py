"""Application settings: financial assumptions and interface preferences.

Settings are stored in ``st.session_state`` (no database in this MVP) and are
mirrored into URL-free, serialisable snapshots so analytics results can be
cached safely without ever caching secrets.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from datetime import date
from typing import Any

from config import (
    DEFAULT_ANALYSIS_YEARS,
    DEFAULT_CAPITAL,
    DEFAULT_RISK_FREE_RATE,
    DEMO_SEED,
    MAX_ANALYSIS_YEARS,
    MIN_ANALYSIS_YEARS,
    DataMode,
    FxRate,
    Market,
    analysis_window,
    default_data_mode,
    default_language,
    default_market,
    load_env_fx_rate,
)

__all__ = ["AppSettings", "default_settings"]


@dataclass
class AppSettings:
    """Everything the user can configure outside of the portfolio itself."""

    language: str = field(default_factory=default_language)
    market: Market = field(default_factory=default_market)
    data_mode: DataMode = field(default_factory=default_data_mode)
    risk_free_rate: float = DEFAULT_RISK_FREE_RATE
    analysis_years: int = DEFAULT_ANALYSIS_YEARS
    start_date: date | None = None
    end_date: date | None = None
    demo_seed: int = DEMO_SEED
    min_coverage: float = 0.6
    allow_multi_currency: bool = False
    capital: float = DEFAULT_CAPITAL
    rolling_window: int = 21
    show_simulated_banner: bool = True

    # -- helpers ----------------------------------------------------------
    @property
    def fx_rate(self) -> FxRate | None:
        """Explicitly declared FX rate, if the environment provides one."""
        return load_env_fx_rate()

    def window(self, *, today: date | None = None) -> tuple[date, date]:
        """Effective ``(start, end)`` analysis window."""
        if self.start_date and self.end_date and self.start_date < self.end_date:
            return self.start_date, self.end_date
        years = max(MIN_ANALYSIS_YEARS, min(MAX_ANALYSIS_YEARS, int(self.analysis_years)))
        return analysis_window(years, today)

    def snapshot(self) -> dict[str, Any]:
        """Serialisable, secret-free representation used as a cache key."""
        payload = asdict(self)
        payload["market"] = self.market.value
        payload["data_mode"] = self.data_mode.value
        payload["start_date"] = self.start_date.isoformat() if self.start_date else None
        payload["end_date"] = self.end_date.isoformat() if self.end_date else None
        return payload

    def to_dict(self) -> dict[str, Any]:
        return self.snapshot()

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "AppSettings":
        if not payload:
            return default_settings()
        data = dict(payload)
        try:
            data["market"] = Market(str(data.get("market", default_market().value)).upper())
        except ValueError:
            data["market"] = default_market()
        try:
            data["data_mode"] = DataMode(str(data.get("data_mode", default_data_mode().value)).lower())
        except ValueError:
            data["data_mode"] = default_data_mode()
        for key in ("start_date", "end_date"):
            value = data.get(key)
            if isinstance(value, str) and value:
                try:
                    data[key] = date.fromisoformat(value)
                except ValueError:
                    data[key] = None
            elif isinstance(value, date):
                data[key] = value
        allowed = {field_name for field_name in cls.__dataclass_fields__}
        filtered = {key: value for key, value in data.items() if key in allowed}
        try:
            return cls(**filtered)
        except TypeError:  # pragma: no cover - defensive
            return default_settings()

    def with_changes(self, **changes: Any) -> "AppSettings":
        return replace(self, **changes)

    def normalised(self) -> "AppSettings":
        """Clamp numeric settings into their supported ranges."""
        return replace(
            self,
            risk_free_rate=float(max(-0.5, min(1.0, self.risk_free_rate))),
            analysis_years=int(max(MIN_ANALYSIS_YEARS, min(MAX_ANALYSIS_YEARS, self.analysis_years))),
            min_coverage=float(max(0.0, min(1.0, self.min_coverage))),
            rolling_window=int(max(2, self.rolling_window)),
            capital=float(max(0.0, self.capital)) or DEFAULT_CAPITAL,
        )


def default_settings() -> AppSettings:
    """Settings for a fresh session."""
    return AppSettings()
