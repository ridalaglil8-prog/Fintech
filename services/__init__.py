"""Service layer: market data, currency, AI and session orchestration.

Nothing in this package renders UI; pages import these helpers so that business
logic stays in one place and remains testable.
"""

from __future__ import annotations

__all__ = [
    "ai_service",
    "cache",
    "currency",
    "engine",
    "market_data",
    "portfolio",
    "settings",
    "state",
]
