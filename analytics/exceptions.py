"""Domain exceptions used across FinPilot AI.

All application errors inherit from :class:`FinPilotError` so the UI layer can
catch a single base class and render a friendly message instead of a traceback.
The analytics errors also inherit from :class:`ValueError` which keeps them
idiomatic for callers (and pytest's ``pytest.raises``).
"""

from __future__ import annotations


class FinPilotError(Exception):
    """Base class for every error raised by the application."""

    #: Translation key used by the interface (``error.<code>``).
    code = "generic"

    def __init__(self, message: str = "", *, code: str | None = None, params: dict | None = None):
        super().__init__(message or self.code)
        self.message = message or self.code
        if code:
            self.code = code
        self.params = params or {}


# --------------------------------------------------------------------------
# Analytics / business logic
# --------------------------------------------------------------------------
class AnalyticsError(FinPilotError, ValueError):
    """Invalid input or impossible calculation in the analytics engine."""


class InvalidInputError(AnalyticsError):
    code = "invalid_input"


class InvalidWeightsError(AnalyticsError):
    code = "invalid_weights"


class InsufficientDataError(AnalyticsError):
    code = "insufficient_data"


# --------------------------------------------------------------------------
# Market data
# --------------------------------------------------------------------------
class MarketDataError(FinPilotError):
    code = "market_data_unavailable"


class DataValidationError(FinPilotError, ValueError):
    code = "invalid_csv"


class ProviderNotConfiguredError(MarketDataError):
    code = "provider_not_configured"


# --------------------------------------------------------------------------
# Optimisation / AI
# --------------------------------------------------------------------------
class OptimizationError(FinPilotError, ValueError):
    code = "optimization_failed"


class AIProviderError(FinPilotError):
    code = "ai_unavailable"


class AINotConfiguredError(AIProviderError):
    code = "ai_not_configured"
