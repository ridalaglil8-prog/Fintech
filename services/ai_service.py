"""LLM integration (OpenAI-compatible chat completions API).

Security
--------
* the API key is read from the environment (``LLM_API_KEY`` or
  ``OPENAI_API_KEY``) at call time — it is never stored, logged or rendered;
* errors are converted into translation-ready codes with a short detail, so no
  secret and no full traceback can leak into the interface;
* the application remains fully usable without a key (this module then reports
  "not configured" and every other page keeps working).

Reliability
-----------
Timeouts, connection errors, HTTP status codes and empty answers are all mapped
to a specific error code. A single retry is attempted on a timeout.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Mapping, Sequence

from config import LLMSettings, load_llm_settings
from utils.i18n import get_language

logger = logging.getLogger(__name__)

__all__ = [
    "AIResponse",
    "AIConfigStatus",
    "config_status",
    "is_configured",
    "chat",
    "explain",
    "health_check",
]

try:  # requests is a Streamlit dependency; the guard keeps the module importable.
    import requests

    REQUESTS_AVAILABLE = True
except Exception:  # pragma: no cover - defensive
    requests = None  # type: ignore[assignment]
    REQUESTS_AVAILABLE = False


@dataclass(frozen=True)
class AIConfigStatus:
    """Secret-free view of the AI configuration for the UI."""

    configured: bool
    base_url: str
    model: str
    masked_key: str
    timeout_seconds: float
    max_tokens: int
    temperature: float
    requests_available: bool

    @property
    def status_key(self) -> str:
        return "settings.ai_status_configured" if self.configured else "settings.ai_status_missing"


@dataclass
class AIResponse:
    """Result of an LLM call."""

    ok: bool
    text: str = ""
    error_key: str | None = None
    error_params: dict = field(default_factory=dict)
    model: str | None = None
    usage: dict = field(default_factory=dict)
    elapsed_ms: int = 0
    attempts: int = 1

    @property
    def failed(self) -> bool:
        return not self.ok


def _settings() -> LLMSettings:
    return load_llm_settings()


def config_status(settings: LLMSettings | None = None) -> AIConfigStatus:
    """Configuration summary safe to display (never the full key)."""
    resolved = settings or _settings()
    return AIConfigStatus(
        configured=resolved.configured and REQUESTS_AVAILABLE,
        base_url=resolved.base_url,
        model=resolved.model,
        masked_key=resolved.masked_key,
        timeout_seconds=resolved.timeout_seconds,
        max_tokens=resolved.max_tokens,
        temperature=resolved.temperature,
        requests_available=REQUESTS_AVAILABLE,
    )


def is_configured() -> bool:
    """True when an API key is present and the HTTP client is available."""
    return config_status().configured


def _fail(key: str, **params) -> AIResponse:
    return AIResponse(ok=False, error_key=key, error_params=params)


def chat(
    messages: Sequence[Mapping[str, str]],
    *,
    settings: LLMSettings | None = None,
    max_tokens: int | None = None,
    retries: int = 1,
) -> AIResponse:
    """Send a chat-completions request and return a structured response.

    Never raises: every failure becomes an ``AIResponse`` carrying a
    translation key (``ai.error_*``) and short parameters.
    """
    resolved = settings or _settings()
    if not resolved.configured:
        return _fail("ai.error_not_configured")
    if not REQUESTS_AVAILABLE:
        return _fail("ai.error_not_configured")
    if not messages:
        return _fail("ai.error_unexpected", detail="empty message list")

    payload = {
        "model": resolved.model,
        "messages": list(messages),
        "temperature": resolved.temperature,
        "max_tokens": max_tokens or resolved.max_tokens,
    }
    headers = {
        "Authorization": f"Bearer {resolved.api_key}",
        "Content-Type": "application/json",
    }

    started = time.perf_counter()
    last_error: AIResponse | None = None
    for attempt in range(1, max(1, retries) + 1):
        try:
            response = requests.post(
                resolved.endpoint,
                json=payload,
                headers=headers,
                timeout=resolved.timeout_seconds,
            )
        except requests.Timeout:  # type: ignore[union-attr]
            last_error = _fail("ai.error_timeout", seconds=int(resolved.timeout_seconds))
            logger.warning("LLM request timed out (attempt %s)", attempt)
            continue
        except requests.RequestException as exc:  # type: ignore[union-attr]
            detail = str(exc)[:200]
            last_error = _fail("ai.error_network", detail=detail)
            logger.warning("LLM request failed: %s", detail)
            continue

        elapsed_ms = int((time.perf_counter() - started) * 1000)
        if response.status_code >= 400:
            detail = _safe_error_detail(response)
            logger.warning("LLM HTTP %s: %s", response.status_code, detail)
            return _fail("ai.error_http", status=response.status_code, detail=detail)

        try:
            body = response.json()
        except ValueError:
            return _fail("ai.error_unexpected", detail="invalid JSON response")

        choices = body.get("choices") or []
        if not choices:
            return _fail("ai.error_empty")
        message = choices[0].get("message") or {}
        text = (message.get("content") or "").strip()
        if not text:
            return _fail("ai.error_empty")
        return AIResponse(
            ok=True,
            text=text,
            model=body.get("model", resolved.model),
            usage=body.get("usage") or {},
            elapsed_ms=elapsed_ms,
            attempts=attempt,
        )

    return last_error or _fail("ai.error_unexpected", detail="no attempt completed")


def _safe_error_detail(response) -> str:
    """Extract a short, secret-free error message from an HTTP response."""
    try:
        payload = response.json()
    except ValueError:
        return (response.text or "")[:200]
    error = payload.get("error") if isinstance(payload, dict) else None
    if isinstance(error, dict):
        return str(error.get("message") or error)[:200]
    if isinstance(error, str):
        return error[:200]
    return str(payload)[:200]


def explain(
    question: str,
    context: Mapping | None,
    *,
    language: str | None = None,
    history: Sequence[Mapping[str, str]] | None = None,
    settings: LLMSettings | None = None,
) -> AIResponse:
    """Ask the assistant to explain the computed figures in the active language."""
    from ai.prompts import build_messages

    resolved = settings or _settings()
    language = language or get_language()
    messages = build_messages(question, context, language=language, history=history)
    return chat(messages, settings=resolved)


def health_check(*, settings: LLMSettings | None = None) -> AIResponse:
    """Minimal call used by the Settings page to validate the configuration."""
    return chat(
        [
            {"role": "system", "content": "You reply with a single word."},
            {"role": "user", "content": "ping"},
        ],
        settings=settings,
        max_tokens=8,
        retries=0,
    )
