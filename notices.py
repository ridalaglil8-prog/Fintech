"""Structured user-facing notices (data quality, errors, warnings).

A notice is rendered by the interface as a translated message: ``code`` is
looked up in the locale files under ``notice.<code>`` and ``params`` are
interpolated. Keeping messages as codes + parameters (instead of pre-formatted
strings) is what makes the bilingual interface possible and testable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

__all__ = ["Notice", "info", "warning", "error", "success", "format_message", "collect"]

LEVELS = ("info", "success", "warning", "error")

LEVEL_ICONS = {
    "info": "ℹ️",
    "success": "✅",
    "warning": "⚠️",
    "error": "⛔",
}


@dataclass(frozen=True)
class Notice:
    """A translatable message with a severity level."""

    level: str
    code: str
    params: dict = field(default_factory=dict)

    @property
    def key(self) -> str:
        """Translation key of the notice."""
        return f"notice.{self.code}"

    @property
    def is_error(self) -> bool:
        return self.level == "error"

    @property
    def icon(self) -> str:
        return LEVEL_ICONS.get(self.level, "")

    def as_dict(self) -> dict:
        return {"level": self.level, "code": self.code, "params": dict(self.params)}


def info(code: str, **params) -> Notice:
    return Notice("info", code, params)


def success(code: str, **params) -> Notice:
    return Notice("success", code, params)


def warning(code: str, **params) -> Notice:
    return Notice("warning", code, params)


def error(code: str, **params) -> Notice:
    return Notice("error", code, params)


def collect(*notices: Notice | None | Iterable[Notice | None]) -> list[Notice]:
    """Flatten ``None``-tolerant notice collections."""
    result: list[Notice] = []
    for item in notices:
        if item is None:
            continue
        if isinstance(item, Notice):
            result.append(item)
        else:
            result.extend(notice for notice in item if notice is not None)
    return result


def format_message(notice: Notice, translate) -> str:
    """Render a notice with a translation function (``utils.i18n.t``).

    Unknown keys fall back to the code itself, so a missing translation never
    breaks a page — it only shows a less friendly message.
    """
    try:
        template = translate(notice.key)
    except Exception:  # pragma: no cover - defensive
        template = notice.code
    if template == notice.key:
        template = notice.code
    if notice.params:
        try:
            return str(template).format(**notice.params)
        except (KeyError, IndexError, ValueError):
            return str(template)
    return str(template)
