"""Single translation helper used by every page and component.

* Dictionaries live in ``locales/en.json`` and ``locales/fr.json`` as flat
  ``"section.key": "text"`` mappings, which makes completeness easy to check
  (see ``tests/test_locales.py``).
* Placeholders use ``str.format`` syntax: ``t("builder.weights_total", total="98%")``.
* A missing key never breaks a page: the key itself is returned and recorded in
  :func:`missing_keys` for developers.
* The current language lives in ``st.session_state["fp_language"]`` when a
  Streamlit runtime is active, with a module-level fallback so the same helper
  works in unit tests and scripts.
"""

from __future__ import annotations

import functools
import json
import logging
from pathlib import Path
from typing import Any, Iterable

from config import LANGUAGES, default_language

logger = logging.getLogger(__name__)

LOCALES_DIR = Path(__file__).resolve().parents[1] / "locales"
DEFAULT_LANGUAGE = "en"
FALLBACK_LANGUAGE = "en"

_missing: set[str] = set()
_current_language: str = default_language()


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
@functools.lru_cache(maxsize=8)
def load_locale(language: str) -> dict[str, str]:
    """Load (and cache) one locale dictionary."""
    path = LOCALES_DIR / f"{_normalise_language(language)}.json"
    if not path.exists():
        logger.warning("Locale file not found: %s", path)
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:  # pragma: no cover - corrupted file
        logger.error("Could not read locale %s: %s", path, exc)
        return {}
    if not isinstance(payload, dict):
        return {}
    return {str(key): str(value) for key, value in payload.items()}


def _normalise_language(language: str | None) -> str:
    if not language:
        return DEFAULT_LANGUAGE
    code = str(language).strip().lower()[:2]
    return code if code in LANGUAGES else DEFAULT_LANGUAGE


def available_languages() -> dict[str, str]:
    """``{"en": "English", "fr": "Français"}``."""
    return dict(LANGUAGES)


def language_label(language: str) -> str:
    return LANGUAGES.get(_normalise_language(language), language)


# ---------------------------------------------------------------------------
# Current language
# ---------------------------------------------------------------------------
def _session_language() -> str | None:
    try:
        import streamlit as st

        value = st.session_state.get("fp_language")
        if value:
            return _normalise_language(value)
    except Exception:
        return None
    return None


def set_language(language: str) -> str:
    """Set the active language (session state when available)."""
    global _current_language
    code = _normalise_language(language)
    _current_language = code
    try:
        import streamlit as st

        st.session_state["fp_language"] = code
    except Exception:
        pass
    return code


def get_language() -> str:
    """Active language code (``en``/``fr``)."""
    return _session_language() or _current_language


def is_rtl(language: str | None = None) -> bool:
    return False  # Both supported languages are left-to-right.


# ---------------------------------------------------------------------------
# Translation
# ---------------------------------------------------------------------------
def translate(key: str, language: str | None = None, **params: Any) -> str:
    """Translate ``key`` into ``language`` (explicit language, no session)."""
    code = _normalise_language(language)
    catalogue = load_locale(code)
    text = catalogue.get(key)
    if text is None and code != FALLBACK_LANGUAGE:
        text = load_locale(FALLBACK_LANGUAGE).get(key)
    if text is None:
        _missing.add(key)
        logger.debug("Missing translation key: %s", key)
        return key
    if params:
        try:
            return text.format(**params)
        except (KeyError, IndexError, ValueError):
            logger.debug("Placeholder mismatch for key %s", key)
            return text
    return text


def t(key: str, **params: Any) -> str:
    """Translate ``key`` into the currently selected language."""
    return translate(key, get_language(), **params)


def t_opt(key: str, **params: Any) -> str | None:
    """Translate ``key`` or return ``None`` when it is not defined.

    Handy for optional UI elements such as tooltips that are only shown when a
    translation exists.
    """
    text = translate(key, get_language(), **params)
    return None if text == key else text


def has_key(key: str, language: str | None = None) -> bool:
    return key in load_locale(_normalise_language(language))


def missing_keys() -> set[str]:
    """Keys requested but not found in any locale (developer diagnostics)."""
    return set(_missing)


def reset_missing_keys() -> None:
    _missing.clear()


def locale_keys(language: str = DEFAULT_LANGUAGE) -> set[str]:
    return set(load_locale(language).keys())


def key_differences(left: str = "en", right: str = "fr") -> dict[str, set[str]]:
    """Compare two locale files (used by the test suite)."""
    left_keys = locale_keys(left)
    right_keys = locale_keys(right)
    return {
        f"missing_in_{right}": left_keys - right_keys,
        f"missing_in_{left}": right_keys - left_keys,
    }


def locale_coverage(pages_dir: Path | None = None) -> dict[str, Any]:
    """Quick diagnostic: how many keys each locale defines."""
    return {
        "languages": {code: len(locale_keys(code)) for code in LANGUAGES},
        "missing_at_runtime": sorted(_missing),
    }


def parameter_names(key: str, language: str = DEFAULT_LANGUAGE) -> Iterable[str]:
    """Extract ``{placeholder}`` names from a translated string."""
    import re

    text = load_locale(language).get(key, "")
    return re.findall(r"\{(\w+)\}", text)
