"""Streamlit caching helpers that stay importable outside a Streamlit runtime.

``analytics``, ``data`` and ``optimization`` never cache anything themselves;
only the service layer (I/O, market data) is cached. When Streamlit is not
installed (unit tests, scripts), the decorators degrade to no-ops so the same
code paths keep working.
"""

from __future__ import annotations

from typing import Any, Callable, TypeVar

F = TypeVar("F", bound=Callable[..., Any])

try:  # pragma: no cover - depends on the environment
    import streamlit as st

    STREAMLIT_AVAILABLE = True
except Exception:  # pragma: no cover - streamlit is a hard requirement in the app
    st = None  # type: ignore[assignment]
    STREAMLIT_AVAILABLE = False


def _noop_cache(*decorator_args: Any, **decorator_kwargs: Any) -> Any:
    """Emulate ``st.cache_data``/``st.cache_resource`` without Streamlit."""

    def decorate(func: F) -> F:
        return func

    if len(decorator_args) == 1 and callable(decorator_args[0]) and not decorator_kwargs:
        return decorator_args[0]
    return decorate


cached_data: Any = st.cache_data if STREAMLIT_AVAILABLE else _noop_cache
cached_resource: Any = st.cache_resource if STREAMLIT_AVAILABLE else _noop_cache


def clear_caches() -> bool:
    """Clear every Streamlit data cache. Returns True when it was available."""
    if not STREAMLIT_AVAILABLE:
        return False
    try:
        st.cache_data.clear()
        return True
    except Exception:  # pragma: no cover - no runtime
        return False


def cache_info() -> dict[str, Any]:
    return {
        "streamlit_available": STREAMLIT_AVAILABLE,
        "cached_data": bool(STREAMLIT_AVAILABLE),
    }
