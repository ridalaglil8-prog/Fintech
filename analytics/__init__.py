"""FinPilot AI analytics package.

Framework-free (no Streamlit import) so every function is unit-testable:

* :mod:`analytics.returns`     — daily / cumulative / annualised returns
* :mod:`analytics.risk`        — volatility, Sharpe, drawdown, tail risk, concentration
* :mod:`analytics.correlation` — correlation / covariance / diversification
* :mod:`analytics.portfolio`   — weights validation, alignment, valuation, summaries
* :mod:`analytics.stress`      — hypothetical scenario analysis
"""

from __future__ import annotations

__all__ = [
    "correlation",
    "exceptions",
    "portfolio",
    "returns",
    "risk",
    "stress",
]
