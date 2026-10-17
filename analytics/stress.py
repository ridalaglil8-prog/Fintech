"""Hypothetical stress-testing of a portfolio.

Method (deliberately simple and transparent)
--------------------------------------------
A shock is a percentage change applied to the *value* of one asset (or to a
group of assets) at a point in time. With weights ``w_i`` summing to 1, an
initial capital ``V0`` and shocks ``s_i`` (decimals, ``-0.10`` = -10%):

    impact_i      = w_i * V0 * s_i                (monetary impact per asset)
    total_impact  = V0 * sum_i (w_i * s_i)
    impact_pct    = sum_i (w_i * s_i)
    value_after   = V0 + total_impact

This is a **deterministic, one-shot** revaluation: no volatility, no
correlation, no recovery path and no probability is implied. Results are
hypothetical scenarios and are labelled as such everywhere in the interface.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

import numpy as np
import pandas as pd

from analytics.exceptions import InvalidInputError, InvalidWeightsError
from analytics.portfolio import normalize_weights

__all__ = [
    "ShockRule",
    "StressResult",
    "apply_shocks",
    "apply_group_shocks",
    "shock_from_group",
]


@dataclass(frozen=True)
class ShockRule:
    """A named shock applied to a group of symbols.

    ``pct`` is expressed as a decimal change (``-0.10`` = -10%).
    """

    key: str
    label_key: str
    pct: float
    symbols: tuple[str, ...] = ()

    @property
    def code(self) -> str:
        return self.key


@dataclass(frozen=True)
class StressResult:
    """Outcome of a hypothetical scenario."""

    per_asset: pd.DataFrame  #: columns: weight, value, shock, impact, value_after
    shocks: dict[str, float]
    value_before: float
    value_after: float
    total_impact: float
    impact_pct: float
    unshocked_weight: float
    missing_symbols: tuple[str, ...] = ()
    rules: tuple[ShockRule, ...] = ()
    notes: tuple[str, ...] = field(default_factory=tuple)

    @property
    def is_loss(self) -> bool:
        return self.total_impact < 0


def _validate_pct(pct: float, symbol: str) -> float:
    if not np.isfinite(pct):
        raise InvalidInputError(
            f"The shock applied to {symbol} is not a finite number.",
            code="non_finite_shock",
            params={"symbol": symbol},
        )
    if pct < -1.0 or pct > 5.0:
        raise InvalidInputError(
            f"The shock applied to {symbol} must be between -100% and +500%.",
            code="shock_out_of_range",
            params={"symbol": symbol, "value": pct},
        )
    return float(pct)


def apply_shocks(
    weights: Mapping[str, float] | pd.Series,
    capital: float,
    shocks: Mapping[str, float],
    *,
    notes: tuple[str, ...] = (),
    rules: tuple[ShockRule, ...] = (),
) -> StressResult:
    """Apply per-asset percentage shocks to a portfolio.

    Assets without an explicit shock keep a 0% shock (they are simply not
    affected by the scenario).
    """
    if not np.isfinite(capital) or capital <= 0:
        raise InvalidInputError(
            "capital must be a strictly positive finite number.",
            code="invalid_capital",
            params={"capital": capital},
        )
    if not shocks:
        raise InvalidInputError("No shock has been defined.", code="no_shocks")

    aligned = normalize_weights(weights)
    if aligned.empty:
        raise InvalidWeightsError("Empty portfolio.", code="empty_weights")

    unknown = [symbol for symbol in shocks if symbol not in aligned.index]
    if unknown:
        raise InvalidInputError(
            f"Shocks reference assets that are not in the portfolio: {unknown}",
            code="unknown_shock_symbols",
            params={"symbols": unknown},
        )

    shock_series = pd.Series(
        {symbol: _validate_pct(float(shocks.get(symbol, 0.0)), symbol) for symbol in aligned.index},
        dtype="float64",
    )
    values = aligned * float(capital)
    impacts = values * shock_series
    frame = pd.DataFrame(
        {
            "weight": aligned,
            "value": values,
            "shock": shock_series,
            "impact": impacts,
            "value_after": values + impacts,
        }
    )
    total_impact = float(impacts.sum())
    value_after = float(capital) + total_impact
    return StressResult(
        per_asset=frame,
        shocks={str(symbol): float(value) for symbol, value in shock_series.items()},
        value_before=float(capital),
        value_after=value_after,
        total_impact=total_impact,
        impact_pct=total_impact / float(capital),
        unshocked_weight=float(aligned[shock_series == 0.0].sum()),
        missing_symbols=(),
        rules=rules,
        notes=notes,
    )


def shock_from_group(
    symbols: Mapping[str, float] | pd.Series, shock_pct: float, *, group: Mapping[str, str] | None = None
) -> dict[str, float]:
    """Build a ``{symbol: shock}`` mapping for every symbol in a selection."""
    series = symbols if isinstance(symbols, pd.Series) else pd.Series(dict(symbols), dtype="float64")
    _validate_pct(float(shock_pct), "group")
    return {str(symbol): float(shock_pct) for symbol in series.index}


def apply_group_shocks(
    weights: Mapping[str, float] | pd.Series,
    capital: float,
    rules: list[ShockRule],
    *,
    combine: str = "additive",
) -> StressResult:
    """Apply several named group shocks at once.

    ``combine="additive"`` sums the shocks applied to the same asset (two rules
    hitting the same asset add up), which is the intuitive behaviour when a
    user defines "US equity -10%" and "Technology -15%" for an overlapping set.
    """
    if combine not in {"additive", "override"}:
        raise InvalidInputError(
            f"Unknown combination mode '{combine}'.",
            code="invalid_combination_mode",
            params={"mode": combine},
        )
    if not rules:
        raise InvalidInputError("No shock has been defined.", code="no_shocks")

    aligned = normalize_weights(weights)
    shocks: dict[str, float] = {}
    notes: list[str] = []
    for rule in rules:
        pct = _validate_pct(float(rule.pct), rule.key)
        if pct == 0.0:
            continue
        targets = [symbol for symbol in rule.symbols if symbol in aligned.index]
        if not targets:
            notes.append(f"stress.no_target:{rule.key}")
            continue
        for symbol in targets:
            if combine == "additive":
                shocks[symbol] = shocks.get(symbol, 0.0) + pct
            else:
                shocks[symbol] = pct

    if not shocks:
        raise InvalidInputError(
            "None of the selected shocks applies to the current portfolio.",
            code="no_applicable_shocks",
            params={"rules": [rule.key for rule in rules]},
        )
    return apply_shocks(weights, capital, shocks, notes=tuple(notes), rules=tuple(rules))
