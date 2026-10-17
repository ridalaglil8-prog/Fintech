"""Stress-testing tests: exact monetary impacts, validation, grouping."""

from __future__ import annotations

import numpy as np
import pytest

from analytics.exceptions import InvalidInputError, InvalidWeightsError
from analytics.stress import ShockRule, apply_group_shocks, apply_shocks, shock_from_group


def test_single_asset_shock_is_purely_arithmetic():
    weights = {"A": 0.6, "B": 0.4}
    result = apply_shocks(weights, 10_000.0, {"A": -0.10})
    # 60% of 10 000 = 6 000; -10% -> -600
    assert result.total_impact == pytest.approx(-600.0)
    assert result.value_after == pytest.approx(9_400.0)
    assert result.impact_pct == pytest.approx(-0.06)
    assert result.value_before == pytest.approx(10_000.0)
    assert result.is_loss is True
    assert result.unshocked_weight == pytest.approx(0.4)
    assert result.shocks == {"A": -0.10, "B": 0.0}


def test_per_asset_frame_details():
    result = apply_shocks({"A": 0.5, "B": 0.5}, 10_000.0, {"A": -0.20, "B": +0.10})
    frame = result.per_asset
    assert list(frame.columns) == ["weight", "value", "shock", "impact", "value_after"]
    assert frame.loc["A", "impact"] == pytest.approx(-1_000.0)
    assert frame.loc["B", "impact"] == pytest.approx(+500.0)
    assert frame.loc["A", "value_after"] == pytest.approx(4_000.0)
    assert frame["value_after"].sum() == pytest.approx(result.value_after)
    assert result.total_impact == pytest.approx(-500.0)
    assert result.value_after == pytest.approx(9_500.0)


def test_weights_are_normalised_before_shocking():
    result = apply_shocks({"A": 60.0, "B": 40.0}, 10_000.0, {"A": -0.10})
    assert result.total_impact == pytest.approx(-600.0)


@pytest.mark.parametrize("capital", [0.0, -100.0, float("nan"), float("inf")])
def test_invalid_capital_is_rejected(capital):
    with pytest.raises(InvalidInputError) as info:
        apply_shocks({"A": 1.0}, capital, {"A": -0.1})
    assert info.value.code == "invalid_capital"


def test_empty_shocks_are_rejected():
    with pytest.raises(InvalidInputError) as info:
        apply_shocks({"A": 1.0}, 1_000.0, {})
    assert info.value.code == "no_shocks"


def test_empty_portfolio_is_rejected():
    with pytest.raises(InvalidWeightsError):
        apply_shocks({}, 1_000.0, {"A": -0.1})


def test_unknown_shock_symbols_are_rejected():
    with pytest.raises(InvalidInputError) as info:
        apply_shocks({"A": 1.0}, 1_000.0, {"ZZZ": -0.1})
    assert info.value.code == "unknown_shock_symbols"
    assert info.value.params["symbols"] == ["ZZZ"]


@pytest.mark.parametrize("pct", [-1.5, 5.05, 100.0])
def test_out_of_range_shocks_are_rejected(pct):
    with pytest.raises(InvalidInputError) as info:
        apply_shocks({"A": 1.0}, 1_000.0, {"A": pct})
    assert info.value.code == "shock_out_of_range"


def test_non_finite_shock_is_rejected():
    with pytest.raises(InvalidInputError) as info:
        apply_shocks({"A": 1.0}, 1_000.0, {"A": np.nan})
    assert info.value.code == "non_finite_shock"


# ---------------------------------------------------------------------------
# Groups
# ---------------------------------------------------------------------------
def test_group_shocks_are_additive_for_overlapping_assets():
    rules = [
        ShockRule("equity", "stress.group.equity", -0.10, ("A", "B")),
        ShockRule("tech", "stress.group.tech", -0.15, ("B",)),
    ]
    result = apply_group_shocks({"A": 0.5, "B": 0.5}, 100_000.0, rules)
    assert result.shocks["A"] == pytest.approx(-0.10)
    assert result.shocks["B"] == pytest.approx(-0.25)
    assert result.per_asset.loc["B", "impact"] == pytest.approx(-0.25 * 50_000.0)
    assert len(result.rules) == 2


def test_group_shocks_can_override():
    rules = [
        ShockRule("equity", "stress.group.equity", -0.10, ("A", "B")),
        ShockRule("tech", "stress.group.tech", -0.15, ("B",)),
    ]
    result = apply_group_shocks({"A": 0.5, "B": 0.5}, 100_000.0, rules, combine="override")
    assert result.shocks["B"] == pytest.approx(-0.15)


def test_group_shocks_ignore_assets_outside_the_portfolio():
    rules = [ShockRule("equity", "stress.group.equity", -0.10, ("A", "ZZZ"))]
    result = apply_group_shocks({"A": 1.0}, 1_000.0, rules)
    assert result.shocks["A"] == pytest.approx(-0.10)


def test_group_shocks_report_rules_without_targets():
    rules = [
        ShockRule("equity", "stress.group.equity", -0.10, ("A",)),
        ShockRule("fx", "stress.group.fx", -0.05, ("ZZZ",)),
    ]
    result = apply_group_shocks({"A": 1.0}, 1_000.0, rules)
    assert result.notes == ("stress.no_target:fx",)


def test_group_shocks_reject_unknown_mode():
    rules = [ShockRule("equity", "stress.group.equity", -0.10, ("A",))]
    with pytest.raises(InvalidInputError) as info:
        apply_group_shocks({"A": 1.0}, 1_000.0, rules, combine="magic")
    assert info.value.code == "invalid_combination_mode"


def test_group_shocks_require_at_least_one_applicable_rule():
    rules = [ShockRule("fx", "stress.group.fx", -0.05, ("ZZZ",))]
    with pytest.raises(InvalidInputError) as info:
        apply_group_shocks({"A": 1.0}, 1_000.0, rules)
    assert info.value.code == "no_applicable_shocks"


def test_group_shocks_skip_zero_pct_rules():
    rules = [
        ShockRule("equity", "stress.group.equity", 0.0, ("A",)),
        ShockRule("tech", "stress.group.tech", -0.10, ("A",)),
    ]
    result = apply_group_shocks({"A": 1.0}, 1_000.0, rules)
    assert result.shocks["A"] == pytest.approx(-0.10)


def test_shock_from_group_covers_every_symbol():
    shocks = shock_from_group({"A": 0.5, "B": 0.5}, -0.2)
    assert shocks == {"A": -0.2, "B": -0.2}
    with pytest.raises(InvalidInputError):
        shock_from_group({"A": 1.0}, -2.0)
