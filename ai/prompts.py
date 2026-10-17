"""Prompt engineering for the AI financial assistant.

Core principle
--------------
**Python computes, the LLM explains.** The context block built here contains
only figures produced by the analytics engine (:mod:`services.engine`,
:mod:`analytics`, :mod:`optimization`, :mod:`analytics.stress`). The system
prompt forbids the model from inventing prices, statistics, market observations
or optimisation outputs, from predicting returns and from giving personalised
investment advice.

Everything in this module is pure (no network, no Streamlit) so the context can
be unit-tested.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

from config import APP_NAME
from utils.i18n import translate

__all__ = [
    "LANGUAGES",
    "build_context",
    "render_context",
    "system_prompt",
    "build_messages",
    "QUICK_PROMPT_KEYS",
    "quick_prompt",
]

#: Language codes supported by the assistant (mirrors the interface).
LANGUAGES = {"en": "English", "fr": "French"}

#: Suggested questions offered as one-click buttons on the assistant page.
QUICK_PROMPT_KEYS: tuple[str, ...] = (
    "ai.quick.metrics",
    "ai.quick.risk",
    "ai.quick.diversification",
    "ai.quick.correlation",
    "ai.quick.optimization",
    "ai.quick.stress",
    "ai.quick.summary",
)

_FOCUS: dict[str, dict[str, str]] = {
    "ai.quick.metrics": {
        "en": "Focus on the return, volatility, Sharpe ratio and maximum drawdown figures.",
        "fr": "Concentrez-vous sur le rendement, la volatilité, le ratio de Sharpe et la perte maximale.",
    },
    "ai.quick.risk": {
        "en": "Focus on volatility, tail risk (VaR/CVaR), drawdown and concentration.",
        "fr": "Concentrez-vous sur la volatilité, le risque extrême (VaR/CVaR), les replis et la concentration.",
    },
    "ai.quick.diversification": {
        "en": "Focus on concentration metrics, the effective number of assets and the diversification ratio.",
        "fr": "Concentrez-vous sur les mesures de concentration, le nombre effectif d'actifs et le ratio de diversification.",
    },
    "ai.quick.correlation": {
        "en": "Focus on the average correlation and the most / least correlated pairs.",
        "fr": "Concentrez-vous sur la corrélation moyenne et les paires les plus et les moins corrélées.",
    },
    "ai.quick.optimization": {
        "en": "Compare the current weights with the optimised weights and the resulting metrics, and restate the caveats.",
        "fr": "Comparez les pondérations actuelles et optimisées ainsi que les indicateurs associés, et rappelez les limites.",
    },
    "ai.quick.stress": {
        "en": "Summarise the hypothetical scenarios, their assumptions and their estimated monetary impact.",
        "fr": "Résumez les scénarios hypothétiques, leurs hypothèses et leur impact monétaire estimé.",
    },
    "ai.quick.summary": {
        "en": "Provide a structured review: performance, risk, diversification, benchmark, data quality and the main caveats.",
        "fr": "Fournissez une revue structurée : performance, risque, diversification, indice de référence, qualité des données et principales limites.",
    },
}

_SYSTEM_EN = """You are the financial assistant of {app}, an educational portfolio analytics application.
You explain portfolio metrics that have ALREADY been computed by the application.

Absolute rules:
1. Use ONLY the figures in the provided context block. They come from a tested analytics engine.
2. Never invent prices, returns, statistics, market observations, dates, tickers, indices or optimisation results.
   If a figure is missing or shown as "n/a", say it could not be computed and explain which data would be needed.
3. Never predict future returns, never give price targets, and never give personalised investment advice
   (no "buy", "sell" or "you should invest" recommendations).
4. Do not perform your own return, risk, correlation or optimisation calculations: refer to the provided numbers instead.
5. If the data are simulated (demo mode), state it explicitly once in your answer.
6. Be concise and pedagogical: short paragraphs or bullets, plain language, define the metric before interpreting it.
7. End with a one-line reminder that this is an educational analysis, not investment advice.

Answer in {language}. Use the currency codes exactly as they appear in the context."""

_SYSTEM_FR = """Vous êtes l'assistant financier de {app}, une application pédagogique d'analyse de portefeuille.
Vous expliquez des indicateurs DÉJÀ calculés par l'application.

Règles absolues :
1. Utilisez UNIQUEMENT les chiffres du bloc de contexte fourni. Ils proviennent d'un moteur d'analyse testé.
2. N'inventez jamais de prix, rendements, statistiques, observations de marché, dates, titres, indices ou résultats d'optimisation.
   Si un chiffre est absent ou indiqué « n/d », dites qu'il n'a pas pu être calculé et expliquez les données nécessaires.
3. Ne prévoyez jamais de rendement futur, ne donnez pas d'objectif de cours et ne prodiguez aucun conseil
   d'investissement personnalisé (pas de « achetez », « vendez » ou « vous devriez investir »).
4. N'effectuez pas vos propres calculs de rendement, de risque, de corrélation ou d'optimisation :
   référez-vous aux chiffres fournis.
5. Si les données sont simulées (mode démo), précisez-le une fois explicitement.
6. Soyez concis et pédagogique : paragraphes courts ou puces, langage simple, définissez l'indicateur avant de l'interpréter.
7. Terminez par un rappel d'une ligne indiquant qu'il s'agit d'une analyse pédagogique, sans valeur de conseil.

Répondez en {language}. Reprenez les codes de devise exactement tels qu'ils figurent dans le contexte."""


def system_prompt(language: str = "en") -> str:
    """System prompt in the requested language (falls back to English)."""
    template = _SYSTEM_FR if str(language).lower().startswith("fr") else _SYSTEM_EN
    return template.format(app=APP_NAME, language=LANGUAGES.get(str(language)[:2], "English"))


def quick_prompt(key: str, language: str = "en", **params: Any) -> str:
    """Build the user question for a suggested prompt button."""
    label = translate(key, language, **params)
    focus = _FOCUS.get(key, {}).get("fr" if str(language).startswith("fr") else "en", "")
    return f"{label}\n{focus}".strip()


# ---------------------------------------------------------------------------
# Context building
# ---------------------------------------------------------------------------
def _clean(value: Any) -> Any:
    """Convert numpy/pandas scalars into JSON-friendly values."""
    if value is None:
        return None
    if isinstance(value, (np.floating,)):
        numeric = float(value)
        return round(numeric, 6) if np.isfinite(numeric) else None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, float):
        return round(value, 6) if np.isfinite(value) else None
    if isinstance(value, (list, tuple)):
        return [_clean(item) for item in value]
    if isinstance(value, Mapping):
        return {str(key): _clean(item) for key, item in value.items()}
    return value


def _pct(value: Any) -> str:
    if value is None:
        return "n/a"
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return "n/a"
    if not np.isfinite(numeric):
        return "n/a"
    return f"{numeric * 100:.2f}%"


def _num(value: Any, decimals: int = 3) -> str:
    if value is None:
        return "n/a"
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return "n/a"
    if not np.isfinite(numeric):
        return "n/a"
    return f"{numeric:.{decimals}f}"


def build_context(
    bundle,
    *,
    optimization=None,
    stress=None,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Collect the computed figures attached to every assistant question.

    Accepts an :class:`~services.engine.AnalyticsBundle` (or ``None``) plus
    optional optimisation and stress-test results. Returns a plain dictionary so
    the payload stays inspectable and testable.
    """
    context: dict[str, Any] = {
        "application": APP_NAME,
        "disclaimer": "Educational analysis based on historical or simulated data. Not investment advice.",
    }
    if bundle is None:
        context["data"] = {"available": False}
        return _clean(context)

    context["data"] = {
        "available": bool(bundle.ok),
        "market": bundle.market.value,
        "data_mode": bundle.data_mode.value,
        "is_simulated": bool(bundle.is_simulated),
        "source": bundle.data_source_label,
        "window_start": str(bundle.first_observation.date()) if bundle.first_observation is not None else None,
        "window_end": str(bundle.last_observation.date()) if bundle.last_observation is not None else None,
        "observations": int(bundle.n_observations),
        "currencies": list(bundle.portfolio.currencies),
        "assets_used": list(bundle.available_symbols),
        "assets_unavailable": list(bundle.missing_symbols),
        "data_warnings": [
            {"code": notice.code, "params": notice.params}
            for notice in bundle.notices
            if notice.level in {"warning", "error"}
        ],
    }

    portfolio = {
        "capital": bundle.capital,
        "current_value": bundle.current_value,
        "cumulative_return": bundle.cumulative_return,
        "base_currency": bundle.currency,
        "weights": {
            symbol: round(float(weight), 6) for symbol, weight in bundle.weights.items()
        },
        "asset_names": {
            symbol: instrument.display_name("en")
            for symbol, instrument in bundle.portfolio.assets.items()
            if symbol in bundle.weights.index
        },
        "asset_classes": {
            symbol: instrument.asset_class.value
            for symbol, instrument in bundle.portfolio.assets.items()
            if symbol in bundle.weights.index
        },
        "allocation_values": {
            symbol: float(value) for symbol, value in bundle.allocation.items()
        },
    }
    if bundle.currency_totals is not None:
        portfolio["allocation_by_currency"] = {
            code: float(value) for code, value in bundle.currency_totals.totals.items()
        }
        portfolio["combined_total"] = bundle.currency_totals.combined_total
        portfolio["combined_currency"] = bundle.currency_totals.combined_currency
    context["portfolio"] = portfolio

    metrics = {
        "total_return": bundle.metric("total_return"),
        "annualized_return": bundle.metric("annualized_return"),
        "annualized_volatility": bundle.metric("annualized_volatility"),
        "sharpe_ratio": bundle.metric("sharpe_ratio"),
        "sortino_ratio": bundle.metric("sortino_ratio"),
        "max_drawdown": bundle.metric("max_drawdown"),
        "value_at_risk_95_daily": bundle.metric("value_at_risk_95"),
        "expected_shortfall_95_daily": bundle.metric("conditional_value_at_risk_95"),
        "best_day": bundle.metric("best_day"),
        "worst_day": bundle.metric("worst_day"),
        "positive_days_share": bundle.metric("positive_days_share"),
        "n_observations": bundle.metric("n_observations"),
        "risk_free_rate": bundle.settings.risk_free_rate,
    }
    context["metrics"] = metrics

    if bundle.benchmark_metrics:
        context["benchmark"] = {
            "symbol": bundle.portfolio.benchmark,
            "name": (
                bundle.portfolio.assets[bundle.portfolio.benchmark].display_name("en")
                if bundle.portfolio.benchmark in bundle.portfolio.assets
                else bundle.portfolio.benchmark
            ),
            **{
                key: value
                for key, value in {
                    "annualized_return": bundle.benchmark_metrics.get("benchmark_annualized_return"),
                    "portfolio_annualized_return": bundle.benchmark_metrics.get("portfolio_annualized_return"),
                    "excess_annualized_return": bundle.benchmark_metrics.get("excess_annualized_return"),
                    "beta": bundle.benchmark_metrics.get("beta"),
                    "alpha": bundle.benchmark_metrics.get("alpha"),
                    "tracking_error": bundle.benchmark_metrics.get("tracking_error"),
                    "information_ratio": bundle.benchmark_metrics.get("information_ratio"),
                    "correlation": bundle.benchmark_metrics.get("correlation"),
                    "observations": bundle.benchmark_metrics.get("n_observations"),
                }.items()
            },
        }

    if bundle.concentration is not None:
        context["concentration"] = {
            "hhi": bundle.concentration.hhi,
            "normalized_hhi": bundle.concentration.normalized_hhi,
            "effective_number_of_assets": bundle.concentration.effective_n,
            "largest_position": bundle.concentration.top1_weight,
            "top_three_positions": bundle.concentration.top3_weight,
            "level": bundle.concentration.label,
            "notes": list(bundle.concentration.warnings),
        }

    if bundle.correlation is not None:
        from analytics.correlation import least_correlated_pairs, most_correlated_pairs

        context["correlation"] = {
            "average_pairwise": bundle.average_correlation,
            "min_pair_observations": bundle.correlation.min_pair_observations,
            "most_correlated_pairs": most_correlated_pairs(bundle.correlation.matrix, 4),
            "least_correlated_pairs": least_correlated_pairs(bundle.correlation.matrix, 4),
            "matrix": bundle.correlation.matrix.round(4).to_dict(),
        }

    context["diversification"] = {
        "diversification_ratio": bundle.diversification_ratio,
        "risk_contributions": (
            {str(symbol): float(value) for symbol, value in bundle.risk_contributions.items()}
            if bundle.risk_contributions is not None
            else None
        ),
    }

    if optimization is not None:
        context["optimization"] = {
            "method": getattr(optimization, "method", None),
            "engine": getattr(optimization, "engine", None),
            "weights": {
                str(symbol): float(value) for symbol, value in getattr(optimization, "weights", {}).items()
            },
            "expected_return": getattr(optimization, "expected_return", None),
            "volatility": getattr(optimization, "volatility", None),
            "sharpe": getattr(optimization, "sharpe", None),
            "observations": getattr(optimization, "n_observations", None),
            "window": getattr(optimization, "price_range", (None, None)),
            "excluded_assets": getattr(optimization, "dropped_assets", {}),
            "caveat": (
                "In-sample historical estimate. Weights depend on the estimation window and the constraints; "
                "they do not guarantee better future performance."
            ),
        }

    if stress is not None:
        context["stress_test"] = {
            "value_before": getattr(stress, "value_before", None),
            "value_after": getattr(stress, "value_after", None),
            "total_impact": getattr(stress, "total_impact", None),
            "impact_percent": getattr(stress, "impact_pct", None),
            "shocks": getattr(stress, "shocks", {}),
            "per_asset": (
                {
                    str(symbol): {
                        "weight": float(row["weight"]),
                        "shock": float(row["shock"]),
                        "impact": float(row["impact"]),
                    }
                    for symbol, row in getattr(stress, "per_asset", {}).iterrows()
                }
                if hasattr(stress, "per_asset")
                else None
            ),
            "caveat": "Hypothetical scenario, not a forecast and not a probability estimate.",
        }

    if extra:
        context["additional"] = _clean(dict(extra))
    return _clean(context)


def render_context(context: Mapping[str, Any]) -> str:
    """Render the context dictionary as a compact, readable text block."""
    lines: list[str] = ["=== COMPUTED DATA (authoritative, produced by the application) ==="]

    data = context.get("data") or {}
    lines.append("-- Data --")
    lines.append(f"available: {data.get('available')}")
    lines.append(f"market: {data.get('market')} | mode: {data.get('data_mode')} | simulated: {data.get('is_simulated')}")
    lines.append(f"source: {data.get('source')} | window: {data.get('window_start')} -> {data.get('window_end')}")
    lines.append(f"observations: {data.get('observations')} | currencies: {data.get('currencies')}")
    lines.append(f"assets used: {data.get('assets_used')}")
    if data.get("assets_unavailable"):
        lines.append(f"assets unavailable: {data.get('assets_unavailable')}")
    for warning_item in data.get("data_warnings") or []:
        lines.append(f"data warning: {warning_item.get('code')} {warning_item.get('params')}")

    portfolio = context.get("portfolio") or {}
    lines.append("-- Portfolio --")
    lines.append(f"capital: {portfolio.get('capital')} {portfolio.get('base_currency')}")
    lines.append(f"current value: {portfolio.get('current_value')} | cumulative return: {_pct(portfolio.get('cumulative_return'))}")
    weights = portfolio.get("weights") or {}
    names = portfolio.get("asset_names") or {}
    for symbol, weight in sorted(weights.items(), key=lambda item: -float(item[1])):
        value = (portfolio.get("allocation_values") or {}).get(symbol)
        lines.append(
            f"  {symbol} ({names.get(symbol, symbol)}): weight {_pct(weight)}, value {value}"
        )
    if portfolio.get("allocation_by_currency"):
        lines.append(f"allocation by currency: {portfolio.get('allocation_by_currency')}")
        if portfolio.get("combined_total") is not None:
            lines.append(
                f"combined total: {portfolio.get('combined_total')} {portfolio.get('combined_currency')}"
            )

    metrics = context.get("metrics") or {}
    lines.append("-- Portfolio metrics (annualised unless stated) --")
    lines.append(f"total return: {_pct(metrics.get('total_return'))}")
    lines.append(f"annualised return: {_pct(metrics.get('annualized_return'))}")
    lines.append(f"annualised volatility: {_pct(metrics.get('annualized_volatility'))}")
    lines.append(f"Sharpe ratio (rf={_pct(metrics.get('risk_free_rate'))}): {_num(metrics.get('sharpe_ratio'), 2)}")
    lines.append(f"Sortino ratio: {_num(metrics.get('sortino_ratio'), 2)}")
    lines.append(f"maximum drawdown: {_pct(metrics.get('max_drawdown'))}")
    lines.append(f"daily VaR 95%: {_pct(metrics.get('value_at_risk_95_daily'))} | daily expected shortfall 95%: {_pct(metrics.get('expected_shortfall_95_daily'))}")
    lines.append(f"best day: {_pct(metrics.get('best_day'))} | worst day: {_pct(metrics.get('worst_day'))}")
    lines.append(f"positive days: {_pct(metrics.get('positive_days_share'))} | observations: {metrics.get('n_observations')}")

    benchmark = context.get("benchmark")
    if benchmark:
        lines.append("-- Benchmark --")
        lines.append(
            f"{benchmark.get('symbol')}: annualised return {_pct(benchmark.get('annualized_return'))}, "
            f"excess return {_pct(benchmark.get('excess_annualized_return'))}, beta {_num(benchmark.get('beta'), 2)}, "
            f"alpha {_pct(benchmark.get('alpha'))}, tracking error {_pct(benchmark.get('tracking_error'))}, "
            f"information ratio {_num(benchmark.get('information_ratio'), 2)}, correlation {_num(benchmark.get('correlation'), 2)}"
        )

    concentration = context.get("concentration")
    if concentration:
        lines.append("-- Concentration --")
        lines.append(
            f"HHI {_num(concentration.get('hhi'), 4)} | normalised HHI {_num(concentration.get('normalized_hhi'), 3)} | "
            f"effective number of assets {_num(concentration.get('effective_number_of_assets'), 2)}"
        )
        lines.append(
            f"largest position {_pct(concentration.get('largest_position'))} | top three {_pct(concentration.get('top_three_positions'))} | "
            f"level: {concentration.get('level')}"
        )

    correlation = context.get("correlation")
    if correlation:
        lines.append("-- Correlation --")
        lines.append(f"average pairwise correlation: {_num(correlation.get('average_pairwise'), 3)}")
        for a, b, value in correlation.get("most_correlated_pairs") or []:
            lines.append(f"  most correlated: {a} / {b} = {_num(value, 3)}")
        for a, b, value in correlation.get("least_correlated_pairs") or []:
            lines.append(f"  least correlated: {a} / {b} = {_num(value, 3)}")

    diversification = context.get("diversification") or {}
    if diversification.get("diversification_ratio") is not None or diversification.get("risk_contributions"):
        lines.append("-- Diversification --")
        lines.append(f"diversification ratio: {_num(diversification.get('diversification_ratio'), 3)}")
        contributions = diversification.get("risk_contributions") or {}
        if contributions:
            top = sorted(contributions.items(), key=lambda item: -abs(float(item[1])))[:5]
            lines.append("largest risk contributions (share of portfolio volatility): " + ", ".join(
                f"{symbol} {_pct(value)}" for symbol, value in top
            ))

    optimization = context.get("optimization")
    if optimization:
        lines.append("-- Optimisation (in-sample) --")
        lines.append(
            f"method {optimization.get('method')} | engine {optimization.get('engine')} | observations {optimization.get('observations')}"
        )
        for symbol, weight in sorted((optimization.get("weights") or {}).items(), key=lambda item: -float(item[1])):
            lines.append(f"  {symbol}: {_pct(weight)}")
        lines.append(
            f"historical annualised return {_pct(optimization.get('expected_return'))} | volatility {_pct(optimization.get('volatility'))} | "
            f"Sharpe {_num(optimization.get('sharpe'), 2)}"
        )
        lines.append(f"caveat: {optimization.get('caveat')}")

    stress = context.get("stress_test")
    if stress:
        lines.append("-- Stress test (hypothetical) --")
        lines.append(f"shocks applied: {stress.get('shocks')}")
        lines.append(
            f"value before {stress.get('value_before')} | impact {stress.get('total_impact')} "
            f"({_pct(stress.get('impact_percent'))}) | value after {stress.get('value_after')}"
        )
        lines.append("scenario only: no probability is attached to these numbers.")

    additional = context.get("additional")
    if additional:
        lines.append("-- Additional context --")
        for key, value in additional.items():
            lines.append(f"{key}: {value}")

    lines.append("=== END OF COMPUTED DATA ===")
    return "\n".join(lines)


def build_messages(
    question: str,
    context: Mapping[str, Any] | None,
    *,
    language: str = "en",
    history: Sequence[Mapping[str, str]] | None = None,
    max_history: int = 6,
) -> list[dict[str, str]]:
    """Assemble the chat payload: system rules + context + recent turns + question."""
    messages: list[dict[str, str]] = [{"role": "system", "content": system_prompt(language)}]
    if context:
        messages.append(
            {
                "role": "system",
                "content": (
                    "Computed figures for the current portfolio. Use ONLY these numbers.\n\n"
                    + render_context(context)
                ),
            }
        )
    for entry in list(history or [])[-max_history:]:
        role = str(entry.get("role", "user"))
        content = str(entry.get("content", ""))
        if role in {"user", "assistant"} and content:
            messages.append({"role": role, "content": content})
    messages.append({"role": "user", "content": question.strip()})
    return messages
