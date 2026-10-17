"""Correlation and co-movement analytics.

Formulas
--------
* Pearson correlation between assets ``i`` and ``j``:
      rho_ij = cov(r_i, r_j) / (sigma_i * sigma_j)
* Average pairwise correlation: mean of the strictly upper triangular part of
  the correlation matrix (the diagonal of 1s is excluded).
* Diversification ratio (Choueifaty & Coignard):
      DR = (sum_i w_i * sigma_i) / sigma_p        (>= 1 for imperfectly correlated assets)
  A ratio of 1 means no diversification benefit at all.
* Covariance matrix: sample covariance (``ddof=1``) of the aligned daily
  returns, annualised by ``periods_per_year`` when requested.

Missing data: pair-wise deletion is used by pandas' ``corr`` (each pair uses the
observations available for that pair) and the number of observations actually
used is returned alongside the matrix so the UI can flag weak estimates.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
import pandas as pd

from analytics.exceptions import (
    InsufficientDataError,
    InvalidInputError,
    InvalidWeightsError,
)
from analytics.risk import portfolio_volatility
from analytics.returns import _check_finite, _require_numeric
from config import TRADING_DAYS_PER_YEAR

__all__ = [
    "CorrelationResult",
    "correlation_matrix",
    "average_pairwise_correlation",
    "most_correlated_pairs",
    "least_correlated_pairs",
    "annualized_covariance",
    "covariance_matrix",
    "diversification_ratio",
    "observation_counts",
]


@dataclass(frozen=True)
class CorrelationResult:
    """Correlation matrix plus the observation counts used to build it."""

    matrix: pd.DataFrame
    observations: pd.DataFrame  #: pairwise number of aligned observations
    average: float
    min_pair_observations: int
    n_assets: int


def _validate_returns_frame(asset_returns: pd.DataFrame, *, minimum_assets: int = 2) -> None:
    if asset_returns is None or asset_returns.empty:
        raise InsufficientDataError(
            "No return history available for correlation analysis.",
            code="insufficient_observations",
            params={"available": 0, "required": 2, "what": "correlation"},
        )
    _require_numeric(asset_returns, "asset_returns")
    _check_finite(asset_returns, "asset_returns")
    if asset_returns.shape[1] < minimum_assets:
        raise InsufficientDataError(
            "At least two assets are required to compute correlations.",
            code="insufficient_assets",
            params={"available": int(asset_returns.shape[1]), "required": minimum_assets, "what": "correlation"},
        )


def observation_counts(asset_returns: pd.DataFrame) -> pd.DataFrame:
    """Pairwise count of non-missing overlapping observations per asset pair."""
    frame = asset_returns.notna().astype("int64")
    counts = frame.T.dot(frame)
    return counts


def correlation_matrix(
    asset_returns: pd.DataFrame,
    *,
    method: str = "pearson",
    min_periods: int = 2,
) -> pd.DataFrame:
    """Correlation matrix of asset returns (pairwise complete observations).

    A pair with fewer than ``min_periods`` overlapping observations produces
    ``NaN`` (flagged as unreliable by the UI) instead of a fabricated number.
    """
    _validate_returns_frame(asset_returns)
    if method not in {"pearson", "spearman", "kendall"}:
        raise InvalidInputError(
            f"Unsupported correlation method '{method}'.",
            code="invalid_correlation_method",
            params={"method": method},
        )
    matrix = asset_returns.corr(method=method, min_periods=min_periods)
    if matrix.isna().all().all():
        raise InsufficientDataError(
            "No overlapping observations: correlation cannot be computed.",
            code="no_overlapping_observations",
            params={"available": 0, "required": min_periods, "what": "correlation"},
        )
    return matrix


def correlation_details(
    asset_returns: pd.DataFrame,
    *,
    method: str = "pearson",
    min_periods: int = 2,
) -> CorrelationResult:
    """Correlation matrix + average pairwise correlation + observation counts."""
    matrix = correlation_matrix(asset_returns, method=method, min_periods=min_periods)
    counts = observation_counts(asset_returns)
    upper = matrix.where(np.triu(np.ones(matrix.shape), k=1).astype(bool))
    stacked = upper.stack()
    average = float(stacked.mean()) if not stacked.empty else float("nan")
    if counts.shape[0] > 1:
        mask = np.triu(np.ones(counts.shape), k=1).astype(bool)
        min_obs = int(counts.where(mask).min().min())
    else:  # pragma: no cover - guarded by _validate_returns_frame
        min_obs = int(counts.min().min())
    return CorrelationResult(
        matrix=matrix,
        observations=counts,
        average=average,
        min_pair_observations=min_obs,
        n_assets=int(matrix.shape[0]),
    )


def average_pairwise_correlation(correlation: pd.DataFrame) -> float:
    """Mean of the off-diagonal correlations (``NaN`` for a single asset)."""
    if correlation is None or correlation.empty:
        raise InvalidInputError("Correlation matrix is empty.", code="empty_correlation")
    if correlation.shape[0] < 2:
        return float("nan")
    upper = correlation.where(np.triu(np.ones(correlation.shape), k=1).astype(bool))
    stacked = upper.stack()
    return float(stacked.mean()) if not stacked.empty else float("nan")


def _ranked_pairs(correlation: pd.DataFrame, n: int, *, highest: bool) -> list[tuple[str, str, float]]:
    if correlation is None or correlation.empty or correlation.shape[0] < 2:
        return []
    upper = correlation.where(np.triu(np.ones(correlation.shape), k=1).astype(bool))
    pairs = [
        (str(a), str(b), float(value))
        for (a, b), value in upper.stack().items()
        if np.isfinite(value)
    ]
    pairs.sort(key=lambda item: item[2], reverse=highest)
    return pairs[:max(0, int(n))]


def most_correlated_pairs(
    correlation: pd.DataFrame, n: int = 3
) -> list[tuple[str, str, float]]:
    """The ``n`` most positively correlated asset pairs."""
    return _ranked_pairs(correlation, n, highest=True)


def least_correlated_pairs(
    correlation: pd.DataFrame, n: int = 3
) -> list[tuple[str, str, float]]:
    """The ``n`` least correlated (best diversifying) asset pairs."""
    return _ranked_pairs(correlation, n, highest=False)


def covariance_matrix(
    asset_returns: pd.DataFrame, *, periods_per_year: int | None = None, ddof: int = 1
) -> pd.DataFrame:
    """Sample covariance of the asset returns (optionally annualised)."""
    _validate_returns_frame(asset_returns)
    covariance = asset_returns.cov(ddof=ddof)
    if periods_per_year is not None:
        if periods_per_year <= 0:
            raise InvalidInputError(
                "periods_per_year must be positive.",
                code="invalid_periods_per_year",
                params={"value": periods_per_year},
            )
        covariance = covariance * periods_per_year
    return covariance


def annualized_covariance(
    asset_returns: pd.DataFrame, *, periods_per_year: int = TRADING_DAYS_PER_YEAR
) -> pd.DataFrame:
    """Annualised covariance matrix (``Sigma_daily * periods_per_year``)."""
    return covariance_matrix(asset_returns, periods_per_year=periods_per_year)


def diversification_ratio(
    weights: Mapping[str, float] | pd.Series,
    asset_returns: pd.DataFrame,
    *,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
) -> float:
    """``(sum_i w_i sigma_i) / sigma_p`` — 1.0 means zero diversification.

    Uses the **aligned** observations so that every weight refers to the same
    return frame; the weights are renormalised over the returned frame.
    """
    from analytics.returns import align_returns_to_weights  # local import: avoid cycles

    frame, aligned = align_returns_to_weights(asset_returns, weights)
    if len(frame) < 2:
        raise InsufficientDataError(
            "At least 2 aligned observations are required for the diversification ratio.",
            code="insufficient_observations",
            params={"available": int(len(frame)), "required": 2, "what": "diversification ratio"},
        )
    covariance = annualized_covariance(frame, periods_per_year=periods_per_year)
    portfolio_sigma = portfolio_volatility(aligned, covariance)
    if not np.isfinite(portfolio_sigma) or portfolio_sigma <= 0:
        return float("nan")
    individual_sigma = frame.std(ddof=1) * np.sqrt(periods_per_year)
    weighted_average_sigma = float((aligned * individual_sigma.reindex(aligned.index)).sum())
    return float(weighted_average_sigma / portfolio_sigma)
