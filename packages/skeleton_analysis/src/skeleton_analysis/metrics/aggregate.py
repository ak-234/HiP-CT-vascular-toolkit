"""Aggregate per-segment metrics by Strahler order and plot distributions.

A focused, reusable re-working of ``Graphs_strahler_against_metrics.m``. The
MATLAB script hard-coded absolute CSV paths for three connected components
(cc1/cc4/cc9) and drove bespoke violin plots via the third-party
``al_goodplot.m``. Here we provide library functions that operate on a tidy
pandas DataFrame (one row per vessel segment, with a Strahler-order column and
any number of metric columns), so callers assemble their own inputs.

Plotting depends on the optional ``[viz]`` extra (matplotlib + seaborn) and is
imported lazily so the core package has no hard viz dependency.
"""

from __future__ import annotations

from typing import Dict, Iterable, Optional

import numpy as np
import pandas as pd


def aggregate_by_strahler(
    df: pd.DataFrame,
    value_columns: Iterable[str],
    order_column: str = "strahler",
) -> pd.DataFrame:
    """Per-Strahler-order summary statistics for the given metric columns.

    Returns a DataFrame indexed by Strahler order with ``<col>_mean``,
    ``<col>_std``, ``<col>_median`` and ``count`` columns. NaNs are ignored
    (``numpy.nanmean``/``nanstd``, replacing MATLAB's Statistics-Toolbox
    ``nanmean``/``nanstd``).
    """
    value_columns = list(value_columns)

    def _nanmean(s):
        a = s.to_numpy(dtype=float)
        return np.nanmean(a) if np.any(~np.isnan(a)) else np.nan

    def _nanstd(s):
        a = s.to_numpy(dtype=float)
        a = a[~np.isnan(a)]
        return np.std(a, ddof=1) if a.size > 1 else np.nan  # sample std; NaN if <2

    def _nanmedian(s):
        a = s.to_numpy(dtype=float)
        return np.nanmedian(a) if np.any(~np.isnan(a)) else np.nan

    agg: Dict[str, tuple] = {}
    for col in value_columns:
        agg[f"{col}_mean"] = (col, _nanmean)
        agg[f"{col}_std"] = (col, _nanstd)
        agg[f"{col}_median"] = (col, _nanmedian)
    grouped = df.groupby(order_column).agg(**agg)
    grouped["count"] = df.groupby(order_column).size()
    return grouped.sort_index()


def branching_ratio(counts_by_order: pd.Series) -> pd.Series:
    """Ratio of segment counts between successive Strahler orders (N_k / N_{k+1}).

    ``counts_by_order`` maps Strahler order -> number of segments. The classic
    bifurcation ratio is the number of order-*k* vessels divided by the number of
    order-*(k+1)* vessels.
    """
    counts = counts_by_order.sort_index()
    orders = counts.index.to_numpy()
    ratios = {}
    for k in orders[:-1]:
        upper = counts.get(k + 1, np.nan)
        ratios[k] = counts[k] / upper if upper else np.nan
    return pd.Series(ratios, name="branching_ratio")


def violin_by_strahler(
    df: pd.DataFrame,
    value_column: str,
    order_column: str = "strahler",
    ax=None,
    **kwargs,
):
    """Violin plot of ``value_column`` grouped by Strahler order.

    Replacement for ``al_goodplot.m``. Requires the ``[viz]`` extra.
    """
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError as exc:  # pragma: no cover - depends on optional extra
        raise ImportError(
            "violin_by_strahler needs the [viz] extra: pip install "
            "'skeleton_analysis[viz]'"
        ) from exc

    if ax is None:
        _fig, ax = plt.subplots()
    sub = df[[order_column, value_column]].dropna()
    sns.violinplot(data=sub, x=order_column, y=value_column, ax=ax, **kwargs)
    ax.set_xlabel("Strahler order")
    ax.set_ylabel(value_column)
    return ax
