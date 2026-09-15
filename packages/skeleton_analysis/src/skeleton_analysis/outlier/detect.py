"""Collapsed-vessel outlier detection (pure numpy, no image dependencies).

Ports ``return_outlier.m`` and the outlier-flagging logic of
``Outliers_spatial_graph.m``, reworked to operate directly on a
:class:`~skeleton_analysis.io.amira.SpatialGraph` (as the MATLAB ``TODO`` note
intended) instead of Amira-exported CSVs. Implements MATLAB's percentile-based
``isoutlier`` / ``filloutliers`` on the low tail (collapsed vessels have
abnormally small radii).
"""

from __future__ import annotations

from typing import Iterable, Optional, Sequence, Tuple

import numpy as np

from skeleton_analysis.io.amira import SpatialGraph


def matlab_prctile(x, p):
    """MATLAB-compatible percentile(s).

    MATLAB ``prctile`` places the sorted samples at percentiles
    ``100*(i-0.5)/n`` and linearly interpolates between them, clamping requests
    outside that range to the min/max sample. ``p`` may be a scalar or array.
    """
    x = np.sort(np.asarray(x, dtype=float))
    n = x.size
    if n == 0:
        return np.nan if np.isscalar(p) else np.full(np.shape(p), np.nan)
    if n == 1:
        return float(x[0]) if np.isscalar(p) else np.full(np.shape(p), x[0])
    q = 100.0 * (np.arange(1, n + 1) - 0.5) / n
    return np.interp(p, q, x)  # np.interp clamps outside [q0, qn] -> matches MATLAB


def isoutlier_percentiles(x, lower: float, upper: float) -> np.ndarray:
    """Boolean mask of ``x`` values below ``lower``-pctile or above ``upper``-pctile.

    Equivalent to MATLAB ``isoutlier(x, "percentiles", [lower upper])``.
    """
    x = np.asarray(x, dtype=float)
    lo = matlab_prctile(x, lower)
    hi = matlab_prctile(x, upper)
    return (x < lo) | (x > hi)


def filloutliers_nearest(x, lower: float, upper: float) -> np.ndarray:
    """Replace percentile-outliers with the nearest (by index) non-outlier value.

    Equivalent to MATLAB ``filloutliers(x, "nearest", "percentiles", [lower upper])``.
    """
    x = np.array(x, dtype=float)
    mask = isoutlier_percentiles(x, lower, upper)
    good = np.flatnonzero(~mask)
    if good.size == 0 or not mask.any():
        return x
    for i in np.flatnonzero(mask):
        j = good[np.argmin(np.abs(good - i))]  # nearest good index (ties -> left)
        x[i] = x[j]
    return x


def along_segment_outliers(
    thickness, lower: float = 5.0, upper: float = 100.0
) -> Tuple[np.ndarray, np.ndarray]:
    """Find low-tail thickness outliers within one vessel segment.

    Port of ``return_outlier.m``. Returns ``(local_indices, replacement_values)``
    where ``local_indices`` index into ``thickness`` and ``replacement_values``
    are the nearest-neighbour fills.
    """
    thickness = np.asarray(thickness, dtype=float)
    mask = isoutlier_percentiles(thickness, lower, upper)
    filled = filloutliers_nearest(thickness, lower, upper)
    idx = np.flatnonzero(mask)
    return idx, filled[idx]


def _edge_point_slices(graph: SpatialGraph):
    nump = np.asarray(graph.num_edge_points, dtype=np.int64)
    starts = np.concatenate([[0], np.cumsum(nump)[:-1]])
    return starts, nump


def correct_along_segment_thickness(
    graph: SpatialGraph,
    lower: float = 5.0,
    upper: float = 100.0,
    thickness_field: str = "thickness",
) -> Tuple[np.ndarray, np.ndarray]:
    """Correct short collapsed stretches within every segment.

    For each edge, low-tail thickness outliers among its points are replaced with
    the nearest non-outlier value. Returns ``(corrected_thickness, changed_point_indices)``
    where ``corrected_thickness`` is a new per-point array (the input graph is not
    modified) and ``changed_point_indices`` are the global point indices altered.
    """
    thickness = np.asarray(graph.point_fields[thickness_field], dtype=float).copy()
    starts, nump = _edge_point_slices(graph)
    changed = []
    for s, n in zip(starts, nump):
        if n <= 0:
            continue
        seg = thickness[s : s + n]
        idx, repl = along_segment_outliers(seg, lower, upper)
        for li, rv in zip(idx, repl):
            thickness[s + li] = rv
            changed.append(int(s + li))
    return thickness, np.asarray(changed, dtype=np.int64)


def detect_collapsed_segments(
    graph: SpatialGraph,
    strahler_field: str = "strahler",
    radius_field: str = "MeanRadius",
    flag_orders: Sequence[int] = (6, 7, 8, 9),
    percentile_orders: Optional[Iterable[int]] = (5,),
    percentile: float = 10.0,
) -> np.ndarray:
    """Flag whole collapsed vessels (large radius outliers).

    Port of the ``genx_outliers`` logic in ``Outliers_spatial_graph.m``: all
    segments at Strahler order >= ``flag_orders`` are flagged outright, and for
    each order in ``percentile_orders`` the segments below the ``percentile``-th
    radius percentile (within that order) are flagged. Returns sorted edge indices.
    """
    strahler = np.asarray(graph.edge_fields[strahler_field]).astype(int)
    radius = np.asarray(graph.edge_fields[radius_field], dtype=float)

    flagged = set(np.flatnonzero(np.isin(strahler, list(flag_orders))).tolist())
    for order in percentile_orders or ():
        idx = np.flatnonzero(strahler == order)
        if idx.size == 0:
            continue
        mask = isoutlier_percentiles(radius[idx], percentile, 100.0)
        flagged.update(idx[mask].tolist())
    return np.array(sorted(flagged), dtype=np.int64)
