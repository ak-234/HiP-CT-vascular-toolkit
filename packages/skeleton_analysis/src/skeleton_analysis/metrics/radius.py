"""Per-edge mean radius from per-point thickness.

Amira exports a per-edge ``MeanRadius`` attribute used by the Murray's-law and
exponent metrics. When a spatial graph carries only per-point ``thickness``
(radius), this helper reconstructs the per-edge mean by averaging the thickness
of the points belonging to each edge (using ``NumEdgePoints`` to slice the flat
point array).
"""

from __future__ import annotations

import numpy as np

from skeleton_analysis.io.amira import SpatialGraph


def mean_radius_per_edge(graph: SpatialGraph, thickness_field: str = "thickness") -> np.ndarray:
    """Return the mean point-thickness (radius) of every edge, shape ``(n_edges,)``."""
    nump = np.asarray(graph.num_edge_points, dtype=np.int64)
    thickness = np.asarray(graph.point_fields[thickness_field], dtype=float)
    starts = np.concatenate([[0], np.cumsum(nump)[:-1]])
    out = np.empty(len(nump), dtype=float)
    for i, (s, n) in enumerate(zip(starts, nump)):
        out[i] = float(np.mean(thickness[s : s + n])) if n > 0 else np.nan
    return out
