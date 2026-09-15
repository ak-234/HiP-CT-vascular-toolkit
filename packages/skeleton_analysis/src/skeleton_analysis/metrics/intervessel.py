"""Inter-vessel distance: nearest centre-to-centre distance between segments.

Port of ``intervessel_distance.m``. For each edge we find its arc-length
midpoint (the point closest to half the segment's total length) and then, for
every edge, the minimum distance to any *other* edge's midpoint.

The MATLAB code did an O(nE^2) brute-force min; here we use a KD-tree
(:class:`scipy.spatial.cKDTree`) for the same result in O(nE log nE).
"""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from skeleton_analysis.io.amira import SpatialGraph


def edge_midpoints(graph: SpatialGraph) -> np.ndarray:
    """Arc-length midpoint coordinate of every edge, shape ``(n_edges, 3)``."""
    nump = np.asarray(graph.num_edge_points, dtype=np.int64)
    pcoords = np.asarray(graph.point_coords, dtype=float)
    starts = np.concatenate([[0], np.cumsum(nump)[:-1]])

    mids = np.zeros((len(nump), 3), dtype=float)
    for i, (s, n) in enumerate(zip(starts, nump)):
        pts = pcoords[s : s + n]
        if n <= 1:
            mids[i] = pts[0] if n == 1 else np.nan
            continue
        seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
        cum = np.concatenate([[0.0], np.cumsum(seg)])  # arc length to each point
        half = cum[-1] / 2.0
        k = int(np.argmin(np.abs(cum - half)))  # point nearest the mid-arc-length
        mids[i] = pts[k]
    return mids


def intervessel_distance(graph: SpatialGraph) -> np.ndarray:
    """Nearest centre-to-centre distance from each edge to any other edge.

    Returns an ``(n_edges,)`` array. Edges with a single midpoint that coincides
    with another still report the true nearest-neighbour distance.
    """
    mids = edge_midpoints(graph)
    n = len(mids)
    if n < 2:
        return np.zeros(n, dtype=float)
    tree = cKDTree(mids)
    # k=2: first neighbour is the point itself (distance 0), second is the nearest other.
    dist, _idx = tree.query(mids, k=2)
    return dist[:, 1]
