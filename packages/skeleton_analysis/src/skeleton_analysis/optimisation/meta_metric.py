"""Bifurcation-matching Dice and the combined skeleton-optimisation metric.

Port of ``meta_metric.m``. Bifurcation points (graph nodes with coordination
number > 2) of a candidate skeleton are matched to those of a binary
ground-truth graph by nearest neighbour (greedy, each ground-truth point used
once) within a distance threshold, yielding TP/FP/FN and a bifurcation Dice.

Fixes vs MATLAB
---------------
* ``bb_coords`` and file paths are real arguments (the MATLAB ``clear all`` on
  line 3 wiped the input argument).
* The bounding-box test ``isInBox`` had a mis-placed parenthesis that folded the
  y-max comparison into the z tests; here every axis is tested correctly.
* The combined ``Metric`` (normalised RMS of relative differences in Volume, CC,
  Euler number, branch-point count and clDice) is provided as a separate,
  fully-defined function :func:`meta_metric` rather than the commented-out,
  undefined-variable expression in the MATLAB source.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Optional, Sequence

import numpy as np

from skeleton_analysis.graph.neighbors import coordination_number
from skeleton_analysis.io.amira import SpatialGraph


def bifurcation_points(
    graph: SpatialGraph,
    bounding_box: Optional[Sequence[float]] = None,
    min_coordination: int = 3,
) -> np.ndarray:
    """Coordinates of branch points (coordination number >= ``min_coordination``).

    Parameters
    ----------
    bounding_box : optional [xmin, xmax, ymin, ymax, zmin, zmax]
        If given, only vertices inside the box are considered.
    """
    edges = np.asarray(graph.edge_connectivity, dtype=np.int64)
    coords = np.asarray(graph.vertex_coords, dtype=float)
    coord = coordination_number(edges)

    node_ids = np.arange(len(coords))
    if bounding_box is not None:
        x0, x1, y0, y1, z0, z1 = bounding_box
        in_box = (
            (coords[:, 0] >= x0) & (coords[:, 0] <= x1)
            & (coords[:, 1] >= y0) & (coords[:, 1] <= y1)
            & (coords[:, 2] >= z0) & (coords[:, 2] <= z1)
        )
        node_ids = node_ids[in_box]

    branch = [n for n in node_ids if coord.get(int(n), 0) >= min_coordination]
    return coords[branch] if branch else np.empty((0, 3))


@dataclass
class BifurcationDice:
    tp: int
    fp: int
    fn: int
    dice: float
    n_candidate: int
    n_reference: int


def bifurcation_dice_points(
    candidate_pts: np.ndarray,
    reference_pts: np.ndarray,
    threshold: float = 900.0,
) -> BifurcationDice:
    """Bifurcation-matching Dice between two sets of bifurcation coordinates.

    Greedy nearest-neighbour matching (each reference point used at most once)
    within ``threshold`` world units, reproducing the MATLAB ``knnsearch`` loop.
    This is the coordinate-level core used by :func:`bifurcation_dice`; it also
    lets a reference derived from a segmentation volume (e.g. skeleton junctions)
    be matched without building a full graph.
    """
    cand_pts = np.asarray(candidate_pts, dtype=float).reshape(-1, 3)
    ref_pts = np.asarray(reference_pts, dtype=float).reshape(-1, 3)

    available = np.ones(len(ref_pts), dtype=bool)
    tp = 0
    for c in cand_pts:
        if not available.any():
            break
        idx = np.flatnonzero(available)
        d = np.linalg.norm(ref_pts[idx] - c, axis=1)
        k = int(np.argmin(d))
        if d[k] < threshold:
            available[idx[k]] = False
            tp += 1

    n_cand = len(cand_pts)
    n_ref = len(ref_pts)
    fp = n_cand - tp
    fn = n_ref - tp
    denom = 2 * tp + fp + fn
    dice = (2 * tp) / denom if denom > 0 else float("nan")
    return BifurcationDice(tp=tp, fp=fp, fn=fn, dice=dice, n_candidate=n_cand, n_reference=n_ref)


def bifurcation_dice(
    candidate: SpatialGraph,
    reference: SpatialGraph,
    bounding_box: Optional[Sequence[float]] = None,
    threshold: float = 900.0,
) -> BifurcationDice:
    """Bifurcation-matching Dice between a candidate and a reference graph.

    Extracts each graph's branch points and delegates to
    :func:`bifurcation_dice_points`. ``bounding_box`` restricts the *candidate*
    bifurcations considered.
    """
    ref_pts = bifurcation_points(reference)
    cand_pts = bifurcation_points(candidate, bounding_box)
    return bifurcation_dice_points(cand_pts, ref_pts, threshold=threshold)


def meta_metric(
    candidate: Mapping[str, float],
    reference: Mapping[str, float],
    keys: Sequence[str] = ("Volume", "CC", "Euler", "BB", "CL"),
) -> float:
    """Combined skeleton-optimisation metric.

    The normalised root-mean-square of the relative differences between candidate
    and reference for each key (Volume, connected components, Euler number,
    branch-point count, clDice). This realises the formula left commented in
    ``meta_metric.m`` (lines 74-78). Lower is better (0 == identical).

    ``candidate`` / ``reference`` are mappings of metric name -> value; only the
    ``keys`` present in ``reference`` with a non-zero value are used.
    """
    terms = []
    for k in keys:
        if k not in reference or k not in candidate:
            continue
        ref = reference[k]
        if ref == 0:
            continue
        terms.append(((reference[k] - candidate[k]) / ref) ** 2)
    if not terms:
        return float("nan")
    return float(np.sqrt(np.sum(terms)))
