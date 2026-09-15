"""Branching angles at bifurcation / trifurcation points.

Port of ``branching_angles_with_strahler.m`` (+ ``branching_ang.m``). For every
branch point (coordination number >= 3) we form vectors from the branch node to
its parent (incoming vessel) and to each child, then measure:

* per-vertex angles between the child branches (the bifurcation angle), and
* per-edge angles between the incoming parent vessel and each outgoing child.

Angles use the straight-line vectors between *vertex* coordinates, matching the
MATLAB implementation. Interactive root selection is replaced by ``root_id``.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from skeleton_analysis.graph.build import reorient_edges, resolve_root
from skeleton_analysis.graph.neighbors import (
    coordination_number,
    find_children,
    find_parents,
    return_edge_index,
)
from skeleton_analysis.io.amira import SpatialGraph


def branching_ang(vec1, vec2) -> float:
    """Angle in degrees between two 3-D vectors (port of ``branching_ang.m``)."""
    v1 = np.asarray(vec1, dtype=float)
    v2 = np.asarray(vec2, dtype=float)
    denom = np.linalg.norm(v1) * np.linalg.norm(v2)
    if denom == 0:
        return float("nan")
    cos = np.clip(np.dot(v1, v2) / denom, -1.0, 1.0)  # clip guards acos domain
    return float(np.degrees(np.arccos(cos)))


def branching_angles(
    graph: SpatialGraph, root_id: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute per-edge and per-vertex branching angles.

    Returns ``(BA_edge, BA_vertex)`` where ``BA_edge`` is ``(n_edges,)`` (angle
    between the parent vessel and the child that terminates on that edge) and
    ``BA_vertex`` is ``(n_vertices, 3)`` (pairwise child-child angles; only the
    first column is used for bifurcations). Entries with no defined angle are
    NaN.
    """
    edges0 = np.asarray(graph.edge_connectivity, dtype=np.int64)
    root = resolve_root(edges0, root_id)
    edges, _flipped = reorient_edges(edges0, root)
    coords = np.asarray(graph.vertex_coords, dtype=float)
    coord = coordination_number(edges)

    BA_edge = np.full(len(edges), np.nan)
    BA_vertex = np.full((graph.n_vertices, 3), np.nan)

    branch_nodes = [n for n in np.unique(edges) if coord[n] >= 3]
    for node in branch_nodes:
        children = find_children(node, edges)
        parents = find_parents(node, edges)
        node_coords = coords[node]
        child_vecs = [node_coords - coords[c] for c in children]

        # Per-vertex: pairwise angles between child branches.
        if len(children) == 2:
            BA_vertex[node, 0] = branching_ang(child_vecs[0], child_vecs[1])
        elif len(children) >= 3:
            BA_vertex[node, 0] = branching_ang(child_vecs[0], child_vecs[1])
            BA_vertex[node, 1] = branching_ang(child_vecs[1], child_vecs[2])
            BA_vertex[node, 2] = branching_ang(child_vecs[0], child_vecs[2])

        # Per-edge: angle between the incoming parent vessel and each child.
        if parents.size > 0:
            parent_vec = node_coords - coords[parents[0]]
            for c, cv in zip(children, child_vecs):
                ei = return_edge_index(c, node, edges)
                if ei.size:
                    BA_edge[ei[0]] = branching_ang(parent_vec, cv)

    return BA_edge, BA_vertex
