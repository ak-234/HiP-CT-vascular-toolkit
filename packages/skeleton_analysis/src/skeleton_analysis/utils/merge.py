"""Merge two Amira spatial graphs into one.

Port of ``add_spatial_graphs.m``. Shared vertices (identical coordinates) are
de-duplicated: graph 2's node IDs are remapped onto graph 1's IDs where they
coincide, and onto freshly appended IDs otherwise. All edge/point attributes
present in *both* graphs (e.g. NumEdgePoints, thickness, strahler, topo,
Identified_Graphs) are concatenated.

``graph2`` should be the smaller graph (as recommended by the MATLAB comment).
The manual, fragile inline reader/writer of the original is replaced by the
generic :mod:`skeleton_analysis.io.amira` round-trip.
"""

from __future__ import annotations

from typing import Dict

import numpy as np

from skeleton_analysis.io.amira import (
    F_EDGE_CONNECTIVITY,
    F_VERTEX_COORDS,
    SpatialGraph,
)


def _match_vertex(coord: np.ndarray, coords1: np.ndarray, tol: float):
    """Index of the row in ``coords1`` equal to ``coord`` (within ``tol``), or None."""
    if tol <= 0:
        hits = np.flatnonzero(np.all(coords1 == coord, axis=1))
    else:
        hits = np.flatnonzero(np.all(np.abs(coords1 - coord) <= tol, axis=1))
    return int(hits[0]) if hits.size else None


def add_spatial_graphs(
    graph1: SpatialGraph,
    graph2: SpatialGraph,
    match_tol: float = 0.0,
) -> SpatialGraph:
    """Return a new SpatialGraph merging ``graph2`` into ``graph1``.

    Parameters
    ----------
    match_tol : float
        Coordinate tolerance for treating two vertices as the same node. The
        default 0 reproduces MATLAB's exact ``ismember(...,'rows')`` matching.
    """
    vc1 = np.asarray(graph1.vertex_coords, dtype=float)
    vc2 = np.asarray(graph2.vertex_coords, dtype=float)
    n1 = len(vc1)

    # Map every graph-2 node ID to its merged ID.
    mapping: Dict[int, int] = {}
    new_coords = []
    next_id = n1
    for j in range(len(vc2)):
        match = _match_vertex(vc2[j], vc1, match_tol)
        if match is not None:
            mapping[j] = match
        else:
            mapping[j] = next_id
            next_id += 1
            new_coords.append(vc2[j])

    merged_coords = np.vstack([vc1, np.asarray(new_coords).reshape(-1, 3)]) if new_coords else vc1.copy()

    edges1 = np.asarray(graph1.edge_connectivity, dtype=np.int64)
    edges2 = np.asarray(graph2.edge_connectivity, dtype=np.int64)
    remapped2 = np.array([[mapping[int(a)], mapping[int(b)]] for a, b in edges2], dtype=np.int64)
    merged_edges = np.vstack([edges1, remapped2]) if len(edges2) else edges1.copy()

    merged = SpatialGraph(header=graph1.header, raw_parameters=graph1.raw_parameters)
    merged.set_vertex_field(F_VERTEX_COORDS, merged_coords)
    # MATLAB sets a single Identified_Graphs=1 per vertex for the merged graph.
    merged.set_vertex_field("Identified_Graphs", np.ones(len(merged_coords), dtype=np.int64))
    merged.set_edge_field(F_EDGE_CONNECTIVITY, merged_edges)

    # Concatenate every edge field shared by both graphs.
    for name in graph1.edge_fields:
        if name == F_EDGE_CONNECTIVITY or name not in graph2.edge_fields:
            continue
        merged.set_edge_field(
            name, np.concatenate([graph1.edge_fields[name], graph2.edge_fields[name]])
        )

    # Concatenate every point field shared by both graphs.
    for name in graph1.point_fields:
        if name not in graph2.point_fields:
            continue
        merged.set_point_field(
            name, np.concatenate([graph1.point_fields[name], graph2.point_fields[name]])
        )

    return merged
