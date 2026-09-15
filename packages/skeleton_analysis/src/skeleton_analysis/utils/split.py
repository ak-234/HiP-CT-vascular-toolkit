"""Split a spatial graph into its connected components (trees).

The kidney workflow historically split a multi-tree Amira graph into separate
connected-component files (cc1/cc4/cc9) so each could be processed with a single
root. This does the same in memory: it returns one :class:`SpatialGraph` per
connected component, with node IDs, edges, points and every attribute field
re-indexed consistently, plus the old→new node-ID map so external references
(e.g. a picked root) can be translated.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import networkx as nx
import numpy as np

from skeleton_analysis.io.amira import (
    F_EDGE_CONNECTIVITY,
    F_NUM_EDGE_POINTS,
    SpatialGraph,
)


@dataclass
class Component:
    graph: SpatialGraph
    node_map: Dict[int, int]  # old (global) node id -> new (local) node id
    edge_indices: np.ndarray  # original edge indices making up this component
    point_indices: np.ndarray  # original point indices (in this component's edge order)


def split_connected_components(graph: SpatialGraph) -> List[Component]:
    """Return one :class:`Component` per connected component, largest first.

    Every vertex/edge/point field is carried over (re-indexed). Edge point ranges
    are recomputed from ``NumEdgePoints`` so ``EdgePointCoordinates``/``thickness``
    stay aligned.
    """
    edges = np.asarray(graph.edge_connectivity, dtype=np.int64)
    nump = np.asarray(graph.num_edge_points, dtype=np.int64)
    edge_starts = np.concatenate([[0], np.cumsum(nump)[:-1]])

    g = nx.Graph()
    g.add_nodes_from(range(graph.n_vertices))
    g.add_edges_from((int(a), int(b)) for a, b in edges)
    comps = sorted(nx.connected_components(g), key=len, reverse=True)

    out: List[Component] = []
    for comp in comps:
        comp_nodes = np.array(sorted(comp), dtype=np.int64)
        node_map = {int(old): new for new, old in enumerate(comp_nodes)}
        comp_set = set(comp_nodes.tolist())

        edge_mask = np.array(
            [int(a) in comp_set and int(b) in comp_set for a, b in edges], dtype=bool
        )
        edge_idx = np.flatnonzero(edge_mask)
        if edge_idx.size == 0:
            continue

        # Gather the point indices for the selected edges (in selected-edge order).
        pt_chunks = [np.arange(edge_starts[e], edge_starts[e] + nump[e]) for e in edge_idx]
        point_idx = np.concatenate(pt_chunks) if pt_chunks else np.empty(0, dtype=np.int64)

        sub = SpatialGraph(header=graph.header, raw_parameters=graph.raw_parameters)

        # Vertex fields (re-indexed by comp_nodes).
        for name, arr in graph.vertex_fields.items():
            sub.set_vertex_field(name, np.asarray(arr)[comp_nodes])
        # Remap connectivity to local node IDs.
        remapped = np.array(
            [[node_map[int(a)], node_map[int(b)]] for a, b in edges[edge_idx]],
            dtype=np.int64,
        )
        sub.set_edge_field(F_EDGE_CONNECTIVITY, remapped)
        # Other edge fields.
        for name, arr in graph.edge_fields.items():
            if name == F_EDGE_CONNECTIVITY:
                continue
            sub.set_edge_field(name, np.asarray(arr)[edge_idx])
        # Point fields.
        for name, arr in graph.point_fields.items():
            sub.set_point_field(name, np.asarray(arr)[point_idx])

        out.append(
            Component(
                graph=sub,
                node_map=node_map,
                edge_indices=edge_idx,
                point_indices=point_idx,
            )
        )
    return out
