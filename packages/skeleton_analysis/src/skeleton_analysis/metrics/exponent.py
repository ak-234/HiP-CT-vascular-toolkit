"""Vascular radius-scaling exponent.

Port of ``Exponent_calculation.m``. For every node we count the terminal tips
(leaves) downstream of it and read the radius of the vessel leaving it, then
regress ``log(tip_count)`` against ``log(radius)`` with model-II (RMA)
regression. The slope estimates the network's radius-scaling exponent.

Fix vs MATLAB
-------------
The original concatenated undefined variables ``log_exponent_cc1/cc4/cc9``. Here
:func:`exponent_calculation` accepts one *or more* graphs (e.g. the cc1/cc4/cc9
connected components) and concatenates their per-node data before regressing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Union

import networkx as nx
import numpy as np

from skeleton_analysis.graph.build import reorient_edges, resolve_root
from skeleton_analysis.graph.neighbors import coordination_number
from skeleton_analysis.io.amira import SpatialGraph
from skeleton_analysis.metrics.regression import GMRegressResult, gmregress


@dataclass
class ExponentResult:
    regression: GMRegressResult
    log_data: np.ndarray  # (n, 2) columns [log(tip_count), log(radius)]

    @property
    def exponent(self) -> float:
        """The fitted RMA slope (the scaling exponent)."""
        return self.regression.slope


def _node_tip_radius(graph: SpatialGraph, root_id: Optional[int], radius_field: str):
    edges0 = np.asarray(graph.edge_connectivity, dtype=np.int64)
    root = resolve_root(edges0, root_id)
    edges, _flipped = reorient_edges(edges0, root)
    if radius_field not in graph.edge_fields:
        raise KeyError(f"Edge field {radius_field!r} not present.")
    radius = np.asarray(graph.edge_fields[radius_field], dtype=float)
    coord = coordination_number(edges)

    # parent -> child directed graph (reverse of the child->parent edges), so
    # descendants(node) == everything downstream of node (toward the leaves).
    gpc = nx.DiGraph()
    gpc.add_nodes_from(np.unique(edges).tolist())
    gpc.add_edges_from((int(parent), int(child)) for child, parent in edges)

    data = []
    for node in np.unique(edges):
        parent_idx = np.flatnonzero(edges[:, 0] == node)  # edge leaving node
        if parent_idx.size == 0:
            continue
        downstream = nx.descendants(gpc, int(node))
        ds_tips = sum(1 for d in downstream if coord[int(d)] == 1)
        rad = float(radius[int(parent_idx[0])])
        data.append((ds_tips, rad))
    return np.asarray(data, dtype=float)


def exponent_calculation(
    graphs: Union[SpatialGraph, Sequence[SpatialGraph]],
    root_ids: Optional[Sequence[Optional[int]]] = None,
    radius_field: str = "MeanRadius",
    alpha: float = 0.05,
) -> ExponentResult:
    """Estimate the radius-scaling exponent across one or more graphs.

    Parameters
    ----------
    graphs : SpatialGraph or sequence of SpatialGraph
        One graph, or several connected components to pool (cc1/cc4/cc9).
    root_ids : sequence, optional
        Root node ID per graph (aligned with ``graphs``); ``None`` entries are
        auto-detected.
    radius_field : str
        Per-edge radius attribute.
    """
    if isinstance(graphs, SpatialGraph):
        graphs = [graphs]
    if root_ids is None:
        root_ids = [None] * len(graphs)

    parts: List[np.ndarray] = []
    for g, rid in zip(graphs, root_ids):
        parts.append(_node_tip_radius(g, rid, radius_field))
    data = np.vstack([p for p in parts if p.size])

    # Zeros -> NaN before taking logs (MATLAB: exponent_data(exponent_data==0)=NaN).
    data = np.where(data == 0, np.nan, data)
    log_data = np.log(data)

    reg = gmregress(log_data[:, 0], log_data[:, 1], alpha=alpha)
    return ExponentResult(regression=reg, log_data=log_data)
