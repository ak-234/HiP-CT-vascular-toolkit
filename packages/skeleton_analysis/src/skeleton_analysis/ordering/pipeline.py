"""End-to-end Strahler + topological ordering pipeline.

Clean port of ``run_ordering.m``. Fixes applied (per the conversion plan):

* the input path is honoured (MATLAB line 5 hard-coded ``filepath_ascii`` over
  the argument);
* the root node is supplied as an argument instead of the interactive
  ``inputdlg``/``menu`` dialogs;
* the computed ``strahler`` / ``topo`` orders are written as ordinary EDGE data
  blocks via :func:`skeleton_analysis.io.amira.write_amira`, replacing the
  fragile "edit the header by hand then append ``@22``/``@23``" workflow. Orders
  are kept aligned to the original edge indexing, so no error-prone node-pair
  matching (``write_back_strahler``) is needed.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple, Union

import networkx as nx
import numpy as np

from skeleton_analysis.graph.build import find_roots, reorient_edges, resolve_root
from skeleton_analysis.graph.neighbors import coordination_number
from skeleton_analysis.io.amira import SpatialGraph, read_amira, write_amira
from skeleton_analysis.ordering.strahler import strahler_order
from skeleton_analysis.ordering.topological import topological_generations

PathLike = Union[str, Path]


@dataclass
class OrderingResult:
    """Outcome of :func:`run_ordering`."""

    graph: SpatialGraph
    root_id: int  # first root (back-compat; == roots[0])
    strahler: np.ndarray  # per-edge, aligned to graph.edge_connectivity
    topo: np.ndarray  # per-edge
    flipped_edges: np.ndarray  # indices of edges whose orientation disagreed with a root
    roots: Optional[List[int]] = None  # all roots used (one per tree/component)


def order_forest(
    edges: np.ndarray, roots: Sequence[int]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Strahler + topological order for a *forest* (one root per component).

    Each root orders its own connected component (edges outside it stay 0), and
    the per-component results are combined into single per-edge arrays aligned to
    the input edge indexing. Returns ``(strahler, topo, flipped_indices)``.
    """
    edges = np.asarray(edges, dtype=np.int64)
    strahler = np.zeros(len(edges), dtype=np.int64)
    topo = np.zeros(len(edges), dtype=np.int64)
    flipped_all: set = set()
    for r in roots:
        s, _ = strahler_order(edges, int(r))
        t, _ = topological_generations(edges, int(r))
        strahler = np.where(s > 0, s, strahler)
        topo = np.where(t > 0, t, topo)
        _re, fl = reorient_edges(edges, int(r))
        flipped_all.update(int(x) for x in fl.tolist())
    return strahler, topo, np.array(sorted(flipped_all), dtype=np.int64)


def auto_roots(graph: SpatialGraph, radius_field: str = "MeanRadius") -> List[int]:
    """Heuristic root per connected component: the degree-1 endpoint of the
    component's largest-radius edge (the inlet trunk), falling back to the
    endpoint of the max-radius edge when neither end is a leaf.

    Used as a non-interactive fallback when roots are not picked manually.
    """
    edges = np.asarray(graph.edge_connectivity, dtype=np.int64)
    coord = coordination_number(edges)
    if radius_field in graph.edge_fields:
        radius = np.asarray(graph.edge_fields[radius_field], dtype=float)
    else:
        radius = np.ones(len(edges), dtype=float)

    g = nx.Graph()
    g.add_nodes_from(np.unique(edges).tolist())
    g.add_edges_from((int(a), int(b)) for a, b in edges)

    roots: List[int] = []
    for comp in nx.connected_components(g):
        comp_nodes = set(comp)
        mask = np.array([a in comp_nodes for a in edges[:, 0]])
        idx = np.flatnonzero(mask)
        if idx.size == 0:
            continue
        e = int(idx[np.argmax(radius[idx])])  # largest-radius edge in this component
        a, b = int(edges[e, 0]), int(edges[e, 1])
        # Prefer a degree-1 (leaf) endpoint as the inlet root.
        root = a if coord.get(a, 0) <= coord.get(b, 0) else b
        roots.append(root)
    return roots


def run_ordering(
    input_path: PathLike,
    output_path: Optional[PathLike] = None,
    root_id: Optional[int] = None,
    roots: Optional[Sequence[int]] = None,
    strahler_field: str = "strahler",
    topo_field: str = "topo",
) -> OrderingResult:
    """Read a spatial graph, compute Strahler + topological orders, write them back.

    Parameters
    ----------
    input_path : path
        Amira/Avizo ``.am`` spatial graph to read.
    output_path : path, optional
        Where to write the ordered graph. If omitted, nothing is written.
    root_id : int, optional
        Single root node ID (0-based) for a single-tree graph. If both this and
        ``roots`` are omitted the unique out-degree-0 node is used.
    roots : sequence of int, optional
        One root per connected component (tree), for a **forest**. Takes
        precedence over ``root_id``.
    strahler_field, topo_field : str
        EDGE field names for the two computed attributes.

    Returns
    -------
    OrderingResult
    """
    graph = read_amira(input_path)
    edges = np.asarray(graph.edge_connectivity, dtype=np.int64)

    if roots is not None:
        root_list = [int(r) for r in roots]
    else:
        root_list = [resolve_root(edges, root_id)]

    strahler_edges, topo_edges, flipped = order_forest(edges, root_list)

    graph.set_edge_field(strahler_field, strahler_edges)
    graph.set_edge_field(topo_field, topo_edges)

    if output_path is not None:
        write_amira(graph, output_path)

    return OrderingResult(
        graph=graph,
        root_id=int(root_list[0]),
        strahler=strahler_edges,
        topo=topo_edges,
        flipped_edges=flipped,
        roots=root_list,
    )


def root_candidates(input_path: PathLike):
    """List out-degree-0 root candidates for a file (helps choose ``root_id``)."""
    graph = read_amira(input_path)
    return find_roots(np.asarray(graph.edge_connectivity, dtype=np.int64))
