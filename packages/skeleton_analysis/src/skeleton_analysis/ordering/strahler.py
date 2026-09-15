"""Strahler ordering of a rooted vascular tree.

Clean re-derivation of ``strahler_graph.m`` + ``return_Strahler.m``. Rather than
the iterative leaf-pruning of the MATLAB code, we root the tree at ``root`` and
apply the standard Strahler rule in reverse-BFS order (children before parents):

    * a leaf has order 1;
    * an internal node's order is ``max(child_orders) + 1`` when that maximum is
      shared by **two or more** children, otherwise ``max(child_orders)``.

Correctness note
----------------
The MATLAB ``return_Strahler`` handled tri-furcations incompletely: when exactly
two of three children shared the maximum order (e.g. children ``[2, 2, 1]``) it
returned ``max`` (2) instead of ``max + 1`` (3). This port applies the standard
rule uniformly for any node degree, so it agrees with the MATLAB code on all
bi-furcations and on all-equal / all-different tri-furcations, and additionally
gives the correct result for the two-of-three-tie case.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np

from skeleton_analysis.graph.build import edge_child_index, resolve_root, rooted_tree


def _node_strahler(children: Dict[int, list], dist: Dict[int, int]) -> Dict[int, int]:
    order: Dict[int, int] = {}
    # Process farthest-from-root first so a node's children are already ordered.
    for node in sorted(dist, key=lambda n: -dist[n]):
        ch = children.get(node, [])
        if not ch:
            order[node] = 1
            continue
        child_orders = [order[c] for c in ch]
        m = max(child_orders)
        order[node] = m + 1 if child_orders.count(m) >= 2 else m
    return order


def strahler_order(
    edges: np.ndarray, root_id: Optional[int] = None
) -> Tuple[np.ndarray, Dict[int, int]]:
    """Compute the Strahler order of every edge.

    Parameters
    ----------
    edges : (nE, 2) int array
        Edge connectivity ``[source, target]`` (0-based node IDs). Orientation
        does not matter: the tree is rooted at ``root`` and each edge is assigned
        the Strahler order of its child endpoint (the one farther from the root),
        matching the MATLAB convention ``edge_nodes(edges_indx, 3) = order``.
    root_id : int, optional
        Root node ID. If omitted, the unique out-degree-0 node is used.

    Returns
    -------
    edge_orders : (nE,) int array
        Strahler order per edge, aligned with the input edge indexing. Edges not
        reachable from the root (other components) are set to 0.
    node_orders : dict
        Strahler order per node (root's component only).
    """
    edges = np.asarray(edges, dtype=np.int64)
    root = resolve_root(edges, root_id)
    dist, _parent, children = rooted_tree(edges, root)
    node_orders = _node_strahler(children, dist)

    child_col = edge_child_index(edges, dist)
    edge_orders = np.zeros(len(edges), dtype=np.int64)
    for i, col in enumerate(child_col):
        if col < 0:
            continue
        child_node = int(edges[i, col])
        edge_orders[i] = node_orders[child_node]
    return edge_orders, node_orders
