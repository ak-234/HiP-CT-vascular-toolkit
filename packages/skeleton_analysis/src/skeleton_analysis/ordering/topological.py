"""Topological-generation numbering of a rooted vascular tree.

Clean re-derivation of ``topological_gen.m``. In the MATLAB code the first edge
leaving the root is generation 1, and the generation increments by one along
every branch, resetting to the branch-point's generation when a new branch
starts. That is exactly the **edge depth from the root**: an edge's generation
is the BFS distance (in edges) from the root to the edge's child endpoint.

We therefore compute BFS distances from the root once and assign each edge the
distance of its child (farther) endpoint. This reproduces ``topological_gen``'s
output on any valid rooted tree via a single, robust pass.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np

from skeleton_analysis.graph.build import edge_child_index, resolve_root, rooted_tree


def topological_generations(
    edges: np.ndarray, root_id: Optional[int] = None
) -> Tuple[np.ndarray, Dict[int, int]]:
    """Compute the topological generation of every edge.

    Parameters
    ----------
    edges : (nE, 2) int array
        Edge connectivity ``[source, target]`` (0-based). Orientation-independent.
    root_id : int, optional
        Root node ID; if omitted the unique out-degree-0 node is used.

    Returns
    -------
    edge_gen : (nE,) int array
        Generation per edge (root's edges = 1, increasing outward). Edges not
        reachable from the root are set to 0.
    node_gen : dict
        Generation per node = 1 + distance from root (root = 1), for parity with
        the MATLAB ``node_gen`` table.
    """
    edges = np.asarray(edges, dtype=np.int64)
    root = resolve_root(edges, root_id)
    dist, _parent, _children = rooted_tree(edges, root)

    child_col = edge_child_index(edges, dist)
    edge_gen = np.zeros(len(edges), dtype=np.int64)
    for i, col in enumerate(child_col):
        if col < 0:
            continue
        child_node = int(edges[i, col])
        edge_gen[i] = dist[child_node]  # depth of child == generation

    node_gen = {int(n): int(d) + 1 for n, d in dist.items()}
    return edge_gen, node_gen
