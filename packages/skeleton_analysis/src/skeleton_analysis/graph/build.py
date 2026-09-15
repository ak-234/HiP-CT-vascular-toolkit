"""Build directed graphs, detect roots, and reorient edges toward a root.

Edge convention (inherited from the MATLAB package): ``edges`` is an ``(nE, 2)``
integer array where each row is ``[source, target]``. Edges point from a child
toward its parent, so the **root** is the node with out-degree 0 (it is never a
source). Node IDs are 0-based.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import networkx as nx
import numpy as np


def to_digraph(edges: np.ndarray, one_based: bool = False) -> nx.DiGraph:
    """Build a directed graph from an ``(nE, 2)`` edge array.

    ``one_based=True`` reproduces the MATLAB ``digraph(s+1, t+1)`` offset; the
    default keeps the native 0-based IDs used everywhere in this package.
    """
    edges = np.asarray(edges, dtype=np.int64)
    offset = 1 if one_based else 0
    g = nx.DiGraph()
    g.add_nodes_from(np.unique(edges) + offset)
    g.add_edges_from((int(a) + offset, int(b) + offset) for a, b in edges)
    return g


def find_roots(edges: np.ndarray) -> List[int]:
    """Return node IDs with out-degree 0 (candidate roots).

    Equivalent to MATLAB ``find(G.outdegree == 0)``: nodes that never appear as a
    source (column 1).
    """
    edges = np.asarray(edges, dtype=np.int64)
    sources = set(edges[:, 0].tolist())
    nodes = set(np.unique(edges).tolist())
    return sorted(n for n in nodes if n not in sources)


def resolve_root(edges: np.ndarray, root_id: Optional[int] = None) -> int:
    """Determine the root node ID.

    If ``root_id`` is given it is returned (validated). Otherwise the unique
    out-degree-0 node is used; if there is not exactly one, a ``ValueError`` is
    raised listing the candidates (the batch-mode replacement for the MATLAB
    ``inputdlg``/``menu`` prompt).
    """
    candidates = find_roots(edges)
    if root_id is not None:
        return int(root_id)
    if len(candidates) == 1:
        return candidates[0]
    raise ValueError(
        "Could not uniquely determine the root node "
        f"(out-degree-0 candidates: {candidates}). "
        "Pass root_id explicitly."
    )


def rooted_tree(
    edges: np.ndarray, root: int
) -> Tuple[Dict[int, int], Dict[int, int], Dict[int, List[int]]]:
    """Root the (undirected) tree at ``root`` via BFS.

    Returns ``(dist, parent, children)`` for the connected component containing
    ``root``:

    * ``dist[node]``   – edge distance from ``root`` (root = 0),
    * ``parent[node]`` – the node one step closer to ``root`` (root absent),
    * ``children[node]`` – nodes one step farther from ``root``.

    Nodes in other connected components are not included.
    """
    edges = np.asarray(edges, dtype=np.int64)
    g = nx.Graph()
    g.add_nodes_from(np.unique(edges).tolist())
    g.add_edges_from((int(a), int(b)) for a, b in edges)
    root = int(root)
    if root not in g:
        raise ValueError(f"Root {root} is not a node in the graph")

    dist = nx.single_source_shortest_path_length(g, root)
    bfs = nx.bfs_tree(g, root)  # directed root -> leaves
    children: Dict[int, List[int]] = {n: [] for n in bfs.nodes}
    parent: Dict[int, int] = {}
    for p in bfs.nodes:
        for c in bfs.successors(p):
            children[p].append(int(c))
            parent[int(c)] = int(p)
    return dist, parent, children


def edge_child_index(edges: np.ndarray, dist: Dict[int, int]) -> np.ndarray:
    """For each edge, return which endpoint (0 or 1) is the child.

    The child is the endpoint farther from the root (larger ``dist``). Edges with
    an endpoint outside the rooted component are marked ``-1``.
    """
    edges = np.asarray(edges, dtype=np.int64)
    out = np.full(len(edges), -1, dtype=np.int64)
    for i, (a, b) in enumerate(edges):
        da = dist.get(int(a))
        db = dist.get(int(b))
        if da is None or db is None:
            continue
        out[i] = 0 if da > db else 1
    return out


def reorient_edges(
    edges: np.ndarray, root_id: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """Reorient every edge to point child -> parent (toward ``root``).

    Clean re-derivation of ``Find_bad_edges.m``: instead of the iterative
    leaf-pruning used there, we BFS from the root and flip any edge whose current
    ``[source, target]`` disagrees with the child->parent direction. Edge indices
    and count are preserved (only orientation changes), so the result stays
    aligned with ``NumEdgePoints`` / point data.

    Returns ``(new_edges, flipped_indices)``.
    """
    edges = np.asarray(edges, dtype=np.int64).copy()
    root = resolve_root(edges, root_id)
    dist, _parent, _children = rooted_tree(edges, root)

    flipped: List[int] = []
    for i, (a, b) in enumerate(edges):
        da = dist.get(int(a))
        db = dist.get(int(b))
        if da is None or db is None:
            continue  # edge outside the root's component; leave as-is
        # Correct orientation is child (farther) -> parent (nearer).
        if da < db:  # a is nearer the root than b => currently reversed
            edges[i] = [b, a]
            flipped.append(i)
    return edges, np.asarray(flipped, dtype=np.int64)
