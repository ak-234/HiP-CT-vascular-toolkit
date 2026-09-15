"""Neighbour-lookup helpers on an ``(nE, 2)`` edge array.

Ports of ``find_children.m``, ``find_parent_vec.m``, ``return_edge_ind.m`` and
the ``histc`` coordination-number counting used throughout the metrics code.
Edges are ``[source, target]`` = ``[child, parent]`` (see graph.build).
"""

from __future__ import annotations

from typing import Dict

import numpy as np


def find_children(node: int, edges: np.ndarray) -> np.ndarray:
    """Child node IDs feeding into ``node`` (rows where target == node).

    Port of ``find_children.m`` (``find(edge_nodes(:,2)==node)`` then take the
    sources), generalised to any number of children.
    """
    edges = np.asarray(edges, dtype=np.int64)
    mask = edges[:, 1] == node
    return edges[mask, 0]


def find_parents(node: int, edges: np.ndarray) -> np.ndarray:
    """Parent node ID(s) of ``node`` (rows where source == node).

    Port of ``find_parent_vec.m`` (``find(edge_nodes(:,1)==node)`` then take the
    targets). In a valid rooted tree every non-root node has exactly one parent.
    """
    edges = np.asarray(edges, dtype=np.int64)
    mask = edges[:, 0] == node
    return edges[mask, 1]


def return_edge_index(node_source: int, node_target: int, edges: np.ndarray) -> np.ndarray:
    """Row index/indices of the edge connecting ``node_source`` -> ``node_target``.

    Port of ``return_edge_ind.m``.
    """
    edges = np.asarray(edges, dtype=np.int64)
    mask = (edges[:, 0] == node_source) & (edges[:, 1] == node_target)
    return np.flatnonzero(mask)


def coordination_number(edges: np.ndarray) -> Dict[int, int]:
    """Degree (coordination number) of every node.

    Equivalent to MATLAB ``histc(edge_nodes(:), unique(edge_nodes))``: how many
    edge endpoints touch each node, counting both columns.
    """
    edges = np.asarray(edges, dtype=np.int64)
    nodes, counts = np.unique(edges.reshape(-1), return_counts=True)
    return {int(n): int(c) for n, c in zip(nodes, counts)}
