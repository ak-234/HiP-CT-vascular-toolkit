"""Graph construction, root detection and edge reorientation utilities.

Replaces the MATLAB ``digraph`` usage plus ``Find_bad_edges.m`` and the small
neighbour-lookup helpers (``find_children.m``, ``find_parent_vec.m``,
``return_edge_ind.m``).
"""

from skeleton_analysis.graph.build import (
    to_digraph,
    find_roots,
    rooted_tree,
    reorient_edges,
    resolve_root,
)
from skeleton_analysis.graph.neighbors import (
    coordination_number,
    find_children,
    find_parents,
    return_edge_index,
)

__all__ = [
    "to_digraph",
    "find_roots",
    "rooted_tree",
    "reorient_edges",
    "resolve_root",
    "coordination_number",
    "find_children",
    "find_parents",
    "return_edge_index",
]
