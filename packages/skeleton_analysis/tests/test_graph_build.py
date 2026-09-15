"""Tests for graph construction, root detection and edge reorientation."""

import numpy as np
import pytest

from skeleton_analysis.graph.build import (
    find_roots,
    reorient_edges,
    resolve_root,
    rooted_tree,
)
from skeleton_analysis.graph.neighbors import (
    coordination_number,
    find_children,
    find_parents,
    return_edge_index,
)


# Balanced binary tree, edges oriented child -> parent:
#        0
#      /   \
#     1     2
#    / \   / \
#   3   4 5   6
BALANCED = np.array(
    [[1, 0], [2, 0], [3, 1], [4, 1], [5, 2], [6, 2]], dtype=np.int64
)


def test_find_roots_unique():
    assert find_roots(BALANCED) == [0]
    assert resolve_root(BALANCED) == 0


def test_find_roots_ambiguous_raises():
    # Two disconnected edges -> two out-degree-0 nodes.
    edges = np.array([[1, 0], [3, 2]], dtype=np.int64)
    assert find_roots(edges) == [0, 2]
    with pytest.raises(ValueError):
        resolve_root(edges)
    # Explicit root_id resolves it.
    assert resolve_root(edges, root_id=2) == 2


def test_rooted_tree_structure():
    dist, parent, children = rooted_tree(BALANCED, 0)
    assert dist == {0: 0, 1: 1, 2: 1, 3: 2, 4: 2, 5: 2, 6: 2}
    assert parent == {1: 0, 2: 0, 3: 1, 4: 1, 5: 2, 6: 2}
    assert sorted(children[0]) == [1, 2]
    assert sorted(children[1]) == [3, 4]
    assert children[3] == []


def test_neighbors():
    assert sorted(find_children(0, BALANCED).tolist()) == [1, 2]
    assert sorted(find_children(1, BALANCED).tolist()) == [3, 4]
    assert find_parents(3, BALANCED).tolist() == [1]
    assert return_edge_index(3, 1, BALANCED).tolist() == [2]
    coord = coordination_number(BALANCED)
    assert coord[0] == 2  # root touches edges to 1 and 2
    assert coord[1] == 3  # 1 touches edges to 0, 3, 4
    assert coord[3] == 1  # leaf


def test_reorient_already_consistent_is_noop():
    new_edges, flipped = reorient_edges(BALANCED, 0)
    np.testing.assert_array_equal(new_edges, BALANCED)
    assert flipped.size == 0


def test_reorient_flips_reversed_edges():
    # Reverse two edges; reorient must flip exactly those back.
    edges = BALANCED.copy()
    edges[2] = [1, 3]  # was [3, 1]
    edges[5] = [2, 6]  # was [6, 2]
    new_edges, flipped = reorient_edges(edges, 0)
    np.testing.assert_array_equal(new_edges, BALANCED)
    assert sorted(flipped.tolist()) == [2, 5]
    # After reorientation the root is the unique out-degree-0 node.
    assert find_roots(new_edges) == [0]
