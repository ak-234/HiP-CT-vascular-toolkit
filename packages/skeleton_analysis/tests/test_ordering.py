"""Strahler + topological ordering tests with hand-computed oracles."""

import numpy as np

from skeleton_analysis.ordering.pipeline import auto_roots, order_forest
from skeleton_analysis.ordering.strahler import strahler_order
from skeleton_analysis.ordering.topological import topological_generations
from skeleton_analysis.io.amira import SpatialGraph


def test_strahler_single_bifurcation():
    # 0 root; 1,2 leaves.  edges [1->0, 2->0]
    edges = np.array([[1, 0], [2, 0]], dtype=np.int64)
    orders, node_orders = strahler_order(edges, 0)
    np.testing.assert_array_equal(orders, [1, 1])
    assert node_orders[1] == 1 and node_orders[2] == 1
    assert node_orders[0] == 2  # root of two equal order-1 children


def test_strahler_balanced_binary_tree():
    edges = np.array(
        [[1, 0], [2, 0], [3, 1], [4, 1], [5, 2], [6, 2]], dtype=np.int64
    )
    orders, node_orders = strahler_order(edges, 0)
    # Edge orders = order of the child endpoint.
    np.testing.assert_array_equal(orders, [2, 2, 1, 1, 1, 1])
    assert node_orders[0] == 3
    assert node_orders[1] == 2 and node_orders[2] == 2


def test_strahler_trifurcation_all_equal():
    # root 0 with three leaf children -> node order 2, edge orders all 1.
    edges = np.array([[1, 0], [2, 0], [3, 0]], dtype=np.int64)
    orders, node_orders = strahler_order(edges, 0)
    np.testing.assert_array_equal(orders, [1, 1, 1])
    assert node_orders[0] == 2


def test_strahler_two_of_three_tie_is_corrected():
    # node 1 has three children with orders [2, 2, 1] -> should be 3 (max+1).
    # (The original MATLAB return_Strahler returned 2 here; this is the fix.)
    edges = np.array(
        [
            [1, 0],  # 0
            [2, 1],  # 1  child 2 (order 2)
            [3, 1],  # 2  child 3 (order 2)
            [4, 1],  # 3  child 4 (leaf, order 1)
            [5, 2],  # 4
            [6, 2],  # 5
            [7, 3],  # 6
            [8, 3],  # 7
        ],
        dtype=np.int64,
    )
    orders, node_orders = strahler_order(edges, 0)
    assert node_orders[2] == 2 and node_orders[3] == 2 and node_orders[4] == 1
    assert node_orders[1] == 3  # two children tie for the max -> +1
    # Edge 0 (1->0) carries node 1's order.
    assert orders[0] == 3


def test_strahler_orientation_independent():
    # Same tree but with some edges written parent->child; result must match.
    edges = np.array([[0, 1], [2, 0], [1, 3], [4, 1], [5, 2], [6, 2]], dtype=np.int64)
    orders, _ = strahler_order(edges, 0)
    # Map back: whatever the orientation, the child endpoint's order is used.
    # Nodes 3,4,5,6 are leaves (order 1); edges to 1 and 2 carry order 2.
    expected = []
    dist = {0: 0, 1: 1, 2: 1, 3: 2, 4: 2, 5: 2, 6: 2}
    node_order = {0: 3, 1: 2, 2: 2, 3: 1, 4: 1, 5: 1, 6: 1}
    for a, b in edges:
        child = a if dist[a] > dist[b] else b
        expected.append(node_order[child])
    np.testing.assert_array_equal(orders, expected)


def test_topological_generations_balanced():
    edges = np.array(
        [[1, 0], [2, 0], [3, 1], [4, 1], [5, 2], [6, 2]], dtype=np.int64
    )
    gen, node_gen = topological_generations(edges, 0)
    np.testing.assert_array_equal(gen, [1, 1, 2, 2, 2, 2])
    assert node_gen[0] == 1  # root generation 1
    assert node_gen[3] == 3  # depth 2 -> generation 3


def test_unreachable_edges_are_zero():
    # Edge [3,2] is a separate component; its order/gen should be 0.
    edges = np.array([[1, 0], [2, 0], [4, 3]], dtype=np.int64)
    orders, _ = strahler_order(edges, 0)
    gen, _ = topological_generations(edges, 0)
    assert orders[2] == 0
    assert gen[2] == 0


def test_order_forest_two_trees():
    # Tree A rooted at 0: a balanced binary tree (edges 0..5).
    # Tree B rooted at 10: a single bifurcation (edges 6..7).
    edges = np.array(
        [
            [1, 0], [2, 0], [3, 1], [4, 1], [5, 2], [6, 2],  # tree A
            [11, 10], [12, 10],  # tree B
        ],
        dtype=np.int64,
    )
    strahler, topo, flipped = order_forest(edges, roots=[0, 10])
    # Tree A: same result as the single-tree balanced case.
    np.testing.assert_array_equal(strahler[:6], [2, 2, 1, 1, 1, 1])
    np.testing.assert_array_equal(topo[:6], [1, 1, 2, 2, 2, 2])
    # Tree B: two leaves -> order 1, generation 1.
    np.testing.assert_array_equal(strahler[6:], [1, 1])
    np.testing.assert_array_equal(topo[6:], [1, 1])
    assert flipped.size == 0  # already consistently oriented


def test_auto_roots_picks_leaf_inlet_per_component():
    # Two trees; the largest-radius edge's leaf endpoint should be chosen.
    g = SpatialGraph()
    g.set_vertex_field(
        "VertexCoordinates",
        np.zeros((13, 3), dtype=float),
    )
    edges = np.array(
        [[1, 0], [2, 0], [3, 1], [4, 1], [5, 2], [6, 2], [11, 10], [12, 10]],
        dtype=np.int64,
    )
    g.set_edge_field("EdgeConnectivity", edges)
    # Trunk edges (leaf->hub) largest radius in each tree.
    radius = np.array([9.0, 1.0, 1, 1, 1, 1, 8.0, 1.0], dtype=float)
    g.set_edge_field("MeanRadius", radius)
    roots = auto_roots(g)
    # Tree A: edge [1,0] radius 9 -> endpoints 1 (deg3) and 0 (deg2) -> pick 0.
    # Tree B: edge [11,10] radius 8 -> endpoints 11 (deg1 leaf) and 10 (deg2) -> pick 11.
    assert set(roots) == {0, 11}
