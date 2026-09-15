"""Tests for splitting a spatial graph into connected components."""

import numpy as np

from skeleton_analysis.io.amira import SpatialGraph
from skeleton_analysis.utils.split import split_connected_components


def _forest_graph():
    """Two trees: A = nodes 0,1,2 (edges 0,1); B = nodes 5,6 (edge 2)."""
    g = SpatialGraph()
    g.set_vertex_field(
        "VertexCoordinates",
        np.array(
            [[0, 0, 0], [1, 0, 0], [2, 0, 0], [9, 9, 9], [9, 9, 9], [5, 0, 0], [6, 0, 0]],
            dtype=float,
        ),
    )
    g.set_edge_field("EdgeConnectivity", np.array([[0, 1], [1, 2], [5, 6]], dtype=np.int64))
    g.set_edge_field("NumEdgePoints", np.array([2, 2, 3], dtype=np.int64))
    g.set_point_field(
        "EdgePointCoordinates",
        np.array(
            [[0, 0, 0], [1, 0, 0],   # edge 0
             [1, 0, 0], [2, 0, 0],   # edge 1
             [5, 0, 0], [5.5, 0, 0], [6, 0, 0]],  # edge 2
            dtype=float,
        ),
    )
    g.set_point_field("thickness", np.array([1, 1, 0.5, 0.5, 2, 2, 2], dtype=float))
    g.set_edge_field("strahler", np.array([2, 1, 1], dtype=np.int64))
    return g


def test_split_two_trees():
    comps = split_connected_components(_forest_graph())
    assert len(comps) == 2

    a, b = comps[0], comps[1]  # largest first
    # Tree A: 3 nodes, 2 edges, 4 points.
    assert a.graph.n_vertices == 3
    assert a.graph.n_edges == 2
    assert a.graph.n_points == 4
    assert int(np.sum(a.graph.num_edge_points)) == a.graph.n_points
    # Tree B: 2 nodes, 1 edge, 3 points.
    assert b.graph.n_vertices == 2
    assert b.graph.n_edges == 1
    assert b.graph.n_points == 3

    # Node IDs remapped to 0-based locals.
    np.testing.assert_array_equal(a.graph.edge_connectivity, [[0, 1], [1, 2]])
    np.testing.assert_array_equal(b.graph.edge_connectivity, [[0, 1]])
    # node_map translates a global root ID to the sub-graph's local ID.
    assert b.node_map == {5: 0, 6: 1}

    # Fields carried and sliced correctly.
    np.testing.assert_array_equal(a.graph.edge_fields["strahler"], [2, 1])
    np.testing.assert_array_equal(b.graph.edge_fields["strahler"], [1])
    np.testing.assert_allclose(b.graph.thickness, [2, 2, 2])
    np.testing.assert_allclose(b.graph.point_coords[:, 0], [5, 5.5, 6])


def test_split_preserves_consistency():
    for comp in split_connected_components(_forest_graph()):
        assert comp.graph.check_consistency() == []
