import numpy as np

from coronary_sdf.adaptive_surface_remesh import (
    SurfaceRemeshConfig,
    _coincident_vertex_groups,
    associate_surface_points,
    build_graph_size_field,
)


def _edge(edge_id, node1, node2, start, end, radius):
    return {
        "edge_id": edge_id,
        "node1": node1,
        "node2": node2,
        "points_mm": [start, end],
        "radii_mm": [radius, radius],
    }


def test_radius_aware_size_increases_with_local_radius():
    graph = {
        "edges": [
            _edge(1, 1, 2, [0, 0, 0], [2, 0, 0], 0.2),
            _edge(2, 3, 4, [0, 10, 0], [2, 10, 0], 0.8),
        ]
    }
    config = SurfaceRemeshConfig(
        min_edge_mm=0.01,
        max_edge_mm=1.0,
        clearance_candidates=2,
    )
    field = build_graph_size_field(graph, config)
    small = np.median(field.target_edges[field.edge_ids == 1])
    large = np.median(field.target_edges[field.edge_ids == 2])
    assert large > 3.5 * small


def test_nonadjacent_branch_clearance_refines_but_true_bifurcation_does_not():
    config = SurfaceRemeshConfig(
        circumferential_segments=12,
        min_edge_mm=0.02,
        max_edge_mm=1.0,
        graph_sample_spacing_mm=0.1,
        clearance_factor=0.35,
        clearance_candidates=64,
    )
    separated = {
        "edges": [
            _edge(1, 1, 2, [0, 0, 0], [2, 0, 0], 0.2),
            _edge(2, 3, 4, [0, 0.45, 0], [2, 0.45, 0], 0.2),
        ]
    }
    connected = {
        "edges": [
            _edge(1, 1, 2, [0, 0, 0], [2, 0, 0], 0.2),
            _edge(2, 1, 3, [0, 0, 0], [2, 0.45, 0], 0.2),
        ]
    }
    separated_field = build_graph_size_field(separated, config)
    connected_field = build_graph_size_field(connected, config)
    assert np.median(separated_field.target_edges) < np.median(
        connected_field.target_edges
    )


def test_surface_association_uses_wall_residual_not_nearest_axis_only():
    graph = {
        "edges": [
            _edge(10, 1, 2, [0, 0, 0], [2, 0, 0], 0.5),
            _edge(20, 3, 4, [0, 0.7, 0], [2, 0.7, 0], 0.1),
        ]
    }
    field = build_graph_size_field(
        graph,
        SurfaceRemeshConfig(
            min_edge_mm=0.01,
            max_edge_mm=1.0,
            graph_sample_spacing_mm=0.05,
            clearance_candidates=2,
        ),
    )
    # This point is on the r=0.5 wall of edge 10 but only 0.2 mm from edge 20's
    # axis.  Euclidean-nearest-axis assignment would incorrectly choose edge 20.
    _target, _radius, edge_id = associate_surface_points(
        np.asarray([[1.0, 0.5, 0.0]]), field, candidates=24
    )
    assert int(edge_id[0]) == 10


def test_coincident_topological_vertices_are_detected_before_stl_welding():
    vertices = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ]
    )
    groups = _coincident_vertex_groups(vertices)
    assert len(groups) == 1
    assert groups[0].tolist() == [0, 2]
