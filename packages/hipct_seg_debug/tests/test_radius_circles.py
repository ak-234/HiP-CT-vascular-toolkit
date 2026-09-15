"""Geometry tests for the graph-wide ideal-radius contour layer."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("pyvista")

from hipct_seg_debug.viewer3d import _point_tangents, radius_circle_polydata


def _graph(points, radii, edge_sizes):
    edge_sizes = np.asarray(edge_sizes, dtype=np.int64)
    return SimpleNamespace(
        points=np.asarray(points, dtype=np.float64),
        thickness=np.asarray(radii, dtype=np.float64),
        n_edge=len(edge_sizes),
        n_edge_points=edge_sizes,
        edge_offsets=np.concatenate([[0], np.cumsum(edge_sizes)]),
    )


def _mixed_graph():
    return _graph(
        [
            # Curved edge.
            [0, 0, 0], [10, 0, 0], [10, 10, 0],
            # A second edge repeats the junction coordinate but leaves along z.
            [10, 10, 0], [10, 10, 10],
            # Singleton and entirely coincident edges exercise both fallbacks.
            [30, 0, 0],
            [40, 0, 0], [40, 0, 0],
            # These records must be skipped.
            [50, 0, 0], [60, 0, 0], [np.nan, 0, 0],
        ],
        [2, 3, 4, 5, 6, 7, 8, 9, 0, np.nan, 10],
        [3, 2, 1, 2, 1, 1, 1],
    )


def test_every_valid_point_gets_one_closed_exact_radius_ring():
    graph = _mixed_graph()
    resolution = 12
    mesh = radius_circle_polydata(graph, resolution=resolution)
    valid = (
        np.isfinite(graph.points).all(axis=1)
        & np.isfinite(graph.thickness)
        & (graph.thickness > 0)
    )
    valid_i = np.flatnonzero(valid)

    assert mesh.n_lines == len(valid_i)
    assert mesh.n_points == len(valid_i) * (resolution + 1)
    assert np.array_equal(mesh.cell_data["radius_um"], graph.thickness[valid].astype(np.float32))

    tangent = _point_tangents(graph)
    rings = np.asarray(mesh.points).reshape(len(valid_i), resolution + 1, 3)
    for ring, point_i in zip(rings, valid_i):
        centre = graph.points[point_i]
        radius = graph.thickness[point_i]
        displacement = ring[:-1] - centre
        assert ring[-1] == pytest.approx(ring[0], abs=2e-5)
        assert displacement.mean(axis=0) == pytest.approx([0, 0, 0], abs=2e-5)
        assert np.linalg.norm(displacement, axis=1) == pytest.approx(radius, abs=2e-5)
        assert displacement @ tangent[point_i] == pytest.approx(
            np.zeros(resolution), abs=2e-5
        )


def test_repeated_junction_records_keep_their_edge_local_planes():
    graph = _mixed_graph()
    tangent = _point_tangents(graph)

    assert np.array_equal(graph.points[2], graph.points[3])
    assert tangent[2] == pytest.approx([0, 1, 0])
    assert tangent[3] == pytest.approx([0, 0, 1])

    mesh = radius_circle_polydata(graph, resolution=12)
    rings = np.asarray(mesh.points).reshape(-1, 13, 3)
    # The first ring is perpendicular to y (constant y); the second to z (constant z).
    assert np.ptp(rings[2, :, 1]) < 1e-5
    assert np.ptp(rings[3, :, 2]) < 1e-5


def test_singleton_and_degenerate_edges_use_the_z_fallback():
    graph = _mixed_graph()
    tangent = _point_tangents(graph)
    assert tangent[5] == pytest.approx([0, 0, 1])
    assert tangent[6] == pytest.approx([0, 0, 1])
    assert tangent[7] == pytest.approx([0, 0, 1])


def test_no_valid_radius_returns_an_empty_mesh():
    graph = _graph([[0, 0, 0], [1, 0, 0]], [0, np.nan], [2])
    mesh = radius_circle_polydata(graph)
    assert mesh.n_points == 0
    assert mesh.n_cells == 0


def test_resolution_must_describe_a_circle():
    with pytest.raises(ValueError, match="at least 3"):
        radius_circle_polydata(_mixed_graph(), resolution=2)
