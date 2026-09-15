"""Tests for per-edge geometry metrics."""

import numpy as np
import pytest

from skeleton_analysis.io.amira import SpatialGraph
from skeleton_analysis.metrics import geometry as geom


def _graph(num_edge_points, point_coords, thickness):
    g = SpatialGraph()
    g.set_edge_field("EdgeConnectivity",
                     np.zeros((len(num_edge_points), 2), dtype=np.int64))
    g.set_edge_field("NumEdgePoints", np.asarray(num_edge_points, np.int64))
    g.set_point_field("EdgePointCoordinates", np.asarray(point_coords, float))
    g.set_point_field("thickness", np.asarray(thickness, float))
    return g


def test_lengths_chords_tortuosity():
    g = _graph(
        num_edge_points=[3, 3],
        point_coords=[
            [0, 0, 0], [1, 0, 0], [2, 0, 0],       # edge 0: straight, length 2, chord 2
            [0, 0, 0], [1, 1, 0], [2, 0, 0],       # edge 1: bent, length 2*sqrt(2), chord 2
        ],
        thickness=[2, 2, 2, 1, 1, 1],
    )
    np.testing.assert_allclose(geom.segment_lengths(g), [2.0, 2 * np.sqrt(2)])
    np.testing.assert_allclose(geom.chord_lengths(g), [2.0, 2.0])
    np.testing.assert_allclose(geom.tortuosity(g), [1.0, np.sqrt(2)])


def test_radius_stats_volume_surface():
    g = _graph(
        num_edge_points=[3, 2],
        point_coords=[[0, 0, 0], [1, 0, 0], [2, 0, 0], [0, 0, 0], [0, 3, 0]],
        thickness=[2, 2, 2, 1, 1],
    )
    rs = geom.radius_stats(g)
    np.testing.assert_allclose(rs.avg, [2.0, 1.0])
    np.testing.assert_allclose(rs.max, [2.0, 1.0])
    np.testing.assert_allclose(rs.min, [2.0, 1.0])
    # volume = pi r^2 L ; edge0 = pi*4*2, edge1 = pi*1*3
    np.testing.assert_allclose(geom.volumes(g), [np.pi * 4 * 2, np.pi * 1 * 3])
    # lateral SA = 2 pi r L ; edge0 = 2pi*2*2, edge1 = 2pi*1*3
    np.testing.assert_allclose(geom.surface_areas(g), [2 * np.pi * 2 * 2, 2 * np.pi * 1 * 3])


def test_loop_tortuosity_zero():
    # A closed edge (first == last point) has ~0 chord -> tortuosity 0.
    g = _graph(
        num_edge_points=[3],
        point_coords=[[0, 0, 0], [1, 0, 0], [0, 0, 0]],
        thickness=[1, 1, 1],
    )
    assert geom.tortuosity(g)[0] == 0.0
