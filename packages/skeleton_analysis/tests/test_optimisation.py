"""Tests for the optimisation metrics and utilities."""

import numpy as np
import pytest

from skeleton_analysis.io.amira import SpatialGraph
from skeleton_analysis.optimisation.meta_metric import (
    bifurcation_dice,
    bifurcation_points,
    meta_metric,
)
from skeleton_analysis.optimisation.cl_dice import cl_score, dice


def _graph(vertex_coords, edges):
    g = SpatialGraph()
    g.set_vertex_field("VertexCoordinates", np.asarray(vertex_coords, float))
    g.set_edge_field("EdgeConnectivity", np.asarray(edges, np.int64))
    return g


def test_bifurcation_points():
    # node 1 is a bifurcation (degree 3); others are not.
    g = _graph(
        [[0, 0, 0], [1, 0, 0], [2, 1, 0], [2, -1, 0]],
        [[0, 1], [2, 1], [3, 1]],
    )
    pts = bifurcation_points(g)
    assert pts.shape == (1, 3)
    np.testing.assert_allclose(pts[0], [1, 0, 0])


def test_bifurcation_points_bounding_box():
    g = _graph(
        [[0, 0, 0], [1, 0, 0], [2, 1, 0], [2, -1, 0], [100, 0, 0]],
        [[0, 1], [2, 1], [3, 1]],
    )
    # Box excludes node 1 -> no bifurcations reported.
    pts = bifurcation_points(g, bounding_box=[50, 200, -10, 10, -10, 10])
    assert pts.shape == (0, 3)


def test_bifurcation_dice_perfect_match():
    g = _graph(
        [[0, 0, 0], [1, 0, 0], [2, 1, 0], [2, -1, 0]],
        [[0, 1], [2, 1], [3, 1]],
    )
    res = bifurcation_dice(g, g, threshold=1.0)
    assert res.tp == 1 and res.fp == 0 and res.fn == 0
    assert res.dice == pytest.approx(1.0)


def test_bifurcation_dice_partial():
    # Reference has 2 bifurcations; candidate matches 1 and adds 1 spurious.
    ref = _graph(
        [[0, 0, 0], [10, 0, 0], [11, 1, 0], [11, -1, 0], [-1, 1, 0], [-1, -1, 0]],
        [[0, 1], [2, 1], [3, 1], [4, 0], [5, 0]],  # nodes 0 and 1 are bifurcations
    )
    cand = _graph(
        [[0, 0.1, 0], [500, 0, 0], [501, 1, 0], [501, -1, 0], [-1, 1, 0], [-1, -1, 0]],
        [[0, 1], [2, 1], [3, 1], [4, 0], [5, 0]],  # bifurcations at node 0 (~match) & 1 (far)
    )
    res = bifurcation_dice(cand, ref, threshold=5.0)
    assert res.tp == 1
    assert res.fp == 1  # the far candidate bifurcation
    assert res.fn == 1  # the unmatched reference bifurcation
    assert res.dice == pytest.approx(2 / 4)


def test_meta_metric_identical_is_zero():
    ref = {"Volume": 100, "CC": 3, "Euler": -5, "BB": 40, "CL": 0.9}
    assert meta_metric(ref, ref) == pytest.approx(0.0)


def test_meta_metric_relative_rms():
    ref = {"Volume": 100, "CL": 1.0}
    cand = {"Volume": 90, "CL": 0.8}
    # sqrt((0.1)^2 + (0.2)^2)
    assert meta_metric(cand, ref, keys=("Volume", "CL")) == pytest.approx(
        np.sqrt(0.01 + 0.04)
    )


def test_cl_score_and_dice():
    v = np.array([255, 255, 0, 0], dtype=float)
    s = np.array([255, 0, 0, 0], dtype=float)
    # overlap sum(v*s)/sum(s) after /255 -> 1/1
    assert cl_score(v, s) == pytest.approx(1.0)
    a = np.array([1, 1, 0, 0], dtype=bool)
    b = np.array([1, 0, 0, 0], dtype=bool)
    assert dice(a, b) == pytest.approx(2 * 1 / (2 + 1))
