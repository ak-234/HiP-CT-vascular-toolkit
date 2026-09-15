"""Tests for the metrics module."""

import numpy as np
import pandas as pd
import pytest

from skeleton_analysis.io.amira import SpatialGraph
from skeleton_analysis.metrics import (
    aggregate_by_strahler,
    branching_ang,
    branching_angles,
    branching_ratio,
    edge_midpoints,
    exponent_calculation,
    find_effective_gamma,
    gmregress,
    gmregresspi,
    intervessel_distance,
    mean_radius_per_edge,
    murray_law,
)

# Sokal & Rohlf Box 14.12 example used to validate the RMA regression ports.
SR_X = np.array([14, 17, 24, 25, 27, 33, 34, 37, 40, 41, 42], dtype=float)
SR_Y = np.array([61, 37, 65, 69, 54, 93, 87, 89, 100, 90, 97], dtype=float)


def _graph(vertex_coords=None, edges=None, num_edge_points=None,
           point_coords=None, thickness=None, edge_fields=None):
    g = SpatialGraph()
    if vertex_coords is not None:
        g.set_vertex_field("VertexCoordinates", np.asarray(vertex_coords, float))
    if edges is not None:
        g.set_edge_field("EdgeConnectivity", np.asarray(edges, np.int64))
    if num_edge_points is not None:
        g.set_edge_field("NumEdgePoints", np.asarray(num_edge_points, np.int64))
    if point_coords is not None:
        g.set_point_field("EdgePointCoordinates", np.asarray(point_coords, float))
    if thickness is not None:
        g.set_point_field("thickness", np.asarray(thickness, float))
    for name, vals in (edge_fields or {}).items():
        g.set_edge_field(name, np.asarray(vals))
    return g


# ---------------------------------------------------------------------------
# Regression (validated against the documented MATLAB example)
# ---------------------------------------------------------------------------
def test_gmregress_matches_reference():
    res = gmregress(SR_X, SR_Y)
    np.testing.assert_allclose([res.intercept, res.slope], [12.1938, 2.1194], atol=1e-3)
    np.testing.assert_allclose(res.ricker_ci, [[-10.6445, 35.0320], [1.3672, 2.8715]], atol=1e-3)
    np.testing.assert_allclose(res.jm_ci, [[-14.5769, 31.0996], [1.4967, 3.0010]], atol=1e-3)


def test_gmregress_slope_sign_matches_correlation():
    res = gmregress(SR_X, -SR_Y)
    assert res.slope < 0
    assert res.r < 0


def test_gmregresspi_matches_reference():
    b, yo, se, pint = gmregresspi(SR_X, SR_Y, xo=20)
    np.testing.assert_allclose(b, [12.1938, 2.1194], atol=1e-3)
    assert yo == pytest.approx(54.5811, abs=1e-3)
    assert se == pytest.approx(7.8954, abs=1e-3)
    np.testing.assert_allclose(pint, [36.7204, 72.4418], atol=1e-3)


# ---------------------------------------------------------------------------
# Effective gamma (Murray's law)
# ---------------------------------------------------------------------------
def test_find_effective_gamma_murray_cubic():
    # Two equal daughters with r_parent^3 == 2 r_child^3 -> gamma == 3.
    c = 0.5 ** (1 / 3)
    gamma = find_effective_gamma(1.0, [c, c])
    assert gamma == pytest.approx(3.0, abs=1e-4)


# ---------------------------------------------------------------------------
# Branching angles (hand-computed geometry)
# ---------------------------------------------------------------------------
def test_branching_ang_orthogonal():
    assert branching_ang([1, 0, 0], [0, 1, 0]) == pytest.approx(90.0)
    assert branching_ang([1, 0, 0], [1, 0, 0]) == pytest.approx(0.0)


def test_branching_angles_symmetric_y():
    # root 0 at origin, branch node 1 above it, children 2 & 3 forming a 90 deg fork.
    g = _graph(
        vertex_coords=[[0, 0, 0], [0, 1, 0], [1, 2, 0], [-1, 2, 0]],
        edges=[[1, 0], [2, 1], [3, 1]],
    )
    BA_edge, BA_vertex = branching_angles(g, root_id=0)
    assert BA_vertex[1, 0] == pytest.approx(90.0)  # angle between the two children
    assert np.isnan(BA_edge[0])  # edge (1->0): parent side, no branch angle
    assert BA_edge[1] == pytest.approx(135.0)  # parent vessel vs each child
    assert BA_edge[2] == pytest.approx(135.0)


# ---------------------------------------------------------------------------
# Murray's law
# ---------------------------------------------------------------------------
def test_murray_law_cubic_balance():
    c = 0.5 ** (1 / 3)
    g = _graph(
        vertex_coords=[[0, 0, 0], [0, 1, 0], [1, 2, 0], [-1, 2, 0]],
        edges=[[1, 0], [2, 1], [3, 1]],
        edge_fields={"MeanRadius": [1.0, c, c], "strahler": [2, 1, 1]},
    )
    df = murray_law(g, root_id=0)
    assert len(df) == 1
    row = df.iloc[0]
    assert row["node"] == 1
    assert row["parent_rad_cubed"] == pytest.approx(1.0)
    assert row["sumchild_cubed"] == pytest.approx(1.0)  # Murray's law satisfied
    assert row["strahler"] == 2
    assert row["gamma_eff"] == pytest.approx(3.0, abs=1e-4)


# ---------------------------------------------------------------------------
# Inter-vessel distance & mean radius
# ---------------------------------------------------------------------------
def test_mean_radius_per_edge():
    g = _graph(
        num_edge_points=[2, 3],
        thickness=[1.0, 3.0, 2.0, 2.0, 2.0],
    )
    r = mean_radius_per_edge(g)
    np.testing.assert_allclose(r, [2.0, 2.0])


def test_intervessel_distance():
    # Three straight vessels; midpoints at x=0, x=1, x=5 (all at y=z=0).
    g = _graph(
        num_edge_points=[2, 2, 2],
        point_coords=[
            [0, 0, 0], [0, 0, 0],   # edge 0 midpoint (0,0,0)
            [1, 0, 0], [1, 0, 0],   # edge 1 midpoint (1,0,0)
            [5, 0, 0], [5, 0, 0],   # edge 2 midpoint (5,0,0)
        ],
    )
    ivd = intervessel_distance(g)
    np.testing.assert_allclose(ivd, [1.0, 1.0, 4.0])


def test_edge_midpoints_arc_length():
    # Single edge of 3 collinear points; midpoint should be the central point.
    g = _graph(num_edge_points=[3], point_coords=[[0, 0, 0], [1, 0, 0], [2, 0, 0]])
    mids = edge_midpoints(g)
    np.testing.assert_allclose(mids[0], [1, 0, 0])


# ---------------------------------------------------------------------------
# Exponent
# ---------------------------------------------------------------------------
def test_exponent_calculation_runs():
    # Caterpillar tree so several internal nodes have downstream tips.
    edges = [[1, 0], [2, 1], [3, 2], [4, 1], [5, 2], [6, 3], [7, 3]]
    g = _graph(
        edges=edges,
        edge_fields={"MeanRadius": [3.0, 2.0, 1.0, 0.5, 0.5, 0.5, 0.5]},
    )
    res = exponent_calculation(g, radius_field="MeanRadius")
    # Nodes 1,2,3 have downstream tips 4,3,2 respectively -> 3 valid data points.
    valid = np.sum(~np.isnan(res.log_data).any(axis=1))
    assert valid == 3
    assert np.isfinite(res.exponent)


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------
def test_aggregate_by_strahler_and_ratio():
    df = pd.DataFrame(
        {
            "strahler": [1, 1, 1, 1, 2, 2, 3],
            "radius": [1.0, 1.2, 0.8, 1.0, 2.0, 2.2, 4.0],
        }
    )
    agg = aggregate_by_strahler(df, ["radius"])
    assert agg.loc[1, "count"] == 4
    assert agg.loc[1, "radius_mean"] == pytest.approx(1.0)
    assert agg.loc[2, "radius_mean"] == pytest.approx(2.1)

    ratio = branching_ratio(agg["count"])
    assert ratio.loc[1] == pytest.approx(4 / 2)  # 4 order-1 vessels / 2 order-2
    assert ratio.loc[2] == pytest.approx(2 / 1)
