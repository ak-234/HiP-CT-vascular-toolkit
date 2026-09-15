"""Tests for the metrics report module."""

import numpy as np
import pandas as pd
import pytest

from skeleton_analysis.io.amira import SpatialGraph, read_amira
from skeleton_analysis.metrics.report import (
    assign_kmeans,
    compare_states,
    edge_metrics_table,
    murray_table,
    plot_report,
    write_metric_graph,
)


def _tree_graph():
    """Root 0; branch node 1; children 2,3. Two points per edge."""
    g = SpatialGraph()
    g.set_vertex_field("VertexCoordinates",
                       np.array([[0, 0, 0], [0, 1, 0], [1, 2, 0], [-1, 2, 0]], float))
    g.set_edge_field("EdgeConnectivity", np.array([[1, 0], [2, 1], [3, 1]], np.int64))
    g.set_edge_field("NumEdgePoints", np.array([2, 2, 2], np.int64))
    g.set_point_field("EdgePointCoordinates",
                      np.array([[0, 1, 0], [0, 0, 0],
                                [1, 2, 0], [0, 1, 0],
                                [-1, 2, 0], [0, 1, 0]], float))
    g.set_point_field("thickness", np.array([2, 2, 1, 1, 1, 1], float))
    return g


def test_spatialgraph_copy_independent():
    g = _tree_graph()
    h = g.copy()
    h.thickness[0] = 999.0
    assert g.thickness[0] == 2.0            # original untouched
    h.set_edge_field("strahler", np.array([9, 9, 9]))
    assert "strahler" not in g.edge_fields  # field added to copy only


def test_edge_metrics_table():
    g = _tree_graph()
    df = edge_metrics_table(g, roots=[0])
    assert len(df) == 3
    expected_cols = {"edge", "tree", "strahler", "topo", "radius", "length",
                     "tortuosity", "volume", "ld_ratio", "branching_angle", "intervessel"}
    assert expected_cols.issubset(df.columns)
    # Trunk edge (1->0) carries the higher Strahler order.
    assert df.loc[df["edge"] == 0, "strahler"].iloc[0] == 2
    # Radius comes from the per-point thickness (trunk r=2, children r=1).
    np.testing.assert_allclose(df.sort_values("edge")["radius"].to_numpy(), [2, 1, 1])
    # Branch-point children get a defined branching angle; trunk edge is NaN.
    assert np.isnan(df.loc[df["edge"] == 0, "branching_angle"].iloc[0])
    assert np.isfinite(df.loc[df["edge"] == 1, "branching_angle"].iloc[0])


def test_assign_kmeans_two_blobs():
    sklearn = pytest.importorskip("sklearn")
    rng = np.random.default_rng(0)
    small = pd.DataFrame({
        "topo": rng.normal(1, 0.1, 20), "radius": rng.normal(1, 0.1, 20),
        "tortuosity": rng.normal(1, 0.1, 20), "branching_angle": rng.normal(90, 1, 20),
        "ld_ratio": rng.normal(2, 0.1, 20), "intervessel": rng.normal(5, 0.1, 20),
    })
    large = pd.DataFrame({
        "topo": rng.normal(8, 0.1, 20), "radius": rng.normal(20, 0.1, 20),
        "tortuosity": rng.normal(3, 0.1, 20), "branching_angle": rng.normal(150, 1, 20),
        "ld_ratio": rng.normal(10, 0.1, 20), "intervessel": rng.normal(50, 0.1, 20),
    })
    table = pd.concat([small, large], ignore_index=True)
    labels, k = assign_kmeans(table, k=2)
    assert k == 2
    # Clusters are ordered by mean radius -> the small-radius rows are cluster 0.
    assert np.all(labels[:20] == 0)
    assert np.all(labels[20:] == 1)


def test_write_metric_graph_roundtrip(tmp_path):
    g = _tree_graph()
    df = edge_metrics_table(g, roots=[0])
    df["kmeans_cluster"] = np.array([1, 0, 0])
    out = tmp_path / "metrics.am"
    write_metric_graph(g, df, out)
    reloaded = read_amira(out)
    for f in ("strahler", "topo", "kmeans_cluster", "ld_ratio", "branching_angle", "radius"):
        assert f in reloaded.edge_fields
    # NaN branching angle on the trunk edge was written as the -1 sentinel.
    assert reloaded.edge_fields["branching_angle"][0] == -1


def test_plot_and_compare_smoke(tmp_path):
    pytest.importorskip("matplotlib")
    pytest.importorskip("seaborn")
    g = _tree_graph()
    df = edge_metrics_table(g, roots=[0])
    murray = murray_table(g, roots=[0])
    plot_report(df, murray, tmp_path / "rep", prefix="")
    assert (tmp_path / "rep" / "metrics_summary.csv").exists()
    assert any((tmp_path / "rep").glob("*.png"))

    comp = compare_states({"before": df, "after": df}, tmp_path / "cmp")
    assert (tmp_path / "cmp" / "state_comparison.csv").exists()
    assert not comp.empty
