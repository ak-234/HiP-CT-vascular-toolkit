"""Tests for graph merging and VesselVio conversion."""

import numpy as np

from skeleton_analysis.io.amira import SpatialGraph, read_amira
from skeleton_analysis.io.vesselvio import vesselvio_to_amira, vesselvio_to_spatial_graph
from skeleton_analysis.utils.merge import add_spatial_graphs


def _graph(vertex_coords, edges, num_edge_points, point_coords, thickness, extra=None):
    g = SpatialGraph()
    g.set_vertex_field("VertexCoordinates", np.asarray(vertex_coords, float))
    g.set_edge_field("EdgeConnectivity", np.asarray(edges, np.int64))
    g.set_edge_field("NumEdgePoints", np.asarray(num_edge_points, np.int64))
    g.set_point_field("EdgePointCoordinates", np.asarray(point_coords, float))
    g.set_point_field("thickness", np.asarray(thickness, float))
    for name, vals in (extra or {}).items():
        g.set_edge_field(name, np.asarray(vals))
    return g


def test_merge_shares_vertices():
    # graph1 vertices 0,1,2 ; graph2 shares vertex at [1,0,0] (== graph1 node 1).
    g1 = _graph(
        vertex_coords=[[0, 0, 0], [1, 0, 0], [2, 0, 0]],
        edges=[[0, 1], [1, 2]],
        num_edge_points=[2, 2],
        point_coords=[[0, 0, 0], [1, 0, 0], [1, 0, 0], [2, 0, 0]],
        thickness=[1.0, 1.0, 0.5, 0.5],
        extra={"strahler": [2, 1]},
    )
    g2 = _graph(
        vertex_coords=[[1, 0, 0], [1, 1, 0]],  # node 0 shared with g1 node 1
        edges=[[0, 1]],
        num_edge_points=[2],
        point_coords=[[1, 0, 0], [1, 1, 0]],
        thickness=[0.3, 0.3],
        extra={"strahler": [1]},
    )
    merged = add_spatial_graphs(g1, g2)

    # One new vertex added (the [1,1,0] node); the shared one is not duplicated.
    assert merged.n_vertices == 4
    assert merged.n_edges == 3
    assert merged.n_points == 6
    # graph2's edge [0,1] remaps to [g1 node 1, new node 3].
    np.testing.assert_array_equal(merged.edge_connectivity[-1], [1, 3])
    # Shared attributes concatenated.
    np.testing.assert_array_equal(merged.edge_fields["strahler"], [2, 1, 1])
    assert int(np.sum(merged.num_edge_points)) == merged.n_points


def test_merge_round_trips_to_file(tmp_path):
    g1 = _graph(
        vertex_coords=[[0, 0, 0], [1, 0, 0]],
        edges=[[0, 1]],
        num_edge_points=[2],
        point_coords=[[0, 0, 0], [1, 0, 0]],
        thickness=[1.0, 1.0],
    )
    g2 = _graph(
        vertex_coords=[[1, 0, 0], [2, 0, 0]],
        edges=[[0, 1]],
        num_edge_points=[2],
        point_coords=[[1, 0, 0], [2, 0, 0]],
        thickness=[0.5, 0.5],
    )
    merged = add_spatial_graphs(g1, g2)
    from skeleton_analysis.io.amira import write_amira

    out = tmp_path / "merged.am"
    write_amira(merged, out)
    reloaded = read_amira(out)
    assert reloaded.n_vertices == 3  # shared [1,0,0]
    assert reloaded.n_edges == 2


def test_vesselvio_conversion(tmp_path):
    # vertices.csv: coordinate string "[z, y, x]" + radius.
    vcsv = tmp_path / "vertices.csv"
    vcsv.write_text(
        "v_coords,v_radius\n"
        '"[10, 20, 30]",1.5\n'
        '"[40, 50, 60]",2.5\n'
        '"[70, 80, 90]",3.5\n',
        encoding="utf-8",
    )
    ecsv = tmp_path / "edges.csv"
    ecsv.write_text("source,target\n0,1\n1,2\n", encoding="utf-8")

    g = vesselvio_to_spatial_graph(vcsv, ecsv, resolution=(50, 50))
    assert g.n_vertices == 3
    assert g.n_edges == 2
    assert g.n_points == 4  # 2 points per edge
    # First vertex was [z,y,x]=[10,20,30] -> swap zx -> [30,20,10] -> *50.
    np.testing.assert_allclose(g.vertex_coords[0], [30 * 50, 20 * 50, 10 * 50])
    np.testing.assert_array_equal(g.num_edge_points, [2, 2])
    # thickness picks endpoint radii for each edge.
    np.testing.assert_allclose(g.thickness, [1.5, 2.5, 2.5, 3.5])

    out = tmp_path / "vv.am"
    vesselvio_to_amira(vcsv, ecsv, out, resolution=(50, 50))
    reloaded = read_amira(out)
    assert reloaded.n_edges == 2
