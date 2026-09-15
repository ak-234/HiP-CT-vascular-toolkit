"""Tests for the Amira/Avizo ``.am`` reader/writer (skeleton_analysis.io.amira)."""

import numpy as np
import pytest

from skeleton_analysis.io.amira import (
    F_EDGE_CONNECTIVITY,
    F_THICKNESS,
    F_VERTEX_COORDS,
    SpatialGraph,
    read_amira,
    write_amira,
)


# ---------------------------------------------------------------------------
# Reading the real Avizo fixture
# ---------------------------------------------------------------------------
def test_read_real_file_counts(test_am_path):
    g = read_amira(test_am_path)
    # Header from Test.am: define VERTEX 148 / EDGE 147 / POINT 19924.
    assert g.n_vertices == 148
    assert g.n_edges == 147
    assert g.n_points == 19924
    assert g.header.startswith("# Avizo 3D ASCII 3.0")


def test_read_real_file_field_shapes(test_am_path):
    g = read_amira(test_am_path)
    assert g.vertex_coords.shape == (148, 3)
    assert g.edge_connectivity.shape == (147, 2)
    assert g.num_edge_points.shape == (147,)
    assert g.point_coords.shape == (19924, 3)
    assert g.thickness.shape == (19924,)
    # dtypes: coords/thickness float, connectivity/counts int.
    assert np.issubdtype(g.vertex_coords.dtype, np.floating)
    assert np.issubdtype(g.edge_connectivity.dtype, np.integer)
    assert np.issubdtype(g.thickness.dtype, np.floating)


def test_read_real_file_is_consistent(test_am_path):
    g = read_amira(test_am_path)
    # sum(NumEdgePoints) must equal the number of points.
    assert int(np.sum(g.num_edge_points)) == g.n_points
    assert g.check_consistency() == []
    # Node IDs are 0-based; first edge in Test.am is "0 1".
    assert g.edge_connectivity[0].tolist() == [0, 1]


def test_parameters_block_preserved(test_am_path):
    g = read_amira(test_am_path)
    assert g.raw_parameters is not None
    # Avizo file carries a TransformationMatrix + HxSpatialGraph content type.
    assert "TransformationMatrix" in g.raw_parameters
    assert "HxSpatialGraph" in g.raw_parameters
    # Brace-balanced capture.
    assert g.raw_parameters.count("{") == g.raw_parameters.count("}")


# ---------------------------------------------------------------------------
# Round-trip: read -> write -> read yields identical arrays
# ---------------------------------------------------------------------------
def test_round_trip_real_file(test_am_path, tmp_path):
    g1 = read_amira(test_am_path)
    out = tmp_path / "round_trip.am"
    write_amira(g1, out)
    g2 = read_amira(out)

    assert g2.n_vertices == g1.n_vertices
    assert g2.n_edges == g1.n_edges
    assert g2.n_points == g1.n_points

    # Same field set per domain.
    assert set(g2.vertex_fields) == set(g1.vertex_fields)
    assert set(g2.edge_fields) == set(g1.edge_fields)
    assert set(g2.point_fields) == set(g1.point_fields)

    # Integer fields survive exactly.
    np.testing.assert_array_equal(g2.edge_connectivity, g1.edge_connectivity)
    np.testing.assert_array_equal(g2.num_edge_points, g1.num_edge_points)
    # Float fields survive to full double precision (we write 15 sig figs).
    np.testing.assert_allclose(g2.vertex_coords, g1.vertex_coords, rtol=0, atol=0)
    np.testing.assert_allclose(g2.point_coords, g1.point_coords, rtol=0, atol=0)
    np.testing.assert_allclose(g2.thickness, g1.thickness, rtol=0, atol=0)

    # Preserved parameters round-trip too.
    assert g2.raw_parameters is not None
    assert "TransformationMatrix" in g2.raw_parameters


def test_round_trip_preserves_field_order(test_am_path, tmp_path):
    g1 = read_amira(test_am_path)
    out = tmp_path / "round_trip.am"
    write_amira(g1, out)
    g2 = read_amira(out)
    assert g2.field_order == g1.field_order


# ---------------------------------------------------------------------------
# Synthetic graphs + adding new fields (strahler/topo style)
# ---------------------------------------------------------------------------
def test_synthetic_round_trip(synthetic_am):
    p = synthetic_am(
        vertex_coords=[[0, 0, 0], [1, 0, 0], [2, 0, 0]],
        edge_connectivity=[[0, 1], [1, 2]],
        num_edge_points=[2, 2],
        point_coords=[[0, 0, 0], [1, 0, 0], [1, 0, 0], [2, 0, 0]],
        thickness=[1.0, 1.0, 0.5, 0.5],
    )
    g = read_amira(p)
    assert g.n_vertices == 3
    assert g.n_edges == 2
    assert g.n_points == 4
    np.testing.assert_array_equal(g.edge_connectivity, [[0, 1], [1, 2]])
    np.testing.assert_allclose(g.thickness, [1.0, 1.0, 0.5, 0.5])


def test_add_edge_field_and_write(synthetic_am, tmp_path):
    p = synthetic_am(
        vertex_coords=[[0, 0, 0], [1, 0, 0], [2, 0, 0]],
        edge_connectivity=[[0, 1], [1, 2]],
        num_edge_points=[2, 2],
        point_coords=[[0, 0, 0], [1, 0, 0], [1, 0, 0], [2, 0, 0]],
        thickness=[1.0, 1.0, 0.5, 0.5],
    )
    g = read_amira(p)
    g.set_edge_field("strahler", np.array([2, 1], dtype=np.int64))
    g.set_edge_field("topo", np.array([1, 2], dtype=np.int64))

    out = tmp_path / "with_orders.am"
    write_amira(g, out)
    g2 = read_amira(out)

    assert "strahler" in g2.edge_fields
    assert "topo" in g2.edge_fields
    np.testing.assert_array_equal(g2.edge_fields["strahler"], [2, 1])
    np.testing.assert_array_equal(g2.edge_fields["topo"], [1, 2])
    # New int fields are declared as int in the written file.
    assert np.issubdtype(g2.edge_fields["strahler"].dtype, np.integer)


def test_missing_data_section_raises(tmp_path):
    bad = tmp_path / "bad.am"
    bad.write_text("# AmiraMesh 3D ASCII 2.0\ndefine VERTEX 1\n", encoding="latin-1")
    with pytest.raises(ValueError):
        read_amira(bad)


def test_wrong_value_count_raises(tmp_path):
    text = (
        "# AmiraMesh 3D ASCII 2.0\n\n"
        "define VERTEX 2\ndefine EDGE 1\ndefine POINT 2\n\n"
        'Parameters {\n    ContentType "HxSpatialGraph"\n}\n\n'
        "VERTEX { float[3] VertexCoordinates } @1\n"
        "EDGE { int[2] EdgeConnectivity } @2\n"
        "EDGE { int NumEdgePoints } @3\n"
        "POINT { float[3] EdgePointCoordinates } @4\n"
        "POINT { float thickness } @5\n\n"
        "# Data section follows\n"
        "@1\n0 0 0\n"  # only ONE vertex row for a 2-vertex graph -> error
        "\n@2\n0 1\n\n@3\n2\n\n@4\n0 0 0\n0 0 0\n\n@5\n1\n1\n"
    )
    bad = tmp_path / "bad_counts.am"
    bad.write_text(text, encoding="latin-1")
    with pytest.raises(ValueError):
        read_amira(bad)
