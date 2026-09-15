"""The adapter must agree with ``coronary_sdf``'s own parser, exactly.

If these two ever disagree, an edit made in the viewer would produce a different
surface than the same graph run through the pipeline directly -- the preview
would be a lie. So the check is equality against ``parse_am`` on the real
dataset, not a tolerance.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from hipct_seg_debug.amira import SpatialGraph
from hipct_seg_debug.edit.adapter import (
    from_spatial_graph,
    to_spatial_graph,
)

from .realdata import GRAPH_REASON, REAL_AM


def synthetic_graph() -> SpatialGraph:
    """A two-edge chain with one int and one float edge attribute."""
    return SpatialGraph(
        path=Path("<synthetic>"),
        n_vertex=3,
        n_edge=2,
        n_point=6,
        vertices=np.array([[0.0, 0, 0], [0, 0, 10], [0, 0, 20]]),
        connectivity=np.array([[0, 1], [1, 2]], dtype=np.int64),
        n_edge_points=np.array([3, 3], dtype=np.int64),
        points=np.array(
            [[0.0, 0, 0], [0, 0, 5], [0, 0, 10], [0, 0, 10], [0, 0, 15], [0, 0, 20]]
        ),
        thickness=np.array([2.0, 2.1, 2.2, 2.2, 1.8, 1.5]),
        edge_attrs={
            "strahler": np.array([2, 1], dtype=np.int64),
            "MeanRadius": np.array([2.1, 1.83]),
        },
        vertex_attrs={},
    )


def assert_same_graph(a: SpatialGraph, b: SpatialGraph):
    assert (a.n_vertex, a.n_edge, a.n_point) == (b.n_vertex, b.n_edge, b.n_point)
    for name in ("vertices", "connectivity", "n_edge_points", "points", "thickness"):
        assert np.array_equal(getattr(a, name), getattr(b, name)), f"{name} differs"
    assert sorted(a.edge_attrs) == sorted(b.edge_attrs)
    for name, arr in a.edge_attrs.items():
        assert np.array_equal(np.ravel(arr), np.ravel(b.edge_attrs[name])), name
    # Point attributes round-trip too. Without this the triple could read a POINT
    # field and drop it on the way out, which is silent -- the graph still opens,
    # it has simply lost a column.
    assert sorted(a.point_attrs) == sorted(b.point_attrs)
    for name, arr in a.point_attrs.items():
        assert np.array_equal(np.ravel(arr), np.ravel(b.point_attrs[name])), name


def test_round_trip_is_exact():
    g = synthetic_graph()
    assert_same_graph(g, to_spatial_graph(from_spatial_graph(g)))


def test_triple_has_the_shape_the_pipeline_expects():
    tri = from_spatial_graph(synthetic_graph())
    assert tri.nodes[1] == (0.0, 0.0, 10.0, 2), "coordination number is the vertex degree"
    assert tri.points[1] == (0.0, 0.0, 5.0, 2.1), "the fourth component is the radius"
    assert tri.segments[0]["point_ids"] == [0, 1, 2]
    assert tri.segments[1]["point_ids"] == [3, 4, 5]
    assert tri.segments[0]["strahler"] == 2
    assert isinstance(tri.segments[0]["strahler"], int)
    assert isinstance(tri.segments[0]["MeanRadius"], float)


def test_strahler_alias_is_canonicalised_and_restored():
    g = synthetic_graph()
    g.edge_attrs["StrahlerOrder"] = g.edge_attrs.pop("strahler")

    tri = from_spatial_graph(g)
    assert tri.strahler_field == "StrahlerOrder"
    # coronary_sdf only ever looks up seg["strahler"].
    assert tri.segments[0]["strahler"] == 2
    assert "StrahlerOrder" not in tri.segments[0]

    back = to_spatial_graph(tri)
    assert "StrahlerOrder" in back.edge_attrs, "the file's own field name was lost"
    assert "strahler" not in back.edge_attrs


def test_copy_does_not_share_mutable_state():
    tri = from_spatial_graph(synthetic_graph())
    copy = tri.copy()
    # bridge_centerline_gaps and densify_sparse_segments both mutate this list
    # in place, so a shallow copy would corrupt the original.
    copy.segments[0]["point_ids"].append(99)
    copy.nodes[0] = (1.0, 1.0, 1.0, 1)
    assert tri.segments[0]["point_ids"] == [0, 1, 2]
    assert tri.nodes[0] == (0.0, 0.0, 0.0, 1)


def test_nodes_with_no_segments_are_dropped():
    tri = from_spatial_graph(synthetic_graph())
    tri.nodes[99] = (1.0, 2.0, 3.0, 0)  # Amira cannot express an isolated vertex
    back = to_spatial_graph(tri)
    assert back.n_vertex == 3
    assert back.connectivity.max() < back.n_vertex


@pytest.mark.skipif(not REAL_AM.is_file(), reason=GRAPH_REASON)
def test_matches_coronary_sdf_parser_on_the_real_graph():
    from hipct_seg_debug.amira import read_spatial_graph
    from hipct_seg_debug.edit._deps import ensure_coronary_sdf

    ensure_coronary_sdf()
    from coronary_sdf.parse_amira import parse_am

    tri = from_spatial_graph(read_spatial_graph(REAL_AM))
    nodes, points, segments = parse_am(REAL_AM)

    assert tri.nodes == nodes
    assert tri.points == points
    assert len(tri.segments) == len(segments)
    for mine, theirs in zip(tri.segments, segments):
        assert mine == theirs, f"segment {mine['id']} differs"


@pytest.mark.skipif(not REAL_AM.is_file(), reason=GRAPH_REASON)
def test_real_graph_round_trips_exactly():
    from hipct_seg_debug.amira import read_spatial_graph

    g = read_spatial_graph(REAL_AM)
    assert_same_graph(g, to_spatial_graph(from_spatial_graph(g)))
