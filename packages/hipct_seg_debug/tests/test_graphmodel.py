"""Every edit operation must be exactly reversible.

The whole undo design rests on one claim -- that recording the first value seen
for each touched id yields a correct inverse -- so every operation is checked the
same way: fingerprint, edit, undo, compare. A fingerprint covers the graph *and*
its derived indices, because an incidence index that drifts out of step with the
segment list produces wrong answers long after the edit that broke it.
"""

from __future__ import annotations

import numpy as np
import pytest

from hipct_seg_debug.edit.adapter import Triple
from hipct_seg_debug.edit.graphmodel import EditableGraph


def make_triple() -> Triple:
    """A Y: trunk 0->1, then two daughters 1->2 and 1->3.

           (2)
          /
    (0)-(1)
          \\
           (3)
    """
    nodes = {
        0: (0.0, 0.0, 0.0, 1),
        1: (0.0, 0.0, 100.0, 3),
        2: (50.0, 0.0, 200.0, 1),
        3: (-50.0, 0.0, 200.0, 1),
    }
    points = {}
    segments = []

    def run(sid, n1, n2, start, end, radius, first_pid):
        n = 5
        ts = np.linspace(0.0, 1.0, n)
        ids = []
        for i, t in enumerate(ts):
            pid = first_pid + i
            xyz = np.asarray(start) * (1 - t) + np.asarray(end) * t
            points[pid] = (float(xyz[0]), float(xyz[1]), float(xyz[2]), radius)
            ids.append(pid)
        segments.append(
            {"id": sid, "node1": n1, "node2": n2, "point_ids": ids, "strahler": 1}
        )

    run(0, 0, 1, (0, 0, 0), (0, 0, 100), 20.0, 0)
    run(1, 1, 2, (0, 0, 100), (50, 0, 200), 12.0, 10)
    run(2, 1, 3, (0, 0, 100), (-50, 0, 200), 12.0, 20)
    return Triple(nodes=nodes, points=points, segments=segments,
                  edge_attr_dtypes={"strahler": np.dtype("int64")})


def fingerprint(g: EditableGraph):
    """Everything that must come back after an undo, derived state included."""
    segs = sorted(
        (s["id"], s["node1"], s["node2"], tuple(s["point_ids"]),
         tuple(sorted((k, v) for k, v in s.items()
                      if k not in ("id", "node1", "node2", "point_ids"))))
        for s in g.segments
    )
    incidence = sorted((nid, tuple(sorted(sids))) for nid, sids in g._node_segs.items()
                       if nid in g.nodes)
    return (
        sorted(g.nodes.items()),
        sorted(g.points.items()),
        segs,
        sorted(g._seg_by_id),
        incidence,
        sorted(g._point_seg.items()),
    )


def check_indices(g: EditableGraph):
    """The derived indices must agree with the segment list they summarise."""
    assert sorted(g._seg_by_id) == sorted(s["id"] for s in g.segments)
    expected_inc: dict[int, set[int]] = {nid: set() for nid in g.nodes}
    expected_pts: dict[int, int] = {}
    for seg in g.segments:
        for key in ("node1", "node2"):
            expected_inc.setdefault(seg[key], set()).add(seg["id"])
        for pid in seg["point_ids"]:
            expected_pts[pid] = seg["id"]
    assert {k: v for k, v in g._node_segs.items() if k in g.nodes} == \
           {k: v for k, v in expected_inc.items() if k in g.nodes}
    assert g._point_seg == expected_pts
    for nid, node in g.nodes.items():
        assert node[3] == len(expected_inc.get(nid, ())), \
            f"node {nid} coordination number is stale"


@pytest.fixture
def graph() -> EditableGraph:
    return EditableGraph(make_triple())


def check_reversible(g: EditableGraph, action, *, expect_change=True):
    """Run `action`, then assert undo restores and redo re-applies exactly."""
    check_indices(g)
    before = fingerprint(g)
    result = action()
    after = fingerprint(g)
    check_indices(g)
    if expect_change:
        assert after != before, "operation reported success but changed nothing"

    assert g.undo() is not None
    assert fingerprint(g) == before, "undo did not restore the graph"
    check_indices(g)

    assert g.redo() is not None
    assert fingerprint(g) == after, "redo did not re-apply the edit"
    check_indices(g)

    assert g.undo() is not None
    assert fingerprint(g) == before
    check_indices(g)
    return result


# --------------------------------------------------------------- reading

def test_reads_report_the_shape_of_the_tree(graph):
    assert graph.degree(1) == 3
    assert sorted(graph.endpoints()) == [0, 2, 3]
    assert graph.node_segments(1) == {0, 1, 2}
    assert len(graph.components()) == 1
    assert graph.coords(0).shape == (5, 3)
    assert np.allclose(graph.radii(0), 20.0)
    assert graph.point_order() == list(range(0, 5)) + list(range(10, 15)) + list(range(20, 25))


def test_bounds_covers_only_the_named_segments(graph):
    box = graph.bounds([1])
    assert np.allclose(box[0], [0, 0, 100])
    assert np.allclose(box[1], [50, 0, 200])


# ------------------------------------------------------------ operations

def test_move_point_is_reversible(graph):
    check_reversible(graph, lambda: graph.move_point(2, (5.0, 5.0, 50.0)))


def test_move_point_on_a_node_drags_the_node(graph):
    # Point 4 is the last point of segment 0, i.e. it sits on node 1.
    graph.move_point(4, (1.0, 2.0, 103.0))
    assert graph.nodes[1][:3] == (1.0, 2.0, 103.0)


def test_move_node_drags_every_incident_boundary_point(graph):
    graph.move_node(1, (7.0, 8.0, 90.0))
    assert graph.points[4][:3] == (7.0, 8.0, 90.0)   # end of segment 0
    assert graph.points[10][:3] == (7.0, 8.0, 90.0)  # start of segment 1
    assert graph.points[20][:3] == (7.0, 8.0, 90.0)  # start of segment 2


def test_move_node_is_reversible(graph):
    check_reversible(graph, lambda: graph.move_node(1, (7.0, 8.0, 90.0)))


def test_set_radius_is_reversible(graph):
    check_reversible(graph, lambda: graph.set_radius(2, 33.0))
    assert graph.points[2][3] == 20.0


def test_scale_radii_is_reversible(graph):
    check_reversible(graph, lambda: graph.scale_radii(0, 1.5))


def test_set_segment_radii_rejects_a_length_mismatch(graph):
    with pytest.raises(ValueError, match="5 points, got 3 radii"):
        graph.set_segment_radii(0, [1.0, 2.0, 3.0])


def test_insert_and_delete_point_are_reversible(graph):
    check_reversible(graph, lambda: graph.insert_point(0, 2, (1.0, 1.0, 45.0), 19.0))
    check_reversible(graph, lambda: graph.delete_point(0, 2))


def test_endpoints_of_a_segment_cannot_be_deleted(graph):
    for index in (0, 4):
        with pytest.raises(ValueError, match="first or last point"):
            graph.delete_point(0, index)


def test_delete_segment_is_reversible_and_orphans_its_node(graph):
    check_reversible(graph, lambda: graph.delete_segment(1))
    graph.delete_segment(1)
    assert 2 not in graph.nodes, "the degree-1 node should go with its only segment"
    assert 1 in graph.nodes, "the junction still has two other segments"
    assert graph.degree(1) == 2
    assert not any(p in graph.points for p in range(10, 15))


def test_delete_subtree_removes_everything_downstream(graph):
    # Walking away from node 0 through segment 0 reaches the whole tree.
    assert graph.subtree(0, 0) == {0, 1, 2}
    # Walking away from node 1 through segment 1 reaches only that daughter.
    assert graph.subtree(1, 1) == {1}
    check_reversible(graph, lambda: graph.delete_subtree(0, 0))
    graph.delete_subtree(0, 0)
    assert graph.segments == []
    assert graph.nodes == {}


def test_delete_subtree_is_one_undo_step(graph):
    graph.delete_subtree(0, 0)
    assert len(graph.history.labels()) == 1
    graph.undo()
    assert len(graph.segments) == 3


def test_split_segment_creates_a_junction(graph):
    before = fingerprint(graph)
    nid, sid_a, sid_b = graph.split_segment(0, 2)

    assert graph.degree(nid) == 2
    assert graph.segment(sid_a)["node2"] == nid
    assert graph.segment(sid_b)["node1"] == nid
    assert not graph.has_segment(0)
    # The joint is duplicated so each edge owns its own run, per Amira's model.
    a_end = graph.segment(sid_a)["point_ids"][-1]
    b_start = graph.segment(sid_b)["point_ids"][0]
    assert a_end != b_start
    assert graph.points[a_end][:3] == graph.points[b_start][:3]
    # No centreline point is lost or duplicated in position.
    assert len(graph.coords(sid_a)) + len(graph.coords(sid_b)) == 6
    assert graph.segment(sid_a)["strahler"] == 1

    graph.undo()
    assert fingerprint(graph) == before


def test_split_must_be_strictly_interior(graph):
    for index in (0, 4):
        with pytest.raises(ValueError, match="strictly inside"):
            graph.split_segment(0, index)


def test_add_segment_snaps_onto_its_nodes(graph):
    coords = np.array([[9.0, 9.0, 9.0], [25.0, 0.0, 150.0], [9.0, 9.0, 9.0]])
    sid = graph.add_segment(2, 3, coords, [5.0, 6.0, 5.0])
    got = graph.coords(sid)
    assert np.allclose(got[0], graph.nodes[2][:3]), "start was not snapped onto node1"
    assert np.allclose(got[-1], graph.nodes[3][:3]), "end was not snapped onto node2"
    assert np.allclose(got[1], [25.0, 0.0, 150.0]), "interior points were disturbed"
    assert graph.degree(2) == 2 and graph.degree(3) == 2
    assert len(graph.components()) == 1


def test_add_segment_is_reversible(graph):
    coords = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]], dtype=float)
    check_reversible(graph, lambda: graph.add_segment(2, 3, coords, [5.0, 6.0, 5.0]))


def test_add_segment_validates_its_input(graph):
    with pytest.raises(ValueError, match="at least two points"):
        graph.add_segment(2, 3, np.zeros((1, 3)), [1.0])
    with pytest.raises(ValueError, match="3 coords but 2 radii"):
        graph.add_segment(2, 3, np.zeros((3, 3)), [1.0, 2.0])
    with pytest.raises(KeyError):
        graph.add_segment(2, 99, np.zeros((2, 3)), [1.0, 2.0])


def test_merge_nodes_moves_segments_and_snaps_geometry(graph):
    check_reversible(graph, lambda: graph.merge_nodes(2, 3))
    graph.merge_nodes(2, 3)
    assert 3 not in graph.nodes
    assert graph.degree(2) == 2
    assert graph.segment(2)["node2"] == 2
    assert graph.points[24][:3] == graph.nodes[2][:3]


def test_weld_coincident_nodes_joins_two_components():
    tri = make_triple()
    # A second, disjoint stub whose start sits ~1 um from node 2.
    tri.nodes[10] = (50.0, 0.0, 200.5, 1)
    tri.nodes[11] = (80.0, 0.0, 260.0, 1)
    tri.points[100] = (50.0, 0.0, 200.5, 8.0)
    tri.points[101] = (80.0, 0.0, 260.0, 8.0)
    tri.segments.append(
        {"id": 9, "node1": 10, "node2": 11, "point_ids": [100, 101], "strahler": 1}
    )
    g = EditableGraph(tri)
    assert len(g.components()) == 2

    g.weld_coincident_nodes(eps=10.0)
    assert len(g.components()) == 1, "the near-coincident nodes should have welded"
    assert len(g.history.labels()) == 1, "a weld is one undo step"

    g.undo()
    assert len(g.components()) == 2


def test_weld_is_a_no_op_when_nothing_is_close(graph):
    before = fingerprint(graph)
    graph.weld_coincident_nodes(eps=1.0)
    assert fingerprint(graph) == before
    assert not graph.history.can_undo


# ---------------------------------------------------------------- batching

def test_batch_is_a_single_undo_step_covering_every_edit(graph):
    before = fingerprint(graph)
    with graph.batch("two edits"):
        graph.set_radius(2, 99.0)
        graph.move_point(12, (26.0, 0.0, 151.0))

    assert len(graph.history.labels()) == 1
    assert graph.history.undo_label == "two edits"
    # The patch covers both edits, not just the last.
    assert graph.last_patch.seg_ids >= {0, 1}

    graph.undo()
    assert fingerprint(graph) == before


def test_nested_batches_collapse_into_the_outermost(graph):
    with graph.batch("outer"):
        graph.set_radius(2, 99.0)
        with graph.batch("inner"):
            graph.set_radius(3, 98.0)
    assert graph.history.labels() == ["outer"]


# ------------------------------------------------------------------ patches

def test_patch_reports_where_the_edit_landed(graph):
    patch = graph.move_point(12, (500.0, 0.0, 150.0))
    assert patch.seg_ids == {1}
    # The box spans both the old and the new position: the surface has to be
    # rebuilt where the point *was*, not only where it went.
    assert patch.aabb[0][0] <= 25.0
    assert patch.aabb[1][0] >= 500.0


def test_patches_merge_into_a_covering_box(graph):
    a = graph.move_point(2, (0.0, 0.0, 50.0))
    b = graph.move_point(12, (25.0, 0.0, 150.0))
    merged = a.merged(b)
    assert merged.seg_ids == {0, 1}
    assert np.all(merged.aabb[0] <= np.minimum(a.aabb[0], b.aabb[0]))
    assert np.all(merged.aabb[1] >= np.maximum(a.aabb[1], b.aabb[1]))


def test_history_limit_drops_the_oldest_edit(graph):
    graph.history.limit = 3
    for r in range(6):
        graph.set_radius(2, 20.0 + r)
    assert len(graph.history.labels()) == 3
    for _ in range(3):
        graph.undo()
    assert graph.points[2][3] == 22.0  # edits 0-2 became permanent
    assert not graph.history.can_undo


def test_redo_branch_is_discarded_by_a_new_edit(graph):
    graph.set_radius(2, 30.0)
    graph.undo()
    assert graph.history.can_redo
    graph.set_radius(3, 40.0)
    assert not graph.history.can_redo


# ----------------------------------------------------- round trip after edits

def test_an_edited_graph_still_converts_to_a_spatial_graph(graph):
    with graph.batch("edits"):
        graph.split_segment(0, 2)
        graph.set_radius(12, 15.0)
        graph.delete_segment(2)

    sg = graph.to_spatial_graph()
    assert sg.n_edge == len(graph.segments)
    assert sg.n_point == sum(len(s["point_ids"]) for s in graph.segments)
    assert int(sg.n_edge_points.sum()) == sg.n_point
    assert sg.connectivity.max() < sg.n_vertex, "connectivity indexes a dropped vertex"
    assert len(sg.edge_attrs["strahler"]) == sg.n_edge
    assert sg.thickness.shape == (sg.n_point,)
