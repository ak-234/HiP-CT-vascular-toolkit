import numpy as np
import pytest

from hipct_seg_debug.crosssection import _PlaneSampler
from hipct_seg_debug.edit import junction_links as jl
from hipct_seg_debug.edit.centreline_refine import bad_edges
from .conftest_geometry import cylinder, graph_from, make_frame

SHAPE = (30, 70, 100)  # (nz, ny, nx), 10 um voxels
CY, CZ = 35, 15


def branched(xs, *, offset_z=0):
    """A 60 um trunk along x with 30 um side branches leaving at each of `xs`.

    Branches alternate +y / -y. The graph's junction nodes sit `offset_z` voxels
    off the trunk axis, so the endpoints' centre of mass is not the lumen centre.
    """
    frame = make_frame(SHAPE)
    mask = cylinder(SHAPE, 6, 5, 95, cy=CY, cz=CZ)
    zz, yy, xx = np.ogrid[:SHAPE[0], :SHAPE[1], :SHAPE[2]]
    tips = []
    for k, x in enumerate(xs):
        up = k % 2 == 0
        span = (yy >= CY) & (yy < 65) if up else (yy > 5) & (yy <= CY)
        mask[((xx-x)**2 + (zz-CZ)**2 <= 9) & span] = 1
        tips.append((x, 62 if up else 8, CZ))
    voxels = [(8, CY, CZ)] + [(x, CY, CZ+offset_z) for x in xs] + [(92, CY, CZ)] + tips
    xyz = frame.seg_to_um(np.asarray(voxels, float))
    n = len(xs)
    edges = [(i, i+1, 12, 60.) for i in range(n+1)]
    edges += [(1+k, n+2+k, 12, 30.) for k in range(n)]
    return graph_from(xyz, edges), frame, mask


def test_split_junction_collapses_to_one_centred_node():
    graph, frame, mask = branched([40, 44], offset_z=2)
    before = {sid: graph.coords(sid).copy() for sid in graph.segment_ids()}
    ends = graph.coords(1)[[0, -1]]
    report = jl.simplify_links(graph, mask, frame)
    assert report.rule == "local"
    [row] = report.clusters
    assert row["status"] == "collapsed" and row["links"] == [1]
    np.testing.assert_allclose(row["p_ca"], ends.mean(axis=0))
    assert not graph.has_segment(1)
    assert set(graph.segment_ids()) == set(before) - {1}
    kept = row["kept_node"]
    assert graph.degree(kept) == 4 and len(graph.nodes) == len(before)
    # p_NewC is the boundary-distance maximum: back on the trunk axis, not at p_CA.
    axis_z = frame.seg_to_um(np.array([[0, CY, CZ]], float))[0, 2]
    assert abs(row["p_new"][2] - axis_z) < abs(row["p_ca"][2] - axis_z)
    assert row["position"] == "p_edt"
    assert row["edt_at_p_edt_um"] >= row["edt_at_p_ca_um"]
    np.testing.assert_allclose(graph.nodes[kept][:3], row["p_new"])
    sampler = _PlaneSampler(mask, frame)
    for sid in graph.segment_ids():
        assert not bad_edges(graph.coords(sid), sampler, frame).any()


def test_one_undo_restores_the_original_graph():
    graph, frame, mask = branched([40, 44])
    before = {sid: graph.coords(sid).copy() for sid in graph.segment_ids()}
    nodes = {n: v[:3] for n, v in graph.nodes.items()}
    jl.simplify_links(graph, mask, frame)
    graph.undo()
    assert set(graph.segment_ids()) == set(before)
    for sid, x in before.items():
        np.testing.assert_array_equal(graph.coords(sid), x)
    assert {n: v[:3] for n, v in graph.nodes.items()} == nodes


def test_long_inner_link_is_left_alone():
    graph, frame, mask = branched([30, 60])  # 300 um link on a 60 um trunk
    assert jl.find_short_links(graph) == []


def test_chain_of_short_links_is_one_cluster():
    graph, frame, mask = branched([38, 42, 46])
    [cluster] = jl.find_short_links(graph)
    assert cluster["links"] == [1, 2] and len(cluster["nodes"]) == 3
    [row] = jl.collapse_links(graph, [cluster], mask, frame)
    assert row["status"] == "collapsed"
    assert graph.degree(row["kept_node"]) == 5


def test_dry_run_reports_without_editing():
    graph, frame, mask = branched([40, 44])
    before = {sid: graph.coords(sid).copy() for sid in graph.segment_ids()}
    report = jl.simplify_links(graph, mask, frame, apply=False)
    assert [c["status"] for c in report.clusters] == ["accepted"]
    assert "p_new" in report.clusters[0]
    assert set(graph.segment_ids()) == set(before)


@pytest.mark.parametrize("end", [0, -1])
def test_blend_moves_the_end_exactly_and_fades_before_the_far_end(end):
    x = np.c_[np.linspace(0., 1000., 101), np.zeros(101), np.zeros(101)]
    point = x[end] + [0., 30., 0.]
    y = jl.blend_end(x, end, point, 40.)
    np.testing.assert_allclose(y[end], point)
    np.testing.assert_array_equal(y[-1 - end if end == 0 else 0], x[-1 - end if end == 0 else 0])
    lateral = y[:, 1] if end == 0 else y[::-1, 1]
    assert np.all(np.diff(lateral) <= 1e-9)            # decays monotonically away from the end
    assert lateral[6] == 0. and 0. < lateral[3] < 30.  # over twice the shift (60 um)


def test_exiting_distance_maximum_falls_back_to_the_endpoint_centre(monkeypatch):
    graph, frame, mask = branched([40, 44])
    outside = frame.seg_to_um(np.array([[42, CY, 2]], float))[0]
    monkeypatch.setattr(jl, "reposition", lambda *a, **k: (outside, 0., 0.))
    [row] = jl.simplify_links(graph, mask, frame).clusters
    assert row["status"] == "collapsed" and row["position"] == "p_ca"
    np.testing.assert_allclose(graph.nodes[row["kept_node"]][:3], row["p_ca"])
    assert not row["at_ball_edge"]


def test_collapse_that_leaves_the_segmentation_is_refused(monkeypatch):
    graph, frame, mask = branched([40, 44])
    before = {sid: graph.coords(sid).copy() for sid in graph.segment_ids()}
    outside = frame.seg_to_um(np.array([[42, CY, 2]], float))[0]
    monkeypatch.setattr(jl, "reposition", lambda *a, **k: (outside, 0., 0.))
    # Every candidate position exits, including where the kept node already is.
    monkeypatch.setattr(jl, "_positions", lambda *a: [("p_edt", outside), ("p_ca", outside)])
    [row] = jl.simplify_links(graph, mask, frame).clusters
    assert row["status"] == "refused" and row["reason"] == "new_segmentation_exit"
    assert set(row["exits_by_position"]) == {"p_edt", "p_ca"}
    assert set(graph.segment_ids()) == set(before)
    for sid, x in before.items():
        np.testing.assert_array_equal(graph.coords(sid), x)


def test_real_vessel_with_both_ends_in_the_cluster_is_not_deleted():
    graph, frame, mask = branched([40, 44])
    a, b = graph.segment(1)["node1"], graph.segment(1)["node2"]
    arc = np.array([graph.nodes[a][:3], graph.nodes[b][:3]], float)
    graph.add_segment(a, b, np.vstack([arc[0], arc.mean(axis=0)+[0, 0, 40], arc[1]]),
                      np.full(3, 30.))
    [row] = jl.simplify_links(graph, mask, frame).clusters
    assert row["reason"] == "would_create_self_loop"


def test_paper_global_thresholds():
    graph, frame, mask = branched([40, 44])
    assert jl.thinnest_diameter_um(graph) == pytest.approx(60.)
    [cluster] = jl.find_short_links(graph, factor=None, min_length_um=50.)
    assert cluster["rules"] == ["global"]
    assert jl.find_short_links(graph, factor=None, min_length_um=30.) == []
    report = jl.simplify_links(graph, mask, frame, factor=None, auto_thinnest=True, apply=False)
    assert report.min_length_um == pytest.approx(60.) and report.rule == "global"
    with pytest.raises(ValueError):
        jl.find_short_links(graph, factor=None)
