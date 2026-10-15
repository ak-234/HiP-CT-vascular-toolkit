"""Junction-aware fairing: staircase removed, vessels straight through, branches unkinked."""
import json

import numpy as np
import pytest

from hipct_seg_debug.edit import centreline_fair as cf
from hipct_seg_debug.edit import skeleton_phantoms as sp
from hipct_seg_debug.edit.graphmodel import EditableGraph
from hipct_seg_debug.edit.skeletonisers import skeletonise

from .conftest_geometry import make_frame


def jin_graph(mask, root_zyx, **params):
    frame = make_frame(mask.shape)
    c = skeletonise("jin-mgf", mask, frame, root_zyx=tuple(int(v) for v in root_zyx),
                    **params)
    return EditableGraph(c.triple), frame


def faired(mask, root_zyx, **params):
    graph, frame = jin_graph(mask, root_zyx)
    radii = {sid: graph.radii(sid).copy() for sid in graph.segment_ids()}
    report = cf.fair(graph, spacing_um=float(frame.seg_spacing[0]),
                     inside=cf.volume_inside(mask, frame.seg_origin, frame.seg_spacing),
                     **params)
    return graph, report, radii


def t_junction():
    """A straight round parent with a side branch leaving its middle at 45 degrees.

    Not 90: jin-mgf's section marking fills the parent's normal slab, which holds the
    whole of a short perpendicular branch, so that one is never extracted.
    """
    shape = (30, 64, 130)
    parent = sp.make_branch(np.array([[15., 20., 8.], [15., 20., 122.]]), 5.0, 1.0, 0, 1)
    side = sp.make_branch(np.array([[15., 20., 50.], [15., 56., 86.]]), 3.5, 1.0, 2, 3)
    return sp.rasterise(shape, [parent, side], caps={(1, 0): False}), (15, 20, 10)


def test_voxel_staircase_is_smoothed_inside_the_mask_with_radii_kept():
    p = sp.build("curved-collapsed")
    graph, report, radii = faired(p.mask, p.root_zyx)
    assert report.before["turn_median_deg"] > 30
    assert report.after["turn_median_deg"] < 5
    assert report.outside_edges_after == 0
    for sid in graph.segment_ids():
        np.testing.assert_array_equal(graph.radii(sid), radii[sid])


def branch_kink(graph, sid, nid):
    near, far = cf._end_chords(graph, sid, nid)
    return float(np.degrees(np.arccos(np.clip(near @ far, -1., 1.))))


def test_vessel_runs_straight_through_and_branch_leaves_without_a_kink():
    mask, root = t_junction()
    graph, frame = jin_graph(mask, root)
    (junction,) = [n for n in graph.nodes if graph.degree(n) == 3]
    branch = max(graph.node_segments(junction),
                 key=lambda s: abs(cf._leaving(graph, s, junction) @ [0., 1., 0.]))
    before = branch_kink(graph, branch, junction)
    report = cf.fair(graph, spacing_um=float(frame.seg_spacing[0]),
                     inside=cf.volume_inside(mask, frame.seg_origin, frame.seg_spacing))
    vessels = {sid: graph.segment(sid)["vessel_id"] for sid in graph.node_segments(junction)}
    # The two halves of the parent are one vessel; the branch is another.
    assert [v for s, v in vessels.items() if s != branch] == [vessels[s] for s in vessels
                                                             if s != branch][:1] * 2
    assert vessels[branch] not in {v for s, v in vessels.items() if s != branch}
    assert report.vessels == 2 and report.branch_vessels == 1
    assert report.after["through_median_deg"] < 5
    after = branch_kink(graph, branch, junction)
    assert before > 20 and after < before / 2
    assert report.outside_edges_after == 0


def test_y_bifurcation_kinks_are_reduced():
    p = sp.build("y-bifurcation")
    _, report, _ = faired(p.mask, p.root_zyx)
    assert report.after["kink_median_deg"] < report.before["kink_median_deg"]
    assert report.after["turn_p95_deg"] < 15
    assert report.outside_edges_after == 0


def test_degree_two_nodes_are_rejoined():
    p = sp.build("curved-collapsed")
    graph, frame = jin_graph(p.mask, p.root_zyx)
    (sid,) = graph.segment_ids()
    graph.split_segment(sid, len(graph.coords(sid)) // 2)
    assert len(graph.segment_ids()) == 2
    report = cf.fair(graph, spacing_um=float(frame.seg_spacing[0]))
    assert report.contracted_degree2 == 1
    assert len(graph.segment_ids()) == 1
    assert all(graph.degree(n) != 2 for n in graph.nodes)


def test_loops_are_faired_with_their_closing_ends_held():
    p = sp.build("loop")
    graph, frame = jin_graph(p.mask, p.root_zyx, allow_loops=True)
    nodes = {n: graph.nodes[n][:3] for n in graph.nodes if graph.degree(n) == 1}
    report = cf.fair(graph, spacing_um=float(frame.seg_spacing[0]),
                     inside=cf.volume_inside(p.mask, frame.seg_origin, frame.seg_spacing))
    assert report.vessels >= 2
    # Two of Jin's own edges on this phantom already sample outside the mask; fairing
    # pins them rather than counting them against itself.
    assert report.outside_edges_after <= report.outside_edges_before
    for n, xyz in nodes.items():  # terminals never move
        np.testing.assert_array_equal(graph.nodes[n][:3], xyz)


def test_oversmoothing_is_held_inside_the_mask():
    p = sp.build("curved-collapsed")
    _, report, _ = faired(p.mask, p.root_zyx, smooth_radii=30., min_smooth_voxels=30.,
                          max_move_radii=50.)
    assert report.outside_edges_after == 0
    assert report.pinned_points or report.backtracked_vessels or report.blocked_vessels


def test_polyline_fit_keeps_fixed_points_and_ignores_unweighted_ones():
    x = np.column_stack([np.arange(20.), np.where(np.arange(20) % 2, 1., 0.), np.zeros(20)])
    y = cf._fair_polyline(x, np.ones(20), np.full(20, 3.), {0, 10, 19})
    np.testing.assert_array_equal(y[[0, 10, 19]], x[[0, 10, 19]])
    assert np.ptp(y[3:8, 1]) < np.ptp(x[3:8, 1])
    w = np.ones(20)
    w[1:6] = 0  # free of data: a straight bridge between the pinned end and the rest
    y = cf._fair_polyline(x, w, np.full(20, 3.), {0, 19})
    assert np.all(np.isfinite(y))


def test_jin_mgf_fair_option(tmp_path):
    p = sp.build("curved-collapsed")
    frame = make_frame(p.mask.shape)
    root = tuple(int(v) for v in p.root_zyx)
    raw = skeletonise("jin-mgf", p.mask, frame, root_zyx=root)
    smooth = skeletonise("jin-mgf", p.mask, frame, root_zyx=root, fair=True)
    assert "faired" in smooth.detail
    assert len(smooth.triple.segments) == len(raw.triple.segments)
    assert (cf.turn_stats(EditableGraph(smooth.triple))["turn_median_deg"]
            < cf.turn_stats(EditableGraph(raw.triple))["turn_median_deg"])


def test_fair_centreline_cli_round_trip(tmp_path):
    from hipct_seg_debug.edit.__main__ import build_parser, cmd_fair_centreline
    from hipct_seg_debug.edit.adapter import read_triple
    from hipct_seg_debug.edit.amira_write import write_spatial_graph
    from hipct_seg_debug.rle_write import write_lattice

    mask, root = t_junction()
    graph, frame = jin_graph(mask, root)
    seg, src, out = tmp_path / "seg.am", tmp_path / "jin.am", tmp_path / "fair.am"
    write_lattice(seg, mask.astype(np.uint8), frame.seg_bbox_um)
    write_spatial_graph(graph.to_spatial_graph(), src)
    args = build_parser().parse_args(["fair-centreline", str(src), "--seg", str(seg),
                                      "--out", str(out)])
    assert cmd_fair_centreline(args) == 0
    triple = read_triple(out)
    assert len(triple.segments) == len(graph.segment_ids())
    report = json.loads((tmp_path / "fair.am.report.json").read_text())
    assert report["outside_edges_after"] == 0
    assert report["after"]["kink_p95_deg"] < report["before"]["kink_p95_deg"]
    with pytest.raises(ValueError, match="separate output"):
        cmd_fair_centreline(build_parser().parse_args(
            ["fair-centreline", str(src), "--seg", str(seg), "--out", str(src)]))
