"""Painting a mask correction, and turning it back into centreline.

The fixture is the case the whole feature exists for: one vessel, broken in two by
a hole in the segmentation, and a graph that reflects the break. Painting the hole
shut and re-skeletonising must join the two components with one new segment welded
at both ends -- and must not disturb the geometry either side, which is Avizo's and
is not in question.
"""

from __future__ import annotations

import numpy as np
import pytest

from hipct_seg_debug.amira import LatticeInfo
from hipct_seg_debug.edit.adapter import Triple
from hipct_seg_debug.edit.graphmodel import EditableGraph
from hipct_seg_debug.edit.maskedit import MaskSource
from hipct_seg_debug.edit.reskeletonise import (
    clear_box,
    reskeletonise_box,
    select_painted,
    trim_to_new,
)
from hipct_seg_debug.frame import WorldFrame

from .test_maskedit import FakeLattice

pytest.importorskip("skimage")

SPACING = 66.0
SHAPE = (26, 26, 70)  # (nz, ny, nx) segmentation voxels
AXIS_Z, AXIS_Y = 13, 13
RADIUS_VOX = 3
GAP = (28, 40)  # x range removed from the mask, exclusive
TUBE = (5, 65)


def _cylinder(shape, x0, x1, radius=RADIUS_VOX):
    """A tube along x, centred on (AXIS_Z, AXIS_Y)."""
    nz, ny, nx = shape
    zz, yy, xx = np.ogrid[:nz, :ny, :nx]
    radial = (zz - AXIS_Z) ** 2 + (yy - AXIS_Y) ** 2 <= radius * radius
    return (radial & (xx >= x0) & (xx < x1)).astype(np.uint8)


@pytest.fixture
def frame():
    nz, ny, nx = SHAPE
    dims = np.array([nx, ny, nz])
    origin = np.zeros(3)
    bbox = np.empty(6)
    bbox[0::2] = origin
    bbox[1::2] = origin + (dims - 1) * SPACING
    info = LatticeInfo(path=None, dims=dims, bbox=bbox, fields={})
    return WorldFrame.from_inputs((nz * 2, ny * 2, nx * 2), SPACING / 2, info)


@pytest.fixture
def broken_source():
    """The mask with the gap: two disconnected tube stubs."""
    volume = _cylinder(SHAPE, TUBE[0], GAP[0]) | _cylinder(SHAPE, GAP[1], TUBE[1])
    return MaskSource(FakeLattice(volume))


def _straight_segment(frame, x0, x1, node_a, node_b, first_pid):
    """Points along the tube axis from voxel x0 to x1 inclusive."""
    xs = np.arange(x0, x1 + 1)
    ijk = np.stack([xs, np.full_like(xs, AXIS_Y), np.full_like(xs, AXIS_Z)], axis=1)
    xyz = frame.seg_to_um(ijk)
    points = {first_pid + n: (*map(float, p), RADIUS_VOX * SPACING)
              for n, p in enumerate(xyz)}
    seg = {"id": 0, "node1": node_a, "node2": node_b,
           "point_ids": list(points)}
    return points, seg, xyz


@pytest.fixture
def broken_graph(frame):
    """Two segments matching the two mask stubs, as separate components."""
    points, nodes, segments = {}, {}, []
    for n, (x0, x1) in enumerate([(TUBE[0], GAP[0] - 1), (GAP[1], TUBE[1] - 1)]):
        pts, seg, xyz = _straight_segment(frame, x0, x1, 2 * n, 2 * n + 1, 1000 * (n + 1))
        points.update(pts)
        nodes[2 * n] = (*map(float, xyz[0]), 0)
        nodes[2 * n + 1] = (*map(float, xyz[-1]), 0)
        seg["id"] = n
        segments.append(seg)
    return EditableGraph(Triple(nodes=nodes, points=points, segments=segments))


def _paint_the_gap(source, x0=GAP[0], x1=GAP[1]):
    """Fill the hole in the mask, exactly as a brush stroke would."""
    patch = _cylinder(SHAPE, x0, x1)
    for k in range(SHAPE[0]):
        rows, cols = np.nonzero(patch[k])
        if len(rows):
            source.edits.set_plane(k, rows, cols, np.ones(len(rows), np.uint8))
    return int(patch.sum())


def _gap_box_um(frame, margin_vox=1):
    lo = frame.seg_to_um([[GAP[0] - margin_vox, AXIS_Y - RADIUS_VOX - margin_vox,
                           AXIS_Z - RADIUS_VOX - margin_vox]])[0]
    hi = frame.seg_to_um([[GAP[1] - 1 + margin_vox, AXIS_Y + RADIUS_VOX + margin_vox,
                           AXIS_Z + RADIUS_VOX + margin_vox]])[0]
    return np.array([lo, hi])


# --------------------------------------------------------------- the fixture


def test_the_fixture_really_is_broken(broken_source, broken_graph):
    from hipct_seg_debug.edit.reconnect.segmentation import components

    volume = np.stack([broken_source.slice_z(k) for k in range(SHAPE[0])])
    assert components(volume > 0).n == 2
    assert len(broken_graph.components()) == 2


def test_painting_reconnects_the_mask(broken_source):
    from hipct_seg_debug.edit.reconnect.segmentation import components

    assert _paint_the_gap(broken_source) > 0
    volume = np.stack([broken_source.slice_z(k) for k in range(SHAPE[0])])
    assert components(volume > 0).n == 1


# ------------------------------------------------------------------ add mode


def test_add_mode_joins_the_two_components(broken_source, broken_graph, frame):
    _paint_the_gap(broken_source)
    before = {seg["id"]: broken_graph.coords(seg["id"]).copy()
              for seg in broken_graph.segments}

    report = reskeletonise_box(broken_graph, broken_source, frame,
                               _gap_box_um(frame), mode="add")

    assert report.applied, report.reason
    assert report.segments_added == 1
    assert report.segments_deleted == 0
    assert report.ends_welded == 2
    assert report.ends_free == 0
    assert len(broken_graph.components()) == 1

    # Avizo's geometry either side is untouched, point for point.
    for sid, coords in before.items():
        assert broken_graph.has_segment(sid)
        assert np.allclose(broken_graph.coords(sid), coords)


def test_the_new_segment_spans_the_gap(broken_source, broken_graph, frame):
    _paint_the_gap(broken_source)
    old = {seg["id"] for seg in broken_graph.segments}
    reskeletonise_box(broken_graph, broken_source, frame, _gap_box_um(frame), mode="add")

    new = [seg["id"] for seg in broken_graph.segments if seg["id"] not in old]
    assert len(new) == 1
    coords = broken_graph.coords(new[0])
    x_lo, x_hi = coords[:, 0].min(), coords[:, 0].max()
    # It must reach both stubs: their tips are at voxels GAP[0]-1 and GAP[1].
    assert x_lo <= (GAP[0] - 1) * SPACING + 1e-6
    assert x_hi >= GAP[1] * SPACING - 1e-6
    # ...and stay on the axis.
    assert np.allclose(coords[:, 1], AXIS_Y * SPACING, atol=SPACING)
    assert np.allclose(coords[:, 2], AXIS_Z * SPACING, atol=SPACING)


def test_the_new_radii_are_sane(broken_source, broken_graph, frame):
    _paint_the_gap(broken_source)
    old = {seg["id"] for seg in broken_graph.segments}
    reskeletonise_box(broken_graph, broken_source, frame, _gap_box_um(frame), mode="add")
    new = [seg["id"] for seg in broken_graph.segments if seg["id"] not in old][0]
    radii = broken_graph.radii(new)
    # The distance transform of a radius-3 tube peaks near 3 voxels on the axis.
    assert (radii > 0).all()
    assert abs(np.median(radii) - RADIUS_VOX * SPACING) < 1.5 * SPACING


def test_add_mode_declines_when_nothing_was_painted(broken_source, broken_graph, frame):
    report = reskeletonise_box(broken_graph, broken_source, frame,
                               _gap_box_um(frame), mode="add")
    assert not report.applied
    assert "painted" in report.reason
    assert len(broken_graph.components()) == 2


def test_the_whole_thing_is_one_undo_step(broken_source, broken_graph, frame):
    _paint_the_gap(broken_source)
    before = len(broken_graph.segments)
    reskeletonise_box(broken_graph, broken_source, frame, _gap_box_um(frame), mode="add")
    assert len(broken_graph.segments) == before + 1

    assert broken_graph.undo() is not None
    assert len(broken_graph.segments) == before
    assert len(broken_graph.components()) == 2


def test_an_unreachable_fragment_is_reported_not_hidden(broken_source, frame):
    """A painted blob with no graph anywhere near it welds nothing, and says so."""
    _paint_the_gap(broken_source)
    empty = EditableGraph(Triple(nodes={}, points={}, segments=[]))
    report = reskeletonise_box(empty, broken_source, frame, _gap_box_um(frame),
                               mode="add", weld_um=1.0)
    assert report.applied
    assert report.ends_welded == 0
    assert report.ends_free == 2


# -------------------------------------------------------------- replace mode


def test_replace_mode_rebuilds_the_box(broken_source, broken_graph, frame):
    _paint_the_gap(broken_source)
    box = _gap_box_um(frame, margin_vox=4)
    report = reskeletonise_box(broken_graph, broken_source, frame, box, mode="replace")

    assert report.applied, report.reason
    assert report.segments_added >= 1
    assert len(broken_graph.components()) == 1
    # Nothing may be left lying inside the box except what was just inserted.
    from hipct_seg_debug.edit.reskeletonise import _inside

    stale = [seg["id"] for seg in broken_graph.segments
             if _inside(broken_graph.coords(seg["id"]), box).all()]
    assert len(stale) <= report.segments_added


def test_replace_mode_needs_no_paint_at_all(broken_source, broken_graph, frame):
    report = reskeletonise_box(broken_graph, broken_source, frame,
                               _gap_box_um(frame, margin_vox=4), mode="replace")
    # The mask still has a hole there, so there may be nothing to trace -- but it
    # must not refuse for the *add*-mode reason.
    assert "painted" not in report.reason


def test_clear_box_splits_a_straddling_segment(broken_graph, frame):
    """A vessel crossing the boundary is cut, not left duplicating the new one."""
    # A box over the middle of the first stub only.
    lo = frame.seg_to_um([[12, 0, 0]])[0]
    hi = frame.seg_to_um([[20, SHAPE[1] - 1, SHAPE[0] - 1]])[0]
    box = np.array([lo, hi])
    with broken_graph.batch("clear"):
        clear_box(broken_graph, box)

    from hipct_seg_debug.edit.reskeletonise import _inside

    for seg in broken_graph.segments:
        assert not _inside(broken_graph.coords(seg["id"]), box).all()


def test_bad_mode_is_rejected(broken_source, broken_graph, frame):
    with pytest.raises(ValueError, match="add.*replace"):
        reskeletonise_box(broken_graph, broken_source, frame,
                          _gap_box_um(frame), mode="sideways")


# ------------------------------------------------------------- voxel stages


def test_a_painted_bump_adds_no_branch(broken_source, broken_graph, frame):
    """A short spur off the vessel is a thinning artefact, not a daughter.

    ``skeleton_to_graph``'s own ``min_branch_voxels`` is what drops it -- a
    voxel-level prune cannot, because a spur one voxel off a 26-connected line is
    itself adjacent to three line voxels and so reads as a junction.
    """
    _paint_the_gap(broken_source)
    # A two-voxel blister on the side of the repaired tube.
    for k in (AXIS_Z,):
        broken_source.edits.set_plane(
            k, [AXIS_Y + RADIUS_VOX + 1, AXIS_Y + RADIUS_VOX + 2], [34, 34], [1, 1]
        )
    report = reskeletonise_box(broken_graph, broken_source, frame,
                               _gap_box_um(frame), mode="add")
    assert report.applied, report.reason
    assert report.segments_added == 1


def test_contract_degree2_rejoins_a_split_chain(frame):
    """What a dropped branch leaves behind must not survive as a node."""
    from hipct_seg_debug.edit.reskeletonise import contract_degree2

    xyz = np.stack([np.arange(6) * SPACING, np.zeros(6), np.zeros(6)], axis=1)
    points = {n: (*map(float, p), 50.0) for n, p in enumerate(xyz)}
    local = Triple(
        nodes={0: (*map(float, xyz[0]), 0), 1: (*map(float, xyz[3]), 0),
               2: (*map(float, xyz[5]), 0)},
        points=points,
        segments=[
            {"id": 0, "node1": 0, "node2": 1, "point_ids": [0, 1, 2, 3]},
            {"id": 1, "node1": 1, "node2": 2, "point_ids": [3, 4, 5]},
        ],
    )
    out = contract_degree2(local, eps_um=SPACING)
    assert len(out.segments) == 1
    assert 1 not in out.nodes
    assert {out.segments[0]["node1"], out.segments[0]["node2"]} == {0, 2}
    coords = np.array([out.points[p][:3] for p in out.segments[0]["point_ids"]])
    assert np.allclose(np.sort(coords[:, 0]), xyz[:, 0])


def test_contract_degree2_leaves_a_would_be_loop_alone(frame):
    """Merging both halves of a ring into itself is never what was meant."""
    from hipct_seg_debug.edit.reskeletonise import contract_degree2

    pts = {n: (float(n), 0.0, 0.0, 5.0) for n in range(6)}
    local = Triple(
        nodes={0: (0.0, 0.0, 0.0, 0), 1: (3.0, 0.0, 0.0, 0)},
        points=pts,
        segments=[
            {"id": 0, "node1": 0, "node2": 1, "point_ids": [0, 1, 2]},
            {"id": 1, "node1": 1, "node2": 0, "point_ids": [3, 4, 5]},
        ],
    )
    assert len(contract_degree2(local, eps_um=1.0).segments) == 2


# ------------------------------------------------------------------- trimming


def test_trim_drops_the_overlap_and_keeps_one_anchor(broken_graph, frame):
    """The fragment is grown into covered territory on purpose; it must come back."""
    xs = np.arange(GAP[0] - 6, GAP[1] + 6)
    ijk = np.stack([xs, np.full_like(xs, AXIS_Y), np.full_like(xs, AXIS_Z)], axis=1)
    xyz = frame.seg_to_um(ijk)
    points = {n: (*map(float, p), 100.0) for n, p in enumerate(xyz)}
    local = Triple(
        nodes={0: (*map(float, xyz[0]), 0), 1: (*map(float, xyz[-1]), 0)},
        points=points,
        segments=[{"id": 0, "node1": 0, "node2": 1, "point_ids": list(points)}],
    )

    trimmed, removed = trim_to_new(local, broken_graph, cover_um=0.75 * SPACING)
    assert removed > 0
    kept = trimmed.segments[0]["point_ids"]
    coords = np.array([trimmed.points[p][:3] for p in kept])
    # One anchor point either side of the genuinely-new run.
    assert coords[:, 0].min() == pytest.approx((GAP[0] - 1) * SPACING)
    assert coords[:, 0].max() == pytest.approx(GAP[1] * SPACING)


def test_trim_drops_a_fragment_that_is_entirely_covered(broken_graph, frame):
    xs = np.arange(TUBE[0] + 2, TUBE[0] + 10)
    ijk = np.stack([xs, np.full_like(xs, AXIS_Y), np.full_like(xs, AXIS_Z)], axis=1)
    xyz = frame.seg_to_um(ijk)
    points = {n: (*map(float, p), 100.0) for n, p in enumerate(xyz)}
    local = Triple(
        nodes={0: (*map(float, xyz[0]), 0), 1: (*map(float, xyz[-1]), 0)},
        points=points,
        segments=[{"id": 0, "node1": 0, "node2": 1, "point_ids": list(points)}],
    )
    trimmed, removed = trim_to_new(local, broken_graph, cover_um=0.75 * SPACING)
    assert not trimmed.segments
    assert removed == len(points)


def test_trim_is_a_no_op_against_an_empty_graph(frame):
    empty = EditableGraph(Triple(nodes={}, points={}, segments=[]))
    local = Triple(nodes={0: (0.0, 0.0, 0.0, 0), 1: (10.0, 0.0, 0.0, 0)},
                   points={0: (0.0, 0.0, 0.0, 5.0), 1: (10.0, 0.0, 0.0, 5.0)},
                   segments=[{"id": 0, "node1": 0, "node2": 1, "point_ids": [0, 1]}])
    trimmed, removed = trim_to_new(local, empty, cover_um=1.0)
    assert removed == 0 and trimmed is local


def test_select_painted_keeps_only_the_reachable_chain():
    skel = np.zeros((5, 5, 40), dtype=bool)
    skel[2, 2, :] = True
    added = np.zeros_like(skel)
    added[2, 2, 20] = True
    kept = select_painted(skel, added, np.array([10.0, 10.0, 10.0]), grow_um=50.0)
    # 5 steps of dilation either side of the seed, restricted to the skeleton.
    assert kept[2, 2, 20]
    assert kept[2, 2, 15] and kept[2, 2, 25]
    assert not kept[2, 2, 0] and not kept[2, 2, 39]


def test_select_painted_is_empty_when_the_paint_misses_the_skeleton():
    skel = np.zeros((5, 5, 20), dtype=bool)
    skel[2, 2, :] = True
    added = np.zeros_like(skel)
    added[0, 0, 0] = True
    assert not select_painted(skel, added, np.array([10.0] * 3), 50.0).any()
