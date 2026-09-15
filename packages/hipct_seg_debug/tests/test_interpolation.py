"""What counts as a point Avizo invented, and what the toolkit is allowed to do about it.

Two things are being defended here. The first is *specificity*: a straight vessel is
common and a straight vessel is not an artefact, so most of these tests are cases the
detector must decline to flag. The second is that the split is reversible -- reconnection
gets to see the break, but nothing is lost if the image cannot justify a better join.

The last two tests run against the real Avizo export and assert the counts measured when
this was written. They are the reason a threshold cannot be loosened quietly.
"""

from __future__ import annotations


import numpy as np
import pytest

from hipct_seg_debug.edit import interpolation as ip
from hipct_seg_debug.edit.adapter import Triple, read_triple, to_spatial_graph
from hipct_seg_debug.edit.amira_write import write_spatial_graph
from hipct_seg_debug.edit.graphmodel import EditableGraph

from .realdata import (
    GRAPH_ALT_REASON,
    GRAPH_REASON,
    REAL_AM,
    REAL_AM_ALT,
    REAL_SEG,
    SEG_REASON,
)


STEP = 100.0  # um between consecutive points, as in the real resampled graphs


def _arc(n: int, *, curvature: float, x0: float = 0.0) -> np.ndarray:
    """`n` points along x, bowed in y so the run has genuine curvature."""
    x = x0 + STEP * np.arange(n, dtype=np.float64)
    return np.column_stack([x, curvature * (x - x0) ** 2 / STEP, np.zeros(n)])


def _one_segment(coords, radii) -> EditableGraph:
    coords = np.asarray(coords, dtype=np.float64)
    radii = np.asarray(radii, dtype=np.float64)
    points = {
        i: (float(p[0]), float(p[1]), float(p[2]), float(r))
        for i, (p, r) in enumerate(zip(coords, radii))
    }
    nodes = {0: (*coords[0], 0), 1: (*coords[-1], 0)}
    segments = [{"id": 0, "node1": 0, "node2": 1, "point_ids": list(range(len(coords)))}]
    return EditableGraph(Triple(nodes, points, segments))


def bridged_graph(*, n_fill: int = 8, discontinuous: bool = True) -> EditableGraph:
    """A curved vessel with a straight, linearly-tapered fill spliced into the middle.

    Shaped like the real artefact rather than merely straight. Edge 183 of LADAF-28 is
    the model: a nearly flat interior ramp (0.1% per step) meeting the real vessel with a
    20-30% step at each end, because the fill was generated without reference to either
    side. ``discontinuous=False`` keeps the same geometry but lets the radius run
    continuously through, which is what an ordinary straight vessel looks like.
    """
    left = _arc(6, curvature=0.35)
    right = _arc(6, curvature=0.35, x0=left[-1, 0] + STEP * (n_fill + 1))
    right[:, 1] += left[-1, 1] - right[0, 1] + STEP  # offset so the join is not straight

    t = np.linspace(0.0, 1.0, n_fill + 2)[1:-1][:, None]
    fill = left[-1] + t * (right[0] - left[-1])

    r_left = np.linspace(400.0, 380.0, len(left))
    r_right = np.linspace(300.0, 280.0, len(right))
    if discontinuous:
        r_fill = np.linspace(250.0, 247.0, n_fill)
    else:
        r_fill = np.linspace(380.0, 300.0, n_fill + 2)[1:-1]

    coords = np.vstack([left, fill, right])
    radii = np.concatenate([r_left, r_fill, r_right])
    return _one_segment(coords, radii)


# --------------------------------------------------------------------------- detection
def test_a_straight_linear_fill_between_two_jumps_is_flagged():
    graph = bridged_graph(n_fill=8)
    found = ip.detect(graph)
    spans = [s for s in found.spans if s.reason & ip.STRAIGHT_BRIDGE]
    assert len(spans) == 1
    span = spans[0]
    assert span.n_points == 8
    # The anchors are real points shared with the vessel either side, and stay unflagged.
    assert span.start == 6
    assert span.stop == 14


def test_the_same_fill_without_an_anchor_jump_is_not_flagged():
    """A straight, linearly-tapered run that agrees with its neighbours is just vessel."""
    graph = bridged_graph(n_fill=8, discontinuous=False)
    found = ip.detect(graph)
    assert not [s for s in found.spans if s.reason & ip.STRAIGHT_BRIDGE]


def test_a_steep_taper_does_not_fake_an_anchor_jump():
    """The absolute gate alone is not enough -- a thin vessel clears 5% per point.

    ``linspace(200, 400)`` over 20 points steps 5.3% each time, so both ends of any
    straight run inside it exceed ANCHOR_JUMP. What saves it is that the ends look
    exactly like the middle.
    """
    coords = np.column_stack([STEP * np.arange(20.0), np.zeros(20), np.zeros(20)])
    graph = _one_segment(coords, np.linspace(200.0, 400.0, 20))
    assert not [s for s in ip.detect(graph).spans if s.reason & ip.STRAIGHT_BRIDGE]


def test_a_plain_straight_constant_radius_segment_is_not_flagged():
    """What half the synthetic fixtures in this suite are made of."""
    coords = np.column_stack(
        [STEP * np.arange(30.0), np.zeros(30), np.zeros(30)]
    )
    graph = _one_segment(coords, np.full(30, 250.0))
    assert not ip.detect(graph).flags


def test_a_smoothly_tapering_straight_segment_is_not_flagged():
    coords = np.column_stack([STEP * np.arange(30.0), np.zeros(30), np.zeros(30)])
    graph = _one_segment(coords, np.linspace(400.0, 200.0, 30))
    assert not ip.detect(graph).flags


def test_a_short_fill_is_below_the_span_floor():
    graph = bridged_graph(n_fill=ip.MIN_SPAN_POINTS - 1)
    found = ip.detect(graph)
    assert not [s for s in found.spans if s.reason & ip.STRAIGHT_BRIDGE]


def test_a_detached_radius_floor_is_flagged_but_a_continuum_is_not():
    coords = np.column_stack([STEP * np.arange(20.0), np.zeros(20), np.zeros(20)])
    radii = np.linspace(200.0, 400.0, 20)

    continuous = _one_segment(coords, radii)
    assert ip.detect(continuous).floor_um is None
    assert not ip.detect(continuous).flags

    detached = radii.copy()
    detached[7] = 40.0  # well under 200 / FLOOR_GAP_RATIO
    graph = _one_segment(coords, detached)
    found = ip.detect(graph)
    assert found.floor_um == pytest.approx(40.0)
    assert found.counts()[ip.FLOOR_RADIUS] == 1


def test_a_zero_radius_is_always_degenerate():
    """A raw Avizo export writes no thickness at all, rather than a fitted intercept."""
    coords = np.column_stack([STEP * np.arange(10.0), np.zeros(10), np.zeros(10)])
    radii = np.full(10, 250.0)
    radii[4] = 0.0
    found = ip.detect(_one_segment(coords, radii))
    assert found.flags and all(v & ip.FLOOR_RADIUS for v in found.flags.values())


class _StripLabels:
    """A 1-D mask: one row of `n` voxels, zero at the given indices."""

    def __init__(self, n, outside):
        self.plane = np.ones((1, n), dtype=np.uint8)
        self.plane[0, list(outside)] = 0

    def slice_z(self, z):
        return self.plane


class _StripFrame:
    """Maps a point at x = i * STEP onto voxel i of that single row."""

    def __init__(self, n):
        self.seg_dims = np.array([n, 1, 1])

    def um_to_seg_index(self, xyz):
        x = np.asarray(xyz, dtype=np.float64)[:, 0]
        return np.column_stack(
            [np.rint(x / STEP).astype(np.int64), np.zeros((len(x), 2), dtype=np.int64)]
        )


def _strip(n=12):
    coords = np.column_stack([STEP * np.arange(float(n)), np.zeros(n), np.zeros(n)])
    return _one_segment(coords, np.full(n, 250.0))


def test_an_isolated_off_mask_point_is_not_a_bridge():
    """One point outside the mask is a smoother clipping a bend, not an invented vessel."""
    found = ip.detect(_strip(), labels=_StripLabels(12, [4]), frame=_StripFrame(12))
    assert found.checked_mask
    assert not found.flags


def test_a_run_outside_the_mask_is_a_bridge():
    found = ip.detect(_strip(), labels=_StripLabels(12, [4, 5, 6]), frame=_StripFrame(12))
    spans = [s for s in found.spans if s.reason & ip.OFF_MASK]
    assert len(spans) == 1
    assert (spans[0].start, spans[0].n_points) == (4, 3)
    assert spans[0].off_mask_fraction == pytest.approx(1.0)


# ------------------------------------------------------------------------ persistence
def test_flags_are_not_computed_on_demand():
    """A graph nobody has flagged reports nothing, however obvious the artefact is."""
    graph = bridged_graph()
    assert not ip.has_flags(graph)
    assert ip.flags(graph) == {}
    assert not ip.mask(graph).any()


def test_annotate_survives_the_amira_round_trip(tmp_path):
    graph = bridged_graph()
    found = ip.detect(graph)
    ip.annotate(graph, found)
    assert ip.has_flags(graph)

    path = write_spatial_graph(to_spatial_graph(graph.triple), tmp_path / "flagged.am")
    back = read_triple(path)

    assert ip.has_flags(back)
    assert ip.flagged_points(back) == ip.flagged_points(graph)
    assert [s.n_points for s in ip.spans(back)] == [s.n_points for s in ip.spans(graph)]


def test_mask_lines_up_with_point_order():
    graph = bridged_graph()
    ip.annotate(graph, ip.detect(graph))
    flat = ip.mask(graph)
    assert flat.sum() == 8
    assert np.array_equal(flat, ip.mask_for_segment(graph, 0))


# ------------------------------------------------------------------------ the split
def test_splitting_an_interior_fill_makes_two_real_free_ends():
    graph = bridged_graph()
    ip.annotate(graph, ip.detect(graph))
    before_ends, before_components = len(graph.endpoints()), len(graph.components())

    records = ip.split_flagged(graph)

    assert len(records) == 1
    assert not records[0].whole_edge
    assert len(graph.endpoints()) == before_ends + 2
    assert len(graph.components()) == before_components + 1
    # The fill is gone; the twelve real points either side, plus the duplicated joint
    # each split leaves behind, are not.
    assert not ip.flagged_points(graph)
    assert not ip.spans(graph)
    assert sum(len(s["point_ids"]) for s in graph.segments) == 12


def test_restore_puts_back_a_fill_that_nothing_replaced():
    graph = bridged_graph()
    ip.annotate(graph, ip.detect(graph))
    ends, components = len(graph.endpoints()), len(graph.components())

    records = ip.split_flagged(graph)
    restored = ip.restore_unbridged(graph, records)

    assert len(restored) == 1
    assert len(graph.endpoints()) == ends
    assert len(graph.components()) == components
    # Restored through `add_segment`, so the point ids are new -- but the interior is
    # flagged again, and the two anchors are still real.
    span = [s for s in ip.spans(graph)]
    assert len(span) == 1
    assert span[0].n_points == 8


def test_restore_leaves_a_fill_alone_once_the_two_sides_are_joined():
    graph = bridged_graph()
    ip.annotate(graph, ip.detect(graph))
    records = ip.split_flagged(graph)

    # Stand in for a DPC-validated bridge across the same break.
    record = records[0]
    coords = np.linspace(record.coords[0], record.coords[-1], 5)
    graph.add_segment(record.node1, record.node2, coords, np.full(5, 320.0))

    assert ip.restore_unbridged(graph, records) == []


def test_a_lone_degenerate_radius_is_masked_but_never_split():
    coords = np.column_stack([STEP * np.arange(20.0), np.zeros(20), np.zeros(20)])
    radii = np.linspace(200.0, 400.0, 20)
    radii[7] = 40.0
    graph = _one_segment(coords, radii)
    ip.annotate(graph, ip.detect(graph))

    assert ip.mask(graph).sum() == 1
    before = len(graph.segments)
    assert ip.split_flagged(graph) == []
    assert len(graph.segments) == before


def test_splitting_a_whole_edge_fill_reports_itself_as_such():
    """A fill occupying an entire edge leaves its junctions at degree two, not one."""
    fill = np.column_stack([STEP * np.arange(10.0), np.zeros(10), np.zeros(10)])
    left = _arc(5, curvature=0.4)
    left = left - left[-1] + fill[0]
    right = _arc(5, curvature=0.4, x0=0.0) + fill[-1]

    points, segments = {}, []
    pid = 0

    def add(coords, radii, n1, n2):
        nonlocal pid
        ids = []
        for p, r in zip(coords, radii):
            points[pid] = (float(p[0]), float(p[1]), float(p[2]), float(r))
            ids.append(pid)
            pid += 1
        segments.append(
            {"id": len(segments), "node1": n1, "node2": n2, "point_ids": ids}
        )

    # The fill edge's own terminal points are shared with the junctions either side and
    # carry the *real* radius there, exactly as Avizo stores it -- which is what puts the
    # discontinuity inside this edge's point list where the detector can see it.
    add(left, np.linspace(500.0, 400.0, 5), 0, 1)
    add(fill, [400.0, *np.linspace(250.0, 247.0, 8), 390.0], 1, 2)
    add(right, np.linspace(390.0, 300.0, 5), 2, 3)
    nodes = {
        0: (*left[0], 0), 1: (*fill[0], 0), 2: (*fill[-1], 0), 3: (*right[-1], 0),
    }
    graph = EditableGraph(Triple(nodes, points, segments))

    found = ip.detect(graph)
    spans = [s for s in found.spans if s.reason & ip.STRAIGHT_BRIDGE]
    assert len(spans) == 1 and spans[0].seg_id == 1

    ip.annotate(graph, found)
    records = ip.split_flagged(graph)
    assert len(records) == 1 and records[0].whole_edge
    assert len(graph.components()) == 2
    assert len(ip.restore_unbridged(graph, records)) == 1
    assert len(graph.components()) == 1


# ------------------------------------------------------------------- the real export
@pytest.mark.skipif(not REAL_AM.is_file(), reason=GRAPH_REASON)
def test_the_real_avizo_graph_flags_exactly_the_two_known_bridges():
    """Measured when this was written. A loosened threshold has to fail somewhere."""
    found = ip.detect(read_triple(REAL_AM))

    bridges = sorted(
        (s for s in found.spans if s.reason & ip.STRAIGHT_BRIDGE),
        key=lambda s: s.seg_id,
    )
    assert [(s.seg_id, s.start, s.n_points) for s in bridges] == [
        (183, 1, 11), (197, 1, 15)
    ]
    # Both sit far above the gate: the nearest non-artefact run is around 0.3%.
    assert all(s.anchor_jump > 0.15 for s in bridges)

    assert found.floor_um == pytest.approx(81.269, abs=1e-3)
    assert found.counts()[ip.FLOOR_RADIUS] == 10
    assert found.n_flagged == 36


@pytest.mark.skipif(not REAL_AM_ALT.is_file(), reason=GRAPH_ALT_REASON)
def test_the_second_scan_agrees_at_the_same_thresholds():
    """A different scan, no retuning: one bridge and a floor cluster of its own."""
    found = ip.detect(read_triple(REAL_AM_ALT))

    bridges = [s for s in found.spans if s.reason & ip.STRAIGHT_BRIDGE]
    assert [(s.seg_id, s.n_points) for s in bridges] == [(216, 6)]
    assert found.floor_um == pytest.approx(78.4242, abs=1e-3)
    assert found.counts()[ip.FLOOR_RADIUS] == 8


# ------------------------------------------------------------------- unsampled jumps
# The fourth signature, and the only one that cannot be seen from the graph alone. Avizo
# joins two traced runs with a single enormous step and *no points between them*, so
# there is nothing for the three point signatures to flag -- on LADAF-28 they miss all of
# these, which is why they were found by measuring step length against the mask instead.


class _FakeComponents:
    """A component index over a 1-D strip: voxel i belongs to ``labels[i]``."""

    def __init__(self, labels):
        self.labels = list(labels)

    def nearest_label(self, z, y, x, radius):
        if 0 <= x < len(self.labels) and self.labels[x]:
            return int(self.labels[x]), 0.0
        return 0, float("inf")


class _JumpFrame:
    """World x -> voxel index, one voxel per :data:`STEP`."""

    seg_spacing = np.array([STEP, STEP, STEP])

    def um_to_seg(self, xyz):
        arr = np.asarray(xyz, dtype=np.float64).reshape(-1, 3)
        return np.column_stack([arr[:, 0] / STEP, np.zeros(len(arr)), np.zeros(len(arr))])


def _jump_graph(gap_um, n=8):
    """Two runs of `n` points with one big step between them, all on one segment."""
    left = STEP * np.arange(n, dtype=np.float64)
    right = left[-1] + gap_um + STEP * np.arange(n, dtype=np.float64)
    x = np.concatenate([left, right])
    coords = np.column_stack([x, np.zeros(len(x)), np.zeros(len(x))])
    return _one_segment(coords, np.full(len(x), 250.0))


def _split_components(gap_um, n=8):
    """Component labels: everything left of the gap is 1, everything right is 2."""
    total = int(round((2 * n * STEP + gap_um) / STEP)) + 2
    left_end = n
    right_start = int(round((STEP * (n - 1) + gap_um) / STEP))
    return _FakeComponents(
        [1 if i < left_end else (2 if i >= right_start else 0) for i in range(total)]
    )


def test_a_big_step_across_a_mask_break_is_a_jump():
    gap = 4000.0
    found = ip.detect(_jump_graph(gap), components=_split_components(gap),
                      frame=_JumpFrame())
    assert found.checked_jumps
    assert len(found.jumps) == 1
    jump = found.jumps[0]
    assert jump.is_jump and jump.n_points == 0 and jump.point_ids == []
    assert jump.reason == ip.UNSAMPLED_JUMP
    assert jump.length_um == pytest.approx(gap)
    # It marks the *step*, so the anchors are ids[start - 1] and ids[start].
    assert jump.start == 8


def test_a_big_step_inside_one_mask_component_is_only_coarse_sampling():
    """Under-sampling is not a break. Without the mask test this would be a false find."""
    gap = 4000.0
    n = 8
    total = int(round((2 * n * STEP + gap) / STEP)) + 2
    found = ip.detect(_jump_graph(gap), components=_FakeComponents([1] * total),
                      frame=_JumpFrame())
    assert found.checked_jumps
    assert found.jumps == []


def test_ordinary_spacing_is_never_a_jump():
    found = ip.detect(_strip(12), components=_FakeComponents([1] * 6 + [2] * 8),
                      frame=_JumpFrame())
    assert found.jumps == []


def test_a_step_below_the_absolute_floor_is_left_alone():
    """Measured: every LADAF-28 step over 600 um crosses a break, none below it does."""
    gap = 400.0
    found = ip.detect(_jump_graph(gap), components=_split_components(gap),
                      frame=_JumpFrame())
    assert found.jumps == []


def test_the_jump_signature_is_off_unless_components_are_supplied():
    found = ip.detect(_jump_graph(4000.0))
    assert not found.checked_jumps
    assert found.jumps == []
    assert "unsampled jumps" in found.describe()


def test_a_jump_does_not_contaminate_the_measurement_mask():
    """The load-bearing separation.

    ``FIELD`` means "this point is invented, keep it out of every measurement". A jump's
    two anchors are ordinary measured centreline -- it is the empty step between them
    that is fabricated -- so folding the jump into ``FIELD`` would silently drop two real
    points from every radius and length statistic in the toolkit.
    """
    gap = 4000.0
    graph = _jump_graph(gap)
    found = ip.detect(graph, components=_split_components(gap), frame=_JumpFrame())
    ip.annotate(graph, found)

    assert found.jumps
    assert not ip.mask(graph).any(), "a jump anchor was marked as an invented point"
    assert ip.flagged_points(graph) == set()
    assert len(ip.jump_flags(graph)) == 1


def test_a_jump_round_trips_through_the_amira_file(tmp_path):
    gap = 4000.0
    graph = _jump_graph(gap)
    ip.annotate(graph, ip.detect(graph, components=_split_components(gap),
                                 frame=_JumpFrame()))
    path = tmp_path / "jump.am"
    write_spatial_graph(to_spatial_graph(graph.to_triple()), path)

    reloaded = EditableGraph(read_triple(path))
    assert ip.has_flags(reloaded)
    jumps = [s for s in ip.spans(reloaded) if s.is_jump]
    assert len(jumps) == 1
    assert jumps[0].length_um == pytest.approx(gap)
    assert jumps[0].start == 8


def test_splitting_a_jump_makes_two_real_free_ends():
    """The whole point: after this the reconnector can finally see the break."""
    gap = 4000.0
    graph = _jump_graph(gap)
    ip.annotate(graph, ip.detect(graph, components=_split_components(gap),
                                 frame=_JumpFrame()))
    before_ends = len(graph.endpoints())
    before_components = len(graph.components())

    records = ip.split_flagged(graph)
    assert len(records) == 1
    assert records[0].reason == ip.UNSAMPLED_JUMP
    assert len(graph.components()) == before_components + 1
    assert len(graph.endpoints()) == before_ends + 2


def test_two_jumps_in_one_segment_are_both_cut():
    """The second one used to vanish without a word.

    `split_segment` retires the segment id, so a span still holding the old one is
    dropped by `has_segment`. For a run of points the stale-index guard catches that --
    but a jump has no points, so the guard compares two empty lists and passes. Three
    LADAF-28 segments carry two jumps each, and all three second cuts were being lost.
    """
    n = 6
    gap = 4000.0
    xs = [STEP * i for i in range(n)]
    xs += [xs[-1] + gap + STEP * i for i in range(n)]
    xs += [xs[-1] + gap + STEP * i for i in range(n)]
    coords = np.column_stack([xs, np.zeros(len(xs)), np.zeros(len(xs))])
    graph = _one_segment(coords, np.full(len(xs), 250.0))

    # Three mask components, one per run, with nothing in the two gaps.
    labels = []
    for x in range(int(round(xs[-1] / STEP)) + 2):
        um = x * STEP
        if um <= xs[n - 1] + 1:
            labels.append(1)
        elif xs[n] - 1 <= um <= xs[2 * n - 1] + 1:
            labels.append(2)
        elif um >= xs[2 * n] - 1:
            labels.append(3)
        else:
            labels.append(0)

    found = ip.detect(graph, components=_FakeComponents(labels), frame=_JumpFrame())
    assert len(found.jumps) == 2
    assert all(j.anchors is not None for j in found.jumps)
    ip.annotate(graph, found)

    records = ip.split_flagged(graph)
    assert len([r for r in records if r.reason == ip.UNSAMPLED_JUMP]) == 2
    assert len(graph.components()) == 3


def test_a_split_jump_is_restored_when_nothing_bridged_it():
    """Cutting is a question, not a decision -- an unanswered one is put back."""
    gap = 4000.0
    graph = _jump_graph(gap)
    ip.annotate(graph, ip.detect(graph, components=_split_components(gap),
                                 frame=_JumpFrame()))
    records = ip.split_flagged(graph)
    assert len(graph.components()) == 2

    restored = ip.restore_unbridged(graph, records)
    assert len(restored) == 1
    assert len(graph.components()) == 1
    # Restoring must not have marked the two real anchors as invented either.
    assert not ip.mask(graph).any()


def test_a_lone_degenerate_radius_still_never_splits_beside_a_jump():
    """The minimum-span rule is relaxed for jumps only, not for everything."""
    gap = 4000.0
    graph = _jump_graph(gap)
    found = ip.detect(graph, components=_split_components(gap), frame=_JumpFrame())
    # Add a single floor-radius point by hand; it must stay unsplit.
    pid = graph.segments[0]["point_ids"][3]
    found.flags[pid] = ip.FLOOR_RADIUS
    found.spans.append(ip.Span(seg_id=graph.segments[0]["id"], start=3, n_points=1,
                               point_ids=[pid], reason=ip.FLOOR_RADIUS))
    ip.annotate(graph, found)

    records = ip.split_flagged(graph)
    assert [r.reason for r in records] == [ip.UNSAMPLED_JUMP]





@pytest.mark.slow
@pytest.mark.skipif(not (REAL_AM.is_file() and REAL_SEG.is_file()),
                    reason=f"{GRAPH_REASON}; {SEG_REASON}")
def test_the_real_export_carries_two_dozen_unsampled_jumps():
    """Measured when this was written, and the reason the signature exists.

    The stock graph reports two connected components. Its own free ends sit on thirty
    different mask components, and these twenty-four steps are where it asserts
    continuity the segmentation does not have -- 72 mm of it, none of which any
    point-based signature can see.

    Marked slow: unlike its neighbours this one needs the 2.34 GB lattice labelled.
    """
    from hipct_seg_debug import amira, rle
    from hipct_seg_debug.edit.reconnect.geodesic import components as components_mod
    from hipct_seg_debug.frame import WorldFrame

    info = amira.read_lattice_header(REAL_SEG)
    labels = rle.ByteRLELattice(REAL_SEG, info.fields["Labels"], info.dims)
    frame = WorldFrame.from_inputs(
        (int(info.dims[2]), int(info.dims[1]), int(info.dims[0])),
        float(info.spacing[0]) / 2.0, info,
    )
    index = components_mod.build(labels)
    assert index.n == 55

    found = ip.detect(read_triple(REAL_AM), components=index, frame=frame)
    assert found.checked_jumps
    assert len(found.jumps) == 24
    assert sum(s.length_um for s in found.jumps) / 1000.0 == pytest.approx(72.4, abs=0.5)

    # The point-based signatures are unchanged by its presence, and still see none of it.
    assert found.n_flagged == 36
    assert found.counts()[ip.STRAIGHT_BRIDGE] == 26


def test_a_zero_point_jump_is_screened_without_the_mask():
    """The artefact the three point signatures are blind to, seen from the graph alone.

    `detect_jumps` needs the segmentation to *confirm* a jump crosses a break, and
    that gate is right. But gating the whole check on having the mask meant a graph
    carrying 24 invented bridges and 72.4 mm of centreline through empty space was
    reported as "3 points in 1 span" -- indistinguishable from clean. The geometric
    half of the gate needs no image and is what makes the count visible.
    """
    coords = np.column_stack([STEP * np.arange(20.0), np.zeros(20), np.zeros(20)])
    coords[10:, 0] += 9000.0  # one enormous step, with no points written across it
    graph = _one_segment(coords, np.full(20, 250.0))

    assert not ip.detect(graph).flags, "no point-based signature can see this"

    screen = ip.candidate_jumps(graph)
    assert len(screen) == 1
    assert screen[0].n_points == 0, "the invented thing is the edge, not a point"
    assert screen[0].length_um == pytest.approx(9000.0 + STEP)


def test_the_screen_judges_a_coarse_segment_against_itself():
    """A uniformly coarse trunk is under-sampled, not bridged, and must stay quiet."""
    coords = np.column_stack([700.0 * np.arange(20.0), np.zeros(20), np.zeros(20)])
    graph = _one_segment(coords, np.full(20, 250.0))
    assert not ip.candidate_jumps(graph)
