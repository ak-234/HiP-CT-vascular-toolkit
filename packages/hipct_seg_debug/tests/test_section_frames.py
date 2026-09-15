"""The debug view of `radius-perimeter`'s cut planes must describe the real cut.

Everything here is checked against geometry known in closed form, because the point of
the view is to answer "is the window in the wrong place, or the wrong size?" and a
view that got either of those wrong would answer it confidently and incorrectly.

The two load-bearing cases are the last two: a centreline deliberately run off the
axis of a straight tube must report the offset it actually has, and a window forced
too small must come back `truncated` rather than quietly measuring a cropped section.
"""

from __future__ import annotations

import numpy as np
import pytest

from hipct_seg_debug.edit import section_frames as sf

from .conftest_geometry import SPACING, axis_graph, cylinder, make_frame

pytest.importorskip("cv2")

SHAPE = (40, 40, 80)


@pytest.fixture
def frame():
    return make_frame(SHAPE)


def _survey(frame, mask, graph, **kw):
    kw.setdefault("stride", 4)
    return sf.survey(graph, frame, mask, **kw)


# ------------------------------------------------------------------- sampling


def test_both_ends_are_always_cut():
    """They are where the junction mask bites, so a sampler that dropped them lies."""
    idx = sf.sample_indices(20, 7)
    assert idx[0] == 0 and idx[-1] == 19
    assert list(idx) == [0, 7, 14, 19]


def test_the_frame_cap_stops_the_walk_and_says_so(frame):
    mask = cylinder(SHAPE, 6, 5, 75)
    graph = axis_graph(frame, 8, 72, 6 * SPACING, cy=20, cz=20)
    survey = _survey(frame, mask, graph, stride=1, max_frames=5)

    assert len(survey.frames) == 5
    assert survey.truncated_by_cap
    assert "frame cap" in survey.describe()


# ------------------------------------------------------------------- geometry


def test_a_window_is_a_square_perpendicular_to_the_vessel(frame):
    mask = cylinder(SHAPE, 6, 5, 75)
    graph = axis_graph(frame, 8, 72, 6 * SPACING, cy=20, cz=20)
    survey = _survey(frame, mask, graph)
    f = survey.frames[len(survey.frames) // 2]

    corners = f.corners_um
    assert corners.shape == (4, 3)
    sides = np.linalg.norm(np.diff(np.vstack([corners, corners[:1]]), axis=0), axis=1)
    assert sides == pytest.approx(sides[0] * np.ones(4), rel=1e-6)
    assert sides[0] == pytest.approx(2 * f.half_um, rel=1e-6)
    # The tube runs along x, so its sections lie in the y-z plane: no corner may sit
    # off the plane through the point.
    assert np.allclose(corners[:, 0], f.point_um[0], atol=1e-6)
    assert f.centroid_um is not None and f.contour_um is not None
    # A closed ring, and one that stays inside its own window.
    assert np.allclose(f.contour_um[0], f.contour_um[-1])
    assert f.extent_ratio < 1.0


def test_a_centred_cut_measures_the_tube_and_reports_no_offset(frame):
    r_vox = 6
    mask = cylinder(SHAPE, r_vox, 5, 75)
    graph = axis_graph(frame, 8, 72, r_vox * SPACING, cy=20, cz=20)
    survey = _survey(frame, mask, graph)

    assert survey.counts()["accepted"] == len(survey.frames)
    radii = np.array([f.radius_um for f in survey.frames])
    assert np.median(radii) == pytest.approx(r_vox * SPACING, rel=0.15)
    # The centreline is on the axis, so its own section's centroid is under it: at
    # most half a voxel of rasterisation.
    offsets = np.array([f.offset_um for f in survey.frames])
    assert offsets.max() < SPACING


# ----------------------------------------------------- the two things it is for


def test_a_centreline_off_the_axis_reports_the_offset_it_has(frame):
    """The re-centring `radius-perimeter` does not do, measured in um.

    The tube is centred on row 20 and the centreline is run along row 23, so every
    section's centroid sits three voxels from the point it was cut at. That is what
    the offset line drawn in 3D is showing, and it must be that number rather than a
    proxy for it.
    """
    r_vox = 7
    mask = cylinder(SHAPE, r_vox, 5, 75, cy=20, cz=20)
    graph = axis_graph(frame, 8, 72, r_vox * SPACING, cy=23, cz=20)
    survey = _survey(frame, mask, graph)

    offsets = np.array([f.offset_um for f in survey.frames if f.measured])
    assert offsets.size
    assert np.median(offsets) == pytest.approx(3 * SPACING, abs=SPACING)
    # And the centroid it points at is the tube's axis, not somewhere else in plane.
    f = next(f for f in survey.frames if f.measured)
    assert f.centroid_um[1] == pytest.approx(20 * SPACING, abs=SPACING)


def test_a_window_too_small_is_reported_truncated_not_measured(frame):
    """A cropped section still has a perimeter, which is exactly the danger."""
    r_vox = 8
    mask = cylinder(SHAPE, r_vox, 5, 75)
    graph = axis_graph(frame, 8, 72, r_vox * SPACING, cy=20, cz=20)

    survey = _survey(frame, mask, graph, initial_half=3, max_half=3)
    assert survey.counts()["truncated"] == len(survey.frames)
    assert all(not f.measured for f in survey.frames)
    # The window that failed is still drawn, at the size that failed: that is the
    # whole point of showing refused sections.
    assert all(f.half == 3 for f in survey.frames)


def test_growing_the_window_turns_a_truncated_section_into_a_measured_one(frame):
    """The direct test the panel's 'start half' box exists for."""
    r_vox = 8
    mask = cylinder(SHAPE, r_vox, 5, 75)
    graph = axis_graph(frame, 8, 72, r_vox * SPACING, cy=20, cz=20)

    tight = _survey(frame, mask, graph, initial_half=3, max_half=3)
    roomy = _survey(frame, mask, graph, initial_half=3, max_half=32)
    assert tight.counts()["accepted"] == 0
    assert roomy.counts()["accepted"] == len(roomy.frames)
    assert all(f.grew for f in roomy.frames), "it had to double to get there"


def test_the_legacy_selector_accepts_a_plain_closed_cut(frame):
    """With branch-aware off the plain cut *is* the selector, not a failed probe."""
    r_vox = 6
    mask = cylinder(SHAPE, r_vox, 5, 75)
    graph = axis_graph(frame, 8, 72, r_vox * SPACING, cy=20, cz=20)

    survey = _survey(frame, mask, graph, branch_aware=False)
    assert survey.counts()["accepted"] == len(survey.frames)
    assert all(f.measured for f in survey.frames)


def test_a_point_outside_the_volume_is_reported_not_skipped(frame):
    mask = cylinder(SHAPE, 6, 5, 75)
    graph = axis_graph(frame, 8, 72, 6 * SPACING, cy=20, cz=20)
    coords = graph.coords(0)
    coords[:, 0] += 10_000.0  # far off the end of the lattice
    for pid, xyz in zip(graph.segment(0)["point_ids"], coords):
        x, y, z, r = graph.triple.points[pid]
        graph.triple.points[pid] = (float(xyz[0]), y, z, r)

    survey = _survey(frame, mask, graph)
    assert survey.counts()["outside"] == len(survey.frames)
    assert all(f.contour_um is None for f in survey.frames)


# --------------------------------------------------------------- what it hands over


def test_drawables_splits_measured_from_refused(frame):
    r_vox = 6
    mask = cylinder(SHAPE, r_vox, 5, 40)  # the tube stops half way along the graph
    graph = axis_graph(frame, 8, 72, r_vox * SPACING, cy=20, cz=20)
    survey = _survey(frame, mask, graph)

    good, bad, contours, offsets, refused = sf.drawables(survey)
    assert len(good) + len(bad) == len(survey.frames)
    assert good and bad, "this fixture must produce both, or it tests nothing"
    assert all(np.asarray(q).shape == (4, 3) for q in good + bad)
    assert all(np.asarray(o).shape == (2, 3) for o in offsets)
    # The two boundary layers *partition* the frames that produced a ring: every ring
    # that exists is drawn exactly once, on the layer its verdict names. An earlier
    # version filtered the refused ones out entirely, which on a real run silently
    # dropped 163 of 203 sections and left the orange windows empty.
    assert len(contours) == sum(
        f.contour_um is not None for f in survey.frames if f.verdict == "accepted"
    )
    assert len(contours) + len(refused) == sum(
        f.contour_um is not None for f in survey.frames
    )
    # Here the refusals are `unmeasurable` -- the graph runs past the end of the tube,
    # so there is no blob to draw and `refused` is legitimately empty. A refused frame
    # contributes a ring only when it had a cut at all; see the truncation test below.
    assert not refused
    # Offsets are the exception and do not follow the boundary: a refused blob's
    # centroid is the centroid of a slab, and that layer reports a distance.
    assert len(offsets) <= len(good)


def test_the_summary_names_the_two_suspects(frame):
    r_vox = 6
    mask = cylinder(SHAPE, r_vox, 5, 75)
    graph = axis_graph(frame, 8, 72, r_vox * SPACING, cy=22, cz=20)
    survey = _survey(frame, mask, graph)

    text = survey.describe()
    assert "centroid offset" in text and "window fill" in text
    assert survey.table()[0].startswith("  seg") or "seg" in survey.table()[0]


# ------------------------------------------------------- collapsed or oblique

# The question the view exists to answer, and the one a shape measure alone cannot:
# an elongated section is either a lumen that is genuinely flat or a plane cut at the
# wrong angle, and the radius means completely different things in the two cases.


def test_a_refused_blob_keeps_its_window_and_goes_to_its_own_layer():
    """A grown slab drawn as a measured lumen is what makes a wrong axis look right.

    So it is drawn on a layer that says what it is instead. The two traps this pins
    are the ring, which must move rather than vanish, and the centroid, which must
    vanish rather than move: `offset_um` is finite for a refused frame, so a rule that
    simply followed the boundary would put a slab's centre on a layer that reports how
    far re-centring would move the point.
    """
    shape = (40, 40, 120)
    frame = make_frame(shape)
    mask = cylinder(shape, 6, 5, 115, cy=20, cz=20)
    graph = axis_graph(frame, 8, 112, 6 * SPACING, cy=20, cz=20)
    survey = _survey(frame, mask, graph)

    slab = sf.SectionFrame(
        sid=0, index=999, verdict="truncated",
        point_um=np.zeros(3), tangent=np.array([1.0, 0.0, 0.0]),
        corners_um=np.zeros((4, 3)), contour_um=np.zeros((5, 3)),
        centroid_um=np.ones(3), half=64, half_um=640.0,
        stored_radius_um=60.0, radius_um=float("nan"),
        offset_um=500.0, extent_ratio=1.0, grew=True, searched=False,
    )
    survey.frames.append(slab)
    good, bad, contours, offsets, refused = sf.drawables(survey)

    assert any(np.array_equal(q, slab.corners_um) for q in bad)
    assert not any(len(c) == 5 for c in contours), "not on the measured-lumen layer"
    assert any(len(c) == 5 for c in refused), "but drawn, on the refused one"
    assert not any(np.array_equal(o[1], slab.centroid_um) for o in offsets)
    assert len(offsets) <= len(good)


def test_a_refused_ring_is_the_blob_inside_its_own_window():
    """What the refused layer is *for*: the shape names which failure it was.

    A cut forced to truncate must still hand over a boundary, and that boundary has to
    sit inside the square it was cut in -- which is what makes "a blob filling its
    window" readable as "the window was too small" rather than as a measurement.
    Checked against the window's own half-width rather than a recorded number, because
    the containment is geometry and not policy.
    """
    shape = (40, 40, 120)
    frame = make_frame(shape)
    mask = cylinder(shape, 9, 5, 115, cy=20, cz=20)
    graph = axis_graph(frame, 8, 112, 9 * SPACING, cy=20, cz=20)
    # A window far smaller than the tube, and forbidden from growing to fit it.
    survey = sf.survey(graph, frame, mask, stride=8, initial_half=4, max_half=4)

    rings = [f for f in survey.frames if f.contour_um is not None]
    assert rings and not any(f.measured for f in rings), "the fixture must truncate"

    _good, _bad, contours, _offsets, refused = sf.drawables(survey)
    assert not contours
    # Same rings, same order: the panel relies on that correspondence and nothing
    # else checks it.
    assert len(refused) == len(rings)
    assert all(np.array_equal(a, f.contour_um) for a, f in zip(refused, rings))

    for f in rings:
        centre = f.corners_um.mean(axis=0)
        reach = np.linalg.norm(np.asarray(f.contour_um) - centre, axis=1).max()
        # The corner-to-centre distance of a square of half-width h is h*sqrt(2); one
        # voxel of slack covers the contour tracing pixel centres.
        assert reach <= f.half_um * np.sqrt(2) + SPACING


def test_a_straight_tube_reports_round_sections_and_no_obliquity(frame):
    r_vox = 6
    mask = cylinder(SHAPE, r_vox, 5, 75)
    graph = axis_graph(frame, 8, 72, r_vox * SPACING, cy=20, cz=20)
    survey = _survey(frame, mask, graph)

    shapes = np.array([f.axis_ratio for f in survey.frames if f.measured])
    assert shapes.size and np.median(shapes) < 1.15
    assert not any(f.searched for f in survey.frames)
    text = survey.describe()
    assert "section shape" in text and "obliquity" in text


def test_the_view_separates_a_collapsed_lumen_from_an_oblique_cut(frame):
    """Both are flat; only one of them straightens when the plane is rotated."""
    from .conftest_geometry import slit

    mask = slit(SHAPE, 8, 1, 5, 75, cy=20, cz=20)
    graph = axis_graph(frame, 8, 72, 5 * SPACING, cy=20, cz=20)
    survey = _survey(frame, mask, graph)

    measured = [f for f in survey.frames if f.measured]
    assert measured
    # Flat, and legitimately so: the cut is already square to the vessel, so nothing
    # in the search cone shortens its boundary.
    assert np.median([f.axis_ratio for f in measured]) > 3.0
    assert np.nanmedian([f.obliquity for f in measured]) == pytest.approx(1.0)
    assert "flatter than 1.5:1" in survey.describe()


def test_the_gate_reaches_the_cut_from_the_survey(frame):
    """`transverse_axis_ratio=inf` is the panel's direct test of "wrong axis".

    What the gate *does* is pinned on exact tilts in `test_crosssection`, where the
    normal can be set rather than fitted. All this has to show is that the knob is
    still connected by the time the survey cuts anything.
    """
    seen = {}
    mask = cylinder(SHAPE, 6, 5, 75)
    graph = axis_graph(frame, 8, 72, 6 * SPACING, cy=20, cz=20)

    import hipct_seg_debug.crosssection as cs

    original = cs.stable_transverse_cut

    def spy(*args, **kw):
        seen["ratio"] = kw.get("transverse_axis_ratio")
        return original(*args, **kw)

    cs.stable_transverse_cut = spy
    try:
        sf.survey(graph, frame, mask, stride=8)
        assert seen["ratio"] == pytest.approx(cs.TRANSVERSE_AXIS_RATIO)
        sf.survey(graph, frame, mask, stride=8, transverse_axis_ratio=float("inf"))
        assert seen["ratio"] == float("inf")
    finally:
        cs.stable_transverse_cut = original


# ------------------------------------------------- reproducing the pass exactly


def test_the_survey_defaults_match_the_pass_it_reproduces():
    """`survey` exists to re-cut what `measure_radii` cuts. Its defaults are the contract.

    This broke silently once, and the only symptom was a wrong number. The pass gained
    `grow_radii` and two stability thresholds; `survey` did not, and nothing failed --
    it simply went on describing a different pass. With the panel at max half 128 and
    no growth ceiling a 260 um vessel's window doubled to half=88 voxels, swallowed the
    lumen beside it and reported r=1062 um at 3.7:1, and the conclusion drawn from that
    was that `radius-perimeter` does not work.

    So every shared parameter is compared by name *and* by default value. Adding one to
    either side now has to be a decision rather than an omission. The single permitted
    disagreement is `transverse_axis_ratio`, where `survey` uses `None` to mean "take
    the module default" and resolves it in its own first lines.
    """
    import inspect

    from hipct_seg_debug.crosssection import TRANSVERSE_AXIS_RATIO
    from hipct_seg_debug.edit import radius_perimeter as rp

    pass_params = inspect.signature(rp.measure_radii).parameters
    view_params = inspect.signature(sf.survey).parameters
    shared = set(pass_params) & set(view_params)

    assert {"grow_radii", "stability_variation", "stability_centroid_radii",
            "stability_centroid_mode", "max_half", "tangent_search_degrees",
            "branch_aware", "min_blob_voxels", "perimeter_correction",
            "transverse_axis_ratio"} <= shared

    differ = {k for k in shared if pass_params[k].default != view_params[k].default}
    assert differ == {"transverse_axis_ratio"}, f"defaults drifted: {differ}"
    assert pass_params["transverse_axis_ratio"].default == TRANSVERSE_AXIS_RATIO
    assert view_params["transverse_axis_ratio"].default is None


def test_the_growth_ceiling_reaches_both_cut_paths(frame):
    """`survey` imports its cutters inside the body, so patching the module works."""
    import hipct_seg_debug.crosssection as cs

    mask = cylinder(SHAPE, 6, 5, 75)
    graph = axis_graph(frame, 8, 72, 6 * SPACING, cy=20, cz=20)
    seen = {}

    real_stable, real_cut = cs.stable_transverse_cut, cs.cut

    def spy_stable(*a, **kw):
        seen.update(stable=kw)
        return real_stable(*a, **kw)

    def spy_cut(*a, **kw):
        seen.setdefault("cut", kw)
        return real_cut(*a, **kw)

    cs.stable_transverse_cut, cs.cut = spy_stable, spy_cut
    try:
        sf.survey(graph, frame, mask, stride=8, rival_check=False)
        assert seen["stable"]["grow_radii"] is None
        assert seen["stable"]["max_variation"] == 1.5
        assert seen["stable"]["max_centroid_radii"] == 0.5
        assert seen["stable"]["centroid_mode"] == "drift"

        seen.clear()
        sf.survey(graph, frame, mask, stride=8, grow_radii=4.0, rival_check=False)
        assert seen["stable"]["grow_radii"] == 4.0

        # branch-aware off leaves the plain `cut` as the only cutter, so its kwargs
        # are reachable; `grow_to` is the per-point form of the same ceiling.
        seen.clear()
        sf.survey(graph, frame, mask, stride=8, grow_radii=4.0, branch_aware=False,
                  rival_check=False)
        assert seen["cut"]["grow_to"] == int(4.0 * 6.0) + 2
    finally:
        cs.stable_transverse_cut, cs.cut = real_stable, real_cut


def test_a_runaway_window_is_named_only_when_it_actually_grew(frame):
    """The advice has to fire on a runaway and stay quiet on an operator's decision.

    A window forced wide with `start half` is as large as one that ran away, and the
    ratio cannot tell them apart -- only `grew` can. Telling someone their own
    deliberate setting is a runaway would train them to ignore the line.
    """
    mask = cylinder(SHAPE, 6, 5, 75)
    graph = axis_graph(frame, 8, 72, 6 * SPACING, cy=20, cz=20)

    forced = sf.survey(graph, frame, mask, stride=8, initial_half=100, max_half=100,
                       rival_check=False)
    assert not any(f.grew for f in forced.frames), "the fixture must not have grown"
    assert max(f.half_radii for f in forced.frames) > sf.RUNAWAY_HALF_RADII
    assert "no growth ceiling" not in forced.describe()


# ------------------------------------------- collapsed lumen, or two merged vessels


def _two_touching_tubes(frame, shared_node=False):
    """Two radius-4 tubes whose surfaces meet, as one graph.

    Lifted from `test_radius_perimeter.test_nonadjacent_touching_vessels_use_local_3d
    _ownership`, which is the pass's own fixture for this. `shared_node` welds the two
    segments at one end so they become topologically adjacent -- the geometry, the
    mask, the blob and the radius are untouched, so it isolates the single variable
    the merge split is about.
    """
    from hipct_seg_debug.edit.adapter import Triple
    from hipct_seg_debug.edit.graphmodel import EditableGraph

    mask = np.maximum(
        cylinder(SHAPE, 4, 5, 75, cy=16, cz=20),
        cylinder(SHAPE, 4, 5, 75, cy=24, cz=20),
    )
    nodes, points, segments = {}, {}, []
    pid = 0
    for sid, cy in enumerate((16, 24)):
        ijk = np.c_[np.arange(8, 72), np.full(64, cy), np.full(64, 20)]
        xyz = frame.seg_to_um(ijk)
        ids = []
        nodes[2 * sid] = (*xyz[0], 0)
        nodes[2 * sid + 1] = (*xyz[-1], 0)
        for p in xyz:
            points[pid] = (*p, 4 * SPACING)
            ids.append(pid)
            pid += 1
        node1 = 0 if shared_node else 2 * sid
        segments.append({"id": sid, "node1": node1, "node2": 2 * sid + 1,
                         "point_ids": ids, "strahler": 1})
    return mask, EditableGraph(Triple(nodes, points, segments))


def test_two_touching_vessels_are_reported_as_one_merged_section(frame):
    """The blind spot the merge column exists to cover.

    Two tubes whose surfaces meet are one 8-connected blob, so the perimeter measured
    is of both and the radius comes back far too large. `shape` sees an elongated
    section and `obliq` sees a plane that rotation cannot improve -- which is exactly
    what a *genuinely collapsed* lumen looks like. Neither column can tell them apart.
    A foreign centreline inside the blob can, and it is the same test `measure_radii`
    uses to decide ownership -- except the pass then re-cuts or refuses, while the
    survey reports the merged reading, which is why the count has to be visible.
    """
    mask, graph = _two_touching_tubes(frame)
    survey = sf.survey(graph, frame, mask, [0], stride=16)

    measured = [f for f in survey.frames if f.measured]
    assert measured, "the fixture must measure something, or it tests nothing"
    for f in measured:
        assert f.merged_rivals == 1
        assert f.merged_adjacent == 0
        assert f.merged_foreign == 1
        assert sf._merge_cell(f) == "1/1"
        # The blind spot itself, pinned: both existing columns read this as a single
        # collapsed lumen.
        assert f.axis_ratio > 1.5
        assert f.obliquity == pytest.approx(1.0, abs=0.05)
        assert f.radius_ratio > 1.2, "the merged reading is of both tubes"
    assert "non-adjacent" in survey.describe()


def test_a_rival_that_shares_a_node_is_a_junction_not_contamination(frame):
    """Only the topology moves, and the verdict about the same blob flips.

    At a branch node the lumens really are continuous, so a neighbour's centreline
    inside this section is what a junction *is*. Counting it as contamination would
    flag every junction in the tree.
    """
    mask, graph = _two_touching_tubes(frame, shared_node=True)
    survey = sf.survey(graph, frame, mask, [0], stride=16)

    measured = [f for f in survey.frames if f.measured]
    assert measured
    for f in measured:
        assert f.merged_rivals == 1
        assert f.merged_adjacent == 1
        assert f.merged_foreign == 0
        assert sf._merge_cell(f) == "0/1"
    assert "what a junction looks like" in survey.describe()


def test_a_lone_tube_finds_no_rivals(frame):
    """'Asked and found none' has to read differently from 'not asked'."""
    mask = cylinder(SHAPE, 6, 5, 75)
    graph = axis_graph(frame, 8, 72, 6 * SPACING, cy=20, cz=20)
    survey = _survey(frame, mask, graph)

    assert survey.rivals_tested
    for f in survey.frames:
        if f.merged_rivals >= 0:
            assert f.merged_rivals == 0 and sf._merge_cell(f) == "0/0"


def test_the_merge_check_can_be_turned_off_and_says_so(frame):
    """An absent column reads as 'no merges', which is the one thing it must not."""
    mask, graph = _two_touching_tubes(frame)
    survey = sf.survey(graph, frame, mask, [0], stride=16, rival_check=False)

    assert not survey.rivals_tested
    assert all(f.merged_rivals == -1 for f in survey.frames)
    assert all(sf._merge_cell(f) == "-" for f in survey.frames)
    assert "merge check off" in survey.describe()
