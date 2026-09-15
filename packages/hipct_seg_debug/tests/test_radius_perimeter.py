"""Radius measured from each point's own cross-section perimeter.

The upstream pipeline's assumption -- that a collapsed lumen's *perimeter* survives
fixation even though its shape does not -- is not in question here. What is tested is
that the measurement is actually taken at each point and written back, rather than
being reduced to a global fit, that the estimator is chosen honestly, and that its
digitisation bias is corrected rather than written out as if it were the vessel.
"""

from __future__ import annotations


import numpy as np
import pytest

from hipct_seg_debug.edit import radius_perimeter as rp

from .conftest_geometry import (
    SPACING, _graph_through, axis_graph, cylinder, graph_from, make_frame, slit,
    wobbly_graph,
)

pytest.importorskip("cv2")

SHAPE = (40, 40, 80)


@pytest.fixture
def frame():
    return make_frame(SHAPE)


def _measure(frame, mask, radius_seed_um, *, cy=20, cz=20, **kw):
    graph = axis_graph(frame, 8, 72, radius_seed_um, cy=cy, cz=cz)
    return graph, rp.measure_radii(graph, frame, mask, **kw)


# ------------------------------------------------------------------ the values


def test_a_known_cylinder_measures_its_own_radius(frame):
    r_vox = 6
    mask = cylinder(SHAPE, r_vox, 5, 75)
    graph, result = _measure(frame, mask, r_vox * SPACING)

    radii = result.radii[0]
    assert np.isfinite(radii).all()
    # The correction is on, so this should land near the truth rather than under it;
    # the tolerance stays loose because a single centred fixture is not the place to
    # pin the estimator. See the digitisation-bias tests below for that.
    assert np.median(radii) == pytest.approx(r_vox * SPACING, rel=0.15)


def test_every_point_is_measured_not_fitted(frame):
    """A tapering tube must produce a tapering radius, which a global fit would not."""
    nz, ny, nx = SHAPE
    zz, yy, xx = np.ogrid[:nz, :ny, :nx]
    # radius falls linearly from 8 to 3 voxels along x
    r_of_x = np.linspace(8.0, 3.0, nx)[None, None, :]
    mask = (((zz - 20) ** 2 + (yy - 20) ** 2) <= r_of_x ** 2).astype(np.uint8)

    graph, result = _measure(frame, mask, 8 * SPACING)
    radii = result.radii[0]

    assert radii[0] > radii[-1], "the measured profile tapers"
    assert radii[0] / radii[-1] == pytest.approx(8.0 / 3.0, rel=0.35)


def test_a_collapsed_slit_reports_the_perimeter_not_the_half_width(frame):
    """The design assumption: a slit re-inflates to the circle of equal perimeter."""
    half_y = 8
    mask = slit(SHAPE, half_y=half_y, half_z=1, x0=5, x1=75, cy=20, cz=20)
    graph, result = _measure(frame, mask, 40.0)

    radii = result.radii[0]
    # Perimeter of a (2*8+1) x 3 voxel rectangle ~ 2*(17+3) = 40 voxels -> r ~ 6.4 vox.
    assert np.median(radii) > half_y * SPACING * 0.6
    assert np.median(radii) < (2 * half_y) * SPACING
    assert not (result.reject_reason[0] == rp.RUNAWAY).any()


# --------------------------------------------------------------- the hybrid gate


def test_the_gate_picks_area_for_a_thin_section_and_perimeter_for_a_thick_one(frame):
    """The crossover still works when asked for -- it is simply not asked for.

    `GATE_VOXELS` defaults to 0 so every point is measured by perimeter; see the
    constant's own note for the measurement that decided it. The mechanism is kept
    and tested because the staircase inflation it corrects is real, and a
    well-resolved dataset with round sections may want it back.
    """
    thick = cylinder(SHAPE, 8, 5, 75)
    _g, thick_result = _measure(frame, thick, 8 * SPACING, gate_voxels=3.0,
                                perimeter_correction=False)
    assert (thick_result.source[0] == rp.PERIMETER).all()

    thin = cylinder(SHAPE, 2, 5, 75)
    _g2, thin_result = _measure(frame, thin, 2 * SPACING, min_blob_voxels=4,
                                gate_voxels=3.0)
    measured = thin_result.source[0] != rp.FILLED
    assert measured.any(), "the thin tube is still measurable"
    assert (thin_result.source[0][measured] == rp.AREA).all()


def test_every_point_is_measured_by_perimeter_by_default(frame):
    """Area under-reads a collapsed section, which is the case this pass is for.

    On LADAF-2024-28 the gate sent 19.1% of points down the area branch, and those
    read a median 0.73x -- worst 0.36x -- the perimeter points in their own segment:
    a stretch half the width of the vessel either side of it, widening again where
    the estimator switched back. The staircase error it avoided is smaller than that.
    """
    assert rp.GATE_VOXELS == 0.0
    thin = cylinder(SHAPE, 2, 5, 75)
    _g, result = _measure(frame, thin, 2 * SPACING, min_blob_voxels=4,
                          perimeter_correction=False)
    measured = result.source[0] != rp.FILLED
    assert measured.any()
    assert (result.source[0][measured] == rp.PERIMETER).all()


def test_the_raw_perimeter_misses_a_thin_section_by_more_than_area_does(frame):
    """The measurement that justifies the gate existing at all.

    Named for over-statement until 2026-08-30, when the direction was measured:
    the raw estimator misses a thin section by reading it *small*, not large. The
    assertion never tested the direction, only the magnitude, so it held either
    way -- which is exactly how the wrong sign survived in the surrounding prose.
    """
    r_vox = 2
    mask = cylinder(SHAPE, r_vox, 5, 75)
    truth = r_vox * SPACING

    _g, area = _measure(frame, mask, truth, min_blob_voxels=4, gate_voxels=99.0)
    _g2, perim = _measure(frame, mask, truth, min_blob_voxels=4, gate_voxels=0.0,
                          perimeter_correction=False)

    area_err = abs(np.median(area.radii[0]) - truth)
    perim_err = abs(np.median(perim.radii[0]) - truth)
    assert perim_err > area_err
    assert np.median(perim.radii[0]) < truth, "raw perimeter under-reads, not over"


# ------------------------------------------------------- the digitisation bias


#: Sub-voxel positions of the tube axis. A vessel axis does not lie on voxel
#: centres, and at these sizes where it lies changes the digitisation entirely, so
#: any statement about the estimator has to be made over a sweep of them rather
#: than at one. See :func:`test_a_centred_tube_of_integer_radius_is_not_typical`.
AXIS_OFFSETS = (0.0, 0.13, 0.27, 0.41, 0.5, 0.68)


def _tube_at(frame, r_vox, off, **kw):
    """`measure_radii` on a tube whose axis sits `off` voxels off the lattice.

    The centreline is placed *on* the axis, so what this varies is the digitisation
    of the section and not the centring of the skeleton within it.
    """
    cy = cz = 20 + off
    mask = cylinder(SHAPE, r_vox, 5, 75, cy=cy, cz=cz)
    xs = np.arange(8, 72, dtype=float)
    ijk = np.stack([xs, np.full_like(xs, cy), np.full_like(xs, cz)], axis=1)
    graph = _graph_through(frame, ijk, r_vox * SPACING)
    return rp.measure_radii(graph, frame, mask, **kw)


def _swept_median(frame, r_vox, **kw):
    return float(np.median([
        np.median(_tube_at(frame, r_vox, off, **kw).radii[0]) for off in AXIS_OFFSETS
    ]))


def test_the_correction_recovers_a_thin_cylinder_the_raw_reading_misses(frame):
    """Ground truth: a cylinder of known radius, measured through the whole pass.

    Two voxels is the size the `MIN_BLOB_VOXELS` floor sits at, so it is the size
    the correction has to earn its place at. `research_scripts/subvoxel_bias.py` in the
    coronary_sdf checkout sweeps this across radii, sub-voxel offsets and
    eccentricities; this pins the one case that matters most.
    """
    r_vox = 2
    truth = r_vox * SPACING

    raw = _swept_median(frame, r_vox, min_blob_voxels=4, perimeter_correction=False)
    fixed = _swept_median(frame, r_vox, min_blob_voxels=4)

    assert raw == pytest.approx(0.82 * truth, rel=0.05)
    assert fixed == pytest.approx(truth, rel=0.05)

    result = _tube_at(frame, r_vox, 0.27, min_blob_voxels=4)
    measured = result.source[0] != rp.FILLED
    assert (result.source[0][measured] == rp.PERIMETER_CORRECTED).all()


def test_a_centred_tube_of_integer_radius_is_not_typical(frame):
    """The fixture the rest of this file uses is the estimator's best case.

    A disc of integer radius centred on a voxel centre has its boundary running
    straight along the axes at the four cardinal points, which is the one
    configuration where the digitised perimeter comes out *long*. At r = 2 it reads
    0.90x true where the swept median is 0.82x -- above the 95th percentile of
    offsets, not near the middle of them. Correcting it therefore overshoots.

    This is pinned rather than avoided because it is the trap: measuring the
    estimator on this one configuration is what produced the belief that the
    perimeter over-states a thin section, and re-tuning the constants until this
    case lands on 1.00 would put every realistic offset 10% high.
    """
    r_vox = 2
    truth = r_vox * SPACING
    aligned = float(np.median(_tube_at(frame, r_vox, 0.0, min_blob_voxels=4,
                                       perimeter_correction=False).radii[0]))
    assert aligned == pytest.approx(0.90 * truth, rel=0.02)
    assert aligned > _swept_median(frame, r_vox, min_blob_voxels=4,
                                   perimeter_correction=False)


def test_the_correction_leaves_a_section_that_traced_no_contour_alone(frame):
    """A one- or two-voxel blob closes a contour of zero length. Correcting that
    would invent half a voxel of radius out of an estimator failure."""
    assert rp.correct_perimeter_radius(0.0, SPACING) == 0.0
    assert rp.correct_perimeter_radius(2.0 * SPACING, SPACING) > 2.0 * SPACING


def test_the_correction_changes_sign_where_the_model_says_it_does(frame):
    """Below 9.6 voxels the estimator under-reads and the correction lifts the
    radius; above it the estimator over-reads and the correction lowers it. A
    correction that only ever pushed one way would be a fudge factor."""
    sp = SPACING
    small = 3.0 * sp
    large = 20.0 * sp
    assert rp.correct_perimeter_radius(small, sp) > small
    assert rp.correct_perimeter_radius(large, sp) < large


# ---------------------------------------------------------------- gaps and I/O


def test_unmeasurable_points_are_filled_and_flagged(frame):
    """A centreline leaving the mask must not silently keep its old radius."""
    mask = cylinder(SHAPE, 6, 5, 40)  # the tube stops half way along the centreline
    graph, result = _measure(frame, mask, 60.0)

    source = result.source[0]
    assert (source == rp.FILLED).any(), "the unmeasured tail is marked, not invented"
    assert np.isfinite(result.radii[0]).all(), "and it still has a usable radius"
    assert (result.radii[0] > 0).all()


def test_a_section_that_never_closes_is_interpolated_not_measured(frame):
    """The contour of a truncated blob is not a cross-section, at any window size.

    On LADAF-2024-28 these were warned about and used anyway: 612 points came back at a
    median radius of 3.5 mm against 0.76 mm stored, and being weighted by r squared they
    carried 28% of the whole network's volume between them.

    Forced here by capping the window far below the tube, so the section runs off every
    edge exactly as an oblique cut through a real vessel does.
    """
    mask = cylinder(SHAPE, 12, 5, 75)
    graph = axis_graph(frame, 6, 74, 120.0, cy=SHAPE[1] // 2, cz=SHAPE[0] // 2)
    result = rp.measure_radii(graph, frame, mask, max_half=4)

    assert result.n_truncated > 0, "the window really was too small to close the section"
    assert (result.source[0] == rp.FILLED).any(), "and those points were not measured"
    # Whatever survived, nothing reports the outline of a truncated window as a radius.
    assert result.radii[0].max() < 4.0 * 120.0
    assert (result.reject_reason[0] == rp.TRUNCATED).any()


def test_branched_endpoints_are_interpolated_not_measured_from_the_carina(frame):
    centre = frame.seg_to_um(np.array([[40, 20, 20]]))[0]
    ends = frame.seg_to_um(np.array([[10, 20, 20], [70, 10, 20], [70, 30, 20]]))
    graph = graph_from([centre, *ends], [(0, 1, 16, 30.0), (0, 2, 16, 30.0),
                                                (0, 3, 16, 30.0)])
    mask = np.ones(SHAPE, dtype=np.uint8)
    result = rp.measure_radii(graph, frame, mask, max_half=4)

    for sid in graph.segment_ids():
        # The coordinate at the branch is duplicated per edge and each copy retains
        # its own edge-local orientation, but none is measured through the carina.
        assert result.reject_reason[sid][0] == rp.JUNCTION
        assert result.source[sid][0] == rp.FILLED


def test_adaptive_junction_run_waits_for_two_exclusive_stable_sections(frame):
    centre = frame.seg_to_um(np.array([[40, 20, 20]]))[0]
    ends = frame.seg_to_um(np.array([[10, 20, 20], [70, 10, 20], [70, 30, 20]]))
    graph = graph_from([centre, *ends], [(0, 1, 12, 40.0), (0, 2, 12, 40.0),
                                                (0, 3, 12, 40.0)])
    stable = np.ones(12, dtype=bool)
    overlap = np.zeros(12, dtype=bool)
    overlap[:5] = True
    arc = np.arange(12, dtype=float) * 20.0

    # `max_fraction=None` so the cap cannot pre-empt the mechanism under test: with
    # the default it would stop this walk at the same index for the wrong reason and
    # the assertion would pass without exercising the exclusivity rule at all.
    mask, lengths = rp._adaptive_junction_mask(
        graph, 0, arc, stable, overlap, max_fraction=None)

    assert np.flatnonzero(mask).tolist() == list(range(5))
    assert lengths == [80.0]


# ------------------------------------------------- the junction mask's own bound


def _both_ends_branched(frame, n_points, radius_um=40.0):
    """A segment whose two end nodes are both degree 3, which is the shape that
    lets two junction walks meet in the middle and consume everything between."""
    a = frame.seg_to_um(np.array([[20, 20, 20]]))[0]
    b = frame.seg_to_um(np.array([[60, 20, 20]]))[0]
    spurs = frame.seg_to_um(np.array([[10, 10, 20], [10, 30, 20],
                                      [70, 10, 20], [70, 30, 20]]))
    graph = graph_from(
        [a, b, *spurs],
        [(0, 1, n_points, radius_um),  # segment 0: the one under test
         (0, 2, 6, radius_um), (0, 3, 6, radius_um),
         (1, 4, 6, radius_um), (1, 5, 6, radius_um)],
    )
    assert graph.degree(0) == 3 and graph.degree(1) == 3
    return graph


def test_an_unbounded_junction_mask_eats_the_whole_segment(frame):
    """The failure this bound exists for, pinned so the fix cannot be undone quietly.

    Every section stable, every section adjacent to a rival -- which is the ordinary
    state of a short segment between two junctions, and is what segments 265 and 268
    of LADAF-2024-28 measured at 19/19 and 9/9. Nothing is ever exclusive, so the
    walk never meets its two-in-a-row stop and the loop's `else` commits everything.
    """
    n = 20
    graph = _both_ends_branched(frame, n)
    arc = np.arange(n, dtype=float) * 20.0
    stable = np.ones(n, dtype=bool)
    overlap = np.ones(n, dtype=bool)

    mask, _ = rp._adaptive_junction_mask(
        graph, 0, arc, stable, overlap, max_fraction=None)

    assert mask.all(), "unbounded, both walks run the full length"


def test_the_bound_keeps_a_measurable_middle_whatever_the_overlap_says(frame):
    n = 20
    graph = _both_ends_branched(frame, n)
    arc = np.arange(n, dtype=float) * 20.0
    stable = np.ones(n, dtype=bool)
    overlap = np.ones(n, dtype=bool)

    mask, _ = rp._adaptive_junction_mask(graph, 0, arc, stable, overlap)

    kept = np.flatnonzero(~mask)
    assert len(kept) > 0, "the bound must leave something to measure"
    # Both ends may take `JUNCTION_MASK_MAX_FRACTION` each, so the survivors are the
    # middle of the segment rather than one arbitrary end of it.
    assert kept.min() > 0 and kept.max() < n - 1
    assert len(kept) / n >= 1 - 2 * rp.JUNCTION_MASK_MAX_FRACTION - 1e-9


def test_the_bound_does_not_extend_a_run_that_would_have_stopped_sooner(frame):
    """The bound is a ceiling, not a target: where the exclusivity rule already
    stops the walk early, the mask must be exactly what it was before."""
    n = 20
    graph = _both_ends_branched(frame, n)
    arc = np.arange(n, dtype=float) * 20.0
    stable = np.ones(n, dtype=bool)
    overlap = np.zeros(n, dtype=bool)
    overlap[:2] = True
    overlap[-2:] = True

    bounded, _ = rp._adaptive_junction_mask(graph, 0, arc, stable, overlap)
    unbounded, _ = rp._adaptive_junction_mask(
        graph, 0, arc, stable, overlap, max_fraction=None)

    assert np.array_equal(bounded, unbounded)
    assert np.flatnonzero(bounded).tolist() == [0, 1, 18, 19]


def test_a_node_ceiling_bounds_the_run_by_the_junction_and_not_the_segment(frame):
    """The fraction measures the wrong thing, and on a short segment it is all there is.

    A junction's influence is a property of the junction: it reaches a couple of the
    node's own radii into each branch, wherever that branch happens to end. A fraction
    of the segment measures how long the segment is instead, so the same node takes
    40% of a long branch and 40% of a short one. Since the exclusivity test is
    unsatisfiable here -- every section stable, every section adjacent -- the walk
    always runs to whichever bound binds first, and that must be the junction's.
    """
    n = 20
    graph = _both_ends_branched(frame, n)
    arc = np.arange(n, dtype=float) * 20.0  # 380 um long; 40% per end is 152 um
    stable = np.ones(n, dtype=bool)
    overlap = np.ones(n, dtype=bool)
    nid1, nid2 = graph.segment(0)["node1"], graph.segment(0)["node2"]

    fraction_only, _ = rp._adaptive_junction_mask(graph, 0, arc, stable, overlap)
    tight, _ = rp._adaptive_junction_mask(
        graph, 0, arc, stable, overlap, node_limits={nid1: 60.0, nid2: 60.0})

    assert tight.sum() < fraction_only.sum(), "the tighter of the two bounds must win"
    # 60 um reaches points 0-3 from each end, and nothing further.
    assert np.flatnonzero(tight).tolist() == [0, 1, 2, 3, 16, 17, 18, 19]


def test_the_node_ceiling_is_per_node_and_never_widens_the_fraction(frame):
    n = 20
    graph = _both_ends_branched(frame, n)
    arc = np.arange(n, dtype=float) * 20.0
    stable = np.ones(n, dtype=bool)
    overlap = np.ones(n, dtype=bool)
    nid1, nid2 = graph.segment(0)["node1"], graph.segment(0)["node2"]

    # One end tight, the other far wider than the fraction allows: the loose end must
    # still stop at the fraction, so a ceiling can only ever remove points.
    mixed, _ = rp._adaptive_junction_mask(
        graph, 0, arc, stable, overlap, node_limits={nid1: 60.0, nid2: 1e9})
    fraction_only, _ = rp._adaptive_junction_mask(graph, 0, arc, stable, overlap)

    # The tight end stops at 60 um (points 0-3); the loose end still stops where
    # the fraction says (152 um of 380, points 12-19), not at its own ceiling.
    assert np.flatnonzero(mixed).tolist() == [0, 1, 2, 3, *range(12, 20)]
    assert not (mixed & ~fraction_only).any()


def test_the_node_ceiling_is_off_until_something_shows_it_helps(frame):
    """Measured on LADAF-2021-17 and it did not help: the points it frees from the
    mask come back `truncated` or `unstable` rather than measured, because the mask
    was covering points that were already unmeasurable. The bound is kept and
    documented; the default stays off until there is evidence for it."""
    import inspect

    assert inspect.signature(rp.measure_radii).parameters["junction_mask_radii"].default is None


def test_noisy_centreline_is_not_moved_and_still_measures_transversely(frame):
    mask = cylinder(SHAPE, 6, 5, 75)
    graph = wobbly_graph(frame, 8, 72, 6 * SPACING, cy=20, cz=20, wobble=1)
    before = graph.coords(0).copy()

    result = rp.measure_radii(graph, frame, mask)

    assert np.array_equal(graph.coords(0), before)
    assert np.median(result.radii[0]) == pytest.approx(6 * SPACING, rel=0.2)
    assert result.radii[0].max() < 2.0 * np.median(result.radii[0])


def test_two_pass_trend_rejects_only_a_grown_oversized_correction():
    n = 40
    arc = np.arange(n, dtype=float) * 50.0
    old = np.full(n, 100.0)
    measured = np.full(n, 180.0)  # sustained correction: legitimate collapsed shape
    measured[20] = 1000.0
    grew = np.zeros(n, dtype=bool)
    grew[20] = True

    bad = rp._robust_high_correction_mask(arc, measured, old, grew)
    assert np.flatnonzero(bad).tolist() == [20]


def test_local_factor_gate_rejects_an_isolated_large_radius_and_interpolates(frame):
    mask = cylinder(SHAPE, 4, 5, 75)
    # A short, oversized component around x=40 closes cleanly, so border truncation
    # alone cannot reject it. Its surrounding branch remains a radius-4 cylinder.
    zz, yy, xx = np.ogrid[:SHAPE[0], :SHAPE[1], :SHAPE[2]]
    bulge = ((zz - 20) ** 2 + (yy - 20) ** 2 <= 12 ** 2) & (np.abs(xx - 40) <= 1)
    mask[bulge] = 1
    # A generous input window lets the bulge close without triggering the separate
    # growth/runaway rule, isolating the local-factor gate in this test.
    graph, result = _measure(frame, mask, 100.0, max_radius_factor=2.0)

    # The three-plane stability gate normally catches this abrupt one-slice bulge
    # before it reaches the second-pass local-factor gate. Either rejection path is
    # valid; importantly, neither outline becomes an accepted radius.
    rejected = np.isin(result.reject_reason[0], (rp.UNSTABLE, rp.CEILING))
    assert rejected.any()
    assert result.source[0][rejected].tolist() == [rp.FILLED] * int(rejected.sum())
    surrounding = np.median(result.radii[0][~rejected])
    assert result.radii[0][rejected].max() < 1.25 * surrounding


def test_local_factor_has_no_global_micrometre_cap():
    n = 40
    arc = np.arange(n, dtype=float) * 50.0
    measured = np.full(n, 2500.0)
    old = np.full(n, 2500.0)
    assert not rp._robust_local_high_mask(arc, measured, old, factor=2.0).any()


def test_local_factor_uses_input_neighbourhood_when_too_few_sections_close():
    arc = np.arange(6, dtype=float) * 50.0
    measured = np.array([np.nan, np.nan, np.nan, np.nan, 2300.0, 1400.0])
    old = np.full(6, 150.0)
    bad = rp._robust_local_high_mask(
        arc, measured, old, factor=2.0, input_calibration=1.35
    )
    assert np.flatnonzero(bad).tolist() == [4, 5]


def test_bifurcation_taper_keeps_parent_and_grows_daughters_from_the_carina(frame):
    centre = frame.seg_to_um(np.array([[40, 20, 20]]))[0]
    ends = frame.seg_to_um(np.array([[10, 20, 20], [70, 10, 20], [70, 30, 20]]))
    graph = graph_from([centre, *ends], [(0, 1, 12, 200.0), (0, 2, 12, 80.0),
                                                (0, 3, 12, 70.0)])
    graph.segment(0)["strahler"] = 2
    graph.segment(1)["strahler"] = 1
    graph.segment(2)["strahler"] = 1
    measured, source, reject, modes, arcs = {}, {}, {}, {}, {}
    for sid, radius in enumerate((200.0, 80.0, 70.0)):
        measured[sid] = np.full(12, radius)
        measured[sid][:3] = np.nan
        source[sid] = np.full(12, rp.PERIMETER, dtype=np.int8)
        reject[sid] = np.zeros(12, dtype=np.int8)
        reject[sid][:3] = rp.JUNCTION
        modes[sid] = np.full(12, rp.INTERPOLATED, dtype=np.int8)
        arcs[sid] = np.arange(12, dtype=float) * 20.0
    # `daughter_carina` is explicit: it defaults off, so the carina model this test
    # exists to exercise has to be asked for.
    fallback = rp._apply_bifurcation_tapers(
        graph, measured, source, reject, modes, arcs,
        spacing_um=SPACING, carina_tip_factor=0.1, daughter_carina=True,
    )
    assert not fallback
    assert np.allclose(measured[0][:4], 200.0)
    assert modes[0][0] == rp.BIF_PARENT
    for sid, anchor in ((1, 80.0), (2, 70.0)):
        assert measured[sid][0] == pytest.approx(0.1 * anchor)
        assert np.all(np.diff(measured[sid][:4]) > 0)
        assert modes[sid][0] == rp.BIF_DAUGHTER


def _junction_fixture(frame):
    """A parent of 200 um shedding two ostial daughters, junction runs unmeasured."""
    centre = frame.seg_to_um(np.array([[40, 20, 20]]))[0]
    ends = frame.seg_to_um(np.array([[10, 20, 20], [70, 10, 20], [70, 30, 20]]))
    graph = graph_from([centre, *ends], [(0, 1, 12, 200.0), (0, 2, 12, 80.0),
                                         (0, 3, 12, 70.0)])
    for sid, order in ((0, 2), (1, 1), (2, 1)):
        graph.segment(sid)["strahler"] = order
    measured, source, reject, modes, arcs = {}, {}, {}, {}, {}
    for sid, radius in enumerate((200.0, 80.0, 70.0)):
        measured[sid] = np.full(12, radius)
        measured[sid][:3] = np.nan
        source[sid] = np.full(12, rp.PERIMETER, dtype=np.int8)
        reject[sid] = np.zeros(12, dtype=np.int8)
        reject[sid][:3] = rp.JUNCTION
        modes[sid] = np.full(12, rp.INTERPOLATED, dtype=np.int8)
        arcs[sid] = np.arange(12, dtype=float) * 20.0
    return graph, measured, source, reject, modes, arcs


def test_the_parent_is_carried_through_without_authoring_a_daughter_carina(frame):
    """The two halves of the junction model are independent, and only one is on.

    Carrying the parent's own trend across a span where its sections were refused
    extrapolates that vessel's own measurements. Narrowing a daughter to a carina
    tip asserts a shape nothing observed -- and asserts it with the wrong sign, since
    it makes the daughter narrowest at the node where an ostium is widest.
    """
    graph, measured, source, reject, modes, arcs = _junction_fixture(frame)

    rp._apply_bifurcation_tapers(
        graph, measured, source, reject, modes, arcs,
        spacing_um=SPACING, carina_tip_factor=0.1, daughter_carina=False,
    )

    assert np.allclose(measured[0][:4], 200.0), "the parent is carried through"
    assert modes[0][0] == rp.BIF_PARENT
    for sid in (1, 2):
        assert np.isnan(measured[sid][:3]).all(), "the daughter run is left to _fill_gaps"
        assert modes[sid][0] == rp.INTERPOLATED
        assert reject[sid][0] == rp.JUNCTION


def test_junction_flare_modes_keep_the_measurement_they_claim_to(frame):
    """`none` discards every junction measurement, `all` keeps them, `parent` keeps
    the parent's and no one else's.

    A tube with a side branch: the section approaching the node is genuinely wider
    because the two lumens are continuous there. Which branch is allowed to report
    that width is the whole question, since a shared region kept on every branch is
    contributed to the union once per branch.
    """
    mask = cylinder(SHAPE, 6, 5, 75)
    counts = {}
    for mode in ("none", "parent", "all"):
        _g, res = _measure(frame, mask, 6 * SPACING, junction_flare=mode)
        counts[mode] = int((res.reject_reason[0] == rp.JUNCTION).sum())
    assert counts["none"] >= counts["parent"] >= counts["all"], counts
    assert rp.JUNCTION_FLARE_MODES == ("none", "parent", "all")

    with pytest.raises(ValueError, match="junction_flare"):
        _measure(frame, mask, 6 * SPACING, junction_flare="both")


def test_the_daughter_carina_is_the_only_thing_the_flag_turns_off(frame):
    """Parent handling must be bit-identical with the carina on and off, so the two
    can be reasoned about separately."""
    on = _junction_fixture(frame)
    off = _junction_fixture(frame)
    for args, carina in ((on, True), (off, False)):
        graph, measured, source, reject, modes, arcs = args
        rp._apply_bifurcation_tapers(
            graph, measured, source, reject, modes, arcs,
            spacing_um=SPACING, carina_tip_factor=0.1, daughter_carina=carina,
        )
    assert np.array_equal(on[1][0], off[1][0])
    assert np.array_equal(on[4][0], off[4][0])


def test_explicit_root_edge_overrides_automatic_parent_inference(frame):
    centre = frame.seg_to_um(np.array([[40, 20, 20]]))[0]
    ends = frame.seg_to_um(np.array([[10, 20, 20], [70, 10, 20], [70, 30, 20]]))
    graph = graph_from([centre, *ends], [(0, 1, 6, 200.0), (0, 2, 6, 80.0),
                                                (0, 3, 6, 70.0)])
    graph.segment(0)["strahler"] = 3
    graph.segment(1)["strahler"] = 1
    graph.segment(2)["strahler"] = 1

    assert rp._junction_parents(graph)[0] == 0
    assert rp._junction_parents(graph, root_edges=[1])[0] == 1


def test_junction_without_a_trusted_anchor_retains_input_with_provenance(frame):
    centre = frame.seg_to_um(np.array([[40, 20, 20]]))[0]
    ends = frame.seg_to_um(np.array([[10, 20, 20], [70, 10, 20], [70, 30, 20]]))
    graph = graph_from([centre, *ends], [(0, 1, 5, 200.0), (0, 2, 5, 80.0),
                                                (0, 3, 5, 70.0)])
    graph.segment(0)["strahler"] = 2
    measured = {sid: np.full(5, np.nan) for sid in graph.segment_ids()}
    source = {sid: np.full(5, rp.FILLED, dtype=np.int8) for sid in graph.segment_ids()}
    reject = {sid: np.full(5, rp.JUNCTION, dtype=np.int8) for sid in graph.segment_ids()}
    modes = {sid: np.full(5, rp.INTERPOLATED, dtype=np.int8) for sid in graph.segment_ids()}
    arcs = {sid: np.arange(5, dtype=float) * 20.0 for sid in graph.segment_ids()}

    fallback = rp._apply_bifurcation_tapers(
        graph, measured, source, reject, modes, arcs, spacing_um=SPACING,
    )

    assert fallback == set(graph.segment_ids())
    for sid in graph.segment_ids():
        assert np.array_equal(measured[sid], graph.radii(sid))
        assert (modes[sid] == rp.INPUT_FALLBACK).all()


def test_nonadjacent_touching_vessels_use_local_3d_ownership(frame):
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
        segments.append(
            {"id": sid, "node1": 2 * sid, "node2": 2 * sid + 1,
             "point_ids": ids, "strahler": 1}
        )
    graph = EditableGraph(Triple(nodes, points, segments))
    result = rp.measure_radii(graph, frame, mask)

    for sid in graph.segment_ids():
        assert (result.resolution_mode[sid] == rp.OWNED_PLANE).all()
        assert (result.reject_reason[sid] == rp.ACCEPTED).all()
        assert np.median(result.radii[sid]) == pytest.approx(4 * SPACING, rel=0.3)


def test_apply_radii_writes_thickness_and_provenance(frame, tmp_path):
    mask = cylinder(SHAPE, 6, 5, 75)
    graph, result = _measure(frame, mask, 60.0)
    before = graph.radii(0).copy()

    rp.apply_radii(graph, result)

    assert not np.allclose(graph.radii(0), before), "the seed radius was replaced"
    attrs = graph.triple.point_attrs["radius_source"]
    assert set(attrs) == set(graph.segment(0)["point_ids"]), "keyed by point id"
    rejects = graph.triple.point_attrs["radius_reject_reason"]
    assert set(rejects) == set(graph.segment(0)["point_ids"])

    # ...and it survives the trip through the .am writer and reader.
    from hipct_seg_debug.amira import read_spatial_graph
    from hipct_seg_debug.edit.amira_write import write_spatial_graph

    out = tmp_path / "radius.am"
    write_spatial_graph(graph.to_spatial_graph(), out)
    back = read_spatial_graph(out)
    assert "radius_source" in back.point_attrs
    assert "radius_reject_reason" in back.point_attrs
    assert "radius_resolution_mode" in back.point_attrs
    assert len(back.point_attrs["radius_source"]) == back.n_point
    assert len(back.point_attrs["radius_reject_reason"]) == back.n_point
    assert len(back.point_attrs["radius_resolution_mode"]) == back.n_point
    assert np.allclose(back.thickness, graph.radii(0))


def test_mean_radius_is_rederived_on_every_edge(frame):
    """Unlike Stage 5 of the port, which leaves untouched edges on the old scale."""
    mask = cylinder(SHAPE, 6, 5, 75)
    graph, result = _measure(frame, mask, 60.0)
    rp.apply_radii(graph, result)

    mean = graph.segment(0)["MeanRadius"]
    assert mean == pytest.approx(float(np.mean(graph.radii(0))), rel=1e-9)


# ------------------------------------------------- iterated measurement scale

def test_second_pass_reports_movement_and_converges(frame):
    """A second pass re-derives the measurement geometry from measured radii.

    Every geometric scale -- tangent-fit window, cut half-width, stability slab
    offsets -- is taken from the input radius, which is the quantity being
    replaced. `n_passes=2` feeds pass 1's accepted radii back so those scales
    come from a measurement instead. On an isolated cylinder the answer is
    already right, so the second pass must barely move it; `pass_movement`
    records by how much.
    """
    r_vox = 6
    mask = cylinder(SHAPE, r_vox, 5, 75)
    # Seed deliberately wrong (half the truth) so pass 1's geometry is badly
    # scaled -- the situation the feedback exists for.
    _graph, one = _measure(frame, mask, r_vox * SPACING * 0.5, n_passes=1)
    _graph, two = _measure(frame, mask, r_vox * SPACING * 0.5, n_passes=2)

    truth = r_vox * SPACING
    err_one = abs(np.median(one.radii[0]) - truth) / truth
    err_two = abs(np.median(two.radii[0]) - truth) / truth
    # The extra pass must not make an isolated, well-resolved vessel worse.
    assert err_two <= err_one + 0.02

    assert one.pass_movement == []           # nothing to compare against
    assert len(two.pass_movement) == 1       # one inter-pass comparison
    assert np.isfinite(two.pass_movement[0])


def test_single_pass_is_the_default(frame):
    """Default stays one pass: feeding a larger radius back grows the cut window,
    which defeats ownership resolution where non-adjacent vessels touch."""
    import inspect

    sig = inspect.signature(rp.measure_radii)
    assert sig.parameters["n_passes"].default == 1


# ------------------------------------------------------- junction authoring

def test_bifurcation_tapers_can_be_switched_off(frame):
    """Junction profiles are a model of carina geometry, not a measurement.

    With tapers off, junction runs stay rejected and are interpolated from the
    surrounding trusted radii instead, so every point still gets a radius but
    none of them claim a carina that was never observed.
    """
    r_vox = 6
    mask = cylinder(SHAPE, r_vox, 5, 75)
    graph = axis_graph(frame, 8, 72, r_vox * SPACING, cy=20, cz=20)
    on = rp.measure_radii(graph, frame, mask, bifurcation_tapers=True)
    off = rp.measure_radii(graph, frame, mask, bifurcation_tapers=False)

    authored = {rp.BIF_PARENT, rp.BIF_DAUGHTER, rp.BIF_CONTINUATION}
    n_off = sum(int(np.isin(m, list(authored)).sum()) for m in off.resolution_mode.values())
    assert n_off == 0, "tapers were disabled but junction profiles were still authored"

    # Every point still receives a finite, positive radius either way.
    for res in (on, off):
        for arr in res.radii.values():
            assert np.all(np.isfinite(arr)) and np.all(arr > 0)


def test_a_daughter_matching_its_parent_is_not_tapered_to_the_carina_tip():
    """A junction daughter of comparable calibre is the vessel continuing.

    Selecting the parent by topological depth alone makes the continuation of a
    main vessel a `daughter`, and tapering it to `carina_tip_factor` pinches a
    lumen that never narrowed. The calibre test is what separates the two.
    """
    assert 0 < rp.CONTINUATION_RATIO <= 1
    import inspect
    sig = inspect.signature(rp.measure_radii)
    assert sig.parameters["continuation_ratio"].default == rp.CONTINUATION_RATIO
    # Off by default: the carina model is unvalidated against the segmentation,
    # so junction runs are interpolated unless the taper is asked for explicitly.
    assert sig.parameters["bifurcation_tapers"].default is False


# --------------------------------------------- segments that measured nothing


def test_drop_anchors_a_leaf_on_a_neighbour_processed_later(frame):
    """The fallback policies read the *neighbours'* radii, so they run last.

    Resolved inside the measuring loop, a segment sees only the neighbours that
    happened to come earlier in the ordering. On a leaf whose one junction sits
    later -- which is most of them, the junction being interior -- that is no
    neighbours at all, and the span silently kept the uncorrected input radius
    the policy was asked to replace.
    """
    centre = frame.seg_to_um(np.array([[40, 20, 20]]))[0]
    ends = frame.seg_to_um(np.array([[10, 20, 20], [70, 20, 20]]))
    # Segment 0 is the leaf and is measured first; segment 1 carries the anchor.
    graph = graph_from([centre, *ends], [(0, 1, 8, 900.0), (0, 2, 8, 200.0)])
    measured = {0: np.full(8, np.nan), 1: np.full(8, 210.0)}

    anchors = rp._junction_anchor_radii(graph, 0, measured)
    assert anchors == [210.0, 210.0], "a single junction anchor is used at both ends"

    # ...and a neighbour that measured nothing itself must not become the anchor,
    # or the spike moves one segment along instead of going away.
    assert rp._junction_anchor_radii(graph, 0, measured, {1}) == []


def test_a_segment_with_no_trustworthy_section_is_not_left_at_its_input(frame):
    """`drop` must replace the calibre of a segment that measured nothing.

    The mask holds one cylinder; the second segment runs outside it, so nothing
    on it can be measured. Left alone it keeps its seeded 900 um -- 4.5x the
    vessel it joins -- which is the bulge this policy exists to remove.
    """
    r_vox = 6
    mask = cylinder(SHAPE, r_vox, 5, 75)
    inside = frame.seg_to_um(np.array([[8, 20, 20], [72, 20, 20]]))
    outside = frame.seg_to_um(np.array([[72, 36, 36]]))[0]
    graph = graph_from([*inside, outside], [(1, 2, 8, 900.0), (0, 1, 20, r_vox * SPACING)])

    kept = rp.measure_radii(graph, frame, mask, fallback_policy="retain")
    dropped = rp.measure_radii(graph, frame, mask, fallback_policy="drop")

    assert 0 in kept.fallback_segments and 0 in dropped.fallback_segments
    assert np.allclose(kept.radii[0], 900.0)
    measured_calibre = float(np.median(dropped.radii[1]))
    assert np.max(dropped.radii[0]) < 900.0
    assert dropped.radii[0] == pytest.approx(measured_calibre, rel=0.35)


def test_a_dropped_span_is_not_tapered_unless_asked():
    """Ramping between the anchors claims a taper nothing observed.

    `drop` hands an un-measured span to its junction neighbours. Interpolating
    between the two ends is smooth, but it asserts a steady narrowing across
    exactly the stretch where no cross-section could be measured -- the same
    unverified claim `bifurcation_tapers` makes, and off for the same reason.
    """
    import inspect
    sig = inspect.signature(rp.measure_radii)
    assert sig.parameters["fallback_taper"].default is False


def test_ownership_runs_next_to_a_junction_not_only_away_from_one(frame):
    """The gate this replaced was a whole-point disable, not a per-rival filter.

    `adjacent_overlap[i]` is `any(item[4] for item in rivals)`. It used to sit inside
    both `connected_nonadjacent` comprehensions, where it does not depend on the loop
    variable -- so one topologically-adjacent branch merely being *near* emptied the
    list, the 3-D watershed never ran, `BRANCH_OVERLAP` was never raised, and the
    merged blob was measured and accepted. Near a junction an adjacent branch always
    is near, so ownership was disabled at exactly the places two lumens fuse. On
    LADAF-2021-17 it resolved ~100 of 336,865 points.

    The fixture is the two touching tubes, plus a third stub welded to segment 0 so
    that segment 0 now *has* an adjacent neighbour. The geometry of the merge is
    unchanged; only the topology around it is. Ownership must still resolve.
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
        segments.append({"id": sid, "node1": 2 * sid, "node2": 2 * sid + 1,
                         "point_ids": ids, "strahler": 1})
    # A short stub sharing segment 0's start node: adjacent to 0, foreign to 1.
    stub = np.c_[np.arange(8, 20), np.full(12, 16), np.arange(20, 32)]
    stub_xyz = frame.seg_to_um(stub)
    nodes[4] = (*stub_xyz[-1], 0)
    ids = []
    for p in stub_xyz:
        points[pid] = (*p, 3 * SPACING)
        ids.append(pid)
        pid += 1
    segments.append({"id": 2, "node1": 0, "node2": 4, "point_ids": ids, "strahler": 1})
    graph = EditableGraph(Triple(nodes, points, segments))

    on = rp.measure_radii(graph, frame, mask)
    off = rp.measure_radii(graph, frame, mask, ownership_near_junctions=False)

    owned_on = int((on.resolution_mode[0] == rp.OWNED_PLANE).sum())
    owned_off = int((off.resolution_mode[0] == rp.OWNED_PLANE).sum())
    assert owned_on > owned_off, "the adjacent stub must no longer disable ownership"
    # And what it resolves to is this tube, not both of them welded together.
    resolved = on.radii[0][on.resolution_mode[0] == rp.OWNED_PLANE]
    assert np.median(resolved) == pytest.approx(4 * SPACING, rel=0.4)
