"""De-looping, spur pruning, smoothing and re-centring.

Each of these deletes or moves something, so each test pins both what it must do and
what it must *not* -- an over-eager prune that removes a real vessel, or a de-loop that
disconnects a component, are silent failures that only show up as a wrong flow
simulation weeks later.
"""

from __future__ import annotations

import numpy as np
import pytest

from hipct_seg_debug.edit import skeleton_optimise as so
from hipct_seg_debug.edit import supermetric as sm

from .conftest_geometry import (
    SPACING,
    axis_graph,
    graph_from,
    make_frame,
    slit,
    staircase_graph,
    wobbly_graph,
)

# ------------------------------------------------------------------ de-looping


def test_a_diamond_loses_its_thinner_arm():
    """The break goes where the segmentation is least sure: the smallest radius."""
    g = graph_from(
        [(0, 0, 0), (1000, 0, 0), (2000, 0, 0), (1000, 800, 0)],
        [(0, 1, 10, 200.0), (1, 2, 10, 200.0), (0, 3, 10, 60.0), (3, 2, 10, 60.0)],
    )
    assert so.count_cycles(g) == 1

    report = so.remove_loops(g)

    assert so.count_cycles(g) == 0
    assert len(g.components()) == 1, "de-looping must never disconnect the network"
    assert len(report.breaks) == 1
    assert report.breaks[0].radius_um == pytest.approx(60.0)


def test_a_figure_of_eight_loses_both_loops_and_stays_connected():
    g = graph_from(
        [(0, 0, 0), (1000, 0, 0), (2000, 0, 0), (1000, 800, 0), (1000, -800, 0)],
        [(0, 1, 8, 200.0), (1, 2, 8, 200.0), (0, 3, 8, 90.0), (3, 2, 8, 90.0),
         (0, 4, 8, 50.0), (4, 2, 8, 50.0)],
    )
    assert so.count_cycles(g) == 2

    so.remove_loops(g)

    assert so.count_cycles(g) == 0
    assert len(g.components()) == 1


def test_two_parallel_segments_are_de_looped():
    """The exact shape of a vessel that collapsed mid-way and was segmented as two.

    `networkx.cycle_basis` needs a simple graph, so it cannot see a parallel pair at
    all -- they have to be found separately or this case passes through untouched,
    which is the one case the de-looping exists for.
    """
    g = graph_from(
        [(0, 0, 0), (2000, 0, 0)],
        [(0, 1, 10, 200.0), (0, 1, 10, 80.0)],  # two edges, same two nodes
    )
    assert so.count_cycles(g) == 1

    report = so.remove_loops(g)

    assert so.count_cycles(g) == 0
    assert len(g.segments) == 1
    assert len(g.components()) == 1
    assert report.breaks[0].radius_um == pytest.approx(80.0), "the thinner lumen goes"


def test_a_self_loop_is_removed():
    g = graph_from(
        [(0, 0, 0), (1000, 0, 0)],
        [(0, 1, 8, 200.0), (1, 1, 8, 50.0)],
    )
    so.remove_loops(g)
    assert so.count_cycles(g) == 0
    assert all(s["node1"] != s["node2"] for s in g.segments)


def test_a_tree_is_left_alone():
    g = graph_from(
        [(0, 0, 0), (1000, 0, 0), (2000, 500, 0), (2000, -500, 0)],
        [(0, 1, 8, 200.0), (1, 2, 8, 120.0), (1, 3, 8, 120.0)],
    )
    before = len(g.segments)
    report = so.remove_loops(g)
    assert report.breaks == []
    assert len(g.segments) == before


# --------------------------------------------------------------------- pruning


def _trunk_with(branch_end, branch_radius, n_points=4):
    """A 2 mm trunk of radius 200 um with one branch off its midpoint."""
    return graph_from(
        [(0, 0, 0), (1000, 0, 0), (2000, 0, 0), branch_end],
        [(0, 1, 12, 200.0), (1, 2, 12, 200.0), (1, 3, n_points, branch_radius)],
    )


def test_a_short_thin_leaf_is_pruned():
    """The user's rule: shorter than the largest local vessel radius, so it goes."""
    g = _trunk_with((1000, 60, 0), 40.0)   # 60 um long against a 200 um parent
    report = so.prune_spurs(g)

    assert report.n_removed == 1
    assert len(g.components()) == 1
    # The junction it hung from is now degree 2, so the trunk is one segment again.
    assert len(g.segments) == 1
    assert report.n_contracted == 1


def test_a_long_leaf_survives():
    g = _trunk_with((1000, 900, 0), 120.0, n_points=15)
    report = so.prune_spurs(g)
    assert report.n_removed == 0
    assert len(g.segments) == 3


def test_a_short_but_thick_leaf_survives():
    """A stub as thick as its parent is a truncated vessel, not a thinning artefact."""
    g = _trunk_with((1000, 60, 0), 190.0)
    report = so.prune_spurs(g)
    assert report.n_removed == 0
    assert report.kept_thick == 1


def test_a_leaf_at_the_lattice_boundary_survives():
    """A vessel the scan cut off is real however short its stub looks."""
    g = _trunk_with((1000, 60, 0), 40.0)
    bbox = np.array([0.0, 2000.0, 0.0, 100.0, 0.0, 100.0])  # tip sits on the y face
    report = so.prune_spurs(g, bbox_um=bbox, boundary_margin_um=200.0)

    assert report.n_removed == 0
    assert report.kept_boundary == 1


def test_pruning_never_deletes_a_lone_segment():
    g = graph_from([(0, 0, 0), (60, 0, 0)], [(0, 1, 3, 200.0)])
    so.prune_spurs(g)
    assert len(g.segments) == 1


# ------------------------------------------------------------------- smoothing


def test_smoothing_flattens_a_saw_tooth_and_pins_the_ends():
    n = 41
    xs = np.linspace(0.0, 4000.0, n)
    ys = np.where(np.arange(n) % 2 == 0, 0.0, 60.0)
    g = graph_from([(xs[0], ys[0], 0), (xs[-1], ys[-1], 0)], [(0, 1, n, 300.0)])
    with g.batch("lay out the saw-tooth"):
        g.set_segment_coords(0, np.stack([xs, ys, np.zeros(n)], axis=1))

    before = g.coords(0)
    so.smooth_centreline(g, window_um=200.0)
    after = g.coords(0)

    assert np.abs(np.diff(after[:, 1])).mean() < 0.1 * np.abs(np.diff(before[:, 1])).mean()
    assert np.allclose(before[0], after[0]), "endpoints belong to nodes"
    assert np.allclose(before[-1], after[-1])


def test_smoothing_cannot_move_a_point_further_than_its_radius_allows():
    n = 21
    xs = np.linspace(0.0, 2000.0, n)
    ys = np.where(np.arange(n) % 2 == 0, 0.0, 500.0)  # a violent zig-zag
    g = graph_from([(xs[0], ys[0], 0), (xs[-1], ys[-1], 0)], [(0, 1, n, 20.0)])
    with g.batch("lay out"):
        g.set_segment_coords(0, np.stack([xs, ys, np.zeros(n)], axis=1))

    before = g.coords(0)
    report = so.smooth_centreline(g, window_um=1500.0, max_move_frac=0.5)
    moved = np.linalg.norm(g.coords(0) - before, axis=1)

    assert report.n_clamped > 0
    assert moved.max() <= 0.5 * 20.0 + 1e-6


def test_a_window_below_the_point_spacing_is_reported_not_silent():
    """A no-op that looks like "smoothing did not help" has to announce itself."""
    n = 21
    xs = np.linspace(0.0, 2000.0, n)          # points 100 um apart in x...
    ys = np.where(np.arange(n) % 2 == 0, 0.0, 500.0)   # ...but ~510 um apart in arc
    g = graph_from([(xs[0], ys[0], 0), (xs[-1], ys[-1], 0)], [(0, 1, n, 20.0)])
    with g.batch("lay out"):
        g.set_segment_coords(0, np.stack([xs, ys, np.zeros(n)], axis=1))

    report = so.smooth_centreline(g, window_um=400.0)

    assert report.n_moved == 0
    assert report.n_window_too_small == 1
    assert "raise --smooth-um" in report.describe()


# ----------------------------------------------------------------- re-centring


def test_recentre_finds_the_middle_of_a_collapsed_slit():
    """The case the area-centroid choice exists for.

    A slit's distance transform has a flat ridge along its major axis, so an
    EDT-maximum rule could leave the centreline anywhere along it. The centroid is
    unique, and it is the middle.

    Undamped and with both move bounds released, so this measures the centroid and
    nothing else; that the damped, bounded default gets there too is
    `test_recentre_converges_rather_than_oscillating`'s job.
    """
    shape = (40, 60, 60)
    frame = make_frame(shape)
    half_y, cy, cz = 8, 30, 20
    mask = slit(shape, half_y=half_y, half_z=1, x0=5, x1=55, cy=cy, cz=cz)

    off_centre = 6  # voxels along the slit's major axis, where a ridge would sit
    g = axis_graph(frame, 6, 54, 40.0, cy=cy + off_centre, cz=cz)
    before = g.coords(0)[1:-1, 1].mean()

    report = so.recentre(g, frame, mask, max_move_frac=6.0, damping=1.0,
                         max_move_spacing=0.0)
    after = g.coords(0)[1:-1, 1].mean()

    assert report.n_moved > 0
    assert abs(before - cy * SPACING) > 50.0, "it really did start off centre"
    assert after == pytest.approx(cy * SPACING, abs=SPACING)
    assert sm.cl_sensitivity(g, frame, mask) == pytest.approx(1.0)


def test_recentre_converges_rather_than_oscillating():
    """Damped passes approach the centre monotonically and settle there.

    A full step re-estimates the plane from the line it just moved, so it can overshoot
    and come back; the damped iteration is the standard remedy. What matters is that the
    error shrinks every pass and that further passes then do nothing, because
    `optimise_skeleton` runs a fixed number of them and cannot check.

    Sampled at one radius per point, which is the order the real graphs use -- `lee.am`
    is 0.58 radii per point and the Avizo graph 0.40. A fixture four times denser than
    that would be bounded by the fold guard rather than by the centroid, and would be
    measuring the wrong thing.
    """
    shape = (40, 60, 60)
    frame = make_frame(shape)
    cy, cz = 30, 20
    mask = slit(shape, half_y=8, half_z=1, x0=5, x1=55, cy=cy, cz=cz)
    g = axis_graph(frame, 6, 54, 40.0, cy=cy + 6, cz=cz, step=4)

    origin = {sid: g.coords(sid).copy() for sid in g.segment_ids()}
    errors = [abs(g.coords(0)[1:-1, 1].mean() - cy * SPACING)]
    for _ in range(5):
        so.recentre(g, frame, mask, max_move_frac=6.0, origin=origin)
        errors.append(abs(g.coords(0)[1:-1, 1].mean() - cy * SPACING))

    # Monotone only while there is something left to fix. The centroid of a *voxelised*
    # slit resolves to about a voxel, so once inside one spacing the remaining motion is
    # quantisation, and demanding monotonicity there is demanding that noise behave.
    for i, (prev, now) in enumerate(zip(errors, errors[1:])):
        assert now < prev or prev < SPACING, f"pass {i + 1} moved away: {errors}"
    assert errors[-1] < SPACING, f"did not settle on the centre: {errors}"
    assert errors[-1] < errors[0] / 4
    # Settled, not orbiting.
    assert abs(errors[-1] - errors[-2]) < SPACING / 4


def test_recentre_holds_points_near_a_junction_but_not_near_a_free_end():
    """The margin exists for the carina, where the section is two lumens, not one.

    A free end is one vessel stopping: its section is a single lumen whose centroid is
    exactly what re-centring wants. Holding the margin there too would abandon every
    vessel tip for nothing.
    """
    shape = (40, 60, 60)
    frame = make_frame(shape)
    mask = slit(shape, half_y=8, half_z=1, x0=5, x1=55, cy=30, cz=20)

    free_ended = axis_graph(frame, 6, 54, 40.0, cy=36, cz=20)
    report = so.recentre(free_ended, frame, mask, max_move_frac=6.0)
    assert report.n_near_junction == 0, "both ends are terminal"

    # Split it so the midpoint becomes a node, then hang a third branch off that node
    # to make it a real junction.
    mid = len(free_ended.coords(0)) // 2
    nid, first, _second = free_ended.split_segment(free_ended.segment_ids()[0], mid)
    stub = np.linspace(free_ended.nodes[nid][:3],
                       np.array(free_ended.nodes[nid][:3]) + [0.0, 200.0, 0.0], 6)
    tip = free_ended.add_node(stub[-1])
    free_ended.add_segment(nid, tip, stub, np.full(len(stub), 40.0))

    junction_report = so.recentre(free_ended, frame, mask, max_move_frac=6.0)
    assert junction_report.n_near_junction > 0


def test_recentre_leaves_an_already_central_line_alone():
    shape = (40, 40, 60)
    frame = make_frame(shape)
    from .conftest_geometry import cylinder

    mask = cylinder(shape, 5, 5, 55)
    g = axis_graph(frame, 6, 54, 50.0, cy=20, cz=20)
    before = g.coords(0).copy()

    so.recentre(g, frame, mask)

    assert np.allclose(g.coords(0), before, atol=SPACING / 2)


# ------------------------------------------------------- the cut plane's normal


def _normal_rotation_deg(coords, radii, **kw):
    """Angle between each consecutive pair of estimated plane normals, in degrees."""
    t = so.plane_normals(coords, radii, so._arclength(coords), **kw)
    t = t / np.linalg.norm(t, axis=1)[:, None]
    cos = np.abs(np.einsum("ij,ij->i", t[:-1], t[1:]))
    return np.degrees(np.arccos(np.clip(cos, 0.0, 1.0)))


def test_a_radius_wide_chord_sees_through_the_voxel_staircase():
    """The measurement that motivated the whole fix, on a fixture that reproduces it.

    A thinned skeleton of a straight vessel staircases, so its adjacent-point tangent
    swings wildly while the vessel does not move at all. Every degree of that swing tilts
    the cut plane, and a tilted plane finds a different centroid -- which is the zig-zag.
    """
    frame = make_frame((40, 60, 60))
    g = staircase_graph(frame, 5, 55, 40.0, cy=30, cz=20)
    coords, radii = g.coords(0), g.radii(0)

    central = np.gradient(coords, axis=0)
    central = central / np.linalg.norm(central, axis=1)[:, None]
    cos = np.abs(np.einsum("ij,ij->i", central[:-1], central[1:]))
    staircase_swing = np.degrees(np.arccos(np.clip(cos, 0.0, 1.0)))

    # 19.5 degrees is what LADAF-2024-28's `lee.am` measures at stride 1, so a fixture
    # above that is representative rather than contrived.
    assert np.median(staircase_swing) > 19.5, "the fixture is not staircased"
    assert np.median(_normal_rotation_deg(coords, radii)) < 5.0


def test_the_tangent_window_is_measured_in_radii_not_points():
    """Scale equivariance: enlarge the whole geometry and the normals do not turn.

    A window counted in *points* would span a different physical length on the same
    vessel sampled more finely, which is precisely how `np.gradient` came to depend on
    the voxel size. Measuring it in each point's own radius removes that dependence, and
    this is the property that says so.
    """
    frame = make_frame((40, 60, 60))
    g = staircase_graph(frame, 5, 55, 40.0, cy=30, cz=20)
    coords, radii = g.coords(0), g.radii(0)

    small = so.plane_normals(coords, radii, so._arclength(coords))
    for k in (2.0, 37.5):
        big = so.plane_normals(k * coords, k * radii, so._arclength(k * coords))
        unit = lambda t: t / np.linalg.norm(t, axis=1)[:, None]  # noqa: E731
        assert np.allclose(unit(big), unit(small), atol=1e-12)


def test_a_short_segment_falls_back_to_the_central_difference():
    """No window fits, so it degrades rather than returning a zero direction."""
    coords = np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0], [20.0, 0.0, 0.0]])
    radii = np.full(3, 1e-6)  # a window far narrower than the point spacing
    t = so.plane_normals(coords, radii, so._arclength(coords))
    assert np.all(np.linalg.norm(t, axis=1) > 0)
    assert np.allclose(t / np.linalg.norm(t, axis=1)[:, None], [1.0, 0.0, 0.0])


def test_recentre_pulls_a_jittering_skeleton_onto_the_axis():
    """End to end, on the shape thinning actually leaves inside a straight tube.

    This is the whole point of re-centring, and the case that used to fail: a line that
    is right to within a voxel but never smooth. It must come out closer to the axis and
    without a single doubling-back, because on LADAF-2024-28 the old code answered this
    by scattering 4.4% of points into out-and-back spikes.
    """
    shape = (40, 60, 60)
    frame = make_frame(shape)
    from .conftest_geometry import cylinder

    mask = cylinder(shape, 5, 5, 55, cy=30, cz=20)
    g = wobbly_graph(frame, 6, 54, 50.0, cy=30, cz=20)
    axis_um = 30 * SPACING
    before = np.max(np.abs(g.coords(0)[1:-1, 1] - axis_um))

    origin = {sid: g.coords(sid).copy() for sid in g.segment_ids()}
    for _ in range(3):
        so.recentre(g, frame, mask, origin=origin)

    assert so.roughness(g).reversing == 0.0, "re-centring introduced a doubling-back"
    after = np.max(np.abs(g.coords(0)[1:-1, 1] - axis_um))
    assert after < before, f"no closer to the axis: {before} -> {after}"
    assert after <= SPACING


def test_no_point_may_step_past_its_own_neighbour():
    """The guard the Avizo graph needed, where the tangent was never the problem.

    Its points sit 0.40 radii apart, so a clamp expressed only in radii let a half-radius
    move carry a point clean past the one in front of it and the segment folded --
    re-centring took its doubling-back from 9.69% to 10.28%. A radius bound cannot see
    this: it is a property of the chain, not of any one point.
    """
    old = np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0], [20.0, 0.0, 0.0],
                    [30.0, 0.0, 0.0]])
    new = old.copy()
    new[1] += [40.0, 0.0, 0.0]  # a leap past points 2 and 3

    out, folded = so._clamp_to_spacing(old, new, so.RECENTRE_MAX_MOVE_SPACING)

    assert folded == 1
    assert np.linalg.norm(out[1] - old[1]) == pytest.approx(5.0), "half the 10 um gap"
    assert np.all(np.diff(out[:, 0]) > 0), "still in order"


def test_a_move_that_deepens_a_doubling_back_is_put_back():
    """The only guard that holds on an oversampled graph.

    The Avizo export samples 0.16 radii per point, so 9.69% of its points already turn
    past 120 degrees untouched. There a 15 um nudge -- a quarter of the 60 um gap, so
    inside every geometric bound -- takes a 110-degree corner to 137. This rule is stated
    in the quantity being protected instead, so it catches that case by construction.
    """
    old = np.array([[0.0, 0.0, 0.0], [100.0, 60.0, 0.0], [200.0, 0.0, 0.0]])
    assert so._turn_deg(old)[1] < so.Roughness.TURN_DEG, "starts merely bent"

    new = old.copy()
    new[1] += [0.0, 120.0, 0.0]  # push the corner out until it folds
    assert so._turn_deg(new)[1] > so.Roughness.TURN_DEG

    out, reverted = so._revert_new_reversals(old, new, so.Roughness.TURN_DEG)
    assert reverted >= 1
    assert np.array_equal(out[1], old[1])


def test_reverting_never_leaves_a_point_worse_than_it_started():
    """The property the whole guard exists to provide, on a line built to break it.

    Putting one point back moves two more angles, so a rule that only examines the
    offender leaves neighbours behind -- which is how the Avizo graph's >150 degree tail
    grew while its >120 degree count fell. Stated as a postcondition instead, over a
    randomised zig-zag dense enough that every point interacts with both neighbours.
    """
    rng = np.random.default_rng(7)
    for _ in range(25):
        n = 40
        old = np.zeros((n, 3))
        old[:, 0] = np.arange(n) * 50.0
        old[:, 1] = rng.normal(0.0, 40.0, n)
        new = old + rng.normal(0.0, 60.0, (n, 3)) * [0.2, 1.0, 1.0]

        out, _ = so._revert_new_reversals(old, new, so.Roughness.TURN_DEG)

        before, after = so._turn_deg(old), so._turn_deg(out)
        bad = (after > so.Roughness.TURN_DEG) & (after > before + 1e-9)
        assert not bad.any(), f"{int(bad.sum())} point(s) left worse than they started"


def test_a_point_already_doubled_back_may_still_be_improved():
    """Reverting is for moves that make it *worse*, not for abandoning bad points."""
    old = np.array([[0.0, 0.0, 0.0], [100.0, 400.0, 0.0], [200.0, 0.0, 0.0]])
    assert so._turn_deg(old)[1] > so.Roughness.TURN_DEG, "starts folded"

    new = old.copy()
    new[1] = [100.0, 150.0, 0.0]  # pulled back towards its neighbours
    out, reverted = so._revert_new_reversals(old, new, so.Roughness.TURN_DEG)

    assert reverted == 0
    assert np.array_equal(out, new)


def test_the_fold_guard_leaves_a_modest_move_alone():
    old = np.array([[0.0, 0.0, 0.0], [100.0, 0.0, 0.0], [200.0, 0.0, 0.0]])
    new = old.copy()
    new[1] += [0.0, 10.0, 0.0]  # a tenth of the gap, sideways
    out, folded = so._clamp_to_spacing(old, new, so.RECENTRE_MAX_MOVE_SPACING)
    assert folded == 0
    assert np.array_equal(out, new)


# ---------------------------------------------------------------- the roughness


def test_roughness_counts_a_doubling_back_and_ignores_a_smooth_line():
    frame = make_frame((40, 60, 60))
    straight = axis_graph(frame, 5, 55, 40.0, cy=30, cz=20)
    assert so.roughness(straight).reversing == 0.0

    coords = straight.coords(0).copy()
    coords[10] += [0.0, 900.0, 0.0]  # one point flung sideways: out and back
    straight.set_segment_coords(0, coords)
    spiked = so.roughness(straight)
    assert spiked.reversing > 0.0
    assert spiked.max_step_um > 800.0
    assert spiked.length_mm > so.roughness(axis_graph(
        frame, 5, 55, 40.0, cy=30, cz=20)).length_mm
