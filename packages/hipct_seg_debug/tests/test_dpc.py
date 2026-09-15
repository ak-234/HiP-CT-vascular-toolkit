"""The DPC walk must follow the image, not the straight line to the target.

Synthetic volumes make the right answer knowable: a tube that bends, a tube with
a break in it, and two tubes that pass close without touching. The last is the
one that matters -- a walk that hops between them would fabricate an anastomosis,
and the geometric gates alone cannot tell that case from a real break.
"""

from __future__ import annotations

import numpy as np
import pytest

from hipct_seg_debug.edit.reconnect.dpc import (
    DpcParams,
    validate,
    walk,
)
from hipct_seg_debug.edit.reconnect.probability import FieldProbability, Roi

SPACING = np.array([10.0, 10.0, 10.0])  # um per voxel, isotropic
ORIGIN = np.zeros(3)


def make_roi(shape=(40, 40, 40)) -> np.ndarray:
    return np.zeros(shape, dtype=np.float32)


def paint_tube(volume, path_zyx, radius_vox=2.5, value=1.0):
    """Draw a tube of `radius_vox` along a (N, 3) polyline in (z, y, x) indices."""
    zz, yy, xx = np.indices(volume.shape)
    grid = np.stack([zz, yy, xx], axis=-1).astype(np.float64)
    path = np.asarray(path_zyx, dtype=np.float64)
    for k in range(len(path) - 1):
        a, b = path[k], path[k + 1]
        ab = b - a
        length_sq = float(ab @ ab)
        t = np.clip(((grid - a) @ ab) / max(length_sq, 1e-9), 0.0, 1.0)
        closest = a + t[..., None] * ab
        volume[np.linalg.norm(grid - closest, axis=-1) <= radius_vox] = value
    return volume


def roi_from(volume, mask=None) -> Roi:
    return Roi(volume=volume, origin_um=ORIGIN, spacing_um=SPACING, mask=mask)


def to_um(zyx):
    """(z, y, x) index -> world um."""
    return ORIGIN + np.asarray(zyx, dtype=np.float64)[::-1] * SPACING


# ------------------------------------------------------------------- the Roi

def test_roi_index_and_world_round_trip():
    roi = roi_from(make_roi())
    pts = np.array([[0.0, 0, 0], [100.0, 200.0, 300.0]])
    assert np.allclose(roi.to_world(roi.to_index(pts)), pts)


def test_roi_index_order_is_zyx():
    roi = roi_from(make_roi())
    # x = 300 um at 10 um/voxel is index 30 on the *last* axis.
    idx = roi.to_index(np.array([[300.0, 100.0, 50.0]]))[0]
    assert np.allclose(idx, [5.0, 10.0, 30.0])


def test_roi_inside_rejects_out_of_bounds():
    roi = roi_from(make_roi((10, 10, 10)))
    # 9.5 is rejected as well as 12: trilinear interpolation there would need
    # voxel 10, which does not exist, so the last valid position is shape - 1.
    idx = np.array([[5.0, 5, 5], [-1.0, 5, 5], [5.0, 5, 9.0], [5.0, 5, 9.5]])
    assert list(roi.inside(idx)) == [True, False, True, False]


# ---------------------------------------------------------------- the walk

def test_the_walk_crosses_a_break_in_a_straight_tube():
    volume = make_roi()
    paint_tube(volume, [(20, 20, 4), (20, 20, 16)])
    paint_tube(volume, [(20, 20, 24), (20, 20, 35)])
    roi = roi_from(volume, mask=volume > 0.5)
    prob = FieldProbability(roi, sigmas=(1.0, 2.0))

    result = walk(
        roi, prob, to_um((20, 20, 16)), to_um((20, 20, 24)),
        start_direction=np.array([1.0, 0.0, 0.0]),  # +x in world
        params=DpcParams(max_steps=100),
    )
    assert result.reached, result.reason
    # It should stay on the tube's axis rather than wander off it.
    idx = roi.to_index(result.path_um)
    assert np.all(np.abs(idx[:, 0] - 20) < 3), "drifted in z"
    assert np.all(np.abs(idx[:, 1] - 20) < 3), "drifted in y"


def test_the_walk_follows_a_bend_rather_than_the_chord():
    """An L-shaped tube: the straight line to the target leaves the vessel."""
    volume = make_roi()
    corner = (20, 32, 8)
    paint_tube(volume, [(20, 8, 8), corner])
    paint_tube(volume, [corner, (20, 32, 32)])
    roi = roi_from(volume, mask=volume > 0.5)
    prob = FieldProbability(roi, sigmas=(1.0, 2.0))

    start, target = (20, 20, 8), (20, 32, 24)
    result = walk(
        roi, prob, to_um(start), to_um(target),
        start_direction=np.array([0.0, 1.0, 0.0]),  # +y in world
        params=DpcParams(max_steps=200),
    )
    assert result.reached, result.reason

    idx = roi.to_index(result.path_um)
    on_tube = volume[
        np.clip(idx[:, 0].round().astype(int), 0, volume.shape[0] - 1),
        np.clip(idx[:, 1].round().astype(int), 0, volume.shape[1] - 1),
        np.clip(idx[:, 2].round().astype(int), 0, volume.shape[2] - 1),
    ]
    assert on_tube.mean() > 0.85, "the walk left the vessel to cut the corner"
    # A chord would go diagonally; a path round the corner is longer.
    chord = np.linalg.norm(np.asarray(target, float) - np.asarray(start, float))
    steps = np.linalg.norm(np.diff(idx, axis=0), axis=1).sum()
    assert steps > chord * 1.15


def test_the_walk_does_not_hop_between_two_close_but_separate_tubes():
    """The false-positive case the geometric gates cannot see.

    Two parallel tubes with clear tissue between them. Crossing is geometrically
    perfectly reasonable -- short, straight, radius-matched -- so only the image
    can rule it out.
    """
    volume = make_roi()
    paint_tube(volume, [(20, 12, 4), (20, 12, 35)], radius_vox=2.0)
    paint_tube(volume, [(20, 28, 4), (20, 28, 35)], radius_vox=2.0)
    roi = roi_from(volume, mask=volume > 0.5)
    prob = FieldProbability(roi, sigmas=(1.0, 2.0))

    start = to_um((20, 12, 20))
    result = walk(
        roi, prob, start, to_um((20, 28, 20)),
        start_direction=np.array([0.0, 1.0, 0.0]),
        params=DpcParams(max_steps=80),
    )
    # Whether or not it arrives, the path must not be accepted: the tissue it
    # crosses has no centreline signal, so the probability series has a hole.
    reference = prob(np.array([to_um((20, 12, z)) for z in range(8, 32)]))
    ok, reason, stats = validate(result, reference)
    assert not ok, f"a hop across empty tissue was accepted: {stats}"
    # Either refusal is correct, and the walk preferring to stay on its own
    # vessel until it runs out of steps is the stronger of the two: the
    # probability term never made crossing look attractive in the first place.
    assert reason in ("ran out of steps", "every neighbour already visited") or any(
        phrase in reason for phrase in ("hole", "below the parent", "too low")
    ), reason


def test_validate_rejects_a_path_with_a_hole_in_it():
    """A mean that looks fine, hiding a few steps through tissue."""
    from hipct_seg_debug.edit.reconnect.dpc import DpcResult

    series = np.concatenate([np.full(8, 0.9), np.full(3, 0.02), np.full(8, 0.9)])
    result = DpcResult(np.zeros((19, 3)), series, True, 18)
    ok, reason, stats = validate(result, np.full(20, 0.9))
    assert stats["mean_probability"] > 0.7, "the mean should look acceptable"
    assert not ok
    assert "hole" in reason


def test_the_walk_stops_at_the_edge_of_the_region():
    volume = make_roi((20, 20, 20))
    paint_tube(volume, [(10, 10, 2), (10, 10, 17)])
    roi = roi_from(volume, mask=volume > 0.5)
    prob = FieldProbability(roi, sigmas=(1.0,))
    result = walk(
        roi, prob, to_um((10, 10, 10)), np.array([1e6, 1e6, 1e6]),
        params=DpcParams(max_steps=200),
    )
    assert not result.reached
    assert result.reason in ("walked out of the region of interest",
                            "ran out of steps",
                            "every neighbour already visited",
                            "no neighbour satisfies the direction constraint")


def test_the_walk_never_revisits_a_voxel():
    volume = make_roi()
    paint_tube(volume, [(20, 20, 4), (20, 20, 35)])
    roi = roi_from(volume, mask=volume > 0.5)
    prob = FieldProbability(roi, sigmas=(1.0, 2.0))
    result = walk(roi, prob, to_um((20, 20, 8)), to_um((20, 20, 30)),
                  params=DpcParams(max_steps=200))
    idx = np.round(roi.to_index(result.path_um)).astype(int)
    assert len(np.unique(idx, axis=0)) == len(idx), "the walk revisited a voxel"


# ------------------------------------------------------------- acceptance

def test_validate_rejects_a_walk_that_never_arrived():
    from hipct_seg_debug.edit.reconnect.dpc import DpcResult

    result = DpcResult(np.zeros((3, 3)), np.array([0.9, 0.9, 0.9]), False, 2, "ran out")
    ok, reason, _ = validate(result, np.array([0.9]))
    assert not ok and reason == "ran out"


def test_validate_rejects_a_low_probability_path():
    from hipct_seg_debug.edit.reconnect.dpc import DpcResult

    result = DpcResult(np.zeros((12, 3)), np.full(12, 0.02), True, 11)
    ok, reason, stats = validate(result, np.full(20, 0.9))
    assert not ok
    assert "too low" in reason
    assert stats["mean_probability"] == pytest.approx(0.02)


def test_validate_rejects_a_path_far_worse_than_its_parent_vessel():
    from hipct_seg_debug.edit.reconnect.dpc import DpcResult

    # Absolutely acceptable, but a big drop from the vessel it continues.
    noise = np.random.default_rng(0).normal(0.22, 0.01, 20)
    result = DpcResult(np.zeros((20, 3)), noise, True, 19)
    ok, reason, stats = validate(result, np.full(30, 0.95))
    assert not ok
    assert "below the parent vessel" in reason
    assert stats["probability_drop"] > 0.6


def test_validate_accepts_a_steady_path_matching_its_parent():
    from hipct_seg_debug.edit.reconnect.dpc import DpcResult

    rng = np.random.default_rng(1)
    series = rng.normal(0.8, 0.02, 40)
    result = DpcResult(np.zeros((40, 3)), series, True, 39)
    ok, reason, stats = validate(result, np.full(40, 0.82))
    assert ok, f"{reason} {stats}"


def test_validate_flags_a_greyscale_step_at_the_join():
    from hipct_seg_debug.edit.reconnect.dpc import DpcResult

    rng = np.random.default_rng(2)
    result = DpcResult(np.zeros((30, 3)), rng.normal(0.8, 0.02, 30), True, 29)
    grey = np.concatenate([np.full(15, 0.2), np.full(15, 0.9)])
    ok, reason, stats = validate(result, np.full(30, 0.8), grey_along_path=grey)
    assert not ok
    assert "discontinuity" in reason


# ------------------------------------------------------------- probability

def test_field_probability_peaks_on_the_vessel_axis():
    volume = make_roi()
    paint_tube(volume, [(20, 20, 4), (20, 20, 35)], radius_vox=4.0)
    roi = roi_from(volume, mask=volume > 0.5)
    prob = FieldProbability(roi, sigmas=(1.0, 2.0, 4.0))

    on_axis = prob(np.array([to_um((20, 20, 20))]))[0]
    off_axis = prob(np.array([to_um((20, 26, 20))]))[0]   # just outside the wall
    background = prob(np.array([to_um((8, 8, 20))]))[0]
    assert on_axis > off_axis > background - 1e-9
    assert on_axis > 0.3


def test_field_probability_works_without_a_mask():
    """Across a true gap there is no segmentation, so it must not need one."""
    volume = make_roi()
    paint_tube(volume, [(20, 20, 4), (20, 20, 35)], radius_vox=3.0)
    prob = FieldProbability(roi_from(volume, mask=None), sigmas=(1.0, 2.0))
    on_axis = prob(np.array([to_um((20, 20, 20))]))[0]
    background = prob(np.array([to_um((8, 8, 20))]))[0]
    assert on_axis > background


def test_learned_probability_separates_axis_from_wall():
    from hipct_seg_debug.edit.reconnect.probability import LearnedProbability

    volume = make_roi((32, 32, 48))
    paint_tube(volume, [(16, 16, 4), (16, 16, 43)], radius_vox=4.0)
    volume += np.random.default_rng(3).normal(0, 0.03, volume.shape).astype(np.float32)
    roi = roi_from(volume)

    axis_um = np.array([to_um((16, 16, z)) for z in range(6, 42)])
    model = LearnedProbability(roi).fit(axis_um, radius_um=40.0, seed=3)
    assert model.training_score > 0.85

    on_axis = model(np.array([to_um((16, 16, 24))]))[0]
    wall = model(np.array([to_um((16, 22, 24))]))[0]
    assert on_axis > wall, f"axis {on_axis:.3f} is not above wall {wall:.3f}"


def test_learned_probability_refuses_to_predict_before_training():
    from hipct_seg_debug.edit.reconnect.probability import LearnedProbability

    model = LearnedProbability(roi_from(make_roi((8, 8, 8))))
    with pytest.raises(RuntimeError, match="fit"):
        model(np.zeros((1, 3)))


# --------------------------------------------------------------- the ROI builder
#
# The walk is only as good as the box it is handed. Two ways to get that wrong are
# invisible until the results are nonsense: an ROI whose `origin_um` is off by the
# crop offset puts every world coordinate in the wrong voxel, and a mask sampled on
# the lattice's own grid rather than the raw one is silently half the resolution.


class FakeStack:
    """A `TiffStack` stand-in: a whole volume in memory, same three methods."""

    def __init__(self, volume):
        self.volume = np.asarray(volume)
        self.dtype = self.volume.dtype
        self.n_slices, self.n_rows, self.n_cols = self.volume.shape
        self.shape = self.volume.shape
        self.reads = 0

    def read_slice(self, z):
        self.reads += 1
        return self.volume[z]

    def read_window(self, z, row0, row1, col0, col1):
        self.reads += 1
        return self.volume[z, row0:row1, col0:col1]

    def read_stack_window(self, z_lo, z_hi, row0, row1, col0, col1):
        return np.stack(
            [self.read_window(z, row0, row1, col0, col1) for z in range(z_lo, z_hi)]
        )


class FakeLattice:
    """A binary label lattice at half the raw resolution, sliced like the real one."""

    def __init__(self, volume):
        self.volume = np.asarray(volume)

    def slice_z(self, z):
        return self.volume[z]


def dpc_frame(raw_shape=(64, 60, 56), voxel=10.0, origin=(50.0, 50.0, 50.0)):
    """A `WorldFrame` with a real 2x binning and a non-zero crop, as on HiP-CT."""
    from hipct_seg_debug.frame import WorldFrame

    class Info:
        # (nx, ny, nz), matching Amira's order.
        dims = (20, 20, 24)
        spacing = (voxel * 2, voxel * 2, voxel * 2)

    Info.origin = origin
    return WorldFrame.from_inputs(raw_shape, voxel, Info)


def test_roi_origin_survives_the_crop_offset():
    """A world point must land on the same voxel through the ROI as through the frame."""
    from hipct_seg_debug.edit.reconnect import roi as roi_mod

    frame = dpc_frame()
    volume = np.arange(64 * 60 * 56, dtype=np.uint16).reshape(64, 60, 56)
    stack = FakeStack(volume)

    probe = np.array([[220.0, 180.0, 260.0]])
    roi = roi_mod.build(stack, frame, probe[0] - 40.0, probe[0] + 40.0)

    # The value at the probe point must be the value the full volume has there.
    z, y, x = np.round(frame.um_to_raw(probe)[0]).astype(int)
    idx = np.round(roi.to_index(probe)[0]).astype(int)
    assert roi.volume[tuple(idx)] == volume[z, y, x]


def test_roi_mask_is_sampled_onto_the_raw_grid_not_the_lattice_grid():
    from hipct_seg_debug.edit.reconnect import roi as roi_mod

    frame = dpc_frame()
    stack = FakeStack(np.zeros((64, 60, 56), dtype=np.uint16))
    labels = np.zeros((24, 20, 20), dtype=np.uint8)
    labels[10, 8, 6] = 1  # one segmentation voxel, which is 2x2x2 raw voxels

    roi = roi_mod.build(
        stack, frame, np.array([50.0, 50.0, 50.0]), np.array([500.0, 500.0, 500.0]),
        labels=FakeLattice(labels),
    )
    assert roi.mask.shape == roi.volume.shape, "mask and volume must share a grid"
    # One lattice voxel at 2x binning covers exactly eight raw voxels.
    assert int(roi.mask.sum()) == 8

    # ...and it must sit where that lattice voxel actually is in the world.
    centre = frame.seg_to_um(np.array([[6.0, 8.0, 10.0]]))
    assert roi.mask[tuple(np.round(roi.to_index(centre)[0]).astype(int))]


def test_build_many_reads_each_slice_once():
    """The whole reason `build_many` exists: overlapping boxes must share reads."""
    from hipct_seg_debug.edit.reconnect import roi as roi_mod

    frame = dpc_frame()
    stack = FakeStack(np.zeros((64, 60, 56), dtype=np.uint16))
    # Three boxes covering the same z range, deliberately.
    spans = [
        (np.array([100.0, 100.0, 100.0]), np.array([200.0, 200.0, 300.0])),
        (np.array([150.0, 150.0, 100.0]), np.array([250.0, 250.0, 300.0])),
        (np.array([200.0, 200.0, 100.0]), np.array([300.0, 300.0, 300.0])),
    ]
    rois = roi_mod.build_many(stack, frame, spans)
    assert len(rois) == 3 and all(r is not None for r in rois)

    distinct_z = set()
    for lo, hi in spans:
        box = roi_mod.box_for(frame, lo, hi)
        distinct_z.update(range(box[0], box[1]))
    assert stack.reads == len(distinct_z), (
        f"read {stack.reads} slices for {len(distinct_z)} distinct ones"
    )
    # And each box still got its own correct geometry.
    for roi, (lo, _hi) in zip(rois, spans):
        assert roi.inside(roi.to_index(lo.reshape(1, 3)))[0]


def test_build_many_returns_none_for_a_box_too_large_to_read():
    from hipct_seg_debug.edit.reconnect import roi as roi_mod

    frame = dpc_frame()
    stack = FakeStack(np.zeros((64, 60, 56), dtype=np.uint16))
    small = (np.array([100.0, 100.0, 100.0]), np.array([200.0, 200.0, 200.0]))
    monkey = roi_mod.MAX_VOXELS
    try:
        roi_mod.MAX_VOXELS = 8  # smaller than any real box
        assert roi_mod.build_many(stack, frame, [small]) == [None]
        with pytest.raises(roi_mod.RoiTooLarge):
            roi_mod.box_for(frame, *small)
    finally:
        roi_mod.MAX_VOXELS = monkey


def test_roi_box_never_collapses_below_the_walk_neighbourhood():
    """A zero-length bridge must still get a box the walk can step around in."""
    from hipct_seg_debug.edit.reconnect import roi as roi_mod

    frame = dpc_frame()
    point = np.array([300.0, 300.0, 300.0])
    z0, z1, y0, y1, x0, x1 = roi_mod.box_for(frame, point, point)
    assert min(z1 - z0, y1 - y0, x1 - x0) >= roi_mod.MIN_SPAN_VOXELS
