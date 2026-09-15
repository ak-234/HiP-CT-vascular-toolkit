"""Mask repair must mend real breaks without welding unrelated vessels."""

from __future__ import annotations

import numpy as np

from hipct_seg_debug.edit.reconnect import segmentation as seg
from hipct_seg_debug.edit.reconnect.candidates import Bridge

SPACING = np.array([10.0, 10.0, 10.0])
ORIGIN = np.zeros(3)


def tube(mask, z, y, x0, x1, r=2):
    zz, yy, xx = np.indices(mask.shape)
    body = (
        (np.abs(zz - z) <= r) & (np.abs(yy - y) <= r) & (xx >= x0) & (xx <= x1)
    )
    mask[body] = True
    return mask


def test_components_counts_and_measures():
    mask = np.zeros((20, 20, 40), dtype=bool)
    tube(mask, 10, 10, 2, 15)
    tube(mask, 10, 10, 25, 37)
    stats = seg.components(mask)
    assert stats.n == 2
    assert stats.sizes[1:].sum() == np.count_nonzero(mask)
    assert len(stats.order_by_size()) == 2


def test_components_uses_26_connectivity():
    """A diagonal step is one vessel, not two beads."""
    mask = np.zeros((6, 6, 6), dtype=bool)
    mask[2, 2, 2] = True
    mask[3, 3, 3] = True
    assert seg.components(mask, connectivity=3).n == 1
    assert seg.components(mask, connectivity=1).n == 2


def test_cull_small_removes_debris_and_keeps_the_vessel():
    mask = np.zeros((20, 20, 40), dtype=bool)
    tube(mask, 10, 10, 2, 37)
    mask[2, 2, 2] = True          # single-voxel speck
    mask[16, 16, 30:32] = True    # two-voxel speck

    out, removed = seg.cull_small(mask, min_voxels=20)
    assert removed == 2
    assert seg.components(out).n == 1
    assert out[10, 10, 20]


def test_cull_small_protects_the_largest_components():
    mask = np.zeros((20, 20, 40), dtype=bool)
    tube(mask, 5, 5, 2, 6, r=1)     # small but real
    tube(mask, 15, 15, 2, 37, r=2)  # large

    out, removed = seg.cull_small(mask, min_voxels=10_000, keep_largest=2)
    assert removed == 0
    assert seg.components(out).n == 2


def test_close_gaps_mends_a_one_voxel_break():
    mask = np.zeros((20, 20, 40), dtype=bool)
    tube(mask, 10, 10, 2, 19)
    tube(mask, 10, 10, 21, 37)
    assert seg.components(mask).n == 2

    out = seg.close_gaps(mask, radius_voxels=2)
    assert seg.components(out).n == 1


def test_close_gaps_is_a_no_op_at_radius_zero():
    mask = np.zeros((10, 10, 10), dtype=bool)
    mask[5, 5, 5] = True
    assert np.array_equal(seg.close_gaps(mask, 0), mask)


def test_paint_bridges_joins_two_components():
    mask = np.zeros((24, 24, 48), dtype=bool)
    tube(mask, 12, 12, 2, 18)
    tube(mask, 12, 12, 30, 45)
    assert seg.components(mask).n == 2

    # A bridge along +x at world y = z = 120 um, spanning the gap.
    coords = np.array([[180.0, 120.0, 120.0], [300.0, 120.0, 120.0]])
    bridge = Bridge("endpoint", 0, coords, np.array([25.0, 25.0]), target_node=1)

    out, painted = seg.paint_bridges(mask, [bridge], ORIGIN, SPACING)
    assert painted == 1
    assert seg.components(out).n == 1, "the painted bridge did not join the two halves"
    assert np.count_nonzero(out) > np.count_nonzero(mask)


def test_paint_bridges_tapers_between_the_two_radii():
    mask = np.zeros((32, 32, 32), dtype=bool)
    coords = np.array([[50.0, 160.0, 160.0], [250.0, 160.0, 160.0]])
    bridge = Bridge("endpoint", 0, coords, np.array([60.0, 10.0]), target_node=1)
    out, _ = seg.paint_bridges(mask, [bridge], ORIGIN, SPACING)

    # Cross-sectional area must shrink from the thick end to the thin one.
    thick = out[:, :, 6].sum()
    thin = out[:, :, 23].sum()
    assert thick > thin > 0, f"no taper: {thick} -> {thin}"


def test_paint_bridges_skips_rejected_ones():
    mask = np.zeros((16, 16, 16), dtype=bool)
    coords = np.array([[20.0, 80.0, 80.0], [140.0, 80.0, 80.0]])
    bridge = Bridge("endpoint", 0, coords, np.array([20.0, 20.0]), target_node=1)
    bridge.reject("test")
    out, painted = seg.paint_bridges(mask, [bridge], ORIGIN, SPACING)
    assert painted == 0
    assert not out.any()


def test_paint_bridges_clips_at_the_volume_edge():
    """A bridge running off the ROI must paint what it can, not raise."""
    mask = np.zeros((16, 16, 16), dtype=bool)
    coords = np.array([[80.0, 80.0, 80.0], [900.0, 80.0, 80.0]])
    bridge = Bridge("endpoint", 0, coords, np.array([20.0, 20.0]), target_node=1)
    out, painted = seg.paint_bridges(mask, [bridge], ORIGIN, SPACING)
    assert painted == 1
    assert out.any()


def test_report_and_compare_expose_a_topology_change():
    mask = np.zeros((20, 20, 40), dtype=bool)
    tube(mask, 10, 10, 2, 19)
    tube(mask, 10, 10, 21, 37)

    before = seg.report(mask, voxel_size_um=10.0)
    after = seg.report(seg.close_gaps(mask, 2), voxel_size_um=10.0)
    assert before["connected_components"] == 2
    assert after["connected_components"] == 1

    text = seg.compare(before, after)
    assert "connected_components" in text
    assert "-1" in text


def test_closing_that_welds_two_vessels_is_visible_in_the_report():
    """Closing is indiscriminate; the report is how you notice it went wrong."""
    mask = np.zeros((24, 24, 40), dtype=bool)
    tube(mask, 10, 7, 2, 37, r=2)    # spans y in [5, 9]
    tube(mask, 10, 16, 2, 37, r=2)   # spans y in [14, 18]: 4 clear voxels between
    before = seg.report(mask, voxel_size_um=10.0)
    assert before["connected_components"] == 2

    after = seg.report(seg.close_gaps(mask, 4), voxel_size_um=10.0)
    assert after["connected_components"] == 1, "the two vessels were welded"
    # Both numbers move, and that pairing is the signal: a closing that was asked
    # to mend one break should not also be adding voxels along the entire length
    # of two vessels. `compare` puts them side by side for exactly this reason.
    assert after["voxel_count"] > before["voxel_count"]
    text = seg.compare(before, after)
    assert "connected_components" in text and "voxel_count" in text
    assert "-1" in text, "the component drop should be reported as a delta"
