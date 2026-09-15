"""The streaming component index: does it agree with the array-at-once answer?

Everything downstream rests on this. A break is classified by comparing the mask
component at each end, so a labelling that is subtly wrong -- 18-connected instead
of 26, or a run table that mis-answers one row -- does not produce a visibly
broken repair. It produces a repair that confidently paints voxels across a gap
that was never there, and nothing in the output says so.

So the index is checked against ``scipy.ndimage.label`` on random volumes rather
than on hand-built cases: agreement on a shape someone chose proves the shape,
agreement on a hundred random ones proves the algorithm.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy import ndimage

from hipct_seg_debug.edit.reconnect.geodesic import components

from .conftest_geodesic import SHAPE, cylinder, decode, mask_source, slit


def _scipy_labels(mask):
    return ndimage.label(mask, structure=np.ones((3, 3, 3), dtype=int))


@pytest.mark.parametrize("seed", range(6))
def test_matches_scipy_on_random_volumes(seed):
    """The whole contract, on volumes nobody chose."""
    rng = np.random.default_rng(seed)
    mask = rng.random((18, 20, 22)) < 0.28
    index = components.from_array(mask)
    reference, n = _scipy_labels(mask)

    assert index.n == n
    # Labels need not be numbered the same way, but the *partition* must match:
    # two voxels share a label here exactly when they share one there.
    coords = np.argwhere(mask)
    mine = index.labels_at(coords)
    theirs = reference[coords[:, 0], coords[:, 1], coords[:, 2]]
    assert len(np.unique(np.stack([mine, theirs], axis=1), axis=0)) == n


def test_background_is_zero_and_sizes_are_exact():
    mask = np.zeros((10, 10, 10), dtype=bool)
    mask[2:5, 2:5, 2:5] = True
    index = components.from_array(mask)

    assert index.n == 1
    assert int(index.sizes[1]) == 27
    assert index.label_at(3, 3, 3) == 1
    assert index.label_at(8, 8, 8) == 0
    assert index.label_at(-1, 0, 0) == 0  # out of range is background, not an error


def test_connectivity_is_26_not_6():
    """A diagonal chain is one vessel, not eight beads.

    The single most consequential parameter in the module: a 6-connected labelling
    splits every obliquely-running capillary, and this package would then be asked
    to repair breaks that the labelling itself invented.
    """
    mask = np.zeros((10, 10, 10), dtype=bool)
    for i in range(8):
        mask[i, i, i] = True
    assert components.from_array(mask).n == 1


def test_window_decodes_labels_in_a_box():
    mask = np.zeros((12, 12, 12), dtype=bool)
    mask[2:4, 2:4, 1:5] = True
    mask[8:10, 8:10, 7:11] = True
    index = components.from_array(mask)

    window = index.window([0, 0, 0], [12, 12, 12])
    assert window.shape == (12, 12, 12)
    assert set(np.unique(window).tolist()) == {0, 1, 2}
    # And a sub-box holds only what it contains.
    near = index.window([2, 2, 0], [4, 4, 6])
    assert set(np.unique(near).tolist()) == {0, 1}


def test_voxels_round_trip_and_respect_the_limit():
    mask = np.zeros((10, 10, 10), dtype=bool)
    mask[3:6, 3:6, 3:6] = True
    index = components.from_array(mask)

    voxels = index.voxels(1)
    assert len(voxels) == 27
    assert set(map(tuple, voxels.tolist())) == set(map(tuple, np.argwhere(mask).tolist()))
    assert len(index.voxels(1, limit=5)) == 5


def test_nearest_label_finds_a_component_just_off_the_endpoint():
    """A centreline vertex one voxel outside its own lumen is routine, not a finding."""
    mask = np.zeros((10, 10, 10), dtype=bool)
    mask[5, 5, 5] = True
    index = components.from_array(mask)

    assert index.nearest_label(5, 5, 5, 3) == (1, 0.0)
    label, distance = index.nearest_label(5, 5, 7, 3)
    assert label == 1 and distance == pytest.approx(2.0)
    assert index.nearest_label(0, 0, 0, 2)[0] == 0


def test_streaming_build_matches_from_array_on_a_mask_source():
    """The path the CLI actually takes: a slice-at-a-time source, edits composited."""
    volume = (cylinder(SHAPE, 3, 5, 25) | cylinder(SHAPE, 3, 33, 55))
    source = mask_source(volume)
    index = components.build(source)

    assert index.n == 2
    assert int(index.sizes.sum()) == int(volume.sum())

    # Painting the gap shut must be visible to a rebuild through the same source,
    # because that is how a repair reaches every other reader in the toolkit.
    for x in range(25, 33):
        source.edits.set_plane(20, np.array([20]), np.array([x]),
                               np.array([1], dtype=np.uint8))
    assert components.build(source).n == 1
    assert decode(source)[20, 20, 28] == 1


def test_z_range_labels_only_the_slab():
    mask = np.zeros((20, 8, 8), dtype=bool)
    mask[:, 3:5, 3:5] = True  # one column through the whole volume
    whole = components.build(_Adapter(mask))
    assert whole.n == 1

    slab = components.build(_Adapter(mask), z_range=(5, 10))
    assert slab.n == 1
    assert int(slab.sizes[1]) == 5 * 4  # only the slab's voxels are counted
    assert slab.label_at(7, 3, 3) == 1
    assert slab.label_at(15, 3, 3) == 0  # outside the slab reads as background


def test_slit_of_one_voxel_thickness_is_a_single_component():
    """The collapsed case. A slit is a legitimate vessel, not a labelling accident."""
    index = components.from_array(slit(SHAPE, 6, 0, 5, 55) > 0)
    assert index.n == 1


class _Adapter:
    def __init__(self, array):
        self.array = np.asarray(array)
        self.nz, self.ny, self.nx = self.array.shape

    def slice_z(self, k):
        return self.array[k]
