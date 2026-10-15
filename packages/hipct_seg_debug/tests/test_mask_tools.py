"""Seeded mask repairs: fill a collapsed notch, split a fused pair, relabel strokes.

Each tool is tested against the defect it exists to fix *and* against the failure it
must refuse, because the refusal is the half nobody notices when it breaks: a fill
that spills into an unrelated dark region, or into the other tree, looks like a
successful fill until someone skeletonises it.
"""
import numpy as np
import pytest
from scipy import ndimage

from hipct_seg_debug.edit import mask_repair as mr
from hipct_seg_debug.edit import mask_tools as mt
from hipct_seg_debug.edit.maskedit import MaskSource

from .conftest_geometry import cylinder as _cylinder
from .test_maskedit import FakeLattice

pytest.importorskip("skimage")

TISSUE, WALL, LUMEN = 200.0, 230.0, 30.0
SHAPE = (32, 40, 80)
CZ, CY = 16, 20


def cylinder(*args, **kw):
    return _cylinder(*args, **kw).astype(bool)


def betti(vol):
    f = np.pad(np.asarray(vol) > 0, 1)
    b0, b2, chi = mr._betti(f)
    return b0, b0 + b2 - chi, b2


def render(lumen, rng_seed=0, noise=6.0):
    """Tissue, a bright two-voxel wall around every lumen, and dark lumen."""
    raw = np.full(lumen.shape, TISSUE)
    wall = ndimage.binary_dilation(lumen, mt.CONN26, iterations=2) & ~lumen
    raw[wall] = WALL
    raw[lumen] = LUMEN
    rng = np.random.default_rng(rng_seed)
    return (raw + rng.normal(0.0, noise, raw.shape)).astype(np.float32)


def notched_tube(value=1):
    """A tube whose middle has collapsed to a slit, half of which the mask missed.

    Outside x in [30, 50) the lumen is a round tube of radius 5; inside, it is a slit
    11 voxels wide and 3 thick. The mask covers the tube and only the y < CY half of
    the slit -- a notch open on one side, which ``repair-mask --guarded`` never
    proposes because it is not a hole in any slice.
    """
    zz, yy, xx = np.ogrid[:SHAPE[0], :SHAPE[1], :SHAPE[2]]
    collapse = (xx >= 30) & (xx < 50)
    tube = cylinder(SHAPE, 5, 4, 76, cy=CY, cz=CZ) & ~collapse
    slit = (np.abs(yy - CY) <= 5) & (np.abs(zz - CZ) <= 1) & collapse
    lumen = tube | slit
    mask = np.zeros(SHAPE, dtype=np.uint8)
    mask[tube | (slit & (yy < CY))] = value
    missing = slit & (yy >= CY)
    return mask, lumen, missing


NOTCH_SEEDS = [(CZ, CY + 3, 32), (CZ, CY + 3, 47)]


# --------------------------------------------------------------------- the fill


@pytest.mark.parametrize("value", [1, 2])
def test_fill_closes_a_collapsed_notch_with_the_tree_s_own_label(value):
    mask, lumen, missing = notched_tube(value)
    result = mt.fill_collapsed(mask, render(lumen), NOTCH_SEEDS)
    assert result.ok, result.describe()
    assert result.value == value
    got = result.write
    assert (got & missing).sum() >= 0.8 * missing.sum()
    assert not (got & (mask > 0)).any()
    capsule = mt.seeds_to_roi(SHAPE, NOTCH_SEEDS, result.stats["roi_radius"])
    assert not (got & ~capsule).any()
    after = mask.copy()
    after[got] = value
    assert betti(after)[1] <= betti(mask)[1]


def test_without_thickening_the_fill_stays_on_the_lumen():
    mask, lumen, _ = notched_tube()
    result = mt.fill_collapsed(mask, render(lumen), NOTCH_SEEDS, thicken=0)
    assert result.ok
    near = ndimage.binary_dilation(lumen, mt.CONN26)
    assert not (result.write & ~near).any()


def test_a_dark_region_behind_a_wall_is_not_filled():
    mask, lumen, missing = notched_tube()
    zz, yy, xx = np.ogrid[:SHAPE[0], :SHAPE[1], :SHAPE[2]]
    # Dark, but behind the slit's own two-voxel bright wall: not reachable.
    blob = (zz >= CZ + 5) & (zz < CZ + 9) & (np.abs(yy - CY) <= 4) & (xx >= 32) & (xx < 46)
    result = mt.fill_collapsed(mask, render(lumen | blob), NOTCH_SEEDS)
    assert not (result.write & blob).any()
    assert (result.write & missing).sum() >= 0.8 * missing.sum()


def test_a_dark_region_running_out_of_the_capsule_is_refused_as_a_leak():
    mask, lumen, missing = notched_tube()
    zz, yy, xx = np.ogrid[:SHAPE[0], :SHAPE[1], :SHAPE[2]]
    # Touching the missing slit and running off to the volume's edge: a chamber,
    # not a slit, and exactly what a brightness criterion alone would pour into.
    blob = (zz >= CZ + 1) & (yy >= CY) & (yy < CY + 15) & (xx >= 28) & (xx < 52)
    result = mt.fill_collapsed(mask, render(lumen | blob), NOTCH_SEEDS)
    assert not (result.write & blob).any()
    assert result.refused.any()
    assert any("leak" in r for r in result.reasons)


def test_a_fill_that_would_join_the_other_tree_is_refused():
    zz, yy, xx = np.ogrid[:SHAPE[0], :SHAPE[1], :SHAPE[2]]
    left = cylinder(SHAPE, 4, 4, 76, cy=12, cz=CZ)
    right = cylinder(SHAPE, 4, 4, 76, cy=28, cz=CZ)
    bridge = (np.abs(zz - CZ) <= 1) & (yy > 16) & (yy < 24) & (xx >= 35) & (xx < 45)
    mask = np.zeros(SHAPE, dtype=np.uint8)
    mask[left] = 1
    mask[right] = 2
    seeds = [(CZ, 20, 37), (CZ, 20, 43)]
    result = mt.fill_collapsed(mask, render(left | right | bridge), seeds,
                               names={1: "Left_Tree", 2: "Right_Tree"})
    assert not result.ok
    assert any("would merge with" in r and "Tree" in r for r in result.reasons)


def test_no_contrast_means_no_fill():
    mask, lumen, _ = notched_tube()
    flat = np.full(SHAPE, TISSUE, dtype=np.float32)
    result = mt.fill_collapsed(mask, flat, NOTCH_SEEDS)
    assert not result.ok
    assert any("contrast" in r for r in result.reasons)


def test_the_volume_cap_refuses_everything():
    mask, lumen, _ = notched_tube()
    result = mt.fill_collapsed(mask, render(lumen), NOTCH_SEEDS, max_vox=5)
    assert not result.ok
    assert result.refused.any()
    assert any("cap" in r for r in result.reasons)


def test_fill_needs_seeds_and_raw():
    mask, lumen, _ = notched_tube()
    assert not mt.fill_collapsed(mask, render(lumen), []).ok
    assert not mt.fill_collapsed(mask, None, NOTCH_SEEDS).ok


def test_seeds_to_roi_is_a_capsule():
    roi = mt.seeds_to_roi((20, 20, 40), [(10, 10, 5), (10, 10, 30)], 3)
    assert roi[10, 10, 5:31].all()
    assert roi[10, 13, 20] and not roi[10, 14, 20]
    assert not roi[10, 10, 34]
    ball = mt.seeds_to_roi((9, 9, 9), [(4, 4, 4)], 2)
    assert ball.sum() == mt._ball(2).sum()


# -------------------------------------------------------------------- the split


def fused_pair():
    """Two parallel tubes joined by a thin band -- a fusion across a wall."""
    zz, yy, xx = np.ogrid[:SHAPE[0], :SHAPE[1], :SHAPE[2]]
    a = cylinder(SHAPE, 5, 4, 76, cy=12, cz=CZ)
    b = cylinder(SHAPE, 5, 4, 76, cy=28, cz=CZ)
    band = (np.abs(zz - CZ) <= 2) & (yy >= 12) & (yy <= 28) & (xx >= 35) & (xx < 46)
    mask = (a | b | band).astype(np.uint8)
    raw = render(a | b)  # the band is wall: bright, segmented by mistake
    return mask, raw, a, b, band


GROUPS = [[(CZ, 12, 10), (CZ, 12, 70)], [(CZ, 28, 10), (CZ, 28, 70)]]


@pytest.mark.parametrize("with_raw", [True, False])
def test_split_separates_two_fused_tubes_along_the_band(with_raw):
    mask, raw, a, b, band = fused_pair()
    assert ndimage.label(mask > 0, mt.CONN26)[1] == 1
    result = mt.split_fused(mask, raw if with_raw else None, GROUPS)
    assert result.ok, result.describe()
    cut = result.write
    after, n = ndimage.label((mask > 0) & ~cut, mt.CONN26)
    assert n >= 2
    ca = set(np.unique(after[a & ~cut])) - {0}
    cb = set(np.unique(after[b & ~cut])) - {0}
    assert len(ca) == 1 and len(cb) == 1 and ca != cb
    assert (cut & (a | b)).sum() <= 0.1 * cut.sum()
    assert cut.sum() <= 1.5 * band[:, 19:22].sum() * 2  # about one layer across the band


def test_split_refuses_to_cut_a_healthy_tube():
    tube = cylinder(SHAPE, 5, 4, 76, cy=CY, cz=CZ)
    mask = tube.astype(np.uint8)
    groups = [[(CZ, CY, 10), (CZ, CY, 30)], [(CZ, CY, 50), (CZ, CY, 70)]]
    result = mt.split_fused(mask, render(tube), groups)
    assert not result.ok
    assert result.refused.any()
    assert any("neck" in r for r in result.reasons)
    forced = mt.split_fused(mask, render(tube), groups, force=True)
    assert forced.ok


def test_split_reports_seeds_that_are_already_apart():
    a = cylinder(SHAPE, 5, 4, 76, cy=12, cz=CZ)
    b = cylinder(SHAPE, 5, 4, 76, cy=28, cz=CZ)
    mask = (a | b).astype(np.uint8)
    result = mt.split_fused(mask, render(a | b), GROUPS)
    assert not result.ok
    assert any("already" in r for r in result.reasons)


def test_split_needs_two_groups_on_the_mask():
    mask, raw, *_ = fused_pair()
    assert not mt.split_fused(mask, raw, GROUPS[:1]).ok
    off = [[(0, 0, 0)], GROUPS[1]]
    assert not mt.split_fused(mask, raw, off).ok


def test_cut_between_leaves_no_26_adjacency():
    rng = np.random.default_rng(5)
    basins = rng.integers(0, 4, size=(12, 13, 14))
    elevation = rng.random(basins.shape)
    cut = mt.cut_between(basins, elevation)
    left = np.where(cut, 0, basins)
    for a in (1, 2, 3):
        grown = ndimage.binary_dilation(left == a, mt.CONN26)
        others = (left > 0) & (left != a)
        assert not (grown & others).any()


# ------------------------------------------------------------------ relabelling


def test_relabel_gives_strokes_their_tree_and_restores_overwrites():
    base = np.zeros((6, 10, 12), dtype=np.uint8)
    base[3, 2:4, 2:10] = 1  # left
    base[3, 7:9, 2:10] = 2  # right
    source = MaskSource(FakeLattice(base))
    e = source.edits
    e.set_plane(3, [6], [5], [1])  # painted next to Right with the old fixed label
    e.set_plane(3, [7], [6], [1])  # painted *over* Right: should give it back
    e.set_plane(3, [4], [5], [1])  # beside Left: already right
    e.set_plane(3, [5], [11], [0])  # an erasure: not ours to touch
    e.set_plane(0, [0], [0], [1])  # nowhere near anything within reach=2
    counts = mt.relabel_added(source, reach=2)
    assert counts == {"relabelled": 1, "restored": 1, "isolated": 1}
    plane = source.slice_z(3)
    assert plane[6, 5] == 2
    assert plane[7, 6] == 2
    assert plane[4, 5] == 1
    k, row, col, val = e.entries()
    assert (3, 7, 6) not in set(zip(k.tolist(), row.tolist(), col.tolist()))


def test_stack_sampler_box_matches_its_own_point_sampler():
    """``box`` must pick the same bin-centre pixel as ``__call__``, and NaN off the stack."""
    from types import SimpleNamespace

    rng = np.random.default_rng(2)
    raw = rng.integers(0, 60000, size=(20, 22, 24)).astype(np.uint16)

    class Stack:
        n_slices, n_rows, n_cols = raw.shape

        def read_window(self, z, r0, r1, c0, c1):
            out = np.zeros((r1 - r0, c1 - c0), dtype=raw.dtype)
            rr0, rr1, cc0, cc1 = max(r0, 0), min(r1, 22), max(c0, 0), min(c1, 24)
            out[rr0 - r0:rr1 - r0, cc0 - c0:cc1 - c0] = raw[z, rr0:rr1, cc0:cc1]
            return out

        def _read_image(self, f):
            return raw[f]

        files = list(range(20))

    frame = SimpleNamespace(bin_factor=np.array([2, 2, 2]), raw_start=np.array([1, 0, 1]))
    sampler = mr.StackSampler(Stack(), frame, threads=2)
    got = sampler.box((5, 6, 7), (6, 7, 8))  # runs off the far faces
    zyx = np.argwhere(np.ones((6, 7, 8), bool)) + np.array([5, 6, 7])
    r = sampler.raw_index(zyx)
    inside = np.all((r >= 0) & (r < np.array(raw.shape)), axis=1)
    expect = np.full(len(zyx), np.nan)
    expect[inside] = sampler(zyx[inside])
    assert np.array_equal(got.ravel(), expect.astype(np.float32), equal_nan=True)
    assert np.isnan(got).any() and np.isfinite(got).any()
