"""The brush: where a painted voxel lands, and whether undo reaches the store."""

from __future__ import annotations

import numpy as np
import pytest

from hipct_seg_debug.amira import LatticeInfo
from hipct_seg_debug.edit.maskedit import MaskSource
from hipct_seg_debug.edit.paint import PaintSession
from hipct_seg_debug.frame import WorldFrame
from hipct_seg_debug.volume import seg_placement, seg_window_placement

from .test_maskedit import FakeLattice

napari = pytest.importorskip("napari")


class FakeViewer:
    """Just enough ``napari.Viewer`` for :class:`PaintSession`, with no window.

    ``Labels`` layers construct perfectly well headless, and every behaviour worth
    testing here -- where a voxel lands, what a stroke does to the array, what undo
    does to the store -- lives in the layer rather than in the canvas. Using
    napari's own ``make_napari_viewer`` would drag in ``pytest-qt`` and a real GUI
    for no extra coverage.
    """

    def __init__(self):
        self.layers = []

    def add_labels(self, data, **kwargs):
        from napari.layers import Labels

        layer = Labels(data, **kwargs)
        self.layers.append(layer)
        return layer

    def add_points(self, data, **kwargs):
        from napari.layers import Points

        layer = Points(data, **kwargs)
        self.layers.append(layer)
        return layer


@pytest.fixture
def viewer():
    return FakeViewer()


# A segmentation binned 2x against the raw grid, like LADAF-2024-28.
SEG_DIMS = (10, 12, 8)  # (nx, ny, nz)
SPACING = 66.0
RAW_VOXEL = 33.0


@pytest.fixture
def frame():
    origin = np.array([SPACING, 2 * SPACING, 3 * SPACING])
    hi = origin + (np.asarray(SEG_DIMS) - 1) * SPACING
    bbox = np.empty(6)
    bbox[0::2] = origin
    bbox[1::2] = hi
    info = LatticeInfo(path=None, dims=np.asarray(SEG_DIMS), bbox=bbox, fields={})
    # Raw stack big enough to contain the lattice: (nz, nrow, ncol).
    raw_shape = (60, 60, 60)
    return WorldFrame.from_inputs(raw_shape, RAW_VOXEL, info)


@pytest.fixture
def source():
    rng = np.random.default_rng(3)
    nx, ny, nz = SEG_DIMS
    return MaskSource(FakeLattice((rng.random((nz, ny, nx)) < 0.25).astype(np.uint8)))


def test_bin_factor_is_what_the_fixture_claims(frame):
    assert tuple(frame.bin_factor) == (2, 2, 2)


# ------------------------------------------------------------------ placement


def test_window_placement_matches_the_whole_volume_placement(frame):
    """A box at the origin must place identically to the full-volume layer."""
    assert seg_window_placement(frame, (0, 0, 0)) == seg_placement(frame)


@pytest.mark.parametrize("origin_kji", [(0, 0, 0), (2, 3, 1), (5, 1, 4)])
def test_painted_voxel_lands_where_the_slab_gather_puts_it(frame, origin_kji):
    """The alignment check.

    ``viewer2d._seg_window`` maps a raw pixel to a segmentation voxel with
    ``raw_axis_to_seg_axis``; napari maps a layer index to a raw pixel with
    ``index * scale + translate``. If those two disagree by the half-voxel term in
    ``seg_placement``, every painted edit is displaced by half a raw voxel and
    still looks entirely plausible. So: place a voxel, walk the napari transform
    forward, and require the gather to come back to the same segmentation index.
    """
    scale, translate = seg_window_placement(frame, origin_kji)
    k0, j0, i0 = origin_kji

    for local in [(0, 0, 0), (1, 2, 3), (2, 1, 0)]:
        # Where napari draws this layer index, in raw (slice, row, col).
        raw = [local[a] * scale[a] + translate[a] for a in range(3)]
        # Where the slab gather says that raw pixel's mask value comes from.
        back_col = int(frame.raw_axis_to_seg_axis(np.array([raw[2]]), 0)[0])
        back_row = int(frame.raw_axis_to_seg_axis(np.array([raw[1]]), 1)[0])
        back_k = int(frame.raw_slice_to_seg_slice(np.array([raw[0]]))[0])
        assert (back_k, back_row, back_col) == (k0 + local[0], j0 + local[1], i0 + local[2])


def test_open_box_is_clipped_to_the_lattice(frame, source):
    session = PaintSession(source, frame, size_vox=999)
    box = session.open_box(frame.seg_to_um([[0, 0, 0]])[0])
    assert box.origin_kji == (0, 0, 0)
    assert box.shape_kji == (SEG_DIMS[2], SEG_DIMS[1], SEG_DIMS[0])


def test_open_box_centres_on_the_pick(frame, source):
    session = PaintSession(source, frame, size_vox=4)
    centre = frame.seg_to_um([[5, 6, 4]])[0]
    box = session.open_box(centre)
    assert box.origin_kji == (2, 4, 3)
    assert box.shape_kji == (4, 4, 4)
    assert np.array_equal(box.base, source.window(2, 6, 4, 8, 3, 7, edited=False))


def test_box_bounds_cover_every_voxel_centre(frame, source):
    session = PaintSession(source, frame, size_vox=4)
    box = session.open_box(frame.seg_to_um([[5, 6, 4]])[0])
    lo, hi = box.bounds_um(frame)
    k0, j0, i0 = box.origin_kji
    nk, nj, ni = box.shape_kji
    for kji in [(0, 0, 0), (nk - 1, nj - 1, ni - 1)]:
        xyz = frame.seg_to_um([[i0 + kji[2], j0 + kji[1], k0 + kji[0]]])[0]
        assert np.all(xyz >= lo) and np.all(xyz <= hi)


# --------------------------------------------------------------------- commit


def _session_with_layer(source, frame, viewer, centre_index=(5, 6, 4), size=6):
    session = PaintSession(source, frame, size_vox=size)
    session.attach(viewer, frame.seg_to_um([list(centre_index)])[0])
    return session


def test_paint_then_commit_reaches_the_store(viewer, frame, source):
    session = _session_with_layer(source, frame, viewer)
    layer = session.layer

    target = tuple(np.argwhere(session.box.base == 0)[0])
    layer.data_setitem(tuple(np.array([t]) for t in target), 1)
    assert session.commit() == 1
    assert source.edits.n_voxels == 1

    k0, j0, i0 = session.box.origin_kji
    assert source.slice_z(k0 + target[0])[j0 + target[1], i0 + target[2]] == 1


def test_napari_undo_empties_the_store(viewer, frame, source):
    """``Labels.undo()`` emits no paint event -- measured, not assumed.

    Anything that accumulated paint events would keep a stroke the user has
    already undone and can no longer see. Committing by diff is what makes this
    come out right, and this is the test that fails if that ever changes.
    """
    session = _session_with_layer(source, frame, viewer)
    layer = session.layer

    target = tuple(np.argwhere(session.box.base == 0)[0])
    idx = tuple(np.array([t]) for t in target)
    layer.data_setitem(idx, 1)
    session.commit()
    assert source.edits.n_voxels == 1

    layer.undo()
    assert session.commit() == 0
    assert source.edits.is_empty


def test_reopening_a_box_keeps_the_edits_visible(viewer, frame, source):
    session = _session_with_layer(source, frame, viewer)
    target = tuple(np.argwhere(session.box.base == 0)[0])
    session.layer.data_setitem(tuple(np.array([t]) for t in target), 1)
    session.commit()
    k0, j0, i0 = session.box.origin_kji
    seg_index = (k0 + target[0], j0 + target[1], i0 + target[2])

    # A new pick elsewhere, then back again: the edit must still be on screen.
    session.attach(viewer, frame.seg_to_um([[1, 1, 1]])[0])
    session.attach(viewer, frame.seg_to_um([[5, 6, 4]])[0])
    nk, nj, ni = session.box.origin_kji
    local = (seg_index[0] - nk, seg_index[1] - nj, seg_index[2] - ni)
    assert np.asarray(session.layer.data)[local] == 1
    assert source.edits.n_voxels == 1


def test_attach_reuses_the_layer(viewer, frame, source):
    session = _session_with_layer(source, frame, viewer)
    first = session.layer
    session.attach(viewer, frame.seg_to_um([[2, 2, 2]])[0])
    assert session.layer is first
    assert sum(1 for lay in viewer.layers if lay.name == first.name) == 1


def test_revert_box_restores_the_lattice(viewer, frame, source):
    session = _session_with_layer(source, frame, viewer)
    target = tuple(np.argwhere(session.box.base == 0)[0])
    session.layer.data_setitem(tuple(np.array([t]) for t in target), 1)
    session.commit()
    assert session.revert_box() == 1
    assert source.edits.is_empty
    assert np.array_equal(np.asarray(session.layer.data), session.box.base)


def test_edited_bounds_cover_the_painted_voxel(viewer, frame, source):
    session = _session_with_layer(source, frame, viewer)
    target = tuple(np.argwhere(session.box.base == 0)[0])
    session.layer.data_setitem(tuple(np.array([t]) for t in target), 1)

    bounds = session.edited_bounds_um()
    k0, j0, i0 = session.box.origin_kji
    xyz = frame.seg_to_um([[i0 + target[2], j0 + target[1], k0 + target[0]]])[0]
    assert np.all(xyz >= bounds[0]) and np.all(xyz <= bounds[1])

    session.revert_box()
    assert session.edited_bounds_um() is None


def test_save_and_reload_through_a_session(viewer, frame, source, tmp_path):
    session = _session_with_layer(source, frame, viewer)
    target = tuple(np.argwhere(session.box.base == 0)[0])
    session.layer.data_setitem(tuple(np.array([t]) for t in target), 1)

    path = session.save(tmp_path / "edits.npz")
    assert path.exists() and not session.unsaved

    from hipct_seg_debug.edit.maskedit import MaskEdits

    back = MaskEdits.load(path, expect_dims=SEG_DIMS)
    assert back.n_voxels == 1


def test_the_slice_slider_stays_on_whole_raw_slices():
    """A mask-grid layer must not shift napari's slider grid onto half-slices.

    ``seg_placement`` puts a segmentation layer at ``raw_start + (bin - 1) / 2`` --
    2101.5 on this dataset, and correctly so, because a mask voxel straddles two raw
    slices. napari builds its slider grid from the union's start, so a layer that
    sets that start makes every step land half a slice off and the browser shows a
    different slice from the one that was picked. Measured before the fix:
    ``--paint`` landed on the right slice 1 pick in 4.
    """
    from collections import namedtuple

    from hipct_seg_debug.viewer2d import _whole_slices

    R = namedtuple("RangeTuple", "start stop step")
    fixed = _whole_slices((R(2719.5, 2813.5, 1.0), R(0.5, 100.5, 1.0)))
    assert fixed[0] == R(2719.0, 2814.0, 1.0)
    assert float(fixed[0].start).is_integer()
    # Every step from the start is then a whole slice.
    assert (2767 - fixed[0].start) % fixed[0].step == 0
    # The displayed axes are continuous and are deliberately left alone.
    assert fixed[1] == R(0.5, 100.5, 1.0)


def test_whole_slices_tolerates_an_empty_viewer():
    from hipct_seg_debug.viewer2d import _whole_slices

    assert _whole_slices(()) == ()


def test_paint_session_refuses_a_bare_lattice(frame):
    with pytest.raises(TypeError, match="MaskSource"):
        PaintSession(FakeLattice(np.zeros((2, 2, 2), np.uint8)), frame)


# ------------------------------------------------------------- labels and tools


def _two_tree_source():
    """Left_Tree (1) in the low rows, Right_Tree (2) -- the majority -- in the high."""
    nx, ny, nz = SEG_DIMS
    vol = np.zeros((nz, ny, nx), dtype=np.uint8)
    vol[:, :3, :] = 1
    vol[:, 6:, :] = 2
    return MaskSource(FakeLattice(vol))


def _materials():
    from hipct_seg_debug.amira import Material

    return (Material("Exterior", 0), Material("Left_Tree", 1), Material("Right_Tree", 2))


def test_every_label_is_coloured_and_the_brush_takes_the_box_majority(viewer, frame):
    session = PaintSession(_two_tree_source(), frame, size_vox=999, materials=_materials())
    session.attach(viewer, frame.seg_to_um([[5, 6, 4]])[0])
    colours = session.layer.colormap.color_dict
    assert 1 in colours and 2 in colours
    assert session.layer.selected_label == 2
    assert session.label_choices() == [("Left_Tree", 1), ("Right_Tree", 2)]
    assert "Right_Tree" in session.describe()
    assert session.layer.preserve_labels


def test_a_pinned_label_survives_a_new_box_and_napari_s_picker_pins(viewer, frame):
    session = PaintSession(_two_tree_source(), frame, size_vox=999, materials=_materials())
    session.attach(viewer, frame.seg_to_um([[5, 6, 4]])[0])
    session.set_label(1)
    session.attach(viewer, frame.seg_to_um([[2, 2, 2]])[0])
    assert session.layer.selected_label == 1
    session.set_label("auto")
    assert session.layer.selected_label == 2
    session.layer.selected_label = 1  # what napari's own label picker does
    assert session.label_mode == 1


def test_attach_adds_each_helper_layer_once(viewer, frame, source):
    from hipct_seg_debug.edit.paint import layer_names

    session = _session_with_layer(source, frame, viewer)
    session.attach(viewer, frame.seg_to_um([[2, 2, 2]])[0])
    names = [lay.name for lay in viewer.layers]
    assert sorted(names) == sorted(layer_names())


def _tool_session(viewer, mask, raw, *, size=999):
    from hipct_seg_debug.edit.mask_repair import ArraySampler

    from .conftest_geometry import make_frame

    frame = make_frame(mask.shape)
    source = MaskSource(FakeLattice(mask))
    session = PaintSession(source, frame, size_vox=size, materials=_materials(),
                           sampler=ArraySampler(raw))
    centre = np.asarray(mask.shape)[::-1] // 2
    session.attach(viewer, frame.seg_to_um([centre])[0])
    assert session.box.origin_kji == (0, 0, 0)
    return session, source


def _seed(session, group, kji_points):
    session.set_seed_group(group)
    for kji in kji_points:
        session.seeds_layer.add(session.layer.data_to_world(np.asarray(kji, float)))


def test_fill_writes_the_tree_s_label_as_one_undo_step(viewer):
    from .test_mask_tools import NOTCH_SEEDS, notched_tube, render

    mask, lumen, missing = notched_tube(value=2)
    session, source = _tool_session(viewer, mask, render(lumen))
    _seed(session, 1, NOTCH_SEEDS)
    result = session.run_fill()
    assert result is not None and result.ok, session.status
    assert result.value == 2
    k, row, col, val = source.edits.entries()
    assert len(val) == int(result.write.sum()) and set(val.tolist()) == {2}
    assert (np.asarray(session.result_layer.data) == 1).sum() == len(val)

    session.layer.undo()  # one Ctrl+Z takes the whole fill back
    assert session.commit() == 0
    assert source.edits.is_empty


def test_split_stores_only_erasures_and_switches_to_replace(viewer):
    from .test_mask_tools import GROUPS, fused_pair

    mask, raw, *_ = fused_pair()
    session, source = _tool_session(viewer, mask, raw)
    _seed(session, 1, GROUPS[0])
    _seed(session, 2, GROUPS[1])
    result = session.run_split()
    assert result is not None and result.ok, session.status
    assert source.edits.stats()["added"] == 0
    assert source.edits.stats()["removed"] == int(result.write.sum())
    assert session.splice_mode == "replace"


def test_tools_explain_missing_seeds(viewer):
    from .test_mask_tools import notched_tube, render

    mask, lumen, _ = notched_tube()
    session, source = _tool_session(viewer, mask, render(lumen))
    assert session.run_fill() is None
    assert "seeds" in session.status
    assert session.run_split() is None
    assert source.edits.is_empty


def test_relabel_strokes_reloads_the_layer(viewer, frame):
    source = _two_tree_source()
    session = PaintSession(source, frame, size_vox=999, materials=_materials())
    session.attach(viewer, frame.seg_to_um([[5, 6, 4]])[0])
    # A stroke with the old fixed label, right beside Right_Tree.
    session.layer.data_setitem((np.array([2]), np.array([5]), np.array([4])), 1)
    session.commit()
    counts = session.relabel_strokes()
    assert counts["relabelled"] == 1
    assert np.asarray(session.layer.data)[2, 5, 4] == 2
    assert session.commit() == 1  # and the reload is what the next diff sees
    assert source.slice_z(2)[5, 4] == 2
