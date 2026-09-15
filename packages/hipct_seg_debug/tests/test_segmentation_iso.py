"""The two isosurface routes must agree, and the stride must be changeable live.

`segmentation_volume` and `segmentation_box` each grew a `volume=` parameter so they
can read the resident mask instead of decoding. That is only safe if the two paths
produce the same mesh, which is what most of this file asserts — on an array small
enough that both run in microseconds.

The contour method changed at the same time, from `vtkContourFilter` to flying edges,
which is the change that made full resolution affordable (19.3 s → 3.0 s on the real
lattice). `test_flying_edges_matches_the_default_contour` pins that it is a speed
change and not a geometry change.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

pv = pytest.importorskip("pyvista")

from hipct_seg_debug.viewer3d import (  # noqa: E402
    Picker3D,
    segmentation_box,
    segmentation_volume,
)
from .test_maskedit import FakeLattice  # noqa: E402

SPACING = (2.0, 2.0, 2.0)


@pytest.fixture
def blob():
    """A solid cube inside a bigger box, so there is a real surface to find."""
    vol = np.zeros((16, 20, 24), dtype=np.uint8)
    vol[4:12, 6:16, 8:20] = 1
    return vol


@pytest.fixture
def frame(blob):
    nz, ny, nx = blob.shape
    return SimpleNamespace(
        seg_dims=(nx, ny, nz),
        seg_spacing=SPACING,
        seg_to_um=lambda idx: np.asarray(idx, dtype=float) * np.asarray(SPACING),
        um_to_seg_index=lambda xyz: (np.asarray(xyz, dtype=float)
                                     / np.asarray(SPACING)).astype(int).reshape(1, 3),
        um_to_raw_index=lambda xyz: (np.asarray(xyz, dtype=float)
                                     / np.asarray(SPACING)).astype(int).reshape(1, 3),
    )


# ------------------------------------------------------- the two routes agree


@pytest.mark.parametrize("stride", [1, 2, 4])
def test_whole_tree_from_the_array_matches_the_lattice(frame, blob, stride):
    lattice = FakeLattice(blob)
    _g1, from_lattice = segmentation_volume(frame, lattice, stride)
    _g2, from_array = segmentation_volume(frame, None, stride, volume=blob)
    assert from_lattice.n_cells == from_array.n_cells
    assert np.allclose(from_lattice.bounds, from_array.bounds)


def test_the_box_from_the_array_matches_the_lattice(frame, blob):
    lattice = FakeLattice(blob)
    centre = [16.0, 20.0, 16.0]
    _g1, from_lattice = segmentation_box(frame, lattice, centre, 8.0)
    _g2, from_array = segmentation_box(frame, None, centre, 8.0, volume=blob)
    assert from_lattice.n_cells == from_array.n_cells
    assert np.allclose(from_lattice.bounds, from_array.bounds)


def test_the_array_route_never_touches_the_lattice(frame, blob):
    """Passing labels=None proves it: the lattice route would raise."""
    lattice = FakeLattice(blob)
    segmentation_volume(frame, lattice, 1, volume=blob)
    assert lattice.reads == 0


def test_the_lattice_route_still_works_without_a_resident_volume(frame, blob):
    lattice = FakeLattice(blob)
    _grid, surf = segmentation_volume(frame, lattice, 1)
    assert surf is not None and surf.n_cells > 0
    assert lattice.reads == blob.shape[0]


def test_the_box_decodes_only_the_planes_it_needs(frame, blob):
    lattice = FakeLattice(blob)
    segmentation_box(frame, lattice, [16.0, 20.0, 16.0], 8.0)
    assert 0 < lattice.reads < blob.shape[0]


def test_an_empty_mask_yields_no_surface(frame):
    empty = np.zeros((16, 20, 24), dtype=np.uint8)
    grid, surf = segmentation_volume(frame, None, 1, volume=empty)
    assert surf is None and grid is not None


def test_striding_scales_the_spacing(frame, blob):
    """Otherwise the decimated mesh would sit in a quarter of the right space."""
    g1, _ = segmentation_volume(frame, None, 1, volume=blob)
    g2, _ = segmentation_volume(frame, None, 2, volume=blob)
    assert np.allclose(np.asarray(g2.spacing), np.asarray(g1.spacing) * 2)


def test_a_strided_view_is_handled_without_a_contiguity_error(frame, blob):
    """`volume[::2, ::2, ::2]` is not contiguous; ravel() on it must still be right."""
    _grid, surf = segmentation_volume(frame, None, 2, volume=blob)
    assert surf is not None and surf.n_cells > 0


# ----------------------------------------------------------- flying edges


def test_flying_edges_matches_the_default_contour(frame, blob):
    """A speed change, not a geometry change. Verified on the real lattice too."""
    grid = pv.ImageData(dimensions=(blob.shape[2], blob.shape[1], blob.shape[0]),
                        spacing=SPACING, origin=(0.0, 0.0, 0.0))
    grid.point_data["mask"] = blob.ravel()
    default = grid.contour([0.5], scalars="mask")
    fast = grid.contour([0.5], scalars="mask", method="flying_edges")
    assert default.n_cells == fast.n_cells
    assert np.allclose(default.bounds, fast.bounds)


# ------------------------------------------------------ the runtime control


def _picker(**kw):
    return Picker3D(plotter_factory=lambda title: pv.Plotter(off_screen=True), **kw)


def test_the_default_is_full_resolution():
    assert _picker().seg_stride == 1


def test_set_seg_stride_drops_the_cached_mesh():
    picker = _picker()
    picker._seg_all_mesh = object()
    picker.set_seg_stride(4)
    assert picker.seg_stride == 4 and picker._seg_all_mesh is None


def test_setting_the_same_stride_keeps_the_mesh():
    """A spin box emits a change per intermediate value; 4 -> 1 must not contour at 3."""
    picker = _picker()
    picker.set_seg_stride(4)
    sentinel = object()
    picker._seg_all_mesh = sentinel
    picker.set_seg_stride(4)
    assert picker._seg_all_mesh is sentinel


def test_a_stride_below_one_is_clamped():
    picker = _picker()
    picker.set_seg_stride(0)
    assert picker.seg_stride == 1


def test_set_seg_stride_does_not_rebuild_on_its_own():
    """Deferred deliberately -- the rebuild is a button, not a side effect."""
    picker = _picker(labels=FakeLattice(np.zeros((4, 4, 4), dtype=np.uint8)))
    picker.build()
    picker.set_seg_stride(2)
    assert picker._seg_all_mesh is None and not picker._seg_all_on


# ------------------------------------------------------------- staleness


class _Edits:
    def __init__(self, version=0):
        self.version = version


def test_a_mesh_built_before_an_edit_reads_as_stale():
    picker = _picker()
    picker.mask = SimpleNamespace(edits=_Edits(0), peek=lambda: None)
    picker._seg_all_mesh = object()
    picker._seg_all_version = 0
    assert not picker.segmentation_all_stale()
    picker.mask.edits.version = 3
    assert picker.segmentation_all_stale()


def test_nothing_is_stale_without_a_mesh():
    picker = _picker()
    picker.mask = SimpleNamespace(edits=_Edits(9), peek=lambda: None)
    assert not picker.segmentation_all_stale()


def test_nothing_is_stale_without_an_edit_store():
    picker = _picker()
    picker._seg_all_mesh = object()
    assert not picker.segmentation_all_stale()


def test_the_status_line_says_when_the_mesh_is_older_than_the_edits():
    from .test_viewer3d_swap import FakeGraph

    picker = _picker(graph=FakeGraph())
    picker.mask = SimpleNamespace(edits=_Edits(2), peek=lambda: None)
    picker._seg_all_mesh = object()
    picker._seg_all_version = 0
    assert "older than your edits" in picker._status()
    picker._seg_all_version = 2
    assert "older than your edits" not in picker._status()


# ------------------------------------------------------------- the swap


def test_the_mask_follows_a_dataset_swap():
    picker = _picker()
    picker.build()
    first = object()
    picker.set_dataset(mask=first)
    assert picker.mask is first
    picker.set_dataset(mask=None)
    assert picker.mask is None


def test_an_unnamed_mask_is_left_alone():
    picker = _picker(mask="keep me")
    picker.build()
    picker.set_dataset(graph=None)
    assert picker.mask == "keep me"


def test_a_swap_resets_the_mesh_version():
    picker = _picker()
    picker.build()
    picker._seg_all_version = 7
    picker.set_dataset(mask=None)
    assert picker._seg_all_version == 0


# ----------------------------------------------- picking survives a bad box


def test_a_failing_segmentation_box_does_not_break_picking(frame, blob):
    """It runs inside the pick handler; an escape would kill every later pick."""
    picker = _picker(frame=frame, labels=FakeLattice(blob))
    picker.build()
    picker._seg_on = True
    picker.mask = SimpleNamespace(peek=lambda: (_ for _ in ()).throw(RuntimeError("boom")))
    picker._set_pick([16.0, 20.0, 16.0])  # must not raise
    assert not picker._seg_on
    assert "segmentation box failed" in picker._note
