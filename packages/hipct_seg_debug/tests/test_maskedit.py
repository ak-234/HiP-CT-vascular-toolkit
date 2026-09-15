"""The edit store, and the reader that composites it onto the lattice."""

from __future__ import annotations

import numpy as np
import pytest

from hipct_seg_debug.edit.maskedit import MaskEdits, MaskSource


class FakeLattice:
    """The slice-at-a-time surface ``ByteRLELattice`` presents, over a real array."""

    def __init__(self, volume):
        self.volume = np.asarray(volume, dtype=np.uint8)
        self.nz, self.ny, self.nx = self.volume.shape
        self.dims = np.array([self.nx, self.ny, self.nz], dtype=np.int64)
        self.reads = 0

    def slice_z(self, k):
        if not 0 <= k < self.nz:
            raise IndexError(k)
        self.reads += 1
        return self.volume[k].copy()

    def decode_sequential(self, n_slices):
        return self.volume[:n_slices].copy()


@pytest.fixture
def lattice():
    rng = np.random.default_rng(7)
    return FakeLattice((rng.random((8, 12, 10)) < 0.2).astype(np.uint8))


# ------------------------------------------------------------------ the store


def test_set_and_apply_round_trips():
    edits = MaskEdits(ny=12, nx=10)
    edits.set_plane(3, [4, 5], [6, 7], [1, 1])
    window = np.zeros((8, 12, 10), dtype=np.uint8)
    assert edits.apply(window, (0, 0, 0)) == 2
    assert window[3, 4, 6] == 1 and window[3, 5, 7] == 1
    assert window.sum() == 2


def test_apply_respects_a_window_offset():
    edits = MaskEdits(ny=12, nx=10)
    edits.set_plane(3, [4], [6], [1])
    window = np.zeros((2, 4, 4), dtype=np.uint8)
    assert edits.apply(window, (2, 3, 5)) == 1
    assert window[1, 1, 1] == 1

    outside = np.zeros((2, 2, 2), dtype=np.uint8)
    assert edits.apply(outside, (0, 0, 0)) == 0


def test_diff_records_only_what_differs(lattice):
    base = lattice.volume[2:5, 3:9, 1:8].copy()
    edited = base.copy()
    edited[1, 2, 3] = 1 - edited[1, 2, 3]
    edited[0, 0, 0] = 1 - edited[0, 0, 0]

    edits = MaskEdits(ny=lattice.ny, nx=lattice.nx)
    assert edits.diff(edited, base, (2, 3, 1)) == 2
    assert edits.n_voxels == 2


def test_reverting_to_the_original_empties_the_store(lattice):
    """The invariant the whole design rests on.

    napari's Ctrl+Z restores the array and emits no paint event; the store stays
    correct only because a diff that agrees with the baseline removes the entry
    rather than leaving the old one behind.
    """
    base = lattice.volume[0:4, 0:6, 0:6].copy()
    edited = base.copy()
    edited[1, 2, 3] = 1 - edited[1, 2, 3]

    edits = MaskEdits(ny=lattice.ny, nx=lattice.nx)
    edits.diff(edited, base, (0, 0, 0))
    assert edits.n_voxels == 1

    # ...and now the user undoes it.
    edits.diff(base.copy(), base, (0, 0, 0))
    assert edits.n_voxels == 0
    assert edits.is_empty


def test_diff_leaves_edits_outside_the_window_alone():
    edits = MaskEdits(ny=20, nx=20)
    edits.set_plane(9, [15], [15], [1])  # far away
    base = np.zeros((2, 4, 4), dtype=np.uint8)
    edited = base.copy()
    edited[0, 1, 1] = 1
    edits.diff(edited, base, (0, 0, 0))
    assert edits.n_voxels == 2


def test_stats_separate_painting_from_erasing():
    edits = MaskEdits(ny=8, nx=8)
    edits.set_plane(0, [1, 2, 3], [1, 2, 3], [1, 1, 0])
    assert edits.stats() == {"added": 2, "removed": 1, "total": 3, "planes": 1}


def test_clear_window_drops_only_what_is_inside():
    edits = MaskEdits(ny=20, nx=20)
    edits.set_plane(1, [2, 15], [2, 15], [1, 1])
    assert edits.clear_window((0, 0, 0), (4, 5, 5)) == 1
    assert edits.n_voxels == 1


def test_added_mask_is_new_lumen_only():
    edits = MaskEdits(ny=8, nx=8)
    base = np.zeros((2, 8, 8), dtype=np.uint8)
    base[0, 1, 1] = 1
    edits.set_plane(0, [1, 4], [1, 4], [0, 1])  # erase one, paint one
    added = edits.added_mask(base, (0, 0, 0))
    assert added[0, 4, 4]
    assert not added[0, 1, 1]
    assert added.sum() == 1


def test_bbox_spans_every_edit():
    edits = MaskEdits(ny=30, nx=30)
    edits.set_plane(2, [5], [7], [1])
    edits.set_plane(9, [20], [3], [1])
    lo, hi = edits.bbox_seg()
    assert tuple(lo) == (2, 5, 3)
    assert tuple(hi) == (9, 20, 7)
    assert MaskEdits(ny=4, nx=4).bbox_seg() is None


def test_save_load_round_trip(tmp_path):
    edits = MaskEdits(ny=12, nx=10, nz=8, source="whatever.am")
    edits.set_plane(3, [4, 5], [6, 7], [1, 0])
    edits.set_plane(6, [1], [2], [1])
    path = edits.save(tmp_path / "edits.npz")

    back = MaskEdits.load(path, expect_dims=(10, 12, 8))
    assert back.n_voxels == 3
    assert back.stats() == edits.stats()
    a = np.zeros((8, 12, 10), dtype=np.uint8)
    b = np.zeros((8, 12, 10), dtype=np.uint8)
    edits.apply(a, (0, 0, 0))
    back.apply(b, (0, 0, 0))
    assert np.array_equal(a, b)


def test_load_refuses_the_wrong_lattice(tmp_path):
    path = MaskEdits(ny=12, nx=10, nz=8).save(tmp_path / "e.npz")
    with pytest.raises(ValueError, match="lattice"):
        MaskEdits.load(path, expect_dims=(99, 12, 8))


def test_empty_store_saves_and_loads(tmp_path):
    path = MaskEdits(ny=4, nx=4, nz=2).save(tmp_path / "empty.npz")
    assert MaskEdits.load(path).is_empty


# ----------------------------------------------------------------- the reader


def test_source_composites_on_read(lattice):
    src = MaskSource(lattice)
    src.edits.set_plane(3, [4], [6], [1])
    plane = src.slice_z(3)
    assert plane[4, 6] == 1
    # Everything else is exactly the lattice, and the source is left alone.
    expected = lattice.volume[3].copy()
    expected[4, 6] = 1
    assert np.array_equal(plane, expected)
    assert np.array_equal(src.base_slice_z(3), lattice.volume[3])


def test_source_composites_onto_a_row_band_too(lattice):
    """`__getattr__` would delegate `slice_rows` to the *unedited* lattice.

    Nothing downstream would say so: the sampler would read bands straight past the
    edits, and a painted correction would appear in the viewer and not in the
    measurement.
    """
    src = MaskSource(lattice)
    src.edits.set_plane(3, [4], [6], [1])

    band = src.slice_rows(3, 2, 7)
    assert np.array_equal(band, src.slice_z(3)[2:7])
    assert band[2, 6] == 1
    assert not np.array_equal(band, lattice.volume[3, 2:7])


def test_a_row_band_uses_the_lattices_own_reader_when_there_is_nothing_to_composite():
    """With no edits the band must come from the fast path, not a whole plane."""

    class Banded(FakeLattice):
        def __init__(self, volume):
            super().__init__(volume)
            self.row_reads = 0

        def slice_rows(self, k, row0, row1):
            self.row_reads += 1
            return self.volume[k, row0:row1].copy()

    rng = np.random.default_rng(21)
    lattice = Banded((rng.random((5, 9, 7)) < 0.3).astype(np.uint8))
    src = MaskSource(lattice)

    assert np.array_equal(src.slice_rows(2, 1, 5), lattice.volume[2, 1:5])
    assert (lattice.row_reads, lattice.reads) == (1, 0)

    src.edits.set_plane(2, [1], [1], [1])
    assert src.slice_rows(2, 1, 5)[0, 1] == 1
    assert lattice.reads == 1, "with edits it falls back to the composited plane"


def test_source_is_a_drop_in_for_decode_volume(lattice):
    from hipct_seg_debug.edit.lattice import decode_volume

    src = MaskSource(lattice)
    src.edits.set_plane(2, [1], [1], [1])
    volume = decode_volume(src)
    assert volume.shape == (lattice.nz, lattice.ny, lattice.nx)
    assert volume[2, 1, 1] == 1
    expected = lattice.volume.copy()
    expected[2, 1, 1] = 1
    assert np.array_equal(volume, expected)


def test_decode_sequential_is_composited_too(lattice):
    src = MaskSource(lattice)
    src.edits.set_plane(1, [0], [0], [1])
    assert src.decode_sequential(4)[1, 0, 0] == 1


def test_window_clips_and_zero_pads(lattice):
    src = MaskSource(lattice)
    w = src.window(-2, 2, -1, 3, 0, 4)
    assert w.shape == (4, 4, 4)
    assert not w[:2].any()  # planes -2, -1 do not exist
    assert not w[:, 0].any()  # row -1 does not exist
    assert np.array_equal(w[2:, 1:, :], lattice.volume[0:2, 0:3, 0:4])


def test_window_can_ask_for_the_pristine_values(lattice):
    src = MaskSource(lattice)
    src.edits.set_plane(1, [1], [1], [1])
    edited = src.window(0, 3, 0, 4, 0, 4, edited=True)
    base = src.window(0, 3, 0, 4, 0, 4, edited=False)
    assert edited[1, 1, 1] == 1
    assert base[1, 1, 1] == lattice.volume[1, 1, 1]


def test_source_rejects_a_store_for_another_lattice(lattice):
    with pytest.raises(ValueError, match="planes"):
        MaskSource(lattice, MaskEdits(ny=99, nx=99))


def test_unknown_attributes_fall_through(lattice):
    src = MaskSource(lattice)
    assert src.reads == lattice.reads
