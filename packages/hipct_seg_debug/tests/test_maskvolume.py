"""The resident mask, and whether a painted correction reaches it.

A volume decoded once from a `MaskSource` is stale the moment a stroke is committed,
and the failure is silent — the isosurface simply shows the mask as it was. These
tests pin the repair, using the `reads` counter `FakeLattice` already carries, which
is what turns "it is correct" into "it is correct *and* it re-read exactly one plane".

`test_removed_edit_returns_to_base` is the one that shaped the design: a scheme built
on `touched_planes` alone passes every other test here and fails that one, because a
plane whose edits were all deleted is no longer in `touched_planes` at all.
"""

from __future__ import annotations

import numpy as np
import pytest

from hipct_seg_debug.edit.lattice import MaskVolume, decode_foreground_crop
from hipct_seg_debug.edit.maskedit import MaskEdits, MaskSource
from .test_maskedit import FakeLattice


@pytest.fixture
def lattice():
    rng = np.random.default_rng(11)
    return FakeLattice((rng.random((8, 12, 10)) < 0.2).astype(np.uint8))


@pytest.fixture
def source(lattice):
    return MaskSource(lattice, MaskEdits(ny=lattice.ny, nx=lattice.nx, nz=lattice.nz))


# ------------------------------------------------------------------ decoding


def test_it_decodes_once(lattice):
    mv = MaskVolume(lattice)
    mv.get()
    mv.get()
    assert lattice.reads == lattice.nz


def test_the_array_matches_the_lattice_plane_by_plane(lattice):
    mv = MaskVolume(lattice)
    array = mv.get()
    for k in range(lattice.nz):
        assert np.array_equal(array[k], lattice.volume[k])


def test_foreground_crop_preserves_values_and_reports_its_sampled_origin():
    volume = np.zeros((10, 12, 14), dtype=np.uint8)
    volume[3:6, 4:8, 5:10] = 255
    lattice = FakeLattice(volume)

    crop, origin = decode_foreground_crop(lattice, padding=1)

    assert origin == (2, 3, 4)
    assert crop.shape == (5, 6, 7)
    assert np.array_equal(crop, volume[2:7, 3:9, 4:11])
    assert np.count_nonzero(crop) == np.count_nonzero(volume)


def test_foreground_crop_origin_is_on_the_decimated_grid():
    volume = np.zeros((13, 15, 17), dtype=np.uint8)
    volume[4:11, 6:13, 8:15] = 1
    lattice = FakeLattice(volume)

    crop, origin = decode_foreground_crop(lattice, stride=2, padding=0)
    expected = volume[::2, ::2, ::2]
    nonzero = np.argwhere(expected)
    lo, hi = nonzero.min(axis=0), nonzero.max(axis=0) + 1

    assert origin == tuple(lo)
    assert np.array_equal(crop, expected[tuple(slice(a, b) for a, b in zip(lo, hi))])


def test_nothing_is_decoded_until_asked(lattice):
    MaskVolume(lattice)
    assert lattice.reads == 0


def test_peek_never_triggers_a_decode(lattice):
    mv = MaskVolume(lattice)
    assert mv.peek() is None
    assert lattice.reads == 0
    mv.get()
    assert mv.peek() is not None


def test_release_frees_and_a_later_get_redecodes(lattice):
    mv = MaskVolume(lattice)
    mv.get()
    mv.release()
    assert not mv.ready and mv.nbytes == 0
    mv.get()
    assert lattice.reads == 2 * lattice.nz


def test_describe_says_whether_it_is_resident(lattice):
    mv = MaskVolume(lattice)
    assert "not resident" in mv.describe()
    mv.get()
    assert "resident" in mv.describe() and "GB" in mv.describe()


def test_slice_z_is_read_only(lattice):
    """It is a view onto the shared array; a caller may believe it owns the result."""
    mv = MaskVolume(lattice)
    plane = mv.slice_z(3)
    assert np.array_equal(plane, lattice.volume[3])
    with pytest.raises(ValueError):
        plane[0, 0] = 1


def test_a_bare_lattice_has_no_edit_store(lattice):
    mv = MaskVolume(lattice)
    mv.get()
    assert mv.edits is None
    assert mv.refresh() == 0


# --------------------------------------------------------------- invalidation


def test_an_edit_after_the_decode_is_visible(source, lattice):
    mv = MaskVolume(source)
    mv.get()
    before = lattice.reads

    source.edits.set_plane(3, [4], [6], [1])
    assert mv.get()[3, 4, 6] == 1
    assert lattice.reads == before + 1, "only the touched plane may be re-read"


def test_removed_edit_returns_to_base(source, lattice):
    """The case a `touched_planes`-only scheme gets wrong.

    Once the edits on a plane are all deleted the plane is not "touched" any more,
    so a reader that iterates `touched_planes` never re-reads it and the resident
    copy keeps showing a correction that no longer exists.
    """
    base = lattice.volume[3, 4, 6]
    source.edits.set_plane(3, [4], [6], [1 - base])
    mv = MaskVolume(source)
    assert mv.get()[3, 4, 6] == 1 - base

    source.edits.clear_window((3, 0, 0), (1, lattice.ny, lattice.nx))
    assert mv.get()[3, 4, 6] == base
    assert np.array_equal(mv.get()[3], lattice.volume[3])


def test_clear_restores_every_plane(source, lattice):
    source.edits.set_plane(2, [1], [1], [1])
    source.edits.set_plane(5, [2], [2], [1])
    mv = MaskVolume(source)
    mv.get()

    source.edits.clear()
    array = mv.get()
    assert np.array_equal(array, lattice.volume)


def test_a_diff_that_adds_and_removes_agrees_with_the_source(source, lattice):
    """One commit that undoes one voxel and paints another, as painting really does."""
    mv = MaskVolume(source)
    mv.get()

    origin = (2, 0, 0)
    window = lattice.volume[2:4].copy()
    edited = window.copy()
    edited[0, 3, 3] = 1 - edited[0, 3, 3]
    source.edits.diff(edited, window, origin)
    array = mv.get()
    assert np.array_equal(array[2], source.slice_z(2))

    # Now undo it in a second commit, which drops the entry entirely.
    source.edits.diff(window, window, origin)
    array = mv.get()
    for k in range(lattice.nz):
        assert np.array_equal(array[k], source.slice_z(k)), k


def test_a_no_op_commit_is_free(source, lattice):
    """`PaintSession.commit()` runs on every pick. This is why a pull is viable."""
    mv = MaskVolume(source)
    mv.get()
    before = lattice.reads

    window = lattice.volume[2:4].copy()
    for _ in range(5):
        source.edits.diff(window, window, (2, 0, 0))
        mv.get()
    assert lattice.reads == before, "an unchanged commit must not cause any re-read"


def test_only_the_changed_planes_are_re_read(source, lattice):
    mv = MaskVolume(source)
    mv.get()
    before = lattice.reads

    source.edits.set_plane(1, [1], [1], [1])
    source.edits.set_plane(6, [2], [2], [1])
    mv.get()
    assert lattice.reads == before + 2


def test_repeated_gets_after_one_edit_repair_once(source, lattice):
    mv = MaskVolume(source)
    mv.get()
    source.edits.set_plane(4, [1], [1], [1])
    mv.get()
    before = lattice.reads
    mv.get()
    mv.get()
    assert lattice.reads == before


def test_peek_also_repairs(source, lattice):
    mv = MaskVolume(source)
    mv.get()
    source.edits.set_plane(3, [4], [6], [1])
    assert mv.peek()[3, 4, 6] == 1


def test_refresh_reports_how_many_planes_it_repaired(source):
    mv = MaskVolume(source)
    mv.get()
    source.edits.set_plane(1, [1], [1], [1])
    source.edits.set_plane(2, [1], [1], [1])
    assert mv.refresh() == 2
    assert mv.refresh() == 0


def test_a_different_store_forces_a_full_redecode(lattice):
    """`MaskEdits.load` returns a new object rather than mutating one."""
    source = MaskSource(lattice, MaskEdits(ny=lattice.ny, nx=lattice.nx, nz=lattice.nz))
    mv = MaskVolume(source)
    mv.get()
    source.edits = MaskEdits(ny=lattice.ny, nx=lattice.nx, nz=lattice.nz)
    assert mv.refresh() == 0
    assert not mv.ready, "it must re-decode rather than trust the old version number"


def test_an_edit_outside_the_array_is_ignored(source, lattice):
    """A store built for a taller lattice must not index past the end."""
    mv = MaskVolume(source)
    mv.get()
    source.edits.set_plane(lattice.nz + 5, [1], [1], [1])
    mv.get()  # must not raise


# ------------------------------------------------------- the version counter


def test_the_version_bumps_only_on_a_real_change(lattice):
    edits = MaskEdits(ny=lattice.ny, nx=lattice.nx, nz=lattice.nz)
    assert edits.version == 0

    edits.set_plane(1, [1], [1], [1])
    assert edits.version == 1

    assert edits.clear_window((7, 0, 0), (1, lattice.ny, lattice.nx)) == 0
    assert edits.version == 1, "a window with nothing in it must not bump"

    assert edits.clear_window((1, 0, 0), (1, lattice.ny, lattice.nx)) == 1
    assert edits.version == 2


def test_set_plane_with_no_voxels_does_not_bump(lattice):
    edits = MaskEdits(ny=lattice.ny, nx=lattice.nx, nz=lattice.nz)
    edits.set_plane(1, [], [], [])
    assert edits.version == 0


def test_clear_bumps_only_when_there_was_something(lattice):
    edits = MaskEdits(ny=lattice.ny, nx=lattice.nx, nz=lattice.nz)
    edits.clear()
    assert edits.version == 0
    edits.set_plane(1, [1], [1], [1])
    edits.clear()
    assert edits.version == 2


def test_touched_at_keeps_a_plane_that_was_emptied(lattice):
    """`touched_planes` drops it; the repair scheme needs it kept."""
    edits = MaskEdits(ny=lattice.ny, nx=lattice.nx, nz=lattice.nz)
    edits.set_plane(3, [1], [1], [1])
    edits.clear_window((3, 0, 0), (1, lattice.ny, lattice.nx))
    assert edits.touched_planes == []
    assert 3 in edits.touched_at
