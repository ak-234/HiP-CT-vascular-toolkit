"""Which files in a raw directory are slices.

Tests for a bug that was silent everywhere and wrong everywhere. Raw directories can
contain sidecars beside slice zero. On the real
LADAF-2024-28 overview that is five files (``.tif.dat``, ``.tif.fcp``, ``.tif.lda`` and
two ``.bck`` twins), and because they sort immediately after ``..._000000.tif`` they
take indices 1-5 and push every real slice down by five.

Nothing downstream could notice. ``shape`` simply reported 4757 slices instead of 4752,
``WorldFrame`` derived its binning from that number, and every raw sample after the
first came from the wrong slice -- for ``connect --dpc``, ``connect --geodesic --raw``,
``train-cfc``, ``evaluate-dpc`` and the viewer alike.
"""

from __future__ import annotations

import numpy as np
import pytest

tifffile = pytest.importorskip("tifffile")

from hipct_seg_debug.tiffstack import TiffStack  # noqa: E402


def _stack_dir(tmp_path, n=6, sidecars=()):
    for z in range(n):
        tifffile.imwrite(
            tmp_path / f"slice_{z:06d}.tif",
            np.full((4, 5), z, dtype=np.uint16),
        )
    for name in sidecars:
        (tmp_path / name).write_bytes(b"not a tiff")
    return tmp_path


def test_sidecars_beside_slice_zero_are_not_slices(tmp_path):
    directory = _stack_dir(tmp_path, n=6, sidecars=(
        "slice_000000.tif.dat", "slice_000000.tif.fcp",
        "slice_000000.tif.fcp.bck", "slice_000000.tif.lda",
        "slice_000000.tif.lda.bck",
    ))
    stack = TiffStack(directory)

    assert stack.shape[0] == 6, "the sidecars were counted as slices"
    assert stack.skipped == 5
    # ...and, the part that actually mattered: slice z really is slice z.
    for z in range(6):
        assert int(stack.read_slice(z)[0, 0]) == z


def test_a_clean_directory_reports_nothing_skipped(tmp_path):
    stack = TiffStack(_stack_dir(tmp_path, n=4))
    assert stack.shape[0] == 4
    assert stack.skipped == 0


def test_tiff_and_tiff_long_suffix_are_both_slices(tmp_path):
    tifffile.imwrite(tmp_path / "a_000000.tif", np.zeros((3, 3), np.uint16))
    tifffile.imwrite(tmp_path / "a_000001.tiff", np.ones((3, 3), np.uint16))
    stack = TiffStack(tmp_path)
    assert stack.shape[0] == 2 and stack.skipped == 0


def test_a_directory_of_only_sidecars_is_refused(tmp_path):
    (tmp_path / "x_000000.tif.dat").write_bytes(b"nope")
    with pytest.raises(FileNotFoundError, match="supported image"):
        TiffStack(tmp_path)


def test_jpeg2000_slices_are_discovered_sorted_and_decoded(tmp_path):
    pillow = pytest.importorskip("PIL.Image")
    features = pytest.importorskip("PIL.features")
    if not features.check("jpg_2000"):
        pytest.skip("Pillow was built without JPEG 2000 support")

    for z in (10, 2, 1):
        image = np.full((4, 5), z, dtype=np.uint8)
        pillow.fromarray(image).save(tmp_path / f"slice_{z}.jp2", format="JPEG2000")
    (tmp_path / "slice_1.jp2.dat").write_bytes(b"not an image")

    stack = TiffStack(tmp_path)
    assert stack.shape == (3, 4, 5)
    assert stack.dtype == np.dtype(np.uint8)
    assert stack.skipped == 1
    assert [int(stack.read_slice(z)[0, 0]) for z in range(3)] == [1, 2, 10]


def test_explicit_pattern_can_select_one_supported_format(tmp_path):
    tifffile.imwrite(tmp_path / "slice_1.tif", np.ones((3, 4), np.uint16))
    tifffile.imwrite(tmp_path / "slice_2.tiff", np.ones((3, 4), np.uint16) * 2)

    stack = TiffStack(tmp_path, pattern="*.tiff")
    assert stack.shape == (1, 3, 4)
    assert int(stack.read_slice(0)[0, 0]) == 2


# ------------------------------------------------------- strip-windowed reads


def _striped(tmp_path, shape=(64, 48), n=3, rowsperstrip=8, **kw):
    """A directory of striped TIFFs with a distinctive per-pixel value."""
    rng = np.random.default_rng(0)
    volume = rng.integers(0, 4000, size=(n, *shape), dtype=np.uint16)
    for z in range(n):
        tifffile.imwrite(
            tmp_path / f"s_{z:04d}.tif", volume[z], rowsperstrip=rowsperstrip, **kw
        )
    return volume


def _expect(volume, z, r0, r1, c0, c1):
    """What `read_window` should return, computed the slow obvious way."""
    nr, nc = volume.shape[1:]
    out = np.zeros((r1 - r0, c1 - c0), dtype=volume.dtype)
    a0, a1, b0, b1 = max(0, r0), min(nr, r1), max(0, c0), min(nc, c1)
    if a0 < a1 and b0 < b1:
        out[a0 - r0:a1 - r0, b0 - c0:b1 - c0] = volume[z, a0:a1, b0:b1]
    return out


@pytest.mark.parametrize("rowsperstrip", [1, 8, 64])
def test_a_strip_window_is_bit_identical_to_the_whole_page(tmp_path, rowsperstrip):
    """The whole point: decoding fewer strips must change nothing but the time.

    Measured on the real LADAF-2024-28 overview (3079 rows, one row per strip, LZW):
    188 ms for the page against 10 ms for a 130-row window. That is only worth having
    if the pixels are the same ones, so this compares against the full decode directly.
    """
    volume = _striped(tmp_path, rowsperstrip=rowsperstrip)
    stack = TiffStack(tmp_path, cache_slices=0)
    for r0, r1, c0, c1 in [
        (10, 26, 8, 24),      # interior
        (0, 16, 0, 16),       # flush with the origin
        (48, 64, 32, 48),     # flush with the far corner
        (17, 23, 5, 41),      # strip-boundary-straddling, odd offsets
    ]:
        got = stack.read_window(1, r0, r1, c0, c1)
        np.testing.assert_array_equal(got, _expect(volume, 1, r0, r1, c0, c1))


def test_a_window_running_off_the_edge_is_still_zero_padded(tmp_path):
    """`read_window`'s padding contract has to survive the new read path."""
    volume = _striped(tmp_path, rowsperstrip=8)
    stack = TiffStack(tmp_path, cache_slices=0)
    for r0, r1, c0, c1 in [
        (-8, 8, -8, 8),        # straddles the origin
        (56, 72, 40, 56),      # straddles the far corner
        (-40, -20, 0, 16),     # entirely outside
    ]:
        got = stack.read_window(0, r0, r1, c0, c1)
        np.testing.assert_array_equal(got, _expect(volume, 0, r0, r1, c0, c1))


def test_a_tiled_tiff_falls_back_and_is_still_right(tmp_path):
    """Tiles are 2D segments, so the row arithmetic does not apply to them."""
    rng = np.random.default_rng(1)
    plane = rng.integers(0, 4000, size=(64, 64), dtype=np.uint16)
    tifffile.imwrite(tmp_path / "s_0000.tif", plane, tile=(16, 16))

    stack = TiffStack(tmp_path, cache_slices=4)
    got = stack.read_window(0, 20, 36, 12, 28)
    np.testing.assert_array_equal(got, plane[20:36, 12:28])
    assert stack.strip_reads == 0
    assert stack.whole_page_reads == 1


def test_a_window_covering_most_of_the_page_uses_the_whole_page_path(tmp_path):
    """Per-strip overhead stops paying once the window is most of the rows."""
    _striped(tmp_path, rowsperstrip=1)
    stack = TiffStack(tmp_path, cache_slices=4)
    stack.read_window(0, 0, 64, 0, 48)      # all 64 rows
    assert stack.whole_page_reads == 1 and stack.strip_reads == 0


def test_a_cached_plane_is_cropped_rather_than_decoded_again(tmp_path):
    """A plane already in the LRU is strictly better than any partial decode."""
    volume = _striped(tmp_path, rowsperstrip=1)
    stack = TiffStack(tmp_path, cache_slices=4)
    stack.read_slice(2)                      # warm the cache
    got = stack.read_window(2, 10, 26, 8, 24)
    np.testing.assert_array_equal(got, _expect(volume, 2, 10, 26, 8, 24))
    assert stack.strip_reads == 0 and stack.whole_page_reads == 0


def test_read_stack_window_still_agrees_with_the_slow_path(tmp_path):
    volume = _striped(tmp_path, n=4, rowsperstrip=8)
    stack = TiffStack(tmp_path, cache_slices=0)
    got = stack.read_stack_window(1, 4, 10, 26, 8, 24)
    want = np.stack([_expect(volume, z, 10, 26, 8, 24) for z in (1, 2, 3)])
    np.testing.assert_array_equal(got, want)
    assert stack.strip_reads == 3


def test_a_non_tiff_slice_falls_back_without_trying_twice(tmp_path):
    """Pillow formats have no strip concept here; the check is made once."""
    pytest.importorskip("PIL")
    from PIL import Image

    for z in range(2):
        Image.fromarray(np.full((32, 32), 100 + z, dtype=np.uint16)).save(
            tmp_path / f"s_{z:04d}.png"
        )
    stack = TiffStack(tmp_path, cache_slices=0)
    got = stack.read_window(1, 4, 20, 4, 20)
    assert (got == 101).all()
    assert stack._strip_reads_ok is False
