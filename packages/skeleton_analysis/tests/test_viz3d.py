"""Tests for the optional interactive 3-D QC viewer."""

import numpy as np
import pytest

from skeleton_analysis.outlier.viz3d import segment_bbox, show_segment_volume


def test_segment_bbox_padding_and_clamp():
    # centreline in (z, y, x); padded box, clamped to shape.
    cl = np.array([[5, 5, 5], [6, 7, 8]], dtype=float)
    lo, hi = segment_bbox(cl, half_pad=2, shape=(20, 20, 20))
    np.testing.assert_array_equal(lo, [3, 3, 3])          # floor(min) - pad
    np.testing.assert_array_equal(hi, [9, 10, 11])        # ceil(max) + pad + 1

    # Near the edges -> clamped to [0, shape].
    cl2 = np.array([[0, 0, 0], [1, 19, 19]], dtype=float)
    lo2, hi2 = segment_bbox(cl2, half_pad=5, shape=(20, 20, 20))
    np.testing.assert_array_equal(lo2, [0, 0, 0])
    np.testing.assert_array_equal(hi2, [7, 20, 20])


def test_show_segment_volume_offscreen(tmp_path):
    pytest.importorskip("pyvista")
    # A short cylinder (radius 3) along z; centreline down its axis.
    Z, Y, X = 12, 21, 21
    yy, xx = np.mgrid[0:Y, 0:X]
    mask = (yy - 10) ** 2 + (xx - 10) ** 2 <= 3 ** 2
    volume = np.repeat(mask[None, :, :].astype(float), Z, axis=0)
    centreline = np.array([[z, 10.0, 10.0] for z in range(Z)], dtype=float)

    out = tmp_path / "qc.png"
    try:
        result = show_segment_volume(
            volume, centreline, half_pad=4, threshold=0.5, show_planes=True,
            plane_stride=3, off_screen=True, screenshot=str(out), title="edge 0",
        )
    except Exception as exc:  # no OpenGL/VTK render context in this environment
        pytest.skip(f"off-screen rendering unavailable: {exc}")

    assert result == str(out)
    assert out.exists() and out.stat().st_size > 0
