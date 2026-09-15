"""Tests for the outlier-correction module."""

import numpy as np
import pytest

from skeleton_analysis.io.amira import SpatialGraph
from skeleton_analysis.outlier.detect import (
    along_segment_outliers,
    correct_along_segment_thickness,
    detect_collapsed_segments,
    filloutliers_nearest,
    isoutlier_percentiles,
    matlab_prctile,
)


def _graph(num_edge_points=None, thickness=None, edge_fields=None):
    g = SpatialGraph()
    if num_edge_points is not None:
        g.set_edge_field("EdgeConnectivity",
                         np.zeros((len(num_edge_points), 2), dtype=np.int64))
        g.set_edge_field("NumEdgePoints", np.asarray(num_edge_points, np.int64))
    if thickness is not None:
        g.set_point_field("thickness", np.asarray(thickness, float))
    for name, vals in (edge_fields or {}).items():
        g.set_edge_field(name, np.asarray(vals))
    return g


# ---------------------------------------------------------------------------
# MATLAB-compatible percentile
# ---------------------------------------------------------------------------
def test_matlab_prctile_matches_reference():
    x = [1, 2, 3, 4, 5]
    assert matlab_prctile(x, 50) == pytest.approx(3.0)
    assert matlab_prctile(x, 25) == pytest.approx(1.75)  # MATLAB prctile([1..5],25)
    assert matlab_prctile(x, 10) == pytest.approx(1.0)
    assert matlab_prctile(x, 5) == pytest.approx(1.0)  # clamps to min
    assert matlab_prctile(x, 100) == pytest.approx(5.0)  # clamps to max


# ---------------------------------------------------------------------------
# isoutlier / filloutliers on the low tail
# ---------------------------------------------------------------------------
def test_low_tail_outlier_and_fill():
    x = np.array([5.0] * 20 + [0.1])  # one collapsed point among 21
    mask = isoutlier_percentiles(x, 5, 100)
    assert mask.sum() == 1 and mask[-1]
    filled = filloutliers_nearest(x, 5, 100)
    assert filled[-1] == pytest.approx(5.0)  # replaced by nearest good value

    idx, repl = along_segment_outliers(x)
    assert idx.tolist() == [20]
    assert repl[0] == pytest.approx(5.0)


def test_no_outliers_for_small_segments():
    # With few points the 5th percentile clamps to the minimum -> nothing flagged.
    x = np.array([1.0, 10.0, 11.0, 12.0, 13.0])
    assert isoutlier_percentiles(x, 5, 100).sum() == 0


# ---------------------------------------------------------------------------
# Whole-graph corrections
# ---------------------------------------------------------------------------
def test_correct_along_segment_thickness():
    seg0 = [5.0] * 20 + [0.1]  # collapsed point at global index 20
    seg1 = [2.0, 2.0, 2.0]
    g = _graph(num_edge_points=[21, 3], thickness=seg0 + seg1)
    corrected, changed = correct_along_segment_thickness(g)
    assert changed.tolist() == [20]
    assert corrected[20] == pytest.approx(5.0)
    # Second segment untouched.
    np.testing.assert_allclose(corrected[21:], [2.0, 2.0, 2.0])
    # Input graph is not mutated.
    assert g.thickness[20] == pytest.approx(0.1)


def test_apply_manual_plane_selection():
    import pandas as pd
    from skeleton_analysis.outlier.correct import apply_manual_plane_selection

    seg_radii = {
        10: np.array([100.0, 50.0, 200.0]),
        20: np.array([5.0, 5.0, 5.0]),
        30: np.array([130.0, 400.0]),
    }
    # seg10: plane index 2 -> auto[1]=50 (single val -> expand +/-5%);
    # seg20: first value NaN -> not corrected;
    # seg30: two literal radii (non-clean decimals) -> snap to nearest.
    table = pd.DataFrame({
        "genx_segment_no": [10, 20, 30],
        "p1": [2.0, np.nan, 123.45],
        "p2": [np.nan, np.nan, 456.78],
    })
    out = apply_manual_plane_selection(seg_radii, table)
    np.testing.assert_allclose(out[10], [52.5, 50.0, 52.5])
    assert np.isnan(out[20]).all()
    np.testing.assert_allclose(out[30], [123.45, 456.78])


def test_detect_collapsed_segments():
    # 6 edges: two at order 8 (flagged wholesale), four at order 5 with one
    # abnormally small radius that must be caught by the percentile rule.
    strahler = [8, 8, 5, 5, 5, 5]
    radius = [1.0, 1.1, 2.0, 2.1, 2.2, 0.2]
    g = _graph(edge_fields={"strahler": strahler, "MeanRadius": radius})
    flagged = detect_collapsed_segments(
        g, flag_orders=(6, 7, 8, 9), percentile_orders=(5,), percentile=25
    )
    assert 0 in flagged and 1 in flagged  # order-8 flagged wholesale
    assert 5 in flagged  # the collapsed order-5 vessel
    assert 2 not in flagged  # a normal order-5 vessel


# ---------------------------------------------------------------------------
# Oblique slicing (requires the [image] extra)
# ---------------------------------------------------------------------------
@pytest.mark.image
def test_cross_section_radius_disk():
    from skimage.draw import disk
    from skeleton_analysis.outlier.oblique import cross_section_radius

    img = np.zeros((41, 41), dtype=float)
    rr, cc = disk((20, 20), 8, shape=img.shape)
    img[rr, cc] = 1.0
    r = cross_section_radius(img, res=1.0, threshold=0.5)
    assert r == pytest.approx(8.0, abs=1.5)  # perimeter/(2 pi); discretization slack


@pytest.mark.image
def test_segment_radii_from_cylinder():
    from skeleton_analysis.outlier.oblique import segment_radii_from_volume

    # Cylinder of radius 5 (voxels) along the z axis; volume axes (z, y, x).
    Z, Y, X = 12, 41, 41
    yy, xx = np.mgrid[0:Y, 0:X]
    mask = (yy - 20) ** 2 + (xx - 20) ** 2 <= 5 ** 2
    volume = np.repeat(mask[None, :, :].astype(float), Z, axis=0)

    centreline = np.array([[z, 20.0, 20.0] for z in range(Z)], dtype=float)
    rads = segment_radii_from_volume(
        volume, centreline, res=1.0, half_size=18, threshold=0.5
    )
    assert rads.shape == (Z,)
    assert np.all(np.isfinite(rads))
    assert np.median(rads) == pytest.approx(5.0, abs=1.5)


@pytest.mark.image
def test_cross_section_details_area_vs_perimeter():
    from skimage.draw import disk
    from skeleton_analysis.outlier.oblique import cross_section_radius

    img = np.zeros((41, 41), dtype=float)
    rr, cc = disk((20, 20), 8, shape=img.shape)
    img[rr, cc] = 1.0
    cs = cross_section_radius(img, res=1.0, threshold=0.5, return_details=True)
    # Both estimates recover the true radius (8) for a well-resolved disk; the
    # area estimate is the tighter one. (Perimeter only over-states at ~1-2 vox.)
    assert cs.r_area == pytest.approx(8.0, abs=0.5)
    assert cs.r_perimeter == pytest.approx(8.0, abs=1.5)
    assert cs.perimeter > 0 and cs.area > 0
    # `method` selects which estimate `.radius` (and the scalar return) reports.
    assert cross_section_radius(img, threshold=0.5, method="area") == pytest.approx(cs.r_area)
    assert cross_section_radius(img, threshold=0.5, method="perimeter") == pytest.approx(cs.r_perimeter)


@pytest.mark.image
def test_segment_radii_area_method_cylinder():
    from skeleton_analysis.outlier.oblique import segment_radii_from_volume

    Z, Y, X = 12, 41, 41
    yy, xx = np.mgrid[0:Y, 0:X]
    mask = (yy - 20) ** 2 + (xx - 20) ** 2 <= 5 ** 2
    volume = np.repeat(mask[None, :, :].astype(float), Z, axis=0)
    centreline = np.array([[z, 20.0, 20.0] for z in range(Z)], dtype=float)
    rads = segment_radii_from_volume(
        volume, centreline, res=1.0, half_size=18, threshold=0.5, method="area"
    )
    # Area-based tracks the true radius tightly.
    assert np.median(rads) == pytest.approx(5.0, abs=0.6)


@pytest.mark.image
def test_oblique_debug_pngs(tmp_path):
    from skeleton_analysis.outlier.oblique import segment_radii_from_volume

    # A thin cylinder (radius ~2 voxels) so slices qualify as "near voxel size".
    Z, Y, X = 8, 21, 21
    yy, xx = np.mgrid[0:Y, 0:X]
    mask = (yy - 10) ** 2 + (xx - 10) ** 2 <= 2 ** 2
    volume = np.repeat(mask[None, :, :].astype(float), Z, axis=0)
    centreline = np.array([[z, 10.0, 10.0] for z in range(Z)], dtype=float)

    debug = {"dir": tmp_path / "dbg", "radius_vox_max": 5.0, "budget": 3,
             "saved": 0, "prefix": "seg_"}
    segment_radii_from_volume(volume, centreline, res=1.0, half_size=9,
                              threshold=0.5, debug=debug)
    pngs = list((tmp_path / "dbg").glob("*.png"))
    assert 1 <= len(pngs) <= 3            # at least one saved, budget cap respected
    assert debug["saved"] == len(pngs)


# --- cut-plane orientation and window sizing --------------------------------
# The radius here is perimeter/(2*pi), so anything that mis-orients or clips the
# cut plane converts directly into radius error. These pin the two mechanisms.

def _cylinder(radius_vox, length=40, pad=14.0):
    """Binary cylinder of known radius along z; volume axes (z, y, x)."""
    half = int(np.ceil(radius_vox + pad))
    size = 2 * half + 1
    yy, xx = np.mgrid[0:size, 0:size]
    disc = ((yy - half) ** 2 + (xx - half) ** 2) <= radius_vox ** 2
    vol = np.repeat(disc[None, :, :].astype(float), length, axis=0)
    centre = np.array([[z, float(half), float(half)] for z in range(length)])
    return vol, centre


@pytest.mark.image
def test_radius_is_stable_under_centreline_jitter():
    """Centreline noise must not change a measured radius.

    A one-step-difference normal tilts the plane, which cuts an ellipse and
    inflates perimeter/(2*pi) by ~1/cos(tilt): on this cylinder it reaches +40%
    at one voxel of jitter. The radius-scaled quadratic tangent is what keeps the
    measurement flat.
    """
    from skeleton_analysis.outlier.oblique import segment_radii_from_volume

    rng = np.random.default_rng(0)
    truth = 8.0
    for amp in (0.0, 0.5, 1.0, 1.5):
        vol, centre = _cylinder(truth)
        jittered = centre.copy()
        if amp:
            jittered[:, 1:] += rng.normal(0.0, amp, size=(len(jittered), 2))
        rads = segment_radii_from_volume(
            vol, jittered, res=1.0, threshold=0.5,
            radii=np.full(len(jittered), truth), max_half=64,
        )
        err = abs(np.nanmedian(rads) - truth) / truth
        # 4% is the digitised-disc perimeter bias, present even at zero jitter;
        # the point is that it does not GROW with jitter.
        assert err < 0.08, f"jitter {amp}: radius moved {err:.1%}"


@pytest.mark.image
def test_window_grows_for_vessels_wider_than_it():
    """A fixed window cannot measure a vessel wider than itself.

    At the historical default (half_size=20) a radius-40 tube reads ~25 -- the
    section is clipped and its traced perimeter is partly the window boundary.
    This is the mechanism behind systematically undersized large vessels.
    """
    from skeleton_analysis.outlier.oblique import segment_radii_from_volume

    truth = 40.0
    vol, centre = _cylinder(truth)
    report = {}
    rads = segment_radii_from_volume(
        vol, centre, res=1.0, threshold=0.5,
        radii=np.full(len(centre), truth), max_half=128, report=report,
    )
    assert abs(np.nanmedian(rads) - truth) / truth < 0.08
    assert report["n_clipped"] == 0
    assert report["n_measured"] == len(centre)


@pytest.mark.image
def test_background_centre_is_rejected_not_guessed():
    """Off-lumen centreline points must yield NaN, not a neighbour's radius."""
    from skeleton_analysis.outlier.oblique import cross_section_radius

    img = np.zeros((41, 41), dtype=float)
    img[4:12, 4:12] = 1.0          # a blob well away from the centre
    assert np.isnan(cross_section_radius(img, res=1.0, threshold=0.5))
    # The historical guess-nearest behaviour stays available explicitly.
    assert cross_section_radius(img, res=1.0, threshold=0.5,
                                require_center=False) > 0
