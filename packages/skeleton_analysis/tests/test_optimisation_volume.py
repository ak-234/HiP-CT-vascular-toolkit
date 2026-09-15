"""Tests for volume-based optimisation metrics (skeleton vs segmentation)."""

import numpy as np
import pytest

from skeleton_analysis.io.amira import SpatialGraph
from skeleton_analysis.io.amira_lattice import AmiraLattice
from skeleton_analysis.optimisation.meta_metric import bifurcation_dice_points
from skeleton_analysis.optimisation.volume_metrics import (
    centreline_sensitivity,
    skeleton_junction_points,
    region_morphometrics,
    region_props_table,
    super_metric,
)


def _unit_lattice(volume):
    """AmiraLattice with unit spacing / zero origin so world == voxel indices."""
    nz, ny, nx = volume.shape
    bbox = np.array([0, nx - 1, 0, ny - 1, 0, nz - 1], dtype=float)  # spacing 1
    return AmiraLattice(volume=volume.astype(np.uint8), dims=(nx, ny, nz),
                        bbox=bbox, block="Labels")


def _points_graph(point_coords):
    g = SpatialGraph()
    g.set_point_field("EdgePointCoordinates", np.asarray(point_coords, dtype=float))
    return g


def test_centreline_sensitivity_fraction():
    # Label the x==2 plane; 3 of 4 skeleton points sit there.
    vol = np.zeros((4, 4, 4), dtype=np.uint8)
    vol[:, :, 2] = 1
    lat = _unit_lattice(vol)
    g = _points_graph([[2, 1, 1], [2, 2, 2], [0, 0, 0], [2, 0, 3]])  # (x,y,z)
    assert centreline_sensitivity(g, lat) == pytest.approx(0.75)


def test_centreline_sensitivity_all_inside():
    vol = np.ones((3, 3, 3), dtype=np.uint8)
    lat = _unit_lattice(vol)
    g = _points_graph([[0, 0, 0], [1, 1, 1], [2, 2, 2]])
    assert centreline_sensitivity(g, lat) == pytest.approx(1.0)


def test_centreline_sensitivity_out_of_bounds_is_nan():
    vol = np.ones((3, 3, 3), dtype=np.uint8)
    lat = _unit_lattice(vol)
    g = _points_graph([[100, 100, 100]])  # outside the volume
    assert np.isnan(centreline_sensitivity(g, lat))


def test_bifurcation_dice_points_perfect():
    pts = np.array([[0, 0, 0], [10, 0, 0], [0, 10, 0]], dtype=float)
    res = bifurcation_dice_points(pts, pts, threshold=1.0)
    assert res.tp == 3 and res.fp == 0 and res.fn == 0
    assert res.dice == pytest.approx(1.0)


def test_bifurcation_dice_points_partial():
    cand = np.array([[0, 0, 0], [100, 0, 0]], dtype=float)  # 2nd is spurious
    ref = np.array([[0, 0.5, 0], [50, 0, 0]], dtype=float)  # 1st matches, 2nd missed
    res = bifurcation_dice_points(cand, ref, threshold=5.0)
    assert res.tp == 1 and res.fp == 1 and res.fn == 1
    assert res.dice == pytest.approx(2 / 4)


@pytest.mark.image
def test_skeleton_junction_points_plus():
    # A thin '+' in the z=3 plane -> one junction voxel at (3,3,3).
    vol = np.zeros((7, 7, 7), dtype=np.uint8)
    vol[3, 3, :] = 1  # line along x
    vol[3, :, 3] = 1  # line along y
    lat = _unit_lattice(vol)
    juncs = skeleton_junction_points(lat, crop_to_nonzero=True, pad=1)
    assert juncs.shape[0] == 1
    np.testing.assert_allclose(juncs[0], [3, 3, 3], atol=1e-6)  # world (x,y,z)


@pytest.mark.image
def test_skeleton_junction_points_none_for_straight_line():
    vol = np.zeros((7, 7, 7), dtype=np.uint8)
    vol[3, 3, :] = 1  # a single straight segment: no junction
    lat = _unit_lattice(vol)
    juncs = skeleton_junction_points(lat)
    assert juncs.shape[0] == 0


@pytest.mark.image
def test_region_morphometrics_two_cubes():
    vol = np.zeros((10, 10, 10), dtype=np.uint8)
    vol[1:4, 1:4, 1:4] = 1  # solid 3x3x3 cube (27 voxels)
    vol[6:9, 6:9, 6:9] = 1  # a second, disjoint 3x3x3 cube
    m = region_morphometrics(vol, voxel_size=1.0)
    assert m["connected_components"] == 2
    assert m["euler_number"] == 2          # two solid blobs, no holes/tunnels
    assert m["volume"] == 54               # 2 * 27 voxels
    assert m["surface_area"] > 0


@pytest.mark.image
def test_region_props_table():
    vol = np.zeros((10, 10, 10), dtype=np.uint8)
    vol[1:4, 1:4, 1:4] = 1
    vol[6:9, 6:9, 6:9] = 1
    df = region_props_table(vol)
    assert len(df) == 2
    assert set(df["area"]) == {27}


@pytest.mark.image
def test_super_metric_identical_is_zero_and_perturbed_positive():
    ref = np.zeros((12, 12, 12), dtype=np.uint8)
    ref[2:10, 2:10, 2:10] = 1              # 8^3 solid cube
    # Identical candidate -> every relative term 0 -> combined 0.
    # (A solid cube has an empty medial skeleton, so clDice is undefined and its
    # term is skipped; Volume/CC/Euler still match exactly.)
    res = super_metric(ref.copy(), ref, voxel_size=1.0)
    assert res["meta_metric"] == pytest.approx(0.0, abs=1e-9)
    assert res["candidate_Volume"] == res["reference_Volume"]

    cand = np.zeros_like(ref)
    cand[2:8, 2:8, 2:8] = 1                # smaller cube -> volume differs
    res2 = super_metric(cand, ref, voxel_size=1.0)
    assert res2["meta_metric"] > 0.0
    assert res2["candidate_Volume"] < res2["reference_Volume"]
