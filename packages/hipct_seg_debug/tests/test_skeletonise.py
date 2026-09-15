"""Skeletonisation must produce a graph with the topology the shape actually has.

The assertion that earns its keep is the junction count: a bifurcation thins to a
*blob* of high-degree voxels, and without clustering that becomes several nodes a
voxel apart joined by zero-length edges. Every downstream measure -- bifurcation
counts, Strahler order, Murray ratios -- is then wrong, and nothing about the
picture looks obviously broken.
"""

from __future__ import annotations

import numpy as np
import pytest

from hipct_seg_debug.edit.skeletonise import (
    neighbour_counts,
    skeleton_to_graph,
    skeletonise,
)

SPACING = np.array([10.0, 10.0, 10.0])  # um per voxel, (x, y, z)
ORIGIN = np.zeros(3)


def draw_tube(volume, a, b, radius):
    """Paint a cylinder between two (z, y, x) points."""
    zz, yy, xx = np.indices(volume.shape)
    grid = np.stack([zz, yy, xx], axis=-1).astype(np.float64)
    a, b = np.asarray(a, float), np.asarray(b, float)
    ab = b - a
    t = np.clip(((grid - a) @ ab) / max(float(ab @ ab), 1e-9), 0.0, 1.0)
    closest = a + t[..., None] * ab
    volume[np.linalg.norm(grid - closest, axis=-1) <= radius] = True
    return volume


def trace(mask):
    skel, edt = skeletonise(mask, SPACING)
    return skel, edt, skeleton_to_graph(skel, edt, ORIGIN, SPACING)


# ---------------------------------------------------------------- primitives

def test_neighbour_counts_on_a_straight_line():
    skel = np.zeros((9, 9, 9), dtype=bool)
    skel[4, 4, 2:7] = True
    counts = neighbour_counts(skel)
    interior = counts[4, 4, 3:6]
    assert list(interior) == [2, 2, 2], "interior voxels of a line have two neighbours"
    assert counts[4, 4, 2] == 1 and counts[4, 4, 6] == 1, "ends have one"
    assert counts[0, 0, 0] == 0, "background must not be counted"


def test_edt_recovers_a_known_radius():
    mask = np.zeros((40, 40, 40), dtype=bool)
    draw_tube(mask, (20, 20, 4), (20, 20, 35), radius=5.0)
    _skel, edt, _ = trace(mask)
    # On the axis, the distance to background is the radius, in um.
    assert edt[20, 20, 20] == pytest.approx(5.0 * SPACING[0], abs=SPACING[0])


# ----------------------------------------------------------------- topology

def test_a_straight_tube_is_one_segment():
    mask = np.zeros((30, 30, 60), dtype=bool)
    draw_tube(mask, (15, 15, 5), (15, 15, 54), radius=4.0)
    _skel, _edt, result = trace(mask)
    tri = result.triple

    assert len(tri.segments) == 1
    assert len(tri.nodes) == 2
    assert sorted(n[3] for n in tri.nodes.values()) == [1, 1]
    assert len(tri.points) > 20


def test_a_y_has_one_junction_not_a_cluster_of_them():
    """The test that catches a missing junction-clustering step."""
    mask = np.zeros((40, 60, 60), dtype=bool)
    fork = (20, 30, 30)
    draw_tube(mask, (20, 30, 5), fork, radius=4.0)
    draw_tube(mask, fork, (20, 10, 54), radius=3.0)
    draw_tube(mask, fork, (20, 50, 54), radius=3.0)

    _skel, _edt, result = trace(mask)
    tri = result.triple
    degrees = sorted(n[3] for n in tri.nodes.values())

    assert degrees.count(3) == 1, f"expected exactly one junction, got degrees {degrees}"
    assert degrees.count(1) == 3, f"expected three free ends, got degrees {degrees}"
    assert len(tri.segments) == 3
    assert len(tri.nodes) == 4
    assert result.n_junction_clusters >= 1


def test_the_junction_node_sits_near_the_real_fork():
    mask = np.zeros((40, 60, 60), dtype=bool)
    fork = (20, 30, 30)
    draw_tube(mask, (20, 30, 5), fork, radius=4.0)
    draw_tube(mask, fork, (20, 10, 54), radius=3.0)
    draw_tube(mask, fork, (20, 50, 54), radius=3.0)

    _skel, _edt, result = trace(mask)
    tri = result.triple
    junction = next(n for n in tri.nodes.values() if n[3] == 3)
    want = ORIGIN + np.asarray(fork, float)[::-1] * SPACING
    assert np.linalg.norm(np.asarray(junction[:3]) - want) < 8 * SPACING[0]


def test_two_separate_tubes_give_two_components():
    mask = np.zeros((40, 40, 60), dtype=bool)
    draw_tube(mask, (12, 12, 5), (12, 12, 54), radius=3.0)
    draw_tube(mask, (28, 28, 5), (28, 28, 54), radius=3.0)

    _skel, _edt, result = trace(mask)
    from hipct_seg_debug.edit.graphmodel import EditableGraph

    g = EditableGraph(result.triple)
    assert len(g.components()) == 2
    assert len(result.triple.segments) == 2


def test_an_empty_mask_gives_an_empty_graph():
    mask = np.zeros((16, 16, 16), dtype=bool)
    _skel, _edt, result = trace(mask)
    assert result.triple.segments == []
    assert result.n_skeleton_voxels == 0


def test_a_closed_loop_is_not_dropped():
    """A torus has no degree-1 or degree-3 voxel, so it needs breaking explicitly."""
    n = 48
    zz, yy, xx = np.indices((20, n, n))
    cy = cx = n / 2.0
    ring = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    mask = (np.abs(ring - 15.0) <= 3.0) & (np.abs(zz - 10) <= 3.0)

    _skel, _edt, result = trace(mask)
    assert len(result.triple.segments) >= 1, "the loop was dropped entirely"
    assert len(result.triple.points) > 20


# ------------------------------------------------------------------- output

def test_the_result_is_a_usable_triple():
    mask = np.zeros((30, 30, 60), dtype=bool)
    draw_tube(mask, (15, 15, 5), (15, 15, 54), radius=4.0)
    _skel, _edt, result = trace(mask)

    from hipct_seg_debug.edit.adapter import to_spatial_graph
    from hipct_seg_debug.edit.graphmodel import EditableGraph

    g = EditableGraph(result.triple)
    assert g.radii(g.segment_ids()[0]).min() > 0, "radii must be positive"

    sg = to_spatial_graph(result.triple)
    assert sg.n_edge == len(result.triple.segments)
    assert int(sg.n_edge_points.sum()) == sg.n_point
    assert sg.connectivity.max() < sg.n_vertex


def test_points_are_in_world_micrometres():
    mask = np.zeros((30, 30, 60), dtype=bool)
    draw_tube(mask, (15, 15, 5), (15, 15, 54), radius=4.0)
    _skel, _edt, result = trace(mask)

    pts = np.array([p[:3] for p in result.triple.points.values()])
    # The tube runs along x (the last axis), so x should span most of the volume
    # in um and y/z should sit near the centre.
    assert pts[:, 0].ptp() > 30 * SPACING[0]
    assert abs(pts[:, 1].mean() - 15 * SPACING[1]) < 3 * SPACING[1]
    assert abs(pts[:, 2].mean() - 15 * SPACING[2]) < 3 * SPACING[2]


def test_an_origin_offset_shifts_every_point():
    mask = np.zeros((30, 30, 60), dtype=bool)
    draw_tube(mask, (15, 15, 5), (15, 15, 54), radius=4.0)
    skel, edt = skeletonise(mask, SPACING)

    a = skeleton_to_graph(skel, edt, np.zeros(3), SPACING)
    b = skeleton_to_graph(skel, edt, np.array([1000.0, 2000.0, 3000.0]), SPACING)
    pa = np.array([p[:3] for p in a.triple.points.values()])
    pb = np.array([p[:3] for p in b.triple.points.values()])
    assert np.allclose(pb.mean(axis=0) - pa.mean(axis=0), [1000.0, 2000.0, 3000.0])


def test_describe_mentions_the_counts():
    mask = np.zeros((30, 30, 60), dtype=bool)
    draw_tube(mask, (15, 15, 5), (15, 15, 54), radius=4.0)
    _skel, _edt, result = trace(mask)
    text = result.describe()
    assert "segments" in text and "skeleton voxels" in text
