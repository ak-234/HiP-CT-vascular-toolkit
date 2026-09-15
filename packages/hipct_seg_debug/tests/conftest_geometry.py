"""Synthetic tubes and graphs shared by the skeleton-optimisation tests.

A real dataset cannot answer "is this radius right?" -- that is the whole reason the
super metric exists. A cylinder can: its volume, radius, component count and Euler
number are known in closed form, so a measurement that disagrees is wrong rather than
merely different.
"""

from __future__ import annotations

import numpy as np

from hipct_seg_debug.amira import LatticeInfo
from hipct_seg_debug.edit.adapter import Triple
from hipct_seg_debug.edit.graphmodel import EditableGraph
from hipct_seg_debug.frame import WorldFrame

SPACING = 10.0  # um per segmentation voxel


def make_frame(shape) -> WorldFrame:
    """A `WorldFrame` over a ``(nz, ny, nx)`` volume at :data:`SPACING`."""
    nz, ny, nx = shape
    dims = np.array([nx, ny, nz], dtype=np.int64)
    bbox = np.empty(6)
    bbox[0::2] = 0.0
    bbox[1::2] = (dims - 1) * SPACING
    info = LatticeInfo(path=None, dims=dims, bbox=bbox, fields={})
    return WorldFrame.from_inputs((nz * 2, ny * 2, nx * 2), SPACING / 2, info)


def cylinder(shape, radius_vox, x0, x1, *, cy=None, cz=None) -> np.ndarray:
    """A round tube along x, of `radius_vox` voxels, spanning ``[x0, x1)``."""
    nz, ny, nx = shape
    cy = ny // 2 if cy is None else cy
    cz = nz // 2 if cz is None else cz
    zz, yy, xx = np.ogrid[:nz, :ny, :nx]
    radial = (zz - cz) ** 2 + (yy - cy) ** 2 <= radius_vox * radius_vox
    return (radial & (xx >= x0) & (xx < x1)).astype(np.uint8)


def two_cylinders(shape) -> np.ndarray:
    """Two disjoint tubes of deliberately different size -- a left and a right tree.

    The sizes differ so the largest-first tree ordering is *decided* rather than
    resolved by whatever order the labeller happened to produce. Their bounding boxes
    do not overlap, which is the simple case; :func:`interleaved_components` is the
    one that catches a bounding-box crop admitting the wrong component.
    """
    nz, ny, nx = shape
    big = cylinder(shape, 4, 5, nx - 5, cy=ny // 4, cz=nz // 2)
    small = cylinder(shape, 2, nx // 2, nx - 8, cy=3 * ny // 4, cz=nz // 2)
    return np.maximum(big, small)


def interleaved_components(shape) -> np.ndarray:
    """Two disjoint components whose bounding boxes overlap heavily.

    An L-shaped tube and a bar threaded through the L's corner without touching it.
    Cropping either component's bounding box and thresholding (``volume[box] > 0``)
    picks up part of the other, which would skeletonise it twice; cropping by label
    does not. That is the only thing this fixture exists to catch.
    """
    nz, ny, nx = shape
    cz = nz // 2
    out = np.zeros(shape, dtype=np.uint8)
    zz, yy, xx = np.ogrid[:nz, :ny, :nx]

    # The L: along x at y=4, turning to run along y at x=nx-6.
    arm_x = (np.abs(yy - 4) <= 1) & (np.abs(zz - cz) <= 1) & (xx >= 3) & (xx < nx - 4)
    arm_y = (np.abs(xx - (nx - 6)) <= 1) & (np.abs(zz - cz) <= 1) & (yy >= 4) & (yy < ny - 3)
    out[arm_x | arm_y] = 1

    # The bar: along x at y=ny-6, inside the L's bounding box, separated from both arms.
    bar = (np.abs(yy - (ny - 6)) <= 1) & (np.abs(zz - cz) <= 1) & (xx >= 3) & (xx < nx - 10)
    out[bar] = 1
    return out


def slit(shape, half_y, half_z, x0, x1, *, cy=None, cz=None) -> np.ndarray:
    """A collapsed tube: a rectangular slit along x, wide in y and thin in z.

    The case the whole re-centring argument turns on. Its distance transform has a
    *flat ridge* along the major axis, so "the centre" is not defined by the distance
    maximum -- but the area centroid is, and sits at the middle of the slit.
    """
    nz, ny, nx = shape
    cy = ny // 2 if cy is None else cy
    cz = nz // 2 if cz is None else cz
    zz, yy, xx = np.ogrid[:nz, :ny, :nx]
    return (
        (np.abs(yy - cy) <= half_y) & (np.abs(zz - cz) <= half_z)
        & (xx >= x0) & (xx < x1)
    ).astype(np.uint8)


def axis_graph(frame, x0, x1, radius_um, *, cy, cz, step=1) -> EditableGraph:
    """A straight centreline along x at voxel row `cy`, slice `cz`."""
    xs = np.arange(x0, x1, step)
    ijk = np.stack([xs, np.full_like(xs, cy), np.full_like(xs, cz)], axis=1)
    xyz = frame.seg_to_um(ijk)
    points = {
        i: (float(p[0]), float(p[1]), float(p[2]), float(radius_um))
        for i, p in enumerate(xyz)
    }
    nodes = {0: (*xyz[0], 0), 1: (*xyz[-1], 0)}
    segments = [{"id": 0, "node1": 0, "node2": 1, "point_ids": list(range(len(xs)))}]
    return EditableGraph(Triple(nodes, points, segments))


def _graph_through(frame, ijk, radius_um) -> EditableGraph:
    """One segment through the given voxel coordinates, all at the same radius."""
    xyz = frame.seg_to_um(np.asarray(ijk))
    points = {
        i: (float(p[0]), float(p[1]), float(p[2]), float(radius_um))
        for i, p in enumerate(xyz)
    }
    nodes = {0: (*xyz[0], 0), 1: (*xyz[-1], 0)}
    segments = [{"id": 0, "node1": 0, "node2": 1, "point_ids": list(range(len(xyz)))}]
    return EditableGraph(Triple(nodes, points, segments))


def staircase_graph(frame, x0, x1, radius_um, *, cy, cz,
                    slope_y=0.5, slope_z=0.3) -> EditableGraph:
    """A thinned skeleton of a vessel that is *not* axis-aligned: a rasterised diagonal.

    Lee thinning walks voxel to voxel, so an oblique vessel comes out as a flight of
    single-voxel steps and the line turns 45 or 90 degrees at nearly every point even
    though the vessel is straight. Measured on LADAF-2024-28 at stride 1 the median turn
    of `lee.am` is exactly 45 degrees, and the median rotation of a central-difference
    tangent from one point to the next is 19.5. This fixture reproduces 26.6, so a test
    of the tangent fix has something real to bite on.

    The slopes are deliberately not 1/2 of each other and not 1: an *evenly* periodic
    staircase is the one case a central difference handles perfectly, because the wobble
    cancels between i-1 and i+1. Real rasterisation is not periodic, and a fixture that
    was would have quietly proved nothing.
    """
    xs = np.arange(x0, x1)
    ys = cy + np.rint(slope_y * (xs - x0)).astype(int)
    zs = cz + np.rint(slope_z * (xs - x0)).astype(int)
    return _graph_through(frame, np.stack([xs, ys, zs], axis=1), radius_um)


def wobbly_graph(frame, x0, x1, radius_um, *, cy, cz, wobble=1) -> EditableGraph:
    """A skeleton that jitters a voxel either side of a straight axis, aperiodically.

    What thinning leaves *inside* a straight tube: the line is in the right place to
    within a voxel but never smooth. Unlike :func:`staircase_graph` this stays inside a
    cylinder of a few voxels' radius, so re-centring can be asked to pull it onto the
    axis and be checked against the answer.
    """
    xs = np.arange(x0, x1)
    jitter = np.rint(wobble * np.sin(0.7 * np.arange(len(xs)))).astype(int)
    ijk = np.stack([xs, cy + jitter, np.full_like(xs, cz)], axis=1)
    return _graph_through(frame, ijk, radius_um)


def graph_from(nodes_xyz, edges) -> EditableGraph:
    """Build a graph from node positions and ``(a, b, n_points, radius_um)`` edges."""
    nodes, points, segments = {}, {}, []
    for i, p in enumerate(nodes_xyz):
        nodes[i] = (float(p[0]), float(p[1]), float(p[2]), 0)
    pid = 0
    for sid, (a, b, n, r) in enumerate(edges):
        ids = []
        for t in np.linspace(0.0, 1.0, n):
            p = np.asarray(nodes_xyz[a], float) * (1 - t) + np.asarray(nodes_xyz[b], float) * t
            points[pid] = (float(p[0]), float(p[1]), float(p[2]), float(r))
            ids.append(pid)
            pid += 1
        segments.append({"id": sid, "node1": a, "node2": b, "point_ids": ids})
    return EditableGraph(Triple(nodes, points, segments))
