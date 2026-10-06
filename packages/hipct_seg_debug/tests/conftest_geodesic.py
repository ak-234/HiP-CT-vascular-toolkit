"""Fixtures for the geodesic reconnector: broken vessels of every shape it claims.

The point of building these rather than testing on the real dataset is that each
one has a *known* right answer. "Did the repair connect the two pieces" is not a
question LADAF-2024-28 can settle -- it has no ground truth for which of its 4000
free ends belong together -- but a slit deliberately cut in two has exactly one.

Shapes matter as much as gaps here. A round tube, a ribbon and a one-voxel slit
are the three cross-sections an ex-vivo coronary actually presents, and a
reconnector that only works on the first is the failure this package exists to
avoid, so all three are fixtures rather than one.
"""

from __future__ import annotations

import numpy as np

from hipct_seg_debug.edit.adapter import Triple
from hipct_seg_debug.edit.graphmodel import EditableGraph
from hipct_seg_debug.edit.maskedit import MaskEdits, MaskSource

from .conftest_geometry import SPACING, cylinder, make_frame, slit

SHAPE = (40, 40, 60)
CY = CZ = 20


class FakeLattice:
    """The slice-at-a-time surface ``ByteRLELattice`` presents, over a real array."""

    def __init__(self, volume):
        self.volume = np.asarray(volume, dtype=np.uint8)
        self.nz, self.ny, self.nx = self.volume.shape
        self.dims = np.array([self.nx, self.ny, self.nz], dtype=np.int64)

    def slice_z(self, k):
        if not 0 <= k < self.nz:
            raise IndexError(k)
        return self.volume[k].copy()

    def decode_sequential(self, n_slices):
        return self.volume[:n_slices].copy()


def mask_source(volume) -> MaskSource:
    """A `MaskSource` with a live edit store, which is what a repair writes into."""
    lattice = FakeLattice(volume)
    return MaskSource(lattice, MaskEdits(ny=lattice.ny, nx=lattice.nx,
                                         nz=lattice.nz, source="<test>"))


def decode(source) -> np.ndarray:
    """The mask as the rest of the toolkit now sees it, edits composited in."""
    nz = int(np.asarray(source.dims)[2])
    return np.stack([np.asarray(source.slice_z(k)) for k in range(nz)])


def ribbon(shape, half_y, half_z, x0, x1, *, cy=CY, cz=CZ) -> np.ndarray:
    """A flattened but not fully collapsed vessel: an ellipse wider than it is tall."""
    nz, ny, nx = shape
    zz, yy, xx = np.ogrid[:nz, :ny, :nx]
    ellipse = ((yy - cy) / half_y) ** 2 + ((zz - cz) / half_z) ** 2 <= 1.0
    return (ellipse & (xx >= x0) & (xx < x1)).astype(np.uint8)


def curved_tube(shape, radius, x0, x1, *, amplitude=6.0, cy=CY, cz=CZ) -> np.ndarray:
    """A tube that bends in y, so a chord between its ends leaves the lumen.

    The case a straight-line proposer gets wrong and a path search should not: the
    correct route bulges, and cutting the chord crosses tissue.
    """
    nz, ny, nx = shape
    out = np.zeros(shape, dtype=np.uint8)
    zz, yy = np.ogrid[:nz, :ny]
    for x in range(max(x0, 0), min(x1, nx)):
        t = (x - x0) / max(x1 - x0 - 1, 1)
        centre = cy + amplitude * np.sin(np.pi * t)
        disc = (zz - cz) ** 2 + (yy - centre) ** 2 <= radius * radius
        out[:, :, x] = disc
    return out


def _merge(*triples) -> EditableGraph:
    """Splice several single-segment graphs into one graph of several components."""
    nodes: dict = {}
    points: dict = {}
    segments: list = []
    node_off = point_off = seg_off = 0
    for triple in triples:
        for k, v in triple.nodes.items():
            nodes[k + node_off] = v
        for k, v in triple.points.items():
            points[k + point_off] = v
        for s in triple.segments:
            segments.append({
                "id": s["id"] + seg_off,
                "node1": s["node1"] + node_off, "node2": s["node2"] + node_off,
                "point_ids": [p + point_off for p in s["point_ids"]],
            })
        node_off = max(nodes) + 1
        point_off = max(points) + 1
        seg_off = max(s["id"] for s in segments) + 1
    return EditableGraph(Triple(nodes, points, segments))


def _run_triple(frame, ijk, radius_um) -> Triple:
    xyz = np.asarray(frame.seg_to_um(np.asarray(ijk, dtype=np.float64)))
    points = {i: (float(p[0]), float(p[1]), float(p[2]), float(radius_um))
              for i, p in enumerate(xyz)}
    nodes = {0: (*xyz[0], 0), 1: (*xyz[-1], 0)}
    return Triple(nodes, points,
                  [{"id": 0, "node1": 0, "node2": 1,
                    "point_ids": list(range(len(xyz)))}])


def axis_run(frame, x0, x1, radius_um, *, cy=CY, cz=CZ) -> Triple:
    """A straight centreline run along x, as a mergeable single-segment triple."""
    xs = np.arange(x0, x1)
    return _run_triple(frame, np.stack(
        [xs, np.full_like(xs, cy), np.full_like(xs, cz)], axis=1), radius_um)


def curved_run(frame, x0, x1, radius_um, *, amplitude=6.0, cy=CY, cz=CZ) -> Triple:
    """The centreline of :func:`curved_tube`, over part of its span."""
    xs = np.arange(x0, x1)
    t = (xs - x0) / max(x1 - x0 - 1, 1)
    ys = np.rint(cy + amplitude * np.sin(np.pi * t)).astype(int)
    return _run_triple(frame, np.stack(
        [xs, ys, np.full_like(xs, cz)], axis=1), radius_um)


def broken_graph(frame, left, right, radius_um=20.0, *, cy=CY, cz=CZ) -> EditableGraph:
    """Two disconnected straight runs: the standard two-free-ends-facing situation."""
    return _merge(axis_run(frame, *left, radius_um, cy=cy, cz=cz),
                  axis_run(frame, *right, radius_um, cy=cy, cz=cz))


def scene(volume, graph_builder, *, radius_um=20.0, shape=SHAPE):
    """``(graph, index, frame, source)`` for one fixture, ready to plan on."""
    from hipct_seg_debug.edit.reconnect.geodesic import components

    frame = make_frame(shape)
    source = mask_source(np.asarray(volume, dtype=np.uint8))
    index = components.build(source)
    return graph_builder(frame), index, frame, source




# ------------------------------------------------------------ raw greyscale
#
# The wavefront tests need what the mask-only fixtures above cannot give: an image
# in which a vessel is *visible* where the segmentation has nothing. HiP-CT lumen is
# dark and the myocardium around it bright, so these are dark-lumen phantoms.

LUMEN, WALL = 10.0, 100.0


def greyscale(mask, *, lumen=LUMEN, wall=WALL, noise=4.0, blur=0.8, seed=0,
              faint=None) -> np.ndarray:
    """A dark-lumen greyscale over a mask, blurred and noised like a scan.

    `faint`, if given, is a second boolean mask painted at ``lumen + faint * (wall -
    lumen)`` -- lumen the segmentation missed, still darker than tissue. That is
    the pinched slit: a vessel the image shows and the mask does not.
    """
    rng = np.random.default_rng(seed)
    image = np.full(np.shape(mask), float(wall))
    if faint is not None:
        level, where = faint
        image[np.asarray(where, dtype=bool)] = lumen + level * (wall - lumen)
    image[np.asarray(mask, dtype=bool)] = float(lumen)
    if blur:
        from scipy import ndimage

        image = ndimage.gaussian_filter(image, blur)
    image += rng.normal(0.0, noise, size=image.shape)
    return image.astype(np.float32)


def ribbon_gap(shape, half_y, half_z, x0, x1, gap, *, faint=0.5, cy=CY, cz=CZ):
    """A ribbon with its mask missing over ``gap = (a, b)`` but its image not.

    Returns ``(mask, image)``: the mask is the ribbon minus the gap, the image is
    the dark ribbon throughout, at `faint` of the wall level inside the gap.
    """
    whole = ribbon(shape, half_y, half_z, x0, x1, cy=cy, cz=cz).astype(bool)
    a, b = gap
    mask = whole.copy()
    mask[:, :, a:b] = False
    trace = whole & ~mask
    return mask.astype(np.uint8), greyscale(mask, faint=(faint, trace))


class FakeStack:
    """The windowed-read surface of ``TiffStack``, over a segmentation-grid image.

    :func:`make_frame` puts the raw grid at twice the segmentation resolution, so
    the image is upsampled by two on every axis to sit where the frame expects it.
    """

    def __init__(self, image_seg_grid):
        img = np.asarray(image_seg_grid, dtype=np.float32)
        self.volume = np.repeat(np.repeat(np.repeat(img, 2, 0), 2, 1), 2, 2)
        self.shape = self.volume.shape
        self.nominal_voxel_um = SPACING / 2

    def read_stack_window(self, z0, z1, y0, y1, x0, x1):
        return self.volume[z0:z1, y0:y1, x0:x1]

    def read_slice(self, z):
        return self.volume[z]


__all__ = [
    "SHAPE", "SPACING", "CY", "CZ", "FakeLattice", "mask_source", "decode",
    "ribbon", "curved_tube", "curved_run", "axis_run", "broken_graph", "scene",
    "cylinder", "slit", "make_frame",
    "LUMEN", "WALL", "greyscale", "ribbon_gap", "FakeStack",
]
