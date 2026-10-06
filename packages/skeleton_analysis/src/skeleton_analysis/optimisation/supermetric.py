"""The skeleton super metric of Walsh et al. (2024), Eq. 10.

    C.L. Walsh, M. Berg, H. West, N.A. Holroyd, S. Walker-Samuel, R.J. Shipley,
    "Reconstructing microvascular network skeletons from 3D images: What is the
    ground truth?", Computers in Biology and Medicine 171 (2024) 108140.

This is the single implementation of the metric in the toolkit. Other packages
(e.g. ``hipct_seg_debug``) adapt their own graph types to :class:`SkeletonGraph`
and call it, choosing a preset of :class:`SuperMetricOptions`.

    M_S = |V_I - V_S| / V_I
        + |cc_I - cc_S| / cc_I
        + |chi_I - chi_S| / chi_I
        + (1 - cl_S) / cl_S**3
        + (1 - B_S) / B_S**2

The subscript I is the binary (segmented) image, the gold standard, and S the
skeleton (spatial graph) being assessed:

    V    total network volume
    cc   number of connected components (26-connectivity / sub-graphs)
    chi  local Euler characteristic of the largest connected component,
         chi = 2 - chi_classical (always >= 1 for a connected object). The
         Supplementary's ``-chi_classical - 2`` is a typo: it is negative for a tree.
    cl   cl-sensitivity: fraction of the rasterised skeleton centreline that
         lies inside the binary image
    B    DICE score of bifurcation points against a reference, in a sub-volume

Presets
-------
:data:`PAPER` reproduces the paper (the default everywhere). :data:`CORONARY`
is the ex vivo coronary pipeline's variant: automatic bifurcation reference,
no sub-volume, a tree as the chi reference, and interpolated points excluded.
Results made with anything other than :data:`PAPER` are flagged
``paper_comparable=False``.

Fixed by the paper and deliberately *not* options: the weights; the largest
component chosen by volume in both image and graph; an isolated node counting
as a component; and a cl or B of zero making the metric infinite.

Bifurcation matching
--------------------
Bifurcations are nodes of degree >= 3. Reference and skeleton bifurcations are
paired one-to-one; with ``matching="hungarian"`` (default) by
``scipy.optimize.linear_sum_assignment``, which maximises the number of matched
pairs and, among those, minimises the total distance. A pair is allowed if the
distance is within ``tolerance_value`` x the local vessel radius at the
reference bifurcation (radius from the Euclidean distance map of the binary
image), or within a fixed distance with ``tolerance="fixed"``.

Because matching is one-to-one, several skeleton bifurcations clustered around
one true bifurcation give one TP and the rest FPs. False positives are split
into ``fp_duplicate`` (within tolerance of a reference bifurcation, i.e.
clustered / spurious extras) and ``fp_isolated`` (not near any).

With a sub-volume, skeleton bifurcations up to one tolerance outside the box may
be matched (so a correct bifurcation straddling the edge is not FP + FN), but
unmatched ones only count as FPs if they are inside the box.

Conventions
-----------
* Graph coordinates are physical (x, y, z), in the same units as voxel_size.
* Images are arrays indexed [z, y, x] (as returned by ``tifffile.imread``).
* ``voxel_size`` and ``origin`` are (x, y, z). ``origin`` is the physical
  coordinate of the centre of voxel [0, 0, 0].
* Any non-zero voxel of the binary image is foreground.

Usage
-----
Command line::

    python -m skeleton_analysis.optimisation.supermetric binary.tif skel_a.am \\
        --gt manual_skeleton.am --voxel-size 50 -o results.csv

Python::

    from skeleton_analysis.optimisation import supermetric as sm
    binary = sm.read_image("binary.tif")
    gt = sm.load_graph("manual_skeleton.am")     # or a CSV folder, or SkeletonGraph(...)
    ref = sm.reference_metrics(binary, voxel_size=50, gt_graph=gt)   # once
    for path in skeleton_paths:
        res = sm.super_metric(sm.load_graph(path), ref)
        print(path, res.total, res.terms)
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
import types
import warnings
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
from scipy import ndimage
from scipy.optimize import linear_sum_assignment
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial.distance import cdist

LARGE_MATCHING_PAIRS = 25_000_000  # reference x candidate pairs above which to warn about memory
RADIUS_FIELD_NAMES = ("thickness", "radius", "Radius", "Thickness")


# --------------------------------------------------------------------------
# Options
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class SuperMetricOptions:
    """How the super metric is computed. The defaults are the paper's.

    bifurcation_reference  "manual": a ground-truth skeleton or annotated points
                           supplied by the user (the paper).
                           "auto": junctions of a skeleton of the binary image.
    matching               "hungarian": optimal one-to-one pairing (the paper's
                           intent; order independent).
                           "greedy": each skeleton bifurcation in turn takes the
                           nearest unused reference point, as the original MATLAB.
                           TODO: remove "greedy" once agreed with Akash; it is
                           kept only for comparison with earlier results.
    tolerance              "radius": tolerance_value x local vessel radius.
                           "fixed": tolerance_value in physical units.
    tolerance_value        the multiplier (radius) or distance (fixed).
    use_bbox               restrict B to a sub-volume (the paper annotates one).
    chi_reference          "image": the segmentation's own local Euler number.
                           "tree": a tree (chi_classical = 1), for anatomy known
                           to have no loops, where loops are artefacts.
    exclude_invalid_points skip sub-segments touching a point flagged invalid
                           (e.g. interpolated by Avizo) in volume and cl.
    """

    bifurcation_reference: str = "manual"
    matching: str = "hungarian"
    tolerance: str = "radius"
    tolerance_value: float = 1.5
    use_bbox: bool = True
    chi_reference: str = "image"
    exclude_invalid_points: bool = False

    def __post_init__(self):
        for name, allowed in [("bifurcation_reference", ("manual", "auto")),
                              ("matching", ("hungarian", "greedy")),
                              ("tolerance", ("radius", "fixed")),
                              ("chi_reference", ("image", "tree"))]:
            if getattr(self, name) not in allowed:
                raise ValueError(f"{name} must be one of {allowed}, got {getattr(self, name)!r}")
        if self.tolerance_value <= 0:
            raise ValueError("tolerance_value must be > 0")

    @property
    def paper_comparable(self) -> bool:
        return self == PAPER


PAPER = SuperMetricOptions()
CORONARY = SuperMetricOptions(
    bifurcation_reference="auto",
    use_bbox=False,
    chi_reference="tree",
    exclude_invalid_points=True,
)
PRESETS = {"paper": PAPER, "coronary": CORONARY}


# --------------------------------------------------------------------------
# Skeleton graph
# --------------------------------------------------------------------------

@dataclass
class SkeletonGraph:
    """A skeleton as a spatial graph, independent of the software that made it.

    Input specification
    -------------------
    All coordinates and radii are in the same physical units as the image
    ``voxel_size`` (e.g. um), with coordinates in (x, y, z) order.

    nodes        (N, 3) float. Node coordinates: branch points, end points and
                 any other graph vertices. A node with no edges is allowed and
                 counts as its own connected component.
    edges        (E, 2) int. For each segment, the indices (0-based row numbers
                 of ``nodes``) of the two nodes it joins. Self-loops (i, i) and
                 repeated node pairs are allowed (they are real loops).
    edge_points  list of E arrays, each (k, 3) float with k >= 2. The polyline
                 of each segment, running from one of its end nodes to the
                 other and *including* both end nodes.
    edge_radii   list of E arrays, each (k,) float >= 0, or None. The vessel
                 radius at each polyline point. Needed for the volume term.
    point_valid  list of E arrays, each (k,) bool, or None. False marks a point
                 that was not skeletonised (e.g. interpolated); used only with
                 ``exclude_invalid_points``.

    Ways to build one
    -----------------
    SkeletonGraph(nodes, edges, edge_points, edge_radii)       one polyline per edge
    SkeletonGraph.from_flat(nodes, edges, points, num_edge_points, radii)
    SkeletonGraph.from_csv(folder)                             nodes.csv, edges.csv, points.csv
    SkeletonGraph.from_amira(path)                             Amira/Avizo ASCII .am
    SkeletonGraph.from_spatial_graph(graph)                    skeleton_analysis.io.amira.SpatialGraph
    """

    nodes: np.ndarray
    edges: np.ndarray
    edge_points: list
    edge_radii: list | None = None
    point_valid: list | None = None

    def __post_init__(self):
        self.nodes = np.asarray(self.nodes, dtype=float).reshape(-1, 3)
        edges = np.asarray(self.edges, dtype=float).reshape(-1, 2)
        if not np.all(edges == np.round(edges)):
            raise ValueError("edges must contain integer node indices")
        self.edges = edges.astype(np.int64)
        self.edge_points = [np.asarray(p, dtype=float).reshape(-1, 3) for p in self.edge_points]
        if self.edge_radii is not None:
            self.edge_radii = [np.asarray(r, dtype=float).ravel() for r in self.edge_radii]
        if self.point_valid is not None:
            self.point_valid = [np.asarray(v, dtype=bool).ravel() for v in self.point_valid]
        self._validate()

    def _validate(self):
        n, e = len(self.nodes), len(self.edges)
        if not np.all(np.isfinite(self.nodes)):
            raise ValueError("nodes contain NaN or inf")
        if e and (self.edges.min() < 0 or self.edges.max() >= n):
            raise ValueError(f"edges refer to node indices outside 0..{n - 1} "
                             "(indices must be 0-based)")
        if len(self.edge_points) != e:
            raise ValueError(f"{len(self.edge_points)} polylines given for {e} edges")
        short = [i for i, p in enumerate(self.edge_points) if len(p) < 2]
        if short:
            raise ValueError(f"{len(short)} edge(s) have fewer than 2 polyline points, "
                             f"e.g. edge {short[0]}")
        for name in ("edge_radii", "point_valid"):
            per_edge = getattr(self, name)
            if per_edge is None:
                continue
            if len(per_edge) != e:
                raise ValueError(f"{len(per_edge)} {name} arrays given for {e} edges")
            bad = [i for i, (p, r) in enumerate(zip(self.edge_points, per_edge))
                   if len(p) != len(r)]
            if bad:
                raise ValueError(f"{len(bad)} edge(s) have a different number of {name} and "
                                 f"points, e.g. edge {bad[0]}")
        if self.edge_radii is not None and e:
            radii = np.concatenate(self.edge_radii)
            if not np.all(np.isfinite(radii)) or np.any(radii < 0):
                raise ValueError("radii must be finite and >= 0")
        if e:
            # Polyline ends should sit on the edge's nodes (either direction).
            first = np.array([p[0] for p in self.edge_points])
            last = np.array([p[-1] for p in self.edge_points])
            a, b = self.nodes[self.edges[:, 0]], self.nodes[self.edges[:, 1]]
            gap = np.minimum(
                np.maximum(np.linalg.norm(first - a, axis=1), np.linalg.norm(last - b, axis=1)),
                np.maximum(np.linalg.norm(first - b, axis=1), np.linalg.norm(last - a, axis=1)))
            extent = np.ptp(np.vstack([self.nodes, first, last]), axis=0).max()
            off = int(np.sum(gap > 1e-3 * max(extent, 1e-12)))
            if off:
                warnings.warn(f"{off} edge polyline(s) do not start/end at their edge's nodes "
                              f"(largest gap {gap.max():.3g}); check node indexing.",
                              stacklevel=3)

    # ---- constructors ----------------------------------------------------

    @classmethod
    def from_flat(cls, nodes, edges, points, num_edge_points, radii=None, valid=None):
        """Build from flat arrays, the layout used by Amira and many tools:
        ``points`` (P, 3) holds every polyline point, edge after edge, and
        ``num_edge_points`` (E,) says how many belong to each edge.
        ``radii`` (P,) and ``valid`` (P,) are optional per-point values."""
        points = np.asarray(points, dtype=float).reshape(-1, 3)
        num = np.asarray(num_edge_points).astype(np.int64).ravel()
        if num.sum() != len(points):
            raise ValueError(f"num_edge_points sums to {num.sum()} but {len(points)} points given")
        splits = np.cumsum(num)[:-1]

        def per_edge(values, name):
            if values is None:
                return None
            values = np.asarray(values).ravel()
            if len(values) != len(points):
                raise ValueError(f"{len(values)} {name} given for {len(points)} points")
            return np.split(values, splits)

        return cls(nodes, edges, np.split(points, splits),
                   per_edge(None if radii is None else np.asarray(radii, float), "radii"),
                   per_edge(None if valid is None else np.asarray(valid, bool), "valid flags"))

    @classmethod
    def from_csv(cls, folder):
        """Build from a folder of three CSV files (with header rows):

        nodes.csv   x,y,z                one row per node; row order = node index
        edges.csv   node1,node2          one row per edge; row order = edge index
        points.csv  edge,x,y,z[,radius]  polyline points, listed in order along
                                         each edge; ``edge`` is the edge index
        """
        folder = Path(folder)
        nodes = _read_csv_columns(folder / "nodes.csv", ["x", "y", "z"])
        edges = _read_csv_columns(folder / "edges.csv", ["node1", "node2"])
        pts = _read_csv_columns(folder / "points.csv", ["edge", "x", "y", "z"],
                                optional=["radius"])
        edge_id = pts[:, 0].astype(np.int64)
        if np.any(np.diff(edge_id) < 0):
            # keep the within-edge order, group points by edge
            order = np.argsort(edge_id, kind="stable")
            pts, edge_id = pts[order], edge_id[order]
        num = np.bincount(edge_id, minlength=len(edges))
        if len(num) > len(edges):
            raise ValueError("points.csv refers to edge indices beyond edges.csv")
        radii = pts[:, 4] if pts.shape[1] > 4 else None
        return cls.from_flat(nodes, edges, pts[:, 1:4], num, radii)

    @classmethod
    def from_spatial_graph(cls, graph, radius_field=None):
        """Build from a :class:`skeleton_analysis.io.amira.SpatialGraph`."""
        if radius_field is None:
            radius_field = next((f for f in RADIUS_FIELD_NAMES if f in graph.point_fields), None)
        radii = None if radius_field is None else graph.point_fields[radius_field]
        return cls.from_flat(graph.vertex_coords, graph.edge_connectivity, graph.point_coords,
                             graph.num_edge_points, radii)

    @classmethod
    def from_amira(cls, path, radius_field=None):
        """Read an Amira/Avizo ASCII spatial graph (.am)."""
        from skeleton_analysis.io.amira import read_amira

        return cls.from_spatial_graph(read_amira(path), radius_field)

    # ---- measures --------------------------------------------------------

    @property
    def degrees(self):
        """Node degree; a self-loop counts twice."""
        return np.bincount(self.edges.ravel(), minlength=len(self.nodes))

    def bifurcations(self):
        """Coordinates of nodes with degree >= 3."""
        return self.nodes[self.degrees >= 3]

    def _subsegment_keep(self, i, exclude_invalid):
        n = len(self.edge_points[i]) - 1
        if not exclude_invalid or self.point_valid is None:
            return np.ones(n, dtype=bool)
        v = self.point_valid[i]
        return v[:-1] & v[1:]

    def edge_volumes(self, exclude_invalid=False):
        """Volume of each segment: the sum of its sub-segments, each a cylinder
        whose length is the distance between consecutive points and whose
        radius is the mean of the two point radii."""
        if self.edge_radii is None:
            raise ValueError("Graph has no radii; cannot compute volume")
        vols = np.empty(len(self.edges))
        for i, (p, r) in enumerate(zip(self.edge_points, self.edge_radii)):
            keep = self._subsegment_keep(i, exclude_invalid)
            length = np.linalg.norm(np.diff(p, axis=0), axis=1)[keep]
            radius = 0.5 * (r[:-1] + r[1:])[keep]
            vols[i] = np.sum(np.pi * radius * radius * length)
        return vols

    def volume(self, exclude_invalid=False):
        return float(self.edge_volumes(exclude_invalid).sum())

    def component_labels(self):
        """(n_components, label per node). Isolated nodes are their own component."""
        n = len(self.nodes)
        adj = coo_matrix(
            (np.ones(len(self.edges)), (self.edges[:, 0], self.edges[:, 1])), shape=(n, n)
        )
        return connected_components(adj, directed=False)

    def n_components(self):
        return int(self.component_labels()[0])

    def largest_component_euler_classical(self):
        """nodes - segments of the largest sub-graph by volume (by centreline
        length if the graph has no radii). 1 for a tree, minus one per loop."""
        n_comp, labels = self.component_labels()
        if len(self.edges) == 0:
            return 1 if len(self.nodes) else 0
        edge_comp = labels[self.edges[:, 0]]
        if self.edge_radii is not None:
            size = self.edge_volumes()
        else:
            size = np.array([np.linalg.norm(np.diff(p, axis=0), axis=1).sum()
                             for p in self.edge_points])
        comp_size = np.bincount(edge_comp, weights=size, minlength=n_comp)
        largest = int(np.argmax(comp_size))
        n_nodes = int(np.sum(labels == largest))
        n_edges = int(np.sum(edge_comp == largest))
        return n_nodes - n_edges

    def largest_component_local_euler(self):
        """Local Euler characteristic, 2 - (nodes - segments), of the largest sub-graph."""
        return local_euler(self.largest_component_euler_classical())


def local_euler(chi_classical):
    """``2 - chi_classical``: strictly positive for any connected graph or solid."""
    return 2 - chi_classical


# --------------------------------------------------------------------------
# I/O
# --------------------------------------------------------------------------

def _read_csv_columns(path, required, optional=()):
    """Read named numeric columns from a CSV with a header row -> (n, k) array."""
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        header = [h.strip() for h in (reader.fieldnames or [])]
        missing = [c for c in required if c not in header]
        if missing:
            raise ValueError(f"{path}: missing column(s) {missing}; found {header}")
        cols = list(required) + [c for c in optional if c in header]
        rows = [[float(row[c]) for c in cols]
                for row in ({k.strip(): v for k, v in r.items()} for r in reader)]
    return np.array(rows, dtype=float).reshape(-1, len(cols))


def read_bifurcation_points(path):
    """Read reference bifurcation coordinates from a CSV with columns x,y,z."""
    return _read_csv_columns(path, ["x", "y", "z"])


def load_graph(path, radius_field=None):
    """Load a skeleton from an Amira .am file or a folder of CSV files."""
    path = Path(path)
    if path.is_dir():
        return SkeletonGraph.from_csv(path)
    if path.suffix.lower() == ".am":
        return SkeletonGraph.from_amira(path, radius_field)
    raise ValueError(f"{path}: expected an Amira .am file or a folder with "
                     "nodes.csv, edges.csv and points.csv")


def read_image(path):
    """Read a 3D TIFF stack as a boolean array (non-zero = foreground)."""
    import tifffile

    return tifffile.imread(path) != 0


# --------------------------------------------------------------------------
# Coordinate helpers
# --------------------------------------------------------------------------

def _xyz(v):
    """Scalar or (x, y, z) sequence -> float array of shape (3,)."""
    return np.broadcast_to(np.asarray(v, dtype=float), (3,)).copy()


def _world_to_index_zyx(pts, voxel_size, origin):
    """(n, 3) physical (x, y, z) -> (n, 3) float voxel indices in (z, y, x)."""
    idx_xyz = (np.asarray(pts, dtype=float) - origin) / voxel_size
    return idx_xyz[:, ::-1]


def _in_bbox(pts, bbox, pad=0.0):
    if bbox is None:
        return np.ones(len(pts), dtype=bool)
    lo = bbox[0::2] - pad
    hi = bbox[1::2] + pad
    return np.all((pts >= lo) & (pts <= hi), axis=1)


# --------------------------------------------------------------------------
# Binary-image measures
# --------------------------------------------------------------------------

_STRUCT_26 = np.ones((3, 3, 3), dtype=bool)


def binary_measures(binary, voxel_size):
    """Volume, number of 26-connected components, and classical Euler number
    (26-connectivity) of the largest component."""
    from skimage.measure import euler_number

    binary = np.asarray(binary) != 0
    labels, n_comp = ndimage.label(binary, structure=_STRUCT_26)
    if n_comp == 0:
        raise ValueError("Binary image is empty")
    sizes = np.bincount(labels.ravel())
    sizes[0] = 0
    largest = int(np.argmax(sizes))
    slc = ndimage.find_objects(labels, max_label=largest)[largest - 1]
    component = np.pad(labels[slc] == largest, 1)
    return dict(
        volume=float(binary.sum() * np.prod(voxel_size)),
        n_components=int(n_comp),
        euler_classical=int(euler_number(component, connectivity=3)),
    )


def local_radius(binary, pts, voxel_size, origin=0.0, margin=16):
    """Distance-map value (physical units) of the binary image at physical points.

    The distance map is computed on a crop around the points, grown until the
    crop margin is larger than every radius found. Points on background give 0.
    """
    binary = np.asarray(binary) != 0
    voxel_size, origin = _xyz(voxel_size), _xyz(origin)
    if len(pts) == 0:
        return np.zeros(0)
    vs_zyx = voxel_size[::-1]
    shape = np.array(binary.shape)
    idx = np.clip(np.rint(_world_to_index_zyx(pts, voxel_size, origin)).astype(np.int64),
                  0, shape - 1)
    while True:
        lo = np.maximum(idx.min(axis=0) - margin, 0)
        hi = np.minimum(idx.max(axis=0) + margin + 1, shape)
        crop = binary[tuple(slice(a, b) for a, b in zip(lo, hi))]
        dist = ndimage.distance_transform_edt(crop, sampling=vs_zyx)
        r = dist[tuple((idx - lo).T)]
        touches_edge = np.any(lo > 0) or np.any(hi < shape)
        if not touches_edge or r.max(initial=0) < (margin - 1) * vs_zyx.min():
            return r
        margin *= 2


def auto_reference_bifurcations(binary, voxel_size, origin=0.0):
    """Reference bifurcations from the binary image's own skeleton (physical x, y, z).

    Junction voxels of a Lee skeleton, clustered so one branch point gives one
    coordinate (:func:`~skeleton_analysis.optimisation.volume_metrics.skeleton_junction_points`).
    """
    from skeleton_analysis.optimisation.volume_metrics import skeleton_junction_points

    lattice = types.SimpleNamespace(volume=np.asarray(binary) != 0,
                                    origin=_xyz(origin), spacing=_xyz(voxel_size))
    return np.asarray(skeleton_junction_points(lattice), dtype=float).reshape(-1, 3)


# --------------------------------------------------------------------------
# Reference (binary-image side, computed once)
# --------------------------------------------------------------------------

@dataclass
class ReferenceMetrics:
    """Everything derived from the binary image and the bifurcation reference."""

    binary: np.ndarray = field(repr=False)
    voxel_size: np.ndarray
    origin: np.ndarray
    volume: float
    n_components: int
    euler_classical: int
    options: SuperMetricOptions = PAPER
    bbox: np.ndarray | None = None              # (xmin, xmax, ymin, ymax, zmin, zmax)
    ref_bifurcations: np.ndarray | None = None  # (n, 3), inside bbox if one is used
    ref_radius: np.ndarray | None = None        # (n,) distance-map radius

    @property
    def local_euler(self):
        return local_euler(1 if self.options.chi_reference == "tree" else self.euler_classical)

    @property
    def ref_tolerance(self):
        if self.ref_bifurcations is None:
            return None
        return bifurcation_tolerance(self.ref_radius, self.options)


def bifurcation_tolerance(radius, options):
    """Match tolerance per reference bifurcation for the given options."""
    radius = np.asarray(radius, dtype=float)
    if options.tolerance == "fixed":
        return np.full(len(radius), float(options.tolerance_value))
    return options.tolerance_value * radius


def reference_metrics(binary, voxel_size, origin=0.0, gt_graph=None, bbox=None,
                      options: SuperMetricOptions = PAPER, gt_bifurcations=None):
    """Compute the binary-image side of the super metric once.

    binary            3D array [z, y, x]; non-zero = vessel
    voxel_size        scalar or (x, y, z), physical units per voxel
    origin            scalar or (x, y, z), physical coordinate of voxel [0,0,0]
    gt_graph          ground-truth skeleton for the bifurcation DICE; its
                      bifurcations are nodes of degree >= 3 (manual reference)
    gt_bifurcations   alternative to ``gt_graph``: (n, 3) annotated bifurcation
                      coordinates (x, y, z), from any tool (manual reference)
    bbox              (xmin, xmax, ymin, ymax, zmin, zmax) sub-volume for B in
                      physical units; defaults to the extent of the reference.
                      Ignored when ``options.use_bbox`` is False.
    options           a :class:`SuperMetricOptions`, e.g. :data:`PAPER`

    With a manual reference and neither ``gt_graph`` nor ``gt_bifurcations``,
    the bifurcation term is skipped.
    """
    if gt_graph is not None and gt_bifurcations is not None:
        raise ValueError("give gt_graph or gt_bifurcations, not both")
    binary = np.asarray(binary) != 0
    if binary.ndim != 3:
        raise ValueError("binary image must be 3D")
    voxel_size, origin = _xyz(voxel_size), _xyz(origin)
    ref = ReferenceMetrics(binary=binary, voxel_size=voxel_size, origin=origin,
                           options=options, **binary_measures(binary, voxel_size))

    if options.bifurcation_reference == "auto":
        if gt_graph is not None or gt_bifurcations is not None:
            raise ValueError("bifurcation_reference='auto' takes no gt_graph/gt_bifurcations")
        ref_bif = auto_reference_bifurcations(binary, voxel_size, origin)
        extent_pts = ref_bif
    elif gt_graph is not None:
        ref_bif = gt_graph.bifurcations()
        extent_pts = np.vstack([gt_graph.nodes] + gt_graph.edge_points)
    elif gt_bifurcations is not None:
        ref_bif = np.asarray(gt_bifurcations, dtype=float).reshape(-1, 3)
        extent_pts = ref_bif
        if bbox is None and options.use_bbox:
            warnings.warn("No bbox given with reference bifurcation points: using the extent "
                          "of the points, which may be smaller than the annotated region.",
                          stacklevel=2)
    else:
        return ref

    if options.use_bbox:
        if bbox is None and len(extent_pts):
            lo, hi = extent_pts.min(axis=0), extent_pts.max(axis=0)
            bbox = np.column_stack([lo, hi]).ravel()
        if bbox is not None:
            bbox = np.asarray(bbox, dtype=float).ravel()
            if bbox.size != 6:
                raise ValueError("bbox must be (xmin, xmax, ymin, ymax, zmin, zmax)")
            ref_bif = ref_bif[_in_bbox(ref_bif, bbox)]
    else:
        bbox = None

    radius = local_radius(binary, ref_bif, voxel_size, origin)
    floor = voxel_size.min()
    n_bg = int(np.sum(radius < floor))
    if n_bg and options.tolerance == "radius":
        warnings.warn(f"{n_bg} reference bifurcation(s) lie on/next to background in the "
                      f"binary image; their radius is floored to one voxel ({floor}).",
                      stacklevel=2)
    radius = np.maximum(radius, floor)
    if len(ref_bif) == 0:
        warnings.warn("No reference bifurcations" + (" inside the bounding box." if bbox is not None
                                                     else "."), stacklevel=2)
    ref.bbox, ref.ref_bifurcations, ref.ref_radius = bbox, ref_bif, radius
    return ref


# --------------------------------------------------------------------------
# Skeleton-side measures
# --------------------------------------------------------------------------

def rasterise_centreline(graph, shape, voxel_size, origin=0.0, exclude_invalid=False):
    """Voxelise all segment polylines with 3D Bresenham lines.

    Returns (voxels_in_image (m, 3) [z, y, x] int array, n_outside) where
    n_outside is the number of distinct line voxels falling outside the image.
    With ``exclude_invalid``, sub-segments touching an invalid point are skipped.
    """
    vox = rasterise_voxels(graph, voxel_size, origin, exclude_invalid)
    inside = np.all((vox >= 0) & (vox < np.array(shape)), axis=1)
    return vox[inside], int(np.sum(~inside))


def rasterise_voxels(graph, voxel_size, origin=0.0, exclude_invalid=False):
    """All distinct voxels [z, y, x] covered by the centreline's Bresenham lines,
    without clipping to any image (for callers that sample the mask themselves)."""
    voxel_size, origin = _xyz(voxel_size), _xyz(origin)
    p0s, p1s = [], []
    for i, pts in enumerate(graph.edge_points):
        v = np.floor(_world_to_index_zyx(pts, voxel_size, origin) + 0.5).astype(np.int64)
        keep = graph._subsegment_keep(i, exclude_invalid)
        p0s.append(v[:-1][keep])
        p1s.append(v[1:][keep])
    if not p0s:
        return np.zeros((0, 3), dtype=np.int64)
    p0 = np.vstack(p0s)
    d = np.vstack(p1s) - p0
    # Integer endpoints, n = steps along the dominant axis, one voxel per step:
    # this reproduces the Bresenham digital line.
    n = np.abs(d).max(axis=1)
    reps = n + 1
    seg = np.repeat(np.arange(len(p0)), reps)
    k = np.arange(reps.sum()) - np.repeat(np.cumsum(reps) - reps, reps)
    t = k / np.maximum(n[seg], 1)
    vox = p0[seg] + np.floor(d[seg] * t[:, None] + 0.5).astype(np.int64)
    return np.unique(vox, axis=0)


def cl_sensitivity(graph, binary, voxel_size, origin=0.0, exclude_invalid=False):
    """Fraction of rasterised centreline voxels lying inside the binary image,
    sum(V_I * l_S) / sum(l_S). Line voxels outside the image count as misses."""
    binary = np.asarray(binary) != 0
    vox, n_out = rasterise_centreline(graph, binary.shape, voxel_size, origin, exclude_invalid)
    total = len(vox) + n_out
    if total == 0:
        return float("nan")
    return float(binary[tuple(vox.T)].sum() / total)


@dataclass
class BifurcationResult:
    dice: float          # NaN when there are no bifurcations on either side
    tp: int
    fp: int
    fn: int
    fp_duplicate: int
    fp_isolated: int
    matched_ref: np.ndarray = field(repr=False)       # indices into the reference points
    matched_skeleton: np.ndarray = field(repr=False)  # coordinates of matched skeleton nodes
    false_positives: np.ndarray = field(repr=False)   # coordinates of FP skeleton nodes


def match_bifurcations(candidates, reference, tolerance, bbox=None, matching="hungarian"):
    """Pair skeleton bifurcations with reference bifurcations and score them.

    candidates  (m, 3) skeleton bifurcation coordinates
    reference   (n, 3) reference bifurcation coordinates
    tolerance   (n,) maximum allowed distance for each reference point
    bbox        optional sub-volume; see the module docstring
    matching    "hungarian" or "greedy" (see :class:`SuperMetricOptions`)
    """
    ref = np.asarray(reference, dtype=float).reshape(-1, 3)
    tol = np.asarray(tolerance, dtype=float).ravel()
    max_tol = float(tol.max(initial=0.0))
    bif = np.asarray(candidates, dtype=float).reshape(-1, 3)
    cand = bif[_in_bbox(bif, bbox, pad=max_tol)]
    cand_inside = _in_bbox(cand, bbox)

    if len(ref) * len(cand) > LARGE_MATCHING_PAIRS:
        warnings.warn(
            f"Bifurcation matching builds a {len(ref)} x {len(cand)} distance table "
            f"(~{3 * 8 * len(ref) * len(cand) / 1e9:.1f} GB); consider a smaller bounding box.",
            stacklevel=2)

    matched_r = np.zeros(0, dtype=np.int64)
    matched_c = np.zeros(0, dtype=np.int64)
    if len(ref) and len(cand):
        dist = cdist(ref, cand)
        allowed = dist <= tol[:, None]
        if matching == "hungarian":
            # Disallowed pairs cost more than any sum of allowed distances, so the
            # assignment first maximises the number of allowed pairs, then
            # minimises their total distance.
            big = (dist[allowed].sum() + 1.0) if allowed.any() else 1.0
            r, c = linear_sum_assignment(np.where(allowed, dist, big))
            keep = allowed[r, c]
            matched_r, matched_c = r[keep], c[keep]
        elif matching == "greedy":
            # TODO: remove once agreed with Akash (kept to compare with earlier results).
            available = np.ones(len(ref), dtype=bool)
            pairs = []
            for j in range(len(cand)):
                if not available.any():
                    break
                idx = np.flatnonzero(available)
                k = idx[int(np.argmin(dist[idx, j]))]
                if allowed[k, j]:
                    available[k] = False
                    pairs.append((k, j))
            if pairs:
                matched_r, matched_c = (np.array(x, dtype=np.int64) for x in zip(*pairs))
        else:
            raise ValueError(f"unknown matching {matching!r}")

    is_matched = np.zeros(len(cand), dtype=bool)
    is_matched[matched_c] = True
    fp_mask = cand_inside & ~is_matched
    fps = cand[fp_mask]
    if len(fps) and len(ref):
        duplicate = np.any(cdist(fps, ref) <= tol[None, :], axis=1)
    else:
        duplicate = np.zeros(len(fps), dtype=bool)

    tp = len(matched_r)
    fp = int(fp_mask.sum())
    fn = len(ref) - tp
    denom = 2 * tp + fp + fn
    return BifurcationResult(
        dice=float("nan") if denom == 0 else 2 * tp / denom, tp=tp, fp=fp, fn=fn,
        fp_duplicate=int(duplicate.sum()), fp_isolated=int((~duplicate).sum()),
        matched_ref=matched_r, matched_skeleton=cand[matched_c], false_positives=fps,
    )


def bifurcation_dice(graph, ref: ReferenceMetrics):
    """Bifurcation DICE of ``graph`` against the reference bifurcations in ``ref``."""
    if ref.ref_bifurcations is None:
        raise ValueError("Reference has no bifurcations; pass gt_graph/gt_bifurcations to "
                         "reference_metrics or use bifurcation_reference='auto'")
    return match_bifurcations(graph.bifurcations(), ref.ref_bifurcations, ref.ref_tolerance,
                              ref.bbox, ref.options.matching)


# --------------------------------------------------------------------------
# Super metric
# --------------------------------------------------------------------------

@dataclass
class SuperMetricResult:
    total: float
    terms: dict          # weighted term per measure (NaN = not applicable, dropped from total)
    skeleton: dict       # raw skeleton measures
    reference: dict      # raw binary-image measures
    options: SuperMetricOptions = PAPER
    bifurcation: BifurcationResult | None = None

    @property
    def paper_comparable(self) -> bool:
        return self.options.paper_comparable

    def as_row(self):
        """Flat dict for tables / CSV."""
        row = {f"skel_{k}": v for k, v in self.skeleton.items()}
        row.update({f"term_{k}": v for k, v in self.terms.items()})
        if self.bifurcation is not None:
            b = self.bifurcation
            row.update(bif_tp=b.tp, bif_fp=b.fp, bif_fp_duplicate=b.fp_duplicate,
                       bif_fp_isolated=b.fp_isolated, bif_fn=b.fn)
        row["super_metric"] = self.total
        row["paper_comparable"] = self.paper_comparable
        return row


def relative_term(ref_value, value):
    """``|f_I - f_S| / |f_I|``; NaN when the reference is zero or undefined."""
    if not np.isfinite(ref_value) or ref_value == 0:
        return float("nan")
    return float(abs(ref_value - value) / abs(ref_value))


def overlap_term(score, power):
    """``(1 - s) / s**power`` for a score whose gold-standard value is 1.

    Infinite at s = 0 (the skeleton is entirely wrong); NaN when the score is
    undefined (e.g. no bifurcations on either side: not applicable).
    """
    if not np.isfinite(score):
        return float("nan")
    return float("inf") if score <= 0 else abs(1.0 - score) / score ** power


def combine_terms(terms):
    """Sum the terms. NaN (not applicable) terms are dropped; an infinite term
    makes the total infinite, as the paper intends."""
    good = [t for t in terms.values() if not math.isnan(t)]
    return float(sum(good)) if good else float("nan")


def super_metric(graph, ref: ReferenceMetrics):
    """Super metric (Eq. 10) of a skeleton against a precomputed reference.

    Uses ``ref.options``. Without any bifurcation reference the B term is
    omitted (with a warning).
    """
    opt = ref.options
    excl = opt.exclude_invalid_points
    skel = dict(
        volume=graph.volume(excl),
        n_components=graph.n_components(),
        local_euler=graph.largest_component_local_euler(),
        cl_sensitivity=cl_sensitivity(graph, ref.binary, ref.voxel_size, ref.origin, excl),
    )
    terms = dict(
        volume=relative_term(ref.volume, skel["volume"]),
        n_components=relative_term(ref.n_components, skel["n_components"]),
        local_euler=relative_term(ref.local_euler, skel["local_euler"]),
        cl_sensitivity=overlap_term(skel["cl_sensitivity"], 3),
    )
    bif = None
    if ref.ref_bifurcations is not None:
        bif = bifurcation_dice(graph, ref)
        skel["bifurcation_dice"] = bif.dice
        terms["bifurcation_dice"] = overlap_term(bif.dice, 2)
    else:
        warnings.warn("No bifurcation reference: bifurcation DICE term omitted.", stacklevel=2)
    reference = dict(volume=ref.volume, n_components=ref.n_components,
                     local_euler=ref.local_euler)
    return SuperMetricResult(total=combine_terms(terms), terms=terms, skeleton=skel,
                             reference=reference, options=opt, bifurcation=bif)


# --------------------------------------------------------------------------
# Command line
# --------------------------------------------------------------------------

def _parse_triplet(values):
    if values is None:
        return None
    return values[0] if len(values) == 1 else values


def main(argv=None):
    p = argparse.ArgumentParser(description="Skeleton super metric (Walsh et al. 2024, Eq. 10)")
    p.add_argument("binary", help="binary segmentation (3D TIFF)")
    p.add_argument("skeletons", nargs="+",
                   help="skeleton(s) to evaluate: Amira ASCII .am files or CSV graph folders")
    gt_group = p.add_mutually_exclusive_group()
    gt_group.add_argument("--gt", help="manually drawn ground-truth skeleton for the bifurcation "
                                       "DICE (.am file or CSV graph folder)")
    gt_group.add_argument("--gt-points", help="ground-truth bifurcation points: CSV with columns x,y,z")
    p.add_argument("--voxel-size", type=float, nargs="+", required=True,
                   help="voxel size: one value or x y z")
    p.add_argument("--origin", type=float, nargs="+", default=[0.0],
                   help="physical coordinate of voxel [0,0,0] centre: one value or x y z")
    p.add_argument("--bbox", type=float, nargs=6,
                   metavar=("XMIN", "XMAX", "YMIN", "YMAX", "ZMIN", "ZMAX"),
                   help="bifurcation sub-volume (physical units); default: extent of the reference")
    p.add_argument("--preset", choices=sorted(PRESETS), default="paper",
                   help="option preset (default: paper)")
    p.add_argument("--matching", choices=["hungarian", "greedy"],
                   help="override the preset's matching (greedy is for comparison only)")
    p.add_argument("--tolerance-factor", type=float,
                   help="override: match tolerance as a multiple of the local radius")
    p.add_argument("--radius-field", help="Amira point field holding radii (default: auto)")
    p.add_argument("-o", "--output", help="write results to this CSV file")
    args = p.parse_args(argv)

    options = PRESETS[args.preset]
    overrides = {}
    if args.matching:
        overrides["matching"] = args.matching
    if args.tolerance_factor is not None:
        overrides.update(tolerance="radius", tolerance_value=args.tolerance_factor)
    if overrides:
        options = SuperMetricOptions(**{**asdict(options), **overrides})

    binary = read_image(args.binary)
    gt = load_graph(args.gt, args.radius_field) if args.gt else None
    gt_points = read_bifurcation_points(args.gt_points) if args.gt_points else None
    ref = reference_metrics(binary, _parse_triplet(args.voxel_size), _parse_triplet(args.origin),
                            gt_graph=gt, bbox=args.bbox, options=options,
                            gt_bifurcations=gt_points)
    print(f"binary: volume={ref.volume:.6g} cc={ref.n_components} local_euler={ref.local_euler}"
          + (f" ref_bifurcations={len(ref.ref_bifurcations)}"
             if ref.ref_bifurcations is not None else "")
          + ("" if options.paper_comparable else "  [options differ from the paper]"))

    rows = []
    for path in args.skeletons:
        res = super_metric(load_graph(path, args.radius_field), ref)
        rows.append({"skeleton": path, **res.as_row()})
        print(f"{Path(path).name}: super_metric={res.total:.4g}  "
              + "  ".join(f"{k}={v:.4g}" for k, v in res.terms.items()))

    if args.output:
        fields = list(dict.fromkeys(k for r in rows for k in r))
        with open(args.output, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(rows)
        print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
