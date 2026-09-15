"""Volume-based optimisation metrics: skeleton-vs-segmentation overlap and
whole-volume morphometrics.

* :func:`centreline_sensitivity` — fraction of the skeleton centreline inside the
  segmentation (clDice sensitivity, by point-sampling).
* :func:`skeleton_junction_points` — reference bifurcation coordinates from the
  segmentation's own skeleton.
* :func:`region_props_table` / :func:`region_morphometrics` — per-region and
  whole-volume connected-components / Euler number / volume / surface area (the
  Python replacement for the Fiji/MorphoLibJ ``super_metric_Euler_and_cc.ijm``).
* :func:`super_metric` — the combined skeleton-optimisation "meta metric"
  (Volume + CC + Euler + branch-point Dice + clDice), completing ``meta_metric.m``.

The morphometric / super-metric functions need the ``[image]`` extra (scikit-image).
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np

from skeleton_analysis.io.amira import SpatialGraph
from skeleton_analysis.io.amira_lattice import AmiraLattice


def _skeleton_voxel_indices(graph: SpatialGraph, lattice: AmiraLattice) -> np.ndarray:
    """Integer ``(iz, iy, ix)`` voxel indices for every in-bounds skeleton point."""
    pts = np.asarray(graph.point_coords, dtype=float)
    idx = np.rint(lattice.world_to_index_zyx(pts)).astype(np.int64)  # (n, 3) = (iz,iy,ix)
    nz, ny, nx = lattice.volume.shape
    inb = (
        (idx[:, 0] >= 0) & (idx[:, 0] < nz)
        & (idx[:, 1] >= 0) & (idx[:, 1] < ny)
        & (idx[:, 2] >= 0) & (idx[:, 2] < nx)
    )
    return idx[inb]


def centreline_sensitivity(
    graph: SpatialGraph, lattice: AmiraLattice, dedupe: bool = True
) -> float:
    """Fraction of the skeleton centreline that lies inside the segmentation.

    Equivalent to ``cl_score(v_l, s_p)`` with ``s_p`` the skeleton voxel set, but
    computed by sampling the segmentation at the skeleton points (no rasterised
    skeleton volume). ``dedupe`` counts each occupied voxel once (the voxel-based
    definition); set ``False`` to weight by point density. Returns NaN if no
    skeleton point falls inside the volume bounds.
    """
    idx = _skeleton_voxel_indices(graph, lattice)
    if idx.shape[0] == 0:
        return float("nan")
    if dedupe:
        idx = np.unique(idx, axis=0)
    vals = lattice.volume[idx[:, 0], idx[:, 1], idx[:, 2]]
    inside = np.count_nonzero(vals > 0)
    return float(inside / len(idx))


def skeleton_junction_points(
    lattice: AmiraLattice,
    crop_to_nonzero: bool = True,
    pad: int = 2,
    min_neighbours: int = 3,
) -> np.ndarray:
    """Reference bifurcation coordinates from the segmentation's own skeleton.

    Skeletonises the binary volume (optionally cropped to its non-zero bounding
    box for speed — the segmentation is typically <1% foreground), finds junction
    voxels (skeleton voxels with ``>= min_neighbours`` skeleton neighbours in the
    26-neighbourhood), and returns their **world** coordinates ``(x, y, z)``.

    Requires the ``[image]`` extra.
    """
    from scipy.spatial import cKDTree
    from skimage.morphology import skeletonize  # lazy: [image] extra

    binary = lattice.volume > 0
    offset = np.zeros(3, dtype=np.int64)  # (z, y, x) crop offset
    if crop_to_nonzero and binary.any():
        nz = np.argwhere(binary)
        lo = np.maximum(nz.min(axis=0) - pad, 0)
        hi = np.minimum(nz.max(axis=0) + pad + 1, binary.shape)
        binary = binary[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]
        offset = lo

    skel = skeletonize(binary)
    coords = np.argwhere(skel)  # (m, 3) as (iz, iy, ix) in cropped frame
    if coords.shape[0] == 0:
        return np.empty((0, 3))

    # A voxel is a junction if it has >= min_neighbours skeleton neighbours in
    # the 26-neighbourhood (Chebyshev distance 1 -> Euclidean <= sqrt(3)).
    r = np.sqrt(3) + 1e-6
    tree = cKDTree(coords)
    counts = np.array([len(n) for n in tree.query_ball_point(coords, r=r)])
    counts -= 1  # exclude self
    junc = coords[counts >= min_neighbours].astype(float)
    if junc.shape[0] == 0:
        return np.empty((0, 3))

    # A single branch point produces a small blob of high-degree voxels (diagonal
    # neighbours inflate counts at crossings). Cluster adjacent junction voxels
    # (26-connectivity) and take one centroid per cluster.
    import networkx as nx

    jt = cKDTree(junc)
    gj = nx.Graph()
    gj.add_nodes_from(range(len(junc)))
    gj.add_edges_from(jt.query_pairs(r=r))
    junc = np.array([junc[list(c)].mean(axis=0) for c in nx.connected_components(gj)])

    # Cropped (iz,iy,ix) -> global (iz,iy,ix) -> world (x,y,z).
    junc_global = junc + offset
    ix = junc_global[:, 2].astype(float)
    iy = junc_global[:, 1].astype(float)
    iz = junc_global[:, 0].astype(float)
    vox_xyz = np.stack([ix, iy, iz], axis=1)
    return lattice.origin + vox_xyz * lattice.spacing


# ---------------------------------------------------------------------------
# Whole-volume morphometrics (Python replacement for the Fiji/MorphoLibJ macro)
# ---------------------------------------------------------------------------
def region_props_table(volume: np.ndarray, area_threshold: int = 0):
    """Per-region area + Euler number of a binary volume (port of ``calc_region_props``).

    Optionally fills small holes first, labels the volume, and returns a table of
    ``area``, ``label``, ``euler_number`` per connected region. Requires ``[image]``.
    """
    import pandas as pd
    from skimage.measure import label, regionprops_table
    from skimage.morphology import remove_small_holes

    v = np.asarray(volume) > 0
    if area_threshold and area_threshold > 0:
        v = remove_small_holes(v, area_threshold=area_threshold)
    labels = label(v)
    props = regionprops_table(labels, properties=("area", "label", "euler_number"))
    return pd.DataFrame(props)


def region_morphometrics(
    volume: np.ndarray, voxel_size: float = 1.0, connectivity: int = 3
) -> Dict[str, float]:
    """Whole-volume connected components / Euler number / volume / surface area.

    Python replacement for the Fiji/MorphoLibJ ``super_metric_Euler_and_cc.ijm``.
    ``connectivity=3`` is 26-connectivity in 3-D (Fiji's setting). Surface area is
    the marching-cubes mesh area (the scikit-image analogue of the Crofton estimate).
    Requires ``[image]``.
    """
    from skimage.measure import (
        euler_number,
        label,
        marching_cubes,
        mesh_surface_area,
    )

    binary = np.asarray(volume) > 0
    n_vox = int(np.count_nonzero(binary))
    labels = label(binary, connectivity=connectivity)
    n_cc = int(labels.max())
    euler = int(euler_number(binary, connectivity=connectivity)) if n_vox else 0

    surface = 0.0
    if n_vox and binary.ndim == 3 and min(binary.shape) > 1:
        try:
            verts, faces, _n, _v = marching_cubes(binary.astype(float), level=0.5)
            surface = float(mesh_surface_area(verts, faces)) * voxel_size ** 2
        except (RuntimeError, ValueError):
            surface = float("nan")

    return {
        "connected_components": n_cc,
        "euler_number": euler,
        "volume": n_vox * voxel_size ** 3,
        "surface_area": surface,
        "voxel_count": n_vox,
    }


def super_metric(
    candidate_volume: np.ndarray,
    reference_volume: np.ndarray,
    candidate_graph: Optional[SpatialGraph] = None,
    reference_graph: Optional[SpatialGraph] = None,
    voxel_size: float = 1.0,
    bb_threshold: float = 900.0,
    normalize: float = 1.0,
) -> Dict[str, float]:
    """Combined skeleton-optimisation "meta metric" (completes ``meta_metric.m``).

    Assembles candidate/reference values for Volume, connected Components (CC),
    Euler number, Branch-point count (BB) and clDice (CL), then combines them with
    :func:`skeleton_analysis.optimisation.meta_metric.meta_metric` (normalised RMS
    of the relative differences; 0 == identical). Branch counts come from the two
    graphs (matched via ``bifurcation_dice``) when both are supplied; CL is the
    clDice of candidate vs reference (reference self-CL = 1.0). Requires ``[image]``.
    """
    from skeleton_analysis.optimisation.cl_dice import cl_dice
    from skeleton_analysis.optimisation.meta_metric import (
        bifurcation_dice,
        bifurcation_points,
        meta_metric,
    )

    cand_m = region_morphometrics(candidate_volume, voxel_size)
    ref_m = region_morphometrics(reference_volume, voxel_size)

    cand = {"Volume": cand_m["volume"], "CC": cand_m["connected_components"],
            "Euler": cand_m["euler_number"]}
    ref = {"Volume": ref_m["volume"], "CC": ref_m["connected_components"],
           "Euler": ref_m["euler_number"]}

    result: Dict[str, float] = {}
    if candidate_graph is not None and reference_graph is not None:
        bb = bifurcation_dice(candidate_graph, reference_graph, threshold=bb_threshold)
        cand["BB"] = bb.n_candidate
        ref["BB"] = bb.n_reference
        result["bifurcation_dice"] = bb.dice

    cl = cl_dice(candidate_volume, reference_volume, normalize=normalize)
    result["clDice"] = cl
    if np.isfinite(cl):  # skip CL when undefined (e.g. an empty skeleton)
        cand["CL"] = cl
        ref["CL"] = 1.0

    result.update({f"candidate_{k}": v for k, v in cand.items()})
    result.update({f"reference_{k}": v for k, v in ref.items()})
    result["meta_metric"] = meta_metric(cand, ref)
    return result
