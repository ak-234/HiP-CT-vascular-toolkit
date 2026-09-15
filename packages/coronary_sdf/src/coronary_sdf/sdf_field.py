"""Narrow-band signed-distance field evaluation.

The vectorised core that was the bulk of ``generate_sdf_surface``:

- Adjacency, bifurcation node tree, terminal endpoint info.
- Bounding box + auto voxel size + grid dims.
- Narrow band via per-capsule AABB union.
- Batched SDF evaluation with topology-aware smooth-min and the
  patch-81..87 anti-bridge carve (wall-band sandwich + cross-section
  gate + junction-ball protection + tangent-angle gate).
- SDF-level flat-cap clamp at terminal endpoints.
- Optional Gaussian smoothing of the final SDF volume.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy.spatial import KDTree

from .config import runtime_config as config
from .capsules import CapsuleArrays
from .splines import branch_tangent_at_node


# ── Smooth-min helpers ────────────────────────────────────────────────────────


def smooth_min_exp(values: np.ndarray, k: float | None = None) -> float:
    """Log-sum-exp smooth minimum: ``-ln(sum(exp(-k*v))) / k``."""
    if k is None:
        k = config.SMIN_K_DEFAULT
    if len(values) == 0:
        return float("inf")
    v_min = values.min()
    return float(v_min - np.log(np.sum(np.exp(-k * (values - v_min)))) / k)


def smooth_min_poly_pair(
    a: np.ndarray, b: np.ndarray, k: float | np.ndarray
) -> np.ndarray:
    """Polynomial 2-ary smooth minimum (iquilezles).

    ``smin(a, b, k) = min(a, b) - max(k - |a-b|, 0)^2 / (4k)``

    Compact support: equals ``min(a, b)`` when ``|a-b| >= k``. ``k`` may be
    scalar or per-voxel; broadcastable against ``a`` and ``b``.
    """
    m = np.minimum(a, b)
    diff = np.abs(a - b)
    h = np.maximum(k - diff, 0.0)
    k_safe = np.maximum(k, 1e-9)
    return m - (h * h) / (4.0 * k_safe)


def soft_cap(x: np.ndarray, cap, knee: float) -> np.ndarray:
    """Smoothly cap ``x`` from above at ``cap`` (corner rounded over ``knee``).

    ``knee <= 0`` -> hard ``np.minimum(x, cap)`` (legacy). Otherwise reuses
    the polynomial smin so the result equals ``x`` for ``x <= cap-knee``,
    equals ``cap`` for ``x >= cap+knee``, and is C1 across the corner in
    between. ``cap`` may be a scalar or an array broadcastable against ``x``.
    """
    if knee <= 0.0:
        return np.minimum(x, cap)
    return smooth_min_poly_pair(x, np.asarray(cap, dtype=x.dtype), knee)


def adaptive_smin_k(local_radius: float, k_base: float | None = None) -> float:
    """Continuous k(r) law clipped to ``[SMIN_K_MIN, SMIN_K_MAX]``."""
    if not config.SMIN_ADAPTIVE_BLEND:
        return float(k_base) if k_base is not None else float(config.BSPLINE_SMIN_K)
    if k_base is None:
        k_base = config.BSPLINE_SMIN_K
    # k has units 1/length. Keeping k*r constant makes the blend equivariant
    # under a global rescaling of coordinates and radii.
    k = k_base * (config.SMIN_K_REF_RADIUS / max(float(local_radius), 1e-12))
    return float(np.clip(k, config.SMIN_K_MIN, config.SMIN_K_MAX))


# ── Endpoint / bif info ──────────────────────────────────────────────────────


def collect_endpoint_info(
    nodes: dict[int, tuple],
    points: dict[int, tuple],
    segments: list[dict[str, Any]],
    node_to_segs: dict[int, set[int]],
) -> list[tuple[np.ndarray, np.ndarray, float]]:
    """Per-terminal (coord==1) endpoint info: ``(position_mm, outward_normal, radius_mm)``."""
    out: list[tuple[np.ndarray, np.ndarray, float]] = []
    for nid, (x, y, z, coord) in nodes.items():
        if coord != 1:
            continue
        pos = np.array([x, y, z], dtype=np.float64) / 1000.0
        seg_indices = node_to_segs.get(nid, set())
        if not seg_indices:
            continue
        seg_idx = next(iter(seg_indices))
        seg = segments[seg_idx]
        pids = seg["point_ids"]
        if not pids or len(pids) < 2:
            continue
        is_node1 = seg["node1"] == nid
        tip_pids = pids[: min(5, len(pids))] if is_node1 else pids[-min(5, len(pids)):]
        tip_coords: list[np.ndarray] = []
        for pid in tip_pids:
            if pid in points:
                tip_coords.append(np.array(points[pid][:3], dtype=np.float64) / 1000.0)
        if len(tip_coords) < 2:
            continue
        tip_arr = np.array(tip_coords)
        if is_node1:
            tangent = tip_arr[0] - tip_arr[-1]
        else:
            tangent = tip_arr[-1] - tip_arr[0]
        tlen = float(np.linalg.norm(tangent))
        if tlen < 1e-10:
            continue
        normal = tangent / tlen
        # Flat-cap reach scales with this radius, so use the median of
        # the last few INTERIOR points rather than the terminal point's
        # own radius. The terminal point often has shrunk raw thickness
        # which would yield a too-narrow reach and a visible taper at
        # the cap. Median over the same tip points already used for the
        # tangent estimate above.
        n_lookahead = min(3, len(pids))
        if is_node1:
            lookahead_pids = pids[:n_lookahead]
        else:
            lookahead_pids = pids[-n_lookahead:]
        lookahead_rs = [
            points[pid][3] / 1000.0 * config.RADIUS_SCALE
            for pid in lookahead_pids
            if pid in points
        ]
        radius = float(np.median(lookahead_rs)) if lookahead_rs else 0.2
        out.append((pos, normal, float(radius)))
    return out


@dataclass
class TerminalSet:
    pos: np.ndarray | None = None
    nrm: np.ndarray | None = None
    rad: np.ndarray | None = None
    tree: KDTree | None = None


def build_terminal_set(endpoint_info: list[tuple[np.ndarray, np.ndarray, float]]) -> TerminalSet:
    if not endpoint_info:
        return TerminalSet()
    pos = np.array([p for p, _, _ in endpoint_info], dtype=np.float64)
    nrm = np.array([n for _, n, _ in endpoint_info], dtype=np.float64)
    rad = np.array([r for _, _, r in endpoint_info], dtype=np.float64)
    return TerminalSet(pos=pos, nrm=nrm, rad=rad, tree=KDTree(pos))


@dataclass
class BifurcationSet:
    positions: np.ndarray = field(default_factory=lambda: np.empty((0, 3)))
    radii: np.ndarray = field(default_factory=lambda: np.empty(0))
    tree: KDTree | None = None
    # Node ids parallel to positions / radii. Used to build the
    # (n_bifs, n_segs) bif_seg_incident matrix for the topology-aware
    # carve shield.
    node_ids: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.int64))


def find_bifurcations(
    nodes: dict[int, tuple],
    points: dict[int, tuple],
    segments: list[dict[str, Any]],
    node_to_segs: dict[int, set[int]],
) -> BifurcationSet:
    """Identify coord>2 nodes and their mean incident radius."""
    positions: list[list[float]] = []
    radii: list[float] = []
    node_ids: list[int] = []
    for nid, (x, y, z, coord) in nodes.items():
        if coord <= 2:
            continue
        positions.append([x / 1000.0, y / 1000.0, z / 1000.0])
        node_ids.append(int(nid))
        seg_rs: list[float] = []
        for seg_idx in node_to_segs.get(nid, []):
            seg = segments[seg_idx]
            pids = seg["point_ids"]
            if not pids:
                continue
            pid = pids[0] if seg["node1"] == nid else pids[-1]
            if pid in points:
                seg_rs.append(points[pid][3] / 1000.0 * config.RADIUS_SCALE)
        radii.append(float(np.mean(seg_rs)) if seg_rs else 0.2)
    if not positions:
        return BifurcationSet()
    pos = np.array(positions, dtype=np.float64)
    rad = np.array(radii, dtype=np.float64)
    nids = np.array(node_ids, dtype=np.int64)
    print(f"  {len(pos)} bifurcation nodes identified")
    return BifurcationSet(positions=pos, radii=rad, tree=KDTree(pos), node_ids=nids)


# ── Adjacency (with near-coincident node merging) ────────────────────────────


def build_adjacency(
    nodes: dict[int, tuple],
    points: dict[int, tuple],
    segments: list[dict[str, Any]],
    node_to_segs: dict[int, set[int]],
) -> tuple[np.ndarray, dict[tuple[int, int], np.ndarray], dict[tuple[int, int], float]]:
    """Per-segment-pair adjacency matrix + shared-node position/radius lookups.

    Merges nodes within ``config.NODE_COINCIDENCE_EPS_MM`` of each other.
    """
    from .topology import node_id_canon_map

    n_segs = len(segments)
    adj_matrix = np.eye(n_segs, dtype=bool)
    shared_node_pos: dict[tuple[int, int], np.ndarray] = {}
    shared_node_radius: dict[tuple[int, int], float] = {}

    canon = node_id_canon_map(nodes, config.NODE_COINCIDENCE_EPS_MM)
    canon_to_members: dict[int, list[int]] = {}
    for nid in node_to_segs.keys():
        canon_to_members.setdefault(canon.get(nid, nid), []).append(nid)

    for _canon, members in canon_to_members.items():
        seg_indices: set[int] = set()
        for nid in members:
            seg_indices.update(node_to_segs.get(nid, set()))
        seg_list = list(seg_indices)
        if not seg_list:
            continue
        node_pos = (
            np.mean(
                np.array(
                    [[nodes[nid][0], nodes[nid][1], nodes[nid][2]] for nid in members],
                    dtype=np.float64,
                ),
                axis=0,
            )
            / 1000.0
        )
        node_radius = 0.2
        for si in seg_list:
            seg = segments[si]
            pids = seg["point_ids"]
            if not pids:
                continue
            if seg["node1"] in members:
                pid = pids[0]
            elif seg["node2"] in members:
                pid = pids[-1]
            else:
                continue
            if pid in points:
                node_radius = max(node_radius, points[pid][3] / 1000.0 * config.RADIUS_SCALE)
        for i, si in enumerate(seg_list):
            for sj in seg_list[i + 1:]:
                adj_matrix[si, sj] = True
                adj_matrix[sj, si] = True
                shared_node_pos[(si, sj)] = node_pos
                shared_node_pos[(sj, si)] = node_pos
                shared_node_radius[(si, sj)] = node_radius
                shared_node_radius[(sj, si)] = node_radius
    return adj_matrix, shared_node_pos, shared_node_radius


# ── Proximity diagnostic ─────────────────────────────────────────────────────


def report_non_adjacent_proximity(
    nodes: dict[int, tuple],
    points: dict[int, tuple],
    segments: list[dict[str, Any]],
    voxel_size_mm: float = 0.1,
    top_n: int | None = None,
) -> list[dict[str, Any]]:
    """List non-adjacent segment pairs whose centerlines come within
    ``r_A + r_B + voxel`` of each other."""
    if top_n is not None and top_n <= 0:
        return []
    n_segs = len(segments)
    if n_segs < 2:
        return []

    seg_pts: list[np.ndarray | None] = []
    seg_pids: list[list[int] | None] = []
    seg_maxr = np.zeros(n_segs, dtype=np.float64)
    seg_ids_ok: list[int] = []
    for si, seg in enumerate(segments):
        pts_local: list[list[float]] = []
        pids_local: list[int] = []
        rmax = 0.0
        for pid in seg["point_ids"]:
            if pid in points:
                p = points[pid]
                pts_local.append([p[0] / 1000.0, p[1] / 1000.0, p[2] / 1000.0])
                pids_local.append(pid)
                rmax = max(rmax, p[3] / 1000.0 * config.RADIUS_SCALE)
        if len(pts_local) < 2 or rmax <= 0.0:
            seg_pts.append(None)
            seg_pids.append(None)
            continue
        seg_pts.append(np.asarray(pts_local, dtype=np.float64))
        seg_pids.append(pids_local)
        seg_maxr[si] = rmax
        seg_ids_ok.append(si)

    node_to_segs: dict[int, set[int]] = {}
    for si, seg in enumerate(segments):
        for nid in (seg["node1"], seg["node2"]):
            node_to_segs.setdefault(nid, set()).add(si)
    adj: list[set[int]] = [set() for _ in range(n_segs)]
    for _nid, eset in node_to_segs.items():
        L = list(eset)
        for a in L:
            for b in L:
                if a != b:
                    adj[a].add(b)

    from scipy.spatial import cKDTree

    seg_kd: list[Any] = [None] * n_segs
    for si in seg_ids_ok:
        seg_kd[si] = cKDTree(seg_pts[si])  # type: ignore[arg-type]

    offenders: list[dict[str, Any]] = []
    for si in seg_ids_ok:
        pts_i = seg_pts[si]
        r_i = seg_maxr[si]
        for sj in seg_ids_ok:
            if sj <= si or sj in adj[si]:
                continue
            r_j = seg_maxr[sj]
            threshold = r_i + r_j + voxel_size_mm
            dists, idxs = seg_kd[sj].query(pts_i, k=1)
            min_idx = int(np.argmin(dists))
            min_d = float(dists[min_idx])
            if min_d < threshold:
                pid_a = -1
                pid_b = -1
                if seg_pids[si] is not None:
                    pid_a = seg_pids[si][min_idx]
                if seg_pids[sj] is not None:
                    pid_b = seg_pids[sj][int(idxs[min_idx])]
                offenders.append(
                    {
                        "min_d": min_d,
                        "threshold": threshold,
                        "seg_a_idx": si,
                        "seg_b_idx": sj,
                        "seg_a_id": segments[si]["id"],
                        "seg_b_id": segments[sj]["id"],
                        "pid_a": pid_a,
                        "pid_b": pid_b,
                    }
                )

    n_off = len(offenders)
    if n_off == 0:
        print("  [PROXIMITY] No non-adjacent segment pairs within r_A + r_B + voxel.")
        return []
    offenders.sort(key=lambda row: row["min_d"])
    shown = offenders[:top_n] if top_n is not None and top_n > 0 else offenders
    print(f"  [PROXIMITY] {n_off} non-adjacent pair(s) closer than r_A + r_B + voxel:")
    print(
        f"  {'rank':>4} {'min_d (mm)':>11} {'threshold':>11}"
        f" {'seg_a_id':>9} {'seg_b_id':>9} {'pid_a':>9} {'pid_b':>9}"
    )
    for k, row in enumerate(shown, 1):
        print(
            f"  {k:>4} {row['min_d']:>11.4f} {row['threshold']:>11.4f}"
            f" {row['seg_a_id']:>9} {row['seg_b_id']:>9}"
            f" {row['pid_a']:>9} {row['pid_b']:>9}"
        )
    if top_n is not None and n_off > top_n:
        print(f"  ... ({n_off - top_n} more not shown)")
    return offenders


# ── Bounding box & narrow band ───────────────────────────────────────────────


# MeshLib's SimpleVolume and VTK's ImageData both address voxels with a signed
# 32-bit index, so 2**31 voxels is a hard ceiling for the dense extraction
# backends regardless of available memory.
_DENSE_VOXEL_INDEX_LIMIT = 2**31


@dataclass
class Grid:
    bbox_min: np.ndarray
    bbox_max: np.ndarray
    voxel_size: float
    dims: np.ndarray            # (3,) int
    x: np.ndarray
    y: np.ndarray
    z: np.ndarray

    @property
    def n_voxels(self) -> int:
        # Multiply as Python ints. ``dims`` is int32 on Windows, so
        # ``np.prod(dims)`` silently overflows past ~2.1e9 voxels and returns a
        # negative count -- which would defeat every size guard downstream.
        return int(self.dims[0]) * int(self.dims[1]) * int(self.dims[2])


def compute_grid(capsules: CapsuleArrays) -> Grid:
    """Auto-pick voxel size from min radius (capped by ``BSPLINE_SDF_MAX_VOXELS``)."""
    all_coords = np.vstack([capsules.starts, capsules.ends])
    max_radius = float(capsules.max_radii.max())
    min_radius = float(capsules.max_radii.min())
    bbox_min = all_coords.min(axis=0) - max_radius - config.PADDING
    bbox_max = all_coords.max(axis=0) + max_radius + config.PADDING
    print(
        f"  Bounding box: {bbox_max[0]-bbox_min[0]:.1f} x "
        f"{bbox_max[1]-bbox_min[1]:.1f} x {bbox_max[2]-bbox_min[2]:.1f} mm"
    )

    bbox_size = bbox_max - bbox_min
    bbox_volume = float(np.prod(bbox_size))
    if config.BSPLINE_SDF_RESOLUTION is None:
        voxel_from_radius = min_radius / 2.5
        voxel_from_limit = (bbox_volume / config.BSPLINE_SDF_MAX_VOXELS) ** (1 / 3)
        candidates = [voxel_from_radius, voxel_from_limit]
        if config.DENSE_MIN_SPACING_MM is not None:
            candidates.append(float(config.DENSE_MIN_SPACING_MM))
        voxel_size = max(candidates)
        print(f"  Auto voxel size: {voxel_size:.3f} mm (min_radius={min_radius:.3f}mm)")
    else:
        voxel_size = config.BSPLINE_SDF_RESOLUTION
        est = bbox_volume / (voxel_size**3)
        if est > config.BSPLINE_SDF_MAX_VOXELS:
            voxel_size = (bbox_volume / config.BSPLINE_SDF_MAX_VOXELS) ** (1 / 3)
            print(f"  [WARN] Resolution too fine, adjusted to {voxel_size:.3f} mm")

    bbox_max_req = bbox_max
    dims = np.ceil((bbox_max_req - bbox_min) / voxel_size).astype(int) + 1
    # The extraction backends use ``voxel_size`` as the grid spacing.  Keep
    # coordinates on exactly that lattice instead of stretching linspace to
    # the pre-rounded bounding box (which previously introduced up to one
    # voxel of coordinate error per axis).
    bbox_max = bbox_min + (dims - 1) * voxel_size
    # Python-int product: np.prod on int32 dims overflows past ~2.1e9 voxels.
    n_voxels = int(dims[0]) * int(dims[1]) * int(dims[2])
    # The dense backends (MeshLib SimpleVolume, VTK ImageData) index voxels with
    # a signed 32-bit type, so a grid at or beyond 2**31 cannot be contoured at
    # all. Coarsen to stay inside that ceiling rather than failing deep inside a
    # C++ extension with an opaque error.
    if n_voxels >= _DENSE_VOXEL_INDEX_LIMIT:
        safe_voxel = (bbox_volume / (_DENSE_VOXEL_INDEX_LIMIT * 0.95)) ** (1 / 3)
        print(
            f"  [WARN] grid of {n_voxels:,} voxels exceeds the {_DENSE_VOXEL_INDEX_LIMIT:,} "
            f"32-bit indexing limit of the dense extraction backends; "
            f"coarsening {voxel_size:.4f} -> {safe_voxel:.4f} mm"
        )
        voxel_size = safe_voxel
        dims = np.ceil((bbox_max_req - bbox_min) / voxel_size).astype(int) + 1
        bbox_max = bbox_min + (dims - 1) * voxel_size
        n_voxels = int(dims[0]) * int(dims[1]) * int(dims[2])
    print(f"  Grid: {dims[0]} x {dims[1]} x {dims[2]} = {n_voxels:,} voxels")
    x = bbox_min[0] + np.arange(dims[0], dtype=np.float64) * voxel_size
    y = bbox_min[1] + np.arange(dims[1], dtype=np.float64) * voxel_size
    z = bbox_min[2] + np.arange(dims[2], dtype=np.float64) * voxel_size
    return Grid(bbox_min, bbox_max, float(voxel_size), dims, x, y, z)


def build_narrow_band(capsules: CapsuleArrays, grid: Grid) -> np.ndarray:
    """Union of per-capsule AABBs, padded by wall + blend half-width. Returns
    an ``(n_band, 3)`` int array of voxel indices."""
    t1 = time.time()
    seg_lo = np.minimum(capsules.starts, capsules.ends)
    seg_hi = np.maximum(capsules.starts, capsules.ends)
    pad = np.maximum(
        grid.voxel_size * 3.0,
        capsules.max_radii * (1.0 + config.SMIN_PROXIMITY_BLEND_FACTOR),
    )
    inv_vs = 1.0 / grid.voxel_size
    lo_vox = np.floor((seg_lo - pad[:, None] - grid.bbox_min[None, :]) * inv_vs).astype(np.int64)
    hi_vox = np.ceil((seg_hi + pad[:, None] - grid.bbox_min[None, :]) * inv_vs).astype(np.int64)
    lo_vox = np.maximum(lo_vox, 0)
    hi_vox = np.minimum(hi_vox, np.array(grid.dims, dtype=np.int64) - 1)

    mask = np.zeros(tuple(grid.dims), dtype=bool)
    for i in range(capsules.n):
        if (hi_vox[i] < lo_vox[i]).any():
            continue
        mask[
            lo_vox[i, 0]:hi_vox[i, 0] + 1,
            lo_vox[i, 1]:hi_vox[i, 1] + 1,
            lo_vox[i, 2]:hi_vox[i, 2] + 1,
        ] = True
    nb = np.stack(np.where(mask), axis=1).astype(np.int64, copy=False)
    n_band = len(nb)
    n_total = grid.n_voxels
    print(
        f"  Narrow band: {n_band:,} voxels ({100*n_band/max(n_total, 1):.1f}%) "
        f"in {time.time()-t1:.1f}s"
    )
    return nb


# ── Main SDF eval ────────────────────────────────────────────────────────────


@dataclass
class SdfVolume:
    sdf: np.ndarray             # (dims) float32
    path_vol: np.ndarray | None  # diagnostic (BLEND_DIAGNOSTIC)
    blend_weight_vol: np.ndarray | None
    # Per-narrow-band debug diagnostics (DETAILED_BLEND_DIAGNOSTIC). nb_idx
    # is (n_band, 3) i,j,k voxel indices into the grid; diag is a dict of
    # named per-band arrays (uint8 / int32 / float32). Consumed by
    # viz.debug_show_blend_diagnostics.
    nb_idx: np.ndarray | None = None
    diag: dict[str, np.ndarray] | None = None


def _evaluate_legacy_kernel(
    points_mm: np.ndarray,
    *,
    length_scale: float,
    capsules: CapsuleArrays,
    cap_is_junction: np.ndarray,
    adj_matrix: np.ndarray,
    bif: BifurcationSet,
    term: TerminalSet,
    shared_node_pos: np.ndarray | None = None,
    shared_node_has: np.ndarray | None = None,
    seg_end_pos: np.ndarray | None = None,
    seg_end_tan: np.ndarray | None = None,
    seg_end_tan_ok: np.ndarray | None = None,
    is_parent: np.ndarray | None = None,
    is_child: np.ndarray | None = None,
    is_sibling: np.ndarray | None = None,
    bif_seg_incident: np.ndarray | None = None,
    batch_size: int = 1_000_000,
    progress: bool = True,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None, dict[str, np.ndarray]]:
    """Evaluate the legacy field at arbitrary world points.

    ``length_scale`` replaces what used to be ``grid.voxel_size`` read directly
    from the evaluation grid. It is a genuine parameter of the *field*, not of
    the sampling: it sets the flat-cap soft band, the capsule candidate search
    radius, and the two carve buffers. Because of that the legacy zero set moves
    when the grid spacing changes, so any comparison across extractors has to
    pin this value explicitly rather than inherit it.

    Returns ``(values, path_flags, blend_weights, diagnostics)`` with one entry
    per input point.

    Implements topology-aware smooth-min plus the patch-81..87 anti-bridge
    carve and the SDF-level flat-cap clamp. Prints carve diagnostic
    counters at the end when ``config.SDF_CARVE_NON_ADJACENT`` is True.

    When ``config.SMIN_GATE_VARIANT == "t_projection"`` and the directed
    topology masks (``is_parent`` / ``is_child`` / ``is_sibling``) are
    supplied, the blend weight is computed from the segment-level
    normalised t of the owner and rival closest points per sdf_plan.md;
    otherwise the multi-gate stack (cross-section + bif-ball + wedge +
    parallel-rival suppression) drives the blend weight.
    """
    points_mm = np.asarray(points_mm, dtype=np.float64)
    if points_mm.ndim == 1:
        points_mm = points_mm.reshape(1, 3)
    if points_mm.ndim != 2 or points_mm.shape[1] != 3:
        raise ValueError("points_mm must have shape (N, 3)")
    n_band = len(points_mm)
    sdf_flat = np.full(n_band, 10.0, dtype=np.float32)
    if config.BLEND_DIAGNOSTIC:
        path_flat = np.zeros(n_band, dtype=np.uint8)
        blend_weight_flat = np.zeros(n_band, dtype=np.float32)
    else:
        path_flat = None
        blend_weight_flat = None

    print("  Evaluating SDF (vectorized)...")
    t2 = time.time()

    # Float32 views to halve memory bandwidth on the gather hot path.
    cap_starts_f32 = capsules.starts.astype(np.float32, copy=False)
    cap_ends_f32 = capsules.ends.astype(np.float32, copy=False)
    cap_radii_start_f32 = capsules.radii_start.astype(np.float32, copy=False)
    cap_radii_end_f32 = capsules.radii_end.astype(np.float32, copy=False)
    cap_seg_idx_i32 = capsules.seg_idx.astype(np.int32, copy=False)
    cap_tangents = capsules.tangents  # already float64
    cap_bif_at_start_bool = np.asarray(
        getattr(capsules, "cap_bif_at_start", np.zeros(capsules.n, dtype=bool)),
        dtype=bool,
    )
    cap_bif_at_end_bool = np.asarray(
        getattr(capsules, "cap_bif_at_end", np.zeros(capsules.n, dtype=bool)),
        dtype=bool,
    )

    # The SDF hot path allocates (batch, K) arrays where K is the per-voxel
    # capsule query count (SDF_MAX_CAPSULE_QUERY). Keep batch*K near the legacy
    # 1M*16 budget so raising K for thick-junction correctness doesn't blow
    # memory; smaller batches cost only a little extra Python overhead.
    _kq = min(int(getattr(config, "SDF_MAX_CAPSULE_QUERY", 16)), max(capsules.n, 1))
    batch_size = min(batch_size, max(100_000, 16_000_000 // max(_kq, 1)))
    n_batches = (n_band + batch_size - 1) // batch_size

    # Patch 83 carve diagnostic totals.
    _band_pre = 0
    _after_gate = 0
    _actually = 0
    # Patch 88 deep-merger carve diagnostic totals.
    _deep_pre = 0
    _deep_actually = 0
    # Adjacent-parallel carve diagnostic totals.
    _adj_pre = 0
    _adj_actually = 0
    # Wedge-pass diagnostic: sum over voxels of wedge_pass weight (in [0,1]).
    _wedge_total_sum = 0.0

    # ── Detailed per-narrow-band diagnostics ─────────────────────────────
    detailed = bool(getattr(config, "DETAILED_BLEND_DIAGNOSTIC", False))
    diag: dict[str, np.ndarray] = {}
    if detailed:
        diag = {
            # Gates
            "gate_xs_active":      np.zeros(n_band, dtype=np.uint8),
            "gate_bif_ball":       np.zeros(n_band, dtype=np.uint8),
            "gate_intersection":   np.zeros(n_band, dtype=np.uint8),
            "gate_proximity":      np.zeros(n_band, dtype=np.float32),
            "gate_wedge":          np.ones(n_band, dtype=np.float32),
            "gate_par_suppress":   np.zeros(n_band, dtype=np.uint8),
            "gate_jprotect":       np.zeros(n_band, dtype=np.uint8),
            # Topology-aware shield (subset of gate_jprotect when
            # SDF_CARVE_PROTECT_JUNCTION_BALL_TOPO_AWARE is True). Recorded
            # for the wall-band rival; the deep-merger shield uses
            # non_adj_rival_seg and is similar in shape.
            "gate_jprotect_topo":  np.zeros(n_band, dtype=np.uint8),
            # T-projection (only meaningful when SMIN_GATE_VARIANT == "t_projection")
            "t_seg_owner":         np.full(n_band, np.nan, dtype=np.float32),
            "t_seg_rival":         np.full(n_band, np.nan, dtype=np.float32),
            "t_relation":          np.zeros(n_band, dtype=np.uint8),
            # Smooth-min
            "final_blend_w":       np.zeros(n_band, dtype=np.float32),
            "smin_depression":     np.zeros(n_band, dtype=np.float32),
            # Topology / proximity
            "owner_seg":           np.full(n_band, -1, dtype=np.int32),
            "non_adj_min":         np.full(n_band, np.inf, dtype=np.float32),
            "non_adj_rival_seg":   np.full(n_band, -1, dtype=np.int32),
            # Carves
            "carve_wb_pre":        np.zeros(n_band, dtype=np.uint8),
            "carve_wb_act":        np.zeros(n_band, dtype=np.uint8),
            "carve_deep_pre":      np.zeros(n_band, dtype=np.uint8),
            "carve_deep_act":      np.zeros(n_band, dtype=np.uint8),
            "carve_adj_pre":       np.zeros(n_band, dtype=np.uint8),
            "carve_adj_act":       np.zeros(n_band, dtype=np.uint8),
            "carve_total_push":    np.zeros(n_band, dtype=np.float32),
        }

    for batch_idx in range(n_batches):
        b_start = batch_idx * batch_size
        b_end = min((batch_idx + 1) * batch_size, n_band)
        batch_coords = points_mm[b_start:b_end]
        if len(batch_coords) == 0:
            continue
        M = len(batch_coords)

        max_query = min(int(getattr(config, "SDF_MAX_CAPSULE_QUERY", 16)), capsules.n)
        dists, indices = capsules.tree.query(batch_coords, k=max_query, workers=-1)
        if dists.ndim == 1:
            dists = dists.reshape(-1, 1)
            indices = indices.reshape(-1, 1)
        K = indices.shape[1]

        flat = indices.ravel()
        p0 = cap_starts_f32[flat].reshape(M, K, 3)
        p1 = cap_ends_f32[flat].reshape(M, K, 3)
        r0 = cap_radii_start_f32[flat].reshape(M, K)
        r1 = cap_radii_end_f32[flat].reshape(M, K)
        seg_ids_mk = cap_seg_idx_i32[flat].reshape(M, K)

        pts = batch_coords[:, None, :]
        d = p1 - p0
        h = np.sum(d * d, axis=2)
        pa = pts - p0
        t = np.clip(np.sum(pa * d, axis=2) / np.maximum(h, 1e-10), 0.0, 1.0)
        closest = p0 + t[..., None] * d
        dist_ax = np.linalg.norm(pts - closest, axis=2)
        rad_at_t = r0 + t * (r1 - r0)
        sdf_caps = dist_ax - rad_at_t

        # Flat-cap at bif-incident capsule endpoints. For capsules whose
        # endpoint sits at a degree>=3 node, replace the hemispherical
        # end-cap with a plane through that endpoint normal to the
        # capsule axis. Per-capsule (applied before owner argmin), so
        # adjacent capsules sharing the bif don't punch holes into each
        # other's interiors.
        if getattr(config, "SDF_FLAT_CAP_BIF", False):
            inv_norm = 1.0 / np.sqrt(np.maximum(h, 1e-12))     # (M, K)
            d_unit = d * inv_norm[..., None]                    # (M, K, 3)
            bif_end_mk = cap_bif_at_end_bool[flat].reshape(M, K)
            bif_start_mk = cap_bif_at_start_bool[flat].reshape(M, K)
            if bif_end_mk.any() or bif_start_mk.any():
                # past-end: (pts - p1) · d_unit. > 0 ⇒ past bif end.
                beyond_end = np.einsum("mki,mki->mk", pts - p1, d_unit)
                # past-start: (pts - p0) · (-d_unit) = -((pts - p0)·d_unit).
                beyond_start = -np.einsum("mki,mki->mk", pa, d_unit)
                neg_inf = np.float32(-1e30)
                cap_amount = np.where(
                    bif_end_mk,
                    beyond_end.astype(sdf_caps.dtype, copy=False),
                    np.where(
                        bif_start_mk,
                        beyond_start.astype(sdf_caps.dtype, copy=False),
                        neg_inf,
                    ),
                )
                # Soft-cap the plane with a smoothstep transition over a
                # ~voxel-sized band, so the cap doesn't introduce a hard
                # discontinuity in sdf_caps at the plane (which produces
                # visible ridges at the carina). At beyond < -h the cap
                # is inactive; at beyond > +h it is fully active; between
                # it ramps with smoothstep so the gradient stays bounded.
                # Transition-band half-width. The hardcoded voxel-sized band
                # made the cap's inward carve a sharp recessed ring at the
                # ostium on thick vessels (the visible indentation). Scaling
                # the band with the local radius spreads it into a smooth
                # shoulder. soft_factor == 0 keeps the legacy scalar band.
                soft_factor = float(getattr(config, "SDF_FLAT_CAP_BIF_SOFT_FACTOR", 0.0))
                if soft_factor > 0.0:
                    h_soft = np.maximum(
                        np.float32(length_scale),
                        (rad_at_t * soft_factor).astype(np.float32),
                    )                                       # (M, K)
                else:
                    h_soft = np.float32(length_scale)    # legacy scalar band
                # Optionally shift the plane PAST the bif node by a fraction of
                # the local radius so the daughter fills the ostium valley a
                # little before truncating. shift == 0 truncates at the node.
                shift_factor = float(getattr(config, "SDF_FLAT_CAP_BIF_SHIFT_FACTOR", 0.0))
                if shift_factor != 0.0:
                    shift = (rad_at_t * shift_factor).astype(np.float32)
                else:
                    shift = np.float32(0.0)
                # Shift smoothstep so full cap (u=1) is at the (shifted)
                # endpoint, transition onset 2*h_soft inside the cylinder body,
                # ramping with smoothstep so the gradient stays bounded.
                u = np.clip(
                    (cap_amount - shift) / (2.0 * h_soft) + 1.0, 0.0, 1.0
                )
                soft_w = (u * u * (3.0 - 2.0 * u)).astype(sdf_caps.dtype, copy=False)
                hard_capped = np.maximum(sdf_caps, cap_amount)
                sdf_caps = sdf_caps + soft_w * (hard_capped - sdf_caps)

        row_max_r = np.maximum(r0, r1).max(axis=1)
        search_r = row_max_r * 3.0 + length_scale * 2.0
        invalid = dists > search_r[:, None]
        sdf_caps = np.where(invalid, 999.0, sdf_caps)

        owner_k = sdf_caps.argmin(axis=1)
        rng = np.arange(M)
        owner_seg_ids = seg_ids_mk[rng, owner_k]
        owner_radius = np.maximum(rad_at_t[rng, owner_k], 1e-6)

        adj_mask = adj_matrix[owner_seg_ids[:, None], seg_ids_mk]
        sdf_topo = np.where(adj_mask, sdf_caps, 999.0)
        same_seg_mask = seg_ids_mk == owner_seg_ids[:, None]

        sdf_owner_only = np.where(same_seg_mask, sdf_caps, 999.0)
        hard_sdf = sdf_owner_only.min(axis=1)

        rival_mask = adj_mask & ~same_seg_mask
        sdf_rival = np.where(rival_mask, sdf_caps, 999.0)
        rival_sdf = sdf_rival.min(axis=1)

        # Hoisted: owner global capsule index, rival selection + rival_radius.
        # Needed by the symmetric polynomial k, the t-projection gate, and
        # the multi-gate stack below.
        owner_global = indices[rng, owner_k]
        rival_k_global = sdf_rival.argmin(axis=1)
        rival_seg_ids_global = seg_ids_mk[rng, rival_k_global]
        rival_radius = np.maximum(rad_at_t[rng, rival_k_global], 1e-6)
        rival_global_caps = indices[rng, rival_k_global]
        has_rival = rival_sdf < 998.0
        if detailed:
            diag["owner_seg"][b_start:b_end] = owner_seg_ids.astype(np.int32, copy=False)

        gap = np.maximum(rival_sdf - hard_sdf, 0.0)
        blend_width = np.maximum(owner_radius * config.SMIN_PROXIMITY_BLEND_FACTOR, 1e-9)
        u = np.clip(gap / blend_width, 0.0, 1.0)
        blend_w = (1.0 - u) * (1.0 - u) * (1.0 + 2.0 * u)
        if detailed:
            diag["gate_proximity"][b_start:b_end] = blend_w.astype(np.float32, copy=False)

        if config.USE_CROSS_SECTION_BLEND_GATE:
            owner_global = indices[rng, owner_k]
            owner_is_junc = cap_is_junction[owner_global]
            if detailed:
                diag["gate_xs_active"][b_start:b_end] = owner_is_junc.astype(np.uint8, copy=False)
            blend_w = blend_w * owner_is_junc.astype(blend_w.dtype, copy=False)

        if config.BIF_BLEND_ENABLE and bif.tree is not None and len(bif.radii) > 0:
            d_bif_blend, i_bif_blend = bif.tree.query(batch_coords)
            r_bif_blend = bif.radii[i_bif_blend]
            in_bif_ball_blend = d_bif_blend < (
                config.BIF_BLEND_RADIUS_FACTOR * r_bif_blend
            )
            if detailed:
                diag["gate_bif_ball"][b_start:b_end] = in_bif_ball_blend.astype(np.uint8, copy=False)
            blend_w = blend_w * in_bif_ball_blend.astype(blend_w.dtype, copy=False)

        # Intersection-curve localization: smin fires only where the voxel is
        # simultaneously inside the band of both the owner-segment surface and
        # at least one adjacent rival's surface. Restricts the smin contribution
        # to a tube of half-thickness SMIN_INTERSECTION_BAND_MM around the
        # carina edge (the geometric intersection of the two capsule surfaces).
        band_mm = float(getattr(config, "SMIN_INTERSECTION_BAND_MM", 0.0))
        if band_mm > 0.0:
            # Smoothstep weight in [0, 1] that ramps from 1.0 at the carina
            # center (both |sdf| ~ 0) down to 0.0 at the band edge. Continuous
            # and C1-smooth across the band boundary, so blend_w doesn't jump
            # there and the smin <-> hard-min transition doesn't produce
            # ridges in the iso=0 surface.
            farther = np.maximum(np.abs(hard_sdf), np.abs(rival_sdf))
            u_inter = np.clip(1.0 - farther / band_mm, 0.0, 1.0)
            soft_inter = u_inter * u_inter * (3.0 - 2.0 * u_inter)
            if detailed:
                diag["gate_intersection"][b_start:b_end] = (
                    (soft_inter * 255.0).astype(np.uint8, copy=False)
                )
            blend_w = blend_w * soft_inter.astype(blend_w.dtype, copy=False)

        # Parallel-adjacent-rival diagnostics. Hoisted so both the blend
        # gate (this block) and the SDF_CARVE_ADJACENT_PARALLEL carve
        # (below) can reuse is_parallel_blend / in_tight_bif.
        need_parallel = (
            config.BLEND_PARALLEL_RIVAL_GATE_ENABLE
            or config.SDF_CARVE_ADJACENT_PARALLEL
        )
        if need_parallel and rival_mask.any():
            rival_k_blend = sdf_rival.argmin(axis=1)
            rival_global_blend = indices[rng, rival_k_blend]
            owner_global_blend = indices[rng, owner_k]
            rival_seg_ids = seg_ids_mk[rng, rival_k_blend]
            cos_angle_blend = np.abs(
                np.einsum(
                    "ij,ij->i",
                    cap_tangents[owner_global_blend],
                    cap_tangents[rival_global_blend],
                )
            )
            is_parallel_blend = (
                cos_angle_blend > config.BLEND_PARALLEL_COS_THRESHOLD
            )

            # Continuous "outside_tight_bif_w" in [0, 1]: 0 inside the
            # carina protect ball, 1 well outside. Smoothstep ramp across
            # BLEND_PARALLEL_BIF_PROTECT_RAMP_FACTOR * r_bif. Binary mask
            # if RAMP_FACTOR == 0.
            if bif.tree is not None and len(bif.radii) > 0:
                d_bif_tight, i_bif_tight = bif.tree.query(batch_coords)
                r_bif_tight = bif.radii[i_bif_tight]
                inner_r = config.BLEND_PARALLEL_BIF_PROTECT_FACTOR * r_bif_tight
                ramp_factor = float(config.BLEND_PARALLEL_BIF_PROTECT_RAMP_FACTOR)
                if ramp_factor > 0.0:
                    outer_r = inner_r + ramp_factor * r_bif_tight
                    u_b = np.clip(
                        (d_bif_tight - inner_r) / np.maximum(outer_r - inner_r, 1e-9),
                        0.0, 1.0,
                    )
                    outside_tight_bif_w = u_b * u_b * (3.0 - 2.0 * u_b)
                else:
                    outside_tight_bif_w = (d_bif_tight >= inner_r).astype(np.float64)
            else:
                outside_tight_bif_w = np.ones(M, dtype=np.float64)

            # Continuous wedge_pass in [0, 1]: smoothstep product of the
            # two downstream half-spaces. Binary mask if RAMP_MM == 0.
            wedge_pass = np.ones(M, dtype=np.float64)
            if (
                config.BIF_WEDGE_GATE_ENABLE
                and shared_node_pos is not None
                and shared_node_has is not None
                and seg_end_pos is not None
                and seg_end_tan is not None
                and seg_end_tan_ok is not None
            ):
                shared_has = shared_node_has[owner_seg_ids, rival_seg_ids]
                if shared_has.any():
                    shared_pos = shared_node_pos[owner_seg_ids, rival_seg_ids]
                    v = batch_coords - shared_pos

                    owner_end_pos = seg_end_pos[owner_seg_ids]
                    owner_end_tan = seg_end_tan[owner_seg_ids]
                    owner_end_ok = seg_end_tan_ok[owner_seg_ids]
                    owner_d0 = np.linalg.norm(shared_pos - owner_end_pos[:, 0, :], axis=1)
                    owner_d1 = np.linalg.norm(shared_pos - owner_end_pos[:, 1, :], axis=1)
                    owner_use0 = owner_d0 <= owner_d1
                    owner_tan = np.where(
                        owner_use0[:, None],
                        owner_end_tan[:, 0, :],
                        owner_end_tan[:, 1, :],
                    )
                    owner_ok = np.where(owner_use0, owner_end_ok[:, 0], owner_end_ok[:, 1])

                    rival_end_pos = seg_end_pos[rival_seg_ids]
                    rival_end_tan = seg_end_tan[rival_seg_ids]
                    rival_end_ok = seg_end_tan_ok[rival_seg_ids]
                    rival_d0 = np.linalg.norm(shared_pos - rival_end_pos[:, 0, :], axis=1)
                    rival_d1 = np.linalg.norm(shared_pos - rival_end_pos[:, 1, :], axis=1)
                    rival_use0 = rival_d0 <= rival_d1
                    rival_tan = np.where(
                        rival_use0[:, None],
                        rival_end_tan[:, 0, :],
                        rival_end_tan[:, 1, :],
                    )
                    rival_ok = np.where(rival_use0, rival_end_ok[:, 0], rival_end_ok[:, 1])

                    wedge_ok = shared_has & owner_ok & rival_ok
                    if wedge_ok.any():
                        dot_owner = np.einsum("ij,ij->i", v, owner_tan)
                        dot_rival = np.einsum("ij,ij->i", v, rival_tan)
                        ramp_mm_floor = float(config.BIF_WEDGE_RAMP_MM)
                        ramp_radius_factor = float(
                            getattr(config, "BIF_WEDGE_RAMP_RADIUS_FACTOR", 0.0)
                        )
                        if ramp_radius_factor > 0.0:
                            r_local = np.minimum(owner_radius, rival_radius)
                            ramp_mm_eff = np.maximum(
                                ramp_mm_floor, r_local * ramp_radius_factor
                            )
                            use_smoothstep = True
                        else:
                            ramp_mm_eff = ramp_mm_floor
                            use_smoothstep = ramp_mm_floor > 0.0
                        if use_smoothstep:
                            u_o = np.clip(dot_owner / ramp_mm_eff + 0.5, 0.0, 1.0)
                            u_r = np.clip(dot_rival / ramp_mm_eff + 0.5, 0.0, 1.0)
                            ramp_o = u_o * u_o * (3.0 - 2.0 * u_o)
                            ramp_r = u_r * u_r * (3.0 - 2.0 * u_r)
                            wedge_w_inner = ramp_o * ramp_r
                        else:
                            wedge_w_inner = (
                                (dot_owner >= 0.0) & (dot_rival >= 0.0)
                            ).astype(np.float64)
                        # 1.0 (no filter) where the wedge couldn't be computed.
                        wedge_pass = np.where(wedge_ok, wedge_w_inner, 1.0)
        else:
            is_parallel_blend = np.zeros(M, dtype=bool)
            outside_tight_bif_w = np.zeros(M, dtype=np.float64)
            wedge_pass = np.ones(M, dtype=np.float64)

        if config.BIF_WEDGE_GATE_ENABLE:
            _wedge_total_sum += float(wedge_pass.sum())
        if detailed:
            diag["gate_wedge"][b_start:b_end] = wedge_pass.astype(np.float32, copy=False)

        # Blend-path: continuous suppression of smooth-min, in proportion
        # to (is_parallel_blend) * (outside_tight_bif_w) * (wedge_pass).
        # Smooth ramps eliminate the SDF discontinuities that the previous
        # binary `np.where(suppress, 0, blend_w)` left behind.
        if config.BLEND_PARALLEL_RIVAL_GATE_ENABLE:
            suppress_mag = (
                is_parallel_blend.astype(blend_w.dtype)
                * outside_tight_bif_w.astype(blend_w.dtype)
            )
            if config.BIF_WEDGE_GATE_ENABLE:
                suppress_mag = suppress_mag * wedge_pass.astype(blend_w.dtype)
            if detailed:
                diag["gate_par_suppress"][b_start:b_end] = (
                    (suppress_mag > 0).astype(np.uint8, copy=False)
                )
            if suppress_mag.any():
                blend_w = blend_w * (blend_w.dtype.type(1.0) - suppress_mag)

        # ── T-projection smin gate (sdf_plan.md target) ─────────────────
        # Override the multi-gate blend_w with the t-projection fade when
        # the directed topology + segment arc lengths are available.
        # Cross-section / bif-ball / parallel-suppression were still
        # computed above so their side-effects (wedge diagnostics, carve
        # gating flags) remain consistent — only blend_w gets replaced.
        if (
            config.SMIN_GATE_VARIANT == "t_projection"
            and is_parent is not None
            and is_child is not None
            and is_sibling is not None
            and capsules.seg_L is not None
            and len(capsules.seg_L) > 0
        ):
            owner_t_cap = t[rng, owner_k]
            rival_t_cap = t[rng, rival_k_global]
            cap_arc_start_f = capsules.arc_start
            cap_arc_end_f = capsules.arc_end
            owner_abs_arc = (
                cap_arc_start_f[owner_global]
                + owner_t_cap
                * (cap_arc_end_f[owner_global] - cap_arc_start_f[owner_global])
            )
            rival_abs_arc = (
                cap_arc_start_f[rival_global_caps]
                + rival_t_cap
                * (cap_arc_end_f[rival_global_caps] - cap_arc_start_f[rival_global_caps])
            )
            L_owner = np.maximum(capsules.seg_L[owner_seg_ids], 1e-9)
            L_rival = np.maximum(capsules.seg_L[rival_seg_ids_global], 1e-9)
            t_seg_owner_raw = np.clip(owner_abs_arc / L_owner, 0.0, 1.0)
            t_seg_rival_raw = np.clip(rival_abs_arc / L_rival, 0.0, 1.0)

            if seg_end_pos is not None:
                # shared_node_pos[owner, rival] is the bif between the pair
                # (NaN where no shared node). Compare to each segment's own
                # endpoint positions to decide which end is at the bif.
                if shared_node_pos is not None:
                    shared_pos_pair = shared_node_pos[owner_seg_ids, rival_seg_ids_global]
                else:
                    shared_pos_pair = np.full((M, 3), np.nan, dtype=np.float64)
                owner_end_pos_pair = seg_end_pos[owner_seg_ids]
                rival_end_pos_pair = seg_end_pos[rival_seg_ids_global]
                owner_d0 = np.linalg.norm(
                    shared_pos_pair - owner_end_pos_pair[:, 0, :], axis=1
                )
                owner_d1 = np.linalg.norm(
                    shared_pos_pair - owner_end_pos_pair[:, 1, :], axis=1
                )
                rival_d0 = np.linalg.norm(
                    shared_pos_pair - rival_end_pos_pair[:, 0, :], axis=1
                )
                rival_d1 = np.linalg.norm(
                    shared_pos_pair - rival_end_pos_pair[:, 1, :], axis=1
                )
                # If shared_pos is NaN (no shared node), these comparisons
                # produce NaN; treat as bif-at-start by default.
                owner_bif_at_start = ~(owner_d0 > owner_d1)
                rival_bif_at_start = ~(rival_d0 > rival_d1)
                t_seg_owner = np.where(
                    owner_bif_at_start, t_seg_owner_raw, 1.0 - t_seg_owner_raw
                )
                t_seg_rival = np.where(
                    rival_bif_at_start, t_seg_rival_raw, 1.0 - t_seg_rival_raw
                )
            else:
                t_seg_owner = t_seg_owner_raw
                t_seg_rival = t_seg_rival_raw

            is_par = is_parent[owner_seg_ids, rival_seg_ids_global]
            is_chi = is_child[owner_seg_ids, rival_seg_ids_global]
            is_sib = is_sibling[owner_seg_ids, rival_seg_ids_global]

            # t_relevant per relation (parent: t_active; child: t_other;
            # sibling: max). Unrelated rows stay at +inf so the fade is 0.
            t_relevant = np.full(M, np.inf, dtype=np.float64)
            t_relevant = np.where(is_par, t_seg_owner, t_relevant)
            t_relevant = np.where(is_chi, t_seg_rival, t_relevant)
            t_relevant = np.where(
                is_sib, np.maximum(t_seg_owner, t_seg_rival), t_relevant
            )

            fade = (1.0 - np.clip(t_relevant * 0.5, 0.0, 1.0)) ** 2
            any_relation = is_par | is_chi | is_sib
            fade = np.where(any_relation & has_rival, fade, 0.0)
            blend_w = fade.astype(blend_w.dtype, copy=False)

            if detailed:
                diag["t_seg_owner"][b_start:b_end] = t_seg_owner.astype(np.float32, copy=False)
                diag["t_seg_rival"][b_start:b_end] = t_seg_rival.astype(np.float32, copy=False)
                # 0 unrelated, 1 parent, 2 child, 3 sibling
                rel_code = np.zeros(M, dtype=np.uint8)
                rel_code = np.where(is_par, np.uint8(1), rel_code)
                rel_code = np.where(is_chi, np.uint8(2), rel_code)
                rel_code = np.where(is_sib, np.uint8(3), rel_code)
                diag["t_relation"][b_start:b_end] = rel_code

        if config.SMIN_VARIANT == "polynomial":
            # Symmetric k per sdf_plan.md: k = min(r_owner, r_rival) * factor.
            # Falls back to owner_radius when there's no rival.
            k_pair = np.where(
                has_rival,
                np.minimum(owner_radius, rival_radius),
                owner_radius,
            )
            k_poly = np.maximum(k_pair * config.SMIN_POLY_K_FACTOR, 1e-6)
            smooth_sdf = smooth_min_poly_pair(hard_sdf, rival_sdf, k_poly)
            if config.BLEND_BULGE_CAP_MM > 0:
                floor = np.minimum(hard_sdf, rival_sdf) - config.BLEND_BULGE_CAP_MM
                knee = config.BLEND_BULGE_CAP_SOFT_KNEE_MM
                if knee > 0.0:
                    smooth_sdf = -soft_cap(-smooth_sdf, -floor, knee)
                else:
                    smooth_sdf = np.maximum(smooth_sdf, floor)
        else:
            if config.SMIN_ADAPTIVE_BLEND:
                k_local = np.clip(
                    config.BSPLINE_SMIN_K * config.SMIN_K_REF_RADIUS / owner_radius,
                    config.SMIN_K_MIN,
                    config.SMIN_K_MAX,
                )
            else:
                k_local = np.full(M, config.BSPLINE_SMIN_K, dtype=np.float64)

            v_min = sdf_topo.min(axis=1)
            diff = sdf_topo - v_min[:, None]
            diff = np.minimum(diff, 50.0)
            exp_terms = np.exp(-k_local[:, None] * diff)
            exp_sum = exp_terms.sum(axis=1)
            depression = np.log(np.maximum(exp_sum, 1e-30)) / k_local
            if config.BLEND_BULGE_CAP_MM > 0:
                cap_eff = config.BLEND_BULGE_CAP_MM
                if config.BLEND_BULGE_CAP_RADIUS_FACTOR > 0.0:
                    cap_eff = np.maximum(
                        config.BLEND_BULGE_CAP_MM,
                        owner_radius * config.BLEND_BULGE_CAP_RADIUS_FACTOR,
                    )
                depression = soft_cap(
                    depression, cap_eff, config.BLEND_BULGE_CAP_SOFT_KNEE_MM
                )
            smooth_sdf = v_min - depression

        if config.SMOOTH_MIN_INTERIOR_ONLY:
            inside_intersection = (hard_sdf < 0.0) & (rival_sdf < 0.0)
            blend_w = np.where(inside_intersection, blend_w, 0.0)

        # Diagnostic mode: bypass smooth-min by forcing blend_w = 0. Smooth-min
        # arrays above were still computed so the diag counters / fields stay
        # consistent across runs.
        if config.FORCE_HARD_MIN_ONLY:
            blend_w = np.zeros_like(blend_w)

        if detailed:
            diag["final_blend_w"][b_start:b_end] = blend_w.astype(np.float32, copy=False)

        sdf_final = hard_sdf + blend_w * (smooth_sdf - hard_sdf)

        if detailed:
            diag["smin_depression"][b_start:b_end] = (
                (hard_sdf - sdf_final).astype(np.float32, copy=False)
            )
            # Snapshot SDF before any carve so we can compute the total push.
            sdf_pre_carves = sdf_final.copy()

        # ── Anti-bridge carve (patches 81-87) ────────────────────────────
        if config.SDF_CARVE_NON_ADJACENT:
            any_rival_mask = ~same_seg_mask
            sdf_any_rival = np.where(any_rival_mask, sdf_caps, np.inf)
            any_rival_min = sdf_any_rival.min(axis=1)

            non_adj_mask = (~adj_mask) & (~same_seg_mask)
            sdf_non_adj = np.where(non_adj_mask, sdf_caps, np.inf)
            non_adj_min = sdf_non_adj.min(axis=1)

            # Promoted out of the detailed-only branch: needed by the
            # topology-aware shield (and still recorded in diag).
            non_adj_k = np.argmin(sdf_non_adj, axis=1)
            has_non_adj = np.isfinite(non_adj_min)
            non_adj_rival_seg = np.where(
                has_non_adj,
                seg_ids_mk[rng, non_adj_k].astype(np.int32, copy=False),
                np.int32(-1),
            )
            if detailed:
                diag["non_adj_min"][b_start:b_end] = non_adj_min.astype(np.float32, copy=False)
                diag["non_adj_rival_seg"][b_start:b_end] = non_adj_rival_seg

            # Shared carve gates — hoisted so both the wall-band carve and
            # the patch-88 deep-merger carve below can reference them.
            owner_global = indices[rng, owner_k]
            if config.USE_CROSS_SECTION_BLEND_GATE and config.SDF_CARVE_HONOR_JUNCTION_GATE:
                owner_is_junc = cap_is_junction[owner_global]
            else:
                owner_is_junc = np.zeros(M, dtype=bool)
            if (
                config.SDF_CARVE_PROTECT_JUNCTION_BALL
                and bif.tree is not None
                and len(bif.radii) > 0
            ):
                d_bif, i_bif = bif.tree.query(batch_coords)
                r_bif = bif.radii[i_bif]
                in_jball = d_bif < (config.SDF_CARVE_JUNCTION_PROTECT_FACTOR * r_bif)
            else:
                d_bif = None
                i_bif = None
                in_jball = np.zeros(M, dtype=bool)
            if detailed:
                diag["gate_jprotect"][b_start:b_end] = in_jball.astype(np.uint8, copy=False)

            # Patch 87: tangent-angle gate — hoisted so rival_global is
            # available for the topology-aware shield's wall-band rival.
            rival_k_local = np.argmin(sdf_any_rival, axis=1)
            rival_global = indices[rng, rival_k_local]
            rival_seg_for_shield = seg_ids_mk[rng, rival_k_local]

            # Topology-aware shield: relax in_jball when the offending rival
            # is NOT incident on the nearest bif. Wall-band uses any-rival
            # (rival_seg_for_shield from sdf_any_rival.argmin); deep-merger
            # uses non_adj_rival_seg. Falls back to the geometric shield
            # when bif_seg_incident isn't provided (legacy behaviour).
            use_topo_shield = (
                config.SDF_CARVE_PROTECT_JUNCTION_BALL_TOPO_AWARE
                and bif_seg_incident is not None
                and i_bif is not None
                and bif_seg_incident.size > 0
            )
            if use_topo_shield:
                # Clamp seg ids to valid index range (some rivals could be -1
                # for the non-adj case, which we map to "not incident" / 0).
                _n_segs_bsi = bif_seg_incident.shape[1]
                _wb_rs = np.clip(rival_seg_for_shield, 0, _n_segs_bsi - 1)
                wb_rival_at_bif = bif_seg_incident[i_bif, _wb_rs]
                _deep_rs = np.where(non_adj_rival_seg >= 0, non_adj_rival_seg, 0)
                _deep_rs = np.clip(_deep_rs, 0, _n_segs_bsi - 1)
                deep_rival_at_bif = bif_seg_incident[i_bif, _deep_rs] & (non_adj_rival_seg >= 0)
                wb_shield = in_jball & wb_rival_at_bif
                deep_shield = in_jball & deep_rival_at_bif
            else:
                wb_shield = in_jball
                deep_shield = in_jball
            if detailed:
                diag["gate_jprotect_topo"][b_start:b_end] = (
                    wb_shield.astype(np.uint8, copy=False)
                )

            band = config.SDF_CARVE_WALL_BAND_FACTOR * owner_radius
            in_bridge_pre = (
                (hard_sdf > -band)
                & (any_rival_min > -band)
                & (any_rival_min < 0.0)
            )
            _band_pre += int(in_bridge_pre.sum())
            if detailed:
                diag["carve_wb_pre"][b_start:b_end] = in_bridge_pre.astype(np.uint8, copy=False)
            in_bridge = in_bridge_pre & (~owner_is_junc) & (~wb_shield)

            # Patch 87: tangent-angle gate.
            owner_tan = cap_tangents[owner_global]
            rival_tan = cap_tangents[rival_global]
            cos_angle = np.abs(np.einsum("ij,ij->i", owner_tan, rival_tan))
            is_parallel = cos_angle > config.SDF_CARVE_PARALLEL_COS_THRESHOLD
            in_bridge = in_bridge & is_parallel

            _after_gate += int(in_bridge.sum())
            if in_bridge.any():
                sdf_before_carve = sdf_final.copy()
                sdf_final = np.where(
                    in_bridge, np.maximum(sdf_final, -any_rival_min), sdf_final
                )
                _actually += int((sdf_final != sdf_before_carve).sum())
                if detailed:
                    diag["carve_wb_act"][b_start:b_end] = (
                        (sdf_final != sdf_before_carve).astype(np.uint8, copy=False)
                    )

            # Patch 88: non-adjacent deep-merger carve.
            # The wall-band carve above gates on any_rival_min > -band, so
            # it only fires when the rival's surface is close to the voxel.
            # When two non-adjacent segments deeply interpenetrate, the
            # rival surface is past the band and the wall-band carve
            # rejects them. Fix: any voxel inside a non-adjacent rival
            # should be pushed exterior so the mesh boundary follows the
            # midpoint between owner and rival rather than wrapping their
            # union into a bridge.
            # No junction-gate or bif-ball protection here: those exist
            # to protect adjacent-segment bif blending. Patch 88 only
            # fires when a non-adjacent rival is inside the voxel — by
            # construction it cannot disturb adjacent-segment bif logic,
            # because non_adj_min excludes adjacent rivals.
            # No parallel-angle gate either — deep overlap is inherently
            # non-tangential.
            # ~in_jball protects bif ostia from being punched through
            # when a non-adjacent third vessel happens to pass near the
            # bif — without this gate, patch-88 was creating small holes
            # at the ostia of bifurcations near unrelated sub-trees.
            if config.SDF_CARVE_NON_ADJACENT_DEEP:
                deep_min_overlap = float(config.SDF_CARVE_DEEP_MIN_OVERLAP_MM)
                deep_buf = float(config.SDF_CARVE_DEEP_BUFFER_VOXELS) * length_scale
                in_deep = (non_adj_min < -deep_min_overlap) & (~deep_shield)
                _deep_pre += int(in_deep.sum())
                if detailed:
                    diag["carve_deep_pre"][b_start:b_end] = in_deep.astype(np.uint8, copy=False)
                if in_deep.any():
                    sdf_before_deep = sdf_final.copy()
                    sdf_final = np.where(
                        in_deep,
                        np.maximum(sdf_final, -non_adj_min + deep_buf),
                        sdf_final,
                    )
                    _deep_actually += int((sdf_final != sdf_before_deep).sum())
                    if detailed:
                        diag["carve_deep_act"][b_start:b_end] = (
                            (sdf_final != sdf_before_deep).astype(np.uint8, copy=False)
                        )

            # Regime (b): tangent-touch buffer for non-adjacent rivals.
            tang_buf = config.SDF_CARVE_TANGENT_BUFFER_VOXELS * length_scale
            if tang_buf > 0:
                near_tangent = (non_adj_min < tang_buf) & (sdf_final < tang_buf)
                if near_tangent.any():
                    sdf_final = np.where(
                        near_tangent,
                        np.maximum(sdf_final, tang_buf - non_adj_min),
                        sdf_final,
                    )

            # Last-resort force-gap: unconditionally push SDF positive at any
            # voxel where a non-adjacent rival is genuinely inside. Bypasses
            # all gates (including the topology-aware shield) — risks
            # puncturing legitimate geometry, so it's off by default.
            fg = float(getattr(config, "FORCE_NON_ADJ_GAP_MM", 0.0))
            if fg > 0.0:
                fg_fire = non_adj_min < 0.0
                if fg_fire.any():
                    sdf_before_fg = sdf_final.copy()
                    sdf_final = np.where(
                        fg_fire,
                        np.maximum(sdf_final, np.float32(fg)),
                        sdf_final,
                    )
                    if detailed:
                        diag["carve_total_push"][b_start:b_end] = (
                            diag["carve_total_push"][b_start:b_end]
                            + (sdf_final - sdf_before_fg).astype(np.float32, copy=False)
                        )

        # ── Adjacent-parallel anti-bridge carve ──────────────────────────
        # Hard-min path analogue of the BLEND_PARALLEL_RIVAL_GATE.
        # Adjacent rivals (parent+sidebranch share a bif node) are excluded
        # from SDF_CARVE_NON_ADJACENT by design. When two adjacent vessels
        # run near-parallel and close, their capsule SDFs both go negative
        # at the same voxel -> union is fused even with blend_w=0. Carve
        # the rival's interior out so the surface follows the owner's wall.
        # Gated by ~in_tight_bif so the carina fillet at real bifurcations
        # is preserved.
        if config.SDF_CARVE_ADJACENT_PARALLEL and rival_mask.any():
            band_adj = config.SDF_CARVE_WALL_BAND_FACTOR * owner_radius
            # Only carve where both owner and rival are inside but close to
            # their respective walls (wall-band overlap). This avoids carving
            # the opposite wall when a parallel rival overlaps deeply.
            in_adj_bridge_pre = (
                (hard_sdf < 0.0)
                & (hard_sdf > -band_adj)
                & (rival_sdf < 0.0)
                & (rival_sdf > -band_adj)
            )
            _adj_pre += int(in_adj_bridge_pre.sum())
            if detailed:
                diag["carve_adj_pre"][b_start:b_end] = (
                    in_adj_bridge_pre.astype(np.uint8, copy=False)
                )
            # Continuous carve weight in [0, 1] built from the same gates
            # used by the blend path. Mix between original and -rival_sdf
            # so the SDF transitions smoothly across the wedge / bif-ball
            # boundaries instead of the binary `np.where` step that the
            # previous version used (which created ridges/bulges).
            carve_w = (
                in_adj_bridge_pre.astype(np.float64)
                * is_parallel_blend.astype(np.float64)
                * outside_tight_bif_w
            )
            if config.BIF_WEDGE_GATE_ENABLE:
                carve_w = carve_w * wedge_pass
            if carve_w.any():
                sdf_before_adj = sdf_final.copy()
                carve_target = np.maximum(sdf_final, -rival_sdf)
                sdf_final = (
                    sdf_final.astype(np.float64)
                    + carve_w * (carve_target.astype(np.float64) - sdf_final.astype(np.float64))
                ).astype(sdf_final.dtype)
                _adj_actually += int((sdf_final != sdf_before_adj).sum())
                if detailed:
                    diag["carve_adj_act"][b_start:b_end] = (
                        (sdf_final != sdf_before_adj).astype(np.uint8, copy=False)
                    )

        if detailed:
            diag["carve_total_push"][b_start:b_end] = (
                (sdf_final - sdf_pre_carves).astype(np.float32, copy=False)
            )

        # ── SDF-level flat-cap clamp ─────────────────────────────────────
        if term.tree is not None:
            td, ti = term.tree.query(batch_coords)
            reach = term.rad[ti] * config.SDF_FLAT_CAP_REACH_FACTOR
            within = td < reach
            if within.any():
                beyond = np.sum((batch_coords - term.pos[ti]) * term.nrm[ti], axis=1)
                mask = within & (beyond > 0.0)
                if mask.any():
                    sdf_final = np.where(mask, np.maximum(sdf_final, beyond), sdf_final)

        sdf_flat[b_start:b_end] = sdf_final
        if path_flat is not None and blend_weight_flat is not None:
            path_flat[b_start:b_end] = np.where(blend_w >= 0.05, 3, 1).astype(np.uint8)
            blend_weight_flat[b_start:b_end] = blend_w.astype(np.float32)

        if progress:
            elapsed = time.time() - t2
            pct = 100 * b_end / n_band
            rate = b_end / max(elapsed, 0.001)
            eta = (n_band - b_end) / max(rate, 1)
            print(f"    {b_end:,}/{n_band:,} ({pct:.0f}%) - {elapsed:.1f}s, ETA: {eta:.0f}s")

    print(f"  SDF evaluation: {time.time()-t2:.1f}s")
    if config.SDF_CARVE_NON_ADJACENT:
        print(f"  [carve] band-sandwich voxels:        {_band_pre:,}")
        print(
            f"  [carve] after junction gate:          {_after_gate:,}"
            f"  (gate honoured={config.SDF_CARVE_HONOR_JUNCTION_GATE})"
        )
        print(f"  [carve] voxels actually pushed exterior: {_actually:,}")
        if _band_pre > 0:
            print(f"  [carve] effective fraction: {100.0*_actually/_band_pre:.1f}%")
        if config.SDF_CARVE_NON_ADJACENT_DEEP:
            print(f"  [carve88] deep-merger voxels inside non-adj rival: {_deep_pre:,}")
            print(f"  [carve88] voxels actually pushed exterior:         {_deep_actually:,}")
    if config.SDF_CARVE_ADJACENT_PARALLEL:
        print(f"  [carve-adj] adjacent rival inside owner band:      {_adj_pre:,}")
        print(f"  [carve-adj] voxels actually pushed exterior:       {_adj_actually:,}")
    if config.BIF_WEDGE_GATE_ENABLE:
        print(
            f"  [wedge] mean wedge_pass over all band voxels: "
            f"{_wedge_total_sum / max(n_band, 1):.3f}"
        )

    return sdf_flat, path_flat, blend_weight_flat, diag


def evaluate_sdf_points(
    points_mm: np.ndarray,
    *,
    length_scale: float,
    capsules: CapsuleArrays,
    cap_is_junction: np.ndarray,
    adj_matrix: np.ndarray,
    bif: BifurcationSet,
    term: TerminalSet,
    shared_node_pos: np.ndarray | None = None,
    shared_node_has: np.ndarray | None = None,
    seg_end_pos: np.ndarray | None = None,
    seg_end_tan: np.ndarray | None = None,
    seg_end_tan_ok: np.ndarray | None = None,
    is_parent: np.ndarray | None = None,
    is_child: np.ndarray | None = None,
    is_sibling: np.ndarray | None = None,
    bif_seg_incident: np.ndarray | None = None,
    batch_size: int = 1_000_000,
    progress: bool = False,
) -> np.ndarray:
    """Point-query form of the legacy field, for grid-free extractors.

    Raises when Gaussian post-smoothing is enabled: that step is defined on a
    voxel grid and cannot be reproduced pointwise, so silently omitting it would
    make this a *different* field from the dense path and invalidate any
    comparison between the two.
    """

    if config.SDF_GAUSSIAN_SIGMA_VOXELS > 0:
        raise ValueError(
            "evaluate_sdf_points cannot reproduce SDF_GAUSSIAN_SIGMA_VOXELS="
            f"{config.SDF_GAUSSIAN_SIGMA_VOXELS}; it is a grid-shaped operation. "
            "Set it to 0 to compare the legacy field against a grid-free extractor."
        )
    values, _paths, _weights, _diag = _evaluate_legacy_kernel(
        points_mm,
        length_scale=length_scale,
        capsules=capsules,
        cap_is_junction=cap_is_junction,
        adj_matrix=adj_matrix,
        bif=bif,
        term=term,
        shared_node_pos=shared_node_pos,
        shared_node_has=shared_node_has,
        seg_end_pos=seg_end_pos,
        seg_end_tan=seg_end_tan,
        seg_end_tan_ok=seg_end_tan_ok,
        is_parent=is_parent,
        is_child=is_child,
        is_sibling=is_sibling,
        bif_seg_incident=bif_seg_incident,
        batch_size=batch_size,
        progress=progress,
    )
    return values


def evaluate_sdf(
    capsules: CapsuleArrays,
    cap_is_junction: np.ndarray,
    adj_matrix: np.ndarray,
    bif: BifurcationSet,
    term: TerminalSet,
    grid: Grid,
    nb_idx: np.ndarray,
    shared_node_pos: np.ndarray | None = None,
    shared_node_has: np.ndarray | None = None,
    seg_end_pos: np.ndarray | None = None,
    seg_end_tan: np.ndarray | None = None,
    seg_end_tan_ok: np.ndarray | None = None,
    is_parent: np.ndarray | None = None,
    is_child: np.ndarray | None = None,
    is_sibling: np.ndarray | None = None,
    bif_seg_incident: np.ndarray | None = None,
    batch_size: int = 1_000_000,
) -> SdfVolume:
    """Batched narrow-band SDF eval on ``grid``.

    Thin driver over :func:`_evaluate_legacy_kernel`: it turns narrow-band voxel
    indices into world coordinates, pins the field's internal length scale to
    the grid spacing (the historical behaviour), scatters the result back into a
    dense volume, and applies the optional grid-shaped Gaussian smoothing.
    """

    dims = tuple(grid.dims)
    nb_idx = np.asarray(nb_idx)
    coords = np.column_stack(
        [grid.x[nb_idx[:, 0]], grid.y[nb_idx[:, 1]], grid.z[nb_idx[:, 2]]]
    )
    sdf_flat, path_flat, blend_flat, diag = _evaluate_legacy_kernel(
        coords,
        length_scale=grid.voxel_size,
        capsules=capsules,
        cap_is_junction=cap_is_junction,
        adj_matrix=adj_matrix,
        bif=bif,
        term=term,
        shared_node_pos=shared_node_pos,
        shared_node_has=shared_node_has,
        seg_end_pos=seg_end_pos,
        seg_end_tan=seg_end_tan,
        seg_end_tan_ok=seg_end_tan_ok,
        is_parent=is_parent,
        is_child=is_child,
        is_sibling=is_sibling,
        bif_seg_incident=bif_seg_incident,
        batch_size=batch_size,
        progress=True,
    )

    iv, jv, kv = nb_idx[:, 0], nb_idx[:, 1], nb_idx[:, 2]
    sdf_vol = np.full(dims, 10.0, dtype=np.float32)
    sdf_vol[iv, jv, kv] = sdf_flat
    path_vol = blend_weight_vol = None
    if path_flat is not None and blend_flat is not None:
        path_vol = np.zeros(dims, dtype=np.uint8)
        blend_weight_vol = np.zeros(dims, dtype=np.float32)
        path_vol[iv, jv, kv] = path_flat
        blend_weight_vol[iv, jv, kv] = blend_flat

    # Optional Gaussian smoothing. Grid-shaped by definition, so it stays in the
    # dense driver rather than the point kernel.
    if config.SDF_GAUSSIAN_SIGMA_VOXELS > 0:
        from scipy.ndimage import gaussian_filter

        t_gs = time.time()
        sdf_vol = gaussian_filter(
            sdf_vol, sigma=config.SDF_GAUSSIAN_SIGMA_VOXELS, mode="nearest"
        ).astype(np.float32)
        print(
            f"  SDF Gaussian smoothing (sigma={config.SDF_GAUSSIAN_SIGMA_VOXELS} "
            f"voxels): {time.time()-t_gs:.1f}s"
        )
    detailed = bool(getattr(config, "DETAILED_BLEND_DIAGNOSTIC", False))
    return SdfVolume(
        sdf=sdf_vol,
        path_vol=path_vol,
        blend_weight_vol=blend_weight_vol,
        nb_idx=nb_idx if detailed else None,
        diag=diag if detailed else None,
    )


__all__ = [
    "smooth_min_exp",
    "adaptive_smin_k",
    "collect_endpoint_info",
    "TerminalSet",
    "build_terminal_set",
    "BifurcationSet",
    "find_bifurcations",
    "build_adjacency",
    "report_non_adjacent_proximity",
    "Grid",
    "compute_grid",
    "build_narrow_band",
    "SdfVolume",
    "evaluate_sdf",
    "evaluate_sdf_points",
]
