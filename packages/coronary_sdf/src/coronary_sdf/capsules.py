"""Per-segment capsule sampling + spatial index.

A "capsule" is a tapered line segment (p0, p1) with radii (r0, r1) used
as the primitive for the SDF field. Each spline (one per vessel
segment) is sampled into capsules spanning consecutive smoothed
centerline points.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.spatial import KDTree

from .config import runtime_config as config


@dataclass
class CapsuleArrays:
    """Flat arrays of all capsules in a graph, plus the KDTree of midpoints."""

    starts: np.ndarray         # (N, 3)
    ends: np.ndarray           # (N, 3)
    radii_start: np.ndarray    # (N,)
    radii_end: np.ndarray      # (N,)
    seg_idx: np.ndarray        # (N,) int — segment index per capsule
    midpoints: np.ndarray      # (N, 3) — (starts + ends) / 2
    tangents: np.ndarray       # (N, 3) — unit (ends - starts)
    max_radii: np.ndarray      # (N,) — max(r_start, r_end)
    tree: KDTree               # spatial index over midpoints
    # Per-capsule arc position within its owning segment (mm). arc_start
    # is the cumulative arc length from coords[0] to the capsule's p0;
    # arc_end is to p1. Used by the t-projection smin gate to compute the
    # segment-level normalised t for the closest point on the capsule.
    arc_start: np.ndarray      # (N,) float64
    arc_end: np.ndarray        # (N,) float64
    # Per-segment total arc length (mm), indexed by seg_idx. Zero for
    # segments not present in this CapsuleArrays.
    seg_L: np.ndarray          # (n_segs,) float64
    # Per-capsule flags: True iff the capsule's p0 (resp. p1) endpoint sits
    # at a bif-incident node (degree >= 3). Only the first capsule of a
    # spline can have cap_bif_at_start=True; only the last can have
    # cap_bif_at_end=True. Consumed by sdf_field.evaluate_sdf to apply a
    # planar truncation at bif-incident capsule endpoints when
    # SDF_FLAT_CAP_BIF is True.
    cap_bif_at_start: np.ndarray  # (N,) bool
    cap_bif_at_end: np.ndarray    # (N,) bool

    @property
    def n(self) -> int:
        return len(self.starts)


def clamp_terminal_capsule_radii(
    valid_splines: list[dict[str, Any]],
    terminal_node_ids: set[int],
) -> dict[str, Any]:
    """Force terminal-end radii of each spline up to the median of the
    next ``TERMINAL_CAPSULE_CLAMP_LOOKAHEAD`` interior radii.

    Mutates ``sp["radii"]`` in-place for every spline whose ``node1_id``
    or ``node2_id`` is in ``terminal_node_ids``. Unconditional (no
    threshold) — the terminal endpoint radius cannot be smaller than the
    median of the lookahead interior radii. If it is already above the
    median, it is left alone. No-op when
    ``FORCE_TERMINAL_CAPSULE_NO_SHRINK`` is False.

    Returns a diagnostic dict::

        {
            "n_clamped": int,         # endpoints actually modified
            "n_checked": int,         # endpoints that matched terminal_node_ids
            "records":   list[dict],  # per-endpoint details
        }

    Each record contains ``seg_id``, ``node_id``, ``end`` ("start"|"end"),
    ``interior_ref``, ``radius_before``, ``radius_after``, ``clamped``.
    """
    report: dict[str, Any] = {"n_clamped": 0, "n_checked": 0, "records": []}
    if not config.FORCE_TERMINAL_CAPSULE_NO_SHRINK:
        return report
    look = int(config.TERMINAL_CAPSULE_CLAMP_LOOKAHEAD)
    if look < 1:
        return report
    for sp in valid_splines:
        radii = sp["radii"]
        n = len(radii)
        if n < 2:
            continue
        k = min(look, n - 1)
        seg_id = sp.get("seg_id")
        if sp["node1_id"] in terminal_node_ids:
            interior = float(np.median(radii[1:1 + k]))
            r_before = float(radii[0])
            clamped = r_before < interior
            if clamped:
                radii[0] = interior
                sp["start_radius"] = float(interior)
                report["n_clamped"] += 1
            report["n_checked"] += 1
            report["records"].append({
                "seg_id": seg_id,
                "node_id": sp["node1_id"],
                "end": "start",
                "interior_ref": interior,
                "radius_before": r_before,
                "radius_after": float(radii[0]),
                "clamped": clamped,
            })
        if sp["node2_id"] in terminal_node_ids:
            interior = float(np.median(radii[-1 - k:-1]))
            r_before = float(radii[-1])
            clamped = r_before < interior
            if clamped:
                radii[-1] = interior
                sp["end_radius"] = float(interior)
                report["n_clamped"] += 1
            report["n_checked"] += 1
            report["records"].append({
                "seg_id": seg_id,
                "node_id": sp["node2_id"],
                "end": "end",
                "interior_ref": interior,
                "radius_before": r_before,
                "radius_after": float(radii[-1]),
                "clamped": clamped,
            })
    return report


def build_capsules(
    valid_splines: list[dict[str, Any]],
    node_to_segs: dict[int, set[int]] | None = None,
) -> CapsuleArrays:
    """Sample every spline into tapered capsules and pack into flat arrays.

    Each capsule spans two consecutive smoothed centerline points
    (``sp["coords"][i]`` to ``sp["coords"][i+1]``) with linearly tapered
    radii from ``sp["radii"][i]`` to ``sp["radii"][i+1]``. Capsule count
    per segment is ``len(sp["coords"]) - 1``; capsule density is therefore
    controlled upstream by the centerline-smoothing knobs.

    Also computes per-capsule arc positions within their owning segment
    and the segment's total arc length, consumed by the t-projection smin
    gate in ``sdf_field.evaluate_sdf``.

    When ``node_to_segs`` is provided, the first/last capsule of each
    spline is tagged with ``cap_bif_at_start`` / ``cap_bif_at_end`` if
    the corresponding node has degree >= 3.
    """
    starts: list[np.ndarray] = []
    ends: list[np.ndarray] = []
    radii_start: list[float] = []
    radii_end: list[float] = []
    seg_idx: list[int] = []
    arc_start: list[float] = []
    arc_end: list[float] = []
    cap_bif_at_start_l: list[bool] = []
    cap_bif_at_end_l: list[bool] = []
    # Track max seg_idx for sizing seg_L. Splines that were filtered out
    # upstream keep their original seg_idx and leave 0.0 in seg_L; the
    # t-projection gate gracefully no-ops for those (it divides by max(L,
    # 1e-9) and the missing segments are never referenced as owner or
    # rival because they have no capsules).
    seg_L_by_idx: dict[int, float] = {}

    for sp in valid_splines:
        coords = sp["coords"]
        radii = sp["radii"]
        sidx = int(sp["seg_idx"])
        n_caps_here = max(0, len(coords) - 1)
        node1_is_bif = (
            node_to_segs is not None
            and len(node_to_segs.get(sp["node1_id"], set())) >= 3
        )
        node2_is_bif = (
            node_to_segs is not None
            and len(node_to_segs.get(sp["node2_id"], set())) >= 3
        )
        cum_arc = 0.0
        for i in range(n_caps_here):
            seg_len = float(np.linalg.norm(coords[i + 1] - coords[i]))
            starts.append(coords[i])
            ends.append(coords[i + 1])
            radii_start.append(float(radii[i]))
            radii_end.append(float(radii[i + 1]))
            seg_idx.append(sidx)
            arc_start.append(cum_arc)
            arc_end.append(cum_arc + seg_len)
            cum_arc += seg_len
            cap_bif_at_start_l.append(bool(node1_is_bif and i == 0))
            cap_bif_at_end_l.append(bool(node2_is_bif and i == n_caps_here - 1))
        seg_L_by_idx[sidx] = cum_arc

    starts_arr = np.asarray(starts, dtype=np.float64)
    ends_arr = np.asarray(ends, dtype=np.float64)
    radii_start_arr = np.asarray(radii_start, dtype=np.float64)
    radii_end_arr = np.asarray(radii_end, dtype=np.float64)
    seg_idx_arr = np.asarray(seg_idx, dtype=np.int64)
    arc_start_arr = np.asarray(arc_start, dtype=np.float64)
    arc_end_arr = np.asarray(arc_end, dtype=np.float64)
    midpoints = (starts_arr + ends_arr) / 2.0

    # Patch 87: per-capsule unit tangent for the carve angle gate.
    tan_raw = ends_arr - starts_arr
    tan_norm = np.linalg.norm(tan_raw, axis=1, keepdims=True)
    tangents = tan_raw / np.maximum(tan_norm, 1e-12)

    max_radii = np.maximum(radii_start_arr, radii_end_arr)
    tree = KDTree(midpoints)

    n_segs_needed = (int(seg_idx_arr.max()) + 1) if len(seg_idx_arr) else 0
    seg_L = np.zeros(n_segs_needed, dtype=np.float64)
    for sidx, L in seg_L_by_idx.items():
        if 0 <= sidx < n_segs_needed:
            seg_L[sidx] = L

    cap_bif_at_start_arr = np.asarray(cap_bif_at_start_l, dtype=bool)
    cap_bif_at_end_arr = np.asarray(cap_bif_at_end_l, dtype=bool)

    return CapsuleArrays(
        starts=starts_arr,
        ends=ends_arr,
        radii_start=radii_start_arr,
        radii_end=radii_end_arr,
        seg_idx=seg_idx_arr,
        midpoints=midpoints,
        tangents=tangents,
        max_radii=max_radii,
        tree=tree,
        arc_start=arc_start_arr,
        arc_end=arc_end_arr,
        seg_L=seg_L,
        cap_bif_at_start=cap_bif_at_start_arr,
        cap_bif_at_end=cap_bif_at_end_arr,
    )


def precompensate_capsule_radii(
    capsules: CapsuleArrays, voxel_size: float, coeff: float | None = None
) -> tuple[CapsuleArrays, float]:
    """Inflate capsule radii to cancel the marching-cubes radius deficit.

    Extracting a curved iso-surface on a finite grid places the surface slightly
    inside the true one. The error is second order in the voxel size,

        delta_r ~ -COEFF * h**2 / r

    so it is a few percent where ``r / h`` is small (the thinnest vessels) and
    negligible once ``r / h`` exceeds ~10. Adding ``+COEFF * h**2 / r`` to every
    capsule radius before the field is evaluated cancels the leading term.

    ``COEFF`` is geometry-dependent and must be measured for a cylinder through
    this exact pipeline -- a cylinder has half a sphere's mean curvature, so a
    sphere-fitted coefficient does not transfer. It is 0.0 (a no-op) until that
    measurement is made.

    ``compute_grid`` chooses ``voxel_size`` from the *uncompensated* radii, so
    there is a weak feedback loop here; it is second order in an already
    second-order correction and is deliberately not iterated.

    Mutates and returns ``capsules`` plus the maximum inflation applied (mm).
    ``midpoints``/``tangents``/``tree`` are radius-independent and stay valid.
    """
    c = float(config.SDF_RADIUS_PRECOMPENSATE_COEFF if coeff is None else coeff)
    h = float(voxel_size)
    if c == 0.0 or h <= 0.0 or capsules.n == 0:
        return capsules, 0.0

    delta_h2 = c * h * h

    def _inflate(r: np.ndarray) -> np.ndarray:
        # Guard r -> 0: a zero-radius capsule has no meaningful correction and
        # would otherwise diverge.
        return np.where(r > 0.0, r + delta_h2 / np.maximum(r, 1e-12), r)

    r0_new = _inflate(capsules.radii_start)
    r1_new = _inflate(capsules.radii_end)
    max_delta = float(
        np.max(
            np.concatenate(
                [r0_new - capsules.radii_start, r1_new - capsules.radii_end]
            )
        )
    )
    capsules.radii_start = r0_new
    capsules.radii_end = r1_new
    capsules.max_radii = np.maximum(r0_new, r1_new)
    return capsules, max_delta


__all__ = [
    "CapsuleArrays",
    "build_capsules",
    "clamp_terminal_capsule_radii",
    "precompensate_capsule_radii",
]
