"""Per-segment B-spline preparation + branch tangents.

A ``spline_data`` dict is produced for each valid segment:

        {
            "cs_pos":   CubicSpline (arc -> 3-vector position)
            "L":        float (segment arc length, mm)
            "coords":   (N, 3) smoothed centerline (mm)
            "radii":    (N,)  per-point radii (mm)
            "node1_id"/"node2_id":  endpoint node ids
            "start_radius"/"end_radius": endpoint radii
            "seg_id":   original segment id
        }
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.interpolate import CubicSpline

from .config import runtime_config as config


def compute_frenet_frame(tangent: np.ndarray, prev_normal: np.ndarray | None = None):
    """Stable Frenet-like frame (tangent, normal, binormal) given a tangent."""
    t = tangent / max(np.linalg.norm(tangent), 1e-12)

    if prev_normal is not None:
        n = prev_normal - np.dot(prev_normal, t) * t
        n_norm = np.linalg.norm(n)
        if n_norm > 1e-6:
            n = n / n_norm
        else:
            prev_normal = None

    if prev_normal is None:
        candidates = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        dots = np.abs(candidates @ t)
        best = candidates[np.argmin(dots)]
        n = best - np.dot(best, t) * t
        n = n / max(np.linalg.norm(n), 1e-12)

    b = np.cross(t, n)
    return t, n, b


def _apply_stride_sampling(
    coords: np.ndarray, radii: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    pct = float(config.CAPSULE_SAMPLE_STRIDE_PCT)
    if pct <= 0:
        return coords, radii
    n = len(coords)
    if n < 2:
        return coords, radii

    stride = int(round(n * pct / 100.0))
    stride = max(1, stride)

    max_pct = float(config.CAPSULE_SAMPLE_STRIDE_MAX_PCT)
    if max_pct > 0:
        max_stride = max(1, int(np.ceil(n * max_pct / 100.0)) - 1)
        if stride >= max_stride:
            stride = max_stride

    if stride <= 1:
        return coords, radii

    idx = np.arange(0, n, stride, dtype=np.int64)
    if idx[-1] != n - 1:
        idx = np.append(idx, n - 1)
    return coords[idx], radii[idx]


def _finalise_spline(
    coords: np.ndarray,
    radii: np.ndarray,
    node1_id: int,
    node2_id: int,
    seg_id: Any,
    strahler: Any = None,
) -> dict[str, Any] | None:
    """Build the spline dict from cleaned ``coords`` / ``radii`` arrays.

    Returns ``None`` if the trimmed arrays no longer admit a valid arc
    parameterisation (zero length or fewer than 2 points). Used by
    ``prepare_segment_spline`` and by ``bif_trim`` after points are
    dropped from a spline's endpoints.
    """
    if len(coords) < 2:
        return None

    diffs = np.diff(coords, axis=0)
    arc = np.zeros(len(coords))
    arc[1:] = np.cumsum(np.linalg.norm(diffs, axis=1))
    L = arc[-1]
    if L < 1e-12:
        return None

    cs_pos = CubicSpline(arc, coords, bc_type="not-a-knot")

    return {
        "cs_pos": cs_pos,
        "L": float(L),
        "coords": coords,
        "radii": radii,
        "node1_id": node1_id,
        "node2_id": node2_id,
        "start_radius": float(radii[0]),
        "end_radius": float(radii[-1]),
        "seg_id": seg_id,
        "strahler": strahler,
    }


def prepare_segment_spline(
    seg: dict[str, Any],
    points: dict[int, tuple],
    nodes: dict[int, tuple],
) -> dict[str, Any] | None:
    """Build the per-segment spline dict. Returns ``None`` for invalid input."""
    pids = seg["point_ids"]
    if len(pids) < 2:
        return None

    coords = np.array(
        [[points[p][0] / 1000.0, points[p][1] / 1000.0, points[p][2] / 1000.0] for p in pids]
    )
    radii = np.array([points[p][3] / 1000.0 * config.RADIUS_SCALE for p in pids])

    # Drop consecutive coincident points (required for CubicSpline).
    if len(coords) > 1:
        d = np.linalg.norm(np.diff(coords, axis=0), axis=1)
        keep = np.concatenate([[True], d > 1e-9])
        coords = coords[keep]
        radii = radii[keep]
    if len(coords) < 2:
        return None

    coords, radii = _apply_stride_sampling(coords, radii)
    if len(coords) < 2:
        return None

    # Centerline smoothing now runs upstream via smooth_segment_centerlines
    # so it happens before radius smoothing. Re-smoothing here would just
    # repeat work on already-smoothed coords.

    return _finalise_spline(
        coords, radii, seg["node1"], seg["node2"], seg["id"], strahler=seg.get("strahler")
    )


def branch_tangent_at_node(
    spline: dict[str, Any], node_id: int, seg: dict[str, Any]
) -> np.ndarray | None:
    """Unit tangent of the centerline pointing away from ``node_id``."""
    if seg["node1"] == node_id:
        tang = spline["cs_pos"](0, 1)
    else:
        tang = -spline["cs_pos"](spline["L"], 1)
    norm = np.linalg.norm(tang)
    return tang / norm if norm > 1e-10 else None


__all__ = [
    "compute_frenet_frame",
    "prepare_segment_spline",
    "branch_tangent_at_node",
    "_finalise_spline",
]
