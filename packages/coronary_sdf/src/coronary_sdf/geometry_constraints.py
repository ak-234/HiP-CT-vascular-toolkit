"""Scale-free feasibility checks for radius-labelled centreline capsules."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .capsules import CapsuleArrays


@dataclass(frozen=True)
class CapsuleConflict:
    capsule_a: int
    capsule_b: int
    segment_a: int
    segment_b: int
    parameter_a: float
    parameter_b: float
    centreline_distance: float
    radius_a: float
    radius_b: float
    clearance: float

    @property
    def normalized_clearance(self) -> float:
        return self.clearance / min(self.radius_a, self.radius_b)


def closest_segment_parameters(
    p0: np.ndarray, p1: np.ndarray, q0: np.ndarray, q1: np.ndarray
) -> tuple[float, float, float]:
    """Return clamped parameters and distance between two 3-D segments."""

    u = np.asarray(p1, dtype=np.float64) - np.asarray(p0, dtype=np.float64)
    v = np.asarray(q1, dtype=np.float64) - np.asarray(q0, dtype=np.float64)
    w = np.asarray(p0, dtype=np.float64) - np.asarray(q0, dtype=np.float64)
    a = float(np.dot(u, u))
    b = float(np.dot(u, v))
    c = float(np.dot(v, v))
    d = float(np.dot(u, w))
    e = float(np.dot(v, w))
    denominator = a * c - b * b
    small = 1e-30

    if a <= small and c <= small:
        s = t = 0.0
    elif a <= small:
        s = 0.0
        t = float(np.clip(e / c, 0.0, 1.0))
    elif c <= small:
        t = 0.0
        s = float(np.clip(-d / a, 0.0, 1.0))
    else:
        s = float(np.clip((b * e - c * d) / max(denominator, small), 0.0, 1.0))
        t = float(np.clip((a * e - b * d) / max(denominator, small), 0.0, 1.0))
        # Clamping one parameter changes the optimum of the other.
        for _ in range(2):
            s = float(np.clip((b * t - d) / a, 0.0, 1.0))
            t = float(np.clip((b * s + e) / c, 0.0, 1.0))
    delta = (p0 + s * u) - (q0 + t * v)
    return s, t, float(np.linalg.norm(delta))


def _share_endpoint(capsules: CapsuleArrays, i: int, j: int) -> bool:
    scale = max(float(capsules.max_radii[i]), float(capsules.max_radii[j]), 1.0)
    tolerance = np.finfo(np.float64).eps * scale * 256.0
    return any(
        np.linalg.norm(a - b) <= tolerance
        for a in (capsules.starts[i], capsules.ends[i])
        for b in (capsules.starts[j], capsules.ends[j])
    )


def _is_doubly_critical_pair(
    capsules: CapsuleArrays,
    i: int,
    j: int,
    s: float,
    t: float,
    tangent_start: np.ndarray,
    tangent_end: np.ndarray,
) -> bool:
    """Return whether a same-curve capsule pair can represent self-contact.

    A local polygonal sweep overlaps by construction.  A genuine smooth-curve
    self-contact is a doubly-critical pair: the connecting chord is normal to
    both local tangents.  Vertex-clamped closest points are deliberately left
    to their adjoining capsules, which avoids classifying polygon corners as
    contacts while retaining parallel-arm/hairpin contacts.
    """

    endpoint_tolerance = 1e-7
    u = (
        tangent_start[i]
        if s <= endpoint_tolerance
        else tangent_end[i]
        if s >= 1.0 - endpoint_tolerance
        else capsules.ends[i] - capsules.starts[i]
    )
    v = (
        tangent_start[j]
        if t <= endpoint_tolerance
        else tangent_end[j]
        if t >= 1.0 - endpoint_tolerance
        else capsules.ends[j] - capsules.starts[j]
    )
    pa = capsules.starts[i] + s * u
    pb = capsules.starts[j] + t * v
    chord = pa - pb
    chord_norm = float(np.linalg.norm(chord))
    if chord_norm <= np.finfo(np.float64).eps:
        return True
    angular_tolerance = 5e-3
    return (
        abs(float(np.dot(chord, u)))
        <= angular_tolerance * chord_norm * float(np.linalg.norm(u))
        and abs(float(np.dot(chord, v)))
        <= angular_tolerance * chord_norm * float(np.linalg.norm(v))
    )


def find_capsule_conflicts(
    capsules: CapsuleArrays,
    adjacency: np.ndarray | None = None,
    *,
    shared_node_positions: dict[tuple[int, int], np.ndarray] | None = None,
    shared_node_radii: dict[tuple[int, int], float] | None = None,
    clearance_fraction: float = 0.0,
    maximum_records: int | None = None,
) -> list[CapsuleConflict]:
    """Find nonadjacent or nonlocal self-overlaps with sweep-and-prune.

    Directly adjoining capsules are exempt. Graph-adjacent segments are exempt
    only inside their shared-node junction region when shared-node metadata is
    supplied; the legacy whole-branch exemption is retained only for callers
    that provide an adjacency matrix without that metadata. All thresholds are
    fractions of local radius.
    """

    if clearance_fraction < 0.0:
        raise ValueError("clearance_fraction must be non-negative")
    count = capsules.n
    radii = np.asarray(capsules.max_radii, dtype=np.float64)
    lo = np.minimum(capsules.starts, capsules.ends) - radii[:, None]
    hi = np.maximum(capsules.starts, capsules.ends) + radii[:, None]
    order = np.argsort(lo[:, 0], kind="mergesort")
    conflicts: list[CapsuleConflict] = []
    tangent_start = capsules.ends - capsules.starts
    tangent_end = tangent_start.copy()
    # Estimate the smooth tangent at polygon vertices.  Using the adjoining
    # chords is essential because segment/segment closest points commonly
    # clamp to a polyline vertex even for a true hairpin self-contact.
    for segment in np.unique(capsules.seg_idx):
        indices = np.where(capsules.seg_idx == segment)[0]
        indices = indices[np.argsort(capsules.arc_start[indices], kind="mergesort")]
        for left, right in zip(indices[:-1], indices[1:]):
            a = tangent_end[left]
            b = tangent_start[right]
            an = a / max(float(np.linalg.norm(a)), 1e-30)
            bn = b / max(float(np.linalg.norm(b)), 1e-30)
            averaged = an + bn
            if np.linalg.norm(averaged) > 1e-12:
                tangent_end[left] = averaged
                tangent_start[right] = averaged
    segment_ends: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    if shared_node_positions is not None and shared_node_radii is not None:
        for segment in np.unique(capsules.seg_idx):
            segment = int(segment)
            indices = np.where(capsules.seg_idx == segment)[0]
            first = indices[np.argmin(capsules.arc_start[indices])]
            last = indices[np.argmax(capsules.arc_end[indices])]
            segment_ends[segment] = (capsules.starts[first], capsules.ends[last])

    for order_i, i_raw in enumerate(order):
        i = int(i_raw)
        for j_raw in order[order_i + 1 :]:
            j = int(j_raw)
            if lo[j, 0] > hi[i, 0]:
                break
            if np.any(lo[j, 1:] > hi[i, 1:]) or np.any(lo[i, 1:] > hi[j, 1:]):
                continue
            segment_i = int(capsules.seg_idx[i])
            segment_j = int(capsules.seg_idx[j])
            if segment_i == segment_j and _share_endpoint(capsules, i, j):
                continue
            s, t, distance = closest_segment_parameters(
                capsules.starts[i], capsules.ends[i], capsules.starts[j], capsules.ends[j]
            )
            radius_i = float(
                capsules.radii_start[i]
                + s * (capsules.radii_end[i] - capsules.radii_start[i])
            )
            radius_j = float(
                capsules.radii_start[j]
                + t * (capsules.radii_end[j] - capsules.radii_start[j])
            )
            if segment_i == segment_j:
                # Nearby capsules on one continuous sweep necessarily overlap;
                # that overlap constructs the tube and is not a self-contact.
                # Only compare pairs separated along intrinsic arc length by
                # more than the two local radii. Hairpin/doubly-critical pairs
                # remain eligible while straight densely sampled runs do not
                # produce thousands of false conflicts.
                arc_i = float(
                    capsules.arc_start[i]
                    + s * (capsules.arc_end[i] - capsules.arc_start[i])
                )
                arc_j = float(
                    capsules.arc_start[j]
                    + t * (capsules.arc_end[j] - capsules.arc_start[j])
                )
                if abs(arc_i - arc_j) <= radius_i + radius_j:
                    continue
                if not _is_doubly_critical_pair(
                    capsules, i, j, s, t, tangent_start, tangent_end
                ):
                    continue
            if (
                segment_i != segment_j
                and adjacency is not None
                and 0 <= segment_i < adjacency.shape[0]
                and 0 <= segment_j < adjacency.shape[1]
                and bool(adjacency[segment_i, segment_j])
            ):
                key = (segment_i, segment_j)
                if shared_node_positions is None or shared_node_radii is None:
                    continue
                node_position = shared_node_positions.get(key)
                node_radius = shared_node_radii.get(key)
                if node_position is not None and node_radius is not None:
                    arc_i = float(capsules.arc_start[i] + s * (
                        capsules.arc_end[i] - capsules.arc_start[i]))
                    arc_j = float(capsules.arc_start[j] + t * (
                        capsules.arc_end[j] - capsules.arc_start[j]))
                    # Identify which intrinsic endpoint is the shared node from
                    # the endpoint geometry, then measure along each segment.
                    start_i, end_i = segment_ends[segment_i]
                    start_j, end_j = segment_ends[segment_j]
                    dist_i = arc_i if np.linalg.norm(start_i-node_position) <= np.linalg.norm(end_i-node_position) else float(capsules.seg_L[segment_i])-arc_i
                    dist_j = arc_j if np.linalg.norm(start_j-node_position) <= np.linalg.norm(end_j-node_position) else float(capsules.seg_L[segment_j])-arc_j
                    if (
                        dist_i <= float(node_radius) + radius_i
                        and dist_j <= float(node_radius) + radius_j
                    ):
                        continue
            clearance = distance - radius_i - radius_j
            required = clearance_fraction * min(radius_i, radius_j)
            numeric_tolerance = 256.0 * np.finfo(np.float64).eps * max(
                distance, radius_i, radius_j, abs(required), 1.0
            )
            if clearance >= required - numeric_tolerance:
                continue
            conflicts.append(
                CapsuleConflict(
                    capsule_a=i,
                    capsule_b=j,
                    segment_a=segment_i,
                    segment_b=segment_j,
                    parameter_a=s,
                    parameter_b=t,
                    centreline_distance=distance,
                    radius_a=radius_i,
                    radius_b=radius_j,
                    clearance=clearance,
                )
            )
            if maximum_records is not None and len(conflicts) >= maximum_records:
                return sorted(conflicts, key=lambda item: item.normalized_clearance)
    return sorted(conflicts, key=lambda item: item.normalized_clearance)


__all__ = [
    "CapsuleConflict",
    "closest_segment_parameters",
    "find_capsule_conflicts",
]
