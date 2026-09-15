"""Scale-equivariant, graph-aware implicit field for vascular trees.

This module deliberately does not rasterize the field.  It provides an
on-demand oracle which can be consumed by an adaptive sampler or an external
implicit mesher (for example CGAL Mesh_3).

The field has three important invariants:

* radii are read-only inputs;
* the hard union is evaluated with a correctness-bounded BVH, never a fixed
  number of nearest capsule midpoints;
* all smoothing lengths are multiples of a junction radius, so scaling a
  graph by ``lambda`` scales the field by ``lambda``.

Two interchangeable primitives are provided: the historical sign-correct
radial taper and the exact Euclidean signed distance to a round cone (the
union of linearly interpolated balls). Both retain correctness-bounded BVH
queries and scale-equivariant junction blending.
"""

from __future__ import annotations

from dataclasses import dataclass
import heapq
import math
from typing import Any, Iterable

import numpy as np

try:
    from numba import njit, prange
except ImportError:  # pragma: no cover - optional acceleration
    njit = None
    prange = range

from .capsules import CapsuleArrays


if njit is not None:
    @njit(cache=True, parallel=True)
    def _evaluate_bvh_batch_numba(
        points: np.ndarray,
        node_lo: np.ndarray,
        node_hi: np.ndarray,
        node_max_radius: np.ndarray,
        node_left: np.ndarray,
        node_right: np.ndarray,
        leaf_offset: np.ndarray,
        leaf_count: np.ndarray,
        leaf_primitives: np.ndarray,
        starts: np.ndarray,
        ends: np.ndarray,
        radii_start: np.ndarray,
        radii_end: np.ndarray,
        segment_ids: np.ndarray,
        clip_plane_normals: np.ndarray,
        clip_plane_offsets: np.ndarray,
        clip_plane_count: np.ndarray,
        primitive_method_code: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        count = len(points)
        values = np.empty(count, dtype=np.float64)
        owners = np.empty(count, dtype=np.int64)
        local_radii = np.empty(count, dtype=np.float64)
        for point_index in prange(count):
            point = points[point_index]
            best = np.inf
            best_primitive = -1
            best_radius = 0.0
            stack = np.empty(128, dtype=np.int64)
            stack_size = 1
            stack[0] = 0
            while stack_size:
                stack_size -= 1
                node = stack[stack_size]
                distance2 = 0.0
                for axis in range(3):
                    delta = 0.0
                    if point[axis] < node_lo[node, axis]:
                        delta = node_lo[node, axis] - point[axis]
                    elif point[axis] > node_hi[node, axis]:
                        delta = point[axis] - node_hi[node, axis]
                    distance2 += delta * delta
                lower_bound = math.sqrt(distance2) - node_max_radius[node]
                if lower_bound > best:
                    continue
                n_leaf = leaf_count[node]
                if n_leaf > 0:
                    offset = leaf_offset[node]
                    for local_index in range(n_leaf):
                        primitive = leaf_primitives[offset + local_index]
                        dx0 = ends[primitive, 0] - starts[primitive, 0]
                        dx1 = ends[primitive, 1] - starts[primitive, 1]
                        dx2 = ends[primitive, 2] - starts[primitive, 2]
                        px0 = point[0] - starts[primitive, 0]
                        px1 = point[1] - starts[primitive, 1]
                        px2 = point[2] - starts[primitive, 2]
                        length2 = dx0 * dx0 + dx1 * dx1 + dx2 * dx2
                        t = (px0 * dx0 + px1 * dx1 + px2 * dx2) / max(length2, 1e-30)
                        t = min(max(t, 0.0), 1.0)
                        rx0 = px0 - t * dx0
                        rx1 = px1 - t * dx1
                        rx2 = px2 - t * dx2
                        radius = radii_start[primitive] + t * (
                            radii_end[primitive] - radii_start[primitive]
                        )
                        if primitive_method_code == 0:
                            value = math.sqrt(rx0 * rx0 + rx1 * rx1 + rx2 * rx2) - radius
                        else:
                            r0 = radii_start[primitive]
                            r1 = radii_end[primitive]
                            rr = r0 - r1
                            a2 = length2 - rr * rr
                            if length2 <= 1e-30 or a2 <= 1e-30:
                                if r0 >= r1:
                                    value = math.sqrt(px0 * px0 + px1 * px1 + px2 * px2) - r0
                                    radius = r0
                                    t = 0.0
                                else:
                                    qx0 = point[0] - ends[primitive, 0]
                                    qx1 = point[1] - ends[primitive, 1]
                                    qx2 = point[2] - ends[primitive, 2]
                                    value = math.sqrt(qx0 * qx0 + qx1 * qx1 + qx2 * qx2) - r1
                                    radius = r1
                                    t = 1.0
                            else:
                                y = px0 * dx0 + px1 * dx1 + px2 * dx2
                                z = y - length2
                                cx0 = px0 * length2 - dx0 * y
                                cx1 = px1 * length2 - dx1 * y
                                cx2 = px2 * length2 - dx2 * y
                                x2 = cx0 * cx0 + cx1 * cx1 + cx2 * cx2
                                y2 = y * y * length2
                                z2 = z * z * length2
                                sign_rr = 1.0 if rr >= 0.0 else -1.0
                                sign_z = 1.0 if z >= 0.0 else -1.0
                                sign_y = 1.0 if y >= 0.0 else -1.0
                                k = sign_rr * rr * rr * x2
                                inv_l2 = 1.0 / length2
                                if sign_z * a2 * z2 > k:
                                    value = math.sqrt(x2 + z2) * inv_l2 - r1
                                    radius = r1
                                    t = 1.0
                                elif sign_y * a2 * y2 < k:
                                    value = math.sqrt(x2 + y2) * inv_l2 - r0
                                    radius = r0
                                    t = 0.0
                                else:
                                    value = (
                                        math.sqrt(max(x2 * a2 * inv_l2, 0.0))
                                        + y * rr
                                    ) * inv_l2 - r0
                        for plane_index in range(clip_plane_count[primitive]):
                            plane_value = (
                                point[0] * clip_plane_normals[primitive, plane_index, 0]
                                + point[1] * clip_plane_normals[primitive, plane_index, 1]
                                + point[2] * clip_plane_normals[primitive, plane_index, 2]
                                - clip_plane_offsets[primitive, plane_index]
                            )
                            value = max(value, plane_value)
                        if value < best:
                            best = value
                            best_primitive = primitive
                            best_radius = radius
                else:
                    left = node_left[node]
                    right = node_right[node]
                    if left >= 0:
                        stack[stack_size] = left
                        stack_size += 1
                    if right >= 0:
                        stack[stack_size] = right
                        stack_size += 1
            values[point_index] = best
            owners[point_index] = segment_ids[best_primitive]
            local_radii[point_index] = best_radius
        return values, owners, local_radii
else:
    _evaluate_bvh_batch_numba = None


@dataclass(frozen=True)
class FieldSample:
    """One implicit-field query result."""

    value: float
    owner_segment: int
    local_radius: float
    gradient: np.ndarray | None = None


@dataclass(frozen=True)
class JunctionBlend:
    """Dimensionless smooth-union definition for one graph junction.

    ``capsule_groups[i]`` contains the local capsule indices belonging to
    ``incident_segments[i]``.  Restricting these groups to the neighbourhood
    of the shared node prevents adjacent branches that later run in parallel
    from blending along their full length.
    """

    node_id: int
    position: np.ndarray
    radius: float
    incident_segments: tuple[int, ...]
    capsule_groups: tuple[np.ndarray, ...]
    blend_fraction: float
    support_factor: float

    @property
    def support_radius(self) -> float:
        return self.support_factor * self.radius

    @property
    def blend_depth(self) -> float:
        return self.blend_fraction * self.radius


@dataclass
class _BVHNode:
    lo: np.ndarray
    hi: np.ndarray
    max_radius: float
    indices: np.ndarray | None = None
    left: int = -1
    right: int = -1


def _distance_to_aabb(point: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> float:
    delta = np.maximum(np.maximum(lo - point, point - hi), 0.0)
    return float(np.linalg.norm(delta))


class CapsuleBVH:
    """A correctness-bounded BVH over capsule centreline AABBs.

    For every primitive in a node,

    ``distance(point, node_aabb) - node.max_radius``

    is a lower bound on the tapered-capsule field.  Branch-and-bound queries
    therefore cannot omit a primitive capable of changing the minimum.
    """

    def __init__(
        self,
        starts: np.ndarray,
        ends: np.ndarray,
        max_radii: np.ndarray,
        leaf_size: int = 8,
    ) -> None:
        self.starts = np.asarray(starts, dtype=np.float64)
        self.ends = np.asarray(ends, dtype=np.float64)
        self.max_radii = np.asarray(max_radii, dtype=np.float64)
        if self.starts.shape != self.ends.shape or self.starts.ndim != 2:
            raise ValueError("starts and ends must both have shape (N, 3)")
        if self.starts.shape[1] != 3 or len(self.max_radii) != len(self.starts):
            raise ValueError("invalid capsule array shapes")
        if len(self.starts) == 0:
            raise ValueError("at least one capsule is required")
        if not np.isfinite(self.starts).all() or not np.isfinite(self.ends).all():
            raise ValueError("capsule coordinates must be finite")
        if not np.isfinite(self.max_radii).all() or np.any(self.max_radii < 0):
            raise ValueError("capsule radii must be finite and non-negative")

        self.leaf_size = max(int(leaf_size), 1)
        self.primitive_lo = np.minimum(self.starts, self.ends)
        self.primitive_hi = np.maximum(self.starts, self.ends)
        self.centroids = 0.5 * (self.primitive_lo + self.primitive_hi)
        self.nodes: list[_BVHNode] = []
        self.root = self._build(np.arange(len(self.starts), dtype=np.int64))

    def _build(self, indices: np.ndarray) -> int:
        lo = self.primitive_lo[indices].min(axis=0)
        hi = self.primitive_hi[indices].max(axis=0)
        max_radius = float(self.max_radii[indices].max())
        node_id = len(self.nodes)
        self.nodes.append(_BVHNode(lo=lo, hi=hi, max_radius=max_radius))
        if len(indices) <= self.leaf_size:
            self.nodes[node_id].indices = indices.copy()
            return node_id

        span = np.ptp(self.centroids[indices], axis=0)
        axis = int(np.argmax(span))
        order = indices[np.argsort(self.centroids[indices, axis], kind="mergesort")]
        middle = len(order) // 2
        self.nodes[node_id].left = self._build(order[:middle])
        self.nodes[node_id].right = self._build(order[middle:])
        return node_id

    def lower_bound(self, node_id: int, point: np.ndarray) -> float:
        node = self.nodes[node_id]
        return _distance_to_aabb(point, node.lo, node.hi) - node.max_radius

    def ordered_leaves(self, point: np.ndarray) -> Iterable[np.ndarray]:
        """Yield leaf primitive arrays in nondecreasing lower-bound order."""

        point = np.asarray(point, dtype=np.float64)
        queue: list[tuple[float, int]] = [(self.lower_bound(self.root, point), self.root)]
        while queue:
            _bound, node_id = heapq.heappop(queue)
            node = self.nodes[node_id]
            if node.indices is not None:
                yield node.indices
                continue
            for child in (node.left, node.right):
                heapq.heappush(queue, (self.lower_bound(child, point), child))

    def candidates_below(self, point: np.ndarray, threshold: float) -> np.ndarray:
        """Return every primitive whose BVH lower bound is at most threshold."""

        point = np.asarray(point, dtype=np.float64)
        found: list[np.ndarray] = []
        stack = [self.root]
        while stack:
            node_id = stack.pop()
            if self.lower_bound(node_id, point) > threshold:
                continue
            node = self.nodes[node_id]
            if node.indices is not None:
                found.append(node.indices)
            else:
                stack.append(node.left)
                stack.append(node.right)
        if not found:
            return np.empty(0, dtype=np.int64)
        return np.concatenate(found)


def tapered_capsule_values(
    point: np.ndarray,
    starts: np.ndarray,
    ends: np.ndarray,
    radii_start: np.ndarray,
    radii_end: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate tapered capsules and return ``(value, radius, t)`` arrays."""

    point = np.asarray(point, dtype=np.float64)
    starts = np.asarray(starts, dtype=np.float64)
    ends = np.asarray(ends, dtype=np.float64)
    radii_start = np.asarray(radii_start, dtype=np.float64)
    radii_end = np.asarray(radii_end, dtype=np.float64)
    direction = ends - starts
    length2 = np.einsum("ij,ij->i", direction, direction)
    offset = point[None, :] - starts
    t_raw = np.einsum("ij,ij->i", offset, direction) / np.maximum(length2, 1e-30)
    t = np.clip(t_raw, 0.0, 1.0)
    closest = starts + t[:, None] * direction
    radial_distance = np.linalg.norm(point[None, :] - closest, axis=1)
    radius = radii_start + t * (radii_end - radii_start)
    return radial_distance - radius, radius, t


def round_cone_values(
    point: np.ndarray,
    starts: np.ndarray,
    ends: np.ndarray,
    radii_start: np.ndarray,
    radii_end: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Exact signed distance to linearly tapered round-cone primitives.

    A primitive is the union of balls whose centre and radius interpolate
    linearly between its endpoints. When one endpoint ball contains the other
    (``abs(r1-r0) >= segment_length``), the primitive reduces exactly to the
    larger ball.
    """

    point = np.asarray(point, dtype=np.float64)
    starts = np.asarray(starts, dtype=np.float64)
    ends = np.asarray(ends, dtype=np.float64)
    r0 = np.asarray(radii_start, dtype=np.float64)
    r1 = np.asarray(radii_end, dtype=np.float64)
    ba = ends - starts
    pa = point[None, :] - starts
    l2 = np.einsum("ij,ij->i", ba, ba)
    y = np.einsum("ij,ij->i", pa, ba)
    t = np.clip(y / np.maximum(l2, 1e-30), 0.0, 1.0)
    radius = r0 + t * (r1 - r0)
    values = np.empty(len(starts), dtype=np.float64)

    rr = r0 - r1
    a2 = l2 - rr * rr
    contained = (l2 <= 1e-30) | (a2 <= 1e-30)
    if np.any(contained):
        use_start = r0[contained] >= r1[contained]
        centres = np.where(use_start[:, None], starts[contained], ends[contained])
        selected_radii = np.where(use_start, r0[contained], r1[contained])
        values[contained] = np.linalg.norm(point[None, :] - centres, axis=1) - selected_radii
        radius[contained] = selected_radii
        t[contained] = np.where(use_start, 0.0, 1.0)

    regular = ~contained
    if np.any(regular):
        ba_r = ba[regular]
        pa_r = pa[regular]
        l2_r = l2[regular]
        y_r = y[regular]
        rr_r = rr[regular]
        a2_r = a2[regular]
        z = y_r - l2_r
        cross_scaled = pa_r * l2_r[:, None] - ba_r * y_r[:, None]
        x2 = np.einsum("ij,ij->i", cross_scaled, cross_scaled)
        y2 = y_r * y_r * l2_r
        z2 = z * z * l2_r
        k = np.where(rr_r >= 0.0, 1.0, -1.0) * rr_r * rr_r * x2
        inv_l2 = 1.0 / l2_r
        end_region = np.where(z >= 0.0, 1.0, -1.0) * a2_r * z2 > k
        start_region = np.where(y_r >= 0.0, 1.0, -1.0) * a2_r * y2 < k
        result = (
            np.sqrt(np.maximum(x2 * a2_r * inv_l2, 0.0)) + y_r * rr_r
        ) * inv_l2 - r0[regular]
        result = np.where(
            start_region,
            np.sqrt(x2 + y2) * inv_l2 - r0[regular],
            result,
        )
        result = np.where(
            end_region,
            np.sqrt(x2 + z2) * inv_l2 - r1[regular],
            result,
        )
        values[regular] = result
        regular_indices = np.flatnonzero(regular)
        start_indices = regular_indices[start_region]
        end_indices = regular_indices[end_region]
        radius[start_indices] = r0[start_indices]
        t[start_indices] = 0.0
        radius[end_indices] = r1[end_indices]
        t[end_indices] = 1.0
    return values, radius, t


class GraphImplicitField:
    """Analytic graph field with safe spatial acceleration and local blends."""

    def __init__(
        self,
        capsules: CapsuleArrays,
        junctions: Iterable[JunctionBlend] = (),
        *,
        bvh_leaf_size: int = 8,
        primitive_method: str = "radial",
        clip_bifurcation_caps: bool = True,
    ) -> None:
        self.starts = np.asarray(capsules.starts, dtype=np.float64)
        self.ends = np.asarray(capsules.ends, dtype=np.float64)
        self.radii_start = np.asarray(capsules.radii_start, dtype=np.float64)
        self.radii_end = np.asarray(capsules.radii_end, dtype=np.float64)
        self.segment_ids = np.asarray(capsules.seg_idx, dtype=np.int64)
        self.cap_bif_at_start = np.asarray(
            getattr(capsules, "cap_bif_at_start", np.zeros(len(self.starts), dtype=bool)),
            dtype=bool,
        )
        self.cap_bif_at_end = np.asarray(
            getattr(capsules, "cap_bif_at_end", np.zeros(len(self.starts), dtype=bool)),
            dtype=bool,
        )
        self.clip_bifurcation_caps = bool(clip_bifurcation_caps)
        self.max_radii = np.maximum(self.radii_start, self.radii_end)
        if primitive_method not in {"radial", "round_cone"}:
            raise ValueError("primitive_method must be 'radial' or 'round_cone'")
        self.primitive_method = primitive_method
        if np.any(self.radii_start < 0) or np.any(self.radii_end < 0):
            raise ValueError("all radii must be non-negative")
        positive = np.concatenate((self.radii_start, self.radii_end))
        positive = positive[positive > 0.0]
        if not len(positive):
            raise ValueError("at least one capsule radius must be positive")
        self.minimum_positive_radius = float(np.min(positive))
        if not (
            len(self.starts)
            == len(self.ends)
            == len(self.radii_start)
            == len(self.radii_end)
            == len(self.segment_ids)
            == len(self.cap_bif_at_start)
            == len(self.cap_bif_at_end)
        ):
            raise ValueError("capsule arrays have inconsistent lengths")

        self.bvh = CapsuleBVH(
            self.starts, self.ends, self.max_radii, leaf_size=bvh_leaf_size
        )
        self.junctions = tuple(junctions)
        (
            self.clip_plane_normals,
            self.clip_plane_offsets,
            self.clip_plane_count,
            self.junction_core_fractions,
        ) = self._build_clip_planes()
        self._junction_core_fraction_by_node = {
            junction.node_id: float(self.junction_core_fractions[index])
            for index, junction in enumerate(self.junctions)
        }
        self._prepare_compiled_bvh()
        self.segment_capsules: dict[int, np.ndarray] = {
            int(segment_id): np.flatnonzero(self.segment_ids == segment_id)
            for segment_id in np.unique(self.segment_ids)
        }
        self.bounds_min = np.min(self.starts - self.max_radii[:, None], axis=0)
        self.bounds_min = np.minimum(
            self.bounds_min,
            np.min(self.ends - self.max_radii[:, None], axis=0),
        )
        self.bounds_max = np.max(self.starts + self.max_radii[:, None], axis=0)
        self.bounds_max = np.maximum(
            self.bounds_max,
            np.max(self.ends + self.max_radii[:, None], axis=0),
        )

        length = np.linalg.norm(self.ends - self.starts, axis=1)
        slope = np.abs(self.radii_end - self.radii_start) / np.maximum(length, 1e-30)
        self.primitive_lipschitz = (
            np.ones_like(slope)
            if self.primitive_method == "round_cone"
            else np.sqrt(1.0 + slope * slope)
        )
        base_lipschitz = float(np.max(self.primitive_lipschitz))
        blend_extra = 0.0
        for junction in self.junctions:
            # The C1 support gate changes over the outer quarter of the support
            # ball. max|smoothstep'|=1.5, hence |grad(w)|<=6/support.
            blend_extra = max(
                blend_extra,
                6.0 * junction.blend_fraction / junction.support_factor,
            )
        self.lipschitz_bound = base_lipschitz + blend_extra

    def _build_clip_planes(
        self,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Build local graph-node half-spaces for junction-near primitives.

        A junction plane must never constrain an entire incident segment.  A
        tortuous branch can legitimately cross its endpoint tangent plane
        again far downstream; applying that half-space globally deletes the
        returning part of the vessel.  ``JunctionBlend.capsule_groups`` is an
        intrinsic, radius-scaled halo around the graph node, so only capsules
        whose rounded ends can influence the junction receive the plane.
        """

        count = len(self.starts)
        normals = np.zeros((count, 2, 3), dtype=np.float64)
        offsets = np.zeros((count, 2), dtype=np.float64)
        plane_count = np.zeros(count, dtype=np.int64)
        core_fractions = np.zeros(len(self.junctions), dtype=np.float64)
        if not self.clip_bifurcation_caps:
            return normals, offsets, plane_count, core_fractions

        covered_segments: set[int] = set()
        for junction_index, junction in enumerate(self.junctions):
            junction_clipped = False
            for segment_id, local_indices in zip(
                junction.incident_segments, junction.capsule_groups
            ):
                indices = np.asarray(local_indices, dtype=np.int64)
                if not len(indices):
                    continue
                start_distance = np.linalg.norm(
                    self.starts[indices] - junction.position[None, :], axis=1
                )
                end_distance = np.linalg.norm(
                    self.ends[indices] - junction.position[None, :], axis=1
                )
                start_local = int(np.argmin(start_distance))
                end_local = int(np.argmin(end_distance))
                if start_distance[start_local] <= end_distance[end_local]:
                    endpoint_primitive = int(indices[start_local])
                    if not self.cap_bif_at_start[endpoint_primitive]:
                        continue
                    direction = (
                        self.ends[endpoint_primitive] - self.starts[endpoint_primitive]
                    )
                    normal = -direction
                else:
                    endpoint_primitive = int(indices[end_local])
                    if not self.cap_bif_at_end[endpoint_primitive]:
                        continue
                    direction = (
                        self.ends[endpoint_primitive] - self.starts[endpoint_primitive]
                    )
                    normal = direction
                norm = float(np.linalg.norm(normal))
                if norm <= 1e-30:
                    continue
                normal = normal / norm
                offset = float(np.dot(junction.position, normal))
                for primitive in indices:
                    slot = int(plane_count[primitive])
                    if slot >= 2:
                        raise ValueError(
                            f"segment {segment_id} has more than two junction clip planes"
                        )
                    normals[primitive, slot] = normal
                    offsets[primitive, slot] = offset
                    plane_count[primitive] += 1
                covered_segments.add(int(segment_id))
                junction_clipped = True
            if junction_clipped:
                # A half-radius core is resolved by three cells at the minimum
                # supported six-cells-per-diameter production resolution.
                core_fractions[junction_index] = 0.5

        # Preserve direct GraphImplicitField use when no JunctionBlend metadata
        # was supplied: clip only the explicitly tagged endpoint primitive.
        for primitive in range(count):
            if int(self.segment_ids[primitive]) in covered_segments:
                continue
            direction = self.ends[primitive] - self.starts[primitive]
            norm = float(np.linalg.norm(direction))
            if norm <= 1e-30:
                continue
            unit = direction / norm
            for active, point, normal in (
                (self.cap_bif_at_start[primitive], self.starts[primitive], -unit),
                (self.cap_bif_at_end[primitive], self.ends[primitive], unit),
            ):
                if not active:
                    continue
                slot = int(plane_count[primitive])
                normals[primitive, slot] = normal
                offsets[primitive, slot] = float(np.dot(point, normal))
                plane_count[primitive] += 1
        return normals, offsets, plane_count, core_fractions

    def _prepare_compiled_bvh(self) -> None:
        node_count = len(self.bvh.nodes)
        self._node_lo = np.asarray([node.lo for node in self.bvh.nodes])
        self._node_hi = np.asarray([node.hi for node in self.bvh.nodes])
        self._node_max_radius = np.asarray(
            [node.max_radius for node in self.bvh.nodes], dtype=np.float64
        )
        self._node_left = np.asarray(
            [node.left for node in self.bvh.nodes], dtype=np.int64
        )
        self._node_right = np.asarray(
            [node.right for node in self.bvh.nodes], dtype=np.int64
        )
        self._leaf_offset = np.full(node_count, -1, dtype=np.int64)
        self._leaf_count = np.zeros(node_count, dtype=np.int64)
        flattened: list[int] = []
        for node_id, node in enumerate(self.bvh.nodes):
            if node.indices is None:
                continue
            self._leaf_offset[node_id] = len(flattened)
            self._leaf_count[node_id] = len(node.indices)
            flattened.extend(int(value) for value in node.indices)
        self._leaf_primitives = np.asarray(flattened, dtype=np.int64)

    def _values_for_indices(
        self, point: np.ndarray, indices: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        evaluator = (
            round_cone_values
            if self.primitive_method == "round_cone"
            else tapered_capsule_values
        )
        values, radii, projected = evaluator(
            point,
            self.starts[indices],
            self.ends[indices],
            self.radii_start[indices],
            self.radii_end[indices],
        )
        for plane_index in range(2):
            active = self.clip_plane_count[indices] > plane_index
            if np.any(active):
                selected = indices[active]
                plane_values = (
                    self.clip_plane_normals[selected, plane_index] @ point
                    - self.clip_plane_offsets[selected, plane_index]
                )
                values[active] = np.maximum(values[active], plane_values)
        return values, radii, projected

    def _hard_min(self, point: np.ndarray) -> tuple[float, int, float, int, float]:
        """Return value, primitive, radius, segment and projected t."""

        point = np.asarray(point, dtype=np.float64)
        best_value = math.inf
        best_index = -1
        best_radius = math.nan
        best_t = math.nan
        queue: list[tuple[float, int]] = [
            (self.bvh.lower_bound(self.bvh.root, point), self.bvh.root)
        ]
        while queue:
            bound, node_id = heapq.heappop(queue)
            if bound > best_value:
                break
            node = self.bvh.nodes[node_id]
            if node.indices is not None:
                values, radii, ts = self._values_for_indices(point, node.indices)
                local = int(np.argmin(values))
                value = float(values[local])
                if value < best_value:
                    best_value = value
                    best_index = int(node.indices[local])
                    best_radius = float(radii[local])
                    best_t = float(ts[local])
                continue
            for child in (node.left, node.right):
                child_bound = self.bvh.lower_bound(child, point)
                if child_bound <= best_value:
                    heapq.heappush(queue, (child_bound, child))
        return (
            best_value,
            best_index,
            best_radius,
            int(self.segment_ids[best_index]),
            best_t,
        )

    def _group_min(self, point: np.ndarray, indices: np.ndarray) -> float:
        values, _radii, _ts = self._values_for_indices(point, indices)
        return float(np.min(values))

    @staticmethod
    def _support_weight(distance: float, support: float) -> float:
        """C1 radial gate: one through 75% support, then smooth to zero."""

        inner = 0.75 * support
        if distance <= inner:
            return 1.0
        if distance >= support:
            return 0.0
        t = (distance - inner) / (support - inner)
        return 1.0 - t * t * (3.0 - 2.0 * t)

    def _junction_candidate(
        self, point: np.ndarray, junction: JunctionBlend
    ) -> tuple[float, float] | None:
        distance = float(np.linalg.norm(point - junction.position))
        weight = self._support_weight(distance, junction.support_radius)
        if weight <= 0.0:
            return None
        branch_values = np.asarray(
            [self._group_min(point, group) for group in junction.capsule_groups],
            dtype=np.float64,
        )
        n = len(branch_values)
        if n < 2:
            return None
        hard = float(np.min(branch_values))
        depth = junction.blend_depth
        candidate = hard
        if depth > 0.0:
            tau = depth / math.log(float(n))
            shifted = branch_values - hard
            soft = hard - tau * math.log(float(np.exp(-shifted / tau).sum()))
            candidate = hard + weight * (soft - hard)
        core_fraction = self._junction_core_fraction_by_node.get(junction.node_id, 0.0)
        if core_fraction > 0.0:
            candidate = min(candidate, distance - core_fraction * junction.radius)
        return candidate, junction.radius

    def _gradient(
        self, point: np.ndarray, primitive_index: int, projected_t: float
    ) -> np.ndarray:
        start = self.starts[primitive_index]
        end = self.ends[primitive_index]
        direction = end - start
        length = float(np.linalg.norm(direction))
        if length <= 1e-30:
            radial = point - start
            norm = float(np.linalg.norm(radial))
            return radial / norm if norm > 1e-30 else np.zeros(3)
        axis = direction / length
        closest = start + projected_t * direction
        radial = point - closest
        radial_norm = float(np.linalg.norm(radial))
        radial_unit = radial / radial_norm if radial_norm > 1e-30 else np.zeros(3)
        if 0.0 < projected_t < 1.0:
            radial_unit = radial_unit - (
                (self.radii_end[primitive_index] - self.radii_start[primitive_index])
                / length
            ) * axis
        norm = float(np.linalg.norm(radial_unit))
        return radial_unit / norm if norm > 1e-30 else np.zeros(3)

    def sample(self, point: np.ndarray, *, with_gradient: bool = False) -> FieldSample:
        point = np.asarray(point, dtype=np.float64)
        if point.shape != (3,) or not np.isfinite(point).all():
            raise ValueError("point must be a finite vector with shape (3,)")
        value, primitive, radius, segment, projected_t = self._hard_min(point)
        junction_owned = False
        for junction in self.junctions:
            candidate = self._junction_candidate(point, junction)
            if candidate is None:
                continue
            candidate_value, candidate_radius = candidate
            if candidate_value < value:
                value = candidate_value
                radius = candidate_radius
                junction_owned = True
                # A junction is owned by its graph node, not an arbitrary
                # capsule. Keep a stable negative encoding for diagnostics.
                segment = -junction.node_id - 1
        if radius <= 0.0:
            # A mathematically valid zero-radius tip must not produce a zero
            # Mesh_3 sizing criterion. Use the same positive lower bound as
            # the native oracle while preserving the exact field value.
            radius = self.minimum_positive_radius
        gradient = None
        if with_gradient:
            if junction_owned:
                # Junction gradients include both log-sum-exp and the compact
                # support gate. A relative central difference is used only for
                # requested Hermite samples; ordinary field queries stay fully
                # analytic and do not pay this cost.
                eps = max(radius * 1e-5, np.finfo(np.float64).eps * 128.0)
                gradient = np.empty(3, dtype=np.float64)
                for axis in range(3):
                    offset = np.zeros(3, dtype=np.float64)
                    offset[axis] = eps
                    gradient[axis] = (
                        self.sample(point + offset, with_gradient=False).value
                        - self.sample(point - offset, with_gradient=False).value
                    ) / (2.0 * eps)
                norm = float(np.linalg.norm(gradient))
                if norm > 1e-30:
                    gradient /= norm
            elif (
                self.primitive_method == "radial"
                and not (
                    self.clip_bifurcation_caps
                    and self.clip_plane_count[primitive] > 0
                )
            ):
                gradient = self._gradient(point, primitive, projected_t)
            else:
                eps = max(radius * 1e-5, np.finfo(np.float64).eps * 128.0)
                gradient = np.empty(3, dtype=np.float64)
                for axis in range(3):
                    offset = np.zeros(3, dtype=np.float64)
                    offset[axis] = eps
                    gradient[axis] = (
                        self.sample(point + offset, with_gradient=False).value
                        - self.sample(point - offset, with_gradient=False).value
                    ) / (2.0 * eps)
                norm = float(np.linalg.norm(gradient))
                if norm > 1e-30:
                    gradient /= norm
        return FieldSample(value, segment, radius, gradient)

    def evaluate(
        self, points: np.ndarray, *, with_gradient: bool = False
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
        """Evaluate ``(value, owner, radius, gradient)`` for an ``(N,3)`` array."""

        points = np.asarray(points, dtype=np.float64)
        if points.ndim == 1:
            points = points.reshape(1, 3)
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError("points must have shape (N, 3)")
        if (
            not with_gradient
            and _evaluate_bvh_batch_numba is not None
            and len(points) >= 256
        ):
            values, owners, radii = _evaluate_bvh_batch_numba(
                np.ascontiguousarray(points),
                self._node_lo,
                self._node_hi,
                self._node_max_radius,
                self._node_left,
                self._node_right,
                self._leaf_offset,
                self._leaf_count,
                self._leaf_primitives,
                self.starts,
                self.ends,
                self.radii_start,
                self.radii_end,
                self.segment_ids,
                self.clip_plane_normals,
                self.clip_plane_offsets,
                self.clip_plane_count,
                0 if self.primitive_method == "radial" else 1,
            )
            # Junction work is sparse and low-degree. Keep it in the reference
            # implementation so both paths use exactly the same blend formula.
            for junction in self.junctions:
                distance = np.linalg.norm(points - junction.position[None, :], axis=1)
                for point_index in np.flatnonzero(distance < junction.support_radius):
                    candidate = self._junction_candidate(points[point_index], junction)
                    if candidate is not None and candidate[0] < values[point_index]:
                        values[point_index] = candidate[0]
                        owners[point_index] = -junction.node_id - 1
                        radii[point_index] = candidate[1]
            radii[radii <= 0.0] = self.minimum_positive_radius
            return values, owners, radii, None

        values = np.empty(len(points), dtype=np.float64)
        owners = np.empty(len(points), dtype=np.int64)
        radii = np.empty(len(points), dtype=np.float64)
        gradients = np.empty((len(points), 3), dtype=np.float64) if with_gradient else None
        for i, point in enumerate(points):
            sample = self.sample(point, with_gradient=with_gradient)
            values[i] = sample.value
            owners[i] = sample.owner_segment
            radii[i] = sample.local_radius
            if gradients is not None and sample.gradient is not None:
                gradients[i] = sample.gradient
        return values, owners, radii, gradients

    def minimum_relevant_radius(self, point: np.ndarray, cell_radius: float) -> float:
        """Smallest radius capable of affecting a cell around ``point``."""

        indices = self.bvh.candidates_below(point, self.lipschitz_bound * cell_radius)
        if len(indices):
            local = np.concatenate(
                (self.radii_start[indices], self.radii_end[indices])
            )
            local = local[local > 0.0]
            if len(local):
                return float(np.min(local))
        return self.minimum_positive_radius


def build_junction_blends(
    capsules: CapsuleArrays,
    nodes: dict[int, tuple],
    node_to_segments: dict[int, set[int]],
    *,
    blend_fraction: float = 0.15,
    support_factor: float = 4.0,
) -> tuple[JunctionBlend, ...]:
    """Build graph-local junction definitions without modifying radii.

    Node coordinates are assumed to use the package's input convention
    (micrometres); capsule coordinates are already in millimetres.
    """

    if blend_fraction < 0.0:
        raise ValueError("blend_fraction must be non-negative")
    if support_factor <= 1.0:
        raise ValueError("support_factor must be greater than one")

    starts = np.asarray(capsules.starts, dtype=np.float64)
    ends = np.asarray(capsules.ends, dtype=np.float64)
    segment_ids = np.asarray(capsules.seg_idx, dtype=np.int64)
    half_lengths = 0.5 * np.linalg.norm(ends - starts, axis=1)
    midpoints = 0.5 * (starts + ends)
    junctions: list[JunctionBlend] = []

    for node_id in sorted(node_to_segments):
        incident = tuple(sorted(int(v) for v in node_to_segments[node_id]))
        if len(incident) < 3 or node_id not in nodes:
            continue
        position = np.asarray(nodes[node_id][:3], dtype=np.float64) / 1000.0

        endpoint_radii: list[float] = []
        nearest_by_segment: list[tuple[int, np.ndarray, float, bool]] = []
        for segment_id in incident:
            indices = np.flatnonzero(segment_ids == segment_id)
            if len(indices) == 0:
                continue
            start_dist = np.linalg.norm(starts[indices] - position, axis=1)
            end_dist = np.linalg.norm(ends[indices] - position, axis=1)
            flat_choice = int(np.argmin(np.minimum(start_dist, end_dist)))
            primitive = int(indices[flat_choice])
            if start_dist[flat_choice] <= end_dist[flat_choice]:
                endpoint_radius = float(capsules.radii_start[primitive])
                junction_at_segment_start = True
            else:
                endpoint_radius = float(capsules.radii_end[primitive])
                junction_at_segment_start = False
            endpoint_radii.append(endpoint_radius)
            nearest_by_segment.append(
                (segment_id, indices, endpoint_radius, junction_at_segment_start)
            )

        if len(nearest_by_segment) < 3:
            continue
        junction_radius = float(min(endpoint_radii))
        support = support_factor * junction_radius
        capsule_groups: list[np.ndarray] = []
        kept_segments: list[int] = []
        arc_start = np.asarray(capsules.arc_start, dtype=np.float64)
        arc_end = np.asarray(capsules.arc_end, dtype=np.float64)
        segment_lengths = np.asarray(capsules.seg_L, dtype=np.float64)
        for (
            segment_id,
            indices,
            _endpoint_radius,
            junction_at_segment_start,
        ) in nearest_by_segment:
            if (
                len(arc_start) == len(starts)
                and len(arc_end) == len(starts)
                and 0 <= segment_id < len(segment_lengths)
                and segment_lengths[segment_id] > 0.0
            ):
                if junction_at_segment_start:
                    intrinsic_near = arc_start[indices]
                else:
                    intrinsic_near = np.maximum(
                        segment_lengths[segment_id] - arc_end[indices], 0.0
                    )
                # The support controls the blend neighbourhood.  Adding each
                # primitive's radius includes any round cap capable of leaking
                # into that neighbourhood, while remaining scale invariant.
                local = indices[
                    intrinsic_near
                    <= support + np.maximum(
                        capsules.radii_start[indices], capsules.radii_end[indices]
                    )
                    + np.finfo(np.float64).eps * 64.0
                ]
            else:
                # Compatibility fallback for hand-constructed CapsuleArrays.
                intersects_support = (
                    np.linalg.norm(midpoints[indices] - position, axis=1)
                    <= support + half_lengths[indices]
                )
                local = indices[intersects_support]
            if len(local) == 0:
                nearest = int(
                    np.argmin(np.linalg.norm(midpoints[indices] - position, axis=1))
                )
                local = indices[nearest : nearest + 1]
            capsule_groups.append(np.asarray(local, dtype=np.int64))
            kept_segments.append(segment_id)

        junctions.append(
            JunctionBlend(
                node_id=int(node_id),
                position=position,
                radius=junction_radius,
                incident_segments=tuple(kept_segments),
                capsule_groups=tuple(capsule_groups),
                blend_fraction=float(blend_fraction),
                support_factor=float(support_factor),
            )
        )
    return tuple(junctions)


def build_graph_implicit_field(
    capsules: CapsuleArrays,
    nodes: dict[int, tuple],
    node_to_segments: dict[int, set[int]],
    *,
    blend_fraction: float = 0.15,
    support_factor: float = 4.0,
    bvh_leaf_size: int = 8,
    primitive_method: str = "radial",
    clip_bifurcation_caps: bool = True,
) -> GraphImplicitField:
    junctions = build_junction_blends(
        capsules,
        nodes,
        node_to_segments,
        blend_fraction=blend_fraction,
        support_factor=support_factor,
    )
    return GraphImplicitField(
        capsules,
        junctions,
        bvh_leaf_size=bvh_leaf_size,
        primitive_method=primitive_method,
        clip_bifurcation_caps=clip_bifurcation_caps,
    )


def evaluate_implicit_on_grid(
    field: GraphImplicitField,
    grid: Any,
    narrow_band_indices: np.ndarray,
    *,
    batch_size: int = 65_536,
) -> Any:
    """Compatibility evaluator for parity against the legacy dense pipeline.

    This is not the scale-adaptive production path. It lets the new field be
    compared using the repository's existing extraction/post-processing stack
    while :class:`RadiusAdaptiveOctree` and the native implicit mesher are
    validated independently.
    """

    from .sdf_field import SdfVolume

    indices = np.asarray(narrow_band_indices, dtype=np.int64)
    extent = float(np.linalg.norm(np.asarray(grid.bbox_max) - np.asarray(grid.bbox_min)))
    sentinel = max(extent * 2.0, np.finfo(np.float32).tiny)
    volume = np.full(tuple(int(v) for v in grid.dims), sentinel, dtype=np.float32)
    batch_size = max(int(batch_size), 1)
    for start in range(0, len(indices), batch_size):
        voxels = indices[start : start + batch_size]
        coordinates = np.column_stack(
            [grid.x[voxels[:, 0]], grid.y[voxels[:, 1]], grid.z[voxels[:, 2]]]
        )
        values, _owners, _radii, _gradients = field.evaluate(coordinates)
        volume[voxels[:, 0], voxels[:, 1], voxels[:, 2]] = values.astype(np.float32)
    return SdfVolume(
        sdf=volume,
        path_vol=None,
        blend_weight_vol=None,
        nb_idx=None,
        diag=None,
    )


__all__ = [
    "CapsuleBVH",
    "FieldSample",
    "GraphImplicitField",
    "JunctionBlend",
    "build_graph_implicit_field",
    "build_junction_blends",
    "evaluate_implicit_on_grid",
    "round_cone_values",
    "tapered_capsule_values",
]
