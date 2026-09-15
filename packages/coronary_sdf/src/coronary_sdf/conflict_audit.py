"""Mask-backed adjudication of fixed-radius capsule conflicts."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from scipy import ndimage

from .amira_lattice import AmiraByteLattice
from .capsules import CapsuleArrays
from .geometry_constraints import CapsuleConflict


@dataclass(frozen=True)
class MaskConflictAudit:
    conflict_index: int
    capsule_a: int
    capsule_b: int
    segment_a: int
    segment_b: int
    classification: str
    input_valid: bool
    point_a_mm: tuple[float, float, float]
    point_b_mm: tuple[float, float, float]
    capsule_clearance_mm: float
    centreline_distance_mm: float
    line_background_gap_mm: float
    locally_connected: bool
    tangent_cosine: float
    mask_radius_a_mm: float
    mask_radius_b_mm: float
    diagnostic_path: str | None

    def to_dict(self) -> dict:
        return asdict(self)


class _LatticeSampler:
    def __init__(self, lattice: AmiraByteLattice) -> None:
        self.lattice = lattice
        self.cache: dict[int, np.ndarray] = {}

    def _slice(self, z: int) -> np.ndarray:
        if z not in self.cache:
            self.cache[z] = self.lattice.slice_z(z) > 0
            # Cross-section/line queries are local; prevent accidental growth
            # when auditing many widely separated conflicts.
            if len(self.cache) > 32:
                self.cache.pop(next(iter(self.cache)))
        return self.cache[z]

    def world_to_index(self, points_mm: np.ndarray) -> np.ndarray:
        return (
            np.asarray(points_mm, dtype=np.float64)
            - self.lattice.header.origin_mm[None, :]
        ) / self.lattice.header.spacing_mm[None, :]

    def sample(self, points_mm: np.ndarray) -> np.ndarray:
        indices = np.rint(self.world_to_index(points_mm)).astype(np.int64)
        valid = (
            (indices[:, 0] >= 0)
            & (indices[:, 0] < self.lattice.nx)
            & (indices[:, 1] >= 0)
            & (indices[:, 1] < self.lattice.ny)
            & (indices[:, 2] >= 0)
            & (indices[:, 2] < self.lattice.nz)
        )
        result = np.zeros(len(indices), dtype=bool)
        for z in np.unique(indices[valid, 2]):
            selection = valid & (indices[:, 2] == z)
            result[selection] = self._slice(int(z))[
                indices[selection, 1], indices[selection, 0]
            ]
        return result


def _orthonormal_plane(tangent: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    tangent = np.asarray(tangent, dtype=np.float64)
    tangent /= max(float(np.linalg.norm(tangent)), 1e-30)
    seed = np.asarray([1.0, 0.0, 0.0])
    if abs(float(np.dot(seed, tangent))) > 0.85:
        seed = np.asarray([0.0, 1.0, 0.0])
    u = np.cross(tangent, seed)
    u /= max(float(np.linalg.norm(u)), 1e-30)
    return u, np.cross(tangent, u)


def _mask_equivalent_radius(
    sampler: _LatticeSampler,
    point: np.ndarray,
    tangent: np.ndarray,
    expected_radius: float,
) -> float:
    step = float(np.min(sampler.lattice.header.spacing_mm))
    half_width = max(1.5 * expected_radius, 3.0 * step)
    count = min(int(np.ceil(2.0 * half_width / step)) + 1, 129)
    offsets = np.linspace(-half_width, half_width, count)
    u, v = _orthonormal_plane(tangent)
    aa, bb = np.meshgrid(offsets, offsets, indexing="ij")
    probes = point[None, :] + aa.ravel()[:, None] * u + bb.ravel()[:, None] * v
    inside = sampler.sample(probes)
    pixel_area = (2.0 * half_width / max(count - 1, 1)) ** 2
    return float(np.sqrt(np.count_nonzero(inside) * pixel_area / np.pi))


def _longest_false_run(values: np.ndarray) -> int:
    longest = current = 0
    for value in values:
        if value:
            current = 0
        else:
            current += 1
            longest = max(longest, current)
    return longest


def _local_connectivity(
    lattice: AmiraByteLattice,
    point_a: np.ndarray,
    point_b: np.ndarray,
    seed_radius: float,
) -> tuple[bool, np.ndarray, np.ndarray]:
    spacing = lattice.header.spacing_mm
    margin = max(2.0 * seed_radius, 3.0 * float(np.max(spacing)))
    lo_world = np.minimum(point_a, point_b) - margin
    hi_world = np.maximum(point_a, point_b) + margin
    lo = np.floor((lo_world - lattice.header.origin_mm) / spacing).astype(int)
    hi = np.ceil((hi_world - lattice.header.origin_mm) / spacing).astype(int) + 1
    lo = np.maximum(lo, 0)
    hi = np.minimum(hi, [lattice.nx, lattice.ny, lattice.nz])
    if np.any(hi <= lo):
        return False, np.empty((0, 0, 0), bool), lo
    volume = np.empty((hi[2] - lo[2], hi[1] - lo[1], hi[0] - lo[0]), dtype=bool)
    for local_z, z in enumerate(range(int(lo[2]), int(hi[2]))):
        volume[local_z] = lattice.slice_z(z)[lo[1] : hi[1], lo[0] : hi[0]] > 0
    labels, _count = ndimage.label(volume, structure=ndimage.generate_binary_structure(3, 1))

    def seed_labels(point: np.ndarray) -> set[int]:
        index = (point - lattice.header.origin_mm) / spacing - lo
        zz, yy, xx = np.indices(volume.shape)
        distance2 = (
            ((xx - index[0]) * spacing[0]) ** 2
            + ((yy - index[1]) * spacing[1]) ** 2
            + ((zz - index[2]) * spacing[2]) ** 2
        )
        values = np.unique(labels[distance2 <= seed_radius**2])
        return {int(value) for value in values if value > 0}

    connected = bool(seed_labels(point_a) & seed_labels(point_b))
    return connected, volume, lo


def audit_capsule_conflicts_against_mask(
    capsules: CapsuleArrays,
    conflicts: Iterable[CapsuleConflict],
    lattice: AmiraByteLattice,
    *,
    diagnostic_dir: str | Path | None = None,
) -> list[MaskConflictAudit]:
    """Classify reported fixed-radius overlaps using the source Labels mask.

    A background run on the shortest centreline connector directly contradicts
    an overlapping capsule pair. Confirmed local foreground connectivity is
    accepted as source-mask contact. Duplicate/indeterminate geometry remains
    invalid and is never corrected automatically.
    """

    sampler = _LatticeSampler(lattice)
    output = Path(diagnostic_dir) if diagnostic_dir is not None else None
    if output is not None:
        output.mkdir(parents=True, exist_ok=True)
    voxel = float(np.max(lattice.header.spacing_mm))
    reports: list[MaskConflictAudit] = []
    for conflict_index, conflict in enumerate(conflicts):
        i, j = conflict.capsule_a, conflict.capsule_b
        point_a = capsules.starts[i] + conflict.parameter_a * (
            capsules.ends[i] - capsules.starts[i]
        )
        point_b = capsules.starts[j] + conflict.parameter_b * (
            capsules.ends[j] - capsules.starts[j]
        )
        tangent_a = capsules.ends[i] - capsules.starts[i]
        tangent_b = capsules.ends[j] - capsules.starts[j]
        tangent_cosine = abs(
            float(np.dot(tangent_a, tangent_b))
            / max(float(np.linalg.norm(tangent_a) * np.linalg.norm(tangent_b)), 1e-30)
        )
        distance = float(np.linalg.norm(point_b - point_a))
        sample_step = max(0.25 * float(np.min(lattice.header.spacing_mm)), 1e-6)
        line_count = min(max(int(np.ceil(distance / sample_step)) + 1, 2), 4096)
        line = np.linspace(point_a, point_b, line_count)
        line_values = sampler.sample(line)
        background_gap = _longest_false_run(line_values) * (
            distance / max(line_count - 1, 1)
        )
        seed_radius = max(1.5 * voxel, 0.35 * min(conflict.radius_a, conflict.radius_b))
        connected, roi, roi_lo = _local_connectivity(
            lattice, point_a, point_b, seed_radius
        )
        duplicate = distance <= voxel and tangent_cosine >= 0.95
        if duplicate:
            classification = "duplicated_graph_geometry"
        elif background_gap >= 0.5 * voxel:
            classification = "radius_overestimation"
        elif connected and bool(np.all(line_values)):
            classification = "mask_confirmed_contact"
        else:
            classification = "indeterminate"
        input_valid = classification == "mask_confirmed_contact"
        diagnostic_path = None
        if output is not None:
            path = output / (
                f"conflict_{conflict_index:04d}_seg{conflict.segment_a}_"
                f"seg{conflict.segment_b}.npz"
            )
            np.savez_compressed(
                path,
                labels=roi.astype(np.uint8),
                roi_index_origin_xyz=roi_lo,
                lattice_origin_mm=lattice.header.origin_mm,
                spacing_mm=lattice.header.spacing_mm,
                point_a_mm=point_a,
                point_b_mm=point_b,
                line_labels=line_values.astype(np.uint8),
            )
            diagnostic_path = str(path)
        reports.append(
            MaskConflictAudit(
                conflict_index=conflict_index,
                capsule_a=i,
                capsule_b=j,
                segment_a=conflict.segment_a,
                segment_b=conflict.segment_b,
                classification=classification,
                input_valid=input_valid,
                point_a_mm=tuple(float(value) for value in point_a),
                point_b_mm=tuple(float(value) for value in point_b),
                capsule_clearance_mm=float(conflict.clearance),
                centreline_distance_mm=distance,
                line_background_gap_mm=float(background_gap),
                locally_connected=connected,
                tangent_cosine=tangent_cosine,
                mask_radius_a_mm=_mask_equivalent_radius(
                    sampler, point_a, tangent_a, conflict.radius_a
                ),
                mask_radius_b_mm=_mask_equivalent_radius(
                    sampler, point_b, tangent_b, conflict.radius_b
                ),
                diagnostic_path=diagnostic_path,
            )
        )
    return reports


__all__ = ["MaskConflictAudit", "audit_capsule_conflicts_against_mask"]
