"""Deterministic geometry and segmentation metrics for reconstruction benchmarks."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Iterable

import numpy as np
import pyvista as pv
from scipy.spatial import cKDTree

from .amira_lattice import AmiraByteLattice


def _deterministic_subsample(points: np.ndarray, maximum: int) -> np.ndarray:
    if len(points) <= maximum:
        return points
    indices = np.linspace(0, len(points) - 1, maximum, dtype=np.int64)
    return points[indices]


def combine_surfaces(surfaces: Iterable[pv.PolyData]) -> pv.PolyData:
    surfaces = list(surfaces)
    if not surfaces:
        return pv.PolyData()
    result = surfaces[0].triangulate()
    for surface in surfaces[1:]:
        result = result.merge(surface.triangulate(), merge_points=False)
    return result


def symmetric_surface_metrics(
    candidate: pv.PolyData,
    reference: pv.PolyData,
    *,
    maximum_samples: int = 250_000,
) -> dict[str, float]:
    """Symmetric vertex-sampled surface distances in model units."""

    a = _deterministic_subsample(np.asarray(candidate.points), maximum_samples)
    b = _deterministic_subsample(np.asarray(reference.points), maximum_samples)
    if not len(a) or not len(b):
        return {"mean": math.inf, "median": math.inf, "hd95": math.inf, "hausdorff": math.inf}
    da = cKDTree(b).query(a, workers=-1)[0]
    db = cKDTree(a).query(b, workers=-1)[0]
    distances = np.concatenate((da, db))
    return {
        "mean": float(np.mean(distances)),
        "median": float(np.median(distances)),
        "hd95": float(np.percentile(distances, 95)),
        "hausdorff": float(np.max(distances)),
    }


@dataclass(frozen=True)
class SegmentationSamples:
    foreground_count: int
    foreground_points_mm: np.ndarray
    boundary_points_mm: np.ndarray


def _voxel_window(
    lattice: AmiraByteLattice, bounds_mm: tuple[np.ndarray, np.ndarray] | None
) -> tuple[int, int, int, int, int, int]:
    """Half-open voxel window ``(x0, x1, y0, y1, z0, z1)`` covering ``bounds_mm``."""

    dims = (lattice.nx, lattice.ny, lattice.nz)
    if bounds_mm is None:
        return 0, dims[0], 0, dims[1], 0, dims[2]
    lower = np.asarray(bounds_mm[0], dtype=float)
    upper = np.asarray(bounds_mm[1], dtype=float)
    origin = lattice.header.origin_mm
    spacing = np.maximum(lattice.header.spacing_mm, 1e-30)
    low = np.floor((lower - origin) / spacing).astype(np.int64)
    high = np.ceil((upper - origin) / spacing).astype(np.int64) + 1
    low = np.clip(low, 0, np.asarray(dims) )
    high = np.clip(high, 0, np.asarray(dims))
    return (
        int(low[0]), int(high[0]),
        int(low[1]), int(high[1]),
        int(low[2]), int(high[2]),
    )


def sample_segmentation(
    lattice: AmiraByteLattice,
    *,
    maximum_foreground: int = 500_000,
    maximum_boundary: int = 500_000,
    bounds_mm: tuple[np.ndarray, np.ndarray] | None = None,
) -> SegmentationSamples:
    """Stream a sparse label volume and deterministically sample foreground/boundary.

    ``bounds_mm`` restricts which samples are *emitted* without changing how
    interior/boundary status is decided: the neighbour stencil still sees the
    full slices, and the z sweep carries a one-slice guard band past each end of
    the window. The emitted boundary set is therefore exactly the full-volume
    boundary set restricted to the box, rather than a set that mistakes the cut
    faces of the box for vessel wall.
    """

    x0, x1, y0, y1, z0, z1 = _voxel_window(lattice, bounds_mm)
    if x0 >= x1 or y0 >= y1 or z0 >= z1:
        empty = np.empty((0, 3), dtype=np.float64)
        return SegmentationSamples(0, empty, empty)

    foreground_voxels = np.empty((0, 3), dtype=np.int64)
    foreground_keys = np.empty(0, dtype=np.uint64)
    boundary_voxels = np.empty((0, 3), dtype=np.int64)
    boundary_keys = np.empty(0, dtype=np.uint64)
    foreground_count = 0

    def _in_window(xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
        if bounds_mm is None:
            return np.ones(len(xs), dtype=bool)
        return (xs >= x0) & (xs < x1) & (ys >= y0) & (ys < y1)

    # Guard band: start one slice early so ``previous`` is correct at z0.
    scan_start = max(z0 - 1, 0)
    previous = (
        (lattice.slice_z(scan_start - 1) > 0)
        if scan_start > 0
        else np.zeros((lattice.ny, lattice.nx), dtype=bool)
    )
    current = lattice.slice_z(scan_start) > 0
    for z in range(scan_start, min(z1 + 1, lattice.nz)):
        following = (
            lattice.slice_z(z + 1) > 0
            if z + 1 < lattice.nz
            else np.zeros_like(current)
        )
        emit = z0 <= z < z1
        if not emit:
            previous, current = current, following
            continue
        yz, xz = np.nonzero(current)
        if len(xz):
            keep_window = _in_window(xz, yz)
            xz, yz = xz[keep_window], yz[keep_window]
        foreground_count += int(len(xz))
        if len(xz):
            incoming = np.column_stack((xz, yz, np.full(len(xz), z, dtype=np.int64)))
            linear = xz.astype(np.uint64) + np.uint64(lattice.nx) * (
                yz.astype(np.uint64) + np.uint64(lattice.ny * z)
            )
            keys = linear * np.uint64(11400714819323198485)
            foreground_voxels = np.vstack((foreground_voxels, incoming))
            foreground_keys = np.concatenate((foreground_keys, keys))
            if len(foreground_keys) > maximum_foreground:
                keep = np.argpartition(foreground_keys, maximum_foreground - 1)[
                    :maximum_foreground
                ]
                foreground_voxels = foreground_voxels[keep]
                foreground_keys = foreground_keys[keep]
        padded = np.pad(current, 1, constant_values=False)
        interior = (
            padded[1:-1, :-2]
            & padded[1:-1, 2:]
            & padded[:-2, 1:-1]
            & padded[2:, 1:-1]
            & previous
            & following
        )
        boundary = current & ~interior
        yb, xb = np.nonzero(boundary)
        if len(xb):
            keep_window = _in_window(xb, yb)
            xb, yb = xb[keep_window], yb[keep_window]
        if len(xb):
            incoming = np.column_stack((xb, yb, np.full(len(xb), z, dtype=np.int64)))
            linear = xb.astype(np.uint64) + np.uint64(lattice.nx) * (
                yb.astype(np.uint64) + np.uint64(lattice.ny * z)
            )
            keys = linear * np.uint64(11400714819323198485)
            boundary_voxels = np.vstack((boundary_voxels, incoming))
            boundary_keys = np.concatenate((boundary_keys, keys))
            if len(boundary_keys) > maximum_boundary:
                keep = np.argpartition(boundary_keys, maximum_boundary - 1)[
                    :maximum_boundary
                ]
                boundary_voxels = boundary_voxels[keep]
                boundary_keys = boundary_keys[keep]
        previous, current = current, following

    def convert(voxels: np.ndarray) -> np.ndarray:
        if not len(voxels):
            return np.empty((0, 3), dtype=np.float64)
        return lattice.header.origin_mm + voxels * lattice.header.spacing_mm

    return SegmentationSamples(
        foreground_count=foreground_count,
        foreground_points_mm=convert(foreground_voxels),
        boundary_points_mm=convert(boundary_voxels),
    )


def _implicit_distances(points: np.ndarray, surfaces: list[pv.PolyData]) -> np.ndarray:
    if not len(points) or not surfaces:
        return np.full(len(points), math.inf)
    distances = np.full(len(points), math.inf)
    cloud = pv.PolyData(points)
    for surface in surfaces:
        evaluated = cloud.compute_implicit_distance(surface, inplace=False)
        distances = np.minimum(distances, np.abs(np.asarray(evaluated["implicit_distance"])))
    return distances


def _inside_any(points: np.ndarray, surfaces: list[pv.PolyData]) -> np.ndarray:
    inside = np.zeros(len(points), dtype=bool)
    if not len(points):
        return inside
    cloud = pv.PolyData(points)
    for surface in surfaces:
        selected = cloud.select_enclosed_points(
            surface,
            tolerance=0.0,
            check_surface=False,
        )
        inside |= np.asarray(selected["SelectedPoints"], dtype=bool)
    return inside


def restrict_samples(
    samples: SegmentationSamples,
    centreline_mm: np.ndarray,
    radii_mm: np.ndarray,
    *,
    capture_factor: float = 2.0,
) -> SegmentationSamples:
    """Keep only mask samples belonging to vessels the graph actually claims.

    A box restriction alone still admits mask voxels from branches this graph
    does not represent — other components, or branches whose segments lie mostly
    outside the region. Those have no mesh anywhere near them and would dominate
    HD95. Samples are kept when they fall within ``capture_factor`` local radii
    of some centreline point.
    """

    centreline_mm = np.asarray(centreline_mm, dtype=float)
    radii_mm = np.asarray(radii_mm, dtype=float)
    if not len(centreline_mm):
        return samples
    tree = cKDTree(centreline_mm)

    def _keep(points: np.ndarray) -> np.ndarray:
        if not len(points):
            return points
        distance, index = tree.query(points, workers=-1)
        return points[distance <= capture_factor * np.maximum(radii_mm[index], 1e-12)]

    kept_foreground = _keep(samples.foreground_points_mm)
    scale = (
        len(kept_foreground) / len(samples.foreground_points_mm)
        if len(samples.foreground_points_mm)
        else 0.0
    )
    return SegmentationSamples(
        foreground_count=int(round(samples.foreground_count * scale)),
        foreground_points_mm=kept_foreground,
        boundary_points_mm=_keep(samples.boundary_points_mm),
    )


def segmentation_surface_metrics(
    surfaces: Iterable[pv.PolyData],
    lattice: AmiraByteLattice,
    *,
    samples: SegmentationSamples | None = None,
    bounds_mm: tuple[np.ndarray, np.ndarray] | None = None,
    metric_scope: str | None = None,
) -> dict[str, Any]:
    """Acquisition-aware surface distance and approximate volumetric Dice.

    ``bounds_mm`` restricts the reference to a region. A restricted score is a
    *conditional* metric — distance to the mask surface given that surface lies
    in the region the graph describes — and is not comparable with a
    whole-volume score, so the scope is reported alongside it.
    """

    surfaces = [surface.triangulate() for surface in surfaces]
    if samples is None:
        samples = sample_segmentation(lattice, bounds_mm=bounds_mm)
    boundary = samples.boundary_points_mm
    if not len(boundary) or not surfaces:
        # A region with no reference boundary cannot be scored; say so rather
        # than returning a flattering zero.
        return {
            "surface_mean_mm": float("inf"),
            "surface_median_mm": float("inf"),
            "hd95_mm": float("inf"),
            "hausdorff_mm": float("inf"),
            "dice_approx": 0.0,
            "foreground_voxels": float(samples.foreground_count),
            "reference_voxel_mm": float(np.max(lattice.header.spacing_mm)),
            "metric_scope": metric_scope
            or ("roi_restricted" if bounds_mm is not None else "full_volume"),
            "unscorable_reason": (
                "no reference boundary samples in scope"
                if not len(boundary)
                else "no surface"
            ),
        }
    segmentation_to_mesh = _implicit_distances(boundary, surfaces)
    mesh_vertices = _deterministic_subsample(
        np.vstack([surface.points for surface in surfaces]), 500_000
    )
    mesh_to_segmentation = cKDTree(boundary).query(mesh_vertices, workers=-1)[0]
    symmetric = np.concatenate((segmentation_to_mesh, mesh_to_segmentation))

    foreground_inside_fraction = float(
        np.mean(_inside_any(samples.foreground_points_mm, surfaces))
    ) if len(samples.foreground_points_mm) else 0.0
    intersection_voxels = foreground_inside_fraction * samples.foreground_count
    voxel_volume = float(np.prod(lattice.header.spacing_mm))
    predicted_voxels = sum(abs(float(surface.volume)) for surface in surfaces) / voxel_volume
    denominator = samples.foreground_count + predicted_voxels
    dice = 2.0 * intersection_voxels / denominator if denominator > 0 else 1.0
    return {
        "surface_mean_mm": float(np.mean(symmetric)),
        "surface_median_mm": float(np.median(symmetric)),
        "hd95_mm": float(np.percentile(symmetric, 95)),
        "hausdorff_mm": float(np.max(symmetric)),
        "dice_approx": float(np.clip(dice, 0.0, 1.0)),
        "foreground_voxels": float(samples.foreground_count),
        "reference_voxel_mm": float(np.max(lattice.header.spacing_mm)),
        "metric_scope": metric_scope
        or ("roi_restricted" if bounds_mm is not None else "full_volume"),
    }


class _SliceCache:
    def __init__(self, lattice: AmiraByteLattice, maximum: int = 16):
        self.lattice = lattice
        self.maximum = maximum
        self.images: dict[int, np.ndarray] = {}
        self.order: list[int] = []

    def get(self, z: int) -> np.ndarray:
        z = int(np.clip(z, 0, self.lattice.nz - 1))
        if z not in self.images:
            self.images[z] = self.lattice.slice_z(z) > 0
            self.order.append(z)
            if len(self.order) > self.maximum:
                oldest = self.order.pop(0)
                self.images.pop(oldest, None)
        return self.images[z]


def _sample_labels_nearest(
    points_mm: np.ndarray, lattice: AmiraByteLattice, cache: _SliceCache
) -> np.ndarray:
    voxels = np.rint(
        (points_mm - lattice.header.origin_mm) / lattice.header.spacing_mm
    ).astype(np.int64)
    valid = np.all(voxels >= 0, axis=1) & np.all(voxels < lattice.header.dims, axis=1)
    result = np.zeros(len(points_mm), dtype=bool)
    for z in np.unique(voxels[valid, 2]):
        selection = valid & (voxels[:, 2] == z)
        v = voxels[selection]
        result[selection] = cache.get(int(z))[v[:, 1], v[:, 0]]
    return result


def cross_section_area_metrics(
    surfaces: Iterable[pv.PolyData],
    lattice: AmiraByteLattice,
    points: dict[int, tuple],
    segments: list[dict],
    *,
    stations_per_segment: int = 3,
) -> dict[str, Any]:
    """Compare source/predicted planar occupancy at resolved branch stations.

    Stations on vessels narrower than three source voxels cannot be scored at
    all. Their count is reported, because a region made entirely of such vessels
    returns no resolved sections — and that is a measurement limit, not a
    geometric failure.
    """

    surfaces = [surface.triangulate() for surface in surfaces]
    voxel = float(np.max(lattice.header.spacing_mm))
    cache = _SliceCache(lattice)
    errors: list[float] = []
    hydraulic_errors: list[float] = []
    segment_errors: dict[int, list[float]] = {}
    skipped_unresolved = 0
    skipped_short_segments = 0
    for segment_index, segment in enumerate(segments):
        ids = [pid for pid in segment.get("point_ids", []) if pid in points]
        if len(ids) < 3:
            skipped_short_segments += 1
            continue
        coords = np.asarray([points[pid][:3] for pid in ids], dtype=float) / 1000.0
        radii = np.asarray([points[pid][3] for pid in ids], dtype=float) / 1000.0
        for fraction in np.linspace(0.25, 0.75, stations_per_segment):
            index = int(round(fraction * (len(ids) - 1)))
            index = int(np.clip(index, 1, len(ids) - 2))
            radius = float(radii[index])
            if 2.0 * radius < 3.0 * voxel:
                skipped_unresolved += 1
                continue
            tangent = coords[index + 1] - coords[index - 1]
            tangent /= max(float(np.linalg.norm(tangent)), 1e-30)
            seed = np.asarray([1.0, 0.0, 0.0])
            if abs(float(np.dot(seed, tangent))) > 0.8:
                seed = np.asarray([0.0, 1.0, 0.0])
            u = np.cross(tangent, seed)
            u /= np.linalg.norm(u)
            v = np.cross(tangent, u)
            half_width = 1.5 * radius + 2.0 * voxel
            axis = np.arange(-half_width, half_width + 0.5 * voxel, voxel)
            aa, bb = np.meshgrid(axis, axis, indexing="xy")
            samples = (
                coords[index]
                + aa.reshape(-1, 1) * u
                + bb.reshape(-1, 1) * v
            )
            reference_inside = _sample_labels_nearest(samples, lattice, cache)
            predicted_inside = _inside_any(samples, surfaces)
            reference_area = float(np.count_nonzero(reference_inside)) * voxel * voxel
            predicted_area = float(np.count_nonzero(predicted_inside)) * voxel * voxel
            if reference_area <= 0:
                continue
            error = abs(predicted_area - reference_area) / reference_area
            errors.append(error)
            segment_errors.setdefault(segment_index, []).append(error)
            ref_diameter = math.sqrt(4.0 * reference_area / math.pi)
            pred_diameter = math.sqrt(4.0 * predicted_area / math.pi)
            hydraulic_errors.append(abs(pred_diameter - ref_diameter) / ref_diameter)
    provenance = {
        "skipped_sections_unresolved": float(skipped_unresolved),
        "skipped_short_segments": float(skipped_short_segments),
        "scored_segments": float(len(segment_errors)),
    }
    if not errors:
        return {
            "resolved_sections": 0.0,
            "median_area_error_fraction": math.inf,
            "p95_area_error_fraction": math.inf,
            "median_hydraulic_diameter_error_fraction": math.inf,
            "per_segment_median_area_error": {},
            **provenance,
        }
    return {
        "resolved_sections": float(len(errors)),
        "median_area_error_fraction": float(np.median(errors)),
        "p95_area_error_fraction": float(np.percentile(errors, 95)),
        "median_hydraulic_diameter_error_fraction": float(np.median(hydraulic_errors)),
        # Per-segment attribution: a single flat list cannot say *which* vessel
        # the area error came from, which is the point of the ablation.
        "per_segment_median_area_error": {
            str(index): float(np.median(values))
            for index, values in sorted(segment_errors.items())
        },
        **provenance,
    }


__all__ = [
    "SegmentationSamples",
    "combine_surfaces",
    "cross_section_area_metrics",
    "restrict_samples",
    "sample_segmentation",
    "segmentation_surface_metrics",
    "symmetric_surface_metrics",
]
