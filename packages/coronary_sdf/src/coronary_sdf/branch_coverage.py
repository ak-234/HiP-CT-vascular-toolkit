"""Coverage of the source centreline by an extracted lumen surface.

Answers a question no existing metric answers: for every source segment, how far
is the nearest wall vertex, measured in units of the *local radius*? A vessel
that the reconstruction dropped entirely shows up here as a segment whose
samples all sit far outside any wall, which surface-to-mask distance and
cross-sectional area both miss because they never sample the centreline.

Kept separate from :mod:`coronary_sdf.benchmark_metrics` because coverage is
graph-versus-mesh and needs no segmentation lattice, so the synthetic screen,
the real screen and the tests can all use it without the mask machinery.

Two measurement caveats are recorded in the report rather than hidden:

* Distances are measured to mesh *vertices*, so the value is an upper bound
  biased by the extractor's triangle size. For an adaptive mesh that bias is
  ``~r/cells_across_diameter`` and therefore scale-free; for a dense mesh it is
  ``~h`` and fixed, which for a 57 µm vessel at ``h = 66 µm`` exceeds the radius
  itself. ``extractor_characteristic_length_mm`` is carried alongside so dense
  and adaptive numbers are never compared blind.
* Component sizes use point-connectivity, which merges components meeting at a
  single vertex. That is conservative for speckle detection.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Iterable, Sequence

import numpy as np
import pyvista as pv
from scipy.spatial import cKDTree

# A vessel whose nearest wall sits beyond this many local radii is not merely
# under-resolved; the surface is not describing that vessel at all.
LOST_BRANCH_RADII = 2.0


@dataclass(frozen=True)
class SegmentCoverage:
    segment_index: int
    segment_id: int
    sample_count: int
    is_terminal: bool
    minimum_radius_mm: float
    median_radius_mm: float
    median_wall_distance_radii: float
    p95_wall_distance_radii: float
    max_wall_distance_radii: float
    median_wall_distance_mm: float
    fraction_beyond_two_radii: float
    inside_fraction: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class TinyComponent:
    """A small isolated surface island, attributed to the geometry near it.

    ``nearest_radius_mm`` is the diagnostic that matters: a cluster of islands
    sitting on sub-micron-radius centreline points indicates degenerate input
    geometry driving the extractor, not a field or extractor defect.
    """

    face_count: int
    centroid_mm: tuple[float, float, float]
    nearest_point_id: int
    nearest_segment_index: int
    nearest_radius_mm: float
    distance_mm: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CoverageReport:
    sample_count: int
    segment_count: int
    median_wall_distance_radii: float
    p95_wall_distance_radii: float
    fraction_beyond_two_radii: float
    missing_segment_indices: tuple[int, ...]
    missing_segment_ids: tuple[int, ...]
    weak_segment_indices: tuple[int, ...]
    terminal_segment_indices: tuple[int, ...]
    missing_terminal_segment_indices: tuple[int, ...]
    component_count: int
    component_face_counts: tuple[int, ...]
    largest_component_face_fraction: float
    tiny_component_count: int
    tiny_component_face_threshold: int
    tiny_components: tuple[TinyComponent, ...]
    extractor_characteristic_length_mm: float | None
    per_segment: tuple[SegmentCoverage, ...]

    @property
    def complete(self) -> bool:
        """True when no source segment lost its wall."""

        return not self.missing_segment_indices

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["complete"] = self.complete
        return result

    def summary(self) -> dict[str, Any]:
        """Compact form for a run record, without the per-segment table."""

        result = self.to_dict()
        result.pop("per_segment", None)
        return result


def segment_centreline_samples(
    points: dict[int, tuple],
    segments: Sequence[dict[str, Any]],
    *,
    include_endpoints: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(coords_mm, radii_mm, segment_index, point_id)`` for all samples.

    Single source for the ``points[pid][:3] / 1000`` and ``points[pid][3] / 1000``
    idiom; the graph stores µm and the surface is in mm.
    """

    coords: list[np.ndarray] = []
    radii: list[float] = []
    seg_index: list[int] = []
    point_ids: list[int] = []
    for index, segment in enumerate(segments):
        ids = [pid for pid in segment.get("point_ids", []) if pid in points]
        if not include_endpoints and len(ids) > 2:
            ids = ids[1:-1]
        for pid in ids:
            record = points[pid]
            coords.append(np.asarray(record[:3], dtype=np.float64) / 1000.0)
            radii.append(float(record[3]) / 1000.0)
            seg_index.append(index)
            point_ids.append(int(pid))
    if not coords:
        return (
            np.empty((0, 3), dtype=np.float64),
            np.empty(0, dtype=np.float64),
            np.empty(0, dtype=np.int64),
            np.empty(0, dtype=np.int64),
        )
    return (
        np.asarray(coords, dtype=np.float64),
        np.asarray(radii, dtype=np.float64),
        np.asarray(seg_index, dtype=np.int64),
        np.asarray(point_ids, dtype=np.int64),
    )


def terminal_segment_indices(segments: Sequence[dict[str, Any]]) -> set[int]:
    """Indices of segments with at least one degree-1 endpoint node."""

    degree: dict[int, int] = {}
    for segment in segments:
        for key in ("node1", "node2"):
            node = segment.get(key)
            if node is not None:
                degree[int(node)] = degree.get(int(node), 0) + 1
    result: set[int] = set()
    for index, segment in enumerate(segments):
        for key in ("node1", "node2"):
            node = segment.get(key)
            if node is not None and degree.get(int(node), 0) <= 1:
                result.add(index)
                break
    return result


def _combined(surfaces: Iterable[pv.PolyData]) -> pv.PolyData | None:
    meshes = [s for s in surfaces if s is not None and s.n_points]
    if not meshes:
        return None
    if len(meshes) == 1:
        return meshes[0].triangulate()
    merged = meshes[0].triangulate()
    for mesh in meshes[1:]:
        merged = merged.merge(mesh.triangulate())
    return merged


def component_face_counts(surface: pv.PolyData) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(per_face_label, face_count_per_label)`` for ``surface``."""

    if surface is None or not surface.n_cells:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
    tagged = surface.connectivity(extraction_mode="all")
    labels = np.asarray(tagged.cell_data["RegionId"], dtype=np.int64)
    if not len(labels):
        return labels, np.empty(0, dtype=np.int64)
    return labels, np.bincount(labels)


def branch_coverage(
    surfaces: Iterable[pv.PolyData],
    points: dict[int, tuple],
    segments: Sequence[dict[str, Any]],
    *,
    tiny_component_faces: int = 64,
    containment: bool = False,
    extractor_characteristic_length_mm: float | None = None,
) -> CoverageReport:
    """Measure how well ``surfaces`` cover the centreline of ``segments``.

    ``containment`` adds a point-in-surface test per sample. It is off by
    default because ``select_enclosed_points`` against a multi-million-face mesh
    dominates the runtime of the whole harness; enable it for small regions.
    """

    surface_list = [s for s in surfaces if s is not None]
    coords, radii, seg_index, point_ids = segment_centreline_samples(points, segments)
    combined = _combined(surface_list)
    terminals = terminal_segment_indices(segments)

    if combined is None or not len(coords):
        # No surface (or no graph) means nothing is covered; report it as such
        # rather than raising, so a failed component still yields a record.
        return CoverageReport(
            sample_count=int(len(coords)),
            segment_count=len(segments),
            median_wall_distance_radii=float("inf"),
            p95_wall_distance_radii=float("inf"),
            fraction_beyond_two_radii=1.0 if len(coords) else 0.0,
            missing_segment_indices=tuple(range(len(segments))),
            missing_segment_ids=tuple(
                int(s.get("id", i)) for i, s in enumerate(segments)
            ),
            weak_segment_indices=tuple(range(len(segments))),
            terminal_segment_indices=tuple(sorted(terminals)),
            missing_terminal_segment_indices=tuple(sorted(terminals)),
            component_count=0,
            component_face_counts=(),
            largest_component_face_fraction=0.0,
            tiny_component_count=0,
            tiny_component_face_threshold=int(tiny_component_faces),
            tiny_components=(),
            extractor_characteristic_length_mm=extractor_characteristic_length_mm,
            per_segment=(),
        )

    vertices = np.asarray(combined.points, dtype=np.float64)
    distance_mm, _ = cKDTree(vertices).query(coords, workers=-1)
    safe_radii = np.maximum(radii, 1e-12)
    distance_radii = distance_mm / safe_radii

    inside: np.ndarray | None = None
    if containment:
        selected = pv.PolyData(coords).select_enclosed_points(
            combined, tolerance=1e-7, check_surface=False
        )
        inside = np.asarray(selected["SelectedPoints"], dtype=bool)

    per_segment: list[SegmentCoverage] = []
    missing: list[int] = []
    weak: list[int] = []
    for index, segment in enumerate(segments):
        mask = seg_index == index
        if not np.any(mask):
            continue
        values = distance_radii[mask]
        beyond = float(np.mean(values > LOST_BRANCH_RADII))
        p95 = float(np.percentile(values, 95))
        coverage = SegmentCoverage(
            segment_index=index,
            segment_id=int(segment.get("id", index)),
            sample_count=int(mask.sum()),
            is_terminal=index in terminals,
            minimum_radius_mm=float(np.min(radii[mask])),
            median_radius_mm=float(np.median(radii[mask])),
            median_wall_distance_radii=float(np.median(values)),
            p95_wall_distance_radii=p95,
            max_wall_distance_radii=float(np.max(values)),
            median_wall_distance_mm=float(np.median(distance_mm[mask])),
            fraction_beyond_two_radii=beyond,
            inside_fraction=(
                float(np.mean(inside[mask])) if inside is not None else None
            ),
        )
        per_segment.append(coverage)
        # A majority of samples beyond two local radii means the surface does not
        # describe this vessel at all; p95 alone only means the tip is ragged.
        if beyond > 0.5:
            missing.append(index)
        elif p95 > LOST_BRANCH_RADII:
            weak.append(index)

    labels, counts = component_face_counts(combined)
    order = np.argsort(counts)[::-1] if len(counts) else np.empty(0, dtype=np.int64)
    sorted_counts = tuple(int(counts[i]) for i in order)
    total_faces = int(combined.n_cells)
    largest_fraction = (
        float(sorted_counts[0]) / float(total_faces) if sorted_counts else 0.0
    )

    tiny: list[TinyComponent] = []
    if len(counts):
        centres = np.asarray(combined.cell_centers().points, dtype=np.float64)
        sample_tree = cKDTree(coords)
        for label in np.flatnonzero(counts <= int(tiny_component_faces)):
            member = labels == label
            centroid = centres[member].mean(axis=0)
            gap, nearest = sample_tree.query(centroid, workers=-1)
            tiny.append(
                TinyComponent(
                    face_count=int(counts[label]),
                    centroid_mm=tuple(float(v) for v in centroid),
                    nearest_point_id=int(point_ids[nearest]),
                    nearest_segment_index=int(seg_index[nearest]),
                    nearest_radius_mm=float(radii[nearest]),
                    distance_mm=float(gap),
                )
            )
        tiny.sort(key=lambda item: (item.face_count, item.distance_mm))

    return CoverageReport(
        sample_count=int(len(coords)),
        segment_count=len(segments),
        median_wall_distance_radii=float(np.median(distance_radii)),
        p95_wall_distance_radii=float(np.percentile(distance_radii, 95)),
        fraction_beyond_two_radii=float(np.mean(distance_radii > LOST_BRANCH_RADII)),
        missing_segment_indices=tuple(missing),
        missing_segment_ids=tuple(
            int(segments[i].get("id", i)) for i in missing
        ),
        weak_segment_indices=tuple(weak),
        terminal_segment_indices=tuple(sorted(terminals)),
        missing_terminal_segment_indices=tuple(
            sorted(i for i in missing if i in terminals)
        ),
        component_count=int(len(counts)),
        component_face_counts=sorted_counts,
        largest_component_face_fraction=largest_fraction,
        tiny_component_count=len(tiny),
        tiny_component_face_threshold=int(tiny_component_faces),
        tiny_components=tuple(tiny),
        extractor_characteristic_length_mm=extractor_characteristic_length_mm,
        per_segment=tuple(per_segment),
    )


__all__ = [
    "LOST_BRANCH_RADII",
    "CoverageReport",
    "SegmentCoverage",
    "TinyComponent",
    "branch_coverage",
    "component_face_counts",
    "segment_centreline_samples",
    "terminal_segment_indices",
]
