"""Create one disposable, radius-aware master STL for Simpleware meshing.

The validated source STL is never modified.  A cached STL-cropped Amira graph
drives a spatial surface-size field.  MeshLib then simplifies coarse-radius
regions while retaining finer triangles on distal vessels, tight bends,
bifurcations, and close non-adjacent branches.

This is deliberately a *master geometry* operation: all mesh-convergence
levels must use the same accepted output STL.  Changing the surface geometry
between convergence levels would confound geometry and volume-mesh error.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from scipy.spatial import cKDTree


@dataclass(frozen=True)
class SurfaceRemeshConfig:
    circumferential_segments: int = 18
    min_edge_mm: float = 0.025
    max_edge_mm: float = 0.20
    graph_sample_spacing_mm: float = 0.06
    clearance_factor: float = 0.35
    bifurcation_factor: float = 0.65
    bifurcation_length_radii: float = 2.0
    relative_error: float = 0.02
    absolute_error_mm: float = 0.010
    number_of_size_bins: int = 7
    association_candidates: int = 24
    clearance_candidates: int = 64
    max_angle_change_deg: float = 25.0
    maximum_triangle_aspect_ratio: float = 15.0

    def validate(self) -> None:
        if self.circumferential_segments < 8:
            raise ValueError("circumferential_segments must be at least 8")
        if not 0.0 < self.min_edge_mm <= self.max_edge_mm:
            raise ValueError("edge bounds must satisfy 0 < min <= max")
        if self.graph_sample_spacing_mm <= 0.0:
            raise ValueError("graph_sample_spacing_mm must be positive")
        if not 0.0 < self.clearance_factor <= 1.0:
            raise ValueError("clearance_factor must lie in (0, 1]")
        if not 0.0 < self.bifurcation_factor <= 1.0:
            raise ValueError("bifurcation_factor must lie in (0, 1]")
        if self.relative_error <= 0.0 or self.absolute_error_mm <= 0.0:
            raise ValueError("geometry-error limits must be positive")
        if self.number_of_size_bins < 2:
            raise ValueError("number_of_size_bins must be at least 2")


@dataclass(frozen=True)
class GraphSizeField:
    points: np.ndarray
    radii: np.ndarray
    target_edges: np.ndarray
    edge_indices: np.ndarray
    edge_ids: np.ndarray
    nearest_nonadjacent_gap: np.ndarray
    tree: cKDTree


def _log(message: str, started: float) -> None:
    elapsed = time.perf_counter() - started
    print(
        f"[{datetime.now():%Y-%m-%d %H:%M:%S} +{elapsed:8.1f}s] {message}",
        flush=True,
    )


def _cumulative(points: np.ndarray) -> np.ndarray:
    if len(points) == 0:
        return np.empty(0, dtype=float)
    return np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))]


def _resample_polyline(
    points: np.ndarray, radii: np.ndarray, spacing: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    points = np.asarray(points, dtype=float)
    radii = np.asarray(radii, dtype=float)
    distance = _cumulative(points)
    if len(points) < 2 or distance[-1] <= 0.0:
        raise ValueError("graph edge must contain a non-zero-length polyline")
    count = max(2, int(math.ceil(distance[-1] / spacing)) + 1)
    sampled_distance = np.linspace(0.0, distance[-1], count)
    sampled_points = np.column_stack(
        [np.interp(sampled_distance, distance, points[:, axis]) for axis in range(3)]
    )
    sampled_radii = np.interp(sampled_distance, distance, radii)
    return sampled_points, sampled_radii, sampled_distance


def _curvature(points: np.ndarray) -> np.ndarray:
    """Stable discrete centreline curvature in 1/mm."""
    if len(points) < 3:
        return np.zeros(len(points), dtype=float)
    distance = _cumulative(points)
    tangent = np.gradient(points, distance, axis=0, edge_order=1)
    tangent /= np.maximum(np.linalg.norm(tangent, axis=1), 1.0e-12)[:, None]
    derivative = np.gradient(tangent, distance, axis=0, edge_order=1)
    result = np.linalg.norm(derivative, axis=1)
    result[~np.isfinite(result)] = 0.0
    return result


def load_cropped_graph(path: str | Path) -> dict[str, Any]:
    path = Path(path).expanduser().resolve()
    with path.open("r", encoding="utf-8") as stream:
        graph = json.load(stream)
    edges = graph.get("edges")
    if not isinstance(edges, list) or not edges:
        raise ValueError(f"cropped graph contains no edges: {path}")
    required = {"edge_id", "node1", "node2", "points_mm", "radii_mm"}
    for edge in edges:
        missing = required.difference(edge)
        if missing:
            raise ValueError(
                f"cropped graph edge is missing {sorted(missing)}: {path}"
            )
    return graph


def _edge_adjacency(edges: list[dict[str, Any]]) -> list[set[int]]:
    nodes_to_edges: dict[int, set[int]] = {}
    for index, edge in enumerate(edges):
        for node in (int(edge["node1"]), int(edge["node2"])):
            nodes_to_edges.setdefault(node, set()).add(index)
    adjacent: list[set[int]] = []
    for index, edge in enumerate(edges):
        values = {index}
        for node in (int(edge["node1"]), int(edge["node2"])):
            values.update(nodes_to_edges[node])
        adjacent.append(values)
    return adjacent


def build_graph_size_field(
    graph: dict[str, Any], config: SurfaceRemeshConfig
) -> GraphSizeField:
    """Return graph samples and a topology-aware target surface-edge field."""
    config.validate()
    edges = graph["edges"]
    node_degree: dict[int, int] = {}
    for edge in edges:
        for node in (int(edge["node1"]), int(edge["node2"])):
            node_degree[node] = node_degree.get(node, 0) + 1

    point_parts: list[np.ndarray] = []
    radius_parts: list[np.ndarray] = []
    target_parts: list[np.ndarray] = []
    edge_index_parts: list[np.ndarray] = []
    edge_id_parts: list[np.ndarray] = []
    for index, edge in enumerate(edges):
        points, radii, along = _resample_polyline(
            np.asarray(edge["points_mm"], dtype=float),
            np.asarray(edge["radii_mm"], dtype=float),
            config.graph_sample_spacing_mm,
        )
        radii = np.maximum(radii, config.min_edge_mm * 0.5)
        # Circumferential resolution of a locally cylindrical vessel.
        target = 2.0 * math.pi * radii / config.circumferential_segments

        # Chord-error control at tight centreline bends.  For a circle,
        # sagitta ~= h^2/(8R); solving for h limits axial faceting error.
        curve = _curvature(points)
        bend_radius = np.divide(
            1.0,
            curve,
            out=np.full_like(curve, np.inf),
            where=curve > 1.0e-10,
        )
        allowed_error = np.minimum(
            config.absolute_error_mm, config.relative_error * radii
        )
        curvature_target = np.sqrt(8.0 * bend_radius * allowed_error)
        target = np.minimum(target, curvature_target)

        # Keep junction/carina geometry finer than a straight vessel of the
        # same radius.  True adjacent branches are deliberately allowed to
        # overlap; only non-adjacent branches constrain clearance below.
        total = float(along[-1])
        near_start = along <= config.bifurcation_length_radii * radii
        near_end = (total - along) <= config.bifurcation_length_radii * radii
        if node_degree[int(edge["node1"])] > 2:
            target[near_start] *= config.bifurcation_factor
        if node_degree[int(edge["node2"])] > 2:
            target[near_end] *= config.bifurcation_factor

        point_parts.append(points)
        radius_parts.append(radii)
        target_parts.append(target)
        edge_index_parts.append(np.full(len(points), index, dtype=np.int32))
        edge_id_parts.append(np.full(len(points), int(edge["edge_id"]), dtype=np.int64))

    points = np.vstack(point_parts)
    radii = np.concatenate(radius_parts)
    target = np.concatenate(target_parts)
    edge_indices = np.concatenate(edge_index_parts)
    edge_ids = np.concatenate(edge_id_parts)
    tree = cKDTree(points)

    # Clearance to a non-topologically-adjacent branch.  The centreline gap is
    # reduced by both wall radii.  A small gap forces a finer surface locally;
    # it never permits simplification to bridge two nearby branches.
    adjacent = _edge_adjacency(edges)
    k = min(max(2, config.clearance_candidates), len(points))
    distances, neighbours = tree.query(points, k=k)
    if k == 1:
        distances = distances[:, None]
        neighbours = neighbours[:, None]
    gap = np.full(len(points), np.inf, dtype=float)
    for row in range(len(points)):
        own = int(edge_indices[row])
        forbidden = adjacent[own]
        for distance, neighbour in zip(distances[row, 1:], neighbours[row, 1:]):
            neighbour = int(neighbour)
            if int(edge_indices[neighbour]) in forbidden:
                continue
            gap[row] = float(distance) - radii[row] - radii[neighbour]
            break
    finite_gap = np.isfinite(gap)
    clearance_target = np.maximum(
        config.min_edge_mm, config.clearance_factor * np.maximum(gap, 0.0)
    )
    target[finite_gap] = np.minimum(target[finite_gap], clearance_target[finite_gap])
    target = np.clip(target, config.min_edge_mm, config.max_edge_mm)
    return GraphSizeField(
        points=points,
        radii=radii,
        target_edges=target,
        edge_indices=edge_indices,
        edge_ids=edge_ids,
        nearest_nonadjacent_gap=gap,
        tree=tree,
    )


def associate_surface_points(
    points: np.ndarray,
    field: GraphSizeField,
    *,
    candidates: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Map surface points to the anatomically most plausible graph sample.

    Euclidean-nearest centreline assignment fails when two branches nearly
    touch.  Among nearby candidates, a vessel's own centreline is instead the
    one whose centre distance best matches its local radius.
    """
    points = np.asarray(points, dtype=float)
    k = min(max(1, int(candidates)), len(field.points))
    distances, indices = field.tree.query(points, k=k)
    if k == 1:
        distances = distances[:, None]
        indices = indices[:, None]
    candidate_radii = field.radii[indices]
    normalised_wall_error = np.abs(distances - candidate_radii) / np.maximum(
        candidate_radii, 1.0e-9
    )
    # A weak distance term breaks ties toward the local rather than a remote
    # large-radius trunk with a similar absolute wall residual.
    score = normalised_wall_error + 0.02 * distances / np.maximum(
        candidate_radii, 1.0e-9
    )
    column = np.argmin(score, axis=1)
    chosen = indices[np.arange(len(points)), column]
    return (
        field.target_edges[chosen],
        field.radii[chosen],
        field.edge_ids[chosen],
    )


def _mesh_arrays(mesh: Any, mrmeshnumpy: Any) -> tuple[np.ndarray, np.ndarray]:
    try:
        mesh.pack()
    except Exception:
        pass
    vertices = np.asarray(mrmeshnumpy.getNumpyVerts(mesh), dtype=np.float64)
    faces = np.asarray(mrmeshnumpy.getNumpyFaces(mesh.topology), dtype=np.int64)
    return vertices, faces


def _face_edge_lengths(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    triangles = vertices[faces]
    return np.column_stack(
        (
            np.linalg.norm(triangles[:, 1] - triangles[:, 0], axis=1),
            np.linalg.norm(triangles[:, 2] - triangles[:, 1], axis=1),
            np.linalg.norm(triangles[:, 0] - triangles[:, 2], axis=1),
        )
    )


def _coincident_vertex_groups(vertices: np.ndarray) -> list[np.ndarray]:
    """Return exact-coordinate vertex groups that STL readers would weld.

    MeshLib can represent two closed surface fans that meet at a coordinate as
    distinct vertices.  Binary STL does not preserve vertex identity, so
    Simpleware welds such a pair and correctly reports a non-manifold node.
    """

    _unique, inverse, counts = np.unique(
        vertices, axis=0, return_inverse=True, return_counts=True
    )
    return [
        np.flatnonzero(inverse == group)
        for group in np.flatnonzero(counts > 1)
    ]


def _separate_coincident_vertices(
    mesh: Any,
    mrmeshnumpy: Any,
    mrmeshpy: Any,
    *,
    displacement_mm: float = 1.0e-4,
) -> tuple[Any, dict[str, Any]]:
    """Separate coincident, topologically distinct vertices without collision.

    The displacement is 0.1 micrometre by default, far below the remeshing
    tolerance.  Both local-normal directions are tested and the direction with
    the fewest true triangle intersections is retained.  A candidate that
    increases the collision count is rejected.
    """

    vertices, faces = _mesh_arrays(mesh, mrmeshnumpy)
    groups = _coincident_vertex_groups(vertices)
    if not groups:
        return mesh, {
            "groups_before": 0,
            "vertices_moved": 0,
            "displacement_mm": displacement_mm,
            "collision_pairs_before": 0,
            "collision_pairs_after": 0,
        }

    baseline = int(
        len(
            mrmeshpy.findSelfCollidingTriangles(
                mrmeshpy.MeshPart(mesh), touchIsIntersection=False
            )
        )
    )
    collisions_before = baseline
    moved = 0
    for group in groups:
        # Keep one topological fan fixed and move each other fan independently.
        for vertex_id in group[1:]:
            incident = np.flatnonzero(np.any(faces == vertex_id, axis=1))
            triangles = vertices[faces[incident]]
            normal = np.cross(
                triangles[:, 1] - triangles[:, 0],
                triangles[:, 2] - triangles[:, 0],
            ).sum(axis=0)
            magnitude = float(np.linalg.norm(normal))
            if magnitude <= 1.0e-15:
                raise RuntimeError(
                    f"cannot separate coincident vertex {int(vertex_id)}: "
                    "its incident-face normal is degenerate"
                )
            normal /= magnitude

            best: tuple[int, Any, np.ndarray] | None = None
            for direction in (-1.0, 1.0):
                trial_vertices = vertices.copy()
                trial_vertices[vertex_id] += direction * displacement_mm * normal
                trial = mrmeshnumpy.meshFromFacesVerts(
                    np.ascontiguousarray(faces, dtype=np.int32),
                    np.ascontiguousarray(trial_vertices, dtype=np.float64),
                    duplicateNonManifoldVertices=False,
                )
                collisions = int(
                    len(
                        mrmeshpy.findSelfCollidingTriangles(
                            mrmeshpy.MeshPart(trial), touchIsIntersection=False
                        )
                    )
                )
                if best is None or collisions < best[0]:
                    best = (collisions, trial, trial_vertices)
            assert best is not None
            if best[0] > baseline:
                raise RuntimeError(
                    f"separating coincident vertex {int(vertex_id)} would "
                    f"increase self-intersections from {baseline} to {best[0]}"
                )
            baseline, mesh, vertices = best
            moved += 1

    remaining = _coincident_vertex_groups(vertices)
    if remaining:
        raise RuntimeError(
            f"{len(remaining)} exact-coordinate vertex group(s) remain after "
            "non-manifold-node repair"
        )
    return mesh, {
        "groups_before": len(groups),
        "vertices_moved": moved,
        "displacement_mm": displacement_mm,
        "collision_pairs_before": collisions_before,
        "collision_pairs_after": baseline,
    }


def radius_aware_decimate(
    source: str | Path,
    output: str | Path,
    field: GraphSizeField,
    config: SurfaceRemeshConfig,
    *,
    started: float,
) -> dict[str, Any]:
    from meshlib import mrmeshnumpy, mrmeshpy

    source = Path(source).expanduser().resolve()
    output = Path(output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    _log(f"Loading source STL with MeshLib: {source}", started)
    mesh = mrmeshpy.loadMesh(source)
    vertices, faces = _mesh_arrays(mesh, mrmeshnumpy)
    initial_vertices = int(len(vertices))
    initial_faces = int(len(faces))
    _log(
        f"Loaded {len(vertices):,} vertices and {initial_faces:,} triangles.",
        started,
    )

    bins = np.geomspace(
        config.min_edge_mm, config.max_edge_mm, config.number_of_size_bins + 1
    )
    passes: list[dict[str, Any]] = []
    for bin_index in range(config.number_of_size_bins - 1, -1, -1):
        vertices, faces = _mesh_arrays(mesh, mrmeshnumpy)
        centres = vertices[faces].mean(axis=1)
        target, local_radius, _edge_ids = associate_surface_points(
            centres, field, candidates=config.association_candidates
        )
        assigned = np.clip(np.searchsorted(bins, target, side="right") - 1, 0, len(bins) - 2)
        region = assigned == bin_index
        region_count = int(np.count_nonzero(region))
        if region_count < 20:
            continue
        target_edge = float(math.sqrt(bins[bin_index] * bins[bin_index + 1]))
        representative_radius = float(np.percentile(local_radius[region], 10.0))
        max_error = min(
            config.absolute_error_mm,
            config.relative_error * representative_radius,
        )
        settings = mrmeshpy.DecimateSettings()
        settings.region = mrmeshnumpy.faceBitSetFromBools(
            np.ascontiguousarray(region, dtype=bool)
        )
        settings.maxEdgeLen = target_edge
        settings.maxError = max_error
        settings.maxDeletedFaces = region_count
        settings.maxDeletedVertices = region_count
        settings.maxAngleChange = math.radians(config.max_angle_change_deg)
        settings.maxTriangleAspectRatio = config.maximum_triangle_aspect_ratio
        settings.touchBdVerts = False
        settings.touchNearBdEdges = False
        settings.packMesh = False
        before = int(len(faces))
        _log(
            f"Decimation bin {bin_index + 1}/{config.number_of_size_bins}: "
            f"{region_count:,} faces, target edge {target_edge:.4f} mm, "
            f"max error {max_error:.4f} mm.",
            started,
        )
        result = mrmeshpy.decimateMesh(mesh, settings)
        _vertices_after, faces_after = _mesh_arrays(mesh, mrmeshnumpy)
        after = int(len(faces_after))
        passes.append(
            {
                "bin_index": bin_index,
                "region_faces_before": region_count,
                "mesh_faces_before": before,
                "mesh_faces_after": after,
                "target_edge_mm": target_edge,
                "max_error_mm": max_error,
                "deleted_faces_reported": int(getattr(result, "facesDeleted", before - after)),
            }
        )
        _log(f"Triangle count after bin: {after:,}.", started)

    # Regional decimation can very occasionally let triangles from two close
    # vessel walls cross even though connectivity remains manifold.  Repair
    # only the detected local patches by relaxation, then let the independent
    # validation below reject any residual collision or excessive displacement.
    collisions_before = int(
        len(
            mrmeshpy.findSelfCollidingTriangles(
                mrmeshpy.MeshPart(mesh), touchIsIntersection=False
            )
        )
    )
    collisions_after = collisions_before
    if collisions_before:
        _log(
            f"Locally relaxing {collisions_before} self-intersecting triangle "
            "pair(s) introduced by simplification.",
            started,
        )
        repair = mrmeshpy.SelfIntersections.Settings()
        repair.method = mrmeshpy.SelfIntersections.Settings.Method.Relax
        repair.touchIsIntersection = False
        repair.relaxIterations = 8
        repair.maxExpand = 4
        mrmeshpy.localFixSelfIntersections(mesh, repair)
        collisions_after = int(
            len(
                mrmeshpy.findSelfCollidingTriangles(
                    mrmeshpy.MeshPart(mesh), touchIsIntersection=False
                )
            )
        )
        _log(
            f"Local collision repair left {collisions_after} intersecting "
            "pair(s).",
            started,
        )

    mesh, coincident_repair = _separate_coincident_vertices(
        mesh, mrmeshnumpy, mrmeshpy
    )
    if coincident_repair["vertices_moved"]:
        _log(
            "Separated "
            f"{coincident_repair['vertices_moved']} coincident topological "
            "vertex/vertices by 0.1 micrometre for STL import safety.",
            started,
        )

    vertices, faces = _mesh_arrays(mesh, mrmeshnumpy)
    _log(f"Saving disposable remeshed STL: {output}", started)
    mrmeshpy.saveMesh(mesh, output)

    centres = vertices[faces].mean(axis=1)
    target, local_radius, edge_ids = associate_surface_points(
        centres, field, candidates=config.association_candidates
    )
    lengths = _face_edge_lengths(vertices, faces)
    longest = lengths.max(axis=1)
    return {
        "source": str(source),
        "output": str(output),
        "initial_vertices": initial_vertices,
        "initial_triangles": initial_faces,
        "output_vertices": int(len(vertices)),
        "output_triangles": int(len(faces)),
        "triangle_reduction_fraction": 1.0 - float(len(faces)) / initial_faces,
        "edge_target_ratio_median": float(np.median(longest / target)),
        "edge_target_ratio_p95": float(np.percentile(longest / target, 95)),
        "edge_target_ratio_max": float(np.max(longest / target)),
        "local_radius_mm_min": float(np.min(local_radius)),
        "local_radius_mm_max": float(np.max(local_radius)),
        "associated_edge_count": int(len(np.unique(edge_ids))),
        "self_intersection_repair": {
            "pairs_before": collisions_before,
            "pairs_after": collisions_after,
            "method": "local_relax" if collisions_before else "not_required",
        },
        "coincident_vertex_repair": coincident_repair,
        "passes": passes,
    }


def _read_surface(path: str | Path):
    import pyvista as pv

    surface = pv.read(str(path)).triangulate().clean()
    if surface.n_cells == 0:
        raise RuntimeError(f"empty surface: {path}")
    return surface


def _topology_summary(surface: Any) -> dict[str, Any]:
    edges = surface.extract_feature_edges(
        boundary_edges=True,
        non_manifold_edges=True,
        feature_edges=False,
        manifold_edges=False,
    )
    connectivity = surface.connectivity()
    regions = np.asarray(connectivity.cell_data.get("RegionId", []), dtype=int)
    return {
        "points": int(surface.n_points),
        "triangles": int(surface.n_cells),
        "boundary_or_nonmanifold_edges": int(edges.n_cells),
        "connected_components": int(len(np.unique(regions))) if len(regions) else 0,
        "bounds": [float(value) for value in surface.bounds],
    }


def _implicit_distances(points: np.ndarray, surface: Any) -> np.ndarray:
    import pyvista as pv

    cloud = pv.PolyData(np.asarray(points, dtype=float))
    measured = cloud.compute_implicit_distance(surface, inplace=False)
    return np.abs(np.asarray(measured.point_data["implicit_distance"], dtype=float))


def validate_remesh(
    source: str | Path,
    output: str | Path,
    *,
    field: GraphSizeField | None = None,
    maximum_distance_samples: int = 100_000,
    check_self_intersections: bool = True,
    maximum_p95_surface_error_mm: float = 0.012,
    maximum_surface_error_mm: float = 0.050,
    maximum_p95_radius_error_mm: float = 0.010,
    started: float,
) -> dict[str, Any]:
    from meshlib import mrmeshnumpy, mrmeshpy

    _log("Loading source and remeshed surfaces for validation.", started)
    source_surface = _read_surface(source)
    output_surface = _read_surface(output)
    source_topology = _topology_summary(source_surface)
    output_topology = _topology_summary(output_surface)

    rng = np.random.default_rng(20260830)
    source_ids = np.arange(source_surface.n_points)
    output_ids = np.arange(output_surface.n_points)
    if len(source_ids) > maximum_distance_samples:
        source_ids = rng.choice(source_ids, maximum_distance_samples, replace=False)
    if len(output_ids) > maximum_distance_samples:
        output_ids = rng.choice(output_ids, maximum_distance_samples, replace=False)
    _log(
        f"Measuring symmetric surface error using {len(source_ids):,} source "
        f"and {len(output_ids):,} output samples.",
        started,
    )
    source_to_output = _implicit_distances(source_surface.points[source_ids], output_surface)
    output_to_source = _implicit_distances(output_surface.points[output_ids], source_surface)
    symmetric = np.r_[source_to_output, output_to_source]

    radius_preservation: dict[str, Any] | None = None
    if field is not None:
        _log(
            f"Checking minimum wall-radius preservation at {len(field.points):,} "
            "cropped-Amira samples.",
            started,
        )
        source_radius = _implicit_distances(field.points, source_surface)
        output_radius = _implicit_distances(field.points, output_surface)
        radius_error = np.abs(output_radius - source_radius)
        valid_radius = source_radius > 0.05
        relative_error = radius_error[valid_radius] / source_radius[valid_radius]
        if not len(relative_error):
            relative_error = np.asarray([math.inf], dtype=float)
        radius_preservation = {
            "sample_count": int(len(radius_error)),
            "absolute_error_mm_median": float(np.median(radius_error)),
            "absolute_error_mm_p95": float(np.percentile(radius_error, 95)),
            "absolute_error_mm_maximum": float(np.max(radius_error)),
            "relative_error_median": float(np.median(relative_error)),
            "relative_error_p95": float(np.percentile(relative_error, 95)),
        }

    _log("Checking raw STL topology before reader vertex welding.", started)
    raw_mesh = mrmeshpy.loadMesh(Path(output).resolve())
    raw_vertices, _raw_faces = _mesh_arrays(raw_mesh, mrmeshnumpy)
    coincident_groups = _coincident_vertex_groups(raw_vertices)
    raw_topology = {
        "vertices": int(raw_mesh.topology.numValidVerts()),
        "faces": int(raw_mesh.topology.numValidFaces()),
        "holes": int(raw_mesh.topology.findNumHoles()),
        "has_multiple_edges": bool(mrmeshpy.hasMultipleEdges(raw_mesh.topology)),
        "coincident_vertex_groups": int(len(coincident_groups)),
        "coincident_vertex_excess": int(
            sum(max(0, len(group) - 1) for group in coincident_groups)
        ),
    }

    self_intersections: int | None = None
    if check_self_intersections:
        _log("Checking remeshed STL for self-intersecting triangle pairs.", started)
        pairs = mrmeshpy.findSelfCollidingTriangles(
            mrmeshpy.MeshPart(raw_mesh), touchIsIntersection=False
        )
        self_intersections = int(len(pairs))

    source_bounds = np.asarray(source_topology["bounds"], dtype=float)
    output_bounds = np.asarray(output_topology["bounds"], dtype=float)
    report = {
        "source_topology": source_topology,
        "output_topology": output_topology,
        "bounds_max_abs_error_mm": float(np.max(np.abs(source_bounds - output_bounds))),
        "surface_distance_mm": {
            "median": float(np.median(symmetric)),
            "p95": float(np.percentile(symmetric, 95)),
            "p99": float(np.percentile(symmetric, 99)),
            "maximum": float(np.max(symmetric)),
            "sample_count": int(len(symmetric)),
        },
        "minimum_wall_radius_preservation": radius_preservation,
        "raw_stl_topology": raw_topology,
        "self_intersections": self_intersections,
    }
    failures: list[str] = []
    if output_topology["boundary_or_nonmanifold_edges"] != source_topology[
        "boundary_or_nonmanifold_edges"
    ]:
        failures.append("boundary/non-manifold edge count changed")
    if output_topology["connected_components"] != source_topology["connected_components"]:
        failures.append("connected-component count changed")
    if self_intersections:
        failures.append(f"output has {self_intersections} self-intersecting pairs")
    if raw_topology["holes"]:
        failures.append(f"raw output STL has {raw_topology['holes']} hole(s)")
    if raw_topology["has_multiple_edges"]:
        failures.append("raw output STL has multiple topological edges")
    if raw_topology["coincident_vertex_groups"]:
        failures.append(
            "raw output STL has "
            f"{raw_topology['coincident_vertex_groups']} coincident, "
            "topologically distinct vertex group(s)"
        )
    if float(np.percentile(symmetric, 95)) > maximum_p95_surface_error_mm:
        failures.append(
            "P95 surface error exceeds "
            f"{maximum_p95_surface_error_mm:.3f} mm"
        )
    if float(np.max(symmetric)) > maximum_surface_error_mm:
        failures.append(
            f"maximum surface error exceeds {maximum_surface_error_mm:.3f} mm"
        )
    if (
        radius_preservation is not None
        and radius_preservation["absolute_error_mm_p95"]
        > maximum_p95_radius_error_mm
    ):
        failures.append(
            "P95 minimum-wall-radius error exceeds "
            f"{maximum_p95_radius_error_mm:.3f} mm"
        )
    report["failures"] = failures
    report["topology_passed"] = not failures
    return report


def default_output_path(source: Path, output_dir: Path) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return output_dir / f"{source.stem}.radius_adaptive_{stamp}.stl"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-stl", required=True, type=Path)
    parser.add_argument("--cropped-graph-json", required=True, type=Path)
    parser.add_argument("--output-stl", type=Path)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("mesh_sensitivity_output/surface_remesh_benchmark"),
    )
    parser.add_argument("--circumferential-segments", type=int, default=18)
    parser.add_argument("--min-edge-mm", type=float, default=0.025)
    parser.add_argument("--max-edge-mm", type=float, default=0.20)
    parser.add_argument("--sample-spacing-mm", type=float, default=0.06)
    parser.add_argument("--distance-samples", type=int, default=100_000)
    parser.add_argument("--skip-self-intersection-check", action="store_true")
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    started = time.perf_counter()
    source = args.source_stl.expanduser().resolve()
    graph_path = args.cropped_graph_json.expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    if not graph_path.is_file():
        raise FileNotFoundError(graph_path)
    output_dir = args.output_dir.expanduser().resolve()
    output = (
        args.output_stl.expanduser().resolve()
        if args.output_stl
        else default_output_path(source, output_dir)
    )
    if output == source:
        raise RuntimeError("refusing to overwrite the validated source STL")
    output.parent.mkdir(parents=True, exist_ok=True)

    config = SurfaceRemeshConfig(
        circumferential_segments=args.circumferential_segments,
        min_edge_mm=args.min_edge_mm,
        max_edge_mm=args.max_edge_mm,
        graph_sample_spacing_mm=args.sample_spacing_mm,
    )
    _log(f"Reading cached cropped Amira graph: {graph_path}", started)
    graph = load_cropped_graph(graph_path)
    field = build_graph_size_field(graph, config)
    finite_gap = field.nearest_nonadjacent_gap[np.isfinite(field.nearest_nonadjacent_gap)]
    _log(
        f"Built {len(field.points):,}-sample size field over {len(graph['edges'])} "
        f"retained edges; target edge {field.target_edges.min():.4f}-"
        f"{field.target_edges.max():.4f} mm.",
        started,
    )
    decimation = radius_aware_decimate(
        source, output, field, config, started=started
    )
    validation = validate_remesh(
        source,
        output,
        field=field,
        maximum_distance_samples=args.distance_samples,
        check_self_intersections=not args.skip_self_intersection_check,
        started=started,
    )
    report = {
        "created_at": datetime.now().isoformat(),
        "elapsed_seconds": time.perf_counter() - started,
        "config": asdict(config),
        "cropped_graph_json": str(graph_path),
        "graph": {
            "edge_count": len(graph["edges"]),
            "sample_count": len(field.points),
            "radius_mm_min": float(np.min(field.radii)),
            "radius_mm_median": float(np.median(field.radii)),
            "radius_mm_max": float(np.max(field.radii)),
            "target_edge_mm_min": float(np.min(field.target_edges)),
            "target_edge_mm_median": float(np.median(field.target_edges)),
            "target_edge_mm_max": float(np.max(field.target_edges)),
            "nonadjacent_gap_mm_min": (
                float(np.min(finite_gap)) if len(finite_gap) else None
            ),
        },
        "decimation": decimation,
        "validation": validation,
        "accepted_for_simpleware_benchmark": bool(validation["topology_passed"]),
        "note": (
            "This is a disposable benchmark STL. Do not update the study manifest "
            "until Simpleware timing and anatomical review pass."
        ),
    }
    report_path = output.with_suffix(".report.json")
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    _log(f"Wrote report: {report_path}", started)
    _log(
        f"Result: {decimation['output_triangles']:,} triangles; topology "
        f"{'PASS' if validation['topology_passed'] else 'FAIL'}; "
        f"P95 surface error {validation['surface_distance_mm']['p95']:.4f} mm.",
        started,
    )
    return 0 if validation["topology_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
