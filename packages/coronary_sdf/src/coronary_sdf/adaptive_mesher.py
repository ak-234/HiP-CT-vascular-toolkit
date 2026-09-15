"""Conforming reference mesher for a radius-adaptive implicit hierarchy.

This backend tetrahedralizes the sparse octree samples with SciPy Delaunay and
extracts the analytic zero set by marching tetrahedra.  All tetrahedra share a
single edge-root cache, so there are no coarse/fine stitching cracks.

It is intended as a correctness/parity backend and for moderate graphs.  Large
production trees should use the same :class:`GraphImplicitField` oracle and
radius sizing function with VTK HyperTreeGrid; global Delaunay construction is not
memory-optimal at that scale.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import numpy as np
from scipy.spatial import Delaunay

from .adaptive_octree import RadiusAdaptiveOctree
from .implicit_field import GraphImplicitField


@dataclass(frozen=True)
class AdaptiveImplicitMesh:
    vertices: np.ndarray
    faces: np.ndarray

    def edge_incidence(self) -> np.ndarray:
        if len(self.faces) == 0:
            return np.empty(0, dtype=np.int64)
        edges = np.sort(
            np.vstack(
                [
                    self.faces[:, [0, 1]],
                    self.faces[:, [1, 2]],
                    self.faces[:, [2, 0]],
                ]
            ),
            axis=1,
        )
        _unique, counts = np.unique(edges, axis=0, return_counts=True)
        return counts

    @property
    def is_watertight(self) -> bool:
        counts = self.edge_incidence()
        return bool(len(counts) > 0 and np.all(counts == 2))

    @property
    def is_edge_manifold(self) -> bool:
        counts = self.edge_incidence()
        return bool(len(counts) > 0 and np.all(counts <= 2))

    @property
    def is_vertex_manifold(self) -> bool:
        """Check that every closed vertex link is one connected cycle."""

        if len(self.faces) == 0:
            return False
        incident: list[list[tuple[int, int]]] = [list() for _ in range(len(self.vertices))]
        for a_raw, b_raw, c_raw in self.faces:
            a, b, c = int(a_raw), int(b_raw), int(c_raw)
            incident[a].append((b, c))
            incident[b].append((c, a))
            incident[c].append((a, b))
        for link_edges in incident:
            if not link_edges:
                continue
            adjacency: dict[int, list[int]] = {}
            for a, b in link_edges:
                adjacency.setdefault(a, []).append(b)
                adjacency.setdefault(b, []).append(a)
            if any(len(neighbours) != 2 for neighbours in adjacency.values()):
                return False
            start = next(iter(adjacency))
            visited = {start}
            stack = [start]
            while stack:
                current = stack.pop()
                for neighbour in adjacency[current]:
                    if neighbour not in visited:
                        visited.add(neighbour)
                        stack.append(neighbour)
            if len(visited) != len(adjacency):
                return False
        return True

    @property
    def is_manifold(self) -> bool:
        return self.is_edge_manifold and self.is_vertex_manifold

    def to_pyvista(self) -> Any:
        """Convert lazily so field/unit tests do not require PyVista."""

        import pyvista as pv

        if len(self.faces) == 0:
            return pv.PolyData()
        packed = np.column_stack(
            [np.full(len(self.faces), 3, dtype=np.int64), self.faces]
        ).reshape(-1)
        return pv.PolyData(self.vertices, packed)


def _unique_octree_points(octree: RadiusAdaptiveOctree) -> np.ndarray:
    if not octree.leaves:
        raise ValueError("adaptive hierarchy has not been built or contains no leaves")
    maximum_depth = max(leaf.depth for leaf in octree.leaves)
    unit = octree.root_size / float(2**maximum_depth)
    lower = octree.root_center - 0.5 * octree.root_size
    integer_points: set[tuple[int, int, int]] = set()
    signs = RadiusAdaptiveOctree._CORNER_SIGNS
    for leaf in octree.leaves:
        corners = leaf.center[None, :] + signs * (0.5 * leaf.size)
        keys = np.rint((corners - lower[None, :]) / unit).astype(np.int64)
        integer_points.update(tuple(int(v) for v in row) for row in keys)
    keys = np.asarray(sorted(integer_points), dtype=np.float64)
    return lower[None, :] + keys * unit


def _deduplicate_points(points: np.ndarray, tolerance: float) -> np.ndarray:
    origin = points.min(axis=0)
    keys = np.rint((points - origin[None, :]) / tolerance).astype(np.int64)
    _unique, first = np.unique(keys, axis=0, return_index=True)
    return points[np.sort(first)]


def _edge_root(
    field: GraphImplicitField,
    a: np.ndarray,
    b: np.ndarray,
    fa: float,
    fb: float,
    relative_tolerance: float,
    maximum_iterations: int,
) -> np.ndarray:
    if (fa < 0.0) == (fb < 0.0):
        raise ValueError("edge root is not bracketed")
    lo = np.asarray(a, dtype=np.float64).copy()
    hi = np.asarray(b, dtype=np.float64).copy()
    flo = float(fa)
    fhi = float(fb)
    edge_length = float(np.linalg.norm(hi - lo))
    tolerance = max(relative_tolerance * edge_length, np.finfo(np.float64).eps * 128)
    for _ in range(maximum_iterations):
        # Safeguarded secant, falling back to the midpoint near equal values.
        denominator = abs(flo) + abs(fhi)
        t = abs(flo) / denominator if denominator > 1e-30 else 0.5
        t = min(max(t, 0.1), 0.9)
        mid = lo * (1.0 - t) + hi * t
        fm = field.sample(mid).value
        if abs(fm) <= tolerance or np.linalg.norm(hi - lo) <= tolerance:
            return mid
        if (fm < 0.0) == (flo < 0.0):
            lo, flo = mid, fm
        else:
            hi, fhi = mid, fm
    return 0.5 * (lo + hi)


def mesh_adaptive_implicit(
    field: GraphImplicitField,
    octree: RadiusAdaptiveOctree | None = None,
    *,
    cells_across_diameter: float = 12.0,
    maximum_points: int = 500_000,
    root_relative_tolerance: float = 1e-7,
    root_maximum_iterations: int = 48,
    require_watertight: bool = True,
) -> AdaptiveImplicitMesh:
    """Generate a conforming triangle mesh of ``field == 0``.

    The returned mesh is validated independently through undirected edge
    incidence. A non-watertight result raises instead of being silently saved.
    """

    if octree is None:
        octree = RadiusAdaptiveOctree(
            field, cells_across_diameter=cells_across_diameter
        )
        octree.build()
    elif not octree.leaves:
        octree.build()

    grid_points = _unique_octree_points(octree)
    # Centreline samples guarantee at least one interior seed per primitive;
    # root-cube corners guarantee an exterior convex hull. They seed a
    # tetrahedralization only and do not define surface connectivity.
    root_corners = (
        octree.root_center[None, :]
        + RadiusAdaptiveOctree._CORNER_SIGNS * (0.5 * octree.root_size)
    )
    centreline_points = np.vstack(
        [field.starts, field.ends, 0.5 * (field.starts + field.ends)]
    )
    all_points = np.vstack([grid_points, centreline_points, root_corners])
    tolerance = max(
        octree.root_size * 1e-12, np.finfo(np.float64).eps * octree.root_size * 128
    )
    points = _deduplicate_points(all_points, tolerance)
    if len(points) > maximum_points:
        raise MemoryError(
            f"reference adaptive mesher requires {len(points):,} Delaunay points, "
            f"exceeding maximum_points={maximum_points:,}; use the vtk_htg backend"
        )

    # Radius-adaptive octree samples contain many axis-aligned/cospherical
    # configurations. Qhull's internal ``QJ`` joggle is not reflected in the
    # coordinates returned to Python, leaving geometrically zero-volume tets
    # when their surface is subsequently reconstructed. Apply a deterministic,
    # scale-relative joggle to the actual coordinates instead.
    minimum_leaf_size = min(leaf.size for leaf in octree.leaves)
    rng = np.random.default_rng(0)
    meshing_points = points + rng.normal(
        scale=minimum_leaf_size * 1e-7, size=points.shape
    )
    tetrahedralization = Delaunay(meshing_points, qhull_options="Qbb Qc Qz Q12")
    values, _owners, _radii, _gradients = field.evaluate(meshing_points)
    simplices = np.asarray(tetrahedralization.simplices, dtype=np.int64)

    surface_vertices: list[np.ndarray] = []
    surface_faces: list[tuple[int, int, int]] = []
    edge_vertices: dict[tuple[int, int], int] = {}

    def crossing_vertex(i: int, j: int) -> int:
        key = (i, j) if i < j else (j, i)
        existing = edge_vertices.get(key)
        if existing is not None:
            return existing
        root = _edge_root(
            field,
            meshing_points[key[0]],
            meshing_points[key[1]],
            float(values[key[0]]),
            float(values[key[1]]),
            root_relative_tolerance,
            root_maximum_iterations,
        )
        vertex_id = len(surface_vertices)
        surface_vertices.append(root)
        edge_vertices[key] = vertex_id
        return vertex_id

    for tet in simplices:
        inside = [int(v) for v in tet if values[v] < 0.0]
        outside = [int(v) for v in tet if values[v] >= 0.0]
        if len(inside) == 0 or len(inside) == 4:
            continue
        if len(inside) == 1:
            ids = [crossing_vertex(inside[0], out) for out in outside]
            surface_faces.append((ids[0], ids[1], ids[2]))
        elif len(inside) == 3:
            ids = [crossing_vertex(outside[0], inn) for inn in inside]
            surface_faces.append((ids[0], ids[2], ids[1]))
        else:
            # Four crossings form a quad. The deterministic diagonal is local
            # to this tetrahedron; all boundary edges are globally shared.
            a, b = inside
            c, d = outside
            q0 = crossing_vertex(a, c)
            q1 = crossing_vertex(a, d)
            q2 = crossing_vertex(b, d)
            q3 = crossing_vertex(b, c)
            surface_faces.append((q0, q1, q2))
            surface_faces.append((q0, q2, q3))

    vertices = np.asarray(surface_vertices, dtype=np.float64)
    faces = np.asarray(surface_faces, dtype=np.int64).reshape(-1, 3)
    if len(faces) == 0:
        raise RuntimeError("adaptive tetrahedralization did not intersect the zero set")

    # Orient each connected triangle locally toward increasing field values.
    for face_index, face in enumerate(faces):
        triangle = vertices[face]
        normal = np.cross(triangle[1] - triangle[0], triangle[2] - triangle[0])
        area2 = float(np.linalg.norm(normal))
        if area2 <= tolerance * tolerance:
            continue
        centroid = triangle.mean(axis=0)
        gradient = field.sample(centroid, with_gradient=True).gradient
        if gradient is not None and float(np.dot(normal, gradient)) < 0.0:
            faces[face_index, 1], faces[face_index, 2] = (
                faces[face_index, 2],
                faces[face_index, 1],
            )

    # Drop numerical zero-area faces before incidence validation.
    tri = vertices[faces]
    area2 = np.linalg.norm(
        np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1
    )
    nondegenerate = area2 > tolerance * tolerance
    if not np.all(nondegenerate):
        faces = faces[nondegenerate]
    mesh = AdaptiveImplicitMesh(vertices=vertices, faces=faces)
    if require_watertight and (not mesh.is_watertight or not mesh.is_manifold):
        counts = mesh.edge_incidence()
        boundary = int(np.count_nonzero(counts == 1))
        nonmanifold = int(np.count_nonzero(counts > 2))
        raise RuntimeError(
            "adaptive implicit extraction failed topology validation: "
            f"{boundary} boundary edges, {nonmanifold} non-manifold edges, "
            f"vertex_manifold={mesh.is_vertex_manifold}"
        )
    return mesh


__all__ = ["AdaptiveImplicitMesh", "mesh_adaptive_implicit"]
