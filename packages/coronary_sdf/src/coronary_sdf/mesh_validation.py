"""Backend-independent validation contract for CFD surface meshes."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class MeshValidation:
    finite: bool
    triangle_count: int
    vertex_count: int
    boundary_edges: int
    nonmanifold_edges: int
    vertex_manifold: bool
    orientable: bool
    connected_components: int
    expected_components: int | None
    expected_genus: float | None
    degenerate_faces: int
    self_intersections: int | None
    euler_characteristic: int | None
    genus: float | None
    errors: tuple[str, ...]
    # Face counts of every face-connected component, largest first. Added after
    # the original contract, so it carries a default and stays optional for
    # callers constructing a report directly.
    component_face_counts: tuple[int, ...] = ()

    @property
    def closed(self) -> bool:
        return self.boundary_edges == 0

    @property
    def largest_component_face_fraction(self) -> float:
        """Fraction of triangles in the dominant component (1.0 when clean)."""

        if not self.component_face_counts or not self.triangle_count:
            return 0.0
        return float(self.component_face_counts[0]) / float(self.triangle_count)

    @property
    def edge_manifold(self) -> bool:
        return self.nonmanifold_edges == 0

    @property
    def valid(self) -> bool:
        return not self.errors

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["closed"] = self.closed
        result["edge_manifold"] = self.edge_manifold
        result["valid"] = self.valid
        result["largest_component_face_fraction"] = self.largest_component_face_fraction
        return result


def _triangles(surface) -> tuple[np.ndarray, np.ndarray]:
    tri = surface.triangulate()
    vertices = np.asarray(tri.points, dtype=np.float64)
    packed = np.asarray(tri.faces, dtype=np.int64)
    if len(packed) == 0:
        return vertices, np.empty((0, 3), dtype=np.int64)
    cells = packed.reshape(-1, 4)
    if np.any(cells[:, 0] != 3):
        raise ValueError("surface could not be represented as triangles")
    return vertices, cells[:, 1:].copy()


def _edge_table(faces: np.ndarray):
    if len(faces) == 0:
        return np.empty((0, 2), dtype=np.int64), np.empty(0, dtype=np.int64), []
    directed = np.vstack(
        (faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]])
    )
    face_ids = np.tile(np.arange(len(faces), dtype=np.int64), 3)
    ordered = np.sort(directed, axis=1)
    unique, inverse, counts = np.unique(
        ordered, axis=0, return_inverse=True, return_counts=True
    )
    incidence: list[list[tuple[int, int]]] = [[] for _ in range(len(unique))]
    direction = np.where(directed[:, 0] == ordered[:, 0], 1, -1)
    for row, edge_id in enumerate(inverse):
        incidence[int(edge_id)].append((int(face_ids[row]), int(direction[row])))
    return unique, counts, incidence


def _orientable_and_components(
    n_faces: int, incidence: list[list[tuple[int, int]]]
) -> tuple[bool, int, np.ndarray]:
    """Return orientability, the component count, and a per-face component label.

    Components are *face*-connected through edges with exactly two incident
    faces, so isolated contour speckles are counted as their own components.
    """

    adjacency: list[list[tuple[int, int]]] = [[] for _ in range(n_faces)]
    for entries in incidence:
        if len(entries) != 2:
            continue
        (a, da), (b, db) = entries
        # sign[b] = relation * sign[a] makes the shared directed edges opposite.
        relation = -da * db
        adjacency[a].append((b, relation))
        adjacency[b].append((a, relation))
    signs = np.zeros(n_faces, dtype=np.int8)
    labels = np.full(n_faces, -1, dtype=np.int64)
    components = 0
    orientable = True
    for seed in range(n_faces):
        if signs[seed]:
            continue
        label = components
        components += 1
        signs[seed] = 1
        labels[seed] = label
        stack = [seed]
        while stack:
            current = stack.pop()
            for neighbour, relation in adjacency[current]:
                wanted = int(signs[current]) * relation
                if signs[neighbour] == 0:
                    signs[neighbour] = wanted
                    labels[neighbour] = label
                    stack.append(neighbour)
                elif int(signs[neighbour]) != wanted:
                    orientable = False
    return orientable, components, labels


def face_components(surface) -> np.ndarray:
    """Return the face-connected component label of every triangle.

    Exposed so artefact reporting can size components without duplicating the
    traversal in :func:`validate_mesh`.
    """

    _vertices, faces = _triangles(surface)
    if not len(faces):
        return np.empty(0, dtype=np.int64)
    _unique, _counts, incidence = _edge_table(faces)
    _orientable, _components, labels = _orientable_and_components(len(faces), incidence)
    return labels


def _vertex_manifold(faces: np.ndarray, n_vertices: int) -> bool:
    incident: list[list[tuple[int, int]]] = [[] for _ in range(n_vertices)]
    for a_raw, b_raw, c_raw in faces:
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
        degrees = [len(neighbours) for neighbours in adjacency.values()]
        # A closed link is one cycle; a boundary link is one path. Boundary
        # status is reported separately, but both are locally manifold.
        if any(degree > 2 or degree == 0 for degree in degrees):
            return False
        endpoints = sum(degree == 1 for degree in degrees)
        if endpoints not in (0, 2):
            return False
        seed = next(iter(adjacency))
        visited = {seed}
        stack = [seed]
        while stack:
            current = stack.pop()
            for neighbour in adjacency[current]:
                if neighbour not in visited:
                    visited.add(neighbour)
                    stack.append(neighbour)
        if len(visited) != len(adjacency):
            return False
    return True


def _self_intersection_count(vertices: np.ndarray, faces: np.ndarray) -> int:
    import meshlib.mrmeshnumpy as mrnumpy
    import meshlib.mrmeshpy as mrmesh

    mesh = mrnumpy.meshFromFacesVerts(
        np.ascontiguousarray(faces, dtype=np.int32),
        np.ascontiguousarray(vertices, dtype=np.float64),
    )
    pairs = mrmesh.findSelfCollidingTriangles(
        mrmesh.MeshPart(mesh), touchIsIntersection=False
    )
    return int(len(pairs))


def validate_mesh(
    surface,
    *,
    expected_components: int | None = 1,
    expected_genus: float | None = 0.0,
    check_self_intersections: bool = True,
) -> MeshValidation:
    """Return a complete validation report without modifying ``surface``."""

    vertices, faces = _triangles(surface)
    finite = bool(np.isfinite(vertices).all())
    unique_edges, counts, incidence = _edge_table(faces)
    boundary = int(np.count_nonzero(counts == 1))
    nonmanifold = int(np.count_nonzero(counts > 2))
    orientable, components, labels = _orientable_and_components(len(faces), incidence)
    vertex_manifold = _vertex_manifold(faces, len(vertices)) if len(faces) else False
    if components:
        face_counts = tuple(
            sorted(
                (int(count) for count in np.bincount(labels, minlength=components)),
                reverse=True,
            )
        )
    else:
        face_counts = ()

    if len(faces) and finite:
        triangles = vertices[faces]
        double_area = np.linalg.norm(
            np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]),
            axis=1,
        )
        scale = max(float(np.ptp(vertices, axis=0).max()), 1.0)
        degenerate = int(np.count_nonzero(double_area <= (scale * 1e-12) ** 2))
    else:
        degenerate = int(len(faces))

    self_intersections: int | None = None
    self_error: str | None = None
    if check_self_intersections and len(faces) and finite:
        try:
            self_intersections = _self_intersection_count(vertices, faces)
        except Exception as exc:  # validation must report unavailable checks
            self_error = f"self-intersection check failed: {exc}"

    euler: int | None = None
    genus: float | None = None
    if len(faces):
        used_vertices = int(len(np.unique(faces)))
        euler = used_vertices - len(unique_edges) + len(faces)
        if boundary == 0 and nonmanifold == 0 and orientable:
            genus = (2.0 * components - float(euler)) / 2.0

    errors: list[str] = []
    if not finite:
        errors.append("mesh contains non-finite vertices")
    if len(vertices) == 0 or len(faces) == 0:
        errors.append("mesh is empty")
    if boundary:
        errors.append(f"mesh has {boundary} boundary edges")
    if nonmanifold:
        errors.append(f"mesh has {nonmanifold} non-manifold edges")
    if not vertex_manifold:
        errors.append("mesh is not vertex-manifold")
    if not orientable:
        errors.append("mesh is not orientable")
    if expected_components is not None and components != expected_components:
        errors.append(
            f"mesh has {components} components; expected {expected_components}"
        )
    if degenerate:
        errors.append(f"mesh has {degenerate} degenerate faces")
    if (
        expected_genus is not None
        and genus is not None
        and not np.isclose(genus, expected_genus, atol=1e-12, rtol=0.0)
    ):
        errors.append(f"mesh has genus {genus:g}; expected {expected_genus:g}")
    if self_intersections:
        errors.append(f"mesh has {self_intersections} self-intersecting triangle pairs")
    if self_error:
        errors.append(self_error)

    return MeshValidation(
        finite=finite,
        triangle_count=int(len(faces)),
        vertex_count=int(len(vertices)),
        boundary_edges=boundary,
        nonmanifold_edges=nonmanifold,
        vertex_manifold=vertex_manifold,
        orientable=orientable,
        connected_components=components,
        expected_components=expected_components,
        expected_genus=expected_genus,
        degenerate_faces=degenerate,
        self_intersections=self_intersections,
        euler_characteristic=euler,
        genus=genus,
        errors=tuple(errors),
        component_face_counts=face_counts,
    )


def enforce_mesh_validation(
    surface,
    *,
    mode: str,
    expected_components: int | None = 1,
    expected_genus: float | None = 0.0,
    check_self_intersections: bool = True,
) -> MeshValidation | None:
    """Validate according to ``off|warn|error`` and return the report."""

    if mode == "off":
        return None
    if mode not in {"warn", "error"}:
        raise ValueError("validation mode must be 'off', 'warn', or 'error'")
    report = validate_mesh(
        surface,
        expected_components=expected_components,
        expected_genus=expected_genus,
        check_self_intersections=check_self_intersections,
    )
    if report.errors:
        message = "mesh validation failed: " + "; ".join(report.errors)
        if mode == "error":
            raise RuntimeError(message)
        print(f"  [VALIDATION][WARN] {message}")
    else:
        print(
            "  [VALIDATION] passed: "
            f"components={report.connected_components}, genus={report.genus}"
        )
    return report


__all__ = [
    "MeshValidation",
    "enforce_mesh_validation",
    "face_components",
    "validate_mesh",
]
