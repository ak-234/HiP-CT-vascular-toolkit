"""Python boundary for the optional native CGAL Mesh_3 extension."""

from __future__ import annotations

import importlib
import math
from typing import Any

import numpy as np
import pyvista as pv

from . import config as config_module
from .implicit_field import GraphImplicitField


NATIVE_API_VERSION = config_module.CGAL_NATIVE_API_VERSION


class CgalMesh3Unavailable(RuntimeError):
    pass


def _extension():
    try:
        native = importlib.import_module("coronary_sdf_cgal")
    except ImportError as exc:
        raise CgalMesh3Unavailable(
            "CGAL Mesh_3 backend requested, but native module "
            "'coronary_sdf_cgal' is not installed. Create the isolated "
            "environment from environment-cgal-win64.yml and install "
            "native/cgal. Visual Studio 2022 Build Tools with the Desktop "
            "development with C++ workload is required on Windows."
        ) from exc
    version = int(getattr(native, "API_VERSION", -1))
    if version != NATIVE_API_VERSION:
        raise CgalMesh3Unavailable(
            f"coronary_sdf_cgal API {version} is incompatible; "
            f"expected {NATIVE_API_VERSION}"
        )
    extension_version = str(getattr(native, "EXTENSION_VERSION", ""))
    if extension_version != config_module.CGAL_EXTENSION_BUILD_VERSION:
        raise CgalMesh3Unavailable(
            f"coronary_sdf_cgal build {extension_version or 'unknown'} is incompatible; "
            f"expected {config_module.CGAL_EXTENSION_BUILD_VERSION}"
        )
    cgal_version = str(getattr(native, "CGAL_VERSION", ""))
    if not cgal_version.startswith(config_module.CGAL_REQUIRED_VERSION):
        raise CgalMesh3Unavailable(
            f"coronary_sdf_cgal uses CGAL {cgal_version or 'unknown'}; "
            f"expected {config_module.CGAL_REQUIRED_VERSION}.x"
        )
    return native


def cgal_mesh3_available() -> bool:
    try:
        _extension()
    except CgalMesh3Unavailable:
        return False
    return True


def cgal_build_info() -> dict[str, Any]:
    native = _extension()
    return {
        "api_version": int(native.API_VERSION),
        "cgal_version": str(native.CGAL_VERSION),
        "extension_version": str(native.EXTENSION_VERSION),
        "compiler": str(native.COMPILER),
        "module": str(native.__file__),
    }


def _junction_arrays(field: GraphImplicitField) -> tuple[np.ndarray, ...]:
    positions = []
    radii = []
    blends = []
    supports = []
    cores = []
    junction_group_offsets = [0]
    group_capsule_offsets = [0]
    group_capsules: list[int] = []
    for junction_index, junction in enumerate(field.junctions):
        positions.append(np.asarray(junction.position, dtype=np.float64))
        radii.append(float(junction.radius))
        blends.append(float(junction.blend_fraction))
        supports.append(float(junction.support_factor))
        cores.append(float(field.junction_core_fractions[junction_index]))
        for group in junction.capsule_groups:
            group_capsules.extend(int(index) for index in group)
            group_capsule_offsets.append(len(group_capsules))
        junction_group_offsets.append(
            junction_group_offsets[-1] + len(junction.capsule_groups)
        )
    return (
        np.asarray(positions, dtype=np.float64).reshape(-1, 3),
        np.asarray(radii, dtype=np.float64),
        np.asarray(blends, dtype=np.float64),
        np.asarray(supports, dtype=np.float64),
        np.asarray(cores, dtype=np.float64),
        np.asarray(junction_group_offsets, dtype=np.int64),
        np.asarray(group_capsule_offsets, dtype=np.int64),
        np.asarray(group_capsules, dtype=np.int64),
    )


def create_native_field(field: GraphImplicitField):
    """Copy one immutable Python graph field into the native oracle."""

    if field.primitive_method != "round_cone":
        raise ValueError("CGAL production backend requires the exact round_cone primitive")
    native = _extension()
    junction_arrays = _junction_arrays(field)
    return native.create_field(
        np.ascontiguousarray(field.starts, dtype=np.float64),
        np.ascontiguousarray(field.ends, dtype=np.float64),
        np.ascontiguousarray(field.radii_start, dtype=np.float64),
        np.ascontiguousarray(field.radii_end, dtype=np.float64),
        np.ascontiguousarray(field.segment_ids, dtype=np.int64),
        np.ascontiguousarray(field.clip_plane_normals, dtype=np.float64),
        np.ascontiguousarray(field.clip_plane_offsets, dtype=np.float64),
        np.ascontiguousarray(field.clip_plane_count, dtype=np.int64),
        *junction_arrays,
        int(field.bvh.leaf_size),
    )


def native_field_values(
    field: GraphImplicitField, points: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    native_field = create_native_field(field)
    values, radii = native_field.evaluate(
        np.ascontiguousarray(points, dtype=np.float64).reshape(-1, 3)
    )
    return np.asarray(values), np.asarray(radii)


def native_sizing_values(
    field: GraphImplicitField, points: np.ndarray, cells_across_diameter: float
) -> np.ndarray:
    native_field = create_native_field(field)
    return np.asarray(
        native_field.sizing(
            np.ascontiguousarray(points, dtype=np.float64).reshape(-1, 3),
            float(cells_across_diameter),
        )
    )


def mesh_cgal_implicit(
    field: GraphImplicitField,
    *,
    cells_across_diameter: float,
    facet_angle_deg: float = 30.0,
    facet_distance_fraction: float = 0.25,
    cell_size_factor: float = 2.0,
    cell_radius_edge_ratio: float = 2.0,
) -> pv.PolyData:
    """Mesh ``field < 0`` with native CGAL Mesh_3 and return its boundary."""

    if cells_across_diameter <= 0:
        raise ValueError("cells_across_diameter must be positive")
    native = _extension()
    native_field = create_native_field(field)
    positive = np.flatnonzero(field.max_radii > 0.0)
    if not len(positive):
        raise ValueError("CGAL field contains no positive-radius capsule")
    seed = int(positive[0])
    bounding_center = 0.5 * (field.starts[seed] + field.ends[seed])
    bounds_corners = np.asarray(
        [
            [x, y, z]
            for x in (field.bounds_min[0], field.bounds_max[0])
            for y in (field.bounds_min[1], field.bounds_max[1])
            for z in (field.bounds_min[2], field.bounds_max[2])
        ],
        dtype=np.float64,
    )
    padding = float(np.max(field.max_radii))
    bounding_radius = float(
        np.max(np.linalg.norm(bounds_corners - bounding_center[None, :], axis=1))
        + 2.0 * padding
    )
    result = native.mesh_implicit_surface(
        native_field,
        np.ascontiguousarray(bounding_center, dtype=np.float64),
        bounding_radius,
        float(cells_across_diameter),
        float(facet_angle_deg),
        float(facet_distance_fraction),
        float(cell_size_factor),
        float(cell_radius_edge_ratio),
    )
    vertices = np.asarray(result["vertices"], dtype=np.float64)
    triangles = np.asarray(result["triangles"], dtype=np.int64)
    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise RuntimeError("CGAL adapter returned invalid vertices")
    if triangles.ndim != 2 or triangles.shape[1] != 3:
        raise RuntimeError("CGAL adapter returned invalid triangles")
    if not len(vertices) or not len(triangles):
        raise RuntimeError("CGAL adapter returned an empty surface")
    faces = np.column_stack(
        (np.full(len(triangles), 3, dtype=np.int64), triangles)
    ).ravel()
    surface = pv.PolyData(vertices, faces).triangulate().clean(tolerance=0.0)
    telemetry = dict(result.get("telemetry", {}))
    telemetry.update(cgal_build_info())
    for key, value in telemetry.items():
        if isinstance(value, str):
            surface.field_data[f"cgal_{key}"] = np.asarray([value])
        elif isinstance(value, (int, np.integer)):
            surface.field_data[f"cgal_{key}"] = np.asarray([value], dtype=np.int64)
        elif isinstance(value, (float, np.floating)) and math.isfinite(float(value)):
            surface.field_data[f"cgal_{key}"] = np.asarray([value], dtype=np.float64)
    return surface


__all__ = [
    "CgalMesh3Unavailable",
    "NATIVE_API_VERSION",
    "cgal_build_info",
    "cgal_mesh3_available",
    "create_native_field",
    "mesh_cgal_implicit",
    "native_field_values",
    "native_sizing_values",
]
