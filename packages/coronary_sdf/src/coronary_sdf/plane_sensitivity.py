"""PDF-style plane sensitivity analysis for completed coronary CFX cases.

Geometry is defined once from the STL-cropped Amira graph and the exact meshing
STL, then reused unchanged for every CFD result.  Distances are millimetres in
geometry files and SI units in CFD-Post results.
"""

from __future__ import annotations

import csv
import json
import math
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from scipy.spatial import cKDTree


LEVELS = ("l1", "l2", "l3", "l4")
METRICS = ("wss_mean_pa", "wss_max_pa", "velocity_mean_m_s", "velocity_max_m_s", "pressure_mean_pa", "pressure_drop_pa")
DOMAIN_DIAGNOSTICS = (
    ("aspect_ratio", "Aspect Ratio", None),
    ("courant_number", "Courant Number", None),
    ("mesh_expansion_factor", "Mesh Expansion Factor", None),
    ("orthogonality_angle_rad", "Orthogonality Angle", "rad"),
)
HOTSPOT_DEFINITIONS = (
    ("courant_gt_100", "Courant Number", 100.0, None),
    ("aspect_ratio_gt_100", "Aspect Ratio", 100.0, None),
    ("expansion_gt_20", "Mesh Expansion Factor", 20.0, None),
    ("orthogonality_gt_85deg", "Orthogonality Angle", 85.0, "degree"),
)


class PlaneSelectionFailure(RuntimeError):
    def __init__(self, message: str, diagnostics: list[dict[str, Any]]):
        super().__init__(message)
        self.diagnostics = diagnostics


def _unit(vector: Iterable[float]) -> np.ndarray:
    value = np.asarray(vector, dtype=float)
    length = float(np.linalg.norm(value))
    if length <= 1.0e-12:
        raise ValueError("zero-length vector")
    return value / length


def polyline_arclength(points: np.ndarray) -> tuple[np.ndarray, float]:
    points = np.asarray(points, dtype=float)
    if len(points) < 2:
        raise ValueError("polyline needs at least two points")
    cumulative = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))]
    return cumulative, float(cumulative[-1])


def interpolate_polyline(points: np.ndarray, values: np.ndarray, distance: float) -> tuple[np.ndarray, float]:
    cumulative, total = polyline_arclength(points)
    distance = min(max(float(distance), 0.0), total)
    index = min(int(np.searchsorted(cumulative, distance, side="right") - 1), len(points) - 2)
    span = cumulative[index + 1] - cumulative[index]
    fraction = 0.0 if span <= 1.0e-12 else (distance - cumulative[index]) / span
    point = (1.0 - fraction) * np.asarray(points[index]) + fraction * np.asarray(points[index + 1])
    scalar = (1.0 - fraction) * float(values[index]) + fraction * float(values[index + 1])
    return point, scalar


def averaged_tangent(points: np.ndarray, centre_distance: float, window_length: float) -> np.ndarray:
    """Length-weighted tangent over a symmetric arc-length window."""
    cumulative, total = polyline_arclength(points)
    half = 0.5 * float(window_length)
    lo, hi = max(0.0, centre_distance - half), min(total, centre_distance + half)
    start, _ = interpolate_polyline(points, np.ones(len(points)), lo)
    end, _ = interpolate_polyline(points, np.ones(len(points)), hi)
    return _unit(end - start)


def local_curvature_score(points: np.ndarray, centre_distance: float, window_length: float) -> float:
    cumulative, total = polyline_arclength(points)
    lo, hi = max(0.0, centre_distance - window_length / 2), min(total, centre_distance + window_length / 2)
    mask = (cumulative >= lo) & (cumulative <= hi)
    sample = np.vstack((interpolate_polyline(points, np.ones(len(points)), lo)[0], points[mask], interpolate_polyline(points, np.ones(len(points)), hi)[0]))
    sample = np.unique(np.round(sample, 10), axis=0)
    if len(sample) < 3:
        return 0.0
    directions = np.diff(sample, axis=0)
    directions /= np.maximum(np.linalg.norm(directions, axis=1)[:, None], 1.0e-12)
    cosines = np.clip(np.sum(directions[:-1] * directions[1:], axis=1), -1.0, 1.0)
    return float(np.sum(np.arccos(cosines)) / max(hi - lo, 1.0e-12))


def plane_basis(normal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    normal = _unit(normal)
    axis = np.array([1.0, 0.0, 0.0]) if abs(normal[0]) < 0.8 else np.array([0.0, 1.0, 0.0])
    u = _unit(np.cross(normal, axis))
    return u, _unit(np.cross(normal, u))


def polygon_area_centroid(xy: np.ndarray) -> tuple[float, np.ndarray]:
    xy = np.asarray(xy, dtype=float)
    following = np.roll(xy, -1, axis=0)
    cross = xy[:, 0] * following[:, 1] - following[:, 0] * xy[:, 1]
    signed_area = 0.5 * float(np.sum(cross))
    if abs(signed_area) <= 1.0e-12:
        raise ValueError("degenerate section loop")
    centroid = np.sum((xy + following) * cross[:, None], axis=0) / (6.0 * signed_area)
    return abs(signed_area), centroid


def point_in_polygon(point: np.ndarray, polygon: np.ndarray) -> bool:
    x, y = map(float, point)
    polygon = np.asarray(polygon, dtype=float)
    inside = False
    j = len(polygon) - 1
    for i in range(len(polygon)):
        xi, yi = polygon[i]
        xj, yj = polygon[j]
        if ((yi > y) != (yj > y)) and x < (xj - xi) * (y - yi) / (yj - yi) + xi:
            inside = not inside
        j = i
    return inside


def _slice_loops(surface: Any, point: np.ndarray, normal: np.ndarray) -> list[np.ndarray]:
    cut = surface.slice(normal=normal, origin=point).clean(tolerance=1.0e-8, absolute=True)
    lines = np.asarray(cut.lines, dtype=np.int64)
    adjacency: dict[int, set[int]] = defaultdict(set)
    cursor = 0
    while cursor < len(lines):
        count = int(lines[cursor]); ids = lines[cursor + 1:cursor + 1 + count]; cursor += count + 1
        for a, b in zip(ids[:-1], ids[1:]):
            adjacency[int(a)].add(int(b)); adjacency[int(b)].add(int(a))
    loops: list[np.ndarray] = []
    unused = {tuple(sorted((a, b))) for a, neighbours in adjacency.items() for b in neighbours}
    while unused:
        first = next(iter(unused)); start, current = first; previous = start
        order = [start]; unused.discard(first)
        while True:
            order.append(current)
            if current == start:
                break
            choices = [n for n in adjacency[current] if n != previous and tuple(sorted((current, n))) in unused]
            if not choices:
                order = []
                break
            following = choices[0]
            unused.discard(tuple(sorted((current, following))))
            previous, current = current, following
            if len(order) > len(adjacency) + 2:
                order = []
                break
        if len(order) >= 4:
            loops.append(np.asarray(cut.points[np.asarray(order[:-1], dtype=int)], dtype=float))
    return loops


def validate_section(surface: Any, midpoint: np.ndarray, normal: np.ndarray, margin: float) -> dict[str, Any]:
    u, v = plane_basis(normal)
    loops = _slice_loops(surface, midpoint, normal)
    described = []
    for loop in loops:
        xy = np.column_stack(((loop - midpoint) @ u, (loop - midpoint) @ v))
        try:
            area, centroid_xy = polygon_area_centroid(xy)
        except ValueError:
            continue
        described.append((loop, xy, area, centroid_xy, point_in_polygon(np.zeros(2), xy)))
    containing = [item for item in described if item[4]]
    if len(containing) != 1:
        raise ValueError("section does not have exactly one closed loop containing the Amira midpoint")
    loop, xy, area, centroid_xy, _ = containing[0]
    centre = midpoint + centroid_xy[0] * u + centroid_xy[1] * v
    loop_radii = np.linalg.norm(loop - centre, axis=1)
    max_radius = float(np.max(loop_radii))
    bound_radius = float(margin) * max_radius
    for other, *_ in described:
        if other is loop:
            continue
        if float(np.min(np.linalg.norm(other - centre, axis=1))) <= bound_radius:
            raise ValueError("bounded plane intersects a second STL branch")
    return {"loop_points_mm": loop, "area_mm2": area, "section_equivalent_radius_mm": math.sqrt(area / math.pi), "centre_mm": centre, "max_radius_mm": max_radius, "bound_radius_mm": bound_radius, "loop_count": len(described)}


def _candidate(edge: dict[str, Any], tangent_window_diameters: float = 1.0) -> dict[str, Any]:
    points = np.asarray(edge["points_mm"], dtype=float)
    radii = np.asarray(edge["radii_mm"], dtype=float)
    cumulative, length = polyline_arclength(points)
    centre_distance = length / 2.0
    point, radius = interpolate_polyline(points, radii, centre_distance)
    diameter = 2.0 * radius
    tangent = averaged_tangent(points, centre_distance, tangent_window_diameters * diameter)
    return {
        "edge": edge, "edge_id": int(edge["edge_id"]), "strahler_order": int(edge["strahler"]),
        "radius_bin": int(edge.get("radius_bin", 0)),
        "midpoint_mm": point, "midpoint_radius_mm": float(radius), "length_mm": length,
        "endpoint_clearance_mm": length / 2.0, "endpoint_clearance_diameters": (length / 2.0) / max(diameter, 1.0e-12),
        "normal": tangent, "curvature_rad_per_mm": local_curvature_score(points, centre_distance, diameter),
    }


def rank_representative_candidates(edges: list[dict[str, Any]], category: str, value: int, target_radius: float, min_clearance_diameters: float, tangent_window_diameters: float = 1.0) -> list[dict[str, Any]]:
    candidates = [_candidate(edge, tangent_window_diameters) for edge in edges]
    if category == "strahler":
        candidates = [c for c in candidates if c["strahler_order"] == int(value)]
    else:
        candidates = [c for c in candidates if int(c["radius_bin"]) == int(value)]
    candidates = [c for c in candidates if c["endpoint_clearance_diameters"] >= min_clearance_diameters]
    return sorted(candidates, key=lambda c: (abs(math.log(max(c["midpoint_radius_mm"], 1e-12) / target_radius)), c["curvature_rad_per_mm"], -c["endpoint_clearance_diameters"], c["edge_id"]))


def select_all_vessel_planes(
    graph: dict[str, Any],
    surface: Any,
    bin_edges: list[float],
    settings: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Validate one bifurcation-clear midpoint section for every eligible edge."""
    minimum = float(settings["minimum_junction_clearance_local_diameters"])
    tangent_window = float(settings["tangent_averaging_window_local_diameters"])
    margin = 1.0 + float(settings["plane_coverage_margin_fraction"])
    accepted: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    for edge in sorted(graph["edges"], key=lambda item: int(item["edge_id"])):
        candidate = _candidate(edge, tangent_window)
        candidate["amira_radius_bin"] = _radius_bin(candidate["midpoint_radius_mm"], bin_edges)
        base = {
            "edge_id": candidate["edge_id"], "strahler_order": candidate["strahler_order"],
            "amira_radius_bin": candidate["amira_radius_bin"],
            "length_mm": candidate["length_mm"], "amira_midpoint_radius_mm": candidate["midpoint_radius_mm"],
            "endpoint_clearance_mm": candidate["endpoint_clearance_mm"],
            "amira_endpoint_clearance_diameters": candidate["endpoint_clearance_diameters"],
        }
        if candidate["endpoint_clearance_diameters"] < minimum:
            excluded.append({**base, "reason": "short_segment_below_two_local_diameters_from_endpoint", "detail": "midpoint clearance evaluated with Amira radius"})
            continue
        try:
            section = validate_section(surface, candidate["midpoint_mm"], candidate["normal"], margin)
        except ValueError as exc:
            excluded.append({**base, "reason": "invalid_or_contaminated_stl_section", "detail": str(exc)})
            continue
        stl_clearance_diameters = candidate["endpoint_clearance_mm"] / max(2.0 * float(section["section_equivalent_radius_mm"]), 1.0e-12)
        if stl_clearance_diameters < minimum:
            excluded.append({
                **base, "reason": "short_segment_after_stl_radius_validation",
                "detail": "STL-based midpoint clearance {:.6g}D is below {:.6g}D".format(stl_clearance_diameters, minimum),
            })
            continue
        radius_bin = _radius_bin(float(section["section_equivalent_radius_mm"]), bin_edges)
        plane = {
            **candidate, **section,
            "plane_id": "VES_E{:04d}".format(candidate["edge_id"]),
            "category": "all_vessels", "category_value": candidate["edge_id"],
            "radius_bin": radius_bin, "amira_radius_bin": candidate["amira_radius_bin"],
            "endpoint_clearance_diameters": stl_clearance_diameters,
            "validation_status": "valid", "candidate_rank": 1,
        }
        accepted.append(plane)
    return accepted, excluded


def _radius_bin(radius: float, edges: list[float]) -> int:
    return min(max(int(np.searchsorted(np.asarray(edges), radius, side="right")), 1), len(edges) - 1)


def select_planes(graph: dict[str, Any], surface: Any, bin_edges: list[float], settings: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    edges = graph["edges"]
    tangent_window = float(settings["tangent_averaging_window_local_diameters"])
    for edge in edges:
        midpoint = _candidate(edge, tangent_window)["midpoint_radius_mm"]
        edge["radius_bin"] = _radius_bin(midpoint, bin_edges)
    minimum = float(settings["minimum_junction_clearance_local_diameters"])
    margin = 1.0 + float(settings["plane_coverage_margin_fraction"])
    selected: dict[str, list[dict[str, Any]]] = {"strahler": [], "radius_bin": []}
    diagnostics: list[dict[str, Any]] = []
    order_radii = defaultdict(list)
    for edge in edges:
        candidate = _candidate(edge, tangent_window)
        if candidate["endpoint_clearance_diameters"] >= minimum:
            order_radii[candidate["strahler_order"]].append(candidate["midpoint_radius_mm"])
    groups = [("strahler", order, float(np.median(order_radii[order]))) for order in sorted(order_radii)]
    groups += [("radius_bin", index, math.sqrt(bin_edges[index - 1] * bin_edges[index])) for index in range(1, len(bin_edges))]
    counters = {"strahler": 0, "radius_bin": 0}
    for category, value, target in groups:
        ranked = rank_representative_candidates(edges, category, value, target, minimum, tangent_window)
        accepted = None
        for rank, candidate in enumerate(ranked, 1):
            try:
                section = validate_section(surface, candidate["midpoint_mm"], candidate["normal"], margin)
                accepted = {**candidate, **section, "candidate_rank": rank, "validation_status": "valid", "category": category, "category_value": value}
                break
            except ValueError as exc:
                diagnostics.append({"category": category, "category_value": value, "edge_id": candidate["edge_id"], "rank": rank, "reason": str(exc), "point_mm": candidate["midpoint_mm"].tolist(), "normal": candidate["normal"].tolist()})
        if accepted is None:
            raise PlaneSelectionFailure("No uncontaminated closed STL section for {} {}. See plane_selection_diagnostics.vtk".format(category, value), diagnostics)
        counters[category] += 1
        accepted["plane_id"] = ("STR_O{:02d}" if category == "strahler" else "RAD_B{:02d}").format(value)
        accepted["radius_bin"] = int(accepted["edge"]["radius_bin"])
        selected[category].append(accepted)
    return selected["strahler"], selected["radius_bin"], diagnostics


def plane_manifest_row(plane: dict[str, Any]) -> dict[str, Any]:
    row = {key: plane[key] for key in ("plane_id", "category", "category_value", "edge_id", "strahler_order", "radius_bin", "midpoint_radius_mm", "section_equivalent_radius_mm", "length_mm", "endpoint_clearance_mm", "endpoint_clearance_diameters", "curvature_rad_per_mm", "area_mm2", "max_radius_mm", "bound_radius_mm", "loop_count", "candidate_rank", "validation_status")}
    for prefix, values in (("point", plane["centre_mm"]), ("amira_midpoint", plane["midpoint_mm"]), ("normal", plane["normal"])):
        for suffix, value in zip("xyz", values): row[prefix + "_" + suffix] = float(value)
    return row


def _plane_patch(plane: dict[str, Any], numeric_id: int) -> Any:
    import pyvista as pv
    loop = np.asarray(plane["loop_points_mm"], dtype=float)
    centre = np.asarray(plane["centre_mm"], dtype=float)
    points = np.vstack((centre, loop))
    faces = []
    for index in range(len(loop)):
        faces.extend((3, 0, index + 1, (index + 1) % len(loop) + 1))
    patch = pv.PolyData(points, np.asarray(faces, dtype=np.int64))
    patch.cell_data["geometry_kind"] = np.full(patch.n_cells, 2, np.int32)
    patch.cell_data["plane_numeric_id"] = np.full(patch.n_cells, numeric_id, np.int32)
    patch.cell_data["edge_id"] = np.full(patch.n_cells, plane["edge_id"], np.int32)
    patch.cell_data["strahler_order"] = np.full(patch.n_cells, plane["strahler_order"], np.int32)
    patch.cell_data["radius_bin"] = np.full(patch.n_cells, plane["radius_bin"], np.int32)
    patch.cell_data["local_radius_mm"] = np.full(patch.n_cells, plane["midpoint_radius_mm"], float)
    patch.cell_data["section_area_mm2"] = np.full(patch.n_cells, plane["area_mm2"], float)
    patch.cell_data["section_equivalent_radius_mm"] = np.full(patch.n_cells, plane["section_equivalent_radius_mm"], float)
    patch.cell_data["bound_radius_mm"] = np.full(patch.n_cells, plane["bound_radius_mm"], float)
    patch.cell_data["endpoint_clearance_mm"] = np.full(patch.n_cells, plane["endpoint_clearance_mm"], float)
    patch.cell_data["endpoint_clearance_diameters"] = np.full(patch.n_cells, plane["endpoint_clearance_diameters"], float)
    for suffix, value in zip("xyz", plane["centre_mm"]): patch.cell_data["plane_point_" + suffix + "_mm"] = np.full(patch.n_cells, value, float)
    for suffix, value in zip("xyz", plane["normal"]): patch.cell_data["plane_normal_" + suffix] = np.full(patch.n_cells, value, float)
    patch.cell_data["validation_status"] = np.ones(patch.n_cells, np.int32)
    return patch


def _bounded_plane_square(
    plane: dict[str, Any],
    numeric_id: int,
    subdivisions: int = 16,
    display_scale: float = 2.5,
) -> Any:
    """Create the visible square sectioning plane used to locate the STL loop."""
    import pyvista as pv

    centre = np.asarray(plane["centre_mm"], float)
    normal = _unit(plane["normal"])
    u, v = plane_basis(normal)
    analysis_half_width = float(plane["bound_radius_mm"])
    display_half_width = max(display_scale * analysis_half_width, 0.75)
    coordinates = np.linspace(-display_half_width, display_half_width, int(subdivisions) + 1)
    points = np.asarray([centre + a * u + b * v for b in coordinates for a in coordinates], float)
    faces = []
    width = int(subdivisions) + 1
    for row in range(int(subdivisions)):
        for column in range(int(subdivisions)):
            lower_left = row * width + column
            faces.extend((4, lower_left, lower_left + 1, lower_left + width + 1, lower_left + width))
    square = pv.PolyData(points, np.asarray(faces, dtype=np.int64))
    square.cell_data["geometry_kind"] = np.full(square.n_cells, 3, np.int32)
    square.cell_data["plane_numeric_id"] = np.full(square.n_cells, numeric_id, np.int32)
    square.cell_data["edge_id"] = np.full(square.n_cells, int(plane["edge_id"]), np.int32)
    square.cell_data["strahler_order"] = np.full(square.n_cells, int(plane["strahler_order"]), np.int32)
    square.cell_data["radius_bin"] = np.full(square.n_cells, int(plane["radius_bin"]), np.int32)
    square.cell_data["analysis_bound_radius_mm"] = np.full(square.n_cells, analysis_half_width, float)
    square.cell_data["display_half_width_mm"] = np.full(square.n_cells, display_half_width, float)
    for suffix, value in zip("xyz", normal):
        square.cell_data["plane_normal_" + suffix] = np.full(square.n_cells, value, float)
    return square


def _centreline_polydata(graph: dict[str, Any]) -> Any:
    import pyvista as pv
    points, lines, edge_ids, orders, bins = [], [], [], [], []
    for edge in graph["edges"]:
        ids = np.arange(len(points), len(points) + len(edge["points_mm"]), dtype=np.int64)
        points.extend(edge["points_mm"]); lines.extend([len(ids), *ids]); edge_ids.append(int(edge["edge_id"])); orders.append(int(edge["strahler"])); bins.append(int(edge["radius_bin"]))
    mesh = pv.PolyData(np.asarray(points, float), lines=np.asarray(lines, np.int64))
    mesh.cell_data["geometry_kind"] = np.ones(mesh.n_cells, np.int32)
    mesh.cell_data["plane_numeric_id"] = np.zeros(mesh.n_cells, np.int32)
    mesh.cell_data["edge_id"] = np.asarray(edge_ids, np.int32); mesh.cell_data["strahler_order"] = np.asarray(orders, np.int32); mesh.cell_data["radius_bin"] = np.asarray(bins, np.int32)
    mesh.cell_data["local_radius_mm"] = np.zeros(mesh.n_cells); mesh.cell_data["section_area_mm2"] = np.zeros(mesh.n_cells); mesh.cell_data["section_equivalent_radius_mm"] = np.zeros(mesh.n_cells); mesh.cell_data["bound_radius_mm"] = np.zeros(mesh.n_cells); mesh.cell_data["endpoint_clearance_mm"] = np.zeros(mesh.n_cells); mesh.cell_data["endpoint_clearance_diameters"] = np.zeros(mesh.n_cells); mesh.cell_data["validation_status"] = np.ones(mesh.n_cells, np.int32)
    for suffix in "xyz": mesh.cell_data["plane_point_" + suffix + "_mm"] = np.zeros(mesh.n_cells); mesh.cell_data["plane_normal_" + suffix] = np.zeros(mesh.n_cells)
    return mesh


def write_geometry_packages(surface: Any, graph: dict[str, Any], sets: dict[str, list[dict[str, Any]]], output_dir: Path) -> None:
    import pyvista as pv
    output_dir.mkdir(parents=True, exist_ok=True)
    for stem, planes in sets.items():
        lumen = surface.copy(deep=False)
        for name, values in {"geometry_kind": 0, "plane_numeric_id": 0, "edge_id": -1, "strahler_order": 0, "radius_bin": 0, "local_radius_mm": 0.0, "section_area_mm2": 0.0, "section_equivalent_radius_mm": 0.0, "bound_radius_mm": 0.0, "endpoint_clearance_mm": 0.0, "endpoint_clearance_diameters": 0.0, "plane_point_x_mm": 0.0, "plane_point_y_mm": 0.0, "plane_point_z_mm": 0.0, "plane_normal_x": 0.0, "plane_normal_y": 0.0, "plane_normal_z": 0.0, "validation_status": 1}.items():
            lumen.cell_data[name] = np.full(lumen.n_cells, values)
        centreline = _centreline_polydata(graph)
        blocks = pv.MultiBlock(); blocks["validated_lumen_stl"] = lumen; blocks["cropped_amira_centreline"] = centreline
        combined = lumen.merge(centreline, merge_points=False)
        square_planes = None
        for index, plane in enumerate(planes, 1):
            square = _bounded_plane_square(plane, index)
            patch = _plane_patch(plane, index)
            square_planes = square if square_planes is None else square_planes.merge(square, merge_points=False)
            blocks[plane["plane_id"] + "_sectioning_square"] = square
            blocks[plane["plane_id"] + "_cross_section"] = patch
            combined = combined.merge(square, merge_points=False)
            combined = combined.merge(patch, merge_points=False)
        blocks.save(output_dir / (stem + ".vtm")); combined.save(output_dir / (stem + ".vtk"), binary=True)
        square_planes.save(output_dir / (stem + "_square_planes_only.vtk"), binary=True)


def write_selection_diagnostic(surface: Any, diagnostics: list[dict[str, Any]], path: Path) -> None:
    """Write failed candidate locations and normals in the exact STL frame."""
    import pyvista as pv
    markers = []
    for index, row in enumerate(diagnostics, 1):
        if "point_mm" not in row:
            continue
        point = np.asarray(row["point_mm"], float); normal = _unit(row["normal"])
        line = pv.Line(point - normal, point + normal)
        line.cell_data["diagnostic_id"] = np.full(line.n_cells, index, np.int32)
        line.cell_data["edge_id"] = np.full(line.n_cells, int(row["edge_id"]), np.int32)
        line.cell_data["candidate_rank"] = np.full(line.n_cells, int(row["rank"]), np.int32)
        line.cell_data["validation_status"] = np.zeros(line.n_cells, np.int32)
        markers.append(line)
    diagnostic = surface.copy(deep=False)
    diagnostic.cell_data["diagnostic_id"] = np.zeros(diagnostic.n_cells, np.int32)
    diagnostic.cell_data["edge_id"] = np.full(diagnostic.n_cells, -1, np.int32)
    diagnostic.cell_data["candidate_rank"] = np.zeros(diagnostic.n_cells, np.int32)
    diagnostic.cell_data["validation_status"] = np.ones(diagnostic.n_cells, np.int32)
    for marker in markers: diagnostic = diagnostic.merge(marker, merge_points=False)
    diagnostic.save(path, binary=True)


def _cfx_session(res: Path, csv_path: Path, planes: list[dict[str, Any]]) -> str:
    lines = ["COMMAND FILE:", "  CFX Post Version = 25.2", "END", ">load filename={}".format(res.as_posix())]
    for plane in planes:
        p = np.asarray(plane["centre_mm"]) / 1000.0; n = plane["normal"]
        lines += ["PLANE: {}".format(plane["plane_id"]), "  Apply Instancing Transform = On", "  Apply Texture = Off", "  Blend Texture = On", "  Bound Radius = {:.12g} [m]".format(plane["bound_radius_mm"] / 1000.0), "  Colour = 0.75, 0.75, 0.75", "  Colour Mode = Constant", "  Lighting = On", "  Normal = {:.12g}, {:.12g}, {:.12g}".format(*n), "  Option = Point and Normal", "  Plane Bound = Circular", "  Point = {:.12g} [m], {:.12g} [m], {:.12g} [m]".format(*p), "END"]
    safe = csv_path.as_posix().replace("'", "")
    lines += ["! open(my $fh, '>', '{}') or die $!;".format(safe), "! print $fh \"plane_id,area_m2,velocity_mean_m_s,velocity_max_m_s,pressure_mean_pa\\n\";"]
    for plane in planes:
        pid = plane["plane_id"]
        # Single quotes are essential here: in a Perl double-quoted string,
        # ``@PLANE_ID`` is interpreted as an array and silently removed before
        # CFD-Post sees the expression.
        lines += ["! my ($a_{0},$au_{0})=evaluate('area()@{0} / (1 [m^2])');".format(pid), "! my ($vm_{0},$vmu_{0})=evaluate('areaAve(Velocity)@{0} / (1 [m s^-1])');".format(pid), "! my ($vx_{0},$vxu_{0})=evaluate('maxVal(Velocity)@{0} / (1 [m s^-1])');".format(pid), "! my ($pr_{0},$pru_{0})=evaluate('areaAve(Pressure)@{0} / (1 [Pa])');".format(pid), "! print $fh \"{0},$a_{0},$vm_{0},$vx_{0},$pr_{0}\\n\";".format(pid)]
    lines += ["! close($fh);", ""]
    return "\n".join(lines)


def extract_cfx_planes(post: str, res: Path, planes: list[dict[str, Any]], output_csv: Path, resume: bool) -> list[dict[str, Any]]:
    complete = False
    if output_csv.exists() and resume:
        try:
            with output_csv.open(newline="", encoding="utf-8") as handle:
                cached = list(csv.DictReader(handle))
            complete = len(cached) == len(planes) and all(
                row.get("plane_id") and all(row.get(key) not in (None, "") for key in ("area_m2", "velocity_mean_m_s", "velocity_max_m_s", "pressure_mean_pa"))
                for row in cached
            )
        except (OSError, ValueError):
            complete = False
    if not complete:
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        try:
            from .cfx_extract import _short_result_link
        except ImportError:
            from cfx_extract import _short_result_link
        short_res = _short_result_link(res)
        session = output_csv.with_suffix(".cse"); session.write_text(_cfx_session(short_res, output_csv, planes), encoding="utf-8")
        process = subprocess.run([post, "-batch", str(session)], cwd=str(output_csv.parent), capture_output=True, text=True)
        (output_csv.parent / "cfxpost_stdout.log").write_text(process.stdout, encoding="utf-8", errors="replace")
        (output_csv.parent / "cfxpost_stderr.log").write_text(process.stderr, encoding="utf-8", errors="replace")
        if process.returncode or not output_csv.is_file():
            raise RuntimeError("CFD-Post plane extraction failed for {} (see {})".format(res.name, output_csv.parent))
    with output_csv.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != len(planes):
        raise RuntimeError("Expected {} CFD planes, found {} in {}".format(len(planes), len(rows), output_csv))
    return [{key: (value if key == "plane_id" else float(value)) for key, value in row.items()} for row in rows]


def _velocity_plane_session(res: Path, csv_path: Path, planes: list[dict[str, Any]]) -> str:
    """CFD-Post extraction containing only section area and velocity metrics."""
    lines = ["COMMAND FILE:", "  CFX Post Version = 25.2", "END", ">load filename={}".format(res.as_posix())]
    for plane in planes:
        point = np.asarray(plane["centre_mm"], float) / 1000.0
        normal = np.asarray(plane["normal"], float)
        lines += [
            "PLANE: {}".format(plane["plane_id"]),
            "  Apply Instancing Transform = On",
            "  Bound Radius = {:.12g} [m]".format(float(plane["bound_radius_mm"]) / 1000.0),
            "  Normal = {:.12g}, {:.12g}, {:.12g}".format(*normal),
            "  Option = Point and Normal",
            "  Plane Bound = Circular",
            "  Point = {:.12g} [m], {:.12g} [m], {:.12g} [m]".format(*point),
            "END",
        ]
    safe = csv_path.as_posix().replace("'", "")
    lines += [
        "! open(my $fh, '>', '{}') or die $!;".format(safe),
        "! print $fh \"plane_id,area_m2,velocity_mean_m_s,velocity_max_m_s\\n\";",
    ]
    for plane in planes:
        plane_id = plane["plane_id"]
        lines += [
            "! my ($a_{0},$au_{0})=evaluate('area()@{0} / (1 [m^2])');".format(plane_id),
            "! my ($vm_{0},$vmu_{0})=evaluate('areaAve(Velocity)@{0} / (1 [m s^-1])');".format(plane_id),
            "! my ($vx_{0},$vxu_{0})=evaluate('maxVal(Velocity)@{0} / (1 [m s^-1])');".format(plane_id),
            "! print $fh \"{0},$a_{0},$vm_{0},$vx_{0}\\n\";".format(plane_id),
        ]
    lines += ["! close($fh);", ""]
    return "\n".join(lines)


def extract_cfx_velocity_planes(
    post: str,
    res: Path,
    planes: list[dict[str, Any]],
    output_csv: Path,
    resume: bool,
) -> list[dict[str, Any]]:
    required = ("area_m2", "velocity_mean_m_s", "velocity_max_m_s")
    complete = False
    if output_csv.exists() and resume:
        try:
            with output_csv.open(newline="", encoding="utf-8") as handle:
                cached = list(csv.DictReader(handle))
            complete = len(cached) == len(planes) and all(all(row.get(key) not in (None, "") for key in required) for row in cached)
        except (OSError, ValueError):
            pass
    if not complete:
        try:
            from .cfx_extract import _short_result_link
        except ImportError:
            from cfx_extract import _short_result_link
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        short_res = _short_result_link(res)
        session = output_csv.with_suffix(".cse")
        session.write_text(_velocity_plane_session(short_res, output_csv, planes), encoding="utf-8")
        process = subprocess.run([post, "-batch", str(session)], cwd=str(output_csv.parent), capture_output=True, text=True)
        (output_csv.parent / "all_vessel_cfxpost_stdout.log").write_text(process.stdout, encoding="utf-8", errors="replace")
        (output_csv.parent / "all_vessel_cfxpost_stderr.log").write_text(process.stderr, encoding="utf-8", errors="replace")
        if process.returncode or not output_csv.is_file():
            raise RuntimeError("CFD-Post all-vessel velocity extraction failed for {}".format(res.name))
    with output_csv.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != len(planes):
        raise RuntimeError("Expected {} all-vessel planes, found {} in {}".format(len(planes), len(rows), output_csv))
    return [{key: (value if key == "plane_id" else float(value)) for key, value in row.items()} for row in rows]


def _domain_diagnostic_session(res: Path, domain: str, csv_path: Path) -> str:
    safe = csv_path.as_posix().replace("'", "")
    lines = [
        "COMMAND FILE:", "  CFX Post Version = 25.2", "END",
        ">load filename={}".format(res.as_posix()),
        "! open(my $fh, '>', '{}') or die $!;".format(safe),
        "! print $fh \"variable,minimum,volume_average,maximum\\n\";",
    ]
    for key, variable, unit in DOMAIN_DIAGNOSTICS:
        divisor = " / (1 [{}])".format(unit) if unit else ""
        lines += [
            "! my ($mn_{0},$mnu_{0})=evaluate('minVal({1})@{2}{3}');".format(key, variable, domain, divisor),
            "! my ($av_{0},$avu_{0})=evaluate('volumeAve({1})@{2}{3}');".format(key, variable, domain, divisor),
            "! my ($mx_{0},$mxu_{0})=evaluate('maxVal({1})@{2}{3}');".format(key, variable, domain, divisor),
            "! print $fh \"{0},$mn_{0},$av_{0},$mx_{0}\\n\";".format(key),
        ]
    lines += ["! close($fh);", ""]
    return "\n".join(lines)


def extract_cfx_domain_diagnostics(post: str, res: Path, output_csv: Path, resume: bool) -> list[dict[str, Any]]:
    expected = {item[0] for item in DOMAIN_DIAGNOSTICS}
    complete = False
    if output_csv.exists() and resume:
        try:
            with output_csv.open(newline="", encoding="utf-8") as handle:
                cached = list(csv.DictReader(handle))
            complete = {row["variable"] for row in cached} == expected and all(
                all(row.get(key) not in (None, "") for key in ("minimum", "volume_average", "maximum")) for row in cached
            )
        except (OSError, ValueError, KeyError):
            pass
    if not complete:
        try:
            from .cfx_extract import _short_result_link, probe_regions
        except ImportError:
            from cfx_extract import _short_result_link, probe_regions
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        domain, _ = probe_regions(res)
        short_res = _short_result_link(res)
        session = output_csv.with_suffix(".cse")
        session.write_text(_domain_diagnostic_session(short_res, domain, output_csv), encoding="utf-8")
        process = subprocess.run([post, "-batch", str(session)], cwd=str(output_csv.parent), capture_output=True, text=True)
        (output_csv.parent / "diagnostics_stdout.log").write_text(process.stdout, encoding="utf-8", errors="replace")
        (output_csv.parent / "diagnostics_stderr.log").write_text(process.stderr, encoding="utf-8", errors="replace")
        if process.returncode or not output_csv.is_file():
            raise RuntimeError("CFD-Post Courant/mesh-quality extraction failed for {}".format(res.name))
    with output_csv.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return [{key: (value if key == "variable" else float(value)) for key, value in row.items()} for row in rows]


def _hotspot_session(res: Path, domain: str, output_dir: Path) -> str:
    """Build CFX-Post isovolumes for diagnostically poor cells and export them."""
    variables = ",".join(item[1] for item in DOMAIN_DIAGNOSTICS)
    lines = [
        "COMMAND FILE:", "  CFX Post Version = 25.2", "END",
        ">load filename={}".format(res.as_posix()),
    ]
    for key, variable, threshold, unit in HOTSPOT_DEFINITIONS:
        location = "HOT_" + key.upper()
        value = "{:.12g}{}".format(threshold, " [{}]".format(unit) if unit else "")
        csv_path = (output_dir / (key + ".csv")).as_posix()
        lines += [
            "VOLUME: {}".format(location),
            "  Domain List = {}".format(domain),
            "  Option = Isovolume",
            "  Isovolume Intersection Mode = Above Value",
            "  Variable = {}".format(variable),
            "  Value 1 = {}".format(value),
            "  Inclusive = On",
            "  Remove Internal Element Faces = On",
            "END",
            "EXPORT:",
            "  Export File = {}".format(csv_path),
            "  Export Geometry = On",
            "  Export Type = Generic",
            "  Include Header = On",
            "  Location = {}".format(location),
            "  Location List = {}".format(location),
            "  Overwrite = On",
            "  Precision = 8",
            '  Separator = ", "',
            "  Spatial Variables = X,Y,Z",
            "  Variable List = {}".format(variables),
            "  Vector Display = Scalar",
            "END",
            ">export",
        ]
    lines.append("")
    return "\n".join(lines)


def _read_hotspot_export(path: Path) -> tuple[list[str], np.ndarray]:
    try:
        from .cfx_extract import read_export_csv
    except ImportError:
        from cfx_extract import read_export_csv
    return read_export_csv(path)


def _nearest_plane_data(points_mm: np.ndarray, planes: list[dict[str, Any]]) -> dict[str, np.ndarray]:
    """Distance from points to each bounded analysis disk, normalized locally."""
    centres = np.asarray([plane["centre_mm"] for plane in planes], float)
    normals = np.asarray([_unit(plane["normal"]) for plane in planes], float)
    bounds = np.asarray([plane["bound_radius_mm"] for plane in planes], float)
    diameters = 2.0 * np.asarray([plane["midpoint_radius_mm"] for plane in planes], float)
    count = len(points_mm)
    best_index = np.empty(count, np.int32)
    best_disk = np.empty(count, float)
    best_centre = np.empty(count, float)
    best_axial = np.empty(count, float)
    best_radial = np.empty(count, float)
    chunk = 100_000
    for start in range(0, count, chunk):
        stop = min(start + chunk, count)
        delta = points_mm[start:stop, None, :] - centres[None, :, :]
        signed = np.einsum("npi,pi->np", delta, normals)
        axial = np.abs(signed)
        radial_vector = delta - signed[:, :, None] * normals[None, :, :]
        radial = np.linalg.norm(radial_vector, axis=2)
        disk = np.sqrt(axial ** 2 + np.maximum(radial - bounds[None, :], 0.0) ** 2)
        normalized = disk / np.maximum(diameters[None, :], 1.0e-12)
        chosen = np.argmin(normalized, axis=1)
        rows = np.arange(stop - start)
        best_index[start:stop] = chosen
        best_disk[start:stop] = disk[rows, chosen]
        best_centre[start:stop] = np.linalg.norm(delta[rows, chosen], axis=1)
        best_axial[start:stop] = axial[rows, chosen]
        best_radial[start:stop] = radial[rows, chosen]
    return {
        "nearest_plane_index": best_index,
        "nearest_plane_disk_distance_mm": best_disk,
        "nearest_plane_center_distance_mm": best_centre,
        "nearest_plane_normal_distance_mm": best_axial,
        "nearest_plane_radial_distance_mm": best_radial,
        "nearest_plane_disk_distance_diameters": best_disk / diameters[best_index],
    }


def extract_cfx_hotspots(
    post: str,
    res: Path,
    output_dir: Path,
    planes: list[dict[str, Any]],
    level: str,
    resume: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Export thresholded hot spots, measure plane proximity and write a VTK package."""
    import pyvista as pv

    output_dir.mkdir(parents=True, exist_ok=True)
    expected = [output_dir / (item[0] + ".csv") for item in HOTSPOT_DEFINITIONS]
    if not (resume and all(path.is_file() and path.stat().st_size > 0 for path in expected)):
        try:
            from .cfx_extract import _short_result_link, probe_regions
        except ImportError:
            from cfx_extract import _short_result_link, probe_regions
        domain, _ = probe_regions(res)
        short_res = _short_result_link(res)
        session = output_dir / "quality_hotspots.cse"
        session.write_text(_hotspot_session(short_res, domain, output_dir), encoding="utf-8")
        process = subprocess.run([post, "-batch", str(session)], cwd=str(output_dir), capture_output=True, text=True)
        (output_dir / "hotspots_stdout.log").write_text(process.stdout, encoding="utf-8", errors="replace")
        (output_dir / "hotspots_stderr.log").write_text(process.stderr, encoding="utf-8", errors="replace")
        if process.returncode or not all(path.is_file() for path in expected):
            raise RuntimeError("CFD-Post quality hot-spot export failed for {}".format(res.name))

    blocks = pv.MultiBlock()
    plane_points = pv.PolyData(np.asarray([plane["centre_mm"] for plane in planes], float))
    plane_points.point_data["plane_numeric_id"] = np.arange(1, len(planes) + 1, dtype=np.int32)
    plane_points.point_data["local_diameter_mm"] = 2.0 * np.asarray([plane["midpoint_radius_mm"] for plane in planes], float)
    blocks["analysis_plane_centres"] = plane_points
    sectioning_squares = None
    section_patches = None
    for plane_index, plane in enumerate(planes, 1):
        square = _bounded_plane_square(plane, plane_index)
        section = _plane_patch(plane, plane_index)
        sectioning_squares = square if sectioning_squares is None else sectioning_squares.merge(square, merge_points=False)
        section_patches = section if section_patches is None else section_patches.merge(section, merge_points=False)
    blocks["analysis_sectioning_square_planes"] = sectioning_squares
    blocks["analysis_section_patches"] = section_patches
    summaries: list[dict[str, Any]] = []
    plane_proximity: list[dict[str, Any]] = []
    column_names = {
        "courant_number": ("Courant Number",),
        "aspect_ratio": ("Aspect Ratio",),
        "mesh_expansion_factor": ("Mesh Expansion Factor",),
        "orthogonality_angle_degrees": ("Orthogonality Angle",),
    }
    for type_id, (key, variable, threshold, unit) in enumerate(HOTSPOT_DEFINITIONS, 1):
        columns, values = _read_hotspot_export(output_dir / (key + ".csv"))
        if values.size == 0:
            summaries.append({
                "level": level, "hotspot": key, "variable": variable, "threshold": threshold,
                "threshold_unit": unit or "dimensionless", "exported_points": 0,
                "target_min": "", "target_p50": "", "target_p95": "", "target_p99": "", "target_max": "",
                "nearest_plane_id": "", "minimum_plane_disk_distance_mm": "",
                "minimum_plane_disk_distance_diameters": "", "points_within_0.5_diameters": 0,
                "points_within_1_diameter": 0,
            })
            continue
        xyz = [_column(columns, (axis,)) for axis in ("X", "Y", "Z")]
        points_mm = values[:, xyz] * 1000.0
        proximity = _nearest_plane_data(points_mm, planes)
        cloud = pv.PolyData(points_mm)
        cloud.point_data["hotspot_type_id"] = np.full(len(points_mm), type_id, np.int32)
        for output_name, alternatives in column_names.items():
            cloud.point_data[output_name] = values[:, _column(columns, alternatives)]
        for output_name, data in proximity.items():
            cloud.point_data[output_name] = data
        cloud.point_data["nearest_plane_numeric_id"] = proximity["nearest_plane_index"] + 1
        blocks[key] = cloud
        target_name = next(name for name, alternatives in column_names.items() if variable in alternatives)
        target = np.asarray(cloud.point_data[target_name], float)
        nearest_row = int(np.argmin(proximity["nearest_plane_disk_distance_diameters"]))
        nearest_index = int(proximity["nearest_plane_index"][nearest_row])
        summaries.append({
            "level": level, "hotspot": key, "variable": variable, "threshold": threshold,
            "threshold_unit": unit or "dimensionless", "exported_points": len(points_mm),
            "target_min": float(np.min(target)), "target_p50": float(np.percentile(target, 50)),
            "target_p95": float(np.percentile(target, 95)), "target_p99": float(np.percentile(target, 99)),
            "target_max": float(np.max(target)), "nearest_plane_id": planes[nearest_index]["plane_id"],
            "minimum_plane_disk_distance_mm": float(proximity["nearest_plane_disk_distance_mm"][nearest_row]),
            "minimum_plane_disk_distance_diameters": float(proximity["nearest_plane_disk_distance_diameters"][nearest_row]),
            "points_within_0.5_diameters": int(np.count_nonzero(proximity["nearest_plane_disk_distance_diameters"] <= 0.5)),
            "points_within_1_diameter": int(np.count_nonzero(proximity["nearest_plane_disk_distance_diameters"] <= 1.0)),
        })
        for plane_index, plane in enumerate(planes):
            centre = np.asarray(plane["centre_mm"], float)
            normal = _unit(plane["normal"])
            delta = points_mm - centre
            signed = delta @ normal
            axial = np.abs(signed)
            radial = np.linalg.norm(delta - signed[:, None] * normal[None, :], axis=1)
            disk = np.sqrt(axial ** 2 + np.maximum(radial - float(plane["bound_radius_mm"]), 0.0) ** 2)
            normalized = disk / max(2.0 * float(plane["midpoint_radius_mm"]), 1.0e-12)
            closest = int(np.argmin(normalized))
            plane_proximity.append({
                "level": level, "hotspot": key, "variable": variable,
                "plane_id": plane["plane_id"], "plane_category": plane["category"],
                "plane_category_value": plane["category_value"],
                "minimum_disk_distance_mm": float(disk[closest]),
                "minimum_disk_distance_diameters": float(normalized[closest]),
                "closest_point_target_value": float(target[closest]),
                "points_within_0.25_diameters": int(np.count_nonzero(normalized <= 0.25)),
                "points_within_0.5_diameters": int(np.count_nonzero(normalized <= 0.5)),
                "points_within_1_diameter": int(np.count_nonzero(normalized <= 1.0)),
            })
    vtm_path = output_dir / ("quality_hotspots_" + level + ".vtm")
    blocks.save(vtm_path)
    sectioning_squares.save(output_dir / ("analysis_square_planes_" + level + ".vtk"), binary=True)
    combined = sectioning_squares.copy(deep=True)
    combined.cell_data["hotspot_type_id"] = np.zeros(combined.n_cells, np.int32)
    for block_index in range(3, blocks.n_blocks):
        block = blocks[block_index]
        if block is not None and block.n_points:
            combined = combined.merge(block, merge_points=False)
    combined.save(output_dir / ("quality_hotspots_" + level + ".vtk"), binary=True)
    (output_dir / "plane_id_map.json").write_text(
        json.dumps({str(index): plane["plane_id"] for index, plane in enumerate(planes, 1)}, indent=2) + "\n",
        encoding="utf-8",
    )
    return summaries, plane_proximity


def _column(columns: list[str], alternatives: tuple[str, ...]) -> int:
    lower = {name.lower(): i for i, name in enumerate(columns)}
    for name in alternatives:
        if name.lower() in lower: return lower[name.lower()]
    raise KeyError("None of {} in {}".format(alternatives, columns))


def _dense_graph_samples(graph: dict[str, Any], spacing_mm: float = 0.1) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    points, edge_ids, arc = [], [], []
    for edge in graph["edges"]:
        xyz = np.asarray(edge["points_mm"], float); cumulative, total = polyline_arclength(xyz)
        sample_s = np.linspace(0, total, max(2, int(math.ceil(total / spacing_mm)) + 1))
        for s in sample_s:
            point, _ = interpolate_polyline(xyz, np.ones(len(xyz)), s); points.append(point); edge_ids.append(int(edge["edge_id"])); arc.append(s)
    return np.asarray(points), np.asarray(edge_ids), np.asarray(arc)


def wall_band_metrics(
    npz_path: Path,
    graph: dict[str, Any],
    planes: list[dict[str, Any]],
    band_length_diameters: float,
    strict: bool = True,
) -> dict[str, dict[str, float]]:
    data = np.load(npz_path, allow_pickle=True); columns = [str(v) for v in data["columns"]]; values = data["values"]
    xyz = values[:, [_column(columns, ("X",)), _column(columns, ("Y",)), _column(columns, ("Z",))]]
    graph_points = np.vstack([np.asarray(edge["points_mm"], float) for edge in graph["edges"]])
    graph_span = float(np.linalg.norm(np.ptp(graph_points, axis=0))); value_span = float(np.linalg.norm(np.ptp(xyz, axis=0)))
    xyz_mm = xyz * (1000.0 if value_span < graph_span / 10 else 1.0)
    wss = values[:, _column(columns, ("Wall Shear", "Wall Shear Magnitude"))]
    weights = np.asarray(data["surface_control_area"], float)
    samples, sample_edges, sample_arcs = _dense_graph_samples(graph)
    distance, nearest = cKDTree(samples).query(xyz_mm)
    nearest_edge = sample_edges[nearest]; nearest_arc = sample_arcs[nearest]
    edge_map = {int(edge["edge_id"]): edge for edge in graph["edges"]}
    result = {}
    for plane in planes:
        edge = edge_map[plane["edge_id"]]; _, length = polyline_arclength(np.asarray(edge["points_mm"], float))
        local_radius = float(plane.get("section_equivalent_radius_mm", plane["midpoint_radius_mm"]))
        half_band = band_length_diameters * local_radius
        mask = (nearest_edge == plane["edge_id"]) & (np.abs(nearest_arc - length / 2.0) <= half_band) & (distance <= 1.8 * plane["midpoint_radius_mm"])
        valid = mask & np.isfinite(wss) & np.isfinite(weights) & (weights > 0)
        if np.count_nonzero(valid) < 8:
            if strict:
                raise RuntimeError("Insufficient positive-area wall nodes for {}".format(plane["plane_id"]))
            continue
        result[plane["plane_id"]] = {"wss_mean_pa": float(np.average(wss[valid], weights=weights[valid])), "wss_max_pa": float(np.max(wss[valid])), "wall_band_nodes": int(np.count_nonzero(valid)), "wall_band_area_weight": float(np.sum(weights[valid]))}
    return result


def adjacent_percent(coarse: float, fine: float) -> float:
    return 100.0 * abs(float(fine) - float(coarse)) / max(abs(float(fine)), 1.0e-30)


def normalized_difference_percent(coarse: float, fine: float, normalization: float) -> float:
    return 100.0 * abs(float(fine) - float(coarse)) / max(abs(float(normalization)), 1.0e-30)


def three_grid_gci(values: list[float], counts: list[int], safety_factor: float = 1.25, normalization: float | None = None) -> dict[str, Any]:
    """Unequal-grid generalized observed order and fine-grid GCI (L2/L3/L4)."""
    f1, f2, f3 = map(float, values[-3:])  # coarse -> fine
    h1, h2, h3 = [float(n) ** (-1.0 / 3.0) for n in counts[-3:]]
    e21, e32 = f2 - f1, f3 - f2
    if e21 == 0 or e32 == 0 or e21 * e32 <= 0:
        return {"status": "oscillatory_or_exact", "observed_order": None, "gci_fine_percent": None, "finest_pair_percent": adjacent_percent(f2, f3)}
    r21, r32 = h1 / h2, h2 / h3
    ratio = abs(e21 / e32)
    # For unequal achieved refinements, the exact power-law relation is
    # |e21/e32| = r32^p (r21^p - 1) / (r32^p - 1).
    from scipy.optimize import brentq
    def residual(order: float) -> float:
        return r32 ** order * (r21 ** order - 1.0) / (r32 ** order - 1.0) - ratio
    lower, upper = 1.0e-4, 20.0
    if residual(lower) * residual(upper) > 0:
        return {"status": "non_asymptotic", "observed_order": None, "gci_fine_percent": None, "finest_pair_percent": adjacent_percent(f2, f3), "r21": r21, "r32": r32}
    p = float(brentq(residual, lower, upper))
    scale = abs(f3) if normalization is None else abs(float(normalization))
    gci = 100.0 * safety_factor * abs(e32) / max(scale * (r32 ** p - 1.0), 1.0e-30)
    return {"status": "monotonic", "observed_order": p, "gci_fine_percent": gci, "finest_pair_percent": adjacent_percent(f2, f3), "r21": r21, "r32": r32}


def _read_inlet_pressure(metrics_csv: Path) -> float:
    with metrics_csv.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["metric_id"] == "POI_001:pressure_static": return float(row["value"])
    raise RuntimeError("POI_001 inlet pressure reference missing from {}".format(metrics_csv))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows: return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)


def _plots_and_tables(report_dir: Path, stem: str, planes: list[dict[str, Any]], values: list[dict[str, Any]], convergence: list[dict[str, Any]], counts: dict[str, int]) -> None:
    import matplotlib.pyplot as plt
    labels = [p["plane_id"] for p in planes]
    figure, axes = plt.subplots(2, 3, figsize=(13.5, 8), constrained_layout=True)
    titles = (("wss_mean_pa", "Mean WSS (Pa)"), ("wss_max_pa", "Maximum WSS (Pa)"), ("velocity_mean_m_s", "Mean velocity (m/s)"), ("velocity_max_m_s", "Maximum velocity (m/s)"), ("pressure_mean_pa", "Mean static pressure (Pa)"), ("pressure_drop_pa", "Inlet-referenced pressure drop (Pa)"))
    lookup = {(row["plane_id"], row["level"]): row for row in values}
    for axis, (metric, title) in zip(axes.flat, titles):
        for pid in labels:
            axis.plot([counts[level] for level in LEVELS], [lookup[(pid, level)][metric] for level in LEVELS], marker="o", label=pid)
        axis.set_xscale("log")
        achieved = [counts[level] for level in LEVELS]
        axis.set_xticks(achieved)
        axis.set_xticklabels(["{:.2f}M".format(value / 1.0e6) for value in achieved])
        axis.minorticks_on()
        axis.tick_params(axis="x", which="major", rotation=25)
        axis.set_title(title); axis.set_xlabel("Achieved volume elements"); axis.grid(alpha=.3, which="both")
    axes.flat[0].legend(fontsize=7, ncol=2)
    figure.savefig(report_dir / (stem + "_convergence.png"), dpi=220); plt.close(figure)
    rows = [row for row in convergence if row["metric"] in ("wss_mean_pa", "wss_max_pa", "velocity_mean_m_s", "velocity_max_m_s", "pressure_mean_pa", "pressure_drop_pa")]
    metric_labels = {
        "wss_mean_pa": "Mean WSS (Pa)",
        "wss_max_pa": "Max WSS (Pa)",
        "velocity_mean_m_s": "Mean velocity (m/s)",
        "velocity_max_m_s": "Max velocity (m/s)",
        "pressure_mean_pa": "Mean pressure (Pa)",
        "pressure_drop_pa": "Pressure drop (Pa)",
    }
    cell_text = [[
        r["plane_id"], metric_labels.get(r["metric"], r["metric"]),
        "{:.4g}".format(r["l1_value"]), "{:.4g}".format(r["l2_value"]),
        "{:.4g}".format(r["l3_value"]), "{:.4g}".format(r["l4_value"]),
        "{:.2f}".format(r["l4_l3_percent"]),
        "" if r["gci_fine_percent"] in (None, "") else "{:.2f}".format(r["gci_fine_percent"]),
        r["status"], "PASS" if r["pass"] else "FAIL",
    ] for r in rows]
    height = max(6, 0.30 * len(cell_text)); fig, ax = plt.subplots(figsize=(18, height)); ax.axis("off")
    table = ax.table(
        cellText=cell_text,
        colLabels=["Plane", "Metric", "L1", "L2", "L3", "L4", "L4/L3 Δ (%)", "Fine GCI (%)", "Sequence", "Criterion"],
        loc="center", cellLoc="center",
    )
    table.auto_set_font_size(False); table.set_fontsize(8); table.scale(1, 1.25)
    fig.savefig(report_dir / (stem + "_table.png"), dpi=220, bbox_inches="tight")
    fig.savefig(report_dir / (stem + "_table.pdf"), bbox_inches="tight")
    plt.close(fig)


def _velocity_wss_criterion_figures(report_dir: Path, convergence: list[dict[str, Any]]) -> None:
    """Render separate 1% and 5% finest-pair figures for velocity and WSS."""
    import matplotlib.pyplot as plt

    families = {
        "velocity": (
            ("velocity_mean_m_s", "Mean velocity"),
            ("velocity_max_m_s", "Maximum velocity"),
        ),
        "wss": (
            ("wss_mean_pa", "Mean WSS"),
            ("wss_max_pa", "Maximum WSS"),
        ),
    }
    category_titles = (("strahler", "Strahler-order planes"), ("radius_bin", "Radius-bin planes"))
    summary_rows: list[dict[str, Any]] = []
    for family, metric_specs in families.items():
        for threshold in (5.0, 1.0):
            figure, axes = plt.subplots(1, 2, figsize=(13.2, 5.2), constrained_layout=True, sharey=True)
            for axis, (category, title) in zip(axes, category_titles):
                category_rows = [row for row in convergence if row["category"] == category]
                plane_ids = sorted(
                    {row["plane_id"] for row in category_rows},
                    key=lambda item: int(item[-2:]),
                )
                x = np.arange(len(plane_ids), dtype=float)
                width = 0.36
                panel_max = threshold
                for metric_index, (metric, label) in enumerate(metric_specs):
                    lookup = {row["plane_id"]: float(row["l4_l3_percent"]) for row in category_rows if row["metric"] == metric}
                    changes = np.asarray([lookup[plane_id] for plane_id in plane_ids])
                    panel_max = max(panel_max, float(np.max(changes)))
                    positions = x + (metric_index - 0.5) * width
                    bars = axis.bar(
                        positions, changes, width=width,
                        color=("#2878B5" if metric_index == 0 else "#E07B39"),
                        edgecolor="black", linewidth=0.45, label=label,
                    )
                    for bar, change in zip(bars, changes):
                        if change > threshold:
                            bar.set_hatch("///")
                            bar.set_alpha(0.82)
                    passed = int(np.count_nonzero(changes <= threshold))
                    summary_rows.append({
                        "quantity": family, "criterion_percent": threshold,
                        "category": category, "metric": metric,
                        "passed_planes": passed, "total_planes": len(changes),
                        "failed_planes": ";".join(
                            plane_id for plane_id, change in zip(plane_ids, changes)
                            if change > threshold
                        ),
                    })
                axis.axhline(threshold, color="#B22222", linestyle="--", linewidth=1.7, label="{}% criterion".format(int(threshold)))
                axis.set_xticks(x); axis.set_xticklabels(plane_ids, rotation=35, ha="right")
                axis.set_title(title); axis.set_xlabel("Analysis plane")
                axis.grid(axis="y", alpha=0.25)
                axis.set_ylim(0.0, max(threshold * 1.25, panel_max * 1.12))
            axes[0].set_ylabel("Absolute L3→L4 change (%)")
            handles, labels = axes[1].get_legend_handles_labels()
            figure.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.98), ncol=3, frameon=False)
            figure.suptitle("{} mesh sensitivity — {}% criterion".format("Velocity" if family == "velocity" else "Wall shear stress", int(threshold)), fontsize=15, fontweight="bold")
            stem = "{}_{}pct_criterion".format(family, int(threshold))
            figure.savefig(report_dir / (stem + ".png"), dpi=240)
            figure.savefig(report_dir / (stem + ".pdf"))
            plt.close(figure)
    _write_csv(report_dir / "velocity_wss_criterion_summary.csv", summary_rows)


def _cfx_diagnostic_figure(
    report_dir: Path,
    diagnostics: list[dict[str, Any]],
    counts: dict[str, int],
) -> None:
    """Compare volume-average and extreme CFX mesh/Courant diagnostics."""
    import matplotlib.pyplot as plt

    specifications = (
        ("aspect_ratio", "Aspect ratio", False),
        ("courant_number", "Courant number", True),
        ("mesh_expansion_factor", "Mesh expansion factor", True),
        ("orthogonality_angle_rad", "Orthogonality angle (degrees)", False),
    )
    lookup = {(row["level"], row["variable"]): row for row in diagnostics}
    achieved = [counts[level] for level in LEVELS]
    figure, axes = plt.subplots(2, 2, figsize=(12.5, 8.2), constrained_layout=True)
    for axis, (variable, title, logarithmic) in zip(axes.flat, specifications):
        rows = [lookup[(level, variable)] for level in LEVELS]
        if variable == "orthogonality_angle_rad":
            average = [float(row["volume_average_degrees"]) for row in rows]
            extreme = [float(row["maximum_degrees"]) for row in rows]
        else:
            average = [float(row["volume_average"]) for row in rows]
            extreme = [float(row["maximum"]) for row in rows]
        axis.plot(achieved, average, "o-", linewidth=1.8, label="Volume average")
        axis.plot(achieved, extreme, "s--", linewidth=1.6, label="Maximum")
        axis.set_xscale("log")
        if logarithmic:
            axis.set_yscale("log")
        axis.set_xticks(achieved)
        axis.set_xticklabels(["{:.2f}M".format(value / 1.0e6) for value in achieved], rotation=25)
        axis.set_title(title)
        axis.set_xlabel("Achieved volume elements")
        axis.grid(alpha=0.3, which="both")
    axes.flat[0].legend(frameon=False)
    figure.suptitle("CFX Courant and mesh-quality diagnostics", fontsize=15, fontweight="bold")
    figure.savefig(report_dir / "cfx_courant_mesh_quality.png", dpi=240)
    figure.savefig(report_dir / "cfx_courant_mesh_quality.pdf")
    plt.close(figure)


def _hotspot_proximity_figure(report_dir: Path, summaries: list[dict[str, Any]]) -> None:
    """Plot the closest thresholded quality location to any analysis plane."""
    import matplotlib.pyplot as plt

    keys = [item[0] for item in HOTSPOT_DEFINITIONS]
    labels = ["Courant >100", "Aspect ratio >100", "Expansion >20", "Orthogonality >85°"]
    lookup = {(row["hotspot"], row["level"]): row for row in summaries}
    distances = np.full((len(keys), len(LEVELS)), np.nan)
    for row_index, key in enumerate(keys):
        for column_index, level in enumerate(LEVELS):
            value = lookup[(key, level)]["minimum_plane_disk_distance_diameters"]
            if value not in (None, ""):
                distances[row_index, column_index] = float(value)
    figure, axis = plt.subplots(figsize=(8.4, 5.3), constrained_layout=True)
    image_handle = axis.imshow(distances, cmap="viridis_r", aspect="auto", vmin=0.0)
    axis.set_xticks(np.arange(len(LEVELS)), [level.upper() for level in LEVELS])
    axis.set_yticks(np.arange(len(keys)), labels)
    axis.set_xlabel("Global mesh level")
    axis.set_title("Nearest quality hot spot to an analysis plane")
    for row in range(distances.shape[0]):
        for column in range(distances.shape[1]):
            value = distances[row, column]
            axis.text(column, row, "n/a" if not np.isfinite(value) else "{:.2f}D".format(value), ha="center", va="center", color="white" if np.isfinite(value) and value < 0.8 else "black")
    colour_bar = figure.colorbar(image_handle, ax=axis)
    colour_bar.set_label("Distance to bounded plane (local diameters)")
    figure.savefig(report_dir / "cfx_quality_hotspot_plane_proximity.png", dpi=240)
    figure.savefig(report_dir / "cfx_quality_hotspot_plane_proximity.pdf")
    plt.close(figure)


def _hotspot_plane_matrix_figures(report_dir: Path, rows: list[dict[str, Any]]) -> None:
    """Name the analysis planes approached by each thresholded hot-spot set."""
    import matplotlib.pyplot as plt

    keys = [item[0] for item in HOTSPOT_DEFINITIONS]
    labels = ["Courant >100", "Aspect >100", "Expansion >20", "Angle >85°"]
    plane_ids = sorted({row["plane_id"] for row in rows}, key=lambda value: (value.split("_")[0], int(value[-2:])))
    lookup = {(row["level"], row["plane_id"], row["hotspot"]): row for row in rows}
    for level in LEVELS:
        distances = np.asarray([
            [float(lookup[(level, plane_id, key)]["minimum_disk_distance_diameters"]) for key in keys]
            for plane_id in plane_ids
        ])
        display = np.minimum(distances, 2.0)
        figure, axis = plt.subplots(figsize=(8.8, 8.4), constrained_layout=True)
        image_handle = axis.imshow(display, cmap="viridis_r", aspect="auto", vmin=0.0, vmax=2.0)
        axis.set_xticks(np.arange(len(keys)), labels, rotation=22, ha="right")
        axis.set_yticks(np.arange(len(plane_ids)), plane_ids)
        axis.set_title("{} quality hot spots near analysis planes".format(level.upper()))
        for row_index in range(len(plane_ids)):
            for column_index in range(len(keys)):
                value = distances[row_index, column_index]
                text_value = ">2D" if value > 2.0 else "{:.2f}D".format(value)
                axis.text(column_index, row_index, text_value, ha="center", va="center", fontsize=7.5, color="white" if value < 0.8 else "black")
        colour_bar = figure.colorbar(image_handle, ax=axis)
        colour_bar.set_label("Minimum distance to bounded plane (local diameters; clipped at 2D)")
        figure.savefig(report_dir / ("cfx_quality_hotspot_plane_matrix_" + level + ".png"), dpi=240)
        figure.savefig(report_dir / ("cfx_quality_hotspot_plane_matrix_" + level + ".pdf"))
        plt.close(figure)


ALL_VESSEL_METRICS = ("wss_mean_pa", "wss_max_pa", "velocity_mean_m_s", "velocity_max_m_s")


def _write_all_vessel_geometry(surface: Any, graph: dict[str, Any], planes: list[dict[str, Any]], output_dir: Path) -> None:
    """Write compact all-vessel geometry with visible planes and exact sections."""
    import pyvista as pv

    output_dir.mkdir(parents=True, exist_ok=True)
    squares = None
    sections = None
    for index, plane in enumerate(planes, 1):
        square = _bounded_plane_square(plane, index)
        section = _plane_patch(plane, index)
        squares = square if squares is None else squares.merge(square, merge_points=False)
        sections = section if sections is None else sections.merge(section, merge_points=False)
    centreline = _centreline_polydata(graph)
    blocks = pv.MultiBlock()
    blocks["validated_lumen_stl"] = surface
    blocks["cropped_amira_centreline"] = centreline
    blocks["all_vessel_sectioning_square_planes"] = squares
    blocks["all_vessel_exact_cross_sections"] = sections
    blocks.save(output_dir / "all_vessel_planes.vtm")
    squares.save(output_dir / "all_vessel_square_planes_only.vtk", binary=True)
    sections.save(output_dir / "all_vessel_cross_sections_only.vtk", binary=True)
    combined = surface.merge(centreline, merge_points=False).merge(squares, merge_points=False).merge(sections, merge_points=False)
    combined.save(output_dir / "all_vessel_planes.vtk", binary=True)


def _distribution(values: np.ndarray, weights: np.ndarray | None = None) -> dict[str, float]:
    values = np.asarray(values, float)
    result = {
        "equal_vessel_mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "q25": float(np.percentile(values, 25)),
        "q75": float(np.percentile(values, 75)),
        "iqr": float(np.percentile(values, 75) - np.percentile(values, 25)),
        "p95": float(np.percentile(values, 95)),
        "maximum": float(np.max(values)),
    }
    if weights is not None:
        weights = np.asarray(weights, float)
        valid = np.isfinite(weights) & (weights > 0)
        result["area_weighted_mean"] = float(np.average(values[valid], weights=weights[valid]))
    else:
        result["area_weighted_mean"] = float("nan")
    return result


def _all_vessel_group_results(
    planes: list[dict[str, Any]],
    values: list[dict[str, Any]],
    counts: dict[str, int],
    safety_factor: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    metadata = {plane["plane_id"]: plane for plane in planes}
    value_lookup = {(row["plane_id"], row["level"]): row for row in values}
    per_vessel: list[dict[str, Any]] = []
    for plane in planes:
        for metric in ALL_VESSEL_METRICS:
            metric_values = [float(value_lookup[(plane["plane_id"], level)][metric]) for level in LEVELS]
            differences = [adjacent_percent(metric_values[index], metric_values[index + 1]) for index in range(3)]
            gci = three_grid_gci(metric_values, [counts[level] for level in LEVELS], safety_factor)
            normalized_gci = {key: gci.get(key, "") for key in ("status", "observed_order", "gci_fine_percent", "finest_pair_percent", "r21", "r32")}
            per_vessel.append({
                "plane_id": plane["plane_id"], "edge_id": plane["edge_id"],
                "strahler_order": plane["strahler_order"], "radius_bin": plane["radius_bin"],
                "section_equivalent_radius_mm": plane["section_equivalent_radius_mm"],
                "metric": metric, **{level + "_value": value for level, value in zip(LEVELS, metric_values)},
                "l2_l1_percent": differences[0], "l3_l2_percent": differences[1], "l4_l3_percent": differences[2],
                "passes_5_percent": differences[2] <= 5.0, "passes_1_percent": differences[2] <= 1.0,
                **normalized_gci,
            })

    group_values: list[dict[str, Any]] = []
    group_convergence: list[dict[str, Any]] = []
    pairs = (("l2_l1", "l1", "l2"), ("l3_l2", "l2", "l3"), ("l4_l3", "l3", "l4"))
    for group_type, attribute in (("strahler_order", "strahler_order"), ("radius_bin", "radius_bin")):
        groups = sorted({int(plane[attribute]) for plane in planes})
        for group_value in groups:
            members = [plane for plane in planes if int(plane[attribute]) == group_value]
            plane_ids = {plane["plane_id"] for plane in members}
            sparse = len(members) < 5
            note = "SPARSE: fewer than 5 eligible vessels" if sparse else ""
            for metric in ALL_VESSEL_METRICS:
                stat_by_level: dict[str, dict[str, float]] = {}
                for level in LEVELS:
                    rows = [value_lookup[(plane_id, level)] for plane_id in plane_ids]
                    metric_array = np.asarray([float(row[metric]) for row in rows])
                    weight_key = "wall_band_area_weight" if metric.startswith("wss") else "cfx_area_m2"
                    weights = np.asarray([float(row[weight_key]) for row in rows])
                    stat_by_level[level] = _distribution(metric_array, weights)
                for statistic in ("equal_vessel_mean", "area_weighted_mean", "median", "q25", "q75", "iqr", "p95", "maximum"):
                    group_values.append({
                        "group_type": group_type, "group_value": group_value,
                        "metric": metric, "statistic": statistic, "vessel_count": len(members),
                        **{level + "_value": stat_by_level[level][statistic] for level in LEVELS},
                        "sparse_group": sparse, "uncertainty_note": note,
                    })
                vessel_rows = [row for row in per_vessel if row["plane_id"] in plane_ids and row["metric"] == metric]
                for pair_name, _, fine_level in pairs:
                    differences = np.asarray([float(row[pair_name + "_percent"]) for row in vessel_rows])
                    fine_weights = np.asarray([
                        float(value_lookup[(row["plane_id"], fine_level)]["wall_band_area_weight" if metric.startswith("wss") else "cfx_area_m2"])
                        for row in vessel_rows
                    ])
                    stats = _distribution(differences, fine_weights)
                    valid_gci = [float(row["gci_fine_percent"]) for row in vessel_rows if row.get("gci_fine_percent") not in (None, "")]
                    group_convergence.append({
                        "group_type": group_type, "group_value": group_value,
                        "metric": metric, "mesh_pair": pair_name, "vessel_count": len(members),
                        **stats,
                        "percent_vessels_passing_5_percent": 100.0 * float(np.mean(differences <= 5.0)),
                        "percent_vessels_passing_1_percent": 100.0 * float(np.mean(differences <= 1.0)),
                        "valid_monotonic_gci_count": len(valid_gci),
                        "median_valid_gci_percent": float(np.median(valid_gci)) if valid_gci else "",
                        "p95_valid_gci_percent": float(np.percentile(valid_gci, 95)) if valid_gci else "",
                        "sparse_group": sparse, "uncertainty_note": note,
                    })
    return per_vessel, group_values, group_convergence


def _all_vessel_group_figures(
    report_dir: Path,
    group_values: list[dict[str, Any]],
    group_convergence: list[dict[str, Any]],
    counts: dict[str, int],
) -> None:
    import matplotlib.pyplot as plt

    titles = {
        "wss_mean_pa": "Mean WSS (Pa)", "wss_max_pa": "Maximum WSS (Pa)",
        "velocity_mean_m_s": "Mean velocity (m/s)", "velocity_max_m_s": "Maximum velocity (m/s)",
    }
    for group_type, stem in (("strahler_order", "strahler"), ("radius_bin", "radius_bin")):
        rows = [row for row in group_values if row["group_type"] == group_type]
        groups = sorted({int(row["group_value"]) for row in rows})
        # Median lines only. The IQR bands these used to carry overlapped
        # heavily between groups and hid the very lines they belonged to; the
        # spread per group is in `group_values.csv` (q25/q75) and in the
        # sensitivity figure, which is where it can actually be read.
        figure, axes = plt.subplots(2, 2, figsize=(15.0, 8.5), constrained_layout=True)
        for axis, metric in zip(axes.flat, ALL_VESSEL_METRICS):
            for group in groups:
                median_row = next(row for row in rows if int(row["group_value"]) == group and row["metric"] == metric and row["statistic"] == "median")
                x = np.asarray([counts[level] for level in LEVELS])
                median = np.asarray([median_row[level + "_value"] for level in LEVELS], float)
                axis.plot(x, median, marker="o", label="{} {} (n={})".format("Order" if stem == "strahler" else "Bin", group, median_row["vessel_count"]))
            axis.set_xscale("log"); axis.set_xticks([counts[level] for level in LEVELS])
            axis.set_xticklabels(["{:.2f}M".format(counts[level] / 1e6) for level in LEVELS], rotation=25)
            axis.set_title(titles[metric]); axis.set_xlabel("Volume elements"); axis.grid(alpha=0.3, which="both")
        # Legend outside the panels: in-axes it sat over the order-1 data.
        handles, labels = axes.flat[0].get_legend_handles_labels()
        figure.legend(handles, labels, loc="outside right upper", fontsize=9,
                      frameon=False, title="Strahler order" if stem == "strahler" else "Radius bin")
        figure.suptitle("All-vessel {} grouped values: median".format("Strahler" if stem == "strahler" else "radius-bin"), fontsize=14, fontweight="bold")
        figure.savefig(report_dir / (stem + "_all_vessel_values.png"), dpi=240)
        figure.savefig(report_dir / (stem + "_all_vessel_values.pdf"))
        plt.close(figure)

        finest = [row for row in group_convergence if row["group_type"] == group_type and row["mesh_pair"] == "l4_l3"]
        figure, axes = plt.subplots(2, 2, figsize=(13.5, 8.5), constrained_layout=True)
        x = np.arange(len(groups), dtype=float)
        for axis, metric in zip(axes.flat, ALL_VESSEL_METRICS):
            metric_rows = {int(row["group_value"]): row for row in finest if row["metric"] == metric}
            medians = [float(metric_rows[group]["median"]) for group in groups]
            p95 = [float(metric_rows[group]["p95"]) for group in groups]
            axis.bar(x - 0.18, medians, width=0.36, label="Median")
            axis.bar(x + 0.18, p95, width=0.36, label="P95")
            axis.axhline(5.0, color="#B22222", linestyle="--", label="5% criterion")
            axis.axhline(1.0, color="#333333", linestyle=":", label="1% criterion")
            axis.set_xticks(x, [str(group) for group in groups]); axis.set_xlabel("Strahler order" if stem == "strahler" else "Radius bin")
            axis.set_title(titles[metric]); axis.set_ylabel("Paired L3→L4 change (%)"); axis.grid(axis="y", alpha=0.25)
        axes.flat[0].legend(fontsize=8)
        figure.suptitle("All-vessel {} mesh sensitivity".format("Strahler" if stem == "strahler" else "radius-bin"), fontsize=14, fontweight="bold")
        figure.savefig(report_dir / (stem + "_all_vessel_sensitivity.png"), dpi=240)
        figure.savefig(report_dir / (stem + "_all_vessel_sensitivity.pdf"))
        plt.close(figure)

        mean_lookup = {(int(row["group_value"]), row["metric"]): row for row in rows if row["statistic"] == "equal_vessel_mean"}
        finest_lookup = {(int(row["group_value"]), row["metric"]): row for row in finest}
        table_rows = []
        for group in groups:
            for metric in ALL_VESSEL_METRICS:
                physical = mean_lookup[(group, metric)]; sensitivity = finest_lookup[(group, metric)]
                table_rows.append([
                    str(group), titles[metric], str(physical["vessel_count"]),
                    *["{:.4g}".format(float(physical[level + "_value"])) for level in LEVELS],
                    "{:.2f}".format(float(sensitivity["median"])), "{:.2f}".format(float(sensitivity["p95"])),
                    "{:.1f}".format(float(sensitivity["percent_vessels_passing_5_percent"])),
                    "{:.1f}".format(float(sensitivity["percent_vessels_passing_1_percent"])),
                    "YES" if physical["sparse_group"] else "NO",
                ])
        figure, axis = plt.subplots(figsize=(18, max(6, 0.36 * len(table_rows)))); axis.axis("off")
        table = axis.table(
            cellText=table_rows,
            colLabels=["Group", "Metric", "n", "L1 mean", "L2 mean", "L3 mean", "L4 mean", "Median Δ%", "P95 Δ%", "%≤5", "%≤1", "Sparse"],
            loc="center", cellLoc="center",
        )
        table.auto_set_font_size(False); table.set_fontsize(7.5); table.scale(1, 1.22)
        figure.savefig(report_dir / (stem + "_all_vessel_table.png"), dpi=240, bbox_inches="tight")
        figure.savefig(report_dir / (stem + "_all_vessel_table.pdf"), bbox_inches="tight")
        plt.close(figure)


def run_all_vessel_group_sensitivity(
    cfg: dict[str, Any],
    graph: dict[str, Any],
    surface: Any,
    bin_edges: list[float],
    settings: dict[str, Any],
    counts: dict[str, int],
    resume: bool,
    family: str = "global",
) -> Path:
    """Analyse every bifurcation-clear vessel midpoint using velocity and WSS only."""
    from collections import Counter

    output = Path(cfg["paths"]["output_dir"])
    if family not in ("global", "adaptive"):
        raise ValueError("family must be 'global' or 'adaptive'")
    root_name = "all_vessels" if family == "global" else "all_vessels_adaptive"
    root = output / "plane_sensitivity" / root_name
    geometry_dir = root / "geometry"
    cfx_dir = root / "cfx"
    report_dir = root / "reports"
    for directory in (root, geometry_dir, cfx_dir, report_dir):
        directory.mkdir(parents=True, exist_ok=True)
    manifest_path = geometry_dir / "all_vessel_plane_definitions.json"
    exclusions_path = geometry_dir / "all_vessel_plane_exclusions.csv"
    global_manifest_path = (
        output / "plane_sensitivity" / "all_vessels" / "geometry"
        / "all_vessel_plane_definitions.json"
    )
    # The matched global/adaptive comparison must use identical anatomical
    # stations.  Seed the adaptive manifest from the already validated global
    # definitions instead of re-running geometric ranking and loop centring.
    if (
        family == "adaptive"
        and not manifest_path.exists()
        and global_manifest_path.is_file()
    ):
        manifest_path.write_text(
            global_manifest_path.read_text(encoding="utf-8"),
            encoding="utf-8",
        )
    if manifest_path.exists() and resume:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        planes = payload["planes"]
        exclusions = payload["exclusions"]
        for plane in planes:
            for key in ("midpoint_mm", "centre_mm", "normal", "loop_points_mm"):
                plane[key] = np.asarray(plane[key], float)
    else:
        print("[all-vessels] validating midpoint sections and bifurcation clearance", flush=True)
        planes, exclusions = select_all_vessel_planes(graph, surface, bin_edges, settings)
        serial = lambda plane: {key: (value.tolist() if isinstance(value, np.ndarray) else value) for key, value in plane.items() if key != "edge"}
        manifest_path.write_text(json.dumps({
            "contract_version": 1,
            "minimum_endpoint_clearance_local_diameters": float(settings["minimum_junction_clearance_local_diameters"]),
            "planes": [serial(plane) for plane in planes], "exclusions": exclusions,
        }, indent=2) + "\n", encoding="utf-8")
    _write_csv(exclusions_path, exclusions)
    definition_rows = []
    for plane in planes:
        row = plane_manifest_row(plane)
        row["amira_radius_bin"] = plane["amira_radius_bin"]
        definition_rows.append(row)
    _write_csv(geometry_dir / "all_vessel_plane_definitions.csv", definition_rows)
    reason_counts = Counter(row["reason"] for row in exclusions)
    selection_summary = {
        "total_retained_amira_edges": len(graph["edges"]),
        "geometrically_eligible_planes": len(planes),
        "excluded_edges": len(exclusions),
        "exclusion_reason_counts": dict(sorted(reason_counts.items())),
        "eligible_by_strahler_order": dict(sorted(Counter(str(plane["strahler_order"]) for plane in planes).items())),
        "eligible_by_stl_radius_bin": dict(sorted(Counter(str(plane["radius_bin"]) for plane in planes).items())),
        "sparse_radius_bins": [value for value, count in sorted(Counter(int(plane["radius_bin"]) for plane in planes).items()) if count < 5],
        "clearance_note": "Midpoint must be at least two STL local diameters from both segment endpoints; this conservatively excludes terminal as well as bifurcation-adjacent short segments.",
    }
    (geometry_dir / "all_vessel_selection_summary.json").write_text(json.dumps(selection_summary, indent=2) + "\n", encoding="utf-8")

    post = cfg["executables"]["cfx_post"]
    values: list[dict[str, Any]] = []
    insufficient: set[str] = set()
    for level in LEVELS:
        case = family + "_" + level
        result_candidates = sorted((output / "cfx" / case).glob(case + "*.res"), key=lambda path: ("continue" in path.stem, path.stat().st_mtime))
        if not result_candidates:
            raise FileNotFoundError("No result for " + case)
        case_output = cfx_dir / case
        print("[all-vessels] CFD-Post {}: {} midpoint planes".format(case, len(planes)), flush=True)
        section_rows = extract_cfx_velocity_planes(post, result_candidates[-1], planes, case_output / "all_vessel_velocity_sections.csv", resume)
        section_lookup = {row["plane_id"]: row for row in section_rows}
        wall = wall_band_metrics(
            output / "extracted" / case / (case + "_wall.npz"), graph, planes,
            float(settings["wss_wall_band_length_local_diameters"]), strict=False,
        )
        missing = {plane["plane_id"] for plane in planes} - set(wall)
        insufficient.update(missing)
        for plane in planes:
            if plane["plane_id"] not in wall:
                continue
            section = section_lookup[plane["plane_id"]]
            stl_area_m2 = float(plane["area_mm2"]) * 1.0e-6
            values.append({
                "plane_id": plane["plane_id"], "edge_id": plane["edge_id"],
                "strahler_order": plane["strahler_order"], "radius_bin": plane["radius_bin"],
                "section_equivalent_radius_mm": plane["section_equivalent_radius_mm"],
                "level": level, "elements": counts[level],
                **wall[plane["plane_id"]],
                "velocity_mean_m_s": section["velocity_mean_m_s"],
                "velocity_max_m_s": section["velocity_max_m_s"],
                "cfx_area_m2": section["area_m2"], "stl_area_m2": stl_area_m2,
                "area_relative_error": abs(float(section["area_m2"]) - stl_area_m2) / max(stl_area_m2, 1.0e-30),
            })
    if insufficient:
        for plane_id in sorted(insufficient):
            plane = next(item for item in planes if item["plane_id"] == plane_id)
            exclusions.append({
                "edge_id": plane["edge_id"], "strahler_order": plane["strahler_order"],
                "amira_radius_bin": plane["amira_radius_bin"], "length_mm": plane["length_mm"],
                "amira_midpoint_radius_mm": plane["midpoint_radius_mm"],
                "endpoint_clearance_mm": plane["endpoint_clearance_mm"],
                "amira_endpoint_clearance_diameters": plane.get("endpoint_clearance_diameters", ""),
                "reason": "insufficient_positive_area_wall_nodes_on_at_least_one_mesh",
                "detail": "fewer than 8 wall nodes in the one-diameter band",
            })
        planes = [plane for plane in planes if plane["plane_id"] not in insufficient]
        values = [row for row in values if row["plane_id"] not in insufficient]
        _write_csv(exclusions_path, exclusions)
    expected_rows = len(planes) * len(LEVELS)
    if len(values) != expected_rows:
        raise RuntimeError("Expected {} complete all-vessel value rows, found {}".format(expected_rows, len(values)))

    analysed_definition_rows = []
    for plane in planes:
        row = plane_manifest_row(plane)
        row["amira_radius_bin"] = plane["amira_radius_bin"]
        analysed_definition_rows.append(row)
    _write_csv(geometry_dir / "all_vessel_analysed_plane_definitions.csv", analysed_definition_rows)
    _write_all_vessel_geometry(surface, graph, planes, geometry_dir)
    _write_csv(report_dir / "all_vessel_values.csv", values)
    per_vessel, group_values, group_convergence = _all_vessel_group_results(
        planes, values, counts, float(settings["gci_safety_factor"]),
    )
    _write_csv(report_dir / "all_vessel_convergence.csv", per_vessel)
    _write_csv(report_dir / "group_values.csv", group_values)
    _write_csv(report_dir / "group_convergence.csv", group_convergence)
    for group_type, stem in (("strahler_order", "strahler"), ("radius_bin", "radius_bin")):
        _write_csv(report_dir / (stem + "_group_values.csv"), [row for row in group_values if row["group_type"] == group_type])
        _write_csv(report_dir / (stem + "_group_convergence.csv"), [row for row in group_convergence if row["group_type"] == group_type])
    _all_vessel_group_figures(report_dir, group_values, group_convergence, counts)
    final_counts = {
        "analysed_vessels": len(planes),
        "excluded_vessels": len(exclusions),
        "by_strahler_order": dict(sorted(Counter(str(plane["strahler_order"]) for plane in planes).items())),
        "by_radius_bin": dict(sorted(Counter(str(plane["radius_bin"]) for plane in planes).items())),
        "sparse_strahler_orders": [value for value, count in sorted(Counter(int(plane["strahler_order"]) for plane in planes).items()) if count < 5],
        "sparse_radius_bins": [value for value, count in sorted(Counter(int(plane["radius_bin"]) for plane in planes).items()) if count < 5],
    }
    summary_path = report_dir / "all_vessel_group_sensitivity_summary.json"
    summary_path.write_text(json.dumps({
        "status": "complete", "family": family, **final_counts,
        "element_counts": counts,
        "metrics": list(ALL_VESSEL_METRICS),
        "pressure_included": False,
        "group_statistics": ["equal_vessel_mean", "area_weighted_mean", "median", "q25", "q75", "iqr", "p95", "maximum"],
        "difference_method": "paired anatomical vessel differences before group aggregation",
        "criteria_percent": [5.0, 1.0],
    }, indent=2) + "\n", encoding="utf-8")
    return summary_path


def run_boundary_layer_thickness_sensitivity(
    cfg: dict[str, Any], graph: dict[str, Any], resume: bool
) -> Path | None:
    """Compare 5/10/15%-of-radius prism envelopes at fixed global size."""
    output = Path(cfg["paths"]["output_dir"])
    definitions = (
        output / "plane_sensitivity" / "all_vessels" / "geometry"
        / "all_vessel_plane_definitions.json"
    )
    if not definitions.is_file():
        return None
    candidate_cases = {
        0.05: "blthick_global_l3_r050",
        0.10: "blthick_global_l3_r100",
        0.15: "global_l3",
    }
    for case in candidate_cases.values():
        if not (
            (output / "meshes" / case / "mesh_stats.json").is_file()
            and (output / "extracted" / case / (case + "_wall.npz")).is_file()
            and any((output / "cfx" / case).glob(case + "*.res"))
        ):
            return None
    raw = json.loads(definitions.read_text(encoding="utf-8"))
    planes = raw["planes"]
    for plane in planes:
        for key in ("midpoint_mm", "centre_mm", "normal", "loop_points_mm"):
            plane[key] = np.asarray(plane[key], dtype=float)
    root = output / "boundary_layer" / "thickness_sensitivity"
    cfx_root = root / "cfx"
    report_dir = root / "reports"
    cfx_root.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    post = cfg["executables"]["cfx_post"]
    settings = cfg["plane_sensitivity"]
    values: list[dict[str, Any]] = []
    for ratio, case in sorted(candidate_cases.items()):
        result = sorted(
            (output / "cfx" / case).glob(case + "*.res"),
            key=lambda path: ("continue" in path.stem, path.stat().st_mtime),
        )[-1]
        sections = extract_cfx_velocity_planes(
            post, result, planes,
            cfx_root / case / "all_vessel_velocity_sections.csv", resume,
        )
        section_lookup = {row["plane_id"]: row for row in sections}
        wall = wall_band_metrics(
            output / "extracted" / case / (case + "_wall.npz"),
            graph, planes,
            float(settings["wss_wall_band_length_local_diameters"]),
            strict=False,
        )
        stats = json.loads((
            output / "meshes" / case / "mesh_stats.json"
        ).read_text(encoding="utf-8"))
        for plane in planes:
            plane_id = plane["plane_id"]
            if plane_id not in wall or plane_id not in section_lookup:
                continue
            section = section_lookup[plane_id]
            values.append({
                "plane_id": plane_id,
                "edge_id": plane["edge_id"],
                "strahler_order": plane["strahler_order"],
                "radius_bin": plane["radius_bin"],
                "section_equivalent_radius_mm": plane["section_equivalent_radius_mm"],
                "maximum_channel_radius_ratio": ratio,
                "case_id": case,
                "elements": int(stats["total_elements"]),
                "global_h_mm": float(stats["global_h_mm"]),
                **wall[plane_id],
                "velocity_mean_m_s": section["velocity_mean_m_s"],
                "velocity_max_m_s": section["velocity_max_m_s"],
            })
    _write_csv(report_dir / "per_vessel_values.csv", values)
    lookup = {
        (row["plane_id"], float(row["maximum_channel_radius_ratio"])): row
        for row in values
    }
    common = sorted({row["plane_id"] for row in values if all(
        (row["plane_id"], ratio) in lookup for ratio in candidate_cases
    )})
    comparisons: list[dict[str, Any]] = []
    for plane_id in common:
        meta = lookup[(plane_id, 0.15)]
        for metric in ALL_VESSEL_METRICS:
            v05 = float(lookup[(plane_id, 0.05)][metric])
            v10 = float(lookup[(plane_id, 0.10)][metric])
            v15 = float(lookup[(plane_id, 0.15)][metric])
            comparisons.append({
                "plane_id": plane_id, "edge_id": meta["edge_id"],
                "strahler_order": meta["strahler_order"],
                "radius_bin": meta["radius_bin"],
                "section_equivalent_radius_mm": meta["section_equivalent_radius_mm"],
                "metric": metric,
                "ratio_005_value": v05, "ratio_010_value": v10,
                "ratio_015_value": v15,
                "r015_r010_percent": adjacent_percent(v15, v10),
                "r010_r005_percent": adjacent_percent(v10, v05),
                "r015_r005_percent": adjacent_percent(v15, v05),
            })
    _write_csv(report_dir / "per_vessel_thickness_comparison.csv", comparisons)
    group_rows: list[dict[str, Any]] = []
    for group_type, key in (
        ("radius_bin", "radius_bin"), ("strahler_order", "strahler_order")
    ):
        group_values = sorted({int(row[key]) for row in comparisons})
        for group_value in group_values:
            for metric in ALL_VESSEL_METRICS:
                subset = [
                    row for row in comparisons
                    if int(row[key]) == group_value and row["metric"] == metric
                ]
                if not subset:
                    continue
                for pair in ("r015_r010_percent", "r010_r005_percent", "r015_r005_percent"):
                    delta = np.asarray([float(row[pair]) for row in subset])
                    group_rows.append({
                        "group_type": group_type, "group_value": group_value,
                        "metric": metric, "comparison": pair,
                        "vessel_count": len(subset),
                        "median_percent": float(np.median(delta)),
                        "p95_percent": float(np.percentile(delta, 95.0)),
                        "maximum_percent": float(np.max(delta)),
                        "percent_vessels_passing_5_percent": 100.0 * float(np.mean(delta <= 5.0)),
                        "percent_vessels_passing_1_percent": 100.0 * float(np.mean(delta <= 1.0)),
                        "sparse_group": len(subset) < 5,
                    })
    _write_csv(report_dir / "group_thickness_comparison.csv", group_rows)
    distal = [
        row for row in group_rows
        if row["group_type"] == "radius_bin"
        and int(row["group_value"]) <= 3
        and row["metric"].startswith("wss")
        and row["comparison"] == "r010_r005_percent"
    ]
    report = {
        "status": "complete",
        "design": "five layers; identical global bulk/surface size; total prism thickness capped at 5%, 10%, or 15% of local radius",
        "common_vessels": len(common),
        "metrics": list(ALL_VESSEL_METRICS),
        "distal_wss_10_to_5_percent_pass": bool(distal) and all(
            float(row["p95_percent"]) <= 5.0 for row in distal
        ),
        "interpretation_rule": (
            "The 15%-radius boundary layer is implicated when distal WSS "
            "changes materially from 15% to 10% but stabilizes from 10% to "
            "5%; otherwise the poor global WSS convergence is more likely "
            "driven by tangential/surface/core resolution or local mesh quality."
        ),
        "per_vessel_values": str(report_dir / "per_vessel_values.csv"),
        "per_vessel_comparison": str(report_dir / "per_vessel_thickness_comparison.csv"),
        "group_comparison": str(report_dir / "group_thickness_comparison.csv"),
    }
    summary = report_dir / "boundary_layer_thickness_sensitivity.json"
    summary.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return summary


def run_plane_sensitivity(cfg: dict[str, Any], resume: bool = True) -> Path:
    import pyvista as pv
    output = Path(cfg["paths"]["output_dir"]); root = output / "plane_sensitivity"; geometry_dir = root / "geometry"; report_dir = root / "reports"
    root.mkdir(parents=True, exist_ok=True); geometry_dir.mkdir(exist_ok=True); report_dir.mkdir(exist_ok=True)
    settings = cfg["plane_sensitivity"]
    graph_path = output / "meshes" / "global_l4" / "cropped_amira_graph.json"
    graph = json.loads(graph_path.read_text(encoding="utf-8"))
    opening = json.loads((output / "poi" / "opening_terminal.json").read_text(encoding="utf-8"))
    bin_edges = opening.get("radius_bin_edges_mm") or opening["radius_bins_mm"]
    # Radius-bin annotations are derived analysis metadata and are not stored in
    # the shared cropped-graph contract.  Restore them on every resumed run so
    # the centreline VTK remains self-describing.
    for edge in graph["edges"]:
        edge["radius_bin"] = _radius_bin(_candidate(edge)["midpoint_radius_mm"], bin_edges)
    print("[planes] loading exact meshing STL", flush=True); surface = pv.read(cfg["paths"]["stl"]).triangulate().clean()
    manifest_json = geometry_dir / "plane_definitions.json"
    if manifest_json.exists() and resume:
        raw = json.loads(manifest_json.read_text(encoding="utf-8")); strahler, radius = raw["strahler"], raw["radius_bin"]
        for plane in strahler + radius:
            plane["midpoint_mm"] = np.asarray(plane["midpoint_mm"]); plane["centre_mm"] = np.asarray(plane["centre_mm"]); plane["normal"] = np.asarray(plane["normal"]); plane["loop_points_mm"] = np.asarray(plane["loop_points_mm"])
            plane.setdefault("section_equivalent_radius_mm", math.sqrt(float(plane["area_mm2"]) / math.pi))
    else:
        print("[planes] selecting and validating representative sections", flush=True)
        try:
            strahler, radius, diagnostics = select_planes(graph, surface, bin_edges, settings)
        except PlaneSelectionFailure as exc:
            (geometry_dir / "plane_selection_diagnostics.json").write_text(json.dumps(exc.diagnostics, indent=2), encoding="utf-8")
            write_selection_diagnostic(surface, exc.diagnostics, geometry_dir / "plane_selection_diagnostics.vtk")
            raise
        serial = lambda p: {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in p.items() if k != "edge"}
        manifest_json.write_text(json.dumps({"strahler": [serial(p) for p in strahler], "radius_bin": [serial(p) for p in radius], "settings": settings}, indent=2) + "\n", encoding="utf-8")
        (geometry_dir / "plane_selection_diagnostics.json").write_text(json.dumps(diagnostics, indent=2) + "\n", encoding="utf-8")
    all_planes = strahler + radius
    _write_csv(geometry_dir / "strahler_plane_definitions.csv", [plane_manifest_row(p) for p in strahler]); _write_csv(geometry_dir / "radius_bin_plane_definitions.csv", [plane_manifest_row(p) for p in radius])
    package_files = [geometry_dir / name for name in ("strahler_planes.vtk", "strahler_planes.vtm", "radius_bin_planes.vtk", "radius_bin_planes.vtm")]
    package_contract = geometry_dir / "geometry_package_contract.json"
    expected_contract = {"version": 4, "strahler_plane_count": len(strahler), "radius_bin_plane_count": len(radius), "visible_plane_shape": "square", "display_scale": 2.5, "minimum_display_half_width_mm": 0.75}
    contract_matches = False
    if package_contract.exists():
        try: contract_matches = json.loads(package_contract.read_text(encoding="utf-8")) == expected_contract
        except (OSError, ValueError): pass
    if resume and contract_matches and all(path.exists() for path in package_files):
        print("[planes] reusing validated geometry VTK packages", flush=True)
    else:
        print("[planes] writing exact STL/centreline/section VTK packages", flush=True)
        write_geometry_packages(surface, graph, {"strahler_planes": strahler, "radius_bin_planes": radius}, geometry_dir)
        package_contract.write_text(json.dumps(expected_contract, indent=2) + "\n", encoding="utf-8")
    post = cfg["executables"]["cfx_post"]; case_values = []; diagnostic_values = []; hotspot_values = []; hotspot_plane_values = []
    counts = {}; pressure_normalizations = {}
    for level in LEVELS:
        case = "global_" + level; case_dir = root / "cfx" / case
        stats = json.loads((output / "meshes" / case / "mesh_stats.json").read_text(encoding="utf-8")); counts[level] = int(stats["total_elements"])
        assignment = json.loads((output / "extracted" / case / "metric_assignment.json").read_text(encoding="utf-8"))
        pressure_normalizations[level] = float(assignment["pressure_normalization_pa"])
        res_candidates = sorted((output / "cfx" / case).glob(case + "*.res"), key=lambda p: ("continue" in p.stem, p.stat().st_mtime))
        if not res_candidates: raise FileNotFoundError("No result for " + case)
        res = res_candidates[-1]
        print("[planes] CFD-Post {} ({:,} elements)".format(case, counts[level]), flush=True)
        cfx_rows = extract_cfx_planes(post, res, all_planes, case_dir / "cross_section_metrics.csv", resume)
        domain_rows = extract_cfx_domain_diagnostics(post, res, case_dir / "courant_mesh_quality.csv", resume)
        for row in domain_rows:
            diagnostic = {"level": level, "case_id": case, "elements": counts[level], **row}
            if row["variable"] == "orthogonality_angle_rad":
                diagnostic.update({
                    "minimum_degrees": math.degrees(row["minimum"]),
                    "volume_average_degrees": math.degrees(row["volume_average"]),
                    "maximum_degrees": math.degrees(row["maximum"]),
                })
            else:
                diagnostic.update({"minimum_degrees": "", "volume_average_degrees": "", "maximum_degrees": ""})
            diagnostic_values.append(diagnostic)
        case_hotspots, case_hotspot_planes = extract_cfx_hotspots(post, res, case_dir / "quality_hotspots", all_planes, level, resume)
        hotspot_values.extend(case_hotspots)
        hotspot_plane_values.extend(case_hotspot_planes)
        wall = wall_band_metrics(output / "extracted" / case / (case + "_wall.npz"), graph, all_planes, float(settings["wss_wall_band_length_local_diameters"]))
        inlet_pressure = _read_inlet_pressure(output / "extracted" / case / "metrics.csv")
        by_id = {r["plane_id"]: r for r in cfx_rows}
        for plane in all_planes:
            cfx = by_id[plane["plane_id"]]; stl_area_m2 = plane["area_mm2"] * 1e-6
            area_error = abs(cfx["area_m2"] - stl_area_m2) / stl_area_m2
            row = {"plane_id": plane["plane_id"], "category": plane["category"], "category_value": plane["category_value"], "level": level, "elements": counts[level], **wall[plane["plane_id"]], "velocity_mean_m_s": cfx["velocity_mean_m_s"], "velocity_max_m_s": cfx["velocity_max_m_s"], "pressure_mean_pa": cfx["pressure_mean_pa"], "pressure_drop_pa": inlet_pressure - cfx["pressure_mean_pa"], "cfx_area_m2": cfx["area_m2"], "stl_area_m2": stl_area_m2, "area_relative_error": area_error}
            if not (row["velocity_mean_m_s"] <= row["velocity_max_m_s"] + 1e-12): raise RuntimeError("Mean velocity exceeds maximum at " + plane["plane_id"])
            case_values.append(row)
    _write_csv(report_dir / "per_plane_values.csv", case_values)
    _write_csv(report_dir / "cfx_courant_mesh_quality.csv", diagnostic_values)
    _cfx_diagnostic_figure(report_dir, diagnostic_values, counts)
    _write_csv(report_dir / "cfx_quality_hotspot_proximity.csv", hotspot_values)
    _hotspot_proximity_figure(report_dir, hotspot_values)
    _write_csv(report_dir / "cfx_quality_hotspot_plane_matrix.csv", hotspot_plane_values)
    _hotspot_plane_matrix_figures(report_dir, hotspot_plane_values)
    safety = float(settings["gci_safety_factor"]); convergence = []
    lookup = {(r["plane_id"], r["level"]): r for r in case_values}
    for plane in all_planes:
        for metric in METRICS:
            vals = [float(lookup[(plane["plane_id"], level)][metric]) for level in LEVELS]
            pressure_metric = metric.startswith("pressure")
            gci = three_grid_gci(
                vals, [counts[level] for level in LEVELS], safety,
                pressure_normalizations["l4"] if pressure_metric else None,
            )
            tolerance = float(settings["pressure_tolerance_fraction"] if metric.startswith("pressure") else settings["wss_velocity_tolerance_fraction"])
            differences = (
                [normalized_difference_percent(vals[i], vals[i + 1], pressure_normalizations[LEVELS[i + 1]]) for i in range(3)]
                if pressure_metric else [adjacent_percent(vals[i], vals[i + 1]) for i in range(3)]
            )
            row = {"plane_id": plane["plane_id"], "category": plane["category"], "category_value": plane["category_value"], "metric": metric, **{level + "_value": value for level, value in zip(LEVELS, vals)}, "l2_l1_percent": differences[0], "l3_l2_percent": differences[1], "l4_l3_percent": differences[2], "difference_normalization": "inlet_to_opening_pressure_drop" if pressure_metric else "fine_mesh_metric_magnitude", **gci, "tolerance_percent": 100 * tolerance, "pass": differences[2] <= 100 * tolerance}
            convergence.append(row)
    _write_csv(report_dir / "adjacent_differences_gci.csv", convergence)
    overlap_rows = []
    sensitivity_metrics = {"wss_mean_pa", "wss_max_pa", "velocity_mean_m_s", "velocity_max_m_s"}
    for result in convergence:
        if result["metric"] not in sensitivity_metrics:
            continue
        nearby_by_level = {}
        for comparison_level in ("l3", "l4"):
            nearby = [
                row["hotspot"] for row in hotspot_plane_values
                if row["level"] == comparison_level
                and row["plane_id"] == result["plane_id"]
                and float(row["minimum_disk_distance_diameters"]) <= 0.5
            ]
            nearby_by_level[comparison_level] = ";".join(sorted(nearby))
        overlap_rows.append({
            "plane_id": result["plane_id"], "category": result["category"],
            "metric": result["metric"], "l4_l3_percent": result["l4_l3_percent"],
            "passes_5_percent": float(result["l4_l3_percent"]) <= 5.0,
            "l3_hotspots_within_0.5_diameters": nearby_by_level["l3"],
            "l4_hotspots_within_0.5_diameters": nearby_by_level["l4"],
            "quality_hotspot_overlap": bool(nearby_by_level["l3"] or nearby_by_level["l4"]),
        })
    _write_csv(report_dir / "quality_hotspot_convergence_overlap.csv", overlap_rows)
    plane_pass = []
    for plane in all_planes:
        relevant = [r for r in convergence if r["plane_id"] == plane["plane_id"]]
        plane_pass.append({"plane_id": plane["plane_id"], "category": plane["category"], "category_value": plane["category_value"], "pass": all(r["pass"] for r in relevant), "failed_metrics": ";".join(r["metric"] for r in relevant if not r["pass"])})
    _write_csv(report_dir / "plane_pass_fail.csv", plane_pass)
    for stem, planes in (("strahler", strahler), ("radius_bin", radius)):
        ids = {p["plane_id"] for p in planes}; _write_csv(report_dir / (stem + "_values.csv"), [r for r in case_values if r["plane_id"] in ids]); _write_csv(report_dir / (stem + "_gci.csv"), [r for r in convergence if r["plane_id"] in ids]); _plots_and_tables(report_dir, stem, planes, [r for r in case_values if r["plane_id"] in ids], [r for r in convergence if r["plane_id"] in ids], counts)
    _velocity_wss_criterion_figures(report_dir, convergence)
    all_vessel_summary = run_all_vessel_group_sensitivity(
        cfg, graph, surface, bin_edges, settings, counts, resume,
        family="global",
    )
    adaptive_summary = None
    adaptive_counts = {}
    adaptive_ready = True
    for level in LEVELS:
        case = "adaptive_" + level
        stats_path = output / "meshes" / case / "mesh_stats.json"
        result_dir = output / "cfx" / case
        wall_path = output / "extracted" / case / (case + "_wall.npz")
        if (
            not stats_path.is_file() or not wall_path.is_file()
            or not any(result_dir.glob(case + "*.res"))
        ):
            adaptive_ready = False
            break
        adaptive_counts[level] = int(json.loads(
            stats_path.read_text(encoding="utf-8")
        )["total_elements"])
    if adaptive_ready:
        adaptive_summary = run_all_vessel_group_sensitivity(
            cfg, graph, surface, bin_edges, settings, adaptive_counts, resume,
            family="adaptive",
        )
    thickness_summary = run_boundary_layer_thickness_sensitivity(
        cfg, graph, resume
    )
    area_errors = [float(row["area_relative_error"]) for row in case_values]
    summary = {"status": "pass" if all(r["pass"] for r in plane_pass) else "non_converged", "plane_counts": {"strahler": len(strahler), "radius_bin": len(radius)}, "element_counts": counts, "pressure_normalizations_pa": pressure_normalizations, "maximum_cfx_to_stl_area_error_fraction": max(area_errors), "finest_cfx_to_stl_area_error_fraction": max(float(row["area_relative_error"]) for row in case_values if row["level"] == "l4"), "settings": settings, "pressure_reference": "existing area-weighted inlet POI_001 cross-section; changes normalized by each fine mesh's inlet-to-opening pressure drop", "steady_quantity_note": "Steady WSS and velocity are not labelled TAWSS or peak-systolic.", "all_vessel_group_analysis": str(all_vessel_summary), "adaptive_all_vessel_group_analysis": str(adaptive_summary) if adaptive_summary else None, "boundary_layer_thickness_analysis": str(thickness_summary) if thickness_summary else None, "planes": plane_pass}
    summary_path = report_dir / "plane_sensitivity_summary.json"; summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary_path
