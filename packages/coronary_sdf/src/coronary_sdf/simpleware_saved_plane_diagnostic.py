"""Validate persisted Simpleware finite-plane transforms against Amira/STL.

The optional third argument writes a CSV report. X-2025.06 finite-plane
``scale`` values are local half extents. The normal check uses those persisted
values; the double-extent check is an additional conservative stress test.
"""

import csv
import math
import re
import sys
import zipfile
import xml.etree.ElementTree as ET

import numpy as np

import simpleware_coronary_regions as regions


def _rotation(axis, degrees):
    axis = np.asarray(axis, dtype=float)
    axis /= np.linalg.norm(axis)
    angle = math.radians(float(degrees))
    skew = np.array([
        [0.0, -axis[2], axis[1]],
        [axis[2], 0.0, -axis[0]],
        [-axis[1], axis[0], 0.0],
    ])
    return (
        np.eye(3) * math.cos(angle)
        + (1.0 - math.cos(angle)) * np.outer(axis, axis)
        + math.sin(angle) * skew
    )


def _finite_plane_status(solid, item, graph_radius, extent_multiplier=1.0):
    search_radius = max(
        float(regions.STL_PLANE_NEIGHBOUR_SEARCH_MIN_RADIUS_MM),
        8.0 * float(graph_radius),
    )
    triangles = solid.nearby_triangles(item["centre"], search_radius)
    segments = regions._triangle_plane_segments(
        triangles, item["centre"], item["normal"]
    )
    loops, open_components = regions._ordered_plane_intersection_loops(
        segments, item["centre"], item["normal"]
    )
    containing = [
        (index, loop) for index, loop in enumerate(loops)
        if regions._point_in_polygon((0.0, 0.0), loop)
    ]
    if len(containing) != 1:
        return False, float("inf"), -1, "containing_loops={}".format(len(containing))

    target_index, target = containing[0]
    auto_first, auto_second = regions._plane_basis(item["normal"])

    def saved_uv(component):
        world = (
            component[:, 0, None] * auto_first[None, :]
            + component[:, 1, None] * auto_second[None, :]
        )
        return np.column_stack((world.dot(item["first"]), world.dot(item["second"])))

    target_uv = saved_uv(target)
    half_x = float(item["half_x"]) * float(extent_multiplier)
    half_y = float(item["half_y"]) * float(extent_multiplier)
    ratios = np.column_stack((
        np.abs(target_uv[:, 0]) / max(half_x, 1.0e-12),
        np.abs(target_uv[:, 1]) / max(half_y, 1.0e-12),
    ))
    coverage_ratio = float(np.max(ratios))
    collisions = 0
    for index, loop in enumerate(loops):
        if index == target_index:
            continue
        if regions._polyline_intersects_rectangle(
            saved_uv(loop), half_x, half_y
        ):
            collisions += 1
    for component in open_components:
        if regions._polyline_intersects_rectangle(
            saved_uv(component), half_x, half_y
        ):
            collisions += 1
    return coverage_ratio <= 1.0 + 1.0e-6 and collisions == 0, coverage_ratio, collisions, None


def _local_polyline_tangent(raw, query, window):
    """Return outward tangent averaged around the nearest polyline location."""
    raw = np.asarray(raw, dtype=float)
    query = np.asarray(query, dtype=float)
    cumulative = regions._cumulative_distances(raw)
    best_distance = float("inf")
    best_arc = 0.0
    for index, (start, end) in enumerate(zip(raw[:-1], raw[1:])):
        delta = end - start
        length2 = float(np.dot(delta, delta))
        fraction = 0.0 if length2 <= 1.0e-15 else float(np.clip(
            np.dot(query - start, delta) / length2, 0.0, 1.0
        ))
        nearest = start + fraction * delta
        distance = float(np.linalg.norm(query - nearest))
        if distance < best_distance:
            best_distance = distance
            best_arc = float(
                cumulative[index] + fraction * (cumulative[index + 1] - cumulative[index])
            )
    half_window = 0.5 * min(float(window), float(cumulative[-1]))
    outward_point = regions._polyline_position(
        raw, cumulative, max(0.0, best_arc - half_window)
    )
    inward_point = regions._polyline_position(
        raw, cumulative, min(float(cumulative[-1]), best_arc + half_window)
    )
    return regions._unit(outward_point - inward_point), best_arc, best_distance


def _plane_crossing_arc(points, cumulative, centre, normal):
    """Return the first arc position where a plane crosses a terminal path."""
    points = np.asarray(points, dtype=float)
    signed = (points - np.asarray(centre, dtype=float)).dot(normal)
    for index in range(len(points) - 1):
        first = float(signed[index])
        second = float(signed[index + 1])
        if first == 0.0:
            return float(cumulative[index])
        if first * second <= 0.0:
            denominator = first - second
            fraction = 0.0 if abs(denominator) <= 1.0e-15 else first / denominator
            return float(
                cumulative[index]
                + fraction * (cumulative[index + 1] - cumulative[index])
            )
    return float(cumulative[int(np.argmin(np.abs(signed)))])


def main(project, surface_path=None, csv_path=None):
    if surface_path:
        regions.STL_SURFACE_PATH = surface_path
    with zipfile.ZipFile(project) as archive:
        root = ET.fromstring(archive.read("RegionsOfInterest.xml"))
    saved = {}
    for roi in root.findall("./CLIPPING/RegionOfInterestVolume"):
        name = roi.attrib.get("name", "")
        if not name.startswith("COR_OUTLET_"):
            continue
        shape = roi.find("FinitePlane")
        axis = [float(shape.attrib[key]) for key in ("ax", "ay", "az")]
        matrix = _rotation(axis, float(shape.attrib["angle"]))
        match = re.search(r"(AmiraNode_\d+)$", name)
        if not match:
            continue
        saved[match.group(1)] = {
            "name": name,
            "centre": np.array([float(shape.attrib[key]) for key in ("cx", "cy", "cz")]),
            "first": matrix[:, 0],
            "second": matrix[:, 1],
            "normal": matrix[:, 2],
            "half_x": float(shape.attrib["sx"]),
            "half_y": float(shape.attrib["sy"]),
        }

    solid = regions._load_stl_solid()
    network = regions._crop_amira_network_to_stl(regions._load_amira_network(), solid)
    records = {
        record["node_name"]: record
        for record in regions._find_terminals(network, None, "amira")
    }
    results = []
    for node_name, item in saved.items():
        record = records.get(node_name)
        graph_radius = max(item["half_x"], item["half_y"])
        tangent_cosine = float("nan")
        centre_to_terminal = float("nan")
        normal = item["normal"]
        if record is not None:
            raw = np.asarray(record["terminal_raw_points"], dtype=float)
            cumulative = regions._cumulative_distances(raw)
            inward = regions._polyline_position(raw, cumulative, min(0.25, cumulative[-1]))
            local_outward = regions._unit(raw[0] - inward)
            if np.dot(normal, local_outward) < 0.0:
                normal = -normal
            tangent_cosine = float(np.dot(normal, local_outward))
            graph_radius = float(record["radius"])
            centre_to_terminal = float(np.linalg.norm(item["centre"] - raw[0]))
        exact, reason = regions._validate_stl_plane_intersection(
            solid, item["centre"], normal, graph_radius
        )
        finite_valid, coverage_ratio, collisions, finite_reason = (
            _finite_plane_status(solid, item, graph_radius)
        )
        double_valid, double_coverage, double_collisions, _double_reason = (
            _finite_plane_status(solid, item, graph_radius, extent_multiplier=2.0)
        )
        local_tangent_cosine = float("nan")
        local_tangent_arc = float("nan")
        centreline_offset = float("nan")
        surface_to_graph_radius_ratio = float("nan")
        if record is not None:
            anchor, _outward, _extension = regions._terminal_surface_anchor(
                record, solid
            )
            raw = np.asarray(record["terminal_raw_points"], dtype=float)
            path = np.vstack((anchor, raw))
            keep = np.concatenate((
                [True], np.linalg.norm(np.diff(path, axis=0), axis=1) > regions.EPS
            ))
            path = path[keep]
            path_cumulative = regions._cumulative_distances(path)
            local_tangent_arc = _plane_crossing_arc(
                path, path_cumulative, item["centre"], item["normal"]
            )
            local_tangent = regions._local_averaged_outward_tangent(
                path,
                path_cumulative,
                local_tangent_arc,
                regions.STL_PLANE_TANGENT_AVERAGING_LENGTH_MM,
            )
            local_tangent_cosine = abs(float(np.dot(item["normal"], local_tangent)))
            plane_point = regions._polyline_position(
                path, path_cumulative, local_tangent_arc
            )
            centreline_offset = float(np.linalg.norm(
                (item["centre"] - plane_point)
                - np.dot(item["centre"] - plane_point, item["normal"])
                * item["normal"]
            ))
            if exact is not None:
                surface_to_graph_radius_ratio = float(
                    exact["surface_loop_radius"] / max(float(record["radius"]), 1.0e-12)
                )
        persisted_normal_cosine = float("nan")
        persisted_centre_error = float("nan")
        cap_graph_deviation_degrees = float("nan")
        cap_planarity_error_mm = float("nan")
        adaptive_inset_mm = float("nan")
        terminal_branch_retained_fraction = float("nan")
        short_branch_limited = False
        normal_source = "unmatched"
        if record is not None:
            try:
                regions._search_stl_validated_terminal_plane(record, solid)
            except RuntimeError:
                pass
            else:
                persisted_normal_cosine = abs(float(np.dot(
                    item["normal"], record["normal"]
                )))
                persisted_centre_error = float(np.linalg.norm(
                    item["centre"] - record["centre"]
                ))
                cap_graph_deviation_degrees = float(
                    record.get("stl_cap_graph_deviation_degrees", float("nan"))
                )
                cap_planarity_error_mm = float(
                    record.get("stl_cap_planarity_error_mm", float("nan"))
                )
                normal_source = record.get("plane_normal_source", "unknown")
                adaptive_inset_mm = float(
                    record.get("surface_plane_inset_mm", float("nan"))
                )
                terminal_branch_retained_fraction = 1.0 - float(
                    record.get("surface_plane_removed_branch_fraction", float("nan"))
                )
                short_branch_limited = bool(
                    record.get("surface_plane_short_branch_limited", False)
                )
        ordinal_match = re.match(r"COR_OUTLET_(\d+)_", item["name"])
        results.append({
            "contact_name": item["name"],
            "plane_label": (
                "P{:03d}".format(int(ordinal_match.group(1)))
                if ordinal_match else ""
            ),
            "tangent_cosine": tangent_cosine,
            "centre_to_terminal": centre_to_terminal,
            "node_name": node_name,
            "exact_valid": exact is not None,
            "reason": reason,
            "finite_valid": finite_valid,
            "coverage_ratio": coverage_ratio,
            "collisions": collisions,
            "double_extent_valid": double_valid,
            "double_extent_coverage_ratio": double_coverage,
            "double_extent_collisions": double_collisions,
            "finite_reason": finite_reason,
            "local_tangent_cosine": local_tangent_cosine,
            "local_tangent_arc_mm": local_tangent_arc,
            "centreline_offset_mm": centreline_offset,
            "surface_to_graph_radius_ratio": surface_to_graph_radius_ratio,
            "persisted_normal_cosine": persisted_normal_cosine,
            "persisted_centre_error": persisted_centre_error,
            "normal_source": normal_source,
            "cap_graph_deviation_degrees": cap_graph_deviation_degrees,
            "cap_planarity_error_mm": cap_planarity_error_mm,
            "adaptive_inset_mm": adaptive_inset_mm,
            "terminal_branch_retained_fraction": (
                terminal_branch_retained_fraction
            ),
            "short_branch_limited": short_branch_limited,
        })

    print("SAVED_PLANES", len(saved), "MATCHED", len(results))
    print("EXACT_VALID", sum(item["exact_valid"] for item in results))
    print("FINITE_VALID", sum(item["finite_valid"] for item in results))
    print(
        "DOUBLE_EXTENT_VALID",
        sum(item["double_extent_valid"] for item in results),
        "COLLIDING",
        sum(item["double_extent_collisions"] > 0 for item in results),
    )
    matched = [item for item in results if math.isfinite(item["tangent_cosine"])]
    if matched:
        print("TANGENT_COS_RANGE", min(item["tangent_cosine"] for item in matched), max(item["tangent_cosine"] for item in matched))
        print("CENTRE_TO_TERMINAL_RANGE", min(item["centre_to_terminal"] for item in matched), max(item["centre_to_terminal"] for item in matched))
        persisted = [
            item for item in matched
            if math.isfinite(item["persisted_normal_cosine"])
        ]
        if persisted:
            print("PERSISTED_NORMAL_COS_RANGE", min(item["persisted_normal_cosine"] for item in persisted), max(item["persisted_normal_cosine"] for item in persisted))
            print("PERSISTED_CENTRE_ERROR_RANGE", min(item["persisted_centre_error"] for item in persisted), max(item["persisted_centre_error"] for item in persisted))
    print("WORST_TANGENT")
    for item in sorted(matched, key=lambda value: value["local_tangent_cosine"])[:15]:
        print(item)
    print("INVALID")
    for item in results:
        if not item["exact_valid"] or not item["finite_valid"]:
            print(item)
    print("DOUBLE_EXTENT_COLLISIONS")
    for item in results:
        if item["double_extent_collisions"]:
            print(item)
    if csv_path:
        fieldnames = list(results[0]) if results else []
        with open(csv_path, "w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)
        print("CSV", csv_path)


if __name__ == "__main__":
    main(
        sys.argv[1],
        sys.argv[2] if len(sys.argv) > 2 else None,
        sys.argv[3] if len(sys.argv) > 3 else None,
    )
