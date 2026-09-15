"""Resumable radius-aware coronary mesh-sensitivity study orchestration.

The module keeps geometry preparation, numerical calibration and convergence
math deterministic and testable outside Simpleware/CFX.  Expensive applications
are launched as isolated jobs and leave stable JSON status files for ``--resume``.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import heapq
import json
import math
import re
import shutil
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
from scipy.spatial import cKDTree

try:
    from .parse_amira import parse_graph
except ImportError:  # direct invocation from the repository directory
    repository_parent = str(Path(__file__).resolve().parent.parent)
    if repository_parent not in sys.path:
        sys.path.insert(0, repository_parent)
    from coronary_sdf.parse_amira import parse_graph


APPROVAL_REQUIRED_EXIT = 3
CASE_LEVELS = ("l1", "l2", "l3", "l4")


class InfeasibleCoarseTarget(RuntimeError):
    def __init__(self, measured_minimum: int):
        super().__init__("Fixed boundary layer prevents the requested coarse target")
        self.measured_minimum = int(measured_minimum)


def load_manifest(path: str | Path) -> tuple[dict[str, Any], Path]:
    manifest_path = Path(path).resolve()
    cfg = json.loads(manifest_path.read_text(encoding="utf-8"))
    required = ("paths", "targets", "boundary_layer", "adaptive", "cfx")
    missing = [key for key in required if key not in cfg]
    if missing:
        raise ValueError("Study manifest is missing: {}".format(", ".join(missing)))
    base = manifest_path.parent
    for key in ("source_sip", "stl", "amira", "cfx_seed"):
        p = Path(cfg["paths"][key])
        if not p.is_absolute():
            p = base / p
        cfg["paths"][key] = str(p.resolve())
    if cfg["paths"].get("pre_import_stl"):
        p = Path(cfg["paths"]["pre_import_stl"])
        if not p.is_absolute():
            p = base / p
        cfg["paths"]["pre_import_stl"] = str(p.resolve())
    out = Path(cfg["paths"].get("output_dir", "mesh_sensitivity_output"))
    if not out.is_absolute():
        out = base / out
    cfg["paths"]["output_dir"] = str(out.resolve())
    return cfg, manifest_path


def validate_manifest(cfg: dict[str, Any], require_apps: bool = False) -> None:
    for key in ("source_sip", "stl", "amira", "cfx_seed"):
        if not Path(cfg["paths"][key]).is_file():
            raise FileNotFoundError("{} not found: {}".format(key, cfg["paths"][key]))
    if cfg["paths"].get("pre_import_stl") and not Path(cfg["paths"]["pre_import_stl"]).is_file():
        raise FileNotFoundError("pre_import_stl not found: {}".format(cfg["paths"]["pre_import_stl"]))
    targets = [int(v) for v in cfg["targets"]["element_counts"]]
    if len(targets) != 4 or any(v <= 0 for v in targets):
        raise ValueError("targets.element_counts must contain four positive counts")
    if targets != sorted(targets):
        raise ValueError("targets.element_counts must increase")
    bl = cfg["boundary_layer"]
    if [int(v) for v in bl["candidate_layers"]] != [4, 6, 8]:
        raise ValueError("boundary_layer.candidate_layers must be [4, 6, 8]")
    if float(bl["growth_ratio"]) < 1.0:
        raise ValueError("boundary_layer.growth_ratio must be at least 1")
    if not (0 < float(bl["maximum_channel_radius_ratio"]) <= 1):
        raise ValueError("maximum_channel_radius_ratio must be in (0,1]")
    adaptive = cfg["adaptive"]
    if not (0 < float(adaptive["min_h_mm"]) <= float(adaptive["max_h_mm"])):
        raise ValueError("adaptive min_h_mm/max_h_mm are invalid")
    if float(adaptive["maximum_sphere_expansion"]) > 1.35:
        raise ValueError("maximum_sphere_expansion cannot exceed validated 1.35")
    if "plane_sensitivity" in cfg:
        plane = cfg["plane_sensitivity"]
        required_plane = (
            "tangent_averaging_window_local_diameters",
            "minimum_junction_clearance_local_diameters",
            "wss_wall_band_length_local_diameters",
            "plane_coverage_margin_fraction",
            "gci_safety_factor",
            "wss_velocity_tolerance_fraction",
            "pressure_tolerance_fraction",
        )
        missing_plane = [key for key in required_plane if key not in plane]
        if missing_plane:
            raise ValueError("plane_sensitivity is missing: {}".format(", ".join(missing_plane)))
        if float(plane["minimum_junction_clearance_local_diameters"]) < 2.0:
            raise ValueError("plane_sensitivity junction clearance must be at least two diameters")
        if float(plane["plane_coverage_margin_fraction"]) < 0.0:
            raise ValueError("plane coverage margin cannot be negative")
    if require_apps:
        for key in ("simpleware_console", "cfx_pre", "cfx_solver", "cfx_post"):
            if not Path(cfg["executables"][key]).is_file():
                raise FileNotFoundError("{} not found".format(key))


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def extract_cfx_seed_contract(path: str | Path) -> dict[str, Any]:
    """Extract the immutable physics controls embedded in a binary ``.cfx``.

    CFX project files contain their CCL as NUL-padded ASCII.  Reading that CCL
    directly makes the audit independent of a licensed CFX-Pre process and, in
    particular, lets us reject a zero/placeholder inlet before starting the
    expensive mesh matrix.
    """
    seed = Path(path)
    raw = seed.read_bytes().replace(b"\x00", b" ").decode("latin1", "ignore")
    compact = " ".join(raw.split())

    def capture(pattern: str, label: str, cast=str):
        match = re.search(pattern, compact, re.IGNORECASE)
        if not match:
            raise RuntimeError("CFX seed does not expose {} in embedded CCL".format(label))
        return cast(match.group(1))

    inlet_block = capture(
        r"(BOUNDARY:\s*Inlet_000\b.*?)(?=\s+BOUNDARY:|\s+END\s+END\s+END)",
        "Inlet_000 boundary",
    )
    inlet_match = re.search(
        r"Mass Flow Rate\s*=\s*([+\-0-9.eE]+)\s*\[kg\s*s\^-1\]",
        inlet_block,
        re.IGNORECASE,
    )
    if not inlet_match:
        raise RuntimeError("CFX seed Inlet_000 has no numeric kg/s mass flow")
    inlet_mass = float(inlet_match.group(1))
    if not np.isfinite(inlet_mass) or inlet_mass <= 0:
        raise RuntimeError("CFX seed inlet mass flow must be positive")

    contract = {
        "inlet_boundary": "Inlet_000",
        "inlet_mass_flow_kg_s": inlet_mass,
        "material_model": "Quemada" if re.search(r"MATERIAL:\s*Quemada\b", compact, re.I) else "",
        "turbulence_option": capture(
            r"TURBULENCE MODEL:\s*.*?Option\s*=\s*([^\s].*?)(?=\s+END)",
            "turbulence model",
        ),
        "residual_target": capture(r"Residual Target\s*=\s*([+\-0-9.eE]+)", "residual target", float),
        "residual_type": capture(r"Residual Type\s*=\s*([^\s].*?)(?=\s+END|\s+[A-Z][A-Za-z ]+\s*=)", "residual type"),
        "minimum_iterations": capture(r"Minimum Number of Iterations\s*=\s*(\d+)", "minimum iterations", int),
        "maximum_iterations": capture(r"Maximum Number of Iterations\s*=\s*(\d+)", "maximum iterations", int),
        "timescale_control": capture(r"Timescale Control\s*=\s*([^\s].*?)(?=\s+Timescale Factor|\s+END)", "timescale control"),
        "timescale_factor": capture(r"Timescale Factor\s*=\s*([+\-0-9.eE]+)", "timescale factor", float),
    }
    if contract["material_model"] != "Quemada":
        raise RuntimeError("CFX seed does not contain the authoritative Quemada material")
    canonical = json.dumps(contract, sort_keys=True, separators=(",", ":"))
    contract["physics_control_sha256"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    contract["seed_sha256"] = file_sha256(seed)
    return contract


def validate_cfx_seed_contract(cfg: dict[str, Any], write_to: Path | None = None) -> dict[str, Any]:
    contract = extract_cfx_seed_contract(cfg["paths"]["cfx_seed"])
    expected_hash = cfg["cfx"].get("expected_seed_sha256")
    expected_physics = cfg["cfx"].get("expected_physics_control_sha256")
    expected_flow = cfg["cfx"].get("expected_inlet_mass_flow_kg_s")
    if expected_hash and contract["seed_sha256"].lower() != str(expected_hash).lower():
        raise RuntimeError("Immutable CFX seed SHA-256 changed")
    if expected_physics and contract["physics_control_sha256"].lower() != str(expected_physics).lower():
        raise RuntimeError("CFX seed physics-control contract changed")
    if expected_flow is None:
        raise RuntimeError("cfx.expected_inlet_mass_flow_kg_s must contain the audited allometric value")
    if not math.isclose(
        contract["inlet_mass_flow_kg_s"], float(expected_flow), rel_tol=1e-10, abs_tol=1e-15
    ):
        raise RuntimeError("CFX seed allometric inlet mass flow changed")
    if float(expected_flow) in (0.0, 0.001):
        raise RuntimeError("Refusing a zero or known placeholder inlet mass flow")
    if not math.isclose(
        contract["residual_target"], float(cfg["cfx"]["residual_target"]),
        rel_tol=1e-10, abs_tol=1e-15,
    ):
        raise RuntimeError("CFX seed residual target changed")
    if write_to is not None:
        write_to.parent.mkdir(parents=True, exist_ok=True)
        write_to.write_text(json.dumps(contract, indent=2) + "\n", encoding="utf-8")
    return contract


def local_boundary_layer_thickness(
    radius_mm: float, requested_mm: float = 0.20, maximum_ratio: float = 0.15
) -> float:
    return min(float(requested_mm), float(maximum_ratio) * float(radius_mm))


def simpleware_width_ratio(num_layers: int, growth_ratio: float) -> float:
    """Convert a consecutive-layer growth factor to Simpleware's width ratio.

    RatioSlicing expects the thinnest-to-thickest layer-width ratio (0, 1],
    whereas the study manifest specifies the conventional consecutive-layer
    growth factor (>= 1).  For geometrically growing layers, the first and last
    widths differ by growth_ratio ** (num_layers - 1).
    """
    num_layers = int(num_layers)
    growth_ratio = float(growth_ratio)
    if num_layers < 2:
        raise ValueError("num_layers must be at least 2 for ratio slicing")
    if not math.isfinite(growth_ratio) or growth_ratio < 1.0:
        raise ValueError("growth_ratio must be finite and at least 1")
    return growth_ratio ** (1 - num_layers)


def radius_mesh_size(
    radius_mm: float, n_d: float, minimum_mm: float = 0.02,
    maximum_mm: float = 0.40,
) -> float:
    if radius_mm <= 0 or n_d <= 0:
        raise ValueError("radius_mm and n_d must be positive")
    return float(np.clip(2.0 * radius_mm / n_d, minimum_mm, maximum_mm))


def calibrated_size(old_h: float, old_count: int, target_count: int) -> float:
    if min(old_h, old_count, target_count) <= 0:
        raise ValueError("calibration inputs must be positive")
    return float(old_h) * (float(old_count) / float(target_count)) ** (1.0 / 3.0)


def log_log_size_fit(trials: Sequence[tuple[float, int]], target: int) -> float:
    """Fit log(N)=a+b*log(h); fall back to cubic scaling if ill-conditioned."""
    if len(trials) < 2:
        return calibrated_size(trials[-1][0], trials[-1][1], target)
    x = np.log([float(v[0]) for v in trials])
    y = np.log([float(v[1]) for v in trials])
    slope, intercept = np.polyfit(x, y, 1)
    # Element count must decrease as characteristic size increases.  A
    # non-negative fitted slope is sampling noise or a fixed boundary-layer
    # floor, never a valid direction for extrapolation.  Reject it explicitly
    # rather than allowing the fit to request a finer and still larger mesh.
    if not np.isfinite(slope) or slope >= -0.5:
        return calibrated_size(trials[-1][0], trials[-1][1], target)
    return float(np.exp((math.log(float(target)) - intercept) / slope))


def boundary_layer_floor_evident(
    trials: Sequence[tuple[float, int]], target: int,
) -> bool:
    """Detect a coarse target that a fixed prism layer prevents reaching."""
    if len(trials) < 2 or any(count <= target for _, count in trials):
        return False
    previous_h, previous_count = trials[-2]
    current_h, current_count = trials[-1]
    return current_h > previous_h and current_count >= 0.98 * previous_count


def is_coarse_volume_mesh_failure(log_path: str | Path) -> bool:
    """Recognise a Simpleware failure caused by attempting a coarser mesh.

    A finer trial can be valid while a subsequent coarser trial cannot resolve
    the fixed prism layers or a narrow distal channel.  In that situation the
    successful finer count is the measured feasible floor; retrying the same
    coarse parameters on resume only repeats the internal mesher failure.
    """

    path = Path(log_path)
    if not path.is_file():
        return False
    text = path.read_text(encoding="utf-8", errors="replace").lower()
    return any(
        marker in text
        for marker in (
            "volume mesh could not be generated because of an internal error",
            "non-manifold mesh",
        )
    )


def feasible_targets(minimum: int, requested: Sequence[int], ceiling: int) -> list[int]:
    if minimum <= requested[0]:
        return [int(v) for v in requested]
    if minimum >= ceiling:
        raise ValueError("Boundary-layer minimum is at or above the finest target")
    values = np.geomspace(float(minimum), float(ceiling), 4)
    return [int(round(v)) for v in values]


def weighted_quantile(values: Sequence[float], weights: Sequence[float], q: float) -> float:
    v = np.asarray(values, dtype=float)
    w = np.asarray(weights, dtype=float)
    mask = np.isfinite(v) & np.isfinite(w) & (w > 0)
    if not mask.any() or not 0 <= q <= 1:
        return math.nan
    order = np.argsort(v[mask])
    vv, ww = v[mask][order], w[mask][order]
    centres = (np.cumsum(ww) - 0.5 * ww) / ww.sum()
    return float(np.interp(q, centres, vv, left=vv[0], right=vv[-1]))


def lumped_vertex_areas(points: np.ndarray, triangles: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=float)
    triangles = np.asarray(triangles, dtype=int)
    result = np.zeros(len(points), dtype=float)
    if not len(triangles):
        return result
    xyz = points[triangles]
    areas = 0.5 * np.linalg.norm(
        np.cross(xyz[:, 1] - xyz[:, 0], xyz[:, 2] - xyz[:, 0]), axis=1
    )
    for column in range(3):
        np.add.at(result, triangles[:, column], areas / 3.0)
    return result


def parallel_transport_frames(points: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return tangent and a minimally rotating orthonormal pair at each point."""
    p = np.asarray(points, dtype=float)
    if len(p) < 2:
        raise ValueError("At least two centreline points are required")
    tangent = np.empty_like(p)
    tangent[0] = p[1] - p[0]
    tangent[-1] = p[-1] - p[-2]
    if len(p) > 2:
        tangent[1:-1] = p[2:] - p[:-2]
    tangent /= np.maximum(np.linalg.norm(tangent, axis=1)[:, None], 1e-15)
    reference = np.array([1.0, 0.0, 0.0])
    if abs(np.dot(reference, tangent[0])) > 0.85:
        reference = np.array([0.0, 1.0, 0.0])
    normal = np.empty_like(p)
    normal[0] = reference - np.dot(reference, tangent[0]) * tangent[0]
    normal[0] /= np.linalg.norm(normal[0])
    for i in range(1, len(p)):
        candidate = normal[i - 1] - np.dot(normal[i - 1], tangent[i]) * tangent[i]
        if np.linalg.norm(candidate) < 1e-10:
            candidate = np.cross(tangent[i - 1], tangent[i])
        normal[i] = candidate / np.linalg.norm(candidate)
    binormal = np.cross(tangent, normal)
    binormal /= np.maximum(np.linalg.norm(binormal, axis=1)[:, None], 1e-15)
    return tangent, normal, binormal


def sector_indices(radial_vectors: np.ndarray, normal: np.ndarray, binormal: np.ndarray) -> np.ndarray:
    radial = np.asarray(radial_vectors, dtype=float)
    angles = np.mod(
        np.arctan2(radial @ np.asarray(binormal), radial @ np.asarray(normal)),
        2.0 * np.pi,
    )
    return np.floor(angles / (np.pi / 4.0)).astype(int).clip(0, 7)


def wss_band_statistics(values: np.ndarray, areas: np.ndarray, sectors: np.ndarray) -> dict[str, Any]:
    values, areas, sectors = map(np.asarray, (values, areas, sectors))
    valid = np.isfinite(values) & np.isfinite(areas) & (areas > 0)
    if not valid.any():
        raise ValueError("Wall band has no positive-area samples")
    vv, aa, ss = values[valid], areas[valid], sectors[valid]
    means = []
    for sector in range(8):
        use = ss == sector
        means.append(float(np.average(vv[use], weights=aa[use])) if use.any() else math.nan)
    return {
        "area_weighted_mean": float(np.average(vv, weights=aa)),
        "area_weighted_p95": weighted_quantile(vv, aa, 0.95),
        "sector_means": means,
        "sector_min": float(np.nanmin(means)),
        "sector_max": float(np.nanmax(means)),
        "raw_min": float(np.min(vv)),
        "raw_max": float(np.max(vv)),
    }


def normalized_change(coarse: float, fine: float, normalization: float) -> float:
    if not np.isfinite(normalization) or abs(normalization) < 1e-15:
        return math.inf
    return abs(float(fine) - float(coarse)) / abs(float(normalization))


def observed_order_gci(values: Sequence[float], counts: Sequence[int]) -> dict[str, Any]:
    """ASME-style observed order/GCI using achieved N^(-1/3) resolutions."""
    if len(values) != 3 or len(counts) != 3:
        raise ValueError("Exactly the finest three values/counts are required")
    phi = np.asarray(values, dtype=float)
    h = np.asarray(counts, dtype=float) ** (-1.0 / 3.0)
    order = np.argsort(h)[::-1]  # coarse -> fine
    phi, h = phi[order], h[order]
    e21, e32 = phi[1] - phi[0], phi[2] - phi[1]
    monotonic = e21 * e32 > 0 and abs(e21) > 0 and abs(e32) > 0
    if not monotonic:
        return {"status": "oscillatory_or_non_asymptotic", "finest_pair": abs(e32)}
    r21, r32 = h[0] / h[1], h[1] / h[2]
    p = max(0.05, abs(math.log(abs(e21 / e32))) / math.log(math.sqrt(r21 * r32)))
    for _ in range(30):
        ratio = (r21 ** p - 1.0) / max(r32 ** p - 1.0, 1e-15)
        new_p = abs(math.log(abs(e21 / e32) * ratio)) / math.log(r21)
        if abs(new_p - p) < 1e-8:
            p = new_p
            break
        p = max(0.05, new_p)
    extrapolated = phi[2] + (phi[2] - phi[1]) / (r32 ** p - 1.0)
    denom = max(abs(phi[2]), 1e-15)
    gci = 1.25 * abs((phi[2] - phi[1]) / denom) / (r32 ** p - 1.0)
    return {
        "status": "monotonic",
        "observed_order": float(p),
        "extrapolated": float(extrapolated),
        "gci_fine": float(gci),
        "refinement_ratios": [float(r21), float(r32)],
    }


def _segment_arrays(points: dict[int, tuple], segment: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    raw = np.asarray([points[i] for i in segment["point_ids"] if i in points], dtype=float)
    return raw[:, :3] / 1000.0, raw[:, 3] / 1000.0


def _arc(coords: np.ndarray) -> np.ndarray:
    if len(coords) == 0:
        return np.empty(0)
    return np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(coords, axis=0), axis=1))]


def select_distal_terminal(nodes: dict[int, tuple], points: dict[int, tuple], segments: list[dict[str, Any]]) -> dict[str, Any]:
    degree = defaultdict(int)
    adjacency = defaultdict(list)
    radii = {}
    lengths = {}
    for segment in segments:
        coords, rr = _segment_arrays(points, segment)
        length = float(_arc(coords)[-1]) if len(coords) else 0.0
        sid = int(segment["id"])
        lengths[sid] = length
        radii[sid] = float(np.nanmedian(rr))
        u, v = int(segment["node1"]), int(segment["node2"])
        degree[u] += 1; degree[v] += 1
        adjacency[u].append((v, length, sid)); adjacency[v].append((u, length, sid))
    terminals = [node for node, value in degree.items() if value == 1]
    if len(terminals) < 2:
        raise ValueError("Cropped graph must have at least two terminal nodes")
    inlet = max(terminals, key=lambda node: radii[adjacency[node][0][2]])
    distance = {inlet: 0.0}
    queue = [(0.0, inlet)]
    while queue:
        d, node = heapq.heappop(queue)
        if d != distance.get(node):
            continue
        for other, length, _ in adjacency[node]:
            candidate = d + length
            if candidate < distance.get(other, math.inf):
                distance[other] = candidate
                heapq.heappush(queue, (candidate, other))
    opening = max((n for n in terminals if n in distance and n != inlet), key=distance.get)
    edge = adjacency[opening][0][2]
    return {
        "inlet_node": int(inlet),
        "opening_node": int(opening),
        "opening_edge_id": int(edge),
        "opening_terminal_id": "edge{}:node{}".format(edge, opening),
        "geodesic_distance_mm": float(distance[opening]),
        "opening_coordinates_mm": [float(v) / 1000.0 for v in nodes[opening][:3]],
        "inlet_coordinates_mm": [float(v) / 1000.0 for v in nodes[inlet][:3]],
    }


def _candidate(segment, coords, radii, index, kind, label, priority=10):
    arc = _arc(coords)
    total = float(arc[-1]) if len(arc) else 0.0
    return {
        "kind": kind,
        "label": label,
        "edge_id": int(segment["id"]),
        "arc_fraction": float(arc[index] / total) if total > 0 else 0.0,
        "x_mm": float(coords[index, 0]), "y_mm": float(coords[index, 1]),
        "z_mm": float(coords[index, 2]), "radius_mm": float(radii[index]),
        "strahler": int(segment.get("strahler", 0)), "priority": int(priority),
        "approved": "",
    }


def generate_poi_candidates(nodes, points, segments) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    arrays = {}
    usable = []
    for segment in segments:
        coords, radii = _segment_arrays(points, segment)
        if len(coords) >= 2 and np.isfinite(radii).any():
            arrays[int(segment["id"])] = (coords, radii)
            usable.append(segment)
    terminal = select_distal_terminal(nodes, points, usable)
    candidates = []
    for role, node_key, edge_key in (
        ("inlet", "inlet_node", None), ("opening", "opening_node", "opening_edge_id")
    ):
        node = terminal[node_key]
        segment = next(s for s in usable if node in (s["node1"], s["node2"]) and (edge_key is None or s["id"] == terminal[edge_key]))
        coords, radii = arrays[int(segment["id"])]
        index = 0 if int(segment["node1"]) == node else len(coords) - 1
        candidates.append(_candidate(segment, coords, radii, index, role, role.upper(), 0))

    by_order = defaultdict(list)
    for segment in usable:
        coords, radii = arrays[int(segment["id"])]
        by_order[int(segment.get("strahler", 0))].append((float(np.median(radii)), segment))
    for order, values in sorted(by_order.items()):
        values.sort(key=lambda value: value[0])
        _, segment = values[len(values) // 2]
        coords, radii = arrays[int(segment["id"])]
        candidates.append(_candidate(segment, coords, radii, len(coords)//2, "strahler", "STRAHLER_{}".format(order)))

    all_r = np.concatenate([arrays[int(s["id"])][1] for s in usable])
    positive = all_r[np.isfinite(all_r) & (all_r > 0)]
    edges = np.geomspace(float(positive.min()), float(positive.max()), 9)
    flat = []
    for segment in usable:
        coords, radii = arrays[int(segment["id"])]
        flat.extend((abs(math.log(max(r, 1e-12))), segment, i) for i, r in enumerate(radii))
    for i in range(8):
        target = math.sqrt(edges[i] * edges[i + 1])
        _, segment, index = min(flat, key=lambda item: abs(item[0] - math.log(target)))
        coords, radii = arrays[int(segment["id"])]
        candidates.append(_candidate(segment, coords, radii, index, "radius_bin", "RADIUS_BIN_{:02d}".format(i + 1)))

    curvature, gradient = [], []
    for segment in usable:
        coords, radii = arrays[int(segment["id"])]
        for i in range(1, len(coords) - 1):
            a, b = coords[i] - coords[i-1], coords[i+1] - coords[i]
            scale = max(0.5 * (np.linalg.norm(a) + np.linalg.norm(b)), 1e-12)
            cosine = np.clip(np.dot(a, b) / max(np.linalg.norm(a)*np.linalg.norm(b), 1e-15), -1, 1)
            curvature.append((math.acos(cosine) / scale, segment, i))
            gradient.append((abs(radii[i+1]-radii[i-1]) / max(np.linalg.norm(coords[i+1]-coords[i-1]), 1e-12), segment, i))
    for rank, (_, segment, index) in enumerate(sorted(curvature, reverse=True, key=lambda x:x[0])[:5], 1):
        coords, radii = arrays[int(segment["id"])]
        candidates.append(_candidate(segment, coords, radii, index, "curvature", "CURVATURE_{:02d}".format(rank)))
    for rank, (_, segment, index) in enumerate(sorted(gradient, reverse=True, key=lambda x:x[0])[:5], 1):
        coords, radii = arrays[int(segment["id"])]
        candidates.append(_candidate(segment, coords, radii, index, "radius_gradient", "RADIUS_GRADIENT_{:02d}".format(rank)))

    incidence = defaultdict(list)
    node_adjacency = defaultdict(list)
    for segment in usable:
        incidence[int(segment["node1"])].append(segment)
        incidence[int(segment["node2"])].append(segment)
        length = arrays[int(segment["id"])][0]
        edge_length = float(_arc(length)[-1])
        node_adjacency[int(segment["node1"])].append((int(segment["node2"]), edge_length))
        node_adjacency[int(segment["node2"])].append((int(segment["node1"]), edge_length))
    node_distance = {int(terminal["inlet_node"]): 0.0}
    queue = [(0.0, int(terminal["inlet_node"]))]
    while queue:
        distance, node = heapq.heappop(queue)
        if distance != node_distance.get(node): continue
        for adjacent, length in node_adjacency[node]:
            candidate = distance + length
            if candidate < node_distance.get(adjacent, math.inf):
                node_distance[adjacent] = candidate
                heapq.heappush(queue, (candidate, adjacent))
    bifurcations = []
    for node, incident in incidence.items():
        if len(incident) >= 3:
            score = sum(float(np.median(arrays[int(s["id"])][1])) ** 2 for s in incident)
            bifurcations.append((score, node, incident))
    for rank, (_, node, incident) in enumerate(sorted(bifurcations, reverse=True, key=lambda x:x[0])[:5], 1):
        upstream = [s for s in incident if node_distance.get(int(s["node1"] if int(s["node2"])==node else s["node2"]), math.inf) < node_distance.get(node, math.inf)]
        downstream = [s for s in incident if s not in upstream]
        upstream.sort(key=lambda s: float(np.median(arrays[int(s["id"])][1])), reverse=True)
        downstream.sort(key=lambda s: float(np.median(arrays[int(s["id"])][1])), reverse=True)
        choices = []
        if upstream: choices.append(("UP", upstream[0]))
        if downstream: choices.append(("DOWN", downstream[0]))
        for side, segment in choices:
            coords, radii = arrays[int(segment["id"])]
            at_start = int(segment["node1"]) == node
            index = min(len(coords)-1, max(0, int(round(0.2*(len(coords)-1)))))
            if not at_start:
                index = len(coords)-1-index
            candidates.append(_candidate(segment, coords, radii, index, "bifurcation", "BIF_{:02d}_{}".format(rank, side)))

    candidates.sort(key=lambda c: (c["priority"], c["label"], c["edge_id"], c["arc_fraction"]))
    kept = []
    for candidate in candidates:
        xyz = np.array([candidate[k] for k in ("x_mm", "y_mm", "z_mm")])
        duplicate = False
        for old in kept:
            old_xyz = np.array([old[k] for k in ("x_mm", "y_mm", "z_mm")])
            threshold = 0.5 * min(candidate["radius_mm"], old["radius_mm"])
            if np.linalg.norm(xyz-old_xyz) < max(threshold, 0.02):
                duplicate = True; break
        if not duplicate:
            kept.append(candidate)
    for index, candidate in enumerate(kept, 1):
        candidate["poi_id"] = "POI_{:03d}".format(index)
    terminal["radius_bin_edges_mm"] = [float(v) for v in edges]
    return kept, terminal


def write_poi_files(candidates: list[dict[str, Any]], csv_path: Path, vtk_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fields = ("poi_id", "approved", "label", "kind", "edge_id", "arc_fraction", "x_mm", "y_mm", "z_mm", "radius_mm", "strahler")
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader(); writer.writerows(candidates)
    with vtk_path.open("w", encoding="utf-8") as handle:
        handle.write("# vtk DataFile Version 3.0\nCoronary POI candidates\nASCII\nDATASET POLYDATA\n")
        handle.write("POINTS {} float\n".format(len(candidates)))
        for c in candidates:
            handle.write("{x_mm:.9g} {y_mm:.9g} {z_mm:.9g}\n".format(**c))
        handle.write("VERTICES {} {}\n".format(len(candidates), 2*len(candidates)))
        for i in range(len(candidates)):
            handle.write("1 {}\n".format(i))
        handle.write("POINT_DATA {}\nSCALARS radius_mm float 1\nLOOKUP_TABLE default\n".format(len(candidates)))
        for c in candidates:
            handle.write("{:.9g}\n".format(c["radius_mm"]))
        for name, caster, key in (
            ("poi_index", int, None), ("amira_edge_id", int, "edge_id"),
            ("strahler_order", int, "strahler"),
        ):
            handle.write("SCALARS {} int 1\nLOOKUP_TABLE default\n".format(name))
            for index, candidate in enumerate(candidates, 1):
                handle.write("{}\n".format(index if key is None else caster(candidate[key])))


def approval_complete(csv_path: Path) -> bool:
    if not csv_path.is_file():
        return False
    with csv_path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    accepted = {"yes", "y", "true", "1", "no", "n", "false", "0"}
    complete = bool(rows) and all(str(row.get("approved", "")).strip().lower() in accepted for row in rows)
    if complete:
        positive = {"yes", "y", "true", "1"}
        for role in ("inlet", "opening"):
            row = next((item for item in rows if item.get("kind") == role), None)
            if row is None or str(row.get("approved", "")).strip().lower() not in positive:
                raise ValueError("The {} POI is mandatory and must be approved".format(role))
    return complete


def build_case_plan(
    cfg: dict[str, Any], selected_bl_layers: int | None = None,
    phase: str = "all",
) -> list[dict[str, Any]]:
    """Build the stable study matrix in its execution order.

    The bulk mesh-convergence matrix deliberately runs before the separate
    boundary-layer convergence study.  Its provisional layer count is fixed in
    the manifest (five for the current study), so a missing 4/6/8-layer
    selection can no longer gate generation of the main meshes.
    """
    if phase not in (
        "all", "main", "boundary_layer", "boundary_layer_thickness"
    ):
        raise ValueError(
            "phase must be 'all', 'main', 'boundary_layer', or "
            "'boundary_layer_thickness'"
        )
    targets = [int(v) for v in cfg["targets"]["element_counts"]]
    bl = cfg["boundary_layer"]
    adaptive = cfg["adaptive"]
    medium = targets[1]
    cases = []
    if phase in ("all", "main"):
        main_layers = int(bl.get("mesh_convergence_layers", 5))
        if main_layers < 1:
            raise ValueError("boundary_layer.mesh_convergence_layers must be positive")
        for family in ("global", "adaptive"):
            for level, target in zip(CASE_LEVELS, targets):
                cases.append({"case_id": "{}_{}".format(family, level), "family": family, "target_elements": target, "layers": main_layers, "purpose": "main"})
    if phase in ("all", "boundary_layer"):
        for layers in bl["candidate_layers"]:
            cases.append({"case_id": "bl_adaptive_{}layers".format(layers), "family": "adaptive", "target_elements": medium, "layers": int(layers), "purpose": "boundary_layer"})
    if phase in ("all", "boundary_layer_thickness"):
        thickness_ratios = bl.get("thickness_ratio_candidates")
        if not thickness_ratios:
            if phase == "boundary_layer_thickness":
                print(
                    "No boundary_layer.thickness_ratio_candidates are "
                    "configured; no thickness cases were added.", flush=True,
                )
            thickness_ratios = ()
        thickness_level = str(bl.get("thickness_test_level", "l3")).lower()
        if thickness_level not in CASE_LEVELS:
            raise ValueError(
                "boundary_layer.thickness_test_level must be one of {}"
                .format(", ".join(CASE_LEVELS))
            )
        thickness_target = targets[CASE_LEVELS.index(thickness_level)]
        baseline_ratio = float(bl.get("maximum_channel_radius_ratio", 0.15))
        for ratio in thickness_ratios:
            ratio = float(ratio)
            if ratio <= 0.0 or ratio > 1.0:
                raise ValueError(
                    "boundary-layer thickness ratios must be in (0, 1]"
                )
            # The matching main global mesh is the immutable 0.15-r baseline;
            # do not generate a duplicate mesh for it.
            if abs(ratio - baseline_ratio) <= 1.0e-12:
                continue
            cases.append({
                "case_id": "blthick_global_{}_r{:03d}".format(
                    thickness_level, int(round(1000.0 * ratio))
                ),
                "family": "global",
                "target_elements": thickness_target,
                "layers": int(bl.get("mesh_convergence_layers", 5)),
                "purpose": "boundary_layer_thickness",
                "maximum_channel_radius_ratio": ratio,
                "baseline_case_id": "global_{}".format(thickness_level),
            })
    for case in cases:
        case.update({"min_h_mm": float(adaptive["min_h_mm"]), "max_h_mm": float(adaptive["max_h_mm"]), "maximum_sphere_expansion": float(adaptive["maximum_sphere_expansion"])})
    return cases


def prepare_pois(cfg: dict[str, Any]) -> bool:
    out = Path(cfg["paths"]["output_dir"])
    csv_path, vtk_path = out/"poi"/"poi_candidates.csv", out/"poi"/"poi_candidates.vtk"
    provenance = out/"poi"/"source_graph.json"
    provenance_data = json.loads(provenance.read_text()) if provenance.exists() else {}
    current_amira_hash = file_sha256(cfg["paths"]["amira"])
    current_stl_hash = file_sha256(cfg["paths"]["stl"])
    regenerate = (
        not csv_path.exists() or not provenance.exists()
        or int(provenance_data.get("poi_algorithm_version", 0)) != 2
        or provenance_data.get("amira_sha256") != current_amira_hash
        or provenance_data.get("stl_sha256") != current_stl_hash
    )
    if regenerate and csv_path.exists() and approval_complete(csv_path):
        old_hash = str(provenance_data.get("stl_sha256", "unknown"))[:12]
        archive = csv_path.with_name("poi_candidates.pre_recrop_{}.csv".format(old_hash))
        if not archive.exists():
            shutil.copy2(csv_path, archive)
        print(
            "POI geometry authority changed; archived prior approvals at {} and "
            "requiring review of regenerated candidates.".format(archive), flush=True,
        )
    if regenerate:
        # Reuse the exact crop implementation that creates the validated planes
        # and refinement spheres.  This is intentionally slower than parsing the
        # raw graph: POIs must not be proposed on vessels absent from the STL.
        import simpleware_coronary_regions as regions
        regions.AMIRA_SPATIAL_GRAPH_PATH = cfg["paths"]["amira"]
        regions.STL_SURFACE_PATH = cfg["paths"]["stl"]
        raw_network = regions._load_amira_network()
        cropped = regions._crop_amira_network_to_stl(
            raw_network, regions._load_stl_solid()
        )
        nodes = {
            int(node.node_id): tuple(float(v) * 1000.0 for v in node.position) + (len(node.splines),)
            for node in cropped.nodes
        }
        points, segments = {}, []
        point_id = 0
        for spline in cropped.splines:
            ids = []
            for xyz, radius in zip(spline.points, spline.radii):
                points[point_id] = tuple(float(v) * 1000.0 for v in xyz) + (float(radius) * 1000.0,)
                ids.append(point_id); point_id += 1
            segments.append({
                "id": int(spline.edge_id), "node1": int(spline.start_node.node_id),
                "node2": int(spline.end_node.node_id), "point_ids": ids,
                "strahler": int(spline.strahler_order or 0),
            })
        candidates, opening = generate_poi_candidates(nodes, points, segments)
        write_poi_files(candidates, csv_path, vtk_path)
        (out/"poi"/"opening_terminal.json").write_text(json.dumps(opening, indent=2)+"\n", encoding="utf-8")
        provenance.write_text(json.dumps({
            "poi_algorithm_version": 2,
            "amira_sha256": current_amira_hash,
            "stl_sha256": current_stl_hash,
            "retained_edges": len(cropped.splines), "retained_nodes": len(cropped.nodes),
        }, indent=2)+"\n", encoding="utf-8")
        print("Generated {} POI candidates: {}".format(len(candidates), csv_path), flush=True)
    return approval_complete(csv_path)


def write_plan(cfg: dict[str, Any]) -> Path:
    out = Path(cfg["paths"]["output_dir"]); out.mkdir(parents=True, exist_ok=True)
    seed_contract = validate_cfx_seed_contract(cfg, out/"cfx_seed_contract.json")
    selection = out/"boundary_layer"/"selection.json"
    chosen = json.loads(selection.read_text(encoding="utf-8"))["selected_layers"] if selection.is_file() else None
    payload = {
        "source_hashes": {key: file_sha256(cfg["paths"][key]) for key in ("source_sip", "stl", "amira", "cfx_seed")},
        "cfx_physics_control_sha256": seed_contract["physics_control_sha256"],
        "cases": build_case_plan(cfg, chosen, phase="all"),
        "selected_boundary_layers": chosen,
        "execution_order": ["mesh_convergence", "boundary_layer_convergence"],
        "mesh_convergence_layers": int(cfg["boundary_layer"].get("mesh_convergence_layers", 5)),
    }
    path = out/"study_plan.json"; path.write_text(json.dumps(payload, indent=2)+"\n", encoding="utf-8")
    return path


def select_boundary_layer(result_4: dict, result_6: dict, result_8: dict, wss_tol=0.05, pressure_tol=0.01) -> int:
    def passes(a, b):
        return (
            normalized_change(a["wss"], b["wss"], b["wss"]) <= wss_tol
            and normalized_change(a["pressure_drop"], b["pressure_drop"], b["pressure_drop"]) <= pressure_tol
        )
    if not passes(result_6, result_8):
        raise RuntimeError("6-to-8 prism-layer comparison failed; wall mesh is unconverged")
    return 4 if passes(result_4, result_6) else 6


def _stream_command(command: list[str], cwd: Path, log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            command, cwd=str(cwd), stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, text=True, bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True); log.write(line); log.flush()
        code = process.wait()
    if code:
        raise RuntimeError("Command failed with exit code {} (log: {})".format(code, log_path))


def _mesh_trial_job(
    cfg: dict[str, Any], case: dict[str, Any], trial: int,
    global_h: float, n_d: float, resume: bool,
) -> dict[str, Any]:
    out = Path(cfg["paths"]["output_dir"])
    trial_dir = out/"meshes"/case["case_id"]/"calibration"/"trial_{:02d}".format(trial)
    stats_path = trial_dir/"mesh_stats.json"
    quality_json_path = trial_dir/"mesh_quality.json"
    quality_csv_path = trial_dir/"mesh_quality.csv"
    if resume and stats_path.is_file():
        return json.loads(stats_path.read_text(encoding="utf-8"))
    trial_dir.mkdir(parents=True, exist_ok=True)
    case_sip, msh = trial_dir/"case.sip", trial_dir/"mesh.msh"
    cropped_graph = trial_dir/"cropped_amira_graph.json"
    shutil.copy2(cfg["paths"]["source_sip"], case_sip)
    job = {
        "repository_dir": str(Path(__file__).resolve().parent),
        "case_id": case["case_id"], "case_sip": str(case_sip),
        "fluent_mesh": str(msh), "mesh_stats_json": str(stats_path),
        "mesh_quality_json": str(quality_json_path),
        "mesh_quality_csv": str(quality_csv_path),
        "cropped_graph_json": str(cropped_graph),
        "family": case["family"], "target_elements": case["target_elements"],
        "global_h_mm": global_h, "n_d": n_d,
        "min_h_mm": case["min_h_mm"], "max_h_mm": case["max_h_mm"],
        "maximum_sphere_expansion": case["maximum_sphere_expansion"],
        "amira_path": cfg["paths"]["amira"], "stl_path": cfg["paths"]["stl"],
        "surface_contract": cfg.get("surface_contract"),
        "boundary_layer": {
            "layers": case["layers"],
            "growth_ratio": cfg["boundary_layer"]["growth_ratio"],
            "simpleware_width_ratio": simpleware_width_ratio(
                case["layers"], cfg["boundary_layer"]["growth_ratio"]
            ),
            "requested_total_thickness_mm": cfg["boundary_layer"]["requested_total_thickness_mm"],
            "maximum_channel_radius_ratio": case.get(
                "maximum_channel_radius_ratio",
                cfg["boundary_layer"]["maximum_channel_radius_ratio"],
            ),
            "minimum_inlet_cell_quality": cfg["boundary_layer"].get("minimum_inlet_cell_quality", 0.1),
            "separate_boundary_layer": False,
        },
        "mesh_quality": {
            # Off-surface additional improvement is unsafe where coronary
            # branches nearly touch: node motion can merge otherwise separate
            # surfaces and Simpleware then reports a non-manifold mesh.  The
            # base +FE quality pass remains active and CFX import is still the
            # downstream positive-volume/quality gate.
            "use_additional_improvement": bool(
                cfg.get("meshing", {}).get(
                    "use_additional_quality_improvement", False
                )
            ),
            "allow_off_surface": bool(
                cfg.get("meshing", {}).get(
                    "additional_quality_allow_off_surface", False
                )
            ),
        },
    }
    job_path = trial_dir/"job.json"
    job_path.write_text(json.dumps(job, indent=2)+"\n", encoding="utf-8")
    console = cfg["executables"]["simpleware_console"]
    entry = str(Path(__file__).resolve().parent/"simpleware_mesh_sensitivity_console.py")
    command = [console, "--run-script={}".format(entry), "--script-lan=python3",
               "--input-value={}".format(job_path), "--exit-after-script", "--disable-undo"]
    _stream_command(command, trial_dir, trial_dir/"simpleware.log")
    return json.loads(stats_path.read_text(encoding="utf-8"))


def calibrate_mesh_case(cfg: dict[str, Any], case: dict[str, Any], resume: bool) -> dict[str, Any]:
    out = Path(cfg["paths"]["output_dir"]); final_dir = out/"meshes"/case["case_id"]
    final_stats = final_dir/"mesh_stats.json"
    if resume and final_stats.is_file():
        return json.loads(final_stats.read_text(encoding="utf-8"))
    target = int(case["target_elements"])
    medium = int(cfg["targets"]["element_counts"][1])
    base_h = float(cfg["targets"].get("initial_global_h_mm", 0.18))
    global_h = base_h * (medium/target) ** (1/3)
    n_d = float(cfg["adaptive"].get("initial_n_d", 6.0)) * (target/medium) ** (1/3)
    if case["family"] == "adaptive" and case.get("purpose") == "main":
        level = case["case_id"].rsplit("_", 1)[-1]
        if level in CASE_LEVELS and CASE_LEVELS.index(level) > 0:
            previous_level = CASE_LEVELS[CASE_LEVELS.index(level) - 1]
            previous_path = out / "meshes" / (
                "adaptive_" + previous_level
            ) / "mesh_stats.json"
            if previous_path.is_file():
                previous = json.loads(previous_path.read_text(encoding="utf-8"))
                previous_count = int(previous["total_elements"])
                previous_n_d = float(previous["n_d"])
                # Extrapolating from the immediately preceding realised mesh
                # is much safer than extrapolating every level from the
                # original medium target.  A small downward margin avoids the
                # Simpleware negative-width failure seen when very fine local
                # refinement collides with the fixed prism-layer envelope.
                extrapolated_n_d = (
                    previous_n_d * (target / previous_count) ** (1.0 / 3.0)
                    * float(cfg["adaptive"].get(
                        "successive_level_initial_n_d_safety_factor", 0.95
                    ))
                )
                if extrapolated_n_d < n_d:
                    print(
                        "{}: seeding n_D={:.6g} from realised {} "
                        "({:,} elements, n_D={:.6g}) instead of {:.6g}."
                        .format(
                            case["case_id"], extrapolated_n_d,
                            "adaptive_" + previous_level, previous_count,
                            previous_n_d, n_d,
                        ),
                        flush=True,
                    )
                    n_d = extrapolated_n_d
    fixed_bulk_size = case.get("purpose") == "boundary_layer_thickness"
    if fixed_bulk_size:
        baseline_case = str(case["baseline_case_id"])
        baseline_stats_path = out / "meshes" / baseline_case / "mesh_stats.json"
        if not baseline_stats_path.is_file():
            raise FileNotFoundError(
                "Boundary-layer thickness testing requires its baseline mesh: {}"
                .format(baseline_stats_path)
            )
        baseline_stats = json.loads(
            baseline_stats_path.read_text(encoding="utf-8")
        )
        global_h = float(baseline_stats["global_h_mm"])
        print(
            "{}: holding global bulk/surface size at {:.9g} mm from {} to "
            "isolate boundary-layer thickness.".format(
                case["case_id"], global_h, baseline_case
            ),
            flush=True,
        )
    trials = []
    selected_trial = None
    source_hash = file_sha256(cfg["paths"]["source_sip"])
    for iteration in range(1, 5):
        attempted_effective = global_h if case["family"] == "global" else 1.0/n_d
        failure_log = (
            out / "meshes" / case["case_id"] / "calibration"
            / "trial_{:02d}".format(iteration) / "simpleware.log"
        )
        existing_coarse_failure = (
            resume
            and case["purpose"] == "main"
            and case["case_id"].endswith("_l1")
            and bool(trials)
            and attempted_effective > trials[-1][0]
            and is_coarse_volume_mesh_failure(failure_log)
        )
        if existing_coarse_failure:
            measured = min(count for _, count in trials)
            print(
                "{}: reusing the recorded coarser-trial failure; treating "
                "{:,} elements as the measured feasible floor."
                .format(case["case_id"], measured),
                flush=True,
            )
            raise InfeasibleCoarseTarget(measured)
        try:
            stats = _mesh_trial_job(
                cfg, case, iteration, global_h, n_d, resume
            )
        except RuntimeError:
            if (
                case["purpose"] == "main"
                and case["case_id"].endswith("_l1")
                and trials
                and attempted_effective > trials[-1][0]
                and is_coarse_volume_mesh_failure(failure_log)
            ):
                measured = min(count for _, count in trials)
                print(
                    "{}: coarser trial {} failed after a valid finer mesh; "
                    "treating {:,} elements as the measured feasible floor."
                    .format(case["case_id"], iteration, measured),
                    flush=True,
                )
                raise InfeasibleCoarseTarget(measured)
            raise
        if file_sha256(cfg["paths"]["source_sip"]) != source_hash:
            raise RuntimeError("Validated source SIP changed during disposable job")
        count = int(stats["total_elements"])
        effective_h = global_h if case["family"] == "global" else 1.0/n_d
        trials.append((effective_h, count))
        selected_trial = iteration
        error = abs(count-target)/target
        print("{} trial {}: {:,} elements ({:+.2%})".format(case["case_id"], iteration, count, (count-target)/target), flush=True)
        if fixed_bulk_size:
            break
        if error <= float(cfg["targets"].get("tolerance_fraction", 0.05)):
            break
        if (
            case["purpose"] == "main"
            and case["case_id"].endswith("_l1")
            and boundary_layer_floor_evident(trials, target)
        ):
            print(
                "{}: coarsening did not reduce the fixed-layer mesh; treating "
                "{:,} elements as the measured feasible floor.".format(
                    case["case_id"], min(value for _, value in trials)
                ),
                flush=True,
            )
            raise InfeasibleCoarseTarget(min(value for _, value in trials))
        new_effective = log_log_size_fit(trials, target) if len(trials) >= 2 else calibrated_size(effective_h, count, target)
        if case["family"] == "global": global_h = new_effective
        else: n_d = 1.0/new_effective
    else:
        if (
            case["purpose"] == "main" and case["case_id"].endswith("_l1")
            and all(count > target for _, count in trials)
            and (trials[-1][1] >= 0.9 * trials[-2][1] or trials[-1][0] >= 2.0)
        ):
            raise InfeasibleCoarseTarget(min(count for _, count in trials))
        raise RuntimeError("{} did not reach element target within four trials".format(case["case_id"]))
    trial_dir = final_dir/"calibration"/"trial_{:02d}".format(selected_trial)
    final_dir.mkdir(parents=True, exist_ok=True)
    artifacts = (
        ("case.sip", "case.sip"),
        ("mesh.msh", "mesh.msh"),
        ("mesh_stats.json", "mesh_stats.json"),
        ("mesh_quality.json", "mesh_quality.json"),
        ("mesh_quality.csv", "mesh_quality.csv"),
        ("job.json", "job.json"),
        ("cropped_amira_graph.json", "cropped_amira_graph.json"),
    )
    for source_name, dest_name in artifacts:
        source_path = trial_dir/source_name
        # Pre-existing calibration trials made before quality export remain
        # resumable. Newly generated trials always contain both quality files.
        if source_path.is_file():
            shutil.copy2(source_path, final_dir/dest_name)
    # A feasible-floor adjustment can promote a valid trial originally run
    # against the requested 1.5M target.  Keep the immutable trial metadata,
    # but make the promoted case state its adjusted target unambiguously.
    promoted_stats = json.loads(final_stats.read_text(encoding="utf-8"))
    if int(promoted_stats.get("target_elements", target)) != target:
        promoted_stats["trial_target_elements"] = int(
            promoted_stats["target_elements"]
        )
        promoted_stats["target_elements"] = target
        final_stats.write_text(
            json.dumps(promoted_stats, indent=2) + "\n", encoding="utf-8"
        )
        final_job = final_dir / "job.json"
        promoted_job = json.loads(final_job.read_text(encoding="utf-8"))
        promoted_job["trial_target_elements"] = int(
            promoted_job["target_elements"]
        )
        promoted_job["target_elements"] = target
        final_job.write_text(
            json.dumps(promoted_job, indent=2) + "\n", encoding="utf-8"
        )
    history = {"case_id":case["case_id"], "target_elements":target, "trials":[{"effective_h":h,"elements":n} for h,n in trials], "selected_trial":selected_trial, "source_sip_sha256":source_hash}
    (final_dir/"calibration.json").write_text(json.dumps(history, indent=2)+"\n", encoding="utf-8")
    return json.loads(final_stats.read_text(encoding="utf-8"))


def run_mesh_stage(cfg: dict[str, Any], resume: bool, phase: str = "main") -> None:
    out = Path(cfg["paths"]["output_dir"])
    selection = out/"boundary_layer"/"selection.json"
    chosen = json.loads(selection.read_text())["selected_layers"] if selection.is_file() else None
    cases = build_case_plan(cfg, chosen, phase=phase)
    try:
        for case in cases:
            calibrate_mesh_case(cfg, case, resume)
    except InfeasibleCoarseTarget as exc:
        old = [int(v) for v in cfg["targets"]["element_counts"]]
        new = feasible_targets(exc.measured_minimum, old, old[-1])
        cfg["targets"]["element_counts"] = new
        adjustment = out/"targets_adjusted_for_boundary_layer.json"
        adjustment.write_text(json.dumps({"requested":old, "measured_minimum_feasible":exc.measured_minimum, "adjusted":new}, indent=2)+"\n", encoding="utf-8")
        print("Adjusted targets for fixed boundary-layer floor: {}".format(new), flush=True)
        write_plan(cfg)
        for case in build_case_plan(cfg, chosen, phase=phase):
            calibrate_mesh_case(cfg, case, True)


def export_existing_simpleware_quality(
    cfg: dict[str, Any], resume: bool
) -> Path:
    """Run the native quality inspector on final SIPs without remeshing."""
    out = Path(cfg["paths"]["output_dir"])
    meshes_dir = out / "meshes"
    report_dir = out / "reports" / "simpleware_mesh_quality"
    report_dir.mkdir(parents=True, exist_ok=True)
    console = cfg["executables"]["simpleware_console"]
    entry = Path(__file__).resolve().with_name(
        "simpleware_mesh_quality_inspection_console.py"
    )
    if not entry.is_file():
        raise FileNotFoundError(entry)
    cases = sorted(meshes_dir.glob("*/case.sip"))
    if not cases:
        raise RuntimeError("No final case SIPs were found below {}".format(meshes_dir))
    summaries = []
    combined_rows = []
    for case_sip in cases:
        case_id = case_sip.parent.name
        json_path = case_sip.parent / "mesh_quality_inspection.json"
        csv_path = case_sip.parent / "mesh_quality_inspection.csv"
        if not (resume and json_path.is_file() and csv_path.is_file()):
            job = {
                "case_id": case_id,
                "case_sip": str(case_sip.resolve()),
                "quality_inspection_json": str(json_path.resolve()),
                "quality_inspection_csv": str(csv_path.resolve()),
            }
            job_path = case_sip.parent / "mesh_quality_inspection_job.json"
            job_path.write_text(json.dumps(job, indent=2) + "\n", encoding="utf-8")
            command = [
                console,
                "--run-script={}".format(entry),
                "--script-lan=python3",
                "--input-value={}".format(job_path),
                "--exit-after-script",
                "--disable-undo",
            ]
            _stream_command(
                command, case_sip.parent,
                case_sip.parent / "mesh_quality_inspection.log",
            )
        payload = json.loads(json_path.read_text(encoding="utf-8"))
        summaries.append({"case_id": case_id, **payload["totals"]})
        combined_rows.extend(payload["metrics"])
    def write_quality_table(path: Path, rows: list[dict[str, Any]]) -> None:
        if not rows:
            raise RuntimeError(
                "Simpleware quality inspection produced no rows for {}".format(path)
            )
        with path.open("w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    write_quality_table(report_dir / "inspection_totals.csv", summaries)
    write_quality_table(report_dir / "inspection_metrics.csv", combined_rows)
    summary_path = report_dir / "inspection_summary.json"
    summary_path.write_text(json.dumps({
        "schema_version": 1,
        "case_count": len(summaries),
        "cases": summaries,
    }, indent=2) + "\n", encoding="utf-8")
    return summary_path


def _render_template(path: Path, values: dict[str, str]) -> str:
    text = path.read_text(encoding="utf-8")
    for key, value in values.items():
        text = text.replace("{{"+key+"}}", value.replace("\\", "/"))
    missing = sorted(set(__import__("re").findall(r"\{\{([A-Z0-9_]+)\}\}", text)))
    if missing:
        raise ValueError("Unresolved CFX-Pre template tokens: {}".format(missing))
    return text


def parse_cfx_solver_output(
    path: str | Path, residual_target: float | None = None
) -> dict[str, Any]:
    """Read the terminal solver state; a zero process exit alone is insufficient."""
    text = Path(path).read_text(encoding="utf-8", errors="replace")
    lower = text.lower()
    converged = any(
        phrase in lower for phrase in (
            "convergence criteria satisfied",
            "convergence criterion satisfied",
            "residual target has been achieved",
        )
    )
    normal = any(
        phrase in lower for phrase in (
            "solver finished normally", "run completed normally", "execution completed successfully"
        )
    )
    iterations = [int(v) for v in re.findall(r"(?:iteration|coefficient loop)\s*[=:]?\s*(\d+)", text, re.I)]
    residual_blocks = re.findall(
        r"OUTER LOOP ITERATION\s*=\s*(\d+).*?"
        r"\|\s*U-Mom\s*\|[^|]*\|\s*([0-9.Ee+\-]+).*?"
        r"\|\s*V-Mom\s*\|[^|]*\|\s*([0-9.Ee+\-]+).*?"
        r"\|\s*W-Mom\s*\|[^|]*\|\s*([0-9.Ee+\-]+).*?"
        r"\|\s*P-Mass\s*\|[^|]*\|\s*([0-9.Ee+\-]+)",
        text,
        re.S,
    )
    last_residuals: dict[str, float] = {}
    if residual_blocks:
        _, u_rms, v_rms, w_rms, p_rms = residual_blocks[-1]
        last_residuals = {
            "U-Mom": float(u_rms), "V-Mom": float(v_rms),
            "W-Mom": float(w_rms), "P-Mass": float(p_rms),
        }
    numeric_converged = bool(
        residual_target is not None
        and len(last_residuals) == 4
        and max(last_residuals.values()) <= float(residual_target)
    )
    converged = bool(converged or numeric_converged)
    return {
        "converged": converged,
        "converged_by_numeric_rms": numeric_converged,
        "acceptance_residual_target": residual_target,
        "normal_completion": bool(normal or converged),
        "last_iteration": max(iterations) if iterations else None,
        "last_rms_residuals": last_residuals,
        "source_output": str(Path(path)),
    }


def _write_cfx_boundary_audit_session(
    session: Path, audit_csv: Path, boundary_names: Sequence[str]
) -> None:
    output = audit_csv.as_posix().replace('"', '\\"')
    lines = [
        "COMMAND FILE:", "  CFX Post Version = 25.2", "END",
        '! open(my $audit, \">\", \"{}\") or die $!;'.format(output),
        '! print $audit \"boundary,mass_flow_kg_s\\n\";',
    ]
    for name in boundary_names:
        escaped_name = str(name).replace('"', '\\"')
        expression = "massFlow()\\@{}".format(escaped_name)
        lines.extend([
            '! ($value, $units) = evaluate(\"{}\");'.format(expression),
            '! print $audit \"{},$value\\n\";'.format(escaped_name),
        ])
    lines.extend(['! close($audit);', '>quit', ''])
    session.write_text("\n".join(lines), encoding="utf-8")


def run_cfx_boundary_audit(
    cfg: dict[str, Any], res: Path, flow_csv: Path, cfx_dir: Path
) -> dict[str, Any]:
    with flow_csv.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    fixed = [row["ccl_name"] for row in rows if row["boundary_role"] == "fixed_outlet"]
    boundaries = ["Inlet_000", *fixed, "Pressure_Opening"]
    values: dict[str, float] = {}
    # The solver's terminal P-Mass boundary summary is authoritative and is
    # substantially faster and more robust than opening every large result in
    # CFD-Post.  Use the newest output that contains all required boundaries.
    for solver_output in sorted(cfx_dir.glob("*.out"), key=lambda p: p.stat().st_mtime, reverse=True):
        solver_text = solver_output.read_text(encoding="utf-8", errors="replace")
        sections = re.findall(
            r"\|\s*P-Mass\s*\|.*?(?=\|\s*Normalised Imbalance Summary\s*\||\Z)",
            solver_text,
            re.S,
        )
        for section in reversed(sections):
            parsed = {
                name.strip(): float(value)
                for name, value in re.findall(
                    r"^\s*Boundary\s*:\s*(.*?)\s+([+\-]?[0-9.]+E[+\-]\d+)\s*$",
                    section,
                    re.M | re.I,
                )
            }
            if set(boundaries).issubset(parsed):
                values = {name: parsed[name] for name in boundaries}
                break
        if values:
            break
    if not values:
        session, raw_csv = cfx_dir/"boundary_audit.cse", cfx_dir/"boundary_mass_flows.csv"
        _write_cfx_boundary_audit_session(session, raw_csv, boundaries)
        _stream_command(
            [cfg["executables"]["cfx_post"], "-batch", str(session), "-res", str(res)],
            cfx_dir,
            cfx_dir/"boundary_audit.log",
        )
        if not raw_csv.is_file():
            raise RuntimeError("CFD-Post did not produce the boundary mass-flow audit")
        with raw_csv.open(newline="", encoding="utf-8-sig") as handle:
            values = {
                row["boundary"]: float(row["mass_flow_kg_s"])
                for row in csv.DictReader(handle)
                if row.get("mass_flow_kg_s")
            }
    missing = sorted(set(boundaries) - set(values))
    if missing or any(not np.isfinite(values[name]) for name in boundaries if name in values):
        raise RuntimeError("Boundary audit is missing/non-finite for: {}".format(", ".join(missing)))
    inlet = values["Inlet_000"]
    opening = values["Pressure_Opening"]
    imbalance = abs(sum(values.values())) / max(abs(inlet), 1e-30)
    opening_outward = inlet * opening < 0.0
    report = {
        "mass_flows_kg_s": values,
        "mass_imbalance_fraction": float(imbalance),
        "opening_outward": bool(opening_outward),
        "pass": bool(
            imbalance <= float(cfg["cfx"]["maximum_mass_imbalance_fraction"])
            and (opening_outward or not cfg["cfx"].get("require_outward_opening_flow", True))
        ),
    }
    (cfx_dir/"boundary_audit.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if not report["pass"]:
        raise RuntimeError(
            "CFX boundary audit failed (imbalance={:.4%}, opening_outward={})"
            .format(imbalance, opening_outward)
        )
    return report


def verify_opening_mapping(flow_dir: Path, expected: dict[str, Any], cfx_dir: Path) -> dict[str, Any]:
    path = flow_dir/"opening_boundary.json"
    if not path.is_file():
        raise RuntimeError("Flow assignment did not emit opening_boundary.json")
    actual = json.loads(path.read_text(encoding="utf-8"))
    stable_ok = actual.get("opening_terminal_id") == expected["opening_terminal_id"]
    position = np.asarray(actual.get("pos", [math.nan]*3), dtype=float)
    expected_position = np.asarray(expected["opening_coordinates_mm"], dtype=float)
    coordinate_error = float(np.linalg.norm(position-expected_position))
    area = float(actual.get("surface_area_mm2", 0.0))
    expected_radius = None
    # The POI record is the STL-cropped terminal radius; area is checked against
    # it with a deliberately broad faceting/non-circularity tolerance.
    poi_csv = cfx_dir.parents[1]/"poi"/"poi_candidates.csv"
    if poi_csv.is_file():
        with poi_csv.open(newline="",encoding="utf-8-sig") as handle:
            opening_row = next((row for row in csv.DictReader(handle) if row["kind"]=="opening"), None)
        if opening_row is not None:
            expected_radius = float(opening_row["radius_mm"])
    expected_area = math.pi*expected_radius**2 if expected_radius else math.nan
    area_ratio = area/expected_area if expected_area > 0 else math.nan
    passed = stable_ok and coordinate_error <= 1.0 and area > 0 and (
        not np.isfinite(area_ratio) or 0.5 <= area_ratio <= 1.5
    )
    report = {
        "pass":bool(passed), "stable_terminal_match":bool(stable_ok),
        "coordinate_error_mm":coordinate_error, "surface_area_mm2":area,
        "expected_area_mm2":expected_area, "area_ratio":area_ratio,
        "actual":actual,
    }
    (cfx_dir/"opening_verification.json").write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")
    if not passed:
        raise RuntimeError("Opening mapping verification failed: {}".format(report))
    return report


def prepare_cfx_seed_physics_ccl(cfg: dict[str, Any], out: Path) -> Path:
    """Export seed physics while discarding stale mesh/GUI state.

    The binary seed embeds a legacy mesh source path that is no longer present,
    so a recorded ``Reload Mesh Files`` action cannot be replayed.  CFX-Pre's
    supported CCL export retains the complete LIBRARY and FLOW definitions.  We
    then change only mesh-location references to the stable names emitted by
    ``flow_fractions.py``; all physical models and solver controls are untouched.
    """
    cfx_root = out / "cfx"
    cfx_root.mkdir(parents=True, exist_ok=True)
    exported = cfx_root / "seed_complete_physics.ccl"
    sanitized = cfx_root / "seed_physics_for_remesh.ccl"
    export_session = cfx_root / "export_seed_physics.pre"
    configured_seed = Path(cfg["paths"]["cfx_seed"])
    configured_hash = file_sha256(configured_seed)
    # Some CFX-Pre builds update case-state metadata even during an export.
    # Never open the authoritative seed directly; use a disposable local copy.
    export_seed = cfx_root / "seed_export_working_copy.cfx"
    shutil.copy2(configured_seed, export_seed)
    export_session.write_text(
        "COMMAND FILE:\n"
        "  CFX Pre Version = 25.2\n"
        "END\n\n"
        ">exportccl filename={}, mode=overwrite\n"
        "> update\n\n"
        ">quit\n".format(str(exported).replace("\\", "/")),
        encoding="utf-8",
    )
    _stream_command(
        [
            cfg["executables"]["cfx_pre"],
            "-cfx",
            str(export_seed),
            "-batch",
            str(export_session),
        ],
        cfx_root,
        cfx_root / "export_seed_physics.log",
    )
    if not exported.is_file():
        raise RuntimeError("CFX-Pre did not export the immutable seed CCL")
    if file_sha256(configured_seed) != configured_hash:
        raise RuntimeError("Authoritative CFX seed changed during CCL export")

    text = exported.read_text(encoding="utf-8", errors="strict")
    end = text.find("\nCOMMAND FILE:")
    if end < 0:
        raise RuntimeError("Seed CCL has no COMMAND FILE boundary after FLOW data")
    text = text[:end].rstrip() + "\n"

    old_volume = "lumen_bspline_cropped_smoothed_meshmixer 2 from surface 2 fluid"
    new_volume = "lumen_bspline_cropped_smoothed_meshmixer 2 fluid"
    wall = "from lumen_bspline_cropped_smoothed_meshmixer 2 to background wall"
    text, volume_count = re.subn(
        r"(?m)^(\s*Location\s*=\s*)" + re.escape(old_volume) + r"\s*$",
        r"\g<1>" + new_volume,
        text,
    )
    text, wall_count = re.subn(
        r"(?m)^(\s*Location\s*=\s*)Primitive 2D\s*$",
        r"\g<1>" + wall,
        text,
    )
    inlet_count = 0
    outlet_count = 0
    lines: list[str] = []
    for line in text.splitlines():
        match = re.match(r"^(\s*Location\s*=\s*)(Inlet|Outlet)_(\d{3})\s*$", line)
        if match:
            line = "{}{}{}".format(match.group(1), match.group(2), match.group(3))
            if match.group(2) == "Inlet":
                inlet_count += 1
            else:
                outlet_count += 1
        lines.append(line)
    text = "\n".join(lines) + "\n"
    if (volume_count, wall_count, inlet_count, outlet_count) != (1, 1, 1, 77):
        raise RuntimeError(
            "Unexpected seed mesh-location contract: volume={}, wall={}, inlet={}, outlets={}".format(
                volume_count, wall_count, inlet_count, outlet_count
            )
        )
    # The legacy seed contains one pressure opening under an old Outlet_010
    # object.  CFX's append import can retain nested children from that object
    # even when the parent boundary is replaced.  Strip every seed cap-boundary
    # object and let the generated boundary-only CCL recreate all caps cleanly.
    stripped: list[str] = []
    skipping_cap = False
    removed_caps = 0
    for line in text.splitlines():
        if not skipping_cap and re.match(
            r"^    BOUNDARY: (?:Inlet_000|Outlet_\d{3})\s*$", line
        ):
            skipping_cap = True
            removed_caps += 1
            continue
        if skipping_cap:
            if line == "    END":
                skipping_cap = False
            continue
        stripped.append(line)
    if skipping_cap or removed_caps != 78:
        raise RuntimeError(
            "Unexpected seed cap-boundary count while sanitizing: {}".format(
                removed_caps
            )
        )
    text = "\n".join(stripped) + "\n"
    # The recorded seed's convenience monitor uses ``Pressure`` rather than
    # the solver variable ``Static Pressure``.  On a remeshed case CFX tries to
    # evaluate it while constructing the boundary-condition database, before
    # DENSITY is registered, and aborts in CAL_BCP_CDB/GET_GVAR.  It is not a
    # physics or convergence control and all pressure metrics are extracted
    # from the result, so omit this remesh-unsafe diagnostic monitor.
    text, monitor_count = re.subn(
        r"(?ms)^      MONITOR POINT: Inlet Pressure\s*\n.*?^      END\s*\n",
        "",
        text,
    )
    if monitor_count != 1:
        raise RuntimeError(
            "Expected one remesh-unsafe Inlet Pressure monitor, found {}".format(
                monitor_count
            )
        )
    sanitized.write_text(text, encoding="utf-8")
    return sanitized


def prepare_and_run_cfx(cfg: dict[str, Any], resume: bool) -> None:
    """Build remeshed definitions from immutable seed physics, then solve."""
    out = Path(cfg["paths"]["output_dir"])
    plan = json.loads((out/"study_plan.json").read_text(encoding="utf-8"))
    template = Path(cfg["cfx"]["pre_session_template"])
    if not template.is_file():
        raise FileNotFoundError(
            "CFX-Pre remesh session template not found: {}. Record one reload/CCL/"
            "write-DEF session using the documented tokens before CFX execution."
            .format(template)
        )
    opening = json.loads((out/"poi"/"opening_terminal.json").read_text())
    seed_contract = validate_cfx_seed_contract(cfg, out/"cfx_seed_contract.json")
    seed_hash = seed_contract["seed_sha256"]
    seed_ccl = prepare_cfx_seed_physics_ccl(cfg, out)
    for case in plan["cases"]:
        case_id = case["case_id"]; mesh_dir = out/"meshes"/case_id
        if not (mesh_dir/"mesh.msh").is_file(): continue
        cfx_dir = out/"cfx"/case_id; cfx_dir.mkdir(parents=True, exist_ok=True)
        res, definition = cfx_dir/(case_id+".res"), cfx_dir/(case_id+".def")
        validation_path = cfx_dir/"case_validation.json"
        if resume and validation_path.is_file():
            prior = json.loads(validation_path.read_text(encoding="utf-8"))
            if (
                prior.get("pass") is True
                and prior.get("seed_sha256") == seed_hash
                and prior.get("physics_control_sha256") == seed_contract["physics_control_sha256"]
                and res.is_file()
            ):
                continue
            if (
                prior.get("pass") is False
                and cfg["cfx"].get("continue_after_case_failure", False)
                and prior.get("seed_sha256") == seed_hash
                and prior.get("physics_control_sha256") == seed_contract["physics_control_sha256"]
                and prior.get("solver", {}).get("acceptance_residual_target")
                    == float(cfg["cfx"].get(
                        "acceptance_residual_target", cfg["cfx"]["residual_target"]
                    ))
                and res.is_file()
            ):
                print("[CFX] Skipping previously rejected diagnostic case {}.".format(case_id), flush=True)
                continue
        flow_dir = cfx_dir/"boundaries"
        flow_csv = flow_dir/"giessen_cfx_outlet_flow_fractions.csv"
        flow_cmd = [sys.executable, "-m", "coronary_sdf.flow_fractions",
                    cfg["paths"]["amira"], str(mesh_dir/"mesh.msh"), str(flow_dir),
                    "--opening-terminal-id={}".format(opening["opening_terminal_id"]),
                    "--inlet-mass-flow-kg-s={:.12g}".format(
                        float(cfg["cfx"]["expected_inlet_mass_flow_kg_s"])
                    ),
                    "--boundary-only"]
        ccl = flow_dir/"giessen_boundary_conditions.ccl"
        ccl_is_current = (
            ccl.is_file()
            and "Option = Opening Pressure and Direction" in ccl.read_text(
                encoding="utf-8", errors="replace"
            )
            and "&replace BOUNDARY:" in ccl.read_text(
                encoding="utf-8", errors="replace"
            )
            and "Numeric fixed-outlet mass flows" in ccl.read_text(
                encoding="utf-8", errors="replace"
            )
            and "Fluent cap zone types enforced (mass-flow inlet)" in ccl.read_text(
                encoding="utf-8", errors="replace"
            )
            and "Mass Flow Rate Area = As Specified" in ccl.read_text(
                encoding="utf-8", errors="replace"
            )
            and "Option = Bulk Mass Flow Rate" not in ccl.read_text(
                encoding="utf-8", errors="replace"
            )
        )
        if not flow_csv.is_file() or not ccl_is_current:
            _stream_command(flow_cmd, Path(__file__).resolve().parent.parent, cfx_dir/"flow_fractions.log")
        opening_verification = verify_opening_mapping(flow_dir, opening, cfx_dir)
        if not res.is_file():
            session = cfx_dir/"prepare.pre"
            session.write_text(_render_template(template, {
                # flow_fractions writes a topology-identical Fluent mesh whose
                # cap zones have the stable OutletNNN/InletNNN names referenced
                # by the boundary-only CCL.  Import that file, not the raw
                # Simpleware export with COR_* region names.
                "SEED_CFX":cfg["paths"]["cfx_seed"], "MESH_MSH":str(flow_dir/"mesh_renamed.msh"),
                "SEED_CCL":str(seed_ccl), "BOUNDARY_CCL":str(ccl),
                "OUTPUT_DEF":str(definition), "CASE_ID":case_id,
            }), encoding="utf-8")
            _stream_command([cfg["executables"]["cfx_pre"], "-batch", str(session)], cfx_dir, cfx_dir/"cfx_pre.log")
            if not definition.is_file(): raise RuntimeError("CFX-Pre did not create {}".format(definition))
            _stream_command([cfg["executables"]["cfx_solver"], "-batch", "-def", str(definition), "-part", str(cfg["cfx"]["cores"]), "-par-local", "-double", "-fullname", str(cfx_dir/case_id)], cfx_dir, cfx_dir/"solver_console.log")
            if not res.is_file():
                found = sorted(cfx_dir.glob(case_id+"*.res"))
                if found: shutil.copy2(found[-1], res)
        if not res.is_file(): raise RuntimeError("CFX solve did not produce a result for {}".format(case_id))
        solved_contract = extract_cfx_seed_contract(definition if definition.is_file() else res)
        if solved_contract["physics_control_sha256"] != seed_contract["physics_control_sha256"]:
            raise RuntimeError("Remeshed CFX definition/result changed authoritative physics controls")
        solver_outputs = list(cfx_dir.glob("*.out"))
        if not solver_outputs:
            solver_outputs = [cfx_dir/"solver_console.log"]
        solver_output = max(solver_outputs, key=lambda path:path.stat().st_mtime)
        acceptance_target = float(
            cfg["cfx"].get("acceptance_residual_target", cfg["cfx"]["residual_target"])
        )
        solver_status = parse_cfx_solver_output(solver_output, acceptance_target)
        # A complex coronary tree can need more than the seed's per-run outer
        # iteration allowance.  Continue on the *same* mesh/result while
        # retaining the immutable DEF and its convergence target.  This does
        # not interpolate between meshes and therefore does not violate the
        # study's prohibition on cross-mesh restart interpolation.
        max_continuations = int(cfg["cfx"].get("max_same_mesh_continuations", 2))
        continuation = 0
        while not solver_status["converged"] and continuation < max_continuations:
            continuation += 1
            continuation_stem = cfx_dir/(case_id+"_continue_{}".format(continuation))
            print(
                "[CFX] {} reached its iteration allowance above the residual target; "
                "continuing the same-mesh solution ({}/{}).".format(
                    case_id, continuation, max_continuations
                ),
                flush=True,
            )
            _stream_command(
                [
                    cfg["executables"]["cfx_solver"], "-batch", "-def", str(definition),
                    "-continue-from-file", str(res), "-part", str(cfg["cfx"]["cores"]),
                    "-par-local", "-double", "-fullname", str(continuation_stem),
                ],
                cfx_dir,
                cfx_dir/("solver_continue_{}.log".format(continuation)),
            )
            continuation_res = continuation_stem.with_suffix(".res")
            continuation_out = continuation_stem.with_suffix(".out")
            if not continuation_res.is_file():
                raise RuntimeError(
                    "CFX continuation {} did not produce a result for {}".format(
                        continuation, case_id
                    )
                )
            shutil.copy2(continuation_res, res)
            solver_output = continuation_out if continuation_out.is_file() else cfx_dir/("solver_continue_{}.log".format(continuation))
            solver_status = parse_cfx_solver_output(solver_output, acceptance_target)
            solver_status["continuations"] = continuation
        solver_status.setdefault("continuations", continuation)
        if not solver_status["converged"]:
            rejection = {
                "case_id": case_id,
                "pass": False,
                "rejection_reasons": ["residual_convergence_target_not_met"],
                "seed_sha256": seed_hash,
                "physics_control_sha256": seed_contract["physics_control_sha256"],
                "solver": solver_status,
                "opening_mapping": opening_verification,
            }
            validation_path.write_text(json.dumps(rejection, indent=2) + "\n", encoding="utf-8")
            if cfg["cfx"].get("continue_after_case_failure", False):
                print(
                    "[CFX][REJECTED] {} did not meet the residual target; "
                    "recorded diagnostic result and continuing to the next mesh.".format(case_id),
                    flush=True,
                )
                continue
            raise RuntimeError("CFX result {} did not meet the residual convergence target".format(case_id))
        boundary_audit = run_cfx_boundary_audit(cfg, res, flow_csv, cfx_dir)
        validation = {
            "case_id": case_id,
            "pass": True,
            "seed_sha256": seed_hash,
            "physics_control_sha256": seed_contract["physics_control_sha256"],
            "solver": solver_status,
            "boundaries": boundary_audit,
            "opening_mapping": opening_verification,
        }
        validation_path.write_text(json.dumps(validation, indent=2) + "\n", encoding="utf-8")
        if file_sha256(cfg["paths"]["cfx_seed"]) != seed_hash:
            raise RuntimeError("Immutable CFX seed changed")


def run_extract_stage(cfg: dict[str, Any], resume: bool) -> None:
    out = Path(cfg["paths"]["output_dir"])
    plan = json.loads((out/"study_plan.json").read_text())
    for case in plan["cases"]:
        case_id = case["case_id"]; res = out/"cfx"/case_id/(case_id+".res")
        if not res.is_file(): continue
        extract = out/"extracted"/case_id
        if not (resume and (extract/(case_id+"_wall.npz")).is_file()):
            cmd = [sys.executable, "-m", "coronary_sdf.cfx_extract", "--res", str(res), "--out", str(extract), "--fluent-mesh", str(out/"meshes"/case_id/"mesh.msh")]
            _stream_command(cmd, Path(__file__).resolve().parent.parent, extract/"extract.log")
        if not (resume and (extract/"metrics.csv").is_file()):
            print("Computing POI/segment/order/radius metrics for {}.".format(case_id), flush=True)
            extract_case_metrics(cfg, case_id)


def _npz_fields(path: Path) -> dict[str, np.ndarray]:
    data = np.load(path, allow_pickle=True)
    fields = {str(name): data["values"][:, i] for i, name in enumerate(data["columns"])}
    if "surface_control_area" in data:
        fields["Surface Control Area"] = np.asarray(data["surface_control_area"])
    return fields


def _station_pressure(volume: dict[str, np.ndarray], xyz_mm: Sequence[float], radius_mm: float) -> float:
    xyz_m = np.asarray(xyz_mm, dtype=float) / 1000.0
    points = np.column_stack([volume[k] for k in ("X", "Y", "Z")])
    distance = np.linalg.norm(points-xyz_m, axis=1)
    use = distance <= max(float(radius_mm)/1000.0, 2e-5)
    if use.sum() < 10:
        use = np.argsort(distance)[:min(50, len(distance))]
    weights = volume.get("Volume of Finite Volumes")
    return float(np.average(volume["Pressure"][use], weights=None if weights is None else weights[use]))


def _load_cropped_graph(path: Path) -> dict[int, dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    result = {}
    for edge in payload["edges"]:
        coords = np.asarray(edge["points_mm"], dtype=float)
        radii = np.asarray(edge["radii_mm"], dtype=float)
        if len(coords) < 2 or len(coords) != len(radii):
            continue
        arc = _arc(coords)
        if arc[-1] <= 0:
            continue
        result[int(edge["edge_id"])] = {
            **edge, "coords": coords, "radii": radii, "arc": arc,
            "length_mm": float(arc[-1]),
            "median_radius_mm": float(np.nanmedian(radii)),
        }
    if not result:
        raise RuntimeError("STL-cropped graph contains no usable edges")
    return result


def _dense_graph_samples(
    edges: dict[int, dict[str, Any]], maximum_spacing_mm: float = 0.15
) -> dict[str, np.ndarray]:
    xyz, edge_ids, arcs, radii = [], [], [], []
    for edge_id, edge in sorted(edges.items()):
        positions = np.linspace(
            0.0, edge["length_mm"],
            max(2, int(math.ceil(edge["length_mm"] / maximum_spacing_mm)) + 1),
        )
        xyz.append(np.column_stack([
            np.interp(positions, edge["arc"], edge["coords"][:, axis])
            for axis in range(3)
        ]))
        edge_ids.append(np.full(len(positions), edge_id, dtype=np.int64))
        arcs.append(positions)
        radii.append(np.interp(positions, edge["arc"], edge["radii"]))
    return {
        "xyz": np.concatenate(xyz), "edge_id": np.concatenate(edge_ids),
        "arc": np.concatenate(arcs), "radius": np.concatenate(radii),
    }


def _edge_station(edge: dict[str, Any], fraction: float) -> dict[str, Any]:
    s = float(np.clip(fraction, 0.0, 1.0)) * edge["length_mm"]
    coords, arc = edge["coords"], edge["arc"]
    xyz = np.array([np.interp(s, arc, coords[:, axis]) for axis in range(3)])
    radius = float(np.interp(s, arc, edge["radii"]))
    tangent, normal, binormal = parallel_transport_frames(coords)
    index = int(np.argmin(abs(arc - s)))
    return {
        "s": s, "xyz": xyz, "radius": radius, "tangent": tangent[index],
        "normal": normal[index], "binormal": binormal[index],
    }


def _approved_pois(path: Path) -> list[dict[str, Any]]:
    accepted = {"yes", "y", "true", "1"}
    with path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    return [row for row in rows if str(row.get("approved", "")).strip().lower() in accepted]


def _write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "metric_id", "metric_class", "value", "normalization", "scope",
        "poi_id", "edge_id", "strahler", "radius_bin", "flag",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def extract_case_metrics(cfg: dict[str, Any], case_id: str) -> Path:
    """Create deterministic station/segment/order/radius metrics for one case."""
    out = Path(cfg["paths"]["output_dir"])
    extract = out/"extracted"/case_id
    wall_files = sorted(extract.glob("*_wall.npz"))
    volume_files = sorted(extract.glob("*_volume.npz"))
    if not wall_files or not volume_files:
        raise RuntimeError("Missing wall/volume extraction for {}".format(case_id))
    graph_path = out/"meshes"/case_id/"cropped_amira_graph.json"
    edges = _load_cropped_graph(graph_path)
    graph = _dense_graph_samples(edges)
    tree = cKDTree(graph["xyz"])
    wall, volume = _npz_fields(wall_files[-1]), _npz_fields(volume_files[-1])
    if "Surface Control Area" not in wall:
        raise RuntimeError("{} wall export has no lumped Fluent-face areas".format(case_id))
    wall_xyz = 1000.0 * np.column_stack([wall[k] for k in ("X", "Y", "Z")])
    volume_xyz = 1000.0 * np.column_stack([volume[k] for k in ("X", "Y", "Z")])
    wall_distance, wall_sample = tree.query(wall_xyz)
    volume_distance, volume_sample = tree.query(volume_xyz)
    wall_owner = graph["edge_id"][wall_sample]
    volume_owner = graph["edge_id"][volume_sample]
    wall_arc = graph["arc"][wall_sample]
    volume_arc = graph["arc"][volume_sample]
    wall_valid = wall_distance <= 2.25 * np.maximum(graph["radius"][wall_sample], 0.01)
    volume_valid = volume_distance <= 1.35 * np.maximum(graph["radius"][volume_sample], 0.01)
    wall_area = np.asarray(wall["Surface Control Area"], dtype=float)
    volume_weight = np.asarray(
        volume.get("Volume of Finite Volumes", np.ones(len(volume_xyz))), dtype=float
    )
    poi_path = out/"poi"/"poi_candidates.csv"
    pois = _approved_pois(poi_path)
    # Inlet and opening stations are deterministic boundary references needed
    # to normalize every pressure metric.  Include those two even when the
    # optional anatomical POI review has not yet approved any interior sites;
    # this does not auto-approve the remaining candidates.
    present_kinds = {row.get("kind") for row in pois}
    if not {"inlet", "opening"}.issubset(present_kinds):
        with poi_path.open(newline="", encoding="utf-8-sig") as handle:
            candidates = list(csv.DictReader(handle))
        used = {row.get("poi_id") for row in pois}
        for row in candidates:
            if row.get("kind") in {"inlet", "opening"} and row.get("poi_id") not in used:
                pois.append(row)
                used.add(row.get("poi_id"))
    if not pois:
        raise RuntimeError("No approved POIs")

    station_data = []
    for poi in pois:
        edge_id = int(poi["edge_id"])
        if edge_id not in edges:
            # A POI on a raw-graph portion removed by the exact STL crop cannot
            # represent this mesh and is explicitly reported rather than moved.
            station_data.append({"poi": poi, "flag": "removed_by_stl_crop"})
            continue
        edge = edges[edge_id]
        station = _edge_station(edge, float(poi["arc_fraction"]))
        axial = (volume_xyz - station["xyz"]) @ station["tangent"]
        radial = np.linalg.norm(
            volume_xyz - station["xyz"] - axial[:, None] * station["tangent"], axis=1
        )
        slab = max(0.02, 0.15 * station["radius"])
        pressure_use = (
            volume_valid & (volume_owner == edge_id) & (abs(axial) <= slab)
            & (radial <= 1.25 * station["radius"])
        )
        if pressure_use.sum() < 10:
            pressure_use = (
                volume_valid & (volume_owner == edge_id) & (abs(axial) <= 2.0 * slab)
                & (radial <= 1.5 * station["radius"])
            )
        boundary_station_shifted = False
        if pressure_use.sum() < 3 and poi.get("kind") in {"inlet", "opening"}:
            # Clipping planes intentionally sit inward from the raw Amira
            # endpoint.  Move the normalization station to the first retained
            # mesh section, then a small radius-scaled distance farther inward
            # so the pressure slab contains volume nodes rather than the cap.
            owned = volume_valid & (volume_owner == edge_id)
            observed_arc = volume_arc[owned]
            if observed_arc.size:
                inward = max(0.02, 0.15 * station["radius"])
                if float(poi["arc_fraction"]) <= 0.5:
                    target_s = min(float(observed_arc.min()) + inward, float(observed_arc.max()))
                else:
                    target_s = max(float(observed_arc.max()) - inward, float(observed_arc.min()))
                station = _edge_station(edge, target_s / edge["length_mm"])
                axial = (volume_xyz - station["xyz"]) @ station["tangent"]
                radial = np.linalg.norm(
                    volume_xyz - station["xyz"] - axial[:, None] * station["tangent"], axis=1
                )
                slab = max(0.02, 0.15 * station["radius"])
                pressure_use = (
                    owned & (abs(axial) <= slab)
                    & (radial <= 1.5 * station["radius"])
                )
                boundary_station_shifted = True
        if pressure_use.sum() < 3:
            station_data.append({"poi": poi, "flag": "insufficient_cross_section_nodes"})
            continue
        pressure = float(np.average(volume["Pressure"][pressure_use], weights=volume_weight[pressure_use]))

        desired = min(2.0 * station["radius"], edge["length_mm"])
        lower = float(np.clip(station["s"] - desired/2.0, 0.0, edge["length_mm"] - desired))
        upper = lower + desired
        wall_use = wall_valid & (wall_owner == edge_id) & (wall_arc >= lower) & (wall_arc <= upper)
        if wall_use.sum() < 8:
            station_data.append({"poi": poi, "flag": "insufficient_wall_band_nodes"})
            continue
        radial_vectors = wall_xyz[wall_use] - station["xyz"]
        axial_wall = radial_vectors @ station["tangent"]
        radial_vectors -= axial_wall[:, None] * station["tangent"]
        sectors = sector_indices(radial_vectors, station["normal"], station["binormal"])
        wss = wss_band_statistics(wall["Wall Shear"][wall_use], wall_area[wall_use], sectors)
        flags = []
        if desired < 2.0 * station["radius"] * 0.999:
            flags.append("short_band")
        if boundary_station_shifted:
            flags.append("boundary_station_shifted_inward")
        flag = ";".join(flags)
        station_data.append({
            "poi": poi, "edge": edge, "station": station, "pressure": pressure,
            "wss": wss, "band_length_mm": desired, "flag": flag,
        })

    inlet = next((item for item in station_data if item["poi"]["kind"] == "inlet" and "pressure" in item), None)
    opening = next((item for item in station_data if item["poi"]["kind"] == "opening" and "pressure" in item), None)
    if inlet is None or opening is None:
        raise RuntimeError("Approved inlet/opening POIs must yield valid pressure sections")
    pressure_scale = abs(inlet["pressure"] - opening["pressure"])
    if pressure_scale <= 1e-15:
        raise RuntimeError("Inlet-to-opening pressure drop is zero")

    rows: list[dict[str, Any]] = []
    station_rows: list[dict[str, Any]] = []
    for item in station_data:
        poi = item["poi"]
        if "pressure" not in item:
            station_rows.append({
                "metric_id": poi["poi_id"] + ":unavailable", "metric_class": "diagnostic",
                "value": math.nan, "normalization": math.nan, "scope": "poi",
                "poi_id": poi["poi_id"], "edge_id": poi["edge_id"], "flag": item["flag"],
            })
            continue
        prefix = poi["poi_id"]
        common = {"scope": "poi", "poi_id": prefix, "edge_id": poi["edge_id"], "strahler": poi["strahler"], "flag": item["flag"]}
        wss = item["wss"]; wss_norm = wss["area_weighted_mean"]
        values = {
            "wss_mean": wss["area_weighted_mean"], "wss_p95": wss["area_weighted_p95"],
            "wss_sector_min": wss["sector_min"], "wss_sector_max": wss["sector_max"],
        }
        for name, value in values.items():
            station_rows.append({**common, "metric_id": prefix+":"+name, "metric_class": name, "value": value, "normalization": wss_norm})
        for sector, value in enumerate(wss["sector_means"]):
            station_rows.append({**common, "metric_id": "{}:wss_sector_{:02d}".format(prefix, sector+1), "metric_class": "diagnostic_wss_sector", "value": value, "normalization": wss_norm})
        for name in ("raw_min", "raw_max"):
            station_rows.append({**common, "metric_id": "{}:wss_{}".format(prefix, name), "metric_class": "diagnostic_wss_raw", "value": wss[name], "normalization": wss_norm})
        station_rows.extend([
            {**common, "metric_id": prefix+":pressure_static", "metric_class": "pressure_static", "value": item["pressure"], "normalization": pressure_scale},
            {**common, "metric_id": prefix+":pressure_drop", "metric_class": "pressure_drop", "value": inlet["pressure"]-item["pressure"], "normalization": pressure_scale},
        ])
    rows.extend(station_rows)

    segment_rows: list[dict[str, Any]] = []
    for edge_id, edge in sorted(edges.items()):
        wu = wall_valid & (wall_owner == edge_id)
        vu = volume_valid & (volume_owner == edge_id)
        if not wu.any() or not vu.any():
            continue
        wss_mean = float(np.average(wall["Wall Shear"][wu], weights=wall_area[wu]))
        wss_p95 = weighted_quantile(wall["Wall Shear"][wu], wall_area[wu], 0.95)
        wall_pressure = float(np.average(wall["Pressure"][wu], weights=wall_area[wu]))
        pressure = float(np.average(volume["Pressure"][vu], weights=volume_weight[vu]))
        common = {"scope":"segment", "edge_id":edge_id, "strahler":edge.get("strahler"), "flag":""}
        for name, value, norm in (
            ("wss_mean", wss_mean, wss_mean), ("wss_p95", wss_p95, wss_mean),
            ("pressure_static", pressure, pressure_scale),
            ("pressure_drop", inlet["pressure"]-pressure, pressure_scale),
            ("pressure_wall", wall_pressure, pressure_scale),
        ):
            segment_rows.append({**common, "metric_id":"edge{}:{}".format(edge_id,name), "metric_class":name, "value":value, "normalization":norm})
    rows.extend(segment_rows)

    radius_edges = np.geomspace(
        min(edge["median_radius_mm"] for edge in edges.values()),
        max(edge["median_radius_mm"] for edge in edges.values()), 9,
    )
    group_rows: list[dict[str, Any]] = []
    for scope, labels in (
        ("strahler", sorted({int(edge.get("strahler") or 0) for edge in edges.values()})),
        ("radius_bin", list(range(1, 9))),
    ):
        for label in labels:
            if scope == "strahler":
                members = {eid for eid, edge in edges.items() if int(edge.get("strahler") or 0) == label}
            else:
                members = {
                    eid for eid, edge in edges.items()
                    if min(7, int(np.searchsorted(radius_edges, edge["median_radius_mm"], side="right")-1)) + 1 == label
                }
            wu = wall_valid & np.isin(wall_owner, list(members)); vu = volume_valid & np.isin(volume_owner, list(members))
            if not wu.any() or not vu.any():
                continue
            wss_mean = float(np.average(wall["Wall Shear"][wu], weights=wall_area[wu]))
            wall_pressure = float(np.average(wall["Pressure"][wu], weights=wall_area[wu]))
            pressure = float(np.average(volume["Pressure"][vu], weights=volume_weight[vu]))
            common = {"scope":scope, "strahler":label if scope=="strahler" else "", "radius_bin":label if scope=="radius_bin" else "", "flag":""}
            prefix = "{}{}".format(scope, label)
            for name, value, norm in (
                ("wss_mean",wss_mean,wss_mean),
                ("wss_p95",weighted_quantile(wall["Wall Shear"][wu],wall_area[wu],0.95),wss_mean),
                ("pressure_static",pressure,pressure_scale),
                ("pressure_drop",inlet["pressure"]-pressure,pressure_scale),
                ("pressure_wall",wall_pressure,pressure_scale),
            ):
                group_rows.append({**common,"metric_id":prefix+":"+name,"metric_class":name,"value":value,"normalization":norm})
    rows.extend(group_rows)
    _write_rows(extract/"station_metrics.csv", station_rows)
    _write_rows(extract/"segment_metrics.csv", segment_rows)
    _write_rows(extract/"group_metrics.csv", group_rows)
    metrics = extract/"metrics.csv"; _write_rows(metrics, rows)
    assignment = {
        "case_id": case_id, "cropped_edge_count": len(edges),
        "wall_nodes": len(wall_xyz), "wall_nodes_assigned": int(wall_valid.sum()),
        "volume_nodes": len(volume_xyz), "volume_nodes_assigned": int(volume_valid.sum()),
        "pressure_normalization_pa": pressure_scale,
    }
    (extract/"metric_assignment.json").write_text(json.dumps(assignment, indent=2)+"\n", encoding="utf-8")
    return metrics


def maybe_select_boundary_layer(cfg: dict[str, Any]) -> int | None:
    out = Path(cfg["paths"]["output_dir"]); selection = out/"boundary_layer"/"selection.json"
    if selection.is_file(): return int(json.loads(selection.read_text())["selected_layers"])
    results = {}
    for layers in (4, 6, 8):
        case_id = "bl_adaptive_{}layers".format(layers)
        metrics = out/"extracted"/case_id/"metrics.csv"
        if not metrics.is_file(): return None
        with metrics.open(newline="", encoding="utf-8-sig") as handle:
            results[layers] = {
                row["metric_id"]: {
                    "class": row["metric_class"], "value": float(row["value"]),
                    "normalization": float(row["normalization"]),
                }
                for row in csv.DictReader(handle)
                if not row["metric_class"].startswith("diagnostic")
            }

    def comparison(coarse: int, fine: int) -> dict[str, Any]:
        common = sorted(set(results[coarse]) & set(results[fine]))
        wss, pressure = [], []
        for metric_id in common:
            fine_row = results[fine][metric_id]
            delta = normalized_change(
                results[coarse][metric_id]["value"], fine_row["value"],
                fine_row["normalization"],
            )
            (wss if fine_row["class"].startswith("wss") else pressure).append(delta)
        if not wss or not pressure:
            raise RuntimeError("Boundary-layer cases have no common WSS/pressure metrics")
        return {
            "coarse_layers": coarse, "fine_layers": fine,
            "maximum_wss_change": max(wss), "maximum_pressure_change": max(pressure),
            "metric_count": len(common),
            "pass": max(wss) <= float(cfg["boundary_layer"]["wss_tolerance"])
                    and max(pressure) <= float(cfg["boundary_layer"]["pressure_tolerance"]),
        }

    compare_46, compare_68 = comparison(4, 6), comparison(6, 8)
    if not compare_68["pass"]:
        raise RuntimeError("6-to-8 prism-layer comparison failed; wall mesh is unconverged")
    selected = 4 if compare_46["pass"] else 6
    selection.parent.mkdir(parents=True, exist_ok=True)
    selection.write_text(json.dumps({"selected_layers":selected, "comparisons":[compare_46, compare_68], "wss_tolerance":cfg["boundary_layer"]["wss_tolerance"], "pressure_tolerance":cfg["boundary_layer"]["pressure_tolerance"]}, indent=2)+"\n", encoding="utf-8")
    print("Boundary-layer study selected {} layers.".format(selected), flush=True)
    return selected


def analyse_global_group_sensitivity(cfg: dict[str, Any]) -> Path:
    """Create Strahler- and radius-resolved global-mesh sensitivity outputs."""
    out = Path(cfg["paths"]["output_dir"])
    reports = out/"reports"
    reports.mkdir(parents=True, exist_ok=True)
    case_ids = ["global_" + level for level in CASE_LEVELS]
    counts: dict[str, int] = {}
    grouped: dict[tuple[str, int, str], dict[str, tuple[float, float]]] = defaultdict(dict)
    for case_id in case_ids:
        stats_path = out/"meshes"/case_id/"mesh_stats.json"
        metrics_path = out/"extracted"/case_id/"group_metrics.csv"
        if not stats_path.is_file() or not metrics_path.is_file():
            continue
        counts[case_id] = int(json.loads(stats_path.read_text(encoding="utf-8"))["total_elements"])
        with metrics_path.open(newline="", encoding="utf-8-sig") as handle:
            for row in csv.DictReader(handle):
                scope = row.get("scope", "")
                label_text = row.get("strahler", "") if scope == "strahler" else row.get("radius_bin", "")
                if scope not in {"strahler", "radius_bin"} or not label_text:
                    continue
                grouped[(scope, int(label_text), row["metric_class"])][case_id] = (
                    float(row["value"]), float(row["normalization"])
                )

    missing = [case_id for case_id in case_ids if case_id not in counts]
    report: dict[str, Any] = {
        "status": "incomplete" if missing else "pass",
        "family": "global", "missing_cases": missing, "scopes": {},
    }
    radius_edges: list[float] = []
    opening_path = out/"poi"/"opening_terminal.json"
    if opening_path.is_file():
        radius_edges = [
            float(value) for value in
            json.loads(opening_path.read_text(encoding="utf-8")).get("radius_bin_edges_mm", [])
        ]

    for scope in ("strahler", "radius_bin"):
        labels = sorted({label for (item_scope, label, _) in grouped if item_scope == scope})
        value_rows: list[dict[str, Any]] = []
        summary_rows: list[dict[str, Any]] = []
        for label in labels:
            if scope == "strahler":
                display = "Order {}".format(label)
                lower = upper = ""
            elif len(radius_edges) >= label + 1:
                lower, upper = radius_edges[label-1], radius_edges[label]
                display = "{:.3f}-{:.3f} mm".format(lower, upper)
            else:
                lower = upper = ""
                display = "Bin {}".format(label)
            for metric in ("wss_mean", "wss_p95", "pressure_static", "pressure_drop", "pressure_wall"):
                values = grouped.get((scope, label, metric), {})
                if not all(case_id in values for case_id in case_ids):
                    continue
                fine_value, fine_norm = values[case_ids[-1]]
                for level, case_id in zip(CASE_LEVELS, case_ids):
                    value, normalization = values[case_id]
                    value_rows.append({
                        "scope": scope, "group": label, "group_label": display,
                        "radius_lower_mm": lower, "radius_upper_mm": upper,
                        "metric": metric, "case_id": case_id, "level": level,
                        "elements": counts[case_id], "value": value,
                        "normalization": normalization,
                        "normalized_difference_from_l4": normalized_change(
                            value, fine_value, fine_norm
                        ),
                    })
                previous_value = values[case_ids[-2]][0]
                tolerance = 0.05 if metric.startswith("wss") else 0.01
                finest_change = normalized_change(previous_value, fine_value, fine_norm)
                gci = observed_order_gci(
                    [values[case_id][0] for case_id in case_ids[-3:]],
                    [counts[case_id] for case_id in case_ids[-3:]],
                )
                row: dict[str, Any] = {
                    "scope": scope, "group": label, "group_label": display,
                    "radius_lower_mm": lower, "radius_upper_mm": upper,
                    "metric": metric, "l4_value": fine_value,
                    "l3_to_l4_normalized_change": finest_change,
                    "tolerance": tolerance, "pass": finest_change <= tolerance,
                    "gci_status": gci.get("status", ""),
                    "observed_order": gci.get("observed_order", ""),
                    "gci_fine": gci.get("gci_fine", ""),
                }
                for level, case_id in zip(CASE_LEVELS, case_ids):
                    row[level + "_elements"] = counts[case_id]
                    row[level + "_value"] = values[case_id][0]
                    row[level + "_difference_from_l4"] = normalized_change(
                        values[case_id][0], fine_value, fine_norm
                    )
                summary_rows.append(row)
                if not row["pass"]:
                    report["status"] = "non_converged" if not missing else "incomplete"

        values_path = reports/("global_{}_sensitivity_values.csv".format(scope))
        summary_path = reports/("global_{}_sensitivity_summary.csv".format(scope))
        if value_rows:
            with values_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(value_rows[0]))
                writer.writeheader(); writer.writerows(value_rows)
        if summary_rows:
            with summary_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0]))
                writer.writeheader(); writer.writerows(summary_rows)
        report["scopes"][scope] = {
            "groups": len(labels), "metrics": len(summary_rows),
            "values_csv": str(values_path), "summary_csv": str(summary_path),
        }

        if not value_rows or not summary_rows:
            continue
        try:
            import matplotlib.pyplot as plt

            metric_specs = (
                ("wss_mean", "Mean WSS", "Pa"),
                ("wss_p95", "WSS P95", "Pa"),
                ("pressure_drop", "Inlet-referenced pressure drop", "Pa"),
                ("pressure_static", "Mean static pressure", "Pa"),
            )
            figure, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
            for axis, (metric, title, unit) in zip(axes.flat, metric_specs):
                for level, case_id in zip(CASE_LEVELS, case_ids):
                    points = [row for row in value_rows if row["metric"] == metric and row["case_id"] == case_id]
                    points.sort(key=lambda row: row["group"])
                    if points:
                        x = [
                            math.sqrt(float(row["radius_lower_mm"]) * float(row["radius_upper_mm"]))
                            if scope == "radius_bin" else row["group"] for row in points
                        ]
                        axis.plot(x, [row["value"] for row in points], marker="o", label=level.upper())
                if scope == "radius_bin":
                    axis.set_xscale("log"); axis.set_xlabel("Median-radius bin midpoint (mm)")
                else:
                    axis.set_xlabel("Strahler order"); axis.set_xticks(labels)
                axis.set_ylabel(unit); axis.set_title(title); axis.grid(True, alpha=.3); axis.legend(fontsize=8)
            figure.savefig(reports/("global_{}_sensitivity.png".format(scope)), dpi=220)
            plt.close(figure)

            heat_metrics = (("wss_mean", "Mean WSS error vs L4"), ("pressure_drop", "Pressure-drop error vs L4"))
            figure, axes = plt.subplots(1, 2, figsize=(11, max(4.5, 0.55*len(labels))), constrained_layout=True)
            for axis, (metric, title) in zip(axes, heat_metrics):
                matrix = np.full((len(labels), 3), np.nan)
                for row_index, label in enumerate(labels):
                    for column, level in enumerate(CASE_LEVELS[:3]):
                        match = next((row for row in value_rows if row["metric"] == metric and row["group"] == label and row["level"] == level), None)
                        if match:
                            matrix[row_index, column] = 100.0 * match["normalized_difference_from_l4"]
                image = axis.imshow(matrix, aspect="auto", cmap="viridis")
                axis.set_xticks(range(3), [level.upper() for level in CASE_LEVELS[:3]])
                axis.set_yticks(range(len(labels)), [
                    next(row["group_label"] for row in value_rows if row["group"] == label) for label in labels
                ])
                axis.set_title(title); figure.colorbar(image, ax=axis, label="Normalized difference (%)")
                for i in range(matrix.shape[0]):
                    for j in range(matrix.shape[1]):
                        if np.isfinite(matrix[i, j]): axis.text(j, i, "{:.2f}".format(matrix[i, j]), ha="center", va="center", color="white" if matrix[i,j] > np.nanmax(matrix)*.45 else "black", fontsize=8)
            figure.savefig(reports/("global_{}_error_heatmap.png".format(scope)), dpi=220)
            plt.close(figure)

            table_rows = []
            for label in labels:
                wss = next((row for row in summary_rows if row["group"] == label and row["metric"] == "wss_mean"), None)
                pressure = next((row for row in summary_rows if row["group"] == label and row["metric"] == "pressure_drop"), None)
                if wss and pressure:
                    table_rows.append([
                        wss["group_label"], "{:.4g}".format(wss["l4_value"]),
                        "{:.2f}%".format(100*wss["l3_to_l4_normalized_change"]),
                        "{:.4g}".format(pressure["l4_value"]),
                        "{:.2f}%".format(100*pressure["l3_to_l4_normalized_change"]),
                        "PASS" if wss["pass"] and pressure["pass"] else "FAIL",
                    ])
            figure, axis = plt.subplots(figsize=(12, max(2.6, .48*len(table_rows)+1.4)))
            axis.axis("off")
            table = axis.table(
                cellText=table_rows,
                colLabels=("Group", "L4 mean WSS (Pa)", "L3-L4 WSS", "L4 pressure drop (Pa)", "L3-L4 pressure", "Status"),
                cellLoc="center", loc="center",
            )
            table.auto_set_font_size(False); table.set_fontsize(9); table.scale(1, 1.35)
            axis.set_title("Global mesh sensitivity by {}".format("Strahler order" if scope == "strahler" else "radius bin"), pad=16)
            figure.savefig(reports/("global_{}_sensitivity_table.png".format(scope)), dpi=220, bbox_inches="tight")
            plt.close(figure)
        except Exception as exc:
            report["scopes"][scope]["plot_warning"] = str(exc)

    report_path = reports/"global_group_sensitivity.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report_path


def analyse_long_metrics(cfg: dict[str, Any]) -> Path:
    """Analyse standardized per-case ``metrics.csv`` files into pass/fail JSON."""
    out = Path(cfg["paths"]["output_dir"])
    group_report_path = analyse_global_group_sensitivity(cfg)
    selection = out/"boundary_layer"/"selection.json"
    selected = (
        json.loads(selection.read_text())["selected_layers"]
        if selection.is_file() else None
    )
    cases = build_case_plan(cfg, selected, phase="main")
    main = [case for case in cases if case["purpose"] == "main"]
    rows = defaultdict(dict); normalizations = defaultdict(dict); counts = {}
    for case in main:
        metrics = out/"extracted"/case["case_id"]/"metrics.csv"
        stats = out/"meshes"/case["case_id"]/"mesh_stats.json"
        if not metrics.is_file() or not stats.is_file():
            continue
        counts[case["case_id"]] = int(json.loads(stats.read_text())["total_elements"])
        with metrics.open(newline="", encoding="utf-8-sig") as handle:
            for row in csv.DictReader(handle):
                rows[(row["metric_id"], row["metric_class"])][case["case_id"]] = float(row["value"])
                normalizations[(row["metric_id"], row["metric_class"])][case["case_id"]] = float(row["normalization"])
    report = {
        "status": "pass", "metrics": [], "missing_cases": [],
        "global_group_sensitivity": str(group_report_path),
    }
    for family in ("global", "adaptive"):
        ids = ["{}_{}".format(family, level) for level in CASE_LEVELS]
        report["missing_cases"].extend(case for case in ids if case not in counts)
        for (metric_id, metric_class), values in sorted(rows.items()):
            if metric_class.startswith("diagnostic"):
                continue
            if not all(case in values for case in ids[-3:]):
                continue
            fine, previous = values[ids[-1]], values[ids[-2]]
            tol = 0.05 if metric_class.startswith("wss") else 0.01
            normalization = normalizations[(metric_id, metric_class)][ids[-1]]
            change = normalized_change(previous, fine, normalization)
            gci = observed_order_gci([values[c] for c in ids[-3:]], [counts[c] for c in ids[-3:]])
            passed = change <= tol
            report["metrics"].append({"family":family, "metric_id":metric_id, "metric_class":metric_class, "finest_pair_normalized_change":change, "tolerance":tol, "pass":passed, "gci":gci, "achieved_counts":{case:counts[case] for case in ids[-3:]}, "values":{case:values[case] for case in ids[-3:]}})
            if not passed:
                report["status"] = "non_converged"
    if report["missing_cases"]:
        report["status"] = "incomplete"
    if not report["metrics"]:
        report["status"] = "incomplete"
    report["recommendation"] = "Add an approximately 32-million-element level." if report["status"] == "non_converged" else ""
    reports = out/"reports"; reports.mkdir(parents=True, exist_ok=True)
    report_path = reports/"mesh_independence.json"
    report_path.write_text(json.dumps(report, indent=2)+"\n", encoding="utf-8")

    convergence_csv = reports/"convergence_metrics.csv"
    with convergence_csv.open("w", newline="", encoding="utf-8") as handle:
        fields = ("family", "metric_id", "metric_class", "finest_pair_normalized_change", "tolerance", "pass", "gci_status", "observed_order", "gci_fine")
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader()
        for item in report["metrics"]:
            writer.writerow({
                **{key:item.get(key, "") for key in fields},
                "gci_status":item["gci"].get("status", ""),
                "observed_order":item["gci"].get("observed_order", ""),
                "gci_fine":item["gci"].get("gci_fine", ""),
            })

    mesh_rows = []
    for case in main:
        case_id = case["case_id"]
        stats_path = out/"meshes"/case_id/"mesh_stats.json"
        validation_path = out/"cfx"/case_id/"case_validation.json"
        if not stats_path.is_file(): continue
        stats = json.loads(stats_path.read_text())
        validation = json.loads(validation_path.read_text()) if validation_path.is_file() else {}
        mesh_rows.append({
            "case_id":case_id, "family":case["family"], "level":case_id.rsplit("_",1)[-1],
            "target_elements":case["target_elements"], "achieved_elements":stats["total_elements"],
            "core_elements":stats.get("core_elements", ""), "boundary_layer_elements":stats.get("boundary_layer_elements", ""),
            "mesh_seconds":stats.get("mesh_seconds", ""), "peak_memory_bytes":stats.get("peak_memory_bytes", ""),
            "global_h_mm":stats.get("global_h_mm", ""), "n_d":stats.get("n_d", ""),
            "cfx_validation_pass":validation.get("pass", False),
            "mass_imbalance_fraction":validation.get("boundaries",{}).get("mass_imbalance_fraction", ""),
            "last_iteration":validation.get("solver",{}).get("last_iteration", ""),
            "quality_gate":"CFX import/positive-volume solve passed" if validation.get("pass") else "pending",
        })
    mesh_csv = reports/"mesh_quality_runtime.csv"
    if mesh_rows:
        with mesh_csv.open("w", newline="", encoding="utf-8") as handle:
            writer=csv.DictWriter(handle, fieldnames=list(mesh_rows[0])); writer.writeheader(); writer.writerows(mesh_rows)

    comparison_rows = []
    for level in CASE_LEVELS:
        global_id, adaptive_id = "global_"+level, "adaptive_"+level
        for key, values in sorted(rows.items()):
            if global_id not in values or adaptive_id not in values or key[1].startswith("diagnostic"):
                continue
            norm = normalizations[key][adaptive_id]
            comparison_rows.append({
                "level":level, "metric_id":key[0], "metric_class":key[1],
                "global_value":values[global_id], "adaptive_value":values[adaptive_id],
                "normalized_difference":normalized_change(values[global_id],values[adaptive_id],norm),
            })
    if comparison_rows:
        with (reports/"matched_global_vs_adaptive.csv").open("w",newline="",encoding="utf-8") as handle:
            writer=csv.DictWriter(handle,fieldnames=list(comparison_rows[0])); writer.writeheader(); writer.writerows(comparison_rows)

    try:
        import matplotlib.pyplot as plt
        figure, axes = plt.subplots(1, 2, figsize=(10, 4.5), constrained_layout=True)
        for axis, prefix, title in zip(axes, ("wss", "pressure"), ("WSS", "Pressure")):
            for family, marker in (("global", "o"), ("adaptive", "s")):
                x, median_error, maximum_error = [], [], []
                ids = [family+"_"+level for level in CASE_LEVELS]
                for case_id in ids:
                    if case_id not in counts: continue
                    deltas=[]
                    for key, values in rows.items():
                        if key[1].startswith(prefix) and case_id in values and ids[-1] in values:
                            deltas.append(normalized_change(values[case_id],values[ids[-1]],normalizations[key][ids[-1]]))
                    if deltas:
                        x.append(counts[case_id]); median_error.append(max(float(np.median(deltas)),1e-12)); maximum_error.append(max(max(deltas),1e-12))
                if x:
                    axis.plot(x,median_error,marker=marker,label=family+" median")
                    axis.plot(x,maximum_error,marker=marker,linestyle="--",label=family+" maximum")
            axis.set_xscale("log"); axis.set_yscale("log"); axis.set_xlabel("Achieved elements")
            axis.set_ylabel("Normalized difference from finest"); axis.set_title(title); axis.grid(True,which="both",alpha=.3); axis.legend(fontsize=8)
        figure.savefig(reports/"convergence_summary.png",dpi=180); plt.close(figure)
    except Exception as exc:
        report["plot_warning"] = "Could not render convergence plot: {}".format(exc)
        report_path.write_text(json.dumps(report, indent=2)+"\n", encoding="utf-8")
    return report_path


def write_boundary_layer_thickness_audit(cfg: dict[str, Any]) -> Path:
    """Quantify the local prism-layer thickness implied by the radius cap.

    This is a geometry/configuration audit, not a substitute for the paired
    CFD cases.  It makes the distal-vessel hypothesis explicit and records the
    first-layer height and occupied radius fraction for every validated vessel
    section used by the all-vessel sensitivity analysis.
    """
    out = Path(cfg["paths"]["output_dir"])
    audit_dir = out / "boundary_layer" / "thickness_sensitivity"
    audit_dir.mkdir(parents=True, exist_ok=True)
    definitions = (
        out / "plane_sensitivity" / "all_vessels" / "geometry"
        / "all_vessel_plane_definitions.csv"
    )
    if not definitions.is_file():
        raise FileNotFoundError(
            "All-vessel plane definitions are required for the boundary-layer "
            "radius audit: {}".format(definitions)
        )
    bl = cfg["boundary_layer"]
    layers = int(bl.get("mesh_convergence_layers", 5))
    growth = float(bl["growth_ratio"])
    requested = float(bl["requested_total_thickness_mm"])
    ratios = [float(value) for value in bl.get(
        "thickness_ratio_candidates", (0.05, 0.10, 0.15)
    )]
    if layers < 1 or growth <= 0.0:
        raise ValueError("Invalid boundary-layer layer count or growth ratio")
    if abs(growth - 1.0) <= 1.0e-12:
        first_fraction = 1.0 / float(layers)
    else:
        first_fraction = (growth - 1.0) / (growth ** layers - 1.0)

    with definitions.open(newline="", encoding="utf-8-sig") as handle:
        sections = [
            row for row in csv.DictReader(handle)
            if str(row.get("validation_status", "valid")).lower() == "valid"
        ]
    rows: list[dict[str, Any]] = []
    for section in sections:
        radius = float(section["section_equivalent_radius_mm"])
        if radius <= 0.0:
            continue
        for ratio in ratios:
            total = min(requested, ratio * radius)
            rows.append({
                "plane_id": section["plane_id"],
                "edge_id": section["edge_id"],
                "strahler_order": section["strahler_order"],
                "radius_bin": section["radius_bin"],
                "local_radius_mm": radius,
                "maximum_channel_radius_ratio": ratio,
                "total_prism_thickness_mm": total,
                "total_thickness_over_radius": total / radius,
                "first_layer_height_mm": total * first_fraction,
                "first_layer_height_over_radius": total * first_fraction / radius,
                "remaining_core_radius_fraction": 1.0 - total / radius,
                "layers": layers,
                "growth_ratio": growth,
            })
    def write_table(path: Path, table: list[dict[str, Any]]) -> None:
        if not table:
            raise RuntimeError("Boundary-layer thickness audit produced no rows")
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(table[0]))
            writer.writeheader()
            writer.writerows(table)

    audit_csv = audit_dir / "local_radius_thickness_audit.csv"
    write_table(audit_csv, rows)

    summary_rows: list[dict[str, Any]] = []
    for ratio in ratios:
        for radius_bin in sorted({row["radius_bin"] for row in rows}, key=int):
            group = [
                row for row in rows
                if row["maximum_channel_radius_ratio"] == ratio
                and row["radius_bin"] == radius_bin
            ]
            if not group:
                continue
            summary_rows.append({
                "maximum_channel_radius_ratio": ratio,
                "radius_bin": radius_bin,
                "vessel_count": len(group),
                "minimum_radius_mm": min(row["local_radius_mm"] for row in group),
                "median_radius_mm": float(np.median([
                    row["local_radius_mm"] for row in group
                ])),
                "maximum_radius_mm": max(row["local_radius_mm"] for row in group),
                "median_total_thickness_mm": float(np.median([
                    row["total_prism_thickness_mm"] for row in group
                ])),
                "maximum_total_thickness_over_radius": max(
                    row["total_thickness_over_radius"] for row in group
                ),
                "median_first_layer_height_mm": float(np.median([
                    row["first_layer_height_mm"] for row in group
                ])),
            })
    summary_csv = audit_dir / "radius_bin_thickness_summary.csv"
    write_table(summary_csv, summary_rows)
    convergence_path = (
        out / "plane_sensitivity" / "all_vessels" / "reports"
        / "all_vessel_convergence.csv"
    )
    wss_radius_diagnostic: dict[str, Any] = {}
    if convergence_path.is_file():
        with convergence_path.open(newline="", encoding="utf-8-sig") as handle:
            convergence_rows = list(csv.DictReader(handle))
        radius_by_plane = {
            row["plane_id"]: float(row["section_equivalent_radius_mm"])
            for row in sections
        }
        for metric in ("wss_mean_pa", "wss_max_pa"):
            pairs = [
                (
                    radius_by_plane[row["plane_id"]],
                    float(row["finest_pair_percent"]),
                )
                for row in convergence_rows
                if row.get("metric") == metric
                and row.get("plane_id") in radius_by_plane
                and row.get("finest_pair_percent") not in (None, "")
            ]
            if len(pairs) >= 3:
                radii = np.asarray([value[0] for value in pairs], dtype=float)
                errors = np.asarray([value[1] for value in pairs], dtype=float)
                wss_radius_diagnostic[metric] = {
                    "vessel_count": len(pairs),
                    "pearson_log_radius_vs_finest_pair_percent": float(
                        np.corrcoef(np.log(radii), errors)[0, 1]
                    ),
                    "median_finest_pair_percent": float(np.median(errors)),
                    "p95_finest_pair_percent": float(np.percentile(errors, 95.0)),
                }
    payload = {
        "status": "geometry_audit_complete_cfd_comparison_pending",
        "layers": layers,
        "growth_ratio": growth,
        "requested_total_thickness_mm": requested,
        "candidate_maximum_channel_radius_ratios": ratios,
        "baseline_ratio": float(bl["maximum_channel_radius_ratio"]),
        "test_level": str(bl.get("thickness_test_level", "l3")),
        "validated_vessel_sections": len(sections),
        "first_layer_fraction_of_total": first_fraction,
        "existing_global_wss_radius_diagnostic": wss_radius_diagnostic,
        "audit_csv": str(audit_csv),
        "summary_csv": str(summary_csv),
        "interpretation": (
            "A cap of 0.15 means the five prism layers occupy at most 15% "
            "of local radius; paired 0.05/0.10/0.15-r CFD cases are needed "
            "to determine whether that fraction biases distal WSS."
        ),
    }
    audit_json = audit_dir / "radius_thickness_audit.json"
    audit_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print("Boundary-layer radius audit: {}".format(audit_json), flush=True)
    return audit_json


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--stage", choices=("all", "poi", "plan", "mesh", "mesh-quality", "boundary-layer", "boundary-layer-thickness", "cfx", "extract", "analyse", "plane-sensitivity", "validate"), default="all")
    parser.add_argument("--cores", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    cfg, _ = load_manifest(args.config)
    validate_manifest(cfg, require_apps=args.stage == "validate")
    if args.cores is not None:
        if args.cores <= 0: raise ValueError("--cores must be positive")
        cfg["cfx"]["cores"] = args.cores
    if args.stage == "validate":
        contract = validate_cfx_seed_contract(cfg)
        print(
            "Study manifest, executables and CFX physics contract passed "
            "(inlet={:.9g} kg/s, physics={}).".format(
                contract["inlet_mass_flow_kg_s"], contract["physics_control_sha256"][:12]
            )
        )
        return 0
    if args.stage in ("all", "poi"):
        if not prepare_pois(cfg):
            print("APPROVAL REQUIRED: complete the 'approved' column in {}/poi/poi_candidates.csv".format(cfg["paths"]["output_dir"]), flush=True)
            return APPROVAL_REQUIRED_EXIT
        if args.stage == "poi": return 0
    if args.stage in ("all", "plan"):
        print("Wrote study plan: {}".format(write_plan(cfg)), flush=True)
        if args.stage == "plan": return 0
    if args.stage in ("all", "mesh"):
        print(
            "Running mesh-convergence matrix first with {} fixed prism layers."
            .format(cfg["boundary_layer"].get("mesh_convergence_layers", 5)),
            flush=True,
        )
        run_mesh_stage(cfg, args.resume, phase="main")
        if args.stage == "mesh": return 0
        if args.stage == "all" and not Path(cfg["cfx"]["pre_session_template"]).is_file():
            print(
                "PAUSED AFTER MESHING: record cfx_remesh_session_template.pre using "
                "one generated BL mesh, then rerun with -Resume.", flush=True,
            )
            return 0
    if args.stage == "boundary-layer-thickness":
        write_boundary_layer_thickness_audit(cfg)
        run_mesh_stage(cfg, args.resume, phase="boundary_layer_thickness")
        return 0
    if args.stage == "mesh-quality":
        print("Wrote Simpleware quality audit: {}".format(
            export_existing_simpleware_quality(cfg, args.resume)
        ), flush=True)
        return 0
    if args.stage in ("all", "cfx"):
        prepare_and_run_cfx(cfg, args.resume)
        if args.stage == "cfx": return 0
    if args.stage in ("all", "extract"):
        run_extract_stage(cfg, args.resume)
        if args.stage == "extract": return 0
    if args.stage in ("all", "analyse"):
        print("Wrote analysis: {}".format(analyse_long_metrics(cfg)), flush=True)
    if args.stage in ("all", "plane-sensitivity") and "plane_sensitivity" in cfg:
        if "plane_sensitivity" not in cfg:
            raise ValueError("The manifest has no plane_sensitivity section")
        try:
            from .plane_sensitivity import run_plane_sensitivity
        except ImportError:
            from plane_sensitivity import run_plane_sensitivity
        print("Wrote plane sensitivity analysis: {}".format(
            run_plane_sensitivity(cfg, resume=args.resume)
        ), flush=True)
    if args.stage == "boundary-layer":
        print(
            "Main convergence is no longer gated by this later 4/6/8-layer study.",
            flush=True,
        )
        run_mesh_stage(cfg, args.resume, phase="boundary_layer")
        prepare_and_run_cfx(cfg, args.resume)
        run_extract_stage(cfg, args.resume)
        selected = maybe_select_boundary_layer(cfg)
        if selected is None:
            raise RuntimeError("Boundary-layer cases did not produce complete metrics")
        write_plan(cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
