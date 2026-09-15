"""Reproducible staged benchmark for implicit coronary reconstruction methods.

Examples
--------
Run the synthetic screen::

    python -m coronary_sdf.benchmark --output benchmark_out --stage synthetic

Run both paired LADAF cases using the checked-in manifest::

    python -m coronary_sdf.benchmark --output benchmark_out --stage all \
        --manifest coronary_sdf/benchmark_cases.json
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
from typing import Any

import numpy as np
import pyvista as pv
from scipy.spatial import KDTree

from . import config as config_module
from .amira_lattice import read_amira_lattice
from .benchmark_metrics import (
    cross_section_area_metrics,
    restrict_samples,
    sample_segmentation,
    segmentation_surface_metrics,
)
from .branch_coverage import branch_coverage, segment_centreline_samples
from .capsules import CapsuleArrays
from .centerline_optimizer import smooth_centerlines_constrained_multiscale
from .centreline_reconnection import find_connected_components
from .conflict_audit import audit_capsule_conflicts_against_mask
from .geometry_constraints import find_capsule_conflicts
from .implicit_field import round_cone_values
from .mesh_validation import validate_mesh
from .parse_amira import parse_xml
from .pipeline import generate_sdf_surface, run_pipeline
from .smoothing import limit_centerline_curvature, smooth_segment_centerlines
from .sdf_field import build_adjacency
from .synthetic_cases import SyntheticCase, synthetic_suite


CANDIDATES: dict[str, dict[str, Any]] = {
    "legacy_dense_meshlib": {
        "SDF_FIELD_METHOD": "legacy",
        "SDF_MESH_METHOD": "meshlib",
        "IMPLICIT_PRIMITIVE_METHOD": "radial",
    },
    "graph_round_cone_dense_mc": {
        "SDF_FIELD_METHOD": "graph_implicit",
        "SDF_MESH_METHOD": "mc",
        "IMPLICIT_PRIMITIVE_METHOD": "round_cone",
    },
    "graph_round_cone_vtk_htg": {
        "SDF_FIELD_METHOD": "graph_implicit",
        "SDF_MESH_METHOD": "vtk_htg",
        "IMPLICIT_PRIMITIVE_METHOD": "round_cone",
    },
    # Completes the field x extractor matrix. The dense column is matched on
    # 'mc' for both fields so the only difference across a row is the field.
    "legacy_dense_mc": {
        "SDF_FIELD_METHOD": "legacy",
        "SDF_MESH_METHOD": "mc",
        "IMPLICIT_PRIMITIVE_METHOD": "radial",
    },
    "legacy_vtk_htg": {
        "SDF_FIELD_METHOD": "legacy",
        "SDF_MESH_METHOD": "vtk_htg",
        "IMPLICIT_PRIMITIVE_METHOD": "radial",
    },
}

PREPROCESSORS = ("none", "savgol", "bspline", "constrained_multiscale")
DENSE_SPACING_FACTORS = (2.0, 1.0, 0.5)
ADAPTIVE_CELLS = (6.0, 12.0, 18.0)
BLEND_FRACTIONS = (0.05, 0.10, 0.15, 0.20)
BLEND_SUPPORTS = (2.0, 3.0, 4.0)


def _json_default(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


@dataclass(frozen=True)
class RunLimits:
    timeout_seconds: float
    memory_mb: float


def _base_config(**overrides) -> config_module.SdfConfig:
    baseline = config_module.SdfConfig(
        DEBUG_VIS=False,
        DEBUG_VIS_BLOCK=False,
        WRITE_REGION_VTK=False,
        OUTPUT_VALIDATION_MODE="off",
        OUTPUT_VALIDATE_SELF_INTERSECTIONS=False,
        PRESERVE_INPUT_RADII=True,
        MESH_REPAIR=False,
        FLAT_CAP_OUTLETS=False,
        SDF_FLAT_TERMINAL_CAPS=False,
        TAUBIN_ITERS=0,
        MIN_STRAHLER_ORDER=0,
        PRUNE_SHORT_TERMINAL_NUBS=False,
        SMOOTH_DRIFT_VERBOSE=False,
        CURVATURE_VERBOSE=False,
        TERMINAL_CLAMP_VERBOSE=False,
        REPORT_RING_GAP_ATTRIBUTION=False,
        BIF_MERGE_VERBOSE=False,
    )
    return baseline.with_overrides(**overrides)


def candidate_config(
    name: str,
    *,
    preprocessor: str = "none",
    resolution: float | None = None,
    cells_across_diameter: float | None = None,
    blend_fraction: float = 0.05,
    blend_support: float = 2.0,
    validation_mode: str | None = None,
    component_failure: str = "continue",
) -> config_module.SdfConfig:
    if name not in CANDIDATES:
        raise KeyError(f"unknown candidate {name!r}")
    overrides = dict(CANDIDATES[name])
    overrides.update(
        CENTERLINE_SMOOTHER=preprocessor,
        LIMIT_CENTERLINE_CURVATURE=preprocessor in {"savgol", "bspline"},
        CENTERLINE_CONSTRAINT_FAILURE="error",
        BSPLINE_SDF_RESOLUTION=resolution,
        IMPLICIT_JUNCTION_BLEND_FRACTION=blend_fraction,
        IMPLICIT_JUNCTION_SUPPORT_FACTOR=blend_support,
        IMPLICIT_FAIL_ON_GEOMETRY_CONFLICT=False,
        # A failing component must be recorded, not allowed to erase the run.
        PIPELINE_COMPONENT_FAILURE=component_failure,
    )
    if name in {"graph_round_cone_dense_mc", "legacy_dense_mc"}:
        overrides["DENSE_MIN_SPACING_MM"] = None
    if name == "graph_round_cone_vtk_htg":
        overrides.update(
            OUTPUT_VALIDATION_MODE="error",
            OUTPUT_VALIDATE_SELF_INTERSECTIONS=True,
        )
    if validation_mode is not None:
        overrides["OUTPUT_VALIDATION_MODE"] = validation_mode
    if cells_across_diameter is not None:
        overrides["IMPLICIT_CELLS_ACROSS_DIAMETER"] = cells_across_diameter
    return _base_config(**overrides)


def _capsules_from_graph(points: dict[int, tuple], segments: list[dict]) -> CapsuleArrays:
    starts = []
    ends = []
    r0 = []
    r1 = []
    segment_ids = []
    for segment_index, segment in enumerate(segments):
        ids = [pid for pid in segment.get("point_ids", []) if pid in points]
        for a, b in zip(ids[:-1], ids[1:]):
            starts.append(np.asarray(points[a][:3], float) / 1000.0)
            ends.append(np.asarray(points[b][:3], float) / 1000.0)
            r0.append(float(points[a][3]) / 1000.0)
            r1.append(float(points[b][3]) / 1000.0)
            segment_ids.append(segment_index)
    starts_a = np.asarray(starts, dtype=np.float64)
    ends_a = np.asarray(ends, dtype=np.float64)
    r0_a = np.asarray(r0, dtype=np.float64)
    r1_a = np.asarray(r1, dtype=np.float64)
    segment_a = np.asarray(segment_ids, dtype=np.int64)
    displacement = ends_a - starts_a
    lengths = np.linalg.norm(displacement, axis=1)
    tangents = displacement / np.maximum(lengths[:, None], 1e-30)
    arc_start = np.zeros(len(starts_a))
    arc_end = np.zeros(len(starts_a))
    segment_length = np.zeros(len(segments))
    for segment_index in range(len(segments)):
        selection = np.flatnonzero(segment_a == segment_index)
        cumulative = np.r_[0.0, np.cumsum(lengths[selection])]
        arc_start[selection] = cumulative[:-1]
        arc_end[selection] = cumulative[1:]
        segment_length[segment_index] = cumulative[-1]
    return CapsuleArrays(
        starts=starts_a,
        ends=ends_a,
        radii_start=r0_a,
        radii_end=r1_a,
        seg_idx=segment_a,
        midpoints=0.5 * (starts_a + ends_a),
        tangents=tangents,
        max_radii=np.maximum(r0_a, r1_a),
        tree=KDTree(0.5 * (starts_a + ends_a)),
        arc_start=arc_start,
        arc_end=arc_end,
        seg_L=segment_length,
        cap_bif_at_start=np.zeros(len(starts_a), dtype=bool),
        cap_bif_at_end=np.zeros(len(starts_a), dtype=bool),
    )


def _analytic_error(surface, case: SyntheticCase) -> dict[str, float]:
    capsules = _capsules_from_graph(case.points, case.segments)
    vertices = np.asarray(surface.points)
    if len(vertices) > 250_000:
        vertices = vertices[np.linspace(0, len(vertices) - 1, 250_000, dtype=np.int64)]
    residual = np.empty(len(vertices), dtype=np.float64)
    for index, point in enumerate(vertices):
        residual[index] = np.min(
            round_cone_values(
                point,
                capsules.starts,
                capsules.ends,
                capsules.radii_start,
                capsules.radii_end,
            )[0]
        )
    residual = np.abs(residual)
    characteristic_radius = float(np.median(capsules.max_radii))
    return {
        "surface_residual_median_mm": float(np.median(residual)),
        "surface_residual_hd95_mm": float(np.percentile(residual, 95)),
        "surface_residual_hd95_radius": float(
            np.percentile(residual, 95) / characteristic_radius
        ),
    }


def run_synthetic(spec: dict[str, Any]) -> dict[str, Any]:
    cases = {case.name: case for case in synthetic_suite()}
    case = cases[spec["case"]]
    cfg = candidate_config(
        spec["candidate"],
        preprocessor=spec.get("preprocessor", "none"),
        resolution=spec.get("resolution"),
        cells_across_diameter=spec.get("cells_across_diameter"),
        blend_fraction=spec.get("blend_fraction", 0.05),
        blend_support=spec.get("blend_support", 2.0),
    )
    # Synthetic fixtures can intentionally contain more than one disconnected
    # lumen in a single in-memory graph (for example the positive-clearance
    # pair). The pipeline's per-connected-component production validator
    # expects one component, so collect the no-write mesh here and apply the
    # fixture's exact topology contract below instead.
    cfg = cfg.with_overrides(OUTPUT_VALIDATION_MODE="off")
    started = time.perf_counter()
    surface = generate_sdf_surface(
        case.nodes,
        case.points,
        case.segments,
        spec["scratch"],
        cfg=cfg,
        write_outputs=False,
        interactive=False,
    )
    if surface is None:
        raise RuntimeError("pipeline returned no surface")
    validation = validate_mesh(
        surface,
        expected_components=case.expected_components,
        check_self_intersections=case.expected_valid,
    )
    branch_loss = 0
    for segment in case.segments:
        ids = segment["point_ids"][1:-1] or segment["point_ids"]
        centreline = np.asarray([case.points[pid][:3] for pid in ids], float) / 1000.0
        selected = pv.PolyData(centreline).select_enclosed_points(
            surface,
            tolerance=1e-7,
            check_surface=False,
        )
        inside = np.asarray(selected["SelectedPoints"], bool)
        if not len(inside) or float(np.mean(inside)) < 0.95:
            branch_loss += 1
    topology_ok = bool(
        validation.valid
        and validation.genus is not None
        and abs(validation.genus) < 1e-12
        and branch_loss == 0
    )
    clearance_metrics: dict[str, float | None] = {}
    if case.category == "clearance" and len(case.segments) >= 2:
        first, second = case.segments[:2]
        p0 = np.asarray(case.points[first["point_ids"][0]][:3], float) / 1000.0
        q0 = np.asarray(case.points[second["point_ids"][0]][:3], float) / 1000.0
        r0 = float(case.points[first["point_ids"][0]][3]) / 1000.0
        r1 = float(case.points[second["point_ids"][0]][3]) / 1000.0
        expected_clearance = float(np.linalg.norm(p0 - q0) - r0 - r1)
        bodies = surface.split_bodies(label=False)
        reconstructed_clearance = None
        if len(bodies) == 2:
            reconstructed_clearance = float(
                KDTree(np.asarray(bodies[0].points)).query(
                    np.asarray(bodies[1].points), workers=-1
                )[0].min()
            )
        clearance_metrics = {
            "expected_clearance_mm": expected_clearance,
            "reconstructed_clearance_mm": reconstructed_clearance,
            "clearance_error_mm": (
                None
                if reconstructed_clearance is None
                else abs(reconstructed_clearance - expected_clearance)
            ),
        }
    result = {
        "kind": "synthetic",
        "case": case.name,
        "category": case.category,
        "expected_valid": case.expected_valid,
        "candidate": spec["candidate"],
        "preprocessor": spec.get("preprocessor", "none"),
        "elapsed_seconds": time.perf_counter() - started,
        "mesh_points": int(surface.n_points),
        "mesh_faces": int(surface.n_cells),
        "validation": validation.to_dict(),
        "branch_loss_count": branch_loss,
        "topology_ok": topology_ok,
        "telemetry": _surface_telemetry([surface]),
        "clearance_metrics": clearance_metrics,
    }
    result.update(_analytic_error(surface, case))
    result["qualified"] = bool(not case.expected_valid or topology_ok)
    if case.expected_valid and not result["qualified"] and spec.get("diagnostic_dir"):
        diagnostic = Path(spec["diagnostic_dir"])
        diagnostic.mkdir(parents=True, exist_ok=True)
        surface.save(str(diagnostic / f"{spec['candidate']}__{case.name}.vtp"))
    return result


def _surface_telemetry(surfaces) -> dict[str, float | int]:
    totals: dict[str, float] = {}
    for surface in surfaces:
        for key in (
            "field_evaluations",
            "hierarchy_seconds",
            "field_seconds",
            "extraction_seconds",
            "total_seconds",
        ):
            if key in surface.field_data:
                totals[key] = totals.get(key, 0.0) + float(
                    np.asarray(surface.field_data[key]).ravel()[0]
                )
    if "field_evaluations" in totals:
        totals["field_evaluations"] = int(totals["field_evaluations"])
    return totals


def _curvature_violations(points: dict[int, tuple], segments: list[dict]) -> int:
    total = 0
    for segment in segments:
        ids = [pid for pid in segment.get("point_ids", []) if pid in points]
        if len(ids) < 3:
            continue
        xyz = np.asarray([points[pid][:3] for pid in ids], float) / 1000.0
        radii = np.asarray([points[pid][3] for pid in ids], float) / 1000.0
        a = xyz[1:-1] - xyz[:-2]
        b = xyz[2:] - xyz[1:-1]
        c = xyz[2:] - xyz[:-2]
        denominator = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) * np.linalg.norm(c, axis=1)
        curvature = np.where(
            denominator > 1e-30,
            2.0 * np.linalg.norm(np.cross(a, c), axis=1) / denominator,
            0.0,
        )
        total += int(np.count_nonzero(curvature * radii[1:-1] >= 1.0))
    return total


def run_preprocessor(spec: dict[str, Any]) -> dict[str, Any]:
    case = {case.name: case for case in synthetic_suite()}[spec["case"]]
    mode = spec["preprocessor"]
    cfg = _base_config(
        CENTERLINE_SMOOTHER=mode,
        LIMIT_CENTERLINE_CURVATURE=mode in {"savgol", "bspline"},
        CENTERLINE_CONSTRAINT_FAILURE="report",
    )
    before = dict(case.points)
    output = dict(case.points)
    report = None
    started = time.perf_counter()
    with config_module.use_config(cfg):
        if mode == "constrained_multiscale":
            output, smoothing = smooth_centerlines_constrained_multiscale(
                case.nodes, output, case.segments
            )
            report = smoothing.to_dict()
        elif mode != "none":
            output, _ = smooth_segment_centerlines(case.nodes, output, case.segments)
            if cfg.LIMIT_CENTERLINE_CURVATURE:
                output, _ = limit_centerline_curvature(case.nodes, output, case.segments)
    ids = sorted(set(before) & set(output))
    raw = np.asarray([before[pid][:3] for pid in ids], float) / 1000.0
    fitted = np.asarray([output[pid][:3] for pid in ids], float) / 1000.0
    radii = np.asarray([before[pid][3] for pid in ids], float) / 1000.0
    displacement = np.linalg.norm(fitted - raw, axis=1) / np.maximum(radii, 1e-30)
    before_conflicts = find_capsule_conflicts(_capsules_from_graph(before, case.segments))
    after_conflicts = find_capsule_conflicts(_capsules_from_graph(output, case.segments))
    before_pairs = {(item.segment_a, item.segment_b) for item in before_conflicts}
    after_pairs = {(item.segment_a, item.segment_b) for item in after_conflicts}
    endpoints_preserved = all(
        output[segment["point_ids"][end]][:3] == before[segment["point_ids"][end]][:3]
        for segment in case.segments
        for end in (0, -1)
    )
    radii_preserved = all(output[pid][3] == before[pid][3] for pid in ids)
    unresolved = int(report["unresolved_constraints"]) if report else 0
    result = {
        "kind": "preprocessor",
        "case": case.name,
        "category": case.category,
        "expected_valid": case.expected_valid,
        "preprocessor": mode,
        "elapsed_seconds": time.perf_counter() - started,
        "max_drift_radius": float(displacement.max(initial=0.0)),
        "p95_drift_radius": float(np.percentile(displacement, 95)),
        "endpoints_preserved": endpoints_preserved,
        "radii_preserved": radii_preserved,
        "new_conflict_pairs": len(after_pairs - before_pairs),
        "curvature_violations_before": _curvature_violations(before, case.segments),
        "curvature_violations_after": _curvature_violations(output, case.segments),
        "unresolved_constraints": unresolved,
        "optimizer_report": report,
    }
    result["qualified"] = bool(
        endpoints_preserved
        and radii_preserved
        and result["max_drift_radius"] <= 0.25 + 1e-9
        and result["new_conflict_pairs"] == 0
        and unresolved == 0
    )
    return result


def _preflight_dense(graph_path: str, spacing: float, maximum_voxels: int) -> int:
    _nodes, points, _segments = parse_xml(graph_path)
    coords = np.asarray([value[:3] for value in points.values()], float) / 1000.0
    radii = np.asarray([value[3] for value in points.values()], float) / 1000.0
    padding = float(np.max(radii)) + 0.01
    extent = np.ptp(coords, axis=0) + 2.0 * padding
    voxels = int(np.prod(np.ceil(extent / spacing).astype(np.int64) + 1))
    if voxels > maximum_voxels:
        raise MemoryError(
            f"requested dense grid requires {voxels:,} voxels, exceeding {maximum_voxels:,}"
        )
    return voxels


#: Sites the mask confirms are genuine vessel contact do not disqualify an
#: input; every other classification does. Same predicate the all-or-nothing
#: gate used, now reported instead of short-circuiting the run.
QUALIFYING_CLASSIFICATION = "mask_confirmed_contact"

#: Cap on the compact per-site extract carried in a run record. The full audit
#: is still recorded verbatim under ``conflict_audit``.
MAX_REPORTED_SITES = 64


def _characteristic_length(manifest) -> float | None:
    """Largest dense voxel size across components, or ``None`` when adaptive.

    Vertex-sampled coverage distances are biased by the extractor's triangle
    size. Recording that length is what allows a dense and an adaptive run to be
    compared at all; without it a coarse dense mesh looks like a coverage
    failure rather than a sampling artefact.
    """

    values = [
        component.voxel_size_mm
        for component in getattr(manifest, "components", ())
        if component.voxel_size_mm
    ]
    return float(max(values)) if values else None


def _summarise_conflict_audit(
    conflict_audit: list[dict[str, Any]],
    segments: list[dict[str, Any]],
    *,
    conflict_count: int | None = None,
) -> dict[str, Any]:
    """Partition a mask-backed audit into a qualification verdict plus evidence.

    The verdict is still all-or-nothing — one unconfirmed site disqualifies the
    input for CFD — but it is now a label on the run rather than a reason to
    skip meshing, and it carries enough locality (segment index *and* source
    Segment ID, plus the graph component) to act on.
    """

    classification_counts: dict[str, int] = {}
    invalid_counts: dict[str, int] = {}
    invalid_segment_indices: set[int] = set()
    sites: list[dict[str, Any]] = []

    # ``segment_a``/``segment_b`` index the segment *list* (they come from
    # ``enumerate(segments)`` in ``_capsules_from_graph``), which is a different
    # number from the source ``Segment ID``. Carry both or attribution is wrong.
    component_of: dict[int, int] = {}
    if segments:
        for component_index, member_indices in enumerate(
            find_connected_components(segments)
        ):
            for segment_index in member_indices:
                component_of[int(segment_index)] = component_index

    def _segment_id(index: Any) -> int | None:
        if index is None:
            return None
        position = int(index)
        if 0 <= position < len(segments):
            return int(segments[position].get("id", position))
        return None

    for record in conflict_audit:
        classification = str(record.get("classification", "unknown"))
        classification_counts[classification] = (
            classification_counts.get(classification, 0) + 1
        )
        if classification == QUALIFYING_CLASSIFICATION:
            continue
        invalid_counts[classification] = invalid_counts.get(classification, 0) + 1
        for key in ("segment_a", "segment_b"):
            value = record.get(key)
            if value is not None:
                invalid_segment_indices.add(int(value))
        if len(sites) < MAX_REPORTED_SITES:
            sites.append(
                {
                    "conflict_index": record.get("conflict_index"),
                    "classification": classification,
                    "segment_a_index": record.get("segment_a"),
                    "segment_b_index": record.get("segment_b"),
                    "segment_a_id": _segment_id(record.get("segment_a")),
                    "segment_b_id": _segment_id(record.get("segment_b")),
                    "point_a_mm": record.get("point_a_mm"),
                    "point_b_mm": record.get("point_b_mm"),
                    "tangent_cosine": record.get("tangent_cosine"),
                    "line_background_gap_mm": record.get("line_background_gap_mm"),
                }
            )

    invalid_total = sum(invalid_counts.values())
    ordered_indices = sorted(invalid_segment_indices)
    return {
        "conflict_count": int(
            conflict_count if conflict_count is not None else len(conflict_audit)
        ),
        "classification_counts": classification_counts,
        "invalid_count": invalid_total,
        "invalid_classification_counts": invalid_counts,
        "invalid_segment_indices": ordered_indices,
        "invalid_segment_ids": [
            value
            for value in (_segment_id(index) for index in ordered_indices)
            if value is not None
        ],
        "invalid_graph_components": sorted(
            {
                component_of[index]
                for index in ordered_indices
                if index in component_of
            }
        ),
        "invalid_sites": sites,
        "invalid_sites_truncated": invalid_total > len(sites),
        "input_qualified": invalid_total == 0,
        "input_failure": (
            None
            if invalid_total == 0
            else "Labels mask contradicts or cannot resolve fixed-radius graph overlap"
        ),
    }


def run_real(spec: dict[str, Any]) -> dict[str, Any]:
    case = spec["real_case"]
    lattice = read_amira_lattice(
        case["segmentation"], "Labels", cache_dir=spec["cache_dir"]
    )
    voxel = float(np.max(lattice.header.spacing_mm))
    resolution_factor = spec.get("resolution_factor")
    cells = spec.get("cells_across_diameter")
    resolution = voxel * resolution_factor if resolution_factor is not None else None
    cfg = candidate_config(
        spec["candidate"],
        preprocessor=spec.get("preprocessor", "none"),
        resolution=resolution,
        cells_across_diameter=cells,
        blend_fraction=spec.get("blend_fraction", 0.05),
        blend_support=spec.get("blend_support", 2.0),
        validation_mode=spec.get("validation_mode"),
    )
    audit_cache = Path(spec["cache_dir"]) / f"{case['name']}.conflict-audit.json"
    graph_stat = Path(case["graph"]).stat()
    mask_stat = Path(case["segmentation"]).stat()
    audit_fingerprint = {
        "graph": str(Path(case["graph"]).resolve()),
        "graph_size": graph_stat.st_size,
        "graph_mtime_ns": graph_stat.st_mtime_ns,
        "segmentation": str(Path(case["segmentation"]).resolve()),
        "segmentation_size": mask_stat.st_size,
        "segmentation_mtime_ns": mask_stat.st_mtime_ns,
        "clearance_fraction": cfg.IMPLICIT_GEOMETRY_CLEARANCE_FRACTION,
    }
    # Parse once and reuse for the audit and the metrics, so the audit's
    # segment indices and the metrics' segment indices are the same by
    # construction rather than by assumption.
    nodes_raw, points_raw, segments_raw = parse_xml(case["graph"])
    cached_audit = None
    if audit_cache.exists():
        candidate_cache = json.loads(audit_cache.read_text(encoding="utf-8"))
        if candidate_cache.get("fingerprint") == audit_fingerprint:
            cached_audit = candidate_cache
    if cached_audit is None:
        node_to_segments: dict[int, set[int]] = {
            int(node): set() for node in nodes_raw
        }
        for segment_index, segment in enumerate(segments_raw):
            node_to_segments.setdefault(int(segment["node1"]), set()).add(
                segment_index
            )
            node_to_segments.setdefault(int(segment["node2"]), set()).add(
                segment_index
            )
        raw_capsules = _capsules_from_graph(points_raw, segments_raw)
        with config_module.use_config(cfg):
            adjacency, shared_positions, shared_radii = build_adjacency(
                nodes_raw, points_raw, segments_raw, node_to_segments
            )
        conflicts = find_capsule_conflicts(
            raw_capsules,
            adjacency,
            shared_node_positions=shared_positions,
            shared_node_radii=shared_radii,
            clearance_fraction=cfg.IMPLICIT_GEOMETRY_CLEARANCE_FRACTION,
            maximum_records=None,
        )
        reports = audit_capsule_conflicts_against_mask(
            raw_capsules,
            conflicts,
            lattice,
            diagnostic_dir=(
                Path(spec["diagnostic_dir"]) / "conflict_rois" / case["name"]
                if conflicts and spec.get("diagnostic_dir")
                else None
            ),
        )
        conflict_audit = [report.to_dict() for report in reports]
        cached_audit = {
            "fingerprint": audit_fingerprint,
            "conflict_count": len(conflicts),
            "audits": conflict_audit,
        }
        audit_cache.write_text(
            json.dumps(cached_audit, indent=2, default=_json_default),
            encoding="utf-8",
        )
    else:
        conflict_audit = list(cached_audit.get("audits", []))

    # The audit still decides CFD qualification, but it no longer decides
    # whether the reconstruction runs. A single unconfirmed site out of
    # thousands used to suppress every fidelity metric for the whole case,
    # which is why no real surface evidence exists yet.
    conflict_summary = _summarise_conflict_audit(
        conflict_audit,
        segments_raw,
        conflict_count=cached_audit.get("conflict_count"),
    )
    input_qualified = bool(conflict_summary["input_qualified"])
    purpose = str(spec.get("purpose", "qualification"))
    roi = spec.get("roi")

    def _record(**fields: Any) -> dict[str, Any]:
        """Assemble a real-run record with the qualification fields fixed."""

        base = {
            "kind": "real",
            "case": case["name"],
            "candidate": spec["candidate"],
            "preprocessor": spec.get("preprocessor", "none"),
            "purpose": purpose,
            "roi": roi,
            "resolution_factor": resolution_factor,
            "cells_across_diameter": cells,
            "input_qualified": input_qualified,
            "input_failure": conflict_summary["input_failure"],
            "conflict_count": conflict_summary["conflict_count"],
            "conflict_summary": conflict_summary,
            "conflict_audit": conflict_audit,
        }
        base.update(fields)
        return base

    if cfg.SDF_MESH_METHOD not in {"adaptive", "vtk_htg", "cgal_mesh3"} and resolution is not None:
        dense_voxels = _preflight_dense(
            case["graph"], resolution, cfg.BSPLINE_SDF_MAX_VOXELS
        )
    else:
        dense_voxels = 0
    started = time.perf_counter()
    try:
        surfaces, manifest = run_pipeline(
            case["graph"],
            spec["scratch"],
            cfg=cfg,
            write_outputs=False,
            interactive=False,
            return_report=True,
        )
    except Exception as exc:
        # The mask audit is part of the scientific result and must survive a
        # later native dependency, timeout-adjacent, or meshing failure.
        return _record(
            dense_voxels=dense_voxels,
            status="error",
            error_type=type(exc).__name__,
            error=str(exc),
            elapsed_seconds=time.perf_counter() - started,
            qualified=False,
            qualification_blockers=["reconstruction_error"],
        )
    surface_list = list(surfaces.values())
    validations = [
        validate_mesh(surface, expected_components=1, check_self_intersections=True).to_dict()
        for surface in surface_list
    ]
    if not all(item["valid"] for item in validations) and spec.get("diagnostic_dir"):
        diagnostic = Path(spec["diagnostic_dir"])
        diagnostic.mkdir(parents=True, exist_ok=True)
        # Keep the real graph id; ``enumerate`` mislabels components once one
        # of them drops out.
        for gid, surface in surfaces.items():
            surface.save(
                str(diagnostic / f"{spec['candidate']}__{case['name']}__g{gid}.vtp")
            )
    # An ROI surface compared against the whole mask scores catastrophically for
    # reasons that have nothing to do with the reconstruction, so restrict the
    # reference to the region and then to the vessels this graph claims.
    bounds = case.get("evaluation_box_mm")
    if bounds is not None:
        bounds = (np.asarray(bounds[0], float), np.asarray(bounds[1], float))
    if bounds is not None:
        samples = sample_segmentation(lattice, bounds_mm=bounds)
        centreline, radii, _seg, _pid = segment_centreline_samples(
            points_raw, segments_raw
        )
        samples = restrict_samples(samples, centreline, radii)
        surface_metrics = segmentation_surface_metrics(
            surface_list,
            lattice,
            samples=samples,
            bounds_mm=bounds,
            metric_scope="roi_graph_restricted",
        )
    else:
        surface_metrics = segmentation_surface_metrics(surface_list, lattice)
    area_metrics = cross_section_area_metrics(
        surface_list, lattice, points_raw, segments_raw
    )
    coverage = branch_coverage(
        surface_list,
        points_raw,
        segments_raw,
        containment=bool(spec.get("coverage_containment", False)),
        extractor_characteristic_length_mm=_characteristic_length(manifest),
    )
    topology_ok = bool(validations and all(item["valid"] for item in validations))
    gate = {
        "input_qualified": input_qualified,
        "components_complete": not manifest.incomplete,
        "topology_ok": topology_ok,
        "hd95_ok": bool(surface_metrics["hd95_mm"] <= 2.0 * voxel),
        "area_ok": bool(area_metrics["median_area_error_fraction"] <= 0.10),
        # Recorded but deliberately not yet folded into ``qualified``: changing
        # the gate and the experiment in one step is what made the previous
        # screen uninterpretable.
        "coverage_ok": bool(coverage.complete),
    }
    blockers = [name for name, passed in gate.items() if not passed]
    if purpose != "qualification":
        blockers.append("diagnostic_purpose")
    if roi:
        blockers.append("roi_scope")
    scoring_gate = {k: v for k, v in gate.items() if k != "coverage_ok"}
    return _record(
        dense_voxels=dense_voxels,
        status="ok",
        elapsed_seconds=time.perf_counter() - started,
        mesh_points=int(sum(surface.n_points for surface in surface_list)),
        mesh_faces=int(sum(surface.n_cells for surface in surface_list)),
        validations=validations,
        telemetry=_surface_telemetry(surface_list),
        component_manifest=manifest.to_dict(),
        coverage_metrics=coverage.summary(),
        topology_ok=topology_ok,
        surface_metrics=surface_metrics,
        area_metrics=area_metrics,
        cfd_gate=gate,
        qualification_blockers=blockers,
        qualified=bool(
            purpose == "qualification" and not roi and all(scoring_gate.values())
        ),
    )


def execute_worker(spec: dict[str, Any]) -> dict[str, Any]:
    kind = spec["kind"]
    if kind == "synthetic":
        return run_synthetic(spec)
    if kind == "preprocessor":
        return run_preprocessor(spec)
    if kind == "real":
        return run_real(spec)
    raise ValueError(f"unknown worker kind {kind!r}")


def _monitor_worker(
    spec: dict[str, Any], result_path: Path, limits: RunLimits
) -> dict[str, Any]:
    # Never let a killed retry inherit a stale success/error payload from a
    # previous attempt with the same deterministic run id.
    if result_path.exists():
        result_path.unlink()
    spec_path = result_path.with_suffix(".spec.json")
    spec_path.write_text(json.dumps(spec, indent=2, default=_json_default), encoding="utf-8")
    command = [
        sys.executable,
        "-m",
        "coronary_sdf.benchmark",
        "--worker-spec",
        str(spec_path),
        "--worker-result",
        str(result_path),
    ]
    process = subprocess.Popen(command, cwd=str(Path(__file__).resolve().parent.parent))
    started = time.monotonic()
    peak = 0
    status = "running"
    try:
        import psutil

        monitored = psutil.Process(process.pid)
        while process.poll() is None:
            rss = 0
            try:
                rss += monitored.memory_info().rss
                for child in monitored.children(recursive=True):
                    rss += child.memory_info().rss
            except psutil.Error:
                pass
            peak = max(peak, rss)
            if rss > limits.memory_mb * 1024**2:
                status = "memory_limit"
                process.kill()
                break
            if time.monotonic() - started > limits.timeout_seconds:
                status = "timeout"
                process.kill()
                break
            time.sleep(0.1)
        process.wait(timeout=10)
    except Exception:
        if process.poll() is None:
            process.kill()
        raise
    if status == "running":
        status = "ok" if process.returncode == 0 else "error"
    if result_path.exists():
        result = json.loads(result_path.read_text(encoding="utf-8"))
    else:
        result = {
            **spec,
            "status": status,
            "error": f"worker exited {process.returncode}",
        }
    if status in {"timeout", "memory_limit"}:
        result["status"] = status
    else:
        result.setdefault("status", status)
    result["peak_rss_mb"] = peak / 1024**2
    result["wall_seconds"] = time.monotonic() - started
    return result


def _worker_main(spec_path: Path, result_path: Path) -> int:
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    try:
        result = execute_worker(spec)
        result.setdefault("status", "ok")
    except MemoryError as exc:
        result = {"status": "resource_skip", "error": str(exc), **spec}
    except Exception as exc:
        result = {
            "status": "error",
            "error": str(exc),
            "traceback": traceback.format_exc(),
            **spec,
        }
    result_path.write_text(
        json.dumps(result, indent=2, default=_json_default), encoding="utf-8"
    )
    return 0 if result["status"] in {"ok", "resource_skip"} else 1


def _flatten(value: Any, prefix: str = "") -> dict[str, Any]:
    if isinstance(value, dict):
        result = {}
        for key, child in value.items():
            result.update(_flatten(child, f"{prefix}.{key}" if prefix else key))
        return result
    if isinstance(value, list):
        return {prefix: json.dumps(value, sort_keys=True, default=_json_default)}
    return {prefix: value}


def _write_outputs(records: list[dict[str, Any]], output: Path) -> None:
    jsonl = output / "runs.jsonl"
    jsonl.write_text(
        "".join(
            json.dumps(record, sort_keys=True, default=_json_default) + "\n"
            for record in records
        ),
        encoding="utf-8",
    )
    rows = [_flatten(record) for record in records]
    columns = sorted({key for row in rows for key in row})
    with (output / "summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def _load_completed_records(output: Path) -> list[dict[str, Any]]:
    completed = []
    runs = output / "runs"
    if not runs.exists():
        return completed
    for path in sorted(runs.glob("*.json")):
        if path.name.endswith(".spec.json"):
            continue
        try:
            completed.append(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError):
            continue
    return completed


def _is_qualification_record(record: dict[str, Any]) -> bool:
    """True only for full-graph runs executed for qualification.

    Filters on two independent keys so a record missing one of them cannot leak
    into winner selection.
    """

    return (
        record.get("purpose", "qualification") == "qualification"
        and not record.get("roi")
    )


def _select_winner(records: list[dict[str, Any]], candidates: list[str]):
    all_real = [record for record in records if record.get("kind") == "real"]
    # A diagnostic or ROI record claiming qualification means the record set is
    # inconsistent; fail loudly rather than silently promoting a partial sweep.
    for record in all_real:
        if record.get("qualified") and not _is_qualification_record(record):
            return None, (
                "inconsistent records: "
                f"{record.get('candidate')}/{record.get('case')} is marked qualified "
                f"but has purpose={record.get('purpose')!r} roi={record.get('roi')!r}"
            )
    real = [record for record in all_real if _is_qualification_record(record)]
    synthetic = [record for record in records if record.get("kind") == "synthetic"]
    if not real:
        if all_real:
            return None, (
                f"{len(all_real)} real run(s) were diagnostic-only; "
                "no qualification run was executed"
            )
        return None, "real-data stage was not run; results are screening-only"
    scored = []
    required_synthetic = {
        case.name for case in synthetic_suite() if case.expected_valid
    }
    required_real_cases = {"LADAF_2024_28", "LADAF_2024_56"}
    for candidate in candidates:
        candidate_synthetic = [
            record for record in synthetic
            if record.get("candidate") == candidate
            and record.get("case") in required_synthetic
        ]
        candidate_real = [record for record in real if record.get("candidate") == candidate]
        if (
            {record.get("case") for record in candidate_synthetic}
            != required_synthetic
            or not all(
            record.get("status") == "ok" and record.get("qualified")
            for record in candidate_synthetic
            )
        ):
            continue
        if {record.get("case") for record in candidate_real} != required_real_cases:
            continue
        mesh_method = CANDIDATES[candidate]["SDF_MESH_METHOD"]
        if mesh_method == "vtk_htg":
            observed_resolution = {
                (record.get("case"), float(record.get("cells_across_diameter")))
                for record in candidate_real
                if record.get("cells_across_diameter") is not None
            }
            required_resolution = {
                (case, cells)
                for case in required_real_cases
                for cells in ADAPTIVE_CELLS
            }
        else:
            observed_resolution = {
                (record.get("case"), float(record.get("resolution_factor")))
                for record in candidate_real
                if record.get("resolution_factor") is not None
            }
            required_resolution = {
                (case, factor)
                for case in required_real_cases
                for factor in DENSE_SPACING_FACTORS
            }
        if observed_resolution != required_resolution:
            continue
        # Redundant with ``qualified`` by construction, but stated explicitly so
        # the promotion criteria are auditable from this function alone.
        if not all(
            record.get("status") == "ok"
            and record.get("qualified")
            and record.get("input_qualified") is True
            and not record.get("component_manifest", {}).get("incomplete", False)
            for record in candidate_real
        ):
            continue
        worst = max(
            max(
                record["surface_metrics"]["hd95_mm"]
                / (2.0 * record["surface_metrics"]["reference_voxel_mm"]),
                record["area_metrics"]["median_area_error_fraction"] / 0.10,
            )
            for record in candidate_real
        )
        peak = max(record.get("peak_rss_mb", math.inf) for record in candidate_real)
        runtime = sum(record.get("wall_seconds", math.inf) for record in candidate_real)
        scored.append((candidate, worst, peak, runtime))
    if not scored:
        return None, "no implicit candidate met every CFD qualification gate"
    scored.sort(key=lambda item: item[1])
    best_error = scored[0][1]
    near = [item for item in scored if item[1] <= 1.05 * best_error]
    near.sort(key=lambda item: (item[2], item[3]))
    return near[0][0], "qualified by topology and acquisition-aware geometry gates"


def _select_preprocessor(records: list[dict[str, Any]]):
    screened = [record for record in records if record.get("kind") == "preprocessor"]
    if not screened:
        return None, "preprocessing screen was not run"
    case_names = {record.get("case") for record in screened}
    qualified = []
    for mode in PREPROCESSORS:
        selected = [record for record in screened if record.get("preprocessor") == mode]
        if {record.get("case") for record in selected} != case_names:
            continue
        if not all(
            record.get("status") == "ok" and record.get("qualified")
            for record in selected
        ):
            continue
        qualified.append(
            (
                max(float(record.get("max_drift_radius", math.inf)) for record in selected),
                sum(float(record.get("wall_seconds", math.inf)) for record in selected),
                mode,
            )
        )
    if not qualified:
        return None, "no preprocessing mode met every preservation/constraint gate"
    drift, _runtime, mode = min(qualified)
    return mode, f"lowest worst-case normalized drift ({drift:.6g} local radii)"


def _write_report(
    records: list[dict[str, Any]], output: Path, candidates: list[str]
) -> None:
    required_real_cases = {"LADAF_2024_28", "LADAF_2024_56"}
    winner, reason = _select_winner(records, candidates)
    preprocessor, preprocessing_reason = _select_preprocessor(records)
    lines = [
        "# Coronary implicit reconstruction benchmark",
        "",
        f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        f"**Winner:** `{winner}`" if winner else "**Winner:** none",
        "",
        reason,
        "",
        (
            f"**Selected preprocessing:** `{preprocessor}` — {preprocessing_reason}"
            if preprocessor
            else f"**Selected preprocessing:** none — {preprocessing_reason}"
        ),
        "",
        "## Candidate summary",
        "",
        "| Candidate | Runs | Passed | Errors/skips | Peak RSS MB | Wall seconds |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for candidate in candidates:
        selected = [record for record in records if record.get("candidate") == candidate]
        passed = sum(bool(record.get("qualified")) for record in selected)
        errors = sum(record.get("status") != "ok" for record in selected)
        peak = max((record.get("peak_rss_mb", 0.0) for record in selected), default=0.0)
        wall = sum(record.get("wall_seconds", 0.0) for record in selected)
        lines.append(
            f"| `{candidate}` | {len(selected)} | {passed} | {errors} | {peak:.1f} | {wall:.1f} |"
        )
    required_synthetic = {case.name for case in synthetic_suite()}
    screening_rows: list[tuple[str, int, int, float, float]] = []
    screening_qualifiers: list[tuple[float, str]] = []
    for candidate in candidates:
        selected = [
            record for record in records
            if record.get("kind") == "synthetic"
            and record.get("candidate") == candidate
        ]
        passed = sum(bool(record.get("qualified")) for record in selected)
        residuals = [
            float(record["surface_residual_hd95_radius"])
            for record in selected
            if record.get("surface_residual_hd95_radius") is not None
        ]
        worst = max(residuals, default=math.inf)
        mean = float(np.mean(residuals)) if residuals else math.inf
        screening_rows.append((candidate, len(selected), passed, worst, mean))
        complete = (
            {record.get("case") for record in selected} == required_synthetic
            and all(
                record.get("status") == "ok" and record.get("qualified")
                for record in selected
            )
        )
        if complete and math.isfinite(worst):
            screening_qualifiers.append((worst, candidate))
    if any(count for _candidate, count, _passed, _worst, _mean in screening_rows):
        lines.extend(
            [
                "",
                "## Synthetic geometry screen",
                "",
                "| Candidate | Cases | Passed | Worst HD95 / radius | Mean HD95 / radius |",
                "|---|---:|---:|---:|---:|",
            ]
        )
        for candidate, count, passed, worst, mean in screening_rows:
            worst_text = f"{worst:.6g}" if math.isfinite(worst) else "n/a"
            mean_text = f"{mean:.6g}" if math.isfinite(mean) else "n/a"
            lines.append(
                f"| `{candidate}` | {count} | {passed} | {worst_text} | {mean_text} |"
            )
        if screening_qualifiers:
            screening_qualifiers.sort()
            screen_error, screen_leader = screening_qualifiers[0]
            lines.extend(
                [
                    "",
                    f"Synthetic screening leader: `{screen_leader}` "
                    f"(worst HD95/radius {screen_error:.6g}). This is not a CFD "
                    "qualification; the real-data gates still control profile promotion.",
                ]
            )
    diagnostic_real = [
        record
        for record in records
        if record.get("kind") == "real" and not _is_qualification_record(record)
    ]
    qualification_real = [
        record
        for record in records
        if record.get("kind") == "real" and _is_qualification_record(record)
    ]
    if diagnostic_real:
        lines.extend(
            [
                "",
                "## Diagnostic reconstruction runs — NOT CFD-qualified",
                "",
                "These runs exist to attribute regressions to the field, the junction "
                "model or the extractor. They are reported even when the input audit "
                "is unqualified, and they can never promote a profile.",
                "",
                "| Candidate | Case | ROI | Input qualified | Status | Complete | Genus-clean | Coverage complete | Missing branches | Tiny components | HD95 mm |",
                "|---|---|---|---|---|---|---|---|---:|---:|---:|",
            ]
        )
        for record in sorted(
            diagnostic_real,
            key=lambda item: (
                str(item.get("candidate")),
                str(item.get("case")),
                str(item.get("roi") or ""),
            ),
        ):
            coverage = record.get("coverage_metrics") or {}
            manifest = record.get("component_manifest") or {}
            gate = record.get("cfd_gate") or {}
            hd95 = (record.get("surface_metrics") or {}).get("hd95_mm")
            lines.append(
                "| `{candidate}` | {case} | {roi} | {input_ok} | {status} | {complete} | "
                "{topology} | {coverage_ok} | {missing} | {tiny} | {hd95} |".format(
                    candidate=record.get("candidate"),
                    case=record.get("case"),
                    roi=record.get("roi") or "—",
                    input_ok="yes" if record.get("input_qualified") else "**no**",
                    status=record.get("status"),
                    complete="no" if manifest.get("incomplete") else "yes",
                    topology="yes" if gate.get("topology_ok") else "no",
                    coverage_ok="yes" if coverage.get("complete") else "no",
                    missing=len(coverage.get("missing_segment_indices") or []),
                    tiny=coverage.get("tiny_component_count", "—"),
                    hd95=f"{hd95:.5g}" if isinstance(hd95, (int, float)) else "—",
                )
            )
        unqualified = sum(
            1 for record in diagnostic_real if not record.get("input_qualified")
        )
        total_real = len(diagnostic_real) + len(qualification_real)
        lines.extend(
            [
                "",
                f"{unqualified} of {total_real} real run(s) reconstructed despite an "
                "unqualified input audit; their metrics are diagnostic evidence only.",
            ]
        )

    audit_by_site: dict[tuple[Any, ...], dict[str, Any]] = {}
    for record in records:
        if record.get("kind") != "real":
            continue
        for audit in record.get("conflict_audit", []):
            key = (
                record.get("case"),
                audit.get("capsule_a"),
                audit.get("capsule_b"),
            )
            audit_by_site[key] = audit
    audits = list(audit_by_site.values())
    if audits:
        classifications: dict[str, int] = {}
        for audit in audits:
            name = str(audit.get("classification", "unknown"))
            classifications[name] = classifications.get(name, 0) + 1
        lines.extend(["", "## Mask-backed input conflict audit", ""])
        for name, count in sorted(classifications.items()):
            lines.append(f"- `{name}`: {count}")
    lines.extend(
        [
            "",
            "## Qualification gates",
            "",
            "- Closed, orientable, edge- and vertex-manifold, self-intersection-free meshes.",
            "- Correct expected topology on every feasible synthetic case.",
            "- HD95 no greater than two source voxels on both real cases.",
            "- Median resolved-vessel cross-sectional area error no greater than 10%.",
            "- Timeouts, memory exits, and unsupported runs count as failures, not missing data.",
        ]
    )
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    qualified_path = output / "cfd_qualified_profile.json"
    winning_path = output / "winning_config.json"
    # This file is the single path by which a configuration reaches production,
    # so re-derive the contributing records and refuse on anything diagnostic.
    contributing = [
        record
        for record in records
        if record.get("kind") == "real" and record.get("candidate") == winner
    ]
    refusal = None
    if winner:
        for record in contributing:
            if not _is_qualification_record(record):
                refusal = (
                    f"{record.get('case')} was run for "
                    f"purpose={record.get('purpose')!r} roi={record.get('roi')!r}"
                )
            elif record.get("input_qualified") is not True:
                refusal = f"{record.get('case')} has an unqualified input audit"
            elif record.get("component_manifest", {}).get("incomplete", False):
                refusal = f"{record.get('case')} did not emit every graph component"
            if refusal:
                break
    if winner and not refusal:
        selected_config = candidate_config(winner).to_dict()
        winning_path.write_text(
            json.dumps(selected_config, indent=2, sort_keys=True), encoding="utf-8"
        )
        if winner == "graph_round_cone_vtk_htg":
            qualified_path.write_text(
                json.dumps(
                    {
                        "qualified": True,
                        "diagnostic_only": False,
                        "winner": winner,
                        "real_cases": sorted(required_real_cases),
                        "input_qualified_cases": sorted(
                            {
                                str(record.get("case"))
                                for record in contributing
                                if record.get("input_qualified") is True
                            }
                        ),
                        "config": selected_config,
                    },
                    indent=2,
                    sort_keys=True,
                ),
                encoding="utf-8",
            )
    else:
        if refusal:
            print(f"  [PROFILE][REFUSED] not emitting a qualification token: {refusal}")
        # These are generated benchmark artifacts. A failed/resumed run must
        # never leave behind a stale production qualification token.
        for stale in (winning_path, qualified_path):
            if stale.exists():
                stale.unlink()
    (output / "preprocessor_selection.json").write_text(
        json.dumps(
            {
                "preprocessor": preprocessor,
                "reason": preprocessing_reason,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def _spec_id(spec: dict[str, Any]) -> str:
    parts = [str(spec["kind"]), str(spec.get("candidate", spec.get("preprocessor", ""))), str(spec.get("case", ""))]
    # An ROI spec and a full-graph spec can otherwise agree on kind/candidate/
    # case/resolution and overwrite each other's runs/<id>.json.
    if spec.get("roi"):
        parts.append(f"roi-{spec['roi']}")
    purpose = spec.get("purpose")
    if purpose and purpose != "qualification":
        parts.append(purpose[:1])
    if spec.get("resolution_factor") is not None:
        parts.append(f"h{spec['resolution_factor']}")
    if spec.get("cells_across_diameter") is not None:
        parts.append(f"n{spec['cells_across_diameter']}")
    if spec.get("blend_fraction") is not None:
        parts.append(f"b{spec['blend_fraction']}")
    if spec.get("blend_support") is not None:
        parts.append(f"s{spec['blend_support']}")
    return "__".join(parts).replace(".", "p").replace("/", "_")


def _tune_junctions(
    output: Path,
    scratch: Path,
    cache: Path,
    limits: RunLimits,
    *,
    resume: bool,
) -> tuple[float, float, list[dict[str, Any]]]:
    junction_cases = [
        case for case in synthetic_suite()
        if case.category == "junction" and case.expected_valid
    ]
    records: list[dict[str, Any]] = []
    for blend in BLEND_FRACTIONS:
        for support in BLEND_SUPPORTS:
            for case in junction_cases:
                spec = {
                    "kind": "synthetic",
                    "case": case.name,
                    "candidate": "graph_round_cone_dense_mc",
                    "preprocessor": "none",
                    "blend_fraction": blend,
                    "blend_support": support,
                    "scratch": str(scratch),
                    "cache_dir": str(cache),
                    "tuning": True,
                }
                run_id = "tune__" + _spec_id(spec)
                result_path = output / "runs" / f"{run_id}.json"
                result_path.parent.mkdir(exist_ok=True)
                if resume and result_path.exists():
                    record = json.loads(result_path.read_text(encoding="utf-8"))
                else:
                    record = _monitor_worker(spec, result_path, limits)
                    result_path.write_text(json.dumps(record, indent=2), encoding="utf-8")
                record["tuning"] = True
                records.append(record)
    scored: list[tuple[float, float, float]] = []
    # Group by deterministic loop order because successful worker results only
    # contain measured fields, not every orchestration key.
    group_size = len(junction_cases)
    cursor = 0
    for blend in BLEND_FRACTIONS:
        for support in BLEND_SUPPORTS:
            selected = records[cursor : cursor + group_size]
            cursor += group_size
            for record in selected:
                record["tuning_key"] = f"{blend}:{support}"
                record["blend_fraction"] = blend
                record["blend_support"] = support
            if selected and all(
                record.get("status") == "ok" and record.get("qualified")
                for record in selected
            ):
                score = max(
                    float(record.get("surface_residual_hd95_radius", math.inf))
                    for record in selected
                )
                scored.append((score, blend, support))
    if scored:
        _score, best_blend, best_support = min(scored)
    else:
        best_blend, best_support = 0.15, 4.0
    (output / "junction_tuning.json").write_text(
        json.dumps(
            {
                "blend_fraction": best_blend,
                "support_factor": best_support,
                "qualified_combinations": len(scored),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return best_blend, best_support, records


def _load_manifest(path: Path | None) -> list[dict[str, Any]]:
    if path is None:
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    cases = data.get("cases", [])
    for case in cases:
        for key in ("name", "graph", "segmentation"):
            if key not in case:
                raise ValueError(f"manifest case missing {key!r}")
            if key != "name" and not Path(case[key]).exists():
                raise FileNotFoundError(case[key])
        # A region of interest is a fraction of the network, so it can never
        # stand in for whole-network qualification. Make that structural rather
        # than a convention someone can forget.
        if case.get("roi") and case.get("purpose") != "diagnostic":
            raise ValueError(
                f"manifest case {case['name']!r} declares an ROI and must set "
                "\"purpose\": \"diagnostic\""
            )
    return cases


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument(
        "--stage",
        choices=("preprocess", "synthetic", "real", "ablation", "all"),
        default="synthetic",
    )
    parser.add_argument("--candidate", action="append", choices=sorted(CANDIDATES))
    parser.add_argument(
        "--case",
        action="append",
        help="limit synthetic/preprocessing runs to named cases (repeatable)",
    )
    parser.add_argument(
        "--real-case",
        action="append",
        help="limit real/ablation runs to named manifest cases (repeatable)",
    )
    parser.add_argument(
        "--ablation-resolution-factor",
        type=float,
        default=1.0,
        help="dense spacing factor used by the ablation stage",
    )
    parser.add_argument(
        "--ablation-cells",
        type=float,
        default=12.0,
        help="cells across diameter used by the ablation stage",
    )
    parser.add_argument("--preprocessor", choices=PREPROCESSORS)
    parser.add_argument("--timeout", type=float, default=7200.0)
    parser.add_argument("--memory-mb", type=float, default=16_384.0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--skip-junction-tuning", action="store_true")
    parser.add_argument("--worker-spec", type=Path)
    parser.add_argument("--worker-result", type=Path)
    args = parser.parse_args(argv)
    if args.worker_spec:
        if args.worker_result is None:
            parser.error("--worker-result is required with --worker-spec")
        return _worker_main(args.worker_spec, args.worker_result)
    if args.output is None:
        parser.error("--output is required")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    scratch = output / "scratch"
    cache = output / "cache"
    scratch.mkdir(exist_ok=True)
    cache.mkdir(exist_ok=True)
    candidates = args.candidate or list(CANDIDATES)
    limits = RunLimits(args.timeout, args.memory_mb)
    records: list[dict[str, Any]] = []
    blend_fraction, blend_support = 0.05, 2.0
    tuning_path = output / "junction_tuning.json"
    if args.skip_junction_tuning and tuning_path.exists():
        frozen = json.loads(tuning_path.read_text(encoding="utf-8"))
        blend_fraction = float(frozen["blend_fraction"])
        blend_support = float(frozen["support_factor"])
    if (
        not args.skip_junction_tuning
        and args.stage in {"synthetic", "real", "all"}
    ):
        blend_fraction, blend_support, tuning_records = _tune_junctions(
            output, scratch, cache, limits, resume=args.resume
        )
        records.extend(tuning_records)
    specs: list[dict[str, Any]] = []
    suite = synthetic_suite()
    if args.case:
        requested = set(args.case)
        available = {case.name for case in suite}
        unknown = requested - available
        if unknown:
            parser.error(f"unknown synthetic case(s): {', '.join(sorted(unknown))}")
        suite = [case for case in suite if case.name in requested]
    if args.stage in {"preprocess", "all"}:
        preprocessors = (args.preprocessor,) if args.preprocessor else PREPROCESSORS
        for preprocessor in preprocessors:
            for case in suite:
                specs.append(
                    {
                        "kind": "preprocessor",
                        "case": case.name,
                        "preprocessor": preprocessor,
                    }
                )
    if args.stage in {"synthetic", "all"}:
        for candidate in candidates:
            for case in suite:
                specs.append(
                    {
                        "kind": "synthetic",
                        "case": case.name,
                        "candidate": candidate,
                        "preprocessor": "none",
                        "blend_fraction": blend_fraction,
                        "blend_support": blend_support,
                    }
                )
    real_cases: list[dict[str, Any]] = []
    if args.stage in {"real", "ablation", "all"}:
        real_cases = _load_manifest(args.manifest)
        if not real_cases:
            parser.error(f"{args.stage} stage requires a non-empty --manifest")
        if args.real_case:
            requested = set(args.real_case)
            available = {case["name"] for case in real_cases}
            unknown = requested - available
            if unknown:
                parser.error(f"unknown real case(s): {', '.join(sorted(unknown))}")
            real_cases = [case for case in real_cases if case["name"] in requested]

    if args.stage in {"real", "all"}:
        for candidate in candidates:
            adaptive = CANDIDATES[candidate]["SDF_MESH_METHOD"] in {
                "adaptive", "vtk_htg"
            }
            values = ADAPTIVE_CELLS if adaptive else DENSE_SPACING_FACTORS
            for case in real_cases:
                for value in values:
                    specs.append(
                        {
                            "kind": "real",
                            "candidate": candidate,
                            "case": case["name"],
                            "real_case": case,
                            "preprocessor": "none",
                            "purpose": case.get("purpose", "qualification"),
                            "roi": case.get("roi"),
                            "resolution_factor": None if adaptive else value,
                            "cells_across_diameter": value if adaptive else None,
                            "blend_fraction": blend_fraction,
                            "blend_support": blend_support,
                        }
                    )

    if args.stage == "ablation":
        # One matched resolution per family, so the four cells differ in the
        # field and the extractor only. Validation is relaxed to 'warn' because
        # 'error' turns a speckled mesh into an exception and destroys exactly
        # the evidence this stage exists to collect.
        for candidate in candidates:
            adaptive = CANDIDATES[candidate]["SDF_MESH_METHOD"] in {
                "adaptive", "vtk_htg"
            }
            for case in real_cases:
                specs.append(
                    {
                        "kind": "real",
                        "candidate": candidate,
                        "case": case["name"],
                        "real_case": case,
                        "preprocessor": "none",
                        "purpose": "diagnostic",
                        "roi": case.get("roi") or "full",
                        "resolution_factor": (
                            None if adaptive else args.ablation_resolution_factor
                        ),
                        "cells_across_diameter": (
                            args.ablation_cells if adaptive else None
                        ),
                        "blend_fraction": blend_fraction,
                        "blend_support": blend_support,
                        "validation_mode": "warn",
                        "coverage_containment": True,
                    }
                )
    for spec_index, spec in enumerate(specs, start=1):
        spec.update(
            scratch=str(scratch),
            cache_dir=str(cache),
            diagnostic_dir=str(output / "diagnostics"),
        )
        run_id = _spec_id(spec)
        result_path = output / "runs" / f"{run_id}.json"
        result_path.parent.mkdir(exist_ok=True)
        if args.resume and result_path.exists():
            record = json.loads(result_path.read_text(encoding="utf-8"))
        else:
            record = _monitor_worker(spec, result_path, limits)
            result_path.write_text(json.dumps(record, indent=2), encoding="utf-8")
        records.append(record)
        all_records = _load_completed_records(output)
        _write_outputs(all_records, output)
        _write_report(all_records, output, list(CANDIDATES))
        print(
            f"[{spec_index}/{len(specs)}] {run_id}: "
            f"{record.get('status')} qualified={record.get('qualified')}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["CANDIDATES", "candidate_config", "execute_worker", "main"]
