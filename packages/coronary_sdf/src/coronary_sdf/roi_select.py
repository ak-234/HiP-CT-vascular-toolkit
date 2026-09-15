"""Feature-seeded regions of interest for real vascular graphs.

Whole-network runs cannot carry the ablation. Both LADAF graphs span roughly
90 mm at a 66 µm source voxel, so every dense resolution the benchmark offers
exceeds ``BSPLINE_SDF_MAX_VOXELS`` and exits as a resource skip before meshing.
Comparing a dense field against an adaptive one therefore requires small,
representative regions.

Regions are located from the graph itself rather than picked by eye, so each one
provably contains the feature it claims to, and the resolved boxes are frozen to
JSON so later runs are reproducible even if the detectors change.

Three boxes matter and are kept distinct:

``seed box``
    radius-scaled box around the detected feature; selects segments.
``graph subset``
    every segment with any point in the seed box, kept **whole**. Clipping a
    segment mid-way invents a dangling terminal at an interior radius, which is
    exactly the carina and terminal behaviour the ablation is measuring.
``evaluation box``
    bounding box of the retained subset plus its radii. This, not the seed box,
    is what mask-restricted metrics must use.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import sys
from typing import Any, Iterable, Sequence

import numpy as np

from .epicardial_annotation import filter_points, recompute_node_degrees, write_amira_xml
from .parse_amira import parse_xml

#: Radii below this multiple of the source voxel are treated as degenerate
#: measurement artefacts rather than as the smallest real vessel. LADAF-28 has
#: 40 of 29,122 points below 1 µm against a 1st-percentile of 57 µm; seeding a
#: region there would target a data defect, not anatomy.
DEGENERATE_RADIUS_VOXELS = 0.5

FEATURE_KINDS = (
    "degree5_junction",
    "trunk_bifurcation",
    "tortuous_return",
    "parallel_pair",
    "minimum_radius",
)


@dataclass(frozen=True)
class RoiFeature:
    kind: str
    node_id: int | None
    segment_indices: tuple[int, ...]
    segment_ids: tuple[int, ...]
    centre_mm: tuple[float, float, float]
    radius_mm: float
    score: float
    detail: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _node_degrees(segments: Sequence[dict[str, Any]]) -> dict[int, int]:
    degree: dict[int, int] = {}
    for segment in segments:
        for key in ("node1", "node2"):
            node = segment.get(key)
            if node is not None:
                degree[int(node)] = degree.get(int(node), 0) + 1
    return degree


def _incident(segments: Sequence[dict[str, Any]]) -> dict[int, list[int]]:
    incident: dict[int, list[int]] = {}
    for index, segment in enumerate(segments):
        for key in ("node1", "node2"):
            node = segment.get(key)
            if node is not None:
                incident.setdefault(int(node), []).append(index)
    return incident


def _segment_arrays(
    points: dict[int, tuple], segment: dict[str, Any]
) -> tuple[np.ndarray, np.ndarray]:
    ids = [pid for pid in segment.get("point_ids", []) if pid in points]
    if not ids:
        return np.empty((0, 3)), np.empty(0)
    coords = np.asarray([points[pid][:3] for pid in ids], dtype=float) / 1000.0
    radii = np.asarray([points[pid][3] for pid in ids], dtype=float) / 1000.0
    return coords, radii


def _segment_mean_radius(points: dict[int, tuple], segment: dict[str, Any]) -> float:
    _coords, radii = _segment_arrays(points, segment)
    return float(np.mean(radii)) if len(radii) else 0.0


def _node_position(
    nodes: dict[int, tuple], node_id: int
) -> tuple[float, float, float]:
    record = nodes[node_id]
    return tuple(float(v) / 1000.0 for v in record[:3])


def _segment_id(segments: Sequence[dict[str, Any]], index: int) -> int:
    return int(segments[index].get("id", index))


def find_roi_features(
    nodes: dict[int, tuple],
    points: dict[int, tuple],
    segments: Sequence[dict[str, Any]],
    *,
    voxel_mm: float = 0.066,
    conflict_audit: Iterable[dict[str, Any]] | None = None,
    kinds: Sequence[str] | None = None,
) -> list[RoiFeature]:
    """Locate the representative features the ablation requires.

    Every detector sorts its candidates on an explicit key, so the result does
    not depend on dictionary ordering.
    """

    wanted = set(kinds) if kinds else set(FEATURE_KINDS)
    degree = _node_degrees(segments)
    incident = _incident(segments)
    features: list[RoiFeature] = []
    radius_floor = DEGENERATE_RADIUS_VOXELS * float(voxel_mm)

    def _node_feature(kind: str, node_id: int, score: float, detail: dict) -> RoiFeature:
        members = sorted(incident.get(node_id, []))
        radii = [_segment_mean_radius(points, segments[i]) for i in members]
        return RoiFeature(
            kind=kind,
            node_id=int(node_id),
            segment_indices=tuple(members),
            segment_ids=tuple(_segment_id(segments, i) for i in members),
            centre_mm=_node_position(nodes, node_id),
            radius_mm=float(max(radii)) if radii else 0.0,
            score=float(score),
            detail=detail,
        )

    # ── highest-degree junction ──────────────────────────────────────────────
    if "degree5_junction" in wanted:
        candidates = [n for n in degree if n in nodes and degree[n] >= 4]
        if candidates:
            best = max(sorted(candidates), key=lambda n: (degree[n], -n))
            features.append(
                _node_feature(
                    "degree5_junction",
                    best,
                    degree[best],
                    {"degree": degree[best], "exact_degree_five": degree[best] == 5},
                )
            )

    # ── trunk bifurcation: degree 3 carrying the largest vessel ──────────────
    if "trunk_bifurcation" in wanted:
        best_node, best_radius = None, -1.0
        for node_id in sorted(n for n in degree if degree[n] == 3 and n in nodes):
            radius = max(
                (_segment_mean_radius(points, segments[i]) for i in incident[node_id]),
                default=0.0,
            )
            if radius > best_radius:
                best_node, best_radius = node_id, radius
        if best_node is not None:
            features.append(
                _node_feature(
                    "trunk_bifurcation",
                    best_node,
                    best_radius,
                    {"max_incident_mean_radius_mm": best_radius},
                )
            )

    # ── tortuous return: a genuine hairpin, not merely high curvature ────────
    if "tortuous_return" in wanted:
        best = None
        for index, segment in enumerate(segments):
            coords, radii = _segment_arrays(points, segment)
            if len(coords) < 12:
                continue
            steps = np.linalg.norm(np.diff(coords, axis=0), axis=1)
            arc = np.concatenate(([0.0], np.cumsum(steps)))
            local = float(np.median(radii))
            if local <= 0.0:
                continue
            # Compare only points far apart along the centreline; nearby pairs
            # are trivially close and say nothing about self-approach.
            window = max(6.0 * local, 0.1 * arc[-1])
            gaps = np.abs(arc[:, None] - arc[None, :])
            distances = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=2)
            mask = gaps > window
            if not np.any(mask):
                continue
            ratio = float(np.min(distances[mask]) / local)
            flat = np.argmin(np.where(mask, distances, np.inf))
            i, j = np.unravel_index(flat, distances.shape)
            if best is None or ratio < best[0]:
                best = (
                    ratio,
                    index,
                    tuple(0.5 * (coords[i] + coords[j])),
                    local,
                    float(arc[-1]),
                )
        if best is not None:
            ratio, index, centre, local, length = best
            features.append(
                RoiFeature(
                    kind="tortuous_return",
                    node_id=None,
                    segment_indices=(index,),
                    segment_ids=(_segment_id(segments, index),),
                    centre_mm=tuple(float(v) for v in centre),
                    radius_mm=local,
                    score=ratio,
                    detail={
                        "nonlocal_separation_radii": ratio,
                        "segment_length_mm": length,
                    },
                )
            )

    # ── close parallel branches, taken from the mask-backed audit ────────────
    if "parallel_pair" in wanted and conflict_audit:
        best = None
        for record in conflict_audit:
            cosine = record.get("tangent_cosine")
            a, b = record.get("segment_a"), record.get("segment_b")
            if cosine is None or a is None or b is None or int(a) == int(b):
                continue
            if float(cosine) < 0.9:
                continue
            clearance = record.get("capsule_clearance_mm")
            if clearance is None:
                continue
            if best is None or float(clearance) < best[0]:
                best = (float(clearance), record)
        if best is not None:
            clearance, record = best
            point_a = record.get("point_a_mm") or [0.0, 0.0, 0.0]
            point_b = record.get("point_b_mm") or point_a
            centre = 0.5 * (np.asarray(point_a, float) + np.asarray(point_b, float))
            indices = tuple(
                sorted({int(record["segment_a"]), int(record["segment_b"])})
            )
            radii = [_segment_mean_radius(points, segments[i]) for i in indices]
            features.append(
                RoiFeature(
                    kind="parallel_pair",
                    node_id=None,
                    segment_indices=indices,
                    segment_ids=tuple(_segment_id(segments, i) for i in indices),
                    centre_mm=tuple(float(v) for v in centre),
                    radius_mm=float(max(radii)) if radii else 0.0,
                    score=clearance,
                    detail={
                        "capsule_clearance_mm": clearance,
                        "tangent_cosine": float(record["tangent_cosine"]),
                        "classification": record.get("classification"),
                    },
                )
            )

    # ── smallest *resolvable* radius ─────────────────────────────────────────
    if "minimum_radius" in wanted:
        best = None
        degenerate = 0
        for index, segment in enumerate(segments):
            ids = [pid for pid in segment.get("point_ids", []) if pid in points]
            for pid in ids:
                radius = float(points[pid][3]) / 1000.0
                if radius < radius_floor:
                    degenerate += 1
                    continue
                if best is None or radius < best[0]:
                    best = (
                        radius,
                        index,
                        pid,
                        tuple(float(v) / 1000.0 for v in points[pid][:3]),
                    )
        if best is not None:
            radius, index, pid, centre = best
            features.append(
                RoiFeature(
                    kind="minimum_radius",
                    node_id=None,
                    segment_indices=(index,),
                    segment_ids=(_segment_id(segments, index),),
                    centre_mm=centre,
                    radius_mm=radius,
                    score=radius,
                    detail={
                        "point_id": int(pid),
                        "radius_floor_mm": radius_floor,
                        "degenerate_points_below_floor": degenerate,
                    },
                )
            )

    return features


def seed_bounds(
    feature: RoiFeature,
    *,
    margin_radii: float = 6.0,
    minimum_margin_mm: float = 0.5,
    maximum_margin_mm: float = 7.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Radius-scaled seed box around ``feature``, clamped to a sane budget."""

    margin = float(
        np.clip(
            margin_radii * max(feature.radius_mm, 0.0),
            minimum_margin_mm,
            maximum_margin_mm,
        )
    )
    centre = np.asarray(feature.centre_mm, dtype=float)
    return centre - margin, centre + margin


def subset_graph_by_bounds(
    nodes: dict[int, tuple],
    points: dict[int, tuple],
    segments: Sequence[dict[str, Any]],
    lower_mm: np.ndarray,
    upper_mm: np.ndarray,
) -> tuple[dict[int, tuple], dict[int, tuple], list[dict[str, Any]], dict[str, Any]]:
    """Keep whole segments touching the box, plus both endpoint nodes of each.

    ``recompute_node_degrees`` invents a placeholder node at the origin for any
    node id a segment references but the node dict lacks, so filtering nodes by
    position would silently teleport endpoints. Endpoint nodes are therefore
    retained regardless of whether they fall inside the box.
    """

    lower = np.asarray(lower_mm, dtype=float)
    upper = np.asarray(upper_mm, dtype=float)
    kept: list[dict[str, Any]] = []
    kept_indices: list[int] = []
    boundary: list[int] = []
    for index, segment in enumerate(segments):
        coords, _radii = _segment_arrays(points, segment)
        if not len(coords):
            continue
        inside = np.all((coords >= lower) & (coords <= upper), axis=1)
        if not np.any(inside):
            continue
        # Strahler is all-or-none in the Amira writer: once any segment carries
        # it, the rest are written as 0, silently rewriting real orders. Drop it.
        trimmed = {k: v for k, v in segment.items() if k != "strahler"}
        trimmed["point_ids"] = list(segment.get("point_ids", []))
        kept.append(trimmed)
        kept_indices.append(index)
        if not np.all(inside):
            boundary.append(index)

    sub_points = filter_points(points, kept)
    sub_nodes = recompute_node_degrees(nodes, kept)
    referenced = {int(s["node1"]) for s in kept} | {int(s["node2"]) for s in kept}
    missing = referenced - set(sub_nodes)
    if missing:
        raise AssertionError(
            f"ROI subset references node(s) {sorted(missing)} that were dropped; "
            "endpoint nodes must always be retained"
        )

    detail = {
        "segment_indices": kept_indices,
        "segment_ids": [_segment_id(segments, i) for i in kept_indices],
        "boundary_segment_indices": boundary,
        "boundary_segment_ids": [_segment_id(segments, i) for i in boundary],
        "n_segments": len(kept),
        "n_points": len(sub_points),
        "n_nodes": len(sub_nodes),
    }
    return sub_nodes, sub_points, kept, detail


def evaluation_bounds(
    points: dict[int, tuple],
    segments: Sequence[dict[str, Any]],
    *,
    padding_radii: float = 2.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Bounding box of the retained subset, padded by its own radii.

    Mask-restricted metrics must use this box, not the seed box, because whole
    segments extend beyond the seed.
    """

    lows, highs = [], []
    for segment in segments:
        coords, radii = _segment_arrays(points, segment)
        if not len(coords):
            continue
        pad = padding_radii * float(np.max(radii)) if len(radii) else 0.0
        lows.append(coords.min(axis=0) - pad)
        highs.append(coords.max(axis=0) + pad)
    if not lows:
        return np.zeros(3), np.zeros(3)
    return np.min(np.asarray(lows), axis=0), np.max(np.asarray(highs), axis=0)


def _dense_voxel_estimate(
    lower: np.ndarray, upper: np.ndarray, spacing_mm: float
) -> int:
    extent = np.maximum(upper - lower, 0.0)
    counts = np.maximum(np.ceil(extent / max(spacing_mm, 1e-12)) + 1, 1)
    return int(np.prod(counts))


def build_roi_cases(
    case: dict[str, str],
    output_dir: Path,
    *,
    voxel_mm: float = 0.066,
    margin_radii: float = 6.0,
    kinds: Sequence[str] | None = None,
    conflict_audit: Iterable[dict[str, Any]] | None = None,
    dense_spacing_factors: Sequence[float] = (2.0, 1.0, 0.5),
    required_factor: float = 1.0,
    maximum_dense_voxels: int = 200_000_000,
) -> list[dict[str, Any]]:
    """Materialise one ROI graph per detected feature and describe each one."""

    nodes, points, segments = parse_xml(case["graph"])
    features = find_roi_features(
        nodes,
        points,
        segments,
        voxel_mm=voxel_mm,
        conflict_audit=conflict_audit,
        kinds=kinds,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    graph_stat = Path(case["graph"]).stat()
    results: list[dict[str, Any]] = []

    for feature in features:
        lower, upper = seed_bounds(feature, margin_radii=margin_radii)
        sub_nodes, sub_points, sub_segments, detail = subset_graph_by_bounds(
            nodes, points, segments, lower, upper
        )
        if not sub_segments:
            print(f"  [ROI][SKIP] {feature.kind}: seed box retained no segments")
            continue
        eval_lo, eval_hi = evaluation_bounds(sub_points, sub_segments)
        # Segments are kept whole, so the evaluation box is driven by the
        # longest retained vessel rather than by the seed margin. Record the
        # dense cost at every resolution the benchmark offers, and require only
        # the resolution the attribution phase actually uses.
        budget = {
            float(factor): _dense_voxel_estimate(
                eval_lo, eval_hi, voxel_mm * float(factor)
            )
            for factor in dense_spacing_factors
        }
        feasible = sorted(
            factor for factor, count in budget.items()
            if count <= maximum_dense_voxels
        )
        required_voxels = budget.get(float(required_factor))
        if required_voxels is None or required_voxels > maximum_dense_voxels:
            print(
                f"  [ROI][SKIP] {feature.kind}: needs {required_voxels:,} dense "
                f"voxels at factor {required_factor} "
                f"(cap {maximum_dense_voxels:,}); feasible factors: {feasible}"
            )
            continue

        name = f"{case['name']}__{feature.kind}"
        graph_path = output_dir / f"{name}.xml"
        write_amira_xml(sub_nodes, sub_points, sub_segments, str(graph_path))
        descriptor = {
            "name": name,
            "source_case": case["name"],
            "roi": feature.kind,
            "purpose": "diagnostic",
            "graph": str(graph_path),
            "segmentation": case["segmentation"],
            "feature": feature.to_dict(),
            "seed_box_mm": [lower.tolist(), upper.tolist()],
            "evaluation_box_mm": [eval_lo.tolist(), eval_hi.tolist()],
            "margin_radii": margin_radii,
            "subset": detail,
            "dense_voxels_by_factor": {str(k): v for k, v in sorted(budget.items())},
            "feasible_dense_factors": feasible,
            "source_graph_fingerprint": {
                "path": str(Path(case["graph"]).resolve()),
                "size": graph_stat.st_size,
                "mtime_ns": graph_stat.st_mtime_ns,
            },
        }
        (output_dir / f"{name}.roi.json").write_text(
            json.dumps(descriptor, indent=2, sort_keys=True), encoding="utf-8"
        )
        results.append(descriptor)
        print(
            f"  [ROI] {feature.kind}: {detail['n_segments']} segment(s), "
            f"{detail['n_points']} point(s), r={feature.radius_mm:.5g} mm, "
            f"{required_voxels:,} dense voxels at factor {required_factor}, "
            f"feasible factors {feasible} -> {graph_path.name}"
        )
    return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate feature-seeded ROI graphs for the ablation harness."
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest-out", type=Path, required=True)
    parser.add_argument("--margin-radii", type=float, default=6.0)
    parser.add_argument("--voxel-mm", type=float, default=0.066)
    parser.add_argument(
        "--feature", action="append", choices=FEATURE_KINDS, default=None
    )
    parser.add_argument(
        "--audit-cache",
        type=Path,
        default=None,
        help="benchmark cache directory holding <case>.conflict-audit.json",
    )
    parser.add_argument("--case", action="append", default=None)
    args = parser.parse_args(argv)

    payload = json.loads(args.manifest.read_text(encoding="utf-8"))
    cases = payload.get("cases", payload if isinstance(payload, list) else [])
    if args.case:
        wanted = set(args.case)
        cases = [case for case in cases if case["name"] in wanted]

    descriptors: list[dict[str, Any]] = []
    for case in cases:
        print(f"[ROI] {case['name']}")
        audit = None
        if args.audit_cache:
            cache_path = args.audit_cache / f"{case['name']}.conflict-audit.json"
            if cache_path.exists():
                audit = json.loads(cache_path.read_text(encoding="utf-8")).get("audits")
                print(f"  reusing {len(audit or [])} cached conflict site(s)")
        descriptors.extend(
            build_roi_cases(
                case,
                args.output,
                voxel_mm=args.voxel_mm,
                margin_radii=args.margin_radii,
                kinds=args.feature,
                conflict_audit=audit,
            )
        )

    args.manifest_out.parent.mkdir(parents=True, exist_ok=True)
    args.manifest_out.write_text(
        json.dumps(
            {
                "cases": [
                    {
                        "name": item["name"],
                        "graph": item["graph"],
                        "segmentation": item["segmentation"],
                        "roi": item["roi"],
                        "purpose": "diagnostic",
                        "evaluation_box_mm": item["evaluation_box_mm"],
                    }
                    for item in descriptors
                ]
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    print(f"[ROI] wrote {len(descriptors)} case(s) to {args.manifest_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())


__all__ = [
    "DEGENERATE_RADIUS_VOXELS",
    "FEATURE_KINDS",
    "RoiFeature",
    "build_roi_cases",
    "evaluation_bounds",
    "find_roi_features",
    "seed_bounds",
    "subset_graph_by_bounds",
]
