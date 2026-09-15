"""Real-region DPC review, omega sweeps, ablations, and paper metrics."""

from __future__ import annotations

import copy
import csv
import json
import time
from pathlib import Path

import numpy as np

from . import dpc
from .candidates import Bridge, endpoint_tangent

MANIFEST_VERSION = 1
PAPER_LABELS = ("TP_b", "TN_b", "FP_b", "FN_b", "TP_s", "FP_s")


def load_regions(path: str | Path) -> dict:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if data.get("version") != MANIFEST_VERSION:
        raise ValueError(f"unsupported DPC region manifest version: {data.get('version')}")
    ids = [case["id"] for case in data.get("regions", [])]
    if len(ids) != len(set(ids)):
        raise ValueError("DPC region case IDs must be unique")
    return data


def parse_omega(spec: str) -> list[int]:
    text = str(spec).strip()
    if ":" in text:
        lo, hi = (int(v) for v in text.split(":", 1))
        if hi < lo:
            raise ValueError("omega range must be ascending")
        return list(range(lo, hi + 1))
    return [int(v) for v in text.split(",") if v.strip()]


def reconnection_metrics(labels) -> dict | None:
    counts = {name: 0 for name in PAPER_LABELS}
    found = False
    for label in labels:
        if label is None:
            continue
        if label not in counts:
            raise ValueError(f"unknown paper reconnection label: {label}")
        counts[label] += 1
        found = True
    if not found:
        return None
    numerator = counts["TP_b"] + counts["TP_s"]
    total = numerator + counts["TN_b"] + counts["FP_b"] + counts["FP_s"] + counts["FN_b"]
    sensitivity_den = numerator + counts["FN_b"]
    specificity_den = counts["TN_b"] + counts["FP_b"] + counts["FP_s"]
    return {
        "counts": counts,
        "RecAcc": (numerator + counts["TN_b"]) / max(total, 1),
        "RecSen": numerator / max(sensitivity_den, 1),
        "RecSpe": counts["TN_b"] / max(specificity_den, 1),
    }


def _region_roi(case, stack, frame, labels=None):
    from . import roi as roi_mod

    lo = np.asarray(case["raw_roi_zyx"][0], dtype=int)
    hi = np.asarray(case["raw_roi_zyx"][1], dtype=int) + 1
    volume = stack.read_stack_window(lo[0], hi[0], lo[1], hi[1], lo[2], hi[2])
    return roi_mod._wrap(volume, frame, labels, (lo[0], hi[0]),
                         (lo[1], hi[1]), (lo[2], hi[2]))


def bridge_for_case(graph, case) -> Bridge:
    source = int(case["source"]["node"])
    start = np.asarray(graph.nodes[source][:3], dtype=np.float64)
    source_info = endpoint_tangent(graph, source)
    r0 = source_info[1] if source_info is not None else 1.0
    reconnect_type = int(case["type"])
    if reconnect_type in (1, 2):
        target_node = int(case["target"]["node"])
        end = np.asarray(graph.nodes[target_node][:3], dtype=np.float64)
        target_info = endpoint_tangent(graph, target_node)
        r1 = target_info[1] if target_info is not None else r0
        bridge = Bridge(
            "endpoint", source, np.vstack([start, end]), np.array([r0, r1]),
            target_node=target_node, reconnection_type=reconnect_type,
        )
    else:
        sid = int(case["target"]["segment"])
        coords = np.asarray(graph.coords(sid), dtype=np.float64)
        anchor = np.asarray(case["target"]["world_um"], dtype=np.float64)
        index = int(np.argmin(np.linalg.norm(coords - anchor, axis=1)))
        end = coords[index]
        r1 = float(graph.radii(sid)[index])
        bridge = Bridge(
            "tjunction", source, np.vstack([start, end]), np.array([r0, r1]),
            target_segment=sid, target_index=index, reconnection_type=3,
        )
        bridge.metrics["target_point_id"] = graph.segment(sid)["point_ids"][index]
    bridge.metrics.update(r_source=float(r0), r_target=float(r1), span_um=bridge.span_um)
    return bridge


def _paper_label(case, run_key: str):
    labels = case.get("adjudication", {}).get("paper_labels", {})
    return labels.get(run_key, labels.get("all"))


def _run_one(graph, roi, probability, base_bridge, *, omega, ablation,
             neighbourhood_policy="two-level") -> dict:
    bridge = copy.deepcopy(base_bridge)
    params = dpc.DpcParams(omega=float(omega), neighbourhood_policy=neighbourhood_policy)
    if "D" not in ablation:
        params.w_distance = 0.0
    if "P" not in ablation:
        params.omega = 0.0
    if "C" not in ablation:
        params.w_cosine = 0.0
    started = time.perf_counter()
    dpc.refine(graph, roi, probability, [bridge], params=params)
    elapsed = time.perf_counter() - started
    return {
        "accepted": bool(bridge.accepted),
        "reached": bridge.metrics.get("dpc_reason") == "reached the target",
        "reason": bridge.reason,
        "walk_reason": bridge.metrics.get("dpc_reason"),
        "steps": int(bridge.metrics.get("dpc_steps", 0)),
        "runtime_seconds": elapsed,
        "path_um": np.asarray(bridge.coords).tolist(),
        "probability_sequence": bridge.metrics.get("dpc_probability_sequence", []),
        "grayscale_sequence": bridge.metrics.get("dpc_grayscale_sequence", []),
        "distance_scores": bridge.metrics.get("dpc_distance_scores", []),
        "probability_scores": bridge.metrics.get("dpc_probability_scores", []),
        "cosine_scores": bridge.metrics.get("dpc_cosine_scores", []),
        "probability_adf_p": bridge.metrics.get("probability_adf_p"),
        "grayscale_adf_p": bridge.metrics.get("grayscale_adf_p"),
        "hipct_safeguards": bridge.metrics.get("hipct_safeguards", {}),
        "paper_validation": bridge.metrics.get("paper_validation"),
    }


def _adjudication_outcome(case) -> dict:
    review = case.get("adjudication", {})
    correct_type = review.get("correct_type")
    correct_target = review.get("correct_target")
    proposed_target = case["target"].get("node", case["target"].get("segment"))
    return {
        "review_status": review.get("status", "pending"),
        "type_correct": None if correct_type is None else int(correct_type) == int(case["type"]),
        "target_correct": None if correct_target is None else str(correct_target) == str(proposed_target),
    }


def _summarise_runs(results, cases, run_key) -> dict | None:
    case_by_id = {case["id"]: case for case in cases}
    labels = [_paper_label(case_by_id[r["case_id"]], run_key) for r in results]
    metrics = reconnection_metrics(labels)
    if metrics is None:
        return None
    metrics["path_failures"] = sum(not r["accepted"] for r in results)
    by_type = {}
    for kind in (1, 2, 3):
        subset = [r for r in results if r["type"] == kind]
        subset_labels = [_paper_label(case_by_id[r["case_id"]], run_key) for r in subset]
        by_type[str(kind)] = reconnection_metrics(subset_labels)
    metrics["by_type"] = by_type
    return metrics


def evaluate_dpc(graph, stack, frame, labels, model_artifact, regions_path, *,
                 omega_spec="0:7", include_ablations=True, case_ids=None) -> dict:
    from .cfc import CfcProbability, load_model

    manifest = load_regions(regions_path)
    cases = manifest["regions"]
    if case_ids:
        wanted = {str(case_id).strip() for case_id in case_ids}
        cases = [case for case in cases if case["id"] in wanted]
        missing = wanted - {case["id"] for case in cases}
        if missing:
            raise ValueError(f"unknown DPC region case(s): {sorted(missing)}")
    omega_values = parse_omega(omega_spec)
    shared_model = load_model(model_artifact)
    providers = {}
    bridges = {}
    for case in cases:
        roi = _region_roi(case, stack, frame, labels)
        probability = CfcProbability(
            roi, model_artifact, model=shared_model, expected_raw_shape=stack.shape
        )
        probability.precompute()
        providers[case["id"]] = (roi, probability)
        bridges[case["id"]] = bridge_for_case(graph, case)

    runs = []
    summaries = {}
    for omega in omega_values:
        run_key = f"omega-{omega}"
        current = []
        for case in cases:
            roi, probability = providers[case["id"]]
            record = _run_one(graph, roi, probability, bridges[case["id"]],
                              omega=omega, ablation="DPC")
            record.update(case_id=case["id"], type=int(case["type"]),
                          omega=omega, ablation="DPC", run_key=run_key,
                          paper_label=_paper_label(case, run_key),
                          **_adjudication_outcome(case))
            runs.append(record)
            current.append(record)
        summaries[run_key] = _summarise_runs(current, cases, run_key)

    ranked = [
        (summary["RecAcc"], summary["RecSpe"], -summary["path_failures"], omega)
        for omega in omega_values
        if (summary := summaries.get(f"omega-{omega}")) is not None
    ]
    selected = max(ranked)[-1] if ranked else (5 if 5 in omega_values else omega_values[0])
    selection_status = "paper labels" if ranked else "default pending expert labels"

    if include_ablations:
        for ablation in ("DP", "PC", "DC"):
            run_key = f"ablation-{ablation}-omega-{selected}"
            current = []
            for case in cases:
                roi, probability = providers[case["id"]]
                record = _run_one(graph, roi, probability, bridges[case["id"]],
                                  omega=selected, ablation=ablation)
                record.update(case_id=case["id"], type=int(case["type"]),
                              omega=selected, ablation=ablation, run_key=run_key,
                              paper_label=_paper_label(case, run_key),
                              **_adjudication_outcome(case))
                runs.append(record)
                current.append(record)
            summaries[run_key] = _summarise_runs(current, cases, run_key)

    return {
        "version": 1,
        "regions": str(Path(regions_path).resolve()),
        "model": str(Path(model_artifact).resolve()),
        "omega_values": omega_values,
        "selected_omega": selected,
        "selection_status": selection_status,
        "paper_default_confirmed": selected == 5 if ranked else None,
        "summaries": summaries,
        "runs": runs,
    }


def export_regions(graph, stack, frame, labels, regions_path, output_directory) -> list[str]:
    """Write six-view contact sheets and a tabular adjudication template."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    manifest = load_regions(regions_path)
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    written = []
    rows = []
    for case in manifest["regions"]:
        roi = _region_roi(case, stack, frame, labels)
        bridge = bridge_for_case(graph, case)
        volume = np.asarray(roi.volume, dtype=np.float32)
        lo, hi = np.percentile(volume, [1, 99])
        image = np.clip((volume - lo) / max(hi - lo, 1e-6), 0, 1)
        mask = np.asarray(roi.mask, dtype=bool) if roi.mask is not None else None
        points = roi.to_index(bridge.coords)
        fig, axes = plt.subplots(2, 3, figsize=(13, 8), constrained_layout=True)
        projections = (
            (image.min(0), image.max(0), 2, 1, "axial (y/x)"),
            (image.min(1), image.max(1), 2, 0, "coronal (z/x)"),
            (image.min(2), image.max(2), 1, 0, "sagittal (z/y)"),
        )
        for col, (minimum, maximum, horizontal, vertical, title) in enumerate(projections):
            for row, panel in enumerate((minimum, maximum)):
                ax = axes[row, col]
                ax.imshow(panel, cmap="gray", origin="lower")
                if mask is not None:
                    mask_projection = mask.max(axis=col)
                    ax.contour(mask_projection.astype(float), levels=[0.5],
                               colors=["cyan"], linewidths=0.5)
                ax.plot(points[:, horizontal], points[:, vertical], "y-", linewidth=1.5)
                ax.scatter(points[:, horizontal], points[:, vertical], c=["lime", "red"], s=18)
                ax.set_title(("min " if row == 0 else "max ") + title)
                ax.set_axis_off()
        fig.suptitle(f"{case['id']} — Type {case['type']} — {case['role']}")
        destination = output / f"{case['id']}.png"
        fig.savefig(destination, dpi=160)
        plt.close(fig)
        written.append(str(destination))
        rows.append({
            "case_id": case["id"], "type": case["type"], "review_status": "pending",
            "reviewer": "", "correct_type": "", "correct_target": "",
            "paper_label": "", "notes": "",
        })
    with (output / "review_template.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    written.append(str(output / "review_template.csv"))
    return written
