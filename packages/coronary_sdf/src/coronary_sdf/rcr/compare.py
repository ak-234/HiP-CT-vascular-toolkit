"""Separate reduced-network comparisons from qualified CFD observations."""
import json
from pathlib import Path
import numpy as np


def compare(results, config):
    reference = config.get("comparison", {}).get("reference_method", "hipct")
    trees = {r["tree_id"] for r in results["results"]}
    output = {"reference_method": reference, "network_sha256": results.get("network_sha256"),
              "network_flow_comparisons": [], "cfd_comparisons": [],
              "cfd_status": "not_supplied"}
    for tree in sorted(trees):
        group = [r for r in results["results"] if r["tree_id"] == tree]
        ref = next((r for r in group if r["method"] == reference), None)
        if ref is None:
            raise ValueError(f"Missing {reference} reference for {tree}")
        truth = {o["outlet_id"]: o["flow_m3_s"] for o in ref["outlets"]}
        for case in group:
            values = {o["outlet_id"]: o["flow_m3_s"] for o in case["outlets"]}
            if values.keys() != truth.keys():
                raise ValueError("Cannot compare unmatched anatomical outlets")
            errors = {k: (values[k]-truth[k])/truth[k] for k in truth}
            output["network_flow_comparisons"].append(dict(tree_id=tree, method=case["method"],
                relative_errors=errors, rms_relative_error=float(np.sqrt(np.mean(np.square(list(errors.values()))))),
                relative_mass_error=case["relative_mass_error"]))
    path = config.get("comparison", {}).get("observations")
    if path:
        observations = json.loads(Path(path).read_text(encoding="utf-8"))
        for obs in observations:
            if obs["network_sha256"] != results.get("network_sha256"):
                raise ValueError("CFD geometry/network provenance mismatch")
            if obs["experiment"] not in ("open_cut", "sealed_branch_removal", "uncorrected_sensitivity"):
                raise ValueError("Explicit experiment type required")
            checks = obs["qualification"]
            if not all(checks.get(k) is True for k in
                       ("mass_conservation", "periodic_convergence", "mesh_convergence", "time_step_convergence")):
                raise ValueError("CFD comparison is not qualified for interpretation")
            metrics = {}
            for kind in ("flow", "impedance", "waveform", "wss"):
                rows = obs.get(kind, [])
                if not rows:
                    continue
                if len({r["matched_id"] for r in rows}) != len(rows):
                    raise ValueError("Duplicate matched region/sample identifier")
                a = np.asarray([r["reference"] for r in rows], float)
                b = np.asarray([r["candidate"] for r in rows], float)
                if a.shape != b.shape or not np.all(np.isfinite([a, b])):
                    raise ValueError("Invalid comparison samples")
                norm = float(np.linalg.norm(a))
                if norm == 0:
                    raise ValueError("Cannot normalise errors by a zero reference")
                metrics[kind] = {"relative_l2_error": float(np.linalg.norm(b-a)/norm), "count": len(rows)}
            output["cfd_comparisons"].append({"method": obs["method"], "experiment": obs["experiment"], "metrics": metrics})
        output["cfd_status"] = "qualified_observations"
    return output
