"""Export Simpleware quality-inspection diagnostics from an existing SIP.

This is intentionally separate from ``simpleware_mesh_quality.py``. A saved
volume mesh does not retain the generation-time ``Mesh`` statistics object, but
the document quality inspector can recompute its configured problem counts.
"""

import csv
import json
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

from simpleware.scripting import App, QualityInspectionTool


STARTED = time.perf_counter()


def log(message):
    print(
        "[{} +{:8.1f}s] {}".format(
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            time.perf_counter() - STARTED,
            message,
        ),
        flush=True,
    )


def _level_name(level):
    mapping = (
        (QualityInspectionTool.InspectionMetricOff, "off"),
        (QualityInspectionTool.InspectionMetricWarning, "warning"),
        (QualityInspectionTool.InspectionMetricError, "error"),
        (QualityInspectionTool.InspectionMetricFeature, "feature"),
    )
    for value, name in mapping:
        if level == value:
            return name
    return "unknown"


def main():
    app = App.GetInstance()
    job_path = Path(str(app.GetInputValue() or "").strip().strip('"')).resolve()
    if not job_path.is_file():
        raise RuntimeError("Quality-inspection job JSON not found: {}".format(job_path))
    job = json.loads(job_path.read_text(encoding="utf-8"))
    project_path = Path(job["case_sip"]).resolve()
    if not project_path.is_file():
        raise RuntimeError("Case SIP not found: {}".format(project_path))
    log("Opening existing mesh SIP: {}".format(project_path))
    document = app.OpenDocument(str(project_path))
    if not document.HasActiveModel():
        models = list(document.GetModels())
        if len(models) != 1:
            raise RuntimeError(
                "SIP has no active model and does not contain exactly one model"
            )
        document.SetActiveModel(models[0])
        log("Activated the SIP's only model for quality inspection.")

    tool = document.GetQualityInspectionTool()
    tool.SetInspectionMode(QualityInspectionTool.InspectCells)
    tool.SetInspectionPartsMode(QualityInspectionTool.InspectionAllParts)
    log("Running Simpleware cell-quality inspection (no remeshing).")
    tool.ActivateInspectionMode(False)
    try:
        levels = (
            ("warning", QualityInspectionTool.InspectionMetricWarning),
            ("error", QualityInspectionTool.InspectionMetricError),
            ("feature", QualityInspectionTool.InspectionMetricFeature),
        )
        totals = {
            name: int(tool.GetNumberOfItemsTotal(value))
            for name, value in levels
        }
        rows = []
        for index in range(int(tool.GetNumberOfMetrics())):
            name = str(tool.GetMetricName(index))
            level = tool.GetMetricLevel(index)
            uses_threshold = bool(tool.GetMetricUsesThreshold(index))
            rows.append({
                "case_id": str(job["case_id"]),
                "metric_index": index,
                "metric": name,
                "level": _level_name(level),
                "uses_threshold": uses_threshold,
                "threshold": (
                    float(tool.GetMetricThreshold(index))
                    if uses_threshold else None
                ),
                "less_than_threshold_is_valid": (
                    bool(tool.GetMetricLessThanThresholdIsValid(index))
                    if uses_threshold else None
                ),
                "problem_count": int(
                    tool.GetNumberOfSpecificMetricProblemsTotal(index)
                ),
            })
    finally:
        tool.DeactivateInspectionMode()

    payload = {
        "schema_version": 1,
        "source": "Simpleware X-2025.06 QualityInspectionTool",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "case_id": str(job["case_id"]),
        "case_sip": str(project_path),
        "inspection_mode": "cells",
        "totals": totals,
        "metrics": rows,
        "note": (
            "This is a post-save inspector audit. Generation-time distribution "
            "statistics are available only in mesh_quality.json created during "
            "new mesh generation."
        ),
    }
    json_path = Path(job["quality_inspection_json"]).resolve()
    csv_path = Path(job["quality_inspection_csv"]).resolve()
    json_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(
        json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    fields = (
        "case_id", "metric_index", "metric", "level", "uses_threshold",
        "threshold", "less_than_threshold_is_valid", "problem_count",
    )
    with csv_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    log(
        "Exported {} inspector metrics (errors={}, warnings={}).".format(
            len(rows), totals["error"], totals["warning"]
        )
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        log("ERROR: {}: {}".format(type(exc).__name__, exc))
        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        raise
