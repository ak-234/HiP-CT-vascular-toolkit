"""Read-only section diagnostics with segmentation caches confined to output.

Run with --case case.json --output NEW_DIRECTORY --segments 67 270 307 308.
Uses the graph consumed by measurement, never the radius-corrected output.
"""
from __future__ import annotations

import argparse
from contextlib import redirect_stdout
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from coronary_sdf.rcr.prepare import digest, open_segmentation, write_json
from hipct_seg_debug.edit import __main__ as edit, diagnostics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--segments", type=int, nargs="+", required=True)
    args = parser.parse_args()
    config = json.loads(args.case.read_text(encoding="utf-8"))
    geometry = config["geometry"]
    paths = {}
    for key in ("graph", "segmentation"):
        path = Path(geometry[key])
        paths[key] = path.resolve() if path.is_absolute() else (args.case.parent/path).resolve()
    # Never overwrite an earlier diagnosis or place caches beside source data.
    args.output.mkdir(parents=True, exist_ok=False)
    hashes = {key: digest(path) for key, path in paths.items()}
    graph = edit._load([str(paths["graph"])])
    for sid in args.segments:
        graph.segment(sid)
    labels, frame, _ = open_segmentation(paths["segmentation"],
                                        geometry["acquisition_voxel_um"], args.output/"cache")
    if not np.allclose(np.asarray(frame.seg_spacing)/geometry["acquisition_voxel_um"],
                       geometry["segmentation_binning"], rtol=1e-5):
        raise ValueError("Configured binning disagrees with segmentation frame")
    with (args.output/"diagnostics.txt").open("w", encoding="utf-8") as stream:
        stream.write(
            "INPUT-GRAPH DIAGNOSTIC, NOT A REPEAT PERIMETER PASS.\n"
            "The legacy diagnostic uses a 64-voxel half-window and its own stability defaults.\n"
            "Its stable/exclusive counts must not replace measurement acceptance.\n"
            "If the input lacks radius provenance, source/reject columns are diagnostic\n"
            "defaults (filled/unmeasurable), not recorded measurement verdicts.\n\n")
        with redirect_stdout(stream):
            edit._correct_units(graph, frame, SimpleNamespace(
                graph=[str(paths["graph"])], voxel_um=geometry["acquisition_voxel_um"]))
        for sid in args.segments:
            print(f"Diagnosing source segment {sid}", flush=True)
            diagnostics.junction_terms(graph, frame, labels, [sid], out=stream)
            diagnostics.segment_report(graph, frame, labels, [sid], out=stream)
            stream.flush()
    after = {key: digest(path) for key, path in paths.items()}
    write_json(args.output/"provenance.json", dict(
        case=str(args.case.resolve()), case_sha256=digest(args.case),
        source_paths={key: str(path) for key, path in paths.items()},
        source_sha256=hashes, sources_unchanged=hashes == after,
        segments=args.segments, diagnostic_source_sha256=digest(diagnostics.__file__),
        note="Diagnostic section recuts do not replace estimator acceptance or hydraulic support."))
    if hashes != after:
        raise RuntimeError("Source data changed during diagnosis")


if __name__ == "__main__":
    main()
