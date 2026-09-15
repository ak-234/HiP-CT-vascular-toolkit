"""Sweep the bifurcation-geometry knobs and keep every surface for comparison.

The complaint this exists to answer: side branches emerge from the parent without a
carina, and the ostium does not flare -- an ostial radius should start large where
the branch meets the parent and taper down, and instead it steps.

Both mechanisms for this already exist in `config.py` and the current surface uses
the one that cannot produce a carina:

* ``SDF_FLAT_CAP_BIF`` (on) replaces the hemispherical end-cap with a plane, which
  removes the bif bulge "without narrowing the daughter's cross-section the way
  carina taper does" -- and equally without producing a flow divider.
* ``BIF_CARINA_ENABLE`` (off) tapers each endpoint to a conical tip so N cones
  smooth-min into a Y/T/X carina, at the price of narrowing the daughter.
* ``SDF_FLAT_CAP_BIF_SHIFT_FACTOR`` (0.0) pushes the cap plane past the node "so the
  daughter extends slightly into the parent before truncating and fills the ostium
  valley", with a suggested 0.2-0.4. Never tried; the sibling knob
  ``SDF_FLAT_CAP_BIF_SOFT_FACTOR`` carries a note saying it was tried and did
  nothing, so the valley is known and unsolved.

**There is no ground truth here.** Every other question this session was settled
against analytic geometry or an independent measurement; "the expected carina" is
anatomical judgement. This sweep can only produce candidates to look at, so it keeps
every mesh rather than scoring one winner.

Configurations are bound through `use_config`/`with_overrides`, never by editing
`config.py`, so the repository is unchanged and each run records exactly what it used.

Run: python research_scripts/bifurcation_sweep.py [name ...]
"""
from __future__ import annotations

import json
import sys
import time
import traceback
from pathlib import Path

import _paths

from coronary_sdf.config import default_config
from coronary_sdf.pipeline import run_pipeline

GRAPH = _paths.graph()
OUT_ROOT = _paths.out_dir() / "bifurcation_sweep"
#: name -> overrides. `baseline` reproduces `final_surface_capped/` and is included
#: so the comparison does not depend on that run having used identical settings.
CONFIGS = {
    "baseline":        {},
    "shift020":        {"SDF_FLAT_CAP_BIF_SHIFT_FACTOR": 0.20},
    "shift030":        {"SDF_FLAT_CAP_BIF_SHIFT_FACTOR": 0.30},
    "shift040":        {"SDF_FLAT_CAP_BIF_SHIFT_FACTOR": 0.40},
    "carina":          {"BIF_CARINA_ENABLE": True},
    "carina_shift030": {"BIF_CARINA_ENABLE": True,
                        "SDF_FLAT_CAP_BIF_SHIFT_FACTOR": 0.30},
}


def main(argv: list[str]) -> int:
    wanted = argv[1:] or list(CONFIGS)
    unknown = [w for w in wanted if w not in CONFIGS]
    if unknown:
        print(f"unknown configuration(s): {unknown}; known: {list(CONFIGS)}")
        return 2
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    summary = []
    for name in wanted:
        overrides = CONFIGS[name]
        out = OUT_ROOT / name
        # `warn` rather than `error`: a config that fails validation is a *result*
        # here, and a mesh that cannot be looked at answers nothing.
        cfg = default_config().with_overrides(
            OUTPUT_VALIDATION_MODE="warn",
            PIPELINE_COMPONENT_FAILURE="continue",
            **overrides,
        )
        print("=" * 70, flush=True)
        print(f"[{name}] {overrides or 'no overrides (baseline)'}", flush=True)
        print("=" * 70, flush=True)
        t0 = time.time()
        try:
            run_pipeline(GRAPH, str(out), cfg=cfg, write_outputs=True,
                         interactive=False)
            status = "ok"
            err = None
        except Exception as exc:  # a failing config is data, not a reason to stop
            status = "failed"
            err = f"{type(exc).__name__}: {exc}"
            traceback.print_exc()
        dt = time.time() - t0
        print(f"[{name}] {status} in {dt / 60:.1f} min", flush=True)
        summary.append({"name": name, "overrides": overrides, "status": status,
                        "error": err, "seconds": round(dt, 1),
                        "output_dir": str(out)})
        (OUT_ROOT / "sweep_summary.json").write_text(
            json.dumps(summary, indent=2), encoding="utf-8")
    print("\n" + "=" * 70)
    for s in summary:
        print(f"  {s['name']:<18} {s['status']:<8} {s['seconds'] / 60:>5.1f} min"
              f"   {s['error'] or ''}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
