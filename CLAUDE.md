# HiP-CT Vascular Toolkit — working notes for agents

Monorepo holding three Python packages that depend on each other. Created
2026-09-15 by consolidating three separate checkouts on `F:\`. Those checkouts
still exist but are **retired** — this repository is the source of truth.

## Layout

```
packages/hipct_seg_debug/     viewer + editing/repair CLI   src/ tests/ docs/
packages/coronary_sdf/        SDF lumen surfacing + CFD     src/ tests/ docs/
                              research_scripts/  one-off diagnostics, not in the wheel
                              native/cgal/       optional pybind11+CGAL ext, opt-in
packages/skeleton_analysis/   ordering + topological metrics  src/ tests/
tools/sync_from_legacy.py     one-way import from the retired checkouts
```

Python **3.12** (the intersection of the three packages' requirements).

## Install

Order matters — `hipct_seg_debug`'s edit/SDF half imports the other two:

```bash
python -m pip install -e packages/skeleton_analysis
python -m pip install -e packages/coronary_sdf
python -m pip install -e "packages/hipct_seg_debug[dev]"
```

None of the three is on PyPI, which is why they are not in each other's
`dependencies`. With all three installed, imports resolve normally and
`HIPCT_CORONARY_SDF` is not needed.

## No dataset paths are hardcoded

This was the main cleanup when publishing, and it is easy to undo by accident.
Never write a drive-letter path into shipped code. Paths come from arguments or
these variables, and anything unset fails immediately naming the variable:

| Variable | Meaning |
|---|---|
| `HIPCT_GRAPH` / `HIPCT_SEG` / `HIPCT_OUT` | graph, segmentation, output dir (research scripts, `Python_port_test.py`) |
| `CORONARY_SDF_INPUT` / `CORONARY_SDF_OUTPUT_DIR` | coronary_sdf input graph and output dir |
| `CORONARY_SDF_STL` / `CORONARY_SDF_MSH` | crop-authority STL, volume mesh |
| `HIPCT_CORONARY_SDF` | only for a `coronary_sdf` outside this repo |

`packages/coronary_sdf/research_scripts/_paths.py` is the helper; follow its
pattern (report what was looked for and what to set) rather than raising a bare
`KeyError`. `hipct_seg_debug/edit/_deps.py` does the same for the sibling
packages.

`D:\data\...` strings in docs and `D:\a b\c.am` in `test_cliform.py` are
deliberate examples, not leaks.

## Things that will bite you

- **Generated data must stay out.** The predecessor `coronary_sdf` repo reached
  a 15.6 GB `.git` with single blobs up to 2.98 GB and could never be pushed.
  The root `.gitignore` excludes output trees by directory name, not just by
  extension. Check `git status` before committing after a pipeline run.
- **`coronary_sdf` was a flat repo.** Its root *was* the package. Anything
  written against the old layout (`F:\coronary_sdf\config.py`) now lives at
  `packages/coronary_sdf/src/coronary_sdf/config.py`. Cross-module imports use
  the absolute `from coronary_sdf.x import y` form; keep it that way.
- **`skeleton_analysis` had two copies** and pip pointed at the stale one. Only
  one exists here. If you see `Skeleton_analysis-main` referenced anywhere, it
  is third-party MATLAB with no licence grant and must not be vendored in.
- **Lint is uneven.** Only `hipct_seg_debug` is clean and CI gates on it alone.
  `coronary_sdf` (~800 violations) and `skeleton_analysis` (~270) predate
  linting; both carry a `[tool.ruff]` target. Do not bulk-autofix them in a
  change about something else.
- **`coronary_sdf` has no LICENCE**, and there is no root one. Do not add or
  imply a licence for it without being asked.

## Tests

```bash
pytest packages/skeleton_analysis/tests -q     # 103
pytest packages/coronary_sdf/tests -q          # 140 passed, 13 skipped
pytest packages/hipct_seg_debug/tests -q       # ~1493; needs a display (xvfb on CI)
ruff check packages/hipct_seg_debug/src packages/hipct_seg_debug/tests
```

None need a dataset. `hipct_seg_debug` has `--runslow` for tests that do;
`coronary_sdf` gates equivalents behind `CORONARY_SDF_RUN_SLOW=1`.

**Known pre-existing failure**, unrelated to any current work:
`test_radius_circles.py::test_repeated_junction_records_keep_their_edge_local_planes`.
It failed identically in the pre-monorepo checkout. Do not treat it as a
regression you caused.

## Syncing from the retired checkouts

If work still happens in `F:\hipct_seg_debug` etc., `tools/sync_from_legacy.py`
imports it (dry run by default, `--apply` to write). **44 files are protected**
because publication edited them; the legacy copies still hold hardcoded `F:\`
paths, so copying them back would reintroduce those into shipped code. The
script blocks them and re-runs the path scan afterwards, exiting non-zero on a
leak. See the README section for details.
