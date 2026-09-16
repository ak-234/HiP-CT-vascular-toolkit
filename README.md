# HiP-CT Vascular Toolkit

Three Python packages for turning HiP-CT vascular imaging into analysable
centrelines and watertight lumen surfaces. They were developed together and
depend on each other, so they live in one repository.

| Package | Import name | What it does |
|---|---|---|
| [`packages/hipct_seg_debug`](packages/hipct_seg_debug) | `hipct_seg_debug` | Interactive viewer and editing/repair CLI for spatial graphs and segmentations |
| [`packages/coronary_sdf`](packages/coronary_sdf) | `coronary_sdf` | Signed-distance-field lumen surface reconstruction, meshing and CFD prep |
| [`packages/skeleton_analysis`](packages/skeleton_analysis) | `skeleton_analysis` | Skeletonisation fixes, ordering and topological metrics (Strahler, Murray) |

`hipct_seg_debug` is the front end most work starts from; it calls into the other
two for surface generation and for the super metric.

## Install

Python **3.12**. One command creates the environment and installs all three
packages editable; a second confirms the result.

```bash
git clone https://github.com/ak-234/HiP-CT-vascular-toolkit.git
cd HiP-CT-vascular-toolkit

conda env create -f environment.yml     # creates `hipct` and installs everything
conda activate hipct
python tools/check_environment.py       # every line should read PASS
```

Into an environment you already have (conda or venv, Python 3.12):

```bash
python -m pip install -r requirements.txt
python tools/check_environment.py
```

[`requirements.txt`](requirements.txt) installs the three packages in a single
pip resolution, so the version pins in
[`hipct_seg_debug`'s `pyproject.toml`](packages/hipct_seg_debug/pyproject.toml)
apply to all of them from the start. Each package is still independently
installable (`pip install -e packages/<name>`) — nothing forces you to take all
three — but `hipct_seg_debug` will raise a descriptive `ImportError` from its
edit and SDF commands until `coronary_sdf` and `skeleton_analysis` are on the
path. None of the three is published to PyPI, which is why they are not listed
in each other's `dependencies`.

### If the checker complains

`tools/check_environment.py` prints one line per check with the command that
fixes it. The failures worth knowing about in advance:

- **`pin numpy … not installed`** (and friends) right after
  `conda env create` — your per-user `site-packages` (`%APPDATA%\Python\...`
  on Windows, `~/.local` on Linux) already had those packages, and conda's
  create-time pip counted them as satisfied. The activated env hides the user
  site, so they are genuinely missing. Run
  `python -m pip install -r requirements.txt` once, in the activated env.
- **`user-site … is on sys.path`** — Python is importing from that per-user
  directory. Anything installed there shadows the environment, including an
  old editable install of these packages pointing at another checkout.
  `environment.yml` sets `PYTHONNOUSERSITE=1` on the env to prevent this; for
  an env you made yourself run
  `conda env config vars set PYTHONNOUSERSITE=1 -n <env>` and re-activate.
- **`editable from <elsewhere>, but this checkout is …`** — the package on the
  import path comes from a different clone. Uninstall it and reinstall from
  here with `python -m pip install -r requirements.txt`.

> **`pip` and `python` can be different environments.** If `pip install` reports
> a Python version you did not expect, use `python -m pip install ...` so the
> install lands in the interpreter you will actually run.

> **Do not let pip upgrade numpy.** `hipct_seg_debug`'s pins are load-bearing:
> numpy 2.x breaks numba (the RLE decoder) and contourpy/matplotlib. See that
> package's README. The checker fails on numpy 2.x for this reason.

### Console scripts

```bash
hipct-seg-debug --help        # the viewer
hipct-edit --help             # the editing / repair CLI
coronary-sdf --help           # the SDF surface pipeline
skeleton-analysis --help      # ordering and metrics
```

## Pointing the tools at data

No dataset paths are baked into this repository. Imaging data is large, is not
redistributable, and lives wherever you put it, so the tools take paths from
arguments or from the environment:

| Variable | Used by | Meaning |
|---|---|---|
| `HIPCT_GRAPH` | research scripts, `Python_port_test.py` | Input Amira SpatialGraph (`.am` / `.xml`) |
| `HIPCT_SEG` | research scripts, `Python_port_test.py` | Matching segmentation lattice (`.am`) |
| `HIPCT_OUT` | research scripts | Output directory (default `./out`) |
| `CORONARY_SDF_INPUT` | `coronary_sdf` | Input spatial graph |
| `CORONARY_SDF_OUTPUT_DIR` | `coronary_sdf` | Output directory (default `./coronary_sdf_out`) |
| `CORONARY_SDF_STL` | `coronary_sdf` | Capped lumen STL used as the crop authority |
| `CORONARY_SDF_MSH` | `coronary_sdf` | Volume mesh for flow-fraction work |
| `HIPCT_CORONARY_SDF` | `hipct_seg_debug` | Only needed when `coronary_sdf` is *not* installed |

Anything unset fails immediately, naming the variable to set, rather than part
way through a run.

## Repository layout

```
packages/
├── hipct_seg_debug/     src/ tests/ docs/       MIT
├── coronary_sdf/        src/ tests/ docs/       licence pending
│   ├── research_scripts/    one-off diagnostics, not part of the wheel
│   └── native/cgal/         optional pybind11 + CGAL extension, opt-in
└── skeleton_analysis/   src/ tests/             MIT
```

Generated data — meshes, logs, solver output, image stacks — is excluded by
`.gitignore` and should stay outside the repository.

## Syncing from the pre-monorepo checkouts

These packages were assembled from separate checkouts on `F:\`. If you still
work in those, `tools/sync_from_legacy.py` carries changes across — one way
only, legacy into this repository:

```bash
python tools/sync_from_legacy.py                 # dry run: what would change
python tools/sync_from_legacy.py --apply
python tools/sync_from_legacy.py --apply --only hipct_seg_debug
```

It handles the `coronary_sdf` layout change (its flat root became `src/`,
`tests/` and `research_scripts/`) and finds a checkout that has been renamed,
e.g. `coronary_sdf_local`. Override any location with `LEGACY_CORONARY_SDF`,
`LEGACY_HIPCT_SEG_DEBUG` or `LEGACY_SKELETON_ANALYSIS`.

**44 files are protected and never overwritten.** They were edited during
publication — dataset paths replaced by environment variables, imports fixed
for the new layout, docs rewritten — and their legacy copies still contain the
originals, so copying them back would silently reintroduce paths like
`F:/Edo_latest_spatial_graph/...` into code that ships in a wheel. The script
reports them as blocked; `--diff` shows what differs and `--force-protected`
overrides once you have read that. After applying, it re-runs the absolute-path
scan and exits non-zero if anything leaked.

This is one-directional, so the two copies drift. Working directly in this
repository avoids that entirely.

## Tests

```bash
python tools/check_environment.py           # the environment itself
pytest tools/tests -q                       # the checker and the legacy sync script
pytest packages/skeleton_analysis/tests -q
pytest packages/coronary_sdf/tests -q
pytest packages/hipct_seg_debug/tests -q    # needs a display; use xvfb-run on CI
ruff check packages/hipct_seg_debug/src packages/hipct_seg_debug/tests   # the only lint-clean package; CI gates on this
```

None of these need a dataset. `hipct_seg_debug` has a `--runslow` flag that opts
into tests requiring real data.

## Licensing

`hipct_seg_debug` and `skeleton_analysis` are MIT (see the `LICENSE` file in
each). **`coronary_sdf` carries no licence yet**, and there is no repository-wide
`LICENSE`; until one is added, default copyright applies and the code is not
reusable by others. Settle this before making the repository public.

`skeleton_analysis` is a Python port of an earlier MATLAB pipeline from the UCL
HiP-CT group; `packages/skeleton_analysis/PORTING.md` documents the derivation
module by module.

Datasets referenced in documentation use anonymised HiP-CT donor identifiers
(`LADAF_…`). No imaging data is included in this repository.
