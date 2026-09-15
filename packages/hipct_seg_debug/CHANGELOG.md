# Changelog

All notable changes to this project are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses
[semantic versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed

- **Repository moved to a `src/` layout.** The package now lives at
  `src/hipct_seg_debug/` instead of being the checkout itself. Import paths are
  unchanged — `import hipct_seg_debug`, `python -m hipct_seg_debug` and
  `python -m hipct_seg_debug.edit` all behave as before — but the checkout is no
  longer importable without installing it. Run `pip install -e .` once; the old
  trick of running from the parent directory with the checkout named
  `hipct_seg_debug` no longer works, and is no longer needed.
- **Tests moved out of the package** to a top-level `tests/`. They are no longer
  shipped in the wheel. Run them with `python -m pytest`; the old
  `python -m pytest --pyargs hipct_seg_debug.edit.tests` form is gone.
- **Long-form documentation moved to `docs/`** — `CLI.md`, `REFORMAT.md`,
  `SKELETONISATION.md`, `GEODESIC_RECONNECTION.md`, and the editor guide
  (previously `edit/README.md`, now `docs/EDITOR.md`). `README.md` stays at the root.
- **`--cache` default changed** from `<package>/cache` to `$HIPCT_CACHE`, falling
  back to `cache/` under the working directory. The old default wrote tens of
  megabytes of decoded slices into the package directory, which lands inside
  `site-packages` — often unwritable — for a non-editable install. Running from a
  checkout, as before, still produces `./cache`.

### Removed

- **No dataset paths are built into the package any more.** `main.DEFAULTS` held four
  absolute paths to one machine's LADAF-2024-28 files, and the editor's `DEFAULT_SEG` /
  `DEFAULT_GRAPH` held two more, so a fresh install with no arguments loaded somebody
  else's scan — and on the authoring machine, found it. Each input now falls back to an
  environment variable instead:

  | variable | flag |
  |---|---|
  | `HIPCT_RAW` | `--raw` |
  | `HIPCT_GRAPH` | `--graph`, and the editor's reference graph |
  | `HIPCT_SEG` | `--seg` |
  | `HIPCT_SURFACE` | `--surface` |

  With neither a flag nor a variable, both entry points name the inputs they are missing
  rather than guessing. The `F:\` fallback in `edit/_deps.py`'s search for `coronary_sdf`
  is gone too; only locations relative to the checkout are searched now, plus
  `HIPCT_CORONARY_SDF`.
- Machine-local paths removed from every docstring, error message, test fixture and
  document. The optional real-data tests read `HIPCT_GRAPH` / `HIPCT_SEG` /
  `HIPCT_GRAPH_ALT` (see `tests/realdata.py`) and skip when those are unset, instead of
  reaching for a hardcoded drive.
- The "run from `F:\`" / `PYTHONPATH` instructions throughout the docs, which existed
  only because of the old flat layout. `pip install -e .` replaces all of them.

### Added

- **The viewer starts without a dataset.** `python -m hipct_seg_debug` with no
  `--raw` / `--graph` / `--seg` now opens an empty 3D window instead of exiting; load a
  dataset from the control dock's Data tab, the same panel that already swaps datasets
  mid-session. `--validate-only`, `--selftest` and the `--goto-*` flags still require
  inputs and say which are missing, because none of them has anywhere to ask.

  Internally this is a new `MissingInputs(InputError)`, so "you named nothing" and "what
  you named will not load" can be told apart; the latter is still fatal everywhere. No
  new tolerance was needed in the window itself -- `ViewerApp`, `Picker3D._populate` and
  every panel already handled the null-session state they reach between datasets.

### Fixed

- A session with no `--surface` raised `TypeError: argument should be a str or an
  os.PathLike object ... not 'NoneType'` from `Path(None)`, after the raw stack, graph
  and lattice had all loaded — which the Data tab reported as a bare "could not load".
  The surface is optional, so it can be *absent* as well as missing from disk; only the
  latter was handled, because the flag used to carry a built-in default that always
  named a real file.

- The Data and Workflows tabs fell back to the bare working directory for their cache
  when no session was loaded, so a session started without a dataset wrote
  `gui_recent.json` (and run directories) into whatever directory it was launched from
  rather than into `cache/`. Both now use the same default a session would.

- `LICENSE` (MIT), and the licence, classifiers, keywords and project URLs that a
  package needs to be published.
- GitHub Actions CI: lint, the test suite under `xvfb`, and a `python -m build`
  plus `twine check` of the resulting distributions.
- `ruff` configuration and a `[dev]` extra, so lint and release tooling install
  with `pip install -e ".[dev]"`.

### Removed

- The committed Amira spatial graph `LADAF_2021_17_left_tree_cropped_strahler_min_3`
  is no longer tracked, matching the existing policy that no dataset lives in the
  source tree. `.gitignore` now covers extensionless `LADAF_*` exports.

## [1.0.0]

- First tagged version: the viewer, the editing/repair CLI, the four reconnection
  proposers, reformatting, and the skeletonisation comparison tooling.
