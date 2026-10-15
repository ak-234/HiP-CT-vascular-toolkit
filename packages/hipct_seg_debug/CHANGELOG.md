# Changelog

All notable changes to this project are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses
[semantic versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **Workflow 13, `jin-mgf` centreline end to end**, in the control panel's Workflows
  tab: `repair-mask --guarded` (with the session's painted edits), a pause to review
  it, `skeletonise-all --algorithms jin-mgf --per-tree --jin-engine sparse`,
  `simplify-skeleton --apply`, then `fair-centreline`. Chains now also receive the
  session's `--voxel-um`, and a placeholder the session cannot fill is omitted
  rather than passed empty. `docs/JIN_MGF_WORKFLOW.md` gives the command-line and GUI
  form of every stage.

- **Seeded mask repairs in the paint panel** (`edit/mask_tools.py`).
  - *Fill collapsed lumen* grows from a vessel's mask into the dark slit its
    segmentation missed, inside a capsule along seeds placed in a new `paint seeds`
    layer. It calibrates lumen and wall on that vessel alone, because a global
    brightness gate failed on LADAF-2021-17. It refuses leaks out of the capsule,
    merges into the other tree, unwanted joins and new tunnels.
  - *Split fused vessels* runs a seeded watershed on distance and raw intensity. The
    cut is safe under 26-connectivity, and a cut across a healthy lumen is refused
    unless forced.
  - Each run is one `Ctrl+Z`, and a `paint result` layer shows what was filled, cut
    and refused.
- **Label picker for the brush.** The brush takes each box's majority label or a
  pinned one, every label is coloured, and *protect other labels* is on by default.
  *Relabel painted voxels to their tree* repairs stores painted with the old fixed
  label.
- **`--goto-edge ID`** (and `--goto-edge-at F`), plus a *go to edge* box in the 3D
  layers panel: go straight to an edge id from a report or the status line.

- **`fair-centreline`** (`edit/centreline_fair.py`) smooths a path-search skeleton
  vessel by vessel through its junctions, and keeps it inside the segmentation.
  - Degree-2 nodes are rejoined, and each segment is labelled with a `vessel_id`.
  - A branch's first stretch inside its parent is not fitted, so the branch leaves
    the junction without a kink.
  - On both LADAF-2021-17 `jin-mgf` trees, the median step-to-step turn went from 45°
    to 1.0° and the median junction kink from 20° to 5.1°, in under 2 minutes.
  - `jin-mgf[fair=true]` applies it inside `skeletonise-all`.
- **Geodesic and curve-based skeletonisers, benchmarked against TEASAR.** See
  `docs/SKELETON_BENCHMARK.md`.
  - **`geodesic-scoop`** (`edit/geodesic_scoop.py`) is the wave-front family, which
    was previously unavailable. A vertex is the centroid of each component of each
    geodesic shell from an inlet root. It keeps loops.
  - **`jin-mgf`** (`edit/jin_mgf.py`) is Jin MCP with a Dey & Sun medial-geodesic
    medialness on sheet voxels and cross-section-filling marking. It is driven
    through new hooks on `jin_mcp.extract`, whose defaults are unchanged.
  - **`teasar-readme`** is the kimimaro README pipeline (`postprocess`,
    `cross_sectional_area`); kimimaro's area is carried as a point attribute.
  - **`skeletonise-all`** accepts parameterised candidates
    (`name[key=value;...]`) and a `--summary-json` report.
  - **`edit/skeleton_synthetic_benchmark.py`** scores extraction against the
    perimeter-conserving phantoms in `edit/skeleton_phantoms.py`.
- **Whole-heart runs.**
  - **`jin_sparse`**: Jin MCP and `jin-mgf` on lumen voxels only.
    - Output is identical to the dense engine when `incremental=False`.
    - Differential (incremental) rounds are the default.
    - `--jin-engine` and `--jin-allow-loops` are new flags.
  - **`geodesic-scoop` is sparse too.** It uses `edt` and `cc3d`, and takes 31 s
    where it took 519 s on a whole LADAF-2024-28 tree.
  - **`skeletonise-all --per-tree --workers N`** runs every (candidate, tree) pair
    in its own process.
  - **The split and scoring context are reused.** `skeletonise-all` computes the
    component split and each tree's scoring context once rather than per candidate.
  - **`summary.json` is rewritten after every candidate.**
  - **`jin-mgf` on the sparse engine is about 2× faster, with identical output.** The
    Hessian classifier's Gaussian and the MGF ridge weight now run on 64³ tiles that
    hold lumen, across 8 threads, instead of over the whole tree box or one box per
    sheet component. On a whole LADAF-2024-28 tree, `exact` mode went from 161 s to
    75 s; both arrays are bit-identical to the old code on two trees.

- **`repair-mask --guarded`: fill hollow lumens and voids without welding vessels or
  filling loops** (`edit/mask_repair.py`, `docs/SKELETONISATION.md` §6.3).
  - **What it proposes:** 2D hole fills along all three axes, one tree at a time,
    repeated for any pockets they seal. A closing is opt-in (`--close-radius`, default
    0): it added about 3% volume (a 0.1-voxel skin) for no topological gain.
  - **What each region must pass:**
    - a local Betti check, Δb0 = 0 and Δb1 ≤ 0;
    - closing patches and pinholes: Δb1 = 0, with a detour bound on closing patches;
    - slice fills: they must be enclosed in 3D, capped on both sides by mask, a
      sleeve around the vessel's own core, or a hollow core (alignment ≥ 0.35 to the
      normals of the slices it is a hole in, and band ≤ 1.0: a ring no thicker than
      itself). On 246 phantom regions this fills no loop or hairpin and 64 of 78
      cores; the previous elongation rule filled 4 tight loops, and a closing
      allowed to remove loops closed 15. The capped and sleeve tests admit the
      unlabelled-wall "donut" sleeves inside large vessels; on LADAF-2021-17, 1% of
      blocks held 55% of the remaining loops this way, up to 214 in one 64³ block.
  - **Inputs and outputs:** it keeps labels and `Materials`, writes an `.am`, a
    per-region `--report` and an `--audit-dir` gallery, and needs `--raw` and
    `--voxel-um`.
  - **Measured on LADAF-2021-17:**
    - on crops, a hollow vessel went from 126 tips to 10;
    - on the whole lattice (exact Betti via `mask_repair.betti_volume`), b0 stays 1 per
      tree, b1 goes from 1,949 / 806 to 1,781 / 699 and b2 from 481 / 196 to 297 / 29.
  - **`skeletonise-all --mask-repair`** runs the same repair before skeletonising an
    ROI.
- **`skeletonise-all --root-xyz X Y Z` and `--root-graph PATH`: extraction roots.**
  - **What they do:** Jin (`jin-mcp`, `jin-mgf`) and `geodesic-scoop` start at a given
    inlet instead of the widest voxel.
  - **Why:** on LADAF-2021-17, the default root sat inside the flared left inlet and
    traced back out to the cut's rim, a false Y with tips 1.8 and 2.2 mm from the
    inlet.
  - **How roots are placed:** `--root-graph` takes each tree's widest free end from a
    reference graph, such as a refined DFS centreline. Roots are world positions,
    snapped onto each tree's lumen through the decoded frame.
- **`connect --wavefront`: tensor-guided wavefront reconnection for large gaps.**
  A sibling of the geodesic connector (`edit/reconnect/wavefront/`) that reuses its
  corridor, calibration, classification, gates, selector, apply and audit, and
  changes what happens before and inside the search. It **refines the whole
  centreline first** (`centreline_refine.refine` + radius remeasurement,
  `--refine-method`, `--refine-workers`) and profiles each free end against the
  segmentation — tangent, ellipse major axis, flatness — because the two-point
  tangent every proposer coned on was 37–50° off on a jittered skeleton and is
  under 1° after refinement. It prices the corridor with an **image intensity
  tensor** (structure tensor for orientation, Hessian for planarity, both gated by
  the calibrated lumen likelihood) so a step along the vessel is cheap, across the
  ribbon dearer and through the collapsed wall dearest, capped at what the lattice
  resolves (`--anisotropy-ratio`, `--stencil 26|98`). It searches by **wave
  propagation** (a numba Dijkstra over the padded corridor with Riemannian edge
  costs; optional `agd` HamiltonFastMarching backend via `--engine agd` and the
  new `eikonal` extra): dual fronts meeting at a saddle, falling back to a keypoint
  chain with a ±15° look-ahead cone sized by the ellipse major axis
  (`--keypoint-step`, `--lookahead-cone-deg`), which records forced bridges across
  dropouts and sends them to review. Proposal reach is 40 radii
  (`--reach-radii`); `--explore-open-ends` chains outward from free ends nothing was
  proposed for. New gates refuse routes that cut across the local orientation or
  through collapsed walls (`alignment`, `normal_crossing` in the evidence).
  Provenance code `ReconnectionOrigin = 5`. `docs/WAVEFRONT_RECONNECTION.md`.
- **Shared plumbing for the two mask-and-graph connectors.** `endpoints.propose`,
  `tjunction.propose`, `geodesic.classify.associate` and `geodesic.lobes.propose`
  take `tangents=` to override the two-point end tangent; `geodesic.corridor.bounds`
  / `for_candidate` take `max_voxels=`; `geodesic.apply_plan` / `apply_one` / `weld`
  take `origin=`. All default to the previous behaviour. `cmd_connect_geodesic`'s
  session setup and write-out are factored into `_reconnect_session` /
  `_finish_reconnect` and shared with `cmd_connect_wavefront`.
- **GUI: connector runs are child processes, and their review files reach the
  Reconnect tab.** `runner.mode_for` now sends `connect --geodesic` and
  `connect --wavefront` to a subprocess like `--dpc` (both index every component
  and search a corridor per candidate; Stop must work), and the Commands tab
  records `review-json` / `decisions-json` as job outputs so a finished run loads
  its work list into the Reconnect tab as the Workflows path already did.
- **Dark-lumen greyscale phantoms for tests.** `conftest_geodesic.greyscale`,
  `ribbon_gap` (a ribbon whose mask is missing but whose image keeps a faint trace —
  the pinched slit) and `FakeStack` (the `TiffStack` window-read surface over a
  segmentation-grid image), so raw-image behaviour is now exercised end to end.

- **Reformat: anti-alias pre-filter when a plane's pitch is coarser than the voxel.**
  `radius`/`fixed` mode on a large vessel (over 16 voxels of radius at 4 radii on
  129 px) and `native_scale > 1` under-sample, and fine texture aliased into a beat
  pattern. The raw block is now low-passed with a Gaussian of
  `sqrt(f² − 1) / 2` voxels (`reformat.anti_alias_sigma`, zero at one voxel per
  pixel so `native` stacks are byte-identical to before) before interpolation. The
  block pad grows by the Gaussian's reach, the block planner ends a run wherever the
  sigma changes rung, and the chunking test is extended to under-sampling pitches.
  Reported in the geometry (`under-sampling at N planes … low-passed`) and in the
  stats (`anti-alias: N plane(s) pre-filtered`); `build(anti_alias=False)` and a panel
  checkbox turn it off for comparison. `docs/REFORMAT.md` §2.3 and §5.1.
- **Reformat: optional post-sharpen.** `build(sharpen=, sharpen_sigma_px=)` — an
  unsharp mask on the sampled planes, off by default, with a `sharpened: …` note so an
  export says so. Panel spinboxes `sharpen` / `sigma px`.
- **Reformat: the interpolating property is now asserted.** `map_coordinates` was
  already prefiltering (the cubic is an interpolant, not the smoothing "Bourke" form);
  `prefilter=True` is now passed explicitly and
  `test_integer_grid_samples_reproduce_the_raw_values` /
  `test_a_step_edge_is_not_smoothed_by_the_spline` prove it against the smoothing
  form as a negative control.

### Changed

- **`reconnect.segmentation.components` labels with `cc3d`.** The components are
  the same as before; 367 s against about 2,500 s on the full LADAF-2021-17 lattice.
- **Reformat refuses an anisotropic raw voxel.** `reformat.isotropic_voxel_um` checks
  the three components of `WorldFrame.raw_voxel` agree to 0.1% and raises naming them;
  `build`, the panel's default step and Match voxel all go through it, where they used
  to take `np.min` silently.
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

- **Painting a two-tree mask wrote every stroke as 1.** Corrections to `Right_Tree`
  became `Left_Tree`, and label 2 had no colour in any slice-browser layer, so the
  right coronary was invisible there.
- **`mask-export` dropped the `Materials` block**, and so did
  `connect --geodesic --out-seg`. Labels survived but their names did not, and
  everything that splits by material fell back to connectivity, which at 32 µm joins
  the two coronaries. Both now copy the block, and geodesic bridges take the label of
  the tree they extend.

- **`refine-centreline --workers N` survives a section worker dying.** One spawn
  worker failing at start-up (`OSError: [WinError 6]` on Windows) broke the pool and
  ended the run with `BrokenProcessPool`. On LADAF-2021-17 this happened in iteration 7
  of 8, after six hours. Finished segments are now kept, and the rest are resubmitted
  to a fresh pool up to twice. The section phase also submits its longest segments
  first.
- **`skeletonise` and `skeletonise-all` stamp their output's voxel size.** Under a
  corrected `--voxel-um` they wrote graphs in true micrometres but without
  `HiPCTVoxelSizeUm`. The viewer read them as file units and applied the correction
  again, so on LADAF-2021-17 only 1.8% of a `jin-mgf` graph sat inside its own mask.
  Graphs are now stamped exactly when the frame was corrected.
- **Viewer start-up on a whole-heart graph.** The mask-containment check decoded each
  slice once per orientation variant (five times). It now decodes each slice once.
  The "segmentation within raw stack" check allowed only 1e-6 µm, so it failed on a
  float32 bounding-box origin of -1.9e-6 µm; it now allows a thousandth of a voxel.
- **`skeletonise-all --workers N` no longer loses every candidate to one dead worker.**
  Four TEASAR jobs on the LADAF-2021-17 tree boxes exhausted memory. The killed worker
  broke the pool, so every job still in it failed. The first failed candidate then
  raised out of the command, and the finished `geodesic-scoop` tree was discarded. Jobs
  from a broken pool are now rerun on half as many workers, down to one. A candidate
  that still fails is recorded as `failed` in `summary.json`, and the others are
  written.

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
