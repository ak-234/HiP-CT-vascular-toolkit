# Radius-aware coronary mesh-sensitivity study

The study is resumable and never overwrites the validated SIP or authoritative
CFX seed. Every generated artifact is stored below `mesh_sensitivity_output`
under a stable case ID (`bl_adaptive_4layers`, `global_l1`, `adaptive_l4`, etc.).

## First run: approve POIs

```powershell
.\run_mesh_sensitivity_study.ps1 `
  -Config .\mesh_sensitivity_study.json `
  -Cores 8 `
  -Resume
```

The first run stops before meshing. Open
`mesh_sensitivity_output\poi\poi_candidates.csv`, inspect the matching VTK in
Amira/ParaView, and enter `yes` or `no` in every `approved` cell. Candidates are
generated from the exact STL-cropped Amira graph, not the raw graph or a
Simpleware centreline.

The manifest's `paths.stl` must be the exact surface persisted inside the
validated SIP (normally the STL produced by
`prepare_simpleware_surface_stl.py`). `paths.pre_import_stl` records the
original source STL for provenance only. Simpleware may alter the imported
surface slightly while persisting it; using that earlier STL for cropping or
validation can therefore produce a false surface-mismatch error.

If local PowerShell policy blocks scripts, enable it only for the current
terminal first:

```powershell
Set-ExecutionPolicy -Scope Process Bypass
```

## CFX-Pre mesh-remap recording

Before the `cfx` stage, create `cfx_remesh_session_template.pre` by following
[cfx_remesh_session_template_README.md](cfx_remesh_session_template_README.md).
This records only the supported CFX-Pre reload/name-remap/write-DEF operation.
The launcher audits the binary seed independently and rejects any change to its
SHA-256, allometric inlet flow, Quemada/laminar model, iteration/timescale
controls, or RMS residual target.

## Stages

Use `-Stage poi`, `plan`, `mesh`, `mesh-quality`, `cfx`, `extract`, `analyse`,
or `validate` to
run one stage. Omitting `-Stage` runs all currently available work. `-Resume`
skips only artifacts with a completed validation record; a `.res` file alone is
not treated as success.

The workflow first runs the global/adaptive mesh-convergence matrix with five
fixed prism layers. The separate 4/6/8-layer adaptive boundary-layer cases run
later with `-Stage boundary-layer`; they no longer gate the main matrix.

For tightly spaced coronary branches, the optional Simpleware **Additional mesh
quality improvement** pass is disabled. Its off-surface node motion can merge
nearby but topologically separate vessel walls and cause a non-manifold mesh.
The normal +FE quality operations remain enabled, and every exported mesh must
still pass the CFX import and positive-volume validation gates. Every completed
solve must also pass residual,
boundary-presence, mass-imbalance (0.1%), outward-opening-flow, and seed-contract
checks.

Each newly generated calibration trial now exports native Simpleware quality
statistics before the transient `Mesh` result is discarded. The selected trial
promotes `mesh_quality.json` and `mesh_quality.csv` beside `case.sip`,
`mesh.msh`, and `mesh_stats.json`. The files separate core-volume,
boundary-layer-volume, and surface-face statistics and include element/sample
counts, sum, mean, minimum, maximum, Simpleware's configured threshold, and the
count and percentage past that threshold. The summary records the minimum
Jacobian and explicitly flags a negative Jacobian. Unsupported metric/element
combinations are labelled `partial` or `unsupported` without losing the valid
statistics from the same mesh.

For SIPs generated before these exports were added, run:

```powershell
.\run_mesh_sensitivity_study.ps1 `
  -Config .\mesh_sensitivity_study.json `
  -Stage mesh-quality `
  -Resume
```

This does not remesh or save the SIP. It runs X-2025.06's quality inspector and
writes per-case `mesh_quality_inspection.json`/`.csv`, plus combined tables in
`reports\simpleware_mesh_quality`. Post-save inspection provides configured
error/warning thresholds and offending-cell counts; full distribution
minima/means/maxima are available only for meshes generated with the updated
console script.

`boundary_layer.growth_ratio` is the conventional ratio between consecutive
prism-layer widths. The launcher converts it to Simpleware's bounded
thinnest-to-thickest `RatioSlicing` width ratio for each requested layer count;
the Simpleware API does not accept a growth factor greater than one directly.
Finite clipping-plane caps are not native lumen-part surface regions in
Simpleware, so they are inherently excluded from prism-layer spawning. The
console runner verifies that every named inlet/outlet/opening cap is a clipping
ROI and that boundary layers remain enabled on the native `External` wall; it
does not pass clipping ROI names to `SetBoundaryLayerRegionByName`.

Post-processing writes area-weighted wall and volume-weighted pressure metrics
for approved POIs, each retained segment, every observed Strahler order, and
eight fixed logarithmic radius bins. The report is written to
`mesh_sensitivity_output\reports\mesh_independence.json`; failure of the finest
pair produces a recommendation for an approximately 32-million-element level.
