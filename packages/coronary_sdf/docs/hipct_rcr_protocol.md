# LADAF_2024_28 coronary RCR protocol

The `coronary-rcr` pipeline separates measured geometry, hydraulic assumptions,
calibration and CFD export. Source datasets are read-only. Each command requires
an explicit new/empty output directory. Relative input paths resolve from the
JSON configuration file. Python module invocation works without reinstalling the
editable package; reinstall to register the `coronary-rcr` console command.

See the [9 October LADAF validation record](hipct_rcr_ladaf_validation.md) for
the resolved acquisition scale, inlet mappings and selected real measurements.
Three proximal hydraulic spans currently fail the support gate.

## Fixed baseline

| Quantity | Active value |
|---|---:|
| Segmented LV + RV myocardium volume | 87.260959 cm³ |
| Myocardial density | 1.05 g/cm³ |
| Myocardial mass | 91.62400695 g |
| Resting perfusion | 0.8 mL/min/g |
| Total coronary flow | 73.29920556 mL/min |
| Left/right allocation | 70% / 30% |
| Left flow | 8.551573982 × 10⁻⁷ m³/s |
| Right flow | 3.664960278 × 10⁻⁷ m³/s |
| Blood density | 1050 kg/m³ |
| Newtonian viscosity, explicit modelling assumption | 0.0035 Pa·s |
| Provisional arterial/distal pressure | 93 / 5 mmHg |
| Left inlet-to-distal resistance | 1.371950477 × 10¹⁰ Pa·s/m³ |
| Right inlet-to-distal resistance | 3.201217779 × 10¹⁰ Pa·s/m³ |

Conversion uses 133.322 Pa/mmHg and 6×10⁷ (mL/min)/(m³/s). These resistances
include the retained coronary vessels. They must not also be assigned in full
to every outlet. Radius correction does not rescale the accepted myocardial
mass or the target flows.

The donor is female, 59 years, 40 kg and 155 cm. There is no paired in-vivo dataset
or validated ex-vivo correction for this case. Keep segmentation mass active.
Patient-variable post-mortem population estimates from the
[proposed population study](https://pmc.ncbi.nlm.nih.gov/articles/PMC9587075/)
and allometric estimates are future sensitivity cases. Compare matching tissue
compartments: whole-heart mass is not interchangeable with segmented LV + RV
myocardium. Do not tune an ex-vivo correction merely to force agreement.

## Run the synthetic example

From the repository root, using the environment with all three toolkit packages:

```powershell
python -m coronary_sdf.rcr calibrate --config packages/coronary_sdf/examples/rcr/synthetic_case.json --out runs/rcr/example_calibration --methods diameter structured hipct legacy-giessen
python -m coronary_sdf.rcr compare --config packages/coronary_sdf/examples/rcr/synthetic_case.json --results runs/rcr/example_calibration/calibration.json --out runs/rcr/example_comparison
python -m coronary_sdf.rcr export-cfx --config packages/coronary_sdf/examples/rcr/synthetic_case.json --results runs/rcr/example_calibration/calibration.json --tree left --method hipct --out runs/rcr/example_cfx --dry-run
```

The example uses synthetic cylindrical networks and explicitly assumed
structured-tree and compliance parameters. It is not a LADAF calibration.
Without a mesh, the dry run writes raw CCL, Fortran sources and preflight status;
it cannot write mesh-resolved CCL, a usable `.def`, or validate a solver run.
Supply a real Fluent mesh and its region mapping to exercise CFX-Pre export.
Tests contain a *zone-table-only* fixture for text resolution; it is not a CFD mesh.

## Geometry preparation

Add this block to a copy of the example JSON, replacing all placeholders:

```json
{
  "geometry": {
    "graph": "path/to/original.attributegraph.am",
    "segmentation": "path/to/matching_labels.am",
    "roots": {"left": {"node": 0}, "right": {"node": 200}},
    "acquisition_voxel_um": null,
    "voxel_size_source": null,
    "segmentation_binning": null,
    "centreline_audit_accepted": false,
    "clinical_radius_m": 0.0005,
    "radius_settings": {"workers": 1, "fallback_policy": "rescale"}
  }
}
```

The illustrated node IDs are placeholders, not anatomical assignments.
First run `prepare --config case.json --out runs/rcr/audit --audit-only`.
Inspect `geometry_audit.json` and overlay centreline/segmentation in the existing
viewer. Confirm axes, origin, coordinates in micrometres, binning, inlet anatomy,
connectivity, placement within the vessel lumen and any previous correction stamp.
Every connected component requires exactly one explicit free-end root. A root edge
alone does not identify which endpoint is the inlet.

The local dataset was found at `F:\Edo_latest_spatial_graph` (different from the
originally typed directory). The original ASCII graph has 309 segments, 311 nodes,
two connected components, and no voxel-correction stamp. The matching resampled
lattice header gives approximately 65.980 µm spacing, consistent with a 2×32.99 µm
coordinate convention. Subsequent acquisition-metadata inspection resolved this
for the supplied dataset: `F:\32_99um_LADAF-2024-28_heart_overview_\metadata.json`
records scan `HEL_HA2100_16.495um_LADAF-2024-28_heart_overview_` and sample-pixel
size 0.016495 mm. Its crop extents, divided by two and rounded down, match the
actual TIFF stack: 4752 slices of 3079×3154 pixels. Thus the working image stack
is 32.99 µm and this segmentation is another 2× binned (65.98 µm). Use 32.99 µm
as the working acquisition-grid input and `[2,2,2]` segmentation binning; record
the native 16.495 µm acquisition and 2× stack downsampling in `voxel_size_source`.
The older toolkit prose recommending 32.04 µm is contradicted by these primary
records. A separate folder named `32.04um_artery_left_right` has different stack
dimensions and supplies no calibrated TIFF resolution tags; its name is not
evidence for rescaling this dataset. Do not rescale the accepted myocardial mass.
The existing toolkit refuses conflicting correction stamps and prevents applying
the same scale change twice. Complete the centreline audit before setting
`centreline_audit_accepted: true`. Lattice caches are written under the output
directory, never beside the source segmentation.

Then run `prepare --config case.json --out runs/rcr/prepared` on a small verified
subtree first. Use an isolated subtree graph with its matching root mapping and
the same segmentation. Only after review process both complete trees. Unsupported
hydraulic spans stop preparation after writing the corrected graph and audit;
they do not silently acquire a small positive radius.

Prefer `geometry.measurement_segment_ids: [67, 270, 307, 308]` for a selected
measurement audit **with the full graph still present**. This preserves neighbouring
branches for ownership and junction checks, avoiding boundary effects from an
isolated patch. It writes `measured_subset.am`, coverage and before/after statistics
for the selected scope, but deliberately emits no hydraulic `network.json`, even
if the selected spans pass. Unselected segments remain unmeasured.

The adapter uses the existing `radius-perimeter` parser and estimator with gate=0,
discretisation correction enabled, section filtering enabled, and bifurcation
tapers disabled. Its correction is `r = (P/2π)/1.0548 + 0.5 × sampling_spacing`.
Effective estimator defaults and source hashes are recorded. `MeanRadius` is
recomputed. Direct and owned-plane accepted samples are trusted; interpolation,
fallback and junction construction are not labelled as direct measurements.
Internal gaps use the existing bounded log-PCHIP interpolation. Short unsupported
junction ends may use that branch's own nearest trusted calibre, up to
`max(8 × segmentation_spacing, 4 × anchor_radius)`. No sibling radius or artificial
carina-tip radius is integrated as an anatomical constriction.

An explicit hydraulic override is possible only with a reason and one positive
radius per original segment sample:

```json
{"hydraulic_overrides": {"17": {"radius_um": [220, 215, 210], "reason": "Reviewed substitute; not directly measured"}}}
```

Put this inside `geometry`. Prefer resolving image support over overrides.
The original point-level source, rejection and resolution-mode fields are carried
into derived graph exports and `network.json`. Original source graph and full
`corrected.am` retain annotations; split graph IDs have mapping sidecars back to
source segment IDs. Derived hydraulic radii have a separate field. Surface radii
remain separate from hydraulic radii; surface reconstruction can make geometric
adjustments without changing the hydraulic network or its measurement evidence.

Preparation emits:

- `corrected.am`, `geometry_audit.json`, estimator log and `network.json`.
- `<tree>_hydraulic.am` and `<tree>_clinical.am` for hydraulic inspection.
- `<tree>_full_surface.am` and `<tree>_clinical_surface.am` for reconstruction,
  each with an anatomical mapping sidecar.

Automatic truncation follows each root-to-leaf path to its first sampled radius
below 500 µm, retaining at least one sample interval as an outlet stub. Actual
cut radius is recorded; this is not interpolation to an exact 500 µm isosurface.
Do not interpret the smallest surviving radius as the resolution limit. Establish
the reliable distal limit from segmentation, section quality and coverage. There
is no enforced 100 µm terminal threshold for the measured tree.
Loops are supported in hydraulics, but automatic clinical cropping requires a
tree. For loops provide explicit `retained_edge_ids` and `outlets` inside each
root entry. Independent loads are refused when clinical outlets share downstream
connections; a coupled multiport model would be needed.

`prepare --surfaces` invokes the existing surface CLI on the separate surface
profiles. Inspect its quality reports before meshing. Volumetric meshing and
anatomical opening/cap labelling require the established external meshing workflow;
this command does not invent a Fluent volume mesh. Regenerate meshes after
correction/cropping and keep their region labels tied to the sidecars. Do not
reuse the previously supplied STL/CFD results as if they were corrected geometry.

## Network contract and calibration

The example `synthetic_network.json` is the executable schema reference.
Each tree has a unique `id`, inlet `root`, edges with stable `id`, `u`, `v`,
strictly increasing `s_m` and positive `radius_m` samples, retained edge IDs,
clinical outlets (`id`, `node`, `radius_m`) and unresolved terminals
(`node`, `radius_m`). Prepared networks also contain `xyz_m`, anatomical attributes,
per-point measurement evidence and separate surface radii. Use metres throughout.
Outlet `area_m2` is optional for calibration; absent areas are explicitly circular
estimates. Replace them with measured mesh opening areas for the SCAPIS impedance
prescription, then recalibrate before export. Inlet velocity always uses the actual
imported CFX inlet area, not a graph-derived area.

Poiseuille resistance is integrated along the complete radius profile:

`R_segment = (8 μ/π) ∫ ds/r(s)^4`.

Integration is exact for a piecewise-linear radius profile. The sparse nodal
conservation solve handles internal loops and parallel edges. Terminal resistance
is represented as a shunt to the explicit distal pressure. Anatomical loops are
not duplicated into independent trees.

| Resistance method | Definition |
|---|---|
| `diameter` | Global clinical-outlet conductance weights proportional to d^2.27 |
| `legacy-giessen` | Recursive daughter-diameter allocation on the clinical tree |
| `structured` | Explicit binary distal tree using configured daughter ratios, L/r and terminal radius |
| `hipct` | Measured downstream connectivity/profile resistance plus unresolved terminal beds |

The legacy allocator reuses `flow_fractions._branch_diameter`: skip ten junction
contours, average the next ten clean contours, and use the minimum radius for a
branch too short for that window. The effective counts and helper source hash
are recorded. It does not rerun legacy radius smoothing on corrected geometry.

For the last two methods, initial terminal conductance is proportional to r^2.27;
`R_terminal = scale × R_reference × (r_reference/r)^2.27`. A bracketed scalar solve
calibrates a common positive scale separately for left and right, **including
retained vessel losses**, to the fixed flow target. Diameter weights prescribe
relative load conductances; actual flow fractions can differ because upstream
losses differ. Infeasible fixed-vessel resistance budgets raise errors, with no
clamping of negative residual resistance. Structured-tree resource/iteration
limits also raise errors rather than silently becoming termination rules.

Calibrate using the `network.json` from preparation:

```powershell
python -m coronary_sdf.rcr calibrate --config case.json --network runs/rcr/prepared/network.json --out runs/rcr/calibrated --methods diameter structured hipct
```

The outlet table records RT, Rp, Rd, C, both RT·C and Rd·C, flows, initial
capacitor/outlet pressure, pressure datum, units, mesh/anatomical identifiers,
method and radius provenance. Network/config/source hashes accompany the table.
The common static target does not determine compliance.

Choose exactly one `rcr` mode:

```json
{"mode": "scapis", "tau_total_s": 1.79}
{"mode": "fixed", "alpha": 0.1, "tau_distal_s": 1.0}
{"mode": "impedance", "harmonics": 10}
```

SCAPIS uses `c=13.3/(diameter_mm)^0.3`, `Rp=ρc/A`, `Rd=RT−Rp` and
`C=tau_total/RT`; nonpositive Rd is rejected. Fixed mode uses `Rp=alpha×RT`,
`Rd=(1−alpha)×RT`, `C=tau_distal/Rd`. Impedance mode fits positive Rp, Rd and C
subject to Rp+Rd=RT against each outlet's supplied `impedance_samples` rows
`[frequency_hz, real_Z, imaginary_Z]` in Pa·s/m³. It uses the first ten positive
frequency samples by default, requires at least two, checks supplied DC against
calibrated RT, and reports the fit residual. Supply harmonics of the same period.
Static radii alone do not provide frequency-dependent impedance data.

## CFX export and transient state

The pinned upstream implementation is
[SCAPIS revision 24f36a5](https://github.com/ak-234/SCAPIS-SCAPIS_GUI_VERSION/tree/24f36a58b9e7a0adea605a39db6d768cbf600ee4).
See [attribution and adaptations](../src/coronary_sdf/rcr/THIRD_PARTY.md).
The adapter supplies explicit RCR values without recalculating them from diameter.
Only one independent coronary tree is exported per case. `Outlet0`, `Outlet1`, …
are used consistently in Python and Fortran; character buffers support more than
100 outlets. User-function arguments have changed; original SCAPIS binaries are
not compatible.

Configure `cfx.mesh`, explicit `mesh_units`, `inlet_regions`, `outlet_regions`
keyed by tree and anatomical outlet ID, and optionally `wall_regions` as a
comma-separated list. Mesh region names are those imported into CFX (Fluent
hyphens become spaces). Every boundary must be accounted for with no duplicates
or overlap; a mismatched tree/mesh is an error. The CCL uses `Q/area@Inlet`.
Density 1050 is propagated through material properties, USER inputs and mass-to-
volume conversion. All pressures share the CFX gauge datum with reference 1 atm.

The implemented passive equations are

`Pout = Pc + Rp Q`, `C dPc/dt = Q − (Pc−Pv)/Rd`.

At start of step n, backward Euler evaluates a trial capacitor pressure using
the last completed step's flow Q[n−1]. CEL evaluations return that trial plus
Rp·Q[n−1] without advancing state during coefficient iterations. The end-of-step
junction box commits Pc once, samples Q[n] and appends a checkpoint. Reverse flow
is allowed. The lagged method is first-order in time and must pass time-step
convergence; Python reference tests do not establish stability of CFX coupling.

`rcr_state.dat` contains repeated `NOUT step dt` headers, zero-based outlet index,
Pc and Q rows, and a `COMMIT step` trailer. Restart export requires `cfx.restart`,
the exact `restart_step` corresponding to the CFD result, and
`restart_mapping_sha256` from the original export status. Mismatched outlet order,
parameters, timestep and partial records are rejected. The native CFD flow-field
restart must still be supplied to the CFX solver separately. Do not restart with
only a CFD `.res` and reset capacitor values. End-to-end CFX restart continuity
and parallel execution remain solver qualification requirements.

The export writes CCL, mesh-resolved CCL, a CFX-Pre `.pre` session, logs and status.
With compatible compiled libraries and CFX-Pre available, omit `--dry-run` to
generate `.def`. Successful text generation, successful `.def` creation and solver
validation are separate flags. The detected installation is CFX 2025 R2; a
compatible compiler and licence have not been established. The three adapted
Fortran sources must be built with this installation's `cfx5mkext`. Add `--compile`
to `export-cfx` to attempt this in the output directory, retaining per-routine logs
and automatically recording source and binary hashes. `cfx.double_precision`
must match the intended solver precision. The library
directory must contain a `build_manifest.json` recording `source_sha256` matching
the exported status; never stamp an old DLL as compatible. A missing executable
or compatible library leaves a dry-run artifact and refuses to claim `.def`
success. CFX solver validation is always false until a separate qualified run.

For a pulsatile inlet provide `cfx.waveform.a` and `.b` Fourier coefficients in
m³/s, `period_s`, and `pulsatile: true`. The zeroth cosine coefficient must equal
that tree's fixed target. The inherited template uses `a*cos(-ωt)+b*sin(-ωt)`;
respect that sine-coefficient sign convention. Default inlet flow is constant. Coronary myocardial
compression is not represented by this passive RCR; a pressure-forcing model
would be a separate extension requiring additional data.

## Comparisons and qualification

Use identical corrected clinical geometry, inlet flow and pressure assumptions
for all resistance methods. `compare` reports anatomical-outlet relative flow
errors against HiP-CT, with conservation errors separately. These are reduced-
network differences, not measured physiological errors and not WSS predictions.

Optional `comparison.observations` points to JSON CFD observations. Each record
names `method`, `network_sha256`, `experiment`, and `qualification` booleans for
`mass_conservation`, `periodic_convergence`, `mesh_convergence`, and
`time_step_convergence`. Supply separately `flow`, `impedance`, `waveform` and
`wss` arrays of `{matched_id, reference, candidate}`. Complex impedance can use
`[real, imaginary]` pairs; waveform arrays must use matched phase samples. WSS
values must refer to matched anatomical surface regions and the same averaging
definition. Relative L2 errors are reported independently. Boolean attestations
must be backed by retained CFD reports; the comparison command cannot infer
convergence from an outlet table.

`experiment` must distinguish `open_cut`, `sealed_branch_removal` and
`uncorrected_sensitivity`. Do not pool these interventions. Preserve full/cropped
geometry hashes, mesh refinement levels, timestep series and last-cycle waveform
change. Reuse prior CFD results only if the geometry, pressure datum, material,
boundary parameters and flow assumptions match.

Before interpretation require: conservation at every node/CFD boundary; periodic
convergence of pressure, flow and WSS; mesh and timestep convergence for each
study family; a small real-subtree validation; then both complete trees.
Thresholds must be preregistered for the study rather than selected after seeing
method differences. Current implementation tests qualify synthetic numerics and
text export. Selected LADAF perimeter measurements have been run, but their
hydraulic support gate has not passed. No complete-tree correction, regenerated
volume mesh, new `.def` or CFD run has been completed.

## Tests

For rejected junction spans, run the read-only diagnostic on the **input** graph:

```powershell
python packages/coronary_sdf/examples/rcr/diagnose_junctions.py --case runs/rcr/ladaf_full_context_subset.json --output runs/rcr/new_junction_diagnosis --segments 67 270 307 308
```

The case must specify acquisition voxel size and segmentation binning. The script
uses the existing section and junction diagnostics, confines segmentation caches
to the new output directory, and records source hashes before and after reading.
The diagnostic's 64-voxel half-window and stability defaults differ from the
perimeter pass: its stability counts are diagnostic evidence, not replacement
acceptance criteria. Missing input provenance displays default source/rejection
labels; those labels are not measured verdicts. Any centreline refinement remains
a separate candidate until repeat perimeter measurement and hydraulic support
checks pass, with geometry displacement and convergence recorded.

Local validation on 2026-10-08: 15 new RCR tests passed, plus 67 existing tests
across the five radius/provenance/surface suites below. Static undefined-name/
unused-import checks and Python compilation passed. A real-data read-only audit
and synthetic calibration/comparison/CCL export completed. The attempted
`cfx5mkext` preflight produced no compatible shared library; no compiler was found
on PATH. CFX-Pre and solver executables exist, but licensing and solver behaviour
remain unverified. These results do not qualify patient CFD or the Fortran ABI.

```powershell
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD='1'
python -m pytest packages/coronary_sdf/tests/test_rcr.py --basetemp=runs/rcr/pytest_new -q
```

Use a fresh test directory for each run. Relevant existing regression suites are
`test_radius_perimeter`, `test_radius_units_cli`, `test_radius_profile`,
`test_radius_surface_provenance` and `test_prepared_surface` in `hipct_seg_debug`.
No original patient dataset or pre-existing local edit is modified by this workflow.
