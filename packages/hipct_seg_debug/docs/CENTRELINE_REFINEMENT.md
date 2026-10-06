# Centreline refinement and circular reconstruction clearance

These are opt-in experimental commands. They preserve segment IDs and connectivity
and use the graph's recorded voxel size. The segmentation constraint applies to
the centreline, including its connecting edges, not to reconstructed circles.
The measured and reconstruction graphs serve different purposes and must be kept
as separate files.

## Measurement geometry

```
py -3.12 -m hipct_seg_debug.edit refine-centreline graph.am --seg labels.am --method centroid-coherent --strength 0.01 --workers 8 --out centred_measured.am
```

The command fits geometry first, then remeasures radii. `--geometry-only` skips
remeasurement and marks old radii on changed segments as unmeasured placeholders.
It is useful for comparing geometry, not for exporting new trusted measurements.
Every written graph gets an adjacent `.report.json`; `--report-json` overrides it.

`--segment ID` can be repeated to select a region while keeping full graph context.
`--roots-json` and repeated `--fixed-node ID` anchor selected nodes. Terminal nodes
are always fixed. Shared junctions may move only when all incident segments are
selected and supply enough trusted section support. `--fixed-junctions` disables
that movement. Existing segmentation gaps remain flagged; the method does not
invent a connection through background.

| Method | Behaviour |
|---|---|
| `none` | Unmodified reference geometry |
| `centroid-spline` | Confidence-weighted cubic fit to section centroids, using the radius estimator's section orientation search |
| `centroid-coherent` | Same fit, but retains the fitted curve's normal when its section passes stability checks; searches alternatives only when it fails |
| `dfs-centroid` | Fits root-to-terminal paths longest first, with shared movable internal nodes and bounded joint fitting at later branch attachments |
| `dfs-centroid-shape` | Adds section-shape diagnostics, physically weighted observations, supported-centroid tangent hints and a fresh final centring audit to joint DFS fitting |
| `laplacian` | Implicit diffusion using physical edge lengths and locally measured section scales |
| `taubin` | Shrinkage-compensated implicit Laplacian, using `2H-H²` for the implicit filter `H` |

The compensated filter is **Taubin-inspired**, not a reproduction of the classical
explicit lambda/mu algorithm. Neither diffusion variant is a recentering algorithm:
both can remove jitter while leaving a systematic offset unchanged. Local scales
vary along each segment, and arclength weighting avoids assigning more influence
to densely sampled regions. The original radius is not a permanent movement cap.

Centreline fitting and opt-in `radius-perimeter --section-filter` use the same
finite-branch ownership and three-plane stability checks. Angles are diagnostic:
a neighbouring axis need not cross a plane for its lumen to contaminate it.
Spatially separate parallel branches remain acceptable. Non-adjacent touching
lumens can be separated by a shared 3D watershed; ambiguous incident junction
sections remain unsupported. Every alternative orientation passes the same checks.

Overlapping junction neighbourhoods are fitted together with fixed outer anchors.
Degree-two joins share a position and derivative; branching nodes retain separate
approach directions and calibres. Unsupported internal links can receive curve
support from exclusive sections on the external approaches. Missing external
support remains an explicit failure, not an inferred anatomical measurement.

Defaults are 25 iterations, 32 sample stations per segment, strength 0.1 and a
256-voxel maximum half-window. Convergence requires two steps below 0.1 voxel.
Non-convergence, insufficient support, blocked moves and pre-existing outside
edges are reported. No new method has been promoted into `optimise-skeleton`.

### Flattened-lumen refinement

```powershell
python -m hipct_seg_debug.edit refine-centreline graph.am --seg labels.am --method dfs-centroid-shape --strength 0.01 --workers 8 --max-samples 32 --geometry-only --out centred_geometry.am --report-json centring.json
```

Use `--roots-json` or repeated `--root-node` when roots are known. For a regional
experiment, select complete junction neighbourhoods with the qualification driver:

```powershell
python -m hipct_seg_debug.edit.junction_qualification --graph graph.am --seg labels.am --method dfs-centroid-shape --segment 3717 --segment 3655 --segment 3612 --workers 8 --geometry-only --out-dir runs/shape-regions
```

This mode retains actual cross-sectional area centroids, including flattened
sections. It uses the shared ownership/stability filter and accepts no centroid
whose position or straight movement leaves the selected lumen. A concave section
can have its centroid outside the lumen; such an observation is rejected, with
curve support left to neighbouring exclusive sections. Section normals use
physical-arclength curve fits, with exclusive centroid observations providing
direction hints on subsequent iterations. All alternate normals still pass the
shared filter. The original Jin shortest-path cost is not used in this fitter.

Independent boundary ellipse fits report axis ratio, centroid disagreement and
normalised radial residual. They do not replace the lumen, its centroid or its
perimeter. Only a fit with an interior centre, at least a two-pixel minor radius
and radial RMS at most 0.12 contributes an agreement weight. These are provisional
diagnostic criteria, not anatomical classifiers. Flattening alone never rejects
or downweights a section. Area/perimeter instability reduces confidence.

Accepted section observations represent their physical support intervals (capped
at four local calibres), instead of the adjacent exported point spacing. Joint
fitting retains locally scaled curvature regularisation, shared movable junction
variables and the exact degree-two derivative constraint. Outer anchors, roots,
terminals, segment IDs, radii and connectivity retain their previous safeguards.
There is no independent point-shifting correction or forced daughter calibre.

The report contains per-section offsets in micrometres and in units of the short
semi-axis, ellipse checks, and per-segment turn/curvature diagnostics. After the
last geometry update, sections are measured again on the final curve. This audit
uses fitted-curve normals without the previous iteration's tangent hints. It can
run in parallel for file-backed segmentation. `centring_final` is one of
`centred`, `off_centre`, or `insufficient_support`. The provisional tolerance is
`max(0.75 voxel, 0.15 * short semi-axis)`. A stationary curve that fails this audit
is not reported as converged. These metrics measure agreement with segmentation,
not recovery of an unknown undeformed anatomical centreline.
Degree-two joins also receive a tangent-continuity audit; a unit-vector residual
above 0.001 prevents a convergence claim. Branching nodes are not forced to share
one direction. Nearest-voxel section sampling has a resolution floor: a reported
zero section offset does not establish zero subvoxel error.

Geometry-only outputs retain radius placeholders. Remeasure through the shared
section filter after geometry review, then derive reconstruction profiles and
apply optional smooth clearance correction. Containment of a centreline does
not certify containment or nonintersection of reconstructed circular surfaces.

## Optional reconstruction layout

### Longest-path-first experiment

```powershell
py -3.12 -m hipct_seg_debug.edit refine-centreline graph.am --seg labels.am --method dfs-centroid --roots-json roots.json --strength 0.01 --workers 8 --out dfs_measured.am
py -3.12 -m hipct_seg_debug.edit prepare-reconstruction dfs_measured.am --seg labels.am --radius-profile dfs-confidence --path-plan-json dfs_measured.am.report.json --skip-clearance --out dfs_reconstruction.am
```

Use the same roots for both stages. Repeated `--root-node ID` is an alternative to
a root sidecar. `--path-plan-json` reuses the refinement report's roots and original
path ordering, verifying graph topology before use. Without a saved plan the
profile ranks paths using the current geometry, which can change the order of
similarly long paths. Without explicit roots, the existing automatic root heuristic is
used; the geometry report records the complete initial path plan. Paths are ranked
by physical arclength once, with deterministic ID tie breaks. Cycles and multiple
explicit roots in one component are refused rather than silently pruning topology.

Each refinement iteration fits the longest path, then the unfitted suffix of each
remaining path. Earlier geometry is fixed outside a bounded attachment region;
all incident approaches inside it participate in the shared-node solve. Roots,
terminals and regional boundary nodes remain fixed. The solve uses section-centroid
observations and a calibre-scaled bending penalty, followed by objective and
containment checks. Degree-two derivatives are coupled. Unsupported daughters do
not veto a supported through path, but remain explicitly unsupported in the report.
Two-point internal links can receive support from accepted sections on both sides.
This experimental method does not establish that the longest path is the anatomical
main vessel. Compare the recorded path order and segmentation overlays.
Progress includes `stage: path_fit` after section sampling. Identical failed
regional prefixes are not retried until another accepted fit changes the fitting
state; their original failure remains in the report.

`dfs-confidence` extends the existing `confidence` profile across segment boundaries
along the first path that owns each segment. It interpolates rejected/unmeasured
spans in log radius with shape-preserving cubic interpolation. Trusted measurements,
including supported narrowing, remain unchanged. A later daughter uses its own
anchors and the existing bounded branch-local endpoint extension; it does not
inherit a parent radius. Conflicting trusted joint records are not averaged.

Cross-segment anchor separation is limited to `--transition-radii` (default 4)
times the smaller anchor radius, with a minimum of eight voxels. Longer gaps remain
unresolved in `paths[].rejected_spans`. Existing within-segment confidence filling
is unchanged. Mere proximity to a bifurcation does not invalidate an accepted
measurement: remeasure with the shared section filter to identify contaminated
sections first. This avoids replacing a supported calibre increase with a guessed
radius. Original measurements stay in `radius_measured_um`; path-derived values
have `radius_adjustment_reason = 6`. Profiles are reconstruction inputs, not new
measurements. The compatibility default remains `preserve`.

For checkpointed regional geometry comparisons, add `--method dfs-centroid` and
the same root options to `python -m hipct_seg_debug.edit.junction_qualification`.
That qualification pipeline still uses its established `confidence` reconstruction
profile; use `prepare-reconstruction` explicitly to compare `dfs-confidence`.
`--skip-clearance` leaves clearance and mesh validation outstanding and therefore
returns the existing review-required exit status.

```
py -3.12 -m hipct_seg_debug.edit prepare-reconstruction centred_measured.am --seg labels.am --out reconstruction_layout.am
```

Run this **after** final radius measurement and **before** surface reconstruction.
The compatibility default, `--radius-profile preserve`, retains the input radii.
`--radius-profile confidence` preserves trusted anchors and fills unsupported spans
with shape-preserving interpolation in log radius. Degree-two boundary conflicts
are remeasured and unresolved disagreements remain flagged. Junction extensions
retain each branch's own calibre. Separate point fields retain measured radius,
reconstruction radius, adjustment and reason. `--skip-clearance` prepares this
profile without moving the graph.

Clearance correction transports the derived radii unchanged on a smooth displacement field.
Do not remeasure radii on the displaced reconstruction layout: they describe the
measurement graph, which the report identifies.

The clearance stage detects overlapping circular-tube bounds and solves for
overlapping smooth displacement fields along vessel spans. Each field and its
first two derivatives vanish at its support boundary; endpoint tapers retain
junction positions and avoid isolated-point kinks. Local foreground checks and
reversal checks constrain the accepted deformation. A branch that cannot move
does not automatically freeze unrelated feasible corrections.

`--max-displacement-radii` defaults to 0.5, `--max-iterations` to 30, and
`--gap-um` to zero. Size-bucketed spatial indexing handles vessels of different
calibres without searching every small branch using the largest artery's radius.
Radii are never reduced. If separation is infeasible inside the segmentation,
the graph remains a review candidate and the command returns status 2 with
remaining contacts, tight bends and containment failures in the report.

The detector uses conservative maximum-radius capsules for each graph edge.
Expected overlap is exempted only within a bounded neighbourhood of a shared
junction; sharing a node does not exempt whole branches. Tapered edges can be
over-flagged. The test is a preflight, not a certificate for the final blended SDF:
surface blending or later mesh smoothing can create additional contacts. Even a
clear result is explicitly named `capsule-clear-surface-unvalidated` and requires
validation of the actual reconstructed surface. Tight local bends are reported
separately from nonlocal tube contacts.

## Qualification

```
py -3.12 -m hipct_seg_debug.edit.centreline_benchmark --graph graph.am --seg labels.am --out-dir runs/centreline_comparison --workers 16
```

The benchmark selects 48 reproducible regions, includes the known failures, and
compares fixed reference planes across methods. Outputs retain every region,
including failed methods and sections without reference support. Completed region
files are checkpoints; rerunning with the same manifest resumes missing regions.

Real-data scores are fixed-section residuals, not independent anatomical ground
truth. The labels `development` and `heldout` record the initial split; inspecting
that split to design another method makes subsequent reuse exploratory. Neither
those scores nor reduced roughness alone qualify a method as anatomically correct.
Known-centre synthetic vessels, untouched controls, junctions, foreground
containment, different sample spacings, and the final surface all need checking.

The synthetic comparison requires no dataset or test-package imports:

```
python -m hipct_seg_debug.edit.centreline_synthetic_benchmark --out runs/synthetic/results.json
```

See [the LADAF-2021-17 qualification report](CENTRELINE_QUALIFICATION_2021_17.md)
for completed comparisons and the remaining acceptance gaps. Generated graphs,
segmentation data and per-region outputs are local artifacts, excluded from Git.

See [junction refinement and qualification](JUNCTION_REFINEMENT.md) for the shared
filter, direct prepared-surface command, checkpointed regional runner and current
rollout limitations.
