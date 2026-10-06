# Junction refinement and qualification

These changes are experimental. Synthetic and software checks do not establish
anatomical accuracy on a real tree. Keep the input, measured and reconstruction
graphs in separate files. A geometry-only graph retains placeholder radii and
must be remeasured before reconstruction.

Add `--quiet-dependency-warnings` to the `junction_qualification` command to hide
the known Paramiko TripleDES/Blowfish import deprecations in the parent and worker
processes. Geometry, numerical and other dependency warnings remain visible.
For any package command, PowerShell users can instead set
`$env:HIPCT_QUIET_DEPENDENCY_WARNINGS="1"` before launching it; remove that environment
variable to restore the default. Neither option changes an already running process.

## Runtime and exact reuse

Refinement automatically reuses section observations for unchanged regions of an
Amira lattice. Reuse requires identical target geometry, local scale and relevant
point flags, plus unchanged geometry/radii for every potentially influencing
vessel. Conservative bounds include the maximum section window, ownership volume,
all candidate orientations and slab offsets. A neighbour entering the region or
growing enough to reach it invalidates the entry. Rejected observations may also
be reused; they remain rejected, never promoted to trusted support. Mutable array
and live-edit inputs do not enable this cache.

Progress records include `reused_section_segments`. Use `--no-section-cache` with
`junction_qualification` or `refine-centreline` for an uncached comparison. Use a
new output directory when changing run settings/code. Cache entries are local to
one run and are not checkpointed.

Decisive incident-branch rejection avoids unnecessary companion-plane work;
read-only diagnostic tracing still examines all three planes. The first recorded
rejection reason can consequently differ where a candidate has multiple faults,
but an accepted section must still pass all checks. Backtracking reuses starting
curve containment checks. Radius workers retain whole-tree indexes and bounded
decoded row caches across provisional batches; those caches never cross a graph
update. Refinement workers limit nested OpenCV/BLAS threads to avoid CPU contention.

The earlier full-tree experiment required about 24.1 hours for refinement and
3.2 hours for remeasurement with 24 workers. These are baseline timings, not a
prediction for the optimised implementation. Five real-data midpoint probes,
each run cold and warm, retained identical selected sections and rejection
diagnostics in the before/after comparison. A full-tree speed-up factor has not
yet been measured; savings depend on how many neighbourhoods remain unchanged.
In a small full-context batch benchmark (two real segments, each measured twice),
worker-state reuse reduced elapsed time from 15.54 to 5.08 seconds with identical
provisional radii and rejection/source arrays. This isolates batch reuse and is
not an estimate of the speed-up for a complete tree.

## Section validation

### DFS regional experiment (2026-09-29)

The opt-in `dfs-centroid` experiment ran two iterations with eight section stations
per segment and eight measurement workers on complete neighbourhoods of 3717,
3612/3655, plus untouched controls 318 and 4390. Internal nodes moved jointly:
endpoint displacements were approximately 160/159 um for 3717, 142/191 um for
3612 and 65/224 um for 3655. No region added segmentation exits; existing outside
edge counts remained 9, 5, 1 and 7 respectively. These are short geometry tests,
not converged or anatomically qualified reconstructions. Radii were preserved as
placeholders and have not been remeasured on these new curves.

The second 3717 iteration rejected its primary path proposal because an incident
approach on segment 3041 would leave the segmentation. Other rejected approaches
included 3339, 3400 and 3551. The objective decreased, so these are containment
failures rather than uphill-objective failures. Projected section overlays still
show residual centre offsets. Resolve these constraints before promoting the
method to a full-tree reconstruction; reducing a visual bulge is not sufficient.

Skipping repeated, identical failed regional prefixes retained exactly the same
first-iteration coordinates while reducing fitting attempts from 94 to 28. Observed
iteration time changed from 196.9 to 171.2 seconds; this is not a full-tree speed
benchmark. The updated two-iteration 3717 geometry run took 306.0 seconds.

The separate `dfs-confidence` profile experiment on the completed v5 measurements
filled unsupported samples on 3717 and 3612. Segment 3655 remained unresolved:
its bracketing anchors were 3600.5 um apart, exceeding the default 3157.8 um local
transition limit. Trusted measurements were retained. This profile experiment
does not qualify the v5 geometry or replace remeasurement after DFS refinement.

See [CENTRELINE_REFINEMENT.md](CENTRELINE_REFINEMENT.md) for the opt-in commands,
saved path ordering, interpolation limits and provenance fields.

### Shared section checks

`section_validation.py` is shared by centreline refinement and
`radius-perimeter --section-filter`. It checks finite branch extent, local calibre,
overlap with the selected segmentation component, and foreground connections to
the neighbouring branch. Angles are recorded but never reject a spatially
exclusive section. No angular rejection threshold has been enabled.

Every candidate orientation uses three parallel sections and the same area,
perimeter and centroid stability criteria. The fitted target normal is preferred.
Topological degree-two continuations are treated as one vessel for ownership.
Non-adjacent touching lumens can be partitioned using one 3D ownership volume for
the slab. Incident branches in a merged junction are rejected instead of assigning
an invented boundary through the shared node. Ownership is an estimate and is
retained as separate provenance, not proof of anatomical separation.

The new ownership path uses float distance and plateau-aware flooding rather than
the legacy quantised inverse-distance watershed. The latter produced an incorrect
outer-shell assignment in the touching-cylinder regression. See the
[scikit-image watershed implementation](https://scikit-image.org/docs/0.24.x/api/skimage.segmentation.html#skimage.segmentation.watershed).
Finite polylines are clipped and rasterised into the ownership ROI, so sparse
sampling does not lose a branch whose stored endpoints are outside the ROI.

Rejection counters distinguish target obliquity (currently zero), neighbouring
lumen contamination, unstable sections, truncation and insufficient support.
Maximum target obliquity is a separate diagnostic. The counters describe rejected
candidates; a point can still have an accepted alternative orientation.

The shared measurement path disables legacy authored junction tapers and the
projection-only junction mask. Existing radius-perimeter defaults remain available
for compatibility; new refinement workflows explicitly enable the shared filter.

## Geometry and profiles

Fitting uses physical arclength and local section calibre. Spline proposals and
step acceptance use the same sampled curvature objective. A numerical coefficient
prior handles unobserved knot intervals under uneven sampling. Containment checks
inspect every crossed voxel on centreline edges and on displacement paths.

Overlapping junction spans share one sparse solve, fixed outer positions and
tangents, and one acceptance step. Degree-two joins have a shared derivative.
Multifurcations keep branch-specific approach observations. Unsupported internal
links can be fitted from supported external approaches; unsupported external
approaches prevent qualification. Roots, terminals, IDs and topology are retained.
Oscillation, changing support, blocked movement and convergence are reported.

`prepare-reconstruction --radius-profile confidence` uses PCHIP interpolation in
log radius without overshoot or changes to trusted anchors. Degree-two conflicts
are remeasured, with the resulting measurement graph saved separately. Bounded
junction extensions use each branch's nearest trusted calibre independently.
Unsupported terminal spans and unresolved conflicts remain review requirements.
The default `preserve` policy keeps existing radii.

Point fields are `radius_measured_um`, `radius_reconstruction_um`,
`radius_adjustment_um`, `radius_adjustment_reason`, and `radius_anchor_trusted`.
Adjustment codes: 0 unchanged, 1 supported gap interpolation, 2 degree-two
continuation, 3 conflict, 4 unsupported, 5 branch-local junction extension.
Discontinuity classifications are diagnostics, not independent anatomical labels.

## Reconstruction

```sh
python -m hipct_seg_debug.edit prepare-reconstruction measured.am --seg labels.am --radius-profile confidence --out reconstruction.am
python -m hipct_seg_debug.edit surface reconstruction.am --prepared-report reconstruction.am.report.json --out-dir surface_review
```

The prepared surface path verifies the graph hash and bypasses legacy centreline
and radius smoothing. Circular branch profiles use bounded, arbitrary-degree
junction patches. Actual mesh checks include components, genus, manifoldness,
self-intersections, branch radius rays, and patch vertex residuals against both
the intended blend and unblended union. They detect numerical necks, bulges and
unintended connections within the checked tolerances. Carina anatomy still needs
segmentation and known-shape review. No subsequent mesh smoothing is applied.

Optional clearance correction follows profile derivation and holds radii fixed.
Infeasible contacts remain reported. The centreline must remain inside segmentation;
the reconstructed circular surface may extend beyond a flattened observed lumen.
A regional preparation report cannot qualify the unprepared remainder of a graph.

## Regional qualification and rollout

```sh
python -m hipct_seg_debug.edit.junction_qualification --graph measured_input.am --seg labels.am --segment 3717 --segment 3655 --segment 3612 --segment CONTROL_ID --out-dir runs/junction_review --workers 16
```

Replace `CONTROL_ID` with independent controls and repeat `--segment` as needed.
Regions expand across short links whose junction neighbourhoods overlap. Targets
covered by the same region share its work. The graph context remains complete for
section filtering and measurement. Exported Amira graphs retain original IDs;
regional surface endpoints are artificial review boundaries.

The runner records input fingerprints and source hashes, saves stage checkpoints,
and refuses stale checkpoints after code or option changes. Iteration checkpoints
and per-segment section progress make long fitting runs inspectable. Completed
regional stages resume with identical inputs and options; iteration files can be
used explicitly as the input to a new geometry experiment. `--geometry-only`
skips radius and mesh work and cannot qualify a full-tree reconstruction.

Compressed-lattice section work uses processes with full graph context; radius
measurement is also parallel. The decoder releases Python's thread lock and an
ownership volume is reused across compatible alternative planes. These changes
reduce repeated work without weakening acceptance criteria.

Only add `--full-tree` after supplying the failure regions and independent controls.
It starts the full sequence only when every supplied region passes geometry,
radius-profile, clearance and actual-mesh checks. It does not certify that the
user-supplied control set is statistically independent or representative.

For an explicitly requested full-tree experiment before regional qualification,
use `--experimental-full-tree` without `--segment` or `--full-tree`.
`--measurement-only` stops after writing the separate measured graph and reports
`review_required`; it does not certify geometry or produce a validated surface.
Iteration checkpoints include current support/convergence fields. Existing gaps
remain reported constraints, not repaired anatomy.

Convergence diagnostics distinguish unsupported approaches, containment failures
and objective increases. Interior centroid fitting uses a smooth spline
displacement with endpoints and points adjoining existing gaps fixed during the
solve. This includes the unchanged sampled curve in the feasible model: an
absolute spline fit could previously propose only uphill or forbidden moves.
Neighbourhood reports identify the specific segments and constraints blocking a
joint step. A stationary but unsupported neighbourhood still requires review.

## Local evidence and limitations, 2026-09-18

The earlier five-iteration regional experiment completed for the known failures
and six controls. The 3612/3655 region contains 29 segments; the 3717 region contains
35. Neither converged. Existing outside edges were retained (5 and 9 respectively),
with no new exits. Controls also retained their pre-existing exits and did not
qualify. These are geometry experiments, not corrected measured trees.

The 3655 centre section can differ in ownership from its parallel neighbours.
Expanding the junction context and resolving non-adjacent ownership recovered
some support on neighbouring segments, but ambiguous sections remain rejected.
The experiment also exposed inconsistent fitting/acceptance objectives, corrected
after that run. Its results must not be attributed to the corrected implementation.

The synthetic comparison before that final numerical change covered round,
flattened and curved vessels at two sampling spacings. Across six cases the median
centre error was 3.07 voxels before fitting and 0.12 after coherent centroid fitting;
Laplacian and Taubin comparators retained about 3.00 voxels of systematic offset.
No method introduced a segmentation exit. This supports centroid fitting as a
candidate, not a claim that it is universally optimal.

Regression coverage includes degree-two joins, three to six incident branches,
unequal calibres, uneven sampling, overlapping junctions with unsupported internal
links, parallel contamination without axis crossing, spatially separate neighbours,
reversed directions, reordered branches, trusted narrowing, artificial boundary
steps, radius provenance, infeasible collisions, and actual junction meshes.

The latest implementation still requires regional qualification and anatomical
overlay review. No full-tree geometry-refined and remeasured reconstruction has
been accepted or published as a corrected result.

## Endpoint bowing and rejected-section investigation, 2026-09-28

The experimental full-tree run completed 25 refinement iterations and all radius
measurements, but did not converge. Segment 3717 retained both original endpoints
while its interior moved by up to 1.036 mm. Segments 3655 and 3612 did not move;
their neighbouring segments accumulated substantial interior movement with fixed
ends. Independent interior acceptance before a failed junction solve could retain
these bowed curves. These outputs are not accepted anatomical corrections.

Refinement now solves shared junctions and approach curves first. Independent
interior fitting preserves the accepted junction spans exactly, including their
approach tangents. Segments in unresolved neighbourhoods defer independent
interior fitting. This prevents that partial-update failure; it does not supply
missing section support or prove that a junction is anatomically centred. New
regional experiments start from the pre-refinement graph, not the bowed output.

Read-only rejection tracing is available through
`python -m hipct_seg_debug.edit.section_debug --graph INPUT --seg LABELS
--segment ID --out-dir OUTPUT`. JSON includes each completed candidate slab,
incident rivals, unpartitioned area/perimeter, and the exact foreign edge,
foreground witness, axis distance and radius that triggered contamination.
`--point INDEX` selects individual points. `--slab-half-span FRACTION` is only a
sensitivity experiment and never writes a corrected graph.

Two issues were documented by that investigation (the first synthetic defect is
addressed by the endpoint-support correction described below):

- Finite foreign tubes use individual sampled edge directions. A synthetic axial
  continuation with a half-voxel endpoint kink falsely contaminates an upstream
  plane, while the straight version passes. This initially had a strict
  expected-failure regression; it now passes without that marker.
- Reducing the slab half-span from 0.5 to 0.125 input radii recovered one section
  on 3655, but its ownership partition retained only 49.1% of the unpartitioned
  perimeter at the **same orientation** (3877 versus 7892 micrometres). Stability
  alone therefore cannot establish anatomically correct ownership. No shortened
  slab default or relaxed contamination threshold has been adopted.

## Flattened-lumen refinement evaluation (2026-10-01)

The opt-in `dfs-centroid-shape` mode adds independent shape evidence and final
centring/degree-two continuity audits. The focused suite passed 124 tests, with
the optional VMTK integration skipped and the existing noisy axial-continuation
filter regression xfailed. A subsequent geometry-export provenance test passed:
geometry checkpoints invalidate radius trust while retaining numeric placeholders.
On a flattened synthetic vessel, maximum error fell from 59.98 to 2.41 micrometres
in four iterations, with fixed endpoints and no segmentation exits.

Complete junction neighbourhoods were then evaluated at eight section stations
per segment. The 3612 neighbourhood (including 3655) resumed a two-iteration
checkpoint for one additional iteration. Other regions ran two iterations. These
are geometry experiments, not remeasured or reconstruction-ready graphs.

| Target | Final centring audit | Maximum accepted-section offset |
|---|---|---|
| 3717 | Off centre at one of six accepted stations | 197.36 um; 0.333 short semi-axes |
| 3612 | Insufficient support: no accepted stations | Unknown |
| 3655 | Insufficient support: no accepted stations | Unknown |
| Control 318 | Within provisional tolerance at four accepted stations | 18.44 um |
| Control 4390 | Within provisional tolerance at three accepted stations | 13.60 um |

The final audit on 3612 recorded 71 neighbouring-lumen rejections and one
insufficient-support rejection across candidate orientations; 3655 recorded 72
neighbouring-lumen rejections. These are candidate counts, not 71/72 independent
cross-sections. Missing support was not replaced with invented measurements.

A subsequent read-only audit requested as many samples as points on 3612 and
3655. It found
accepted sections at points 22 and 23 on 3612, between the coarse audit stations,
but both were off centre: maximum offset 333.10 micrometres, or 0.878 short
semi-axes. No section was accepted on 3655. This diagnostic used stored radii for
initial section/tangent scales rather than the refinement's iteratively measured
scales, so it is not an exact resampling of the earlier audit. It demonstrates
that the coarse sampling can miss a narrow support window; it does not qualify
3612. Production defaults use 32 stations, and neither sparse audit establishes
centring along the entire curve. The dense-sampling bug discovered below also
affected this diagnostic: requesting that budget did not actually visit every
point on an unevenly sampled curve.

Exported graphs preserve IDs, connectivity, shared endpoint coordinates, fixed
roots/terminals, numeric radii and unselected geometry. Exact edge traversal found
no new segmentation exits. Existing outside-edge counts stayed 12/12 for the
expanded 3612 neighbourhood, 9/9 for 3717, 1/1 for control 318 and 7/7 for control
4390. Neither target neighbourhood nor the complete control neighbourhoods
converged. The target centring results therefore block full-tree promotion;
remeasurement and surface qualification have not been performed on these outputs.

The implemented Jin hybrid is a post-extraction centroid/curvature fit, not a
modified Jin shortest-path cost. The optional VMTK baseline remains unexecuted
because its dependency is absent. See `JIN_MCP.md` for these distinctions.

## Rejected-section and sampling fixes (2026-10-01)

The finite foreign-tube model now clips endpoint-local support using an inward
chord over physical arclength, rather than allowing a tiny first-edge kink to
project the tube backwards. Each endpoint cap applies only within two endpoint
radii, limited to half the branch length. Distant returning curves retain their
support. This is a bounded geometric model using stored calibre, not a claim that
the cap is an observed anatomical boundary. Angles alone still do not reject a
section; candidate overlap and the foreground connection remain required.

The former axial-continuation expected failure now passes, including reversed
directions and point subdivision. Tests also retain detection of a curved return,
a parallel contaminating daughter, overlap without an axis crossing, and a
spatially separate daughter with an overestimated radius.

A second defect affected dense audits: mapping N uniformly spaced arclength
stations to N exported points could repeat indices and skip tightly sampled
points. A budget covering the point count now visits every non-invented point
directly; lower budgets still use physical arclength. Section diagnostics record
the actual sampled indices. A regression exercises a tightly sampled window.

On the unchanged 3612/3655 geometry, a true every-point before/after comparison
using identical stored-radius scales found:

| Target | Before endpoint correction | After endpoint correction |
|---|---|---|
| 3612 | Points 21-24 accepted; max offset 344.44 um | Points 18 and 21-24 accepted; max offset 366.85 um |
| 3655 | No accepted points | No accepted points |

All accepted 3612 points remain off centre. The larger maximum after correction
comes from newly exposed support, not a geometry change. At 3655 point 22, the
fitted orientation still encounters incident branches 794 and 3661 on its +0.5
radius companion plane, plus non-incident branch 1046 on other planes. The cap
correction does not remove those witnesses. Ownership remains unresolved; no
slab-width or contamination-threshold relaxation has been adopted.

The regional runner accepts `--dense-target-sections`: every non-invented point
on each requested `--segment` is sampled during fitting and final auditing,
while neighbouring segments retain `--max-samples`. Full graph context and the
joint neighbourhood fit are unchanged. This avoids paying for dense sections
on every neighbouring branch when investigating a narrow target support window.
For example:

```powershell
python -m hipct_seg_debug.edit.junction_qualification --graph INPUT.am --seg LABELS.am --segment 3612 --segment 3655 --method dfs-centroid-shape --dense-target-sections --max-samples 8 --workers 8 --max-iterations 2 --geometry-only --out-dir runs/dense_target_review
```

Use the dataset's established `--root-node`/`--roots-json` configuration as well.
Dense checks of the unchanged control graphs retained identical decisions before
and after the endpoint fix: 318 was centred at 11 accepted points, but 4390 was
off centre at three of 13 accepted points (maximum offset 137.01 micrometres).
The earlier coarse control audit had missed these failures. An accepted sample
is not a certificate of centring along the entire curve.

The targeted dense experiment then completed two additional fitting iterations
on the 33-segment 3612/3655 neighbourhood in 513.56 seconds. All 46 points of
3612 and all 31 points of 3655 were attempted; neighbours used eight stations.
The final audit accepted 16 sections on 3612, all within provisional tolerance,
with maximum offset 42.80 micrometres. A separate check of the exported geometry
using the original stored-radius sampling scales accepted eight sections, with
maximum offset 39.50 micrometres, also within tolerance. This supports a centring
improvement on observed exclusive sections, not a whole-vessel accuracy claim.
Segment 3655 still had no accepted sections. The full neighbourhood did not
converge: the second pass reported six oscillating segments and 18 blocked
segments. Numeric radii and structural invariants were preserved, and exact
edge traversal found no new exits (12 existing outside edges before and after).
Radii have not been remeasured. A preceding run with 64 stations on every
neighbour was stopped before completing its first fit and was not used as a
geometry result.

## Complex-junction topology audit (2026-10-02)

The refinement preserves imported connectivity. It can move a shared node but
does not independently identify which branches anatomically join. Misassigned
connectivity or several graph nodes representing one junction can therefore
persist through refinement.

A read-only audit identifies short links between branching nodes and records
their section evidence. The provisional proximity limit is the sum of stored
endpoint radii; these radii may themselves be inaccurate in flattened vessels.
Only links with an explicit final audit reporting zero accepted sections are
grouped as unresolved complexes. Missing audit evidence is distinct from zero
support, and observed exclusive sections keep two nodes separate in this
diagnostic. Direct nodes with four or more incident branches are also reported.

```powershell
python -m hipct_seg_debug.edit.junction_topology_audit --graph REGION/geometry.am --geometry-report REGION/geometry.json --out REGION/junction_topology.json
```

The graph and report must describe the same geometry. The audit exports node
degrees, endpoint consistency, short-link lengths, external branch IDs and input
fingerprints. It never merges nodes or changes the graph. Six tests cover
three-to-six-way nodes, reversed edges, reordered records, absent versus negative
section evidence, supported connectors and graph preservation.

The known failure regions contain concrete candidates:

| Region | Candidate nodes | Internal links | External branches |
|---|---|---|---|
| Near 3717 | 3713, 3716 | 3716 (498.58 um; no accepted sections in the coarse audit) | 3593, 3643, 3715, 3717 |
| Around 3655 | 1547, 3333, 3588 | 3655 and 3661 (1654.18 and 2082.60 um; no accepted sections) | 794, 2200, 2974, 3648, 3667 |

The first has four external approaches and the second five. Either may describe
closely spaced bifurcations rather than one multifurcation. These are candidates
for segmentation review, not proof that the imported topology is wrong. In
contrast, 3612's short connector now has exclusive-section support and is not
grouped by this criterion. Native-voxel segmentation surfaces with every incident
branch and node label were generated locally for review. The next topology
decision requires tracing those approaches through the segmentation and comparing
one-junction and multiple-junction hypotheses; radius-based proximity alone must
not author a connectivity change.
