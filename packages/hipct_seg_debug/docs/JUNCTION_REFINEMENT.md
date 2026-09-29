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

Two unresolved issues are documented by that investigation:

- Finite foreign tubes use individual sampled edge directions. A synthetic axial
  continuation with a half-voxel endpoint kink falsely contaminates an upstream
  plane, while the straight version passes. A strict expected-failure regression
  records this defect until finite branch support is made robust to point noise.
- Reducing the slab half-span from 0.5 to 0.125 input radii recovered one section
  on 3655, but its ownership partition retained only 49.1% of the unpartitioned
  perimeter at the **same orientation** (3877 versus 7892 micrometres). Stability
  alone therefore cannot establish anatomically correct ownership. No shortened
  slab default or relaxed contamination threshold has been adopted.
