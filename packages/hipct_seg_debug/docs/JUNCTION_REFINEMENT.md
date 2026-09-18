# Junction refinement and qualification

These changes are experimental. Synthetic and software checks do not establish
anatomical accuracy on a real tree. Keep the input, measured and reconstruction
graphs in separate files. A geometry-only graph retains placeholder radii and
must be remeasured before reconstruction.

## Section validation

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
