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

`connect --wavefront` runs this refinement itself over the whole graph before
proposing any reconnection (`--refine-method`, default `centroid-coherent`;
`none` to skip on an already refined input), remeasures radii the same way, and
writes the same `centreline_displacement_um` field. See
[WAVEFRONT_RECONNECTION.md](WAVEFRONT_RECONNECTION.md).

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
Progress includes `stage: path_fit` after section sampling. A failed run is not
retried until an accepted fit changes its inputs: the geometry of the run and of
every segment incident to its nodes. Before this, any accepted fit anywhere triggered
a retry, so one blocked 3717 cluster was solved six times per iteration. The
original failure remains in the report.

The containment line search accepts or refuses the whole neighbourhood. A polyline
that already lies on the segmentation boundary leaves it under any step, however
small. That let one grazing side approach (3041 on 3717) veto the primary path. When
the only objection is `new_segmentation_exit`, the solver now pins the offending
vertices where they are and solves again, for up to `PIN_ROUNDS` (4) rounds. Pinned
vertices are reported per segment in `pinned_points`. A pinned junction node is
reported separately in `pinned_nodes` and does not move. Any other objection still
blocks the neighbourhood. A blocked row carries `blocked_detail` for each vetoing
segment:

| Field | Meaning |
|---|---|
| `exit_edges` | edges that would newly leave the segmentation at the last tried step |
| `exit_near_moved_node` | an exit within two edges of a jointly moved node |
| `contained_alpha` | largest tried step this segment alone accepts; `null` means even 1/256 exits, i.e. it grazes |
| `accepted_sections`, `released_span`, `max_move_um` | its support, the released sample range, and its largest move at the last step |

`converged` requires every segment in the selection to be curve-supported. A
region that contains a segment with no usable section therefore never converges,
however still the rest of it is. `supported_stable` applies the same 0.1-voxel,
two-iteration test to the supported, unblocked segments only. `unsupported_segments`
names the segments it excludes. Each `history` row also records:

| Field | Meaning |
|---|---|
| `median_move_um` | median per-segment peak move |
| `peak_segment` | the segment that moved furthest |
| `supported_max_move_um` | the largest move among supported, unblocked segments |
| `pinned_points` | vertices held by the active set in this iteration |
| `unsupported_segments` | how many segments lacked section support |

A `pinned_points` count that grows from iteration to iteration means the curves are
held against the boundary rather than settling inside it.

### Skeleton simplification before DFS fitting

A skeleton often splits one branch point into several degree-3 nodes joined by
links shorter than the vessel they lie in. On the left tree, 3695 (792 um) and 3698
(1291 um) sit inside a vessel of about 1.6 mm radius. These links have no cross-section
of their own. They route the longest path through a stub a third of the parent's
calibre, and force the joint fit to move all of those nodes at once.
`simplify-skeleton` follows the two topology phases of
[PMC10182136, Fig. 4](https://pmc.ncbi.nlm.nih.gov/articles/PMC10182136/);
phase three, smoothing, is `dfs-centroid` itself:

1. **Short leaves**: the existing `prune_spurs` (see [SKELETONISATION.md](SKELETONISATION.md)).
   `--leaf-min-um` is the paper's user tolerance. It runs over the whole tree and is
   skipped when `--segment` limits the run to a region.
2. **Contained leaves** (`edit/contained_leaves.py`, not in the paper): a leaf that lies
   inside another vessel's lumen is a medial-sheet spur of a flattened vessel, not a
   branch. At up to 12 points along the leaf, each nearby vessel whose cross-section
   plane holds the point (within 1.5 voxels) is asked whether its lumen contains it.
   The section is cut at an interior station of that vessel, at least two radii from
   its junction ends: at a junction every branch is joined to the lumen, so a real
   branch leaving it square would otherwise also read as contained. A leaf is removed
   when at least half its points were decided, 80% of those are contained, and the
   tip half is as well. Any nearby vessel may contain a point (1046 lies in 3655's
   sections but nearer stations of other vessels); `--contained-nearest-host` asks
   only the nearest. A leaf ending at a `--root-node`, or with a recorded Strahler
   order above 1 (the trunk's root end), is never removed. Removal keeps the other
   ids: the node left at degree two is not rejoined. `--no-contained-prune` skips
   this phase.
3. **Short inner links** (`edit/junction_links.py`): the links B0..Bk of a cluster are
   removed and their end nodes welded. The paper's p_CA is the centre of mass of the
   removed links' endpoints. The node is then moved to p_NewC, the point of greatest
   distance to the segmentation boundary within the cluster's own extent around p_CA,
   inside the same foreground component.

```powershell
py -3.12 -m hipct_seg_debug.edit simplify-skeleton graph.am --seg labels.am --segment 3717 --report-json dry.json
py -3.12 -m hipct_seg_debug.edit simplify-skeleton graph.am --seg labels.am --segment 3717 --apply --out simplified.am
```

It is a dry run unless `--apply` is given. Deviations from the paper:

- **The threshold is local.** A link qualifies when it is shorter than `--link-factor`
  (default 1) times the larger radius of the other vessels at its ends, that is,
  when it lies inside the parent lumen. The paper's automatic threshold, the thinnest
  vessel's diameter (`--auto-thinnest`), is 54 um on the left tree. It selects 8 of 3045
  inner links and none of the three above. `--min-length-um` is the paper's manual
  threshold. The report records which rule selected each link.
- **Attached segments are reshaped only as far as containment needs.** Each segment
  first gets the paper's snap: only its end point moves. If that new first edge leaves
  the segmentation, the shift is instead faded along the curve with a raised cosine
  over twice its length (at least four voxels, at most half the segment).
  `blended_segments` lists these. Neither shape works everywhere. On the left
  tree, snapping alone refused 39 of 143 clusters, and blending alone refused 58,
  because it moves a run of samples that then leaves a thin daughter. Trying the snap
  first and blending only on failure refuses 31.
- **Position fallback.** If the distance maximum (`p_edt`) still exits, the collapse
  retries at p_CA, then at the kept node's current position. `position` records
  which was used, and `exits_by_position` records why each candidate failed.
- **A collapse that exits at every candidate is refused** (`new_segmentation_exit`)
  instead of applied. So is one where a real vessel has both ends in the cluster
  (`would_create_self_loop`). Most remaining refusals are clusters of two or more links.
- `at_ball_edge` marks a p_NewC on the search boundary. In a thick vessel the distance
  maximum tends to slide towards the thickest incident branch, so review these in
  the overlays. In the 3719/3720 cluster, DFS fitting moved the node about 1 mm back
  towards p_CA.

**Ids change on save.** Amira stores vertices and edges by index, so every node and
segment after a removed link is renumbered. After simplifying 3717, the old segment 3717
is saved as 3713, and root 6107 as 6101. `--apply` prints the new id of each
`--segment` and writes the full old-to-new map under `renumbered` in the report. Map
roots, regions and `--path-plan-json` through it; a stale path plan is refused.

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
