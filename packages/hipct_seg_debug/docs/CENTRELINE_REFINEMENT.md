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
| `laplacian` | Implicit diffusion using physical edge lengths and locally measured section scales |
| `taubin` | Shrinkage-compensated implicit Laplacian, using `2H-H²` for the implicit filter `H` |

The compensated filter is **Taubin-inspired**, not a reproduction of the classical
explicit lambda/mu algorithm. Neither diffusion variant is a recentering algorithm:
both can remove jitter while leaving a systematic offset unchanged. Local scales
vary along each segment, and arclength weighting avoids assigning more influence
to densely sampled regions. The original radius is not a permanent movement cap.

Accepted section centroids must belong to an exclusive, closed lumen. A nearby
branch is considered a rival only if its centreline actually intersects the cut
plane inside that lumen; projecting an axially displaced continuation into the
plane would incorrectly exclude short segments.

Defaults are 25 iterations, 32 sample stations per segment, strength 0.1 and a
256-voxel maximum half-window. Convergence requires two steps below 0.1 voxel.
Non-convergence, insufficient support, blocked moves and pre-existing outside
edges are reported. No new method has been promoted into `optimise-skeleton`.

## Optional reconstruction layout

```
py -3.12 -m hipct_seg_debug.edit prepare-reconstruction centred_measured.am --seg labels.am --out reconstruction_layout.am
```

Run this **after** final radius measurement and **before** surface reconstruction.
It transports those radii unchanged onto a separate smooth displacement field.
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
