# LADAF-2021-17 centreline qualification

## Decision

Section-centroid spline fitting is the strongest candidate in this comparison
for centring on the observed segmented lumen. It remains experimental: most
real regions did not converge within 25 iterations. No full-tree geometry
replacement or anatomical radius validation is claimed.

Locally scaled Laplacian diffusion is useful for point noise, but it did not
remove systematic centreline offsets in this experiment. The coherent section
variant avoids searching for a different plane when the fitted normal already
gives a stable section. It is a candidate for further evaluation, not a proven
winner over the ordinary centroid fit.

## Protocol and limits

The real-data benchmark used 48 regions on one heart: 24 development regions
(including segments 3717, 3655 and 3612) and 24 initially held-out regions.
Selection used seed 202117 and six radius strata. It was not stratified by
curvature. The input was `drift_full_remeasured.am`, using the graph's recorded
32.04 µm voxel scale and the arterial label segmentation. "Original" below means
the input geometry before centreline refinement. Thirteen fixed reference
planes were attempted per region, at
10–90% of its point-index range. Reference centroids were obtained from the
segmentation once, before comparing candidate curves.

Each fit used at most 16 section stations and 25 iterations. Junctions were
pinned for this comparison; real-data movement of shared junctions is therefore
not qualified by these results. The legacy baselines used isolated segments
and lack the full junction context of the new methods.

The score is the distance between a candidate's intersection with a fixed
plane and that plane's segmentation centroid. The table reports the median
of region median distances. It is a proxy for centrality, not independent
anatomical ground truth. The initially held-out set was inspected while
developing the coherent variant; this reuse is exploratory, not blinded
validation. Two held-out regions had no usable reference sections. The EDT
variant additionally failed one region. Unsupported regions and failed
methods remain recorded in the benchmark outputs.

## Real-data results

| Method | Held-out median residual (µm) | Evaluated regions / 24 | New outside edges |
|---|---:|---:|---:|
| Original geometry | 55.9 | 22 | 0 |
| Legacy recentering without smoothing | 20.2 | 22 | 10 |
| Legacy Gaussian | 23.9 | 22 | 9 |
| Legacy Savitzky–Golay | 61.0 | 22 | 20 |
| Legacy B-spline | 52.0 | 22 | 15 |
| Legacy multiscale | 22.8 | 22 | 10 |
| EDT-weighted geodesic | 23.7 | 21 | 1 |
| Centroid spline, strength 0.01 | 16.5 | 22 | 0 |
| Centroid spline, strength 0.1 | 26.4 | 22 | 0 |
| Centroid spline, strength 1 | 45.3 | 22 | 0 |
| Locally scaled Laplacian | 55.8 | 22 | 0 |
| Compensated implicit Laplacian | 54.4 | 22 | 0 |
| Coherent centroid spline, strength 0.01 | 16.5 | 22 | 0 |

There were no new reversals reported in this comparison. Only one of the 24
held-out regions converged for centroid spline at 0.01, and two converged for
the coherent variant. The strict criterion requires two successive maximum
point movements below 0.1 voxel. Reduced residual alone does not satisfy it.

| Reported segment | Original median (µm) | Centroid spline 0.01 (µm) | Coherent 0.01 (µm) | Coherent 95th percentile (µm) |
|---|---:|---:|---:|---:|
| 3717, bulge | 539.0 | 68.6 | 81.7 | 260.3 |
| 3655, constriction | 987.2 | 44.3 | 55.7 | 488.9 |
| 3612, constriction | 671.3 | 87.6 | 70.7 | 320.1 |

None of these three fits converged. Their large remaining tail residuals need
inspection, particularly near anchored endpoints. These are geometry results;
radii require remeasurement after a geometry change.

## Known-centre synthetic comparison

Six cases combine round, flat and curved flat lumens with point spacing of one
or three voxels. The voxel spacing is 10 µm; each input has a three-voxel offset
plus sinusoidal jitter, true anchored endpoints, and an underestimated 15 µm
radius. The flat cross-section has semiaxes of 10 and 3 voxels. Position scores
are transverse errors against the analytical centre away from the endpoints,
not shortest distances to the analytical curve.

| Method | Median of case median errors (voxels) | Worst case 95th percentile (voxels) |
|---|---:|---:|
| Original | 3.071 | 3.686 |
| Legacy Gaussian | 3.095 | 3.686 |
| Legacy multiscale | 3.016 | 3.664 |
| EDT-weighted geodesic | 0.000 | 5.246 |
| Centroid spline 0.01 | 0.108 | 0.531 |
| Coherent centroid spline 0.01 | 0.110 | 0.448 |
| Locally scaled Laplacian 0.01 | 3.015 | 3.651 |
| Compensated implicit Laplacian 0.01 | 3.015 | 3.684 |

All synthetic outputs retained foreground containment. The EDT aggregate hides
a failure to follow the curved flat phantom's analytical centre. These phantoms
do not represent the full range of coronary anatomy or segmentation defects.

## Optional collision correction

The separate reconstruction preparation command preserves every measured radius
and applies smooth displacement fields over vessel spans. Tests cover feasible
separation, an infeasible constrained layout, foreground containment, fixed
junctions and smooth support boundaries. One synthetic fixture also verifies
that initially intersecting circular tube meshes no longer intersect after
correction. Real full-tree reconstruction and blended SDF surfaces remain
unvalidated. Infeasible separation is reported instead of reducing radii.

## Reproduction and next acceptance checks

Run the commands in [CENTRELINE_REFINEMENT.md](CENTRELINE_REFINEMENT.md).
For the ordinary real-data comparison, select all benchmark variants except
`centroid-coherent:0.01`; run that variant in a second output directory.
The benchmark checkpoints completed regions and uses 16 independent worker
processes when `--workers 16` is supplied. Quoted aggregate region times are
not end-to-end throughput estimates; one difficult region can dominate a run.

Before promotion, resolve oscillation/non-convergence, evaluate movable
junction neighbourhoods and degree-two tangent continuity, inspect the three
reported failures against segmentation, then remeasure and validate radii.
Check an untouched region set and actual reconstructed surface collisions.
Foreground containment alone does not exclude every nonlocal centreline
crossing within a connected segmented lumen.

## Relation to established methods

[VMTK's centreline method](https://vmtk.github.io/tutorials/Centerlines.html)
uses paths on a surface Voronoi diagram with an inscribed-sphere radius metric.
Its associated radius is a maximum-inscribed-sphere radius. Our inference is
that this objective need not select the section-area centre of a flattened
lumen. The benchmark's voxel EDT route is a comparator, not an implementation
or test of VMTK itself.

[Taubin's smoothing paper](https://graphics.stanford.edu/courses/cs468-01-fall/Papers/taubin-smoothing.pdf)
addresses shrinkage using a low-pass filtering construction. The implemented
`2H-H²` implicit compensation is inspired by that goal and is not the paper's
canonical explicit lambda/mu filter.
