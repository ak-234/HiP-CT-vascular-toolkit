# Handoff — epicardial radius: an estimator bias, and two approaches to the floor

Written 2026-08-30, revised the same day once the synthetic ground truth was run.
Continues work on perimeter-based radius correction for the LADAF-2024-28 coronary
tree. Everything below is measured on that dataset unless stated otherwise; the
estimator results are measured on analytic geometry and say so where they appear.

**What changed in the revision:** the synthetic test that section A called for has
been run. It contradicts A's stated premise — the estimator under-reads thin
sections rather than over-reading them — and it turned up a larger, separate bias
affecting nearly every measured point. That correction is now **applied and on by
default** in `radius_perimeter`, and the tree has been re-measured with it; see
"Applied to the tree". Section A, "Which to prefer", the pitfalls, the code tables
and "Still open" are all updated accordingly.

Neither approach A nor approach B has been implemented. Both are still open, and
the correction changes what A is worth — read section A before starting it.

---

## The problem to solve

Thin epicardial branches render with radii that do not follow the segmentation:
too small over a stretch, then widening where the profile resumes agreeing with
the mask.

**Cause (established, not hypothesis).** `crosssection.MIN_BLOB_VOXELS = 12`
refuses any cross-section smaller than 12 voxels. At the 65.98 µm segmentation
spacing that is an area floor of 52,240 µm², **an equivalent radius of 129 µm —
about 2 voxels**. Vessels below that cannot be measured at all, so their radii
come from interpolation or retained input rather than from the segmentation.

Evidence:

| observation | value |
|---|---|
| points carrying a radius below the 129 µm floor | **14.6%** |
| `unmeasurable` points | 3,154 (10.8%) |
| median radius *assigned* to `unmeasurable` points | **122 µm** — below the floor |
| 129 µm as p5 of every estimator in the run logs | the floor showing through |
| `unmeasurable` points actually sitting on background | **0.7%** (so there *is* vessel there) |

The centreline is inside the mask; the pass simply refuses to measure sections
that small, and the back-fill then invents a number.

---

## The estimator under-reads, and by how much

Measured 2026-08-30 on analytic discs and cylinders of known radius, over 576
sub-voxel offsets per radius, against the pipeline's own estimator
(`cv2.arcLength` on `blob4`, `crosssection._perimeter_um`). Part 3 of the script
runs the real `measure_radii` and reproduces these to three decimals, so this is
the pipeline's behaviour and not a property of the harness.

| true r | 1.5 vox (99 µm) | 2.0 (132) | 2.5 (165) | 3.0 (198) | 5.0 (330) | 8.0 (528) |
|---|---|---|---|---|---|---|
| `r_perim` / true | **0.725** | **0.815** | 0.869 | 0.906 | 0.959 | 0.996 |
| `r_area` / true | 0.995 | 1.017 | 0.984 | 0.995 | 1.003 | 1.000 |

**The mechanism is two errors of opposite sign**, and closed-form:

* `cv2` walks the **centres** of the boundary pixels, so it traces a circle of
  radius `r − 0.5`, not `r`. At r = 2 that alone is −25%.
* the traced path is a chain code, longer than a smooth one by
  `mean(cos t + (√2−1) sin t)` over `t ∈ [0°, 45°)` = **1.0548**.

`r_est ≈ 1.0548 · (r − 0.5)` fits every measured radius from 0.75 to 8 voxels to
within 0.08 voxels. It **changes sign at r = 9.6 voxels = 635 µm**: above that the
estimator over-reads, below it under-reads. Almost the whole tree is below 635 µm,
so this is a systematic under-read across the tree, not a floor artefact.

**The correction is the inverse, and it works.** `r / 1.0548 + 0.5` (in voxels;
in µm the inset is half a spacing). Both constants are derived, not fitted, and it
was checked on radii appearing nowhere in the table above:

| true r (vox) | 1.1 | 1.4 | 1.9 | 2.3 | 2.8 | 3.6 | 4.5 | 7.0 |
|---|---|---|---|---|---|---|---|---|
| raw | 0.579 | 0.710 | 0.809 | 0.830 | 0.890 | 0.906 | 0.945 | 0.996 |
| corrected | 1.003 | 1.030 | 1.030 | 1.005 | 1.022 | 0.997 | 1.007 | 1.016 |

Through the whole pass at floor 4 it takes 0.787–0.981 to 1.013–1.079. It holds on
collapsed sections to 4:1 (at r = 3, 4:1: 0.875 → 0.996) and fails at 8:1, where
the blob is down to a handful of voxels and no estimator works. Residual is a
consistent **+2%**; do not tune it away, the remaining error is the digitisation.

### Applied to the tree

`perimeter_correction` is now on by default in `measure_radii`, and the tree has
been re-measured into `analysis_out/radius_perim_corrected.am` (234 s). Coverage is
unchanged — 16,837 measured against the baseline's 16,835, both 57.8% — which is
the expected result: the correction changes what a section reads, not whether it
can be read. All 16,837 carry `radius_source = PERIMETER_CORRECTED`.

| | baseline (raw) | corrected | predicted |
|---|---|---|---|
| median radius | 257 µm | **277** | 276.6 |
| p95 | 911 µm | **897** | 896.7 |
| p99 | 1545 µm | **1497** | 1497.7 |
| max | 1795 µm | **1734** | 1734.7 |
| `r_new/r_old` p5 / median / p95 | 0.84 / 0.99 / 1.13 | 0.94 / 1.05 / 1.25 | — |

Every quantile lands within 1 µm of `r/1.0548 + 33 µm`, which is what a monotone
transform must do and is worth nothing as evidence — it confirms the arithmetic,
not the physics. **What is worth something is the sign change showing up in real
data**: the median rose 7.8% while the max *fell* 3.4%, because the tree straddles
the 633 µm crossover. A fudge factor would have moved everything one way.

The `r_new/r_old` p5 of 0.94 is the number to watch. It was 0.84 — that population
is the thin end this whole document is about, and it is no longer being written out
a sixth smaller than the mask.

Surfaces have been built from both graphs and compared; the correction costs one
topological handle in the larger tree. See "Find the handle the correction
introduced in graph 0" under "Still open" — and note that graph 0's corrected mesh
was written under a validation **warning**, so `final_surface/lumen_bspline.stl` is
a saved-but-unvalidated mesh, not a passed one.

---

## Severe collapse defeats both estimators

**This section is analytic geometry, and it is *not* the cause of the narrow
trifurcation branch in the viewer.** That was diagnosed separately and turned out to
be the junction mask — see "The junction mask discards good measurements". What
follows is a real failure mode of the estimator, established on known shapes and
worth knowing about, but nothing yet shows how much of the tree is in it.

Measured on ellipses of known perimeter, orientation and sub-voxel offset swept,
taken down to sub-voxel thickness (`subvoxel_bias.py` part 5):

| target r | aspect | thickness | voxels | admitted @12 | pinched (blob4) | raw | area |
|---|---|---|---|---|---|---|---|
| 3.0 vox (198 µm) | 4:1 | 2.20 vox | 15 | 100% | 0% | 0.875 | 0.728 |
| 3.0 | **8:1** | **1.15** | **6** | **0%** | 68% | — | — |
| 3.0 | 16:1 | 0.59 | 2 | 0% | 27% | — | — |
| 5.0 (330 µm) | 8:1 | 1.92 | 23 | 100% | 29% | 0.896 | 0.541 |
| 5.0 | **16:1** | **0.98** | **4** | **7%** | 79% | 0.828 | 0.437 |
| 8.0 (528 µm) | 16:1 | 1.56 | 28 | 100% | 70% | **0.785** | 0.373 |

**Three failures, and they compound.**

1. **`MIN_BLOB_VOXELS` is an area floor, but the pass measures perimeter.** A
   collapsed lumen is small-area and large-perimeter *by construction*, so an area
   floor preferentially refuses exactly the sections the perimeter estimator was
   chosen to handle. A 198 µm branch at 8:1 has six voxels of area and is refused at
   every orientation and every offset — 0% admitted. It is then back-filled, which
   is what puts a narrow stretch next to a correct one. **This is the single
   clearest mechanism for the opening symptom yet measured.**
2. **Below roughly two voxels of lumen thickness the estimator degrades even when
   admitted** — 0.785 at 16:1, worse than any round section at any size, because
   tracing pixel *centres* through a two-voxel-thick blob leaves almost no enclosed
   width to trace.
3. **4-connectivity fragments the slit** in 68–79% of orientations at those aspect
   ratios, and the fragments fall under the floor. See the amended `blob4` pitfall.

**Neither estimator is the answer here.** Area reads 0.37–0.54 on the same sections
— it is not a fallback, it is a different failure. Below ~2 voxels of thickness the
mask does not carry a measurable cross-section at all, and no choice of estimator
recovers one. The honest options are to *mark* those points rather than measure them
(approach B), or to accept that the segmentation's resolution bounds what can be
said there.

**What this does not yet establish** is how much of the real tree is in that regime.
The numbers above are analytic geometry. The measurement that would settle it is the
distribution of minor-axis thickness in voxels over the tree's own sections —
`crosssection.measure()` already computes `minor_ratio` and `isoperimetric` per
point, so the data is one pass away and has not been looked at.

---

## The junction mask discards good measurements

**This is the cause of the symptom this document opens with**, diagnosed 2026-08-30
on segments 265 and 268 of `radius_perim_corrected.am` — the narrow stretch at the
rightmost trifurcation of the left tree. Reproduce with
`python research_scripts/segment_diagnosis.py 265 268 270`, which reads each point's
`radius_source` and `radius_reject_reason` back out of the .am and re-cuts its
section, so the verdict is the pass's own and not a reconstruction.

| | segment 265 | segment 268 |
|---|---|---|
| points measured | **0 / 19** | **0 / 9** |
| reject reason, every point | `junction` | `junction` |
| radius written | **269 µm, constant** | **222 µm, constant** |
| section area | 326–444 voxels | 341–470 voxels |
| minor axis | **19–21 voxels** | ~20 voxels |
| `blob8/blob4` | 1.00 (no fragmentation) | 1.00 |
| `r_area` from the mask | **670–780 µm** | **690–760 µm** |
| `stable_transverse_cut` succeeds | **18 / 19 points** | 7 / 9 |
| length, end node degrees | 1994 µm, 3 and 3 | 797 µm, 3 and 3 |

**The written radius is about a third of the vessel's actual calibre**, and none of
the usual suspects is responsible: these sections are 27× above the area floor, ten
times thicker than the collapse regime, essentially round, and `blob4` is not
splitting anything. Nearly every point yields a stable, closed cross-section. The
pass measured them and then threw the measurements away.

**Mechanism, confirmed by measurement** (`research_scripts/junction_mask_confirm.py`,
which rebuilds the pass's own `_BranchContext` and asks it the same question at the
same points, using the **input** graph because `rivals()` scales its search by the
radius it is handed):

| | segment 265 | segment 268 |
|---|---|---|
| `stable` | **19/19** | **9/9** |
| `adjacent_overlap` | **19/19** | **9/9** |
| `exclusive` (`stable and not adjacent_overlap`) | **0/19** | **0/9** |
| longest run of consecutive exclusive sections | **0** | **0** |
| `_adaptive_junction_mask` masks | 19/19, runs 1994 µm ×2 | 9/9, runs 797 µm ×2 |

`_adaptive_junction_mask` walks inward from each degree-3 end node masking points
until it meets **two consecutive** exclusive sections; if it never finds two in a
row, the loop's `else` branch commits the lot. Every point here is stable and every
point is topology-adjacent to a rival, so nothing is ever exclusive, both walks run
the full length, and the whole segment is masked. The adjacent rivals of 268 are 265
and 270 — **these are not three independent failures but one trifurcation cluster
whose members disqualify each other.**

**Then the fill makes it worse.** With `fallback_policy=drop` the segment takes a
constant `sqrt(lo·hi)` from its junction anchors, and the anchors are whichever
neighbours *were* measured — which are the thin distal branches, because those are
the ones far enough from a junction to survive the mask. Segment 265 meets 161 µm
and 167 µm daughters, and a 735 µm trunk that is itself only 22/57 measured. So a
~700 µm trunk is filled from ~165 µm daughters.

**Tree-wide: 66 of 309 segments (21%) have zero measured points, every one
rejected as `junction` — 1,679 points over 166.9 mm.** Only 28 of those have a
degree-3 node at *both* ends, so one junction end is often enough to consume a whole
branch. Each was re-measured independently on `reformat` planes
(`research_scripts/junction_damage.py`; 4 could not be measured at all and are genuinely
below what the mask resolves):

| min usable planes | segments | written/measured p5 / median / p95 | wrong by >25% |
|---|---|---|---|
| any | 62 (163.2 mm) | 0.36 / 0.93 / 2.95 | **79%** (25 under, 24 over) |
| ≥ 10 | 51 (139.9 mm) | 0.35 / 0.91 / 2.66 | **78%** (20 under, 20 over) |
| ≥ 20 | 22 (75.4 mm) | 0.52 / 0.75 / 2.25 | **82%** (10 under, 8 over) |

**The error is two-sided, and that is the important part.** An earlier draft of this
section said "large short segments get their calibre from small long ones" — true of
265 and 268, but it does not generalise. The median ratio sits near 1.0 not because
the fill is usually right but because its errors cancel in aggregate: roughly as many
segments are written a third too wide as a quarter too narrow. **The junction mask
does not bias the radius, it randomises it.** Four fifths of wholly-masked segments
carry a calibre wrong by more than 25%, in either direction, and the tree-wide
summary statistics hide that completely.

The comparison is against a fresh measurement from the segmentation, not against the
input graph — the input is what this pass exists to distrust (p99 1740 µm, max
4348 µm), so a large written/input ratio could as easily be a bad input as a bad
output. The input is carried as a third column only.

So this is not a passive gap. It actively assigns wrong radii, from the wrong
vessels, to a fifth of the tree.

### Fixed: the mask is now bounded

`JUNCTION_MASK_MAX_FRACTION = 0.4` caps what one junction may consume of a segment,
per end, so at least a fifth of every segment stays measurable however the
exclusivity test votes. It is a ceiling, not a target — where the two-in-a-row rule
already stops the walk early, the mask is bit-identical to before. Re-run into
`analysis_out/radius_perim_capped.am`.

| | before | after |
|---|---|---|
| segments wholly masked as `junction` | 66 (166.9 mm) | **2 (10.7 mm)** |
| measured points | 16,837 (57.8%) | **17,108 (58.7%)** |
| `junction` rejections | 7,702 | **6,701** |
| segments with no trustworthy measurement | 66 | **36** |
| segment 265 / 268 / 270 | 269 / 222 / 522 µm | **763 / 752 / 828 µm** |
| their independent targets | — | 788 / 774 / 853 µm |

Scored against the independent `reformat` measurements of all 62 measurable
previously-wholly-masked segments:

| | before | after |
|---|---|---|
| wrong by >25%, all 62 | 82% (141.8 mm) | **47% (83.5 mm)** |
| wrong by >25%, ≥20 usable planes | 91% | **32%** |
| median \|log ratio\| | 0.529 | **0.248** |

**It is not uniformly better, and the split matters.** 30 segments improved, 22 are
unchanged, 10 got worse. The improvements are about four times the size of the
regressions — median |log| change −0.371 against +0.088, totals −12.61 against
+1.22 — so the aggregate halves, but a tenth of the set moved the wrong way. The two
worst regressions (segments 43 and 300) rest on 3 and 7 usable planes respectively,
so their targets are the least trustworthy in the set; segments 209 and 182 moved
modestly further and have no such excuse.

**Two costs, both expected.** `unmeasurable` rose 3,154 → 3,794 and unstable
tangents 1,427 → 1,513: points the mask used to hide now reach the other gates and
some fail there. That is the honest accounting rather than a regression — a point
refused by a gate that examined it is not the same as a point discarded unexamined.
Max radius rose 1734 → 1951 µm and p95 897 → 930 µm; the median is unchanged at 277.

Still wholly masked: segments 276 and 297. 297's independent target is 654 µm
against 234 written, so the bound does not reach every case.

**The surface got topologically worse, and the trend is monotone.** Built from the
capped graph into `analysis_out/final_surface_capped/`:

| | raw baseline | corrected, unbounded mask | capped |
|---|---|---|---|
| graph 0, 130 segments | genus **0**, passed | genus **1**, warn | genus **3**, warn |
| graph 1, 119 segments | genus 0, passed | genus 0, passed | genus 0, passed |
| graph 0 mesh | 830,792 v / 2,509,228 f | 770,354 v / 2,305,443 f | 791,660 v / 2,405,396 f |
| graph 1 mesh | 724,438 v / 2,233,374 f | 714,490 v / 2,134,006 f | 708,952 v / 2,117,721 f |

Each step that made the radii more faithful to the segmentation added handles to the
left tree: 0 with the raw estimator, 1 with the digitisation correction, 3 with the
junction bound as well. Both trees stay manifold and single-component throughout, and
graph 1 is untouched at genus 0.

**The obvious reading is that this is the price of correctness, not a regression in
the radii** — a branch restored from 269 µm to 763 µm is three times wider, and at a
trifurcation three such branches converge, so the SDF fuses lumens that previously
passed each other with room to spare. If the vessels really are that wide and really
do touch, a handle is the anatomically honest outcome and it is the surface
pipeline's merge handling that needs attention, not the calibre.

**That reading is untested.** Nobody has located a single one of the three handles.
The alternative — that some restored radius is simply too large — is not excluded by
anything measured so far, and `p95` rose 897 → 930 µm and `max` 1734 → 1951 µm under
the bound. Find the handles before accepting either story.

**Three independent plane constructions agree on the true calibre, and the pass
agrees with none of them.** `reformat` builds planes a different way — resample to
uniform arclength, smooth until the plane stack is provably collision-free, then
carry a parallel-transport frame, so a plane's normal depends on the whole path
rather than on its immediate neighbours. Run at `mode="native"`, which pins one
output pixel to one segmentation voxel so the estimator has the same staircase to
trace (`research_scripts/reformat_radius.py`):

| | segment 265 | segment 268 |
|---|---|---|
| `reformat` planes, r_perimeter corrected | **788 µm** | **774 µm** |
| `reformat` planes, r_area | 698 µm | 692 µm |
| `crosssection.cut` re-measure, r_area | 670–780 µm | 690–760 µm |
| input graph, `flagged_recentred.am` | 790 µm | 770 µm |
| **written out** | **269 µm** | **222 µm** |
| ratio written / measured | **0.34** | **0.29** |

Every plane was usable; none truncated, none with its centre off the mask. Note the
correction barely moves these (796 → 788 µm), because at ~12 voxels they sit just
above the 9.6-voxel crossover — an incidental check that its sign is right.

**The pass made the radius worse than the input it exists to distrust.** The input
graph already carried 770–790 µm here; `r_new/r_old` for these segments is ~0.3.

**This outranks everything else in this document.** It is larger than the estimator
bias already corrected (a third of true calibre, against 10–25%), it has a concrete
reproduction, and it explains the opening symptom directly.

---

## The perimeter-conservation assumption, tested

The premise under `radius_perimeter` is that a lumen collapses when the pressure
goes — the wall folds, the enclosed *area* falls, the wall's *length* is preserved —
so `perimeter/2pi` still reports the in-vivo calibre and `sqrt(area/pi)` does not. It
had been asserted throughout and never tested. Tested 2026-08-31
(`research_scripts/collapse_conservation.py`, `research_scripts/shrinkage_scale.py`).

**It cannot be tested by comparing the estimators against collapse severity.** With
`Q = P^2/(4*pi*A)`, `r_area / r_perim = Q^(-1/2)` *exactly*, for every section. Any
regression of one against the other on Q recovers an identity and would "confirm" the
hypothesis on random noise. Two external references avoid it.

### Relative accuracy: both references favour area

**Longitudinal smoothness.** A vessel's calibre varies smoothly along its own length;
collapse varies section to section. Median |step| in log radius, over 289 segments:
r_perimeter **0.0663**, r_area **0.0430** — perimeter smoother in only 4% of
segments. The control is the part that matters:

| segments containing | n | perimeter smoother in |
|---|---|---|
| almost no collapsed sections | 80 | 2% |
| some | 121 | 4% |
| mostly collapsed | 88 | 5% |

**The gap does not widen with collapse.** It is already there on near-circular
vessels and stays flat, which is the signature of the perimeter estimator being
noisier, not of area being corrupted. Were the hypothesis operating, perimeter would
get relatively smoother exactly where collapse is present.

**Murray's law**, 123 degree-3 nodes, `sum(daughter^3)/parent^3`:

| | median | median \|log\| | within 25% |
|---|---|---|---|
| r_perimeter | 0.629 | 0.473 | 25% |
| r_area | **0.715** | **0.346** | **31%** |

Area is closer at 70% of junctions. Both sit well below 1.0 — daughters consistently
sum to less than the parent, most likely because the segmentation misses small side
branches — so only the comparison means anything, not the absolute fit.

### Absolute scale: perimeter is better centred

After correcting the estimator's own digitisation bias (**required**: raw, the uplift
is dominated by centre-tracing and reports `Q < 1` on small vessels, which the
isoperimetric inequality forbids), `r_perim / r_area` is median **1.117**, and it does
*not* trend with vessel size — 1.105 in the largest decile against 1.082 in the
smallest. It behaves more like a scale factor than expected. But it still spreads
1.34x between segments (p5 1.041, p95 1.392), because it is driven by section shape:
as a shrinkage correction it would add 4% to some vessels and 39% to others for
reasons unrelated to shrinkage.

The five largest segments measure **2.78 mm** (area) against **3.16 mm** (perimeter)
in diameter, where in-vivo proximal LAD is roughly 3.0–3.9 mm. Perimeter lands inside
the range and area below it. **Caveat: nobody has identified which vessel these
segments are.** If the largest is left main (4.0–5.0 mm) rather than proximal LAD,
both estimators are far too small and this reads differently.

### What follows

The two results are not in conflict — they measure different things. Perimeter is
better *centred* and noisier *locally*; area is better behaved and systematically
~12% low. That points at an option neither had been arguing for: **take the area
radius and apply an explicit global shrinkage factor.** It keeps the two error types
separable — one a measurement, the other a stated constant that can be revised — and
folding shrinkage into the choice of estimator hides it inside a shape-dependent
quantity where it cannot be inspected.

**What none of this settles.** The physiological claim is about tissue, and at 66 µm
the folds of a collapsed wall are sub-voxel: `cv2` traces the outline of the
digitised blob, not the path of the folded wall, so the quantity said to be conserved
is not the quantity being measured. Consistent with "Severe collapse defeats both
estimators", where perimeter reads 0.16–0.47x true on genuinely flattened sections.
Settling it needs a **pressurised-versus-unpressurised scan of the same specimen**;
short of that, higher-resolution imaging of one collapsed segment would at least say
whether the folds are resolvable in principle.

This bears on `GATE_VOXELS = 0`, which disabled the area estimator on the grounds
that area under-reads collapsed lumens. That reasoning is now contradicted from two
directions — here and in the synthetic work — and every surface downstream inherits
the choice.

---

## Junctions: what can and cannot be measured there

Three things were established 2026-08-31 while trying to make the estimator
branch-aware at bifurcations. Two are dead ends, recorded so they are not re-derived.

**Trifurcations are not a separate problem.** Node degrees across the tree are
{1: 161, 3: 144, 4: 5, 5: 1} — **six** nodes of degree ≥ 4 against 144 bifurcations.
The ownership watershed is already N-ary, so a trifurcation is only more markers.
There is a *junction* problem, at all 150.

**The ownership machinery was structurally unable to fire at a junction, and now
can.** `_resolve_owned_cut` separates fused lumens with a marker watershed, but was
only ever handed rivals that are *not* topology-adjacent. The reason turned out to be
mechanical: `_raster_markers` aborted on marker collision, and branches sharing a
node share that node's coordinates exactly, so adjacent branches *always* collide.
Seeding is now two-pass — a voxel claimed by one branch seeds it, a contested voxel
is left for the watershed — which is the right treatment, since contested territory
is not a contradiction. Segment 265 went 0/19 to 6/19 resolvable, 9/19 once the slab
offset is bounded by distance-to-node rather than radius alone (on segment 268, 797 µm
long with a 770 µm radius, a ±0.5r slab spans 771 µm and straddles both junctions).

**But nearest-centreline ownership is the wrong instrument here — measured, not
argued.** Walking segment 265 away from its node the owned radius reads 591, 618,
642, 671, 692 µm: smallest *at* the junction. And 265 is the **parent** at both its
nodes, so this is a parent narrowing before a bifurcation, which is anatomically
wrong. The partition divides contested lumen and charges both parties for it. It is
well-posed only between *siblings*, where the equidistant locus is a real carina;
against a parent it invents a boundary. Ownership is therefore **not** wired into
`measure_radii`: enabling it would pinch parents tree-wide, worse than masking.

**A relative-calibre gate on `adjacent_overlap` does not work.** The idea — a 161 µm
daughter should not disqualify a 790 µm parent's section — is measured and false:
over 486 points, relaxing the gate from 0.00 to 1.00 moves measurable points 317 to
333, and with the mask bound removed 265/268/270 still yield **zero**. Adjacent
rivals of comparable or larger calibre sit at essentially every junction point, so
**the exclusivity test is unsatisfiable near a junction** however it is tuned, and
`JUNCTION_MASK_MAX_FRACTION` is the only thing that ever stops the walk.

### What was done instead: parent-through, without a carina

A through-vessel cannot be *measured* across a node, so the remaining lever is what
fills the run. `_apply_bifurcation_tapers` already authored both a parent trend and a
daughter carina taper, welded to one flag. They are now separable, because only one
is an extrapolation of something measured:

* `junction_parent_profile` (**on**) carries a parent's — and a same-calibre
  continuation's — own log-linear trend across its junction run. It says nothing the
  vessel did not already say on both sides of the node.
* `bifurcation_tapers` (**off**, unchanged) is the daughter carina model, and its
  sign is wrong for an ostium: it narrows the daughter to `carina_tip_factor` **at**
  the node and widens it distally, where an ostium is widest at the parent.

Measured on the tree (`analysis_out/radius_perim_parent.am`): parent-through taper
**2,147** points, through-junction continuation **2,276**, daughter-emergence taper
**0**; ordinary interpolation 11,068 to 6,645. Measured coverage is unchanged at
17,108 (58.7%) — this changes only how un-measured runs are filled. Of 181 segments
with an authored run of 3+ points, the run was perfectly flat before on **83** and
after on **10**: junction spans now follow their vessel instead of holding the last
value. Max radius 1951 to 2048 µm, p99 1496 to 1473.

**This is still authored, not observed**, and `radius_resolution_mode` says so per
point. The claim is only that extrapolating one vessel's own measurements across a
gap is weaker than inventing a carina — not that it is a measurement.

The surface is built (`analysis_out/final_surface_parent/`), and **parent-through
added no handles**:

| | raw | corrected | capped | parent-through |
|---|---|---|---|---|
| graph 0, 130 segments | genus 0 | genus 1 | genus 3 | **genus 3** |
| graph 1, 119 segments | genus 0 | genus 0 | genus 0 | genus 0 |
| graph 0 vertices | 830,792 | 770,354 | 791,660 | 826,883 |

Genus held at 3 even though max radius rose 1951 → 2048 µm, so the left tree's
handles came from the radius corrections that preceded this and not from carrying
the trend through. Graph 0's mesh is still written under a validation **warning**;
`final_surface_parent/lumen_bspline.stl` is saved-but-unvalidated, not passed.

### The ostium flare is real, and it is being thrown away

Measured over 261 segments with a branched end, **each normalised by its own
interior** so vessel calibre cancels (`research_scripts/ostium_flare.py`):

| distance to node (local radii) | segments | r_perim / own | aspect / own |
|---|---|---|---|
| 0.0–0.5 | 183 | **1.32** | 1.13 |
| 0.5–1.0 | 185 | 1.21 | 1.08 |
| 1.0–1.5 | 189 | 1.11 | 1.00 |
| 1.5–2.0 | 176 | 1.07 | 0.96 |
| 2.0–3.0 | 215 | 1.05 | 0.97 |

The section really is **32% wider** at under half a radius from a branched node,
decaying smoothly to 5% by three radii — an ostium, with the shape the anatomy
predicts. And it is **mostly round**: the aspect ratio rises only 1.13x, which for a
typical interior aspect of 1.4 puts the minor axis up ~23% and the major ~39%. An
isotropic radius carries most of it. Ordinary interpolation discards all of it.

**Normalise within segment or this measurement lies.** Binning raw distance-to-node
across the tree compares near-node *proximal* sections against far-from-node *distal*
ones and reports a 2.2x flare with the aspect ratio moving the wrong way, because the
far bin is a different vessel population (median 283 µm, where few-voxel sections make
the axes noisy).

### `--junction-flare`: who may report the shared lumen

The section is wider near a node *because the lumens are continuous there*, so each
branch measures the shared region. `junction_flare` decides who keeps it: `none`
(default, previous behaviour), `parent` (the parent at that node only), `all`.

| | `none` | `parent` | `all` |
|---|---|---|---|
| measured points | 17,108 (58.7%) | 18,890 (64.9%) | **21,110 (72.5%)** |
| `junction` rejections | 6,701 | 4,919 | **2,649** |
| max radius | 2048 µm | 2676 µm | 2676 µm |
| graph 0 genus | 3 | **4**, + 8 self-intersections | **1** |
| graph 1 genus | 0 | 0 | 0 |
| graph 0 faces | 2,461,572 | 1,682,734 | 2,477,306 |

**A prediction was made and it failed.** The expectation was that `all` would bulge
at junctions, since an N-way node contributes the shared region N times to the union.
On genus `all` is the *best* of the three and `parent` the worst — the only build so
far with self-intersections. **Genus cannot adjudicate this**: inflating radii can
*close* a gap between two near-touching branches, which lowers genus while producing
exactly the fused blob the prediction was about. So the metric neither confirms nor
refutes it, and this needs an eye on the mesh.

The gain that does not depend on the surface is coverage: under `all`, ~4,000 points
that were being invented by interpolation now carry a measured radius.

`--junction-flare` defaults to `none`, so nothing moved without being asked. Note
`parent` has 32% fewer faces than the other two at similar vertex count, because
larger radii coarsen the adaptive cells — some of any visible difference is
resolution, not geometry.

---

## The two approaches to test

### A. Lower the floor — measure the thin vessels

Drop `MIN_BLOB_VOXELS` (`packages/hipct_seg_debug\crosssection.py:61`) from 12 to
roughly 5–6, so sections down to ~83–91 µm equivalent radius are measured.

**The predicted cost does not exist. Measured, and it is the opposite sign.**
This section previously said a perimeter traced around a 4–8 voxel blob is
staircase-dominated and *over*-reads, so thin sections should come back too wide.
They come back **too narrow**. See "The estimator under-reads" above for the
numbers and the mechanism; `research_scripts/subvoxel_bias.py` is the test and
`research_scripts/subvoxel_bias.log` the run.

That changes what A is worth. Admitting sub-floor sections *does* work — at
`MIN_BLOB_VOXELS = 4`, sections of true radius 1.5 voxels are admitted at every
sub-voxel offset — but they read 0.73–0.79× true. Sub-floor points are currently
back-filled at a median 122 µm (≈ 1.85 voxels), so measuring them would return
something **smaller** than what is filled in now. **A alone makes the opening
symptom worse, not better.** Lower the floor only together with the correction.

Sweep: `MIN_BLOB_VOXELS` in {4, 6, 8, 12} — the CLI does not expose it, so thread
the parameter through `radius_perimeter.measure_radii(min_blob_voxels=...)`, which
**already accepts one** (`edit/radius_perimeter.py:886`). Prefer the parameter; do
not edit the constant for a sweep.


### B. Keep the floor — stop back-filling below it

Leave `MIN_BLOB_VOXELS` alone and change what happens to points it rejects.
Currently they are filled by `_fill_gaps` and end up indistinguishable from
measurements in the surface. Instead, keep them honestly marked and let the
consumer decide.

Two variants worth separating:

- **B1 — provenance only.** Points below the floor keep a radius (the surface
  needs one) but the `radius_source` / `radius_resolution_mode` fields say plainly
  it was not measured. Much of this plumbing exists; check whether
  `INPUT_FALLBACK` vs `INTERPOLATED` is already distinguishing them correctly for
  sub-floor points, since the numbers above suggest sub-floor points are currently
  labelled `INTERPOLATED` like any other gap.
- **B2 — refuse to extend.** Do not interpolate *across* a sub-floor run from
  thicker bracketing neighbours, which is what pulls the profile away from the
  mask. Hold the last measured calibre, or let the branch terminate. This is the
  same argument that made `fallback_taper` default to False.

**Success criterion for B** is not a better radius — it is that no point claims a
measured radius it never had, and that the surface stops widening across
stretches nothing supports.

### Which to prefer

They are not exclusive. A raises the fraction of points that are genuinely
measured; B makes the remainder honest. Test them independently so each one's
effect is attributable.

**But the synthetic test has added a third thing that comes before both, and is
larger than either.** The estimator under-reads every measured point below 635 µm,
which is nearly all of them — that is a bias on the points the pipeline is
*already* confident about, not on the sub-floor remainder. Order of work:

1. **Apply the correction** and re-run. This is the only change so far whose effect
   is known in advance to within 2%, and it touches ~58% of points rather than the
   14.6% below the floor.
2. **Then A**, with the floor the corrected estimator can support. Correction and
   floor interact: at 1.5 voxels the raw read is 0.73× and the corrected read
   1.08×, so the floor that is defensible moves once the correction is in.
3. **Then B**, for whatever still falls below it.

Doing A before the correction moves the thin end in the wrong direction — see A.

---

## How to run things

Voxel spacing 65.98 µm; segmentation dims (1500, 1250, 1250); 4,589,552 foreground
voxels in **55 disconnected components**.

```powershell
# Inputs
$RAW = "%DATA_DIR%\perimeter radius test\ASCII_smooth_thick_adj_LADAF_2024_28.Spatial-Graph.attributegraph.am"
$SEG = "%DATA_DIR%\clean_inference_segmentation_LADAF_2024_28_cropped.filtered.resampled 1.am"

# 1. flag the unsampled jumps (needs --seg explicitly, see pitfalls)
python -m hipct_seg_debug.edit flag-interpolation $RAW --seg $SEG --out %HIPCT_OUT%\flagged.am

# 2. re-centre onto the lumen centroid  (~36 s)
python -m hipct_seg_debug.edit optimise-skeleton %HIPCT_OUT%\flagged.am --no-deloop --no-prune --smoother none --recentre-passes 3 --out %HIPCT_OUT%\flagged_recentred.am

# 3. measure  (~234 s). The digitisation correction is on by default;
#    add --no-perimeter-correction to reproduce the raw-estimator baseline.
python -m hipct_seg_debug.edit radius-perimeter %HIPCT_OUT%\flagged_recentred.am --fallback-policy drop --out %HIPCT_OUT%\radius_perim_capped.am

# 4. surface
python -m coronary_sdf %HIPCT_OUT%\radius_perim_corrected.am %HIPCT_OUT%\final_surface

# view (the console script is not on PATH; use the module)
python -m hipct_seg_debug --graph %HIPCT_OUT%\radius_perim_corrected.am
```

Current best graph: **`analysis_out\radius_perim_parent.am`** (correction on, junction
mask bounded, parent trend carried through). Previously
**`analysis_out\radius_perim_capped.am`** (correction on, junction
mask bounded). `analysis_out\radius_perim_corrected.am` is the same run with the
unbounded mask, kept as the before-column of the table in "Fixed: the mask is now
bounded".
`analysis_out\radius_perim_only.am` is the same run with the raw estimator, kept as
the baseline the table above compares against — do not overwrite it.
A surface run from `radius_final.am` was left running and may not have completed.

### Tests

```powershell
cd packages/hipct_seg_debug ; python -m pytest edit/tests/test_radius_perimeter.py edit/tests/test_interpolation.py edit/tests/test_interpolation_pipelines.py -q
cd packages/coronary_sdf   ; python -m pytest test_jump_split.py -q
```
Last run: 38 + 40 + 5 passing. Full `edit/tests` is 1,242 passing with one
pre-existing failure in `test_radius_circles.py` — see "Still open".

### The synthetic ground truth

```powershell
cd packages/coronary_sdf ; python research_scripts\subvoxel_bias.py   # ~3 min, last run in research_scripts\subvoxel_bias.log
```

Four parts: the estimator alone on rasterised discs over 576 sub-voxel offsets;
what each candidate `MIN_BLOB_VOXELS` admits and what the admitted sections read;
collapsed sections as ellipses of known perimeter, orientation and offset swept;
the whole `measure_radii` pass on the same tubes; then the correction, on radii
held out from the rest. It needs no dataset — the answers are closed-form — so it
is the cheap place to settle an estimator question before spending 230 s on the
tree. Note `test_radius_perimeter.py` asserts only that perimeter errs *more* than
area on a thin section, not in which direction, so it passes either way.

---

## State of the code (changed this session)

### `packages/hipct_seg_debug`

| file | change |
|---|---|
| `crosssection.py` | unchanged — `MIN_BLOB_VOXELS = 12` is the thing under test |
| `edit/radius_perimeter.py` | `GATE_VOXELS` 3.0 → **0.0** (area estimator off, every point perimeter); `fallback_policy` ∈ {retain, rescale, drop}; fallback resolution moved to a **second pass** after the measuring loop; `_junction_anchor_radii(..., exclude)`; `fallback_taper=False` (constant fill, not a log-linear ramp); **`perimeter_correction=True`** with `CHAIN_CODE_FACTOR` / `correct_perimeter_radius()` and a fourth `radius_source` value `PERIMETER_CORRECTED`; module docstring's "perimeter over-states a thin section" claim corrected; **`JUNCTION_MASK_MAX_FRACTION = 0.4`** bounding `_adaptive_junction_mask` per end; **`junction_parent_profile=True`** split from `bifurcation_tapers` via `daughter_carina`; **`junction_flare`** ∈ {none, parent, all}; `_raster_markers` two-pass unambiguous seeding |
| `edit/interpolation.py` | new `candidate_jumps()` — the mask-free half of the jump gate; `detect_jumps` now calls it so there is one definition |
| `edit/__main__.py` | `--fallback-policy`, `--fallback-taper`, **`--no-perimeter-correction`**, **`--junction-mask-max-fraction`**, **`--no-junction-parent-profile`**, **`--junction-flare`**; `flag-interpolation` and `_interpolation_hint` now report jump candidates without `--seg` |
| `edit/tests/test_radius_perimeter.py` | 28 → **38 tests**. Tests that pin the *raw* estimator now pass `perimeter_correction=False` explicitly; `test_perimeter_overstates_a_thin_section...` renamed to `..._misses_a_thin_section_by_more_than_area_does` and now asserts the direction, which it never did; new `test_a_centred_tube_of_integer_radius_is_not_typical` pins the lattice-aligned trap so nobody re-tunes the constants against it |

`bifurcation_tapers` defaults **False** (carina model unverified). All taper
counters read 0 in the current output — confirmed, not assumed.

### `packages/coronary_sdf`

| file | change |
|---|---|
| `config.py` | `BRIDGE_CENTERLINE_GAPS = False`; `SPLIT_UNSAMPLED_JUMPS = True`; `SPLIT_JUMP_STEP_RATIO = 5.0`; `MIN_COMPONENT_LENGTH_MM = 5.0`; `MIN_COMPONENT_LENGTH_FRACTION = 0.25`; all mirrored into `SdfConfig` |
| `centreline_reconnection.py` | new `split_unsampled_jumps()`, `drop_small_components()` |
| `topology.py`, `pipeline.py` | re-exports and wiring |
| `test_jump_split.py` | new, 5 tests |
| `research_scripts/subvoxel_bias.py` | new — the synthetic ground truth for the estimator; analysis only, changes no pipeline behaviour |
| `research_scripts/segment_diagnosis.py` | new — per-point verdict and section geometry for named segments, read back out of the .am. Bound the window growth: unbounded, a mis-perpendicular plane becomes a streak *along* the vessel and reports thousands of voxels |
| `research_scripts/junction_mask_confirm.py` | new — rebuilds the pass's `_BranchContext` to measure `stable` and `adjacent_overlap` per point. Must be run against the **input** graph; `rivals()` scales its search by the radius handed to it |
| `research_scripts/reformat_radius.py` | new — perimeter radius on `reformat`'s parallel-transport planes, an independent plane construction. `mode="native"` so one pixel is one voxel and the estimator sees the same staircase |
| `research_scripts/junction_damage.py` | new — re-measures every wholly-junction-masked segment on reformat planes. A 1–2 voxel blob traces a zero-length contour, so those report 0 and are counted separately rather than averaged in |

Effect: 18 jumps cut instead of bridged, 18 disconnected components (457.9 mm)
dropped, 2 trees kept (130 and 121 segments).

---

## Pitfalls — things already tried that did not work

Read these before re-deriving them.

- **Do not characterise the estimator on a centred disc of integer radius.** That
  is the most lattice-friendly configuration there is — the boundary runs straight
  along the axes at the four cardinal points — and it is the *only* configuration
  in which the perimeter estimator over-reads at small size. It is where the
  `radius_perimeter` docstring's "5.25 vs 5.08 on a radius-5 disc" comes from; the
  figure reproduces exactly, and is unrepresentative. Swept over sub-voxel offsets
  the same radius reads **4.80**. Always sweep the offset: a vessel axis does not
  pass through voxel centres.
- **`research_scripts/oblique_synthetic.py` does not test this pipeline's estimator.**
  It exercises `skeleton_analysis.outlier.oblique`, which takes its perimeter from
  skimage `regionprops`. The pipeline uses `cv2.arcLength` on the 4-connected blob.
  The two agree on a clean disc but are different code. `subvoxel_bias.py` calls
  `crosssection._perimeter_um` directly, and part 3 calls `measure_radii` itself.
- **`blob4` vs `blob8` is not the cause of thin readings — among sections that were
  accepted.** Measured over 907 *accepted* sections: median ratio **1.000**, only
  1.0% differ by >5%. That result stands where it was taken and is not worth
  re-deriving. **It does not generalise to the refused population, and the
  difference matters**: at 8:1–16:1 collapse, 4-connectivity fragments the slit in
  68–79% of orientations, and the fragments then fall below `MIN_BLOB_VOXELS` and
  are refused. A section split small enough to be rejected could never appear in a
  survey of accepted sections, so the original test was structurally unable to see
  this. See "Severe collapse defeats both estimators".
- **A 6×-radius re-cut reporting "99% of unmeasurable points have no lumen" was a
  bug in the test**, not a finding. Sampling the mask directly at those coordinates
  shows 0.7% on background. Do not conclude the centreline leaves the vessel.
- **`--fallback-policy rescale` is a no-op on this data.** Bit-identical to
  `retain`, because the global `r_new/r_old` calibration is 1.00. It can only fix a
  scale error, and these radii are right on average and wrong locally.
- **Point-sampled background tests miss the jumps.** The bridging segments carry
  *no interior points*, so any per-point check is blind to them. Resample the
  polyline densely (0.5 voxel) or use `candidate_jumps()`.
- **`flag-interpolation` needs `--seg` passed explicitly** to confirm jumps
  against the mask; the default does not trigger it. It now reports candidates
  without the mask, so it no longer looks clean when it is not.
- **There is now one `skeleton_analysis`.** Before the monorepo there were two
  checkouts, and pip's editable install pointed at a stale March snapshot while
  `import skeleton_analysis` resolved the newer one — editing the wrong copy cost
  a cycle and the tests still passed. `packages/skeleton_analysis` is the
  surviving copy; the stale snapshot was not carried over.
- **coronary_sdf does not read hipct_seg_debug's flags** (`avizo_interpolated`,
  `unsampled_jump`). It has its own gate. If you add a flag in one repo, wire it in
  the other or it does nothing.
- **Topology metrics cannot see a smooth bulge.** Self-intersection and genus
  counts will call a surface with large spherical blobs "clean". Look at it.

---

## Context worth keeping

Earlier findings this work rests on, all measured:

- **Re-centring is the single biggest win so far.** TEASAR places the skeleton
  toward an ellipse focus, not the centroid. Three damped `recentre` passes moved
  the previously-rejected population from a median centroid offset of 0.63 r to
  0.22 r (fraction over the `max_centroid_radii = 0.5` gate: 75% → 7%). Accepted
  measurements 12,327 → 16,835; unstable-tangent rejections 4,609 → 1,427;
  measured coverage 42.3% → 57.8%; runtime 1073 s → 229 s.
- **The Avizo graph bridges disconnected segmentation.** 55 mask components, 24
  point-less jumps, 72.4 mm total, longest single step 9.8 mm against a 93 µm
  median spacing. Now cut rather than bridged.
- **Radius spikes are fixed.** Segments peaking >3× their own median: 33 → 1.
  Max radius 3694 µm → 1795 µm, now below the 1824 µm ceiling of what direct-plane
  measurement ever observed.
- **The area estimator was under-reading — but the synthetic disagrees, and this
  needs re-measuring.** The observation stands as recorded: within a single
  segment, area-branch points read a median 0.73× (worst 0.36×) the perimeter
  points beside them, `GATE_VOXELS` is now 0, and it accounted for only 3 of the 15
  deep within-segment dips. What is now in doubt is the *reading* of it. On
  synthetic sections the area estimator recovers `sqrt(area/π)` to 1–2% down to
  1.25 voxels, and near the old gate it reads **higher** than perimeter, not lower
  (r = 2 vox: `r_area/r_perim` = 1.23 round, 0.99 at 4:1) — because it is the
  perimeter estimator that under-reads there. The 0.73× compares points on
  opposite sides of a size boundary, so it may be reading a genuine taper rather
  than estimator disagreement. Re-measure it within a fixed size band before
  relying on it either way.
- **The collapse argument for `GATE_VOXELS = 0` does not apply at the sizes the
  floor is about.** Area's under-read on a flattened lumen is real and reproduces
  (r = 5 vox at 8:1: `r_area/r_perim` = 0.604, against the ~0.57 at 10:1 the
  docstring cites) — but it is size-dependent. At r = 2 vox and 4:1 the gap is
  0.994: a section that small cannot be very flattened once digitised. So the
  reason the gate was switched off is a large-section argument being applied to
  small sections. Revisit the gate once the correction is in — a corrected
  perimeter may make the gate unnecessary rather than merely mis-set.

## Still open

- **The junction mask is bounded now, but 47% of the recovered segments are still
  more than 25% off** — see "Fixed: the mask is now bounded". The remaining work is
  the residual, not the mechanism: segments 276 and 297 are still wholly masked
  (297's target is 654 µm against 234 written), and segments 209 and 182 moved
  modestly *further* from their measurement under the cap for reasons not yet
  looked into. `junction_damage.py` regenerates the scoring; `segment_diagnosis.py
  265 268` is the regression check and should now read ~760 µm, not 269 and 222.
- **`JUNCTION_MASK_MAX_FRACTION = 0.4` is a bound, not a measurement.** Unlike the
  perimeter correction it rests on no ground truth — it says only "a junction may
  not eat a whole branch", and 0.4 was chosen to leave a fifth of every segment
  rather than derived from anything. A sweep against the `reformat` targets would
  say whether 0.3 or 0.45 does better, and nobody has run one.
- **Measure the tree's own lumen thickness distribution.** "Severe collapse defeats
  both estimators" shows that failure mode analytically, but nothing yet says how much of LADAF-2024-28 is in it.
  `crosssection.measure()` already returns `minor_ratio` and `isoperimetric` per
  point. The number wanted is the fraction of sections under ~2 voxels of minor
  axis, split by whether they were accepted, refused by the floor, or rejected as
  unstable. If that fraction is small the collapse story explains one branch; if it
  is large it explains the 3,154 `unmeasurable` and much of the 1,427 unstable, and
  it outranks both approach A and approach B.
- **Re-express the floor in the quantity being measured.** `MIN_BLOB_VOXELS = 12`
  gates on area while the pass reports perimeter. A gate on minor-axis thickness, or
  on perimeter with a separate thickness guard, would refuse the sections that are
  genuinely unmeasurable without refusing every collapsed one. Do not simply lower
  the number — that admits the fragments too, and they read worse than nothing.
- **Revisit `GATE_VOXELS = 0`.** It was set to disable the area estimator because
  area under-reads a collapsed lumen. Two independent lines now contradict the
  reasoning: on synthetic sections area recovers its own target to 1–2% down to 1.25
  voxels while perimeter under-reads badly, and on this tree area is both smoother
  along vessels and closer to Murray. Perimeter remains better *centred* in absolute
  terms. The candidate that fits every measurement so far is **area plus an explicit
  global shrinkage factor**, which nobody has tried. Every surface inherits this
  choice, so it sits upstream of the bifurcation work.
- **Identify the named vessels.** Several conclusions are gated on it: whether the
  largest segments are proximal LAD or left main decides whether the ex-vivo calibres
  are plausible or far too small, and the Murray deficit cannot be read without
  knowing which branches are missing. Nothing in either repo names a vessel.
- **Bifurcation geometry sweep** (`research_scripts/bifurcation_sweep.py`, results under
  `analysis_out/bif_sweep/`): `SDF_FLAT_CAP_BIF_SHIFT_FACTOR` in {0, 0.2, 0.3, 0.4}
  crossed with `BIF_CARINA_ENABLE`. Unlike everything else here **there is no ground
  truth** — "the expected carina" is anatomical judgement — so the sweep produces
  candidates to look at rather than a winner. Note the radius pass's own carina model
  is *backwards* for this purpose: `BIF_DAUGHTER` sets the daughter to
  `0.1 x anchor` **at** the junction and widens it distally, pinching the ostium
  shut, where the anatomy wants it flared. Inverting that is a separate job from the
  surface knobs.
- **Locate the three handles in graph 0.** Genus
  in the left tree went 0 → 1 → 3 across raw, corrected, and corrected-plus-bounded,
  tracking radius accuracy upward step for step (see "Fixed: the mask is now
  bounded"). Two readings fit and nothing yet separates them: either the restored
  calibres are right and adjacent branches at the trifurcations genuinely touch, in
  which case the surface pipeline's merge handling is what needs work; or some
  restored radius is too large. Locating even one handle would decide it, and no one
  has looked. The three surfaces are `final_surface_raw_baseline/`,
  `final_surface/` and `final_surface_capped/`, all built from the same settings,
  so the diff is attributable.
- **Then look at the surface — actually look at it.** Genus is the metric least able
  to judge this work: by this document's own pitfall, *topology metrics cannot see a
  smooth bulge*. A surface can be genus 0, manifold and single-component and still
  be wrong everywhere the radius is wrong — which is exactly what the raw baseline
  was. None of the three meshes has been opened.
- Whether the correction survives real sections. It is derived for a boundary the
  mask actually resolves; a lumen that is genuinely one voxel across, or one whose
  mask boundary is noise rather than anatomy, is outside what was tested. The tree
  run says nothing about this either way — a monotone transform of a wrong radius
  is still wrong.
- **Decide the failing `test_radius_circles` test — it is a real fork, not a stale
  assertion to delete.** `edit/tests/test_radius_circles.py::test_repeated_junction_records_keep_their_edge_local_planes`
  fails, and it predates this work: an earlier uncommitted session changed
  `viewer3d._point_tangents` from `np.gradient` to `crosssection.robust_edge_tangents`,
  and the test still asserts the old estimator's exact `[0, 1, 0]`. The viewer change
  has a good argument behind it — the rings are being oriented with the same tangent
  the radius was measured with, instead of a cruder one that made a correct radius
  look wrong. So the likely answer is that the test wants rewriting against the
  fitted tangent. Confirm that is what is wanted before editing it: the alternative
  reading is that `_point_tangents` should not have changed, and picking the wrong
  one silently is how a deliberate change gets reverted by accident. Everything else
  in `edit/tests` passes — 1,242 of them.

- Right tree surface quality. The genus 11 and ~21 self-intersections at 0.12 mm
  recorded earlier were measured on an older graph and are **superseded**: on
  `radius_perim_only.am` both trees now build at genus 0, and on
  `radius_perim_corrected.am` at genus 1 and 0. Self-intersections have not been
  re-counted on either.
- **New dataset, not started:** `%DATA_DIR%/LADAF_2021_17_heart_coronaries`.
  Manual segmentation needing fusion/component cleanup, then TEASAR parameter
  tuning to suppress spurious branching at collapsed vessels, then re-centring and
  perimeter radii. Two things to settle first: skeletonise from the manual
  segmentation or start from the existing `...Spatial-Graph_corrected.am`, and
  which mask is authoritative — the 47 GB `labels.am` or the 768 MB
  `Regions.am`.
