# Skeletonisation: algorithms, the super metric, and what this repo does with them

The methodology of

> C.L. Walsh, M. Berg, H. West, N.A. Holroyd, S. Walker-Samuel, R.J. Shipley,
> **"Reconstructing microvascular network skeletons from 3D images: What is the ground truth?"**,
> *Computers in Biology and Medicine* **171** (2024) 108140.
> <https://doi.org/10.1016/j.compbiomed.2024.108140> (CC BY)

translated into what it means for this pipeline, plus the places where this repo deliberately
departs from it and why. `skeleton_analysis` is a port of the MATLAB code accompanying this
paper; [the divergence section](#where-the-existing-port-differs-from-the-paper) records where that
port and the paper disagree.

---

## 1. The problem the paper is about

Extracting a vascular network from an image is two steps: **segmentation** (which voxels are
vessel) and **skeletonisation** (reduce those voxels to a graph of nodes, segments and radii).

Segmentation has an accepted validation culture — manual gold standards, consensus by STAPLE, and
Dice/Jaccard/Hausdorff to score against them. **Skeletonisation has none.** The usual validation is
visual inspection of the skeleton overlaid on the image. That does not scale, and the paper shows
it hides differences large enough to change the conclusion.

How large: running four algorithms on the same consensus segmentation, *the skeleton with the most
nodes had more than double the nodes and segments of the one with the fewest*. Radius, length-to-
diameter ratio, branching angle, intervessel distance and tortuosity distributions differed with
p ≤ 0.0001 across nearly every algorithm pair. Simulated perfusion flow rate differed by **an order
of magnitude**, and which algorithm gave the highest flow was not even consistent between datasets.

The paper's resolution: there is no ground-truth skeleton and building one is impractical — the
only published manual consensus centrelines, Schaap et al.'s coronary set, took **over 500 hours of
expert time for the first three branches**. So treat the **segmentation as the gold standard** and
ask a different question: *how much of the information in the segmented image did this
skeletonisation lose?*

## 2. The four algorithm families

| family | how it works | representative | in Python? |
|---|---|---|---|
| **Thinning** | iteratively peel voxels without changing topology until one voxel thick | Lee et al. medial-axis thinning; **VesselVio** | **yes** — `skimage.morphology.skeletonize(..., method="lee")` |
| **Thinning + distance ordering (DTHO)** | peel in order of distance-transform value, which preserves medialness and is *proved* to preserve homology | Pudney, Palágyi; **Amira AutoSkeleton** (parallelised Fouard) | **no** — commercial; reachable only by ingesting an Avizo `.am` |
| **Minimum-cost path** | shortest-path spanning tree through the volume; produces a **strict tree** | **TEASAR**; Amira **Centerline Tree** | **yes** — `kimimaro` |
| **Wave-front / vessel scooping** | grow clusters from seeds, take each cluster's centre of mass | Rodriguez, Wu; **MOST** / Vaa3D | **no** |

The three stated goals of a skeletonisation are **thinness** (one voxel wide), **medialness**
(equidistant from the boundaries) and **homology** (same connected components and loops as the
original). The paper's measurements show these goals are not actually met:

- Lee thinning **does not preserve homology**, despite being the most widely deployed method.
- Neither AutoSkeleton nor VesselVio reproduced the segmentation's connected-component count.
- **Centerline Tree (TEASAR) does preserve component count but destroys loops by construction**,
  because it forces a tree. On the FaDu network that pushed the terminal-node fraction to **90%**.
- Efficiency work can break the guarantee: a parallelised implementation of an algorithm proved
  homotopic may no longer be.

Radius estimation differs just as much, and is worth knowing per algorithm because flow goes as the
fourth power of it:

| algorithm | radius estimator | consequence the paper measured |
|---|---|---|
| Amira AutoSkeleton | 1/5 of the maximum Chamfer distance | systematically **lower** radii than the others |
| Centerline Tree | maximum inscribed sphere | **hard floor at 2 × pixel size**, which distorts the distribution wherever vessels approach the resolution limit |
| VesselVio | modified Euclidean distance map | **largest** radii; correspondingly the smallest intervessel distance |
| MOST | radius-adjustable sphere | — |

## 3. The super metric

Five terms, each a comparison between the reconstructed skeleton (`S`) and the binary image (`I`).
Each was chosen because the paper's flow simulations showed it drives functional behaviour.

| | measure | from the binary image `I` | from the spatial graph `S` |
|---|---|---|---|
| `V` | total network volume | labelled voxels × voxel volume | `Σ_s π R_s² L_s` over subsegments |
| `cc` | connected components | 26-connected components | number of subnetworks |
| `χ` | local Euler characteristic | tunnels/holes in the **largest** component | `N − E` of the **largest subgraph** |
| `B` | bifurcation Dice | bifurcation points annotated in a small subvolume | nodes with coordination ≥ 3, matched to them within a tolerance |
| `cl` | cl-sensitivity | — (`cl_I ≡ 1` by definition) | `Σ(V_I · l_S) / Σ l_S`, with `l_S` the centreline rasterised by **Bresenham's algorithm** |

`χ` is *reformulated*: the classical Euler characteristic can be legitimately zero (a network with
one loop), which would make a relative difference undefined. The paper's main text defines the
local form as

```
χ = 2 − χ_classical            (always strictly positive)
```

> **Discrepancy.** Supplementary §7 prints `χ = −χ_classical − 2`. That is not positive for a tree
> (`χ_classical = N − E = 1` gives `−3`), so it contradicts the stated requirement. The main text's
> form is the one used here.

Combined (Eq. 9) as a scalar product of a distance vector with a weight vector:

```
M_S = Σ_i  w_i · | Δ_IS f_i / f_I,i |
```

With `w_V = w_cc = w_χ = 1`, `w_B = 1/B_S²`, `w_cl = 1/cl_S³`, and using `B_I = cl_I = 1`, this is
Eq. 10:

```
M_S = |V_I − V_S|/V_I  +  |cc_I − cc_S|/cc_I  +  |χ_I − χ_S|/χ_I
      +  |1 − cl_S| / cl_S³  +  |1 − B_S| / B_S²
```

**Lower is better; 0 is identical.** The non-linear weights are the design, not a detail. Because
`cl` enters as `1/cl³`, even a small drop in centreline-inside-mask fraction dominates the sum and
rejects the skeleton — the paper's reasoning being that an algorithm may reasonably cut a corner,
but large stretches of centreline *outside* the segmentation are never plausible. `B` is weighted
more softly at `1/B²` because the bifurcation reference is manually annotated and therefore
somewhat subjective.

Reading the *terms separately* is as useful as the total. In the paper's Fig. 8B the AutoSkeleton
run is dominated by its Volume term, which pins the fault on its 1/5-Chamfer radius estimator
rather than on its topology.

### Parameter optimisation

The four algorithms lived in software that was "hardly scriptable", so the paper used a surrogate:
sample each algorithm's parameter space **10 times by Latin hypercube**, evaluate `M_S` at each
sample, fit **universal Kriging with a spherical variogram**, and optimise the surrogate. Swept
ranges (Table S14):

| algorithm | parameters swept | range |
|---|---|---|
| Centerline Tree | slope; zeroVal | 1–6; 1–10 |
| AutoSkeleton | smooth; attach-to-data; iterations | 0–1; 0–1; 1–15 |
| VesselVio | isolated-segment filter; **pruning length** | 0–10 px; **0–5 px** |
| MOST | threshold; seed; slip | 1–40; 2–10; 2–10 |

Optimisation mattered: `M_S` for Centerline Tree fell from 3.13 to 2.13, VesselVio 2.55 → 1.61,
MOST 6.48 → 9.85 at best and 10⁸ at worst. AutoSkeleton won overall (1.39) and was the most
parameter-insensitive. After optimisation, the spread of simulated perfusion flow rate across
algorithms tightened measurably.

Note VesselVio's **pruning length** is the paper's version of the spur removal this repo does in
`edit/skeleton_optimise.py::prune_spurs`.

---

## 4. What this repo implements, and where it departs

Implemented in `edit/supermetric.py`, faithfully to Eq. 10 except where noted.

**Departure 1 — `χ` is scored against a tree, not against the segmentation.**
This pipeline images *coronary arteries* ex vivo. At this calibre the true anatomy is a tree: there
are no anastomoses to recover. The loops that appear in a skeleton here come from two artefacts the
viewer exists to find — a vessel that collapsed in the middle and was segmented as two, and a
segmentation that touches a neighbouring vessel. So `edit/skeleton_optimise.py::remove_loops`
deletes them deliberately, and scoring `χ` against the segmentation's loop count would make the
metric fight that prior. With `tree_chi=True` the reference is `χ_classical = 1`.
**Consequence: `M_S` values computed here are not comparable with the paper's.** Pass
`tree_chi=False` for a comparable number.

**Departure 2 — the `B` reference is automatic.**
The paper annotates bifurcations by hand in a subvolume. Here the reference is the segmentation's
own skeleton junctions, via
`skeleton_analysis.optimisation.volume_metrics.skeleton_junction_points` — Lee-skeletonise the
mask, cluster the junction voxels, take one centroid per cluster. It keeps the gold standard the
binary image, as the paper intends, and makes a sweep runnable without a day of annotation first.
It inherits Lee thinning's own errors, which is why `B` is the softly weighted term.

**Not implemented:** DTHO/AutoSkeleton and vessel scooping. The first is reachable by ingesting an
Avizo export as a candidate (`skeletonise-all --algorithms amira`), the second not at all.

### Measured on LADAF-2024-28, at stride 1

**Measure at the resolution you will run at.** Everything below was first measured at stride 4
(264 µm voxels, vessels 1–2 voxels across) because it iterates in seconds. Almost every conclusion
drawn there was an artefact of the decimation, and the stride-4 figures are kept further down only
as evidence of *how badly* that misleads.

The image itself, at the two strides:

| | `V` | `cc` | `χ`<sub>classical</sub> | reference bifurcations |
|---|---|---|---|---|
| stride 4 | 1317.475 mm³ (71,668 vox) | **412** | −12 | 208 |
| stride 1 | 1318.279 mm³ (4,589,552 vox) | **55** | −10 | 645 |

Volume agrees to 0.06%, but decimation shatters the mask into **seven times** as many connected
components. Any `cc` term measured at stride 4 is mostly measuring the stride.

Lee thinning at stride 1, `optimise-skeleton` with each smoother (all runs de-loop and prune):

| stage / smoother | `M_S` | `V` | `cc` | `χ` | `cl` | `B` | cl-sens |
|---|---|---|---|---|---|---|---|
| Lee, raw | 12.353 | 0.170 | 0.055 | **12.000** | 0.003 | 0.126 | 0.9974 |
| **+ gaussian** | **0.419** | **0.010** | 0.055 | 0.000 | 0.004 | 0.351 | 0.9962 |
| + bspline | 0.440 | 0.030 | 0.055 | 0.000 | 0.005 | 0.351 | 0.9952 |
| + none | 0.468 | 0.059 | 0.055 | 0.000 | 0.003 | 0.351 | 0.9966 |
| + multiscale | 0.470 | 0.059 | 0.055 | 0.000 | 0.005 | 0.351 | 0.9948 |
| + savgol | 0.504 | 0.095 | 0.055 | 0.000 | 0.004 | 0.351 | 0.9960 |

**De-looping remains the whole story** — `χ` 12.0 → 0 takes `M_S` from 12.353 to under 0.5.
After that, the only term that separates the smoothers is `V`; `cl` and `B` are flat to three
decimal places.

> **Read the `V` column with care.** These rows were measured with the *old* re-centring, and its
> `V` is an artefact — see [below](#re-centring-and-the-zig-zag-it-used-to-cause). The smoother
> ranking itself still holds; what does not is any reading of `V` as evidence that the geometry
> was right. Re-run with the current code and every `V` here rises, because the term stops being
> cancelled by a compensating error and starts reporting the radius problem the pipeline exists
> to fix.

**Smoothing does help at full resolution, and re-centring is why.** Measured directly:

| graph | segments | length | network `V` | `V` term |
|---|---|---|---|---|
| Lee, raw | 1226 | 3238 mm | 1093.8 mm³ | 0.170 |
| de-loop + prune + re-centre | 896 | 3602 mm | 1395.9 mm³ | 0.059 |
| …+ gaussian smooth | 896 | 3342 mm | 1305.7 mm³ | **0.010** |
| …+ multiscale | 896 | 3601 mm | 1395.8 mm³ | 0.059 |

Re-centring moves points laterally onto the lumen centre, which **lengthens** the path by 11% and
overshoots the image volume to +6%. Smoothing removes that added wiggle and lands within 0.8%.
They are a pair: at stride 4 re-centring had almost nothing to move, so there was no overshoot to
correct and smoothing only did harm.

> **Superseded.** That 11% was not re-centring finding the lumen centre — it was re-centring
> *scattering*, and smoothing was earning its keep by cleaning up after it. See
> [Re-centring, and the zig-zag it used to cause](#re-centring-and-the-zig-zag-it-used-to-cause).
> With that fixed the path *shortens* 6.8%, which is what removing a rasterised staircase does.
> The `V` column above is still the right way to read the smoothers against each other; the causal
> story attached to it was wrong.

### Re-centring, and the zig-zag it used to cause

The optimised centreline used to spike and cross itself in dense regions. It was not the smoother:
the damage is fully present with smoothing disabled, and `opt_multiscale.am` matched `opt_none.am`
to within 0.01% on every measure. De-looping, pruning and contraction never move a point.

`reversing` below is the share of interior points turning more than 120° — doubling back, which a
vessel does not do, so any non-zero value is artefact:

| backend | graph | edges | max step | **reversing** | >150° | length |
|---|---|---|---|---|---|---|
| `lee` | the input | 1226 | 114 µm | 0.00% | 0.00% | 3238 mm |
| `lee` | old re-centring | 896 | 1647 µm | **4.41%** | 2.03% | 3602 mm (+11.2%) |
| `lee` | **new re-centring** | 896 | **494 µm** | **0.03%** | **0.01%** | 3017 mm (−6.8%) |
| `teasar` | the input | 901 | 148 µm | 0.00% | 0.00% | 3284 mm |
| `teasar` | **new re-centring** | 826 | **199 µm** | **0.00%** | **0.00%** | 3121 mm |
| `amira` | the Avizo export | 307 | 9811 µm | 9.69% | 5.89% | 3613 mm |
| `amira` | **new re-centring** | 307 | 9811 µm | **9.25%** | **5.58%** | 3558 mm |

Points thrown a full radius or more off the input skeleton, on `lee`: **3.045% → 0.014%**.

All three backends qualify on the same defaults, which is the result that says the diagnosis was
right rather than curve-fitted. **TEASAR is not the smooth alternative it sounds like** — kimimaro
traces through voxel centres too, so its central-difference tangent rotates a median 18.4°
(p95 45.0°) against Lee's 19.5° (p95 39.2°). It arrives cleaner than Lee only because it does not
thin, and it needs the same tangent fix. The Avizo export is the hard case: it arrives already
doubling back on 9.69% of its points — those 9.8 mm steps are Amira's and are untouched — because
it is sampled at 0.16 radii per point, dense enough that the turn angle is measuring quantisation.
Re-centring improves it rather than adding to it, which is all that can be asked there.

**Four faults compounded.** The cut plane's normal came from `np.gradient(coords, axis=0)`, a chord
between the two *adjacent* points, which on a Lee staircase — median turn 45° — rotates a median
**19.5° (p95 39°) from one point to the next**, so neighbours cut differently tilted sections of the
same vessel and find unrelated centroids. The cut window doubled to `max_half` = 64 voxels (4.2 mm,
26× the median radius) chasing a streak that touches the border at any width. Nothing bounded a
move relative to the *point spacing*, so a point could step past its own neighbour. And the clamp
was a full radius — onto the vessel wall — applied per pass over two passes.

Four bounds now hold it, one for each:

| constant | what it stops |
|---|---|
| `RECENTRE_TANGENT_RADII` | the plane direction is a chord over 2 local radii; neighbour-to-neighbour rotation 19.5° → under 5° |
| `RECENTRE_GROW_RADII` | the window stops growing, and a section still touching the border is refused rather than measured |
| `RECENTRE_MAX_MOVE_SPACING` | no point moves past the two it sits between |
| `_revert_new_reversals` | stated directly: re-centring may not leave the line doubling back worse than it found it |

plus `RECENTRE_DAMPING` (half-steps) and a `RECENTRE_MAX_MOVE_FRAC` of 0.5 measured against where
re-centring *started*, so N passes cannot walk N radii.

#### Which of them actually does the work

Measured by ablation, each guard alone against no guards at all, two re-centring passes on `lee.am`:

| enabled | reversing | max step | length |
|---|---|---|---|
| nothing — the old code | 3.768% | 1250 µm | 3410 mm |
| only `RECENTRE_GROW_RADII` | 2.269% | 1110 µm | 3227 mm |
| only `RECENTRE_TANGENT_RADII` | 0.571% | 1107 µm | 3135 mm |
| only `RECENTRE_MAX_MOVE_SPACING` | 0.270% | **513 µm** | **3079 mm** |
| only `_revert_new_reversals` | **0.032%** | 873 µm | 3183 mm |
| **all four** | **0.026%** | **494 µm** | **3017 mm** |
| all four, `damping` 1 and clamp 1 | 0.026% | 491 µm | 3048 mm |

Three things to read out of that, two of which correct earlier claims here:

1. **No single guard is redundant, and none is sufficient on its own** — leave-one-out barely
   moves the number (0.023–0.049%) because the remaining three cover for it, but each *alone*
   leaves between 0.03% and 2.3%. They are cheap; keep all four.
2. **The reversal guard's 0.032% is partly circular.** It is defined in terms of the statistic it
   is being scored on, so of course it minimises it. Judge it on the independent columns instead:
   its max step is 873 µm and its path 3183 mm, both worse than the fold guard's. It is a backstop,
   not the mechanism.
3. **On the independent columns the fold guard is the strongest single fix** — "no point may step
   past its neighbour" gets max step to 513 µm and the path to 3079 mm on its own. An earlier
   version of this section called the tangent estimate "the one that matters most". That was the
   first thing measured, not the biggest thing; the tangent window alone leaves 1107 µm steps.
4. **Damping and the tighter clamp are not what fixed this.** With the four guards in place,
   `damping=1` and `max_move_frac=1` give the same 0.026%. They stay because a damped fixed-point
   iteration is the right shape for something that re-estimates its own input, not because they
   are carrying the result.

**`radius-perimeter` had the same flaw and it mattered more there.** An oblique plane cuts an
ellipse, whose perimeter exceeds the true section's by roughly 1/cos of the tilt — so
`r = perimeter/2π` was over-stated systematically, on the one number this whole pipeline exists to
correct. It now uses the same chord estimator.

**The super metric never caught any of it — it actively preferred it.** This is the part worth
reading twice:

| graph | length | network `V` | vs image | `V` term |
|---|---|---|---|---|
| Lee, raw | 3238 mm | 1093.8 mm³ | −17.0% | 0.170 |
| old re-centring | 3602 mm | 1395.9 mm³ | +5.9% | 0.059 |
| …+ gaussian | 3342 mm | 1305.7 mm³ | **−1.0%** | **0.010** |
| **new re-centring** | 3017 mm | 1020.3 mm³ | −22.6% | 0.226 |
| …+ gaussian | 2857 mm | 964.4 mm³ | −26.8% | 0.268 |

`V` is `Σ π r̄² × subsegment length`. Hold the radii fixed and **extra path length is extra
volume** — so the scattered centreline's 11% of wiggle carried the network volume from 17% under
the image to within 1% of it, and `M_S` recorded that as the best result in the table. The volume
was right for a reason that has nothing to do with vasculature.

Correct the geometry and the `V` term jumps to 0.27, because it is now measuring the error it
should have been measuring all along: **the radii are Lee's EDT inscribed radii, and an inscribed
circle under-states a collapsed elliptical lumen.** `radius-perimeter` puts the median radius up
by 1.44×, and π r² makes that a factor of two on volume. The whole premise of this pipeline is that
those radii are wrong; `V` was hiding it behind a compensating geometry error.

So **do not tune re-centring on `M_S`.** `cl`-sensitivity cannot see a spike that leaves the lumen
and comes back — 4.41% of points doubling back costs about 0.005 — and `V` rewards the spike
outright. That is not a fault in the paper's metric, which was built to rank *skeletonisation
algorithms* against a fixed radius estimator, but it is why `roughness()` now runs on every
`optimise-skeleton` and is printed as its own ranked table in every sweep.

### A section that never closes — or closes around a junction — is not a cross-section

The same runaway window was in `radius-perimeter`, and it mattered more there. 612 points reached
`max_half` — a 4.2 mm half-width — with the blob still running off the edge, so what
`cv2.arcLength` traced was the outline of everything the plane had swallowed. They came back at a
**median radius of 3.5 mm against 0.76 mm stored, a 4.8× inflation**, and being r²-weighted those
612 points carried **28% of the entire network's volume**. The largest ratio was 46.9×.

They were warned about and used anyway. They are now left unmeasured, so `_fill_gaps` interpolates
them from neighbours whose section did close — the same treatment every other unmeasurable point
already got.

Closure alone is not sufficient: an expanding plane can eventually close around a carina or an
oblique multi-vessel component and return a very large but perfectly closed contour. The perimeter
pass therefore leaves a junction zone extending two times the largest incident input radius
unmeasured. This is especially important for a small side branch: using its own small radius would
start measuring while its centreline is still inside, or nearly parallel to, the parent lumen. A
second, segment-local gate applies outside that zone. When the sampling half-width grew beyond four input radii, a correction
larger than three times the robust log-space correction trend is rejected. The trend is refitted on
the lower residual half, so runaway measurements cannot vote themselves normal. Rejected runs are
interpolated from trusted neighbours and recorded in `radius_reject_reason`.

There is no absolute-radius ceiling. `--max-radius-factor` (2.0 by default) compares each provisional
measurement with a robust local median over an arclength window scaled by the local input radius.
It refits after removing first-pass highs, then interpolates rejected contours from non-outlier
neighbours. A uniformly large vessel therefore survives; only a local jump is filtered.

The carina points themselves are reconstructed from the stable branch measurements. At a
degree-three node the largest branch is treated as the parent, its junction target is the
Murray-equivalent `(r1³+r2³)^(1/3)` of the daughters, and each daughter ostium is bounded by the
same local factor. Smooth log-radius blends across the excluded zone give the capsule surface a
gradual parent/daughter transition while retaining separate endpoint values for each incident edge.

Two terms still to read with care:

1. **`B` gets *worse* when de-looping.** Dice falls 0.898 → 0.784, because the reference bifurcation
   set comes from the mask's own Lee skeleton, which still contains the loop junctions. The
   de-looping prior is not one the reference shares, so `B` penalises it for the same reason `χ`
   would without `tree_chi`. Read `B` as "agreement with a Lee skeleton", not as truth.
2. **`V` and the perimeter radius disagree by design.** After `radius-perimeter` the network volume
   rises above the segmented volume, so the `V` term gets worse. That is the re-inflation premise
   working: a collapsed lumen is deliberately reconstructed as the circle of equal *perimeter*,
   which has more area than the slit it came from. **Do not optimise `V` after applying perimeter
   radii** — score the skeleton before the radius pass, which is the order workflow 12 uses.

`cc` no longer belongs on that list. At stride 1 the graph gives 52 subnetworks against the mask's
55 and the term is 0.055 — an earlier claim here that `cc` "never improves" and is purely a
segmentation problem was measuring decimation, not segmentation.

### What stride 4 got wrong

Kept as a caution, not as a result. Same pipeline, same data, only the decimation differs:

| smoother | stride 4 | stride 1 |
|---|---|---|
| gaussian | 0.966 (4th) | **0.419 (1st)** |
| bspline | 0.943 (3rd) | 0.440 (2nd) |
| none | **0.923 (1st)** | 0.468 (3rd) |
| multiscale | 1.016 (5th) | 0.470 (4th) |
| savgol | 0.937 (2nd) | **0.504 (5th)** |

The ranking essentially inverts. At stride 4 the honest reading was "no smoother beats not
smoothing"; at stride 1 three of the four do, and the best of them halves `M_S` against the raw
skeleton. `savgol` goes from best-of-the-smoothers to worst. **A sweep is only as good as the
resolution it was run at.**

### Where the existing port differs from the paper

`skeleton_analysis.optimisation` does **not** implement Eq. 10. Its `meta_metric` docstring is
candid that it "realises the formula left commented in `meta_metric.m`" — it was reconstructed from
a commented-out MATLAB expression rather than from the paper.

| | paper (Eq. 10 / Table S13) | `skeleton_analysis` |
|---|---|---|
| combination | weighted sum of \|relative differences\| | `sqrt(Σ (relative difference)²)` — an RMS |
| weights | `w_B = 1/B²`, `w_cl = 1/cl³` | all unit |
| `B` term | bifurcation **Dice** | branch-point **count** |
| `χ` | `2 − χ_classical` on the largest component; `χ_graph = N − E` | classical Euler on the whole volume |
| `cl` | cl-**sensitivity**, Bresenham-rasterised | full clDice between two volumes |
| `V`, `cc`, `χ` from | the **spatial graph** vs the binary image | two **volumes** |

The last row is the substantive one, and `edit/optimise.py` already records its symptom:
`super_metric` "compares two segmentations, not two skeletons… its Volume/CC/Euler terms are
identical for both and contribute exactly nothing." Computing `V`, `cc` and `χ` *from the graph* is
what turns the metric into an objective a skeleton can be optimised against.

One more, smaller: the port's `centreline_sensitivity` samples the mask **at the graph's points**.
The paper rasterises the **lines between** them. Wherever consecutive points are more than a voxel
apart — which is everywhere on a smoothed Avizo graph — point sampling scores a centreline as fully
inside the mask when the segment between two points may leave it entirely.

`skeleton_analysis` is left unmodified; it is a shared package with its own test suite
asserting the current behaviour. The pieces of it that *are* correct and reused here are
`bifurcation_dice_points` (greedy matching), `skeleton_junction_points`, `region_morphometrics`,
and the whole of `outlier/oblique.py`'s perimeter-radius reasoning.

---

## 5. Choosing a smoother

`optimise-skeleton --smoother` selects between five centreline smoothers: the built-in
`gaussian`, `coronary_sdf`'s two Gen-1 filters (`savgol`, `bspline`), its Gen-2
`multiscale` optimiser, and `none`. All are sweepable, which is the point — `coronary_sdf`'s
own `coronary_sdf_refinement_plan.txt` rates the multiscale one *"Repaired but not qualified…
it still fails synthetic curvature/contact constraints and the measured LADAF-56
convergence/performance audit, so it remains opt-in"*, and a super metric is exactly the
qualification it was missing.

**`multiscale`** minimises a penalised-spline energy summed over three bandwidths of
`(1, 2, 4)` *local radii* in radius-normalised arc length, solved graph-wide by ADMM against
three projections: a trust region (drift ≤ 0.25 r), a curvature bound (κr ≤ 0.95), and
tube-tube clearance. Measuring everything in radii makes it globally scale-equivariant. The
constraints participate in the fit rather than moving points after it — which is the
criticism the refinement plan levels at Gen 1's fixed 51-sample window, *"not invariant to
sampling density or vessel length… segmentwise fitting also cannot prevent one branch moving
into another"*.

### Ranking

Measured at stride 1; see [§4](#4-what-this-repo-implements-and-where-it-departs) for the full
table and for how badly the stride-4 version of this misled.

| smoother | `M_S` | `V` | cl-sens |
|---|---|---|---|
| **gaussian** | **0.419** | **0.010** | 0.9962 |
| bspline | 0.440 | 0.030 | 0.9952 |
| none | 0.468 | 0.059 | 0.9966 |
| multiscale | 0.470 | 0.059 | 0.9948 |
| savgol | 0.504 | 0.095 | 0.9960 |

`gaussian` remains the default: it wins here, it costs 0.7 s against multiscale's 258 s, and it
depends on nothing outside this repository.

### The multiscale smoother is inert on this data, and the freeze is not why

It ranks fourth, but the score is not the interesting part — its own report is:

```
smooth [multiscale]: 24039 point(s) moved, median 0.1 um, max 161.6 um (257.9s)
  [warning] did not converge in 20 iteration(s)
  [warning] 20794 constraint(s) left unresolved (20794 curvature, 0 new branch conflicts)
  39947 pre-existing overlap(s): those points and their one-radius halo were frozen
  certified line search kept 0% of the solution; the rest would have created new contact
  drift max 0.001 r, p95 0.001 r
```

**It solved, then rejected essentially all of its own answer.** `backtrack_fraction` is 0.0046 —
the certified line search accepts the largest fraction of the solution introducing no *new* tube
contact anywhere, and here that is under half a percent. 24,037 points move, by 0.00116 of a
radius: 0.2 µm. Path length confirms it, 3600.9 mm against 3601.6 mm for no smoothing at all.

An earlier version of this section blamed the preserve-only freeze — "its preconditions are
violated across the whole tree". **That was wrong, and the report says so:** `frozen_points` is
12,533 of 36,613, so 66% of the graph was free to move and did. The bottleneck is that
`new_count(a)` bisects **one scalar over the entire graph**, so a single new touch among 37k
points vetoes every other point's correction. In a tree where roughly half of all points have
another vessel's centreline within six radii (median gap 1.72 r), that veto is close to certain.

Fixing re-centring halved the overlaps and did not free it:

| on | `input_overlaps` | `frozen_points` | `curvature_violations_before` | `backtrack_fraction` |
|---|---|---|---|---|
| the old, scattered skeleton | 39,947 | 12,533 | 20,816 | 0.0046 |
| the re-centred one | **17,386** | **9,083** | **16,712** | **0.0040** |

`curvature_violations_before` is the reason it will not converge either. κ·r ≤ 0.95 with points
87 µm apart in a 162 µm-radius vessel means **any turn over about 28°** is a violation, and a
centreline sampled at half a radius per point cannot avoid that — the constraint is measuring
sampling density, not geometry. 16,712 of 36,613 points violate it on a centreline with 0.04%
doubling-back. Qualifying this smoother needs the solver to resample in radius-normalised arc
length before it starts, which is a `coronary_sdf` change, not a parameter.

Its clearance constraint — the thing Gen 1 cannot do at all — held perfectly throughout
(0 new branch conflicts).

### The drift cap, and why it is not declared

`CENTERLINE_MAX_DRIFT_RADIUS_FACTOR` is the multiscale trust region, in local radii.
`centerline_optimizer.py:355` reads it as `getattr(config, ..., .25)` and **`config.py` never
declares it**, so it cannot be reached through `SdfConfig` at all. `edit/sdfconfig.py` names it in
an `UNDECLARED` allowlist so it can be set and swept via `--drift-radius-factor`:

| | 0.05 | 0.1 | 0.25 (default) | 0.5 |
|---|---|---|---|---|
| `M_S`, stride 4 | 0.983 | 0.976 | 1.016 | **0.921** |
| `M_S`, stride 1 | **0.466** | **0.466** | 0.470 | 0.469 |

At stride 4 the cap spanned 0.095 and looked like the binding constraint — drift came back with max
*and* p95 pinned at exactly 0.250 r. At stride 1 the span is 0.004, and the reason is *not* that
the cap stopped mattering now vessels are resolved: it is that the line search rejects the solution
regardless of how much drift the trust region would have permitted. The knob is still worth having
exposed, but on this data it is not what limits the result.

### Do not smooth after `radius-perimeter`

Smoothing never rewrites a radius — `edit/smoothers.py` asserts that on every call. But it
moves the point the radius was *measured at*, so the cross-section is re-cut somewhere
slightly different. Measured on the stride-8 chain, `r_stored / r_perimeter` widened from
p5/median/p95 **1.00 / 1.00 / 1.00** to **0.93 / 1.00 / 1.08** after a multiscale pass.

Hence the ordering in workflow 12: all graph, taper, and geometry work — including
`repair-radius` when needed, then de-loop, prune, re-centre, and smooth — happens before
`radius-perimeter`. The perimeter pass runs **last**, so every radius is measured at the
position the point finally occupies. If `repair-radius` is run afterward, `radius_source`
suppresses redundant image-collapse detection; high spans are reported, but changing them
requires `--allow-decrease`.

## 6. Commands

See [CLI.md](CLI.md) for every flag.

```
python -m hipct_seg_debug.edit skeletonise-all   --seg SEG --out-dir DIR   # run and score each algorithm
python -m hipct_seg_debug.edit pick-roots        GRAPH --roots-json R.json # click each tree's inlet
#   ... or --pick-roots on skeletonise / skeletonise-all / optimise-skeleton
python -m hipct_seg_debug.edit optimise-skeleton GRAPH --out OUT [--sweep] # de-loop, prune, re-centre, smooth
python -m hipct_seg_debug.edit radius-perimeter  GRAPH --out OUT --seg SEG # perimeter radius per point
python -m hipct_seg_debug.edit score             GRAPH --seg SEG          # the five terms and M_S
```

### 6.1 One tree at a time

A coronary mask holds **two** anatomically distinct trees, left and right, and by
default everything above treats them as one object. `--per-tree` splits the mask
26-connected, skeletonises each component inside its own bounding box, and tags every
edge with a `tree` index:

```
python -m hipct_seg_debug.edit skeletonise-all --seg SEG --out-dir DIR \
    --per-tree --min-component-voxels 2000
```

The index is a property of the **mask** labelling (largest component is 0), not of the
graph, so `lee.am`, `teasar.am` and `amira.am` from one run all agree about which tree
is which. It is *not* portable across masks, strides or thresholds — a different
labelling renumbers everything, which is why the roots sidecar records each tree's
voxel count and bounding box.

Two things to know before turning it on:

* **The skeleton itself changes slightly.** Lee thinning inside a tight box and the
  slab-wise EDT see a different neighbourhood than a whole-volume run, so segment
  counts move by a fraction of a percent. This is a different skeleton, not a
  regrouping of the same one.
* **A restricted scope is not comparable with `M_S`.** See below.

### Choosing what a score is computed over

The default `M_S` is a *mixed* scope, and that is the published definition rather than
an oversight: `V`, `cc`, `cl` and `B` are taken over the whole graph, but `χ` over the
**largest component alone** (Table S13). On a two-tree coronary mask that means a
change confined to the right coronary tree does not move `χ` at all.

`--scope`, on `score`, `skeletonise-all` and `optimise-skeleton`, makes the choice
explicit:

| `--scope` | what is scored |
|---|---|
| `whole` (default) | the published definition, exactly as above |
| `per-tree` | each mask component against its own gold standard, one row per tree, combined by `--objective` |
| `largest` | the largest component only — the same restriction `χ` already had, now applied to **all five** terms |

`largest` is `per-tree` capped at one component, so its numbers match that scope's
`tree 0` exactly. Use it when the question really is about one tree; use `per-tree`
when it is about the skeleton as a whole and you want the second tree to count.

The whole-graph table is always printed first, whichever scope is chosen, because a
restricted scope redefines two terms: `cc` becomes `|1 - cc_s| / 1`, so a tree in
three fragments scores 2.0 where the same fragmentation against a global count of
hundreds was a small ratio.

Measured on LADAF-2024-28 at stride 4, the whole-graph `M_S` of 13.564 hides a
bifurcation Dice of 0.798 that is really **0.980 and 0.947** per tree — the global
figure was matching each tree's bifurcations against all 412 components' references,
410 of which are debris totalling under 8,000 voxels. After `optimise-skeleton`
de-looped, per-tree `χ` went to 0 for *both* trees; the whole-graph number cannot show
that the right tree was de-looped at all.

`--scope` defaults to `per-tree` on `skeletonise-all` when `--per-tree` is given,
since ranking a per-tree skeleton on the largest tree's `χ` is the blind spot the
split exists to close. Pass `--scope whole` to split the skeleton but still rank on
the published metric.

Candidates are ranked by `--objective` (`weighted` by voxel count, the default;
`mean`; or `sum`). An unweighted mean lets a 3,000-voxel fragment outvote the left
main, and a sum rescales with the tree count, so sweeps on different masks stop being
comparable.

### 6.2 Choosing the roots

Everything that turns a skeleton into anatomy needs to know where the blood comes in:
parent/child inference at a bifurcation, Strahler order, and which end `crop` must not
remove. By default these are guessed — `auto_roots` takes the largest-radius edge of
each component, then its lower-coordination endpoint — and the only manual override
was `--root-edge`, a segment id read out of a viewer by hand.

`pick-roots` opens **one 3-D window per tree**, largest first. Left-click the inlet
segment (it turns green), `q` to confirm and move to the next tree, `x` to stop, `c`
to clear. `--style` chooses what is drawn: `contour` rings plus centreline dots,
`tube` a solid tube per segment, or `lines` the graph itself — an edge per polyline
with a point at each node, nothing swept, and the only one that stays responsive on a
whole coronary tree:

```
python -m hipct_seg_debug.edit pick-roots skeletons/lee.am --roots-json roots.json
```

Then hand the sidecar to anything downstream, in place of `--root-edge`. **Every
command that assigns a Strahler order takes it**, and guesses with `auto_roots` when it
is not given:

```
python -m hipct_seg_debug.edit skeletonise --order --roots-json roots.json --out cand.am
python -m hipct_seg_debug.edit optimise cand.am --roots-json roots.json --out ordered.am
python -m hipct_seg_debug.edit optimise-skeleton lee.am --roots-json roots.json --out refined.am
python -m hipct_seg_debug.edit radius-perimeter refined.am --roots-json roots.json --out radius.am
python -m hipct_seg_debug.edit crop radius.am --roots-json roots.json --min-strahler 2
```

On `skeletonise` the flag is a *destination* under `--pick-roots` and a *source*
without it — re-deriving a skeleton is precisely what the sidecar's world coordinates
were recorded to survive.

Or skip the separate command: `skeletonise`, `skeletonise-all` and `optimise-skeleton`
all take **`--pick-roots`**, which opens the same picker once their own work is
finished and writes the same sidecar.

```
python -m hipct_seg_debug.edit skeletonise --order --pick-roots     --roots-json roots.json --out candidate.am
python -m hipct_seg_debug.edit skeletonise-all --out-dir cand/ --pick-roots     --roots-json roots.json
python -m hipct_seg_debug.edit optimise-skeleton lee.am --pick-roots     --roots-json roots.json --out refined.am
```

The window never opens mid-run. `skeletonise-all` derives and scores every candidate
first and then roots **the winner, once** — the sidecar keys on world coordinates, so
roots picked on the winning candidate resolve onto the others too, and clicking the
same two inlets per algorithm would be pure repeat. `optimise-skeleton` roots *after*
pruning and de-looping, because a root clicked on the input graph can sit on a spur
that pruning then deletes. `--pick-roots` and `--sweep` are refused together: a sweep
writes nothing, so there is no graph for the root to belong to.

Masks that already name their trees
-----------------------------------

An Avizo `.Regions.am` carries a `Materials` block — `Exterior`, `Left_Tree`,
`Right_Tree` — and the voxels hold the material's **index**, `0/1/2`. Where such a mask
is given, the split runs **by material first and by connectivity inside each**, so tree
0 is `Left_Tree` and tree 1 is `Right_Tree`, whatever the fragments do.

This is not cosmetic. At 32 µm the two coronaries touch, so the binarised `mask > 0`
labels them as one 26-connected component: connectivity alone *cannot* recover the
split, and "the two largest components" are then two fragments of one artery. The file
has said which is which all along. `--ignore-materials` restores the old behaviour.

`pick-roots` takes **several skeletons**, which is what makes a left-tree graph and a
right-tree graph rootable in one session:

```
python -m hipct_seg_debug.edit pick-roots left_tree.am right_tree.am     --seg 32.04um_artery_left_right.labels.Regions.am --roots-json roots.json
```

The GUI takes several the same way: `--graph left.am right.am`, or multi-select in the
Data tab's `skeleton (.am)` field. There they are **merged into one graph, each source
held as its own tree**, so the edit tools, `crop` and the writer all reach every one of
them — see [CLI.md](CLI.md#data).

The mask is decoded and split once and shared by both, so each graph is numbered
against the same labelling — the left one arrives as tree 0 and the right as tree 1 by
themselves — and one sidecar covers both, each root tagged with its material name.
Without a mask to anchor them, two single-tree graphs would both call themselves tree
0; they are renumbered apart, and the record says what it was renumbered from.

Both trees, and only the trees you care about
---------------------------------------------

The picker opens **one window per connected component, largest first**, so a mask
holding the left and right coronaries gives two windows and two roots. Debris
components are windows 3 onward — root the trees you want and press `x`. Every
component you skip keeps its automatic root: `order` fills them from `auto_roots`
before `order_forest` runs, because an unrooted component keeps order 0 throughout,
which is indistinguishable from a graph that was never ordered. Picking nothing at all
records nothing, rather than quietly writing the automatic roots you opened the window
to override.

No skeletonisation algorithm can be *driven* by a root — Lee thinning has no root
concept and `kimimaro` has no root argument — so the pick is recorded against the
graph they produce. That is also why it works identically for all three.

The sidecar records each root's **world coordinate** first, its segment key second and
its ids only as a note, because it has to survive a re-skeletonisation: a geometric
key survives a renumbering, but only a coordinate survives the segments being derived
afresh. A root that cannot be placed earns a note and falls back to the automatic
pick — never an error, since re-picking roots for a repaired graph is the intended
workflow. Add `--auto` (or run headless without PyVista) to record the automatic roots
without opening a window.

Qualify a smoother on your own data:

```
python -m hipct_seg_debug.edit optimise-skeleton candidate.am \
    --sweep "smoother=none,gaussian,savgol,bspline,multiscale"
python -m hipct_seg_debug.edit optimise-skeleton candidate.am \
    --smoother multiscale --sweep "drift-radius-factor=0.05,0.1,0.25,0.5"
```

## 7. Seeing a result in the GUI

Run the multiscale smoother and look at what it did:

```
python -m hipct_seg_debug.edit optimise-skeleton candidate.am \
    --smoother multiscale --out refined.am
```

Then in the 3D window's **control** dock: **Data** tab → **Load result**. The button
appears the moment a command writes a `.am`, is labelled with the filename, and reloads
the viewer on it in about 0.4 s — the frame, the lattice and the decoded slice caches are
kept, so only the skeleton changes. If you would rather open something else, put its path
in the `skeleton (.am)` field and press **Reload graph** instead.

The offer is never taken automatically. A graph you are in the middle of inspecting can
hold unsaved skeleton edits, and reloading would discard them, so the Log says what was
written and the button waits for you.

**Where things are written.** `skeletonise-all` writes every candidate to `--out-dir`,
default `skeletons/` relative to the working directory, and prints the absolute path when
it finishes. Run through the **Workflows** tab instead and everything lands in the run
directory shown there, which defaults to `cache/runs/<YYYYMMDD-HHMMSS>/` — workflow 12
puts the stride-4 sanity run in `quick/` and the stride-1 ranking in `candidates/`.

**Expect multiscale to do nothing here.** Its certified line search keeps under 0.5% of
its solution, so the graph it writes is within 0.02% of the unsmoothed one. That is
*not* the preserve-only freeze — two thirds of the graph is free to move and does — it is
that one new tube contact anywhere in 37k points vetoes the whole step. `smooth()` now
says so rather than reporting 24,037 points moved and leaving you to notice they moved
0.2 µm. Fixing re-centring halved the input overlaps and did not free it; see
[the multiscale section](#the-multiscale-smoother-is-inert-on-this-data-and-the-freeze-is-not-why).
