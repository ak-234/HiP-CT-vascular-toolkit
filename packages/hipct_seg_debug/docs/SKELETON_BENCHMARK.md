# Geodesic and curve-based skeletonisers against TEASAR

This page compares TEASAR (`kimimaro`) with three other candidates for collapsed
ex-vivo coronaries:

- the curve-based Jin minimum-cost path, as published (`jin-mcp`);
- a collapse-aware Jin variant (`jin-mgf`);
- a geodesic wave-front method (`geodesic-scoop`).

They are compared on phantoms with a known centreline and on two regions of
LADAF-2024-28.

The question comes from the methodology note on unpressurised HiP-CT. It argues that
a collapsed lumen breaks minimum-cost-path skeletonisation in two ways:

1. **Ladders.** A slit's medial set is a *sheet*. Collision impact is high across all
   of it, so the path has no unique ridge, and the 2 × FDT marking covers only the
   slit's half-thickness. The ribbons left unmarked on either side come back as
   parallel "ladder" branches.
2. **Radius bias.** Every distance-transform radius is the inscribed ball, which is
   the slit's *minor* semi-axis. Perimeter is what collapse conserves.

## The candidates

| name | family | module | what it is |
|---|---|---|---|
| `teasar` | minimum-cost path | `skeletonisers._teasar` | kimimaro TEASAR with this repo's defaults (scale 2.5, const 300 µm) |
| `teasar-readme` | minimum-cost path | `skeletonisers._teasar_readme` | the pipeline the [kimimaro README](https://github.com/seung-lab/kimimaro) documents: `skeletonize`, then `postprocess` (dust, loops, close-component joins, ticks), then `cross_sectional_area`. Soma detection stays off, because its defaults are for neurons |
| `jin-mcp` | minimum-cost path | `jin_mcp` | Jin et al., PRL 2016, as published |
| `jin-mcp-centroid` | hybrid | `skeletonisers._jin_mcp_centroid` | `jin-mcp` followed by centroid/curvature refinement |
| `jin-mgf` | minimum-cost path | `jin_mgf` | `jin-mcp` with a medial-geodesic medialness (Dey & Sun 2006) on sheet voxels and cross-section-filling marking; see [JIN_MCP.md](JIN_MCP.md#collapse-aware-variant-jin-mgf) |
| `geodesic-scoop` | wave-front | `geodesic_scoop` | geodesic shells from an inlet root; each shell component's centroid is a vertex. Keeps loops |

Every candidate returns a `Triple` and is scored identically.

## Running it

```powershell
# Phantoms. No dataset needed; about 25 minutes, mostly jin-mcp-centroid.
python -m hipct_seg_debug.edit.skeleton_synthetic_benchmark --out-dir $env:HIPCT_OUT\skel-bench --write-graphs

# A real region: every candidate, a TEASAR grid, and a JSON summary.
python -m hipct_seg_debug.edit skeletonise-all --seg $env:HIPCT_SEG --roi-zyx Z0 Z1 Y0 Y1 X0 X1 `
  --algorithms "teasar,teasar-readme[scale=2.5;const_um=132],jin-mcp,jin-mgf,geodesic-scoop" `
  --out-dir $env:HIPCT_OUT\roi --summary-json $env:HIPCT_OUT\roi\summary.json
```

`name[key=value;...]` sets a backend parameter. The output file is named after it, so
a whole grid fits in one `--algorithms` list.

## Phantoms

`edit/skeleton_phantoms.py` sweeps analytic centrelines with a perimeter-conserving
ellipse. For inflated radius `R0` and collapse ratio `λ = b/a`, Ramanujan's formula
gives `a = 2R0 / h(λ)`. The phantoms are rasterised at 2× supersampling with rounded
(half-ellipsoid) caps, at 10 µm voxels.

| phantom | geometry | truth |
|---|---|---|
| `round` | straight, R0 = 5 voxels | 2 terminals |
| `collapse-bands` | straight, R0 = 8, λ = 1, 0.5, 0.3, 0.2, 0.1 in bands | 2 terminals |
| `curved-collapsed` | sinusoid, R0 = 6, λ = 0.3, major axis half-twists | 2 terminals |
| `y-bifurcation` | round parent R0 = 6, Murray daughters, one at λ = 0.3 | 3 terminals, 1 junction |
| `loop` | anastomosis ring, R0 = 4 | 2 terminals, 2 junctions, β1 = 1 |
| `protrusions` | round tube with 12 bumps and 8 dents | 2 terminals |

The caps are rounded because a flat-ended slit's medial sheet genuinely forks into its
corners. With flat caps, every method grew a "false branch" that was really the
phantom.

### Metrics

- **err**: median and p95 distance from skeleton to the true centreline, in voxels.
  The end caps (1.5 R0) are excluded.
- **coverage**: the fraction of the true centreline within max(1.5, R0/2) voxels of
  the skeleton.
- **false / missing**: terminals matched one-to-one to the true ones within
  2R0 + 2 voxels.
- **ladder**: the share of true samples, away from junctions, whose lumen
  cross-section (within the major semi-axis) holds two or more skeleton segments.

## Results on the phantoms

Selected rows. The full tables, all 12 `teasar-readme` grid points and every `jin-mgf`
ablation are in the run's `summary.md`. `teasar-readme*` is the best grid point, scale
2.5 with const 100 µm (10 voxels), which the benchmark selects automatically.

**collapse-bands** (2 terminals, no junction):

| candidate | false | missing | ladder | err med / p95 |
|---|---|---|---|---|
| `teasar` (defaults) | 1 | 1 | 0.00 | 1.00 / 3.65 |
| `teasar-readme*` | 0 | 0 | 0.00 | 0.97 / 3.61 |
| `teasar-readme[scale=1.5;const_um=20]` | 18 | 0 | 0.73 | 7.56 / 10.77 |
| `jin-mcp` | **10** | 0 | **0.42** | 2.00 / 12.00 |
| `jin-mgf[medialness=lsf;marking=section]` | 0 | 0 | 0.00 | 1.01 / 4.00 |
| `jin-mgf[medialness=mgf;marking=dilate]` | 0 | 0 | 0.00 | 0.08 / 0.22 |
| `jin-mgf` (both) | 0 | 0 | 0.00 | **0.08 / 0.22** |
| `geodesic-scoop` | 0 | 0 | 0.00 | 0.16 / 0.24 |

Median position error (voxels) and ladder share by band:

| candidate | λ 1 | 0.5 | 0.3 | 0.2 | 0.1 |
|---|---|---|---|---|---|
| `teasar-readme*` | 0.22 / 0 | 0.17 / 0 | 1.00 / 0 | 1.59 / 0 | 1.24 / 0 |
| `jin-mcp` | 0.03 / 0 | 0.08 / 0 | 1.00 / 0.17 | 2.97 / **1.00** | 4.00 / **1.00** |
| `jin-mgf` | 0.05 / 0 | 0.12 / 0 | 0.19 / 0 | 0.06 / 0 | **0.01** / 0 |
| `geodesic-scoop` | 0.05 / 0 | 0.13 / 0 | 0.21 / 0 | 0.20 / 0 | 0.13 / 0 |

**The other phantoms** (false, missing, err med; β1 for the loop):

| candidate | curved-collapsed | y-bifurcation (junction err) | loop | protrusions |
|---|---|---|---|---|
| `teasar-readme*` | 0, 0, 0.82 | 0, 0, 0.25 (2.00) | 1, 0, β1 **0** | 0, 0, 0.80 |
| `jin-mcp` | 1, 0, 0.65 (ladder 0.15) | 0, 0, 0.22 (1.00) | refused (Euler ≠ 1) | 0, 0, 1.00 |
| `jin-mcp-centroid` | 1, 0, 0.23 | 0, 0, 0.10 (**0.38**) | refused | 0, 0, 0.29 |
| `jin-mgf` | 0, 0, **0.24** | 0, 0, 0.17 (1.00) | refused | 0, 0, 1.00 |
| `jin-mgf[mgf_mode=rim]` | 0, 0, 1.05 | 0, 0, 0.23 (1.00) | refused | 0, 0, 1.00 |
| `geodesic-scoop` | 0, 0, **0.24** | 0, 0, 0.14 (6.17) | 0, 0, β1 **1** | 0, 0, **0.21** |

What this supports:

- **The ladder prediction holds for published Jin.** It ladders through every
  cross-section at λ ≤ 0.2 and gives 10 false branches on one straight vessel.
  `jin-mgf` removes the ladders and puts the path on the slit's mid-line (0.01 voxels
  at λ = 0.1).
- **The two `jin-mgf` modifications do different jobs.** Section marking alone stops
  the ladders but leaves the path wandering across the sheet (4.0 voxels at
  λ = 0.1). The MGF medialness alone fixes both, because ridge-weighted
  significance no longer accepts sheet branches. Together is the default.
- **TEASAR is not ladder-free by nature; it is parameter-dependent.** At small `const`
  it ladders worse than Jin (18 false branches, 73 % ladder). At the best grid point
  it is clean but runs about 1–1.6 voxels off-centre in collapsed bands: its PDRF
  ridge is the same flat sheet. The note's argument applies to TEASAR too.
- **Only `geodesic-scoop` keeps a loop.** Both Jin variants refuse non-tree input by
  design, and TEASAR returns a tree with a false branch where the ring was cut.
- **Weak spots, stated plainly.**
  - `geodesic-scoop` places junctions 6 voxels off, because a geodesic shell is oblique
    at a fork and its centroid lags. That is the family's known bias.
    `jin-mcp-centroid` has the best junctions.
  - The `rim` MGF proxy is cheaper but loses centring on the twisted phantom; `exact`
    is the default.

### Radii (collapse-bands)

Median `r / R0` by band: r_DT (EDT inscribed), r_A (stable transverse cut area), and
r_inflated (`radius_perimeter.measure_radii`). These are identical across candidates
to two decimals, because on a straight slit every centred path cuts the same sections.

| | λ 1 | 0.5 | 0.3 | 0.2 | 0.1 |
|---|---|---|---|---|---|
| analytic r_DT, r_A, r_inflated | 1.00, 1.00, 1.00 | 0.65, 0.92, 1.00 | 0.43, 0.78, 1.00 | 0.30, 0.67, 1.00 | 0.15, 0.49, 1.00 |
| measured | 1.03, 1.03, 1.05 | 0.75, 0.93, 1.02 | 0.50, 0.80, 0.99 | 0.38, 0.71, 1.03 | 0.25, 0.56, 1.00 |
| kimimaro `cross_sectional_area` | 1.03 | 0.93 | 0.80 | 0.71 | 0.57 |

- **The perimeter radius recovers R0 within 5 % at every collapse ratio.** The
  inscribed radius falls to a quarter of it.
- kimimaro's own section area agrees with this repo's r_A to 0.01, and is equally
  biased: area is not conserved by collapse.
- At λ = 0.1, voxelisation keeps the measured r_DT (0.25) above the analytic value
  (0.15). The slit is only 2.5 voxels thick.
- In Poiseuille terms (r⁻⁴), r_DT overstates segment resistance **29×** at λ = 0.3
  and **125×** at λ = 0.2 analytically (16× and 48× with the voxelised r_DT measured
  here). That is not "an order of magnitude"; at λ = 0.2 it is two.

## Results on LADAF-2024-28

The source is a stride-1 lattice at 66 µm. The Avizo ASCII graph supplied candidate
junctions, and each ROI was screened for tree topology and EDT-Hessian sheet fraction.
Scores are the super metric (`coronary` preset, whole-graph scope). There is no true
centreline here, so these rank agreement with the mask.

- **node 306**: ROI `1047 1167 707 827 399 519`; one component, χ = 1; median radius
  9.8 voxels; 96 % sheet-like; 4 reference bifurcations.
- **node 88**: ROI `889 1009 1033 1153 322 442`; run with `--per-tree`, 2 trees kept;
  94 % sheet-like.

| candidate | node 306 M_S | B dice | segments | node 88 M_S | B dice | segments |
|---|---|---|---|---|---|---|
| `teasar` (defaults) | ∞ (2 pieces) | 0.00 | 2 | 36.1 | 0.15 | 45 |
| `teasar-readme[scale=2.5;const_um=132]` | **1.04** | 0.67 | 5 | 156.5 | 0.08 | 42 |
| `teasar-readme[scale=1.5;const_um=660]` | 1.22 | 0.60 | 13 | ∞ | 0.00 | 16 |
| `teasar-readme[scale=4;const_um=132]` | 4.07 | 0.40 | 3 | 30.5 | 0.17 | 14 |
| `teasar-readme[scale=1.5;const_um=132]` | 26.8 | 0.20 | 80 | 72.1 | 0.11 | 63 |
| `jin-mcp` | 6.38 | 0.33 | 5 | 133.0 | 0.08 | 39 |
| `jin-mgf` | 4.12 | 0.40 | 3 | ∞ | 0.00 | 2 |
| `jin-mgf[mgf_mode=rim]` | 6.31 | 0.33 | 5 | 16.7 | 0.22 | 8 |
| `geodesic-scoop` | 1.66 | 0.57 | 7 | **1.39** | **0.67** | 8 |

What this supports:

- **No TEASAR setting is good in both regions.** The best node-306 point is
  second-worst on node 88. Across the 12-point grid, M_S spans 1.04 to 26.8 on one
  region and 30.5 to ∞ on the other. This is the Walsh et al. finding, reproduced: the
  parameters have to be qualified per dataset.
- **Plain `teasar` returned a single-component vessel in two disconnected pieces** on
  node 306. The README `postprocess` re-joins them.
- **`geodesic-scoop` is the only candidate near the top in both regions,** at its
  default parameters. Its V term (0.30–0.34) reflects max-EDT placeholder radii, not
  geometry.
- **`jin-mgf` under-detects branches on real data.** On node 88's larger tree it
  stops after 2 rounds with 23,626 voxels unexplained. The ridge weighting that
  removes ladders also lowers the significance of real side branches leaving a sheet,
  below Jin's α + β·DT. Published `jin-mcp` stalls similarly (24,312 voxels) and
  adds false branches instead. Re-deriving α for this resolution, as the note says,
  is the open item.

**Perimeter radii on node 306** (`radius-perimeter` run on each candidate):

| candidate | placeholder median (µm) | perimeter median (µm) | ratio median (p5–p95) |
|---|---|---|---|
| `teasar-readme[scale=2.5;const_um=132]` | 681 | 1150 | 1.66 (1.18–1.84) |
| `geodesic-scoop` | 663 | 1180 | 1.74 (1.39–3.46) |
| `jin-mgf` | 638 | 1142 | 1.82 (1.26–4.28) |
| `jin-mcp` | 608 | 1134 | 1.81 (1.26–5.53) |

The perimeter radius agrees to within 4 % whichever skeleton it is measured on. The
inscribed placeholders differ by 12 %, and all are about 0.58 of the perimeter
radius, which is what the phantoms show at λ ≈ 0.3. Here, choosing a skeleton matters
for topology and much less for the radius once it is measured by perimeter.

## Caveats

- Phantoms test the behaviour of each method, not anatomy. Real-data scores measure
  agreement with the segmentation, whose own bifurcation reference comes from a Lee
  skeleton.
- Two ROIs are not a qualification. The ROIs were chosen to be tree-like so that Jin
  could run at all.
- `jin-mgf` and `jin-mcp` are bounded-ROI (≤ 2 M voxels), isotropic-only, one tree
  per extraction. `geodesic-scoop` has none of those limits but is O(N) in memory with
  several voxel-sized arrays.
- **kimimaro under the pinned osteoid.** osteoid 0.6.0 is pinned for NumPy 1.26.
  `_osteoid_uint32_edges` works around three incompatibilities with kimimaro 5.8:
  - `uint32` edges in `create_distance_graph`;
  - float indices in `components()`;
  - `postprocess` returning voxel-space skeletons, which made every
    `cross_sectional_area` zero.

  Without these, `teasar-readme` crashes or silently reports zero areas.
