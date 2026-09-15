# Strahler-order morphometry, haemodynamics and surface validation

Three modules turn a spatial graph, a CFX solution and a reconstructed surface into
publication figures, with every plotted number backed by a CSV.

| Module | Role |
|---|---|
| `cfx_extract.py` | pull wall + volume fields out of an ANSYS CFX `.res` via CFD-Post batch |
| `strahler_analysis.py` | join graph + CFD into per-segment and per-order tables |
| `strahler_plots.py` | publication figures from those tables |
| `strahler_combined.py` | the same tables for left alone, right alone and the two pooled |
| `strahler_combined_plots.py` | overlay figures for that three-way comparison |
| `surface_validation.py` | score a reconstructed STL against the graph |

---

## 1. Extract CFD fields

`.res` files are a proprietary container, so fields are exported by driving
CFD-Post in batch. ANSYS CFX must be installed; the module finds `cfx5post`
automatically (override with `CFX5POST`).

```bash
python -m coronary_sdf.cfx_extract \
  --res ".../ratio_6/left_tree_ratio_6_001.res" \
  --res ".../ratio_6/right_tree_ratio_6_001.res" \
  --out analysis_out/cfd_extract
```

Per run it writes `<stem>_wall.npz` (`X, Y, Z, Pressure, Wall Shear`) and
`<stem>_volume.npz` (`… Velocity u/v/w, Velocity, Volume of Finite Volumes`),
keeping the intermediate CSVs so the export stays auditable. Re-running skips
work already done unless `--force` is passed.

**Why the control volume is exported.** CFX inflates the boundary layer, so nodes
are far denser near the wall than in the core. Averaging velocity over raw node
counts weights the slow near-wall fluid several times too heavily — on the ratio_6
trees that halved the reported velocity. The analysis therefore forms a
volume-weighted mean using `Volume of Finite Volumes`.

## 2. Build the tables

Anatomy and haemodynamics may come from different graphs: the imaged tree is
normally larger than the geometry that was meshed and solved. `--graph` supplies
the anatomical tree (best statistics), `--cfd-graph` the tree the solver ran on.

```bash
python -m coronary_sdf.strahler_analysis \
  --graph ".../pruned.am.xml" \
  --cfd-graph ".../ratio_6/model.am.xml" \
  --cfd-run left:left_tree_ratio_6_001 \
  --cfd-run right:right_tree_ratio_6_001 \
  --cfd-dir analysis_out/cfd_extract \
  --out analysis_out/ratio_6
```

Outputs:

| File | Contents |
|---|---|
| `segments.csv` | one row per anatomical vessel segment — the audit trail |
| `segments_cfd.csv` | the same for the solved geometry, with the CFD columns |
| `by_strahler.csv` | per-order mean / SD / SEM / median / min / max / sum / n |
| `by_strahler_cfd.csv` | the same for the CFD tree |
| `by_radius_bin.csv` | the metrics binned by vessel radius (log bins by default) |
| `bifurcations.csv` | bifurcation counts per order |
| `provenance.json` | inputs, units, options, bin edges |

**Conventions.**

- Coordinates and radii arrive in µm from `parse_amira` and are converted to mm.
- Points within `--bif-skip` (10) contours of a degree ≥ 3 node are excluded from
  radius and field statistics: the skeletoniser inflates thickness at junctions
  and the solver sees three-dimensional flow there, so neither describes a vessel.
- A CFD node joins a segment only within `--radius-factor` (1.5) × the local graph
  radius, which stops a neighbouring vessel bleeding into the average.
- A bifurcation is a node of degree ≥ 3, attributed to the **parent** order (the
  highest order incident on it), so "bifurcations of order n" means "places where
  an order-n vessel divides".
- Cross-sectional area is `π r̄²` per segment; the per-order **sum** is the total
  CSA of that order, the **mean** is the average vessel.
- Flow rate is `Q = ⟨|v|⟩ · A` from the volume-weighted mean speed.

### Metric definitions and formulas

This section defines the quantities written by `strahler_analysis.py`. Unless
stated otherwise, coordinates and radii are converted from micrometres to
millimetres before calculation. A bar labelled "vessel segment count" in the
figures is `n_vessels` in the tables. With `--elements` (the recommended
morphometry setting), one counted vessel segment is one **Strahler element**: a
maximal connected run of graph segments having the same order. Without
`--elements`, one row is one raw graph segment, so counts and per-order calibre
statistics will differ.

#### Strahler order and analysis rows

`strahler_analysis.py` reads each graph segment's stored `strahler` value; it
does not recompute the order. The conventional definition represented by that
field is:

```text
S(parent) = max(S1, S2)          if S1 != S2
S(parent) = S1 + 1               if S1 == S2
S(terminal) = 1
```

Here `S1` and `S2` are daughter orders. For a multifurcation, the equivalent
rule is to increase the maximum daughter order by one only when that maximum
occurs at least twice.

When `--elements` is enabled, two same-order graph segments are joined only when
they are the only two segments of that highest order incident at their shared
node. This preserves a genuine equal-order trifurcation as separate elements.
The element identifier is the smallest constituent graph-segment id.

#### Junction masking

Let `k = --bif-skip` (default 10). At any segment end whose recomputed node degree
is at least three, the nearest `min(k, n_points)` centreline contours are excluded
from radius and CFD assignment. If masking both ends would remove every point,
the complete segment is retained. This mask prevents junction dilation from
biasing calibre and prevents locally three-dimensional junction flow from being
reported as ordinary vessel flow. Segment length still uses the complete
polyline.

#### Per-segment anatomical metrics

For retained centreline radii `r_i` and full centreline coordinates `x_i`:

| CSV metric | Calculation |
|---|---|
| `n_points` | Number of centreline points before junction masking. |
| `n_points_used` | Number of centreline points retained after junction masking. |
| `length_mm` | Polyline arc length: `L = sum_i ||x_(i+1) - x_i||`. All centreline points are used. |
| `radius_mean_mm` | Arithmetic mean: `r_bar = (1/N) sum_i r_i`, using retained points. |
| `radius_sd_mm` | Sample SD: `sqrt(sum_i (r_i-r_bar)^2 / (N-1))`; zero when `N = 1`. |
| `radius_min_mm`, `radius_max_mm` | Minimum and maximum retained radius. |
| `diameter_mean_mm_*` | Per-order reporting columns derived as `d = 2r`. Every radius statistic is doubled; the associated `_n` is unchanged. |
| `csa_mm2` | Representative cross-sectional area: `A = pi r_bar^2`. |
| `volume_mm3` | Cylindrical segment estimate: `V = A L = pi r_bar^2 L`. |

The area is based on the segment mean radius, not the mean of the pointwise areas
`pi r_i^2`. Consequently `volume_mm3` is a centreline-derived geometric estimate,
not a voxel count or a surface-mesh volume.

#### Collapsing graph segments into Strahler elements

For an element containing graph segments `j`, with segment lengths `L_j`, any
intensive metric `y` is collapsed using the length-weighted mean:

```text
y_element = sum_j (L_j y_j) / sum_j L_j.
```

In particular, `r_element` is the length-weighted mean segment radius and the
element CSA is `pi r_element^2`. Element length and volume are additive:

```text
L_element = sum_j L_j
V_element = sum_j V_j.
```

The element radius variance includes variation within segments and taper between
segment means:

```text
variance_element = sum_j L_j [sd_j^2 + (r_j - r_element)^2] / sum_j L_j
sd_element = sqrt(variance_element).
```

`n_points`, `n_points_used`, `n_wall_nodes`, and `n_volume_nodes` are summed over
the constituent graph segments. CFD means and SD columns are length-weighted in
the same way as other intensive metrics after their initial node-level
calculation.

#### Counts and extensive per-order metrics

| CSV metric | Calculation |
|---|---|
| `n_vessels` | Number of analysis rows in the order: Strahler elements with `--elements`, otherwise raw graph segments. |
| `n_terminal_branches` | Number of segments with at least one endpoint of degree one. Segments named by `--root-seg` are omitted because the ostial inlet is not a distal terminal. |
| `n_bifurcations` | Number of nodes with degree at least three, assigned to `max(incident Strahler orders)`, i.e. the parent order. |
| `total_length_mm` / `length_total_mm` | `sum_j L_j` for all rows in the order. |
| `csa_mm2_sum` / `csa_total_mm2` | `sum_j pi r_bar,j^2`: one representative cross-section per analysis row. It is not the area of a single physical plane. |
| `volume_total_mm3` | `sum_j V_j` for all rows in the order. |

The compact `morphometry_table.csv` contains the same counts and extensive
totals. Its diameter median, Q25 and Q75 are calculated over the individual
analysis-row radii and then multiplied by two. The `total` row pools all analysis
rows before taking those percentiles; it does not average the per-order
percentiles.

#### Assignment of CFD nodes to vessels

Each wall or volume node is assigned to the segment containing its nearest
retained centreline point. If the node-to-centreline distance is `d` and the
local graph radius is `r`, assignment is accepted only when:

```text
d <= (--radius-factor) r,
```

with a default radius factor of 1.5. Nodes failing the gate receive no owner and
do not contribute to any vessel statistic. `n_wall_nodes` and `n_volume_nodes`
are accepted node counts, not weighted effective sample sizes.

For values `y_i` with weights `w_i`, node-level means and SDs are:

```text
mean_w(y) = sum_i w_i y_i / sum_i w_i
sd_w(y)   = sqrt(max(0, sum_i w_i y_i^2 / sum_i w_i - mean_w(y)^2)).
```

Wall quantities use `Surface Control Area` as `w_i` when available and otherwise
use equal weights. Lumen quantities use `Volume of Finite Volumes`; this avoids
overweighting the densely sampled boundary layer. These are weighted population
SDs at node level, unlike the sample SD used across vessels in the per-order
tables.

#### Per-segment haemodynamic metrics

Let `v_i = (u_i, v_i, w_i)` be velocity, `|v_i|` its speed, `t_i` the unit local
centreline tangent, `v_ax,i = v_i . t_i`, `A = pi r_bar^2`, and `C = 6.0e7` the
conversion from cubic metres per second to millilitres per minute.

| CSV metric | Calculation |
|---|---|
| `wss_mean_pa`, `wss_sd_pa` | Area-weighted mean and SD of CFX `Wall Shear`, the wall-shear-stress magnitude, in Pa. |
| `pressure_wall_mean_mmhg`, `pressure_wall_sd_mmhg` | Area-weighted wall pressure and SD, multiplied by `1/133.322387415` to convert Pa to mmHg. `--pressure-offset-mmhg` is added to the mean only. |
| `velocity_mean_ms`, `velocity_sd_ms` | Control-volume-weighted mean and SD of speed `|v|`, in m/s. If the scalar is absent, speed is reconstructed as `sqrt(u^2+v^2+w^2)`. |
| `pressure_lumen_mean_mmhg`, `pressure_lumen_sd_mmhg` | Control-volume-weighted lumen pressure and SD, converted from Pa to mmHg; the requested pressure offset is added to the mean only. |
| `velocity_axial_mean_ms` | `mean_w(|v_ax|)`. The absolute value removes the arbitrary sign introduced by graph point ordering. |
| `axial_fraction` | `mean_w(|v_ax|) / mean_w(|v|)`. Values near one indicate predominantly centreline-aligned motion. |
| `flow_coherence` | `|mean_w(sign(v_ax))|`. One means all accepted nodes move in one axial direction; zero means equal weighted motion in both directions. |
| `flow_speed_ml_min` | `mean_w(|v|) A 1e-6 C`, equivalently `60 mean_w(|v|) A` for `A` in mm^2. This is a mean-speed x area estimate, not a conserved flux. |
| `flow_axial_ml_min` | `mean_w(|v_ax|) A 1e-6 C`, equivalently `60 mean_w(|v_ax|) A`. This removes non-axial speed but still is not a cross-sectional integral. |

Both flow estimates are non-negative. Neither evaluates the physical flux
`integral_A v . n dA`; use CFX `massFlow()` when conservation or inlet/outlet
flow splits are required.

#### Per-order and per-radius summary statistics

For each metric, non-finite values are removed independently. Therefore a
metric's `*_n` can be smaller than `n_vessels`, especially when CFD does not
cover the whole anatomical tree. For the remaining `N` values `y_i`, the output
suffixes mean:

| Suffix | Formula or definition |
|---|---|
| `_n` | Number of finite values, `N`. |
| `_mean` | `(1/N) sum_i y_i`. |
| `_sd` | Sample SD, `sqrt(sum_i (y_i-y_bar)^2/(N-1))`; zero for one value. |
| `_sem` | `_sd / sqrt(N)`. |
| `_median` | 50th percentile. |
| `_q25`, `_q75` | 25th and 75th percentiles using NumPy's default linear percentile interpolation. |
| `_iqr` | `_q75 - _q25`. |
| `_min`, `_max` | Smallest and largest finite value. |
| `_sum` | Sum of finite values. For intensive metrics this is retained for completeness and is not generally a physical total. |

`by_strahler.csv` groups rows by stored Strahler order. By default,
`by_radius_bin.csv` uses `B` logarithmically spaced edges from the minimum
positive mean radius to `1.001 x` the maximum:

```text
edge_j = 10^[log10(r_min) + (j/B)(log10(1.001 r_max)-log10(r_min))]
radius_mid = sqrt(edge_j edge_(j+1)).
```

`--linear-bins` instead uses equally spaced edges and the arithmetic midpoint.
The factor 1.001 ensures that the largest observed vessel falls inside the last
half-open bin.

## 3. Figures

```bash
python -m coronary_sdf.strahler_plots \
  --in analysis_out/ratio_6 \
  --validation analysis_out/validation
```

| Figure | Panels |
|---|---|
| `fig1_morphometry` | mean radius, total CSA, vessel count, bifurcations |
| `fig2_haemodynamics` | WSS, wall pressure, velocity, flow rate by order |
| `fig3_by_radius` | the same four against vessel radius |
| `fig4_distributions` | per-order box plots of the raw per-segment values |
| `fig5_surface_validation` | graph vs reconstructed radius, and Bland–Altman |
| `single_panels/` | one single-column panel per metric, for slides |

Each is written as 600 dpi PNG plus vector PDF and SVG.

**Bars show the median with the interquartile range.** This is the convention in
cardiovascular-mechanics papers for these quantities, and it matters here: the
distributions are strongly right-skewed, so the mean sits far above the median and
the standard deviation exceeds it. On the left full tree, order-1 wall shear has a
mean of 11.1 Pa but a median of 2.4 Pa — the mean is set by a handful of extreme
segments, and a symmetric mean ± SD bar reaches below zero, which is impossible for
a magnitude. `--summary mean-sd` restores mean ± SD (the morphometry convention)
and `mean-sem` the standard error; every table carries `mean`, `sd`, `sem`,
`median`, `q25`, `q75`, `iqr`, `min`, `max`, `sum` and `n`, so any of these can be
plotted without recomputing.

`fig4` (box plots) remains the fullest view — median, IQR, whiskers and the
individual outliers behind the summary.

### The solves are steady state

The CFX runs behind these figures are **steady-state**, so wall shear is the
steady WSS at the converged operating point — *not* a cycle-averaged TAWSS, and no
oscillatory measure (OSI, RRT, transverse WSS) is defined for it. Label it plainly
as steady WSS, and note the operating point: these are resting-flow solves with a
prescribed inlet mass flow, not hyperaemic.

### Sign conventions: what CFX exports, and what these scripts do with it

Checked 2026-08-31 against the actual exports in `analysis_out/cfd_extract/`
(913,147 wall and 5,452,329 volume nodes on the left tree; 245,872 and 1,446,328
on the right), not from documentation. Percentages below are the fraction of
exported nodes carrying a negative value.

| CFX variable | in the export | sign |
|---|---|---|
| `Wall Shear` | 0.0019 – 139.2 Pa | **never negative** (0.0% of 1.16 M wall nodes) |
| `Pressure` | −7508 – 1206 Pa | negative at 6.5% (left) / 26.6% (right) of nodes |
| `Velocity u/v/w` | ±0.86 m s⁻¹ | signed; 15–56% negative depending on component and tree |
| `Velocity` | 2.6 × 10⁻⁸ – 1.003 m s⁻¹ | **never negative** |

**`Wall Shear` is a magnitude, not a signed quantity.** CFD-Post's scalar
`Wall Shear` is `|τ_w|`; the direction lives in `Wall Shear X/Y/Z`, which
`cfx_extract.WALL_VARIABLES` does **not** export. The measured 0.0% negative over
1.16 M nodes confirms it. Two consequences: taking `abs()` of it anywhere is a
no-op, and no directional wall metric — OSI, RRT, transverse WSS, reversal
fraction — can be computed from what is currently exported, independently of the
steady-state argument above. Add the components to `WALL_VARIABLES` first if any
such metric is ever wanted.

**`Pressure` is relative, but not for the reason the code says.** CFX reports
`Pressure` relative to the domain `Reference Pressure`, and absolute pressure is
`Pressure + Reference Pressure`. The comment at `strahler_analysis.py:64` states
that 1 atm is subtracted "in these runs" — but `flow_fractions.py:623` writes
`Reference Pressure = 0 [atm]` into the CCL, so nothing is subtracted and the
exported number *is* the absolute pressure as far as the solver is concerned. What
actually makes the field gauge is the boundary condition: every opening and outlet
is pinned at `Relative Pressure = 0 [Pa]` (`flow_fractions.py:698, 717`), so the
field is referenced **to the outlets**. That is why negative values appear at all —
they are static pressures below the outlet reference, not below vacuum.

The reported conclusion is unchanged (pressure differences are what mean anything,
and left/right are separate solves so only within-tree comparisons hold), but the
stated mechanism is wrong and matters for one thing: `pressure_offset`. If you add
an offset believing 1 atm was already removed, you double-count. Treat
`pressure_offset` as "shift the outlet-referenced field to a physiological
coronary pressure", and pick it from the intended distal pressure, not from
atmospheric.

**Flow rate is computed from a speed magnitude, so it cannot be signed.**
`strahler_analysis.py:410` computes `Q = <|v|> · A` from the scalar `Velocity`,
which the table above confirms is strictly positive. This is *not* the flux
`∫ v·n̂ dA`, and the difference is a bias, not just a sign convention:

* `|v| ≥ v·n̂` at every node, so any secondary, swirl or radial component inflates
  `Q`. The velocity components are 15–56% negative, so the flow is genuinely not
  axis-aligned everywhere.
* the average is taken over **volume** nodes owned by the segment (weighted by
  `Volume of Finite Volumes`), not over a cross-section, so it mixes near-wall and
  core velocities rather than integrating a profile.
* retrograde flow is counted as forward. Recirculation adds to `Q` instead of
  subtracting from it.

So `flow_ml_min` is an upper bound on the throughput, and the tree's flows will not
sum consistently across a bifurcation. If flow conservation or a flow split is
wanted, take it from CFX directly — `massFlow()` on the inlet/outlet locators, via
`flow_fractions.py`, which is the module built for it — rather than from this
column. **Do not present `flow_ml_min` as a conserved flow rate**; present it as a
mean-speed × area estimate, or replace it before it goes in a figure.

**If you ever use `massFlow()` or `areaInt()` in CFD-Post**, note that CFX's
boundary normals point **out of the fluid domain**, so an inlet with flow entering
returns a negative value under that convention. This has not been relied on
anywhere in these scripts — it is recorded here because it is the sign convention
that bites when flow splits are computed — and it should be confirmed against the
run rather than assumed, by checking that the inlet and the outlets carry opposite
signs and sum to ~0.

## 3b. Left, right, and the two pooled

`strahler_analysis` takes one graph and one output directory, which cannot answer
"does pooling the two trees change the answer?". Two facts about the inputs are
why, and both are easy to miss:

**A coronary spatial graph holds both trees.** `pruned.am.xml` and the `ratio_8`
model each carry the left and right trees as *disjoint connected components* (the
full skeleton also carries two orphan single-segment fragments). So
`--graph pruned.am.xml --cfd-run left:...` gives morphometry over **both** trees
while the haemodynamics cover only the left — the other tree's CFD columns stay
NaN and drop out of `_stats`. The `left_full` and `strahler_right_ratio8` outputs
have exactly this shape: their `by_strahler_cfd.csv` is per-tree, but their
`by_strahler.csv` is not.

**The two trees came off different graphs.** The left solve ran on the full
skeleton, the right on the ratio-8 prune, so no single `--graph` serves both. The
combined analysis therefore pools *per-segment records*, not graph files — which
is also the correct operation, since Strahler order is a property of a rooted
tree and must be assigned within each tree before anything is pooled.

```bash
python -m coronary_sdf.strahler_combined \
  --left-graph  ".../pruned.am.xml" --left-root-seg 67 \
  --left-run    "meshmixer_..._001" --left-cfd-dir  analysis_out/cfd_extract_full \
  --left-inflow-kgs 0.000897915 \
  --right-graph ".../ratio_8/model.am.xml" --right-root-seg 291 \
  --right-run   "right_tree_ratio_8_001" \
  --right-cfd-dir analysis_out/cfd_extract_right_ratio8 \
  --right-inflow-kgs 0.000384821 \
  --out analysis_out/combined_lr

python -m coronary_sdf.strahler_combined_plots --in analysis_out/combined_lr
```

`run_strahler_combined_lr.ps1` wraps both with the LADAF_2024_28 paths filled in
and also runs `strahler_plots` over each of the three directories.

**Trees are named by a root segment id**, not by size or order — which component
is larger depends on the pruning, so "largest component" would silently swap the
two between graphs. Segment 67 names the left tree (its order-5 ostial trunk),
291 the right. Each root is then **dropped from every statistic**: an ex vivo
ostial stub is cannulated, so its segmented radius is corrupt, and being the sole
member of its order it would otherwise define that order single-handedly.
`root_check` in `provenance.json` records whether the named segment really was
top-order with a free end.

Outputs — `left/`, `right/` and `combined/` each in the layout
`strahler_plots --in <dir>` expects, plus:

| File | Contents |
|---|---|
| `by_radius_bin.csv` | bins fitted to that table's own radius range — what a standalone run gives |
| `by_radius_bin_shared.csv` | one edge set fitted to the pooled records, the only bin-for-bin comparable version |
| `comparison/compare_by_strahler.csv` | left \| right \| combined per order, every metric and statistic |
| `comparison/compare_by_radius.csv` | the same over the shared radius bins |
| `comparison/pooling_effect.csv` | the headline columns alone, for both scopes |
| `comparison/summary.md` | inputs, operating point, and what actually differs |
| `comparison/figures/` | the figures below |

| Figure | Panels |
|---|---|
| `fig_compare_by_strahler` | radius, CSA, WSS, axial velocity by order — left, right and pooled |
| `fig_compare_by_radius` | vessel count, WSS, axial velocity, axial flow over shared radius bins, all three series |
| `fig_cfd_by_radius` | WSS, velocity, axial velocity, axial flow by radius — **left and right only**, no pooled curve |
| `fig_pooling_shift` | pooled median vs the n-weighted mean of the parts, per order |
| `single_panels/fig_radius_<metric>` | one slide-sized panel per CFD metric, left and right overlaid |

`fig_cfd_by_radius` and the single panels are the ones to read for "how do the two
trees differ": nothing is mixed across the solves, so every curve is one tree's own
result. The pooled series appears only where the question is what pooling does.
Both pressures get a single panel — the two curves sit on different zeros, so the
panel is annotated and only the within-tree gradient down the calibre range should
be read.

Dots with IQR whiskers rather than bars: three of these four metrics span more
than a decade and are drawn on a log axis, where a bar has no zero to measure its
length from.

Two derived columns carry the comparison. `<metric>_right_vs_left_pct` is the
contrast between the trees — the reason pooling can move a number at all.
`<metric>_pool_delta_pct` is how far the pooled median sits from the n-weighted
mean of the two per-tree medians; a pooled median is *not* the weighted mean of
its parts, so this is not an error term but the size of the effect pooling itself
introduces.

**Three things not to read off these tables.**

- **Pressure does not pool.** Left and right are separate solves, each with its
  static pressure referenced to its own outlets at 0 Pa (see "Sign conventions").
  The combined pressure columns show how far apart the two solves sit; they are
  not a coronary pressure distribution, and the comparison figures omit them.
- **Strahler order is relative to its own tree.** After the roots are dropped both
  trees span orders 1–4, but the left is a 5-order tree and the right a 4-order
  one, so order *n* sits one generation nearer the ostium on the right. The radius
  tables are the comparison that carries no such assumption, since a radius bin
  means the same physical calibre in both.
- **The flow split is an input.** Each solve has its inlet mass flow prescribed —
  here 0.000897915 and 0.000384821 kg s⁻¹, a 70/30 left/right split at
  1050 kg m⁻³. Wall shear and velocity scale with it, so a left-vs-right
  haemodynamic contrast reports anatomy *and* the flow each tree was given.
  `--left-inflow-kgs` / `--right-inflow-kgs` only record it into `provenance.json`
  and `summary.md`; no computed quantity depends on them.

## 4. Surface validation

The graph is the ground truth: it carries a measured radius at every centreline
point, so no second segmentation is needed.

```bash
python -m coronary_sdf.surface_validation \
  --graph ".../pruned.am.xml" \
  --stl ".../LADAF_2024_28_SDF/lumen_bspline.stl" \
  --out analysis_out/validation --voxel-mm 0.15
```

Only **non-bifurcation** stations are scored, for the same reason they are dropped
from the morphometry: at a junction the graph describes one vessel while the
surface is a blended confluence, so any disagreement measures the blend.

### Cropping: absent geometry is not failed geometry

A surface is normally built from a *cropped* graph — an ROI, a clipped distal
extent, one artery of two. Graph the reconstruction never covered must not be
scored as reconstruction error, so every station is first classified:

| State | Test | Meaning |
|---|---|---|
| `inside` | enclosed by the surface | the reconstruction covers this vessel |
| `missed` | not enclosed, but surface within the margin | a real defect — a break, a gap, or a wall pulled inside the centreline |
| `cropped` | not enclosed and no surface within the margin | removed before reconstruction; excluded from every score |

The margin is `max(3 × local radius, 0.5 mm)`, scaling with vessel size because a
large vessel leaves a correspondingly large gap when it breaks. Both knobs are
module constants (`CROP_MARGIN_RADIUS_FACTOR`, `CROP_MARGIN_MIN_MM`) and the
resolved values are echoed into `validation_summary.json`.

The classification drives everything downstream: radius agreement uses `inside`
stations, and the ground-truth volume behind clDice and Dice is rasterised from
`inside + missed` only. Rasterising the cropped vessels too would inflate the true
volume with geometry never requested, depressing Dice and Tsens for no real
defect. Because only enclosed stations are ray-cast, this also makes the radius
measurement substantially faster.

**The coverage split is a diagnostic, not a result.** It is deliberately kept out
of the figures and the score CSVs — a reader of the figures should not have to
interpret coverage. It surfaces in three places instead: the
`surface_validation` console log, the `coverage_per_tree` block of
`validation_summary.json`, and a flag printed by the figure script:

```
[plots][coverage] 8992 inside, 295 missed, 7009 cropped of 16296 stations
[plots][WARN] 43% of centreline stations lie outside the reconstruction and are
              excluded from every score; the figures describe only the part of
              the graph the surface spans.
[plots][coverage] tree1 is entirely outside the surface and carries no scores.
```

The warnings fire above 20% cropped, or above 10% missed among in-domain
stations — the latter pointing at breaks or gaps rather than at cropping.

**Radius.** At each station 16 rays leave the centreline in the plane normal to the
vessel axis and the first surface crossing is recorded; the station radius is the
**median** hit distance. A plain nearest-surface (inscribed) distance was tried
first and proved far too fragile — stray surface fragments and junction blends
swung it between 0.03× and 5.9× the true radius. The inscribed figures are still
written to `radius_agreement_inscribed.csv` for comparison.

**Metrics reported**

| Metric | Meaning |
|---|---|
| bias, LoA, RMSE, MAE, relative error | per Strahler order, plus Bland–Altman |
| `cl_dice` | topology-aware overlap (Shit et al., CVPR 2021) |
| `topology_sensitivity` | fraction of the **true** centreline inside the reconstruction — exact here, since the graph *is* the ground-truth skeleton |
| `topology_precision` | fraction of the reconstruction's skeleton inside the ground-truth tube volume |
| `dice`, `jaccard` | volumetric overlap, reported because clDice alone hides volume error |
| `beta0_surface_components` | connected components — the topological defect clDice is least sensitive to; extractor speckle shows here first |
| `graph_coverage_fraction` | how much of the graph the surface spans at all |
| `fraction_vessels_resolved` | share of vessels thicker than one voxel at the chosen spacing |

`--skip-voxel` runs the radius agreement alone (seconds rather than minutes);
`--voxel-mm` trades clDice resolution against memory — the grid is allocated over
the surface bounding box, so 0.15 mm on a whole tree is a few hundred MB.

### clDice is reported per tree

A coronary spatial graph normally holds the left and right trees as two disjoint
components, while a given STL usually represents only one of them. Scored
together, the overlap metrics charge the surface for a whole tree it never
claimed to reconstruct — on the `ratio_6` pair that dragged clDice down to 0.68
purely through absent coverage.

The graph is therefore split into connected components (largest first, so
`tree0` is the bigger tree) and every voxel metric is computed per component,
each inside its own bounding box:

| File | Contents |
|---|---|
| `cldice_per_tree.csv` | clDice, Tprec, Tsens, Dice, Jaccard and voxel counts per tree |
| `radius_agreement_per_tree.csv` | the radius agreement table, split the same way |
| `validation_summary.json` | the whole-graph score, kept only for reference |

Quote the per-tree row for the tree the surface actually represents. A tree the
surface never contained still gets a row, with blank scores — that is the correct
reading of "this surface does not contain that tree", not a reconstruction error,
and the figure script flags it by name on the console.

### Voxel Tsens under-reads; an exact version is reported alongside

`topology_sensitivity` is resolution-limited in a way `topology_precision` is not:
a vessel thinner than roughly two voxels rasterises away even where an exact
point-in-mesh test says the centreline is enclosed. On the `ratio_6` pair at
0.15 mm that is the difference between 0.72 (voxel) and 0.97 (exact) — an
artefact of the grid, not a defect in the surface.

Because the graph *is* the ground-truth skeleton, that term can be evaluated
exactly on the mesh, so each tree row carries both:

| Column | Meaning |
|---|---|
| `topology_sensitivity` | voxel estimate, comparable with published clDice values |
| `topology_sensitivity_exact` | exact fraction of in-domain stations enclosed by the surface |
| `cl_dice` | clDice from the voxel terms — use when comparing against the literature |
| `cl_dice_exact` | the same with the exact sensitivity term — the better estimate of this surface |

Report `cl_dice` for comparability and `cl_dice_exact` as the truer figure, or
drop the voxel spacing if the thinnest vessels matter.

The reconstruction is voxelised and skeletonised **once** on the full grid and
then sliced per component; skeletonising a cropped volume would introduce
spurious endpoints at the crop faces and inflate `Tprec`.

## Caveats worth stating in a paper

- **Different n per panel.** Anatomy comes from the full imaged tree, haemodynamics
  from the solved subset. `by_strahler.csv` and `by_strahler_cfd.csv` are kept
  separate rather than merged so the difference is explicit.
- **The root order is a single vessel.** Order 5 is one segment, so it has no
  spread; `fig4` draws such groups as points rather than a degenerate box.
- **Pressure is relative** to the outlets, which are pinned at 0 Pa, and left and
  right trees are separate simulations — compare within a tree, not across. See
  "Sign conventions" above; the reference is the boundary condition, not 1 atm.
- **Wall shear is a magnitude.** Only `|τ_w|` is exported, so every WSS statement
  is about magnitude and no directional or oscillatory measure is available.
- **`flow_ml_min` is not a conserved flow rate.** It is `<|v|> · A` from the speed
  magnitude, so it over-reads wherever flow is not axis-aligned and will not
  balance across a bifurcation. Use `flow_fractions.py` / CFX `massFlow()` for
  anything that has to conserve.
- **Voxel metrics are resolution-limited.** Vessels thinner than about two voxels
  cannot be represented; `fraction_vessels_resolved` states the coverage.
