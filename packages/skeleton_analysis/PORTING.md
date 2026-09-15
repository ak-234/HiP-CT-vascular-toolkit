# Porting `Skeleton_analysis` from MATLAB to Python

This document is the detailed reference for the MATLAB → Python port of the
`Skeleton_analysis` vascular spatial-graph package. It covers:

1. [Overview](#1-overview)
2. [Package architecture](#2-package-architecture)
3. [MATLAB → Python mapping](#3-matlab--python-mapping)
4. [Algorithm explanations](#4-algorithm-explanations)
5. [External dependencies](#5-external-dependencies)
6. [Bug fixes (detailed)](#6-bug-fixes-detailed)
7. [Design decisions / intentional deviations](#7-design-decisions--intentional-deviations)
8. [Validation & testing](#8-validation--testing)
9. [Known limitations](#9-known-limitations)

For a quickstart (install, CLI, examples) see [README.md](README.md).

---

## 1. Overview

The original is a MATLAB package (plus a few Python/ImageJ helpers) for analysing vascular
**spatial graphs** — vessel networks exported from Amira/Avizo as ASCII `.am`
(`HxSpatialGraph`) files, derived from HiP-CT kidney imaging. It is organised as five loosely
coupled, manually-run stages (I/O + ordering, metrics, outlier correction, optimisation metrics,
and utilities), with no top-level orchestrator, hard-coded absolute paths, and interactive dialogs.

**Scope of the port:** the *entire* package, as an installable Python package
(`skeleton_analysis`, `src/` layout) with a programmatic API, a CLI, `pytest` tests, and an
end-to-end driver ([Python_port_test.py](Python_port_test.py)).

**Approach:** a **clean, corrected** port, not a 1:1 translation. Interactive dialogs become
function arguments; MATLAB-toolbox features become open-source equivalents; and the documented
MATLAB bugs are **fixed** rather than replicated (see [§6](#6-bug-fixes-detailed)).

**Status:** 34 modules, **91 passing tests**. Validation is structural (no MATLAB numeric golden
reference was available) plus real-data runs on the LADAF-2024-28 kidney (see
[§8](#8-validation--testing)). The most decisive check: the re-derived Strahler order matches the
`strahler` field already embedded in a real file on **98.7 %** of edges.

---

## 2. Package architecture

```
src/skeleton_analysis/
  io/
    amira.py          SpatialGraph model + generic .am read/write
    amira_lattice.py  Amira BINARY lattice (image volume) reader (HxByteRLE)
    vesselvio.py      VesselVio vertices/edges CSV -> SpatialGraph/.am
  graph/
    build.py          SpatialGraph<->networkx, root detection, edge reorientation, rooted tree
    neighbors.py      children / parents / edge-index / coordination-number lookups
  ordering/
    strahler.py       Strahler order (per edge)
    topological.py    topological generation (per edge)
    pipeline.py       run_ordering, order_forest, auto_roots (forest-aware)
    root_picker.py    interactive per-tree root picker (PyVista; [viz3d]+[viz])
  metrics/
    regression.py     gmregress / gmregresspi (RMA / model-II regression)
    radius.py         mean radius per edge (from per-point thickness)
    geometry.py       length / chord / tortuosity / radius-stats / volume / surface-area
    branching_angles.py  branching angles (per edge + per vertex)
    murray.py         Murray's law + effective gamma
    intervessel.py    nearest centre-to-centre distance between segments
    exponent.py       radius-scaling exponent (log tip-count vs log radius, RMA)
    aggregate.py      per-Strahler aggregation, branching ratio, violin plots
    report.py         full metric table, k-means order, plots, Amira export, state comparison
  outlier/
    detect.py         percentile isoutlier/filloutliers, collapsed-vessel detection
    oblique.py        oblique cross-section resampling + radius, debug PNGs
    correct.py        thickness write-back, manual plane-selection snapping
    viz3d.py          optional interactive 3-D QC viewer (PyVista, [viz3d])
  optimisation/
    meta_metric.py    bifurcation points + bifurcation Dice + meta-metric formula
    volume_metrics.py centreline sensitivity, skeleton junctions, morphometrics, super_metric
    cl_dice.py        clDice family (cleaned VesselVio-era script)
  utils/
    merge.py          merge two spatial graphs (dedupe shared vertices)
    split.py          split a forest into per-tree SpatialGraphs
  cli.py              console entry points (info/roots/order/merge/vesselvio)
```

| Python module | Purpose | MATLAB origin |
|---|---|---|
| `io.amira` | `SpatialGraph` + generic `.am` read/write | `ultimate_amira_read.m`, `make_dict.m`, `run_ordering.m` (read_data/write_back), `write_corrected_data.m`, `write_amira_file.m`, `add_spatial_graphs.m` I/O |
| `io.amira_lattice` | binary lattice (image) reader | (new) — reads the Avizo segmentation `.am` |
| `io.vesselvio` | VesselVio CSV → `.am` | `VVToAmira_v3.m` |
| `graph.build` / `graph.neighbors` | digraph build, roots, reorientation, neighbours | `Find_bad_edges.m`, `find_children.m`, `find_parent_vec.m`, `return_edge_ind.m` |
| `ordering.*` | Strahler + topological ordering | `strahler_graph.m`, `return_Strahler.m`, `topological_gen.m`, `run_ordering.m` |
| `metrics.regression` | RMA regression | `gmregress.m`, `gmregresspi.m` |
| `metrics.geometry` | length/tortuosity/volume/SA/radius | `feature_extraction.py` (VesselVio) definitions |
| `metrics.branching_angles` | branching angles | `branching_angles_with_strahler.m`, `branching_ang.m` |
| `metrics.murray` | Murray's law + gamma | `murray_law.m`, `findEffectiveGamma.m` |
| `metrics.intervessel` | intervessel distance | `intervessel_distance.m` |
| `metrics.exponent` | scaling exponent | `Exponent_calculation.m` |
| `metrics.aggregate` / `metrics.report` | per-Strahler stats + plots + k-means + Amira export | `Graphs_strahler_against_metrics.m`, `al_goodplot.m`, external k-means |
| `outlier.detect` / `oblique` / `correct` | collapsed-vessel detection + oblique correction | `Outliers_spatial_graph.m`, `return_outlier.m`, `oblique_slice_vessel.m`, `give_oblique_slice_info.m`, `replace_thickness_vals.m`, `within_range.m` |
| `optimisation.meta_metric` / `volume_metrics` / `cl_dice` | bifurcation Dice, morphometrics, meta-metric, clDice | `meta_metric.m`, `super_metric_Euler_and_cc.ijm`, `cl_dice.py` |
| `utils.merge` / `utils.split` | merge / split graphs | `add_spatial_graphs.m`, cc1/cc4/cc9 splitting |

---

## 3. MATLAB → Python mapping

### 3.1 Data model

MATLAB stored the graph as `table`s whose columns are **cell-wrapped arrays** accessed like
`edge_network.EdgeConnectivity_EDGE{:}` — the product of `ultimate_amira_read.m` building
`cell2table` from regex-parsed field declarations.

The port replaces this with a single [`SpatialGraph`](src/skeleton_analysis/io/amira.py) dataclass:

* `vertex_fields`, `edge_fields`, `point_fields` — insertion-ordered dicts of numpy arrays
  (shape `(count,)` for scalars, `(count, dim)` for vectors such as `VertexCoordinates`);
* `field_order` — declaration order, so writes reproduce the layout;
* `raw_parameters` — the **verbatim** `Parameters { ... }` block, so writes round-trip cleanly back
  into Amira/Avizo (units, `TransformationMatrix`, colours, history). *The MATLAB writers discarded
  this, losing the spatial transform.*
* convenience properties (`vertex_coords`, `edge_connectivity`, `num_edge_points`, `point_coords`,
  `thickness`, `n_vertices/n_edges/n_points`).

The `.am` format handled by the reader/writer:

```
# Avizo 3D ASCII 3.0            (or "# AmiraMesh 3D ASCII 2.0")
define VERTEX <nV> / define EDGE <nE> / define POINT <nP>
Parameters { ... }              (captured verbatim, brace-matched, quotes respected)
VERTEX { float[3] VertexCoordinates } @1
EDGE   { int[2]  EdgeConnectivity  } @3
POINT  { float   thickness         } @7
# Data section follows
@1
<rows of numbers, one per VERTEX/EDGE/POINT>
```

Node IDs in `EdgeConnectivity` are **0-based** in the file and in the port; MATLAB added `+1` only
when building `digraph` objects.

### 3.2 Toolbox / function mapping

| MATLAB | Python |
|---|---|
| `digraph`/`graph`/`toposort`/`nearest`/`outdegree`/`indegree` | `networkx` (`DiGraph`, `topological_sort`, `descendants`, `degree`) |
| `knnsearch` (Stats & ML Toolbox) | `scipy.spatial.cKDTree` |
| `fsolve` (Optimization Toolbox) | `scipy.optimize.fsolve` |
| `tinv` / `finv` / `nanmean` / `nanstd` | `scipy.stats.t/f` / `numpy.nanmean` / `numpy` |
| `obliqueslice` (Image Processing) | `scipy.ndimage.map_coordinates` on a computed plane grid |
| `regionprops` / `imbinarize` / `bwskel` | `scikit-image` (`measure.regionprops`, threshold, `morphology.skeletonize`) |
| `tiffreadVolume` | `tifffile` |
| Amira **binary** lattice (`HxByteRLE`) | custom reader `io.amira_lattice` |
| `table` / `readtable` / `writetable` / `readcell` | `pandas` |
| `al_goodplot` (violin) | `seaborn` / `matplotlib` |
| MorphoLibJ macro (`.ijm`) | `skimage.measure` (`label`, `euler_number`, marching-cubes area) |
| external Python k-means (`kmeans_optimal.csv`) | `sklearn.cluster.KMeans` + silhouette selection |

### 3.3 Interactive → batch

The MATLAB pipeline blocks on human input: `inputdlg`/`menu` (root selection), `input`/`msgbox`
(oblique QC), `volshow`/`viewer3d` (3-D preview). The port removes all blocking dialogs:

* **Root selection**: pass `root_id` / `roots`; or auto-detect (`graph.build.resolve_root` /
  `ordering.auto_roots`); or use the built-in interactive **PyVista picker**
  `ordering.pick_roots` (the driver's GUI default) — one window per tree, click the inlet segment,
  and its degree-1 endpoint becomes that tree's root. See [fix #20](#6-bug-fixes-detailed) for why
  this replaced the external `coronary_sdf` picker the driver originally borrowed.
* **Oblique QC**: replaced by direct plane sampling; optional debug PNGs
  (`outlier.oblique.save_cross_section_png`) instead of `volshow`.

---

## 4. Algorithm explanations

Only the non-obvious re-derivations are described here; the rest are direct translations.

### 4.1 Generic Amira `.am` read/write (`io.amira`)
The reader discovers each field from its declaration line (`DOMAIN { dtype[dim] Name } @N`) rather
than assuming a fixed `@N` order — the same idea as `ultimate_amira_read.m`, but into numpy. The
`Parameters` block is captured by brace-matching that ignores braces inside double-quoted strings
(base64 blobs, quoted paths). Files are read/written as `latin-1` so the raw parameter bytes survive
losslessly. The writer emits header + preserved parameters + **renumbered** contiguous `@1..@k`
declarations + data — a complete writer that replaces the fragile MATLAB "append `@22`" flow
([bug #14](#6-bug-fixes-detailed)).

### 4.2 HxByteRLE lattice decode (`io.amira_lattice`)
The Avizo segmentation is a **binary** `define Lattice NX NY NZ` with `byte` data compressed as
`HxByteRLE`. Decode: control byte `c`; if `c & 0x80` copy the next `c & 0x7f` bytes verbatim
(literal), else repeat the next byte `c` times (run). Data is **X-fastest**, so the flat stream is
reshaped to `(NZ, NY, NX)` giving `vol[z, y, x]`. World↔voxel uses the header `BoundingBox`
(uniform spacing `(hi-lo)/(dim-1)`), not a hard-coded resolution.

### 4.3 Edge reorientation (`graph.build.reorient_edges`)
Avizo stores edges in mixed orientation. `Find_bad_edges.m` detected mis-oriented edges by an
iterative leaf-pruning while-loop. The port instead BFS-roots the (undirected) tree at `root` and
flips any edge whose `[source, target]` disagrees with the child→parent direction. Edge indices and
count are preserved (only orientation changes), so results stay aligned with `NumEdgePoints`/points.

### 4.4 Strahler order (`ordering.strahler`)
`strahler_graph.m` computed Strahler by repeatedly pruning coordination-1 leaves. The port roots the
tree and applies the **standard rule** in reverse-BFS order (children before parents): a leaf is 1;
an internal node is `max(child_orders) + 1` iff that maximum is shared by ≥2 children, else
`max(child_orders)`. Each edge takes the order of its child endpoint. This matches the MATLAB on all
bifurcations and additionally fixes the trifurcation tie ([bug #2](#6-bug-fixes-detailed)).

### 4.5 Topological generation (`ordering.topological`)
`topological_gen.m` walked a `toposort` order assigning generations along branches. Equivalently,
the generation of an edge is its BFS **edge-distance from the root** (root's edges = 1). The port
computes BFS distances once and assigns each edge the depth of its child endpoint.

### 4.6 Forest ordering (`ordering.pipeline`)
Real files are forests (e.g. 2 trees). `order_forest(edges, roots)` orders each connected component
with its own root and merges the per-edge results; `auto_roots` picks a root per component (the
leaf endpoint of the largest-radius edge); `utils.split.split_connected_components` extracts each
tree as a standalone `SpatialGraph` so per-tree metrics run with the existing single-root functions.

### 4.7 Oblique cross-section radius (`outlier.oblique`)
Replaces `obliqueslice`: build an orthonormal basis `(u, v)` perpendicular to the local centreline
tangent, sample a square grid `point + i·u + j·v` with `scipy.ndimage.map_coordinates`, binarise,
pick the region containing the centre, and estimate radius from its **perimeter** (`perimeter/(2π)`,
matching MATLAB) or **area** (`sqrt(area/π)`, more robust near voxel size). Optional debug PNGs
overlay the marching-squares boundary skimage's perimeter is based on.

### 4.8 K-means cluster order (`metrics.report.assign_kmeans`)
The MATLAB exported the feature matrix and read back an external `kmeans_optimal.csv`. The port
standardises the features, selects k∈2..8 by silhouette (reconstructing the "optimal" intent;
`--kmeans-k` overrides), then relabels clusters ordered by mean radius so the "cluster order"
increases with vessel size.

### 4.9 Metrics from geometry vs Amira fields (`metrics.geometry`, `metrics.report`)
Amira's stored `CurvedLength`/`Tortuosity`/`Volume`/`MeanRadius` are static and cannot be recomputed
after a correction. The report instead **computes** length (polyline arc-length), tortuosity
(length/chord), volume (`π r² L`), surface area (`2π r L`) and radius (mean of per-point thickness)
from the graph — the VesselVio `feature_extraction.py` definitions — so volume/SA/L-D/radius respond
to the corrections. Amira's fields are still read and reported as cross-checks.

### 4.10 Morphometrics + super-metric (`optimisation.volume_metrics`)
`region_morphometrics` is the Python replacement for the Fiji/MorphoLibJ macro: connected-components
count (`skimage.measure.label`, 26-conn), Euler number (`skimage.measure.euler_number`), volume
(foreground voxels × voxel³), surface area (marching-cubes mesh area ≈ the Crofton estimate).
`super_metric` assembles candidate/reference `Volume, CC, Euler, BB (branch count), CL (clDice)` and
combines them via the normalised-RMS `meta_metric`, completing what `meta_metric.m` left commented.

### 4.11 Bifurcation Dice + junctions (`optimisation.meta_metric`, `volume_metrics`)
`bifurcation_dice[_points]` greedily matches candidate branch points to reference ones within a
distance threshold (reproducing the `knnsearch` loop). `skeleton_junction_points` derives reference
bifurcations from a segmentation: skeletonise (cropped to the nonzero bbox), find 26-neighbourhood
junction voxels, **cluster** adjacent junction voxels into one centroid each, convert to world
coordinates.

---

## 5. External dependencies

### 5.1 Python packages

| Extra | Packages | Used for |
|---|---|---|
| core | `numpy`, `scipy`, `networkx`, `pandas` | arrays, cKDTree/optimize/ndimage/stats, graphs, tables |
| `[viz]` | `matplotlib`, `seaborn`, `scikit-learn` | plots, violins, k-means |
| `[viz3d]` | `pyvista` | optional interactive 3-D QC viewer (`outlier.show_segment_volume`) |
| `[image]` | `scikit-image`, `tifffile`, `SimpleITK` | oblique slicing, morphometrics, skeletonise, TIFF/volume I/O |
| `[dev]` | `pytest` | tests |

Core stays lightweight: `[image]`/`[viz]` are imported lazily so the ordering/metrics path never
requires them.

### 5.2 External tools / steps (stay external)

| External step | What it did | In the port |
|---|---|---|
| **VesselVio** (`feature_extraction.py` + its `library`, igraph/geomdl/numba) | skeletonise a segmentation, export vertex/edge CSVs | run externally; `io.vesselvio` reads the CSVs; metric *definitions* reproduced in `metrics.geometry` |
| **Amira/Avizo GUI** | multiscale smoothing, filament-editor simplification, spatial-graph stats export | manual GUI steps; the port reads/writes the `.am` directly |
| **Fiji + MorphoLibJ** (`super_metric_Euler_and_cc.ijm`) | CC, Euler, voxel count, Crofton surface area | replaced by `optimisation.region_morphometrics` |
| **external Python k-means** (`kmeans_optimal.csv`) | cluster segments by metric profile | replaced by `metrics.report.assign_kmeans` |
| **manual QC CSV** (`within_range.m`) | human selection of which oblique planes/radii to keep | `outlier.apply_manual_plane_selection` applies such a table |
| `addpath …\Image_processing_scripts_claire_under_development` | an external MATLAB helper directory | **not in the repo**, no observed call — treated as vestigial |

> The driver originally borrowed an interactive 3-D root picker from an external `coronary_sdf`
> package; it is now ported in-package as `ordering.pick_roots` (needs `[viz3d]`+`[viz]`). See
> [fix #20](#6-bug-fixes-detailed).

---

## 6. Bug fixes (detailed)

Each entry: **where** → **what was wrong & why** → **impact** → **fix**.

### MATLAB source bugs

**1. `run_ordering.m:5` — argument silently overridden.**
Line 5 hard-codes `filepath_ascii = "F:\...am"`, immediately overwriting the `filepath_ascii`
function argument. Impact: the function always processed one hard-coded file regardless of what was
passed. Fix: `ordering.pipeline.run_ordering(input_path, ...)` uses its argument.

**2. `return_Strahler.m:23-25` — trifurcation "two of three tie" understated.**
For a node with three children, only the *all-equal* branch added `+1`; the "not all equal" branch
returned `max` even when two of the three children shared the maximum (e.g. children `[2,2,1]`
returned 2 instead of 3). Impact: Strahler orders too low at such trifurcations, propagating up the
tree. Fix: `ordering/strahler.py` applies the standard rule uniformly — `max + 1` iff the maximum is
shared by ≥2 children — for any node degree. (Verified: on real data our order matches the file's
embedded `strahler` on 98.7 % of edges; the residual are exactly cases like this.)

**3. `oblique_slice_vessel.m:29` — `for i=1:1` debug limiter.**
The main correction loop was capped at the first outlier (a leftover debug limiter). Impact: only
one collapsed vessel was ever corrected. Fix: `outlier.oblique` / the driver iterate **all** flagged
segments.

**4. `oblique_slice_vessel.m:23` — hard-coded `res=50`.**
Voxel size fixed at 50 µm. Impact: wrong physical radii on any other dataset (the LADAF-2024-28
segmentation is ~66 µm). Fix: `res` is an argument; the driver derives it from the lattice's
`BoundingBox` spacing.

**5. `meta_metric.m:3` — `clear all` wipes the input.**
`function [results] = meta_metric(bb_coords)` immediately calls `clear all`, destroying the
`bb_coords` argument (then re-hard-codes it on line 34). Impact: the argument is meaningless. Fix:
the port takes real arguments (`optimisation.meta_metric`, `super_metric`).

**6. `meta_metric.m:37` — malformed `isInBox` parenthesis.**
`isInBox = @(M,B) (M(:,1)>=B(1)).*(M(:,1)<=B(2)).*(M(:,2)>=B(3)).*(M(:,2)<=B(4).*(M(:,3)>=B(5)).*(M(:,3)<=B(6)))`
mis-places a parenthesis so the y-max test `M(:,2)<=B(4)` is multiplied into the z tests instead of
standing alone. Impact: the bounding-box membership is wrong on the y/z axes. Fix:
`optimisation/meta_metric.py::bifurcation_points` tests all six bounds correctly.

**7. `meta_metric.m:74-81` — combined metric never computed.**
The combined `Metric` (normalised RMS of relative differences in Volume/CC/Euler/BB/CL) is
commented out, and the final `results = table(name_stat, volume, conncom, Euler, ..., Metric)`
references variables that are never assigned. Impact: the meta-metric did not exist in the MATLAB.
Fix: implemented end-to-end as `optimisation.super_metric` (+ `region_morphometrics` for
Volume/CC/Euler and `cl_dice`/`bifurcation_dice` for CL/BB).

**8. `within_range.m:3-4` — `data_path` vs `datapath`.**
Defines `data_path` but the next line reads `fullfile(datapath, ...)` (undefined). Impact: the
function errors immediately. Fix: `outlier.apply_manual_plane_selection(segment_radii, plane_table)`
takes the table as an argument (no path variable at all).

**9. `Exponent_calculation.m:61` — undefined variables.**
`log_exponent_all = vertcat(log_exponent_cc1, log_exponent_cc4, log_exponent_cc9)` references three
variables that are never defined (an unfinished multi-component concatenation). Impact: the function
errors. Fix: `metrics.exponent_calculation` accepts one *or more* graphs and pools their per-node
`(downstream-tips, radius)` data before the RMA regression.

**10. `murray_law.m` — filename vs function-name mismatch.**
The file `murray_law.m` declares `function [...] = murrays_law(...)`. MATLAB dispatches on the
*filename*, so callers use `murray_law` while the declared name is `murrays_law`. Impact: confusing;
fragile if the file is renamed. Fix: single canonical `metrics.murray.murray_law`.

**11. `murray_law.m:59` (Metrics variant) — wrong argument to `findEffectiveGamma`.**
`gamma_eff(i) = findEffectiveGamma(parent_rad, children)` passes the whole **accumulating**
`parent_rad` vector instead of the current scalar `parent_rad(i)`. Impact: the effective-gamma solve
uses the wrong (growing) parent radius. Fix: `metrics.murray` passes the scalar parent radius to
`find_effective_gamma`.

**12. `Find_bad_edges` — inconsistent signature.**
Called with 2 arguments in the `Metrics/` variants (`Find_bad_edges(edges, rootIDs)`) but 3 in
`Initial_Ordering/` (`..., update_root`). Impact: the same-named function behaves differently per
caller; the `update_root` flag is a workaround for the leaf-pruning approach. Fix: one BFS-based
`graph.build.reorient_edges(edges, root_id)` with a single behaviour.

**13. `ultimate_amira_read.m:93` — wrong empty check.**
`if isempty(strng)` guards a scalar-vs-vector decision but should be `isempty(strng{1})` (`strng`
is a non-empty cell whose first element may be empty). Impact: a field's component count could be
mis-inferred, mis-sizing the read. Fix: the generic reader `io.amira.read_amira` infers the
dimension from the parsed `[dim]` directly and validates the value count.

**14. Fragile write flow (`run_ordering.m`, `write_amira_file.m`).**
Writing `strahler`/`topo` required the user to *manually* insert header declaration lines in the
`.am`, then the code appended out-of-band data blocks (`@22`/`@23`) in `a+` append mode. Impact:
error-prone, order-dependent, breaks if the manual edit is wrong. Fix: `io.amira.write_amira` is a
complete writer — new fields are declared and written as normal, contiguously renumbered blocks.

### Port-implementation gotchas (found & handled while porting)

**15. HxByteRLE run/literal convention (inverted).**
The lattice decoder was first written with the run/literal bit inverted; the real 37.7 MB `Labels`
block failed to decode. A read-only dry-run over the block confirmed the Amira/`ahds` convention
(high-bit-set = literal, clear = run) decodes to exactly `NX·NY·NZ` bytes consuming exactly the
declared byte count. Fixed in `io.amira_lattice._decode_hxbyterle`; unit tests use the real
convention.

**16. X-fastest reshape + bbox mapping.**
Avizo lattice data is X-fastest, so the flat stream reshapes to `(nz, ny, nx)` (`vol[z,y,x]`);
world→voxel uses the header `BoundingBox` (uniform node-centred spacing), not a hard-coded resolution.
Getting this wrong would place oblique samples on the wrong voxels.

**17. `thickness` ≠ `MeanRadius` scale.**
In the LADAF-2024-28 file, per-point `thickness` and per-edge `MeanRadius` differ by a **non-constant
~2.46×** (median), because the thickness was smoothed/adjusted after `MeanRadius` was computed
(`smooth_thick_adj`). Consequences handled: (a) the metrics report uses a *consistent*
`radius = mean(thickness)` across all correction states so before/after reflects the corrections, not
the definitional gap (Amira's `MeanRadius` is reported as a cross-check column); (b) the Stage-5
oblique write-back re-derives `MeanRadius` **only on corrected edges**, so untouched edges keep
Amira's original value instead of being globally rescaled ~2.5×.

**18. Skeleton junction over-count.**
Naive 26-neighbourhood junction detection flags a *blob* of voxels at each crossing (diagonal
neighbours inflate the count), yielding several "bifurcations" per real one. Fix: cluster adjacent
junction voxels (26-connectivity) and take one centroid per cluster
(`volume_metrics.skeleton_junction_points`).

**19. clDice undefined on solid blobs.**
A solid (non-tubular) region has an empty medial skeleton, so `clDice` divides by zero → NaN.
`super_metric` skips the CL term when clDice is non-finite so the combined metric stays well-defined.

**20. Root picker showed "prune / removal (red)" wording.**
The driver borrowed `coronary_sdf.epicardial_annotation.run_prune_picker` for per-tree root
selection, but that picker is a *side-branch pruner*: its legend/instructions read "marked for
removal", "prune", and it highlights clicks **red**. Here the click is not a prune — the selected
segment's degree-1 endpoint is **assigned as the Strahler root** — so the wording was misleading and
implied destructive intent. Fix: the picker is ported in-package as `ordering.root_picker.pick_roots`
with **root-selection** semantics — title "ROOT SELECTION", "click the inlet (root) segment",
selection highlighted **green**, no removal language — and the external `coronary_sdf` dependency is
dropped entirely (the port reuses our own `SpatialGraph`, `graph.coordination_number`, and the
`auto_roots` root rule).

---

## 7. Design decisions / intentional deviations

These are deliberate changes, **not** bug fixes:

* **Interactive → batch** everywhere (see [§3.3](#33-interactive--batch)).
* **Report radius = mean of per-point thickness**, applied consistently across correction states, so
  before/after comparisons isolate the correction effect (see [gotcha #17](#6-bug-fixes-detailed)).
* **Oblique radius method**: `perimeter/(2π)` by default (matching MATLAB) with an `area = sqrt(area/π)`
  option — the perimeter estimate over-states the radius for cross-sections only 1-2 voxels across
  (staircase boundary), which the debug PNGs make visible.
* **Scoped `MeanRadius` re-derivation** on write-back (corrected edges only).
* **Crofton surface area** approximated by the marching-cubes mesh area (the standard scikit-image
  route).
* **`super_metric` needs a candidate + reference pair**, so it is a library + tested function rather
  than wired to a single-segmentation run.
* **Root picker internalised** (no `coronary_sdf`): the interactive per-tree picker is ported into
  the package as `ordering.pick_roots` — Frenet cross-section contour rings + the skeletonisation
  centreline points (Strahler-coloured) + legend, click the inlet segment, green highlight — instead
  of borrowing an external package's *prune* picker (see [fix #20](#6-bug-fixes-detailed)). The Frenet
  frame + contour-ring construction (`compute_frenet_frame`, `_segment_contour_mesh`) are ported from
  `coronary_sdf.splines` / `epicardial_annotation`; a lighter `style="tube"` is also available. It is
  opt-in (`[viz3d]`+`[viz]`, imported lazily) so headless `--auto-roots`/`--roots` and the core install
  are unaffected.

### 7.1 Why interactive 3-D QC is opt-in (not in the core path)

The MATLAB outlier stage had interactive 3-D QC: `getboundingbox.m` used
`volshow(subvol, OverlayData=label)` + `msgbox`/`input` to rotate a vessel sub-volume and type a
multiplier to grow/shrink the box (or `NaN` to skip the vessel); `vizualisation.m` used
`viewer3d`/`volshow` to show the sub-volume with the centreline points and the oblique cutting plane.
The core Python path deliberately does **not** run these, for five reasons:

1. **QC, not computation.** The viewers only let a human *look*; the radius is computed separately
   (`obliqueslice → regionprops → perimeter/(2π)`). Dropping them changes no output.
2. **Headless/batch by design.** `volshow`/`viewer3d`/`msgbox`/`input` block on a human and need a
   display — incompatible with running the pipeline programmatically, on a server, or in CI.
3. **The bounding-box step is unnecessary.** `getboundingbox` existed to extract/resize a sub-volume
   before slicing; the port samples the oblique plane **directly from the full volume**
   (`scipy.ndimage.map_coordinates`), so there is nothing to pre-extract.
4. **Heavy GUI dependency.** MATLAB's viewers map only to napari/PyVista/VTK — a large, display-bound
   stack that does not belong in an otherwise-light, headless package.
5. **Reproducibility.** "Rotate and type a multiplier / `NaN` to skip" are non-reproducible human
   decisions; the port replaces them with explicit parameters (`--max-oblique-segs`, detection
   thresholds) and `outlier.apply_manual_plane_selection` (a CSV), so a run is deterministic.

The genuinely-useful inspection — *seeing* the cross-sections that drive the radius — is preserved
**headlessly** via `outlier.oblique.save_cross_section_png` (`--oblique-debug`), which writes PNGs of
the binarised cross-section + its boundary, no display required.

**Now available opt-in.** For users who do want the 3-D view, it is provided behind the **`[viz3d]`**
extra (PyVista): `outlier.show_segment_volume(volume, centreline, show_planes=...)` renders the
vessel sub-volume + centreline (+ optional cutting planes), and the driver exposes `--oblique-qc`
(`--oblique-qc-max`, `--oblique-qc-planes`). It is **inspection only** — skip/resize decisions stay
reproducible via the parameters and the manual-correction CSV above. `off_screen=True` renders to a
PNG for headless capture/tests.

---

## 8. Validation & testing

No MATLAB numeric golden reference was available, so validation is **structural** plus real-data
sanity:

* **Amira round-trip**: read → write → read yields identical arrays; the `Parameters` block and
  field order are preserved; counts stay consistent (`Σ NumEdgePoints == nPoints`).
* **Hand-computed oracles**: Strahler / topological generation on synthetic Y-, balanced- and
  trifurcation trees (including the two-of-three tie that exercises [bug #2](#6-bug-fixes-detailed)).
* **Embedded-order match**: on a real file that already carried `strahler`, the re-derived order
  matches on **98.7 %** of 309 edges (auto-picked roots).
* **RMA regression** checked against the published Sokal & Rohlf worked example
  (`b = [12.1938, 2.1194]`, and the documented CIs / prediction interval).
* **Oblique**: radius ≈ 5 recovered from a synthetic radius-5 cylinder; area vs perimeter behaviour.
* **Morphometrics**: two disjoint cubes → components 2, Euler 2, exact volume; `super_metric`
  identical pair → combined 0, perturbed → positive.
* **HxByteRLE**: synthetic encode/decode round-trip + the real-block dry-run.
* **Real data**: the LADAF-2024-28 kidney runs end-to-end — 2-tree forest, Strahler 1-4, ~66 µm
  voxels; segmentation morphometrics 55 components / Euler 33 / 1.32×10¹² µm³.

Run the suite:

```bash
pip install -e ".[image,viz,dev]"
pytest                 # 91 passed
pytest -m "not image"  # skip tests needing the [image] extra
```

---

## 9. Known limitations

* **Coarse-voxel oblique radius** over-estimates for vessels ~1-2 voxels in radius (perimeter
  discretisation); use a finer segmentation or `--oblique-radius-method area`.
* **`super_metric`** requires a candidate + reference pair (two volumes/graphs) — not derivable from a
  single segmentation, so it is a library/test deliverable.
* **Memory / time**: decoding the full 2.3 GB `Labels` volume takes ~10 s; `region_morphometrics`
  (marching cubes on that volume) ~90 s; both are opt-in (`--image` / `--optimisation`).
* **Interactive 3-D QC** is not in the core/headless path (see [§7.1](#71-why-interactive-3-d-qc-is-opt-in-not-in-the-core-path)); it is available opt-in via the `[viz3d]` extra.
* The vestigial external MATLAB `addpath` directory is absent; no functionality depends on it in the
  scripts provided.
