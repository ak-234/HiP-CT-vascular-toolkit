# Skeleton_analysis

Skeletonisation fixes and topological analysis of vascular **spatial graphs**
(Amira/Avizo `HxSpatialGraph` `.am` files), derived from HiP-CT image data.

This package is the **Python port** — a clean, batch-friendly reimplementation
of an earlier MATLAB pipeline, installable as the `skeleton_analysis` package
(in `src/`). The port's MATLAB origins (module-by-module mapping, algorithm
explanations, and every bug fix) are documented in [PORTING.md](PORTING.md), which
references the original MATLAB scripts by name throughout.

It is one of three packages in the
[HiP-CT Vascular Toolkit](https://github.com/ak-234/HiP-CT-vascular-toolkit).

### Attribution

The original MATLAB skeletonisation and topological-analysis pipeline was written
by the **UCL HiP-CT group (Claire Walsh and colleagues)**. This is an independent
Python reimplementation of those algorithms; the MATLAB sources are not
redistributed here. The MIT licence on this package covers the port only — if you
intend to redistribute or build on the original MATLAB work, seek permission from
its authors.

The original MATLAB write-up, with images (access-controlled — not public):
<https://docs.google.com/document/d/1eW96BHCpdBfnYGSXVgjeTqCNGPLj0TTpnx4Xdsut-H8/edit?usp=sharing>

---

## Python package

A clean, batch-friendly port of the MATLAB code. Interactive dialogs are
replaced by function arguments, MATLAB toolbox features by open-source
equivalents (`networkx`, `scipy`, `scikit-image`, `pandas`), and several
MATLAB bugs are fixed (see [Fixes](#fixes-applied-during-the-port)).

> **See [PORTING.md](PORTING.md)** for the detailed MATLAB→Python porting guide:
> the full module map, algorithm explanations, external dependencies, and a
> detailed explanation of every bug fix.

### Install

```bash
pip install -e .                     # core: numpy, scipy, networkx, pandas
pip install -e ".[image]"            # + scikit-image, tifffile, SimpleITK (outlier / clDice)
pip install -e ".[viz]"              # + matplotlib, seaborn, scikit-learn (plots + k-means)
pip install -e ".[viz3d]"            # + pyvista (optional interactive 3-D QC viewer)
pip install -e ".[image,viz,dev]"    # everything + pytest
```

### Quick start

```python
from skeleton_analysis.ordering import run_ordering

# Strahler + topological ordering, written back into a new .am file.
result = run_ordering("input.Spatial-Graph.am", "ordered.am")   # root auto-detected
print(result.root_id, result.strahler.max(), result.topo.max())
```

```python
from skeleton_analysis.io import read_amira, write_amira
from skeleton_analysis.metrics import (
    branching_angles, murray_law, intervessel_distance,
    exponent_calculation, mean_radius_per_edge,
)

g = read_amira("ordered.am")
BA_edge, BA_vertex = branching_angles(g, root_id=21)
ivd = intervessel_distance(g)

g.set_edge_field("MeanRadius", mean_radius_per_edge(g))   # if not already present
murray = murray_law(g, root_id=21)                        # DataFrame per branch point
scaling = exponent_calculation(g, radius_field="MeanRadius")
print("radius-scaling exponent:", scaling.exponent)
```

### Command line

```bash
skeleton-analysis info      input.am
skeleton-analysis roots     input.am
skeleton-analysis order     input.am ordered.am [--root 21]
skeleton-analysis merge     big.am small.am merged.am
skeleton-analysis vesselvio vertices.csv edges.csv out.am --res-xy 50 --res-z 50
```

### Package layout

| Module | Purpose | MATLAB origin |
|---|---|---|
| `skeleton_analysis.io.amira` | `SpatialGraph` model + generic `.am` read/write | `ultimate_amira_read.m`, `make_dict.m`, all bespoke readers/writers |
| `skeleton_analysis.io.amira_lattice` | Amira **binary lattice** (image volume) reader (HxByteRLE) | (new — reads Avizo segmentation `.am`) |
| `skeleton_analysis.io.vesselvio` | VesselVio CSV -> `.am` | `VVToAmira_v3.m` |
| `skeleton_analysis.graph` | digraph build, root detection, edge reorientation, neighbours | `Find_bad_edges.m`, `find_children.m`, `find_parent_vec.m`, `return_edge_ind.m` |
| `skeleton_analysis.ordering` | Strahler order, topological generations, `run_ordering`, auto/interactive per-tree root selection (`auto_roots`, `pick_roots`) | `strahler_graph.m`, `return_Strahler.m`, `topological_gen.m`, `run_ordering.m` |
| `skeleton_analysis.metrics` | branching angles, Murray's law, intervessel distance, RMA regression, scaling exponent, aggregation | `branching_angles_with_strahler.m`, `murray_law.m`, `findEffectiveGamma.m`, `intervessel_distance.m`, `gmregress(pi).m`, `Exponent_calculation.m`, `Graphs_strahler_against_metrics.m` |
| `skeleton_analysis.outlier` | collapsed-vessel detection, oblique cross-section radius correction | `Outliers_spatial_graph.m`, `return_outlier.m`, `oblique_slice_vessel.m` + helpers |
| `skeleton_analysis.optimisation` | bifurcation Dice, clDice, whole-volume morphometrics (CC/Euler/volume/surface), combined `super_metric` | `meta_metric.m`, `cl_dice.py`, `super_metric_Euler_and_cc.ijm` |
| `skeleton_analysis.utils.merge` | merge two spatial graphs | `add_spatial_graphs.m` |
| `skeleton_analysis.utils.split` | split a forest into per-tree graphs | (new — mirrors the cc1/cc4/cc9 split) |

### Full-pipeline test on real data

`Python_port_test.py` runs the whole pipeline on a real LADAF-2024-28 graph
(order → metrics → outlier detection → optional image radius correction). It roots
each tree with the built-in per-tree picker (`ordering.pick_roots`; needs the
`[viz3d]`+`[viz]` extras) — click the inlet segment of each tree — or
`--auto-roots` / `--roots` to skip the GUI, and **validates the port by comparing
the re-derived Strahler order against the `strahler` field already embedded in the
file** (98.7% match on the LADAF-2024-28 skeleton). Forest ordering
(`ordering.order_forest`, `ordering.auto_roots`) handles multi-tree graphs with a
root per component.

```bash
python Python_port_test.py --auto-roots            # stages 0-4, no GUI
python Python_port_test.py                          # opens the per-tree root picker
python Python_port_test.py --auto-roots --image     # + oblique correction (decodes ~2.3 GB)
python Python_port_test.py --auto-roots --optimisation  # + clDice sensitivity & bifurcation Dice
```

Stage 6 (`--optimisation`) compares the skeleton to the segmentation *volume*: **clDice centreline
sensitivity** (fraction of the skeleton inside the segmentation, by point-sampling) and
**bifurcation Dice** (our graph's branch points matched against junctions of the segmentation's own
3-D skeleton). On LADAF-2024-28: sensitivity ≈ 0.9996 (the skeleton lies inside the segmentation, as
expected).

### The Amira `.am` format (as handled here)

The reader is generic: every field is discovered from its declaration line
(`DOMAIN { dtype[dim] Name } @N`), so `@N` numbering is never assumed. The full
`Parameters { ... }` block (units, `TransformationMatrix`, colours, history) is
captured verbatim and re-emitted, so written files round-trip cleanly back into
Amira/Avizo — unlike the MATLAB writers, which discarded it. Node IDs in
`EdgeConnectivity` are 0-based.

### MATLAB toolbox -> Python mapping

`digraph`/`toposort`/`nearest` -> `networkx` &middot; `knnsearch` ->
`scipy.spatial.cKDTree` &middot; `fsolve` -> `scipy.optimize` &middot;
`tinv`/`finv`/`nanmean` -> `scipy.stats`/`numpy` &middot; `obliqueslice` ->
`scipy.ndimage.map_coordinates` &middot; `regionprops`/`imbinarize` ->
`scikit-image` &middot; `tiffreadVolume` -> `tifffile` &middot;
`table`/`readtable` -> `pandas` &middot; `al_goodplot` -> `seaborn`.

### External pipeline steps (not in this package)

The original workflow used several external tools/scripts. Here is how each maps to the Python port:

| External step | What it did | In the port |
|---|---|---|
| **VesselVio** (`Other_useful_scripts/feature_extraction.py` + the VesselVio `library`) | skeletonise a segmentation and export per-segment vertex/edge CSVs | run VesselVio externally; `skeleton_analysis.io.vesselvio` reads its CSVs into a `.am`. Metric *definitions* (length/tortuosity/volume/SA) are reproduced in `metrics.geometry`. |
| **Amira/Avizo GUI** | multiscale smoothing, filament-editor simplification, spatial-graph statistics export to CSV | manual GUI steps; the port reads/writes the resulting `.am` directly (no CSV export needed). |
| **Fiji/ImageJ + MorphoLibJ** (`super_metric_Euler_and_cc.ijm`) | connected components, Euler number, voxel count, Crofton surface area of a binary mask | replaced by `optimisation.region_morphometrics` (scikit-image; surface area via marching cubes). |
| **External Python k-means** (`kmeans_optimal.csv`) | cluster segments by their metric profile | replaced by `metrics.report.assign_kmeans` (silhouette-selected k). |
| **Manual QC CSV** (`within_range.m`, `Skeleton_manual_correction_from_genx_outliers.csv`) | human selection of which oblique planes/radii to keep | `outlier.apply_manual_plane_selection` applies such a table (you still create it by hand). |
| `addpath …\Image_processing_scripts_claire_under_development` (Metrics) | an external MATLAB helper directory | **not in the repo**; no observed call in the scripts — treated as vestigial. |

`super_metric` (Volume + CC + Euler + bifurcation-Dice + clDice) completes the combined "meta metric"
that `meta_metric.m` left commented out; it compares a *candidate* skeletonisation volume/graph to a
*reference* one.

### Fixes applied during the port

* `run_ordering.m` honoured a hard-coded path over its argument — the port uses the argument.
* Strahler tri-furcation rule corrected for the "two of three children tie" case
  (`return_Strahler.m` returned `max` instead of `max + 1`).
* `oblique_slice_vessel.m` `for i=1:1` debug limiter removed; `res` and window size are arguments.
* `meta_metric.m` `clear all` (which wiped the input) removed; the `isInBox`
  bounding-box parenthesis fixed; the combined metric fully defined.
* `within_range.m` `data_path`/`datapath` typo, `Exponent_calculation.m` undefined
  `log_exponent_cc*`, and the `murray_law`/`murrays_law` name mismatch all resolved.
* Fragile "hand-edit the header then append `@22`/`@23`" write flow replaced by a
  complete writer.

### Tests

```bash
pytest                 # runs all tests (image tests auto-run if [image] is installed)
pytest -m "not image"  # skip tests that need the [image] extra
```

Validation is structural (no MATLAB golden reference): `.am` round-trip
equality, hand-computed Strahler/topology oracles on synthetic trees,
RMA regression checked against the published Sokal & Rohlf example, and radius
recovery from a synthetic cylinder.
