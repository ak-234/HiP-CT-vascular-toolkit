# `coronary_sdf`

The production defaults remain the compatibility path:
`CENTERLINE_SMOOTHER="savgol"`, `SDF_FIELD_METHOD="legacy"`, and
`SDF_MESH_METHOD="meshlib"`. The constrained multiscale smoother, graph
implicit fields, exact round-cone primitive, and adaptive extractors are
experimental until they pass the benchmark's CFD qualification gates.
`python -m coronary_sdf` therefore still runs the compatibility profile.
`--profile cfd` is deliberately unavailable unless it is given a qualification
file emitted after every synthetic and both LADAF gates pass.

**Reconstruct a watertight 3-D coronary-artery lumen surface from an Amira/Avizo
centerline graph, then drive CFD boundary conditions and wall-shear-stress analysis off it.**

Given an Amira **SpatialGraph** (a skeleton of vessel centerlines with a radius at every point),
`coronary_sdf` builds a smooth, closed triangle mesh of the vessel *lumen* — suitable for meshing
and computational fluid dynamics (CFD) — using a **smooth-minimum capsule signed-distance field
(SDF)**. On top of the surface pipeline it ships a set of downstream tools for outlet flow-split
boundary conditions, resistance-based pressure outlets, wall-shear-stress (WSS) post-processing,
and interactive vessel annotation / pruning.

> **New here?** Read [How it works](#how-it-works-the-logic-behind-it) for the core idea, then open
> the visual explainer at [docs/dashboard.html](docs/dashboard.html) (double-click — it runs
> offline in any browser). The architecture reference with Mermaid diagrams lives in
> [docs/uml_diagrams.md](docs/uml_diagrams.md).

---

## Table of contents

- [What it is](#what-it-is)
- [Requirements](#requirements)
- [Installation](#installation)
- [Quick start](#quick-start)
- [How it works (the logic behind it)](#how-it-works-the-logic-behind-it)
- [Pipeline stages](#pipeline-stages)
- [Configuration](#configuration)
- [Outputs](#outputs)
- [Downstream tools](#downstream-tools)
- [Package layout](#package-layout)
- [Further reading](#further-reading)

---

## What it is

A coronary tree, after segmentation and skeletonisation in Avizo/Amira, is stored as a
**SpatialGraph**: a set of *nodes* (branch points and endpoints), *points* (densely sampled
centerline coordinates, each carrying a local vessel *thickness* = radius), and *segments* (the
edges connecting nodes, each an ordered list of points, optionally tagged with a Strahler order).

`coronary_sdf` turns that graph into a **single closed surface mesh of the lumen wall**. The
scale-equivariant backend represents branch interiors by tapered analytic primitives, evaluates
their implicit field on demand through a correctness-bounded BVH, and blends only graph-incident
branches inside compact junction neighbourhoods. Bifurcation endpoints are one-sided plane-clipped
before blending, so a large parent endpoint sphere cannot occupy the daughter carina. The zero set
can then be sampled adaptively without allocating a dense whole-tree voxel grid.

The package is a focused refactor of a legacy monolith (`Coronary_lumen_octree.py`): it keeps only
the SDF surface method and drops the older HRBF / loft / hybrid / octree / dual-contouring
generators. It is **function-oriented** — behaviour lives in free functions across single-purpose
modules, with a handful of plain-data dataclasses; there is no class hierarchy.

---

## Requirements

- **Python 3.11+**
- **Required:** [`numpy`](https://numpy.org), [`scipy`](https://scipy.org),
  [`pyvista`](https://pyvista.org) (pulls in VTK).
- **Iso-surface backends** (choose via `SDF_MESH_METHOD`):
  - `vtk_htg` — diagnostic compiler-free adaptive VTK HyperTreeGrid backend.
    It has good analytic-fixture accuracy, but real LADAF output produced tiny
    contour speckles and incorrect genus, so it is not a production candidate.
    A controlled ablation on a real region has since attributed those speckles
    to the extractor rather than the field: with the dense extractor matched,
    the legacy and graph fields are indistinguishable, while swapping dense for
    adaptive on a fixed field turns one clean component into twenty.
  - `cgal_mesh3` — optional compiled CGAL 6.2 implementation retained for
    research comparison. See [native/cgal/README.md](native/cgal/README.md).
  - `adaptive` — explicitly experimental SciPy
    Delaunay/marching-tetrahedra diagnostic extractor; it is excluded from
    production candidate selection.
  - [`meshlib`](https://pypi.org/project/meshlib/) — marching cubes (default) and the
    `tube_union` boolean mesher.
  - [`open3d`](http://www.open3d.org) — Screened Poisson reconstruction.
  - VTK FlyingEdges marching cubes — no extra dependency (bundled with `pyvista`).
- **Mesh repair:** [`pymeshfix`](https://pypi.org/project/pymeshfix/) (optional; gated by
  `MESH_REPAIR` / `USE_PYMESHFIX`).
- **Topology / region tagging:** [`networkx`](https://networkx.org).
- **Plots & interactive pickers (optional):** [`matplotlib`](https://matplotlib.org) — only needed
  by the WSS plotting and the annotation/prune pickers.

The package is run as a **module from the directory that contains the `coronary_sdf/` folder**
(i.e. `coronary_sdf`'s parent), so that `python -m coronary_sdf …` resolves the package.

---

## Installation

From the project directory, install the runnable compatibility and compiler-free
VTK pipelines with:

```bash
cd packages/coronary_sdf
python -m pip install .
```

This also installs the `coronary-sdf` and `coronary-sdf-benchmark` commands.
Large optional backends, repair, plotting, acceleration, benchmarking, and test
dependencies can be installed together with:

```bash
python -m pip install ".[all]"
```

For an editable development installation with the focused test dependencies:

```bash
python -m pip install -e ".[test]"
```

Individual extras are `benchmark`, `repair`, `poisson`, `plot`, `speed`, and
`test`. The optional native CGAL extension is not installed by pip; see
[native/cgal/README.md](native/cgal/README.md) for its separate toolchain.

---

## Quick start

Generate the lumen surface from a SpatialGraph:

```bash
# from the parent directory of coronary_sdf/
python -m coronary_sdf  path/to/tree.am.xml  path/to/output_dir

# equivalent installed console command
coronary-sdf path/to/tree.am.xml path/to/output_dir
```

That command is equivalent to `--profile compat`. A qualified adaptive profile
is invoked only with the benchmark token that certified it:

```bash
python -m coronary_sdf input.am output_dir --profile cfd \
  --qualified-config benchmark_out/cfd_qualified_profile.json
```

If the qualification file is absent or does not certify the VTK HyperTreeGrid
candidate on both LADAF cases, the CLI exits before reconstruction.

The same graph-implicit, round-cone, VTK HyperTreeGrid candidate can be run for
development and visual assessment without claiming CFD qualification:

```bash
python -m coronary_sdf input.am output_dir --profile experimental --non-interactive
```

For very large multiscale graphs, start with six cells across the smallest
diameter and retain the five-million-cell safety limit:

```bash
python -m coronary_sdf input.am output_dir --profile experimental \
  --cells-across-diameter 6 --max-cells 5000000 --non-interactive
```

The experimental profile rejects a mesh that fails topology validation before
writing it. To retain such a mesh strictly for visual diagnosis, add
`--validation-mode warn`; the output is then explicitly unqualified and must
not be used for CFD. The qualified `cfd` profile cannot weaken validation.

A multi-component graph aborts on the first component that fails, which for a
real network usually means the largest component is written and the rest are
silently absent. Add `--component-failure continue` to reconstruct every
remaining component and record the failure instead:

```bash
python -m coronary_sdf input.am output_dir --profile experimental \
  --non-interactive --component-failure continue --validation-mode warn
```

`component_manifest.json` is written either way, so completeness is always
checkable. The `cfd` profile cannot weaken this either — a partial network is
not CFD-qualified.

- The input may be a **native Amira ASCII `.am`** or an **Excel-XML SpatialGraph `.xml`** export;
  the format is auto-detected by `parse_amira.parse_xml`.
- With **no arguments**, the CLI falls back to `config.INPUT_PATH` and `config.OUTPUT_DIR`.
- Each **connected component** of the graph is meshed independently and written with a `_g{id}`
  suffix (the first component has no suffix).

After it runs, the output directory contains:

```
output_dir/
  lumen_bspline.stl            # watertight lumen surface (STL)
  lumen_bspline.vtk            # same surface (VTK PolyData)
  lumen_bspline_regions.vtk    # per-face region/Strahler/junction tags for colouring
  lumen_bspline_g1.stl ...     # additional connected components, if any
  component_manifest.json      # requested / completed / failed components
  INCOMPLETE                   # present only when a component did not finish
```

> **Reading a partial run.** Component 0 is written without a `_g` suffix, so a
> run that aborted on component 1 leaves exactly the files a single-component
> graph would leave. `component_manifest.json` is written on every exit path —
> including the fail-fast one — and records which components were *requested*
> against which completed, with the exception type for each failure. If any are
> missing, an `INCOMPLETE` marker names them in plain text. Never infer
> completeness from the file listing alone.

---

## How it works (the logic behind it)

The whole pipeline exists to answer one question at every point in space:
**"how far am I from the nearest vessel wall, and am I inside or outside?"** That scalar field is
the signed-distance field (SDF); the lumen surface is simply where it equals zero.

### 1. A vessel is a union of tapered capsules
Each vessel segment is a smoothed centerline with a radius at every point. Split it into short
spans and model each span as a **capsule**: a line segment `(p0, p1)` with radii `(r0, r1)` — a
cone with hemispherical end-caps. The distance from a point to that capsule is cheap and exact. The
lumen is the **union** of all capsules in the tree. *(See `capsules.build_capsules`,
`sdf_field.evaluate_sdf`.)*

### 2. Graph-local smooth union instead of a global smooth minimum
The distance to a union of shapes is the **minimum** of the individual distances. A plain `min`
produces a sharp crease where vessels meet, while applying smooth-min globally can bridge unrelated
vessels. The graph implicit field therefore uses hard union globally and evaluates a smooth union
(log-sum-exp) only between branches incident to the same graph node and only inside a compact,
radius-scaled junction support:

```
smin(d₁…dₙ) = −ln( Σ exp(−k·dᵢ) ) / k
```

The graph backend parameterizes blend depth and support entirely as fractions of the local junction
radius. Scaling a 30-micron branch or a 2-mm branch to unit radius therefore produces the same
normalized field. The legacy dense evaluator retains its historical radius-adaptive smooth-min.

### 3. Preventing bridges between non-adjacent branches
Two vessels can be close in space yet far apart in the tree. The graph field never smooths such a
pair, so initially positive surface clearance remains positive. The input is audited for capsule
overlap before meshing. If two perfect-circle tubes already overlap, no scalar union can keep both
radii unchanged and also produce separate lumens; the conflict must be resolved from segmentation
contours, constrained centreline geometry, or an explicitly accepted local geometry change. The
legacy dense path retains its topology-aware carve for compatibility.

### 4. Clean carinas: flat caps and carina taper
At a bifurcation, the incident capsule end-caps are hemispheres whose union balloons outward. Two
complementary fixes keep the carina (the ridge where daughters split) sharp and realistic:
- **One-sided branch clipping** (`SDF_FLAT_CAP_BIF`): intersect every bif-incident analytic
  primitive with its tangent half-space before hard union or junction blending. The same
  radius-normalized zero set is used by scalar, accelerated batch, adaptive, VTK HTG, and native
  CGAL field consumers.
- **Carina taper** (`bif_trim.taper_bifurcation_carina`, optional): narrow the last few capsule
  radii into a cone, so *N* cones meeting at a node smooth-min into a clean Y/T/X.
Terminal (leaf) endpoints get a matching **flat cap** so outlets are planar and CFD-ready.

### 5. On-demand adaptive evaluation
The recommended backend does not allocate a dense SDF volume. A BVH supplies a proven lower bound
for each primitive group, and an adaptive octree, VTK HyperTreeGrid, or CGAL implicit mesher queries
the field only where a cell can contain the zero set. Target cell size is a fraction of the local
radius, so the cost follows vessel surface complexity rather than the cube of the smallest radius.
The dense narrow-band evaluator remains available as the compatibility path.

### 6. Why constrained centreline regularization happens up front
Raw skeleton centerlines are **noisy and unevenly sampled**. Left alone they cause two failure
modes the surface can't recover from: on a tight bend the swept tube **self-intersects** and the
same-segment union deletes the inner wall; at a junction, stair-step **radius jumps** produce necks
and dishes. The constrained multiscale smoother works in radius-normalized arc length, fixes graph
nodes, limits displacement to a fraction of local radius, and checks curvature reach plus nonlocal
and non-adjacent clearance. Input radii remain immutable; infeasible overlaps are reported rather
than hidden by unconstrained smoothing. Legacy Savitzky–Golay and B-spline modes remain available.

### 7. From field to mesh
The compatibility profile extracts the dense field with MeshLib marching cubes. The graph backend
supports adaptive octree, VTK HyperTreeGrid, and native CGAL implicit extraction from the shared
oracle. Every production candidate must pass finite-coordinate, watertightness, manifoldness,
self-intersection, component-count, and contour-fidelity checks before CFD qualification.

---

### Scale-equivariant graph backends

Set `SDF_MESH_METHOD = "vtk_htg"` to bypass the dense volume completely. The
backend uses `implicit_field.GraphImplicitField`, which evaluates a hard union
with a correctness-bounded capsule BVH and blends only graph-incident segment
fields near a junction. Its N-ary log-sum-exp has a maximum depression of
`IMPLICIT_JUNCTION_BLEND_FRACTION * local_radius`, independent of junction
degree and capsule sampling density.

The hierarchy targets
`cell_size = 2 * local_radius / IMPLICIT_CELLS_ACROSS_DIAMETER`; consequently a
uniform rescaling of coordinates and radii produces the same hierarchy and
mesh topology. `PRESERVE_INPUT_RADII = True` disables every preprocessing pass
that rewrites measured radii. Fixed-radius overlaps outside graph junctions are
reported before field evaluation because no scalar union can keep genuinely
overlapping volumes separate.

The included SciPy and VTK HyperTreeGrid adaptive extractors are retained only
as diagnostic parity backends. HTG builds a radius-adaptive octree in Python
and contours it with the prebuilt VTK wheel, but compiler-free deployment does
not outweigh its failed real-network topology. The optional CGAL extension
remains available for research comparison. No adaptive backend is currently
selected for production.

## Pipeline stages

Run order, as orchestrated by `pipeline.run_pipeline` (graph-wide preprocessing) and
`pipeline.generate_sdf_surface` (per connected component):

| # | Stage | Module · key function | Purpose |
|---|-------|-----------------------|---------|
| 1 | Parse | `parse_amira.parse_xml` | Read SpatialGraph → `nodes`, `points`, `segments` (µm). |
| 2 | Validate | `parse_amira.find_degenerate_segments` | Flag collapsed / large-gap centerlines before the expensive work. |
| 3 | Topology filter | `centreline_reconnection.merge_degree2_segments`, `merge_split_multifurcations` | Strahler filter, contract pass-through nodes, collapse split multifurcations. |
| 4 | Gap bridge / densify | `centreline_reconnection.bridge_centerline_gaps`, `smoothing.densify_sparse_segments` | Fill point-less spans and up-sample sparse segments. |
| 5 | Split by component | `centreline_reconnection.split_by_graph` | One independent sub-problem per connected tree. |
| 6 | Centerline smoothing | `smoothing.smooth_segment_centerlines`, `limit_centerline_curvature` | Radius-adaptive B-spline smoothing; straighten self-intersecting bends. |
| 7 | Radius smoothing | `smoothing.smooth_segment_radii`, `smooth_radius_transitions`, `prune_*_shrink` | Denoise r(s); blend junction transitions; clamp shrink outliers. |
| 8 | Topology & adjacency | `sdf_field.build_adjacency`, `find_bifurcations`, `topology.build_directed_topology` | Segment adjacency, bifurcation set, parent/child/sibling masks. |
| 9 | Splines | `splines.prepare_segment_spline`, `branch_tangent_at_node` | Per-segment cubic spline + branch tangents. |
| 10 | Carina taper (opt.) | `bif_trim.taper_bifurcation_carina` | Cone-taper bif-incident capsule ends. |
| 11 | Capsules | `capsules.build_capsules`, `clamp_terminal_capsule_radii` | Sample splines into tapered capsules + KD-tree. |
| 12 | Grid & narrow band | `sdf_field.compute_grid`, `build_narrow_band` | Auto voxel size, bounding grid, thin evaluation shell. |
| 13 | **SDF evaluation** | `sdf_field.evaluate_sdf` | Smooth-min field + anti-bridge carve + flat caps. |
| 14 | Iso-surface | `mesh_extract.extract_isosurface`, `vtk_htg_mesher.mesh_vtk_hyper_tree_grid` | Dense marching cubes / Poisson or adaptive HyperTreeGrid contouring at SDF = 0. |
| 15 | Mesh post | `mesh_extract.cut_non_adjacent_bridges`, `mesh_repair.radius_constrained_taubin`, `repair_mesh` | Bridge cut, clean, Taubin smooth, repair, flat caps. |
| 16 | Region tag & save | `region_vtk.emit_region_vtk_for_surface` | Write STL + VTK + region-tagged VTK. |

A rendered Mermaid version of this flow (plus the module dependency graph and dataclass model) is in
[docs/uml_diagrams.md](docs/uml_diagrams.md).

---

## Configuration

All tunable behaviour lives in [config.py](config.py), which is the **single source of truth**:

- Module-level constants define backward-compatible defaults.
- Every constant carries a **`# Consumed by:`** comment naming the function(s) that read it;
  **`# UNUSED`** marks knobs with no consumer (cleanup candidates). This makes it easy to trace what
  a knob actually affects.
- **`SdfConfig`** is a `frozen=True` dataclass containing every active runtime
  option. Pass it to `run_pipeline(..., cfg=...)` or
  `generate_sdf_surface(..., cfg=...)`; nested and concurrent runs do not
  mutate module defaults. Both entry points accept `write_outputs=False` and
  `interactive=False` for non-interactive experiments.

A few load-bearing knobs to know first:

| Knob | Default | What it controls |
|------|---------|------------------|
| `SDF_MESH_METHOD` | `"meshlib"` | Iso-surface backend: `vtk_htg` \| `adaptive` \| `cgal_mesh3` \| `meshlib` \| `poisson` \| `mc`. |
| `SDF_FIELD_METHOD` | `"legacy"` | Dense-path field used for parity: `legacy` \| `graph_implicit`. |
| `IMPLICIT_CELLS_ACROSS_DIAMETER` | `12` | Dimensionless local resolution for `vtk_htg` and adaptive diagnostics. |
| `IMPLICIT_JUNCTION_BLEND_FRACTION` | `0.15` | Maximum junction bulge as a fraction of local radius. |
| `PRESERVE_INPUT_RADII` | `True` | Prevent smoothing, tapering and clamping from changing measured radii. |
| `PIPELINE_COMPONENT_FAILURE` | `"error"` | What a failing connected component does: `error` (fail fast, the historical default) \| `continue` (record it and reconstruct the rest). The manifest is written either way. |
| `LEGACY_ORACLE_PRUNE_MODE` | `"geometric"` | How the legacy point oracle excludes cells: `geometric` (hard-union distance certificate) \| `none` (refine everything; slower reference). |
| `BSPLINE_SMIN_K` | `6.0` | Base smooth-min sharpness (higher = sharper junctions). |
| `MIN_STRAHLER_ORDER` | `1` | Drop vessels below this Strahler order (prune the finest twigs). |
| `INPUT_VOXEL_SIZE_UM` | `1.0` | Physical size of one input unit in µm — set to the source voxel size if the graph stores voxel indices instead of µm. |
| `BSPLINE_SDF_RESOLUTION` | `None` (auto) | SDF voxel size in mm; `None` auto-selects from geometry. |
| `MESH_REPAIR` | `False` | Run the `pymeshfix` repair pass after extraction. |
| `OUTPUT_VALIDATION_MODE` | `"off"` | Common final validator: `off` \| `warn` \| `error`; use `error` for CFD runs. |

The compatibility default remains blend `0.15`/support `4`. Synthetic tuning
froze the unqualified CFD candidate at blend `0.05`/support `2`; this does not
alter no-flag behavior. `SdfConfig` snapshots every VTK HyperTreeGrid limit and
strategy flag as well as the optional CGAL criteria.

### Reproducible benchmark

The staged benchmark uses process-isolated timeout/memory limits and records
resource exits as failures:

```bash
python -m coronary_sdf.benchmark \
  --stage all \
  --manifest coronary_sdf/benchmark_cases.json \
  --output benchmark_out \
  --timeout 7200 --memory-mb 16384
```

### Controlled field-versus-extractor ablation

Reconstruction regressions are attributed by varying one thing at a time across
a field × extractor matrix on small real regions:

| | dense (`mc`) | adaptive (`vtk_htg`) |
|---|---|---|
| **legacy field** | `legacy_dense_mc` | `legacy_vtk_htg` |
| **graph field** | `graph_round_cone_dense_mc` | `graph_round_cone_vtk_htg` |

`legacy_dense_meshlib` is retained unchanged as the behavioural baseline, outside
the matrix — it uses MeshLib marching cubes rather than FlyingEdges, so pairing
it against a graph-field run varies the extractor as well as the field.

Generate the regions, then run the matrix:

```bash
python -m coronary_sdf.roi_select \
  --manifest coronary_sdf/benchmark_cases.json \
  --output coronary_sdf/roi_cases \
  --manifest-out coronary_sdf/benchmark_roi_cases.json \
  --audit-cache benchmark_out/cache

python -m coronary_sdf.benchmark --stage ablation \
  --manifest coronary_sdf/benchmark_roi_cases.json \
  --output ablation_out --skip-junction-tuning
```

Regions are located from the graph itself (highest-degree junction, trunk
bifurcation, hairpin self-approach, closest non-adjacent pair, smallest
resolvable radius) and frozen to JSON with provenance. Segments are kept whole,
so a region is a complete sub-network with its own terminals. Region metrics are
restricted both to the evaluation box and to the vessels the graph claims, and
are reported with `metric_scope` — a region HD95 is a conditional metric and is
not comparable with a whole-network one.

Ablation runs are always `purpose: "diagnostic"`; they report full fidelity and
completeness metrics but can never promote a profile.

Useful controls are repeatable `--candidate`, `--case`, `--real-case`,
`--preprocessor`, and `--resume`. Outputs are `runs.jsonl`, `summary.csv`, `report.md`, a frozen
`winning_config.json` and `cfd_qualified_profile.json` only when every CFD gate
passes, plus diagnostic VTP/NPZ artifacts for failed runs and input conflicts.
The selected adaptive candidate requires only the VTK wheel installed with
PyVista. Neither an unavailable VTK backend nor the optional native CGAL
backend is silently substituted.

> **Scale gotcha.** The pipeline hard-codes a `µm → mm` divide by 1000. Coronary `.am`/`.xml`
> exports are already in µm, so keep `INPUT_VOXEL_SIZE_UM = 1.0`. For voxel-indexed graphs (e.g.
> HiP-CT), set it to the acquisition voxel size in µm.

---

## Outputs

Per connected component (`{suffix}` = `_g{id}` for components after the first):

| File | Written by | Contents |
|------|-----------|----------|
| `lumen_bspline{suffix}.stl` | `pipeline.generate_sdf_surface` | Watertight lumen surface (STL). |
| `lumen_bspline{suffix}.vtk` | `pipeline.generate_sdf_surface` | Same surface as VTK PolyData. |
| `lumen_bspline{suffix}_regions.vtk` | `region_vtk.emit_region_vtk_for_surface` | Per-face `region_id` / `is_junction` / `strahler` / `bif_level` tags for colouring. |
| `lumen_vtk_htg{suffix}.stl/.vtk` | `vtk_htg_mesher.mesh_vtk_hyper_tree_grid` | Strictly validated adaptive exact-field candidate output. |

The downstream tools add their own CSV / CCL / JSON sidecars (see below).

---

## Downstream tools

These build on the surface pipeline and the same parser/topology code. Each is a runnable module.

### `flow_fractions` — outlet flow-split boundary conditions
Computes **Van der Giessen** diameter-law flow splits at every bifurcation, propagates them to the
terminal outlets, and matches each outlet to the corresponding boundary zone in an ANSYS `.msh`
generated from the SDF surface. Writes a per-outlet flow-fraction CSV and a CFX `.ccl` mass-flow
boundary-condition file, and can rename the `.msh` zones to match.

```bash
python -m coronary_sdf.flow_fractions  <input.am.xml>  <input.msh>  <output_dir>
```

### `resistance_based_outlet_BC` — resistance-coupled pressure outlets
Companion to `flow_fractions`. Instead of mass-flow outlets, writes **resistance-coupled pressure
outlets**: each outlet's static pressure is tied to the live inlet flow via Ohm's law for fluids
(`P_i = P_distal + Q_i·R_i`), using the Giessen fractions from `flow_fractions`' CSV. Produces a
stable, near-uniform prescribed outlet pressure and lets the CFD mesh set the actual split.

### `epicardial_annotation` + `manual_prune` — vessel annotation & pruning series
`epicardial_annotation` lets you annotate the main epicardial vessels (LAD, LCx, RCA…) **once** in a
3-D picker; they are then always preserved while side branches are pruned at a sequence of radius
ratios, producing a *series* of models plus flow splits — to study how side-branch pruning affects
main-vessel flow. `manual_prune` is the operator-driven counterpart: colour the tree by
Strahler / radius / index and click branches to cut. See
[epicardial_annotation_README.md](epicardial_annotation_README.md) for full usage.

### `wss_postprocess` + `wss_contour_compare` — wall-shear-stress analysis
`wss_postprocess` consumes a CFD wall-node WSS CSV plus a pruned annotated model and reports, at
stations along each main vessel, the worst / shielded / mean WSS from a swept 90° arc window
(excluding bifurcation regions). `wss_contour_compare` samples WSS in fixed 3 mm segments on a
**shared reference centerline** so every pruned model yields measurements at the same locations —
the prerequisite for a paired statistical test (e.g. `scipy.stats.wilcoxon`).

### `tube_union` — alternative boolean-union mesher
An alternative to the volumetric SDF path for graphs with a very wide radius dynamic range (e.g.
HiP-CT microvasculature), where a uniform SDF grid would fuse or fragment sub-voxel vessels. It
builds a capped tube solid per segment plus a small sphere at each junction and merges them with a
robust MeshLib boolean union into a single closed manifold. Cost is `O(edges)` and scale-independent.

```bash
python -m coronary_sdf.tube_union  <input.am|.xml>  <output.stl>  [options]
```

### `adaptive_surface_remesh` — disposable Simpleware master STL

Reduces an over-tessellated lumen STL before Simpleware volume meshing while
retaining finer triangles on small vessels, bends, bifurcations, and close
non-adjacent branches. It uses the cached STL-cropped Amira graph, preserves
the validated source STL, and writes a topology/geometric-error JSON report.

```powershell
.\run_adaptive_surface_remesh.ps1
```

Run this once to create a candidate **master geometry**. Do not use a different
surface remesh at each convergence level. The mesh-sensitivity manifest should
only be changed after the report passes and a disposable Simpleware timing run
confirms the expected speed-up.

After the current Simpleware process has exited, create a disposable SIP whose
selected surface is replaced by the newest accepted candidate:

```powershell
.\prepare_adaptive_surface_benchmark_sip.ps1
```

This copies the source SIP first, imports without implicit Simpleware repair,
and fails if Simpleware reports an open, erroneous, or warning-bearing surface.

To prepare that SIP, create an isolated study configuration, and start the mesh
matrix without mixing it with original-surface results:

```powershell
.\start_adaptive_surface_mesh_study.ps1 -Cores 16
```

Use `-PrepareOnly` to stop after import/configuration, or `-Resume` to continue
an interrupted adaptive-master-surface study.

---

## Package layout

```
coronary_sdf/
├── __main__.py         CLI entry (python -m coronary_sdf)
├── pipeline.py         orchestrator: run_pipeline, generate_sdf_surface
├── config.py           all tunables (single source of truth) + SdfConfig
│
├── parse_amira.py      Amira SpatialGraph reader (.am / .am.xml)
├── topology.py         directed topology, junction labels, NetworkX tree builder
├── centreline_reconnection.py  gap bridging, node weld, contraction, component split
├── pruning.py          Strahler / radius / terminal-nub pruning, radius stats
├── smoothing.py        centerline + radius smoothing, densify, curvature limit
├── splines.py          per-segment cubic spline + branch tangents
├── bif_trim.py         carina taper at bifurcations
├── capsules.py         capsule sampling + KD-tree (CapsuleArrays)
├── sdf_field.py        smooth-min SDF evaluation + anti-bridge carve  ← core
├── implicit_field.py   scale-equivariant analytic field + correctness-bounded BVH
├── adaptive_octree.py  sparse radius-adaptive field hierarchy
├── adaptive_mesher.py  conforming reference tetrahedral extractor
├── mesh_extract.py     dense iso-surface backends, bridge cut, flat caps
├── mesh_repair.py      Taubin smoothing, pymeshfix, hole fill
├── region_vtk.py       per-face region / Strahler / junction tagging
├── viz.py              PyVista debug viewers (gated by DEBUG_VIS)
│
├── flow_fractions.py             Van der Giessen outlet flow split → CFX CCL
├── resistance_based_outlet_BC.py resistance-coupled pressure outlets
├── epicardial_annotation.py      annotate main vessels + pruning series
├── manual_prune.py               interactive click-to-prune picker
├── wss_postprocess.py            radially-resolved WSS along main vessels
├── wss_contour_compare.py        paired-comparison WSS segment sampling
├── tube_union.py                 alternative boolean-union mesher
│
└── docs/
    ├── uml_diagrams.md   Mermaid architecture reference
    └── dashboard.html    visual explainer (this README, illustrated)
```

Files prefixed `_` (`_smoke_test.py`, `_probe_multifurc.py`, `_test_*.py`) are developer scripts,
not part of the production pipeline.

---

## Further reading

- **[docs/dashboard.html](docs/dashboard.html)** — illustrated, interactive walkthrough of the
  pipeline and the logic (opens offline in a browser).
- **[docs/uml_diagrams.md](docs/uml_diagrams.md)** — module dependency graph, pipeline data-flow,
  dataclass model, and per-module API maps (Mermaid).
- **[epicardial_annotation_README.md](epicardial_annotation_README.md)** — full guide to the
  annotation + pruning-series workflow.
- **[config.py](config.py)** — every knob, each with a `# Consumed by:` pointer to its consumers.
