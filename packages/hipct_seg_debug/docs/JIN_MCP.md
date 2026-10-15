# Jin minimum-cost-path skeletonisation (experimental)

`jin-mcp` implements the method in Jin et al., *A Robust and Efficient Curve
Skeletonization Algorithm for Tree-Like Objects Using Minimum Cost Paths*,
Pattern Recognition Letters 76, 32–40 (2016),
[doi:10.1016/j.patrec.2015.04.002](https://pmc.ncbi.nlm.nih.gov/articles/PMC4860741/).
It is a separate skeletonisation candidate. It does not replace `dfs-centroid`
or refine an existing graph while retaining its segment IDs.

> **To run it end to end** (mask correction, skeletonisation, pruning, simplification
> and smoothing, from the command line or the GUI), follow
> [JIN_MGF_WORKFLOW.md](JIN_MGF_WORKFLOW.md). This page describes the method.

## Implemented method

The core accepts fuzzy membership in [0,1] on a cubic grid. The label-lattice
adapter explicitly converts nonzero labels to binary membership. A 26-neighbour
fuzzy distance transform uses edge length times mean endpoint membership,
including the half-membership foreground/background boundary step. Equation 2
defines local significance (LSF) from the largest positive normalised FDT slope.
Strong quench voxels have LSF > 0.5.

Each round finds connected unmarked subtrees, selects their geodesically farthest
strong quench voxel, and traces minimum-cost paths to the current skeleton.
Equation 6 uses `length / (0.01 + mean(LSF_p, LSF_q)**2)`.
Branch significance is the sum of LSF outside the previously marked volume;
acceptance requires significance greater than `3 + 0.5*FDT(attachment)`.
Geodesic dilation by twice FDT marks the represented volume. Multiple subtrees
share the same distance/predecessor maps in each round (section 2.5).

Numba compiles the implicit-grid Dijkstra search; no dense voxel adjacency graph
is constructed. The first call includes compilation. This is a bounded-region
implementation, with a default limit of two million input voxels. That limit is
an input guard, not a guarantee of peak memory: priority queues can grow too.
It is not an out-of-core or full-tree implementation.

## Run a region

From an environment with this monorepo's `hipct_seg_debug` package installed:

```powershell
python -m hipct_seg_debug.edit skeletonise-all `
  --seg "path/to/artery.labels.am" --voxel-um 32.04 `
  --algorithms jin-mcp --stride 1 `
  --roi-zyx 100 200 200 300 300 400 `
  --per-tree --min-component-voxels 100 `
  --no-score --out-dir "runs/jin-region"
```

The bounds above are illustrative: replace them with the intended source-voxel
region. Bounds are Z0 Z1 Y0 Y1 X0 X1, upper bounds exclusive. Only those slices
and windows are decoded. The world origin is adjusted on export. A cut vessel
at a crop face has an artificial boundary; include substantial surrounding
segmentation and do not interpret cut-face endpoints as anatomical terminals.
This local extraction is not equivalent to full-volume extraction.

Output is `jin-mcp.am`, with branch acceptance reports in `jin-mcp-reports/`.
The default extraction root is the maximum-FDT voxel of each component. For a
single connected ROI, omit `--per-tree` and optionally specify
`--jin-root-zyx Z Y X` **relative to that ROI**. The existing root picker and
graph root sidecars operate after extraction; they do not set this root.

The Python API `edit.jin_mcp.extract(membership, root_zyx=...)` retains fuzzy
memberships. `edit.skeletonisers.skeletonise('jin-mcp', volume, frame, ...)`
converts label data to binary and returns a candidate graph. Anisotropic
spacing is rejected; the published scale thresholds are in cubic-grid voxels.

## Qualification and limits

Analytic binary/fuzzy distance tests and round, curved, three-arm and six-arm
phantoms check the fields, centreline location, branching, containment and tree
connectivity. A shallow protrusion test checks suppression of a false branch.
The isolated-root, world-coordinate export and rejection cases are also tested.

**A strongly flattened rectangular tube produces extra branches on its medial
sheet.** This counterexample is retained in the test suite. Passing these tests
therefore does not mean this method solves the flattened-coronary problem.
Reports explicitly require review. No subvoxel smoothing, perimeter measurement,
surface reconstruction or collision correction is performed here.

Input must be one connected component per extraction. A non-unit Euler
characteristic is rejected to catch obvious tunnels/cavities; a unit value alone
does not prove their absence. This method assumes tree-like objects and outputs
a tree with new IDs. It is not suitable for preserving anatomical loops.

Exported radii are **FDT thickness placeholders**, not measured perimeter radii.
Keep this candidate separate from measured and reconstruction graphs. Compare
the complete junction neighbourhoods and branch correspondence against the
segmentation before choosing geometry, then remeasure with the shared section
filter. Neither the three clinical target regions nor a full tree have been
qualified with this backend. Do not use its output directly as a final surface.

## Collapse-aware variant: `jin-mgf`

`jin-mgf` (`edit/jin_mgf.py`) keeps the published control loop above unchanged and
adds two switchable modifications through hooks on `jin_mcp.extract`. With both
off (`medialness=lsf;marking=dilate`) the output is identical to `jin-mcp`, which
a test pins.

- **`medialness=mgf`.** Strong-quench voxels are labelled curve or sheet by the
  ridge dimension of the smoothed EDT Hessian (`classifier=hessian`, the default),
  or by PCA of the strong set (`classifier=pca`, the methodology note's form). On
  sheet voxels a voxelised *medial geodesic function* (Dey & Sun, SGP 2006) is
  computed. `mgf_mode=exact`, the default, takes the boundary geodesic between
  antipodal contacts; `rim` is the cheap in-sheet distance-from-rim proxy.
  Medialness is then `LSF x w`, where `w` is near 1 on the MGF ridge (negative
  divergence of its unit gradient) and `w_min` elsewhere on the sheet. Curves
  keep LSF.
- **`marking=section`.** After the paper's 2 x FDT dilation, the lumen in a
  one-voxel slab normal to each accepted path is also marked. A slab still open
  at its window edge is refused.

The Hessian classifier is the default because PCA labelled most of a voxelised
round tube as sheet: LSF has digital "spokes" there. Sheet rims are given the MGF
too (`sheet_support`). The Hessian calls them "curve", and with full LSF the path
rode along the slit's edge.

On the flattened tube that `test_flattened_tube_exposes_medial_sheet_limitation`
keeps for `jin-mcp`, each modification alone gives exactly two terminals. The
phantom results are in [SKELETON_BENCHMARK.md](SKELETON_BENCHMARK.md). The same
limits as `jin-mcp` apply: an isotropic grid, one component, a tree, and FDT
placeholder radii.

```powershell
python -m hipct_seg_debug.edit skeletonise-all --seg $env:HIPCT_SEG --roi-zyx Z0 Z1 Y0 Y1 X0 X1 `
  --algorithms "jin-mcp,jin-mgf,jin-mgf[mgf_mode=rim],jin-mgf[medialness=lsf;marking=section]" `
  --out-dir runs/jin-mgf --summary-json runs/jin-mgf/summary.json
```

### Smoothing the output

A Jin path runs through voxel centres, so consecutive steps turn a median 45°.
`fair-centreline` (`edit/centreline_fair.py`) smooths it inside the mask. It fits each
vessel as one curve through its junctions, and it ignores a branch's first stretch
inside its parent, so the branch leaves the junction without a kink. Run it on a
written graph (see [CLI.md](CLI.md#fair-centreline--smooth-each-vessel-through-its-junctions)),
or ask for it in the skeletoniser as `jin-mgf[fair=true]`.

A branch at right angles to a vessel and shorter than about `fill_max_radii` (12) of
the vessel's radii is not extracted at all. `marking=section` fills the parent's
normal slab, and that slab holds the whole branch. Branches at 45° are extracted.

## Whole trees: the sparse engine

`edit/jin_sparse.py` runs the same method on lumen voxels only. The dense
implementation keeps about six float64 fields over the bounding box, which is why it
has a 2 M-voxel guard; a LADAF-2021-17 tree box is 24 G voxels.

- **Memory.** The only box-sized arrays are a 4-byte lookup and the mask.
  Everything else is per lumen voxel, and every per-voxel loop is numba.
- **Identical to dense.** With `incremental=False` the output matches the dense
  engine's exactly: the same vertices, edges and FDT on every test shape and
  phantom. Raster order is preserved, so heap and component ties break as they do
  in the dense code.
- **`incremental=True`, the default.** Each round continues from the last round's
  shortest-path maps instead of starting again; this is the differential IFT the
  methodology note proposes. Both seed sets only grow, so the distances are exact.
  Only tied predecessors can differ.
- **`allow_loops`.** This lets a mask with loops through the tree-only Euler guard.
  The output is still a tree, with the loops cut. The 2021-17 mask has
  χ = −1467, so it needs this.
- **`jin-mgf` on the sparse engine.**
  - The Hessian classifier uses a float32 `edt` plus a Gaussian over the box, and
    the same `np.gradient` stencils, evaluated at strong-quench voxels only. It
    labels exactly the same sheet voxels as dense.
  - `rim` mode is identical to dense.
  - `exact` mode finds contacts by local search rather than a box-sized feature
    transform, and runs its per-voxel boundary geodesics across all cores.
    Equidistant contacts tie-break by scan order, so its output can differ from
    dense by sub-voxel amounts; the leaf and sheet counts are unchanged on the
    phantoms.
  - The PCA classifier is dense-only.
- **Choosing the engine.** `--jin-engine auto` (the default) uses dense up to
  `--jin-max-voxels` and sparse beyond. `--jin-engine dense` keeps the old
  decode-time refusal.
- **Measured.** On a whole LADAF-2024-28 tree (2.6 M lumen voxels), `jin-mcp`
  takes 36 s over 14 rounds and `jin-mgf` 146 s.

```powershell
python -m hipct_seg_debug.edit skeletonise-all --seg $env:HIPCT_SEG --voxel-um 32.04 `
  --per-tree --workers 3 --jin-engine sparse --jin-allow-loops `
  --algorithms "jin-mcp,jin-mgf" --no-score --out-dir $env:HIPCT_OUT\jin-full
```

## Rooting at the inlet

By default Jin roots each tree at its maximum-FDT voxel. On a coronary tree that is
inside the widest vessel, near the inlet but short of it. Jin then traces back out to
the inlet's cut face, and on a flared inlet it reaches the face's rim twice.
- **On LADAF-2021-17:** the left tree's inlet came out as a Y, a junction 2.0 mm from
  the inlet with tips 1.8 and 2.2 mm away.
- **The fix:** give the inlet. `--root-graph` takes each tree's widest free end from
  a graph you trust, and `--root-xyz` takes a world position, e.g. from the viewer's
  pick readout. Roots are snapped onto each tree's lumen within 12 voxels.

```powershell
python -m hipct_seg_debug.edit skeletonise-all --seg $env:HIPCT_SEG --voxel-um 32.04 `
  --per-tree --jin-engine sparse --jin-allow-loops --algorithms jin-mgf `
  --root-graph refined_dfs.am --out-dir $env:HIPCT_OUT\jin-rooted
```

## Centroid/curvature hybrid and VMTK comparison

`--algorithms jin-mcp,jin-mcp-centroid` writes separate raw and refined candidates.
The hybrid applies `dfs-centroid-shape` after Jin extraction, with
`--jin-refine-iterations 5` by default. Its section-centre data term and curvature
term favour a coherent centred curve; they do **not** modify Jin's published
shortest-path cost. Each stage has a separate report. Refinement preserves the
extracted topology and radii, so it does not silently remove medial-sheet branches.
The hybrid uses the refinement stage's automatic graph roots and fixed terminals;
the extraction root can be an interior voxel that is not an exported graph node.
For controlled anatomical roots, refine the exported graph separately with an
explicit root sidecar. See [CENTRELINE_REFINEMENT.md](CENTRELINE_REFINEMENT.md).

An optional `--algorithms vmtk` adapter provides an independent surface-based
baseline when VMTK is installed. It requires one connected ROI (omit `--per-tree`)
and explicit world-coordinate seeds:

```powershell
python -m hipct_seg_debug.edit skeletonise-all --seg region.am --algorithms vmtk --vmtk-source-xyz 100 200 300 --vmtk-target-xyz 900 200 300 --no-score --out-dir runs/vmtk-region
```

Replace these illustrative coordinates with suitable inlet/outlet surface seeds.
Repeat `--vmtk-target-xyz` for additional outlets. Following VMTK's point-list
interface, seeds snap to the nearest surface vertex. The adapter extracts a
marching-cubes surface with an exterior background layer, without mesh smoothing
or centreline resampling. Cropping introduces artificial caps. VMTK's radii are
maximal-inscribed-sphere values, requiring separate perimeter remeasurement.
The resulting graph still requires containment and centring checks.

VMTK is not installed in the development environment. The optional integration
test therefore skips; no VMTK accuracy or timing result is claimed. A missing
dependency is reported explicitly. The adapter does not install or substitute
another algorithm automatically.
