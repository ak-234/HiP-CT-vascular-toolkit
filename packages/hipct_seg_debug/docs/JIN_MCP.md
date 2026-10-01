# Jin minimum-cost-path skeletonisation (experimental)

`jin-mcp` implements the method in Jin et al., *A Robust and Efficient Curve
Skeletonization Algorithm for Tree-Like Objects Using Minimum Cost Paths*,
Pattern Recognition Letters 76, 32–40 (2016),
[doi:10.1016/j.patrec.2015.04.002](https://pmc.ncbi.nlm.nih.gov/articles/PMC4860741/).
It is a separate skeletonisation candidate. It does not replace `dfs-centroid`
or refine an existing graph while retaining its segment IDs.

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
