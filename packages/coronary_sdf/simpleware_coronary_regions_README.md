# Coronary boundary and refinement automation

The runnable script is `simpleware_coronary_regions.py`. It targets the
installed Simpleware **X-2025.06** Python 3 API. The recorded macro in
`simpleware_boundary_region_mesh_refinement_API.py` is preserved as the source
example and is not modified.

## What the script creates

The script supports separate centreline sources for boundary planes and mesh
refinement. For this project, the cropped Amira graph is authoritative for both.
Every degree-one endpoint of the retained cropped graph becomes a boundary
candidate, irrespective of Strahler order. This is essential when the STL ends
midway along an order-2/3 vessel. Strahler order is used only for refinement.
The exact capped STL remains the crop authority for the Amira graph: every edge
is clipped to the part inside the STL, new graph terminals are inserted at STL
crossings, and fully external portions are discarded. This mirrors the
crop/truncate behavior in `flow_fractions.py`.

- The largest terminal is treated as the inlet by default. It receives a finite
  `COR_INLET_...` clipping plane and a `Model.VelocityInletContact`; set
  `CREATE_INLET_PLANES = False` only when intentionally inspecting outlets
  without an inlet boundary.
- A finite `Doc.Clipping` plane is also created at every outlet terminal. Starting at
  the actual STL cap/exit, the script targets at least 0.20 mm inward (or 0.75
  local radii) and searches in 0.05 mm increments when needed. No plane may
  remove more than 20% of its retained cap-to-junction terminal branch; a very
  short cropped fragment therefore receives a smaller inset automatically.
  The planar outlet
  cap triangles are clustered and area-fitted; their normal and centre are the
  plane authority. The local Amira tangent constrains the fit to the terminal
  branch and remains the fallback when a coherent cap cannot be found.
- Each candidate is intersected with the STL triangle-by-triangle. It is accepted
  only when it produces exactly one complete closed loop around the branch, the
  full loop fits inside the finite plane, and no second branch or hairpin
  crossing enters the ROI. The ROI is recentered on the measured surface loop
  rather than on a potentially off-centre graph point. Unsafe candidates are
  named and rejected before model mutation.
- Plane sizing calculations use physical half-widths internally and pass those
  values directly as X-2025.06 ROI `scale` values. Doubling a plane scale makes
  its footprint reach adjacent vessels even when the intended rectangle was
  safe. Refinement primitives retain their own documented dimension handling.
- Each outlet ROI is connected to the active model using
  `Model.AddCfdSurfaceContact(..., Model.PressureOutletContact)`. This step is
  required for the ROI to clip the full mesh and appear as an exported CFD
  boundary.
- When the Annotations tool is available, each outlet is given a persistent 3-D
  label `P001`, `P002`, ... at its plane. The annotation name contains the full
  matching `COR_OUTLET_...` ROI name, and the same short label is written to the
  diagnostic CSV. ConsoleSimpleware does not load this tool; after opening a
  console-generated SIP, run `simpleware_add_plane_labels.py` once from the GUI
  Scripting tab to add the labels.
- Every plane normal is explicitly checked against the vector from its inset
  centre to its terminal and flipped when necessary. Outward normals use the
  recorded X-2025.06 macro's
  `inverted=False` setting so the short distal cap, rather than the whole
  proximal lumen, is selected.
- Centreline segments selected by diameter, Amira Strahler order, either
  criterion, or both receive overlapping +FE Free refinement volumes. The
  configured project defaults to all Strahler order 1 and 2 edges, represented
  by overlapping cylinders at 0.5 mm centreline spacing. Full diameters and
  full overlapped lengths are passed to Simpleware, avoiding both gaps and
  half-scale refinement primitives.
  Each volume is assigned to the coronary model part and given an absolute
  target mesh size.

Created objects use these prefixes:

- `COR_OUTLET_`
- `COR_INLET_`
- `COR_SMALL_`

Rerunning the script replaces refinement objects with those prefixes. For this
project, `REMOVE_ALL_EXISTING_CLIPPING_PLANES = True` also removes the 79 legacy
`Finite planeN` objects left by manual macro recording before recreating the
terminal planes. Disable that option in projects containing unrelated clipping
ROIs that must be preserved.

## Run from the VS Code PowerShell terminal

Simpleware X-2025.06 includes `ConsoleSimpleware.exe`, which initializes the
application/API while keeping script output visible in a terminal. Close the
Simpleware GUI first, then run:

```powershell
Set-Location packages/coronary_sdf
.\run_simpleware_coronary_regions.ps1
```

The launcher validates the executable, project, and both scripts; refuses to
run while a `Simpleware.exe` or `ConsoleSimpleware.exe` process exists; makes a
timestamped `.sip` backup next to the project; and writes the combined console
output to `packages/coronary_sdf\logs`. It opens and ultimately overwrites:

```text
%CORONARY_SDF_OUTPUT_DIR%\LADAF_2024_28_full_model_test\meshmixer_full_tree.sip
```

The console entry point calls `document.Save()` only after the full automation
returns successfully. An exception is printed and re-raised without that save.
To check paths and the process guard without making a backup or starting
Simpleware, use:

```powershell
.\run_simpleware_coronary_regions.ps1 -ValidateOnly
```

Do not run the automation with Simpleware's bundled `Python.exe`: that
interpreter can compile the files, but it does not initialize a Simpleware
application or open document. The console run uses the project's persisted
active CFD model and the same automatic surface/part selection as a GUI run.
The console entry point suppresses viewer-only visibility/highlight calls because
ConsoleSimpleware has no Dataset/3D viewer; this does not change the saved ROIs,
CFD contacts, or refinement volumes.

## Identify the surface region selected by one plane

X-2025.06 does not expose a pre-mesh triangle list or area for an ROI-defined
CFD contact. It can report which ROI is assigned to a part, but the surface
highlighting API operates on the complete clipping-ROI set. The diagnostic
runner therefore copies the SIP, removes every other clipping ROI/contact from
the copy, and assigns only the requested named plane:

```powershell
.\run_simpleware_contact_diagnostic.ps1 `
  -ProjectPath "C:\path\to\completed_project.sip" `
  -PlaneName "COR_OUTLET_019_amira_AmiraNode_327"
```

Close all Simpleware processes first. The command prints the new diagnostic SIP
path under `.simpleware_contact_diagnostics`; the source project is unchanged.
When that copy is opened, any highlighted surface comes from the one named
plane. In newly generated projects, read the short `P###` label beside a bad
plane and map it directly to `COR_OUTLET_###_...` in the Document tree or CSV.
If the SIP was generated in console mode, first run
`simpleware_add_plane_labels.py` from the GUI Scripting tab; it does not change
the planes, contacts, mesh settings, or surface.

`simpleware_saved_plane_diagnostic.py` also produces a read-only CSV containing
the plane/contact name, exact-loop coverage, adjacent-loop collisions, local
tangent cosine, and surface-to-graph radius ratio.

## Before running

1. Open the coronary project and create/activate its **CFD model** using the
   **+FE Free** algorithm.
2. Add the coronary fluid surface to that model.
3. Select the coronary surface in the Dataset browser.
4. Choose the centreline source near the top of the script:

   - For Amira, set `CENTRELINE_SOURCE = "amira"` and provide
     `AMIRA_SPATIAL_GRAPH_PATH`. Both native ASCII `.am` and the repository's
     Excel-XML export are accepted by `parse_amira.py`.
     `CORONARY_SDF_REPOSITORY_DIR` must point to the package directory containing
     `__init__.py` and `parse_amira.py`. This is explicit because Simpleware's
     Scripting tab does not provide the source file path through `__file__`.
   - For Simpleware, leave `CENTRELINE_SOURCE = "simpleware"` and ensure only
     the matching network is visible, or set `CENTRELINE_NETWORK_NAME`.

   For Amira mode, also set `STL_SURFACE_PATH` to the exact capped/watertight
   STL used to create the active model surface. `CROP_AMIRA_TO_STL = True` is the
   default. Binary and ASCII STL are supported without extra Python packages.

   The Amira graph and STL must be registered to the Simpleware surface. Parser
   output is in micrometres and `AMIRA_OUTPUT_UM_TO_MM = 0.001` converts it to
   mm. Use `AMIRA_TO_SIMPLEWARE_AFFINE` if graph registration is needed. The STL
   defaults to mm; use `STL_SCALE_TO_MM` and `STL_TO_SIMPLEWARE_AFFINE` if needed.
   Before creating anything, the script checks STL/selected-surface bounds,
   crops the graph, checks cropped-graph/surface overlap, and reports the median
   graph-radius versus surface-wall distance.
5. Review these configuration values:

   - `INLET_COUNT`: use `2` if the network contains two coronary roots; use `0`
     when the network contains outlets only.
   - `INLET_NODE_NAMES`: safer than the size heuristic when node names are
     stable. Explicit names override `INLET_COUNT`.
   - `REFINEMENT_SELECTION_MODE`: choose `"radius"`, `"strahler"`,
     `"radius_or_strahler"`, or `"radius_and_strahler"`. The legacy names
     `"diameter"`, `"either"`, and `"both"` remain accepted.
   - `REFINEMENT_RADIUS_SOURCE`: `"surface_cross_section"` measures the exact
     STL loop at each local sample; `"graph"` uses the Amira radius.
   - `SMALL_VESSEL_CROSS_SECTION_RADIUS_MM`: equivalent cross-sectional radius
     threshold when the surface source is active. `SMALL_VESSEL_DIAMETER_MM`
     remains the threshold for graph-radius mode.
   - `REFINEMENT_STRAHLER_ORDERS`: exact edge orders selected by modes that use
     Strahler metadata, for example `{1, 2}` for distal orders 1 and 2.
   - `REFINEMENT_PRIMITIVE`: `"sphere"` is the project default. A dense chain
     avoids long-chord orientation artefacts and follows tortuous distal
     vessels smoothly. `"ellipsoid"` and `"cylinder"` remain available.
   - `REFINEMENT_SAMPLE_SPACING_MM`: maximum spacing along selected edges. The
     configured 0.30 mm value is now the radius/selection measurement grid,
     not necessarily the final sphere-centre spacing.
   - `REFINEMENT_OPTIMIZE_SPHERE_COUNT`: greedily merge consecutive selected
     measurement intervals without crossing an unselected gap or Amira edge
     node. `REFINEMENT_SPHERE_MAX_RADIUS_EXPANSION_FACTOR = 1.35` limits the
     resulting sphere relative to the largest local wall envelope, while
     `REFINEMENT_SPHERE_MAX_ARC_LENGTH_MM = 0.90` provides an absolute cap.
     Wider/straighter segments therefore use fewer spheres; small or tortuous
     segments retain the density required for complete coverage.
   - `REFINEMENT_PADDING_RADIUS_FACTOR` and `REFINEMENT_PADDING_MM`: set the
     preferred local wall envelope. The default is local radius × 1.25 plus
     0.05 mm, instead of adding a fixed 0.35 mm to every vessel.
   - `REFINEMENT_SURFACE_RADIUS_MIN_GRAPH_FACTOR` and
     `REFINEMENT_SURFACE_RADIUS_MAX_GRAPH_FACTOR`: reject implausible exact-STL
     loop radii caused by oblique/merged sections and fall back to Amira.
   - `REFINEMENT_AXIAL_OVERLAP_RADIUS_FACTOR`: extends cylinders beyond each
     chord end to avoid gaps at bends and graph nodes.
   - `REFINEMENT_NEIGHBOUR_AWARE`: adapt each primitive against unselected
     Amira branches. `REFINEMENT_MIN_PADDING_MM` is the least wall margin;
     primitives that cannot retain that margin without touching an unselected
     vessel are skipped and named in the console log.
   - `REFINEMENT_MESH_SIZE_MM`: desired +FE Free edge length in those regions.
   - `STL_PLANE_SEARCH_MIN_INSET_MM`, `STL_PLANE_SEARCH_STEP_MM`, and
     `STL_PLANE_SEARCH_MAX_INSET_RADIUS_FACTOR`: exact STL plane search range.
   - `STL_PLANE_DESIRED_INSET_MM` and
     `STL_PLANE_DESIRED_INSET_RADIUS_FACTOR`: preferred inward placement.
   - `STL_PLANE_MAX_TERMINAL_BRANCH_FRACTION`: maximum fraction of a retained
     terminal branch that clipping may remove; the default is `0.20`.
   - `STL_PLANE_TANGENT_AVERAGING_LENGTH_MM`: Amira arc length averaged around
     each candidate plane position.
   - `STL_CAP_NORMAL_MAX_GRAPH_DEVIATION_DEGREES`: maximum permitted correction
     from the Amira terminal tangent to the fitted planar STL cap normal.
   - `CREATE_PLANE_ID_ANNOTATIONS`: create persistent `P###` 3-D plane labels.
   - `BOUNDARY_CENTRELINE_SOURCES`: boundary-terminal sources; the project
     default is Amira only: `("amira",)`.
   - `INLET_COUNT`/`INLET_NODE_NAMES`: classify the inlet automatically by
     largest terminal radius or explicitly by Amira node name.
   - `CREATE_INLET_PLANES`: create the inlet clipping ROI; enabled by default.
     `INLET_CONTACT_TYPE` defaults to `"velocity_inlet"`.
   - `BOUNDARY_OUTLET_STRAHLER_ORDERS`: permitted terminal-edge orders for
     boundary planes; the project default is `None`, because an STL crop can
     terminate midway through a vessel of any order.
   - `BOUNDARY_TERMINAL_MERGE_DISTANCE_MM`: endpoint tolerance used to suppress
     a lower-priority duplicate while retaining unique terminals from both.
   - `REFINE_PLANE_NORMAL_FROM_SURFACE`: locally fits the lumen-wall principal
     axis to correct an oblique centreline tangent.
   - `SURFACE_AXIS_MAX_CORRECTION_DEGREES`: rejects implausibly large local-axis
     corrections that could snap a plane toward a neighbouring branch.
   - `PLANE_CLEARANCE_SAFETY_FACTOR`: conservative cap imposed by nearby
     non-local vessels.
   - `GRAPH_CROP_SAMPLE_SPACING_MM`: maximum graph sampling interval used to
     find an STL crossing; the default is `0.20` mm.
   - `GRAPH_CROP_MIN_FRAGMENT_LENGTH_MM`: rejects tiny numerical fragments after
     clipping.
   - `STL_GRAPH_COMPONENT_MODE`: `"relative"` keeps substantial disconnected
     coronary trees while rejecting cropped-branch ostia; use `"largest"` for
     one tree or `"all"` only for diagnostics.
   - `SKIP_UNSAFE_SHORT_TERMINALS`: safely omit branches that cannot accommodate
     the configured node/physical inset without reaching their junction.
   - `SKIP_UNSAFE_CLEARANCE_TERMINALS`: safely omit planes that would reach a
     neighbouring vessel. Set either skip option to `False` for strict failure.
   - `PART_NAME` and `SURFACE_NAME`: only needed when automatic selection is
     ambiguous.

## Recommended first run

For visual inspection before activating clipping, temporarily set:

```python
ADD_CFD_BOUNDARY_CONDITIONS = False
```

Run the script from Simpleware's Scripting tab. Check that:

- every outlet plane fully crosses only its intended vessel;
- no pruned/internal centreline endpoint has been mistaken for an outlet;
- the plane is far enough inside the terminal vessel to make a clean cut;
- refinement spheres enclose the intended small vessels without capturing
  unrelated nearby branches.

Then set `ADD_CFD_BOUNDARY_CONDITIONS = True` and rerun. The script removes its
inspection objects and recreates them with CFD pressure-outlet assignments.
Generate a **Full model**, display **Boundary conditions** under surface
entities, and verify every exported outlet patch.

If a plane highlights the vessel side rather than its terminal side, toggle
`INVERT_CLIPPING_PLANES`. If a plane is too small or large, adjust
`OUTLET_PLANE_DIAMETER_FACTOR` and the minimum/maximum diameter limits.
The configuration uses full physical diameters; the script converts them to
Simpleware's primitive half-extents when constructing the ROI.

## Refinement behaviour

The default selection combines Strahler orders 1/2 with an exact STL
cross-sectional-radius threshold. For each sampled chord, the equivalent loop
radius controls selection and the local sphere envelope. Measurements outside
0.40-1.60 times the Amira radius are treated as oblique/merged-loop outliers and
fall back to the local graph radius. At junctions where one unique loop cannot
be measured, the Amira radius is also the fallback.

Refinement spheres are then checked against densely sampled *unselected* Amira
branches. Their radius-relative preferred padding is reduced locally, down to
`REFINEMENT_MIN_PADDING_MM`, before creation. Selected
neighbouring segments are allowed to overlap because they are part of the same
requested refinement region. If the minimum wall-covering sphere still
touches an unselected branch, that segment is omitted and logged rather than
silently refining outside the region of interest.

The sphere radius includes the small chord half-length in quadrature, so every
sampled vessel section remains covered while the envelope stays rotationally
symmetric. This creates more objects than ellipsoids, but removes the visibly
irregular orientation and long-axis overshoot seen with sparse primitives.
Before clearance validation, consecutive selected intervals are combined using
their sampled tangents, curvature offsets, and local radii. The covering-radius
calculation includes the farthest point of each circular vessel section, so the
optimisation does not merely skip centreline nodes or assume a straight chord.

Strahler selection is available for Amira edges containing a `strahler`,
`StrahlerOrder`, `Strahler`, or `StrahlerNumber` field. If a Strahler-based mode
is requested and an edge lacks that value, the script stops before changing the
model. Refinement object names include `_oNN` when an order is available.

The STL containment test requires a closed, capped surface. An open STL has no
well-defined inside and must be capped before use. Cropped boundary points are
found by bisection, and the retained graph is densified at
`GRAPH_CROP_SAMPLE_SPACING_MM`. Final plane placement is measured from the
actual STL exit and accepted only after the exact closed-loop test; it does not
depend on the spacing of those densified graph nodes.
Cropped side-branch ostia can remain geometrically inside the lumen while being
disconnected from its centreline tree. The default `"relative"` component policy
removes these fragments using the same restrict-to-meshed-tree principle as
`flow_fractions.py`.

The refinement regions are overlapping short primitives rather than a single
primitive per branch. Exact surface radii plus clearance-aware tapered
ellipsoids follow curved vessels while avoiding nearby unselected branches.
If the script reaches `MAX_REFINEMENT_VOLUMES`, increase
`REFINEMENT_SAMPLE_SPACING_MM` or narrow the small-vessel criterion.

## Local validation

The API calls were checked against the X-2025.06 `scripting.pyi` installed with
Simpleware. Geometry helper tests can be run outside Simpleware with:

```powershell
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD='1'
C:\ProgramData\anaconda3\python.exe -m pytest -q test_simpleware_coronary_regions.py
```

Both Python files can be syntax-checked with Simpleware's bundled interpreter:

```powershell
& 'C:\Program Files\Synopsys\Simpleware\X-2025.06\Python3\Python.exe' `
  -m py_compile simpleware_coronary_regions.py simpleware_coronary_regions_console.py
```

The final integration check must use either `ConsoleSimpleware.exe` or an open
GUI project because model, surface, centreline, and licensing state are only
available in an initialized Simpleware application.
