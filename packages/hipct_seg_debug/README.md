# hipct_seg_debug

Overlay a HiP-CT image stack with its Amira segmentation, its skeleton spatial graph
and the reconstructed lumen surface — in one physical coordinate frame — to find where
the pipeline has mis-handled a **collapsed vessel**.

HiP-CT is imaged ex vivo, so vessels are unpressurised and fixed and some lumens are
collapsed. Two failure modes matter:

1. the segmentation **misses** a collapsed vessel entirely;
2. the segmentation **splits one vessel into two** where it collapsed only in the
   middle, after which each half is re-inflated into its own round tube.

Neither is visible in the STL, because generating circular contours from the per-node
radius erases the evidence. The only ground truth is the raw greyscale, so the tool
exists to put the raw image back underneath everything else.

> **Looking for a command?** [CLI.md](docs/CLI.md) lists every one, with the workflows
> that chain them. This file explains why they work the way they do.
>
> **Looking along a vessel instead of across it?** [REFORMAT.md](docs/REFORMAT.md) covers
> the Reformat tab — perpendicular cross-sections stacked along a centreline, why the
> planes have to be kept from folding through each other, and what every control does.
>
> **Choosing a skeletonisation?** [SKELETONISATION.md](docs/SKELETONISATION.md) translates
> the Walsh–Berg super metric — the answer to "there is no ground-truth skeleton, so
> how do I know this one is right?" — and records where `skeleton_analysis`
> diverges from the paper it is a port of.
>
> You do not have to type them. The 3D window carries a **control** dock that runs
> every command as a form, chains the workflows, and loads a different dataset
> without restarting — [CLI.md Part 7](docs/CLI.md#part-7--the-control-panel).

> ### Re-inflation is by design — do not read it as a bug
>
> `adjust_thickness.py` finds the minimum-area reslice through each centreline
> point, keeps the connected component containing the plane centre, and assigns
> **`r = cv2.arcLength(contour) / (2π)`** — on the assumption that the lumen
> *perimeter* survives fixation even though the lumen shape does not. A collapsed slit
> is therefore *supposed* to come out as a much rounder, much larger circle.
>
> It then discards the per-point measurements and reduces them to a **global linear
> fit** `r ≈ slope × (Amira distance-transform thickness) + intercept`, applied
> everywhere. Measured over 11,094 cross-sections, `r_stored / r_perimeter` has median
> **0.96** — the assumption holds in the bulk — but correlation is only ~0.66 with a
> p5–p95 spread of **0.49–2.98**, so at any single location the radius can be out by a
> factor of two or more.
>
> So the question is never "is this section rounder than the segmentation" (it should
> be) but **"does the assigned radius honour the perimeter measured *here*"**. That is
> what `perimeter_mismatch` reports, and what the green-vs-orange comparison shows.

```
python -m hipct_seg_debug
```

Click a region in the 3D window, press `v`, and the slice browser opens alongside it
on the matching raw image slices with every derived representation drawn on top. The 3D
window stays open, so you can keep picking.

---

## Install

### A fresh environment, from nothing

Python **3.12**. The pins in `pyproject.toml` are complete and resolve into an empty
environment (155 packages; verified with `pip install --dry-run --ignore-installed`).

```powershell
conda create -n hipct python=3.12 -y
conda activate hipct

git clone https://github.com/ak-234/HiP-CT-vascular-toolkit.git
cd HiP-CT-vascular-toolkit

# The two sibling packages first, so the edit/SDF features resolve normally.
python -m pip install -e packages/skeleton_analysis
python -m pip install -e packages/coronary_sdf
python -m pip install -e "packages/hipct_seg_debug[test]"
```

This package lives at `packages/hipct_seg_debug` in the
[HiP-CT Vascular Toolkit](https://github.com/ak-234/HiP-CT-vascular-toolkit)
monorepo alongside `coronary_sdf` and `skeleton_analysis`. Installing all three
puts them on the import path, so the entry points run **from any directory** —
no `PYTHONPATH`, no `HIPCT_CORONARY_SDF`, no running from the parent. Drop
`[test]` if you do not want pytest.

Installing `hipct_seg_debug` alone works too, but the edit/SDF half raises a
descriptive `ImportError` until the other two are present.

```powershell
python -m hipct_seg_debug --help        # the viewer
python -m hipct_seg_debug.edit --help   # the editing/repair CLI
```

`pip install -r requirements.txt` still works — that file now just defers to
`pyproject.toml`. The install is editable (`-e`), so edits to the checkout take effect
without reinstalling; drop the `-e` for a copy into `site-packages` instead.

Two console scripts are installed as well, `hipct-seg-debug` and `hipct-edit`, equivalent
to the two `python -m` forms. pip warns if it puts them in a `Scripts`/`bin` directory
that is not on `PATH`; the `python -m` forms need no `PATH` change.

> **`pip` and `python` can be different environments.** If `pip install` reports a
> Python version you did not expect (`requires a different Python: 3.13.x`), use
> `python -m pip install ...` so the install lands in the interpreter you will run.

Check the environment is sound before pointing it at data:

```powershell
python -m pytest   # ~1100 tests, about 4 minutes
```

That needs no dataset and no GUI. To check a *dataset* instead, see
[Validation](#validation) — `--validate-only` proves the four inputs share one
coordinate frame.

> ### Do not let pip upgrade numpy
>
> The pins are load-bearing. `numpy` 2.x breaks `numba` (the RLE decoder) and
> `contourpy`/matplotlib, and requires `scipy>=1.14`. Both `napari>=0.8` and an
> unpinned `opencv-python` (which resolves to 5.x) drag it in. The decisive pins are:
>
> ```
> numpy==1.26.4  scipy==1.13.1  napari[pyqt5]==0.5.6
> pyvistaqt==0.12.0  opencv-python==4.10.0.84
> ```
>
> Check with `pip install --dry-run <pkg>` before adding anything to the environment.

### The two sibling packages

Both live beside this package in the monorepo. Neither is on PyPI, so neither is
a dependency in `pyproject.toml`; install them from the checkout as shown in
[Install](#install). Both are optional — the read-only auditing half of the tool runs
without them, and `src/hipct_seg_debug/edit/_deps.py` fails with an explanation rather than a bare
`ModuleNotFoundError` if something needs one.

| package | needed for | how it is found |
|---|---|---|
| `coronary_sdf` | live SDF surface regeneration (`--edit`), the `surface` command, the `savgol`/`bspline`/`multiscale` smoothers | ordinary import once installed; else `HIPCT_CORONARY_SDF`, else the sibling `packages/coronary_sdf/src` |
| `skeleton_analysis` | the Walsh–Berg super metric (`score`, `optimise-skeleton`) | ordinary import once installed |

```powershell
# From the repository root -- this is the whole setup:
pip install -e packages/skeleton_analysis
pip install -e packages/coronary_sdf

# Only if coronary_sdf lives somewhere outside the monorepo:
$env:HIPCT_CORONARY_SDF = "D:\src\coronary_sdf"
```

### Cascade Forest environment

**Optional — skip it unless you are training or applying a CFC model.**

The official `deep-forest==0.1.7` Windows build does not support the Python 3.12
environment used by the viewer. CFC training, inference, and DPC evaluation therefore
run in a separate Python 3.9 environment; the rest of the application stays on Python
3.12. Create it from the pinned specification:

```powershell
conda env create --prefix .conda-cfc --file environment-cfc.yml
```

`--prefix .conda-cfc` keeps it beside the package rather than in the conda envs
directory, and it is git-ignored for that reason — but **nothing discovers it
automatically**. The interpreter has to be named explicitly, either per run or once in
the environment:

```powershell
python -m hipct_seg_debug --cfc-python .conda-cfc\python.exe
# or
$env:HIPCT_CFC_PYTHON = ".conda-cfc\python.exe"
```

It is used only for CFC workflow steps; everything else stays on 3.12.

---

## Usage

```
python -m hipct_seg_debug --validate-only     # prove the inputs share one frame
python -m hipct_seg_debug --selftest          # end-to-end checks against the greyscale
python -m hipct_seg_debug                     # 3D pick -> slice browser -> repeat
python -m hipct_seg_debug --goto-slice 2596 --goto-row 1709 --goto-col 2182
python -m hipct_seg_debug --goto-candidate 1  # jump straight to a flagged site
```

The four inputs, three of which are required:

```
--raw       directory of raw image slices (TIFF, JPEG 2000, PNG, JPEG, or BMP)
--graph     ASCII Amira spatial graph (.am)
--seg       binary Amira label lattice (.am)
--surface   reconstructed lumen surface (.stl) -- optional; missing is a warning
```

### Naming your data once

Retyping four long paths on every run is unpleasant, so each flag falls back to an
environment variable. Set them once and every command below works bare:

```powershell
$env:HIPCT_RAW     = "D:\data\raw_slices"
$env:HIPCT_GRAPH   = "D:\data\skeleton.am"
$env:HIPCT_SEG     = "D:\data\segmentation.am"
$env:HIPCT_SURFACE = "D:\data\lumen.stl"
```

| variable | flag | used by |
|---|---|---|
| `HIPCT_RAW` | `--raw` | the viewer |
| `HIPCT_GRAPH` | `--graph` | the viewer; the editor's reference graph |
| `HIPCT_SEG` | `--seg` | both entry points |
| `HIPCT_SURFACE` | `--surface` | the viewer |
| `HIPCT_CACHE` | `--cache` | both; defaults to `cache/` under the working directory |
| `HIPCT_CORONARY_SDF` | — | where to find the `coronary_sdf` checkout |
| `HIPCT_CFC_PYTHON` | `--cfc-python` | the Python 3.9 interpreter for Cascade Forest steps |

A flag always beats the variable. There are no built-in paths, so nothing ever loads a
dataset you did not name.

### Starting with nothing

The GUI does not need any of them. With no `--raw` / `--graph` / `--seg`:

```
python -m hipct_seg_debug
```

the 3D window opens empty, and the control dock's **Data** tab loads a dataset into it —
the same panel that swaps datasets mid-session, with a file browser on each of the five
inputs and a history of what you have opened before. Everything that needs data stays
disabled until something is loaded.

The other modes still need their inputs up front, because none of them has anywhere to
ask: `--validate-only`, `--selftest`, and the `--goto-*` flags all exit with a message
naming what is missing.

The ones you reach for most: `--slab N` (slices above/below the pick, default 5),
`--roi PX` (crop width, default 400; **`--roi 0` for the whole slice**), and
`--volume` (see below).

**[CLI.md](docs/CLI.md) is the complete reference** — every flag of both entry points,
every interactive key, and the numbered workflows including CFC-backed variants.

### Seeing more than the ROI

The slab is a window — `--slab` slices by `--roi` pixels around one pick. That is the
right unit for auditing a vessel and the wrong one for asking whether a defect continues
outside the box. Two ways to widen it:

```
python -m hipct_seg_debug --roi 0             # the whole slice, still +/-5 slices
python -m hipct_seg_debug --roi 0 --volume    # ...plus layers spanning the whole dataset
```

**`--roi 0`** makes every slab layer cover the full slice. It costs **no extra decode** —
both readers already decode a whole slice and crop afterwards (`tifffile` has no windowed
read, and `ByteRLELattice.slice_z` always returns the full slice), so the only cost is
memory and a gather. Measured: 0.69 s versus 0.06 s for a 400 px crop, and 427 MB versus
7 MB per slab. The camera recentres on the pick instead of reframing, which at full slice
would zoom out to the whole heart on every pick.

**`--volume`** adds two more layers — `raw (all)` and `segmentation (all)` — that span the
*entire* dataset rather than the slab, so the z slider can leave the slab and follow a
defect through the volume. They are lazy dask arrays chunked one slice at a time, so
nothing is materialised until it is displayed: a segmentation slice costs ~15 ms and a
raw slice ~0.4 s cold. Both start hidden; turn them on in napari's layer list.

The whole-volume mask stays on the **segmentation's own grid** and is placed by napari's
`scale`/`translate` rather than being upsampled onto the raw grid the way the slab's is —
60× less memory for the same picture. The placement carries a half-voxel term, because
`raw_start` is a corner and napari maps pixel *centres*; `selftest.test_lazy_seg_alignment`
checks the two routes agree exactly, over 400 probes.

In the 3D window, **`a`** shows an isosurface of the whole segmentation rather than the
`--seg-box-um` box around the pick — **at full resolution by default**: 3.75 M
triangles, 6.9 s on the first press and 4.7 s after that.

`--seg-stride` still decimates it if you want speed instead (4 → 213 k triangles in
0.2 s), and the **layer panel's stride box and rebuild button change it live** rather
than needing a restart. The mesh is built on first use and then kept, so a stride
change or a painted correction reaches it only when you press rebuild — the status
line says so when it is out of date.

Full resolution used to be documented as "unrenderable". It never was, and the reason
was not the 2.34 GB: `contour()` defaults to `vtkContourFilter`, and switching to
flying edges took it from **19.3 s to 3.0 s for a bit-identical mesh** — same cell
count, same bounds.

The first `a` also makes the mask **resident** — decoded once, 2.2 s — after which
rebuilds and the `g` box read it straight out of memory (a box drops from 61 decodes
to 0.01 s). Corrections painted afterwards are folded back in a plane at a time, so
the resident copy never disagrees with what `slice_z` would return.

### Editing (`edit/`)

Everything above is read-only: this tool finds problems, it does not change anything.
The `edit/` subpackage adds the other half — correct the skeleton **or the mask** and
watch the SDF lumen surface regenerate around the edit, in under a second, because only
the box the edit touched is rebuilt.

```
python -m hipct_seg_debug --edit                       # 3D window, with editing
python -m hipct_seg_debug --edit --paint --edits e.npz # plus a brush on the mask
python -m hipct_seg_debug.edit report  graph.am        # headless: inspect
python -m hipct_seg_debug.edit gaps    graph.am --out fixed.am
python -m hipct_seg_debug.edit connect graph.am --tjunction
```

`--paint` makes the segmentation writable in the slice window, on its own 65.98 µm
grid. Corrections go to a sparse sidecar rather than into the 2.34 GB lattice, and
everything downstream sees them because they all read the mask through one
`slice_z`. *Re-skeletonise painted region* turns a painted correction into
centreline and splices it into the graph, which is what carries it through to the
surface — `coronary_sdf` reads the graph and never the voxels. `mask-export` writes
the result back out as a real `HxByteRLE` Amira lattice.

It also carries scriptable reconnection for skeletons that came out of Avizo in pieces —
intra-segment gaps, end-to-end joins, T-junctions onto the side of a vessel, and the DPC
walk, which decides by reading the greyscale rather than by interpolating — plus
`skeletonise`, which derives an independent centreline from the mask and scores it
against Avizo's. See [edit/README.md](docs/EDITOR.md) for setup, the full command
reference and the Python API. Only the surface half needs `coronary_sdf`; painting,
skeletonising and exporting do not.

### Cropping (`crop`)

Every downstream consumer wants a tree cut down to the vessels actually of interest, and
that cut has to be *explainable* six months later. `crop` makes it, and records why in a
JSON sidecar — the named main vessels, the rule, and the selection it produced. **The
sidecar is the artefact; the `.am` is derived from it**, so a crop can be reviewed,
argued with, and re-run against a repaired graph.

```
python -m hipct_seg_debug.edit crop radius.am --min-strahler 2 --out cropped.am
python -m hipct_seg_debug.edit crop radius.am --crop-json crop.json --ratio-denominator 4 --out cropped.am
python -m hipct_seg_debug.edit crop radius.am --crop-json crop.json   # re-run the recorded crop
```

Three rules, composable in one run, each of which drops a branch **and everything
downstream of it**:

* **`--min-strahler`** — below this order, go.
* **`--min-ostium-um`** — below this take-off radius, go. Needs no main vessels.
* **`--ratio` / `--ratio-denominator`** — a side branch below this fraction of the
  *ostial* radius of the main vessel it descends from. This is
  `coronary_sdf.epicardial_annotation`'s rule: a twig five generations off the LAD is
  judged against the LAD's own proximal radius, not against its immediate parent's.

Main vessels are **named in the Crop tab**, either one segment at a time or two picks at
a time: tick `trace between two picks`, click the ostium, click the far end, and every
segment on the path between them joins the vessel. On a tree that path is unique; where
the segmentation has fused two vessels into a cycle, `prefer thick` routes the trace down
the vessel rather than over the thin bridge between them. A trace never doubles back and
never goes round a loop — it is a simple path or it is a refusal. Either way the vessels
are stored geometrically (endpoint and mid-point coordinates hashed at 1 µm) rather than
by edge number, because every write renumbers edges. The tab also takes hand marks: `o`
drops the picked segment, `h` prunes everything past the pick, and `Preview` draws what
the rule would take in dulled red over the tree it would take it from. Nothing in the tab
writes to the dataset — it hands you the `crop` command line and queues it.

Main vessels, and the path from the root down to one, are never dropped: a rule that
removed an unannotated left main would take the LAD with it. Strahler orders are
recomputed after the crop by default, because the stored ones describe the tree as it
was. Full flag table in [CLI.md](docs/CLI.md).

### CFC-guided DPC reconnection

The code and paper call this the **DPC walk** (Distance-Probability-Cosine); “DCP” in
notes or discussion refers to the same algorithm. The scan-specific Cascade Forest
Classifier (CFC) supplies `P`, the probability that a candidate raw-image voxel lies on
a true vascular centreline. It is intended for collapsed gaps where the segmentation
contains no foreground to follow.

Each classifier input contains exactly 686 unnormalised raw-intensity values:

1. a centred raw `15x15x15` patch, with its first `14x14x14` samples max-pooled in
   non-overlapping `2x2x2` blocks to `7x7x7`;
2. the centred raw `7x7x7` patch;
3. pooled-large then small, flattened in C order with axes `(z, y, x)`.

Training uses `N` unique skeleton voxels, `2N` lumen negatives from segmentation
foreground away from the centreline, and `2N` outside-lumen negatives within seven
segmentation voxels of the wall. Sampling is deterministic. Validation is grouped by
skeleton component (with spatial groups as a fallback) so adjacent voxels do not leak
between training and validation. The reported metrics are accuracy, sensitivity,
specificity, balanced accuracy, ROC-AUC, and PR-AUC; the deployable DF21 model is then
refitted on all samples.

For the full LADAF-2024-28 skeleton, omit `--max-positive`:

```powershell
& '.conda-cfc\python.exe' -u -m hipct_seg_debug.edit train-cfc `
  'D:\data\candidate.am' `
  --seg 'D:\data\segmentation.am' `
  --raw 'D:\data\raw_slices' `
  --out-model 'models\LADAF-2024-28-cfc' `
  --seed 0
```

The artifact directory contains the native DF21 `model/`, `manifest.json`,
`metrics.json`, and `samples.npz`. The manifest pins the raw geometry, the 686-value
feature contract, sampling and DF21 settings, source fingerprints, runtime versions,
and seed. Inference rejects a model trained for a different stack shape, voxel spacing,
or feature contract. `--max-positive N` is useful for a smoke test; a capped model is
not the final full-skeleton model. An existing non-empty artifact directory is refused
unless `--overwrite` is explicitly supplied.

Launch the Python 3.12 GUI with both the model and its Python 3.9 interpreter:

```powershell
py -3.12 -m hipct_seg_debug `
  --raw 'D:\data\raw_slices' `
  --graph 'D:\data\candidate.am' `
  --seg 'D:\data\segmentation.am' `
  --cfc-model 'models\LADAF-2024-28-cfc' `
  --cfc-python '.conda-cfc\python.exe' `
  --no-surface
```

In **control -> Workflows**, select **8C. Repair a graph with CFC-guided DPC**,
choose a new run directory, and run the chain. It produces:

```text
step1-cfc.am   intra-edge gaps repaired
step2-cfc.am   CFC-guided Type 1 -> Type 2 -> Type 3 reconnections
step3-cfc.am   reconnected graph with repaired radii
```

The source graph is never overwritten. To inspect a result, put `step2-cfc.am` or
`step3-cfc.am` in the Data tab's graph field and press **Reload graph**. Workflow
**11C. Full run with CFC-guided DPC** includes the same reconnection stage inside the
full skeleton/mask/surface pipeline and pauses for human review.

At each walk step the score is `D + omega*P + C`, with `(1, 5, 1)` as the default
weights. Only the neighbourhood probability is min-max normalised. The corrected walk
uses a coarse/fine two-level `5x5x5` neighbourhood, the paper's cosine gate, explicit
type-specific reach, distance to the target polyline for Type 3, and sequential routing:

- Type 1: endpoint to endpoint between two disconnected non-backbone components;
- Type 2: disconnected endpoint to a backbone endpoint;
- Type 3: disconnected endpoint to the interior of a target vessel segment.

Accepted connections are incorporated before the next type is proposed, so topology,
components, and endpoints are recomputed between stages. Completed paths are validated
with probability and raw-greyscale sequences that include five known vessel voxels at
both ends. The report separates paper-style ADF stationarity failures from the existing
HiP-CT mean/drop/trough/greyscale safeguards.

Nine versioned LADAF-2024-28 review regions live in
`edit/reconnect/dpc_eval_regions.json`: two candidates and one hard negative for each
type. `export-dpc-regions` creates blinded orthogonal views, projections, overlays, and
a review CSV. A reviewer must enter the correct type/target and the paper labels
`TP_b`, `TN_b`, `FP_b`, `FN_b`, `TP_s`, or `FP_s` in the manifest; geometric roles are
not clinical ground truth. `evaluate-dpc` then sweeps one global integer `omega=0..7`,
reports the paper's `RecAcc`, `RecSen`, and `RecSpe` overall and per type, and runs DPC,
DP, PC, and DC ablations. The legacy full-neighbourhood mode remains available through
`connect --dpc-neighbourhood full` for a separate comparison. Without reviewed
labels it deliberately reports `omega=5` only as a pending-label default, not a measured
optimum. See [CLI.md](docs/CLI.md#train-and-evaluate-the-cascade-forest) for exact commands.

### 3D window

| key | action |
|---|---|
| double-click | pick the nearest centreline point (or a candidate marker) |
| drag | rotate the camera — never picks |
| `v` | open / update the slice browser at the pick |
| `i` | raw image plane through the pick (off by default) |
| `g` | segmentation mask around the pick (off by default) |
| `a` | segmentation mask, whole tree (off by default; built on first use) |
| `n` / `b` | jump to the next / previous flagged candidate |
| `s` | allow / forbid picking the surface (off by default) |
| `L` | show / hide the legend box (shift+l; on by default) |
| `1` / `2` / `4` | Reformat tab: add the picked segment (or trace between two) / clear / show the stack |
| `8` | Reformat tab: cancel a half-finished trace |
| `5` / `6` / `7` | Sections tab: add the picked segment / clear / cut the debug sections |
| `c` | clear the pick |
| `r` | reset the camera |
| `q` | close the 3D window |

### Layer panel

Docked on the right of the 3D window: one row per layer — surface, centreline,
candidates, graph-wide radius circles, image plane, segmentation (around the pick, and
whole-tree), the three
per-slice shapes, and the two crop overlays — each with a visibility checkbox and an opacity slider. Rows for inputs this session does not have
(no surface loaded, no candidates found) are greyed out rather than hidden, so the panel
reads the same every run.

`colour by` at the top of the panel maps **both** the centreline and the radius circles
— one scale for the two, because a ring is the cross-section *of* the centreline it sits
on. `radius (um)` is the stored point radius, on a continuous viridis ramp. `Strahler
order` is the order of each point's owning edge, drawn as discrete bands with one label
per order rather than as a ramp: it counts branching generations, so a continuous bar
would invite reading a 2.5 off it. Junction points, which appear once per incident edge,
take the order of the branch they belong to. The row is greyed out on a graph that
carries no Strahler attribute (`strahler`, `StrahlerOrder`, `Strahler` or
`StrahlerNumber` — whichever name the file used), and a dataset swap onto such a graph
falls back to radius.

`i` and `g` are the same state as the corresponding checkboxes, in both directions.
Opacities are *remembered*, not just applied: the image plane and the mask isosurface are
rebuilt from scratch on every pick, so a value written only to the actor would snap back
to the default the next time you picked. `--plane-opacity` and `--surface-opacity` set
the starting values.

`save figure...` writes the 3D view to SVG (or PDF, EPS, PS, TeX) as a publication
figure rather than a screenshot: the keybinding block and the pick readout are left out,
and the legend box and the colour bar are kept exactly as they are drawn — so a tree
banded by Strahler order arrives with the bar that says which band is which order. A
legend switched off with `L` stays off; the export is of the view you set up. What lands
in the file is a split: VTK exports through GL2PS, whose OpenGL2 backend writes the
geometry as one embedded raster image, while the overlay labels — legend entries, the
colour-bar title, the per-order annotations — come out as real `<text>`, sharp at any
zoom and editable in Inkscape or Illustrator.

`radius circles` is a graph-wide diagnostic layer, separate from the picked-slab
`assumed cross-section`. It draws one closed ideal circle per stored spatial-graph point,
using that point's `thickness` as the radius and a plane perpendicular to the local
centreline tangent. The rings use the same viridis scale as the centreline, whichever scalar that is. At a
bifurcation, repeated endpoint records retain their incident edge's orientation, so
several differently oriented rings may share one centre. The layer is off by default
and its approximately 1.8-million-vertex mesh on the full graph is built and cached only
when its checkbox is first enabled. It is not pickable.

### Image and mask overlays in 3D

`i` drops the raw slice through the pick into the scene as a translucent plane, in the
same 1–99.5th-percentile window the slice browser uses, so the two agree. **It follows
the browser's z slider** — scroll there and the plane moves with you. Full slices are
already decoded and LRU-cached by the slab builder, so it costs nothing across a slab.

`g` runs marching cubes over the Amira mask in a box around the pick and draws it in the
same blue the 2D view uses, rebuilt on each pick. The whole lattice is 2.34 GB decoded,
but a slice decodes in ~1 ms, so a local box is cheap. Together they put the greyscale,
the mask, the skeleton and the reconstructed surface in one picture.

The box is `--seg-box-um` (default ±2000 µm) **or three times the local radius,
whichever is larger**. A box smaller than the lumen it sits in is entirely mask, so
marching cubes finds no boundary and draws nothing — and the fattest vessel here has
r = 1.55 mm, so a fixed extent would come up empty on exactly the vessels most worth
looking at. If it still does, the status says so rather than leaving you guessing.

Neither overlay is pickable — the image plane spans the entire field of view and would
otherwise swallow every double-click aimed at a vessel behind it.

### Slice shapes in 3D

The three per-slice overlays the browser draws — the red `STL contour`, the green
`assumed cross-section` and the orange `perimeter circle` — are mirrored into the 3D
scene as polylines lying in the image plane, with the same colours and the same
defaults (contour on, the diagnostic pair off).

They are **handed over by the browser rather than recomputed**, which is the point: the
STL has already been cut against each plane and each cross-section measured, over the
same window, so the two windows cannot disagree about what they are drawing. It also
bounds the work — every slice the slab covers is already in hand and scrolling is free.
The window is whatever the slab used, so under `--roi 0` these are the whole tree's
cross-section rather than one vessel's; under `--volume` the slider can move past the
slab, and those slices simply have no shapes.

The consequence is that the three rows stay greyed out until `v` has been pressed, and a
pick somewhere else clears them until the slab is rebuilt: they belong to a slab, not to
the scene. Where the surface does not reach — it is a partial model covering ~42 % of
this skeleton — the STL contour is legitimately empty.

> **`add_key_event` appends; it does not replace.** pyvista binds its own defaults, and
> our handler runs *in addition* to them unless the key is cleared first. `v` was bound
> to `isometric_view_interactive()`, so opening the slice browser also snapped the camera
> to a fixed viewpoint; `b` installs a fresh `LeftButtonPressEvent` observer on every
> press. `Picker3D.build` now calls `clear_events_for_key` for every key it binds, and
> leaves only `q` (pyvista's close) and `r` (VTK's reset-camera) inherited.

**Selection is a double-click.** A single left press is how VTK starts a camera
rotation, so picking on it drops a pick wherever each drag begins and the pick creeps
around while you are only looking at the tree. pyvista's double-click detector gates on
distance *and* time (~6 px apart, within 0.8 s), so a press-drag-press does not count
either.

**A pick is always a spatial-graph point.** Picking uses a `vtkPointPicker` restricted
to the centreline and the candidate clouds, so a pick snaps to a skeleton vertex and
the status reports which one, with its radius and owning edge. Double-clicking further
than about 10 px from any vessel reports nothing rather than guessing. `s` re-enables
free picking on the surface if you ever need a point the skeleton does not cover.

Two actors are deliberately **not** pickable, and both must stay that way:

- the reconstructed surface is drawn translucent, so a ray aimed at a vessel visible
  *through* the shell would stop on the near shell instead — millimetres from the vessel
  that was clicked;
- the cyan pick marker, or it intercepts the *next* click in the same region and walks
  the pick towards the camera by its own radius, one click at a time. That is enough to
  leave the ±5-slice slab, so the second and later picks would open the wrong slice.

Both windows stay open and share one Qt event loop, so you can pick, look, and pick
again without losing your camera. Repeated picks reuse the same slice window. The
status text says whether the active point came from a click or from candidate
browsing, so a stale candidate can never be mistaken for a fresh click.

### Slice browser

napari's layer list gives a visibility checkbox per layer; the opacity slider applies
to whichever layer is selected. The dims slider scrolls z, and the overlay in the
corner shows the true TIFF slice number.

| layer | colour | meaning |
|---|---|---|
| `raw` | greyscale | the ground truth |
| `segmentation` | blue | the Amira binary mask, upsampled onto the raw grid |
| `reconstructed lumen (r)` | violet | union of the spheres the skeleton radii imply — what the model *claims* the lumen is |
| `surface (STL)` | red | cross-section of the reconstructed surface on this plane |
| `assumed cross-section` | green | the ellipse the assigned radius implies (off by default) |
| `perimeter circle` | orange | the circle the pipeline's own rule (`P/2π`) would give from *this* slice (off by default) |
| `skeleton` | yellow | where the centreline crosses this slice |
| `probability` | magma | second lattice field, off by default (see caveat below) |

**How to read it.** Turn on `assumed cross-section` (green) and `perimeter circle`
(orange) together — that pair is the whole diagnostic:

- **green ≈ orange** → the assigned radius honours the perimeter measured here. If both
  are much larger and rounder than the blue segmentation, that is a collapsed lumen
  being re-inflated exactly as intended. **Not a bug.**
- **green ≫ orange, or orange missing** → the assigned radius is not supported by this
  cross-section. Reported as `perimeter_mismatch`; this is a real error.
- **two separate blue blobs inside one dark lumen in the raw image** → one collapsed
  vessel segmented as two. Reported as `companion_lumen`, and only one of the two is
  measured by the pipeline.
- **a lumen visible in the raw image with no blue on it at all** → a missed collapsed
  segment. Nothing can flag this automatically; it is why the raw layer is underneath.

---

## Flagged candidates

Candidate detection is on by default; it writes `cache/candidates.csv` and shows the
sites as clickable markers in 3D. These are leads, not verdicts — the raw image
decides.

### From the image (`crosssection.py`)

Each cross-section is measured in the plane **perpendicular to the local centreline
tangent** — for a prismatic tube that is the minimum-area plane, so it reproduces the
pipeline's own three-angle search cheaply. Perimeter uses `cv2.arcLength`, the same
estimator `adjust_thickness.py` used, so a disagreement is real and not a difference of
method.

The fitted tangent is not taken on trust when the section it cuts comes back
elliptical. The three-plane stability slab that validates a normal is stepped *along*
that normal, so a tilted plane through a straight vessel cuts three identical ellipses
and scores perfectly while over-reading the radius by `1 / cos(tilt)`. An elongated
section is therefore re-cut over the bounded search cone and the shortest boundary
wins, which straightens an oblique cut and leaves a genuinely collapsed lumen where it
is — see `--transverse-axis-ratio` in [docs/CLI.md](docs/CLI.md).

- **`perimeter_mismatch`** (red) — **the actual error detector.** The assigned radius
  differs from the perimeter measured at this cross-section by more than
  `--mismatch-factor` (default 1.5×), sustained along the branch. The model is not
  doing what it claims to do here.
- **`companion_lumen`** (purple) — a second lumen beside the centreline one.
  `clean_and_measure_slice` only ever measures the component containing the plane
  centre, so a companion is silently discarded: the signature of one collapsed vessel
  segmented as two.
- **`collapse_severity`** (pink) — **informational, not an error.** Isoperimetric ratio
  `P²/(4πA)` above `--severity-iso` (default 1.6). Ranks where the perimeter assumption
  is carrying the most weight, so the sites most dependent on it can be eyeballed.

Sites are emitted only for sustained runs along an edge, never single-point dips.

Two connectivity subtleties, both deliberate: the measured component uses
4-connectivity to match the pipeline, while companion detection uses 8-connectivity —
without that, every diagonally-pinched lumen reports a spurious companion one voxel
away. Where the two disagree the pipeline measured only part of a pinched lumen, and
the `perimeter_mismatch` detail says so.

### From the graph (`candidates.py`) — sparse on a cleaned tree

- **`premature_end`** (orange): a branch of Strahler order ≥ 2 simply stops. Vessels
  taper through the orders, so a trunk that terminates is the strongest topological
  sign the segmentation lost it.
- **`murray_deficit`** (blue): a bifurcation whose daughters cannot carry the parent —
  `sum(r_child³)` far below `r_parent³`, judged against the tree's own distribution
  (`--murray-percentile`). Consistent with a lost collapsed daughter branch.
- **`endpoint_gap`** (yellow): a dead-end pointing at a vessel that is far away *in
  the tree* but close in space.
- **`parallel_pair`** (green): two topologically distant branches running alongside
  each other, closer than the sum of their radii, over a sustained length.

**Both pair detectors are tree-aware and must stay that way.** Branches meeting at a
bifurcation are close and near-parallel by nature. Filtering only on "shares a vertex"
leaves every uncle/grandparent pair one hop further out, which on this tree produced
33 candidates of which all 33 were false positives. Pairs therefore require a minimum
separation in the edge-adjacency graph (`--min-hops`, default 4) and comparable
Strahler orders. Pairs in different trees have infinite separation and always qualify.

**Murray's law must be sampled away from the junction.** Every edge at a vertex shares
that point, so reading the three radii *at* the vertex returns the same number three
times and the ratio is identically 2.00 — an earlier version did that and wrongly
concluded the test was unusable on this data. Sampled two local radii along each branch
(`candidates._radius_along`) the tree gives median 0.83, p5–p95 0.14–1.90, and nothing
at 2.00. A self-test guards this.

**Terminal radius is genuinely uninformative**, though: 145 of 161 terminals sit at
exactly 156.640 µm (six distinct values in all), a floor left by `adjust_thickness.py`.
That is why `premature_end` keys off Strahler order rather than size.

---

## Coordinate model

Everything is reduced to **micrometres**, with the raw stack defining the grid
(Amira uniform convention, voxel 0 at the origin):

```
x_um = col * vx      y_um = row * vy      z_um = slice * vz
```

The segmentation is a binned, cropped lattice placed by its `BoundingBox`; the STL is
in millimetres. `frame.WorldFrame` derives the binning and crop offset at runtime —
for LADAF-2024-28 that comes out as **2×2×2 binning cropped from raw
(col 154, row 501, slice 2101)**.

**The voxel size itself is stated, not inferred.** `--voxel-um` is required, and the
Data tab refuses to load until it is confirmed. The two things that used to supply it
— the folder name (`32_99um…`) or a TIFF tag, and the bounding box's own
`spacing = (hi - lo) / (dims - 1)` — are wrong in the way nothing downstream can see:
a bounding box written from a rounded voxel size is internally consistent, so every
check passes while every radius, length and volume carries the same error. 32.99 µm
recorded for a 32.04 µm scan is 2.96%, which is larger than most of the corrections
`radius-perimeter` exists to make.

Stated, the number is the authority: the bounding box is read as the same lattice
measured in the wrong units, and the whole µm world — lattice origin and spacing,
graph coordinates and radii, STL scale — is rescaled onto it. Voxel indices do not
move, so nothing comes out of alignment; every length changes. Outputs are written on
the corrected scale and stamped with `HiPCTVoxelSizeUm`, so re-loading one does not
correct it twice. [CLI.md](docs/CLI.md#voxel-size) has the details.

`POINT { float thickness }` in the spatial graph is a **radius in µm**, matching
`RADIUS_SCALE = 1.0` in `Coronary_lumen_octree.py:559`. The edge attribute
`MeanRadius` is a separate Avizo statistic and is *not* what generated the surface.

### Validation

`--validate-only` proves the frame rather than assuming it. On this dataset:

```
[PASS] raw voxel size consistent      nominal 32.99 um, bin 2x2x2, 0.000% off
[PASS] crop offset on raw grid        raw start (col 154, row 501, slice 2101)
[PASS] segmentation within raw stack  seg z 69328-151738 um, slices 2101-4600
[PASS] graph within segmentation bbox
[PASS] surface within segmentation bbox
[PASS] surface radius matches thickness   median ratio 1.013
[PASS] centreline inside mask             99.9% of 20000 points
[PASS] orientation unambiguous            flips: y 1.7%, x 0.7%, z 1.4%, xy-swap 0.8%
```

The orientation check is the decisive one: it re-runs the inside-mask test under every
axis flip and transpose, and each must fail badly for the direct mapping to be
trusted.

`--selftest` goes further and checks the *interpretation* against the greyscale: the
cached RLE index must reproduce a plain sequential decode byte-for-byte; a µm→raw→µm
round trip must land within a voxel; the mask must be ~2r wide at the fattest
centreline point; and the bright vessel wall must ring the centreline far more
strongly than at a decoy position 6 radii away (11.7× here).

---

## Performance

Nothing is loaded whole — the segmentation alone would be 2.34 GB decoded.

- A one-off pass builds a per-slice entry point into each `HxByteRLE` stream (~1 s,
  cached in `cache/*.sliceidx.npz`); after that any segmentation slice decodes in ~1 ms.
- Raw image slices are read one at a time with a small LRU cache. TIFF uses
  `tifffile`; JPEG 2000 (`.jp2`, `.j2k`, `.j2c`, `.jpc`) and conventional image
  formats use Pillow.
- The 2 M-triangle surface is reduced to the ROI by cell-centroid selection (~0.09 s)
  before z planes are cut out of it (~5 ms per slice).

A pick on the reference data takes about 4–6 s end to end, dominated by decoding 11
LZW TIFF slices. Decode time for other formats depends on their codec and compression.

---

## Caveats for this dataset

- **`Probability` carries no signal.** 99.96 % of its voxels are exactly 255 across
  only five quantisation levels — it is Avizo's label-confidence field, not the
  network's pre-threshold output. It cannot reveal sub-threshold vessels. The layer is
  supported and off by default; validation prints a warning when it detects this.
- **`lumen_bspline.stl` covers ~42 % of the skeleton.** It is a partial model, so the
  red contour is simply absent over much of the tree. Point `--surface` at
  `lumen_bspline_g1_full_skeleton.stl` in the same folder for fuller coverage. The
  violet `reconstructed lumen` layer is derived from the skeleton and is available
  everywhere.
- The skeleton was computed on the 2× binned lattice, so its points sit at half-integer
  raw z. The browser intersects each edge polyline with every slice plane rather than
  snapping, so markers appear on every slice.

## GUI traps, in case this code is extended

The first two cost real debugging time and both fail as a hard process kill, not an
exception.

1. **Never build the napari viewer inside a VTK callback.** Pressing `v` runs inside
   VTK's key handler with VTK's OpenGL context current; creating vispy's canvas there
   initialises GL against the wrong context and the process dies with
   `access violation reading 0x...` inside `glDrawArrays`. `Picker3D._open` defers the
   work with `QTimer.singleShot(0, ...)` so VTK finishes first.
2. **Never `viewer.layers.clear()` and re-add to refresh.** That destroys visuals with
   a draw still pending and crashes vispy the same way. `viewer2d._update_layers`
   assigns to `.data` in place.

And one that fails silently instead, which is worse:

3. **A hidden layer never recomputes its extent.** `Layer._refresh_sync` returns *before*
   clearing the extent cache when `visible` is False, so a hidden layer keeps the
   previous slab's bounds. That stale extent is unioned into `dims.range` (widening the
   z slider across both slabs) and into the extent `reset_view` frames (zooming out over
   empty space) — and because `Dims` silently *clips* `point` into `range`, asking for a
   slice napari does not believe in is not an error, it just shows a different slice.
   `viewer2d._force_extents` refreshes every layer with `force=True`, which is the
   documented way past the visibility gate, then pushes `dims.range` across by hand
   (napari only recomputes it from data/transform events, not extent events).
   `_show_slice` then reads the slider position back and warns if it did not land.
4. **Never key per-viewer state on `id(viewer)`.** napari's `Viewer` is a pydantic model
   and rejects arbitrary attributes, so the bookkeeping has to live outside it — but
   CPython reuses ids, so once the slice window is closed and a new viewer is built, it
   can inherit the dead one's entry: the slice-number overlay never reconnects to the
   slider, and the info panel's `QLabel` has no C++ half left to `setText`.
   `viewer2d._viewer_state` hangs the dict off `viewer.window._qt_window` instead, which
   has exactly the right lifetime.

---

## Repository layout

```
src/hipct_seg_debug/     the package -- `pip install -e .` puts this on the import path
    edit/                skeleton and mask editing, reconnection, local surface regeneration
    edit/reconnect/      the four reconnection proposers (gaps, CFC, DPC, geodesic)
tests/                   the pytest suite; `python -m pytest` runs it
docs/                    CLI.md, EDITOR.md, REFORMAT.md, SKELETONISATION.md, GEODESIC_RECONNECTION.md
environment-cfc.yml      the separate Python 3.9 environment the Cascade Forest steps need
```

Sources live under `src/` rather than at the repository root, so the package is
importable only once installed. That is deliberate: it stops a stale working copy
shadowing the installed one, and it means the tests always exercise what a user
would actually get.

## Module map

Paths are relative to `src/hipct_seg_debug/`.

| file | role |
|---|---|
| `amira.py` | ASCII spatial graph and binary lattice header readers |
| `rle.py` | numba `HxByteRLE` slice index and random-access decode, by whole slice or by row band |
| `frame.py` | `WorldFrame` conversions, the stated-voxel unit correction, and the validation table |
| `tiffstack.py` | lazy raw slice/window access |
| `stl_slice.py` | ROI clipping and z-plane cutting of the surface |
| `candidates.py` | tree-aware graph heuristics, hop distances, CSV export |
| `crosssection.py` | perpendicular-plane cut of the lumen (`cut`), and the audit of the perimeter rule built on it |
| `viewer3d.py` | PyVista pick window (non-blocking `BackgroundPlotter`) |
| `controls3d.py` | docked layer panel for the 3D window |
| `controls_crop.py` | the Crop tab: name main vessels, set a rule, preview what it would take |
| `controls_sections.py` | the Sections tab: re-cut `radius-perimeter`'s planes at sampled points and draw them |
| `edit/section_frames.py` | those cuts as geometry: window corners, the measured lumen, and the centroid the point was not moved to |
| `edit/crop.py` | the crop rules, the take-off measurement and the crop sidecar |
| `viewer2d.py` | slab assembly and the napari browser |
| `volume.py` | lazy whole-dataset layers (`--volume`) |
| `rle_write.py` | `HxByteRLE` **encoder** — writes a corrected mask back as an Amira lattice |
| `selftest.py` | end-to-end checks against the greyscale |
| `main.py` | CLI and session wiring |
| `workflows.py` | GUI command chains, including additive workflows 8C and 11C |
| `edit/` | skeleton and mask editing, live local surface regeneration, reconnection ([README](docs/EDITOR.md)) |
| `edit/reconnect/cfc.py` | 686-value raw patch features, deterministic sampling, DF21 artifacts and cached inference |
| `edit/reconnect/dpc.py` | corrected Type 1/2/3 Distance-Probability-Cosine walk and validation |
| `edit/reconnect/evaluation.py` | blinded review export, omega sweep, paper metrics and ablations |
| `edit/skeletonisers.py` | Lee, TEASAR and Avizo-ingest behind one signature |
| `edit/supermetric.py` | the Walsh–Berg super metric ([SKELETONISATION.md](docs/SKELETONISATION.md)) |
| `edit/skeleton_optimise.py` | de-loop, prune, re-centre, smooth |
| `edit/smoothers.py` | five selectable centreline smoothers, three of them `coronary_sdf`'s |
| `edit/radius_perimeter.py` | per-point radius from the cross-section perimeter |

Caches (RLE indices, `candidates.csv`) go to `cache/` under the working directory —
override with `--cache` or `$HIPCT_CACHE`. Delete it to force a rebuild.
