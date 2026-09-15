# CLI reference

Every command, every flag, every key, and the workflows that chain them.

This is the reference. The two READMEs explain *why* things work the way they do —
[README.md](../README.md) for the auditing tool, [edit/README.md](EDITOR.md) for
correction and surface regeneration — and point here for the flags.

There are exactly two entry points:

| | |
|---|---|
| `python -m hipct_seg_debug` | the **viewer**: validate, audit, browse, edit, paint. Opens windows. |
| `python -m hipct_seg_debug.edit` | the **toolkit**: twelve headless subcommands over a graph, mask, CFC model, or DPC evaluation set. |

**Every command below has a GUI form.** The 3D window carries a
`control` dock — three tabs (Data, Commands, Workflows) above a Log pane — that runs
every command here as a form, chains the workflows in [Part 3](#part-3--workflows), and
loads a different dataset without restarting the process. Its forms are generated
from the same argparse definitions this document describes, so the two cannot
disagree. DF21-only commands must use the separate Python 3.9 interpreter; workflows
8C and 11C do that automatically. See [Part 7](#part-7--the-control-panel).

---

## Conventions

**Install once, then run from anywhere.** `pip install -e .` in the checkout puts
`hipct_seg_debug` on the import path, so both entry points work in any working
directory. Nothing needs to be on `PYTHONPATH`, and no directory has to be current.

**Everything is dry-run unless you ask for output.** This is a package of
heuristics: a bridge that should not exist silently reroutes flow in whatever CFD
run follows, so the default is to print what would happen and write nothing.

| command | without an output flag | with one |
|---|---|---|
| `report` | prints; there is no output flag | — |
| `gaps` `connect` `flag-interpolation` `skeletonise` `optimise` `repair-radius` `repair-mask` | prints a plan, writes nothing | `--out PATH` |
| `connect --geodesic` | prints a plan, writes nothing | `--out PATH` **and** `--out-seg PATH`, together |
| `skeletonise-all` | **always writes** to `--out-dir` (default `skeletons/`) | `--out-dir DIR` |
| `train-cfc` | refuses a non-empty artifact directory | `--out-model DIR` (required) |
| `export-dpc-regions` | refuses to run | `--output DIR` (required) |
| `evaluate-dpc` | writes `dpc-evaluation.json` | `--output PATH` |
| `surface` | **always writes** to `--out-dir` (default `surface/`) | `--out-dir DIR` |
| `mask-export` | refuses to run | `--out PATH` (required) |

**Micrometres, everywhere.** Graph coordinates, radii, box sizes, thresholds. The
only exceptions are `--edit-voxel-mm` / `--voxel-mm`, which are millimetres because
that is what `coronary_sdf` works in internally, and `--stl-scale`, which is the
multiplier taking surface units to µm (1000 for a mesh written in mm).

**Paths come from you, not from the source.** There are no built-in dataset paths.
Pass `--raw` / `--graph` / `--seg` / `--surface`, or set `HIPCT_RAW` / `HIPCT_GRAPH` /
`HIPCT_SEG` / `HIPCT_SURFACE` once and run every command bare. A flag always beats the
variable.

**The viewer starts without them.** `python -m hipct_seg_debug` with nothing named opens
an empty 3D window; load a dataset from the control dock's Data tab. `--validate-only`,
`--selftest` and the `--goto-*` flags still require inputs, and say which are missing.

**Exit codes.** `0` success, `1` a self-test or a command failed, `2` argparse
rejected the command line. `--validate-only` exits `1` on a failed check unless
`--no-strict`.

---

## Cheat sheet

| I want to… | command |
|---|---|
| do any of the below without retyping paths | `python -m hipct_seg_debug`, then the **control** dock ([Part 7](#part-7--the-control-panel)) |
| check the inputs share one coordinate frame | `python -m hipct_seg_debug --validate-only` |
| prove the whole thing against the greyscale | `python -m hipct_seg_debug --selftest` |
| find and rank suspect sites | `python -m hipct_seg_debug` (writes `cache/candidates.csv`) |
| look at one known location | `python -m hipct_seg_debug --goto-slice 3361 --goto-row 2638 --goto-col 969` |
| browse and pick interactively | `python -m hipct_seg_debug` |
| correct the skeleton, surface following | `python -m hipct_seg_debug --edit` |
| correct the mask by hand | `python -m hipct_seg_debug --edit --paint --edits work/edits.npz` |
| see the whole volume, not just the slab | add `--volume`, or `--roi 0` for whole slices |
| describe a graph | `edit report GRAPH.am` |
| mark what Avizo interpolated rather than measured | `edit flag-interpolation GRAPH.am --show-spans --out flagged.am` |
| mend jumps inside one edge | `edit gaps GRAPH.am --out fixed.am` |
| join broken vessels | `edit connect GRAPH.am --tjunction --out fixed.am` |
| train the scan-specific CFC on every skeleton voxel | `edit train-cfc GRAPH.am --seg SEG.am --raw RAW --out-model MODEL` in Python 3.9 |
| reconnect with CFC-guided DPC | `edit connect GRAPH.am --tjunction --dpc --raw RAW --seg SEG.am --cfc-model MODEL --out fixed.am` |
| repair graph *and* mask together | `edit connect GRAPH.am --geodesic --seg SEG.am --raw RAW --out fixed.am --out-seg fixed-seg.am` |
| export the nine blinded DPC review cases | `edit export-dpc-regions GRAPH.am --seg SEG.am --raw RAW --output REVIEW_DIR` |
| sweep global `omega=0..7` and run ablations | `edit evaluate-dpc GRAPH.am --seg SEG.am --raw RAW --cfc-model MODEL --omega 0:7` |
| restore collapsed radii | `edit repair-radius GRAPH.am --source both --out fixed.am` |
| derive a skeleton from the mask | `edit skeletonise --order --out candidate.am` |
| skeletonise the left and right trees separately | `edit skeletonise --per-tree --min-component-voxels 2000 --out candidate.am` |
| choose each tree's inlet by clicking it | `edit pick-roots candidate.am --roots-json roots.json` |
| score each tree on its own terms | `edit score radius.am --scope per-tree` |
| score only the largest tree, all five terms | `edit score radius.am --scope largest` |
| score a skeleton against Avizo's | `edit optimise candidate.am --sensitivity` |
| cut the tree down to the vessels of interest | `edit crop GRAPH.am --crop-json crop.json --out cropped.am` |
| clean debris out of the mask | `edit repair-mask --min-voxels 500 --out mask.tif` |
| write the corrected mask back out | `edit mask-export --edits work/edits.npz --out corrected.am` |
| regenerate the lumen STL | `edit surface GRAPH.am --out-dir surfaces/` |

(`edit` above is short for `python -m hipct_seg_debug.edit`.)

---

# Part 1 — the viewer

```
python -m hipct_seg_debug [flags]
```

## The five modes

`main()` checks them in this order, and the first match wins:

| # | trigger | what happens |
|---|---|---|
| 1 | `--validate-only` | coordinate checks, print, exit. ~15 s. |
| 2 | `--selftest` | 21 end-to-end checks against the greyscale, exit. ~2 min. |
| 3 | any `--goto-*` | validate, detect candidates, open **one** slab in napari, block. |
| 4 | *(default)* | validate, detect candidates, open the 3D pick window + slice browser. |

Modes 3 and 4 both run validation and candidate detection first, so a default run
costs ~18 s before a window appears (3 s loading, 15 s detecting).

`--goto-*` precedence, when more than one is given: **`--goto-um` › `--goto-candidate`
› `--goto-slice`**. `--goto-row` / `--goto-col` only qualify `--goto-slice`, and
default to the middle of the stack.

```
python -m hipct_seg_debug --goto-um 68000 55000 96000
python -m hipct_seg_debug --goto-candidate 1
python -m hipct_seg_debug --goto-slice 3361 --goto-row 2638 --goto-col 969
python -m hipct_seg_debug --goto-slice 2727                 # centre of that slice
```

## Inputs

| flag | default | meaning |
|---|---|---|
| `--raw DIR` | `$HIPCT_RAW` | directory of raw image slices (TIFF, JPEG 2000, PNG, JPEG, or BMP) |
| `--graph PATH [PATH ...]` | `$HIPCT_GRAPH` | ASCII Amira spatial graph; pass several and they are merged into one editable graph, each held as its own tree |
| `--seg PATH` | `$HIPCT_SEG` | Amira label lattice, binary or a `.Regions.am` naming its trees |
| `--surface PATH` | `$HIPCT_SURFACE` | reconstructed lumen surface (optional) |
| `--pattern GLOB` | `*` | which supported image files in `--raw` are slices |
| `--cache DIR` | `$HIPCT_CACHE`, else `./cache` | RLE slice indices and `candidates.csv` |
| `--voxel-um N` | **required** | the acquisition's own raw voxel size. Never inferred — see [Voxel size](#voxel-size) |
| `--stl-scale N` | `1000.0` | multiplier taking surface units to µm |
| `--labels-field NAME` | `Labels` | lattice field holding the binary mask |
| `--probability-field NAME` | `Probability` | optional second lattice field |

## Voxel size

`--voxel-um` is required, and is the one input the loader will not guess at.

Two sources used to fill it in: the raw folder name or a TIFF resolution tag, and the
segmentation's own `BoundingBox` (`spacing = (hi - lo) / (dims - 1)`). Both can be
wrong, and they are wrong in the way that cannot be noticed — a bounding box written
from a rounded voxel size is *internally consistent*, so every conversion agrees with
every other, the validation table passes, and every radius, length and volume in the
session carries the same error. On LADAF-2024-28 the recorded 32.99 µm against the
acquisition's 32.04 µm is 2.96%, larger than most of the corrections
`radius-perimeter` exists to make; the only place the true figure appeared was a
skeleton file name, which nothing reads.

So the value is stated. The load error names what each source suggests, so a correct
one can be pasted; the Data tab pre-fills the box with the same suggestion and refuses
to load until it is confirmed.

**What stating it does.** The number is taken as the truth and the bounding box is
read as the same lattice measured in the wrong units, so the whole micrometre world is
rescaled by `stated x bin / bbox_spacing`:

* voxel *indices* do not move, which is what keeps the images, the mask, the graph and
  the surface aligned with each other;
* every *length* does — the lattice origin and spacing, the graph's coordinates and
  radii (`thickness`, `MeanRadius`), and the STL scale.

Outputs are written on the corrected scale, and carry a `HiPCTVoxelSizeUm` parameter
in their `Parameters` block saying so. That stamp is load-bearing: the scale factor
comes from the *segmentation*, which does not change when a graph is written, so
without it re-loading a corrected graph beside the same mask would apply the same
correction a second time — and the result would still sit inside the mask and still
pass every check. A stamped graph is left alone; a selection mixing stamped and
unstamped skeletons is refused rather than guessed at.

In the `edit` CLI the same flag has the same meaning, and the same rescale, wherever a
command opens the lattice. It stays optional there: omitted, nothing is corrected and
the files' own units are used. The Commands tab fills it in from the session, so a job
launched from the GUI measures on the same scale the viewer is showing.

The validation table reports whichever reading was taken:

```
  [PASS] voxel size corrected to yours      stated 32.0400 um; bounding box implied
                                            32.9900 um; every length scaled by
                                            0.971203 (-2.88%) at bin 1x1x1
```

## What is shown

| flag | default | meaning |
|---|---|---|
| `--slab N` | `5` | slices above *and* below the pick (so 11 in total) |
| `--roi PX` | `400` | crop width in raw pixels. **`--roi 0` shows the whole slice.** |
| `--volume` | off | add lazy whole-dataset layers, so the z slider can leave the slab |
| `--probability` | off | load the probability field as an overlay |
| `--no-surface` | off | skip the STL entirely |
| `--no-tube` | off | skip the rasterised skeleton-radius layer |
| `--seg-box-um N` | `2000.0` | half-extent of the 3D mask isosurface around a pick (`g`) |
| `--seg-stride N` | `1` | resolution of the whole-tree 3D mask (`a`). `1` is full resolution — 3.75 M triangles, 6.9 s cold, 4.7 s after. `4` is 213 k in 0.2 s. Changeable live from the layer panel. |
| `--plane-opacity N` | `0.55` | starting opacity of the 3D raw image plane (`i`) |
| `--surface-opacity N` | `0.35` | starting opacity of the 3D reconstructed surface |

## Detection

| flag | default | meaning |
|---|---|---|
| `--no-candidates` | off | skip detection entirely (saves ~15 s) |
| `--no-crosssection` | off | graph heuristics only; skip the image-based collapse detector |
| `--min-hops N` | `4` | minimum tree separation for a reported pair; below this it is a bifurcation, not a defect |
| `--murray-percentile N` | `0.10` | flag bifurcations in this lowest fraction of the tree's own `Σr_child³/r_parent³` distribution |
| `--severity-iso N` | `1.6` | isoperimetric ratio above which a lumen is called strongly collapsed |
| `--mismatch-factor N` | `1.5` | flag where the assigned radius differs from the measured perimeter by more than this factor |

Detection writes `cache/candidates.csv` and prints a summary. On LADAF-2024-28 it
finds **249 sites in 14.6 s** across six kinds — `premature_end`, `murray_deficit`,
`endpoint_gap`, `parallel_pair` from the graph; `collapse_severity`,
`perimeter_mismatch`, `companion_lumen` from the image.

## Editing

| flag | default | meaning |
|---|---|---|
| `--edit` | off | enable skeleton editing with live SDF surface regeneration |
| `--edit-box-um N` | `4000.0` | side of the box rebuilt around an edit, µm |
| `--edit-voxel-mm N` | derived | pin the SDF voxel size, **mm** |

Needs `coronary_sdf`; see [edit/README.md](EDITOR.md) for how it is located.

## Painting

| flag | default | meaning |
|---|---|---|
| `--paint` | off | add a writable segmentation layer to the slice browser |
| `--paint-box N` | `192` | side of the paintable block, in **segmentation voxels** (192³ ≈ 12.7 mm, 7.1 MB) |
| `--edits PATH` | none | mask edit store `.npz`; loaded at start, saved on exit |

`--paint` works without `--edit` and never imports `coronary_sdf`. Only
*Re-skeletonise painted region* needs it, and that button is absent without `--edit`.

**Give `--edits` a path.** Without one, corrections live only in memory and the
session warns on exit that they were never saved.

## Other

| flag | meaning |
|---|---|
| `--no-strict` | warn instead of aborting when a coordinate check fails |
| `-h`, `--help` | the authoritative list; this document is checked against it |

---

# Part 2 — the toolkit

```
python -m hipct_seg_debug.edit {report,gaps,connect,surface,
                                skeletonise,optimise,repair-radius,
                                repair-mask,mask-export,train-cfc,
                                export-dpc-regions,evaluate-dpc} [flags]
```

Graphs must be **ASCII** Amira `HxSpatialGraph`. Binary `.am` is not supported —
re-export as ASCII from Avizo.

### Shared flags

Commands that take a graph take it as a **positional** argument, and most also take
`--out PATH`. `report` and `surface` do not: `report` writes nothing, and `surface`
writes a mesh to `--out-dir`.

### Skeletons: every graph command takes several

The `GRAPH` positional is variadic on **every** command that processes a skeleton —
`report`, `gaps`, `connect`, `optimise`, `optimise-skeleton`, `radius-perimeter`,
`crop`, `score`, `surface`, the diagnostics, all of them:

```
python -m hipct_seg_debug.edit optimise-skeleton left_tree.am right_tree.am --out refined.am
```

Several are **merged into one graph, each source held as its own tree**, and `--out`
receives that one graph — the same arrangement as the GUI's `skeleton (.am)` field,
for the same reason: every command works on one graph, so merging is what makes all of
them reach every tree. The `tree` field keeps the sources apart, which is what `crop`,
per-tree scoring (`--scope per-tree`) and the roots sidecar already read. A source that
already carries trees keeps them, shifted past the ones before it.

**A merge renumbers ids**, and the run says so. `--segment`, `--root-edge` and anything
else quoting a number read off a viewer then names a position in the *merged* graph.
Nothing is renumbered when one graph is given, so the usual case is unchanged.

`pick-roots` is the exception that does not merge: it opens the picker on each skeleton
in turn and writes one sidecar covering all of them.

Commands that read the segmentation share these six:

| flag | default | meaning |
|---|---|---|
| `--seg PATH` | the LADAF-2024-28 lattice | Amira label lattice, binary or a `.Regions.am` naming its trees |
| `--labels-field NAME` | `Labels` | which field to read |
| `--voxel-um N` | recorded / derived | the true raw voxel size. Given, it is the authority ([Voxel size](#voxel-size)); omitted, use the input graph's recorded voxel size, falling back to the bounding box only for unstamped graphs |
| `--stride N` | `1` | decimate every axis before decoding. 2 → 293 MB, 4 → 37 MB, 8 → 4.6 MB. |
| `--edits PATH` | none | composite a painting session's corrections before anything reads the mask |
| `--ignore-materials` | off | split by connectivity alone, ignoring a `Materials` block that names the trees |

#### Masks that already name their trees

An Avizo `.Regions.am` carries a `Materials` block — on this data `Exterior`,
`Left_Tree`, `Right_Tree` — and the voxels hold the material's **index**, `0/1/2`.
(Not the `Id` field beside it: `Left_Tree` says `Id "9"` and its voxels hold `1`.)

When a mask names more than the Exterior, every command that splits it now splits **by
material first and by connectivity inside each**. That matters because connectivity
alone answers a different question: at 32 µm the two coronaries touch, so `mask > 0`
labels them as a *single* component, and "tree 0" and "tree 1" end up being the two
largest fragments — often both from the same artery. With materials, tree 0 is
`Left_Tree` and tree 1 is `Right_Tree`, their detached fragments follow, and
`--max-trees` caps what is kept **per material** so a global cap cannot drop a whole
coronary. The material name is carried into the `tree` field, the roots sidecar and
the 3-D view. `--ignore-materials` restores the old behaviour for comparison.

`--edits` is available on `skeletonise`, `optimise`, `repair-mask`, `mask-export`
and `repair-radius`. It refuses a store built for a differently-shaped lattice
rather than scattering corrections into unrelated tissue.

`--stride` is a *decimation*, not a crop: at stride 4 only every fourth plane is
read, so corrections on the planes in between do not appear. Use stride 1 when the
output matters.

**Budget for stride 1 on a full lattice.** Anything that splits the mask into
components runs one 26-connected labelling over every decoded voxel, single-threaded.
On the 3400x2964x4748 LADAF-2021-17 lattice that is 47.8 G voxels and takes tens of
minutes — measured at 56 s for the 0.75 G voxels of stride 4, and it scales with
voxel count. The command says `labelling N G voxels...` before it starts and reports
how long it took, so a long pass reads as busy rather than hung. Work at `--stride 4`
while you are deciding what to run, and pay for stride 1 once.

---

### `report` — describe a graph

```
python -m hipct_seg_debug.edit report GRAPH.am
```

No flags beyond the positional. Prints components and their sizes, free ends,
radius quantiles, the ten largest intra-segment gaps, and how much of the graph
Avizo interpolated. Writes nothing — it is the "what am I dealing with" command.
Instant.

---

### `flag-interpolation` — mark what Avizo invented

```
python -m hipct_seg_debug.edit flag-interpolation GRAPH.am --show-spans --out flagged.am
```

| flag | default | meaning |
|---|---|---|
| `--show-spans` | off | list every span with its evidence, not just a tally |
| `--seg PATH` | — | also check each span against the segmentation (ground truth, costs a decode) |
| `--anchor-jump N` | 0.05 | minimum radius step at each end of a straight run |
| `--min-span-points N` | 3 | shortest run that counts as an invented vessel |
| `--min-jump-um N` | 600 | shortest intra-edge step that can count as an unsampled jump; needs `--seg` |
| `--no-jumps` | off | skip the unsampled-jump signature |
| `--out PATH` | — | write the annotated graph |

Where the segmentation has a hole, Avizo writes points across it so the edge stays
continuous. Those points are not skeletonisation output — nothing measured them, the
radius is generated, and a tree that is genuinely in two pieces looks like one. This
command finds them and records the answer as a per-point `avizo_interpolated` field
in the `.am`, which every other stage then reads.

Three signatures, each with its own reason code:

| code | signature |
|---|---|
| 1 | **straight bridge** — no curvature, an exact linear radius ramp, and a radius step ≥ `--anchor-jump` at *both* ends that dwarfs the run's own taper |
| 2 | **degenerate radius** — a radius sitting on a detached floor at the bottom of the distribution (`adjust_thickness`'s clamped intercept, or a raw zero) |
| 4 | **off-mask** — a *run* of points outside the segmentation; needs `--seg` |

On LADAF-2024-28 this flags 36 of 29,122 points (0.12%): two straight bridges, on
edges 183 and 197, plus ten points on the 81.27 µm radius floor. The both-ends test
is what does the work — collinearity alone would flag 941 runs, and adding the linear
ramp still leaves 941.

**Nothing detects on its own.** Every pipeline stage reads the stored field and
treats an unflagged graph as clean, so this command is the only way in (the 3D
viewer is the one exception, and only for its own display layer). Run it first if
you want the rest of the toolkit to honour the distinction.

Two different consequences follow, and the difference matters:

* a **span** — three or more consecutive points — is an invented piece of vessel.
  `connect` cuts it out so the break becomes two real free ends, and `surface`
  refuses to mesh it.
* a **lone** flagged point is an invented *radius* on a real point. It is kept out
  of every measurement, and nothing is cut.

---

### `gaps` — fill jumps inside a single edge

```
python -m hipct_seg_debug.edit gaps GRAPH.am --out fixed.am
```

| flag | default | meaning |
|---|---|---|
| `--min-gap-um N` | 500 | absolute floor for what counts as a gap |
| `--big-jump-ratio N` | 5 | …and it must also exceed this many vessel widths |
| `--out PATH` | — | write the repaired graph |

**Both** thresholds must be exceeded. The relative one is what stops normal wide
sampling on a thin vessel being mistaken for a gap.

---

### `connect` — bridge disconnected ends

```
python -m hipct_seg_debug.edit connect GRAPH.am --tjunction --out fixed.am
```

| flag | default | meaning |
|---|---|---|
| `--tjunction` | off | also attach free ends onto the *side* of other vessels |
| `--same-component` | off | allow joins inside one component (creates loops) |
| `--cone-deg N` | 50 | the target must lie within this angle of the vessel's heading |
| `--reach-factor N` | 15 | max span, in multiples of the endpoint radius |
| `--radius-ratio N` | 5 | max thick:thin ratio across a join |
| `--tortuosity N` | 1.8 | max path length / straight-line distance |
| `--show-rejected` | off | list every refused candidate, not just a tally |
| `--keep-interpolation` | off | leave Avizo's fills in place instead of cutting them out first |
| `--dpc --raw DIR --seg PATH` | off | run sequential Type 1, 2 and 3 image-guided walks |
| `--cfc-model DIR` | — | use a persisted DF21 raw-patch classifier (exclusive with `--dpc-learned`) |
| `--dpc-omega N` | 5 | probability weight; only P is neighbourhood-normalised |
| `--dpc-distance-weight N` / `--dpc-cosine-weight N` | 1 / 1 | D and C ablation weights |
| `--dpc-neighbourhood {two-level,full}` | `two-level` | paper coarse/fine walk or legacy full 5-cube |
| `--dpc-min-probability N` | 0.15 | reject a completed walk whose mean centreline probability is lower |
| `--dpc-pad-factor N` | library default | raw ROI padding around a proposal, in source radii |
| `--dpc-bright-lumen` | off | use for contrast-enhanced scans; native HiP-CT assumes a darker lumen |
| `--out PATH` | — | write the reconnected graph |
| `--geodesic --seg PATH [--raw DIR]` | off | collapse-aware connector; repairs the mask too (exclusive with `--dpc`) |
| `--out-seg PATH` | — | write the repaired segmentation; **required** whenever `--geodesic` is given `--out` |
| `--review-json PATH` | — | write the candidates needing a human decision |
| `--decisions-json PATH` | — | write the full record; if it exists, operator rulings in it are applied first |
| `--max-unsupported-factor N` | 4 | reject a route with more than N local radii of contiguous unsupported path |
| `--alternatives N` | 3 | how many distinct routes to look for; the second decides ambiguity |
| `--no-tjunction-geodesic` | off | with `--geodesic`, propose end-to-end joins only |

Loosening `--cone-deg` and `--reach-factor` is the usual way to get candidates out
of a graph that yields none. **Use `--show-rejected` first** — it names the gate
that refused each one, so you widen the right threshold instead of guessing.

`--out` writes even when nothing was accepted, so the output is always a complete
graph you can chain onwards. On the stock Avizo graph, the default gates accept
nothing — it is already connected apart from two far-apart components.

**On a graph that has been through `flag-interpolation`, every fill is removed
before anything is proposed.** A fill that is still in place hides its own break, so
every gate downstream answers the wrong question. Cutting it makes the two sides
genuine free ends and lets the image decide. Afterwards, any fill whose two sides are
still separate goes back exactly as Avizo wrote it — still flagged, so it stays out
of every measurement — and any fill whose sides are joined again is left out. Nothing
is lost either way. `--keep-interpolation` opts out.

#### `--geodesic` — repair the graph and the mask together

```
python -m hipct_seg_debug.edit connect GRAPH.am --geodesic `
  --seg SEG.am --raw RAW\ `
  --out fixed.am --out-seg fixed-seg.am `
  --review-json review.json --decisions-json decisions.json
```

Opt-in and mutually exclusive with `--dpc`. Four things behave differently:

1. **Paired outputs.** `--out` and `--out-seg` are required together — a graph
   written beside the original mask is two files that disagree about where the
   vessels are. Passing one alone exits with that message rather than writing.
2. **Classification first.** Two free ends already inside one mask component are
   re-skeletonised and **add no voxels**; only genuinely separate components get a
   path search. The run prints the split.
3. **Ambiguity is an outcome.** A second route within 15% of the best sends the
   candidate to `--review-json` instead of being resolved by a threshold. So does
   an endpoint that does not sit on the mask, and — without `--raw` — any mask gap
   wider than two segmentation voxels.
4. **Provenance is written.** Every created edge carries `ReconnectionOrigin`
   (0 original, 1 geometry, 2 DPC, 3 geodesic, 4 re-skeletonised), `RouteScore`
   and `RouteReviewed`, and they survive the `.am` round trip.

Dry run is still the default: with neither output path, everything is proposed,
searched and decided and nothing is written. That plus `--review-json` is the
useful first run.

The review loop: run with `--decisions-json decisions.json`, adjudicate in the
GUI's **Reconnect** tab (or edit the file — set `decision.operator` to
`{"accept": true}`, optionally with `"waypoints_um"`), then re-run with the same
`--decisions-json` and the rulings apply. Matching is by endpoint, not by list
position, so a re-run on a graph that has moved on reports the unmatched rulings
rather than applying them to the wrong candidate.

Writing `--out-seg` re-encodes the whole lattice, which is the honest cost of
`HxByteRLE`: it is one sequential stream, so a byte written near the start shifts
everything after it.

One case cannot be helped: where the fill was an entire edge between two junctions,
removing it leaves both at degree two rather than degree one, and every proposer here
starts from a free end. The split is still reported, and the component count still
tells you the tree was never whole.

With `--dpc`, endpoint candidates are explicitly routed through Type 1 then Type 2,
and `--tjunction` adds Type 3. Accepted bridges are applied between stages so every
later proposal sees the updated topology. A CFC run uses raw image intensities only
for its probability term; the segmentation is not consulted across the gap.

The three types are unambiguous in the result metadata:

- **Type 1:** endpoint-to-endpoint between disconnected non-backbone components;
- **Type 2:** disconnected component endpoint to a backbone endpoint;
- **Type 3:** disconnected endpoint to the interior of a target vessel polyline.

Their active DPC reach limits are respectively 8, 15, and 15 times the source radius;
they are no longer inert configuration fields. The ordinary proposal gate still uses
`--reach-factor` before image-guided refinement.

The corrected score is `D + omega*P + C`, defaulting to `(1, 5, 1)`. `D` is negative
Euclidean distance to the target, or negative minimum distance to the whole target
polyline for Type 3. Only `P` is min-max normalised over the current neighbourhood.
`C` is included when the previous two directions have cosine similarity `<= 0.5` and
omitted when they are already closely aligned. Type 3 candidates must be compatible
with both the latest direction and the sum of the latest two directions.

The default two-level walk uses offsets at Euclidean distance 2 through 3 while the
target is more than three raw voxels away, then offsets below distance 2 near the
target. `--dpc-neighbourhood full` selects the previous full `5x5x5` behaviour for
comparison. Both modes reject visited points and turns of 90 degrees or more.

Completed paths carry probability and raw-greyscale sequences containing the walk plus
five voxels from the known connected tail and five from the target head. Validation
records ADF p-values and reports two independent decisions: `paper_validation` for the
probability/greyscale stationarity checks, and `hipct_safeguards` for the conservative
mean, relative drop, trough, and greyscale-continuity checks. `--show-rejected` exposes
the responsible gate.

### Train and evaluate the Cascade Forest

DF21's official Windows wheel is limited to the dedicated Python 3.9 environment;
do not install it into the Python 3.12 viewer environment. Create the pinned
environment once:

```powershell
# from the checkout
conda env create --prefix .conda-cfc --file environment-cfc.yml
```

The examples below run from any directory, provided the package is installed.

#### `train-cfc` — train one scan-specific DF21 model

```text
python -m hipct_seg_debug.edit train-cfc GRAPH.am --seg SEG.am --raw RAW_DIR
    --out-model MODEL_DIR [--max-positive N] [--seed N] [--overwrite]
```

| flag | default | meaning |
|---|---|---|
| `--seg PATH` | `$HIPCT_SEG` | sampling mask and geometry source |
| `--labels-field NAME` | `Labels` | binary lattice field |
| `--raw DIR` | required | authoritative raw image stack used for every feature |
| `--out-model DIR` | required | artifact directory; must be empty unless `--overwrite` |
| `--voxel-um N` | derived | the true raw voxel size; given, it is the authority and rescales every length ([Voxel size](#voxel-size)) |
| `--max-positive N` | all | deterministic positive cap for a smoke test; omit for full training |
| `--seed N` | `0` | sampling, grouped split, and DF21 random seed |
| `--overwrite` | off | replace only known artifact files in an existing directory |

The LADAF-2024-28 full-skeleton command is:

```powershell
& '.conda-cfc\python.exe' -u -m hipct_seg_debug.edit train-cfc `
  'D:\data\candidate.am' `
  --seg 'D:\data\segmentation.am' `
  --raw 'D:\data\raw_slices' `
  --out-model 'models\LADAF-2024-28-cfc' `
  --seed 0
```

There is deliberately no `--max-positive` in that command. A smoke run can add, for
example, `--max-positive 100`, but its model is not a full-skeleton result.

Before sampling, training checks positive x/y/z spacing, dimensions and bounds, that
the segmentation crop fits the raw stack, that graph points fit the raw stack, that
complete patches exist, and that the segmentation-to-raw binning is `2x2x2`.
Coordinates are transformed through the Amira bounding box rather than by assuming
matching array indices. Run the viewer's stronger axis-flip/transpose check with
`--validate-only` before training a new scan.

Each input row is a rigid 686-value raw-intensity feature:

```text
15x15x15 raw patch
  -> first 14x14x14
  -> non-overlapping 2x2x2 max pool
  -> 7x7x7, C-order (z,y,x), 343 values

centred raw 7x7x7
  -> C-order (z,y,x), 343 values

concatenate pooled-large then small; no per-patch normalisation
```

The classes are `N` unique centreline voxels, `2N` lumen foreground negatives away
from the centreline, and `2N` outside-lumen negatives no farther than seven
segmentation voxels from vessel foreground. The seed makes sampling reproducible.
Validation holds out whole skeleton-component groups, falling back to 64-raw-voxel
spatial groups when necessary, then reports accuracy, sensitivity, specificity,
balanced accuracy, ROC-AUC, PR-AUC, and the confusion counts. The deployable model is
refitted on all samples with:

```text
n_bins=255, max_layers=20, n_estimators=2, n_trees=100,
n_tolerant_rounds=2, random_state=SEED, n_jobs=-1, partial_mode=True
```

The artifact is self-describing:

| path | contents |
|---|---|
| `model/` | native DF21 model |
| `manifest.json` | feature contract, geometry, sample counts, DF21 settings, source fingerprints, seed, runtime |
| `metrics.json` | grouped held-out metrics and validation groups |
| `samples.npz` | raw `(z,y,x)` voxel IDs, binary labels, three-class sampling IDs, groups |

`CfcProbability` loads the model once per connection job, reads raw patches lazily,
and caches repeated candidate coordinates. It refuses a different artifact version,
feature width/order, raw stack shape, or voxel spacing.

#### Run CFC-guided DPC directly

```powershell
# from the checkout
New-Item -ItemType Directory -Force 'runs\manual-cfc' | Out-Null

& '.conda-cfc\python.exe' -u -m hipct_seg_debug.edit connect `
  'D:\data\candidate.am' `
  --tjunction --dpc --show-rejected `
  --seg 'D:\data\segmentation.am' `
  --raw 'D:\data\raw_slices' `
  --cfc-model 'models\LADAF-2024-28-cfc' `
  --out 'runs\manual-cfc\connected.am'
```

`--cfc-model` requires `--dpc` and is mutually exclusive with `--dpc-learned`.
The persisted model supplies probabilities from raw patches; the current segmentation
is still used for geometry/ROI context, not as a substitute signal inside a collapsed
gap. The default is the paper weight `--dpc-omega 5` and corrected two-level
neighbourhood. Set `--dpc-neighbourhood full` only for the legacy-neighbourhood
comparison.

#### `export-dpc-regions` — make blinded review material

```powershell
& '.conda-cfc\python.exe' -u -m hipct_seg_debug.edit export-dpc-regions `
  'D:\data\candidate.am' `
  --seg 'D:\data\segmentation.am' `
  --raw 'D:\data\raw_slices' `
  --regions 'src\hipct_seg_debug\edit\reconnect\dpc_eval_regions.json' `
  --output 'dpc_review_regions'
```

The version-1 manifest contains nine initial real-data regions: two plausible
candidates and one hard negative for each reconnection type. The command writes one
six-view PNG per case with orthogonal minimum/maximum projections, segmentation
contours, source/target markers, and the proposed bridge, plus `review_template.csv`.
These geometric roles are review hints, not truth.

| case | role | source -> target | inclusive raw ROI `(z,y,x)` |
|---|---|---|---|
| T1-A | candidate | node 575 -> node 583 | `[3488,1552,1389]` to `[3535,1591,1420]` |
| T1-B | candidate | node 740 -> node 751 | `[3784,2303,425]` to `[3827,2335,458]` |
| T1-N | hard negative; target faces away | node 1116 -> node 1124 | `[4332,1436,2357]` to `[4369,1483,2402]` |
| T2-A | candidate | node 977 -> node 983 | `[4218,1702,1361]` to `[4253,1763,1400]` |
| T2-B | candidate; weak segmented support | node 328 -> node 341 | `[3034,2420,1057]` to `[3081,2465,1098]` |
| T2-N | hard negative; endpoint faces away | node 1171 -> node 1174 | `[4422,2230,2753]` to `[4463,2273,2786]` |
| T3-A | candidate | node 1171 -> segment 1150 | `[4422,2230,2755]` to `[4469,2275,2788]` |
| T3-B | candidate; no segmented core | node 576 -> segment 587 | `[3490,778,2161]` to `[3535,811,2206]` |
| T3-N | hard negative; outside direction cone | node 621 -> segment 569 | `[3514,886,1737]` to `[3583,945,1776]` |

After blinded raw-image review, record `adjudication.status`, `reviewer`,
`correct_type`, `correct_target`, and paper labels in the JSON manifest. Valid labels
are `TP_b`, `TN_b`, `FP_b`, `FN_b`, `TP_s`, and `FP_s`. `paper_labels` is a mapping:
use `"all"` for one label shared by all runs, or a run key such as `"omega-5"` or
`"ablation-DP-omega-5"` when the label differs by result.

#### `evaluate-dpc` — sweep one global probability weight

```powershell
& '.conda-cfc\python.exe' -u -m hipct_seg_debug.edit evaluate-dpc `
  'D:\data\candidate.am' `
  --seg 'D:\data\segmentation.am' `
  --raw 'D:\data\raw_slices' `
  --cfc-model 'models\LADAF-2024-28-cfc' `
  --regions 'src\hipct_seg_debug\edit\reconnect\dpc_eval_regions.json' `
  --omega 0:7 `
  --output 'runs\dpc-evaluation.json'
```

| flag | default | meaning |
|---|---|---|
| `--regions PATH` | bundled version-1 manifest | reviewed Type 1/2/3 cases |
| `--omega SPEC` | `0:7` | inclusive range (`0:7`) or comma-separated integers (`0,3,5`) |
| `--cases IDS` | all | comma-separated case IDs for a targeted diagnostic run |
| `--output PATH` | `dpc-evaluation.json` | complete JSON report |
| `--no-ablations` | off | skip DP, PC, and DC runs at the selected weight |

Every run records case/type/target correctness, completion and acceptance, rejection
source, steps, runtime, path, D/P/C scores, probability and greyscale sequences, both
ADF p-values, paper validation, and HiP-CT safeguards. Metrics are computed exactly as:

```text
RecAcc = ((TP_b + TP_s) + TN_b) /
         ((TP_b + TP_s) + TN_b + (FP_b + FP_s) + FN_b)
RecSen = (TP_b + TP_s) / ((TP_b + TP_s) + FN_b)
RecSpe = TN_b / (TN_b + (FP_b + FP_s))
```

One global omega is selected by pooled `RecAcc`, then `RecSpe`, then fewer failed
paths. The report includes overall and per-type summaries, whether the paper default
five was confirmed, and DP/PC/DC ablations at the selected weight. It never invents
truth from geometry: if no reviewed paper labels exist, `selected_omega` falls back to
five (when included), `selection_status` says `default pending expert labels`, and
`paper_default_confirmed` is null.

---

### `repair-radius` — restore collapsed radii from the local taper

```
python -m hipct_seg_debug.edit repair-radius GRAPH.am --source both --out fixed.am
```

| flag | default | meaning |
|---|---|---|
| `--source {outlier,image,both}` | `both` | how to find collapsed spans |
| `--factor N` | 0.6 | a span is collapsed below this fraction of the local trend |
| `--margin N` | 6 | points at each segment end never treated as collapsed |
| `--max-taper-per-mm N` | 0.5 | clamp on the fitted taper |
| `--allow-decrease` | off | let the fit *lower* an over-inflated span, not just raise a collapsed one |
| `--seg PATH` `--labels-field` `--voxel-um` `--edits` | | needed by `--source image` / `both` |
| `--out PATH` | — | write the repaired graph |

`--source image` measures cross-sections against the lattice and is slow (minutes);
`--source outlier` needs no image at all. High-radius spans are always reported using
the healthy/lower half of each segment's log-radius residuals as the baseline. They are
left unchanged unless `--allow-decrease` is supplied.

Run graph and taper repair **before** `radius-perimeter`; perimeter measurement is the
last radius-changing step. If `repair-radius` is run on a graph carrying the
`radius_source` field, image-collapse detection is skipped with an explicit message:
the perimeter pass has already compensated for collapsed shape. Graph-only low/high
auditing remains available. Overlapping or adjacent image, low and high spans are
merged before fitting so a bad high cannot anchor a neighbouring repair. One-sided
extrapolation is capped at 1.05 times its trusted healthy-window maximum.

**When no spans are found this writes no file, even with `--out`** — it prints
"no repairable spans found" and stops rather than emitting an identical copy. So an
absent output here means "nothing needed fixing", not "it failed". (`connect` takes
the opposite view and always writes; the two differ.)

---

### `surface` — regenerate the lumen mesh

```
python -m hipct_seg_debug.edit surface GRAPH.am --out-dir surfaces/
```

| flag | default | meaning |
|---|---|---|
| `--out-dir DIR` | `surface` | where to write `lumen_bspline.stl` |
| `--voxel-mm N` | derived | pin the SDF voxel size, **mm** |
| `--keep-interpolation` | off | mesh Avizo's fills too, instead of cutting them out |

**This one always writes** — it has no dry run, because producing the mesh *is* the
command. There is no `--out`; the mesh goes to `--out-dir`, as both
`lumen_bspline.stl` and `lumen_bspline.vtk`.

Flagged fills are *cut*, not masked, for a specific reason: leaving a hole inside an
edge would be closed straight back up by `bridge_centerline_gaps`, which
`sdfpatch.preprocess_graph` runs before anything is meshed. Splitting removes the hole
and the edge together, so a capsule is never swept along an invented centreline.

Measured on the stock Avizo graph at the derived 0.146 mm voxel: **1,275,058
vertices / 6,769,610 triangles**, a 127 MB STL, pipeline time **87.8 s** (SDF
evaluation 20 s, iso-surface 49 s). A finer `--voxel-mm` costs sharply more in both
time and memory — the grid here is already 648 × 528 × 589.

---

### `skeletonise` — derive a centreline from the mask

```
python -m hipct_seg_debug.edit skeletonise --order --out candidate.am
```

| flag | default | meaning |
|---|---|---|
| `--order` | off | also assign Strahler order and topological generation |
| `--pick-roots` | off | open the 3-D picker once the skeleton is finished and root each tree by hand |
| `--style` `--color-by` | `tube`, `strahler` | with `--pick-roots`: how the segments are drawn (`contour`/`tube`/`lines`) and coloured |
| `--roots-json PATH` | — | with `--pick-roots`, write the chosen roots here |
| `--out PATH` | — | write the generated graph |
| `--per-tree` | off | skeletonise each mask component separately, tagging every edge with its `tree` index |
| `--min-component-voxels N` | 0 | with `--per-tree`, ignore components smaller than this |
| `--max-trees N` | — | with `--per-tree`, keep only the N largest |
| *plus the six shared segmentation flags* | | |

Lee thinning, radii from the distance transform, then a trace into a graph with
junction voxels clustered into single nodes. **~12 minutes at stride 1** on the full
lattice; use `--stride 4` for a fast sanity run.

The generated skeleton is a *candidate*, not a replacement — score it before
trusting it.

`--per-tree` labels the mask 26-connected and skeletonises each component inside its
own bounding box, so the left and right coronary trees carry an identity everything
downstream can read. The index comes from the *mask* labelling (largest is 0), not
from the graph, and is not portable across masks or strides. Note the skeleton itself
shifts slightly — thinning inside a tight box sees a different neighbourhood.
`--min-component-voxels` and `--max-trees` are refused without it rather than
silently implying it. See [SKELETONISATION.md](SKELETONISATION.md) §6.1.

---

### `optimise` — order, clean radii, and score against a reference

```
python -m hipct_seg_debug.edit optimise candidate.am --sensitivity
```

| flag | default | meaning |
|---|---|---|
| `--reference PATH` | the Avizo graph | score against this |
| `--oblique` | off | also re-measure flagged radii from oblique cross-sections |
| `--sensitivity` | off | report centreline sensitivity — needs the lattice decoded (2.34 GB at stride 1) |
| `--bb-threshold N` | 900.0 | bifurcation match distance, µm |
| `--out PATH` | — | write the ordered, radius-corrected graph |
| *plus the six shared segmentation flags* | | |

`--seg`, `--oblique` or `--sensitivity` each force the lattice to be decoded;
without any of them the command is graph-only and fast.

`--roots-json` roots the ordering from the sidecar `pick-roots` wrote; without it
`auto_roots` guesses, which is what it did unconditionally before. Any component the
sidecar does not cover is still filled in automatically, so no tree is left at order 0.

Ordering runs after the lattice is opened when there is one, because the sidecar snaps
a recorded root to the nearest node within a few *voxels* and only the frame knows how
big a voxel is. It still runs before `--oblique`, which is what needs the
Strahler field to pick which segments to re-measure. Calling
`optimise.correct_radii` directly on an *unordered* graph re-measures every segment
instead and says so.

The number that discriminates two skeletons over one mask is the **bifurcation
Dice**. `super_metric` compares two *segmentations*, so its volume, component and
Euler terms are identical for both candidates and contribute nothing. The four
commands below score a skeleton properly instead — see
[`score`](#score--the-five-super-metric-terms).

---

### `skeletonise-all` — run several algorithms and score each

```
python -m hipct_seg_debug.edit skeletonise-all --stride 4 --out-dir cand/
python -m hipct_seg_debug.edit skeletonise-all --algorithms lee,teasar,amira \
    --amira-graph avizo.am --out-dir cand/
```

| flag | default | meaning |
|---|---|---|
| `--algorithms LIST` | `lee` | comma-separated: `lee`, `teasar`, `amira` |
| `--amira-graph PATH` | the Avizo graph | existing `.am` to score as the `amira` candidate |
| `--teasar-scale N` | 2.5 | TEASAR penalty scale — Centerline Tree's `slope` |
| `--teasar-const-um N` | 300.0 | TEASAR fixed penalty, µm — its `zeroVal` |
| `--bb-threshold N` | 900.0 | bifurcation match distance, µm |
| `--no-tree-chi` | off | score χ against the mask's loops, not the tree ideal |
| `--no-score` | off | write candidates without the dense image topology/super-metric pass |
| `--pick-roots` | off | open the 3-D picker once the skeleton is finished and root each tree by hand |
| `--style` `--color-by` | `tube`, `strahler` | with `--pick-roots`: how the segments are drawn (`contour`/`tube`/`lines`) and coloured |
| `--roots-json PATH` | — | with `--pick-roots`, write the chosen roots here |
| `--out-dir DIR` | `skeletons` | where every candidate `.am` goes |
| *plus the six shared segmentation flags* | | |

**This one always writes**, unlike the proposers above. It spends up to 25 minutes at
stride 1 deriving skeletons, and a blank flag costing all of that is not a safe default —
`surface` sets the same precedent. The absolute destination is printed on completion, so
the Log tab tells you where the files went.

Prints the five super-metric terms per algorithm and ranks them, lowest `M_S` first.
Reading the *terms* matters as much as the total: a candidate dominated by its `V`
term has a radius-estimator problem, one dominated by `cl` has put its centreline
outside the lumen. [SKELETONISATION.md](SKELETONISATION.md) explains all five.

`teasar` needs `pip install kimimaro`; without it that candidate is skipped with a
message and the others still run. `amira` runs no algorithm — it ingests an existing
Avizo export so the commercial AutoSkeleton/Centerline Tree result can be scored on
the same terms as the ones that can be scripted.

---

### `optimise-skeleton` — de-loop, prune, smooth, re-centre

```
python -m hipct_seg_debug.edit optimise-skeleton candidate.am --out refined.am
python -m hipct_seg_debug.edit optimise-skeleton candidate.am \
    --sweep "prune-factor=1,2,3;smooth-um=0,100,200"
```

`candidate.am` is the spatial graph. `--seg`, when supplied or required for
re-centring/scoring, must be the binary Amira **Lattice** containing `define Lattice`;
passing another spatial graph produces the expected “no define Lattice” error.

| flag | default | meaning |
|---|---|---|
| `--no-deloop` | off | keep cycles; by default every one is broken |
| `--no-prune` | off | keep short leaves |
| `--no-recentre` | off | leave the centreline where it is |
| `--prune-factor N` | 2.0 | drop a leaf shorter than this many *local vessel radii* |
| `--prune-radius-ratio N` | 0.8 | never prune a leaf this thick relative to its parent |
| `--smoother NAME` | `gaussian` | `none`, `gaussian`, `savgol`, `bspline`, `multiscale` |
| `--smooth-um N` | data-derived | smoothing window, `--smoother gaussian` only; `0` disables |
| `--drift-radius-factor N` | 0.25 | trust region in local radii, `--smoother multiscale` only |
| `--recentre-passes N` | 2 | re-centring rounds |
| `--recentre-tangent-radii N` | 2.0 | length of the chord the cut plane's normal is taken over, in local radii |
| `--recentre-damping N` | 0.5 | fraction of the way to the centroid moved per pass |
| `--recentre-max-move-frac N` | 0.5 | furthest a point may end up from where re-centring started, in its own radii |
| `--recentre-grow-radii N` | 4.0 | how far the cut window may grow chasing a section that reaches its edge |
| `--recentre-blob NAME` | `blob8` | connectivity the centroid is taken over: `blob4` or `blob8` |
| `--sweep SPEC` | — | score a grid instead of writing |
| `--roots-json PATH` | — | roots sidecar from `pick-roots`: one chosen root per tree. An explicit `--root-edge` wins for its own component. With `--pick-roots`, written rather than read |
| `--pick-roots` | off | open the 3-D picker once the skeleton is finished and root each tree by hand |
| `--style` `--color-by` | `tube`, `strahler` | with `--pick-roots`: how the segments are drawn (`contour`/`tube`/`lines`) and coloured |
| `--scope` | `whole` | what `M_S` is computed over: `whole`, `per-tree`, or `largest` |
| `--min-component-voxels N` | 0 | with a restricted scope, ignore components smaller than this |
| `--max-trees N` | — | with `--scope per-tree`, score only the N largest |
| `--objective` | `weighted` | how per-tree scores combine when ranking a sweep |
| `--lhs N` | 0 | Latin-hypercube samples across each swept *numeric* range |
| `--out PATH` | — | write the refined graph |
| *plus the six shared segmentation flags* | | |

For segmentation-constrained geometry fitting without pruning or de-looping, use
`refine-centreline`. The optional `prepare-reconstruction` command then addresses
circular-vessel collisions using smooth displacement fields and fixed measured
radii. Both commands write diagnostic reports and remain opt-in; see
[CENTRELINE_REFINEMENT.md](CENTRELINE_REFINEMENT.md) for the workflow and limits.

Runs **de-loop → prune → re-centre × N → smooth**. Topology before geometry, so the
smoother is never asked to smooth a segment about to be deleted — and smoothing goes
*last* because `multiscale` certifies its output as it returns (κr ≤ 0.95, no new branch
contact, drift within the trust region), and re-centring afterwards would void all three.

`savgol`, `bspline` and `multiscale` come from `coronary_sdf` and need it importable;
without it they fail with the path it looked in, and `gaussian` still works.
[SKELETONISATION.md](SKELETONISATION.md) has the measured ranking — at stride 4 on
LADAF-2024-28 no smoother beat `none`, and `multiscale` at its default drift cap was the
worst of the five. Sweep before trusting one:

```
--sweep "smoother=none,gaussian,savgol,bspline,multiscale"
```

*De-looping is a prior about the anatomy.* Coronary arteries at this calibre are a
tree, so a cycle is either a vessel that collapsed in the middle and was segmented as
two, or a segmentation that ran into its neighbour. The thinnest edge of each cycle
goes, and every break is reported with its radius and position.

*Pruning uses the local largest radius*, measured on the thickest other branch at the
junction and read *away* from it — the distance transform at a carina is the distance
out of the whole junction, not either vessel's radius. A leaf at the lattice boundary
is never pruned: that is a vessel the scan cut off.

*Re-centring is bounded five ways*, because unbounded it was the one stage that made the
skeleton worse — on LADAF-2024-28 at stride 1 it scattered 4.41% of points into
out-and-back spikes against 0.00% in the Lee skeleton it was given. The
`--recentre-tangent-radii` chord is the important one: a plane whose normal comes from
the two adjacent points of a voxel staircase rotates 19.5° between neighbours, so
neighbours cut differently tilted sections and land on unrelated centroids. Every run now
prints a **roughness** line — the share of points turning more than 120°, which is the
number that catches this — and it should stay near zero:

```
roughness: 0.03% of points turn >120 deg, step median 87.4 um max 493.8 um, 3016.6 mm total
```

`--sweep` writes nothing. It scores each parameter set with the super metric, prints a
ranked table *and* a second table ranked by roughness, and you re-run the winning flags
with `--out`. Both tables matter: the super metric barely moves on this defect — 4.41% of
points doubling back costs about 0.005 of cl-sensitivity — so `M_S` alone would call the
broken result a tie. Sweep terms are named after the flags, minus the leading dashes, and
a term nothing reads is refused before the volume is decoded rather than minutes later.

---

### `radius-perimeter` — replace every radius with its own cross-section

```
python -m hipct_seg_debug.edit radius-perimeter refined.am --seg labels.am --out radius.am
```

| flag | default | meaning |
|---|---|---|
| `--gate-voxels N` | 3.0 | below this radius in voxels use `sqrt(area/π)` instead |
| `--max-half N` | 128 | largest sampling-window half-width in voxels; a section must close inside the window to be accepted. Increase this for larger vessels if cuts remain truncated |
| `--stability-centroid-mode NAME` | `drift` | check centroid movement across adjacent sections; `offset` restores the legacy distance-from-centreline check |
| `--workers N` | 1 | parallel processes for cross-section measurement; full branch context is retained, and final calibration/filtering runs once on the combined measurements |
| `--max-radius-factor N` | 2.0 | reject a radius above this multiple of its robust local surrounding radius |
| `--no-branch-aware` | off | restore legacy single-plane selection; disable slab validation and 3D ownership |
| `--root-edge EDGE_ID` | inferred | root one component for parent/daughter inference; repeat for multiple components |
| `--roots-json PATH` | — | roots sidecar from `pick-roots`: one chosen root per tree. An explicit `--root-edge` wins for its own component |
| `--tangent-search-deg N` | 20 | bounded correction searched when the fitted tangent is unstable |
| `--no-ownership-near-junctions` | on | restore the old behaviour, where local 3-D branch ownership was skipped at any point with a topologically-adjacent branch nearby. Near a junction one always is, so the watershed never ran and no `unresolved branch overlap` was raised at exactly the places two lumens fuse: the merged blob was measured and accepted. On LADAF-2021-17 ownership resolved ~100 of 336,865 points |
| `--grow-radii N` | off | ceiling on how far one window may double, in multiples of that point's own radius. Off, a window grows to `--max-half` whenever the blob touches its border — which for a plane that is not transverse is *always*, since a streak along the vessel touches at any width. So raising `--max-half` to reach the widest vessels taxes every narrow one, which then gets rejected anyway: on a 1,729-point subgraph, 64 → 256 was 35× slower. Pair `--max-half 192 --grow-radii 4` to buy the wide sections without that. The Sections tab's `grow ceiling` is the same number; set it there first to see which windows it binds on |
| `--junction-mask-radii N` | off | second ceiling on a junction run, in multiples of the node's own input radius, applied alongside `--junction-mask-max-fraction` so the tighter wins. A junction reaches a few parent radii into each branch wherever it sits, so a bound stated as a fraction of the segment over-masks a short branch. **Off by default: measured on LADAF-2021-17 it frees a third of all points from the mask and none of them became a measurement — they came back `truncated` or `unstable`, so the mask was covering points that were already unmeasurable** |
| `--junction-mask-max-fraction N` | 0.4 | most of a segment's own arclength one junction may consume, per end |
| `--transverse-axis-ratio N` | 1.10 | how elliptical a *stable* section may be before it is re-cut over the cone anyway. The stability test is blind to tilt; raise this to disable the re-cut (cheaper, and over-reads every oblique cut) |
| `--carina-tip-factor N` | 0.1 | daughter support radius at the node relative to its first trusted radius |
| `--out PATH` | — | write the re-measured graph |
| *plus the six shared segmentation flags* | | |

Leave `--grow-radii` unset when investigating severely underestimated input radii:
its window cap is derived from those input values and can prevent the true lumen
from closing, even when `--max-half` is large enough.

Radius measurement checks centroid **movement across the slab**, allowing a consistent
section whose centreline is off-centre. The legacy offset check could reject a valid
large lumen when the stored input radius was too small. Sustained corrections are
also checked against surrounding measured radii rather than capped by that same
incorrect input radius. Isolated spikes still fail the local radius gate. This
changes radius estimation; it does not move the centreline or guarantee a circular
reconstruction will follow a flattened lumen boundary.

For parallel measurement, add `--workers 8` (or another process count). Each process
keeps the full graph context and its own lattice cache. Confidence filtering,
junction profiles and fallbacks use the combined measurements, so splitting the work
does not change the calibration. Use `--workers 1` for live mask edits or multiple
measurement passes. On a 4,136-point LADAF-2021-17 sample, eight workers reduced the
measurement from 64.3 s to 16.9 s; all exported radii and provenance fields were
identical. Small graphs may gain little because worker startup has a fixed cost.

**This is what fixes the thickness values.** `adjust_thickness.py` measures
`r = arcLength/(2π)` at every point and then throws the measurements away, keeping
only a global fit `r ≈ slope·(Amira DT thickness) + intercept`. Over 11,094
cross-sections that fit has median ratio 0.96 but correlation only ~0.66 and a p5–p95
spread of 0.49–2.98 — locally it can be out by a factor of two. This command keeps the
measurement made *at each point*.

The estimator is hybrid and says which it used. `perimeter/(2π)` is inflated by the
staircase boundary when a section is only a voxel or two across, so below
`--gate-voxels` the area estimator is used instead, and a per-point `radius_source`
field (`0` perimeter, `1` area, `2` filled) is written into the `.am` beside the
thickness. `MeanRadius` is re-derived on **every** edge, which is only sound because
every point was re-measured.

Measurements pass geometric and statistical confidence gates before interpolation.
The normal comes from an edge-local, radius-scaled quadratic fit. Its section and two
nearby parallel sections must close, contain the centreline and agree in area and
perimeter within 50%; an unstable normal gets a bounded nine-direction search. Sections
that never close remain rejected.

**Those three metrics cannot see tilt.** The slab is stepped along the candidate
normal, so on a straight vessel of constant calibre an oblique plane cuts three
identical ellipses: area and perimeter agree to 1.000 and the cut is pronounced
stable while its perimeter — and so its radius — is too long by `1 / cos(tilt)`,
up to a quarter at 40 degrees. That is a smooth bias, not a spike, so it also
passes the runaway and local-factor gates below. What tilt does change is the
section's *shape*: an oblique cut of a tube of radius `r` at angle `t` is an
ellipse of semi-axes `r` and `r / cos t`. So any section flatter than
`--transverse-axis-ratio` (default 1.10, about 25 degrees) is re-cut over the same
nine-direction cone whether or not it was stable, and the shortest boundary wins —
ties going to the fitted tangent. Minimising the perimeter is right for a collapsed
lumen too, since tilting away from transverse can only lengthen the boundary, which
is why one gate serves both and why `adjust_thickness.py` resliced for minimum area
in the first place. A round section short-circuits the search as before, so the
common case costs what it always did. A section that grows beyond four input radii is also rejected
when its measured/input correction exceeds three times the robust segment trend. That
trend is fitted in log space and re-fitted on the lower residual half so high values
cannot establish their own baseline. Independently, the local-factor gate uses a
two-pass median over an arclength neighbourhood scaled to the local input radius; first-
pass highs are removed before the local baseline is recomputed. Rejected runs are then
interpolated only from the surrounding trusted radii. If an entire segment is
unmeasurable, its input radii are retained.

A plane is not made branch-specific by making it larger. If a non-adjacent segmented
vessel is connected to the target component, a local 3D marker watershed assigns the
foreground to the competing centrelines and reconstructs their missing separating wall.
Disconnected companion vessels remain outside the selected component. This ownership is
never used between topology-adjacent parent and daughter branches because it would invent
a wall inside a real shared lumen.

A carina has no unique circular cross-section, so its radii are inferred rather than
measured. The junction run grows out from the node until two consecutive stable sections
no longer intersect an adjacent branch; this covers a side-branch centreline that remains
inside and initially parallel to its parent. Components are rooted by `--root-edge`, then
Strahler order, free-end status and calibre. The parent calibre continues through the
shared region, while each daughter grows monotonically from a small support radius at its
duplicated node point to its first trusted post-ostial measurement. GUI preview and full
surface export preserve these authored tapers instead of replacing them with generic SDF
radius smoothing.

The audit field `radius_reject_reason` records `0` accepted, `1` unmeasurable,
`2` truncated, `3` junction, `4` sampling-growth runaway, `5` local-factor outlier,
`6` unstable tangent/cross-section, and `7` unresolved branch overlap.
`radius_resolution_mode` additionally records direct, 3D-owned, interpolated,
parent-through, daughter-emergence, or retained-input values. The command prints counts
by reason and resolution mode plus before/after radius quantiles and maxima. There is
deliberately no global micrometre cap: a uniformly large vessel is valid when it agrees
with its own neighbourhood.

**`--stride` is inherited from the shared flags but does not apply here.** The radius is
the quantity being fixed, so it is always measured against the full-resolution lattice —
decimating it would defeat the purpose. That costs nothing in memory: the sections are
cut by streaming **row bands** out of the RLE lattice, never by decoding the volume.

A cross-section is a few tens of voxels across, so decoding the plane it sits in --
10 MB and 5.6 ms on a 3400x2964 mask -- throws away 98% of the work. Bands are 43x
cheaper (0.13 ms for 64 rows), cached by bytes rather than by count, and evicted
least-recently-used because one point tries up to nine candidate tangents over three
slab offsets through the same few bands. Measured on the LADAF-2021-17 left tree,
788 segments: **438 s to 17.5 s over the first 609 points, a 25x speedup, with every
measured radius bit-identical** -- this is an I/O change, not a change of method.
The remaining cost is the plane sampling itself, and it is concentrated: the 12% of
sections whose window grew past half=32 take 46% of the time, so `--max-half 64`
trades a shorter measurement for the widest sections against roughly another quarter
off the clock.

Verified on LADAF-2024-28, measured against the same lattice — `r_stored / r_perimeter`
as p5 / median / p95:

| skeleton | before (EDT radii) | after | within 1.5× |
|---|---|---|---|
| stride 8 | 0.55 / 1.16 / 2.76 | **1.00 / 1.00 / 1.00** | 57.4% → 99.9% |
| stride 1 | 0.22 / 0.71 / 1.03 | **0.93 / 1.00 / 1.14** | 57.1% → 99.7% |

`perimeter_mismatch` becomes true by construction rather than something to audit for. The
residual spread at stride 1 is the hybrid gate doing its job, not an error: 24.3% of points
fell below `--gate-voxels` and took `sqrt(area/π)`, while the audit always measures pure
perimeter, so those points are *expected* to differ. `radius_source` records which is which.

---

### Diagnostics — why is this radius what it is?

Four read-only commands that explain a written graph rather than editing one. They
take no `--out`. All four appear in the GUI's Commands panel, because that panel is
generated from this parser.

```
python -m hipct_seg_debug.edit segment-diagnosis radius.am --segment 265 --segment 268
python -m hipct_seg_debug.edit junction-mask     refined.am --segment 265
python -m hipct_seg_debug.edit reformat-radius   refined.am --segment 265
python -m hipct_seg_debug.edit ostium-flare      radius.am
```

* **`segment-diagnosis`** — per point: the radius written, its `radius_source` and
  `radius_reject_reason` read back out of the `.am`, and the section re-cut to give
  voxel count, minor and major axis, `blob8/blob4`, and both estimators. This is what
  distinguishes "the vessel is narrow here" from "the pass could not measure it".
* **`junction-mask`** — per point: `stable`, `adjacent_overlap`, and whether the
  section was ever *exclusive*, plus how many points `_adaptive_junction_mask` takes.
  **Give it the graph the measurement consumed**, not the one it wrote:
  `_BranchContext.rivals` scales its search by the radii in the file, so running it on
  the output asks a different question using the radius under suspicion. The command
  says so when it starts.
* **`reformat-radius`** — measures the named segments again on `reformat`'s
  parallel-transport planes. `cut` shares its plane construction with the pass being
  audited, so agreement between them is weaker evidence than it looks; this is
  genuinely independent. Runs at `mode="native"`, one pixel per segmentation voxel, so
  both see the same digitisation.
* **`ostium-flare`** — tree-wide: how much wider the section becomes approaching a
  branched node, and whether the flare is round. **Each segment is normalised by its
  own interior**; binning raw distance-to-node across the tree compares near-node
  proximal sections against far-from-node distal ones and reports a flare twice the
  real size with the aspect ratio moving the wrong way.

Cut with a bounded window in all of them: unbounded, a plane that is not truly
perpendicular doubles out to `max_half` and encloses a streak *along* the vessel —
thousands of voxels and a major axis of 100+, which is not a section at all.

---

### `crop` — cut the tree down to the vessels of interest

```
python -m hipct_seg_debug.edit crop radius.am --crop-json crop.json --out cropped.am
```

| flag | default | meaning |
|---|---|---|
| `--crop-json PATH` | — | the crop sidecar: the named main vessels, the rule, and the selection it produced. Read first if it exists, and rewritten every run — **this file is the artefact; the `.am` is derived from it** |
| `--min-strahler N` | off | drop every branch below this Strahler order |
| `--min-ostium-um X` | off | drop every branch whose take-off radius is below this. Needs no main vessels |
| `--ratio F` | off | drop a side branch below this fraction of the ostial radius of the main vessel it descends from |
| `--ratio-denominator N` | off | the same threshold as `1/N` — `coronary_sdf`'s `--ratios`, one value per run |
| `--prune-unattributed` | off | also drop subtrees that descend from no named main vessel |
| `--takeoff-factor F` | 2.0 | local radii to walk from a junction before measuring a branch |
| `--root-edge N` | — | force a component's root, deciding which end is proximal; repeatable |
| `--roots-json PATH` | — | roots sidecar from `pick-roots`: one chosen root per tree. An explicit `--root-edge` wins for its own component |
| `--no-reorder` | off | write the pre-crop Strahler orders instead of recomputing them |
| `--replay` | off | drop exactly what the sidecar recorded, re-evaluating nothing |
| `--report-csv PATH` | — | one row per dropped take-off |

**Every rule drops a branch *and everything downstream of it*.** A kept twig hanging off
a dropped parent is not a smaller tree, it is a broken one. Main vessels themselves, and
the path from the root down to one, are never dropped — a rule that removed an
unannotated left main would take the LAD with it.

The three rules compose in one run and their take-offs are unioned. `--min-strahler`
refuses on a graph carrying no orders rather than dropping everything, and `--ratio`
refuses with no main vessel named, since it would have no denominator.

**Main vessels are named in the Crop tab** ([Part 7](#part-7--the-control-panel)), under
a name of your choosing — segment by segment, or by tracing: `trace between two picks`
takes the ostium and the far end and adds the whole path between them, which is a simple
path by construction (no backtracking, no loop) or a refusal when the two picks are in
different trees. `prefer thick` decides the route where a fused cycle offers two, sending
the trace down the vessel instead of over the artefact joining it to its neighbour. They are identified
*geometrically* — endpoint and mid-point coordinates at 1 µm, hashed — not by edge
number, because every write renumbers edges and a sidecar keyed by number would silently
name different vessels after one `gaps` run. A vessel key that no longer resolves is a
hard error; a hand-marked branch that no longer resolves is a note.

Strahler orders are **recomputed** after the crop by default: the stored ones describe
the tree as it was, and every branch removed changes the order of everything proximal to
it. The recomputation uses `--roots-json`, re-resolved against the *cropped* graph —
the sidecar keys on world coordinates, so the root survives the segments it was recorded
on being deleted, which is exactly what a crop does. Without it the re-ordering would
answer with a different root than the crop was planned against, and the file's crop and
its `strahler` column would disagree about which end of the tree is proximal. That needs
`skeleton_analysis`; if it is unavailable the run stops without writing and points you at
`--no-reorder`.

One pass, and no degree-2 contraction — the nodes left mid-vessel by a removed branch are
counted and reported, and `optimise-skeleton` is what contracts them.

`--takeoff-factor` exists because a radius read *at* a junction measures the whole carina
rather than either vessel. `coronary_sdf` skips a fixed number of contours where this
walks a distance, so the radii here will not match its `pruned_branches.csv`.

---

### `score` — the five super-metric terms

```
python -m hipct_seg_debug.edit score radius.am
```

| flag | default | meaning |
|---|---|---|
| `--bb-threshold N` | 900.0 | bifurcation match distance, µm |
| `--no-tree-chi` | off | score χ against the mask's loops (comparable to the paper) |
| `--scope` | `whole` | what `M_S` is computed over: `whole`, `per-tree`, or `largest` |
| `--min-component-voxels N` | 0 | with a restricted scope, ignore components smaller than this |
| `--max-trees N` | — | with `--scope per-tree`, score only the N largest |
| `--objective` | `weighted` | how per-tree scores combine: `weighted` by voxels, `mean`, or `sum` |
| *plus the six shared segmentation flags* | | |

Writes nothing. Prints `V`, `cc`, `χ`, `cl` and `B` and their weighted sum `M_S`,
lower being better. Because de-looping deliberately moves the graph away from the
segmentation's Euler characteristic, χ is scored against the tree ideal by default;
`--no-tree-chi` gives a number comparable with the published values.

By default `χ` is measured on the **largest component alone**, on both the graph and
the image side, so on a two-tree mask a change confined to the right coronary tree
does not move that term at all. `--scope` chooses what to do about it:

| `--scope` | what is scored |
|---|---|
| `whole` (default) | the published definition: `V`/`cc`/`cl`/`B` over the whole graph, `χ` over the largest component alone |
| `per-tree` | each mask component against its own gold standard, one row per tree, combined by `--objective` |
| `largest` | the largest component only — the same restriction `χ` already had, now applied to **all five** terms |

`largest` is `per-tree` capped at one component, so its numbers match that scope's
`tree 0` exactly. The whole-graph table is always printed first, because the other two
define `cc` and `χ` differently and the three must not be quoted against each other.
See [SKELETONISATION.md](SKELETONISATION.md) §6.1.

---

### `pick-roots` — click each tree's inlet

```
python -m hipct_seg_debug.edit pick-roots lee.am --roots-json roots.json
python -m hipct_seg_debug.edit pick-roots left.am right.am --seg regions.am     --roots-json roots.json
```

Takes **several skeletons**. The mask is decoded and split once and shared by all of
them, so a left-tree graph and a right-tree graph are numbered against one labelling
and land in one sidecar. `--out` applies only when a single graph is given.

**`--seg` is optional, and the picker never reads it.** The 3-D window is built from
the graph alone and the root is recorded by world coordinate, so the roots you get
with and without it are identical. It does exactly one thing: it anchors the *tree
numbering* to the mask labelling — and to `Left_Tree` / `Right_Tree` where the mask
names them — instead of to the graph's own component order. Without it the sidecar
records `tree_source: "graph"` and numbers components largest-first.

It is not free: passing it decodes the lattice and runs a connected-component
labelling, 3 s against 73 s at `--stride 4`, and tens of minutes at `--stride 1` on a
whole heart. Reach for it when you are rooting **several** skeletons at once (it is
what makes the left graph tree 0 and the right graph tree 1 by themselves), or when
you want the tree names in the sidecar. For a single graph, skip it.

| flag | default | meaning |
|---|---|---|
| `--roots-json PATH` | — | write the chosen roots here; read first if it exists |
| `--out PATH` | — | also write the graph with Strahler orders taken from those roots |
| `--auto` | off | skip the window and record the automatic roots |
| `--color-by` | `strahler` | colour segments by Strahler order, or `none` for flat grey |
| `--style` | `contour` | `contour` rings plus centreline dots, `tube` one solid tube per segment, or `lines` — the graph edges as bare polylines with a point at each node |
| `--n-sides N` | 16 | ring tessellation for `--style contour` |
| `--ring-stride N` | 1 | draw a ring every N centreline points |
| `--seg PATH` | — | anchor the tree indices to the mask labelling rather than component size |
| `--labels-field` `--voxel-um` `--stride` `--edits` | | the usual lattice flags, read only when `--seg` is given |
| `--ignore-materials` | off | split by connectivity alone, ignoring a `Materials` block that already names the trees |
| `--min-component-voxels N` | 0 | with `--seg`, ignore components smaller than this |
| `--screenshot PATH` | — | render the **first** tree to a PNG instead of opening a window |
| `--preselect N` | — | with `--screenshot`, the edge index to draw as chosen |

Opens one 3-D window per tree, largest first. Left-click the inlet segment (it turns
green), `q` confirms and moves to the next tree, `x` stops, `c` clears. The root is
the clicked segment's degree-1 end, so a trunk's free end wins over its junction.

**`--style lines` draws the graph itself** — one polyline per edge, a point at each
node, no radius consulted and nothing swept. It is by far the lightest of the three: a
3,722-segment coronary tree is 3,722 polylines rather than 3,722 tubes, so the scene
orbits at once instead of redrawing in steps. Use it when you are looking for a trunk
on a whole tree; `tube` and `contour` show calibre, which matters when the inlet is
ambiguous and you want to see which branch is thickest.

The sidecar records each root's world coordinate first and its ids only as a note, so
it survives a re-skeletonisation; `--roots-json` then replaces hand-typed
`--root-edge` values on `optimise-skeleton`, `radius-perimeter` and `crop`. A root
that cannot be placed on a later graph earns a note and falls back to the automatic
pick rather than failing. Without PyVista and matplotlib the command says so and
records the automatic roots, so it is safe in a headless pipeline.

---

### `repair-mask` — cull debris and close small breaks

```
python -m hipct_seg_debug.edit repair-mask --min-voxels 500 --close 1 --out mask.tif
```

| flag | default | meaning |
|---|---|---|
| `--min-voxels N` | 0 | drop connected components smaller than this |
| `--keep-largest N` | 2 | components this large are protected regardless of `--min-voxels` |
| `--close N` | 0 | radius in voxels for morphological closing; 0 to skip |
| `--out PATH` | — | write the repaired mask (`.tif`, scaled to 0/255) |
| *plus the six shared segmentation flags* | | |

Prints before/after morphometrics. **Read component count and Euler number as a
pair**: a closing that mends a real break lowers the component count and leaves
Euler roughly alone; one that welds two unrelated vessels lowers both.

Closing is indiscriminate — keep `--close` at 1 or 2 and use painting for anything
longer.

---

### `mask-export` — write the corrected mask back out

```
python -m hipct_seg_debug.edit mask-export --edits work/edits.npz --out corrected.am
python -m hipct_seg_debug.edit mask-export --edits work/edits.npz --out corrected.tif
```

| flag | default | meaning |
|---|---|---|
| `--out PATH` | **required** | `.am` → `HxByteRLE` Amira lattice; `.tif` → uint8 stack |
| `--field NAME` | `Labels` | field name to write into the `.am` |
| *plus the six shared segmentation flags* | | |

Label values are **preserved**, not scaled, so an `.am` written here is a drop-in
replacement for the source lattice. (`repair-mask --out` scales to 0/255, because
what it writes is a thresholded boolean.)

Measured on LADAF-2024-28: decoding all 2.34 GB and re-encoding gives **37,725,961
bytes where Avizo wrote 37,709,212** — 0.04% larger, byte-identical when read back,
1.8 s.

---

# Part 3 — workflows

## 1. First contact — does this dataset hang together?

```
python -m hipct_seg_debug --validate-only          # ~15 s
python -m hipct_seg_debug --selftest               # ~2 min
```

`--validate-only` proves the raw stack, graph, lattice and STL share one coordinate
frame, and exits non-zero if not. `--selftest` goes further: 21 checks against the
greyscale itself, including that the mask width matches 2r, that the vessel wall
ring sits just outside the assigned radius, and that the lumen is darker than the
tissue around it.

Run both on any new dataset before trusting anything else.

## 2. Audit — what is wrong, and where?

```
python -m hipct_seg_debug                          # writes cache/candidates.csv
python -m hipct_seg_debug --no-crosssection        # graph heuristics only, much faster
```

249 sites on LADAF-2024-28 in 14.6 s. The CSV has one row per site with its kind,
score, world position and — for image-detected collapses — the point range on its
segment. Sort by score; walk them with `n` / `b` in the 3D window.

## 3. Inspect one site, headlessly

```
python -m hipct_seg_debug --goto-candidate 1
python -m hipct_seg_debug --goto-slice 2727 --roi 0        # whole slice
python -m hipct_seg_debug --goto-slice 2727 --slab 20      # 41 slices deep
```

Opens one napari window on that location and blocks. Good for a screenshot or a
second opinion on a single site.

## 4. Interactive audit

```
python -m hipct_seg_debug --volume
```

3D pick window and slice browser side by side. Double-click a vessel, press `v` for
slices. `--volume` adds whole-dataset layers so the z slider can leave the slab —
worth it for asking whether a defect continues past the crop.

## 5. Correct the skeleton by hand

```
python -m hipct_seg_debug --edit
python -m hipct_seg_debug --edit --edit-box-um 8000        # rebuild a bigger box
```

Press `e` to enable, then edit with the keys in Part 4. Each edit reports the box it
touched and queues a rebuild of only that box, off the GUI thread — 0.5–0.8 s rather
than the 5–10 minutes a full run costs. Export from the edit panel.

## 6. Correct the mask by hand

```
python -m hipct_seg_debug --edit --paint --edits work/edits.npz
```

Select the `segmentation (editable)` layer, `2` to paint, `4` to erase, then
**Commit**, then **Re-skeletonise painted region**. The graph gains a centreline
through what you painted and the surface follows.

Corrections go to `work/edits.npz` and never into the source lattice. Feed that file
to anything downstream with `--edits`.

## 7. Derive an independent skeleton and score it

```
python -m hipct_seg_debug.edit skeletonise --stride 4 --out quick.am     # 10 s, a sanity run
python -m hipct_seg_debug.edit skeletonise --order --out candidate.am    # ~12 min, the real one
python -m hipct_seg_debug.edit optimise candidate.am --sensitivity
```

`--stride` is worth knowing here, because the cost is steeply non-linear — and so is
the answer, since decimation thins exactly the small vessels a skeletoniser is meant
to find:

| stride | time | segments |
|---|---|---|
| 8 | 4 s | 205 |
| 4 | 10 s | 597 |
| 1 | ~12 min | 1226 |

Use a strided run to check the command works and the geometry looks right; use
stride 1 for anything you intend to keep.

Measured against Avizo on LADAF-2024-28:

```
reference    309 seg   311 nodes   2 comp  150 bifs  2933 mm  r 247 um  sens 0.911
candidate   1226 seg  1254 nodes  52 comp  526 bifs  3238 mm  r 162 um  sens 0.941
bifurcation Dice 0.425  (tp 144, fp 383, fn 6)
```

96% recall on Avizo's branch points, but 383 invented ones. Read that as "it finds
the same vessels and over-branches", not "it is better because sensitivity is
higher".

## 8. Repair a graph, end to end

```
python -m hipct_seg_debug.edit report        AVIZO.am
python -m hipct_seg_debug.edit gaps          AVIZO.am --out step1.am
python -m hipct_seg_debug.edit connect       step1.am --tjunction --show-rejected --out step2.am
python -m hipct_seg_debug.edit repair-radius step2.am --source both --out step3.am
```

Drop `--out` from any of them to see the plan without committing to it. Inspect
between steps with `report`.

### 8C. Repair a graph with CFC-guided DPC

Train the model once as described in
[Train and evaluate the Cascade Forest](#train-and-evaluate-the-cascade-forest), then
run the graph repair, using Python 3.9 for its DF21 connection step:

```powershell
# from the checkout
New-Item -ItemType Directory -Force 'runs\LADAF-2024-28-cfc-full' | Out-Null

py -3.12 -u -m hipct_seg_debug.edit gaps `
  'D:\data\candidate.am' `
  --out 'runs\LADAF-2024-28-cfc-full\step1-cfc.am'

& '.conda-cfc\python.exe' -u -m hipct_seg_debug.edit connect `
  'runs\LADAF-2024-28-cfc-full\step1-cfc.am' `
  --tjunction --dpc --show-rejected `
  --seg 'D:\data\segmentation.am' `
  --raw 'D:\data\raw_slices' `
  --cfc-model 'models\LADAF-2024-28-cfc' `
  --out 'runs\LADAF-2024-28-cfc-full\step2-cfc.am'

py -3.12 -u -m hipct_seg_debug.edit repair-radius `
  'runs\LADAF-2024-28-cfc-full\step2-cfc.am' `
  --source both `
  --seg 'D:\data\segmentation.am' `
  --out 'runs\LADAF-2024-28-cfc-full\step3-cfc.am'
```

The GUI workflow with the same name builds these commands for you. `step2-cfc.am` is
the graph immediately after sequential Type 1 -> Type 2 -> Type 3 DPC; `step3-cfc.am`
also contains radius repair. Neither replaces the source graph.

## 9. Repair a mask, end to end

```
python -m hipct_seg_debug.edit repair-mask  --min-voxels 500 --close 1
python -m hipct_seg_debug.edit repair-mask  --min-voxels 500 --close 1 --out mask.tif
python -m hipct_seg_debug.edit mask-export  --edits work/edits.npz --out corrected.am
```

The first line is the dry run — read the morphometrics before writing anything.

## 10. Regenerate the surface

```
python -m hipct_seg_debug.edit surface step3.am --out-dir surfaces/
```

5–10 minutes. The exported STL always comes from a full run, never from accumulated
preview patches: patches are spliced for display and are not welded at their seams,
which is fine to look at and not fine to mesh.

## 11. The full run

```
# 1. prove the inputs
python -m hipct_seg_debug --validate-only

# 2. an independent opinion on the skeleton
python -m hipct_seg_debug.edit skeletonise --order --out candidate.am
python -m hipct_seg_debug.edit optimise candidate.am --sensitivity

# 3. repair the working graph
python -m hipct_seg_debug.edit gaps          AVIZO.am --out step1.am
python -m hipct_seg_debug.edit connect       step1.am --tjunction --out step2.am
python -m hipct_seg_debug.edit repair-radius step2.am --source both --out step3.am

# 4. and the mask
python -m hipct_seg_debug.edit repair-mask --min-voxels 500 --close 1 --out mask.tif

# 5. fix by hand what no heuristic should be trusted with
python -m hipct_seg_debug --edit --paint --edits work/edits.npz

# 6. take the corrected mask back out
python -m hipct_seg_debug.edit mask-export --edits work/edits.npz --out corrected.am

# 7. the surface for CFD
python -m hipct_seg_debug.edit surface step3.am --out-dir surfaces/
```

### 11C. Full run with CFC-guided DPC

Workflow **11C** keeps workflow 11 intact and substitutes the CFC version of graph
repair. It uses the dedicated Python 3.9 executable only for the CFC `connect` step,
then pauses after reconnection so the proposed bridges can be reviewed before mask
export and surface generation. Its repaired graph is `step3-cfc.am` and its final
surface directory is `surfaces-cfc/` inside the selected run directory.

## 12. Choose a skeletonisation, then measure it

The [Walsh–Berg procedure](SKELETONISATION.md): there is no ground-truth skeleton, so
the segmentation is the gold standard and the algorithm is chosen by how little of it
the skeletonisation lost.

```
# 1. run the algorithms and rank them on the five-term super metric (strided, ~1 min)
python -m hipct_seg_debug.edit skeletonise-all --stride 4 --out-dir work/candidates

# 2. the winner at full resolution (~12 min)
python -m hipct_seg_debug.edit skeletonise --out work/candidate.am

# 3. break the collapse loops, drop leaves shorter than the local vessel radius,
#    smooth, and re-centre on the lumen
python -m hipct_seg_debug.edit optimise-skeleton work/candidate.am --seg labels.am --out work/refined.am

# 4. after all graph/taper repair, replace each thickness from its cross-section;
#    values above 2x their robust local neighbourhood are interpolated by default
python -m hipct_seg_debug.edit radius-perimeter work/refined.am --seg labels.am --out work/radius.am

# 5. confirm the terms improved
python -m hipct_seg_debug.edit score work/radius.am
```

Step 4 is the one that fixes the thickness values. Step 1 is worth running once per
dataset even if you keep Lee thinning: the ranked table says *which* term each
algorithm is losing on, which is the paper's main practical use for the metric.
Do not run taper or geometry repair after step 4. If post-perimeter repair is
unavoidable, image collapse detection is skipped and high correction is opt-in via
`--allow-decrease`.

To tune the refinement rather than accept the defaults, sweep it:

```
python -m hipct_seg_debug.edit optimise-skeleton work/candidate.am \
    --sweep "prune-factor=1,2,3;smooth-um=0,100,200"
```

## 13. Tests

```
python -m pytest                # 460 tests, ~33 s
python -m pytest --runslow      # + real pipeline runs, minutes
python -m hipct_seg_debug --selftest                          # 21 checks, ~2 min
python -m compileall -q hipct_seg_debug/
```

The fast tests need no dataset. Run the self-test as well after touching
`viewer2d.py` or `volume.py` — they are shared with the read-only auditing path.

---

# Part 4 — interactive keys

## 3D pick window

| key | action |
|---|---|
| double-click | pick the nearest centreline point |
| drag | rotate — never picks |
| `v` | open / update the slice viewer |
| `i` | raw image plane at the pick |
| `g` | segmentation mask around the pick |
| `a` | segmentation mask, whole tree — full resolution, built once and kept |
| `n` / `b` | next / previous candidate |
| `s` | allow / forbid picking the surface |
| `L` | show / hide the legend box *(shift+l)* |
| `c` | clear the pick |
| `r` | reset the camera *(VTK's own, not rebindable)* |
| `q` | close the window *(VTK's own)* |

The docked **layers** panel carries a visibility checkbox and an opacity slider per
layer, plus `save figure...`, which writes the view to SVG (or PDF / EPS / PS / TeX)
without the keybinding block and the pick readout but *with* the legend box and the
colour bar — the Strahler bands are unreadable without the bar that names them. The
geometry lands as an embedded raster image and the overlay labels as vector text; that
is GL2PS's OpenGL2 backend, not a setting. One of the layer rows is worth knowing about
before you see it:

**`interpolated (Avizo)`** draws the spans Avizo invented in flat grey, and the
viridis-by-radius centreline is *broken* either side of them, so the real skeleton
visibly stops where the real skeleton stops. Grey is deliberate — it is absent from
viridis, so it cannot be misread as a radius, which these points do not have. Those
points are also left out of the radius-circle layer and out of the colour range, so
one point pinned to a calibration intercept cannot flatten the contrast across the
whole tree. The row is greyed out on a graph with nothing invented in it.

This is the one place in the toolkit that detects on its own: the viewer opens
whatever file it is handed, and if the graph has not been through
`flag-interpolation` it says so in the status line rather than showing nothing. It is
display-only — no pipeline reads it. Picking is unaffected either way; a picked
vertex id is still an index straight into the graph's own point order, which
`selftest.test_centreline_point_ids` checks on the real dataset.

## Edit mode (`--edit`)

Off until you press `e`, so a stray keypress over the render window cannot change
the graph.

| key | action |
|---|---|
| `e` | toggle edit mode |
| `z` / `y` | undo / redo |
| `d` | delete the picked segment |
| `t` | delete the picked branch and everything past it |
| `x` | split the segment at the picked point |
| `k` / `j` | widen / narrow the picked segment (×1.1) |
| `f` | fill the collapsed radii around the pick |
| `u` | rebuild the surface around the pick |

A docked **edit** panel carries the same operations as buttons, plus a radius factor
and a *rebuild after every edit* toggle — turn that off to make several edits in one
region and rebuild once.

## Slice browser (napari)

| key | action |
|---|---|
| `2` | paint |
| `3` | fill |
| `4` | erase |
| `[` / `]` | brush size |
| `Ctrl+Z` / `Ctrl+Shift+Z` | undo / redo the brush |

These are napari's own `Labels` bindings and only act on the selected layer — select
**`segmentation (editable)`** first. The paint dock adds buttons rather than more
keys, because napari already owns `1`–`5`, `[`, `]` and `Ctrl+Z`, and a binding that
silently loses to one of those is worse than no binding:

**Commit** · **Revert this box** · **Save edits…** · **Re-skeletonise painted
region** · a `add`/`replace` splice selector · a 3-D brush toggle.

Layer visibility and opacity are napari's own layer list, on the left.

---

# Part 5 — measured runtimes

LADAF-2024-28: raw 4752 × 3079 × 3154 at 32.99 µm, mask 1500 × 1250 × 1250 at
65.98 µm, graph 309 segments / 29,122 points.

| operation | time |
|---|---|
| load all four inputs | 3.0 s |
| `--validate-only`, total | 15 s |
| candidate detection (249 sites) | 14.6 s |
| `--selftest`, total | ~2 min |
| decode the whole lattice (2.34 GB) | 2.2 s |
| whole-tree isosurface, stride 1 — first press / after | 6.9 s / 4.7 s |
| whole-tree isosurface, stride 4 (mask resident) | 0.2 s |
| re-encode it to `.am` (37.7 MB) | 1.8 s |
| open a 192³ paint box | 0.2 s |
| commit painted edits | ~5 ms |
| re-skeletonise a painted gap (`add`) | 0.7 s |
| re-skeletonise a 3 mm box (`replace`) | 1.5 s |
| SDF session prepare (once, at startup) | ~1 s |
| local patch rebuild after an edit | 0.5–0.8 s |
| `skeletonise`, stride 8 / 4 / 1 | 4 s / 10 s / ~12 min |
| `surface` on the Avizo graph (6.77 M triangles) | 88 s |
| `report` from the panel, in-process / subprocess | 0.5 s / 0.7 s |
| swap the dataset (control panel) | 2.9 s |
| **Reload graph** alone (control panel) | 0.06 s |

Subprocess overhead is 0.2 s, not the seconds a cold `python -m` costs — the child
imports `argparse` and little else, because every handler imports its own
dependencies inside its body.

The whole design follows from the last row against the two above it: a full surface
run is minutes, the same code over a 10 mm box is sub-second, so every edit reports
the box it touched and only that box is rebuilt. `--voxel-mm` moves the surface
figure sharply — 0.146 mm is what the graph's own minimum radius derives.

---

# Part 6 — troubleshooting

**`ModuleNotFoundError: No module named 'hipct_seg_debug'`** — the package is not
installed in the interpreter you are running. From the checkout, `python -m pip install -e .`

**`ImportError: cannot find the 'coronary_sdf' package`** — only `--edit`,
`surface` and the re-skeletonise button need it. Set `HIPCT_CORONARY_SDF` to the
directory containing it. Painting, skeletonising, scoring and exporting do not need
it at all.

**`no 'define Lattice nx ny nz'` / parse errors on a graph** — the graph is binary
`.am`. Re-export as ASCII from Avizo.

**"it printed a plan and wrote nothing"** — that is the design. Add `--out`.

**`edits are for a 1500x1250x1250 lattice, this one is …`** — the `.npz` was made
against a different segmentation. This check exists because applying it anyway would
scatter corrections into unrelated tissue and look entirely plausible.

**`unrecognized arguments: --candidates`** — that flag was removed; it never did
anything. Detection runs by default, and `--no-candidates` turns it off.

**Several PyVista windows open and everything freezes** — something called
`coronary_sdf` outside `sdfconfig.sdf_config`. `DEBUG_VIS` and `DEBUG_VIS_BLOCK`
both default to `True` there and open about six *blocking* windows.

**A pick takes 4–6 s** — that is the raw TIFF read, and it is expected. The second
pick in the same region is fast; the readers cache decoded slices.

**The Run button is greyed out** — an input path does not exist, or a required flag
is empty. The reason is printed just above the button.

**A command wrote a file but the viewer still shows the old one** — by design. The
Log says which file, and Data → **Load result** picks it up in one click (or paste the
path and press **Reload graph**). Nothing reloads on its own, because doing so
mid-inspection would also discard any skeleton edits.

---

# Part 7 — the control panel

The 3D window carries a `control` dock — tabbed alongside `layers` and, in edit mode,
`edit`. It is always there; there is no flag to enable it. Everything in Parts 1–3 is
reachable from it, and unlike the command line it does not have to reload 2.34 GB of
lattice between two commands.

For workflows 8C and 11C, start the Python 3.12 GUI with the persisted model and the
dedicated Python 3.9 executable:

```powershell
py -3.12 -m hipct_seg_debug `
  --raw 'D:\data\raw_slices' `
  --graph 'D:\data\candidate.am' `
  --seg 'D:\data\segmentation.am' `
  --cfc-model 'models\LADAF-2024-28-cfc' `
  --cfc-python '.conda-cfc\python.exe' `
  --no-surface
```

`--no-surface` is optional. `--cfc-model` and `--cfc-python` are workflow inputs, not
entries in the Data tab. They may instead be supplied as `HIPCT_CFC_MODEL` and
`HIPCT_CFC_PYTHON`. The workflow launches only its DF21 `connect` step with Python 3.9;
the viewer and the remaining repair steps retain Python 3.12.

## Data

The five inputs, with a Browse button each and a drop-down of paths you have used
before (kept in `cache/gui_recent.json` — the only settings file in the package).

**skeleton (.am) takes several.** Its Browse dialog multi-selects and *appends*, the
field holds them `;`-separated, and `--graph` is variadic on the command line:

```
python -m hipct_seg_debug --graph left_tree.am right_tree.am --seg regions.am --raw ...
```

They are **merged into one graph, each source held as its own tree** — not drawn
beside it as scenery. That is what makes them editable: picking, the edit handles,
`crop`, `reformat` and the writer are every one of them defined against a single
graph, so a second graph next to it would have to be read-only to stay honest. Merged,
all of them are editable, and the `tree` field keeps them apart — the same field
`--per-tree` skeletonisation, per-tree scoring and the roots sidecar already use. A
source that already carries trees keeps them, shifted past the ones before it, so a
per-tree skeleton loaded beside a single-tree one gives three trees rather than two.
A single path is not merged at all, so it still round trips through the writer exactly
as before.

**Reload graph** takes the whole selection too, so swapping which skeletons are loaded
still costs 0.4 s rather than a full reload.

A segmentation that names its trees (an Avizo `.Regions.am` with `Left_Tree` and
`Right_Tree`) is contoured **once per material** in the `segmentation (whole tree)`
layer, each in Avizo's own colour, instead of one fused surface over `mask > 0`. Two
coronaries that touch are then two objects on screen rather than one. A plain binary
mask is unchanged.

| button | what it does | cost |
|---|---|---|
| **Load all** | rebuilds the whole session from the five paths | 2.9 s |
| **Reload graph** | re-reads the skeleton(s) only, keeping the frame, the lattice and the warm slice caches | 0.06 s |
| **Validate** | the eight coordinate-frame checks, non-fatally | ~12 s |
| **Load result** | the same reload, on the `.am` the last command wrote | 0.06 s |

**Reload graph is the one to use between repair steps.** `gaps` → `connect` →
`repair-radius` changes nothing but the graph, and a full reload there would throw
away the decoded slice caches and the whole-tree mask mesh for no reason.

**Load result** is that same reload without the copy-paste. It is greyed out until a
command writes a graph, then names the file — *Load refined.am* — and opens it. It is
an offer and never fires by itself: a graph you are inspecting may hold unsaved
skeleton edits, so a file already open in the viewer is only ever mentioned in the Log.

The inputs are not symmetric, and the panel treats them accordingly: the raw folder
is a directory, a missing surface is a warning rather than an error, and the edits
`.npz` is as often a file painting is about to *create* as one to open.

**Unsaved work blocks a swap.** If you have painted edits with no `--edits` path, or
skeleton edits you have not exported, Load asks before discarding them. Nothing in
the package persists either automatically.

## Commands

One form per subcommand, **generated from argparse** — the same definitions Part 2
documents. Each flag's own help text is its tooltip, its default is prefilled, and
the command line that will run is shown underneath with a Copy button.

The CFC command forms expose and validate every argument, but standalone `train-cfc`,
`evaluate-dpc`, `export-dpc-regions`, and CFC-backed `connect` jobs require the Python
3.9 interpreter. Use **Copy** and run those lines with `.conda-cfc\python.exe`, or use
workflow 8C/11C for CFC-backed connection; those workflows apply `--cfc-python`
automatically.

Three things are worth knowing:

- **Only what you changed is passed.** A flag left alone is omitted, not sent at its
  default. That matters because 26 of the 62 flags default to `None` meaning "keep
  the library's own default", and `--cone-deg 0.0` is not the same as saying nothing.
  Tick *show every flag* for a fully explicit line to paste into a script.
- **Numeric fields have an unset position** at the far left, shown as
  `(library default)`. That is what "don't pass this flag" looks like in a spin box.
- **Run is disabled while an input does not exist**, rather than letting argparse
  fail later in a subprocess.

**Where a command runs** depends on what it costs:

| | runs | Stop |
|---|---|---|
| `report`, `gaps`, `connect`, `repair-radius --source outlier` | in this process, on a thread | clears the queue; the running command finishes |
| everything else, including `repair-radius` at its default | as `python -u -m hipct_seg_debug.edit …` | kills the child |

`repair-radius` is split by its flags, not its name: with `--source image` or `both`
it runs the image detector over the whole graph, which is minutes.

The Stop button is honest about the difference. Nothing in `find_sites`,
`decode_volume` or `fill_spans` has a cancellation point, so an in-process job cannot
be interrupted — Stop clears the queue and marks the result cancelled, which stops a
chain, but the current command runs to completion.

Jobs run one at a time, in order. Nothing is coalesced away: each writes files the
next may read.

## Workflows

Workflows 7–11C run as chains. Intermediates go to one directory per run
(`cache/runs/<timestamp>`), and each step's output feeds the next by name rather than
by scraping the log.

A chain stops at the first step that fails **or that returns 0 without writing what
it said it would**. That second case is real: `repair-radius` writes nothing when it
finds no collapsed spans, and a chain that trusted the exit code would fail three
minutes later with a confusing "file not found". Instead it stops and says so,
naming the graph that is still current.

Two chains pause deliberately — workflow 9 after its dry run so you can read the
morphometrics, and workflow 11 at the hand-correction step. Press **Continue**.

The CFC variants are deliberately additive; workflows 8 and 11 remain unchanged:

| workflow | result |
|---|---|
| **8C. Repair a graph with CFC-guided DPC** | `step1-cfc.am` -> `step2-cfc.am` -> `step3-cfc.am` |
| **11C. Full run with CFC-guided DPC** | the complete chain, a review pause, then `surfaces-cfc/` |

8C runs `gaps`, then CFC-backed `connect --tjunction --dpc --show-rejected`, then
`repair-radius --source both`. The connection step performs Type 1, Type 2, and Type 3
in order and updates graph topology between stages. Use a fresh run directory because
intermediate outputs are intended to be inspectable and reproducible, not silently
overwritten.

When the chain finishes, set the Data tab's graph field to `step2-cfc.am` to inspect
the reconnections alone or `step3-cfc.am` for the radius-repaired result, then press
**Reload graph**. The viewer never swaps graphs automatically because that would
discard unexported manual edits.

Workflows 1–6 are not chains; they are "open the viewer and look". Those buttons
configure the session and print what to do next, and are listed separately so it is
clear which is which. **Audit** is worth singling out: its candidates go straight
into the 3D view, so `n` and `b` walk them — which a session started with
`--no-candidates` could not do at all before.

## Sections — why a radius came out the way it did

`radius-perimeter` measures each point's radius on a square of segmentation cut
perpendicular to the centreline. It reports the radius and a reason code; it does not
report the square. When the corrections come back wrong the square is usually why, and
there are only three ways for it to be wrong: it is in the **wrong place**, it is the
**wrong size**, or it is at the **wrong angle**. This tab re-cuts it at sampled points
and draws it.

Select segments with `5` (or leave the list empty for the whole tree), then `7` to cut.
`6` clears the selection, `9` cancels a half-finished trace.

| control | what it does |
|---|---|
| trace between two picks | Add pick twice and every segment on the path between the two joins the selection, so a whole vessel is two clicks rather than forty. The same `crop.trace_path` the Crop tab uses — a Dijkstra over nodes, so the route is a simple path. Unlike a plain pick it is a **union**: a second trace adds to the selection rather than toggling off the overlap |
| prefer thick | weight that route by each segment's own radius instead of arclength, so it goes down the vessel rather than over a thin false bridge where the segmentation fused two crossing vessels. On a tree the path is unique and this changes nothing |
| every N points | sections are cut every N points of each segment; **both ends are always cut**, because that is where the junction mask bites |
| max sections | hard cap per run, so a whole tree cannot cost what the measurement pass costs |
| max half (vox) | largest half-width the window may grow to — `radius-perimeter --max-half`. A vessel needs `2.5 x` its radius in voxels; short of that every section reaches the border and is refused |
| start half (vox) | force the starting window instead of `2.5 x` the stored radius. **This is the direct test of "the planes are not big enough"**: raise it and see whether orange windows turn violet and their amber rings close into green ones |
| grow ceiling (radii) | how far a window may double, in multiples of *that point's own* stored radius — the same number as `radius-perimeter --grow-radii`, and off by default because the pass's is. **Off is also what produces a runaway**: a plane that is not transverse cuts a streak along the vessel, which touches the border at any width, so the window doubles until it holds the neighbour too. Measured: a 260 µm vessel at max half 128 grew to half=88 and reported 1,062 µm. `h/r` and the summary say when that happened; `4` is the usual cap and does not bind on a vessel that genuinely needs a wide window |
| check for merged neighbours | count, per section, how many *other* segments' centrelines lie inside this section's own blob — the same test the pass uses to decide ownership. Costs one KD-tree over every centreline point of the whole graph whether one segment is selected or all of them, so turn it off on a big tree when the question is about window size rather than ownership |
| tangent search | maximum angular correction searched when the fitted tangent gives an unstable section |
| branch-aware | off uses the legacy single plain cut, with no three-plane stability slab |
| trust the fitted tangent | **the direct test of "the planes are cut at the wrong axis"**: skips the re-cut of flat sections, so any radius that grows was being measured on an oblique plane |

Five things are drawn, each its own layer row:

| colour | meaning |
|---|---|
| violet square | the window a radius **was** measured in |
| orange square | a window that yielded nothing usable — the section reached the border (`truncated`) or no stable section existed (`unstable`). The radius here is interpolated from neighbours, not measured |
| green ring | the lumen boundary `cv2.arcLength` actually measured — the same contour the perimeter came from. **Measured sections only**: a refused window keeps its orange square, but its blob is a slab through whatever the plane grazed, grown until it stopped touching the border, and on this row that ribbon would read as a cross-section of impossible size. It gets the amber row below instead |
| amber ring | the boundary of a blob the pass **refused**, drawn inside its own orange square. Not a measurement — and the *shape* is the diagnostic: a long streak means the plane was cut on the wrong axis, a blob filling its window means the window was too small, a blob merged with a neighbour means the cut caught a second vessel. No centroid line is drawn for these, deliberately: a slab's centroid is not a centre. The `merge` column now says the third case as a number rather than by eye |
| magenta line | centreline point → that section's own area centroid. Measured sections only, and deliberately so — see the amber row |

**The magenta lines are the re-centring that does not happen.** `radius-perimeter`
takes the centreline point as the section's centre and never moves it, so where the
skeleton runs off the lumen's axis the section is cut off-centre and its perimeter is
not the cross-section's perimeter. `optimise-skeleton` is the pass that re-centres;
long magenta lines are the argument for running it first.

The summary and the log table give the same thing numerically: centroid offset in um
and in radii, how much of each window the section filled, how many windows had to grow,
and `measured / stored` per point.

Two further columns answer *why* a section is elliptical, which a shape measure alone
cannot because a collapsed lumen and an oblique cut both give an ellipse:

| column | meaning |
|---|---|
| `shape` | `major / minor` semi-axis of the measured section; 1.00 is round |
| `obliq` | how much longer the boundary was on the *fitted* tangent than on the one finally used. 1.00 means the fitted tangent was already the best cut in the cone, so a flat section there is a genuinely flat lumen. Above 1.00 the fitted tangent was tilted and would have over-read the radius by that factor |

`shape` and `obliq` still cannot separate a collapsed lumen from two vessels measured
as one: both are elongated at obliquity 1.00, because rotating the plane shortens
neither. Two more columns can.

| column | meaning |
|---|---|
| `h/r` | the window's half-width in multiples of *this point's own* stored radius. It starts at 2.5 and doubles while the blob touches its border, so 5 is one doubling and anything past 6 is a window that has stopped being about this vessel. `grow ceiling` is the cap |
| `merge` | `non-adjacent / total` rival centrelines lying inside this section's own blob. `0/2` is two neighbours at a branch node, which is what a junction is. `1/1` is a vessel sharing no node with this one, inside the blob the radius was measured from — the radius beside it is of both. `-` means the merge check was off, or there was no blob to test |

Three caveats, all deliberate. The junction mask and the bifurcation tapers run over
*whole* segments after every section is known, so a section marked `accepted` here can
still be discarded as `junction` by the real pass. The geometry scale is the stored
radius, which is what pass 1 of the real pass uses. And the survey does **not** run
ownership resolution: where `radius-perimeter` would re-cut a merged section through
its local 3-D watershed or refuse it as `unresolved branch overlap`, this reports the
merged section and flags it in `merge`. A radius here with a non-zero `merge` is not a
radius the pass would have written.

## Log

**Below the tabs, not among them** — a splitter pane that stays visible whichever tab
is in front. It is the only panel that reports on what the other three are doing, and
as a fourth tab it hid exactly when it mattered: starting a chain from Workflows meant
switching away from the only view of its progress. Drag the divider down to reclaim the
render area; the tabs themselves cannot be collapsed, so there is always a handle to
drag back.

Everything the commands print, streamed as they print it. `\r` progress lines
overwrite rather than accumulate, as they do in a terminal.

One limitation: capture is Python-level. VTK, numba and other C extensions write to
the file descriptor directly and keep going to the terminal you launched from. That
applies only to in-process jobs — subprocess jobs are read from a real pipe.
