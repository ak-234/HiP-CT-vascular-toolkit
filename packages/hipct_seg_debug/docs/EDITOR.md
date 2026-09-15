# `hipct_seg_debug.edit` — correcting skeletons with a live surface

The parent package tells you *where* a HiP-CT coronary segmentation or skeleton is
wrong. This subpackage lets you fix it, and shows you the lumen surface changing as you
do.

Correcting a skeleton used to mean editing it, writing a new `.am`, and re-running
`coronary_sdf.run_pipeline` — minutes per edit. Here an edit rebuilds only the box it
touched, in under a second, using the same pipeline code on the same lattice. It also
ships scriptable tools for reconnecting skeletons and segmentations that came out of
Avizo in pieces.

> **Looking for a command or a flag?** **[CLI.md](CLI.md)** is the complete
> reference for both entry points. This file explains what the commands are *for*.
>
> Every command below is also a form in the 3D window's **control** dock, which runs
> them, chains them into the documented workflows, and swaps datasets without
> restarting — [CLI.md Part 7](CLI.md#part-7--the-control-panel).

---

## Prerequisites

### Install the package once

`hipct_seg_debug` is pip-installable — one command from the checkout, and every command
below runs from any working directory:

```
python -m pip install -e ".[test]"     # from the checkout root
python -m hipct_seg_debug.edit report graph.am
```

See [README.md "Install"](../README.md#install) for the full environment. If you skipped
the install you get:

```
ModuleNotFoundError: No module named 'hipct_seg_debug'
```

which the checkout's parent directory on `PYTHONPATH` also fixes, if you would rather
not install:

```
python -m pip install -e .        # from the checkout, once
export PYTHONPATH=/f                   # Git Bash
```

### `coronary_sdf`

Also not installed, and also has no packaging metadata — but you do not have to do
anything about it. `_deps.ensure_coronary_sdf()` locates it on first use, trying
`HIPCT_CORONARY_SDF` first, then looking beside the checkout and inside it. If it
lives anywhere else, point the variable at the directory *containing* the package:

```
set HIPCT_CORONARY_SDF=D:\some\other\place
```

Because the checkout-relative locations are tried afterwards, a wrong value here
fails over rather than erroring — you only see `ImportError: cannot find the
'coronary_sdf' package` when no candidate works.

### Packages

Everything below is already present in the Anaconda environment on this machine.

| | |
|---|---|
| **required** | `numpy`, `scipy`, `pyvista`, plus `coronary_sdf` |
| **interactive only** | `qtpy`, `napari`, `pyvistaqt` (as for the parent package) |
| **optional** | `skimage` → `FieldProbability`; `sklearn` → `LearnedProbability`; `statsmodels` → the DPC stationarity test; `skeleton_analysis` → richer `segmentation.report` metrics |

The optional four are imported inside the functions that need them, so
`import hipct_seg_debug.edit` works without any of them, and the two that are only used
for scoring degrade to a simpler result rather than raising.

> **Do not let pip upgrade numpy.** It is pinned at 1.26.4 for the same reasons the
> parent README gives — numpy 2.x breaks `numba` (the RLE decoder) and changes
> `np.unique(..., axis=0, return_inverse=True)` semantics that `coronary_sdf` relies on.

---

## Running it: the 3D window

```
python -m hipct_seg_debug --edit
```

The four inputs default to the LADAF-2024-28 paths and are overridable exactly as in the
parent package (`--raw`, `--graph`, `--seg`, `--surface`). `--edit-box-um` sizes the box
rebuilt around an edit and `--edit-voxel-mm` pins the SDF voxel size; see
**[CLI.md](CLI.md)** for those and every other flag.

Startup builds the SDF session once — roughly a second on LADAF-2024-28 — and prints the
capsule count and voxel size. After that, editing is local.

Press `e` to enable editing (it starts off, so a stray keypress over the render window
cannot change the graph), then:

| key | action |
|---|---|
| `e` | toggle edit mode |
| `d` | delete the picked segment |
| `t` | prune the picked branch and everything past it |
| `x` | split the segment at the picked point — makes a node for a T-junction |
| `k` / `j` | widen / narrow the picked segment (×1.1) |
| `f` | fill the collapsed radii around the pick |
| `u` | rebuild the surface around the pick |
| `z` / `y` | undo / redo |

Picking is unchanged from the parent package: double-click a centreline point. Every
edit reports the box it touched and queues a rebuild of just that box, off the GUI
thread, so a burst of edits coalesces into one rebuild.

A docked **edit** panel carries the same operations as buttons, an adjustable radius
factor, and a *rebuild after every edit* toggle — turn it off to make several edits in
one region and rebuild once. Three new rows appear in the **layers** panel:
`edit_surface` (the regenerated lumen), `edit_handles` (the picked segment's nodes) and
`reconnect` (proposed bridges). They are greyed out until something registers them.

---

## Running it: painting the mask

```
python -m hipct_seg_debug --edit --paint --edits work/edits.npz
```

`--paint` adds a **writable** segmentation layer to the slice window. It needs no
`--edit` and never imports `coronary_sdf`; only turning paint back into centreline
does, so painting and exporting a corrected mask work in a plain read-only session.

`--paint-box` sizes the paintable block (192 segmentation voxels by default) and
`--edits` is where corrections are kept. **Give `--edits` a path** — without one they
live only in memory, and the session can only warn you on the way out.

Select the layer named **`segmentation (editable)`**, then use napari's own tools:
`2` paint, `3` fill, `4` erase, `[` / `]` brush size, `Ctrl+Z` undo. The docked
**paint** panel adds *Commit*, *Revert this box*, *Save edits…*, a 3-D brush
toggle, and **Re-skeletonise painted region**.

Three things about it are worth knowing before you start.

**You are painting the mask's own grid, not the image's.** One pixel of that layer
is one segmentation voxel — 65.98 µm on LADAF-2024-28, four raw pixels across. The
brush looks chunky because the mask *is* chunky; painting on the raw grid would
need a lossy many-to-one reduction to get back to the mask. The read-only
`segmentation` overlay, which is the mask resampled onto the raw grid for display,
is hidden by default when `--paint` is on so the two are not confused. Turn it back
on from the layer list whenever you want to compare.

**Corrections are a sidecar, never a rewrite.** The store holds exactly the voxels
that differ from the source file, keyed by segmentation index, and is composited on
read. Nothing writes to the 2.34 GB lattice, and every consumer in the package —
the overlays, the perimeter-circle measurement, the collapse detector,
`skeletonise`, `repair-mask` — sees the corrected mask because they all reach it
through the same `slice_z`.

**Undo is napari's, and it works.** `Labels.undo()` emits no paint event, so nothing
here accumulates strokes; committing compares the layer against the pristine block
it was opened from and stores the difference. A voxel painted and then undone is
simply not different any more, so it leaves no trace. Commits happen on the button,
on every pick, and on exit.

### Re-skeletonising what you painted

The lumen surface is a function of the skeleton graph alone — `coronary_sdf` never
reads voxel data — so a mask edit reaches the surface only by becoming graph.
*Re-skeletonise painted region* does that for the box around your edits:

```
decode the box  →  skeletonize (lee)  →  distance transform for radii
                →  trace  →  trim  →  weld into the graph  →  local SDF rebuild
```

Two modes, chosen in the panel:

**`add` (default)** keeps only the chain running through voxels you actually
painted and welds its ends onto the existing graph. Nothing already in the graph is
touched. This is the default because of a measurement: scoring a whole-volume
re-skeletonisation against Avizo's graph gave 96% bifurcation recall but **383
false bifurcations** against Avizo's 150, so re-deriving a region wholesale imports
that noise into a graph you have reason to trust.

**`replace`** deletes every centreline inside the box and inserts the freshly
derived one, cutting segments that straddle the boundary so nothing is left to
duplicate the new geometry. Truest to "the mask is ground truth", and there when
that is what you want.

Either way it is **one undo step** — `z` in the 3D window puts the graph back.

The report says what happened, including what it declined to do:

```
re-skeletonise (add): +1 segment(s), 2.32 mm, 2 end(s) welded, 0.67s
```

An end with no existing centreline within the weld tolerance is **left free and
counted**, not attached to the nearest thing available. A fragment that reaches
nothing is a real finding about the mask, and inventing a join would hide it.

### Getting the corrected mask out

```
python -m hipct_seg_debug.edit mask-export --edits work/edits.npz --out corrected.am
python -m hipct_seg_debug.edit mask-export --edits work/edits.npz --out corrected.tif
```

`.am` writes a real `HxByteRLE` Amira lattice through `hipct_seg_debug/rle_write.py`
— the encoder this package did not have until now. Label values are preserved
rather than scaled, so the result is a drop-in replacement input for Avizo and for
your existing pipeline. Measured on LADAF-2024-28: decoding all 2.34 GB and
re-encoding gives **37,725,961 bytes where Avizo wrote 37,709,212** (0.04% larger),
byte-identical when read back, in 1.8 s.

`--edits` also works on `skeletonise`, `optimise`, `repair-mask` and
`repair-radius`, so a correction can flow straight into any of them without being
exported first.

---

## Running it: the command line

```
python -m hipct_seg_debug.edit {report,gaps,connect,surface} GRAPH.am [flags]
```

`GRAPH.am` is an ASCII Amira `HxSpatialGraph`. Binary `.am` is not supported — re-export
as ASCII from Avizo.

> **`gaps` and `connect` are dry runs unless you pass `--out`.** Every gate in this
> package is a heuristic tuned on one dataset. A bridge that should not exist silently
> reroutes flow in whatever CFD run follows, so the default is to print what would
> happen and write nothing.

### `report` — what shape is this graph in?

```
python -m hipct_seg_debug.edit report GRAPH.am
```

Prints segment and node counts, connected components and their sizes, free ends, the
radius range, every intra-segment gap sorted by size, and how much of the graph Avizo
interpolated. No flags; nothing is written.

### `flag-interpolation` — mark what Avizo invented rather than measured

```
python -m hipct_seg_debug.edit flag-interpolation GRAPH.am --show-spans --out flagged.am
```

Where the segmentation has a hole, Avizo writes points across it so the edge stays
continuous. They are not skeletonisation output: nothing measured them, the radius is
generated, and — the part that costs something — a tree that is genuinely in two pieces
looks like one, so the reconnection stages are never asked to rebuild the join from the
greyscale.

Three signatures, each with its own reason code, all documented with their measurements
in `interpolation.py`'s docstring: a **straight bridge** (no curvature, an exact linear
radius ramp, and a big radius step at *both* ends that dwarfs the run's own taper); a
**degenerate radius** (sitting on a detached floor at the bottom of the distribution —
`adjust_thickness`'s clamped intercept, or a raw zero); and, with `--seg`, a **run**
outside the mask.

On LADAF-2024-28 that is 36 of 29,122 points: edges 183 and 197, plus ten points on the
81.27 µm floor. The both-ends test is what makes it specific — collinearity alone would
flag 941 runs, a fifth of the graph, and requiring the linear ramp as well still leaves
941. Worth knowing, because it contradicts the obvious story: both straight bridges lie
*entirely inside* the segmentation this graph is paired with today.

The answer is stored as a per-point `avizo_interpolated` field and travels with the
`.am`. **Nothing detects on its own** — every stage reads the stored field and treats an
unflagged graph as clean. That is deliberate: a perfectly straight, linearly interpolated
segment is what half the fixtures in `tests/conftest_geometry.py` are made of, and a
detector running inside every pipeline would quietly change what those tests mean. The 3D
viewer is the single exception, for its own display layer only.

Two consequences, and the difference is the whole design:

- a **span** — three or more consecutive points — is an invented piece of vessel.
  `connect` cuts it out so the break becomes two genuine free ends, `surface` refuses to
  mesh it, and the viewer draws it flat grey with the centreline broken either side.
- a **lone** flagged point is an invented *radius* on a real point. `radius-perimeter`,
  `repair-radius`, `optimise`, `score` and the smoothers all skip it; nothing is cut.

### `flag-interpolation` — the fourth signature: unsampled jumps

`flag-interpolation --seg SEG.am` now also looks for **unsampled jumps**: a single
enormous step inside one edge, with *no points between its ends*, whose two ends land
in **different mask components**.

This is the artefact the other three signatures cannot see, because there is nothing
to flag — Avizo joined two traced runs without sampling between them, so there is no
run of invented points and no degenerate radius. On LADAF-2024-28 the point signatures
find 36 points in 12 spans; the jump signature finds **24 more breaks, 72 mm in total**,
and every one of them is a place the graph asserts continuity that the segmentation
does not have.

Two gates, and the second is a measurement rather than a heuristic:

- the step is an outlier **against its own segment** (5× that segment's median spacing,
  and at least 600 µm). Median spacing on LADAF-28 is 93 µm, p99 is 114;
- the two ends sit in **different mask components**. A long step *inside* one component
  is under-sampling, not a break, and is left alone — eleven of the 38 large steps on
  LADAF-28 are exactly that.

The 600 µm floor sits in an empty band: of the 38 steps above 500 µm, every one above
600 crosses a break (24 of them) and the eleven that do not are all 510–590 µm.

**It is recorded in its own field, `unsampled_jump`, not as another bit in
`avizo_interpolated`.** The two make different claims. `avizo_interpolated` means *this
point is invented, keep it out of every measurement*; a jump's two anchors are ordinary
measured centreline and it is the empty step between them that is fabricated. Folding
one into the other would silently drop two real points from every radius and length
statistic in the toolkit.

`connect` and `surface` cut these like any other flagged span, and `restore_unbridged`
puts back the ones nothing bridged. Cutting the 24 on LADAF-28 takes the graph from
2 components to 25 and gives the reconnector its first real work on that dataset —
without it, `connect --geodesic` proposes nothing at all.

Pass `--no-jumps` to skip the signature, or `--min-jump-um` to move the floor.

### `gaps` — fill jumps inside a single edge

```
python -m hipct_seg_debug.edit gaps GRAPH.am --out fixed.am
```

Both thresholds must be exceeded. The relative one is what stops normal wide sampling on
a thin vessel being mistaken for a gap.

**`0 gap(s) to fill` is not "the graph is whole".** This command only looks *inside* one
edge's point list, and a skeletonised graph has no such jumps by construction — so it
prints 0 on a graph with 52 disconnected components. It now names the components and free
ends it did not examine, and points at `connect`. It also still writes `--out` in that
case, so a chain that declared the file does not stop on a clean pass.

### `connect` — bridge disconnected ends

```
python -m hipct_seg_debug.edit connect GRAPH.am --tjunction --show-rejected --out fixed.am
```

Loosening `--cone-deg` and `--reach-factor` is the usual way to get candidates out of a
graph that yields none; `--show-rejected` tells you which gate was responsible before
you start guessing. When it finds nothing at all, the report names the stage that
consumed everything — pairs out of reach, or pairs pruned as same-component — because
those two happen before a candidate exists and so never reach `--show-rejected`.

Add `--dpc` (with `--raw` and `--seg`) to put every surviving proposal through the DPC
walk, which reads the image instead of trusting the geometry. That is minutes rather than
seconds, and the GUI runs it as a child process so Stop still works.

### `connect --geodesic` — repair the graph and the mask together

```
python -m hipct_seg_debug.edit connect GRAPH.am --geodesic \
    --seg SEG.am --raw RAW/ --out fixed.am --out-seg fixed-seg.am \
    --review-json review.json --decisions-json decisions.json
```

Opt-in, and mutually exclusive with `--dpc` — they are two answers to the same
question, and running one over the other's topology gives a result neither chose.
The differences that matter:

- **It repairs the segmentation too.** `--out` and `--out-seg` are required
  together: a graph written next to the *original* mask is two files that disagree
  about where the vessels are, and nothing downstream can tell which is right.
- **It classifies before it searches.** Two free ends already inside one mask
  component do not need a route at all — the lumen is continuous and only the
  centreline broke — so those are re-skeletonised and **no voxel is invented**.
  On this dataset that is most of them.
- **It does not assume a round vessel.** The cross-section is measured at both
  intact ends and transported across the gap on a rotation-minimising frame, so a
  collapsed slit stays a slit. A capsule sweep would insert a few hundred microns
  of anatomically impossible round lumen between two flattened ends.
- **It says when it does not know.** Ambiguous routes — a second route within 15%
  of the best — go to `--review-json` instead of being resolved by a threshold.

Without `--raw` it will still re-skeletonise mask-connected breaks and close gaps
of a voxel or two, and sends anything longer to review: the mask alone cannot
justify inventing a vessel.

Dry run is still the default. With neither output path given, everything is
proposed, searched and decided, and nothing is written — which is the useful first
run, together with `--review-json`.

The review loop is: run with `--decisions-json decisions.json`, adjudicate in the
GUI's **Reconnect** tab (or by editing the file), then re-run with the same
`--decisions-json` and the rulings are applied. Decisions are matched by endpoint
rather than by position, so a re-run on a graph that has moved on cannot apply a
ruling to the wrong candidate — it reports the unmatched ones instead.

### `skeletonise` — derive a centreline from the mask

```
python -m hipct_seg_debug.edit skeletonise --out candidate.am --order
```

`skimage`'s Lee thinning, then radii from the distance transform, then a trace
into a graph. Adjacent junction voxels are **clustered into one node** — without
that a bifurcation becomes a cloud of degree-3 nodes a voxel apart and every
branch-point measure downstream is wrong.

Measured on the full LADAF-2024-28 lattice: decode 2.9 s, thinning 107 s
(36,925 skeleton voxels), distance transform 570 s, trace 61 s — about 12 minutes,
producing 1226 segments against Avizo's 309. The distance transform runs in
overlapping z slabs and is sampled only at skeleton voxels, because asking
`distance_transform_edt` for the whole volume returns **float64** — 18.7 GB for an
answer that occupies two.

`sknw` is deliberately not used: it is unmaintained, not installed, and the part
that matters (junction clustering) has to be written anyway.

### `optimise` — order, clean radii, and score against a reference

```
python -m hipct_seg_debug.edit optimise candidate.am --reference AVIZO.am --sensitivity
```

Two things about `skeleton_analysis` are worth knowing before reading its output:

- **its `optimisation` subpackage does not optimise a skeleton, it scores one.**
  There is no sweep and no minimiser; `meta_metric` and `super_metric` are
  objective *functions*. The code that changes a skeleton lives in `outlier`.
- **`super_metric` compares two segmentations, not two skeletons.** Given one mask
  and two centrelines its Volume/CC/Euler terms are identical for both and cancel
  exactly. So the headline here is the **bifurcation Dice** — greedy
  nearest-neighbour matching of the two skeletons' branch points — plus each
  skeleton's centreline sensitivity against the mask, and a table of segment,
  node, component, length and radius statistics.

Scoring a graph against itself returns Dice 1.000, which is the sanity check that
the conversion between the two packages' graph classes is faithful.

Measured on LADAF-2024-28, generated skeleton against Avizo's:

```
                 segments     nodes    comp    ends    bifs     length   radius p5/50/95      sens
  reference       309 seg   311 nodes  2 comp  161     150     2933 mm   162/ 247/ 970 um   0.9109
  candidate      1226 seg  1254 nodes  52 comp 609     526     3238 mm    66/ 162/ 723 um   0.9408

  bifurcation Dice 0.425  (tp 144, fp 383, fn 6; 527 candidate vs 150 reference)
```

Read that as: the generated skeleton **finds 144 of Avizo's 150 branch points and
misses 6** — 96% recall — but invents 383 more, which is the classic thinning
signature of spurious twigs off a rough mask surface. Its centreline sits inside
the mask slightly *better* than Avizo's (0.941 against 0.911), which it should,
being derived from it. Total length agrees to 10%, so it is following the same
vessels. The radii are systematically smaller (median 162 against 247 µm) because
a distance transform measures the lumen while `adjust_thickness` re-inflates it
from the perimeter — the difference the parent README calls "re-inflation is by
design".

This is why the generated skeleton is scored rather than substituted: 383 false
bifurcations and 52 components would have to be pruned and reconnected before it
could carry a surface.

### `repair-radius` — fill collapsed radii from the local taper

```
python -m hipct_seg_debug.edit repair-radius GRAPH.am --source both --out fixed.am
```

See [the section below](#restoring-a-collapsed-vessels-radius) for what it does and
why it declines as often as it acts.

Apply graph/taper repair before `radius-perimeter`, which must remain the last
radius-changing step. On perimeter-tagged graphs image-collapse detection is skipped;
high spans are still reported and are repaired only with `--allow-decrease`.

`radius-perimeter` is branch-aware by default. It validates a short stack of planes,
uses a local 3D marker watershed only to separate non-adjacent touching vessels, and
leaves the genuinely shared parent/daughter lumen unmeasured. The resulting graph stores
a parent-through/daughter-emergence taper plus `radius_reject_reason` and
`radius_resolution_mode` audit fields. There is no global radius cap; the default
`--max-radius-factor 2` remains relative to each edge's robust local neighbourhood.

The graph argument and segmentation argument are different file types:
`optimise-skeleton GRAPH.am --seg LABELS.am` expects `LABELS.am` to be an Amira lattice
with `define Lattice`, not another spatial graph.

### `repair-mask` — cull debris and close small breaks

```
python -m hipct_seg_debug.edit repair-mask --min-voxels 500 --close 1 --out fixed.tif
```

Reports component count, Euler number and volume before and after. **Read those as
a pair**: a closing that mends a real break lowers the component count and leaves
Euler alone, while one that welds two unrelated vessels lowers both.

### `surface` — regenerate the lumen mesh

```
python -m hipct_seg_debug.edit surface GRAPH.am --out-dir surfaces/
```

A full, unmodified `generate_sdf_surface` run — this is the export path, not a preview.
Writes `lumen_bspline.stl` and `lumen_bspline.vtk`. On LADAF-2024-28 at the derived
0.146 mm voxel that is ~88 s and 6.8 M triangles (a 127 MB STL); a finer `--voxel-mm`
costs sharply more in both.

---

## The whole workflow

```
# 1. what shape is the graph in?
python -m hipct_seg_debug.edit report AVIZO.am

# 2. derive an independent skeleton from the mask and score it against Avizo's
python -m hipct_seg_debug.edit skeletonise --order --out candidate.am    # ~12 min
python -m hipct_seg_debug.edit optimise candidate.am --sensitivity

# 3. repair the working graph
#    flag first: every later step reads the field, and treats an unflagged graph as clean
python -m hipct_seg_debug.edit flag-interpolation AVIZO.am --out step0.am
python -m hipct_seg_debug.edit gaps          step0.am --out step1.am     # intra-edge gaps
python -m hipct_seg_debug.edit connect       step1.am --tjunction --out step2.am
python -m hipct_seg_debug.edit repair-radius step2.am --source both --out step3.am

# 4. and the mask, if you want it consistent with the graph
python -m hipct_seg_debug.edit repair-mask --min-voxels 500 --close 1 --out mask.tif

# 5. fix by hand what no heuristic should be trusted with: paint the mask,
#    re-skeletonise the painted region, watch the surface follow
python -m hipct_seg_debug --edit --paint --edits work/edits.npz

# 6. take the corrected mask back out, and feed it to anything
python -m hipct_seg_debug.edit mask-export  --edits work/edits.npz --out corrected.am
python -m hipct_seg_debug.edit skeletonise  --edits work/edits.npz --out candidate2.am

# 7. regenerate the surface
python -m hipct_seg_debug.edit surface step3.am --out-dir surfaces/
```

`flag-interpolation` comes first because it changes what every later step is willing
to touch: `connect` cuts each invented span so the break becomes two real free ends,
`repair-radius` refuses to extrapolate a taper from an invented radius, and `surface`
will not mesh a capsule along a centreline nobody observed. Run without it and the
chain still works — it simply treats Avizo's fills as data.

Steps 3 and 4 are dry runs without `--out`, so each can be inspected before it is
committed to. Step 5 writes only to its `--edits` sidecar; the source lattice is
never modified by anything here.

## A worked run

Real output from LADAF-2024-28, so you can check you are seeing the same thing.

**1. Look before touching anything.**

```
$ python -m hipct_seg_debug.edit report ASCII_smooth_thick_adj_LADAF_2024_28...am
ASCII_smooth_thick_adj_LADAF_2024_28.Spatial-Graph.attributegraph.am: 309 segments, 311 nodes
  components : 2  sizes=[160, 149]
  free ends  : 161
  radii (um) : min=119 median=240 max=1505
  intra-segment gaps: 11
    segment   297 at point     0: 9.81 mm
    segment   295 at point   302: 9.78 mm
    ...
```

Two components here are the anatomically distinct left and right coronary trees, not a
defect. The 11 gaps are the defect.

That distinction can be made explicit rather than inferred: `skeletonise --per-tree`
tags every edge with a `tree` index taken from the mask's own connected-component
labelling, and `pick-roots` records which node roots each one. See
[SKELETONISATION.md](SKELETONISATION.md) §6.1 and §6.2.

**2. Fill the gaps.**

```
$ python -m hipct_seg_debug.edit gaps <graph>.am --out fixed.am
11 gap(s) to fill
  segment   297 at point     0: 9.81 mm -> 49 points
  ...
filled 11 gap(s)
wrote fixed.am
```

**3. Confirm.**

```
$ python -m hipct_seg_debug.edit report fixed.am
  components : 2  sizes=[160, 149]
  intra-segment gaps: 0
```

**4. Look for genuine disconnections.** With the default `same_component=False` this
graph yields nothing — no free end of one tree lies within reach of the other, which is
correct. Within a tree:

```
$ python -m hipct_seg_debug.edit connect fixed.am --tjunction --same-component
end-to-end:
0 accepted of 34 candidates
  rejected: outside the 50deg search cone (31)
  rejected: the target end faces away (3)

end-to-vessel (T-junction):
21 accepted of 5618 candidates
  tjunction  21
  rejected: outside the 50deg search cone (4538)
    <Bridge tjunction 144 -> seg 92[20] 3549um score=0.593 ok>
    ...
```

Review those before adding `--out`. Then regenerate:

```
$ python -m hipct_seg_debug.edit surface fixed.am --out-dir surfaces/
```

---

## Restoring a collapsed vessel's radius

A collapsed vessel does not lose its centreline — the skeleton runs straight
through it — it loses its *calibre*. The pipeline assigns radius from the
cross-section perimeter, and where the lumen has flattened that under-states it
badly, so the surface pinches to a thread over a span that should be a smoothly
tapering tube.

Nothing that existed could fix it. `skeleton_analysis`'s `filloutliers_nearest`
copies the nearest good *value*, which flattens the region instead of continuing
the taper; `adjust_thickness.fit_line` fits one global thickness→radius line for
the whole tree; and `coronary_sdf.smoothing`'s four radius passes are all
junction- or endpoint-anchored — `smooth_segment_radii` deliberately preserves the
level and the other three only reach a few points in from a node. A mid-segment
collapse is out of range of every one of them.

So the model is: **radius falls at a roughly constant fraction per unit length
along an unbranched vessel**, i.e. `log r` is linear in arclength. Fitting in log
space is what makes it scale-free — a trunk and a twig get the same treatment —
and means an extrapolated radius can never come out negative, which a linear fit
across a long gap readily does.

Both sides of the gap are fitted and blended by distance, so the fill meets healthy
tissue exactly at *both* boundaries rather than stepping at the far one. With too
few healthy points on one side (a collapse running into a terminal) it falls back
to one-sided extrapolation.

### Four guards, each for a failure that is otherwise silent

| guard | why |
|---|---|
| `min_healthy` (5) | refuse to fit from too few points, and say so, rather than guess |
| `max_taper_per_mm` (0.5) | a short noisy window must not explode over a long gap |
| `only_increase` (on) | a collapse *under*-states radius, so the fit is a floor, not a replacement |
| `JUNCTION_MARGIN` (6) | **a lumen at a bifurcation is legitimately non-circular** |
| `MAX_BULGE` (1.05) | between two healthy anchors a tube has no reason to swell above both |

The last two are not tuning knobs, and both were added because the real data
showed the failure:

- Without the junction margin, 15 of 83 image-detected collapse runs on
  LADAF-2024-28 began at point 0 of their segment — exactly on a junction. That is
  a carina, not a collapse, and inflating it would be inventing anatomy. The
  junction neighbourhood already belongs to `coronary_sdf`'s
  `prune_bifurcation_shrink` and `smooth_radius_transitions`.
- Without the bulge cap, `seg 237[6:39]` was filled from 375 µm to **572 µm** at
  the centre of a span whose two ends measured 353 and 375. Six healthy points
  spanning 0.3 mm gave a badly-determined slope which was then carried across a
  2 mm gap. r-squared does not catch this, because the problem is the lever arm,
  not the scatter. With the cap it fills to 394 µm.

### What it does on this dataset

`--source image` finds 66 spans (from 228 cross-section candidates) and fills 37;
the other 29 decline with *"the fit did not move any radius"* — the fit sat below
the stored radii and `only_increase` kept them. That is the correct outcome, not a
failure.

`--source outlier` finds **nothing**, which is also correct and worth
understanding: `adjust_thickness` has already re-inflated these radii from the
cross-section perimeter, so they do not dip below their own trend. The collapses
here are *shape* collapses — a flattened lumen whose perimeter still reads large —
which is exactly what the image detector measures and the radius detector cannot
see. On a graph whose radii came straight from a distance transform, the reverse
would be true.

## Using it from Python

Install the package first (`pip install -e .`); no cwd or `sys.path` setup is needed.

### Edit and rebuild locally

```python
from hipct_seg_debug.edit.adapter import read_triple
from hipct_seg_debug.edit.graphmodel import EditableGraph
from hipct_seg_debug.edit.sdfpatch import SdfSession

graph = EditableGraph(read_triple("graph.am"))
session = SdfSession(graph.snapshot())          # preprocessing, once

sid = graph.segment_ids()[0]
patch = graph.scale_radii(sid, 1.2)             # every edit returns a Patch

session.set_graph(graph.snapshot())             # re-prepare against the edit
result = session.rebuild_around(patch)          # rebuild only that box
print(f"{result.surface.n_cells:,} triangles in {result.seconds:.2f}s")

graph.undo()                                    # exactly reversible
```

`EditableGraph` also has `move_point`, `move_node`, `set_radius`,
`set_segment_radii`, `insert_point`, `delete_point`, `delete_segment`,
`delete_subtree`, `split_segment`, `add_segment`, `merge_nodes`,
`weld_coincident_nodes` and `reroot`. Wrap several in `with graph.batch("label"):` to
make them one undo step and one rebuild.

### Propose and apply reconnections

```python
from hipct_seg_debug.edit.reconnect import apply_bridges, endpoints, summarise, tjunction

stats = {}
proposals = endpoints.propose(graph, keep_rejected=True, stats=stats)
proposals += tjunction.propose(graph, keep_rejected=True)
print(summarise(proposals, stats))              # accepted, and why the rest were not

accepted = [b for b in proposals if b.accepted]
apply_bridges(graph, accepted)                  # one undo step
```

Each `Bridge` carries `coords`, `radii`, `score` and a `metrics` dict with the measured
span, radius ratio, tortuosity and cone angle — enough to draw or to audit.

**Pass `stats` if you might get nothing back.** Two of the ways a pair dies happen
*before* a `Bridge` exists — nothing within reach, and the same-component prune — so
without the counters an empty list is indistinguishable from a graph with no free ends,
and `--show-rejected` has nothing to show either. On the Avizo LADAF-2024-28 graph the
bare message was `no reconnection candidates`; with `stats` it is:

```
no reconnection candidates (161 free end(s) examined)
  34 pair(s) within reach
  pruned: 34 pair(s) inside one component (pass --same-component to allow those)
```

which names the knob to turn. When nothing is in reach at all it reports the distance
between the nearest two pieces instead, so you can tell a reach that is too short from a
cone that is too narrow.

### Let the image decide (the DPC walk)

The geometric proposers above never look at the image. `dpc.refine` re-walks each
proposal through the greyscale and accepts or rejects it. From the command line — which
is also how the GUI's Commands tab runs it, since those forms are generated from
argparse:

```
python -m hipct_seg_debug.edit connect graph.am --tjunction \
    --dpc --raw <dir of TIFF slices> --seg <lattice.am>
```

On the 1226-segment skeletonised LADAF-2024-28 graph, the geometry alone proposes 40
bridges (52 components → 30). The walk keeps **16** of them (52 → 41), refusing 12 whose
probability series is not stationary, 8 that pass through a hole in the signal — the
shortcut-through-tissue case the trough test exists for — and 4 walks that never arrived.
Accepted bridges also come back with the path the *walk* took rather than the
interpolated one, so their spans change.

`--dpc-learned` swaps `FieldProbability` for a `LearnedProbability` trained on this
graph's own skeleton. Worth trying where the segmentation is absent across the gap, which
is exactly where a reconnection has to walk; it trains on whichever ROI holds the most
centreline and warns when that is a thin training set.

Directly, in Python:

```python
import numpy as np
from hipct_seg_debug.edit.reconnect import dpc, roi as roi_mod
from hipct_seg_debug.edit.reconnect.probability import FieldProbability, Roi

# Either build the Roi by hand...
roi = Roi(
    volume=raw_zyx,                       # [z, y, x] greyscale sub-volume
    origin_um=np.array([x0, y0, z0]),     # world position of voxel (0, 0, 0)
    spacing_um=np.array([vx, vy, vz]),
    mask=seg_zyx,                         # optional segmentation on the same grid
)
# ...or let `roi` cut one out of the TIFF stack around each proposal. `build_many`
# reads every needed slice exactly once rather than once per bridge, which on a
# 40-bridge run is 1233 slice decodes instead of 2615.
spans = [roi_mod.span_for(b, frame) for b in proposals]
rois = roi_mod.build_many(stack, frame, spans, labels=labels)

refined = dpc.refine(graph, roi, FieldProbability(roi, dark_vessels=True), proposals)
apply_bridges(graph, [b for b in refined if b.accepted])
```

`dark_vessels=True` is right for native HiP-CT, whose lumen is *darker* than the
myocardium around it (`selftest.py:36` measures exactly that); the vesselness filter
responds to bright tubes, so the field is inverted first. `--dpc-bright-lumen` is the
CLI's escape hatch for a contrast-enhanced scan.

### Export

```python
from hipct_seg_debug.edit.amira_write import write_spatial_graph

write_spatial_graph(graph.to_spatial_graph(), "edited.am", parameters_from="graph.am")
session.set_graph(graph.snapshot())
session.rebuild_full("surfaces/")               # the real pipeline, for the STL you ship
```

`parameters_from` carries the source file's `Parameters` block across, including the
`TransformationMatrix`. A graph that has quietly lost its transform still opens in Avizo
and still looks right — it is simply in the wrong place, which is a bad thing to discover
after meshing it.

---

## Tests

```
python -m pytest             # 460 tests, ~33 s
python -m pytest --runslow   # + full pipeline runs, minutes
python -m hipct_seg_debug --selftest                       # the parent's own checks
```

The fast tests need no dataset. The `--runslow` ones drive a real
`generate_sdf_surface` and are what guard the agreement figures below. Synthetic-volume
tests cover the two cases that matter for reconnection: a break that must be mended,
and — more importantly — two vessels running close together that must *not* be welded.

Four of them exist because the failure they catch is invisible rather than loud:

- **`test_napari_undo_empties_the_store`** — `Labels.undo()` emits no paint event, so
  a store built by accumulating events would keep strokes the user has already
  undone and can no longer see.
- **`test_painted_voxel_lands_where_the_slab_gather_puts_it`** — walks napari's
  layer transform forward and the slab's gather back, and requires them to meet.
  A half-voxel disagreement here displaces every edit and still looks plausible.
- **`test_add_mode_joins_the_two_components`** — asserts, point for point, that
  Avizo's geometry either side of a painted bridge is *unchanged*.
- **`test_encode_output_decodes_with_the_existing_decoder`** — the encoder is
  checked against `rle.py`, which was written months earlier and independently, not
  against a decoder written alongside it.

Run the parent self-test too after touching `viewer2d.py` or `volume.py`: they are
shared with the read-only auditing path, and `lazy mask alignment` and
`full-slice slab` are the regression guards for the layer changes.

---

## Troubleshooting

**`ModuleNotFoundError: No module named 'hipct_seg_debug'`** — the package is not
installed; run `python -m pip install -e .` in the checkout. See
[Prerequisites](#run-from-f).

**`ImportError: cannot find the 'coronary_sdf' package`** — set `HIPCT_CORONARY_SDF` to
the directory containing it.

**Several PyVista windows open and everything freezes.** `coronary_sdf.config.DEBUG_VIS`
and `DEBUG_VIS_BLOCK` both default to `True`, so an unguarded pipeline call opens about
six *blocking* windows. Everything in this package goes through
`sdfconfig.sdf_config`, which forces them off and restores the globals afterwards — if
you are calling `coronary_sdf` directly, do the same.

**A config override does nothing.** `SdfConfig` (`config.py:875`) looks like the
configuration API but nothing in the pipeline reads it. Settings are module globals;
`sdf_config` is the supported way to change them.

**Rebuilds are slower than a second.** Shrink `--edit-box-um`; drop
`SDF_MAX_CAPSULE_QUERY` from 64 to 16 (the legacy value); turn off `THIN_VESSEL_REFINE`.
All three trade preview fidelity only — the export is a separate full run.

**`connect` proposes nothing.** Usually correct. Joins inside one component are refused
by default (`--same-component` allows them), and the reach is 15 × the endpoint radius,
so a thin vessel cannot bridge far. Run with `--show-rejected` to see which gate fired,
and check `report` first — the break may be an intra-segment gap, which is `gaps`' job.

---

## How the live rebuild works

`coronary_sdf.Grid` is a plain dataclass with no invariants, and `build_narrow_band` /
`evaluate_sdf` take one as an ordinary argument — nothing assumes it covers the whole
tree. On LADAF-2024-28 the full grid is 201 M voxels with a 2.5 M-voxel narrow band; a
4 mm box is ~700 k voxels with a band under 100 k.

Two details make the patch trustworthy rather than merely fast.

**The local grid is a literal sub-block of the full grid.** `evaluate_sdf` samples world
positions out of `grid.x/y/z` (`sdf_field.py:616`), so those arrays are *sliced*, never
rebuilt.

**The mesher gets a different origin than the field.** `compute_grid` samples at
`linspace(bbox_min, bbox_max, dims)` — step `extent/(dims-1)` — but hands the mesher
`voxel_size`, and those differ by ~0.15%. A full run accumulates that from index 0, so a
patch restarting its indexing at `i0` lands `i0 × (voxel_size − step)` away. A few
hundred voxels in, that is tens of micrometres of pure translation. The fix is one line
in `_subgrid`, and it is the difference between the two columns below.

### Measured agreement with a full pipeline run

| | before the origin fix | after |
|---|---|---|
| capsules, grid, SDF field near the surface | identical | identical |
| mesh, `SDF_MESH_METHOD='mc'` | 42 µm mean (29% voxel) | **0.001 µm mean, 0.019 µm max** |
| mesh, `SDF_MESH_METHOD='meshlib'` | 41 µm mean (28%) | **0.79 µm mean (0.5% voxel), 30 µm max** |

With the local mesher the preview *is* the export. With meshlib the residual is its
global `relaxKeepVolume` (30 iterations over the whole mesh), which a patch cannot
reproduce and which does not shrink with margin — 0.5% of a voxel, far below the
resolution the graph was derived from.

Margin only needs to cover the post-field stencils (marching cubes, Taubin), not the
smooth-min blend: every capsule reaches `evaluate_sdf` whatever the grid is, so the field
inside the box is already correct on its first voxel. Six voxels.

---

## Modules

| | |
|---|---|
| `adapter` | `SpatialGraph` ⇄ the `(nodes, points, segments)` triple. Verified bit-identical to `coronary_sdf.parse_am` on the real graph. |
| `graphmodel` | `EditableGraph`: move, split, delete, prune, add, merge, weld, reroot — every one reversible. |
| `history` | `Patch` (what an edit touched, and *where*) plus undo/redo. |
| `sdfpatch` | `SdfSession`: graph-level preprocessing once, SDF evaluation per box. |
| `sdfconfig` | Scoped, restorable overrides for `coronary_sdf.config`. |
| `worker` | Off-thread rebuilds, coalescing a burst of edits into one. |
| `controller` | Binds the above to the 3D pick window. |
| `controls_edit` | The docked edit panel. |
| `amira_write` | ASCII `.am` writer that preserves the source's `Parameters` block. |
| `lattice` | decodes the RLE lattice to an array, and adapts it to what `skeleton_analysis` expects. |
| `skeletonise` | mask → centreline graph: Lee thinning, slab-wise distance transform, junction-clustered tracing. |
| `components` | splits the mask into its connected components (the left and right trees), skeletonises each in its own bounding box, and merges the pieces back with a per-edge `tree` field. |
| `roots` | the roots sidecar: which node roots each tree, recorded by coordinate so it survives a re-skeletonisation, and placed back onto a graph by `resolve`. |
| `skeletonisers` | several algorithms behind one signature — Lee, TEASAR (`kimimaro`), and ingest of an Avizo export — so they can be scored against each other. |
| `supermetric` | the Walsh–Berg five-term super metric, Eq. 10, computed from the graph against the binary image. See [SKELETONISATION.md](SKELETONISATION.md). |
| `skeleton_optimise` | de-loop, prune leaves shorter than the local vessel radius, re-centre on the lumen cross-section, then smooth. |
| `smoothers` | five selectable centreline smoothers behind one call — the built-in gaussian, `coronary_sdf`'s savgol/bspline and its constrained multiscale optimiser — so the super metric can choose between them. Radii are asserted untouched. |
| `radius_perimeter` | replaces every radius with the perimeter of that point's own cross-section, hybridising to area below the resolution gate. |
| `optimise` | Strahler/topological ordering, radius cleanup, scoring against a reference skeleton. |
| `radius_repair` | collapsed-span detection and the log-taper fill. |
| `interpolation` | finds the points Avizo invented across a gap, records them as a per-point `.am` field, and offers the two ways to honour it: a mask for anything that measures, a split for anything that needs the topology to be honest. |
| `maskedit` | `MaskEdits`, the sparse correction store, and `MaskSource`, which composites it onto the lattice behind the same `slice_z` everything already calls. |
| `paint` | `PaintSession`: the writable napari layer, commit-by-diff, and the docked controls. |
| `reskeletonise` | a painted region back into centreline — pad, trace, trim to what is new, weld, splice. |
| `_deps` | Locates `coronary_sdf`. |
| `reconnect/` | `gaps`, `endpoints`, `tjunction`, `dpc`, `probability`, `roi`, `segmentation`. |

One module lives in the parent package rather than here, because it is the
counterpart of a reader that has always been there: **`hipct_seg_debug/rle_write.py`**,
the `HxByteRLE` encoder.

---

## Reconnection

Four kinds of break:

- **`gaps`** — a jump inside one edge's point list. Wraps
  `coronary_sdf.bridge_centerline_gaps`, which already does this well; this module makes
  it undoable and reports where it acted.
- **`endpoints`** — two free ends that are one vessel. Gates ported from
  the earlier skeleton-graph-editing-toolkit script (cone 50°, reach 15×radius, radius ratio ≤5,
  tortuosity ≤1.8), with three fixes: that file defines `reconnect_end_points` **twice**,
  so its newer Hermite version is dead code; the search is O(n²) with no spatial index;
  and its `.am` reader swaps x↔z and drops `Parameters`.
- **`tjunction`** — a free end belonging on the *side* of another vessel. Needs a new node
  mid-vessel, so it goes through `split_segment`. Not expressible in any existing tool
  here.
- **`dpc`** — the DPC walk ([arXiv:2504.01597](https://arxiv.org/abs/2504.01597),
  Med. Image Anal. 2025). Rather than interpolating between two ends, it *reads the
  image*: stepping voxel by voxel, scoring each neighbour by distance to target,
  centreline probability, and agreement with the last two steps — `DPC = D + 5·P_N + C`
  over a 5×5×5 neighbourhood. Type 3 ("branch occurrence") aims at a whole centreline
  rather than its end, which is the T-junction case.

The persisted production provider is a DF21 Cascade Forest over the concatenated
max-pooled `15³` and raw `7³` HiP-CT patches; `environment-cfc.yml` pins its Python
3.9 runtime. The geometric proposers are cheap and generate candidate pairs;
`dpc.refine` is expensive and decides. Legacy alternatives remain available through
`FieldProbability` (vesselness × EDT) and `LearnedProbability` (gradient-boosted local
statistics), but neither is substituted for CFC when `--cfc-model` is supplied.

**`roi`** is what connects the walk to the data: the walk wants a greyscale array, and
the greyscale is a directory of 4753 LZW TIFFs. It cuts one box per proposal — padded, so
a walk that bulges out to follow the vessel does not fall off the edge — and samples the
label lattice onto the *raw* grid rather than its own, so `volume` and `mask` are
index-for-index comparable the way `FieldProbability` assumes when it sums them.
`build_many` does the whole run in one ordered pass, because a HiP-CT slice costs a full
19 MB decode whether you want all of it or an 84×76 window, and the boxes overlap heavily
in z — they are all on the same coronary tree.

`segmentation` repairs the voxel mask instead: label, cull debris, close small breaks,
and burn accepted bridges in as tapered capsules so mask and surface agree. Read its
`report` output as a pair — closing that mends a real break lowers the component count
and leaves the Euler number alone, while closing that welds two unrelated vessels lowers
both.

### `geodesic` — the collapse-aware connector

A fifth kind of repair, and the only one that treats the graph and the mask as one
object. It exists because of three facts about this data that the four above do
not encode.

**The specimen is collapsed.** Ex-vivo HiP-CT coronaries are slits and ribbons, not
tubes. A Frangi-style vesselness filter scores a one-voxel slit near zero, so
`cost` scores tube, ribbon *and* sheet responses and takes the best; and it adds a
flux term — the divergence of the unit gradient field — which measures "two walls
facing each other across a gap" and is indifferent to the cross-section's shape.
`shape` then transports the measured cross-section along the accepted route on a
rotation-minimising frame (Wang et al., ACM TOG 27(1)) rather than sweeping a
capsule. A Frenet frame is unusable here: it is undefined on a straight run and
flips through every inflection, which is most of a bridge.

**Most "disconnections" are not mask gaps.** `classify` splits candidates four
ways by comparing the mask component at each end — same component (re-skeletonise,
invent nothing), different components (search), an unskeletonised fragment in
between (skeletonise it and include it), or no usable association (review). The
component index is built by `components`, which labels the mask 26-connected
**from its run-length structure**, two planes resident at a time, rather than
decoding 2.34 GB and handing `ndimage.label` another 9.4 GB.

**Greedy is not good enough, and neither is confident.** `astar` searches globally
with the arrival direction in the state, so curvature has an explicit price and a
dropout mid-gap costs what it costs instead of ending the walk the way the DPC
step does. It runs coarse-to-fine — the coarse pass finds *which corridor* and is
deliberately not orientation-aware, which makes it 27× smaller — and returns up to
three spatially distinct alternatives. The second one is the only honest basis for
calling a route ambiguous.

Everything is calibrated per candidate on the intact vessel either side of the
break: lumen and wall intensity distributions from its own foreground, and the
evidence scale from the two intact tails. No trained model, and no threshold that
has to be retuned between scans. `select` then picks a maximum-confidence
**forest** — one continuation per free end, no cycles, T-junction attachments kept
a vessel-width apart — because a candidate cannot see the endpoint it is competing
for. Forest rather than tree: a piece nothing supports connecting stays
disconnected.

Foreground belonging to an unrelated component is **blocked, not penalised**. It is
the cheapest material in the volume by every term in the field, so a shortest path
would dive into a neighbouring vessel and run along it — the exact false connection
this package exists to refuse.

`apply` commits the mask edit, re-derives the centreline from the *edited* mask,
welds it in and stamps per-edge provenance (`ReconnectionOrigin`, `RouteScore`,
`RouteReviewed`), all inside one `graph.batch` so a repair is one press of undo.
The mask records the collapsed observation; the graph carries a
perimeter-equivalent radius in the convention `crosssection.py` already uses.
Those are deliberately two different numbers — conflating them is what silently
re-inflates a specimen.

---

## Notes and limitations

- **A repaired segmentation is an observation; a graph radius is a calibre.** This
  is the one thing to keep straight about `connect --geodesic`, because the two
  numbers disagree by a lot on this specimen and conflating them re-inflates it.
  The mask records the **observed ex-vivo morphology** — collapsed, slit-like,
  whatever the block actually contains — and the repair transports that shape
  across the gap rather than assuming a circular lumen. The graph carries a
  **perimeter-equivalent radius**, `perimeter / 2π`, the same convention
  `crosssection.py` and `radius_perimeter.py` use and the one the SDF surface
  pipeline consumes. So the regenerated **surface may be rounder than the mask**,
  and that is not a bug in either: a surface swept from perimeter-equivalent radii
  is a statement about anatomical calibre, while the voxels are a statement about
  what was measured. Compare a repair against the mask, and a CFD run against the
  radii; do not compare one to the other and read the difference as error.
- **Fully invisible gaps cannot be resolved from topology alone.** Where the image
  gives no support and the continuation is not unique, the honest answers are a
  waypoint or a rejection — there is nothing in the data that prefers one route,
  and a tool that picked one anyway would be inventing anatomy with a confident
  face. Those land in `--review-json`.
- **A painted correction only reaches the surface once it is re-skeletonised.**
  `coronary_sdf` reads the graph and never the voxels, so painting alone changes
  the overlays, the measurements and any export — but not the STL. Press
  *Re-skeletonise painted region* to convert it, or accept that the mask and the
  surface now say different things.
- **Painting is bounded by the paint box** (192 segmentation voxels cubed by
  default, ≈12.7 mm). A `Labels` layer needs a real array and the mask is 2.34 GB,
  so a correction that runs further than that takes more than one pick. The store
  is global; only the window is not.
- **`add` mode needs the painted region to touch existing centreline** at each end
  it should attach to. It follows the skeleton a few radii past what you painted to
  find something to weld to, then trims the overlap back off; if the graph really
  is absent there, both ends come back free and the report says so.
- **The generated skeleton is a candidate, not a replacement.** With 383 false
  bifurcations and 52 components against Avizo's 2, it needs pruning and
  reconnecting before it could carry a surface. `connect` and the `prune_*`
  helpers in `coronary_sdf.pruning` are the tools for that; nothing here does it
  automatically, because deciding which twigs are real is exactly the judgement
  the scoring exists to inform.
- **VesselVio is not on this machine** and is not used. Its own graphs carry only
  two points per edge — endpoints, no interior centreline — which the SDF pipeline
  cannot consume; `skeletonise` implements the same underlying algorithm
  (`skeletonize(method='lee')` → graph → radii from `distance_transform_edt`)
  directly instead.
- Exported STLs come from `rebuild_full`, not from accumulated patches: patches are
  spliced for display and are not welded at their seams.
- The SDF surface is a function of the skeleton graph alone. `coronary_sdf` never reads
  voxel data, so segmentation edits only reach the surface once they are re-skeletonised
  into the graph.
- `preprocess_graph` runs the same chain `run_pipeline` does before meshing (Strahler
  filter, degree-2 contraction, multifurcation collapse, nub pruning, gap bridging,
  densification). Skipping it moves the surface by about a third of a voxel, which is why
  the preview and the export are both fed from it.
- A graph with a complete authored bifurcation taper disables the generic downstream
  segment-radius smoother, bifurcation-shrink repair, radius-transition smoother and
  second carina taper for that preview/export session. Flat bifurcation caps and
  topology-aware SDF blending remain active.
