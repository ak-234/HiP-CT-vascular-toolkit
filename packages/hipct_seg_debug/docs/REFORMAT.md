# Reformat: looking along a vessel instead of through the acquisition grid

Every other view in this package reads the data the way it was acquired. The slice
browser shows raw `(slice, row, col)`; the only oblique sampling that existed before
this was mask-side and nearest-neighbour (`crosssection._PlaneSampler`). A coronary
crosses slices at an arbitrary angle, so **an axial slice cuts it obliquely** and every
cross-section it shows is a smear whose apparent calibre depends on the angle rather
than on the vessel.

The **Reformat** tab builds the other view: a stack of square images sampled on the
planes *perpendicular to the centreline*, one per step along it. Scrolling the stack
walks down the vessel, and each image is a true cross-section.

Nothing here writes anything. There is no output file and no dataset mutation, so
nothing a mis-click can cost you.

> **Where the code is:** `reformat.py` (geometry and sampling, no Qt, no napari),
> `viewer_reformat.py` (the napari window), `controls_reformat.py` (the tab),
> `src/hipct_seg_debug/tiffstack.py` (the strip-windowed reader). Tests in `tests/test_reformat*.py`.

---

## 1. Using it

1. Open the 3D window (`python -m hipct_seg_debug`) and pick the **Reformat** tab in
   the *control* dock.
2. **Double-click a centreline point** in the 3D view, then press **`1`** to add that
   segment to the selection. Repeat for adjacent segments — they are chained into one
   continuous run.

   Or tick **trace between two picks** and give it the two ends: `1` on one, `1` on the
   other, and every segment on the path between them joins the selection. See
   [§1.1](#11-tracing-a-run-from-two-picks).
3. Press **Check geometry** to see what the run will produce. This reads no images and
   takes milliseconds.
4. Press **`4`** (or **Show stack**) to build it. The napari window opens with the
   z-slider running along the vessel.

### Keys bound in the 3D window

| key | action |
|---|---|
| `1` | add the picked segment to the selection (picking it again removes it) |
| `2` | clear the selection |
| `4` | build and show the stack |
| `8` | cancel a half-finished trace |

Digits because **every letter is taken**: `viewer3d` owns `v n b s c i g a r q`, the
edit controller `e z y d t x k j u f`, the Crop tab `m l o h`, and VTK's interactor
binds `p w r q`. `3` is avoided deliberately — it is VTK's stereo toggle, which lives
in the interactor style where `clear_events_for_key` cannot reach it. `5 6 7` are the
Sections tab's, and `bind_keys` *clears* a key before binding it, so taking one of
those would steal it silently rather than bind it twice.

### 1.1 Tracing a run from two picks

Naming a 40 mm run one segment at a time is forty picks, and a single miss is not a
smaller selection but a **broken** one: `chain_segments` splits an interrupted selection
into two runs, and the build then samples only the longest and says so. You get a
shorter stack than you asked for, and the only sign is a note.

**Trace mode** removes the possibility. Tick it, pick one end of the run, pick the
other, and `edit.crop.trace_path` — the same Dijkstra over nodes the Crop tab uses to
name a main vessel — returns the simple path between them. What lands in the selection
is a path, so it is connected by construction and `chain_segments` has nothing left to
split.

* **In path order.** The selection is an ordered list because that order breaks ties in
  the chain walk, and a traced path is already in run order.
* **It extends, it does not replace.** Segments picked by hand before the trace stay,
  and a second trace adds only what is new — so a long run can be traced in two goes,
  or a trace topped up by hand at either end.
* **On a tree there is exactly one path**, so no weighting can change the answer. On a
  graph carrying a fused cycle — two vessels the segmentation joined where they merely
  cross — there is a choice, and **prefer thick** makes it: routing by length in units
  of each segment's own radius takes the fat detour rather than the thin artefact.
* **A failed trace disarms.** Picks in two different components have no path between
  them; that is reported and the pending start is dropped, rather than left to become
  the far end of a trace you have stopped expecting. Leaving the mode, clearing the
  selection and loading another dataset drop it too.

---

## 2. The three things that decide whether the stack means anything

### 2.1 The frame must not twist

The two in-plane axes have to be carried along the centreline with **no rotation about
the tangent**, or the image spins as you scroll and a feature that is standing still
looks like it is moving.

`frames()` uses the **rotation-minimizing frame** (double reflection, Wang–Jüttler–
Zheng–Liu 2008) already in `edit/reconnect/geodesic/shape.py`. It is the only
twist-free frame in the tree. `crosssection._plane_axes` and
`viewer3d.radius_circle_polydata` both pick an arbitrary seed per point, which is fine
for drawing one circle and wrong for a stack.

The difference is not subtle, and is measured in `test_the_frame_does_not_twist_...`:
along a helix, per-step twist is 0.006° for the RMF against 0.38° for an arbitrary
seed — neither of which you would notice. But the seeded error is *systematic*, so it
accumulates to **over 300°** across the run while the RMF stays under 5°.

One consequence: the **seed normal fixes the rotation of the entire stack**, because
the frame transports it without twist. It is deterministic, so a rebuild of the same
path gives the same pictures.

### 2.2 The planes must not intersect each other

This is the constraint that is easy to miss and impossible to see once it has happened.
It is worth doing properly, so this section gives the geometry, the discrete form the
code actually evaluates, and what the pipeline does when the bound is violated.

#### Where the bound comes from

Let the centreline be `p(s)`, parameterised by arclength, with unit tangent
`T = dp/ds`, curvature `κ = |dT/ds|` and **radius of curvature `R = 1/κ`**. The plane
this tool samples at `s` is the *normal plane*

```
Π(s) = { x : (x − p(s)) · T(s) = 0 }
```

Take the osculating circle at `s` — the circle of radius `R` that matches the curve to
second order. **Every normal plane of a circle contains its centre.** So two normal
planes at `s` and `s + ds` both contain that centre, which means their line of
intersection passes through a point at distance exactly `R` from the curve, on the
concave side.

That is the whole result. A square of half-width `h` centred on the curve inside the
normal plane reaches out to `h`, so it touches its neighbour's plane exactly when
`h` reaches the intersection line:

```
    h  <  R          free of self-intersection
    h  ≥  R          the planes fold through each other
```

Past the bound, the same tissue appears in two images and the tissue that should have
been between them appears in none. (In general the envelope of the normal planes is the
curve's *focal surface*, the locus of centres of curvature; `R` is its distance along
the principal normal.)

#### The discrete form the code evaluates

Given resampled points `pᵢ` and the frame's own unit tangents `Tᵢ`:

```
    dsᵢ  =  ‖pᵢ₊₁ − pᵢ‖                       arclength step
    θᵢ   =  2·arcsin( ‖Tᵢ₊₁ − Tᵢ‖ / 2 )       turn between consecutive tangents
    Rᵢ   =  dsᵢ / θᵢ                          local radius of curvature  (∞ if θᵢ ≈ 0)
```

`Rᵢ = dsᵢ/θᵢ` is exact for a circular arc, where `θ = ds/R` by definition of radian
measure. So the criterion becomes a comparison with no division at all:

```
    h · θᵢ  <  dsᵢ
```

Each *point* is then bounded by whichever of the two steps meeting at it turns harder,
which is the conservative choice:

```
    R_point[i]  =  min( R_step[i−1] , R_step[i] )
```

and the test carries a safety factor (`safety`, default **0.8**) because `R` is
estimated from a discrete polyline, so meeting the bound exactly means meeting it only
to within the estimate's own error:

```
    h  ≤  safety · R_point[i]        for every i
```

`collision_free()` returns the worst ratio `max( h / (safety·R) )`; 1.0 is the boundary.

#### Two implementation details that change the numbers

- **The two end steps are discarded** and replaced by their neighbours. The frame's end
  tangents come from a one-sided difference, i.e. a chord, whose direction is the
  tangent at the *midpoint* of the step rather than at its end — so the turn measured
  across the first step is only half the real one and `Rᵢ` comes out at **twice** the
  truth. That is an *optimistic* bound at precisely the two places a path is most likely
  to have been cut mid-bend. Measured on a 5 mm arc: 9,999 µm reported against a true
  5,000 µm.
- **The turn uses the half-chord arcsin form.** `‖Tᵢ₊₁ − Tᵢ‖ = 2·sin(θ/2)` for unit
  vectors, so `θ = 2·arcsin(‖ΔT‖/2)` is an exact identity. This is *defensive rather
  than a fix*: `arccos(Tᵢ·Tᵢ₊₁)` is ill-conditioned near 1, with relative error growing
  like `ε/θ²`, but in float64 at the turn angles this data produces (a 33 µm step on a
  2 mm radius gives θ = 0.0165 rad) the two agree to **4e-14**, and `arccos` only costs
  more than 1% of `R` below θ ≈ 3e-8 — a radius of curvature of a kilometre. It would
  matter if the tangents were ever float32, where `arccos` is already 2.3% out at a
  milliradian.

#### How reformat handles a violated bound

A voxel-derived centreline turns a **median 19.5°** between consecutive points, which
puts `R` at a fraction of a voxel and makes *every* usable half-width illegal. So the
bound is not a check that passes or fails — it is something the pipeline actively works
to satisfy, in four stages:

**1 — Resample to uniform arclength.** Done first, so `ds` is a single physical number,
the test is a per-index comparison, and the smoothing window below means the same thing
at every point.

**2 — Smooth until the bound is met.** Up to `smooth passes` iterations of

```
    window ← max( 1.6 · window , 3 · step )        (window starts from the graph's own
                                                    spacing and radii, or your value)
    p ← gaussian(p, window)                        arclength-parameterised, σ = window/2
    resample, re-frame, re-measure
```

The smoothing is arclength-parameterised (`exp(−½(Δs/σ)²)` over `|Δs| ≤ window`) rather
than index-based, because a voxel skeleton steps √3 times further on a diagonal than on
an axis — an index window would be a physically variable one. It runs on the **whole
concatenated path**, not per segment: the kink at a junction seam is exactly where the
bound bites, and per-segment smoothing pins both sides of it.

Two guards on this stage, because smoothing is how the bound gets *cheated* rather than
met:

- each point's displacement is capped at `max_move_frac · r` (default **half its own
  radius**), so smoothing cannot drag the sampling centre out of the lumen;
- the total displacement is always reported (`moved a median X µm, at most Y µm`). A
  curvature target met by relocating the vessel is not a curvature target met.

`_gaussian_smooth` **pins both end points** to their nodes, so a kink in the first or
last pair survives every pass. The loop detects that it has stopped changing anything
and stops early rather than burning the remaining budget.

**3 — Clamp what is left.** Per plane, `h_used = min(h_wanted, safety·R)`, counted and
reported. In `fixed` mode the tightest bend sets one width for the whole stack, since
keeping a per-plane width there would quietly turn it into `radius` mode.

> **`native` mode does not clamp** — see §3. With the pitch pinned to the voxel,
> shrinking `h` would shrink the *pixel* rather than the view, which is the
> magnification that mode exists to prevent. There the bound is reported instead, with
> the `size_px` that would stay inside it.

**4 — Verify directly.** `planes_disjoint()` does not trust the curvature proxy; it
checks the squares themselves, and covers both ways two images can show the same tissue:

- squares that genuinely **cross** — each has corners on both sides of the other's
  plane;
- near-parallel squares that **coincide** — parallel planes never cross however close
  they are, so a crossing test alone is blind to a hairpin, whose two limbs come back
  *anti*-parallel. Pairs close along the vessel are exempt, since overlapping there is
  the design; only pairs whose arclength separation exceeds their combined reach are
  judged.

This last check is **advisory**. The curvature bound is local — it certifies
neighbouring planes and cannot see a vessel folding back on itself at low curvature.
Nothing can sample both limbs of a hairpin without showing the same tissue twice, so
smoothing does not help and the only honest remedy is a narrower `h`, which is your
decision to make.

### 2.3 The grid must not ask for more than the data has

A plane's pixel pitch is a free choice, and choosing it finer than the voxel reveals
nothing — it enlarges the interpolation kernel. This was the original cause of
"the sections look blurry": in `radius` mode at the median graph radius (247 µm), a
129 px frame samples at 15.4 µm/px against a **33 µm voxel** — 2.1× magnified, 3.3× at
the 5th percentile.

`native` mode exists for this and is the default. See §3.

Whichever mode is in use, the report states the factor rather than leaving a soft
picture unexplained.

---

## 3. Plane size: the four modes

The modes differ in **which of the three quantities is derived**. Half-width, pixel
count and pixel pitch are related by `pitch = half / (size_px // 2)`; fix any two and
the third follows.

| mode | you fix | derived | can it magnify? |
|---|---|---|---|
| **`native`** *(default)* | pixel count | `half = (size/2) × voxel` | **structurally no** |
| `radius` | pixel count | `pitch = k·r / (size/2)` | **yes** — a small radius means a tiny pixel |
| `fixed` | pixel count | `pitch = k·r_max / (size/2)` | **yes** |
| `manual` | half + pitch | pixel count | only if you ask it to |

### `native` — a fixed frame at the acquisition's own resolution

The pitch is pinned to one raw voxel and the **half-width follows from the frame
size**, rather than the other way round. One output pixel is one voxel by
construction, whatever the vessel is doing. The frame is a *window on the data* rather
than an enlargement of it.

`native_scale` samples at that many voxels per pixel — 2.0 gives a wider view at half
resolution. That undersamples; it never magnifies.

> **`native` does not apply the curvature clamp, and that is deliberate.** In the
> width-driven modes clamping is right because the pitch is derived from the
> half-width. Here the pitch is pinned, so shrinking the half-width would not shrink
> the field of view — it would shrink the *pixel*, which is precisely the magnification
> the mode exists to prevent. So the bound becomes a **report**: how many planes exceed
> it, and what `size_px` would stay inside. Reducing the frame is your decision.

### `radius`

Half-width is `radii_k` × the *local* radius on a fixed pixel grid, so a capillary and
an artery both fill the frame. The physical scale then varies down the stack — which is
why the napari window cannot show a physical ruler in this mode (§6).

### `fixed`

One half-width for the whole stack, `radii_k` × the largest radius on the path.
Physically comparable end to end, at the price of a mostly-empty frame wherever the
vessel is small. The tightest bend clamps it for everyone; keeping a per-plane width
here would quietly turn it back into `radius` mode.

### `manual`

`half_um` and `px_um` exactly as given; `size_px` follows.

---

## 4. Every control on the tab

### Selection

| control | what it does |
|---|---|
| **selected segments** | the ordered list. Pick order is kept, because it breaks ties when a chain walk is ambiguous. Segments that will not be part of the built stack are marked **`[not in the run]`** — see below. |
| **Add pick (1)** | adds the segment under the last 3D pick. A **toggle** — picking the same segment again removes it, since that is far more likely a correction than a request to add it twice. |
| **Clear (2)** | empties the selection. |

Selected segments are chained into **one continuous run**. What decides the topology is
each node's degree *within the selection*, not in the tree — so picking two segments
either side of a bifurcation is a chain, even though the node between them has degree 3.
A node joining more than two selected segments has no total order through it, so the run
is split there and the node is named, rather than guessed at.

### What happens if the selection is not one chain

**A stack is one continuous path by definition** — the slider walks *along* a vessel —
so a selection spanning two disconnected vessels can only build one of them. Nothing is
refused and nothing fails: `chain_segments` splits the selection into maximal runs,
longest first, and the build takes the first.

Because that silently leaves segments out, all three views of the selection say so:

- **the segment list** marks each one `[not in the run]`. *Marked*, not removed or
  reordered — deselecting something **else** may well bring it back into the run, so
  the mark is advisory rather than a rejection;
- **the 3D preview** draws the run that will be sampled in cyan and everything else in
  a dimmed slate (§7), because drawing them alike would show you one thing while the
  build did another;
- **the report** names both halves explicitly (§8).

All of it is available from **Check geometry**, before any image is decoded. If you
want the other vessels too, reformat them as separate stacks.

### Plane geometry

| control | what it does |
|---|---|
| **size** | the mode (§3). `native` is the default. |
| **px** (`size_px`) | the output frame, in pixels. Forced odd, so the centreline lands on an exact centre pixel rather than between two. |
| **interpolation** | spline order for the raw greyscale: nearest (0), linear (1), **cubic (3, default)**, quintic (5). See §5.1. |
| **Match voxel** | sizes the grid to the data. In `native` the pitch is already matched, so it sets the *frame* to about `radii_k` median radii across. In the other modes it sets `size_px` so `um/px` lands near one voxel — computed from the half-width the build will **actually use**, including any curvature clamp. |
| **half-width (radii)** (`radii_k`) | the multiplier for `radius` and `fixed`. Inert in `native` (except as the "how much context" hint for Match voxel) and in `manual`. |
| **half um** / **um/px** | `manual` mode only. |

### Path

| control | what it does |
|---|---|
| **step um** | the spacing **along** the vessel between consecutive planes — the stack's third dimension. `0` means one raw voxel, which is where the information runs out. Sets the plane count (`n ≈ length/step + 1`), the `max_planes` refusal, the tolerance for a graph point counting as "in" a plane (`step/2` along the tangent), and the floor for the smoothing window (`3 × step`). See §5.3 for why it is a weak lever on runtime and *not* a lever on plane collision at all. |
| **curvature safety** | keeps the half-width this far inside the curvature bound (default 0.8). `R` is estimated from a discrete polyline, so meeting the bound exactly is meeting it only to within the estimate's own error. |
| **smooth passes** | how many times the smoothing loop may widen its window and retry (default 8). |
| **smoothing window um** | `0` derives it from the graph's own point spacing and radii (`skeleton_optimise.default_smooth_window`). Set it to force a particular window. |

The smoothing loop widens by ×1.6 per pass and re-checks. It is arclength-parameterised
(so the window is a physical distance, not a point count) and runs on the **whole
concatenated path** rather than per segment — the kink at a junction seam is exactly
where the bound bites, and per-segment smoothing pins both sides of it.

Each point's displacement is capped at half its own radius, so smoothing cannot drag
the sampling centre out of the lumen. And `_gaussian_smooth` **pins both end points**,
so a kink in the first or last pair survives every pass; the loop detects that it has
stopped changing anything and says so rather than burning its remaining budget.

### Output

| control | what it does |
|---|---|
| **segmentation overlay** | also sample the mask on the same planes, as a napari Labels layer. Nearest-neighbour, never linear — a label is not a quantity to average, and an interpolated 0.5 would draw a boundary that is not in the mask. |
| **draw the section stack in 3D (slower)** | draw a couple of dozen cross-sections as textured quads in the 3D window. Off by default: it costs a texture upload per plane. The *current* section always follows the slider regardless (§7). |
| **Check geometry** | everything except the image sampling — chaining, smoothing, the clamp, the disjointness check. Milliseconds, no decode. This is why the geometry is separable from the sampler: you learn the half-width will be clamped to a third of what you asked *before* committing to a long build. |
| **Show stack (4)** | build and open. Runs on a worker thread; the napari window is created on the Qt thread when it lands. |

### Save and load

Building is the expensive part — a 26 mm run at 61 px is ~21 s, a 47 mm run at 129 px
~37 s, almost all of it TIFF decode. A saved stack **loads in about 0.05 s**, so looking
at the same vessel again costs nothing.

| control | what it does |
|---|---|
| **name** | the base. The rest of the filename is appended for you, and the hint underneath shows the whole thing *before* the dialog opens. |
| **format** | `.npz (one file)`, `folder: TIFF + JSON`, or `folder: .npy + JSON`. |
| **Save stack…** | disabled until there is a stack. Asks where to put it every time — nothing is written unless you ask, and there is no remembered directory to overwrite by surprise. |
| **Load stack…** | **enabled with no dataset open at all**, which is the case the feature exists for. One dialog serves all three formats: pick the `.npz`, or a folder's `geometry.json`. |

The name carries what decides whether two stacks are comparable:

```
LAD__native_61px__seg306          one segment
LAD__radius_129px__seg0-1-2       a short chain
RCA__fixed_129px__seg12+7more     a long one collapses to a count
```

Mode and frame size first, segments last because that is the part that can run long. The
segment ids are those of the run **actually sampled**, not of the selection — those
differ whenever the selection was not one connected chain (§4), and a filename that
describes the request rather than the file is worse than none.

**What is saved is not just the images.** The overlay readout and the 3D plane frames all
come from the `Centreline` and `PlaneGeometry` beside the arrays, so those travel with
them; a reloaded stack is the same object the builder produced. The tests assert this by
comparing `describe()` and the 3D plane corners, not only the pixels.

#### Choosing a format

| | when |
|---|---|
| `.npz` | the default. One file to move or attach, nothing to get out of sync. |
| folder: TIFF + JSON | when the stack is going somewhere else. `raw.tif` and `mask.tif` open **directly in Fiji** with no script; `README.txt` beside them says what the axes are and gives both scales, which is exactly what Fiji cannot tell you. |
| folder: `.npy` + JSON | fastest to load and exact dtypes, but no better for Fiji than the npz. |

Measured on a real 707-plane stack: 4.7 MB as npz, 5.5 MB as a TIFF folder, all three
byte-exact on reload.

#### What happens when a stack does not match what is open

Three tiers, and the difference between them is the point:

- **A different schema raises.** The file cannot be understood, and a stack silently
  misread is worse than one that will not open.
- **A different coordinate frame is a note, and the 3D overlays are skipped.** The images
  are self-contained and worth looking at; their *world* positions are not meaningful
  here, so plane frames would be drawn somewhere unrelated and look plausible doing it.
  The napari window opens as normal and the summary says why the 3D layers are absent.
- **A changed graph is a note only.** Reformatting a repaired graph is the intended
  workflow. Provenance records the segments twice — by id, and by the reversal-invariant
  geometric key from `edit/crop.py` — so a stack can still say which vessel it is after a
  repair has renumbered things.

---

## 5. How the sampling works

### 5.1 Interpolation

Cubic B-spline (`order=3`) by default. Measured by a round-trip on a real slice —
box-average 2× to a coarse grid, restore onto the original grid, compare to truth:

| order | RMSE | vs linear | edge energy recovered |
|---|---|---|---|
| 1 linear | 234.9 | — | 55.2% |
| **3 cubic** | **173.3** | **−26%** | **72.6%** |
| 5 quintic | 168.3 | −28% | 75.4% |

Quintic buys 2% more for three times the cost. Windowed sinc (as ITK offers) would land
near that same ceiling — which, with cubic already at 73%, is not worth a dependency.

> **The pad is not cosmetic above order 1.** `map_coordinates` prefilters for
> `order ≥ 2`, and the spline prefilter is an **IIR filter** — it is *not local*, so a
> per-block prefilter differs from a whole-volume one well in from the block edge.
> Measured at order 3: pad 1 leaves 4.3e-3 of error at every seam, pad 8 leaves 3.7e-6,
> pad 16 is exact. `SPLINE_PAD` carries the per-order table and
> `test_chunking_does_not_change_the_answer` is the guard.

The one genuine exception: a block clipped at the **true volume boundary** cannot carry
its pad, so its prefilter differs there. That is at the edge of the data, where there is
nothing to recover anyway.

### 5.2 Reading the images

The raw stack is ~92 GB read one TIFF at a time, so the sampler is organised around
*which slices a run of planes needs*, not around the planes. A contiguous run shares a
small `(slice, row, col)` box; the run grows until that box would exceed the memory
budget, is read once, and every sample in it is taken with a single `map_coordinates`
call.

Because `WorldFrame.um_to_raw` is a pure per-axis scale with an axis reversal, the raw
box of a sampled plane is *exactly* the box of its four corners — so the planner costs
O(N) rather than O(N·size²), and is exact rather than conservative.

**The decode used to dominate everything.** `tifffile.imread` decodes a whole 18 MB LZW
page (188 ms) and then discards everything outside a ~130 px window. But these files are
striped with `rows_per_strip = 1` — 3079 independently compressed strips per page — so
`TiffStack.read_window` now decodes only the strips it needs:

| read | time | speedup |
|---|---|---|
| full page | 188.5 ms | — |
| 130-row window | 10.0 ms | **18.8×** |
| 400-row window | 22.8 ms | 8.3× |

Bit-exact against the full decode. It falls back to the whole page for tiled files,
non-TIFFs, or a window covering most of the rows, and reports how many windows took
which path — a directory that quietly fell back is an order of magnitude slower and
otherwise looks identical.

### 5.3 What `step_um` costs

Measured on a real 26 mm segment in `native` mode:

| step | planes | build | R_min | decode | interp |
|---|---|---|---|---|---|
| 16.5 µm | 1415 | 19.9 s | 1320 µm | 13.5 s | 2.5 s |
| **33.0 µm** | 707 | 16.2 s | 1582 µm | 13.5 s | 1.9 s |
| 66.0 µm | 355 | 15.3 s | 1312 µm | 13.4 s | 1.5 s |
| 132.0 µm | 178 | 14.7 s | 1545 µm | 13.3 s | 1.3 s |

Two things follow.

**Halving the step does not halve the speed.** Decode is flat — the planes cover the
same z range whatever their spacing, so the same strips get read either way. Only
interpolation scales, and it is the small term.

**Step is not a lever on plane collision.** `R_min` wanders with no trend, because
`R = ds/θ` is scale-invariant on a smooth curve: both terms shrink together. Use the
half-width (or `size_px` in `native`) for that.

---

## 6. The napari window

Its own viewer, separate from the slice browser: the arrays live in
`(plane index, v, u)`, and every placement rule in `viewer2d` is built around raw
`(slice, row, col)`.

**Axis 0 stays an integer plane index.** napari's slider steps in integers, so giving
axis 0 a physical scale would put the readout half a step off the plane actually shown —
the failure `viewer2d._whole_slices` documents at length. Arclength goes in the text
overlay instead, which is also the more honest number: the resample pins both endpoints,
so the step is only nominally uniform.

**The two in-plane axes are scaled when they can honestly be scaled.** In `native`,
`fixed` and `manual` every plane shares one pitch, so the layers carry a real `scale`
and `translate` and the cursor readout is micrometres from the vessel centre. In
`radius` mode the pitch changes plane to plane and napari's `scale` is a single constant
per axis — there is no honest value to give it, so the axes stay in pixels and the
per-plane pitch goes in the overlay. Quietly applying one plane's scale to the whole
stack would be exactly the kind of lie the rest of this package works to avoid.

Layers: `reformat (raw)`, `reformat (mask)`, `graph points` (other graph points falling
in the plane — a branch leaving the vessel shows as a mark walking out of frame over a
few planes), and `centreline` (the fixed centre mark, which is what lets the eye see the
lumen drift off it — the first sign of a tangent that is not quite right).

The overlay reads:

```
plane 412 / 707    s = 13.60 mm    segment 306
centre  x 41,203  y 18,772  z 92,431 um
r_graph 640 um    half 990 um    33.0 um/px = 1.0x the 33.0 um voxel    R_curv 2,302 um
```

---

## 7. The 3D window

Three layers, all in the *layers* dock, none of them pickable — a quad lying across the
vessel would swallow the double-clicks the selection is built from.

| layer | default | what it is |
|---|---|---|
| `reformat: selected run` | on | the selection, in **two colours**: cyan for the run that will actually be sampled, dimmed slate for selected segments not connected to it. One layer row, two actors. |
| `reformat: plane frames` | on | the outline of every plane. One actor whatever the plane count, and **the only place plane collision is visible** — squares fanning through each other on the inside of a bend. |
| `reformat: current section` | on | the cross-section the napari window is showing, textured, with an outline (at a glancing angle the quad is nearly edge-on, which is when you most need to see where it is). |
| `reformat: section stack` | off | a decimated set of sections as textured quads, when the checkbox asks for it. |

**Scrolling the napari stack moves two things in 3D**: the axial image plane goes to the
raw slice this section was cut from, and the current-section quad walks down the vessel.
Only that one actor is rebuilt per slider step.

---

## 8. Reading the report

A finished build prints, e.g.:

```
707 planes over 22.91 mm at 33.0 um; curvature: tightest radius 2,302 um at plane 53
planes: 61x61 px, half-width 990-990 um (33.0-33.0 um/px), mode 'native'
  sampling 1.00-1.00x the 33.0 um voxel
sampling: 1 block(s), 348 slice reads over 348 distinct slices
  order 3; decode 13.5s, interpolate 1.9s
```

| line | what to look at |
|---|---|
| **curvature** | the tightest radius on the run. If the half-width is near it, the planes are near folding. |
| **sampling Nx the voxel** | above ~1.5 the section is *magnified rather than resolved*. This is the line that explains a soft picture. |
| **half-width clamped at N planes** | the curvature bound cut the field of view. Because `size_px` is fixed, that turns straight into magnification in the width-driven modes — on one real segment a 982 µm request clamped to 181 µm, which at 129 px is 2.8 µm/px: **11.65× magnified**. |
| **the fixed frame reaches past the curvature bound** | `native` only. Not clamped; reduce `size_px` to the suggested value if those planes matter. |
| **slice reads over distinct slices** | a gap between them is re-reading at block seams. |
| **whole-page decode** | the strip reader fell back. Expect ~19× the cost. |
| **decode / interpolate** | which half to attack. Decode responds to how the TIFFs are stored, interpolation to the spline order. |
| **the selection is N separate runs** | the selection was not one chain (§4). |
| **built the longest run only** | which segments went into the stack, and which were left out. |
| **notes** | smoothing passes, move limits, disjointness failures. |

A selection spanning three vessels reports both halves of that decision — *that* it
split, and what was done about it:

```
note: the selection is 3 separate runs (2 seg / 0.1 mm, 1 seg / 0.0 mm, 1 seg / 0.0 mm)
note: built the longest run only: segments [0, 1], 0.06 mm. Segments [2, 3] are not
      connected to it and were not sampled -- reformat them separately.
```

The first line comes from the chaining, the second from the build. Both are needed: the
first alone tells you the selection was odd without telling you which segments you are
actually looking at.

---

## 9. Known limits

- **The disjointness check is advisory.** A hairpin genuinely has two limbs; nothing
  can sample both without showing the same tissue twice. Narrow the half-width.
- **Smoothing moves the centreline.** Meeting a curvature target by relocating the
  vessel is not meeting it. The displacement is capped at half a radius per point and
  always reported (`moved a median X um, at most Y um`) — read it.
- **A kink at the very first or last point cannot be smoothed out**, because the ends
  are pinned to their nodes. Only the clamp helps there.
- **`radius` mode has no honest in-plane ruler** (§6).
- **No parallel decode.** The strip reader is the structural win; a thread pool over
  slices would likely give several × more, since LZW decode releases the GIL. Deferred
  deliberately — there is no precedent for I/O parallelism in this package, and the
  strip change should be measured on its own first.
- **ITK is not used and would not help.** The bottleneck is TIFF decode, which ITK
  would pay identically; its `sitkBSpline` is the same B-spline scipy already provides;
  and its resampler wants an image object, so an oblique reformat through a 92 GB stack
  would need the whole volume resident or a streaming pipeline — a worse fit than the
  block sampler that already exists.
