# Collapse-aware geodesic reconnection

An opt-in, training-free connector that repairs the spatial graph and the
segmentation together. It replaces the greedy voxel walk of DPC-style
reconnection (arXiv:2504.01597) with a global, direction-aware path search over
image evidence, designed for the deflated tube-, ribbon- and slit-like vessels
that ex-vivo HiP-CT coronary data actually contains.

Enable it with `edit connect --geodesic`. Everything below is off by default.

## Why not a greedy walk

A collapsed vessel is not a tube. Its cross-section is a slit, its centre is not
its brightest point, and the gap between two free ends is frequently longer than
the local radius. A greedy walk from one endpoint commits to a direction before
it has seen the evidence that would justify it, and cannot revisit that choice.

The search here is global over a region of interest, carries arrival direction in
the state so it can be penalised for turning, and is scored against alternatives
so that an ambiguous reconnection can be refused rather than guessed.

## Pipeline

| Stage | Module | What it does |
|---|---|---|
| Component index | `edit/reconnect/geodesic/components.py` | 26-connected labelling of the segmentation, streamed |
| Classification | `edit/reconnect/geodesic/classify.py` | four-way split of what needs repair |
| Cost field | `edit/reconnect/geodesic/cost.py` | self-calibrated, per-ROI image evidence |
| Search | `edit/reconnect/geodesic/astar.py` | orientation-aware A\*, coarse-to-fine |
| Scoring and gates | `edit/reconnect/geodesic/route.py` | accept / review / reject with a reason |
| Selection | `edit/reconnect/geodesic/select.py` | maximum-confidence spanning forest |
| Apply | `edit/reconnect/geodesic/apply.py` | transactional write-back with provenance |

### Component index

The segmentation lattice is `HxByteRLE`-encoded and around 2.3 GB decoded, so it
is never decoded whole. `components.py` labels run intervals directly, streaming
one plane at a time with two planes resident, and unions them with path-halving
union-find. On the reference dataset this is ~13 s and a 9.9 MB run table, and it
reproduces `scipy.ndimage.label` exactly.

`ComponentIndex` exposes `label_at`, `labels_at`, `nearest_label`, `window`,
`voxels`, `boxes`, `sizes`; built with `build(labels, progress, z_range)` or
`from_array(mask)` for tests.

### Classification

Every candidate is one of four things: **reskeletonise** (mask present, skeleton
missing), **geodesic** (a genuine gap to route across), **fragment** (a detached
piece of vessel), or **unassociated** (debris). `MIN_FRAGMENT_ELONGATION = 2.0`
and `DEBRIS_VOXELS = 12` separate the last two.

> **Known limitation.** `fragment_candidates()` tests whole components. Measured
> against the reference dataset, undescribed mask amounts to 354 radius-scaled
> lobes against 161 skeleton free ends, but only 19 orphan *components* (1.1%) —
> so a component-level test misses roughly 95% of the fragments. A lobe-level
> test is the open work here.

### Cost field

Cheap where the image supports a vessel, infinite where passage is blocked, and
deliberately *expensive* where a centreline already exists:

```python
cost = BASE_COST + (1.0 - clip(support, 0, 1))
cost += redundancy_weight * described * mine
cost[blocked] = inf
```

`support` combines a gamma-normalised multiscale Hessian tube/ribbon/sheet
response with flux medialness, calibrated per ROI on intact vessel tails — there
is no trained model. `described` is 1 on an existing centreline and decays over
one local radius, so `REDUNDANCY_WEIGHT = 0.6` pushes routes off centrelines that
are already explained.

Two sign conventions here are easy to get backwards and fail silently:

- **Flux medialness is `+div`, not `-div`.** For a dark lumen the unit-gradient
  field *diverges* from the axis, which is opposite the classical bright-vessel
  form. With the classical sign, medialness reads zero inside every vessel; with
  the correct sign the inside/outside ratio is 22–64×.
- **The no-raw surrogate has the opposite polarity.** When raw data is absent the
  mask is used as a surrogate, and it is foreground-*bright*. It must not inherit
  `dark_lumen`.

A related trap: restricting the allowed foreground to endpoint neighbourhoods
looks like a useful narrowing, but it blocks exactly the pruned branches that
routes need to pass through. The correct arrangement is the one above —
described foreground expensive, undescribed foreground cheap — not a hard
restriction.

### Search

`astar.py` searches a padded grid of `inf` with flat integer states
(`voxel * n_dir + dir`), expanding one vectorised numpy pass at a time. Carrying
arrival direction lets `turn_weight` penalise wandering.

A naive implementation took over 120 s per ROI. The padded-grid rewrite plus an
orientation-free coarse pass, with `coarse_threshold` lowered from 250 000 to
20 000, brings it to about 2.9 s.

`routes()` returns alternatives with `SUPPRESSION = 6.0`, setting
`suppressed_cost = inf` when no alternative exists.

### Gates and confidence

Gates apply to the **route**, not to the proposal that generated it — applying
them to the proposal was the cause of visibly wrong accepted paths, together with
a 21× discount inside allowed foreground. `tortuosity_max` is re-applied to the
route and rejects with a readable reason:

```
the route wanders: 4180 um across a 900 um gap (tortuosity 4.64, above 1.8)
```

Confidence renormalises over the terms actually measured: with
`--alternatives 1` there is no uniqueness term, and dropping it avoids forcing
every candidate into review. Support is clipped at 1.0 rather than scaled.

An ambiguity margin of 15% separates accept from review.

### Apply

Transactional: the graph and segmentation are written together or not at all.
Provenance lands as per-edge Amira scalar fields — `ReconnectionOrigin`,
`RouteScore`, `RouteReviewed` — and `graphmodel.set_segment_attrs(sid, attrs)`
makes that undoable (it refuses topology keys).

## Jump detection

The connector only has work to do once gaps are *found*. On the reference dataset
the stock graph yields **zero** proposals; `flag-interpolation` is what creates
the work.

Amira's own interpolated points were already flagged, but a second signature was
missing: artificial points inserted to bridge a gap *within* one segment, which
carry no interpolation flag. `edit/interpolation.py` adds it:

```python
UNSAMPLED_JUMP = 8
JUMP_FIELD = "unsampled_jump"   # separate from FIELD: protects measurement semantics
JUMP_MIN_UM = 600.0
JUMP_STEP_RATIO = 5.0
```

`Span` gains `is_jump` (`n_points == 0`) and `anchors`. Jumps are resolved through
anchor *point* ids rather than segment ids, because `split_segment` retires a
segment id and the point-ids guard is vacuous for a zero-point span — without
that, a second jump in the same segment was silently dropped.

On the reference dataset: 24 jumps over 72.4 mm, cutting 2 components into 28 and
yielding 18 proposals. Raw image evidence lifts acceptance from 4 to 11.

## Viewer

Jumps and routes render in the 3D view: `JUMP_COLOR #ff453a`,
`ROUTE_COLOR #ff375f`, `ALTERNATIVE_COLOR #ff9f0a`, `WAYPOINT_COLOR #32d74b`.
`centreline_polydata(graph, flagged=None, breaks=None)` breaks *after* point *i*,
keeping both anchors.

> `pv.PolyData(points)` creates one vertex cell per point, so `n_cells != n_lines`.
> Guards must test `n_lines`.

The Reconnect tab (`controls_reconnect.py`) shows each candidate with its
post-selection status — `final_status()` and `decision_note()` exist because the
panel otherwise reported pre-selector counts (13 accept where the selector had
settled on 11).

## CLI

```
edit connect --geodesic --out GRAPH.am --out-seg SEG.am
             [--review-json F] [--decisions-json F]
             [--max-unsupported-factor 4] [--alternatives 3]
             [--no-tjunction-geodesic] [--min-jump-um 600] [--no-jumps]
```

## Deferred

- **Lobe-level fragment detection** — see the limitation above.
- **Blocked-remnant handling.** Measured as not firing on the reference dataset
  (2 of 18 corridors, both real vessels, both accepted), so it was not worth
  fixing on that evidence.

## Related

A `TiffStack` bug found during this work affects every `--raw` consumer, not just
this one: globbing `*.tif*` matched five non-TIFF sidecar files that sorted into
indices 1–5, shifting every slice by 5 (165 µm in z). Fixed by filtering on
`TIFF_SUFFIXES = (".tif", ".tiff")`. The damage was confined to slice lookup —
`WorldFrame.from_inputs` yields identical `bin_factor` and `raw_start` either way.

## Tests

`test_geodesic.py` (42), `test_geodesic_components.py` (14),
`test_geodesic_cli.py` (14), `test_geodesic_panel.py` (19),
`test_tiffstack.py` (4), shared fixtures in `conftest_geodesic.py`, plus
additions to `test_interpolation.py` (32) and `test_viewer3d_swap.py` (43).
