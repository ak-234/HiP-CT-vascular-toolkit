# `epicardial_annotation.py`

Build a **series of coronary tree models**, each keeping a different level of side branches,
to study the effect on flow in the **main epicardial vessels** (LAD, LCx, RCA…). You annotate
the main vessels **once** in an interactive 3D picker; they are then **always preserved**,
while side branches are pruned at a sequence of radius thresholds. For each model the script
writes a pruned Amira spatial graph, generates the SDF lumen surface, and computes the
Van der Giessen flow split so you can see how much flow reaches each main vessel.

It lives inside the `coronary_sdf` package and reuses its parser, topology, smoothing,
surface pipeline, and flow-split code. It does **not** modify any of those shared modules.

---

## What it produces

For each radius ratio you request, one model is written under `--out`:

```
<out>/
  epicardial.json                # annotation sidecar (which segments are main vessels)
  summary.csv                    # one row per (ratio, vessel): ostial/distal/retained flow
  pruned_branches.csv            # every pruned take-off branch across all ratios + its radius
  vessel_radius_stats.csv        # per-vessel radius min/mean/max: total, proximal, distal
  ratio_10/
    model.am.xml                 # pruned spatial graph (ALL trees, combined)
    main_vessel_flow.json        # per-vessel Giessen flow for this model
    pruned_branches.csv          # take-off branches pruned at THIS ratio (id, radius, ...)
    surface/                     # SDF lumen meshes (lumen_bspline*.stl/.vtk, one per tree)
  ratio_8/  ratio_6/  ratio_4/  ratio_2/   ...
```

---

## Requirements

- Run as a module from the parent of `coronary_sdf`: `python -m coronary_sdf.epicardial_annotation …`
- Dependencies are those of `coronary_sdf`: **numpy**, **scipy**, **pyvista** (+ the SDF
  surface pipeline deps).
- The interactive picker needs a **display** (it opens PyVista windows). Only the first
  annotation run needs it; later runs reuse the saved sidecar headlessly.

---

## Quick start

**1. First run — annotate the main vessels** (opens the picker, saves the sidecar):

```
python -m coronary_sdf.epicardial_annotation ^
  --out "%CORONARY_SDF_OUTPUT_DIR%\Epicardial_Proximal_Radius_Ratio_Prune" ^
  --ratios "10,8,6,4,2" --pick
```

**2. Later runs — regenerate from the saved annotation** (no picker, no display):

```
python -m coronary_sdf.epicardial_annotation ^
  --out "%CORONARY_SDF_OUTPUT_DIR%\Epicardial_Proximal_Radius_Ratio_Prune" ^
  --ratios "10,8,6,4,2"
```

Add `--no-pipeline` to skip the (slow) surface generation while you iterate on the
annotation and check flow numbers.

---

## CLI reference

| Flag | Default | Meaning |
|------|---------|---------|
| `--xml` | `config.INPUT_XML` | Input Amira `.am.xml` spatial graph. |
| `--out` | `epicardial_models` | Output directory. |
| `--sidecar` | `<out>/epicardial.json` | Annotation JSON path. |
| `--ratios` | `2,5,10` | Comma-separated **denominators**. Threshold for a model = `R_ostial / d`, i.e. `2` → ½, `10` → 1⁄10 of the proximal radius. Processed in the order listed. |
| `--pick` | off | Force the interactive picker to (re)run even if a sidecar exists. |
| `--no-pipeline` | off | Write XML + flow only; skip SDF surface generation. |
| `--prune-unattributed` | off | Also drop branches/trees with no annotated ancestor (default keeps them). |
| `--frac` | `0.8` | Containment fraction for the swallowed-leaf prune (see below). |
| `--ostium-skip` | `BIF_SKIP_POINTS` (10) | Contours skipped at the ostium when measuring a branch radius (avoids the inflated junction). |
| `--radius-navg` | `FRAMES_DOWNSTREAM` (10) | Downstream contours averaged for a branch radius. |
| `--no-hover` | off | Disable the picker's green hover-preview highlight. |

---

## How it works (pipeline overview)

1. **Parse + preprocess once.** `parse_xml` then `flow_fractions.preprocess_topology`
   (Strahler filter → degree-2 contraction → split-multifurcation merge → nub prune) produce
   one **fixed base topology**. Annotation and pruning are computed against this so segment
   identity is stable.
2. **Annotate** the main vessels in the picker (or load the sidecar).
3. **Per-vessel ostial radius** — for each annotated vessel, find its most-proximal
   (root-side) segment and measure its radius with `flow_fractions._branch_diameter` (the
   denominator for that vessel's threshold).
4. **Radius-ratio prune** — drop side branches below the threshold (see *Pruning*).
5. **Containment prune** — drop "swallowed" leaf stubs, length-independently.
6. **Write** the pruned forest to `model.am.xml` (round-trips through `parse_xml`).
7. **Surface** — `pipeline.run_pipeline` generates the SDF lumen mesh, one per connected tree.
8. **Flow** — `flow_fractions.compute_flow_fractions` computes the Giessen split per tree.

**Per-tree vs combined.** The forest can contain several disconnected trees (e.g. LCA and
RCA). Pruning is forest-aware (each tree rooted independently; each vessel judged against its
own ostial radius). **Flow** and **surface meshes** are produced **separately per tree**.
The written **`model.am.xml` is combined** (all trees in one file per ratio).

---

## The interactive picker — how it works

The picker (`run_picker` → `_pick_one_tree`) is where you mark which segments make up each
main epicardial vessel. Its job is to turn mouse clicks into a stable, reusable annotation.

### One window per tree

`centreline_reconnection.find_connected_components` splits the forest into trees (largest first). The picker
opens **one window per tree**, titled `Tree k/K — n segments`. You annotate the vessels in
that tree, press `q` to advance to the next tree, or `x` to stop once the large trees are
done (skipping tiny fragments). Selections from every tree accumulate into a single combined
annotation — so the output stays combined even though you pick tree-by-tree.

### What you see: points + contours (not a solid surface)

Each segment is drawn by `_segment_contour_mesh` as:

- **Centerline points** — the segment's smoothed centerline nodes, as dots.
- **Cross-section contour rings** — at each point, a circle of the local lumen radius `r`
  lying in the plane **perpendicular to the centerline tangent**. The ring orientation uses
  `splines.compute_frenet_frame` with parallel transport, so consecutive rings don't twist.

This shows the lumen as a "stack of rings" outline rather than an opaque tube, so you can see
the vessel calibre and the centerline at the same time. (Same ring construction as the flow
visualiser's `_build_diameter_contours`.)

### How clicking selects a whole segment

The contour rings + dots are the **pickable** geometry (no hidden surface). Each segment's
contour mesh is tagged with `field_data["seg_idx"]`.

- **Click vs drag.** Custom `LeftButtonPress`/`LeftButtonRelease` observers record the press
  pixel and only treat it as a selection if the cursor moved ≤ ~7 px between press and
  release. A click-and-drag is a **camera rotate** and never selects — fixing the old
  behaviour where starting a rotate picked a segment.
- **Exact front-most hit.** On a genuine click, a `vtkCellPicker` (tolerance ~0.008) resolves
  the segment from the **picked mesh's** `seg_idx` (the front-most geometry under the cursor),
  falling back to nearest tagged point. This is what makes dense, non-planar multifurcation
  regions selectable — you get the segment you see in front, not a neighbour.
- **Hover preview.** As you move the mouse (no button held), the segment a click would select
  is outlined by a bright **green** polyline along its centerline, so you can confirm the
  target before committing. Rotate/zoom until the right segment lights up, then click.
  Disable with `--no-hover`.

The matched segment's rings/dots recolour grey → yellow (pending) → vessel colour (committed).

> Earlier versions resolved the click from an invisible `opacity=0` tube via the picked
> actor's name; that failed silently because `pyvista.Actor` exposes no public `.name`. The
> field_data + observer approach above replaces it.

### Selection → vessel state machine

- **Click** a vessel → toggles that segment into the current pending set (`state["selected"]`);
  the segment recolours grey → **yellow**.
- **`n`** → commit the pending set as a **named vessel**: a name is taken from the preset list
  `LAD, LCx, RCA, LM, Diag, OM, PDA, PLB, Ramus` (cycling; auto-deduped to `LAD_2`, etc. if a
  name repeats), a colour is assigned, and the entry is appended to `vessels_all` as a set of
  **global** segment indices. Those segments recolour to the vessel colour and become
  **locked** — clicking them again is ignored (use `u` to undo).
- **`c`** → clear the pending (un-committed) selection.
- **`u`** → undo the most recently committed vessel **from this tree**.
- **`q`** → finish this tree, open the next.
- **`x`** → stop annotating; skip any remaining (smaller) trees.

An on-screen text overlay shows the current tree, the key bindings, the pending count, and the
list of committed vessels with their segment counts.

### Saving + reloading: stable identity

When picking finishes, the annotation is written to the sidecar JSON. To survive re-runs it
stores, per vessel, a list of **`seg_key`** fingerprints rather than raw indices. A `seg_key`
(`build_segment_keys`) is a hash of the segment's two endpoint coordinates (sorted) plus its
midpoint, at 1 µm resolution. Because the base preprocessing is deterministic, the same XML
re-preprocessed yields the same keys, so a later run resolves each key back to the right
segment (`resolve_annotation_to_indices`). The sidecar also records:

- `source_xml_sha1` — detects if the input XML changed.
- `preprocess_signature` — the config flags that affect topology.

If either differs you get a warning, and if **any** main-vessel key fails to resolve the run
aborts and asks you to re-pick (a silently mis-resolved main vessel would corrupt every model).

Sidecar shape:

```json
{
  "schema_version": 1,
  "source_xml": "…smooth_thick_adj_LADAF_2024_28….am.xml",
  "source_xml_sha1": "…",
  "preprocess_signature": { "MIN_STRAHLER_ORDER": 1, "MERGE_DEGREE2_SEGMENTS": true, … },
  "vessels": {
    "LAD": { "seg_keys": ["a1b2…", "c3d4…"], "color": "#d62728" },
    "RCA": { "seg_keys": ["…"], "color": "#2ca02c" }
  }
}
```

> **Tip:** vessel names auto-cycle through the preset list. If you want anatomically correct
> names, just edit the `vessels` keys in `epicardial.json` after picking — the geometry keys
> are what matter.

### Why downstream uses point-ids, not the picked indices

The picker records segment **indices** (resolved from `seg_key`) on the base topology. But
pruning a side branch at a bifurcation can leave the two flanking main segments as a degree-2
node, which the contraction step **merges** into one new segment — changing indices and keys.
So for every pruned model the main vessels are re-identified by **point-id set membership**
(`assign_vessel`): each vessel is the union of its segments' centerline point ids, and a
(possibly merged) segment is "that vessel" if a majority of its point ids belong to the set.
Point ids are preserved across merges, so this is robust where indices/keys are not.

---

## Pruning logic

**Radius-ratio subtree prune** (`prune_by_radius_ratio`). Walking down each tree from the
root: a non-main segment whose **take-off radius** is `< ratio × (ostial radius of the main
vessel it descends from)` is removed **together with its whole downstream subtree**. Main
vessels are never removed. After removal, newly-exposed degree-2 nodes are contracted and the
pass repeats until stable.

Both the take-off radius and each vessel's ostial radius are measured by
`flow_fractions.branch_radius_mm`, which **skips the first `--ostium-skip` contours** of the
branch — where it intersects the parent and the segmented radius is an inflated outlier — then
**averages** the radius over the next `--radius-navg` clean contours (a robust median fallback
covers very short branches). Every pruned take-off branch is exported with its measured radius
to `pruned_branches.csv` (per ratio and combined): `seg_id, vessel, radius_mm, threshold_mm,
n_subtree_removed`.

**Containment ("swallowed stub") prune** (`prune_contained_leaves`). After Strahler/length
pruning you can be left with short leaf stubs whose centerline lies almost entirely inside a
neighbouring vessel's lumen. This prune is **length-independent**: a leaf is dropped only if a
fraction ≥ `--frac` (default 0.8) of its centerline points — **including its distal tip** —
fall inside another segment's lumen (capsule signed-distance < 0). A genuine thin distal
vessel running in open space has ~0 containment and is kept, regardless of how short it is.
Annotated main vessels are never removed.

---

## Flow metrics (`main_vessel_flow.json`, `summary.csv`)

Flow is the Van der Giessen diameter-law split (`compute_flow_fractions`), normalised so each
tree's inflow = 1.0. Per main vessel:

| Field | Meaning |
|-------|---------|
| `ostial_flow_fraction` | Total share entering the vessel's ostium. **Invariant** to pruning the vessel's own side branches (flow into the ostium is fixed by the split at its parent junction — conservation). |
| `distal_flow_fraction` | Flow reaching the vessel's most-distal main segment. **Rises** as side branches are pruned — their flow reroutes down the trunk. **This is the main signal of interest.** |
| `retained_fraction` | `distal / ostial` — the fraction of the vessel's inflow that reaches its distal end. |
| `along_vessel_flow` | Flow at each main segment, proximal → distal. |

Example: pruning aggressively (ratio 1/2) might retain ~27 % of inflow to the distal LAD,
versus ~1 % when all small branches are kept (1/10), while the ostial share stays constant.

---

## Vessel radius stats (`vessel_radius_stats.csv`)

One row per annotated vessel with the radius (mm) **min / mean / max** over three regions:

| Region | Segments used |
|--------|---------------|
| `total_*` | All of the vessel's segments. |
| `prox_*` | The **ostial** (most-proximal / min-depth) segment only. |
| `dist_*` | The **distal-most** (max-depth) segment only. |

Each region also reports the contour count (`*_n`). Radii **exclude junction-inflated
contours** — `--ostium-skip` (default 10) contours are dropped at every bifurcation end
(node degree ≥ 3) so the bulges where side branches join don't skew the stats (especially
`max`); true ostium/tip ends (degree 1) are kept. These stats are a property of the annotation
(the main vessel is preserved across pruning), so the file is written **once** at the top
level, not per ratio.

---

## Caveats

- **Display required** for `--pick`; later runs are headless from the sidecar.
- **Surface == flow tree:** the script disables short-terminal-nub pruning for the surface
  runs so `run_pipeline` doesn't drop thin distal vessels you intentionally kept.
- **Containment prune is geometric and global** across all trees. For anatomically separated
  LCA/RCA this is fine; if two trees physically overlapped, a leaf of one could be flagged
  inside the other.
- **Unattributed branches** (no annotated ancestor, e.g. an un-annotated tree) are **kept** by
  default; use `--prune-unattributed` to drop them.
- **Picking precision:** clicks resolve to the nearest segment within the cell-picker
  tolerance (~3 % of the window diagonal); zoom in if two thin vessels are very close before
  selecting.
