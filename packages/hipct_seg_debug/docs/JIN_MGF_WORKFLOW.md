# jin-mgf skeletonisation, end to end

This page takes a coronary segmentation to a smooth `jin-mgf` centreline. Every
stage has a command-line form and a GUI form. The method itself is described in
[JIN_MCP.md](JIN_MCP.md); each command's full flag table is in [CLI.md](CLI.md).

| # | stage | command line | GUI |
|---|---|---|---|
| 1 | correct the mask by hand | `mask-export` (writes out what you painted) | paint panel: brush, *Fill collapsed lumen*, *Split fused vessels* |
| 2 | guarded mask repair | `repair-mask --guarded` | Commands tab → `repair-mask` |
| 3 | skeletonise | `skeletonise-all --algorithms jin-mgf` | Commands tab → `skeletonise-all` |
| 4 | prune spurious branches, simplify junctions | `simplify-skeleton --apply` | Commands tab → `simplify-skeleton` |
| 5 | smooth | `fair-centreline` | Commands tab → `fair-centreline` |
| 6 | inspect | — | Data tab → **Load result** |

Stages 2–5 are also one chain in the GUI: **Workflows** tab → **13. jin-mgf
centreline, end to end**. See [Run it as one chain](#run-it-as-one-chain-in-the-gui).

The order matters:

- **Mask first.** Jin extracts paths through the mask. A notch, a hollow lumen or a
  fused neck becomes a false branch or a loop, and no post-processing removes those
  reliably.
- **Topology before geometry.** `simplify-skeleton` deletes and welds segments, so it
  runs before `fair-centreline`. Smoothing a segment that is about to be deleted is
  wasted work, and a weld would leave a kink in a curve that was already smooth.
- **Nothing that moves points runs after `fair-centreline`.** Do not run
  `optimise-skeleton` re-centring or smoothing on its output.

## Before you start

Install the monorepo as described in the root README, then check the environment:

```powershell
python tools/check_environment.py
```

Set the inputs once, so that neither the commands nor the viewer need to be told
again. Nothing in the package has a default dataset path.

```powershell
$env:HIPCT_RAW  = "D:\data\raw_slices"            # folder of raw slices
$env:HIPCT_SEG  = "D:\data\artery.labels.am"      # label lattice (Left_Tree / Right_Tree)
$env:HIPCT_GRAPH = "D:\data\any_skeleton.am"      # only the viewer needs this
```

You also need the scan's raw voxel size. The value is never inferred: pass it as
`--voxel-um` (32.04 for LADAF-2021-17). Use the same value on every command. If it
differs, the graph and the mask stop lining up.

The examples write to `work\`. Any directory will do.

## Open the GUI

```powershell
python -m hipct_seg_debug --voxel-um 32.04 --paint --edits work\edits.npz --no-surface
```

- **`--paint`** adds the writable segmentation layer and the paint panel for stage 1.
- **`--edits`** is where the stage 1 corrections are kept. Always give it a path.
  Without one the corrections live only in memory.
- **A skeleton is required to load.** On a first pass, use whatever skeleton you have
  for the scan, such as the Avizo export or an earlier candidate. It is only something
  to navigate by.
- **The dock.** The 3D window carries a **control** dock with tabs **Data**,
  **Commands**, **Workflows** and **Log**.
  - Every subcommand used below has a form in **Commands**. The forms are generated
    from the same argparse definitions as the command line.
  - The form shows the exact command line it will run, with a **Copy** button. You can
    build a command in the GUI and paste it into a script.

## 1. Correct the mask by hand

**GUI only.** Painting needs the slice browser. The full guide is
[EDITOR.md — painting the mask](EDITOR.md#running-it-painting-the-mask). In brief:

1. **Go to the problem.** Type an edge id into *go to edge* in the 3D window's
   **layers** panel and press *go*, or pick a vessel and press `v`. This opens the
   slice browser there.
2. **Check the label box before painting.** Each tree has its own label (`Left_Tree` 1,
   `Right_Tree` 2). Leave *protect other labels* on.
3. **Paint or erase.** Select the `segmentation (editable)` layer. Keys: `2` paint,
   `4` erase, `[` / `]` brush size, `Ctrl+Z` undo. Press **Commit**.
4. **A collapsed vessel with a missing chunk.** Select the `paint seeds` layer, press
   `2`, and click along the collapsed run, beyond both ends of the notch. Then press
   **Fill collapsed lumen**.
5. **Two vessels fused across a neck.** Place seed group 1 along one vessel and seed
   group 2 along the other. Then press **Split fused vessels**.
6. **Check the result.** The `paint result` layer shows what each tool did: green
   filled, yellow cut, red refused. The Log gives the reason for any refusal.
7. **Save.** Press **Save edits…**. Edits are also committed on exit.

The corrections are stored in `work\edits.npz`. The source lattice is never
modified. You do not need *Re-skeletonise painted region* here. Stage 3 skeletonises
the whole corrected mask again.

**Command line.** There is no painting on the command line. You can do two things
with the edits store:

- **Pass it on.** Every stage-2 command accepts `--edits work\edits.npz`.
- **Write it out** as a standalone lattice, keeping labels and `Materials`:

```powershell
python -m hipct_seg_debug.edit mask-export --seg $env:HIPCT_SEG --edits work\edits.npz `
  --out work\corrected.labels.am
```

## 2. Guarded mask repair

This fills hollow lumens (where only the wall was labelled) and enclosed voids, one
tree at a time. It refuses any addition that would merge two pieces or create a loop.
How each region is tested is in
[SKELETONISATION.md §6.3](SKELETONISATION.md#63-guarded-mask-repair).

```powershell
python -m hipct_seg_debug.edit repair-mask --guarded `
  --seg $env:HIPCT_SEG --edits work\edits.npz --raw $env:HIPCT_RAW --voxel-um 32.04 `
  --close-radius 2 `
  --out work\repaired.labels.am --report work\repair.json --audit-dir work\repair-audit
```

- **`--close-radius`** is off by default, and on its own adds about 3% volume for no
  topological gain. The LADAF-2021-17 run used 2, which also closed small dents. Leave
  it out unless dents are leaving spurs.
- **`--edits`** brings in the stage 1 corrections. Leave it out if you painted nothing.
- **Read before going on.** Check `repair.json` (per-tree totals and every region's
  verdict) and the `repair-audit\` gallery of the largest accepted and refused
  regions. If a vessel still has a notch or a fusion, go back to stage 1.

**GUI.** Go to **Commands** → `repair-mask`. Tick `--guarded`, and fill `--raw`,
`--voxel-um`, `--out`, `--report` and `--audit-dir`. Fill `--edits` if you painted.
Press **Run**. The command runs in a subprocess, and **Stop** kills it.

## 3. Skeletonise with jin-mgf

Run this on the repaired lattice, not the original:

```powershell
python -m hipct_seg_debug.edit skeletonise-all `
  --seg work\repaired.labels.am --voxel-um 32.04 `
  --algorithms jin-mgf --per-tree --workers 2 `
  --jin-engine sparse --jin-allow-loops --no-score `
  --out-dir work\jin --summary-json work\jin\summary.json
```

| flag | why |
|---|---|
| `--per-tree` | each tree (each `Materials` label) is extracted on its own; Jin needs one connected component |
| `--workers 2` | one process per (candidate, tree) pair; each tree takes its own memory, so raise it only if memory allows |
| `--jin-engine sparse` | works on lumen voxels only. Needed for whole trees: the dense engine refuses above 2 M voxels |
| `--jin-allow-loops` | a real mask has loops (LADAF-2021-17 has χ = −1467); the output is still a tree, with the loops cut |
| `--no-score` | skips the super-metric pass, which does not fit in memory at full resolution |

The output is `work\jin\jin-mgf.am`. The per-round branch acceptance reports are in
`work\jin\jin-mgf-reports\`. Radii are **FDT thickness placeholders**, not measured
radii.

**Root each tree at its inlet.** By default each tree is rooted at its widest voxel.
On a flared inlet that produces a false Y: on LADAF-2021-17 its tips were 1.8 and
2.2 mm from the inlet. To avoid it, add one of these:

- `--root-graph refined.am` takes each tree's widest free end from a graph you trust.
- `--root-xyz X Y Z` takes a world position. Repeat it once per tree. The viewer's
  pick readout gives the coordinates.

**Try a region first.** `--roi-zyx Z0 Z1 Y0 Y1 X0 X1` (upper bounds exclusive)
decodes only that box, so you can check the settings in minutes. Drop `--per-tree`
for a single connected ROI. Variants can be compared in one run:

```powershell
python -m hipct_seg_debug.edit skeletonise-all --seg work\repaired.labels.am --voxel-um 32.04 `
  --roi-zyx 100 300 200 400 200 400 --no-score --out-dir work\jin-roi `
  --algorithms "jin-mgf,jin-mgf[mgf_mode=rim],jin-mgf[medialness=lsf;marking=section]"
```

**Known limit.** A branch leaving its parent at a right angle is not extracted if it
is shorter than about 12 parent radii (`marking=section` fills the parent's normal
slab, and that slab contains the branch). Branches at 45° are extracted.

**GUI.** Go to **Commands** → `skeletonise-all`.

- Set `--seg` to the repaired lattice, and `--algorithms` to `jin-mgf`.
- Set `--jin-engine` to `sparse`.
- Tick `--per-tree`, `--jin-allow-loops` and `--no-score`.
- Set `--out-dir`, and optionally `--root-graph`.

## 4. Prune spurious branches and simplify junctions

`simplify-skeleton` removes branches that are not vessels and merges junctions that
were split. It works in three phases:

1. **Short leaves.** A leaf is pruned when it is shorter than `--leaf-factor` times
   the radius of the thickest other branch at its junction, or than `--leaf-min-um`.
   Three kinds are kept: a leaf at least 0.8 of its parent's radius (a real vessel cut
   short), a leaf ending at the lattice boundary (a vessel the scan cut off), and the
   last segment of a component.
2. **Contained leaves.** A leaf that lies inside another vessel's lumen is removed. On
   a flattened vessel these are medial-sheet spurs. A leaf ending at `--root-node` is
   never removed.
3. **Short inner links.** A link shorter than `--link-factor` (default 1) times the
   larger neighbouring radius lies inside the parent lumen. Each such cluster is
   collapsed into one junction, which is placed at the deepest point of the lumen.

Without `--apply` it is a dry run. Read the dry run first:

```powershell
python -m hipct_seg_debug.edit simplify-skeleton work\jin\jin-mgf.am `
  --seg work\repaired.labels.am --voxel-um 32.04 --report-json work\simplify_dry.json

python -m hipct_seg_debug.edit simplify-skeleton work\jin\jin-mgf.am `
  --seg work\repaired.labels.am --voxel-um 32.04 `
  --apply --out work\jin-mgf_simplified.am --report-json work\simplify.json
```

Ids are renumbered on save. The phases and their limits are described in
[CENTRELINE_REFINEMENT.md](CENTRELINE_REFINEMENT.md#skeleton-simplification-before-dfs-fitting).

| flag | effect |
|---|---|
| `--no-leaf-prune` | skip phase 1 |
| `--no-contained-prune` | skip phase 2 |
| `--contained-nearest-host` | phase 2 asks only the nearest vessel whether it contains a leaf point |
| `--link-factor 0` | skip phase 3 |
| `--auto-thinnest`, `--min-length-um N` | also collapse links shorter than the thinnest vessel's diameter, or than N µm (the paper's thresholds) |
| `--segment ID` | phase 3 only, around one segment's junctions; phase 1 is skipped |

**GUI.** Go to **Commands** → `simplify-skeleton`. Run it once without `--apply` and
read the Log. Then tick `--apply`, set `--out`, and run it again.

## 5. Smooth through the junctions

A Jin path runs through voxel centres, so consecutive steps turn a median 45°.
`fair-centreline` removes that staircase:

- **One curve per vessel.** Each vessel is fitted as one curve through its junctions.
- **No kink at a branch.** A branch's first stretch inside its parent is not fitted,
  so the branch leaves the junction without a kink.
- **Stays in the mask.** Every point stays inside the segmentation.
- **Radii are not changed.**

```powershell
python -m hipct_seg_debug.edit fair-centreline work\jin-mgf_simplified.am `
  --seg work\repaired.labels.am --voxel-um 32.04 --out work\jin-mgf_fair.am
```

Measured on both LADAF-2021-17 trees (7763 segments, under 2 minutes):

| | step-to-step turn, median | junction kink, median |
|---|---|---|
| `jin-mgf` | 45.0° | 20.1° |
| after `fair-centreline` | 1.0° | 5.1° |

Each segment gets a `vessel_id` attribute naming the vessel it belongs to. The report
(`jin-mgf_fair.report.json`, or `--report-json`) lists any vessel that the
containment check left unchanged.

| flag | default | effect |
|---|---|---|
| `--smooth-radii` | 2.0 | smoothing length, in local radii; larger is smoother |
| `--through-radii` | 1.5 | unfitted stretch either side of a junction a vessel passes through |
| `--emergence-radii` | 1.5 | a branch's unfitted stretch next to its junction, in parent radii |
| `--max-deflection-deg` | 60 | a vessel continues only into a branch turning less than this |
| `--max-move-radii` | 1.0 | largest move outside those stretches |

`jin-mgf[fair=true]` in stage 3 runs this pass inside the skeletoniser. It then
smooths before any pruning, so use it only when you will not run stage 4.

**GUI.** Go to **Commands** → `fair-centreline`. Set the graph to the simplified
graph, set `--seg`, `--voxel-um` and `--out`, and press **Run**.

## 6. Inspect the result

When a command writes a graph, the **Data** tab's **Load result** button names the
file. Press it to reload the viewer on that graph. This takes about 0.4 s: the frame,
the lattice and the slice caches are kept. To open a different file, put its path in
the `skeleton (.am)` field and press **Reload graph**.

From the command line, open the viewer on the new graph directly:

```powershell
python -m hipct_seg_debug --graph work\jin-mgf_fair.am --seg work\repaired.labels.am `
  --voxel-um 32.04 --no-surface
```

What to check:

- **Leftover spurs, loops and fusions.** These come from the mask. Fix them in stage
  1 and run stages 2–5 again. Editing the skeleton by hand (`--edit`, see
  [EDITOR.md](EDITOR.md)) is the alternative for a one-off.
- **Radii.** They are still Jin's FDT placeholders. Do not use them as measured radii.
  [CLI.md](CLI.md#radius-perimeter--replace-every-radius-with-its-own-cross-section)
  describes `radius-perimeter`, which replaces each radius with its measured
  cross-section. It must be the last step that changes the graph. It has not been
  qualified on `jin-mgf` output.

## Run it as one chain in the GUI

Go to **Workflows** → **13. jin-mgf centreline, end to end** → **Run chain**. The chain
runs stages 2–5 in one run directory (`cache\runs\<timestamp>\`):

| step | command | writes |
|---|---|---|
| 1 | `repair-mask --guarded` (with the session's `--edits`, if any) | `repaired.labels.am`, `repair.json`, `repair-audit\` |
| 2 | pause: review the repair, then **Continue** | — |
| 3 | `skeletonise-all --algorithms jin-mgf --per-tree --jin-engine sparse --jin-allow-loops --no-score` | `jin\jin-mgf.am` |
| 4 | `simplify-skeleton --apply` | `jin-mgf_simplified.am`, `simplify.json` |
| 5 | `fair-centreline` | `jin-mgf_fair.am` |

- **Where the inputs come from.** The chain takes `--seg`, `--raw`, `--voxel-um` and
  `--edits` from the loaded session. Without `--voxel-um` or `--raw`, step 1 stops
  and names the missing flag.
- **What it leaves at defaults.** `--close-radius` and the root options are not set.
  Use the Commands tab, or the script below, when you need them.
- **When a step stops the chain.** A step that fails, or that exits cleanly without
  writing its output, stops the chain. The Log then names the reason.
- **Loading the result.** When the chain finishes, put `jin-mgf_fair.am` in the Data
  tab's skeleton field and press **Reload graph**.

## The whole chain as a script

The same five commands, for a headless machine or a batch job:

```bash
#!/bin/bash
set -euo pipefail
SEG=$HIPCT_SEG; RAW=$HIPCT_RAW; UM=32.04; W=work
mkdir -p $W/jin

python -m hipct_seg_debug.edit repair-mask --guarded --close-radius 2 \
  --seg "$SEG" --raw "$RAW" --voxel-um $UM \
  --out $W/repaired.labels.am --report $W/repair.json --audit-dir $W/repair-audit
# add  --edits $W/edits.npz  above if you painted corrections

python -m hipct_seg_debug.edit skeletonise-all --seg $W/repaired.labels.am --voxel-um $UM \
  --algorithms jin-mgf --per-tree --workers 2 --jin-engine sparse --jin-allow-loops \
  --no-score --out-dir $W/jin --summary-json $W/jin/summary.json

python -m hipct_seg_debug.edit simplify-skeleton $W/jin/jin-mgf.am \
  --seg $W/repaired.labels.am --voxel-um $UM --apply --out $W/jin-mgf_simplified.am

python -m hipct_seg_debug.edit fair-centreline $W/jin-mgf_simplified.am \
  --seg $W/repaired.labels.am --voxel-um $UM --out $W/jin-mgf_fair.am
```

On LADAF-2021-17 this is the chain that was run on the hand-corrected mask. Most of
the time goes to skeletonisation. On a whole LADAF-2024-28 tree, `jin-mgf` on the
sparse engine takes 75–150 s; the 2021-17 trees are much larger.
