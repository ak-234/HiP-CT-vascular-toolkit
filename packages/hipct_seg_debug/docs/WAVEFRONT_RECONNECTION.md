# Tensor-guided wavefront reconnection

An opt-in connector for the gaps the [geodesic connector](GEODESIC_RECONNECTION.md)
cannot reach: breaks tens of radii long in collapsed, ribbon- and slit-shaped
ex-vivo HiP-CT vessels, with dropouts in the middle and, sometimes, no known far
side. It shares the geodesic connector's corridor, calibration, classification,
gates, global selection, transactional apply, cross-section transport and audit
files, and changes what happens before and inside the search.

Enable it with `edit connect --wavefront`. Everything below is off by default.

## Why a tensor, and why a wave

The geodesic connector prices every voxel with one number and runs an A\* whose
only sense of direction is a turn penalty. On a round tube that is enough. On a
collapsed vessel it is not: the lumen is a slit two voxels thick and a dozen wide,
and the cheapest way from one end to the other is very often *through* a wall, or
sideways across the ribbon, rather than along it. The image knows the difference
— two wall gradients pressed together have a very definite orientation — and the
scalar cost throws that away.

A point-to-point search also cannot answer the questions a long gap asks. Where
does the vessel *go* when the far end is unknown? Which of two continuations does
the image prefer, and by how much? How far can the evidence be trusted past a
dropout? Those are questions about a *map* of arrival costs over the corridor,
which is what a propagated front produces and a path search does not.

## Pipeline

| Stage | Module | What it does |
|---|---|---|
| Clean | `edit/reconnect/wavefront/prepare.py` | whole-graph centreline refinement, radius remeasurement, one `EndProfile` per free end |
| Propose | `edit/reconnect/wavefront/route.py` | the sibling's geometric gates with the profiled tangents and a 40-radius reach; optional open-end exploration |
| Price | `..geodesic/cost.py` + `wavefront/tensor.py` | the calibrated scalar cost, then a Riemannian metric from the structure tensor and Hessian |
| Propagate | `wavefront/propagate.py` | numba lattice wave propagation (26/98-stencil), optional `agd` backend |
| Bridge | `wavefront/bridge.py` | dual fronts meeting at a saddle; keypoint chain with a look-ahead cone |
| Gate | `..geodesic/route.py` + `wavefront/route.py` | the sibling's evidence gates, then alignment and normal-crossing against the tensor |
| Select, apply, audit | `..geodesic/{select,apply,audit}.py` | unchanged; provenance code `WAVEFRONT = 5` |

### Clean

Every proposal cone and every seed direction starts at a free end, and the free
end's direction from `candidates.endpoint_tangent` is the difference of two
centreline points at least a micrometre apart. On a voxel-staircased skeleton that
is noise: measured on the synthetic `flat` benchmark phantom, the two-point
tangent is 37–50° off the true axis. `prepare.clean` runs
`centreline_refine.refine` over the whole graph (terminal nodes held fixed, so tips
do not move; topology and segment ids untouched), remeasures the radii exactly as
`refine-centreline` does, and stamps `centreline_displacement_um`. After it the
same tangents are within half a degree.

`prepare.profile_ends` then cuts the component's cross-section a little behind
each tip (`geodesic.shape.extract_section`) and fits an ellipse to it. The
`EndProfile` carries the tangent, the major axis, the half-lengths, the flatness
and the intact tail points, and those set the tensor scales and the chain step.

### Price: the image intensity tensor

Two tensors with two jobs, both from scikit-image:

- **Structure tensor** `J = G_rho * (grad I grad I^T)` for orientation. Its smallest
  eigenvector `nu_1` is the direction of least intensity variation — along the
  vessel; its largest `nu_3` is across the two pressed-together walls; `nu_2` is
  across the ribbon's width. The eigenvalue gaps give a `coherence` and an
  `axis_confidence`. On a perfectly flat sheet `mu_1 ~ mu_2`, the in-plane
  direction is legitimately undetermined, and the metric is made planar-isotropic
  rather than guessed.
- **Hessian** for planarity `P`: the sheet/ribbon signature (`|l2| << |l3|`,
  `l3 < 0` on the inverted, lumen-bright image), gamma-normalised over three
  scales. A round tube scores low here on purpose; the scalar term already rewards
  tubes.

The scales follow the ends: the derivative scale is the slit's half-thickness, the
integration scale its half-width, both clipped to sensible voxel ranges.

The metric is `M = c^2 A` with `c` the calibrated scalar cost and

```
A = I + width_weight * a * coh * L * (I - nu_1 nu_1^T)
      + normal_weight * P * max(coh, P) * (nu_3 nu_3^T)
```

where `L` is the calibrated lumen likelihood. Three consequences worth knowing:

- **Both terms are gated by lumen-likeness.** A HiP-CT ring artefact is planar and
  bright; the cut faces either side of a mask gap are coherent and background.
  Without `L`, both would be handed a confident direction that means nothing for
  a vessel — the second was measured to reject every mask-only slit repair before
  the gate was written this way.
- **The anisotropy is capped** (`max_ratio`, default 5). A 26-neighbour lattice
  approximates the continuous distance to within ~13 % and resolves about a 5:1
  anisotropy faithfully; asking for more gives a metric the solver cannot honour.
  `--stencil 98` and the `agd` engine both allow a larger cap.
- **The sign conventions are inherited.** Dark lumen throughout; the mask-only
  surrogate is foreground-bright and flips exactly as in `geodesic/cost.py`.

### Propagate

A Dijkstra over the padded corridor with Riemannian edge costs
`0.5 (c_v ||e||_{A_v} + c_w ||e||_{A_w}) |e|`, compiled with numba, state = voxel.
Direction is priced by the metric, not by a turn table, so the state space is 27
times smaller than the A\*'s and a corridor forty radii long can be swept whole
(1.7 M voxels in ~1.5 s). The `Front` it returns holds arrival cost, path length,
parent pointers and seed label for every voxel. A soft asymmetric cone —
`(axis, degrees, weight)` — multiplies the cost of steps outside the cone, which is
how a restarted front is told which way the vessel was going.

Large corridors get the sibling's coarse-to-fine treatment: an isotropic pass on a
block-minimum of the scalar cost decides the corridor, and the fine anisotropic
pass is confined to a tube around it.

`--engine agd` uses HamiltonFastMarching's `Riemann3` model when the optional
`agd` package is importable (`pip install "hipct_seg_debug[eikonal]"`); its value
map is turned into the same `Front` by steepest descent. Nothing imports it
unconditionally, `auto` prefers it when present, and asking for it without it is
refused with a message rather than worked around. It has not been exercised on
Windows conda here.

### Bridge

**Dual fronts**, for a pair with a known far side. One front from the source along
its tangent, one from the target set along its; the route is the cheapest place
they meet — the saddle of `U_s + U_t` — and the two backtracks joined there.
Alternatives are found under suppression as in `astar.routes`, so a
single-corridor gap reads as unambiguous rather than as unexamined.

**Keypoint chain**, when the dual fronts fail the unsupported gate or when there is
no far side. The front is swept from the tip, restricted to the forward half-space
and coned along the tangent, allowed to travel one step (`keypoint_step_major` ×
the end's major half-axis, default 4), and the cheapest voxel on that shell becomes
the next keypoint; the next leg's direction is the leg just taken blended with the
local `nu_1`. When the shell has no support, the same map is asked for a cluster of
planar, lumen-like voxels further out in the cone — the **ribbon look-ahead** — and
if one exists its cheapest voxel becomes the keypoint and the leg is recorded as a
**forced bridge**. Otherwise a bounded number of weak keypoints are tolerated
(`MAX_WEAK_KEYPOINTS = 2`) before the chain stops and says why:

```
no planar lumen signature within the look-ahead cone after 2 weak keypoint(s)
the chain exceeded its 800 um budget without reaching a target
```

A chain that stops early is returned with its partial path so an audit can show
where. The construction is Benmansour & Cohen (2009) with the direction carried
between restarts as in Kaul, Yezzi & Tsai (2012).

### Gate

The sibling's gates run first and unchanged: tortuosity, contiguous unsupported
path (allowance 8 radii here rather than 4 — the metric and the chain refuse wall
crossings on their own, so a longer dropout *along* the vessel is the case this
package exists for), unrelated-component crossing, ambiguity margin, confidence.
Then the tensor's:

- **alignment** — mean `|t . nu_1|` over route steps where the metric is anisotropic
  *and* the in-plane axis is determined (`axis_confidence > 0.3`); below 0.5 the
  route "cuts across the local vessel orientation".
- **normal crossing** — the fraction of anisotropic, planar steps with
  `|t . nu_3| > 0.7`; above 30 % the route "passes through collapsed walls".
- a chain with **forced bridges** never goes straight to accept; it goes to review.

Both numbers are in the candidate's evidence whether or not they fire, alongside
`strategy` (`dual_front` / `chain`), the chain's stop reason and keypoint count,
and the end profiles.

### Open-end exploration

`--explore-open-ends` runs the chain from every associated free end nothing was
proposed for, with its own component allowed and the finite voxels touching every
*other* component as goals. Whatever the front reaches becomes an ordinary
`Bridge` to the nearest graph point there — a free end within two radii, else a
T-junction onto the segment — and goes through the same classification and gates
as any other proposal. Nothing is applied from exploration directly.

## Measured on the phantoms

`tests/test_wavefront.py` pins these:

- A ribbon with a 60-voxel (30-radius) mask gap whose image keeps a faint dark
  trace: `--geodesic` proposes nothing (reach gate); `--wavefront` accepts one
  route, alignment 0.99, in the slit plane throughout, and transports the
  cross-section across it.
- The same gap with a 20-voxel dropout in the middle: the dual fronts fail the
  unsupported gate, the chain reaches the far side, and the candidate is refused
  with the chain reported — a 200 µm dropout is above the 8-radius allowance, and
  that is the right answer for a 20 µm vessel.
- A parallel vessel two radii away is blocked, never routed through.
- With a one-degree proposal cone nothing is proposed; exploration supplies the
  pair and it is accepted.

## CLI

```
edit connect --wavefront --seg SEG.am [--raw DIR] [--out GRAPH.am --out-seg SEG.am]
             [--refine-method M] [--refine-strength F] [--refine-workers N]
             [--reach-radii 40] [--keypoint-step 4] [--lookahead-cone-deg 15]
             [--anisotropy-ratio 5] [--stencil 26|98] [--engine auto|lattice|agd]
             [--explore-open-ends] [--no-chain-fallback]
             [--review-json F] [--decisions-json F] [--alternatives 3]
             [--max-unsupported-factor 8]
```

## In the GUI

```
python -m hipct_seg_debug --edit --graph GRAPH.am --seg SEG.am --raw RAW_DIR
```

1. **Commands tab** → pick `connect`, press **Use loaded paths** (fills `graph`,
   `seg`, `raw` from what the viewer has open), tick **wavefront**, set
   `refine-workers` to your core count, and give `review-json` and
   `decisions-json` paths. Leave `out` / `out-seg` empty for a dry run. **Run**.
   The job runs as a child process, so **Stop** works and the log streams the
   refinement iterations, the profiled ends, then one line per candidate.
2. When it finishes, a review file with anything to adjudicate is loaded into the
   **Reconnect tab** automatically; walk the candidates (the evidence shows
   `strategy`, `alignment`, `normal_crossing` and the chain's stop reason), accept
   or refuse each, **Save decisions** to the same `decisions-json` path.
3. Back on the **Commands** tab, add `out` and `out-seg` paths to the same form and
   **Run** again: the saved rulings are applied alongside the automatic accepts,
   and the graph and mask are written together. The Data tab then offers
   **Load result** for the new graph.

## Tests

`test_wavefront_propagate.py` (15 + 1 optional), `test_wavefront_tensor.py` (9),
`test_wavefront_prepare.py` (7), `test_wavefront_bridge.py` (11),
`test_wavefront.py` (11), `test_wavefront_cli.py` (12), with the dark-lumen
greyscale phantoms `greyscale`, `ribbon_gap` and `FakeStack` added to
`conftest_geodesic.py`.
