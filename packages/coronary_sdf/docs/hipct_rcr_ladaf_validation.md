# LADAF_2024_28 validation — 9 October 2026

See the [RCR protocol](hipct_rcr_protocol.md) for the implementation. Four real
left proximal branches and three right proximal branches have now been measured, but their hydraulic support gate has
not passed. No complete patient network or CFD result is claimed.

## Acquisition scale

`F:\32_99um_LADAF-2024-28_heart_overview_\metadata.json` names scan
`HEL_HA2100_16.495um_LADAF-2024-28_heart_overview_` and records sample-pixel size
0.016495 mm. Native crop differences are 6309×6159×9504 pixels. Dividing by two
and taking integer image sizes gives 3154×3079×4752, exactly matching the TIFF
stack's dimensions and slice count.

Use **32.99 µm working-stack spacing**, from 2×16.495 µm native acquisition.
The matching segmentation is another 2× binned, approximately 65.98 µm. There
is no supported 32.99→32.04 rescaling for this dataset. The older toolkit prose
asserting 32.04 µm is contradicted by these primary records. A separate
`32.04um_artery_left_right` folder has different dimensions and no calibrated
TIFF resolution tags; its name cannot establish this dataset's scale.

The accepted **91.62400695 g** myocardial mass and prescribed flows remain
unchanged. Evidence: `runs/rcr/acquisition_evidence.json`.

## Inlets and placement

Legacy Fluent meshes were inspected read-only for anatomical inlet mapping.
They are not reused as corrected study geometry.

| Tree | Root node | Root edge | Legacy region | Opening area (mm²) |
|---|---:|---:|---|---:|
| Left | 133 | 67 | `Inlet_000` | 1.918216843 |
| Right | 310 | 291 | `Inlet000` | 3.825811164 |

These root profiles are uniquely nearest to the inlet centroids (0.252 and
0.161 mm; next nearest branches 6.33 and 4.02 mm away). Endpoint-to-centroid
distances are 1.119 and 0.974 mm, showing that the old meshes cut back the graph
stubs. New inlet planes need to account for this difference. Old CCL records
also use 1060 kg/m³ rather than the active 1050 kg/m³, so previous solutions
cannot simply be imported into this comparison.

The graph contains 309 segments, 311 nodes and 29,122 centreline samples.
All samples lie inside the lattice bounds; 99.9588% lie on foreground voxels,
and every segment has at least 95% foreground coverage. This checks alignment,
not perfect centring or junction ownership.

Evidence: `runs/rcr/legacy_mesh_inlet_audit.json`,
`runs/rcr/centreline_placement_audit.json`, `runs/rcr/ladaf_case.json`.
Graph SHA-256: `97b3a55a5b387b9c8cf35806e07929b6fd92f9ffb25ae77e8a08a27bbd8f15d4`.
The authoritative complete hashes are in the placement audit.

## Perimeter measurements

An isolated four-branch patch was tested, then all four source branches were
remeasured at full length **with all 309 segments present as context**. The
latter avoids interpreting early patch clipping as absent branch support.

Settings: 32.99 µm working voxel; 65.98 µm segmentation; root edges 67/291;
perimeter gate zero; discretisation correction and section filtering enabled;
bifurcation tapers off; one pass and one worker. Effective settings and code
hashes accompany the run.

| Source segment | Trusted / total samples | Hydraulic result | Unresolved sampled interval |
|---|---:|---|---|
| 67 | 10 / 18 | Rejected | Samples 0–2; arclength 0–0.971 mm |
| 270 | 6 / 26 | Rejected | Samples 0–13; 0–3.910 mm |
| 307 | 37 / 56 | Rejected | Samples 0–15; 0–4.324 mm |
| 308 | 35 / 49 | Supported | Bounded gaps and short own-branch junction extensions |

Total: **88 trusted measurements / 149 samples (59.1%)**. Rejections include
56 unresolved branch-overlap samples, two unmeasurable sections and three
unstable sections. The other 61 radii are ordinary interpolation, not direct
measurements. All four full-length branches have anchors; the zero-anchor
result for one short isolated branch was a clipping effect.

After bounded interpolation and short branch-specific junction support, three
spans remain unresolved. These intervals describe sampled support, not proposed
cut planes. No arbitrary radius, borrowed sibling calibre, extended extrapolation
limit or artificial carina radius was inserted to make the hydraulic gate pass.

Artifacts under `runs/rcr/real_full_context_subset/`:

- `measured_subset.am`: selected radii and their provenance; other segments unmeasured.
- `geometry_audit.json` and `radius_perimeter.log`: coverage, rejection and settings.
- `radius_review.png` and `radius_review.json`: visual review and exact unsupported ranges.

Reproduce measurements with `prepare`, using
`runs/rcr/ladaf_full_context_subset.json` and a new output directory. The plotting
script is `packages/coronary_sdf/examples/rcr/review_measurements.py`. The original
run preceded the addition of explicit span ranges to the preparation audit;
the companion review JSON records them. No hydraulic `network.json` was emitted.

## Junction diagnosis and refinement experiment

Read-only diagnostics were run on source segments 67, 270, 307 and 308 using
`examples/rcr/diagnose_junctions.py`. Both source graph and segmentation hashes
were unchanged. Artifacts are `runs/rcr/junction_diagnosis/diagnostics.txt` and
`provenance.json`. These legacy diagnostic recuts use a 64-voxel half-window
and different stability defaults; their counts must not replace acceptance in
the perimeter pass. The source graph has no measurement provenance, so the
diagnostic's default filled/unmeasurable columns are not measurement verdicts.

At the inlet, the first two cuts contain only 4 and 39 foreground pixels,
respectively. Near node 134, proximal sections of 270 and 307 overlap neighbouring
branches. This supports investigating both inlet placement and branch ownership;
it does not establish a new anatomical cut plane or justify radius extrapolation.

A separate candidate used the existing `centroid-coherent` refinement, selecting
all four incident branches with the full graph present. Roots 133/310 were fixed,
junction movement allowed, strength 0.1, maximum half-window 256, maximum samples
64, and one worker. A ten-iteration experiment moved one junction and 138 points,
with maximum displacement about 1.36 mm. No sampled centreline edges left the
segmentation, but the refinement reported **not converged**.

Repeat perimeter measurement with unchanged acceptance settings yielded 90/149
trusted samples. Segment 270 gained hydraulic support; 67 remained unsupported
at samples 0-2 (0-1.003 mm), and 307 at samples 0-15 (0-3.545 mm). This candidate
is not the accepted geometry. Artifacts: `runs/rcr/refinement_candidate/` and
`runs/rcr/refinement_candidate_measured/`, including the plotted radius review.
The exact experiment script is `runs/rcr/refine_candidate.py`.

Repeating from the original geometry with the default **25-iteration** limit
still reported `converged=false` and `supported_stable=false`. One junction and
139 points moved; no sampled edges left the segmentation. The independent
perimeter pass produced **94/149 trusted samples** (67: 12/18, 270: 8/26,
307: 38/56, 308: 36/49). This differs from the refinement's internal section
count, which is not the perimeter acceptance count. Segments 270/308 were
hydraulically supported. Segment 67 still lacked samples 0-2 (0-1.003 mm),
and 307 lacked samples 0-14 (0-2.774 mm). Increasing the iteration budget did
not qualify this candidate, and it was not promoted to the reference geometry.

Artifacts: `runs/rcr/refinement_candidate_25/`,
`runs/rcr/refinement_candidate_25_measured/`, and the exact experiment script
`runs/rcr/refine_candidate_25.py`. The scripts save their settings, implementation
hash, convergence history and per-segment displacement. Original graph hashes
were unchanged in both experiments. No radius acceptance threshold or hydraulic
extrapolation bound was relaxed; no patient RCR parameters were generated.

## Right proximal subset

Three original right-tree branches were measured with all 309 source segments
present as context, using the same perimeter settings and explicit roots.

| Source segment | Trusted / total samples | Hydraulic result |
|---|---:|---|
| 291 (root) | 72 / 97 | Supported |
| 219 | 60 / 105 | Rejected: samples 0-25, arclength 0-2.698 mm |
| 230 | 53 / 105 | Supported |

Total: **185/307 trusted samples (60.3%)**. Supported profiles still contain
explicitly modelled gaps/junction extensions; support is not 100% measurement
coverage. No hydraulic network was emitted. Configuration:
`runs/rcr/ladaf_right_subset.json`; artifacts: `runs/rcr/real_right_subset/`,
including `geometry_audit.json`, `radius_review.png` and `radius_review.json`.

## Remaining qualification

1. Resolve the segment-67 inlet stub against the segmentation and existing inlet cut.
2. Review node 134 and the shared proximal spans of branches 270/307, and the
   right-tree unresolved span on 219. Resolve
   geometry/ownership or explicitly define a separate hydraulic modelling assumption.
3. Repeat this gate before processing both complete trees; establish the reliable
   distal limit from coverage rather than enforcing a 100 µm radius cutoff.
4. Regenerate surfaces, opening mappings and volume meshes after correction.
5. Compile and qualify the adapted Fortran, including CFX licensing, equilibrium,
   periodic convergence, restart and mesh/time-step convergence.

Fifteen new tests and 67 relevant existing radius/provenance/surface tests passed.
The 15 RCR tests were rerun successfully on 9 October after this investigation;
the new diagnostic example also passed static checks, compilation and its real-data run.
Fortran compilation was attempted but produced no compatible shared library;
no Intel compiler was found on PATH. CFX executables exist, but no new `.def`,
CFD solution or complete patient RCR table has been produced.
