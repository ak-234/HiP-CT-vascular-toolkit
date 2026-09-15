# BioMedEng 2026 talk guide

## One-sentence story

HiP-CT does not merely provide a higher-resolution coronary geometry; it provides a reference anatomy with which to test the distal-boundary assumptions that conventional coronary CFD cannot observe.

## Research gap

Clinical coronary imaging resolves the epicardial arteries but truncates the model before much of the resistance vasculature. Consequently, outlet flow split, resistance and compliance are usually inferred from scaling laws or lumped-parameter models. The missing study is a controlled test of how much resolved distal anatomy is required before those reduced outlet models become sufficient.

This framing is stronger than saying only that “clinical CT has limited resolution.” It identifies the modelling consequence of that limit and makes the next experiment explicit.

## Timing and purpose

| Slide | Time | Purpose |
|---|---:|---|
| 1. Title | 0:20 | State the question and the hook. |
| 2. Imaging boundary | 0:45 | Establish the research gap. |
| 3. HiP-CT opportunity | 0:40 | Explain what becomes measurable and state the objective. |
| 4. Pipeline | 0:40 | Show the graph-to-domain technical contribution. |
| 5. CFD methodology | 0:50 | State the mesh, physics, boundary conditions and interpretation boundary. |
| 6. Morphometry | 0:50 | Give the first quantitative result. |
| 7. Surface validation | 0:55 | Demonstrate fidelity and disclose the proximal limitation. |
| 8. CFD feasibility | 1:00 | Show branch-resolved outputs without claiming physiological validation. |
| 9. Next experiment | 0:55 | Close the loop back to the research gap. |
| 10. Take-home | 0:35 | Leave three claims and one memorable sentence. |
| **Total** | **7:30** | Leaves about 30 seconds of safety margin. |

The full speaker script is embedded in the PowerPoint notes.

## Results to emphasise

- One ex vivo left coronary tree: 264 segments, 127 bifurcations and 2.36 m of centreline.
- Analysed diameter range: 0.26–3.01 mm across four Strahler orders.
- Surface comparison at 8,673 in-domain centreline stations: overall mean radius bias +0.017 mm and mean absolute error 0.028 mm.
- Important counter-result: order-4 regions were over-expanded by 0.253 mm (23%), so proximal/junction reconstruction remains a limitation.
- Steady CFD fields were mapped to 123 branches. Flow accumulated towards proximal orders; distal branches had the highest and most variable WSS.

## Conclusions from the left full tree and right ratio-8 runs

Measured 2026-08-31 from `analysis_out/biomedeng_left_full/` and
`analysis_out/biomedeng_right8/` (left: 124 of 265 anatomical segments solved;
right ratio-8: 95 of 243). Medians per Strahler order.

### 1. The epicardial tree is not the resistance — and this is the gap, quantified

Static pressure across the whole solved tree spans **0.58 → 3.87 mmHg (left)** and
**−1.94 → 0.51 mmHg (right)**. Against a driving pressure of order 80–100 mmHg,
the entire resolved epicardial tree accounts for a few percent at most. **Over 95%
of coronary resistance lies distal to geometry that even HiP-CT resolves here.**

This is the strongest result for the gap framing: it converts "outlet BCs matter"
from an assertion into a measurement on one specimen. At this resolution the
solution is set by the outlet model, not by the geometry — so adding resolved
anatomy has diminishing returns *unless* it is used to constrain those outlets.
That is the argument for the next experiment rather than for more resolution.

### 2. WSS rises into the smaller vessels, monotonically in the right tree

| order | right ratio-8 WSS (Pa) | left full WSS (Pa) |
|---|---|---|
| 4 | 0.20 | 0.63 |
| 3 | 0.48 | 1.02 |
| 2 | 0.63 | 0.86 |
| 1 | **1.28** | **2.40** |

The right tree is cleanly monotonic over four orders, which is what an
approximately constant-WSS (Murray-type) design predicts and is worth showing as
an independent check on the reconstruction: it was not imposed anywhere.

**Do not show the left order-5 value (14.91 Pa).** That is the ostial stub, whose
radius (0.785 mm) is *smaller than its own order-4 daughters* (1.150 mm) — the
cannulation artefact `filter_segments` exists to remove. Its velocity, and so its
WSS, is inflated by the undersized inlet, not by physiology.

### 3. Resting flow is axial and unidirectional — the complexity does not show up

Axial fraction ⟨|v·t̂|⟩/⟨|v|⟩ is **0.92–1.00** and flow coherence **0.97–1.00** at
every order in both trees. Despite the tortuosity HiP-CT reveals, steady resting
flow is essentially simple through-flow; secondary and recirculating motion is not
a first-order effect here. The exception is **order 4, where the speed-magnitude
and axial flow estimates diverge by 16%** (19.12 vs 16.03 mL/min left; 5.56 vs
4.74 right) — the junction-rich order. If secondary flow matters anywhere in these
data, it is at the bifurcations, which is also where the reconstruction is weakest.

### 4. Morphometry is self-consistent but shows the tree is truncated

Strahler ratios over orders 1–4, ostial stub excluded (geometric means):
**R_b = 2.19, R_r = 1.78, R_l = 1.17**. R_b sits at the low end of published
coronary values (~2.6–3.0), and the diameter exponent
`D = log R_b / log R_r` = **1.35**, well below Murray's 3.0.

The reason is visible directly in the summed cross-sectional area, which *falls*
toward the periphery — 61.9 → 45.8 → 38.1 → **25.2 mm²** from order 4 to order 1.
In a complete arterial tree total CSA *increases* distally; that is why velocity
falls. A decreasing total CSA is a truncation signature: the order-1 vessels here
are terminals **of the imaged tree**, not true terminal arterioles, and small
daughters are being missed.

**Report D = 1.35 as a measurement of how far the reconstruction is from a
complete tree, not as a property of coronary anatomy.** It is consistent with the
independent finding that radius measurement has a hard floor at ~129 µm
(≈2 voxels at 65.98 µm) — see `HANDOFF_epicardial_radius.md` — and the order-1
median radius is 0.203 mm, close enough to that floor for both detection and
calibre to be degrading there.

### How these connect to the gap

The three results compose into one argument: the resolved anatomy is *sufficient*
to show that resistance is not in it (1), *good enough* to reproduce a scaling law
nobody imposed (2), and *demonstrably incomplete* at the small end (4). So the
useful next step is not more geometry for its own sake — it is using the resolved
tree to calibrate outlet models, with the truncation quantified rather than
assumed. That is exactly the controlled test named in "Research gap" above.

## Claim boundaries

Say:

- “This demonstrates technical feasibility.”
- “The CFD patterns are descriptive and hypothesis-generating.”
- “Validation is against the radius encoded in the spatial graph, not histology.”
- “Steady, resting, one specimen; left and right are separate solves, so compare
  within a tree and never across.”
- “Wall shear is the steady magnitude |τ_w| — CFX exports no direction here, so no
  OSI, RRT or reversal measure exists in these data.”
- “The tree is truncated at the small end; total CSA falls distally, which it
  should not, and D = 1.35 against Murray's 3.0 measures that gap.”

Do **not** say:

- “Flow rate” for the `flow_speed_ml_min` column — it is ⟨|v|⟩·A, an upper bound
  that does not conserve at a bifurcation. Use `flow_axial_ml_min`, or take flow
  from CFX `massFlow()` via `flow_fractions.py`.
- Anything about the ostial/order-5 segment: its radius is a cannulation artefact.
- “HiP-CT reduces anatomical uncertainty but does not restore in-vivo physiology to an ex vivo organ.”

Avoid:

- Calling the current WSS distribution physiologically validated.
- Presenting a single ex vivo tree as population-level coronary morphometry.
- Treating the overall radius bias as proof that all orders are equally accurate.
- Saying HiP-CT removes the need for outlet models; even HiP-CT does not explicitly resolve the complete microcirculation in the present CFD domain.

## Likely questions

**Why use 3D CFD rather than a 1D network?**  
The 3D model is needed for local velocity and wall-shear patterns at bends and bifurcations. A coupled 3D–0D or 3D–1D model is likely the scalable end point for the full distal tree.

**How is flow prescribed?**  
The feasibility model uses measured run-specific inlet mass flows (approximately 51 mL/min left and 22 mL/min right), a prescribed outlet allocation proportional to diameter to the power 2.27, laminar Quemada rheology and rigid walls. This is why the CFD result is presented as feasibility rather than physiological validation.

**Why does order 4 show radius expansion?**  
The implicit blending needed for watertight junctions can over-expand large proximal or bifurcation regions. The order-resolved validation exposes this even though the overall bias is small.

**What is the decisive next experiment?**  
Progressively truncate the high-resolution tree at clinical-like diameter thresholds, compare each reduced model with the high-resolution reference, and determine how resistance/compliance must change to preserve pressure, flow split and WSS.

**Can ex vivo HiP-CT be translated directly to in-vivo flow?**  
No. Ex vivo fixation, vessel collapse and the absence of active microvascular tone must be accounted for. HiP-CT is best used here as a high-resolution anatomical reference and a platform for sensitivity analysis.

## Evidence used for the gap

- Achenbach et al. describe the quantitative spatial-resolution limit of coronary CTA and illustrate how a 0.5 mm voxel samples a 3 mm lumen with only about six voxels: <https://www.jacc.org/doi/10.1016/j.jacc.2009.11.013>
- Sankaran et al. quantify the impact of geometry, boundary resistance and viscosity uncertainty in coronary simulations: <https://www.sciencedirect.com/science/article/pii/S0021929016000117>
- van der Giessen et al. describe coronary outlet-flow scaling and show its influence on WSS distributions: <https://www.sciencedirect.com/science/article/pii/S0021929011000789>
- Walsh et al. introduce HiP-CT for non-destructive multiscale imaging of intact human organs: <https://www.nature.com/articles/s41592-021-01317-x>
- Brunet et al. demonstrate whole-heart HiP-CT at approximately 20 micrometre voxels with higher-resolution local imaging of coronary anatomy: <https://pubs.rsna.org/doi/10.1148/radiol.232731>
- Expert recommendations on coronary WSS modelling emphasise the importance of specialised coronary boundary conditions for pulsatile analysis: <https://pmc.ncbi.nlm.nih.gov/articles/PMC6823616/>
