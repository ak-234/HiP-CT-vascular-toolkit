#!/usr/bin/env python3
"""resistance_based_outlet_BC.py - resistance-coupled pressure-outlet CCL writer.

Companion to flow_fractions.py. Where flow_fractions.py writes mass-flow outlets
(``MoutletNode = massFlow()@Inlet * fraction``), this script writes *resistance-
coupled pressure outlets*: each outlet's static pressure is tied to the live inlet
flow, split by its Giessen fraction, via Ohm's law for fluids,

    P_i = P_distal + (Q_i * R_i),   with   Q_i = fraction_i * massFlow()@Inlet / Rho

so each outlet's driving flow is its anatomical share of the (stable) total inflow
rather than its own noisy measured outflow. No ``abs()`` on the inlet: CFX reports
inflow as positive (mirrors flow_fractions.py).

Resistance method (flow-fraction / Ohm):

    R_i = dP / Q_i,   Q_i = fraction_i * Q_total,   dP = P_mean_aortic - P_distal

where ``fraction_i`` is the Van der Giessen flow split already computed by
flow_fractions.py and stored in ``giessen_cfx_outlet_flow_fractions.csv``. Because
``R_i = dP / (fraction_i * Q_total)``, the ``fraction_i`` cancels in ``Q_i * R_i``:
every outlet in a tree tracks the same live inlet flow and reaches P_mean_aortic at
the design operating point (``massFlow()@Inlet = Rho * Q_total``). The result is a
stable, near-uniform prescribed outlet pressure; the CFD mesh sets the actual split.

Two outlet modes (``--mode``):
  * ``full`` (default)  - plain BC ``P_i = Pdistal + (fraction_i * massFlow()@Inlet / Rho) * R_i``,
    no step/clamp/ramp.
  * ``stabilized``      - ramp the resistance in over RAMP_DURATION_ITERS iterations
    (RampFactor, default 400) for a soft start at highly twisted ~150 um branch cuts
    (no pressure clamp).

Reads:  giessen_cfx_outlet_flow_fractions.csv  (per-CFX-outlet table)
Writes: a complete CFX-Pre CCL (domain + Quemada material + inlet + pressure outlets).

Usage:
    python resistance_based_outlet_BC.py --csv <...csv> --q-total <m^3/s> --out <...ccl>
"""
from __future__ import annotations

import argparse
import csv
from collections import OrderedDict
from pathlib import Path

# -- Physical / modelling constants (edit here or override via the CLI) --------

# Mean aortic (coronary perfusion) pressure and downstream microvascular/venous
# reference pressure. The design pressure drop is dP = P_MEAN_AORTIC - P_DISTAL.
P_MEAN_AORTIC_MMHG: float = 93.0
P_DISTAL_MMHG: float = 5.0
MMHG_TO_PA: float = 133.322

# Blood density (kg/m^3). Source of truth: flow_fractions.FLUID_DENSITY = 1060.0.
# Used both in R_i = dP/Q_i and as the CEL ``Rho`` that converts massFlow -> Q,
# and is written as the fluid material density so the two stay consistent.
RHO: float = 1060.0

# Total volumetric inflow PER TREE at the design point (m^3/s). REQUIRED:
# set Q_TOTAL_M3S (applies to every tree) and/or override individual trees in
# Q_TOTAL_BY_TREE keyed by the tree_id string from the CSV, e.g.
#   Q_TOTAL_BY_TREE = {"(0, 19)": 2.5e-6}
# A left coronary tree is typically ~1-4 mL/s = 1e-6 .. 4e-6 m^3/s.
Q_TOTAL_M3S: float | None = None
Q_TOTAL_BY_TREE: dict[str, float] = {}

# Outlet pressure formulation (toggle with --mode):
#   "full"       -> plain BC: P<i> = Pdistal + (frac<i> * massFlow()@Inlet / Rho) * R<i>
#                   (no step, no clamp, no ramp).
#   "stabilized" -> ramp the resistance in over RAMP_DURATION_ITERS iters (RampFactor)
#                   for a soft start (no pressure clamp).
OUTLET_MODE: str = "full"

# Solution ramp: outlets act as a constant P_distal boundary for the first
# RAMP_ITERATIONS iterations (resistance switches on at RAMP_ITERATIONS + 1) via
# step(aitern - RAMP_ITERATIONS). Set to 0 to disable the ramp. Only used in
# "stabilized" mode.
RAMP_ITERATIONS: int = 30

# Duration (iterations) over which RampFactor grows 0 -> 1, starting at RAMP_ITERATIONS.
# Longer = gentler introduction of the stiff resistance coupling. Stabilized mode only.
RAMP_DURATION_ITERS: int = 400

# Average Static Pressure profile blend: fraction of local profile variation
# allowed across the outlet face while the area-average target is enforced.
PRESSURE_PROFILE_BLEND: float = 0.05

# Explicit physical timescale (s) for the steady-state run. Do NOT use Auto
# Timescale with pressure-flow coupling. Set ~ L_tree / (2 * V_inlet); reduce by
# 2-5x if the pressure-flow feedback loop oscillates.
PHYSICAL_TIMESCALE_S: float = 0.0001

# Solver iteration cap. Must comfortably exceed RAMP_ITERATIONS + RAMP_DURATION_ITERS so
# the run has room to converge after the resistance reaches full strength.
MAX_ITERATIONS: int = 1000

CFX_DOMAIN: str = "Default Domain"   # mirrors flow_fractions.CFX_DOMAIN
INLET_PREFIX: str = "Inlet"          # mirrors flow_fractions.RENAME_MSH_INLET_PREFIX
INLET_NUM_DIGITS: int = 3            # mirrors flow_fractions.RENAME_MSH_NUM_DIGITS

DEFAULT_CSV_NAME = "giessen_cfx_outlet_flow_fractions.csv"
DEFAULT_CCL_NAME = "resistance_boundary_conditions.ccl"

EPS_FRACTION = 1e-6   # floor for orphan/zero-fraction outlets

# Build-time sanity monitor: warn if an outlet's design-point pressure
# (Pdistal + Q_i,design * R_i) strays this far from P_mean_aortic. At the design
# point it should equal P_mean_aortic exactly, so any deviation flags a
# units/Rho/fraction inconsistency.
P_DESIGN_TOL_MMHG: float = 0.5


# -- Input -------------------------------------------------------------------


def read_outlets(csv_path: str | Path) -> list[dict]:
    """Read the per-CFX-outlet table written by flow_fractions.run().

    Returns one dict per outlet with keys: idx, ccl_name, fraction, diam_mm,
    tree_id. ``tree_id`` is kept as the opaque string from the CSV (e.g.
    ``"(0, 19)"``) and used only as a grouping key.
    """
    rows: list[dict] = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append({
                "idx": int(r["idx"]),
                "ccl_name": r["ccl_name"].strip(),
                "fraction": float(r["fraction"]),
                "diam_mm": float(r["diam_mm"]),
                "tree_id": r["tree_id"].strip(),
            })
    if not rows:
        raise SystemExit(f"[ERROR] No outlet rows found in {csv_path}")
    return rows


# -- Resistance computation --------------------------------------------------


def _resolve_q_total(tree_id: str, q_total_default: float | None) -> float:
    q = Q_TOTAL_BY_TREE.get(tree_id, q_total_default)
    if q is None:
        raise SystemExit(
            f"[ERROR] No Q_total set for tree {tree_id}. Pass --q-total <m^3/s>, "
            f"set Q_TOTAL_M3S, or add an entry to Q_TOTAL_BY_TREE."
        )
    if q <= 0:
        raise SystemExit(f"[ERROR] Q_total for tree {tree_id} must be positive (got {q}).")
    if q > 1e-2:
        print(f"[WARN] Q_total for tree {tree_id} = {q:g} m^3/s looks large "
              f"(coronary trees are ~1e-6 m^3/s) - did you mean mL/s? 1 mL/s = 1e-6 m^3/s.")
    return q


def compute_resistances(outlets: list[dict], q_total_default: float | None,
                        dp_pa: float, p_distal_pa: float,
                        p_aortic_mmHg: float) -> "OrderedDict[str, list[dict]]":
    """Group outlets by tree and assign ``R`` [Pa s m^-3] = R_total / fraction, where
    ``R_total = dP / Q_total`` is the total coronary resistance for the tree, split per
    outlet by the inverse Giessen fraction.

    Also stores ``q_i``, ``q_total`` and ``frac_eff`` per outlet, prints a summary
    table, and warns if a tree's fractions do not sum to ~1.0.

    Build-time pressure monitor: for each outlet the design-point pressure
    ``Pdistal + Q_i,design * R_i`` is reconstructed the same way the CEL computes
    it (``Pdistal + (frac_i * massFlow()@Inlet / Rho) * R_i`` at the design inlet
    flow ``massFlow()@Inlet = Rho * Q_total``) and printed in the summary. Because
    ``R_i = dP / Q_i,design``, this must equal P_mean_aortic
    for every outlet; a deviation beyond ``P_DESIGN_TOL_MMHG`` flags a
    units/Rho/fraction bug. A per-tree footer then confirms the individual
    10^11..10^13 resistances recombine in parallel to ``dP / Q_total``.
    """
    by_tree: "OrderedDict[str, list[dict]]" = OrderedDict()
    for o in outlets:
        by_tree.setdefault(o["tree_id"], []).append(o)

    for tid, outs in by_tree.items():
        q_total = _resolve_q_total(tid, q_total_default)
        fsum = sum(o["fraction"] for o in outs)
        if abs(fsum - 1.0) > 0.02:
            print(f"[WARN] tree {tid}: outlet fractions sum to {fsum:.4f}, expected ~1.0")

        # Total coronary bed resistance for this tree; each outlet then gets
        # R_i = R_total / fraction_i (split by the inverse Giessen fraction).
        r_total = dp_pa / q_total
        r_total_clin = r_total * (1.0 / MMHG_TO_PA) * 1e-6   # Pa.s/m^3 -> mmHg.s/mL

        print(f"\nTree {tid}:  Q_total = {q_total:.4e} m^3/s,  dP = {dp_pa:.1f} Pa,  "
              f"outlets = {len(outs)}")
        print(f"  R_total = dP/Q_total = {r_total:.4e} Pa.s/m^3 = {r_total_clin:.3f} "
              f"mmHg.s/mL  (split per outlet by 1/fraction)")
        print(f"  {'outlet':<11}{'frac':>10}{'d_mm':>9}{'Q_i (m^3/s)':>16}"
              f"{'R_i (Pa.s/m^3)':>18}{'P_dsgn(mmHg)':>15}")
        for o in outs:
            frac = o["fraction"]
            if frac <= 0:
                print(f"[WARN] {o['ccl_name']}: fraction <= 0 (orphan zone?); "
                      f"clamping to {EPS_FRACTION:g}")
                frac = EPS_FRACTION
            q_i = frac * q_total
            o["q_i"] = q_i
            o["q_total"] = q_total
            o["frac_eff"] = frac   # clamped fraction that R_i / q_i were built from
            o["r_total"] = r_total
            o["R"] = r_total / frac   # split total resistance by inverse Giessen fraction
            # Design-point outlet pressure, reconstructed exactly as the CEL does
            # (Pdistal + (frac_i * massFlow()@Inlet / Rho) * R_i). At the design inlet
            # flow massFlow()@Inlet = Rho*Q_total, so frac_i*Q_total*R_i = Q_i*R_i.
            # By construction R = dP/Q_i, so this must read P_mean_aortic.
            p_design_pa = p_distal_pa + q_i * o["R"]
            p_design_mmhg = p_design_pa / MMHG_TO_PA
            o["p_design_mmHg"] = p_design_mmhg
            print(f"  {o['ccl_name']:<11}{o['fraction']:>10.5f}{o['diam_mm']:>9.4f}"
                  f"{q_i:>16.4e}{o['R']:>18.4e}{p_design_mmhg:>15.2f}")
            if abs(p_design_mmhg - p_aortic_mmHg) > P_DESIGN_TOL_MMHG:
                print(f"[WARN] {o['ccl_name']}: design-point pressure {p_design_mmhg:.2f} mmHg "
                      f"deviates from P_mean_aortic {p_aortic_mmHg:.2f} mmHg by more than "
                      f"{P_DESIGN_TOL_MMHG:g} mmHg - check Rho/units/fraction.")

        # Per-tree footer: the individual 10^11..10^13 resistances must recombine
        # in parallel to R_total = dP/Q_total (parallel consistency of the split).
        r_parallel = 1.0 / sum(1.0 / o["R"] for o in outs)
        resid = abs(r_parallel - r_total) / r_total
        r_vals = [o["R"] for o in outs]
        print(f"  parallel sum of per-outlet R = {r_parallel:.4e} Pa.s/m^3 "
              f"(residual {resid:.2e} vs R_total); R_i range [{min(r_vals):.3e}, {max(r_vals):.3e}]")
        if resid > 1e-6:
            print(f"[WARN] tree {tid}: parallel combination of per-outlet R "
                  f"({r_parallel:.4e}) does not match R_total ({r_total:.4e}); "
                  f"residual {resid:.2e} - check grouping/summation.")
    return by_tree


# -- CCL generation ----------------------------------------------------------


def _fmt_inlet_name(n: int) -> str:
    return f"{INLET_PREFIX}{n:0{INLET_NUM_DIGITS}d}"


def generate_resistance_ccl(by_tree: "OrderedDict[str, list[dict]]",
                            params: dict, out_path: str | Path) -> Path:
    """Write a complete CFX-Pre CCL with resistance-coupled pressure outlets.

    Mirrors the structure of flow_fractions.generate_cfx_ccl (domain + Quemada
    material + inlet block), but replaces every mass-flow outlet with an Average
    Static Pressure outlet driven by the ``P_<idx>`` Ohm's-law expression.
    """
    dp_pa = params["dp_pa"]
    ramp = params["ramp"]
    dur = params["ramp_duration"]
    max_iters = params["max_iters"]
    stabilize = params["mode"] == "stabilized"
    lines: list[str] = []

    # One inlet per distinct tree, named in CSV order.
    inlet_name_by_tree = {tid: _fmt_inlet_name(n) for n, tid in enumerate(by_tree)}

    lines.append("# ============================================================")
    lines.append("# Resistance-coupled pressure-outlet boundary conditions for ANSYS CFX-Pre")
    lines.append("# Generated by resistance_based_outlet_BC.py")
    lines.append(f"# Fluid density Rho = {RHO:g} kg/m^3")
    lines.append(f"# P_mean_aortic = {params['p_aortic_mmHg']:g} mmHg, "
                 f"P_distal = {params['p_distal_mmHg']:g} mmHg, "
                 f"design dP = {dp_pa:.1f} Pa")
    lines.append("# Outlet pressure:  P_i = Pdistal + (fraction_i * massFlow()@Inlet / Rho) * R_i")
    lines.append("#             R_i = R_total / fraction_i  (R_total = dP / Q_total)  [Pa s m^-3]")
    lines.append("#             Driving flow is the outlet's Giessen share of the LIVE inlet")
    lines.append("#             inflow (not the measured outlet outflow). Since fraction_i")
    lines.append("#             cancels against R_i, all outlets in a tree track the inlet flow")
    lines.append("#             and reach P_mean_aortic at the design point (massFlow@Inlet=Rho*Q_total).")
    if stabilize:
        lines.append(f"# Mode: stabilized - resistance ramped in over {dur} iters via RampFactor "
                     "(no clamp).")
        if ramp > 0:
            lines.append(f"# Resistance ramps on from iteration {ramp} via the RampFactor expression.")
    else:
        lines.append("# Mode: full - plain full-resistance outlets (no step, clamp, or ramp).")
    lines.append("# Each Inlet mass flow below is pre-filled to Rho*Q_total; confirm it in CFX-Pre.")
    lines.append("# ============================================================\n")

    # ---- LIBRARY / CEL / EXPRESSIONS ----
    lines.append("LIBRARY:")
    lines.append("  CEL:")
    lines.append("    EXPRESSIONS:")
    # Quemada non-Newtonian rheology (matches flow_fractions.generate_cfx_ccl).
    lines.append("      gr = Shear Strain Rate/grc")
    lines.append("      grc = 2.3 [s^-1]")
    lines.append("      k0 = 3.691")
    lines.append("      kinf = 1.778")
    lines.append("      muF = 1.32e-3 [Pa s]")
    lines.append("      muQmd = muF*(1-0.5*phi*(k0+kinf*sqrt(gr))/(1+sqrt(gr)))^(-2)")
    lines.append("      phi = 0.43")
    lines.append("")
    lines.append("      # Downstream reference pressure and blood density")
    lines.append(f"      Pdistal = {params['p_distal_mmHg']:g} [mmHg]")
    lines.append(f"      Rho = {RHO:g} [kg m^-3]")
    lines.append("")
    lines.append("      # Per-outlet numerical resistance  R_i = R_total / fraction_i  (R_total = dP / Q_total)")
    for outs in by_tree.values():
        for o in outs:
            lines.append(
                f"      R{o['idx']} = {o['R']:.6e} [Pa s m^-3]"
                f" # {o['ccl_name']}  d={o['diam_mm']:.4f}mm  frac={o['fraction']:.6f}"
            )
    lines.append("")
    # Live inlet volumetric flow per tree, split by the Giessen fraction. No abs():
    # CFX reports inlet inflow as positive (mirrors flow_fractions.generate_cfx_ccl).
    lines.append("      # Live inlet volumetric flow per tree  Qin = massFlow()@Inlet / Rho")
    for n, (tid, outs) in enumerate(by_tree.items()):
        lines.append(f"      Qin{n} = massFlow()@{inlet_name_by_tree[tid]} / Rho"
                     f" # tree {tid}")
    lines.append("")
    lines.append("      # Giessen flow fraction per outlet (clamped to EPS for orphans)")
    for outs in by_tree.values():
        for o in outs:
            lines.append(f"      frac{o['idx']} = {o['frac_eff']:.6f} # {o['ccl_name']}")
    lines.append("")
    if stabilize:
        lines.append("      # Smooth convergence ramp using native CFX if-logic")
        # Linear ramp from 0 to 1 between 'ramp' and 'ramp + dur' iterations
        lines.append(
            f"      RampFactor = if(aitern < {ramp}, 0.0, "
            f"if(aitern > ({ramp} + {dur}), 1.0, (aitern - {ramp}) / {dur}))"
        )
        lines.append("")
        lines.append("      # Outlet pressure: driving flow Q_i = frac_i * (massFlow()@Inlet / Rho), ramped (no clamp)")
        for n, (tid, outs) in enumerate(by_tree.items()):
            for o in outs:
                idx = o['idx']
                lines.append(f"      P{idx} = Pdistal + (frac{idx} * Qin{n}) * R{idx} * RampFactor")
    else:
        lines.append("      # Outlet pressure: driving flow Q_i = frac_i * (massFlow()@Inlet / Rho)")
        for n, (tid, outs) in enumerate(by_tree.items()):
            for o in outs:
                idx = o['idx']
                lines.append(f"      P{idx} = Pdistal + (frac{idx} * Qin{n}) * R{idx}")
    lines.append("    END")
    lines.append("  END")
    # Quemada material; density matched to RHO so massFlow/Rho gives true Q.
    lines.append("  MATERIAL: Quemada")
    lines.append("    Material Group = User")
    lines.append("    Option = Pure Substance")
    lines.append("    PROPERTIES:")
    lines.append("      Option = General Material")
    lines.append("      EQUATION OF STATE:")
    lines.append(f"        Density = {RHO:g} [kg m^-3]")
    lines.append("        Molar Mass = 1.0 [kg kmol^-1]")
    lines.append("        Option = Value")
    lines.append("      END")
    lines.append("      DYNAMIC VISCOSITY:")
    lines.append("        Dynamic Viscosity = muQmd")
    lines.append("        Option = Value")
    lines.append("      END")
    lines.append("    END")
    lines.append("  END")
    lines.append("END")
    lines.append("")

    # ---- FLOW / DOMAIN ----
    lines.append("FLOW: Flow Analysis 1")
    lines.append(f"  DOMAIN: {CFX_DOMAIN}")
    lines.append("    Coord Frame = Coord 0")
    lines.append("    Domain Type = Fluid")
    lines.append("    # Location = <SET 3D MESH REGION HERE> # bind to the imported fluid zone")
    lines.append("    DOMAIN MODELS:")
    lines.append("      BUOYANCY MODEL:")
    lines.append("        Option = Non Buoyant")
    lines.append("      END")
    lines.append("      DOMAIN MOTION:")
    lines.append("        Option = Stationary")
    lines.append("      END")
    lines.append("      MESH DEFORMATION:")
    lines.append("        Option = None")
    lines.append("      END")
    lines.append("      REFERENCE PRESSURE:")
    lines.append("        Reference Pressure = 1 [atm]"
                 " # gauge datum: prescribed 5-93 mmHg are relative to atmospheric")
    lines.append("      END")
    lines.append("    END")
    lines.append("    FLUID DEFINITION: Fluid 1")
    lines.append("      Material = Quemada")
    lines.append("      Option = Material Library")
    lines.append("      MORPHOLOGY:")
    lines.append("        Option = Continuous Fluid")
    lines.append("      END")
    lines.append("    END")
    lines.append("    FLUID MODELS:")
    lines.append("      COMBUSTION MODEL:")
    lines.append("        Option = None")
    lines.append("      END")
    lines.append("      HEAT TRANSFER MODEL:")
    lines.append("        Option = None")
    lines.append("      END")
    lines.append("      THERMAL RADIATION MODEL:")
    lines.append("        Option = None")
    lines.append("      END")
    lines.append("      TURBULENCE MODEL:")
    lines.append("        Option = Laminar")
    lines.append("      END")
    lines.append("    END")

    # ---- INLET boundary per tree ----
    for tid, outs in by_tree.items():
        inlet_name = inlet_name_by_tree[tid]
        q_total = outs[0]["q_total"]
        mdot = RHO * q_total
        lines.append(f"    BOUNDARY: {inlet_name}")
        lines.append("      Boundary Type = INLET")
        lines.append(f"      Location = {inlet_name}"
                     f" # tree {tid}; rename to the mesh inlet zone if different")
        lines.append("      BOUNDARY CONDITIONS:")
        lines.append("        FLOW DIRECTION:")
        lines.append("          Option = Normal to Boundary Condition")
        lines.append("        END")
        lines.append("        FLOW REGIME:")
        lines.append("          Option = Subsonic")
        lines.append("        END")
        lines.append("        MASS AND MOMENTUM:")
        lines.append("          Option = Bulk Mass Flow Rate")
        lines.append(f"          Mass Flow Rate = {mdot:.6e} [kg s^-1]"
                     f" # = Rho*Q_total (Q_total={q_total:.3e} m^3/s); confirm in CFX-Pre")
        lines.append("        END")
        lines.append("      END")
        lines.append("    END")

    # ---- OUTLET boundaries: Average Static Pressure driven by P_<idx> ----
# ---- OUTLET boundaries: Average Static Pressure driven by P<idx> ----
    for outs in by_tree.values():
        for o in outs:
            lines.append(f"    BOUNDARY: {o['ccl_name']}")
            lines.append("      Boundary Type = OUTLET")
            lines.append(f"      Location = {o['ccl_name']}")
            lines.append("      BOUNDARY CONDITIONS:")
            lines.append("        FLOW REGIME:")
            lines.append("          Option = Subsonic")
            lines.append("        END")
            lines.append("        MASS AND MOMENTUM:")
            lines.append("          Option = Average Static Pressure")
            lines.append(f"          Relative Pressure = P{o['idx']}") 
            lines.append(f"          Pressure Profile Blend = {params['blend']:g}")
            lines.append("        END")            
            lines.append("        PRESSURE AVERAGING:")
            lines.append("          Option = Average Over Whole Outlet")
            lines.append("        END")
            lines.append("      END")  # Closes BOUNDARY CONDITIONS
            lines.append("    END")    # Closes BOUNDARY


    lines.append("  END")  # end DOMAIN

    # ---- SOLVER CONTROL: explicit physical timescale (never Auto here) ----
    lines.append("  SOLVER CONTROL:")
    lines.append("    ADVECTION SCHEME:")
    lines.append("      Option = High Resolution")
    lines.append("    END")
    lines.append("    CONVERGENCE CONTROL:")
    lines.append("      Length Scale Option = Conservative")
    lines.append(f"      Maximum Number of Iterations = {max_iters}")
    lines.append("      Minimum Number of Iterations = 1")
    lines.append("      Timescale Control = Physical Timescale")
    lines.append(f"      Physical Timescale = {params['timescale']:g} [s]"
                 " # set ~ L_tree/(2*V_inlet); reduce 2-5x if pressure-flow oscillates")
    lines.append("    END")
    lines.append("    CONVERGENCE CRITERIA:")
    lines.append("      Residual Target = 1e-4")
    lines.append("      Residual Type = RMS")
    lines.append("    END")
    lines.append("  END")

    # ---- INITIALISATION: start the field at the distal pressure ----
    lines.append("  INITIALISATION:")
    lines.append("    Option = Automatic")
    lines.append("    INITIAL CONDITIONS:")
    lines.append("      Velocity Type = Cartesian")
    lines.append("      CARTESIAN VELOCITY COMPONENTS:")
    lines.append("        Option = Automatic with Value")
    lines.append("        U = 0 [m s^-1]")
    lines.append("        V = 0 [m s^-1]")
    lines.append("        W = 0 [m s^-1]")
    lines.append("      END")
    lines.append("      STATIC PRESSURE:")
    lines.append("        Option = Automatic with Value")
    lines.append("        Relative Pressure = Pdistal")
    lines.append("      END")
    lines.append("    END")
    lines.append("  END")
    lines.append("END")

    out_path = Path(out_path)
    out_path.write_text("\n".join(lines))
    print(f"\n[CCL] Wrote resistance boundary conditions to {out_path}")
    return out_path


# -- CLI ---------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--csv", type=Path, default=None,
                   help=f"path to {DEFAULT_CSV_NAME} (default: alongside this script)")
    p.add_argument("--out", type=Path, default=None,
                   help=f"output .ccl path (default: <csv dir>/{DEFAULT_CCL_NAME})")
    p.add_argument("--q-total", type=float, default=Q_TOTAL_M3S,
                   help="total volumetric inflow per tree (m^3/s); required if not set in-file")
    p.add_argument("--p-aortic", type=float, default=P_MEAN_AORTIC_MMHG,
                   help="mean aortic pressure (mmHg)")
    p.add_argument("--p-distal", type=float, default=P_DISTAL_MMHG,
                   help="distal reference pressure (mmHg)")
    p.add_argument("--mode", choices=["full", "stabilized"], default=OUTLET_MODE,
                   help="outlet pressure formulation: 'full' (plain resistance, no ramp/clamp) "
                        "or 'stabilized' (RampFactor ramp, no clamp)")
    p.add_argument("--ramp", type=int, default=RAMP_ITERATIONS,
                   help="ramp start iteration for the RampFactor expression (stabilized mode only)")
    p.add_argument("--blend", type=float, default=PRESSURE_PROFILE_BLEND,
                   help="Average Static Pressure profile blend (0-1)")
    p.add_argument("--timescale", type=float, default=PHYSICAL_TIMESCALE_S,
                   help="physical timescale (s) for Solver Control")
    p.add_argument("--ramp-duration", type=int, default=RAMP_DURATION_ITERS,
                   help="iterations over which RampFactor ramps 0->1 (stabilized mode)")
    p.add_argument("--max-iters", type=int, default=MAX_ITERATIONS,
                   help="Maximum Number of Iterations for Solver Control")
    args = p.parse_args(argv)

    if args.max_iters <= args.ramp + args.ramp_duration:
        print(f"[WARN] max-iters ({args.max_iters}) <= ramp start ({args.ramp}) + duration "
              f"({args.ramp_duration}); resistance never reaches full strength within the run.")

    csv_path = args.csv or (Path(__file__).resolve().parent / DEFAULT_CSV_NAME)
    if not Path(csv_path).is_file():
        raise SystemExit(
            f"[ERROR] CSV not found: {csv_path}\n"
            f"        Run flow_fractions.py first, or pass --csv <path to {DEFAULT_CSV_NAME}>."
        )
    out_path = args.out or (Path(csv_path).parent / DEFAULT_CCL_NAME)

    dp_pa = (args.p_aortic - args.p_distal) * MMHG_TO_PA
    if dp_pa <= 0:
        raise SystemExit(
            f"[ERROR] Non-positive design pressure drop: P_aortic ({args.p_aortic} mmHg) "
            f"must exceed P_distal ({args.p_distal} mmHg)."
        )

    outlets = read_outlets(csv_path)
    by_tree = compute_resistances(outlets, args.q_total, dp_pa,
                                  args.p_distal * MMHG_TO_PA, args.p_aortic)

    params = {
        "dp_pa": dp_pa,
        "p_aortic_mmHg": args.p_aortic,
        "p_distal_mmHg": args.p_distal,
        "mode": args.mode,
        "ramp": max(0, args.ramp),
        "ramp_duration": max(1, args.ramp_duration),
        "blend": args.blend,
        "timescale": args.timescale,
        "max_iters": args.max_iters,
    }
    generate_resistance_ccl(by_tree, params, out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
