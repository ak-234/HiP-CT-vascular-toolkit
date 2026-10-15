"""One common terminal-load scale per independently supplied coronary tree."""
from __future__ import annotations

import math
import networkx as nx

from . import MMHG_TO_PA
from .network import positive, partition, solve, calibrate_scale, structured_tree
from .dynamics import parameters

BASELINE = dict(myocardial_mass_g=91.62400695, perfusion_ml_min_g=0.8,
                density_kg_m3=1050.0, viscosity_pa_s=0.0035,
                arterial_mmhg=93.0, distal_mmhg=5.0)


def physiology(config):
    p = {**BASELINE, **config.get("physiology", {})}
    for key in ("myocardial_mass_g", "perfusion_ml_min_g", "density_kg_m3", "viscosity_pa_s"):
        positive(p[key], key)
    if not math.isfinite(p["distal_mmhg"]) or not math.isfinite(p["arterial_mmhg"]):
        raise ValueError("Invalid pressure datum")
    p["delta_p_pa"] = positive((p["arterial_mmhg"]-p["distal_mmhg"])*MMHG_TO_PA, "pressure drop")
    p["flow_m3_s"] = p["myocardial_mass_g"]*p["perfusion_ml_min_g"]/6e7
    p["whole_resistance_Pa_s_m3"] = p["delta_p_pa"]/p["flow_m3_s"]
    return p


def legacy_fractions(tree, retained):
    """Local daughter-diameter^m splitting, propagated only on the clinical tree."""
    from coronary_sdf.flow_fractions import _branch_diameter
    g = nx.Graph()
    by_pair = {}
    for e in tree["edges"]:
        if str(e["id"]) in set(map(str, tree["retained_edge_ids"])):
            u, v = str(e["u"]), str(e["v"])
            g.add_edge(u, v)
            by_pair[frozenset((u, v))] = e
    if not nx.is_tree(g):
        raise ValueError("Recursive Giessen allocation requires an acyclic clinical tree")
    root = str(tree["root"])
    weight = {root: 1.0}
    m = float(tree.get("diameter_exponent", 2.27))
    for node, children in nx.bfs_successors(g, root):
        sizes = []
        for child in children:
            edge = by_pair[frozenset((node, child))]
            # Reuse the established clean-tube window (skip junction contours,
            # average downstream samples, minimum-radius short-branch fallback).
            points = {i: (0., 0., 0., r*1e6) for i, r in enumerate(edge["radius_m"])}
            segment = dict(node1=str(edge["u"]), node2=str(edge["v"]), point_ids=list(points))
            nodes = {n: (0., 0., 0., g.degree(n)) for n in (str(edge["u"]), str(edge["v"]))}
            diameter, _ = _branch_diameter(segment, node, points, nodes)
            sizes.append(positive(diameter, "daughter diameter")**m)
        for child, size in zip(children, sizes):
            weight[child] = weight[node]*size/sum(sizes)
    f = {str(o["node"]): weight[str(o["node"])] for o in tree["outlets"]}
    if not math.isclose(sum(f.values()), 1, rel_tol=1e-8):
        raise ValueError("Legacy outlets do not cover all clinical leaves")
    return f


def calibrate_tree(tree, config, method, target_flow):
    p = physiology(config)
    pv = p["distal_mmhg"]*MMHG_TO_PA
    pa = p["arterial_mmhg"]*MMHG_TO_PA
    retained, measured_beds = partition(tree, p["viscosity_pa_s"])
    outlets = sorted(tree["outlets"], key=lambda o: str(o["id"]))
    exponent = positive(config.get("diameter_exponent", 2.27), "diameter exponent")
    rref = positive(config.get("terminal_reference_radius_m", 0.0001), "terminal reference radius")
    rbase = positive(config.get("terminal_reference_resistance", 1e12), "terminal reference resistance")
    if method == "diameter":
        weights = {str(o["node"]): positive(o["radius_m"], "outlet radius")**exponent for o in outlets}
        weights = {node: value/sum(weights.values()) for node, value in weights.items()}
    elif method == "legacy-giessen":
        weights = legacy_fractions({**tree, "diameter_exponent": exponent}, retained)
    elif method == "hipct":
        beds = measured_beds
    elif method == "structured":
        beds = {str(o["node"]): structured_tree(o["radius_m"], config["structured_tree"],
                str(o["node"]), p["viscosity_pa_s"]) for o in outlets}
    else:
        raise ValueError(f"Unknown resistance method {method}")

    def loads_at(scale):
        if method in ("diameter", "legacy-giessen"):
            return {node: scale*rbase/f for node, f in weights.items()}
        loads = {}
        for node, (edges, radii) in beds.items():
            terminal = {n: scale*rbase*(rref/r)**exponent for n, r in radii.items()}
            # Unit pressure makes this independent of the chosen physiological state.
            bed = solve(edges, node, terminal, 1., 0.)
            loads[node] = 1/positive(bed.inlet_flow, "subtree conductance")
        return loads

    root = str(tree["root"])
    def flow(scale):
        return solve(retained, root, loads_at(scale), pa, pv).inlet_flow
    scale = calibrate_scale(flow, target_flow)
    loads = loads_at(scale)
    solution = solve(retained, root, loads, pa, pv)
    if not math.isclose(solution.inlet_flow, target_flow, rel_tol=1e-7):
        raise ValueError("Calibration missed the target flow")
    rows = []
    for k, outlet in enumerate(outlets):
        node, outlet_id = str(outlet["node"]), str(outlet["id"])
        rt = loads[node]
        r = positive(outlet["radius_m"], "radius")
        area = positive(outlet.get("area_m2", math.pi*r*r), "outlet area")
        fit = parameters(rt, r, area, p["density_kg_m3"], config["rcr"],
                         samples=outlet.get("impedance_samples"))
        qi = solution.terminal_flow[node]
        rows.append(dict(tree_id=str(tree["id"]), outlet_id=outlet_id, node=node,
            cfx_outlet=f"Outlet{k}", region_name=outlet.get("region_name", ""),
            radius_m=r, area_m2=area, flow_m3_s=qi, mass_flow_kg_s=qi*p["density_kg_m3"],
            flow_fraction=qi/target_flow, distal_pressure_pa=pv,
            pressure_datum="CFX gauge, reference 1 atm", units="SI: Pa, m, s, kg",
            radius_provenance=outlet.get("radius_provenance", "hydraulic profile; see network provenance"),
            initial_pc_pa=pv+qi*fit["Rd_Pa_s_per_m3"],
            initial_outlet_pressure_pa=solution.pressure[node], resistance_method=method,
            terminal_scale=scale, **fit))
    legacy_settings = None
    if method == "legacy-giessen":
        from coronary_sdf import flow_fractions
        from .prepare import digest
        legacy_settings = dict(skip_points=flow_fractions.BIF_SKIP_POINTS,
            downstream_points=flow_fractions.FRAMES_DOWNSTREAM,
            source_sha256=digest(flow_fractions.__file__))
    return dict(tree_id=str(tree["id"]), method=method, target_flow_m3_s=target_flow,
                legacy_settings=legacy_settings,
                achieved_flow_m3_s=solution.inlet_flow, terminal_scale=scale,
                whole_resistance_Pa_s_m3=p["delta_p_pa"]/target_flow,
                relative_mass_error=solution.relative_mass_error,
                retained_pressure_pa=solution.pressure, retained_edge_flow_m3_s=solution.edge_flow,
                outlets=rows)


def calibrate_case(network, config, methods):
    p = physiology(config)
    trees = network["trees"]
    fractions = config.get("tree_fractions", {"left": 0.7, "right": 0.3})
    if set(fractions) != {str(t["id"]) for t in trees}:
        raise ValueError("tree_fractions must name every independent network tree exactly once")
    if not math.isclose(sum(positive(x, "tree fraction") for x in fractions.values()), 1, abs_tol=1e-10):
        raise ValueError("Tree fractions must sum to one (do not apply total flow to each tree)")
    return dict(schema="coronary-rcr-results/1", physiology=p,
                network_provenance=network.get("provenance", {}),
                results=[calibrate_tree(t, config, method, p["flow_m3_s"]*fractions[str(t["id"])])
                         for method in methods for t in trees])
