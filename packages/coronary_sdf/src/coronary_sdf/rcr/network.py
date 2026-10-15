"""SI hydraulic networks; terminal loads remain attached to their actual nodes."""
from __future__ import annotations

from dataclasses import dataclass
import math

import networkx as nx
import numpy as np
from scipy.optimize import brentq
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve


def positive(value, name):
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive, got {value}")
    return value


def segment_resistance(edge, viscosity_pa_s=0.0035):
    """Integrate 8 mu / (pi r^4); exact for piecewise-linear radius profiles.

    This avoids sampling-dependent trapezoidal bias in sharply tapering vessels.
    Radii must be hydraulic measurements/support, never surface carina tips.
    """
    if edge.get("hydraulic_supported", True) is not True:
        raise ValueError(f"Unsupported hydraulic radius on edge {edge.get('id')}")
    r = np.asarray(edge["radius_m"], dtype=float)
    s = np.asarray(edge["s_m"], dtype=float)
    if r.ndim != 1 or len(r) < 2 or r.shape != s.shape:
        raise ValueError("Each edge requires equally sized s_m/radius_m profiles (>=2)")
    if not np.all(np.isfinite(r)) or np.any(r <= 0) or not np.all(np.isfinite(s)):
        raise ValueError("Invalid radius/arclength samples")
    ds = np.diff(s)
    if np.any(ds <= 0):
        raise ValueError("Arclength samples must increase strictly")
    a, b = r[:-1], r[1:]
    integral = np.sum(ds * (a*a + a*b + b*b) / (3*a**3*b**3))
    return float(8 * positive(viscosity_pa_s, "viscosity") / np.pi * integral)


@dataclass(frozen=True)
class Solution:
    pressure: dict
    edge_flow: dict
    terminal_flow: dict
    inlet_flow: float
    relative_mass_error: float


def solve(edges, root, terminal_loads, inlet_pa, distal_pa=0.0):
    """Linear resistor graph with a Dirichlet inlet and shunts to distal_pa.

    edges: (stable id, u, v, resistance); loads: node -> resistance.
    Distinct edges between the same nodes are supported. Internal loops are legal.
    """
    if not math.isfinite(inlet_pa) or not math.isfinite(distal_pa):
        raise ValueError("Non-finite boundary pressure")
    nodes = {root, *terminal_loads}
    topology = nx.Graph()
    seen = set()
    for key, u, v, resistance in edges:
        if key in seen or u == v:
            raise ValueError("Duplicate edge id or self-loop")
        seen.add(key)
        positive(resistance, f"edge {key} resistance")
        nodes.update((u, v))
        topology.add_edge(u, v)
    topology.add_nodes_from(nodes)
    if not terminal_loads or not nx.is_connected(topology):
        raise ValueError("Network must be connected and have terminal loads")
    for resistance in terminal_loads.values():
        positive(resistance, "terminal resistance")
    unknown = sorted(nodes - {root}, key=str)
    index = {node: i for i, node in enumerate(unknown)}
    rows, cols, vals = [], [], []
    rhs = np.zeros(len(unknown))
    def add(u, v, g):
        if u == root:
            return
        i = index[u]
        rows.append(i); cols.append(i); vals.append(g)
        if v == root:
            rhs[i] += g * (inlet_pa - distal_pa)
        else:
            rows.append(i); cols.append(index[v]); vals.append(-g)
    for _, u, v, resistance in edges:
        add(u, v, 1/resistance)
        add(v, u, 1/resistance)
    for node, resistance in terminal_loads.items():
        if node != root:
            i = index[node]
            rows.append(i); cols.append(i); vals.append(1/resistance)
    matrix = coo_matrix((vals, (rows, cols)), shape=(len(unknown), len(unknown))).tocsr()
    values = spsolve(matrix, rhs) if unknown else np.array([])
    if not np.all(np.isfinite(values)):
        raise ValueError("Singular network")
    relative = {node: float(values[i]) for node, i in index.items()}
    relative[root] = float(inlet_pa-distal_pa)
    p = {node: value+distal_pa for node, value in relative.items()}
    q = {key: (relative[u]-relative[v])/r for key, u, v, r in edges}
    terminal_q = {node: relative[node]/r for node, r in terminal_loads.items()}
    inlet_q = sum(q[key] if u == root else -q[key] if v == root else 0
                  for key, u, v, _ in edges) + terminal_q.get(root, 0.0)
    error = abs(inlet_q-sum(terminal_q.values())) / max(abs(inlet_q), 1e-30)
    imbalance = {node: terminal_q.get(node, 0.) for node in nodes}
    for key, u, v, _ in edges:
        imbalance[u] += q[key]
        imbalance[v] -= q[key]
    error = max(error, max((abs(value)/max(abs(inlet_q), 1e-30)
                           for node, value in imbalance.items() if node != root), default=0.))
    if error > 1e-7:
        raise ValueError(f"Network mass balance failed: {error:g}")
    return Solution(p, q, terminal_q, float(inlet_q), float(error))


def calibrate_scale(evaluate, target_flow):
    """Find positive terminal-load scale; fail rather than clip infeasible targets."""
    target = positive(target_flow, "target flow")
    def residual(log_scale):
        q = float(evaluate(math.exp(log_scale)))
        if not math.isfinite(q):
            raise ValueError("Non-finite calibration flow")
        return q / target - 1
    # Expand from unit scale: extreme arbitrary endpoints destroy precision in
    # almost occluded networks before the physically relevant bracket is found.
    lo = hi = 0.0
    flo = fhi = residual(0.0)
    if abs(flo) < 1e-12:
        return 1.0
    while flo <= 0 and lo > -30:
        lo -= 1
        flo = residual(lo)
    while fhi >= 0 and hi < 30:
        hi += 1
        fhi = residual(hi)
    if flo <= 0:
        raise ValueError("Infeasible target: fixed vessel resistance exceeds the budget")
    if fhi >= 0:
        raise ValueError("Target cannot be bracketed within terminal-load scale bounds")
    return float(math.exp(brentq(residual, lo, hi, xtol=1e-11)))


def structured_tree(root_radius, spec, prefix, viscosity):
    """Explicit binary tree. Daughter ratios and L/r are user-supplied assumptions."""
    radius = positive(root_radius, "structured root radius")
    ratios = list(map(float, spec["daughter_radius_ratios"]))
    if len(ratios) != 2 or any(not 0 < x < 1 for x in ratios):
        raise ValueError("Provide two daughter radius ratios strictly between 0 and 1")
    lr = positive(spec["length_radius_ratio"], "length/radius ratio")
    cutoff = positive(spec["terminal_radius_m"], "terminal radius")
    limit = int(spec.get("max_segments", 100000))
    max_depth = int(spec.get("max_generations", 32))
    edges, leaves = [], {}
    stack = [(prefix, radius, 0)]
    while stack:
        parent, r, depth = stack.pop()
        node = f"{prefix}:st:{len(edges)}"
        length = lr*r
        edges.append((node, parent, node, 8*viscosity*length/(np.pi*r**4)))
        if len(edges) > limit:
            raise ValueError("Structured tree exceeded max_segments; change explicit parameters")
        if r <= cutoff:
            leaves[node] = r
        elif depth >= max_depth:
            raise ValueError("Structured tree reached max_generations before terminal radius")
        else:
            stack.extend((node, r*f, depth+1) for f in ratios)
    return edges, leaves


def partition(tree, viscosity):
    """Partition at explicit clinical cuts and detect shared downstream circuits."""
    profiles = tree["edges"]
    by_id = {str(e["id"]): e for e in profiles}
    if len(by_id) != len(profiles):
        raise ValueError("Duplicate edge identifiers")
    retained_ids = set(map(str, tree["retained_edge_ids"]))
    if not retained_ids <= by_id.keys():
        raise ValueError("Unknown retained edge id")
    weighted = {k: (k, str(e["u"]), str(e["v"]), segment_resistance(e, viscosity))
                for k, e in by_id.items()}
    retained = [weighted[k] for k in sorted(retained_ids)]
    downstream = [edge for k, edge in weighted.items() if k not in retained_ids]
    retained_nodes = {str(tree["root"])} | {n for _, u, v, _ in retained for n in (u, v)}
    outlets = tree["outlets"]
    keys = [str(o["id"]) for o in outlets]
    ports = [str(o["node"]) for o in outlets]
    if len(set(keys)) != len(keys) or len(set(ports)) != len(ports) or not outlets:
        raise ValueError("Clinical outlets require unique ids and nodes")
    if not set(ports) <= retained_nodes:
        raise ValueError("Outlet is not on the retained network")
    clinical = nx.MultiGraph()
    clinical.add_nodes_from(retained_nodes)
    clinical.add_edges_from((u, v) for _, u, v, _ in retained)
    if str(tree["root"]) in ports or any(clinical.degree(n) != 1 for n in ports):
        raise ValueError("Clinical outlet cuts must be separate free ends away from the inlet")
    if any(clinical.degree(n) == 1 and n not in set(ports) | {str(tree["root"])} for n in clinical):
        raise ValueError("Unmapped clinical leaf: open-cut truncation requires an outlet at every cut")
    graph = nx.Graph()
    graph.add_nodes_from(ports)
    for _, u, v, _ in downstream:
        graph.add_edge(u, v)
    beds = {}
    terminal = {str(t["node"]): positive(t["radius_m"], "terminal radius")
                for t in tree["terminals"]}
    if len(terminal) != len(tree["terminals"]):
        raise ValueError("Duplicate terminal node")
    claimed = set()
    for component in nx.connected_components(graph):
        cuts = component & set(ports)
        if len(cuts) != 1 or (component & retained_nodes) != cuts:
            raise ValueError("Shared/unmapped downstream network: independent RCR loads would lose coupling")
        root = next(iter(cuts))
        loads = {n: r for n, r in terminal.items() if n in component}
        if not loads:
            raise ValueError(f"Downstream circuit at {root} has no terminal load")
        bed_edges = [e for e in downstream if e[1] in component]
        for node in component - cuts:
            if graph.degree(node) == 1 and node not in loads:
                raise ValueError(f"Unloaded downstream leaf {node}")
        beds[root] = (bed_edges, loads)
        claimed.update(loads)
    if claimed != set(terminal):
        raise ValueError("Terminal is outside mapped downstream beds")
    return retained, beds
