"""Centreline reconnection: gap filling and graph-connectivity passes.

Groups every operation that reconnects or contracts the spatial graph so the
behaviour can be imported and stepped through on its own. Two categories:

Disconnections *within* a segment (point-less spans in a single centerline):
- ``bridge_centerline_gaps`` -- fill a large empty step with a C1 cubic Hermite
  curve, inserting fresh interior points.

Reconnection / contraction *between* segments (graph topology):
- ``node_id_canon_map`` -- weld near-coincident node ids (KDTree union-find).
- ``merge_degree2_segments`` -- contract degree-2 pass-through nodes.
- ``merge_split_multifurcations`` -- collapse bif-bif stubs that represent one
  anatomical N-furcation artificially split into two degree-3 nodes.
- ``find_connected_components`` / ``split_by_graph`` -- union-find over node
  connectivity and per-component splitting.

Depends only on ``config`` + numpy/scipy (no other package modules), so it is
free of import cycles. ``topology`` and ``smoothing`` re-export these names for
backward compatibility.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.spatial import KDTree

from .config import runtime_config as config


# ── Connected components ──────────────────────────────────────────────────────


def find_connected_components(segments: list[dict[str, Any]]) -> list[set[int]]:
    """Union-find over segment node connectivity.

    Returns a list of sets (one per component), each containing segment
    indices into ``segments``. Components are sorted largest-first.
    """
    parent: dict[int, int] = {}

    def _find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def _union(a: int, b: int) -> None:
        ra, rb = _find(a), _find(b)
        if ra != rb:
            parent[ra] = rb

    all_ids: set[int] = set()
    for s in segments:
        all_ids.add(s["node1"])
        all_ids.add(s["node2"])
    for nid in all_ids:
        parent[nid] = nid
    for s in segments:
        _union(s["node1"], s["node2"])

    comp_map: dict[int, list[int]] = {}
    for i, s in enumerate(segments):
        root = _find(s["node1"])
        comp_map.setdefault(root, []).append(i)

    return [set(c) for c in sorted(comp_map.values(), key=len, reverse=True)]


def split_by_graph(
    nodes: dict[int, tuple],
    points: dict[int, tuple],
    segments: list[dict[str, Any]],
) -> dict[int, tuple[dict, dict, list]]:
    """Split into per-graph subsets using topological connectivity."""
    if not segments:
        return {0: (nodes, points, segments)}

    components = find_connected_components(segments)
    if len(components) <= 1:
        return {0: (nodes, points, segments)}

    print(
        f"  [GRAPH] Topology: {len(components)} connected component(s), "
        f"sizes: {[len(c) for c in components]}"
    )

    result: dict[int, tuple[dict, dict, list]] = {}
    for comp_idx, seg_indices in enumerate(components):
        g_segments = [segments[i] for i in seg_indices]
        g_node_ids: set[int] = set()
        for s in g_segments:
            g_node_ids.add(s["node1"])
            g_node_ids.add(s["node2"])
        g_nodes = {nid: data for nid, data in nodes.items() if nid in g_node_ids}
        g_point_ids: set[int] = set()
        for s in g_segments:
            g_point_ids.update(s["point_ids"])
        g_points = {pid: data for pid, data in points.items() if pid in g_point_ids}
        result[comp_idx] = (g_nodes, g_points, g_segments)
    return result


# ── Degree-2 contraction ──────────────────────────────────────────────────────


def merge_degree2_segments(
    nodes: dict[int, tuple],
    points: dict[int, tuple],
    segments: list[dict[str, Any]],
) -> tuple[dict[int, tuple], list[dict[str, Any]], int]:
    """Contract degree-2 pass-through nodes; return ``(nodes, segments, n_merged)``.

    Smooths the radius across the join across
    ``config.RADIUS_BLEND_POINTS`` to avoid sub-mm steps at the merge.
    """
    if not segments:
        return dict(nodes), list(segments), 0

    segs = [dict(s, point_ids=list(s["point_ids"])) for s in segments]
    n_orig = len(segs)

    node_to_segs: dict[int, list[int]] = {}
    for i, s in enumerate(segs):
        for nid in (s["node1"], s["node2"]):
            node_to_segs.setdefault(nid, []).append(i)

    active = [True] * n_orig
    deg2_nodes = [nid for nid, lst in node_to_segs.items() if len(lst) == 2]
    n_merged = 0

    for nid in deg2_nodes:
        live = [i for i in node_to_segs.get(nid, []) if active[i]]
        if len(live) != 2:
            continue
        a_idx, b_idx = live
        if a_idx == b_idx:
            continue

        sa = segs[a_idx]
        sb = segs[b_idx]
        sa_at_n1 = sa["node1"] == nid
        sb_at_n1 = sb["node1"] == nid

        pa = list(sa["point_ids"])
        pb = list(sb["point_ids"])
        if sa_at_n1:
            pa = pa[::-1]
        if not sb_at_n1:
            pb = pb[::-1]

        drop_first_of_b = False
        if pa and pb:
            if pa[-1] == pb[0]:
                drop_first_of_b = True
            elif pa[-1] in points and pb[0] in points:
                ca = points[pa[-1]]
                cb = points[pb[0]]
                d2 = (ca[0] - cb[0]) ** 2 + (ca[1] - cb[1]) ** 2 + (ca[2] - cb[2]) ** 2
                if d2 < 1.0:
                    drop_first_of_b = True
        merged_pids = pa + (pb[1:] if drop_first_of_b else pb)

        # Linear-blend radii across the join.
        join_local_idx = len(pa)
        blend_n = max(1, config.RADIUS_BLEND_POINTS)
        i_lo = max(0, join_local_idx - blend_n)
        i_hi = min(len(merged_pids) - 1, join_local_idx + blend_n)
        if i_hi > i_lo + 1:
            r_lo = float(points[merged_pids[i_lo]][3])
            r_hi = float(points[merged_pids[i_hi]][3])
            for k in range(i_lo + 1, i_hi):
                t = (k - i_lo) / (i_hi - i_lo)
                blended_r = r_lo * (1.0 - t) + r_hi * t
                pid_k = merged_pids[k]
                if pid_k in points:
                    rec = list(points[pid_k])
                    rec[3] = blended_r
                    points[pid_k] = tuple(rec) if isinstance(points[pid_k], tuple) else rec

        new_node1 = sa["node2"] if sa_at_n1 else sa["node1"]
        new_node2 = sb["node2"] if sb_at_n1 else sb["node1"]

        sa["point_ids"] = merged_pids
        sa["node1"] = new_node1
        sa["node2"] = new_node2
        if "strahler" in sa or "strahler" in sb:
            sa["strahler"] = min(sa.get("strahler", 0), sb.get("strahler", 0))

        active[b_idx] = False
        n_merged += 1
        node_to_segs[new_node2] = [
            a_idx if x == b_idx else x for x in node_to_segs.get(new_node2, [])
        ]
        node_to_segs.pop(nid, None)

    new_segments = [segs[i] for i in range(n_orig) if active[i]]

    deg: dict[int, int] = {}
    for s in new_segments:
        deg[s["node1"]] = deg.get(s["node1"], 0) + 1
        deg[s["node2"]] = deg.get(s["node2"], 0) + 1
    new_nodes: dict[int, tuple] = {}
    for nid, ndata in nodes.items():
        d = deg.get(nid, 0)
        if d == 0:
            continue
        x, y, z = ndata[0], ndata[1], ndata[2]
        new_nodes[nid] = (x, y, z, d)
    return new_nodes, new_segments, n_merged


# ── Split-multifurcation contraction ──────────────────────────────────────────


def _endpoint_tangent(
    seg: dict[str, Any], nid: int, points: dict[int, tuple], lookahead: int = 3
) -> np.ndarray:
    """Unit tangent at the given endpoint of ``seg``, pointing outward from
    ``nid``. Zero vector if the segment is too short or degenerate."""
    pids = seg["point_ids"]
    n_pts = len(pids)
    if n_pts < 2:
        return np.zeros(3)
    if nid == seg["node1"]:
        i_near = 0
        i_far = min(lookahead, n_pts - 1)
    elif nid == seg["node2"]:
        i_near = n_pts - 1
        i_far = max(0, n_pts - 1 - lookahead)
    else:
        return np.zeros(3)
    pn = points.get(pids[i_near])
    pf = points.get(pids[i_far])
    if pn is None or pf is None:
        return np.zeros(3)
    tan = np.array([pf[0] - pn[0], pf[1] - pn[1], pf[2] - pn[2]], dtype=np.float64)
    nrm = float(np.linalg.norm(tan))
    if nrm < 1e-9:
        return np.zeros(3)
    return tan / nrm


def _endpoint_radius_mm(
    seg: dict[str, Any], nid: int, points: dict[int, tuple]
) -> float:
    """Radius (mm) at the given endpoint of ``seg``."""
    pids = seg["point_ids"]
    if not pids:
        return 0.0
    if nid == seg["node1"]:
        pid = pids[0]
    elif nid == seg["node2"]:
        pid = pids[-1]
    else:
        return 0.0
    pt = points.get(pid)
    if pt is None or len(pt) < 4:
        return 0.0
    return float(pt[3]) / 1000.0 * config.RADIUS_SCALE


def _segment_arc_length_mm(
    seg: dict[str, Any], points: dict[int, tuple]
) -> float:
    pids = seg["point_ids"]
    if len(pids) < 2:
        return 0.0
    total = 0.0
    p0 = points.get(pids[0])
    if p0 is None:
        return 0.0
    px, py, pz = p0[0], p0[1], p0[2]
    for pid in pids[1:]:
        pt = points.get(pid)
        if pt is None:
            continue
        x, y, z = pt[0], pt[1], pt[2]
        dx, dy, dz = x - px, y - py, z - pz
        total += (dx * dx + dy * dy + dz * dz) ** 0.5
        px, py, pz = x, y, z
    return total / 1000.0


def merge_split_multifurcations(
    nodes: dict[int, tuple],
    points: dict[int, tuple],
    segments: list[dict[str, Any]],
    max_len_factor: float,
    require_strahler: bool = True,
    tangent_cos_min: float = 0.7,
) -> tuple[dict[int, tuple], list[dict[str, Any]], int, list[dict]]:
    """Contract bif-bif stubs that represent a single anatomical N-furcation
    artificially split into two degree-3 nodes by the centerline extractor.

    Criteria (all must hold for contraction):

    1. Both endpoints are degree exactly 3.
    2. Strahler signature: connector at trunk Strahler, the proximal node's
       other branch at lower Strahler, and BOTH distal-node daughters at lower
       Strahler. (Real adjacent bifs would have one distal daughter still at
       trunk Strahler — the trunk continues past D.)
    3. Connector arc length < ``max_len_factor`` * min(endpoint radius, mm).
    4. Connector tangent collinear with the same-Strahler trunk-continuation
       at the proximal node (``|cos| >= tangent_cos_min``).

    Returns ``(nodes, segments, n_collapsed, records)``. ``records`` lists
    per-candidate diagnostics: decision, segment id, lengths, radii, Strahler
    fields, tangent cosine — for verbose inspection.
    """
    if not segments:
        return dict(nodes), list(segments), 0, []

    segs = [dict(s, point_ids=list(s["point_ids"])) for s in segments]
    nodes_out = dict(nodes)
    records: list[dict] = []
    n_collapsed_total = 0

    max_iters = 10
    for _iteration in range(max_iters):
        node_to_segs: dict[int, list[int]] = {}
        for i, s in enumerate(segs):
            for nid in (s["node1"], s["node2"]):
                node_to_segs.setdefault(nid, []).append(i)

        collapse_ids: list[int] = []
        for ci, s in enumerate(segs):
            n1, n2 = s["node1"], s["node2"]
            if n1 == n2:
                continue
            deg_n1 = len(node_to_segs.get(n1, []))
            deg_n2 = len(node_to_segs.get(n2, []))
            if deg_n1 != 3 or deg_n2 != 3:
                continue

            arc_len = _segment_arc_length_mm(s, points)
            r_n1 = _endpoint_radius_mm(s, n1, points)
            r_n2 = _endpoint_radius_mm(s, n2, points)
            r_min = min(r_n1, r_n2)
            if r_min <= 0.0:
                continue
            if arc_len >= max_len_factor * r_min:
                records.append({
                    "seg_id": s.get("id"),
                    "action": "reject_length",
                    "arc_len_mm": arc_len,
                    "r_min_mm": r_min,
                    "threshold_mm": max_len_factor * r_min,
                })
                continue

            others_n1 = [j for j in node_to_segs[n1] if j != ci]
            others_n2 = [j for j in node_to_segs[n2] if j != ci]
            if len(others_n1) != 2 or len(others_n2) != 2:
                continue

            s_con = s.get("strahler")
            if require_strahler:
                if s_con is None:
                    records.append({
                        "seg_id": s.get("id"),
                        "action": "reject_no_strahler",
                    })
                    continue
                strahlers_n1 = sorted(
                    [(j, segs[j].get("strahler", 0)) for j in others_n1],
                    key=lambda x: x[1],
                    reverse=True,
                )
                parent_n1_idx, s_pp = strahlers_n1[0]
                _, s_pc = strahlers_n1[1]
                if s_con != s_pp:
                    records.append({
                        "seg_id": s.get("id"),
                        "action": "reject_strahler_not_continuator",
                        "s_con": s_con, "s_pp": s_pp,
                    })
                    continue
                if s_pc > s_pp - 1:
                    records.append({
                        "seg_id": s.get("id"),
                        "action": "reject_strahler_proximal_real_bif",
                        "s_pp": s_pp, "s_pc": s_pc,
                    })
                    continue
                max_dd = max(segs[j].get("strahler", 0) for j in others_n2)
                if max_dd > s_pp - 1:
                    records.append({
                        "seg_id": s.get("id"),
                        "action": "reject_strahler_distal_real_bif",
                        "s_pp": s_pp, "max_dd": max_dd,
                    })
                    continue
            else:
                parent_n1_idx = max(
                    others_n1,
                    key=lambda j: _endpoint_radius_mm(segs[j], n1, points),
                )
                s_pp = segs[parent_n1_idx].get("strahler", 0)
                s_pc = 0

            t_con = _endpoint_tangent(s, n1, points)
            t_par = _endpoint_tangent(segs[parent_n1_idx], n1, points)
            if float(np.linalg.norm(t_con)) < 1e-9 or float(np.linalg.norm(t_par)) < 1e-9:
                records.append({
                    "seg_id": s.get("id"),
                    "action": "reject_tangent_degenerate",
                })
                continue
            cos_n1 = float(abs(np.dot(t_con, t_par)))
            if cos_n1 < tangent_cos_min:
                records.append({
                    "seg_id": s.get("id"),
                    "action": "reject_tangent_misaligned",
                    "cos_n1": cos_n1,
                })
                continue

            collapse_ids.append(ci)
            records.append({
                "seg_id": s.get("id"),
                "action": "collapse",
                "arc_len_mm": arc_len,
                "r_min_mm": r_min,
                "cos_n1": cos_n1,
                "s_con": s_con,
                "s_pp": s_pp,
                "s_pc": s_pc,
                "node1": n1, "node2": n2,
            })

        if not collapse_ids:
            break

        uf: dict[int, int] = {}

        def _find(x: int) -> int:
            while uf.get(x, x) != x:
                uf[x] = uf.get(uf[x], uf[x])
                x = uf[x]
            return x

        def _union(a: int, b: int) -> None:
            ra, rb = _find(a), _find(b)
            if ra != rb:
                uf[ra] = rb

        for ci in collapse_ids:
            _union(segs[ci]["node1"], segs[ci]["node2"])

        for s in segs:
            s["node1"] = _find(s["node1"])
            s["node2"] = _find(s["node2"])

        kept: list[dict[str, Any]] = []
        for ci, s in enumerate(segs):
            if ci in set(collapse_ids):
                continue
            kept.append(s)
        segs = kept
        n_collapsed_total += len(collapse_ids)

    referenced_nodes: set[int] = set()
    deg: dict[int, int] = {}
    for s in segs:
        referenced_nodes.add(s["node1"])
        referenced_nodes.add(s["node2"])
        deg[s["node1"]] = deg.get(s["node1"], 0) + 1
        deg[s["node2"]] = deg.get(s["node2"], 0) + 1
    new_nodes: dict[int, tuple] = {}
    for nid in referenced_nodes:
        nd = nodes_out.get(nid)
        if nd is None:
            continue
        new_nodes[nid] = (nd[0], nd[1], nd[2], deg.get(nid, 0))

    return new_nodes, segs, n_collapsed_total, records


# ── Near-coincident node canonicalisation ─────────────────────────────────────


def node_id_canon_map(
    nodes_dict: dict[int, tuple], eps_mm: float
) -> dict[int, int]:
    """Map each node id to a canonical id (merging within ``eps_mm`` of each other).

    Operates in mm by dividing the stored micrometers by 1000.
    """
    if not nodes_dict or eps_mm <= 0:
        return {}
    ids = list(nodes_dict.keys())
    pos = np.array(
        [
            [nodes_dict[i][0] / 1000.0, nodes_dict[i][1] / 1000.0, nodes_dict[i][2] / 1000.0]
            for i in ids
        ],
        dtype=np.float64,
    )
    if len(ids) < 2:
        return {nid: nid for nid in ids}
    parent = {nid: nid for nid in ids}

    def _find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def _union(a, b):
        ra, rb = _find(a), _find(b)
        if ra != rb:
            parent[ra] = rb

    kd = KDTree(pos)
    try:
        pairs = kd.query_pairs(float(eps_mm))
    except Exception:
        pairs = set()
    for i, j in pairs:
        _union(ids[i], ids[j])
    return {nid: _find(nid) for nid in ids}


# ── Within-segment gap bridging ───────────────────────────────────────────────


def bridge_centerline_gaps(
    points: dict[int, tuple],
    segments: list[dict[str, Any]],
    *,
    target_spacing_mm: float,
    big_jump_ratio: float,
    min_gap_um: float,
    radius_scale: float = 1.0,
    verbose: bool = False,
) -> tuple[dict[int, tuple], int, list[dict[str, Any]]]:
    """Fill point-less centerline spans with curvature-following interpolation.

    Some ``.am`` exports store an edge as two point clusters separated by a large
    jump with no intermediate points (Avizo hides this by drawing the edge as a
    straight line). The pipeline samples cross-sections only at stored points, so
    such a span produces an empty, fragmented tube. This pass detects each span
    and bridges it with a **C1 cubic Hermite** curve that leaves each side along
    its local tangent -- a curved (not straight) fill -- inserting fresh interior
    points at ``target_spacing_mm``.

    A step between consecutive present points ``i, i+1`` is a gap when both
    ``step_um > min_gap_um`` and ``step_um > big_jump_ratio * (r_i + r_j) * 1000``
    (the BIG-JUMP criterion from ``report_ring_gap_attribution``), so thin-vessel
    sample spacing is left alone. Radius is linearly interpolated across the span.

    Mirrors :func:`densify_sparse_segments`: allocates fresh point ids past
    ``max(points)`` and mutates ``seg["point_ids"]`` in place, preserving every
    original point (so node anchoring is unaffected -- only interior points are
    added). Returns ``(new_points, n_gaps_bridged, records)`` where each record is
    ``{seg_id, i, gap_mm, n_inserted}``. Pure aside from the in-place point_ids.
    """
    from scipy.interpolate import CubicHermiteSpline

    if not segments:
        return dict(points), 0, []

    pts_out: dict[int, tuple] = dict(points)
    next_pid = (max(pts_out.keys()) + 1) if pts_out else 0
    n_bridged = 0
    records: list[dict[str, Any]] = []

    if verbose:
        print("[BRIDGE] seg_id     i   gap_mm  inserted")

    for seg in segments:
        sid = seg.get("id", "?")
        pids = [p for p in seg.get("point_ids", []) if p in pts_out]
        if len(pids) < 2:
            continue
        coords = np.array(
            [(pts_out[p][0], pts_out[p][1], pts_out[p][2]) for p in pids],
            dtype=np.float64,
        )  # micrometers
        radii_um = np.array([float(pts_out[p][3]) for p in pids], dtype=np.float64)
        steps = np.linalg.norm(np.diff(coords, axis=0), axis=1)
        n_p = len(coords)

        out_pids: list[int] = [pids[0]]
        seg_bridged = 0
        for i in range(n_p - 1):
            step = float(steps[i])
            sum_r_um = (radii_um[i] + radii_um[i + 1]) * radius_scale
            is_gap = step > min_gap_um and step > big_jump_ratio * sum_r_um
            if is_gap:
                pa, pb = coords[i], coords[i + 1]
                ta = coords[i] - coords[i - 1] if i - 1 >= 0 else pb - pa
                tb = coords[i + 2] - coords[i + 1] if i + 2 <= n_p - 1 else pb - pa
                na, nb = np.linalg.norm(ta), np.linalg.norm(tb)
                chord = pb - pa
                ta = ta / na if na > 1e-12 else chord / max(step, 1e-12)
                tb = tb / nb if nb > 1e-12 else chord / max(step, 1e-12)
                n_interior = max(1, int(np.ceil((step / 1000.0) / max(target_spacing_mm, 1e-9))) - 1)
                t = np.linspace(0.0, 1.0, n_interior + 2)[1:-1]
                new_xyz = np.empty((n_interior, 3), dtype=np.float64)
                for ax in range(3):
                    hs = CubicHermiteSpline(
                        [0.0, 1.0], [pa[ax], pb[ax]], [ta[ax] * step, tb[ax] * step]
                    )
                    new_xyz[:, ax] = hs(t)
                new_r = np.interp(t, [0.0, 1.0], [radii_um[i], radii_um[i + 1]])
                for k in range(n_interior):
                    pts_out[next_pid] = (
                        float(new_xyz[k, 0]),
                        float(new_xyz[k, 1]),
                        float(new_xyz[k, 2]),
                        float(new_r[k]),
                    )
                    out_pids.append(next_pid)
                    next_pid += 1
                seg_bridged += 1
                records.append(
                    dict(seg_id=sid, i=i, gap_mm=step / 1000.0, n_inserted=n_interior)
                )
                if verbose:
                    print(f"[BRIDGE] {sid:>6}  {i:>4}  {step / 1000.0:>7.2f}  {n_interior:>8}")
            out_pids.append(pids[i + 1])

        if seg_bridged:
            seg["point_ids"] = out_pids
            n_bridged += seg_bridged

    return pts_out, n_bridged, records


def split_unsampled_jumps(
    nodes: dict[int, tuple],
    points: dict[int, tuple],
    segments: list[dict[str, Any]],
    *,
    big_jump_ratio: float,
    min_gap_um: float,
    step_ratio: float = 0.0,
    radius_scale: float = 1.0,
    verbose: bool = False,
) -> tuple[dict[int, tuple], list[dict[str, Any]], int, list[dict[str, Any]]]:
    """Cut each segment at every point-less jump instead of bridging it.

    Uses the same test as :func:`bridge_centerline_gaps` -- so the spans this cuts
    are the spans that function would have filled -- and then one more. Only the
    action differs, and which action is right is a question about the image, not
    about the graph: on LADAF-2024-28 the segmentation is in 55 disconnected
    components and the graph's 24 jumps (72.4 mm in total) each cross between two
    of them. Bridging them draws lumen through 72.4 mm of proven background;
    cutting them lets the surface end where the evidence ends.

    Two gates, and a step needs the absolute floor plus *either* of the others:

    * ``big_jump_ratio * (r_i + r_j)`` -- the jump is many vessel-widths wide;
    * ``step_ratio`` times the segment's **own median spacing** -- the jump is an
      outlier against how that segment is actually sampled.

    The second exists because the first is radius-scaled, and a break in a thin
    vessel is a small absolute distance. Measured here: the radius gate alone finds
    18 of the 24 jumps `hipct_seg_debug`'s `flag-interpolation --seg` confirms
    against the mask, missing six of 0.6-1.2 mm in vessels 140-230 um across --
    every one a genuine break, just a narrow one. Set ``step_ratio=0`` for the
    radius gate alone.

    A cut promotes the two points either side of the jump to nodes, so the halves
    become topologically separate and :func:`split_by_graph` sees them as the
    distinct components they are. Nothing is deleted here -- dropping the fragments
    is a separate decision, taken once the component sizes are known.

    Returns ``(new_nodes, new_segments, n_split, records)``; ``points`` is not
    modified, since a cut adds no geometry.
    """
    if not segments:
        return dict(nodes), list(segments), 0, []

    nodes_out = dict(nodes)
    next_nid = (max(nodes_out.keys()) + 1) if nodes_out else 0
    out_segments: list[dict[str, Any]] = []
    records: list[dict[str, Any]] = []
    n_split = 0

    for seg in segments:
        pids = [p for p in seg.get("point_ids", []) if p in points]
        if len(pids) < 2:
            out_segments.append(seg)
            continue
        coords = np.array([points[p][:3] for p in pids], dtype=np.float64)
        radii_um = np.array([float(points[p][3]) for p in pids], dtype=np.float64)
        steps = np.linalg.norm(np.diff(coords, axis=0), axis=1)

        median_step = float(np.median(steps)) if len(steps) else 0.0
        cuts = [
            i for i in range(len(steps))
            if steps[i] > min_gap_um
            and (
                steps[i] > big_jump_ratio * (radii_um[i] + radii_um[i + 1]) * radius_scale
                or (step_ratio > 0.0 and steps[i] > step_ratio * median_step)
            )
        ]
        if not cuts:
            out_segments.append(seg)
            continue

        # Point index ranges of the pieces: cut i separates [.., i] from [i+1, ..].
        bounds = [0] + [i + 1 for i in cuts] + [len(pids)]
        pieces = [pids[bounds[k]:bounds[k + 1]] for k in range(len(bounds) - 1)]
        first_node, last_node = seg["node1"], seg["node2"]

        for k, piece in enumerate(pieces):
            if len(piece) < 2:
                # A jump landing one point from the end leaves a single orphan
                # point, which is not a polyline. Dropping it loses nothing: the
                # piece carries no length, and keeping it would make a node that
                # anchors nothing.
                continue
            if k == 0:
                n1 = first_node
            else:
                x, y, z = points[piece[0]][:3]
                nodes_out[next_nid] = (float(x), float(y), float(z), 0)
                n1 = next_nid
                next_nid += 1
            if k == len(pieces) - 1:
                n2 = last_node
            else:
                x, y, z = points[piece[-1]][:3]
                nodes_out[next_nid] = (float(x), float(y), float(z), 0)
                n2 = next_nid
                next_nid += 1
            child = dict(seg)
            child["node1"], child["node2"] = n1, n2
            child["point_ids"] = list(piece)
            if k > 0:
                # Ids must stay unique; the original keeps its id so anything
                # already recorded against it still resolves.
                child["id"] = f"{seg.get('id', '?')}:{k}"
            out_segments.append(child)

        n_split += len(cuts)
        for i in cuts:
            records.append(dict(seg_id=seg.get("id", "?"), i=i,
                                gap_mm=float(steps[i]) / 1000.0))
            if verbose:
                print(f"[SPLIT] {seg.get('id', '?'):>8}  at point {i:>5}  "
                      f"{steps[i] / 1000.0:>7.2f} mm")

    return nodes_out, out_segments, n_split, records


def drop_small_components(
    nodes: dict[int, tuple],
    points: dict[int, tuple],
    segments: list[dict[str, Any]],
    *,
    min_length_mm: float = 0.0,
    min_fraction_of_largest: float = 0.0,
    verbose: bool = True,
) -> tuple[dict[int, tuple], list[dict[str, Any]], list[dict[str, Any]]]:
    """Discard connected components too small to be a tree in their own right.

    Cutting at the jumps leaves the fragments those jumps were holding on by. A
    fragment is not evidence of a vessel -- it is the part of one whose connection
    to the tree was never imaged -- and surfacing it produces a free-floating tube
    that no inlet feeds, which is worse than useless for CFD.

    Length, not segment count, is the measure: one long segment matters and twenty
    short ones need not. A component is kept when it clears **both** floors, and
    `min_fraction_of_largest` is the one that does the work, because it is the only
    one that transfers between hearts. Measured on LADAF-2024-28 after cutting: the
    two trees are 1248 mm and 973 mm (100% and 78% of the largest) and the biggest
    stranded fragment is 127 mm (10%), so any fraction in the wide empty band
    between them separates them -- the default sits in the middle of it.

    Returns ``(nodes, segments, dropped)``; each dropped record is
    ``{n_segments, length_mm}``.
    """
    if not segments:
        return dict(nodes), list(segments), []

    components = find_connected_components(segments)
    lengths: list[float] = []
    for comp in components:
        total_um = 0.0
        for i in comp:
            pids = [p for p in segments[i].get("point_ids", []) if p in points]
            if len(pids) < 2:
                continue
            c = np.array([points[p][:3] for p in pids], dtype=np.float64)
            total_um += float(np.linalg.norm(np.diff(c, axis=0), axis=1).sum())
        lengths.append(total_um / 1000.0)

    largest = max(lengths) if lengths else 0.0
    floor = max(float(min_length_mm), float(min_fraction_of_largest) * largest)

    keep_idx: set[int] = set()
    dropped: list[dict[str, Any]] = []
    for comp, length_mm in zip(components, lengths):
        if length_mm >= floor:
            keep_idx |= comp
        else:
            dropped.append(dict(n_segments=len(comp), length_mm=length_mm))

    if not dropped:
        return dict(nodes), list(segments), []

    kept = [s for i, s in enumerate(segments) if i in keep_idx]
    live_nodes = {s["node1"] for s in kept} | {s["node2"] for s in kept}
    nodes_out = {nid: d for nid, d in nodes.items() if nid in live_nodes}
    if verbose:
        total = sum(d["length_mm"] for d in dropped)
        print(f"  Dropped {len(dropped)} disconnected component(s), {total:.1f} mm "
              f"below the {floor:.1f} mm floor "
              f"({len(segments)} -> {len(kept)} segments, "
              f"{len(components) - len(dropped)} component(s) kept)")
    return nodes_out, kept, dropped


__all__ = [
    "find_connected_components",
    "split_by_graph",
    "merge_degree2_segments",
    "merge_split_multifurcations",
    "node_id_canon_map",
    "bridge_centerline_gaps",
    "split_unsampled_jumps",
    "drop_small_components",
]
