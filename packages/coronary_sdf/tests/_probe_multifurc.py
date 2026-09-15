"""Diagnostic probe: locate the trunk radius/centerline dip at multifurcations.

Reproduces the run_pipeline preprocessing (parse -> strahler -> degree-2 merge ->
split-multifurc merge -> nub prune -> densify -> split_by_graph) and then the per-graph
smoothing chain (centerline -> segment-radii -> bif-shrink -> terminal-shrink ->
radius-transition), snapshotting `points` after each stage. For every degree>=4 node it
prints the node-ward radius profile of each incident segment at each stage, so we can see
whether (and at which stage) the through-trunk radius necks at the node.

Read-only w.r.t. the repo; writes nothing. Run: python _probe_multifurc.py
"""
from __future__ import annotations

import copy
import os
import sys

import numpy as np

# Ensure the package parent (…\) is importable when run as a plain script.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from coronary_sdf import config

# Quiet the per-junction verbose chatter from the smoothing stages.
config.RADIUS_TRANSITION_VERBOSE = False
config.SMOOTH_DRIFT_VERBOSE = False
config.DENSIFY_VERBOSE = False

from coronary_sdf.parse_amira import parse_xml
from coronary_sdf.topology import (
    merge_degree2_segments,
    merge_split_multifurcations,
    split_by_graph,
)
from coronary_sdf.pruning import prune_short_terminal_nubs
from coronary_sdf.smoothing import (
    densify_sparse_segments,
    smooth_segment_centerlines,
    smooth_segment_radii,
    smooth_radius_transitions,
    prune_terminal_shrink,
    prune_bifurcation_shrink,
)
from coronary_sdf.splines import prepare_segment_spline
from coronary_sdf.capsules import build_capsules

LOOK = int(config.RADIUS_ENDPOINT_LOOKAHEAD)
NPROF = 8           # node-ward points to print per segment
TOP_NODES = 10      # worst (thickest) deg>=4 nodes to show


def node_ward_pids(seg, nid):
    """Ordered pids from the node inward (index 0 == at the node)."""
    pids = seg["point_ids"]
    if seg["node1"] == nid:
        return list(pids)
    if seg["node2"] == nid:
        return list(reversed(pids))
    # canonicalised / merged node: pick the geometrically nearer end
    return list(pids)


def r_at(points, pid):
    return float(points[pid][3])


def coord_mm(points, pid):
    p = points[pid]
    return np.array([p[0], p[1], p[2]], dtype=float) / 1000.0


def seg_caps_near(points, segments, center_mm, reach=6.0):
    """All capsules (consecutive point pairs) whose midpoint is within `reach`
    mm of center_mm. Returns starts,ends,(r0,r1) in mm."""
    S, E, R0, R1 = [], [], [], []
    for seg in segments:
        pids = seg["point_ids"]
        for a, b in zip(pids[:-1], pids[1:]):
            ca, cb = coord_mm(points, a), coord_mm(points, b)
            if np.linalg.norm(0.5 * (ca + cb) - center_mm) <= reach:
                S.append(ca); E.append(cb)
                R0.append(points[a][3] / 1000.0 * config.RADIUS_SCALE)
                R1.append(points[b][3] / 1000.0 * config.RADIUS_SCALE)
    return (np.asarray(S), np.asarray(E), np.asarray(R0), np.asarray(R1))


def union_sdf(P, caps):
    """Pure capsule-union SDF at points P (M,3). No smin/flat-cap/carve."""
    S, E, R0, R1 = caps
    if len(S) == 0:
        return np.full(len(P), np.inf)
    d = E - S
    dd = np.sum(d * d, axis=1)
    PS = P[:, None, :] - S[None, :, :]
    t = np.clip(np.einsum("mki,ki->mk", PS, d) / np.maximum(dd, 1e-12), 0.0, 1.0)
    closest = S[None, :, :] + t[..., None] * d[None, :, :]
    dist = np.linalg.norm(P[:, None, :] - closest, axis=2)
    rad = R0[None, :] + t * (R1[None, :] - R0[None, :])
    return (dist - rad).min(axis=1)


def node_tangent_out(points, seg, nid, npts=4):
    pids = node_ward_pids(seg, nid)
    if len(pids) < 2:
        return None
    a = coord_mm(points, pids[0])
    b = coord_mm(points, pids[min(npts, len(pids) - 1)])
    v = b - a
    n = np.linalg.norm(v)
    return v / n if n > 1e-9 else None


def _tangents(points, segments, segs, nid):
    out = []
    for si in segs:
        t = node_tangent_out(points, segments[si], nid)
        out.append(t)
    return out


def _trunk_pair(tans):
    best = (-2.0, 0, 1)
    for i in range(len(tans)):
        for j in range(i + 1, len(tans)):
            if tans[i] is None or tans[j] is None:
                continue
            d = float(tans[i] @ tans[j])
            if -d > best[0]:
                best = (-d, i, j)
    return best[1], best[2]


def _kink(tans, i, j):
    if tans[i] is None or tans[j] is None:
        return float("nan")
    return float(np.degrees(np.arccos(np.clip(tans[i] @ tans[j], -1, 1))))  # 180 = straight


def _trunk_polyline(points, segments, segs, nid, i, j, npts=12):
    """Ordered (coord_mm, radius_mm) list through the node: far_i .. node .. far_j."""
    a = node_ward_pids(segments[segs[i]], nid)[:npts]   # [node, inward_i...]
    b = node_ward_pids(segments[segs[j]], nid)[:npts]   # [node, inward_j...]
    chain = list(reversed(a)) + b[1:]                   # far_i .. node .. far_j
    pts = [coord_mm(points, p) for p in chain]
    rad = [points[p][3] / 1000.0 * config.RADIUS_SCALE for p in chain]
    node_k = len(a) - 1                                  # index of the node in chain
    return np.array(pts), np.array(rad), node_k


def analyze_geometry(points, raw_points, nodes, segments, n2s, nid, label):
    segs = n2s[nid]
    tans_s = _tangents(points, segments, segs, nid)
    tans_r = _tangents(raw_points, segments, segs, nid)
    if sum(t is not None for t in tans_s) < 2:
        return
    i, j = _trunk_pair(tans_s)
    kink_s = _kink(tans_s, i, j)
    kink_r = _kink(tans_r, i, j)
    sids = [segments[si].get("id") for si in segs]

    poly, rad, k0 = _trunk_polyline(points, segments, segs, nid, i, j)
    R = float(np.median(rad))
    caps = seg_caps_near(points, segments, poly[k0], reach=max(6.0, 4 * R))
    th = np.linspace(0, 2 * np.pi, 24, endpoint=False)

    # Curvature-robust dish: at each centerline station, ring of local radius in
    # the plane perpendicular to the LOCAL tangent; max union SDF over angles.
    dish = []
    for k in range(1, len(poly) - 1):
        tan = poly[k + 1] - poly[k - 1]
        n = np.linalg.norm(tan)
        if n < 1e-9:
            dish.append(0.0); continue
        tan /= n
        ref = np.array([1.0, 0, 0]) if abs(tan[0]) < 0.9 else np.array([0, 1.0, 0])
        u = np.cross(tan, ref); u /= max(np.linalg.norm(u), 1e-9)
        v = np.cross(tan, u)
        rdir = np.cos(th)[:, None] * u[None, :] + np.sin(th)[:, None] * v[None, :]
        P = poly[k][None, :] + rad[k] * rdir
        dish.append(float((union_sdf(P, caps) * 1000.0).max()))   # um, >0 = inward dish
    dish = np.array(dish)
    node_dish = dish[k0 - 1] if 0 <= k0 - 1 < len(dish) else float("nan")
    print(f"  [GEO] {label} NODE {nid} deg={len(segs)} trunk=seg{sids[i]}/seg{sids[j]} R={R*1000:.0f}um | "
          f"kink raw={kink_r:5.1f} -> smoothed={kink_s:5.1f} deg (180=straight) | "
          f"dish@node={node_dish:4.0f}um peak±2pts={dish[max(0,k0-3):k0+2].max():4.0f}um")


def interior_ref(points, pids):
    rs = [r_at(points, pid) for pid in pids[1:1 + LOOK] if r_at(points, pid) > 0]
    return float(np.median(rs)) if rs else (r_at(points, pids[0]) if pids else 0.0)


def main():
    print(f"[probe] parsing {config.INPUT_XML}")
    nodes, points, segments = parse_xml(config.INPUT_XML)

    if config.MIN_STRAHLER_ORDER > 0:
        segments = [s for s in segments if s.get("strahler", 0) >= config.MIN_STRAHLER_ORDER]
    if config.MERGE_DEGREE2_SEGMENTS:
        nodes, segments, _ = merge_degree2_segments(nodes, points, segments)
    if config.MERGE_SPLIT_MULTIFURCATIONS:
        nodes, segments, _, _ = merge_split_multifurcations(
            nodes, points, segments,
            max_len_factor=config.SPLIT_MULTIFURC_MAX_LEN_FACTOR,
            require_strahler=config.SPLIT_MULTIFURC_REQUIRE_STRAHLER,
            tangent_cos_min=config.SPLIT_MULTIFURC_TANGENT_COS_MIN,
        )
    if config.PRUNE_SHORT_TERMINAL_NUBS and config.MIN_TERMINAL_LENGTH_MM > 0:
        nodes, segments, _ = prune_short_terminal_nubs(
            nodes, points, segments,
            min_length_mm=config.MIN_TERMINAL_LENGTH_MM, max_iters=config.PRUNE_ITER_MAX,
        )
    if config.DENSIFY_SPARSE_SEGMENTS:
        points, _ = densify_sparse_segments(
            points, segments,
            target_spacing_mm=config.DENSIFY_TARGET_SPACING_MM,
            min_points=config.DENSIFY_MIN_POINTS, verbose=False,
        )

    graphs = split_by_graph(nodes, points, segments)
    gid = sorted(graphs)[0]
    g_nodes, g_points, g_segments = graphs[gid]
    print(f"[probe] graph {gid}: {len(g_nodes)} nodes, {len(g_segments)} segments")

    # Per-graph smoothing chain, snapshotting points after each stage.
    stages: dict[str, dict] = {"raw": copy.deepcopy(g_points)}
    p, _ = smooth_segment_centerlines(g_nodes, g_points, g_segments)
    stages["cl"] = copy.deepcopy(p)
    p, _ = smooth_segment_radii(g_nodes, p, g_segments)
    stages["segR"] = copy.deepcopy(p)
    p, _ = prune_bifurcation_shrink(g_nodes, p, g_segments)
    stages["bifPrune"] = copy.deepcopy(p)
    p, _ = prune_terminal_shrink(g_nodes, p, g_segments)
    stages["termPrune"] = copy.deepcopy(p)
    p, _ = smooth_radius_transitions(g_nodes, p, g_segments)
    stages["radTrans"] = copy.deepcopy(p)
    final = p
    stage_order = ["raw", "segR", "bifPrune", "termPrune", "radTrans"]

    # node -> incident segment indices
    n2s: dict[int, list[int]] = {}
    for si, seg in enumerate(g_segments):
        for nid in (seg["node1"], seg["node2"]):
            n2s.setdefault(nid, []).append(si)

    multif = [(nid, segs) for nid, segs in n2s.items() if len(segs) >= 4]
    # rank by thickest incident segment (final radii) so we see the worst trunks first
    def node_thickness(nid, segs):
        rr = []
        for si in segs:
            pids = node_ward_pids(g_segments[si], nid)
            if pids:
                rr.append(interior_ref(final, pids))
        return max(rr) if rr else 0.0
    multif.sort(key=lambda t: node_thickness(*t), reverse=True)

    print(f"\n[probe] {len(multif)} degree>=4 nodes. Showing top {TOP_NODES} by trunk radius.")
    print("        radii are raw thickness units (microns); col0 = AT node, increasing inward.\n")

    for nid, segs in multif[:TOP_NODES]:
        # parent / target_r as smooth_radius_transitions would choose (parent mode)
        info = []
        for si in segs:
            seg = g_segments[si]
            pids = node_ward_pids(seg, nid)
            info.append({
                "si": si, "id": seg.get("id"), "strahler": int(seg.get("strahler", 0)),
                "pids": pids, "iref_raw": interior_ref(stages["raw"], pids),
                "iref_fin": interior_ref(final, pids),
            })
        parent = max(range(len(info)), key=lambda i: (info[i]["strahler"], info[i]["iref_raw"]))
        target_r = info[parent]["iref_raw"]

        print("=" * 96)
        print(f"NODE {nid}  degree={len(segs)}  parent=seg{info[parent]['id']} "
              f"(strahler={info[parent]['strahler']}, iref_raw={target_r:.0f})  "
              f"target_r={target_r:.0f}")
        for k, e in enumerate(info):
            tag = "TRUNK?" if (e["strahler"] == info[parent]["strahler"] and k != parent) else \
                  ("PARENT" if k == parent else "branch")
            drop = 100.0 * (e["iref_fin"] - e["iref_raw"]) / max(e["iref_raw"], 1e-9)
            row_raw = " ".join(f"{r_at(stages['raw'], pid):5.0f}" for pid in e["pids"][:NPROF])
            row_fin = " ".join(f"{r_at(final, pid):5.0f}" for pid in e["pids"][:NPROF])
            print(f"  seg{str(e['id']):>5} S{e['strahler']} {tag:<6} "
                  f"iref raw={e['iref_raw']:5.0f} fin={e['iref_fin']:5.0f} ({drop:+5.1f}%)")
            print(f"      raw : {row_raw}")
            print(f"      fin : {row_fin}")
        # node-at radius vs interior for the parent/trunk: is the AT-node value below interior?
        for k, e in enumerate(info):
            at = r_at(final, e["pids"][0]) if e["pids"] else 0.0
            if e["iref_fin"] > 0 and at < 0.85 * e["iref_fin"]:
                print(f"  >>> seg{e['id']} NECKS at node: r_at_node={at:.0f} < 0.85*interior={e['iref_fin']:.0f}")

    # quick deg-3 controls
    deg3 = [(nid, segs) for nid, segs in n2s.items() if len(segs) == 3]
    deg3.sort(key=lambda t: node_thickness(*t), reverse=True)
    print("\n" + "=" * 96)
    print(f"[probe] {len(deg3)} degree-3 nodes; top 3 by trunk radius (controls):")
    for nid, segs in deg3[:3]:
        rr = []
        for si in segs:
            pids = node_ward_pids(g_segments[si], nid)
            rr.append((g_segments[si].get("id"), interior_ref(final, pids),
                       r_at(final, pids[0]) if pids else 0.0))
        desc = "  ".join(f"seg{i}:iref={ir:.0f},at={at:.0f}" for i, ir, at in rr)
        print(f"  NODE {nid}: {desc}")

    # ── Geometry check: union-surface dip + centerline kink/offset ──────────
    print("\n" + "#" * 96)
    print("# GEOMETRY: does the PURE capsule union dish at the node? (smin/flat-cap/carve OFF)")
    print("#" * 96)
    raw_pts = stages["raw"]
    for nid, segs in multif[:3]:
        analyze_geometry(final, raw_pts, g_nodes, g_segments, n2s, nid, "deg4")
    for nid, segs in deg3[:5]:
        analyze_geometry(final, raw_pts, g_nodes, g_segments, n2s, nid, "deg3")

    # ── EXACT pipeline capsules (prepare_segment_spline -> build_capsules) ──
    print("\n" + "#" * 96)
    print("# EXACT pipeline capsules: union dish at the same nodes (gold-standard geometry)")
    print("#" * 96)
    n2s_full: dict[int, set[int]] = {}
    for idx, seg in enumerate(g_segments):
        for nd in (seg["node1"], seg["node2"]):
            n2s_full.setdefault(nd, set()).add(idx)
    splines = []
    for i, seg in enumerate(g_segments):
        sp = prepare_segment_spline(seg, final, g_nodes)
        if sp is not None:
            sp["seg_idx"] = i
            splines.append(sp)
    caps = build_capsules(splines, node_to_segs=n2s_full)
    from coronary_sdf.sdf_field import compute_grid
    _grid = compute_grid(caps)
    print(f"  [GRID] graph0 auto voxel = {_grid.voxel_size*1000:.0f} um  "
          f"(thick trunk R~1270um -> ~{2*1270/_grid.voxel_size/1000:.1f} voxels across diameter)")
    GS, GE = np.asarray(caps.starts), np.asarray(caps.ends)
    GR0, GR1 = np.asarray(caps.radii_start), np.asarray(caps.radii_end)
    GBS = np.asarray(getattr(caps, "cap_bif_at_start", np.zeros(len(GS), bool)), bool)
    GBE = np.asarray(getattr(caps, "cap_bif_at_end", np.zeros(len(GS), bool)), bool)
    gmid = 0.5 * (GS + GE)
    th = np.linspace(0, 2 * np.pi, 24, endpoint=False)

    def union_with_flatcap(P, idx):
        """Union SDF over local capsules idx, WITH SDF_FLAT_CAP_BIF applied."""
        S, E, R0, R1 = GS[idx], GE[idx], GR0[idx], GR1[idx]
        BS, BE = GBS[idx], GBE[idx]
        d = E - S
        dd = np.sum(d * d, axis=1)
        du = d / np.sqrt(np.maximum(dd, 1e-12))[:, None]
        PS = P[:, None, :] - S[None, :, :]
        t = np.clip(np.einsum("mki,ki->mk", PS, d) / np.maximum(dd, 1e-12), 0.0, 1.0)
        closest = S[None, :, :] + t[..., None] * d[None, :, :]
        sdf = np.linalg.norm(P[:, None, :] - closest, axis=2) - (R0[None, :] + t * (R1 - R0)[None, :])
        beyond_end = np.einsum("mki,ki->mk", P[:, None, :] - E[None, :, :], du)
        sdf = np.where(BE[None, :], np.maximum(sdf, beyond_end), sdf)
        beyond_start = -np.einsum("mki,ki->mk", PS, du)
        sdf = np.where(BS[None, :], np.maximum(sdf, beyond_start), sdf)
        return sdf.min(axis=1)

    def exact_dish(nid):
        segs = n2s[nid]
        tans = _tangents(final, g_segments, segs, nid)
        if sum(t is not None for t in tans) < 2:
            return None
        i, j = _trunk_pair(tans)
        poly, rad, k0 = _trunk_polyline(final, g_segments, segs, nid, i, j)
        R = float(np.median(rad))
        m = np.linalg.norm(gmid - poly[k0][None, :], axis=1) <= max(6.0, 4 * R)
        loc = (GS[m], GE[m], GR0[m], GR1[m])
        peak_plain, peak_fc = -1e9, -1e9
        for k in range(1, len(poly) - 1):
            tan = poly[k + 1] - poly[k - 1]
            nrm = np.linalg.norm(tan)
            if nrm < 1e-9:
                continue
            tan /= nrm
            ref = np.array([1.0, 0, 0]) if abs(tan[0]) < 0.9 else np.array([0, 1.0, 0])
            u = np.cross(tan, ref); u /= max(np.linalg.norm(u), 1e-9)
            v = np.cross(tan, u)
            rdir = np.cos(th)[:, None] * u[None, :] + np.sin(th)[:, None] * v[None, :]
            P = poly[k][None, :] + rad[k] * rdir
            peak_plain = max(peak_plain, float((union_sdf(P, loc) * 1000.0).max()))
            peak_fc = max(peak_fc, float((union_with_flatcap(P, m) * 1000.0).max()))
        return peak_plain, peak_fc

    for nid, _ in (multif[:3] + deg3[:5]):
        r = exact_dish(nid)
        if r is not None:
            pk, pkfc = r
            print(f"  NODE {nid}: peak inward  union={pk:5.0f}um   union+FLATCAP={pkfc:5.0f}um   "
                  f"-> {'FLAT-CAP carves the trunk (DISH)' if pkfc > 80 and pkfc > pk + 60 else 'no flat-cap dish'}")

    # ── REAL evaluate_sdf on a local grid (production code, no approximation) ──
    print("\n" + "#" * 96)
    print("# REAL evaluate_sdf on a local grid at the node (full smin/flat-cap/gates)")
    print("#" * 96)
    from coronary_sdf.sdf_field import (
        build_adjacency, find_bifurcations, collect_endpoint_info,
        build_terminal_set, build_narrow_band, evaluate_sdf, Grid,
    )
    adj, _sp, _sr = build_adjacency(g_nodes, final, g_segments, n2s_full)
    bif = find_bifurcations(g_nodes, final, g_segments, n2s_full)
    term = build_terminal_set(collect_endpoint_info(g_nodes, final, g_segments, n2s_full))
    cap_is_junc = np.zeros(caps.n, dtype=bool)
    vsz = float(_grid.voxel_size)

    def trilerp(vol, bmin, vs, P):
        g = (P - bmin) / vs
        i0 = np.floor(g).astype(int)
        f = g - i0
        out = np.empty(len(P))
        dims = vol.shape
        for n in range(len(P)):
            x, y, z = i0[n]
            if not (0 <= x < dims[0] - 1 and 0 <= y < dims[1] - 1 and 0 <= z < dims[2] - 1):
                out[n] = np.nan; continue
            fx, fy, fz = f[n]
            c = vol[x:x + 2, y:y + 2, z:z + 2]
            out[n] = (
                c[0, 0, 0] * (1 - fx) * (1 - fy) * (1 - fz) + c[1, 0, 0] * fx * (1 - fy) * (1 - fz)
                + c[0, 1, 0] * (1 - fx) * fy * (1 - fz) + c[0, 0, 1] * (1 - fx) * (1 - fy) * fz
                + c[1, 1, 0] * fx * fy * (1 - fz) + c[1, 0, 1] * fx * (1 - fy) * fz
                + c[0, 1, 1] * (1 - fx) * fy * fz + c[1, 1, 1] * fx * fy * fz
            )
        return out

    def real_dish(nid):
        segs = n2s[nid]
        tans = _tangents(final, g_segments, segs, nid)
        if sum(t is not None for t in tans) < 2:
            return
        i, j = _trunk_pair(tans)
        poly, rad, k0 = _trunk_polyline(final, g_segments, segs, nid, i, j)
        center = poly[k0]
        half = 5.0
        bmin = center - half
        bmax = center + half
        dims = np.ceil((bmax - bmin) / vsz).astype(int) + 1
        x = np.linspace(bmin[0], bmax[0], dims[0])
        y = np.linspace(bmin[1], bmax[1], dims[1])
        z = np.linspace(bmin[2], bmax[2], dims[2])
        grid = Grid(bmin, bmax, vsz, dims, x, y, z)
        nb = build_narrow_band(caps, grid)
        vol = evaluate_sdf(caps, cap_is_junc, adj, bif, term, grid, nb).sdf
        # sample REAL sdf at nominal-surface rings along the trunk
        worst = -1e9
        for k in range(1, len(poly) - 1):
            tan = poly[k + 1] - poly[k - 1]
            n = np.linalg.norm(tan)
            if n < 1e-9:
                continue
            tan /= n
            ref = np.array([1.0, 0, 0]) if abs(tan[0]) < 0.9 else np.array([0, 1.0, 0])
            u = np.cross(tan, ref); u /= max(np.linalg.norm(u), 1e-9)
            v = np.cross(tan, u)
            rdir = np.cos(th)[:, None] * u[None, :] + np.sin(th)[:, None] * v[None, :]
            P = poly[k][None, :] + rad[k] * rdir
            s = trilerp(vol, bmin, vsz, P) * 1000.0
            s = s[np.isfinite(s)]
            if len(s):
                worst = max(worst, float(s.max()))
        print(f"  NODE {nid}: REAL-SDF peak inward at nominal surface = {worst:5.0f}um "
              f"-> {'TRUE DISH in evaluate_sdf' if worst > 80 else 'clean'}")

    for nid, _ in (multif[:2] + deg3[:2]):
        real_dish(nid)

    # ── Smin raise profile + transition smoothness (symmetric fix ON vs OFF) ──
    print("\n" + "#" * 96)
    print("# SMIN RAISE: surface offset along trunk through bif + transition crease metric")
    print("#" * 96)

    def _vol(nid_poly, force_hard, symmetric):
        config.FORCE_HARD_MIN_ONLY = force_hard
        config.SMIN_SYMMETRIC_BLEND_RADIUS = symmetric
        poly, rad, k0, center = nid_poly
        bmin = center - 5.0
        bmax = center + 5.0
        dims = np.ceil((bmax - bmin) / vsz).astype(int) + 1
        grid = Grid(bmin, bmax, vsz, dims,
                    np.linspace(bmin[0], bmax[0], dims[0]),
                    np.linspace(bmin[1], bmax[1], dims[1]),
                    np.linspace(bmin[2], bmax[2], dims[2]))
        nb = build_narrow_band(caps, grid)
        return evaluate_sdf(caps, cap_is_junc, adj, bif, term, grid, nb).sdf, bmin

    def smin_raise(nid, symmetric):
        """Isolated smin raise = sdf_hard - sdf_smin at the nominal surface,
        sampled along the trunk. Peak (um) + crease (max |2nd diff|)."""
        segs = n2s[nid]
        tans = _tangents(final, g_segments, segs, nid)
        if sum(t is not None for t in tans) < 2:
            return None
        i, j = _trunk_pair(tans)
        poly, rad, k0 = _trunk_polyline(final, g_segments, segs, nid, i, j, npts=18)
        npoly = (poly, rad, k0, poly[k0])
        vol_h, bmin = _vol(npoly, True, symmetric)
        vol_s, _ = _vol(npoly, False, symmetric)
        rz = []
        for k in range(1, len(poly) - 1):
            tan = poly[k + 1] - poly[k - 1]
            nn = np.linalg.norm(tan)
            if nn < 1e-9:
                continue
            tan /= nn
            ref = np.array([1.0, 0, 0]) if abs(tan[0]) < 0.9 else np.array([0, 1.0, 0])
            u = np.cross(tan, ref); u /= max(np.linalg.norm(u), 1e-9)
            v = np.cross(tan, u)
            rdir = np.cos(th)[:, None] * u[None, :] + np.sin(th)[:, None] * v[None, :]
            P = poly[k][None, :] + rad[k] * rdir
            sh = trilerp(vol_h, bmin, vsz, P)
            ss = trilerp(vol_s, bmin, vsz, P)
            d = (sh - ss)  # >0 where smin pushed surface outward (raise), mm
            d = d[np.isfinite(d)]
            rz.append(float(np.nanmax(d)) * 1000.0 if len(d) else 0.0)
        rz = np.array(rz)
        crease = float(np.abs(np.diff(rz, 2)).max()) if len(rz) > 2 else 0.0
        return rz.max(), crease

    base_cap = config.BLEND_BULGE_CAP_MM
    base_prox = config.SMIN_PROXIMITY_BLEND_FACTOR
    sweep = [(0.05, 0.2, "baseline   "), (0.03, 0.2, "cap=0.03   "),
             (0.05, 0.4, "prox=0.4   "), (0.03, 0.4, "cap.03+pr.4"),
             (0.02, 0.5, "cap.02+pr.5")]
    for nid, _ in (deg3[:2]):
        print(f"  NODE {nid}:")
        for cap, prox, label in sweep:
            config.BLEND_BULGE_CAP_MM = cap
            config.SMIN_PROXIMITY_BLEND_FACTOR = prox
            r = smin_raise(nid, True)
            if r:
                print(f"      {label}: peak raise={r[0]:4.0f}um  crease(max|d2|)={r[1]:5.1f}um")
    config.BLEND_BULGE_CAP_MM = base_cap
    config.SMIN_PROXIMITY_BLEND_FACTOR = base_prox
    config.FORCE_HARD_MIN_ONLY = False
    config.SMIN_SYMMETRIC_BLEND_RADIUS = True


if __name__ == "__main__":
    main()
