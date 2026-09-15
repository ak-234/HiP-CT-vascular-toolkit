"""Epicardial-vessel annotation + side-branch pruning model series.

Builds a *series* of coronary tree models, each keeping fewer side branches, to
study the effect on flow in the **main epicardial vessels** (LAD, LCx, RCA...).

Workflow
--------
1. Parse + canonical-preprocess the spatial graph ONCE (same chain as the surface
   pipeline / flow_fractions, via :func:`flow_fractions.preprocess_topology`).
2. Manually annotate which segments form the main epicardial vessels, grouped into
   named vessels, in an interactive 3D PyVista picker. The selection is persisted
   to a sidecar JSON so models regenerate without re-picking.  Main vessels are
   ALWAYS preserved.
3. For each radius ratio in {1/2, 1/5, 1/10, ...}: remove every side branch (and
   its whole downstream subtree) whose take-off radius is below
   ``ratio * (ostial radius of the main vessel it descends from)``; then run a
   length-independent containment prune that drops "swallowed" leaf stubs whose
   centerline lies almost entirely inside a neighbouring segment's lumen.
4. Write the pruned tree back to an Amira ``.am.xml`` (round-trips through
   :func:`parse_amira.parse_xml`), run the surface pipeline, and compute the
   Van der Giessen flow split, reporting the flow carried by each main vessel.

Main vessels are re-identified across pruning by **point-id set membership**
(stable across degree-2 merges), not by segment index.  The sidecar stores a
geometric ``seg_key`` per annotated segment which resolves against the fixed
(deterministic) base topology.

Usage::

    python -m coronary_sdf.epicardial_annotation --xml <in.am.xml> --out <dir> --pick
    # first run: pick + save sidecar; subsequent runs reuse the sidecar:
    python -m coronary_sdf.epicardial_annotation --xml <in.am.xml> --out <dir>
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any
from xml.dom import minidom
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial import KDTree

from . import config
from .parse_amira import parse_xml
from .splines import compute_frenet_frame, prepare_segment_spline
from .topology import (
    build_directed_topology,
    find_connected_components,
    merge_degree2_segments,
    split_by_graph,
)
from .flow_fractions import (
    BIF_SKIP_POINTS,
    _radius_mm,
    branch_radius_mm,
    compute_flow_fractions,
    preprocess_topology,
    smooth_graph,
)
from .pruning import segment_arc_length

SS = "urn:schemas-microsoft-com:office:spreadsheet"

# Preset vessel names cycled by the picker's "n" key (Other -> typed in console).
PRESET_VESSEL_NAMES = ["LAD", "LCx", "IM", "RCA", "Diag", "OM", "PDA", "PLB", "Ramus"]
VESSEL_COLORS = [
    "#d62728", "#1f77b4", "#2ca02c", "#9467bd", "#ff7f0e",
    "#17becf", "#bcbd22", "#e377c2", "#8c564b",
]


# ── Base-topology helpers ─────────────────────────────────────────────────────


def build_node_to_segs(segments: list[dict[str, Any]]) -> dict[int, set[int]]:
    """``node_id -> {segment indices incident to that node}``."""
    n2s: dict[int, set[int]] = {}
    for i, s in enumerate(segments):
        for nid in (s["node1"], s["node2"]):
            n2s.setdefault(nid, set()).add(i)
    return n2s


def recompute_node_degrees(
    nodes: dict[int, tuple], segments: list[dict[str, Any]]
) -> dict[int, tuple]:
    """Return a fresh nodes dict with Coordination Number recomputed from
    ``segments``; nodes no longer referenced are dropped."""
    deg: dict[int, int] = {}
    for s in segments:
        deg[s["node1"]] = deg.get(s["node1"], 0) + 1
        deg[s["node2"]] = deg.get(s["node2"], 0) + 1
    out: dict[int, tuple] = {}
    for nid, d in deg.items():
        if nid in nodes:
            x, y, z = nodes[nid][0], nodes[nid][1], nodes[nid][2]
        else:  # node referenced but missing — keep a placeholder at origin
            x, y, z = 0.0, 0.0, 0.0
        out[nid] = (x, y, z, d)
    return out


def filter_points(
    points: dict[int, tuple], segments: list[dict[str, Any]]
) -> dict[int, tuple]:
    """Keep only points referenced by ``segments``."""
    keep: set[int] = set()
    for s in segments:
        keep.update(s["point_ids"])
    return {pid: data for pid, data in points.items() if pid in keep}


def copy_points(points: dict[int, tuple]) -> dict[int, tuple]:
    """Deep-ish copy so radius blending in merges never leaks across ratios."""
    return {pid: tuple(v) for pid, v in points.items()}


def copy_segments(segments: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [dict(s, point_ids=list(s["point_ids"])) for s in segments]


def children_of(parent_idx: np.ndarray) -> dict[int, list[int]]:
    ch: dict[int, list[int]] = {i: [] for i in range(len(parent_idx))}
    for c in range(len(parent_idx)):
        p = int(parent_idx[c])
        if p >= 0:
            ch[p].append(c)
    return ch


def seg_depths(parent_idx: np.ndarray) -> np.ndarray:
    """BFS depth of each segment from its component root (root depth = 0)."""
    n = len(parent_idx)
    depth = np.zeros(n, dtype=np.int64)
    for c in range(n):
        d = 0
        cur = int(parent_idx[c])
        seen = 0
        while cur >= 0 and seen <= n:
            d += 1
            cur = int(parent_idx[cur])
            seen += 1
        depth[c] = d
    return depth


def shared_node_with_parent(
    c: int, segments: list[dict[str, Any]], parent_idx: np.ndarray
) -> int | None:
    p = int(parent_idx[c])
    if p < 0:
        return None
    cn = {segments[c]["node1"], segments[c]["node2"]}
    pn = {segments[p]["node1"], segments[p]["node2"]}
    common = cn & pn
    return next(iter(common)) if common else None


def prox_node(
    c: int, segments: list[dict[str, Any]], parent_idx: np.ndarray, nodes: dict[int, tuple]
) -> int:
    """Proximal (parent / inlet) node of segment ``c``."""
    sh = shared_node_with_parent(c, segments, parent_idx)
    if sh is not None:
        return sh
    n1, n2 = segments[c]["node1"], segments[c]["node2"]

    def deg(nid: int) -> int:
        return nodes[nid][3] if nid in nodes else 0

    return n1 if deg(n1) == 1 else (n2 if deg(n2) == 1 else n1)


# ── Stable segment identity (sidecar <-> base resolution) ──────────────────────


def _pt_key(points: dict[int, tuple], pid: int) -> tuple[int, int, int]:
    """Integer-micrometre coordinate of a point (1 µm resolution)."""
    x, y, z = points[pid][0], points[pid][1], points[pid][2]
    return (int(round(x)), int(round(y)), int(round(z)))


def build_segment_keys(
    segments: list[dict[str, Any]], points: dict[int, tuple]
) -> list[str]:
    """A geometric fingerprint per segment, stable across deterministic re-runs.

    Uses the two endpoint coordinates (sorted, undirected) plus the mid-point
    coordinate, all at 1 µm resolution, hashed to 16 hex chars.
    """
    keys: list[str] = []
    for s in segments:
        pids = s["point_ids"]
        if not pids:
            keys.append("")
            continue
        e0 = _pt_key(points, pids[0])
        e1 = _pt_key(points, pids[-1])
        lo, hi = (e0, e1) if e0 <= e1 else (e1, e0)
        mid = _pt_key(points, pids[len(pids) // 2])
        raw = f"{lo}|{hi}|{mid}"
        keys.append(hashlib.sha1(raw.encode()).hexdigest()[:16])
    return keys


# ── Amira XML writer (round-trips through parse_amira.parse_xml) ───────────────


def _cell(row: ET.Element, value: Any, type_: str) -> None:
    c = ET.SubElement(row, f"{{{SS}}}Cell")
    d = ET.SubElement(c, f"{{{SS}}}Data")
    d.set(f"{{{SS}}}Type", type_)
    d.text = str(value)


def _header_row(table: ET.Element, names: list[str]) -> None:
    row = ET.SubElement(table, f"{{{SS}}}Row")
    for nm in names:
        _cell(row, nm, "String")


def _num(x: float) -> str:
    # Full round-trippable float repr; ints stay clean.
    return repr(float(x))


def write_amira_xml(
    nodes: dict[int, tuple],
    points: dict[int, tuple],
    segments: list[dict[str, Any]],
    xml_path: str | Path,
) -> None:
    """Serialise (nodes, points, segments) to an Amira SpatialGraph XML.

    Emits exactly the named columns the header-driven parser looks up; only
    nodes/points referenced by ``segments`` are written. Original integer ids
    are preserved so ``Point IDs`` strings stay valid.
    """
    xml_path = Path(xml_path)
    xml_path.parent.mkdir(parents=True, exist_ok=True)

    nodes = recompute_node_degrees(nodes, segments)
    points = filter_points(points, segments)

    ET.register_namespace("ss", SS)
    wb = ET.Element(f"{{{SS}}}Workbook")

    # Nodes worksheet
    ws = ET.SubElement(wb, f"{{{SS}}}Worksheet")
    ws.set(f"{{{SS}}}Name", "Nodes")
    tbl = ET.SubElement(ws, f"{{{SS}}}Table")
    _header_row(tbl, ["Node ID", "X Coord", "Y Coord", "Z Coord", "Coordination Number"])
    for nid in sorted(nodes):
        x, y, z, coord = nodes[nid]
        row = ET.SubElement(tbl, f"{{{SS}}}Row")
        _cell(row, int(nid), "Number")
        _cell(row, _num(x), "Number")
        _cell(row, _num(y), "Number")
        _cell(row, _num(z), "Number")
        _cell(row, int(coord), "Number")

    # Points worksheet
    ws = ET.SubElement(wb, f"{{{SS}}}Worksheet")
    ws.set(f"{{{SS}}}Name", "Points")
    tbl = ET.SubElement(ws, f"{{{SS}}}Table")
    _header_row(tbl, ["Point ID", "thickness", "X Coord", "Y Coord", "Z Coord"])
    for pid in sorted(points):
        x, y, z, th = points[pid]
        row = ET.SubElement(tbl, f"{{{SS}}}Row")
        _cell(row, int(pid), "Number")
        _cell(row, _num(th), "Number")
        _cell(row, _num(x), "Number")
        _cell(row, _num(y), "Number")
        _cell(row, _num(z), "Number")

    # Segments worksheet
    ws = ET.SubElement(wb, f"{{{SS}}}Worksheet")
    ws.set(f"{{{SS}}}Name", "Segments")
    tbl = ET.SubElement(ws, f"{{{SS}}}Table")
    has_strahler = any("strahler" in s for s in segments)
    hdr = ["Segment ID", "Node ID #1", "Node ID #2"]
    if has_strahler:
        hdr.append("strahler")
    hdr.append("Point IDs")
    _header_row(tbl, hdr)
    for s in segments:
        row = ET.SubElement(tbl, f"{{{SS}}}Row")
        _cell(row, int(s["id"]), "Number")
        _cell(row, int(s["node1"]), "Number")
        _cell(row, int(s["node2"]), "Number")
        if has_strahler:
            _cell(row, int(s.get("strahler", 0)), "Number")
        _cell(row, ",".join(str(int(p)) for p in s["point_ids"]), "String")

    rough = ET.tostring(wb, encoding="utf-8")
    pretty = minidom.parseString(rough).toprettyxml(indent="  ", encoding="utf-8")
    xml_path.write_bytes(pretty)
    print(f"[WRITE] {xml_path}  ({len(nodes)} nodes, {len(points)} points, "
          f"{len(segments)} segments)")


# ── Vessel membership / ostia / descent ───────────────────────────────────────


def assign_vessel(
    seg: dict[str, Any], vessel_points: dict[str, set[int]], min_frac: float = 0.5
) -> str | None:
    """Vessel a segment belongs to, by majority point-id membership (or None)."""
    pids = seg["point_ids"]
    if not pids:
        return None
    best_v, best_overlap = None, 0
    pset = set(pids)
    for v, vp in vessel_points.items():
        ov = len(pset & vp)
        if ov > best_overlap:
            best_overlap, best_v = ov, v
    if best_v is not None and best_overlap / len(pids) >= min_frac:
        return best_v
    return None


def seg_vessel_map(
    segments: list[dict[str, Any]], vessel_points: dict[str, set[int]]
) -> dict[int, str]:
    """``seg_idx -> vessel name`` for segments that are part of a main vessel."""
    out: dict[int, str] = {}
    for i, s in enumerate(segments):
        v = assign_vessel(s, vessel_points)
        if v is not None:
            out[i] = v
    return out


def compute_vessel_ostia(
    vessels_idx: dict[str, set[int]],
    nodes: dict[int, tuple],
    points: dict[int, tuple],
    segments: list[dict[str, Any]],
    skip_points: int | None = None,
    n_average: int | None = None,
) -> dict[str, dict[str, Any]]:
    """Per-vessel ostial (most-proximal) segment + radius, measured on the base.

    The ostial radius is the denominator for that vessel's side-branch threshold.
    Measured with :func:`branch_radius_mm` so it skips the inflated ostium
    contours and averages a controlled number of downstream contours.
    """
    nodes = recompute_node_degrees(nodes, segments)
    dtopo = build_directed_topology(segments, build_node_to_segs(segments))
    parent_idx = dtopo["parent_seg_idx"]
    depth = seg_depths(parent_idx)

    ostia: dict[str, dict[str, Any]] = {}
    for v, idxs in vessels_idx.items():
        if not idxs:
            continue
        ostial = min(idxs, key=lambda i: int(depth[i]))
        onode = prox_node(ostial, segments, parent_idx, nodes)
        r, _ = branch_radius_mm(segments[ostial], onode, points, nodes,
                                skip_points=skip_points, n_average=n_average)
        ostia[v] = {
            "ostial_seg_idx": int(ostial),
            "ostial_node": int(onode),
            "radius_mm": float(r),
        }
        print(f"  [OSTIUM] {v}: seg#{ostial} ostial radius = {r:.4f} mm "
              f"({len(idxs)} segment(s))")
    return ostia


def _clean_seg_radii(
    seg: dict[str, Any], points: dict[int, tuple], nodes: dict[int, tuple], skip: int
) -> list[float]:
    """Per-point radii (mm) of a segment, dropping ``skip`` contours at any
    bifurcation end (node coordination >= 3) so junction-inflated contours are
    excluded. Degree-1 ends (true ostium / tip) are kept. Falls back to the
    central contour if a short junction-to-junction segment trims to empty."""
    pids = seg["point_ids"]
    n = len(pids)
    if n == 0:
        return []

    def deg(nid: int) -> int:
        return nodes[nid][3] if nid in nodes else 0

    lo = skip if deg(seg["node1"]) >= 3 else 0
    hi = n - (skip if deg(seg["node2"]) >= 3 else 0)
    sub = pids[lo:hi] if lo < hi else pids[n // 2:n // 2 + 1]
    return [_radius_mm(points, p) for p in sub if p in points]


def compute_vessel_radius_stats(
    vessels_idx: dict[str, set[int]],
    nodes: dict[int, tuple],
    points: dict[int, tuple],
    segments: list[dict[str, Any]],
    skip_points: int | None = None,
) -> list[dict[str, Any]]:
    """Per-vessel radius (mm) min/mean/max over the total vessel, its proximal
    (ostial / min-depth) segment, and its distal (max-depth) segment. Excludes
    junction-inflated contours (see :func:`_clean_seg_radii`)."""
    skip = BIF_SKIP_POINTS if skip_points is None else int(skip_points)
    nodes = recompute_node_degrees(nodes, segments)
    dtopo = build_directed_topology(segments, build_node_to_segs(segments))
    parent_idx = dtopo["parent_seg_idx"]
    depth = seg_depths(parent_idx)

    def _stats(rs: list[float]) -> tuple[int, Any, Any, Any]:
        if not rs:
            return 0, None, None, None
        a = np.asarray(rs, dtype=np.float64)
        return (len(a), round(float(a.min()), 5),
                round(float(a.mean()), 5), round(float(a.max()), 5))

    rows: list[dict[str, Any]] = []
    for v, idxs in vessels_idx.items():
        if not idxs:
            continue
        idxs = list(idxs)
        ostial = min(idxs, key=lambda i: int(depth[i]))
        distal = max(idxs, key=lambda i: (int(depth[i]),
                                          segment_arc_length(segments[i], points)))
        total_r = [r for i in idxs for r in _clean_seg_radii(segments[i], points, nodes, skip)]
        prox_r = _clean_seg_radii(segments[ostial], points, nodes, skip)
        dist_r = _clean_seg_radii(segments[distal], points, nodes, skip)
        tn, tmin, tmean, tmax = _stats(total_r)
        pn, pmin, pmean, pmax = _stats(prox_r)
        dn, dmin, dmean, dmax = _stats(dist_r)
        rows.append({
            "vessel": v,
            "total_n": tn, "total_min_mm": tmin, "total_mean_mm": tmean, "total_max_mm": tmax,
            "prox_n": pn, "prox_min_mm": pmin, "prox_mean_mm": pmean, "prox_max_mm": pmax,
            "dist_n": dn, "dist_min_mm": dmin, "dist_mean_mm": dmean, "dist_max_mm": dmax,
        })
        print(f"  [RADIUS] {v}: total mean={tmean} mm (min={tmin}, max={tmax}); "
              f"prox mean={pmean}, dist mean={dmean}")
    return rows


def build_descent_map(
    segments: list[dict[str, Any]],
    parent_idx: np.ndarray,
    seg_vessel: dict[int, str],
) -> dict[int, str | None]:
    """Every segment -> the main vessel it descends from (walk parents to the
    first annotated main segment), or None if no annotated ancestor."""
    descends: dict[int, str | None] = {}
    for start in range(len(segments)):
        path: list[int] = []
        cur = start
        result: str | None = None
        while cur >= 0:
            if cur in descends:
                result = descends[cur]
                break
            if cur in seg_vessel:
                result = seg_vessel[cur]
                break
            path.append(cur)
            cur = int(parent_idx[cur])
        for c in path:
            descends[c] = result
    return descends


def _collect_subtree(c: int, children: dict[int, list[int]]) -> set[int]:
    out: set[int] = set()
    stack = [c]
    while stack:
        x = stack.pop()
        if x in out:
            continue
        out.add(x)
        stack.extend(children.get(x, ()))
    return out


# ── Radius-ratio subtree prune ────────────────────────────────────────────────


def prune_by_radius_ratio(
    nodes: dict[int, tuple],
    points: dict[int, tuple],
    segments: list[dict[str, Any]],
    vessel_points: dict[str, set[int]],
    ostia: dict[str, dict[str, Any]],
    ratio: float,
    prune_unattributed: bool = False,
    max_iters: int = 8,
    skip_points: int | None = None,
    n_average: int | None = None,
) -> tuple[dict[int, tuple], dict[int, tuple], list[dict[str, Any]], list[dict[str, Any]]]:
    """Remove every side branch (and its whole downstream subtree) whose take-off
    radius is below ``ratio * ostial_radius`` of the vessel it descends from.

    Main (annotated) segments are never removed. Iterates: remove subtrees ->
    contract newly-exposed degree-2 nodes -> repeat until stable. ``points`` is
    mutated by the degree-2 radius blend (pass a per-model copy).

    Returns ``(nodes, points, segments, pruned)`` where ``pruned`` is one record
    per pruned take-off branch: ``{seg_id, vessel, radius_mm, threshold_mm,
    n_subtree_removed}``. Take-off radius is measured with :func:`branch_radius_mm`
    (skips the inflated ostium contours).
    """
    segments = copy_segments(segments)
    total_removed = 0
    pruned: list[dict[str, Any]] = []
    for _it in range(max_iters):
        nodes = recompute_node_degrees(nodes, segments)
        dtopo = build_directed_topology(segments, build_node_to_segs(segments))
        parent_idx = dtopo["parent_seg_idx"]
        children = children_of(parent_idx)
        roots = [i for i in range(len(segments)) if int(parent_idx[i]) < 0]
        svm = seg_vessel_map(segments, vessel_points)
        main_set = set(svm)
        descends = build_descent_map(segments, parent_idx, svm)

        # Segments on the path between a root and a main vessel (e.g. an
        # un-annotated LM proximal to the LAD) must never be dropped.
        ancestors_of_main: set[int] = set()
        for m in main_set:
            cur = int(parent_idx[m])
            while cur >= 0 and cur not in ancestors_of_main:
                ancestors_of_main.add(cur)
                cur = int(parent_idx[cur])

        remove: set[int] = set()
        for root in roots:
            stack = [root]
            while stack:
                c = stack.pop()
                if c in remove:
                    continue
                if c in main_set:
                    stack.extend(children[c])
                    continue
                v = descends.get(c)
                drop = False
                r_takeoff = thr = None
                if v is None:
                    drop = prune_unattributed and c not in ancestors_of_main
                else:
                    sh = shared_node_with_parent(c, segments, parent_idx)
                    if sh is not None:
                        r_takeoff, _ = branch_radius_mm(
                            segments[c], sh, points, nodes,
                            skip_points=skip_points, n_average=n_average)
                        thr = ratio * ostia[v]["radius_mm"]
                        drop = r_takeoff < thr
                if drop:
                    subtree = _collect_subtree(c, children)
                    remove |= subtree
                    pruned.append({
                        "seg_id": int(segments[c]["id"]),
                        "vessel": v if v is not None else "",
                        "radius_mm": None if r_takeoff is None else round(r_takeoff, 5),
                        "threshold_mm": None if thr is None else round(thr, 5),
                        "n_subtree_removed": len(subtree),
                    })
                else:
                    stack.extend(children[c])

        if not remove:
            break
        total_removed += len(remove)
        kept = [s for i, s in enumerate(segments) if i not in remove]
        nodes, segments, _ = merge_degree2_segments(nodes, points, kept)

    nodes = recompute_node_degrees(nodes, segments)
    points = filter_points(points, segments)
    print(f"  [RATIO {ratio:.4g}] removed {total_removed} segment(s) in "
          f"{len(pruned)} take-off branch(es) -> {len(segments)} remain")
    return nodes, points, segments, pruned


# ── Containment-based stub prune (length-independent) ──────────────────────────


def _segment_capsules(
    segments: list[dict[str, Any]], points: dict[int, tuple]
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Capsule arrays ``(p0, p1, r0, r1, seg_idx)`` (mm) over all segments."""
    p0s, p1s, r0s, r1s, idxs = [], [], [], [], []
    for i, s in enumerate(segments):
        pids = [p for p in s["point_ids"] if p in points]
        if len(pids) < 2:
            continue
        coords = np.array(
            [[points[p][0], points[p][1], points[p][2]] for p in pids]
        ) / 1000.0
        radii = np.array([_radius_mm(points, p) for p in pids])
        p0s.append(coords[:-1])
        p1s.append(coords[1:])
        r0s.append(radii[:-1])
        r1s.append(radii[1:])
        idxs.append(np.full(len(pids) - 1, i, dtype=np.int64))
    if not p0s:
        z3 = np.empty((0, 3))
        z1 = np.empty(0)
        return z3, z3, z1, z1, np.empty(0, dtype=np.int64)
    return (np.vstack(p0s), np.vstack(p1s), np.concatenate(r0s),
            np.concatenate(r1s), np.concatenate(idxs))


def _min_sdf_to_capsules(
    q: np.ndarray, p0: np.ndarray, p1: np.ndarray, r0: np.ndarray,
    r1: np.ndarray, cand: np.ndarray,
) -> float:
    """Minimum capsule SDF of point ``q`` over candidate capsules (<0 = inside)."""
    if len(cand) == 0:
        return np.inf
    a = p0[cand]
    d = p1[cand] - a
    pa = q - a
    dd = np.einsum("ij,ij->i", d, d)
    t = np.clip(np.einsum("ij,ij->i", pa, d) / np.maximum(dd, 1e-12), 0.0, 1.0)
    closest = a + t[:, None] * d
    dist = np.linalg.norm(q - closest, axis=1)
    rad = r0[cand] + t * (r1[cand] - r0[cand])
    return float(np.min(dist - rad))


def prune_contained_leaves(
    nodes: dict[int, tuple],
    points: dict[int, tuple],
    segments: list[dict[str, Any]],
    vessel_points: dict[str, set[int]] | None = None,
    frac_threshold: float = 0.8,
    tip_frac: float = 0.5,
    margin_mm: float = 0.0,
    max_iters: int = 8,
) -> tuple[dict[int, tuple], dict[int, tuple], list[dict[str, Any]]]:
    """Drop "swallowed" leaf stubs whose centerline lies almost entirely inside
    another segment's lumen.

    Length-independent: a leaf is removed iff a fraction >= ``frac_threshold`` of
    its points are inside another segment's lumen (capsule SDF < -margin) AND the
    distal tip (outer ``tip_frac``) is contained. A genuine thin distal vessel that
    runs in open space (containment ~ 0) always survives, regardless of length.
    Annotated main-vessel segments are never removed.
    """
    vessel_points = vessel_points or {}
    segments = copy_segments(segments)
    total = 0
    for _it in range(max_iters):
        nodes = recompute_node_degrees(nodes, segments)
        deg = {nid: nodes[nid][3] for nid in nodes}
        main_set = set(seg_vessel_map(segments, vessel_points))
        p0, p1, r0, r1, segidx = _segment_capsules(segments, points)
        if len(p0) == 0:
            break
        mids = 0.5 * (p0 + p1)
        kd = KDTree(mids)
        cap_len = np.linalg.norm(p1 - p0, axis=1)
        max_cap_r = float(np.max(np.maximum(r0, r1)))
        max_cap_len = float(np.max(cap_len)) if len(cap_len) else 0.0

        remove: set[int] = set()
        for i, s in enumerate(segments):
            if i in main_set:
                continue
            d1, d2 = deg.get(s["node1"], 0), deg.get(s["node2"], 0)
            if not (d1 == 1 or d2 == 1):
                continue  # not a leaf
            pids = [p for p in s["point_ids"] if p in points]
            if not pids:
                continue
            coords = np.array(
                [[points[p][0], points[p][1], points[p][2]] for p in pids]
            ) / 1000.0
            radii = np.array([_radius_mm(points, p) for p in pids])
            if d1 == 1:  # orient so the degree-1 tip is last
                coords = coords[::-1]
                radii = radii[::-1]
            contained = np.zeros(len(coords), dtype=bool)
            for k in range(len(coords)):
                q = coords[k]
                search_r = radii[k] + max_cap_r + 0.5 * max_cap_len
                cand = [c for c in kd.query_ball_point(q, search_r) if segidx[c] != i]
                sdf = _min_sdf_to_capsules(
                    q, p0, p1, r0, r1, np.asarray(cand, dtype=np.int64))
                contained[k] = sdf < -margin_mm
            frac = float(np.mean(contained)) if len(contained) else 0.0
            tip_n = max(1, int(round(len(coords) * tip_frac)))
            tip_ok = bool(np.all(contained[-tip_n:]))
            if frac >= frac_threshold and tip_ok:
                remove.add(i)

        if not remove:
            break
        total += len(remove)
        kept = [s for i, s in enumerate(segments) if i not in remove]
        nodes, segments, _ = merge_degree2_segments(nodes, points, kept)

    nodes = recompute_node_degrees(nodes, segments)
    points = filter_points(points, segments)
    print(f"  [CONTAINED] removed {total} swallowed leaf stub(s) "
          f"-> {len(segments)} remain")
    return nodes, points, segments


# ── Per-model flow report ──────────────────────────────────────────────────────


def compute_main_vessel_flow(
    nodes: dict[int, tuple],
    points: dict[int, tuple],
    segments: list[dict[str, Any]],
    vessel_points: dict[str, set[int]],
    ostia: dict[str, dict[str, Any]] | None = None,
) -> dict[str, dict[str, Any]]:
    """Giessen flow split on the pruned tree; per main vessel report the flow
    fraction entering its ostial segment (root-normalised to 1.0 per tree)."""
    nodes = recompute_node_degrees(nodes, segments)
    graphs = split_by_graph(nodes, points, segments)
    result: dict[str, dict[str, Any]] = {}
    for gid, (g_nodes, g_points, g_segments) in sorted(graphs.items()):
        g_points = smooth_graph(g_nodes, g_points, g_segments)
        _outlets, _inlets, tree_ctx = compute_flow_fractions(
            g_nodes, g_points, g_segments)
        flow = np.asarray(tree_ctx["flow"], dtype=float)
        parent_idx = np.asarray(tree_ctx["parent_idx"])
        depth = seg_depths(parent_idx)
        children = children_of(parent_idx)
        leaves = {int(x) for x in tree_ctx["leaves"]}

        members_by_v: dict[str, list[int]] = {}
        for i, v in seg_vessel_map(g_segments, vessel_points).items():
            members_by_v.setdefault(v, []).append(i)

        for v, members in members_by_v.items():
            ordered = sorted(members, key=lambda i: int(depth[i]))
            ostial, distal = ordered[0], ordered[-1]
            subtree = _collect_subtree(ostial, children)
            leaf_sum = float(sum(flow[ll] for ll in subtree if ll in leaves))
            ostial_flow = float(flow[ostial])
            distal_flow = float(flow[distal])
            entry = {
                # Total share entering the vessel — fixed by the parent split,
                # invariant to pruning the vessel's own side branches.
                "ostial_flow_fraction": ostial_flow,
                # Flow reaching the vessel's distal-most main segment — RISES as
                # side branches are pruned (their flow reroutes down the trunk).
                "distal_flow_fraction": distal_flow,
                # Fraction of the vessel's inflow that reaches its distal end.
                "retained_fraction": (distal_flow / ostial_flow) if ostial_flow > 0 else 0.0,
                # Flow at each main segment, proximal -> distal.
                "along_vessel_flow": [float(flow[m]) for m in ordered],
                "leaf_flow_sum": leaf_sum,
                "n_members": len(members),
                "graph": int(gid),
            }
            if v not in result or entry["ostial_flow_fraction"] > result[v][
                "ostial_flow_fraction"
            ]:
                result[v] = entry
    return result


# ── Annotation sidecar I/O ─────────────────────────────────────────────────────


def _preprocess_signature() -> dict[str, Any]:
    return {
        "MIN_STRAHLER_ORDER": config.MIN_STRAHLER_ORDER,
        "MERGE_DEGREE2_SEGMENTS": config.MERGE_DEGREE2_SEGMENTS,
        "MERGE_SPLIT_MULTIFURCATIONS": config.MERGE_SPLIT_MULTIFURCATIONS,
        "PRUNE_SHORT_TERMINAL_NUBS": config.PRUNE_SHORT_TERMINAL_NUBS,
        "MIN_TERMINAL_LENGTH_MM": config.MIN_TERMINAL_LENGTH_MM,
        "DENSIFY_SPARSE_SEGMENTS": config.DENSIFY_SPARSE_SEGMENTS,
    }


def _file_sha1(path: str | Path) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def save_annotation(
    path: str | Path,
    xml_path: str | Path,
    vessels_idx: dict[str, set[int]],
    seg_keys: list[str],
    colors: dict[str, str] | None = None,
) -> None:
    colors = colors or {}
    vessels: dict[str, Any] = {}
    for k, (v, idxs) in enumerate(vessels_idx.items()):
        vessels[v] = {
            "seg_keys": sorted({seg_keys[i] for i in idxs if seg_keys[i]}),
            "color": colors.get(v, VESSEL_COLORS[k % len(VESSEL_COLORS)]),
        }
    data = {
        "schema_version": 1,
        "source_xml": str(xml_path),
        "source_xml_sha1": _file_sha1(xml_path),
        "preprocess_signature": _preprocess_signature(),
        "vessels": vessels,
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(data, indent=2))
    n = sum(len(d["seg_keys"]) for d in vessels.values())
    print(f"[ANNOT] saved {n} segment key(s) across {len(vessels)} vessel(s) -> {path}")


def resolve_annotation_to_indices(
    data: dict[str, Any], segments: list[dict[str, Any]], seg_keys: list[str]
) -> tuple[dict[str, set[int]], dict[str, str], int]:
    key_to_idx: dict[str, list[int]] = {}
    for i, k in enumerate(seg_keys):
        if k:
            key_to_idx.setdefault(k, []).append(i)
    vessels_idx: dict[str, set[int]] = {}
    colors: dict[str, str] = {}
    unmatched = 0
    for v, d in data["vessels"].items():
        idxs: set[int] = set()
        for k in d.get("seg_keys", []):
            if k in key_to_idx:
                idxs.update(key_to_idx[k])
            else:
                unmatched += 1
        vessels_idx[v] = idxs
        colors[v] = d.get("color", "")
    return vessels_idx, colors, unmatched


def load_or_create_annotation(
    sidecar: str | Path,
    xml_path: str | Path,
    nodes: dict[int, tuple],
    points: dict[int, tuple],
    segments: list[dict[str, Any]],
    seg_keys: list[str],
    force_pick: bool = False,
    hover: bool = True,
) -> tuple[dict[str, set[int]], dict[str, str]]:
    sidecar = Path(sidecar)
    if sidecar.exists() and not force_pick:
        data = json.loads(sidecar.read_text())
        if data.get("preprocess_signature") != _preprocess_signature() or \
                data.get("source_xml_sha1") != _file_sha1(xml_path):
            print("[ANNOT][WARN] sidecar was made against a different XML/preprocess "
                  "config; resolving anyway — re-run with --pick if vessels look wrong.")
        vessels_idx, colors, unmatched = resolve_annotation_to_indices(
            data, segments, seg_keys)
        if unmatched:
            raise SystemExit(
                f"[ANNOT][ERROR] {unmatched} annotated segment key(s) did not resolve "
                "against the current topology. Re-pick with --pick.")
        total = sum(len(s) for s in vessels_idx.values())
        if total == 0:
            raise SystemExit(
                "[ANNOT][ERROR] sidecar resolved to 0 segments. Re-pick with --pick.")
        print(f"[ANNOT] loaded {total} segment(s) across {len(vessels_idx)} "
              f"vessel(s) from {sidecar}")
        return vessels_idx, colors

    vessels_idx, colors = run_picker(nodes, points, segments, hover=hover)
    if not vessels_idx or sum(len(s) for s in vessels_idx.values()) == 0:
        raise SystemExit("[ANNOT][ERROR] no segments selected in picker.")
    save_annotation(sidecar, xml_path, vessels_idx, seg_keys, colors)
    return vessels_idx, colors


# ── Interactive 3D picker ──────────────────────────────────────────────────────


def _segment_contour_mesh(
    coords: np.ndarray,
    radii: np.ndarray,
    n_sides: int = 16,
    ring_stride: int = 1,
) -> Any:
    """Cross-section contour rings + centerline points as one ``pv.PolyData``.

    Per (subsampled) centerline point, a closed ring polyline of radius ``r``
    perpendicular to the local tangent (parallel-transported frame). Centerline
    points are added as vertex cells so they render as dots. Mirrors the ring
    construction in ``flow_fractions._build_diameter_contours``.
    """
    import pyvista as pv

    coords = np.asarray(coords, dtype=np.float64)
    radii = np.asarray(radii, dtype=np.float64)
    n = len(coords)
    theta = np.linspace(0.0, 2.0 * np.pi, n_sides, endpoint=False)
    cos_t, sin_t = np.cos(theta)[:, None], np.sin(theta)[:, None]

    ring_pts: list[np.ndarray] = []
    ring_lines: list[int] = []
    offset = 0
    prev_normal = None
    sample = list(range(0, n, max(1, ring_stride)))
    if n >= 2 and (n - 1) not in sample:
        sample.append(n - 1)
    for i in sample:
        if n < 2:
            break
        if i == 0:
            tang = coords[1] - coords[0]
        elif i == n - 1:
            tang = coords[-1] - coords[-2]
        else:
            tang = coords[i + 1] - coords[i - 1]
        if np.linalg.norm(tang) < 1e-12 or radii[i] <= 0:
            continue
        _t, n_hat, b_hat = compute_frenet_frame(tang, prev_normal)
        prev_normal = n_hat
        ring = coords[i] + radii[i] * (cos_t * n_hat + sin_t * b_hat)
        ring_pts.append(ring)
        ring_lines.append(n_sides + 1)
        ring_lines.extend(range(offset, offset + n_sides))
        ring_lines.append(offset)  # close the loop
        offset += n_sides

    all_pts = np.vstack(ring_pts + [coords]) if ring_pts else coords
    poly = pv.PolyData(all_pts)
    if ring_lines:
        poly.lines = np.asarray(ring_lines, dtype=np.int64)
    # Vertex cells only for the centerline points (they begin at ``offset``).
    verts = np.empty((n, 2), dtype=np.int64)
    verts[:, 0] = 1
    verts[:, 1] = np.arange(offset, offset + n)
    poly.verts = verts.ravel()
    return poly


def run_picker(
    nodes: dict[int, tuple],
    points: dict[int, tuple],
    segments: list[dict[str, Any]],
    hover: bool = True,
) -> tuple[dict[str, set[int]], dict[str, str]]:
    """Interactive picker, one window per tree (connected component).

    Each window shows centerline points + cross-section contour rings (no solid
    surface). Left-click a vessel to toggle it, 'n' to name+commit the current
    group as a vessel, 'c' clear, 'u' undo last vessel, 'q' finish this tree,
    'x' stop annotating remaining trees. Output is combined across trees:
    returns ``({vessel: {global seg_idx}}, {vessel: color})``.
    """
    comps = sorted((list(c) for c in find_connected_components(segments)),
                   key=len, reverse=True)
    print(f"[PICK] {len(comps)} tree(s): sizes {[len(c) for c in comps]}")

    vessels_all: list[tuple[str, set[int], str]] = []  # (name, global idxs, color)
    name_counter = [0]
    stop = {"all": False}
    for ti, comp in enumerate(comps):
        if stop["all"]:
            print(f"[PICK] skipping remaining {len(comps) - ti} smaller tree(s).")
            break
        _pick_one_tree(nodes, points, segments, comp, ti, len(comps),
                       vessels_all, name_counter, stop, hover=hover)

    vessels_idx = {nm: idxs for (nm, idxs, _c) in vessels_all}
    colors = {nm: col for (nm, _i, col) in vessels_all}
    return vessels_idx, colors


def _pick_one_tree(
    nodes: dict[int, tuple],
    points: dict[int, tuple],
    segments: list[dict[str, Any]],
    comp: list[int],
    tree_i: int,
    n_trees: int,
    vessels_all: list[tuple[str, set[int], str]],
    name_counter: list[int],
    stop: dict[str, bool],
    hover: bool = True,
    mode: str = "annotate",
    prune_acc: set[int] | None = None,
    protect: frozenset[int] = frozenset(),
    base_color_fn: Any = None,
    legend_entries: Any = None,
    root_acc: list[int | None] | None = None,
    preselect: frozenset[int] = frozenset(),
) -> None:
    """Render one tree's segments (rings + points) for interactive selection.

    ``mode="annotate"`` (default): append named vessels (as global segment
    indices) to ``vessels_all``. ``mode="prune"``: toggle a removal set, mark it
    red, and union it into ``prune_acc`` on close; segments in ``protect`` are
    rendered non-pickable (main vessels) so they can't be removed.
    ``mode="root"``: single-select -- one click moves the choice (lime) and the
    chosen global segment index is written to ``root_acc[0]`` on close (``None``
    if the selection was cleared). ``preselect`` seeds the initial selection.

    ``base_color_fn``: optional ``seg_idx -> color`` callback giving each
    non-protected segment its resting colour (e.g. by Strahler order / radius).
    Defaults to a flat grey. ``legend_entries``: optional list of
    ``(label, color)`` pairs drawn as a legend (e.g. the Strahler/radius key).
    Both are backward-compatible no-ops when ``None``.
    """
    import pyvista as pv
    from pyvista import _vtk

    is_prune = mode == "prune"
    is_root = mode == "root"

    pl = pv.Plotter(title=f"Tree {tree_i + 1}/{n_trees} - {len(comp)} segments")
    pl.set_background("white")
    base_color = (0.72, 0.72, 0.72)

    vis: dict[int, Any] = {}        # global seg idx -> visible rings/points actor
    seg_coords: dict[int, np.ndarray] = {}  # global seg idx -> centerline (mm)
    tag_pts: list[np.ndarray] = []   # rendered geometry points, tagged by seg idx
    tag_idx: list[np.ndarray] = []
    for i in comp:
        sp = prepare_segment_spline(segments[i], points, nodes)
        if sp is None:
            continue
        coords, radii = sp["coords"], sp["radii"]
        seg_coords[i] = np.asarray(coords, dtype=np.float64)
        # Visible + pickable: contour rings + centerline points (no solid surface).
        contour = _segment_contour_mesh(coords, radii, n_sides=16, ring_stride=1)
        # Tag the mesh so a pick resolves to the exact front-most segment hit.
        contour.field_data["seg_idx"] = np.array([i], dtype=np.int64)
        protected = (is_prune or is_root) and i in protect
        resting = base_color_fn(i) if base_color_fn is not None else base_color
        vis[i] = pl.add_mesh(
            contour,
            color=(0.55, 0.70, 0.95) if protected else resting,
            pickable=not protected,
            render_points_as_spheres=True, point_size=6.0, line_width=2.0)
        if protected:
            continue  # main vessels: visible but never selectable / hoverable
        pts = np.asarray(contour.points, dtype=np.float64)
        tag_pts.append(pts)
        tag_idx.append(np.full(len(pts), i, dtype=np.int64))

    pick_pts = np.vstack(tag_pts) if tag_pts else np.empty((0, 3))
    pick_tag = np.concatenate(tag_idx) if tag_idx else np.empty(0, dtype=np.int64)
    pick_kd = KDTree(pick_pts) if len(pick_pts) else None

    state = {"selected": set(preselect) & set(vis), "txt": None}

    def seg_color(i: int):
        resting = base_color_fn(i) if base_color_fn is not None else base_color
        if is_root:
            return "lime" if i in state["selected"] else resting
        if is_prune:
            return "red" if i in state["selected"] else resting
        for (_n, idxs, col) in vessels_all:
            if i in idxs:
                return col
        return "yellow" if i in state["selected"] else resting

    def recolor(i: int) -> None:
        a = vis.get(i)
        if a is not None:
            try:
                a.prop.color = seg_color(i)
            except Exception:
                pass

    def update_text() -> None:
        if is_root:
            sel = next(iter(state["selected"]), None)
            sel_s = (f"seg id {int(segments[sel]['id'])}" if sel is not None
                     else "none -- q keeps the auto pick")
            lines = [
                f"INLET / ROOT  TREE {tree_i + 1}/{n_trees}  ({len(comp)} segments)",
                "CLICK=set the inlet segment (lime) | DRAG=rotate",
                "c=clear | q=confirm | x=cancel (keep auto pick)",
                f"inlet: {sel_s}",
            ]
        elif is_prune:
            lines = [
                f"PRUNE  TREE {tree_i + 1}/{n_trees}  ({len(comp)} segments)",
                "CLICK=toggle removal (red) | DRAG=rotate | blue=main (locked)",
                "c=clear | q=next tree | x=stop",
                f"marked for removal: {len(state['selected'])} segment(s)",
            ]
        else:
            lines = [
                f"TREE {tree_i + 1}/{n_trees}  ({len(comp)} segments)",
                "CLICK=toggle (hover=green preview) | DRAG=rotate | n=name+commit",
                "c=clear | u=undo | q=next tree | x=stop",
                f"current selection: {len(state['selected'])} segment(s)",
            ]
            lines += [f"  {nm}: {len(idxs)}" for (nm, idxs, _c) in vessels_all]
        if state["txt"] is not None:
            pl.remove_actor(state["txt"])
        state["txt"] = pl.add_text("\n".join(lines), font_size=9, color="black")

    def resolve_seg(picker) -> int | None:
        # Prefer the exact front-most hit segment (its mesh's tagged seg_idx);
        # fall back to nearest tagged point to the pick position.
        ds = picker.GetDataSet()
        if ds is not None:
            try:
                fd = pv.wrap(ds).field_data
                if "seg_idx" in fd:
                    return int(fd["seg_idx"][0])
            except Exception:
                pass
        if pick_kd is not None:
            pos = np.asarray(picker.GetPickPosition(), dtype=np.float64)
            return int(pick_tag[pick_kd.query(pos)[1]])
        return None

    def toggle_seg(i: int | None) -> None:
        if i is None:
            return
        if (is_prune or is_root) and i in protect:
            return  # main vessel: never selectable for removal
        if is_root:
            # Single-select: a click moves the choice rather than adding to it.
            old_sel = set(state["selected"])
            state["selected"] = set() if i in old_sel else {i}
            for j in old_sel | state["selected"]:
                recolor(j)
            update_text()
            return
        if not is_prune and any(i in idxs for (_n, idxs, _c) in vessels_all):
            return  # already committed to a vessel
        if i in state["selected"]:
            state["selected"].discard(i)
        else:
            state["selected"].add(i)
        recolor(i)
        update_text()

    def _unique_name(base: str) -> str:
        existing = {nm for (nm, _i, _c) in vessels_all}
        if base not in existing:
            return base
        k = 2
        while f"{base}_{k}" in existing:
            k += 1
        return f"{base}_{k}"

    def commit_vessel() -> None:
        if not state["selected"]:
            print("[PICK] nothing selected")
            return
        base = PRESET_VESSEL_NAMES[name_counter[0] % len(PRESET_VESSEL_NAMES)]
        name = _unique_name(base)
        col = VESSEL_COLORS[len(vessels_all) % len(VESSEL_COLORS)]
        idxs = set(state["selected"])
        vessels_all.append((name, idxs, col))
        state["selected"] = set()
        name_counter[0] += 1
        for i in idxs:
            recolor(i)
        print(f"[PICK] committed vessel '{name}' ({len(idxs)} seg). "
              "Rename in the sidecar JSON if needed.")
        update_text()

    def clear_sel() -> None:
        old = set(state["selected"])
        state["selected"] = set()
        for i in old:
            recolor(i)
        update_text()

    def undo_vessel() -> None:
        # Undo the most recent vessel from THIS tree only.
        for j in range(len(vessels_all) - 1, -1, -1):
            nm, idxs, _c = vessels_all[j]
            if idxs & set(comp):
                vessels_all.pop(j)
                for i in idxs:
                    recolor(i)
                print(f"[PICK] removed vessel '{nm}'")
                update_text()
                return

    def stop_all() -> None:
        stop["all"] = True
        pl.close()

    # ── Mouse handling: distinguish a click from a drag-rotate, and hover. ──
    cell_picker = _vtk.vtkCellPicker()
    cell_picker.SetTolerance(0.008)
    press = {"xy": None}          # pixel of the last left-button press
    button = {"down": False}      # left button currently held (rotating)
    hover_state = {"seg": None, "xy": None}
    CLICK_TOL2 = 49               # (<=7 px)^2 counts as a click, not a drag

    def _seg_at(xy) -> int | None:
        try:
            cell_picker.Pick(float(xy[0]), float(xy[1]), 0, pl.renderer)
        except Exception:
            return None
        return resolve_seg(cell_picker)

    def on_press(*_a) -> None:
        button["down"] = True
        try:
            press["xy"] = pl.iren.get_event_position()
        except Exception:
            press["xy"] = None

    def on_release(*_a) -> None:
        button["down"] = False
        p = press["xy"]
        press["xy"] = None
        try:
            rel = pl.iren.get_event_position()
        except Exception:
            return
        if p is None:
            return
        dx, dy = rel[0] - p[0], rel[1] - p[1]
        if dx * dx + dy * dy <= CLICK_TOL2:   # a click, not a rotate
            toggle_seg(_seg_at(rel))

    def on_move(*_a) -> None:
        if button["down"]:        # don't hover while rotating
            return
        try:
            xy = pl.iren.get_event_position()
        except Exception:
            return
        lx = hover_state["xy"]
        if lx is not None and (xy[0] - lx[0]) ** 2 + (xy[1] - lx[1]) ** 2 < 16:
            return                # throttle: ignore < 4 px moves
        hover_state["xy"] = xy
        seg = _seg_at(xy)
        if seg == hover_state["seg"]:
            return
        hover_state["seg"] = seg
        try:
            if seg is None or seg not in seg_coords:
                pl.remove_actor("__hover__", reset_camera=False)
            else:
                cc = seg_coords[seg]
                conn = np.concatenate([[len(cc)], np.arange(len(cc), dtype=np.int64)])
                pl.add_mesh(pv.PolyData(cc, lines=conn), color="lime",
                            line_width=6.0, pickable=False, name="__hover__",
                            reset_camera=False)
            pl.render()
        except Exception:
            pass

    pl.iren.add_observer("LeftButtonPressEvent", on_press)
    pl.iren.add_observer("LeftButtonReleaseEvent", on_release)
    if hover:
        pl.iren.track_mouse_position(on_move)

    pl.add_key_event("c", clear_sel)
    pl.add_key_event("x", stop_all)
    if not (is_prune or is_root):
        pl.add_key_event("n", commit_vessel)
        pl.add_key_event("u", undo_vessel)
    update_text()
    if legend_entries:
        try:
            pl.add_legend(legend_entries, bcolor="white")
        except Exception as exc:  # legend is a nicety; never block picking
            print(f"[PICK][WARN] could not draw colour legend: {exc}")
    if is_root:
        print(f"[INLET] tree {tree_i + 1}/{n_trees}: left-click the segment the "
              "inflow enters through (lime = chosen); 'c' clears, 'q' confirms, "
              "'x' cancels (keeps the auto pick).")
    elif is_prune:
        print(f"[PRUNE] tree {tree_i + 1}/{n_trees}: left-click a stub/side "
              "branch to mark for removal (red), 'q' next tree, 'x' stop. "
              "Blue = main vessel (locked).")
    else:
        print(f"[PICK] tree {tree_i + 1}/{n_trees}: left-click a vessel to toggle "
              "(drag = rotate), 'n' names a vessel, 'q' next tree, 'x' stop.")
    pl.show()

    if is_root:
        if root_acc is not None and not stop["all"]:
            root_acc[0] = next(iter(state["selected"]), None)
    elif is_prune:
        if prune_acc is not None:
            prune_acc |= state["selected"]
    elif state["selected"]:
        commit_vessel()


# ── Inlet / root picker ─────────────────────────────────────────────────────────


def run_root_picker(
    nodes: dict[int, tuple],
    points: dict[int, tuple],
    segments: list[dict[str, Any]],
    candidates: set[int] | None = None,
    auto: int | None = None,
    hover: bool = True,
    label: str = "",
) -> int | None:
    """Pick one tree's inlet/root segment in 3D; return its index or ``None``.

    ``segments`` is a single graph's segment list (indices are local to it, as in
    :func:`coronary_sdf.flow_fractions._resolve_root_pref`). ``candidates`` — the
    max-Strahler tie set — are drawn orange against the grey rest, ``auto`` (the
    automatic pick) is seeded as the current selection so pressing 'q' straight
    away keeps it. ``None`` comes back when the user clears the selection or
    cancels with 'x'.
    """
    n = len(segments)
    if n == 0:
        return None
    cands = set(candidates or ())
    resting_other = (0.72, 0.72, 0.72)
    resting_cand = (0.95, 0.60, 0.15)

    def base_color_fn(i: int):
        return resting_cand if i in cands else resting_other

    legend = [("inlet / root (chosen)", "lime")]
    if cands:
        legend.append(("max-Strahler candidate", resting_cand))
    legend.append(("other segment", resting_other))

    root_acc: list[int | None] = [auto]
    stop = {"all": False}
    if label:
        print(f"[INLET] 3D picker: {label}")
    _pick_one_tree(
        nodes, points, segments, list(range(n)), 0, 1,
        vessels_all=[], name_counter=[0], stop=stop, hover=hover,
        mode="root", root_acc=root_acc,
        preselect=frozenset({auto}) if auto is not None else frozenset(),
        base_color_fn=base_color_fn, legend_entries=legend)
    if stop["all"]:
        print("[INLET] picker cancelled ('x') — keeping the automatic pick.")
        return auto
    return root_acc[0]


# ── Manual prune picker ─────────────────────────────────────────────────────────


def run_prune_picker(
    nodes: dict[int, tuple],
    points: dict[int, tuple],
    segments: list[dict[str, Any]],
    vessel_points: dict[str, set[int]],
    hover: bool = True,
    base_color_fn: Any = None,
    legend_entries: Any = None,
) -> set[int]:
    """Open the picker per tree; return the union of global segment indices the
    user marked (red) for removal. Annotated main-vessel segments are shown but
    non-pickable so they can't be removed. Left-click toggles, 'q' next tree,
    'x' stop, 'c' clear.

    ``base_color_fn`` / ``legend_entries`` are forwarded to
    :func:`_pick_one_tree` to colour non-protected segments by a scalar (e.g.
    Strahler order or radius) and draw the matching legend."""
    protect = frozenset(seg_vessel_map(segments, vessel_points))
    comps = sorted((list(c) for c in find_connected_components(segments)),
                   key=len, reverse=True)
    print(f"[PRUNE] {len(comps)} tree(s): sizes {[len(c) for c in comps]}")
    acc: set[int] = set()
    stop = {"all": False}
    for ti, comp in enumerate(comps):
        if stop["all"]:
            print(f"[PRUNE] skipping remaining {len(comps) - ti} smaller tree(s).")
            break
        _pick_one_tree(nodes, points, segments, comp, ti, len(comps),
                       vessels_all=[], name_counter=[0], stop=stop,
                       hover=hover, mode="prune", prune_acc=acc, protect=protect,
                       base_color_fn=base_color_fn, legend_entries=legend_entries)
    return acc


def apply_manual_prune(
    nodes: dict[int, tuple],
    points: dict[int, tuple],
    segments: list[dict[str, Any]],
    vessel_points: dict[str, set[int]],
    remove_idx: set[int],
    skip_points: int | None = None,
    n_average: int | None = None,
) -> tuple[dict[int, tuple], dict[int, tuple], list[dict[str, Any]], list[dict[str, Any]]]:
    """Remove user-selected segments and their whole downstream subtrees.

    Mirrors :func:`prune_by_radius_ratio`'s removal: expand each selected index
    to its subtree, drop annotated main-vessel segments from the removal set
    (belt-and-braces), then contract degree-2 nodes and filter orphaned points.
    Returns ``(nodes, points, segments, removed)`` where ``removed`` is one
    record per selected take-off ``{seg_id, vessel, radius_mm, n_subtree_removed,
    removed_seg_ids}`` (``removed_seg_ids`` lists every base seg id in that
    take-off's subtree, for logging).
    """
    segments = copy_segments(segments)
    nodes = recompute_node_degrees(nodes, segments)
    parent_idx = build_directed_topology(
        segments, build_node_to_segs(segments))["parent_seg_idx"]
    children = children_of(parent_idx)
    svm = seg_vessel_map(segments, vessel_points)
    main_set = set(svm)
    descends = build_descent_map(segments, parent_idx, svm)

    remove: set[int] = set()
    removed: list[dict[str, Any]] = []
    for c in sorted(remove_idx):
        if c < 0 or c >= len(segments) or c in main_set or c in remove:
            continue
        subtree = _collect_subtree(c, children) - main_set
        if not subtree:
            continue
        v = descends.get(c)
        r_takeoff = None
        sh = shared_node_with_parent(c, segments, parent_idx)
        if sh is not None:
            r_takeoff, _ = branch_radius_mm(
                segments[c], sh, points, nodes,
                skip_points=skip_points, n_average=n_average)
        removed.append({
            "seg_id": int(segments[c]["id"]),
            "vessel": v if v is not None else "",
            "radius_mm": None if r_takeoff is None else round(float(r_takeoff), 5),
            "n_subtree_removed": len(subtree),
            "removed_seg_ids": sorted(int(segments[j]["id"]) for j in subtree),
        })
        remove |= subtree

    if remove:
        kept = [s for i, s in enumerate(segments) if i not in remove]
        nodes, segments, _ = merge_degree2_segments(nodes, points, kept)
    nodes = recompute_node_degrees(nodes, segments)
    points = filter_points(points, segments)
    print(f"  [MANUAL] removed {len(remove)} segment(s) in {len(removed)} "
          f"selection(s) -> {len(segments)} remain")
    return nodes, points, segments, removed


# ── Sidebranch radius histograms ────────────────────────────────────────────────


def plot_sidebranch_radius_histograms(
    nodes: dict[int, tuple],
    points: dict[int, tuple],
    segments: list[dict[str, Any]],
    vessel_points: dict[str, set[int]],
    ostia: dict[str, dict[str, Any]],
    denoms: list[float],
    out_dir: Path,
    skip_points: int | None = None,
    n_average: int | None = None,
) -> None:
    """Histogram the take-off radii of every attributed sidebranch, with the
    radius-ratio cutoffs (``1/d`` for each ``d`` in ``denoms``) overlaid.

    The universe is built once on the base tree (ratio-independent): every
    non-main segment that descends from an annotated main vessel, measured at
    its shared node with its parent via :func:`branch_radius_mm` (same skip /
    average as the prune). This is a per-branch radius test, not the cascade --
    a branch above its own cutoff whose parent is pruned still disappears in the
    real model; that cascade is captured by ``pruned_branches.csv``.

    Writes ``sidebranch_radii.csv`` plus three figures into ``out_dir``:
    ``sidebranch_radii_normalized.png`` (pooled r/R_ostial, global 1/d lines),
    ``sidebranch_radii_per_ratio.png`` (one panel per cutoff, kept vs pruned),
    and ``sidebranch_radii_by_vessel.png`` (absolute mm per main vessel, with
    that vessel's per-ratio threshold lines).
    """
    try:
        import matplotlib
        matplotlib.use("Agg")          # file output, no display needed
        import matplotlib.pyplot as plt
    except Exception as exc:           # pragma: no cover - env dependent
        print(f"  [HIST] matplotlib unavailable ({exc}); skipping radius histograms")
        return

    import csv
    import math

    parent_idx = build_directed_topology(
        segments, build_node_to_segs(segments))["parent_seg_idx"]
    svm = seg_vessel_map(segments, vessel_points)      # seg_idx -> main vessel
    main_set = set(svm)
    descends = build_descent_map(segments, parent_idx, svm)

    records: list[dict[str, Any]] = []
    for c in range(len(segments)):
        if c in main_set:
            continue
        v = descends.get(c)            # which main vessel it descends from
        if v is None:                  # unattributed: cutoffs undefined -> skip
            continue
        sh = shared_node_with_parent(c, segments, parent_idx)
        if sh is None:
            continue
        r_takeoff, _ = branch_radius_mm(
            segments[c], sh, points, nodes,
            skip_points=skip_points, n_average=n_average)
        R = float(ostia[v]["radius_mm"]) if v in ostia else 0.0
        if r_takeoff is None or r_takeoff <= 0 or R <= 0:
            continue
        records.append({
            "seg_id": int(segments[c]["id"]), "vessel": v,
            "r_mm": float(r_takeoff), "R_ostial_mm": R,
            "r_norm": float(r_takeoff) / R,
        })

    if not records:
        print("  [HIST] no attributed sidebranches found; skipping radius histograms")
        return

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Dump the universe for reproducibility (matches the tool's CSV convention).
    with open(out_dir / "sidebranch_radii.csv", "w", newline="") as fh:
        w = csv.DictWriter(
            fh, fieldnames=["seg_id", "vessel", "r_mm", "R_ostial_mm", "r_norm"])
        w.writeheader()
        w.writerows(records)

    denoms_sorted = sorted({float(d) for d in denoms})
    cutoffs = [(d, 1.0 / d) for d in denoms_sorted]          # (denom, normalized line)
    r_norm = np.array([rec["r_norm"] for rec in records], dtype=float)
    n_total = len(records)
    cut_colors = (
        plt.get_cmap("viridis")(np.linspace(0.0, 0.85, len(cutoffs)))
        if cutoffs else []
    )

    # ── Figure 1: pooled normalized overlay ──────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)
    ax.hist(r_norm, bins=40, color="#4477aa", edgecolor="white", alpha=0.85)
    for (d, line), col in zip(cutoffs, cut_colors):
        ax.axvline(line, color=col, ls="--", lw=1.6,
                   label=f"d={d:g} (≥{line:.2f})")
    ax.set_xlabel("take-off radius / ostial radius  (r_norm)")
    ax.set_ylabel("sidebranch count")
    ax.set_title(f"Sidebranch take-off radii (normalized), N={n_total}")
    ax.legend(title="ratio cutoff", fontsize=8, loc="upper left",
              bbox_to_anchor=(1.02, 1.0), borderaxespad=0.0)
    fig.savefig(out_dir / "sidebranch_radii_normalized.png", dpi=150,
                bbox_inches="tight")
    plt.close(fig)

    # ── Figure 2: one panel per ratio cutoff (kept vs pruned) ────────────────
    if cutoffs:
        ncols = min(len(cutoffs), int(math.ceil(math.sqrt(len(cutoffs)))))
        nrows = int(math.ceil(len(cutoffs) / ncols))
        bins = np.linspace(0.0, float(r_norm.max()) * 1.05 + 1e-9, 41)
        fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4.0 * nrows),
                                 squeeze=False, sharex=True, constrained_layout=True)
        flat = axes.flatten()
        legend_handles = None
        for ax, (d, line) in zip(flat, cutoffs):
            pruned = r_norm[r_norm < line]
            kept = r_norm[r_norm >= line]
            _, _, patches = ax.hist([pruned, kept], bins=bins, stacked=True,
                                    color=["#cc6677", "#4477aa"], edgecolor="white",
                                    label=["pruned", "kept"])
            if legend_handles is None:
                legend_handles = [patches[0][0], patches[1][0]]
            ax.axvline(line, color="black", ls="--", lw=1.4)
            pct = 100.0 * len(pruned) / n_total
            ax.set_title(f"d={d:g}  (cut <{line:.3f})\n"
                         f"pruned {len(pruned)}/{n_total} ({pct:.0f}%)", fontsize=10)
            ax.set_xlabel("r_norm")
            ax.set_ylabel("count")
        for ax in flat[len(cutoffs):]:
            ax.set_visible(False)
        if legend_handles is not None:
            fig.legend(legend_handles, ["pruned", "kept"], loc="outside upper right")
        fig.suptitle("Sidebranch radii per ratio cutoff (normalized)", fontsize=12)
        fig.savefig(out_dir / "sidebranch_radii_per_ratio.png", dpi=150)
        plt.close(fig)

    # ── Figure 3: per-vessel absolute radii (mm) ─────────────────────────────
    by_vessel: dict[str, list[float]] = {}
    for rec in records:
        by_vessel.setdefault(rec["vessel"], []).append(rec["r_mm"])
    vessels = sorted(by_vessel)
    ncols = min(len(vessels), int(math.ceil(math.sqrt(len(vessels))))) or 1
    nrows = int(math.ceil(len(vessels) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4.0 * nrows),
                             squeeze=False, constrained_layout=True)
    flat = axes.flatten()
    for ax, v in zip(flat, vessels):
        rs = np.array(by_vessel[v], dtype=float)
        R = float(ostia[v]["radius_mm"]) if v in ostia else 0.0
        ax.hist(rs, bins=30, color="#117733", edgecolor="white", alpha=0.85)
        for (d, _), col in zip(cutoffs, cut_colors):
            ax.axvline(R / d, color=col, ls="--", lw=1.4)
        ax.set_title(f"{v}  (R_ostial={R:.3f}mm, n={len(rs)})", fontsize=10)
        ax.set_xlabel("take-off radius (mm)")
        ax.set_ylabel("count")
    for ax in flat[len(vessels):]:
        ax.set_visible(False)
    # One shared legend mapping cutoff colour -> denominator (per-vessel mm
    # thresholds are read off each panel's x-axis where the dashed lines sit).
    if cutoffs:
        from matplotlib.lines import Line2D
        cut_legend = [Line2D([0], [0], color=col, ls="--", lw=1.4, label=f"d={d:g}")
                      for (d, _), col in zip(cutoffs, cut_colors)]
        fig.legend(handles=cut_legend, title="ratio cutoff", loc="outside upper right")
    fig.suptitle("Sidebranch take-off radii per main vessel (absolute)", fontsize=12)
    fig.savefig(out_dir / "sidebranch_radii_by_vessel.png", dpi=150)
    plt.close(fig)

    print(f"  [HIST] wrote sidebranch_radii.csv + 3 figures to {out_dir} "
          f"({n_total} sidebranches, {len(vessels)} vessels, {len(cutoffs)} cutoffs)")


# ── Orchestration ──────────────────────────────────────────────────────────────


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Epicardial annotation + side-branch pruning model series")
    ap.add_argument("--xml", default=config.INPUT_XML, help="input Amira .am.xml")
    ap.add_argument("--out", default="epicardial_models", help="output directory")
    ap.add_argument("--sidecar", default=None,
                    help="annotation JSON (default <out>/epicardial.json)")
    ap.add_argument("--ratios", default="2,5,10",
                    help="comma-separated denominators; threshold = R_ostial / d")
    ap.add_argument("--pick", action="store_true", help="force the picker to re-run")
    ap.add_argument("--no-pipeline", action="store_true",
                    help="write XML + flow only, skip surface generation")
    ap.add_argument("--prune-unattributed", action="store_true",
                    help="also drop trees/branches with no annotated ancestor")
    ap.add_argument("--frac", type=float, default=0.8,
                    help="containment fraction threshold for swallowed-leaf prune")
    ap.add_argument("--ostium-skip", type=int, default=None,
                    help="contours to skip at the ostium when measuring a branch "
                         "radius (default: flow_fractions.BIF_SKIP_POINTS)")
    ap.add_argument("--radius-navg", type=int, default=None,
                    help="downstream contours to average for a branch radius "
                         "(default: flow_fractions.FRAMES_DOWNSTREAM)")
    ap.add_argument("--no-hover", action="store_true",
                    help="disable the picker's hover preview highlight")
    ap.add_argument("--no-hist", action="store_true",
                    help="skip sidebranch radius histogram diagnostics")
    ap.add_argument("--manual-prune", action="store_true",
                    help="after each ratio prune, open the picker to manually "
                         "select extra segments (and their subtree) to remove")
    ap.add_argument("--hist-only", action="store_true",
                    help="produce only the sidebranch radius histograms (+ CSV), then exit")
    args = ap.parse_args(argv)

    xml_path = Path(args.xml)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    sidecar = Path(args.sidecar) if args.sidecar else out / "epicardial.json"
    denoms = [float(x) for x in args.ratios.split(",") if x.strip()]

    nodes, points, segments = parse_xml(xml_path)
    nodes, points, segments = preprocess_topology(nodes, points, segments)
    seg_keys = build_segment_keys(segments, points)

    vessels_idx, _colors = load_or_create_annotation(
        sidecar, xml_path, nodes, points, segments, seg_keys,
        force_pick=args.pick, hover=not args.no_hover)

    vessel_points = {
        v: {pid for i in idxs for pid in segments[i]["point_ids"]}
        for v, idxs in vessels_idx.items()
    }
    ostia = compute_vessel_ostia(vessels_idx, nodes, points, segments,
                                 skip_points=args.ostium_skip,
                                 n_average=args.radius_navg)

    # Per-vessel radius stats (invariant to pruning -> computed once on the base).
    radius_stats = compute_vessel_radius_stats(
        vessels_idx, nodes, points, segments, skip_points=args.ostium_skip)
    radius_fields = [
        "vessel", "total_n", "total_min_mm", "total_mean_mm", "total_max_mm",
        "prox_n", "prox_min_mm", "prox_mean_mm", "prox_max_mm",
        "dist_n", "dist_min_mm", "dist_mean_mm", "dist_max_mm"]

    # Sidebranch radius distribution across all ratio cutoffs (ratio-independent;
    # computed once on the base tree, so it runs even with --no-pipeline).
    if args.hist_only or not args.no_hist:
        plot_sidebranch_radius_histograms(
            nodes, points, segments, vessel_points, ostia, denoms, out,
            skip_points=args.ostium_skip, n_average=args.radius_navg)
    if args.hist_only:
        print(f"[DONE] histograms only -> {out}")
        return 0

    if not args.no_pipeline:
        # Keep the surface in step with the flow tree: don't let run_pipeline's
        # own nub-prune drop thin distal vessels we intentionally kept.
        config.PRUNE_SHORT_TERMINAL_NUBS = False
        config.MIN_TERMINAL_LENGTH_MM = 0.0
        print("[CFG] disabled short-terminal-nub pruning for surface runs "
              "(surface == flow tree).")

    import csv
    pruned_fields = ["ratio_denom", "seg_id", "vessel", "radius_mm",
                     "threshold_mm", "n_subtree_removed"]
    all_pruned: list[dict[str, Any]] = []

    summary_rows: list[dict[str, Any]] = []
    for denom in denoms:
        ratio = 1.0 / denom
        tag = f"ratio_{denom:g}"
        ratio_dir = out / tag
        ratio_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n{'=' * 60}\n[MODEL] {tag} (threshold = R_ostial / {denom:g})\n{'=' * 60}")

        p_points = copy_points(points)
        p_nodes, p_points, p_segs, pruned = prune_by_radius_ratio(
            nodes, p_points, segments, vessel_points, ostia, ratio,
            prune_unattributed=args.prune_unattributed,
            skip_points=args.ostium_skip, n_average=args.radius_navg)
        p_nodes, p_points, p_segs = prune_contained_leaves(
            p_nodes, p_points, p_segs, vessel_points, frac_threshold=args.frac)

        # Optional human-in-the-loop pass: pick extra stubs / swallowed side
        # branches the automatic prunes missed, on this ratio's pruned tree.
        if args.manual_prune:
            sel = run_prune_picker(p_nodes, p_points, p_segs, vessel_points,
                                   hover=not args.no_hover)
            if sel:
                p_nodes, p_points, p_segs, manual_recs = apply_manual_prune(
                    p_nodes, p_points, p_segs, vessel_points, sel,
                    skip_points=args.ostium_skip, n_average=args.radius_navg)
                # Re-run the auto containment pass to clean newly-exposed stubs.
                p_nodes, p_points, p_segs = prune_contained_leaves(
                    p_nodes, p_points, p_segs, vessel_points, frac_threshold=args.frac)
                with open(ratio_dir / "manual_pruned.csv", "w", newline="") as fh:
                    mw = csv.DictWriter(
                        fh, fieldnames=["ratio_denom", "seg_id", "vessel",
                                        "radius_mm", "n_subtree_removed"])
                    mw.writeheader()
                    mw.writerows({"ratio_denom": denom, **r} for r in manual_recs)

        # Export the pruned take-off branches (id + measured radius) for this ratio.
        for rec in pruned:
            rec = {"ratio_denom": denom, **rec}
            all_pruned.append(rec)
        with open(ratio_dir / "pruned_branches.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=pruned_fields)
            w.writeheader()
            w.writerows({k: r.get(k) for k in pruned_fields}
                        for r in all_pruned if r["ratio_denom"] == denom)

        xml_out = ratio_dir / "model.am.xml"
        write_amira_xml(p_nodes, p_points, p_segs, xml_out)

        if not args.no_pipeline:
            from .pipeline import run_pipeline
            run_pipeline(xml_out, ratio_dir / "surface")

        flow = compute_main_vessel_flow(p_nodes, p_points, p_segs, vessel_points, ostia)
        (ratio_dir / "main_vessel_flow.json").write_text(json.dumps(flow, indent=2))
        for v, f in flow.items():
            summary_rows.append({
                "ratio_denom": denom, "vessel": v,
                "ostial_flow_fraction": f["ostial_flow_fraction"],
                "distal_flow_fraction": f["distal_flow_fraction"],
                "retained_fraction": f["retained_fraction"],
                "leaf_flow_sum": f["leaf_flow_sum"],
                "n_segments": len(p_segs),
            })
        print("[MODEL] flow (ostial -> distal): " + ", ".join(
            f"{v}: {f['ostial_flow_fraction']:.4f} -> {f['distal_flow_fraction']:.4f}"
            for v, f in flow.items()))

    with open(out / "summary.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=[
            "ratio_denom", "vessel", "ostial_flow_fraction", "distal_flow_fraction",
            "retained_fraction", "leaf_flow_sum", "n_segments"])
        w.writeheader()
        w.writerows(summary_rows)
    with open(out / "pruned_branches.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=pruned_fields)
        w.writeheader()
        w.writerows({k: r.get(k) for k in pruned_fields} for r in all_pruned)
    with open(out / "vessel_radius_stats.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=radius_fields)
        w.writeheader()
        w.writerows(radius_stats)
    print(f"\n[DONE] wrote {out / 'summary.csv'} ({len(summary_rows)} rows), "
          f"{out / 'pruned_branches.csv'} ({len(all_pruned)} pruned branches), and "
          f"{out / 'vessel_radius_stats.csv'} ({len(radius_stats)} vessels)")
    return 0


__all__ = [
    "write_amira_xml",
    "build_segment_keys",
    "prune_by_radius_ratio",
    "prune_contained_leaves",
    "compute_vessel_ostia",
    "compute_vessel_radius_stats",
    "compute_main_vessel_flow",
    "plot_sidebranch_radius_histograms",
    "run_picker",
    "run_prune_picker",
    "apply_manual_prune",
    "main",
]


if __name__ == "__main__":
    raise SystemExit(main())
