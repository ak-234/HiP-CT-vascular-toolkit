"""Radially-resolved wall-shear-stress (WSS) post-processing along annotated
main epicardial vessels.

Consumes a CFX/CFD-Post wall-node CSV (X, Y, Z + Wall Shear magnitude, Pa) plus
the pruned coronary model produced by :mod:`coronary_sdf.epicardial_annotation`
(``model.am.xml`` + ``epicardial.json`` sidecar) and reports, at stations spaced
every *N* mm of arc length along each main vessel:

* the **max** 90 deg arc sector (worst WSS) and its angular position,
* the **min** 90 deg arc sector (shielded WSS) and its angular position,
* the ring-mean WSS and supporting counts.

A 90 deg window is swept around the lumen circumference at each station; the WSS
magnitude is averaged inside the window at each rotation, bounding the
circumferential WSS distribution. Bifurcation regions (within ``BIF_SKIP_POINTS``
contours of a degree>=3 node) are excluded so junction flow does not contaminate
the main-vessel measurement.

Usage::

    python -m coronary_sdf.wss_postprocess \
        --xml <base.am.xml> --model-dir epicardial_models/ratio_2 \
        --wss-csv cfx_wall.csv --out wss_out

The numeric core (:func:`evaluate_vessel`, :func:`sweep_station`) is file-IO-free
so it can be unit-tested on synthetic geometry (see ``_test_wss_postprocess.py``).
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial import KDTree

from .parse_amira import parse_xml
from .splines import prepare_segment_spline, compute_frenet_frame
from .flow_fractions import preprocess_topology, BIF_SKIP_POINTS
from .topology import build_directed_topology
from .epicardial_annotation import (
    build_node_to_segs,
    build_segment_keys,
    recompute_node_degrees,
    resolve_annotation_to_indices,
    seg_depths,
    seg_vessel_map,
    prox_node,
)

CSV_FIELDS = [
    "vessel", "s_mm", "x", "y", "z", "local_radius_mm", "n_wall_pts",
    "ring_mean_wss", "wss_max_window", "angle_max_deg",
    "wss_min_window", "angle_min_deg",
]

# Per-wall-point export (one row per CFD wall node, tagged by owning interval
# segment). ``seg_idx == -1`` / empty ``vessel`` flags an unassigned node.
POINT_CSV_FIELDS = ["vessel", "seg_idx", "x", "y", "z", "wss", "s_mm"]


# ── CFX / CFD-Post CSV reader ──────────────────────────────────────────────────


_UNIT_TO_MM = {"m": 1000.0, "metre": 1000.0, "meter": 1000.0,
               "cm": 10.0, "mm": 1.0, "um": 1e-3, "micron": 1e-3}


def _clean_col(name: str) -> str:
    """Lower-cased column name with the trailing ``[ unit ]`` stripped."""
    name = name.strip().strip('"').strip()
    if "[" in name:
        name = name[: name.index("[")]
    return name.strip().lower()


def _coord_unit(name: str) -> str | None:
    """Bracketed unit token from a coord header, e.g. ``'X [ m ]'`` -> ``'m'``.

    Returns ``None`` when the header carries no ``[ unit ]`` annotation."""
    name = name.strip().strip('"')
    if "[" in name and "]" in name:
        return name[name.index("[") + 1: name.index("]")].strip().lower() or None
    return None


def read_wss_csv(
    path: str | Path, wss_col: str | None = None
) -> tuple[np.ndarray, np.ndarray, str | None]:
    """Parse a CFD-Post wall-node CSV into ``(points (N,3), wss (N,), unit)``.

    Tolerant of CFD-Post's section preamble: scans for the first row that names
    X/Y/Z coordinate columns *and* a wall-shear column, then reads the numeric
    rows beneath it until a blank / non-numeric line. ``wss_col`` overrides the
    auto-detected shear column (matched on the cleaned, unit-stripped name).

    ``unit`` is the bracketed coordinate unit from the X header (e.g. ``'m'`` for
    ``X [ m ]``), or ``None`` if unannotated; the caller uses it to rescale to mm.
    """
    path = Path(path)
    rows = list(csv.reader(path.read_text().splitlines()))

    want = None if wss_col is None else _clean_col(wss_col)
    header_i = ix = iy = iz = iw = None
    for ri, row in enumerate(rows):
        if not row:
            continue
        cleaned = [_clean_col(c) for c in row]

        def _find(pred):
            for ci, c in enumerate(cleaned):
                if pred(c):
                    return ci
            return None

        cx = _find(lambda c: c in ("x", "x coord", "x coordinate"))
        cy = _find(lambda c: c in ("y", "y coord", "y coordinate"))
        cz = _find(lambda c: c in ("z", "z coord", "z coordinate"))
        if want is not None:
            cw = _find(lambda c: c == want)
        else:
            cw = _find(lambda c: ("wall shear" in c or c == "wss" or
                                  ("shear" in c and "x" not in c.split()
                                   and "y" not in c.split() and "z" not in c.split())))
        if None not in (cx, cy, cz, cw):
            header_i, ix, iy, iz, iw = ri, cx, cy, cz, cw
            break

    if header_i is None:
        raise SystemExit(
            f"[WSS][ERROR] could not find X/Y/Z + Wall Shear columns in {path}. "
            "Pass --wss-col with the exact CFD-Post column name.")

    pts: list[tuple[float, float, float]] = []
    wss: list[float] = []
    for row in rows[header_i + 1:]:
        if not row or all(not c.strip() for c in row):
            break
        try:
            x = float(row[ix]); y = float(row[iy]); z = float(row[iz])
            w = float(row[iw])
        except (ValueError, IndexError):
            continue  # skip stray non-numeric lines, keep reading
        pts.append((x, y, z))
        wss.append(w)

    if not pts:
        raise SystemExit(f"[WSS][ERROR] no numeric data rows parsed from {path}.")
    unit = _coord_unit(rows[header_i][ix])
    print(f"[WSS] read {len(pts)} wall nodes from {path}"
          + (f" (coords in [{unit}])" if unit else ""))
    return (np.asarray(pts, dtype=np.float64),
            np.asarray(wss, dtype=np.float64), unit)


def check_alignment(wall_pts: np.ndarray, cl_pts: np.ndarray) -> None:
    """Warn if the wall cloud and centreline extents differ by ~1000x (CFX in
    metres) or otherwise look mis-scaled. Mirrors flow_fractions' [ALIGN] check."""
    def _span(a: np.ndarray) -> float:
        return float(np.linalg.norm(a.max(axis=0) - a.min(axis=0)))

    sw, sc = _span(wall_pts), _span(cl_pts)
    if sc <= 0 or sw <= 0:
        return
    ratio = sc / sw
    if 700 < ratio < 1400:
        print("[ALIGN][WARN] wall cloud looks ~1000x smaller than the centreline "
              "— CFX is probably in METRES; pass --coord-scale 1000.")
    elif not (0.5 < ratio < 2.0):
        print(f"[ALIGN][WARN] wall-cloud / centreline extent ratio = {ratio:.4g}; "
              "check --coord-scale / --coord-offset (centreline is in mm).")


def resolve_coord_scale(
    unit: str | None,
    wall_pts: np.ndarray,
    cl_pts: np.ndarray,
    user_scale: float | None = None,
) -> tuple[float, str]:
    """Resolve the factor that brings the wall cloud into mm. Returns
    ``(scale, reason)``.

    Priority: an explicit ``user_scale`` always wins; otherwise the bracketed CSV
    coordinate ``unit`` (e.g. ``'m'`` -> x1000); otherwise infer from the
    wall/centreline extent ratio (~1000x smaller => CFX exported in metres),
    falling back to no rescale (assume mm)."""
    if user_scale is not None:
        return float(user_scale), f"--coord-scale {user_scale}"
    if unit and unit in _UNIT_TO_MM:
        return _UNIT_TO_MM[unit], f"header unit [{unit}]"

    def _span(a: np.ndarray) -> float:
        return float(np.linalg.norm(a.max(axis=0) - a.min(axis=0)))

    sw, sc = _span(wall_pts), _span(cl_pts)
    if sw > 0 and sc > 0 and 700 < sc / sw < 1400:
        return 1000.0, f"extent ratio {sc / sw:.0f}x ~ metres"
    return 1.0, "assumed mm"


# ── Vessel centreline construction (pruned model + sidecar) ────────────────────


def _segment_bif_mask(
    coords: np.ndarray, node1: int, node2: int, nodes: dict[int, tuple], bif_skip: int
) -> np.ndarray:
    """Boolean mask over ``coords`` flagging contours within ``bif_skip`` of an
    end whose node has degree>=3. coords[0] is the node1 end (Amira point order)."""
    n = len(coords)
    mask = np.zeros(n, dtype=bool)
    k = min(bif_skip, n)

    def deg(nid: int) -> int:
        return nodes[nid][3] if nid in nodes else 0

    if deg(node1) >= 3:
        mask[:k] = True
    if deg(node2) >= 3:
        mask[n - k:] = True
    return mask


def build_vessel_centrelines(
    base_xml: str | Path,
    model_xml: str | Path | None,
    sidecar: str | Path,
    bif_skip: int = BIF_SKIP_POINTS,
    return_competitors: bool = False,
) -> dict[str, dict[str, np.ndarray]] | tuple[dict[str, dict[str, np.ndarray]], np.ndarray]:
    """Return ``{vessel: {coords (M,3) mm, radii (M,), bif (M,) bool}}``.

    Vessel membership is recovered by point-id set membership against the sidecar
    (same scheme as ``compute_main_vessel_flow``). The centrelines are built on
    the *pruned* model geometry when ``model_xml`` is given; pass ``model_xml=None``
    to build them on the un-pruned base tree instead (no ratio model required).
    Each vessel's segments are ordered proximal->distal and stitched into one
    polyline; ``bif`` flags bifurcation-zone points.
    """
    # 1. Resolve sidecar -> vessel point-id sets on the base preprocessed tree.
    b_nodes, b_points, b_segments = parse_xml(base_xml)
    b_nodes, b_points, b_segments = preprocess_topology(b_nodes, b_points, b_segments)
    seg_keys = build_segment_keys(b_segments, b_points)
    data = json.loads(Path(sidecar).read_text())
    vessels_idx, _colors, unmatched = resolve_annotation_to_indices(
        data, b_segments, seg_keys)
    if unmatched:
        print(f"[WSS][WARN] {unmatched} sidecar key(s) did not resolve against the "
              "base topology; vessel membership may be partial.")
    vessel_points = {
        v: {pid for i in idxs for pid in b_segments[i]["point_ids"]}
        for v, idxs in vessels_idx.items()
    }

    # 2. Geometry source: the pruned (meshed) model, or the un-pruned base tree.
    if model_xml is not None:
        nodes, points, segments = parse_xml(model_xml)
    else:
        nodes, points, segments = b_nodes, b_points, b_segments
    return _centrelines_from_geometry(
        vessel_points, nodes, points, segments, bif_skip,
        return_competitors=return_competitors)


def _centrelines_from_geometry(
    vessel_points: dict[str, set[int]],
    nodes: dict[int, tuple],
    points: dict[int, tuple],
    segments: list[dict[str, Any]],
    bif_skip: int = BIF_SKIP_POINTS,
    return_competitors: bool = False,
) -> dict[str, dict[str, np.ndarray]] | tuple[dict[str, dict[str, np.ndarray]], np.ndarray]:
    """Stitch per-vessel centreline polylines from resolved point-id sets and a
    parsed geometry. Source-agnostic: works on the pruned model or the base tree.

    Maps ``segments`` to vessels (majority point-id membership), orders each
    vessel's segments proximal->distal, and stitches their splines into one
    polyline with a bifurcation-zone mask. Returns ``{vessel: {coords, radii, bif}}``.

    When ``return_competitors`` is set, also returns a ``(K,3)`` array of centreline
    points from every segment **not** mapped to an annotated vessel (the side
    branches) — used to Voronoi-exclude side-branch wall points downstream."""
    nodes = recompute_node_degrees(nodes, segments)
    svm = seg_vessel_map(segments, vessel_points)            # seg_idx -> vessel
    members: dict[str, list[int]] = {}
    for i, v in svm.items():
        members.setdefault(v, []).append(i)

    parent_idx = build_directed_topology(
        segments, build_node_to_segs(segments))["parent_seg_idx"]
    depth = seg_depths(parent_idx)

    out: dict[str, dict[str, np.ndarray]] = {}
    for v, idxs in members.items():
        chain = _order_vessel_segments(idxs, segments, nodes, parent_idx, depth)
        if not chain:
            continue
        all_c: list[np.ndarray] = []
        all_r: list[np.ndarray] = []
        all_b: list[np.ndarray] = []
        for seg_i, prox_first in chain:
            seg = segments[seg_i]
            sp = prepare_segment_spline(seg, points, nodes)
            if sp is None:
                continue
            coords = np.asarray(sp["coords"], dtype=np.float64)
            radii = np.asarray(sp["radii"], dtype=np.float64)
            bif = _segment_bif_mask(coords, seg["node1"], seg["node2"], nodes, bif_skip)
            if not prox_first:                       # orient proximal->distal
                coords, radii, bif = coords[::-1], radii[::-1], bif[::-1]
            # Drop the duplicated shared node between consecutive segments.
            if all_c and len(coords) and np.linalg.norm(coords[0] - all_c[-1][-1]) < 1e-6:
                coords, radii, bif = coords[1:], radii[1:], bif[1:]
            all_c.append(coords); all_r.append(radii); all_b.append(bif)
        if not all_c:
            continue
        out[v] = {
            "coords": np.vstack(all_c),
            "radii": np.concatenate(all_r),
            "bif": np.concatenate(all_b),
        }
    if not return_competitors:
        return out

    # Side-branch (non-annotated) segment centrelines, for Voronoi exclusion.
    annotated = set(svm)
    comp: list[np.ndarray] = []
    for i, seg in enumerate(segments):
        if i in annotated:
            continue
        sp = prepare_segment_spline(seg, points, nodes)
        if sp is None:
            continue
        comp.append(np.asarray(sp["coords"], dtype=np.float64))
    competitor_pts = np.vstack(comp) if comp else np.empty((0, 3), dtype=np.float64)
    return out, competitor_pts


def _order_vessel_segments(
    idxs: list[int],
    segments: list[dict[str, Any]],
    nodes: dict[int, tuple],
    parent_idx: np.ndarray,
    depth: np.ndarray,
) -> list[tuple[int, bool]]:
    """Order a vessel's segment indices proximal->distal, returning
    ``[(seg_idx, proximal_node_is_node1), ...]`` for orientation.

    Walks shared nodes from the ostial (min-depth) segment; falls back to plain
    depth order if the chain breaks (e.g. a non-contiguous annotation)."""
    if not idxs:
        return []
    remaining = set(idxs)
    ostial = min(idxs, key=lambda i: (int(depth[i])))
    p0 = prox_node(ostial, segments, parent_idx, nodes)
    chain: list[tuple[int, bool]] = []
    cur_seg, cur_prox = ostial, p0
    while True:
        seg = segments[cur_seg]
        prox_is_n1 = (seg["node1"] == cur_prox)
        distal = seg["node2"] if prox_is_n1 else seg["node1"]
        chain.append((cur_seg, prox_is_n1))
        remaining.discard(cur_seg)
        nxt = None
        for j in remaining:
            sj = segments[j]
            if sj["node1"] == distal or sj["node2"] == distal:
                nxt = j
                break
        if nxt is None:
            break
        cur_seg, cur_prox = nxt, distal
    # Append any unreachable members in depth order (best-effort).
    for j in sorted(remaining, key=lambda i: int(depth[i])):
        sj = segments[j]
        chain.append((j, True))
    return chain


# ── Per-point interval-segment assignment + export ─────────────────────────────


def assign_points_to_segments(
    centrelines: dict[str, dict[str, np.ndarray]],
    wall_pts: np.ndarray,
    interval_mm: float,
    competitor_pts: np.ndarray | None = None,
    radius_factor: float = 1.5,
) -> dict[str, np.ndarray]:
    """Tag each wall node with the interval segment it belongs to.

    Every vessel centreline is partitioned into fixed ``interval_mm`` arc-length
    bins (``seg_idx = floor(s / interval_mm)``, matching the measurement stations)
    and all vessels' centreline points are pooled. Each wall point is assigned the
    ``(vessel, seg_idx)`` of its **nearest** centreline point — but only when that
    centreline point is the *global* nearest over the annotated points **and** the
    side-branch ``competitor_pts`` (so branch nodes are not stolen into a main
    vessel) and the node lies within ``radius_factor * local_radius`` of it. Mirrors
    :func:`wss_contour_compare._nearest_annotated` (kept self-contained here to avoid
    a circular import).

    Nodes failing either test are left unassigned. Returns a dict of per-wall-point
    arrays (all length ``N = len(wall_pts)``):

    * ``vessel`` (object/str, ``""`` when unassigned),
    * ``seg_idx`` (int, ``-1`` when unassigned),
    * ``vessel_id`` (int, ``-1`` when unassigned) — stable per-vessel index,
    * ``s_mm`` (float, NaN when unassigned) — arc length of the owning centreline pt,
    * ``seg_global`` (float, NaN when unassigned) — ``vessel_id*1000 + seg_idx``, a
      distinct value per ``(vessel, segment)`` used to colour the debug viz,
    * ``keep`` (bool) — whether the node was assigned.
    """
    wall_pts = np.asarray(wall_pts, dtype=np.float64)
    n_wall = len(wall_pts)

    cl_pts: list[np.ndarray] = []
    cl_radius: list[float] = []
    cl_seg: list[int] = []
    cl_vid: list[int] = []
    cl_s: list[float] = []
    cl_vessel: list[str] = []
    vessel_ids: dict[str, int] = {}
    for v, d in centrelines.items():
        coords = np.asarray(d["coords"], dtype=np.float64)
        radii = np.asarray(d["radii"], dtype=np.float64)
        if len(coords) == 0:
            continue
        vid = vessel_ids.setdefault(v, len(vessel_ids))
        s = _polyline_arclength(coords)
        seg_of = np.floor(s / float(interval_mm)).astype(int)
        for i in range(len(coords)):
            cl_pts.append(coords[i])
            cl_radius.append(float(radii[i]))
            cl_seg.append(int(seg_of[i]))
            cl_vid.append(vid)
            cl_s.append(float(s[i]))
            cl_vessel.append(v)

    vessel = np.full(n_wall, "", dtype=object)
    seg_idx = np.full(n_wall, -1, dtype=np.int64)
    vessel_id = np.full(n_wall, -1, dtype=np.int64)
    s_mm = np.full(n_wall, np.nan, dtype=np.float64)
    seg_global = np.full(n_wall, np.nan, dtype=np.float64)
    keep = np.zeros(n_wall, dtype=bool)
    if not cl_pts or n_wall == 0:
        return {"vessel": vessel, "seg_idx": seg_idx, "vessel_id": vessel_id,
                "s_mm": s_mm, "seg_global": seg_global, "keep": keep}

    cl_pts_a = np.asarray(cl_pts, dtype=np.float64)
    cl_radius_a = np.asarray(cl_radius, dtype=np.float64)
    cl_seg_a = np.asarray(cl_seg, dtype=np.int64)
    cl_vid_a = np.asarray(cl_vid, dtype=np.int64)
    cl_s_a = np.asarray(cl_s, dtype=np.float64)
    cl_vessel_a = np.asarray(cl_vessel, dtype=object)
    n_ann = len(cl_pts_a)

    comp = (np.asarray(competitor_pts, dtype=np.float64)
            if competitor_pts is not None and len(competitor_pts) else
            np.empty((0, 3), dtype=np.float64))
    allpts = np.vstack([cl_pts_a, comp]) if len(comp) else cl_pts_a
    dist, gidx = KDTree(allpts).query(wall_pts)
    is_ann = gidx < n_ann
    idx = np.where(is_ann, gidx, 0)          # 0 is a safe placeholder; gated out below
    keep = is_ann & (dist <= radius_factor * cl_radius_a[idx])

    ki = idx[keep]
    vessel[keep] = cl_vessel_a[ki]
    seg_idx[keep] = cl_seg_a[ki]
    vessel_id[keep] = cl_vid_a[ki]
    s_mm[keep] = cl_s_a[ki]
    seg_global[keep] = cl_vid_a[ki].astype(np.float64) * 1000.0 + cl_seg_a[ki]
    return {"vessel": vessel, "seg_idx": seg_idx, "vessel_id": vessel_id,
            "s_mm": s_mm, "seg_global": seg_global, "keep": keep}


def write_points_csv(
    path: str | Path,
    wall_pts: np.ndarray,
    wall_wss: np.ndarray,
    assign: dict[str, np.ndarray],
) -> tuple[int, int]:
    """Write one row per wall node (``POINT_CSV_FIELDS``), tagged by its owning
    ``(vessel, seg_idx)`` from :func:`assign_points_to_segments`. Rows are grouped
    by segment (assigned nodes first, ordered by vessel then ``seg_idx``; unassigned
    nodes last with ``vessel=""``, ``seg_idx=-1`` and a blank ``s_mm``). Coordinates
    are the scaled/offset wall-cloud coordinates. Returns ``(n_assigned, n_total)``.
    """
    vessel = assign["vessel"]
    seg_idx = assign["seg_idx"]
    vessel_id = assign["vessel_id"]
    s_mm = assign["s_mm"]
    keep = np.asarray(assign["keep"], dtype=bool)
    n = len(wall_pts)
    # Group per segment: assigned first (-keep sorts True before False), then by
    # vessel_id, then seg_idx. Unassigned share vessel_id/seg_idx == -1 and trail.
    order = np.lexsort((seg_idx, vessel_id, ~keep)) if n else np.empty(0, dtype=int)
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=POINT_CSV_FIELDS)
        w.writeheader()
        for i in order:
            assigned = bool(keep[i])
            w.writerow({
                "vessel": vessel[i] if assigned else "",
                "seg_idx": int(seg_idx[i]),
                "x": round(float(wall_pts[i, 0]), 5),
                "y": round(float(wall_pts[i, 1]), 5),
                "z": round(float(wall_pts[i, 2]), 5),
                "wss": round(float(wall_wss[i]), 6),
                "s_mm": round(float(s_mm[i]), 4) if assigned else "",
            })
    return int(np.count_nonzero(keep)), int(n)


# ── Station sampling + arc sweep ───────────────────────────────────────────────


def _polyline_arclength(coords: np.ndarray) -> np.ndarray:
    s = np.zeros(len(coords))
    if len(coords) > 1:
        s[1:] = np.cumsum(np.linalg.norm(np.diff(coords, axis=0), axis=1))
    return s


def sample_stations(
    coords: np.ndarray, radii: np.ndarray, bif: np.ndarray, interval_mm: float
) -> list[dict[str, Any]]:
    """Stations every ``interval_mm`` of arc length. Each is a dict with
    ``s, pos, tangent, radius``; bifurcation-zone stations are dropped."""
    s = _polyline_arclength(coords)
    total = float(s[-1]) if len(s) else 0.0
    if total <= 0:
        return []
    targets = np.arange(0.0, total + 1e-9, float(interval_mm))
    stations: list[dict[str, Any]] = []
    for t in targets:
        j = int(np.searchsorted(s, t, side="right") - 1)
        j = max(0, min(j, len(coords) - 2))
        seg_len = s[j + 1] - s[j]
        u = 0.0 if seg_len <= 1e-12 else (t - s[j]) / seg_len
        pos = coords[j] * (1 - u) + coords[j + 1] * u
        rad = float(radii[j] * (1 - u) + radii[j + 1] * u)
        if bool(bif[j]) or bool(bif[j + 1]):
            continue                                   # inside a bifurcation zone
        tang = coords[j + 1] - coords[j]
        nrm = np.linalg.norm(tang)
        if nrm < 1e-12:
            continue
        stations.append({
            "s": float(t), "pos": pos, "tangent": tang / nrm, "radius": rad,
        })
    return stations


def arc_window_min_max(
    phi_deg: np.ndarray,
    wss: np.ndarray,
    arc_deg: float = 90.0,
    rot_step_deg: float = 5.0,
) -> dict[str, Any]:
    """Rotate an ``arc_deg`` window over [0, 360) in ``rot_step_deg`` steps and
    report the max/min windowed-**mean** WSS and their centre angles, plus the
    ring mean over all points.

    ``phi_deg`` are the circumferential angles (deg) of the points and ``wss``
    their WSS. Returns NaN windows when no rotation catches a point. The max/min
    *values* are invariant to the frame that defined ``phi``; only the reported
    angles rotate with it."""
    nan = float("nan")
    phi = np.asarray(phi_deg, dtype=np.float64) % 360.0
    w = np.asarray(wss, dtype=np.float64)
    if len(w) == 0:
        return {"ring_mean_wss": nan, "wss_max_window": nan, "angle_max_deg": nan,
                "wss_min_window": nan, "angle_min_deg": nan}
    ring_mean = float(np.mean(w))
    half = arc_deg / 2.0
    alphas = np.arange(0.0, 360.0, rot_step_deg)
    means: list[float] = []
    used_alpha: list[float] = []
    for a in alphas:
        d = np.abs((phi - a + 180.0) % 360.0 - 180.0)   # circular distance
        sel = d <= half
        if np.any(sel):
            means.append(float(np.mean(w[sel])))
            used_alpha.append(float(a))
    if not means:
        return {"ring_mean_wss": ring_mean, "wss_max_window": nan,
                "angle_max_deg": nan, "wss_min_window": nan, "angle_min_deg": nan}
    means_a = np.asarray(means)
    ai_max = int(np.argmax(means_a))
    ai_min = int(np.argmin(means_a))
    return {
        "ring_mean_wss": ring_mean,
        "wss_max_window": float(means_a[ai_max]),
        "angle_max_deg": float(used_alpha[ai_max]),
        "wss_min_window": float(means_a[ai_min]),
        "angle_min_deg": float(used_alpha[ai_min]),
    }


def sweep_station(
    pos: np.ndarray,
    tangent: np.ndarray,
    n_hat: np.ndarray,
    b_hat: np.ndarray,
    kd: KDTree,
    wall_pts: np.ndarray,
    wall_wss: np.ndarray,
    local_radius: float,
    slab_half: float,
    arc_deg: float = 90.0,
    rot_step_deg: float = 5.0,
    min_pts: int = 8,
) -> dict[str, Any]:
    """Sweep a ``arc_deg`` window around the lumen at one station.

    Selects wall points within an axial slab ``|(p-pos).t| <= slab_half`` of the
    cross-section, bins them by circumferential angle, then rotates the window
    centre over [0, 360) reporting the max/min windowed-mean WSS and their angles.
    Returns NaN fields when fewer than ``min_pts`` wall points are in the slab.
    """
    nan = float("nan")
    empty = {
        "n_wall_pts": 0, "ring_mean_wss": nan,
        "wss_max_window": nan, "angle_max_deg": nan,
        "wss_min_window": nan, "angle_min_deg": nan,
    }
    search_r = float(local_radius) + float(slab_half) + max(float(local_radius), 1e-6)
    cand = kd.query_ball_point(pos, search_r)
    if not cand:
        return empty
    cand = np.asarray(cand, dtype=np.int64)
    rel = wall_pts[cand] - pos
    axial = rel @ tangent
    in_slab = np.abs(axial) <= slab_half
    cand = cand[in_slab]
    rel = rel[in_slab]
    if len(cand) < min_pts:
        return dict(empty, n_wall_pts=int(len(cand)))
    # Remove the axial component; keep points near the lumen wall (within ~1.5 r).
    radial = rel - np.outer(rel @ tangent, tangent)
    rdist = np.linalg.norm(radial, axis=1)
    keep = rdist > 1e-9
    cand, radial, rdist = cand[keep], radial[keep], rdist[keep]
    if len(cand) < min_pts:
        return dict(empty, n_wall_pts=int(len(cand)))
    phi = np.degrees(np.arctan2(radial @ b_hat, radial @ n_hat)) % 360.0
    w = wall_wss[cand]
    res = arc_window_min_max(phi, w, arc_deg, rot_step_deg)
    return {"n_wall_pts": int(len(cand)), **res}


def recenter_station(
    pos: np.ndarray,
    tangent: np.ndarray,
    kd: KDTree,
    wall_pts: np.ndarray,
    slab_half: float,
    local_radius: float,
    radius_factor: float = 1.5,
    min_pts: int = 8,
) -> np.ndarray:
    """Lateral snap of a station ``pos`` onto the local lumen centroid of the wall
    points in its cross-section slab, **preserving axial position** (the
    along-tangent component of the shift is removed).

    Corrects the drift between the un-smoothed graph centreline and the smoothed
    centreline the CFD mesh was built from, so the slab selection and radial φ in
    :func:`sweep_station` are taken from the true lumen axis. Returns ``pos``
    unchanged when fewer than ``min_pts`` wall points fall in the slab."""
    search_r = float(local_radius) * float(radius_factor) + float(slab_half)
    cand = kd.query_ball_point(pos, search_r)
    if len(cand) < min_pts:
        return pos
    cand = np.asarray(cand, dtype=np.int64)
    rel = wall_pts[cand] - pos
    axial = rel @ tangent
    radial = rel - np.outer(axial, tangent)
    rdist = np.linalg.norm(radial, axis=1)
    sel = (np.abs(axial) <= slab_half) & (rdist <= radius_factor * float(local_radius))
    pts = wall_pts[cand[sel]]
    if len(pts) < min_pts:
        return pos
    delta = pts.mean(axis=0) - pos
    return pos + (delta - (delta @ tangent) * tangent)


def sample_anchor_wss(
    pos: np.ndarray,
    local_radius: float,
    kd: KDTree,
    wall_wss: np.ndarray,
    radius_factor: float = 1.0,
    min_pts: int = 1,
) -> dict[str, Any]:
    """Ring-mean WSS of all wall nodes within ``radius_factor*local_radius`` of an
    anchor (a sphere query) — no axial slab, no arc window, no bifurcation
    exclusion.

    Unlike :func:`sweep_station`, this places a fixed measurement at the anchor
    regardless of geometry, so a shared anchor grid yields the same number of
    measurements across models (for paired comparison). Returns
    ``{'n_wall_pts', 'ring_mean_wss'}``; ``ring_mean_wss`` is NaN when fewer than
    ``min_pts`` nodes fall inside the sphere.
    """
    cand = kd.query_ball_point(pos, float(local_radius) * float(radius_factor))
    n = len(cand)
    if n < min_pts:
        return {"n_wall_pts": int(n), "ring_mean_wss": float("nan")}
    w = wall_wss[np.asarray(cand, dtype=np.int64)]
    return {"n_wall_pts": int(n), "ring_mean_wss": float(np.mean(w))}


def evaluate_vessel(
    coords: np.ndarray,
    radii: np.ndarray,
    bif: np.ndarray,
    wall_pts: np.ndarray,
    wall_wss: np.ndarray,
    kd: KDTree | None = None,
    interval_mm: float = 1.0,
    arc_deg: float = 90.0,
    rot_step_deg: float = 5.0,
    slab_mm: float | None = None,
    min_pts: int = 8,
    recenter: bool = True,
    recenter_factor: float = 1.5,
) -> list[dict[str, Any]]:
    """Full per-vessel evaluation: place stations, build parallel-transported
    frames, sweep the arc window at each. Returns one row dict per station.

    When ``recenter`` is set, each station is first snapped laterally onto the
    local lumen centroid (:func:`recenter_station`) so the measurement is taken on
    the mesh's true axis rather than the drifted graph centreline."""
    if kd is None:
        kd = KDTree(wall_pts)
    slab_half = (interval_mm / 2.0) if slab_mm is None else float(slab_mm)
    stations = sample_stations(coords, radii, bif, interval_mm)
    rows: list[dict[str, Any]] = []
    prev_normal: np.ndarray | None = None
    for st in stations:
        _t, n_hat, b_hat = compute_frenet_frame(st["tangent"], prev_normal)
        prev_normal = n_hat
        pos = st["pos"]
        if recenter:
            pos = recenter_station(pos, st["tangent"], kd, wall_pts, slab_half,
                                   st["radius"], recenter_factor, min_pts)
        res = sweep_station(
            pos, st["tangent"], n_hat, b_hat, kd, wall_pts, wall_wss,
            st["radius"], slab_half, arc_deg, rot_step_deg, min_pts)
        rows.append({
            "s_mm": round(st["s"], 4),
            "x": round(float(pos[0]), 5),
            "y": round(float(pos[1]), 5),
            "z": round(float(pos[2]), 5),
            "local_radius_mm": round(float(st["radius"]), 5),
            **res,
        })
    return rows


# ── Plotting ───────────────────────────────────────────────────────────────────


def plot_vessel_wss(vessel: str, rows: list[dict[str, Any]], out_dir: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - env dependent
        print(f"  [PLOT] matplotlib unavailable ({exc}); skipping {vessel} plot")
        return
    s = np.array([r["s_mm"] for r in rows], dtype=float)
    mx = np.array([r["wss_max_window"] for r in rows], dtype=float)
    mn = np.array([r["wss_min_window"] for r in rows], dtype=float)
    rm = np.array([r["ring_mean_wss"] for r in rows], dtype=float)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.fill_between(s, mn, mx, color="#cce0f5", label="90 deg min-max band")
    ax.plot(s, mx, color="#cc3311", lw=1.6, label="max sector")
    ax.plot(s, rm, color="#333333", lw=1.4, ls="--", label="ring mean")
    ax.plot(s, mn, color="#0077bb", lw=1.6, label="min sector")
    ax.set_xlabel("arc length along vessel  s (mm)")
    ax.set_ylabel("wall shear stress (Pa)")
    ax.set_title(f"{vessel}: radially-resolved WSS ({len(rows)} stations)")
    ax.legend(fontsize=8)
    fig.savefig(out_dir / f"wss_{vessel}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


# ── Orchestration ──────────────────────────────────────────────────────────────


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Radially-resolved WSS along annotated main epicardial vessels")
    ap.add_argument("--xml", required=True, help="base input Amira .am.xml")
    ap.add_argument("--model-dir", default=None,
                    help="pruned model dir (contains model.am.xml); sidecar "
                         "defaults to <model-dir>/../epicardial.json. Omit both "
                         "--model-dir and --model-xml to run on the un-pruned base "
                         "tree (no ratio cropping); --sidecar is then required.")
    ap.add_argument("--model-xml", default=None,
                    help="pruned model XML (overrides <model-dir>/model.am.xml)")
    ap.add_argument("--sidecar", default=None, help="epicardial.json annotation")
    ap.add_argument("--wss-csv", required=True, help="CFD-Post wall-node CSV")
    ap.add_argument("--wss-col", default=None,
                    help="exact CFD-Post WSS column name (default: auto-detect)")
    ap.add_argument("--out", default="wss_out", help="output directory")
    ap.add_argument("--interval-mm", type=float, default=3.0,
                    help="station spacing along the centreline (mm)")
    ap.add_argument("--arc-deg", type=float, default=90.0, help="arc window width")
    ap.add_argument("--rot-step-deg", type=float, default=5.0,
                    help="window rotation step")
    ap.add_argument("--slab-mm", type=float, default=None,
                    help="axial half-thickness for wall-point selection "
                         "(default: interval/2)")
    ap.add_argument("--bif-skip", type=int, default=BIF_SKIP_POINTS,
                    help="contours excluded near a degree>=3 node")
    ap.add_argument("--min-pts", type=int, default=8,
                    help="min wall points in a slab to evaluate a station")
    ap.add_argument("--recenter", action=argparse.BooleanOptionalAction, default=True,
                    help="snap each station onto the local lumen centroid of the "
                         "cloud (corrects centreline drift vs the smoothed mesh)")
    ap.add_argument("--recenter-factor", type=float, default=1.5,
                    help="radial gate (xlocal_radius) for the station recentre")
    ap.add_argument("--coord-scale", type=float, default=None,
                    help="multiply CFX coords to mm (e.g. 1000 if CFX is in "
                         "metres); default auto-detects from the CSV unit header "
                         "or the wall/centreline extent ratio")
    ap.add_argument("--coord-offset", default="0,0,0",
                    help="add to CFX coords (mm) after scaling: 'dx,dy,dz'")
    ap.add_argument("--no-plot", action="store_true", help="skip per-vessel plots")
    ap.add_argument("--viz", action="store_true",
                    help="interactive pyvista overlay of stations over the cloud")
    args = ap.parse_args(argv)

    model_dir = Path(args.model_dir) if args.model_dir else None
    model_xml = Path(args.model_xml) if args.model_xml else (
        model_dir / "model.am.xml" if model_dir else None)
    # model_xml may be None -> build centrelines on the un-pruned base tree.
    if args.sidecar:
        sidecar = Path(args.sidecar)
    elif model_dir:
        sidecar = model_dir.parent / "epicardial.json"
    elif model_xml:
        sidecar = model_xml.parent.parent / "epicardial.json"
    else:
        sidecar = Path(args.xml).parent / "epicardial.json"
    if not sidecar.exists():
        raise SystemExit(f"[WSS][ERROR] sidecar not found: {sidecar} (pass --sidecar)")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # Vessel centrelines (mm).
    print(f"[WSS] geometry: "
          + (str(model_xml) if model_xml else "base tree (no ratio pruning)"))
    centrelines, competitor_pts = build_vessel_centrelines(
        args.xml, model_xml, sidecar, bif_skip=args.bif_skip,
        return_competitors=True)
    if not centrelines:
        raise SystemExit("[WSS][ERROR] no annotated main vessels resolved on the "
                         + ("model." if model_xml else "base tree."))
    print(f"[WSS] {len(centrelines)} vessel(s): "
          + ", ".join(f"{v}({len(d['coords'])} pts)" for v, d in centrelines.items()))

    # WSS cloud (resolve scale, apply scale + offset, then sanity-check alignment).
    wall_pts, wall_wss, coord_unit = read_wss_csv(args.wss_csv, wss_col=args.wss_col)
    all_cl = np.vstack([d["coords"] for d in centrelines.values()])
    scale, why = resolve_coord_scale(coord_unit, wall_pts, all_cl, args.coord_scale)
    print(f"[WSS] coord scale x{scale:g} ({why})")
    offset = np.array([float(x) for x in args.coord_offset.split(",")], dtype=np.float64)
    wall_pts = wall_pts * scale + offset
    check_alignment(wall_pts, all_cl)
    kd = KDTree(wall_pts)

    all_rows: list[dict[str, Any]] = []
    for v, d in centrelines.items():
        rows = evaluate_vessel(
            d["coords"], d["radii"], d["bif"], wall_pts, wall_wss, kd=kd,
            interval_mm=args.interval_mm, arc_deg=args.arc_deg,
            rot_step_deg=args.rot_step_deg, slab_mm=args.slab_mm, min_pts=args.min_pts,
            recenter=args.recenter, recenter_factor=args.recenter_factor)
        for r in rows:
            r["vessel"] = v
        with open(out / f"wss_{v}.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
            w.writeheader()
            w.writerows({k: r.get(k) for k in CSV_FIELDS} for r in rows)
        n_valid = sum(1 for r in rows if not math.isnan(r["wss_max_window"]))
        print(f"  [{v}] {len(rows)} stations ({n_valid} with WSS data) "
              f"-> wss_{v}.csv")
        if not args.no_plot and rows:
            plot_vessel_wss(v, rows, out)
        all_rows.extend(rows)

    with open(out / "wss_all.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
        w.writeheader()
        w.writerows({k: r.get(k) for k in CSV_FIELDS} for r in all_rows)
    print(f"[DONE] wrote {out / 'wss_all.csv'} ({len(all_rows)} stations across "
          f"{len(centrelines)} vessels)")

    # Per-wall-point export, tagged by owning interval segment.
    assign = assign_points_to_segments(
        centrelines, wall_pts, args.interval_mm,
        competitor_pts=competitor_pts, radius_factor=args.recenter_factor)
    n_assigned, n_total = write_points_csv(
        out / "wss_points.csv", wall_pts, wall_wss, assign)
    print(f"[DONE] wrote {out / 'wss_points.csv'} ({n_assigned}/{n_total} wall "
          f"nodes assigned to a segment, {n_total - n_assigned} unassigned)")

    if args.viz:
        _viz_overlay(centrelines, wall_pts, wall_wss, all_rows, assign)
    return 0


def _viz_overlay(centrelines, wall_pts, wall_wss, rows, assign) -> None:  # pragma: no cover
    """Interactive overlay of the wall cloud + centrelines + station markers, with
    the cloud colouring toggled between **WSS** (turbo) and **interval segment**
    (tab20) via the ``t`` key or an on-screen checkbox. In WSS mode the whole cloud
    (main vessels *and* side branches) is coloured by WSS; unassigned/side-branch
    nodes (``seg_global`` NaN) render grey **only** in segment mode. Falls back to a
    screenshot on a headless box (via ``viz._show_plotter``)."""
    try:
        import pyvista as pv
    except Exception as exc:
        print(f"[VIZ][WARN] pyvista unavailable: {exc}")
        return
    pl = pv.Plotter(title="WSS along main vessels")
    pl.set_background("white")

    # Separate PolyData per mode so neither actor steals the other's active scalar
    # (a shared mesh would make WSS mode render the segment array + grey NaN nodes).
    cloud_wss = pv.PolyData(wall_pts)
    cloud_wss["WSS"] = np.asarray(wall_wss, dtype=np.float64)
    cloud_seg = pv.PolyData(wall_pts)
    cloud_seg["segment"] = np.asarray(assign["seg_global"], dtype=np.float64)
    wss_actor = pl.add_mesh(
        cloud_wss, scalars="WSS", cmap="turbo", point_size=4,
        render_points_as_spheres=True, opacity=0.5, name="cloud_wss",
        scalar_bar_args={"title": "WSS (Pa)"})
    seg_actor = pl.add_mesh(
        cloud_seg, scalars="segment", cmap="tab20", point_size=6,
        render_points_as_spheres=True, opacity=0.9,
        nan_color="lightgrey", nan_opacity=1.0,
        name="cloud_seg", show_scalar_bar=False)
    seg_actor.SetVisibility(False)

    for d in centrelines.values():
        c = np.asarray(d["coords"], dtype=np.float64)
        conn = np.concatenate([[len(c)], np.arange(len(c), dtype=np.int64)])
        pl.add_mesh(pv.PolyData(c, lines=conn), color="black", line_width=3)
    valid = [r for r in rows if not math.isnan(r["wss_max_window"])]
    if valid:
        spts = np.array([[r["x"], r["y"], r["z"]] for r in valid])
        sp = pv.PolyData(spts)
        sp["max_sector_WSS"] = np.array([r["wss_max_window"] for r in valid])
        pl.add_mesh(sp, scalars="max_sector_WSS", cmap="turbo", point_size=12,
                    render_points_as_spheres=True, show_scalar_bar=False)

    state = {"seg": False, "hud": None}

    def _refresh_hud() -> None:
        if state["hud"] is not None:
            pl.remove_actor(state["hud"], render=False)
        mode = "segment (tab20)" if state["seg"] else "WSS (turbo, Pa)"
        state["hud"] = pl.add_text(
            f"[t] / checkbox  colour: {mode}", position="upper_left",
            font_size=10, color="black")

    def set_mode(seg: bool) -> None:
        state["seg"] = bool(seg)
        wss_actor.SetVisibility(not state["seg"])
        seg_actor.SetVisibility(state["seg"])
        try:                                   # only the WSS bar; hide in seg mode
            pl.scalar_bars["WSS (Pa)"].SetVisibility(not state["seg"])
        except Exception:
            pass
        _refresh_hud()
        pl.render()

    _refresh_hud()
    pl.add_key_event("t", lambda: set_mode(not state["seg"]))
    try:
        pl.add_checkbox_button_widget(
            lambda flag: set_mode(bool(flag)), value=False, position=(10, 10),
            size=30, color_on="green", color_off="lightgray")
        pl.add_text("colour by segment", position=(48, 12), font_size=9,
                    color="black")
    except Exception as exc:
        print(f"[VIZ][WARN] checkbox widget unavailable ({exc}); use the 't' key.")

    try:
        from .viz import _show_plotter
        _show_plotter(pl, "wss_overlay")
    except Exception:
        pl.show()


__all__ = [
    "read_wss_csv",
    "check_alignment",
    "resolve_coord_scale",
    "build_vessel_centrelines",
    "assign_points_to_segments",
    "write_points_csv",
    "sample_stations",
    "sweep_station",
    "arc_window_min_max",
    "recenter_station",
    "sample_anchor_wss",
    "evaluate_vessel",
    "plot_vessel_wss",
    "main",
]


if __name__ == "__main__":
    raise SystemExit(main())
