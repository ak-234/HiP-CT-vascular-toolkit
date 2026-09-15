"""Strahler-order and radius-resolved morphometry + haemodynamics.

Joins three sources into one per-segment table and two aggregate tables:

* the **spatial graph** (Amira ``.am``/``.am.xml``) — Strahler order, radius,
  length, cross-sectional area, vessel and bifurcation counts;
* **CFD wall nodes** (``X,Y,Z,Pressure,Wall Shear``) — wall shear and wall
  pressure, extracted by :mod:`coronary_sdf.cfx_extract`;
* **CFD volume nodes** (``X,Y,Z,Pressure,Velocity …``) — lumen pressure and
  velocity. Flow is reported two ways: ``mean|v| * A`` (an upper bound, since a
  speed magnitude counts swirl and reversal as forward transport) and
  ``mean|v.t| * A`` along the local centreline tangent, which is the component
  that actually moves fluid along the vessel. See the README's sign-convention
  section; only the axial pair should be read as transport.

Anatomy and haemodynamics may come from *different* graphs: the imaged tree is
usually larger than the geometry that was meshed and solved.  ``--graph`` supplies
the anatomical tree (best statistics) and ``--cfd-graph`` the tree the solver ran
on; when only ``--graph`` is given it serves both roles.

Every figure in :mod:`coronary_sdf.strahler_plots` is driven by the CSVs written
here, so each plotted number is reproducible from a table:

``segments.csv``        one row per vessel segment — the audit trail
``by_strahler.csv``     per-order mean / SD / n for every metric
``by_radius_bin.csv``   the same metrics binned by vessel radius
``bifurcations.csv``    bifurcation counts per order
``provenance.json``     inputs, units and options behind the tables

Usage::

    python -m coronary_sdf.strahler_analysis \
        --graph pruned.am.xml \
        --cfd-graph ratio_6/model.am.xml \
        --cfd-run left:left_tree_ratio_6_001 \
        --cfd-run right:right_tree_ratio_6_001 \
        --cfd-dir analysis_out/cfd_extract \
        --out analysis_out/ratio_6
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass, asdict, field, fields
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from scipy.spatial import KDTree

from .parse_amira import parse_xml
from .flow_fractions import BIF_SKIP_POINTS

# parse_amira normalises every source graph to micrometres; the analysis works
# in millimetres throughout (radii, lengths, areas) to match the CFD export.
UM_PER_MM = 1000.0

# A CFD node counts towards a segment only within this multiple of the local
# graph radius, so nodes in a neighbouring vessel are never absorbed.
DEFAULT_RADIUS_FACTOR = 1.5

# m^3/s -> mL/min
M3S_TO_ML_MIN = 6.0e7

# CFX exports ``Pressure`` in Pa, relative to the domain ``Reference Pressure``;
# absolute pressure is ``Pressure + Reference Pressure``. These runs set
# ``Reference Pressure = 0 [atm]`` (`flow_fractions.py`), so *nothing is
# subtracted* -- an earlier version of this comment claimed 1 atm was, which was
# wrong. What makes the field gauge is the boundary condition: every opening and
# outlet is pinned at ``Relative Pressure = 0 [Pa]``, so the field is referenced
# **to the outlets**, and the negative values it contains (6.5% of left-tree
# nodes, 26.6% of right) are static pressures below that reference rather than
# below vacuum. `pressure_offset` therefore shifts an outlet-referenced field to a
# physiological coronary pressure; it is not an atmospheric correction, and adding
# one atmosphere would double-count. Reported in mmHg, the unit used clinically.
PA_TO_MMHG = 1.0 / 133.322387415


# ── per-segment record ────────────────────────────────────────────────────────


@dataclass
class SegmentRecord:
    """One vessel segment: geometry from the graph, fields from the CFD run.

    CFD columns are NaN when the segment lies outside the solved geometry (the
    anatomical tree usually extends past it) or when every candidate node failed
    the radius gate."""

    seg_id: int
    strahler: int
    tree: str = ""

    # geometry (graph)
    n_points: int = 0
    n_points_used: int = 0
    length_mm: float = math.nan
    radius_mean_mm: float = math.nan
    radius_sd_mm: float = math.nan
    radius_min_mm: float = math.nan
    radius_max_mm: float = math.nan
    csa_mm2: float = math.nan
    volume_mm3: float = math.nan

    # haemodynamics (CFD)
    n_wall_nodes: int = 0
    wss_mean_pa: float = math.nan
    wss_sd_pa: float = math.nan
    pressure_wall_mean_mmhg: float = math.nan
    pressure_wall_sd_mmhg: float = math.nan

    n_volume_nodes: int = 0
    velocity_mean_ms: float = math.nan
    velocity_sd_ms: float = math.nan
    pressure_lumen_mean_mmhg: float = math.nan
    pressure_lumen_sd_mmhg: float = math.nan
    # `<|v|> * A`. An upper bound on throughput, not a flux: it counts swirl and
    # retrograde motion as forward transport, and does not conserve at a
    # bifurcation. Named for what it is so no figure can treat it as a flow rate.
    flow_speed_ml_min: float = math.nan
    # `<|v . t|> * A` with `t` the local centreline tangent -- the component that
    # actually moves fluid along the vessel.
    flow_axial_ml_min: float = math.nan
    velocity_axial_mean_ms: float = math.nan
    # `<|v . t|> / <|v|>`: 1 where flow is purely axial, lower where it is not.
    axial_fraction: float = math.nan
    # `|<sign(v . t)>|`: 1 where every node moves the same way along the vessel,
    # 0 where as much goes back as forward. Separates "fast but disordered" from
    # "fast and through-flowing", which the speed magnitude alone cannot.
    flow_coherence: float = math.nan


# Metrics aggregated by Strahler order and by radius bin. ``label`` is the axis
# label used by the plotting module, so units live in exactly one place.
METRICS: list[tuple[str, str]] = [
    ("radius_mean_mm", "Vessel radius (mm)"),
    ("csa_mm2", "Cross-sectional area (mm$^2$)"),
    ("length_mm", "Segment length (mm)"),
    ("wss_mean_pa", "Wall shear stress (Pa)"),
    # "Static pressure", not "wall pressure": this is the ordinary static pressure
    # of the flow, sampled on the wall boundary and through the lumen. The two
    # differ only in where they were sampled, and agree closely because pressure
    # is near-uniform across a vessel cross-section at these Reynolds numbers.
    ("pressure_wall_mean_mmhg", "Static pressure (mmHg)"),
    ("pressure_lumen_mean_mmhg", "Static pressure (mmHg)"),
    ("velocity_mean_ms", "Velocity (m s$^{-1}$)"),
    # Not a flux -- see `SegmentRecord.flow_speed_ml_min`. The axial pair beside it
    # is the one that means transport along the vessel.
    ("flow_speed_ml_min", "Mean-speed flow estimate (mL min$^{-1}$)"),
    ("flow_axial_ml_min", "Axial flow estimate (mL min$^{-1}$)"),
    ("velocity_axial_mean_ms", "Axial velocity (m s$^{-1}$)"),
    ("axial_fraction", r"Axial fraction $\langle|v\cdot t|\rangle/\langle|v|\rangle$"),
    ("flow_coherence", "Flow coherence"),
]


# ── graph geometry ────────────────────────────────────────────────────────────


def _polyline_length_mm(coords_mm: np.ndarray) -> float:
    if len(coords_mm) < 2:
        return 0.0
    return float(np.linalg.norm(np.diff(coords_mm, axis=0), axis=1).sum())


def filter_segments(
    segments: list[dict[str, Any]], exclude: set[int]
) -> list[dict[str, Any]]:
    """Drop segments whose ``id`` is in ``exclude``.

    Used to remove geometry that is known-bad rather than merely unusual — the
    ostial inlet stub of an ex vivo specimen, for instance, where cannulation
    corrupts the segmented radius over the first few millimetres. Such a segment
    fails junction conservation (its measured diameter is smaller than its own
    daughter's), so leaving it in would bias its whole Strahler order."""
    if not exclude:
        return segments
    return [s for i, s in enumerate(segments) if int(s.get("id", i)) not in exclude]


def _bif_mask(
    n: int, node1: int, node2: int, degrees: dict[int, int], bif_skip: int
) -> np.ndarray:
    """Points within ``bif_skip`` contours of a degree>=3 end.

    Radius and wall fields are both unreliable inside a junction: the
    skeletoniser inflates the thickness where branches meet, and the solver sees
    a locally three-dimensional flow there rather than vessel flow."""
    mask = np.zeros(n, dtype=bool)
    k = min(bif_skip, n)
    # Degrees are recomputed from the surviving segments, not read from the file:
    # excluding a segment can turn a junction into a plain continuation, and a
    # stale stored degree would keep masking points that are no longer near one.
    if degrees.get(node1, 0) >= 3:
        mask[:k] = True
    if degrees.get(node2, 0) >= 3:
        mask[n - k:] = True
    return mask


def node_degrees(segments: list[dict[str, Any]]) -> dict[int, int]:
    """Segment-incidence degree per node id, recomputed from ``segments``."""
    deg: dict[int, int] = {}
    for s in segments:
        deg[s["node1"]] = deg.get(s["node1"], 0) + 1
        deg[s["node2"]] = deg.get(s["node2"], 0) + 1
    return deg


def segment_geometry(
    nodes: dict[int, tuple],
    points: dict[int, tuple],
    segments: list[dict[str, Any]],
    bif_skip: int = BIF_SKIP_POINTS,
) -> tuple[list[SegmentRecord], list[np.ndarray], list[np.ndarray]]:
    """Per-segment geometry plus the centreline arrays used for CFD assignment.

    Returns ``(records, coords_mm, radii_mm)`` where the two lists are parallel to
    ``records`` and hold only the **non-bifurcation** points of each segment, so
    the same masking governs the radius statistics and the field sampling."""
    records: list[SegmentRecord] = []
    coords_out: list[np.ndarray] = []
    radii_out: list[np.ndarray] = []
    degrees = node_degrees(segments)

    for si, seg in enumerate(segments):
        pids = list(seg["point_ids"])
        if not pids:
            continue
        raw = np.array([points[p] for p in pids if p in points], dtype=np.float64)
        if len(raw) == 0:
            continue
        coords_mm = raw[:, :3] / UM_PER_MM
        radii_mm = raw[:, 3] / UM_PER_MM

        keep = ~_bif_mask(len(coords_mm), seg["node1"], seg["node2"], degrees, bif_skip)
        if not keep.any():  # a segment shorter than two bif zones: keep it whole
            keep = np.ones(len(coords_mm), dtype=bool)

        r = radii_mm[keep]
        rec = SegmentRecord(
            seg_id=int(seg.get("id", si)),
            strahler=int(seg.get("strahler", 0)),
            n_points=len(coords_mm),
            n_points_used=int(keep.sum()),
            length_mm=_polyline_length_mm(coords_mm),
            radius_mean_mm=float(r.mean()),
            radius_sd_mm=float(r.std(ddof=1)) if len(r) > 1 else 0.0,
            radius_min_mm=float(r.min()),
            radius_max_mm=float(r.max()),
        )
        # Area/volume use the mean radius so a segment contributes one CSA, which
        # is what "cross-sectional area of this vessel" means when summed per order.
        rec.csa_mm2 = math.pi * rec.radius_mean_mm ** 2
        rec.volume_mm3 = rec.csa_mm2 * rec.length_mm
        records.append(rec)
        coords_out.append(coords_mm[keep])
        radii_out.append(r)

    return records, coords_out, radii_out


# -- Strahler elements ---------------------------------------------------------


def strahler_elements(segments: list[dict[str, Any]]) -> list[list[int]]:
    """Segment ids grouped into Strahler *elements*, largest order first.

    An element is a maximal run of consecutive same-order segments -- what the
    morphometry literature calls a vessel. The graph splits a trunk wherever a
    side branch joins it, so a single RCA arrives as a dozen segments all
    carrying the parent order. Summing a per-vessel quantity over segments then
    counts the same lumen once per side branch, and because side branches
    concentrate on the large vessels the bias lands almost entirely on the top
    orders: it is what makes total cross-sectional area fall towards the
    periphery instead of rising.

    Two same-order segments meeting at a node continue one vessel only if they
    are the *only* two of that order there; three would be a genuine trifurcation
    of equals and each limb starts its own element."""
    parent: dict[int, int] = {}

    def find(x: int) -> int:
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    ids = [int(seg.get("id", i)) for i, seg in enumerate(segments)]
    order_of = {sid: int(seg.get("strahler", 0))
                for sid, seg in zip(ids, segments)}

    incident: dict[int, list[int]] = {}
    for sid, seg in zip(ids, segments):
        for k in ("node1", "node2"):
            incident.setdefault(seg[k], []).append(sid)

    for at_node in incident.values():
        top_order = max(order_of[sid] for sid in at_node)
        top = [sid for sid in at_node if order_of[sid] == top_order]
        if len(top) != 2:
            continue
        a, b = find(top[0]), find(top[1])
        if a != b:
            parent[a] = b

    groups: dict[int, list[int]] = {}
    for sid in ids:
        groups.setdefault(find(sid), []).append(sid)
    return sorted(groups.values(), key=lambda g: (-order_of[g[0]], min(g)))


def _weighted(values: np.ndarray, weights: np.ndarray) -> float:
    """Length-weighted mean over the entries that carry a value."""
    ok = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    if not ok.any():
        ok = np.isfinite(values)
        return float(values[ok].mean()) if ok.any() else math.nan
    return float(np.average(values[ok], weights=weights[ok]))


def collapse_to_elements(
    records: list[SegmentRecord], groups: list[list[int]]
) -> list[SegmentRecord]:
    """One record per element, built from the segment records it contains.

    Lengths, volumes and node counts add; every intensive quantity -- radius,
    the CFD fields -- is averaged along the element weighted by segment length,
    so a 30 mm run does not count the same as the 0.8 mm stub beside it. The
    element's CSA is taken from its mean radius, exactly as a segment's is, so
    "total CSA of order n" sums one cross-section per vessel.

    ``seg_id`` becomes the lowest segment id in the run, and ``n_points`` the
    run's total, so an element remains traceable back to ``segments_raw.csv``."""
    by_id = {r.seg_id: r for r in records}
    out: list[SegmentRecord] = []
    for group in groups:
        members = [by_id[i] for i in group if i in by_id]
        if not members:
            continue
        if len(members) == 1:
            out.append(members[0])
            continue
        L = np.array([m.length_mm for m in members], dtype=np.float64)
        L = np.where(np.isfinite(L), L, 0.0)

        def wmean(attr: str) -> float:
            return _weighted(
                np.array([getattr(m, attr) for m in members], dtype=np.float64), L)

        el = SegmentRecord(seg_id=min(m.seg_id for m in members),
                           strahler=members[0].strahler,
                           tree=members[0].tree)
        el.n_points = sum(m.n_points for m in members)
        el.n_points_used = sum(m.n_points_used for m in members)
        el.length_mm = float(L.sum())
        el.radius_mean_mm = wmean("radius_mean_mm")
        # Spread along the whole element: the variance within each segment plus
        # the variance between segment means, which is the taper the run carries.
        var = _weighted(
            np.array([m.radius_sd_mm ** 2 + (m.radius_mean_mm - el.radius_mean_mm) ** 2
                      for m in members], dtype=np.float64), L)
        el.radius_sd_mm = math.sqrt(var) if math.isfinite(var) else math.nan
        el.radius_min_mm = min(m.radius_min_mm for m in members)
        el.radius_max_mm = max(m.radius_max_mm for m in members)
        el.csa_mm2 = math.pi * el.radius_mean_mm ** 2
        # Volume adds -- it is the lumen the run actually encloses, taper and all,
        # not the cylinder implied by the element's mean radius.
        el.volume_mm3 = float(np.nansum([m.volume_mm3 for m in members]))

        el.n_wall_nodes = sum(m.n_wall_nodes for m in members)
        el.n_volume_nodes = sum(m.n_volume_nodes for m in members)
        for attr in ("wss_mean_pa", "wss_sd_pa", "pressure_wall_mean_mmhg",
                     "pressure_wall_sd_mmhg", "velocity_mean_ms", "velocity_sd_ms",
                     "pressure_lumen_mean_mmhg", "pressure_lumen_sd_mmhg",
                     "flow_speed_ml_min", "flow_axial_ml_min",
                     "velocity_axial_mean_ms", "axial_fraction", "flow_coherence"):
            setattr(el, attr, wmean(attr))
        out.append(el)
    return out


def count_terminals(
    segments: list[dict[str, Any]], exclude: set[int] | None = None
) -> tuple[dict[int, int], int]:
    """Terminal branches per Strahler order, and the total.

    A terminal branch is a segment with a free end -- a node no other segment
    touches. ``exclude`` drops segment ids that own a free end without being a
    terminal, which in practice means the ostial root when it is kept: its
    inlet node has degree 1, but it is where the tree starts rather than where
    it ends.

    In a well-ordered tree this total equals the order-1 count, since order 1 is
    by definition a vessel with no daughters. The two are computed independently
    -- one from node degree, one from the stored Strahler field -- so a mismatch
    between them is a real signal that the ordering does not match the topology,
    and is worth surfacing rather than assuming away."""
    exclude = exclude or set()
    deg = node_degrees(segments)
    per_order: dict[int, int] = {}
    total = 0
    for i, seg in enumerate(segments):
        if int(seg.get("id", i)) in exclude:
            continue
        if deg.get(seg["node1"], 0) == 1 or deg.get(seg["node2"], 0) == 1:
            order = int(seg.get("strahler", 0))
            per_order[order] = per_order.get(order, 0) + 1
            total += 1
    return per_order, total


#: Columns of the compact morphometry table, in order. The wide `by_strahler.csv`
#: carries every statistic of every metric; this is the subset a paper or a slide
#: actually quotes, with a totals row, so it can be pasted without re-cutting.
TABLE_COLUMNS = [
    "strahler", "n_vessels", "n_terminal_branches", "n_bifurcations",
    "diameter_median_mm", "diameter_q25_mm", "diameter_q75_mm",
    "csa_total_mm2", "length_total_mm", "volume_total_mm3",
]


def morphometry_table(
    records: list[SegmentRecord],
    bifurcations: dict[int, int] | None = None,
    terminals: dict[int, int] | None = None,
) -> list[dict[str, Any]]:
    """Per-order morphometry as a compact table, with a ``total`` row.

    Counts and extensive quantities add down the column; the diameter quartiles on
    the total row are taken over every record rather than averaged across orders,
    which would weight a 3-vessel order the same as an 82-vessel one."""
    bifurcations = bifurcations or {}
    terminals = terminals or {}

    def block(group: list[SegmentRecord], label: Any,
              n_bif: int, n_term: int) -> dict[str, Any]:
        radii = np.array([r.radius_mean_mm for r in group], dtype=np.float64)
        radii = radii[np.isfinite(radii)]
        q25, med, q75 = (np.percentile(radii, [25, 50, 75])
                         if len(radii) else (math.nan,) * 3)
        return {
            "strahler": label,
            "n_vessels": len(group),
            "n_terminal_branches": n_term,
            "n_bifurcations": n_bif,
            "diameter_median_mm": float(2.0 * med),
            "diameter_q25_mm": float(2.0 * q25),
            "diameter_q75_mm": float(2.0 * q75),
            "csa_total_mm2": float(np.nansum([r.csa_mm2 for r in group])),
            "length_total_mm": float(np.nansum([r.length_mm for r in group])),
            "volume_total_mm3": float(np.nansum([r.volume_mm3 for r in group])),
        }

    rows = [
        block([r for r in records if r.strahler == order], order,
              int(bifurcations.get(order, 0)), int(terminals.get(order, 0)))
        for order in sorted({r.strahler for r in records})
    ]
    rows.append(block(records, "total",
                      int(sum(bifurcations.values())),
                      int(sum(terminals.values()))))
    return rows


def count_bifurcations(
    nodes: dict[int, tuple], segments: list[dict[str, Any]]
) -> tuple[dict[int, int], int, dict[int, int]]:
    """Bifurcations per Strahler order.

    A bifurcation is a node where three or more segments meet.  It is attributed
    to the **parent** order — the highest Strahler order incident on that node —
    which is the convention that makes "bifurcations of order n" mean "places
    where an order-n vessel divides".  Returns ``(per_order, total, degree_hist)``."""
    deg = node_degrees(segments)
    incident: dict[int, list[int]] = {}
    for s in segments:
        order = int(s.get("strahler", 0))
        incident.setdefault(s["node1"], []).append(order)
        incident.setdefault(s["node2"], []).append(order)

    per_order: dict[int, int] = {}
    degree_hist: dict[int, int] = {}
    total = 0
    for nid, d in deg.items():
        degree_hist[d] = degree_hist.get(d, 0) + 1
        if d < 3:
            continue
        total += 1
        parent = max(incident.get(nid, [0]))
        per_order[parent] = per_order.get(parent, 0) + 1
    return per_order, total, degree_hist


# ── CFD sampling ──────────────────────────────────────────────────────────────


def load_cfd_npz(path: Path) -> dict[str, np.ndarray]:
    """Load a :mod:`coronary_sdf.cfx_extract` ``.npz`` as ``{column: array}``.

    Coordinates are converted from the CFX export unit (metres) to millimetres so
    they share the graph's frame; every other column keeps its SI unit."""
    d = np.load(path, allow_pickle=True)
    cols = [str(c) for c in d["columns"]]
    vals = d["values"]
    out = {c: vals[:, i] for i, c in enumerate(cols)}
    if "surface_control_area" in d:
        out["Surface Control Area"] = np.asarray(d["surface_control_area"])
    for axis in ("X", "Y", "Z"):
        out[axis] = out[axis] * 1000.0
    return out


def assign_nodes_to_segments(
    seg_coords: list[np.ndarray],
    seg_radii: list[np.ndarray],
    query_pts: np.ndarray,
    radius_factor: float = DEFAULT_RADIUS_FACTOR,
) -> np.ndarray:
    """Owning segment index per query point, ``-1`` when unassigned.

    Each CFD node takes the segment of its nearest centreline point, but only if
    it lies within ``radius_factor x`` that point's graph radius.  The gate is what
    keeps a node in a neighbouring vessel — or in a branch that was pruned out of
    the graph — from being counted into this segment."""
    if not seg_coords or len(query_pts) == 0:
        return np.full(len(query_pts), -1, dtype=np.int64)

    pool = np.vstack(seg_coords)
    pool_radius = np.concatenate(seg_radii)
    pool_seg = np.concatenate(
        [np.full(len(c), i, dtype=np.int64) for i, c in enumerate(seg_coords)]
    )

    dist, idx = KDTree(pool).query(np.asarray(query_pts, dtype=np.float64))
    owner = np.where(dist <= radius_factor * pool_radius[idx], pool_seg[idx], -1)
    return owner.astype(np.int64)


def centreline_tangents(seg_coords: list[np.ndarray]) -> list[np.ndarray]:
    """Unit tangent at every centreline point, per segment.

    Oriented along increasing point index, which is the segment's stored direction
    and *not* necessarily the flow direction -- see :func:`axial_velocity`."""
    out: list[np.ndarray] = []
    for c in seg_coords:
        if len(c) < 2:
            out.append(np.tile(np.array([1.0, 0.0, 0.0]), (max(len(c), 1), 1)))
            continue
        t = np.gradient(c, axis=0)
        norm = np.linalg.norm(t, axis=1, keepdims=True)
        out.append(t / np.maximum(norm, 1e-12))
    return out


def axial_velocity(
    seg_coords: list[np.ndarray],
    seg_radii: list[np.ndarray],
    query_pts: np.ndarray,
    uvw: np.ndarray,
    radius_factor: float = DEFAULT_RADIUS_FACTOR,
) -> np.ndarray:
    """Velocity component along the local centreline, signed, per query point.

    ``|v|`` is not a flow rate. Flux through a section is ``int v.n dA``, and the
    speed magnitude exceeds ``v.n`` at every node where the flow is not aligned
    with the vessel -- swirl, secondary motion and recirculation all inflate it,
    and retrograde flow is counted as forward. Measured on LADAF-2024-28, the
    velocity components are negative at 15-56% of nodes depending on component and
    tree, so the flow is demonstrably not axis-aligned and the difference is not
    academic.

    Projecting onto the centreline tangent recovers the component that actually
    transports fluid along the vessel, and keeps its sign. The tangent's *direction*
    is the segment's stored point order, which is arbitrary with respect to flow, so
    the sign here is per-segment-arbitrary; :func:`_signed_axial_stats` resolves it
    by taking the segment's dominant direction. What survives is the distinction
    between flow that moves along the vessel and flow that does not."""
    n = len(query_pts)
    if not seg_coords or n == 0:
        return np.zeros(n)

    pool = np.vstack(seg_coords)
    pool_radius = np.concatenate(seg_radii)
    tangents = np.vstack(centreline_tangents(seg_coords))

    dist, idx = KDTree(pool).query(np.asarray(query_pts, dtype=np.float64))
    axial = np.einsum("ij,ij->i", np.asarray(uvw, dtype=np.float64), tangents[idx])
    return np.where(dist <= radius_factor * pool_radius[idx], axial, 0.0)


def _grouped_stats(
    owner: np.ndarray,
    values: np.ndarray,
    n_segments: int,
    weights: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-segment ``(count, mean, sd)`` of ``values`` grouped by ``owner``.

    ``weights`` should be the nodal control volumes for a volume field: CFX
    inflates the boundary layer, so an unweighted mean counts the slow near-wall
    fluid many times over and biases velocity low.  Wall fields pass
    ``weights=None`` — there the nodes tile the surface roughly evenly.

    ``np.bincount`` keeps this linear in the number of CFD nodes; a Python loop
    over millions of nodes would dominate the runtime."""
    ok = owner >= 0
    o, v = owner[ok], values[ok]
    count = np.bincount(o, minlength=n_segments).astype(np.float64)
    w = np.ones_like(v) if weights is None else np.asarray(weights)[ok]

    wsum = np.bincount(o, weights=w, minlength=n_segments)
    total = np.bincount(o, weights=w * v, minlength=n_segments)
    total_sq = np.bincount(o, weights=w * v * v, minlength=n_segments)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = np.where(wsum > 0, total / wsum, np.nan)
        var = np.where(count > 1, total_sq / wsum - mean ** 2, np.nan)
    sd = np.sqrt(np.clip(var, 0.0, None))
    return count, mean, sd


def apply_cfd_run(
    records: list[SegmentRecord],
    seg_coords: list[np.ndarray],
    seg_radii: list[np.ndarray],
    wall: dict[str, np.ndarray] | None,
    volume: dict[str, np.ndarray] | None,
    tree: str,
    radius_factor: float = DEFAULT_RADIUS_FACTOR,
    pressure_offset: float = 0.0,
) -> int:
    """Fill the CFD columns of ``records`` from one solver run.

    Returns the number of segments that received any CFD data.  A segment already
    claimed by an earlier run (the other tree) is left alone, so left and right
    runs compose without overwriting each other."""
    n = len(records)
    touched = np.zeros(n, dtype=bool)

    if wall is not None:
        pts = np.column_stack([wall["X"], wall["Y"], wall["Z"]])
        owner = assign_nodes_to_segments(seg_coords, seg_radii, pts, radius_factor)
        # Not every solve writes Wall Shear — it depends on the run's extra-output
        # list — so the column is optional and its absence leaves WSS as NaN
        # rather than failing the whole join.
        shear = wall.get("Wall Shear")
        if shear is None:
            print(f"[strahler][WARN] '{tree}' wall export has no Wall Shear; "
                  "WSS will be blank for this tree")
        wall_area = wall.get("Surface Control Area")
        if wall_area is None:
            print(
                f"[strahler][WARN] '{tree}' wall export has no Surface Control "
                "Area; falling back to unweighted wall statistics"
            )
        cnt, wss_m, wss_s = (
            _grouped_stats(owner, shear, n, wall_area) if shear is not None
            else (np.bincount(owner[owner >= 0], minlength=n).astype(np.float64),
                  np.full(n, np.nan), np.full(n, np.nan))
        )
        _, p_m, p_s = _grouped_stats(owner, wall["Pressure"], n, wall_area)
        for i, rec in enumerate(records):
            if cnt[i] == 0 or rec.n_wall_nodes:
                continue
            rec.n_wall_nodes = int(cnt[i])
            rec.wss_mean_pa, rec.wss_sd_pa = float(wss_m[i]), float(wss_s[i])
            rec.pressure_wall_mean_mmhg = float(p_m[i]) * PA_TO_MMHG + pressure_offset
            rec.pressure_wall_sd_mmhg = float(p_s[i]) * PA_TO_MMHG
            touched[i] = True

    if volume is not None:
        pts = np.column_stack([volume["X"], volume["Y"], volume["Z"]])
        owner = assign_nodes_to_segments(seg_coords, seg_radii, pts, radius_factor)
        speed = volume.get("Velocity")
        if speed is None:  # older exports carry components only
            speed = np.sqrt(
                volume["Velocity u"] ** 2
                + volume["Velocity v"] ** 2
                + volume["Velocity w"] ** 2
            )
        cell_vol = volume.get("Volume of Finite Volumes")
        cnt, v_m, v_s = _grouped_stats(owner, speed, n, cell_vol)
        _, p_m, p_s = _grouped_stats(owner, volume["Pressure"], n, cell_vol)

        # The axial component, for a flow estimate that is not a speed magnitude.
        # Needs the components; an export carrying only the `Velocity` scalar
        # cannot support this and leaves `flow_axial_ml_min` as NaN.
        have_uvw = all(k in volume for k in ("Velocity u", "Velocity v", "Velocity w"))
        a_m = sgn_m = None
        if have_uvw:
            uvw = np.column_stack([volume["Velocity u"], volume["Velocity v"],
                                   volume["Velocity w"]])
            v_ax = axial_velocity(seg_coords, seg_radii, pts, uvw, radius_factor)
            _, a_m, _ = _grouped_stats(owner, np.abs(v_ax), n, cell_vol)
            _, sgn_m, _ = _grouped_stats(owner, np.sign(v_ax), n, cell_vol)

        for i, rec in enumerate(records):
            if cnt[i] == 0 or rec.n_volume_nodes:
                continue
            rec.n_volume_nodes = int(cnt[i])
            rec.velocity_mean_ms, rec.velocity_sd_ms = float(v_m[i]), float(v_s[i])
            rec.pressure_lumen_mean_mmhg = float(p_m[i]) * PA_TO_MMHG + pressure_offset
            rec.pressure_lumen_sd_mmhg = float(p_s[i]) * PA_TO_MMHG
            # `flow_speed_ml_min` is <|v|> * A -- an UPPER BOUND on throughput, kept
            # because the older tables report it, and renamed so no figure can
            # mistake it for a flux. It does not conserve across a bifurcation.
            if np.isfinite(rec.csa_mm2):
                rec.flow_speed_ml_min = (
                    rec.velocity_mean_ms * rec.csa_mm2 * 1e-6 * M3S_TO_ML_MIN
                )
                if have_uvw and a_m is not None and sgn_m is not None:
                    rec.velocity_axial_mean_ms = float(a_m[i])
                    rec.axial_fraction = (
                        float(a_m[i]) / float(v_m[i]) if v_m[i] > 0 else math.nan
                    )
                    # Sign is per-segment arbitrary (see `axial_velocity`), so the
                    # magnitude of the mean sign is what says whether the segment
                    # carries coherent through-flow: 1 = every node one way,
                    # 0 = as much back as forward.
                    rec.flow_coherence = abs(float(sgn_m[i]))
                    rec.flow_axial_ml_min = (
                        rec.velocity_axial_mean_ms * rec.csa_mm2 * 1e-6 * M3S_TO_ML_MIN
                    )
            touched[i] = True

    for i, rec in enumerate(records):
        if touched[i] and not rec.tree:
            rec.tree = tree
    return int(touched.sum())


# ── aggregation ───────────────────────────────────────────────────────────────


_STAT_KEYS = ("n", "mean", "sd", "sem", "median", "q25", "q75", "iqr",
              "min", "max", "sum")


def _stats(values: np.ndarray) -> dict[str, float]:
    """Summary statistics for one metric over a group of segments.

    Both the parametric (mean/SD/SEM) and the order statistics (median/quartiles)
    are stored so a figure can be drawn either way from the same table. The
    haemodynamic distributions are strongly right-skewed, so median with the
    interquartile range is the summary the cardiovascular literature expects;
    mean +/- SD is kept for the morphometry, where it is the convention."""
    v = np.asarray(values, dtype=np.float64)
    v = v[np.isfinite(v)]
    if len(v) == 0:
        return {k: (0 if k == "n" else math.nan) for k in _STAT_KEYS}
    sd = float(v.std(ddof=1)) if len(v) > 1 else 0.0
    q25, q75 = (float(x) for x in np.percentile(v, [25, 75]))
    return {
        "n": int(len(v)),
        "mean": float(v.mean()),
        "sd": sd,
        "sem": sd / math.sqrt(len(v)) if len(v) else math.nan,
        "median": float(np.median(v)),
        "q25": q25,
        "q75": q75,
        "iqr": q75 - q25,
        "min": float(v.min()),
        "max": float(v.max()),
        "sum": float(v.sum()),
    }


def aggregate_by_strahler(
    records: list[SegmentRecord], bifurcations: dict[int, int] | None = None
) -> list[dict[str, Any]]:
    """Per-order mean/SD/n for every metric, plus vessel and bifurcation counts."""
    bifurcations = bifurcations or {}
    orders = sorted({r.strahler for r in records})
    rows: list[dict[str, Any]] = []
    for order in orders:
        group = [r for r in records if r.strahler == order]
        row: dict[str, Any] = {
            "strahler": order,
            "n_vessels": len(group),
            "n_bifurcations": int(bifurcations.get(order, 0)),
            "total_length_mm": float(
                np.nansum([r.length_mm for r in group])
            ),
        }
        for key, _label in METRICS:
            st = _stats(np.array([getattr(r, key) for r in group], dtype=np.float64))
            for stat, val in st.items():
                row[f"{key}_{stat}"] = val
        # Diameter is the reporting calibre for Strahler-order morphometry.
        # Radius stays in this table because it is the graph's native measurement
        # and remains the basis of the separate radius-binned analysis.
        for stat in _STAT_KEYS:
            value = row[f"radius_mean_mm_{stat}"]
            row[f"diameter_mean_mm_{stat}"] = (
                value if stat == "n" else 2.0 * value
            )
        rows.append(row)
    return rows


def radius_bin_edges(
    records: list[SegmentRecord], n_bins: int = 8, log: bool = True
) -> np.ndarray:
    """Bin edges over the observed radius range.

    Coronary radii span more than a decade, so the default is logarithmic —
    linear bins would put almost every vessel in the first bin."""
    r = np.array([x.radius_mean_mm for x in records], dtype=np.float64)
    r = r[np.isfinite(r) & (r > 0)]
    if len(r) == 0:
        return np.array([0.0, 1.0])
    lo, hi = float(r.min()), float(r.max())
    if log and lo > 0:
        return np.logspace(math.log10(lo), math.log10(hi * 1.001), n_bins + 1)
    return np.linspace(lo, hi * 1.001, n_bins + 1)


def aggregate_by_radius(
    records: list[SegmentRecord], edges: np.ndarray
) -> list[dict[str, Any]]:
    """The same metrics binned by segment mean radius rather than Strahler order."""
    r = np.array([x.radius_mean_mm for x in records], dtype=np.float64)
    idx = np.digitize(r, edges) - 1
    rows: list[dict[str, Any]] = []
    for b in range(len(edges) - 1):
        group = [rec for rec, k in zip(records, idx) if k == b]
        row: dict[str, Any] = {
            "bin": b,
            "radius_lo_mm": float(edges[b]),
            "radius_hi_mm": float(edges[b + 1]),
            "radius_mid_mm": float(math.sqrt(edges[b] * edges[b + 1]))
            if edges[b] > 0 else float((edges[b] + edges[b + 1]) / 2),
            "n_vessels": len(group),
        }
        for key, _label in METRICS:
            st = _stats(np.array([getattr(x, key) for x in group], dtype=np.float64))
            for stat, val in st.items():
                row[f"{key}_{stat}"] = val
        rows.append(row)
    return rows


# ── CSV output ────────────────────────────────────────────────────────────────


def write_csv(path: Path, rows: Iterable[dict[str, Any]], columns: list[str] | None = None) -> int:
    """Write ``rows`` as CSV; NaN is written as an empty cell so spreadsheets and
    pandas both read it as missing rather than the string 'nan'."""
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return 0
    cols = columns or list(rows[0].keys())

    def fmt(v: Any) -> str:
        if isinstance(v, float):
            if not math.isfinite(v):
                return ""
            return f"{v:.6g}"
        return str(v)

    with path.open("w", newline="") as fh:
        fh.write(",".join(cols) + "\n")
        for row in rows:
            fh.write(",".join(fmt(row.get(c, "")) for c in cols) + "\n")
    return len(rows)


# ── driver ────────────────────────────────────────────────────────────────────


def _parse_run(spec: str) -> tuple[str, str]:
    """``'left:left_tree_ratio_6_001'`` -> ``('left', 'left_tree_ratio_6_001')``."""
    if ":" in spec:
        tree, stem = spec.split(":", 1)
        return tree.strip(), stem.strip()
    return Path(spec).stem, spec.strip()


def run_analysis(
    graph_xml: Path,
    out_dir: Path,
    cfd_graph_xml: Path | None = None,
    cfd_runs: list[tuple[str, str]] | None = None,
    cfd_dir: Path | None = None,
    bif_skip: int = BIF_SKIP_POINTS,
    radius_factor: float = DEFAULT_RADIUS_FACTOR,
    n_radius_bins: int = 8,
    log_bins: bool = True,
    pressure_offset: float = 0.0,
    exclude_segments: set[int] | None = None,
    use_elements: bool = False,
    root_segments: set[int] | None = None,
) -> dict[str, Any]:
    """Build every table and write them under ``out_dir``. Returns a summary dict."""
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Anatomy — from the imaged tree, which is usually larger than the mesh.
    print(f"[strahler] anatomy graph: {graph_xml}")
    a_nodes, a_points, a_segments = parse_xml(graph_xml)
    exclude_segments = exclude_segments or set()
    if exclude_segments:
        before = len(a_segments)
        a_segments = filter_segments(a_segments, exclude_segments)
        print(f"[strahler]   excluded segment(s) {sorted(exclude_segments)}: "
              f"{before} -> {len(a_segments)} segments")
    anat, _c, _r = segment_geometry(a_nodes, a_points, a_segments, bif_skip)
    anat_raw = anat
    if use_elements:
        anat = collapse_to_elements(anat, strahler_elements(a_segments))
    bif_per_order, bif_total, degree_hist = count_bifurcations(a_nodes, a_segments)
    # The root's ostial node has degree 1, so a kept root would be counted as a
    # terminal branch unless it is named. Naming it excludes it from that count
    # only -- it stays in every statistic.
    term_per_order, term_total = count_terminals(a_segments, root_segments)
    print(f"[strahler]   {len(anat_raw)} segments"
          + (f" -> {len(anat)} Strahler elements" if use_elements else "")
          + f", {bif_total} bifurcations, {term_total} terminal branches")

    # 2. Haemodynamics — on the geometry the solver actually saw.
    cfd_records: list[SegmentRecord] = []
    cfd_runs = cfd_runs or []
    if cfd_runs:
        cfd_xml = cfd_graph_xml or graph_xml
        print(f"[strahler] CFD graph: {cfd_xml}")
        c_nodes, c_points, c_segments = parse_xml(cfd_xml)
        c_segments = filter_segments(c_segments, exclude_segments)
        cfd_records, seg_coords, seg_radii = segment_geometry(
            c_nodes, c_points, c_segments, bif_skip)
        for tree, stem in cfd_runs:
            wall_p = (cfd_dir or Path(".")) / f"{stem}_wall.npz"
            vol_p = (cfd_dir or Path(".")) / f"{stem}_volume.npz"
            wall = load_cfd_npz(wall_p) if wall_p.exists() else None
            volume = load_cfd_npz(vol_p) if vol_p.exists() else None
            if wall is None and volume is None:
                print(f"[strahler][WARN] no export found for '{stem}' in {cfd_dir}")
                continue
            n_hit = apply_cfd_run(cfd_records, seg_coords, seg_radii,
                                  wall, volume, tree, radius_factor,
                                  pressure_offset)
            print(f"[strahler]   {tree}: wall="
                  f"{0 if wall is None else len(wall['X'])} vol="
                  f"{0 if volume is None else len(volume['X'])} "
                  f"-> {n_hit} segments covered")
        # Collapse only after the CFD pass, which assigns mesh nodes to the
        # nearest segment centreline and so needs the segments themselves.
        if use_elements:
            cfd_records = collapse_to_elements(
                cfd_records, strahler_elements(c_segments))

    # 3. Tables. Anatomy metrics come from the anatomical tree; CFD metrics from
    #    the solved tree. Keeping them in separate files makes the differing n
    #    explicit rather than hiding it inside one merged row.
    n_seg = write_csv(out_dir / "segments.csv",
                      [asdict(r) for r in anat],
                      [f.name for f in fields(SegmentRecord)])
    if use_elements:
        write_csv(out_dir / "segments_raw.csv",
                  [asdict(r) for r in anat_raw],
                  [f.name for f in fields(SegmentRecord)])
    n_cfd = 0
    if cfd_records:
        n_cfd = write_csv(out_dir / "segments_cfd.csv",
                          [asdict(r) for r in cfd_records],
                          [f.name for f in fields(SegmentRecord)])

    anat_rows = aggregate_by_strahler(anat, bif_per_order)
    write_csv(out_dir / "by_strahler.csv", anat_rows)
    if cfd_records:
        cfd_bif, _t, _d = count_bifurcations(c_nodes, c_segments)
        write_csv(out_dir / "by_strahler_cfd.csv",
                  aggregate_by_strahler(cfd_records, cfd_bif))

    edges = radius_bin_edges(cfd_records or anat, n_radius_bins, log_bins)
    write_csv(out_dir / "by_radius_bin.csv",
              aggregate_by_radius(cfd_records or anat, edges))

    write_csv(out_dir / "bifurcations.csv", [
        {"strahler": o, "n_bifurcations": bif_per_order.get(o, 0)}
        for o in sorted(bif_per_order)
    ] + [{"strahler": "total", "n_bifurcations": bif_total}])

    write_csv(out_dir / "morphometry_table.csv",
              morphometry_table(anat, bif_per_order, term_per_order),
              TABLE_COLUMNS)

    provenance = {
        "graph": str(graph_xml),
        "cfd_graph": str(cfd_graph_xml) if cfd_graph_xml else None,
        "cfd_runs": [{"tree": t, "stem": s} for t, s in cfd_runs],
        "cfd_dir": str(cfd_dir) if cfd_dir else None,
        "excluded_segments": sorted(exclude_segments),
        "root_segments": sorted(root_segments or set()),
        "row_unit": "strahler_element" if use_elements else "graph_segment",
        "n_anatomy_elements": len(anat) if use_elements else None,
        "bif_skip_points": bif_skip,
        "radius_factor": radius_factor,
        "radius_bin_edges_mm": [float(e) for e in edges],
        "log_radius_bins": log_bins,
        "n_anatomy_segments": len(anat_raw),
        "n_analysis_rows": n_seg,
        "n_cfd_segments": n_cfd,
        "n_bifurcations_total": bif_total,
        "n_terminal_branches": term_total,
        "n_terminal_branches_per_order": {str(k): v for k, v in
                                          sorted(term_per_order.items())},
        "node_degree_histogram": {str(k): v for k, v in sorted(degree_hist.items())},
        "units": {
            "length": "mm", "radius": "mm", "area": "mm^2", "volume": "mm^3",
            "pressure": "mmHg", "wall_shear": "Pa", "velocity": "m/s",
            "flow": "mL/min",
        },
        "pressure_note": (
            "CFX relative (gauge) static pressure, referenced to the domain "
            "reference pressure (1 atm in these runs), converted at "
            f"1 mmHg = 133.322387415 Pa and offset by {pressure_offset} mmHg."
        ),
        "pressure_offset_mmhg": pressure_offset,
    }
    (out_dir / "provenance.json").write_text(json.dumps(provenance, indent=2))
    print(f"[strahler] wrote tables to {out_dir}")
    return provenance


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--graph", type=Path, required=True,
                    help="anatomical spatial graph (.am / .am.xml)")
    ap.add_argument("--root-seg", type=int, action="append", default=[],
                    help="segment id of an ostial root; repeat as needed. Its "
                         "free end is not a terminal branch, so naming it keeps "
                         "the terminal count right. It is excluded from that "
                         "count only and stays in every statistic -- use "
                         "--exclude-seg to drop it from the analysis")
    ap.add_argument("--elements", action="store_true",
                    help="aggregate over Strahler elements -- maximal runs of "
                         "consecutive same-order segments, one row per vessel -- "
                         "rather than over graph segments")
    ap.add_argument("--cfd-graph", type=Path, default=None,
                    help="graph matching the CFD geometry (defaults to --graph)")
    ap.add_argument("--cfd-run", action="append", default=[],
                    help="'<tree>:<export stem>', e.g. left:left_tree_ratio_6_001")
    ap.add_argument("--cfd-dir", type=Path, default=None,
                    help="directory holding the cfx_extract .npz files")
    ap.add_argument("--out", type=Path, required=True, help="output directory")
    ap.add_argument("--bif-skip", type=int, default=BIF_SKIP_POINTS,
                    help="contours excluded either side of a bifurcation node")
    ap.add_argument("--radius-factor", type=float, default=DEFAULT_RADIUS_FACTOR,
                    help="CFD node accepted within this multiple of local radius")
    ap.add_argument("--radius-bins", type=int, default=8)
    ap.add_argument("--linear-bins", action="store_true",
                    help="use linear rather than logarithmic radius bins")
    ap.add_argument("--exclude-seg", type=int, action="append", default=[],
                    help="segment id to drop from every statistic; repeat as "
                         "needed. Use for known-bad geometry such as a cannulated "
                         "ostial stub whose segmented radius is corrupt")
    ap.add_argument("--pressure-offset-mmhg", type=float, default=0.0,
                    help="added to every pressure. CFX pressure is gauge, "
                         "referenced to the domain reference pressure, so the "
                         "values sit around zero; set e.g. 100 to express them "
                         "about a mean aortic pressure instead")
    args = ap.parse_args(argv)

    run_analysis(
        graph_xml=args.graph,
        out_dir=args.out,
        cfd_graph_xml=args.cfd_graph,
        cfd_runs=[_parse_run(s) for s in args.cfd_run],
        cfd_dir=args.cfd_dir,
        bif_skip=args.bif_skip,
        radius_factor=args.radius_factor,
        n_radius_bins=args.radius_bins,
        log_bins=not args.linear_bins,
        pressure_offset=args.pressure_offset_mmhg,
        exclude_segments=set(args.exclude_seg),
        use_elements=args.elements,
        root_segments=set(args.root_seg),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
