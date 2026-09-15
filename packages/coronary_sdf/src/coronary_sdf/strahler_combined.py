"""Left tree, right tree, and the two pooled -- the same morphometry three ways.

:mod:`coronary_sdf.strahler_analysis` takes one graph and one output directory.
That is the wrong shape for the question "does pooling the two coronary trees
change the answer?", for two reasons that only show up once the graphs are
opened:

* **A coronary spatial graph holds both trees.** ``pruned.am.xml`` and the
  ``ratio_8`` model each contain the left and right trees as *disjoint connected
  components* (plus, in the full skeleton, two orphan single-segment fragments).
  Running ``strahler_analysis --graph pruned.am.xml --cfd-run left:...``
  therefore produces morphometry over **both** trees while the haemodynamics
  cover only the left -- the CFD columns of the other tree stay NaN and drop out
  of ``_stats``. The existing ``left_full`` / ``strahler_right_ratio8`` outputs
  have exactly this shape, so their ``by_strahler.csv`` is not a per-tree table.
  This module splits the graph into components first and names them, so "the
  left tree" means the left tree in every table.

* **The two trees come from different graphs.** The left solve ran on the full
  skeleton (``pruned.am.xml``); the right on the ratio-8 prune
  (``ratio_8/model.am.xml``). No single ``--graph`` can serve both, so the
  combined analysis is built by *pooling per-segment records*, not by merging
  graph files. That is also the correct operation: Strahler order is a property
  of a rooted tree, so it must be assigned within each tree and only then
  pooled.

Trees are identified by a **root segment id** rather than by size or order,
since both are accidents of the pruning: the component containing segment 67 is
the left tree (its order-5 ostial trunk), the component containing 291 is the
right. Each root segment is then dropped from every statistic -- the "ignore the
inlet/root segments" requirement. Dropping it is not cosmetic: an ex vivo ostial
stub is cannulated, so its segmented radius is corrupt, and being the sole
member of the top order it would otherwise define that order single-handedly.
``--keep-root`` reverses that choice for a morphometric run that wants the
ostial trunk counted: the top order of each tree then appears in every table,
with one vessel per tree and the cannulation caveat attached.

Outputs -- each of ``left/``, ``right/`` and ``combined/`` in the layout
``strahler_plots --in <dir>`` expects::

    segments.csv, segments_cfd.csv, by_strahler.csv, by_strahler_cfd.csv,
    by_radius_bin.csv, by_radius_bin_shared.csv, bifurcations.csv,
    provenance.json

``by_radius_bin.csv`` uses bins fitted to that table's own radius range -- what a
standalone run would produce. ``by_radius_bin_shared.csv`` uses one edge set
fitted to the pooled records, which is the only way the three are comparable bin
for bin; the comparison tables are built from it.

::

    comparison/compare_by_strahler.csv   left | right | combined, per order
    comparison/compare_by_radius.csv     the same over the shared radius bins
    comparison/pooling_effect.csv        combined vs the n-weighted expectation
    comparison/summary.md                what actually differs, in prose

**Pressure does not pool.** Left and right are separate solves and CFX
references static pressure to each solve's own outlets (pinned at 0 Pa), so a
combined pressure statistic mixes two different data. The pressure columns are
carried through the per-tree tables and written into the combined table only so
the comparison can show *how far apart* they are; ``summary.md`` flags them.
Radius, CSA, length, WSS and velocity are solve-independent and pool
legitimately.

Usage::

    python -m coronary_sdf.strahler_combined \
        --left-graph  ".../pruned.am.xml" --left-root-seg 67 \
        --left-run    "meshmixer_..._001" \
        --left-cfd-dir analysis_out/cfd_extract_full \
        --right-graph ".../ratio_8/model.am.xml" --right-root-seg 291 \
        --right-run   "right_tree_ratio_8_001" \
        --right-cfd-dir analysis_out/cfd_extract_right_ratio8 \
        --out analysis_out/combined_lr

When one graph holds both trees at the same distal cut-off -- ``radius_final.am``
does -- pass it as both ``--left-graph`` and ``--right-graph``; the components
are still named by root segment, so nothing else changes.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from dataclasses import asdict, fields
from pathlib import Path
from typing import Any

import numpy as np

from .parse_amira import parse_xml
from .flow_fractions import BIF_SKIP_POINTS
from .strahler_analysis import (
    DEFAULT_RADIUS_FACTOR,
    METRICS,
    SegmentRecord,
    aggregate_by_radius,
    aggregate_by_strahler,
    apply_cfd_run,
    TABLE_COLUMNS,
    collapse_to_elements,
    count_bifurcations,
    count_terminals,
    filter_segments,
    load_cfd_npz,
    morphometry_table,
    node_degrees,
    radius_bin_edges,
    segment_geometry,
    strahler_elements,
    write_csv,
)

# Components below this size are graph debris -- the full skeleton carries two
# single-segment fragments belonging to neither tree. They are reported and
# dropped rather than silently swept into whichever tree is nearest.
MIN_COMPONENT_SEGMENTS = 3

# Blood density used by the solves, for turning a prescribed inlet mass flow into
# a volumetric one. Only ever used to *report* the operating point -- no analysis
# quantity depends on it.
BLOOD_DENSITY_KG_M3 = 1050.0

# kg/s -> mL/min at a given density.
KGS_TO_ML_MIN = 6.0e7

# Metrics that survive pooling across two independent solves. Pressure is left
# out on purpose: see the module docstring.
NON_POOLABLE = ["pressure_wall_mean_mmhg", "pressure_lumen_mean_mmhg"]
POOLABLE = [m for m, _ in METRICS if m not in NON_POOLABLE]

# Columns that belong to the row itself rather than to a metric, so `_merge_rows`
# never takes them from the CFD table (whose n is the covered subset).
_ROW_COLS = ("n_vessels", "n_bifurcations", "total_length_mm",
             "radius_lo_mm", "radius_hi_mm", "radius_mid_mm")

# Metric-column prefixes that come from the CFD pass rather than the graph.
_CFD_PREFIXES = ("wss_", "pressure_", "velocity_", "flow_", "axial_")


# -- graph components ---------------------------------------------------------


def connected_components(segments: list[dict[str, Any]]) -> list[list[int]]:
    """Segment indices grouped by connected component, largest first."""
    parent: dict[int, int] = {}

    def find(x: int) -> int:
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for s in segments:
        a, b = find(s["node1"]), find(s["node2"])
        if a != b:
            parent[a] = b

    groups: dict[int, list[int]] = defaultdict(list)
    for i, s in enumerate(segments):
        groups[find(s["node1"])].append(i)
    return sorted(groups.values(), key=len, reverse=True)


def component_containing(
    segments: list[dict[str, Any]], seg_id: int
) -> list[dict[str, Any]]:
    """The connected component holding segment ``seg_id``.

    Trees are named by a segment they contain rather than by rank or size: which
    component is larger depends on the pruning -- the left tree is the smaller
    component in the full skeleton and the larger one in the ratio-8 graph, so
    "largest component" would silently swap the two between runs."""
    for idxs in connected_components(segments):
        if any(int(segments[i].get("id", i)) == seg_id for i in idxs):
            if len(idxs) < MIN_COMPONENT_SEGMENTS:
                raise ValueError(
                    f"segment {seg_id} lies in a {len(idxs)}-segment fragment, "
                    "not a tree; check the root segment id")
            return [segments[i] for i in idxs]
    raise ValueError(f"no component contains segment id {seg_id}")


def root_check(segments: list[dict[str, Any]], seg_id: int) -> dict[str, Any]:
    """Confirm the named root really is one: top order, and a free end.

    A root segment reaches the ostium, so one of its nodes has degree 1, and it
    carries the component's maximum Strahler order. Reported rather than
    enforced -- a manually re-rooted graph may legitimately fail one test -- but
    a silent mismatch would move the exclusion onto an ordinary vessel."""
    deg = node_degrees(segments)
    max_order = max(int(s.get("strahler", 0)) for s in segments)
    for i, s in enumerate(segments):
        if int(s.get("id", i)) != seg_id:
            continue
        d1, d2 = deg.get(s["node1"], 0), deg.get(s["node2"], 0)
        order = int(s.get("strahler", 0))
        info = {
            "seg_id": seg_id,
            "strahler": order,
            "component_max_strahler": max_order,
            "node_degrees": [d1, d2],
            "is_top_order": order == max_order,
            "has_free_end": d1 == 1 or d2 == 1,
        }
        if not (info["is_top_order"] and info["has_free_end"]):
            print(f"[combined][WARN] segment {seg_id} does not look like a root: "
                  f"order {order} of {max_order}, node degrees {d1}/{d2}")
        return info
    raise ValueError(f"segment {seg_id} is not in this component")


# -- one tree -----------------------------------------------------------------


def build_tree(
    label: str,
    graph_xml: Path,
    root_seg: int,
    cfd_dir: Path | None,
    cfd_stem: str | None,
    bif_skip: int,
    radius_factor: float,
    pressure_offset: float,
    inflow_kgs: float | None = None,
    keep_root: bool = False,
    use_elements: bool = False,
) -> dict[str, Any]:
    """Anatomy + haemodynamics for one tree.

    The root is removed *before* :func:`segment_geometry`, so node degrees are
    recomputed without it and the junction it used to form is no longer masked
    as a bifurcation -- the same ordering ``strahler_analysis`` uses for
    ``--exclude-seg``.

    ``keep_root`` retains it instead. The root is the sole member of the tree's
    top Strahler order, so dropping it deletes that order from every table; a
    morphometric run that wants the ostial trunk reported -- order 5 on a
    5-order tree -- has to keep it and accept that an ex vivo cannulated stub
    carries the radius the segmentation gave it."""
    print(f"[combined] {label}: {graph_xml}")
    nodes, points, segments = parse_xml(graph_xml)
    comp = component_containing(segments, root_seg)
    info = root_check(comp, root_seg)
    kept = comp if keep_root else filter_segments(comp, {root_seg})
    verb = ("keeping" if keep_root else "dropping")
    print(f"[combined]   component {len(comp)} segments -> {len(kept)} after "
          f"{verb} root segment {root_seg} (order {info['strahler']})")

    records, seg_coords, seg_radii = segment_geometry(nodes, points, kept, bif_skip)

    n_hit = 0
    if cfd_stem and cfd_dir:
        wall_p = cfd_dir / f"{cfd_stem}_wall.npz"
        vol_p = cfd_dir / f"{cfd_stem}_volume.npz"
        wall = load_cfd_npz(wall_p) if wall_p.exists() else None
        volume = load_cfd_npz(vol_p) if vol_p.exists() else None
        if wall is None and volume is None:
            print(f"[combined][WARN] no export for '{cfd_stem}' in {cfd_dir}")
        else:
            n_hit = apply_cfd_run(records, seg_coords, seg_radii, wall, volume,
                                  label, radius_factor, pressure_offset)
            print(f"[combined]   {label}: wall="
                  f"{0 if wall is None else len(wall['X'])} vol="
                  f"{0 if volume is None else len(volume['X'])} "
                  f"-> {n_hit}/{len(records)} segments covered")

    # `apply_cfd_run` stamps `tree` only on segments it touched. Every record
    # here belongs to this tree whether or not the solve reached it, and the
    # combined table is split on this column.
    for rec in records:
        rec.tree = label

    # Collapsing happens *after* the CFD pass: a mesh node is assigned to the
    # segment whose centreline it sits nearest, which needs the segments, and an
    # element then inherits the length-weighted average of its segments' fields.
    raw = records
    if use_elements:
        records = collapse_to_elements(records, strahler_elements(kept))
        print(f"[combined]   {len(raw)} segments -> {len(records)} Strahler "
              f"elements (maximal same-order runs)")

    bif_per_order, bif_total, degree_hist = count_bifurcations(nodes, kept)
    # The root owns a degree-1 node when it is kept, but it is where the tree
    # starts, not a terminal branch.
    term_per_order, term_total = count_terminals(
        kept, {root_seg} if keep_root else None)
    print(f"[combined]   {label}: {bif_total} bifurcations, "
          f"{term_total} terminal branches")
    return {
        "label": label,
        "graph": str(graph_xml),
        "root_seg": root_seg,
        "root_info": info,
        "keep_root": keep_root,
        "cfd_dir": str(cfd_dir) if cfd_dir else None,
        "cfd_stem": cfd_stem,
        "inflow_kgs": inflow_kgs,
        "records": records,
        "raw_records": raw,
        "use_elements": use_elements,
        "n_elements": len(records) if use_elements else None,
        "n_component_segments": len(comp),
        "n_segments": len(raw),
        "n_rows": len(records),
        "n_cfd_covered": n_hit,
        "bif_per_order": bif_per_order,
        "bif_total": bif_total,
        "term_per_order": term_per_order,
        "term_total": term_total,
        "degree_hist": degree_hist,
    }


def cfd_records(records: list[SegmentRecord]) -> list[SegmentRecord]:
    """Segments the solver actually reached -- the rows a CFD statistic may use."""
    return [r for r in records if r.n_wall_nodes or r.n_volume_nodes]


# -- output -------------------------------------------------------------------


def write_tree_dir(
    out_dir: Path,
    records: list[SegmentRecord],
    bif_per_order: dict[int, int],
    bif_total: int,
    shared_edges: np.ndarray,
    provenance: dict[str, Any],
    n_radius_bins: int,
    log_bins: bool,
    raw_records: list[SegmentRecord] | None = None,
    terminals: dict[int, int] | None = None,
) -> None:
    """Write one analysis directory in the layout ``strahler_plots`` expects.

    ``records`` are the rows every table is built from -- segments, or elements
    when the run collapsed them. ``segments.csv`` always holds those rows so the
    plotting module needs no mode flag; when they are elements the underlying
    per-segment table is written alongside as ``segments_raw.csv``."""
    out_dir.mkdir(parents=True, exist_ok=True)
    cols = [f.name for f in fields(SegmentRecord)]
    cfd = cfd_records(records)

    write_csv(out_dir / "segments.csv", [asdict(r) for r in records], cols)
    if raw_records is not None and len(raw_records) != len(records):
        write_csv(out_dir / "segments_raw.csv", [asdict(r) for r in raw_records], cols)
    if cfd:
        write_csv(out_dir / "segments_cfd.csv", [asdict(r) for r in cfd], cols)

    write_csv(out_dir / "by_strahler.csv",
              aggregate_by_strahler(records, bif_per_order))
    if cfd:
        write_csv(out_dir / "by_strahler_cfd.csv",
                  aggregate_by_strahler(cfd, bif_per_order))

    own_edges = radius_bin_edges(records, n_radius_bins, log_bins)
    write_csv(out_dir / "by_radius_bin.csv",
              aggregate_by_radius(records, own_edges))
    write_csv(out_dir / "by_radius_bin_shared.csv",
              aggregate_by_radius(records, shared_edges))

    write_csv(out_dir / "bifurcations.csv", [
        {"strahler": o, "n_bifurcations": bif_per_order.get(o, 0)}
        for o in sorted(bif_per_order)
    ] + [{"strahler": "total", "n_bifurcations": bif_total}])

    write_csv(out_dir / "morphometry_table.csv",
              morphometry_table(records, bif_per_order, terminals),
              TABLE_COLUMNS)

    prov = dict(provenance)
    prov["radius_bin_edges_mm"] = [float(e) for e in own_edges]
    prov["shared_radius_bin_edges_mm"] = [float(e) for e in shared_edges]
    (out_dir / "provenance.json").write_text(json.dumps(prov, indent=2))


# -- comparison ---------------------------------------------------------------


def _index(rows: list[dict[str, Any]], key: str) -> dict[Any, dict[str, Any]]:
    return {r[key]: r for r in rows}


def _f(v: Any) -> float:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return math.nan
    return f


def _ok(v: float) -> bool:
    return isinstance(v, float) and math.isfinite(v)


def merge_rows(a: list[dict[str, Any]], b: list[dict[str, Any]],
               key: str) -> list[dict[str, Any]]:
    """Anatomy row + CFD row for the same order/bin, CFD columns winning.

    Anatomy is counted over every segment, haemodynamics only over the ones the
    solver reached, so the two carry different ``n``. The per-metric ``_n``
    columns preserve that difference; the row-level ``n_vessels`` stays the
    anatomical count, which is what "how many vessels of this order" means."""
    ia, ib = _index(a, key), _index(b, key)
    out: list[dict[str, Any]] = []
    for k in sorted(set(ia) | set(ib)):
        row = dict(ia.get(k, {key: k}))
        for col, val in ib.get(k, {}).items():
            if col == key or col in _ROW_COLS:
                continue
            if col.startswith(_CFD_PREFIXES):
                row[col] = val
        out.append(row)
    return out


def compare_tables(
    per_tree: dict[str, list[dict[str, Any]]],
    combined: list[dict[str, Any]],
    key: str,
    extra_cols: list[str],
) -> list[dict[str, Any]]:
    """One row per order (or bin) carrying left, right and combined side by side.

    Two derived columns say what the comparison is for:

    ``<metric>_right_vs_left_pct``
        the contrast between the trees at that order or calibre -- the reason
        pooling can move a number at all.
    ``<metric>_pool_delta_pct``
        how far the pooled median sits from the n-weighted mean of the two
        per-tree medians. A pooled median is not the weighted mean of the parts,
        so this is not an error term: it measures how much of the combined
        figure comes from the mixing rather than from either tree."""
    idx = {name: _index(rows, key) for name, rows in per_tree.items()}
    idx["combined"] = _index(combined, key)
    names = ("left", "right", "combined")
    keys = sorted({k for m in idx.values() for k in m})

    out: list[dict[str, Any]] = []
    for k in keys:
        row: dict[str, Any] = {key: k}
        for col in extra_cols:
            for name in names:
                row[f"{col}_{name}"] = idx.get(name, {}).get(k, {}).get(col, "")
        for name in names:
            row[f"n_vessels_{name}"] = idx.get(name, {}).get(k, {}).get("n_vessels", 0)

        for metric, _label in METRICS:
            got: dict[str, float] = {}
            for name in names:
                r = idx.get(name, {}).get(k, {})
                # q25/q75 as well as the IQR: the quartiles are asymmetric about
                # the median for these skewed distributions, so a figure needs
                # both arms, not one half-width.
                for stat in ("n", "median", "iqr", "q25", "q75", "mean", "sd"):
                    v = r.get(f"{metric}_{stat}", "")
                    row[f"{metric}_{stat}_{name}"] = v
                    if stat in ("median", "n"):
                        got[f"{stat}_{name}"] = _f(v)

            nl, nr = got.get("n_left", math.nan), got.get("n_right", math.nan)
            ml, mr = got.get("median_left", math.nan), got.get("median_right", math.nan)
            mc = got.get("median_combined", math.nan)

            expected = math.nan
            if _ok(ml) and _ok(mr) and _ok(nl) and _ok(nr) and nl + nr > 0:
                expected = (nl * ml + nr * mr) / (nl + nr)
            elif _ok(ml) and not _ok(mr):
                expected = ml
            elif _ok(mr) and not _ok(ml):
                expected = mr
            row[f"{metric}_median_expected"] = (
                f"{expected:.6g}" if _ok(expected) else "")
            row[f"{metric}_pool_delta_pct"] = (
                f"{100.0 * (mc - expected) / abs(expected):.4g}"
                if _ok(expected) and _ok(mc) and expected != 0 else "")
            row[f"{metric}_right_vs_left_pct"] = (
                f"{100.0 * (mr - ml) / abs(ml):.4g}"
                if _ok(ml) and _ok(mr) and ml != 0 else "")
        out.append(row)
    return out


def _fmt(v: Any) -> str:
    f = _f(v)
    return "--" if not math.isfinite(f) else f"{f:.3g}"


def _pct(v: Any) -> str:
    f = _f(v)
    return "--" if not math.isfinite(f) else f"{f:+.1f}%"


def _operating_point(trees: list[dict[str, Any]], density: float) -> list[str]:
    """The prescribed inflow of each solve, and the split between them.

    Written out because it is the confounder in every left-vs-right haemodynamic
    contrast below: wall shear scales with the flow the solver was told to push
    through the tree, so a difference between the trees is a difference in
    anatomy *and* in the prescribed split, never anatomy alone."""
    have = [t for t in trees if t.get("inflow_kgs")]
    if not have:
        return []
    L = ["## Operating point\n",
         f"Steady-state solves, blood density {density:g} kg m^-3. "
         "Each tree has its inlet mass flow prescribed and its outlets set to "
         "flow fractions of it.\n",
         "| tree | inlet mass flow (kg s^-1) | inlet flow (mL min^-1) | share |",
         "|---|---|---|---|"]
    total = sum(float(t["inflow_kgs"]) for t in have)
    for t in have:
        m = float(t["inflow_kgs"])
        q = m / density * KGS_TO_ML_MIN
        share = f"{100.0 * m / total:.1f}%" if total > 0 else "--"
        L.append(f"| {t['label']} | {m:.6g} | {q:.1f} | {share} |")
    L.append("")
    L.append("**The flow split is an input, not a result.** Wall shear and "
             "velocity scale with it, so a left-vs-right haemodynamic contrast "
             "below reports the combined effect of the two anatomies *and* the "
             "flow each was given. Only the morphometry is free of it.\n")
    return L


def write_summary_md(
    path: Path,
    trees: list[dict[str, Any]],
    strahler_rows: list[dict[str, Any]],
    radius_rows: list[dict[str, Any]],
    shared_edges: np.ndarray,
    density: float = BLOOD_DENSITY_KG_M3,
) -> None:
    """Prose summary: what pooling changes, and where it must not be read."""
    L: list[str] = []
    L.append("# Left, right and combined -- what pooling changes\n")

    L.append("## Inputs\n")
    els = any(t.get("use_elements") for t in trees)
    L.append("| tree | graph | root segment | segments |"
             + (" elements |" if els else "")
             + " terminal branches | bifurcations | CFD-covered |")
    L.append("|---|---|---|---|---|---|---|" + ("---|" if els else ""))
    for t in trees:
        state = "kept" if t.get("keep_root") else "dropped"
        L.append(f"| {t['label']} | `{Path(t['graph']).name}` | "
                 f"{t['root_seg']} (order {t['root_info']['strahler']}, "
                 f"{state}) | {t['n_segments']} |"
                 + (f" {t['n_elements']} |" if els else "")
                 + f" {t['term_total']} | {t['bif_total']} |"
                 + f" {t['n_cfd_covered']} |")
    L.append(f"| **total** | | | **{sum(t['n_segments'] for t in trees)}** |"
             + (f" **{sum(t['n_elements'] or 0 for t in trees)}** |" if els else "")
             + f" **{sum(t['term_total'] for t in trees)}** |"
             + f" **{sum(t['bif_total'] for t in trees)}** |"
             + f" **{sum(t['n_cfd_covered'] for t in trees)}** |")
    L.append("")
    if els:
        L.append("Every table below has **one row per Strahler element** -- a "
                 "maximal run of consecutive same-order segments, which is the "
                 "vessel. The graph cuts a trunk at every side branch, so "
                 "summing a per-vessel quantity such as cross-sectional area "
                 "over segments counts the same lumen once per side branch, and "
                 "because side branches concentrate on the large vessels it "
                 "biases the top orders hardest.\n")
    if all(t.get("keep_root") for t in trees):
        L.append("Root segments are **kept**, so the top Strahler order of each "
                 "tree is its ostial trunk. That order holds one vessel per "
                 "tree: read it as a single vessel, not a population, and note "
                 "that an ex vivo ostium is cannulated, so its radius is the "
                 "least trustworthy in the table.\n")
    elif any(t.get("keep_root") for t in trees):
        L.append("The root is kept for one tree and dropped for the other, so "
                 "the top orders of the two are not comparable.\n")
    else:
        L.append("Root segments are excluded from every statistic below, so the "
                 "top Strahler order of each tree is its highest *non-ostial* "
                 "order.\n")

    L.extend(_operating_point(trees, density))

    L.append("## Morphometry by Strahler order\n")
    L.append("| order | n left | n right | radius left (mm) | radius right (mm) "
             "| right vs left | radius combined (mm) | pooling shift |")
    L.append("|---|---|---|---|---|---|---|---|")
    for r in strahler_rows:
        L.append(
            f"| {r['strahler']} | {r['n_vessels_left']} | {r['n_vessels_right']} | "
            f"{_fmt(r.get('radius_mean_mm_median_left'))} | "
            f"{_fmt(r.get('radius_mean_mm_median_right'))} | "
            f"{_pct(r.get('radius_mean_mm_right_vs_left_pct'))} | "
            f"{_fmt(r.get('radius_mean_mm_median_combined'))} | "
            f"{_pct(r.get('radius_mean_mm_pool_delta_pct'))} |")
    L.append("")

    L.append("## Wall shear by Strahler order\n")
    L.append("| order | WSS left (Pa) | WSS right (Pa) | right vs left | "
             "WSS combined (Pa) | pooling shift |")
    L.append("|---|---|---|---|---|---|")
    for r in strahler_rows:
        L.append(
            f"| {r['strahler']} | {_fmt(r.get('wss_mean_pa_median_left'))} | "
            f"{_fmt(r.get('wss_mean_pa_median_right'))} | "
            f"{_pct(r.get('wss_mean_pa_right_vs_left_pct'))} | "
            f"{_fmt(r.get('wss_mean_pa_median_combined'))} | "
            f"{_pct(r.get('wss_mean_pa_pool_delta_pct'))} |")
    L.append("")

    L.append("## Wall shear by radius (shared bins)\n")
    L.append("Radius is a physical calibre, so a bin means the same vessel size "
             "in both trees -- unlike a Strahler order, whose meaning depends on "
             "how many generations sit below it.\n")
    L.append("| radius bin (mm) | n left | n right | WSS left (Pa) | "
             "WSS right (Pa) | right vs left | WSS combined (Pa) |")
    L.append("|---|---|---|---|---|---|---|")
    for r in radius_rows:
        b = int(r["bin"])
        lo, hi = float(shared_edges[b]), float(shared_edges[b + 1])
        L.append(
            f"| {lo:.3f}-{hi:.3f} | {r['n_vessels_left']} | {r['n_vessels_right']} | "
            f"{_fmt(r.get('wss_mean_pa_median_left'))} | "
            f"{_fmt(r.get('wss_mean_pa_median_right'))} | "
            f"{_pct(r.get('wss_mean_pa_right_vs_left_pct'))} | "
            f"{_fmt(r.get('wss_mean_pa_median_combined'))} |")
    L.append("")

    L.append("## Read with care\n")
    L.append("- **Pressure must not be pooled.** Left and right are separate "
             "solves and CFX references static pressure to each solve's own "
             "outlets, pinned at 0 Pa. The combined pressure columns exist in "
             "the tables only to show the offset between the two solves; they "
             "are not a coronary pressure distribution.")
    L.append("- **Strahler order is relative to its own tree.** Order *n* in a "
             "4-order tree sits one generation nearer the ostium than order *n* "
             "in a 5-order tree, so pooling by raw order number compares "
             "unequal anatomical positions. The radius tables are the "
             "assumption-free comparison.")
    if len({str(t["graph"]) for t in trees}) > 1:
        L.append("- **The two trees came off different graphs** ("
                 + ", ".join(f"{t['label']} `{Path(t['graph']).name}`"
                             for t in trees)
                 + "), so they are truncated at different distal cut-offs and "
                   "their order-1 populations are not the same population.")
    else:
        L.append("- **Both trees came off the same graph** "
                 f"(`{Path(trees[0]['graph']).name}`), so their distal cut-off "
                 "is common and the order-1 populations are comparable.")
    if els:
        L.append("- **An element's haemodynamics are length-weighted averages** "
                 "of its segments, so a covered element may be only partly "
                 "covered by the mesh. The per-segment table is kept as "
                 "`segments_raw.csv` in each directory.")
    L.append("- **`flow_speed_ml_min` is not a flux** and does not conserve "
             "across a bifurcation; the axial pair is the transport estimate. "
             "Wall shear is a magnitude from a steady-state solve, so no OSI or "
             "RRT is defined for it.")
    path.write_text("\n".join(L) + "\n", encoding="utf-8")


# -- driver -------------------------------------------------------------------


def run(
    left: dict[str, Any],
    right: dict[str, Any],
    out_dir: Path,
    bif_skip: int = BIF_SKIP_POINTS,
    radius_factor: float = DEFAULT_RADIUS_FACTOR,
    n_radius_bins: int = 8,
    log_bins: bool = True,
    pressure_offset: float = 0.0,
    density_kgm3: float = BLOOD_DENSITY_KG_M3,
    keep_root: bool = False,
    use_elements: bool = False,
) -> dict[str, Any]:
    """Build the three analyses and the comparison; returns a summary dict."""
    out_dir.mkdir(parents=True, exist_ok=True)

    trees = [
        build_tree("left", Path(left["graph"]), int(left["root_seg"]),
                   left["cfd_dir"], left["run"],
                   bif_skip, radius_factor, pressure_offset,
                   left.get("inflow_kgs"), keep_root, use_elements),
        build_tree("right", Path(right["graph"]), int(right["root_seg"]),
                   right["cfd_dir"], right["run"],
                   bif_skip, radius_factor, pressure_offset,
                   right.get("inflow_kgs"), keep_root, use_elements),
    ]
    pooled = [r for t in trees for r in t["records"]]
    pooled_raw = [r for t in trees for r in t["raw_records"]]

    # One edge set over the pooled radii: three tables binned on three different
    # ranges cannot be read against each other bin for bin.
    shared_edges = radius_bin_edges(pooled, n_radius_bins, log_bins)
    print(f"[combined] shared radius bins: {shared_edges[0]:.4f} .. "
          f"{shared_edges[-1]:.4f} mm, {len(shared_edges) - 1} bins")

    common = {
        "bif_skip_points": bif_skip,
        "radius_factor": radius_factor,
        "log_radius_bins": log_bins,
        "pressure_offset_mmhg": pressure_offset,
        "density_kg_m3": density_kgm3,
        "units": {"length": "mm", "radius": "mm", "area": "mm^2",
                  "volume": "mm^3", "pressure": "mmHg", "wall_shear": "Pa",
                  "velocity": "m/s", "flow": "mL/min"},
        "row_unit": "strahler_element" if use_elements else "graph_segment",
        "row_unit_note": (
            "Rows are Strahler elements -- maximal runs of consecutive "
            "same-order segments, i.e. one row per vessel. A trunk is split by "
            "every side branch that joins it, so summing a per-vessel quantity "
            "over segments would count the same lumen once per side branch."
            if use_elements else
            "Rows are graph segments. A vessel crossed by side branches "
            "contributes one row per piece, so `csa_mm2_sum` and `n_vessels` "
            "over-count the large orders; --elements collapses them."),
    }

    def _inflow(t: dict[str, Any]) -> dict[str, Any]:
        m = t.get("inflow_kgs")
        if not m:
            return {"inlet_mass_flow_kg_s": None, "inlet_flow_ml_min": None}
        return {"inlet_mass_flow_kg_s": float(m),
                "inlet_flow_ml_min": float(m) / density_kgm3 * KGS_TO_ML_MIN}

    for t in trees:
        write_tree_dir(
            out_dir / t["label"], t["records"], t["bif_per_order"], t["bif_total"],
            shared_edges,
            {**common,
             "mode": f"{t['label']} tree alone",
             "graph": t["graph"],
             "root_segment": t["root_seg"],
             "root_segment_excluded": None if t["keep_root"] else t["root_seg"],
             "root_segment_included": t["keep_root"],
             "root_check": t["root_info"],
             "cfd_dir": t["cfd_dir"],
             "cfd_run": t["cfd_stem"],
             **_inflow(t),
             "n_component_segments": t["n_component_segments"],
             "n_segments": t["n_segments"],
             "n_elements": t["n_elements"],
             "n_terminal_branches": t["term_total"],
             "n_terminal_branches_per_order": {
                 str(k): v for k, v in sorted(t["term_per_order"].items())},
             "n_cfd_covered": t["n_cfd_covered"],
             "n_bifurcations_total": t["bif_total"],
             "node_degree_histogram": {
                 str(k): v for k, v in sorted(t["degree_hist"].items())}},
            n_radius_bins, log_bins, t["raw_records"], t["term_per_order"])

    comb_bif: dict[int, int] = defaultdict(int)
    for t in trees:
        for order, count in t["bif_per_order"].items():
            comb_bif[order] += count
    comb_bif = dict(comb_bif)
    comb_total = sum(t["bif_total"] for t in trees)

    comb_term: dict[int, int] = defaultdict(int)
    for t in trees:
        for order, count in t["term_per_order"].items():
            comb_term[order] += count
    comb_term = dict(comb_term)
    comb_term_total = sum(t["term_total"] for t in trees)

    write_tree_dir(
        out_dir / "combined", pooled, comb_bif, comb_total, shared_edges,
        {**common,
         "mode": "left + right pooled",
         "graphs": {t["label"]: t["graph"] for t in trees},
         "root_segments": {t["label"]: t["root_seg"] for t in trees},
         "root_segments_excluded": {
             t["label"]: t["root_seg"] for t in trees if not t["keep_root"]},
         "root_segments_included": {
             t["label"]: t["root_seg"] for t in trees if t["keep_root"]},
         "cfd_runs": [{"tree": t["label"], "stem": t["cfd_stem"],
                       "dir": t["cfd_dir"], **_inflow(t)} for t in trees],
         "n_segments": len(pooled_raw),
         "n_elements": len(pooled) if use_elements else None,
         "n_segments_per_tree": {t["label"]: t["n_segments"] for t in trees},
         "n_cfd_covered": sum(t["n_cfd_covered"] for t in trees),
         "n_bifurcations_total": comb_total,
         "n_terminal_branches": comb_term_total,
         "n_terminal_branches_per_tree": {
             t["label"]: t["term_total"] for t in trees},
         "n_terminal_branches_per_order": {
             str(k): v for k, v in sorted(comb_term.items())},
         "poolable_metrics": POOLABLE,
         "non_poolable_metrics": NON_POOLABLE,
         "non_poolable_note": (
             "Left and right are separate CFX solves, each with its static "
             "pressure referenced to its own outlets at 0 Pa. Pooled pressure "
             "statistics mix two references and are not physical.")},
        n_radius_bins, log_bins, pooled_raw, comb_term)

    # -- comparison tables ----------------------------------------------------
    cmp_dir = out_dir / "comparison"
    cmp_dir.mkdir(parents=True, exist_ok=True)

    merged_tree = {
        t["label"]: merge_rows(
            aggregate_by_strahler(t["records"], t["bif_per_order"]),
            aggregate_by_strahler(cfd_records(t["records"]), t["bif_per_order"]),
            "strahler")
        for t in trees
    }
    merged_comb = merge_rows(
        aggregate_by_strahler(pooled, comb_bif),
        aggregate_by_strahler(cfd_records(pooled), comb_bif),
        "strahler")
    strahler_cmp = compare_tables(merged_tree, merged_comb, "strahler",
                                  ["n_bifurcations", "total_length_mm"])
    write_csv(cmp_dir / "compare_by_strahler.csv", strahler_cmp)

    rad_tree = {
        t["label"]: merge_rows(
            aggregate_by_radius(t["records"], shared_edges),
            aggregate_by_radius(cfd_records(t["records"]), shared_edges),
            "bin")
        for t in trees
    }
    rad_comb = merge_rows(
        aggregate_by_radius(pooled, shared_edges),
        aggregate_by_radius(cfd_records(pooled), shared_edges),
        "bin")
    radius_cmp = compare_tables(rad_tree, rad_comb, "bin",
                                ["radius_lo_mm", "radius_hi_mm", "radius_mid_mm"])
    write_csv(cmp_dir / "compare_by_radius.csv", radius_cmp)

    # The pooling-effect table alone, so the headline numbers are not buried in
    # the several-hundred columns of the full comparison.
    pool_rows: list[dict[str, Any]] = []
    for scope, rows, kcol in (("strahler", strahler_cmp, "strahler"),
                              ("radius_bin", radius_cmp, "bin")):
        for r in rows:
            for metric, label in METRICS:
                if metric in NON_POOLABLE:
                    continue
                pool_rows.append({
                    "scope": scope,
                    "group": r[kcol],
                    "metric": metric,
                    "label": label,
                    "n_left": r.get(f"{metric}_n_left", ""),
                    "n_right": r.get(f"{metric}_n_right", ""),
                    "median_left": r.get(f"{metric}_median_left", ""),
                    "median_right": r.get(f"{metric}_median_right", ""),
                    "right_vs_left_pct": r.get(f"{metric}_right_vs_left_pct", ""),
                    "median_combined": r.get(f"{metric}_median_combined", ""),
                    "median_expected": r.get(f"{metric}_median_expected", ""),
                    "pool_delta_pct": r.get(f"{metric}_pool_delta_pct", ""),
                })
    write_csv(cmp_dir / "pooling_effect.csv", pool_rows)

    write_summary_md(cmp_dir / "summary.md", trees, strahler_cmp, radius_cmp,
                     shared_edges, density_kgm3)

    print(f"[combined] wrote {out_dir}\\{{left,right,combined,comparison}}")
    return {
        "out_dir": str(out_dir),
        "trees": {t["label"]: {"n_segments": t["n_segments"],
                               "n_elements": t["n_elements"],
                               "n_terminal_branches": t["term_total"],
                               "n_cfd_covered": t["n_cfd_covered"],
                               "root_seg": t["root_seg"],
                               "graph": t["graph"],
                               **_inflow(t)} for t in trees},
        "n_pooled_segments": len(pooled),
        "shared_radius_bin_edges_mm": [float(e) for e in shared_edges],
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--left-graph", type=Path, required=True,
                    help="graph containing the left tree")
    ap.add_argument("--left-root-seg", type=int, required=True,
                    help="segment id of the left ostial/root segment; it names "
                         "the component and is dropped from every statistic")
    ap.add_argument("--left-run", default=None, help="cfx_extract export stem")
    ap.add_argument("--left-cfd-dir", type=Path, default=None)
    ap.add_argument("--left-inflow-kgs", type=float, default=None,
                    help="the solve's prescribed inlet mass flow, kg/s. Recorded "
                         "and reported only: it is the confounder behind every "
                         "left-vs-right haemodynamic contrast, so the tables "
                         "should state it")
    ap.add_argument("--right-graph", type=Path, required=True)
    ap.add_argument("--right-root-seg", type=int, required=True)
    ap.add_argument("--right-run", default=None)
    ap.add_argument("--right-cfd-dir", type=Path, default=None)
    ap.add_argument("--right-inflow-kgs", type=float, default=None)
    ap.add_argument("--density-kgm3", type=float, default=BLOOD_DENSITY_KG_M3,
                    help="blood density used by the solves; only converts the "
                         "reported inlet mass flow to mL/min")
    ap.add_argument("--out", type=Path, required=True, help="output directory")
    ap.add_argument("--bif-skip", type=int, default=BIF_SKIP_POINTS,
                    help="contours excluded either side of a bifurcation node")
    ap.add_argument("--radius-factor", type=float, default=DEFAULT_RADIUS_FACTOR,
                    help="CFD node accepted within this multiple of local radius")
    ap.add_argument("--radius-bins", type=int, default=8)
    ap.add_argument("--linear-bins", action="store_true",
                    help="use linear rather than logarithmic radius bins")
    ap.add_argument("--pressure-offset-mmhg", type=float, default=0.0)
    ap.add_argument("--elements", action="store_true",
                    help="aggregate over Strahler elements -- maximal runs of "
                         "consecutive same-order segments, one row per vessel -- "
                         "rather than over graph segments. Without it a trunk "
                         "split by its side branches contributes one row per "
                         "piece and `csa_mm2_sum` counts the same lumen many "
                         "times over")
    ap.add_argument("--keep-root", action="store_true",
                    help="keep the ostial root segment of each tree instead of "
                         "dropping it, so the tree's top Strahler order (order "
                         "5 on the left tree) appears in every table and figure")
    args = ap.parse_args(argv)

    run(
        left={"graph": args.left_graph, "root_seg": args.left_root_seg,
              "run": args.left_run, "cfd_dir": args.left_cfd_dir,
              "inflow_kgs": args.left_inflow_kgs},
        right={"graph": args.right_graph, "root_seg": args.right_root_seg,
               "run": args.right_run, "cfd_dir": args.right_cfd_dir,
               "inflow_kgs": args.right_inflow_kgs},
        out_dir=args.out,
        bif_skip=args.bif_skip,
        radius_factor=args.radius_factor,
        n_radius_bins=args.radius_bins,
        log_bins=not args.linear_bins,
        pressure_offset=args.pressure_offset_mmhg,
        density_kgm3=args.density_kgm3,
        keep_root=args.keep_root,
        use_elements=args.elements,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
