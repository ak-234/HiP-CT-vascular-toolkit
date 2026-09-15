"""Segment-based contour WSS sampling for paired cross-ratio comparison.

Samples wall-shear-stress (WSS) along the annotated main vessels in **3 mm
centreline segments**, reporting per segment the **min and max "predominant" WSS**
from a 90 deg rotating arc window. Because the segments are defined once on a
**shared reference centreline** (the full un-pruned tree), every pruned-model CFD
cloud yields the *same number of measurements at the same locations* — the
prerequisite for a paired statistical test of "does ratio-pruning change the WSS?"

Point selection (no double counting): each wall-cloud point is assigned to its
single **nearest centreline point** (so it is used at most once) and kept only if
it lies within ``radius_factor * local_radius`` of that contour (excludes points
from nearby other branches). The kept points are pooled per 3 mm segment and the
arc window (:func:`~coronary_sdf.wss_postprocess.arc_window_min_max`) is swept over
their circumferential angles.

Outputs into ``--out``:

* ``wss_seg_long.csv``     — one row per (vessel, segment, ratio), all metrics;
* ``wss_seg_max_wide.csv`` — row per (vessel, seg_idx), a ``wss_<ratio>`` column
  holding the max windowed-mean WSS (paired, equal length);
* ``wss_seg_min_wide.csv`` — same for the min windowed-mean WSS.

Drop the wide tables into e.g. ``scipy.stats.wilcoxon`` for the paired test.

Usage::

    python -m coronary_sdf.wss_contour_compare \
        --xml base.am.xml --sidecar epicardial.json \
        --ratio full=full_wall.csv --ratio r2=ratio2_wall.csv \
        --segment-mm 3 --out wss_compare --viz
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Any, cast

import numpy as np
from scipy.spatial import KDTree

from .flow_fractions import BIF_SKIP_POINTS
from .splines import compute_frenet_frame
from .wss_postprocess import (
    _polyline_arclength,
    arc_window_min_max,
    build_vessel_centrelines,
    check_alignment,
    read_wss_csv,
    resolve_coord_scale,
)

LONG_FIELDS = [
    "vessel", "seg_idx", "s_start_mm", "s_end_mm", "has_bif", "ratio", "n_wall_pts",
    "ring_mean_wss", "wss_max_window", "angle_max_deg",
    "wss_min_window", "angle_min_deg",
]
ANCHOR_FIELDS = ["vessel", "seg_idx", "s_start_mm", "s_end_mm", "has_bif"]


def parse_ratio_arg(spec: str) -> tuple[str, Path]:
    """Parse a ``label=path`` ``--ratio`` value. If ``=`` is absent, the label is
    inferred from the CSV's parent directory name (or its stem)."""
    if "=" in spec:
        label, path = spec.split("=", 1)
        label, path = label.strip(), path.strip()
    else:
        path = spec.strip()
        p = Path(path)
        label = p.parent.name or p.stem
    if not label:
        raise SystemExit(f"[COMPARE][ERROR] could not derive a label from --ratio {spec!r}")
    return label, Path(path)


def filter_vessels(
    centrelines: dict[str, dict[str, np.ndarray]], names: list[str] | None
) -> tuple[dict[str, dict[str, np.ndarray]], list[str]]:
    """Keep only the named vessels (case-insensitive exact match). Returns
    ``(filtered, missing)`` where ``missing`` lists requested names not present.
    With no ``names`` the centrelines are returned unchanged."""
    if not names:
        return centrelines, []
    want = {n.strip().lower() for n in names if n.strip()}
    filt = {v: d for v, d in centrelines.items() if v.lower() in want}
    present = {v.lower() for v in centrelines}
    missing = sorted(w for w in want if w not in present)
    return filt, missing


def vessels_without_data(long_rows: list[dict[str, Any]]) -> list[str]:
    """Vessels whose ``wss_max_window`` is NaN for every row (all ratios) — i.e.
    no wall points fell near them (e.g. the wrong side's cloud)."""
    all_nan: dict[str, bool] = {}
    for r in long_rows:
        v = r["vessel"]
        is_nan = isinstance(r["wss_max_window"], float) and math.isnan(r["wss_max_window"])
        all_nan[v] = all_nan.get(v, True) and is_nan
    return sorted(v for v, nan in all_nan.items() if nan)


def build_segment_table(
    centrelines: dict[str, dict[str, np.ndarray]], segment_mm: float,
    trim_proximal_bif: bool = True,
    competitor_pts: np.ndarray | None = None,
) -> dict[str, Any]:
    """Partition every vessel centreline into fixed ``segment_mm`` arc-length bins.

    Built **once** and shared across ratios, so the segment set (and therefore the
    measurement count/locations) is identical for every cloud. A
    **parallel-transported** frame is computed at every centreline point (the
    normal is carried smoothly along each vessel) so the circumferential angle is
    referenced consistently along/across segments.

    When ``trim_proximal_bif`` is set, the leading run of bifurcation-zone points
    (e.g. the LM junction at a vessel's ostium) is dropped so the first segment
    starts outside the bifurcation. Returns:

    * ``cl_pts (M,3)``, ``cl_radius (M,)`` — all vessels' centreline points;
    * ``cl_tan / cl_nhat / cl_bhat (M,3)`` — per-point transported frame;
    * ``cl_key`` — list of ``(vessel, seg_idx)`` per centreline point;
    * ``seg_keys`` — ordered unique ``(vessel, seg_idx)`` (output row order);
    * ``seg_span``  — ``{(vessel, seg_idx): (s_start, s_end)}`` (mm);
    * ``seg_bif``   — ``{(vessel, seg_idx): has_bifurcation (bool)}``.
    """
    cl_pts: list[np.ndarray] = []
    cl_radius: list[float] = []
    cl_tan: list[np.ndarray] = []
    cl_nhat: list[np.ndarray] = []
    cl_bhat: list[np.ndarray] = []
    cl_key: list[tuple[str, int]] = []
    seg_keys: list[tuple[str, int]] = []
    seg_span: dict[tuple[str, int], tuple[float, float]] = {}
    seg_bif: dict[tuple[str, int], bool] = {}

    for v, d in centrelines.items():
        coords = np.asarray(d["coords"], dtype=np.float64)
        radii = np.asarray(d["radii"], dtype=np.float64)
        bif = np.asarray(d["bif"], dtype=bool)
        if len(coords) < 2:
            continue
        # Start outside the proximal (e.g. LM) bifurcation: drop the leading bif run.
        if trim_proximal_bif and len(bif) >= 2 and bif[0] and not bif.all():
            i0 = int(np.argmax(~bif))               # first non-bif index
            if len(coords) - i0 >= 2:
                coords, radii, bif = coords[i0:], radii[i0:], bif[i0:]
                print(f"[COMPARE] {v}: trimmed {i0} proximal bifurcation point(s)")
        s = _polyline_arclength(coords)
        seg_of = np.floor(s / float(segment_mm)).astype(int)

        # Flag a discontinuous (branching/lumped) annotation: a large jump in
        # arc length between consecutive centreline points.
        if len(s) > 1:
            gaps = np.diff(s)
            gmax = float(gaps.max())
            if gmax > max(5.0, 2.0 * float(segment_mm)):
                print(f"[COMPARE][WARN] vessel '{v}' centreline has a {gmax:.1f} mm "
                      "gap - likely a branching/lumped annotation; annotate its "
                      "branches as separate vessels for clean per-vessel stats.")

        # Per-point parallel-transported frame along the whole vessel.
        tang = np.gradient(coords, axis=0)
        prev_normal = None
        prev_tan = np.array([0.0, 0.0, 1.0])
        for i in range(len(coords)):
            nrm = np.linalg.norm(tang[i])
            t_in = tang[i] / nrm if nrm > 1e-12 else prev_tan
            prev_tan = t_in
            t_i, n_i, b_i = compute_frenet_frame(t_in, prev_normal)
            prev_normal = n_i
            cl_pts.append(coords[i])
            cl_radius.append(float(radii[i]))
            cl_tan.append(t_i); cl_nhat.append(n_i); cl_bhat.append(b_i)
            cl_key.append((v, int(seg_of[i])))

        # Only bins that actually contain centreline points — a large arc-length
        # gap (discontinuous stitched vessel) leaves intermediate bins empty.
        for k in np.unique(seg_of):
            mask = seg_of == k
            seg_span[(v, int(k))] = (float(s[mask].min()), float(s[mask].max()))
            seg_bif[(v, int(k))] = bool(bif[mask].any())
            seg_keys.append((v, int(k)))

    return {
        "cl_pts": np.asarray(cl_pts, dtype=np.float64),
        "cl_radius": np.asarray(cl_radius, dtype=np.float64),
        "cl_tan": np.asarray(cl_tan, dtype=np.float64),
        "cl_nhat": np.asarray(cl_nhat, dtype=np.float64),
        "cl_bhat": np.asarray(cl_bhat, dtype=np.float64),
        "cl_key": cl_key,
        "seg_keys": seg_keys,
        "seg_span": seg_span,
        "seg_bif": seg_bif,
        "comp_pts": (np.asarray(competitor_pts, dtype=np.float64)
                     if competitor_pts is not None else np.empty((0, 3))),
    }


def _nearest_annotated(
    centers: np.ndarray,
    comp_pts: np.ndarray,
    wall_pts: np.ndarray,
    cl_radius: np.ndarray,
    radius_factor: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Nearest-centreline assignment where non-annotated (side-branch) centrelines
    can steal their own wall points. Returns ``(idx, keep)``: ``idx`` = nearest
    annotated centre index, ``keep`` = True only when an annotated centre is the
    **global** nearest (over annotated + competitors) and within the radius gate."""
    n = len(centers)
    allpts = np.vstack([centers, comp_pts]) if len(comp_pts) else centers
    dist, gidx = KDTree(allpts).query(wall_pts)
    is_ann = gidx < n
    idx = np.where(is_ann, gidx, 0)
    keep = is_ann & (dist <= radius_factor * cl_radius[idx])
    return idx, keep


def recenter_on_cloud(
    cl_pts: np.ndarray,
    cl_tan: np.ndarray,
    cl_radius: np.ndarray,
    wall_pts: np.ndarray,
    radius_factor: float,
    comp_pts: np.ndarray | None = None,
    min_pts: int = 4,
) -> np.ndarray:
    """Lateral snap of each centreline point onto the centroid of the wall points
    gated to it, **preserving axial position** (the along-tangent component of the
    shift is removed). Corrects the drift between the un-smoothed graph centreline
    and the smoothed centreline the CFD mesh was built from. Side-branch wall points
    (claimed by ``comp_pts``) are excluded so they do not pull the centroid. Returns
    new centres ``(M,3)``; a point with ``< min_pts`` gated wall points keeps its
    position."""
    if comp_pts is None:
        comp_pts = np.empty((0, 3))
    idx, keep = _nearest_annotated(cl_pts, comp_pts, wall_pts, cl_radius, radius_factor)
    ki, kp = idx[keep], wall_pts[keep]
    m = len(cl_pts)
    sums = np.zeros((m, 3))
    cnt = np.zeros(m)
    np.add.at(sums, ki, kp)
    np.add.at(cnt, ki, 1.0)
    have = cnt >= min_pts
    delta = np.zeros((m, 3))
    delta[have] = sums[have] / cnt[have, None] - cl_pts[have]
    axial = np.sum(delta * cl_tan, axis=1)              # remove along-tangent part
    lateral = delta - axial[:, None] * cl_tan
    new = cl_pts.copy()
    new[have] = cl_pts[have] + lateral[have]
    return new


def assign_and_sample(
    seg_table: dict[str, Any],
    label: str,
    wall_pts: np.ndarray,
    wall_wss: np.ndarray,
    radius_factor: float,
    arc_deg: float,
    rot_step_deg: float,
    min_pts: int,
    recenter: bool = True,
    recenter_min_pts: int = 4,
    want_assignment: bool = False,
) -> tuple[list[dict[str, Any]], np.ndarray | None, np.ndarray | None, np.ndarray]:
    """Assign each wall point to its nearest contour centre (kept within the gate,
    no duplicates), pool per 3 mm segment, sweep the arc window.

    Wall points nearer to a non-annotated side-branch centreline
    (``seg_table['comp_pts']``) are excluded (:func:`_nearest_annotated`). When
    ``recenter`` is set, the contour centres are first snapped onto the local lumen
    centroid of this cloud (:func:`recenter_on_cloud`) so the gate and radial φ are
    taken from the mesh's true axis rather than the drifted graph centreline. Emits
    one row per segment key (NaN metrics when ``< min_pts`` points) so the count
    matches every other ratio. Returns ``(rows, keep, idx, centres)``; ``keep`` and
    ``idx`` are ``None`` unless ``want_assignment`` is set, ``centres`` is always
    returned (for drift logging / viz)."""
    cl_pts = seg_table["cl_pts"]
    cl_radius = seg_table["cl_radius"]
    cl_key = seg_table["cl_key"]
    cl_tan = seg_table["cl_tan"]
    cl_nhat = seg_table["cl_nhat"]
    cl_bhat = seg_table["cl_bhat"]
    comp_pts = seg_table.get("comp_pts", np.empty((0, 3)))

    if recenter:
        centers = recenter_on_cloud(cl_pts, cl_tan, cl_radius, wall_pts,
                                    radius_factor, comp_pts, recenter_min_pts)
    else:
        centers = cl_pts

    idx, keep = _nearest_annotated(centers, comp_pts, wall_pts, cl_radius, radius_factor)
    kept_w = np.nonzero(keep)[0]

    # Circumferential angle of every kept wall point, in the parallel-transported
    # frame at its own assigned contour (origin = the recentred centre).
    ci = idx[kept_w]
    rel = wall_pts[kept_w] - centers[ci]
    T, N, B = cl_tan[ci], cl_nhat[ci], cl_bhat[ci]
    axial = np.sum(rel * T, axis=1)
    radial = rel - axial[:, None] * T
    phi_all = np.degrees(np.arctan2(np.sum(radial * B, axis=1),
                                    np.sum(radial * N, axis=1))) % 360.0

    # Bucket kept-point *positions* by their assigned (vessel, seg_idx).
    by_seg: dict[tuple[str, int], list[int]] = {}
    for p, wi in enumerate(kept_w):
        by_seg.setdefault(cl_key[idx[wi]], []).append(p)

    rows: list[dict[str, Any]] = []
    nan = float("nan")
    for key in seg_table["seg_keys"]:
        v, k = key
        s0, s1 = seg_table["seg_span"][key]
        pos = by_seg.get(key, [])
        base = {
            "vessel": v, "seg_idx": k,
            "s_start_mm": round(s0, 4), "s_end_mm": round(s1, 4),
            "has_bif": seg_table["seg_bif"][key],
            "ratio": label, "n_wall_pts": len(pos),
        }
        if len(pos) < min_pts:
            rows.append(dict(base, ring_mean_wss=nan, wss_max_window=nan,
                             angle_max_deg=nan, wss_min_window=nan, angle_min_deg=nan))
            continue
        p = np.asarray(pos, dtype=np.int64)
        res = arc_window_min_max(phi_all[p], wall_wss[kept_w[p]], arc_deg, rot_step_deg)
        rows.append(dict(base, **res))

    if want_assignment:
        return rows, keep, idx, centers
    return rows, None, None, centers


def write_wide_csv(
    path: Path,
    seg_table: dict[str, Any],
    labels: list[str],
    value_by_ratio: dict[str, dict[tuple[str, int], float]],
) -> int:
    """Write a paired wide table: one row per (vessel, seg_idx) with a
    ``wss_<label>`` column per ratio. Returns the number of rows written."""
    cols = ANCHOR_FIELDS + [f"wss_{lab}" for lab in labels]
    n = 0
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for key in seg_table["seg_keys"]:
            v, k = key
            s0, s1 = seg_table["seg_span"][key]
            row = {"vessel": v, "seg_idx": k,
                   "s_start_mm": round(s0, 4), "s_end_mm": round(s1, 4),
                   "has_bif": seg_table["seg_bif"][key]}
            for lab in labels:
                row[f"wss_{lab}"] = value_by_ratio[lab].get(key, float("nan"))
            w.writerow(row)
            n += 1
    return n


def _viz_segments(centrelines, seg_table, wall_pts, wall_wss, keep, idx, label,
                  centers=None):  # pragma: no cover
    """Interactive overlay: translucent WSS cloud + the kept points coloured by
    segment + centreline + 3 mm boundary markers. When ``centers`` (the recentred
    contour centres) are given, they are drawn in red so the drift correction is
    visible. Falls back to a screenshot when no display is available (via
    viz._show_plotter)."""
    try:
        import pyvista as pv
    except Exception as exc:
        print(f"[VIZ][WARN] pyvista unavailable: {exc}")
        return
    pl = pv.Plotter(title=f"WSS segments — {label}")
    pl.set_background("white")

    cloud = pv.PolyData(wall_pts)
    cloud["WSS"] = wall_wss
    pl.add_mesh(cloud, scalars="WSS", cmap="turbo", point_size=3,
                render_points_as_spheres=True, opacity=0.25,
                scalar_bar_args={"title": "WSS (Pa)"})

    if keep is not None and np.any(keep):
        kw = np.nonzero(keep)[0]
        kpts = pv.PolyData(wall_pts[kw])
        seg_of_pt = np.array([seg_table["cl_key"][idx[wi]][1] for wi in kw], dtype=float)
        kpts["segment"] = seg_of_pt
        pl.add_mesh(kpts, scalars="segment", cmap="tab20", point_size=8,
                    render_points_as_spheres=True, show_scalar_bar=False)

    starts = []
    for d in centrelines.values():
        c = np.asarray(d["coords"], dtype=np.float64)
        conn = np.concatenate([[len(c)], np.arange(len(c), dtype=np.int64)])
        pl.add_mesh(pv.PolyData(c, lines=conn), color="black", line_width=2)
    for (v, k), (s0, _s1) in seg_table["seg_span"].items():
        # nearest centreline point to the segment start, for a boundary marker
        cl = seg_table["cl_pts"]
        keys = seg_table["cl_key"]
        cand = [i for i, kk in enumerate(keys) if kk == (v, k)]
        if cand:
            starts.append(cl[cand[0]])
    if starts:
        pl.add_mesh(pv.PolyData(np.asarray(starts)), color="black", point_size=12,
                    render_points_as_spheres=True, show_scalar_bar=False)

    if centers is not None and not np.array_equal(centers, seg_table["cl_pts"]):
        moved = np.linalg.norm(centers - seg_table["cl_pts"], axis=1) > 1e-9
        if np.any(moved):
            pl.add_mesh(pv.PolyData(centers[moved]), color="red", point_size=6,
                        render_points_as_spheres=True, show_scalar_bar=False)

    try:
        from .viz import _show_plotter
        _show_plotter(pl, f"wss_segments_{label}")
    except Exception:
        pl.show()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Segment-based contour WSS sampling for paired cross-ratio comparison")
    ap.add_argument("--xml", required=True, help="base input Amira .am.xml")
    ap.add_argument("--sidecar", default=None, help="epicardial.json annotation")
    ap.add_argument("--model-dir", default=None,
                    help="model dir used only to default the sidecar to "
                         "<model-dir>/../epicardial.json")
    ap.add_argument("--ref-model-xml", default=None,
                    help="reference geometry for the shared centreline "
                         "(default: the full un-pruned base tree)")
    ap.add_argument("--ratio", action="append", default=None, metavar="LABEL=CSV",
                    help="a ratio's wall-node CSV as 'label=path' (repeatable). "
                         "Omit 'label=' to infer the label from the CSV's folder.")
    ap.add_argument("--vessels", default=None,
                    help="comma-separated vessel names to sample (default: all). "
                         "Use to run one side at a time, e.g. 'LAD,LCx' with the "
                         "left-tree clouds and 'RCA' with the right-tree clouds.")
    ap.add_argument("--segment-mm", type=float, default=3.0,
                    help="centreline segment length (mm)")
    ap.add_argument("--trim-proximal-bif", action=argparse.BooleanOptionalAction,
                    default=True,
                    help="start each vessel outside its proximal (e.g. LM) "
                         "bifurcation zone by dropping the leading bif points")
    ap.add_argument("--radius-factor", type=float, default=1.5,
                    help="keep a wall point if within radius_factor*local_radius "
                         "of its nearest centreline point")
    ap.add_argument("--arc-deg", type=float, default=90.0, help="arc window width")
    ap.add_argument("--rot-step-deg", type=float, default=5.0,
                    help="window rotation step")
    ap.add_argument("--min-pts", type=int, default=8,
                    help="min wall points in a segment to report values (else NaN)")
    ap.add_argument("--recenter", action=argparse.BooleanOptionalAction, default=True,
                    help="snap each contour centre onto the local lumen centroid of "
                         "the cloud (corrects centreline drift vs the smoothed mesh)")
    ap.add_argument("--recenter-min-pts", type=int, default=4,
                    help="min gated wall points to recentre a contour (else kept)")
    ap.add_argument("--exclude-sidebranches", action=argparse.BooleanOptionalAction,
                    default=True,
                    help="exclude wall points nearer to a side-branch (or non-selected "
                         "vessel) centreline than to the annotated vessel")
    ap.add_argument("--bif-skip", type=int, default=BIF_SKIP_POINTS,
                    help="passed through to centreline construction (unused here)")
    ap.add_argument("--coord-scale", type=float, default=None,
                    help="multiply CFX coords to mm (default: auto-detect per cloud)")
    ap.add_argument("--coord-offset", default="0,0,0",
                    help="add to CFX coords (mm) after scaling: 'dx,dy,dz'")
    ap.add_argument("--viz", action="store_true",
                    help="interactive pyvista overlay of the cloud + segments")
    ap.add_argument("--viz-ratio", default=None,
                    help="only visualise this ratio label (default: all, with --viz)")
    ap.add_argument("--out", default="wss_compare", help="output directory")
    args = ap.parse_args(argv)

    if not args.ratio:
        raise SystemExit("[COMPARE][ERROR] provide at least one --ratio label=cloud.csv")
    ratios = [parse_ratio_arg(s) for s in args.ratio]
    labels = [lab for lab, _ in ratios]
    if len(set(labels)) != len(labels):
        raise SystemExit(f"[COMPARE][ERROR] duplicate ratio labels: {labels}")

    # Sidecar resolution (mirrors wss_postprocess.main, base-tree fallback).
    model_dir = Path(args.model_dir) if args.model_dir else None
    if args.sidecar:
        sidecar = Path(args.sidecar)
    elif model_dir:
        sidecar = model_dir.parent / "epicardial.json"
    else:
        sidecar = Path(args.xml).parent / "epicardial.json"
    if not sidecar.exists():
        raise SystemExit(f"[COMPARE][ERROR] sidecar not found: {sidecar} (pass --sidecar)")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # Shared reference centrelines (full tree by default) + fixed 3 mm segments.
    ref_model_xml = Path(args.ref_model_xml) if args.ref_model_xml else None
    print("[COMPARE] reference geometry: "
          + (str(ref_model_xml) if ref_model_xml else "full tree (no ratio pruning)"))
    centrelines, comp_pts = cast(
        "tuple[dict[str, dict[str, np.ndarray]], np.ndarray]",
        build_vessel_centrelines(args.xml, ref_model_xml, sidecar,
                                 bif_skip=args.bif_skip, return_competitors=True))
    if not centrelines:
        raise SystemExit("[COMPARE][ERROR] no annotated main vessels resolved.")
    centrelines_all = centrelines
    if args.vessels:
        available = sorted(centrelines)
        centrelines, missing = filter_vessels(centrelines, args.vessels.split(","))
        if missing:
            print(f"[COMPARE][WARN] requested vessel(s) not found: {missing}; "
                  f"available: {available}")
        if not centrelines:
            raise SystemExit(f"[COMPARE][ERROR] --vessels matched none; available: {available}")

    # Voronoi competitors = side branches + any annotated vessel filtered out by
    # --vessels (so e.g. LCx wall points are excluded from a LAD-only run).
    if args.exclude_sidebranches:
        dropped = [d["coords"] for v, d in centrelines_all.items() if v not in centrelines]
        competitor_pts = np.vstack([comp_pts, *dropped]) if dropped else comp_pts
        print(f"[COMPARE] side-branch exclusion on: {len(competitor_pts)} competitor points")
    else:
        competitor_pts = np.empty((0, 3))
    seg_table = build_segment_table(centrelines, args.segment_mm,
                                    trim_proximal_bif=args.trim_proximal_bif,
                                    competitor_pts=competitor_pts)
    seg_keys = seg_table["seg_keys"]
    if not seg_keys:
        raise SystemExit("[COMPARE][ERROR] no segments produced (empty centrelines?).")
    per_vessel = {}
    for (v, _k) in seg_keys:
        per_vessel[v] = per_vessel.get(v, 0) + 1
    print(f"[COMPARE] {len(per_vessel)} vessel(s), {len(seg_keys)} segments @ "
          f"{args.segment_mm:g} mm: " + ", ".join(f"{v}({n})" for v, n in per_vessel.items()))

    all_ref_coords = seg_table["cl_pts"]
    offset = np.array([float(x) for x in args.coord_offset.split(",")], dtype=np.float64)

    long_rows: list[dict[str, Any]] = []
    max_by_ratio: dict[str, dict[tuple[str, int], float]] = {}
    min_by_ratio: dict[str, dict[tuple[str, int], float]] = {}
    for label, csv_path in ratios:
        wall_pts, wall_wss, unit = read_wss_csv(csv_path)
        scale, why = resolve_coord_scale(unit, wall_pts, all_ref_coords, args.coord_scale)
        print(f"[COMPARE] [{label}] coord scale x{scale:g} ({why})")
        wall_pts = wall_pts * scale + offset
        check_alignment(wall_pts, all_ref_coords)
        want_viz = args.viz and (args.viz_ratio is None or args.viz_ratio == label)
        rows, keep, idx, centers = assign_and_sample(
            seg_table, label, wall_pts, wall_wss, args.radius_factor,
            args.arc_deg, args.rot_step_deg, args.min_pts,
            recenter=args.recenter, recenter_min_pts=args.recenter_min_pts,
            want_assignment=want_viz)
        assert len(rows) == len(seg_keys), (
            f"[{label}] produced {len(rows)} rows != {len(seg_keys)} segments")
        if args.recenter:
            shift = np.linalg.norm(centers - all_ref_coords, axis=1)
            moved = int(np.sum(shift > 1e-9))
            med = float(np.median(shift[shift > 1e-9])) if moved else 0.0
            print(f"  [{label}] recentred {moved}/{len(shift)} contours "
                  f"(median shift {med:.4g} mm)")
        n_valid = sum(1 for r in rows if not np.isnan(r["wss_max_window"]))
        print(f"  [{label}] {len(rows)} segments ({n_valid} with WSS data)")
        long_rows.extend(rows)
        max_by_ratio[label] = {(r["vessel"], r["seg_idx"]): r["wss_max_window"] for r in rows}
        min_by_ratio[label] = {(r["vessel"], r["seg_idx"]): r["wss_min_window"] for r in rows}
        if want_viz:
            _viz_segments(centrelines, seg_table, wall_pts, wall_wss, keep, idx,
                          label, centers=centers)

    empty = vessels_without_data(long_rows)
    if empty:
        print(f"[COMPARE][WARN] no WSS data for: {', '.join(empty)} - wrong "
              "side/cloud? (pass --vessels to restrict to this side's vessels)")

    with open(out / "wss_seg_long.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=LONG_FIELDS)
        w.writeheader()
        w.writerows({k: r.get(k) for k in LONG_FIELDS} for r in long_rows)
    n_max = write_wide_csv(out / "wss_seg_max_wide.csv", seg_table, labels, max_by_ratio)
    write_wide_csv(out / "wss_seg_min_wide.csv", seg_table, labels, min_by_ratio)

    print(f"[DONE] wrote {out / 'wss_seg_long.csv'} ({len(long_rows)} rows), "
          f"wss_seg_max_wide.csv + wss_seg_min_wide.csv "
          f"({n_max} segments x {len(labels)} ratios)")
    return 0


__all__ = [
    "parse_ratio_arg",
    "filter_vessels",
    "vessels_without_data",
    "build_segment_table",
    "recenter_on_cloud",
    "assign_and_sample",
    "write_wide_csv",
    "main",
]


if __name__ == "__main__":
    raise SystemExit(main())
