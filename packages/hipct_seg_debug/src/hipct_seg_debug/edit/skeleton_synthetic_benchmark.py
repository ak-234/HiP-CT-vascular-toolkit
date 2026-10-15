"""Score skeleton *extraction* against phantoms with a known centreline.

Run with::

    python -m hipct_seg_debug.edit.skeleton_synthetic_benchmark --out-dir DIR

Every candidate is run through :func:`~.skeletonisers.skeletonise` on every phantom of
:mod:`~.skeleton_phantoms`, and scored on the same terms:

* **position** -- skeleton-to-truth distance (the spurious side) and truth-to-skeleton
  coverage (the missed side), in voxels, with the flat end caps excluded (1.5 R0);
* **branches** -- terminals matched one-to-one to the true ones within 2 R0 + 2 voxels;
  unmatched skeleton terminals are false branches, unmatched true ones missing;
  junctions matched the same way, with their position error;
* **ladders** -- the share of true centreline samples whose lumen cross-section (within
  the major semi-axis) holds two or more skeleton segments, away from junctions. This
  is the failure the collapsed-lumen argument predicts for minimum-cost paths;
* **topology** -- components and first Betti number against the phantom's;
* **containment and smoothness** -- ``supermetric.cl_sensitivity`` and
  ``skeleton_optimise.roughness``;
* **radii** -- at skeleton points, ``r_DT`` (EDT inscribed ball), ``r_A`` (area of a
  stable transverse cut), ``r_inflated`` (``radius_perimeter.measure_radii``) and, for
  ``teasar-readme``, kimimaro's own cross-sectional area, each as a ratio to the
  phantom's inflated radius ``R0`` and grouped by the collapse ratio there.

These are phantoms: the scores say how each method behaves on a known geometry, not
how it does on anatomy. ``skeletonise-all --summary-json`` is the real-data side.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np

from . import skeleton_phantoms as sp
from . import skeletonisers as sk

ROOTED = ("jin-mcp", "jin-mcp-centroid", "jin-mgf", "geodesic-scoop")

TEASAR_SCALES = (1.5, 2.5, 4.0, 6.0)
TEASAR_CONST_VOX = (2, 5, 10)


def default_candidates(spacing_um=sp.SPACING_UM) -> list[str]:
    grid = [f"teasar-readme[scale={s:g};const_um={c * spacing_um:g}]"
            for s in TEASAR_SCALES for c in TEASAR_CONST_VOX]
    # jin-mgf ablations: each modification alone, both, the cheap rim MGF, and the
    # methodology note's PCA classifier in place of the Hessian one.
    return ["teasar", *grid, "jin-mcp", "jin-mcp-centroid", "geodesic-scoop",
            "jin-mgf[medialness=mgf;marking=dilate]",
            "jin-mgf[medialness=lsf;marking=section]",
            "jin-mgf", "jin-mgf[mgf_mode=rim]", "jin-mgf[classifier=pca]"]


def phantom_frame(shape, spacing_um):
    from ..amira import LatticeInfo
    from ..frame import WorldFrame

    dims = np.array(shape[::-1])
    bbox = np.zeros(6)
    bbox[1::2] = (dims - 1) * spacing_um
    info = LatticeInfo(path=None, dims=dims, bbox=bbox, fields={})
    return WorldFrame.from_inputs(tuple(2 * n for n in shape), spacing_um / 2, info)


# ---------------------------------------------------------------- geometry helpers


def _segments_zyx(triple, frame):
    """``[(sid, (n,3) zyx voxels)]`` and node degrees/positions from a triple."""
    segs = []
    for s in triple.segments:
        xyz = np.array([triple.points[p][:3] for p in s["point_ids"]], dtype=np.float64)
        if len(xyz):
            segs.append((s["id"], frame.um_to_seg(xyz)[:, ::-1]))
    nodes = {nid: (frame.um_to_seg(np.array(n[:3]))[0, ::-1], int(n[3]))
             for nid, n in triple.nodes.items()}
    return segs, nodes


def _densify(p, step=0.5):
    if len(p) < 2:
        return p
    s = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))])
    n = max(2, int(np.ceil(s[-1] / step)) + 1)
    t = np.linspace(0, s[-1], n)
    return np.column_stack([np.interp(t, s, p[:, k]) for k in range(3)])


def _match(found, truth, tol):
    """Optimal one-to-one matching within per-truth tolerance; returns matched pairs."""
    from scipy.optimize import linear_sum_assignment

    if not len(found) or not len(truth):
        return []
    d = np.linalg.norm(found[:, None, :] - truth[None, :, :], axis=2)
    cost = np.where(d <= tol[None, :], d, 1e9)
    rows, cols = linear_sum_assignment(cost)
    return [(r, c, float(d[r, c])) for r, c in zip(rows, cols) if d[r, c] <= tol[c]]


def _band(lam):
    bands = np.asarray(sp.COLLAPSE_BANDS)
    return bands[np.argmin(np.abs(bands[None, :] - np.asarray(lam)[:, None]), axis=1)]


def _median(x):
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    return float(np.median(x)) if len(x) else None


def _pct(x, q):
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    return float(np.percentile(x, q)) if len(x) else None


# ------------------------------------------------------------------------ scoring


def score_geometry(phantom, triple, frame) -> dict:
    """Position, branch, ladder and topology terms for one skeleton on one phantom."""
    import networkx as nx
    from scipy.spatial import cKDTree

    gt, r0, lam, _bid = phantom.samples()
    a_gt, _ = sp.semi_axes(r0, lam)
    terms = phantom.terminals
    term_r0 = r0[cKDTree(gt).query(terms)[1]] if len(terms) else np.zeros(0)
    # Exclude the flat end caps from position statistics: every method treats a flat
    # cap differently (stop short, fork into its corners) and none of that is anatomy.
    near_term = np.zeros(len(gt), dtype=bool)
    for t, r in zip(terms, term_r0):
        near_term |= np.linalg.norm(gt - t, axis=1) <= 1.5 * r
    junc = phantom.junctions
    near_junc = np.zeros(len(gt), dtype=bool)
    for j in junc:
        rj = float(r0[cKDTree(gt).query(j)[1]])
        near_junc |= np.linalg.norm(gt - j, axis=1) <= 2.0 * rj

    segs, nodes = _segments_zyx(triple, frame)
    out = dict(segments=len(segs), nodes=len(nodes))
    if not segs:
        out.update(false_branches=0, missing_branches=int(len(terms)), coverage=0.0,
                   ladder_fraction=0.0, beta1=0, components=len(nodes))
        return out

    dense = [(sid, _densify(p)) for sid, p in segs]
    sk_pts = np.vstack([p for _, p in dense])
    sk_sid = np.concatenate([np.full(len(p), sid) for sid, p in dense])
    gtree = cKDTree(gt)
    d_s, nearest = gtree.query(sk_pts)
    keep_s = ~near_term[nearest]
    out["error_median_vox"] = _median(d_s[keep_s])
    out["error_p95_vox"] = _pct(d_s[keep_s], 95)
    out["error_max_vox"] = float(np.max(d_s[keep_s])) if keep_s.any() else None

    stree = cKDTree(sk_pts)
    d_g, _ = stree.query(gt)
    tol = np.maximum(1.5, 0.5 * r0)
    body = ~near_term
    out["coverage"] = float(np.mean(d_g[body] <= tol[body])) if body.any() else None
    out["missed_median_vox"] = _median(d_g[body])

    # Ladders: a second segment running through the same lumen cross-section.
    reach = a_gt + 1.0
    probe = body & ~near_junc
    multi = np.zeros(len(gt), dtype=bool)
    for i in np.flatnonzero(probe):
        hits = stree.query_ball_point(gt[i], reach[i])
        multi[i] = len(set(sk_sid[hits].tolist())) >= 2
    out["ladder_fraction"] = float(multi[probe].mean()) if probe.any() else 0.0

    # Per-lambda breakdown (collapse phantoms).
    if np.ptp(lam) > 0 or lam[0] < 1:
        band_s = _band(lam[nearest])
        band_g = _band(lam)
        out["by_lambda"] = {
            f"{b:g}": dict(error_median_vox=_median(d_s[keep_s & (band_s == b)]),
                           coverage=float(np.mean(d_g[body & (band_g == b)]
                                                  <= tol[body & (band_g == b)]))
                           if (body & (band_g == b)).any() else None,
                           ladder_fraction=float(multi[probe & (band_g == b)].mean())
                           if (probe & (band_g == b)).any() else None)
            for b in np.unique(band_g)}

    deg = np.array([d for _, d in nodes.values()])
    pos = np.array([p for p, _ in nodes.values()])
    leaves = pos[deg == 1]
    forks = pos[deg >= 3]
    tol_t = 2.0 * term_r0 + 2.0
    mt = _match(leaves, terms, tol_t)
    out["terminals"] = int(len(leaves))
    out["false_branches"] = int(len(leaves) - len(mt))
    out["missing_branches"] = int(len(terms) - len(mt))
    mj = []
    if len(junc):
        tol_j = np.array([2.0 * float(r0[gtree.query(j)[1]]) + 2.0 for j in junc])
        mj = _match(forks, junc, tol_j)
        out["junction_error_vox"] = _median([d for *_, d in mj]) if mj else None
        out["missing_junctions"] = int(len(junc) - len(mj))
    out["extra_junctions"] = int(len(forks) - len(mj))
    g = nx.MultiGraph()
    g.add_nodes_from(nodes)
    g.add_edges_from((s["node1"], s["node2"]) for s in triple.segments)
    cc = nx.number_connected_components(g)
    out["components"] = int(cc)
    out["beta1"] = int(g.number_of_edges() - g.number_of_nodes() + cc)
    return out


def score_radii(phantom, triple, frame, *, max_points=150) -> dict:
    """``r / R0`` for each estimator at skeleton points, grouped by the local lambda."""
    from scipy.spatial import cKDTree

    from ..crosssection import _PlaneSampler, robust_edge_tangents, stable_transverse_cut
    from .graphmodel import EditableGraph
    from .radius_perimeter import measure_radii
    from .skeletonise import edt_radii

    gt, r0, lam, _ = phantom.samples()
    spacing = float(phantom.spacing_um)
    terms, junc = phantom.terminals, phantom.junctions
    graph = EditableGraph(triple)
    try:
        perim = measure_radii(graph, frame, phantom.mask).radii
    except Exception as exc:  # report, do not abort the whole row
        perim, perim_error = {}, f"{type(exc).__name__}: {exc}"
    else:
        perim_error = None
    sampler = _PlaneSampler(phantom.mask, frame)
    area_attr = triple.point_attrs.get(sk.KIMIMARO_AREA, {})

    rows = []
    for seg in triple.segments:
        sid, pids = seg["id"], seg["point_ids"]
        if len(pids) < 2:
            continue
        xyz = np.array([triple.points[p][:3] for p in pids])
        rad = np.array([triple.points[p][3] for p in pids])
        tang = robust_edge_tangents(xyz, rad, spacing_um=spacing)
        zyx = frame.um_to_seg(xyz)[:, ::-1]
        r_dt = edt_radii(phantom.mask, np.rint(zyx).astype(np.int64),
                         frame.seg_spacing)
        r_in = np.asarray(perim.get(sid, np.full(len(pids), np.nan)), dtype=np.float64)
        for i in range(len(pids)):
            rows.append((zyx[i], xyz[i], tang[i], rad[i], r_dt[i],
                         r_in[i] if i < len(r_in) else np.nan,
                         area_attr.get(pids[i], np.nan)))
    if not rows:
        return dict(radius_points=0, perimeter_error=perim_error)
    zyx = np.array([r[0] for r in rows])
    _, nearest = cKDTree(gt).query(zyx)
    away = np.ones(len(rows), dtype=bool)
    for p in np.vstack([terms, junc]) if len(junc) else terms:
        k = cKDTree(gt).query(p)[1]
        away &= np.linalg.norm(zyx - p, axis=1) > 2.0 * r0[k]
    use = np.flatnonzero(away)
    if len(use) > max_points:
        use = use[np.linspace(0, len(use) - 1, max_points).astype(int)]

    est = {"r_dt": [], "r_a": [], "r_inflated": [], "r_a_kimimaro": []}
    bands = []
    for i in use:
        _, xyz_i, t_i, rad_i, rdt, rin, karea = rows[i]
        R0 = r0[nearest[i]] * spacing
        bands.append(lam[nearest[i]])
        est["r_dt"].append(rdt / R0)
        est["r_inflated"].append(rin / R0)
        est["r_a_kimimaro"].append(np.sqrt(karea / np.pi) / R0 if np.isfinite(karea)
                                   else np.nan)
        cut = stable_transverse_cut(sampler, frame.um_to_seg(xyz_i)[0], t_i,
                                    max(float(rad_i) / spacing, 2.0), spacing_um=spacing,
                                    max_half=64, centroid_mode="drift")
        if cut is None or cut.cut.touches_border:
            est["r_a"].append(np.nan)
        else:
            area = float(cut.cut.blob8.sum()) * spacing * spacing
            est["r_a"].append(np.sqrt(area / np.pi) / R0)
    band = _band(np.array(bands))
    out = dict(radius_points=int(len(use)), perimeter_error=perim_error, by_lambda={})
    for b in np.unique(band):
        sel = band == b
        out["by_lambda"][f"{b:g}"] = {
            k: _median(np.asarray(v)[sel]) for k, v in est.items()}
    return out


# ------------------------------------------------------------------------- driver


def run_one(spec, phantom, frame, *, radii=True) -> dict:
    name, params = sk.parse_candidate(spec)
    if name in ROOTED and "root_zyx" not in params:
        params["root_zyx"] = phantom.root_zyx
    row = dict(phantom=phantom.name, candidate=spec, algorithm=name,
               gt_beta1=phantom.beta1, gt_terminals=int(len(phantom.terminals)),
               gt_junctions=int(len(phantom.junctions)))
    t0 = time.monotonic()
    try:
        cand = sk.skeletonise(name, phantom.mask, frame, **params)
    except (ValueError, RuntimeError) as exc:
        row.update(status="refused", error=f"{type(exc).__name__}: {exc}",
                   seconds=time.monotonic() - t0)
        return row
    except ImportError as exc:
        row.update(status="unavailable", error=str(exc), seconds=time.monotonic() - t0)
        return row
    row.update(status="evaluated", seconds=cand.seconds, detail=cand.detail)
    row.update(score_geometry(phantom, cand.triple, frame))
    from .graphmodel import EditableGraph
    from .skeleton_optimise import roughness
    from .supermetric import cl_sensitivity

    graph = EditableGraph(cand.triple)
    if cand.triple.segments:
        row["cl_sensitivity"] = float(cl_sensitivity(graph, frame, phantom.mask))
        row["reversing"] = float(roughness(graph).reversing)
    if radii and cand.triple.segments:
        row["radii"] = score_radii(phantom, cand.triple, frame)
    row["_triple"] = cand.triple
    return row


def run(phantoms=sp.PHANTOMS, candidates=None, *, radii=True, out_dir=None,
        write_graphs=False, verbose=True) -> list[dict]:
    candidates = default_candidates() if candidates is None else list(candidates)
    rows = []
    for pname in phantoms:
        phantom = sp.build(pname)
        frame = phantom_frame(phantom.mask.shape, phantom.spacing_um)
        for spec in candidates:
            row = run_one(spec, phantom, frame, radii=radii)
            triple = row.pop("_triple", None)
            if write_graphs and out_dir is not None and triple is not None:
                from .adapter import to_spatial_graph
                from .amira_write import write_spatial_graph

                name, params = sk.parse_candidate(spec)
                dest = Path(out_dir) / "graphs" / pname / f"{sk.candidate_label(name, params)}.am"
                dest.parent.mkdir(parents=True, exist_ok=True)
                write_spatial_graph(to_spatial_graph(triple), dest)
            rows.append(row)
            if verbose:
                print(f"{pname:18s} {spec:45s} {row['status']:10s} "
                      f"false={row.get('false_branches', '-')} miss={row.get('missing_branches', '-')} "
                      f"ladder={_fmt(row.get('ladder_fraction'))} "
                      f"err={_fmt(row.get('error_median_vox'))} b1={row.get('beta1', '-')}"
                      f" ({row['seconds']:.1f}s)", flush=True)
    return rows


def _fmt(v, nd=2):
    return "-" if v is None else (f"{v:.{nd}f}" if isinstance(v, float) else str(v))


def best_teasar(rows) -> str | None:
    """The ``teasar-readme`` grid point with the fewest branch errors, then least error."""
    by: dict = {}
    for r in rows:
        if r["algorithm"] != "teasar-readme" or r["status"] != "evaluated":
            continue
        s = by.setdefault(r["candidate"], [0.0, 0.0])
        s[0] += r.get("false_branches", 0) + r.get("missing_branches", 0)
        s[1] += r.get("error_median_vox") or 0.0
    return min(by, key=lambda k: tuple(by[k])) if by else None


def summary_markdown(rows) -> str:
    lines = []
    for pname in dict.fromkeys(r["phantom"] for r in rows):
        group = [r for r in rows if r["phantom"] == pname]
        g0 = group[0]
        lines += [f"\n### {pname} (truth: {g0['gt_terminals']} terminals, "
                  f"{g0['gt_junctions']} junctions, beta1 {g0['gt_beta1']})\n",
                  "| candidate | false | missing | ladder | beta1 | err med | err p95 | "
                  "coverage | junction err | cl | rev | s |",
                  "|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for r in group:
            if r["status"] != "evaluated":
                lines.append(f"| {r['candidate']} | {r['status']}: "
                             f"{r.get('error', '')[:60]} |||||||||||")
                continue
            lines.append("| " + " | ".join([
                r["candidate"], _fmt(r.get("false_branches")), _fmt(r.get("missing_branches")),
                _fmt(r.get("ladder_fraction")), _fmt(r.get("beta1")),
                _fmt(r.get("error_median_vox")), _fmt(r.get("error_p95_vox")),
                _fmt(r.get("coverage")), _fmt(r.get("junction_error_vox")),
                _fmt(r.get("cl_sensitivity"), 3), _fmt(r.get("reversing"), 3),
                _fmt(r.get("seconds"), 1)]) + " |")
    band_rows = [r for r in rows if r["phantom"] == "collapse-bands"
                 and r["status"] == "evaluated" and "by_lambda" in r]
    if band_rows:
        lines += ["\n### collapse-bands, per lambda: median position error (vox) / ladder\n",
                  "| candidate | " + " | ".join(f"lambda {b:g}" for b in sp.COLLAPSE_BANDS) + " |",
                  "|---|" + "---|" * len(sp.COLLAPSE_BANDS)]
        for r in band_rows:
            cells = []
            for b in sp.COLLAPSE_BANDS:
                v = r["by_lambda"].get(f"{b:g}", {})
                cells.append(f"{_fmt(v.get('error_median_vox'))} / {_fmt(v.get('ladder_fraction'))}")
            lines.append(f"| {r['candidate']} | " + " | ".join(cells) + " |")
    radius_rows = [r for r in band_rows if r.get("radii", {}).get("by_lambda")]
    if radius_rows:
        lines += ["\n### collapse-bands, radius / R0 per lambda (r_DT, r_A, r_inflated"
                  " [, kimimaro r_A])\n",
                  "| candidate | " + " | ".join(f"lambda {b:g}" for b in sp.COLLAPSE_BANDS) + " |",
                  "|---|" + "---|" * len(sp.COLLAPSE_BANDS)]
        analytic = {f"{a['lam']:g}": a for a in sp.analytic_radius_table()}
        lines.append("| *analytic* | " + " | ".join(
            f"{analytic[f'{b:g}']['inscribed']:.2f}, {analytic[f'{b:g}']['area']:.2f}, 1.00"
            for b in sp.COLLAPSE_BANDS) + " |")
        for r in radius_rows:
            cells = []
            for b in sp.COLLAPSE_BANDS:
                v = r["radii"]["by_lambda"].get(f"{b:g}")
                if not v:
                    cells.append("-")
                    continue
                c = f"{_fmt(v['r_dt'])}, {_fmt(v['r_a'])}, {_fmt(v['r_inflated'])}"
                if v.get("r_a_kimimaro") is not None:
                    c += f" [{_fmt(v['r_a_kimimaro'])}]"
                cells.append(c)
            lines.append(f"| {r['candidate']} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--phantoms", nargs="+", choices=sp.PHANTOMS, default=list(sp.PHANTOMS))
    p.add_argument("--candidates", default=None,
                   help="comma-separated candidate specs, e.g. "
                        "'teasar-readme[scale=4;const_um=50],geodesic-scoop'")
    p.add_argument("--no-radii", action="store_true", help="skip the radius estimators")
    p.add_argument("--no-pdrf-stage", action="store_true",
                   help="do not re-run the best teasar-readme grid point at pdrf_exponent 8")
    p.add_argument("--write-graphs", action="store_true", help="write each skeleton as .am")
    args = p.parse_args(argv)
    candidates = (sk.split_candidates(args.candidates) if args.candidates
                  else default_candidates())
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows = run(args.phantoms, candidates, radii=not args.no_radii, out_dir=args.out_dir,
               write_graphs=args.write_graphs)
    best = None if args.no_pdrf_stage else best_teasar(rows)
    if best is not None:
        name, params = sk.parse_candidate(best)
        params["pdrf_exponent"] = 8
        spec = name + "[" + ";".join(f"{k}={v}" for k, v in params.items()) + "]"
        print(f"\nbest teasar-readme grid point: {best}; re-running at pdrf_exponent 8")
        rows += run(args.phantoms, [spec], radii=not args.no_radii, out_dir=args.out_dir,
                    write_graphs=args.write_graphs)
    (args.out_dir / "results.json").write_text(json.dumps(
        dict(phantoms=args.phantoms, candidates=candidates, best_teasar=best,
             analytic_radii=sp.analytic_radius_table(), rows=rows), indent=2, default=float),
        encoding="utf-8")
    md = summary_markdown(rows)
    (args.out_dir / "summary.md").write_text(md, encoding="utf-8")
    print(md)
    return int(any(r["status"] == "unavailable" for r in rows))


if __name__ == "__main__":
    raise SystemExit(main())
