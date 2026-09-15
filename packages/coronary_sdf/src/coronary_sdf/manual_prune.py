"""Manual side-branch pruning with a scalar-coloured 3D picker.

A standalone, operator-driven counterpart to
:mod:`coronary_sdf.epicardial_annotation`. Instead of pruning automatically by a
radius ratio, you *look at* the tree — with every segment coloured by **Strahler
order**, **radius**, or **segment index** (the same colour conventions as
:func:`coronary_sdf.viz._add_scalar_lines`) — and click the branches to cut.

Workflow
--------
1. Parse + canonical-preprocess the spatial graph (same chain as the surface
   pipeline, via :func:`flow_fractions.preprocess_topology`).
2. Annotate the main epicardial vessels in the interactive picker (reuses
   :func:`epicardial_annotation.load_or_create_annotation`; persisted to a sidecar
   JSON). Main vessels are ALWAYS preserved and rendered locked in the pruner.
3. Open the prune picker (:func:`epicardial_annotation.run_prune_picker`) with
   every non-main segment coloured by the chosen scalar + a legend. Left-click a
   branch to mark it (red); it and its whole downstream subtree are removed.
4. Write the pruned tree to a new Amira ``.am.xml`` (round-trips through
   :func:`parse_amira.parse_xml`), ready for surface generation, and write a log
   detailing exactly which segments were removed. Optionally run the surface
   pipeline (``--pipeline``).

Usage::

    python -m coronary_sdf.manual_prune --xml <in.am.xml> --out <dir> --pick \
        --color-by strahler
    # regenerate protection from a saved sidecar (still opens the prune picker):
    python -m coronary_sdf.manual_prune --xml <in.am.xml> --out <dir>
"""

from __future__ import annotations

import argparse
import csv
import datetime as _dt
import json
from pathlib import Path
from typing import Any

import numpy as np

from . import config
from .parse_amira import parse_xml
from .flow_fractions import preprocess_topology
from .splines import prepare_segment_spline
from .epicardial_annotation import (
    _file_sha1,
    _preprocess_signature,
    apply_manual_prune,
    build_segment_keys,
    load_or_create_annotation,
    run_prune_picker,
    write_amira_xml,
)

_GREY = (0.72, 0.72, 0.72)


# ── Per-segment scalar colouring (mirrors viz._add_scalar_lines) ───────────────


def _seg_mean_radius_mm(
    seg: dict[str, Any], points: dict[int, tuple], nodes: dict[int, tuple]
) -> float:
    """Representative (mean) radius (mm) of a segment from its smoothed spline,
    or ``nan`` when the segment has no usable spline."""
    sp = prepare_segment_spline(seg, points, nodes)
    if sp is None:
        return float("nan")
    radii = np.asarray(sp.get("radii", []), dtype=np.float64)
    return float(np.mean(radii)) if radii.size else float("nan")


def build_segment_colors(
    segments: list[dict[str, Any]],
    points: dict[int, tuple],
    nodes: dict[int, tuple],
    color_by: str,
) -> tuple[Any, list[tuple[str, tuple]], str]:
    """Build ``(base_color_fn, legend_entries, effective_mode)`` for the picker.

    ``base_color_fn(seg_idx) -> (r, g, b)`` gives each segment its resting colour;
    ``legend_entries`` is a list of ``(label, rgb)`` pairs for ``pl.add_legend``.

    Colour conventions match :func:`coronary_sdf.viz._add_scalar_lines`:
    ``strahler`` -> discrete ``tab10``; ``radius`` -> ``viridis``; ``segment``
    (and every fallback) -> ``turbo`` by index. Falls back to ``segment`` when the
    requested scalar is unavailable (missing Strahler values / no radii).
    """
    import matplotlib.pyplot as plt

    n = len(segments)
    mode = color_by

    if mode == "strahler":
        vals = [s.get("strahler") for s in segments]
        if not vals or any(v is None for v in vals):
            print("[COLOR] Strahler order unavailable on some segments; "
                  "falling back to '--color-by segment'.")
            mode = "segment"

    if mode == "strahler":
        arr = np.asarray([int(s.get("strahler", 0)) for s in segments], dtype=np.int64)
        lo, hi = int(arr.min()), int(arr.max())
        ncol = max(hi - lo + 1, 1)
        cmap = plt.get_cmap("tab10", ncol)
        colors = {i: tuple(cmap(int(arr[i]) - lo)[:3]) for i in range(n)}
        legend = [(f"order {k}", tuple(cmap(k - lo)[:3])) for k in range(lo, hi + 1)]

        def fn(i: int, _c=colors) -> tuple:
            return _c.get(i, _GREY)

        print(f"[COLOR] by Strahler order (tab10), orders {lo}..{hi}")
        return fn, legend, mode

    if mode == "radius":
        radii = np.array(
            [_seg_mean_radius_mm(s, points, nodes) for s in segments], dtype=np.float64)
        finite = np.isfinite(radii)
        if not finite.any():
            print("[COLOR] no segment radii available; falling back to "
                  "'--color-by segment'.")
            mode = "segment"
        else:
            rmin, rmax = float(np.nanmin(radii)), float(np.nanmax(radii))
            span = max(rmax - rmin, 1e-9)
            cmap = plt.get_cmap("viridis")
            colors = {
                i: (tuple(cmap((radii[i] - rmin) / span)[:3])
                    if np.isfinite(radii[i]) else _GREY)
                for i in range(n)
            }
            nbins = 5
            edges = np.linspace(rmin, rmax, nbins + 1)
            legend = [
                (f"{edges[b]:.3f}-{edges[b + 1]:.3f} mm",
                 tuple(cmap((b + 0.5) / nbins)[:3]))
                for b in range(nbins)
            ]

            def fn(i: int, _c=colors) -> tuple:
                return _c.get(i, _GREY)

            print(f"[COLOR] by radius (viridis), {rmin:.3f}..{rmax:.3f} mm")
            return fn, legend, mode

    # segment index (turbo) — the default fallback.
    cmap = plt.get_cmap("turbo")
    denom = float(max(n - 1, 1))

    def fn(i: int, _cmap=cmap, _d=denom) -> tuple:
        return tuple(_cmap(i / _d)[:3])

    legend = [("seg 0", tuple(cmap(0.0)[:3])),
              (f"seg {max(n - 1, 0)}", tuple(cmap(1.0)[:3]))]
    print("[COLOR] by segment index (turbo)")
    return fn, legend, "segment"


# ── Removal log ────────────────────────────────────────────────────────────────


def write_removal_logs(
    out: Path,
    xml_path: Path,
    color_mode: str,
    vessels_idx: dict[str, set[int]],
    base_n_segs: int,
    remaining_n_segs: int,
    removed: list[dict[str, Any]],
    id_to_strahler: dict[int, Any],
    out_xml: Path,
) -> tuple[Path, Path]:
    """Write ``removed_segments.csv`` (structured) and ``manual_prune_log.txt``
    (human-readable) into ``out``. Returns their paths."""
    csv_path = out / "removed_segments.csv"
    log_path = out / "manual_prune_log.txt"

    total_removed = sum(int(r["n_subtree_removed"]) for r in removed)

    fields = ["takeoff_seg_id", "strahler", "radius_mm", "vessel",
              "n_subtree_removed", "removed_seg_ids"]
    with open(csv_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for r in removed:
            sid = int(r["seg_id"])
            w.writerow({
                "takeoff_seg_id": sid,
                "strahler": id_to_strahler.get(sid, ""),
                "radius_mm": r.get("radius_mm"),
                "vessel": r.get("vessel", ""),
                "n_subtree_removed": r["n_subtree_removed"],
                "removed_seg_ids": " ".join(str(x) for x in r.get("removed_seg_ids", [])),
            })

    lines: list[str] = []
    lines.append("Manual side-branch pruning log")
    lines.append("=" * 60)
    lines.append(f"timestamp:        {_dt.datetime.now().isoformat(timespec='seconds')}")
    lines.append(f"input xml:        {xml_path}")
    try:
        lines.append(f"input sha1:       {_file_sha1(xml_path)}")
    except OSError:
        lines.append("input sha1:       <unavailable>")
    lines.append(f"colour mode:      {color_mode}")
    lines.append(f"preprocess:       {json.dumps(_preprocess_signature())}")
    lines.append("")
    lines.append("Protected main vessels (never removed):")
    if vessels_idx:
        for v, idxs in vessels_idx.items():
            lines.append(f"  {v}: {len(idxs)} segment(s)")
    else:
        lines.append("  (none)")
    lines.append("")
    lines.append(f"base segments:                    {base_n_segs}")
    lines.append(f"segments removed (subtrees):      {total_removed}")
    lines.append(f"remaining after degree-2 merges:  {remaining_n_segs}")
    lines.append(f"take-off selections:              {len(removed)}")
    lines.append(f"output spatial graph:             {out_xml}")
    lines.append("")
    lines.append("Removed take-off branches")
    lines.append("-" * 60)
    if not removed:
        lines.append("(none)")
    for k, r in enumerate(removed, 1):
        sid = int(r["seg_id"])
        rad = r.get("radius_mm")
        rad_s = "n/a" if rad is None else f"{rad:.5f} mm"
        vessel = r.get("vessel") or "(unattributed)"
        ids = r.get("removed_seg_ids", [])
        lines.append(
            f"[{k}] take-off seg_id={sid}  strahler={id_to_strahler.get(sid, '?')}  "
            f"radius={rad_s}  descends_from={vessel}  "
            f"subtree_removed={r['n_subtree_removed']}")
        lines.append(f"     removed seg_ids: {', '.join(str(x) for x in ids)}")
    lines.append("")
    log_path.write_text("\n".join(lines))

    print(f"[LOG] wrote {csv_path} and {log_path} "
          f"({len(removed)} selection(s), {total_removed} segment(s) removed)")
    return csv_path, log_path


# ── Orchestration ──────────────────────────────────────────────────────────────


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Manual side-branch pruning with a scalar-coloured 3D picker")
    ap.add_argument("--xml", default=config.INPUT_XML, help="input Amira .am/.am.xml")
    ap.add_argument("--out", default="manual_prune_out", help="output directory")
    ap.add_argument("--sidecar", default=None,
                    help="annotation JSON (default <out>/epicardial.json)")
    ap.add_argument("--color-by", default=config.DEBUG_VIS_COLOR_BY,
                    choices=["strahler", "radius", "segment"],
                    help="scalar used to colour segments in the prune picker")
    ap.add_argument("--pick", action="store_true",
                    help="force the main-vessel annotation picker to re-run")
    ap.add_argument("--no-hover", action="store_true",
                    help="disable the picker's hover preview highlight")
    ap.add_argument("--out-name", default="pruned.am.xml",
                    help="filename for the pruned spatial graph inside --out")
    ap.add_argument("--ostium-skip", type=int, default=None,
                    help="contours skipped at a take-off when measuring its radius "
                         "(default: flow_fractions.BIF_SKIP_POINTS)")
    ap.add_argument("--radius-navg", type=int, default=None,
                    help="downstream contours averaged for a take-off radius "
                         "(default: flow_fractions.FRAMES_DOWNSTREAM)")
    ap.add_argument("--pipeline", action="store_true",
                    help="after pruning, run the surface pipeline on the output")
    args = ap.parse_args(argv)

    xml_path = Path(args.xml)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    sidecar = Path(args.sidecar) if args.sidecar else out / "epicardial.json"

    # 1. Parse + canonical preprocess -> fixed base topology.
    nodes, points, segments = parse_xml(xml_path)
    nodes, points, segments = preprocess_topology(nodes, points, segments)
    seg_keys = build_segment_keys(segments, points)
    base_n_segs = len(segments)
    id_to_strahler = {int(s["id"]): s.get("strahler") for s in segments}

    # 2. Annotate (or reload) the main vessels; they become the locked set.
    vessels_idx, _colors = load_or_create_annotation(
        sidecar, xml_path, nodes, points, segments, seg_keys,
        force_pick=args.pick, hover=not args.no_hover)
    vessel_points = {
        v: {pid for i in idxs for pid in segments[i]["point_ids"]}
        for v, idxs in vessels_idx.items()
    }

    # 3. Scalar colours + legend, then the manual prune picker.
    base_color_fn, legend_entries, color_mode = build_segment_colors(
        segments, points, nodes, args.color_by)
    remove_idx = run_prune_picker(
        nodes, points, segments, vessel_points, hover=not args.no_hover,
        base_color_fn=base_color_fn, legend_entries=legend_entries)

    if not remove_idx:
        print("[MANUAL] nothing selected; no file or log written.")
        return 0

    # 4. Apply the removal (subtree expansion + degree-2 contraction).
    p_nodes, p_points, p_segs, removed = apply_manual_prune(
        nodes, points, segments, vessel_points, remove_idx,
        skip_points=args.ostium_skip, n_average=args.radius_navg)

    out_xml = out / args.out_name
    write_amira_xml(p_nodes, p_points, p_segs, out_xml)

    write_removal_logs(
        out, xml_path, color_mode, vessels_idx, base_n_segs, len(p_segs),
        removed, id_to_strahler, out_xml)

    # 5. Optional surface generation on the pruned graph.
    if args.pipeline:
        # Keep the surface in step with the pruned tree: don't let the pipeline's
        # own nub-prune drop thin distal vessels we intentionally kept.
        config.PRUNE_SHORT_TERMINAL_NUBS = False
        config.MIN_TERMINAL_LENGTH_MM = 0.0
        print("[CFG] disabled short-terminal-nub pruning for surface run "
              "(surface == pruned tree).")
        from .pipeline import run_pipeline
        run_pipeline(out_xml, out / "surface")

    print(f"[DONE] pruned graph -> {out_xml}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
