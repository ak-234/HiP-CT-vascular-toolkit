"""End-to-end test of the skeleton_analysis Python port on real LADAF-2024-28 data.

Runs the full pipeline on an Avizo ASCII spatial graph (the skeletonisation) and,
optionally, an Avizo binary segmentation lattice:

  0. read the spatial graph
  1. pick a root per tree  (built-in PyVista picker,
     skeleton_analysis.ordering.pick_roots; or --roots / --auto-roots to skip
     the GUI)
  2. Strahler + topological ordering (forest), validated against the strahler
     field already embedded in the file
  3. metrics: branching angles, intervessel distance, Murray's law, scaling exponent
  4. collapsed-vessel detection + along-segment thickness correction
  5. (optional, --image) oblique cross-section radius correction using the
     segmentation lattice
  6. (optional, --optimisation) clDice centreline sensitivity + bifurcation Dice
     (skeleton vs the segmentation volume)
  7. (optional, --report) full metrics report (plots + Amira-colourable .am) at
     each correction state, with before/after comparison

Examples
--------
    python Python_port_test.py --auto-roots
    python Python_port_test.py                      # opens the root picker, one window per tree
    python Python_port_test.py --roots 21,140       # skip the GUI, use these roots
    python Python_port_test.py --auto-roots --image # also run the (heavy) image stage
    python Python_port_test.py --auto-roots --optimisation   # clDice + bifurcation Dice
    python Python_port_test.py --auto-roots --image --report # 3-state metrics report
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np

from skeleton_analysis.io import read_amira, write_amira, read_amira_lattice, lattice_info
from skeleton_analysis.ordering.pipeline import auto_roots, order_forest
from skeleton_analysis.ordering import pick_roots
from skeleton_analysis.metrics import (
    branching_angles,
    intervessel_distance,
    murray_law,
    exponent_calculation,
    mean_radius_per_edge,
    aggregate_by_strahler,
)
from skeleton_analysis.outlier.detect import (
    detect_collapsed_segments,
    correct_along_segment_thickness,
)
from skeleton_analysis.optimisation import (
    centreline_sensitivity,
    skeleton_junction_points,
    bifurcation_points,
    bifurcation_dice_points,
    region_morphometrics,
)
from skeleton_analysis.metrics.report import (
    edge_metrics_table,
    murray_table,
    assign_kmeans,
    plot_report,
    write_metric_graph,
    compare_states,
)
from skeleton_analysis.utils.split import split_connected_components

# No dataset path is baked in: pass --skeleton/--segmentation, or set these in
# the environment once per shell. Both are required; argparse reports a missing
# one by name.
DEFAULT_SKELETON = os.environ.get("HIPCT_GRAPH")
DEFAULT_SEGMENTATION = os.environ.get("HIPCT_SEG")


def hr(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def dedupe_roots_by_component(graph, roots):
    """Keep one root per connected component (fall back to auto_roots elsewhere)."""
    comps = split_connected_components(graph)
    auto = auto_roots(graph)
    chosen = []
    for comp in comps:
        picked = [r for r in roots if r in comp.node_map]
        if picked:
            chosen.append(picked[0])
        else:
            a = [r for r in auto if r in comp.node_map]
            chosen.append(a[0] if a else min(comp.node_map))
    return chosen


# ── main ─────────────────────────────────────────────────────────────────────
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    # Required unless supplied via HIPCT_GRAPH / HIPCT_SEG, so a run without a
    # dataset fails at argument parsing with a named flag rather than later.
    ap.add_argument("--skeleton", default=DEFAULT_SKELETON,
                    required=DEFAULT_SKELETON is None,
                    help="input Amira SpatialGraph (.am); env: HIPCT_GRAPH")
    ap.add_argument("--segmentation", default=DEFAULT_SEGMENTATION,
                    required=DEFAULT_SEGMENTATION is None,
                    help="matching segmentation lattice (.am); env: HIPCT_SEG")
    ap.add_argument("--out", default="pipeline_test_out")
    ap.add_argument("--roots", default=None, help="comma-separated root node IDs (skip GUI)")
    ap.add_argument("--auto-roots", action="store_true",
                    help="pick roots automatically (largest-radius inlet per tree)")
    ap.add_argument("--image", action="store_true",
                    help="run the image oblique-correction stage (decodes ~2.3 GB)")
    ap.add_argument("--max-oblique-segs", type=int, default=0,
                    help="cap #flagged segments to oblique-correct (0 = all flagged)")
    ap.add_argument("--oblique-radius-method", choices=["perimeter", "area"],
                    default="perimeter", help="cross-section radius estimate to write back")
    ap.add_argument("--oblique-debug", action="store_true",
                    help="save debug PNGs of near-voxel-size oblique cross-sections")
    ap.add_argument("--oblique-debug-dir", default=None,
                    help="debug PNG directory (default <out>/oblique_debug)")
    ap.add_argument("--oblique-debug-radius-vox", type=float, default=2.5,
                    help="save a PNG when the perimeter radius <= this many voxels")
    ap.add_argument("--oblique-debug-max", type=int, default=40,
                    help="maximum number of debug PNGs to write")
    ap.add_argument("--oblique-qc", action="store_true",
                    help="open the interactive 3-D QC viewer per flagged segment ([viz3d]; needs a display)")
    ap.add_argument("--oblique-qc-max", type=int, default=5,
                    help="max #segments to show in the 3-D QC viewer")
    ap.add_argument("--oblique-qc-planes", action="store_true",
                    help="overlay the oblique cutting planes in the 3-D QC viewer")
    ap.add_argument("--optimisation", action="store_true",
                    help="run Stage 6: clDice centreline sensitivity + bifurcation Dice")
    ap.add_argument("--bb-threshold", type=float, default=900.0,
                    help="bifurcation matching distance threshold (world units, um)")
    ap.add_argument("--report", action="store_true",
                    help="Stage 7: full metrics report (plots + Amira .am) before/after correction")
    ap.add_argument("--report-dir", default=None, help="report directory (default <out>/report)")
    ap.add_argument("--kmeans-k", type=int, default=None,
                    help="fixed k for the k-means cluster order (default: auto via silhouette)")
    args = ap.parse_args(argv)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # ── Stage 0: read ────────────────────────────────────────────────────────
    hr("STAGE 0  read spatial graph")
    graph = read_amira(args.skeleton)
    print(f"file: {args.skeleton}")
    print(f"V/E/P: {graph.n_vertices}/{graph.n_edges}/{graph.n_points}")
    print("edge fields:", list(graph.edge_fields))
    print("consistency:", graph.check_consistency() or "OK")
    embedded_strahler = (
        np.asarray(graph.edge_fields["strahler"]).astype(int)
        if "strahler" in graph.edge_fields else None
    )

    # ── Stage 1: roots ───────────────────────────────────────────────────────
    hr("STAGE 1  root selection")
    if args.roots:
        roots = [int(x) for x in args.roots.split(",")]
        print("roots from --roots:", roots)
    elif args.auto_roots:
        roots = auto_roots(graph)
        print("auto roots (largest-radius inlet per tree):", roots)
    else:
        try:
            print("[PICK] click the inlet (root) segment of each tree "
                  "(q=next tree, x=stop)")
            roots = pick_roots(graph)
            print("picked roots:", roots)
        except Exception as exc:
            print(f"[WARN] picker unavailable ({exc}); falling back to auto roots.")
            roots = auto_roots(graph)
            print("auto roots:", roots)
    roots = dedupe_roots_by_component(graph, roots)
    print("roots (one per tree):", roots)

    # ── Stage 2: order + validate ────────────────────────────────────────────
    hr("STAGE 2  Strahler + topological ordering (forest)")
    edges = np.asarray(graph.edge_connectivity, dtype=np.int64)
    strahler, topo, flipped = order_forest(edges, roots)
    print(f"edges flipped to orient toward roots: {flipped.size}")
    u, c = np.unique(strahler, return_counts=True)
    print("Strahler order -> #edges:", dict(zip(u.tolist(), c.tolist())))
    print("max generation:", int(topo.max()))
    if embedded_strahler is not None:
        both = (strahler > 0) & (embedded_strahler > 0)
        match = float(np.mean(strahler[both] == embedded_strahler[both])) if both.any() else float("nan")
        print(f"VALIDATION: our Strahler matches embedded on "
              f"{match*100:.1f}% of {int(both.sum())} edges")
    graph.set_edge_field("strahler", strahler)
    graph.set_edge_field("topo", topo)
    ordered_path = out / "LADAF_2024_28_ordered.am"
    write_amira(graph, ordered_path)
    print("written:", ordered_path)

    # Snapshot BEFORE any collapsed-vessel correction (has strahler/topo, original radius).
    snap_before = graph.copy() if args.report else None

    # ── Stage 3: metrics (per tree) ──────────────────────────────────────────
    hr("STAGE 3  metrics")
    if "MeanRadius" not in graph.edge_fields:
        graph.set_edge_field("MeanRadius", mean_radius_per_edge(graph))
    comps = split_connected_components(graph)
    metric_graphs, metric_roots = [], []
    for ci, comp in enumerate(comps):
        root_global = next((r for r in roots if r in comp.node_map), None)
        if root_global is None:
            continue
        rl = comp.node_map[root_global]
        metric_graphs.append(comp.graph)
        metric_roots.append(rl)
        try:
            ba_edge, _ = branching_angles(comp.graph, root_id=rl)
            angs = ba_edge[~np.isnan(ba_edge)]
            df = murray_law(comp.graph, root_id=rl)
            print(f"tree {ci}: {comp.graph.n_edges} edges | "
                  f"branch angles n={angs.size} mean={np.nanmean(angs):.1f} deg | "
                  f"Murray branch points={len(df)} median gamma={df['gamma_eff'].median():.2f}")
        except Exception as exc:
            print(f"tree {ci}: metric error: {exc}")
    ivd = intervessel_distance(graph)
    print(f"intervessel distance (all edges): min={ivd.min():.1f} "
          f"median={np.median(ivd):.1f} max={ivd.max():.1f}")
    try:
        exp = exponent_calculation(metric_graphs, root_ids=metric_roots, radius_field="MeanRadius")
        n_valid = int(np.sum(~np.isnan(exp.log_data).any(axis=1)))
        print(f"radius-scaling exponent (pooled RMA): {exp.exponent:.3f} (n={n_valid})")
    except Exception as exc:
        print(f"exponent error: {exc}")

    # Strahler aggregation table.
    import pandas as pd
    seg_df = pd.DataFrame({
        "strahler": strahler,
        "MeanRadius": np.asarray(graph.edge_fields["MeanRadius"], dtype=float),
    })
    seg_df = seg_df[seg_df["strahler"] > 0]
    print("\nper-Strahler radius summary:")
    print(aggregate_by_strahler(seg_df, ["MeanRadius"]).round(2).to_string())

    # ── Stage 4: outlier detect + thickness correction ───────────────────────
    hr("STAGE 4  collapsed-vessel detection + thickness correction")
    flagged = detect_collapsed_segments(graph, flag_orders=(5,4,3,2,1),strahler_field="strahler",
                                        radius_field="MeanRadius")
    print(f"collapsed-vessel candidates flagged: {flagged.size} edges")
    corrected_thickness, changed = correct_along_segment_thickness(graph)
    print(f"along-segment thickness points corrected: {changed.size}")
    graph.set_point_field("thickness", corrected_thickness)

    # Snapshot AFTER along-segment thickness correction (Stage 4).
    snap_thickness = graph.copy() if args.report else None

    # ── Stages 5-6 both need the segmentation volume; decode it once. ─────────
    lat = None
    if args.image or args.optimisation:
        hr("IMAGE  decode segmentation lattice")
        print("segmentation:", args.segmentation)
        print("lattice info:", lattice_info(args.segmentation))
        print("decoding Labels volume (~2.3 GB, please wait) ...")
        lat = read_amira_lattice(args.segmentation, block="Labels")
        print("volume:", lat.volume.shape, "spacing(um):", lat.spacing.round(2))

    # ── Stage 5: image oblique correction (optional) ─────────────────────────
    if args.image:
        hr("STAGE 5  oblique cross-section radius correction (image)")
        from skeleton_analysis.outlier.oblique import segment_radii_from_volume
        res = float(np.mean(lat.spacing))
        nump = np.asarray(graph.num_edge_points, dtype=np.int64)
        starts = np.concatenate([[0], np.cumsum(nump)[:-1]])
        pcoords = np.asarray(graph.point_coords, dtype=float)
        thickness = np.asarray(graph.thickness, dtype=float).copy()

        todo = flagged if args.max_oblique_segs <= 0 else flagged[: args.max_oblique_segs]

        debug = None
        if args.oblique_debug:
            dbg_dir = Path(args.oblique_debug_dir) if args.oblique_debug_dir else (out / "oblique_debug")
            debug = {"dir": dbg_dir, "radius_vox_max": args.oblique_debug_radius_vox,
                     "budget": args.oblique_debug_max, "saved": 0, "prefix": ""}
            print(f"debug PNGs -> {dbg_dir}  (r_perim <= {args.oblique_debug_radius_vox} vox, "
                  f"max {args.oblique_debug_max})")

        # Keep Amira's MeanRadius; only the oblique-corrected edges get updated
        # (thickness/MeanRadius are on different scales here, so a global
        # re-derivation would rescale untouched edges ~2.5x).
        mean_radius = np.asarray(graph.edge_fields["MeanRadius"], float).copy()
        print(f"obliquing {len(todo)} flagged segment(s), method='{args.oblique_radius_method}' "
              f"(this is the slow step) ...")
        n_seg, n_pts = 0, 0
        for e in todo:
            s, n = int(starts[e]), int(nump[e])
            centre_vox = lat.world_to_index_zyx(pcoords[s : s + n])  # (n,3) as (iz,iy,ix)
            if debug is not None:
                debug["prefix"] = f"edge{int(e)}_"
            rads = segment_radii_from_volume(lat.volume, centre_vox, res=res,
                                             half_size=12, threshold=0.5,
                                             method=args.oblique_radius_method, debug=debug)
            finite = np.isfinite(rads)
            if not finite.any():
                continue  # whole segment failed -> keep prior thickness
            seg = thickness[s : s + n]
            seg[finite] = rads[finite]
            thickness[s : s + n] = seg
            mean_radius[int(e)] = float(np.mean(seg))  # corrected edge only
            n_seg += 1
            n_pts += int(finite.sum())

            # Optional interactive 3-D QC (blocking; needs a display).
            if args.oblique_qc and n_seg <= args.oblique_qc_max:
                from skeleton_analysis.outlier.viz3d import show_segment_volume
                show_segment_volume(lat.volume, centre_vox, half_pad=12,
                                    show_planes=args.oblique_qc_planes,
                                    title=f"edge {int(e)}  (corrected r~{np.nanmedian(rads):.0f} um)")

        graph.set_point_field("thickness", thickness)
        graph.set_edge_field("MeanRadius", mean_radius)
        print(f"segments corrected: {n_seg} | thickness points overwritten: {n_pts}")
        print(f"MeanRadius updated on {n_seg} corrected edge(s); "
              f"untouched edges keep Amira's original value")
        if debug is not None:
            print(f"debug PNGs saved: {debug['saved']}")
        corrected_path = out / "LADAF_2024_28_radius_corrected.am"
        write_amira(graph, corrected_path)
        print("written:", corrected_path)
        snap_oblique = graph.copy() if args.report else None
    else:
        snap_oblique = None
        print("\n(STAGE 5 image oblique correction skipped; pass --image to run it.)")

    # ── Stage 6: optimisation metrics (skeleton vs segmentation) ─────────────
    if args.optimisation:
        hr("STAGE 6  optimisation metrics (skeleton vs segmentation)")
        sens = centreline_sensitivity(graph, lat)
        print(f"clDice centreline sensitivity: {sens:.4f} "
              f"(fraction of skeleton points inside the segmentation)")
        print("skeletonising segmentation to derive reference bifurcations ...")
        ref_pts = skeleton_junction_points(lat)
        cand_pts = bifurcation_points(graph)
        res_bb = bifurcation_dice_points(cand_pts, ref_pts, threshold=args.bb_threshold)
        print(f"bifurcation Dice: {res_bb.dice:.4f}  "
              f"(TP={res_bb.tp} FP={res_bb.fp} FN={res_bb.fn}; "
              f"candidate bifs={res_bb.n_candidate} reference bifs={res_bb.n_reference}; "
              f"threshold={args.bb_threshold:.0f} um)")
        # Whole-volume morphometrics (Python replacement for the Fiji/MorphoLibJ macro).
        vox = float(np.mean(lat.spacing))
        morph = region_morphometrics(lat.volume, voxel_size=vox)
        print(f"segmentation morphometrics: components={morph['connected_components']} | "
              f"euler={morph['euler_number']} | volume={morph['volume']:.3e} um^3 | "
              f"surface_area={morph['surface_area']:.3e} um^2 "
              f"(feeds super_metric / meta-metric)")
    else:
        print("\n(STAGE 6 optimisation metrics skipped; pass --optimisation to run it.)")

    # ── Stage 7: full metrics report, before/after correction ────────────────
    if args.report:
        hr("STAGE 7  metrics report (before/after correction)")
        report_dir = Path(args.report_dir) if args.report_dir else (out / "report")
        states = {"before": snap_before, "after_thickness": snap_thickness}
        if snap_oblique is not None:
            states["after_oblique"] = snap_oblique

        tables = {}
        for name, gsnap in states.items():
            table = edge_metrics_table(gsnap, roots)
            labels, kk = assign_kmeans(table, k=args.kmeans_k)
            table["kmeans_cluster"] = labels
            mdf = murray_table(gsnap, roots)
            plot_report(table, mdf, report_dir / name)
            write_metric_graph(gsnap, table, report_dir / f"{name}_metrics.am")
            tables[name] = table
            print(f"[{name}] {len(table)} edges | k-means order k={kk} | "
                  f"mean radius={table['radius'].mean():.1f} -> {report_dir / name}")

        compare_states(tables, report_dir / "comparison")
        print(f"report (plots + *_metrics.am + comparison) -> {report_dir}")
    else:
        print("\n(STAGE 7 metrics report skipped; pass --report to run it.)")

    hr("DONE")
    print(f"outputs in: {out.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
