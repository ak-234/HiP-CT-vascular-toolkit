"""Top-level orchestrator for the SDF lumen pipeline.

    parse XML
      |
      v
    Strahler filter -> degree-2 contraction -> terminal-nub pruning
      |
      v
    split into connected-component graphs
      |
      v  (per graph)
    radius transition smoothing
      |
      v
    adjacency + bifurcation set + endpoint info
      |
      v
    spline prep -> capsule sampling -> cross-section junction labels
      |
      v
    grid + narrow band -> evaluate_sdf (with carve)
      |
      v
    extract iso-surface (Poisson | MeshLib | MC)
      |
      v
    bridge cut (optional) -> triangulate / clean -> radius-Taubin
      |
      v
    repair_mesh (optional) -> flat caps (optional)
      |
      v
    save STL + VTK + region-tagged VTK
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
import pyvista as pv

from .config import SdfConfig, default_config, runtime_config as config, use_config
from .pipeline_report import (
    ComponentReport,
    PipelineReport,
    PipelineReportBuilder,
    component_suffix,
    config_digest,
    surface_telemetry,
)
from .parse_amira import (
    parse_xml,
    find_degenerate_segments,
    report_ring_gap_attribution,
)
from .pruning import (
    report_segment_radius_range,
    prune_short_terminal_nubs,
)
from .topology import (
    merge_degree2_segments,
    merge_split_multifurcations,
    split_by_graph,
    split_unsampled_jumps,
    drop_small_components,
    label_capsules_by_cross_section,
    node_id_canon_map,
    build_directed_topology,
)
from .smoothing import (
    densify_sparse_segments,
    bridge_centerline_gaps,
    smooth_segment_centerlines,
    limit_centerline_curvature,
    smooth_segment_radii,
    smooth_radius_transitions,
    prune_terminal_shrink,
    prune_bifurcation_shrink,
)
from .splines import prepare_segment_spline, branch_tangent_at_node
from .bif_trim import taper_bifurcation_carina
from .capsules import (
    build_capsules,
    clamp_terminal_capsule_radii,
    precompensate_capsule_radii,
)
from .implicit_field import build_graph_implicit_field, evaluate_implicit_on_grid
from .legacy_field_adapter import LegacyPointField
from .adaptive_octree import RadiusAdaptiveOctree
from .adaptive_mesher import mesh_adaptive_implicit
from .cgal_adapter import mesh_cgal_implicit
from .vtk_htg_mesher import mesh_vtk_hyper_tree_grid
from .geometry_constraints import find_capsule_conflicts
from .centerline_optimizer import smooth_centerlines_constrained_multiscale
from .sdf_field import (
    build_adjacency,
    find_bifurcations,
    collect_endpoint_info,
    build_terminal_set,
    compute_grid,
    build_narrow_band,
    evaluate_sdf,
    report_non_adjacent_proximity,
)
from .mesh_extract import (
    extract_isosurface,
    cut_non_adjacent_bridges,
    create_flat_caps,
    drop_nonfinite_vertices,
)
from .mesh_repair import repair_mesh, radius_constrained_taubin
from .mesh_validation import enforce_mesh_validation
from .viz import (
    debug_show_capsule_tree,
    debug_show_sdf_preview,
    debug_show_blend_paths,
    debug_show_blend_diagnostics,
    debug_show_mesh,
    debug_show_capsule_tubes,
    debug_show_raw_data_contours,
    debug_show_smoothed_centerlines,
)
from .region_vtk import emit_region_vtk_for_surface


# ── Per-graph SDF surface generation ─────────────────────────────────────────


def _record_path(sink: list[str] | None, path: Path) -> None:
    """Record a file the reconstruction actually wrote, for the manifest."""

    if sink is not None:
        sink.append(str(path))


def _label_junction_capsules(capsules, nodes, points, segments) -> np.ndarray:
    """Per-capsule junction flags used by the legacy field.

    Shared by the dense path and the legacy point oracle so both evaluate the
    same field; the oracle needs it before the dense grid exists.
    """

    if not config.USE_CROSS_SECTION_BLEND_GATE:
        return np.zeros(capsules.n, dtype=bool)
    t_xs = time.time()
    cap_is_junction = label_capsules_by_cross_section(
        capsules.midpoints, nodes, points, segments
    )
    print(
        f"  Cross-section labelling: {int(cap_is_junction.sum()):,}/"
        f"{len(cap_is_junction):,} junction capsules ({time.time()-t_xs:.1f}s)"
    )
    return cap_is_junction


def _shared_node_arrays(shared_pos: dict, n_segs: int) -> tuple[np.ndarray, np.ndarray]:
    """Dense ``(n_segs, n_segs)`` views of the shared-node lookup."""

    shared_node_pos = np.full((n_segs, n_segs, 3), np.nan, dtype=np.float64)
    shared_node_has = np.zeros((n_segs, n_segs), dtype=bool)
    for (si, sj), pos in shared_pos.items():
        shared_node_pos[si, sj] = pos
        shared_node_has[si, sj] = True
    return shared_node_pos, shared_node_has


def _component_report(
    graph_id: int,
    status: str,
    g_nodes: dict[int, tuple],
    g_points: dict[int, tuple],
    g_segments: list[dict[str, Any]],
    surface: pv.PolyData | None,
    written: list[str],
    *,
    error: BaseException | None = None,
) -> ComponentReport:
    """Summarise one component's outcome for the run manifest."""

    telemetry = surface_telemetry(surface)
    return ComponentReport(
        graph_id=int(graph_id),
        status=status,
        n_nodes=len(g_nodes),
        n_points=len(g_points),
        n_segments=len(g_segments),
        segment_ids=tuple(
            int(segment.get("id", index))
            for index, segment in enumerate(g_segments)
        ),
        mesh_points=int(surface.n_points) if surface is not None else 0,
        mesh_faces=int(surface.n_cells) if surface is not None else 0,
        field_method=str(config.SDF_FIELD_METHOD),
        mesh_method=str(config.SDF_MESH_METHOD),
        primitive_method=str(config.IMPLICIT_PRIMITIVE_METHOD),
        voxel_size_mm=telemetry.pop("voxel_size_mm", None),
        cells_across_diameter=telemetry.pop("cells_across_diameter", None),
        length_scale_mm=telemetry.pop("length_scale_mm", None),
        telemetry=telemetry,
        output_paths=tuple(written),
        error_type=type(error).__name__ if error is not None else None,
        error=str(error) if error is not None else None,
    )


def _generate_sdf_surface(
    nodes: dict[int, tuple],
    points: dict[int, tuple],
    segments: list[dict[str, Any]],
    output_dir: Path,
    graph_id: int = 0,
    *,
    write_outputs: bool = True,
    interactive: bool = False,
    written_paths: list[str] | None = None,
) -> pv.PolyData | None:
    """Build the smooth-min capsule SDF + iso-surface mesh for a single graph.

    Replaces the legacy ``generate_sdf_surface`` (B-Spline SDF Surface
    pipeline). Always uses ``SdfPipeline=SDF``; other surface methods are
    dropped in this package.
    """
    print(f"\n[SDF] Generating surface for graph {graph_id} (optimized vectorized)...")
    report_segment_radius_range(f"graph-{graph_id}", segments, points)
    res_str = "auto" if config.BSPLINE_SDF_RESOLUTION is None else f"{config.BSPLINE_SDF_RESOLUTION} mm"
    print(f"  Centerline smoother: {config.CENTERLINE_SMOOTHER}")
    print(f"  SDF resolution: {res_str}")
    print(f"  Field method: {config.SDF_FIELD_METHOD}")
    print(f"  Smooth-min k: {config.BSPLINE_SMIN_K}")
    t_start = time.time()

    out = Path(output_dir)
    if write_outputs:
        out.mkdir(parents=True, exist_ok=True)

    if config.REPORT_RING_GAP_ATTRIBUTION:
        report_ring_gap_attribution(
            points, segments,
            radius_scale=config.RADIUS_SCALE,
            gap_alpha=config.GAP_VIS_ALPHA,
            big_jump_ratio=config.GAP_BIG_JUMP_RATIO,
            max_gap_um=config.CENTERLINE_MAX_GAP_UM,
            gap_ratio=config.CENTERLINE_GAP_RATIO,
            label=f"Graph {graph_id}",
        )

    if interactive:
        debug_show_raw_data_contours(
            nodes, points, segments,
            title=f"Graph {graph_id}: 1. Raw Data Contours",
        )

    # Centerline smoothing first — smooths each segment's (x, y, z) coords
    # in place. Pinned endpoints keep node positions stable for downstream
    # node-to-pids distance checks in the radius passes.
    if str(config.CENTERLINE_SMOOTHER).lower() == "constrained_multiscale":
        points, smoothing_report = smooth_centerlines_constrained_multiscale(
            nodes, points, segments
        )
        print(
            "  [CENTERLINE] constrained multiscale: "
            f"moved={smoothing_report.modified_points}, "
            f"max_drift/r={smoothing_report.max_displacement_radius:.4f}, "
            f"curvature={smoothing_report.curvature_violations_before}"
            f"->{smoothing_report.curvature_violations_after}, "
            f"self_contact={smoothing_report.self_distance_violations_before}"
            f"->{smoothing_report.self_distance_violations_after}, "
            f"input_overlaps={smoothing_report.input_overlaps}, "
            f"unresolved={smoothing_report.unresolved_constraints}, "
            f"converged={smoothing_report.converged} "
            f"({smoothing_report.iterations} iterations)"
        )
        if (
            smoothing_report.unresolved_constraints
            and str(config.CENTERLINE_CONSTRAINT_FAILURE).lower() != "report"
        ):
            raise ValueError("constrained centreline smoothing remained infeasible")
        if smoothing_report.unresolved_constraints:
            print(
                "  [CENTERLINE][WARN] constrained smoothing did not satisfy "
                f"{smoothing_report.unresolved_constraints} constraint(s); "
                "the run is diagnostic-only unless this is resolved"
            )
    else:
        points, _ = smooth_segment_centerlines(nodes, points, segments)

    # Straighten sharp bends where the centreline radius of curvature drops below
    # the local vessel radius (the swept tube would self-intersect and the
    # same-segment hard union would fuse the arms, deleting the inner wall).
    if (
        config.LIMIT_CENTERLINE_CURVATURE
        and str(config.CENTERLINE_SMOOTHER).lower() != "constrained_multiscale"
    ):
        points, _ = limit_centerline_curvature(nodes, points, segments)

    if interactive:
        debug_show_raw_data_contours(
            nodes, points, segments,
            title=f"Graph {graph_id}: 2. After Centerline Smoothing",
        )

    # Radius smoothing next: denoise the in-segment r(s) profile first (with
    # endpoints pinned), then prune endpoint shrink outliers (terminals +
    # bifurcations) so smooth_radius_transitions sees anatomically meaningful
    # endpoint values and its mean-blend doesn't drag the parent down toward
    # a noisy daughter.
    if config.PRESERVE_INPUT_RADII:
        print("  Input radii: preserved (all radius rewrite passes disabled)")
    else:
        points, _ = smooth_segment_radii(nodes, points, segments)
        points, _ = prune_bifurcation_shrink(nodes, points, segments)
        points, _ = prune_terminal_shrink(nodes, points, segments)
        points, _ = smooth_radius_transitions(nodes, points, segments)

    if interactive:
        debug_show_raw_data_contours(
            nodes, points, segments,
            title=f"Graph {graph_id}: 3. After Radius Smoothing",
        )

    # node_to_segs (used by adjacency + bif + endpoint info).
    node_to_segs: dict[int, set[int]] = {}
    for idx, seg in enumerate(segments):
        for nid in (seg["node1"], seg["node2"]):
            node_to_segs.setdefault(nid, set()).add(idx)

    # Terminal endpoints + their KDTree.
    if config.SDF_FLAT_TERMINAL_CAPS:
        endpoint_info = collect_endpoint_info(nodes, points, segments, node_to_segs)
        term = build_terminal_set(endpoint_info)
        if term.tree is not None:
            print(f"  Flat-cap terminals: {len(term.pos)} endpoints")
    else:
        endpoint_info = []
        term = build_terminal_set([])

    # Segment-pair adjacency + bifurcation set.
    adj_matrix, shared_pos, _shared_r = build_adjacency(nodes, points, segments, node_to_segs)
    bif = find_bifurcations(nodes, points, segments, node_to_segs)

    # Directed topology (parent / child / sibling masks) for the t-projection
    # smin gate per sdf_plan.md. Strahler-rooted BFS through shared-node
    # adjacency. The masks are (n_segs, n_segs) bool; consumed by
    # sdf_field.evaluate_sdf when SMIN_GATE_VARIANT == "t_projection".
    _dtopo = build_directed_topology(segments, node_to_segs)
    is_parent = _dtopo["is_parent"]
    is_child = _dtopo["is_child"]
    is_sibling = _dtopo["is_sibling"]
    print(
        f"  [DIR TOPO] roots: {int((_dtopo['parent_seg_idx'] < 0).sum())} | "
        f"parent edges: {int(is_parent.sum())} | "
        f"sibling pairs: {int(is_sibling.sum() // 2)}"
    )

    # Bif-to-segment incidence matrix: bif_seg_incident[bif_idx, seg_idx] is
    # True iff seg is one of the segments meeting at that bif node. Used by
    # sdf_field.evaluate_sdf to make SDF_CARVE_PROTECT_JUNCTION_BALL
    # topology-aware: shield the carve only for rivals that are actually
    # incident on the nearest bif, so accidental fusion between unrelated
    # sub-trees that happen to pass close to a bif still gets carved.
    n_segs_full = len(segments)
    n_bifs = int(len(bif.node_ids)) if bif.node_ids is not None else 0
    bif_seg_incident = np.zeros((n_bifs, n_segs_full), dtype=bool)
    for _bi in range(n_bifs):
        _nid = int(bif.node_ids[_bi])
        for _sj in node_to_segs.get(_nid, set()):
            if 0 <= _sj < n_segs_full:
                bif_seg_incident[_bi, _sj] = True

    # Per-segment splines.
    print(f"  Preparing {len(segments)} segment splines...")
    seg_id_to_idx = {seg["id"]: i for i, seg in enumerate(segments)}
    segment_splines: list[dict[str, Any] | None] = []
    for seg in segments:
        sp = prepare_segment_spline(seg, points, nodes)
        if sp is not None:
            sp["seg_idx"] = seg_id_to_idx[seg["id"]]
            segment_splines.append(sp)
        else:
            segment_splines.append(None)
    valid_splines = [s for s in segment_splines if s is not None]
    if not valid_splines:
        print("[WARN] No valid segments for surface")
        return None
    print(f"  {len(valid_splines)} valid segment splines")

    # Bifurcation carina taper: at each spline endpoint that terminates at
    # a deg>=3 node, rewrite the last K radii with a linear taper from the
    # interior reference radius down to a small "carina tip" at the bif
    # endpoint. Converts hemispherical capsule end-caps into conical tips so
    # the N-ary log-sum-exp smooth-min in sdf_field.evaluate_sdf constructs
    # a clean Y/T/X carina rather than a ball-shaped union of hemispheres.
    # Mutates only the radii of dicts in valid_splines (coords unchanged).
    if config.BIF_CARINA_ENABLE and not config.PRESERVE_INPUT_RADII:
        taper_report = taper_bifurcation_carina(
            valid_splines, nodes, points, segments
        )
        print(
            f"  [BIF CARINA] tapered {taper_report['n_tapered_endpoints']} endpoints, "
            f"modified {taper_report['n_radii_modified']} radii"
        )
        if config.BIF_CARINA_VERBOSE:
            for rec in taper_report["records"]:
                print(
                    f"    tapered  seg {str(rec['seg_id']):>5}  "
                    f"node {str(rec['node_id']):>5}  deg {rec['deg']}  "
                    f"{rec['end']:<5}  K={rec['k_tapered']}  "
                    f"r_interior={rec['r_interior']:.4f}  "
                    f"r_tip={rec['r_tip']:.4f}"
                )

    # Per-segment endpoint positions + tangents for bifurcation wedge gating.
    # Coords reach the bif node again (taper modifies only radii), so the
    # spline endpoint and the bif node coincide.
    n_segs = len(segments)
    seg_end_pos = np.full((n_segs, 2, 3), np.nan, dtype=np.float64)
    seg_end_tan = np.full((n_segs, 2, 3), np.nan, dtype=np.float64)
    seg_end_tan_ok = np.zeros((n_segs, 2), dtype=bool)
    for seg, sp in zip(segments, segment_splines):
        if sp is None:
            continue
        si = int(sp["seg_idx"])
        n1 = seg["node1"]
        n2 = seg["node2"]
        if n1 in nodes:
            seg_end_pos[si, 0] = np.array(nodes[n1][:3], dtype=np.float64) / 1000.0
        if n2 in nodes:
            seg_end_pos[si, 1] = np.array(nodes[n2][:3], dtype=np.float64) / 1000.0
        t1 = branch_tangent_at_node(sp, n1, seg)
        if t1 is not None:
            seg_end_tan[si, 0] = t1
            seg_end_tan_ok[si, 0] = True
        t2 = branch_tangent_at_node(sp, n2, seg)
        if t2 is not None:
            seg_end_tan[si, 1] = t2
            seg_end_tan_ok[si, 1] = True

    if interactive:
        debug_show_smoothed_centerlines(
            valid_splines,
            title=f"Graph {graph_id}: Smoothed Centerlines",
        )

    # Capsules.
    print("  Building capsule representation...")
    t_cap = time.time()
    # Terminal-contour clamp: force radii[0] / radii[-1] of every spline
    # whose endpoint is a degree-1 node to >= median of the next-N
    # interior radii. Uses canonicalised node IDs so near-coincident
    # Amira node duplicates aren't falsely treated as terminals.
    _canon = node_id_canon_map(nodes, config.NODE_COINCIDENCE_EPS_MM)
    canonical_segs: dict[int, set[int]] = {}
    for nid, segs in node_to_segs.items():
        canonical_segs.setdefault(_canon.get(nid, nid), set()).update(segs)
    terminal_node_ids = {
        nid for nid in node_to_segs
        if len(canonical_segs.get(_canon.get(nid, nid), set())) == 1
    }
    print(f"  [TERMINAL CLAMP] terminal_node_ids count: {len(terminal_node_ids)}")
    if config.PRESERVE_INPUT_RADII:
        term_report = {"n_checked": 0, "n_clamped": 0, "records": []}
    else:
        term_report = clamp_terminal_capsule_radii(valid_splines, terminal_node_ids)
    print(
        f"  [TERMINAL CLAMP] checked {term_report['n_checked']} terminal endpoints, "
        f"clamped {term_report['n_clamped']}"
    )
    if config.TERMINAL_CLAMP_VERBOSE:
        for rec in term_report["records"]:
            marker = "CLAMPED" if rec["clamped"] else "ok     "
            print(
                f"    {marker}  seg {str(rec['seg_id']):>5}  "
                f"node {str(rec['node_id']):>5}  "
                f"{rec['end']:<5}  "
                f"r_before={rec['radius_before']:.4f}  "
                f"interior_ref={rec['interior_ref']:.4f}  "
                f"r_after={rec['radius_after']:.4f}"
            )
    capsules = build_capsules(valid_splines, node_to_segs=node_to_segs)
    print(f"  {capsules.n} capsules in {time.time()-t_cap:.1f}s")

    if config.IMPLICIT_GEOMETRY_CHECK:
        conflicts = find_capsule_conflicts(
            capsules,
            adj_matrix,
            shared_node_positions=shared_pos,
            shared_node_radii=_shared_r,
            clearance_fraction=config.IMPLICIT_GEOMETRY_CLEARANCE_FRACTION,
            maximum_records=config.IMPLICIT_GEOMETRY_MAX_REPORT,
        )
        if conflicts:
            print(
                f"  [GEOMETRY] {len(conflicts)} nonlocal tube conflict(s) "
                "found (report may be capped)"
            )
            for conflict in conflicts:
                print(
                    f"    seg {conflict.segment_a} vs {conflict.segment_b}: "
                    f"clearance/r={conflict.normalized_clearance:.3f}, "
                    f"clearance={conflict.clearance:.6g} mm"
                )
            if config.IMPLICIT_FAIL_ON_GEOMETRY_CONFLICT:
                raise ValueError(
                    "fixed-radius tube geometry overlaps outside a junction; "
                    "no scalar union can preserve separation"
                )

    graph_field = None
    adaptive_hierarchy = None
    field_method = str(config.SDF_FIELD_METHOD).lower()
    if field_method not in {"legacy", "graph_implicit"}:
        raise ValueError(
            f"Unknown SDF_FIELD_METHOD={config.SDF_FIELD_METHOD!r}; "
            "expected 'legacy' or 'graph_implicit'"
        )
    mesh_method = str(config.SDF_MESH_METHOD).lower()
    adaptive_mesh_requested = mesh_method == "adaptive"
    vtk_htg_mesh_requested = mesh_method == "vtk_htg"
    cgal_mesh_requested = mesh_method == "cgal_mesh3"
    if (
        field_method == "graph_implicit"
        or config.IMPLICIT_ADAPTIVE_AUDIT
        or adaptive_mesh_requested
        or vtk_htg_mesh_requested
        or cgal_mesh_requested
    ):
        t_field = time.time()
        graph_field = build_graph_implicit_field(
            capsules,
            nodes,
            node_to_segs,
            blend_fraction=config.IMPLICIT_JUNCTION_BLEND_FRACTION,
            support_factor=config.IMPLICIT_JUNCTION_SUPPORT_FACTOR,
            bvh_leaf_size=config.IMPLICIT_BVH_LEAF_SIZE,
            primitive_method=config.IMPLICIT_PRIMITIVE_METHOD,
            clip_bifurcation_caps=config.SDF_FLAT_CAP_BIF,
        )
        print(
            f"  Graph implicit field: {len(graph_field.junctions)} junctions, "
            f"{len(graph_field.bvh.nodes):,} BVH nodes in {time.time()-t_field:.1f}s"
        )

    # Which field the grid-free extractors sample. For the graph field this is
    # the field itself; for the legacy field it is a point-query adapter that
    # shares this same object for sizing, so the two ablation cells build an
    # identical hierarchy and differ only in the value returned.
    extraction_field = graph_field
    legacy_length_scale: float | None = None
    if (
        field_method == "legacy"
        and (adaptive_mesh_requested or vtk_htg_mesh_requested or cgal_mesh_requested)
    ):
        assert graph_field is not None
        # Pin the field's internal length scale to what the matching dense run
        # would use, so 'legacy + dense' and 'legacy + adaptive' evaluate the
        # same field and the comparison isolates the extractor.
        legacy_length_scale = (
            float(config.BSPLINE_SDF_RESOLUTION)
            if config.BSPLINE_SDF_RESOLUTION is not None
            else float(compute_grid(capsules).voxel_size)
        )
        shared_pos_array, shared_has_array = _shared_node_arrays(shared_pos, n_segs)
        extraction_field = LegacyPointField(
            capsules,
            length_scale=legacy_length_scale,
            sizing_field=graph_field,
            cap_is_junction=_label_junction_capsules(capsules, nodes, points, segments),
            adj_matrix=adj_matrix,
            bif=bif,
            term=term,
            shared_node_pos=shared_pos_array,
            shared_node_has=shared_has_array,
            seg_end_pos=seg_end_pos,
            seg_end_tan=seg_end_tan,
            seg_end_tan_ok=seg_end_tan_ok,
            is_parent=is_parent,
            is_child=is_child,
            is_sibling=is_sibling,
            bif_seg_incident=bif_seg_incident,
            prune_mode=str(config.LEGACY_ORACLE_PRUNE_MODE),
        )
        print(
            f"  Legacy point oracle: length_scale={legacy_length_scale:.6g} mm, "
            f"prune_mode={config.LEGACY_ORACLE_PRUNE_MODE}"
        )
    if config.IMPLICIT_ADAPTIVE_AUDIT and graph_field is not None:
        t_octree = time.time()
        adaptive_hierarchy = RadiusAdaptiveOctree(
            graph_field,
            cells_across_diameter=config.IMPLICIT_CELLS_ACROSS_DIAMETER,
            maximum_depth=config.IMPLICIT_ADAPTIVE_MAX_DEPTH,
            maximum_active_leaves=config.IMPLICIT_ADAPTIVE_MAX_LEAVES,
        )
        hierarchy_stats = adaptive_hierarchy.build()
        print(
            "  Adaptive implicit audit: "
            f"{hierarchy_stats.active_leaves:,} active leaves, "
            f"{hierarchy_stats.sign_change_leaves:,} sign-changing, "
            f"depth {hierarchy_stats.maximum_depth}, "
            f"{hierarchy_stats.field_evaluations:,} field evaluations "
            f"in {time.time()-t_octree:.1f}s"
        )

    # Compiler-free radius-adaptive production candidate. The Python layer
    # samples the shared field oracle and the prebuilt VTK wheel contours the
    # HyperTreeGrid; the dense grid is not allocated on this path.
    if vtk_htg_mesh_requested:
        assert extraction_field is not None
        t_mesh = time.time()
        surface, htg_stats = mesh_vtk_hyper_tree_grid(
            extraction_field,
            cells_across_diameter=config.IMPLICIT_CELLS_ACROSS_DIAMETER,
            padding_radius_factor=config.VTK_HTG_PADDING_RADIUS_FACTOR,
            maximum_depth=config.IMPLICIT_ADAPTIVE_MAX_DEPTH,
            maximum_cells=config.VTK_HTG_MAX_CELLS,
            decomposed_polyhedra=config.VTK_HTG_DECOMPOSED_POLYHEDRA,
        )
        surface.field_data["field_evaluations"] = np.asarray(
            [htg_stats.field_evaluations], dtype=np.int64
        )
        surface.field_data["hierarchy_seconds"] = np.asarray(
            [htg_stats.hierarchy_seconds], dtype=np.float64
        )
        surface.field_data["extraction_seconds"] = np.asarray(
            [htg_stats.contour_seconds], dtype=np.float64
        )
        surface.field_data["total_seconds"] = np.asarray(
            [time.time() - t_start], dtype=np.float64
        )
        surface.field_data["cells_across_diameter"] = np.asarray(
            [float(config.IMPLICIT_CELLS_ACROSS_DIAMETER)], dtype=np.float64
        )
        print(
            f"  VTK HTG mesh: {surface.n_points:,} vertices, "
            f"{surface.n_cells:,} faces, {htg_stats.leaf_cells:,} leaves "
            f"in {time.time()-t_mesh:.1f}s"
        )
        enforce_mesh_validation(
            surface,
            mode=config.OUTPUT_VALIDATION_MODE,
            expected_components=1,
            check_self_intersections=config.OUTPUT_VALIDATE_SELF_INTERSECTIONS,
        )
        if write_outputs:
            suffix = component_suffix(graph_id)
            for path in (
                out / f"lumen_vtk_htg{suffix}.stl",
                out / f"lumen_vtk_htg{suffix}.vtk",
            ):
                surface.save(str(path))
                _record_path(written_paths, path)
        return surface

    if cgal_mesh_requested:
        assert extraction_field is not None
        if not config.CGAL_SEQUENTIAL:
            raise ValueError("only deterministic sequential CGAL meshing is supported")
        t_mesh = time.time()
        surface = mesh_cgal_implicit(
            extraction_field,
            cells_across_diameter=config.IMPLICIT_CELLS_ACROSS_DIAMETER,
            facet_angle_deg=config.CGAL_FACET_ANGLE_DEG,
            facet_distance_fraction=config.CGAL_FACET_DISTANCE_FRACTION,
            cell_size_factor=config.CGAL_CELL_SIZE_FACTOR,
            cell_radius_edge_ratio=config.CGAL_CELL_RADIUS_EDGE_RATIO,
        )
        surface.compute_normals(inplace=True, consistent_normals=True)
        surface.field_data["extraction_seconds"] = np.asarray(
            [time.time() - t_mesh], dtype=np.float64
        )
        surface.field_data["total_seconds"] = np.asarray(
            [time.time() - t_start], dtype=np.float64
        )
        enforce_mesh_validation(
            surface,
            mode=config.OUTPUT_VALIDATION_MODE,
            expected_components=1,
            check_self_intersections=config.OUTPUT_VALIDATE_SELF_INTERSECTIONS,
        )
        if write_outputs:
            suffix = component_suffix(graph_id)
            for path in (
                out / f"lumen_cgal_implicit{suffix}.stl",
                out / f"lumen_cgal_implicit{suffix}.vtk",
            ):
                surface.save(str(path))
                _record_path(written_paths, path)
        return surface

    if adaptive_mesh_requested:
        assert graph_field is not None
        if adaptive_hierarchy is None:
            t_octree = time.time()
            adaptive_hierarchy = RadiusAdaptiveOctree(
                graph_field,
                cells_across_diameter=config.IMPLICIT_CELLS_ACROSS_DIAMETER,
                maximum_depth=config.IMPLICIT_ADAPTIVE_MAX_DEPTH,
                maximum_active_leaves=config.IMPLICIT_ADAPTIVE_MAX_LEAVES,
            )
            hierarchy_stats = adaptive_hierarchy.build()
            hierarchy_elapsed = time.time() - t_octree
            print(
                f"  Adaptive hierarchy: {hierarchy_stats.active_leaves:,} active "
                f"leaves at depth <= {hierarchy_stats.maximum_depth} "
                f"in {time.time()-t_octree:.1f}s"
            )
        t_mesh = time.time()
        adaptive_mesh = mesh_adaptive_implicit(
            graph_field,
            adaptive_hierarchy,
            maximum_points=config.IMPLICIT_ADAPTIVE_MAX_POINTS,
        )
        surface = adaptive_mesh.to_pyvista().triangulate().clean(tolerance=0.0)
        mesh_elapsed = time.time() - t_mesh
        print(
            f"  Adaptive implicit mesh: {surface.n_points:,} vertices, "
            f"{surface.n_cells:,} faces in {time.time()-t_mesh:.1f}s"
        )
        if config.MESH_REPAIR:
            characteristic_size = min(
                leaf.size for leaf in adaptive_hierarchy.leaves
            )
            surface = repair_mesh(surface, characteristic_size)
        if (
            (config.FLAT_CAP_OUTLETS or config.SDF_FLAT_TERMINAL_CAPS)
            and endpoint_info
        ):
            surface, n_capped = create_flat_caps(surface, endpoint_info)
            print(f"  [FLAT CAPS] Added {n_capped} flat caps")
        surface = surface.triangulate().clean(tolerance=0.0)
        surface.field_data["field_evaluations"] = np.asarray(
            [hierarchy_stats.field_evaluations], dtype=np.int64
        )
        surface.field_data["hierarchy_seconds"] = np.asarray(
            [locals().get("hierarchy_elapsed", 0.0)], dtype=np.float64
        )
        surface.field_data["extraction_seconds"] = np.asarray(
            [mesh_elapsed], dtype=np.float64
        )
        surface.field_data["total_seconds"] = np.asarray(
            [time.time() - t_start], dtype=np.float64
        )
        surface.field_data["cells_across_diameter"] = np.asarray(
            [float(config.IMPLICIT_CELLS_ACROSS_DIAMETER)], dtype=np.float64
        )
        surface.compute_normals(inplace=True, consistent_normals=True)
        boundary_edges = int(
            surface.extract_feature_edges(
                boundary_edges=True,
                feature_edges=False,
                manifold_edges=False,
                non_manifold_edges=False,
            ).n_cells
        )
        if not surface.is_manifold or boundary_edges:
            raise RuntimeError(
                "adaptive implicit mesh failed final validation: "
                f"manifold={surface.is_manifold}, open_edges={boundary_edges}"
            )
        enforce_mesh_validation(
            surface,
            mode=config.OUTPUT_VALIDATION_MODE,
            expected_components=1,
            check_self_intersections=config.OUTPUT_VALIDATE_SELF_INTERSECTIONS,
        )
        if write_outputs:
            suffix = component_suffix(graph_id)
            stl_path = out / f"lumen_adaptive_implicit{suffix}.stl"
            vtk_path = out / f"lumen_adaptive_implicit{suffix}.vtk"
            surface.save(str(stl_path))
            surface.save(str(vtk_path))
            _record_path(written_paths, stl_path)
            _record_path(written_paths, vtk_path)
            print(f"  {stl_path}")
            print(f"  {vtk_path}")
            if config.WRITE_REGION_VTK:
                region_path = out / f"lumen_adaptive_implicit{suffix}_regions.vtk"
                emit_region_vtk_for_surface(
                    surface, nodes, points, segments, str(region_path)
                )
                _record_path(written_paths, region_path)
        return surface

    # Cross-section junction labelling.
    cap_is_junction = _label_junction_capsules(capsules, nodes, points, segments)

    if interactive:
        debug_show_capsule_tree(
            capsules.starts,
            capsules.ends,
            capsules.radii_start,
            capsules.radii_end,
            capsules.seg_idx,
            nodes,
            segments,
            adj_matrix=adj_matrix,
            title=f"Graph {graph_id}: Capsule Tree",
        )
        debug_show_capsule_tubes(
            capsules,
            title=f"Graph {graph_id}: SDF Capsule Tubes",
        )

    # Grid + proximity diagnostic + narrow band.
    grid = compute_grid(capsules)

    # Cancel the marching-cubes radius deficit before the field is evaluated.
    # Must run after compute_grid (needs the voxel size) and before
    # build_narrow_band so the band is padded for the inflated radii.
    if config.SDF_RADIUS_PRECOMPENSATE:
        capsules, max_delta = precompensate_capsule_radii(capsules, grid.voxel_size)
        print(
            f"  Radius pre-compensation: coeff="
            f"{config.SDF_RADIUS_PRECOMPENSATE_COEFF:g}, "
            f"max inflation {max_delta * 1000:.2f} um"
        )
    if config.PROXIMITY_DIAGNOSTIC_TOP_N != 0:
        report_non_adjacent_proximity(
            nodes,
            points,
            segments,
            voxel_size_mm=grid.voxel_size,
            top_n=(
                config.PROXIMITY_DIAGNOSTIC_TOP_N
                if config.PROXIMITY_DIAGNOSTIC_TOP_N > 0
                else None
            ),
        )
    nb_idx = build_narrow_band(capsules, grid)

    # SDF evaluation.
    shared_node_pos, shared_node_has = _shared_node_arrays(shared_pos, n_segs)

    t_evaluate = time.time()
    if field_method == "graph_implicit":
        assert graph_field is not None
        sdf = evaluate_implicit_on_grid(graph_field, grid, nb_idx)
    else:
        sdf = evaluate_sdf(
            capsules=capsules,
            cap_is_junction=cap_is_junction,
            adj_matrix=adj_matrix,
            shared_node_pos=shared_node_pos,
            shared_node_has=shared_node_has,
            seg_end_pos=seg_end_pos,
            seg_end_tan=seg_end_tan,
            seg_end_tan_ok=seg_end_tan_ok,
            is_parent=is_parent,
            is_child=is_child,
            is_sibling=is_sibling,
            bif_seg_incident=bif_seg_incident,
            bif=bif,
            term=term,
            grid=grid,
            nb_idx=nb_idx,
        )
    field_elapsed = time.time() - t_evaluate

    # Pre-MC preview.
    if config.PREVIEW_SDF_BEFORE_MC and interactive:
        _pv = sdf.path_vol if config.BLEND_DIAGNOSTIC else None
        iso = (-grid.voxel_size * 2, 0.0, grid.voxel_size * 2)
        debug_show_sdf_preview(
            sdf.sdf,
            grid.bbox_min,
            grid.bbox_max,
            grid.voxel_size,
            path_vol=_pv,
            iso_levels=iso,
            title=f"Graph {graph_id}: SDF Preview (pre-MC)",
        )
    if config.BLEND_DIAGNOSTIC and interactive and sdf.path_vol is not None:
        debug_show_blend_paths(
            sdf.path_vol,
            grid.bbox_min,
            grid.voxel_size,
            valid_splines=valid_splines,
            blend_weight_vol=sdf.blend_weight_vol,
            title=f"Graph {graph_id}: Blend Paths",
        )

    # Iso-surface extraction.
    t3 = time.time()
    print(f"  Extracting iso-surface (method={config.SDF_MESH_METHOD!r})...")
    surface = extract_isosurface(
        sdf.sdf, grid.bbox_min, grid.voxel_size, grid.dims,
        capsule_tree=capsules.tree, cap_max_radii=capsules.max_radii,
    )
    extraction_elapsed = time.time() - t3
    print(
        f"  Iso-surface: {surface.n_points:,} vertices, "
        f"{surface.n_cells:,} faces in {time.time()-t3:.1f}s"
    )

    # Drop NaN/inf vertices (degenerate meshlib relaxation on sub-voxel vessels
    # can emit non-finite coords) so every downstream consumer — bridge cut,
    # post-processing, Taubin's capsule KD-tree query, repair, caps — sees a
    # finite mesh.
    n_verts_before_drop = int(surface.n_points)
    surface, n_bad_verts = drop_nonfinite_vertices(surface)
    if n_bad_verts:
        bad_frac = n_bad_verts / max(n_verts_before_drop, 1)
        print(f"  [WARN] dropped {n_bad_verts:,} non-finite mesh vertices "
              f"({100 * bad_frac:.2f}% of {n_verts_before_drop:,}) "
              f"-> {surface.n_points:,} verts remain")
        # Dropping a vertex also drops every face touching it, so a large
        # fraction does not "repair" the mesh -- it perforates it, splitting a
        # single-component lumen into many pieces and letting validation rays
        # escape through the holes. That is a failed extraction, not a warning.
        max_frac = float(getattr(config, "MESH_MAX_NONFINITE_FRACTION", 1.0))
        if bad_frac > max_frac:
            raise ValueError(
                f"extraction produced {n_bad_verts:,} non-finite vertices "
                f"({100 * bad_frac:.2f}% of {n_verts_before_drop:,}), above the "
                f"{100 * max_frac:.2f}% limit set by MESH_MAX_NONFINITE_FRACTION; "
                "the surface would be perforated rather than repaired"
            )

    if surface.n_points == 0:
        print("[WARN] No surface generated!")
        return None

    # Mesh-level bridge cut.
    if config.MESH_CUT_NON_ADJACENT_BRIDGES:
        t_bcut = time.time()
        surface, n_bridges = cut_non_adjacent_bridges(surface, segments, points, adj_matrix)
        if n_bridges > 0:
            print(
                f"  Bridge cut: removed {n_bridges:,} non-adjacent bridge faces "
                f"in {time.time()-t_bcut:.1f}s"
            )
        else:
            print("  Bridge cut: no bridges detected")

    # Post-processing: triangulate, clean, optional largest-component, normals.
    print("  Post-processing...")
    surface = surface.triangulate()
    # tolerance=0: meshlib MC / FlyingEdges MC / Open3D Poisson all return
    # shared-vertex meshes already. Any nonzero tolerance fuses opposite-wall
    # verts across thin microvessels (radius can be < voxel_size) and turns a
    # manifold mesh into one with thousands of open edges.
    surface = surface.clean(tolerance=0.0)

    # Speckle cull. Marching cubes leaves a tail of sub-voxel fragments where the
    # narrow band clips a capsule or two rivals nearly touch. They are geometric
    # dust -- measured at 0.004-0.01% of surface area each -- but they dominate
    # the raw component count (27-65 components of which only 3 exceed 0.01% of
    # area), which makes beta0 useless as a topology check and hides genuine
    # breaks. Dropping by SIZE keeps every substantive piece, unlike
    # SDF_KEEP_LARGEST_ONLY which would also discard the real ~1.1%-of-area
    # second component this tree reproducibly contains at every resolution.
    # Implemented in numpy/scipy rather than via vtkConnectivityFilter +
    # extract_cells: those crashed the process here (silent exit, no traceback,
    # so not catchable by try/except) on the full pre-Taubin mesh, while working
    # standalone on smaller ones. Labelling with scipy's connected_components
    # over the triangle-edge graph is pure Python-level code, cheap at this size,
    # and cannot take the interpreter down.
    min_frac = float(getattr(config, "MESH_MIN_COMPONENT_AREA_FRACTION", 0.0))
    if min_frac > 0.0 and surface.n_cells:
        try:
            import scipy.sparse as _sp
            from scipy.sparse.csgraph import connected_components as _cc

            tri = surface.faces.reshape(-1, 4)[:, 1:]
            pts = np.asarray(surface.points)
            n_pts = len(pts)
            rows = np.concatenate([tri[:, 0], tri[:, 1], tri[:, 2]])
            cols = np.concatenate([tri[:, 1], tri[:, 2], tri[:, 0]])
            adj = _sp.coo_matrix(
                (np.ones(len(rows), dtype=np.int8), (rows, cols)),
                shape=(n_pts, n_pts),
            )
            n_reg, labels = _cc(adj, directed=False)
            face_lab = labels[tri[:, 0]]
            v0, v1, v2 = pts[tri[:, 0]], pts[tri[:, 1]], pts[tri[:, 2]]
            area = 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0), axis=1)
            sizes = np.bincount(face_lab, weights=area, minlength=n_reg)
            total = float(sizes.sum())
            if n_reg > 1 and total > 0.0:
                keep_reg = sizes / total >= min_frac
                keep_face = keep_reg[face_lab]
                if keep_face.any() and not keep_face.all():
                    new_tri = tri[keep_face]
                    used = np.unique(new_tri)
                    remap = np.full(n_pts, -1, dtype=np.int64)
                    remap[used] = np.arange(len(used))
                    surface = pv.PolyData(
                        pts[used],
                        np.column_stack([
                            np.full(len(new_tri), 3, dtype=np.int64),
                            remap[new_tri],
                        ]).ravel(),
                    )
                    print(
                        f"    Speckle cull: dropped "
                        f"{int(n_reg - keep_reg.sum())} of {n_reg} components "
                        f"(< {min_frac * 100:g}% of area, "
                        f"{total - float(sizes[keep_reg].sum()):.4f} mm^2); "
                        f"{int(keep_reg.sum())} kept"
                    )
        except Exception as e:
            print(f"    [WARN] Speckle cull failed: {e}")

    if config.SDF_KEEP_LARGEST_ONLY:
        try:
            surface = surface.connectivity(extraction_mode="largest")
            if not isinstance(surface, pv.PolyData):
                surface = surface.extract_surface()
        except Exception as e:
            print(f"    [WARN] Connectivity extraction failed: {e}")
    surface.compute_normals(inplace=True, consistent_normals=True)

    # Radius-constrained Taubin smoothing.
    surface = radius_constrained_taubin(
        surface,
        capsules.tree,
        capsules.max_radii,
        grid.voxel_size,
        term_pos=term.pos,
        term_nrm=term.nrm,
        term_rad=term.rad,
        term_tree=term.tree,
        bif_pos=bif.positions,
        bif_rad=bif.radii,
    )

    # Mesh repair.
    if config.MESH_REPAIR:
        print("  Mesh repair...")
        surface = repair_mesh(surface, grid.voxel_size)

    # Flat caps (post-repair so caps are not altered by the repair pass).
    if config.FLAT_CAP_OUTLETS and endpoint_info:
        can_cap = True
        if config.FLAT_CAP_REQUIRE_MANIFOLD and not surface.is_manifold:
            print("  [FLAT CAPS][WARN] Skipping: surface is non-manifold after repair")
            can_cap = False
        if config.FLAT_CAP_MAX_FACES and surface.n_cells > config.FLAT_CAP_MAX_FACES:
            print(
                f"  [FLAT CAPS][WARN] Skipping: {surface.n_cells:,} faces exceeds "
                f"limit ({config.FLAT_CAP_MAX_FACES:,})"
            )
            can_cap = False
        if can_cap:
            try:
                surface, n_capped = create_flat_caps(surface, endpoint_info)
                print(f"  [FLAT CAPS] Added {n_capped} flat caps")
            except Exception as e:
                print(f"  [FLAT CAPS][WARN] Failed: {e}; keeping uncapped mesh")

    # Summary + save.
    print(
        f"  Final mesh: {surface.n_points:,} vertices, {surface.n_cells:,} faces"
    )
    print(f"  Manifold: {surface.is_manifold}")
    try:
        n_open = int(
            surface.extract_feature_edges(
                boundary_edges=True,
                feature_edges=False,
                manifold_edges=False,
                non_manifold_edges=False,
            ).n_cells
        )
        if n_open > 0:
            print(f"  Open edges: {n_open}")
    except Exception:
        pass
    print(f"  Total time: {time.time()-t_start:.1f}s")

    surface.field_data["field_evaluations"] = np.asarray(
        [len(nb_idx)], dtype=np.int64
    )
    surface.field_data["field_seconds"] = np.asarray(
        [field_elapsed], dtype=np.float64
    )
    surface.field_data["extraction_seconds"] = np.asarray(
        [extraction_elapsed], dtype=np.float64
    )
    surface.field_data["total_seconds"] = np.asarray(
        [time.time() - t_start], dtype=np.float64
    )
    surface.field_data["voxel_size_mm"] = np.asarray(
        [float(grid.voxel_size)], dtype=np.float64
    )

    enforce_mesh_validation(
        surface,
        mode=config.OUTPUT_VALIDATION_MODE,
        expected_components=1,
        check_self_intersections=config.OUTPUT_VALIDATE_SELF_INTERSECTIONS,
    )

    if write_outputs:
        suffix = component_suffix(graph_id)
        stl_path = out / f"lumen_bspline{suffix}.stl"
        vtk_path = out / f"lumen_bspline{suffix}.vtk"
        print("\n[SAVE] Writing mesh files...")
        surface.save(str(stl_path))
        surface.save(str(vtk_path))
        _record_path(written_paths, stl_path)
        _record_path(written_paths, vtk_path)
        print(f"  {stl_path}")
        print(f"  {vtk_path}")

        if config.WRITE_REGION_VTK:
            region_path = out / f"lumen_bspline{suffix}_regions.vtk"
            try:
                emit_region_vtk_for_surface(surface, nodes, points, segments, str(region_path))
                _record_path(written_paths, region_path)
            except Exception as e:
                print(f"  [REGION VTK][WARN] Failed: {e}")

    if interactive:
        debug_show_mesh(surface, title=f"Graph {graph_id}: B-Spline SDF Surface")
        if getattr(config, "DETAILED_BLEND_DIAGNOSTIC", False):
            debug_show_blend_diagnostics(
                sdf,
                surface,
                grid,
                bif,
                title=f"Graph {graph_id}: Blend Diagnostics",
            )
    return surface


# ── Top-level entry point ────────────────────────────────────────────────────


def _run_pipeline(
    xml_path: str | Path,
    output_dir: str | Path,
    *,
    write_outputs: bool,
    interactive: bool,
    return_report: bool = False,
) -> dict[int, pv.PolyData] | tuple[dict[int, pv.PolyData], PipelineReport]:
    """Parse the spatial-graph XML and generate surfaces for every connected
    component. Returns ``{graph_id: PolyData}``.
    """
    out = Path(output_dir)
    if write_outputs:
        out.mkdir(parents=True, exist_ok=True)

    nodes, points, segments = parse_xml(xml_path)
    report_segment_radius_range("parse", segments, points)

    if config.VALIDATE_CENTERLINE_GEOMETRY and config.CENTERLINE_VALIDATION_MODE != "off":
        bad = find_degenerate_segments(
            points,
            segments,
            max_gap_um=config.CENTERLINE_MAX_GAP_UM,
            gap_ratio=config.CENTERLINE_GAP_RATIO,
        )
        if bad:
            print(
                f"  [GEOMETRY WARNING] {len(bad)} segment(s) have degenerate "
                f"centerlines (interior points collapsed / large internal gaps)."
            )
            for d in bad[:10]:
                print(
                    f"    seg id={d['id']} nodes={d['node1']}->{d['node2']} "
                    f"pts={d['n_points']} max_gap={d['max_gap_um'] / 1000:.2f}mm "
                    f"ratio={d['gap_ratio']:.2f}"
                )
            print(
                "    Likely a bad .am export; prefer the .xml or regenerate the "
                "resampled graph."
            )
            if config.CENTERLINE_VALIDATION_MODE == "error":
                raise ValueError(
                    f"{len(bad)} degenerate centerline segment(s) detected in "
                    f"{xml_path}; aborting (CENTERLINE_VALIDATION_MODE='error')."
                )

    if config.MIN_STRAHLER_ORDER > 0:
        before = len(segments)
        segments = [s for s in segments if s.get("strahler", 0) >= config.MIN_STRAHLER_ORDER]
        if len(segments) < before:
            print(
                f"  Strahler filter (>= {config.MIN_STRAHLER_ORDER}): "
                f"{before} -> {len(segments)} segments"
            )
        report_segment_radius_range("after-strahler", segments, points)

    if config.MERGE_DEGREE2_SEGMENTS:
        before = len(segments)
        nodes, segments, n_merged = merge_degree2_segments(nodes, points, segments)
        if n_merged > 0:
            print(
                f"  Degree-2 contraction: merged {n_merged} segments "
                f"({before} -> {len(segments)})"
            )
        report_segment_radius_range("after-merge", segments, points)

    if config.MERGE_SPLIT_MULTIFURCATIONS:
        before = len(segments)
        nodes, segments, n_collapsed, records = merge_split_multifurcations(
            nodes,
            points,
            segments,
            max_len_factor=config.SPLIT_MULTIFURC_MAX_LEN_FACTOR,
            require_strahler=config.SPLIT_MULTIFURC_REQUIRE_STRAHLER,
            tangent_cos_min=config.SPLIT_MULTIFURC_TANGENT_COS_MIN,
        )
        if n_collapsed > 0:
            print(
                f"  Split-multifurcation contraction: collapsed {n_collapsed} stubs "
                f"({before} -> {len(segments)} segments)"
            )
        if config.BIF_MERGE_VERBOSE and records:
            for rec in records:
                print(f"    {rec}")
        report_segment_radius_range("after-multifurc-merge", segments, points)

    if config.PRUNE_SHORT_TERMINAL_NUBS and config.MIN_TERMINAL_LENGTH_MM > 0:
        before = len(segments)
        nodes, segments, n_pruned = prune_short_terminal_nubs(
            nodes,
            points,
            segments,
            min_length_mm=config.MIN_TERMINAL_LENGTH_MM,
            max_iters=config.PRUNE_ITER_MAX,
        )
        if n_pruned > 0:
            print(
                f"  Pruned {n_pruned} short terminal segments "
                f"(< {config.MIN_TERMINAL_LENGTH_MM} mm): {before} -> {len(segments)}"
            )
        report_segment_radius_range("after-prune", segments, points)

    if config.BRIDGE_CENTERLINE_GAPS:
        before_bridge = len(points)
        points, n_bridged, _ = bridge_centerline_gaps(
            points,
            segments,
            target_spacing_mm=config.DENSIFY_TARGET_SPACING_MM,
            big_jump_ratio=config.GAP_BIG_JUMP_RATIO,
            min_gap_um=config.CENTERLINE_MAX_GAP_UM,
            radius_scale=config.RADIUS_SCALE,
            verbose=config.DENSIFY_VERBOSE,
        )
        if n_bridged > 0:
            print(
                f"  Bridged {n_bridged} centerline gap(s) "
                f"(+{len(points) - before_bridge} interpolated points)"
            )

    # Before densify, and mutually exclusive with bridging: densify would otherwise
    # sample a span that is about to stop existing, and the two passes give opposite
    # answers to the same question.
    if config.SPLIT_UNSAMPLED_JUMPS and not config.BRIDGE_CENTERLINE_GAPS:
        before_seg = len(segments)
        nodes, segments, n_cut, _ = split_unsampled_jumps(
            nodes,
            points,
            segments,
            big_jump_ratio=config.GAP_BIG_JUMP_RATIO,
            min_gap_um=config.CENTERLINE_MAX_GAP_UM,
            step_ratio=config.SPLIT_JUMP_STEP_RATIO,
            radius_scale=config.RADIUS_SCALE,
            verbose=config.DENSIFY_VERBOSE,
        )
        if n_cut > 0:
            print(
                f"  Cut {n_cut} unsampled jump(s) rather than bridging them "
                f"({before_seg} -> {len(segments)} segments)"
            )
        report_segment_radius_range("after-split", segments, points)

    if config.MIN_COMPONENT_LENGTH_MM > 0.0 or config.MIN_COMPONENT_LENGTH_FRACTION > 0.0:
        nodes, segments, _dropped = drop_small_components(
            nodes, points, segments,
            min_length_mm=config.MIN_COMPONENT_LENGTH_MM,
            min_fraction_of_largest=config.MIN_COMPONENT_LENGTH_FRACTION,
        )
        if _dropped:
            report_segment_radius_range("after-drop", segments, points)

    if config.DENSIFY_SPARSE_SEGMENTS:
        before_n_pts = len(points)
        points, n_densified = densify_sparse_segments(
            points,
            segments,
            target_spacing_mm=config.DENSIFY_TARGET_SPACING_MM,
            min_points=config.DENSIFY_MIN_POINTS,
            verbose=config.DENSIFY_VERBOSE,
        )
        if n_densified > 0:
            added = len(points) - before_n_pts
            print(
                f"  Densified {n_densified} sparse segments "
                f"(+{added} interpolated points)"
            )

    graphs = split_by_graph(nodes, points, segments)
    print(f"  {len(graphs)} graph(s) detected")

    surfaces: dict[int, pv.PolyData] = {}
    builder = PipelineReportBuilder(
        requested=sorted(graphs),
        graph_source=str(xml_path),
        output_dir=str(out),
        config_digest=config_digest(config),
    )
    # A component that fails must not make the run indistinguishable from a
    # single-component graph, so the manifest is written on every exit path.
    fail_fast = str(config.PIPELINE_COMPONENT_FAILURE).lower() == "error"
    try:
        for gid, (g_nodes, g_points, g_segments) in sorted(graphs.items()):
            print("\n" + "=" * 60)
            print(f"  PROCESSING GRAPH {gid}")
            print(
                f"  {len(g_nodes)} nodes, {len(g_points)} points, {len(g_segments)} segments"
            )
            print("=" * 60)
            written: list[str] = []
            try:
                surface = _generate_sdf_surface(
                    g_nodes,
                    g_points,
                    g_segments,
                    out,
                    gid,
                    write_outputs=write_outputs,
                    interactive=interactive,
                    written_paths=written,
                )
            except Exception as exc:
                builder.record(
                    _component_report(
                        gid,
                        "failed",
                        g_nodes,
                        g_points,
                        g_segments,
                        None,
                        written,
                        error=exc,
                    )
                )
                print(
                    f"  [COMPONENT][FAIL] graph {gid}: "
                    f"{type(exc).__name__}: {exc}"
                )
                if fail_fast:
                    raise
                continue
            builder.record(
                _component_report(
                    gid,
                    "ok" if surface is not None else "empty",
                    g_nodes,
                    g_points,
                    g_segments,
                    surface,
                    written,
                )
            )
            if surface is not None:
                surfaces[gid] = surface
    finally:
        report = builder.finish()
        if write_outputs:
            report.write(out)

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Processed {len(graphs)} graph(s), generated {len(surfaces)} surface(s)")
    for gid, surf in surfaces.items():
        print(f"    Graph {gid}: {surf.n_points:,} vertices, {surf.n_cells:,} faces")
    if report.incomplete:
        missing = sorted(set(report.requested_components) - set(report.completed_components))
        print(f"  [INCOMPLETE] graph component(s) {missing} were not emitted")
    if return_report:
        return surfaces, report
    return surfaces


def generate_sdf_surface(
    nodes: dict[int, tuple],
    points: dict[int, tuple],
    segments: list[dict[str, Any]],
    output_dir: str | Path,
    graph_id: int = 0,
    *,
    cfg: SdfConfig | None = None,
    write_outputs: bool = True,
    interactive: bool | None = None,
) -> pv.PolyData | None:
    """Generate one component using an immutable, context-local configuration."""

    resolved = default_config() if cfg is None else cfg
    show = resolved.DEBUG_VIS if interactive is None else bool(interactive)
    with use_config(resolved):
        return _generate_sdf_surface(
            nodes,
            points,
            segments,
            Path(output_dir),
            graph_id,
            write_outputs=bool(write_outputs),
            interactive=show,
        )


def run_pipeline(
    xml_path: str | Path,
    output_dir: str | Path,
    *,
    cfg: SdfConfig | None = None,
    write_outputs: bool = True,
    interactive: bool | None = None,
    return_report: bool = False,
) -> dict[int, pv.PolyData] | tuple[dict[int, pv.PolyData], PipelineReport]:
    """Run all components with a reproducible immutable configuration.

    The default return type remains ``{graph_id: PolyData}``. Pass
    ``return_report=True`` to also receive the :class:`PipelineReport` recording
    which components were requested, completed and failed.
    ``write_outputs=False`` executes the same reconstruction path without
    creating directories or mesh files, which is used by the benchmark harness.
    """

    resolved = default_config() if cfg is None else cfg
    show = resolved.DEBUG_VIS if interactive is None else bool(interactive)
    with use_config(resolved):
        return _run_pipeline(
            xml_path,
            output_dir,
            write_outputs=bool(write_outputs),
            interactive=show,
            return_report=bool(return_report),
        )


__all__ = ["generate_sdf_surface", "run_pipeline"]
