"""Import-only smoke test for the package."""

from __future__ import annotations

import sys

print("python:", sys.executable)
print("Importing coronary_sdf modules...")
import coronary_sdf
from coronary_sdf import config
from coronary_sdf.parse_amira import parse_xml
from coronary_sdf.topology import (
    find_connected_components,
    split_by_graph,
    merge_degree2_segments,
    merge_split_multifurcations,
    node_id_canon_map,
    build_nx_tree,
    label_capsules_by_cross_section,
    build_directed_topology,
)
from coronary_sdf.smoothing import (
    smooth_centerline,
    smooth_radius_transitions,
    densify_sparse_segments,
)
from coronary_sdf.pruning import (
    segment_mean_radius,
    prune_short_terminal_nubs,
    prune_by_radius,
)
from coronary_sdf.splines import prepare_segment_spline, branch_tangent_at_node
from coronary_sdf.bif_trim import taper_bifurcation_carina
from coronary_sdf.capsules import build_capsules, CapsuleArrays
from coronary_sdf.sdf_field import (
    smooth_min_exp,
    adaptive_smin_k,
    collect_endpoint_info,
    build_adjacency,
    find_bifurcations,
    compute_grid,
    build_narrow_band,
    evaluate_sdf,
)
from coronary_sdf.mesh_extract import (
    extract_isosurface,
    cut_non_adjacent_bridges,
    create_flat_caps,
    mesh_from_sdf_poisson,
    mesh_from_sdf_meshlib,
    fast_contour_zero,
)
from coronary_sdf.mesh_repair import (
    repair_mesh,
    radius_constrained_taubin,
    fast_clean_triangle_mesh,
)
from coronary_sdf.viz import (
    debug_show_mesh,
    debug_show_capsule_tree,
    debug_show_sdf_preview,
    debug_show_blend_paths,
    debug_show_blend_diagnostics,
    debug_show_capsule_tubes,
    debug_show_raw_data_contours,
    debug_show_smoothed_centerlines,
)
from coronary_sdf.region_vtk import (
    label_surface_topology,
    generate_region_vtk,
    emit_region_vtk_for_surface,
)
from coronary_sdf.pipeline import run_pipeline, generate_sdf_surface
from coronary_sdf.config import SdfConfig, default_config

print("All modules imported OK.")
cfg = default_config()
print("Sample SdfConfig defaults:")
print("  SDF_CARVE_PARALLEL_COS_THRESHOLD :", cfg.SDF_CARVE_PARALLEL_COS_THRESHOLD)
print("  SDF_MESH_METHOD                  :", cfg.SDF_MESH_METHOD)
print("  SDF_POISSON_DEPTH                :", cfg.SDF_POISSON_DEPTH)
print("  XS_JUNC_NODE_PROXIMITY_FACTOR    :", cfg.XS_JUNC_NODE_PROXIMITY_FACTOR)
print("  BIF_CARINA_ENABLE                :", cfg.BIF_CARINA_ENABLE)
print("  BIF_CARINA_TIP_RADIUS_FACTOR     :", cfg.BIF_CARINA_TIP_RADIUS_FACTOR)
print("  BIF_CARINA_TIP_MIN_MM            :", cfg.BIF_CARINA_TIP_MIN_MM)
print("  BIF_CARINA_TAPER_MAX_PTS         :", cfg.BIF_CARINA_TAPER_MAX_PTS)
print("  SMIN_VARIANT                     :", cfg.SMIN_VARIANT)
print("  SMIN_POLY_K_FACTOR               :", cfg.SMIN_POLY_K_FACTOR)
print("  SMIN_GATE_VARIANT                :", cfg.SMIN_GATE_VARIANT)
print("  DETAILED_BLEND_DIAGNOSTIC        :", cfg.DETAILED_BLEND_DIAGNOSTIC)
print("  FORCE_HARD_MIN_ONLY              :", cfg.FORCE_HARD_MIN_ONLY)
print("  SDF_CARVE_PROTECT_JUNCTION_BALL_TOPO_AWARE :", cfg.SDF_CARVE_PROTECT_JUNCTION_BALL_TOPO_AWARE)

# ── Functional check: build_directed_topology on a synthetic Y ──────────────
# Root segment 0 (Strahler 2) connects nodes 1<->2.
# Daughters 1, 2 connect nodes 2<->3, 2<->4 (both Strahler 1).
synth_segments = [
    {"id": 0, "node1": 1, "node2": 2, "strahler": 2, "point_ids": []},
    {"id": 1, "node1": 2, "node2": 3, "strahler": 1, "point_ids": []},
    {"id": 2, "node1": 2, "node2": 4, "strahler": 1, "point_ids": []},
]
synth_node_to_segs = {1: {0}, 2: {0, 1, 2}, 3: {1}, 4: {2}}
dt = build_directed_topology(synth_segments, synth_node_to_segs)
assert dt["parent_seg_idx"][0] == -1, "Root must have parent -1"
assert dt["parent_seg_idx"][1] == 0, "Daughter 1 parent must be root"
assert dt["parent_seg_idx"][2] == 0, "Daughter 2 parent must be root"
assert dt["is_parent"][1, 0] and dt["is_parent"][2, 0], "Daughters see root as parent"
assert dt["is_child"][0, 1] and dt["is_child"][0, 2], "Root sees daughters as children"
assert dt["is_sibling"][1, 2] and dt["is_sibling"][2, 1], "Daughters are siblings"
assert not dt["is_sibling"][1, 1], "Diagonal must be False"
print("  build_directed_topology synthetic Y check: PASS")

# ── merge_split_multifurcations checks ──────────────────────────────────────
# Coordinates in micrometers (matches parse_amira convention); radii in
# micrometers too (the function divides by 1000 internally via _endpoint_radius_mm
# and RADIUS_SCALE).
def _mk_points(records):
    # records: list of (pid, x_mm, y_mm, z_mm, r_mm)
    return {
        pid: (x * 1000.0, y * 1000.0, z * 1000.0, r * 1000.0 / max(config.RADIUS_SCALE, 1e-9))
        for pid, x, y, z, r in records
    }

# Case A: split trifurcation. Trunk Strahler=3 enters node 1.
# Stub 0.05 mm (well under 0.6 * r_min ~ 0.6*0.4 = 0.24 mm) connects two
# degree-3 nodes (B and D). Three "real" daughters all at Strahler=2.
nodes_a = {
    1: (0.0, 0.0, 0.0, 1),      # trunk root (terminal here)
    2: (0.0, 0.0, 2.0, 3),      # proximal bif (B)
    3: (0.0, 0.0, 2.05, 3),     # distal bif (D), 0.05 mm above B along trunk
    4: (-1.0, 0.0, 3.0, 1),     # daughter 1 (off B)
    5: (1.0, 0.0, 3.0, 1),      # daughter 2 (off D)
    6: (0.5, 1.0, 3.0, 1),      # daughter 3 (off D)
}
points_a = _mk_points([
    (10, 0.0, 0.0, 0.0, 0.4), (11, 0.0, 0.0, 2.0, 0.4),
    (20, 0.0, 0.0, 2.0, 0.4), (21, 0.0, 0.0, 2.05, 0.4),  # stub
    (30, 0.0, 0.0, 2.0, 0.4), (31, -1.0, 0.0, 3.0, 0.3),
    (40, 0.0, 0.0, 2.05, 0.4), (41, 1.0, 0.0, 3.0, 0.3),
    (50, 0.0, 0.0, 2.05, 0.4), (51, 0.5, 1.0, 3.0, 0.3),
])
segs_a = [
    {"id": 0, "node1": 1, "node2": 2, "point_ids": [10, 11], "strahler": 3},   # trunk
    {"id": 1, "node1": 2, "node2": 3, "point_ids": [20, 21], "strahler": 3},   # stub (continuator)
    {"id": 2, "node1": 2, "node2": 4, "point_ids": [30, 31], "strahler": 2},   # daughter 1
    {"id": 3, "node1": 3, "node2": 5, "point_ids": [40, 41], "strahler": 2},   # daughter 2
    {"id": 4, "node1": 3, "node2": 6, "point_ids": [50, 51], "strahler": 2},   # daughter 3
]
_, segs_a_out, n_a, _ = merge_split_multifurcations(
    nodes_a, points_a, segs_a, max_len_factor=0.6, require_strahler=True,
    tangent_cos_min=0.7,
)
assert n_a == 1, f"split trifurcation should collapse 1 stub, got {n_a}"
assert len(segs_a_out) == 4, f"expected 4 segments after collapse, got {len(segs_a_out)}"
print("  merge_split_multifurcations split-trifurcation collapse: PASS")

# Case B: two REAL adjacent bifs (trunk continues through one of D's daughters
# at Strahler=3). Should NOT collapse despite the same length / tangent.
segs_b = [
    {"id": 0, "node1": 1, "node2": 2, "point_ids": [10, 11], "strahler": 3},
    {"id": 1, "node1": 2, "node2": 3, "point_ids": [20, 21], "strahler": 3},   # stub
    {"id": 2, "node1": 2, "node2": 4, "point_ids": [30, 31], "strahler": 2},
    {"id": 3, "node1": 3, "node2": 5, "point_ids": [40, 41], "strahler": 3},   # trunk continues!
    {"id": 4, "node1": 3, "node2": 6, "point_ids": [50, 51], "strahler": 2},
]
_, segs_b_out, n_b, _ = merge_split_multifurcations(
    nodes_a, points_a, segs_b, max_len_factor=0.6, require_strahler=True,
    tangent_cos_min=0.7,
)
assert n_b == 0, f"real adjacent bifs must NOT collapse, got {n_b}"
print("  merge_split_multifurcations real-adjacent-bifs reject: PASS")

# Case C: Strahler signature matches a split trifurcation, but the stub turns
# sharply (90 deg from trunk axis). Tangent guard should reject.
nodes_c = dict(nodes_a)
nodes_c[3] = (0.05, 0.0, 2.0, 3)   # stub goes sideways instead of axial
points_c = _mk_points([
    (10, 0.0, 0.0, 0.0, 0.4), (11, 0.0, 0.0, 2.0, 0.4),
    (20, 0.0, 0.0, 2.0, 0.4), (21, 0.05, 0.0, 2.0, 0.4),     # stub sideways
    (30, 0.0, 0.0, 2.0, 0.4), (31, -1.0, 0.0, 3.0, 0.3),
    (40, 0.05, 0.0, 2.0, 0.4), (41, 1.0, 0.0, 3.0, 0.3),
    (50, 0.05, 0.0, 2.0, 0.4), (51, 0.5, 1.0, 3.0, 0.3),
])
_, segs_c_out, n_c, _ = merge_split_multifurcations(
    nodes_c, points_c, segs_a, max_len_factor=0.6, require_strahler=True,
    tangent_cos_min=0.7,
)
assert n_c == 0, f"sharp-turn stub must NOT collapse (tangent guard), got {n_c}"
print("  merge_split_multifurcations sharp-turn reject: PASS")

# ── densify_sparse_segments check ───────────────────────────────────────────
# A 5-point segment spanning 10 mm with radii 0.3 mm should densify to >= 40
# points after the pass. First/last point ids are preserved.
sparse_points = {
    100: (0.0,    0.0, 0.0, 300.0),    # micrometers; 0.3 mm radius
    101: (2500.0, 0.0, 0.0, 300.0),
    102: (5000.0, 0.0, 0.0, 300.0),
    103: (7500.0, 0.0, 0.0, 300.0),
    104: (10000.0, 0.0, 0.0, 300.0),
}
sparse_segs = [{"id": 0, "node1": 1, "node2": 2, "point_ids": [100, 101, 102, 103, 104]}]
new_points, n_d = densify_sparse_segments(
    sparse_points, sparse_segs,
    target_spacing_mm=0.2, min_points=40,
)
assert n_d == 1, f"sparse segment should densify, got n_d={n_d}"
new_pids = sparse_segs[0]["point_ids"]
assert len(new_pids) >= 40, f"densified count >= 40 expected, got {len(new_pids)}"
assert new_pids[0] == 100 and new_pids[-1] == 104, "endpoint pids must be preserved"
# Interior points must have y=z=0 (linearly interpolated along x axis) and
# radius ~= 300 (constant in this case).
for pid in new_pids[1:-1]:
    x, y, z, r = new_points[pid]
    assert abs(y) < 1e-9 and abs(z) < 1e-9, f"interp y/z must be 0, got ({y},{z})"
    assert 0.0 <= x <= 10000.0, f"interp x in [0, 10000], got {x}"
    assert abs(r - 300.0) < 1e-6, f"interp radius ~= 300, got {r}"
print("  densify_sparse_segments basic densification: PASS")

# A dense segment (already 50 points along 5 mm) should NOT be densified.
dense_points = {200 + i: (i * 100.0, 0.0, 0.0, 250.0) for i in range(50)}
dense_segs = [{"id": 1, "node1": 3, "node2": 4, "point_ids": list(range(200, 250))}]
_, n_d2 = densify_sparse_segments(
    dense_points, dense_segs,
    target_spacing_mm=0.2, min_points=40,
)
assert n_d2 == 0, f"already-dense segment must NOT densify, got n_d2={n_d2}"
assert dense_segs[0]["point_ids"] == list(range(200, 250)), "dense segment point_ids unchanged"
print("  densify_sparse_segments already-dense skip: PASS")

print("OK")
