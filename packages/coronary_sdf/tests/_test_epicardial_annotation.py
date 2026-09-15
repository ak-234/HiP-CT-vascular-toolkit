"""Smoke tests for epicardial_annotation (run: python -m coronary_sdf._test_epicardial_annotation).

Covers the parts that need no display and no heavy SDF pipeline:
  1. Amira XML writer round-trips through parse_amira.parse_xml.
  2. Radius-ratio subtree prune keeps main vessels + prunes thin side branches.
  3. Containment prune removes a swallowed stub and keeps a *shorter* genuine
     distal vessel (length-independence).
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np

from coronary_sdf.parse_amira import parse_xml
from coronary_sdf.topology import find_connected_components
from coronary_sdf.epicardial_annotation import (
    write_amira_xml,
    prune_by_radius_ratio,
    prune_contained_leaves,
    compute_vessel_ostia,
    compute_vessel_radius_stats,
    _segment_contour_mesh,
)


def _all_pids(segments) -> set[int]:
    out: set[int] = set()
    for s in segments:
        out.update(s["point_ids"])
    return out


# ── 1. Writer round-trip ───────────────────────────────────────────────────────

def test_writer_roundtrip() -> None:
    nodes = {
        0: (0.0, 0.0, 0.0, 1),
        1: (0.0, 0.0, 5000.0, 3),
        2: (3000.0, 0.0, 8000.0, 1),
        3: (-3000.0, 0.0, 8000.0, 1),
    }
    points = {
        0: (0.0, 0.0, 0.0, 2000.0),
        1: (0.0, 0.0, 2500.0, 2000.0),
        2: (0.0, 0.0, 5000.0, 2000.0),
        3: (1500.0, 0.0, 6500.0, 1000.0),
        4: (3000.0, 0.0, 8000.0, 1000.0),
        5: (-1500.0, 0.0, 6500.0, 1000.0),
        6: (-3000.0, 0.0, 8000.0, 1000.0),
    }
    segments = [
        {"id": 0, "node1": 0, "node2": 1, "point_ids": [0, 1, 2], "strahler": 3},
        {"id": 1, "node1": 1, "node2": 2, "point_ids": [3, 4], "strahler": 2},
        {"id": 2, "node1": 1, "node2": 3, "point_ids": [5, 6], "strahler": 2},
    ]

    with tempfile.TemporaryDirectory() as td:
        xml = Path(td) / "rt.am.xml"
        write_amira_xml(nodes, points, segments, xml)
        n2, p2, s2 = parse_xml(xml)

    assert len(n2) == 4, n2
    assert len(p2) == 7, p2
    assert len(s2) == 3, s2
    # node degrees recomputed correctly
    assert {nid: n2[nid][3] for nid in n2} == {0: 1, 1: 3, 2: 1, 3: 1}
    # points preserved (coords + thickness)
    for pid, rec in points.items():
        for a, b in zip(rec, p2[pid]):
            assert abs(a - b) < 1e-3, (pid, rec, p2[pid])
    # segments preserved (by id)
    by_id = {s["id"]: s for s in s2}
    for s in segments:
        assert by_id[s["id"]]["point_ids"] == s["point_ids"]
        assert by_id[s["id"]]["strahler"] == s["strahler"]
        assert {by_id[s["id"]]["node1"], by_id[s["id"]]["node2"]} == {s["node1"], s["node2"]}
    print("  [1] writer round-trip: PASS")


# ── Shared synthetic trunk + 2 side branches + 1 main daughter ─────────────────

def _ratio_fixture():
    # trunk (r=2) -> bif node 1; main daughter (r=1), side1 (r=0.3), side2 (r=0.15)
    nodes = {
        0: (0.0, 0.0, 0.0, 1),
        1: (0.0, 0.0, 10000.0, 4),
        2: (0.0, 0.0, 20000.0, 1),
        3: (5000.0, 0.0, 12000.0, 1),
        4: (-5000.0, 0.0, 12000.0, 1),
    }
    points = {}
    # radius_mm = thickness_um / 1000. trunk r=2mm -> thickness 2000.
    points.update({0: (0, 0, 0, 2000.0), 1: (0, 0, 5000.0, 2000.0), 2: (0, 0, 10000.0, 2000.0)})
    # main daughter pids 10,11,12 (r 1mm)
    points.update({10: (0, 0, 10000.0, 1000.0), 11: (0, 0, 15000.0, 1000.0), 12: (0, 0, 20000.0, 1000.0)})
    # side1 pids 20,21,22 (r 0.3mm)
    points.update({20: (0, 0, 10000.0, 300.0), 21: (2500.0, 0, 11000.0, 300.0), 22: (5000.0, 0, 12000.0, 300.0)})
    # side2 pids 30,31,32 (r 0.15mm)
    points.update({30: (0, 0, 10000.0, 150.0), 31: (-2500.0, 0, 11000.0, 150.0), 32: (-5000.0, 0, 12000.0, 150.0)})
    segments = [
        {"id": 0, "node1": 0, "node2": 1, "point_ids": [0, 1, 2], "strahler": 3},
        {"id": 1, "node1": 1, "node2": 2, "point_ids": [10, 11, 12], "strahler": 2},
        {"id": 2, "node1": 1, "node2": 3, "point_ids": [20, 21, 22], "strahler": 1},
        {"id": 3, "node1": 1, "node2": 4, "point_ids": [30, 31, 32], "strahler": 1},
    ]
    vessel_points = {"MAIN": set([0, 1, 2]) | set([10, 11, 12])}
    return nodes, points, segments, vessel_points


def test_ratio_prune() -> None:
    nodes, points, segments, vessel_points = _ratio_fixture()
    ostia = compute_vessel_ostia(vessel_points_to_idx(vessel_points, segments),
                                 nodes, points, segments)
    assert abs(ostia["MAIN"]["radius_mm"] - 2.0) < 1e-6, ostia

    side1 = {20, 21, 22}
    side2 = {30, 31, 32}
    main = {0, 1, 2, 10, 11, 12}

    # ratio 1/2 -> threshold 1.0 mm: both side branches (0.3, 0.15) removed.
    # Removing both daughters leaves node 1 degree-2, so trunk+main contract into
    # one segment (the shared bifurcation point id is dropped by the merge).
    _n, _p, s_half, pruned_half = prune_by_radius_ratio(
        dict(nodes), {k: tuple(v) for k, v in points.items()},
        segments, vessel_points, ostia, 0.5)
    kept = _all_pids(s_half)
    assert len(pruned_half) == 2 and all("radius_mm" in r for r in pruned_half), pruned_half
    assert len(s_half) == 1, f"trunk+main should contract to 1 segment, got {len(s_half)}"
    assert {0, 1, 11, 12} <= kept, "main vessel endpoints must survive"
    assert len(main & kept) >= len(main) - 1, "at most the shared boundary point dropped"
    assert not (side1 & kept) and not (side2 & kept), "both side branches should be gone"
    print("  [2a] ratio 1/2 prunes 0.3+0.15 side branches, keeps MAIN: PASS")

    # ratio 1/10 -> threshold 0.2 mm: 0.3 kept, 0.15 removed.
    _n, _p, s_tenth, pruned_tenth = prune_by_radius_ratio(
        dict(nodes), {k: tuple(v) for k, v in points.items()},
        segments, vessel_points, ostia, 0.1)
    kept = _all_pids(s_tenth)
    assert len(pruned_tenth) == 1, pruned_tenth   # only the 0.15 branch pruned
    assert main <= kept and side1 <= kept, "0.3 branch should survive at 1/10"
    assert not (side2 & kept), "0.15 branch should be gone at 1/10"
    assert len(s_tenth) == 3, f"expected trunk+main+side1, got {len(s_tenth)}"
    print("  [2b] ratio 1/10 keeps 0.3, prunes 0.15: PASS")


def vessel_points_to_idx(vessel_points, segments):
    """Helper: vessel_points (pid sets) -> vessel_idx (seg index sets) for ostia."""
    out = {}
    for v, vp in vessel_points.items():
        idxs = {i for i, s in enumerate(segments) if set(s["point_ids"]) & vp}
        out[v] = idxs
    return out


# ── 3. Containment prune (length-independent) ──────────────────────────────────

def test_containment_prune() -> None:
    # trunk radius 2 mm along z 0..10 mm, 11 points.
    nodes = {
        0: (0.0, 0.0, 0.0, 1),         # trunk proximal (deg1)
        1: (0.0, 0.0, 10000.0, 3),     # far node (trunk + 2 leaves)
        2: (300.0, 0.0, 7000.0, 1),    # swallowed leaf tip
        3: (3500.0, 0.0, 10000.0, 1),  # genuine leaf tip
    }
    points = {}
    trunk_pids = []
    for k in range(11):
        pid = k
        points[pid] = (0.0, 0.0, k * 1000.0, 2000.0)  # r 2mm (thickness 2000 um)
        trunk_pids.append(pid)
    # swallowed leaf: 3 points, all within 0.3mm of axis, inside trunk (LONGER span 3mm)
    points.update({100: (300.0, 0, 10000.0, 600.0), 101: (300.0, 0, 8500.0, 600.0),
                   102: (300.0, 0, 7000.0, 600.0)})
    # genuine leaf: 2 points poking out to x=3.5mm > trunk r (SHORTER span 0.5mm)
    points.update({200: (3000.0, 0, 10000.0, 600.0), 201: (3500.0, 0, 10000.0, 600.0)})
    segments = [
        {"id": 0, "node1": 0, "node2": 1, "point_ids": trunk_pids, "strahler": 3},
        {"id": 1, "node1": 1, "node2": 2, "point_ids": [100, 101, 102], "strahler": 1},
        {"id": 2, "node1": 1, "node2": 3, "point_ids": [200, 201], "strahler": 1},
    ]

    _n, _p, out = prune_contained_leaves(
        dict(nodes), {k: tuple(v) for k, v in points.items()}, segments,
        vessel_points={})
    kept = _all_pids(out)
    assert 201 in kept, "genuine distal vessel tip must survive (shorter but outside)"
    assert 102 not in kept, "swallowed stub tip must be removed (longer but inside)"
    assert set(trunk_pids) <= kept, "trunk must survive"
    print("  [3] containment prune removes swallowed stub, keeps shorter genuine: PASS")


# ── 4. Contour mesh + connected components (picker building blocks) ────────────

def test_contour_mesh() -> None:
    # straight segment along z, constant radius 1 mm
    n_sides = 16
    coords = np.array([[0.0, 0.0, float(k)] for k in range(5)])
    radii = np.full(5, 1.0)
    poly = _segment_contour_mesh(coords, radii, n_sides=n_sides, ring_stride=1)
    # one ring per centerline point, centerline points as vertex cells
    assert poly.n_lines == 5, f"expected 5 rings, got {poly.n_lines}"
    assert poly.n_verts == 5, f"expected 5 centerline dots, got {poly.n_verts}"
    ring_pts = poly.points[: n_sides * 5]
    # rings perpendicular to z-axis: ring points lie at radius 1 in xy, z = center
    xy = np.linalg.norm(ring_pts[:, :2], axis=1)
    assert np.allclose(xy, 1.0, atol=1e-6), f"ring radius off: {xy.min()}..{xy.max()}"
    assert ring_pts[:, 2].min() >= -1e-9 and ring_pts[:, 2].max() <= 4 + 1e-9
    print("  [4] contour mesh: rings at r=1 perpendicular to tangent + dots: PASS")


def test_connected_components_two_trees() -> None:
    # tree A: segs 0,1 share node 1; tree B: seg 2 (nodes 10-11) disjoint
    segments = [
        {"id": 0, "node1": 0, "node2": 1, "point_ids": [0, 1]},
        {"id": 1, "node1": 1, "node2": 2, "point_ids": [2, 3]},
        {"id": 2, "node1": 10, "node2": 11, "point_ids": [4, 5]},
    ]
    comps = find_connected_components(segments)
    assert len(comps) == 2, f"expected 2 trees, got {len(comps)}"
    assert {0, 1} in comps and {2} in comps, comps
    print("  [5] connected components splits 2 trees by global index: PASS")


# ── 6. Off-screen pick path (clicks resolve to the right segment) ──────────────

def test_offscreen_pick() -> None:
    """Reproduce the picker's click path: pickable contour meshes + a tagged
    KDTree + a vtkCellPicker fired at the screen projection of a known point.
    Proves selection resolves by pick *position* (no dependence on actor.name)."""
    import pyvista as pv
    from pyvista import _vtk
    from coronary_sdf.epicardial_annotation import _segment_contour_mesh

    # Two well-separated segments along z, at x=0 and x=10 mm.
    segs = {
        0: np.array([[0.0, 0.0, float(k)] for k in range(5)]),
        1: np.array([[10.0, 0.0, float(k)] for k in range(5)]),
    }
    pl = pv.Plotter(off_screen=True)
    pl.set_background("white")
    for i, coords in segs.items():
        contour = _segment_contour_mesh(coords, np.ones(len(coords)), n_sides=16)
        contour.field_data["seg_idx"] = np.array([i], dtype=np.int64)
        pl.add_mesh(contour, color=(0.72, 0.72, 0.72), pickable=True,
                    render_points_as_spheres=True, point_size=6.0, line_width=2.0)

    pl.camera_position = "xz"  # look down -y so x/z map to screen
    pl.render()
    ren = pl.renderer

    def pick_segment(world_xyz):
        ren.SetWorldPoint(world_xyz[0], world_xyz[1], world_xyz[2], 1.0)
        ren.WorldToDisplay()
        dx, dy, _dz = ren.GetDisplayPoint()
        picker = _vtk.vtkCellPicker()
        picker.SetTolerance(0.008)
        picker.Pick(dx, dy, 0, ren)
        ds = picker.GetDataSet()
        if ds is None:
            return None
        # exact front-most hit resolved via the tagged mesh (matches the picker)
        fd = pv.wrap(ds).field_data
        return int(fd["seg_idx"][0]) if "seg_idx" in fd else None

    got_a = pick_segment([0.0, 0.0, 2.0])    # midpoint of segment 0
    got_b = pick_segment([10.0, 0.0, 2.0])   # midpoint of segment 1
    pl.close()
    assert got_a == 0, f"click over segment 0 resolved to {got_a}"
    assert got_b == 1, f"click over segment 1 resolved to {got_b}"
    print("  [6] off-screen pick resolves clicks to the correct segment: PASS")


# ── 7. branch_radius_mm skips the inflated ostium contours ─────────────────────

def test_branch_radius_mm() -> None:
    from coronary_sdf.flow_fractions import branch_radius_mm
    # 8-point branch from the ostium (node1) outward: first 2 contours inflated
    # (r=5 mm, where the branch intersects the parent), the rest r=1 mm.
    radii_um = [5000, 5000, 1000, 1000, 1000, 1000, 1000, 1000]
    points = {10 + k: (float(k) * 1000.0, 0.0, 0.0, float(radii_um[k])) for k in range(8)}
    nodes = {1: (0.0, 0.0, 0.0, 3), 2: (7000.0, 0.0, 0.0, 1)}  # distal is a leaf
    seg = {"node1": 1, "node2": 2, "point_ids": list(range(10, 18))}
    r, _used = branch_radius_mm(seg, 1, points, nodes, skip_points=2, n_average=10)
    assert abs(r - 1.0) < 1e-6, f"ostium outliers leaked into the average: r={r}"

    # Short 3-point branch, first contour inflated -> fallback still drops it.
    pts2 = {0: (0.0, 0, 0, 5000.0), 1: (1000.0, 0, 0, 1000.0), 2: (2000.0, 0, 0, 1000.0)}
    seg2 = {"node1": 1, "node2": 2, "point_ids": [0, 1, 2]}
    r2, _ = branch_radius_mm(seg2, 1, pts2, {1: (0, 0, 0, 3), 2: (2000.0, 0, 0, 1)})
    assert abs(r2 - 1.0) < 1e-6, f"short-branch ostium outlier leaked: r2={r2}"
    print("  [7] branch_radius_mm skips inflated ostium contours: PASS")


# ── 8. compute_vessel_radius_stats excludes junction-inflated contours ─────────

def test_vessel_radius_stats() -> None:
    # Vessel = ostial seg (root -> bif) + distal seg (bif -> leaf); a side branch
    # makes the shared node 2 a bifurcation (deg 3). Inflated r=5 contours sit at
    # the bifurcation ends; r=2 proximal, r=1 distal elsewhere.
    points = {}
    points.update({10: (0, 0, 0, 2000.0), 11: (0, 0, 1000.0, 2000.0),
                   12: (0, 0, 2000.0, 2000.0), 13: (0, 0, 3000.0, 5000.0)})  # last=bif
    points.update({20: (0, 0, 3000.0, 5000.0), 21: (0, 0, 4000.0, 1000.0),  # first=bif
                   22: (0, 0, 5000.0, 1000.0), 23: (0, 0, 6000.0, 1000.0)})
    points.update({30: (0, 0, 3000.0, 1000.0), 31: (3000.0, 0, 4000.0, 1000.0)})
    segments = [
        {"id": 0, "node1": 1, "node2": 2, "point_ids": [10, 11, 12, 13], "strahler": 3},
        {"id": 1, "node1": 2, "node2": 3, "point_ids": [20, 21, 22, 23], "strahler": 2},
        {"id": 2, "node1": 2, "node2": 4, "point_ids": [30, 31], "strahler": 1},
    ]
    nodes = {1: (0, 0, 0, 1), 2: (0, 0, 3000.0, 3), 3: (0, 0, 6000.0, 1), 4: (3000.0, 0, 4000.0, 1)}
    rows = compute_vessel_radius_stats({"LAD": {0, 1}}, nodes, points, segments, skip_points=1)
    assert len(rows) == 1, rows
    r = rows[0]
    # inflated r=5 contours at the bifurcation are excluded by skip_points=1
    assert (r["prox_min_mm"], r["prox_mean_mm"], r["prox_max_mm"]) == (2.0, 2.0, 2.0), r
    assert (r["dist_min_mm"], r["dist_mean_mm"], r["dist_max_mm"]) == (1.0, 1.0, 1.0), r
    assert (r["total_min_mm"], r["total_mean_mm"], r["total_max_mm"]) == (1.0, 1.5, 2.0), r
    print("  [8] vessel_radius_stats excludes junction contours (prox/dist/total): PASS")


if __name__ == "__main__":
    test_writer_roundtrip()
    test_ratio_prune()
    test_containment_prune()
    test_contour_mesh()
    test_connected_components_two_trees()
    test_offscreen_pick()
    test_branch_radius_mm()
    test_vessel_radius_stats()
    print("\nAll epicardial_annotation smoke tests passed.")
