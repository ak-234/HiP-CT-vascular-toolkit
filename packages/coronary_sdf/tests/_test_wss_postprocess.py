"""Synthetic-geometry tests for the WSS arc-sweep core (file-IO-free).

Run: ``python tests/_test_wss_postprocess.py``

Builds a straight cylinder centreline (along +z) plus a dense wall point cloud
whose WSS varies as ``base + amp*cos(phi - phi0)`` around the circumference, then
checks that:
  * the max sector lands near ``phi0`` and the min sector near ``phi0 + 180``,
  * the ring mean sits inside the [min, max] band,
  * stations within the bifurcation-flagged zone are excluded,
  * the station arc length spans the cylinder.
"""

from __future__ import annotations

import math
import tempfile
from pathlib import Path

import numpy as np
from scipy.spatial import KDTree

from .wss_postprocess import (
    _centrelines_from_geometry,
    arc_window_min_max,
    evaluate_vessel,
    read_wss_csv,
    recenter_station,
    resolve_coord_scale,
    sample_anchor_wss,
    sample_stations,
)
from .wss_contour_compare import (
    assign_and_sample,
    build_segment_table,
    filter_vessels,
    recenter_on_cloud,
    vessels_without_data,
)


def _build_cylinder(length=20.0, radius=1.5, phi0_deg=90.0, amp=1.0, base=2.0):
    # Centreline along +z (coarser than the wall cloud).
    zc = np.arange(0.0, length + 1e-9, 0.5)
    coords = np.column_stack([np.zeros_like(zc), np.zeros_like(zc), zc])
    radii = np.full(len(zc), radius)
    bif = np.zeros(len(zc), dtype=bool)
    bif[zc < 2.0] = True                       # flag a proximal bifurcation zone

    # Wall cloud: dense rings along z, WSS depends only on circumferential angle.
    zw = np.arange(0.0, length + 1e-9, 0.25)
    phis = np.deg2rad(np.arange(0.0, 360.0, 5.0))
    phi0 = math.radians(phi0_deg)
    pts, wss = [], []
    for z in zw:
        for ph in phis:
            pts.append([radius * math.cos(ph), radius * math.sin(ph), z])
            wss.append(base + amp * math.cos(ph - phi0))
    return coords, radii, bif, np.asarray(pts), np.asarray(wss)


def _ang_close(a, b, tol=12.0):
    d = abs((a - b + 180.0) % 360.0 - 180.0)
    return d <= tol


def test_arc_sweep_finds_max_min_sectors():
    phi0 = 90.0
    coords, radii, bif, pts, wss = _build_cylinder(phi0_deg=phi0)
    rows = evaluate_vessel(coords, radii, bif, pts, wss,
                           interval_mm=1.0, arc_deg=90.0, rot_step_deg=5.0)
    valid = [r for r in rows if not math.isnan(r["wss_max_window"])]
    assert valid, "no valid stations produced"

    for r in valid:
        assert _ang_close(r["angle_max_deg"], phi0), \
            f"max sector {r['angle_max_deg']} not near phi0={phi0}"
        assert _ang_close(r["angle_min_deg"], phi0 + 180.0), \
            f"min sector {r['angle_min_deg']} not near {phi0 + 180}"
        assert r["wss_min_window"] <= r["ring_mean_wss"] + 1e-6 <= r["wss_max_window"] + 1e-3, \
            "ring mean outside [min, max] band"
        assert r["wss_max_window"] > r["wss_min_window"], "max not above min"
    print(f"[PASS] max/min sectors correct across {len(valid)} stations")


def test_bifurcation_zone_excluded():
    coords, radii, bif, pts, wss = _build_cylinder()
    stations = sample_stations(coords, radii, bif, interval_mm=1.0)
    assert stations, "no stations"
    # All bif-flagged centreline points are at z < 2.0 -> no station should start there.
    s_min = min(st["s"] for st in stations)
    assert s_min >= 1.5, f"station leaked into bifurcation zone (s_min={s_min})"
    s_max = max(st["s"] for st in stations)
    assert s_max >= 18.0, f"stations do not span the cylinder (s_max={s_max})"
    print(f"[PASS] bif zone excluded (s in [{s_min:.1f}, {s_max:.1f}] mm)")


def test_min_pts_returns_nan():
    coords, radii, bif, pts, wss = _build_cylinder()
    # Sparse cloud far from the centreline -> slabs empty -> NaN, no crash.
    far = pts + np.array([1000.0, 0.0, 0.0])
    rows = evaluate_vessel(coords, radii, bif, far, wss, interval_mm=2.0, min_pts=8)
    assert all(math.isnan(r["wss_max_window"]) for r in rows), \
        "expected NaN WSS when no wall points are near the centreline"
    print(f"[PASS] empty-slab stations return NaN ({len(rows)} stations)")


def test_resolve_coord_scale():
    # Centreline ~mm extent; wall cloud ~m extent (1000x smaller).
    cl = np.array([[0.0, 0.0, 0.0], [50.0, 0.0, 0.0]])
    wall_m = cl / 1000.0
    wall_mm = cl.copy()

    # Header unit drives the scale.
    s, _ = resolve_coord_scale("m", wall_m, cl)
    assert s == 1000.0, f"metres header should give x1000, got {s}"
    s, _ = resolve_coord_scale("mm", wall_mm, cl)
    assert s == 1.0, f"mm header should give x1, got {s}"
    s, _ = resolve_coord_scale("cm", cl / 10.0, cl)
    assert s == 10.0, f"cm header should give x10, got {s}"

    # Explicit user scale overrides the header.
    s, _ = resolve_coord_scale("m", wall_m, cl, user_scale=1.0)
    assert s == 1.0, f"explicit --coord-scale should win, got {s}"

    # No unit -> infer from the extent ratio.
    s, _ = resolve_coord_scale(None, wall_m, cl)
    assert s == 1000.0, f"~1000x extent ratio should infer metres, got {s}"
    s, _ = resolve_coord_scale(None, wall_mm, cl)
    assert s == 1.0, f"matched extents should assume mm, got {s}"
    print("[PASS] resolve_coord_scale picks header unit / override / extent ratio")


def test_read_wss_csv_metres_with_duplicate_cols():
    # Mimic the user's CFD-Post export: [Name]/[Data] preamble, X/Y/Z in metres,
    # a Wall Shear column, then a duplicated trailing X/Y/Z triple.
    csv_text = (
        "[Name]\n"
        "Default Domain Default\n"
        "\n"
        "[Data]\n"
        "X [ m ], Y [ m ], Z [ m ], Wall Shear [ Pa ], X [ m ], Y [ m ], Z [ m ]\n"
        "7.6945453e-02, 5.986025e-02, 8.56789872e-02, 3.790051, "
        "7.6945453e-02, 5.986025e-02, 8.56789872e-02\n"
        "5.5952783e-02, 4.420738e-02, 9.18809772e-02, 3.474968, "
        "5.5952783e-02, 4.420738e-02, 9.18809772e-02\n"
    )
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "wall.csv"
        p.write_text(csv_text)
        pts, wss, unit = read_wss_csv(p)

    assert unit == "m", f"expected coord unit 'm', got {unit!r}"
    assert pts.shape == (2, 3), f"expected (2,3) coords, got {pts.shape}"
    # Coords come from the first X/Y/Z triple (not the duplicate), in metres.
    assert np.allclose(pts[0], [7.6945453e-02, 5.986025e-02, 8.56789872e-02])
    assert np.allclose(wss, [3.790051, 3.474968]), f"WSS mismatch: {wss}"
    print(f"[PASS] read_wss_csv parses metres + duplicate XYZ ({len(pts)} nodes, [{unit}])")


def _build_two_segment_vessel(add_distal_branch: bool = False):
    """Synthetic LAD made of two segments stitched at a degree-2 node, running
    straight along +z from 0 to 18 mm. Geometry is in micrometres (as parse_xml
    returns it); prepare_segment_spline converts to mm. Optionally add a third
    segment off the distal node so that node becomes a degree-3 bifurcation.

    Returns ``(vessel_points, nodes, points, segments)`` ready for
    ``_centrelines_from_geometry``."""
    points: dict[int, tuple] = {}
    for i in range(19):                      # ids 0..18, z = 0..18 mm
        points[i] = (0.0, 0.0, i * 1000.0, 2000.0)   # thickness 2000 um
    nodes: dict[int, tuple] = {n: (0.0, 0.0, 0.0, 0) for n in (0, 1, 2)}
    seg_a = {"id": 0, "node1": 0, "node2": 1,
             "point_ids": list(range(0, 10)), "strahler": 2}     # z 0..9
    seg_b = {"id": 1, "node1": 1, "node2": 2,
             "point_ids": list(range(9, 19)), "strahler": 1}     # z 9..18
    segments = [seg_a, seg_b]
    if add_distal_branch:
        # Two side branches off the distal node -> node 2 has 3 incident
        # segments (B + C + D) -> degree 3 -> a real bifurcation there.
        nid = 100
        for sid, free_node in ((2, 3), (3, 4)):
            pids = list(range(nid, nid + 10))
            for j, i in enumerate(pids):
                points[i] = (1000.0 * (sid - 1), 0.0, (18 + j) * 1000.0, 2000.0)
            nodes[free_node] = (0.0, 0.0, 0.0, 0)
            segments.append({"id": sid, "node1": 2, "node2": free_node,
                             "point_ids": pids, "strahler": 1})
            nid += 10
    # LAD owns only the two main segments' points (not the side branch's).
    vessel_points = {"LAD": set(range(0, 19))}
    return vessel_points, nodes, points, segments


def test_centrelines_from_geometry_full_tree():
    vessel_points, nodes, points, segments = _build_two_segment_vessel()
    out = _centrelines_from_geometry(vessel_points, nodes, points, segments,
                                     bif_skip=3)
    assert set(out) == {"LAD"}, f"expected one vessel 'LAD', got {set(out)}"
    d = out["LAD"]
    coords, radii, bif = d["coords"], d["radii"], d["bif"]
    assert coords.shape[1] == 3 and len(coords) >= 2, f"bad coords {coords.shape}"
    assert len(radii) == len(coords) == len(bif), "ragged coords/radii/bif"
    # Two segments stitched into one polyline spanning ~0..18 mm along +z.
    assert coords[:, 2].min() < 1.0 and coords[:, 2].max() > 17.0, \
        f"polyline does not span the vessel: z in [{coords[:,2].min():.2f}, {coords[:,2].max():.2f}]"
    assert np.all(radii > 0), "radii must be positive (mm)"
    assert not bif.any(), "no degree>=3 ends -> no bifurcation zone expected"
    print(f"[PASS] full-tree centreline stitched on base geometry "
          f"({len(coords)} pts, z span {coords[:,2].max() - coords[:,2].min():.1f} mm)")


def test_centrelines_from_geometry_flags_bifurcation_end():
    # Distal node now has a third segment -> degree 3 -> bif zone at that end.
    vessel_points, nodes, points, segments = _build_two_segment_vessel(
        add_distal_branch=True)
    out = _centrelines_from_geometry(vessel_points, nodes, points, segments,
                                     bif_skip=3)
    bif = out["LAD"]["bif"]
    assert bif.any(), "expected a bifurcation zone at the degree-3 distal end"
    assert bif[-1] and not bif[0], \
        "bif zone should sit at the distal (degree-3) end, not the proximal end"
    print(f"[PASS] degree-3 distal end flags a bif zone ({int(bif.sum())} contours)")


def test_sample_anchor_wss_basic():
    # Sphere of radius 1.0 (local_radius=1, radius_factor=1) around the origin.
    pts = np.array([[0.5, 0.0, 0.0],    # dist 0.5 -> in
                    [0.0, 0.9, 0.0],    # dist 0.9 -> in
                    [2.0, 0.0, 0.0]])   # dist 2.0 -> out
    wss = np.array([10.0, 20.0, 100.0])
    kd = KDTree(pts)
    res = sample_anchor_wss(np.zeros(3), 1.0, kd, wss, radius_factor=1.0, min_pts=1)
    assert res["n_wall_pts"] == 2, f"expected 2 in-sphere pts, got {res['n_wall_pts']}"
    assert abs(res["ring_mean_wss"] - 15.0) < 1e-9, \
        f"ring-mean should be (10+20)/2=15, got {res['ring_mean_wss']}"
    # Empty sphere -> NaN; and min_pts not met -> NaN.
    far = sample_anchor_wss(np.array([100.0, 0, 0]), 1.0, kd, wss, min_pts=1)
    assert far["n_wall_pts"] == 0 and math.isnan(far["ring_mean_wss"])
    sparse = sample_anchor_wss(np.zeros(3), 1.0, kd, wss, min_pts=3)
    assert math.isnan(sparse["ring_mean_wss"]), "min_pts=3 with 2 pts should be NaN"
    print("[PASS] sample_anchor_wss sphere ring-mean + NaN guards")


def test_sample_anchor_wss_ring_mean_on_cylinder():
    # WSS = base + amp*cos(phi - phi0); a full circumferential ring averages to base.
    coords, radii, bif, pts, wss = _build_cylinder(radius=1.5, base=2.0, amp=1.0)
    kd = KDTree(pts)
    res = sample_anchor_wss(np.array([0.0, 0.0, 10.0]), 1.5, kd, wss,
                            radius_factor=1.5, min_pts=8)
    assert res["n_wall_pts"] > 0, "expected wall points within the sphere"
    assert abs(res["ring_mean_wss"] - 2.0) < 0.1, \
        f"full-ring mean should be ~base=2.0, got {res['ring_mean_wss']}"
    print(f"[PASS] anchor ring-mean ~ base on cylinder "
          f"({res['n_wall_pts']} pts, mean {res['ring_mean_wss']:.3f})")


def test_anchor_grid_count_stable():
    # Straight 10 mm centreline; all-False bif -> nothing dropped.
    zc = np.arange(0.0, 10.0 + 1e-9, 0.5)
    coords = np.column_stack([np.zeros_like(zc), np.zeros_like(zc), zc])
    radii = np.ones_like(zc)
    no_bif = np.zeros(len(coords), dtype=bool)
    sts = sample_stations(coords, radii, no_bif, interval_mm=1.0)
    assert len(sts) == 11, f"expected floor(10/1)+1=11 anchors, got {len(sts)}"
    s = [st["s"] for st in sts]
    assert all(b > a for a, b in zip(s, s[1:])), "anchor arc length must increase"
    print(f"[PASS] fixed anchor grid is stable ({len(sts)} anchors, no bif drop)")


def test_arc_window_min_max():
    phi0 = 90.0
    phi = np.arange(0.0, 360.0, 5.0)
    w = 2.0 + 1.0 * np.cos(np.deg2rad(phi - phi0))      # peak at phi0
    res = arc_window_min_max(phi, w, arc_deg=90.0, rot_step_deg=5.0)
    assert _ang_close(res["angle_max_deg"], phi0), \
        f"max window {res['angle_max_deg']} not near phi0={phi0}"
    assert _ang_close(res["angle_min_deg"], phi0 + 180.0), \
        f"min window {res['angle_min_deg']} not near {phi0 + 180}"
    assert res["wss_max_window"] > res["wss_min_window"], "max not above min"
    assert abs(res["ring_mean_wss"] - 2.0) < 1e-9, "ring mean should equal base"
    empty = arc_window_min_max(np.array([]), np.array([]))
    assert math.isnan(empty["wss_max_window"]) and math.isnan(empty["ring_mean_wss"])
    print("[PASS] arc_window_min_max locates max/min sectors + NaN on empty")


def test_segment_assignment_no_duplicates():
    # Straight 9 mm vessel along +z, radius 1.5 mm.
    zc = np.arange(0.0, 9.0 + 1e-9, 0.5)
    coords = np.column_stack([np.zeros_like(zc), np.zeros_like(zc), zc])
    centrelines = {"LAD": {"coords": coords, "radii": np.full(len(zc), 1.5),
                           "bif": np.zeros(len(zc), bool)}}
    seg_table = build_segment_table(centrelines, segment_mm=3.0)
    assert len(seg_table["seg_keys"]) == 4, \
        f"expected floor(9/3)+1=4 segments, got {len(seg_table['seg_keys'])}"

    # Dense wall ring cloud (all within radius_factor*r), WSS varies with angle,
    # plus one far outlier that must be gated out.
    pts, w = [], []
    for z in np.arange(0.0, 9.0 + 1e-9, 0.25):
        for ph in np.deg2rad(np.arange(0.0, 360.0, 20.0)):
            pts.append([1.5 * np.cos(ph), 1.5 * np.sin(ph), z])
            w.append(3.0 + np.cos(ph))
    pts.append([10.0, 0.0, 4.0]); w.append(999.0)        # far outlier -> dropped
    wall_pts, wall_wss = np.asarray(pts), np.asarray(w)

    rows, keep, idx, _centers = assign_and_sample(
        seg_table, "r1", wall_pts, wall_wss, radius_factor=1.5,
        arc_deg=90.0, rot_step_deg=5.0, min_pts=4, recenter=False,
        want_assignment=True)
    assert len(rows) == 4, f"expected 4 segment rows, got {len(rows)}"
    assert not keep[-1], "far outlier should fail the radius gate"
    # No duplicates: every kept point lands in exactly one segment.
    assert sum(r["n_wall_pts"] for r in rows) == int(keep.sum()), \
        "kept points double-counted across segments"
    populated = [r for r in rows if not math.isnan(r["wss_max_window"])]
    assert populated, "no segment produced WSS values"
    for r in populated:
        assert r["wss_max_window"] >= r["wss_min_window"]

    # Equal count for a second cloud (different WSS level) -> paired.
    rows2, _, _, _ = assign_and_sample(
        seg_table, "r2", wall_pts, wall_wss * 2.0, radius_factor=1.5,
        arc_deg=90.0, rot_step_deg=5.0, min_pts=4, recenter=False)
    assert len(rows2) == len(rows), "ratios must yield equal segment counts"

    # Recenter path: same segment count, outlier still gated out.
    rows3, keep3, _, _ = assign_and_sample(
        seg_table, "r3", wall_pts, wall_wss, radius_factor=1.5,
        arc_deg=90.0, rot_step_deg=5.0, min_pts=4, recenter=True,
        want_assignment=True)
    assert len(rows3) == len(rows), "recenter must keep equal segment counts"
    assert not keep3[-1], "far outlier should stay gated with recenter on"
    print(f"[PASS] 3mm segment assignment: {len(rows)} segments, no duplicates, "
          f"outlier gated ({int(keep.sum())} kept)")


def test_parallel_transport_frame():
    # Quarter-circle centreline in the xy-plane: a strong curvature test.
    theta = np.linspace(0.0, np.pi / 2.0, 41)
    R = 10.0
    coords = np.column_stack([R * np.cos(theta), R * np.sin(theta), np.zeros_like(theta)])
    centrelines = {"LAD": {"coords": coords, "radii": np.full(len(theta), 1.5),
                           "bif": np.zeros(len(theta), bool)}}
    st = build_segment_table(centrelines, segment_mm=3.0)
    T, N, B = st["cl_tan"], st["cl_nhat"], st["cl_bhat"]

    # Orthonormal frame at every point.
    assert np.allclose(np.linalg.norm(T, axis=1), 1.0, atol=1e-9)
    assert np.allclose(np.linalg.norm(N, axis=1), 1.0, atol=1e-9)
    assert np.allclose(np.linalg.norm(B, axis=1), 1.0, atol=1e-9)
    assert np.allclose(np.sum(T * N, axis=1), 0.0, atol=1e-9)
    assert np.allclose(np.sum(T * B, axis=1), 0.0, atol=1e-9)
    assert np.allclose(np.sum(N * B, axis=1), 0.0, atol=1e-9)

    # Parallel transport: the normal must not flip between adjacent points.
    dots = np.sum(N[:-1] * N[1:], axis=1)
    assert np.all(dots > 0), f"normal flips along the curve (min dot {dots.min():.3f})"

    # Tangents follow the local centreline direction: d/dtheta(cos,sin)=(-sin,cos).
    analytic = np.column_stack([-np.sin(theta), np.cos(theta), np.zeros_like(theta)])
    align = np.sum(T[1:-1] * analytic[1:-1], axis=1)        # interior points
    assert np.all(align > 0.99), f"tangent off the curve (min align {align.min():.3f})"
    print(f"[PASS] parallel-transported frame is orthonormal + flip-free "
          f"({len(coords)} pts, min adj-normal dot {dots.min():.3f})")


def test_recenter_on_cloud():
    # Reference centreline drifted +0.3 mm in x off the true lumen axis (x=0).
    zc = np.arange(0.0, 10.0 + 1e-9, 0.5)
    cl_pts = np.column_stack([np.full_like(zc, 0.3), np.zeros_like(zc), zc])
    cl_tan = np.tile([0.0, 0.0, 1.0], (len(zc), 1))         # along +z
    cl_radius = np.full(len(zc), 1.5)

    # On-axis wall ring cloud (centred at x=y=0).
    pts = []
    for z in np.arange(0.0, 10.0 + 1e-9, 0.25):
        for ph in np.deg2rad(np.arange(0.0, 360.0, 20.0)):
            pts.append([1.5 * np.cos(ph), 1.5 * np.sin(ph), z])
    wall_pts = np.asarray(pts)

    new = recenter_on_cloud(cl_pts, cl_tan, cl_radius, wall_pts,
                            radius_factor=1.5, min_pts=4)
    # Interior contours snap back onto the true axis (x ~ 0), z preserved.
    interior = (zc > 0.5) & (zc < 9.5)
    assert np.all(np.abs(new[interior, 0]) < 0.05), \
        f"recentre did not reach the lumen axis (max |x| {np.abs(new[interior,0]).max():.3f})"
    assert np.allclose(new[:, 2], cl_pts[:, 2]), "axial position must be preserved"

    # A point with no nearby cloud keeps its (drifted) position.
    far_cl = np.array([[0.3, 0.0, 100.0]])
    kept = recenter_on_cloud(far_cl, np.array([[0.0, 0.0, 1.0]]),
                             np.array([1.5]), wall_pts, radius_factor=1.5, min_pts=4)
    assert np.allclose(kept[0], far_cl[0]), "isolated contour must keep its position"
    print(f"[PASS] recenter_on_cloud snaps drift to axis "
          f"(|x| {np.abs(new[interior,0]).max():.3f} mm, z preserved)")


def test_recenter_station():
    from scipy.spatial import KDTree
    # Station drifted +0.3 mm in x off the true lumen axis (x=0), tangent +z.
    pos = np.array([0.3, 0.0, 5.0])
    tangent = np.array([0.0, 0.0, 1.0])
    # On-axis wall ring cloud filling the station's slab (z in 4.5..5.5).
    pts = []
    for z in np.arange(4.5, 5.5 + 1e-9, 0.1):
        for ph in np.deg2rad(np.arange(0.0, 360.0, 20.0)):
            pts.append([1.5 * np.cos(ph), 1.5 * np.sin(ph), z])
    kd = KDTree(np.asarray(pts))
    new = recenter_station(pos, tangent, kd, np.asarray(pts), slab_half=0.5,
                           local_radius=1.5, radius_factor=1.5, min_pts=8)
    assert abs(new[0]) < 0.05, f"station not snapped to axis (x={new[0]:.3f})"
    assert abs(new[2] - 5.0) < 1e-6, "axial position must be preserved"
    # No nearby cloud -> unchanged.
    far = recenter_station(np.array([0.0, 0.0, 100.0]), tangent, kd, np.asarray(pts),
                           slab_half=0.5, local_radius=1.5, min_pts=8)
    assert np.allclose(far, [0.0, 0.0, 100.0]), "isolated station must keep its position"
    print(f"[PASS] recenter_station snaps drift to axis (x {new[0]:.3f} mm, z preserved)")


def test_segment_table_handles_arclength_gap():
    # Discontinuous vessel: a cluster near z=0..2 mm, then a big jump to z=20..22 mm
    # (e.g. LAD + LCx best-effort stitched). Intermediate 3 mm bins are empty.
    za = np.arange(0.0, 2.0 + 1e-9, 0.5)
    zb = np.arange(20.0, 22.0 + 1e-9, 0.5)
    zc = np.concatenate([za, zb])
    coords = np.column_stack([np.zeros_like(zc), np.zeros_like(zc), zc])
    cl = {"LAD": {"coords": coords, "radii": np.full(len(zc), 1.5),
                  "bif": np.zeros(len(zc), bool)}}
    st = build_segment_table(cl, segment_mm=3.0)            # must NOT raise
    seg_idx = sorted(k for _v, k in st["seg_keys"])
    # Populated bins only: s up to ~2 -> bin 0; jump adds ~18 -> bins ~6,7.
    assert seg_idx == [0, 6, 7], f"expected populated bins [0,6,7], got {seg_idx}"
    assert all((s0 <= s1) for s0, s1 in st["seg_span"].values())

    # assign_and_sample still yields one row per produced segment.
    pts, w = [], []
    for z in np.concatenate([np.arange(0, 2.01, 0.25), np.arange(20, 22.01, 0.25)]):
        for ph in np.deg2rad(np.arange(0, 360, 20.0)):
            pts.append([1.5 * np.cos(ph), 1.5 * np.sin(ph), z]); w.append(3.0)
    rows, _, _, _ = assign_and_sample(
        st, "r", np.asarray(pts), np.asarray(w), radius_factor=1.5,
        arc_deg=90.0, rot_step_deg=5.0, min_pts=4, recenter=False)
    assert len(rows) == len(st["seg_keys"]) == 3
    print(f"[PASS] arc-length gap handled (populated bins {seg_idx}, {len(rows)} rows)")


def test_filter_vessels():
    cl = {"LAD": {"coords": 1}, "LCx": {"coords": 2}, "RCA": {"coords": 3}}
    # Case-insensitive keep of a subset.
    filt, missing = filter_vessels(cl, ["lad", "LCX"])
    assert set(filt) == {"LAD", "LCx"} and missing == []
    # Missing names reported; present ones kept.
    filt, missing = filter_vessels(cl, ["RCA", "OM1"])
    assert set(filt) == {"RCA"} and missing == ["om1"]
    # No names -> unchanged.
    filt, missing = filter_vessels(cl, None)
    assert filt is cl and missing == []
    print("[PASS] filter_vessels keeps subset (case-insensitive) + reports missing")


def test_vessels_without_data():
    nan = float("nan")
    rows = [
        {"vessel": "LAD", "wss_max_window": 3.2},
        {"vessel": "LAD", "wss_max_window": nan},      # LAD has some data
        {"vessel": "RCA", "wss_max_window": nan},
        {"vessel": "RCA", "wss_max_window": nan},      # RCA all NaN (wrong side)
    ]
    assert vessels_without_data(rows) == ["RCA"], "RCA should be flagged, LAD not"
    print("[PASS] vessels_without_data flags the all-NaN (wrong-side) vessel")


def test_segment_bif_label_and_proximal_trim():
    zc = np.arange(0.0, 12.0 + 1e-9, 0.5)              # 25 points, z = 0..12 mm
    coords = np.column_stack([np.zeros_like(zc), np.zeros_like(zc), zc])
    bif = np.zeros(len(zc), bool)
    bif[0:4] = True        # proximal LM zone (z 0..1.5)
    bif[14:18] = True      # a mid-vessel side branch (z 7..8.5)
    cl = {"LAD": {"coords": coords, "radii": np.full(len(zc), 1.5), "bif": bif}}

    # Trim on: start past the leading bif; seg 0 is bif-free; a mid segment flagged.
    st = build_segment_table(cl, segment_mm=3.0, trim_proximal_bif=True)
    assert abs(st["cl_pts"][:, 2].min() - 2.0) < 1e-9, "proximal bif not trimmed"
    assert not st["seg_bif"][("LAD", 0)], "first segment should be bif-free after trim"
    assert any(st["seg_bif"].values()), "the mid-vessel bifurcation should flag a segment"

    # Trim off: proximal segment retained and flagged has_bif.
    st2 = build_segment_table(cl, segment_mm=3.0, trim_proximal_bif=False)
    assert abs(st2["cl_pts"][:, 2].min()) < 1e-9, "no trim expected"
    assert st2["seg_bif"][("LAD", 0)], "proximal segment should be has_bif"

    # has_bif propagates to the sampled rows.
    pts, w = [], []
    for z in np.arange(2.0, 12.0 + 1e-9, 0.25):
        for ph in np.deg2rad(np.arange(0, 360, 20.0)):
            pts.append([1.5 * np.cos(ph), 1.5 * np.sin(ph), z]); w.append(3.0)
    rows, _, _, _ = assign_and_sample(
        st, "r", np.asarray(pts), np.asarray(w), radius_factor=1.5, arc_deg=90.0,
        rot_step_deg=5.0, min_pts=4, recenter=False)
    for r in rows:
        assert r["has_bif"] == st["seg_bif"][(r["vessel"], r["seg_idx"])]
    print(f"[PASS] bif label + proximal trim (trim min-z {st['cl_pts'][:,2].min():.1f} mm, "
          f"{sum(st['seg_bif'].values())} bif segment(s))")


def test_sidebranch_points_excluded():
    # Main vessel along +z (r=1.5); a side branch leaves at z=5 going +x.
    zc = np.arange(0.0, 10.0 + 1e-9, 0.5)
    coords = np.column_stack([np.zeros_like(zc), np.zeros_like(zc), zc])
    cl = {"LAD": {"coords": coords, "radii": np.full(len(zc), 1.5),
                  "bif": np.zeros(len(zc), bool)}}
    branch = np.array([[x, 0.0, 5.0] for x in (2.0, 2.5, 3.0, 3.5, 4.0)])

    # Wall cloud: main lumen ring + a branch-lumen ring near the ostium that sits
    # within radius_factor*r of the main centreline (the contamination).
    pts = []
    for z in np.arange(0.0, 10.0 + 1e-9, 0.25):
        for th in np.deg2rad(np.arange(0, 360, 20.0)):
            pts.append([1.5 * np.cos(th), 1.5 * np.sin(th), z])
    for ph in np.deg2rad(np.arange(0, 360, 20.0)):
        pts.append([2.0, 0.6 * np.cos(ph), 5.0 + 0.6 * np.sin(ph)])   # branch ring
    wall = np.asarray(pts)
    wss = np.full(len(wall), 3.0)

    # With competitors -> branch points excluded.
    st = build_segment_table(cl, 3.0, trim_proximal_bif=False, competitor_pts=branch)
    _, keep, _, _ = assign_and_sample(st, "r", wall, wss, radius_factor=1.5,
                                      arc_deg=90.0, rot_step_deg=5.0, min_pts=4,
                                      recenter=False, want_assignment=True)
    assert wall[keep][:, 0].max() < 1.7, "side-branch points leaked into the vessel"

    # Without competitors -> the same branch points are captured (the bug).
    st0 = build_segment_table(cl, 3.0, trim_proximal_bif=False, competitor_pts=None)
    _, keep0, _, _ = assign_and_sample(st0, "r", wall, wss, radius_factor=1.5,
                                       arc_deg=90.0, rot_step_deg=5.0, min_pts=4,
                                       recenter=False, want_assignment=True)
    assert wall[keep0][:, 0].max() > 1.9, "expected branch capture without competitors"
    print(f"[PASS] side-branch exclusion (kept max-x {wall[keep][:,0].max():.2f} mm vs "
          f"{wall[keep0][:,0].max():.2f} without competitors)")


if __name__ == "__main__":
    test_arc_sweep_finds_max_min_sectors()
    test_bifurcation_zone_excluded()
    test_min_pts_returns_nan()
    test_resolve_coord_scale()
    test_read_wss_csv_metres_with_duplicate_cols()
    test_centrelines_from_geometry_full_tree()
    test_centrelines_from_geometry_flags_bifurcation_end()
    test_sample_anchor_wss_basic()
    test_sample_anchor_wss_ring_mean_on_cylinder()
    test_anchor_grid_count_stable()
    test_arc_window_min_max()
    test_segment_assignment_no_duplicates()
    test_parallel_transport_frame()
    test_recenter_on_cloud()
    test_recenter_station()
    test_segment_table_handles_arclength_gap()
    test_filter_vessels()
    test_vessels_without_data()
    test_segment_bif_label_and_proximal_trim()
    test_sidebranch_points_excluded()
    print("\nAll WSS post-processing tests passed.")
