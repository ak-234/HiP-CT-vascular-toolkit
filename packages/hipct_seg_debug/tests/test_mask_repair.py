"""Guarded mask repair: it must fix the two defects and refuse the two failure modes.

The failure modes are the reason the guards exist, so each has phantoms built to
trigger it: parallel and crossing vessels that a plain closing welds (risk 1), and a
vessel loop whose interior a plain slice fill turns into lumen (risk 2). Every test
checks the whole volume's Betti numbers, not only the repair's own report.
"""
import numpy as np
import pytest
from scipy import ndimage

from hipct_seg_debug.edit import mask_repair as mr
from .conftest_geometry import SPACING, make_frame
from .conftest_geometry import cylinder as _cylinder

TISSUE, WALL, LUMEN = 200.0, 230.0, 30.0


def cylinder(*args, **kw):
    return _cylinder(*args, **kw).astype(bool)


def betti(vol):
    """``(b0, b1, b2)`` of the 26-connected foreground of the whole volume."""
    f = np.pad(np.asarray(vol) > 0, 1)
    b0, b2, chi = mr._betti(f)
    return b0, b0 + b2 - chi, b2


def raw_for(vol, dark=None):
    """Tissue everywhere, wall-bright on the mask, and dark where `dark` says."""
    raw = np.full(vol.shape, TISSUE)
    raw[vol > 0] = WALL
    if dark is not None:
        raw[dark] = LUMEN
    return raw


def repair(vol, raw, **kw):
    vol = vol.astype(np.uint8)
    kw.setdefault("max_hole_um", 40 * SPACING)
    # The closing is opt-in (CLOSE_RADIUS 0); these phantoms test its guards, so on.
    kw.setdefault("close_radius", 2)
    report = mr.repair_volume(vol, make_frame(vol.shape), sampler=mr.ArraySampler(raw), **kw)
    return vol, report


def ring_tube(shape, outer, inner, x0, x1):
    solid = cylinder(shape, outer, x0, x1)
    core = cylinder(shape, inner, x0, x1)
    return solid & ~core, core


# ------------------------------------------------------------------ the measure


def test_betti_matches_skimage_and_labels_on_random_volumes():
    from skimage.measure import euler_number

    rng = np.random.default_rng(1)
    for p in (0.2, 0.45, 0.7):
        f = rng.random((9, 10, 11)) < p
        b0, b2, chi = mr._betti(f)
        assert chi == euler_number(f, connectivity=3)
        assert b0 == ndimage.label(f, structure=np.ones((3, 3, 3)))[1]
        bg, n = ndimage.label(~f)
        border = set(np.unique(np.concatenate([bg[0].ravel(), bg[-1].ravel(), bg[:, 0].ravel(),
                                               bg[:, -1].ravel(), bg[:, :, 0].ravel(),
                                               bg[:, :, -1].ravel()])))
        assert b2 == len(set(range(1, n + 1)) - border)


# ------------------------------------------------------------------ the two defects


def test_a_hollow_tube_gets_its_dark_core_and_becomes_one_solid_tube():
    shape = (24, 24, 60)
    wall, core = ring_tube(shape, 7, 5, 4, 56)
    assert betti(wall) == (1, 1, 0)
    vol, report = repair(wall, raw_for(wall, dark=core))
    assert (vol[core] > 0).all(), "the unlabelled lumen is filled"
    assert betti(vol) == (1, 0, 0)
    assert report.count("fill", accepted=True) >= 1

    from hipct_seg_debug.edit import jin_mgf
    import networkx as nx

    r = jin_mgf.extract(vol > 0, engine="sparse", allow_loops=True)
    tips = sum(1 for _, d in nx.Graph(r.edges.tolist()).degree if d == 1)
    before = jin_mgf.extract(wall, engine="sparse", allow_loops=True)
    tips_before = sum(1 for _, d in nx.Graph(before.edges.tolist()).degree if d == 1)
    assert tips == 2 and tips_before > tips


def test_an_enclosed_void_is_filled():
    shape = (24, 24, 50)
    tube = cylinder(shape, 7, 4, 46)
    void = np.zeros(shape, dtype=bool)
    void[11:13, 11:13, 20:30] = True
    vol0 = tube & ~void
    assert betti(vol0) == (1, 0, 1)
    vol, report = repair(vol0, raw_for(vol0, dark=void))
    assert (vol[void] > 0).all()
    assert betti(vol) == (1, 0, 0)


# ------------------------------------------------------------------ risk 1: welding


def u_tube(shape, r, gap, x0, x1):
    """Two parallel arms `gap` voxels apart, joined by a bridge at x0: one tree."""
    nz, ny, nx = shape
    cy1 = ny // 2 - (r + 1 + gap // 2)
    cy2 = cy1 + 2 * r + 1 + gap
    a = cylinder(shape, r, x0, x1, cy=cy1)
    b = cylinder(shape, r, x0, x1, cy=cy2)
    bend = np.zeros(shape, dtype=bool)
    bend[nz // 2 - r:nz // 2 + r + 1, cy1:cy2 + 1, x0:x0 + 2 * r + 1] = True
    return a | b | bend, (cy1, cy2)


def test_a_plain_closing_welds_parallel_branches_and_the_guarded_one_does_not():
    shape = (20, 40, 70)
    vol0, (cy1, cy2) = u_tube(shape, 4, 2, 4, 66)
    gap = np.zeros(shape, dtype=bool)  # the closest approach: the arms' mid-plane
    gap[shape[0] // 2 - 1:shape[0] // 2 + 2, cy1 + 5:cy2 - 4, 20:66] = True
    gap &= ~vol0
    plain = ndimage.binary_closing(np.pad(vol0, 3), mr._ball(2))[3:-3, 3:-3, 3:-3]
    assert plain[gap].mean() > 0.3, "the phantom does provoke a plain closing"
    vol, report = repair(vol0, raw_for(vol0))
    assert not (vol[gap] > 0).any(), "nothing between the arms"
    assert betti(vol) == betti(vol0)
    assert report.count("close", accepted=False, reason="detour") >= 1


def test_separate_vessels_passing_close_are_never_joined():
    shape = (36, 30, 50)
    a = cylinder(shape, 4, 2, 48, cy=15, cz=13)  # along x, z 9..17
    b = np.zeros(shape, dtype=bool)  # along y, z 20..28: a 2-voxel gap where they cross
    zz, xx = np.mgrid[:36, :50]
    b[:, 3:27, :] = ((zz - 24) ** 2 + (xx - 25) ** 2 <= 16)[:, None, :]
    vol0 = (a | b).astype(np.uint8)
    assert betti(vol0)[0] == 2
    plain = ndimage.binary_closing(np.pad(vol0 > 0, 3), mr._ball(2))[3:-3, 3:-3, 3:-3]
    assert betti(plain)[0] == 1, "a plain closing joins them"
    vol, report = repair(vol0, raw_for(vol0))
    # Each component is closed on its own, so nothing is even proposed between them;
    # the topology guard below is the second line, tested directly.
    assert betti(vol) == betti(vol0)
    assert not (vol[ndimage.binary_dilation(a, iterations=3) & b] != vol0[
        ndimage.binary_dilation(a, iterations=3) & b]).any()
    assert all(r.db0 == 0 for r in report.regions if r.accepted)


def test_topology_change_flags_a_merge_a_tunnel_and_passes_a_void():
    f = np.zeros((7, 7, 9), dtype=bool)
    f[2:5, 2:5, 0:3] = True
    f[2:5, 2:5, 5:9] = True
    bridge = np.zeros_like(f)
    bridge[3, 3, 3:5] = True
    assert mr.topology_change(f, bridge)[0] == -1, "joins two pieces"

    ring = np.zeros((5, 9, 9), dtype=bool)
    ring[1:4, 1:8, 1:8] = True
    ring[1:4, 3:6, 3:6] = False
    ring[1:4, 4, 1:3] = False  # a cut through one side: a C, not an O
    seal = np.zeros_like(ring)
    seal[1:4, 4, 1:3] = True
    db0, db1, _ = mr.topology_change(ring, seal)
    assert (db0, db1) == (0, 1), "closing the C makes a tunnel"

    solid = np.ones((7, 7, 7), dtype=bool)
    void = np.zeros_like(solid)
    void[3, 3, 3] = True
    solid[3, 3, 3] = False
    assert mr.topology_change(solid, void) == (0, 0, -1)


def test_a_bridge_that_would_close_a_loop_is_refused():
    """Arms bowed toward each other at one spot: the weld would make a tunnel."""
    shape = (20, 40, 70)
    vol0, (cy1, cy2) = u_tube(shape, 4, 6, 4, 66)
    bump = np.zeros(shape, dtype=bool)
    bump[8:13, cy1 + 4:cy1 + 7, 40:45] = True  # arm 1 reaches to within 2 of arm 2
    bump[8:13, cy2 - 6:cy2 - 3, 40:45] = True
    vol0 = vol0 | bump
    plain = ndimage.binary_closing(np.pad(vol0, 3), mr._ball(2))[3:-3, 3:-3, 3:-3]
    assert betti(plain)[1] == betti(vol0)[1] + 1, "a plain closing makes the tunnel"
    vol, report = repair(vol0, raw_for(vol0))
    assert betti(vol) == betti(vol0)
    assert report.count("close", accepted=False, reason="topology") >= 1
    assert all(r.db1 <= 0 and r.db0 == 0 for r in report.regions if r.accepted)


# ------------------------------------------------------------------ risk 2: loop interiors


def flat_loop(shape, r_ring, r_tube):
    """A vessel loop (torus) lying in the z = const plane through the centre."""
    nz, ny, nx = shape
    z, y, x = np.mgrid[:nz, :ny, :nx]
    rho = np.hypot(y - ny // 2, x - nx // 2)
    torus = (rho - r_ring) ** 2 + (z - nz // 2) ** 2 <= r_tube ** 2
    inside = (rho < r_ring - r_tube) & (np.abs(z - nz // 2) <= r_tube)
    return torus, inside


def test_a_loop_interior_of_tissue_is_not_filled():
    torus, inside = flat_loop((16, 40, 40), 12, 3)
    assert betti(torus) == (1, 1, 0)
    vol, report = repair(torus, raw_for(torus))  # interior is bright tissue
    assert not (vol[inside & ~torus] > 0).any()
    assert betti(vol) == (1, 1, 0)
    assert report.count("fill", accepted=False) >= 1
    assert report.count("fill", accepted=True) == 0


def test_a_dark_loop_interior_is_refused_as_a_disc():
    torus, inside = flat_loop((16, 40, 40), 12, 3)
    vol, report = repair(torus, raw_for(torus, dark=inside & ~torus))
    assert not (vol[inside & ~torus] > 0).any()
    assert betti(vol) == (1, 1, 0)
    assert report.count("fill", accepted=False, reason="shape") >= 1


def test_a_loop_wider_than_the_hole_cap_is_not_even_a_candidate():
    torus, inside = flat_loop((16, 40, 40), 12, 3)
    vol, report = repair(torus, raw_for(torus, dark=inside & ~torus),
                         max_hole_um=4 * SPACING)
    assert not (vol[inside & ~torus] > 0).any()
    assert report.count("fill") == 0


def test_a_bifurcation_crotch_patch_does_not_change_topology():
    shape = (24, 40, 60)
    parent = cylinder(shape, 5, 2, 30, cy=20)
    zz, yy, xx = np.mgrid[:24, :40, :60]
    d1 = ((zz - 12) ** 2 + (yy - 20 - (xx - 28) * 0.6) ** 2 <= 16) & (xx >= 28) & (xx < 58)
    d2 = ((zz - 12) ** 2 + (yy - 20 + (xx - 28) * 0.6) ** 2 <= 16) & (xx >= 28) & (xx < 58)
    vol0 = parent | d1 | d2
    vol, report = repair(vol0, raw_for(vol0))
    assert betti(vol) == betti(vol0) == (1, 0, 0)
    assert all(r.db0 == 0 and r.db1 <= 0 for r in report.regions if r.accepted)


def test_no_sampler_needed_and_raw_is_only_recorded():
    shape = (24, 24, 60)
    wall, core = ring_tube(shape, 7, 5, 4, 56)
    vol = wall.astype(np.uint8)
    report = mr.repair_volume(vol, make_frame(shape), sampler=None, max_hole_um=40 * SPACING)
    assert (vol[core] > 0).all()
    assert all(r.region_median is None for r in report.regions)
    # the same with a raw image that would have failed any brightness test
    vol2 = wall.astype(np.uint8)
    raw = raw_for(wall)  # core as bright as the tissue
    mr.repair_volume(vol2, make_frame(shape), sampler=mr.ArraySampler(raw),
                     max_hole_um=40 * SPACING)
    assert np.array_equal(vol, vol2), "brightness decides nothing"


def hairpin(shape, gap, r, x0, x1):
    """Two parallel arms joined at both ends: a closed loop enclosing a long strip."""
    nz, ny, nx = shape
    z, y, x = np.mgrid[:nz, :ny, :nx]
    y1 = ny // 2 - (r + 1 + gap // 2)
    y2 = y1 + 2 * r + 1 + gap

    def arm(yc):
        return ((z - nz // 2) ** 2 + (y - yc) ** 2 <= r * r) & (x >= x0) & (x < x1)

    ends = (np.abs(z - nz // 2) <= r) & (y >= y1) & (y <= y2) & (
        ((x >= x0) & (x < x0 + 2 * r)) | ((x >= x1 - 2 * r) & (x < x1)))
    strip = (np.abs(z - nz // 2) <= r) & (y > y1 + r) & (y < y2 - r) & (x >= x0 + 2 * r) & (
        x < x1 - 2 * r)
    return arm(y1) | arm(y2) | ends, strip


def test_a_hairpin_loops_elongated_interior_is_not_filled():
    """Long for its width like a core, but it lies in its slices' plane."""
    shape = (16, 30, 60)
    loop, strip = hairpin(shape, 4, 3, 4, 56)
    assert betti(loop) == (1, 1, 0)
    plain = mr.slice_holes(loop, np.zeros_like(loop), 10**6) > 0
    assert plain[strip].mean() > 0.3, "a plain slice fill fills the strip"
    vol, report = repair(loop, raw_for(loop, dark=strip))
    assert report.count("fill", accepted=True) == 0
    assert report.count("fill", accepted=False, reason="shape") >= 1
    # the closing may round the four inside corners, as it does a crotch -- nothing more
    middle = strip.copy()
    middle[:, :, :14] = middle[:, :, 46:] = False
    assert not (vol[middle] > 0).any()
    assert (vol[strip] > 0).sum() <= 0.05 * strip.sum()
    assert betti(vol) == (1, 1, 0)


@pytest.mark.parametrize("tilt", [10, 20, 30])
def test_a_tilted_loops_interior_is_not_filled(tilt):
    from scipy.spatial.transform import Rotation

    n, R, r = 32, 8, 4
    g = np.mgrid[:n, :n, :n].astype(float) - n / 2
    p = np.tensordot(Rotation.from_euler("y", tilt, degrees=True).as_matrix().T, g, 1)
    torus = (np.hypot(p[1], p[2]) - R) ** 2 + p[0] ** 2 <= r * r
    vol, report = repair(torus, raw_for(torus))
    assert betti(vol) == betti(torus)
    assert report.count("fill", accepted=True) == 0


def test_an_oblique_hollow_core_is_filled():
    n, ro, ri, L = 40, 6, 4, 30
    g = np.mgrid[:n, :n, :n].astype(float) - n / 2
    axis = np.array([0.3, 0.5, 0.81])
    axis /= np.linalg.norm(axis)
    t = np.tensordot(axis, g, 1)
    perp = np.sqrt(np.maximum((g ** 2).sum(0) - t ** 2, 0))
    seg = np.abs(t) < L / 2
    wall, core = (perp <= ro) & (perp > ri) & seg, (perp <= ri - 0.5) & (np.abs(t) < L / 2 - 4)
    vol, report = repair(wall, raw_for(wall))
    assert (vol[core] > 0).mean() > 0.95
    assert betti(vol)[1] == 0 and betti(wall)[1] == 1


def test_another_label_is_never_claimed_or_joined():
    shape = (24, 24, 60)
    wall, core = ring_tube(shape, 7, 5, 4, 56)
    vol0 = wall.astype(np.uint8)
    other = np.zeros(shape, dtype=bool)
    other[12, 12, 25:35] = True  # a second tree threading the hollow core
    vol0[other] = 2
    vol, report = repair(vol0, raw_for(vol0, dark=core & ~other))
    assert (vol[other] == 2).all()
    assert betti(vol == 2)[0] == 1 and betti(vol == 1)[0] == 1
    assert betti(vol)[0] == betti(vol0)[0], "the repair joins no two labels"


# ------------------------------------------------------------------ the command


MATERIALS = """Materials {
    Exterior {
        Id "0"
    }
    Left_Tree {
        Color 0 0 1,
        Id "9"
    }
}"""


def _write_inputs(tmp_path, volume, raw):
    import tifffile

    from hipct_seg_debug import rle_write

    seg = tmp_path / "seg.am"
    nz, ny, nx = volume.shape
    bbox = np.empty(6)
    bbox[0::2] = 0.0
    bbox[1::2] = (np.array([nx, ny, nz]) - 1) * SPACING
    rle_write.write_lattice(seg, volume, bbox, extra=MATERIALS)
    rawdir = tmp_path / "raw"
    rawdir.mkdir()
    for z in range(nz):
        tifffile.imwrite(rawdir / f"s_{z:04d}.tif", raw[z].astype(np.uint16))
    return seg, rawdir


def test_repair_mask_guarded_needs_raw(tmp_path):
    from hipct_seg_debug.edit import __main__ as cli

    wall, _core = ring_tube((24, 24, 60), 7, 5, 4, 56)
    seg, _ = _write_inputs(tmp_path, wall.astype(np.uint8), raw_for(wall))
    args = cli.build_parser().parse_args(["repair-mask", "--seg", str(seg), "--guarded",
                                          "--voxel-um", str(SPACING)])
    with pytest.raises(SystemExit, match="--raw"):
        cli.cmd_repair_mask(args)
    args = cli.build_parser().parse_args(["repair-mask", "--seg", str(seg), "--guarded",
                                          "--raw", str(tmp_path)])
    with pytest.raises(SystemExit, match="--voxel-um"):
        cli.cmd_repair_mask(args)


def test_repair_mask_guarded_writes_labels_materials_report_and_audit(tmp_path):
    import json

    from hipct_seg_debug import amira, rle
    from hipct_seg_debug.edit import __main__ as cli

    shape = (24, 24, 60)
    wall, core = ring_tube(shape, 7, 5, 4, 56)
    seg, rawdir = _write_inputs(tmp_path, wall.astype(np.uint8), raw_for(wall, dark=core))
    out, report, audit = tmp_path / "repaired.am", tmp_path / "r.json", tmp_path / "audit"
    args = cli.build_parser().parse_args(
        ["repair-mask", "--seg", str(seg), "--guarded", "--raw", str(rawdir),
         "--voxel-um", str(SPACING), "--max-hole-um", str(40 * SPACING), "--out", str(out), "--report", str(report),
         "--audit-dir", str(audit)])
    assert cli.cmd_repair_mask(args) == 0
    info = amira.read_lattice_header(out)
    assert [m.name for m in info.materials] == ["Exterior", "Left_Tree"]
    lattice = rle.open_lattice(out, info.fields["Labels"], info.dims)
    got = np.stack([lattice.slice_z(z) for z in range(shape[0])])
    assert set(np.unique(got)) == {0, 1}, "labels kept, not rescaled to 255"
    assert (got[core] == 1).all() and (got[wall] == 1).all()
    data = json.loads(report.read_text())
    assert data["trees"][0]["added"] > 0
    assert any(r["kind"] == "fill" and r["accepted"] for r in data["regions"])
    assert list(audit.glob("*.png"))


# ------------------------------------------------------------------ the wider sweep


def _frame_grid(n):
    return np.mgrid[:n[0], :n[1], :n[2]].astype(float) - (np.array(n)[:, None, None, None] - 1) / 2


def _torus(R, r, tilt=0):
    from scipy.spatial.transform import Rotation

    n = int(2 * (R + r) + 10)
    g = _frame_grid((n, n, n))
    p = np.tensordot(Rotation.from_euler("y", tilt, degrees=True).as_matrix().T, g, 1)
    return (np.hypot(p[1], p[2]) - R) ** 2 + p[0] ** 2 <= r * r


@pytest.mark.parametrize("R, r, tilt", [(6, 4, 0), (6, 4, 15), (8, 6, 0), (8, 6, 30),
                                        (5, 3, 15), (4, 2, 30)])
@pytest.mark.parametrize("close", [0, 2])
def test_a_tight_loop_of_a_thick_vessel_keeps_its_loop(R, r, tilt, close):
    """Its hole is a short fat pipe -- a core by shape -- ringed by a whole vessel.

    The old gate (elongation + alignment) filled four of these, and a closing allowed
    to remove loops filled them all.
    """
    t = _torus(R, r, tilt)
    assert betti(t) == (1, 1, 0)
    vol, report = repair(t, raw_for(t), close_radius=close)
    assert betti(vol) == (1, 1, 0)


@pytest.mark.parametrize("gap, r", [(2, 3), (2, 4), (1, 3)])
def test_a_closing_does_not_zip_a_narrow_hairpin_shut(gap, r):
    shape = (4 * r + 8, 4 * r + gap + 12, 60)
    loop, strip = hairpin(shape, gap, r, 4, 56)
    plain = ndimage.binary_closing(np.pad(loop, 3), mr._ball(2))[3:-3, 3:-3, 3:-3]
    assert betti(plain)[1] == 0, "a plain closing does zip it"
    vol, report = repair(loop, raw_for(loop), close_radius=2)
    assert betti(vol) == betti(loop) == (1, 1, 0)
    assert report.count("close", accepted=False, reason="topology") >= 1


def test_a_curved_hollow_core_is_filled():
    from scipy.spatial.transform import Rotation

    ro, ri, Rc, angle = 7, 5, 15, 200
    n = int(2 * (Rc + ro) + 10)
    g = _frame_grid((n, n, n))
    p = np.tensordot(Rotation.from_euler("x", 30, degrees=True).as_matrix().T, g, 1)
    phi = np.degrees(np.arctan2(p[1], p[2])) % 360
    d = np.hypot(np.hypot(p[1], p[2]) - Rc, p[0])
    wall = (d <= ro) & (d > ri) & (phi < angle)
    core = (d <= ri - 1) & (phi > 15) & (phi < angle - 15)
    assert betti(wall)[1] == 1
    vol, report = repair(wall, raw_for(wall))
    assert betti(vol) == (1, 0, 0)
    assert (vol[core] > 0).mean() > 0.95


def test_whole_volume_euler_and_betti_match_the_direct_count():
    from skimage.measure import euler_number

    rng = np.random.default_rng(3)
    for _ in range(12):
        f = rng.random(tuple(rng.integers(4, 30, 3))) < rng.uniform(0.1, 0.6)
        for tile in (4, 7, 64):
            assert mr.euler_volume(f, tile=tile) == euler_number(f, connectivity=3)
        assert mr.betti_volume(f, tile=5) == betti(f)


def test_the_default_repair_fills_without_closing():
    shape = (24, 24, 60)
    wall, core = ring_tube(shape, 7, 5, 4, 56)
    vol = wall.astype(np.uint8)
    report = mr.repair_volume(vol, make_frame(shape), max_hole_um=40 * SPACING)
    assert report.count("close") == 0
    assert (vol[core] > 0).all() and betti(vol) == (1, 0, 0)


def _collapsed_in_plane(a=14, b=4, wall=2, L=50):
    """A flattened hollow vessel lying in the slice plane, opened by a gap in one side:
    its slit lumen is a hole only in slices along the vessel, so it lies in their
    plane like a hairpin's strip -- but its own wall caps it above and below."""
    nz, ny, nx = 2 * b + 12, 2 * a + 12, L + 12
    z, y, x = np.mgrid[:nz, :ny, :nx].astype(float)
    cz, cy = nz // 2, ny // 2
    outer = ((y - cy) / a) ** 2 + ((z - cz) / b) ** 2 <= 1
    inner = ((y - cy) / (a - wall)) ** 2 + ((z - cz) / max(b - wall, 0.8)) ** 2 <= 1
    tube = outer & (x >= 6) & (x < 6 + L)
    ends = outer & (((x >= 6) & (x < 9)) | ((x >= 3 + L) & (x < 6 + L)))
    m = tube & (~inner | ends)
    m[:, cy + a - wall - 1:cy + a + 1, 20:24] = False
    lumen = tube & inner & ~ends & (x >= 10) & (x < L)
    return m, lumen


def test_a_collapsed_hollow_vessel_in_the_slice_plane_is_filled_because_it_is_capped():
    wall, lumen = _collapsed_in_plane()
    assert betti(wall)[2] == 0, "the gap opens it: not an enclosed void"
    vol, report = repair(wall, raw_for(wall), close_radius=0)
    assert (vol[lumen] > 0).mean() > 0.9
    assert any(r.capped is not None and r.capped >= mr.CAP_MIN and r.accepted
               for r in report.regions)


def _sleeved_vessel(ro=10, wall=3, L=50, bridges=6):
    """Outer band + core with the wall between unlabelled, joined by a few bridges:
    each bridge is a loop. Open at both ends, so neither enclosed nor capped."""
    nz, ny, nx = 2 * ro + 10, 2 * ro + 10, L + 10
    z, y, x = np.mgrid[:nz, :ny, :nx].astype(float)
    d = np.hypot(z - nz // 2, y - ny // 2)
    run = (x >= 5) & (x < 5 + L)
    band = (d <= ro) & (d > ro - 2) & run
    gap = (d <= ro - 2) & (d > ro - 2 - wall) & run
    core = (d <= ro - 2 - wall) & run
    m = band | core
    for k in range(bridges):
        xb = 8 + k * (L - 6) // bridges
        m |= gap & (np.abs(x - xb) <= 1) & (np.abs(y - ny // 2) <= 1) & (z > nz // 2)
    return m, gap & ~m


def test_an_unlabelled_wall_sleeve_around_its_own_core_is_filled():
    m, gap = _sleeved_vessel()
    assert betti(m)[1] >= 4
    vol, report = repair(m, raw_for(m), close_radius=0)
    assert betti(vol) == (1, 0, 0)
    assert (vol[gap] > 0).mean() > 0.9


def test_a_loop_threaded_by_a_separate_vessel_is_not_filled():
    """Its interior holds mask -- the other vessel -- so it looks like a sleeve; the
    fill would weld the two, which guard 1 refuses."""
    t = _torus(10, 3)
    n = t.shape
    zz, yy, xx = np.mgrid[:n[0], :n[1], :n[2]]
    rod = (np.hypot(yy - n[1] // 2, xx - n[2] // 2) <= 3)
    vol0 = t.astype(np.uint8)
    vol0[rod & ~t] = 1
    assert betti(vol0)[0] == 2, "two separate vessels"
    vol, report = repair(vol0, raw_for(vol0), close_radius=0)
    assert betti(vol) == betti(vol0)
