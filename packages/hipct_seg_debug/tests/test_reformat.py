"""The reformat: does it cut where it says it cuts, and do the cuts stay apart.

Three claims are pinned here, and they fail in three different ways.

**The coordinate chain.** Trilinear interpolation is *exact* for a field linear in
each axis, so ``test_a_multilinear_field_is_reproduced_exactly`` recovers the three
coefficients and compares them to the ones the phantom was built with. Any transposed
axis, any missed reversal in ``WorldFrame.um_to_raw``, any block-local offset error
permutes or shifts them and the test fails loudly. Nothing else in this file matters
if that one is broken.

**The frame.** A plane at angle ``phi`` to the true normal cuts a circular tube in an
ellipse of axis ratio ``1 / cos phi``, so a tilted cylinder whose sections come out
round is a proof that the planes really are perpendicular. And a frame that twists
shows up on a helix as a section that rotates while its shape stays put, which is why
the RMF is compared against the arbitrary-seed axes it replaced.

**The collision bound.** ``h < R`` is not a rule of thumb, it is where the normals of
a circular arc meet. So it is tested against the geometry directly -- squares that do
and do not cross -- rather than against itself.
"""

from __future__ import annotations

import numpy as np
import pytest

from hipct_seg_debug import reformat as rf
from hipct_seg_debug.frame import WorldFrame

from .conftest_geometry import graph_from


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #


class FakeStack:
    """The ``TiffStack`` surface the sampler uses, over an array, counting reads.

    Duck-typed rather than a real directory of TIFFs: what is being tested is the
    coordinate chain and the block planner, and a real decoder only slows that down.
    ``test_a_real_tiff_directory_round_trips`` covers the seam to the real class.
    """

    def __init__(self, volume):
        self.volume = np.asarray(volume)
        self.n_slices, self.n_rows, self.n_cols = self.volume.shape
        self.dtype = self.volume.dtype
        self.reads: list[int] = []

    @property
    def shape(self):
        return self.volume.shape

    def read_slice(self, z: int):
        if not 0 <= z < self.n_slices:
            raise IndexError(f"slice {z} out of range")  # matches TiffStack
        self.reads.append(int(z))
        return self.volume[z]

    def read_window(self, z, row0, row1, col0, col1):
        out = np.zeros((row1 - row0, col1 - col0), dtype=self.dtype)
        r0, r1 = max(0, row0), min(self.n_rows, row1)
        c0, c1 = max(0, col0), min(self.n_cols, col1)
        if r0 >= r1 or c0 >= c1:
            return out
        out[r0 - row0:r1 - row0, c0 - col0:c1 - col0] = self.read_slice(z)[r0:r1, c0:c1]
        return out

    def read_stack_window(self, z_lo, z_hi, row0, row1, col0, col1):
        return np.stack(
            [self.read_window(z, row0, row1, col0, col1) for z in range(z_lo, z_hi)], axis=0
        )


def unit_frame(shape) -> WorldFrame:
    """A frame where one micrometre is one voxel on both grids.

    Keeps the arithmetic in the tests readable: a world point ``(x, y, z)`` is raw
    ``(z, y, x)`` and segmentation ``(x, y, z)`` with no scaling, so an expected value
    can be written down rather than derived.
    """
    nz, ny, nx = shape
    return WorldFrame(
        raw_shape=(nz, ny, nx),
        raw_voxel=np.ones(3),
        seg_dims=np.array([nx, ny, nz], dtype=np.int64),
        seg_origin=np.zeros(3),
        seg_spacing=np.ones(3),
        nominal_voxel=np.ones(3),
    )


def arc(radius_um, sweep_rad=0.8, n=400) -> np.ndarray:
    """An exact circular arc in the z = 0 plane."""
    a = np.linspace(0.0, sweep_rad, n)
    return np.column_stack([radius_um * np.cos(a), radius_um * np.sin(a), np.zeros(n)])


def helix(a_um, c_um, turns=3.0, n=900) -> np.ndarray:
    """A helix whose radius of curvature is exactly ``(a^2 + c^2) / a``."""
    s = np.linspace(0.0, turns * 2 * np.pi, n)
    return np.column_stack([a_um * np.cos(s), a_um * np.sin(s), c_um * s])


def tilted_cylinder(shape, axis_xyz, radius_vox, *, centre=None) -> np.ndarray:
    """A round tube of ``radius_vox`` along an arbitrary direction, as ``(nz,ny,nx)``.

    Deliberately not axis-aligned: an axis-aligned tube is reproduced correctly by a
    sampler with two axes swapped, so it proves nothing.
    """
    nz, ny, nx = shape
    centre = np.array([nx / 2, ny / 2, nz / 2]) if centre is None else np.asarray(centre)
    axis = np.asarray(axis_xyz, dtype=np.float64)
    axis = axis / np.linalg.norm(axis)
    zz, yy, xx = np.mgrid[:nz, :ny, :nx]
    d = np.stack([xx - centre[0], yy - centre[1], zz - centre[2]], axis=-1).astype(np.float64)
    along = d @ axis
    perp2 = (d * d).sum(axis=-1) - along * along
    return (perp2 <= radius_vox * radius_vox).astype(np.uint8)


# --------------------------------------------------------------------------- #
# Curvature and the collision bound
# --------------------------------------------------------------------------- #


def test_a_circular_arc_reports_its_own_radius():
    pts = arc(5000.0)
    tangents, _n, _b = rf.frames(pts)
    assert rf.curvature(pts, tangents).r_min_um == pytest.approx(5000.0, rel=1e-3)


def test_the_end_steps_are_not_optimistic():
    """The frame's end tangents are chords, which halves the turn measured there.

    Untreated, ``R = ds / theta`` comes out at *twice* the true radius at each end --
    an optimistic bound at exactly the two places a path is most likely to have been
    cut mid-bend. Measured on a 5 mm arc: 9,999 um against a true 5,000 um.
    """
    pts = arc(5000.0)
    tangents, _n, _b = rf.frames(pts)
    curv = rf.curvature(pts, tangents)
    assert curv.r_step_um[0] == pytest.approx(5000.0, rel=1e-3)
    assert curv.r_step_um[-1] == pytest.approx(5000.0, rel=1e-3)

    # ...and this is what it would have said without the fix.
    raw = curv.ds_um / curv.theta_rad
    assert raw[0] == pytest.approx(10000.0, rel=1e-3)


def test_a_helix_matches_its_closed_form_radius_of_curvature():
    a, c = 3000.0, 1000.0
    pts = helix(a, c)
    tangents, _n, _b = rf.frames(pts)
    assert rf.curvature(pts, tangents).r_min_um == pytest.approx((a * a + c * c) / a, rel=1e-3)


def test_a_straight_line_puts_no_bound_on_the_half_width():
    pts = np.column_stack([np.arange(200.0), np.zeros(200), np.zeros(200)])
    tangents, _n, _b = rf.frames(pts)
    curv = rf.curvature(pts, tangents)
    assert not np.isfinite(curv.r_min_um)
    assert rf.collision_free(curv, 1e6)[0]


def test_the_bound_is_where_the_planes_actually_start_crossing():
    """``h < R`` is checked against the geometry, not against itself."""
    pts = arc(5000.0, n=200)
    t, n, b = rf.frames(pts)
    curv = rf.curvature(pts, t)
    r = curv.r_min_um

    ok_below, below = rf.planes_disjoint(pts, t, n, b, 0.9 * r)
    ok_above, above = rf.planes_disjoint(pts, t, n, b, 1.3 * r)
    assert ok_below, f"planes cross below the bound: {below[:3]}"
    assert not ok_above and above

    # ...and the cheap proxy agrees with it about which side of the bound we are on.
    assert rf.collision_free(curv, 0.9 * r, safety=1.0)[0]
    assert not rf.collision_free(curv, 1.3 * r, safety=1.0)[0]


def test_a_hairpin_is_caught_even_though_neighbouring_planes_are_fine():
    """The curvature bound is local; a vessel coming back on itself is not."""
    straight = np.linspace(0.0, 4000.0, 120)
    out = np.column_stack([straight, np.zeros(120), np.zeros(120)])
    back = np.column_stack([straight[::-1], np.full(120, 600.0), np.zeros(120)])
    bend = arc(300.0, np.pi, 60) + np.array([4000.0, 300.0, 0.0])
    pts = np.vstack([out, bend, back])

    t, n, b = rf.frames(pts)
    # Well inside the bound of the two straight limbs, which are what is colliding.
    ok, offenders = rf.planes_disjoint(pts, t, n, b, 800.0)
    assert not ok and offenders
    # The offending pairs are far apart along the vessel, which is the whole point.
    assert max(abs(a - c) for a, c, _d in offenders) > 100


# --------------------------------------------------------------------------- #
# Smoothing
# --------------------------------------------------------------------------- #


def test_smoothing_buys_back_the_curvature_bound():
    rng = np.random.default_rng(0)
    pts = arc(5000.0, n=300)
    noisy = pts + rng.normal(scale=30.0, size=pts.shape)

    t, _n, _b = rf.frames(noisy)
    before = rf.curvature(noisy, t).r_min_um

    line = rf.build_centreline(
        noisy, np.full(len(noisy), 400.0), step_um=40.0,
        half_of=lambda r: 4.0 * r, safety=0.8,
    )
    assert line.curvature.r_min_um > before
    assert line.smooth_iters > 0
    assert rf.collision_free(line.curvature, 4.0 * line.radii_um, safety=0.8)[0]


def test_smoothing_reports_how_far_it_moved_the_vessel():
    """Meeting a curvature target by relocating the centreline is not meeting it."""
    rng = np.random.default_rng(1)
    pts = arc(5000.0, n=300) + rng.normal(scale=30.0, size=(300, 3))
    line = rf.build_centreline(pts, np.full(300, 400.0), step_um=40.0)
    assert line.max_move_um > 0.0
    assert any("moved a median" in note for note in line.notes)


def test_an_unfixable_bend_is_clamped_and_said_so():
    """A tight arc cannot be smoothed straight, so the half-width has to give."""
    pts = arc(400.0, sweep_rad=2.0, n=200)
    line = rf.build_centreline(
        pts, np.full(len(pts), 900.0), step_um=20.0,
        half_of=lambda r: 4.0 * r, safety=0.8, max_smooth_iters=4,
    )
    geom = rf.plane_geometry(line, mode="radius", radii_k=4.0, safety=0.8)
    assert geom.clamped
    assert "clamped" in geom.describe()

    # The clamp is per plane, not global: where smoothing did straighten the run the
    # full width is kept, and only the planes still inside the bend give any up. So
    # the contract is the per-plane invariant, not a single worst-case number.
    bound = 0.8 * line.curvature.r_point_um
    bent = np.isfinite(bound)
    assert (geom.half_um[bent] <= bound[bent] + 1e-6).all()
    assert geom.half_um[~bent].size == 0 or (geom.half_um[~bent] == geom.requested_um[~bent]).all()


# --------------------------------------------------------------------------- #
# Frames
# --------------------------------------------------------------------------- #


def test_the_frame_is_orthonormal_everywhere():
    pts = helix(3000.0, 1000.0)
    t, n, b = rf.frames(pts)
    for a, c in ((t, n), (t, b), (n, b)):
        assert np.abs(np.einsum("ij,ij->i", a, c)).max() < 1e-9
    for v in (t, n, b):
        assert np.abs(np.linalg.norm(v, axis=1) - 1.0).max() < 1e-9


def test_the_frame_does_not_twist_where_an_arbitrary_seed_does():
    """The reason ``crosssection._plane_axes`` is not reused for a stack.

    Twist is rotation of the in-plane axes *about the tangent*: transport each frame's
    normal onto the next frame's plane and read the signed angle it moved.

    What matters is the **accumulated** twist, not the per-step one. Per step both
    frames look fine -- 0.006 degrees for the RMF against 0.38 for an arbitrary seed,
    neither of which you would notice. But the seeded error is *systematic*: it has the
    same sign at every step, so over a few hundred planes it sums to most of a full
    turn and the image visibly spins as you scroll, while the feature it is showing
    stays put. The RMF's residual is discretisation and does not accumulate.
    """
    from hipct_seg_debug.crosssection import _plane_axes

    pts = helix(3000.0, 1000.0)
    t, n, _b = rf.frames(pts)

    def total_twist(normals):
        total = 0.0
        for i in range(len(normals) - 1):
            p = normals[i] - (normals[i] @ t[i + 1]) * t[i + 1]
            p /= max(np.linalg.norm(p), 1e-12)
            nxt = normals[i + 1]
            total += np.degrees(np.arctan2(np.cross(p, nxt) @ t[i + 1], p @ nxt))
        return abs(total)

    seeded = np.array([_plane_axes(tv)[0] for tv in t])
    assert total_twist(n) < 5.0
    assert total_twist(seeded) > 300.0


def test_the_seed_normal_fixes_the_rotation_of_the_whole_stack():
    pts = helix(3000.0, 1000.0, turns=1.0, n=200)
    t, n_a, b_a = rf.frames(pts, seed_normal=np.array([0.0, 0.0, 1.0]))
    _t, n_b, _b = rf.frames(pts, seed_normal=b_a[0])

    # A different seed rotates every plane by the same in-plane angle, not by a
    # drifting one -- that is what "transported with zero twist" buys.
    ang = np.degrees(np.arctan2(
        np.einsum("ij,ij->i", n_b, b_a), np.einsum("ij,ij->i", n_b, n_a)
    ))
    assert np.ptp(ang) < 1e-6


# --------------------------------------------------------------------------- #
# The oblique sampler
# --------------------------------------------------------------------------- #


def _ramp(shape, a, b, c):
    """``v = a*z + b*y + c*x`` -- linear in each axis, so trilinear is exact."""
    nz, ny, nx = shape
    zz, yy, xx = np.mgrid[:nz, :ny, :nx]
    return (a * zz + b * yy + c * xx).astype(np.float32)


@pytest.mark.parametrize("order", [1, 3, 5])
def test_a_multilinear_field_is_reproduced_exactly(order):
    """The single most load-bearing test here: it pins the whole coordinate chain.

    Every spline order of 1 or above reproduces ``a*z + b*y + c*x`` exactly, so the
    recovered coefficients are a direct read-out of which world axis reached which array
    axis. A transposed pair permutes them; a missed reversal in ``um_to_raw`` swaps a
    and c. Parametrised so the guard survives a change of default order.
    """
    shape = (60, 70, 80)
    a, b, c = 3.0, 5.0, 7.0
    stack = FakeStack(_ramp(shape, a, b, c))
    frame = unit_frame(shape)
    sampler = rf.ObliqueSampler(stack, frame, order=order)

    rng = np.random.default_rng(3)
    pts = rng.uniform([25, 25, 25], [55, 45, 35], size=(500, 3))  # world (x, y, z)
    got = sampler.sample(pts)
    want = a * pts[:, 2] + b * pts[:, 1] + c * pts[:, 0]
    assert np.allclose(got, want, rtol=0, atol=1e-2)


def test_an_unsupported_order_is_refused_rather_than_silently_wrong():
    stack = FakeStack(np.zeros((10, 10, 10), dtype=np.float32))
    with pytest.raises(rf.ReformatError, match="order"):
        rf.ObliqueSampler(stack, unit_frame((10, 10, 10)), order=7)


def test_the_block_pad_grows_with_the_spline_order():
    """The prefilter for order >= 2 is an IIR filter, so it is not local."""
    stack = FakeStack(np.zeros((10, 10, 10), dtype=np.float32))
    frame = unit_frame((10, 10, 10))
    assert rf.ObliqueSampler(stack, frame, order=1).pad == 1
    assert rf.ObliqueSampler(stack, frame, order=3).pad >= 16
    assert rf.ObliqueSampler(stack, frame, order=5).pad >= rf.SPLINE_PAD[3]


def test_samples_outside_the_volume_are_zero_on_every_axis():
    """z raises in ``read_slice`` while rows and cols zero-pad; both must give 0."""
    shape = (20, 20, 20)
    stack = FakeStack(np.full(shape, 1000.0, dtype=np.float32))
    sampler = rf.ObliqueSampler(stack, unit_frame(shape))

    outside = np.array([
        [10.0, 10.0, -5.0],   # past z, the axis that would raise
        [10.0, 10.0, 40.0],
        [-5.0, 10.0, 10.0],   # past x, the axis that pads
        [10.0, 40.0, 10.0],   # past y
    ])
    assert np.allclose(sampler.sample(outside), 0.0)
    assert sampler.sample(np.array([[10.0, 10.0, 10.0]]))[0] == pytest.approx(1000.0)


def test_a_request_entirely_off_the_stack_reads_nothing():
    shape = (20, 20, 20)
    stack = FakeStack(np.ones(shape, dtype=np.float32))
    sampler = rf.ObliqueSampler(stack, unit_frame(shape))
    stats = rf.SampleStats()
    got = sampler.sample(np.array([[10.0, 10.0, 100.0], [10.0, 10.0, 110.0]]), stats=stats)
    assert np.allclose(got, 0.0)
    assert stack.reads == []
    assert stats.outside == 1.0


@pytest.mark.parametrize("order", [0, 1, 3, 5])
def test_chunking_does_not_change_the_answer(order):
    """Exactly equal, not approximately -- the block pad is what makes it so.

    Two separate failures are guarded here, and the second is why this is parametrised.

    ``map_coordinates(mode="constant")`` returns ``cval`` for anything outside the
    block, so without *any* pad every block seam gets a stripe of zeros.

    And for ``order >= 2`` ``map_coordinates`` prefilters, with an IIR filter that is
    **not local** -- so a per-block prefilter differs from a whole-volume one for far
    more than one voxel in from the edge. Measured at order 3: a pad of 1 leaves 4.3e-3
    of error, 8 leaves 4.5e-7, and 16 is exact. Since this asserts *exact* equality
    between a one-block run and a many-block one, it is the guard on
    :data:`reformat.SPLINE_PAD`.
    """
    shape = (60, 60, 60)
    rng = np.random.default_rng(7)
    # Textured rather than a smooth ramp: a ramp is reproduced exactly by every order,
    # so a wrong prefilter pad would not show up in it.
    stack = FakeStack(rng.random(shape).astype(np.float32))
    frame = unit_frame(shape)
    line, geom = _straight_stack_geometry()

    one = rf.ObliqueSampler(stack, frame, budget_mb=4096.0, order=order)
    many = rf.ObliqueSampler(stack, frame, budget_mb=0.02, order=order)
    assert len(many.plan(line, geom)) > 3
    a = one.sample_planes(line, geom)
    b = many.sample_planes(line, geom)

    if order <= 1:
        # No prefilter, so the arithmetic is genuinely identical block by block and
        # anything less than exact means a missing pad.
        np.testing.assert_array_equal(a, b)
        return

    # With a prefilter, blocks of different extents run the IIR recursion over
    # different numbers of terms, so the two agree only to float32 rounding -- measured
    # at one ULP (5.96e-08) on a single plane of 53. The tolerance is set well inside
    # what a *wrong* pad costs: at order 3 a pad of 1 leaves 3.1e-02 and a pad of 8
    # leaves 3.7e-06, so this still fails by orders of magnitude if SPLINE_PAD shrinks.
    np.testing.assert_allclose(a, b, rtol=0, atol=1e-6)


def test_the_block_plan_covers_every_plane_exactly_once():
    line, geom = _straight_stack_geometry()
    stack = FakeStack(np.zeros((60, 60, 60), dtype=np.float32))
    runs = rf.ObliqueSampler(stack, unit_frame((60, 60, 60)), budget_mb=0.02).plan(line, geom)
    assert runs[0][0] == 0
    assert runs[-1][1] == len(line.coords_um)
    assert all(a[1] == b[0] for a, b in zip(runs, runs[1:]))


def _straight_stack_geometry():
    """A short diagonal run through the middle of a 60^3 volume."""
    pts = np.linspace([15.0, 15.0, 15.0], [45.0, 45.0, 45.0], 40)
    line = rf.build_centreline(pts, np.full(len(pts), 3.0), step_um=1.0)
    geom = rf.plane_geometry(line, mode="fixed", radii_k=2.0, size_px=11)
    return line, geom


def test_a_real_tiff_directory_round_trips(tmp_path):
    """One test through the real ``TiffStack``, including the z clip.

    ``read_window`` pads rows and columns but ``read_slice`` *raises* on an
    out-of-range z, so z is the one axis the sampler has to clip itself. The plane
    here deliberately straddles slice 0.

    Values are checked at ``order=1``, because that is the claim being made -- that the
    clip happens and the right pixels come back -- and a spline near a clipped boundary
    legitimately overshoots against the ``mode="constant"`` extension. That the default
    order also survives the clip is asserted separately, below.
    """
    tifffile = pytest.importorskip("tifffile")
    from hipct_seg_debug.tiffstack import TiffStack

    shape = (12, 16, 16)
    volume = _ramp(shape, 1.0, 2.0, 4.0).astype(np.uint16)
    for z in range(shape[0]):
        tifffile.imwrite(tmp_path / f"s_{z:04d}.tif", volume[z])

    stack = TiffStack(tmp_path)
    pts = np.array([[8.0, 8.0, 1.5], [8.0, 8.0, -3.0], [8.0, 8.0, 6.25]])
    got = rf.ObliqueSampler(stack, unit_frame(shape), order=1).sample(pts)
    assert got[0] == pytest.approx(1.0 * 1.5 + 2.0 * 8 + 4.0 * 8, abs=1e-3)
    assert got[1] == 0.0
    assert got[2] == pytest.approx(1.0 * 6.25 + 2.0 * 8 + 4.0 * 8, abs=1e-3)

    # The default order asks for a 16-voxel pad this 12-slice volume cannot give, so
    # the block is clipped on both sides. That must still come back without an
    # IndexError escaping from `read_slice`.
    deep = rf.ObliqueSampler(stack, unit_frame(shape)).sample(pts)
    assert np.isfinite(deep).all()
    assert deep[1] == 0.0


def test_the_mask_is_sampled_nearest_and_stays_a_label():
    shape = (30, 30, 30)
    labels = tilted_cylinder(shape, (1.0, 0.4, 0.2), 6.0)
    frame = unit_frame(shape)
    sampler = rf.LabelSampler(labels, frame)

    rng = np.random.default_rng(4)
    pts = rng.uniform(5, 25, size=(400, 3))
    got = sampler.sample(pts)
    assert set(np.unique(got)) <= {0, 1}

    ijk = np.rint(frame.um_to_seg(pts)).astype(int)
    want = labels[ijk[:, 2], ijk[:, 1], ijk[:, 0]]
    np.testing.assert_array_equal(got, want)


# --------------------------------------------------------------------------- #
# Phantoms: is the cut actually perpendicular
# --------------------------------------------------------------------------- #


def _sections(mask_stack):
    """``(r_area, major/minor ratio)`` per plane of a reformatted mask stack."""
    out = []
    for plane in mask_stack:
        blob = plane > 0
        area = float(blob.sum())
        if area < 8:
            continue
        coords = np.argwhere(blob).astype(np.float64)
        coords -= coords.mean(axis=0)
        eig = np.linalg.eigvalsh(np.cov(coords.T))
        ratio = np.sqrt(max(eig[1], 1e-12) / max(eig[0], 1e-12))
        out.append((np.sqrt(area / np.pi), ratio))
    assert out, "no plane produced a measurable section"
    return np.asarray(out)


def test_a_tilted_cylinder_gives_round_sections_of_the_right_size():
    """The perpendicularity test.

    A plane at angle ``phi`` to the true normal cuts a circular tube in an ellipse of
    axis ratio ``1 / cos phi``. So a section that comes out round is a statement that
    the plane really is perpendicular -- 1.1 would already mean the tangent is 25
    degrees out.
    """
    shape = (70, 70, 70)
    axis = np.array([1.0, 0.6, 0.45])
    radius = 7.0
    labels = tilted_cylinder(shape, axis, radius)
    frame = unit_frame(shape)

    direction = axis / np.linalg.norm(axis)
    centre = np.array([35.0, 35.0, 35.0])
    pts = centre + np.linspace(-20, 20, 60)[:, None] * direction
    line = rf.build_centreline(pts, np.full(len(pts), radius), step_um=1.0)
    geom = rf.plane_geometry(line, mode="fixed", radii_k=3.0, size_px=45)

    stack = rf.LabelSampler(labels, frame).sample_planes(line, geom)
    r_area, ratio = _sections(stack).T
    assert ratio.max() < 1.12, f"sections are elliptical: worst ratio {ratio.max():.3f}"
    assert np.median(r_area) * geom.px_um[0] == pytest.approx(radius, rel=0.12)


def test_the_lumen_stays_on_the_centre_pixel():
    shape = (70, 70, 70)
    axis = np.array([1.0, 0.6, 0.45])
    labels = tilted_cylinder(shape, axis, 7.0)
    direction = axis / np.linalg.norm(axis)
    pts = np.array([35.0, 35.0, 35.0]) + np.linspace(-20, 20, 60)[:, None] * direction

    line = rf.build_centreline(pts, np.full(len(pts), 7.0), step_um=1.0)
    geom = rf.plane_geometry(line, mode="fixed", radii_k=3.0, size_px=45)
    stack = rf.LabelSampler(labels, unit_frame(shape)).sample_planes(line, geom)

    centre = geom.size_px // 2
    for plane in stack:
        rows, cols = np.nonzero(plane)
        assert abs(rows.mean() - centre) < 1.5
        assert abs(cols.mean() - centre) < 1.5


def test_a_curved_tube_keeps_a_constant_section():
    """A bend is where an axial slice lies most and a reformat should not."""
    shape = (40, 80, 80)  # (nz, ny, nx): world z is short, x and y hold the bend
    radius, tube = 24.0, 5.0
    a = np.linspace(0.4, 1.2, 200)
    pts = np.column_stack([40 + radius * np.cos(a), 40 + radius * np.sin(a), np.full(200, 20.0)])

    # A torus about world z = 20, centred on (40, 40) in world x and y.
    zz, yy, xx = np.mgrid[:shape[0], :shape[1], :shape[2]]
    ring = np.sqrt((xx - 40.0) ** 2 + (yy - 40.0) ** 2)
    labels = (np.sqrt((ring - radius) ** 2 + (zz - 20.0) ** 2) <= tube).astype(np.uint8)

    line = rf.build_centreline(pts, np.full(len(pts), tube), step_um=1.0)
    geom = rf.plane_geometry(line, mode="fixed", radii_k=3.0, size_px=41)
    stack = rf.LabelSampler(labels, unit_frame(shape)).sample_planes(line, geom)

    r_area = _sections(stack)[:, 0]
    assert r_area.std() / r_area.mean() < 0.08


# --------------------------------------------------------------------------- #
# Chaining a selection
# --------------------------------------------------------------------------- #


NODES = [(0.0, 0.0, 0.0), (100.0, 0.0, 0.0), (200.0, 0.0, 0.0), (300.0, 0.0, 0.0),
         (200.0, 100.0, 0.0)]


def test_two_segments_meeting_at_a_node_become_one_run():
    graph = graph_from(NODES, [(0, 1, 11, 5.0), (1, 2, 11, 5.0)])
    chains, notes = rf.chain_segments(graph, [0, 1])
    assert len(chains) == 1
    assert chains[0].segment_ids == (0, 1)
    assert not notes


def test_the_junction_point_appears_once():
    """Amira gives each edge its own copy of the shared node position."""
    graph = graph_from(NODES, [(0, 1, 11, 5.0), (1, 2, 11, 5.0)])
    chains, _ = rf.chain_segments(graph, [0, 1])
    coords, radii, sids = rf.chain_arrays(graph, chains[0].steps)
    assert len(coords) == 11 + 11 - 1
    assert len(radii) == len(coords) == len(sids)
    # No zero-length step, which is what a duplicate would leave behind.
    assert np.linalg.norm(np.diff(coords, axis=0), axis=1).min() > 1e-9


def test_a_segment_reached_at_node2_is_reversed():
    """Built so that walking the chain requires flipping the second segment."""
    graph = graph_from(NODES, [(1, 0, 11, 5.0), (1, 2, 11, 5.0)])
    chains, _ = rf.chain_segments(graph, [0, 1])
    coords, _r, _s = rf.chain_arrays(graph, chains[0].steps)
    steps = np.linalg.norm(np.diff(coords, axis=0), axis=1)
    # A missed flip shows up as one enormous step where the run jumps back.
    assert steps.max() < 1.5 * np.median(steps)


def test_a_branch_inside_the_selection_is_split_not_guessed():
    graph = graph_from(
        NODES, [(0, 1, 11, 5.0), (1, 2, 11, 5.0), (2, 3, 11, 5.0), (2, 4, 11, 5.0)]
    )
    chains, notes = rf.chain_segments(graph, [0, 1, 2, 3])
    assert len(chains) > 1
    assert any("node 2" in n for n in notes)


def test_a_disconnected_selection_yields_one_run_each_longest_first():
    graph = graph_from(NODES, [(0, 1, 11, 5.0), (2, 3, 21, 5.0)])
    chains, notes = rf.chain_segments(graph, [0, 1])
    assert len(chains) == 2
    assert chains[0].length_um >= chains[1].length_um
    assert any("separate runs" in n for n in notes)


def test_a_gap_at_a_seam_is_reported_rather_than_welded():
    """Two segments whose shared node is not actually shared."""
    nodes = [(0.0, 0.0, 0.0), (100.0, 0.0, 0.0), (500.0, 0.0, 0.0)]
    graph = graph_from(nodes, [(0, 1, 11, 5.0), (1, 2, 11, 5.0)])
    # Move the second segment's first point well away from the junction.
    seg = graph.segment(1)
    pid = seg["point_ids"][0]
    x, y, z, r = graph.points[pid]
    graph.move_point(pid, (x + 300.0, y, z))

    chains, _ = rf.chain_segments(graph, [0, 1])
    coords, _r, _s = rf.chain_arrays(graph, chains[0].steps)
    # Kept, not silently dropped: the gap is real and belongs in the geometry.
    assert len(coords) == 22


def test_a_segment_that_is_gone_is_reported_not_crashed_on():
    graph = graph_from(NODES, [(0, 1, 11, 5.0)])
    chains, notes = rf.chain_segments(graph, [0, 99])
    assert len(chains) == 1
    assert any("no longer in the graph" in n for n in notes)


def test_an_empty_selection_is_empty_not_an_error():
    graph = graph_from(NODES, [(0, 1, 11, 5.0)])
    assert rf.chain_segments(graph, [])[0] == []


# --------------------------------------------------------------------------- #
# End to end
# --------------------------------------------------------------------------- #


def test_build_produces_a_stack_with_everything_lined_up():
    shape = (70, 70, 70)
    axis = np.array([1.0, 0.6, 0.45])
    direction = axis / np.linalg.norm(axis)
    labels = tilted_cylinder(shape, axis, 7.0)
    volume = (labels * 500 + 100).astype(np.uint16)
    frame = unit_frame(shape)

    ends = np.array([35.0, 35.0, 35.0]) + np.array([-18.0, 18.0])[:, None] * direction
    graph = graph_from([tuple(ends[0]), tuple(ends[1])], [(0, 1, 40, 7.0)])

    out = rf.build(
        graph, frame, FakeStack(volume), [0], labels=labels,
        mode="fixed", radii_k=3.0, size_px=41, step_um=1.0,
    )
    assert out.raw.shape == out.mask.shape == (out.n_planes, 41, 41)
    assert out.raw.dtype == volume.dtype
    centre = 41 // 2
    assert out.mask[:, centre, centre].mean() > 0.95  # the lumen is on the centre pixel
    assert out.raw[:, centre, centre].mean() > 400    # ...and so is the bright core
    assert "planes" in out.describe()


def test_build_refuses_a_run_that_would_take_too_many_planes():
    graph = graph_from(NODES, [(0, 1, 11, 5.0)])
    with pytest.raises(rf.ReformatError, match="planes"):
        rf.build(graph, unit_frame((10, 10, 10)), FakeStack(np.zeros((10, 10, 10))),
                 [0], step_um=0.1, max_planes=10)


def test_graph_points_land_on_the_centre_pixel():
    """The chain's own points are the plane centres, by construction."""
    shape = (40, 40, 40)
    graph = graph_from([(10.0, 20.0, 20.0), (30.0, 20.0, 20.0)], [(0, 1, 21, 3.0)])
    out = rf.build(
        graph, unit_frame(shape), FakeStack(np.zeros(shape, dtype=np.uint16)), [0],
        mode="fixed", radii_k=2.0, size_px=21, step_um=1.0,
    )
    centre = 21 // 2
    assert len(out.graph_points)
    on_centre = np.abs(out.graph_points[:, 1:] - centre).max(axis=1) < 1.0
    assert on_centre.mean() > 0.5


# --------------------------------------------------------------------------- #
# native mode: a fixed frame at the acquisition's own resolution
# --------------------------------------------------------------------------- #


def _native_geometry(radius_um, size_px, voxel_um=33.0, curve_radius=None):
    pts = (np.linspace([0.0, 0.0, 0.0], [4000.0, 0.0, 0.0], 120)
           if curve_radius is None else arc(curve_radius, n=120))
    line = rf.build_centreline(pts, np.full(len(pts), radius_um), step_um=voxel_um)
    return rf.plane_geometry(line, mode="native", size_px=size_px, voxel_um=voxel_um)


@pytest.mark.parametrize("radius_um", [80.0, 250.0, 1500.0])
@pytest.mark.parametrize("size_px", [21, 129, 401])
def test_native_never_magnifies_whatever_the_vessel_or_the_frame(radius_um, size_px):
    """The whole point of the mode: one output pixel is one voxel, always.

    Every other mode derives the pitch from a half-width, so a small radius -- or a
    curvature clamp cutting the width -- turns straight into magnification. Here the
    pitch is pinned and the half-width follows, so the ratio is 1 by construction and
    cannot be moved by the vessel or by the frame size.
    """
    geom = _native_geometry(radius_um, size_px)
    assert np.allclose(geom.px_um, 33.0)
    assert np.allclose(geom.oversampling, 1.0)
    assert geom.size_px == size_px


def test_native_takes_its_half_width_from_the_frame_not_the_radius():
    geom = _native_geometry(250.0, 129, voxel_um=33.0)
    assert np.allclose(geom.half_um, 64 * 33.0)
    # ...and a different vessel in the same frame gets exactly the same extent.
    assert np.allclose(_native_geometry(1500.0, 129).half_um, geom.half_um)


def test_native_scale_widens_the_view_without_magnifying():
    """Two voxels per pixel is a wider window, not a blurrier one."""
    pts = np.linspace([0.0, 0.0, 0.0], [4000.0, 0.0, 0.0], 120)
    line = rf.build_centreline(pts, np.full(len(pts), 250.0), step_um=33.0)
    one = rf.plane_geometry(line, mode="native", size_px=65, voxel_um=33.0)
    two = rf.plane_geometry(line, mode="native", size_px=65, voxel_um=33.0,
                            native_scale=2.0)
    assert np.allclose(two.half_um, 2 * one.half_um)
    assert np.allclose(two.oversampling, 0.5)  # undersampled, never magnified


def test_native_reports_the_curvature_bound_rather_than_clamping_to_it():
    """Clamping here would shrink the pixel, not the view -- the opposite of the point."""
    geom = _native_geometry(250.0, 401, voxel_um=33.0, curve_radius=3000.0)
    assert geom.n_over_bound > 0
    assert not geom.clamped, "native must not clamp"
    assert np.allclose(geom.oversampling, 1.0), "clamping would have magnified"
    assert 3 <= geom.suggested_size_px < geom.size_px
    text = geom.describe()
    assert "would stay inside it" in text and "not clamped" in text.lower()


def test_the_suggested_frame_size_actually_stays_inside_the_bound():
    geom = _native_geometry(250.0, 401, voxel_um=33.0, curve_radius=3000.0)
    smaller = _native_geometry(250.0, geom.suggested_size_px, voxel_um=33.0,
                               curve_radius=3000.0)
    assert smaller.n_over_bound == 0


def test_native_needs_the_voxel_size():
    pts = np.linspace([0.0, 0.0, 0.0], [4000.0, 0.0, 0.0], 120)
    line = rf.build_centreline(pts, np.full(len(pts), 250.0), step_um=33.0)
    with pytest.raises(rf.ReformatError, match="voxel"):
        rf.plane_geometry(line, mode="native", size_px=65)


def test_an_unknown_mode_names_the_ones_that_exist():
    pts = np.linspace([0.0, 0.0, 0.0], [4000.0, 0.0, 0.0], 120)
    line = rf.build_centreline(pts, np.full(len(pts), 250.0), step_um=33.0)
    with pytest.raises(rf.ReformatError, match="native"):
        rf.plane_geometry(line, mode="nonsense", size_px=65, voxel_um=33.0)


def test_build_in_native_mode_produces_an_unmagnified_stack():
    shape = (70, 70, 70)
    axis = np.array([1.0, 0.6, 0.45])
    direction = axis / np.linalg.norm(axis)
    labels = tilted_cylinder(shape, axis, 7.0)
    volume = (labels * 500 + 100).astype(np.uint16)
    ends = np.array([35.0, 35.0, 35.0]) + np.array([-18.0, 18.0])[:, None] * direction
    graph = graph_from([tuple(ends[0]), tuple(ends[1])], [(0, 1, 40, 7.0)])

    out = rf.build(graph, unit_frame(shape), FakeStack(volume), [0], labels=labels,
                   mode="native", size_px=41, step_um=1.0)
    assert out.raw.shape == (out.n_planes, 41, 41)
    assert np.allclose(out.geometry.oversampling, 1.0)
    centre = 41 // 2
    assert out.mask[:, centre, centre].mean() > 0.95
