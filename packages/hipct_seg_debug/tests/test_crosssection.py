"""The perpendicular plane cut, and the three things measured in it.

`crosssection.measure` is the audit that decides whether a stored radius honours the
perimeter measured at its own cross-section, and `cut` underneath it is now shared with
re-centring and the radius pass. It had no tests; the companion-lumen branch in
particular was reachable only on real data, which is how a stale variable survived a
refactor there.
"""

from __future__ import annotations

import numpy as np
import pytest

from hipct_seg_debug import crosssection as cs

from .conftest_geometry import SPACING, axis_graph, cylinder, make_frame

pytest.importorskip("cv2")

SHAPE = (40, 40, 80)


@pytest.fixture
def frame():
    return make_frame(SHAPE)


def _sampler(frame, volume):
    return cs._PlaneSampler(volume, frame)


# -------------------------------------------------------------- the sampler

# `_PlaneSampler` reads through one of three doors -- a decoded array, a lattice that
# can hand back a row band, and a lattice that can only hand back a whole slice -- and
# the middle one is the only one the fast path uses on real data. All three must
# return the same labels, or a measurement silently depends on how the mask was
# stored.


class _SliceOnly:
    """A lattice that can only produce whole planes, like an older reader."""

    def __init__(self, volume):
        self.volume = np.asarray(volume, dtype=np.uint8)
        self.nz, self.ny, self.nx = self.volume.shape
        self.reads = 0

    def slice_z(self, k):
        self.reads += 1
        return self.volume[k].copy()


class _WithRows(_SliceOnly):
    """A lattice that decodes row bands, which is what `rle.ByteRLELattice` does."""

    def __init__(self, volume):
        super().__init__(volume)
        self.row_reads = 0

    def slice_rows(self, k, row0, row1):
        self.row_reads += 1
        return self.volume[k, max(row0, 0):min(row1, self.ny)].copy()


@pytest.fixture
def noisy():
    return (np.random.default_rng(19).random(SHAPE) < 0.4).astype(np.uint8)


def _points(rng, n, shape):
    nz, ny, nx = shape
    return np.stack([rng.uniform(-3, nx + 3, n),
                     rng.uniform(-3, ny + 3, n),
                     rng.uniform(-3, nz + 3, n)], axis=1)


@pytest.mark.parametrize("door", ["rows", "slices"])
def test_every_reader_samples_the_same_labels(frame, noisy, door):
    lattice = _WithRows(noisy) if door == "rows" else _SliceOnly(noisy)
    points = _points(np.random.default_rng(2), 400, SHAPE)

    direct = cs._PlaneSampler(noisy, frame).at(points)
    assert np.array_equal(cs._PlaneSampler(lattice, frame).at(points), direct)
    if door == "rows":
        assert lattice.row_reads and not lattice.reads, "the band door was the one used"


def test_a_plane_is_sampled_the_same_through_a_lattice_as_through_an_array(frame, noisy):
    u, v = cs._plane_axes(np.array([0.3, 0.8, 0.5]))
    centre = np.array([38.0, 19.0, 21.0])
    direct = cs._PlaneSampler(noisy, frame)
    banded = cs._PlaneSampler(_WithRows(noisy), frame, band_rows=7)

    for half in (1, 3, 12, 30):
        assert np.array_equal(banded.plane(centre, u, v, half),
                              direct.plane(centre, u, v, half)), half


def test_samples_outside_the_lattice_read_zero(frame, noisy):
    sampler = cs._PlaneSampler(_WithRows(noisy), frame)
    outside = np.array([[-5.0, 3.0, 3.0], [3.0, -5.0, 3.0], [3.0, 3.0, -5.0],
                        [1e6, 3.0, 3.0], [3.0, 1e6, 3.0], [3.0, 3.0, 1e6]])
    assert not sampler.at(outside).any()
    # And a wholly-outside plane is all zeros rather than an index error.
    u, v = cs._plane_axes(np.array([0.0, 0.0, 1.0]))
    assert not sampler.plane(np.array([-500.0, -500.0, -500.0]), u, v, 4).any()


def test_a_full_cache_keeps_answering_correctly(frame, noisy):
    """Eviction is where a cache stops being an implementation detail."""
    lattice = _WithRows(noisy)
    tiny = cs._PlaneSampler(lattice, frame, cache_bytes=1, band_rows=4)
    points = _points(np.random.default_rng(8), 500, SHAPE)

    assert np.array_equal(tiny.at(points), cs._PlaneSampler(noisy, frame).at(points))
    assert len(tiny._cache) <= tiny._limit


def test_a_band_is_read_once_however_many_samples_land_in_it(frame, noisy):
    """The whole point of the grouping: one read per band, not one per sample."""
    lattice = _WithRows(noisy)
    sampler = cs._PlaneSampler(lattice, frame, band_rows=SHAPE[1])
    u, v = cs._plane_axes(np.array([0.0, 0.0, 1.0]))  # one slice, one band

    plane = sampler.plane(np.array([20.0, 20.0, 10.0]), u, v, 8)
    assert plane.size == 289
    assert lattice.row_reads == 1
    sampler.plane(np.array([20.0, 20.0, 10.0]), u, v, 8)
    assert lattice.row_reads == 1, "and cached for the next candidate tangent"


def test_a_cut_is_the_same_through_a_lattice_as_through_an_array(frame):
    """The measurement itself, not just the samples under it."""
    mask = cylinder(SHAPE, 5, 5, 75)
    args = (np.array([40.0, 20.0, 20.0]), np.array([1.0, 0.0, 0.0]), 14)
    direct = cs.cut(cs._PlaneSampler(mask, frame), *args)
    banded = cs.cut(cs._PlaneSampler(_WithRows(mask), frame, band_rows=6), *args)

    assert direct is not None and banded is not None
    assert np.array_equal(direct.blob4, banded.blob4)
    assert direct.half == banded.half and direct.touches_border == banded.touches_border


# ------------------------------------------------------------------------ cut


def test_cut_finds_the_component_holding_the_centre(frame):
    mask = cylinder(SHAPE, 5, 5, 75)
    c = cs.cut(_sampler(frame, mask), np.array([40.0, 20.0, 20.0]),
               np.array([1.0, 0.0, 0.0]), 14)

    assert c is not None
    assert c.n_components8 == 1
    assert c.blob4.sum() == c.blob8.sum()
    assert not c.touches_border
    assert not c.pinched


def test_cut_grows_the_window_until_the_section_fits(frame):
    """A window sized from a radius we distrust must not crop the section silently."""
    mask = cylinder(SHAPE, 5, 5, 75)
    sampler = _sampler(frame, mask)
    centre, tangent = np.array([40.0, 20.0, 20.0]), np.array([1.0, 0.0, 0.0])

    generous = cs.cut(sampler, centre, tangent, 14)
    cramped = cs.cut(sampler, centre, tangent, 3, max_half=32)

    assert cramped.half > 3, "it grew"
    assert cramped.grew and not generous.grew
    assert cramped.blob4.sum() == generous.blob4.sum(), "and recovered the whole section"


def test_grow_to_stops_the_window_running_away(frame):
    """A plane cut along a vessel touches the border at *any* width.

    That is the runaway `grow_to` exists to stop: without it the window doubles to
    `max_half` -- 64 voxels, a 4.2 mm half-width at stride 1 -- and the centroid of the
    resulting streak is somewhere down the vessel rather than across it. Callers that
    need a position out of the cut cap the growth and then refuse the result.
    """
    mask = cylinder(SHAPE, 5, 5, 75)
    sampler = _sampler(frame, mask)
    centre = np.array([40.0, 20.0, 20.0])
    along = np.array([0.0, 0.0, 1.0])  # plane *contains* the vessel axis

    runaway = cs.cut(sampler, centre, along, 4, max_half=64)
    capped = cs.cut(sampler, centre, along, 4, max_half=64, grow_to=8)

    assert runaway.half == 64, "it chased the streak all the way to the ceiling"
    assert capped.half == 8, "grow_to stopped it"
    # Either way the section never closes, which is the signal a caller acts on.
    assert capped.touches_border and not capped.trustworthy


def test_a_proper_section_is_trustworthy(frame):
    mask = cylinder(SHAPE, 5, 5, 75)
    c = cs.cut(_sampler(frame, mask), np.array([40.0, 20.0, 20.0]),
               np.array([1.0, 0.0, 0.0]), 14, grow_to=20)
    assert c.trustworthy and not c.grew


def test_cut_returns_none_outside_the_mask(frame):
    mask = cylinder(SHAPE, 5, 5, 75)
    c = cs.cut(_sampler(frame, mask), np.array([40.0, 2.0, 2.0]),
               np.array([1.0, 0.0, 0.0]), 10)
    assert c is None


def test_cut_returns_none_for_a_section_below_the_blob_floor(frame):
    mask = cylinder(SHAPE, 1, 5, 75)
    c = cs.cut(_sampler(frame, mask), np.array([40.0, 20.0, 20.0]),
               np.array([1.0, 0.0, 0.0]), 6)
    assert c is None, "three voxels cannot support a shape fit"


# -------------------------------------------------------------------- measure


def test_measure_recovers_a_known_radius(frame):
    r_vox = 6
    mask = cylinder(SHAPE, r_vox, 5, 75)
    graph = axis_graph(frame, 8, 72, r_vox * SPACING, cy=20, cz=20).to_spatial_graph()

    prof = cs.measure(graph, frame, mask, min_radius_um=0.0)

    assert prof.measured.any()
    assert np.nanmedian(prof.r_area[prof.measured]) == pytest.approx(
        r_vox * SPACING, rel=0.15
    )
    # A round section: isoperimetric ratio near 1, and the stored radius honoured.
    assert np.nanmedian(prof.isoperimetric[prof.measured]) < 1.35
    ratio = prof.perimeter_ratio(graph.thickness)[prof.measured]
    assert np.nanmedian(ratio) == pytest.approx(1.0, abs=0.2)


def test_measure_flags_a_companion_lumen(frame):
    """The signature of one collapsed vessel segmented as two.

    This is the branch that a stale `lab8` reference survived in, because nothing
    exercised a plane carrying a second component.
    """
    mask = cylinder(SHAPE, 4, 5, 75, cy=14, cz=20)
    mask = mask | cylinder(SHAPE, 4, 5, 75, cy=26, cz=20)  # a parallel neighbour
    graph = axis_graph(frame, 8, 72, 4 * SPACING, cy=14, cz=20).to_spatial_graph()

    prof = cs.measure(graph, frame, mask, min_radius_um=0.0)

    assert prof.measured.any()
    seen = prof.companions[prof.measured]
    assert np.isfinite(seen).any(), "the second lumen is reported, not crashed on"
    assert np.nanmin(seen) > 0


def test_measure_reports_a_collapsed_section_as_non_circular(frame):
    from .conftest_geometry import slit

    mask = slit(SHAPE, half_y=8, half_z=1, x0=5, x1=75, cy=20, cz=20)
    graph = axis_graph(frame, 8, 72, 40.0, cy=20, cz=20).to_spatial_graph()

    prof = cs.measure(graph, frame, mask, min_radius_um=0.0)

    assert prof.measured.any()
    assert np.nanmedian(prof.isoperimetric[prof.measured]) > 1.6, "far from circular"
    assert np.nanmedian(prof.minor_ratio[prof.measured]) < \
        np.nanmedian(prof.major_ratio[prof.measured])


# ----------------------------------------------------- the transverse search

# The stability test cannot see tilt. Its three slabs are stepped *along the candidate
# normal*, so on a straight vessel of constant calibre an oblique plane cuts three
# identical ellipses and scores a perfect 1.000/1.000. Everything below is that hole
# and the gate that closes it; see `cs.TRANSVERSE_AXIS_RATIO`.


def _long_tube():
    shape = (40, 40, 400)
    frame = make_frame(shape)
    return frame, cylinder(shape, 6, 5, 395, cy=20, cz=20)


def _tilted(off_degrees):
    """A unit normal `off_degrees` away from the x axis the tube runs along."""
    a = np.deg2rad(off_degrees)
    return np.array([np.cos(a), np.sin(a), 0.0])


def _radius_um(chosen):
    return cs._perimeter_um(chosen.cut.blob4, SPACING) / (2.0 * np.pi)


def test_the_stability_metrics_are_blind_to_a_tilted_plane():
    """The premise. Without this the gate below looks like belt and braces."""
    frame, mask = _long_tube()
    chosen = cs.stable_transverse_cut(
        _sampler(frame, mask), np.array([200.0, 20.0, 20.0]), _tilted(40),
        6.0, spacing_um=SPACING, max_half=64, search_degrees=0.0,
    )
    assert chosen is not None
    assert chosen.area_ratio == pytest.approx(1.0, abs=0.05)
    assert chosen.perimeter_ratio == pytest.approx(1.0, abs=0.05)
    # ... and yet the section it pronounces stable is a visibly flattened ellipse
    # whose perimeter is a quarter too long for the tube it was cut from.
    assert chosen.axis_ratio > 1.3
    assert _radius_um(chosen) > 1.2 * 6 * SPACING


def test_an_oblique_cut_is_straightened_and_a_round_one_is_left_alone():
    frame, mask = _long_tube()
    sampler = _sampler(frame, mask)
    centre = np.array([200.0, 20.0, 20.0])
    true_r = 6 * SPACING

    on_axis = cs.stable_transverse_cut(
        sampler, centre, _tilted(0), 6.0, spacing_um=SPACING,
        max_half=64, search_degrees=20.0,
    )
    assert on_axis is not None and not on_axis.searched
    assert _radius_um(on_axis) == pytest.approx(true_r, rel=0.1)

    # 30 degrees off is inside the 20-degree cone once the fitted tangent is rejected
    # for being elongated, so the search must recover the same radius.
    oblique = cs.stable_transverse_cut(
        sampler, centre, _tilted(30), 6.0, spacing_um=SPACING,
        max_half=64, search_degrees=20.0,
    )
    assert oblique is not None and oblique.searched
    assert _radius_um(oblique) == pytest.approx(_radius_um(on_axis), rel=1e-6)
    assert oblique.obliquity > 1.05, "the fitted tangent's over-read must be reported"


def test_raising_the_gate_restores_the_over_read():
    """The knob `--transverse-axis-ratio` turns, so its effect is stated here."""
    frame, mask = _long_tube()
    sampler = _sampler(frame, mask)
    kw = dict(spacing_um=SPACING, max_half=64, search_degrees=20.0)

    gated = cs.stable_transverse_cut(
        sampler, np.array([200.0, 20.0, 20.0]), _tilted(30), 6.0,
        transverse_axis_ratio=cs.TRANSVERSE_AXIS_RATIO, **kw)
    ungated = cs.stable_transverse_cut(
        sampler, np.array([200.0, 20.0, 20.0]), _tilted(30), 6.0,
        transverse_axis_ratio=float("inf"), **kw)

    assert not ungated.searched
    assert _radius_um(ungated) > 1.05 * _radius_um(gated)


def test_a_collapsed_lumen_cut_square_is_not_rotated_away_from_square():
    """Minimising the perimeter is right in both regimes, which is why one gate does.

    A slit is elongated, so the gate fires and the whole cone is searched; but a
    transverse cut of a slit already has the shortest boundary available, so the
    search must come back to where it started rather than inventing a tilt.
    """
    from .conftest_geometry import slit

    shape = (40, 40, 400)
    frame = make_frame(shape)
    mask = slit(shape, 8, 1, 5, 395, cy=20, cz=20)
    sampler = _sampler(frame, mask)

    square = cs.stable_transverse_cut(
        sampler, np.array([200.0, 20.0, 20.0]), _tilted(0), 5.0,
        spacing_um=SPACING, max_half=64, search_degrees=20.0)
    assert square is not None
    assert square.axis_ratio > 3.0, "the fixture must be flat, or this proves nothing"
    assert not square.searched
    assert square.obliquity == pytest.approx(1.0)

    # And a slit cut obliquely is still straightened, by the same rule.
    tilted = cs.stable_transverse_cut(
        sampler, np.array([200.0, 20.0, 20.0]), _tilted(30), 5.0,
        spacing_um=SPACING, max_half=64, search_degrees=20.0)
    assert tilted is not None and tilted.searched
    assert _radius_um(tilted) == pytest.approx(_radius_um(square), rel=1e-6)


def test_grow_radii_caps_the_search_without_costing_the_wide_sections():
    """What makes a large `max_half` affordable.

    Growth is unbounded relative to the vessel, so a plane that is not transverse --
    a streak along the axis, which touches the border at any width -- escalates to
    `max_half` and is refused anyway, having paid for every doubling. Raising
    `max_half` to reach the few proximal vessels that need it therefore taxes every
    narrow one. `grow_radii` is the ceiling that separates the two: a vessel that
    genuinely needs a wide window has a wide radius, so the cap does not bind on it.
    """
    frame, mask = _long_tube()          # radius 6 voxels
    sampler = _sampler(frame, mask)
    centre = np.array([200.0, 20.0, 20.0])

    # The tube is 6 voxels; a 4-radius ceiling is 26 and never binds on a good cut.
    capped = cs.stable_transverse_cut(
        sampler, centre, _tilted(0), 6.0, spacing_um=SPACING,
        max_half=256, search_degrees=20.0, grow_radii=4.0)
    free = cs.stable_transverse_cut(
        sampler, centre, _tilted(0), 6.0, spacing_um=SPACING,
        max_half=256, search_degrees=20.0)

    assert capped is not None and free is not None
    assert capped.cut.half == free.cut.half
    assert _radius_um(capped) == pytest.approx(_radius_um(free), rel=1e-9)

    # And it does bind where the cut is hopeless. A normal square to the axis cuts
    # *along* the vessel, so the section is a streak that touches the border at every
    # width and doubles all the way to the ceiling -- where, on a tube that ends
    # inside a 513-voxel window, it finally "closes" around the whole vessel and
    # reports a radius many times the truth. That is the runaway, and its cost is
    # paid on every narrow vessel the moment `max_half` is raised for the wide ones.
    runaway = cs.cut(sampler, centre, _tilted(90), 17, max_half=256)
    bounded = cs.cut(sampler, centre, _tilted(90), 17, max_half=256,
                     grow_to=int(4.0 * 6.0) + 2)

    assert runaway.half == 256, "unbounded, it chases the streak to the ceiling"
    assert bounded.half == 26, "grow_radii stops it at four radii"
    assert bounded.touches_border, "and the caller refuses it, which is the point"
    true_r = 6 * SPACING
    assert cs._perimeter_um(runaway.blob4, SPACING) / (2 * np.pi) > 10 * true_r
