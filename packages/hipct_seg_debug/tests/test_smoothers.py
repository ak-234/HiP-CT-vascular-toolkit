"""The selectable centreline smoothers.

Four backends, one contract: move the centreline, never the radii, never a node. The
``coronary_sdf`` ones are skipped when that package is absent, exactly as the TEASAR
skeletoniser is skipped without ``kimimaro`` -- an optional backend must not be able to
fail the suite for someone who does not have it.
"""

from __future__ import annotations

import numpy as np
import pytest

from hipct_seg_debug.edit import smoothers

from .conftest_geometry import graph_from


def _has_coronary_sdf() -> bool:
    from hipct_seg_debug.edit._deps import ensure_coronary_sdf

    try:
        ensure_coronary_sdf()
    except ImportError:
        return False
    return True


needs_sdf = pytest.mark.skipif(
    not _has_coronary_sdf(), reason="coronary_sdf is not importable"
)
EXTERNAL = ("savgol", "bspline", "multiscale")


def noisy_line(n=61, amplitude=30.0, radius=200.0, seed=0):
    """A straight vessel with sub-radius jitter -- what smoothing is meant to remove."""
    rng = np.random.default_rng(seed)
    xs = np.linspace(0.0, 6000.0, n)
    g = graph_from([(xs[0], 0, 0), (xs[-1], 0, 0)], [(0, 1, n, radius)])
    coords = np.stack([xs, np.zeros(n), np.zeros(n)], axis=1)
    coords[1:-1, 1] += rng.normal(0.0, amplitude, n - 2)
    coords[1:-1, 2] += rng.normal(0.0, amplitude, n - 2)
    with g.batch("lay out"):
        g.set_segment_coords(0, coords)
    return g


# ------------------------------------------------------------------- dispatch


def test_none_is_a_no_op():
    g = noisy_line()
    before = g.coords(0).copy()
    result = smoothers.smooth("none", g)
    assert result.n_moved == 0
    assert np.array_equal(g.coords(0), before)


def test_an_unknown_smoother_is_refused():
    with pytest.raises(ValueError, match="unknown smoother"):
        smoothers.smooth("taubin", noisy_line())


def test_every_backend_is_listed():
    assert set(smoothers.SMOOTHERS) == {"none", "gaussian", "savgol", "bspline",
                                        "multiscale"}


# ------------------------------------------------------------------- gaussian


def test_gaussian_reduces_jitter_and_pins_the_ends():
    g = noisy_line()
    before = g.coords(0).copy()
    result = smoothers.smooth("gaussian", g, window_um=600.0)
    after = g.coords(0)

    assert result.n_moved > 0
    assert np.std(after[:, 1]) < np.std(before[:, 1])
    assert np.allclose(before[0], after[0])
    assert np.allclose(before[-1], after[-1])


# ------------------------------------------------------ the coronary_sdf ones


def _roughness(coords: np.ndarray) -> float:
    """Mean second difference -- the high-frequency content smoothing removes.

    Per-axis standard deviation is the wrong measure here: for ``savgol`` and
    ``bspline`` a curvature post-filter runs *after* the fit and moves points again
    (on this fixture, 57 of them), so the spread about the axis is not monotonic even
    though the line is demonstrably less jagged.
    """
    return float(np.abs(np.diff(coords, n=2, axis=0)).mean())


@needs_sdf
@pytest.mark.parametrize("name", EXTERNAL)
def test_external_smoother_reduces_roughness(name):
    g = noisy_line()
    before = _roughness(g.coords(0))
    smoothers.smooth(name, g)
    assert _roughness(g.coords(0)) < before


@needs_sdf
@pytest.mark.parametrize("name", EXTERNAL)
def test_external_smoother_never_touches_a_radius(name):
    """The invariant that protects `radius_perimeter`'s whole output."""
    g = noisy_line()
    before = g.radii(0).copy()
    smoothers.smooth(name, g)
    assert np.array_equal(g.radii(0), before)


@needs_sdf
@pytest.mark.parametrize("name", EXTERNAL)
def test_external_smoother_keeps_node_positions(name):
    """Segment endpoints belong to nodes; moving one is a topological act."""
    g = noisy_line()
    ends_before = {n: g.nodes[n][:3] for n in g.nodes}
    smoothers.smooth(name, g)
    for nid, xyz in ends_before.items():
        assert g.nodes[nid][:3] == pytest.approx(xyz, abs=1e-6)


@needs_sdf
def test_multiscale_reports_its_constraint_diagnostics():
    """The report *is* the qualification evidence; it must reach the caller."""
    g = noisy_line()
    result = smoothers.smooth("multiscale", g)

    for key in ("converged", "iterations", "unresolved_constraints",
                "curvature_violations_before", "curvature_violations_after",
                "new_branch_conflicts", "backtrack_fraction", "modified_points",
                "max_displacement_radius"):
        assert key in result.detail, key
    assert any("drift" in n for n in result.notes)


@needs_sdf
def test_multiscale_respects_its_trust_region():
    """Drift is capped at a quarter radius, and it reports what it used."""
    g = noisy_line(amplitude=150.0, radius=200.0)
    result = smoothers.smooth("multiscale", g)
    assert result.detail["max_displacement_radius"] <= 0.25 + 1e-9


def test_a_backend_that_moves_everything_by_nothing_is_called_inert():
    """`n_moved` alone reads a scaled-to-zero solution as success.

    Measured on LADAF-2024-28 at stride 1, ``multiscale`` reported 24,037 of 36,613
    points moved and moved them 0.2 um -- a thousandth of a radius -- because its own
    line search kept 0.46% of what it computed. The output was identical to no smoothing
    at all while every number said otherwise, which is how this came to be reported as a
    broken smoother rather than an inert one.
    """
    g = noisy_line(radius=200.0)
    result = smoothers.SmoothResult(smoother="multiscale", n_moved=24037,
                                    median_move_um=0.2,
                                    detail={"backtrack_fraction": 0.0046})
    smoothers._warn_if_inert(g, result)

    assert any("inert" in n for n in result.notes)
    assert any("0.46%" in n for n in result.notes), "it names the constraint, not the fit"


def test_a_real_move_is_not_called_inert():
    g = noisy_line(radius=200.0)
    result = smoothers.SmoothResult(smoother="gaussian", n_moved=59,
                                    median_move_um=25.0)
    smoothers._warn_if_inert(g, result)
    assert result.notes == []


@needs_sdf
def test_a_radius_rewrite_would_be_caught():
    """The guard itself is tested, or it is decoration."""
    g = noisy_line()
    before = smoothers._radii_snapshot(g)
    with g.batch("corrupt a radius"):
        g.scale_radii(0, 1.5)
    with pytest.raises(AssertionError, match="must leave the measured radius alone"):
        smoothers._check_radii_untouched(
            g, before, smoothers.SmoothResult(smoother="test")
        )


# ------------------------------------------------------------------ pipeline


def test_optimise_skeleton_runs_the_smoother_last_and_once():
    """Nothing may follow a smoother that certifies its own output."""
    from hipct_seg_debug.edit import skeleton_optimise as so

    g = noisy_line()
    report = so.optimise_skeleton(g, smoother="gaussian", verbose=False)

    assert report.smooth is not None
    assert report.smooth.smoother == "gaussian"
    # One smoothing entry in the timings, not one per re-centre pass.
    assert [k for k in report.seconds if k.startswith("smooth")] == ["smooth"]


def test_optimise_skeleton_can_skip_smoothing_entirely():
    from hipct_seg_debug.edit import skeleton_optimise as so

    g = noisy_line()
    before = g.coords(0).copy()
    report = so.optimise_skeleton(g, smoother="none", deloop=False, prune=False)

    assert report.smooth is None
    assert np.array_equal(g.coords(0), before)
