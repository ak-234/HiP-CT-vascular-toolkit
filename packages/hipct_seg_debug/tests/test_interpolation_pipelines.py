"""Every stage that measures something must decline to measure an invented point.

One test per consumer, each shaped so that ignoring the flag gives a visibly different
answer rather than a slightly different one. The flag is always written explicitly with
``annotate`` -- no stage detects on its own, and a test that relied on detection would be
testing the detector twice instead of testing the exclusion.
"""

from __future__ import annotations

import numpy as np
import pytest

from hipct_seg_debug.edit import interpolation as ip
from hipct_seg_debug.edit import radius_repair as rr
from hipct_seg_debug.edit import skeleton_optimise as so
from hipct_seg_debug.edit import supermetric as sm
from hipct_seg_debug.edit.adapter import Triple
from hipct_seg_debug.edit.graphmodel import EditableGraph

STEP = 100.0


def _segment(coords, radii) -> EditableGraph:
    coords = np.asarray(coords, dtype=np.float64)
    radii = np.asarray(radii, dtype=np.float64)
    points = {
        i: (float(p[0]), float(p[1]), float(p[2]), float(r))
        for i, (p, r) in enumerate(zip(coords, radii))
    }
    nodes = {0: (*coords[0], 0), 1: (*coords[-1], 0)}
    segments = [{"id": 0, "node1": 0, "node2": 1, "point_ids": list(range(len(coords)))}]
    return EditableGraph(Triple(nodes, points, segments))


def _flag(graph, indices):
    """Mark these segment-0 positions as invented, without running the detector."""
    ids = graph.segment(0)["point_ids"]
    found = ip.Detection(
        flags={ids[i]: ip.STRAIGHT_BRIDGE for i in indices},
        n_points=len(ids),
    )
    ip.annotate(graph, found)
    return graph


def _straight(n=40, radius=300.0):
    coords = np.column_stack([STEP * np.arange(float(n)), np.zeros(n), np.zeros(n)])
    return coords, np.full(n, float(radius))


# ------------------------------------------------------------------ radius repair
def test_an_invented_low_radius_is_not_mistaken_for_a_collapse():
    """It is not a collapsed vessel; there is no vessel there to have collapsed."""
    coords, radii = _straight()
    radii[18:24] = 60.0  # far below the segment's own trend

    unflagged = _segment(coords, radii)
    assert rr.find_outlier_spans(unflagged), "the fixture must look like a collapse"

    flagged = _flag(_segment(coords, radii), range(18, 24))
    assert rr.find_outlier_spans(flagged) == []


def test_a_taper_is_never_extrapolated_from_an_invented_radius():
    arc = STEP * np.arange(40.0)
    radii = np.full(40, 300.0)
    radii[10:20] = 45.0   # the invented fill, wildly off the vessel's calibre
    radii[20:26] = 120.0  # the genuine collapse to be repaired

    exclude = np.zeros(40, dtype=bool)
    exclude[10:20] = True

    naive, _ = rr.taper_fill(arc, radii, 20, 25, min_healthy=4)
    careful, info = rr.taper_fill(arc, radii, 20, 25, min_healthy=4, exclude=exclude)

    assert info["reason"] == ""
    # With the fill admitted as evidence the proximal side is mostly 45 um, and the
    # repair lands nowhere near the 300 um vessel it is supposed to continue.
    assert naive[22] < 200.0
    assert careful[22] == pytest.approx(300.0, rel=0.2)


# --------------------------------------------------------------------- super metric
def test_invented_length_does_not_count_towards_graph_volume():
    coords, radii = _straight(n=21)
    graph = _segment(coords, radii)
    whole = sm.graph_volume(graph)

    _flag(graph, range(5, 16))
    part = sm.graph_volume(graph)

    # 20 spans, of which the 12 touching an invented point are dropped.
    assert part == pytest.approx(whole * 8 / 20, rel=1e-9)


def test_invented_spans_are_not_rasterised_as_centreline():
    from .conftest_geometry import make_frame

    frame = make_frame((40, 40, 80))
    coords = np.column_stack([
        np.linspace(100.0, 600.0, 21), np.full(21, 200.0), np.full(21, 200.0)
    ])
    graph = _segment(coords, np.full(21, 30.0))
    whole = len(sm.rasterise_centreline(graph, frame))

    _flag(graph, range(5, 16))
    part = len(sm.rasterise_centreline(graph, frame))
    assert 0 < part < whole


# ---------------------------------------------------------------------- smoothing
def test_smoothing_is_not_dragged_towards_an_invented_straight_line():
    """The artefact must not reshape the data it is excluded from.

    A Gaussian window wide enough to span the fill would average the real, bowed
    centreline together with the straight invented one, pulling it flat.
    """
    n = 31
    x = STEP * np.arange(float(n))
    y = np.where((np.arange(n) >= 12) & (np.arange(n) < 19), 0.0, 400.0)
    coords = np.column_stack([x, y, np.zeros(n)])

    naive = _segment(coords, np.full(n, 300.0))
    so.smooth_centreline(naive, window_um=600.0)
    pulled = naive.coords(0)[11, 1]

    careful = _flag(_segment(coords, np.full(n, 300.0)), range(12, 19))
    so.smooth_centreline(careful, window_um=600.0)
    held = careful.coords(0)[11, 1]

    # The pull is bounded by `_clamp_moves`, so what matters is that it happens at all.
    assert pulled < 380.0, "without the flag the fill drags its neighbour flat"
    assert held == pytest.approx(400.0, abs=1e-9)
    # ...and the invented points themselves are left exactly where they were.
    assert np.allclose(careful.coords(0)[12:19], coords[12:19])


# ------------------------------------------------------------------------- gaps
def test_a_gap_beside_an_invented_fill_is_not_filled_again():
    pytest.importorskip("coronary_sdf")
    from hipct_seg_debug.edit.reconnect import gaps as gaps_mod

    n = 30
    coords = np.column_stack([STEP * np.arange(float(n)), np.zeros(n), np.zeros(n)])
    coords[15:, 0] += 8000.0  # a jump inside the edge, far past any gap threshold
    graph = _segment(coords, np.full(n, 300.0))

    assert gaps_mod.find(graph), "the fixture must look like a fillable gap"
    _flag(graph, range(15, 20))
    assert gaps_mod.find(graph) == []


# --------------------------------------------------------------- radius perimeter
def test_an_invented_point_keeps_its_input_radius_and_says_so():
    pytest.importorskip("cv2")
    from hipct_seg_debug.edit import radius_perimeter as rp

    from .conftest_geometry import SPACING, axis_graph, cylinder, make_frame

    shape = (40, 40, 80)
    frame = make_frame(shape)
    mask = cylinder(shape, 6, 5, 75)

    seed = 6 * SPACING
    graph = axis_graph(frame, 8, 72, seed, cy=20, cz=20)
    ids = graph.segment(0)["point_ids"]
    # Give the invented run a radius no measurement of this cylinder would produce.
    with graph.batch("fixture"):
        for i in range(20, 30):
            graph.set_radius(ids[i], 12.5)
    _flag(graph, range(20, 30))

    result = rp.measure_radii(graph, frame, mask)
    radii = result.radii[0]
    reject = result.reject_reason[0]
    modes = result.resolution_mode[0]

    assert np.all(reject[20:30] == rp.INTERPOLATED_INPUT)
    assert np.all(modes[20:30] == rp.INPUT_FALLBACK)
    assert np.allclose(radii[20:30], 12.5), "the input radius must survive untouched"
    # ...while the real points either side are still measured against the image.
    assert np.median(radii[5:15]) == pytest.approx(seed, rel=0.2)
    assert result.rejection_counts()[rp.INTERPOLATED_INPUT] == 10
