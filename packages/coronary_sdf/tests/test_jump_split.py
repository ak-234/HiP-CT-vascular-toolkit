"""Cutting an unsampled jump rather than bridging it, and dropping what strands.

On LADAF-2024-28 the segmentation is in 55 disconnected components and the graph
crosses between them 24 times, 72.4 mm in total, with no points written across any
of it. Bridging draws lumen through proven background; these two passes are the
alternative, and what is defended here is that they fire on exactly those spans and
leave ordinary vessel alone.
"""

from __future__ import annotations

import numpy as np

from coronary_sdf.centreline_reconnection import (
    drop_small_components,
    find_connected_components,
    split_unsampled_jumps,
)

STEP = 100.0  # um between points, as in the resampled graphs
R = 120.0     # um radius, so 5 * (r_i + r_j) = 1200 um is the ratio gate


def _line(x0: float, n: int, pid0: int, r: float = R):
    return {pid0 + k: (x0 + STEP * k, 0.0, 0.0, r) for k in range(n)}


def _graph(points: dict, seg_pids: list[list[int]]):
    nodes, segments = {}, []
    for i, pids in enumerate(seg_pids):
        nodes[2 * i] = (*points[pids[0]][:3], 0)
        nodes[2 * i + 1] = (*points[pids[-1]][:3], 0)
        segments.append(dict(id=i, node1=2 * i, node2=2 * i + 1,
                             point_ids=list(pids)))
    return nodes, segments


def test_a_pointless_jump_becomes_two_segments():
    points = {**_line(0.0, 10, 0), **_line(9000.0, 10, 100)}
    nodes, segments = _graph(points, [list(range(10)) + list(range(100, 110))])

    nodes, out, n_cut, records = split_unsampled_jumps(
        nodes, points, segments, big_jump_ratio=5.0, min_gap_um=500.0)

    assert n_cut == 1
    assert len(out) == 2
    assert [len(s["point_ids"]) for s in out] == [10, 10]
    # The halves must be topologically separate, or nothing downstream changes.
    assert len(find_connected_components(out)) == 2
    assert records[0]["gap_mm"] > 8.0


def test_ordinary_vessel_is_not_cut():
    """Both gates must hold: a long step in a wide vessel is sampling, not a break."""
    points = _line(0.0, 20, 0)
    nodes, segments = _graph(points, [list(range(20))])
    _n, out, n_cut, _r = split_unsampled_jumps(
        nodes, points, segments, big_jump_ratio=5.0, min_gap_um=500.0)
    assert n_cut == 0 and len(out) == 1

    # A 900 um step clears the absolute floor but not 5 * (r_i + r_j) on a 2 mm
    # vessel, which is under-sampling rather than a bridge.
    wide = {k: (v[0], v[1], v[2], 2000.0) for k, v in _line(0.0, 5, 0).items()}
    wide[5] = (900.0 + 4 * STEP, 0.0, 0.0, 2000.0)
    nodes2, segments2 = _graph(wide, [list(range(6))])
    _n2, out2, n_cut2, _r2 = split_unsampled_jumps(
        nodes2, wide, segments2, big_jump_ratio=5.0, min_gap_um=500.0)
    assert n_cut2 == 0 and len(out2) == 1


def test_the_stranded_fragment_is_dropped_and_the_tree_kept():
    points = {**_line(0.0, 60, 0), **_line(9000.0, 5, 100)}
    nodes, segments = _graph(points, [list(range(60)) + list(range(100, 105))])
    nodes, out, _n, _r = split_unsampled_jumps(
        nodes, points, segments, big_jump_ratio=5.0, min_gap_um=500.0)
    assert len(find_connected_components(out)) == 2

    # 60 points at 100 um is 5.9 mm; 5 points is 0.4 mm.
    nodes, kept, dropped = drop_small_components(
        nodes, points, out, min_length_mm=5.0, verbose=False)

    assert len(kept) == 1 and len(dropped) == 1
    assert dropped[0]["length_mm"] < 1.0
    assert len(kept[0]["point_ids"]) == 60
    # Nodes belonging only to the dropped piece must go with it.
    live = {s["node1"] for s in kept} | {s["node2"] for s in kept}
    assert set(nodes) == live


def test_the_fraction_floor_separates_trees_from_fragments():
    """The criterion that transfers between hearts: relative, not absolute.

    Two trees and one stranded fragment, in the proportions measured on
    LADAF-2024-28 (100%, 78%, 10% of the largest). A millimetre floor tuned on
    this heart would not survive a bigger or smaller one; the fraction does.
    """
    points, seg_pids, pid = {}, [], 0
    for n in (100, 78, 10):
        points.update(_line(1e6 * len(seg_pids), n, pid))
        seg_pids.append(list(range(pid, pid + n)))
        pid += n
    nodes, segments = _graph(points, seg_pids)

    nodes_out, kept, dropped = drop_small_components(
        nodes, points, segments, min_fraction_of_largest=0.25, verbose=False)

    assert len(kept) == 2 and len(dropped) == 1
    assert [len(s["point_ids"]) for s in kept] == [100, 78]

    # ...and with the floors off, nothing is dropped at all.
    _n, kept_all, none_dropped = drop_small_components(
        nodes, points, segments, verbose=False)
    assert len(kept_all) == 3 and not none_dropped


def test_the_spacing_gate_catches_a_break_in_a_thin_vessel():
    """The width gate is radius-scaled, so a narrow vessel's break is a small step.

    140 um vessel, 0.62 mm jump: `5 * (r_i + r_j)` is 1.4 mm, so the width gate
    alone declines it, while the step is 6x the segment's own 100 um spacing.
    Six of the 24 mask-confirmed breaks on LADAF-2024-28 look exactly like this.
    """
    thin = {k: (v[0], v[1], v[2], 140.0) for k, v in _line(0.0, 10, 0).items()}
    thin.update({100 + k: (620.0 + 9 * STEP + STEP * k, 0.0, 0.0, 140.0)
                 for k in range(10)})
    nodes, segments = _graph(thin, [list(range(10)) + list(range(100, 110))])

    _n, width_only, n_width, _r = split_unsampled_jumps(
        nodes, thin, segments, big_jump_ratio=5.0, min_gap_um=500.0,
        step_ratio=0.0)
    assert n_width == 0 and len(width_only) == 1

    _n2, both, n_both, _r2 = split_unsampled_jumps(
        nodes, thin, segments, big_jump_ratio=5.0, min_gap_um=500.0,
        step_ratio=5.0)
    assert n_both == 1 and len(both) == 2
