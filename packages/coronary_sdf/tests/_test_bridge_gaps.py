"""Smoke tests for smoothing.bridge_centerline_gaps
(run: python -m coronary_sdf._test_bridge_gaps).

Covers a mid-segment point-less gap and a start gap (i=0), checking that interior
points are inserted at ~target spacing, all original points/endpoints survive, the
bridge is C1 (leaves the incoming side along the local tangent), the fill is curved
(not straight) when the two sides pull in different directions, and radius is
monotone across the span. Also confirms thin-vessel spacing is left untouched.
"""
from __future__ import annotations

import numpy as np

from coronary_sdf.smoothing import bridge_centerline_gaps

UM = 1000.0  # mm -> um


def _run(points, seg):
    """Bridge one segment in place; returns (out_points, n_bridged, records)."""
    return bridge_centerline_gaps(
        dict(points), [seg],
        target_spacing_mm=0.2, big_jump_ratio=5.0, min_gap_um=500.0,
        radius_scale=1.0, verbose=False,
    )


def test_mid_gap_curved_and_spaced() -> None:
    # Left cluster runs +x with a +y drift; a 6 mm jump in +x; right cluster runs
    # +x with a -y drift -> the two tangents differ, so a C1 Hermite bridge must
    # bow off the straight chord (curved), not lie on it.
    r = 200.0
    left = [(0.0, 0.0), (100.0, 20.0), (200.0, 40.0), (300.0, 60.0)]
    right = [(6300.0, -60.0), (6400.0, -40.0), (6500.0, -20.0), (6600.0, 0.0)]
    pts = {}
    ids = []
    for pid, (x, y) in enumerate(left + right):
        pts[pid] = (x, y, 0.0, r)
        ids.append(pid)
    seg = {"id": 0, "node1": 0, "node2": 1, "point_ids": list(ids)}

    out_pts, n_bridged, recs = _run(pts, seg)
    new_ids = seg["point_ids"]

    assert n_bridged == 1, n_bridged
    assert recs[0]["seg_id"] == 0 and recs[0]["i"] == 3, recs
    # interior count = ceil(gap_mm / target_spacing) - 1
    n_ins = recs[0]["n_inserted"]
    assert n_ins == int(np.ceil(recs[0]["gap_mm"] / 0.2)) - 1, recs[0]
    assert new_ids[0] == ids[0] and new_ids[-1] == ids[-1]
    assert set(ids) <= set(new_ids), "original points must all survive"
    assert len(new_ids) == len(ids) + n_ins, len(new_ids)

    coords = np.array([out_pts[p][:3] for p in new_ids])
    radii = np.array([out_pts[p][3] for p in new_ids])

    steps_mm = np.linalg.norm(np.diff(coords, axis=0), axis=1) / UM
    assert steps_mm.max() < 0.35, steps_mm.max()          # ~target spacing
    assert np.allclose(radii, r, atol=1e-6)               # constant radius here

    # C1 at the incoming side: first inserted point continues the left tangent.
    pa = np.array(out_pts[ids[3]][:3])
    local_tan = pa - np.array(out_pts[ids[2]][:3])
    local_tan /= np.linalg.norm(local_tan)
    first_bridge = np.array(out_pts[new_ids[4]][:3])       # index 4 = first inserted
    bridge_dir = first_bridge - pa
    bridge_dir /= np.linalg.norm(bridge_dir)
    assert float(np.dot(local_tan, bridge_dir)) > 0.9, np.dot(local_tan, bridge_dir)

    # Curved, not straight: interior deviates from the straight chord.
    pb = np.array(out_pts[ids[4]][:3])
    chord = pb - pa
    chord_hat = chord / np.linalg.norm(chord)
    interior = coords[4:4 + n_ins]
    rel = interior - pa
    perp = rel - np.outer(rel @ chord_hat, chord_hat)
    max_dev_mm = np.linalg.norm(perp, axis=1).max() / UM
    assert max_dev_mm > 0.05, f"bridge should bow off the chord, got {max_dev_mm:.3f}mm"
    print(f"  [1] mid-gap: {n_ins} pts, max bow {max_dev_mm:.3f}mm, C1 ok: PASS")


def test_start_gap_and_radius_taper() -> None:
    # Gap at i=0 (node -> far cluster) with tapering radius; start-side tangent
    # falls back to the chord, end side uses the cluster tangent.
    pts = {
        0: (0.0, 0.0, 0.0, 400.0),        # node endpoint, r=0.4mm
        1: (5000.0, 0.0, 0.0, 200.0),     # 5mm jump, r=0.2mm
        2: (5100.0, 50.0, 0.0, 200.0),
        3: (5200.0, 100.0, 0.0, 200.0),
    }
    seg = {"id": 7, "node1": 0, "node2": 1, "point_ids": [0, 1, 2, 3]}
    out_pts, n_bridged, recs = _run(pts, seg)
    new_ids = seg["point_ids"]

    assert n_bridged == 1 and recs[0]["i"] == 0, recs
    assert new_ids[0] == 0 and new_ids[-1] == 3 and 1 in new_ids
    assert new_ids.count(1) == 1

    # radius monotonically decreases across the bridge from 0.4 -> 0.2 mm.
    span_r = np.array([out_pts[p][3] for p in new_ids[: new_ids.index(1) + 1]])
    assert span_r[0] == 400.0 and span_r[-1] == 200.0
    assert np.all(np.diff(span_r) <= 1e-6), span_r
    print(f"  [2] start-gap: inserted {recs[0]['n_inserted']} pts, radius taper ok: PASS")


def test_thin_vessel_not_bridged() -> None:
    # 0.4mm spacing on a 0.05mm vessel: step 400um < 500um floor -> not bridged.
    pts = {i: (i * 400.0, 0.0, 0.0, 50.0) for i in range(6)}
    seg = {"id": 9, "node1": 0, "node2": 1, "point_ids": list(range(6))}
    _out, n_bridged, recs = _run(pts, seg)
    assert n_bridged == 0, (n_bridged, recs)
    assert seg["point_ids"] == list(range(6))
    print("  [3] thin vessel (step<floor) left untouched: PASS")


if __name__ == "__main__":
    test_mid_gap_curved_and_spaced()
    test_start_gap_and_radius_taper()
    test_thin_vessel_not_bridged()
    print("\nAll bridge_centerline_gaps smoke tests passed.")
