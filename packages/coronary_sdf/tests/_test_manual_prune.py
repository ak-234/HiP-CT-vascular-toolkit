"""Smoke tests for manual_prune (run: python tests/_test_manual_prune.py).

Covers the display-free path: scalar colouring, apply_manual_prune's new
removed_seg_ids field, the Amira writer round-trip, and the removal log writers.
The interactive picker (run_prune_picker / _pick_one_tree) needs a display and is
exercised by the existing epicardial off-screen pick test.
"""
from __future__ import annotations

import tempfile
from pathlib import Path

from coronary_sdf.parse_amira import parse_xml
from coronary_sdf.epicardial_annotation import apply_manual_prune, write_amira_xml
from coronary_sdf.manual_prune import build_segment_colors, write_removal_logs


def _fixture():
    # trunk (r=2) -> bif node 1; main daughter (r=1), side1 (r=0.3), side2 (r=0.15)
    nodes = {
        0: (0.0, 0.0, 0.0, 1),
        1: (0.0, 0.0, 10000.0, 4),
        2: (0.0, 0.0, 20000.0, 1),
        3: (5000.0, 0.0, 12000.0, 1),
        4: (-5000.0, 0.0, 12000.0, 1),
    }
    points = {
        0: (0, 0, 0, 2000.0), 1: (0, 0, 5000.0, 2000.0), 2: (0, 0, 10000.0, 2000.0),
        10: (0, 0, 10000.0, 1000.0), 11: (0, 0, 15000.0, 1000.0), 12: (0, 0, 20000.0, 1000.0),
        20: (0, 0, 10000.0, 300.0), 21: (2500.0, 0, 11000.0, 300.0), 22: (5000.0, 0, 12000.0, 300.0),
        30: (0, 0, 10000.0, 150.0), 31: (-2500.0, 0, 11000.0, 150.0), 32: (-5000.0, 0, 12000.0, 150.0),
    }
    segments = [
        {"id": 0, "node1": 0, "node2": 1, "point_ids": [0, 1, 2], "strahler": 3},
        {"id": 1, "node1": 1, "node2": 2, "point_ids": [10, 11, 12], "strahler": 2},
        {"id": 2, "node1": 1, "node2": 3, "point_ids": [20, 21, 22], "strahler": 1},
        {"id": 3, "node1": 1, "node2": 4, "point_ids": [30, 31, 32], "strahler": 1},
    ]
    vessel_points = {"MAIN": {0, 1, 2, 10, 11, 12}}
    vessels_idx = {"MAIN": {0, 1}}
    return nodes, points, segments, vessel_points, vessels_idx


def test_segment_colors() -> None:
    nodes, points, segments, _vp, _vi = _fixture()
    for mode in ("strahler", "radius", "segment"):
        fn, legend, eff = build_segment_colors(segments, points, nodes, mode)
        assert eff == mode, (mode, eff)
        for i in range(len(segments)):
            c = fn(i)
            assert len(c) == 3 and all(0.0 <= x <= 1.0 for x in c), (mode, i, c)
        assert legend and all(len(e) == 2 for e in legend), (mode, legend)
    # Missing Strahler on any segment -> graceful fallback to segment-index colour.
    segs_missing = [dict(s) for s in segments]
    segs_missing[0].pop("strahler")
    _fn, _lg, eff = build_segment_colors(segs_missing, points, nodes, "strahler")
    assert eff == "segment", eff
    print("  [1] scalar colouring (strahler/radius/segment + fallback): PASS")


def test_prune_write_log() -> None:
    nodes, points, segments, vessel_points, vessels_idx = _fixture()
    base_n = len(segments)
    id_to_strahler = {int(s["id"]): s.get("strahler") for s in segments}

    # Remove side1 (seg idx 2). Main vessels + the unselected side branch survive.
    p_nodes, p_points, p_segs, removed = apply_manual_prune(
        nodes, points, segments, vessel_points, {2})

    assert len(removed) == 1, removed
    rec = removed[0]
    assert rec["seg_id"] == 2 and rec["n_subtree_removed"] == 1, rec
    assert rec["removed_seg_ids"] == [2], rec["removed_seg_ids"]
    kept_pids = {p for s in p_segs for p in s["point_ids"]}
    assert {0, 1, 2, 10, 11, 12} <= kept_pids, "MAIN must survive"
    assert not ({20, 21, 22} & kept_pids), "side1 must be gone"
    assert {30, 31, 32} <= kept_pids, "side2 (not selected) must remain"

    with tempfile.TemporaryDirectory() as td:
        out = Path(td)
        out_xml = out / "pruned.am.xml"
        write_amira_xml(p_nodes, p_points, p_segs, out_xml)
        _n2, _p2, s2 = parse_xml(out_xml)                 # round-trips
        assert len(s2) == len(p_segs), (len(s2), len(p_segs))

        csv_path, log_path = write_removal_logs(
            out, out / "in.am.xml", "strahler", vessels_idx, base_n, len(p_segs),
            removed, id_to_strahler, out_xml)
        csv_txt = csv_path.read_text()
        log_txt = log_path.read_text()
        assert "takeoff_seg_id" in csv_txt and "\n2," in ("\n" + csv_txt), csv_txt
        assert "seg_id=2" in log_txt and "MAIN: 2 segment(s)" in log_txt, log_txt
        assert "segments removed (subtrees):      1" in log_txt, log_txt
    print("  [2] prune + writer round-trip + removal logs: PASS")


if __name__ == "__main__":
    test_segment_colors()
    test_prune_write_log()
    print("\nAll manual_prune smoke tests passed.")
