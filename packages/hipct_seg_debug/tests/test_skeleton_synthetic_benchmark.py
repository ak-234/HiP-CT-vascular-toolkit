"""The extraction benchmark's scoring, checked on skeletons whose errors are known."""
import numpy as np
import pytest

from hipct_seg_debug.edit import skeleton_phantoms as sp
from hipct_seg_debug.edit import skeleton_synthetic_benchmark as bench
from hipct_seg_debug.edit.skeletonisers import edges_to_triple


@pytest.fixture(scope="module")
def round_phantom():
    p = sp.build("round")
    return p, bench.phantom_frame(p.mask.shape, p.spacing_um)


def _triple(frame, *polylines_zyx):
    """A triple through the given ``zyx`` voxel polylines (each its own chain)."""
    verts, edges, off = [], [], 0
    for pl in polylines_zyx:
        pl = np.asarray(pl, dtype=float)
        verts.append(frame.seg_to_um(pl[:, ::-1]))
        edges += [(off + i, off + i + 1) for i in range(len(pl) - 1)]
        off += len(pl)
    v = np.vstack(verts)
    return edges_to_triple(v, np.array(edges), np.full(len(v), 50.0))


def _truth(p):
    return np.vstack([b.points for b in p.branches])


def test_the_truth_scores_perfectly(round_phantom):
    p, frame = round_phantom
    s = bench.score_geometry(p, _triple(frame, _truth(p)), frame)
    assert s["error_median_vox"] < 0.05 and s["coverage"] == 1.0
    assert s["false_branches"] == 0 and s["missing_branches"] == 0
    assert s["ladder_fraction"] == 0.0 and s["beta1"] == 0 and s["components"] == 1


def test_a_spur_is_one_false_branch(round_phantom):
    p, frame = round_phantom
    truth = _truth(p)
    mid = truth[len(truth) // 2]
    spur = np.array([mid, mid + [0, 4, 0]])
    # Share the junction vertex so the spur attaches rather than floating.
    t = _triple(frame, truth[: len(truth) // 2 + 1], truth[len(truth) // 2:], spur)
    t2 = bench.score_geometry(p, _merge_at(t), frame)
    assert t2["false_branches"] == 1 and t2["missing_branches"] == 0


def _merge_at(triple):
    """Weld coincident nodes (the test polylines share endpoints by construction)."""
    from hipct_seg_debug.edit.adapter import Triple

    keyed, remap = {}, {}
    for nid, n in triple.nodes.items():
        key = tuple(np.round(n[:3], 6))
        remap[nid] = keyed.setdefault(key, nid)
    segs = [dict(s, node1=remap[s["node1"]], node2=remap[s["node2"]]) for s in triple.segments]
    deg = {}
    for s in segs:
        for k in ("node1", "node2"):
            deg[s[k]] = deg.get(s[k], 0) + 1
    nodes = {nid: (*triple.nodes[nid][:3], deg.get(nid, 0)) for nid in set(remap.values())}
    return Triple(nodes, triple.points, segs)


def test_a_parallel_duplicate_is_a_ladder(round_phantom):
    p, frame = round_phantom
    truth = _truth(p)
    twin = truth + [0, 2, 0]
    s = bench.score_geometry(p, _triple(frame, truth, twin), frame)
    assert s["ladder_fraction"] > 0.9
    assert s["components"] == 2


def test_an_offset_skeleton_reports_its_offset(round_phantom):
    p, frame = round_phantom
    s = bench.score_geometry(p, _triple(frame, _truth(p) + [0, 2, 0]), frame)
    assert s["error_median_vox"] == pytest.approx(2.0, abs=0.05)


def test_smoke_run_scores_and_radii(round_phantom):
    rows = bench.run(["round"], ["geodesic-scoop"], radii=True, verbose=False)
    (row,) = rows
    assert row["status"] == "evaluated"
    assert row["false_branches"] == 0 and row["missing_branches"] == 0
    r = row["radii"]["by_lambda"]["1"]
    assert r["r_inflated"] == pytest.approx(1.0, abs=0.12)
    assert r["r_dt"] == pytest.approx(1.0, abs=0.12)
    md = bench.summary_markdown(rows)
    assert "geodesic-scoop" in md


def test_teasar_smoke_or_skip():
    pytest.importorskip("kimimaro")
    rows = bench.run(["round"], ["teasar-readme[scale=4;const_um=50]"], radii=False,
                     verbose=False)
    assert rows[0]["status"] == "evaluated" and rows[0]["missing_branches"] == 0


def test_jin_refuses_the_loop_and_the_row_says_so():
    rows = bench.run(["loop"], ["jin-mcp"], radii=False, verbose=False)
    assert rows[0]["status"] == "refused" and "Euler" in rows[0]["error"]


def test_default_candidates_cover_every_family():
    names = {c.split("[")[0] for c in bench.default_candidates()}
    assert {"teasar", "teasar-readme", "jin-mcp", "jin-mgf", "geodesic-scoop"} <= names
    assert "lee" not in names
