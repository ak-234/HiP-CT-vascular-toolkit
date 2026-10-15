"""The multi-algorithm front end.

The paper's finding is that algorithm choice alone can double a network's node count,
so the value of this module is that every backend hands back the *same* structure and
can therefore be scored on the same terms. These tests pin that contract; the TEASAR
backend needs ``kimimaro``, which is optional, so it is skipped when absent.
"""

from __future__ import annotations

import numpy as np
import pytest

from hipct_seg_debug.edit import skeletonisers as sk
from hipct_seg_debug.edit import supermetric as sm

from .conftest_geometry import cylinder, make_frame

SHAPE = (30, 30, 60)


@pytest.fixture
def frame():
    return make_frame(SHAPE)


@pytest.fixture
def tube():
    return cylinder(SHAPE, 4, 5, 55)


# --------------------------------------------------------------- edges_to_triple


def test_a_y_becomes_three_segments_and_one_junction():
    """Nodes are the vertices of degree != 2; the chains between them are segments."""
    verts = np.array(
        [[0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0],
         [4, 1, 0], [5, 2, 0], [4, -1, 0], [5, -2, 0]], dtype=float
    ) * 100.0
    edges = np.array([[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [3, 6], [6, 7]])

    triple = sk.edges_to_triple(verts, edges, np.full(len(verts), 50.0))

    assert len(triple.segments) == 3
    assert sorted(n[3] for n in triple.nodes.values()) == [1, 1, 1, 3]
    assert sum(len(s["point_ids"]) for s in triple.segments) == 10


def test_a_ring_with_no_junction_survives_as_a_self_loop():
    """Dropping it would lose a whole cycle silently."""
    angles = np.linspace(0.0, 2 * np.pi, 9)[:-1]
    verts = np.stack([np.cos(angles), np.sin(angles), np.zeros(8)], axis=1) * 100.0
    edges = np.array([[i, (i + 1) % 8] for i in range(8)])

    triple = sk.edges_to_triple(verts, edges, np.full(8, 20.0))

    assert len(triple.segments) == 1
    seg = triple.segments[0]
    assert seg["node1"] == seg["node2"], "closed on itself"


def test_the_origin_offset_is_applied():
    verts = np.array([[0.0, 0.0, 0.0], [100.0, 0.0, 0.0], [200.0, 0.0, 0.0]])
    edges = np.array([[0, 1], [1, 2]])
    triple = sk.edges_to_triple(verts, edges, np.full(3, 10.0), origin_um=(1000, 20, 3))
    assert triple.nodes[0][:3] == (1000.0, 20.0, 3.0)


# ------------------------------------------------------------------- backends


def test_lee_produces_a_scoreable_skeleton(frame, tube):
    from hipct_seg_debug.edit.graphmodel import EditableGraph

    cand = sk.skeletonise("lee", tube, frame)

    assert cand.name == "lee"
    assert len(cand.triple.segments) >= 1
    graph = EditableGraph(cand.triple)
    assert sm.cl_sensitivity(graph, frame, tube) > 0.9, "a thinned axis is inside"


def test_the_frame_describes_the_volume_with_no_stride_applied_twice(frame, tube):
    """A decimated volume's stride belongs in its frame, and nowhere else.

    Applying it here as well put every coordinate eight times too far out at stride 8:
    the skeleton derived *from* the mask scored a cl-sensitivity of exactly zero
    against it, and a network volume 600 times too large. Both are the shape of error
    that looks like a bad algorithm rather than a bad conversion.
    """
    import inspect

    from hipct_seg_debug.edit.graphmodel import EditableGraph

    assert "stride" not in inspect.signature(sk.skeletonise).parameters

    cand = sk.skeletonise("lee", tube, frame)
    coords = np.vstack([
        EditableGraph(cand.triple).coords(sid)
        for sid in EditableGraph(cand.triple).segment_ids()
    ])
    lo, hi = frame.seg_bbox_um[0::2], frame.seg_bbox_um[1::2]
    assert np.all(coords >= lo - 1.0) and np.all(coords <= hi + 1.0), \
        "every centreline point lies inside the lattice it came from"


def test_an_unknown_algorithm_is_refused(frame, tube):
    with pytest.raises(ValueError, match="unknown skeletoniser"):
        sk.skeletonise("autoskeleton", tube, frame)


def test_amira_without_a_path_says_what_is_missing(frame, tube):
    with pytest.raises(ValueError, match="--amira-graph"):
        sk.skeletonise("amira", tube, frame)


def test_teasar_traces_a_tube_or_says_why_it_cannot(frame, tube):
    """`kimimaro` is installed as of 2026-08-17, so this now exercises the real backend.

    The ImportError branch is kept rather than deleted: it is the path a fresh checkout
    takes, and it is the only thing asserting the message tells you what to install.
    It applies only when kimimaro is genuinely absent. Installed but unimportable --
    the osteoid 0.7.1 wheel lacks a module its own skeleton.py imports -- means a
    broken pin, and passing through this branch hid exactly that.
    """
    import importlib.metadata

    try:
        import kimimaro  # noqa: F401
    except ImportError as exc:
        try:
            installed = importlib.metadata.version("kimimaro")
        except importlib.metadata.PackageNotFoundError:
            installed = None
        if installed is not None:
            pytest.fail(f"kimimaro {installed} is installed but does not import ({exc}); "
                        "check the pinned osteoid in hipct_seg_debug's pyproject.toml")
        with pytest.raises(ImportError, match="pip install kimimaro"):
            sk.skeletonise("teasar", tube, frame)
        return
    cand = sk.skeletonise("teasar", tube, frame)
    assert len(cand.triple.segments) >= 1


# ------------------------------------------------------------ candidate specs


def test_parse_candidate_reads_a_parameter_grid_point():
    name, params = sk.parse_candidate("teasar-readme[scale=4;const-um=100;postprocess=false]")
    assert name == "teasar-readme"
    assert params == {"scale": 4, "const_um": 100, "postprocess": False}
    assert sk.parse_candidate("lee") == ("lee", {})
    assert sk.candidate_label(name, params) == "teasar-readme_scale-4_const_um-100_postprocess-False"


def test_split_candidates_respects_brackets():
    assert sk.split_candidates("a, b[x=1;y=2] ,c") == ["a", "b[x=1;y=2]", "c"]


@pytest.mark.parametrize("bad", ["teasar[scale=4", "teasar[scale]"])
def test_malformed_candidate_is_refused(bad):
    with pytest.raises(ValueError):
        sk.parse_candidate(bad)


def test_new_backends_are_registered():
    for name in ("teasar-readme", "geodesic-scoop", "jin-mgf"):
        assert name in sk.ALGORITHMS and name in sk._BACKENDS


def test_point_attrs_follow_their_vertices():
    verts = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 1, 0], [3, -1, 0]], float) * 100
    edges = np.array([[0, 1], [1, 2], [2, 3], [2, 4]])
    area = np.array([10.0, 11.0, 12.0, 13.0, 14.0])
    t = sk.edges_to_triple(verts, edges, np.full(5, 5.0), point_attrs={"a": area})
    values = t.point_attrs["a"]
    assert set(values) == set(t.points)
    for pid, (x, y, _z, _r) in t.points.items():
        vi = int(np.argmin(np.linalg.norm(verts[:, :2] - [x, y], axis=1)))
        assert values[pid] == area[vi]
    with pytest.raises(ValueError, match="values for"):
        sk.edges_to_triple(verts, edges, np.full(5, 5.0), point_attrs={"a": area[:3]})


def test_skeletonise_all_runs_a_grid_and_writes_a_summary(monkeypatch, tmp_path, frame, tube):
    import json

    from hipct_seg_debug.edit import __main__ as cli

    monkeypatch.setattr(cli, "_decoded", lambda _: (tube, frame, None, "fake.am"))
    monkeypatch.setattr(cli, "_seg_materials", lambda _: None)
    summary = tmp_path / "summary.json"
    args = cli.build_parser().parse_args([
        "skeletonise-all", "--algorithms",
        "geodesic-scoop,geodesic-scoop[min_branch_radii=4]", "--jin-root-zyx", "15", "15", "6",
        "--no-score", "--out-dir", str(tmp_path), "--summary-json", str(summary)])
    assert cli.cmd_skeletonise_all(args) == 0
    assert (tmp_path / "geodesic-scoop.am").exists()
    assert (tmp_path / "geodesic-scoop_min_branch_radii-4.am").exists()
    rows = json.loads(summary.read_text())["candidates"]
    assert [r["candidate"] for r in rows] == ["geodesic-scoop", "geodesic-scoop_min_branch_radii-4"]
    assert rows[1]["params"]["min_branch_radii"] == 4
    assert rows[0]["params"]["root_zyx"] == [15, 15, 6]
    assert rows[0]["terminals"] == 2 and "roughness" in rows[0]


def test_split_and_tree_context_are_reused_only_inside_the_block(monkeypatch, frame):
    """skeletonise-all splits once and scores each tree's context once, not per candidate."""
    from hipct_seg_debug.edit import components as comp
    from hipct_seg_debug.edit import supermetric

    from .conftest_geometry import two_cylinders

    vol = two_cylinders(SHAPE)
    calls = {"split": 0, "refs": 0}
    real_split, real_refs = comp._split_components, supermetric.reference_bifurcations

    def counting_split(*a, **k):
        calls["split"] += 1
        return real_split(*a, **k)

    def counting_refs(*a, **k):
        calls["refs"] += 1
        return real_refs(*a, **k)

    monkeypatch.setattr(comp, "_split_components", counting_split)
    monkeypatch.setattr(supermetric, "reference_bifurcations", counting_refs)
    comp.split_components(vol, frame)
    comp.split_components(vol, frame)
    assert calls["split"] == 2, "uncached by default: a caller may edit the mask"

    cand = sk.skeletonise_per_component("geodesic-scoop", vol, frame)
    with comp.reuse_splits(), supermetric.reuse_tree_context():
        parts, stats = comp.split_components(vol, frame)
        again, _ = comp.split_components(vol, frame)
        assert again is parts and calls["split"] == 4  # 2 above, 1 in skeletonise_per_component
        from hipct_seg_debug.edit.graphmodel import EditableGraph
        for _ in range(2):
            supermetric.super_metric_per_tree(EditableGraph(cand.triple), frame,
                                              stats.labels, parts)
    assert calls["refs"] == len(parts) == 2
    comp.split_components(vol, frame)
    assert calls["split"] == 5, "the cache ends with the block"


def test_parallel_per_tree_matches_the_serial_path(frame, tmp_path):
    """skeletonise_many runs (candidate, tree) pairs in processes; the result must not change."""
    from .conftest_geometry import two_cylinders

    vol = two_cylinders(SHAPE)
    specs = [("geodesic-scoop", "geodesic-scoop", {}),
             ("geodesic-scoop_min_branch_radii-4", "geodesic-scoop", {"min_branch_radii": 4})]
    many = sk.skeletonise_many(specs, vol, frame, workers=2, scratch_dir=tmp_path / "crops")
    for label, alg, params in specs:
        serial = sk.skeletonise_per_component(alg, vol, frame, **params)
        got = many[label]
        assert got.trees == serial.trees == 2
        assert got.triple.points == serial.triple.points
        assert [s["point_ids"] for s in got.triple.segments] == \
               [s["point_ids"] for s in serial.triple.segments]
    assert not list((tmp_path / "crops").glob("*.npy")), "crops are removed afterwards"


def test_parallel_reports_a_failing_backend_per_candidate(frame, tmp_path):
    from .conftest_geometry import two_cylinders

    many = sk.skeletonise_many([("bad", "geodesic-scoop", {"shell_um": 1.0})],
                               two_cylinders(SHAPE), frame, workers=2,
                               scratch_dir=tmp_path / "crops")
    assert isinstance(many["bad"], ValueError)


def test_parallel_reruns_jobs_from_a_broken_pool_on_fewer_workers(frame, tmp_path, monkeypatch):
    """A worker killed for memory breaks the pool; its jobs rerun narrower, not fail."""
    import concurrent.futures as cf
    from concurrent.futures.process import BrokenProcessPool

    from .conftest_geometry import two_cylinders

    widths = []

    class Pool:  # in-process; the widest pool breaks after its first job
        def __init__(self, max_workers, **_):
            self.width, self.done = max_workers, 0
            widths.append(max_workers)

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def submit(self, fn, *args):
            future = cf.Future()
            if self.width > 1 and self.done:
                future.set_exception(BrokenProcessPool("killed"))
            else:
                future.set_result(fn(*args))
            self.done += 1
            return future

    monkeypatch.setattr(cf, "ProcessPoolExecutor", Pool)
    vol = two_cylinders(SHAPE)
    specs = [("geodesic-scoop", "geodesic-scoop", {}),
             ("geodesic-scoop_min_branch_radii-4", "geodesic-scoop", {"min_branch_radii": 4})]
    many = sk.skeletonise_many(specs, vol, frame, workers=4, scratch_dir=tmp_path / "crops")
    assert widths[0] == 4 and widths[-1] == 1 and widths == sorted(widths, reverse=True)
    for label, alg, params in specs:
        serial = sk.skeletonise_per_component(alg, vol, frame, **params)
        assert many[label].trees == 2
        assert many[label].triple.points == serial.triple.points


def test_skeletonise_all_writes_the_others_when_one_parallel_candidate_fails(frame, monkeypatch,
                                                                             tmp_path):
    import json

    from hipct_seg_debug.edit import __main__ as cli
    from .conftest_geometry import two_cylinders

    vol = two_cylinders(SHAPE)

    def fake_many(specs, volume, frame_, **_):
        from concurrent.futures.process import BrokenProcessPool
        good = sk.skeletonise_per_component("geodesic-scoop", volume, frame_)
        return {specs[0][0]: BrokenProcessPool("killed"), specs[1][0]: good}

    monkeypatch.setattr(cli, "_decoded", lambda _: (vol, frame, None, "fake.am"))
    monkeypatch.setattr(cli, "_seg_materials", lambda _: None)
    monkeypatch.setattr(sk, "skeletonise_many", fake_many)
    summary = tmp_path / "summary.json"
    args = cli.build_parser().parse_args(
        ["skeletonise-all", "--algorithms", "teasar,geodesic-scoop", "--per-tree",
         "--workers", "2", "--no-score", "--out-dir", str(tmp_path),
         "--summary-json", str(summary), "--seg", "unused.am"])
    assert cli.cmd_skeletonise_all(args) == 0
    rows = {r["candidate"]: r for r in json.loads(summary.read_text())["candidates"]}
    assert rows["teasar"]["status"] == "failed"
    assert "BrokenProcessPool" in rows["teasar"]["error"]
    assert rows["geodesic-scoop"]["status"] == "written"
    assert (tmp_path / "geodesic-scoop.am").exists()


def test_skeletonise_all_refuses_an_unknown_or_malformed_candidate(monkeypatch, tmp_path):
    from hipct_seg_debug.edit import __main__ as cli

    for spec in ("autoskeleton", "teasar[scale=4"):
        args = cli.build_parser().parse_args(
            ["skeletonise-all", "--algorithms", spec, "--out-dir", str(tmp_path)])
        assert cli.cmd_skeletonise_all(args) == 2


def test_teasar_readme_runs_postprocess_and_carries_kimimaro_area(frame, tube):
    pytest.importorskip("kimimaro")
    cand = sk.skeletonise("teasar-readme", tube, frame)
    assert "postprocess" in cand.detail and "cross-sectional area" in cand.detail
    area = np.array(list(cand.triple.point_attrs[sk.KIMIMARO_AREA].values()))
    r_a = np.sqrt(np.nanmedian(area) / np.pi)
    assert r_a == pytest.approx(4 * 10.0, rel=0.15)


def test_kimimaro_area_survives_a_postprocess_merge():
    """At scale 2.5 TEASAR fragments this phantom and postprocess re-joins it.

    The merge came back tagged as voxel space, so every kimimaro area was 0.
    """
    pytest.importorskip("kimimaro")
    from hipct_seg_debug.edit import skeleton_phantoms as sp
    from hipct_seg_debug.edit.skeleton_synthetic_benchmark import phantom_frame

    p = sp.build("collapse-bands")
    cand = sk.skeletonise("teasar-readme", p.mask, phantom_frame(p.mask.shape, p.spacing_um),
                          scale=2.5, const_um=100)
    area = np.array(list(cand.triple.point_attrs[sk.KIMIMARO_AREA].values()))
    assert np.mean(area > 0) > 0.9


def test_plain_teasar_is_unchanged_by_the_readme_options(frame, tube):
    """No point attribute and the same vertices as before the options existed."""
    pytest.importorskip("kimimaro")
    a = sk.skeletonise("teasar", tube, frame)
    b = sk.skeletonise("teasar", tube, frame, postprocess=False, cross_section=False)
    assert not a.triple.point_attrs
    assert a.triple.points == b.triple.points


# ------------------------------------------------------------------ extraction roots


def test_root_for_part_snaps_a_world_point_onto_its_own_tree_only(frame):
    from hipct_seg_debug.edit import components as comp
    from .conftest_geometry import two_cylinders

    vol = two_cylinders(SHAPE)
    parts, _ = comp.split_components(vol, frame)
    big, small = parts[0], parts[1]
    nz, ny, nx = SHAPE
    inlet_um = frame.seg_to_um([[3, ny // 4, nz // 2]])[0]  # 2 voxels short of the cut face
    got = sk.root_for_part(big, frame, [inlet_um])
    assert got is not None and big.volume[got]
    assert np.abs(np.array(got) + big.origin_zyx - [nz // 2, ny // 4, 5]).max() <= 1
    assert sk.root_for_part(small, frame, [inlet_um]) is None, "not on this tree"
    far = frame.seg_to_um([[nx // 2, ny // 2, 2]])[0]
    assert sk.root_for_part(big, frame, [far], snap_voxels=3) is None


def test_skeletonise_per_component_roots_jin_where_it_is_told(frame):
    from .conftest_geometry import two_cylinders

    vol = two_cylinders(SHAPE)
    nz, ny, nx = SHAPE
    inlet = frame.seg_to_um([[5, ny // 4, nz // 2]])[0]
    rooted = sk.skeletonise_per_component("jin-mcp", vol, frame, roots_um=[inlet],
                                          engine="sparse", allow_loops=True)
    assert "root at crop voxel" in rooted.detail
    assert "automatic root" in rooted.detail, "the second tree had no root given"
    plain = sk.skeletonise_per_component("jin-mcp", vol, frame, engine="sparse",
                                         allow_loops=True)
    assert "root at crop voxel" not in plain.detail
    # the given root is a node of the big tree's skeleton
    nodes = np.array([n[:3] for n in rooted.triple.nodes.values()])
    assert np.linalg.norm(nodes - inlet, axis=1).min() <= 1.5 * frame.seg_spacing[0]


def test_root_graph_takes_each_trees_widest_free_end(tmp_path):
    from hipct_seg_debug.amira import SpatialGraph
    from hipct_seg_debug.edit import __main__ as cli
    from hipct_seg_debug.edit.amira_write import write_spatial_graph

    # tree A: an inlet end (radius 50) feeding a junction with two thin ends; tree B:
    # one edge whose second end is the wider. Each edge is two points, end to end.
    vertices = np.array([(0, 0, 0), (100, 0, 0), (200, 50, 0), (200, -50, 0),
                         (0, 500, 0), (100, 500, 0)], dtype=float)
    edges = np.array([(0, 1), (1, 2), (1, 3), (4, 5)])
    radii = [(50, 40), (12, 10), (12, 10), (10, 30)]
    points = np.concatenate([vertices[list(e)] for e in edges])
    thickness = np.array([r for pair in radii for r in pair], dtype=float)
    sg = SpatialGraph(path=None, n_vertex=6, n_edge=4, n_point=8, vertices=vertices,
                      connectivity=edges, n_edge_points=np.full(4, 2), points=points,
                      thickness=thickness)
    path = tmp_path / "ref.am"
    write_spatial_graph(sg, path)
    args = cli.build_parser().parse_args(["skeletonise-all", "--root-graph", str(path)])
    pts = np.array(cli._extraction_roots(args, None))
    assert sorted(map(tuple, pts)) == sorted([(0.0, 0.0, 0.0), (100.0, 500.0, 0.0)])


def test_root_for_part_holds_on_a_cropped_decode(frame):
    """A whole-heart decode is a foreground crop with a shifted frame; a world root
    must still land on the same lumen voxel. A full-lattice index used against the
    crop put a trial root 21 mm off on LADAF-2021-17."""
    from dataclasses import replace

    from hipct_seg_debug.edit import components as comp
    from .conftest_geometry import two_cylinders

    vol = two_cylinders(SHAPE)
    nz, ny, nx = SHAPE
    inlet_um = frame.seg_to_um([[5, ny // 4, nz // 2]])[0]
    full = sk.root_for_part(comp.split_components(vol, frame)[0][0], frame, [inlet_um])
    off = np.array([3, 2, 4])  # z, y, x
    crop = np.ascontiguousarray(vol[off[0]:, off[1]:, off[2]:])
    cframe = replace(frame, seg_dims=np.asarray(crop.shape[::-1], dtype=np.int64),
                     seg_origin=np.asarray(frame.seg_origin) + off[::-1] * frame.seg_spacing)
    part = comp.split_components(crop, cframe)[0][0]
    got = sk.root_for_part(part, cframe, [inlet_um])
    assert got is not None
    full_part = comp.split_components(vol, frame)[0][0]
    assert np.array_equal(np.array(got) + part.origin_zyx + off,
                          np.array(full) + full_part.origin_zyx)
