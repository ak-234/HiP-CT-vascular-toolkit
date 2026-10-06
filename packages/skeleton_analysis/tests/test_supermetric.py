"""The paper's super metric (Walsh et al. 2024, Eq. 10) on geometry with known answers."""

from __future__ import annotations

import csv
import math
import warnings
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("skimage")

from skeleton_analysis.optimisation import supermetric as sm  # noqa: E402

DATA = Path(__file__).parent / "data"
VS = 2.0  # voxel size used by the synthetic images


# ----------------------------------------------------------------- helpers


def tube_image(shape=(60, 60, 120), r=6, x0=10, x1=110, yc=30, zc=30):
    z, y, x = np.indices(shape)
    return ((y - yc) ** 2 + (z - zc) ** 2 <= r * r) & (x >= x0) & (x <= x1)


def line_graph(p0, p1, r, n=20):
    pts = np.linspace(p0, p1, n)
    return sm.SkeletonGraph([p0, p1], [[0, 1]], [pts], [np.full(n, r)])


def star_graph(centres):
    """A graph whose nodes at ``centres`` each have degree 3."""
    nodes, edges, pts = [], [], []
    for c in centres:
        c = np.asarray(c, float)
        i = len(nodes)
        nodes.append(c)
        for d in np.eye(3) * 0.01:
            j = len(nodes)
            nodes.append(c + d)
            edges.append([i, j])
            pts.append(np.array([c, c + d]))
    return sm.SkeletonGraph(nodes, edges, pts)


def match(cands, refs, radius=4.0, options=sm.PAPER, bbox=(0, 100, 0, 100, 0, 100)):
    refs = np.asarray(refs, float).reshape(-1, 3)
    tol = sm.bifurcation_tolerance(np.full(len(refs), radius), options)
    return sm.match_bifurcations(star_graph(cands).bifurcations(), refs, tol,
                                 None if bbox is None else np.asarray(bbox, float),
                                 options.matching)


# --------------------------------------------------------------- graph terms


def test_tube_scores_near_zero():
    b = tube_image()
    g = line_graph([10 * VS, 30 * VS, 30 * VS], [110 * VS, 30 * VS, 30 * VS], 6 * VS)
    ref = sm.reference_metrics(b, VS)
    assert ref.n_components == 1 and ref.local_euler == 1
    with pytest.warns(UserWarning, match="bifurcation DICE term omitted"):
        res = sm.super_metric(g, ref)
    assert res.skeleton["cl_sensitivity"] == 1.0
    assert res.skeleton["volume"] == pytest.approx(math.pi * 12 ** 2 * 200)
    assert res.terms["volume"] < 0.05
    assert res.paper_comparable


def test_cl_counts_lines_leaving_the_vessel_or_the_image():
    b = tube_image()
    off = line_graph([10 * VS, 30 * VS, 30 * VS], [110 * VS, 30 * VS, 50 * VS], 6 * VS)
    assert sm.cl_sensitivity(off, b, VS) < 0.8
    out = line_graph([10 * VS, 30 * VS, 30 * VS], [230 * VS, 30 * VS, 30 * VS], 6 * VS)
    assert sm.cl_sensitivity(out, b, VS) < 0.5


def test_cylinder_volume_uses_mean_radius_per_subsegment():
    g = sm.SkeletonGraph([[0, 0, 0], [3, 0, 0]], [[0, 1]],
                         [np.array([[0, 0, 0], [2, 0, 0], [3, 0, 0]])], [np.array([1., 3, 3])])
    assert g.volume() == pytest.approx(math.pi * 2 ** 2 * 2 + math.pi * 3 ** 2 * 1)


def test_torus_local_euler_and_components():
    shape = (40, 100, 100)
    z, y, x = np.indices(shape)
    R, r = 30, 5
    torus = (np.sqrt((x - 50) ** 2 + (y - 50) ** 2) - R) ** 2 + (z - 20) ** 2 <= r * r
    blob = (x - 5) ** 2 + (y - 5) ** 2 + (z - 5) ** 2 <= 9
    ref = sm.reference_metrics(torus | blob, 1.0)
    assert ref.n_components == 2 and ref.local_euler == 2  # classical 0 -> local 2

    ang = np.linspace(0, 2 * np.pi, 5)[:-1]
    nodes = np.c_[50 + R * np.cos(ang), 50 + R * np.sin(ang), np.full(4, 20)]
    edges = [[i, (i + 1) % 4] for i in range(4)]
    pts = [np.linspace(nodes[a], nodes[b], 10) for a, b in edges]
    nodes = np.vstack([nodes, [[4, 5, 5], [6, 5, 5], [90, 90, 5]]])  # + a line + an isolated node
    edges.append([4, 5])
    pts.append(np.linspace([4, 5, 5], [6, 5, 5], 3))
    g = sm.SkeletonGraph(nodes, edges, pts, [np.full(len(p), r) for p in pts])
    assert g.n_components() == 3, "an isolated node is a component"
    assert g.largest_component_local_euler() == 2

    loop = sm.SkeletonGraph([[0, 0, 0]], [[0, 0]],
                            [np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 0]])], [np.ones(4)])
    assert loop.degrees[0] == 2 and loop.largest_component_local_euler() == 2


# ----------------------------------------------------------- bifurcations


def test_hungarian_finds_the_pairing_greedy_misses():
    G1, G2 = [50., 50, 50], [57., 50, 50]
    S1, S2 = [53., 50, 50], [46., 47, 50]  # S1: 3 from G1, 4 from G2; S2: 5 from G1 only
    res = match([S1, S2], [G1, G2])
    assert (res.tp, res.fp, res.fn) == (2, 0, 0)
    greedy = match([S1, S2], [G1, G2], options=replace(sm.PAPER, matching="greedy"))
    assert (greedy.tp, greedy.fp, greedy.fn) == (1, 1, 1)


def test_clustered_bifurcations_are_duplicate_false_positives():
    G = np.array([50., 50, 50])
    res = match([G + [1, 0, 0], G + [0, 2, 0], G + [0, 0, -3], [10, 10, 10]], [G])
    assert (res.tp, res.fp, res.fn, res.fp_duplicate, res.fp_isolated) == (1, 3, 0, 2, 1)
    assert res.dice == pytest.approx(2 / 5)


def test_box_edge_and_tolerance():
    # GT just inside the box, skeleton just outside: matched, not FP + FN.
    res = match([[102., 50, 50], [120., 50, 50]], [[99., 50, 50]])
    assert (res.tp, res.fp, res.fn) == (1, 0, 0)
    # Beyond 1.5 x radius: an FN and an isolated FP.
    res = match([[57., 50, 50]], [[50., 50, 50]])
    assert (res.tp, res.fp, res.fn, res.fp_isolated) == (0, 1, 1, 1)
    # A fixed tolerance replaces the radius-based one.
    res = match([[57., 50, 50]], [[50., 50, 50]],
                options=replace(sm.PAPER, tolerance="fixed", tolerance_value=10.0))
    assert res.tp == 1


def test_no_bifurcations_anywhere_is_not_applicable():
    res = match([], [])
    assert math.isnan(res.dice)
    assert math.isnan(sm.overlap_term(res.dice, 2))


def test_zero_overlap_makes_the_total_infinite():
    assert sm.overlap_term(0.0, 2) == math.inf
    assert sm.overlap_term(0.5, 2) == pytest.approx(2.0)
    assert sm.combine_terms({"a": 0.1, "b": math.inf, "c": math.nan}) == math.inf
    assert sm.combine_terms({"a": 0.1, "c": math.nan}) == pytest.approx(0.1)


def test_reference_radius_comes_from_the_distance_map():
    b = tube_image()
    gt = star_graph([[60 * VS, 30 * VS, 30 * VS]])
    ref = sm.reference_metrics(b, VS, gt_graph=gt, bbox=[0, 240, 0, 120, 0, 120])
    assert 11 <= ref.ref_radius[0] <= 15
    assert ref.ref_tolerance[0] == pytest.approx(1.5 * ref.ref_radius[0])

    fat = tube_image(shape=(80, 80, 80), r=30, x0=0, x1=79, yc=40, zc=40)
    full = sm.ndimage.distance_transform_edt(fat)[40, 40, 40]
    ref2 = sm.reference_metrics(fat, 1.0, gt_graph=star_graph([[40., 40, 40]]),
                                bbox=[30, 50, 30, 50, 30, 50])
    assert ref2.ref_radius[0] == pytest.approx(full)


def test_gt_points_equal_gt_graph():
    b = tube_image()
    gt = star_graph([[60 * VS, 30 * VS, 30 * VS]])
    box = [0, 240, 0, 120, 0, 120]
    r1 = sm.reference_metrics(b, VS, gt_graph=gt, bbox=box)
    r2 = sm.reference_metrics(b, VS, gt_bifurcations=gt.bifurcations(), bbox=box)
    assert np.allclose(r1.ref_bifurcations, r2.ref_bifurcations)
    assert np.allclose(r1.ref_radius, r2.ref_radius)
    with pytest.raises(ValueError):
        sm.reference_metrics(b, VS, gt_graph=gt, gt_bifurcations=gt.bifurcations())


# ------------------------------------------------------------- options


def test_presets():
    assert sm.PAPER.paper_comparable and not sm.CORONARY.paper_comparable
    assert sm.CORONARY.bifurcation_reference == "auto" and sm.CORONARY.chi_reference == "tree"
    with pytest.raises(ValueError):
        sm.SuperMetricOptions(matching="nearest")


def test_tree_chi_reference():
    shape = (40, 100, 100)
    z, y, x = np.indices(shape)
    torus = (np.sqrt((x - 50) ** 2 + (y - 50) ** 2) - 30) ** 2 + (z - 20) ** 2 <= 25
    image = sm.reference_metrics(torus, 1.0)
    tree = sm.reference_metrics(torus, 1.0, options=replace(sm.PAPER, chi_reference="tree"))
    assert image.local_euler == 2 and tree.local_euler == 1


def test_invalid_points_are_excluded_from_volume_and_cl():
    pts = np.array([[0., 0, 0], [10, 0, 0], [20, 0, 0], [30, 0, 0]])
    g = sm.SkeletonGraph([pts[0], pts[-1]], [[0, 1]], [pts], [np.ones(4)],
                         [np.array([True, True, False, True])])
    assert g.volume() == pytest.approx(math.pi * 30)
    assert g.volume(exclude_invalid=True) == pytest.approx(math.pi * 10)
    assert len(sm.rasterise_voxels(g, 1.0, exclude_invalid=True)) == 11


def test_auto_reference_finds_a_y_junction():
    shape = (40, 80, 80)
    z, y, x = np.indices(shape).astype(float)

    def tube(p0, p1, r=4):
        p0, p1 = np.array(p0, float), np.array(p1, float)
        vox = np.stack([x, y, z], -1)
        d = p1 - p0
        t = np.clip(((vox - p0) @ d) / (d @ d), 0, 1)
        return np.linalg.norm(vox - (p0 + t[..., None] * d), axis=-1) <= r

    b = tube([5, 40, 20], [40, 40, 20]) | tube([40, 40, 20], [75, 15, 20]) | tube([40, 40, 20], [75, 65, 20])
    ref = sm.reference_metrics(b, 1.0, options=sm.CORONARY)
    assert ref.bbox is None
    assert len(ref.ref_bifurcations) >= 1
    assert np.min(np.linalg.norm(ref.ref_bifurcations - [40, 40, 20], axis=1)) < 6


# ---------------------------------------------------------------- inputs


def _amira_test_graph():
    return sm.SkeletonGraph.from_amira(DATA / "Test.am")


def test_reads_a_real_avizo_spatial_graph():
    g = _amira_test_graph()
    assert (len(g.nodes), len(g.edges), sum(len(p) for p in g.edge_points)) == (148, 147, 19924)
    assert g.edge_radii is not None
    assert g.n_components() == 1 and g.largest_component_local_euler() == 1  # a tree
    assert np.bincount(g.degrees).tolist() == [0, 75, 0, 73]


def test_csv_and_flat_inputs_match_amira(tmp_path):
    g = _amira_test_graph()
    pts = np.vstack(g.edge_points)
    num = [len(p) for p in g.edge_points]
    flat = sm.SkeletonGraph.from_flat(g.nodes, g.edges, pts, num, np.concatenate(g.edge_radii))
    assert flat.volume() == pytest.approx(g.volume())

    def write(name, header, rows):
        with open(tmp_path / name, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(header)
            w.writerows(rows)

    write("nodes.csv", ["x", "y", "z"], g.nodes.tolist())
    write("edges.csv", ["node1", "node2"], g.edges.tolist())
    eid = np.repeat(np.arange(len(num)), num)
    rows = [[e, *p, r] for e, p, r in zip(eid, pts.tolist(), np.concatenate(g.edge_radii))]
    cut = int(np.cumsum(num)[10])  # move whole edges: within-edge order must be kept
    write("points.csv", ["edge", "x", "y", "z", "radius"], rows[cut:] + rows[:cut])
    from_csv = sm.load_graph(tmp_path)
    assert np.allclose(np.vstack(from_csv.edge_points), pts)
    assert from_csv.volume() == pytest.approx(g.volume())


@pytest.mark.parametrize("build, message", [
    (lambda V, E, P, T: sm.SkeletonGraph(V, [[0, 3]], [P[:3]]), "outside"),
    (lambda V, E, P, T: sm.SkeletonGraph(V, E, [P[:3]]), "polylines"),
    (lambda V, E, P, T: sm.SkeletonGraph(V, E, [P[:1], P[3:]]), "fewer than 2"),
    (lambda V, E, P, T: sm.SkeletonGraph(V, E, [P[:3], P[3:]], [T[:3], T[3:5]]), "edge_radii"),
    (lambda V, E, P, T: sm.SkeletonGraph.from_flat(V, E, P, [3, 3], T), "sums to"),
    (lambda V, E, P, T: sm.SkeletonGraph(V, [[0.5, 1]], [P[:3]]), "integer"),
])
def test_input_validation(build, message):
    V = np.array([[0, 0, 0], [10, 0, 0], [10, 10, 0]], float)
    E = np.array([[0, 1], [1, 2]])
    P = np.array([[0, 0, 0], [5, 0, 0], [10, 0, 0], [10, 0, 0], [10, 3, 0], [10, 6, 0],
                  [10, 10, 0]], float)
    T = np.ones(7)
    with pytest.raises(ValueError, match=message):
        build(V, E, P, T)


def test_polyline_off_its_nodes_warns_but_reversed_is_fine():
    V = np.array([[0, 0, 0], [10, 0, 0]], float)
    P = np.linspace(V[0], V[1], 5)
    with pytest.warns(UserWarning, match="do not start/end"):
        sm.SkeletonGraph(V, [[0, 1]], [P + 5])
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        sm.SkeletonGraph(V, [[0, 1]], [P[::-1]])


def test_command_line(tmp_path):
    tifffile = pytest.importorskip("tifffile")
    tifffile.imwrite(tmp_path / "tube.tif", tube_image().astype(np.uint8) * 255)
    out = tmp_path / "res.csv"
    rc = sm.main([str(tmp_path / "tube.tif"), str(DATA / "Test.am"), "--voxel-size", "2",
                  "--preset", "coronary", "-o", str(out)])
    assert rc == 0
    row = next(csv.DictReader(open(out)))
    assert row["paper_comparable"] == "False"
