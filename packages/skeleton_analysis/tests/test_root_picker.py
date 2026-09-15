"""Tests for the interactive per-tree root picker (ordering.root_picker)."""

import numpy as np
import pytest

from skeleton_analysis.graph.neighbors import coordination_number
from skeleton_analysis.io.amira import (
    F_EDGE_CONNECTIVITY,
    F_NUM_EDGE_POINTS,
    F_POINT_COORDS,
    F_THICKNESS,
    SpatialGraph,
)
from skeleton_analysis.ordering.root_picker import (
    compute_frenet_frame,
    pick_roots,
    root_from_edge,
    strahler_edge_colors,
    tree_components,
)


def _two_tree_graph():
    """Two disjoint Y-trees. Tree A: nodes 0..3, edges (0-1,1-2,1-3); node 0 is a
    degree-1 inlet. Tree B: nodes 4..6, edges (4-5,5-6). Two points per edge."""
    edges = np.array(
        [[0, 1], [1, 2], [1, 3],      # tree A (node 1 is the bifurcation)
         [4, 5], [5, 6]],             # tree B
        dtype=np.int64,
    )
    coords = {
        0: (0.0, 0.0, 0.0), 1: (10.0, 0.0, 0.0), 2: (20.0, 5.0, 0.0),
        3: (20.0, -5.0, 0.0), 4: (0.0, 50.0, 0.0), 5: (10.0, 50.0, 0.0),
        6: (20.0, 50.0, 0.0),
    }
    pts, nump = [], []
    for a, b in edges:
        pts.append(coords[int(a)])
        pts.append(coords[int(b)])
        nump.append(2)
    g = SpatialGraph()
    g.set_vertex_field("GraphVertices", np.array([coords[i] for i in range(7)], float))
    g.set_edge_field(F_EDGE_CONNECTIVITY, edges)
    g.set_edge_field(F_NUM_EDGE_POINTS, np.array(nump, np.int64))
    g.set_edge_field("strahler", np.array([2, 1, 1, 1, 1], np.int64))
    g.set_point_field(F_POINT_COORDS, np.array(pts, float))
    g.set_point_field(F_THICKNESS, np.full(len(pts), 3.0))
    return g


def test_tree_components_groups_edges_per_tree():
    g = _two_tree_graph()
    comps = tree_components(g)
    assert len(comps) == 2
    # Largest tree first: tree A has 3 edges, tree B has 2.
    assert sorted(comps[0].tolist()) == [0, 1, 2]
    assert sorted(comps[1].tolist()) == [3, 4]


def test_root_from_edge_picks_degree1_inlet():
    g = _two_tree_graph()
    edges = np.asarray(g.edge_connectivity, dtype=np.int64)
    coord = coordination_number(edges)
    # Inlet edge 0 = (0,1): node 0 is degree-1, node 1 is the bifurcation (degree 3).
    assert root_from_edge(0, edges, coord) == 0
    # Inlet edge 3 = (4,5): node 4 is degree-1.
    assert root_from_edge(3, edges, coord) == 4
    # A distal edge 1 = (1,2): node 2 (degree 1) wins over node 1 (degree 3).
    assert root_from_edge(1, edges, coord) == 2


def test_strahler_edge_colors_one_per_edge_and_legend():
    pytest.importorskip("matplotlib")
    g = _two_tree_graph()
    fn, legend = strahler_edge_colors(g)
    cols = [fn(i) for i in range(g.n_edges)]
    assert len(cols) == g.n_edges
    assert all(len(c) == 3 for c in cols)          # RGB triples
    # Strahler orders present are {1, 2} -> legend has 2 entries.
    assert [lbl for lbl, _rgb in legend] == ["order 1", "order 2"]


def test_strahler_edge_colors_falls_back_without_field():
    g = _two_tree_graph()
    del g.edge_fields["strahler"]
    fn, legend = strahler_edge_colors(g)
    assert legend == []
    assert len(fn(0)) == 3                          # flat grey RGB


def test_compute_frenet_frame_orthonormal():
    rng = np.random.default_rng(0)
    prev = None
    for _ in range(20):
        tangent = rng.normal(size=3)
        if np.linalg.norm(tangent) < 1e-9:
            continue
        t, n, b = compute_frenet_frame(tangent, prev)
        # Unit vectors.
        for v in (t, n, b):
            assert abs(np.linalg.norm(v) - 1.0) < 1e-9
        # t aligns with the tangent direction.
        assert np.allclose(t, tangent / np.linalg.norm(tangent))
        # Mutually orthogonal (n ⟂ t, b ⟂ t, b ⟂ n) and right-handed.
        assert abs(np.dot(t, n)) < 1e-9
        assert abs(np.dot(t, b)) < 1e-9
        assert abs(np.dot(n, b)) < 1e-9
        assert np.allclose(np.cross(t, n), b, atol=1e-9)
        prev = n  # continuity across calls


def test_segment_contour_mesh_shapes():
    pv = pytest.importorskip("pyvista")
    from skeleton_analysis.ordering.root_picker import _segment_contour_mesh

    P, n_sides, radius = 6, 8, 2.0
    coords = np.stack([np.arange(P, dtype=float), np.zeros(P), np.zeros(P)], axis=1)
    radii = np.full(P, radius)
    poly = _segment_contour_mesh(pv, coords, radii, n_sides=n_sides, ring_stride=1)

    # One ring (n_sides pts) per centreline point + all P centreline points.
    assert poly.n_points == P * n_sides + P
    assert poly.n_verts == P                       # every centreline point is a dot
    # Each ring point sits ~radius from its centreline point (straight line along x).
    ring_pts = np.asarray(poly.points)[: P * n_sides].reshape(P, n_sides, 3)
    for i in range(P):
        d = np.linalg.norm(ring_pts[i] - coords[i], axis=1)
        assert np.allclose(d, radius, atol=1e-6)


def test_pick_roots_offscreen_preselect(tmp_path):
    pytest.importorskip("pyvista")
    pytest.importorskip("matplotlib")
    g = _two_tree_graph()
    out = tmp_path / "root_pick.png"
    try:
        # Headless: pre-select the inlet edge of the first (largest) tree.
        roots = pick_roots(g, off_screen=True, screenshot=str(out), preselect=0)
    except Exception as exc:  # no OpenGL/VTK render context in this environment
        pytest.skip(f"off-screen rendering unavailable: {exc}")

    assert roots == [0]                             # edge 0 -> root node 0
    assert out.exists() and out.stat().st_size > 0
