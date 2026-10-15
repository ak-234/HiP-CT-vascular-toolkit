"""The sparse Jin engine: the same method as the dense one, on lumen voxels only."""
import networkx as nx
import numpy as np
import pytest

from hipct_seg_debug.edit import jin_mcp, jin_mgf, jin_sparse
from hipct_seg_debug.edit import skeleton_phantoms as sp

from .conftest_geometry import make_frame


def _shapes():
    m = np.zeros((11, 11, 61))
    m[3:8, 3:8, 2:59] = 1
    yield "bar", m, (5, 5, 4)
    z, y, x = np.indices((13, 71, 71))
    m = ((z - 6) ** 2 + (y - 35) ** 2 <= 4) & (x >= 4) & (x <= 66)
    m |= ((z - 6) ** 2 + (x - 35) ** 2 <= 4) & (y >= 35) & (y <= 66)
    yield "three-arm", m.astype(float), (6, 35, 6)
    m = np.zeros((9, 19, 61))
    m[3:6, 3:16, 2:59] = 1
    yield "flattened", m, (4, 9, 4)
    m = np.zeros((11, 11, 61))
    m[3:8, 3:8, 2:59] = 0.4
    yield "fuzzy", m, (5, 5, 4)
    for name in ("collapse-bands", "y-bifurcation"):
        p = sp.build(name)
        yield name, p.mask.astype(float), p.root_zyx


SHAPES = list(_shapes())


@pytest.mark.parametrize("name,m,root", SHAPES, ids=[s[0] for s in SHAPES])
def test_identical_to_the_dense_engine(name, m, root):
    dense = jin_mcp.extract(m, root_zyx=root, max_voxels=10**9)
    sparse = jin_sparse.extract(m, root_zyx=root, incremental=False)
    np.testing.assert_array_equal(dense.coordinates_zyx, sparse.coordinates_zyx)
    np.testing.assert_array_equal(dense.edges, sparse.edges)
    np.testing.assert_allclose(dense.fdt, sparse.fdt)
    assert len(dense.report["rounds"]) == len(sparse.report["rounds"])


@pytest.mark.parametrize("name,m,root", SHAPES, ids=[s[0] for s in SHAPES])
def test_incremental_rounds_give_the_same_tree(name, m, root):
    """Distances are exact when continued; only tied predecessors could differ."""
    fresh = jin_sparse.extract(m, root_zyx=root, incremental=False)
    inc = jin_sparse.extract(m, root_zyx=root, incremental=True)
    degree = lambda r: sorted(d for _, d in nx.Graph(r.edges.tolist()).degree)  # noqa: E731
    assert degree(fresh) == degree(inc)
    assert abs(len(fresh.fdt) - len(inc.fdt)) <= 0.02 * len(fresh.fdt)


def test_loops_are_refused_unless_allowed_and_then_cut():
    p = sp.build("loop")
    with pytest.raises(ValueError, match="Euler"):
        jin_sparse.extract(p.mask, root_zyx=p.root_zyx)
    r = jin_sparse.extract(p.mask, root_zyx=p.root_zyx, allow_loops=True)
    assert nx.is_tree(nx.Graph(r.edges.tolist()))
    assert r.report["allow_loops"] and "loops in the mask are cut" in " ".join(
        r.report["limitations"])
    dense = jin_mcp.extract(p.mask.astype(float), root_zyx=p.root_zyx, max_voxels=10**9,
                            allow_loops=True)
    assert nx.is_tree(nx.Graph(dense.edges.tolist()))


def test_disconnected_input_is_refused():
    m = np.zeros((11, 11, 61))
    m[3:8, 3:8, 2:59] = 1
    m[0, 0, 0] = 1
    with pytest.raises(ValueError, match="one nonempty connected component"):
        jin_sparse.extract(m)


@pytest.mark.parametrize("name", ["collapse-bands", "y-bifurcation"])
def test_sparse_mgf_rim_is_identical_and_exact_agrees(name):
    """rim: bit-identical. exact: contacts by local search, ties broken by scan order --
    same tree shape, positions within half a voxel."""
    p = sp.build(name)
    m = p.mask.astype(float)
    for mode in ("rim", "exact"):
        dense = jin_mgf.extract(m, root_zyx=p.root_zyx, mgf_mode=mode)
        sparse = jin_mgf.extract(m, root_zyx=p.root_zyx, mgf_mode=mode, engine="sparse",
                                 incremental=False)
        assert dense.report["sheet_voxels"] == sparse.report["sheet_voxels"]
        if mode == "rim":
            np.testing.assert_array_equal(dense.coordinates_zyx, sparse.coordinates_zyx)
            np.testing.assert_array_equal(dense.edges, sparse.edges)
        degree = lambda r: sorted(d for _, d in nx.Graph(r.edges.tolist()).degree if d != 2)  # noqa: E731
        assert degree(dense) == degree(sparse)


def test_sparse_hessian_matches_dense():
    p = sp.build("curved-collapsed")
    m = p.mask.astype(float)
    fdt, lsf = jin_mcp.fields(m)
    dim_d, _ = jin_mgf.hessian_dimension(p.mask, lsf)
    fg = np.flatnonzero(p.mask.ravel())
    dim_s, _ = jin_mgf.hessian_dimension_sparse(p.mask > 0, fg, lsf.ravel()[fg] > .5)
    np.testing.assert_array_equal(dim_d.ravel()[fg], dim_s)


def test_backend_switches_to_sparse_beyond_max_voxels():
    from hipct_seg_debug.edit.skeletonisers import skeletonise

    m = np.zeros((9, 19, 61))
    m[3:6, 3:16, 2:59] = 1
    frame = make_frame(m.shape)
    dense = skeletonise("jin-mcp", m, frame, root_zyx=(4, 9, 4))
    auto = skeletonise("jin-mcp", m, frame, root_zyx=(4, 9, 4), max_voxels=1000,
                       incremental=False)
    assert dense.triple.points == auto.triple.points
    with pytest.raises(ValueError, match="engine=sparse"):
        skeletonise("jin-mcp", m, frame, max_voxels=1000, engine="dense")
    mgf = skeletonise("jin-mgf", m, frame, root_zyx=(4, 9, 4), max_voxels=1000)
    assert len(mgf.triple.segments) == 1
