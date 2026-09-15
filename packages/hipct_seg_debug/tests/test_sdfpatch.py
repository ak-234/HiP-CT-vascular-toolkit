"""Is the live preview telling the truth?

The whole design rests on one claim: a patch rebuilt over a box is the same
geometry a full ``generate_sdf_surface`` run would put there. If that is wrong,
the viewer shows the user something they will not get when they export, which is
worse than being slow.

So the headline test runs the real pipeline over the real graph, rebuilds a box
out of the same session, and measures point-to-surface deviation between them.
Point-to-*point* distance is not a useful measure here: meshlib re-tessellates,
so two identical surfaces differ by roughly half an edge length in that metric.

These tests need the LADAF-2024-28 graph and take a couple of minutes; they skip
cleanly when it is absent.
"""

from __future__ import annotations


import numpy as np
import pytest

from .realdata import GRAPH_REASON, REAL_AM

pytestmark = pytest.mark.skipif(
    not REAL_AM.is_file(), reason=GRAPH_REASON
)


def surface_deviation(sample, reference, box=None):
    """Distance from `sample`'s vertices to the nearest point *on* `reference`.

    Restricted to `box` when given, so the trim boundary is not what gets
    measured. Returns ``(mean, max, n)`` in the meshes' own units (um here).
    """
    pts = np.asarray(sample.points, dtype=np.float64)
    if box is not None and len(pts):
        keep = np.all((pts >= box[0]) & (pts <= box[1]), axis=1)
        pts = pts[keep]
    if not len(pts) or reference.n_cells == 0:
        return float("nan"), float("nan"), 0
    _, closest = reference.find_closest_cell(pts, return_closest_point=True)
    d = np.linalg.norm(pts - closest, axis=1)
    return float(d.mean()), float(d.max()), len(pts)


@pytest.fixture(scope="module")
def session():
    from hipct_seg_debug.edit.adapter import read_triple
    from hipct_seg_debug.edit.sdfpatch import SdfSession

    return SdfSession(read_triple(REAL_AM))


@pytest.fixture(scope="module")
def busy_node(session):
    """A high-degree junction -- the hardest place for a patch to agree."""
    from hipct_seg_debug.edit.graphmodel import EditableGraph

    g = EditableGraph(read_triple_cached())
    deg = {n: g.degree(n) for n in g.nodes}
    nid = max(deg, key=deg.get)
    return np.array(g.nodes[nid][:3], dtype=np.float64)


def read_triple_cached():
    from hipct_seg_debug.edit.adapter import read_triple

    return read_triple(REAL_AM)


def test_session_freezes_one_voxel_size(session):
    # compute_grid derives the voxel size from the smallest capsule anywhere in
    # the tree; a patch must inherit it rather than deriving its own from
    # whatever happens to be in the box, or patches would not tile.
    assert 0.05 < session.voxel_size_mm < 0.5
    first = session.voxel_size_mm
    session.set_graph(read_triple_cached())
    assert session.voxel_size_mm == first


def test_patch_grid_is_a_sub_block_of_the_full_grid(session, busy_node):
    full = session._ctx.full_grid
    box_mm = np.array([busy_node - 3000.0, busy_node + 3000.0]) / 1000.0
    sub, _origin = session._subgrid(box_mm)

    assert sub.voxel_size == full.voxel_size
    for axis_sub, axis_full in ((sub.x, full.x), (sub.y, full.y), (sub.z, full.z)):
        # Every sample position must appear verbatim in the full lattice.
        i = int(np.searchsorted(axis_full, axis_sub[0]))
        assert np.array_equal(axis_sub, axis_full[i:i + len(axis_sub)])
    assert sub.bbox_min[0] == sub.x[0] and sub.bbox_max[0] == sub.x[-1]


def test_patch_is_deterministic(session, busy_node):
    box = np.array([busy_node - 2000.0, busy_node + 2000.0])
    a = session.rebuild(box)
    b = session.rebuild(box)
    assert a.surface.n_points == b.surface.n_points
    assert np.allclose(np.asarray(a.surface.points), np.asarray(b.surface.points))


def test_patch_is_confined_to_its_box(session, busy_node):
    box = np.array([busy_node - 2000.0, busy_node + 2000.0])
    res = session.rebuild(box)
    assert res.surface.n_cells > 0
    cc = np.asarray(res.surface.cell_centers().points)
    assert np.all(cc >= box[0]) and np.all(cc <= box[1]), \
        "the patch was not trimmed back to the requested box"


def test_margin_size_barely_changes_the_geometry(session, busy_node):
    """The SDF field does not depend on the box, so margin should not either."""
    box = np.array([busy_node - 2000.0, busy_node + 2000.0])
    core = np.array([busy_node - 1200.0, busy_node + 1200.0])
    vox_um = session.voxel_size_mm * 1000.0

    generous = session.rebuild(box, pad_um=30 * vox_um)
    tight = session.rebuild(box, pad_um=2 * vox_um)
    mean_d, max_d, n = surface_deviation(tight.surface, generous.surface, core)

    assert n > 100, "not enough geometry in the core to be a meaningful comparison"
    assert mean_d < 0.05 * vox_um, f"mean deviation {mean_d:.2f} um vs voxel {vox_um:.1f} um"
    assert max_d < 0.5 * vox_um, f"max deviation {max_d:.2f} um vs voxel {vox_um:.1f} um"


@pytest.mark.slow
def test_patch_capsules_and_field_match_a_full_run_exactly(session, busy_node, tmp_path):
    """The load-bearing test.

    Compares what a patch is meshing against what a full ``generate_sdf_surface``
    run meshes, at the only level where "identical" is a meaningful word: the
    capsules and the signed distance field. Meshes cannot be compared this
    strictly because meshlib's remesh and ``relaxKeepVolume`` are global
    iterative passes -- see the mesh test below.

    Instruments ``evaluate_sdf`` in place rather than reimplementing the
    pipeline, so the comparison is against the real thing.
    """
    import coronary_sdf.pipeline as pipeline
    import coronary_sdf.sdf_field as sdf_field

    captured: dict = {}
    real = sdf_field.evaluate_sdf

    def spy(**kw):
        out = real(**kw)
        captured.update(capsules=kw["capsules"], grid=kw["grid"], sdf=out.sdf)
        return out

    sdf_field.evaluate_sdf = spy
    pipeline.evaluate_sdf = spy
    try:
        session.rebuild_full(tmp_path)
    finally:
        sdf_field.evaluate_sdf = real
        pipeline.evaluate_sdf = real

    ctx = session._ctx
    theirs, full_grid = captured["capsules"], captured["grid"]

    # 1. The capsules -- the entire geometric input to the field.
    assert ctx.capsules.n == theirs.n
    for name in ("starts", "ends", "radii_start", "radii_end", "seg_idx",
                 "max_radii", "arc_start", "arc_end"):
        assert np.array_equal(
            np.asarray(getattr(ctx.capsules, name)), np.asarray(getattr(theirs, name))
        ), f"capsule field {name!r} differs between the patch session and a full run"

    # 2. The lattice.
    assert ctx.full_grid.voxel_size == full_grid.voxel_size
    for ax in ("x", "y", "z"):
        assert np.array_equal(getattr(ctx.full_grid, ax), getattr(full_grid, ax))

    # 3. The field itself, on a sub-block, where it matters: near the surface.
    from hipct_seg_debug.edit.sdfconfig import sdf_config
    from hipct_seg_debug.edit.sdfpatch import _quiet

    box_mm = np.array([busy_node - 3000.0, busy_node + 3000.0]) / 1000.0
    sub, _origin = session._subgrid(box_mm)
    with _quiet(True), sdf_config(session.profile):
        nb = sdf_field.build_narrow_band(ctx.capsules, sub)
        local = real(
            capsules=ctx.capsules, cap_is_junction=ctx.cap_is_junction,
            adj_matrix=ctx.adj_matrix, shared_node_pos=ctx.shared_node_pos,
            shared_node_has=ctx.shared_node_has, seg_end_pos=ctx.seg_end_pos,
            seg_end_tan=ctx.seg_end_tan, seg_end_tan_ok=ctx.seg_end_tan_ok,
            is_parent=ctx.is_parent, is_child=ctx.is_child, is_sibling=ctx.is_sibling,
            bif_seg_incident=ctx.bif_seg_incident, bif=ctx.bif, term=ctx.term,
            grid=sub, nb_idx=nb,
        ).sdf

    i0 = int(np.searchsorted(full_grid.x, sub.x[0]))
    j0 = int(np.searchsorted(full_grid.y, sub.y[0]))
    k0 = int(np.searchsorted(full_grid.z, sub.z[0]))
    block = captured["sdf"][i0:i0 + sub.dims[0], j0:j0 + sub.dims[1], k0:k0 + sub.dims[2]]

    diff = np.abs(local.astype(np.float64) - block.astype(np.float64))
    # Only voxels the iso-surface can pass through matter. Far-field voxels
    # differ where the full grid's band reached them and the local one's did not,
    # which is the 10.0 sentinel and has no effect on any mesh.
    near = np.abs(block) < 0.5
    assert near.any(), "the test box contains no surface"
    assert diff[near].max() == 0.0, (
        f"the field near the surface differs by up to {diff[near].max()*1000:.3f} um"
    )


@pytest.mark.slow
def test_patch_mesh_matches_a_full_run_with_a_local_mesher(busy_node, tmp_path):
    """With a per-voxel mesher the patch reproduces a full run exactly.

    ``fast_contour_zero`` (vtkFlyingEdges) decides each cell from its own eight
    corners, so given an identical field on an identical lattice there is nothing
    left to diverge. This is the test that would have caught the mesh-origin bug:
    before it was fixed, this deviation was ~42 um rather than ~0.02 um.
    """
    from hipct_seg_debug.edit.adapter import read_triple
    from hipct_seg_debug.edit.sdfconfig import PREVIEW
    from hipct_seg_debug.edit.sdfpatch import SdfSession

    session = SdfSession(read_triple(REAL_AM), profile={**PREVIEW, "SDF_MESH_METHOD": "mc"})
    full = session.rebuild_full(tmp_path)
    assert full is not None and full.n_cells > 0
    full_um = full.scale(1000.0, inplace=False)

    box = np.array([busy_node - 3000.0, busy_node + 3000.0])
    core = np.array([busy_node - 2000.0, busy_node + 2000.0])
    patch = session.rebuild(box)

    vox_um = session.voxel_size_mm * 1000.0
    mean_d, max_d, n = surface_deviation(patch.surface, full_um, core)
    print(f"\nmc patch vs full: n={n} mean={mean_d:.4f}um max={max_d:.4f}um")
    assert n > 100
    assert max_d < 0.005 * vox_um, (
        f"max deviation {max_d:.4f} um is more than floating-point noise "
        f"(voxel {vox_um:.1f} um) -- the patch lattice has drifted from the full one"
    )


@pytest.mark.slow
def test_patch_mesh_is_a_fraction_of_a_voxel_from_a_full_run(session, busy_node, tmp_path):
    """Mesh agreement under the default mesher, which is looser -- by design.

    The field is bit-identical, but ``mesh_from_sdf_meshlib`` remeshes and then
    runs ``relaxKeepVolume`` for 30 iterations over the whole mesh. Both passes
    are global, so a patch and a full run relax differently and the residual does
    not shrink with margin. Measured on LADAF-2024-28: ~0.5% of a voxel mean,
    ~21% max, i.e. far below the resolution the graph was derived from.
    """
    full = session.rebuild_full(tmp_path)
    assert full is not None and full.n_cells > 0
    full_um = full.scale(1000.0, inplace=False)

    box = np.array([busy_node - 3000.0, busy_node + 3000.0])
    core = np.array([busy_node - 2000.0, busy_node + 2000.0])
    patch = session.rebuild(box)
    assert patch.surface.n_cells > 0

    vox_um = session.voxel_size_mm * 1000.0
    mean_d, max_d, n = surface_deviation(patch.surface, full_um, core)
    print(
        f"\npatch vs full mesh: n={n} mean={mean_d:.3f}um ({100*mean_d/vox_um:.1f}% voxel) "
        f"max={max_d:.3f}um ({100*max_d/vox_um:.1f}% voxel); patch {patch.seconds:.2f}s"
    )
    assert mean_d < 0.05 * vox_um, f"mean deviation {mean_d:.2f} um vs voxel {vox_um:.1f} um"
    assert max_d < 0.5 * vox_um, f"max deviation {max_d:.2f} um vs voxel {vox_um:.1f} um"


@pytest.mark.slow
def test_mesh_origin_compensates_the_lattice_spacing_mismatch(session, busy_node):
    """Guards the specific arithmetic that made patches land tens of um off.

    ``compute_grid`` samples at ``linspace(bbox_min, bbox_max, dims)`` but tells
    the mesher the spacing is ``voxel_size``; those differ by ~1/(dims-1). The
    mesher origin must therefore be ``bbox_min + index * voxel_size``, not the
    sample position ``x[index]``.
    """
    full = session._ctx.full_grid
    box_mm = np.array([busy_node - 3000.0, busy_node + 3000.0]) / 1000.0
    sub, mesh_origin = session._subgrid(box_mm)

    step = (full.x[-1] - full.x[0]) / (len(full.x) - 1)
    assert step != full.voxel_size, "no mismatch to compensate; this test is vacuous"

    i0 = int(np.searchsorted(full.x, sub.x[0]))
    assert mesh_origin[0] == pytest.approx(full.bbox_min[0] + i0 * full.voxel_size)
    # The naive choice, and the size of the error it would introduce.
    naive_error_um = abs(sub.x[0] - mesh_origin[0]) * 1000.0
    assert naive_error_um > 1.0, (
        "the lattice mismatch is too small here for this test to prove anything"
    )


def test_rebuild_around_expands_a_degenerate_patch(session, busy_node):
    from hipct_seg_debug.edit.history import Patch

    # A radius edit on one point reports a zero-volume box.
    flat = Patch(frozenset({0}), np.array([busy_node, busy_node]))
    res = session.rebuild_around(flat, min_extent_um=3000.0)
    extent = res.box_um[1] - res.box_um[0]
    assert np.allclose(extent, 3000.0)
    assert res.surface.n_cells > 0


def test_rebuild_in_empty_space_returns_nothing(session):
    far = np.array([-1e6, -1e6, -1e6])
    res = session.rebuild(np.array([far, far + 1000.0]))
    assert res.surface.n_cells == 0


def test_splice_replaces_only_the_patch_box(session, busy_node):
    from hipct_seg_debug.edit.sdfpatch import splice

    base = session.rebuild(np.array([busy_node - 8000.0, busy_node + 8000.0]))
    inner_box = np.array([busy_node - 2000.0, busy_node + 2000.0])
    patch = session.rebuild(inner_box)

    merged = splice(base.surface, patch)
    cc = np.asarray(merged.cell_centers().points)
    inside = np.all((cc >= inner_box[0]) & (cc <= inner_box[1]), axis=1)

    # Everything inside the box came from the patch; nothing of the base survives.
    assert inside.sum() == patch.surface.n_cells
    assert merged.n_cells == (base.surface.n_cells
                              - np.all((np.asarray(base.surface.cell_centers().points)
                                        >= inner_box[0])
                                       & (np.asarray(base.surface.cell_centers().points)
                                          <= inner_box[1]), axis=1).sum()
                              + patch.surface.n_cells)


def test_config_is_restored_after_a_rebuild(session, busy_node):
    from hipct_seg_debug.edit._deps import ensure_coronary_sdf

    ensure_coronary_sdf()
    from coronary_sdf import config

    before = (config.DEBUG_VIS, config.DEBUG_VIS_BLOCK, config.SDF_MAX_CAPSULE_QUERY)
    session.rebuild(np.array([busy_node - 1500.0, busy_node + 1500.0]))
    after = (config.DEBUG_VIS, config.DEBUG_VIS_BLOCK, config.SDF_MAX_CAPSULE_QUERY)
    assert before == after, "a rebuild leaked its config overrides into the process"


def test_an_edit_changes_the_surface_where_it_landed(session):
    """End to end: edit the graph, re-prepare, and see the patch move."""
    from hipct_seg_debug.edit.graphmodel import EditableGraph

    g = EditableGraph(read_triple_cached())
    # Pick a mid-tree segment with enough points to have a safe interior.
    sid = next(s["id"] for s in g.segments if len(s["point_ids"]) > 40)
    pid = g.segment(sid)["point_ids"][len(g.segment(sid)["point_ids"]) // 2]
    centre = np.array(g.points[pid][:3])
    box = np.array([centre - 1500.0, centre + 1500.0])

    session.set_graph(g.snapshot())
    before = session.rebuild(box)

    patch = g.scale_radii(sid, 1.6)
    session.set_graph(g.snapshot())
    after = session.rebuild(box)

    assert patch.seg_ids == {sid}, "a radius edit must name the segment it changed"
    mean_d, _, n = surface_deviation(after.surface, before.surface, box)
    assert n > 50
    assert mean_d > 0.2 * session.voxel_size_mm * 1000.0, \
        "widening a vessel by 60% should visibly move its surface"
