import numpy as np

from hipct_seg_debug.edit import contained_leaves as cl
from .conftest_geometry import graph_from, make_frame, slit

SHAPE = (40, 80, 120)  # (nz, ny, nx), 10 um voxels


def flattened_vessel():
    """A host along one edge of a wide flat slit, a spur across it, a real side branch.

    Segment 2 runs across the slit (y 25 -> 55) inside the lumen: the 3655/794 case.
    Segment 4 leaves through its own tube (z 15 -> 34): a genuine branch.
    Segments 0, 1, 3 are the host, split at the two junctions.
    """
    frame = make_frame(SHAPE)
    mask = slit(SHAPE, 20, 4, 5, 115, cy=40, cz=15)
    zz, yy, xx = np.ogrid[:SHAPE[0], :SHAPE[1], :SHAPE[2]]
    mask[((xx - 90) ** 2 + (yy - 25) ** 2 <= 9) & (zz >= 15) & (zz < 36)] = 1
    vox = [(10, 25, 15), (60, 25, 15), (90, 25, 15), (110, 25, 15), (60, 55, 15), (90, 25, 34)]
    xyz = frame.seg_to_um(np.asarray(vox, float))
    g = graph_from(xyz, [(0, 1, 26, 100.), (1, 2, 16, 100.), (1, 4, 16, 50.),
                         (2, 3, 11, 100.), (2, 5, 11, 50.)])
    return g, frame, mask


def rows_by_segment(g, frame, mask, **kw):
    return {r["segment"]: r for r in cl.find_contained_leaves(g, mask, frame, **kw)}


def test_spur_across_a_flattened_lumen_is_contained_and_a_real_branch_is_not():
    g, frame, mask = flattened_vessel()
    rows = rows_by_segment(g, frame, mask)
    assert set(rows) == {0, 2, 3, 4}  # every degree-1 segment; 0 and 3 are the host's ends
    assert rows[2]["contained"] and rows[2]["fraction_inside"] >= .8
    # Square to the host and joined to its lumen at the junction, but outside the
    # host's footprint once clear of it: a real branch.
    assert not rows[4]["contained"] and rows[4]["fraction_inside"] < .5
    assert not rows[0]["contained"] and not rows[3]["contained"]


def test_pruning_removes_only_the_spur_and_keeps_ids():
    g, frame, mask = flattened_vessel()
    before = set(g.segment_ids())
    out = cl.prune_contained_leaves(g, cl.find_contained_leaves(g, mask, frame))
    assert out["removed"] == [2]
    assert set(g.segment_ids()) == before - {2}  # no rejoin, so no renumbering
    g.undo()
    assert set(g.segment_ids()) == before


def test_root_end_and_higher_order_leaves_are_protected():
    g, frame, mask = flattened_vessel()
    free = 4  # node index of the spur's free end
    assert rows_by_segment(g, frame, mask, protected_nodes={free})[2]["protected"] == "root"
    g.set_segment_attrs(2, {"strahler": 3})
    row = rows_by_segment(g, frame, mask)[2]
    assert not row["contained"] and row["protected"] == "strahler 3"
