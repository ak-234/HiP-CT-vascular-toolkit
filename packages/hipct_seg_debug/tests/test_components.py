"""Splitting a mask into trees, and putting the skeletons back together.

The two things that can go silently wrong here are both geometric, and both produce a
graph that looks perfectly well-formed:

* the sub-volume origin, which places every coordinate of a tree in world space. Get
  it wrong by a stride factor and the skeleton lands somewhere else entirely, which is
  exactly what happened when a stride was applied twice in ``skeletonisers``;
* the bounding-box crop, which decides *which* voxels belong to a tree. Threshold
  instead of label and a neighbouring component is skeletonised in both boxes.

So the tests below check coordinates against the frame and voxel counts against the
labelling, rather than checking that the code ran.
"""

from __future__ import annotations

import numpy as np
import pytest

from hipct_seg_debug.edit import components as comp
from hipct_seg_debug.edit.adapter import Triple, read_triple, to_spatial_graph
from hipct_seg_debug.edit.amira_write import write_spatial_graph
from hipct_seg_debug.edit.graphmodel import EditableGraph
from hipct_seg_debug.edit.reconnect.segmentation import components as label_components

from .conftest_geometry import (
    SPACING,cylinder, graph_from, interleaved_components, make_frame, two_cylinders,
)

SHAPE = (30, 30, 60)


@pytest.fixture
def frame():
    return make_frame(SHAPE)


@pytest.fixture
def pair():
    return two_cylinders(SHAPE)


# ------------------------------------------------------------- split_components


def test_two_tubes_split_into_two_parts_largest_first(pair, frame):
    parts, stats = comp.split_components(pair, frame)

    assert stats.n == 2
    assert [p.index for p in parts] == [0, 1]
    assert parts[0].voxels > parts[1].voxels, "tree 0 is the largest component"
    assert sum(p.voxels for p in parts) == int((pair > 0).sum())


def test_each_part_holds_exactly_its_own_component(pair, frame):
    parts, stats = comp.split_components(pair, frame)
    for part in parts:
        assert int(part.volume.sum()) == part.voxels == int(stats.sizes[part.label])


def test_a_bounding_box_crop_does_not_admit_the_neighbouring_component(frame):
    """The label-vs-threshold crop, on two components with overlapping boxes.

    ``volume[box] > 0`` would put part of the other component in this part's volume,
    and it would then be skeletonised once here and once in its own box.
    """
    volume = interleaved_components(SHAPE)
    parts, stats = comp.split_components(volume, frame)
    assert len(parts) == 2

    # The fixture is only meaningful if the boxes really do overlap.
    (_az0, ay0, ax0), (_bz0, by0, bx0) = parts[0].origin_zyx, parts[1].origin_zyx
    az1, ay1, ax1 = np.array(parts[0].origin_zyx) + parts[0].volume.shape
    bz1, by1, bx1 = np.array(parts[1].origin_zyx) + parts[1].volume.shape
    assert ax0 < bx1 and bx0 < ax1 and ay0 < by1 and by0 < ay1, "boxes must overlap"

    for part in parts:
        assert int(part.volume.sum()) == part.voxels

    # And between them they account for the mask exactly once.
    total = np.zeros(SHAPE, dtype=np.int64)
    for part in parts:
        z, y, x = part.origin_zyx
        nz, ny, nx = part.volume.shape
        total[z:z + nz, y:y + ny, x:x + nx] += part.volume
    assert total.max() == 1, "no voxel belongs to two parts"
    assert int(total.sum()) == int((volume > 0).sum())


@pytest.mark.parametrize("stride", [1, 2, 4])
def test_the_part_origin_is_the_frame_origin_with_no_stride_applied_twice(stride):
    """The bug that put every coordinate eight times too far out, guarded directly.

    ``origin_zyx`` indexes the *already decoded* grid, whose pitch is
    ``frame.seg_spacing`` -- which already carries the stride. Multiplying by the
    stride again here would scale the offset a second time.
    """
    shape = (SHAPE[0] // stride, SHAPE[1] // stride, SHAPE[2] // stride)
    frame = make_frame(shape)
    # A frame decoded at `stride` has spacing already multiplied, exactly as `_decoded`
    # builds it; `make_frame` gives spacing SPACING, so scale to mimic that.
    volume = two_cylinders(shape)

    parts, _ = comp.split_components(volume, frame)
    for part in parts:
        i, j, k = part.origin_zyx[2], part.origin_zyx[1], part.origin_zyx[0]
        expected = frame.seg_to_um([[i, j, k]])[0]
        assert np.allclose(part.origin_um, expected)
        # The offset is one spacing per voxel -- never two.
        assert np.allclose(part.origin_um - frame.seg_origin,
                           np.array([i, j, k]) * frame.seg_spacing)


def test_min_voxels_drops_the_small_tree_and_max_trees_caps_the_count(pair, frame):
    parts, stats = comp.split_components(pair, frame)
    small = min(p.voxels for p in parts)

    kept, _ = comp.split_components(pair, frame, min_voxels=small + 1)
    assert len(kept) == 1
    assert kept[0].voxels > small

    capped, _ = comp.split_components(pair, frame, max_trees=1)
    assert len(capped) == 1
    assert capped[0].index == 0


def test_padding_leaves_a_background_border(pair, frame):
    """Lee thinning curls toward a cut face, so the box is never drawn tight."""
    parts, _ = comp.split_components(pair, frame, pad=1)
    for part in parts:
        vol = part.volume
        # Interior only -- a component touching the volume edge is clipped there.
        z, y, x = part.origin_zyx
        nz, ny, nx = vol.shape
        if z > 0:
            assert not vol[0].any()
        if y > 0:
            assert not vol[:, 0].any()
        if x > 0:
            assert not vol[:, :, 0].any()


def test_an_empty_volume_yields_no_parts(frame):
    parts, stats = comp.split_components(np.zeros(SHAPE, dtype=np.uint8), frame)
    assert parts == []
    assert stats.n == 0


# ---------------------------------------------------------------- merge_triples


def _line_triple(x0, x1, *, y, radius=30.0, base=0):
    """A single straight segment, with ids deliberately not starting at zero."""
    xs = np.arange(x0, x1, dtype=float) * SPACING
    points = {base + i: (float(x), float(y), 0.0, radius) for i, x in enumerate(xs)}
    nodes = {base: (float(xs[0]), float(y), 0.0, 1),
             base + 1: (float(xs[-1]), float(y), 0.0, 1)}
    segs = [{"id": base, "node1": base, "node2": base + 1,
             "point_ids": sorted(points)}]
    return Triple(nodes=nodes, points=points, segments=segs)


def test_merging_renumbers_ids_and_tags_every_edge():
    a = _line_triple(0, 5, y=0.0, base=0)
    b = _line_triple(0, 4, y=500.0, base=100)  # colliding id space on purpose

    merged = comp.merge_triples([(0, a), (1, b)])

    ids = [s["id"] for s in merged.segments]
    assert ids == list(range(len(ids))), "contiguous from zero"
    assert len(set(ids)) == len(ids), "unique -- a clash makes a segment disappear"
    assert [s[comp.TREE_FIELD] for s in merged.segments] == [0, 1]
    assert len(merged.points) == len(a.points) + len(b.points)
    assert merged.edge_attr_dtypes[comp.TREE_FIELD].kind in "iu"


def test_a_merged_segment_never_references_another_parts_node():
    a = _line_triple(0, 5, y=0.0, base=0)
    b = _line_triple(0, 4, y=500.0, base=100)
    merged = comp.merge_triples([(0, a), (1, b)])

    graph = EditableGraph(merged)
    assert len(graph.segments) == 2, "both survived the renumbering"
    assert len(graph.components()) == 2
    for seg in merged.segments:
        for key in ("node1", "node2"):
            assert seg[key] in merged.nodes
        for pid in seg["point_ids"]:
            assert pid in merged.points


def test_merging_tolerates_a_part_that_traced_to_nothing():
    a = _line_triple(0, 5, y=0.0, base=0)
    empty = Triple(nodes={}, points={}, segments=[])
    merged = comp.merge_triples([(0, a), (1, empty), (2, None)])

    assert [s["id"] for s in merged.segments] == [0]
    assert [s[comp.TREE_FIELD] for s in merged.segments] == [0]


def test_the_tree_field_survives_a_write_and_read(tmp_path):
    a = _line_triple(0, 5, y=0.0, base=0)
    b = _line_triple(0, 4, y=500.0, base=100)
    merged = comp.merge_triples([(0, a), (1, b)])

    dest = tmp_path / "merged.am"
    write_spatial_graph(to_spatial_graph(merged), dest)
    back = read_triple(dest)

    assert [int(s[comp.TREE_FIELD]) for s in back.segments] == [0, 1]


# ------------------------------------------------- reading the field back again


def _two_tree_graph():
    a = _line_triple(0, 5, y=0.0, base=0)
    b = _line_triple(0, 4, y=500.0, base=100)
    return EditableGraph(comp.merge_triples([(0, a), (1, b)]))


def test_tree_helpers_read_what_the_merge_wrote():
    graph = _two_tree_graph()
    assert comp.tree_indices(graph) == [0, 1]
    assert set(comp.tree_of_edge(graph).values()) == {0, 1}

    one = comp.subgraph_by_tree(graph, 1)
    assert len(one.segments) == 1
    assert comp.tree_indices(one) == [1]
    # Ids are preserved, so anything already resolved still names the same segment.
    assert one.segments[0]["id"] == graph.segments[1]["id"]


def test_tree_helpers_are_empty_on_a_graph_that_carries_no_field():
    graph = EditableGraph(_line_triple(0, 5, y=0.0))
    assert comp.tree_of_edge(graph) == {}
    assert comp.tree_indices(graph) == []


# --------------------------------------------------------- root_edge_for_node


def test_a_free_end_names_its_one_segment_unambiguously():
    graph = graph_from(
        [(0, 0, 0), (100, 0, 0), (200, 100, 0), (200, -100, 0)],
        [(0, 1, 3, 40.0), (1, 2, 3, 20.0), (1, 3, 3, 20.0)],
    )
    sid, ambiguous = comp.root_edge_for_node(graph, 0)
    assert sid == 0
    assert ambiguous is False


def test_a_junction_root_takes_the_thickest_edge_and_says_it_is_ambiguous():
    graph = graph_from(
        [(0, 0, 0), (100, 0, 0), (200, 100, 0), (200, -100, 0)],
        [(0, 1, 3, 40.0), (1, 2, 3, 20.0), (1, 3, 3, 30.0)],
    )
    sid, ambiguous = comp.root_edge_for_node(graph, 1)
    assert sid == 0, "the 40 um trunk"
    assert ambiguous is True


def test_an_isolated_node_cannot_root_a_tree():
    graph = graph_from([(0, 0, 0), (100, 0, 0)], [(0, 1, 3, 40.0)])
    graph.triple.nodes[9] = (500.0, 500.0, 0.0, 0)
    with pytest.raises(ValueError, match="no incident segment"):
        comp.root_edge_for_node(graph, 9)


# ------------------------------------------------------------------ assign_trees


def test_edges_are_assigned_to_the_mask_component_they_lie_in(frame):
    """The ``amira`` path: a graph that arrived without a per-component run."""
    volume = two_cylinders(SHAPE)
    stats = label_components(volume > 0)
    order = [int(v) for v in stats.order_by_size()]

    # One segment down each tube's axis, at the centres `two_cylinders` used.
    nz, ny, nx = SHAPE
    big = frame.seg_to_um(np.stack(
        [np.arange(6, nx - 6), np.full(nx - 12, ny // 4), np.full(nx - 12, nz // 2)], 1))
    small = frame.seg_to_um(np.stack(
        [np.arange(nx // 2 + 1, nx - 9), np.full(nx - 10 - nx // 2, 3 * ny // 4),
         np.full(nx - 10 - nx // 2, nz // 2)], 1))
    graph = graph_from(
        [big[0], big[-1], small[0], small[-1]],
        [(0, 1, len(big), 40.0), (2, 3, len(small), 20.0)],
    )

    trees = comp.assign_trees(graph, stats.labels, frame, order=order)

    assert trees.tolist() == [0, 1], "the big tube is tree 0, the small one tree 1"


# ----------------------------------------------- the whole per-component run


def test_a_per_component_skeleton_lands_where_its_own_tube_is(frame, pair):
    """The test an eight-times-too-far-out origin cannot pass.

    Every coordinate must lie inside the frame, *and* each tree's points must lie
    inside the tube that tree was derived from -- a global bbox check alone would
    pass even if the two trees had been swapped or collapsed onto each other.
    """
    from hipct_seg_debug.edit import skeletonisers as sk

    cand = sk.skeletonise_per_component("lee", pair, frame)
    graph = EditableGraph(cand.triple)

    assert cand.trees == 2
    assert comp.tree_indices(graph) == [0, 1]

    x0, x1, y0, y1, z0, z1 = frame.seg_bbox_um
    nz, ny, nx = SHAPE
    # `two_cylinders` puts the big tube at ny//4 and the small one at 3*ny//4.
    expected_y = {0: (ny // 4) * SPACING, 1: (3 * ny // 4) * SPACING}

    for tree, want_y in expected_y.items():
        pts = np.vstack([comp.subgraph_by_tree(graph, tree).coords(s)
                         for s in comp.subgraph_by_tree(graph, tree).segment_ids()])
        assert pts[:, 0].min() >= x0 - SPACING and pts[:, 0].max() <= x1 + SPACING
        assert pts[:, 2].min() >= z0 - SPACING and pts[:, 2].max() <= z1 + SPACING
        assert np.allclose(pts[:, 1], want_y, atol=SPACING), (
            f"tree {tree} should run along y={want_y}"
        )


def test_the_per_component_centreline_stays_inside_the_mask(frame, pair):
    from hipct_seg_debug.edit import skeletonisers as sk
    from hipct_seg_debug.edit.supermetric import cl_sensitivity

    cand = sk.skeletonise_per_component("lee", pair, frame)
    graph = EditableGraph(cand.triple)

    assert cl_sensitivity(graph, frame, pair) > 0.9


def test_dropping_the_small_component_leaves_one_tree(frame, pair):
    from hipct_seg_debug.edit import skeletonisers as sk

    cand = sk.skeletonise_per_component("lee", pair, frame, min_component_voxels=1000)

    assert cand.trees == 1
    assert comp.tree_indices(EditableGraph(cand.triple)) == [0]


def test_an_empty_volume_produces_an_empty_candidate_rather_than_raising(frame):
    from hipct_seg_debug.edit import skeletonisers as sk

    cand = sk.skeletonise_per_component("lee", np.zeros(SHAPE, np.uint8), frame)

    assert cand.trees == 0
    assert cand.triple.segments == []


def test_a_component_that_touches_no_kept_label_is_left_unplaced(frame):
    volume = cylinder(SHAPE, 4, 5, 55)
    stats = label_components(volume > 0)
    # A segment far outside the mask entirely.
    graph = graph_from([(9e4, 9e4, 9e4), (9.1e4, 9e4, 9e4)], [(0, 1, 5, 10.0)])

    trees = comp.assign_trees(graph, stats.labels, frame,
                              order=[int(v) for v in stats.order_by_size()])

    assert trees.tolist() == [-1], "guessed into no tree at all"


# ------------------------------------------------------------ the scoring scope


@pytest.mark.parametrize("argv, expected", [
    (["score", "g.am"], "whole"),
    (["score", "g.am", "--scope", "whole"], "whole"),
    (["score", "g.am", "--scope", "per-tree"], "per-tree"),
    (["score", "g.am", "--scope", "largest"], "largest"),
    (["optimise-skeleton", "g.am"], "whole"),
    (["optimise-skeleton", "g.am", "--scope", "largest"], "largest"),
    # On `skeletonise-all`, splitting the skeleton carries the scope with it unless
    # the scope is named -- otherwise a per-tree skeleton would be ranked on the
    # largest tree's chi, which is the blind spot --per-tree exists to close.
    (["skeletonise-all"], "whole"),
    (["skeletonise-all", "--per-tree"], "per-tree"),
    (["skeletonise-all", "--per-tree", "--scope", "whole"], "whole"),
    (["skeletonise-all", "--per-tree", "--scope", "largest"], "largest"),
])
def test_the_scoring_scope_resolves_from_the_flags(argv, expected):
    from hipct_seg_debug.edit.__main__ import _resolve_scope, build_parser

    assert _resolve_scope(build_parser().parse_args(argv)) == expected


def test_largest_scope_scores_exactly_what_per_tree_calls_tree_zero(frame, pair):
    """`largest` is `per-tree` capped at one, so the numbers must agree exactly."""
    from hipct_seg_debug.edit import skeletonisers as sk
    from hipct_seg_debug.edit import supermetric as sm
    from hipct_seg_debug.edit.reconnect.segmentation import components as label_components

    cand = sk.skeletonise_per_component("lee", pair, frame)
    graph = EditableGraph(cand.triple)
    stats = label_components(pair > 0)

    every, _ = comp.split_components(pair, frame)
    just_one, _ = comp.split_components(pair, frame, max_trees=1)

    all_trees = sm.super_metric_per_tree(graph, frame, stats.labels, every)
    largest = sm.super_metric_per_tree(graph, frame, stats.labels, just_one)

    assert sorted(all_trees) == [0, 1]
    assert sorted(largest) == [0], "capped to the biggest component"
    assert largest[0].total == all_trees[0].total


def test_max_trees_is_reported_apart_from_the_voxel_threshold(frame, pair, capsys):
    """A tree set aside by the cap is not a tree the threshold ate."""
    comp.split_components(pair, frame, min_voxels=100, max_trees=1, verbose=True)

    out = capsys.readouterr().out
    assert "set aside 1 component(s) beyond the 1 kept" in out
    # The small cylinder is ~286 voxels, well above the 100 threshold.
    assert "286" in out
