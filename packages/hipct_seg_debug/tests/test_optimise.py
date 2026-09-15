"""Ordering and radius cleanup — and where the ordering actually ends up.

The regression these guard against is a quiet one. ``_to_spatial`` *builds* a
``SpatialGraph`` from a ``Triple`` rather than returning a view, so
``_to_spatial(graph).edge_attrs[name] = ...`` writes to a temporary that is
discarded on return. ``order()` did exactly that, so ``skeletonise --order`` wrote
files with no ordering in them and ``optimise --oblique`` then died looking up a
``strahler`` field that had been computed and thrown away. Nothing failed loudly;
the flag simply had no effect.
"""

from __future__ import annotations

import numpy as np
import pytest

from hipct_seg_debug.edit.adapter import Triple, to_spatial_graph
from hipct_seg_debug.edit.graphmodel import EditableGraph
from hipct_seg_debug.edit.optimise import set_edge_field


def _chain(x0, x1, node_a, node_b, first_pid, sid):
    xs = np.arange(x0, x1 + 1, dtype=float)
    points = {first_pid + n: (float(x), 0.0, 0.0, 100.0) for n, x in enumerate(xs)}
    seg = {"id": sid, "node1": node_a, "node2": node_b, "point_ids": list(points)}
    return points, seg


@pytest.fixture
def tree():
    """A Y: one trunk splitting into two daughters."""
    points, segments, nodes = {}, [], {}
    specs = [(0, 5, 0, 1, 1000, 0), (5, 9, 1, 2, 2000, 1), (5, 9, 1, 3, 3000, 2)]
    for x0, x1, a, b, pid, sid in specs:
        pts, seg = _chain(x0, x1, a, b, pid, sid)
        points.update(pts)
        segments.append(seg)
    nodes = {0: (0.0, 0.0, 0.0, 0), 1: (5.0, 0.0, 0.0, 0),
             2: (9.0, 0.0, 0.0, 0), 3: (9.0, 1.0, 0.0, 0)}
    # Give the daughters a lateral offset so they are not coincident.
    for pid in list(points):
        if pid >= 3000:
            x, _y, z, r = points[pid]
            points[pid] = (x, 1.0, z, r)
    return Triple(nodes=nodes, points=points, segments=segments)


# ------------------------------------------------------------ set_edge_field


def test_writes_onto_a_triples_segments(tree):
    set_edge_field(tree, "strahler", [3, 1, 2], np.int64)
    assert [seg["strahler"] for seg in tree.segments] == [3, 1, 2]
    assert tree.edge_attr_dtypes["strahler"] == np.dtype(np.int64)


def test_values_are_plain_python_scalars(tree):
    """`to_spatial_graph` rebuilds a typed column from these; numpy scalars leak
    their dtype into the written file."""
    set_edge_field(tree, "strahler", np.array([3, 1, 2]), np.int64)
    assert all(isinstance(seg["strahler"], int) for seg in tree.segments)


def test_survives_the_round_trip_a_write_performs(tree):
    set_edge_field(tree, "strahler", [3, 1, 2], np.int64)
    sg = to_spatial_graph(tree)
    assert "strahler" in sg.edge_attrs
    assert sg.edge_attrs["strahler"].dtype.kind == "i"
    assert list(sg.edge_attrs["strahler"]) == [3, 1, 2]


def test_overwrite_false_leaves_an_existing_field_alone(tree):
    set_edge_field(tree, "MeanRadius", [1.0, 2.0, 3.0])
    set_edge_field(tree, "MeanRadius", [9.0, 9.0, 9.0], overwrite=False)
    assert [seg["MeanRadius"] for seg in tree.segments] == [1.0, 2.0, 3.0]


def test_overwrite_false_still_writes_when_absent(tree):
    set_edge_field(tree, "MeanRadius", [1.0, 2.0, 3.0], overwrite=False)
    assert [seg["MeanRadius"] for seg in tree.segments] == [1.0, 2.0, 3.0]


def test_reaches_the_triple_behind_an_editable_graph(tree):
    graph = EditableGraph(tree)
    set_edge_field(graph, "strahler", [3, 1, 2], np.int64)
    assert [seg["strahler"] for seg in graph.triple.segments] == [3, 1, 2]


def test_mutates_a_spatial_graph_in_place(tree):
    sg = to_spatial_graph(tree)
    set_edge_field(sg, "strahler", [3, 1, 2], np.int64)
    assert list(sg.edge_attrs["strahler"]) == [3, 1, 2]


def test_wrong_length_is_refused(tree):
    with pytest.raises(ValueError, match="2 values for 3 segments"):
        set_edge_field(tree, "strahler", [1, 2], np.int64)


def test_a_graph_it_cannot_handle_is_refused():
    with pytest.raises(TypeError):
        set_edge_field(object(), "strahler", [1], np.int64)


# -------------------------------------------------------------------- order


def _skeleton_analysis_or_skip():
    try:
        from hipct_seg_debug.edit._deps import ensure_skeleton_analysis

        ensure_skeleton_analysis()
    except Exception as exc:  # noqa: BLE001 - optional dependency
        pytest.skip(f"skeleton_analysis is not importable: {exc}")


def test_order_persists_onto_the_graph_it_was_given(tree):
    """The regression. ``order()`` used to report an ordering it had discarded."""
    _skeleton_analysis_or_skip()
    from hipct_seg_debug.edit.optimise import order

    report = order(tree)
    assert report.n_roots >= 1

    assert all("strahler" in seg for seg in tree.segments), \
        "order() computed a Strahler order and did not keep it"
    assert all("topo" in seg for seg in tree.segments)
    assert all("MeanRadius" in seg for seg in tree.segments)

    # ...and it is still there after the conversion that `--out` goes through.
    sg = to_spatial_graph(tree)
    assert "strahler" in sg.edge_attrs
    assert list(sg.edge_attrs["strahler"]) == [seg["strahler"] for seg in tree.segments]


def test_order_gives_the_trunk_the_highest_strahler(tree):
    _skeleton_analysis_or_skip()
    from hipct_seg_debug.edit.optimise import order

    order(tree)
    trunk = next(seg for seg in tree.segments if seg["id"] == 0)
    daughters = [seg for seg in tree.segments if seg["id"] in (1, 2)]
    assert trunk["strahler"] >= max(d["strahler"] for d in daughters)


def test_order_reaches_an_editable_graphs_triple(tree):
    _skeleton_analysis_or_skip()
    from hipct_seg_debug.edit.optimise import order

    graph = EditableGraph(tree)
    order(graph)
    assert all("strahler" in seg for seg in graph.triple.segments)
