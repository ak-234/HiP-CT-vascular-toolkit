"""Several skeletons loaded as one editable graph, each held as its own tree.

The Data tab's `skeleton (.am)` field takes several paths. They are **merged**, not
drawn side by side: picking, the edit tools, `crop`, `reformat` and the writer are all
defined against a single `session.graph`, so a second graph beside it would have to be
read-only to stay honest. Merged, all of them are editable, and the `tree` field --
the same one `--per-tree` skeletonisation and per-tree scoring already use -- is what
keeps them apart.

What is pinned: nothing is lost in the merge, ids stay unique, each source lands on
its own tree, a graph that already carries trees is not flattened, and a single path
still loads byte-for-byte as it did.
"""

from __future__ import annotations

import numpy as np
import pytest

from hipct_seg_debug.edit import components as comp
from hipct_seg_debug.edit.adapter import from_spatial_graph, to_spatial_graph
from hipct_seg_debug.edit.amira_write import write_spatial_graph
from hipct_seg_debug.main import graph_paths, load_graphs

from .conftest_geometry import graph_from


def one_tree(offset=0.0):
    """A single Y: three segments, one component."""
    return graph_from(
        [(0 + offset, 0, 0), (400 + offset, 0, 0),
         (800 + offset, 300, 0), (800 + offset, -300, 0)],
        [(0, 1, 5, 90.0), (1, 2, 5, 50.0), (1, 3, 5, 45.0)],
    )


def two_trees(offset=0.0):
    """Two disjoint Ys already tagged as trees 0 and 1, as `--per-tree` writes them."""
    graph = graph_from(
        [(0 + offset, 0, 0), (400 + offset, 0, 0),
         (800 + offset, 300, 0), (800 + offset, -300, 0),
         (0 + offset, 9000, 0), (400 + offset, 9000, 0),
         (800 + offset, 9300, 0), (800 + offset, 8700, 0)],
        [(0, 1, 5, 90.0), (1, 2, 5, 50.0), (1, 3, 5, 45.0),
         (4, 5, 5, 60.0), (5, 6, 5, 30.0), (5, 7, 5, 28.0)],
    )
    for seg in graph.segments:
        seg[comp.TREE_FIELD] = 0 if seg["id"] < 3 else 1
    graph.triple.edge_attr_dtypes[comp.TREE_FIELD] = np.dtype(np.int64)
    return graph


def written(tmp_path, name, graph):
    path = tmp_path / name
    write_spatial_graph(to_spatial_graph(graph.triple), path)
    return path


# --------------------------------------------------------------- the field itself


@pytest.mark.parametrize("value,expected", [
    ("one.am", ["one.am"]),
    (["one.am", "two.am"], ["one.am", "two.am"]),
    ("left.am; right.am", ["left.am", "right.am"]),   # the Data tab's field
    ("  ;  ", []),
    (None, []),
])
def test_the_skeleton_field_normalises_however_it_arrives(value, expected):
    """A list from the CLI, ';'-separated from the panel, a bare string from a
    session saved before either. All three mean the same thing."""
    assert graph_paths(value) == expected


# ------------------------------------------------------------------- the merge


def test_a_single_skeleton_is_returned_untouched(tmp_path):
    """Not merged, so an unedited graph still round trips through the writer."""
    path = written(tmp_path, "one.am", one_tree())

    graph, trees = load_graphs([str(path)])

    assert (graph.n_vertex, graph.n_edge) == (4, 3)
    assert trees == [[0]]
    assert comp.TREE_FIELD not in graph.edge_attrs, "no tree invented for one graph"


def test_two_skeletons_become_one_graph_of_two_trees(tmp_path):
    left = written(tmp_path, "left.am", one_tree())
    right = written(tmp_path, "right.am", one_tree(offset=9000.0))

    graph, trees = load_graphs([str(left), str(right)])

    assert trees == [[0], [1]]
    assert graph.n_edge == 6 and graph.n_vertex == 8, "nothing lost in the merge"
    triple = from_spatial_graph(graph)
    assert comp.tree_indices(triple) == [0, 1]


def test_the_merged_graph_is_editable_not_a_backdrop(tmp_path):
    """The whole point: one `EditableGraph` reaching every segment of every source."""
    from hipct_seg_debug.edit.graphmodel import EditableGraph

    left = written(tmp_path, "left.am", one_tree())
    right = written(tmp_path, "right.am", one_tree(offset=9000.0))
    graph, _trees = load_graphs([str(left), str(right)])

    editable = EditableGraph(from_spatial_graph(graph))

    assert len(editable.segments) == 6
    assert len(editable.components()) == 2, "two trees, both reachable"
    # And an edit lands, on the second source as readily as the first.
    editable.delete_segment(editable.segments[-1]["id"])
    assert len(editable.segments) == 5


def test_ids_are_unique_across_the_sources(tmp_path):
    """Two parts sharing a segment id makes one of them silently disappear."""
    left = written(tmp_path, "left.am", one_tree())
    right = written(tmp_path, "right.am", one_tree(offset=9000.0))

    triple = from_spatial_graph(load_graphs([str(left), str(right)])[0])

    for field in ("id", "node1", "node2"):
        assert len({s["id"] for s in triple.segments}) == len(triple.segments)
    assert len(triple.nodes) == 8 and len(triple.points) == 30


def test_a_graph_that_already_has_trees_is_not_flattened(tmp_path):
    """A `--per-tree` skeleton loaded beside a single-tree one gives three trees."""
    both = written(tmp_path, "both.am", two_trees())
    extra = written(tmp_path, "extra.am", one_tree(offset=20000.0))

    graph, trees = load_graphs([str(both), str(extra)])

    assert trees == [[0, 1], [2]]
    assert comp.tree_indices(from_spatial_graph(graph)) == [0, 1, 2]


def test_the_trees_are_the_ones_every_per_tree_tool_already_reads(tmp_path):
    """`tree` is not a new field: crop, per-tree scoring and the roots sidecar use it."""
    left = written(tmp_path, "left.am", one_tree())
    right = written(tmp_path, "right.am", one_tree(offset=9000.0))

    from hipct_seg_debug.edit.graphmodel import EditableGraph

    merged = EditableGraph(from_spatial_graph(load_graphs([str(left), str(right)])[0]))

    of_edge = comp.tree_of_edge(merged)
    assert sorted(of_edge.values()) == [0, 0, 0, 1, 1, 1]
    assert len(comp.subgraph_by_tree(merged, 1).segments) == 3


def test_merging_survives_a_write_and_a_re_read(tmp_path):
    """The merged graph is what gets saved, so the trees have to reach the file."""
    left = written(tmp_path, "left.am", one_tree())
    right = written(tmp_path, "right.am", one_tree(offset=9000.0))
    graph, _trees = load_graphs([str(left), str(right)])

    out = tmp_path / "merged.am"
    write_spatial_graph(graph, out)

    from hipct_seg_debug.amira import read_spatial_graph

    back = from_spatial_graph(read_spatial_graph(out))
    assert comp.tree_indices(back) == [0, 1]
    assert len(back.segments) == 6


# ------------------------------------------------- the commands that write them
#
# Every `edit` subcommand that processes a skeleton takes several now, through the
# same `graph_arg` positional, and `_load` merges them. What is pinned here is that
# the merge reaches the *output*: one file, both trees, nothing dropped.


def test_every_graph_command_accepts_several(tmp_path):
    from hipct_seg_debug.edit.__main__ import build_parser

    parser = build_parser()
    for command in ("report", "gaps", "optimise", "optimise-skeleton", "crop",
                    "radius-perimeter", "score", "surface", "flag-interpolation"):
        args = parser.parse_args([command, "left.am", "right.am"])
        assert args.graph == ["left.am", "right.am"], command


def test_the_cli_loader_merges_and_says_the_ids_moved(tmp_path, capsys):
    from hipct_seg_debug.edit.__main__ import _load

    left = written(tmp_path, "left.am", one_tree())
    right = written(tmp_path, "right.am", one_tree(offset=9000.0))

    graph = _load([str(left), str(right)])

    assert len(graph.segments) == 6
    assert comp.tree_indices(graph) == [0, 1]
    out = capsys.readouterr().out
    assert "merged 2 skeletons" in out
    assert "ids are renumbered" in out, "--segment means something else afterwards"


def test_one_graph_is_not_merged_and_gains_no_tree_field(tmp_path):
    """The usual case has to stay exactly as it was, ids included."""
    from hipct_seg_debug.edit.__main__ import _load

    graph = _load([str(written(tmp_path, "one.am", one_tree()))])

    assert comp.tree_indices(graph) == []
    assert [s["id"] for s in graph.segments] == [0, 1, 2]


def test_a_command_writes_one_output_holding_every_tree(tmp_path):
    from hipct_seg_debug.amira import read_spatial_graph
    from hipct_seg_debug.edit.__main__ import build_parser, cmd_flag_interpolation

    left = written(tmp_path, "left.am", one_tree())
    right = written(tmp_path, "right.am", one_tree(offset=9000.0))
    out = tmp_path / "flagged.am"

    args = build_parser().parse_args(
        ["flag-interpolation", str(left), str(right), "--out", str(out)]
    )
    assert cmd_flag_interpolation(args) == 0

    back = from_spatial_graph(read_spatial_graph(out))
    assert len(back.segments) == 6
    assert comp.tree_indices(back) == [0, 1], "both trees reached the file"


def test_the_form_keeps_a_path_with_spaces_in_one_piece():
    """A repeatable *path* field splits on ';', never on whitespace."""
    from hipct_seg_debug import cliform

    spec = {s.name: s for s in cliform.describe_parser(
        __import__("hipct_seg_debug.edit.__main__", fromlist=["build_parser"])
        .build_parser())}["report"]
    field = spec.field("graph")

    assert cliform.split_repeat(field, r"D:\data dir\a b.am; C:\x y\c.am") == [
        r"D:\data dir\a b.am", r"C:\x y\c.am"
    ]
    # And a bare string is one value, not four characters.
    assert cliform.repeat_values(field, "g.am") == ["g.am"]
