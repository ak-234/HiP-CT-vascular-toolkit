"""Recording which node roots each tree, and finding it again later.

The sidecar exists to survive things that destroy ids: a rewrite, an edit, and above
all a **re-skeletonisation**, after which the segments are new objects in roughly the
same place. So the tests here mostly attack the resolution path -- exact key, snapped
coordinate, and the give-up case -- rather than the writing.

The failure that matters is a root placed on the *wrong* node. It reverses parent and
child through a whole subtree and nothing about the output looks wrong, so
:func:`~.roots.pick` cross-checks every vertex-to-node translation against the
coordinate and raises instead of guessing.
"""

from __future__ import annotations

import json

from pathlib import Path

import numpy as np
import pytest

from hipct_seg_debug.amira import read_spatial_graph
from hipct_seg_debug.edit import components as comp
from hipct_seg_debug.edit import roots as roots_mod
from hipct_seg_debug.edit.adapter import Triple, to_spatial_graph, vertex_node_ids
from hipct_seg_debug.edit.graphmodel import EditableGraph

import types

from .conftest_geometry import graph_from, make_frame

SHAPE = (30, 30, 60)


@pytest.fixture
def frame():
    return make_frame(SHAPE)


def two_trees(shift=(0.0, 0.0, 0.0)) -> EditableGraph:
    """Two disjoint Y's, a big one and a small one -- a left and a right tree.

    `shift` moves the whole graph, which moves its *points* and therefore changes
    every segment key -- the way a re-skeletonisation does, and unlike merely editing
    the node table.
    """
    dx, dy, dz = shift
    return graph_from(
        [(0 + dx, 0 + dy, 0 + dz), (400 + dx, 0 + dy, 0 + dz),
         (800 + dx, 300 + dy, 0 + dz), (800 + dx, -300 + dy, 0 + dz),      # tree A
         (0 + dx, 2000 + dy, 0 + dz), (300 + dx, 2000 + dy, 0 + dz),
         (600 + dx, 2200 + dy, 0 + dz), (600 + dx, 1800 + dy, 0 + dz)],    # tree B
        [(0, 1, 5, 90.0), (1, 2, 5, 50.0), (1, 3, 5, 45.0),
         (4, 5, 5, 60.0), (5, 6, 5, 30.0), (5, 7, 5, 28.0)],
    )


def tagged_two_trees(shift=(0.0, 0.0, 0.0)) -> EditableGraph:
    """The same graph, carrying the ``tree`` field a per-component run would write."""
    graph = two_trees(shift)
    for seg in graph.segments:
        seg[comp.TREE_FIELD] = 0 if seg["id"] < 3 else 1
    graph.triple.edge_attr_dtypes[comp.TREE_FIELD] = np.dtype(np.int64)
    return graph


# ------------------------------------------------------- the vertex-to-node map


def test_vertex_node_ids_inverts_the_spatial_graph_renumbering():
    graph = two_trees()
    triple = graph.triple
    sa = to_spatial_graph(triple)
    keep = vertex_node_ids(triple)

    assert len(keep) == sa.n_vertex
    for i, nid in enumerate(keep):
        assert np.allclose(sa.vertices[i], triple.nodes[nid][:3])


def test_vertex_node_ids_skips_the_isolated_node_that_the_writer_drops():
    """An isolated node is dropped on the way out, so it must not consume an index."""
    graph = two_trees()
    graph.triple.nodes[999] = (9e4, 9e4, 9e4, 0)

    keep = vertex_node_ids(graph.triple)
    sa = to_spatial_graph(graph.triple)

    assert 999 not in keep
    assert len(keep) == sa.n_vertex
    for i, nid in enumerate(keep):
        assert np.allclose(sa.vertices[i], graph.triple.nodes[nid][:3])


def test_vertex_node_ids_handles_non_contiguous_node_ids():
    nodes = {5: (0.0, 0.0, 0.0, 1), 11: (100.0, 0.0, 0.0, 1)}
    points = {0: (0.0, 0.0, 0.0, 10.0), 1: (100.0, 0.0, 0.0, 10.0)}
    triple = Triple(nodes, points, [{"id": 3, "node1": 5, "node2": 11,
                                     "point_ids": [0, 1]}])

    assert vertex_node_ids(triple) == [5, 11]


# ---------------------------------------------------------------- the sidecar


def test_describe_roots_names_one_tree_per_component():
    graph = tagged_two_trees()
    records = roots_mod.describe_roots(graph, [0, 4])

    assert [r["index"] for r in records] == [0, 1]
    assert all(r["tree_source"] == "mask" for r in records)
    assert records[0]["root"]["seg_id"] == 0
    assert records[0]["root"]["ambiguous"] is False


def test_describe_roots_falls_back_to_component_order_without_a_tree_field():
    records = roots_mod.describe_roots(two_trees(), [0, 4])

    assert [r["tree_source"] for r in records] == ["graph", "graph"]


def test_a_second_root_in_one_component_is_ignored():
    """`order_forest` takes one root per component; two would silently fight."""
    records = roots_mod.describe_roots(two_trees(), [0, 2])

    assert len(records) == 1


def test_the_document_round_trips_through_a_file(tmp_path):
    graph = tagged_two_trees()
    records = roots_mod.describe_roots(graph, [0, 4])
    document = roots_mod.document(graph, records, source="graph.am")

    dest = tmp_path / "roots.json"
    roots_mod.write(dest, document)
    back = roots_mod.load(dest)

    assert back["schema"] == roots_mod.SCHEMA
    assert [r["index"] for r in back["trees"]] == [0, 1]
    assert back["source_counts"]["segments"] == len(graph.segments)


def test_a_foreign_schema_is_refused(tmp_path):
    dest = tmp_path / "other.json"
    dest.write_text(json.dumps({"schema": "hipct.crop/1"}), encoding="utf-8")

    with pytest.raises(ValueError, match="expected schema"):
        roots_mod.load(dest)


# ----------------------------------------------------------------- resolution


def _document_for(graph, nodes):
    return roots_mod.document(graph, roots_mod.describe_roots(graph, nodes))


def test_the_same_graph_resolves_back_to_the_same_nodes():
    graph = tagged_two_trees()
    document = _document_for(graph, [0, 4])

    resolved = roots_mod.resolve(graph, document)

    assert set(resolved.root_nodes) == {0, 4}
    assert set(resolved.root_edges) == {0, 3}
    assert resolved.notes == []


def test_a_moved_graph_still_snaps_to_the_nearest_node(frame):
    """A re-skeletonisation puts the free end in roughly, not exactly, the same place."""
    graph = tagged_two_trees()
    document = _document_for(graph, [0, 4])

    # Every point moves, so no segment key matches and resolution must fall through
    # to the recorded coordinate. Well inside the four-voxel snap limit.
    moved = tagged_two_trees(shift=(4.0, -4.0, 4.0))

    resolved = roots_mod.resolve(moved, document, frame=frame)

    assert set(resolved.root_nodes) == {0, 4}
    assert resolved.snapped_um and max(resolved.snapped_um) > 0.0


def test_a_root_too_far_from_any_node_is_refused_with_a_note():
    graph = tagged_two_trees()
    document = _document_for(graph, [0, 4])
    document["trees"][0]["root"]["node_um"] = [9e5, 9e5, 9e5]
    document["trees"][0]["root"]["seg_key"] = "not-a-real-key"

    resolved = roots_mod.resolve(graph, document, max_snap_um=50.0)

    assert set(resolved.root_nodes) == {4}, "the other tree still resolved"
    assert any("beyond the" in n for n in resolved.notes)


def test_resolving_against_a_graph_missing_that_segment_does_not_raise():
    graph = tagged_two_trees()
    document = _document_for(graph, [0, 4])

    stripped = tagged_two_trees()
    stripped.triple.segments = [s for s in stripped.triple.segments if s["id"] != 0]
    stripped = EditableGraph(stripped.triple)

    resolved = roots_mod.resolve(stripped, document, max_snap_um=1.0)

    assert resolved.notes, "it said what it could not place"
    assert 4 in resolved.root_nodes, "and still placed the tree it could"


# ------------------------------------------------------------ root_edges_for


class _Args:
    def __init__(self, **kw):
        self.root_edge = []
        self.roots_json = None
        self.__dict__.update(kw)


def test_an_explicit_root_edge_wins_over_the_sidecar_for_its_component(tmp_path):
    """Both naming one component must not look like the two-roots-in-one error."""
    graph = tagged_two_trees()
    dest = tmp_path / "roots.json"
    roots_mod.write(dest, _document_for(graph, [0, 4]))

    # Segment 1 is in the same component as the sidecar's root edge 0.
    edges = roots_mod.root_edges_for(
        graph, _Args(root_edge=[1], roots_json=str(dest))
    )

    assert 1 in edges, "the hand-picked edge survived"
    assert 0 not in edges, "the sidecar did not also root that component"
    # One root per component, which is what `_directed_topology` requires.
    component_of = {sid: i for i, c in enumerate(graph.components()) for sid in c}
    seen = [component_of[s] for s in edges]
    assert len(seen) == len(set(seen))


def test_without_a_sidecar_only_the_explicit_edges_come_back():
    assert roots_mod.root_edges_for(two_trees(), _Args(root_edge=[2])) == (2,)


def test_a_missing_sidecar_path_is_not_an_error():
    args = _Args(root_edge=[], roots_json="does-not-exist.json")
    assert roots_mod.root_edges_for(two_trees(), args) == ()


# ---------------------------------------------------------- the picker itself


def test_the_picker_reports_a_vertex_to_node_mismatch_rather_than_guessing(monkeypatch):
    """A root on the wrong node reverses a whole subtree and never looks wrong."""
    graph = tagged_two_trees()

    import skeleton_analysis.ordering.root_picker as rp

    monkeypatch.setattr(roots_mod, "__name__", roots_mod.__name__)  # keep module intact
    monkeypatch.setattr(rp, "pick_roots", lambda *a, **k: [10 ** 6])

    with pytest.raises(ValueError, match="no node for"):
        roots_mod.pick(graph)


def test_auto_roots_come_back_as_node_ids_this_graph_knows():
    graph = tagged_two_trees()

    nodes = roots_mod.auto(graph)

    assert len(nodes) == 2, "one per component"
    assert all(n in graph.nodes for n in nodes)
    # And they are usable as roots.
    records = roots_mod.describe_roots(graph, nodes, method="auto")
    assert [r["index"] for r in records] == [0, 1]


@pytest.mark.parametrize("style", ["contour", "tube", "lines"])
def test_the_headless_picker_renders_and_returns_the_preselected_root(tmp_path, style):
    """`off_screen` renders only the FIRST tree -- it is a capture hook, not a mode."""
    pytest.importorskip("pyvista")
    pytest.importorskip("matplotlib")
    graph = tagged_two_trees()
    shot = tmp_path / "tree0.png"

    nodes = roots_mod.pick(graph, style=style, color_by="none",
                           off_screen=True, screenshot=str(shot), preselect=0)

    assert len(nodes) == 1
    assert nodes[0] in graph.nodes
    assert shot.exists() and shot.stat().st_size > 0


def test_the_lines_style_draws_the_graph_and_nothing_else():
    """One polyline per edge, a vertex at each end, no radius consulted.

    The mesh being *the graph* rather than a sweep of it is the whole point: a tube
    per segment is what makes a 3,722-segment coronary tree redraw in steps.
    """
    pytest.importorskip("pyvista")
    import pyvista as pv

    from skeleton_analysis.ordering.root_picker import _segment_line_mesh

    points = np.array([[0.0, 0, 0], [100, 0, 0], [200, 50, 0], [300, 50, 0]])

    mesh = _segment_line_mesh(pv, points)

    assert mesh.n_points == 4, "the centreline points, and no others"
    assert mesh.n_lines == 1 and mesh.n_cells == 3, "one polyline plus two nodes"
    assert mesh.n_faces_strict == 0, "nothing is tessellated"
    assert np.allclose(mesh.points[mesh.verts.reshape(-1, 2)[:, 1]],
                       points[[0, -1]]), "the vertices are the segment's two nodes"


def test_a_single_point_segment_is_a_vertex_not_a_sphere():
    pytest.importorskip("pyvista")
    import pyvista as pv

    from skeleton_analysis.ordering.root_picker import _segment_line_mesh

    mesh = _segment_line_mesh(pv, np.array([[5.0, 6.0, 7.0]]))

    assert mesh.n_points == 1 and mesh.n_lines == 0
    assert mesh.n_faces_strict == 0


# ------------------------------------- the shared picker path, used by every command
#
# `pick-roots`, `skeletonise`, `skeletonise-all` and `optimise-skeleton` all root a
# graph through `_roots_for_graph`, so what a *skipped* tree means is decided in one
# place. These pin that contract, and in particular that a graph with a left and a
# right tree ends up with a root for each even when only one was clicked.


class _Args:
    """A stand-in for the argparse namespace the helpers read with `getattr`."""

    def __init__(self, **kw):
        self.color_by = "none"
        self.style = "tube"
        self.roots_json = None
        self.__dict__.update(kw)


def test_the_shared_path_returns_one_automatic_root_per_tree():
    from hipct_seg_debug.edit.__main__ import _roots_for_graph

    records, method = _roots_for_graph(tagged_two_trees(), _Args(), interactive=False)

    assert method == "auto"
    assert [r["index"] for r in records] == [0, 1], "a root for the left AND the right"


def test_a_tree_the_operator_skipped_still_gets_ordered(monkeypatch):
    """Click one inlet, press `x`: the other tree must not silently stay at order 0."""
    from hipct_seg_debug.edit.__main__ import _roots_for_graph
    from hipct_seg_debug.edit.optimise import order

    graph = tagged_two_trees()
    picked = roots_mod.auto(graph)[:1]
    monkeypatch.setattr(roots_mod, "pick", lambda *a, **k: list(picked))

    records, method = _roots_for_graph(graph, _Args(), interactive=True)

    assert method == "manual"
    assert len(records) == 1, "only the picked tree is recorded as manual"

    report = order(graph.triple, roots=[r["root"]["node_id"] for r in records])
    assert report.n_roots == 2, "the skipped component is filled from auto_roots"
    assert all(seg["strahler"] > 0 for seg in graph.segments)


def test_picking_nothing_records_nothing(monkeypatch):
    """An empty pick is a decision, not a reason to write the roots the operator refused."""
    from hipct_seg_debug.edit.__main__ import _roots_for_graph

    monkeypatch.setattr(roots_mod, "pick", lambda *a, **k: [])

    records, method = _roots_for_graph(tagged_two_trees(), _Args(), interactive=True)

    assert records == []
    assert method == "manual"


def test_a_headless_machine_falls_back_to_the_automatic_roots(monkeypatch):
    from hipct_seg_debug.edit.__main__ import _roots_for_graph

    def no_pyvista(*a, **k):
        raise ImportError("No module named 'pyvista'")

    monkeypatch.setattr(roots_mod, "pick", no_pyvista)

    records, method = _roots_for_graph(tagged_two_trees(), _Args(), interactive=True)

    assert method == "auto"
    assert [r["index"] for r in records] == [0, 1]


def test_the_shared_writer_produces_a_sidecar_resolve_can_read(tmp_path):
    from hipct_seg_debug.edit.__main__ import _roots_for_graph, _write_roots

    graph = tagged_two_trees()
    dest = tmp_path / "roots.json"
    records, method = _roots_for_graph(graph, _Args(), interactive=False)

    assert _write_roots(graph, records, _Args(roots_json=str(dest)),
                        method=method, source="candidate.am")

    resolved = roots_mod.resolve(graph, roots_mod.load(dest))
    assert len(resolved.root_nodes) == 2
    assert set(resolved.root_nodes) == {r["root"]["node_id"] for r in records}


def test_without_a_roots_json_the_shared_writer_writes_nothing(tmp_path):
    from hipct_seg_debug.edit.__main__ import _roots_for_graph, _write_roots

    graph = tagged_two_trees()
    records, method = _roots_for_graph(graph, _Args(), interactive=False)

    assert not _write_roots(graph, records, _Args(), method=method)
    assert not list(tmp_path.iterdir())


def test_a_sweep_refuses_to_open_the_picker(capsys):
    """Every sweep trial is discarded, so a root clicked on one belongs to nothing."""
    from hipct_seg_debug.edit.__main__ import build_parser, cmd_optimise_skeleton

    args = build_parser().parse_args(
        ["optimise-skeleton", "g.am", "--sweep", "prune-factor=1,2", "--pick-roots"]
    )

    assert cmd_optimise_skeleton(args) == 2
    assert "do not combine" in capsys.readouterr().out


def test_optimise_skeleton_orders_saves_then_records(tmp_path, monkeypatch):
    """End to end: `--pick-roots` orders the refined graph and records it afterwards.

    The write order matters. The sidecar hashes the graph its roots belong to, and
    that graph only exists on disk once `_save` has run -- a sidecar written first
    would carry `source_sha1: null` and lose the one field that says *which* graph was
    rooted.
    """
    import hashlib
    import types

    from hipct_seg_debug.edit import __main__ as main_mod
    from hipct_seg_debug.edit import skeleton_optimise as so
    from hipct_seg_debug.edit.amira_write import write_spatial_graph

    src, out, sidecar = tmp_path / "cand.am", tmp_path / "ref.am", tmp_path / "r.json"
    write_spatial_graph(to_spatial_graph(tagged_two_trees().triple), src)

    monkeypatch.setattr(main_mod, "_decoded", lambda args: (None, None, None, None))
    monkeypatch.setattr(so, "optimise_skeleton",
                        lambda *a, **k: types.SimpleNamespace(describe=lambda: "stub"))
    # One tree clicked, the other skipped -- the usual outcome on a real coronary mask.
    monkeypatch.setattr(roots_mod, "pick",
                        lambda graph, **k: roots_mod.auto(graph)[:1])

    args = main_mod.build_parser().parse_args(
        ["optimise-skeleton", str(src), "--pick-roots", "--color-by", "none",
         "--roots-json", str(sidecar), "--out", str(out)]
    )
    assert main_mod.cmd_optimise_skeleton(args) == 0

    document = roots_mod.load(sidecar)
    assert len(document["trees"]) == 1, "only the picked tree is recorded"
    assert document["trees"][0]["root"]["method"] == "manual"
    assert document["source_sha1"] == hashlib.sha1(out.read_bytes()).hexdigest(), (
        "the sidecar must hash the refined graph, which means it is written after it"
    )

    # And the skipped tree was still ordered, from its automatic root.
    written = read_spatial_graph(out)
    assert (np.asarray(written.edge_attrs["strahler"]) > 0).all()


# ------------------------------------------------- several skeletons at once
#
# A left-tree graph and a right-tree graph rooted in one session, against one mask
# labelling, into one sidecar. What has to hold: the two roots keep distinct tree
# indices, both source graphs are recorded, and `resolve` still places each root on
# the graph it belongs to -- it keys on the coordinate, never on the provenance.


def test_two_graphs_rooting_the_same_tree_index_do_not_collide():
    """Without a mask, a left-only and a right-only graph both call themselves 0."""
    from hipct_seg_debug.edit.__main__ import _renumber_trees

    left = roots_mod.describe_roots(two_trees(), roots_mod.auto(two_trees())[:1])
    right = roots_mod.describe_roots(two_trees((0, 5000, 0)),
                                     roots_mod.auto(two_trees((0, 5000, 0)))[:1])
    assert left[0]["index"] == right[0]["index"] == 0

    taken: dict = {}
    out = (_renumber_trees(left, taken, source="left.am")
           + _renumber_trees(right, taken, source="right.am"))

    assert [r["index"] for r in out] == [0, 1]
    assert out[1]["renumbered_from"] == 0


def test_a_tree_re_picked_under_the_same_material_keeps_its_number():
    """Renumbering a genuine second pick of one tree would invent a tree."""
    from hipct_seg_debug.edit.__main__ import _renumber_trees

    first = [{"index": 0, "material": "Left_Tree"}]
    again = [{"index": 0, "material": "Left_Tree"}]

    taken: dict = {}
    out = (_renumber_trees(first, taken, source="left.am")
           + _renumber_trees(again, taken, source="also_left.am"))

    assert [r["index"] for r in out] == [0, 0], "the mask says these are one tree"


def test_the_sidecar_records_every_graph_that_was_rooted(tmp_path):
    from hipct_seg_debug.edit.amira_write import write_spatial_graph

    left, right = tagged_two_trees(), tagged_two_trees((0.0, 5000.0, 0.0))
    paths = []
    for name, graph in (("left.am", left), ("right.am", right)):
        path = tmp_path / name
        write_spatial_graph(to_spatial_graph(graph.triple), path)
        paths.append(path)

    document = roots_mod.document(
        left, roots_mod.describe_roots(left, roots_mod.auto(left)),
        source=paths[0],
        sources=[roots_mod.describe_source(p, g)
                 for p, g in zip(paths, (left, right))],
    )

    assert [Path(s["path"]).name for s in document["sources"]] == ["left.am", "right.am"]
    assert all(s["sha1"] for s in document["sources"])
    # The top-level `source` still names the first, for a reader written before this.
    assert document["source_sha1"] == document["sources"][0]["sha1"]


def test_resolve_ignores_provenance_and_places_by_coordinate(tmp_path):
    """A combined sidecar handed to one of its graphs still roots that graph."""
    left = tagged_two_trees()
    combined = roots_mod.document(
        left, roots_mod.describe_roots(left, roots_mod.auto(left)), source="left.am",
        sources=[{"path": "left.am"}, {"path": "right.am"}],
    )

    resolved = roots_mod.resolve(left, combined)

    assert len(resolved.root_nodes) == 2 and not resolved.notes


# ------------------------------------------ every command that orders reads them
#
# A root is only worth picking if the commands that assign Strahler order actually
# use it. Three of them were still calling `order()` with no roots at all, so the
# sidecar was written, carried around, and quietly ignored.


def _sidecar(tmp_path, graph, node, name="roots.json"):
    path = tmp_path / name
    roots_mod.write(path, roots_mod.document(
        graph, roots_mod.describe_roots(graph, [node], method="manual"), source=None))
    return path


def test_optimise_orders_from_the_sidecar(tmp_path, capsys):
    """`optimise --order` is where most graphs first get a Strahler column."""
    from hipct_seg_debug.edit.__main__ import build_parser, cmd_optimise
    from hipct_seg_debug.edit.amira_write import write_spatial_graph

    graph = tagged_two_trees()
    source = tmp_path / "g.am"
    write_spatial_graph(to_spatial_graph(graph.triple), source)
    # A root `auto_roots` would not choose, so "used it" is distinguishable from
    # "ordered it somehow".
    automatic = set(roots_mod.auto(graph))
    chosen = next(n for n in graph.nodes
                  if graph.degree(n) == 1 and n not in automatic)
    out = tmp_path / "ordered.am"

    args = build_parser().parse_args(
        ["optimise", str(source), "--reference", str(source),
         "--roots-json", str(_sidecar(tmp_path, graph, chosen)), "--out", str(out)]
    )
    assert cmd_optimise(args) == 0

    reported = capsys.readouterr().out
    assert "from 1 chosen root(s)" in reported
    assert f"[{chosen}" in reported, "the chosen node is the root it ordered from"
    assert str(sorted(automatic)) not in reported, "not the automatic pair"
    assert (np.asarray(read_spatial_graph(out).edge_attrs["strahler"]) > 0).all(), \
        "the second component is filled in from auto_roots, not left at 0"


def test_optimise_without_a_sidecar_still_orders(tmp_path):
    """The flag is optional; nothing may start depending on it."""
    from hipct_seg_debug.edit.__main__ import build_parser, cmd_optimise
    from hipct_seg_debug.edit.amira_write import write_spatial_graph

    source = tmp_path / "g.am"
    write_spatial_graph(to_spatial_graph(tagged_two_trees().triple), source)
    out = tmp_path / "ordered.am"

    args = build_parser().parse_args(
        ["optimise", str(source), "--reference", str(source), "--out", str(out)]
    )
    assert cmd_optimise(args) == 0
    assert (np.asarray(read_spatial_graph(out).edge_attrs["strahler"]) > 0).all()


def test_crop_reorders_from_the_chosen_root():
    """A crop is defined against the root; re-ordering from a different one would
    ship a file whose crop and whose Strahler column disagree about which end is
    proximal."""
    from hipct_seg_debug.edit import crop as crop_mod

    graph = tagged_two_trees()
    automatic = set(roots_mod.auto(graph))
    chosen = next(n for n in graph.nodes
                  if graph.degree(n) == 1 and n not in automatic)

    described = crop_mod.reorder(graph, [chosen])

    assert str(chosen) in described, "the chosen root is the one reported"
    assert all(seg["strahler"] > 0 for seg in graph.segments)


def test_crop_reorder_without_roots_is_unchanged():
    from hipct_seg_debug.edit import crop as crop_mod

    graph = tagged_two_trees()

    assert "2 root(s)" in crop_mod.reorder(graph), "one automatic root per component"


def test_skeletonise_reads_an_existing_sidecar(tmp_path, monkeypatch):
    """`--roots-json` is a destination under `--pick-roots` and a source without it.

    Re-deriving a skeleton is exactly what the sidecar's world coordinates were
    recorded to survive, so ordering the new graph from `auto_roots` while the answer
    sits unread beside it is the one outcome nobody wants.
    """
    from hipct_seg_debug.edit import __main__ as main_mod

    graph = tagged_two_trees()
    automatic = set(roots_mod.auto(graph))
    chosen = next(n for n in graph.nodes
                  if graph.degree(n) == 1 and n not in automatic)
    sidecar = _sidecar(tmp_path, graph, chosen)

    monkeypatch.setattr(main_mod, "_open_lattice",
                        lambda args: (None, make_frame((30, 30, 60)), "seg.am"))
    monkeypatch.setattr("hipct_seg_debug.edit.skeletonise.skeletonise_lattice",
                        lambda *a, **k: types.SimpleNamespace(
                            triple=tagged_two_trees().triple))

    out = tmp_path / "skel.am"
    args = main_mod.build_parser().parse_args(
        ["skeletonise", "--roots-json", str(sidecar), "--out", str(out)]
    )
    assert main_mod.cmd_skeletonise(args) == 0

    written = read_spatial_graph(out)
    assert (np.asarray(written.edge_attrs["strahler"]) > 0).all()
