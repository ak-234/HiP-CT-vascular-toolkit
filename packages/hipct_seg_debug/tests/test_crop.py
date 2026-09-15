"""What a crop removes, and what it must refuse to remove.

Two claims carry the design and are pinned here rather than left to the docstrings:

* **a sidecar keyed by ``seg["id"]`` would name the wrong vessels** the moment its
  graph has been written and re-read, because `to_spatial_graph` renumbers edges to
  array indices. `test_a_sidecar_survives_a_write_and_a_re_read` shows the geometric
  key surviving that round trip and the id not.
* **`Topology.descendants` is not `EditableGraph.subtree`.** On a graph with a cycle the
  undirected flood walks back around into the trunk, and the trunk is exactly what a
  crop must never take by accident -- `test_descendants_of_a_diamond_do_not_escape`.
"""

from __future__ import annotations

import numpy as np
import pytest

from hipct_seg_debug.edit import crop
from hipct_seg_debug.edit.adapter import from_spatial_graph, read_triple
from hipct_seg_debug.edit.amira_write import write_spatial_graph
from hipct_seg_debug.edit.graphmodel import EditableGraph

from .conftest_geometry import graph_from

# A coronary-shaped fixture, radii in um:
#
#   0 --trunk(800)-- 1 --LADprox(700)-- 2 --LADdist(600)-- 3
#                    |                  |
#                    +--big(500)-- 4    +--branch(300)-- 5 --(280)-- 6 --twig(100)-- 7
#
# `big` hangs off the *trunk*, which nobody annotated, so it descends from no main
# vessel. `twig` is three generations below the LAD and is judged against the LAD's
# ostium, not against its own 280 um parent -- which is the whole point of the rule.
NODES = [
    (0.0, 0.0, 0.0), (2000.0, 0.0, 0.0), (4000.0, 0.0, 0.0), (6000.0, 0.0, 0.0),
    (2000.0, -2000.0, 0.0), (4000.0, 2000.0, 0.0), (5000.0, 3500.0, 0.0),
    (6000.0, 5000.0, 0.0),
]
EDGES = [
    (0, 1, 20, 800.0),   # 0 trunk
    (1, 2, 20, 700.0),   # 1 LAD proximal
    (2, 3, 20, 600.0),   # 2 LAD distal
    (1, 4, 20, 500.0),   # 3 off the unannotated trunk
    (2, 5, 20, 300.0),   # 4 a side branch of the LAD
    (5, 6, 20, 280.0),   # 5
    (6, 7, 20, 100.0),   # 6 the twig
]
LAD = {1, 2}


def tree() -> EditableGraph:
    return graph_from(NODES, EDGES)


def ordered_tree() -> EditableGraph:
    """The same tree with a plausible Strahler order on every segment."""
    graph = tree()
    for sid, order in {0: 3, 1: 3, 2: 2, 3: 2, 4: 2, 5: 2, 6: 1}.items():
        graph.segments[sid]["strahler"] = order
    return graph


# ------------------------------------------------------------------- identity


def test_a_key_is_undirected():
    """The same segment built the other way round is the same segment."""
    forward = graph_from(NODES, [(1, 2, 20, 700.0)])
    backward = graph_from(NODES, [(2, 1, 20, 700.0)])
    assert crop.segment_key(forward, 0) == crop.segment_key(backward, 0)


def test_a_key_survives_the_renumbering_a_deletion_causes():
    graph = tree()
    before = {sid: crop.segment_key(graph, sid) for sid in (1, 2)}
    graph.delete_subtree(4, 2)  # take an unrelated branch out
    assert {sid: crop.segment_key(graph, sid) for sid in (1, 2)} == before


def test_a_key_two_segments_share_is_unresolved_not_guessed():
    """A two-point segment has ``mid == lo``; two in one place would alias."""
    graph = graph_from([(0.0, 0.0, 0.0), (100.0, 0.0, 0.0)],
                       [(0, 1, 2, 50.0), (0, 1, 2, 50.0)])
    key = crop.segment_key(graph, 0)
    assert crop.segment_key(graph, 1) == key
    found, unresolved = crop.resolve_keys(graph, [key])
    assert found == {} and unresolved == [key]


def test_an_unknown_key_is_reported():
    found, unresolved = crop.resolve_keys(tree(), ["0" * 16])
    assert found == {} and unresolved == ["0" * 16]


def test_node_at_picks_the_nearer_endpoint():
    graph = tree()
    seg = graph.segment(1)
    assert crop.node_at(graph, 1, NODES[1]) == seg["node1"]
    assert crop.node_at(graph, 1, NODES[2]) == seg["node2"]


# ---------------------------------------------------------------- measurement


def test_the_takeoff_radius_is_read_away_from_the_junction():
    """The contours at a junction measure the carina, not the branch."""
    graph = tree()
    radii = np.full(20, 120.0)
    radii[:3] = 900.0  # what the distance transform reports across the carina
    graph.set_segment_radii(6, radii)

    node = crop.topology(graph).prox_node(graph, 6)
    assert crop.takeoff_radius_um(graph, 6, node) == pytest.approx(120.0)


def test_the_takeoff_radius_is_in_micrometres():
    """Guards against a stray /1000 carried over from ``branch_radius_mm``."""
    graph = tree()
    node = crop.topology(graph).prox_node(graph, 1)
    assert crop.takeoff_radius_um(graph, 1, node) == pytest.approx(700.0)


def test_an_unmeasurable_branch_reports_nan_and_is_not_dropped():
    """`nan` means "not measured", and that is not evidence for removing anything."""
    graph = tree()
    graph.set_segment_radii(6, np.full(20, np.nan))
    topo = crop.topology(graph)
    assert np.isnan(crop.takeoff_radius_um(graph, 6, topo.prox_node(graph, 6)))

    # 150 um would drop the twig on its measured radius alone; unmeasured, it stays.
    assert crop.plan(graph, crop.Rule(min_ostium_um=150.0)).drop == set()


# ------------------------------------------------------------------- topology


def test_topology_agrees_with_the_rooting_the_rest_of_the_package_uses():
    from hipct_seg_debug.edit.radius_perimeter import _directed_topology

    graph = ordered_tree()
    depth, parent = _directed_topology(graph)
    topo = crop.topology(graph)
    assert topo.depth == depth and topo.parent == parent


def test_the_root_is_the_thickest_free_ended_branch():
    topo = crop.topology(tree())
    assert topo.roots == [0]


def test_a_forced_root_edge_wins():
    topo = crop.topology(tree(), root_edges=(6,))
    assert topo.roots == [6]
    assert topo.depth[0] > topo.depth[6]


def test_two_forced_roots_in_one_component_raise():
    with pytest.raises(ValueError, match="same component"):
        crop.topology(tree(), root_edges=(0, 6))


def test_descendants_of_a_loop_do_not_escape_upstream():
    """The test that justifies not using ``subtree`` inside the rules.

    A skeletonisation loop that closes back near the inlet: walking away from the
    branch point along one arm comes round the loop, in through the inlet node, and
    back down the trunk -- so an undirected flood would prune the whole tree.
    """
    nodes = [(0.0, 0.0, 0.0), (1000.0, 0.0, 0.0),
             (2000.0, 1000.0, 0.0), (1500.0, 2000.0, 0.0)]
    graph = graph_from(nodes, [(0, 1, 10, 800.0),   # 0 trunk, inlet at node 0
                               (1, 2, 10, 400.0),   # 1 the arm
                               (2, 3, 10, 300.0),   # 2
                               (3, 0, 10, 300.0)])  # 3 closes back onto the inlet
    topo = crop.topology(graph)
    arm = 1

    assert 0 in graph.subtree(arm, graph.segment(arm)["node1"])
    assert 0 not in topo.descendants(arm)


# -------------------------------------------------------------------- tracing


def _diamond() -> EditableGraph:
    r"""A fork with two routes to the same meeting point: one short and thin, one long
    and fat. Every real epicardial tree with a kissing artefact in it looks like this,
    and it is the only shape where the trace's weighting can change the answer.

        0 --start(600)-- 1 --thin(20), 1000 um-------------- 2 --end(600)-- 5
                          \                                 /
                           3 --fat(500)-- ... --fat(500)----
    """
    nodes = [(0.0, 0.0, 0.0), (2000.0, 0.0, 0.0), (3000.0, 0.0, 0.0),
             (2000.0, 3000.0, 0.0), (5000.0, 0.0, 0.0)]
    return graph_from(nodes, [
        (0, 1, 10, 600.0),   # 0 start
        (1, 2, 10, 20.0),    # 1 the thin bridge, 1000 um
        (1, 3, 10, 500.0),   # 2 the long way, 3000 um
        (3, 2, 10, 500.0),   # 3 and back down, ~3162 um
        (2, 4, 10, 600.0),   # 4 end
    ])


def test_a_trace_takes_the_one_path_between_two_picks():
    """On a tree there is exactly one simple path, so this is not a choice at all."""
    traced = crop.trace_path(tree(), 0, 6)
    assert traced.segments == [0, 1, 4, 5, 6]
    assert traced.nodes == [1, 2, 5, 6]


def test_a_trace_runs_up_over_a_fork_and_back_down():
    """Two branches either side of a bifurcation, which is where backtracking would show."""
    traced = crop.trace_path(tree(), 3, 4)
    assert traced.segments == [3, 1, 4]
    assert traced.nodes == [1, 2]


def test_a_trace_is_symmetric_in_its_two_picks():
    there = crop.trace_path(tree(), 0, 6)
    back = crop.trace_path(tree(), 6, 0)
    assert back.segments == there.segments[::-1]
    assert back.nodes == there.nodes[::-1]
    assert back.length_um == pytest.approx(there.length_um)


def test_a_trace_to_itself_is_that_segment():
    traced = crop.trace_path(tree(), 2, 2)
    assert traced.segments == [2] and traced.nodes == []
    assert traced.length_um == pytest.approx(2000.0)


def test_two_neighbours_trace_to_the_pair_with_nothing_between():
    traced = crop.trace_path(tree(), 1, 2)
    assert traced.segments == [1, 2]
    assert traced.nodes == [2], "the node they share, and only it"
    assert traced.length_um == pytest.approx(4000.0)


def test_the_length_is_arc_length_along_the_centreline():
    traced = crop.trace_path(tree(), 0, 2)
    assert traced.length_um == pytest.approx(6000.0)  # three 2000 um segments


def test_a_trace_between_two_trees_refuses_rather_than_guessing():
    graph = graph_from([(0.0, 0.0, 0.0), (1000.0, 0.0, 0.0),
                        (0.0, 5000.0, 0.0), (1000.0, 5000.0, 0.0)],
                       [(0, 1, 10, 400.0), (2, 3, 10, 400.0)])
    with pytest.raises(crop.CropError, match="not connected"):
        crop.trace_path(graph, 0, 1)


def test_a_trace_from_a_segment_that_is_not_there_refuses():
    with pytest.raises(crop.CropError, match="not in this graph"):
        crop.trace_path(tree(), 0, 99)


def test_a_trace_round_a_loop_takes_the_shorter_arm_and_repeats_nothing():
    traced = crop.trace_path(_diamond(), 0, 4)
    assert traced.segments == [0, 1, 4], "the 1000 um bridge, not the 6000 um detour"
    assert len(set(traced.segments)) == len(traced.segments)
    assert len(set(traced.nodes)) == len(traced.nodes), "a simple path visits no node twice"


def test_tracing_by_thickness_goes_round_a_thin_bridge():
    """The reason the option exists: the short way between two epicardial vessels is
    often the artefact joining them, and length alone will always take it."""
    traced = crop.trace_path(_diamond(), 0, 4, prefer_thick=True)
    assert traced.segments == [0, 2, 3, 4]
    assert traced.weight == "thickness"
    assert traced.length_um > crop.trace_path(_diamond(), 0, 4).length_um


def test_an_unmeasured_segment_is_not_a_free_shortcut():
    """A radius of zero must cost at least its length, or it becomes the cheapest
    thing in the graph and every trace is routed through it."""
    graph = _diamond()
    for pid in graph.segment(1)["point_ids"]:
        graph.set_radius(pid, 0.0)
    assert crop.trace_path(graph, 0, 4, prefer_thick=True).segments == [0, 2, 3, 4]


def test_a_traced_path_names_a_vessel_the_rules_then_use():
    """The point of the whole thing: a trace is a selection, and nothing below it
    can tell it was not clicked segment by segment."""
    graph = ordered_tree()
    traced = crop.trace_path(graph, 1, 2)
    plan = crop.plan(graph, crop.Rule(ratio=0.25), vessels={"LAD": set(traced.segments)})
    assert set(traced.segments) == LAD
    assert plan.ostia["LAD"]["radius_um"] == pytest.approx(700.0)
    assert plan.drop == {6}


# ---------------------------------------------------------------------- rules


def test_a_strahler_crop_takes_the_whole_subtree():
    graph = ordered_tree()
    graph.segments[4]["strahler"] = 1  # the branch, with two segments below it
    plan = crop.plan(graph, crop.Rule(min_strahler=2))
    assert plan.drop == {4, 5, 6}
    assert plan.takeoffs[4]["n_subtree_removed"] == 3
    assert 5 not in plan.takeoffs, "a take-off inside another's subtree is not a take-off"


def test_a_graph_with_no_order_refuses_rather_than_dropping_everything():
    with pytest.raises(crop.CropError, match="no Strahler order"):
        crop.plan(tree(), crop.Rule(min_strahler=2))


def test_the_order_is_read_under_the_files_own_field_name():
    """`adapter` canonicalises every alias, so the rule needs no alias table."""
    spatial = ordered_tree().to_spatial_graph()
    spatial.edge_attrs["StrahlerOrder"] = spatial.edge_attrs.pop("strahler")
    graph = EditableGraph(from_spatial_graph(spatial))

    assert crop.plan(graph, crop.Rule(min_strahler=2)).drop == {6}


def test_an_absolute_radius_crop_needs_no_main_vessels():
    plan = crop.plan(tree(), crop.Rule(min_ostium_um=150.0))
    assert plan.drop == {6}
    assert plan.vessels == {}


def test_the_ratio_judges_a_deep_twig_against_its_main_vessel_not_its_parent():
    graph = tree()
    plan = crop.plan(graph, crop.Rule(ratio=0.25), vessels={"LAD": set(LAD)})

    assert plan.ostia["LAD"]["radius_um"] == pytest.approx(700.0)
    # 100 um against 0.25 x 700 = 175. Judged against its own 280 um parent it would
    # have cleared 70 um comfortably and stayed.
    assert plan.drop == {6}
    assert plan.takeoffs[6]["threshold_um"] == pytest.approx(175.0)
    assert plan.takeoffs[6]["vessel"] == "LAD"


def test_the_ratio_threshold_is_strict():
    """A branch measuring exactly the threshold is kept, as in epicardial_annotation."""
    graph = tree()
    graph.set_segment_radii(6, np.full(20, 175.0))
    assert crop.plan(graph, crop.Rule(ratio=0.25), vessels={"LAD": set(LAD)}).drop == set()

    graph.set_segment_radii(6, np.full(20, 174.0))
    assert crop.plan(graph, crop.Rule(ratio=0.25), vessels={"LAD": set(LAD)}).drop == {6}


def test_a_main_vessel_is_never_dropped_even_when_it_is_thin():
    graph = tree()
    graph.set_segment_radii(2, np.full(20, 10.0))  # the LAD's distal half, collapsed
    plan = crop.plan(graph, crop.Rule(min_ostium_um=500.0), vessels={"LAD": set(LAD)})
    assert 2 not in plan.drop


def test_the_path_from_the_root_to_a_main_vessel_is_protected():
    """An unannotated left main proximal to the LAD would take the LAD with it.

    Only the LAD's distal half is annotated here, so segment 1 is an ordinary
    unannotated branch on paper -- and is spared because the annotated part hangs
    off it.
    """
    graph = tree()
    graph.set_segment_radii(1, np.full(20, 50.0))
    plan = crop.plan(graph, crop.Rule(min_ostium_um=100.0), vessels={"LAD": {2}})
    assert 1 in plan.protected and 1 not in plan.drop
    assert 2 not in plan.drop


def test_an_unattributed_subtree_is_kept_by_default_and_dropped_on_request():
    graph = tree()
    kept = crop.plan(graph, crop.Rule(ratio=0.25), vessels={"LAD": set(LAD)})
    assert 3 not in kept.drop

    pruned = crop.plan(graph, crop.Rule(ratio=0.25, prune_unattributed=True),
                       vessels={"LAD": set(LAD)})
    assert 3 in pruned.drop


def test_a_ratio_with_no_main_vessels_refuses():
    with pytest.raises(crop.CropError, match="named main vessel"):
        crop.plan(tree(), crop.Rule(ratio=0.25))


def test_the_rules_compose_to_their_union():
    graph = ordered_tree()
    graph.segments[3]["strahler"] = 1
    rule_a = crop.Rule(min_strahler=2)
    rule_b = crop.Rule(ratio=0.25)
    both = crop.Rule(min_strahler=2, ratio=0.25)

    a = crop.plan(graph, rule_a, vessels={"LAD": set(LAD)}).drop
    b = crop.plan(graph, rule_b, vessels={"LAD": set(LAD)}).drop
    assert a and b and a != b
    assert crop.plan(graph, both, vessels={"LAD": set(LAD)}).drop == a | b


# ----------------------------------------------------------------- hand marks


def test_a_hand_marked_segment_drops_only_itself():
    plan = crop.plan(tree(), crop.Rule(), drop_segments=[5])
    assert plan.drop == {5}


def test_a_hand_marked_prune_takes_everything_past_the_pick():
    graph = tree()
    plan = crop.plan(graph, crop.Rule(), prune_at=[(4, graph.segment(4)["node1"])])
    assert plan.drop == {4, 5, 6}


def test_a_hand_mark_on_a_main_vessel_is_refused_with_a_note():
    plan = crop.plan(tree(), crop.Rule(), vessels={"LAD": set(LAD)}, drop_segments=[1])
    assert plan.drop == set()
    assert any("refused" in note for note in plan.notes)


def test_a_hand_prune_that_reaches_a_main_vessel_is_refused():
    graph = tree()
    # Walking from the far end of the trunk reaches the whole rest of the tree.
    plan = crop.plan(graph, crop.Rule(), vessels={"LAD": set(LAD)},
                     prune_at=[(0, graph.segment(0)["node1"])])
    assert plan.drop == set()
    assert any("protected" in note for note in plan.notes)


# ----------------------------------------------------------------- plan/apply


def test_apply_is_one_undo_step_and_restores_everything():
    graph = ordered_tree()
    before = sorted(graph.segment_ids())
    plan = crop.plan(graph, crop.Rule(min_strahler=2))

    assert crop.apply(graph, plan) == len(plan.drop)
    assert graph.undo() is not None
    assert sorted(graph.segment_ids()) == before


def test_apply_leaves_no_orphaned_node():
    graph = tree()
    crop.apply(graph, crop.plan(graph, crop.Rule(), prune_at=[(4, graph.segment(4)["node1"])]))
    assert all(graph.degree(nid) > 0 for nid in graph.nodes)


def test_a_second_pass_finds_nothing():
    """Pins the single-pass argument: without contraction there is nothing left to find."""
    graph = tree()
    rule = crop.Rule(ratio=0.25)
    crop.apply(graph, crop.plan(graph, rule, vessels={"LAD": set(LAD)}))
    assert crop.plan(graph, rule, vessels={"LAD": set(LAD)}).drop == set()


def test_the_degree_two_nodes_a_crop_creates_are_reported():
    plan = crop.plan(tree(), crop.Rule(), drop_segments=[6])
    assert plan.degree2_after >= 1
    assert any("degree-2" in note for note in plan.notes)


def test_the_after_counts_match_what_apply_produces():
    graph = ordered_tree()
    plan = crop.plan(graph, crop.Rule(min_strahler=2))
    crop.apply(graph, plan)
    assert plan.segments_after == len(graph.segments)
    assert plan.components_after == len(graph.components())


# -------------------------------------------------------------------- sidecar


def _document(graph, rule, vessels=None, **kw):
    plan = crop.plan(graph, rule, vessels=vessels)
    return crop.document(graph, plan, rule, **kw), plan


def test_the_document_round_trips(tmp_path):
    graph = tree()
    rule = crop.Rule(ratio=0.25, takeoff_factor=2.0)
    document, _plan = _document(graph, rule, {"LAD": set(LAD)})
    back = crop.load(crop.write(tmp_path / "c.json", document))
    resolved = crop.resolve(graph, back)

    assert resolved.vessels == {"LAD": set(LAD)}
    assert resolved.rule.ratio == pytest.approx(0.25)
    assert resolved.colors["LAD"] == crop.VESSEL_COLORS[0]


def test_a_foreign_schema_is_refused(tmp_path):
    path = tmp_path / "c.json"
    path.write_text('{"schema": "somebody.else/1"}', encoding="utf-8")
    with pytest.raises(ValueError, match="expected schema"):
        crop.load(path)


def test_a_missing_vessel_key_is_fatal():
    graph = tree()
    document, _plan = _document(graph, crop.Rule(ratio=0.25), {"LAD": set(LAD)})
    document["vessels"]["LAD"]["seg_keys"][0] = "0" * 16
    with pytest.raises(crop.CropError, match="main vessel 'LAD'"):
        crop.resolve(graph, document)


def test_a_missing_hand_mark_is_a_note_not_a_failure():
    graph = tree()
    document, _plan = _document(graph, crop.Rule())
    document["manual"]["drop_segments"] = [{"seg_key": "0" * 16, "seg_id": 99}]
    resolved = crop.resolve(graph, document)
    assert resolved.drop_segments == []
    assert any("no longer resolves" in note for note in resolved.notes)


def test_a_sidecar_survives_a_write_and_a_re_read(tmp_path):
    """The schema decision, demonstrated: keys survive renumbering and ids do not."""
    graph = tree()
    # The named vessel sits *above* the segment being dropped, so the write renumbers
    # it -- which is the situation identity by id gets wrong.
    named = {4, 5}
    plan = crop.plan(graph, crop.Rule(), vessels={"branch": named}, drop_segments=[3])
    document = crop.document(graph, plan, crop.Rule())
    crop.apply(graph, plan)

    path = tmp_path / "cropped.am"
    write_spatial_graph(graph.to_spatial_graph(), path)
    reread = EditableGraph(read_triple(path))

    resolved = crop.resolve(reread, document)
    found = sorted(resolved.vessels["branch"])
    assert {crop.segment_key(reread, sid) for sid in found} == set(
        document["vessels"]["branch"]["seg_keys"]
    )
    # The ids the document carries as a hint no longer name those segments: the write
    # renumbered every edge, which is exactly what identity by id would have missed.
    assert found == [3, 4] != sorted(named)


def test_a_fingerprint_mismatch_warns_but_does_not_refuse(tmp_path):
    graph = tree()
    path = tmp_path / "g.am"
    write_spatial_graph(graph.to_spatial_graph(), path)
    graph = EditableGraph(read_triple(path))

    document, _plan = _document(graph, crop.Rule(), keys=crop.segment_keys(graph))
    document["source_sha1"] = "f" * 40

    resolved = crop.resolve(graph, document)
    assert any("different graph" in note for note in resolved.notes)


def test_replay_drops_exactly_what_was_recorded():
    graph = tree()
    rule = crop.Rule(ratio=0.25)
    plan = crop.plan(graph, rule, vessels={"LAD": set(LAD)})
    document = crop.document(graph, plan, rule)

    replayed = crop.replay_plan(graph, document)
    assert replayed.drop == plan.drop
    assert replayed.takeoffs == {}, "a replay re-evaluates nothing"


def test_a_replay_that_cannot_resolve_everything_refuses():
    graph = tree()
    rule = crop.Rule(ratio=0.25)
    plan = crop.plan(graph, rule, vessels={"LAD": set(LAD)})
    document = crop.document(graph, plan, rule)
    crop.apply(graph, plan)  # the recorded drop is now gone

    with pytest.raises(crop.CropError, match="not be a replay"):
        crop.replay_plan(graph, document)


def test_an_unmeasured_radius_serialises_as_null():
    """`_plain` turns non-finite floats into null; JSON has no NaN token."""
    graph = tree()
    plan = crop.plan(graph, crop.Rule(), drop_segments=[6])
    plan.takeoffs[6]["radius_um"] = float("nan")
    document = crop.document(graph, plan, crop.Rule())
    assert document["selection"]["takeoffs"][0]["radius_um"] is None


def test_the_csv_names_every_dropped_takeoff(tmp_path):
    graph = tree()
    plan = crop.plan(graph, crop.Rule(ratio=0.25), vessels={"LAD": set(LAD)})
    rows = crop.write_csv(tmp_path / "c.csv", graph, plan).read_text().strip().splitlines()

    assert rows[0].split(",") == list(crop.CSV_COLUMNS)
    assert len(rows) == 1 + len(plan.takeoffs)
    assert rows[1].startswith("6,")


def test_carry_manual_keys_the_hand_marks_geometrically():
    graph = tree()
    document, _plan = _document(graph, crop.Rule())
    node = graph.segment(4)["node1"]
    crop.carry_manual(document, graph, drop_segments=[5], prune_at=[(4, node)])

    assert document["manual"]["drop_segments"][0]["seg_key"] == crop.segment_key(graph, 5)
    resolved = crop.resolve(graph, document)
    assert resolved.drop_segments == [5]
    assert resolved.prune_at == [(4, node)]
