"""Manufacturing the endpoints the skeletoniser did not leave behind.

Every test here is built around the same asymmetry: the mask knows about a vessel
and the graph does not. That is the situation the rest of the reconnector cannot
see -- not because it decides wrongly about it, but because no candidate is ever
generated -- so the assertions are mostly about what gets *proposed*, and only then
about what happens to it.

The scenes are deliberately small and deliberately literal. A pruned side branch, a
run of trunk past a free end, a calcification-shaped blob, a speck: each is a thing
the sweep must respond to differently, and a single fixture that mixed them could
pass while getting two of the three backwards.
"""

from __future__ import annotations

import numpy as np

from hipct_seg_debug.edit.reconnect import candidates as candidates_mod
from hipct_seg_debug.edit.reconnect.geodesic import (
    audit,
    classify,
    components,
    lobes,
    route,
    select,
)

from .conftest_geodesic import SHAPE, axis_run, mask_source
from .conftest_geometry import SPACING, cylinder, make_frame

CY = CZ = 20


# --------------------------------------------------------------------- fixtures


def branched(shape=SHAPE, *, trunk=(5, 46), branch_x=25, branch_y=36,
             radius_vox=1) -> np.ndarray:
    """One mask component: a trunk along x with a side branch running out in y."""
    nz, ny, nx = shape
    out = cylinder(shape, radius_vox, trunk[0], trunk[1], cy=CY, cz=CZ)
    zz, yy, xx = np.ogrid[:nz, :ny, :nx]
    limb = (((zz - CZ) ** 2 + (xx - branch_x) ** 2 <= radius_vox * radius_vox)
            & (yy >= CY) & (yy < branch_y))
    return (out | limb).astype(np.uint8)


def two_pieces(shape=SHAPE, *, left=(5, 25), right=(30, 50),
               radius_vox=1) -> np.ndarray:
    """Two mask components, one of which the graph will never describe."""
    return (cylinder(shape, radius_vox, left[0], left[1], cy=CY, cz=CZ)
            | cylinder(shape, radius_vox, right[0], right[1], cy=CY, cz=CZ)
            ).astype(np.uint8)


def overrun(shape=SHAPE, *, radius_vox=1) -> np.ndarray:
    """One component whose lumen runs on past where the graph stops."""
    return cylinder(shape, radius_vox, 5, 35, cy=CY, cz=CZ)


def described_left(frame, x0=5, x1=25, radius_um=10.0):
    """A graph that covers the left piece and nothing else."""
    from hipct_seg_debug.edit.graphmodel import EditableGraph

    return EditableGraph(axis_run(frame, x0, x1, radius_um, cy=CY, cz=CZ))


def scene_for(volume, x0=5, x1=25, radius_um=10.0):
    frame = make_frame(SHAPE)
    source = mask_source(np.asarray(volume, dtype=np.uint8))
    index = components.build(source)
    graph = described_left(frame, x0, x1, radius_um)
    return graph, index, frame, source


def sweep(graph, index, frame, **kwargs):
    associations = classify.associate(index, frame, graph)
    return lobes.find(index, frame, graph, associations, **kwargs)


def bridge_to(graph, node, end):
    """A mask-end bridge built by hand, bypassing the facing gate.

    `propose` refuses a mask end that points away from the source, which is right
    for a proposal and beside the point for a test about what happens *after* one.
    """
    p0 = np.asarray(graph.nodes[node][:3], dtype=np.float64)
    coords = candidates_mod.hermite_path(
        p0, np.array([1.0, 0.0, 0.0]), end.point_um, -np.asarray(end.tangent), 16)
    bridge = candidates_mod.Bridge(
        kind="mask-end", source_node=node, coords=coords,
        radii=np.full(len(coords), end.radius_um))
    bridge.target_mask_end = end
    return bridge


# ------------------------------------------------------------ what it must find


def test_finds_the_tip_of_a_pruned_side_branch():
    """The four-ends case in miniature: the mask branches, the graph does not."""
    graph, index, frame, _ = scene_for(branched())
    ends, report = sweep(graph, index, frame)

    assert report.anchors == 2  # both ends of the described run
    assert ends, lobes.summarise(ends, report)
    branch = [e for e in ends if e.point_um[1] > (CY + 8) * SPACING]
    assert branch, f"no end out along the branch: {lobes.summarise(ends, report)}"
    # The branch runs out in +y, so its outward tangent must too.
    assert branch[0].tangent[1] > 0.8


def test_finds_undescribed_trunk_beyond_a_free_end():
    """Lumen the graph stops short of is a free end even with nothing branching."""
    graph, index, frame, _ = scene_for(overrun())
    ends, report = sweep(graph, index, frame)

    assert ends, report.describe()
    ahead = [e for e in ends if e.point_um[0] > 25 * SPACING]
    assert ahead, lobes.summarise(ends, report)
    assert ahead[0].tangent[0] > 0.8  # pointing on down the trunk, not back


def test_a_lobe_knows_which_graph_component_it_hangs_off():
    """The cycle test needs this, and a lobe on a described vessel is not free."""
    graph, index, frame, _ = scene_for(branched())
    ends, _ = sweep(graph, index, frame)

    attached = [e for e in ends if e.attach_node is not None]
    assert attached
    nodes = set(graph.nodes)
    assert all(e.attach_node in nodes for e in attached)


def test_a_separate_component_reports_no_attachment():
    """A piece of mask with nothing described leading to it is genuinely free."""
    graph, index, frame, _ = scene_for(two_pieces())
    ends, _ = sweep(graph, index, frame)

    right = [e for e in ends if e.component != ends[0].component or
             e.point_um[0] > 27 * SPACING]
    assert right
    assert all(e.attach_node is None for e in right), \
        "a component the graph never touches cannot hang off a node"


# --------------------------------------------------------- what it must refuse


def test_fully_described_lumen_yields_nothing():
    """The commonest case by far, and the one a false positive here would spoil."""
    frame = make_frame(SHAPE)
    volume = cylinder(SHAPE, 1, 5, 46, cy=CY, cz=CZ)
    source = mask_source(volume)
    index = components.build(source)
    graph = described_left(frame, 5, 46)

    ends, report = sweep(graph, index, frame)
    assert ends == [], lobes.summarise(ends, report)


def test_a_blob_is_not_a_vessel():
    """Twenty voxels of calcification beside a vessel is not a free end."""
    nz, ny, nx = SHAPE
    zz, yy, xx = np.ogrid[:nz, :ny, :nx]
    blob = ((zz - CZ) ** 2 + (yy - (CY + 8)) ** 2 + (xx - 20) ** 2 <= 9)
    volume = (cylinder(SHAPE, 1, 5, 25, cy=CY, cz=CZ) | blob).astype(np.uint8)

    graph, index, frame, _ = scene_for(volume)
    ends, report = sweep(graph, index, frame)

    near_blob = [e for e in ends
                 if abs(e.point_um[1] - (CY + 8) * SPACING) < 3 * SPACING
                 and abs(e.point_um[0] - 20 * SPACING) < 4 * SPACING]
    assert near_blob == [], lobes.summarise(ends, report)
    assert report.tips_blob_lobe or report.tips_small_lobe


def test_a_speck_is_below_the_size_floor():
    graph, index, frame, _ = scene_for(branched())
    ends, report = sweep(graph, index, frame, min_voxels=10_000)
    assert ends == []
    assert report.tips_small_lobe > 0


def test_a_tip_on_the_window_face_is_discarded():
    """It is where the box ended, not where the vessel did."""
    graph, index, frame, _ = scene_for(cylinder(SHAPE, 1, 5, 60, cy=CY, cz=CZ))
    _, report = sweep(graph, index, frame)
    assert report.tips_on_window_face > 0


def test_raising_the_describedness_scale_finds_fewer_ends():
    """The knob does what it says, in the direction it says."""
    graph, index, frame, _ = scene_for(branched())
    loose, _ = sweep(graph, index, frame, describe_radii=1.5)
    tight, _ = sweep(graph, index, frame, describe_radii=40.0)
    assert len(tight) < len(loose)


def test_an_unassociated_endpoint_is_not_an_anchor():
    """A guessed premise must not be compounded with a second guess."""
    graph, index, frame, _ = scene_for(branched())
    associations = classify.associate(index, frame, graph)
    for association in associations.values():
        association.component = 0
    ends, report = lobes.find(index, frame, graph, associations)
    assert ends == []
    assert report.anchors == 0
    assert "no associated free end" in report.describe()


# ------------------------------------------------------------------- proposing


def test_proposes_to_a_mask_end_through_the_same_gates():
    graph, index, frame, _ = scene_for(two_pieces())
    associations = classify.associate(index, frame, graph)
    ends, _ = lobes.find(index, frame, graph, associations)
    bridges = lobes.propose(graph, ends, associations)

    assert bridges
    bridge = bridges[0]
    assert bridge.kind == "mask-end"
    assert bridge.target_node is None and bridge.target_segment is None
    assert bridge.target_mask_end is not None
    assert "mask end" in repr(bridge)
    assert bridge.metrics["lobe_voxels"] >= lobes.MIN_LOBE_VOXELS


def test_a_mask_end_behind_the_endpoint_is_outside_the_cone():
    """The same refusal an end-to-end pair would get, for the same reason."""
    graph, index, frame, _ = scene_for(two_pieces())
    associations = classify.associate(index, frame, graph)
    ends, _ = lobes.find(index, frame, graph, associations)
    assert ends

    kept = lobes.propose(graph, ends, associations, cone_angle_deg=5.0)
    behind = [b for b in kept
              if b.metrics.get("cone_deg", 0.0) > 5.0 and b.accepted]
    assert behind == []


def test_a_mask_end_pointing_away_is_not_a_continuation():
    """Lumen that runs *on* past a free end wants re-skeletonising, not a route.

    Its tip faces down the vessel rather than back at the endpoint, so the same
    facing test that stops two parallel free ends being joined stops this. The end
    is still found -- it is real -- it simply is not offered as a continuation.
    """
    graph, index, frame, _ = scene_for(overrun())
    associations = classify.associate(index, frame, graph)
    ends, _ = lobes.find(index, frame, graph, associations)
    assert ends

    kept = lobes.propose(graph, ends, associations)
    assert [b for b in kept if b.accepted] == []
    rejected = lobes.propose(graph, ends, associations, keep_rejected=True)
    assert any("faces away" in b.reason for b in rejected)


def test_a_mask_end_is_scored_below_an_equivalent_graph_end():
    """A weaker premise loses a tie, which is what the damping is for."""
    p0 = np.zeros(3)
    coords = np.stack([p0, np.array([100.0, 0.0, 0.0])])
    bridge = candidates_mod.Bridge(kind="mask-end", source_node=0, coords=coords,
                                   radii=np.array([10.0, 10.0]))
    candidates_mod.gate_geometry(bridge, 10.0, 10.0,
                                 source_tangent=np.array([1.0, 0.0, 0.0]))
    from hipct_seg_debug.edit.reconnect import endpoints as endpoints_mod

    assert lobes._score(bridge, 10.0, 10.0) < endpoints_mod._score(bridge, 10.0, 10.0)


def test_the_endpoints_own_tip_is_not_a_target():
    """A tip found through the mask that *is* this free end is not a continuation."""
    graph, index, frame, _ = scene_for(two_pieces())
    associations = classify.associate(index, frame, graph)
    ends, _ = lobes.find(index, frame, graph, associations)
    node = graph.endpoints()[0]
    here = np.asarray(graph.nodes[node][:3], dtype=np.float64)
    ends.append(lobes.MaskEnd(
        key=(0, 0, 0), point_um=here, tangent=np.array([1.0, 0.0, 0.0]),
        radius_um=10.0, component=1, lobe_voxels=100, elongation=5.0,
        skeleton_um=here[None, :],
    ))
    for bridge in lobes.propose(graph, ends, associations):
        assert bridge.target_mask_end.key != (0, 0, 0)


# ----------------------------------------------------------------- classifying


def test_a_mask_end_in_another_component_is_a_geodesic_repair():
    graph, index, frame, _ = scene_for(two_pieces())
    associations = classify.associate(index, frame, graph)
    ends, _ = lobes.find(index, frame, graph, associations)
    bridge = lobes.propose(graph, ends, associations)[0]

    classified = classify.classify(bridge, associations, index, frame)
    assert classified.kind == "geodesic"
    assert classified.target_mask_end is bridge.target_mask_end
    assert classified.target.node == -1
    assert classified.target.component != classified.source.component


def test_a_mask_end_in_the_same_component_needs_no_route():
    """The lumen is continuous; only the centreline is missing."""
    graph, index, frame, _ = scene_for(overrun())
    associations = classify.associate(index, frame, graph)
    ends, _ = lobes.find(index, frame, graph, associations)
    bridge = bridge_to(graph, graph.endpoints()[-1], ends[0])

    classified = classify.classify(bridge, associations, index, frame)
    assert classified.kind == "reskeletonise"
    assert classified.target.component == classified.source.component


def test_a_mask_end_carries_its_own_calibration_tail():
    """It has no node to walk back from, so one-sided calibration is the risk."""
    graph, index, frame, _ = scene_for(two_pieces())
    associations = classify.associate(index, frame, graph)
    ends, _ = lobes.find(index, frame, graph, associations)
    bridge = lobes.propose(graph, ends, associations)[0]
    classified = classify.classify(bridge, associations, index, frame)

    tail = classified.target.tail_points_um
    assert tail is not None and len(tail) >= 2


def test_a_long_reskeletonise_into_a_mask_end_goes_to_review():
    """Replace mode clears its box first, and the box grows with the span."""
    graph, index, frame, source = scene_for(overrun())
    associations = classify.associate(index, frame, graph)
    ends, _ = lobes.find(index, frame, graph, associations)
    bridge = bridge_to(graph, graph.endpoints()[-1], ends[0])
    classified = classify.classify(bridge, associations, index, frame)
    assert classified.kind == "reskeletonise"

    params = route.GeodesicParams(mask_end_reskeletonise_radii=0.1)
    candidate = route.Candidate(classified=classified, proposal=bridge)
    route.evaluate(candidate, index, frame, params=params, graph=graph)
    assert candidate.status == "review"
    assert "existing centreline" in candidate.reason


# --------------------------------------------------------------- global choice


def _mask_end_candidate(key, *, attach_node=None, confidence=0.9):
    end = lobes.MaskEnd(key=key, point_um=np.zeros(3),
                        tangent=np.array([1.0, 0.0, 0.0]), radius_um=10.0,
                        component=1, lobe_voxels=100, elongation=5.0,
                        skeleton_um=np.zeros((2, 3)), attach_node=attach_node)
    source = classify.Association(node=key[0], point_um=np.zeros(3),
                                  tangent=np.zeros(3), radius_um=10.0,
                                  index_zyx=np.zeros(3, np.int64), component=1,
                                  distance_vox=0.0)
    target = classify.Association(node=-1, point_um=np.zeros(3),
                                  tangent=np.zeros(3), radius_um=10.0,
                                  index_zyx=np.asarray(key, np.int64), component=2,
                                  distance_vox=0.0)
    classified = classify.Classified(kind="geodesic", source=source, target=target,
                                     target_mask_end=end)
    return route.Candidate(classified=classified, confidence=confidence,
                           status="accept")


def test_two_routes_cannot_claim_one_mask_end():
    """It is a free end like any other, and a free end has one continuation."""
    graph, index, frame, _ = scene_for(two_pieces())
    a = _mask_end_candidate((9, 20, 30), confidence=0.9)
    b = _mask_end_candidate((11, 20, 30), confidence=0.5)
    b.classified.target_mask_end.key = (9, 20, 30)  # the same tip
    scored = [(c, c.confidence, c.status, c.reason) for c in (a, b)]

    decisions = select.select(graph, scored)
    assert decisions[0].accepted
    assert not decisions[1].accepted
    assert "mask free end" in decisions[1].reason


def test_a_lobes_own_component_is_used_for_the_cycle_test():
    """Joining a lobe joins whatever the lobe hangs off, skeletonised or not."""
    graph, index, frame, _ = scene_for(branched())
    node = graph.endpoints()[0]
    other = graph.endpoints()[1]
    candidate = _mask_end_candidate((5, 20, 30), attach_node=other)
    candidate.classified.source.node = node

    decisions = select.select(graph, [(candidate, 0.9, "accept", "")])
    assert not decisions[0].accepted
    assert "close a loop" in decisions[0].reason


def test_a_free_floating_lobe_has_no_cycle_to_close():
    graph, index, frame, _ = scene_for(branched())
    candidate = _mask_end_candidate((5, 20, 30), attach_node=None)
    candidate.classified.source.node = graph.endpoints()[0]

    decisions = select.select(graph, [(candidate, 0.9, "accept", "")])
    assert decisions[0].accepted


def test_a_sentinel_node_does_not_collide_with_a_component_index():
    """`-node - 1` mapped node -5 onto component 4, which is a real component."""
    forest = select._Forest({7: 4, 8: 4})
    assert forest.component(-5) != forest.component(7)
    assert not forest.would_cycle(-5, 7)


# ------------------------------------------------------------------ the driver


def test_plan_reaches_a_break_that_has_no_endpoint_on_its_far_side():
    """The whole point: without this there is nothing to propose and no repair."""
    graph, index, frame, source = scene_for(two_pieces())

    off = route.plan(graph, index, frame,
                     params=route.GeodesicParams(mask_endpoints=False))
    assert off.stats["proposals"] == 0
    assert off.candidates == []

    on = route.plan(graph, index, frame,
                    params=route.GeodesicParams(mask_endpoints=True))
    assert on.stats["mask_ends"] >= 1
    assert on.stats["mask_end_proposals"] >= 1
    mask_end_candidates = [c for c in on.candidates
                           if c.classified.target_mask_end is not None]
    assert mask_end_candidates
    assert on.lobe_report is not None and on.lobe_report.windows >= 1
    assert "mask free end" in on.summarise()


def test_explicit_proposals_switch_the_sweep_off():
    """"These pairs and no others" has to mean it."""
    graph, index, frame, _ = scene_for(two_pieces())
    plan = route.plan(graph, index, frame, proposals=[])
    assert plan.mask_ends == []
    assert plan.lobe_report is None
    assert "mask_ends" not in plan.stats


def test_gate_kwargs_reach_the_mask_end_proposer_without_choking_it():
    """`gate_kwargs` also carries flags only ..endpoints understands."""
    graph, index, frame, _ = scene_for(two_pieces())
    plan = route.plan(graph, index, frame,
                      gate_kwargs={"cone_angle_deg": 50.0,
                                   "backbone_nodes": {1, 2}})
    assert plan.stats.get("mask_end_proposals", 0) >= 0  # it ran at all
    assert route._lobe_gate_kwargs({"cone_angle_deg": 50.0,
                                    "backbone_nodes": set()}) == {
        "cone_angle_deg": 50.0}


# ---------------------------------------------------------------------- record


def test_the_record_carries_the_evidence_for_the_manufactured_end():
    graph, index, frame, _ = scene_for(two_pieces())
    plan = route.plan(graph, index, frame)
    candidate = next(c for c in plan.candidates
                     if c.classified.target_mask_end is not None)

    record = audit.candidate_record(candidate, frame)
    assert record["mask_end"]["lobe_voxels"] >= lobes.MIN_LOBE_VOXELS
    assert record["mask_end"]["elongation"] > 1.0
    assert len(record["mask_end"]["key"]) == 3


def test_two_mask_ends_from_one_source_do_not_share_a_ruling():
    """Both serialise with target node -1 and no segment; only the tip separates them."""
    a = _mask_end_candidate((9, 20, 30))
    b = _mask_end_candidate((9, 20, 44))
    b.classified.source.node = a.classified.source.node
    assert audit._key_of_candidate(a) != audit._key_of_candidate(b)

    record_a = audit.candidate_record(a)
    record_b = audit.candidate_record(b)
    assert audit._key_of_record(record_a) == audit._key_of_candidate(a)
    assert audit._key_of_record(record_b) != audit._key_of_record(record_a)


def test_a_ruling_on_a_mask_end_still_matches_after_a_re_run():
    """The tip voxel is the identity, and it survives the graph moving on."""
    candidate = _mask_end_candidate((9, 20, 30))
    record = audit.candidate_record(candidate)
    record["decision"] = {"operator": {"accept": True}}
    document = {"schema": audit.SCHEMA, "candidates": [record]}

    plan = route.Plan(candidates=[candidate], decisions=[], associations={},
                      fragments=[])
    approved, unmatched = audit.apply_decisions(plan, document)
    assert approved == [candidate]
    assert unmatched == []


# -------------------------------------------------------------- the review panel


def test_the_panel_names_a_mask_end_rather_than_node_minus_one():
    """"node -1" would be true and useless."""
    from hipct_seg_debug import controls_reconnect

    record = audit.candidate_record(_mask_end_candidate((9, 20, 30)))
    line = controls_reconnect.describe_candidate(record)
    assert "node -1" not in line
    assert "mask end 9/20/30" in line


def test_the_panel_shows_why_anyone_thought_a_vessel_was_there():
    """A reviewer approving a repair into undescribed lumen is owed the evidence."""
    from hipct_seg_debug import controls_reconnect

    record = audit.candidate_record(_mask_end_candidate((9, 20, 30), attach_node=4))
    lines = controls_reconnect.evidence_lines(record)
    assert any("undescribed lumen" in line for line in lines)
    assert any("node 4" in line for line in lines)


def test_the_panel_is_unchanged_for_an_ordinary_candidate():
    from hipct_seg_debug import controls_reconnect

    record = {"kind": "geodesic", "source": {"node": 3, "component": 1},
              "target": {"node": 9, "component": 2}, "confidence": 0.8}
    assert "node 9" in controls_reconnect.describe_candidate(record)
    assert controls_reconnect.evidence_lines(record) == []


def test_the_traced_axis_covers_the_whole_undescribed_run():
    """It sizes the re-derivation box and the growth, not just the tangent."""
    graph, index, frame, _ = scene_for(overrun())
    ends, _ = sweep(graph, index, frame)
    end = max(ends, key=lambda e: e.point_um[0])

    axis = np.asarray(end.skeleton_um)
    assert len(axis) >= 4
    length = float(np.linalg.norm(np.diff(axis, axis=0), axis=1).sum())
    # The undescribed run here is the trunk from about x=27 to its tip at x=34.
    assert length > 4.0 * end.radius_um
    assert np.allclose(axis[0], end.point_um)  # tip first


def test_the_growth_for_a_mask_end_covers_its_lobe():
    """Three radii would describe the first tenth of a lobe and stop."""
    from hipct_seg_debug.edit.reconnect.geodesic import apply as apply_mod

    graph, index, frame, _ = scene_for(overrun())
    ends, _ = sweep(graph, index, frame)
    end = max(ends, key=lambda e: e.point_um[0])
    bridge = bridge_to(graph, graph.endpoints()[-1], end)
    associations = classify.associate(index, frame, graph)
    classified = classify.classify(bridge, associations, index, frame)
    candidate = route.Candidate(classified=classified, proposal=bridge)

    axis = np.asarray(end.skeleton_um)
    length = float(np.linalg.norm(np.diff(axis, axis=0), axis=1).sum())
    assert apply_mod._grow_for(candidate) >= length


def test_an_ordinary_candidate_keeps_the_default_growth():
    from hipct_seg_debug.edit.reconnect.geodesic import apply as apply_mod

    candidate = _mask_end_candidate((9, 20, 30))
    candidate.classified.target_mask_end = None
    assert apply_mod._grow_for(candidate) is None


def test_the_proposer_says_which_gate_refused_what():
    """A handful of proposals from a hundred ends is expected, not a misfire."""
    graph, index, frame, _ = scene_for(overrun())
    associations = classify.associate(index, frame, graph)
    ends, _ = lobes.find(index, frame, graph, associations)

    stats: dict = {}
    lobes.propose(graph, ends, associations, stats=stats)
    assert stats["ends"] == len(ends)
    assert stats["pairs_in_reach"] >= 1
    assert stats["rejected_facing"] >= 1
    assert stats["proposed"] == 0


def test_the_summary_names_the_binding_gate():
    graph, index, frame, _ = scene_for(overrun())
    plan = route.plan(graph, index, frame)
    assert "refused:" in plan.summarise()
    assert "facing" in plan.summarise()


def test_a_collapsed_end_is_measured_in_the_graphs_own_convention():
    """A slit has the perimeter of the vessel it was and the half-width of nothing.

    This one is 13 voxels wide and 3 thick, so distance-to-background at its axis is
    20 um while its perimeter-equivalent radius -- what the graph carries -- is about
    45. Reporting the first would make the radius-ratio gate read a mismatch between
    a vessel and itself.
    """
    from .conftest_geometry import slit

    volume = slit(SHAPE, 6, 1, 5, 45, cy=CY, cz=CZ)
    graph, index, frame, _ = scene_for(volume, radius_um=60.0)
    ends, report = sweep(graph, index, frame)

    assert ends, lobes.summarise(ends, report)
    end = max(ends, key=lambda e: e.point_um[0])
    assert end.radius_um > 1.8 * (2.0 * SPACING)  # well past the half-thickness
    assert end.radius_um < 100.0
