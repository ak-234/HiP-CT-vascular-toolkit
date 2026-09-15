"""What the geodesic reconnector decides, and what it refuses.

The tests are organised by the way a repair can be *wrong*, not by module, because
that is how this code fails. Nothing here crashes; it quietly connects two vessels
that were never connected, or inflates a collapsed lumen into a round one, or
repairs a break that did not exist. Each of those is a section below.

Runtime note: most tests pass ``alternatives=1``. Searching for alternatives means
re-running A* over a corridor that has been told its best route is expensive, which
is deliberately near-exhaustive and is the slowest thing in the package. The two
tests that are *about* ambiguity ask for alternatives; the rest do not need them.
"""

from __future__ import annotations

import numpy as np
import pytest

from hipct_seg_debug.edit.reconnect.geodesic import (
    apply as apply_mod,
    astar,
    classify,
    components,
    corridor,
    cost,
    select,
    shape as shape_mod,
)
from hipct_seg_debug.edit.reconnect.geodesic import GeodesicParams, plan

from .conftest_geodesic import (
    CY,
    CZ,
    SHAPE,
    axis_run,
    broken_graph,
    curved_run,
    curved_tube,
    cylinder,
    decode,
    make_frame,
    mask_source,
    ribbon,
    slit,
)


FAST = GeodesicParams(alternatives=1)


def _plan(volume, graph, *, params=FAST, stack=None):
    frame = make_frame(SHAPE)
    source = mask_source(np.asarray(volume, dtype=np.uint8))
    index = components.build(source)
    return plan(graph, index, frame, stack=stack, params=params), index, frame, source


def _accepted_kinds(result):
    return sorted(c.kind for c in result.accepted())


# ===================================================================== classify
# The first way to be wrong: repairing a break that is not there, or missing that
# a break is of a kind that needs no new voxels at all.


def test_mask_connected_break_is_reskeletonised_not_painted():
    """Both ends in one mask component: the lumen is continuous already.

    This is the case a purely geometric proposer cannot see and the one whose
    failure is silent -- painting a route through voxels that are *already*
    foreground succeeds by every count anyone checks afterwards, while having
    overwritten a real lumen with a synthetic one.
    """
    frame = make_frame(SHAPE)
    volume = cylinder(SHAPE, 3, 5, 55)  # unbroken tube
    graph = broken_graph(frame, (5, 25), (33, 55))  # ...but a broken centreline

    result, _index, _frame, source = _plan(volume, graph)
    accepted = result.accepted()

    assert accepted, result.summarise()
    assert all(c.kind == "reskeletonise" for c in accepted)
    assert accepted[0].evidence["adds_voxels"] is False
    assert accepted[0].route is None  # no search was run at all
    # And nothing was written to the mask.
    assert source.edits.is_empty


def test_distinct_mask_components_are_classified_geodesic():
    frame = make_frame(SHAPE)
    volume = cylinder(SHAPE, 3, 5, 25) | cylinder(SHAPE, 3, 33, 55)
    graph = broken_graph(frame, (5, 25), (33, 55))

    result, _i, _f, _s = _plan(volume, graph)
    assert {c.kind for c in result.candidates} == {"geodesic"}


def test_an_endpoint_off_the_mask_is_unassociated_and_goes_to_review():
    """A graph and a mask that disagree about where a vessel is, is a finding.

    Snapping to whatever component happens to be nearest would turn a real
    disagreement into a confident repair built on a guessed premise.
    """
    frame = make_frame(SHAPE)
    volume = cylinder(SHAPE, 3, 5, 25) | cylinder(SHAPE, 3, 33, 55)
    # The right-hand run's centreline is eight voxels off its own lumen in z.
    graph = broken_graph(frame, (5, 25), (33, 55))
    from .conftest_geodesic import _merge  # noqa: PLC0415 - local to this case

    graph = _merge(axis_run(frame, 5, 25, 20.0),
                   axis_run(frame, 33, 55, 20.0, cz=CZ + 8))

    result, _i, _f, _s = _plan(volume, graph)
    assert any(c.kind == "unassociated" for c in result.candidates)
    for candidate in result.candidates:
        if candidate.kind == "unassociated":
            assert candidate.status == "review"


def test_debris_is_excluded_but_an_elongated_orphan_is_plausible():
    """Size alone is not enough: a compact blob near a gap is not a vessel."""
    frame = make_frame(SHAPE)
    volume = cylinder(SHAPE, 3, 5, 22) | cylinder(SHAPE, 3, 38, 55)
    volume = volume.copy()
    volume[CZ - 1:CZ + 2, CY - 1:CY + 2, 26:34] = 1  # an orphan strand in the gap
    volume[5:8, 5:8, 5:8] = 1  # ...and a compact lump of debris elsewhere

    source = mask_source(volume)
    index = components.build(source)
    graph = broken_graph(frame, (5, 22), (38, 55))

    fragments = classify.fragment_candidates(index, frame, graph, reach_um=2000.0)
    by_plausible = {f.plausible for f in fragments}
    assert by_plausible == {True, False}, [(f.voxels, f.elongation) for f in fragments]

    strand = next(f for f in fragments if f.plausible)
    lump = next(f for f in fragments if not f.plausible)
    assert strand.elongation > lump.elongation
    assert "blob" in lump.reason or "voxels" in lump.reason


def test_a_fragment_between_two_ends_is_carried_into_the_classification():
    frame = make_frame(SHAPE)
    volume = cylinder(SHAPE, 3, 5, 22) | cylinder(SHAPE, 3, 38, 55)
    volume = volume.copy()
    volume[CZ - 1:CZ + 2, CY - 1:CY + 2, 26:34] = 1

    source = mask_source(volume)
    index = components.build(source)
    graph = broken_graph(frame, (5, 22), (38, 55))
    associations = classify.associate(index, frame, graph)
    fragments = classify.fragment_candidates(index, frame, graph, reach_um=2000.0)

    from hipct_seg_debug.edit.reconnect import endpoints

    pair = next(b for b in endpoints.propose(graph) if b.accepted)
    classified = classify.classify(pair, associations, index, frame,
                                   fragments=fragments)
    assert classified.kind == "fragment"
    assert [f.component for f in classified.fragments]


# ========================================================================= cost
# The second way to be wrong: believing the image says something it does not.


def test_unrelated_foreground_is_blocked_not_cheap():
    """The false-connection guard.

    Foreground is the lowest-cost material in the volume by every term in the
    field, so a neighbouring vessel is the most attractive route available. If it
    is merely penalised rather than removed, a shortest path dives into it and
    runs along it, and that is precisely the wrong answer this package exists to
    refuse.
    """
    frame = make_frame(SHAPE)
    volume = cylinder(SHAPE, 3, 5, 25) | cylinder(SHAPE, 3, 33, 55)
    volume = volume | cylinder(SHAPE, 3, 0, 60, cy=CY + 9)  # a neighbour running past

    source = mask_source(volume)
    index = components.build(source)
    points = np.asarray(frame.seg_to_um([[24, CY, CZ], [33, CY, CZ]]))
    box = corridor.for_candidate(frame, index, points, radius_um=30.0)

    mine = {index.label_at(CZ, CY, 10), index.label_at(CZ, CY, 50)}
    field = cost.build(box, index, mine, radius_um=30.0)

    assert field.blocked.any()
    assert not np.isfinite(field.cost[field.blocked]).any()
    assert field.competing  # and it says which component it is competing with
    # The vessel's own foreground is emphatically not blocked.
    assert not field.blocked[field.mine].any()


def test_calibration_separates_lumen_from_wall_and_says_when_it_cannot():
    volume = np.zeros((12, 12, 12), dtype=np.float32)
    volume[4:8, 4:8, 4:8] = 10.0  # a dark lumen in a bright surround
    volume[volume == 0] = 100.0
    inside = np.zeros((12, 12, 12), dtype=bool)
    inside[4:8, 4:8, 4:8] = True

    calibration = cost.calibrate(volume, inside)
    assert calibration.separated
    assert calibration.lumen_mu < calibration.wall_mu
    assert calibration.lumen_likelihood(np.float32(10.0)) > 0.9
    assert calibration.lumen_likelihood(np.float32(100.0)) < 0.1

    flat = cost.calibrate(np.full((8, 8, 8), 5.0, np.float32),
                          np.ones((8, 8, 8), bool))
    assert not flat.separated
    # No contrast means no opinion, not a confident one.
    assert flat.lumen_likelihood(np.float32(5.0)) == pytest.approx(0.5)


def test_flux_medialness_responds_to_a_slit_a_tube_and_a_ribbon():
    """The collapse-aware claim, made concrete.

    A Frangi-style tube filter is near-blind to a one-voxel slit. Converging wall
    gradients are not, and that is the whole reason the flux term is in the field.
    """
    responses = {}
    for name, mask in (
        ("tube", cylinder(SHAPE, 3, 10, 50)),
        ("ribbon", ribbon(SHAPE, 6, 2, 10, 50)),
        ("slit", slit(SHAPE, 6, 0, 10, 50)),
    ):
        volume = np.where(np.asarray(mask, bool), 10.0, 100.0).astype(np.float32)
        flux = cost.flux_medialness(volume, sigma=1.5, dark_lumen=True)
        inside = np.asarray(mask, bool)
        responses[name] = float(flux[inside].mean()) / max(float(flux[~inside].mean()),
                                                           1e-9)
    for name, ratio in responses.items():
        assert ratio > 1.5, f"{name} scored {ratio:.2f} inside vs outside"


def test_support_is_calibrated_against_the_intact_tails():
    """Support near 1 means "as good as this vessel's own intact centreline".

    Without this the field is in arbitrary units and every gate downstream becomes
    a per-scan constant again.
    """
    frame = make_frame(SHAPE)
    volume = cylinder(SHAPE, 3, 5, 25) | cylinder(SHAPE, 3, 33, 55)
    source = mask_source(volume)
    index = components.build(source)
    points = np.asarray(frame.seg_to_um([[24, CY, CZ], [33, CY, CZ]]))
    box = corridor.for_candidate(frame, index, points, radius_um=30.0)

    tails = np.asarray(frame.seg_to_um(
        [[x, CY, CZ] for x in list(range(18, 25)) + list(range(33, 40))]
    ))
    field = cost.build(box, index, {1, 2}, radius_um=30.0, calibration_points=tails)

    on_tail = field.support[tuple((box.to_global(tails) - field.lo_zyx).T)]
    assert np.median(on_tail) == pytest.approx(1.0, abs=0.35)
    # ...and the gap, which has no vessel in it, scores materially worse.
    gap = np.asarray(frame.seg_to_um([[29, CY, CZ]]))
    in_gap = field.support[tuple((box.to_global(gap) - field.lo_zyx).T)]
    assert float(in_gap[0]) < 0.9 * float(np.median(on_tail))


# ======================================================================== astar
# The third way to be wrong: finding a route that is short rather than right.


def test_search_prefers_the_supported_corridor_over_the_short_chord():
    """A cheap detour beats an expensive straight line, which greedy cannot do."""
    field_cost = np.full((5, 21, 21), 5.0, dtype=np.float64)
    field_cost[2, 2, :] = 0.01   # a cheap lane along the far edge
    field_cost[2, :, 0] = 0.01
    field_cost[2, :, 20] = 0.01
    field_cost[2, 18, :] = 0.01

    field = cost.CostField(
        cost=field_cost, support=np.ones_like(field_cost, np.float32),
        blocked=~np.isfinite(field_cost), mine=np.zeros(field_cost.shape, bool),
        described=np.zeros(field_cost.shape, np.float32),
        labels=np.zeros(field_cost.shape, np.int32),
        lo_zyx=np.zeros(3, np.int64), spacing_zyx=np.ones(3),
        calibration=cost.calibrate(field_cost, np.zeros(field_cost.shape, bool)),
    )
    path, total, reason, _stats = astar.solve(
        field, np.array([2, 2, 0]), np.array([[2, 18, 20]])
    )
    assert reason == "found"
    # The straight diagonal would be ~24 units of cost-5 material; the lane is ~40
    # steps of cost-0.01. Taking the lane is the whole point.
    assert total < 5.0
    assert len(path) > 25


def test_blocked_material_is_impassable():
    field_cost = np.full((5, 9, 9), 0.1)
    field_cost[:, :, 4] = np.inf  # a wall straight across
    field = cost.CostField(
        cost=field_cost, support=np.ones(field_cost.shape, np.float32),
        blocked=~np.isfinite(field_cost), mine=np.zeros(field_cost.shape, bool),
        described=np.zeros(field_cost.shape, np.float32),
        labels=np.zeros(field_cost.shape, np.int32),
        lo_zyx=np.zeros(3, np.int64), spacing_zyx=np.ones(3),
        calibration=cost.calibrate(np.ones(field_cost.shape, np.float32),
                                   np.zeros(field_cost.shape, bool)),
    )
    path, _c, reason, _s = astar.solve(field, np.array([2, 4, 0]),
                                       np.array([[2, 4, 8]]))
    assert path is None and reason != "found"


def test_turning_costs_something():
    """Direction is in the state, so curvature has a price rather than being free."""
    field_cost = np.full((3, 3, 12), 0.1)
    field = cost.CostField(
        cost=field_cost, support=np.ones(field_cost.shape, np.float32),
        blocked=np.zeros(field_cost.shape, bool),
        mine=np.zeros(field_cost.shape, bool),
        described=np.zeros(field_cost.shape, np.float32),
        labels=np.zeros(field_cost.shape, np.int32),
        lo_zyx=np.zeros(3, np.int64), spacing_zyx=np.ones(3),
        calibration=cost.calibrate(np.ones(field_cost.shape, np.float32),
                                   np.zeros(field_cost.shape, bool)),
    )
    straight, _c, _r, _s = astar.search(
        field.cost, np.array([1, 1, 0]), np.array([[1, 1, 11]]),
        spacing_zyx=np.ones(3), turn_weight=0.0,
    )
    _p, with_turns, _r, _s = astar.search(
        field.cost, np.array([1, 1, 0]), np.array([[1, 1, 11]]),
        spacing_zyx=np.ones(3), start_direction=np.array([0.0, 1.0, 0.0]),
        turn_weight=5.0,
    )
    assert straight is not None
    # Starting off pointing across the tube, a heavy turn weight must show up as
    # extra cost rather than being absorbed silently.
    assert with_turns > 0.1 * 11


def test_alternatives_are_spatially_distinct():
    field_cost = np.full((3, 15, 15), 8.0)
    field_cost[1, 2, :] = 0.05   # two separate cheap lanes
    field_cost[1, 12, :] = 0.05
    field_cost[1, :, 0] = 0.05
    field_cost[1, :, 14] = 0.05
    field = cost.CostField(
        cost=field_cost, support=np.ones(field_cost.shape, np.float32),
        blocked=np.zeros(field_cost.shape, bool),
        mine=np.zeros(field_cost.shape, bool),
        described=np.zeros(field_cost.shape, np.float32),
        labels=np.zeros(field_cost.shape, np.int32),
        lo_zyx=np.zeros(3, np.int64), spacing_zyx=np.ones(3),
        calibration=cost.calibrate(np.ones(field_cost.shape, np.float32),
                                   np.zeros(field_cost.shape, bool)),
    )
    routes = astar.routes(field, np.array([1, 2, 0]), np.array([[1, 12, 14]]),
                          alternatives=3)
    assert len(routes) >= 2
    assert routes[0].deviation_from(routes[1]) >= astar.DISTINCT_VOXELS
    assert routes[1].cost >= routes[0].cost


def test_waypoints_redirect_the_route_deterministically():
    """An operator's correction has to be repeatable, or it is not a correction."""
    field_cost = np.full((3, 15, 15), 0.5)
    field = cost.CostField(
        cost=field_cost, support=np.ones(field_cost.shape, np.float32),
        blocked=np.zeros(field_cost.shape, bool),
        mine=np.zeros(field_cost.shape, bool),
        described=np.zeros(field_cost.shape, np.float32),
        labels=np.zeros(field_cost.shape, np.int32),
        lo_zyx=np.zeros(3, np.int64), spacing_zyx=np.ones(3),
        calibration=cost.calibrate(np.ones(field_cost.shape, np.float32),
                                   np.zeros(field_cost.shape, bool)),
    )
    start, goal = np.array([1, 7, 0]), np.array([[1, 7, 14]])
    plain = astar.routes(field, start, goal, alternatives=1)[0]
    via = np.array([1, 13, 7])
    steered = astar.routes(field, start, goal, alternatives=1, waypoints=[via])[0]
    again = astar.routes(field, start, goal, alternatives=1, waypoints=[via])[0]

    assert any(np.array_equal(v, via) for v in steered.path_zyx)
    assert not any(np.array_equal(v, via) for v in plain.path_zyx)
    np.testing.assert_array_equal(steered.path_zyx, again.path_zyx)


def test_unsupported_run_measures_the_longest_contiguous_stretch():
    """Contiguous, not total: scattered weakness is noise, one long hole is not."""
    path = np.stack([np.zeros(10, int), np.zeros(10, int), np.arange(10)], axis=1)
    support = np.array([1, 1, 0.0, 1, 0.0, 0.0, 0.0, 1, 1, 1])
    route = astar.Route(path_zyx=path, cost=1.0, support=support, length_um=9.0)
    # Three consecutive weak steps of length 1 each.
    assert route.unsupported_um(np.ones(3), fraction=0.25) == pytest.approx(3.0)


# ======================================================================= select
# The fourth way to be wrong: two locally-good routes that cannot both be right.


def _stub(source_node, target_node=None, target_segment=None, radius=10.0):
    association = classify.Association(
        node=source_node, point_um=np.zeros(3), tangent=np.array([1.0, 0, 0]),
        radius_um=radius, index_zyx=np.zeros(3, np.int64), component=1,
        distance_vox=0.0,
    )
    target = None if target_node is None else classify.Association(
        node=target_node, point_um=np.array([100.0, 0, 0]),
        tangent=np.array([-1.0, 0, 0]), radius_um=radius,
        index_zyx=np.zeros(3, np.int64), component=2, distance_vox=0.0,
    )
    if target_segment is not None and target is None:
        target = classify.Association(
            node=-1, point_um=np.array([100.0, 0, 0]), tangent=np.zeros(3),
            radius_um=radius, index_zyx=np.zeros(3, np.int64), component=2,
            distance_vox=0.0,
        )
    return classify.Classified(kind="geodesic", source=association, target=target,
                               target_segment=target_segment)


def test_one_continuation_per_free_end():
    frame = make_frame(SHAPE)
    graph = broken_graph(frame, (5, 25), (33, 55))
    ends = graph.endpoints()
    scored = [
        (_stub(ends[1], ends[2]), 0.9, "accept", "best"),
        (_stub(ends[1], ends[3]), 0.5, "accept", "also plausible"),
    ]
    decisions = select.select(graph, scored)
    assert [d.accepted for d in decisions] == [True, False]
    assert "already continued" in decisions[1].reason


def test_a_second_route_between_joined_components_is_a_cycle():
    frame = make_frame(SHAPE)
    graph = broken_graph(frame, (5, 25), (33, 55))
    ends = graph.endpoints()
    scored = [
        (_stub(ends[1], ends[2]), 0.9, "accept", "joins the two pieces"),
        (_stub(ends[0], ends[3]), 0.8, "accept", "joins them again"),
    ]
    decisions = select.select(graph, scored)
    assert decisions[0].accepted and not decisions[1].accepted
    assert "loop" in decisions[1].reason


def test_two_fragments_attaching_to_each_others_side_is_also_a_cycle():
    """The hole an endpoint-only cycle test leaves open.

    A T-junction consumes no target endpoint, so a check that only looks at
    endpoints sees no conflict -- while the two attachments close a loop just as
    firmly as an end-to-end pair would.
    """
    frame = make_frame(SHAPE)
    graph = broken_graph(frame, (5, 25), (33, 55))
    ends = graph.endpoints()
    sids = graph.segment_ids()
    scored = [
        (_stub(ends[1], target_segment=sids[1]), 0.9, "accept", "a onto b"),
        (_stub(ends[2], target_segment=sids[0]), 0.8, "accept", "b onto a"),
    ]
    decisions = select.select(graph, scored)
    assert decisions[0].accepted and not decisions[1].accepted
    assert "loop" in decisions[1].reason


def test_two_daughters_may_attach_to_one_parent_if_far_enough_apart():
    """Separate branches, separate landings.

    The two daughters must come from *different* components. Two free ends of the
    same fragment attaching to one parent is a cycle, not a pair of daughters, and
    :func:`test_two_fragments_attaching_to_each_others_side_is_also_a_cycle` is
    where that case belongs.
    """
    from .conftest_geodesic import _merge

    frame = make_frame(SHAPE)
    graph = _merge(axis_run(frame, 5, 55, 20.0),                 # the parent
                   axis_run(frame, 10, 20, 12.0, cy=CY + 8),     # daughter one
                   axis_run(frame, 35, 45, 12.0, cy=CY + 8))     # daughter two
    parent = graph.segment_ids()[0]
    daughters = [s for s in graph.segment_ids() if s != parent]
    first = graph.segment(daughters[0])["node1"]
    second = graph.segment(daughters[1])["node1"]

    near = _stub(first, target_segment=parent)
    near.target.point_um = np.array([100.0, 0.0, 0.0])
    far = _stub(second, target_segment=parent)
    far.target.point_um = np.array([900.0, 0.0, 0.0])
    twin = _stub(second, target_segment=parent)
    twin.target.point_um = np.array([105.0, 0.0, 0.0])

    ok = select.select(graph, [(near, 0.9, "accept", ""), (far, 0.8, "accept", "")])
    assert all(d.accepted for d in ok), [d.reason for d in ok]

    clash = select.select(graph, [(near, 0.9, "accept", ""),
                                  (twin, 0.8, "accept", "")])
    assert clash[0].accepted and not clash[1].accepted
    assert "already attaches" in clash[1].reason


def test_review_and_reject_do_not_consume_an_endpoint():
    """An unruled candidate must not spend the endpoint a confident one needs."""
    frame = make_frame(SHAPE)
    graph = broken_graph(frame, (5, 25), (33, 55))
    ends = graph.endpoints()
    scored = [
        (_stub(ends[1], ends[2]), 0.95, "review", "ambiguous"),
        (_stub(ends[1], ends[3]), 0.60, "accept", "clear"),
    ]
    decisions = select.select(graph, scored)
    assert decisions[0].status == "review"
    assert decisions[1].accepted


# ======================================================================== shape
# The fifth way to be wrong: re-inflating a collapsed specimen.


def test_rotation_minimizing_frame_does_not_twist_on_a_straight_run():
    points = np.stack([np.arange(20.0), np.zeros(20), np.zeros(20)], axis=1)
    tangents, normals, binormals = shape_mod.rotation_minimizing_frame(points)

    assert np.allclose(np.linalg.norm(normals, axis=1), 1.0)
    # Orthonormal everywhere...
    assert np.allclose(np.einsum("ij,ij->i", tangents, normals), 0.0, atol=1e-9)
    assert np.allclose(np.einsum("ij,ij->i", normals, binormals), 0.0, atol=1e-9)
    # ...and, crucially, unrotated. A Frenet frame is undefined here.
    assert np.allclose(normals, normals[0], atol=1e-9)


def test_rotation_minimizing_frame_survives_an_inflection():
    """Where a Frenet frame flips through 180 degrees, this one must not."""
    t = np.linspace(0.0, 4 * np.pi, 120)
    points = np.stack([t, np.sin(t), np.zeros_like(t)], axis=1)
    _tangents, normals, _b = shape_mod.rotation_minimizing_frame(points)
    steps = np.einsum("ij,ij->i", normals[:-1], normals[1:])
    assert steps.min() > 0.9, "the transported normal jumped at an inflection"


def test_a_slit_keeps_its_shape_across_the_gap():
    """The central claim: transport the observed cross-section, do not assume one.

    A capsule sweep would fill the gap with a circular lumen. What must come out
    instead is a section as flat as the ends it came from.
    """
    frame = make_frame(SHAPE)
    volume = slit(SHAPE, 6, 0, 5, 28) | slit(SHAPE, 6, 0, 32, 55)
    source = mask_source(volume)
    index = components.build(source)

    left = classify.Association(
        node=0, point_um=np.asarray(frame.seg_to_um([[27, CY, CZ]]))[0],
        tangent=np.array([1.0, 0, 0]), radius_um=20.0,
        index_zyx=np.array([CZ, CY, 27]), component=index.label_at(CZ, CY, 27),
        distance_vox=0.0,
    )
    right = classify.Association(
        node=1, point_um=np.asarray(frame.seg_to_um([[32, CY, CZ]]))[0],
        tangent=np.array([-1.0, 0, 0]), radius_um=20.0,
        index_zyx=np.array([CZ, CY, 32]), component=index.label_at(CZ, CY, 32),
        distance_vox=0.0,
    )
    section_a, section_b = shape_mod.measure_ends(index, frame, left, right)
    assert section_a.valid and section_b.valid
    # A slit is flat: its perimeter-equivalent radius far exceeds its area one.
    assert section_a.flatness > 1.2
    assert section_a.r_perimeter > section_a.r_area

    path = np.asarray(frame.seg_to_um([[x, CY, CZ] for x in range(27, 33)]))
    completion = shape_mod.transport(path, section_a, section_b, frame)
    assert len(completion.voxels_zyx)

    # The painted cross-section must stay flat, not become a disc.
    painted = np.zeros(SHAPE, dtype=bool)
    v = completion.voxels_zyx
    inside = np.all((v >= 0) & (v < np.asarray(SHAPE)), axis=1)
    painted[tuple(v[inside].T)] = True
    column = painted[:, :, 30]
    if column.any():
        extent_z = np.ptp(np.flatnonzero(column.any(axis=1))) + 1
        extent_y = np.ptp(np.flatnonzero(column.any(axis=0))) + 1
        assert extent_y > 2 * extent_z, (
            f"the repaired section is {extent_y} x {extent_z}: too round for a slit"
        )


def test_minimum_core_is_26_connected_and_minimal():
    frame = make_frame(SHAPE)
    path = np.asarray(frame.seg_to_um([[10, 10, 10], [13, 12, 11], [16, 12, 11]]))
    core = shape_mod.minimum_core(path, frame)

    assert len(core) >= 3
    steps = np.abs(np.diff(core, axis=0)).max(axis=1)
    assert set(steps.tolist()) <= {1}, "the core is not a 26-connected chain"
    assert len(core) == len(np.unique(core, axis=0)), "the core repeats a voxel"


def test_graph_radii_are_perimeter_equivalent_and_separate_from_the_mask():
    """The mask records the collapsed observation; the graph carries calibre.

    Conflating the two is what silently re-inflates a specimen, so the two numbers
    are produced by different code paths and this pins that they differ.
    """
    frame = make_frame(SHAPE)
    volume = slit(SHAPE, 6, 0, 5, 55)
    index = components.build(mask_source(volume))
    section = shape_mod.extract_section(
        index, frame, np.asarray(frame.seg_to_um([[30, CY, CZ]]))[0],
        np.array([1.0, 0, 0]), component=1, radius_um=20.0,
    )
    assert section.valid
    assert section.r_perimeter > section.r_area * 1.2


# ===================================================================== the plan
# End to end, on the shapes the package claims to handle.


@pytest.mark.parametrize("name", ["tube", "ribbon", "slit"])
def test_a_two_voxel_gap_is_repaired_in_every_cross_section(name):
    frame = make_frame(SHAPE)
    builder = {
        "tube": lambda a, b: cylinder(SHAPE, 3, a, b),
        "ribbon": lambda a, b: ribbon(SHAPE, 6, 2, a, b),
        "slit": lambda a, b: slit(SHAPE, 6, 0, a, b),
    }[name]
    volume = builder(5, 28) | builder(30, 55)
    graph = broken_graph(frame, (5, 28), (30, 55))

    result, index, _frame, _source = _plan(volume, graph)
    assert index.n == 2
    accepted = result.accepted()
    assert len(accepted) == 1, result.summarise()
    assert accepted[0].kind == "geodesic"
    assert accepted[0].evidence["mask_gap_voxels"] <= 2


def test_a_curved_vessel_is_followed_rather_than_chorded():
    """The route must bulge with the vessel; the chord leaves the lumen."""
    frame = make_frame(SHAPE)
    volume = curved_tube(SHAPE, 3, 5, 26) | curved_tube(SHAPE, 3, 30, 55)
    from .conftest_geodesic import _merge

    graph = _merge(curved_run(frame, 5, 26, 20.0),
                   curved_run(frame, 30, 55, 20.0))

    result, _index, frame_, _source = _plan(volume, graph)
    routed = [c for c in result.candidates if c.route is not None]
    assert routed, result.summarise()
    best = max(routed, key=lambda c: c.confidence)
    path = best.route.path_um(frame_)
    chord = np.linalg.norm(path[-1] - path[0])
    assert best.route.length_um >= chord * 0.99


def test_a_long_gap_with_no_image_support_is_not_accepted_automatically():
    """Without raw greyscale the mask alone cannot justify inventing a vessel."""
    frame = make_frame(SHAPE)
    volume = cylinder(SHAPE, 3, 5, 20) | cylinder(SHAPE, 3, 40, 55)
    graph = broken_graph(frame, (5, 20), (40, 55), radius_um=40.0)

    result, _i, _f, _s = _plan(volume, graph)
    assert not result.accepted(), result.summarise()
    for candidate in result.candidates:
        assert candidate.status in ("review", "reject")
        if candidate.status == "review":
            assert "raw" in candidate.reason


def test_two_parallel_vessels_are_not_welded_together():
    """The classic false connection: near, but running alongside rather than into."""
    frame = make_frame(SHAPE)
    volume = cylinder(SHAPE, 3, 5, 55) | cylinder(SHAPE, 3, 5, 55, cy=CY + 9)
    from .conftest_geodesic import _merge

    graph = _merge(axis_run(frame, 5, 55, 20.0),
                   axis_run(frame, 5, 55, 20.0, cy=CY + 9))

    result, _i, _f, _s = _plan(volume, graph)
    # The cone gates should refuse these outright; anything that survives them must
    # not be applied on the strength of proximity alone.
    for candidate in result.accepted():
        assert candidate.kind == "reskeletonise", (
            f"a parallel neighbour was joined: {candidate!r}"
        )


@pytest.mark.parametrize(
    "second_cost, expected", [(1.05, "review"), (2.00, "accept")]
)
def test_a_near_tie_between_routes_forces_review(second_cost, expected):
    """Two comparably good ways through is a question for a person, not a coin toss.

    Tested on the decision rule directly rather than through a fixture engineered
    to produce a tie. Building a mask with two genuinely equal-cost corridors means
    tuning a synthetic volume until the numbers land where the test wants them,
    which proves the fixture rather than the rule; feeding the gate two routes
    whose costs differ by a known margin proves the rule.
    """
    from hipct_seg_debug.edit.reconnect.geodesic import route as route_mod

    frame = make_frame(SHAPE)
    volume = slit(SHAPE, 6, 1, 5, 28) | slit(SHAPE, 6, 1, 30, 55)
    source = mask_source(volume)
    index = components.build(source)
    graph = broken_graph(frame, (5, 28), (30, 55))

    from hipct_seg_debug.edit.reconnect import endpoints

    pair = next(b for b in endpoints.propose(graph) if b.accepted)
    associations = classify.associate(index, frame, graph)
    candidate = route_mod.Candidate(
        classified=classify.classify(pair, associations, index, frame), proposal=pair
    )
    route_mod.evaluate(candidate, index, frame, params=GeodesicParams(alternatives=1),
                       graph=graph)
    assert candidate.route is not None

    # One route, and a rival at a controlled distance behind it.
    rival = astar.Route(
        path_zyx=candidate.route.path_zyx + np.array([3, 0, 0]),
        cost=candidate.route.cost * second_cost,
        support=candidate.route.support, length_um=candidate.route.length_um,
    )
    candidate.alternatives = [rival]

    points = np.asarray([candidate.classified.source.point_um,
                         candidate.classified.target.point_um])
    box = corridor.for_candidate(frame, index, points, radius_um=20.0)
    field = cost.build(box, index,
                       {candidate.classified.source.component,
                        candidate.classified.target.component}, radius_um=20.0)
    params = GeodesicParams(alternatives=3, mask_only_gap_voxels=99)
    route_mod._gate(candidate, field, box, params, 20.0)

    assert candidate.status == expected, candidate.reason
    margin = candidate.evidence["alternative_margin"]
    assert margin == pytest.approx(second_cost - 1.0, abs=0.02)
    if expected == "review":
        assert "second route" in candidate.reason


# ======================================================================== apply
# The sixth way to be wrong: writing a graph and a mask that disagree.


def test_applying_reduces_both_component_counts_and_is_one_undo():
    frame = make_frame(SHAPE)
    volume = slit(SHAPE, 6, 1, 5, 28) | slit(SHAPE, 6, 1, 30, 55)
    graph = broken_graph(frame, (5, 28), (30, 55))
    source = mask_source(volume)
    index = components.build(source)

    before_graph = len(graph.components())
    result = plan(graph, index, frame, params=FAST)
    applied = apply_mod.apply_plan(graph, result, source, frame)

    assert any(a.ok for a in applied), [a.describe() for a in applied]
    assert len(graph.components()) == before_graph - 1
    after = components.from_array(decode(source) > 0)
    assert after.n == index.n - 1

    # One press of undo, and the graph is back.
    assert graph.undo() is not None
    assert len(graph.components()) == before_graph


def test_the_repair_does_not_merge_an_unrelated_neighbour():
    """A component count that falls by more than the number of repairs is a bug."""
    frame = make_frame(SHAPE)
    volume = (slit(SHAPE, 5, 1, 5, 28) | slit(SHAPE, 5, 1, 30, 55)
              | cylinder(SHAPE, 2, 5, 55, cy=CY + 11))
    graph = broken_graph(frame, (5, 28), (30, 55))
    source = mask_source(volume)
    index = components.build(source)

    result = plan(graph, index, frame, params=FAST)
    applied = apply_mod.apply_plan(graph, result, source, frame)
    repairs = sum(1 for a in applied if a.ok and a.voxels_added)

    after = components.from_array(decode(source) > 0)
    assert index.n - after.n <= repairs, (
        f"{index.n} -> {after.n} components after {repairs} mask repair(s): "
        "something unrelated was merged"
    )
    assert after.n >= 2, "the unrelated neighbour was absorbed"


def test_provenance_is_written_on_every_created_edge():
    frame = make_frame(SHAPE)
    volume = slit(SHAPE, 6, 1, 5, 28) | slit(SHAPE, 6, 1, 30, 55)
    graph = broken_graph(frame, (5, 28), (30, 55))
    source = mask_source(volume)
    index = components.build(source)

    original = set(graph.segment_ids())
    result = plan(graph, index, frame, params=FAST)
    apply_mod.apply_plan(graph, result, source, frame)

    created = [s for s in graph.segments if s["id"] not in original]
    assert created
    for segment in created:
        assert segment[apply_mod.ORIGIN_FIELD] in (
            apply_mod.GEODESIC, apply_mod.RESKELETONISED
        )
        assert apply_mod.SCORE_FIELD in segment
        assert apply_mod.REVIEWED_FIELD in segment
    counts = apply_mod.origin_counts(graph)
    assert counts.get("original") == len(original)


def test_a_mask_connected_repair_adds_no_voxels():
    """The safety property of the re-skeletonise path, asserted rather than assumed."""
    frame = make_frame(SHAPE)
    volume = cylinder(SHAPE, 3, 5, 55)
    graph = broken_graph(frame, (5, 25), (33, 55))
    source = mask_source(volume)
    index = components.build(source)

    result = plan(graph, index, frame, params=FAST)
    applied = apply_mod.apply_plan(graph, result, source, frame)

    assert all(a.voxels_added == 0 for a in applied)
    np.testing.assert_array_equal(decode(source), np.asarray(volume, np.uint8))


def test_an_orphan_fragment_joins_the_chain_and_shortens_the_claim():
    """A dropped piece of vessel is evidence, not an obstacle.

    The gap between the two stubs is far too long to close on the mask alone. With
    the orphan admitted to the chain it becomes two short gaps either side of a
    piece of real foreground -- so the same route that would have been sent to
    review is now supported, and the repair claims far fewer invented voxels than
    the straight-line distance suggests.
    """
    frame = make_frame(SHAPE)
    volume = (slit(SHAPE, 5, 1, 5, 22) | slit(SHAPE, 5, 1, 38, 55)).astype(np.uint8)
    volume[CZ - 1:CZ + 2, CY - 1:CY + 2, 24:36] = 1  # a long, thin dropped strand
    source = mask_source(volume)
    index = components.build(source)
    graph = broken_graph(frame, (5, 22), (38, 55))
    assert index.n == 3

    fragments = classify.fragment_candidates(index, frame, graph, reach_um=2000.0)
    assert [f.plausible for f in fragments] == [True]

    result = plan(graph, index, frame, params=FAST)
    accepted = result.accepted()
    assert len(accepted) == 1, result.summarise()
    assert accepted[0].kind == "fragment"
    assert [f.component for f in accepted[0].classified.fragments]
    # The whole point: the longest run of invented voxels is short, even though the
    # two graph ends are 160 um apart.
    assert accepted[0].evidence["mask_gap_voxels"] <= 4
    assert accepted[0].evidence["span_um"] > 100.0

    apply_mod.apply_plan(graph, result, source, frame)
    assert len(graph.components()) == 1
    # All three mask components become one -- two joins from one accepted repair,
    # which is exactly what admitting the fragment to the chain means.
    assert components.from_array(decode(source) > 0).n == 1


def test_paint_and_revert_are_inverse():
    volume = np.zeros((6, 8, 8), dtype=np.uint8)
    volume[2, 2, 2] = 1
    source = mask_source(volume)
    voxels = np.array([[2, 3, 3], [2, 3, 4], [3, 3, 4]], dtype=np.int64)

    added, planes = apply_mod.paint(source, voxels)
    assert added == 3 and set(planes) == {2, 3}
    assert decode(source).sum() == 4

    apply_mod.revert(source, voxels)
    np.testing.assert_array_equal(decode(source), volume)


# ============================================================ redundancy and gating
# The seventh way to be wrong, and the one the real data found: a route that is
# genuinely supported along its whole length because it spent that length inside a
# vessel the graph already describes.


def test_described_lumen_costs_more_than_undescribed_lumen():
    """The ordering the whole fix rests on.

    Undescribed foreground is pruned or never-skeletonised vessel and routing through
    it is the point. Described foreground already has a centreline, so a second one
    along it is a duplicate. Tissue is worse than both -- a route must still prefer to
    hug a vessel rather than flee into myocardium.
    """
    frame = make_frame(SHAPE)
    volume = cylinder(SHAPE, 3, 5, 55)
    source = mask_source(volume)
    index = components.build(source)
    points = np.asarray(frame.seg_to_um([[20, CY, CZ], [40, CY, CZ]]))
    box = corridor.for_candidate(frame, index, points, radius_um=30.0)

    # Centreline over the left half only: the right half is foreground nobody described.
    described_half = np.asarray(frame.seg_to_um([[x, CY, CZ] for x in range(8, 26)]))
    field = cost.build(box, index, {1}, radius_um=30.0,
                       centreline_points=described_half)

    def at(x, y=CY, z=CZ):
        """Local index of a segmentation voxel, for reading the field directly."""
        return tuple(box.to_global(
            np.asarray(frame.seg_to_um([[x, y, z]])))[0] - field.lo_zyx)

    on_centreline = float(field.cost[at(17)])
    undescribed = float(field.cost[at(45)])
    tissue = float(field.cost[at(45, CY + 12)])

    assert undescribed < on_centreline < tissue, (
        f"undescribed={undescribed:.3f} described={on_centreline:.3f} "
        f"tissue={tissue:.3f}"
    )
    assert field.described[at(17)] > 0.5
    assert field.described[at(45)] < 0.5


def test_the_penalty_is_confined_to_our_own_foreground():
    """Outside the mask the route already pays full price; charging twice would push
    it away from the vessel it is meant to be following."""
    frame = make_frame(SHAPE)
    volume = cylinder(SHAPE, 3, 5, 55)
    index = components.build(mask_source(volume))
    points = np.asarray(frame.seg_to_um([[20, CY, CZ], [40, CY, CZ]]))
    box = corridor.for_candidate(frame, index, points, radius_um=30.0)
    line = np.asarray(frame.seg_to_um([[x, CY, CZ] for x in range(8, 50)]))

    plain = cost.build(box, index, {1}, radius_um=30.0, centreline_points=line,
                       redundancy_weight=0.0)
    charged = cost.build(box, index, {1}, radius_um=30.0, centreline_points=line,
                         redundancy_weight=0.6)

    outside = ~charged.mine
    np.testing.assert_allclose(charged.cost[outside], plain.cost[outside], atol=1e-6)
    assert (charged.cost[charged.mine] > plain.cost[charged.mine]).any()


def test_a_wandering_route_is_rejected_even_when_fully_supported():
    """The gate that was applied to the proposal and never to the route.

    `tjunction.propose` gates a synthetic Hermite curve; A* then returns a different
    path and nothing re-checked it. On LADAF-2024-28 three of eleven accepted routes
    came out at tortuosity 1.86, 2.30 and 2.92 against a stated limit of 1.8.
    """
    from hipct_seg_debug.edit.reconnect.geodesic import route as route_mod

    frame = make_frame(SHAPE)
    volume = slit(SHAPE, 6, 1, 5, 28) | slit(SHAPE, 6, 1, 30, 55)
    source = mask_source(volume)
    index = components.build(source)
    graph = broken_graph(frame, (5, 28), (30, 55))

    from hipct_seg_debug.edit.reconnect import endpoints

    pair = next(b for b in endpoints.propose(graph) if b.accepted)
    associations = classify.associate(index, frame, graph)
    candidate = route_mod.Candidate(
        classified=classify.classify(pair, associations, index, frame), proposal=pair
    )
    route_mod.evaluate(candidate, index, frame, params=FAST, graph=graph)
    assert candidate.route is not None
    straight = candidate.route

    points = np.asarray([candidate.classified.source.point_um,
                         candidate.classified.target.point_um])
    box = corridor.for_candidate(frame, index, points, radius_um=20.0)
    field = cost.build(box, index,
                       {candidate.classified.source.component,
                        candidate.classified.target.component}, radius_um=20.0)
    params = GeodesicParams(alternatives=1, mask_only_gap_voxels=99)

    # As found, the route is short and passes.
    route_mod._gate(candidate, field, box, params, 20.0)
    assert candidate.status == "accept", candidate.reason
    assert candidate.evidence["route_tortuosity"] < params.tortuosity_max

    # Same endpoints, same support, three times the length: refused.
    candidate.route = astar.Route(
        path_zyx=straight.path_zyx, cost=straight.cost, support=straight.support,
        length_um=straight.length_um * 4.0,
    )
    route_mod._gate(candidate, field, box, params, 20.0)
    assert candidate.status == "reject"
    assert "wanders" in candidate.reason
    assert candidate.evidence["route_tortuosity"] > params.tortuosity_max
