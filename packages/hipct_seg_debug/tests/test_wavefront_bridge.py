"""Dual fronts and keypoint chains on gaps whose answer is known."""

from __future__ import annotations

import numpy as np
import pytest

from hipct_seg_debug.edit.reconnect.geodesic import components, corridor, cost
from hipct_seg_debug.edit.reconnect.wavefront import bridge, prepare, tensor

from .conftest_geodesic import (
    CY,
    CZ,
    WALL,
    FakeStack,
    broken_graph,
    cylinder,
    make_frame,
    mask_source,
    ribbon,
    ribbon_gap,
    slit,
)

SHAPE = (40, 40, 60)
LONG = (40, 40, 120)


def _scene(mask, image, shape, left, right, *, radius_um=20.0, allowed=None):
    frame = make_frame(shape)
    graph = broken_graph(frame, left, right)
    index = components.build(mask_source(mask))
    profiles = prepare.profile_ends(graph, frame, index)
    source, target = profiles[1], profiles[2]
    box = corridor.for_candidate(
        frame, index, np.vstack([source.point_um, target.point_um]), radius_um=radius_um,
        stack=None if image is None else FakeStack(image), pad_factor=4.0,
    )
    allowed = {source.component, target.component} if allowed is None else allowed
    scalar = cost.build(box, index, allowed, radius_um=radius_um,
                        calibration_points=np.vstack([source.tail_points_um,
                                                      target.tail_points_um]))
    field = tensor.build(scalar, box, radius_um=radius_um, profiles=[source, target])
    start = box.to_global(source.point_um[None, :])[0]
    goal = box.to_global(target.point_um[None, :])
    return field, box, source, target, start, goal


def _zyx(tangent_xyz):
    return np.asarray(tangent_xyz, dtype=np.float64)[::-1]


@pytest.mark.parametrize("name", ["tube", "ribbon", "slit"])
def test_dual_front_meets_in_a_two_voxel_gap(name):
    builder = {
        "tube": lambda a, b: cylinder(SHAPE, 3, a, b),
        "ribbon": lambda a, b: ribbon(SHAPE, 6, 2, a, b),
        "slit": lambda a, b: slit(SHAPE, 6, 0, a, b),
    }[name]
    mask = builder(5, 28) | builder(30, 55)
    field, box, source, target, start, goal = _scene(mask, None, SHAPE, (5, 28), (30, 55))
    routes = bridge.dual_front(field, start, goal, source_tangent=_zyx(source.tangent),
                               target_tangent=_zyx(target.tangent), alternatives=2,
                               engine="lattice")
    best = routes[0]
    assert best.reason == "found"
    assert tuple(best.path_zyx[0]) == tuple(start)
    assert tuple(best.path_zyx[-1]) == tuple(goal[0])
    assert 27 <= best.metrics["meeting_zyx"][2] <= 31  # met inside the gap
    assert np.isfinite(best.metrics["saddle_cost"])
    assert np.all(best.path_zyx[:, 0] == CZ) and np.all(best.path_zyx[:, 1] == CY)
    # The second search returns the same corridor dearer, or a distinct one.
    assert len(routes) == 1 and "suppressed_cost" in best.metrics or len(routes) == 2


def test_dual_front_crosses_a_long_pinched_slit():
    mask, image = ribbon_gap(LONG, 6, 2, 5, 115, gap=(30, 90), faint=0.5)
    field, box, source, target, start, goal = _scene(mask, image, LONG, (5, 30), (90, 115))
    routes = bridge.dual_front(field, start, goal, source_tangent=_zyx(source.tangent),
                               target_tangent=_zyx(target.tangent), alternatives=2,
                               engine="lattice")
    best = routes[0]
    assert best.reason == "found"
    assert best.length_um > 590.0
    assert best.unsupported_um(field.spacing_zyx, 0.25) < 100.0
    assert np.all(np.abs(best.path_zyx[:, 0] - CZ) <= 1)  # stays in the slit plane
    assert best.metrics.get("corridor_voxels", 1) > 0


def test_dual_front_refuses_a_target_outside_the_corridor():
    mask = ribbon(SHAPE, 6, 2, 5, 28) | ribbon(SHAPE, 6, 2, 30, 55)
    field, box, source, target, start, goal = _scene(mask, None, SHAPE, (5, 28), (30, 55))
    routes = bridge.dual_front(field, start, np.array([[0, 0, 500]]), engine="lattice")
    assert not len(routes[0].path_zyx) and "outside" in routes[0].reason


def test_dual_front_never_enters_a_blocked_neighbour():
    mask = ribbon(SHAPE, 6, 2, 5, 28) | ribbon(SHAPE, 6, 2, 30, 55)
    rival = ribbon(SHAPE, 6, 2, 5, 55, cz=CZ + 8)  # a parallel vessel just above
    field, box, source, target, start, goal = _scene(mask | rival, None, SHAPE,
                                                     (5, 28), (30, 55), allowed=None)
    routes = bridge.dual_front(field, start, goal, engine="lattice")
    best = routes[0]
    assert best.reason == "found"
    local = best.path_zyx - field.lo_zyx
    assert not field.blocked[local[:, 0], local[:, 1], local[:, 2]].any()
    assert field.blocked.any()


def test_chain_reaches_a_goal_across_a_long_gap():
    mask, image = ribbon_gap(LONG, 6, 2, 5, 115, gap=(30, 90), faint=0.5)
    field, box, source, target, start, goal = _scene(mask, image, LONG, (5, 30), (90, 115))
    route = bridge.chain(field, start, _zyx(source.tangent), goal,
                         step_um=2.0 * source.major_um, max_length_um=1500.0,
                         engine="lattice")
    assert route.reason == "found"
    assert tuple(route.path_zyx[-1]) == tuple(goal[0])
    assert len(route.metrics["keypoints_zyx"]) >= 2
    assert route.metrics["stop_reason"] == "found"
    assert route.metrics["forced_bridges"] == 0
    assert route.length_um > 590.0
    xs = route.path_zyx[:, 2]
    assert np.all(np.diff(xs) >= 0)  # never turns back


def test_chain_bridges_a_dropout_and_records_it():
    mask, image = ribbon_gap(LONG, 6, 2, 5, 115, gap=(30, 90), faint=0.5)
    rng = np.random.default_rng(1)
    image[:, :, 55:63] = WALL + rng.normal(0.0, 4.0, (40, 40, 8))  # nothing to see
    field, box, source, target, start, goal = _scene(mask, image, LONG, (5, 30), (90, 115))
    route = bridge.chain(field, start, _zyx(source.tangent), goal,
                         step_um=2.0 * source.major_um, max_length_um=1500.0,
                         engine="lattice")
    assert route.reason == "found"
    kinds = [leg["kind"] for leg in route.metrics["legs"]]
    assert "forced" in kinds or "weak" in kinds
    assert route.unsupported_um(field.spacing_zyx, 0.25) >= 60.0


def test_chain_explores_without_a_goal_and_stops_on_its_budget():
    mask, image = ribbon_gap(LONG, 6, 2, 5, 115, gap=(30, 90), faint=0.5)
    field, box, source, target, start, goal = _scene(mask, image, LONG, (5, 30), (90, 115))
    route = bridge.chain(field, start, _zyx(source.tangent), None,
                         step_um=2.0 * source.major_um, max_length_um=300.0,
                         engine="lattice")
    assert route.reason != "found"
    assert "budget" in route.metrics["stop_reason"]
    assert 300.0 < route.metrics["chain_length_um"] < 500.0
    assert route.path_zyx[-1][2] > start[2] + 20


def test_chain_stops_when_the_trace_ends():
    mask, image = ribbon_gap(LONG, 6, 2, 5, 115, gap=(30, 90), faint=0.5)
    image[:, :, 60:] = WALL  # the faint trace ends half way and never resumes
    mask[:, :, 90:] = 0  # and there is no far side either
    frame = make_frame(LONG)
    graph = broken_graph(frame, (5, 30), (90, 115))
    index = components.build(mask_source(mask))
    profiles = prepare.profile_ends(graph, frame, index)
    source = profiles[1]
    box = corridor.for_candidate(frame, index, np.vstack([source.point_um,
                                                          source.point_um + [900, 0, 0]]),
                                 radius_um=20.0, stack=FakeStack(image), pad_factor=4.0)
    scalar = cost.build(box, index, {source.component}, radius_um=20.0,
                        calibration_points=source.tail_points_um)
    field = tensor.build(scalar, box, radius_um=20.0, profiles=[source])
    start = box.to_global(source.point_um[None, :])[0]
    route = bridge.chain(field, start, _zyx(source.tangent), None,
                         step_um=2.0 * source.major_um, max_length_um=2000.0,
                         engine="lattice")
    assert route.reason != "found"
    assert ("planar lumen signature" in route.metrics["stop_reason"]
            or "full step" in route.metrics["stop_reason"])
    assert route.path_zyx[-1][2] < 75 + field.lo_zyx[2]


def test_chain_refuses_a_start_with_no_direction():
    mask = ribbon(SHAPE, 6, 2, 5, 28) | ribbon(SHAPE, 6, 2, 30, 55)
    field, box, source, target, start, goal = _scene(mask, None, SHAPE, (5, 28), (30, 55))
    route = bridge.chain(field, start, np.zeros(3), goal, step_um=50.0,
                        max_length_um=500.0, engine="lattice")
    assert not len(route.path_zyx) and "direction" in route.reason
