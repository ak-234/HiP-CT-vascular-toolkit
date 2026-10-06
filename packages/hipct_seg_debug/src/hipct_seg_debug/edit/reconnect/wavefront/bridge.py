"""Two ways to get across a gap with a wave.

Both return :class:`~..geodesic.astar.Route`, so every gate, the global
selector, the transactional apply and the audit files consume them unchanged.

**Dual fronts** (:func:`dual_front`), for a pair with a known far side. One front
is swept from the source along its tangent, another from the target set along
its; the route is the cheapest place they meet -- the saddle of ``U_s + U_t`` --
and the two backtracks joined there. The saddle is a natural statement of
confidence: with the winning corridor made expensive and both fronts re-run, the
next saddle says how much worse the second-best way through is.

**Keypoint chaining** (:func:`chain`), for gaps that are long, or whose far side
is unknown, or that contain a genuine dropout. The front is swept from the tip
with a look-ahead cone along the vessel axis, allowed to travel one step -- a few
major-axis lengths of the collapsed ellipse -- and the cheapest voxel on that
isodistance shell becomes the next keypoint, from which the front is restarted
with the direction the leg just took, blended with the local structure-tensor
axis. That is the keypoint construction of Benmansour & Cohen (2009) with the
direction carried between restarts as in Kaul, Yezzi & Tsai (2012). When the
front stalls -- the shell it reaches has no support -- the same map is asked a
narrower question: is there a *cluster* of planar, lumen-like voxels further
ahead inside the cone? If there is, the cheapest voxel of it becomes the next
keypoint and the leg is recorded as a forced bridge; if there is not, a bounded
number of weak keypoints are tolerated before the chain stops and says why.
"""

from __future__ import annotations

import numpy as np

from ..geodesic import astar
from . import propagate as prop

#: Cone applied to a seeded front in :func:`dual_front`: wide and mild, because it
#: biases every step of the propagation and the route may legitimately bend.
SEED_CONE_DEG = 45.0
SEED_CONE_WEIGHT = 0.5
#: Look-ahead cone for :func:`chain`: narrow and firm, because a restarted front
#: should continue the vessel and the tensor is what lets it bend.
LOOKAHEAD_CONE_DEG = 15.0
LOOKAHEAD_CONE_WEIGHT = 4.0
#: A keypoint whose leg has less than this support (calibrated units) is weak.
MIN_KEYPOINT_SUPPORT = 0.25
#: Consecutive weak keypoints tolerated before the chain stops.
MAX_WEAK_KEYPOINTS = 2
#: A forced bridge needs this many planar lumen-like voxels ahead, 26-connected.
MIN_CLUSTER_VOXELS = 3
#: ...with at least this planarity, and lumen likelihood above one half.
LOOKAHEAD_PLANARITY = 0.3
#: How far past the step the front is swept, so a probe has somewhere to look.
PROBE_FACTOR = 2.5
#: Hard cap on keypoints, so a chain cannot loop forever inside a big corridor.
MAX_KEYPOINTS = 400


def _measure(field, path_local, cost, reason, stats) -> astar.Route:
    if not len(path_local):
        return astar.Route(np.empty((0, 3), np.int64), np.inf, np.array([]), 0.0,
                           reason=reason, metrics=dict(stats))
    return astar._measure(field, np.asarray(path_local, dtype=np.int64) + field.lo_zyx,
                          cost, reason, stats)


def _unit(v) -> np.ndarray | None:
    v = np.asarray(v, dtype=np.float64).reshape(3)
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-9 and np.isfinite(n) else None


# ------------------------------------------------------------------ dual front


def _sweep(field, seeds, tangent, *, allowed, engine, stencil, seed_cone):
    cone = None
    if tangent is not None and seed_cone is not None:
        deg, weight = seed_cone
        cone = (tangent, deg, weight)
    return prop.propagate(field, seeds, stop="exhaust", cone=cone, allowed=allowed,
                          engine=engine, stencil=stencil)


def _saddle(front_a, front_b):
    """The voxel where the two fronts meet most cheaply, or ``None``."""
    total = front_a.arrival + front_b.arrival
    both = np.isfinite(total)
    if not both.any():
        return None, np.inf
    flat = int(np.argmin(np.where(both, total, np.inf)))
    return np.array(np.unravel_index(flat, total.shape), dtype=np.int64), float(
        total.ravel()[flat])


def _join(front_a, front_b, meeting) -> np.ndarray:
    a = front_a.backtrack(meeting)
    b = front_b.backtrack(meeting)
    if not len(a) or not len(b):
        return np.empty((0, 3), dtype=np.int64)
    return np.vstack([a, b[::-1][1:]]) if len(b) > 1 else a


def dual_front(field, source_zyx, goals_zyx, *, source_tangent=None,
               target_tangent=None, alternatives: int = 3,
               distinct_voxels: float = astar.DISTINCT_VOXELS,
               seed_cone=(SEED_CONE_DEG, SEED_CONE_WEIGHT), coarse: bool = True,
               engine: str = "auto", stencil: int = 26) -> list[astar.Route]:
    """Up to `alternatives` distinct routes between `source_zyx` and `goals_zyx`.

    Both are **global** indices; `goals_zyx` is ``(M, 3)`` -- a point for an
    end-to-end join, a stretch of parent for a T-junction. Tangents are unit
    directions in local ``(z, y, x)`` order, each pointing *into* the gap.

    The same contract as :func:`~..geodesic.astar.routes`: alternatives are found
    under suppression rather than blocking, so a single-corridor gap reports the
    one way through at a visibly higher cost and reads as unambiguous.
    """
    source = field.to_local(source_zyx)
    goals = np.asarray(goals_zyx, dtype=np.int64).reshape(-1, 3) - field.lo_zyx
    shape = np.asarray(field.cost.shape)
    goals = goals[np.all((goals >= 0) & (goals < shape), axis=1)]
    if not len(goals):
        return [astar.Route(np.empty((0, 3), np.int64), np.inf, np.array([]), 0.0,
                            reason="the target lies outside the corridor")]
    if np.any(source < 0) or np.any(source >= shape):
        return [astar.Route(np.empty((0, 3), np.int64), np.inf, np.array([]), 0.0,
                            reason="the start voxel is outside the corridor")]

    stats: dict = {"engine": prop.select_engine(engine)}
    allowed = None
    if coarse:
        allowed = prop.coarse_corridor(field, source[None, :], goals)
        if allowed is not None:
            stats["corridor_voxels"] = int(allowed.sum())
            if not allowed.any():
                return [astar.Route(np.empty((0, 3), np.int64), np.inf, np.array([]),
                                    0.0, reason="coarse pass: no route reaches the "
                                                "target", metrics=stats)]

    t_source = _unit(source_tangent) if source_tangent is not None else None
    t_target = _unit(target_tangent) if target_tangent is not None else None

    out: list[astar.Route] = []
    original = field.cost
    working = original.copy()
    for _ in range(max(int(alternatives), 1)):
        field.cost = working
        try:
            front_a = _sweep(field, source[None, :], t_source, allowed=allowed,
                             engine=engine, stencil=stencil, seed_cone=seed_cone)
            front_b = _sweep(field, goals, t_target, allowed=allowed,
                             engine=engine, stencil=stencil, seed_cone=seed_cone)
        finally:
            field.cost = original
        meeting, saddle = _saddle(front_a, front_b)
        leg_stats = dict(stats, expanded_source=front_a.expanded,
                         expanded_target=front_b.expanded, saddle_cost=saddle)
        if meeting is None:
            if not out:
                out.append(astar.Route(np.empty((0, 3), np.int64), np.inf, np.array([]),
                                       0.0, reason="no route reaches the target",
                                       metrics=leg_stats))
            else:
                out[-1].metrics.setdefault("suppressed_cost", float("inf"))
            break
        path = _join(front_a, front_b, meeting)
        leg_stats["meeting_zyx"] = [int(v) for v in (meeting + field.lo_zyx)]
        leg_stats["fine_expanded"] = front_a.expanded + front_b.expanded
        route = _measure(field, path, saddle, "found", leg_stats)
        if out and min(route.deviation_from(prev) for prev in out) < distinct_voxels:
            out[-1].metrics.setdefault("suppressed_cost", route.cost)
            break
        out.append(route)
        working = astar._suppress(working, path, distinct_voxels)
    return out


# --------------------------------------------------------------- keypoint chain


def _clusters(mask: np.ndarray, min_voxels: int) -> np.ndarray:
    """Keep only 26-connected clusters of at least `min_voxels`."""
    from scipy import ndimage

    if not mask.any():
        return mask
    labels, n = ndimage.label(mask, structure=np.ones((3, 3, 3), dtype=bool))
    if not n:
        return mask
    counts = np.bincount(labels.ravel())
    keep = counts >= int(min_voxels)
    keep[0] = False
    return keep[labels]


def _forward_half_space(shape, kp, tangent, spacing_zyx) -> np.ndarray:
    """Voxels on the far side of the plane through `kp` normal to `tangent`."""
    grid = np.stack(np.meshgrid(*[np.arange(s) for s in shape], indexing="ij"), axis=-1)
    rel = (grid - np.asarray(kp)) * np.asarray(spacing_zyx, dtype=np.float64)
    return (rel @ np.asarray(tangent, dtype=np.float64)) > 0.0


def _blend_tangent(field, leg_direction, kp_local, coherence_floor: float = 0.3):
    """The next leg's direction: what this leg did, pulled toward the local axis."""
    t = _unit(leg_direction)
    if t is None:
        return None
    z, y, x = (int(v) for v in kp_local)
    if float(field.coherence[z, y, x]) < coherence_floor:
        return t
    axis = np.asarray(field.axis[z, y, x], dtype=np.float64)
    if float(np.dot(axis, t)) < 0:
        axis = -axis
    blended = _unit(0.5 * t + 0.5 * axis)
    return blended if blended is not None else t


def chain(field, start_zyx, start_tangent, goals_zyx=None, *, step_um: float,
          max_length_um: float, cone_deg: float = LOOKAHEAD_CONE_DEG,
          cone_weight: float = LOOKAHEAD_CONE_WEIGHT,
          min_keypoint_support: float = MIN_KEYPOINT_SUPPORT,
          max_weak_keypoints: int = MAX_WEAK_KEYPOINTS,
          lookahead_planarity: float = LOOKAHEAD_PLANARITY,
          min_cluster_voxels: int = MIN_CLUSTER_VOXELS,
          engine: str = "auto", stencil: int = 26) -> astar.Route:
    """March a front from `start_zyx` keypoint by keypoint toward `goals_zyx`.

    `start_zyx` and `goals_zyx` are global; `start_tangent` is a unit direction in
    local ``(z, y, x)`` order pointing into the gap; `step_um` is how far each leg
    travels before the front is restarted, and `max_length_um` the total budget.
    With no goals the chain explores until the budget or the evidence runs out,
    which is how an open end is followed to whatever it reaches.

    ``reason == "found"`` means a goal was reached. Anything else is a chain that
    stopped early, returned with its partial path so an audit can show where and
    why -- ``metrics["stop_reason"]`` -- rather than an empty result.
    """
    kp = field.to_local(start_zyx)
    shape = np.asarray(field.cost.shape)
    if np.any(kp < 0) or np.any(kp >= shape):
        return astar.Route(np.empty((0, 3), np.int64), np.inf, np.array([]), 0.0,
                           reason="the start voxel is outside the corridor")
    goals = None
    if goals_zyx is not None and len(goals_zyx):
        goals = np.asarray(goals_zyx, dtype=np.int64).reshape(-1, 3) - field.lo_zyx
        goals = goals[np.all((goals >= 0) & (goals < shape), axis=1)]
        if not len(goals):
            goals = None
    tangent = _unit(start_tangent)
    if tangent is None:
        return astar.Route(np.empty((0, 3), np.int64), np.inf, np.array([]), 0.0,
                           reason="the start has no direction")

    step = float(max(step_um, 1.5 * float(np.max(field.spacing_zyx))))
    probe = PROBE_FACTOR * step

    pieces: list[np.ndarray] = [kp[None, :]]
    keypoints = [kp.copy()]
    supports: list[float] = []
    legs: list[dict] = []
    total_cost = 0.0
    total_length = 0.0
    forced = 0
    weak_run = 0
    stop_reason = "found"
    reached_zyx = None
    reached_origin = None
    engine_name = prop.select_engine(engine)

    for _n in range(MAX_KEYPOINTS):
        # Ahead of the keypoint only. The cone makes going back dear, not
        # impossible, and the lumen behind the tip is the cheapest material in the
        # corridor; without this a leg can loop home before setting out.
        forward = _forward_half_space(field.cost.shape, kp, tangent, field.spacing_zyx)
        forward[tuple(kp)] = True
        front = prop.propagate(field, kp[None, :], goals_zyx=goals, stop="first_goal",
                               max_length_um=probe, cone=(tangent, cone_deg, cone_weight),
                               allowed=forward, engine=engine, stencil=stencil)
        if front.reached:
            goal_local, origin = front.reached[0]
            leg = front.backtrack(goal_local)
            pieces.append(leg[1:])
            total_cost += float(front.arrival[tuple(goal_local)])
            total_length += float(front.length[tuple(goal_local)])
            reached_zyx = goal_local + field.lo_zyx
            reached_origin = origin
            legs.append(dict(kind="goal", length_um=float(front.length[tuple(goal_local)])))
            break

        settled = np.isfinite(front.arrival)
        shell = settled & (front.length >= step)
        if not shell.any():
            stop_reason = "the front could not travel a full step from the last keypoint"
            break
        flat = int(np.argmin(np.where(shell, front.arrival, np.inf)))
        candidate = np.array(np.unravel_index(flat, front.shape), dtype=np.int64)
        leg = front.backtrack(candidate)
        leg_support = float(np.mean(field.support[leg[:, 0], leg[:, 1], leg[:, 2]])) \
            if len(leg) else 0.0
        kind = "step"

        if leg_support < min_keypoint_support:
            # The look-ahead: planar, lumen-like voxels further out in the cone.
            ahead = (settled & (front.length >= step) & (front.length <= probe)
                     & (field.planarity >= lookahead_planarity) & (field.lumen >= 0.5))
            ahead = _clusters(ahead, min_cluster_voxels)
            if ahead.any():
                flat = int(np.argmin(np.where(ahead, front.arrival, np.inf)))
                candidate = np.array(np.unravel_index(flat, front.shape), dtype=np.int64)
                leg = front.backtrack(candidate)
                leg_support = float(np.mean(
                    field.support[leg[:, 0], leg[:, 1], leg[:, 2]]))
                kind = "forced"
                forced += 1
                weak_run = 0
            else:
                weak_run += 1
                kind = "weak"
                if weak_run > max_weak_keypoints:
                    stop_reason = ("no planar lumen signature within the look-ahead "
                                   f"cone after {weak_run - 1} weak keypoint(s)")
                    break
        else:
            weak_run = 0

        if len(leg) < 2:
            stop_reason = "the next keypoint coincides with the last"
            break
        leg_length = float(front.length[tuple(candidate)])
        total_cost += float(front.arrival[tuple(candidate)])
        total_length += leg_length
        pieces.append(leg[1:])
        keypoints.append(candidate.copy())
        supports.append(leg_support)
        legs.append(dict(kind=kind, length_um=leg_length, support=leg_support))
        if total_length > max_length_um:
            stop_reason = (f"the chain exceeded its {max_length_um:.0f} um budget "
                           f"without reaching a target")
            break
        direction = (candidate - kp).astype(np.float64) * field.spacing_zyx
        blended = _blend_tangent(field, direction, candidate)
        tangent = tangent if blended is None else blended
        kp = candidate
    else:
        stop_reason = f"the chain exceeded {MAX_KEYPOINTS} keypoints"

    path = np.vstack(pieces)
    stats = dict(
        engine=engine_name, keypoints_zyx=[[int(v) for v in k + field.lo_zyx]
                                           for k in keypoints],
        keypoint_support=supports, forced_bridges=forced, legs=legs,
        stop_reason=stop_reason, chain_length_um=total_length,
        fine_expanded=sum(1 for _ in legs),
    )
    if reached_zyx is not None:
        stats["reached_zyx"] = [int(v) for v in reached_zyx]
        stats["reached_origin"] = int(reached_origin)
    reason = "found" if stop_reason == "found" else stop_reason
    return _measure(field, path, total_cost, reason, stats)
