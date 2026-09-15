"""A global, direction-aware path search, replacing the greedy DPC walk.

:func:`~..dpc.walk` picks the best next voxel and never reconsiders. That is fast
and it is why it fails on exactly the cases this package was built for: a greedy
walk cannot pay a few voxels of poor evidence now to reach a strongly supported
continuation just beyond, so a vessel with a genuine dropout in the middle of the
gap is abandoned partway with "no neighbour satisfies the direction constraint".
It also has no way to say how *confident* it is, because it never saw the second
best route.

A* over the same grid fixes both. It is optimal for the cost field it is given, so
a local trough costs what it actually costs rather than ending the search; and
re-running it with the winner penalised yields genuinely distinct alternatives,
which is the only honest basis for calling a route ambiguous.

Two things make it more than a shortest path:

**Direction is part of the state.** A search over voxels alone has no memory of
where it was going, so curvature cannot be priced and the cheapest route through a
Y-junction is free to arrive backwards. Here a state is ``(voxel, arrival
direction)`` and a turn costs ``turn_weight * (1 - cos)``, which is what keeps a
route continuing the vessel rather than merely reaching the target.

**Coarse to fine.** The full-resolution search over a corridor big enough to
contain a wandering vessel is tens of millions of states. Solving on a halved grid
first and then re-solving at full resolution inside a narrow tube around that
answer costs a small fraction of it, and the refinement is what recovers the
voxel-accurate route the coarse pass could only approximate.
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass, field

import numpy as np

#: Cost of a right-angle turn, relative to one voxel of fully unsupported travel.
TURN_WEIGHT = 0.6
#: Corridor half-width for the fine pass, in coarse voxels.
REFINE_HALO = 3
#: A candidate alternative must differ from every accepted one by at least this
#: many voxels somewhere along its length, or it is the same route re-expressed.
DISTINCT_VOXELS = 3.0
#: Multiplier on the cost of voxels near an already-returned route, when looking
#: for the next alternative. Large enough to push the search elsewhere, finite so
#: that a genuinely single-corridor gap still returns its one route twice rather
#: than failing.
SUPPRESSION = 6.0

_OFFSETS = np.array(
    [(dz, dy, dx)
     for dz in (-1, 0, 1) for dy in (-1, 0, 1) for dx in (-1, 0, 1)
     if (dz, dy, dx) != (0, 0, 0)],
    dtype=np.int64,
)
_N_DIR = len(_OFFSETS)  # 26


@dataclass
class Route:
    """One path through the cost field, with the evidence it collected."""

    path_zyx: np.ndarray  # (N, 3) global segmentation indices
    cost: float
    support: np.ndarray  # (N,) calibrated support at each step
    length_um: float
    turn_cost: float = 0.0
    reason: str = "found"
    expanded: int = 0
    metrics: dict = field(default_factory=dict)

    @property
    def mean_support(self) -> float:
        return float(np.mean(self.support)) if len(self.support) else 0.0

    @property
    def min_support(self) -> float:
        return float(np.min(self.support)) if len(self.support) else 0.0

    def path_um(self, frame) -> np.ndarray:
        """World micrometres, via the segmentation frame."""
        return np.asarray(frame.seg_to_um(self.path_zyx[:, ::-1]), dtype=np.float64)

    def unsupported_um(self, spacing_zyx, fraction: float) -> float:
        """Longest *contiguous* stretch the image gives no support for.

        Contiguous rather than total, because the two are different findings. A
        route that is weak in scattered single voxels is a route through a noisy
        but real vessel; a route with one unbroken unsupported run is a route that
        left the vessel and came back, and only the second is disqualifying.
        """
        if len(self.path_zyx) < 2:
            return 0.0
        weak = self.support < fraction
        steps = np.linalg.norm(
            np.diff(self.path_zyx, axis=0) * np.asarray(spacing_zyx), axis=1
        )
        # A step counts as unsupported when the voxel it arrives at is.
        best = run = 0.0
        for k, step in enumerate(steps):
            if weak[k + 1]:
                run += float(step)
                best = max(best, run)
            else:
                run = 0.0
        return best

    def deviation_from(self, other: "Route") -> float:
        """Largest distance, in voxels, from this route to the nearest point of another."""
        from scipy.spatial import cKDTree

        if not len(other.path_zyx) or not len(self.path_zyx):
            return float("inf")
        distance, _ = cKDTree(other.path_zyx.astype(np.float64)).query(
            self.path_zyx.astype(np.float64), k=1
        )
        return float(np.max(distance))


def _direction_index(step: np.ndarray) -> int:
    """Which of the 26 unit offsets this step is. Steps are always neighbours."""
    matches = np.flatnonzero(np.all(_OFFSETS == step, axis=1))
    return int(matches[0]) if len(matches) else _N_DIR


def _nearest_direction(vector) -> int:
    """The offset best aligned with an arbitrary direction, for a seeded start."""
    v = np.asarray(vector, dtype=np.float64)
    n = float(np.linalg.norm(v))
    if n < 1e-9:
        return _N_DIR
    unit = _OFFSETS / np.linalg.norm(_OFFSETS, axis=1)[:, None]
    return int(np.argmax(unit @ (v / n)))


def search(cost: np.ndarray, start, goals, *, spacing_zyx, start_direction=None,
           turn_weight: float = TURN_WEIGHT, base_cost: float = 0.05,
           allowed: np.ndarray | None = None, orientation: bool = True,
           max_expansions: int = 4_000_000
           ) -> tuple[np.ndarray | None, float, int, str]:
    """A* from `start` to the nearest of `goals`, over a per-voxel cost field.

    `cost` may contain ``inf`` for blocked material. `goals` is an ``(M, 3)`` set
    of local indices -- a single point for an end-to-end join, a whole local
    stretch of the parent vessel for a T-junction, which is what makes the two
    cases one code path.

    With `orientation`, a state is ``(voxel, arrival direction)`` and turning
    costs ``turn_weight * (1 - cos)``. Without it, a state is a voxel and the
    search is 27 times smaller -- which is the right trade for the coarse pass in
    :func:`solve`, whose job is to find *which corridor*, not to shape the route
    inside it.

    Three implementation choices carry the runtime, and all three matter because
    an alternatives search degenerates towards Dijkstra by construction -- it has
    been told the good route is expensive, so it has to look everywhere else:

    * the grid is padded with ``inf``, so the 26 neighbours of any voxel are
      always valid indices and the per-expansion bounds test disappears;
    * states are flat integers into preallocated arrays rather than tuples into
      dicts, which is most of the constant factor;
    * each expansion relaxes all 26 neighbours in one vectorised pass, so the
      Python-level work per expansion is a handful of numpy calls instead of a
      26-iteration loop.

    Returns ``(path, cost, expansions, reason)``; `path` is ``None`` when no route
    exists, and the reason says which of the several ways that happens it was.
    """
    shape = np.asarray(cost.shape, dtype=np.int64)
    spacing = np.asarray(spacing_zyx, dtype=np.float64)
    start = np.asarray(start, dtype=np.int64)
    goal_set = np.asarray(goals, dtype=np.int64).reshape(-1, 3)

    if np.any(start < 0) or np.any(start >= shape):
        return None, np.inf, 0, "the start voxel is outside the corridor"
    if not np.isfinite(cost[tuple(start)]):
        return None, np.inf, 0, "the start voxel is inside an unrelated component"
    inside = np.all((goal_set >= 0) & (goal_set < shape), axis=1)
    goal_set = goal_set[inside]
    if not len(goal_set):
        return None, np.inf, 0, "the target lies outside the corridor"

    goal_flags = np.zeros(tuple(shape), dtype=bool)
    goal_flags[goal_set[:, 0], goal_set[:, 1], goal_set[:, 2]] = True
    # Heuristic distance to the *nearest* goal. One EDT over the goal set is exact
    # and costs one pass, where a per-state min over M goals costs M per expansion.
    heuristic = (_goal_distance(goal_flags, spacing) * base_cost).astype(np.float64)

    # -- pad, so every neighbour index is in range ----------------------------
    padded = np.pad(np.asarray(cost, dtype=np.float64), 1, constant_values=np.inf)
    if allowed is not None:
        reachable = np.pad(np.asarray(allowed, dtype=bool), 1, constant_values=False)
        padded = np.where(reachable, padded, np.inf)
    flat_cost = padded.ravel()
    flat_goal = np.pad(goal_flags, 1, constant_values=False).ravel()
    flat_h = np.pad(heuristic, 1, constant_values=0.0).ravel()
    strides = np.array([padded.shape[1] * padded.shape[2], padded.shape[2], 1],
                       dtype=np.int64)
    neighbour_offset = (_OFFSETS * strides).sum(axis=1)

    step_length = np.linalg.norm(_OFFSETS * spacing, axis=1)
    unit = (_OFFSETS * spacing) / step_length[:, None]
    # cos between every pair of arrival directions, precomputed: this is the whole
    # curvature term, and it becomes one row lookup per expansion.
    turn = turn_weight * (1.0 - unit @ unit.T) if orientation else None

    n_dir = _N_DIR + 1 if orientation else 1
    n_states = flat_cost.size * n_dir
    g = np.full(n_states, np.inf, dtype=np.float64)
    came = np.full(n_states, -1, dtype=np.int64)

    start_flat = int(((start + 1) * strides).sum())
    start_dir = (_N_DIR if start_direction is None
                 else _nearest_direction(start_direction)) if orientation else 0
    origin = start_flat * n_dir + start_dir
    g[origin] = 0.0
    queue = [(float(flat_h[start_flat]), 0.0, origin)]
    expanded = 0
    reached = -1
    directions = np.arange(_N_DIR, dtype=np.int64)

    while queue:
        _priority, gv, state = heapq.heappop(queue)
        if gv > g[state] + 1e-12:
            continue
        voxel, d = divmod(state, n_dir)
        if flat_goal[voxel]:
            reached = state
            break
        expanded += 1
        if expanded > max_expansions:
            return None, np.inf, expanded, "the search space was exhausted"

        targets = voxel + neighbour_offset
        there = flat_cost[targets]
        ok = np.isfinite(there)
        if not ok.any():
            continue
        edge = 0.5 * (flat_cost[voxel] + there[ok]) * step_length[ok]
        if orientation and d != _N_DIR:
            edge = edge + turn[d][ok]
        candidate = gv + edge
        keys = (targets[ok] * n_dir
                + (directions[ok] if orientation else 0))
        better = candidate + 1e-12 < g[keys]
        if not better.any():
            continue
        keys, candidate = keys[better], candidate[better]
        g[keys] = candidate
        came[keys] = state
        priority = candidate + flat_h[targets[ok][better]]
        for f, gg, key in zip(priority.tolist(), candidate.tolist(), keys.tolist()):
            heapq.heappush(queue, (f, gg, key))

    if reached < 0:
        return None, np.inf, expanded, "no route reaches the target"

    chain = [reached]
    while came[chain[-1]] >= 0:
        chain.append(int(came[chain[-1]]))
    chain.reverse()
    voxels = np.array([s // n_dir for s in chain], dtype=np.int64)
    path = np.stack(np.unravel_index(voxels, padded.shape), axis=1) - 1
    return path.astype(np.int64), float(g[reached]), expanded, "found"


def _goal_distance(goal_flags: np.ndarray, spacing_zyx) -> np.ndarray:
    """Euclidean distance from every voxel to the nearest goal, in micrometres."""
    from scipy import ndimage

    return ndimage.distance_transform_edt(~goal_flags, sampling=spacing_zyx)


# ------------------------------------------------------------ coarse then fine


def _downsample(cost: np.ndarray, factor: int) -> np.ndarray:
    """Block-minimum. Minimum, not mean: the coarse pass must not lose a narrow
    but genuine passage between two blocks of blocked material, and a mean would
    fill it in with the walls on either side."""
    f = int(factor)
    pad = [(0, (-s) % f) for s in cost.shape]
    padded = np.pad(cost, pad, mode="edge")
    shape = [s // f for s in padded.shape]
    return padded.reshape(shape[0], f, shape[1], f, shape[2], f).min(axis=(1, 3, 5))


def solve(field, start_zyx, goal_zyx, *, start_direction=None,
          turn_weight: float = TURN_WEIGHT, coarse_factor: int = 2,
          halo: int = REFINE_HALO, coarse_threshold: int = 20_000
          ) -> tuple[np.ndarray | None, float, str, dict]:
    """One route from `start_zyx` to `goal_zyx`, both **global** indices.

    Solves on a halved grid first, then re-solves at full resolution inside a tube
    around that answer. The tube is what bounds the cost; the halo is what keeps
    the refinement free to differ from the coarse route rather than merely tracing
    it.

    **The coarse pass is deliberately not orientation-aware.** Its question is
    which corridor, and a coarse voxel is two fine voxels across -- wide enough
    that a turn measured on it says little about the curvature of the route that
    will finally be traced through it. Dropping direction from the state makes
    that pass 27 times smaller, and the fine pass, which is orientation-aware,
    then does the shaping inside a tube small enough to afford it.
    """
    start = field.to_local(start_zyx)
    goals = np.asarray(goal_zyx, dtype=np.int64).reshape(-1, 3) - field.lo_zyx
    stats: dict = {}

    allowed = None
    if field.cost.size > coarse_threshold and min(field.cost.shape) >= 2 * coarse_factor:
        coarse = _downsample(field.cost, coarse_factor)
        coarse_path, _c, expanded, reason = search(
            coarse, start // coarse_factor, np.unique(goals // coarse_factor, axis=0),
            spacing_zyx=field.spacing_zyx * coarse_factor, orientation=False,
        )
        stats["coarse_expanded"] = expanded
        stats["coarse_reason"] = reason
        if coarse_path is None:
            return None, np.inf, f"coarse pass: {reason}", stats
        allowed = _corridor_mask(field.cost.shape, coarse_path, coarse_factor, halo)
        # The endpoints must be in the corridor whatever the rounding did.
        for point in np.vstack([start[None, :], goals]):
            lo = np.maximum(point - halo, 0)
            hi = np.minimum(point + halo + 1, np.asarray(field.cost.shape))
            allowed[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]] = True
        stats["corridor_voxels"] = int(allowed.sum())

    path, cost, expanded, reason = search(
        field.cost, start, goals, spacing_zyx=field.spacing_zyx,
        start_direction=start_direction, turn_weight=turn_weight, allowed=allowed,
    )
    stats["fine_expanded"] = expanded
    if path is None:
        return None, np.inf, reason, stats
    return path + field.lo_zyx, cost, reason, stats


def _corridor_mask(shape, coarse_path, factor: int, halo: int) -> np.ndarray:
    """A tube of full-resolution voxels around a coarse route."""
    mask = np.zeros(shape, dtype=bool)
    reach = halo + factor
    for voxel in coarse_path:
        centre = voxel * factor
        lo = np.maximum(centre - reach, 0)
        hi = np.minimum(centre + reach + 1, np.asarray(shape))
        mask[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]] = True
    return mask


# ----------------------------------------------------------------- the frontend


def routes(field, start_zyx, goal_zyx, *, start_direction=None,
           alternatives: int = 3, turn_weight: float = TURN_WEIGHT,
           waypoints=None, distinct_voxels: float = DISTINCT_VOXELS) -> list[Route]:
    """Up to `alternatives` spatially distinct routes, best first.

    Alternatives exist to answer one question: *is there a second, comparably good
    way through?* If there is, the route is ambiguous however good the winner
    looks in isolation, and no amount of examining the winner alone reveals that.

    They are found by re-searching with a tube around each accepted route made
    expensive -- not blocked. Blocking would make a single-corridor gap report "no
    alternative" when the truth is "only one way through", and those are different
    findings; suppression lets the second search return the same corridor at a
    visibly higher cost, which reads correctly as unambiguous.

    `waypoints`, when given, are ordered global indices the route must pass
    through, and the search runs leg by leg. That makes an operator's correction
    deterministic: the same waypoints always produce the same route.
    """
    out: list[Route] = []
    working = field.cost.copy()
    original = field.cost

    for _ in range(max(int(alternatives), 1)):
        field.cost = working
        try:
            if waypoints is not None and len(waypoints):
                path, cost, reason, stats = _solve_legs(
                    field, start_zyx, goal_zyx, waypoints,
                    start_direction=start_direction, turn_weight=turn_weight,
                )
            else:
                path, cost, reason, stats = solve(
                    field, start_zyx, goal_zyx, start_direction=start_direction,
                    turn_weight=turn_weight,
                )
        finally:
            field.cost = original

        if path is None:
            if not out:
                out.append(Route(np.empty((0, 3), np.int64), np.inf, np.array([]),
                                 0.0, reason=reason, metrics=stats))
            else:
                # No second route exists even with the first one made expensive.
                # That is "the only way through", which makes the first route
                # unambiguous -- a materially different finding from "we did not
                # look", and the two must not both read as a missing margin.
                out[-1].metrics.setdefault("suppressed_cost", float("inf"))
            break

        route = _measure(field, path, cost, reason, stats)
        if out and min(route.deviation_from(prev) for prev in out) < distinct_voxels:
            # Same corridor. Its cost under suppression is still worth recording,
            # because "the only way through" is the answer that makes the first
            # route unambiguous.
            out[-1].metrics.setdefault("suppressed_cost", route.cost)
            break
        out.append(route)
        working = _suppress(working, path - field.lo_zyx, distinct_voxels)

    return out


def _solve_legs(field, start_zyx, goal_zyx, waypoints, *, start_direction, turn_weight):
    """Search start -> w1 -> ... -> goal, carrying the arrival direction across."""
    legs = [np.asarray(start_zyx, dtype=np.int64)]
    legs.extend(np.asarray(w, dtype=np.int64).reshape(3) for w in waypoints)
    pieces: list[np.ndarray] = []
    total = 0.0
    stats: dict = {"legs": len(legs)}
    direction = start_direction

    for k in range(len(legs)):
        target = goal_zyx if k == len(legs) - 1 else legs[k + 1][None, :]
        path, cost, reason, leg_stats = solve(
            field, legs[k], target, start_direction=direction, turn_weight=turn_weight
        )
        if path is None:
            return None, np.inf, f"leg {k + 1}: {reason}", stats
        total += cost
        stats[f"leg{k + 1}_expanded"] = leg_stats.get("fine_expanded", 0)
        pieces.append(path if not pieces else path[1:])
        if len(path) >= 2:
            direction = (path[-1] - path[-2]).astype(np.float64) * field.spacing_zyx
    return np.vstack(pieces), total, "found", stats


def _measure(field, path, cost, reason, stats) -> Route:
    """Attach the support series and the geometry to a raw path."""
    local = path - field.lo_zyx
    support = field.support[local[:, 0], local[:, 1], local[:, 2]].astype(np.float64)
    steps = np.linalg.norm(np.diff(path, axis=0) * field.spacing_zyx, axis=1) \
        if len(path) > 1 else np.array([])
    return Route(
        path_zyx=path, cost=float(cost), support=support,
        length_um=float(steps.sum()), reason=reason, metrics=dict(stats),
        expanded=int(stats.get("fine_expanded", 0)),
    )


def _suppress(cost: np.ndarray, local_path: np.ndarray, radius: float) -> np.ndarray:
    """Multiply the cost of a tube around one route, to look for a different one."""
    out = cost.copy()
    r = int(np.ceil(radius))
    shape = np.asarray(cost.shape)
    for voxel in local_path:
        lo = np.maximum(voxel - r, 0)
        hi = np.minimum(voxel + r + 1, shape)
        block = out[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]
        finite = np.isfinite(block)
        block[finite] *= SUPPRESSION
    return out
