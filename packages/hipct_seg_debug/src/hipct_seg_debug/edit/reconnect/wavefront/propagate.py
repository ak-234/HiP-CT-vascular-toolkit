"""Wave propagation over a tensor-priced corridor.

A front is swept outward from one or more seeds until it has covered the region
of interest, reached a goal, or travelled a given physical length. What comes back
is not one route but a *map*: the minimum accumulated cost to every voxel, the
path length it took to get there, which seed got there first, and the parent
pointers that turn any voxel into a route by walking back. The bridging strategies
in :mod:`.bridge` are all questions asked of that map -- where two fronts meet,
which voxel on an isodistance shell is cheapest to reach -- and none of them could
be asked of a point-to-point search.

The state is a voxel. :mod:`..geodesic.astar` carries the arrival direction so it
can price turning; here turning is priced by the metric instead -- a step through
the collapse normal is dear whichever way the route was going -- and the state
space is 27 times smaller for it, which is what makes sweeping a corridor forty
radii long affordable.

Two engines:

**lattice** (default, always available). A Dijkstra over the 26- or 98-neighbour
stencil with Riemannian edge costs ``0.5 * (c_v ||e||_{A_v} + c_w ||e||_{A_w}) * |e|``,
compiled with numba. This is the standard graph approximation to the anisotropic
eikonal equation; it is exact on the lattice and over-estimates the continuous
distance by up to ~13% on 26 neighbours (~5% on 98), which is why
:data:`~.tensor.MAX_RATIO` caps the anisotropy at what the stencil can resolve.

**agd** (optional). The HamiltonFastMarching solver behind the ``agd`` package
solves the Riemannian eikonal equation properly, with adaptive stencils and
sub-voxel geodesics. It is used when importable and asked for; nothing here
imports it unconditionally, and a missing install is reported, not worked around.
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass, field
from math import gcd

import numpy as np
from numba import njit

STOP_EXHAUST, STOP_FIRST_GOAL, STOP_ALL_GOALS = 0, 1, 2
_STOP = {"exhaust": STOP_EXHAUST, "first_goal": STOP_FIRST_GOAL,
         "all_goals": STOP_ALL_GOALS}

ENGINES = ("auto", "lattice", "agd")


def offsets(stencil: int = 26) -> np.ndarray:
    """The neighbour offsets of a stencil, ``(K, 3)`` in ``(dz, dy, dx)``."""
    if stencil == 26:
        reach = 1
    elif stencil == 98:
        reach = 2
    else:
        raise ValueError("stencil must be 26 or 98")
    out = []
    for dz in range(-reach, reach + 1):
        for dy in range(-reach, reach + 1):
            for dx in range(-reach, reach + 1):
                if (dz, dy, dx) == (0, 0, 0):
                    continue
                if gcd(gcd(abs(dz), abs(dy)), abs(dx)) != 1:
                    continue  # a multiple of a shorter offset
                out.append((dz, dy, dx))
    return np.array(out, dtype=np.int64)


@dataclass
class Front:
    """The map one propagation produced, on the corridor's local grid."""

    arrival: np.ndarray  # (dz, dy, dx) float64, inf where never reached
    length: np.ndarray  # (dz, dy, dx) float64 um of path from the seed
    parent: np.ndarray  # (dz, dy, dx) int64 flat *local* index, -1 at seeds / unreached
    origin: np.ndarray  # (dz, dy, dx) int16 seed label, -1 unreached
    reached: list = field(default_factory=list)  # [(local zyx, origin)] goals settled
    expanded: int = 0
    engine: str = "lattice"
    metrics: dict = field(default_factory=dict)

    @property
    def shape(self) -> tuple[int, int, int]:
        return tuple(self.arrival.shape)

    def settled(self) -> np.ndarray:
        return np.isfinite(self.arrival)

    def backtrack(self, zyx) -> np.ndarray:
        """The route from the seed to `zyx`, as ``(N, 3)`` local indices."""
        shape = np.asarray(self.shape, dtype=np.int64)
        strides = np.array([shape[1] * shape[2], shape[2], 1], dtype=np.int64)
        flat = int((np.asarray(zyx, dtype=np.int64) * strides).sum())
        if not np.isfinite(self.arrival.ravel()[flat]):
            return np.empty((0, 3), dtype=np.int64)
        chain = [flat]
        parent = self.parent.ravel()
        while parent[chain[-1]] >= 0:
            chain.append(int(parent[chain[-1]]))
            if len(chain) > self.arrival.size:
                raise RuntimeError("parent pointers form a cycle")
        chain.reverse()
        return np.stack(np.unravel_index(np.array(chain, dtype=np.int64), self.shape),
                        axis=1).astype(np.int64)


# ------------------------------------------------------------- lattice engine


@njit(cache=True)
def _dijkstra(cost, aniso, off_flat, unit, step, seeds, seed_labels, goal,
             stop_mode, max_length, cone_axis, cone_cos, cone_weight, use_cone,
             arrival, length, parent, origin):
    n_goals = 0
    for i in range(goal.shape[0]):
        if goal[i]:
            n_goals += 1
    n_reached = 0
    reached = np.empty(max(n_goals, 1), dtype=np.int64)
    heap = [(0.0, 0)]
    heap.pop()
    for s in range(seeds.shape[0]):
        v = seeds[s]
        if not np.isfinite(cost[v]):
            continue
        if arrival[v] > 0.0:
            arrival[v] = 0.0
            length[v] = 0.0
            parent[v] = -1
            origin[v] = seed_labels[s]
            heapq.heappush(heap, (0.0, v))
    expanded = 0
    k_dirs = off_flat.shape[0]
    while len(heap) > 0:
        gv, v = heapq.heappop(heap)
        if gv > arrival[v] + 1e-12:
            continue
        if goal[v]:
            goal[v] = False
            reached[n_reached] = v
            n_reached += 1
            if stop_mode == 1 or (stop_mode == 2 and n_reached == n_goals):
                break
        if length[v] >= max_length:
            continue
        expanded += 1
        cv = cost[v]
        av0 = aniso[v, 0]
        av1 = aniso[v, 1]
        av2 = aniso[v, 2]
        av3 = aniso[v, 3]
        av4 = aniso[v, 4]
        av5 = aniso[v, 5]
        for k in range(k_dirs):
            w = v + off_flat[k]
            cw = cost[w]
            if not np.isfinite(cw):
                continue
            e0 = unit[k, 0]
            e1 = unit[k, 1]
            e2 = unit[k, 2]
            qv = (av0 * e0 * e0 + av3 * e1 * e1 + av5 * e2 * e2
                  + 2.0 * (av1 * e0 * e1 + av2 * e0 * e2 + av4 * e1 * e2))
            qw = (aniso[w, 0] * e0 * e0 + aniso[w, 3] * e1 * e1
                  + aniso[w, 5] * e2 * e2
                  + 2.0 * (aniso[w, 1] * e0 * e1 + aniso[w, 2] * e0 * e2
                           + aniso[w, 4] * e1 * e2))
            if qv < 0.0:
                qv = 0.0
            if qw < 0.0:
                qw = 0.0
            edge = 0.5 * (cv * np.sqrt(qv) + cw * np.sqrt(qw)) * step[k]
            if use_cone:
                along = e0 * cone_axis[0] + e1 * cone_axis[1] + e2 * cone_axis[2]
                if along < cone_cos:
                    edge *= 1.0 + cone_weight * (cone_cos - along)
            candidate = gv + edge
            if candidate + 1e-12 < arrival[w]:
                arrival[w] = candidate
                length[w] = length[v] + step[k]
                parent[w] = v
                origin[w] = origin[v]
                heapq.heappush(heap, (candidate, w))
    return expanded, reached[:n_reached]


def _pad_field(field, allowed, pad: int):
    """Cost padded with ``inf`` and ``A`` padded with ``I`` so every neighbour
    index is valid and the bounds test disappears from the inner loop."""
    cost = np.pad(np.asarray(field.cost, dtype=np.float64), pad, constant_values=np.inf)
    if allowed is not None:
        keep = np.pad(np.asarray(allowed, dtype=bool), pad, constant_values=False)
        cost = np.where(keep, cost, np.inf)
    aniso = np.pad(np.asarray(field.aniso, dtype=np.float64),
                   ((pad, pad), (pad, pad), (pad, pad), (0, 0)), constant_values=0.0)
    aniso[..., 0] = np.where(aniso[..., 0] == 0.0, 1.0, aniso[..., 0])
    aniso[..., 3] = np.where(aniso[..., 3] == 0.0, 1.0, aniso[..., 3])
    aniso[..., 5] = np.where(aniso[..., 5] == 0.0, 1.0, aniso[..., 5])
    return cost, aniso


def _propagate_lattice(field, seeds, seed_labels, goals, stop_mode, max_length_um,
                       cone, allowed, stencil) -> Front:
    shape = np.asarray(field.cost.shape, dtype=np.int64)
    off = offsets(stencil)
    pad = int(np.max(np.abs(off)))
    padded_shape = shape + 2 * pad
    spacing = np.asarray(field.spacing_zyx, dtype=np.float64)

    cost, aniso = _pad_field(field, allowed, pad)
    flat_cost = cost.ravel()
    flat_aniso = np.ascontiguousarray(aniso.reshape(-1, 6))
    strides = np.array([padded_shape[1] * padded_shape[2], padded_shape[2], 1],
                       dtype=np.int64)

    off_flat = (off * strides).sum(axis=1)
    physical = off * spacing
    step = np.linalg.norm(physical, axis=1)
    unit = physical / step[:, None]

    n = int(np.prod(padded_shape))
    arrival = np.full(n, np.inf, dtype=np.float64)
    length = np.full(n, np.inf, dtype=np.float64)
    parent = np.full(n, -1, dtype=np.int64)
    origin = np.full(n, -1, dtype=np.int16)

    seeds = np.asarray(seeds, dtype=np.int64).reshape(-1, 3)
    inside = np.all((seeds >= 0) & (seeds < shape), axis=1)
    seeds_flat = ((seeds[inside] + pad) * strides).sum(axis=1)
    labels = (np.zeros(len(seeds), dtype=np.int16) if seed_labels is None
              else np.asarray(seed_labels, dtype=np.int16).reshape(-1))[inside]

    goal_flags = np.zeros(n, dtype=np.bool_)
    if goals is not None and len(goals):
        g = np.asarray(goals, dtype=np.int64).reshape(-1, 3)
        g = g[np.all((g >= 0) & (g < shape), axis=1)]
        if len(g):
            goal_flags[((g + pad) * strides).sum(axis=1)] = True

    if cone is None:
        axis, cone_cos, cone_weight, use_cone = np.zeros(3), -1.0, 0.0, False
    else:
        axis_in, deg, weight = cone
        axis = np.asarray(axis_in, dtype=np.float64).reshape(3)
        axis = axis / max(float(np.linalg.norm(axis)), 1e-12)
        cone_cos = float(np.cos(np.radians(float(deg))))
        cone_weight, use_cone = float(weight), True

    expanded, reached = _dijkstra(
        flat_cost, flat_aniso, off_flat, unit, step, seeds_flat, labels, goal_flags,
        int(stop_mode), float(max_length_um), axis, cone_cos, cone_weight, use_cone,
        arrival, length, parent, origin,
    )

    # Strip the padding, remapping parents onto the unpadded grid.
    def unpad(flat, dtype):
        core = flat.reshape(tuple(padded_shape))[pad:-pad, pad:-pad, pad:-pad]
        return core.astype(dtype, copy=True)

    parent_zyx = np.stack(np.unravel_index(np.where(parent >= 0, parent, 0),
                                           tuple(padded_shape)), axis=1) - pad
    local_strides = np.array([shape[1] * shape[2], shape[2], 1], dtype=np.int64)
    parent_local = np.where(parent >= 0, (parent_zyx * local_strides).sum(axis=1), -1)

    front = Front(
        arrival=unpad(arrival, np.float64), length=unpad(length, np.float64),
        parent=unpad(parent_local, np.int64), origin=unpad(origin, np.int16),
        expanded=int(expanded), engine="lattice",
    )
    for flat in reached:
        zyx = np.array(np.unravel_index(int(flat), tuple(padded_shape))) - pad
        front.reached.append((zyx.astype(np.int64), int(origin[int(flat)])))
    return front


# ------------------------------------------------------------------ agd engine


def agd_available() -> bool:
    try:
        import agd  # noqa: F401
        from agd import Eikonal  # noqa: F401
    except Exception:  # noqa: BLE001 - any import failure means "not available"
        return False
    return True


def _propagate_agd(field, seeds, seed_labels, goals, stop_mode, max_length_um,
                   cone, allowed, stencil) -> Front:
    """The HamiltonFastMarching Riemann3 model, mapped onto :class:`Front`.

    The solver returns arrival values only, so parents are reconstructed by
    steepest descent over the 26-stencil and lengths by walking that tree in
    arrival order. A cone is applied as a hard domain restriction, which is the
    closest Riemann3 can get to the asymmetric penalty the lattice engine uses.
    """
    from agd import Eikonal
    from agd.Metrics import Riemann

    shape = tuple(int(v) for v in field.cost.shape)
    spacing = np.asarray(field.spacing_zyx, dtype=np.float64)
    if not np.allclose(spacing, spacing[0]):
        raise ValueError("the agd engine needs isotropic voxels")
    h = float(spacing[0])

    cost = np.asarray(field.cost, dtype=np.float64).copy()
    if allowed is not None:
        cost = np.where(np.asarray(allowed, dtype=bool), cost, np.inf)
    seeds = np.asarray(seeds, dtype=np.int64).reshape(-1, 3)
    if cone is not None:
        axis_in, deg, _weight = cone
        axis = np.asarray(axis_in, dtype=np.float64)
        axis /= max(float(np.linalg.norm(axis)), 1e-12)
        grid = np.stack(np.meshgrid(*[np.arange(s) for s in shape], indexing="ij"), -1)
        rel = (grid - seeds[0]).astype(np.float64) * spacing
        norm = np.linalg.norm(rel, axis=-1)
        along = (rel @ axis) / np.maximum(norm, 1e-9)
        outside = (norm > 1.5 * h) & (along < np.cos(np.radians(float(deg))))
        cost = np.where(outside, np.inf, cost)
    finite = np.isfinite(cost)
    big = float(np.nanmax(np.where(finite, cost, np.nan))) if finite.any() else 1.0
    cost_solver = np.where(finite, cost, 1e6 * max(big, 1.0))

    mats = field.matrices().astype(np.float64)  # (dz, dy, dx, 3, 3)
    metric = np.moveaxis(mats, (-2, -1), (0, 1)) * cost_solver[None, None] ** 2

    hfm = Eikonal.dictIn({
        "model": "Riemann3",
        "exportValues": 1,
        "seeds": seeds.astype(np.float64) * h,
        "verbosity": 0,
    })
    hfm.SetRect(sides=[[0.0, (s - 1) * h] for s in shape], dims=np.array(shape))
    hfm["metric"] = Riemann(metric)
    if max_length_um is not None and np.isfinite(max_length_um):
        hfm["stopAtDistance"] = float(max_length_um) * float(np.min(cost_solver))
    out = hfm.Run()
    values = np.asarray(out["values"], dtype=np.float64)
    values = np.where(finite, values, np.inf)
    values[np.isnan(values)] = np.inf

    # Parents by steepest descent; lengths by walking the tree in arrival order.
    return _front_from_values(field, values, seeds, seed_labels, goals, stop_mode,
                              max_length_um, engine="agd")


def _front_from_values(field, values, seeds, seed_labels, goals, stop_mode,
                       max_length_um, *, engine: str) -> Front:
    shape = np.asarray(values.shape, dtype=np.int64)
    padded = np.pad(values, 1, constant_values=np.inf)
    off = offsets(26)
    spacing = np.asarray(field.spacing_zyx, dtype=np.float64)
    step = np.linalg.norm(off * spacing, axis=1)
    core = padded[1:-1, 1:-1, 1:-1]
    best = np.full(core.shape, -np.inf)
    best_k = np.full(core.shape, -1, dtype=np.int64)
    for k, (dz, dy, dx) in enumerate(off):
        neighbour = padded[1 + dz:1 + dz + shape[0], 1 + dy:1 + dy + shape[1],
                           1 + dx:1 + dx + shape[2]]
        # Descent rate per unit length, so a long diagonal is not preferred merely
        # for spanning more of the drop.
        rate = (core - neighbour) / step[k]
        better = np.isfinite(rate) & (rate > 0) & (rate > best)
        best = np.where(better, rate, best)
        best_k = np.where(better, k, best_k)

    strides = np.array([shape[1] * shape[2], shape[2], 1], dtype=np.int64)
    grid = np.stack(np.meshgrid(*[np.arange(s) for s in shape], indexing="ij"), -1)
    has = best_k >= 0
    pz = grid + off[np.where(has, best_k, 0)]
    parent = np.where(has & np.isfinite(core), (pz * strides).sum(axis=-1), -1)
    seeds = np.asarray(seeds, dtype=np.int64).reshape(-1, 3)
    for s in seeds:
        if np.all(s >= 0) and np.all(s < shape):
            parent[tuple(s)] = -1

    order = np.argsort(values, axis=None)
    flat_parent = parent.ravel()
    flat_values = values.ravel()
    length = np.full(values.size, np.inf)
    origin = np.full(values.size, -1, dtype=np.int16)
    labels = (np.zeros(len(seeds), dtype=np.int16) if seed_labels is None
              else np.asarray(seed_labels, dtype=np.int16))
    for s, lab in zip(seeds, labels):
        if np.all(s >= 0) and np.all(s < shape):
            f = int((s * strides).sum())
            length[f] = 0.0
            origin[f] = lab
    step_of = {tuple(o): st for o, st in zip(map(tuple, off), step)}
    for f in order:
        if not np.isfinite(flat_values[f]):
            break
        p = flat_parent[f]
        if p < 0:
            continue
        if not np.isfinite(length[p]):
            continue
        d = np.array(np.unravel_index(f, tuple(shape))) - np.array(
            np.unravel_index(p, tuple(shape)))
        length[f] = length[p] + step_of.get(tuple(d), float(np.linalg.norm(d * spacing)))
        origin[f] = origin[p]
    length = np.where(np.isfinite(values), length.reshape(tuple(shape)), np.inf)
    front = Front(arrival=values, length=length,
                  parent=parent, origin=origin.reshape(tuple(shape)),
                  expanded=int(np.isfinite(values).sum()), engine=engine)
    if goals is not None and len(goals):
        g = np.asarray(goals, dtype=np.int64).reshape(-1, 3)
        g = g[np.all((g >= 0) & (g < shape), axis=1)]
        settled = [(zyx, float(values[tuple(zyx)])) for zyx in g
                   if np.isfinite(values[tuple(zyx)])]
        settled.sort(key=lambda item: item[1])
        if stop_mode == STOP_FIRST_GOAL:
            settled = settled[:1]
        front.reached = [(zyx, int(front.origin[tuple(zyx)])) for zyx, _ in settled]
    return front


# ------------------------------------------------------------------- frontend


def select_engine(engine: str = "auto") -> str:
    if engine not in ENGINES:
        raise ValueError(f"unknown engine {engine!r}; choose from {ENGINES}")
    if engine == "auto":
        return "agd" if agd_available() else "lattice"
    if engine == "agd" and not agd_available():
        raise RuntimeError(
            "the agd engine was requested but the `agd` package is not installed; "
            "install it (pip install agd) or use --engine lattice"
        )
    return engine


def propagate(field, seeds_zyx, *, seed_labels=None, goals_zyx=None,
              stop: str = "exhaust", max_length_um: float = np.inf, cone=None,
              allowed=None, engine: str = "auto", stencil: int = 26) -> Front:
    """Sweep a front from `seeds_zyx` (local indices) over `field`.

    `stop` is ``"exhaust"`` (cover everything reachable within `max_length_um`),
    ``"first_goal"`` or ``"all_goals"``. `cone` is ``(axis_zyx, degrees, weight)``:
    a soft, asymmetric penalty on steps outside the cone around `axis`, which is
    how a restarted front is told which way the vessel was going. `allowed` is a
    boolean mask restricting the domain, used by the coarse-to-fine pass.
    """
    if stop not in _STOP:
        raise ValueError(f"unknown stop rule {stop!r}")
    chosen = select_engine(engine)
    seeds = np.asarray(seeds_zyx, dtype=np.int64).reshape(-1, 3)
    if not len(seeds):
        raise ValueError("at least one seed is required")
    if chosen == "agd":
        return _propagate_agd(field, seeds, seed_labels, goals_zyx, _STOP[stop],
                              max_length_um, cone, allowed, stencil)
    return _propagate_lattice(field, seeds, seed_labels, goals_zyx, _STOP[stop],
                              max_length_um, cone, allowed, stencil)


# ----------------------------------------------------------- coarse to fine


def coarse_corridor(field, seeds_zyx, goals_zyx, *, factor: int = 2, halo: int = 3,
                    threshold: int = 20_000) -> np.ndarray | None:
    """A tube of fine voxels around the coarse scalar route, or ``None``.

    The same trade :func:`~..geodesic.astar.solve` makes: the coarse pass answers
    *which corridor* on a block-minimum of the scalar cost, isotropically, and the
    fine anisotropic pass is then confined to a tube around that answer. ``None``
    means the corridor is small enough to sweep whole, or that no coarse route
    exists (in which case the caller should not bother with a fine pass either,
    and can tell the two apart by the corridor size).
    """
    from ..geodesic import astar
    from . import tensor as tensor_mod

    if field.cost.size <= threshold or min(field.cost.shape) < 2 * factor:
        return None
    coarse_cost = astar._downsample(np.asarray(field.cost, dtype=np.float64), factor)

    class _Scalar:
        cost = coarse_cost
        spacing_zyx = np.asarray(field.spacing_zyx, dtype=np.float64) * factor

    coarse = tensor_mod.isotropic(_Scalar())
    seeds = np.asarray(seeds_zyx, dtype=np.int64).reshape(-1, 3) // factor
    goals = np.unique(np.asarray(goals_zyx, dtype=np.int64).reshape(-1, 3) // factor,
                      axis=0)
    front = propagate(coarse, seeds, goals_zyx=goals, stop="first_goal",
                      engine="lattice")
    if not front.reached:
        return np.zeros(field.cost.shape, dtype=bool)
    path = front.backtrack(front.reached[0][0])
    allowed = astar._corridor_mask(field.cost.shape, path, factor, halo)
    shape = np.asarray(field.cost.shape)
    for point in np.vstack([np.asarray(seeds_zyx, dtype=np.int64).reshape(-1, 3),
                            np.asarray(goals_zyx, dtype=np.int64).reshape(-1, 3)]):
        lo = np.maximum(point - halo, 0)
        hi = np.minimum(point + halo + 1, shape)
        allowed[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]] = True
    return allowed
