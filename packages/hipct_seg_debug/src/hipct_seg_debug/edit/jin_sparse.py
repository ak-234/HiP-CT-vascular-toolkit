"""Jin et al. (PRL 2016) on lumen voxels only, for whole coronary trees.

:func:`~.jin_mcp.extract` holds about six float64 fields over the bounding box and
loops over candidates in Python, which is why it carries a two-million-voxel guard:
a LADAF-2021-17 tree box is 24 G voxels, over a terabyte at that rate. This engine
runs the *same* method on compact arrays of the lumen voxels -- a 4-byte lookup from
box index to lumen index is the only box-sized structure -- with every per-voxel loop
in numba.

**Same method, same answer.** With ``incremental=False`` the output is identical to
the dense engine's: the lumen voxels are kept in raster order, so the heap breaks
ties exactly as the dense Dijkstra does, and components are numbered by their first
voxel exactly as ``scipy.ndimage.label`` numbers them. The tests pin that equality.

``incremental=True`` (the default here) is the speed-up the methodology note asks for
-- the differential image foresting transform. Both per-round shortest-path maps are
from a seed set that only grows (the marked volume, the skeleton), so each round
continues from the last round's distances instead of starting over. Distances are
exactly those of a fresh run; where two predecessors tie, the one kept may differ,
so paths can differ between equal-cost routes.

``allow_loops`` skips the Euler-characteristic guard. The method still returns a
tree -- loops are cut, as TEASAR cuts them -- which is what the paper's simply-
connected assumption means for a mask that is not.
"""
from __future__ import annotations

import heapq

import numpy as np
from numba import njit

from .jin_mcp import LENGTHS, OFFSETS, JinSkeleton

HALF_OFFSETS = OFFSETS[:13]


# ------------------------------------------------------------------- primitives


@njit(cache=True)
def _neighbour(p, k, offsets, nz, ny, nx):
    z, y, x = p // (ny * nx), (p // nx) % ny, p % nx
    zz, yy, xx = z + offsets[k, 0], y + offsets[k, 1], x + offsets[k, 2]
    if zz < 0 or zz >= nz or yy < 0 or yy >= ny or xx < 0 or xx >= nx:
        return -1
    return (zz * ny + yy) * nx + xx


@njit(cache=True)
def _dijkstra(lookup, fg, nz, ny, nx, offsets, lengths, mu, val, seeds, initial, mode,
              cutoff, dist, prev):
    """The dense engine's three costs on compact arrays, continuing from `dist`.

    mode 0: geodesic length; 1: fuzzy distance (``len * mean mu``); 2: Eq. 6,
    ``len / (0.01 + mean(val)^2)``. `dist`/`prev` are updated in place: pass ``inf``
    and ``-1`` for a fresh run, or the previous round's for an incremental one (a
    grown seed set only lowers distances, so continuing is exact).
    """
    heap = [(0.0, np.int64(0))]
    heap.pop()
    for j in range(seeds.size):
        s = seeds[j]
        if initial[j] < dist[s]:
            dist[s] = initial[j]
            prev[s] = -1
            heapq.heappush(heap, (initial[j], np.int64(s)))
    while heap:
        d, i = heapq.heappop(heap)
        if d != dist[i]:
            continue
        if d > cutoff:
            break
        p = fg[i]
        for k in range(offsets.shape[0]):
            q = _neighbour(p, k, offsets, nz, ny, nx)
            if q < 0:
                continue
            j = lookup[q]
            if j < 0:
                continue
            step = lengths[k]
            if mode == 1:
                step *= 0.5 * (mu[i] + mu[j])
            elif mode == 2:
                step /= 0.01 + (0.5 * (val[i] + val[j])) ** 2
            new = d + step
            if new <= cutoff and new < dist[j]:
                dist[j] = new
                prev[j] = i
                heapq.heappush(heap, (new, np.int64(j)))


@njit(cache=True)
def _boundary_initial(lookup, fg, nz, ny, nx, offsets, lengths, mu):
    """FDT seed values: half the membership times the step to the nearest background."""
    out = np.full(fg.size, np.inf)
    for i in range(fg.size):
        for k in range(offsets.shape[0]):
            q = _neighbour(fg[i], k, offsets, nz, ny, nx)
            if q < 0 or lookup[q] < 0:  # outside the box counts as background
                v = 0.5 * mu[i] * lengths[k]
                if v < out[i]:
                    out[i] = v
    return out


@njit(cache=True)
def _lsf(lookup, fg, nz, ny, nx, offsets, lengths, mu, fdt):
    """Eq. 2: ``1 - max positive normalised FDT slope``, clipped to [0, 1]."""
    out = np.zeros(fg.size)
    for i in range(fg.size):
        slope = 0.0
        for k in range(offsets.shape[0]):
            q = _neighbour(fg[i], k, offsets, nz, ny, nx)
            j = -1 if q < 0 else lookup[q]
            mq = mu[j] if j >= 0 else 0.0
            fq = fdt[j] if j >= 0 else 0.0
            den = 0.5 * (mu[i] + mq) * lengths[k]
            if den > 0:
                g = (fq - fdt[i]) / den
                if g > slope:
                    slope = g
        out[i] = min(max(1.0 - slope, 0.0), 1.0)
    return out


@njit(cache=True)
def _find(parent, p):
    while parent[p] != p:
        parent[p] = parent[parent[p]]
        p = parent[p]
    return p


@njit(cache=True)
def _components(lookup, fg, nz, ny, nx, offsets, active):
    """26-connected components of the `active` lumen voxels; root = first voxel.

    Union keeps the smaller index, so every root is its component's first voxel in
    raster order -- the order ``scipy.ndimage.label`` numbers components in.
    """
    n = fg.size
    parent = np.arange(n)
    for i in range(n):
        if not active[i]:
            continue
        for k in range(offsets.shape[0]):
            q = _neighbour(fg[i], k, offsets, nz, ny, nx)
            if q < 0:
                continue
            j = lookup[q]
            if j < 0 or not active[j]:
                continue
            a, b = _find(parent, i), _find(parent, j)
            if a != b:
                if a < b:
                    parent[b] = a
                else:
                    parent[a] = b
    roots = np.full(n, -1, dtype=np.int64)
    for i in range(n):
        if active[i]:
            roots[i] = _find(parent, i)
    return roots


@njit(cache=True)
def _farthest_per_component(roots, candidates, gd):
    """For each component, its candidate of largest `gd` (first wins a tie)."""
    best = np.full(roots.size, -1, dtype=np.int64)
    for c in candidates:
        r = roots[c]
        if best[r] < 0 or gd[c] > gd[best[r]]:
            best[r] = c
    return best


@njit(cache=True)
def _trace(start, prev, in_skeleton, marked, xi):
    """The predecessor path to the skeleton, and its significance outside `marked`."""
    n = 1
    p = start
    while not in_skeleton[p]:
        p = prev[p]
        if p < 0:
            return np.empty(0, dtype=np.int64), 0.0
        n += 1
    path = np.empty(n, dtype=np.int64)
    p = start
    score = 0.0
    for k in range(n):
        path[k] = p
        if not marked[p]:
            score += xi[p]
        if k < n - 1:
            p = prev[p]
    return path, score


class _FlatValues:
    """``array.ravel()[flat box index]`` over a compact array -- the access pattern
    :func:`~.jin_mgf.section_marker` uses on the dense FDT."""

    def __init__(self, values, lookup):
        self.values, self.lookup = values, lookup

    def ravel(self):
        return self

    def __getitem__(self, flat):
        return self.values[self.lookup[np.asarray(flat)]]


# --------------------------------------------------------------------- extract


def extract(membership, *, root_zyx=None, max_iterations=100_000, medialness=None,
            marker=None, allow_loops=False, incremental=True,
            max_lumen_voxels=500_000_000, decisions_limit=200, verbose=False):
    """Jin MCP on lumen voxels; see the module docstring and :func:`~.jin_mcp.extract`.

    `medialness(mask, fg, lookup, fdt, lsf) -> xi` is the compact form of the dense hook
    (``fg`` the flat box index of each lumen voxel, ``lookup`` its inverse, the fields
    aligned with ``fg``).
    `marker(paths, fdt, mask)` is the dense hook unchanged: `paths` are flat box
    indices and `fdt` answers ``fdt.ravel()[flat]``.

    A round with more than `decisions_limit` subtrees records counts rather than every
    decision, since a whole tree has thousands per round.
    """
    import time

    raw = np.asarray(membership)
    if raw.ndim != 3 or raw.size == 0:
        raise ValueError('Jin MCP requires a nonempty 3-D membership array')
    if max_iterations < 1:
        raise ValueError('max_iterations must be positive')
    fg = np.flatnonzero(raw.ravel() > 0)
    if len(fg) > max_lumen_voxels:
        raise ValueError(f'Jin MCP input has {len(fg):,} lumen voxels; limit is '
                         f'{max_lumen_voxels:,}')
    if not len(fg):
        raise ValueError('Jin MCP requires one nonempty connected component; use --per-tree')
    mu = raw.ravel()[fg].astype(np.float64)
    if not np.isfinite(mu).all() or np.any(mu > 1):
        raise ValueError('membership must be finite in [0, 1]; convert labels explicitly')
    shape = raw.shape
    nz, ny, nx = shape
    lookup = np.full(raw.size, -1, dtype=np.int32 if len(fg) < 2**31 else np.int64)
    lookup[fg] = np.arange(len(fg))
    mask = raw > 0
    t0 = time.time()

    every = np.ones(len(fg), dtype=np.bool_)
    roots = _components(lookup, fg, nz, ny, nx, HALF_OFFSETS, every)
    if len(np.unique(roots)) != 1:
        raise ValueError('Jin MCP requires one nonempty connected component; use --per-tree')
    if not allow_loops:
        from skimage.measure import euler_number

        if euler_number(mask, connectivity=3) != 1:
            raise ValueError('Jin MCP tree-like-input check failed (Euler characteristic '
                             '!= 1); inspect tunnels/cavities, or pass allow_loops')

    initial = _boundary_initial(lookup, fg, nz, ny, nx, OFFSETS, LENGTHS, mu)
    seeds = np.flatnonzero(np.isfinite(initial)).astype(np.int64)
    fdt = np.full(len(fg), np.inf)
    scratch = np.full(len(fg), -1, dtype=np.int64)
    _dijkstra(lookup, fg, nz, ny, nx, OFFSETS, LENGTHS, mu, mu, seeds, initial[seeds], 1,
              np.inf, fdt, scratch)
    lsf = _lsf(lookup, fg, nz, ny, nx, OFFSETS, LENGTHS, mu, fdt)
    if verbose:
        print(f"    jin sparse: {len(fg):,} lumen voxels; fields in {time.time() - t0:.0f}s",
              flush=True)
    xi = lsf if medialness is None else np.asarray(medialness(mask, fg, lookup, fdt, lsf),
                                                   dtype=np.float64)
    strong = lsf > .5 if medialness is None else (lsf > .5) & (xi > .5)

    if root_zyx is None:
        root = int(np.argmax(fdt))
    else:
        coords = np.asarray(root_zyx)
        if coords.shape != (3,) or not np.equal(coords, np.floor(coords)).all():
            raise ValueError('root_zyx must contain three integer voxel indices')
        if np.any(coords < 0) or np.any(coords >= np.asarray(shape)):
            raise ValueError('root is outside the ROI')
        flat = int(np.ravel_multi_index(tuple(coords.astype(int)), shape))
        if lookup[flat] < 0:
            raise ValueError('root must be inside the segmentation')
        root = int(lookup[flat])

    in_skeleton = np.zeros(len(fg), dtype=np.bool_)
    in_skeleton[root] = True
    marked = np.zeros(len(fg), dtype=np.bool_)
    marked[root] = True
    edges: set = set()
    rounds = []
    fdt_view = _FlatValues(fdt, lookup)
    gd = np.full(len(fg), np.inf)
    gd_prev = np.full(len(fg), -1, dtype=np.int64)
    cost = np.full(len(fg), np.inf)
    previous = np.full(len(fg), -1, dtype=np.int64)
    new_marked = np.array([root], dtype=np.int64)
    new_skeleton = np.array([root], dtype=np.int64)

    for iteration in range(max_iterations):
        unmarked = ~marked
        if not unmarked.any():
            break
        comp = _components(lookup, fg, nz, ny, nx, HALF_OFFSETS, unmarked)
        n_sub = int(len(np.unique(comp[unmarked])))
        if not incremental:
            gd[:] = np.inf
            gd_prev[:] = -1
            cost[:] = np.inf
            previous[:] = -1
            new_marked = np.flatnonzero(marked).astype(np.int64)
            new_skeleton = np.flatnonzero(in_skeleton).astype(np.int64)
        _dijkstra(lookup, fg, nz, ny, nx, OFFSETS, LENGTHS, mu, lsf, new_marked,
                  np.zeros(len(new_marked)), 0, np.inf, gd, gd_prev)
        _dijkstra(lookup, fg, nz, ny, nx, OFFSETS, LENGTHS, mu, xi, new_skeleton,
                  np.zeros(len(new_skeleton)), 2, np.inf, cost, previous)

        candidates = np.flatnonzero(strong & unmarked).astype(np.int64)
        best = _farthest_per_component(comp, candidates, gd)
        endpoints = [(int(r), int(best[r])) for r in np.flatnonzero(best >= 0)]
        branches, decisions = [], []
        accepted_n = 0
        for label, endpoint in endpoints:
            path, score = _trace(endpoint, previous, in_skeleton, marked, xi)
            if not len(path):
                raise RuntimeError('Disconnected minimum-cost path')
            threshold = 3 + .5 * float(fdt[path[-1]])
            accepted = score > threshold
            accepted_n += int(accepted)
            if len(endpoints) <= decisions_limit:
                decisions.append(dict(
                    component=label, endpoint_zyx=list(map(int, np.unravel_index(fg[endpoint], shape))),
                    significance=float(score), threshold=threshold, accepted=bool(accepted)))
            if accepted:
                branches.append(path)
        row = dict(iteration=iteration + 1, subtrees=n_sub)
        row.update(branches=decisions) if len(endpoints) <= decisions_limit else \
            row.update(candidates=len(endpoints), accepted=accepted_n)
        rounds.append(row)
        if verbose:
            print(f"    round {iteration + 1}: {n_sub} subtree(s), {accepted_n} branch(es) "
                  f"accepted, {int(marked.sum()):,} marked ({time.time() - t0:.0f}s)", flush=True)
        if not branches:
            break

        added = []
        grown = []
        for path in branches:
            # Common predecessor paths can coalesce; stop on the existing tree.
            for p, q in zip(path[:-1], path[1:]):
                if in_skeleton[p]:
                    break
                edges.add((min(int(p), int(q)), max(int(p), int(q))))
                in_skeleton[p] = True
                grown.append(int(p))
            added.extend(int(v) for v in path)
        added = np.unique(np.asarray(added, dtype=np.int64))
        dist = np.full(len(fg), np.inf)
        _dijkstra(lookup, fg, nz, ny, nx, OFFSETS, LENGTHS, mu, lsf, added,
                  -2 * fdt[added], 0, 0.0, dist, scratch)
        newly = (dist <= 0) & ~marked
        if marker is not None:
            flat_paths = [fg[p] for p in branches]
            extra = np.asarray(marker(flat_paths, fdt_view, mask), dtype=bool)
            newly |= extra.ravel()[fg] & ~marked
        marked |= newly
        new_marked = np.flatnonzero(newly).astype(np.int64)
        new_skeleton = np.asarray(sorted(set(grown)), dtype=np.int64)
        if marked.all():
            break
    else:
        raise ValueError('Jin MCP iteration limit reached; no completed candidate returned')

    vertices = np.flatnonzero(in_skeleton)
    index = {int(v): i for i, v in enumerate(vertices)}
    edge_array = np.array([(index[a], index[b]) for a, b in sorted(edges)],
                          dtype=np.int64).reshape(-1, 2)
    coords = np.column_stack(np.unravel_index(fg[vertices], shape))
    return JinSkeleton(coords, edge_array, fdt[vertices], dict(
        method='jin-mcp', engine='sparse', incremental=bool(incremental),
        allow_loops=bool(allow_loops), doi='10.1016/j.patrec.2015.04.002',
        review_required=True,
        limitations=['experimental; not full-tree qualified',
                     'flattened lumens can produce medial sheets and extra branches',
                     'voxel paths are not subvoxel smoothed curves']
        + (['loops in the mask are cut: the output is a tree'] if allow_loops else []),
        root_zyx=list(map(int, np.unravel_index(fg[root], shape))),
        rounds=rounds, unmarked_voxels=int(np.count_nonzero(~marked)),
        lumen_voxels=int(len(fg)), seconds=time.time() - t0,
        radius_source='fuzzy_distance_voxels_not_perimeter',
        topology='new_tree_ids; assumes tree-like input without tunnels or cavities',
    ))
