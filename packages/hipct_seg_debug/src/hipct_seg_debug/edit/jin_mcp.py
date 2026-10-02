"""Jin et al., PRL 76 (2016), doi:10.1016/j.patrec.2015.04.002.

Experimental cubic-grid implementation of equations 1--9 and simultaneous
subtree growth. Distances are in voxels. Input is a fuzzy membership, not labels.
The padded exterior is background. This extracts a NEW tree, not an ID-preserving
refinement; FDT values are local thickness estimates, not perimeter radii.
"""
from __future__ import annotations

import heapq
from dataclasses import dataclass
from itertools import product

import numpy as np
from numba import njit
from scipy import ndimage

OFFSETS = np.array([p for p in product((-1, 0, 1), repeat=3) if any(p)], dtype=np.int64)
LENGTHS = np.linalg.norm(OFFSETS, axis=1)


@njit(cache=True)
def _distances(mu, offsets, lengths, seeds, initial, values, mode, cutoff=np.inf):
    """Implicit 26-neighbour Dijkstra; mode 0=GD, 1=FDT, 2=LSF cost."""
    nz, ny, nx = mu.shape
    m = mu.ravel()
    val = values.ravel()
    dist = np.full(m.size, np.inf)
    prev = np.full(m.size, -1, dtype=np.int64)
    heap = [(0.0, np.int64(0))]
    heap.pop()
    for j in range(len(seeds)):
        p = seeds[j]
        dist[p] = initial[j]
        heapq.heappush(heap, (initial[j], p))
    while heap:
        d, p = heapq.heappop(heap)
        if d != dist[p]:
            continue
        if d > cutoff:
            break
        z, y, x = p // (ny * nx), (p // nx) % ny, p % nx
        for k in range(26):
            zz, yy, xx = z + offsets[k, 0], y + offsets[k, 1], x + offsets[k, 2]
            if zz < 0 or zz >= nz or yy < 0 or yy >= ny or xx < 0 or xx >= nx:
                continue
            q = (zz * ny + yy) * nx + xx
            if m[q] <= 0:
                continue
            step = lengths[k]
            if mode == 1:
                step *= 0.5 * (m[p] + m[q])
            elif mode == 2:
                step /= 0.01 + (0.5 * (val[p] + val[q])) ** 2
            new = d + step
            if new <= cutoff and new < dist[q]:
                dist[q], prev[q] = new, p
                heapq.heappush(heap, (new, q))
    return dist.reshape(mu.shape), prev


def fields(membership):
    """Fuzzy shortest-distance transform and Eq. 2 local significance.

    Background centres have zero distance; foreground/background edges have
    half the foreground membership times their length. Unlike Euclidean EDT,
    this uses the same 26-neighbour metric as the quench test.
    """
    mu = np.pad(np.asarray(membership, dtype=np.float64), 1)
    initial = np.full(mu.shape, np.inf)
    centre = (slice(1, -1),) * 3
    for off, length in zip(OFFSETS, LENGTHS):
        other = tuple(slice(1 + int(d), mu.shape[a] - 1 + int(d)) for a, d in enumerate(off))
        seed = (mu[centre] > 0) & (mu[other] == 0)
        initial[centre] = np.minimum(initial[centre], np.where(seed, .5 * mu[centre] * length, np.inf))
    seeds = np.flatnonzero(np.isfinite(initial))
    fdt, _ = _distances(mu, OFFSETS, LENGTHS, seeds, initial.ravel()[seeds], mu, 1)
    fdt[mu == 0] = 0
    slope = np.zeros(mu[centre].shape)
    for off, length in zip(OFFSETS, LENGTHS):
        other = tuple(slice(1 + int(d), mu.shape[a] - 1 + int(d)) for a, d in enumerate(off))
        denominator = .5 * (mu[centre] + mu[other]) * length
        gradient = np.divide(fdt[other] - fdt[centre], denominator,
                             out=np.zeros_like(slope), where=denominator > 0)
        slope = np.maximum(slope, gradient)
    lsf = np.where(mu[centre] > 0, np.clip(1 - slope, 0, 1), 0)
    return fdt[centre].copy(), lsf


@dataclass
class JinSkeleton:
    coordinates_zyx: np.ndarray
    edges: np.ndarray
    fdt: np.ndarray
    report: dict


def extract(membership, *, root_zyx=None, max_voxels=2_000_000, max_iterations=256):
    """Extract one connected tree on an isotropic grid, with bounded ROI size.

    All candidate paths in a round share the same cost/predecessor map and
    marked-volume snapshot (section 2.5). Only traced edges are exported: adding
    all adjacencies between skeleton voxels would introduce spurious loops.
    """
    raw = np.asarray(membership)
    if raw.ndim != 3 or raw.size == 0:
        raise ValueError('Jin MCP requires a nonempty 3-D membership array')
    if raw.size > max_voxels:
        raise ValueError(f'Jin MCP ROI has {raw.size:,} voxels; limit is {max_voxels:,}. Use a smaller ROI.')
    if max_iterations < 1:
        raise ValueError('max_iterations must be positive')
    mu = np.asarray(raw, dtype=np.float64)
    if not np.isfinite(mu).all() or np.any((mu < 0) | (mu > 1)):
        raise ValueError('membership must be finite in [0, 1]; convert labels explicitly')
    mask = mu > 0
    _, count = ndimage.label(mask, structure=np.ones((3, 3, 3)))
    if count != 1:
        raise ValueError('Jin MCP requires one nonempty connected component; use --per-tree')
    from skimage.measure import euler_number
    if euler_number(mask, connectivity=3) != 1:
        raise ValueError('Jin MCP tree-like-input check failed (Euler characteristic != 1); inspect tunnels/cavities')
    fdt, lsf = fields(mu)
    if root_zyx is None:
        root = int(np.argmax(fdt))
    else:
        coords = np.asarray(root_zyx)
        if coords.shape != (3,) or not np.equal(coords, np.floor(coords)).all():
            raise ValueError('root_zyx must contain three integer voxel indices')
        if np.any(coords < 0) or np.any(coords >= np.asarray(mu.shape)):
            raise ValueError('root is outside the ROI')
        root = int(np.ravel_multi_index(tuple(coords.astype(int)), mu.shape))
        if not mask.ravel()[root]:
            raise ValueError('root must be inside the segmentation')
    skeleton = {root}
    edges = set()
    marked = np.zeros(mu.shape, dtype=bool)
    marked.ravel()[root] = True  # paper initialises O_marked with the root only
    rounds = []
    for iteration in range(max_iterations):
        labels, n = ndimage.label(mask & ~marked, structure=np.ones((3, 3, 3)))
        if n == 0:
            break
        seeds = np.flatnonzero(marked)
        gd, _ = _distances(mu, OFFSETS, LENGTHS, seeds, np.zeros(len(seeds)), lsf, 0)
        seeds = np.array(sorted(skeleton), dtype=np.int64)
        _, previous = _distances(mu, OFFSETS, LENGTHS, seeds, np.zeros(len(seeds)), lsf, 2)
        branches = []
        decisions = []
        # Group strong quench candidates once; avoid one full-volume scan per subtree.
        candidates = np.flatnonzero((lsf > .5) & ~marked)
        groups = {}
        for p in candidates:
            label = int(labels.ravel()[p])
            old = groups.get(label)
            if old is None or gd.ravel()[p] > gd.ravel()[old]:
                groups[label] = int(p)
        for label, endpoint in sorted(groups.items()):
            path = [endpoint]
            while path[-1] not in skeleton:
                q = int(previous[path[-1]])
                if q < 0:
                    raise RuntimeError('Disconnected minimum-cost path')
                path.append(q)
            score = float(sum(lsf.ravel()[p] for p in path if not marked.ravel()[p]))
            threshold = 3 + .5 * float(fdt.ravel()[path[-1]])
            accepted = score > threshold
            decisions.append(dict(component=label, endpoint_zyx=list(map(int, np.unravel_index(endpoint, mu.shape))),
                                  significance=score, threshold=threshold, accepted=accepted))
            if accepted:
                branches.append(path)
        rounds.append(dict(iteration=iteration + 1, subtrees=n, branches=decisions))
        if not branches:
            break
        added = set()
        for path in branches:
            # Common predecessor paths can coalesce; stop on the existing tree.
            for p, q in zip(path[:-1], path[1:]):
                if p in skeleton:
                    break
                edges.add((min(p, q), max(p, q)))
                skeleton.add(p)
                added.add(p)
            added.update(path)
        seeds = np.array(sorted(added), dtype=np.int64)
        dilation, _ = _distances(mu, OFFSETS, LENGTHS, seeds, -2 * fdt.ravel()[seeds], lsf, 0, 0.)
        marked |= dilation <= 0
        if np.all(marked[mask]):
            break
    else:
        raise ValueError('Jin MCP iteration limit reached; no completed candidate returned')
    vertices = np.array(sorted(skeleton), dtype=np.int64)
    lookup = {int(v): i for i, v in enumerate(vertices)}
    edge_array = np.array([(lookup[a], lookup[b]) for a, b in sorted(edges)], dtype=np.int64).reshape(-1, 2)
    coords = np.column_stack(np.unravel_index(vertices, mu.shape))
    return JinSkeleton(coords, edge_array, fdt.ravel()[vertices], dict(
        method='jin-mcp', doi='10.1016/j.patrec.2015.04.002',
        review_required=True,
        limitations=['experimental; not full-tree qualified',
                     'flattened lumens can produce medial sheets and extra branches',
                     'voxel paths are not subvoxel smoothed curves'],
        root_zyx=list(map(int, np.unravel_index(root, mu.shape))),
        rounds=rounds, unmarked_voxels=int(np.count_nonzero(mask & ~marked)),
        radius_source='fuzzy_distance_voxels_not_perimeter',
        topology='new_tree_ids; assumes tree-like input without tunnels or cavities',
    ))
