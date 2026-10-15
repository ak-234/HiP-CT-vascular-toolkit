"""Wave-front ("vessel scooping") skeletonisation from geodesic shells.

The fourth family in Walsh et al. 2024 (Rodriguez; Wu; MOST/Vaa3D), and the one this
toolkit previously had no Python route to. A wave is grown from a root through the
lumen: the geodesic distance from the root is binned into shells of fixed width, each
shell is split into its 26-connected pieces ("clusters"), and the centroid of every
cluster is a centreline vertex. Clusters in consecutive shells that touch are joined.

What that buys on ex-vivo coronaries is the reason it is here. Far from the root a
shell is a slab across the vessel, so its centroid is the **area centroid of the
cross-section** -- the middle of a collapsed slit, where a distance-transform ridge is
flat and undefined. And the cluster graph keeps topology the wave meets: a shell that
splits is a bifurcation, two clusters that merge into one is a **loop closing**, so an
anastomosis survives instead of being cut to make a tree.

Its known weaknesses are kept visible rather than tuned away:

* the shell is a geodesic sphere about the root, so in a bend or at a junction it is
  oblique to the vessel and its centroid lags towards the inside;
* a boundary bump can split a shell for a shell or two and create a small cycle. Cycles
  shorter than `min_loop_radii` local radii are contracted to one vertex, since no lumen
  ring around tissue can be that short;
* side bumps, a flat face's corners and the wave running backwards from a root placed
  mid-lumen all create short chains that never leave the parent's cross-section; a
  terminal chain whose tip does not protrude past the junction cluster's extent by
  `min_branch_radii` of its own radii is pruned here, and the backend then applies
  ``skeleton_optimise.prune_spurs`` (which spares ends on the lattice edge and spurs
  nearly as thick as their parent), as every graph here gets;
* the default root is an inlet: the centre of the widest ROI cut face, else the end
  farthest from the thickest point. A wave seeded mid-vessel grows as a sphere and,
  in a flattened lumen, cuts the slit into several arcs at once.

Radii are the maximum EDT within each cluster -- inscribed-ball placeholders, to be
remeasured by ``radius-perimeter`` like every other backend's.
"""
from __future__ import annotations

import heapq
from dataclasses import dataclass

import numpy as np
from numba import njit
from scipy import ndimage

from .jin_mcp import OFFSETS

# The 13 offsets lexicographically before (0, 0, 0): each neighbour pair once.
HALF_OFFSETS = OFFSETS[:13]


@dataclass
class ScoopSkeleton:
    coordinates_zyx: np.ndarray  # (N, 3) voxel, float (centroids)
    edges: np.ndarray  # (M, 2)
    radii_um: np.ndarray  # (N,)
    report: dict


@njit(cache=True)
def _find(parent, p):
    while parent[p] != p:
        parent[p] = parent[parent[p]]
        p = parent[p]
    return p


# Everything below runs on *foreground* voxels only. A whole coronary tree's bounding
# box is billions of voxels of which well under 1% are lumen, so per-voxel float64
# fields over the box -- what the first version allocated, ~70 bytes a voxel -- do not
# fit even in a terabyte. `fg` holds the flat box index of each lumen voxel, and
# `lookup` (int32, 4 bytes a box voxel) maps a box index back to its place in `fg`.


@njit(cache=True)
def _neighbour(p, k, offsets, nz, ny, nx):
    z, y, x = p // (ny * nx), (p // nx) % ny, p % nx
    zz, yy, xx = z + offsets[k, 0], y + offsets[k, 1], x + offsets[k, 2]
    if zz < 0 or zz >= nz or yy < 0 or yy >= ny or xx < 0 or xx >= nx:
        return -1
    return (zz * ny + yy) * nx + xx


@njit(cache=True)
def _compact_geodesic(lookup, fg, nz, ny, nx, offsets, lengths, seeds):
    """26-neighbour Dijkstra over foreground voxels, from compact `seeds`."""
    dist = np.full(fg.size, np.inf)
    heap = [(0.0, np.int64(0))]
    heap.pop()
    for j in range(seeds.size):
        dist[seeds[j]] = 0.0
        heapq.heappush(heap, (0.0, np.int64(seeds[j])))
    while heap:
        d, i = heapq.heappop(heap)
        if d != dist[i]:
            continue
        p = fg[i]
        for k in range(offsets.shape[0]):
            q = _neighbour(p, k, offsets, nz, ny, nx)
            if q < 0:
                continue
            j = lookup[q]
            if j < 0:
                continue
            new = d + lengths[k]
            if new < dist[j]:
                dist[j] = new
                heapq.heappush(heap, (new, np.int64(j)))
    return dist


@njit(cache=True)
def _compact_clusters(lookup, fg, shell, nz, ny, nx, offsets):
    """Union-find of equal-shell neighbours, plus touching pairs across adjacent shells.

    `offsets` is half the 26-neighbourhood so each pair is visited once. Returns each
    voxel's root (a compact index; -1 where unreached) and the touching root pairs.
    """
    n = fg.size
    parent = np.arange(n)
    for i in range(n):
        if shell[i] < 0:
            continue
        for k in range(offsets.shape[0]):
            q = _neighbour(fg[i], k, offsets, nz, ny, nx)
            if q < 0 or lookup[q] < 0:
                continue
            j = lookup[q]
            if shell[j] == shell[i]:
                a, b = _find(parent, i), _find(parent, j)
                if a != b:
                    if a < b:
                        parent[b] = a
                    else:
                        parent[a] = b
    m = 0
    for i in range(n):
        if shell[i] < 0:
            continue
        for k in range(offsets.shape[0]):
            q = _neighbour(fg[i], k, offsets, nz, ny, nx)
            if q >= 0 and lookup[q] >= 0 and shell[lookup[q]] >= 0 \
                    and abs(shell[lookup[q]] - shell[i]) == 1:
                m += 1
    pairs = np.empty((m, 2), dtype=np.int64)
    c = 0
    for i in range(n):
        if shell[i] < 0:
            continue
        for k in range(offsets.shape[0]):
            q = _neighbour(fg[i], k, offsets, nz, ny, nx)
            if q >= 0 and lookup[q] >= 0 and shell[lookup[q]] >= 0 \
                    and abs(shell[lookup[q]] - shell[i]) == 1:
                pairs[c, 0] = _find(parent, i)
                pairs[c, 1] = _find(parent, lookup[q])
                c += 1
    roots = np.full(n, -1, dtype=np.int64)
    for i in range(n):
        if shell[i] >= 0:
            roots[i] = _find(parent, i)
    return roots, pairs


#: Threads for the `edt` transform. It is memory-bound: on a whole LADAF-2024-28 tree
#: 8 threads took 14.2 s, 36 took 19.6 s and 72 took 22.0 s.
EDT_THREADS = 8


def _edt_at(mask, coords, sp):
    """Distance to background (um) at `coords`, the volume edge counting as background.

    One whole-box transform with the ``edt`` package (a kimimaro dependency), its
    ``black_border`` making the box face background -- which is what a crop of one tree
    means. That costs 4 bytes a box voxel, and it replaced a slab-wise
    ``scipy.ndimage.distance_transform_edt`` that was 93% of this module's run time:
    482 of 519 s on a whole LADAF-2024-28 tree, and nearly all of the 6.5 h the scoop
    took on LADAF-2021-17. The two agree to 3e-4 um. SciPy remains the fallback.
    """
    try:
        import edt
    except ImportError:  # pragma: no cover - kimimaro brings it in
        return _edt_at_scipy(mask, coords, sp)
    dist = edt.edt(np.ascontiguousarray(mask), anisotropy=tuple(float(v) for v in sp),
                   black_border=True, parallel=EDT_THREADS)
    return dist[coords[:, 0], coords[:, 1], coords[:, 2]].astype(np.float64)


def _edt_at_scipy(mask, coords, sp, *, slab=192, max_radius_um=3000.0):
    """:func:`_edt_at` without ``edt``: slab-wise SciPy, each slab padded."""
    nz = mask.shape[0]
    halo = int(np.ceil(max_radius_um / float(sp[0]))) + 2
    out = np.zeros(len(coords), dtype=np.float64)
    order = np.argsort(coords[:, 0], kind="stable")
    zs = coords[order, 0]
    for z0 in range(0, nz, slab):
        z1 = min(z0 + slab, nz)
        first, last = np.searchsorted(zs, z0), np.searchsorted(zs, z1)
        if last <= first:
            continue
        lo, hi = max(z0 - halo, 0), min(z1 + halo, nz)
        pz = (1 if lo == 0 else 0, 1 if hi == nz else 0)
        block = np.pad(mask[lo:hi], (pz, (1, 1), (1, 1)))
        dist = ndimage.distance_transform_edt(block, sampling=sp)
        sel = order[first:last]
        c = coords[sel]
        out[sel] = dist[c[:, 0] - lo + pz[0], c[:, 1] + 1, c[:, 2] + 1]
    return out


def _roots(mask, fg, lookup, edt_fg, root_zyx, sp, lengths):
    """One seed per component, as box indices.

    `root_zyx` for its own component. Elsewhere the centre of the widest cut face
    (:func:`_face_root`), and for a component that touches no face the voxel
    geodesically farthest from its maximum-EDT voxel -- an end, which is how TEASAR
    chooses its root too.
    """
    try:  # cc3d: 1.9 s against SciPy's 12.6 s on a whole LADAF-2024-28 tree
        import cc3d

        lab, n = cc3d.connected_components(mask, connectivity=26, return_N=True)
    except ImportError:  # pragma: no cover - kimimaro brings it in
        lab, n = ndimage.label(mask, structure=np.ones((3, 3, 3)))
    if n == 0:
        return []
    lab_fg = lab.ravel()[fg]
    dz, dy, dx = mask.shape
    seeds = []
    for i in range(1, n + 1):
        members = np.flatnonzero(lab_fg == i)
        face = _face_root(lab, i, sp)
        if face is not None:
            seeds.append(face)
            continue
        centre = members[int(np.argmax(edt_fg[members]))]
        d = _compact_geodesic(lookup, fg, dz, dy, dx, OFFSETS, lengths,
                              np.array([centre], dtype=np.int64))
        d[~np.isfinite(d)] = -1
        seeds.append(tuple(int(v) for v in np.unravel_index(int(fg[np.argmax(d)]), mask.shape)))
    if root_zyx is not None:
        r = tuple(int(v) for v in root_zyx)
        if len(r) != 3 or any(not 0 <= r[k] < mask.shape[k] for k in range(3)):
            raise ValueError("root_zyx is outside the volume")
        if not mask[r]:
            raise ValueError("root_zyx must be inside the segmentation")
        seeds[lab[r] - 1] = r
    return seeds


def _face_root(lab, i, sp):
    """Centre of component `i`'s largest cut face, or ``None`` if it touches no face.

    A wave started mid-vessel grows as a sphere, and in a flattened lumen that sphere
    cuts the slit into several arcs at once -- a false junction at the root, measured
    as a four-way node on LADAF-2024-28. Seeding at an inlet avoids it, and an ROI's
    widest cut face is where a vessel enters it. Only the six box faces are read; the
    in-plane EDT of each picks the centre of that section.
    """
    best, where = 0.0, None
    for axis in range(3):
        for index in (0, lab.shape[axis] - 1):
            face = np.take(lab, index, axis=axis) == i
            if not face.any():
                continue
            other = [sp[k] for k in range(3) if k != axis]
            d = ndimage.distance_transform_edt(np.pad(face, 1), sampling=other)[1:-1, 1:-1]
            k = int(np.argmax(d))
            if d.flat[k] > best:
                best = float(d.flat[k])
                ij = list(np.unravel_index(k, face.shape))
                ij.insert(axis, index)
                where = tuple(int(v) for v in ij)
    return where


def _chain_length(nodes, xyz):
    return float(sum(np.linalg.norm(xyz[a] - xyz[b]) for a, b in zip(nodes[:-1], nodes[1:])))


def _contract_small_cycles(g, xyz, count, rmax, min_loop_radii):
    """Merge every cycle shorter than `min_loop_radii` local radii into one vertex."""
    import networkx as nx

    contracted = 0
    while True:
        merged_any = False
        used: set = set()
        for cyc in nx.cycle_basis(g):
            if used.intersection(cyc):
                continue
            length = _chain_length(cyc + [cyc[0]], xyz)
            r = max(float(rmax[c]) for c in cyc)
            if length >= min_loop_radii * max(r, 1e-9):
                continue
            keep = cyc[0]
            w = np.array([count[c] for c in cyc], dtype=np.float64)
            xyz[keep] = (xyz[cyc] * w[:, None]).sum(0) / w.sum()
            count[keep] = w.sum()
            rmax[keep] = r
            for c in cyc[1:]:
                for nb in list(g.neighbors(c)):
                    if nb != keep:
                        g.add_edge(keep, nb)
                g.remove_node(c)
            used.update(cyc)
            contracted += 1
            merged_any = True
        if not merged_any:
            return contracted


def _prune_spurs(g, xyz, rmax, cov, protected, min_branch_radii):
    """Remove terminal chains whose tip does not protrude from the parent lumen.

    A chain is a spur when its tip lies within the junction cluster's own extent in
    the tip's direction (``2 sqrt(u' C u)``, the half-width of a uniform section) plus
    `min_branch_radii` times the chain's inscribed radius. Measured in the *tip's
    direction* because the alternatives both failed: the inscribed radius alone is a
    slit's half-thickness, so a slit's fork into its two tips at a flat face survived;
    the cluster's largest extent alone pruned real side branches of a flattened
    vessel on LADAF-2024-28, which leave across its narrow direction.
    """
    pruned = 0
    changed = True
    while changed:
        changed = False
        for leaf in [n for n in g if g.degree(n) == 1 and n not in protected]:
            if leaf not in g or g.degree(leaf) != 1:
                continue
            chain, prev, cur = [leaf], None, leaf
            while True:
                nxt = [n for n in g.neighbors(cur) if n != prev]
                if not nxt:
                    break
                prev, cur = cur, nxt[0]
                chain.append(cur)
                if g.degree(cur) != 2:
                    break
            end = chain[-1]
            if g.degree(end) < 3:
                continue  # an isolated chain is the whole vessel, not a spur
            reach = xyz[leaf] - xyz[end]
            dist = float(np.linalg.norm(reach))
            u = reach / dist if dist > 1e-9 else np.zeros(3)
            within = 2.0 * float(np.sqrt(max(u @ cov[end] @ u, 0.0)))
            r_branch = max(float(rmax[c]) for c in chain[:-1])
            if dist < within + min_branch_radii * r_branch:
                g.remove_nodes_from(chain[:-1])
                pruned += 1
                changed = True
    return pruned


def extract(mask, spacing_xyz_um, *, root_zyx=None, shell_um=None,
            min_branch_radii: float = 2.0, min_loop_radii: float = 8.0,
            max_voxels: int = 500_000_000) -> ScoopSkeleton:
    """Scoop a binary mask from its root(s); see the module docstring.

    `spacing_xyz_um` is the voxel size in ``(x, y, z)``, as everywhere else in this
    package; anisotropic spacing is supported because both the geodesic step lengths
    and the EDT are physical. `shell_um` defaults to the *largest* voxel side and may
    not be smaller: a shell thinner than one step along the coarse axis is skipped
    there, and clusters two shells apart are never joined, so the graph falls apart.

    Memory is about 9 bytes per box voxel (mask, lookup, component labels) plus about
    100 per *lumen* voxel; `max_voxels` bounds the lumen count.
    """
    import networkx as nx

    mask = np.asarray(mask) > 0
    if mask.ndim != 3:
        raise ValueError("geodesic scooping needs a 3-D mask")
    sp = np.asarray(spacing_xyz_um, dtype=np.float64)[::-1]  # zyx
    if sp.shape != (3,) or np.any(~np.isfinite(sp)) or np.any(sp <= 0):
        raise ValueError("spacing must be three positive numbers")
    shell = float(np.max(sp) if shell_um is None else shell_um)
    if shell < np.max(sp) - 1e-9:
        raise ValueError(f"shell_um {shell:g} is thinner than the largest voxel side "
                         f"{np.max(sp):g} um; shells would be skipped")

    fg = np.flatnonzero(mask.ravel())
    if len(fg) > max_voxels:
        raise ValueError(f"geodesic scooping input has {len(fg):,} lumen voxels; limit is "
                         f"{max_voxels:,}. Use --roi-zyx or --per-tree.")
    empty = ScoopSkeleton(np.zeros((0, 3)), np.zeros((0, 2), np.int64), np.zeros(0),
                          dict(method="geodesic-scoop", shells=0, clusters=0, loops=0,
                               pruned=0, contracted=0, roots=[]))
    if not len(fg):
        return empty
    lookup = np.full(mask.size, -1, dtype=np.int32 if len(fg) < 2**31 else np.int64)
    lookup[fg] = np.arange(len(fg))
    dz, dy, dx = mask.shape
    coords_i = np.column_stack(np.unravel_index(fg, mask.shape))
    edt_fg = _edt_at(mask, coords_i, sp)
    lengths = np.linalg.norm(OFFSETS * sp, axis=1)

    seeds = _roots(mask, fg, lookup, edt_fg, root_zyx, sp, lengths)
    flat = np.array([lookup[np.ravel_multi_index(s, mask.shape)] for s in seeds],
                    dtype=np.int64)
    gd = _compact_geodesic(lookup, fg, dz, dy, dx, OFFSETS, lengths, flat)
    reached = np.isfinite(gd)
    shell_idx = np.full(len(fg), -1, dtype=np.int64)
    shell_idx[reached] = np.floor(gd[reached] / shell).astype(np.int64)
    del gd

    roots, pairs = _compact_clusters(lookup, fg, shell_idx, dz, dy, dx, HALF_OFFSETS)
    del lookup
    ids, inv = np.unique(roots[reached], return_inverse=True)
    coords = coords_i[reached].astype(np.float64)
    count = np.bincount(inv).astype(np.float64)
    centroid = np.column_stack([np.bincount(inv, weights=coords[:, k]) / count
                                for k in range(3)])
    rmax = np.zeros(len(ids))
    np.maximum.at(rmax, inv, edt_fg[reached])
    # The scale used to judge spurs and small loops. Max EDT is the *minor* semi-axis
    # of a collapsed slit -- the very collapse this method is here for -- so a slit's
    # two tips at a flat end would pass as branches. A uniform ellipse has variance
    # a^2/4 along its major axis, so 2 sigma recovers `a`; take whichever is larger.
    phys = coords * sp
    mean = centroid * sp
    cov = np.empty((len(ids), 3, 3))
    for i in range(3):
        for j in range(i, 3):
            c = np.bincount(inv, weights=phys[:, i] * phys[:, j]) / count - mean[:, i] * mean[:, j]
            cov[:, i, j] = cov[:, j, i] = c
    half_extent = 2 * np.sqrt(np.maximum(np.linalg.eigvalsh(cov)[:, -1], 0))
    scale = np.maximum(rmax, half_extent)

    g = nx.Graph()
    g.add_nodes_from(range(len(ids)))
    if len(pairs):
        cp = np.sort(np.searchsorted(ids, pairs), axis=1)
        # One int64 key per pair: a 1-D unique, not a row-wise one (12 s -> under 1 s
        # on a whole LADAF-2024-28 tree, where there are ~30 M touching pairs).
        key = np.unique(cp[:, 0] * np.int64(len(ids)) + cp[:, 1])
        cp = np.column_stack([key // len(ids), key % len(ids)])
        cp = cp[cp[:, 0] != cp[:, 1]]
        g.add_edges_from(map(tuple, cp.tolist()))

    xyz = centroid * sp  # physical, zyx order, for lengths
    contracted = _contract_small_cycles(g, xyz, count, scale, min_loop_radii)
    protected = {int(np.searchsorted(ids, roots[f])) for f in flat}
    # Longer spurs are left to `skeleton_optimise.prune_spurs`, which the backend runs
    # and which knows not to prune an end cut by the lattice edge.
    pruned = _prune_spurs(g, xyz, rmax, cov, protected, min_branch_radii)

    keep = sorted(g.nodes)
    index = {n: i for i, n in enumerate(keep)}
    edges = np.array([(index[a], index[b]) for a, b in g.edges],
                     dtype=np.int64).reshape(-1, 2)
    loops = g.number_of_edges() - g.number_of_nodes() + nx.number_connected_components(g)
    return ScoopSkeleton(
        coordinates_zyx=xyz[keep] / sp,
        edges=edges,
        radii_um=rmax[keep],
        report=dict(method="geodesic-scoop", shell_um=shell,
                    shells=int(shell_idx.max() + 1), clusters=int(len(ids)),
                    loops=int(loops), contracted=int(contracted), pruned=int(pruned),
                    unreached_voxels=int(np.count_nonzero(~reached)),
                    roots=[list(s) for s in seeds],
                    radius_source="max EDT per shell cluster; placeholder"),
    )
