"""Mask -> centreline graph.

The pipeline has always taken its skeleton from Avizo's Centerline Tree module,
and nothing on this machine derives one from a mask. This does, so a skeleton can
be generated from the segmentation and *scored against* the Avizo one rather than
taken on trust.

Three steps:

1. ``skimage.morphology.skeletonize(mask, method="lee")`` -- Lee's 3-D thinning,
   which preserves topology, so the skeleton has the same connectivity as the mask.
2. ``distance_transform_edt(mask, sampling=spacing)`` -- the distance to the
   nearest background voxel, which on the medial axis *is* the radius, already in
   micrometres. This is why the EDT is computed on the mask and sampled at the
   skeleton rather than computed on the skeleton.
3. voxel skeleton -> graph, below.

``sknw`` is the usual answer to step 3 and is **not** used: it is not installed,
it is unmaintained, and the part that actually matters -- collapsing each junction
blob into a single node -- has to be written regardless.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from .adapter import Triple

# 26-connectivity: every neighbour of a voxel in 3-D, excluding itself.
_OFFSETS = np.array(
    [(dz, dy, dx)
     for dz in (-1, 0, 1) for dy in (-1, 0, 1) for dx in (-1, 0, 1)
     if (dz, dy, dx) != (0, 0, 0)],
    dtype=np.int64,
)


@dataclass
class SkeletonResult:
    """A generated skeleton and what it cost."""

    triple: Triple
    n_skeleton_voxels: int
    n_junction_clusters: int
    seconds: dict = field(default_factory=dict)

    def describe(self) -> str:
        t = self.seconds
        return (
            f"{len(self.triple.segments)} segments, {len(self.triple.nodes)} nodes, "
            f"{len(self.triple.points)} points "
            f"from {self.n_skeleton_voxels:,} skeleton voxels "
            f"({self.n_junction_clusters} junction clusters); "
            + ", ".join(f"{k} {v:.1f}s" for k, v in t.items())
        )


def skeletonise(mask: np.ndarray, spacing_um) -> tuple[np.ndarray, np.ndarray]:
    """``(skeleton, edt)`` for a binary mask indexed ``[z, y, x]``.

    `spacing_um` is ``(x, y, z)`` to match ``WorldFrame``; the EDT wants ``(z, y, x)``.

    Materialises the whole distance transform, so it is for volumes that comfortably
    fit: ``distance_transform_edt`` returns **float64**, which is 8 bytes a voxel and
    18.7 GB on the full LADAF lattice. Use :func:`edt_radii` at that scale.
    """
    from scipy import ndimage
    from skimage.morphology import skeletonize

    mask = np.asarray(mask) > 0
    skel = skeletonize(mask, method="lee").astype(bool)
    sampling = np.asarray(spacing_um, dtype=np.float64)[::-1]
    edt = ndimage.distance_transform_edt(mask, sampling=sampling)
    return skel, edt


def edt_radii(mask: np.ndarray, coords: np.ndarray, spacing_um, *,
              slab: int = 192, max_radius_um: float | None = None,
              progress=None) -> np.ndarray:
    """Distance-to-background at `coords` only, computed in z slabs.

    The radius is wanted at a few hundred thousand skeleton voxels, not at all
    2.3 billion of them -- but ``distance_transform_edt`` has no windowed form and
    returns float64, so asking for it directly costs 18.7 GB on the full lattice
    for an answer that occupies a couple of megabytes.

    So the transform is run over overlapping z slabs and sampled immediately, and
    only the samples are kept. The overlap is what makes this *exact* rather than
    approximate: a voxel's nearest background voxel can be at most one radius away,
    so a halo of `max_radius_um` guarantees each slab's interior sees every
    competitor it would have seen in the whole volume. The halo is sized from the
    data when not given, by measuring the largest distance found so far and
    growing if it ever reaches the halo -- which would mean the halo was too small.

    Returns ``(N,)`` float32 in micrometres, aligned with `coords`.
    """
    from scipy import ndimage

    mask = np.asarray(mask) > 0
    coords = np.asarray(coords, dtype=np.int64).reshape(-1, 3)
    sampling = np.asarray(spacing_um, dtype=np.float64)[::-1]
    nz = mask.shape[0]

    if max_radius_um is None:
        # A generous default: the largest vessel in a coronary tree is a couple of
        # millimetres across. Verified below and reported if it binds.
        max_radius_um = 3000.0
    halo = int(np.ceil(max_radius_um / max(float(sampling[0]), 1e-9))) + 2

    out = np.zeros(len(coords), dtype=np.float32)
    order = np.argsort(coords[:, 0])
    z_sorted = coords[order, 0]
    reached_halo = False

    for z0 in range(0, nz, slab):
        z1 = min(z0 + slab, nz)
        lo, hi = max(z0 - halo, 0), min(z1 + halo, nz)
        block = mask[lo:hi]
        if not block.any():
            continue
        dist = ndimage.distance_transform_edt(block, sampling=sampling)

        first = int(np.searchsorted(z_sorted, z0, side="left"))
        last = int(np.searchsorted(z_sorted, z1, side="left"))
        if last > first:
            sel = order[first:last]
            c = coords[sel]
            vals = dist[c[:, 0] - lo, c[:, 1], c[:, 2]]
            out[sel] = vals.astype(np.float32)
            if vals.size and float(vals.max()) >= (halo - 2) * float(sampling[0]):
                reached_halo = True
        del dist, block
        if progress is not None:
            progress(z1, nz)

    if reached_halo:
        print(
            f"    [warning] a radius reached the {max_radius_um:.0f} um slab halo; "
            f"pass a larger max_radius_um or the largest vessels are under-measured"
        )
    return out


def neighbour_counts(skel: np.ndarray) -> np.ndarray:
    """Number of 26-neighbours each skeleton voxel has, as an int8 volume.

    A convolution rather than a per-voxel loop: the skeleton of this dataset is
    still hundreds of thousands of voxels, and looping over them in Python to look
    at 26 neighbours each is minutes rather than seconds.
    """
    from scipy import ndimage

    kernel = np.ones((3, 3, 3), dtype=np.uint8)
    kernel[1, 1, 1] = 0
    counts = ndimage.convolve(skel.astype(np.uint8), kernel, mode="constant", cval=0)
    return np.where(skel, counts, 0).astype(np.int8)


def _cluster_nodes(node_voxels: np.ndarray) -> list[np.ndarray]:
    """Group touching node voxels into one cluster each.

    A single anatomical bifurcation thins to a *blob* of high-degree voxels, not
    one. Left alone that becomes several degree-3 nodes a voxel apart joined by
    zero-length edges, and every downstream measure -- bifurcation counts, Strahler
    order, Murray ratios -- is wrong. Same fix
    ``skeleton_analysis.optimisation.volume_metrics`` applies for the same reason.
    """
    import networkx as nx
    from scipy.spatial import cKDTree

    if len(node_voxels) == 0:
        return []
    tree = cKDTree(node_voxels)
    graph = nx.Graph()
    graph.add_nodes_from(range(len(node_voxels)))
    # sqrt(3) + eps reaches every 26-neighbour and nothing further.
    graph.add_edges_from(tree.query_pairs(np.sqrt(3) + 1e-6))
    return [np.array(sorted(c), dtype=np.int64) for c in nx.connected_components(graph)]


def skeleton_to_graph(
    skel: np.ndarray,
    edt: np.ndarray,
    origin_um,
    spacing_um,
    *,
    min_branch_voxels: int = 0,
) -> SkeletonResult:
    """Trace a voxel skeleton into a :class:`~.adapter.Triple`, in micrometres.

    Nodes are the voxels whose degree is not 2 -- degree 1 is a free end, degree 3
    or more a junction -- with touching ones merged. Edges are the degree-2 chains
    between them.

    `edt` supplies the radii and may be either the full 3-D distance transform, or
    a 1-D array already sampled at ``np.argwhere(skel)`` order -- which is what
    :func:`edt_radii` returns, and the only affordable form at full resolution.

    Cycles with no node on them (a closed loop) would otherwise be dropped
    entirely, so each is broken at an arbitrary voxel and emitted as an edge from
    that voxel back to itself.
    """
    t0 = time.time()
    skel = np.asarray(skel) > 0
    origin = np.asarray(origin_um, dtype=np.float64)
    spacing = np.asarray(spacing_um, dtype=np.float64)

    counts = neighbour_counts(skel)
    coords = np.argwhere(skel)  # (N, 3) in (z, y, x)
    if len(coords) == 0:
        return SkeletonResult(Triple({}, {}, []), 0, 0, {"total": time.time() - t0})

    edt = np.asarray(edt)
    if edt.ndim == 1:
        if len(edt) != len(coords):
            raise ValueError(
                f"radii has {len(edt)} entries but the skeleton has {len(coords)} voxels"
            )
        radius_of = dict(zip(map(tuple, coords), edt))

        def radius_at(z, y, x):
            return float(radius_of[(z, y, x)])
    else:
        def radius_at(z, y, x):
            return float(edt[z, y, x])

    # A dense lookup from voxel -> index into `coords`. -1 means "not skeleton".
    index = np.full(skel.shape, -1, dtype=np.int64)
    index[coords[:, 0], coords[:, 1], coords[:, 2]] = np.arange(len(coords))
    degree = counts[coords[:, 0], coords[:, 1], coords[:, 2]].astype(np.int64)
    t_prep = time.time() - t0

    # -- adjacency, once, as a ragged list ---------------------------------- #
    t = time.time()
    shape = np.asarray(skel.shape)
    neighbours: list[list[int]] = [[] for _ in range(len(coords))]
    for off in _OFFSETS:
        shifted = coords + off
        ok = np.all((shifted >= 0) & (shifted < shape), axis=1)
        hits = np.flatnonzero(ok)
        if not len(hits):
            continue
        s = shifted[hits]
        other = index[s[:, 0], s[:, 1], s[:, 2]]
        real = other >= 0
        for a, b in zip(hits[real], other[real]):
            neighbours[a].append(int(b))
    t_adj = time.time() - t

    # -- node clusters ------------------------------------------------------- #
    t = time.time()
    node_idx = np.flatnonzero(degree != 2)
    clusters = _cluster_nodes(coords[node_idx]) if len(node_idx) else []
    cluster_of = np.full(len(coords), -1, dtype=np.int64)
    for cid, members in enumerate(clusters):
        cluster_of[node_idx[members]] = cid
    t_cluster = time.time() - t

    nodes: dict[int, tuple] = {}
    for cid, members in enumerate(clusters):
        centre = coords[node_idx[members]].mean(axis=0)  # (z, y, x)
        xyz = origin + centre[::-1] * spacing
        nodes[cid] = (float(xyz[0]), float(xyz[1]), float(xyz[2]), 0)

    # -- trace the chains ---------------------------------------------------- #
    t = time.time()
    points: dict[int, tuple] = {}
    segments: list[dict] = []
    next_pid = 0
    visited_chain = np.zeros(len(coords), dtype=bool)

    def emit(chain: list[int], node_a: int, node_b: int) -> None:
        nonlocal next_pid
        if len(chain) < 2 or (min_branch_voxels and len(chain) < min_branch_voxels
                              and node_a != node_b
                              and (degree[chain[0]] == 1 or degree[chain[-1]] == 1)):
            return
        pids = []
        for vi in chain:
            z, y, x = coords[vi]
            xyz = origin + np.array([x, y, z], dtype=np.float64) * spacing
            points[next_pid] = (
                float(xyz[0]), float(xyz[1]), float(xyz[2]), radius_at(z, y, x)
            )
            pids.append(next_pid)
            next_pid += 1
        segments.append({
            "id": len(segments), "node1": int(node_a), "node2": int(node_b),
            "point_ids": pids,
        })

    for start in node_idx:
        cid = cluster_of[start]
        for first in neighbours[start]:
            if cluster_of[first] >= 0:
                # Node touching node: a chain of length zero. Only emit it when the
                # two belong to different clusters, otherwise it is inside a blob.
                if cluster_of[first] != cid and first > start:
                    emit([start, first], cid, cluster_of[first])
                continue
            if visited_chain[first]:
                continue
            chain = [start, first]
            visited_chain[first] = True
            prev, cur = start, first
            while True:
                nxt = [n for n in neighbours[cur] if n != prev and not visited_chain[n]]
                # Prefer stepping onto a node: that ends the chain cleanly.
                node_step = [n for n in neighbours[cur] if n != prev and cluster_of[n] >= 0]
                if node_step:
                    chain.append(node_step[0])
                    emit(chain, cid, cluster_of[node_step[0]])
                    break
                if not nxt:
                    # A chain that runs out without reaching a node: keep it, ending
                    # on its own last voxel as a new degree-1 node.
                    end_cid = len(nodes)
                    z, y, x = coords[cur]
                    xyz = origin + np.array([x, y, z], dtype=np.float64) * spacing
                    nodes[end_cid] = (float(xyz[0]), float(xyz[1]), float(xyz[2]), 0)
                    emit(chain, cid, end_cid)
                    break
                prev, cur = cur, nxt[0]
                visited_chain[cur] = True
                chain.append(cur)

    # -- closed loops carrying no node --------------------------------------- #
    for vi in range(len(coords)):
        if visited_chain[vi] or cluster_of[vi] >= 0 or degree[vi] != 2:
            continue
        # Break the loop here and walk all the way round.
        cid = len(nodes)
        z, y, x = coords[vi]
        xyz = origin + np.array([x, y, z], dtype=np.float64) * spacing
        nodes[cid] = (float(xyz[0]), float(xyz[1]), float(xyz[2]), 0)
        chain = [vi]
        visited_chain[vi] = True
        prev, cur = vi, None
        nxt = [n for n in neighbours[vi] if not visited_chain[n]]
        if not nxt:
            continue
        cur = nxt[0]
        visited_chain[cur] = True
        chain.append(cur)
        while True:
            step = [n for n in neighbours[cur] if n != prev and not visited_chain[n]]
            if not step:
                chain.append(vi)  # close it
                break
            prev, cur = cur, step[0]
            visited_chain[cur] = True
            chain.append(cur)
        emit(chain, cid, cid)
    t_trace = time.time() - t

    # Coordination numbers are derived, exactly as EditableGraph maintains them.
    degrees = {nid: 0 for nid in nodes}
    for seg in segments:
        for key in ("node1", "node2"):
            degrees[seg[key]] = degrees.get(seg[key], 0) + 1
    for nid, node in nodes.items():
        nodes[nid] = (node[0], node[1], node[2], int(degrees.get(nid, 0)))

    return SkeletonResult(
        triple=Triple(nodes=nodes, points=points, segments=segments),
        n_skeleton_voxels=int(len(coords)),
        n_junction_clusters=len(clusters),
        seconds={
            "prep": t_prep, "adjacency": t_adj,
            "cluster": t_cluster, "trace": t_trace, "total": time.time() - t0,
        },
    )


def skeletonise_lattice(labels, frame, *, stride: int = 1, verbose: bool = True
                        ) -> SkeletonResult:
    """Decode, skeletonise and trace, in one call.

    `stride` decimates the volume first, which trades resolution for memory: the
    full lattice is 2.34 GB before the skeleton and EDT are allocated on top.
    """
    from skimage.morphology import skeletonize

    from .lattice import decode_volume

    t = time.time()
    if verbose:
        print("  decoding the lattice...")
    volume = decode_volume(labels, stride=stride) > 0
    if verbose:
        print(f"    {volume.shape} = {volume.nbytes / 1e9:.2f} GB in {time.time()-t:.1f}s, "
              f"{int(volume.sum()):,} foreground voxels "
              f"({100 * volume.mean():.2f}%)")

    spacing = np.asarray(frame.seg_spacing, dtype=np.float64) * max(int(stride), 1)

    t = time.time()
    if verbose:
        print("  skeletonising (lee)...")
    skel = skeletonize(volume, method="lee").astype(bool)
    coords = np.argwhere(skel)
    if verbose:
        print(f"    {len(coords):,} skeleton voxels in {time.time()-t:.1f}s")

    t = time.time()
    if verbose:
        print("  distance transform, in slabs...")

    def report(done, total):
        if verbose:
            print(f"    {done}/{total} slices", end="\r")

    radii = edt_radii(volume, coords, spacing, progress=report if verbose else None)
    if verbose:
        print(f"    radii {radii.min():.0f}-{radii.max():.0f} um "
              f"in {time.time()-t:.1f}s" + " " * 20)
    del volume

    if verbose:
        print("  tracing the graph...")
    result = skeleton_to_graph(skel, radii, frame.seg_origin, spacing)
    if verbose:
        print("   ", result.describe())
    return result
