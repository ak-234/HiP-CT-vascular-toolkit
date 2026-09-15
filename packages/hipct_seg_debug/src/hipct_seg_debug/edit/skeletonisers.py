"""Several skeletonisation algorithms behind one signature, so they can be scored.

The Walsh-Berg paper's central finding is that the choice of skeletonisation algorithm
alone changes the answer enough to reverse a conclusion: on one consensus segmentation,
the skeleton with the most nodes had **more than double** the nodes and segments of the
one with the fewest, radius/LDR/branching-angle/IVD/tortuosity distributions differed at
p <= 0.0001 across nearly every pair, and simulated perfusion flow differed by an order
of magnitude -- with no consistency about which algorithm gave the largest. So the
algorithm cannot be chosen by reputation; it has to be *measured*, per dataset, which is
what :mod:`~.supermetric` is for and what this module supplies candidates to.

The paper's four families, and what is reachable from Python:

===================  ===========================  ==============================
family               representative               here
===================  ===========================  ==============================
thinning             Lee et al.; VesselVio        ``"lee"`` -- ``skimage``
thinning + distance  Palagyi/Fouard; **Amira**    ``"amira"`` -- ingest an export
minimum-cost path    **TEASAR**; Amira Centerline ``"teasar"`` -- ``kimimaro``
wave-front scooping  Rodriguez/Wu; MOST, Vaa3D    not available
===================  ===========================  ==============================

DTHO is the family the paper found best overall (super metric 1.39, and the least
parameter-sensitive of the four), and it exists here only as an Avizo export -- the
paper hit the same wall, describing all four implementations as "hardly scriptable".
Ingesting the ``.am`` at least lets the commercial result be *scored* on the same terms
as the ones that can be run, which is the question that matters.

Every backend returns a :class:`~.adapter.Triple`, so nothing downstream --
:mod:`~.skeleton_optimise`, :mod:`~.radius_perimeter`, :mod:`~.supermetric` -- needs to
know which produced it.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

ALGORITHMS = ("lee", "teasar", "amira")

# TEASAR's two shape parameters, named after Amira Centerline Tree's `slope` and
# `zeroVal` because they play the same role: how far from the boundary a voxel has to
# be before it is allowed to be an endpoint. The paper swept slope 1-6 and zeroVal
# 1-10 (Table S14) and improved Centerline Tree's super metric from 3.13 to 2.13.
TEASAR_SCALE = 2.5
TEASAR_CONST_UM = 300.0


@dataclass
class SkeletonCandidate:
    """One algorithm's skeleton, and what it cost to produce."""

    name: str
    triple: object
    params: dict = field(default_factory=dict)
    seconds: float = 0.0
    detail: str = ""
    # 1 for a whole-volume run; the number of mask components when the skeleton was
    # derived per tree. Defaulted so nothing that builds a candidate has to care.
    trees: int = 1
    tree_voxels: tuple = ()

    def describe(self) -> str:
        t = self.triple
        p = ", ".join(f"{k}={v}" for k, v in self.params.items())
        return (
            f"{self.name}: {len(t.segments)} segments, {len(t.nodes)} nodes, "
            f"{len(t.points)} points in {self.seconds:.1f}s"
            + (f", {self.trees} trees" if self.trees != 1 else "")
            + (f" [{p}]" if p else "")
            + (f" -- {self.detail}" if self.detail else "")
        )


def edges_to_triple(vertices, edges, radii, *, origin_um=None):
    """A vertex/edge skeleton -> :class:`~.adapter.Triple`.

    The counterpart of :func:`~.skeletonise.skeleton_to_graph` for algorithms that
    already return a graph rather than a voxel mask. Same rule: **nodes are the
    vertices whose degree is not 2**, and the degree-2 chains between them become
    segments -- otherwise every vertex would become a node and every measure that
    counts bifurcations would be meaningless.

    `vertices` is ``(N, 3)`` in micrometres, `edges` ``(M, 2)`` indices, `radii`
    ``(N,)`` in micrometres. A cycle carrying no node is broken at an arbitrary vertex
    and emitted as a segment from that vertex back to itself, exactly as
    :func:`~.skeletonise.skeleton_to_graph` does, because dropping it would silently
    lose a whole ring.
    """
    import networkx as nx

    from .adapter import Triple

    verts = np.asarray(vertices, dtype=np.float64).reshape(-1, 3)
    if origin_um is not None:
        verts = verts + np.asarray(origin_um, dtype=np.float64)
    rad = np.asarray(radii, dtype=np.float64).ravel()
    g = nx.Graph()
    g.add_nodes_from(range(len(verts)))
    g.add_edges_from(np.asarray(edges, dtype=np.int64).reshape(-1, 2).tolist())

    nodes: dict[int, tuple] = {}
    points: dict[int, tuple] = {}
    segments: list[dict] = []
    node_of_vertex: dict[int, int] = {}
    next_pid = 0

    def node_for(vi: int) -> int:
        nid = node_of_vertex.get(vi)
        if nid is None:
            nid = len(nodes)
            x, y, z = verts[vi]
            nodes[nid] = (float(x), float(y), float(z), 0)
            node_of_vertex[vi] = nid
        return nid

    def emit(chain, na, nb):
        nonlocal next_pid
        if len(chain) < 2:
            return
        pids = []
        for vi in chain:
            x, y, z = verts[vi]
            points[next_pid] = (float(x), float(y), float(z), float(rad[vi]))
            pids.append(next_pid)
            next_pid += 1
        segments.append({"id": len(segments), "node1": na, "node2": nb,
                         "point_ids": pids})

    branch = [v for v in g.nodes if g.degree(v) != 2]
    walked: set[tuple[int, int]] = set()

    for start in branch:
        for nxt in g.neighbors(start):
            if (start, nxt) in walked:
                continue
            chain = [start, nxt]
            walked.add((start, nxt))
            walked.add((nxt, start))
            prev, cur = start, nxt
            while g.degree(cur) == 2:
                step = [n for n in g.neighbors(cur) if n != prev]
                if not step:
                    break
                prev, cur = cur, step[0]
                walked.add((prev, cur))
                walked.add((cur, prev))
                chain.append(cur)
            emit(chain, node_for(start), node_for(cur))

    # Rings with no branch point on them: break each at an arbitrary vertex.
    for comp in nx.connected_components(g):
        if any(g.degree(v) != 2 for v in comp):
            continue
        ring = sorted(comp)
        start = ring[0]
        nid = node_for(start)
        chain = [start]
        prev, cur = start, next(iter(g.neighbors(start)))
        chain.append(cur)
        while cur != start:
            step = [n for n in g.neighbors(cur) if n != prev]
            if not step:
                break
            prev, cur = cur, step[0]
            chain.append(cur)
        emit(chain, nid, nid)

    degrees: dict[int, int] = {nid: 0 for nid in nodes}
    for seg in segments:
        for key in ("node1", "node2"):
            degrees[seg[key]] = degrees.get(seg[key], 0) + 1
    for nid, node in nodes.items():
        nodes[nid] = (node[0], node[1], node[2], int(degrees.get(nid, 0)))

    return Triple(nodes=nodes, points=points, segments=segments)


# ------------------------------------------------------------------- backends


def _lee(volume, frame, spacing, *, origin_um=None, **params) -> tuple:
    """Lee medial-axis thinning, with the EDT as the radius. The existing path."""
    from skimage.morphology import skeletonize

    from .skeletonise import edt_radii, skeleton_to_graph

    mask = np.asarray(volume) > 0
    skel = skeletonize(mask, method="lee").astype(bool)
    coords = np.argwhere(skel)
    radii = edt_radii(mask, coords, spacing)
    origin = frame.seg_origin if origin_um is None else origin_um
    result = skeleton_to_graph(skel, radii, origin, spacing)
    return result.triple, f"{result.n_skeleton_voxels:,} skeleton voxels"


def _teasar(volume, frame, spacing, *, scale: float = TEASAR_SCALE,
            const_um: float = TEASAR_CONST_UM, dust_threshold: int = 100,
            fix_branching: bool = True, origin_um=None, **params) -> tuple:
    """TEASAR via ``kimimaro`` -- the minimum-cost-path family.

    TEASAR produces a **strict tree**, which is the family's defining behaviour and,
    per the paper, its defining cost: forcing a tree breaks every loop and drove the
    terminal-node fraction to 90% on the FaDu network. Here that is close to free,
    because :func:`~.skeleton_optimise.remove_loops` wants the loops gone anyway --
    coronary arteries at this calibre are a tree. It is also the one family that
    *preserved* the segmentation's connected-component count in the paper's tests.

    ``kimimaro`` indexes ``(x, y, z)`` while everything here is ``(z, y, x)``, so the
    volume is transposed on the way in and the vertices come back already in
    micrometres relative to voxel zero, needing only the frame's origin added.
    """
    try:
        import kimimaro
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise ImportError(
            "the 'teasar' skeletoniser needs kimimaro, which is not installed.\n"
            "    pip install kimimaro\n"
            "Every other part of this toolkit works without it; pass "
            "--algorithms lee to skip TEASAR."
        ) from exc

    # Keep one C-contiguous copy in kimimaro's x/y/z order.  Values need only be
    # labels (the source is 0/255 as often as 0/1), so converting to boolean and then
    # back to uint8 would create another volume-sized temporary for no benefit.
    mask = np.ascontiguousarray(np.asarray(volume).transpose(2, 1, 0), dtype=np.uint8)
    skels = kimimaro.skeletonize(
        mask,
        teasar_params={
            "scale": float(scale),
            "const": float(const_um),
            "pdrf_scale": 100000,
            "pdrf_exponent": 4,
            "soma_detection_threshold": 0,
            "soma_acceptance_threshold": 0,
        },
        anisotropy=tuple(float(v) for v in spacing),
        dust_threshold=int(dust_threshold),
        fix_branching=bool(fix_branching),
        progress=False,
    )
    if not skels:
        from .adapter import Triple

        return Triple({}, {}, []), "kimimaro returned no skeleton"

    # kimimaro keys by label; the mask is binary so there is one, but join defensively.
    verts, edges, radii, offset = [], [], [], 0
    for skel in skels.values():
        verts.append(np.asarray(skel.vertices, dtype=np.float64))
        edges.append(np.asarray(skel.edges, dtype=np.int64) + offset)
        r = getattr(skel, "radii", None)
        radii.append(
            np.asarray(r, dtype=np.float64) if r is not None
            else np.zeros(len(skel.vertices))
        )
        offset += len(skel.vertices)

    triple = edges_to_triple(
        np.vstack(verts), np.vstack(edges), np.concatenate(radii),
        origin_um=frame.seg_origin if origin_um is None else origin_um,
    )
    return triple, f"{offset:,} TEASAR vertices, scale={scale}, const={const_um} um"


def _amira(volume, frame, spacing, *, graph: str | None = None, origin_um=None,
           **params) -> tuple:
    """Ingest an externally produced Avizo ``.am`` as a candidate.

    AutoSkeleton (parallelised DTHO) and Centerline Tree (TEASAR) are not scriptable,
    so the only way to put the commercial results on the same footing as the ones that
    can be run here is to read the export and score it identically.
    """
    from pathlib import Path

    from .adapter import read_triple

    if not graph:
        raise ValueError(
            "the 'amira' skeletoniser scores an existing export, so it needs the "
            "path to one: pass --amira-graph"
        )
    triple = read_triple(graph)
    return triple, f"ingested {Path(graph).name}"


_BACKENDS = {"lee": _lee, "teasar": _teasar, "amira": _amira}


def skeletonise(name: str, volume, frame, **params) -> SkeletonCandidate:
    """Run one algorithm by name over an already-decoded volume.

    The volume is decoded **once** by the caller and handed to every backend, because
    at stride 1 it is 2.34 GB and decoding it per algorithm is minutes of pure repeat.

    **`frame` must describe `volume`.** There is no `stride` argument on purpose: a
    decimated volume needs a frame whose spacing is already multiplied, which is what
    ``__main__._decoded`` builds, and taking a stride here as well applied it twice.
    That put every coordinate eight times too far out at stride 8 -- far enough that
    the centreline scored a cl-sensitivity of exactly zero against the mask it was
    derived from, and a network volume 600 times too large.
    """
    if name not in _BACKENDS:
        raise ValueError(f"unknown skeletoniser {name!r}; choose from {ALGORITHMS}")
    spacing = np.asarray(frame.seg_spacing, dtype=np.float64)
    t0 = time.time()
    triple, detail = _BACKENDS[name](volume, frame, spacing, **params)
    return SkeletonCandidate(
        name=name, triple=triple, params=dict(params),
        seconds=time.time() - t0, detail=detail,
    )


def skeletonise_per_component(name: str, volume, frame, *, min_component_voxels: int = 0,
                              max_trees: int | None = None, connectivity: int = 3,
                              verbose: bool = False, materials=None,
                              **params) -> SkeletonCandidate:
    """Skeletonise each mask component separately and tag every edge with its tree.

    Each component is cropped to its own bounding box and skeletonised there, then the
    pieces are merged with a per-edge ``tree`` field naming the mask component they
    came from. See :mod:`~.components` for why the tree index is a property of the
    *mask* and not of the graph.

    `materials` is the mask's ``Materials`` block, when it has one. With it the split
    runs per material first, so a file that already separates ``Left_Tree`` from
    ``Right_Tree`` is skeletonised per *tree* rather than per connected blob -- which
    matters most exactly where connectivity fails, on two coronaries that touch.

    **The skeleton itself moves slightly.** Lee thinning inside a tight box and
    ``edt_radii``'s slab halo see a different neighbourhood than a whole-volume run
    does, so segment counts shift by a fraction of a percent. This is a different
    skeleton, not a regrouping of the same one.

    ``amira`` has no mask to split -- it ingests an existing export -- so it is run
    once and its edges are assigned to trees afterwards by
    :func:`~.components.assign_trees`.
    """
    from . import components as comp
    from .adapter import Triple
    from .graphmodel import EditableGraph
    from .optimise import set_edge_field

    if name not in _BACKENDS:
        raise ValueError(f"unknown skeletoniser {name!r}; choose from {ALGORITHMS}")
    spacing = np.asarray(frame.seg_spacing, dtype=np.float64)
    t0 = time.time()

    parts, stats = comp.split_components(
        volume, frame, min_voxels=min_component_voxels, max_trees=max_trees,
        connectivity=connectivity, verbose=verbose, materials=materials,
    )

    if name == "amira":
        triple, detail = _amira(volume, frame, spacing, **params)
        if parts:
            graph = EditableGraph(triple)
            trees = comp.assign_trees(
                graph, stats.labels, frame,
                order=[p.label for p in parts], verbose=verbose,
            )
            set_edge_field(triple, comp.TREE_FIELD, trees, np.int64)
            detail += f", assigned to {len(set(trees.tolist()))} tree(s) from the mask"
        return SkeletonCandidate(
            name=name, triple=triple, params=dict(params), seconds=time.time() - t0,
            detail=detail, trees=len(parts),
            tree_voxels=tuple(p.voxels for p in parts),
        )

    if not parts:
        return SkeletonCandidate(
            name=name, triple=Triple({}, {}, []), params=dict(params),
            seconds=time.time() - t0, detail="no component to skeletonise", trees=0,
        )

    pieces, notes = [], []
    for part in parts:
        if verbose:
            print(f"  -- {part.describe()}")
        triple, detail = _BACKENDS[name](
            part.volume, frame, spacing, origin_um=part.origin_um, **params
        )
        pieces.append((part.index, triple))
        notes.append(f"tree {part.index}: {detail}")

    merged = comp.merge_triples(pieces)
    return SkeletonCandidate(
        name=name, triple=merged, params=dict(params), seconds=time.time() - t0,
        detail="; ".join(notes), trees=len(parts),
        tree_voxels=tuple(p.voxels for p in parts),
    )
