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
minimum-cost path    **TEASAR**; Amira Centerline ``"teasar"``, ``"teasar-readme"``
                                                  -- ``kimimaro``
minimum-cost path    Jin et al. 2016              ``"jin-mcp"``; ``"jin-mgf"``
                                                  (collapse-aware)
wave-front scooping  Rodriguez/Wu; MOST, Vaa3D    ``"geodesic-scoop"``
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
from contextlib import contextmanager
from dataclasses import dataclass, field

import numpy as np

from ..frame import is_isotropic

ALGORITHMS = ("lee", "teasar", "teasar-readme", "amira", "jin-mcp", "jin-mcp-centroid",
              "vmtk", "geodesic-scoop", "jin-mgf")

# TEASAR's two shape parameters, named after Amira Centerline Tree's `slope` and
# `zeroVal` because they play the same role: how far from the boundary a voxel has to
# be before it is allowed to be an endpoint. The paper swept slope 1-6 and zeroVal
# 1-10 (Table S14) and improved Centerline Tree's super metric from 3.13 to 2.13.
TEASAR_SCALE = 2.5
TEASAR_CONST_UM = 300.0

# Point attribute carrying kimimaro's own cross-sectional area, in um^2.
KIMIMARO_AREA = "kimimaro_area_um2"


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


def edges_to_triple(vertices, edges, radii, *, origin_um=None, point_attrs=None):
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

    `point_attrs` is ``{name: (N,) array}`` of per-vertex scalars to carry beside the
    radius -- kimimaro's cross-sectional area, for one. A vertex shared by several
    segments is written as one point per segment, and each copy gets the vertex's value.
    """
    import networkx as nx

    from .adapter import Triple

    verts = np.asarray(vertices, dtype=np.float64).reshape(-1, 3)
    if origin_um is not None:
        verts = verts + np.asarray(origin_um, dtype=np.float64)
    rad = np.asarray(radii, dtype=np.float64).ravel()
    vertex_values = {name: np.asarray(v, dtype=np.float64).ravel()
                     for name, v in (point_attrs or {}).items()}
    for name, v in vertex_values.items():
        if len(v) != len(verts):
            raise ValueError(f"point attribute {name!r} has {len(v)} values for "
                             f"{len(verts)} vertices")
    attrs: dict[str, dict[int, float]] = {name: {} for name in vertex_values}
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
            for name, v in vertex_values.items():
                attrs[name][next_pid] = float(v[vi])
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

    triple = Triple(nodes=nodes, points=points, segments=segments)
    for name, values in attrs.items():
        triple.point_attrs[name] = values
        triple.point_attr_dtypes[name] = np.dtype(np.float32)
    return triple


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


def _jin_engine(volume, max_voxels, engine):
    """``"dense"`` up to `max_voxels`, ``"sparse"`` beyond, unless one is named."""
    if engine not in ("auto", "dense", "sparse"):
        raise ValueError("Jin engine must be 'auto', 'dense' or 'sparse'")
    if engine == "auto":
        return "sparse" if np.asarray(volume).size > max_voxels else "dense"
    if engine == "dense" and np.asarray(volume).size > max_voxels:
        raise ValueError('Jin MCP input exceeds max_voxels; use --roi-zyx, or engine=sparse')
    return engine


def _jin_mcp(volume, frame, spacing, *, origin_um=None, root_zyx=None,
             max_voxels=2_000_000, report_path=None, report_dir=None, engine="auto",
             allow_loops=False, incremental=True, **params) -> tuple:
    """Binary-label adapter for Jin's fuzzy-object algorithm (opt-in).

    ``engine="auto"`` runs the dense reference implementation up to `max_voxels` and
    :mod:`~.jin_sparse` -- the same method on lumen voxels only -- beyond it, which is
    what makes a whole tree possible. `allow_loops` lets a mask with loops through the
    tree-only guard; the output is still a tree.
    """
    import json
    from pathlib import Path

    from . import jin_sparse
    from .jin_mcp import extract

    if not is_isotropic(spacing):
        raise ValueError('Jin MCP currently requires isotropic voxel spacing')
    if _jin_engine(volume, max_voxels, engine) == "sparse":
        result = jin_sparse.extract(np.asarray(volume) > 0, root_zyx=root_zyx,
                                    allow_loops=allow_loops, incremental=incremental,
                                    verbose=True)
    else:
        result = extract(np.asarray(volume) > 0, root_zyx=root_zyx, max_voxels=max_voxels,
                         allow_loops=allow_loops)
    origin = frame.seg_origin if origin_um is None else origin_um
    result.report.update(origin_xyz_um=list(map(float, origin)), spacing_xyz_um=list(map(float, spacing)),
                         shape_zyx=list(volume.shape), membership='binary nonzero labels')
    triple = edges_to_triple(result.coordinates_zyx[:, ::-1] * spacing,
                            result.edges, result.fdt * spacing[0], origin_um=origin)
    if not len(result.edges):
        xyz = result.coordinates_zyx[0, ::-1] * spacing + origin
        triple.nodes[0] = (*map(float, xyz), 0)
    if report_dir is not None:
        import hashlib
        digest = hashlib.sha256(np.packbits(np.asarray(volume) > 0).tobytes()
                                + np.asarray(origin, dtype=np.float64).tobytes()).hexdigest()[:16]
        Path(report_dir).mkdir(parents=True, exist_ok=True)
        report_path = Path(report_dir) / f'component_{digest}.json'
    if report_path is not None:
        # Intended for a single ROI. Per-component callers should use separate paths.
        Path(report_path).write_text(json.dumps(result.report, indent=2), encoding='utf-8')
    return triple, (f'Jin MCP experimental: {len(result.report["rounds"])} rounds, '
                    f'{result.report["unmarked_voxels"]} unmarked voxels; '
                    'new IDs; FDT thickness placeholders require perimeter remeasurement')


def _jin_mcp_centroid(volume, frame, spacing, *, origin_um=None, refine_iterations=5,
                      report_dir=None, **params):
    """Jin extraction followed by constrained centroid/curvature fitting.

    A deliberately named hybrid: the published shortest-path cost is unchanged.
    This does not prune the extra branches a medial sheet may have generated.
    """
    import hashlib
    import json
    from pathlib import Path
    from dataclasses import replace
    from .centreline_refine import refine
    from .graphmodel import EditableGraph
    triple, detail = _jin_mcp(volume, frame, spacing, origin_um=origin_um,
                              report_dir=report_dir, **params)
    if not triple.segments:
        return triple, detail + '; no curve to refine'
    origin = frame.seg_origin if origin_um is None else origin_um
    local_frame = replace(frame, seg_origin=np.asarray(origin),
                          seg_dims=np.array(volume.shape[::-1]), seg_spacing=np.asarray(spacing))
    graph = EditableGraph(triple)
    report = refine(graph, local_frame, volume, method='dfs-centroid-shape',
                    max_iterations=refine_iterations, strength=.01).to_dict()
    if report_dir is not None:
        digest = hashlib.sha256(np.packbits(np.asarray(volume) > 0).tobytes()
                                + np.asarray(origin, dtype=np.float64).tobytes()).hexdigest()[:16]
        Path(report_dir, f'component_{digest}_refinement.json').write_text(
            json.dumps(report, indent=2), encoding='utf-8')
    return triple, detail + f'; centroid/curvature hybrid converged={report["converged"]}'


def _vmtk(volume, frame, spacing, *, origin_um=None, source_points=None, target_points=None, **params):
    from .vmtk_baseline import extract
    return extract(volume, spacing, frame.seg_origin if origin_um is None else origin_um,
                   source_points, target_points)


def _teasar(volume, frame, spacing, *, scale: float = TEASAR_SCALE,
            const_um: float = TEASAR_CONST_UM, dust_threshold: int = 100,
            fix_branching: bool = True, origin_um=None,
            pdrf_scale: float = 100000, pdrf_exponent: float = 4,
            max_paths: int | None = None, fill_holes: bool = False,
            fix_borders: bool = True, parallel: int = 1,
            postprocess: bool = False, postprocess_dust_um: float | None = None,
            tick_radii: float = 2.0, cross_section: bool = False,
            smoothing_window: int = 5, **params) -> tuple:
    """TEASAR via ``kimimaro`` -- the minimum-cost-path family.

    The keywords after `origin_um` are the rest of the pipeline the kimimaro README
    documents, all off or at kimimaro's values by default so a plain ``teasar`` run is
    unchanged. ``teasar-readme`` switches them on:

    * `postprocess` runs ``kimimaro.postprocess``, which joins fragments and removes
      ticks shorter than `tick_radii` times the skeleton's median radius (both
      thresholds are physical units, so they are converted from radii here);
    * `cross_section` runs ``kimimaro.cross_sectional_area`` on the final vertices and
      carries the area per point as the ``kimimaro_area_um2`` point attribute, so
      TEASAR's own section area can be scored beside the perimeter pass.

    Soma detection stays disabled whatever is passed: the README's soma defaults are
    for neurons, and a coronary ostium would trip them.

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
    teasar_params = {
        "scale": float(scale),
        "const": float(const_um),
        # kimimaro bit-tests the exponent for a power of two, so it must stay an int.
        "pdrf_scale": int(pdrf_scale) if float(pdrf_scale).is_integer() else float(pdrf_scale),
        "pdrf_exponent": int(pdrf_exponent),
        "soma_detection_threshold": 0,
        "soma_acceptance_threshold": 0,
    }
    if max_paths is not None:
        teasar_params["max_paths"] = int(max_paths)
    anisotropy = tuple(float(v) for v in spacing)
    skels = kimimaro.skeletonize(
        mask,
        teasar_params=teasar_params,
        anisotropy=anisotropy,
        dust_threshold=int(dust_threshold),
        fix_branching=bool(fix_branching),
        fix_borders=bool(fix_borders),
        fill_holes=bool(fill_holes),
        parallel=int(parallel),
        progress=False,
    )
    if not skels:
        from .adapter import Triple

        return Triple({}, {}, []), "kimimaro returned no skeleton"

    steps = []
    with _osteoid_uint32_edges(kimimaro):
        if postprocess:
            for label, skel in list(skels.items()):
                r = np.asarray(getattr(skel, "radii", np.zeros(0)), dtype=np.float64)
                median_r = float(np.median(r)) if len(r) else float(np.min(spacing))
                dust = (float(postprocess_dust_um) if postprocess_dust_um is not None
                        else 2.0 * tick_radii * median_r)
                done = kimimaro.postprocess(
                    skel, dust_threshold=dust, tick_threshold=tick_radii * median_r)
                # When postprocess merges pieces, osteoid 0.6.0 returns the result
                # tagged `space="voxel"` with an identity transform although its
                # vertices are still micrometres. cross_sectional_area then treats
                # them as voxel indices, finds every one out of bounds and leaves
                # every area 0 -- measured on the collapse phantom at scale 2.5.
                done.space, done.transform = skel.space, np.copy(skel.transform)
                skels[label] = done
            steps.append(f"postprocess (ticks < {tick_radii} median radii)")
        if cross_section:
            skels = kimimaro.cross_sectional_area(
                mask, skels, anisotropy=np.asarray(anisotropy, dtype=np.float32),
                smoothing_window=int(smoothing_window), progress=False)
            steps.append("cross-sectional area")

    # kimimaro keys by label; the mask is binary so there is one, but join defensively.
    verts, edges, radii, areas, offset = [], [], [], [], 0
    for skel in skels.values():
        if not len(skel.vertices):
            continue
        verts.append(np.asarray(skel.vertices, dtype=np.float64))
        edges.append(np.asarray(skel.edges, dtype=np.int64).reshape(-1, 2) + offset)
        r = getattr(skel, "radii", None)
        radii.append(
            np.asarray(r, dtype=np.float64) if r is not None
            else np.zeros(len(skel.vertices))
        )
        a = getattr(skel, "cross_sectional_area", None)
        areas.append(np.asarray(a, dtype=np.float64) if a is not None
                     else np.full(len(skel.vertices), np.nan))
        offset += len(skel.vertices)
    if not verts:
        from .adapter import Triple

        return Triple({}, {}, []), "kimimaro postprocess removed every vertex"

    triple = edges_to_triple(
        np.vstack(verts), np.vstack(edges), np.concatenate(radii),
        origin_um=frame.seg_origin if origin_um is None else origin_um,
        point_attrs={KIMIMARO_AREA: np.concatenate(areas)} if cross_section else None,
    )
    detail = f"{offset:,} TEASAR vertices, scale={scale}, const={const_um} um"
    if steps:
        detail += "; " + ", ".join(steps)
    return triple, detail


@contextmanager
def _osteoid_uint32_edges(kimimaro):
    """Run kimimaro's post-processing under the pinned osteoid.

    Two places assume ``uint32`` edges, and osteoid 0.6.0 -- pinned because later
    releases assume NumPy 2 promotion -- carries ``int64``:

    * kimimaro 5.8's ``create_distance_graph`` is typed for ``uint32`` and refuses them;
    * ``Skeleton.components()`` gets ``uint64`` index arrays back for ``int64`` edges,
      and under NumPy 1.26 ``uint64 + 1`` is a float, so a skeleton with more than one
      component cannot be split at all -- which ``postprocess`` and
      ``cross_sectional_area`` both do.

    Both are looked up at call time, so each is wrapped for the duration of the block
    only, casting the skeleton's edges to ``uint32`` first.
    """
    from osteoid import Skeleton

    def as_uint32(skeleton):
        skeleton.edges = np.asarray(skeleton.edges).astype(np.uint32)

    tricks = kimimaro.skeletontricks
    original_graph = tricks.create_distance_graph
    original_components = Skeleton.components

    def graph(skeleton):
        as_uint32(skeleton)
        return original_graph(skeleton)

    def components(self):
        as_uint32(self)
        return original_components(self)

    tricks.create_distance_graph = graph
    Skeleton.components = components
    try:
        yield
    finally:
        tricks.create_distance_graph = original_graph
        Skeleton.components = original_components


def _teasar_readme(volume, frame, spacing, **params) -> tuple:
    """TEASAR as the kimimaro README runs it: README parameters, postprocess, areas.

    The README's ``scale=1.5, const=300`` are its own defaults (in nm on connectomics
    data); here `const` stays in micrometres and is the parameter a sweep should move.
    """
    settings = dict(scale=1.5, const_um=TEASAR_CONST_UM, pdrf_scale=100000,
                    pdrf_exponent=4, postprocess=True, cross_section=True,
                    smoothing_window=5)
    settings.update(params)
    triple, detail = _teasar(volume, frame, spacing, **settings)
    return triple, "README pipeline: " + detail


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


def _geodesic_scoop(volume, frame, spacing, *, origin_um=None, prune=True,
                    **params) -> tuple:
    """Wave-front scooping: centroids of geodesic shells from a root. Keeps loops.

    With `prune`, ``skeleton_optimise.prune_spurs`` follows, with this volume's box as
    the lattice edge so a vessel cut by an ROI face is not pruned as a spur.
    """
    from .geodesic_scoop import extract
    from .graphmodel import EditableGraph
    from .skeleton_optimise import BOUNDARY_MARGIN_UM, prune_spurs

    vol = np.asarray(volume) > 0
    result = extract(vol, spacing, **params)
    origin = np.asarray(frame.seg_origin if origin_um is None else origin_um, dtype=np.float64)
    triple = edges_to_triple(result.coordinates_zyx[:, ::-1] * spacing, result.edges,
                             result.radii_um, origin_um=origin)
    if not len(result.edges) and len(result.coordinates_zyx):
        xyz = result.coordinates_zyx[0, ::-1] * spacing + origin
        triple.nodes[0] = (*map(float, xyz), 0)
    r = result.report
    detail = (f"{r['shells']} shells, {r['clusters']} clusters, {r['loops']} loop(s) "
              f"closed, {r['pruned']} micro-spur(s) pruned")
    if prune and triple.segments:
        hi = origin + (np.array(vol.shape[::-1]) - 1) * spacing
        bbox = np.ravel(np.column_stack([origin, hi]))
        graph = EditableGraph(triple)
        report = prune_spurs(graph, bbox_um=bbox,
                             boundary_margin_um=max(BOUNDARY_MARGIN_UM, 2 * float(np.max(spacing))))
        triple = graph.triple
        detail += f", then {report.n_removed} spur(s) by prune_spurs"
    return triple, detail + "; radii are max-EDT placeholders"


def _jin_mgf(volume, frame, spacing, *, origin_um=None, root_zyx=None,
             max_voxels=2_000_000, engine="auto", fair=False, fair_smooth_radii=None,
             fair_emergence_radii=None, **params) -> tuple:
    """Jin MCP with collapse-aware medialness and cross-section marking (opt-in).

    `engine` as for ``jin-mcp``: the sparse engine beyond `max_voxels`. With `fair`,
    the voxel path is smoothed vessel by vessel through its junctions and kept inside
    this volume (:mod:`~.centreline_fair`); `fair_smooth_radii` and
    `fair_emergence_radii` override that module's defaults.
    """
    from .jin_mgf import extract

    if not is_isotropic(spacing):
        raise ValueError('Jin MGF currently requires isotropic voxel spacing')
    chosen = _jin_engine(volume, max_voxels, engine)
    result = extract(np.asarray(volume) > 0, root_zyx=root_zyx, max_voxels=max_voxels,
                     engine=chosen, verbose=chosen == "sparse", **params)
    origin = frame.seg_origin if origin_um is None else origin_um
    triple = edges_to_triple(result.coordinates_zyx[:, ::-1] * spacing,
                             result.edges, result.fdt * spacing[0], origin_um=origin)
    if not len(result.edges):
        xyz = result.coordinates_zyx[0, ::-1] * spacing + origin
        triple.nodes[0] = (*map(float, xyz), 0)
    r = result.report
    detail = (f"Jin MGF [{r['medialness']}/{r['marking']}]: {len(r['rounds'])} rounds, "
              f"{r['sheet_voxels']} sheet voxels, {r['unmarked_voxels']} unmarked; "
              "FDT thickness placeholders")
    if fair and triple.segments:
        from . import centreline_fair as cf
        from .graphmodel import EditableGraph

        graph = EditableGraph(triple)
        roots = ()
        if r.get("root_zyx") is not None:
            # The extraction root is a path end, so it is a node; take the nearest one.
            xyz = np.asarray(r["root_zyx"], dtype=np.float64)[::-1] * spacing + origin
            ids = list(graph.nodes)
            at = np.array([graph.nodes[n][:3] for n in ids])
            roots = (ids[int(np.argmin(np.linalg.norm(at - xyz, axis=1)))],)
        options = {k: v for k, v in (("smooth_radii", fair_smooth_radii),
                                     ("emergence_radii", fair_emergence_radii))
                   if v is not None}
        report = cf.fair(graph, spacing_um=float(spacing[0]), roots=roots,
                         inside=cf.volume_inside(volume, origin, spacing), **options)
        triple = graph.triple
        detail += (f"; faired {report.vessels} vessels, turn median "
                   f"{report.before['turn_median_deg']:.1f} -> "
                   f"{report.after['turn_median_deg']:.1f} deg")
    return triple, detail


_BACKENDS = {"lee": _lee, "teasar": _teasar, "teasar-readme": _teasar_readme,
             "amira": _amira, "jin-mcp": _jin_mcp,
             "jin-mcp-centroid": _jin_mcp_centroid, "vmtk": _vmtk,
             "geodesic-scoop": _geodesic_scoop, "jin-mgf": _jin_mgf}


def _literal(text: str):
    low = text.strip().lower()
    if low in ("true", "false"):
        return low == "true"
    if low in ("none", "null"):
        return None
    for cast in (int, float):
        try:
            return cast(text)
        except ValueError:
            pass
    return text.strip()


def parse_candidate(spec: str) -> tuple[str, dict]:
    """``"teasar[scale=4;const_um=100]"`` -> ``("teasar", {"scale": 4, "const_um": 100})``.

    Lets one ``--algorithms`` list hold a parameter grid. Semicolons separate the
    parameters because commas already separate the candidates.
    """
    spec = spec.strip()
    if "[" not in spec:
        return spec, {}
    if not spec.endswith("]"):
        raise ValueError(f"candidate {spec!r}: expected name[key=value;...]")
    name, body = spec[:-1].split("[", 1)
    params = {}
    for item in filter(None, (s.strip() for s in body.split(";"))):
        if "=" not in item:
            raise ValueError(f"candidate {spec!r}: {item!r} is not key=value")
        key, value = item.split("=", 1)
        params[key.strip().replace("-", "_")] = _literal(value)
    return name.strip(), params


def candidate_label(name: str, params: dict) -> str:
    """A file-name-safe label: ``teasar_scale-4_const_um-100``."""
    if not params:
        return name
    parts = [f"{k}-{v:g}" if isinstance(v, float) else f"{k}-{v}" for k, v in params.items()]
    return "_".join([name, *parts]).replace("/", "-").replace("\\", "-")


def split_candidates(text: str) -> list[str]:
    """Split a comma-separated candidate list without splitting inside brackets."""
    out, depth, cur = [], 0, []
    for ch in text:
        if ch == "[":
            depth += 1
        elif ch == "]":
            depth -= 1
        if ch == "," and depth == 0:
            out.append("".join(cur).strip())
            cur = []
        else:
            cur.append(ch)
    out.append("".join(cur).strip())
    return [s for s in out if s]


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


#: Backends that take an extraction root, as ``root_zyx`` in the tree crop's voxels.
#: Without one, Jin roots at the maximum-FDT voxel -- inside a wide inlet, short of
#: its cut face -- and traces back out to the face's rim: on LADAF-2021-17 the left
#: tree's inlet came out as a Y, its two tips 1.8 and 2.2 mm from the true inlet.
ROOTED = ("jin-mcp", "jin-mcp-centroid", "jin-mgf", "geodesic-scoop")
#: How far a given root may be from the tree's lumen and still be snapped onto it. An
#: inlet picked on a graph sits at the cut face, a voxel or so outside the mask.
ROOT_SNAP_VOXELS = 12


def root_for_part(part, frame, roots_um, snap_voxels=ROOT_SNAP_VOXELS):
    """The first of `roots_um` (world xyz, um) on `part`'s lumen, as crop ``zyx``.

    Each point goes through `frame` onto the decoded grid, then into the part's box,
    and is snapped to the nearest lumen voxel within `snap_voxels`. ``None`` when no
    point lands on this tree.
    """
    if roots_um is None or not len(roots_um):
        return None
    ijk = frame.um_to_seg_index(np.asarray(roots_um, dtype=np.float64).reshape(-1, 3))
    shape = np.array(part.volume.shape)
    for p in ijk[:, ::-1] - np.asarray(part.origin_zyx):
        lo = np.maximum(p - snap_voxels, 0)
        hi = np.minimum(p + snap_voxels + 1, shape)
        if np.any(hi <= lo):
            continue
        sub = np.argwhere(part.volume[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]])
        if not len(sub):
            continue
        sub = sub + lo
        d = np.linalg.norm(sub - p, axis=1)
        k = int(np.argmin(d))
        if d[k] <= snap_voxels:
            return tuple(int(v) for v in sub[k])
    return None


def _rooted_params(name, part, frame, roots_um, params):
    """`params` plus this tree's ``root_zyx``, when one was given and `name` takes one."""
    if name not in ROOTED or not roots_um:
        return params, ""
    root = root_for_part(part, frame, roots_um)
    if root is None:
        return params, " (no given root on this tree; automatic root)"
    return {**params, "root_zyx": root}, f" (root at crop voxel {root})"


def _tree_job(path: str, origin_um, frame, algorithm: str, params: dict):
    """One (candidate, tree) pair in a worker process. Module-level so it pickles."""
    vol = np.load(path, mmap_mode="r")
    spacing = np.asarray(frame.seg_spacing, dtype=np.float64)
    t0 = time.time()
    triple, detail = _BACKENDS[algorithm](np.asarray(vol), frame, spacing,
                                          origin_um=np.asarray(origin_um), **params)
    return triple, detail, time.time() - t0


def skeletonise_many(specs, volume, frame, *, workers: int, scratch_dir,
                     min_component_voxels: int = 0, max_trees: int | None = None,
                     connectivity: int = 3, verbose: bool = False, materials=None,
                     roots_um=None) -> dict:
    """Every ``(candidate, tree)`` pair at once, each in its own process.

    `specs` is ``[(label, algorithm, params), ...]``. Returns ``{label: candidate}``,
    or ``{label: exception}`` where a backend raised, for the caller to report as
    :func:`skeletonise_per_component` would.

    Why processes, and why per tree: on a whole heart the backends are single-threaded
    and independent -- kimimaro parallelises only across labels, and each call here
    has one -- so the two coronaries and every grid point can run side by side. The
    mask is split **once** and each tree's crop saved to `scratch_dir`, where workers
    memory-map it, instead of every worker re-decoding and re-labelling 30 GB. Jobs
    start largest tree first so the longest ones are not left until last.

    Memory is the limit, not cores: a TEASAR job on a LADAF-2021-17 tree box holds
    several times the box size, so choose `workers` from that, not from the core count.
    The result is identical to :func:`skeletonise_per_component`, candidate by
    candidate.
    """
    import multiprocessing
    from concurrent.futures import ProcessPoolExecutor, as_completed
    from concurrent.futures.process import BrokenProcessPool
    from pathlib import Path

    from . import components as comp

    unknown =[alg for _, alg, _ in specs if alg not in _BACKENDS or alg == "amira"]
    if unknown:
        raise ValueError(f"cannot run {unknown} per tree in parallel")
    parts, _stats = comp.split_components(
        volume, frame, min_voxels=min_component_voxels, max_trees=max_trees,
        connectivity=connectivity, verbose=verbose, materials=materials,
    )
    scratch = Path(scratch_dir)
    scratch.mkdir(parents=True, exist_ok=True)
    paths = {}
    for part in parts:
        paths[part.index] = scratch / f"tree_{part.index}.npy"
        np.save(paths[part.index], np.ascontiguousarray(part.volume))

    # Each tree's extraction root is resolved here, once, so a worker needs only its
    # params; `roots_um` are world positions (see `root_for_part`).
    rooted = {(label, part.index): _rooted_params(alg, part, frame, roots_um, params)
              for label, alg, params in specs for part in parts}
    jobs = sorted(((label, alg, rooted[(label, part.index)][0], part)
                   for label, alg, params in specs for part in parts),
                  key=lambda j: -j[3].voxels)
    pieces: dict = {label: [] for label, _, _ in specs}
    notes: dict = {label: [] for label, _, _ in specs}
    seconds: dict = {label: 0.0 for label, _, _ in specs}
    failed: dict = {}
    t0 = time.time()
    # A worker killed outright -- out of memory, as four TEASAR jobs on the
    # LADAF-2021-17 tree boxes were -- breaks the whole pool, and every job still in
    # it fails with BrokenProcessPool, finished work elsewhere notwithstanding. Those
    # jobs are run again on half as many workers, down to one; only a job that breaks
    # a one-worker pool, or raises on its own, fails.
    pending, width = jobs, max(int(workers), 1)
    try:
        while pending:
            broken = []
            with ProcessPoolExecutor(max_workers=min(width, len(pending)),
                                     mp_context=multiprocessing.get_context("spawn")) as pool:
                futures = {
                    pool.submit(_tree_job, str(paths[job[3].index]),
                                np.asarray(job[3].origin_um, dtype=np.float64), frame,
                                job[1], job[2]): job
                    for job in pending
                }
                for future in as_completed(futures):
                    job = futures[future]
                    label, part = job[0], job[3]
                    try:
                        triple, detail, took = future.result()
                    except BrokenProcessPool as exc:
                        if width > 1:
                            broken.append(job)
                            continue
                        failed.setdefault(label, exc)
                        if verbose:
                            print(f"  {label} / tree {part.index}: failed: {exc}", flush=True)
                        continue
                    except Exception as exc:  # reported per candidate, like the serial path
                        failed.setdefault(label, exc)
                        if verbose:
                            print(f"  {label} / tree {part.index}: failed: {exc}", flush=True)
                        continue
                    pieces[label].append((part.index, triple))
                    notes[label].append((part.index, f"tree {part.index}: {detail}"
                                                     f"{rooted[(label, part.index)][1]}"))
                    seconds[label] += took
                    if verbose:
                        print(f"  {label} / tree {part.index}: {len(triple.segments)} segments "
                              f"in {took:.0f}s ({time.time() - t0:.0f}s elapsed)", flush=True)
            pending = sorted(broken, key=lambda j: -j[3].voxels)
            if pending:
                width = max(width // 2, 1)
                if verbose:
                    print(f"  a worker died (out of memory?); rerunning {len(pending)} job(s) "
                          f"on {width} worker(s)", flush=True)
    finally:
        for path in paths.values():
            path.unlink(missing_ok=True)

    out: dict = {}
    for label, _alg, params in specs:
        if label in failed:
            out[label] = failed[label]
            continue
        merged = comp.merge_triples(sorted(pieces[label], key=lambda p: p[0]))
        out[label] = SkeletonCandidate(
            name=label, triple=merged, params=dict(params), seconds=seconds[label],
            detail="; ".join(d for _, d in sorted(notes[label])), trees=len(parts),
            tree_voxels=tuple(p.voxels for p in parts),
        )
    return out


def skeletonise_per_component(name: str, volume, frame, *, min_component_voxels: int = 0,
                              max_trees: int | None = None, connectivity: int = 3,
                              verbose: bool = False, materials=None, roots_um=None,
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
        tree_params, root_note = _rooted_params(name, part, frame, roots_um, params)
        triple, detail = _BACKENDS[name](
            part.volume, frame, spacing, origin_um=part.origin_um, **tree_params
        )
        pieces.append((part.index, triple))
        notes.append(f"tree {part.index}: {detail}{root_note}")

    merged = comp.merge_triples(pieces)
    return SkeletonCandidate(
        name=name, triple=merged, params=dict(params), seconds=time.time() - t0,
        detail="; ".join(notes), trees=len(parts),
        tree_voxels=tuple(p.voxels for p in parts),
    )
