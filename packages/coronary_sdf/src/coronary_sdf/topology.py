"""Tree topology operations.

The reconnection / contraction / component passes now live in
:mod:`coronary_sdf.centreline_reconnection` and are re-exported here for
backward compatibility:

- Connected-component splitting (``split_by_graph``).
- Degree-2 contraction (``merge_degree2_segments``).
- Split-multifurcation contraction (``merge_split_multifurcations``).
- Near-coincident node merging (``node_id_canon_map``).

Retained here:

- NetworkX tree builder used by region tagging and junction labelling.
- Topology-aware cross-section junction labelling
  (``label_capsules_by_cross_section``) with the patch-84 node-proximity
  gate.
- Directed topology (parent / child / sibling masks).
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.spatial import KDTree

from .config import runtime_config as config

# Reconnection / contraction / component passes moved to their own module;
# re-exported so existing ``from .topology import ...`` callers keep working.
from .centreline_reconnection import (
    find_connected_components,
    split_by_graph,
    merge_degree2_segments,
    merge_split_multifurcations,
    node_id_canon_map,
    split_unsampled_jumps,
    drop_small_components,
)


# ── NetworkX tree builder (for region tagging / junction labelling) ───────────


def build_nx_tree(
    nodes: dict[int, tuple],
    points: dict[int, tuple],
    segments: list[dict[str, Any]],
    mm_scale: float = 1.0 / 1000.0,
):
    """Return a ``networkx.Graph`` with edge attributes ``points``, ``radii``,
    ``strahler``. Centerlines are passed through :func:`coronary_sdf.smoothing.
    smooth_centerline` so the surface and the per-face labels agree.
    """
    try:
        import networkx as nx
    except ImportError as exc:
        raise ImportError(
            "networkx is required; install with `pip install networkx`"
        ) from exc

    g = nx.Graph()
    for nid, nd in nodes.items():
        x, y, z = nd[0], nd[1], nd[2]
        g.add_node(
            nid,
            pos=np.array([x * mm_scale, y * mm_scale, z * mm_scale], dtype=np.float64),
        )

    for seg in segments:
        pids = seg["point_ids"]
        if len(pids) < 2:
            continue
        coords = np.array(
            [
                [points[p][0] * mm_scale, points[p][1] * mm_scale, points[p][2] * mm_scale]
                for p in pids
            ],
            dtype=np.float64,
        )
        radii = np.array(
            [points[p][3] * mm_scale * config.RADIUS_SCALE for p in pids], dtype=np.float64
        )
        # Coordinates were already smoothed by the pipeline.  Re-smoothing here
        # changes region labels relative to the surface and is especially wrong
        # for the graph-aware constrained method.
        g.add_edge(
            seg["node1"],
            seg["node2"],
            points=coords,
            radii=radii,
            strahler=int(seg.get("strahler", 0)),
        )
    return g


# ── Topology-aware cross-section junction labelling (patch 84) ───────────────


def label_centreline_topology_aware(tree) -> tuple[np.ndarray, np.ndarray]:
    """Per-centerline-point junction labels, restricted to topology-adjacent
    edges + the patch-84 node-proximity gate.

    Returns ``(is_junction[N], all_pts[N, 3])`` where ``all_pts`` is the
    concatenation of every edge's centerline points (in mm).
    """
    edges_list = list(tree.edges(data=True))
    n_edges = len(edges_list)
    if n_edges == 0:
        return np.empty(0, dtype=bool), np.empty((0, 3), dtype=np.float64)

    # Edge adjacency: ej is adjacent to ei if they share a node.
    node_to_edges: dict[Any, set[int]] = {}
    for ei, (u, v, _) in enumerate(edges_list):
        node_to_edges.setdefault(u, set()).add(ei)
        node_to_edges.setdefault(v, set()).add(ei)
    edge_adj = [set() for _ in range(n_edges)]
    for _nid, eset in node_to_edges.items():
        elist = list(eset)
        for ei in elist:
            for ej in elist:
                if ej != ei:
                    edge_adj[ei].add(ej)

    parts_pts: list[np.ndarray] = []
    parts_r: list[np.ndarray] = []
    parts_tan: list[np.ndarray] = []
    parts_eidx: list[np.ndarray] = []
    for ei, (_, _, d) in enumerate(edges_list):
        pts = np.asarray(d["points"], dtype=np.float64)
        r = np.asarray(d["radii"], dtype=np.float64)
        n = len(pts)
        tans = np.empty_like(pts)
        if n == 1:
            tans[0] = np.array([0.0, 0.0, 1.0])
        else:
            tans[0] = pts[1] - pts[0]
            tans[-1] = pts[-1] - pts[-2]
            if n > 2:
                tans[1:-1] = pts[2:] - pts[:-2]
        norms = np.linalg.norm(tans, axis=1, keepdims=True)
        tans /= np.where(norms < 1e-12, 1.0, norms)
        parts_pts.append(pts)
        parts_r.append(r)
        parts_tan.append(tans)
        parts_eidx.append(np.full(n, ei, dtype=np.int32))

    if not parts_pts:
        return np.empty(0, dtype=bool), np.empty((0, 3), dtype=np.float64)

    all_pts = np.vstack(parts_pts)
    all_r = np.concatenate(parts_r)
    all_tans = np.vstack(parts_tan)
    pt_edge_idx = np.concatenate(parts_eidx)

    # Patch 84: per-point distance to nearer endpoint of its own edge.
    parts_to_node: list[np.ndarray] = []
    for _pts in parts_pts:
        if len(_pts) == 1:
            parts_to_node.append(np.zeros(1, dtype=np.float64))
            continue
        _du = np.linalg.norm(_pts - _pts[0], axis=1)
        _dv = np.linalg.norm(_pts - _pts[-1], axis=1)
        parts_to_node.append(np.minimum(_du, _dv))
    all_to_node = np.concatenate(parts_to_node)
    N = len(all_pts)
    max_r = float(all_r.max())

    kd = KDTree(all_pts)
    search_radii = np.sqrt(all_r**2 + max_r**2)
    try:
        candidates_all = kd.query_ball_point(all_pts, search_radii)
    except TypeError:
        candidates_all = kd.query_ball_point(all_pts, float(search_radii.max()))

    is_junction = np.zeros(N, dtype=bool)
    for pi in range(N):
        ei = int(pt_edge_idx[pi])
        adj_e = edge_adj[ei]
        if not adj_e:
            continue
        r_i = all_r[pi]
        if all_to_node[pi] >= config.XS_JUNC_NODE_PROXIMITY_FACTOR * r_i:
            continue
        t_i = all_tans[pi]
        p_i = all_pts[pi]
        for pj in candidates_all[pi]:
            ej = int(pt_edge_idx[pj])
            if ej == ei or ej not in adj_e:
                continue
            diff = all_pts[pj] - p_i
            d_j = float(diff.dot(t_i))
            if abs(d_j) >= all_r[pj]:
                continue
            if float(np.linalg.norm(diff - d_j * t_i)) < r_i:
                is_junction[pi] = True
                break
    return is_junction, all_pts


def label_capsules_by_cross_section(
    cap_midpoints: np.ndarray,
    nodes: dict[int, tuple],
    points: dict[int, tuple],
    segments: list[dict[str, Any]],
) -> np.ndarray:
    """Per-capsule ``is_junction`` flag via topology-aware cross-section test."""
    try:
        tree = build_nx_tree(nodes, points, segments)
    except ImportError:
        return np.zeros(len(cap_midpoints), dtype=bool)
    if tree.number_of_edges() == 0:
        return np.zeros(len(cap_midpoints), dtype=bool)
    is_junc_pt, all_pts = label_centreline_topology_aware(tree)
    if len(all_pts) == 0:
        return np.zeros(len(cap_midpoints), dtype=bool)
    cl_kd = KDTree(all_pts)
    _, nn_idx = cl_kd.query(cap_midpoints)
    return is_junc_pt[nn_idx].astype(bool, copy=False)


# ── Directed topology (parent / child / sibling masks) ───────────────────────


def build_directed_topology(
    segments: list[dict[str, Any]],
    node_to_segs: dict[int, set[int]],
    root_pref: set[int] | None = None,
) -> dict[str, np.ndarray]:
    """Per-segment parent assignment + (n_segs, n_segs) topology masks.

    Roots each component at its trunk — a caller-forced segment (``root_pref``)
    wins, else the highest-Strahler segment, breaking ties toward a segment that
    has a free (degree-1) end (the ostium) and then toward the thickest
    (``MeanRadius``) — then BFS outward through the shared-node adjacency. Each
    segment's parent is the neighbour it was discovered from. Siblings are
    segments that share the same parent (excluding the root). Forest-friendly:
    disconnected components are processed independently, each rooted at its own
    trunk.

    The free-end tie-break matters: several segments can tie at the max Strahler
    order, and a plain argmax can land on an *internal* one (no degree-1 node),
    which leaves the tree with no proper inlet.

    ``root_pref`` (optional) is a set of segment indices to force as their
    component's root, overriding the Strahler/free-end/radius tie-break. A forced
    index only affects the component it belongs to.

    Returns a dict with arrays sized over ``n_segs``::

        parent_seg_idx : (n_segs,)        int32 — parent index, -1 for roots
        is_parent      : (n_segs, n_segs) bool  — j is i's parent
        is_child       : (n_segs, n_segs) bool  — j is i's child
        is_sibling     : (n_segs, n_segs) bool  — i and j share a parent
    """
    n = len(segments)
    parent_idx = np.full(n, -1, dtype=np.int32)
    if n == 0:
        return {
            "parent_seg_idx": parent_idx,
            "is_parent": np.zeros((0, 0), dtype=bool),
            "is_child": np.zeros((0, 0), dtype=bool),
            "is_sibling": np.zeros((0, 0), dtype=bool),
        }

    seg_strahler = np.array(
        [int(seg.get("strahler", 0)) for seg in segments], dtype=np.int32
    )
    pref = {int(r) for r in root_pref} if root_pref else set()

    def _has_free_end(si: int) -> bool:
        s = segments[si]
        return any(len(node_to_segs.get(nid, ())) == 1
                   for nid in (s["node1"], s["node2"]))

    visited = np.zeros(n, dtype=bool)
    # Process every connected component independently. Root each component at its
    # trunk: a caller-forced segment (root_pref) wins, else highest Strahler,
    # then prefer a free (degree-1) end — the ostium — over an equal-order
    # internal segment, then the thickest (MeanRadius, µm). Without the free-end
    # tie-break, argmax can pick an internal max-Strahler segment (no degree-1
    # node), leaving the tree with no proper inlet.
    while not visited.all():
        candidates = np.where(~visited)[0]
        root = int(max(
            candidates,
            key=lambda si: (si in pref,
                            int(seg_strahler[si]),
                            _has_free_end(si),
                            int(segments[si].get("MeanRadius", 0))),
        ))
        visited[root] = True
        parent_idx[root] = -1
        queue: list[int] = [root]
        while queue:
            si = queue.pop(0)
            seg = segments[si]
            for nid in (seg["node1"], seg["node2"]):
                for sj in node_to_segs.get(nid, set()):
                    if sj == si or visited[sj]:
                        continue
                    visited[sj] = True
                    parent_idx[sj] = si
                    queue.append(sj)

    is_parent = np.zeros((n, n), dtype=bool)
    is_child = np.zeros((n, n), dtype=bool)
    for i in range(n):
        p = int(parent_idx[i])
        if p >= 0:
            is_parent[i, p] = True
            is_child[p, i] = True

    has_parent = parent_idx >= 0
    same_parent = (parent_idx[:, None] == parent_idx[None, :]) & (
        has_parent[:, None] & has_parent[None, :]
    )
    np.fill_diagonal(same_parent, False)
    is_sibling = same_parent

    return {
        "parent_seg_idx": parent_idx,
        "is_parent": is_parent,
        "is_child": is_child,
        "is_sibling": is_sibling,
    }


__all__ = [
    "find_connected_components",
    "split_by_graph",
    "merge_degree2_segments",
    "merge_split_multifurcations",
    "node_id_canon_map",
    "split_unsampled_jumps",
    "drop_small_components",
    "build_nx_tree",
    "label_centreline_topology_aware",
    "label_capsules_by_cross_section",
    "build_directed_topology",
]
