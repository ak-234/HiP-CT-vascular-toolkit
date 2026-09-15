"""Per-face region tagging + VTK PolyData writer.

Labels each triangle of the output surface with:

- ``region_id``    -- unique per junction / branch run
- ``is_junction``  -- 1 if the face is in a junction zone
- ``strahler``     -- Strahler order of the owning edge
- ``bif_level``    -- generation depth from the root
- ``composite_id`` -- packed strahler/level/junction for fast colouring

Uses the legacy (un-gated) cross-section labelling -- per-face tagging
needs the looser definition to colour junction regions, distinct from
the carve gate's patch-84 narrow labelling.
"""

from __future__ import annotations

import logging
from collections import deque
from typing import Any

import numpy as np
import pyvista as pv
from scipy.spatial import KDTree

from .topology import build_nx_tree

_log = logging.getLogger(__name__)


# ── Legacy (un-gated) cross-section labelling ────────────────────────────────


def _label_centreline_by_cross_section(
    tree,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Per-centerline-point junction labels via cross-section disk test.

    Returns ``(is_junction[N], pt_edge_idx[N], all_pts[N, 3], all_r[N])``.
    Used by ``label_surface_topology``; deliberately wider than the
    patch-84 topology-aware variant so faces near a junction get the
    junction tag.
    """
    edges_list = list(tree.edges(data=True))
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
        empty3 = np.empty((0, 3), dtype=np.float64)
        return (
            np.empty(0, dtype=bool),
            np.empty(0, dtype=np.int32),
            empty3,
            np.empty(0, dtype=np.float64),
        )

    all_pts = np.vstack(parts_pts)
    all_r = np.concatenate(parts_r)
    all_tans = np.vstack(parts_tan)
    pt_edge_idx = np.concatenate(parts_eidx)
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
        r_i = all_r[pi]
        t_i = all_tans[pi]
        p_i = all_pts[pi]
        for pj in candidates_all[pi]:
            if int(pt_edge_idx[pj]) == ei:
                continue
            diff = all_pts[pj] - p_i
            d_j = float(diff.dot(t_i))
            if abs(d_j) >= all_r[pj]:
                continue
            if float(np.linalg.norm(diff - d_j * t_i)) < r_i:
                is_junction[pi] = True
                break
    return is_junction, pt_edge_idx, all_pts, all_r


# ── Bifurcation generation depth ─────────────────────────────────────────────


def compute_bifurcation_levels(tree, root: int | None = None):
    """Per-edge bifurcation depth (BFS from root, increments at deg>=3)."""
    import networkx as nx_local

    levels: dict[tuple[int, int], int] = {}
    if tree.number_of_edges() == 0:
        return levels, root

    def _canon(a, b):
        return (a, b) if a <= b else (b, a)

    def _edge_strahler(d):
        try:
            return int(d.get("strahler", 0))
        except (TypeError, ValueError):
            return 0

    for component in nx_local.connected_components(tree):
        sub_nodes = set(component)
        sub_edges = [
            (u, v, d) for u, v, d in tree.edges(data=True) if u in sub_nodes and v in sub_nodes
        ]
        if not sub_edges:
            continue
        if root is not None and root in sub_nodes:
            start = root
        else:
            max_s = max((_edge_strahler(d) for _, _, d in sub_edges), default=0)
            top_edges = [(u, v) for u, v, d in sub_edges if _edge_strahler(d) == max_s]
            top_nodes: set[int] = set()
            for u, v in top_edges:
                top_nodes.add(u)
                top_nodes.add(v)
            leaves = [n for n in top_nodes if tree.degree(n) == 1]
            if leaves:
                start = min(leaves)
            elif top_nodes:
                start = min(top_nodes)
            else:
                start = min(sub_nodes)
        if root is None:
            root = start

        visited_edges: set[tuple[int, int]] = set()
        node_level = {start: 0}
        q = deque([start])
        while q:
            u = q.popleft()
            for v in tree.neighbors(u):
                key = _canon(u, v)
                if key in visited_edges:
                    continue
                visited_edges.add(key)
                lvl = node_level[u]
                if tree.degree(u) >= 3 and u != start:
                    lvl += 1
                levels[key] = lvl
                if v not in node_level:
                    node_level[v] = lvl
                    q.append(v)
    return levels, root


# ── Per-face region tagging ──────────────────────────────────────────────────


def label_surface_topology(
    verts: np.ndarray,
    faces: np.ndarray,
    tree,
    root: int | None = None,
) -> dict[str, Any]:
    """Per-face arrays: region_id, is_junction, strahler, bif_level, composite_id."""
    is_junction_pt, pt_edge_idx, all_pts, _ = _label_centreline_by_cross_section(tree)
    edges_list = list(tree.edges(data=True))
    n_edges = len(edges_list)
    junc_nodes = sorted([n for n in tree.nodes() if tree.degree(n) >= 2])
    n_junc = len(junc_nodes)

    levels, used_root = compute_bifurcation_levels(tree, root=root)

    def _canon(a, b):
        return (a, b) if a <= b else (b, a)

    edge_strahler = np.zeros(max(n_edges, 1), dtype=np.int32)
    edge_level = np.zeros(max(n_edges, 1), dtype=np.int32)
    for ei, (u, v, d) in enumerate(edges_list):
        try:
            edge_strahler[ei] = int(d.get("strahler", 0))
        except (TypeError, ValueError):
            edge_strahler[ei] = 0
        edge_level[ei] = int(levels.get(_canon(u, v), 0))

    junc_strahler = np.zeros(n_junc, dtype=np.int32)
    junc_level = np.zeros(n_junc, dtype=np.int32)
    if n_junc > 0:
        edge_idx_by_pair = {_canon(u, v): ei for ei, (u, v, _) in enumerate(edges_list)}
        for ji, node in enumerate(junc_nodes):
            inc = []
            for nb in tree.neighbors(node):
                key = _canon(node, nb)
                if key in edge_idx_by_pair:
                    inc.append(edge_idx_by_pair[key])
            if inc:
                junc_strahler[ji] = int(edge_strahler[inc].max())
                junc_level[ji] = int(edge_level[inc].min())

    N = len(all_pts)
    pt_region = np.zeros(N, dtype=np.int32)
    pt_strahler = np.zeros(N, dtype=np.int32)
    pt_level = np.zeros(N, dtype=np.int32)
    pt_is_junc = is_junction_pt.astype(np.int32)

    if n_junc > 0 and np.any(is_junction_pt):
        junc_pos = np.array([tree.nodes[n]["pos"] for n in junc_nodes], dtype=np.float64)
        junc_kd = KDTree(junc_pos)
        _, nn = junc_kd.query(all_pts[is_junction_pt])
        pt_region[is_junction_pt] = (nn + 1).astype(np.int32)
        pt_strahler[is_junction_pt] = junc_strahler[nn]
        pt_level[is_junction_pt] = junc_level[nn]

    seg_counter = n_junc + 1
    offset = 0
    for ei, (_, _, d) in enumerate(edges_list):
        n_pts = len(d["points"])
        branch_here = ~is_junction_pt[offset:offset + n_pts]
        in_run = False
        run_id = 0
        for k in range(n_pts):
            gi = offset + k
            if branch_here[k]:
                if not in_run:
                    run_id = seg_counter
                    seg_counter += 1
                    in_run = True
                pt_region[gi] = run_id
                pt_strahler[gi] = edge_strahler[ei]
                pt_level[gi] = edge_level[ei]
            else:
                in_run = False
        offset += n_pts

    cl_kd = KDTree(all_pts)
    face_cents = verts[faces].mean(axis=1)
    _, nn_cl = cl_kd.query(face_cents)

    region_ids = pt_region[nn_cl]
    is_junction = pt_is_junc[nn_cl]
    strahler = pt_strahler[nn_cl]
    bif_level = pt_level[nn_cl]

    zero_mask = region_ids == 0
    if np.any(zero_mask) and np.any(~zero_mask):
        nz_idx = np.where(~zero_mask)[0]
        nz_kd = KDTree(face_cents[nz_idx])
        _, nn_nz = nz_kd.query(face_cents[zero_mask])
        donors = nz_idx[nn_nz]
        region_ids[zero_mask] = region_ids[donors]
        is_junction[zero_mask] = is_junction[donors]
        strahler[zero_mask] = strahler[donors]
        bif_level[zero_mask] = bif_level[donors]

    composite = (
        ((strahler.astype(np.uint32) & 0xFFFF) << 16)
        | ((bif_level.astype(np.uint32) & 0xFF) << 8)
        | (is_junction.astype(np.uint32) & 0x1)
    )

    return {
        "region_id": region_ids.astype(np.int32),
        "is_junction": is_junction.astype(np.int32),
        "strahler": strahler.astype(np.int32),
        "bif_level": bif_level.astype(np.int32),
        "composite_id": composite.astype(np.uint32),
        "root": used_root,
        "n_junc_nodes": int(n_junc),
    }


# ── VTK writers ──────────────────────────────────────────────────────────────


def _write_region_vtk_ascii(
    verts: np.ndarray, faces: np.ndarray, cell_arrays: dict[str, np.ndarray], path: str
) -> None:
    """ASCII VTK PolyData fallback writer (used when PyVista save fails)."""
    if isinstance(cell_arrays, np.ndarray):
        cell_arrays = {"region_id": cell_arrays}
    n_faces = len(faces)
    with open(path, "w") as fh:
        fh.write("# vtk DataFile Version 3.0\n")
        fh.write("Coronary surface regions\n")
        fh.write("ASCII\n")
        fh.write("DATASET POLYDATA\n")
        fh.write(f"POINTS {len(verts)} float\n")
        for v in verts:
            fh.write(f"{float(v[0]):.6f} {float(v[1]):.6f} {float(v[2]):.6f}\n")
        fh.write(f"POLYGONS {n_faces} {n_faces * 4}\n")
        for f in faces:
            fh.write(f"3 {int(f[0])} {int(f[1])} {int(f[2])}\n")
        fh.write(f"CELL_DATA {n_faces}\n")
        for name, arr in cell_arrays.items():
            arr = np.asarray(arr).reshape(-1)
            if np.issubdtype(arr.dtype, np.unsignedinteger):
                vtk_type, fmt = "unsigned_int", "{}\n"
            elif np.issubdtype(arr.dtype, np.integer):
                vtk_type, fmt = "int", "{}\n"
            else:
                vtk_type, fmt = "float", "{:.6f}\n"
            fh.write(f"SCALARS {name} {vtk_type} 1\n")
            fh.write("LOOKUP_TABLE default\n")
            for x in arr:
                fh.write(fmt.format(int(x) if vtk_type != "float" else float(x)))
    _log.info("Saved region VTK (ASCII): %s", path)


def generate_region_vtk(
    verts: np.ndarray,
    faces: np.ndarray,
    tree,
    output_path: str,
    root: int | None = None,
) -> dict[str, Any]:
    """Label every face with topology data and save the VTK file."""
    _log.info(
        "Labelling surface topology (region / is_junction / strahler / bif_level) ..."
    )
    topo = label_surface_topology(verts, faces, tree, root=root)

    n_junc = topo["n_junc_nodes"]
    n_edges = tree.number_of_edges()
    n_unique = int(len(np.unique(topo["region_id"])))
    s_max = int(topo["strahler"].max()) if len(topo["strahler"]) else 0
    l_max = int(topo["bif_level"].max()) if len(topo["bif_level"]) else 0
    print(
        f"  [REGION VTK] {n_junc} junction nodes, {n_edges} edges, "
        f"{n_unique} unique region IDs, strahler 0..{s_max}, bif_level 0..{l_max}, "
        f"root={topo['root']}"
    )

    cell_arrays = {
        "region_id": topo["region_id"],
        "is_junction": topo["is_junction"],
        "strahler": topo["strahler"],
        "bif_level": topo["bif_level"],
        "composite_id": topo["composite_id"],
    }
    try:
        faces_pv = np.column_stack(
            [np.full(len(faces), 3, dtype=np.int32), faces.astype(np.int32)]
        ).ravel()
        mesh = pv.PolyData(verts.astype(np.float32), faces_pv)
        for name, arr in cell_arrays.items():
            mesh.cell_data[name] = arr
        mesh.save(str(output_path))
        print(f"  [REGION VTK] saved {output_path}")
    except Exception as exc:
        print(f"  [REGION VTK] PyVista path failed ({exc}); falling back to ASCII")
        _write_region_vtk_ascii(verts, faces, cell_arrays, str(output_path))
    return topo


def _surface_to_verts_faces(surface: pv.PolyData) -> tuple[np.ndarray, np.ndarray]:
    """Extract ``(verts, faces)`` numpy arrays from a triangle ``pv.PolyData``."""
    verts = np.asarray(surface.points, dtype=np.float64)
    fa = np.asarray(surface.faces).reshape(-1)
    faces: list[list[int]] = []
    i = 0
    while i < len(fa):
        n = int(fa[i])
        if n == 3 and i + 3 < len(fa) + 1:
            faces.append([int(fa[i + 1]), int(fa[i + 2]), int(fa[i + 3])])
        i += n + 1
    return verts, np.asarray(faces, dtype=np.int32)


def emit_region_vtk_for_surface(
    surface: pv.PolyData,
    nodes: dict[int, tuple],
    points: dict[int, tuple],
    segments: list[dict[str, Any]],
    output_path: str,
    root: int | None = None,
) -> dict[str, Any] | None:
    """Build the NetworkX tree from the parsed XML and write a labelled VTK."""
    try:
        tree = build_nx_tree(nodes, points, segments)
    except ImportError as exc:
        print(f"  [REGION VTK][SKIP] {exc}")
        return None
    if tree.number_of_edges() == 0:
        print("  [REGION VTK][SKIP] no edges in tree")
        return None
    verts, faces = _surface_to_verts_faces(surface)
    if len(faces) == 0:
        print("  [REGION VTK][SKIP] mesh has no triangle faces")
        return None
    return generate_region_vtk(verts, faces, tree, output_path, root=root)


__all__ = [
    "compute_bifurcation_levels",
    "label_surface_topology",
    "generate_region_vtk",
    "emit_region_vtk_for_surface",
]
