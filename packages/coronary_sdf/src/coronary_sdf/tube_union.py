"""Watertight full-tree lumen via boolean union of per-branch tube solids.

An alternative to the volumetric SDF pipeline for spatial-graphs with a very wide
radius dynamic range (e.g. HiP-CT microvasculature), where a uniform SDF grid
either fuses or fragments the sub-voxel vessels. This mesher's cost is O(edges)
and scale-independent, so it captures the *entire* tree — including the finest
vessels — and merges the branches with a robust boolean union
(``meshlib.uniteManyMeshes``) into a single closed manifold.

Pipeline (:func:`run`)::

    parse_xml (voxel->µm scaling, field aliases)
      -> optional centerline / radius smoothing (self-intersection guard)
      -> per segment: closed capped tube solid (RMF sweep)
      -> small sphere solid at each degree>=3 node (junction overlap guard)
      -> meshlib.uniteManyMeshes  (watertight union)
      -> saveMesh (STL)

CLI::

    python -m coronary_sdf.tube_union <input.am|.xml> <output.stl> [options]
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from . import config
from .parse_amira import parse_xml
from .splines import prepare_segment_spline, compute_frenet_frame
from .smoothing import (
    smooth_segment_centerlines,
    smooth_segment_radii,
    limit_centerline_curvature,
    prune_bifurcation_shrink,
    prune_terminal_shrink,
    smooth_radius_transitions,
)


# ── Primitive solids (numpy verts/faces, outward-consistent winding) ──────────


def build_tube_solid(
    coords: np.ndarray, radii: np.ndarray, n_circle: int
) -> tuple[np.ndarray, np.ndarray] | None:
    """Closed, capped tube swept along ``coords`` with per-sample ``radii``.

    Uses a parallel-transport (RMF) frame so the ring does not twist. Returns
    ``(verts (V,3), faces (F,3) int)`` with outward-facing winding, or ``None``
    if the centerline is too short.
    """
    coords = np.asarray(coords, dtype=np.float64)
    radii = np.asarray(radii, dtype=np.float64)
    n = len(coords)
    if n < 2:
        return None

    tangents = np.gradient(coords, axis=0)
    nrm = np.linalg.norm(tangents, axis=1, keepdims=True)
    tangents = tangents / np.maximum(nrm, 1e-12)

    ang = np.linspace(0.0, 2.0 * np.pi, n_circle, endpoint=False)
    cos, sin = np.cos(ang), np.sin(ang)

    rings = np.empty((n, n_circle, 3), dtype=np.float64)
    prev_normal = None
    for i in range(n):
        _t, nrm_v, bnv = compute_frenet_frame(tangents[i], prev_normal)
        prev_normal = nrm_v
        rings[i] = coords[i] + radii[i] * (np.outer(cos, nrm_v) + np.outer(sin, bnv))

    verts = rings.reshape(n * n_circle, 3)
    c_start = n * n_circle          # start-cap centre vertex index
    c_end = n * n_circle + 1        # end-cap centre vertex index
    verts = np.vstack([verts, coords[0][None, :], coords[-1][None, :]])

    faces: list[tuple[int, int, int]] = []
    # Tube wall (outward-facing).
    for i in range(n - 1):
        base0 = i * n_circle
        base1 = (i + 1) * n_circle
        for j in range(n_circle):
            j1 = (j + 1) % n_circle
            v00, v01 = base0 + j, base0 + j1
            v10, v11 = base1 + j, base1 + j1
            faces.append((v00, v11, v10))
            faces.append((v00, v01, v11))
    # Start cap (normal -> -tangent).
    for j in range(n_circle):
        j1 = (j + 1) % n_circle
        faces.append((c_start, j1, j))
    # End cap (normal -> +tangent).
    base = (n - 1) * n_circle
    for j in range(n_circle):
        j1 = (j + 1) % n_circle
        faces.append((c_end, base + j, base + j1))

    return verts, np.asarray(faces, dtype=np.int32)


def sphere_mesh(center: np.ndarray, radius: float, res: int = 16):
    """Closed meshlib UV-sphere (guaranteed manifold) centred at ``center``."""
    import meshlib.mrmeshpy as mp
    m = mp.makeUVSphere(float(radius), res, res)
    m.transform(mp.AffineXf3f.translation(
        mp.Vector3f(float(center[0]), float(center[1]), float(center[2]))))
    return m


# ── meshlib bridge + union ────────────────────────────────────────────────────


def _to_meshlib(verts: np.ndarray, faces: np.ndarray):
    import meshlib.mrmeshnumpy as mn
    return mn.meshFromFacesVerts(
        np.ascontiguousarray(faces, dtype=np.int32),
        np.ascontiguousarray(verts, dtype=np.float64),
    )


def union_meshes(mesh_list: list) -> Any:
    """Robust boolean union of many closed solids via meshlib.uniteManyMeshes.

    ``uniteManyMeshes`` accepts a plain Python list of meshes across meshlib
    builds; the bound ``std::vector`` wrapper is named differently between
    versions (``std_vector_Mesh_const_ptr`` vs ``vectorConstMeshPtr``), so a
    list is the portable choice.
    """
    import meshlib.mrmeshpy as mp
    params = mp.UniteManyMeshesParams()
    params.mergeOnFail = True         # keep going if a local boolean fails
    params.useRandomShifts = True     # break coplanar/tangent degeneracies
    params.fixDegenerations = True
    return mp.uniteManyMeshes(list(mesh_list), params)


# ── Orchestrator ──────────────────────────────────────────────────────────────


def run(
    am_path: str | Path,
    out_path: str | Path,
    *,
    n_circle: int = 16,
    out_units: str = "mm",
    junction_spheres: bool = True,
    sphere_scale: float = 1.0,
    smooth_method: str = "taubin",
    smooth_iters: int = 20,
    remesh_edge: float = 0.0,
    remesh_relax: int = 10,
    smooth: bool = True,
    radius_taper: bool = False,
) -> str:
    """Build the watertight union lumen for ``am_path`` and save to ``out_path``.

    ``smooth_method`` selects the post-union facet smoothing: ``"taubin"``
    (pyvista vertex relaxation, lightweight), ``"remesh"`` (meshlib
    curvature-adaptive remesh + volume-preserving relax — refines + smooths,
    heavier), or ``"none"``. Loop subdivision is intentionally unsupported: the
    boolean union leaves non-manifold pinch edges that vtk's subdivider rejects.
    """
    print("=" * 60)
    print("CORONARY LUMEN -- boolean-union tube mesher")
    print(f"  voxel size: {config.INPUT_VOXEL_SIZE_UM} µm/unit | "
          f"n_circle={n_circle} | out units: {out_units} | "
          f"junction_spheres={junction_spheres} | smooth_method={smooth_method}")
    print("=" * 60)
    t0 = time.time()

    nodes, points, segments = parse_xml(am_path)

    if smooth:
        points, _ = smooth_segment_centerlines(nodes, points, segments)
        points, _ = limit_centerline_curvature(nodes, points, segments)
        points, _ = smooth_segment_radii(nodes, points, segments)
        if radius_taper:
            # Anatomical parent->daughter radius interpolation at junctions (mirrors
            # the SDF pipeline). Daughters flare toward the parent radius at the node
            # then taper to their own calibre, so overlapping tubes fuse into a smooth
            # tapered bifurcation.
            points, _ = prune_bifurcation_shrink(nodes, points, segments)
            points, _ = prune_terminal_shrink(nodes, points, segments)
            points, _ = smooth_radius_transitions(nodes, points, segments)
            # Re-limit curvature: flaring raised radii, which can push a bend below
            # its (now larger) radius of curvature -> self-intersecting tube that
            # breaks the boolean. Straighten those again after the flare.
            points, _ = limit_centerline_curvature(nodes, points, segments)

    # prepare_segment_spline returns coords/radii in mm (µm/1000). Convert to the
    # requested output unit (mm default; um keeps the parsed µm scale).
    unit_scale = 1.0 if out_units == "mm" else 1000.0

    solids: list = []
    node_end_radius_mm: dict[int, float] = {}
    node_deg: dict[int, int] = {}
    n_skipped = 0
    min_radius = float("inf")

    for seg in segments:
        for nid in (seg["node1"], seg["node2"]):
            node_deg[nid] = node_deg.get(nid, 0) + 1
        sp = prepare_segment_spline(seg, points, nodes)
        if sp is None:
            n_skipped += 1
            continue
        coords = sp["coords"] * unit_scale
        radii = sp["radii"] * unit_scale
        min_radius = min(min_radius, float(np.min(radii)))
        # Track max endpoint radius per node for the junction spheres.
        node_end_radius_mm[seg["node1"]] = max(
            node_end_radius_mm.get(seg["node1"], 0.0), float(radii[0]))
        node_end_radius_mm[seg["node2"]] = max(
            node_end_radius_mm.get(seg["node2"], 0.0), float(radii[-1]))
        tube = build_tube_solid(coords, radii, n_circle)
        if tube is None:
            n_skipped += 1
            continue
        solids.append(_to_meshlib(*tube))

    if junction_spheres:
        n_sph = 0
        for nid, deg in node_deg.items():
            if deg >= 3 and nid in nodes and nid in node_end_radius_mm:
                r = node_end_radius_mm[nid]
                if r <= 0.0:
                    continue
                center = np.asarray(nodes[nid][:3], dtype=np.float64) / 1000.0 * unit_scale
                # Oversize the sphere so every (radius-flared) tube pierces it
                # cleanly. Equal-radius tube/sphere surfaces are tangent and make
                # the boolean fail; sphere_scale > 1 restores clean intersections.
                solids.append(sphere_mesh(center, r * sphere_scale))
                n_sph += 1
        print(f"  junction spheres: {n_sph} (deg>=3 nodes, scale={sphere_scale})")

    print(f"  built {len(solids)} solids "
          f"({n_skipped} degenerate segments skipped) in {time.time()-t0:.1f}s")

    print("  boolean union (meshlib.uniteManyMeshes)...")
    t_u = time.time()
    union = union_meshes(solids)
    print(f"  union done in {time.time()-t_u:.1f}s")

    # meshlib-side smoothing runs on the union mesh before saving (no STL round-trip).
    if smooth_method == "remesh":
        edge = remesh_edge if remesh_edge > 0 else max(min_radius / 2.0, 1e-4)
        _remesh_smooth(union, edge, remesh_relax)

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    import meshlib.mrmeshpy as mp
    mp.saveMesh(union, str(out))
    print(f"\n[SAVE] {out}")
    _report(str(out))

    # pyvista-side smoothing runs on the saved STL (guarded against clobbering).
    if smooth_method == "taubin" and smooth_iters > 0:
        _smooth_mesh(str(out), smooth_iters)

    print(f"\nDone in {time.time()-t0:.1f}s")
    return str(out)


def _smooth_mesh(stl_path: str, n_iter: int) -> None:
    """Taubin-smooth the (watertight) union in place to round the polygonal tube
    cross-sections + junction spheres. Boolean-safe: only relaxes vertex positions
    on the finished mesh, so topology (and 0 open edges) is preserved. Taubin's
    alternating +/- passes avoid the shrinkage that plain Laplacian would inflict
    on thin tubes, and it tolerates the union's few non-manifold edges (unlike loop
    subdivision, which requires a strictly manifold input). Guarded: a degenerate
    result never overwrites the good union mesh already on disk."""
    try:
        import pyvista as pv
        print(f"\n[SMOOTH] Taubin x{n_iter} (rounding tube facets + junction spheres)...")
        t = time.time()
        m = pv.read(stl_path)
        ms = m.smooth_taubin(n_iter=n_iter, pass_band=0.1,
                             non_manifold_smoothing=True, boundary_smoothing=False)
        if ms.n_points == 0 or ms.n_cells == 0:
            print("  [SMOOTH][WARN] empty result; keeping the raw union mesh")
            return
        ms.save(stl_path)
        print(f"  -> {ms.n_points:,} verts, {ms.n_cells:,} faces in {time.time()-t:.1f}s")
        _report(stl_path)
    except Exception as e:
        print(f"  [SMOOTH][WARN] failed: {e}; keeping the raw union mesh")


def _remesh_smooth(mesh, target_edge: float, relax_iters: int) -> None:
    """Curvature-adaptive remesh + volume-preserving relax on the meshlib union,
    in place. Refines the faceted cross-sections/junctions (curvature-adaptive so
    straight tube runs stay coarse and the face count stays bounded), projecting
    onto the union surface, then smooths with ``relaxKeepVolume`` (not plain relax)
    so ~0.05 mm vessels don't collapse. Mirrors mesh_extract.mesh_from_sdf_meshlib.
    Stays in meshlib — tolerant of the union's non-manifold pinch edges (unlike vtk
    loop subdivision) and preserves closedness."""
    import meshlib.mrmeshpy as mp
    print(f"\n[REMESH] meshlib remesh (target edge {target_edge*1000:.1f} µm, "
          f"curvature-adaptive) + relaxKeepVolume x{relax_iters}...")
    t = time.time()
    rs = mp.RemeshSettings()
    rs.targetEdgeLen = float(target_edge)
    rs.projectOnOriginalMesh = True
    rs.useCurvature = True
    mp.remesh(mesh, rs)
    if relax_iters > 0:
        rp = mp.MeshRelaxParams()
        rp.iterations = int(relax_iters)
        rp.force = 0.4
        mp.relaxKeepVolume(mesh, rp)
    print(f"  -> {mesh.topology.numValidFaces():,} faces in {time.time()-t:.1f}s")


def _report(stl_path: str) -> None:
    """Manifold / open-edge / bbox / component diagnostics via pyvista."""
    try:
        import pyvista as pv
        m = pv.read(stl_path)
        b = m.bounds
        n_open = int(m.extract_feature_edges(
            boundary_edges=True, feature_edges=False,
            manifold_edges=False, non_manifold_edges=False).n_cells)
        print(f"  mesh: {m.n_points:,} verts, {m.n_cells:,} faces | "
              f"bbox {b[1]-b[0]:.1f} x {b[3]-b[2]:.1f} x {b[5]-b[4]:.1f}")
        print(f"  manifold={m.is_manifold}  open_edges={n_open}")
    except Exception as e:
        print(f"  [report][WARN] {e}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="python -m coronary_sdf.tube_union",
        description="Watertight full-tree lumen via boolean union of tube solids",
    )
    ap.add_argument("input", help="native Avizo ASCII .am or Excel-XML spatial-graph")
    ap.add_argument("output", help="destination .stl")
    ap.add_argument("--n-circle", type=int, default=16, help="circumference vertices per tube "
                    "(>~16 makes the boolean union fail; use --smooth-method to smooth instead)")
    ap.add_argument("--out-units", choices=("mm", "um"), default="mm")
    ap.add_argument("--no-junction-spheres", dest="junction_spheres",
                    action="store_false", help="disable the deg>=3 overlap spheres "
                    "(spheres are needed for watertight boolean fusion at junctions)")
    ap.add_argument("--sphere-scale", type=float, default=1.0,
                    help="junction sphere radius as a multiple of the max incident tube radius")
    ap.add_argument("--smooth-method", choices=("none", "taubin", "remesh"), default="taubin",
                    help="post-union facet smoothing: 'taubin' (pyvista vertex relax, light), "
                    "'remesh' (meshlib curvature-adaptive remesh + volume-preserving relax, "
                    "heavier/smoother), or 'none'. (Loop subdivision is unsupported: the union's "
                    "non-manifold pinch edges make vtk's subdivider reject the mesh.)")
    ap.add_argument("--smooth-iters", type=int, default=20,
                    help="Taubin iterations for --smooth-method taubin (~15-25)")
    ap.add_argument("--remesh-edge", type=float, default=0.0,
                    help="target edge length (mm) for --smooth-method remesh; 0 = auto (~min "
                    "vessel radius / 2). Smaller = smoother but far more faces")
    ap.add_argument("--remesh-relax", type=int, default=10,
                    help="volume-preserving relax iterations for --smooth-method remesh")
    ap.add_argument("--no-smooth", dest="smooth", action="store_false",
                    help="skip centerline/radius smoothing preprocessing")
    ap.add_argument("--radius-taper", dest="radius_taper", action="store_true",
                    help="parent->daughter junction radius interpolation (note: currently "
                    "worsens boolean watertightness; off by default)")
    args = ap.parse_args(argv)

    run(
        args.input, args.output,
        n_circle=args.n_circle,
        out_units=args.out_units,
        junction_spheres=args.junction_spheres,
        sphere_scale=args.sphere_scale,
        smooth_method=args.smooth_method,
        smooth_iters=args.smooth_iters,
        remesh_edge=args.remesh_edge,
        remesh_relax=args.remesh_relax,
        smooth=args.smooth,
        radius_taper=args.radius_taper,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
