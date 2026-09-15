"""Iso-surface extraction from an SDF volume.

Three backends:

- ``poisson``  -- Open3D Screened Poisson reconstruction sampled from
  zero-crossing edges with gradient-derived normals (patch 80).
- ``meshlib``  -- MeshLib dual contouring + volume-preserving relaxation.
- ``mc``       -- vtkFlyingEdges3D at SDF grid resolution (legacy).

Plus mesh-space post-processing the legacy script did inline:

- ``cut_non_adjacent_bridges`` -- delete triangles spanning
  topology-disconnected segments.
- ``create_flat_caps`` -- clip every terminal endpoint perpendicular to
  the vessel axis and fan-triangulate the resulting circular boundary.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np
import pyvista as pv

from .config import runtime_config as config


# ── MC fallback ──────────────────────────────────────────────────────────────


def fast_contour_zero(grid: pv.ImageData, scalars: str = "sdf") -> pv.PolyData:
    """Multi-threaded zero-iso surface via vtkFlyingEdges3D, drop-in for
    ``grid.contour(isosurfaces=[0.0])``."""
    try:
        return grid.contour(isosurfaces=[0.0], scalars=scalars, method="flying_edges")
    except TypeError:
        pass
    try:
        from vtkmodules.vtkFiltersCore import vtkFlyingEdges3D  # type: ignore
    except ImportError:
        try:
            from vtk import vtkFlyingEdges3D  # type: ignore
        except ImportError:
            return grid.contour(isosurfaces=[0.0], scalars=scalars)
    fe = vtkFlyingEdges3D()
    fe.SetInputData(grid)
    fe.SetValue(0, 0.0)
    fe.SetComputeNormals(False)
    fe.SetComputeGradients(False)
    fe.SetComputeScalars(False)
    if scalars is not None and scalars in grid.point_data:
        grid.set_active_scalars(scalars, preference="point")
    fe.Update()
    return pv.wrap(fe.GetOutput())


# ── Screened Poisson via Open3D (patch 80) ───────────────────────────────────


def mesh_from_sdf_poisson(
    sdf_vol: np.ndarray,
    bbox_min: np.ndarray,
    voxel_size: float,
    dims: np.ndarray,
    depth: int | None = None,
    density_quantile: float | None = None,
    band_sentinel: float = 10.0,
) -> pv.PolyData:
    """Step D Option 1 of the SDF rebuild prompt: replace MC with Open3D
    Screened Poisson sampled from zero-crossings of the SDF.

    Decouples mesh resolution from the SDF grid spacing -- small tubes
    can be reconstructed at sub-voxel resolution without the staircase
    MC produces at grid resolution.
    """
    try:
        import open3d as o3d
    except ImportError as exc:
        raise RuntimeError(
            "SDF_MESH_METHOD='poisson' requires open3d. Install with: pip install open3d"
        ) from exc

    if depth is None:
        depth = config.SDF_POISSON_DEPTH
    if density_quantile is None:
        density_quantile = config.SDF_POISSON_DENSITY_QUANTILE

    nx, ny, nz = int(dims[0]), int(dims[1]), int(dims[2])
    sdf = np.asarray(sdf_vol)
    bbox_min = np.asarray(bbox_min, dtype=np.float64)
    h = float(voxel_size)

    s = np.where(sdf >= 0.0, 1, -1).astype(np.int8)
    cross = [
        s[:-1, :, :] != s[1:, :, :],
        s[:, :-1, :] != s[:, 1:, :],
        s[:, :, :-1] != s[:, :, 1:],
    ]
    gx, gy, gz = np.gradient(sdf.astype(np.float32), h)
    grid_max = np.array([nx - 1, ny - 1, nz - 1])

    all_pts: list[np.ndarray] = []
    all_nrm: list[np.ndarray] = []

    for axis in range(3):
        c = cross[axis]
        if not c.any():
            continue
        i_lo = np.argwhere(c)
        i_hi = i_lo.copy()
        i_hi[:, axis] += 1
        v_lo = sdf[i_lo[:, 0], i_lo[:, 1], i_lo[:, 2]]
        v_hi = sdf[i_hi[:, 0], i_hi[:, 1], i_hi[:, 2]]
        keep = (np.abs(v_lo) < 0.9 * band_sentinel) & (np.abs(v_hi) < 0.9 * band_sentinel)
        if not keep.any():
            continue
        i_lo = i_lo[keep]
        v_lo = v_lo[keep]
        v_hi = v_hi[keep]
        denom = v_hi - v_lo
        t = np.where(np.abs(denom) > 1e-12, -v_lo / denom, 0.5)
        t = np.clip(t, 0.0, 1.0)
        ijk = i_lo.astype(np.float64)
        ijk[:, axis] += t
        xyz = bbox_min + ijk * h
        ijk_c = np.clip(ijk, 0.0, grid_max.astype(np.float64) - 1e-6)
        i0 = np.floor(ijk_c).astype(np.int64)
        f = ijk_c - i0
        i1 = np.minimum(i0 + 1, grid_max)
        f0 = 1.0 - f

        def _trilerp(field: np.ndarray) -> np.ndarray:
            c000 = field[i0[:, 0], i0[:, 1], i0[:, 2]]
            c100 = field[i1[:, 0], i0[:, 1], i0[:, 2]]
            c010 = field[i0[:, 0], i1[:, 1], i0[:, 2]]
            c110 = field[i1[:, 0], i1[:, 1], i0[:, 2]]
            c001 = field[i0[:, 0], i0[:, 1], i1[:, 2]]
            c101 = field[i1[:, 0], i0[:, 1], i1[:, 2]]
            c011 = field[i0[:, 0], i1[:, 1], i1[:, 2]]
            c111 = field[i1[:, 0], i1[:, 1], i1[:, 2]]
            c00 = c000 * f0[:, 0] + c100 * f[:, 0]
            c10 = c010 * f0[:, 0] + c110 * f[:, 0]
            c01 = c001 * f0[:, 0] + c101 * f[:, 0]
            c11 = c011 * f0[:, 0] + c111 * f[:, 0]
            c0 = c00 * f0[:, 1] + c10 * f[:, 1]
            c1 = c01 * f0[:, 1] + c11 * f[:, 1]
            return c0 * f0[:, 2] + c1 * f[:, 2]

        normals = np.stack([_trilerp(gx), _trilerp(gy), _trilerp(gz)], axis=1)
        mag = np.linalg.norm(normals, axis=1)
        good = (mag > 0.3) & (mag < 3.0)
        if not good.any():
            continue
        normals = normals[good] / mag[good, None]
        all_pts.append(xyz[good])
        all_nrm.append(normals)

    if not all_pts:
        print("  [poisson][WARN] No valid iso-surface samples")
        return pv.PolyData()

    pts = np.concatenate(all_pts, axis=0)
    nrms = np.concatenate(all_nrm, axis=0)
    print(f"  [poisson] {len(pts):,} iso-crossing samples (after gradient-validity filter)")

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    pcd.normals = o3d.utility.Vector3dVector(nrms)

    t_p = time.time()
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd, depth=int(depth)
    )
    print(
        f"  [poisson] depth={depth}: {len(mesh.vertices):,} verts, "
        f"{len(mesh.triangles):,} faces in {time.time()-t_p:.1f}s"
    )
    if len(mesh.triangles) == 0:
        print("  [poisson][WARN] Poisson returned empty mesh")
        return pv.PolyData()

    if density_quantile > 0:
        d = np.asarray(densities)
        thr = float(np.quantile(d, density_quantile))
        keep_mask = d > thr
        if (~keep_mask).any():
            mesh.remove_vertices_by_mask(np.logical_not(keep_mask))
            print(
                f"  [poisson] density crop q={density_quantile} thr={thr:.3f}: "
                f"removed {int((~keep_mask).sum()):,} verts "
                f"({100.0*(~keep_mask).sum()/len(keep_mask):.1f}%)"
            )

    verts = np.asarray(mesh.vertices)
    tris = np.asarray(mesh.triangles)
    if len(tris) == 0:
        return pv.PolyData()
    faces_flat = np.column_stack([np.full(len(tris), 3, dtype=np.int64), tris]).reshape(-1)
    return pv.PolyData(verts, faces_flat)


# ── MeshLib dual contouring + relaxation ─────────────────────────────────────


import time
import numpy as np
import pyvista as pv
from typing import Optional

# Assuming 'config' is defined in your broader scope
# import config 

def _count_nonfinite(mesh: Any, mrmeshnumpy: Any) -> int:
    """Number of vertices with a non-finite coordinate."""
    verts = mrmeshnumpy.getNumpyVerts(mesh)
    return int((~np.isfinite(verts).all(axis=1)).sum())


def _guarded_relax(
    mesh: Any,
    params: Any,
    mrmeshpy: Any,
    mrmeshnumpy: Any,
    chunk: int = 5,
    label: str = "relax",
) -> tuple[Any, int]:
    """``relaxKeepVolume`` run in chunks, reverting the first bad chunk.

    ``relaxKeepVolume`` rescales each vertex neighbourhood to hold its volume
    fixed. Where the local volume has collapsed -- sub-voxel vessels, or the
    sliver triangles an over-fine remesh leaves behind -- that scale factor
    diverges and the vertex comes back NaN/inf. Left unguarded this silently
    destroyed ~30% of the vertices on the left tree and ~24% on the right;
    ``drop_nonfinite_vertices`` then removed every face touching them, which is
    what shattered a single-component lumen into 40+ pieces.

    Running in chunks and restoring the last all-finite snapshot keeps whatever
    smoothing was safely achievable and lets the caller see how far it got.
    Topology is untouched by relaxation, so the revert rebuilds from the same
    face array. Returns ``(mesh, iterations_applied)``.
    """
    total = int(params.iterations)
    if total <= 0:
        return mesh, 0

    last_verts = mrmeshnumpy.getNumpyVerts(mesh).astype(np.float64)
    faces = mrmeshnumpy.getNumpyFaces(mesh.topology).astype(np.int32)

    step = mrmeshpy.MeshRelaxParams()
    step.force = float(params.force)
    region = getattr(params, "region", None)
    if region is not None:
        step.region = region

    done = 0
    while done < total:
        n = min(int(chunk), total - done)
        step.iterations = n
        mrmeshpy.relaxKeepVolume(mesh, step)
        verts = mrmeshnumpy.getNumpyVerts(mesh).astype(np.float64)
        n_bad = int((~np.isfinite(verts).all(axis=1)).sum())
        if n_bad:
            print(
                f"  [meshlib][GUARD] {label}: iterations {done + 1}-{done + n} "
                f"produced {n_bad:,} non-finite vertices; reverted to {done} "
                f"clean iteration(s)"
            )
            return mrmeshnumpy.meshFromFacesVerts(faces, last_verts), done
        last_verts = verts
        done += n
    return mesh, done


def mesh_from_sdf_meshlib(
    sdf_vol: np.ndarray,
    bbox_min: np.ndarray,
    voxel_size: float,
    dims: np.ndarray,
    relax_iterations: Optional[int] = None,
    relax_force: Optional[float] = None,
    remesh_target_edge: Optional[float] = None,
    capsule_tree: Any = None,
    cap_max_radii: np.ndarray | None = None,
) -> pv.PolyData:
    """MeshLib volume extraction optimized for thin, smooth tubular vessels."""
    try:
        import meshlib.mrmeshpy as mrmeshpy
        import meshlib.mrmeshnumpy as mrmeshnumpy
    except ImportError as exc:
        raise RuntimeError(
            "SDF_MESH_METHOD='meshlib' requires meshlib. Install with: pip install meshlib"
        ) from exc

    # Config governs; the explicit arguments remain per-call overrides. These
    # used to be hardcoded to 30 iterations at force 0.5 with the config
    # constants marked UNUSED, so there was no way to turn the relaxation down.
    if relax_iterations is None:
        relax_iterations = int(getattr(config, "SDF_MESHLIB_RELAX_ITERS", 5))
    if relax_force is None:
        relax_force = float(getattr(config, "SDF_MESHLIB_RELAX_FORCE", 0.4))

    t_start = time.time()
    nx, ny, nz = int(dims[0]), int(dims[1]), int(dims[2])
    h = float(voxel_size)
    bbox_min = np.asarray(bbox_min, dtype=np.float64)

    # Target edge as a multiple of the voxel size. The SDF carries no detail
    # below h, so edges much shorter than h add no geometry -- they only
    # manufacture the sliver triangles that make relaxKeepVolume diverge. The
    # previous hardcoded ``h * 0.5`` contradicted this function's own comment
    # ("~1.25x voxel size") and asked for 60 um edges on a 120 um grid.
    if remesh_target_edge is None:
        remesh_target_edge = h * float(
            getattr(config, "MESHLIB_TARGET_EDGE_FACTOR", 1.25)
        )

    print("[meshlib] Initializing dense grid allocation...")
    sdf_data = np.asarray(sdf_vol, dtype=np.float32)
    volume = mrmeshpy.SimpleVolume()
    volume.dims = mrmeshpy.Vector3i(nx, ny, nz)
    volume.voxelSize = mrmeshpy.Vector3f(h, h, h)
    
    # PERFORMANCE FIX: Avoid .tolist() on large HiP-CT volumes
    # Pybind11 automatically handles the conversion from contiguous numpy arrays
    flat_array = sdf_data.ravel(order="F")
    volume.data = mrmeshpy.std_vector_float(flat_array)

    print("[meshlib] Running marching cubes...")
    mc_params = mrmeshpy.MarchingCubesParams()
    mc_params.iso = 0.0
    # MeshLib places sample (i,j,k) at ``origin + voxelSize * (i + 0.5)`` -- it
    # reads ``origin`` as the *corner* of the first voxel cell. The SDF here is
    # sampled *at* ``bbox_min`` (``x = bbox_min + arange(dims) * h``), so passing
    # bbox_min straight through shifts the whole mesh by +half a voxel on every
    # axis. Verified against an analytic sphere: the extracted surface came out
    # at +0.501/+0.494/+0.502 voxels before this correction, and the resulting
    # ~0.095 mm translation was the dominant non-bifurcation error against the
    # spatial graph. Shift the origin back by half a voxel so sample i lands on
    # bbox_min + i*h as intended.
    mc_origin = bbox_min - 0.5 * h
    mc_params.origin = mrmeshpy.Vector3f(
        float(mc_origin[0]), float(mc_origin[1]), float(mc_origin[2])
    )
    mc_params.lessInside = True 
    mc_params.maxVertices = 2_147_483_647
    mesh = mrmeshpy.marchingCubes(volume, mc_params)
    
    if mesh.topology.numValidFaces() == 0:
        print("  [meshlib][WARN] Volume extraction returned empty mesh")
        return pv.PolyData()

# OPTIMIZATION 1: Isotropic Remeshing for Tubular Geometry
    print(f"  [meshlib] Isotropic remeshing (target edge: {remesh_target_edge:.4f})...")
    
    # MeshLib uses RemeshSettings instead of RemeshParams
    remesh_settings = mrmeshpy.RemeshSettings()
    remesh_settings.targetEdgeLen = float(remesh_target_edge)
    
    # Optional: project vertices back to the original surface to strictly maintain the lumen boundary
    remesh_settings.projectOnOriginalMesh = True  
    
    mrmeshpy.remesh(mesh, remesh_settings)
    _stage_nf = {"marching_cubes+remesh": _count_nonfinite(mesh, mrmeshnumpy)}

    # OPTIMIZATION 2: True Volume-Preserving Relaxation
    print(
        f"  [meshlib] Smoothing walls via volume-preserving relaxation "
        f"({relax_iterations} iterations, force {relax_force})..."
    )
    relax_params = mrmeshpy.MeshRelaxParams()
    relax_params.iterations = int(relax_iterations)
    relax_params.force = float(relax_force)

    # Use relaxKeepVolume instead of standard relax to prevent microvessel
    # collapse, chunked + reverting so a diverging neighbourhood cannot emit
    # non-finite vertices into the mesh.
    mesh, _n_applied = _guarded_relax(
        mesh, relax_params, mrmeshpy, mrmeshnumpy, label="global relax"
    )
    _stage_nf["global_relax"] = _count_nonfinite(mesh, mrmeshnumpy)

    # OPTIMIZATION 3: Thin-vessel region refinement (native — no numpy rebuild).
    # Sub-voxel vessels get too few edges around their small circumference at
    # the global target edge, so they render faceted. Region-restrict a second
    # remesh (smaller edge, curvature-adaptive) + relax to the faces whose
    # nearest capsule radius is below the threshold, rounding their cross-
    # section. Operates on the native (manifold) MeshLib mesh, so no faces are
    # dropped. mesh.pack() first so valid FaceIds are contiguous and the bool
    # mask from getNumpyFaces aligns with faceBitSetFromBools.
    if (
        getattr(config, "THIN_VESSEL_REFINE", False)
        and capsule_tree is not None
        and cap_max_radii is not None
    ):
        try:
            mesh.pack()
        except Exception:
            pass
        vb = mrmeshnumpy.getNumpyVerts(mesh)
        fb = mrmeshnumpy.getNumpyFaces(mesh.topology).astype(np.int64)
        thr = float(config.THIN_VESSEL_REFINE_RADIUS_MM)
        _d, vcap = capsule_tree.query(vb, k=1)
        thin_v = np.asarray(cap_max_radii)[vcap] < thr
        thin_f = thin_v[fb].all(axis=1)
        n_thin = int(thin_f.sum())
        if n_thin > 0:
            circ = max(int(config.THIN_VESSEL_CIRCUMF_TARGET), 6)
            rep_r = max(thr * 0.5, h)
            # Never ask for edges materially finer than the grid: the SDF holds
            # no detail below h, so a sub-h target only produces slivers. The
            # old h*0.2 floor let this reach h/3.2 (37.8 um on a 120 um grid).
            min_edge = h * float(getattr(config, "THIN_VESSEL_MIN_EDGE_FACTOR", 0.5))
            thin_edge = max(2.0 * np.pi * rep_r / circ, min_edge)
            print(
                f"  [meshlib] Thin-vessel refine: {n_thin:,}/{len(fb):,} faces "
                f"(< {thr*1000:.0f} um), target edge {thin_edge*1000:.1f} um "
                f"(floor {min_edge*1000:.1f} um)..."
            )
            rs_thin = mrmeshpy.RemeshSettings()
            rs_thin.targetEdgeLen = float(thin_edge)
            rs_thin.region = mrmeshnumpy.faceBitSetFromBools(thin_f)
            rs_thin.projectOnOriginalMesh = True
            rs_thin.useCurvature = True
            mrmeshpy.remesh(mesh, rs_thin)
            _stage_nf["thin_remesh"] = _count_nonfinite(mesh, mrmeshnumpy)
            iters = int(getattr(config, "THIN_VESSEL_REFINE_RELAX_ITERS", 0))
            if iters > 0:
                vb2 = mrmeshnumpy.getNumpyVerts(mesh)
                _d2, vcap2 = capsule_tree.query(vb2, k=1)
                thin_v2 = np.asarray(cap_max_radii)[vcap2] < thr
                if thin_v2.any():
                    rp = mrmeshpy.MeshRelaxParams()
                    rp.iterations = iters
                    rp.force = float(config.THIN_VESSEL_REFINE_RELAX_FORCE)
                    rp.region = mrmeshnumpy.vertBitSetFromBools(thin_v2)
                    mesh, _ = _guarded_relax(
                        mesh, rp, mrmeshpy, mrmeshnumpy, label="thin-vessel relax"
                    )
                    _stage_nf["thin_relax"] = _count_nonfinite(mesh, mrmeshnumpy)

    verts = mrmeshnumpy.getNumpyVerts(mesh).astype(np.float64)
    tris = mrmeshnumpy.getNumpyFaces(mesh.topology).astype(np.int64)

    # Per-stage attribution. A single warning after every stage (as before)
    # made it impossible to tell which of marching cubes, the global relax,
    # the thin remesh or the thin relax introduced non-finite vertices.
    print(
        "  [meshlib] non-finite vertices by stage: "
        + ", ".join(f"{k}={v:,}" for k, v in _stage_nf.items())
    )
    n_nf = int((~np.isfinite(verts).all(axis=1)).sum())
    if n_nf:
        print(f"  [meshlib][WARN] extraction produced {n_nf:,} non-finite vertices")

    if len(tris) == 0:
        return pv.PolyData()

    faces_flat = np.column_stack([np.full(len(tris), 3, dtype=np.int64), tris]).reshape(-1)
    print(
        f"  [meshlib] Extraction complete: {len(verts):,} verts, "
        f"{len(tris):,} faces in {time.time()-t_start:.1f}s"
    )
    return pv.PolyData(verts, faces_flat)


# ── Vertex sanitization ──────────────────────────────────────────────────────


def drop_nonfinite_vertices(surface: pv.PolyData) -> tuple[pv.PolyData, int]:
    """Remove vertices with NaN/inf coordinates and any faces referencing them.

    Returns ``(cleaned_surface, n_dropped)``. No-op (returns the same object)
    when all vertices are finite. Guards against degenerate meshlib relaxation /
    Poisson interpolation that can emit non-finite vertex coordinates, which
    would otherwise crash the downstream capsule KD-tree query.
    """
    pts = np.asarray(surface.points)
    if pts.size == 0:
        return surface, 0
    finite = np.isfinite(pts).all(axis=1)
    n_bad = int((~finite).sum())
    if n_bad == 0:
        return surface, 0
    kept = surface.extract_points(finite, adjacent_cells=False).extract_surface()
    return kept, n_bad


# ── Dispatcher ───────────────────────────────────────────────────────────────


def extract_isosurface(
    sdf_vol: np.ndarray,
    bbox_min: np.ndarray,
    voxel_size: float,
    dims: np.ndarray,
    method: str | None = None,
    capsule_tree: Any = None,
    cap_max_radii: np.ndarray | None = None,
) -> pv.PolyData:
    """Dispatch on ``config.SDF_MESH_METHOD`` (or the override ``method``).

    ``capsule_tree`` / ``cap_max_radii`` (when given, meshlib path only) drive
    the native thin-vessel region refinement inside ``mesh_from_sdf_meshlib``.
    """
    if method is None:
        method = config.SDF_MESH_METHOD
    if method == "poisson":
        return mesh_from_sdf_poisson(sdf_vol, bbox_min, voxel_size, dims)
    if method == "meshlib":
        return mesh_from_sdf_meshlib(
            sdf_vol, bbox_min, voxel_size, dims,
            capsule_tree=capsule_tree, cap_max_radii=cap_max_radii,
        )
    if method == "mc":
        grid = pv.ImageData()
        grid.dimensions = tuple(int(d) for d in dims)
        grid.origin = tuple(float(v) for v in bbox_min)
        grid.spacing = (voxel_size, voxel_size, voxel_size)
        grid.point_data["sdf"] = np.asarray(sdf_vol).ravel(order="F")
        return fast_contour_zero(grid, scalars="sdf")
    if method == "adaptive":
        raise ValueError(
            "the adaptive implicit backend is field-driven and must be invoked "
            "through pipeline.generate_sdf_surface"
        )
    raise ValueError(f"Unknown SDF_MESH_METHOD: {method!r}")


# ── Mesh-level anti-bridge cut ───────────────────────────────────────────────


def cut_non_adjacent_bridges(
    surface: pv.PolyData,
    segments: list[dict[str, Any]],
    points: dict[int, tuple],
    adj_matrix: np.ndarray,
) -> tuple[pv.PolyData, int]:
    """Delete mesh faces that span topology-disconnected segments.

    Labels each vertex by the nearest centerline segment, then drops
    triangles whose three vertices don't all lie on a pairwise-adjacent
    segment set. Optional ``fill_holes`` afterwards.
    """
    from scipy.spatial import KDTree

    verts = np.asarray(surface.points, dtype=np.float64)
    faces_raw = np.asarray(surface.faces, dtype=np.int64).reshape(-1)
    if verts.size == 0 or faces_raw.size == 0:
        return surface, 0

    if faces_raw.size % 4 == 0 and (faces_raw[::4] == 3).all():
        tris = faces_raw.reshape(-1, 4)[:, 1:].copy()
    else:
        tris_list: list[np.ndarray] = []
        i = 0
        while i < faces_raw.size:
            n = int(faces_raw[i])
            if n == 3 and i + 3 < faces_raw.size + 1:
                tris_list.append(faces_raw[i + 1: i + 4])
            i += n + 1
        if not tris_list:
            return surface, 0
        tris = np.array(tris_list, dtype=np.int64)

    cl_pts: list[list[float]] = []
    cl_segs: list[int] = []
    for seg_idx, seg in enumerate(segments):
        for pid in seg["point_ids"]:
            if pid in points:
                p = points[pid]
                cl_pts.append([p[0] / 1000.0, p[1] / 1000.0, p[2] / 1000.0])
                cl_segs.append(seg_idx)
    if not cl_pts:
        return surface, 0
    cl_pts_arr = np.asarray(cl_pts, dtype=np.float64)
    cl_segs_arr = np.asarray(cl_segs, dtype=np.int64)

    print(f"  [BRIDGE CUT] Assigning {len(verts):,} mesh vertices to centerlines...")
    cl_kd = KDTree(cl_pts_arr)
    n_verts = len(verts)
    vert_seg = np.empty(n_verts, dtype=np.int64)
    chunk = min(max(50_000, n_verts // 20), 250_000)
    n_chunks = (n_verts + chunk - 1) // chunk
    t_query = time.time()
    for ci in range(n_chunks):
        start = ci * chunk
        end = min((ci + 1) * chunk, n_verts)
        _, nn = cl_kd.query(verts[start:end])
        vert_seg[start:end] = cl_segs_arr[nn]
        if n_chunks > 1:
            elapsed = time.time() - t_query
            rate = end / max(elapsed, 1e-6)
            print(
                f"    [BRIDGE CUT] {ci + 1}/{n_chunks} chunks, "
                f"{end:,}/{n_verts:,} verts ({rate:,.0f} verts/s)"
            )
    print(f"  [BRIDGE CUT] KDTree query done in {time.time() - t_query:.1f}s")
    face_segs = vert_seg[tris]
    s0, s1, s2 = face_segs[:, 0], face_segs[:, 1], face_segs[:, 2]
    pair01 = adj_matrix[s0, s1]
    pair12 = adj_matrix[s1, s2]
    pair02 = adj_matrix[s0, s2]
    bridge_mask = ~(pair01 & pair12 & pair02)
    n_bridges = int(bridge_mask.sum())
    if n_bridges == 0:
        return surface, 0

    keep = tris[~bridge_mask]
    if len(keep) == 0:
        return surface, n_bridges
    flat_faces = np.column_stack([np.full(len(keep), 3, dtype=np.int64), keep]).ravel()
    new_surface = pv.PolyData(verts, flat_faces)
    if config.MESH_BRIDGE_CUT_HOLE_FILL:
        try:
            new_surface = new_surface.fill_holes(hole_size=10.0)
        except Exception as exc:
            print(f"    [BRIDGE CUT] hole-fill failed: {exc}")
    return new_surface, n_bridges


# ── Flat caps ────────────────────────────────────────────────────────────────


def _extract_boundary_loops(mesh: pv.PolyData) -> list[np.ndarray]:
    """Walk open boundary edges into closed (N, 3) loops."""
    boundary = mesh.extract_feature_edges(
        boundary_edges=True,
        feature_edges=False,
        manifold_edges=False,
        non_manifold_edges=False,
    )
    if boundary.n_cells == 0:
        return []
    pts = np.array(boundary.points)
    from collections import defaultdict

    adj: dict[int, list[int]] = defaultdict(list)
    lines = np.array(boundary.lines)
    i = 0
    while i < len(lines):
        n_verts = int(lines[i])
        seg = [int(v) for v in lines[i + 1: i + 1 + n_verts]]
        for k in range(len(seg) - 1):
            a, b = seg[k], seg[k + 1]
            adj[a].append(b)
            adj[b].append(a)
        i += n_verts + 1

    visited: set[int] = set()
    loops: list[np.ndarray] = []
    for start in sorted(adj.keys()):
        if start in visited:
            continue
        loop = [start]
        visited.add(start)
        current = start
        while True:
            cands = [nb for nb in adj[current] if nb not in visited]
            if not cands:
                break
            nxt = cands[0]
            loop.append(nxt)
            visited.add(nxt)
            current = nxt
        if len(loop) >= 3 and start in adj[current]:
            loops.append(pts[loop])
        elif len(loop) >= 3:
            print(f"  [WARN] Boundary walk: open chain of {len(loop)} vertices skipped")
    return loops


def _make_flat_cap(loop_pts: np.ndarray, outward_normal: np.ndarray) -> pv.PolyData | None:
    """Fan-triangulate a coplanar boundary ring with consistent outward winding."""
    pts = np.array(loop_pts, dtype=np.float64)
    n = len(pts)
    if n < 3:
        return None
    centroid = pts.mean(axis=0)

    t = np.asarray(outward_normal, dtype=np.float64)
    t /= max(np.linalg.norm(t), 1e-12)
    up = np.array([0.0, 0.0, 1.0])
    if abs(np.dot(t, up)) > 0.9:
        up = np.array([1.0, 0.0, 0.0])
    u = np.cross(t, up)
    u /= np.linalg.norm(u)
    v = np.cross(t, u)

    rel = pts - centroid
    angles = np.arctan2(rel @ v, rel @ u)
    order = np.argsort(angles)
    ordered = pts[order]

    all_pts = np.vstack([ordered, centroid])
    cen_idx = n
    trial_n = np.cross(ordered[1] - ordered[0], centroid - ordered[0])
    flip = np.dot(trial_n, outward_normal) < 0
    faces: list[int] = []
    for i in range(n):
        j = (i + 1) % n
        if flip:
            faces.extend([3, j, i, cen_idx])
        else:
            faces.extend([3, i, j, cen_idx])
    return pv.PolyData(all_pts, np.array(faces, dtype=np.int64))


def create_flat_caps(
    mesh: pv.PolyData,
    endpoint_info: list[tuple[np.ndarray, np.ndarray, float]],
) -> tuple[pv.PolyData, int]:
    """Plane-clip every terminal endpoint perpendicular to its tangent and
    fan-triangulate the resulting circular boundary into a flat cap."""
    if not endpoint_info:
        return mesh, 0

    print(
        f"[FLAT CAPS] Clipping {len(endpoint_info)} vessel endpoints "
        "perpendicular to axis..."
    )
    ep_positions = np.array([p for p, _, _ in endpoint_info])
    ep_normals = np.array([n for _, n, _ in endpoint_info])

    n_clipped = 0
    for ep_idx, (pos, outward_normal, radius) in enumerate(endpoint_info):
        clip_origin = pos - outward_normal * max(radius * 0.05, 0.005)
        try:
            clipped = mesh.clip(
                normal=outward_normal,
                origin=clip_origin,
                invert=False,
            )
            if clipped is not None and clipped.n_points > 0:
                mesh = clipped
                n_clipped += 1
        except Exception as e:
            print(f"    [WARN] Clip failed for endpoint {ep_idx}: {e}")

    print(f"  Clipped {n_clipped}/{len(endpoint_info)} endpoints")
    if n_clipped == 0:
        return mesh, 0

    loops = _extract_boundary_loops(mesh)
    print(f"  Found {len(loops)} boundary loops")
    if not loops:
        return mesh, 0

    cap_meshes: list[pv.PolyData] = []
    n_capped = 0
    for loop_pts in loops:
        loop_centroid = loop_pts.mean(axis=0)
        dists = np.linalg.norm(ep_positions - loop_centroid, axis=1)
        nearest = int(dists.argmin())
        cap = _make_flat_cap(loop_pts, ep_normals[nearest])
        if cap is not None and cap.n_cells > 0:
            cap_meshes.append(cap)
            n_capped += 1

    if cap_meshes:
        result = mesh
        for cap in cap_meshes:
            result = result.merge(cap)
        result = result.clean(tolerance=1e-6)
        print(f"  Added {n_capped} flat caps — mesh watertight: {result.is_manifold}")
        return result, n_capped
    return mesh, 0


# ── Triangle-face helper ─────────────────────────────────────────────────────


def extract_triangle_faces(mesh: pv.PolyData) -> tuple[pv.PolyData, np.ndarray]:
    """Return a triangulated PolyData mesh and ``(N, 3)`` triangle index array."""
    if not isinstance(mesh, pv.PolyData):
        mesh = mesh.extract_surface()
    mesh = mesh.triangulate()
    if mesh.n_cells == 0:
        return mesh, np.empty((0, 3), dtype=np.int64)
    if hasattr(mesh, "regular_faces"):
        tri = np.asarray(mesh.regular_faces, dtype=np.int64)
    else:
        tri = np.asarray(mesh.faces).reshape(-1, 4)[:, 1:4].astype(np.int64, copy=False)
    return mesh, tri


__all__ = [
    "fast_contour_zero",
    "mesh_from_sdf_poisson",
    "mesh_from_sdf_meshlib",
    "extract_isosurface",
    "drop_nonfinite_vertices",
    "cut_non_adjacent_bridges",
    "create_flat_caps",
    "extract_triangle_faces",
]
