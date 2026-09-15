"""Mesh repair + smoothing.

- ``run_pymeshfix`` -- subprocess wrapper with hard timeout (pymeshfix
  hangs on some non-manifold inputs).
- ``repair_mesh``  -- clean / fill-holes / pymeshfix / fix-normals
  pipeline gated by ``config.MESH_REPAIR`` etc.
- ``radius_constrained_taubin`` -- Taubin smoothing with per-vertex
  displacement clamped to a fraction of the local vessel radius and
  cap-edge vertices pinned to their pre-smoothing positions.
- ``fast_clean_triangle_mesh`` -- numpy-based clean equivalent to
  ``surface.clean(tolerance=atol)`` for large refinement outputs.
"""

from __future__ import annotations

import multiprocessing as mp
import queue
from typing import Any

import numpy as np
import pyvista as pv

from .config import runtime_config as config
from .mesh_extract import extract_triangle_faces


# ── Fast clean (numpy-based dedupe + degeneracy removal) ─────────────────────


def fast_clean_triangle_mesh(surface: pv.PolyData, atol: float) -> pv.PolyData:
    """Drop-in replacement for ``surface.clean(tolerance=atol)`` based on
    ``np.unique`` instead of VTK's spatial hash. Faster for large meshes."""
    pts = np.asarray(surface.points, dtype=np.float64)
    faces_raw = np.asarray(surface.faces, dtype=np.int64).reshape(-1)
    if faces_raw.size == 0 or pts.size == 0:
        return surface

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
            return surface
        tris = np.array(tris_list, dtype=np.int64)

    scale = 1.0 / max(atol, 1e-12)
    pts_q = np.round(pts * scale).astype(np.int64)
    _, unique_idx, inverse = np.unique(pts_q, axis=0, return_index=True, return_inverse=True)
    new_pts = pts[unique_idx]
    new_tris = inverse[tris]

    deg = (
        (new_tris[:, 0] == new_tris[:, 1])
        | (new_tris[:, 1] == new_tris[:, 2])
        | (new_tris[:, 0] == new_tris[:, 2])
    )
    new_tris = new_tris[~deg]
    if len(new_tris) == 0:
        return pv.PolyData()

    used = np.unique(new_tris.ravel())
    if len(used) < len(new_pts):
        remap = np.full(len(new_pts), -1, dtype=np.int64)
        remap[used] = np.arange(len(used))
        new_pts = new_pts[used]
        new_tris = remap[new_tris]

    vtk_faces = np.column_stack(
        [np.full(len(new_tris), 3, dtype=np.int64), new_tris]
    ).ravel()
    return pv.PolyData(new_pts, vtk_faces)


# ── pymeshfix subprocess wrapper ─────────────────────────────────────────────


def _pymeshfix_worker(points: np.ndarray, tri: np.ndarray, out_queue: Any) -> None:
    try:
        import pymeshfix

        pts = np.asarray(points, dtype=np.float64)
        faces = np.asarray(tri, dtype=np.int64)
        mf = pymeshfix.MeshFix(pts, faces)
        try:
            mf.repair(verbose=False)
        except TypeError:
            mf.repair()

        result = getattr(mf, "mesh", None)
        if result is not None and hasattr(result, "points") and result.n_points > 0:
            v = np.asarray(result.points, dtype=np.float64)
            if hasattr(result, "regular_faces"):
                f = np.asarray(result.regular_faces, dtype=np.int64)
            else:
                f = np.asarray(result.faces).reshape(-1, 4)[:, 1:4].astype(np.int64, copy=False)
            out_queue.put(("ok", (v, f)))
            return

        v = getattr(mf, "v", getattr(mf, "vertices", None))
        f = getattr(mf, "f", getattr(mf, "faces", None))
        if v is None or f is None or len(v) == 0:
            out_queue.put(("none", None))
            return

        f = np.asarray(f, dtype=np.int64)
        if f.ndim == 1:
            if f.size % 4 == 0:
                f = f.reshape(-1, 4)[:, 1:4]
            elif f.size % 3 == 0:
                f = f.reshape(-1, 3)
            else:
                raise ValueError("Unsupported pymeshfix face array layout")
        elif f.ndim == 2 and f.shape[1] == 4:
            if np.all(f[:, 0] == 3):
                f = f[:, 1:4]
            else:
                raise ValueError("Unsupported pymeshfix face array layout")

        out_queue.put(("ok", (np.asarray(v, dtype=np.float64), f)))
    except Exception as e:
        out_queue.put(("err", repr(e)))


def run_pymeshfix(
    mesh: pv.PolyData, timeout_sec: int | None = None
) -> pv.PolyData | None:
    """Run pymeshfix in a subprocess. Raises ``TimeoutError`` past the timeout."""
    if timeout_sec is None:
        timeout_sec = config.PYMESHFIX_TIMEOUT_SEC

    mesh, tri = extract_triangle_faces(mesh)
    if mesh.n_points == 0 or tri.shape[0] == 0:
        return None

    if config.PYMESHFIX_MAX_FACES and tri.shape[0] > config.PYMESHFIX_MAX_FACES:
        print(
            f"    [WARN] Skipping pymeshfix: {tri.shape[0]:,} faces exceeds "
            f"limit ({config.PYMESHFIX_MAX_FACES:,}); set PYMESHFIX_MAX_FACES=0 to disable"
        )
        return None

    ctx = mp.get_context("spawn")
    out_queue = ctx.Queue(maxsize=1)
    worker = ctx.Process(
        target=_pymeshfix_worker,
        args=(np.asarray(mesh.points, dtype=np.float64), tri, out_queue),
    )
    worker.start()
    worker.join(timeout_sec)

    if worker.is_alive():
        worker.terminate()
        worker.join(timeout=2.0)
        if worker.is_alive() and hasattr(worker, "kill"):
            worker.kill()
            worker.join(timeout=1.0)
        out_queue.close()
        out_queue.join_thread()
        raise TimeoutError(f"pymeshfix timed out after {timeout_sec}s")

    try:
        status, payload = out_queue.get_nowait()
    except queue.Empty:
        out_queue.close()
        out_queue.join_thread()
        if worker.exitcode == 0:
            return None
        raise RuntimeError(f"pymeshfix worker exited with code {worker.exitcode}")

    out_queue.close()
    out_queue.join_thread()

    if status == "none":
        return None
    if status == "err":
        raise RuntimeError(payload)
    if status != "ok":
        raise RuntimeError(f"Unexpected pymeshfix worker status: {status}")

    v, f = payload
    if len(v) == 0 or len(f) == 0:
        return None
    faces_vtk = np.hstack(
        [np.full((f.shape[0], 1), 3, dtype=np.int64), f.astype(np.int64, copy=False)]
    ).ravel()
    return pv.PolyData(v, faces_vtk)


# ── Repair pipeline ──────────────────────────────────────────────────────────


def repair_mesh(mesh: pv.PolyData, voxel_size: float = 0.1) -> pv.PolyData:
    """Clean -> fill-holes -> pymeshfix -> fix-normals -> retry."""
    repaired = mesh

    if config.MESH_REMOVE_DEGENERATE:
        try:
            repaired = repaired.clean(tolerance=voxel_size * 0.05)
        except Exception as e:
            print(f"    [WARN] Clean failed: {e}")

    if config.MESH_FILL_HOLES:
        try:
            hole_thresh = min(voxel_size * 50, voxel_size * 20 + 1.0)
            repaired = repaired.fill_holes(hole_size=hole_thresh)
        except Exception as e:
            print(f"    [WARN] Fill holes failed: {e}")

    if config.USE_PYMESHFIX:
        try:
            print("    Using pymeshfix...")
            result = run_pymeshfix(repaired, timeout_sec=config.PYMESHFIX_TIMEOUT_SEC)
            if result is not None and result.n_points > 0:
                repaired = result
                print(f"    pymeshfix: {repaired.n_points:,} vertices, {repaired.n_cells:,} faces")
            else:
                print("    [WARN] pymeshfix returned empty result, keeping pre-repair mesh")
        except TimeoutError as e:
            print(f"    [WARN] {e}, using fallback repair")
        except Exception as e:
            if "No module named 'pymeshfix'" in str(e):
                print("    [INFO] pymeshfix not available, using fallback repair")
            else:
                print(f"    [WARN] pymeshfix failed: {e}, using fallback repair")

    if config.MESH_FIX_NORMALS:
        try:
            repaired.compute_normals(
                inplace=True, consistent_normals=True, auto_orient_normals=True
            )
        except Exception as e:
            print(f"    [WARN] Fix normals failed: {e}")

    try:
        repaired = repaired.clean(tolerance=voxel_size * 0.01)
    except Exception:
        pass

    if not repaired.is_manifold:
        print("    [WARN] Mesh non-manifold after repair, retrying (without largest-component drop)...")
        try:
            if not isinstance(repaired, pv.PolyData):
                repaired = repaired.extract_surface()
            repaired = repaired.clean(tolerance=voxel_size * 0.02)
            if config.USE_PYMESHFIX:
                try:
                    retry = run_pymeshfix(
                        repaired, timeout_sec=max(30, config.PYMESHFIX_TIMEOUT_SEC // 2)
                    )
                    if retry is not None and retry.n_points > 0:
                        repaired = retry
                        print(
                            f"    Retry: {repaired.n_points:,} verts, "
                            f"manifold={repaired.is_manifold}"
                        )
                except Exception:
                    pass
        except Exception as e:
            print(f"    [WARN] Retry failed: {e}")

    return repaired


# ── Radius-constrained Taubin smoothing ──────────────────────────────────────


def radius_constrained_taubin(
    surface: pv.PolyData,
    capsule_tree: Any,
    cap_max_radii: np.ndarray,
    voxel_size: float,
    term_pos: np.ndarray | None = None,
    term_nrm: np.ndarray | None = None,
    term_rad: np.ndarray | None = None,
    term_tree: Any | None = None,
    bif_pos: np.ndarray | None = None,
    bif_rad: np.ndarray | None = None,
) -> pv.PolyData:
    """Run Taubin smoothing then clamp each vertex's displacement to a fraction
    of the nearest capsule's radius. Optionally pin cap-edge vertices to keep
    flat-cap 90 deg corners crisp.

    When ``config.TAUBIN_JUNCTION_ONLY`` and ``bif_pos`` are supplied, each
    vertex's (clamped) displacement is additionally scaled by a feathered
    junction weight — full inside ``TAUBIN_JUNCTION_FACTOR * r_bif`` of a
    bifurcation node, ramping to 0 over the next ``TAUBIN_JUNCTION_FEATHER *
    r_bif`` — so the volume-preserving smoothing polishes the ridge at
    junctions while leaving the rest of the vessel exactly as extracted.
    """
    if config.TAUBIN_ITERS <= 0:
        return surface

    print(f"  Radius-constrained Taubin smoothing ({config.TAUBIN_ITERS} iters)...")
    verts_before = np.array(surface.points).copy()
    if not np.isfinite(verts_before).all():
        print("  [WARN] Taubin: non-finite vertices present; skipping smoothing")
        return surface
    _vert_dists, vert_caps = capsule_tree.query(verts_before, k=1)
    local_radii = cap_max_radii[vert_caps]

    surface = surface.smooth_taubin(
        n_iter=config.TAUBIN_ITERS, pass_band=config.TAUBIN_BAND
    )

    verts_after = np.array(surface.points)
    displacement = verts_after - verts_before
    disp_mag = np.linalg.norm(displacement, axis=1)
    max_disp = local_radii * float(getattr(config, "TAUBIN_MAX_DISP_FACTOR", 0.2))
    excess = disp_mag > max_disp
    if excess.any():
        scale = np.where(excess, max_disp / np.maximum(disp_mag, 1e-12), 1.0)
        displacement = displacement * scale[:, None]

    # Junction localization: feather the displacement to a ball around each
    # bifurcation node so only the junction surface (where the residual ridge
    # sits) is smoothed; everything else keeps its exact extracted position.
    if (
        getattr(config, "TAUBIN_JUNCTION_ONLY", False)
        and bif_pos is not None
        and len(bif_pos) > 0
    ):
        from scipy.spatial import cKDTree

        d_bif, i_bif = cKDTree(np.asarray(bif_pos, dtype=np.float64)).query(verts_before)
        if bif_rad is not None and len(bif_rad) > 0:
            r_bif = np.asarray(bif_rad, dtype=np.float64)[i_bif]
        else:
            r_bif = np.full(len(verts_before), float(np.mean(local_radii)))
        inner = float(config.TAUBIN_JUNCTION_FACTOR) * r_bif
        feather = max(float(config.TAUBIN_JUNCTION_FEATHER), 1e-6) * r_bif
        u = np.clip((d_bif - inner) / np.maximum(feather, 1e-9), 0.0, 1.0)
        w = 1.0 - (u * u * (3.0 - 2.0 * u))   # 1 inside the ball -> 0 past the feather
        displacement = displacement * w[:, None]
        print(
            f"    Junction-localized: {(w > 0.01).sum():,} vertices "
            f"({100*(w > 0.01).mean():.1f}%) within smoothing zone"
        )

    surface.points = verts_before + displacement
    n_clamped = int(excess.sum())

    if (
        config.SDF_FLAT_TERMINAL_CAPS
        and term_tree is not None
        and term_pos is not None
        and term_nrm is not None
        and term_rad is not None
    ):
        td, ti = term_tree.query(np.array(surface.points))
        within = td < (term_rad[ti] * config.SDF_FLAT_CAP_REACH_FACTOR)
        if within.any():
            v = np.array(surface.points)
            beyond = np.abs(np.sum((v - term_pos[ti]) * term_nrm[ti], axis=1))
            at_cap = within & (beyond < voxel_size * 1.5)
            if at_cap.any():
                v[at_cap] = verts_before[at_cap]
                surface.points = v
                print(f"    Protected {at_cap.sum():,} cap-edge vertices from Taubin")

    print(
        f"    Clamped {n_clamped:,} vertices ({100*n_clamped/max(len(verts_before),1):.1f}%) "
        f"to protect small vessels"
    )
    return surface


__all__ = [
    "fast_clean_triangle_mesh",
    "run_pymeshfix",
    "repair_mesh",
    "radius_constrained_taubin",
]
