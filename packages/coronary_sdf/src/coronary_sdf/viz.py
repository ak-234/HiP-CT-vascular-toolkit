"""Optional PyVista debug visualisation.

Every function is a no-op when ``config.DEBUG_VIS`` is False, so callers
can sprinkle them through the pipeline without conditional checks.

The plotter helper falls back to a PNG screenshot when an interactive
window cannot be opened (e.g. headless CI).
"""

from __future__ import annotations

import re
import time
from pathlib import Path
from typing import Any

import numpy as np
import pyvista as pv

from .config import runtime_config as config


# ── Plotter helper ────────────────────────────────────────────────────────────


def _show_plotter(pl: pv.Plotter, title: str) -> None:
    try:
        if config.DEBUG_VIS_BLOCK:
            pl.show(auto_close=True)
        else:
            pl.show(interactive_update=True, auto_close=False)
    except Exception as e:
        print(f"  [VIS][WARN] Interactive visualization failed for '{title}': {e}")
        if config.DEBUG_VIS_SAVE_FALLBACK:
            try:
                out_dir = Path(config.OUTPUT_DIR)
                out_dir.mkdir(parents=True, exist_ok=True)
                safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", title).strip("_")
                png_path = out_dir / f"debug_{safe}_{int(time.time())}.png"
                pl.screenshot(str(png_path))
                print(f"  [VIS] Saved fallback screenshot: {png_path}")
            except Exception as e2:
                print(f"  [VIS][WARN] Screenshot fallback failed: {e2}")
    finally:
        try:
            pl.close()
        except Exception:
            pass


# ── Scalar colouring helper ────────────────────────────────────────────────────


def _add_scalar_lines(
    pl: pv.Plotter,
    poly: pv.PolyData,
    *,
    seg_idx: Any,
    radius: Any,
    line_width: float,
    radius_on_points: bool = False,
    strahler: Any = None,
) -> None:
    """Colour a polyline mesh by segment idx (turbo), radius (viridis), or
    Strahler order (discrete tab10), selected by ``config.DEBUG_VIS_COLOR_BY``.
    Falls back to seg_idx when the requested mode's data is unavailable
    (no radius array, or missing/``None`` Strahler values). Strahler is a
    per-segment value, so it is always applied per polyline cell."""
    if config.DEBUG_VIS_COLOR_BY == "radius" and radius is not None:
        target = poly.point_data if radius_on_points else poly.cell_data
        target["radius"] = np.asarray(radius, dtype=np.float64)
        pl.add_mesh(
            poly,
            scalars="radius",
            cmap="viridis",
            line_width=line_width,
            scalar_bar_args={"title": "Radius (mm)"},
        )
        return

    if config.DEBUG_VIS_COLOR_BY == "strahler" and strahler is not None:
        vals = list(strahler)
        if vals and all(v is not None for v in vals):
            arr = np.asarray(vals, dtype=np.int32)
            lo, hi = int(arr.min()), int(arr.max())
            n = hi - lo + 1
            poly.cell_data["strahler"] = arr
            pl.add_mesh(
                poly,
                scalars="strahler",
                cmap="tab10",
                n_colors=max(n, 1),
                clim=[lo - 0.5, hi + 0.5],
                line_width=line_width,
                scalar_bar_args={"title": "Strahler order", "n_labels": n, "fmt": "%.0f"},
            )
            return

    poly.cell_data["seg_idx"] = np.asarray(seg_idx, dtype=np.int32)
    pl.add_mesh(
        poly,
        scalars="seg_idx",
        cmap="turbo",
        line_width=line_width,
        scalar_bar_args={"title": "Segment idx"},
    )


# ── Mesh / point cloud ────────────────────────────────────────────────────────


def debug_show_mesh(mesh: pv.PolyData, title: str = "Mesh", show_edges: bool = False) -> None:
    if not config.DEBUG_VIS:
        return
    print(f"  [VIS] {title}...")
    pl = pv.Plotter()
    pl.set_background("white")
    if mesh is not None and mesh.n_points > 0:
        pl.add_mesh(
            mesh,
            color="coral",
            show_edges=show_edges,
            edge_color="gray",
            opacity=1.0,
            smooth_shading=True,
        )
    info = f"{mesh.n_points:,} verts, {mesh.n_cells:,} faces"
    if hasattr(mesh, "is_manifold"):
        info += f" | Manifold: {mesh.is_manifold}"
    pl.add_title(f"{title}\n{info}")
    _show_plotter(pl, title)


# ── Capsule tree ──────────────────────────────────────────────────────────────


def debug_show_capsule_tree(
    cap_starts: np.ndarray,
    cap_ends: np.ndarray,
    cap_radii_start: np.ndarray,
    cap_radii_end: np.ndarray,
    cap_seg_idx: np.ndarray,
    nodes: dict[int, tuple],
    segments: list[dict[str, Any]],
    adj_matrix: np.ndarray | None = None,
    title: str = "Capsule Tree",
) -> None:
    """Capsule centerlines (one colour per seg_idx) + bif/endpoint spheres
    + optional adjacency links between segment midpoints."""
    if not config.DEBUG_VIS:
        return
    print(f"  [VIS] {title}...")
    pl = pv.Plotter()
    pl.set_background("white")

    n_caps = len(cap_starts)
    if n_caps == 0:
        print("  [VIS] no capsules to draw")
        return

    line_pts = np.empty((2 * n_caps, 3), dtype=np.float64)
    line_pts[0::2] = cap_starts
    line_pts[1::2] = cap_ends
    line_conn = np.empty((n_caps, 3), dtype=np.int64)
    line_conn[:, 0] = 2
    line_conn[:, 1] = np.arange(0, 2 * n_caps, 2)
    line_conn[:, 2] = np.arange(1, 2 * n_caps, 2)
    poly = pv.PolyData(line_pts, lines=line_conn.ravel())
    cap_strahler = [segments[int(si)].get("strahler") for si in cap_seg_idx]
    _add_scalar_lines(
        pl,
        poly,
        seg_idx=cap_seg_idx,
        radius=0.5 * (np.asarray(cap_radii_start) + np.asarray(cap_radii_end)),
        line_width=2,
        strahler=cap_strahler,
    )

    bif_pts: list[np.ndarray] = []
    bif_radii: list[float] = []
    end_pts: list[np.ndarray] = []
    for nid, ndata in nodes.items():
        x, y, z, coord = ndata
        pos = np.array([x / 1000.0, y / 1000.0, z / 1000.0])
        if coord == 1:
            end_pts.append(pos)
        elif coord >= 2:
            seg_ids_here = [
                i for i, s in enumerate(segments) if s["node1"] == nid or s["node2"] == nid
            ]
            max_r = 0.05
            for si in seg_ids_here:
                mask = cap_seg_idx == si
                if mask.any():
                    max_r = max(
                        max_r,
                        float(cap_radii_start[mask].max()),
                        float(cap_radii_end[mask].max()),
                    )
            bif_pts.append(pos)
            bif_radii.append(max_r)

    if bif_pts:
        cloud = pv.PolyData(np.asarray(bif_pts))
        cloud["radius"] = np.asarray(bif_radii)
        glyph_src = pv.Sphere(radius=1.0, theta_resolution=12, phi_resolution=12)
        glyphs = cloud.glyph(orient=False, scale="radius", geom=glyph_src, factor=1.0)
        pl.add_mesh(glyphs, color="red", opacity=0.85, label=f"Bifurcations (n={len(bif_pts)})")
    if end_pts:
        cloud = pv.PolyData(np.asarray(end_pts))
        ep_r = float(np.mean(bif_radii)) if bif_radii else 0.05
        glyph_src = pv.Sphere(radius=ep_r * 0.6, theta_resolution=10, phi_resolution=10)
        glyphs = cloud.glyph(orient=False, scale=False, geom=glyph_src)
        pl.add_mesh(glyphs, color="green", opacity=0.9, label=f"Endpoints (n={len(end_pts)})")

    if adj_matrix is not None:
        n_segs = adj_matrix.shape[0]
        seg_mids = np.zeros((n_segs, 3), dtype=np.float64)
        seg_has = np.zeros(n_segs, dtype=bool)
        for si in range(n_segs):
            mask = cap_seg_idx == si
            if mask.any():
                seg_mids[si] = (0.5 * (cap_starts[mask] + cap_ends[mask])).mean(axis=0)
                seg_has[si] = True
        link_pts: list[np.ndarray] = []
        link_conn: list[int] = []
        idx_off = 0
        for i in range(n_segs):
            if not seg_has[i]:
                continue
            for j in range(i + 1, n_segs):
                if adj_matrix[i, j] and seg_has[j]:
                    link_pts.append(seg_mids[i])
                    link_pts.append(seg_mids[j])
                    link_conn.extend([2, idx_off, idx_off + 1])
                    idx_off += 2
        if link_pts:
            link_poly = pv.PolyData(np.asarray(link_pts), lines=link_conn)
            pl.add_mesh(link_poly, color="black", line_width=1, opacity=0.35, label="Adjacency")

    pl.add_legend(bcolor="white")
    pl.add_title(title)
    _show_plotter(pl, title)


# ── SDF iso preview ──────────────────────────────────────────────────────────


def debug_show_sdf_preview(
    sdf_grid: np.ndarray,
    bbox_min: np.ndarray,
    bbox_max: np.ndarray,
    voxel_size: float,
    path_vol: np.ndarray | None = None,
    iso_levels: tuple[float, float, float] = (-0.05, 0.0, 0.05),
    title: str = "SDF Preview (pre-MC)",
) -> None:
    """Three-iso surface render to expose fusion artifacts before the mesher.

    Negative iso = interior (red, mostly opaque). 0 = MC zero surface (gold).
    Positive iso = exterior shell (blue, low opacity). When ``path_vol`` is
    provided, the zero iso is coloured by evaluation path (1/3) for debugging.
    """
    if not config.DEBUG_VIS:
        return
    print(f"  [VIS] {title} (iso={list(iso_levels)})...")
    dims = sdf_grid.shape
    grid = pv.ImageData()
    grid.dimensions = tuple(dims)
    grid.origin = tuple(bbox_min)
    grid.spacing = (voxel_size, voxel_size, voxel_size)
    grid.point_data["sdf"] = sdf_grid.ravel(order="F").astype(np.float32)
    if path_vol is not None:
        grid.point_data["path"] = path_vol.ravel(order="F").astype(np.uint8)

    pl = pv.Plotter()
    pl.set_background("white")
    cmap = {
        "neg": ("firebrick", 0.85),
        "zero": ("gold", 0.95),
        "pos": ("royalblue", 0.30),
    }
    for iso in iso_levels:
        try:
            surf = grid.contour(isosurfaces=[float(iso)], scalars="sdf")
        except Exception as exc:
            print(f"    [WARN] iso={iso}: {exc}")
            continue
        if surf.n_points == 0:
            continue
        if iso < -1e-9:
            colour, opacity = cmap["neg"]
            label = f"iso {iso:+.3f} (interior)"
        elif iso > 1e-9:
            colour, opacity = cmap["pos"]
            label = f"iso {iso:+.3f} (exterior)"
        else:
            colour, opacity = cmap["zero"]
            label = "iso  0.000 (MC surface)"
        if path_vol is not None and abs(iso) < 1e-9 and "path" in surf.point_data:
            pl.add_mesh(
                surf,
                scalars="path",
                cmap="viridis",
                opacity=opacity,
                clim=(0, 4),
                show_scalar_bar=True,
                scalar_bar_args={"title": "eval path"},
                label=label,
            )
        else:
            pl.add_mesh(surf, color=colour, opacity=opacity, label=label)
        print(f"    iso {iso:+.3f}: {surf.n_points:,} verts, {surf.n_cells:,} faces")

    pl.add_legend(bcolor="white")
    pl.add_title(title)
    _show_plotter(pl, title)


# ── Blend paths ──────────────────────────────────────────────────────────────


def debug_show_blend_paths(
    path_vol: np.ndarray | None,
    bbox_min: np.ndarray,
    voxel_size: float,
    valid_splines: list[dict[str, Any]] | None = None,
    refine_threshold: float | None = None,
    blend_weight_vol: np.ndarray | None = None,
    max_points: int = 200_000,
    title: str = "SDF Evaluation Paths",
) -> None:
    """Per-voxel SDF evaluation strategy point cloud."""
    if not config.DEBUG_VIS:
        return
    if path_vol is None:
        print("  [VIS] no path_vol — skipping blend-path viz")
        return
    print(f"  [VIS] {title}...")
    pl = pv.Plotter()
    pl.set_background("white")

    nz = np.argwhere(path_vol != 0)
    if nz.size == 0:
        print("    no in-band voxels")
        return
    if len(nz) > max_points:
        sel = np.random.default_rng(0).choice(len(nz), max_points, replace=False)
        nz = nz[sel]
    coords = bbox_min[None, :] + nz.astype(np.float64) * voxel_size
    paths = path_vol[nz[:, 0], nz[:, 1], nz[:, 2]].astype(np.int32)

    table = {
        1: ("royalblue", "FAST hard-min (interior)"),
        2: ("seagreen", "SLOW no-blend (fallback)"),
        3: ("crimson", "SLOW smooth-min (blended)"),
    }
    for p_val, (col, lbl) in table.items():
        mask = paths == p_val
        if not mask.any():
            continue
        sub = pv.PolyData(coords[mask])
        if p_val == 3 and blend_weight_vol is not None:
            nz_sub = nz[mask]
            w = blend_weight_vol[nz_sub[:, 0], nz_sub[:, 1], nz_sub[:, 2]]
            sub["blend_w"] = w
            pl.add_mesh(
                sub,
                scalars="blend_w",
                cmap="Reds",
                point_size=4,
                render_points_as_spheres=True,
                scalar_bar_args={"title": "smooth-min weight"},
                label=f"{lbl} (n={int(mask.sum()):,})",
            )
        else:
            pl.add_mesh(
                sub,
                color=col,
                point_size=3,
                render_points_as_spheres=True,
                label=f"{lbl} (n={int(mask.sum()):,})",
            )

    if valid_splines is not None:
        thresh = refine_threshold
        if thresh is None:
            thresh = voxel_size * config.MULTIRES_VOXEL_FACTOR
        centres: list[np.ndarray] = []
        radii: list[float] = []
        for sp in valid_splines:
            if sp is None:
                continue
            avg_r = 0.5 * (sp["start_radius"] + sp["end_radius"])
            if avg_r < thresh:
                mid_pt = sp["cs_pos"](np.array([sp["L"] / 2.0]))[0]
                centres.append(mid_pt)
                radii.append(max(avg_r * 4.0, voxel_size * 4))
        if centres:
            rc = pv.PolyData(np.asarray(centres))
            rc["radius"] = np.asarray(radii)
            ball = pv.Sphere(radius=1.0, theta_resolution=12, phi_resolution=12)
            glyphs = rc.glyph(orient=False, scale="radius", geom=ball, factor=1.0)
            pl.add_mesh(
                glyphs,
                color="magenta",
                opacity=0.15,
                label=f"Refined smooth-min zones (n={len(centres)})",
            )

    pl.add_legend(bcolor="white")
    pl.add_title(title)
    _show_plotter(pl, title)


# ── Tapered SDF capsules as tubes ────────────────────────────────────────────


def debug_show_capsule_tubes(
    capsules: Any,
    n_sides: int = 12,
    title: str = "SDF Capsule Tubes",
) -> None:
    """Render each segment's capsule chain as a tapered tube via
    ``vtkTubeFilter`` with a per-vertex radius scalar.

    Unlike :func:`debug_show_capsule_tree` (which draws centerline lines),
    this matches the *actual* SDF primitive surface a single capsule
    would generate (sans smooth-min blending).
    """
    if not config.DEBUG_VIS:
        return
    print(f"  [VIS] {title}...")
    if capsules.n == 0:
        print("  [VIS] no capsules to draw")
        return

    starts = np.asarray(capsules.starts, dtype=np.float64)
    ends = np.asarray(capsules.ends, dtype=np.float64)
    r_s = np.asarray(capsules.radii_start, dtype=np.float64)
    r_e = np.asarray(capsules.radii_end, dtype=np.float64)
    seg_idx = np.asarray(capsules.seg_idx, dtype=np.int64)

    pl = pv.Plotter()
    pl.set_background("white")

    seg_ids = np.unique(seg_idx)
    n_tubes = 0
    n_skipped = 0
    for si in seg_ids:
        mask = seg_idx == si
        if not mask.any():
            continue
        # Capsules within a segment are consecutive samples, so the per-segment
        # polyline goes start[0] -> end[0] -> end[1] -> ... -> end[-1].
        seg_starts = starts[mask]
        seg_ends = ends[mask]
        seg_rs = r_s[mask]
        seg_re = r_e[mask]
        order = np.argsort(np.where(mask)[0])  # already sorted by capsule build order
        seg_starts = seg_starts[order]
        seg_ends = seg_ends[order]
        seg_rs = seg_rs[order]
        seg_re = seg_re[order]

        line_pts = np.vstack([seg_starts[0:1], seg_ends])
        line_rad = np.concatenate([seg_rs[0:1], seg_re])
        if len(line_pts) < 2 or np.any(line_rad <= 0):
            n_skipped += 1
            continue

        n_pts = len(line_pts)
        line_conn = np.concatenate([[n_pts], np.arange(n_pts, dtype=np.int64)])
        poly = pv.PolyData(line_pts, lines=line_conn)
        poly.point_data["radius"] = line_rad.astype(np.float32)
        try:
            tube = poly.tube(scalars="radius", radius_factor=1.0, n_sides=n_sides)
        except Exception as e:
            print(f"    [VIS][WARN] tube failed for seg {si}: {e}")
            n_skipped += 1
            continue
        if tube.n_cells == 0:
            n_skipped += 1
            continue
        tube.cell_data["seg_idx"] = np.full(tube.n_cells, si, dtype=np.int32)
        pl.add_mesh(
            tube,
            scalars="seg_idx",
            cmap="turbo",
            clim=(0, int(seg_ids.max())),
            show_scalar_bar=(n_tubes == 0),
            scalar_bar_args={"title": "Segment idx"} if n_tubes == 0 else None,
            smooth_shading=True,
        )
        n_tubes += 1

    info = f"{n_tubes} segment tubes ({capsules.n} capsules)"
    if n_skipped:
        info += f" — {n_skipped} skipped"
    pl.add_title(f"{title}\n{info}")
    _show_plotter(pl, title)


# ── Raw input points + per-point cross-section contours ──────────────────────


def debug_show_raw_data_contours(
    nodes: dict[int, tuple],
    points: dict[int, tuple],
    segments: list[dict[str, Any]],
    n_circle_pts: int = 16,
    title: str = "Raw Data Contours",
) -> None:
    """Render the raw Amira input as point dots + per-point circular
    cross-sections perpendicular to the local centerline tangent.

    Coordinates and thicknesses come straight from the XML (mm conversion
    applied), so this is the ground-truth source data before any
    smoothing.
    """
    if not config.DEBUG_VIS:
        return
    print(f"  [VIS] {title}...")
    if not segments or not points:
        print("  [VIS] no raw data to draw")
        return

    # Lazy import to avoid a circular dependency between viz and splines.
    from .splines import compute_frenet_frame

    pl = pv.Plotter()
    pl.set_background("white")

    ring_pts_all: list[np.ndarray] = []
    ring_lines: list[int] = []
    ring_segidx: list[int] = []
    ring_radius: list[float] = []
    ring_strahler: list = []
    dot_pts: list[np.ndarray] = []
    offset = 0
    prev_normal: np.ndarray | None = None

    for s_idx, seg in enumerate(segments):
        pids = seg.get("point_ids", [])
        coords = [points[p][:3] for p in pids if p in points]
        thick = [points[p][3] for p in pids if p in points]
        if len(coords) < 2:
            continue
        coords_mm = np.asarray(coords, dtype=np.float64) / 1000.0
        radii_mm = np.asarray(thick, dtype=np.float64) / 1000.0 * config.RADIUS_SCALE
        n_p = len(coords_mm)
        dot_pts.append(coords_mm)

        prev_normal = None
        for i in range(n_p):
            if i == 0:
                tang = coords_mm[1] - coords_mm[0]
            elif i == n_p - 1:
                tang = coords_mm[-1] - coords_mm[-2]
            else:
                tang = coords_mm[i + 1] - coords_mm[i - 1]
            if np.linalg.norm(tang) < 1e-12 or radii_mm[i] <= 0:
                continue
            t_hat, n_hat, b_hat = compute_frenet_frame(tang, prev_normal)
            prev_normal = n_hat
            theta = np.linspace(0.0, 2.0 * np.pi, n_circle_pts, endpoint=False)
            ring = (
                coords_mm[i]
                + radii_mm[i] * (np.cos(theta)[:, None] * n_hat + np.sin(theta)[:, None] * b_hat)
            )
            ring_pts_all.append(ring)
            # Closed polyline: n_circle_pts+1 vertex indices, last = first.
            ring_lines.append(n_circle_pts + 1)
            ring_lines.extend(range(offset, offset + n_circle_pts))
            ring_lines.append(offset)
            ring_segidx.append(s_idx)
            ring_radius.append(float(radii_mm[i]))
            ring_strahler.append(seg.get("strahler"))
            offset += n_circle_pts

    if ring_pts_all:
        all_pts = np.vstack(ring_pts_all)
        poly = pv.PolyData(all_pts, lines=np.asarray(ring_lines, dtype=np.int64))
        # One scalar per ring (cell): each closed ring is one polyline cell, so
        # both seg_idx and the ring radius are per-cell values.
        _add_scalar_lines(
            pl,
            poly,
            seg_idx=ring_segidx,
            radius=ring_radius,
            line_width=1,
            strahler=ring_strahler,
        )

    if dot_pts:
        all_dots = np.vstack(dot_pts)
        dot_poly = pv.PolyData(all_dots)
        pl.add_mesh(
            dot_poly,
            color="black",
            point_size=3,
            render_points_as_spheres=True,
            label=f"Raw centerline points (n={len(all_dots):,})",
        )
        pl.add_legend(bcolor="white")

    print(
        f"  [VIS] {title}: rendering {len(ring_segidx)} cross-sections, "
        f"{len(segments)} segments, point dict has {len(points)} entries"
    )
    pl.add_title(f"{title}\n{len(ring_segidx)} cross-sections, {len(segments)} segments")
    _show_plotter(pl, title)


# ── Smoothed centerlines per segment ─────────────────────────────────────────


def debug_show_smoothed_centerlines(
    valid_splines: list[dict[str, Any]] | None,
    show_radii: bool = True,
    title: str = "Smoothed Centerlines",
) -> None:
    """Render each spline's post-smoothing centerline as a coloured polyline.

    When ``show_radii=True`` also drops a small sphere at every smoothed
    sample, scaled by the per-sample radius.
    """
    if not config.DEBUG_VIS:
        return
    if not valid_splines:
        print("  [VIS] no splines to draw")
        return
    print(f"  [VIS] {title}...")

    pl = pv.Plotter()
    pl.set_background("white")

    line_pts_all: list[np.ndarray] = []
    line_conn: list[int] = []
    line_segidx: list[int] = []
    line_strahler: list = []
    line_radius_all: list[float] = []   # per-point radius aligned with all_pts
    radius_pts: list[np.ndarray] = []
    radius_vals: list[float] = []
    offset = 0
    for sp in valid_splines:
        coords = np.asarray(sp.get("coords"), dtype=np.float64)
        radii = np.asarray(sp.get("radii", []), dtype=np.float64)
        if coords.ndim != 2 or len(coords) < 2:
            continue
        n_p = len(coords)
        line_pts_all.append(coords)
        line_conn.append(n_p)
        line_conn.extend(range(offset, offset + n_p))
        line_segidx.append(int(sp.get("seg_idx", -1)))
        line_strahler.append(sp.get("strahler"))
        if len(radii) == n_p:
            line_radius_all.extend(radii.tolist())
        offset += n_p
        if show_radii and len(radii) == n_p:
            radius_pts.append(coords)
            radius_vals.extend(radii.tolist())

    if line_pts_all:
        all_pts = np.vstack(line_pts_all)
        poly = pv.PolyData(all_pts, lines=np.asarray(line_conn, dtype=np.int64))
        # Each segment is one long polyline cell, so colour radius per *point*
        # for a smooth gradient. Only usable when every drawn segment supplied
        # matching per-point radii; otherwise fall back to seg_idx.
        pt_radius = line_radius_all if len(line_radius_all) == len(all_pts) else None
        _add_scalar_lines(
            pl,
            poly,
            seg_idx=line_segidx,
            radius=pt_radius,
            line_width=2,
            radius_on_points=True,
            strahler=line_strahler,
        )

    if show_radii and radius_pts:
        r_pts = np.vstack(radius_pts)
        r_poly = pv.PolyData(r_pts)
        r_poly["radius"] = np.asarray(radius_vals, dtype=np.float64)
        ball = pv.Sphere(radius=1.0, theta_resolution=10, phi_resolution=10)
        glyphs = r_poly.glyph(orient=False, scale="radius", geom=ball, factor=1.0)
        pl.add_mesh(glyphs, color="cornflowerblue", opacity=0.3, label="Radii")
        pl.add_legend(bcolor="white")

    pl.add_title(f"{title}\n{len(line_segidx)} segments")
    _show_plotter(pl, title)


# ── Detailed blend / carve / gate / smooth-min diagnostics ──────────────────


def debug_show_blend_diagnostics(
    sdf_volume: Any,           # SdfVolume with nb_idx + diag populated
    mesh: pv.PolyData | None,
    grid: Any,                 # Grid with .x .y .z .voxel_size
    bif: Any = None,           # BifurcationSet with .positions .radii
    title: str = "Blend Diagnostics",
    max_points_per_layer: int = 200_000,
) -> None:
    """Overlay per-narrow-band gate / carve / smin / proximity layers onto
    the extracted mesh. Each layer is a toggleable point cloud (default off);
    user activates one or two at a time to isolate which component is
    contributing to a bif bulge or vessel-fusion bridge.

    Requires ``config.DETAILED_BLEND_DIAGNOSTIC = True`` at SDF eval time
    so the SdfVolume carries ``nb_idx`` and the ``diag`` dict.
    """
    if not config.DEBUG_VIS:
        return
    if sdf_volume is None or sdf_volume.diag is None or sdf_volume.nb_idx is None:
        print(
            "  [VIS] no detailed diag arrays — "
            "set DETAILED_BLEND_DIAGNOSTIC=True to populate them"
        )
        return

    diag = sdf_volume.diag
    nb_idx = sdf_volume.nb_idx
    print(f"  [VIS] {title}...")

    pl = pv.Plotter()
    pl.set_background("white")

    world_coords = np.column_stack([
        grid.x[nb_idx[:, 0]],
        grid.y[nb_idx[:, 1]],
        grid.z[nb_idx[:, 2]],
    ])

    if mesh is not None and mesh.n_points > 0:
        pl.add_mesh(
            mesh,
            color="coral",
            opacity=0.20,
            smooth_shading=True,
            name="mesh_bg",
        )

    if (
        bif is not None
        and getattr(bif, "positions", None) is not None
        and len(bif.positions) > 0
    ):
        for i, pos in enumerate(bif.positions):
            r = float(bif.radii[i]) * 0.5
            pl.add_mesh(
                pv.Sphere(radius=r, center=pos, theta_resolution=12, phi_resolution=12),
                color="red",
                opacity=0.5,
                name=f"bif_{i}",
            )

    # Layer specs: (diag_name, kind, palette_or_color, label).
    #   kind = "bool" for uint8 mask layers (filter to value != 0).
    #   kind = "float" for continuous fields (mask + colormap).
    layers: list[tuple[str, str, str, str]] = [
        # Gates (binary)
        ("gate_xs_active",    "bool",  "skyblue",     "cross-section gate active"),
        ("gate_bif_ball",     "bool",  "lightgreen",  "inside bif-ball (BIF_BLEND_RADIUS_FACTOR)"),
        ("gate_par_suppress", "bool",  "purple",      "parallel-rival blend suppressed"),
        ("gate_jprotect",     "bool",  "gold",        "junction-ball protect (carve shield)"),
        # Carves (binary)
        ("carve_wb_pre",      "bool",  "lightblue",   "wall-band carve PRE-gate"),
        ("carve_wb_act",      "bool",  "blue",        "wall-band carve ACTUAL push"),
        ("carve_deep_pre",    "bool",  "khaki",       "deep-merger carve PRE-gate"),
        ("carve_deep_act",    "bool",  "yellow",      "deep-merger carve ACTUAL push"),
        ("carve_adj_pre",     "bool",  "pink",        "adj-parallel carve PRE-gate"),
        ("carve_adj_act",     "bool",  "magenta",     "adj-parallel carve ACTUAL push"),
        # Continuous gates
        ("gate_proximity",    "float", "Reds",        "proximity smoothstep weight"),
        ("gate_wedge",        "float", "Greens",      "wedge_pass weight"),
        # Smooth-min
        ("final_blend_w",     "float", "Oranges",     "final blend_w applied"),
        ("smin_depression",   "float", "viridis",     "smin depression (hard - final)"),
        # T-projection
        ("t_seg_owner",       "float", "plasma",      "t along owner segment"),
        ("t_seg_rival",       "float", "plasma",      "t along rival segment"),
        # Proximity / fusion risk
        ("non_adj_min",       "float", "coolwarm",    "non-adj SDF (<0 = fusion risk)"),
        # Total carve push
        ("carve_total_push",  "float", "OrRd",        "total carve push (sdf_final - sdf_pre)"),
    ]

    actors: dict[str, Any] = {}

    def _world_pts(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray | None]:
        n = int(mask.sum())
        if n == 0:
            return np.empty((0, 3), dtype=np.float64), None
        pts = world_coords[mask]
        sel = None
        if n > max_points_per_layer:
            sel = np.random.default_rng(0).choice(n, max_points_per_layer, replace=False)
            pts = pts[sel]
        return pts, sel

    for name, kind, palette, label in layers:
        arr = diag.get(name)
        if arr is None:
            continue
        if kind == "bool":
            mask = arr.astype(bool)
            if not mask.any():
                continue
            pts, _ = _world_pts(mask)
            cloud = pv.PolyData(pts)
            actor = pl.add_mesh(
                cloud,
                color=palette,
                point_size=4,
                render_points_as_spheres=True,
                opacity=0.7,
                name=name,
            )
        else:
            finite = np.isfinite(arr)
            if name == "non_adj_min":
                mask = finite & (arr < 0.0)
            elif name in ("gate_proximity", "gate_wedge", "final_blend_w"):
                mask = finite & (arr > 1e-3)
            elif name == "smin_depression":
                mask = finite & (np.abs(arr) > 1e-4)
            elif name == "carve_total_push":
                mask = finite & (np.abs(arr) > 1e-4)
            elif name in ("t_seg_owner", "t_seg_rival"):
                # Only show where the t-projection actually computed t (NaN
                # elsewhere means the gate wasn't active for that voxel).
                mask = finite
            else:
                mask = finite
            if not mask.any():
                continue
            pts, sel = _world_pts(mask)
            vals = arr[mask]
            if sel is not None:
                vals = vals[sel]
            cloud = pv.PolyData(pts)
            cloud[name] = vals
            actor = pl.add_mesh(
                cloud,
                scalars=name,
                cmap=palette,
                point_size=4,
                render_points_as_spheres=True,
                opacity=0.85,
                name=name,
                show_scalar_bar=False,
            )
        try:
            actor.SetVisibility(False)
        except Exception:
            pass
        actors[name] = (actor, label, int(mask.sum()))

    # Checkbox widgets per layer.
    def _make_callback(actor):
        def cb(flag: bool) -> None:
            try:
                actor.SetVisibility(bool(flag))
            except Exception:
                pass
        return cb

    button_y = 10
    button_size = 20
    button_gap = 26
    for name, (actor, label, count) in actors.items():
        try:
            pl.add_checkbox_button_widget(
                _make_callback(actor),
                value=False,
                position=(10, button_y),
                size=button_size,
                border_size=1,
                color_on="green",
                color_off="lightgray",
            )
            pl.add_text(
                f"{name}  n={count:,}  ({label})",
                position=(10 + button_size + 8, button_y + 2),
                font_size=8,
                color="black",
            )
        except Exception as e:
            print(f"  [VIS][WARN] checkbox failed for '{name}': {e}")
            try:
                actor.SetVisibility(True)
            except Exception:
                pass
        button_y += button_gap

    pl.add_title(title)
    _show_plotter(pl, title)


__all__ = [
    "debug_show_mesh",
    "debug_show_capsule_tree",
    "debug_show_sdf_preview",
    "debug_show_blend_paths",
    "debug_show_blend_diagnostics",
    "debug_show_capsule_tubes",
    "debug_show_raw_data_contours",
    "debug_show_smoothed_centerlines",
]
