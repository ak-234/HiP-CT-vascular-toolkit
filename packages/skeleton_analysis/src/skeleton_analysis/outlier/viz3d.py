"""Optional interactive 3-D QC for oblique vessel correction.

Renders a flagged vessel's sub-volume together with its centreline (and,
optionally, the oblique cutting planes) so the correction can be eyeballed — the
opt-in Python counterpart of the MATLAB ``getboundingbox.m`` / ``vizualisation.m``
(``volshow``/``viewer3d``) QC. This is **inspection only**: skip/resize decisions
stay reproducible via the detection parameters and
:func:`skeleton_analysis.outlier.correct.apply_manual_plane_selection`.

Uses **PyVista** (the ``[viz3d]`` extra), imported lazily so the core package
never requires it. ``off_screen=True`` renders headlessly to a PNG (for tests /
batch capture).
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np


def segment_bbox(
    centreline: np.ndarray, half_pad: int, shape: Tuple[int, int, int]
) -> Tuple[np.ndarray, np.ndarray]:
    """Padded, clamped crop indices around a centreline's voxel bounding box.

    ``centreline`` is ``(P, 3)`` in ``(z, y, x)`` voxel coordinates matching a
    ``vol[z, y, x]`` array of the given ``shape``. Returns ``(lo, hi)`` integer
    index arrays (``hi`` exclusive) suitable for ``vol[lo[0]:hi[0], ...]``.
    """
    c = np.asarray(centreline, dtype=float)
    shape = np.asarray(shape, dtype=np.int64)
    lo = np.floor(c.min(axis=0)).astype(np.int64) - half_pad
    hi = np.ceil(c.max(axis=0)).astype(np.int64) + half_pad + 1
    lo = np.clip(lo, 0, shape)
    hi = np.clip(hi, 0, shape)
    return lo, hi


def show_segment_volume(
    volume: np.ndarray,
    centreline: np.ndarray,
    half_pad: int = 8,
    threshold: float = 0.5,
    show_planes: bool = False,
    plane_stride: int = 10,
    plane_radius: Optional[float] = None,
    off_screen: bool = False,
    screenshot: Optional[str] = None,
    title: Optional[str] = None,
) -> Optional[str]:
    """Render a vessel sub-volume + centreline (+ optional cutting planes).

    Parameters
    ----------
    volume : 3-D array (z, y, x)
        The segmentation / image volume.
    centreline : (P, 3) array
        Centreline points in ``(z, y, x)`` voxel coordinates (as fed to the
        oblique slicer).
    half_pad : int
        Voxels of padding around the centreline bounding box.
    threshold : float
        Iso-level for the vessel surface (``volume > threshold``).
    show_planes : bool
        Draw the oblique cutting-plane discs (every ``plane_stride`` points).
    off_screen / screenshot :
        Render headlessly to ``screenshot`` (PNG) instead of opening a window.

    Returns the screenshot path when ``off_screen`` + ``screenshot`` are set,
    else ``None``. Requires the ``[viz3d]`` extra (PyVista).
    """
    import pyvista as pv  # lazy: [viz3d] extra

    volume = np.asarray(volume)
    c = np.asarray(centreline, dtype=float)
    lo, hi = segment_bbox(c, half_pad, volume.shape)
    sub = volume[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]].astype(float)
    # Centreline in the cropped frame, as (x, y, z) for PyVista point geometry.
    c_local = c - lo  # (P,3) in (z,y,x)
    pts_xyz = c_local[:, ::-1]  # -> (x, y, z)

    pl = pv.Plotter(off_screen=off_screen, title=title or "vessel QC")
    pl.set_background("white")

    # Vessel iso-surface (ImageData is indexed x,y,z, so transpose z,y,x -> x,y,z).
    grid = pv.ImageData(dimensions=np.array(sub.shape)[::-1])
    grid.point_data["v"] = np.transpose(sub, (2, 1, 0)).ravel(order="F")
    if float(sub.max()) > threshold:
        surf = grid.contour([threshold], scalars="v")
        if surf.n_points:
            pl.add_mesh(surf, color=(0.7, 0.75, 0.95), opacity=0.35,
                        smooth_shading=True)

    # Centreline polyline + point spheres.
    if len(pts_xyz) >= 2:
        line = pv.lines_from_points(pts_xyz)
        pl.add_mesh(line, color="red", line_width=4)
    pl.add_mesh(pv.PolyData(pts_xyz), color="red", point_size=8,
                render_points_as_spheres=True)

    # Oblique cutting planes.
    if show_planes and len(pts_xyz) >= 2:
        r = float(plane_radius) if plane_radius else max(half_pad, 4)
        for k in range(0, len(pts_xyz) - 1, max(plane_stride, 1)):
            tangent = c_local[k] - c_local[k + 1]  # (z,y,x)
            if np.linalg.norm(tangent) == 0:
                continue
            normal_xyz = tangent[::-1]
            disc = pv.Disc(center=pts_xyz[k], normal=normal_xyz,
                           inner=0.0, outer=r, r_res=1, c_res=24)
            pl.add_mesh(disc, color=(0.2, 0.7, 0.2), opacity=0.4)

    if title:
        pl.add_text(title, font_size=10, color="black")

    if off_screen:
        pl.show(screenshot=screenshot, auto_close=True)
        pl.close()
        return screenshot
    pl.show()
    return None
