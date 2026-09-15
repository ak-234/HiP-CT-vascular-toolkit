"""Cross-sections of the reconstructed lumen surface at raw-image z planes.

The full surface is ~2 M triangles, so a plane cut across the whole mesh is far too
slow to do per slice. Instead the mesh is clipped once to the region of interest and
the small residue is cut repeatedly.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pyvista as pv


def load_surface(path: str | Path, scale: float = 1000.0) -> pv.PolyData:
    """Read the surface and scale it into micrometres (STL files here are in mm)."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"surface not found: {path}")
    mesh = pv.read(str(path))
    if scale != 1.0:
        mesh = mesh.scale(scale, inplace=False)
    return mesh


def clip_to_roi(mesh: pv.PolyData, bounds_um, cell_centers=None) -> pv.PolyData:
    """Keep whole cells whose centroid lies inside ``(xmin, xmax, ymin, ymax, zmin, zmax)``.

    Selecting on precomputed centroids is ~40x faster than ``clip_box(crinkle=True)`` on
    a 2 M-triangle mesh, and keeping cells whole means the cut lines are unaffected by
    the ROI boundary.
    """
    cc = np.asarray(mesh.cell_centers().points) if cell_centers is None else cell_centers
    b = np.asarray(bounds_um, dtype=np.float64)
    sel = np.all((cc >= b[0::2]) & (cc <= b[1::2]), axis=1)
    idx = np.flatnonzero(sel)
    if idx.size == 0:
        return pv.PolyData()
    return mesh.extract_cells(idx).extract_surface()


def _polylines_from_cut(cut: pv.PolyData) -> list[np.ndarray]:
    """Split a cut result into ordered (N, 3) polylines in micrometres."""
    if cut.n_points == 0 or cut.n_cells == 0:
        return []
    stripped = cut.strip(join=True)
    pts = np.asarray(stripped.points, dtype=np.float64)
    lines = stripped.lines
    out: list[np.ndarray] = []
    i = 0
    while i < len(lines):
        n = int(lines[i])
        ids = lines[i + 1 : i + 1 + n]
        if n >= 2:
            out.append(pts[ids])
        i += 1 + n
    return out


def slice_at_z(mesh: pv.PolyData, z_um: float) -> list[np.ndarray]:
    """Contours of ``mesh`` on the plane z = ``z_um``, as a list of (N, 3) um polylines."""
    zmin, zmax = mesh.bounds[4], mesh.bounds[5]
    if not (zmin <= z_um <= zmax):
        return []
    try:
        cut = mesh.slice(normal="z", origin=(0.0, 0.0, float(z_um)))
    except Exception:
        return []
    return _polylines_from_cut(cut)


class SurfaceSlicer:
    """Caches an ROI-clipped copy of the surface and cuts z planes out of it."""

    def __init__(self, mesh_um: pv.PolyData):
        self.mesh = mesh_um
        self._roi = None
        self._clipped: pv.PolyData | None = None
        self._centers = None

    @property
    def cell_centers(self) -> np.ndarray:
        if self._centers is None:
            self._centers = np.asarray(self.mesh.cell_centers().points)
        return self._centers

    def set_roi(self, bounds_um) -> int:
        """Clip to a region of interest. Returns the triangle count retained."""
        bounds_um = tuple(float(v) for v in bounds_um)
        if self._roi == bounds_um and self._clipped is not None:
            return self._clipped.n_cells
        self._clipped = clip_to_roi(self.mesh, bounds_um, self.cell_centers)
        self._roi = bounds_um
        return self._clipped.n_cells

    def contours(self, z_um: float) -> list[np.ndarray]:
        target = self._clipped if self._clipped is not None else self.mesh
        if target.n_cells == 0:
            return []
        return slice_at_z(target, z_um)
