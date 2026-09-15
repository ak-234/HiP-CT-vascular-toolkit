"""Repair the voxel mask, not the graph.

Nothing on this machine did this before: the whole pipeline treats the
segmentation as read-only input and every existing "reconnection" tool works on
the centreline graph. But a graph bridge and a mask bridge are different claims --
the graph says *a vessel runs here*, the mask says *these voxels are lumen* -- and
the second is what a re-skeletonisation, a clDice score or a volume measurement
will read.

Four operations, all on an ROI rather than the 2.34 GB volume:

:func:`components`      label and measure, so small debris can be told from vessels
:func:`cull_small`      drop components below a size, the destructive half of repair
:func:`close_gaps`      morphological closing, for breaks a voxel or two wide
:func:`paint_bridges`   rasterise accepted centreline bridges as tapered capsules

The last is the one that matters. It is the mask counterpart of applying a
:class:`~.candidates.Bridge`: the same geometry that becomes a new graph segment
is burned into the mask, so the two stay consistent. Sphere-painting a centreline
is what ``generate_skeleton_graph_volume.py`` does; this tapers between
the two endpoint radii instead of using one radius per point, because a bridge
usually spans a step in calibre.

:func:`report` wraps ``skeleton_analysis``'s morphometrics so a repair can be
checked for the thing that actually goes wrong -- a "fix" that quietly merged two
vessels that were never connected shows up as a drop in component count with no
corresponding change in Euler number.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ComponentStats:
    labels: np.ndarray  # (Z, Y, X) int32, 0 = background
    sizes: np.ndarray  # (n_components + 1,) voxel counts, index 0 unused
    n: int

    def order_by_size(self) -> np.ndarray:
        """Component labels, largest first."""
        return np.argsort(self.sizes[1:])[::-1] + 1


def components(mask: np.ndarray, connectivity: int = 3) -> ComponentStats:
    """Label connected components of a binary mask.

    `connectivity` 3 means 26-neighbour in 3D, which is the right choice for
    vessels: an 18- or 6-connected labelling splits a diagonal capillary into
    beads.
    """
    from scipy import ndimage

    mask = np.asarray(mask, dtype=bool)
    structure = ndimage.generate_binary_structure(3, connectivity)
    labels, n = ndimage.label(mask, structure=structure)
    sizes = np.bincount(labels.ravel(), minlength=n + 1)
    return ComponentStats(labels=labels.astype(np.int32), sizes=sizes, n=int(n))


def cull_small(mask: np.ndarray, min_voxels: int, *, keep_largest: int = 0
               ) -> tuple[np.ndarray, int]:
    """Remove components smaller than `min_voxels`. Returns ``(mask, n_removed)``.

    `keep_largest` protects the N biggest components regardless of size, so a
    threshold chosen for debris cannot delete a genuine but small vessel tree.
    """
    stats = components(mask)
    if stats.n == 0:
        return np.asarray(mask, dtype=bool), 0

    protected = set(stats.order_by_size()[:keep_largest].tolist()) if keep_largest else set()
    doomed = [
        label for label in range(1, stats.n + 1)
        if stats.sizes[label] < min_voxels and label not in protected
    ]
    if not doomed:
        return np.asarray(mask, dtype=bool), 0
    out = np.asarray(mask, dtype=bool).copy()
    out[np.isin(stats.labels, doomed)] = False
    return out, len(doomed)


def close_gaps(mask: np.ndarray, radius_voxels: int = 1) -> np.ndarray:
    """Morphological closing with a ball, for breaks a voxel or two wide.

    Deliberately small by default. Closing is indiscriminate -- it will just as
    happily weld two vessels that merely pass close together as it will mend a
    break -- so it is for dropouts on the scale of the voxel, and
    :func:`paint_bridges` is for anything longer.
    """
    from scipy import ndimage

    if radius_voxels < 1:
        return np.asarray(mask, dtype=bool)
    r = int(radius_voxels)
    grid = np.ogrid[-r:r + 1, -r:r + 1, -r:r + 1]
    ball = (grid[0] ** 2 + grid[1] ** 2 + grid[2] ** 2) <= r * r
    return ndimage.binary_closing(np.asarray(mask, dtype=bool), structure=ball)


def paint_bridges(
    mask: np.ndarray,
    bridges,
    origin_um: np.ndarray,
    spacing_um: np.ndarray,
    *,
    radius_scale: float = 1.0,
) -> tuple[np.ndarray, int]:
    """Burn accepted bridges into the mask as tapered capsules.

    `origin_um` and `spacing_um` are ``(x, y, z)`` and `mask` is ``[z, y, x]``,
    matching :class:`~.probability.Roi` and ``WorldFrame``.

    Works one capsule at a time inside that capsule's own bounding box, so cost
    scales with the bridge rather than with the volume.
    """
    out = np.asarray(mask, dtype=bool).copy()
    origin = np.asarray(origin_um, dtype=np.float64)
    spacing = np.asarray(spacing_um, dtype=np.float64)
    painted = 0

    for bridge in bridges:
        if not getattr(bridge, "accepted", True):
            continue
        coords = np.asarray(bridge.coords, dtype=np.float64).reshape(-1, 3)
        radii = np.asarray(bridge.radii, dtype=np.float64).ravel() * radius_scale
        if len(coords) < 2 or len(radii) != len(coords):
            continue
        for k in range(len(coords) - 1):
            _paint_capsule(out, origin, spacing,
                           coords[k], coords[k + 1], radii[k], radii[k + 1])
        painted += 1
    return out, painted


def _paint_capsule(mask, origin, spacing, p0, p1, r0, r1) -> None:
    """Set voxels inside one tapered capsule (a cone with hemispherical caps).

    The same solid ``coronary_sdf`` sweeps for the surface, so a painted mask and
    a regenerated surface agree about where the bridge is.
    """
    r_max = max(r0, r1)
    lo_um = np.minimum(p0, p1) - r_max
    hi_um = np.maximum(p0, p1) + r_max
    lo = np.floor((lo_um - origin) / spacing).astype(int)
    hi = np.ceil((hi_um - origin) / spacing).astype(int) + 1

    shape_xyz = np.asarray(mask.shape)[::-1]
    lo = np.clip(lo, 0, shape_xyz)
    hi = np.clip(hi, 0, shape_xyz)
    if np.any(hi <= lo):
        return

    xs = np.arange(lo[0], hi[0])
    ys = np.arange(lo[1], hi[1])
    zs = np.arange(lo[2], hi[2])
    gx, gy, gz = np.meshgrid(xs, ys, zs, indexing="ij")
    pts = np.stack([gx, gy, gz], axis=-1).reshape(-1, 3) * spacing + origin

    axis = p1 - p0
    length_sq = float(axis @ axis)
    if length_sq < 1e-12:
        t = np.zeros(len(pts))
    else:
        t = np.clip(((pts - p0) @ axis) / length_sq, 0.0, 1.0)
    closest = p0 + t[:, None] * axis
    inside = np.linalg.norm(pts - closest, axis=1) <= (r0 + t * (r1 - r0))
    if not inside.any():
        return

    idx = np.stack([gx, gy, gz], axis=-1).reshape(-1, 3)[inside]
    mask[idx[:, 2], idx[:, 1], idx[:, 0]] = True


def report(mask: np.ndarray, voxel_size_um: float = 1.0) -> dict:
    """Morphometrics of a mask, for comparing before and after a repair.

    Uses ``skeleton_analysis.optimisation.region_morphometrics`` where available,
    so the numbers match what the rest of that toolchain reports. Falls back to a
    component count when it is not installed, because a repair should still be
    checkable without the optional dependency.

    Read the Euler number alongside the component count: closing that mends a
    real break lowers the component count and leaves Euler roughly alone, while
    closing that welds two unrelated vessels lowers the component count *and*
    creates a handle, which shows up as a drop in Euler.
    """
    try:
        from skeleton_analysis.optimisation import region_morphometrics

        return dict(region_morphometrics(np.asarray(mask), voxel_size=voxel_size_um))
    except Exception:  # noqa: BLE001 - optional dependency, optional detail
        stats = components(mask)
        return {
            "connected_components": stats.n,
            "voxel_count": int(np.count_nonzero(mask)),
            "volume": float(np.count_nonzero(mask) * voxel_size_um**3),
        }


def compare(before: dict, after: dict) -> str:
    """A one-line verdict on what a repair did to the mask's topology."""
    lines = []
    for key in sorted(set(before) | set(after)):
        a, b = before.get(key), after.get(key)
        if a is None or b is None:
            continue
        delta = b - a
        arrow = "" if abs(delta) < 1e-9 else f"  ({delta:+g})"
        lines.append(f"  {key:<22} {a:g} -> {b:g}{arrow}")
    return "\n".join(lines) if lines else "  (no comparable metrics)"
