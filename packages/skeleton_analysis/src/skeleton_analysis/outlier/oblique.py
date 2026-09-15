"""Oblique cross-section resampling to recompute collapsed-vessel radii.

Headless port of ``oblique_slice_vessel.m`` (+ ``give_oblique_slice_info.m``).
The interactive MATLAB workflow (``volshow``/``viewer3d`` QC, ``msgbox`` bounding
box resize, ``obliqueslice``) is replaced by:

* :func:`oblique_slice` — arbitrary-plane resampling via
  :func:`scipy.ndimage.map_coordinates` (the ``obliqueslice`` replacement);
* :func:`cross_section_radius` — measure a cross-section's radius from its
  perimeter (``perimeter/(2*pi)``) or area (``sqrt(area/pi)``) using
  :func:`skimage.measure.regionprops`;
* :func:`segment_radii_from_volume` — walk a centreline and produce a corrected
  per-point radius profile.

The perimeter estimate over-states the radius for cross-sections only a voxel or
two across (a staircase boundary inflates the perimeter); :func:`save_cross_section_png`
renders the binarised slice + boundary so this can be inspected, and ``method="area"``
gives the more robust area-based radius.

``scipy`` is a core dependency; ``scikit-image`` (and ``matplotlib`` for the debug
PNGs) are imported lazily so this module only needs the ``[image]``/``[viz]`` extras
when a radius is actually measured / a PNG saved.

Fixes vs MATLAB: no ``for i=1:1`` limiter, ``res`` and the sampling window are
explicit arguments, and there are no hard-coded paths or blocking dialogs.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np
from scipy.ndimage import map_coordinates


@dataclass
class CrossSection:
    """Details of one measured oblique cross-section (see :func:`cross_section_radius`)."""

    radius: float  # the selected radius (perimeter- or area-based) in world units
    r_perimeter: float  # perimeter/(2*pi)*res
    r_area: float  # sqrt(area/pi)*res
    perimeter: float  # region perimeter (pixels), from skimage regionprops
    area: float  # region area (pixels^2)
    label: int  # chosen region label (0 if none found)
    center: Tuple[int, int]  # slice centre pixel (row, col)
    binary: np.ndarray  # the binarised slice (slice > threshold)
    # The chosen region reaches the edge of the sampling window, so its traced
    # boundary is partly the window itself and its perimeter is meaningless.
    # Callers must grow the window and re-cut rather than record this radius.
    touches_border: bool = False
    # The centreline point landed on background: no section is owned here.
    center_background: bool = False


def robust_tangents(
    coords: np.ndarray,
    radii: Optional[np.ndarray] = None,
    *,
    spacing: float = 1.0,
    window_radii: float = 4.0,
) -> np.ndarray:
    """Radius-scaled local quadratic tangents along a centreline.

    Fits each coordinate against arclength over a window of ``window_radii``
    local radii and takes the linear term, so voxel staircase turns are
    suppressed while genuine curvature is still followed.

    **Why this exists.** The obvious estimator -- the difference to the next
    point -- measures the skeletonisation lattice rather than the vessel. On the
    LADAF_2024_28 graph (28,812 point pairs) the angle between consecutive
    one-step tangents has median **35.27 deg**, and identically so at every
    Strahler order: that is ``arccos(sqrt(2/3)) = 35.264 deg``, the angle between
    a cube face-diagonal and body-diagonal, i.e. pure staircase. This estimator
    gives median 1.88 deg on the same data.

    That matters because an oblique plane cuts an *ellipse* whose perimeter
    exceeds the true section's by roughly ``1/cos(tilt)``, and the radius here is
    ``perimeter / 2pi``. The tilt implied by the one-step tangent inflates the
    radius by a median 2.5% and a p99 of 21.6%; this estimator, by 0.01% and
    0.45%. Ported from ``hipct_seg_debug.crosssection.robust_edge_tangents``.

    ``radii`` is in the same units as ``coords``. When omitted the window falls
    back to a fixed multiple of the point spacing, which is still far better than
    a one-step difference but is not scale-aware.
    """
    xyz = np.asarray(coords, dtype=np.float64).reshape(-1, 3)
    n = len(xyz)
    if n == 0:
        return np.zeros((0, 3), dtype=np.float64)
    if n == 1:
        return np.array([[0.0, 0.0, 1.0]])
    ds = np.linalg.norm(np.diff(xyz, axis=0), axis=1)
    arc = np.concatenate([[0.0], np.cumsum(ds)])
    positive = ds[ds > 1e-9]
    step = float(np.median(positive)) if len(positive) else float(spacing)
    rr = (np.asarray(radii, dtype=np.float64).reshape(-1)
          if radii is not None else np.full(n, np.nan))

    out = np.zeros_like(xyz)
    for i in range(n):
        r = rr[i] if i < len(rr) and np.isfinite(rr[i]) and rr[i] > 0 else step
        width = max(float(window_radii) * float(r), 6.0 * step)
        ids = np.flatnonzero(np.abs(arc - arc[i]) <= width)
        if len(ids) < 3:
            ids = np.sort(np.argsort(np.abs(arc - arc[i]))[: min(5, n)])
        x = arc[ids] - arc[i]
        degree = 2 if len(ids) >= 3 and np.ptp(x) > 1e-9 else 1
        try:
            deriv = np.array(
                [np.polynomial.polynomial.polyfit(x, xyz[ids, ax], degree)[1]
                 for ax in range(3)],
                dtype=np.float64,
            )
        except (ValueError, np.linalg.LinAlgError):
            deriv = xyz[min(i + 1, n - 1)] - xyz[max(i - 1, 0)]
        norm = float(np.linalg.norm(deriv))
        if norm < 1e-9:
            deriv = xyz[min(i + 1, n - 1)] - xyz[max(i - 1, 0)]
            norm = float(np.linalg.norm(deriv))
        out[i] = deriv / norm if norm >= 1e-9 else np.array([0.0, 0.0, 1.0])
    # Independent per-point fits can disagree on sign; make the field continuous.
    for i in range(1, n):
        if float(np.dot(out[i - 1], out[i])) < 0:
            out[i] *= -1.0
    return out


def _plane_basis(normal: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Two orthonormal vectors spanning the plane perpendicular to ``normal``."""
    n = np.asarray(normal, dtype=float)
    norm = np.linalg.norm(n)
    if norm == 0:
        raise ValueError("normal vector must be non-zero")
    n = n / norm
    seed = np.array([1.0, 0.0, 0.0]) if abs(n[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    u = np.cross(n, seed)
    u /= np.linalg.norm(u)
    v = np.cross(n, u)
    return u, v


def oblique_slice(
    volume: np.ndarray,
    point,
    normal,
    half_size: int = 20,
    spacing: float = 1.0,
    order: int = 1,
    cval: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Resample a square oblique slice of ``volume`` through ``point``.

    The slice lies in the plane through ``point`` with the given ``normal``
    (all in volume index coordinates). Returns ``(slice2d, coords)`` where
    ``slice2d`` has shape ``(2*half_size+1, 2*half_size+1)`` and ``coords`` is the
    matching ``(H, W, 3)`` array of sampled index coordinates. The slice centre
    (``half_size, half_size``) corresponds to ``point``.
    """
    volume = np.asarray(volume)
    point = np.asarray(point, dtype=float)
    u, v = _plane_basis(normal)

    rng = np.arange(-half_size, half_size + 1) * spacing
    ii, jj = np.meshgrid(rng, rng, indexing="ij")
    coords = point[None, None, :] + ii[..., None] * u + jj[..., None] * v  # (H,W,3)

    sample = [coords[..., 0].ravel(), coords[..., 1].ravel(), coords[..., 2].ravel()]
    flat = map_coordinates(volume, sample, order=order, mode="constant", cval=cval)
    slice2d = flat.reshape(ii.shape)
    return slice2d, coords


def cross_section_radius(
    slice2d: np.ndarray,
    res: float = 1.0,
    center: Optional[Tuple[int, int]] = None,
    threshold: float = 0.0,
    method: str = "perimeter",
    return_details: bool = False,
    require_center: bool = True,
) -> Union[float, CrossSection]:
    """Radius of the vessel cross-section at the centre of ``slice2d``.

    Binarises the slice (``> threshold``), labels connected regions and selects
    the region containing (or nearest to) ``center``. Two radius estimates are
    computed: ``r_perimeter = perimeter/(2*pi)*res`` and ``r_area =
    sqrt(area/pi)*res``. ``method`` (``"perimeter"`` or ``"area"``) selects which
    is returned. With ``return_details=True`` a :class:`CrossSection` is returned
    instead of a float. Returns NaN (or a NaN-filled ``CrossSection``) if no
    region is found. Requires the ``[image]`` extra.

    ``require_center`` (default True) returns NaN when the slice centre falls on
    background instead of guessing the nearest region, which on a coronary tree
    may belong to a different vessel entirely. Pass False for the historical
    behaviour. The returned ``CrossSection`` also reports ``touches_border``: a
    clipped section's perimeter is partly the window boundary, so callers should
    grow the window rather than record that radius.
    """
    from skimage.measure import label, regionprops  # lazy: [image] extra

    if center is None:
        center = (slice2d.shape[0] // 2, slice2d.shape[1] // 2)
    binary = slice2d > threshold
    labels = label(binary)
    if labels.max() == 0:
        nan = float("nan")
        if return_details:
            return CrossSection(nan, nan, nan, nan, 0.0, 0, center, binary)
        return nan

    props = regionprops(labels)
    center_label = int(labels[center])
    if center_label > 0:
        chosen = next(p for p in props if p.label == center_label)
    elif require_center:
        # The centreline point is outside the lumen -- common on a jagged
        # skeleton. The old fallback took the region whose centroid was nearest,
        # which on a coronary tree can be an entirely different vessel running
        # alongside, recorded as a confident measurement. Return NaN instead and
        # let the caller's fill path handle a gap it can see.
        nan = float("nan")
        if return_details:
            return CrossSection(nan, nan, nan, nan, 0.0, 0, center, binary,
                                center_background=True)
        return nan
    else:
        cr, cc = center
        chosen = min(
            props,
            key=lambda p: (p.centroid[0] - cr) ** 2 + (p.centroid[1] - cc) ** 2,
        )

    # A region reaching the window edge is clipped: its perimeter includes the
    # straight cut and under-states the true section. With the historical fixed
    # 20-voxel window this silently affected every vessel above ~1.0 mm radius at
    # res=50 um -- 20-52% of points on the LADAF graph at 13-26 um voxels.
    region = labels == chosen.label
    touches = bool(
        region[0, :].any() or region[-1, :].any()
        or region[:, 0].any() or region[:, -1].any()
    )

    perimeter = float(chosen.perimeter)
    area = float(chosen.area)
    r_perimeter = perimeter / (2 * np.pi) * res
    r_area = float(np.sqrt(area / np.pi) * res)
    radius = r_area if method == "area" else r_perimeter

    if return_details:
        return CrossSection(
            radius=radius, r_perimeter=r_perimeter, r_area=r_area,
            perimeter=perimeter, area=area, label=int(chosen.label),
            center=center, binary=binary, touches_border=touches,
        )
    return radius


def save_cross_section_png(cross: CrossSection, out_path, title: Optional[str] = None) -> None:
    """Render a binarised cross-section + its boundary to a PNG (debugging).

    Shows the binarised slice (nearest-neighbour, pixel grid), overlays the
    region boundary via ``skimage.measure.find_contours(mask, 0.5)`` (the
    marching-squares polygon whose length the perimeter approximates), marks the
    slice centre, and titles it with the perimeter/area and both radius estimates
    so the staircase over-estimate for near-voxel-size vessels is visible.
    Requires ``matplotlib`` (the ``[viz]`` extra).
    """
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    from skimage.measure import find_contours

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    mask = np.asarray(cross.binary)
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(mask, cmap="gray", interpolation="nearest", origin="upper")
    # Boundary of the chosen region only.
    region = mask if cross.label == 0 else _region_mask(mask, cross)
    for contour in find_contours(region.astype(float), 0.5):
        ax.plot(contour[:, 1], contour[:, 0], "-r", linewidth=1.2)
    ax.plot(cross.center[1], cross.center[0], "+", color="cyan", markersize=10)
    # Light pixel grid so individual voxels are visible for tiny regions.
    ax.set_xticks(np.arange(-0.5, mask.shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-0.5, mask.shape[0], 1), minor=True)
    ax.grid(which="minor", color="0.4", linewidth=0.3)
    ax.set_xticks([]); ax.set_yticks([])
    if title is None:
        title = (
            f"perim={cross.perimeter:.1f}px  area={cross.area:.0f}px^2\n"
            f"r_perim={cross.r_perimeter:.2f}  r_area={cross.r_area:.2f}"
        )
    ax.set_title(title, fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def _region_mask(mask: np.ndarray, cross: CrossSection) -> np.ndarray:
    """Boolean mask of just the chosen labelled region (for boundary drawing)."""
    from skimage.measure import label

    labels = label(mask)
    return labels == cross.label


def segment_radii_from_volume(
    volume: np.ndarray,
    centreline: np.ndarray,
    res: float = 50.0,
    half_size: int = 20,
    threshold: float = 0.0,
    fill_percentiles: Tuple[float, float] = (5.0, 90.0),
    method: str = "perimeter",
    debug: Optional[dict] = None,
    radii: Optional[np.ndarray] = None,
    max_half: int = 128,
    require_center: bool = True,
    report: Optional[dict] = None,
) -> np.ndarray:
    """Corrected per-point radius profile along a vessel centreline.

    Parameters
    ----------
    volume : 3-D array
        Raw image sub-volume (index coordinates match ``centreline``).
    centreline : (P, 3) array
        Ordered centreline points in *voxel* coordinates (world / ``res``), as
        produced by ``give_oblique_slice_info``.
    res : float
        Voxel size (micrometres). Radius = perimeter/(2*pi)*res (or, with
        ``method="area"``, sqrt(area/pi)*res).
    half_size : int
        Half-width of the resampled cross-section window (voxels). Used only as
        the fallback when ``radii`` is not supplied; otherwise the window is
        sized per point from the local radius and grown while the section is
        clipped. A *fixed* window cannot measure a vessel wider than itself.
    radii : (P,) array, optional
        Prior per-point radius in the same units as ``res``, used to size the
        cut window and the tangent-fit window. It need not be accurate -- it
        only has to set a scale -- but supplying it is what allows large vessels
        to be measured at all. Without it the window stays fixed at
        ``half_size`` and vessels above ``half_size * res`` are silently clipped.
    max_half : int
        Upper bound on window growth (voxels).
    require_center : bool
        Reject rather than guess when the centreline point lands on background.
    report : dict, optional
        Mutated in place with ``n_points``, ``n_measured``, ``n_clipped`` and
        ``n_background`` so a caller can see how much of the profile is real
        measurement rather than fill.
    threshold : float
        Binarisation threshold for the raw intensities.
    method : {"perimeter", "area"}
        Which cross-section radius estimate to use.
    debug : dict, optional
        When given, saves debug PNGs for near-voxel-size cross-sections. Keys
        (mutated in place so a total cap is shared across calls):
        ``dir`` (output directory), ``radius_vox_max`` (save when
        ``r_perimeter/res <= this``; default 2.5), ``budget`` (max PNGs total;
        default 40), ``saved`` (running count), ``prefix`` (filename prefix).

    Returns
    -------
    (P,) array of corrected radii. Along-profile outliers are filled
    (nearest, percentiles ``fill_percentiles``) and NaNs interpolated, matching
    the MATLAB ``filloutliers``/``fillmissing`` post-processing.
    """
    from skeleton_analysis.outlier.detect import filloutliers_nearest

    centreline = np.asarray(centreline, dtype=float)
    p = len(centreline)
    rads = np.full(p, np.nan)
    # Tangents for the whole centreline at once. A per-point one-step difference
    # measures the skeletonisation lattice, not the vessel (see robust_tangents).
    radii_vox = None
    if radii is not None:
        radii_vox = np.asarray(radii, dtype=float).reshape(-1) / float(res)
    tangents = robust_tangents(centreline, radii_vox, spacing=1.0)

    n_clipped = 0
    n_background = 0
    for k in range(p):
        point = centreline[k]
        normal = tangents[k]
        if np.linalg.norm(normal) == 0:
            continue

        # Window sized from the local radius, grown while the section is clipped.
        # A fixed window cannot measure a vessel wider than itself: at the old
        # default (half_size=20, res=50um) that capped the measurable radius at
        # 1.0mm, so the largest vessels -- the ones whose radius matters most --
        # were the ones guaranteed to be wrong.
        if radii_vox is not None and np.isfinite(radii_vox[k]) and radii_vox[k] > 0:
            half = min(int(2.5 * radii_vox[k]) + 2, max_half)
        else:
            half = half_size

        cross = None
        while True:
            slice2d, _coords = oblique_slice(volume, point, normal, half_size=half)
            cross = cross_section_radius(
                slice2d, res=res, threshold=threshold, method=method,
                return_details=True, require_center=require_center,
            )
            if not cross.touches_border or half >= max_half:
                break
            half = min(int(half * 1.6) + 1, max_half)

        if cross.center_background:
            n_background += 1
            continue          # leave NaN; the fill path below handles the gap
        if cross.touches_border:
            n_clipped += 1
            continue          # clipped at max_half: a clipped perimeter is not a radius
        rads[k] = cross.radius
        if debug is not None and np.isfinite(cross.r_perimeter):
            _maybe_save_debug(cross, res, k, debug)

    if report is not None:
        report.update(n_points=p, n_clipped=n_clipped, n_background=n_background,
                      n_measured=int(np.isfinite(rads).sum()))

    # Fill along-profile outliers, then any remaining NaNs (nearest).
    valid = ~np.isnan(rads)
    if valid.sum() >= 2:
        filled = filloutliers_nearest(np.where(valid, rads, np.nanmean(rads)),
                                      fill_percentiles[0], fill_percentiles[1])
        rads = np.where(valid, filled, np.nan)
        rads = _fill_nan_nearest(rads)
    return rads


def _maybe_save_debug(cross: CrossSection, res: float, k: int, debug: dict) -> None:
    """Save a debug PNG for a near-voxel-size cross-section, honouring the cap."""
    radius_vox_max = debug.get("radius_vox_max", 2.5)
    budget = debug.get("budget", 40)
    saved = debug.get("saved", 0)
    if saved >= budget:
        return
    radius_vox = cross.r_perimeter / res if res else float("inf")
    if radius_vox > radius_vox_max:
        return
    prefix = debug.get("prefix", "")
    out_dir = Path(debug["dir"])
    name = f"{prefix}pt{k:04d}_rvox{radius_vox:04.1f}.png"
    title = (
        f"{prefix}point {k}\n"
        f"perim={cross.perimeter:.1f}px  area={cross.area:.0f}px^2\n"
        f"r_perim={cross.r_perimeter:.1f}um ({radius_vox:.1f} vox)  "
        f"r_area={cross.r_area:.1f}um"
    )
    save_cross_section_png(cross, out_dir / name, title=title)
    debug["saved"] = saved + 1


def _fill_nan_nearest(x: np.ndarray) -> np.ndarray:
    """Replace NaNs with the nearest (by index) non-NaN value (fillmissing 'nearest')."""
    x = np.array(x, dtype=float)
    good = np.flatnonzero(~np.isnan(x))
    if good.size == 0:
        return x
    for i in np.flatnonzero(np.isnan(x)):
        j = good[np.argmin(np.abs(good - i))]
        x[i] = x[j]
    return x


def centreline_voxels(point_coords: np.ndarray, res: float) -> np.ndarray:
    """World coordinates -> voxel coordinates (port of ``give_oblique_slice_info``)."""
    return np.asarray(point_coords, dtype=float) / res
