"""Validate a reconstructed lumen surface against the spatial graph.

The spatial graph is the ground truth: it carries a measured radius at every
centreline point, so the reconstruction can be scored without a second
segmentation.  Reported metrics, and why each is here:

**Radius agreement (per Strahler order)** — the headline check.  At every
centreline point the *inscribed* radius of the reconstruction is measured as the
distance from the centreline to the nearest surface point, and compared with the
graph radius.  Summarised per order as bias, limits of agreement, RMSE and
relative error, and plotted as a Bland-Altman figure.  Bias by order is what shows
whether the reconstruction systematically inflates the small distal vessels — the
usual failure mode of a smooth-minimum SDF blend.

**clDice** (Shit et al., CVPR 2021) — the topology-aware overlap score,
``2*Tprec*Tsens/(Tprec+Tsens)``:

* ``Tsens`` = fraction of the **true** centreline lying inside the reconstruction.
  Because the graph *is* the ground-truth skeleton, this term is exact — no
  skeletonisation error enters it.
* ``Tprec`` = fraction of the reconstruction's own skeleton lying inside the
  ground-truth tube volume, which needs a voxelisation and a 3-D thinning.

**Supporting metrics from the same literature**

* ``Dice`` / ``Jaccard`` on the voxelised volumes — standard overlap, reported
  alongside clDice because clDice alone hides volumetric error.
* ``beta0`` — connected-component count of the reconstruction against the graph's
  own component count.  A Betti-0 error is the topological defect clDice is least
  sensitive to, and speckle from an adaptive extractor shows up here first.
* Surface-to-graph distance (mean / RMS / 95th percentile) — the ASSD and HD95
  analogues, measured against the graph's implied tube rather than a second mesh.

Voxel-based metrics are resolution-limited: vessels thinner than about two voxels
cannot be represented, so the reported ``fraction_vessels_resolved`` states how
much of the tree the score actually covers.

Usage::

    python -m coronary_sdf.surface_validation \
        --graph pruned.am.xml --stl lumen_bspline.stl \
        --out analysis_out/validation --voxel-mm 0.15
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pyvista as pv
import vtk
from scipy.signal import savgol_filter
from scipy.spatial import KDTree
from vtkmodules.util import numpy_support

from .parse_amira import parse_xml
from .flow_fractions import BIF_SKIP_POINTS
from .strahler_analysis import (UM_PER_MM, write_csv, _stats,
                                filter_segments, node_degrees)

# Query points this far outside the surface bounding box are treated as outside
# the reconstruction's coverage rather than as reconstruction errors.
BBOX_MARGIN_MM = 1.0

# A centreline point that is not enclosed by the surface is only counted as a
# reconstruction *failure* when a surface exists nearby; past this margin the
# vessel was cropped away before reconstruction and scoring it would penalise the
# method for geometry it was never asked to build. The margin scales with the
# local radius (a big vessel has a correspondingly big gap when it breaks) with an
# absolute floor for the smallest vessels.
CROP_MARGIN_RADIUS_FACTOR = 3.0
CROP_MARGIN_MIN_MM = 0.5

# Nearest centreline neighbours tested when deciding whether a voxel is inside the
# ground-truth tube union. Testing only the single nearest point would miss a
# voxel that sits inside a fatter neighbour a little further away.
GT_NEIGHBOURS = 8


# ── graph side ────────────────────────────────────────────────────────────────


def _component_labels(segments: list[dict[str, Any]]) -> list[int]:
    """Connected-component id per segment, numbered largest tree first.

    A coronary spatial graph normally holds the left and right trees as two
    disjoint components. Scoring them together penalises a surface that only ever
    represented one of them, so every overlap metric is reported per component."""
    parent: dict[int, int] = {}

    def find(x: int) -> int:
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for s in segments:
        ra, rb = find(s["node1"]), find(s["node2"])
        if ra != rb:
            parent[ra] = rb

    roots = [find(s["node1"]) for s in segments]
    counts: dict[int, int] = {}
    for r in roots:
        counts[r] = counts.get(r, 0) + 1
    order = sorted(counts, key=lambda r: -counts[r])
    remap = {r: i for i, r in enumerate(order)}
    return [remap[r] for r in roots]


def _smooth_centreline(coords: np.ndarray, window: int = 11, poly: int = 2) -> np.ndarray:
    """Savitzky-Golay smoothing along a centreline, for tangent estimation only.

    The returned curve is used solely to orient the measurement plane; radii are
    always sampled at the original, unsmoothed station positions."""
    n = len(coords)
    if n < 5:
        return coords
    w = min(window, n if n % 2 else n - 1)
    if w < poly + 2:
        return coords
    if w % 2 == 0:
        w -= 1
    return savgol_filter(coords, window_length=w, polyorder=poly, axis=0)


#: Bifurcation exclusion as a multiple of the LOCAL VESSEL RADIUS, applied as an
#: arc-length distance from any degree>=3 node and unioned with the fixed
#: ``bif_skip`` point count. Set from the measured profile in
#: ``radius_bias_vs_bif_distance.csv`` rather than from the blend's nominal
#: reach: the median bias is still +115% (order 1) and +7.8% (order 2) in the
#: 4-6 radius bin and only reaches ~0 in the 6-8 bin, so the blend's influence
#: extends about twice as far as ``XS_JUNC_NODE_PROXIMITY_FACTOR`` (3.0) alone
#: would suggest.
BIF_SKIP_RADII = 8.0


def graph_centreline(
    graph_xml: Path, bif_skip: int = BIF_SKIP_POINTS,
    exclude_segments: set[int] | None = None,
    bif_skip_radii: float = BIF_SKIP_RADII,
) -> dict[str, np.ndarray]:
    """Centreline points with a local tangent and a bifurcation flag.

    Returns arrays keyed ``coords`` (mm), ``radii`` (mm), ``tangent`` (unit),
    ``strahler``, ``seg_id`` and ``is_bif``.  The tangent comes from central
    differences along each segment, and is what lets the radius be measured in the
    true cross-sectional plane rather than as a nearest-surface distance."""
    nodes, points, segments = parse_xml(graph_xml)
    segments = filter_segments(segments, exclude_segments or set())
    coords, radii, orders, seg_ids, tangents, bif, comps = [], [], [], [], [], [], []
    bif_dist, bif_dist_r = [], []
    seg_component = _component_labels(segments)
    # Degrees come from the surviving segments, not the file: dropping a segment
    # can turn a junction into a plain continuation.
    degrees = node_degrees(segments)

    def deg(nid: int) -> int:
        return degrees.get(nid, 0)

    for si, seg in enumerate(segments):
        pts = [points[p] for p in seg["point_ids"] if p in points]
        if len(pts) < 2:
            continue
        arr = np.asarray(pts, dtype=np.float64)
        c = arr[:, :3] / UM_PER_MM
        r = arr[:, 3] / UM_PER_MM
        # Differentiating the raw centreline propagates skeletonisation jitter
        # straight into the tangent, tilting the measurement plane and inflating
        # every radius by 1/cos(theta). Smooth first: a Savitzky-Golay filter
        # keeps the curve's shape while removing point-to-point noise.
        t = np.gradient(_smooth_centreline(c), axis=0)
        norm = np.linalg.norm(t, axis=1, keepdims=True)
        t = np.divide(t, norm, out=np.zeros_like(t), where=norm > 0)

        n = len(c)
        mask = np.zeros(n, dtype=bool)
        k = min(bif_skip, n)
        # A fixed point count is the wrong shape for this exclusion. The SDF's
        # junction blend reaches a few LOCAL RADII from the node
        # (XS_JUNC_NODE_PROXIMITY_FACTOR), so ``bif_skip`` points -- about 2 mm
        # at 0.2 mm spacing -- clears a 0.2 mm-radius vessel but leaves a
        # 1.3 mm-radius one measuring the fused parent+daughter blob while still
        # labelling it "non-bifurcation". That produced a radius bias rising
        # monotonically with Strahler order (+1.9% at order 1 to +69% at order
        # 4); raising bif_skip alone collapsed it to +1.9%/+5.6%/+11.6% but
        # deleted orders 4-5 outright, because thick segments are not long
        # enough in POINTS to survive a large fixed skip. Excluding a
        # radius-scaled arc distance instead clears thick vessels properly and
        # keeps their interiors, since a thick vessel is also a long one.
        arc = np.concatenate(
            [[0.0], np.cumsum(np.linalg.norm(np.diff(c, axis=0), axis=1))]
        )
        reach = float(bif_skip_radii) * r
        if deg(seg["node1"]) >= 3:
            mask[:k] = True
            if bif_skip_radii > 0:
                mask |= arc <= reach
        if deg(seg["node2"]) >= 3:
            mask[n - k:] = True
            if bif_skip_radii > 0:
                mask |= (arc[-1] - arc) <= reach

        # Arc distance from each station to the nearest bifurcation, in units of
        # the local radius. Any fixed cutoff on this quantity is a free parameter
        # that moves the answer, so the distance itself is carried through and
        # the bias is reported as a PROFILE against it: the value to quote is the
        # asymptote where the profile flattens, read off the data rather than
        # chosen. inf marks a segment with no bifurcation end at all.
        d_bif = np.full(n, np.inf)
        if deg(seg["node1"]) >= 3:
            d_bif = np.minimum(d_bif, arc)
        if deg(seg["node2"]) >= 3:
            d_bif = np.minimum(d_bif, arc[-1] - arc)

        coords.append(c)
        radii.append(r)
        tangents.append(t)
        orders.append(np.full(n, int(seg.get("strahler", 0)), dtype=np.int32))
        seg_ids.append(np.full(n, int(seg.get("id", si)), dtype=np.int64))
        bif.append(mask)
        comps.append(np.full(n, seg_component[si], dtype=np.int32))
        bif_dist.append(d_bif)
        bif_dist_r.append(d_bif / np.maximum(r, 1e-9))

    return {
        "coords": np.vstack(coords),
        "radii": np.concatenate(radii),
        "tangent": np.vstack(tangents),
        "strahler": np.concatenate(orders),
        "seg_id": np.concatenate(seg_ids),
        "is_bif": np.concatenate(bif),
        "component": np.concatenate(comps),
        "bif_dist_mm": np.concatenate(bif_dist),
        "bif_dist_radii": np.concatenate(bif_dist_r),
    }


def _perpendicular_basis(tangent: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Two unit vectors spanning the plane normal to ``tangent`` (per row)."""
    ref = np.tile(np.array([0.0, 0.0, 1.0]), (len(tangent), 1))
    # Pick a different reference where the tangent is near-parallel to z, else the
    # cross product degenerates.
    near_z = np.abs(tangent[:, 2]) > 0.9
    ref[near_z] = np.array([1.0, 0.0, 0.0])
    u = np.cross(tangent, ref)
    u /= np.maximum(np.linalg.norm(u, axis=1, keepdims=True), 1e-12)
    v = np.cross(tangent, u)
    v /= np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-12)
    return u, v


def cross_section_radius(
    mesh: pv.PolyData,
    coords: np.ndarray,
    tangent: np.ndarray,
    r_hint: np.ndarray,
    n_rays: int = 32,
    max_factor: float = 6.0,
    min_hit_fraction: float = 0.5,
) -> dict[str, np.ndarray]:
    """Lumen radius at each station, by three equivalent-radius definitions.

    ``n_rays`` rays leave the centreline in the plane normal to the vessel axis
    and the first surface crossing on each is recorded.  The hit points form a
    polygonal cross-section, from which three radii are derived:

    ``r_perimeter``  ``P / 2pi`` -- **the one to compare against an Amira graph
                     built with perimeter-based thickness.**  HiP-CT vessels are
                     collapsed ex vivo, so the graph stores a perimeter-equivalent
                     radius; comparing that against an area- or median-based
                     measurement compares two different definitions and shows a
                     deficit even for a perfect reconstruction, because for any
                     non-circular section ``r_perimeter > r_area``.
    ``r_area``       ``sqrt(A / pi)`` -- the hydraulically meaningful radius.
    ``r_median``     median ray length -- robust, but a mixture of the two.

    Also returned: ``anisotropy`` (max/min ray length) and ``hit_fraction``.
    Anisotropy separates the two ways a section departs from a perpendicular
    circle -- an oblique cut plane, or a genuinely elliptical (collapsed) lumen.

    Note the sign of the obliquity error: an oblique cut through a round tube is
    an ellipse with semi-axes ``r`` and ``r / cos(theta)``, so tilt can only make
    the measured radius **larger**.  It cannot manufacture an apparent deficit.

    ``r_perimeter`` and ``r_area`` need a closed polygon and so are only defined
    where every ray hits; ``r_median`` tolerates partial hits down to
    ``min_hit_fraction``."""
    obb = vtk.vtkOBBTree()
    obb.SetDataSet(mesh)
    obb.BuildLocator()

    u, v = _perpendicular_basis(tangent)
    angles = np.linspace(0.0, 2.0 * np.pi, n_rays, endpoint=False)
    cos_a, sin_a = np.cos(angles), np.sin(angles)
    # An inscribed regular n-gon under-measures a circle's circumference by
    # sinc(pi/n); undo that so r_perimeter is unbiased for a round section.
    gon_correction = (np.pi / n_rays) / np.sin(np.pi / n_rays)

    n = len(coords)
    out = {k: np.full(n, np.nan) for k in
           ("r_median", "r_area", "r_perimeter", "anisotropy", "axis_offset_mm",
            "axis_offset_frac")}
    out["hit_fraction"] = np.zeros(n)
    pts = vtk.vtkPoints()

    for i, (p, ui, vi, rh) in enumerate(zip(coords, u, v, r_hint)):
        reach = max(float(rh) * max_factor, 1e-3)
        lengths = np.full(n_rays, np.nan)
        for j, (ca, sa) in enumerate(zip(cos_a, sin_a)):
            end = p + (ui * ca + vi * sa) * reach
            pts.Reset()
            if obb.IntersectWithLine(p, end, pts, None) == 0 or pts.GetNumberOfPoints() == 0:
                continue
            lengths[j] = float(np.linalg.norm(np.asarray(pts.GetPoint(0)) - p))

        ok = np.isfinite(lengths)
        out["hit_fraction"][i] = ok.mean()
        if ok.sum() < 2 or out["hit_fraction"][i] < min_hit_fraction:
            continue
        out["r_median"][i] = float(np.median(lengths[ok]))
        out["anisotropy"][i] = float(lengths[ok].max() / max(lengths[ok].min(), 1e-9))
        if not ok.all():
            continue
        x, y = lengths * cos_a, lengths * sin_a
        perim = float(np.hypot(np.diff(np.r_[x, x[0]]), np.diff(np.r_[y, y[0]])).sum())
        area = 0.5 * abs(float(np.dot(x, np.roll(y, -1)) - np.dot(np.roll(x, -1), y)))
        out["r_perimeter"][i] = perim * gon_correction / (2.0 * np.pi)
        out["r_area"][i] = math.sqrt(area / np.pi)
        # Offset between the graph centreline point and the centroid of the
        # reconstructed cross-section: the vessel *axis* displacement. Radius
        # measured by perimeter or area is independent of where the rays started,
        # so this isolates centreline drift from radius error. Reported as a
        # fraction of the local radius, since a 0.1 mm offset means something very
        # different in a 2 mm vessel and a 0.3 mm one.
        # Use the polygon's AREA centroid, not the mean of its vertices. Rays at
        # uniform angles from an off-centre origin land unevenly around the
        # section, and the vertex mean recovers only half the true offset (checked
        # against an analytic circle); the shoelace centroid is unbiased.
        cross = x * np.roll(y, -1) - np.roll(x, -1) * y
        signed_area = 0.5 * float(cross.sum())
        if abs(signed_area) < 1e-12:
            continue
        cx = float(((x + np.roll(x, -1)) * cross).sum() / (6.0 * signed_area))
        cy = float(((y + np.roll(y, -1)) * cross).sum() / (6.0 * signed_area))
        off = math.hypot(cx, cy)
        out["axis_offset_mm"][i] = off
        rr_local = out["r_area"][i]
        out["axis_offset_frac"][i] = off / rr_local if rr_local > 0 else math.nan
    return out


# ── surface side ──────────────────────────────────────────────────────────────


def inscribed_radius(mesh: pv.PolyData, query_mm: np.ndarray) -> np.ndarray:
    """Distance from each query point to the nearest surface point.

    For a locally circular lumen this is the reconstructed radius; it is the
    quantity the SDF pipeline itself controls, so it is the fair comparison
    against the graph's thickness value."""
    if len(query_mm) == 0:
        return np.empty(0, dtype=np.float64)
    locator = vtk.vtkStaticCellLocator()
    locator.SetDataSet(mesh)
    locator.BuildLocator()
    out = np.empty(len(query_mm), dtype=np.float64)
    closest = [0.0, 0.0, 0.0]
    gen_cell = vtk.vtkGenericCell()
    cell_id = vtk.reference(0)
    sub_id = vtk.reference(0)
    dist2 = vtk.reference(0.0)
    for i, p in enumerate(query_mm):
        locator.FindClosestPoint(p, closest, gen_cell, cell_id, sub_id, dist2)
        out[i] = math.sqrt(float(dist2))
    return out


def points_inside(mesh: pv.PolyData, query_mm: np.ndarray) -> np.ndarray:
    """Boolean mask of query points enclosed by ``mesh``."""
    if len(query_mm) == 0:
        return np.zeros(0, dtype=bool)
    cloud = pv.PolyData(np.asarray(query_mm, dtype=np.float64))
    sel = cloud.select_enclosed_points(mesh, tolerance=0.0, check_surface=False)
    return np.asarray(sel["SelectedPoints"], dtype=bool)


def voxelize_surface(
    mesh: pv.PolyData, origin: np.ndarray, spacing: float, dims: tuple[int, int, int]
) -> np.ndarray:
    """Rasterise a closed surface to a boolean occupancy grid.

    Uses VTK's scanline stencil rather than a point-in-polygon test per voxel,
    which is the difference between seconds and hours on a grid of this size."""
    img = vtk.vtkImageData()
    img.SetOrigin(*[float(v) for v in origin])
    img.SetSpacing(spacing, spacing, spacing)
    img.SetDimensions(*dims)
    img.AllocateScalars(vtk.VTK_UNSIGNED_CHAR, 1)
    numpy_support.vtk_to_numpy(img.GetPointData().GetScalars())[:] = 1

    stencil = vtk.vtkPolyDataToImageStencil()
    stencil.SetInputData(mesh)
    stencil.SetOutputOrigin(*[float(v) for v in origin])
    stencil.SetOutputSpacing(spacing, spacing, spacing)
    stencil.SetOutputWholeExtent(img.GetExtent())
    stencil.SetTolerance(0.0)
    stencil.Update()

    carve = vtk.vtkImageStencil()
    carve.SetInputData(img)
    carve.SetStencilConnection(stencil.GetOutputPort())
    carve.ReverseStencilOff()
    carve.SetBackgroundValue(0)
    carve.Update()

    arr = numpy_support.vtk_to_numpy(carve.GetOutput().GetPointData().GetScalars())
    # VTK ravels z-slowest; transpose back to (nx, ny, nz) index order.
    return arr.reshape(dims[2], dims[1], dims[0]).transpose(2, 1, 0).astype(bool)


def rasterize_graph_volume(
    coords_mm: np.ndarray,
    radii_mm: np.ndarray,
    origin: np.ndarray,
    spacing: float,
    dims: tuple[int, int, int],
) -> np.ndarray:
    """Ground-truth tube volume as a union of spheres, one per centreline point.

    The graph samples its centreline about every 0.08 mm, which is finer than the
    smallest vessel radius, so a sphere union reproduces the swept-capsule volume
    closely while costing one splat per point instead of a full SDF evaluation."""
    grid = np.zeros(dims, dtype=bool)
    for p, r in zip(coords_mm, radii_mm):
        if r <= 0:
            continue
        lo = np.floor((p - r - origin) / spacing).astype(int)
        hi = np.ceil((p + r - origin) / spacing).astype(int) + 1
        lo = np.clip(lo, 0, np.array(dims))
        hi = np.clip(hi, 0, np.array(dims))
        if np.any(hi <= lo):
            continue
        ax = [
            (origin[d] + np.arange(lo[d], hi[d]) * spacing - p[d]) ** 2
            for d in range(3)
        ]
        d2 = ax[0][:, None, None] + ax[1][None, :, None] + ax[2][None, None, :]
        grid[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]] |= d2 <= r * r
    return grid


def rasterize_points(
    coords_mm: np.ndarray, origin: np.ndarray, spacing: float,
    dims: tuple[int, int, int],
) -> np.ndarray:
    """Boolean grid with the voxel containing each centreline point set."""
    grid = np.zeros(dims, dtype=bool)
    idx = np.floor((coords_mm - origin) / spacing).astype(int)
    ok = np.all((idx >= 0) & (idx < np.array(dims)), axis=1)
    idx = idx[ok]
    grid[idx[:, 0], idx[:, 1], idx[:, 2]] = True
    return grid


# ── metrics ───────────────────────────────────────────────────────────────────


def radius_agreement(
    r_graph: np.ndarray, r_recon: np.ndarray, orders: np.ndarray
) -> list[dict[str, Any]]:
    """Per-Strahler-order agreement between graph and reconstructed radius.

    ``bias`` is mean(recon - graph); ``loa_lo/hi`` are the Bland-Altman 95% limits
    of agreement (bias +/- 1.96 SD of the difference)."""
    rows: list[dict[str, Any]] = []
    for order in sorted(set(int(o) for o in orders)) + ["all"]:
        sel = np.ones(len(orders), dtype=bool) if order == "all" else (orders == order)
        g, rc = r_graph[sel], r_recon[sel]
        ok = np.isfinite(g) & np.isfinite(rc)
        g, rc = g[ok], rc[ok]
        if len(g) == 0:
            continue
        diff = rc - g
        bias, sd = float(diff.mean()), float(diff.std(ddof=1)) if len(diff) > 1 else 0.0
        with np.errstate(divide="ignore", invalid="ignore"):
            rel = np.where(g > 0, diff / g, np.nan)
        rows.append({
            "strahler": order,
            "n_points": int(len(g)),
            "radius_graph_mean_mm": float(g.mean()),
            "radius_graph_sd_mm": float(g.std(ddof=1)) if len(g) > 1 else 0.0,
            "radius_recon_mean_mm": float(rc.mean()),
            "radius_recon_sd_mm": float(rc.std(ddof=1)) if len(rc) > 1 else 0.0,
            "bias_mm": bias,
            "bias_sd_mm": sd,
            "loa_lo_mm": bias - 1.96 * sd,
            "loa_hi_mm": bias + 1.96 * sd,
            "rmse_mm": float(np.sqrt(np.mean(diff ** 2))),
            "mae_mm": float(np.mean(np.abs(diff))),
            "rel_bias_pct": float(np.nanmean(rel) * 100.0),
            "rel_mae_pct": float(np.nanmean(np.abs(rel)) * 100.0),
        })
    return rows


#: Bin edges for the bias-vs-bifurcation-distance profile, in local radii.
BIF_DISTANCE_BINS = (0.0, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0, math.inf)

#: Stations at least this far (in local radii) from any bifurcation are treated
#: as far-field for the headline number. Justified only if the profile is flat
#: there, which ``radius_bias_vs_bif_distance.csv`` is what lets you check.
FAR_FIELD_RADII = 8.0


def bias_vs_bif_distance(
    d_over_r: np.ndarray,
    r_graph: np.ndarray,
    r_recon: np.ndarray,
    orders: np.ndarray,
    bins: tuple[float, ...] = BIF_DISTANCE_BINS,
) -> list[dict[str, Any]]:
    """Radius bias binned by distance from the nearest bifurcation.

    Excluding a fixed zone around each junction forces a choice of cutoff, and
    the measured bias on this data moves monotonically with that choice (+7.0%
    at 4 radii to -1.7% at 8) with no value that is obviously right. Profiling
    the bias against distance removes the choice: the near bins show the
    junction blend decaying, and the value to quote is the level the profile
    settles at. If it never settles, that is itself the finding, and the CSV
    shows it rather than hiding it behind a cutoff.

    ``rel_bias_median_pct`` is the robust companion to the mean: the relative
    error distribution is right-skewed, so a handful of near-junction stations
    can carry the mean on their own.
    """
    rows: list[dict[str, Any]] = []
    finite = np.isfinite(r_graph) & np.isfinite(r_recon) & np.isfinite(d_over_r)
    for order in sorted({int(o) for o in orders}) + ["all"]:
        omask = np.ones(len(orders), dtype=bool) if order == "all" else (orders == order)
        for lo, hi in zip(bins[:-1], bins[1:]):
            sel = finite & omask & (d_over_r >= lo) & (d_over_r < hi)
            n = int(sel.sum())
            if n == 0:
                continue
            g, rc = r_graph[sel], r_recon[sel]
            diff = rc - g
            with np.errstate(divide="ignore", invalid="ignore"):
                rel = np.where(g > 0, diff / g, np.nan)
            rows.append({
                "strahler": order,
                # Readable label as well as the numeric edges: the open-ended
                # top bin has an infinite upper edge, which serialises to an
                # empty CSV cell.
                "bin_radii": f"{lo:g}-{hi:g}" if math.isfinite(hi) else f">={lo:g}",
                "d_lo_radii": lo,
                "d_hi_radii": hi,
                "n_points": n,
                "radius_graph_mean_mm": float(g.mean()),
                "radius_recon_mean_mm": float(rc.mean()),
                "bias_mm": float(diff.mean()),
                "rmse_mm": float(np.sqrt(np.mean(diff ** 2))),
                "rel_bias_mean_pct": float(np.nanmean(rel) * 100.0),
                "rel_bias_median_pct": float(np.nanmedian(rel) * 100.0),
            })
    return rows


def classify_coverage(
    enclosed: np.ndarray,
    dist_to_surface: np.ndarray,
    radii: np.ndarray,
    factor: float = CROP_MARGIN_RADIUS_FACTOR,
    min_mm: float = CROP_MARGIN_MIN_MM,
) -> np.ndarray:
    """Label each centreline point ``inside`` / ``missed`` / ``cropped``.

    A surface is normally built from a *cropped* graph — an ROI, a clipped distal
    extent, a single artery — so graph absent from the reconstruction is not the
    same as graph the reconstruction got wrong:

    ``inside``   enclosed by the surface: the reconstruction covers this vessel.
    ``missed``   not enclosed, but surface exists within the margin — a genuine
                 defect (a break, a gap, or a wall pulled inside the centreline).
    ``cropped``  not enclosed and no surface within the margin — this vessel was
                 removed before reconstruction and must not be scored.

    Only the first two are "in domain". Scoring `cropped` points would make a
    tightly-cropped ROI look like a failed reconstruction."""
    margin = np.maximum(np.asarray(radii) * factor, min_mm)
    state = np.where(
        enclosed, "inside",
        np.where(np.asarray(dist_to_surface) <= margin, "missed", "cropped"),
    )
    return state.astype(object)


def per_tree_voxel_metrics(
    pred_vol: np.ndarray,
    pred_skel: np.ndarray,
    origin: np.ndarray,
    spacing: float,
    dims: tuple[int, int, int],
    coords: np.ndarray,
    radii: np.ndarray,
    component: np.ndarray,
    margin_mm: float = 1.0,
) -> list[dict[str, Any]]:
    """clDice / Dice for each graph component against the same reconstruction.

    Each component is scored inside its own bounding box (dilated by
    ``margin_mm``), so the surface is only ever asked about the region that tree
    occupies.  The reconstruction is voxelised and skeletonised **once** on the
    full grid and then sliced — skeletonising a cropped volume would introduce
    spurious endpoints at the crop faces."""
    rows: list[dict[str, Any]] = []
    dims_arr = np.asarray(dims)
    for c in sorted({int(x) for x in component}):
        sel = component == c
        pts, rr = coords[sel], radii[sel]
        if len(pts) == 0:
            continue
        pad = float(rr.max()) + margin_mm
        lo = np.clip(np.floor((pts.min(0) - pad - origin) / spacing).astype(int),
                     0, dims_arr)
        hi = np.clip(np.ceil((pts.max(0) + pad - origin) / spacing).astype(int) + 1,
                     0, dims_arr)
        if np.any(hi <= lo):
            continue
        sub_dims = tuple(int(v) for v in (hi - lo))
        sub_origin = origin + lo * spacing

        gt_vol = rasterize_graph_volume(pts, rr, sub_origin, spacing, sub_dims)
        gt_skel = rasterize_points(pts, sub_origin, spacing, sub_dims)
        sl = (slice(lo[0], hi[0]), slice(lo[1], hi[1]), slice(lo[2], hi[2]))

        row: dict[str, Any] = {"tree": f"tree{c}", "n_centreline_points": int(sel.sum())}
        row.update(cl_dice(gt_vol, pred_vol[sl], gt_skel, pred_skel[sl]))
        rows.append(row)
    return rows


def cl_dice(
    gt_volume: np.ndarray, pred_volume: np.ndarray,
    gt_skeleton: np.ndarray, pred_skeleton: np.ndarray,
) -> dict[str, float]:
    """clDice plus the volumetric overlap scores on the same grids."""

    def frac(skel: np.ndarray, vol: np.ndarray) -> float:
        n = int(skel.sum())
        return float((skel & vol).sum() / n) if n else math.nan

    t_sens = frac(gt_skeleton, pred_volume)
    t_prec = frac(pred_skeleton, gt_volume)
    denom = t_prec + t_sens
    if not np.isfinite(denom):
        cldice = math.nan          # one side had no skeleton to score at all
    elif denom == 0:
        cldice = 0.0               # both sides scored zero: no overlap, not undefined
    else:
        cldice = 2.0 * t_prec * t_sens / denom

    inter = int((gt_volume & pred_volume).sum())
    a, b = int(gt_volume.sum()), int(pred_volume.sum())
    dice = 2.0 * inter / (a + b) if (a + b) else math.nan
    union = int((gt_volume | pred_volume).sum())
    return {
        "cl_dice": cldice,
        "topology_precision": t_prec,
        "topology_sensitivity": t_sens,
        "dice": dice,
        "jaccard": inter / union if union else math.nan,
        "n_voxels_gt": a,
        "n_voxels_pred": b,
        "n_voxels_intersection": inter,
    }


# ── driver ────────────────────────────────────────────────────────────────────


def load_surfaces(stl_paths: list[Path]) -> tuple[pv.PolyData, list[dict[str, Any]]]:
    """Read one or more surfaces and merge them into a single mesh.

    A full-tree reconstruction is often delivered as one STL per arterial tree.
    They describe one geometry, so they are merged and validated together: the
    enclosure test, the ray casting and the stencil voxelisation all handle a
    multi-shell closed surface, and the per-component scoring then splits the
    result back out by *graph* tree rather than by input file."""
    meshes: list[pv.PolyData] = []
    info: list[dict[str, Any]] = []
    for path in stl_paths:
        m = pv.read(path).extract_surface().triangulate()
        meshes.append(m)
        info.append({"stl": str(path), "n_vertices": int(m.n_points),
                     "n_triangles": int(m.n_cells),
                     "n_components": int(len(m.split_bodies()))})
        print(f"[valid]   {Path(path).name}: {m.n_points} vertices, "
              f"{m.n_cells} triangles, {info[-1]['n_components']} component(s)")
    merged = meshes[0] if len(meshes) == 1 else meshes[0].merge(meshes[1:])
    if len(meshes) > 1:
        print(f"[valid]   merged: {merged.n_points} vertices, "
              f"{merged.n_cells} triangles")
    return merged, info


def run_validation(
    graph_xml: Path,
    stl_path: Path | list[Path],
    out_dir: Path,
    voxel_mm: float = 0.15,
    skip_voxel: bool = False,
    bif_skip: int = BIF_SKIP_POINTS,
    n_rays: int = 16,
    exclude_segments: set[int] | None = None,
    bif_skip_radii: float = BIF_SKIP_RADII,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[valid] graph: {graph_xml}")
    cl = graph_centreline(graph_xml, bif_skip, exclude_segments, bif_skip_radii)
    coords, radii, orders = cl["coords"], cl["radii"], cl["strahler"]
    print(f"[valid]   {len(coords)} centreline points "
          f"({int(cl['is_bif'].sum())} in bifurcation zones)")

    stl_paths = [stl_path] if isinstance(stl_path, (str, Path)) else list(stl_path)
    print(f"[valid] surface: {len(stl_paths)} file(s)")
    mesh, mesh_info = load_surfaces([Path(p) for p in stl_paths])

    # The reconstruction may cover only part of the imaged tree (e.g. one artery),
    # so restrict every comparison to the graph inside its bounding box and report
    # the coverage rather than scoring absent geometry as error.
    b = np.asarray(mesh.bounds, dtype=np.float64).reshape(3, 2)
    inside_bbox = np.all(
        (coords >= b[:, 0] - BBOX_MARGIN_MM) & (coords <= b[:, 1] + BBOX_MARGIN_MM),
        axis=1,
    )
    # Junction regions are excluded outright: there the graph radius describes a
    # single vessel while the surface is a blended confluence, so any disagreement
    # there measures the blend rather than the reconstruction of a vessel.
    sel = ~cl["is_bif"]
    coverage = float(inside_bbox.mean())
    c_in, r_in, o_in = coords[sel], radii[sel], orders[sel]
    t_in = cl["tangent"][sel]
    comp_in = cl["component"][sel]

    # Classify before measuring: the surface was built from a cropped graph, so
    # separate "not reconstructed here" from "reconstructed wrongly here".
    print("[valid] classifying coverage (inside / missed / cropped) ...")
    r_inscribed = inscribed_radius(mesh, c_in)
    enclosed = points_inside(mesh, c_in)
    state = classify_coverage(enclosed, r_inscribed, r_in)
    in_domain = state != "cropped"
    n_state = {s: int((state == s).sum()) for s in ("inside", "missed", "cropped")}
    print(f"[valid]   bbox coverage {inside_bbox.mean() * 100:.1f}%; "
          f"inside={n_state['inside']} missed={n_state['missed']} "
          f"cropped={n_state['cropped']} "
          f"(in-domain {int(in_domain.sum())} of {len(state)})")

    # Only enclosed stations have a lumen to measure, so only they are ray-cast.
    print(f"[valid] measuring cross-sectional radius ({n_rays} rays/station) ...")
    keys = ("r_median", "r_area", "r_perimeter", "anisotropy", "hit_fraction",
            "axis_offset_mm", "axis_offset_frac")
    meas = {k: np.full(len(c_in), np.nan) for k in keys}
    meas["hit_fraction"] = np.zeros(len(c_in))
    idx = np.flatnonzero(enclosed)
    if len(idx):
        got = cross_section_radius(
            mesh, c_in[idx], t_in[idx], r_in[idx], n_rays=n_rays)
        for k in keys:
            meas[k][idx] = got[k]
    # The graph stores a PERIMETER-equivalent radius (HiP-CT vessels collapse ex
    # vivo), so the perimeter-equivalent measurement is the like-for-like
    # comparison; fall back to the median where the polygon is incomplete.
    r_recon = np.where(np.isfinite(meas["r_perimeter"]),
                       meas["r_perimeter"], meas["r_median"])
    hit_frac = meas["hit_fraction"]

    rad_rows = radius_agreement(r_in, r_recon, o_in)
    write_csv(out_dir / "radius_agreement.csv", rad_rows)
    write_csv(out_dir / "radius_agreement_inscribed.csv",
              radius_agreement(r_in, np.where(enclosed, r_inscribed, np.nan), o_in))
    # Same stations scored under the other two radius definitions, so the effect
    # of the definition itself is visible rather than assumed.
    write_csv(out_dir / "radius_agreement_area.csv",
              radius_agreement(r_in, meas["r_area"], o_in))
    write_csv(out_dir / "radius_agreement_median.csv",
              radius_agreement(r_in, meas["r_median"], o_in))

    write_csv(out_dir / "radius_points.csv", [
        {"tree": f"tree{int(cp)}", "strahler": int(o),
         "radius_graph_mm": float(g), "radius_recon_mm": float(rc),
         "radius_perimeter_mm": float(rp), "radius_area_mm": float(ra),
         "radius_median_mm": float(rm), "radius_inscribed_mm": float(ri),
         "anisotropy": float(an), "axis_offset_mm": float(ao),
         "axis_offset_frac": float(af), "hit_fraction": float(hf),
         "inside_surface": int(e)}
        for cp, o, g, rc, rp, ra, rm, ri, an, ao, af, hf, e in
        zip(comp_in, o_in, r_in, r_recon, meas["r_perimeter"], meas["r_area"],
            meas["r_median"], r_inscribed, meas["anisotropy"],
            meas["axis_offset_mm"], meas["axis_offset_frac"], hit_frac, enclosed)
    ])

    # Per-tree radius agreement, for the same reason the voxel scores are split.
    per_tree_rad: list[dict[str, Any]] = []
    for c in sorted({int(x) for x in comp_in}):
        m = comp_in == c
        for row in radius_agreement(r_in[m], r_recon[m], o_in[m]):
            per_tree_rad.append({"tree": f"tree{c}", **row})
    write_csv(out_dir / "radius_agreement_per_tree.csv", per_tree_rad)

    # --- bias vs distance from the nearest bifurcation -------------------------
    # Everything above deliberately drops the junction zones, but the profile
    # needs exactly those stations: without the near bins there is no way to see
    # where the blend's influence ends, and therefore no way to justify any
    # cutoff. Measure the excluded stations too, for this profile only -- the
    # CSVs above keep their existing (non-bifurcation) semantics.
    d_all = cl["bif_dist_radii"]
    excl = ~sel
    d_prof, g_prof, rc_prof, o_prof = d_all[sel], r_in, r_recon, o_in
    if excl.any():
        print(f"[valid] profiling bias vs bifurcation distance "
              f"({int(excl.sum())} junction-zone stations added) ...")
        c_ex, r_ex, o_ex = coords[excl], radii[excl], orders[excl]
        t_ex = cl["tangent"][excl]
        enc_ex = points_inside(mesh, c_ex)
        rc_ex = np.full(len(c_ex), np.nan)
        idx_ex = np.flatnonzero(enc_ex)
        if len(idx_ex):
            got_ex = cross_section_radius(
                mesh, c_ex[idx_ex], t_ex[idx_ex], r_ex[idx_ex], n_rays=n_rays)
            rc_ex[idx_ex] = np.where(
                np.isfinite(got_ex["r_perimeter"]),
                got_ex["r_perimeter"], got_ex["r_median"])
        d_prof = np.concatenate([d_prof, d_all[excl]])
        g_prof = np.concatenate([g_prof, r_ex])
        rc_prof = np.concatenate([rc_prof, rc_ex])
        o_prof = np.concatenate([o_prof, o_ex])
    write_csv(out_dir / "radius_bias_vs_bif_distance.csv",
              bias_vs_bif_distance(d_prof, g_prof, rc_prof, o_prof))

    far = (np.isfinite(g_prof) & np.isfinite(rc_prof)
           & (d_prof >= FAR_FIELD_RADII) & (g_prof > 0))
    if far.any():
        rel_far = (rc_prof[far] - g_prof[far]) / g_prof[far]
        far_median = float(np.median(rel_far) * 100.0)
        far_mean = float(np.mean(rel_far) * 100.0)
    else:
        far_median = far_mean = math.nan

    bodies = mesh.split_bodies()
    summary: dict[str, Any] = {
        "graph": str(graph_xml),
        "stl": [str(p) for p in stl_paths],
        "surfaces": mesh_info,
        "n_centreline_points": int(len(coords)),
        "graph_coverage_fraction": coverage,
        "n_points_compared": int(sel.sum()),
        "n_inside": n_state["inside"],
        "n_missed": n_state["missed"],
        "n_cropped": n_state["cropped"],
        "cropped_fraction": float(n_state["cropped"] / max(len(state), 1)),
        "in_domain_fraction": float(in_domain.mean()),
        "crop_margin_radius_factor": CROP_MARGIN_RADIUS_FACTOR,
        "crop_margin_min_mm": CROP_MARGIN_MIN_MM,
        "excluded_segments": sorted(exclude_segments or set()),
        "bif_skip_points": bif_skip,
        "bif_skip_radii": bif_skip_radii,
        # Headline radius bias, taken from the far field rather than from a
        # chosen exclusion zone. Only meaningful if the profile in
        # radius_bias_vs_bif_distance.csv is actually flat by this distance.
        "far_field_radii": FAR_FIELD_RADII,
        "radius_rel_bias_far_field_median_pct": far_median,
        "radius_rel_bias_far_field_mean_pct": far_mean,
        "n_far_field_points": int(far.sum()),
        "n_rays_per_station": n_rays,
        "mean_ray_hit_fraction": float(np.mean(hit_frac)),
        "median_axis_offset_frac": float(
            np.nanmedian(meas["axis_offset_frac"])
            if np.isfinite(meas["axis_offset_frac"]).any() else math.nan),
        "median_axis_offset_mm": float(
            np.nanmedian(meas["axis_offset_mm"])
            if np.isfinite(meas["axis_offset_mm"]).any() else math.nan),
        "median_cross_section_anisotropy": float(
            np.nanmedian(meas["anisotropy"])
            if np.isfinite(meas["anisotropy"]).any() else math.nan),
        "centreline_containment": float(np.mean(enclosed)),
        "beta0_surface_components": int(len(bodies)),
        "surface_n_vertices": int(mesh.n_points),
        "surface_n_triangles": int(mesh.n_cells),
        "voxel_mm": voxel_mm,
    }
    summary["radius_abs_error_mm"] = _stats(np.abs(r_recon - r_in))

    if not skip_voxel:
        origin = b[:, 0] - 2.0 * voxel_mm
        extent = (b[:, 1] + 2.0 * voxel_mm) - origin
        dims = tuple(int(math.ceil(e / voxel_mm)) + 1 for e in extent)
        n_vox = dims[0] * dims[1] * dims[2]
        print(f"[valid] voxelising at {voxel_mm} mm -> {dims} ({n_vox / 1e6:.0f}M voxels)")

        # Ground truth is built from in-domain graph only. Including the cropped
        # vessels would inflate the true volume with geometry the reconstruction
        # was never asked for, depressing Dice and Tsens for no real defect.
        c_dom, r_dom, comp_dom = c_in[in_domain], r_in[in_domain], comp_in[in_domain]
        pred_vol = voxelize_surface(mesh, origin, voxel_mm, dims)
        gt_vol = rasterize_graph_volume(c_dom, r_dom, origin, voxel_mm, dims)
        gt_skel = rasterize_points(c_dom, origin, voxel_mm, dims)

        from skimage.morphology import skeletonize
        print("[valid] skeletonising reconstruction ...")
        pred_skel = skeletonize(pred_vol).astype(bool)

        # Whole-graph score, kept for reference; it charges the surface for every
        # tree it never represented, so read the per-tree table instead.
        summary.update(cl_dice(gt_vol, pred_vol, gt_skel, pred_skel))
        # Vessels thinner than two voxels cannot be represented at this spacing.
        summary["fraction_vessels_resolved"] = float(
            np.mean(r_in >= voxel_mm)
        )
        summary["voxel_grid_dims"] = list(dims)

        tree_rows = per_tree_voxel_metrics(
            pred_vol, pred_skel, origin, voxel_mm, dims, c_dom, r_dom, comp_dom)
        # Carry the coverage split onto each row so a low score can be read as
        # "mostly cropped" rather than mistaken for a poor reconstruction. Every
        # component gets a row, including one that was cropped away entirely —
        # dropping it would hide that the surface omits a whole tree.
        by_name = {str(r["tree"]): r for r in tree_rows}
        for c in sorted({int(x) for x in comp_in}):
            name = f"tree{c}"
            row = by_name.get(name)
            if row is None:
                row = {"tree": name, "n_centreline_points": 0,
                       "cl_dice": math.nan, "topology_precision": math.nan,
                       "topology_sensitivity": math.nan, "dice": math.nan,
                       "jaccard": math.nan}
                tree_rows.append(row)
            m = comp_in == c
            n_tot = int(m.sum())
            for s in ("inside", "missed", "cropped"):
                row[f"n_{s}"] = int((state[m] == s).sum())
            row["n_graph_points"] = n_tot
            row["cropped_fraction"] = (row["n_cropped"] / n_tot) if n_tot else math.nan

            # Voxel Tsens is resolution-limited: a vessel thinner than ~2 voxels
            # rasterises away even where the exact point-in-mesh test says the
            # centreline is enclosed. Since the graph *is* the ground-truth
            # skeleton, that term can be evaluated exactly on the mesh instead.
            n_dom = row["n_inside"] + row["n_missed"]
            t_sens_exact = (row["n_inside"] / n_dom) if n_dom else math.nan
            row["topology_sensitivity_exact"] = t_sens_exact
            t_prec = row["topology_precision"]
            denom = (t_prec + t_sens_exact) if np.isfinite(t_prec) else math.nan
            row["cl_dice_exact"] = (
                2.0 * t_prec * t_sens_exact / denom
                if np.isfinite(denom) and denom else math.nan
            )
        tree_rows.sort(key=lambda r: str(r["tree"]))
        # The coverage split is diagnostic, not a result: it stays in the console
        # log and in validation_summary.json, and is stripped from the published
        # table so the scores are not read as coverage statistics.
        crop_keys = ("n_inside", "n_missed", "n_cropped", "n_graph_points",
                     "cropped_fraction")
        summary["coverage_per_tree"] = {
            str(r["tree"]): {k: r[k] for k in crop_keys if k in r}
            for r in tree_rows
        }
        write_csv(out_dir / "cldice_per_tree.csv",
                  [{k: v for k, v in r.items() if k not in crop_keys}
                   for r in tree_rows])
        summary["n_graph_components"] = len(tree_rows)
        for row in tree_rows:
            print(f"[valid]   {row['tree']}: clDice={row['cl_dice']:.3f} "
                  f"(exact {row['cl_dice_exact']:.3f})  "
                  f"Tprec={row['topology_precision']:.3f} "
                  f"Tsens={row['topology_sensitivity']:.3f} "
                  f"(exact {row['topology_sensitivity_exact']:.3f})  "
                  f"Dice={row['dice']:.3f}  "
                  f"n_in={row['n_inside']} missed={row['n_missed']} "
                  f"cropped={row['n_cropped']} "
                  f"({row['cropped_fraction'] * 100:.0f}%)")

    (out_dir / "validation_summary.json").write_text(json.dumps(summary, indent=2))
    # Coverage counts are diagnostic; they stay in the JSON but are kept out of the
    # results CSV.
    _diag = {"n_inside", "n_missed", "n_cropped", "cropped_fraction",
             "in_domain_fraction", "crop_margin_radius_factor", "crop_margin_min_mm"}
    write_csv(out_dir / "validation_summary.csv", [
        {"metric": k, "value": v} for k, v in summary.items()
        if isinstance(v, (int, float)) and k not in _diag
    ])
    print(f"[valid] wrote results to {out_dir}")
    return summary


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--graph", type=Path, required=True)
    ap.add_argument("--stl", type=Path, action="append", required=True,
                    help="reconstructed surface; repeat for a tree delivered as "
                         "several STLs (they are merged and validated together)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--voxel-mm", type=float, default=0.15,
                    help="voxel size for clDice / Dice (default 0.15 mm)")
    ap.add_argument("--skip-voxel", action="store_true",
                    help="radius agreement only; skip the voxel-based scores")
    ap.add_argument("--bif-skip", type=int, default=BIF_SKIP_POINTS,
                    help="centreline points excluded either side of a bifurcation")
    ap.add_argument("--bif-skip-radii", type=float, default=BIF_SKIP_RADII,
                    help="additionally exclude stations within this multiple of "
                         "the local radius (arc length) of a bifurcation; 0 "
                         "disables. A fixed point count under-excludes thick "
                         "vessels, which inflates their measured radius")
    ap.add_argument("--exclude-seg", type=int, action="append", default=[],
                    help="segment id to drop before scoring; repeat as needed")
    ap.add_argument("--rays", type=int, default=32,
                    help="rays cast per station for the cross-sectional radius")
    args = ap.parse_args(argv)
    run_validation(args.graph, args.stl, args.out, args.voxel_mm, args.skip_voxel,
                   args.bif_skip, args.rays, set(args.exclude_seg),
                   args.bif_skip_radii)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
