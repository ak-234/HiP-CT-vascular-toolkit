"""Collapse-aware Jin MCP: a medial-geodesic medialness and cross-section marking.

:mod:`~.jin_mcp` is Jin et al. 2016 as published, and it keeps a known failure on
view: a flattened lumen's medial set is a *sheet*, collision impact (LSF) is high
across the whole sheet, so the minimum-cost path has no unique ridge to follow, and
the 2 x FDT dilation -- FDT being the slit's half-*thickness* -- leaves ribbons either
side unmarked, which the next round explains with parallel "ladder" branches.

This module adds the two changes the methodology note proposes, as hooks on the
unchanged control loop (:func:`~.jin_mcp.extract`), each switchable for ablation:

``medialness="mgf"``
    Classify the strong-quench set locally by PCA (:func:`medial_dimension`): a curve
    where it is 1-D, a sheet where it is 2-D. On sheet voxels compute a voxelised
    **medial geodesic function** in the sense of Dey & Sun (SGP 2006) -- ``f(x)`` is
    the shortest geodesic *along the lumen boundary* between the two points where the
    maximal ball at ``x`` touches it. It has no local minimum on the sheet, and its
    ridge (negative divergence of the unit gradient) is the curve down the middle.
    ``xi* = LSF * w(div)`` there, LSF unchanged on curves. ``mgf_mode="exact"`` does
    the boundary geodesic per sheet voxel; ``"rim"`` is the cheap proxy, in-sheet
    geodesic distance from the sheet's in-plane rim, which has the same ridge on a slit.

``marking="section"``
    After the paper's dilation, also mark the connected lumen in a one-voxel slab
    normal to each accepted path, so the whole slit width is explained rather than an
    inscribed ball. A slab whose component reaches its window edge is **not** marked
    -- the rule ``radius-perimeter`` uses: a section that never closes is not a
    cross-section, and marking it could swallow a neighbouring vessel.

Everything is in cubic-grid voxels, like :mod:`~.jin_mcp`; FDT values exported are
thickness placeholders, not perimeter radii.
"""
from __future__ import annotations

import heapq

import numpy as np
from numba import njit, prange
from scipy import ndimage

from . import jin_mcp, jin_sparse
from .jin_mcp import LENGTHS, OFFSETS, _distances
from .jin_sparse import _neighbour

MEDIALNESS = ("lsf", "mgf")
MARKING = ("dilate", "section")
MGF_MODES = ("rim", "exact")
CLASSIFIERS = ("hessian", "pca")


# ---------------------------------------------------------------- medial dimension


@njit(cache=True)
def _medial_pca(strong, fdt, window_radii, min_w, max_w, sheet_ratio, sheet_width_radii):
    nz, ny, nx = strong.shape
    dim = np.zeros(strong.shape, dtype=np.int8)
    normal = np.zeros((nz, ny, nx, 3), dtype=np.float32)
    width = np.zeros(strong.shape, dtype=np.float32)
    for z in range(nz):
        for y in range(ny):
            for x in range(nx):
                if not strong[z, y, x]:
                    continue
                w = int(np.ceil(window_radii * fdt[z, y, x]))
                w = min(max(w, min_w), max_w)
                n = 0
                s = np.zeros(3)
                ss = np.zeros((3, 3))
                for dz in range(-w, w + 1):
                    zz = z + dz
                    if zz < 0 or zz >= nz:
                        continue
                    for dy in range(-w, w + 1):
                        yy = y + dy
                        if yy < 0 or yy >= ny:
                            continue
                        for dx in range(-w, w + 1):
                            xx = x + dx
                            if xx < 0 or xx >= nx or dz * dz + dy * dy + dx * dx > w * w:
                                continue
                            if strong[zz, yy, xx]:
                                v = np.array([dz, dy, dx], dtype=np.float64)
                                s += v
                                ss += np.outer(v, v)
                                n += 1
                if n < 4:
                    dim[z, y, x] = 1
                    continue
                mean = s / n
                cov = ss / n - np.outer(mean, mean)
                e, vec = np.linalg.eigh(cov)
                # A uniform strip of width W has variance W^2/12 across it.
                wide = np.sqrt(12.0 * max(e[1], 0.0))
                width[z, y, x] = wide
                sheet = e[1] > sheet_ratio * e[2] and wide > sheet_width_radii * fdt[z, y, x]
                dim[z, y, x] = 2 if sheet else 1
                for k in range(3):
                    normal[z, y, x, k] = vec[k, 0]
    return dim, normal, width


def hessian_dimension(mask, lsf, *, strong_threshold=0.5, sigma=1.0, sheet_ratio=0.4):
    """Label strong-quench voxels by the ridge dimension of the smoothed EDT.

    At a tube's axis the distance function falls off in *both* cross-section
    directions, so the two most negative Hessian eigenvalues are comparable; on a
    slit's mid-plane it falls off across the thickness only. The ratio of the second
    to the most negative eigenvalue is ~1 on a curve ridge and ~0 on a sheet ridge:
    on the phantoms, median 0.95-1.00 on round segments, 0.12 at lambda 0.5 and
    0.03-0.11 at lambda <= 0.3. The sheet normal is the most negative eigenvector.

    This is the default rather than the PCA of :func:`medial_dimension` because LSF
    on a voxelised round tube has digital "spokes": the strong-quench set there is a
    broad, roughly planar cluster, and PCA labelled most round segments as sheets.
    """
    mask = np.asarray(mask, dtype=bool)
    strong = (np.asarray(lsf) > strong_threshold) & mask
    edt = ndimage.distance_transform_edt(np.pad(mask, 2))[2:-2, 2:-2, 2:-2]
    if sigma:
        edt = ndimage.gaussian_filter(edt, sigma)
    idx = np.nonzero(strong)
    grads = np.gradient(edt)
    hess = np.empty((len(idx[0]), 3, 3))
    for i in range(3):
        gi = np.gradient(grads[i])
        for j in range(3):
            hess[:, i, j] = gi[j][idx]
    hess = 0.5 * (hess + np.swapaxes(hess, 1, 2))
    ev, vec = np.linalg.eigh(hess)
    ratio = ev[:, 1] / np.minimum(ev[:, 0], -1e-9)
    dim = np.zeros(mask.shape, dtype=np.int8)
    dim[idx] = np.where(ratio < sheet_ratio, 2, 1)
    normal = np.zeros(mask.shape + (3,), dtype=np.float32)
    normal[idx] = vec[:, :, 0]
    return dim, normal


def sheet_support(dim):
    """Sheet voxels plus the strong voxels touching them.

    The rim rows of a sheet bend across its width, so the Hessian labels them
    "curve" -- and a curve keeps its full LSF, which let the path ride along the
    slit's *edge*. Giving them the MGF too weights them as what they are, the low
    rim of the sheet; at a genuine tube-to-sheet transition the MGF ridge simply
    continues along the axis.
    """
    sheet = dim == 2
    return sheet | (ndimage.binary_dilation(sheet, structure=np.ones((3, 3, 3))) & (dim > 0))


def medial_dimension(lsf, fdt, *, strong_threshold=0.5, window_radii=2.0, min_window=3,
                     max_window=16, sheet_ratio=0.25, sheet_width_radii=2.0,
                     return_width=False):
    """Label strong-quench voxels 1 (curve) or 2 (sheet) by PCA; 0 elsewhere.

    The methodology note's classifier, kept as ``classifier="pca"`` for comparison;
    :func:`hessian_dimension` is the default for the reason given there.

    PCA of the strong set inside a ball of ``window_radii * FDT`` (clamped to
    `min_window`..`max_window` voxels). A sheet needs **both** two comparable
    eigenvalues and an in-plane width over `sheet_width_radii` times the local FDT.
    The second test is what separates a collapsed lumen from a round one: a voxelised
    round tube's strong-quench set is a broad cluster about the axis, comparable in
    every direction but no wider than the radius, while a slit's medial sheet is
    ``2(a - b^2/a)`` wide against an FDT of ``b``.

    Also returns each voxel's local normal (smallest eigenvector), which is how the
    rim of a sheet is told apart from its two faces.
    """
    strong = np.asarray(lsf) > strong_threshold
    dim, normal, width = _medial_pca(
        strong, np.asarray(fdt, dtype=np.float64), float(window_radii), int(min_window),
        int(max_window), float(sheet_ratio), float(sheet_width_radii))
    return (dim, normal, width) if return_width else (dim, normal)


# ----------------------------------------------------------- medial geodesic fn


@njit(cache=True)
def _rim(sheet, strong, mask, normal, offsets, lengths, in_plane):
    nz, ny, nx = sheet.shape
    rim = np.zeros(sheet.shape, dtype=np.bool_)
    for z in range(nz):
        for y in range(ny):
            for x in range(nx):
                if not sheet[z, y, x]:
                    continue
                for k in range(offsets.shape[0]):
                    proj = abs(offsets[k, 0] * normal[z, y, x, 0] + offsets[k, 1] * normal[z, y, x, 1]
                               + offsets[k, 2] * normal[z, y, x, 2]) / lengths[k]
                    if proj > in_plane:
                        continue
                    zz, yy, xx = z + offsets[k, 0], y + offsets[k, 1], x + offsets[k, 2]
                    if zz < 0 or zz >= nz or yy < 0 or yy >= ny or xx < 0 or xx >= nx:
                        rim[z, y, x] = True
                        break
                    if not mask[zz, yy, xx] or not strong[zz, yy, xx]:
                        rim[z, y, x] = True
                        break
    return rim


def _mgf_rim(mask, sheet, strong, normal):
    """In-sheet geodesic distance from the sheet's in-plane rim."""
    rim = _rim(sheet, strong, mask, normal, OFFSETS, LENGTHS, 0.5)
    seeds = np.flatnonzero(rim)
    if not len(seeds):
        return np.where(sheet, 0.0, np.nan)
    mu = sheet.astype(np.float64)
    f, _ = _distances(mu, OFFSETS, LENGTHS, seeds, np.zeros(len(seeds)), mu, 0)
    return np.where(sheet & np.isfinite(f), f, np.nan)


@njit(cache=True)
def _boundary_geodesic(boundary, a, b, half, offsets, lengths):
    """Dijkstra on boundary voxels from `a` to `b` inside a cube about their midpoint."""
    nz, ny, nx = boundary.shape
    lo = np.empty(3, dtype=np.int64)
    hi = np.empty(3, dtype=np.int64)
    dims = np.array([nz, ny, nx])
    for k in range(3):
        c = (a[k] + b[k]) // 2
        lo[k] = max(c - half, 0)
        hi[k] = min(c + half + 1, dims[k])
    sz, sy, sx = hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]
    dist = np.full(sz * sy * sx, np.inf)
    start = ((a[0] - lo[0]) * sy + (a[1] - lo[1])) * sx + (a[2] - lo[2])
    goal = ((b[0] - lo[0]) * sy + (b[1] - lo[1])) * sx + (b[2] - lo[2])
    if start < 0 or goal < 0 or start >= dist.size or goal >= dist.size:
        return np.inf
    dist[start] = 0.0
    heap = [(0.0, start)]
    while heap:
        d, p = heapq.heappop(heap)
        if d != dist[p]:
            continue
        if p == goal:
            return d
        z, y, x = p // (sy * sx), (p // sx) % sy, p % sx
        for k in range(offsets.shape[0]):
            zz, yy, xx = z + offsets[k, 0], y + offsets[k, 1], x + offsets[k, 2]
            if zz < 0 or zz >= sz or yy < 0 or yy >= sy or xx < 0 or xx >= sx:
                continue
            if not boundary[zz + lo[0], yy + lo[1], xx + lo[2]]:
                continue
            q = (zz * sy + yy) * sx + xx
            new = d + lengths[k]
            if new < dist[q]:
                dist[q] = new
                heapq.heappush(heap, (new, q))
    return np.inf


@njit(cache=True)
def _mgf_exact_kernel(sheet_idx, feat, boundary, offsets, lengths, start_half, max_half):
    n = sheet_idx.shape[0]
    out = np.full(n, np.nan)
    nz, ny, nx = feat.shape[1], feat.shape[2], feat.shape[3]
    for i in range(n):
        z, y, x = sheet_idx[i, 0], sheet_idx[i, 1], sheet_idx[i, 2]
        a = np.array([feat[0, z, y, x], feat[1, z, y, x], feat[2, z, y, x]])
        here = np.array([z, y, x])
        ax = (a - here).astype(np.float64)
        na = np.sqrt(np.sum(ax * ax))
        if na <= 0:
            continue
        # The most *opposite* contact, not the farthest: "farthest from a" prefers a
        # diagonal neighbour's contact on ties and shifts the ridge half a voxel.
        best = 2.0
        b = a.copy()
        for k in range(offsets.shape[0]):
            zz, yy, xx = z + offsets[k, 0], y + offsets[k, 1], x + offsets[k, 2]
            if zz < 0 or zz >= nz or yy < 0 or yy >= ny or xx < 0 or xx >= nx:
                continue
            c = np.array([feat[0, zz, yy, xx], feat[1, zz, yy, xx], feat[2, zz, yy, xx]])
            cx = (c - here).astype(np.float64)
            nc = np.sqrt(np.sum(cx * cx))
            if nc <= 0:
                continue
            cos = np.sum(cx * ax) / (nc * na)
            if cos < best - 1e-9:
                best = cos
                b = c
        if best > 0:
            continue  # no contact across the sheet: not a two-sided point
        half = start_half
        while half <= max_half:
            g = _boundary_geodesic(boundary, a, b, half, offsets, lengths)
            if np.isfinite(g):
                out[i] = g
                break
            half *= 2
    return out


def _mgf_exact(mask, sheet, *, start_half=8, max_half=64):
    """Dey & Sun's MGF on voxels: boundary geodesic between antipodal contacts.

    The first contact is ``x``'s nearest background voxel (EDT feature transform);
    the second is the neighbours' feature point farthest from it -- across the sheet.
    Paths run over background voxels 26-adjacent to the lumen, in a padded frame so
    a vessel touching the ROI face still has a boundary to walk around.
    """
    pad = np.pad(np.asarray(mask, dtype=bool), 1)
    _, feat = ndimage.distance_transform_edt(pad, return_indices=True)
    boundary = ~pad & ndimage.binary_dilation(pad, structure=np.ones((3, 3, 3)))
    idx = np.argwhere(np.pad(sheet, 1)).astype(np.int64)
    values = _mgf_exact_kernel(idx, feat.astype(np.int64), boundary, OFFSETS, LENGTHS,
                               int(start_half), int(max_half))
    f = np.full(pad.shape, np.nan)
    f[tuple(idx.T)] = values
    return f[1:-1, 1:-1, 1:-1]


def ridge_weight(f, sheet, *, theta=0.3, w_min=0.05, sigma=1.0):
    """``w`` in (0, 1]: near 1 on the MGF ridge, `w_min` on the rest of the sheet.

    ``w = w_min + (1 - w_min)(1 - exp(-max(0, -div g) / theta))`` with ``g`` the unit
    gradient of `f` after a support-normalised Gaussian of width `sigma` voxels. Off
    the sheet ``w = 1``, so a tubular segment's medialness is untouched.
    """
    support = sheet & np.isfinite(f)
    if not support.any():
        return np.ones(sheet.shape)
    filled = np.where(support, f, 0.0)
    if sigma:
        num = ndimage.gaussian_filter(filled, sigma)
        den = ndimage.gaussian_filter(support.astype(np.float64), sigma)
        filled = np.where(den > 1e-6, num / np.maximum(den, 1e-6), 0.0)
    grads = np.gradient(filled)
    norm = np.sqrt(sum(g * g for g in grads))
    unit = [np.where(norm > 1e-9, g / np.maximum(norm, 1e-9), 0.0) for g in grads]
    div = sum(np.gradient(u, axis=k) for k, u in enumerate(unit))
    w = w_min + (1 - w_min) * (1 - np.exp(-np.maximum(-div, 0) / theta))
    return np.where(sheet, w, 1.0)


# --------------------------------------------------------- cross-section marking


@njit(cache=True)
def _fill_section(mask, p, t, half, slab, out, offsets):
    """Mark the lumen component through `p` in the slab normal to `t`, if it closes."""
    nz, ny, nx = mask.shape
    side = 2 * half + 1
    seen = np.zeros(side * side * side, dtype=np.bool_)
    stack = [(p[0], p[1], p[2])]
    found = []
    seen[(half * side + half) * side + half] = True
    while stack:
        z, y, x = stack.pop()
        found.append((z, y, x))
        for k in range(offsets.shape[0]):
            zz, yy, xx = z + offsets[k, 0], y + offsets[k, 1], x + offsets[k, 2]
            dz, dy, dx = zz - p[0], yy - p[1], xx - p[2]
            if abs(dz * t[0] + dy * t[1] + dx * t[2]) > slab:
                continue
            if zz < 0 or zz >= nz or yy < 0 or yy >= ny or xx < 0 or xx >= nx:
                continue
            if not mask[zz, yy, xx]:
                continue
            if abs(dz) >= half or abs(dy) >= half or abs(dx) >= half:
                return False  # still open at the window edge: not a cross-section
            key = ((dz + half) * side + (dy + half)) * side + (dx + half)
            if seen[key]:
                continue
            seen[key] = True
            stack.append((zz, yy, xx))
    for z, y, x in found:
        out[z, y, x] = True
    return True


def section_marker(*, fill_max_radii=12.0, max_half=64, slab=0.87, tangent_radii=2.0,
                   stats=None):
    """A :func:`~.jin_mcp.extract` marker filling each path's normal sections.

    The tangent is a chord over ``tangent_radii * FDT`` either side (at least two
    voxels) -- the same reasoning as ``RECENTRE_TANGENT_RADII``: a neighbour-to-
    neighbour difference on a voxel path rotates by tens of degrees.
    """
    stats = {} if stats is None else stats

    def marker(paths, fdt, mask):
        out = np.zeros(mask.shape, dtype=bool)
        m = np.asarray(mask, dtype=bool)
        for path in paths:
            c = np.column_stack(np.unravel_index(np.asarray(path, dtype=np.int64), m.shape))
            if len(c) < 2:
                continue
            s = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(c, axis=0), axis=1))])
            f = fdt.ravel()[np.asarray(path)]
            for i in range(len(c)):
                span = max(2.0, tangent_radii * float(f[i]))
                j0 = int(np.searchsorted(s, s[i] - span))
                j1 = min(int(np.searchsorted(s, s[i] + span)), len(c) - 1)
                t = (c[j1] - c[j0]).astype(np.float64)
                nt = np.linalg.norm(t)
                if nt < 1e-9:
                    continue
                half = int(min(max_half, np.ceil(fill_max_radii * max(float(f[i]), 1.0)) + 1))
                ok = _fill_section(m, c[i].astype(np.int64), t / nt, half, float(slab), out,
                                   OFFSETS)
                key = "sections_marked" if ok else "sections_refused"
                stats[key] = stats.get(key, 0) + 1
        return out

    return marker


# ------------------------------------------------------------- sparse versions
#
# The same medialness for :mod:`~.jin_sparse`, on lumen voxels. The dense functions
# above hold several float64 fields and an int64 feature transform over the box --
# 24 bytes a voxel for the transform alone, impossible on a 24 G-voxel tree box. Here
# only the smoothed EDT is box-sized (float32), everything else is per lumen voxel,
# and the per-sheet-voxel MGF runs across all cores.


TILE = 64
TILE_THREADS = 8


def _tiles(zyx, shape, *, halo, tile=TILE):
    """``(lo, hi, own, near)`` for each `tile`-cube holding any of the points `zyx`.

    `lo`/`hi` bound the tile plus `halo`, clipped to `shape`; `own` indexes the points
    in the tile itself and `near` every point inside the haloed box. A crop at the box
    face ends where the box does, so a filter run on it sees the box's own boundary.
    """
    shape = np.asarray(shape, dtype=np.int64)
    grid = (shape + tile - 1) // tile
    keys = np.ravel_multi_index(tuple((zyx // tile).T), tuple(grid))
    order = np.argsort(keys, kind="stable")
    sorted_keys = keys[order]
    occupied, starts = np.unique(sorted_keys, return_index=True)
    ends = np.r_[starts[1:], len(sorted_keys)]
    span = {int(k): (int(a), int(b)) for k, a, b in zip(occupied, starts, ends)}
    reach = -(-halo // tile)
    steps = np.array([(a, b, c) for a in range(-reach, reach + 1)
                      for b in range(-reach, reach + 1) for c in range(-reach, reach + 1)])
    for key in occupied.tolist():
        cell = np.array(np.unravel_index(key, tuple(grid)))
        lo = np.maximum(cell * tile - halo, 0)
        hi = np.minimum(cell * tile + tile + halo, shape)
        own = order[slice(*span[key])]
        near = []
        for c in cell + steps:
            if np.all(c >= 0) and np.all(c < grid):
                hit = span.get(int(np.ravel_multi_index(tuple(c), tuple(grid))))
                if hit is not None:
                    near.append(order[slice(*hit)])
        near = np.concatenate(near)
        near = near[np.all((zyx[near] >= lo) & (zyx[near] < hi), axis=1)]
        yield lo, hi, own, near


def _run_tiles(work, tiles, threads=TILE_THREADS):
    """Run `work` on every tile across threads; SciPy, NumPy and numba release the GIL."""
    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(max_workers=threads) as pool:
        for _ in pool.map(work, tiles):
            pass


@njit(cache=True, nogil=True)
def _d1(a, z, y, x, axis):
    """``np.gradient`` along `axis` at one voxel: central inside, one-sided at the edge."""
    n = a.shape[axis]
    i = (z, y, x)[axis]
    if n < 2:
        return 0.0
    lo = i - 1 if i > 0 else i
    hi = i + 1 if i < n - 1 else i
    if axis == 0:
        va, vb = a[lo, y, x], a[hi, y, x]
    elif axis == 1:
        va, vb = a[z, lo, x], a[z, hi, x]
    else:
        va, vb = a[z, y, lo], a[z, y, hi]
    return (vb - va) / (hi - lo)


@njit(cache=True, nogil=True)
def _hessian_at(a, coords):
    """``np.gradient(np.gradient(a)[i])[j]`` at each of `coords`, as the dense code takes."""
    n = coords.shape[0]
    out = np.empty((n, 3, 3))
    for t in range(n):
        z, y, x = coords[t, 0], coords[t, 1], coords[t, 2]
        for i in range(3):
            for j in range(3):
                m = a.shape[j]
                c = (z, y, x)[j]
                lo = c - 1 if c > 0 else c
                hi = c + 1 if c < m - 1 else c
                if m < 2:
                    out[t, i, j] = 0.0
                    continue
                if j == 0:
                    ga, gb = _d1(a, lo, y, x, i), _d1(a, hi, y, x, i)
                elif j == 1:
                    ga, gb = _d1(a, z, lo, x, i), _d1(a, z, hi, x, i)
                else:
                    ga, gb = _d1(a, z, y, lo, i), _d1(a, z, y, hi, i)
                out[t, i, j] = (gb - ga) / (hi - lo)
    return out


def hessian_dimension_sparse(mask, fg, strong, *, sigma=1.0, sheet_ratio=0.4):
    """:func:`hessian_dimension` on lumen voxels: ``(dim, normal)`` aligned with `fg`.

    The EDT is the multithreaded ``edt`` package with the box face as background --
    what the dense code's 2-voxel zero pad means -- and float32. The Gaussian is taken
    only on the tiles holding strong-quench voxels (:func:`_tiles`, a 6-voxel margin:
    the Gaussian's 4 plus the Hessian's 2), across threads. Over the whole box it was a
    single-threaded pass over 24 G voxels on a LADAF-2021-17 tree for 70 M of lumen;
    every value the Hessian reads is the same as it was.
    """
    import edt

    d = edt.edt(np.ascontiguousarray(mask), black_border=True, parallel=TILE_THREADS)
    idx = np.flatnonzero(strong)
    coords = np.column_stack(np.unravel_index(fg[idx], mask.shape)).astype(np.int64)
    hess = np.empty((len(idx), 3, 3))

    def tile(job):
        lo, hi, own, _near = job
        crop = d[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]
        if sigma:
            crop = ndimage.gaussian_filter(crop, sigma, output=np.float32)
        hess[own] = _hessian_at(crop, coords[own] - lo)

    _run_tiles(tile, _tiles(coords, mask.shape, halo=6))
    d = None  # the box-sized EDT, released before the eigen-decomposition
    hess = 0.5 * (hess + np.swapaxes(hess, 1, 2))
    ev, vec = np.linalg.eigh(hess)
    ratio = ev[:, 1] / np.minimum(ev[:, 0], -1e-9)
    dim = np.zeros(len(fg), dtype=np.int8)
    dim[idx] = np.where(ratio < sheet_ratio, 2, 1)
    normal = np.zeros((len(fg), 3), dtype=np.float32)
    normal[idx] = vec[:, :, 0]
    return dim, normal


@njit(cache=True)
def _sheet_support_compact(lookup, fg, nz, ny, nx, offsets, dim):
    """:func:`sheet_support` on lumen voxels."""
    out = dim == 2
    for i in range(fg.size):
        if dim[i] != 1:
            continue
        for k in range(offsets.shape[0]):
            q = _neighbour(fg[i], k, offsets, nz, ny, nx)
            if q >= 0 and lookup[q] >= 0 and dim[lookup[q]] == 2:
                out[i] = True
                break
    return out


@njit(cache=True)
def _nearest_background(mp, z, y, x, cap):
    """Nearest background voxel of padded mask `mp` to (z, y, x); first in scan order."""
    nz, ny, nx = mp.shape
    if not mp[z, y, x]:
        return z, y, x
    best = 1e18
    bz, by, bx = -1, -1, -1
    for r in range(1, cap + 1):
        for dz in range(-r, r + 1):
            zz = z + dz
            if zz < 0 or zz >= nz:
                continue
            for dy in range(-r, r + 1):
                yy = y + dy
                if yy < 0 or yy >= ny:
                    continue
                for dx in range(-r, r + 1):
                    if max(abs(dz), abs(dy), abs(dx)) != r:
                        continue  # the shell; the inside was searched already
                    xx = x + dx
                    if xx < 0 or xx >= nx or mp[zz, yy, xx]:
                        continue
                    d2 = dz * dz + dy * dy + dx * dx
                    if d2 < best:
                        best, bz, by, bx = d2, zz, yy, xx
        if best <= r * r:  # nothing outside this cube can be nearer
            return bz, by, bx
    return bz, by, bx


@njit(cache=True)
def _is_boundary(mp, z, y, x, offsets):
    if mp[z, y, x]:
        return False
    nz, ny, nx = mp.shape
    for k in range(offsets.shape[0]):
        zz, yy, xx = z + offsets[k, 0], y + offsets[k, 1], x + offsets[k, 2]
        if 0 <= zz < nz and 0 <= yy < ny and 0 <= xx < nx and mp[zz, yy, xx]:
            return True
    return False


@njit(cache=True)
def _boundary_geodesic_fly(mp, a, b, half, offsets, lengths):
    """:func:`_boundary_geodesic` testing "boundary" on the fly from the mask."""
    nz, ny, nx = mp.shape
    lo = np.empty(3, dtype=np.int64)
    hi = np.empty(3, dtype=np.int64)
    dims = np.array([nz, ny, nx])
    for k in range(3):
        c = (a[k] + b[k]) // 2
        lo[k] = max(c - half, 0)
        hi[k] = min(c + half + 1, dims[k])
    sz, sy, sx = hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]
    dist = np.full(sz * sy * sx, np.inf)
    start = ((a[0] - lo[0]) * sy + (a[1] - lo[1])) * sx + (a[2] - lo[2])
    goal = ((b[0] - lo[0]) * sy + (b[1] - lo[1])) * sx + (b[2] - lo[2])
    if start < 0 or goal < 0 or start >= dist.size or goal >= dist.size:
        return np.inf
    dist[start] = 0.0
    heap = [(0.0, start)]
    while heap:
        d, p = heapq.heappop(heap)
        if d != dist[p]:
            continue
        if p == goal:
            return d
        z, y, x = p // (sy * sx), (p // sx) % sy, p % sx
        for k in range(offsets.shape[0]):
            zz, yy, xx = z + offsets[k, 0], y + offsets[k, 1], x + offsets[k, 2]
            if zz < 0 or zz >= sz or yy < 0 or yy >= sy or xx < 0 or xx >= sx:
                continue
            if not _is_boundary(mp, zz + lo[0], yy + lo[1], xx + lo[2], offsets):
                continue
            q = (zz * sy + yy) * sx + xx
            new = d + lengths[k]
            if new < dist[q]:
                dist[q] = new
                heapq.heappush(heap, (new, q))
    return np.inf


@njit(cache=True, parallel=True)
def _mgf_exact_sparse_kernel(mp, coords, offsets, lengths, start_half, max_half, cap):
    """:func:`_mgf_exact_kernel` with contacts found by local search, across cores."""
    n = coords.shape[0]
    out = np.full(n, np.nan)
    nz, ny, nx = mp.shape
    for i in prange(n):
        z, y, x = coords[i, 0], coords[i, 1], coords[i, 2]
        az, ay, ax_ = _nearest_background(mp, z, y, x, cap)
        if az < 0:
            continue
        a = np.array([az, ay, ax_])
        here = np.array([z, y, x])
        av = (a - here).astype(np.float64)
        na = np.sqrt(np.sum(av * av))
        if na <= 0:
            continue
        best = 2.0
        b = a.copy()
        for k in range(offsets.shape[0]):
            zz, yy, xx = z + offsets[k, 0], y + offsets[k, 1], x + offsets[k, 2]
            if zz < 0 or zz >= nz or yy < 0 or yy >= ny or xx < 0 or xx >= nx:
                continue
            cz, cy, cx_ = _nearest_background(mp, zz, yy, xx, cap)
            if cz < 0:
                continue
            c = np.array([cz, cy, cx_])
            cv = (c - here).astype(np.float64)
            nc = np.sqrt(np.sum(cv * cv))
            if nc <= 0:
                continue
            cos = np.sum(cv * av) / (nc * na)
            if cos < best - 1e-9:
                best = cos
                b = c
        if best > 0:
            continue
        half = start_half
        while half <= max_half:
            g = _boundary_geodesic_fly(mp, a, b, half, offsets, lengths)
            if np.isfinite(g):
                out[i] = g
                break
            half *= 2
    return out


def _mgf_exact_sparse(mask, fg, sheet, fdt, *, start_half=8, max_half=64):
    """:func:`_mgf_exact` on lumen voxels: ``f`` aligned with `fg` (NaN off the sheet).

    Contacts come from a local search rather than a box-sized feature transform. A
    sheet voxel's thickness is its FDT, so the search radius is small; equidistant
    contacts are broken by scan order, which may differ from scipy's transform.
    """
    mp = np.pad(np.asarray(mask, dtype=bool), 1)
    idx = np.flatnonzero(sheet)
    coords = (np.column_stack(np.unravel_index(fg[idx], mask.shape)) + 1).astype(np.int64)
    cap = int(np.ceil(float(np.max(fdt[idx])) if len(idx) else 1.0)) + 3
    values = _mgf_exact_sparse_kernel(mp, coords, OFFSETS, LENGTHS, int(start_half),
                                      int(max_half), cap)
    f = np.full(len(fg), np.nan)
    f[idx] = values
    return f


@njit(cache=True)
def _rim_compact(lookup, fg, nz, ny, nx, offsets, lengths, sheet, strong, normal, in_plane):
    out = np.zeros(fg.size, dtype=np.bool_)
    for i in range(fg.size):
        if not sheet[i]:
            continue
        for k in range(offsets.shape[0]):
            proj = abs(offsets[k, 0] * normal[i, 0] + offsets[k, 1] * normal[i, 1]
                       + offsets[k, 2] * normal[i, 2]) / lengths[k]
            if proj > in_plane:
                continue
            q = _neighbour(fg[i], k, offsets, nz, ny, nx)
            if q < 0 or lookup[q] < 0 or not strong[lookup[q]]:
                out[i] = True
                break
    return out


@njit(cache=True)
def _geodesic_within(lookup, fg, nz, ny, nx, offsets, lengths, allowed, seeds):
    dist = np.full(fg.size, np.inf)
    heap = [(0.0, np.int64(0))]
    heap.pop()
    for s in seeds:
        dist[s] = 0.0
        heapq.heappush(heap, (0.0, np.int64(s)))
    while heap:
        d, i = heapq.heappop(heap)
        if d != dist[i]:
            continue
        for k in range(offsets.shape[0]):
            q = _neighbour(fg[i], k, offsets, nz, ny, nx)
            if q < 0:
                continue
            j = lookup[q]
            if j < 0 or not allowed[j]:
                continue
            new = d + lengths[k]
            if new < dist[j]:
                dist[j] = new
                heapq.heappush(heap, (new, np.int64(j)))
    return dist


def _mgf_rim_sparse(mask, fg, lookup, sheet, strong, normal):
    """:func:`_mgf_rim` on lumen voxels."""
    nz, ny, nx = mask.shape
    rim = _rim_compact(lookup, fg, nz, ny, nx, OFFSETS, LENGTHS, sheet, strong, normal, 0.5)
    seeds = np.flatnonzero(rim).astype(np.int64)
    if not len(seeds):
        return np.where(sheet, 0.0, np.nan)
    f = _geodesic_within(lookup, fg, nz, ny, nx, OFFSETS, LENGTHS, sheet, seeds)
    return np.where(sheet & np.isfinite(f), f, np.nan)


def ridge_weight_sparse(shape, fg, lookup, sheet, f, *, theta=0.3, w_min=0.05, sigma=1.0,
                        halo=8):
    """:func:`ridge_weight` on lumen voxels, computed tile by tile.

    Each tile holding sheet voxels is cut out with an 8-voxel halo -- the Gaussian's
    reach (4 sigma) plus the two derivative stencils -- together with every other sheet
    voxel inside it, so the dense routine sees exactly the neighbourhood it would see
    over the whole volume. It used to cut one box per sheet component, and a long
    collapsed vessel's box is mostly empty: on a LADAF-2021-17 tree that was over a
    third of the run.
    """
    w = np.ones(len(fg))
    idx = np.flatnonzero(sheet)
    if not len(idx):
        return w
    zyx = np.column_stack(np.unravel_index(fg[idx], shape)).astype(np.int64)

    def tile(job):
        lo, hi, own, near = job
        crop = tuple(hi - lo)
        sheet_c = np.zeros(crop, dtype=bool)
        f_c = np.full(crop, np.nan)
        local = tuple((zyx[near] - lo).T)
        sheet_c[local] = True
        f_c[local] = f[idx[near]]
        w_c = ridge_weight(f_c, sheet_c, theta=theta, w_min=w_min, sigma=sigma)
        w[idx[own]] = w_c[tuple((zyx[own] - lo).T)]

    _run_tiles(tile, _tiles(zyx, shape, halo=halo))
    return w


def sparse_medialness(*, mgf_mode="exact", sheet_ratio=0.4, theta=0.3, w_min=0.05,
                      info=None):
    """The ``medialness`` hook for :func:`~.jin_sparse.extract` (Hessian classifier)."""
    info = {} if info is None else info

    def mgf(mask, fg, lookup, fdt, lsf):
        strong = lsf > .5
        dim, normal = hessian_dimension_sparse(mask, fg, strong, sheet_ratio=sheet_ratio)
        info.update(sheet_voxels=int((dim == 2).sum()), curve_voxels=int((dim == 1).sum()))
        nz, ny, nx = mask.shape
        sheet = _sheet_support_compact(lookup, fg, nz, ny, nx, OFFSETS, dim)
        if not sheet.any():
            return lsf
        if mgf_mode == "rim":
            f = _mgf_rim_sparse(mask, fg, lookup, sheet, strong, normal)
        else:
            f = _mgf_exact_sparse(mask, fg, sheet, fdt)
        return lsf * ridge_weight_sparse(mask.shape, fg, lookup, sheet, f, theta=theta,
                                         w_min=w_min)

    return mgf


# ------------------------------------------------------------------- extraction


def extract(membership, *, root_zyx=None, max_voxels=2_000_000, max_iterations=256,
            medialness="mgf", marking="section", mgf_mode="exact", classifier="hessian",
            sheet_ratio=None, window_radii=2.0, theta=0.3, w_min=0.05,
            fill_max_radii=12.0, engine="dense", allow_loops=False, incremental=True,
            verbose=False):
    """Jin MCP with the selected medialness and marking; see the module docstring.

    `sheet_ratio` defaults per `classifier`: 0.4 for the Hessian eigenvalue ratio,
    0.25 for the PCA one. ``engine="sparse"`` runs on lumen voxels only
    (:mod:`~.jin_sparse`), for whole trees; it supports the Hessian classifier.
    """
    if engine not in ("dense", "sparse"):
        raise ValueError("engine must be 'dense' or 'sparse'")
    if engine == "sparse":
        return _extract_sparse(
            membership, root_zyx=root_zyx, max_iterations=max_iterations,
            medialness=medialness, marking=marking, mgf_mode=mgf_mode,
            classifier=classifier, sheet_ratio=sheet_ratio, theta=theta, w_min=w_min,
            fill_max_radii=fill_max_radii, allow_loops=allow_loops,
            incremental=incremental, verbose=verbose)
    if medialness not in MEDIALNESS:
        raise ValueError(f"medialness must be one of {MEDIALNESS}")
    if marking not in MARKING:
        raise ValueError(f"marking must be one of {MARKING}")
    if mgf_mode not in MGF_MODES:
        raise ValueError(f"mgf_mode must be one of {MGF_MODES}")
    if classifier not in CLASSIFIERS:
        raise ValueError(f"classifier must be one of {CLASSIFIERS}")
    if sheet_ratio is None:
        sheet_ratio = 0.4 if classifier == "hessian" else 0.25
    info: dict = dict(sheet_voxels=0, curve_voxels=0)

    def mgf(mask, fdt, lsf):
        if classifier == "hessian":
            dim, normal = hessian_dimension(mask, lsf, sheet_ratio=sheet_ratio)
        else:
            dim, normal = medial_dimension(lsf, fdt, window_radii=window_radii,
                                           sheet_ratio=sheet_ratio)
        info.update(sheet_voxels=int((dim == 2).sum()), curve_voxels=int((dim == 1).sum()))
        sheet = sheet_support(dim)
        if not sheet.any():
            return lsf
        if mgf_mode == "rim":
            f = _mgf_rim(mask, sheet, np.asarray(lsf) > .5, normal)
        else:
            f = _mgf_exact(mask, sheet)
        return lsf * ridge_weight(f, sheet, theta=theta, w_min=w_min)

    stats: dict = {}
    result = jin_mcp.extract(
        membership, root_zyx=root_zyx, max_voxels=max_voxels, max_iterations=max_iterations,
        medialness=mgf if medialness == "mgf" else None,
        marker=section_marker(fill_max_radii=fill_max_radii, stats=stats)
        if marking == "section" else None, allow_loops=allow_loops,
    )
    result.report.update(
        method="jin-mgf", medialness=medialness, marking=marking,
        mgf_mode=mgf_mode if medialness == "mgf" else None,
        classifier=classifier if medialness == "mgf" else None,
        parameters=dict(sheet_ratio=sheet_ratio, window_radii=window_radii, theta=theta,
                        w_min=w_min, fill_max_radii=fill_max_radii),
        sections_marked=stats.get("sections_marked", 0),
        sections_refused=stats.get("sections_refused", 0), **info,
    )
    return result


def _extract_sparse(membership, *, root_zyx, max_iterations, medialness, marking, mgf_mode,
                    classifier, sheet_ratio, theta, w_min, fill_max_radii, allow_loops,
                    incremental, verbose):
    """:func:`extract` on :mod:`~.jin_sparse`, for whole trees."""
    for name, value, allowed in (("medialness", medialness, MEDIALNESS),
                                 ("marking", marking, MARKING), ("mgf_mode", mgf_mode, MGF_MODES)):
        if value not in allowed:
            raise ValueError(f"{name} must be one of {allowed}")
    if classifier != "hessian":
        raise ValueError("the sparse engine supports classifier='hessian' only")
    sheet_ratio = 0.4 if sheet_ratio is None else sheet_ratio
    info: dict = dict(sheet_voxels=0, curve_voxels=0)
    stats: dict = {}
    result = jin_sparse.extract(
        membership, root_zyx=root_zyx, max_iterations=max_iterations,
        medialness=sparse_medialness(mgf_mode=mgf_mode, sheet_ratio=sheet_ratio,
                                     theta=theta, w_min=w_min, info=info)
        if medialness == "mgf" else None,
        marker=section_marker(fill_max_radii=fill_max_radii, stats=stats)
        if marking == "section" else None,
        allow_loops=allow_loops, incremental=incremental, verbose=verbose,
    )
    result.report.update(
        method="jin-mgf", medialness=medialness, marking=marking,
        mgf_mode=mgf_mode if medialness == "mgf" else None,
        classifier=classifier if medialness == "mgf" else None,
        parameters=dict(sheet_ratio=sheet_ratio, theta=theta, w_min=w_min,
                        fill_max_radii=fill_max_radii),
        sections_marked=stats.get("sections_marked", 0),
        sections_refused=stats.get("sections_refused", 0), **info,
    )
    return result
