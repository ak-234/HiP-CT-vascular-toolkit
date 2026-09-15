"""Audit the reconstruction against the segmented lumen, on the pipeline's own terms.

**Re-inflation is intended, not a bug.** HiP-CT is ex vivo, so lumens are collapsed by
fixation. ``adjust_thickness.py`` compensates deliberately: it finds the
minimum-area reslice through each centreline point, keeps the connected component
containing the plane centre, and assigns

    r = cv2.arcLength(contour) / (2 * pi)

on the assumption that the lumen *perimeter* survives fixation even though the lumen
shape does not. A collapsed slit therefore *should* be reconstructed as a much rounder,
much larger circle. Flagging that as an error -- as an earlier version of this module
did -- is simply flagging the design.

What can be audited is whether the assigned radius actually honours that assumption
*here*, at this cross-section. It often does not, because ``adjust_thickness.py`` does
not keep its per-point measurements: it reduces them to a global linear fit
``r ~ slope * (Amira distance-transform thickness) + intercept`` and applies that
everywhere. Measured over the tree, ``r_stored / r_perimeter`` has median 0.95 -- the
assumption holds on average -- but correlation only ~0.66 and a p5-p95 spread of
0.49-2.08, so at any single location the radius may be out by a factor of two.

Hence three separate things, deliberately not conflated:

``collapse_severity``
    How far from circular the segmented lumen is. **Expected**, informational, ranked so
    the sites leaning hardest on the perimeter assumption can be reviewed by eye.

``perimeter_mismatch``
    The assigned radius disagrees with the perimeter measured at this cross-section.
    This is the actual error: the model is not doing what it claims to do here.

``companion_lumen``
    A second lumen component beside the centreline one. ``clean_and_measure_slice``
    only ever measures the component containing the plane centre, so a companion is
    silently discarded -- the signature of one collapsed vessel segmented as two.

Cross-sections are measured in the plane perpendicular to the local centreline tangent.
For a prismatic tube that is the minimum-area plane, so it closely reproduces the
pipeline's own 3-angle search at a tiny fraction of the cost.

:func:`cut` is that plane cut on its own, because two other things need it:
:mod:`~.edit.skeleton_optimise` re-centres a point on the lumen's in-plane centroid,
and :mod:`~.edit.radius_perimeter` replaces its radius with the perimeter of the same
section. All three ask the same question of the mask, so they ask it in one place.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage

from .candidates import Candidate, _arclength, _edge_tangents, _strahler

# Below roughly four segmentation voxels the binned mask cannot support a shape fit:
# a blob two or three voxels across yields a near-zero minor eigenvalue whatever its
# true shape, and its digitised perimeter is dominated by staircase artefacts.
MIN_RADIUS_UM = 300.0
MIN_BLOB_VOXELS = 12


def _perimeter_um(blob: np.ndarray, spacing: float) -> float:
    """Contour length using the same estimator as the pipeline (``cv2.arcLength``)."""
    import cv2

    contours, _ = cv2.findContours(
        (blob.astype(np.uint8) * 255), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )
    if not contours:
        return 0.0
    return float(max(cv2.arcLength(c, True) for c in contours)) * spacing


@dataclass
class CrossSectionProfile:
    """Per-centreline-point measurements. NaN where no measurement was possible."""

    r_perimeter: np.ndarray  # (P,) perimeter / 2pi, um -- what the design would assign
    r_area: np.ndarray  # (P,) sqrt(area / pi), um
    minor_ratio: np.ndarray  # (P,) minor semi-axis / assumed radius
    major_ratio: np.ndarray  # (P,) major semi-axis / assumed radius
    isoperimetric: np.ndarray  # (P,) P^2 / (4 pi A); 1.0 = circle, higher = collapsed
    fill: np.ndarray  # (P,) fraction of the assumed disc that is segmented
    companions: np.ndarray  # (P,) nearest genuinely separate lumen, um (inf if none)
    pinched: np.ndarray  # (P,) the pipeline's 4-connectivity split this lumen in two
    measured: np.ndarray  # (P,) bool

    def perimeter_ratio(self, thickness: np.ndarray) -> np.ndarray:
        """``r_stored / r_perimeter`` -- 1.0 means the design assumption is honoured."""
        with np.errstate(divide="ignore", invalid="ignore"):
            return thickness / self.r_perimeter

    def summary(self, thickness: np.ndarray) -> str:
        m = self.measured
        if not m.any():
            return "no cross-sections measured"
        ratio = self.perimeter_ratio(thickness)[m]
        ratio = ratio[np.isfinite(ratio)]
        q = np.percentile(ratio, [5, 50, 95])
        iso = np.percentile(self.isoperimetric[m], [50, 95])
        return (
            f"{int(m.sum())} cross-sections measured; "
            f"r_stored/r_perimeter p5 {q[0]:.2f}  median {q[1]:.2f}  p95 {q[2]:.2f}; "
            f"isoperimetric median {iso[0]:.2f}  p95 {iso[1]:.2f}"
        )


@dataclass
class PlaneCut:
    """One perpendicular cut through the lumen, and the two labellings of it.

    Both labellings are kept because they answer different questions -- see
    :func:`cut`. Callers that only want a shape measure want `blob4`; callers
    asking whether a *separate* vessel is present want `blob8`.
    """

    plane: np.ndarray  # (2h+1, 2h+1) uint8, the sampled labels
    blob4: np.ndarray  # bool, the 4-connected component holding the centre
    blob8: np.ndarray  # bool, the 8-connected component holding the centre
    n_components8: int  # how many 8-connected components the plane holds
    u: np.ndarray  # (3,) unit vector, first in-plane axis, in um
    v: np.ndarray  # (3,) unit vector, second in-plane axis, in um
    half: int  # the half-width actually used, in segmentation voxels
    touches_border: bool  # blob8 reaches the window edge, so it is truncated
    grew: bool = False  # the window had to be doubled at least once to hold the blob

    @property
    def pinched(self) -> bool:
        """The 4-connectivity labelling split this lumen where 8- did not."""
        return bool(self.blob8.sum() > self.blob4.sum())

    @property
    def trustworthy(self) -> bool:
        """The section closes inside its own window, so its centroid means something.

        A cut that still reaches the border is not a cross-section, it is a slab through
        whatever the plane happened to graze -- most often the vessel *along* its axis,
        when the tangent is wrong. Its area centroid can be arbitrarily far from the
        lumen. Measures of *size* can still be salvaged from a truncated blob, which is
        why :func:`measure` uses one; measures of *position* cannot, which is why
        :mod:`~.edit.skeleton_optimise` checks this.
        """
        return not self.touches_border


@dataclass
class StablePlaneCut:
    """A transverse cut validated against a short three-plane slab.

    ``cut`` is the section at the requested centre.  The companion slab cuts are
    intentionally not retained: callers need the selected normal and the stability
    diagnostics, not three copies of every sampled image.
    """

    cut: PlaneCut
    tangent: np.ndarray
    area_ratio: float
    perimeter_ratio: float
    centroid_ratio: float
    searched: bool = False
    perimeter_um: float = float("nan")  # of `cut.blob4`; the search's objective
    axis_ratio: float = float("nan")  # major/minor semi-axis of `cut.blob4`
    #: ``perimeter on the fitted tangent / perimeter on the chosen one``, or NaN when
    #: the fitted tangent produced no stable section to compare against. Above 1 the
    #: fitted tangent was oblique and would have over-read the radius by this factor;
    #: see :data:`TRANSVERSE_AXIS_RATIO`.
    obliquity: float = float("nan")
    centroid_offset_ratio: float = float("nan")


#: How elliptical a section may be before the bounded tangent search is worth running.
#:
#: The three-plane stability test cannot see tilt. Its slabs are taken *along the
#: candidate normal*, so on a straight vessel of constant calibre an oblique plane
#: yields three identical ellipses: ``area_ratio`` and ``perimeter_ratio`` both come
#: back at exactly 1.000 and the cut is pronounced stable. Measured on an analytic
#: cylinder, a normal 40 degrees off the vessel axis scores a perfect 1.000/1.000 and
#: reports a radius 26% too large -- a smooth, plausible bias that sits below every
#: downstream guard (`RUNAWAY_FACTOR`, `LOCAL_RADIUS_FACTOR`) because it is neither a
#: spike nor an outlier.
#:
#: What tilt does change is the *shape*: an oblique cut of a tube of radius ``r`` at
#: ``theta`` off its axis is an ellipse of semi-axes ``r`` and ``r / cos theta``. So an
#: elongated section is the signal that rotating the plane might still help, and a
#: round one is proof that it cannot. Gating the search on it keeps the common case at
#: its old cost -- one candidate, three plane samples -- and spends the nine-candidate
#: search only where there is something to find.
#:
#: Minimising the perimeter over the cone is the right objective in *both* regimes,
#: which is why the same gate serves a collapsed lumen as well as a round one: tilting
#: away from transverse stretches whichever axis the tilt lies in and can only lengthen
#: the boundary. It is also what ``adjust_thickness.py`` did originally, by reslicing
#: for minimum area.
#:
#: 1.10 is roughly a 5% over-read, and 25 degrees of tilt.
TRANSVERSE_AXIS_RATIO = 1.10


def _blob_axis_ratio(blob: np.ndarray) -> float:
    """``major / minor`` semi-axis of a blob, from its second moments. 1.0 is round."""
    uu, vv = np.nonzero(blob)
    if len(uu) < 3:
        return 1.0
    pts = np.column_stack([uu - uu.mean(), vv - vv.mean()]).astype(np.float64)
    w = np.linalg.eigvalsh(pts.T @ pts / len(pts))
    lo, hi = float(max(w[0], 0.0)), float(max(w[1], 0.0))
    return float(np.sqrt(hi / lo)) if lo > 1e-12 else float("inf")


#: The two connectivities :func:`cut` labels with, built once. Passing them saves
#: `ndimage.label` deriving the same two structures on every call, which at one call
#: per plane sample is not nothing.
_STRUCT4 = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=int)
_STRUCT8 = np.ones((3, 3), dtype=int)


def _plane_axes(tangent: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    """Two orthonormal in-plane axes for the plane perpendicular to `tangent`."""
    nt = np.linalg.norm(tangent)
    if nt < 1e-9:
        return None
    t = tangent / nt
    u = np.cross(t, [0.0, 0.0, 1.0])
    if np.linalg.norm(u) < 1e-6:
        u = np.cross(t, [0.0, 1.0, 0.0])
    u = u / np.linalg.norm(u)
    return u, np.cross(t, u)


def cut(sampler, centre_ijk, tangent, half: int, *, max_half: int = 64,
        grow_to: int | None = None,
        min_blob_voxels: int = MIN_BLOB_VOXELS) -> PlaneCut | None:
    """Cut the lumen perpendicular to `tangent` at `centre_ijk`; None if unmeasurable.

    Factored out of :func:`measure` so that re-centring
    (:mod:`~.edit.skeleton_optimise`) and radius measurement
    (:mod:`~.edit.radius_perimeter`) share one cut rather than growing a third
    copy of "sample a plane, label it, pick the component holding the centre".

    **The window grows when the blob reaches its edge.** `half` is sized from the
    radius the graph already carries, which for an Avizo graph is exactly the value
    being distrusted -- if it is too small the section is cropped and every measure
    taken from it is wrong, quietly. Doubling until the blob stops touching the
    border costs one extra cut in the rare case and removes a whole class of silent
    under-measurement. `max_half` bounds it, and `touches_border` reports the cases
    where even that was not enough.

    **`grow_to` bounds the growth separately, and it matters.** Growth is only a good
    idea while the thing being chased is *this* lumen. When the plane is not actually
    perpendicular -- the usual case with a tangent estimated across a voxel staircase --
    the section is a streak *along* the vessel that touches the border at any width, so
    doubling escalates all the way to `max_half`. At stride 1 that is a 4.2 mm window,
    26x the median vessel radius, and its centroid is meaningless. Callers who need a
    *position* out of the cut pass a `grow_to` of a few expected radii and treat
    `touches_border` as a refusal; callers who need a *size* leave it at `max_half`,
    because a large vessel legitimately needs a large window.
    """
    axes = _plane_axes(np.asarray(tangent, dtype=float))
    if axes is None:
        return None
    u, v = axes
    half = max(int(half), 2)
    ceiling = max(half, int(max_half if grow_to is None else min(grow_to, max_half)))
    grew = False

    while True:
        plane = sampler.plane(centre_ijk, u, v, half)
        if not plane[half, half]:
            return None

        # Two labellings, deliberately. 4-connectivity reproduces what the pipeline's
        # `clean_and_measure_slice` actually measured, so the perimeter is comparable to
        # the radius it assigned. 8-connectivity is the honest notion of one lumen, and
        # is what decides whether a *separate* vessel is present: without it every
        # diagonally-pinched lumen reports a spurious companion one voxel away.
        #
        # 8 first, and 4 only once the window has settled: whether to grow is decided
        # on the 8-connected blob alone, and over half of all plane samples here are
        # growth iterations whose 4-connected labelling was thrown away unused.
        lab8, n8 = ndimage.label(plane, structure=_STRUCT8)
        blob8 = lab8 == lab8[half, half]

        touches = bool(
            blob8[0].any() or blob8[-1].any() or blob8[:, 0].any() or blob8[:, -1].any()
        )
        if touches and half < ceiling:
            half = min(half * 2, ceiling)
            grew = True
            continue

        lab4, _ = ndimage.label(plane, structure=_STRUCT4)
        blob4 = lab4 == lab4[half, half]
        if blob4.sum() < min_blob_voxels:
            return None
        return PlaneCut(plane, blob4, blob8, int(n8), u, v, half, touches, grew)


class _PlaneSampler:
    """Samples the segmentation on arbitrary planes, caching decoded row bands.

    `labels` is either the RLE lattice -- anything with ``slice_z(k)``, which is
    what the viewer passes and what keeps the full 2.34 GB volume off the heap --
    or an already-decoded ``(nz, ny, nx)`` array, which is what a test fixture and
    any caller that has already paid for :func:`~.edit.lattice.decode_volume` has.
    Indexing an array needs no cache, so it bypasses one.

    **The unit of decoding is a row band, not a slice**, wherever the lattice offers
    ``slice_rows``. A cross-section plane is a few tens of voxels across and crosses
    a few tens of slices, so decoding a whole 3400x2964 plane for it -- 10 MB, 5.6 ms
    -- throws away 98% of the work. Profiled on `radius-perimeter` over the left tree,
    that one decode was 86% of the run. Bands make it 43x cheaper and are cached by
    bytes rather than by count, which also bounds the memory the old whole-slice cache
    did not: 160 slices of that mask is 1.6 GB.

    Eviction is LRU rather than FIFO because of how the caller asks: a point tries up
    to nine candidate tangents over three slab offsets, all through the same few bands,
    and FIFO evicts the band that is about to be wanted again.
    """

    def __init__(self, labels, frame, cache_bytes: int = 256 << 20,
                 band_rows: int = 64):
        self.labels = labels
        self._array = labels if isinstance(labels, np.ndarray) else None
        self.nx, self.ny, self.nz = (int(v) for v in frame.seg_dims)
        self._band_rows = max(int(band_rows), 1)
        self._n_bands = max((self.ny + self._band_rows - 1) // self._band_rows, 1)
        self._read_rows = (
            None if self._array is not None else getattr(labels, "slice_rows", None)
        )
        self._cache: dict[tuple[int, int], np.ndarray] = {}
        self._limit = max(int(cache_bytes) // max(self._band_rows * self.nx, 1), 8)

    def _rows(self, k: int, row0: int, row1: int) -> np.ndarray:
        """Rows ``[row0, row1)`` of slice ``k``, uncached, by whatever route exists."""
        if self._array is not None:
            return self._array[k, row0:row1]
        if self._read_rows is not None:
            return np.asarray(self._read_rows(k, row0, row1))
        return np.asarray(self.labels.slice_z(k))[row0:row1]

    def _band(self, k: int, band: int) -> np.ndarray:
        """One cached row band of one slice."""
        row0 = band * self._band_rows
        if self._array is not None:
            return self._array[k, row0:row0 + self._band_rows]
        key = (k, band)
        hit = self._cache.get(key)
        if hit is not None:
            self._cache[key] = self._cache.pop(key)  # most recently used, at the end
            return hit
        hit = self._rows(k, row0, min(row0 + self._band_rows, self.ny))
        self._cache[key] = hit
        if len(self._cache) > self._limit:
            self._cache.pop(next(iter(self._cache)))
        return hit

    def _slice(self, k: int) -> np.ndarray:
        """A whole plane. Kept for callers that genuinely want one; bands are cheaper."""
        return self._rows(k, 0, self.ny)

    def _gather(self, i, j, k, shape):
        """Labels at integer ``(i, j, k)``, grouped so each row band is read once.

        Split out of :meth:`at` because the callers that dominate -- :meth:`plane`
        and everything through it -- can produce the three index arrays directly and
        should not have to build, and then re-slice, an ``(..., 3)`` float array to
        get here.

        Two fast paths, both of which are the common case rather than a flourish: a
        plane that lies wholly inside the lattice needs no compaction, and one that
        lies inside a single band of a single slice needs no grouping at all.
        """
        out = np.zeros(i.size, dtype=np.uint8)
        inside = (
            (i >= 0) & (i < self.nx) & (j >= 0) & (j < self.ny) & (k >= 0) & (k < self.nz)
        )
        if inside.all():
            where, ii, jj, kk = None, i, j, k
        else:
            where = np.flatnonzero(inside)
            if where.size == 0:
                return out.reshape(shape)
            ii, jj, kk = i[where], j[where], k[where]

        rows = self._band_rows
        keys = kk * self._n_bands + jj // rows
        low, high = int(keys.min()), int(keys.max())
        if low == high:
            band = low % self._n_bands
            values = self._band(low // self._n_bands, band)[jj - band * rows, ii]
            if where is None:
                out[:] = values
            else:
                out[where] = values
            return out.reshape(shape)

        # Sorting rather than one full-length mask per distinct slice: a grown
        # 257x257 plane crosses hundreds of them, and the mask form is quadratic in
        # exactly the case that already hurts.
        order = np.argsort(keys, kind="stable")
        sorted_keys = keys[order]
        starts = np.flatnonzero(
            np.concatenate(([True], sorted_keys[1:] != sorted_keys[:-1]))
        )
        for start, stop in zip(starts, np.concatenate((starts[1:], [order.size]))):
            sel = order[start:stop]
            key = int(sorted_keys[start])
            band = key % self._n_bands
            values = self._band(key // self._n_bands, band)[jj[sel] - band * rows, ii[sel]]
            out[sel if where is None else where[sel]] = values
        return out.reshape(shape)

    def at(self, points_ijk) -> np.ndarray:
        """Nearest-neighbour labels at fractional ``(i, j, k)`` positions.

        Any shape ending in 3; the result has the leading shape. Outside the lattice
        is 0. Kept as the general entry point -- :mod:`~.reformat` samples a grid of
        its own choosing through it rather than growing a fourth copy of the gather.
        """
        pts = np.asarray(points_ijk, dtype=np.float64)
        shape = pts.shape[:-1]
        flat = pts.reshape(-1, 3)
        return self._gather(
            np.rint(flat[:, 0]).astype(np.int64),
            np.rint(flat[:, 1]).astype(np.int64),
            np.rint(flat[:, 2]).astype(np.int64),
            shape,
        )

    def plane(self, centre_ijk, u, v, half: int) -> np.ndarray:
        """(2*half+1, 2*half+1) binary plane spanned by unit vectors u, v.

        The indices are built one axis at a time by broadcasting, rather than by
        meshgrid into an ``(n, n, 3)`` array: at half=128 that array is 1.6 MB of
        float64 whose three columns are then read back strided, and this is the
        single hottest allocation in `radius-perimeter`.
        """
        centre = np.asarray(centre_ijk, dtype=np.float64)
        ax = np.arange(-half, half + 1, dtype=np.float64)
        col = ax[:, None]
        row = ax[None, :]
        shape = (ax.size, ax.size)
        idx = [
            np.rint(centre[d] + col * float(u[d]) + row * float(v[d]))
            .astype(np.int64)
            .reshape(-1)
            for d in (0, 1, 2)
        ]
        return self._gather(idx[0], idx[1], idx[2], shape)

    def box(self, lo_ijk, hi_ijk) -> tuple[np.ndarray, np.ndarray]:
        """Return a clipped ``(z, y, x)`` binary ROI and its global ijk origin."""
        lo = np.maximum(np.floor(lo_ijk).astype(int), 0)
        hi = np.minimum(
            np.ceil(hi_ijk).astype(int), np.array([self.nx, self.ny, self.nz])
        )
        if np.any(hi <= lo):
            return np.zeros((0, 0, 0), dtype=np.uint8), lo
        out = np.empty((hi[2] - lo[2], hi[1] - lo[1], hi[0] - lo[0]), dtype=np.uint8)
        for n, kz in enumerate(range(lo[2], hi[2])):
            out[n] = self._rows(kz, lo[1], hi[1])[:, lo[0] : hi[0]] > 0
        return out, lo


def robust_edge_tangents(
    coords: np.ndarray,
    radii: np.ndarray,
    *,
    spacing_um: float,
    window_radii: float = 4.0,
) -> np.ndarray:
    """Radius-scaled edge-local quadratic derivatives without moving the edge.

    Fitting coordinate against arclength suppresses voxel staircase turns while a
    quadratic term still follows genuine curvature.  Endpoints naturally receive a
    one-sided fit because the window is clipped to the current edge.
    """
    xyz = np.asarray(coords, dtype=np.float64).reshape(-1, 3)
    n = len(xyz)
    if n == 0:
        return np.zeros((0, 3), dtype=np.float64)
    if n == 1:
        return np.array([[0.0, 0.0, 1.0]])
    ds = np.linalg.norm(np.diff(xyz, axis=0), axis=1)
    arc = np.concatenate([[0.0], np.cumsum(ds)])
    positive_ds = ds[ds > 1e-9]
    step = float(np.median(positive_ds)) if len(positive_ds) else float(spacing_um)
    rr = np.asarray(radii, dtype=np.float64).reshape(-1)
    out = np.zeros_like(xyz)
    for i in range(n):
        r = rr[i] if i < len(rr) and np.isfinite(rr[i]) and rr[i] > 0 else spacing_um
        width = max(float(window_radii) * float(r), 6.0 * step)
        ids = np.flatnonzero(np.abs(arc - arc[i]) <= width)
        if len(ids) < 3:
            order = np.argsort(np.abs(arc - arc[i]))[: min(5, n)]
            ids = np.sort(order)
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
    # Avoid arbitrary sign flips from independent polynomial fits.
    for i in range(1, n):
        if float(np.dot(out[i - 1], out[i])) < 0:
            out[i] *= -1.0
    return out


def _tangent_candidates(tangent: np.ndarray, search_degrees: float) -> list[np.ndarray]:
    """Primary tangent followed by an eight-direction bounded cone search."""
    t = np.asarray(tangent, dtype=np.float64)
    nt = float(np.linalg.norm(t))
    if nt < 1e-9:
        return []
    t /= nt
    axes = _plane_axes(t)
    if axes is None or search_degrees <= 0:
        return [t]
    u, v = axes
    angle = np.deg2rad(float(search_degrees))
    candidates = [t]
    for au, av in ((1, 0), (-1, 0), (0, 1), (0, -1),
                   (1, 1), (1, -1), (-1, 1), (-1, -1)):
        offset = au * u + av * v
        offset /= np.linalg.norm(offset)
        cand = np.cos(angle) * t + np.sin(angle) * offset
        candidates.append(cand / np.linalg.norm(cand))
    return candidates


def _cut_metrics(c: PlaneCut, spacing_um: float) -> tuple[float, float, float]:
    area = float(c.blob4.sum()) * spacing_um * spacing_um
    perimeter = _perimeter_um(c.blob4, spacing_um)
    yy, xx = np.nonzero(c.blob4)
    if not len(xx):
        return area, perimeter, np.inf
    centroid = float(np.hypot(xx.mean() - c.half, yy.mean() - c.half)) * spacing_um
    return area, perimeter, centroid


def stable_transverse_cut(
    sampler: _PlaneSampler,
    centre_ijk,
    tangent,
    radius_vox: float,
    *,
    spacing_um: float,
    max_half: int = 64,
    search_degrees: float = 20.0,
    max_variation: float = 1.5,
    max_centroid_radii: float = 0.5,
    min_blob_voxels: int = MIN_BLOB_VOXELS,
    slab_offsets=(-0.5, 0.0, 0.5),
    initial_half: int | None = None,
    transverse_axis_ratio: float = TRANSVERSE_AXIS_RATIO,
    grow_radii: float | None = None,
    centroid_mode: str = "offset",
) -> StablePlaneCut | None:
    """Choose a closed, stable transverse section from a bounded tangent search.

    The fitted tangent is evaluated first. The eight alternates are sampled when it
    fails -- and, since the stability test is blind to tilt, also when it succeeds
    with a section elongated enough that a rotation could still shorten the boundary.
    A round section short-circuits the search as before, so a clean cut stays at the
    cost of the original pass; `transverse_axis_ratio` is where that line is drawn and
    :data:`TRANSVERSE_AXIS_RATIO` is why it has to exist at all.

    The winner is the stable candidate with the shortest boundary, ties going to the
    fitted tangent. The perimeter is what the radius is taken from, so minimising it
    minimises the tilt the radius is inflated by.

    ``centroid_mode="offset"`` checks distance from the centreline, for callers
    validating centring. ``"drift"`` checks transverse centroid movement across
    the slab divided by its axial span (with a one-voxel floor). Radius measurement
    uses drift: a fixed off-centre path must not invalidate a closed lumen perimeter.
    ``centroid_ratio`` records the selected check; ``centroid_offset_ratio`` always
    records the original offset relative to the input radius.

    `grow_radii` bounds how far each sample's window may double, in multiples of
    this point's own radius, and it is what makes a large `max_half` affordable.
    :func:`cut` grows while the blob touches the border, which is right when the
    vessel is genuinely bigger than the window and catastrophic when the plane is
    not transverse: the section is then a streak *along* the vessel that touches at
    any width, so it escalates to `max_half` and is refused anyway, having paid for
    every doubling on the way. Raising `max_half` from 64 to 256 to reach the few
    proximal vessels that need it therefore made a 1,729-point subgraph 35x slower,
    almost all of it spent on small vessels running away to a 513x513 window before
    being rejected. A ceiling of a few radii stops that without touching the case it
    is raised for: a vessel that genuinely needs a wide window has a wide radius.

    `initial_half` overrides the window the search starts from, which is otherwise
    ``2.5 * radius_vox + 2`` -- the stored radius again, and so wrong in the same
    places the stored radius is. It only moves the *starting* size: `cut` still
    doubles up to `max_half` while the blob touches the border, so a larger value
    buys a section that is closed on the first sample rather than one that could
    not be had at all. :mod:`~.edit.section_frames` exposes it so "the window was
    too small" can be tested by making it bigger, rather than argued about.
    """
    if centroid_mode not in ("offset", "drift"):
        raise ValueError("centroid_mode must be 'offset' or 'drift'")
    centre = np.asarray(centre_ijk, dtype=np.float64)
    radius_vox = max(float(radius_vox), 1.0)
    initial_half = (
        min(int(radius_vox * 2.5) + 2, int(max_half))
        if initial_half is None
        else min(max(int(initial_half), 2), int(max_half))
    )
    candidates = _tangent_candidates(np.asarray(tangent, dtype=float), search_degrees)
    stable: list[StablePlaneCut] = []
    offsets = tuple(float(x) for x in slab_offsets)
    if len(offsets) != 3 or not any(abs(x) < 1e-12 for x in offsets):
        raise ValueError("slab_offsets must contain three values including zero")
    centre_slot = next(i for i, x in enumerate(offsets) if abs(x) < 1e-12)
    best_perimeter = np.inf
    for ci, cand in enumerate(candidates):
        cuts: list[PlaneCut | None] = [None, None, None]
        failed = False
        # The requested centre first, and the two slab companions only if it is still
        # in the running. The objective is the centre section's perimeter; the slab
        # exists to validate the winner, so an alternate that already reads longer
        # than the incumbent cannot be chosen however stable it turns out to be, and
        # its two extra plane samples would be spent to learn nothing. That is two
        # thirds of the search's cost on every alternate that does not win.
        for slot in (centre_slot, *(i for i in range(3) if i != centre_slot)):
            shifted = centre + offsets[slot] * radius_vox * cand
            c = cut(
                sampler, shifted, cand, initial_half,
                max_half=max_half, min_blob_voxels=min_blob_voxels,
                grow_to=(None if grow_radii is None
                         else int(float(grow_radii) * radius_vox) + 2),
            )
            if c is None or c.touches_border:
                failed = True
                break
            cuts[slot] = c
            if slot == centre_slot and _perimeter_um(c.blob4, spacing_um) >= best_perimeter:
                failed = True
                break
        if failed:
            continue
        metrics = [_cut_metrics(c, spacing_um) for c in cuts]
        areas = np.array([m[0] for m in metrics])
        perimeters = np.array([m[1] for m in metrics])
        centroids = np.array([m[2] for m in metrics])
        if np.any(areas <= 0) or np.any(perimeters <= 0):
            continue
        area_ratio = float(areas.max() / areas.min())
        perimeter_ratio = float(perimeters.max() / perimeters.min())
        centroid_offset_ratio = float(centroids.max() / (radius_vox * spacing_um))
        centroid_ratio = centroid_offset_ratio
        if centroid_mode == "drift":
            # A fixed offset from the centreline does not change a closed section's
            # perimeter. Test transverse movement across the slab instead. An
            # oblique plane drifts across a straight cylinder as it advances; a
            # transverse plane through an off-centre path retains the same offset.
            centres = np.array([
                np.argwhere(c.blob4).mean(axis=0) - c.half for c in cuts
            ]) * spacing_um
            motion = np.linalg.norm(centres[:, None] - centres[None, :], axis=2).max()
            span = max(float(np.ptp(offsets)) * radius_vox * spacing_um, spacing_um)
            centroid_ratio = float(motion / span)
        if (
            area_ratio <= max_variation
            and perimeter_ratio <= max_variation
            and centroid_ratio <= max_centroid_radii
        ):
            chosen_cut = cuts[centre_slot]
            axis_ratio = _blob_axis_ratio(chosen_cut.blob4)
            stable.append(
                StablePlaneCut(
                    chosen_cut, cand.copy(), area_ratio, perimeter_ratio,
                    centroid_ratio, searched=ci > 0,
                    perimeter_um=float(perimeters[centre_slot]),
                    axis_ratio=axis_ratio,
                    centroid_offset_ratio=centroid_offset_ratio,
                )
            )
            # The robust fitted tangent is the preferred answer when it is stable
            # *and* round: a round section is already the shortest boundary through
            # this point, so the eight alternates cannot improve on it. An elongated
            # one may be a collapsed lumen, which the search will leave alone, or an
            # oblique cut, which it will straighten -- and the three metrics just
            # computed cannot tell those apart. See :data:`TRANSVERSE_AXIS_RATIO`.
            best_perimeter = min(best_perimeter, float(perimeters[centre_slot]))
            if ci == 0 and axis_ratio <= float(transverse_axis_ratio):
                break
    if not stable:
        return None
    best = min(stable, key=lambda item: item.perimeter_um)
    fitted = next((item for item in stable if not item.searched), None)
    if fitted is not None and best.perimeter_um > 0:
        best.obliquity = float(fitted.perimeter_um / best.perimeter_um)
    return best


def sample_label_plane(
    volume_zyx: np.ndarray,
    origin_ijk: np.ndarray,
    centre_ijk,
    u: np.ndarray,
    v: np.ndarray,
    half: int,
) -> np.ndarray:
    """Nearest-neighbour sample of an integer ROI on an arbitrary global plane."""
    ax = np.arange(-half, half + 1)
    uu, vv = np.meshgrid(ax, ax, indexing="ij")
    pts = (
        np.asarray(centre_ijk)[None, None, :]
        + uu[..., None] * u[None, None, :]
        + vv[..., None] * v[None, None, :]
        - np.asarray(origin_ijk)[None, None, :]
    )
    i = np.rint(pts[..., 0]).astype(int)
    j = np.rint(pts[..., 1]).astype(int)
    k = np.rint(pts[..., 2]).astype(int)
    nz, ny, nx = volume_zyx.shape
    ok = (i >= 0) & (i < nx) & (j >= 0) & (j < ny) & (k >= 0) & (k < nz)
    out = np.zeros(i.shape, dtype=volume_zyx.dtype)
    out[ok] = volume_zyx[k[ok], j[ok], i[ok]]
    return out


def measure(
    graph,
    frame,
    labels,
    min_radius_um: float = MIN_RADIUS_UM,
    max_half: int = 64,
    stride: int = 1,
) -> CrossSectionProfile:
    """Measure every centreline cross-section large enough to support a shape fit."""
    pts = graph.points
    rad = graph.thickness
    n = len(pts)
    nan = lambda: np.full(n, np.nan)  # noqa: E731
    prof = CrossSectionProfile(
        nan(), nan(), nan(), nan(), nan(), nan(),
        np.full(n, np.inf), np.zeros(n, bool), np.zeros(n, bool),
    )

    tan = _edge_tangents(graph)
    ijk = frame.um_to_seg(pts)
    sp = float(frame.seg_spacing[0])
    nx, ny, nz = (int(v) for v in frame.seg_dims)
    sampler = _PlaneSampler(labels, frame)

    ok = (
        (rad >= min_radius_um)
        & (ijk[:, 0] >= 0) & (ijk[:, 0] < nx)
        & (ijk[:, 1] >= 0) & (ijk[:, 1] < ny)
        & (ijk[:, 2] >= 0) & (ijk[:, 2] < nz)
    )
    idx = np.flatnonzero(ok)[:: max(1, stride)]
    if idx.size == 0:
        return prof
    # Sorting by z keeps the slice cache warm: neighbouring planes share slices.
    idx = idx[np.argsort(np.rint(ijk[idx, 2]).astype(int), kind="stable")]

    for p in idx:
        # Floored at one voxel so that `min_radius_um=0` -- which the radius pass
        # passes, to measure everywhere -- cannot divide the shape ratios by zero.
        rp = max(rad[p] / sp, 1.0)
        c = cut(sampler, ijk[p], tan[p], min(int(rp * 2.5) + 2, max_half),
                max_half=max_half)
        if c is None:
            continue
        plane, blob, centre8, half = c.plane, c.blob4, c.blob8, c.half
        n8 = c.n_components8
        prof.pinched[p] = c.pinched

        yy, xx = np.nonzero(blob)
        cov = np.cov(np.stack([yy - yy.mean(), xx - xx.mean()]))
        ev = np.sort(np.linalg.eigvalsh(cov))
        # For a filled ellipse the covariance eigenvalues are (semi-axis)^2 / 4.
        prof.minor_ratio[p] = 2.0 * np.sqrt(max(ev[0], 0.0)) / rp
        prof.major_ratio[p] = 2.0 * np.sqrt(max(ev[1], 0.0)) / rp

        per = _perimeter_um(blob, sp)
        area = float(blob.sum()) * sp * sp
        prof.r_perimeter[p] = per / (2.0 * np.pi)
        prof.r_area[p] = np.sqrt(area / np.pi)
        prof.isoperimetric[p] = (per * per) / (4.0 * np.pi * area) if area > 0 else np.nan

        gy, gx = np.mgrid[0 : plane.shape[0], 0 : plane.shape[1]]
        disc = ((gy - half) ** 2 + (gx - half) ** 2) <= rp * rp
        prof.fill[p] = (blob & disc).sum() / max(disc.sum(), 1)

        # Nearest genuinely separate lumen: the pipeline would have discarded it.
        if n8 > 1:
            # Every labelled voxel outside the centre component. `lab8 > 0` and
            # `plane > 0` mark the same voxels, so the labelling itself need not be
            # carried out of `cut`.
            others = (plane > 0) & ~centre8
            if others.any():
                dist = ndimage.distance_transform_edt(~centre8)
                prof.companions[p] = float(dist[others].min()) * sp
        prof.measured[p] = True

    return prof


def _rolling_median(v: np.ndarray, k: int) -> np.ndarray:
    """Median filter of width k that tolerates NaN, returning NaN only where all-NaN."""
    n = len(v)
    out = np.full(n, np.nan)
    h = k // 2
    for a in range(n):
        w = v[max(0, a - h) : min(n, a + h + 1)]
        w = w[np.isfinite(w)]
        if len(w):
            out[a] = np.median(w)
    return out


def _runs(graph, values, predicate, smooth, min_run_factor, arc, rad):
    """Contiguous stretches along an edge where a smoothed metric satisfies predicate.

    Yields ``(worst_point, edge, run_length_um, median_value, first_point, last_point)``.

    Anchoring the marker on a *measured* point matters: the smoothing spreads
    anomalies onto neighbours that were never measured, and reporting one of those
    would put the marker in the wrong place.

    The run's own bounds are reported alongside it because a collapse is a region,
    not a point -- a repair has to know where it starts and ends, not just where it
    is worst. Both are global point indices, inclusive.
    """
    off = graph.edge_offsets
    for e in range(graph.n_edge):
        a, b = int(off[e]), int(off[e + 1])
        if b - a < 3:
            continue
        raw = values[a:b]
        m = _rolling_median(raw, smooth)
        bad = np.isfinite(m) & predicate(m)
        if not bad.any():
            continue
        edges = np.flatnonzero(np.diff(np.concatenate([[0], bad.view(np.int8), [0]])))
        for s, t in zip(edges[0::2], edges[1::2]):
            local = raw[s:t]
            have = np.flatnonzero(np.isfinite(local))
            if len(have) < 3:
                continue
            span = arc[a + s : a + t][have]
            length = float(span.max() - span.min())
            local_r = float(np.nanmedian(rad[a + s : a + t][have]))
            if length < min_run_factor * local_r:
                continue
            k = a + s + int(have[np.argmax(np.abs(local[have]))])
            yield (k, e, length, float(np.nanmedian(local[have])),
                   int(a + s), int(a + t - 1))


def find_sites(
    graph,
    frame,
    labels,
    profile: CrossSectionProfile | None = None,
    severity_iso: float = 1.6,
    mismatch_factor: float = 1.5,
    companion_max_um: float = 400.0,
    smooth: int = 5,
    min_run_factor: float = 1.0,
    max_per_kind: int = 200,
    start_id: int = 0,
    **measure_kw,
) -> tuple[list[Candidate], CrossSectionProfile]:
    """Emit candidates for sustained runs of each signature, worst first.

    severity_iso
        Isoperimetric ratio above which a lumen counts as strongly collapsed. This is
        **not** an error -- see the module docstring -- it marks where the perimeter
        assumption is carrying the most weight.
    mismatch_factor
        Report where ``r_stored / r_perimeter`` leaves ``[1/f, f]``: the assigned radius
        no longer reflects the perimeter measured at that cross-section.
    companion_max_um
        Largest gap to a discarded neighbouring lumen component still worth reporting.
    """
    if profile is None:
        profile = measure(graph, frame, labels, **measure_kw)

    pts = graph.points
    rad = graph.thickness
    arc = _arclength(graph)
    stra = _strahler(graph)
    out: list[Candidate] = []
    cid = start_id

    # Points Avizo interpolated carry a radius nobody measured, so every signature here
    # is meaningless at them: "the assigned radius disagrees with the perimeter" is not a
    # finding when the assigned radius was made up. Blanked rather than dropped, so the
    # indices still line up with `graph.points` -- `spans_from_candidates` resolves
    # `point_a`/`point_b` through the flat point order.
    from .edit.interpolation import flags_array

    invented = flags_array(graph).astype(bool)

    def emit(kind, k, e, length, score, detail, point_a=-1, point_b=-1):
        nonlocal cid
        p = pts[k]
        zyx = frame.um_to_raw_index(p)[0]
        cid += 1
        out.append(
            Candidate(
                id=cid,
                kind=kind,
                x_um=float(p[0]),
                y_um=float(p[1]),
                z_um=float(p[2]),
                raw_slice=int(zyx[0]),
                raw_row=int(zyx[1]),
                raw_col=int(zyx[2]),
                gap_um=float(profile.companions[k]) if np.isfinite(profile.companions[k]) else 0.0,
                dist_um=0.0,
                radius_um=float(rad[k]),
                partner_radius_um=float(profile.r_perimeter[k])
                if np.isfinite(profile.r_perimeter[k])
                else 0.0,
                edge_a=int(e),
                edge_b=-1,
                hops=float("inf"),
                strahler_a=int(stra[e]),
                strahler_b=-1,
                contact_um=float(length),
                cos_angle=0.0,
                same_component=True,
                score=float(score),
                detail=detail,
                point_a=int(point_a),
                point_b=int(point_b),
            )
        )

    # ------------------------------------------------------- perimeter_mismatch
    ratio = profile.perimeter_ratio(rad)
    with np.errstate(divide="ignore", invalid="ignore"):
        log_ratio = np.log2(ratio)
    log_ratio = np.where(invented, np.nan, log_ratio)
    lim = np.log2(mismatch_factor)
    hits = list(
        _runs(graph, log_ratio, lambda m: np.abs(m) > lim, smooth, min_run_factor, arc, rad)
    )
    hits.sort(key=lambda h: -abs(h[3]))
    for k, e, length, med, pa, pb in hits[:max_per_kind]:
        f = 2.0 ** med
        which = "larger" if f > 1 else "smaller"
        note = (
            "; the pipeline's 4-connectivity split this lumen, so it measured only part of it"
            if profile.pinched[k]
            else ""
        )
        emit(
            "perimeter_mismatch", k, e, length, -abs(med),
            f"edge {e}: assigned radius {rad[k]:.0f} um is {f:.2f}x {which} than the "
            f"perimeter measured here implies ({profile.r_perimeter[k]:.0f} um), "
            f"sustained over {length:.0f} um{note}",
            point_a=pa, point_b=pb,
        )

    # -------------------------------------------------------- collapse_severity
    hits = list(
        _runs(graph, np.where(invented, np.nan, profile.isoperimetric),
              lambda m: m > severity_iso, smooth, min_run_factor, arc, rad)
    )
    hits.sort(key=lambda h: -h[3])
    for k, e, length, med, pa, pb in hits[:max_per_kind]:
        emit(
            "collapse_severity", k, e, length, -med,
            f"edge {e}: lumen is strongly non-circular (isoperimetric {med:.2f}, "
            f"minor/r {profile.minor_ratio[k]:.2f}); the perimeter assumption is doing "
            f"the work here over {length:.0f} um - expected, worth an eyeball",
            point_a=pa, point_b=pb,
        )

    # ---------------------------------------------------------- companion_lumen
    comp = profile.companions.copy()
    comp[~profile.measured] = np.inf
    comp[invented] = np.inf
    hits = list(
        _runs(graph, np.where(np.isfinite(comp), comp, np.nan),
              lambda m: m < companion_max_um, smooth, min_run_factor, arc, rad)
    )
    hits.sort(key=lambda h: h[3])
    for k, e, length, med, pa, pb in hits[:max_per_kind]:
        emit(
            "companion_lumen", k, e, length, med,
            f"edge {e}: a second lumen sits {med:.0f} um away for {length:.0f} um and is "
            f"discarded by the pipeline (only the centre component is measured) - "
            f"possible single collapsed vessel segmented as two",
            point_a=pa, point_b=pb,
        )

    return out, profile


# Backwards-compatible alias: the old name framed re-inflation as a defect.
find_collapse_sites = find_sites
