"""Image evidence for a route, calibrated on the vessel that is asking for it.

Every threshold in :mod:`..dpc` is a number tuned on one scan. HiP-CT greyscale is
not calibrated between datasets and barely calibrated within one -- exposure drifts
down the stack and a distal twig sits at a different level from the trunk that feeds
it -- so a fixed threshold is a per-scan retune waiting to happen.

The alternative used here needs no training and no examples of the thing being
repaired: **the intact vessel on either side of the break is the calibration**. Its
lumen voxels give the lumen distribution, a shell just outside it gives the wall
distribution, and the evidence along the two intact tails sets the scale that the
gap is then measured against. A route is not judged against "what a vessel looks
like"; it is judged against *what these two ends look like*.

Four kinds of evidence, chosen for what a collapsed ex-vivo vessel actually is:

**mask interior distance**   inside foreground, the EDT peaks on the medial axis.
                             Free, exact, and the strongest term wherever the
                             segmentation exists -- which is most of the route when
                             the break is a graph break rather than a mask break.
**distance outside the mask**  past the foreground, cost rises with how far the
                             route strays from any segmentation at all. This is what
                             keeps a path hugging the vessel it is completing rather
                             than cutting a chord through myocardium.
**paired wall gradients**    a deflated tube is two walls pressed together with a
                             dark slit between them, and its two wall gradients point
                             *at each other*. The divergence of the unit gradient
                             field is exactly that quantity, and unlike a vesselness
                             filter it does not care whether the cross-section is
                             round, elliptical or a line.
**multiscale structure**     Hessian eigenvalues at several scales, scored for tube,
                             ribbon and sheet at once. A Frangi filter tuned for
                             tubes scores a collapsed vessel near zero, which is the
                             single most common way an off-the-shelf vesselness term
                             fails on this data.

And one rule that is not evidence at all: **foreground belonging to an unrelated
component is blocked, not cheap.** It is the lowest-cost material in the volume by
every measure above, so a shortest path will dive into a neighbouring vessel and run
along it. That is the false connection this whole package exists to avoid, so those
voxels are removed from the search and recorded as competing targets instead.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

#: Relative weights. Deliberately flat -- the terms are each normalised to the
#: calibration tails before they are combined, so a weight is a statement about
#: how much to trust a kind of evidence, not a scale factor.
WEIGHTS = {"interior": 1.0, "outside": 0.8, "flux": 0.9, "structure": 0.7}
#: Scales for the Hessian, in units of the local vessel radius.
STRUCTURE_SCALES = (0.5, 1.0, 2.0)
#: Support below this fraction of the calibration level counts as unsupported --
#: the path is crossing material the image gives no reason to believe is vessel.
UNSUPPORTED_FRACTION = 0.25
#: Floor on the per-step cost, so a fully supported route still prefers a short
#: path to a long one and the search cannot wander for free.
BASE_COST = 0.05
#: Penalty for travelling through foreground the graph **already describes**.
#:
#: Without it the search has a 21x discount inside any allowed component -- cost
#: `BASE_COST` against ~1.05 in tissue -- and since a component can be the entire
#: coronary tree (2.6 M voxels on LADAF-2024-28), a route may enter the trunk near its
#: source and ride the lumen for millimetres before exiting at whatever goal voxel is
#: convenient. Measured: 8 of 11 accepted routes touched that component, one running
#: 12.4 mm of which 75% was inside existing foreground.
#:
#: The ordering this restores is the one that matters:
#:
#:     undescribed foreground  <<  described foreground  <  tissue
#:
#: Undescribed lumen stays cheap, because that is pruned or never-skeletonised vessel
#: and routing through it is the whole point. Described lumen becomes nearly as
#: expensive as tissue, because the graph already says a vessel runs there and a second
#: centreline along it is a duplicate rather than a repair. It stays *below* tissue so a
#: route still prefers to hug a vessel rather than flee into myocardium.
REDUNDANCY_WEIGHT = 0.6


@dataclass
class Calibration:
    """Lumen and wall statistics measured on the intact ends of one candidate."""

    lumen_mu: float
    lumen_sigma: float
    wall_mu: float
    wall_sigma: float
    n_lumen: int
    n_wall: int
    dark_lumen: bool
    support_reference: float = 1.0
    support_tail_min: float = 0.0

    @property
    def separated(self) -> bool:
        """Is there a measurable contrast between lumen and wall here at all?

        When there is not, the intensity terms carry no information and the
        geometric ones have to do the work alone. Worth knowing explicitly rather
        than discovering it as a route that scores well everywhere.
        """
        spread = np.hypot(self.lumen_sigma, self.wall_sigma)
        return abs(self.lumen_mu - self.wall_mu) > 0.5 * max(spread, 1e-9)

    def lumen_likelihood(self, values: np.ndarray) -> np.ndarray:
        """A soft [0, 1] score for "this intensity belongs to lumen, not wall".

        A logistic on the log-likelihood ratio of two Gaussians. Soft on purpose:
        a hard threshold here reappears as a hard edge in the cost field, and the
        A* then follows that edge instead of the vessel.
        """
        if not self.separated:
            return np.full(np.shape(values), 0.5, dtype=np.float32)
        sl = max(self.lumen_sigma, 1e-6)
        sw = max(self.wall_sigma, 1e-6)
        v = np.asarray(values, dtype=np.float32)
        ratio = (((v - self.wall_mu) / sw) ** 2 - ((v - self.lumen_mu) / sl) ** 2) / 2.0
        return (1.0 / (1.0 + np.exp(-np.clip(ratio, -30.0, 30.0)))).astype(np.float32)

    def describe(self) -> str:
        return (f"lumen {self.lumen_mu:.1f}+-{self.lumen_sigma:.1f} (n={self.n_lumen}), "
                f"wall {self.wall_mu:.1f}+-{self.wall_sigma:.1f} (n={self.n_wall})"
                + ("" if self.separated else "; no usable contrast"))


@dataclass
class CostField:
    """A corridor of the volume, priced for the path search.

    Everything is on the segmentation grid and indexed ``[z, y, x]`` relative to
    ``lo_zyx``. ``cost`` is per-voxel and an edge costs the mean of its two ends
    times the step length, which is the standard way to keep a grid A* consistent
    with the continuous problem it approximates.
    """

    cost: np.ndarray  # (dz, dy, dx) float32, >= BASE_COST
    support: np.ndarray  # (dz, dy, dx) float32, calibrated so ~1 means "as good as the intact vessel"
    blocked: np.ndarray  # (dz, dy, dx) bool -- unrelated components
    mine: np.ndarray  # (dz, dy, dx) bool -- foreground of the allowed components
    described: np.ndarray  # (dz, dy, dx) float32 in [0, 1] -- already carries centreline
    labels: np.ndarray  # (dz, dy, dx) int32 -- component id per voxel, 0 = background
    lo_zyx: np.ndarray
    spacing_zyx: np.ndarray
    calibration: Calibration
    allowed: set = field(default_factory=set)
    competing: dict = field(default_factory=dict)  # label -> voxel count in the corridor

    @property
    def shape(self) -> tuple[int, int, int]:
        return tuple(self.cost.shape)

    def contains(self, zyx) -> bool:
        idx = np.asarray(zyx, dtype=np.int64) - self.lo_zyx
        return bool(np.all(idx >= 0) and np.all(idx < np.asarray(self.cost.shape)))

    def to_local(self, zyx) -> np.ndarray:
        return np.asarray(zyx, dtype=np.int64) - self.lo_zyx

    def to_global(self, local) -> np.ndarray:
        return np.asarray(local, dtype=np.int64) + self.lo_zyx

    def unsupported_mask(self, fraction: float = UNSUPPORTED_FRACTION) -> np.ndarray:
        return self.support < fraction

    def describe(self) -> str:
        blocked = int(self.blocked.sum())
        rivals = ", ".join(f"#{k}({v})" for k, v in
                           sorted(self.competing.items(), key=lambda kv: -kv[1])[:3])
        return (f"corridor {self.shape}, {blocked:,} blocked voxel(s)"
                + (f", competing {rivals}" if rivals else "")
                + f"; {self.calibration.describe()}")


# ------------------------------------------------------------------ calibration


def calibrate(volume, mask_component, *, dark_lumen: bool = True,
              wall_dilation: int = 2) -> Calibration:
    """Lumen and wall intensity statistics from one corridor's intact foreground.

    `mask_component` is the boolean foreground of the components this candidate is
    allowed to use. The wall sample is the shell the mask grows into under a small
    dilation -- the tissue immediately outside the lumen, which is what a route
    leaving the vessel would be crossing.

    Medians and a robust spread rather than mean and standard deviation: the lumen
    sample includes partial-volume voxels at its own boundary and the wall sample
    includes whatever else happens to sit nearby, and both are exactly the kind of
    contamination that moves a mean.
    """
    from scipy import ndimage

    values = np.asarray(volume, dtype=np.float32)
    inside = np.asarray(mask_component, dtype=bool)
    if not inside.any():
        v = values.ravel()
        lo, hi = np.percentile(v, [25.0, 75.0]) if v.size else (0.0, 1.0)
        return Calibration(float(lo), float(hi - lo) or 1.0, float(hi),
                           float(hi - lo) or 1.0, 0, 0, dark_lumen)

    grown = ndimage.binary_dilation(
        inside, ndimage.generate_binary_structure(3, 1), iterations=max(wall_dilation, 1)
    )
    shell = grown & ~inside

    lumen = values[inside]
    wall = values[shell] if shell.any() else values[~inside]
    return Calibration(
        lumen_mu=float(np.median(lumen)), lumen_sigma=_robust_sigma(lumen),
        wall_mu=float(np.median(wall)) if wall.size else float(np.median(lumen)),
        wall_sigma=_robust_sigma(wall) if wall.size else _robust_sigma(lumen),
        n_lumen=int(lumen.size), n_wall=int(wall.size), dark_lumen=dark_lumen,
    )


def _robust_sigma(values: np.ndarray) -> float:
    """Normal-consistent spread from the median absolute deviation."""
    v = np.asarray(values, dtype=np.float64).ravel()
    if v.size < 2:
        return 1.0
    mad = float(np.median(np.abs(v - np.median(v))))
    return max(1.4826 * mad, 1e-6)


# -------------------------------------------------------------------- evidence


def flux_medialness(volume, sigma: float, dark_lumen: bool = True) -> np.ndarray:
    """Where the wall gradients on either side point at each other.

    The divergence of the unit gradient field. The two walls of a collapsed vessel
    have gradients that face each other across the slit between them, and this is
    the sharpest available statement of that; unlike a Hessian ridge measure it
    does not care whether the cross-section is round, elliptical or a line, which
    is the whole reason it is here.

    **The sign follows the contrast, and it is the opposite of the classical
    formulation.** Flux medialness was defined for *bright* objects, where the
    gradient points inward and the field converges on the axis, so the measure is
    ``-div``. HiP-CT coronary lumen is darker than the myocardium around it, so
    the gradient points *outward* from the axis and the field diverges there --
    making ``+div`` the medialness and ``-div`` a measure of the wall. Getting
    this backwards is silent: the term simply reads zero inside every vessel, the
    other three terms carry the field, and nothing reports that one quarter of the
    evidence stopped contributing.
    """
    from scipy import ndimage

    v = np.asarray(volume, dtype=np.float32)
    smooth = ndimage.gaussian_filter(v, sigma)
    gradients = np.gradient(smooth)
    magnitude = np.sqrt(sum(g * g for g in gradients)) + 1e-6
    unit = [g / magnitude for g in gradients]
    divergence = sum(np.gradient(unit[k], axis=k) for k in range(3))
    flux = divergence if dark_lumen else -divergence
    # Weight by how much gradient there was to begin with: a divergence computed
    # from a unit field is defined even in flat noise, and there it means nothing.
    weight = magnitude / (np.percentile(magnitude, 95.0) + 1e-6)
    return (np.maximum(flux, 0.0) * np.clip(weight, 0.0, 1.0)).astype(np.float32)


def structure_response(volume, sigmas, dark_lumen: bool = True) -> np.ndarray:
    """Tube, ribbon and sheet responses at several scales, combined by maximum.

    From the sorted Hessian eigenvalues ``|l1| <= |l2| <= |l3|``. A tube has one
    small and two large same-signed eigenvalues; a sheet has two small and one
    large; a ribbon is between them. Scoring all three and taking the best is what
    lets one filter follow a vessel from round trunk into flattened distal
    collapse without a shape assumption changing under it.
    """
    from scipy import ndimage

    v = np.asarray(volume, dtype=np.float32)
    if dark_lumen:
        v = -v
    best = np.zeros(v.shape, dtype=np.float32)
    for sigma in sigmas:
        if sigma <= 0:
            continue
        smooth = ndimage.gaussian_filter(v, sigma)
        hessian = _hessian_eigenvalues(smooth, sigma)
        l1, l2, l3 = hessian  # by ascending |value|
        # Bright ridge on the inverted image => l2, l3 strongly negative.
        magnitude = np.sqrt(l1 * l1 + l2 * l2 + l3 * l3)
        scale = np.percentile(magnitude, 99.0) + 1e-6
        structure = 1.0 - np.exp(-(magnitude ** 2) / (2.0 * scale ** 2))

        negative = (l3 < 0)
        tube = np.where(negative & (l2 < 0),
                        np.abs(l2) / (np.abs(l3) + 1e-6), 0.0)
        sheet = np.where(negative, 1.0 - np.abs(l2) / (np.abs(l3) + 1e-6), 0.0)
        ribbon = np.where(negative,
                          1.0 - np.abs(np.abs(l2) / (np.abs(l3) + 1e-6) - 0.5) * 2.0,
                          0.0)
        shape = np.maximum(np.maximum(tube, sheet), ribbon)
        best = np.maximum(best, (shape * structure).astype(np.float32))
    return best


def _hessian_eigenvalues(smooth, sigma: float):
    """Hessian eigenvalues, gamma-normalised, sorted by ascending magnitude."""
    axes = [np.gradient(smooth, axis=k) for k in range(3)]
    rows = [[np.gradient(axes[i], axis=j) for j in range(3)] for i in range(3)]
    # Gamma = 2 normalisation, so responses at different sigmas are comparable and
    # the max over scales means "the scale that fits best" rather than "the largest".
    h = np.stack([np.stack(row, axis=-1) for row in rows], axis=-2) * (sigma ** 2)
    h = 0.5 * (h + np.swapaxes(h, -1, -2))  # symmetrise against numerical drift
    values = np.linalg.eigvalsh(h)
    order = np.argsort(np.abs(values), axis=-1)
    values = np.take_along_axis(values, order, axis=-1)
    return values[..., 0], values[..., 1], values[..., 2]


# ------------------------------------------------------------------------ build


def describedness(shape, lo_zyx, centreline_zyx, spacing_zyx, radius_um: float
                  ) -> np.ndarray:
    """[0, 1] per voxel: how thoroughly the graph already describes this lumen.

    1 on an existing centreline, decaying over one local radius. The scale is the
    vessel's own radius rather than a fixed distance because "the graph already covers
    this" means "within the tube that centreline stands for", and that tube is 120 um
    across on a twig and 1.5 mm on the trunk.
    """
    from scipy import ndimage

    out = np.zeros(tuple(shape), dtype=np.float32)
    idx = np.asarray(centreline_zyx, dtype=np.int64).reshape(-1, 3) - np.asarray(lo_zyx)
    keep = np.all((idx >= 0) & (idx < np.asarray(shape)), axis=1)
    if not keep.any():
        return out
    seed = np.zeros(tuple(shape), dtype=bool)
    seed[idx[keep, 0], idx[keep, 1], idx[keep, 2]] = True
    distance = ndimage.distance_transform_edt(~seed, sampling=spacing_zyx)
    scale = max(float(radius_um), float(np.min(spacing_zyx)))
    return np.exp(-distance / scale).astype(np.float32)


def build(roi, index, allowed, *, radius_um: float, calibration_points=None,
          centreline_points=None, redundancy_weight: float = REDUNDANCY_WEIGHT,
          dark_lumen: bool = True, weights=None,
          scales=STRUCTURE_SCALES) -> CostField:
    """Price one corridor for the search.

    `roi` supplies greyscale and geometry on the **segmentation** grid (see
    :func:`corridor_roi`); `index` is a :class:`~.components.ComponentIndex`; and
    `allowed` is the set of component labels the route may travel through -- the
    source's, the target's, and any fragment that has been admitted to the chain.
    Everything else that is foreground becomes a wall.
    """
    from scipy import ndimage

    weights = dict(WEIGHTS if weights is None else weights)
    volume = np.asarray(roi.volume, dtype=np.float32)
    # `dark_lumen` describes the *raw scan*. The mask surrogate a raw-less corridor
    # falls back to is foreground-bright by construction, so it must not inherit
    # that convention -- doing so inverts the flux and structure terms and both
    # then read zero inside every vessel, silently and without complaint.
    dark_lumen = bool(dark_lumen) and bool(getattr(roi, "has_raw", True))
    spacing_zyx = np.asarray(roi.spacing_um, dtype=np.float64)[::-1]
    lo_zyx = np.asarray(roi.lo_zyx, dtype=np.int64)
    allowed = {int(v) for v in allowed}

    labels = index.window(lo_zyx, lo_zyx + np.asarray(volume.shape, dtype=np.int64))
    mine = np.isin(labels, list(allowed)) if allowed else np.zeros(labels.shape, bool)
    blocked = (labels > 0) & ~mine
    competing = {int(k): int(v) for k, v in
                 zip(*np.unique(labels[blocked], return_counts=True))} if blocked.any() \
        else {}

    calibration = calibrate(volume, mine, dark_lumen=dark_lumen)

    # -- the four terms, each on [0, 1] and each meaning "vessel-like here" -----
    interior = ndimage.distance_transform_edt(mine, sampling=spacing_zyx)
    interior = np.clip(interior / max(radius_um, 1e-6), 0.0, 1.0).astype(np.float32)

    outside = ndimage.distance_transform_edt(~mine, sampling=spacing_zyx)
    # Decays over a couple of radii: at three radii from any foreground the route
    # is no longer completing this vessel, it is inventing a new one.
    proximity = np.exp(-outside / max(2.0 * radius_um, 1e-6)).astype(np.float32)

    sigma_vox = float(np.clip(radius_um / max(spacing_zyx.min(), 1e-6), 0.6, 6.0))
    flux = flux_medialness(volume, sigma_vox, dark_lumen=dark_lumen)
    flux = _unit_scale(flux)
    structure = structure_response(
        volume, [s * sigma_vox for s in scales], dark_lumen=dark_lumen
    )
    structure = _unit_scale(structure)

    intensity = calibration.lumen_likelihood(volume)

    support = (
        weights["interior"] * interior
        + weights["outside"] * proximity * intensity
        + weights["flux"] * flux
        + weights["structure"] * structure
    ).astype(np.float32)
    support /= max(sum(weights.values()), 1e-9)

    # -- calibrate the scale against the intact tails --------------------------
    reference, tail_min = _tail_reference(support, roi, calibration_points)
    calibration.support_reference = reference
    calibration.support_tail_min = tail_min
    support = (support / max(reference, 1e-6)).astype(np.float32)

    # -- what the graph already covers ----------------------------------------
    described = np.zeros(volume.shape, dtype=np.float32)
    if centreline_points is not None and len(centreline_points):
        described = describedness(
            volume.shape, lo_zyx,
            np.round(np.asarray(roi.to_index(centreline_points))).astype(np.int64)
            + lo_zyx,
            spacing_zyx, radius_um,
        )

    cost = (BASE_COST + (1.0 - np.clip(support, 0.0, 1.0))).astype(np.float32)
    # Only inside our own foreground: outside it the route is already paying full
    # price, and charging twice there would push it away from the vessel it is meant
    # to be following.
    cost += (redundancy_weight * described * mine).astype(np.float32)
    cost[blocked] = np.inf
    return CostField(
        cost=cost, support=support, blocked=blocked, mine=mine, described=described,
        labels=labels, lo_zyx=lo_zyx, spacing_zyx=spacing_zyx,
        calibration=calibration, allowed=allowed, competing=competing,
    )


def _unit_scale(field: np.ndarray) -> np.ndarray:
    """Scale a response to roughly [0, 1] by its own 99th percentile.

    Not min-max: one bright artefact would then compress the whole field into the
    bottom of the range, and the route would see a flat cost.
    """
    hi = float(np.percentile(field, 99.0))
    if hi <= 1e-9:
        return np.zeros_like(field, dtype=np.float32)
    return np.clip(field / hi, 0.0, 1.0).astype(np.float32)


def _tail_reference(support, roi, calibration_points) -> tuple[float, float]:
    """The support level the *intact* vessel achieves, as the unit of measurement.

    Without this the cost field is in arbitrary units and every gate downstream is
    a per-scan constant again. With it, "half as well supported as the vessel it
    is completing" means the same thing on a trunk and on a twig, and between two
    scans with different exposure.

    Falls back to the corridor's own upper quartile when no tail is available --
    a weaker calibration, and reported as such rather than silently substituted.
    """
    if calibration_points is None or not len(calibration_points):
        return float(np.percentile(support, 75.0)) or 1.0, 0.0
    idx = np.round(np.asarray(roi.to_index(calibration_points))).astype(np.int64)
    shape = np.asarray(support.shape)
    keep = np.all((idx >= 0) & (idx < shape), axis=1)
    if not keep.any():
        return float(np.percentile(support, 75.0)) or 1.0, 0.0
    values = support[idx[keep, 0], idx[keep, 1], idx[keep, 2]]
    # The median of the tails, not the mean: a few points of the tail may already
    # be in the damaged region, and those are precisely what should not set the bar.
    return float(np.median(values)) or 1.0, float(np.min(values))
