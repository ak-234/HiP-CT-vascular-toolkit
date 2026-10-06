"""The image intensity tensor, and the metric it induces.

:mod:`..geodesic.cost` prices every voxel with one number. That number is good --
it is calibrated on the vessel being repaired and it already scores tube, ribbon
and sheet alike -- but it says nothing about *direction*, and on a collapsed
vessel direction is most of the information. A slit-shaped lumen is easy to cross
and hard to follow: two wall gradients a couple of voxels apart, facing each
other, with nothing in between. A search that pays the same to step through the
wall as to step along the lumen will happily do the former whenever it shortens
the route, and the only thing stopping it in the A* connector is a turn penalty.

This module keeps the scalar and adds the direction. Two tensors, for two
different jobs:

**The structure tensor** ``J = G_rho * (grad I  grad I^T)`` says which way the
intensity does *not* change. Its smallest eigenvector ``nu_1`` is the direction of
least variation -- along the vessel; its largest ``nu_3`` is the direction of most
-- across the two pressed-together walls; ``nu_2`` is across the ribbon's width.
The eigenvalue gaps say how well each of those is determined: on a perfectly flat
sheet ``mu_1 ~ mu_2`` and the in-plane direction is legitimately unknown, which is
reported rather than guessed.

**The Hessian** says whether the local shape is a sheet or a ribbon at all: a
dark slit on a bright background has one strongly negative eigenvalue (on the
inverted image) and two near zero. That planarity is what *switches the
anisotropy on* -- and it does so multiplied by the calibrated lumen likelihood,
because HiP-CT ring artefacts are also planar and must not be given a cheap
direction to run along.

The metric is Riemannian: the cost of a unit step ``e`` at a voxel is
``c * sqrt(e^T A e)`` with ``c`` the scalar cost and

    A = I + width_weight * a * coh * L * (I - nu_1 nu_1^T)
          + normal_weight * P * max(coh, P) * (nu_3 nu_3^T)

so along ``nu_1`` the step costs ``c``, across the width more, and through the
collapse normal most (``a`` is the axis confidence, ``coh`` the coherence, ``L``
the calibrated lumen likelihood, which also multiplies ``P``; the
``max`` lets a sheet the Hessian sees clearly keep its normal even where the
structure tensor is undecided). The ratio between dearest and cheapest direction is capped:
a 26-neighbour lattice only resolves about a 5:1 anisotropy faithfully, and asking
it for more gives a metric the solver cannot honour (:mod:`.propagate`).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

#: Default anisotropy. ``width_weight`` prices a step across the ribbon width,
#: ``normal_weight`` a step through the collapse normal, both relative to a step
#: along the axis and scaled by how confidently the direction was measured.
WIDTH_WEIGHT = 3.0
NORMAL_WEIGHT = 15.0
#: Cap on the eigenvalue ratio of ``A``. See the module docstring.
MAX_RATIO = 5.0
#: Below this structure-tensor coherence the orientation is noise and ``A = I``.
COHERENCE_FLOOR = 0.1
#: Multiscale factors on the derivative scale for the planarity response.
PLANARITY_SCALES = (0.7, 1.0, 1.4)
#: Bounds on the derivative and integration scales, in voxels.
SIGMA_RANGE = (0.7, 4.0)
RHO_RANGE = (1.5, 8.0)

#: Index of the upper-triangular components, the order scikit-image returns them.
_TRI = ((0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2))


@dataclass
class AnisotropyParams:
    width_weight: float = WIDTH_WEIGHT
    normal_weight: float = NORMAL_WEIGHT
    max_ratio: float = MAX_RATIO
    coherence_floor: float = COHERENCE_FLOOR


@dataclass
class TensorField:
    """A priced corridor with a direction at every voxel.

    ``aniso`` holds the six upper-triangular components of ``A`` in scikit-image
    order ``(zz, zy, zx, yy, yx, xx)`` -- everything here is ``(z, y, x)``, the
    corridor's own convention. The scalar field is kept whole rather than copied
    so every gate written against a :class:`~..geodesic.cost.CostField` still works.
    """

    scalar: object  # geodesic.cost.CostField
    aniso: np.ndarray  # (dz, dy, dx, 6) float32
    axis: np.ndarray  # (dz, dy, dx, 3) float32, unit nu_1 (sign arbitrary)
    normal: np.ndarray  # (dz, dy, dx, 3) float32, unit nu_3
    planarity: np.ndarray  # (dz, dy, dx) float32 in [0, 1]
    coherence: np.ndarray  # (dz, dy, dx) float32 in [0, 1]
    axis_confidence: np.ndarray  # (dz, dy, dx) float32 in [0, 1]
    lumen: np.ndarray | None = None  # (dz, dy, dx) float32, calibrated lumen likelihood
    sigma_vox: float = 1.0
    rho_vox: float = 2.0
    params: AnisotropyParams = field(default_factory=AnisotropyParams)

    # -- delegation to the scalar field ---------------------------------------
    @property
    def cost(self) -> np.ndarray:
        return self.scalar.cost

    @cost.setter
    def cost(self, value) -> None:
        self.scalar.cost = value

    @property
    def support(self) -> np.ndarray:
        return self.scalar.support

    @property
    def blocked(self) -> np.ndarray:
        return self.scalar.blocked

    @property
    def mine(self) -> np.ndarray:
        return self.scalar.mine

    @property
    def described(self) -> np.ndarray:
        return self.scalar.described

    @property
    def labels(self) -> np.ndarray:
        return self.scalar.labels

    @property
    def lo_zyx(self) -> np.ndarray:
        return self.scalar.lo_zyx

    @property
    def spacing_zyx(self) -> np.ndarray:
        return self.scalar.spacing_zyx

    @property
    def calibration(self):
        return self.scalar.calibration

    @property
    def shape(self) -> tuple[int, int, int]:
        return self.scalar.shape

    def contains(self, zyx) -> bool:
        return self.scalar.contains(zyx)

    def to_local(self, zyx) -> np.ndarray:
        return self.scalar.to_local(zyx)

    def to_global(self, local) -> np.ndarray:
        return self.scalar.to_global(local)

    def unsupported_mask(self, fraction: float) -> np.ndarray:
        return self.scalar.unsupported_mask(fraction)

    def describe(self) -> str:
        planar = float(np.mean(self.planarity > 0.5)) if self.planarity.size else 0.0
        return (f"{self.scalar.describe()}; tensor sigma {self.sigma_vox:.1f} "
                f"rho {self.rho_vox:.1f} vox, {100 * planar:.0f}% planar")

    def matrices(self) -> np.ndarray:
        """``A`` as a full ``(dz, dy, dx, 3, 3)`` array, for inspection and tests."""
        out = np.empty(self.aniso.shape[:-1] + (3, 3), dtype=np.float32)
        for k, (i, j) in enumerate(_TRI):
            out[..., i, j] = self.aniso[..., k]
            out[..., j, i] = self.aniso[..., k]
        return out

    def step_cost(self, local_zyx, direction_zyx) -> float:
        """``sqrt(e^T A e)`` for a unit direction at one local voxel."""
        z, y, x = (int(v) for v in local_zyx)
        e = np.asarray(direction_zyx, dtype=np.float64)
        e = e / max(np.linalg.norm(e), 1e-12)
        a = self.aniso[z, y, x].astype(np.float64)
        return float(np.sqrt(max(
            a[0] * e[0] * e[0] + a[3] * e[1] * e[1] + a[5] * e[2] * e[2]
            + 2.0 * (a[1] * e[0] * e[1] + a[2] * e[0] * e[2] + a[4] * e[1] * e[2]),
            0.0)))


# ------------------------------------------------------------------ the tensors


def _stack_symmetric(elems) -> np.ndarray:
    """Six upper-triangular arrays -> ``(..., 3, 3)``."""
    out = np.empty(np.shape(elems[0]) + (3, 3), dtype=np.float64)
    for k, (i, j) in enumerate(_TRI):
        out[..., i, j] = elems[k]
        out[..., j, i] = elems[k]
    return out


def orientation(volume, *, sigma_vox: float, rho_vox: float):
    """Structure-tensor orientation: ``(axis, normal, coherence, axis_confidence)``.

    The image is smoothed at the derivative scale first; scikit-image's ``sigma``
    is the *integration* scale (it smooths the products), which is what ``rho``
    means here. ``rho`` should be about the ribbon's half-width: large enough that
    both lateral edges of the slit fall in one window, so the width direction is
    seen as a direction of change and the axis is left as the one that is not.
    """
    from scipy import ndimage
    from skimage.feature import structure_tensor

    v = np.asarray(volume, dtype=np.float32)
    smooth = ndimage.gaussian_filter(v, float(sigma_vox))
    j = _stack_symmetric(structure_tensor(smooth, sigma=float(rho_vox), order="rc"))
    values, vectors = np.linalg.eigh(j)  # ascending
    mu1, mu2, mu3 = values[..., 0], values[..., 1], values[..., 2]
    eps = 1e-6 * (float(np.max(mu3)) if mu3.size else 1.0) + 1e-12
    coherence = ((mu3 - mu1) / (mu3 + mu1 + eps)).astype(np.float32)
    axis_confidence = ((mu2 - mu1) / (mu2 + mu1 + eps)).astype(np.float32)
    axis = vectors[..., :, 0].astype(np.float32)
    normal = vectors[..., :, 2].astype(np.float32)
    return axis, normal, np.clip(coherence, 0, 1), np.clip(axis_confidence, 0, 1)


def planarity(volume, *, sigmas, dark_lumen: bool = True):
    """Hessian planarity ``P`` in [0, 1] and the Hessian's own normal.

    The sheet/ribbon signature only: ``l3`` strongly negative on the (inverted, so
    lumen-bright) image with ``|l2| << |l3|``. A round tube scores low here on
    purpose -- it has no collapse normal, and the metric for it should be
    isotropic in the cross-plane. The scalar term in :mod:`..geodesic.cost`
    already rewards tubes.
    """
    from skimage.feature import hessian_matrix

    v = np.asarray(volume, dtype=np.float32)
    if dark_lumen:
        v = -v
    best = np.zeros(v.shape, dtype=np.float32)
    best_normal = np.zeros(v.shape + (3,), dtype=np.float32)
    best_normal[..., 0] = 1.0
    for sigma in sigmas:
        if sigma <= 0:
            continue
        h = _stack_symmetric(hessian_matrix(v, sigma=float(sigma), order="rc",
                                            use_gaussian_derivatives=True))
        h *= float(sigma) ** 2  # gamma = 2 normalisation, comparable across scales
        values, vectors = np.linalg.eigh(h)
        order = np.argsort(np.abs(values), axis=-1)
        values = np.take_along_axis(values, order, axis=-1)
        l2, l3 = values[..., 1], values[..., 2]
        magnitude = np.sqrt((values ** 2).sum(axis=-1))
        scale = float(np.percentile(magnitude, 99.0)) + 1e-6
        structure = 1.0 - np.exp(-(magnitude ** 2) / (2.0 * scale ** 2))
        sheet = np.where(l3 < 0, 1.0 - np.abs(l2) / (np.abs(l3) + 1e-6), 0.0)
        p = (np.clip(sheet, 0.0, 1.0) * structure).astype(np.float32)
        better = p > best
        best = np.where(better, p, best)
        n = np.take_along_axis(vectors, order[..., None, :], axis=-1)[..., :, 2]
        best_normal = np.where(better[..., None], n.astype(np.float32), best_normal)
    return best, best_normal


# -------------------------------------------------------------------- the metric


def scales_for(profiles, radius_um: float, spacing_um: float) -> tuple[float, float]:
    """Derivative and integration scales, in voxels, from the ends' sections.

    ``sigma`` follows the slit's half-thickness and ``rho`` its half-width; with no
    measured section they both fall back to the radius, which is the round-tube
    assumption the rest of the package makes when it knows nothing better.
    """
    minors = [p.minor_um for p in profiles if p is not None and p.major_axis is not None]
    majors = [p.major_um for p in profiles if p is not None and p.major_axis is not None]
    minor = float(np.mean(minors)) if minors else float(radius_um)
    major = float(np.mean(majors)) if majors else float(radius_um)
    sigma = float(np.clip(minor / max(spacing_um, 1e-9), *SIGMA_RANGE))
    rho = float(np.clip(major / max(spacing_um, 1e-9), *RHO_RANGE))
    return sigma, max(rho, sigma)


def anisotropy_ratio(aniso: np.ndarray) -> np.ndarray:
    """Largest over smallest eigenvalue of ``A`` for ``(..., 6)`` components."""
    a = np.asarray(aniso, dtype=np.float64)
    full = np.empty(a.shape[:-1] + (3, 3))
    for k, (i, j) in enumerate(_TRI):
        full[..., i, j] = a[..., k]
        full[..., j, i] = a[..., k]
    values = np.linalg.eigvalsh(full)
    return values[..., 2] / np.maximum(values[..., 0], 1e-9)


def _outer(vectors: np.ndarray) -> np.ndarray:
    """``(..., 3)`` unit vectors -> ``(..., 3, 3)`` outer products."""
    return vectors[..., :, None] * vectors[..., None, :]


def _cap_ratio(a: np.ndarray, max_ratio: float) -> np.ndarray:
    """Clamp the eigenvalues of each ``A`` to ``[1, max_ratio]``."""
    values, vectors = np.linalg.eigh(a)
    values = np.clip(values, 1.0, float(max_ratio))
    return np.einsum("...ij,...j,...kj->...ik", vectors, values, vectors)


def build(scalar, roi, *, radius_um: float, profiles=(), params=None,
          dark_lumen: bool = True) -> TensorField:
    """Attach a direction to every voxel of a priced corridor.

    `scalar` is the :class:`~..geodesic.cost.CostField` for this corridor and
    `roi` the :class:`~..geodesic.corridor.Corridor` it was built from. `profiles`
    are the :class:`~.prepare.EndProfile` objects of the ends involved, which set the
    scales. Without raw greyscale the corridor's mask surrogate is used: it is
    foreground-bright, so the polarity is flipped just as :mod:`..geodesic.cost`
    flips it.
    """
    p = params or AnisotropyParams()
    volume = np.asarray(roi.volume, dtype=np.float32)
    dark = bool(dark_lumen) and bool(getattr(roi, "has_raw", True))
    spacing = float(np.min(np.asarray(roi.spacing_um, dtype=np.float64)))
    sigma, rho = scales_for(profiles, radius_um, spacing)

    axis, normal_st, coherence, axis_conf = orientation(volume, sigma_vox=sigma,
                                                        rho_vox=rho)
    plan, normal_h = planarity(volume, sigmas=[s * sigma for s in PLANARITY_SCALES],
                               dark_lumen=dark)

    # Orientation and planarity only count where the intensity is lumen-like. A
    # ring artefact is planar and bright and must not be handed a cheap direction;
    # the cut faces either side of a mask gap are coherent and *background*, and
    # must not be handed one either.
    lumen = scalar.calibration.lumen_likelihood(volume).astype(np.float32)
    lumenlike = np.clip(2.0 * lumen, 0.0, 1.0).astype(np.float32)
    plan = (plan * lumenlike).astype(np.float32)

    # Where the structure tensor is incoherent but the Hessian still sees a sheet,
    # the Hessian's normal is the better estimate of the collapse direction.
    weak = coherence < p.coherence_floor
    normal = np.where(weak[..., None], normal_h, normal_st).astype(np.float32)

    eye = np.eye(3, dtype=np.float32)
    width = (p.width_weight * axis_conf * coherence * lumenlike)[..., None, None]
    through = (p.normal_weight * plan * np.maximum(coherence, plan))[..., None, None]
    a = eye + width * (eye - _outer(axis)) + through * _outer(normal)
    a = np.where(weak[..., None, None] & (plan[..., None, None] < 0.2), eye, a)
    a = _cap_ratio(a.astype(np.float64), p.max_ratio).astype(np.float32)

    aniso = np.stack([a[..., i, j] for i, j in _TRI], axis=-1)
    return TensorField(
        scalar=scalar, aniso=np.ascontiguousarray(aniso), axis=axis, normal=normal,
        planarity=plan, coherence=coherence, axis_confidence=axis_conf,
        lumen=lumen, sigma_vox=sigma, rho_vox=rho, params=p,
    )


def isotropic(scalar) -> TensorField:
    """A tensor field with ``A = I`` everywhere: the scalar problem, for tests
    and for the coarse pass."""
    shape = tuple(scalar.cost.shape)
    aniso = np.zeros(shape + (6,), dtype=np.float32)
    aniso[..., 0] = aniso[..., 3] = aniso[..., 5] = 1.0
    axis = np.zeros(shape + (3,), dtype=np.float32)
    axis[..., 2] = 1.0
    normal = np.zeros(shape + (3,), dtype=np.float32)
    normal[..., 0] = 1.0
    zero = np.zeros(shape, dtype=np.float32)
    return TensorField(scalar=scalar, aniso=aniso, axis=axis, normal=normal,
                       planarity=zero, coherence=zero.copy(),
                       axis_confidence=zero.copy(), lumen=np.full(shape, 0.5, np.float32))
