"""Centreline probability: how likely is this voxel to be inside a vessel axis?

The ``P`` term of the DPC walk. The paper trains a Cascade Forest (Deep Forest) on
image patches. Three interchangeable providers are available:

:class:`CfcProbability`
    The persisted, scan-specific DF21 model. It consumes the paper's concatenated
    pooled-15 and raw-7 patches and is the production provider across true gaps.

:class:`FieldProbability`
    No training at all. Combines a vesselness filter on the raw greyscale with the
    Euclidean distance transform of the segmentation, which peaks exactly on the
    centreline. Excellent where the segmentation is present -- and *absent across a
    true gap*, which is precisely where a reconnection has to walk. It carries the
    walk over short breaks on vesselness alone.

:class:`LearnedProbability`
    A gradient-boosted classifier on small raw-image patches, which is the
    practical stand-in for a Cascade Forest. The training data is free: the
    existing skeleton supplies millions of confident positives, and negatives are
    sampled from the vessel wall and from background. This is the one that works
    across a gap, because it never looks at the segmentation.

All expose ``__call__(points_um) -> (N,) in [0, 1]`` so the walk does not care
which it is holding.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _normalise(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    lo, hi = float(np.nanmin(values)), float(np.nanmax(values))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi - lo < 1e-12:
        return np.zeros_like(values)
    return (values - lo) / (hi - lo)


@dataclass
class Roi:
    """A raw-image sub-volume and where it sits in world micrometres.

    ``volume`` is indexed ``[z, y, x]`` (the TIFF stack's own order) and
    ``origin_um`` / ``spacing_um`` are ``(x, y, z)``, matching ``WorldFrame``.
    """

    volume: np.ndarray
    origin_um: np.ndarray
    spacing_um: np.ndarray
    mask: np.ndarray | None = None  # segmentation on the same grid, if available

    def to_index(self, points_um: np.ndarray) -> np.ndarray:
        """World um -> fractional ``(z, y, x)`` indices."""
        p = np.asarray(points_um, dtype=np.float64).reshape(-1, 3)
        ijk = (p - self.origin_um) / self.spacing_um  # (x, y, z) order
        return ijk[:, ::-1]

    def to_world(self, zyx: np.ndarray) -> np.ndarray:
        """``(z, y, x)`` indices -> world um."""
        idx = np.asarray(zyx, dtype=np.float64).reshape(-1, 3)[:, ::-1]
        return self.origin_um + idx * self.spacing_um

    def inside(self, zyx: np.ndarray) -> np.ndarray:
        idx = np.asarray(zyx).reshape(-1, 3)
        shape = np.asarray(self.volume.shape)
        return np.all((idx >= 0) & (idx <= shape - 1), axis=1)


class FieldProbability:
    """Vesselness x distance transform. No training, no labels, no model file.

    Both terms are precomputed over the ROI once, then sampled by trilinear
    interpolation, so a walk of a few hundred steps costs nothing.
    """

    def __init__(self, roi: Roi, *, sigmas=(1.0, 2.0, 4.0), dark_vessels: bool = False):
        from scipy import ndimage
        from skimage.filters import sato

        self.roi = roi
        volume = np.asarray(roi.volume, dtype=np.float32)
        # `sato` responds to bright tubes; invert when the lumen is dark.
        image = -volume if dark_vessels else volume
        vesselness = sato(image, sigmas=sigmas, black_ridges=False)
        field = _normalise(vesselness)

        if roi.mask is not None and roi.mask.any():
            # The EDT of the mask peaks on the medial axis, which is exactly the
            # quantity the walk wants -- where the segmentation exists.
            edt = ndimage.distance_transform_edt(
                np.asarray(roi.mask, dtype=bool), sampling=roi.spacing_um[::-1]
            )
            inside = _normalise(edt)
            # Sum rather than product: a product would be zero everywhere the
            # mask is absent, which is the whole gap the walk has to cross.
            field = 0.5 * field + 0.5 * inside
        self.field = field.astype(np.float32)

    def __call__(self, points_um: np.ndarray) -> np.ndarray:
        return sample_trilinear(self.field, self.roi.to_index(points_um))


class LearnedProbability:
    """Gradient-boosted classifier over raw-image patches.

    ``HistGradientBoostingClassifier`` stands in for the paper's Cascade Forest:
    both are ensembles of shallow trees over a modest feature vector, both train
    in seconds on this much data, and neither needs a GPU.

    Features are a small multi-scale descriptor rather than raw voxels -- mean and
    standard deviation in nested cubes plus the local gradient magnitude. Raw
    patches would overfit the intensity range of whichever scan trained it, and
    HiP-CT greyscale is not calibrated between datasets.
    """

    SCALES = (1, 2, 4)

    def __init__(self, roi: Roi, model=None):
        self.roi = roi
        self.model = model
        self._pyramid = None

    # -------------------------------------------------------------- features

    def _levels(self):
        from scipy import ndimage

        if self._pyramid is None:
            volume = np.asarray(self.roi.volume, dtype=np.float32)
            lo, hi = np.percentile(volume, [1.0, 99.5])
            volume = np.clip((volume - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
            grad = ndimage.gaussian_gradient_magnitude(volume, 1.0)
            levels = [volume, grad]
            for s in self.SCALES:
                levels.append(ndimage.uniform_filter(volume, size=2 * s + 1))
                levels.append(
                    np.sqrt(
                        np.maximum(
                            ndimage.uniform_filter(volume**2, size=2 * s + 1)
                            - ndimage.uniform_filter(volume, size=2 * s + 1) ** 2,
                            0.0,
                        )
                    )
                )
            self._pyramid = [level.astype(np.float32) for level in levels]
        return self._pyramid

    def features(self, zyx: np.ndarray) -> np.ndarray:
        idx = np.asarray(zyx, dtype=np.float64).reshape(-1, 3)
        return np.column_stack([sample_trilinear(level, idx) for level in self._levels()])

    # -------------------------------------------------------------- training

    def fit(
        self,
        centreline_um: np.ndarray,
        *,
        n_negative: int | None = None,
        wall_fraction: float = 0.5,
        radius_um: np.ndarray | float = 100.0,
        seed: int = 0,
    ) -> "LearnedProbability":
        """Train on centreline points as positives, wall and background as negatives.

        Half the negatives are drawn just outside each positive's own radius --
        the vessel *wall*. Without those the classifier only learns "vessel versus
        air", scores the whole lumen highly, and gives the walk no gradient to
        follow toward the axis.
        """
        from sklearn.ensemble import HistGradientBoostingClassifier

        rng = np.random.default_rng(seed)
        pos = self.roi.to_index(centreline_um)
        pos = pos[self.roi.inside(pos)]
        if len(pos) < 20:
            raise ValueError(f"only {len(pos)} centreline points fall inside the ROI")

        n_negative = n_negative or len(pos)
        n_wall = int(n_negative * wall_fraction)
        radii = np.broadcast_to(
            np.asarray(radius_um, dtype=np.float64), (len(pos),)
        )
        # Wall negatives: step off the axis by ~1.4 radii in a random direction.
        pick = rng.integers(0, len(pos), n_wall)
        direction = rng.normal(size=(n_wall, 3))
        direction /= np.maximum(np.linalg.norm(direction, axis=1, keepdims=True), 1e-9)
        offset_um = direction * (radii[pick] * 1.4)[:, None]
        wall = self.roi.to_index(self.roi.to_world(pos[pick]) + offset_um)

        shape = np.asarray(self.roi.volume.shape) - 1
        background = rng.uniform(0, 1, size=(n_negative - n_wall, 3)) * shape

        neg = np.vstack([wall, background])
        neg = neg[self.roi.inside(neg)]

        x = np.vstack([self.features(pos), self.features(neg)])
        y = np.concatenate([np.ones(len(pos)), np.zeros(len(neg))])
        self.model = HistGradientBoostingClassifier(
            max_iter=150, learning_rate=0.1, max_depth=6, random_state=seed
        ).fit(x, y)
        self.training_score = float(self.model.score(x, y))
        return self

    def __call__(self, points_um: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("call fit() before using a LearnedProbability")
        idx = self.roi.to_index(points_um)
        inside = self.roi.inside(idx)
        out = np.zeros(len(idx))
        if inside.any():
            out[inside] = self.model.predict_proba(self.features(idx[inside]))[:, 1]
        return out


def sample_trilinear(volume: np.ndarray, zyx: np.ndarray) -> np.ndarray:
    """Trilinear sample of `volume` at fractional ``(z, y, x)``; 0 outside."""
    from scipy import ndimage

    idx = np.asarray(zyx, dtype=np.float64).reshape(-1, 3)
    return ndimage.map_coordinates(
        volume, idx.T, order=1, mode="constant", cval=0.0
    ).astype(np.float64)


# Publicly colocated with the legacy providers while its training/artifact code
# remains in a separate module with optional DF21 imports.
from .cfc import CfcProbability  # noqa: E402,F401
