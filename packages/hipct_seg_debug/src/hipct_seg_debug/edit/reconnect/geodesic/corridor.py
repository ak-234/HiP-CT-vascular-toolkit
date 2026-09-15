"""The box one repair works in, with everything resampled onto one grid.

:mod:`..roi` builds its ROI on the **raw** grid, because the DPC walk steps in raw
voxels. This package cannot: it has to reason about mask components, and a
component is defined on the segmentation grid, which is 2x2x2-binned relative to
the raw stack here. Doing it the other way -- upsampling the labels to raw -- would
octuple the memory for no gain and would invent sub-voxel component boundaries that
do not exist.

So the segmentation grid is the common frame, and the raw greyscale is sampled onto
it. That is a real loss of resolution and it is the right trade: the evidence terms
in :mod:`.cost` are all smoothed at the scale of the vessel radius, which is several
segmentation voxels even for a distal twig, so the binning is well below the scale
anything is measured at.

**Raw is optional.** With no raw stack the greyscale surrogate is the mask itself,
and the cost field then carries only geometric evidence. That is enough to
re-skeletonise a mask-connected break and enough to close a gap of a voxel or two,
and it is *not* enough to justify inventing a longer route -- so
:func:`~.route.plan` caps what a raw-less corridor may accept automatically rather
than letting the same thresholds through on weaker evidence.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

#: Never build a corridor thinner than this on any axis, in segmentation voxels.
MIN_SPAN = 8
#: Refuse rather than thrash. A corridor past this is not a vessel repair.
MAX_VOXELS = 40_000_000


class CorridorTooLarge(RuntimeError):
    """The box this candidate asked for is too big to be worth building."""


@dataclass
class Corridor:
    """A sub-box of the segmentation grid, plus greyscale sampled onto it."""

    volume: np.ndarray  # (dz, dy, dx) float32
    lo_zyx: np.ndarray  # global segmentation index of volume[0, 0, 0]
    spacing_um: np.ndarray  # (x, y, z), the segmentation spacing
    frame: object
    has_raw: bool = True

    @property
    def shape(self):
        return tuple(self.volume.shape)

    @property
    def hi_zyx(self) -> np.ndarray:
        return self.lo_zyx + np.asarray(self.volume.shape, dtype=np.int64)

    def to_index(self, points_um) -> np.ndarray:
        """World um -> fractional local ``(z, y, x)``."""
        points = np.asarray(points_um, dtype=np.float64).reshape(-1, 3)
        ijk = np.asarray(self.frame.um_to_seg(points), dtype=np.float64)
        return ijk[:, ::-1] - self.lo_zyx

    def to_global(self, points_um) -> np.ndarray:
        """World um -> nearest global segmentation ``(z, y, x)``."""
        points = np.asarray(points_um, dtype=np.float64).reshape(-1, 3)
        ijk = np.asarray(self.frame.um_to_seg(points), dtype=np.float64)
        return np.round(ijk[:, ::-1]).astype(np.int64)

    def to_world(self, zyx_global) -> np.ndarray:
        idx = np.asarray(zyx_global, dtype=np.float64).reshape(-1, 3)
        return np.asarray(self.frame.seg_to_um(idx[:, ::-1]), dtype=np.float64)

    def contains(self, zyx_global) -> np.ndarray:
        idx = np.asarray(zyx_global, dtype=np.int64).reshape(-1, 3)
        return np.all((idx >= self.lo_zyx) & (idx < self.hi_zyx), axis=1)

    def describe(self) -> str:
        source = "raw greyscale" if self.has_raw else "mask only (no --raw)"
        return (f"corridor {self.shape} at {tuple(int(v) for v in self.lo_zyx)}, "
                f"{source}")


def bounds(points_um, frame, pad_um: float) -> tuple[np.ndarray, np.ndarray]:
    """A padded segmentation-index box enclosing a set of world points."""
    points = np.asarray(points_um, dtype=np.float64).reshape(-1, 3)
    ijk = np.asarray(frame.um_to_seg(points), dtype=np.float64)[:, ::-1]
    spacing_zyx = np.asarray(frame.seg_spacing, dtype=np.float64)[::-1]
    pad = np.ceil(pad_um / np.maximum(spacing_zyx, 1e-9)).astype(np.int64)
    lo = np.floor(ijk.min(axis=0)).astype(np.int64) - pad
    hi = np.ceil(ijk.max(axis=0)).astype(np.int64) + pad + 1

    # Grow anything degenerate: a straight axis-aligned bridge has zero extent on
    # two axes, and a corridor with no room either side of the route cannot bulge
    # around an obstruction.
    short = (hi - lo) < MIN_SPAN
    if short.any():
        centre = (lo + hi) // 2
        lo = np.where(short, centre - MIN_SPAN // 2, lo)
        hi = np.where(short, centre + MIN_SPAN // 2 + 1, hi)

    dims = np.asarray(frame.seg_dims, dtype=np.int64)[::-1]  # (nz, ny, nx)
    lo = np.clip(lo, 0, dims)
    hi = np.clip(hi, lo + 1, dims)
    voxels = int(np.prod(hi - lo))
    if voxels > MAX_VOXELS:
        raise CorridorTooLarge(
            f"{tuple(int(v) for v in (hi - lo))} = {voxels:,} segmentation voxels"
        )
    return lo, hi


def build(frame, index, lo_zyx, hi_zyx, *, stack=None) -> Corridor:
    """Assemble the corridor, reading raw greyscale if a stack is available."""
    lo = np.asarray(lo_zyx, dtype=np.int64)
    hi = np.asarray(hi_zyx, dtype=np.int64)
    if stack is None:
        volume = index.window(lo, hi).astype(np.float32)
        # Presence, not identity: the cost field's intensity terms want "is this
        # foreground", and component identity reaches them through `blocked`.
        volume = (volume > 0).astype(np.float32)
        return Corridor(volume=volume, lo_zyx=lo,
                        spacing_um=np.asarray(frame.seg_spacing, dtype=np.float64),
                        frame=frame, has_raw=False)
    return Corridor(volume=_sample_raw(frame, stack, lo, hi), lo_zyx=lo,
                    spacing_um=np.asarray(frame.seg_spacing, dtype=np.float64),
                    frame=frame, has_raw=True)


def _sample_raw(frame, stack, lo, hi) -> np.ndarray:
    """Raw greyscale at each segmentation voxel centre in the box.

    One windowed read of the enclosing raw box, then a strided pick. Reading the
    raw stack slice by slice for each segmentation voxel would be thousands of
    19 MB TIFF decodes for a box a few hundred voxels on a side.
    """
    centres_zyx = [np.arange(lo[k], hi[k]) for k in range(3)]
    grid = np.stack(np.meshgrid(*centres_zyx, indexing="ij"), axis=-1)
    flat = grid.reshape(-1, 3).astype(np.float64)
    world = np.asarray(frame.seg_to_um(flat[:, ::-1]), dtype=np.float64)
    raw = np.asarray(frame.um_to_raw(world), dtype=np.float64)
    raw_idx = np.round(raw).astype(np.int64)

    nz, ny, nx = (int(v) for v in frame.raw_shape)
    raw_idx[:, 0] = np.clip(raw_idx[:, 0], 0, nz - 1)
    raw_idx[:, 1] = np.clip(raw_idx[:, 1], 0, ny - 1)
    raw_idx[:, 2] = np.clip(raw_idx[:, 2], 0, nx - 1)

    z0, y0, x0 = raw_idx.min(axis=0)
    z1, y1, x1 = raw_idx.max(axis=0) + 1
    window = np.asarray(stack.read_stack_window(int(z0), int(z1), int(y0), int(y1),
                                                int(x0), int(x1)))
    picked = window[raw_idx[:, 0] - z0, raw_idx[:, 1] - y0, raw_idx[:, 2] - x0]
    return picked.reshape(tuple(hi - lo)).astype(np.float32)


def for_candidate(frame, index, points_um, *, radius_um: float, stack=None,
                  pad_factor: float = 6.0, min_pad_um: float = 0.0) -> Corridor:
    """The corridor one candidate needs: its own geometry, plus room to wander.

    Generous on purpose, and for the same reason :func:`~..roi.pad_for` is: a
    corridor cropped to the straight line between two ends can only contain the
    routes that cut the chord, so the ones that follow the vessel round a bend --
    the correct ones -- would be squeezed out by the box rather than by the
    evidence.
    """
    spacing = np.asarray(frame.seg_spacing, dtype=np.float64)
    pad = max(pad_factor * float(radius_um), float(min_pad_um),
              MIN_SPAN * float(spacing.max()))
    lo, hi = bounds(points_um, frame, pad)
    return build(frame, index, lo, hi, stack=stack)
