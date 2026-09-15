"""Lazy whole-dataset layers, for scrolling past the slab.

The slab is a window: ``--slab`` slices by ``--roi`` pixels around one pick. That
is the right unit for auditing a vessel and the wrong one for asking "does this
segmentation defect continue outside the box?", because the z slider cannot leave
the slab at all.

These two arrays span the entire dataset instead. Both are dask arrays chunked one
slice at a time, so napari materialises only the slice being displayed:

* ``lazy_raw``          4753 x 3079 x 3154 uint16 -- 92 GB if it were real
* ``lazy_segmentation`` 1250 x 1250 x 1500 uint8  -- 2.34 GB if it were real

Neither is ever real. The readers underneath are already slice-at-a-time
(``TiffStack.read_slice``, ``ByteRLELattice.slice_z``), and both already keep a
cache, so scrolling costs one decode per new slice and nothing else. Measured on
LADAF-2024-28: a segmentation slice is 1.1 ms decoded directly and 15 ms through
dask's task machinery, a raw slice ~0.4 s cold and ~65 ms from the LRU. Building
the two graphs takes about 1.4 s once, at startup.

The segmentation array stays on **its own grid** rather than being upsampled onto
the raw one the way ``viewer2d._seg_window`` does. That is what makes this cheap
-- a raw-grid copy of one slab is 60x the memory -- and napari places it
correctly through ``scale`` and ``translate`` instead. See :func:`seg_placement`
for the half-voxel detail that makes those two numbers right.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


def _require_dask():
    try:
        import dask
        import dask.array as da
    except ImportError as exc:  # pragma: no cover - environment issue, not logic
        raise ImportError(
            "the whole-volume layers need dask; install it with `pip install dask`"
        ) from exc
    return dask, da


def lazy_raw(stack):
    """The whole raw stack as a dask array, one chunk per slice.

    Chunked ``(1, n_rows, n_cols)`` because that is exactly what the reader can
    produce: ``tifffile`` decodes a whole LZW page and there is no windowed read,
    so a smaller chunk would decode the same data and throw most of it away.
    """
    dask, da = _require_dask()

    nz, n_rows, n_cols = stack.shape
    read = dask.delayed(stack.read_slice, pure=True)
    planes = [
        da.from_delayed(read(z), shape=(n_rows, n_cols), dtype=stack.dtype)
        for z in range(nz)
    ]
    return da.stack(planes, axis=0)


def lazy_segmentation(lattice):
    """One ``HxByteRLE`` field as a dask array on the segmentation's own grid.

    Shape is ``(nz, ny, nx)`` to match napari's ``(z, row, col)``; the lattice
    itself is indexed x-fastest, which ``slice_z`` already resolves by returning
    ``(ny, nx)``.
    """
    dask, da = _require_dask()

    read = dask.delayed(lattice.slice_z, pure=True)
    planes = [
        da.from_delayed(read(k), shape=(lattice.ny, lattice.nx), dtype=np.uint8)
        for k in range(lattice.nz)
    ]
    return da.stack(planes, axis=0)


def seg_placement(frame) -> tuple[tuple, tuple]:
    """``(scale, translate)`` in napari ``(z, row, col)`` that puts the
    segmentation grid onto the raw pixel grid.

    The half-voxel term is the whole subtlety. napari maps
    ``world = index * scale + translate`` at pixel **centres**, but
    ``frame.raw_start`` is the *first raw index covered by* segmentation voxel 0
    -- a corner, defined as ``round(seg_origin / raw_voxel - (bin - 1) / 2)``
    (``frame.py:81-85``). Segmentation voxel ``s`` therefore spans raw indices
    ``raw_start + s*bin`` to ``raw_start + (s+1)*bin - 1``, whose centre sits
    ``(bin - 1) / 2`` further on.

    Drop that term and the mask is displaced by half a raw voxel everywhere --
    which looks entirely plausible and is wrong. ``selftest.test_lazy_seg_alignment``
    checks this against the gather in ``viewer2d._seg_window``.
    """
    bin_zyx = np.asarray(frame.bin_factor, dtype=np.float64)[::-1]
    start_zyx = np.asarray(frame.raw_start, dtype=np.float64)[::-1]
    scale = tuple(float(b) for b in bin_zyx)
    translate = tuple(float(s + (b - 1.0) / 2.0) for s, b in zip(start_zyx, bin_zyx))
    return scale, translate


def seg_window_placement(frame, origin_kji) -> tuple[tuple, tuple]:
    """``(scale, translate)`` for a *sub-block* of the segmentation grid.

    A paintable layer cannot be the whole 2.34 GB mask, so it is a box starting at
    segmentation index ``origin_kji = (k0, row0, col0)``. Index ``i`` of that box is
    segmentation voxel ``origin + i``, so the placement is :func:`seg_placement`
    with the origin folded into the offset:

        world = (origin + i) * scale + translate
              = i * scale + (translate + origin * scale)

    Sharing the derivation with :func:`seg_placement` rather than restating it is
    the point -- the half-voxel term documented there is easy to get right once and
    very easy to get wrong twice, and a painted voxel that lands half a raw voxel
    from where it was drawn looks entirely plausible.
    """
    scale, translate = seg_placement(frame)
    origin = np.asarray(origin_kji, dtype=np.float64).reshape(3)
    return scale, tuple(float(t + o * s) for t, o, s in zip(translate, origin, scale))


@dataclass
class VolumeSource:
    """The lazy arrays and their placement, ready to hand to napari.

    Built once per session; the arrays are graphs, not data, so this is cheap and
    holding it costs nothing.
    """

    raw: object | None = None
    seg: object | None = None
    seg_scale: tuple = (1.0, 1.0, 1.0)
    seg_translate: tuple = (0.0, 0.0, 0.0)
    names: tuple = field(default_factory=tuple)

    @classmethod
    def build(cls, stack=None, labels=None, frame=None) -> "VolumeSource":
        names = []
        raw = seg = None
        scale = translate = None
        if stack is not None:
            raw = lazy_raw(stack)
            names.append("raw (all)")
        if labels is not None and frame is not None:
            seg = lazy_segmentation(labels)
            scale, translate = seg_placement(frame)
            names.append("segmentation (all)")
        return cls(
            raw=raw,
            seg=seg,
            seg_scale=scale or (1.0, 1.0, 1.0),
            seg_translate=translate or (0.0, 0.0, 0.0),
            names=tuple(names),
        )

    def describe(self) -> str:
        bits = []
        if self.raw is not None:
            bits.append(f"raw {tuple(int(v) for v in self.raw.shape)}")
        if self.seg is not None:
            bits.append(
                f"segmentation {tuple(int(v) for v in self.seg.shape)} "
                f"scale {tuple(int(v) for v in self.seg_scale)}"
            )
        return "whole-volume layers: " + (", ".join(bits) if bits else "none")
