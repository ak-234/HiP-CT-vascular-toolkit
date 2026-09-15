"""A raw-image sub-volume around one proposed bridge, ready for the DPC walk.

The walk needs greyscale, and the greyscale lives in a TIFF stack of thousands of
slices. Two things follow, and they decide the whole shape of this module:

* **One ROI per bridge, not one for the dataset.** ``FieldProbability`` runs a Sato
  vesselness filter over the whole of ``roi.volume`` in its constructor, so a
  dataset-sized ROI is not merely slow, it is not loadable. A bridge is a few
  hundred micrometres of vessel, and a box around it is a few megabytes.
* **The box has to be generous.** :func:`~.dpc.walk` gives up with "walked out of
  the region of interest" the moment its 5x5x5 neighbourhood leaves the array, so a
  box cropped tight to the straight line between two ends would abort exactly the
  walks that are doing their job -- the ones that bulge out to follow the vessel
  rather than cutting the chord.

The mask is read through the same lattice object every other command uses, so a
painting session's corrections are already composited in and the walk sees the mask
the user actually meant.
"""

from __future__ import annotations

import numpy as np

from .probability import Roi

#: Never build a box smaller than this many raw voxels on a side. A handful of
#: voxels would be inside the walk's own neighbourhood radius.
MIN_SPAN_VOXELS = 12
#: Refuse rather than thrash. A bridge needing more than this is centimetres long,
#: which the geometric gates should have refused before the image was ever asked.
MAX_VOXELS = 24_000_000


class RoiTooLarge(RuntimeError):
    """The box a bridge asked for is too big to be worth reading."""


def bounds_for(points_um, pad_um: float) -> tuple[np.ndarray, np.ndarray]:
    """``(lo, hi)`` corners in world um, padded, in ``(x, y, z)`` order."""
    pts = np.asarray(points_um, dtype=np.float64).reshape(-1, 3)
    return pts.min(axis=0) - pad_um, pts.max(axis=0) + pad_um


def box_for(frame, lo_um, hi_um) -> tuple[int, int, int, int, int, int]:
    """World corners -> a clamped integer raw box ``(z0, z1, y0, y1, x0, x1)``.

    Both corners go through the same transform, so an axis flip in the frame
    cannot silently produce an inside-out box.
    """
    corners = frame.um_to_raw(np.vstack([lo_um, hi_um]))
    z0, y0, x0 = np.floor(corners.min(axis=0)).astype(int)
    z1, y1, x1 = np.ceil(corners.max(axis=0)).astype(int) + 1

    nz_raw, ny_raw, nx_raw = frame.raw_shape
    z0, z1 = _clamp_span(z0, z1, nz_raw)
    y0, y1 = _clamp_span(y0, y1, ny_raw)
    x0, x1 = _clamp_span(x0, x1, nx_raw)

    voxels = (z1 - z0) * (y1 - y0) * (x1 - x0)
    if voxels > MAX_VOXELS:
        raise RoiTooLarge(f"{z1 - z0} x {y1 - y0} x {x1 - x0} = {voxels:,} voxels")
    return z0, z1, y0, y1, x0, x1


def build(stack, frame, lo_um, hi_um, *, labels=None) -> Roi:
    """Read the raw box spanning ``lo_um``..``hi_um`` and wrap it as an :class:`Roi`.

    `stack` is a :class:`~...tiffstack.TiffStack`, `frame` a
    :class:`~...frame.WorldFrame`, and `labels` an optional lattice (or
    ``MaskSource``) sampled onto the *same* grid as the raw box -- not its own --
    so that ``Roi.volume`` and ``Roi.mask`` are index-for-index comparable, which
    is what ``FieldProbability`` assumes when it adds the two fields together.
    """
    z0, z1, y0, y1, x0, x1 = box_for(frame, lo_um, hi_um)
    volume = stack.read_stack_window(z0, z1, y0, y1, x0, x1)
    return _wrap(volume, frame, labels, (z0, z1), (y0, y1), (x0, x1))


def build_many(stack, frame, spans, *, labels=None, progress=None) -> list:
    """Build many ROIs in **one ordered pass over the raw stack**.

    `spans` is a sequence of ``(lo_um, hi_um)`` corner pairs; the return is a list
    of :class:`Roi` aligned with it, with ``None`` wherever the box was refused as
    too large.

    Reading each ROI on its own costs one full TIFF decode per slice *per ROI*, and
    a HiP-CT slice is 19 MB whether you want all of it or an 84x76 window of it. The
    boxes overlap heavily in z -- they are all on the same coronary tree -- so
    reading every needed slice exactly once, and filling every box that covers it,
    halves the I/O on a typical run and cannot do worse than the naive order.
    """
    boxes: list[tuple | None] = []
    volumes: list[np.ndarray | None] = []
    for lo_um, hi_um in spans:
        try:
            box = box_for(frame, lo_um, hi_um)
        except RoiTooLarge:
            boxes.append(None)
            volumes.append(None)
            continue
        z0, z1, y0, y1, x0, x1 = box
        boxes.append(box)
        volumes.append(np.zeros((z1 - z0, y1 - y0, x1 - x0), dtype=stack.dtype))

    wanted: dict[int, list[int]] = {}
    for i, box in enumerate(boxes):
        if box is None:
            continue
        for z in range(box[0], box[1]):
            wanted.setdefault(z, []).append(i)

    for n, z in enumerate(sorted(wanted)):
        plane = stack.read_slice(z)
        for i in wanted[z]:
            z0, _z1, y0, y1, x0, x1 = boxes[i]
            volumes[i][z - z0] = plane[y0:y1, x0:x1]
        if progress is not None:
            progress(n + 1, len(wanted))

    return [
        None if box is None else _wrap(
            volume, frame, labels, (box[0], box[1]), (box[2], box[3]), (box[4], box[5])
        )
        for box, volume in zip(boxes, volumes)
    ]


def _wrap(volume, frame, labels, z_span, y_span, x_span) -> Roi:
    """Attach the world geometry -- and the mask -- to a box of raw voxels."""
    origin_um = frame.raw_to_um(np.array([[z_span[0], y_span[0], x_span[0]]]))[0]
    spacing_um = np.asarray(frame.raw_voxel, dtype=np.float64)
    mask = None
    if labels is not None:
        mask = _mask_on_raw_grid(labels, frame, z_span, y_span, x_span)
    return Roi(volume=volume, origin_um=origin_um, spacing_um=spacing_um, mask=mask)


def pad_for(bridge, frame, pad_factor: float = 4.0) -> float:
    """How much room to leave around a bridge, in um.

    Proportional to the vessel, with a floor: :func:`~.dpc.walk` aborts the moment
    its 5x5x5 neighbourhood leaves the array, so a box that merely contains the
    proposed path would kill the walks that are doing their job -- the ones that
    bulge out to follow the vessel instead of cutting the chord.
    """
    radius = float(bridge.metrics.get("r_source", 0.0)) or float(
        np.max(bridge.radii) if len(bridge.radii) else 0.0
    )
    return max(pad_factor * radius, MIN_SPAN_VOXELS * float(np.max(frame.raw_voxel)))


def span_for(bridge, frame, *, pad_factor: float = 4.0, extra_points_um=None):
    """``(lo_um, hi_um)`` for one bridge: its own path, plus room to wander.

    `extra_points_um` is for a T-junction, whose goal is a whole polyline rather
    than a point: the target centreline has to be inside the box or the distance
    term aims at somewhere the walk can never reach.
    """
    points = [np.asarray(bridge.coords, dtype=np.float64).reshape(-1, 3)]
    if extra_points_um is not None and len(extra_points_um):
        points.append(np.asarray(extra_points_um, dtype=np.float64).reshape(-1, 3))
    return bounds_for(np.vstack(points), pad_for(bridge, frame, pad_factor))


def for_bridge(stack, frame, bridge, *, labels=None, pad_factor: float = 4.0,
               extra_points_um=None) -> Roi:
    """The ROI a single bridge needs. See :func:`build_many` for a whole run."""
    lo, hi = span_for(bridge, frame, pad_factor=pad_factor,
                      extra_points_um=extra_points_um)
    return build(stack, frame, lo, hi, labels=labels)


def near_target_segment(graph, bridge, *, keep_um: float) -> np.ndarray:
    """The part of a T-junction's target vessel worth putting inside the ROI.

    The whole parent vessel can be centimetres long; only the neighbourhood of the
    attachment point is a plausible destination, and boxing the rest in would blow
    up the ROI for nothing.
    """
    if bridge.target_segment is None:
        return np.empty((0, 3))
    coords = np.asarray(graph.coords(bridge.target_segment), dtype=np.float64)
    if not len(coords):
        return coords
    anchor = np.asarray(bridge.coords[-1], dtype=np.float64)
    return coords[np.linalg.norm(coords - anchor, axis=1) <= keep_um]


def _clamp_span(lo: int, hi: int, limit: int) -> tuple[int, int]:
    """Clip to ``[0, limit]`` while keeping at least `MIN_SPAN_VOXELS` of extent."""
    lo, hi = int(lo), int(max(hi, lo + 1))
    if hi - lo < MIN_SPAN_VOXELS:
        centre = (lo + hi) // 2
        lo, hi = centre - MIN_SPAN_VOXELS // 2, centre + MIN_SPAN_VOXELS // 2
    lo = max(lo, 0)
    hi = min(hi, limit)
    return lo, max(hi, lo + 1)


def _mask_on_raw_grid(labels, frame, z_span, y_span, x_span) -> np.ndarray:
    """Sample the segmentation lattice onto the raw box's grid.

    The lattice is binned relative to the raw stack (2x2x2 here), so this is a
    nearest-neighbour lookup rather than a crop: every raw voxel asks which
    segmentation voxel contains it. Done per raw slice so only the rows and columns
    in the box are ever touched.
    """
    z0, z1 = z_span
    y0, y1 = y_span
    x0, x1 = x_span
    start = np.asarray(frame.raw_start, dtype=np.int64)      # (x, y, z)
    binf = np.asarray(frame.bin_factor, dtype=np.int64)      # (x, y, z)
    nx_seg, ny_seg, nz_seg = (int(v) for v in frame.seg_dims)

    cols = np.clip((np.arange(x0, x1) - start[0]) // binf[0], 0, nx_seg - 1)
    rows = np.clip((np.arange(y0, y1) - start[1]) // binf[1], 0, ny_seg - 1)
    slices = np.clip((np.arange(z0, z1) - start[2]) // binf[2], 0, nz_seg - 1)

    out = np.zeros((z1 - z0, y1 - y0, x1 - x0), dtype=bool)
    # Consecutive raw slices share a segmentation slice under binning, so decode
    # each distinct one once.
    for seg_z in np.unique(slices):
        plane = np.asarray(labels.slice_z(int(seg_z)))[np.ix_(rows, cols)] > 0
        out[slices == seg_z] = plane
    return out
