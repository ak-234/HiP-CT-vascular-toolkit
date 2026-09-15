"""napari slice browser: raw image with every derived representation overlaid.

All layers are placed in **global raw-stack coordinates** (slice, row, col) via each
image layer's ``translate``, so napari's own slider reads the true TIFF slice number
and the point/shape layers need no local bookkeeping.

napari supplies the requested controls natively: the layer list gives a visibility
checkbox per overlay and an opacity slider for the selected layer, and the dims slider
scrolls the slab.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage

# Layer draw order is bottom-to-top; keep the greyscale underneath everything.
SEG_COLOR = {1: "#00b0ff"}
STL_COLOR = "#ff3b30"
SKEL_COLOR = "#ffd400"
CIRCLE_COLOR = "#00e676"
PERIM_COLOR = "#ff9500"
TUBE_COLOR = {1: "#c158dc"}
# A deliberately different blue from SEG_COLOR: the whole-volume mask and the
# slab's mask overlap wherever the slab is, and telling them apart is the point
# of showing both.
SEG_ALL_COLOR = {1: "#4dd0e1"}


@dataclass
class Slab:
    """A cube of raw data around a picked point, plus everything that overlays it."""

    z_lo: int
    z_hi: int  # exclusive
    row0: int
    row1: int
    col0: int
    col1: int
    centre_um: np.ndarray
    raw: np.ndarray
    seg: np.ndarray | None = None
    prob: np.ndarray | None = None
    tube: np.ndarray | None = None
    contours: list | None = None
    skel_pts: np.ndarray | None = None
    skel_r: np.ndarray | None = None
    circles: list | None = None
    perim_circles: list | None = None
    # Where the pick landed, as a raw (slice, row, col). Kept because once the
    # window is the whole slice, row0/col0 no longer say anything about it -- and
    # the camera and the screenshot name both still need to know.
    pick_zrc: tuple = (0, 0, 0)
    # True when the window is the whole slice rather than an ROI crop.
    full_slice: bool = False

    @property
    def z_centre(self) -> int:
        return (self.z_lo + self.z_hi - 1) // 2

    @property
    def translate(self) -> tuple:
        return (self.z_lo, self.row0, self.col0)


class SlabBuilder:
    """Assembles a `Slab` from the four inputs for a given world point."""

    def __init__(self, frame, stack, graph, labels=None, probability=None, slicer=None):
        self.frame = frame
        self.stack = stack
        self.graph = graph
        self.labels = labels
        self.probability = probability
        self.slicer = slicer
        self._seg_cache: dict[int, np.ndarray] = {}
        self._prob_cache: dict[int, np.ndarray] = {}

    # -- segmentation resampling ------------------------------------------- #
    def _seg_window(self, lattice, cache, z_range, rows, cols) -> np.ndarray:
        """Gather a segmentation window onto the raw pixel grid.

        The lattice is binned relative to the raw stack, so each raw index maps to one
        lattice index by integer division. Gathering with those index arrays handles the
        upsampling and the edge clipping in one step.
        """
        f = self.frame
        nx, ny, nz = (int(v) for v in f.seg_dims)
        si = np.clip(f.raw_axis_to_seg_axis(cols, 0), 0, nx - 1)
        sj = np.clip(f.raw_axis_to_seg_axis(rows, 1), 0, ny - 1)
        in_x = (cols >= f.raw_start[0]) & (f.raw_axis_to_seg_axis(cols, 0) < nx)
        in_y = (rows >= f.raw_start[1]) & (f.raw_axis_to_seg_axis(rows, 1) < ny)

        out = np.zeros((len(z_range), len(rows), len(cols)), dtype=np.uint8)
        for n, z in enumerate(z_range):
            k = int(f.raw_slice_to_seg_slice(z))
            if not 0 <= k < nz:
                continue
            sl = cache.get(k)
            if sl is None:
                sl = lattice.slice_z(k)
                cache[k] = sl
                if len(cache) > 32:
                    cache.pop(next(iter(cache)))
            out[n] = sl[np.ix_(sj, si)]
        out[:, ~in_y, :] = 0
        out[:, :, ~in_x] = 0
        return out

    # -- skeleton ----------------------------------------------------------- #
    def _centreline_on_slices(self, z_range, rows, cols):
        """Where the centreline meets each slice, with the local radius and tangent.

        The skeleton was computed on the binned segmentation, so its points sit at
        half-integer raw z and land on only every other slice. Treating each edge as
        the polyline it is and intersecting it with every slice plane gives a marker on
        every slice, and an interpolated radius to go with it. Segments that run
        nearly within the plane produce no crossing, so points lying inside the slice
        are included as well.

        Returns ``(zyx, radius, tangent)`` in global (slice, row, col) coordinates.
        """
        f = self.frame
        pts_raw = f.um_to_raw(self.graph.points)  # (P, 3) as (slice, row, col)
        rad = self.graph.thickness
        off = self.graph.edge_offsets

        r_pad = float(rad.max()) / f.raw_voxel[0]
        row_lo, row_hi = rows[0] - r_pad, rows[-1] + r_pad
        col_lo, col_hi = cols[0] - r_pad, cols[-1] + r_pad
        z_lo, z_hi = float(z_range[0]), float(z_range[-1])

        out_p, out_r, out_t = [], [], []
        for e in range(self.graph.n_edge):
            a, b = int(off[e]), int(off[e + 1])
            if b - a < 1:
                continue
            seg = pts_raw[a:b]
            # Cheap reject: skip edges nowhere near this slab.
            if (
                seg[:, 0].max() < z_lo - 1
                or seg[:, 0].min() > z_hi + 1
                or seg[:, 1].max() < row_lo
                or seg[:, 1].min() > row_hi
                or seg[:, 2].max() < col_lo
                or seg[:, 2].min() > col_hi
            ):
                continue
            sr = rad[a:b]
            for z in z_range:
                z = float(z)
                if b - a >= 2:
                    z0, z1 = seg[:-1, 0], seg[1:, 0]
                    dz = z1 - z0
                    with np.errstate(divide="ignore", invalid="ignore"):
                        t = np.where(np.abs(dz) > 1e-9, (z - z0) / dz, -1.0)
                    hit = (t >= 0.0) & (t <= 1.0)
                    for k in np.flatnonzero(hit):
                        tk = t[k]
                        p = seg[k] + tk * (seg[k + 1] - seg[k])
                        r = sr[k] + tk * (sr[k + 1] - sr[k])
                        d = seg[k + 1] - seg[k]
                        n = np.linalg.norm(d)
                        out_p.append([z, p[1], p[2]])
                        out_r.append(r)
                        out_t.append(d / n if n > 1e-9 else np.array([1.0, 0.0, 0.0]))
                # Segments running within this plane never "cross" it.
                flat = np.flatnonzero(np.abs(seg[:, 0] - z) <= 0.5)
                for k in flat:
                    d = (
                        seg[min(k + 1, b - a - 1)] - seg[max(k - 1, 0)]
                        if b - a >= 2
                        else np.array([1.0, 0.0, 0.0])
                    )
                    n = np.linalg.norm(d)
                    out_p.append([z, seg[k, 1], seg[k, 2]])
                    out_r.append(sr[k])
                    out_t.append(d / n if n > 1e-9 else np.array([1.0, 0.0, 0.0]))

        if not out_p:
            return np.empty((0, 3)), np.empty(0), np.empty((0, 3))
        P = np.array(out_p)
        R = np.array(out_r)
        T = np.array(out_t)
        # Markers are clipped to the visible crop exactly -- the rasterised tube layer
        # already accounts for vessels centred just outside it.
        keep = (
            (P[:, 1] >= rows[0])
            & (P[:, 1] <= rows[-1])
            & (P[:, 2] >= cols[0])
            & (P[:, 2] <= cols[-1])
        )
        return P[keep], R[keep], T[keep]

    def _radius_ellipses(self, zyx, radius, tangent):
        """The cross-section each radius implies on this slice, as napari ellipse corners.

        A tube of radius r whose axis is oblique to the slice cuts the plane as an
        ellipse: semi-minor r, semi-major r/|t_z|, elongated along the in-plane
        direction of the axis. Drawing that rather than a plain circle is what makes
        the overlay directly comparable to the segmented blob. The elongation is capped
        so near-in-plane vessels do not produce runaway shapes -- the rasterised tube
        layer covers that case properly.
        """
        f = self.frame
        vx = f.raw_voxel[0]
        out = []
        for (z, row, col), r, t in zip(zyx, radius, tangent):
            rp = r / vx
            inplane = t[1:]
            n = np.linalg.norm(inplane)
            if n < 1e-6:
                u = np.array([1.0, 0.0])
            else:
                u = inplane / n
            v = np.array([-u[1], u[0]])
            stretch = min(1.0 / max(abs(t[0]), 0.34), 3.0)
            a = rp * stretch  # along the vessel axis
            bb = rp  # across it
            c = np.array([row, col])
            corners = [c - a * u - bb * v, c + a * u - bb * v, c + a * u + bb * v, c - a * u + bb * v]
            out.append(np.array([[z, p[0], p[1]] for p in corners]))
        return out

    def _tube_mask(self, z_range, rows, cols) -> np.ndarray:
        """Rasterised union of the spheres the skeleton radii imply.

        This is what the reconstruction asserts the lumen to be, independent of the
        surface mesh -- and unlike the mesh it exists everywhere the skeleton does.
        Comparing it against the greyscale is the quickest way to see a re-inflated
        collapsed vessel.
        """
        f = self.frame
        pts, rad = self.graph.points, self.graph.thickness
        vx, vy, vz = f.raw_voxel
        x_lo, x_hi = cols[0] * vx, cols[-1] * vx
        y_lo, y_hi = rows[0] * vy, rows[-1] * vy
        z_lo, z_hi = z_range[0] * vz, z_range[-1] * vz
        r_max = float(rad.max())
        m = (
            (pts[:, 0] >= x_lo - r_max)
            & (pts[:, 0] <= x_hi + r_max)
            & (pts[:, 1] >= y_lo - r_max)
            & (pts[:, 1] <= y_hi + r_max)
            & (pts[:, 2] >= z_lo - r_max)
            & (pts[:, 2] <= z_hi + r_max)
        )
        idx = np.flatnonzero(m)
        out = np.zeros((len(z_range), len(rows), len(cols)), dtype=np.uint8)
        if idx.size == 0:
            return out
        cc = cols[None, :] * vx
        rr = rows[:, None] * vy
        for n, z in enumerate(z_range):
            zc = z * vz
            dz = np.abs(pts[idx, 2] - zc)
            hit = dz < rad[idx]
            if not hit.any():
                continue
            sel = idx[hit]
            reff = np.sqrt(np.maximum(rad[sel] ** 2 - (pts[sel, 2] - zc) ** 2, 0.0))
            plane = out[n]
            for p, r in zip(pts[sel], reff):
                if r <= 0:
                    continue
                c_lo = np.searchsorted(cols * vx, p[0] - r)
                c_hi = np.searchsorted(cols * vx, p[0] + r, side="right")
                r_lo = np.searchsorted(rows * vy, p[1] - r)
                r_hi = np.searchsorted(rows * vy, p[1] + r, side="right")
                if c_lo >= c_hi or r_lo >= r_hi:
                    continue
                d2 = (cc[:, c_lo:c_hi] - p[0]) ** 2 + (rr[r_lo:r_hi] - p[1]) ** 2
                plane[r_lo:r_hi, c_lo:c_hi] |= (d2 <= r * r).astype(np.uint8)
        return out

    # -- assembly ----------------------------------------------------------- #
    def build(
        self,
        centre_um,
        half_slices: int = 5,
        roi_px: int = 400,
        with_probability: bool = False,
        with_tube: bool = True,
        with_surface: bool = True,
    ) -> Slab:
        f = self.frame
        centre_um = np.asarray(centre_um, dtype=np.float64).reshape(3)
        zc, rc, cc = f.um_to_raw_index(centre_um)[0]
        nz, nrow, ncol = self.stack.shape

        z_lo = int(np.clip(zc - half_slices, 0, nz - 1))
        z_hi = int(np.clip(zc + half_slices + 1, 1, nz))
        if roi_px and int(roi_px) > 0:
            half = int(roi_px) // 2
            row0, row1 = int(rc - half), int(rc + half)
            col0, col1 = int(cc - half), int(cc + half)
        else:
            # roi_px <= 0 means the whole slice. Nothing else in the coordinate
            # model has to change: array layers are placed by `translate`, which
            # becomes (z_lo, 0, 0), and every point/shape product is already in
            # global (slice, row, col). It costs no extra decode either -- both
            # readers already decode a whole slice and crop afterwards.
            row0, row1 = 0, int(nrow)
            col0, col1 = 0, int(ncol)

        z_range = np.arange(z_lo, z_hi)
        rows = np.arange(row0, row1)
        cols = np.arange(col0, col1)

        slab = Slab(
            z_lo=z_lo,
            z_hi=z_hi,
            row0=row0,
            row1=row1,
            col0=col0,
            col1=col1,
            centre_um=centre_um,
            raw=self.stack.read_stack_window(z_lo, z_hi, row0, row1, col0, col1),
            pick_zrc=(int(zc), int(rc), int(cc)),
            full_slice=not (roi_px and int(roi_px) > 0),
        )

        if self.labels is not None:
            slab.seg = self._seg_window(self.labels, self._seg_cache, z_range, rows, cols)
        if with_probability and self.probability is not None:
            slab.prob = self._seg_window(self.probability, self._prob_cache, z_range, rows, cols)
        if with_tube:
            slab.tube = self._tube_mask(z_range, rows, cols)

        # Surface contours, in global (slice, row, col).
        contours = []
        if with_surface and self.slicer is not None:
            vx, vy, vz = f.raw_voxel
            pad = 2.0 * vz
            self.slicer.set_roi(
                (
                    col0 * vx,
                    col1 * vx,
                    row0 * vy,
                    row1 * vy,
                    z_lo * vz - pad,
                    (z_hi - 1) * vz + pad,
                )
            )
            for z in z_range:
                for pl in self.slicer.contours(z * vz):
                    if len(pl) < 2:
                        continue
                    arr = np.empty((len(pl), 3))
                    arr[:, 0] = z
                    arr[:, 1] = pl[:, 1] / vy
                    arr[:, 2] = pl[:, 0] / vx
                    contours.append(arr)
        slab.contours = contours

        zyx, rr, tt = self._centreline_on_slices(z_range, rows, cols)
        slab.skel_pts, slab.skel_r = zyx, rr
        slab.circles = self._radius_ellipses(zyx, rr, tt)
        slab.perim_circles = self._perimeter_circles(slab, zyx, rr, tt)
        return slab

    def _perimeter_circles(self, slab, zyx, radius, tangent):
        """The circle the pipeline's own rule would draw from the segmented perimeter.

        ``adjust_thickness.py`` assigns ``r = perimeter / (2*pi)`` on the assumption that
        the lumen perimeter survives fixation even though the lumen collapses. Drawing
        that circle next to the assumed-radius circle and the segmentation is what makes
        it visible whether re-inflation is behaving here, or whether the radius has
        drifted away from what the image actually supports.
        """
        if slab.seg is None or not len(zyx):
            return []
        from skimage.measure import find_contours

        f = self.frame
        vx = f.raw_voxel[0]
        out = []
        for (z, row, col), r, t in zip(zyx, radius, tangent):
            n = int(round(z)) - slab.z_lo
            if not (0 <= n < slab.seg.shape[0]):
                continue
            rp = r / vx
            j = int(round(row - slab.row0))
            i = int(round(col - slab.col0))
            half = int(rp * 2.5) + 3
            j0, i0 = max(0, j - half), max(0, i - half)
            sub = slab.seg[n, j0 : j + half, i0 : i + half]
            if sub.size == 0:
                continue
            cj, ci = j - j0, i - i0
            if not (0 <= cj < sub.shape[0] and 0 <= ci < sub.shape[1]) or not sub[cj, ci]:
                continue
            lab, _ = ndimage.label(sub, structure=np.ones((3, 3), dtype=int))
            blob = lab == lab[cj, ci]
            if blob.sum() < 8:
                continue
            per = max(
                (
                    np.linalg.norm(np.diff(c, axis=0), axis=1).sum()
                    for c in find_contours(blob.astype(float), 0.5)
                ),
                default=0.0,
            )
            if per <= 0:
                continue
            rq = per / (2.0 * np.pi)  # already in raw pixels: seg was gathered onto that grid
            out.append(
                np.array(
                    [
                        [int(round(z)), row - rq, col - rq],
                        [int(round(z)), row - rq, col + rq],
                        [int(round(z)), row + rq, col + rq],
                        [int(round(z)), row + rq, col - rq],
                    ]
                )
            )
        return out


def show(slab: Slab, frame, title: str = "HiP-CT segmentation debugger",
         block: bool = True, paint=None):
    """Open the slab in a fresh napari viewer."""
    import napari

    viewer = napari.Viewer(title=title)
    build_layers(viewer, slab, frame, paint=paint)
    if block:
        napari.run()
    return viewer


def build_layers(viewer, slab: Slab, frame, on_slice=None, volume=None, paint=None):
    """(Re)populate a viewer with this slab. Safe to call on a viewer already in use.

    Reusing one window across picks keeps the session tidy and avoids leaking a viewer
    (and its slab) per pick.

    ``on_slice(z)`` is called with the raw slice number whenever the z slider moves, and
    once here. The 3D window uses it to keep its image plane on the slice being read.

    ``volume`` is an optional :class:`~.volume.VolumeSource`; its layers span the
    whole dataset rather than the slab, are added once, and are never touched
    again -- they are lazy graphs, so there is nothing per-pick to refresh.

    ``paint`` is an optional :class:`~.edit.paint.PaintSession`. Unlike every other
    layer here it is *writable*, and it is on the segmentation's own grid rather
    than the raw one -- see that module for why. It commits itself before the slab
    is swapped underneath it.
    """
    reused = _expected_names(slab, volume, paint) == {layer.name for layer in viewer.layers}
    if reused:
        _update_layers(viewer, slab, paint)
    else:
        viewer.layers.clear()
        _create_layers(viewer, slab, volume, paint)

    _show_slice(viewer, slab.z_centre)
    if reused:
        # Consecutive picks are usually nowhere near each other, so re-frame. This has
        # to follow _show_slice: it frames the layers' *augmented* extents, which the
        # forced refresh in there is what invalidates.
        _reframe(viewer, slab)

    # napari's slider reads out a step index; show the real TIFF slice number instead.
    viewer.text_overlay.visible = True
    viewer.text_overlay.font_size = 11
    viewer.text_overlay.color = "white"

    state = _viewer_state(viewer)
    # Dispatched through the state rather than captured, so that the connection made on
    # the first pick always calls the *current* pick's callback instead of stacking one
    # connection per pick.
    state["on_slice"] = on_slice

    def _on_step(_event=None):
        z = int(round(viewer.dims.point[0]))
        viewer.text_overlay.text = f"slice {z}   z = {z * frame.raw_voxel[2]:,.0f} um"
        cb = _viewer_state(viewer).get("on_slice")
        if cb is not None:
            cb(z)

    _on_step()
    if not state.get("overlay_hooked"):
        viewer.dims.events.current_step.connect(_on_step)
        state["overlay_hooked"] = True

    _attach_info(viewer, slab, paint)
    return viewer


def _force_extents(viewer):
    """Recompute every layer's cached extent, including the hidden ones.

    ``Layer._refresh_sync`` returns *before* it clears the extent cache when the layer
    is not visible, so a hidden layer keeps the previous slab's extent indefinitely.
    That stale extent is then unioned into ``dims.range`` and into the extent
    ``reset_view`` frames, which widens the z slider across both slabs and zooms the
    camera out over empty space. ``force=True`` is the documented way past that gate.

    The layer's ``events.extent`` invalidates the LayerList's own cached extent, but
    ``dims.range`` is only recomputed from data/scale/transform events, so it has to be
    pushed across by hand afterwards.
    """
    for layer in viewer.layers:
        layer.refresh(
            thumbnail=False, data_displayed=False, highlight=False, extent=True, force=True
        )
    viewer.dims.range = _whole_slices(viewer.layers._ranges)


def _whole_slices(ranges) -> tuple:
    """Pin the z axis to whole raw slices.

    Any layer on the *segmentation* grid sits at raw coordinate
    ``raw_start + (bin - 1) / 2``, which is a half-integer whenever the bin factor is
    even -- 2101.5 on LADAF-2024-28. That is correct: a mask voxel really does
    straddle two raw slices, and :func:`~.volume.seg_placement` exists to say so.

    But napari lays its slider grid out from the union's ``start``, so a layer that
    *sets* that start shifts every step by half a slice: asking for slice 2767 lands
    on 2767.5, which displays 2768. The browser then disagrees with the pick, with
    the info panel, and with the 3D window's image plane. This module's whole premise
    is that the slider reads the true TIFF slice number (see the module docstring),
    so put it back.

    Measured on LADAF-2024-28, 4 picks each: without this, ``--paint`` lands on the
    right slice 1 time in 4. ``--volume`` is unaffected -- it also adds ``raw (all)``,
    which starts at 0 and anchors the union on an integer -- but it is only luck that
    it does, and the paint box is the first layer to be alone on the mask grid.

    Only axis 0 needs it. The displayed axes are continuous -- the camera does not
    step -- so a half-voxel start there is invisible and, more to the point, right.
    """
    ranges = list(ranges)
    if not ranges:
        return tuple(ranges)
    z = ranges[0]
    ranges[0] = type(z)(float(np.floor(z.start)), float(np.ceil(z.stop)), 1.0)
    return tuple(ranges)


def _show_slice(viewer, z: int) -> int:
    """Put the dims slider on raw slice ``z``, and prove that it landed there.

    ``Dims`` silently clips ``point`` into ``range``, so asking for a slice outside the
    range napari currently believes in is not an error -- it just shows a different
    slice. Reading the value back turns that into something visible.
    """
    _force_extents(viewer)
    viewer.dims.set_point(0, z)
    got = int(round(viewer.dims.point[0]))
    if got != int(z):
        print(f"[viewer] warning: asked for slice {z}, dims clamped to {got}")
    return got


def _add_volume_layers(viewer, volume, contrast) -> None:
    """Add the lazy whole-dataset layers.

    Hidden by default: they exist so the z slider can leave the slab and so a
    defect can be followed across the volume, not to change what a pick shows.
    Turn them on in napari's layer list.

    They are never updated. The data is a dask graph over the whole dataset, so
    there is nothing per-pick to refresh -- which is also why ``_update_layers``
    does not mention them.
    """
    import napari

    if volume.raw is not None:
        viewer.add_image(
            volume.raw,
            name="raw (all)",
            colormap="gray",
            contrast_limits=tuple(float(v) for v in contrast),
            visible=False,
        )
    if volume.seg is not None:
        viewer.add_labels(
            volume.seg,
            name="segmentation (all)",
            scale=volume.seg_scale,
            translate=volume.seg_translate,
            opacity=0.35,
            visible=False,
            colormap=napari.utils.DirectLabelColormap(
                color_dict={None: "transparent", 1: SEG_ALL_COLOR[1]}
            ),
        )


def _reframe(viewer, slab: Slab) -> None:
    """Point the camera at the new pick.

    For an ROI crop, framing the layers is right -- the crop *is* the region of
    interest. For a full slice it is not: ``reset_view`` would zoom out to all
    ~9.7 M pixels on every pick, so you would land looking at the whole heart and
    have to zoom back in each time. Keep the zoom and recentre instead.
    """
    if not slab.full_slice:
        viewer.reset_view()
        return
    _, row, col = slab.pick_zrc
    camera = viewer.camera
    centre = list(camera.center)
    # napari's camera centre is (z, y, x) in world coordinates; the displayed
    # dims are the last two, which are row and column here.
    centre[-2:] = [float(row), float(col)]
    camera.center = tuple(centre)


def _contrast_limits(raw: np.ndarray) -> tuple[float, float]:
    """The 1st-99.5th percentile window for the raw layer, from a sub-sample.

    The same window ``viewer3d.slice_texture_array`` uses, so the 3D image plane
    and this layer show one contrast -- and computed the same cheap way it does,
    on a single slice rather than the whole slab.

    Sub-sampling is not an optimisation here, it is what makes a full-slice slab
    usable: over 11 full slices ``np.percentile`` is 107 M elements, which costs
    about a second and a 214 MB copy *per pick*. One strided slice is ~60 k
    elements and gives the same window to well within a display level.
    """
    a = np.asarray(raw)
    if a.size == 0:
        return 0.0, 1.0
    sample = a[a.shape[0] // 2, ::4, ::4] if a.ndim == 3 else a
    lo, hi = np.percentile(sample, [1.0, 99.5])
    if hi <= lo:
        hi = lo + 1.0
    return float(lo), float(hi)


def _expected_names(slab: Slab, volume=None, paint=None) -> set:
    """The layer names this slab should produce.

    Compared against the live viewer to decide update-in-place vs rebuild, so it
    has to account for the whole-volume layers too: if it does not, every pick
    sees a name mismatch, takes the ``viewer.layers.clear()`` branch, and kills
    the process on the next paint (see ``_update_layers``).
    """
    names = {
        "raw", "surface (STL)", "assumed cross-section", "perimeter circle", "skeleton",
    }
    if slab.prob is not None:
        names.add("probability")
    if slab.seg is not None:
        names.add("segmentation")
    if slab.tube is not None:
        names.add("reconstructed lumen (r)")
    if volume is not None:
        names.update(volume.names)
    if paint is not None:
        from .edit.paint import PAINT_LAYER

        names.add(PAINT_LAYER)
    return names


def _create_layers(viewer, slab: Slab, volume=None, paint=None):
    """First population of a viewer. Every layer is created even when empty, so that
    later picks can update data in place rather than tearing visuals down."""
    import napari

    tr = slab.translate
    lo, hi = _contrast_limits(slab.raw)
    if volume is not None:
        # Added first so they sit *under* the slab layers: the slab is the detail
        # view and should win wherever the two overlap.
        _add_volume_layers(viewer, volume, (lo, hi))
    viewer.add_image(
        slab.raw,
        name="raw",
        colormap="gray",
        translate=tr,
        contrast_limits=(float(lo), float(hi)),
    )

    if slab.prob is not None:
        viewer.add_image(
            slab.prob,
            name="probability",
            colormap="magma",
            blending="additive",
            translate=tr,
            visible=False,
            opacity=0.6,
        )

    if slab.seg is not None:
        viewer.add_labels(
            slab.seg,
            name="segmentation",
            translate=tr,
            opacity=0.35,
            # Hidden when painting is on. This layer is *derived* -- the mask
            # gathered onto the raw grid for display -- and the editable layer
            # below is the same mask on its own grid. Showing both means two
            # colours over identical anatomy, and only one of them can be changed.
            visible=paint is None,
            colormap=napari.utils.DirectLabelColormap(
                color_dict={None: "transparent", 1: SEG_COLOR[1]}
            ),
        )

    if paint is not None:
        paint.attach(viewer, slab.centre_um)

    if slab.tube is not None:
        # On by default: comparing this against the greyscale is the fastest way to see
        # a collapsed lumen that the circular cross-section has re-inflated.
        viewer.add_labels(
            slab.tube,
            name="reconstructed lumen (r)",
            translate=tr,
            opacity=0.30,
            colormap=napari.utils.DirectLabelColormap(
                color_dict={None: "transparent", 1: TUBE_COLOR[1]}
            ),
        )

    stl = viewer.add_shapes(name="surface (STL)", ndim=3, edge_color=STL_COLOR, opacity=0.9)
    if slab.contours:
        stl.add_paths(slab.contours, edge_color=STL_COLOR, edge_width=1.5)

    # Off by default: informative on a single vessel, but cluttered wherever several
    # run through the ROI. The rasterised lumen layer says the same thing more legibly.
    circ = viewer.add_shapes(
        name="assumed cross-section", ndim=3, edge_color=CIRCLE_COLOR, opacity=0.9
    )
    if slab.circles:
        circ.add_ellipses(
            slab.circles, edge_color=CIRCLE_COLOR, face_color="transparent", edge_width=1.5
        )
    circ.visible = False

    # What the pipeline's own rule (perimeter / 2pi) would draw from this slice. Where
    # it agrees with "assumed cross-section", re-inflation is behaving; where it does
    # not, the assigned radius has drifted from what the image supports.
    per = viewer.add_shapes(
        name="perimeter circle", ndim=3, edge_color=PERIM_COLOR, opacity=0.9
    )
    if slab.perim_circles:
        per.add_ellipses(
            slab.perim_circles,
            edge_color=PERIM_COLOR,
            face_color="transparent",
            edge_width=1.5,
        )
    per.visible = False

    pts = slab.skel_pts if slab.skel_pts is not None and len(slab.skel_pts) else np.empty((0, 3))
    viewer.add_points(
        pts,
        name="skeleton",
        face_color=SKEL_COLOR,
        border_color="black",
        size=7,
        opacity=0.95,
        ndim=3,
    )


def _update_layers(viewer, slab: Slab, paint=None):
    """Refresh an existing viewer in place.

    Clearing the layer list and re-adding crashes vispy: the visuals are destroyed
    while a draw is still pending, and the next paint dereferences freed GPU state
    (an access violation inside glDrawArrays). Assigning to ``.data`` keeps the same
    visuals alive and simply re-uploads.
    """
    tr = slab.translate
    layers = viewer.layers

    # First, before anything overwrites an array: the paint layer is the one thing
    # here holding state the user created rather than state derived from the pick.
    if paint is not None:
        paint.attach(viewer, slab.centre_um)

    raw = layers["raw"]
    raw.data = slab.raw
    raw.translate = tr
    raw.contrast_limits = _contrast_limits(slab.raw)

    for name, data in (
        ("probability", slab.prob),
        ("segmentation", slab.seg),
        ("reconstructed lumen (r)", slab.tube),
    ):
        if name in layers and data is not None:
            layers[name].data = data
            layers[name].translate = tr

    _replace_shapes(
        layers["surface (STL)"], slab.contours, "add_paths",
        edge_color=STL_COLOR, edge_width=1.5,
    )
    _replace_shapes(
        layers["assumed cross-section"], slab.circles, "add_ellipses",
        edge_color=CIRCLE_COLOR, face_color="transparent", edge_width=1.5,
    )
    _replace_shapes(
        layers["perimeter circle"], slab.perim_circles, "add_ellipses",
        edge_color=PERIM_COLOR, face_color="transparent", edge_width=1.5,
    )

    layers["skeleton"].data = (
        slab.skel_pts if slab.skel_pts is not None and len(slab.skel_pts) else np.empty((0, 3))
    )


def _replace_shapes(layer, shapes, adder: str, **kw):
    """Swap a Shapes layer's contents and make sure its extent follows.

    A hidden layer skips its extent recompute (see ``_force_extents``), and these two
    overlays are hidden by default, so the mutation has to be followed by a forced
    refresh or the layer keeps reporting the previous slab's bounds.
    """
    layer.data = []
    if shapes:
        getattr(layer, adder)(shapes, **kw)
    layer.refresh(
        thumbnail=False, data_displayed=False, highlight=False, extent=True, force=True
    )


def _info_html(slab: Slab) -> str:
    x, y, z = slab.centre_um
    return (
        f"<b>Picked</b><br>"
        f"{x:,.0f}, {y:,.0f}, {z:,.0f} &micro;m<br>"
        f"slice <b>{slab.z_centre}</b> "
        f"(showing {slab.z_lo}&ndash;{slab.z_hi - 1})<br>"
        f"row {slab.row0}&ndash;{slab.row1}, col {slab.col0}&ndash;{slab.col1}<br><br>"
        f"<b>Overlays</b><br>"
        f"<span style='color:{SEG_COLOR[1]}'>&#9632;</span> segmentation<br>"
        f"<span style='color:{STL_COLOR}'>&#9632;</span> surface cross-section<br>"
        f"<span style='color:{SKEL_COLOR}'>&#9632;</span> skeleton centreline<br>"
        f"<span style='color:{CIRCLE_COLOR}'>&#9632;</span> assumed cross-section<br>"
        f"<span style='color:{PERIM_COLOR}'>&#9632;</span> perimeter circle (P/2&pi;)<br>"
        f"<span style='color:{TUBE_COLOR[1]}'>&#9632;</span> reconstructed lumen<br><br>"
        f"<i>Re-inflation is by design: a collapsed<br>"
        f"lumen should give a bigger, rounder<br>"
        f"section. Compare green vs orange &mdash;<br>"
        f"they should agree.</i><br><br>"
        f"Toggle each with its checkbox;<br>the opacity slider applies to<br>"
        f"the selected layer."
    )


def _viewer_state(viewer) -> dict:
    """Per-viewer bookkeeping, hung off the Qt window.

    napari's ``Viewer`` is a pydantic model and rejects arbitrary attributes, so this
    cannot live on the viewer itself. It must not be keyed on ``id(viewer)`` either:
    CPython reuses ids, so once the user closes the slice window and picks again, a
    fresh viewer can inherit the dead one's entry -- ``overlay_hooked`` already True, so
    the slice-number overlay never reconnects to the slider, and ``info_label`` pointing
    at a QLabel whose C++ half is gone, so ``setText`` raises. The Qt window has exactly
    the lifetime this state should have.
    """
    win = viewer.window._qt_window
    state = getattr(win, "_hipct_state", None)
    if state is None:
        state = {}
        win._hipct_state = state
    return state


def _attach_info(viewer, slab: Slab, paint=None):
    """Add the info panel once, then just refresh its text on later picks."""
    from qtpy.QtWidgets import QLabel, QPushButton, QVBoxLayout, QWidget

    state = _viewer_state(viewer)
    state["slab"] = slab
    state["paint"] = paint
    _attach_paint_panel(viewer, paint, state)
    label = state.get("info_label")
    if label is not None:
        label.setText(_info_html(slab))
        return

    label = QLabel(_info_html(slab))
    box = QWidget()
    lay = QVBoxLayout(box)
    lay.addWidget(label)
    shot = QPushButton("Save screenshot")

    def _save():
        s = _viewer_state(viewer)["slab"]
        # Named after the pick, not the window: at full slice every window starts
        # at row 0, col 0, so those would name every screenshot identically.
        _, prow, pcol = s.pick_zrc
        out = f"pick_z{s.z_centre}_r{prow}_c{pcol}.png"
        viewer.screenshot(out, canvas_only=True)
        print(f"[viewer] screenshot -> {out}")

    shot.clicked.connect(_save)
    lay.addWidget(shot)
    lay.addStretch(1)

    state["info_label"] = label
    viewer.window.add_dock_widget(box, area="right", name="pick")


def _attach_paint_panel(viewer, paint, state) -> None:
    """Dock the brush controls once, and let them catch up on a late wiring.

    The re-skeletonise callback is only attached when ``--edit`` is on, and that
    happens after the first pick has already built this panel, so the button has to
    be re-enabled rather than fixed at construction.
    """
    if paint is None:
        return
    from .edit import paint as paint_mod

    panel = state.get("paint_panel")
    if panel is None:
        panel = paint_mod.build_paint_panel(paint)
        state["paint_panel"] = panel
        viewer.window.add_dock_widget(panel, area="right", name="paint")
    paint_mod.refresh_panel(panel, paint)
