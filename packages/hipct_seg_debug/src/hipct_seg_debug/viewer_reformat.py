"""The napari window for a reformatted stack.

Shaped like :mod:`~.viewer2d` -- ``show`` opens one, ``build_layers`` (re)populates a
viewer already in use, and layers are updated in place rather than cleared and re-added,
because clearing the layer list and re-adding crashes vispy. But it is deliberately a
*separate* window from the slice browser, for a reason that is not stylistic: the arrays
here live in ``(plane index, v, u)``, and everything in ``viewer2d`` is built around raw
``(slice, row, col)``. In particular ``viewer2d._whole_slices`` pins axis 0 to whole raw
slice numbers, which is exactly right there and meaningless here.

**Axis 0 stays an integer plane index.** napari's slider steps in integers, so giving
axis 0 a physical scale would put the readout half a step off the plane actually being
shown -- the failure ``viewer2d._whole_slices`` documents at length. Arclength is
reported through the text overlay from the stack's own ``arclen_um`` instead, which is
also the more honest number: the resample pins both endpoints, so the step is only
nominally uniform.

**The two in-plane axes are scaled, when they can be honestly scaled.** In ``fixed`` and
``manual`` modes every plane shares one sample pitch, so the layers carry a real
``scale`` and ``translate`` and napari's cursor readout is micrometres from the vessel
centre -- which is what makes this a measuring instrument rather than a picture. In
``radius`` mode the pitch changes from plane to plane and napari's ``scale`` is a single
constant per axis, so there is no honest value to give it: the axes stay in pixels and
the per-plane pitch goes in the overlay. Quietly applying one plane's scale to the whole
stack would be a lie of exactly the kind the rest of this package works to avoid.
"""

from __future__ import annotations

import numpy as np

from .viewer2d import CIRCLE_COLOR, SEG_COLOR, SKEL_COLOR, _contrast_limits, _viewer_state

RAW_LAYER = "reformat (raw)"
MASK_LAYER = "reformat (mask)"
GRAPH_LAYER = "graph points"
CENTRE_LAYER = "centreline"

TITLE = "HiP-CT segmentation debugger - reformat"


def show(reformat, *, title: str = TITLE, block: bool = True, viewer=None):
    """Open (or reuse) a viewer on this stack."""
    import napari

    if viewer is None:
        viewer = napari.Viewer(title=title)
    build_layers(viewer, reformat)
    if block:
        napari.run()
    return viewer


def _expected_names(reformat) -> set:
    names = {RAW_LAYER, CENTRE_LAYER}
    if reformat.mask is not None:
        names.add(MASK_LAYER)
    if len(reformat.graph_points):
        names.add(GRAPH_LAYER)
    return names


def build_layers(viewer, reformat, on_plane=None):
    """(Re)populate a viewer with this stack. Safe on a viewer already in use.

    ``on_plane(i)`` is called with the plane index whenever the slider moves, and once
    here. The 3D window uses it to keep its image plane on the slice the current
    cross-section was cut from.
    """
    reused = _expected_names(reformat) == {layer.name for layer in viewer.layers}
    if reused:
        _update_layers(viewer, reformat)
    else:
        viewer.layers.clear()
        _create_layers(viewer, reformat)

    _show_plane(viewer, len(reformat.raw) // 2)

    viewer.text_overlay.visible = True
    viewer.text_overlay.font_size = 11
    viewer.text_overlay.color = "white"

    state = _viewer_state(viewer)
    # Dispatched through the state rather than captured, so a rebuild replaces the
    # callback instead of stacking one connection per build.
    state["on_plane"] = on_plane
    state["reformat"] = reformat

    def _on_step(_event=None):
        current = _viewer_state(viewer).get("reformat")
        if current is None:
            return
        i = int(np.clip(round(viewer.dims.point[0]), 0, len(current.raw) - 1))
        viewer.text_overlay.text = readout(current, i)
        cb = _viewer_state(viewer).get("on_plane")
        if cb is not None:
            cb(i)

    _on_step()
    if not state.get("reformat_hooked"):
        viewer.dims.events.current_step.connect(_on_step)
        state["reformat_hooked"] = True
    return viewer


def readout(reformat, i: int) -> str:
    """The overlay line: where this plane is, how big it is, and how bent."""
    line = reformat.centreline
    geom = reformat.geometry
    x, y, z = line.coords_um[i]
    r_curv = line.curvature.r_point_um[i]
    bend = "straight" if not np.isfinite(r_curv) else f"R_curv {r_curv:,.0f} um"
    clamp = ""
    if geom.requested_um[i] > geom.half_um[i] + 1e-6:
        clamp = f" (clamped from {geom.requested_um[i]:,.0f})"

    # The oversampling factor is what explains a soft-looking section: above 1 the grid
    # is asking for detail finer than the acquisition, so what is on screen is largely
    # the interpolation kernel rather than the data.
    scale = f"{geom.px_um[i]:.1f} um/px"
    if geom.voxel_um > 0:
        over = geom.oversampling[i]
        scale += f" = {over:.1f}x the {geom.voxel_um:.1f} um voxel"
        if over > 1.5:
            scale += " (magnified)"
    return (
        f"plane {i + 1} / {len(reformat.raw)}    s = {line.arclen_um[i] / 1000:.2f} mm"
        f"    segment {int(line.seg_ids[i])}\n"
        f"centre  x {x:,.0f}  y {y:,.0f}  z {z:,.0f} um\n"
        f"r_graph {line.radii_um[i]:,.0f} um    half {geom.half_um[i]:,.0f} um{clamp}"
        f"    {scale}    {bend}"
    )


def placement(reformat):
    """``(scale, translate)`` for the image layers, or pixels where um would lie.

    See the module docstring: a per-plane pitch cannot be expressed in napari's single
    constant ``scale``, so ``radius`` mode stays in pixels rather than pretending.
    """
    geom = reformat.geometry
    if geom.mode == "radius":
        return (1.0, 1.0, 1.0), (0.0, 0.0, 0.0)
    px = float(geom.px_um[0])
    half = float(geom.half_um[0])
    return (1.0, px, px), (0.0, -half, -half)


def _create_layers(viewer, reformat) -> None:
    import napari

    scale, translate = placement(reformat)
    lo, hi = _contrast_limits(reformat.raw)
    viewer.add_image(
        reformat.raw,
        name=RAW_LAYER,
        colormap="gray",
        contrast_limits=(float(lo), float(hi)),
        scale=scale,
        translate=translate,
    )
    if reformat.mask is not None:
        viewer.add_labels(
            reformat.mask.astype(np.uint8),
            name=MASK_LAYER,
            opacity=0.35,
            scale=scale,
            translate=translate,
            colormap=napari.utils.DirectLabelColormap(
                color_dict={None: "transparent", **SEG_COLOR}
            ),
        )

    # The chain's own point is always the exact centre pixel, by construction. Drawn
    # anyway: it is the fixed mark the eye needs to see the lumen drift off it, which
    # is the first sign of a tangent that is not quite right.
    centre = reformat.geometry.size_px // 2
    n = len(reformat.raw)
    centres = np.column_stack([np.arange(n, dtype=np.float64),
                               np.full(n, centre, dtype=np.float64),
                               np.full(n, centre, dtype=np.float64)])
    viewer.add_points(
        _to_world(centres, scale, translate),
        name=CENTRE_LAYER, ndim=3, size=5, face_color=SKEL_COLOR, border_width=0.0,
    )
    if len(reformat.graph_points):
        viewer.add_points(
            _to_world(reformat.graph_points, scale, translate),
            name=GRAPH_LAYER, ndim=3, size=4, face_color=CIRCLE_COLOR,
            border_width=0.0,
        )


def _to_world(points_px, scale, translate) -> np.ndarray:
    """Points layers carry no scale of their own, so place them by hand.

    Giving the points layer its own ``scale`` would work too, but then the two
    placements could drift apart; one conversion at the point of use cannot.
    """
    pts = np.asarray(points_px, dtype=np.float64).reshape(-1, 3)
    return pts * np.asarray(scale) + np.asarray(translate)


def _update_layers(viewer, reformat) -> None:
    """Reassign data in place. Clearing and re-adding crashes vispy."""
    scale, translate = placement(reformat)
    layers = {layer.name: layer for layer in viewer.layers}

    raw = layers[RAW_LAYER]
    raw.data = reformat.raw
    raw.scale, raw.translate = scale, translate
    raw.contrast_limits = tuple(float(v) for v in _contrast_limits(reformat.raw))

    if reformat.mask is not None and MASK_LAYER in layers:
        mask = layers[MASK_LAYER]
        mask.data = reformat.mask.astype(np.uint8)
        mask.scale, mask.translate = scale, translate

    centre = reformat.geometry.size_px // 2
    n = len(reformat.raw)
    centres = np.column_stack([np.arange(n, dtype=np.float64),
                               np.full(n, centre, dtype=np.float64),
                               np.full(n, centre, dtype=np.float64)])
    layers[CENTRE_LAYER].data = _to_world(centres, scale, translate)
    if GRAPH_LAYER in layers:
        layers[GRAPH_LAYER].data = _to_world(reformat.graph_points, scale, translate)


def _show_plane(viewer, i: int) -> int:
    """Put the slider on plane ``i``, and prove that it landed there.

    ``Dims`` silently clips ``point`` into ``range``, so asking for a plane outside the
    range napari currently believes in is not an error -- it just shows a different one.
    Reading the value back turns that into something visible.
    """
    for layer in viewer.layers:
        layer.refresh(
            thumbnail=False, data_displayed=False, highlight=False, extent=True, force=True
        )
    viewer.dims.range = tuple(viewer.layers._ranges)
    viewer.dims.set_point(0, i)
    got = int(round(viewer.dims.point[0]))
    if got != int(i):
        print(f"[reformat] warning: asked for plane {i}, dims clamped to {got}")
    return got
