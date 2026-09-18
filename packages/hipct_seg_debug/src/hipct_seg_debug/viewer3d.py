"""PyVista window for choosing where to look.

Shows the reconstructed surface, the skeleton centreline and any flagged candidate
sites in micrometre world space. Double-click puts a marker down; ``v`` opens the slice
browser on it. Selection is a double-click precisely so that a single click-and-drag
stays a camera gesture and cannot leave a pick behind wherever the drag began.

Two overlays put the evidence next to the model. ``i`` drops the raw slice through the
pick into the scene as a translucent plane, and it follows the slice browser's z slider;
``g`` builds an isosurface of the Amira mask in a box around the pick. Both are off by
default, and neither is pickable.

The three per-slice shapes the browser draws -- the STL cross-section and the
assumed/perimeter pair -- are mirrored into the scene by ``set_slice_shapes``, taken from
the slab rather than recomputed so the two windows cannot disagree.

Every layer's visibility and opacity is adjustable from the docked panel that
``controls3d`` builds. It drives the accessors at the end of ``Picker3D`` rather than
touching actors, because those two overlays are rebuilt on each pick: an opacity written
only to an actor would be lost the moment you picked again.

The graph-wide ``radius circles`` layer draws the ideal circular cross-section implied
by every stored point radius, perpendicular to that point's edge-local tangent. It is
off by default and built on first use because a full graph contains tens of thousands
of rings.

The centreline and those rings share **one** colour mapping, chosen from the layer
panel: the stored radius, or the Strahler order of each point's owning edge on a graph
that carries one. They share it because a ring is the cross-section *of* the centreline
it sits on, and two different maps in the same scene would be read as one. Strahler is
drawn in discrete bands with one annotation per order rather than as a ramp -- it counts
branching generations, so a continuous bar would invite reading a 2.5 off it.

``save_figure`` writes the scene to SVG for publication. It divides the text over the
render in two: the keybinding block and the pick readout drive the window and are left
out, while the legend box and the colour bar are the key to what is drawn and are kept
as they stand -- a tree banded by Strahler order is unreadable without the bar that
names the bands. Only the labels are vector; the geometry is an embedded raster image,
which is what GL2PS's OpenGL2 backend writes whatever it is asked for.

**A pick is always a spatial-graph point.** Only the centreline and the candidate
clouds are pickable, and the picker is a ``vtkPointPicker``, which snaps to dataset
vertices and reports the vertex id. ``centreline_polydata`` keeps ``graph.points`` in
order, so that id *is* the graph point index and the pick carries its radius and owning
edge with it. Everything else in the scene is deliberately unpickable:

* the reconstructed surface is drawn translucent, so a ray aimed at a vessel you can see
  *through* the shell would otherwise stop on the near shell, millimetres away;
* the pick marker itself would otherwise intercept the *next* click in the same region
  and walk the pick towards the camera by its own radius, one click at a time.

The plotter is a ``pyvistaqt.BackgroundPlotter`` so that it shares napari's Qt event
loop: the 3D view **stays open** while you inspect slices, keeps its camera, and can
be picked from again immediately. A blocking ``pv.Plotter`` would have to be closed
before napari could run, and creating napari after it tore down its VTK interactor is
a good way to crash the session.
"""

from __future__ import annotations

import traceback
from pathlib import Path

import numpy as np
import pyvista as pv

from .viewer2d import CIRCLE_COLOR, PERIM_COLOR, STL_COLOR

SURFACE_COLOR = "#d9534f"
CANDIDATE_COLOR = {
    # graph / topology
    "premature_end": "#ff9500",
    "murray_deficit": "#0a84ff",
    "endpoint_gap": "#ffd400",
    "parallel_pair": "#00e676",
    # image, measured against the pipeline's own perimeter rule
    "perimeter_mismatch": "#ff2d55",
    "companion_lumen": "#af52de",
    # informational: re-inflation of a collapsed lumen is by design
    "collapse_severity": "#ff7ab6",
}

INSTRUCTIONS = (
    "double-click  pick the nearest centreline point\n"
    "drag          rotate - never picks\n"
    "v             open / update the slice viewer\n"
    "i             raw image plane at the pick\n"
    "g             segmentation mask around the pick\n"
    "a             segmentation mask, whole tree\n"
    "n / b         next / previous candidate\n"
    "m / l         add pick to main vessel / next vessel name\n"
    "o / h         crop: drop the pick / prune past it\n"
    "s             allow / forbid picking the surface\n"
    "L (shift+l)   show / hide the legend box\n"
    "c             clear the pick\n"
    "r             reset the camera\n"
    "q             close this window"
)

# Pick radius as a fraction of the render window diagonal: about 10 px on a 1200 px
# window. Wide enough to hit a thin vessel without a steady hand, narrow enough that a
# click on empty tissue reports nothing rather than snapping to a vessel you cannot see.
PICK_TOLERANCE = 0.008

#: Field names Avizo has used for the Strahler order, most common first. The order is
#: stored under whatever the file called it -- ``edit.adapter.STRAHLER_ALIASES`` is the
#: same list for the same reason -- so the viewer has to look under all of them.
STRAHLER_ALIASES = ("strahler", "StrahlerOrder", "Strahler", "StrahlerNumber")

#: What the centreline and the radius rings can be coloured by: key, panel label, and
#: the scalar-bar title, which doubles as the key the bar is removed under. Both layers
#: are coloured through `color_kwargs` so they cannot disagree -- one bar describes both.
COLOR_MODES = (
    ("radius", "radius (um)", "radius (um)"),
    ("strahler", "Strahler order", "Strahler order"),
)
COLOR_BAR_TITLES = tuple(title for _key, _label, title in COLOR_MODES)

#: What `save_figure` will write, which is what `Plotter.save_graphic` can write.
#: ``.svg`` is first because it is the default a path with no suffix at all gets.
VECTOR_SUFFIXES = (".svg", ".pdf", ".eps", ".ps", ".tex")

SEG_COLOR = "#00b0ff"  # the blue the slice browser uses for the segmentation
# Deliberately flat, desaturated, and absent from viridis. The centreline is coloured by
# radius, so any colour that *could* be a viridis value would read as a measurement --
# which is the one thing these points do not have. Grey reads as annotation.
INTERPOLATED_COLOR = "#9aa0a6"
#: Unsampled jumps. Brighter than the Hermite fills and deliberately alarming: a fill
#: is invented centreline through a hole, a jump is a straight line across up to 9.8 mm
#: of nothing at all.
JUMP_COLOR = "#ff453a"
# Reconnection review. The winner and its rivals are deliberately close in
# brightness: the point of drawing them together is that the operator judges them
# on their path through the image, not on which one the viewer made look important.
ROUTE_COLOR = "#ff375f"
ALTERNATIVE_COLOR = "#ff9f0a"
WAYPOINT_COLOR = "#32d74b"
#: The crop preview. Deliberately a *dulled* red rather than another alarm colour:
#: `JUMP_COLOR` means "this geometry was never measured" and can be on screen at the
#: same time, so the two must not read as the same claim. This one means "this is on
#: its way out", which is a decision the operator is making, not a fault in the data.
CROP_DROP_COLOR = "#b3392f"
#: The reformat selection. A cyan that none of the decision colours above use, because
#: it can be on screen beside a crop preview and the two are unrelated claims: this one
#: says "these are the segments about to be sampled", not "these are on their way out".
REFORMAT_COLOR = "#00d0ff"
#: Selected, but not part of the run that will actually be sampled -- a segment that is
#: not connected to the longest chain. Deliberately drawn *dimmer* rather than in
#: another alarm colour: it is not a fault, it is a selection that cannot all be one
#: stack, and the eye needs to see at a glance which half is going to be built.
REFORMAT_DROPPED_COLOR = "#4a6272"
#: The outline of the plane the reformat window is currently showing. Warmer than
#: `REFORMAT_COLOR` so it reads as "you are here" against the run it sits on.
REFORMAT_CURRENT_COLOR = "#ffd400"
#: The `radius-perimeter` debug sections (`edit.section_frames`). A violet family of
#: its own: these squares are drawn *over* a reformat's cyan frames when both are up,
#: and the two are different constructions of the same idea -- one is the stack a
#: reformat would sample, the other is the window a radius was actually measured in.
SECTION_FRAME_COLOR = "#8e8cff"
#: A window whose section did not close inside it, or that yielded nothing. Warm,
#: because unlike the other alarm colours here this one is a *measurement* failure:
#: the radius at this point was interpolated rather than measured.
SECTION_FAIL_COLOR = "#ff6b35"
#: The boundary `cv2.arcLength` measured. Drawn from the same contour the perimeter
#: came from, so what is on screen is the measurement rather than a second opinion.
SECTION_CONTOUR_COLOR = "#7ee787"
#: The boundary of a blob the pass **refused**. Same warm family as
#: `SECTION_FAIL_COLOR`, because it belongs to that failure, but lighter and yellower:
#: the ring is drawn *inside* the orange square it came from, and a same-value orange
#: on orange would not separate at a two-pixel line width. Its own colour rather than
#: the green one because a refused blob is not a measurement -- and its *shape* is the
#: diagnostic, which is why it is drawn at all. A streak is the wrong axis, a blob
#: filling its window is a window too small, a blob merged with its neighbour is a cut
#: that caught a second vessel.
SECTION_REFUSED_COLOR = "#ffb454"
#: Centreline point -> its own section's area centroid. The re-centring that
#: `radius-perimeter` does not do, drawn at the length it would have moved.
SECTION_OFFSET_COLOR = "#ff4fd8"

#: How many of a reformat's cross-sections get drawn as textured quads. Each one is its
#: own actor and its own texture upload, so a 2,700-plane run drawn in full would cost
#: more than the whole tree it is drawn over. Evenly spaced, so the number of them does
#: not depend on how long the run is.
REFORMAT_MAX_TEXTURED = 24
PLANE_OPACITY = 0.55
SURFACE_OPACITY = 0.35
SEG_BOX_UM = 2000.0  # half-extent of the segmentation box built around a pick
# Decimation of the whole-tree mask isosurface. 1 is full resolution: 3.75 M
# triangles, 6.5 s cold and 4.5 s once the mask is resident. The old default of 4
# (213 k triangles, 0.5 s) existed because the undecimated lattice was assumed
# unrenderable; it was never measured, and flying-edges contouring plus a machine
# with room for 2.34 GB makes it ordinary. The layer panel changes this live.
SEG_STRIDE = 1

# The per-slice shapes the slice browser draws, mirrored into 3D: key, label, colour.
# Colours come from viewer2d so the two windows agree about what red/green/orange mean.
SHAPE_LAYERS = (
    ("stl_contour", "STL contour", STL_COLOR),
    ("assumed", "assumed cross-section", CIRCLE_COLOR),
    ("perimeter", "perimeter circle", PERIM_COLOR),
)

# What the layer panel offers, in the order it is drawn:
#   key, label, default opacity, visible by default.
# `plane`, `segmentation` and the three shape layers are *built and torn down* rather
# than merely hidden, so everything goes through Picker3D's accessors rather than poking
# actors directly.
LAYERS = (
    ("surface", "surface (STL)", SURFACE_OPACITY, True),
    ("centreline", "centreline", 1.0, True),
    # On by default and drawn flat grey: the centreline is broken wherever Avizo invented
    # points, so with this off the tree shows a hole rather than a bridge, and a hole is
    # the more misleading of the two.
    ("interpolated", "interpolated (Avizo)", 1.0, True),
    # Built lazily: the full LADAF graph has ~37k points, which produces ~1.8M
    # ring vertices at the default resolution. Do not make every startup pay for it.
    ("radius_circles", "radius circles", 1.0, False),
    ("candidates", "candidates", 1.0, True),
    ("plane", "image plane", PLANE_OPACITY, False),
    ("segmentation", "segmentation", 0.5, False),
    ("segmentation_all", "segmentation (whole tree)", 0.35, False),
    # Defaults mirror the slice browser: the STL contour on, the diagnostic pair off
    # until asked for, because they clutter wherever several vessels cross the ROI.
    ("stl_contour", "STL contour", 1.0, True),
    ("assumed", "assumed cross-section", 1.0, False),
    ("perimeter", "perimeter circle", 1.0, False),
    # Owned by the `edit` subpackage rather than by this module: it registers its
    # actors through `set_extra_actors` and they are absent (so the panel greys
    # the row out) in a plain read-only session.
    ("edit_surface", "edited surface", 1.0, True),
    ("edit_handles", "edit handles", 1.0, True),
    ("reconnect", "reconnect candidates", 1.0, True),
    # Owned by the Crop tab. Both are previews of a decision, not of the data, so
    # they are drawn over the centreline rather than replacing it: the operator is
    # judging what a rule *would* take, against the tree it would take it from.
    ("crop_preview", "crop: to be dropped", 0.9, True),
    ("crop_vessels", "crop: main vessels", 1.0, True),
    # Owned by the Reformat tab. Its own layer rather than a second caller of
    # `show_crop`: that method clears every actor it has drawn, so two panels sharing
    # it would silently erase each other's preview.
    ("reformat_path", "reformat: selected run", 1.0, True),
    # The sampling geometry itself. Frames are on by default because they are what
    # makes plane collision *visible* -- a stack whose squares fan through each other
    # on the inside of a bend is obvious here and impossible to see in the 2D window.
    ("reformat_frames", "reformat: plane frames", 0.9, True),
    ("reformat_slice", "reformat: current section", 1.0, True),
    # Off by default: a couple of dozen textured quads is the one thing here that
    # costs anything, and it is a "show me what I built" gesture rather than a
    # working overlay.
    ("reformat_planes", "reformat: section stack", 1.0, False),
    # Owned by the Sections tab: the planes `radius-perimeter` cuts, at sampled
    # points. Four layers because they answer four separate questions -- where and how
    # big the window was, what was measured in it, what was *refused* in it, and how
    # far off its own section's centre the point sat. The two boundary rows sit
    # adjacent deliberately: they are the same geometry sorted by verdict, and a
    # reader toggling between them wants them next to each other.
    ("section_frames", "sections: plane windows", 1.0, True),
    ("section_contours", "sections: measured lumen", 1.0, True),
    ("section_refused", "sections: refused blob", 1.0, True),
    ("section_offsets", "sections: centroid offset", 1.0, True),
)
SHAPE_KEYS = tuple(key for key, _, _ in SHAPE_LAYERS)

# `set_dataset` has to distinguish "leave this input alone" from "set it to None",
# and None is a meaningful value for every one of them.
_KEEP = object()

# Layers this module does not build itself. Keeping them in `LAYERS` means the
# edit tools get the same panel row, key handling and opacity memory as anything
# else, without `viewer3d` needing to know what they are.
EXTRA_KEYS = ("edit_surface", "edit_handles", "reconnect",
              "crop_preview", "crop_vessels", "reformat_path",
              "reformat_frames", "reformat_slice", "reformat_planes",
              "section_frames", "section_contours", "section_refused",
              "section_offsets")


def slice_texture_array(img: np.ndarray) -> np.ndarray:
    """Turn a raw greyscale slice into an (M, N, 3) uint8 texture array.

    Two things happen here, both load-bearing:

    * the same robust window the slice browser uses for its ``raw`` layer
      (1st to 99.5th percentile), so the 3D plane and the 2D view show one contrast;
    * **a row flip.** ``pv.Texture`` places array row 0 at the *high* end of the quad's
      second texture axis, but this frame puts row 0 at ``y = 0`` (``y_um = row * vy``).
      Handing the slice over unflipped mirrors the anatomy top-to-bottom, which looks
      entirely plausible and is wrong. ``selftest.test_slice_plane_geometry`` guards it.
    """
    a = np.asarray(img)
    lo, hi = np.percentile(a, [1.0, 99.5])
    if hi <= lo:
        hi = lo + 1.0
    g = np.clip((a.astype(np.float32) - lo) / (hi - lo), 0.0, 1.0)
    g = (g * 255.0).astype(np.uint8)[::-1]
    return np.repeat(g[:, :, None], 3, axis=2)


def slice_plane_quad(frame, stack_shape, z: int) -> pv.PolyData:
    """The raw slice's footprint in world micrometres, as a two-triangle textured quad.

    A full slice is ~9.7 M pixels; as a ``pv.ImageData`` that is 9.7 M rendered cells and
    the view stops being interactive. As a texture it is two triangles and one upload.
    """
    _, n_rows, n_cols = stack_shape
    vx, vy, vz = frame.raw_voxel
    x1, y1 = (n_cols - 1) * vx, (n_rows - 1) * vy
    zw = float(z) * vz
    quad = pv.PolyData(
        np.array([[0.0, 0.0, zw], [x1, 0.0, zw], [x1, y1, zw], [0.0, y1, zw]]),
        faces=np.array([4, 0, 1, 2, 3]),
    )
    quad.active_texture_coordinates = np.array(
        [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]
    )
    return quad


def corner_quad(corners) -> pv.PolyData:
    """A ``(4, 3)`` set of world-um corners as a textured two-triangle quad.

    The oblique sibling of :func:`slice_plane_quad`: same texture-coordinate
    convention, but the corners are given rather than derived from an axial slice
    index, because a reformat's planes are perpendicular to the vessel rather than to
    ``z``. ``reformat.plane_corners`` emits them in this order.
    """
    quad = pv.PolyData(
        np.asarray(corners, dtype=np.float64).reshape(4, 3),
        faces=np.array([4, 0, 1, 2, 3]),
    )
    quad.active_texture_coordinates = np.array(
        [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]
    )
    return quad


def corner_outlines(corners) -> pv.PolyData:
    """``(N, 4, 3)`` corner sets as one PolyData of closed square rings.

    One actor for the whole stack, however many planes it has: the frames are there to
    show where the sampling went, and N actors would cost more than the answer.
    """
    rings = [np.asarray(c, dtype=np.float64).reshape(4, 3) for c in corners]
    return polylines_polydata([np.vstack([r, r[:1]]) for r in rings])


def ellipse_from_corners(corners, n: int = 48) -> np.ndarray:
    """Turn a napari 4-corner ellipse into an (n, 3) ring in the same coordinates.

    ``viewer2d`` stores both shape overlays the way a napari Shapes layer wants them --
    the four corners of the parallelogram the ellipse is inscribed in
    (``_radius_ellipses`` builds them as ``c -a-b, c +a-b, c +a+b, c -a+b``). PyVista has
    no such notion, so recover the centre and the two semi-axes and walk the ring.
    """
    c = np.asarray(corners, dtype=np.float64)
    centre = c.mean(axis=0)
    semi_a = (c[1] - c[0]) / 2.0
    semi_b = (c[3] - c[0]) / 2.0
    t = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    return centre + np.outer(np.cos(t), semi_a) + np.outer(np.sin(t), semi_b)


def polylines_polydata(polylines) -> pv.PolyData:
    """One PolyData holding several (N, 3) polylines, closed rings included."""
    pts, cells, off = [], [], 0
    for pl in polylines:
        a = np.asarray(pl, dtype=np.float64)
        if len(a) < 2:
            continue
        pts.append(a)
        cells.append(np.concatenate([[len(a)], np.arange(off, off + len(a))]))
        off += len(a)
    if not pts:
        return pv.PolyData()
    poly = pv.PolyData(np.vstack(pts).astype(np.float32))
    poly.lines = np.concatenate(cells)
    return poly


def _contour(grid):
    """Marching cubes over an ImageData, by the fast route.

    ``contour`` defaults to ``vtkContourFilter``, which is general and single
    threaded. On uniform grids ``flying_edges`` (``vtkFlyingEdges3D``) is threaded
    and produces a bit-identical mesh: measured on the full LADAF-2024-28 lattice,
    **19.3 s against 3.0 s** for the same 3,751,672 cells and the same bounds. It is
    the single reason full resolution is affordable at all.
    """
    return grid.contour([0.5], scalars="mask", method="flying_edges")


def segmentation_box(frame, labels, xyz_um, half_um: float, volume=None):
    """Isosurface of the binary mask in a box around ``xyz_um``, or ``None`` if empty.

    ``volume`` is the resident full-resolution mask when one exists
    (``edit.lattice.MaskVolume``), in which case the box is a slice out of it.
    Without one, the sixty-odd planes are decoded here -- about a millisecond each,
    which is why this was always affordable even before the mask could be resident.
    """
    i, j, k = (int(v) for v in frame.um_to_seg_index(xyz_um)[0])
    nx, ny, nz = (int(v) for v in frame.seg_dims)
    hi_, hj, hk = (max(int(round(half_um / s)), 1) for s in frame.seg_spacing)

    i0, i1 = max(i - hi_, 0), min(i + hi_ + 1, nx)
    j0, j1 = max(j - hj, 0), min(j + hj + 1, ny)
    k0, k1 = max(k - hk, 0), min(k + hk + 1, nz)
    if i0 >= i1 or j0 >= j1 or k0 >= k1:
        return None, None

    if volume is not None:
        vol = volume[k0:k1, j0:j1, i0:i1]
    else:
        vol = np.empty((k1 - k0, j1 - j0, i1 - i0), dtype=np.uint8)
        for n, z in enumerate(range(k0, k1)):
            vol[n] = labels.slice_z(z)[j0:j1, i0:i1]
    grid = pv.ImageData(
        dimensions=(i1 - i0, j1 - j0, k1 - k0),
        spacing=tuple(float(s) for s in frame.seg_spacing),
        origin=tuple(float(v) for v in frame.seg_to_um([[i0, j0, k0]])[0]),
    )
    # vol is [z, y, x]; a C-order ravel is x-fastest, which is ImageData's point order.
    grid.point_data["mask"] = np.ascontiguousarray(vol).ravel()
    if not vol.any():
        return grid, None
    return grid, _contour(grid)


def segmentation_volume(frame, labels, stride: int = SEG_STRIDE, volume=None):
    """Isosurface of the *entire* mask, at ``stride`` on every axis.

    ``segmentation_box`` answers "is the mask right here?"; this answers "what
    does the whole segmentation look like?" -- which the box cannot, because the
    tree spans ~50 mm and the box is a couple of millimetres.

    Measured on LADAF-2024-28, decode + contour + adding it to the scene:

    ======  ========  ==========  ===========
    stride  resident  cold        triangles
    ======  ========  ==========  ===========
    4       0.1 s     0.5 s         213,000
    2       0.7 s     1.6 s         923,872
    1       4.5 s     6.5 s       3,751,672
    ======  ========  ==========  ===========

    So the stride now trades triangle count against contour time, not against
    memory: 2.34 GB is nothing on the machine this runs on, and the claim that
    stride 1 was "unrenderable" was never measured. Full resolution is the default.

    ``volume`` is the resident full-resolution mask (``edit.lattice.MaskVolume``)
    when one exists; striding it is a view and costs nothing. Note that at stride 1
    the grid then *borrows* that buffer rather than copying it -- fine, because the
    caller contours immediately and drops the grid, but it must not be held across a
    ``MaskVolume.refresh()``.

    Decoding is still whole-slice (``slice_z`` has no windowed form), so striding
    in-plane discards most of each slice. That is fine: the decode is the cheap part.
    """
    grid, vol, _s = _whole_grid(frame, labels, stride, volume)
    # vol is [z, y, x]; a C-order ravel is x-fastest, which is ImageData's point order.
    grid.point_data["mask"] = np.ascontiguousarray(vol).ravel()
    if not vol.any():
        return grid, None
    return grid, _contour(grid)


def _whole_grid(frame, labels, stride: int, volume=None):
    """``(grid, vol, stride)`` -- the whole mask, decoded or borrowed, and its grid.

    Split out of :func:`segmentation_volume` so :func:`material_surfaces` can contour
    the same decode several times, once per material, rather than paying for it twice.
    The grid carries no scalars yet: whoever contours it says what it is contouring.
    """
    s = max(int(stride), 1)
    if volume is not None:
        vol = volume[::s, ::s, ::s]
    else:
        # `decode_volume` preallocates and fills in place; `np.stack` used to hold
        # the list *and* the result, a 4.7 GB peak for a 2.34 GB array.
        from .edit.lattice import decode_volume

        vol = decode_volume(labels, stride=s)
    grid = pv.ImageData(
        dimensions=(vol.shape[2], vol.shape[1], vol.shape[0]),
        spacing=tuple(float(v) * s for v in frame.seg_spacing),
        origin=tuple(float(v) for v in frame.seg_to_um([[0, 0, 0]])[0]),
    )
    return grid, vol, s


def material_color(material) -> str:
    """The colour to draw a material in: Avizo's own, or the viewer's blue.

    Avizo stores each material's display colour in the header, and it is the colour
    the operator has been looking at in Avizo all along -- reusing it means the left
    coronary is the same colour in both applications. Only a material with no colour,
    or none at all, falls back to ``SEG_COLOR``.
    """
    rgb = getattr(material, "color", None) if material is not None else None
    if not rgb:
        return SEG_COLOR
    r, g, b = (min(max(float(v), 0.0), 1.0) for v in rgb)
    if r == g == b == 0.0:  # Avizo writes black for a material it never coloured
        return SEG_COLOR
    return "#%02x%02x%02x" % (round(r * 255), round(g * 255), round(b * 255))


def material_surfaces(frame, labels, materials, stride: int = SEG_STRIDE, volume=None):
    """One isosurface per foreground material -- ``[(Material, mesh), ...]``.

    :func:`segmentation_volume` contours ``mask > 0``, which is right for a binary
    lattice and wrong for a ``.Regions.am``: it fuses ``Left_Tree`` and ``Right_Tree``
    into a single surface everywhere they touch, and leaves no way to colour them
    apart. This contours ``mask == value`` per material instead, so the two coronaries
    arrive as two meshes that can carry Avizo's own colours.

    One decode, one grid, re-scalared per material. A material with no voxels at this
    stride is skipped rather than returned empty, since an actor with no cells is just
    an entry in the legend for something invisible.
    """
    regions = [m for m in (materials or ()) if int(getattr(m, "value", 0)) != 0]
    if not regions:
        return []
    grid, vol, _s = _whole_grid(frame, labels, stride, volume)
    out = []
    for material in regions:
        selected = vol == int(material.value)
        if not selected.any():
            continue
        grid.point_data["mask"] = np.ascontiguousarray(selected).ravel().view(np.uint8)
        surf = _contour(grid)
        if surf is not None and surf.n_cells:
            out.append((material, surf))
    return out


def centreline_polydata(graph, flagged=None, breaks=None) -> pv.PolyData:
    """The skeleton as a single PolyData of polylines, with radius as a point scalar.

    ``flagged`` is a (P,) boolean of points Avizo invented; the polyline is *broken* at
    each of them, so the viridis-by-radius centreline stops where the real skeleton
    stops and the grey bridge layer takes over.

    ``breaks`` is a (P,) boolean of *steps* -- True at ``i`` means the span from ``i``
    to ``i + 1`` is an unsampled jump. It cuts the cell **after** ``i`` while keeping
    both points, which is the difference that matters: a jump's two anchors are real
    measured centreline and dropping either would erase vessel that was actually
    observed. ``flagged`` cannot express this, because it marks points and a jump has
    none of its own.

    **Every point stays in the point array, flagged or not.** The pick contract in this
    module's docstring is that a picked vertex id is the index into the graph's own point
    order; dropping points to shorten the array would silently misreport every pick past
    the first bridge. Cells are what changes, never the points.
    """
    pts = graph.points
    off = graph.edge_offsets
    keep = (
        np.zeros(len(pts), dtype=bool) if flagged is None
        else ~np.asarray(flagged, dtype=bool).ravel()
    )
    if flagged is None:
        keep[:] = True

    brk = (np.zeros(len(pts), dtype=bool) if breaks is None
           else np.asarray(breaks, dtype=bool).ravel())

    cells = []
    for e in range(graph.n_edge):
        a, b = int(off[e]), int(off[e + 1])
        run = a
        for i in range(a, b + 1):
            if i < b and keep[i] and not brk[i]:
                continue
            if i < b and keep[i]:
                # Break *after* i: i closes this cell and i + 1 opens the next.
                if (i + 1) - run >= 2:
                    cells.append(
                        np.concatenate([[(i + 1) - run], np.arange(run, i + 1)])
                    )
                run = i + 1
                continue
            if i - run >= 2:
                cells.append(np.concatenate([[i - run], np.arange(run, i)]))
            run = i + 1
    poly = pv.PolyData(pts.astype(np.float32))
    poly.lines = np.concatenate(cells) if cells else np.empty(0, dtype=np.int64)
    poly.point_data["radius_um"] = graph.thickness.astype(np.float32)
    order = point_strahler(graph)
    if order is not None:
        poly.point_data["strahler"] = order
    return poly


def jump_polydata(graph, breaks) -> pv.PolyData:
    """The unsampled jumps themselves, as one two-point line each.

    Drawn separately and in their own colour because they are a different claim from a
    Hermite fill: Avizo wrote *no* points here at all, so what is on screen is a
    straight line between two real ends and nothing was ever measured along it. On
    LADAF-2024-28 these run to 9.8 mm, which is far too long a piece of invented
    geometry to leave looking like ordinary centreline.
    """
    brk = np.asarray(breaks, dtype=bool).ravel()
    idx = np.flatnonzero(brk)
    poly = pv.PolyData(np.asarray(graph.points, dtype=np.float32))
    if not len(idx):
        poly.lines = np.empty(0, dtype=np.int64)
        return poly
    poly.lines = np.concatenate([[2, int(i), int(i + 1)] for i in idx])
    return poly


def interpolation_polydata(graph, flagged) -> pv.PolyData:
    """The invented spans as their own polylines, one point of real vessel either side.

    The padding is what makes the layer read as a bridge rather than as debris: without
    it each grey run stops one point short of the vessel it was invented to join, and
    the eye reads the resulting sliver of background as a gap in the data.
    """
    from .edit.interpolation import runs_in_edges

    flagged = np.asarray(flagged, dtype=bool).ravel()
    if not flagged.any():
        return pv.PolyData()
    runs = runs_in_edges(flagged, graph.edge_offsets, pad=1)
    return polylines_polydata([graph.points[a:b] for a, b in runs if b - a >= 2])


def _radius_clim(graph, flagged=None) -> tuple[float, float] | None:
    """Finite positive radius range shared by the centreline and radius rings.

    Invented radii are excluded. One point pinned to a calibration intercept -- which is
    what ``FLOOR_RADIUS`` finds -- would otherwise stretch the bottom of the colour range
    down to a value nothing measured, and flatten the contrast across the real tree.
    """
    radius = np.asarray(graph.thickness, dtype=np.float64).ravel()
    good = np.isfinite(radius) & (radius > 0.0)
    if flagged is not None:
        good &= ~np.asarray(flagged, dtype=bool).ravel()
    valid = radius[good]
    if not len(valid):
        return None
    return float(valid.min()), float(valid.max())


def edge_strahler(graph) -> np.ndarray | None:
    """(E,) Strahler order per edge, under whatever name the file used, or None.

    Absent is a real answer, not a failure: a graph Avizo never ordered has no orders
    to draw, and the panel greys the mode out rather than inventing ones.
    """
    attrs = getattr(graph, "edge_attrs", None) or {}
    for name in STRAHLER_ALIASES:
        if name not in attrs:
            continue
        order = np.asarray(attrs[name], dtype=np.float64).ravel()
        if len(order) == int(graph.n_edge):
            return order
    return None


def point_strahler(graph) -> np.ndarray | None:
    """(P,) order of each point's owning edge, or None when the graph has no orders.

    Strahler is an **edge** property, so this is a lookup rather than a measurement:
    every point of an edge carries its edge's order. A junction appears once per
    incident edge in ``graph.points``, and each of those copies therefore takes the
    order of the branch it belongs to rather than one blended value -- the same
    reason ``_point_tangents`` differences each edge independently.
    """
    order = edge_strahler(graph)
    if order is None:
        return None
    owner = np.asarray(graph.edge_of_point(), dtype=np.int64)
    if len(owner) != len(np.asarray(graph.points)):
        return None
    return order[owner].astype(np.float32)


def _strahler_clim(graph, flagged=None) -> tuple[float, float] | None:
    """Order range over the points that are actually drawn, or None if there are none.

    Invented points are excluded for the same reason ``_radius_clim`` excludes them,
    though it can only ever narrow the range here: an invented point inherits its
    edge's order rather than carrying one of its own.
    """
    order = point_strahler(graph)
    if order is None:
        return None
    good = np.isfinite(order) & (order > 0.0)
    if flagged is not None:
        good &= ~np.asarray(flagged, dtype=bool).ravel()
    valid = order[good]
    if not len(valid):
        return None
    return float(valid.min()), float(valid.max())


def color_kwargs(mode: str, graph, flagged=None) -> dict | None:
    """``add_mesh`` colour arguments for one colour mode, or None if unavailable.

    None means "this graph cannot be coloured that way" -- no Strahler attribute, or
    no valid radii -- and is what the panel greys the mode out on.
    """
    if mode == "strahler":
        clim = _strahler_clim(graph, flagged)
        if clim is None:
            return None
        lo, hi = int(round(clim[0])), int(round(clim[1]))
        orders = list(range(lo, hi + 1))
        return {
            "scalars": "strahler",
            "cmap": "viridis",
            # Bands rather than a ramp, and ticks replaced by one annotation per band:
            # the order is a count of branching generations, so a continuous bar would
            # invite reading a 2.5 off a scale that has no such value in it.
            "n_colors": len(orders),
            "clim": (lo - 0.5, hi + 0.5),
            "annotations": {float(o): str(o) for o in orders},
            "scalar_bar_args": {
                "title": "Strahler order",
                "color": "white",
                "n_labels": 0,
            },
        }
    clim = _radius_clim(graph, flagged)
    if clim is None:
        return None
    return {
        "scalars": "radius_um",
        "cmap": "viridis",
        "clim": clim,
        "scalar_bar_args": {"title": "radius (um)", "color": "white"},
    }


def _point_tangents(graph) -> np.ndarray:
    """Robust edge-local unit tangent for every stored graph point.

    Graph junctions occur once per incident edge in ``graph.points``. Differencing
    each edge independently therefore preserves the separate cross-section plane each
    branch implies. Coincident runs are uncommon but legal; a zero derivative inherits
    the nearest usable tangent on its own edge, and an entirely degenerate/single-point
    edge uses +z as a deterministic fallback.
    """
    points = np.asarray(graph.points, dtype=np.float64)
    tangent = np.zeros_like(points)
    offsets = np.asarray(graph.edge_offsets, dtype=np.int64)
    # The tangent window scales with the local radius, so the fit spans a fixed
    # number of vessel widths rather than a fixed number of points.
    try:
        radii_for_tangents = np.asarray(graph.thickness, dtype=np.float64).ravel()
        if radii_for_tangents.size != len(points):
            radii_for_tangents = np.full(len(points), np.nan)
    except Exception:
        radii_for_tangents = np.full(len(points), np.nan)
    step = np.linalg.norm(np.diff(points, axis=0), axis=1) if len(points) > 1 else np.zeros(0)
    step = step[np.isfinite(step) & (step > 0)]
    spacing_hint = float(np.median(step)) if len(step) else 1.0

    for a, b in zip(offsets[:-1], offsets[1:]):
        a, b = int(a), int(b)
        n = b - a
        if n <= 0:
            continue
        if n == 1:
            tangent[a] = [0.0, 0.0, 1.0]
            continue

        # `np.gradient` differences the raw skeleton, which on a thinned
        # centreline measures the voxel lattice rather than the vessel: the angle
        # between neighbouring tangents came out at a median 19.5 degrees, with
        # 40% of rings tilted more than 20 degrees from their neighbour, so the
        # rings splayed at junctions and read as a measurement error that was not
        # there. The radii themselves were measured on a radius-scaled quadratic
        # fit (median swing 1.9 degrees); orienting their rings with a cruder
        # estimator made a correct radius look wrong. Use the same estimator the
        # measurement used, falling back to the difference if it is unavailable.
        try:
            from .crosssection import robust_edge_tangents

            if n < 4:
                # A quadratic through three corner samples extrapolates the end
                # direction beyond the last edge. There is no supported smooth
                # approach fit here: retain the incident edge's one-sided plane.
                derivative = np.gradient(points[a:b], axis=0)
            else:
                derivative = robust_edge_tangents(
                    points[a:b], radii_for_tangents[a:b], spacing_um=spacing_hint
                )
        except Exception:
            derivative = np.gradient(points[a:b], axis=0)
        norm = np.linalg.norm(derivative, axis=1)
        good = np.isfinite(derivative).all(axis=1) & (norm > 1e-12)
        if not good.any():
            tangent[a:b] = [0.0, 0.0, 1.0]
            continue

        derivative[good] /= norm[good, None]
        good_i = np.flatnonzero(good)
        bad_i = np.flatnonzero(~good)
        if len(bad_i):
            # Pick the closest valid derivative without crossing an edge boundary.
            insertion = np.searchsorted(good_i, bad_i)
            right = good_i[np.minimum(insertion, len(good_i) - 1)]
            left = good_i[np.maximum(insertion - 1, 0)]
            nearest = np.where(bad_i - left <= right - bad_i, left, right)
            derivative[bad_i] = derivative[nearest]
        tangent[a:b] = derivative

    bad = ~np.isfinite(tangent).all(axis=1) | (np.linalg.norm(tangent, axis=1) <= 1e-12)
    tangent[bad] = [0.0, 0.0, 1.0]
    return tangent


def radius_circle_polydata(graph, resolution: int = 48, flagged=None) -> pv.PolyData:
    """Ideal radius circle at every valid graph point as one compact line mesh.

    Every output cell is a closed polyline in the plane perpendicular to its point's
    edge-local tangent. ``radius_um`` is cell data (rather than point data), so the
    viridis mapping is constant around each ring. Non-finite coordinates and non-finite
    or non-positive radii are omitted without changing the source graph, and so are
    invented ones: drawing a crisp cross-section for a radius nobody ever measured is
    the precise false impression this whole layer is meant to avoid giving.
    """
    resolution = int(resolution)
    if resolution < 3:
        raise ValueError("circle resolution must be at least 3")

    centres = np.asarray(graph.points, dtype=np.float64)
    radius = np.asarray(graph.thickness, dtype=np.float64).ravel()
    valid = (
        np.isfinite(centres).all(axis=1)
        & np.isfinite(radius)
        & (radius > 0.0)
    )
    if flagged is not None:
        valid &= ~np.asarray(flagged, dtype=bool).ravel()
    if not valid.any():
        return pv.PolyData()

    centres = centres[valid].astype(np.float32)
    radius = radius[valid].astype(np.float32)
    tangent = _point_tangents(graph)[valid].astype(np.float32)

    # Choose a reference vector that is never close to parallel with the tangent,
    # then complete a right-handed orthonormal basis for the cross-section plane.
    reference = np.zeros_like(tangent)
    reference[:, 2] = 1.0
    near_z = np.abs(tangent[:, 2]) > 0.9
    reference[near_z] = [0.0, 1.0, 0.0]
    u = np.cross(tangent, reference)
    u /= np.linalg.norm(u, axis=1, keepdims=True)
    v = np.cross(tangent, u)

    angle = np.linspace(0.0, 2.0 * np.pi, resolution + 1, dtype=np.float32)
    rings = (
        centres[:, None, :]
        + radius[:, None, None]
        * (
            np.cos(angle)[None, :, None] * u[:, None, :]
            + np.sin(angle)[None, :, None] * v[:, None, :]
        )
    )

    n_ring = len(centres)
    points_per_ring = resolution + 1
    lines = np.empty((n_ring, points_per_ring + 1), dtype=np.int64)
    lines[:, 0] = points_per_ring
    lines[:, 1:] = np.arange(n_ring * points_per_ring, dtype=np.int64).reshape(
        n_ring, points_per_ring
    )
    # ``pv.PolyData(points)`` also creates one vertex cell per point. Start empty so
    # the only cells are the rings and one cell scalar maps to exactly one circle.
    mesh = pv.PolyData()
    mesh.points = rings.reshape(-1, 3)
    mesh.lines = lines.ravel()
    mesh.cell_data["radius_um"] = radius
    order = point_strahler(graph)
    if order is not None:
        mesh.cell_data["strahler"] = order[valid]
    return mesh


def _clean_point(pt) -> np.ndarray | None:
    """Accept a pick only if it is three finite numbers.

    Clicking empty space hands back an empty array, and PyVista will also pass along
    whatever the picker produced on a failed hit; either would otherwise blow up on the
    first reshape.
    """
    if pt is None:
        return None
    arr = np.asarray(pt, dtype=float).ravel()
    if arr.size != 3 or not np.all(np.isfinite(arr)):
        return None
    return arr


def _background_plotter(title):
    """The real window. Imported here so `viewer3d` loads without pyvistaqt."""
    from pyvistaqt import BackgroundPlotter

    return BackgroundPlotter(title=title, auto_update=False)


class Picker3D:
    """Persistent 3D chooser. Calls ``on_open(xyz_um)`` when the user presses ``v``."""

    def __init__(self, graph=None, mesh_um=None, cands=None, frame=None, on_open=None,
                 stack=None, labels=None, seg_box_um=SEG_BOX_UM, seg_stride=SEG_STRIDE,
                 plane_opacity=PLANE_OPACITY, surface_opacity=SURFACE_OPACITY,
                 title="HiP-CT segmentation debugger - 3D", plotter_factory=None,
                 mask=None, materials=()):
        self.graph = graph
        self.mesh = mesh_um
        self.cands = list(cands or [])
        self.frame = frame
        self.on_open = on_open
        self.stack = stack
        self.labels = labels
        # The resident full-resolution mask (`edit.lattice.MaskVolume`), or None to
        # decode per use. Held rather than reached through the session because the
        # picker takes inputs, never a session -- which is what lets `set_dataset`
        # swap them one at a time.
        self.mask = mask
        # The lattice's `Materials` block. When it names more than the Exterior, the
        # whole-tree layer is contoured once per material and coloured with Avizo's
        # own colours, instead of one fused surface over `mask > 0`.
        self.materials = tuple(materials or ())
        self.seg_box_um = float(seg_box_um)
        self.seg_stride = max(int(seg_stride), 1)
        self.title = title
        # Injected so a test can drive the whole class against
        # `pv.Plotter(off_screen=True)` -- there is no window to close and no Qt loop
        # to enter, which is what makes the swap regression test cheap enough to run.
        self.plotter_factory = plotter_factory
        self._opacity = {key: default for key, _, default, _vis in LAYERS}
        self._opacity["plane"] = float(plane_opacity)
        self._opacity["surface"] = float(surface_opacity)
        # Set by the layer panel so that the `i` / `g` keys move its checkboxes too.
        self.on_layers_changed = None
        # Fired before a dataset is torn down, so whoever owns actors registered
        # through `set_extra_actors` can take them back first. A single slot, like
        # `on_layers_changed`.
        self.on_dataset_changing = None
        self.panel = None
        self._instructions_actor = None
        self.picked: np.ndarray | None = None
        self.pick_source = ""
        self._cand_i = -1
        self._point_i = -1  # index into graph.points, -1 when the pick is not a vertex
        self._note = ""
        self.plotter = None
        self._marker = None
        self._label = None
        self._surface_actor = None
        self._centreline_actor = None
        #: The centreline mesh is kept so that recolouring can re-add the same
        #: geometry: rebuilding it walks every edge in Python, and the colour is a
        #: property of the mapper, not of the polylines.
        self._centreline_mesh = None
        #: Which scalar both the centreline and the radius rings are coloured by.
        #: One setting for the two, because a ring is a cross-section *of* the
        #: centreline it sits on and two different maps would be read as one.
        self._color_by = "radius"
        self._interpolated_actor = None
        # (P,) reason codes for the current graph, computed once per dataset. Several
        # layers need it and detection is not free enough to repeat per rebuild.
        self._interpolated: np.ndarray | None = None
        self._interpolated_stored = False  # came from the file rather than detected here
        #: (P,) bool of *steps* that are unsampled jumps. Separate from
        #: `_interpolated` because it is indexed by step, not by point.
        self._jump_breaks: np.ndarray | None = None
        self._jump_actor = None
        self._radius_circles_actor = None
        self._radius_circles_mesh = None
        self._radius_circles_on = False
        #: The key in the corner of the render. An actor rather than a layer: it is
        #: rebuilt from the labelled actors on every `_populate`, so what has to
        #: survive a dataset swap is the preference, not the actor.
        self._legend_actor = None
        self._legend_on = True
        self._cand_actors: dict = {}  # actor -> candidate index per point in its cloud
        # Routes drawn for the Reconnect panel. Keyed by actor name so one candidate
        # replaces the previous one wholesale -- a review session steps through
        # dozens, and a leftover alternative from the last one is worse than
        # drawing nothing, because it looks like a route for the current one.
        self._route_actors: dict = {}
        #: Crop preview actors. Built by `show_crop` and therefore owned here, unlike
        #: the edit tools' extras, which their controller takes back for itself.
        self._crop_actors: list = []
        self._reformat_actors: list = []
        self._reformat_stack_actors: list = []
        self._reformat_current_actors: list = []
        self._reformat_current = None
        #: Debug actors for the `radius-perimeter` cut planes. Owned here rather than
        #: registered as extras by their panel, for the same reason the crop preview
        #: is: they are built from the session's graph, so they die with the dataset.
        self._section_actors: list = []
        self._surface_pickable = False
        self._plane_actor = None
        self._plane_on = False
        self._slice_z = -1  # raw slice the plane and the shape overlays are showing
        self._plane_z = -1  # raw slice the plane actor was actually built for
        self._seg_actor = None
        self._seg_on = False
        # The whole-tree mask is a fixed piece of geometry, so the mesh is built
        # once and kept even while the layer is off.
        # A list, one actor per material: a `.Regions.am` draws Left_Tree and
        # Right_Tree separately. A binary mask makes a one-element list.
        self._seg_all_actor = []
        self._seg_all_on = False
        # [(Material or None, mesh)], built once and kept even while the layer is off.
        self._seg_all_mesh = None
        # Edit version the cached mesh was built at, so the status line can say when
        # the isosurface no longer shows what has been painted.
        self._seg_all_version = 0
        # Per-slice shapes taken from the slab the browser last built: for each layer,
        # {raw slice -> [ (N, 3) um polylines ]}.
        self._shapes: dict = {key: {} for key in SHAPE_KEYS}
        self._shape_actors: dict = {}
        self._shape_on = {
            key: vis for key, _l, _o, vis in LAYERS if key in SHAPE_KEYS
        }
        # Actors owned by another module (see EXTRA_KEYS), keyed by layer.
        self._extra_actors: dict[str, list] = {}
        # (P,) owning edge of every graph point; a repeat(), rebuilt on every swap.
        self._set_derived()

    # -- ui ---------------------------------------------------------------- #
    def _status(self) -> str:
        if self.graph is None:
            return "no dataset loaded - pick your inputs in the Data tab" + (
                f"\n\n{self._note}" if self._note else ""
            )
        if self.picked is None:
            base = "no point picked"
        else:
            x, y, z = self.picked
            base = f"pick ({self.pick_source})  {x:,.0f}, {y:,.0f}, {z:,.0f} um"
            if self.frame is not None:
                zyx = self.frame.um_to_raw_index(self.picked)[0]
                base += f"\nslice {zyx[0]}  row {zyx[1]}  col {zyx[2]}"
            if 0 <= self._point_i < len(self.graph.points):
                base += (
                    f"\npoint {self._point_i}  r = {self.graph.thickness[self._point_i]:,.0f} um"
                    f"  edge {self._edge_of_point[self._point_i]}"
                )
            if 0 <= self._cand_i < len(self.cands):
                c = self.cands[self._cand_i]
                base += f"\n{self._cand_i + 1}/{len(self.cands)}  {c.kind}\n{c.detail}"
        if self.segmentation_all_stale():
            # The mesh is cached for the session, so a correction painted since it
            # was built is simply not in it. Saying nothing would be showing a lie.
            base += "\n(whole-tree mask is older than your edits - press rebuild)"
        return base + (f"\n\n{self._note}" if self._note else "")

    def _marker_radius(self) -> float:
        """Size the marker from the *local* radius where we know it.

        A tree-wide mean swallows every small vessel it is placed on, which makes it
        impossible to see whether the pick landed where it was aimed.
        """
        if self.graph is None:
            return 60.0 * 1.5
        if 0 <= self._point_i < len(self.graph.thickness):
            r = float(self.graph.thickness[self._point_i])
        else:
            r = float(self.graph.thickness.mean())
        return max(r, 60.0) * 1.5

    def _refresh(self):
        p = self.plotter
        if p is None:
            return
        if self._marker is not None:
            p.remove_actor(self._marker, render=False)
            self._marker = None
        if self.picked is not None:
            self._marker = p.add_mesh(
                pv.Sphere(radius=self._marker_radius(), center=self.picked),
                color="cyan",
                opacity=0.55,
                name="_pick_marker",
                reset_camera=False,
                # Never pickable: otherwise the next click in this region lands on the
                # marker's near face and drags the pick towards the camera.
                pickable=False,
            )
        if self._label is not None:
            p.remove_actor(self._label, render=False)
        self._label = p.add_text(self._status(), position="lower_right", font_size=9, color="white")
        p.render()

    def _set_pick(self, xyz, source="click", point_i=-1, cand_i=-1):
        pt = _clean_point(xyz)
        if pt is None:
            return  # click on empty space: not an error, just nothing to do
        self.picked = pt
        self.pick_source = source
        self._point_i = int(point_i)
        self._cand_i = int(cand_i)
        self._note = ""
        # Overlays follow the pick when they are switched on. The shapes are the
        # exception: they were handed over with the last slab, so a pick somewhere else
        # invalidates them until 'v' rebuilds it.
        self._update_segmentation()
        self._shapes = {key: {} for key in SHAPE_KEYS}
        z = self._pick_slice()
        if z >= 0:
            self.set_current_slice(z)
        self._refresh()
        self._layers_changed()

    # -- reconnection review -------------------------------------------------- #
    def show_reconnect_candidate(self, record, *, waypoints=(),
                                 actor_prefix="reconnect-review"):
        """Draw one reviewed route, its alternatives and its waypoints.

        Called by the Reconnect panel as the operator steps through the work list.
        The alternatives are the reason this exists: a route that looks entirely
        reasonable on its own usually has a second, equally reasonable one beside
        it, and the whole judgement being asked for is which of them is the vessel.
        Drawing only the winner would hide the question.

        Everything is unpickable. These are annotations on top of the skeleton, and
        a route lying along the centreline would otherwise swallow the picks that
        place waypoints -- which is the one gesture this view has to support.
        """
        import pyvista as pv

        self.clear_reconnect_candidate()
        p = self.plotter
        if p is None or record is None:
            return

        routes = [(record.get("route"), ROUTE_COLOR, 5, "route")]
        for i, alternative in enumerate(record.get("alternatives") or (), 1):
            routes.append((alternative, ALTERNATIVE_COLOR, 3, f"alternative {i}"))

        for route, colour, width, label in routes:
            points = (route or {}).get("path_um")
            if not points or len(points) < 2:
                continue
            name = f"{actor_prefix}-{label.replace(' ', '-')}"
            self._route_actors[name] = p.add_mesh(
                pv.MultipleLines(np.asarray(points, dtype=np.float32)),
                color=colour, line_width=width, name=name, label=label,
                reset_camera=False, lighting=False, pickable=False,
            )

        if len(waypoints):
            name = f"{actor_prefix}-waypoints"
            self._route_actors[name] = p.add_mesh(
                pv.PolyData(np.asarray(waypoints, dtype=np.float32)),
                color=WAYPOINT_COLOR, point_size=14, render_points_as_spheres=True,
                name=name, label=f"waypoints ({len(waypoints)})",
                reset_camera=False, pickable=False,
            )
        p.render()

    def show_crop(self, drop_polylines=(), vessels=()) -> None:
        """Draw a crop preview: what a rule would take, and the vessels it is judged by.

        ``drop_polylines`` is a sequence of (N, 3) um centrelines; ``vessels`` a
        sequence of ``(name, colour, [polyline, ...])``.

        The whole preview is **one** actor, because a crop routinely selects thousands
        of segments and one actor per segment would cost more than the contour it is
        previewing. Each named vessel gets its own actor instead: there are a handful,
        and the panel's row has to be able to colour them apart.

        Registered through ``set_extra_actors``, so these get the same panel row and
        opacity memory as any other layer -- and, like every extra, are never pickable.
        A fat tube lying along the centreline would otherwise swallow the picks that
        build the selection in the first place.
        """
        p = self.plotter
        if p is None:
            return
        # `set_extra_actors(key, None)` only *unregisters* -- the edit tools remove
        # their own actors, because they own them. These are built here, so they are
        # removed here, or an emptied preview would stay on screen unreachable by any
        # panel row.
        self._clear_crop_actors()

        drop = [np.asarray(line, dtype=np.float32) for line in drop_polylines]
        drop = [line for line in drop if line.ndim == 2 and len(line) >= 2]
        if drop:
            actor = p.add_mesh(
                polylines_polydata(drop), color=CROP_DROP_COLOR, line_width=6,
                name="_crop_preview", label=f"to be dropped ({len(drop)})",
                reset_camera=False, lighting=False, pickable=False,
            )
            self._crop_actors.append(actor)
            self.set_extra_actors("crop_preview", [actor])
        else:
            self.set_extra_actors("crop_preview", None)

        actors = []
        for i, (name, colour, lines) in enumerate(vessels):
            lines = [np.asarray(line, dtype=np.float32) for line in lines]
            lines = [line for line in lines if line.ndim == 2 and len(line) >= 2]
            if not lines:
                continue
            actors.append(p.add_mesh(
                polylines_polydata(lines), color=colour, line_width=9,
                name=f"_crop_vessel_{i}", label=str(name), reset_camera=False,
                lighting=False, pickable=False,
            ))
        self._crop_actors.extend(actors)
        self.set_extra_actors("crop_vessels", actors or None)
        p.render()

    def _clear_crop_actors(self) -> None:
        p = self.plotter
        for actor in self._crop_actors:
            if actor is not None and p is not None:
                p.remove_actor(actor, render=False)
        self._crop_actors = []

    def clear_crop(self) -> None:
        """Take both crop layers down. Safe to call with nothing drawn."""
        self._clear_crop_actors()
        for key in ("crop_preview", "crop_vessels"):
            self.set_extra_actors(key, None)
        if self.plotter is not None:
            self.plotter.render()

    def show_reformat(self, polylines=(), dropped=()) -> None:
        """Draw the reformat selection: the run that will be sampled, and what will not.

        ``polylines`` is the chain the stack will actually be built from; ``dropped``
        is everything else the user has selected -- segments not connected to that
        chain, which cannot be part of the same stack.

        **Two actors, two colours, and that distinction is the point.** A reformat is
        one continuous path by definition, so a selection spanning two disconnected
        vessels can only produce a stack of one of them. Drawing all of it identically
        would show the operator one thing while the build did another.

        Deliberately a sibling of :meth:`show_crop` rather than another caller of it:
        that method clears every actor it has drawn, so a second panel using it would
        wipe the crop preview each time the reformat selection changed, and vice versa.

        Never pickable, like every extra -- a fat line lying along the centreline would
        swallow the very picks that build the selection.
        """
        p = self.plotter
        if p is None:
            return
        self._clear_reformat_actors()

        def _clean(lines):
            out = [np.asarray(line, dtype=np.float32) for line in lines]
            return [line for line in out if line.ndim == 2 and len(line) >= 2]

        kept, lost = _clean(polylines), _clean(dropped)
        if not kept and not lost:
            self.set_extra_actors("reformat_path", None)
            p.render()
            return

        actors = []
        if lost:
            actors.append(p.add_mesh(
                polylines_polydata(lost), color=REFORMAT_DROPPED_COLOR, line_width=5,
                name="_reformat_dropped",
                label=f"reformat: not in the run ({len(lost)})",
                reset_camera=False, lighting=False, pickable=False,
            ))
        if kept:
            actors.append(p.add_mesh(
                polylines_polydata(kept), color=REFORMAT_COLOR, line_width=9,
                name="_reformat_path", label=f"reformat ({len(kept)} segment(s))",
                reset_camera=False, lighting=False, pickable=False,
            ))
        self._reformat_actors.extend(actors)
        self.set_extra_actors("reformat_path", actors)
        p.render()

    def _clear_reformat_actors(self) -> None:
        p = self.plotter
        for actor in self._reformat_actors:
            if actor is not None and p is not None:
                p.remove_actor(actor, render=False)
        self._reformat_actors = []

    def show_reformat_stack(self, corners=None, sections=(), current=None) -> None:
        """Draw a reformat's sampling planes in 3D.

        ``corners``  -- ``(N, 4, 3)`` um, the outline of every plane in the stack.
        ``sections`` -- ``[(corners (4,3), image (H, W)), ...]``, drawn as textured
                        quads. Keep this to a couple of dozen; see
                        :data:`REFORMAT_MAX_TEXTURED`.
        ``current``  -- ``(corners (4,3), image (H, W))`` or ``None``, the plane the
                        2D reformat window is showing right now.

        Three layers rather than one because they answer different questions. The
        frames say *where the stack sampled and how it is oriented*, and are the only
        place plane collision is visible at all -- squares fanning through each other
        on the inside of a bend. The textured sections say *what it found*. The current
        section ties the two windows together: scrolling the napari stack walks a lit
        square down the vessel here.

        None of them are pickable, like every extra. A stack of quads lying across the
        vessel would otherwise swallow every double-click aimed at the centreline
        behind it -- the same reason the axial image plane is not pickable either.
        """
        p = self.plotter
        if p is None:
            return
        self._clear_reformat_stack_actors()

        frames = None
        if corners is not None and len(corners):
            frames = p.add_mesh(
                corner_outlines(corners), color=REFORMAT_COLOR, line_width=2,
                name="_reformat_frames", label=f"plane frames ({len(corners)})",
                reset_camera=False, lighting=False, pickable=False,
            )
            self._reformat_stack_actors.append(frames)
        self.set_extra_actors("reformat_frames", [frames] if frames else None)

        planes = []
        for i, (quad_corners, image) in enumerate(sections):
            planes.append(p.add_mesh(
                corner_quad(quad_corners),
                texture=pv.Texture(slice_texture_array(image)),
                name=f"_reformat_plane_{i}", reset_camera=False, lighting=False,
                pickable=False,
            ))
        self._reformat_stack_actors.extend(planes)
        self.set_extra_actors("reformat_planes", planes or None)

        self._reformat_current = current
        self._draw_reformat_current()
        p.render()

    def set_reformat_section(self, corners, image) -> None:
        """Move the current-section quad. Cheap enough for every slider step.

        Only this one actor is rebuilt -- the frames and the textured stack do not move
        when the 2D window scrolls, and tearing them down and back up at slider rate
        would make scrolling unusable.
        """
        if self.plotter is None:
            return
        self._reformat_current = (corners, image)
        self._draw_reformat_current()
        self.plotter.render()

    def _draw_reformat_current(self) -> None:
        p = self.plotter
        if p is None:
            return
        for actor in self._reformat_current_actors:
            p.remove_actor(actor, render=False)
        self._reformat_current_actors = []
        if self._reformat_current is None:
            self.set_extra_actors("reformat_slice", None)
            return

        corners, image = self._reformat_current
        actors = [
            p.add_mesh(
                corner_quad(corners), texture=pv.Texture(slice_texture_array(image)),
                name="_reformat_slice", reset_camera=False, lighting=False,
                pickable=False,
            ),
            # An outline as well as the image: at a glancing angle the quad is nearly
            # edge-on and all but invisible, which is exactly when you most want to know
            # where it is.
            p.add_mesh(
                corner_outlines([corners]), color=REFORMAT_CURRENT_COLOR, line_width=4,
                name="_reformat_slice_edge", reset_camera=False, lighting=False,
                pickable=False,
            ),
        ]
        self._reformat_current_actors = actors
        self.set_extra_actors("reformat_slice", actors)

    def _clear_reformat_stack_actors(self) -> None:
        p = self.plotter
        for actor in (*self._reformat_stack_actors, *self._reformat_current_actors):
            if actor is not None and p is not None:
                p.remove_actor(actor, render=False)
        self._reformat_stack_actors = []
        self._reformat_current_actors = []
        self._reformat_current = None

    # -- radius-perimeter debug sections -------------------------------------- #
    def show_cross_sections(self, corners=(), truncated=(), contours=(),
                            offsets=(), refused=()) -> None:
        """Draw the planes ``radius-perimeter`` cuts, at sampled points.

        ``corners``   -- ``(N, 4, 3)`` um, windows whose section was measured.
        ``truncated`` -- ``(M, 4, 3)`` um, windows that yielded nothing usable: the
                         section reached the border, or no stable one existed.
        ``contours``  -- ``[(K, 3), ...]`` um, the measured lumen boundary per window.
        ``offsets``   -- ``[(2, 3), ...]`` um, centreline point -> section centroid.
        ``refused``   -- ``[(K, 3), ...]`` um, the boundary of a blob that was refused.

        Five arrays rather than a list of objects so the split between "measured" and
        "refused" is decided by the caller -- :func:`~.edit.section_frames.drawables`
        makes it from the verdicts, where the verdict vocabulary lives. That is also
        why `refused` is its own parameter rather than more `contours`: which layer a
        ring belongs on is a statement about its verdict, and this module does not
        know the verdicts.

        The two window colours are the point of the layer. A radius that came back
        wrong is either measured in the wrong place, which the offset lines show, or
        not measured at all, which is every orange square. Nothing here is pickable,
        like every overlay: a wall of squares across the vessel would otherwise eat
        the double-click aimed at the centreline behind it.
        """
        p = self.plotter
        if p is None:
            return
        self._clear_section_actors()

        def _outlines(quads, color, name, label):
            if quads is None or not len(quads):
                return None
            return p.add_mesh(
                corner_outlines(quads), color=color, line_width=2,
                name=name, label=label, reset_camera=False, lighting=False,
                pickable=False,
            )

        frames = [
            a for a in (
                _outlines(corners, SECTION_FRAME_COLOR, "_section_frames",
                          f"section windows ({len(corners)})"),
                _outlines(truncated, SECTION_FAIL_COLOR, "_section_truncated",
                          f"sections not measured ({len(truncated)})"),
            ) if a is not None
        ]
        self._section_actors.extend(frames)
        self.set_extra_actors("section_frames", frames or None)

        rings = None
        if len(contours):
            rings = p.add_mesh(
                polylines_polydata(contours), color=SECTION_CONTOUR_COLOR,
                line_width=2, name="_section_contours",
                label=f"measured lumen ({len(contours)})",
                reset_camera=False, lighting=False, pickable=False,
            )
            self._section_actors.append(rings)
        self.set_extra_actors("section_contours", [rings] if rings else None)

        # Guarded like every sibling: `polylines_polydata([])` is a valid empty mesh
        # and `add_mesh` would still hand back a live actor, leaving a registered row
        # controlling nothing and a legend entry reading "refused blob (0)".
        ribbons = None
        if len(refused):
            ribbons = p.add_mesh(
                polylines_polydata(refused), color=SECTION_REFUSED_COLOR,
                line_width=2, name="_section_refused",
                label=f"refused blob ({len(refused)})",
                reset_camera=False, lighting=False, pickable=False,
            )
            self._section_actors.append(ribbons)
        self.set_extra_actors("section_refused", [ribbons] if ribbons else None)

        arrows = None
        if len(offsets):
            arrows = p.add_mesh(
                polylines_polydata(offsets), color=SECTION_OFFSET_COLOR,
                line_width=3, name="_section_offsets",
                label=f"centroid offset ({len(offsets)})",
                reset_camera=False, lighting=False, pickable=False,
            )
            self._section_actors.append(arrows)
        self.set_extra_actors("section_offsets", [arrows] if arrows else None)

        self._apply_pickable()
        p.render()

    def clear_cross_sections(self) -> None:
        """Take the section debug layers down. Safe to call with nothing drawn."""
        self._clear_section_actors()
        for key in ("section_frames", "section_contours", "section_refused",
                    "section_offsets"):
            self.set_extra_actors(key, None)
        if self.plotter is not None:
            self.plotter.render()

    def _clear_section_actors(self) -> None:
        p = self.plotter
        for actor in self._section_actors:
            if actor is not None and p is not None:
                p.remove_actor(actor, render=False)
        self._section_actors = []

    def clear_reformat(self) -> None:
        """Take every reformat layer down. Safe to call with nothing drawn."""
        self._clear_reformat_actors()
        self._clear_reformat_stack_actors()
        for key in ("reformat_path", "reformat_frames", "reformat_slice",
                    "reformat_planes"):
            self.set_extra_actors(key, None)
        if self.plotter is not None:
            self.plotter.render()

    def clear_reconnect_candidate(self) -> None:
        """Remove the previously drawn route. Safe to call with nothing drawn."""
        p = self.plotter
        for actor in self._route_actors.values():
            if actor is not None and p is not None:
                p.remove_actor(actor, render=False)
        self._route_actors = {}

    def _candidate_at(self, point_i: int) -> int:
        """Index of the candidate flagged at this graph point, or -1.

        Candidate markers sit *on* the skeleton, so a click aimed at one resolves to the
        centreline underneath it rather than to the marker. Matching the pick back to the
        candidate is what keeps the marker's annotation -- its kind and detail -- which
        is the only reason to click one.
        """
        if not len(self._cand_xyz) or self.graph is None:
            return -1
        d = np.linalg.norm(self._cand_xyz - self.graph.points[point_i], axis=1)
        j = int(np.argmin(d))
        return j if d[j] <= max(float(self.graph.thickness[point_i]), 1.0) else -1

    def _clear(self):
        self.picked = None
        self.pick_source = ""
        self._cand_i = -1
        self._point_i = -1
        self._note = ""
        # The mask box is anchored on the pick, so it goes with it. The image plane is a
        # slice of the stack and stays put.
        if self._seg_actor is not None and self.plotter is not None:
            self.plotter.remove_actor(self._seg_actor, render=False)
            self._seg_actor = None
        self._refresh()

    # -- picking ------------------------------------------------------------- #
    def _pick_here(self, _viewport_xy):
        """Run a pick at the double-clicked position.

        Registered through ``track_click_position(..., double=True)``, so a single click
        never gets here and click-and-drag stays purely a camera gesture. pyvista decides
        what counts as a double from both distance and time -- ``_MAX_CLICK_DELTA`` (~6 px
        between the two presses) and ``_MAX_CLICK_DELAY`` (0.8 s) in its
        ``RenderWindowInteractor._click_event`` -- which is also what rejects a
        press-drag-press. It stores the click before dispatching, so ``click_position``
        is this click in display pixels.
        """
        p = self.plotter
        if p is None or p.click_position is None:
            return
        x, y = p.click_position
        p.iren.picker.Pick(x, y, 0, p.iren.get_poked_renderer())

    def _on_pick(self, picker, _event):
        """Resolve a pick to a spatial-graph point. Observes the picker's EndPickEvent.

        ``picker`` is a ``vtkPointPicker``, so ``GetPointId()`` is the index of the
        vertex it snapped to *within the picked actor's dataset*. Which dataset that is
        decides how to read the id, so dispatch on the actor rather than trusting the
        reported world position -- ``GetPickPosition`` is a point on the ray, not on the
        vertex.
        """
        actor = picker.GetActor()
        pid = int(picker.GetPointId())

        if actor is not None and pid >= 0:
            if actor is self._centreline_actor:
                self._set_pick(
                    self.graph.points[pid],
                    source="centreline",
                    point_i=pid,
                    cand_i=self._candidate_at(pid),
                )
                return
            ids = self._cand_actors.get(actor)
            if ids is not None and 0 <= pid < len(ids):
                self._select_candidate(ids[pid])
                return
            if self._surface_pickable and actor is self._surface_actor:
                # Free picking was asked for explicitly: take the surface point as-is.
                self._set_pick(picker.GetPickPosition(), source="surface")
                return

        self._note = (
            "no centreline within ~10 px - zoom in, or press 's' to pick the surface"
        )
        self._refresh()

    # -- overlays ------------------------------------------------------------ #
    def set_current_slice(self, z: int, defer: bool = False):
        """Move everything that lives on one slice -- the image plane and the shapes.

        ``defer`` hands the work to the Qt event loop first. The napari z slider calls
        this from inside a vispy event with napari's GL context current; touching VTK's
        actors from there is the same mistake that used to take the process down (see
        ``_open``), so the slider path always defers.
        """
        if defer:
            from qtpy.QtCore import QTimer

            QTimer.singleShot(0, lambda: self.set_current_slice(int(z), defer=False))
            return
        self._slice_z = int(z)
        self._set_plane_slice(int(z))
        self._update_shape_actors()

    def _set_plane_slice(self, z: int):
        p = self.plotter
        if p is None or not self._plane_on:
            return
        if self.stack is None or self.frame is None:
            self._note = "no raw stack loaded - image plane unavailable"
            self._refresh()
            return

        z = int(np.clip(int(z), 0, self.stack.n_slices - 1))
        if z == self._plane_z and self._plane_actor is not None:
            return
        img = self.stack.read_slice(z)
        quad = slice_plane_quad(self.frame, self.stack.shape, z)
        self._plane_actor = p.add_mesh(
            quad,
            texture=pv.Texture(slice_texture_array(img)),
            opacity=self._opacity["plane"],
            name="_slice_plane",
            reset_camera=False,
            lighting=False,
            # The plane spans the whole field of view; pickable it would swallow every
            # double-click aimed at a vessel behind it.
            pickable=False,
        )
        self._plane_z = z
        p.render()

    def set_slice_shapes(self, slab):
        """Take the per-slice overlay shapes from a slab into the 3D scene.

        These come from the slab rather than being recomputed here, and that is the
        whole point: the browser has already cut the STL against each plane and measured
        each cross-section, over the same window, so the two windows cannot disagree
        about what they are drawing. It also bounds the work -- every slice the slab
        covers is already in hand.

        The window is whatever the slab used: ``--roi`` pixels around the pick, or the
        whole slice under ``--roi 0``, in which case these shapes span the entire
        cross-section of the tree and line up with the full-slice image plane. Under
        ``--volume`` the z slider can move outside the slab, and there are simply no
        shapes for those slices -- the browser has not measured them.
        """
        self._shapes = {key: {} for key in SHAPE_KEYS}
        if slab is None or self.frame is None:
            self._layers_changed()
            return

        def _add(key, raw_pts):
            a = np.asarray(raw_pts, dtype=np.float64)
            if len(a) < 2:
                return
            z = int(round(float(a[0, 0])))
            self._shapes[key].setdefault(z, []).append(self.frame.raw_to_um(a))

        for arr in slab.contours or []:
            _add("stl_contour", arr)
        for corners in slab.circles or []:
            _add("assumed", ellipse_from_corners(corners))
        for corners in slab.perim_circles or []:
            _add("perimeter", ellipse_from_corners(corners))

        self.set_current_slice(slab.z_centre)
        self._layers_changed()

    def _update_shape_actors(self):
        """Redraw the shape overlays for the current slice."""
        p = self.plotter
        if p is None:
            return
        for key, _label, color in SHAPE_LAYERS:
            actor = self._shape_actors.pop(key, None)
            if actor is not None:
                p.remove_actor(actor, render=False)
            if not self._shape_on.get(key):
                continue
            polys = self._shapes.get(key, {}).get(self._slice_z, [])
            if not polys:
                continue
            mesh = polylines_polydata(polys)
            if mesh.n_cells == 0:
                continue
            self._shape_actors[key] = p.add_mesh(
                mesh,
                color=color,
                line_width=3,
                opacity=self._opacity[key],
                name=f"_shape_{key}",
                reset_camera=False,
                lighting=False,
                pickable=False,
            )
            # These lie exactly in the image plane. Without a depth-buffer offset the
            # two fight for the same fragments and the contour breaks up into dashes.
            mapper = self._shape_actors[key].GetMapper()
            if hasattr(mapper, "SetRelativeCoincidentTopologyLineOffsetParameters"):
                mapper.SetResolveCoincidentTopologyToPolygonOffset()
                mapper.SetRelativeCoincidentTopologyLineOffsetParameters(0.0, -8.0)
        p.render()

    def _pick_slice(self) -> int:
        """Raw slice index of the current pick, or -1."""
        if self.picked is None or self.frame is None:
            return -1
        return int(self.frame.um_to_raw_index(self.picked)[0][0])

    def _set_plane_enabled(self, on: bool):
        """Build or tear down the image plane. The key and the panel both come here.

        An absolute setter rather than a flip, so a checkbox cannot drift out of step
        with the key when one of them refuses (no pick yet, no stack loaded) -- which is
        also why the panel is told to re-read state on *every* path out of here.
        """
        try:
            self._apply_plane_enabled(on)
        finally:
            self._layers_changed()

    def _apply_plane_enabled(self, on: bool):
        p = self.plotter
        if p is None:
            return
        self._plane_on = bool(on)
        if not self._plane_on:
            if self._plane_actor is not None:
                p.remove_actor(self._plane_actor, render=False)
                self._plane_actor = None
            self._plane_z = -1
            self._note = "image plane off"
            self._refresh()
            return
        z = self._pick_slice()
        if z < 0:
            self._plane_on = False
            self._note = "pick a point first - the plane shows the slice through it"
            self._refresh()
            return
        self._note = "image plane on - scroll the slice viewer to move it"
        self.set_current_slice(z)
        self._refresh()

    def _toggle_slice_plane(self):
        self._set_plane_enabled(not self._plane_on)

    def _seg_half_um(self) -> float:
        """Box half-extent: the configured minimum, but never inside the vessel wall.

        A box smaller than the lumen it is centred in is entirely mask, so marching
        cubes finds no boundary and draws nothing. The fattest vessel here has r = 1.55
        mm, so a fixed default would silently come up empty on the very vessels most
        worth looking at.
        """
        half = self.seg_box_um
        if self.graph is not None and 0 <= self._point_i < len(self.graph.thickness):
            half = max(half, 3.0 * float(self.graph.thickness[self._point_i]))
        return half

    def _update_segmentation(self):
        """(Re)build the mask isosurface around the current pick.

        Wrapped, because this runs inside the pick handler: an exception escaping
        here would propagate out of a VTK callback and break every subsequent pick,
        not just this one.
        """
        try:
            self._rebuild_segmentation_box()
        except Exception as exc:  # noqa: BLE001 - a bad box must not kill picking
            self._seg_on = False
            self._note = f"segmentation box failed: {type(exc).__name__}: {exc}"
            self._layers_changed()

    def _rebuild_segmentation_box(self):
        p = self.plotter
        if p is None or not self._seg_on:
            return
        if self._seg_actor is not None:
            p.remove_actor(self._seg_actor, render=False)
            self._seg_actor = None
        if self.picked is None or self.labels is None or self.frame is None:
            return
        # `peek`, not `get`: 'g' is pressed far more often than 'a', and it must not
        # be the thing that suddenly spends two seconds decoding 2.34 GB.
        volume = self.mask.peek() if self.mask is not None else None
        grid, surf = segmentation_box(
            self.frame, self.labels, self.picked, self._seg_half_um(), volume=volume
        )
        if surf is None or surf.n_cells == 0:
            filled = grid is not None and int(np.asarray(grid.point_data["mask"]).min()) > 0
            self._note = (
                "box lies entirely inside the lumen - raise --seg-box-um"
                if filled
                else "no segmentation in this box"
            )
            return
        self._seg_actor = p.add_mesh(
            surf,
            color=SEG_COLOR,
            opacity=self._opacity["segmentation"],
            smooth_shading=True,
            name="_seg_iso",
            reset_camera=False,
            pickable=False,
        )

    def _set_seg_enabled(self, on: bool):
        """Build or tear down the mask isosurface. See ``_set_plane_enabled``."""
        try:
            self._apply_seg_enabled(on)
        finally:
            self._layers_changed()

    def _apply_seg_enabled(self, on: bool):
        p = self.plotter
        if p is None:
            return
        if self.labels is None:
            self._seg_on = False
            self._note = "no segmentation loaded"
            self._refresh()
            return
        self._seg_on = bool(on)
        if not self._seg_on:
            if self._seg_actor is not None:
                p.remove_actor(self._seg_actor, render=False)
                self._seg_actor = None
            self._note = "segmentation off"
        elif self.picked is None:
            self._seg_on = False
            self._note = "pick a point first - the mask is built in a box around it"
        else:
            self._note = f"segmentation on (+/-{self._seg_half_um():,.0f} um box)"
            self._update_segmentation()
        self._refresh()

    def _toggle_segmentation(self):
        self._set_seg_enabled(not self._seg_on)

    # -- whole-tree mask ------------------------------------------------------ #
    def _say(self, message: str):
        """Put a message on screen *before* a blocking step, not after it."""
        self._note = message
        self._refresh()
        if self.plotter is not None:
            self.plotter.render()

    def _resident_mask(self):
        """The full-resolution array, decoding it on this first ask if need be.

        The decode reports progress into the status text, because 2 s with a frozen
        window and no explanation reads as a hang.
        """
        if self.mask is None:
            return None
        if not self.mask.ready:
            self._say("decoding the mask at full resolution...")

            def report(done, total, elapsed):
                self._say(f"decoding the mask: {done}/{total} planes ({elapsed:.0f}s)")

            return self.mask.get(progress=report)
        return self.mask.get()

    def set_seg_stride(self, stride: int):
        """Change the whole-tree stride, dropping the cached mesh but not rebuilding.

        Deliberately deferred: a spin box emits a value change per intermediate step,
        so going 4 -> 1 would otherwise contour at 3 and 2 on the way. The old mesh
        stays on screen until `rebuild_segmentation_all`, which is better than a
        blank view while you decide.
        """
        stride = max(int(stride), 1)
        if stride == self.seg_stride:
            return
        self.seg_stride = stride
        self._seg_all_mesh = None
        if self._seg_all_on:
            self._note = f"stride {stride} - press rebuild to apply it"
            self._refresh()
        self._layers_changed()

    def _clear_seg_all_actors(self):
        """Remove every whole-tree actor -- one per material, or none at all."""
        p = self.plotter
        for actor in self._seg_all_actor:
            if actor is not None and p is not None:
                p.remove_actor(actor, render=False)
        self._seg_all_actor = []

    def segmentation_all_stale(self) -> bool:
        """Has the mask been painted since the whole-tree mesh was built?"""
        if self._seg_all_mesh is None:
            return False
        edits = getattr(self.mask, "edits", None)
        if edits is None:
            return False
        return getattr(edits, "version", 0) != self._seg_all_version

    def rebuild_segmentation_all(self):
        """Rebuild the whole-tree isosurface from scratch.

        Also the "pick up what I just painted" action: the mesh is cached for the
        session, so a correction reaches it only here. Rebuilding automatically on
        every commit would mean a full contour per brush stroke.
        """
        self._seg_all_mesh = None
        self._set_seg_all_enabled(True)

    def _set_seg_all_enabled(self, on: bool):
        """Build or tear down the whole-tree isosurface.

        Built on first use and then kept: the decode is fast but the contour is
        not, and this is a fixed piece of geometry -- nothing about it depends on
        the pick, so rebuilding it per toggle would be pure waste.
        """
        p = self.plotter
        if p is None:
            return
        if self.labels is None:
            self._seg_all_on = False
            self._note = "no segmentation loaded"
            self._refresh()
            self._layers_changed()
            return

        self._seg_all_on = bool(on)
        if not self._seg_all_on:
            self._clear_seg_all_actors()
            self._note = "whole-tree segmentation off"
            self._refresh()
            self._layers_changed()
            return

        if self._seg_all_mesh is None:
            try:
                volume = self._resident_mask()
            except Exception as exc:  # noqa: BLE001 - fall back to decoding per use
                self._note = f"could not hold the mask resident ({exc}); decoding"
                volume = None
            regions = [m for m in self.materials if int(getattr(m, "value", 0)) != 0]
            what = (f"{len(regions)} materials" if len(regions) > 1 else "the mask")
            self._say(f"contouring {what} at stride {self.seg_stride}...")
            try:
                if len(regions) > 1:
                    # A `.Regions.am` naming Left_Tree and Right_Tree: one surface per
                    # material, so two coronaries that touch are still two objects.
                    self._seg_all_mesh = material_surfaces(
                        self.frame, self.labels, regions, self.seg_stride, volume=volume
                    )
                else:
                    _grid, surf = segmentation_volume(
                        self.frame, self.labels, self.seg_stride, volume=volume
                    )
                    self._seg_all_mesh = [(None, surf)] if surf is not None else []
            except Exception as exc:  # noqa: BLE001 - a failure here must not kill the window
                self._seg_all_mesh = None
                self._seg_all_on = False
                self._note = f"whole-tree mask failed: {exc}"
                self._refresh()
                self._layers_changed()
                return
            edits = getattr(self.mask, "edits", None)
            self._seg_all_version = getattr(edits, "version", 0) if edits is not None else 0

        drawn = [(m, s) for m, s in (self._seg_all_mesh or ())
                 if s is not None and s.n_cells]
        if not drawn:
            self._seg_all_on = False
            self._note = "the segmentation is empty"
        else:
            self._seg_all_actor = []
            for i, (material, surf) in enumerate(drawn):
                self._seg_all_actor.append(p.add_mesh(
                    surf,
                    color=material_color(material),
                    opacity=self._opacity["segmentation_all"],
                    smooth_shading=True,
                    name=f"_seg_all_iso_{i}",
                    reset_camera=False,
                    pickable=False,
                ))
            total = sum(s.n_cells for _m, s in drawn)
            named = ", ".join(f"{m.name} {s.n_cells:,}" for m, s in drawn if m)
            self._note = (
                f"whole-tree segmentation on "
                f"({total:,} triangles, stride {self.seg_stride})"
                + (f"\n  {named}" if named else "")
            )
        self._refresh()
        self._layers_changed()

    def _toggle_segmentation_all(self):
        self._set_seg_all_enabled(not self._seg_all_on)

    # -- what Avizo invented -------------------------------------------------- #
    def _interpolated_mask(self):
        """(P,) bool of invented points for the current graph, or ``None``.

        Cached per dataset. This is the one place in the toolkit that will *detect* when
        the file carries no flags: the layer is display-only, and a viewer that stayed
        silent about a bridge because nobody had run ``flag-interpolation`` yet would be
        hiding exactly what it exists to show. Every pipeline stage reads the stored
        field and only the stored field.
        """
        if self.graph is None:
            return None
        if self._interpolated is None:
            from .edit.interpolation import FIELD, flags_array

            self._interpolated_stored = FIELD in getattr(self.graph, "point_attrs", {})
            try:
                self._interpolated = flags_array(self.graph, detect_if_absent=True)
            except Exception as exc:  # noqa: BLE001 - a layer must not break the window
                print(f"[viewer3d] interpolation detection failed: {exc}")
                self._interpolated = np.zeros(int(self.graph.n_point), dtype=np.int64)
        return self._interpolated.astype(bool)

    def _jump_break_mask(self):
        """(P,) bool of jump steps for the current graph, or ``None``.

        Read from the file and never detected: the signature needs the segmentation
        labelled, and opening a graph should not quietly spend thirteen seconds
        decoding 2.34 GB to colour a line.
        """
        if self.graph is None:
            return None
        if self._jump_breaks is None:
            from .edit.interpolation import jump_breaks_array

            try:
                self._jump_breaks = jump_breaks_array(self.graph)
            except Exception as exc:  # noqa: BLE001 - a layer must not break the window
                print(f"[viewer3d] jump mask unavailable: {exc}")
                self._jump_breaks = np.zeros(int(self.graph.n_point), dtype=bool)
        return self._jump_breaks

    def interpolation_note(self) -> str:
        """One line for the status bar about where the flags came from."""
        flagged = self._interpolated_mask()
        if flagged is None or not flagged.any():
            return ""
        n = int(flagged.sum())
        if self._interpolated_stored:
            return f"{n} interpolated point(s), from the file"
        return (f"{n} interpolated point(s), detected here -- run `flag-interpolation` "
                "to record them")

    # -- what the tree is coloured by ---------------------------------------- #
    def color_modes(self) -> list[tuple[str, str]]:
        """``(key, label)`` of the colour modes *this* dataset can actually serve.

        A graph Avizo never ordered has no Strahler attribute, and the panel greys the
        mode out on this rather than offering a scale with nothing behind it.
        """
        if self.graph is None:
            return []
        flagged = self._interpolated_mask()
        return [
            (key, label)
            for key, label, _title in COLOR_MODES
            if color_kwargs(key, self.graph, flagged) is not None
        ]

    def color_by(self) -> str:
        """The scalar the centreline and the radius rings are currently mapped by."""
        return self._color_by

    def _color_kwargs(self) -> dict:
        """``add_mesh`` colour arguments for the current mode, falling back to radius.

        The fallback is not defensive noise: a dataset swap can land on a graph with no
        Strahler orders while the panel still says Strahler, and a colourless centreline
        there would be a worse answer than quietly showing the radius again.
        """
        flagged = self._interpolated_mask()
        kwargs = color_kwargs(self._color_by, self.graph, flagged)
        if kwargs is None:
            self._color_by = "radius"
            kwargs = color_kwargs("radius", self.graph, flagged)
        return kwargs or {
            "scalars": "radius_um",
            "cmap": "viridis",
            "scalar_bar_args": {"title": "radius (um)", "color": "white"},
        }

    def _remove_scalar_bars(self):
        """Drop whichever colour bar is on screen. Its title is the key it lives under."""
        p = self.plotter
        if p is None:
            return
        for title in COLOR_BAR_TITLES:
            if title in getattr(p, "scalar_bars", {}):
                p.remove_scalar_bar(title, render=False)

    def set_color_by(self, mode: str):
        """Colour the centreline and the radius rings by radius or by Strahler order.

        One setting drives both layers, because a ring is the cross-section *of* the
        centreline it sits on: two different maps in one scene would be read as one.
        """
        mode = str(mode)
        if mode not in {key for key, _label, _title in COLOR_MODES}:
            raise ValueError(f"unknown colour mode {mode!r}")
        if (self.graph is not None
                and color_kwargs(mode, self.graph, self._interpolated_mask()) is None):
            self._note = f"this graph carries no {mode} to colour by"
            self._refresh()
            self._layers_changed()
            return
        self._color_by = mode
        self._apply_color_by()

    def _apply_color_by(self):
        """Re-add the two coloured meshes under their own names, keeping their state.

        ``add_mesh`` replacing by name is how pyvista is meant to be restyled, but it
        builds a *fresh* actor, so visibility, opacity and pickability are re-applied
        here rather than inherited. The old colour bar goes first: bars are keyed by
        title, so switching modes would otherwise leave both stacked on screen.
        """
        p = self.plotter
        if p is None or self.graph is None:
            self._layers_changed()
            return
        self._remove_scalar_bars()

        if self._centreline_actor is not None and self._centreline_mesh is not None:
            visible = bool(self._centreline_actor.GetVisibility())
            self._centreline_actor = p.add_mesh(
                self._centreline_mesh,
                line_width=3,
                opacity=self._opacity["centreline"],
                name="centreline",
                reset_camera=False,
                **self._color_kwargs(),
            )
            self._centreline_actor.SetVisibility(visible)

        if (self._radius_circles_actor is not None
                and self._radius_circles_mesh is not None):
            self._radius_circles_actor = p.add_mesh(
                self._radius_circles_mesh,
                preference="cell",
                show_scalar_bar=False,
                line_width=2,
                opacity=self._opacity["radius_circles"],
                name="_radius_circles",
                reset_camera=False,
                lighting=False,
                pickable=False,
                **self._color_kwargs(),
            )
            self._radius_circles_actor.SetVisibility(self._radius_circles_on)

        # The pick contract lives on the centreline actor, which was just replaced.
        self._apply_pickable()
        labels = {key: label for key, label, _title in COLOR_MODES}
        self._note = f"coloured by {labels[self._color_by]}"
        self._refresh()
        self._layers_changed()

    # -- graph-wide radius circles ------------------------------------------ #
    def _set_radius_circles_enabled(self, on: bool):
        """Lazily build and show the ideal-radius ring at every graph point."""
        on = bool(on)
        if not self.layer_available("radius_circles"):
            self._radius_circles_on = False
            self._note = "no valid graph radii to draw"
            self._refresh()
            self._layers_changed()
            return

        self._radius_circles_on = on
        p = self.plotter
        if p is None:
            self._layers_changed()
            return

        if on and self._radius_circles_actor is None:
            if self._radius_circles_mesh is None:
                self._say("building radius circles for every graph point...")
                try:
                    self._radius_circles_mesh = radius_circle_polydata(
                        self.graph, flagged=self._interpolated_mask()
                    )
                except Exception as exc:  # noqa: BLE001 - never escape a Qt checkbox slot
                    self._radius_circles_on = False
                    self._note = f"radius circles failed: {exc}"
                    self._refresh()
                    self._layers_changed()
                    return

            if self._radius_circles_mesh.n_cells:
                self._radius_circles_actor = p.add_mesh(
                    self._radius_circles_mesh,
                    preference="cell",
                    show_scalar_bar=False,
                    line_width=2,
                    opacity=self._opacity["radius_circles"],
                    name="_radius_circles",
                    reset_camera=False,
                    lighting=False,
                    pickable=False,
                    **self._color_kwargs(),
                )
                self._apply_pickable()

        if self._radius_circles_actor is not None:
            self._radius_circles_actor.SetVisibility(on)
        self._note = (
            f"radius circles on ({self._radius_circles_mesh.n_lines:,} rings)"
            if on and self._radius_circles_mesh is not None
            else "radius circles off"
        )
        self._refresh()
        self._layers_changed()

    # -- the legend box ------------------------------------------------------- #
    def legend_available(self) -> bool:
        """Whether there is a key to show at all.

        `_populate` only builds one when something on screen carries a label, so a
        scene of nothing but the centreline has no legend to toggle.
        """
        return self._legend_actor is not None

    def legend_visible(self) -> bool:
        return self._legend_on

    def set_legend_visible(self, on: bool):
        """Show or hide the key in the corner of the render.

        Hidden rather than removed: `add_legend` derives the box from whatever is
        labelled *at the moment it is called*, so removing it would mean rebuilding it
        later from a scene that may have changed. The flag is the state that matters,
        and `_populate` applies it again to the box it builds for the next dataset.
        """
        self._legend_on = bool(on)
        if self._legend_actor is not None:
            self._legend_actor.SetVisibility(self._legend_on)
        self._note = "legend on" if self._legend_on else "legend off"
        self._refresh()
        self._layers_changed()

    def _toggle_legend(self):
        self.set_legend_visible(not self._legend_on)

    # -- saving the render as a figure ---------------------------------------- #
    def save_figure(self, path) -> str:
        """Write the current render to a vector file and return the path written.

        Two kinds of text sit over this scene and only one of them belongs in a
        figure. The keybinding block and the pick readout are *controls*: they say
        how to drive the window, and they name a pick that means nothing away from
        it. The legend box and the colour bar are the opposite -- they are the key
        to what is drawn, and a tree banded by Strahler order with no bar to say
        which band is which order cannot be read at all. So the two control overlays
        are hidden for the write and everything else is left exactly as it is on
        screen, including hidden: a legend switched off with ``L`` stays off, because
        the export is of the view you set up rather than of a different one.

        What lands in the file is not all vector, and cannot be. VTK exports through
        GL2PS, whose OpenGL2 backend writes 3D props as one embedded raster image --
        ``save_graphic``'s ``raster`` and ``painter`` flags make no difference to it,
        which is why neither is offered here. The overlays *are* real vector: the
        legend entries, the colour-bar title and its per-order annotations come out
        as ``<text>`` in the SVG, at whatever size the reader zooms to. That split is
        the useful one for a figure -- the geometry is a picture either way, and the
        labels are the part that has to stay sharp and stay editable.
        """
        p = self.plotter
        if p is None:
            raise RuntimeError("the 3D window is not open")
        out = Path(path)
        if not out.suffix:
            out = out.with_suffix(".svg")
        if out.suffix.lower() not in VECTOR_SUFFIXES:
            raise ValueError(
                f"{out.suffix!r} is not a vector format; "
                f"use one of {', '.join(VECTOR_SUFFIXES)}"
            )

        # Hidden by visibility rather than removed: `_label` is rebuilt by `_refresh`
        # on every pick and `_instructions_actor` belongs to the window rather than to
        # the dataset, so removing either would mean putting it back by hand. The
        # render in between is what the raster half of the file is captured from, so
        # it has to happen before the write, not after it.
        chrome = [a for a in (self._instructions_actor, self._label) if a is not None]
        was = [a.GetVisibility() for a in chrome]
        try:
            for actor in chrome:
                actor.SetVisibility(False)
            p.render()
            p.save_graphic(str(out), title=self.title)
        finally:
            for actor, visible in zip(chrome, was):
                actor.SetVisibility(visible)
            p.render()
        self._note = f"figure -> {out.name}"
        self._refresh()
        return str(out)

    # -- layer registry ------------------------------------------------------- #
    def set_extra_actors(self, key: str, actors) -> None:
        """Hand this window a layer built elsewhere, or clear it with ``None``.

        The `edit` subpackage owns the surface it regenerates and the handles it
        draws, but they belong in the same layer panel as everything else. New
        actors inherit the remembered opacity and the row's current visibility,
        which matters because they are torn down and rebuilt on every edit --
        the same reason `set_layer_opacity` remembers rather than only applies.
        """
        if key not in EXTRA_KEYS:
            raise KeyError(f"{key!r} is not an extra layer; add it to EXTRA_KEYS")
        was_visible = self.layer_visible(key) if self._extra_actors.get(key) else None
        actors = [a for a in (actors or []) if a is not None]
        if actors:
            self._extra_actors[key] = actors
        else:
            self._extra_actors.pop(key, None)
        default_vis = next(vis for k, _l, _o, vis in LAYERS if k == key)
        visible = default_vis if was_visible is None else was_visible
        for actor in actors:
            actor.GetProperty().SetOpacity(self.layer_opacity(key))
            actor.SetVisibility(visible)
        self._apply_pickable()
        self._layers_changed()
        if self.plotter is not None:
            self.plotter.render()

    def layer_actors(self, key: str) -> list:
        """Live actors behind a panel row. Empty when the layer is absent or off."""
        if key == "candidates":
            return list(self._cand_actors)
        if key in EXTRA_KEYS:
            return list(self._extra_actors.get(key, ()))
        if key == "interpolated":
            # One row, two actors. A Hermite fill and an unsampled jump are both "what
            # Avizo invented rather than measured", so hiding one without the other
            # would leave a misleading half of that answer on screen.
            return [a for a in (self._interpolated_actor, self._jump_actor)
                    if a is not None]
        if key in SHAPE_KEYS:
            actor = self._shape_actors.get(key)
        else:
            actor = {
                "surface": self._surface_actor,
                "centreline": self._centreline_actor,
                "radius_circles": self._radius_circles_actor,
                "plane": self._plane_actor,
                "segmentation": self._seg_actor,
            }.get(key)
        if key == "segmentation_all":
            # Several actors, one per material, so the opacity slider and the
            # visibility checkbox reach all of them rather than only the first.
            return [a for a in self._seg_all_actor if a is not None]
        return [actor] if actor is not None else []

    def layer_available(self, key: str) -> bool:
        """Whether this session has the input the layer needs at all."""
        if key == "surface":
            return self._surface_actor is not None
        if key == "centreline":
            # Not simply `True`: with no dataset loaded there is no actor behind it,
            # and the panel row would offer a slider that moved nothing.
            return self._centreline_actor is not None
        if key == "interpolated":
            # Greyed out on a clean graph, which is itself the useful reading: the row
            # being available at all means this tree has invented geometry in it --
            # either points Avizo wrote across a hole, or a step it wrote nothing along.
            return (self._interpolated_actor is not None
                    or self._jump_actor is not None)
        if key == "radius_circles":
            if self.graph is None:
                return False
            points = np.asarray(self.graph.points)
            radius = np.asarray(self.graph.thickness).ravel()
            return bool(
                len(points)
                and np.any(
                    np.isfinite(points).all(axis=1)
                    & np.isfinite(radius)
                    & (radius > 0.0)
                )
            )
        if key == "candidates":
            return bool(self._cand_actors)
        if key == "plane":
            return self.stack is not None and self.frame is not None
        if key in ("segmentation", "segmentation_all"):
            return self.labels is not None and self.frame is not None
        if key in EXTRA_KEYS:
            # Absent until the edit tools register something, which is how a
            # read-only session greys these rows out.
            return bool(self._extra_actors.get(key))
        if key in SHAPE_KEYS:
            # The shapes are handed over by the slice browser, so until 'v' has been
            # pressed there is nothing to show.
            return bool(self._shapes.get(key))
        return True

    def layer_visible(self, key: str) -> bool:
        if key == "plane":
            return self._plane_on
        if key == "segmentation":
            return self._seg_on
        if key == "segmentation_all":
            return self._seg_all_on
        if key == "radius_circles":
            return self._radius_circles_on
        if key in SHAPE_KEYS:
            return bool(self._shape_on.get(key))
        actors = self.layer_actors(key)
        return bool(actors) and bool(actors[0].GetVisibility())

    def layer_opacity(self, key: str) -> float:
        return float(self._opacity.get(key, 1.0))

    def set_layer_opacity(self, key: str, value: float):
        """Remember the opacity, and apply it to whatever is on screen now.

        Remembering is the half that matters: the image plane and the mask isosurface
        are rebuilt from scratch on every pick, so an opacity only pushed to the actor
        would silently snap back to the default the next time you picked.
        """
        value = float(np.clip(value, 0.0, 1.0))
        self._opacity[key] = value
        for actor in self.layer_actors(key):
            actor.GetProperty().SetOpacity(value)
        if self.plotter is not None:
            self.plotter.render()

    def set_layer_visible(self, key: str, on: bool):
        """Show or hide a layer. The two overlays are built/torn down, not just hidden."""
        if key == "plane":
            self._set_plane_enabled(on)  # notifies the panel itself
        elif key == "segmentation":
            self._set_seg_enabled(on)
        elif key == "segmentation_all":
            self._set_seg_all_enabled(on)
        elif key == "radius_circles":
            self._set_radius_circles_enabled(on)
        elif key in SHAPE_KEYS:
            self._shape_on[key] = bool(on)
            self._update_shape_actors()
            self._layers_changed()
        else:
            for actor in self.layer_actors(key):
                actor.SetVisibility(bool(on))
            if self.plotter is not None:
                self.plotter.render()
            self._layers_changed()

    def _layers_changed(self):
        """Tell the panel to re-read state, so the keys and its checkboxes agree."""
        if self.on_layers_changed is not None:
            self.on_layers_changed()

    def _apply_pickable(self):
        """Restrict picking to the graph. Must run after every actor has been added."""
        p = self.plotter
        if p is None:
            return
        actors = [a for a in [self._centreline_actor, *self._cand_actors] if a is not None]
        if self._surface_pickable and self._surface_actor is not None:
            actors.append(self._surface_actor)
        p.pickable_actors = actors

    def _toggle_surface_pick(self):
        if self._surface_actor is None:
            self._note = "no surface loaded"
            self._refresh()
            return
        self._surface_pickable = not self._surface_pickable
        self._apply_pickable()
        self._note = (
            "surface picking ON - clicks may land anywhere on the shell"
            if self._surface_pickable
            else "surface picking off - clicks snap to the centreline"
        )
        self._refresh()

    def _goto_candidate(self, step: int):
        if not self.cands:
            self._note = "no candidates loaded"
            self._refresh()
            return
        self._select_candidate((self._cand_i + step) % len(self.cands), recentre=True)

    def _select_candidate(self, index: int, recentre: bool = False):
        """Make candidate ``index`` the active pick.

        Walking the list with ``n``/``b`` flies the camera there; clicking a marker that
        is already on screen must not, or the view jumps out from under the mouse.
        """
        if not 0 <= index < len(self.cands):
            return
        c = self.cands[index]
        self._set_pick(c.xyz, source="candidate", cand_i=index)
        p = self.plotter
        if recentre and p is not None:
            span = max(4.0 * max(c.radius_um, c.partner_radius_um, 200.0), 2000.0)
            p.camera.focal_point = tuple(c.xyz)
            p.camera.position = tuple(c.xyz + np.array([span * 3, span * 3, span * 2]))
            p.reset_camera_clipping_range()
            p.render()

    def _open(self):
        if self.picked is None:
            self._note = "nothing picked yet - double-click a vessel"
            self._refresh()
            return
        if self.on_open is None:
            return
        self._note = "loading slices..."
        self._refresh()

        # Hand off to the Qt event loop rather than building the slice viewer here.
        # This runs inside VTK's key-event handler, with VTK's OpenGL context current;
        # creating napari's vispy canvas at that moment initialises GL against the
        # wrong context and takes the process down with an access violation. Deferring
        # by one event-loop turn lets VTK finish and release the context first.
        from qtpy.QtCore import QTimer

        pt = self.picked.copy()
        QTimer.singleShot(0, lambda: self._run_open(pt))

    def _run_open(self, pt):
        try:
            self.on_open(pt)
            self._note = ""
        except Exception as exc:  # noqa: BLE001 - a bad pick must not kill the session
            traceback.print_exc()
            self._note = f"slice viewer failed: {type(exc).__name__}: {exc}"
        self._refresh()

    # -- run ---------------------------------------------------------------- #
    def build(self):
        """Create the background plotter and populate it. Returns the plotter."""
        p = self._build_window()
        self._populate(reset_camera=True)
        self._refresh()
        return p

    def _build_window(self):
        """Everything that does not depend on which dataset is loaded.

        Split out from `build` so a second dataset can be swapped in without
        destroying the window: the plotter, its docks, the key bindings and the
        camera all belong to the session, not to the graph currently in it.
        """
        p = (self.plotter_factory or _background_plotter)(self.title)
        self.plotter = p
        p.set_background("#101014")

        self._instructions_actor = p.add_text(
            INSTRUCTIONS, position="upper_left", font_size=9, color="#b0b0c0"
        )

        # Picking is wired by hand rather than through `enable_point_picking`, whose only
        # two triggers are a pick on every *left* button press -- which is also how the
        # camera starts a rotation, so every drag to spin the view drops a pick where the
        # drag began -- or on every right press, which is the dolly drag, same problem on
        # the other button. Everything else that call does is public API.
        p.iren.picker = "point"  # vtkPointPicker: snaps to vertices and reports their id
        p.iren.picker.SetTolerance(PICK_TOLERANCE)
        p.iren.add_pick_observer(self._on_pick)
        p.track_click_position(self._pick_here, side="left", double=True, viewport=True)

        # pyvista binds its own defaults on some of these, and `add_key_event` *appends*
        # rather than replaces ("These are non-unique - thus a key could map to many
        # callback functions"), so ours would run in addition to:
        #   'v' -> isometric_view_interactive(), which snaps the camera to a fixed
        #          viewpoint and throws away the view you had lined up;
        #   'b' -> installs another LeftButtonPressEvent observer on every press.
        # `clear_events_for_key` is a no-op for keys with nothing bound. 'q' (pyvista's
        # close) and 'r' (VTK's own reset-camera) are left alone: both are wanted.
        for key in ("v", "n", "b", "s", "c", "i", "g", "a", "L"):
            p.clear_events_for_key(key)

        p.add_key_event("v", self._open)
        p.add_key_event("a", self._toggle_segmentation_all)
        p.add_key_event("n", lambda: self._goto_candidate(1))
        p.add_key_event("b", lambda: self._goto_candidate(-1))
        p.add_key_event("s", self._toggle_surface_pick)
        p.add_key_event("i", self._toggle_slice_plane)
        p.add_key_event("g", self._toggle_segmentation)
        p.add_key_event("c", self._clear)
        # Shift+L rather than a bare letter: every free lower-case key is either
        # taken by the edit tools (see `edit.controller`) or by VTK's own char
        # handling -- 'w', for one, wireframes the whole scene on the way past.
        p.add_key_event("L", self._toggle_legend)

        self._dock_panel(p)
        return p

    def _populate(self, *, reset_camera=False):
        """Add the actors that come from the current dataset.

        ``reset_camera`` is passed explicitly because pyvista's default -- reset if
        this is the first actor -- is right on the first build and wrong on every
        swap after it, where the camera is a view the user has already chosen.
        """
        p = self.plotter
        if p is None:
            return

        if self.mesh is not None and self.mesh.n_cells:
            self._surface_actor = p.add_mesh(
                self.mesh,
                color=SURFACE_COLOR,
                opacity=self._opacity["surface"],
                smooth_shading=True,
                name="surface",
                label="surface",
                reset_camera=reset_camera,
            )

        if self.graph is not None:
            flagged = self._interpolated_mask()
            breaks = self._jump_break_mask()
            self._centreline_mesh = centreline_polydata(self.graph, flagged, breaks)
            self._centreline_actor = p.add_mesh(
                self._centreline_mesh,
                line_width=3,
                name="centreline",
                reset_camera=reset_camera,
                **self._color_kwargs(),
            )
            if breaks is not None and breaks.any():
                jumps = jump_polydata(self.graph, breaks)
                if jumps.n_lines:
                    self._jump_actor = p.add_mesh(
                        jumps,
                        color=JUMP_COLOR,
                        line_width=2,
                        opacity=self._opacity["interpolated"],
                        name="unsampled_jumps",
                        label=f"unsampled jumps ({int(breaks.sum())})",
                        reset_camera=False,
                        lighting=False,
                        pickable=False,
                    )
            if flagged is not None and flagged.any():
                bridges = interpolation_polydata(self.graph, flagged)
                if bridges.n_lines:
                    self._interpolated_actor = p.add_mesh(
                        bridges,
                        color=INTERPOLATED_COLOR,
                        line_width=2,
                        opacity=self._opacity["interpolated"],
                        name="interpolated",
                        label=f"interpolated ({int(flagged.sum())} pts)",
                        reset_camera=False,
                        lighting=False,
                        pickable=False,
                    )

        seen = []
        for kind, color in CANDIDATE_COLOR.items():
            sub = [i for i, c in enumerate(self.cands) if c.kind == kind]
            if not sub:
                continue
            cloud = pv.PolyData(np.array([self.cands[i].xyz for i in sub], dtype=np.float32))
            actor = p.add_mesh(
                cloud,
                color=color,
                point_size=16,
                render_points_as_spheres=True,
                name=f"cand_{kind}",
                label=f"{kind} ({len(sub)})",
                reset_camera=False,
            )
            # Point id within this cloud -> index into self.cands, so a clicked marker
            # can select its own candidate.
            self._cand_actors[actor] = sub
            seen.append(kind)
        # The grey bridges carry a label too, and a flat grey line in a scene coloured by
        # radius needs the legend to say what it is more than a candidate marker does.
        if seen or self._interpolated_actor is not None or self._jump_actor is not None:
            self._legend_actor = p.add_legend(bcolor="#202028", face="circle")
            self._legend_actor.SetVisibility(self._legend_on)

        # Last: the setter sweeps the renderer's current actors and unsets the pickable
        # flag on everything it is not given.
        self._apply_pickable()

    def _teardown_dataset(self):
        """Remove every actor built from the current dataset, and forget its state.

        Two things here are not covered by pyvista's own name-based replacement, and
        both leave the previous dataset visible on screen if they are skipped:

        * the candidate actors are named ``cand_<kind>``, so a kind that the *next*
          dataset does not have is never re-added and therefore never replaced;
        * ``add_legend`` replaces itself only when it is called, and it is not called
          at all when there are no candidates.
        """
        p = self.plotter
        if p is not None:
            for actor in (self._surface_actor, self._centreline_actor,
                          self._interpolated_actor,
                          self._radius_circles_actor, self._marker,
                          self._label, self._plane_actor, self._seg_actor,
                          *self._seg_all_actor, *self._cand_actors,
                          *self._shape_actors.values(),
                          *self._route_actors.values(), *self._crop_actors,
                          *self._reformat_actors, *self._reformat_stack_actors,
                          *self._reformat_current_actors, *self._section_actors,
                          self._jump_actor):
                if actor is not None:
                    p.remove_actor(actor, render=False)
            p.remove_legend(render=False)
            self._remove_scalar_bars()
        # The actor goes with the dataset that labelled it; `_legend_on` does not,
        # because it is a preference about the window rather than state read off a graph.
        self._legend_actor = None

        self._surface_actor = None
        self._centreline_actor = None
        self._centreline_mesh = None
        self._interpolated_actor = None
        self._interpolated = None
        self._interpolated_stored = False
        self._jump_actor = None
        self._jump_breaks = None
        self._radius_circles_actor = None
        self._radius_circles_mesh = None
        self._radius_circles_on = False
        self._cand_actors = {}
        self._marker = None
        self._label = None
        self._plane_actor = None
        self._seg_actor = None
        self._seg_all_actor = []
        self._seg_all_mesh = None
        self._seg_all_version = 0
        self._shape_actors = {}
        # A review route belongs to the dataset it was measured on, so it goes with
        # it. Left behind, it would be drawn over a different tree at coordinates
        # that mean nothing there.
        self._route_actors = {}
        # Same for a crop preview: it is a selection over *this* graph's segments, and
        # nothing about it carries over to the next one. The reformat selection is the
        # same kind of thing, and its sampled planes are images cut out of the dataset
        # that is going away, so both go for the same reason.
        self._crop_actors = []
        self._reformat_actors = []
        self._reformat_stack_actors = []
        self._reformat_current_actors = []
        self._reformat_current = None
        # Sections are cut out of *this* dataset's mask at *this* graph's points, so
        # they mean nothing over the next one -- the same argument again.
        self._section_actors = []
        # Actors owned by the edit tools go with their controller, which disposes of
        # itself through `on_dataset_changing` -- removing them here would leave it
        # holding dangling VTK pointers.
        self._extra_actors = {}

        # A toggle left on would rebuild its overlay from the *new* inputs at the
        # *old* slice, with the panel checkbox agreeing that it is on.
        self.picked = None
        self.pick_source = ""
        self._cand_i = -1
        self._point_i = -1
        self._note = ""
        self._slice_z = -1
        self._plane_z = -1
        self._plane_on = False
        self._seg_on = False
        self._seg_all_on = False
        self._surface_pickable = False
        self._shapes = {key: {} for key in SHAPE_KEYS}

    def set_dataset(self, *, graph=_KEEP, mesh=_KEEP, cands=_KEEP, frame=_KEEP,
                    stack=_KEEP, labels=_KEEP, seg_box_um=_KEEP, seg_stride=_KEEP,
                    mask=_KEEP, materials=_KEEP, reset_camera=False):
        """Swap in a new dataset, keeping the window, its docks and its camera.

        Only the inputs named are changed; the rest are kept. That matters for more
        than tidiness -- the common case, stepping through a repair chain, changes
        the graph alone, and rebuilding the whole scene there would throw away the
        whole-tree mask mesh (~1 s to rebuild) for nothing.

        Pass ``graph=None`` for the empty state. Nothing loaded yet, a load that
        failed and the moment between two datasets are then the same code path,
        which is the one that gets exercised.
        """
        if self.on_dataset_changing is not None:
            self.on_dataset_changing()

        self._teardown_dataset()

        if graph is not _KEEP:
            self.graph = graph
        if mesh is not _KEEP:
            self.mesh = mesh
        if cands is not _KEEP:
            self.cands = list(cands or [])
        if frame is not _KEEP:
            self.frame = frame
        if stack is not _KEEP:
            self.stack = stack
        if labels is not _KEEP:
            self.labels = labels
        if mask is not _KEEP:
            self.mask = mask
        if materials is not _KEEP:
            self.materials = tuple(materials or ())
        if seg_box_um is not _KEEP:
            self.seg_box_um = float(seg_box_um)
        if seg_stride is not _KEEP:
            self.seg_stride = max(int(seg_stride), 1)

        self._set_derived()
        self._populate(reset_camera=reset_camera)
        self._refresh()
        self._layers_changed()

    def _set_derived(self):
        """Recompute what is cached off the graph and the candidate list."""
        self._edge_of_point = (
            self.graph.edge_of_point() if self.graph is not None else np.empty(0, dtype=int)
        )
        self._cand_xyz = (
            np.array([c.xyz for c in self.cands], dtype=float)
            if self.cands
            else np.empty((0, 3))
        )

    def _dock_panel(self, p):
        """Dock the layer controls, if this plotter has a window to dock them in.

        A plain ``pv.Plotter`` -- what the test harness substitutes -- has no
        ``app_window``, so the panel is simply skipped there.
        """
        if not hasattr(p, "app_window"):
            return
        from qtpy.QtCore import Qt
        from qtpy.QtWidgets import QDockWidget

        from . import controls3d

        self.panel = controls3d.build_layer_panel(self)
        dock = QDockWidget("layers", p.app_window)
        dock.setWidget(self.panel)
        p.app_window.addDockWidget(Qt.RightDockWidgetArea, dock)
