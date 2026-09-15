"""The mutable half of an interactive session.

`interactive` used to keep its state in a closure -- `open_slices` captured the
builder, the args, the volume source, the paint session and the frame. That is fine
for a process that loads one dataset and exits, and it is the whole obstacle to
loading a second: **a closure cannot be rebound.** Swapping the dataset would leave
`open_slices` serving slabs out of the previous one's builder, against the previous
one's coordinate frame, and there is no way to reach in and fix it.

So the captured variables become fields on an object that the `Picker3D` outlives.
`picker.on_open = app.open_slices` is a bound method, and a bound method reads
`self.session` at the moment it runs rather than at the moment it was defined.

Nothing here knows about Qt. The panels drive it, and `main.interactive` is now a
thin wrapper around it, so the command line behaves exactly as it did.
"""

from __future__ import annotations

import gc
from pathlib import Path

from . import viewer2d


class ViewerApp:
    """Owns the loaded dataset and everything derived from it."""

    def __init__(self, args, session=None, picker=None):
        self.args = args
        self.session = session
        self.picker = picker
        self.builder = session.builder() if session is not None else None
        self.viewer = None  # the napari window, created lazily on the first pick
        #: The reformat window, its own viewer because its arrays are in
        #: (plane, v, u) rather than raw (slice, row, col). Created on the first build.
        self.reformat_viewer = None
        self.reformat = None  # the stack it is currently showing
        self.volume = None
        self.paint = None
        self.controller = None
        self.cands: list = []
        #: Called with a status string whenever something worth reporting happens.
        self.on_status = None
        #: Called after a dataset is loaded, unloaded, or fails to load.
        self.on_dataset = None

    # -- reporting ---------------------------------------------------------

    def status(self, message: str) -> None:
        print(message)
        if self.on_status is not None:
            self.on_status(message)

    @property
    def loaded(self) -> bool:
        return self.session is not None

    def describe(self) -> str:
        if self.session is None:
            return "no dataset loaded"
        from .main import graph_paths

        g = self.session.graph
        paths = graph_paths(self.args.graph)
        named = ", ".join(Path(p).name for p in paths) or "no skeleton"
        trees = getattr(self.session, "graph_trees", None) or []
        if len(paths) > 1:
            named += f" ({sum(len(t) for t in trees)} trees)"
        return (f"{named}: {g.n_vertex} vertices, "
                f"{g.n_edge} edges, {g.n_point} points")

    # -- the pick callback -------------------------------------------------

    def open_slices(self, xyz_um) -> None:
        """Build a slab at this point and show it in the slice browser.

        Was the `open_slices` closure in `main.interactive`, unchanged in behaviour.
        """
        import napari

        if self.session is None or self.builder is None:
            self.status("no dataset loaded")
            return

        args = self.args
        slab = self.builder.build(
            xyz_um,
            half_slices=args.slab,
            roi_px=args.roi,
            with_probability=args.probability,
            with_tube=not args.no_tube,
            with_surface=not args.no_surface,
        )
        print(f"  slices {slab.z_lo}-{slab.z_hi - 1}, "
              f"{len(slab.contours or [])} contours, {len(slab.skel_pts)} centreline points")

        v = self.viewer
        # A viewer the user closed still exists as an object; detect it and start fresh.
        if v is None or not viewer_alive(v):
            v = napari.Viewer(title="HiP-CT segmentation debugger - slices")
            self.viewer = v
        # The 3D view mirrors the same per-slice shapes rather than recomputing them,
        # so the two windows cannot disagree about what they are drawing.
        if self.picker is not None:
            self.picker.set_slice_shapes(slab)
        # Scrolling the z slider drags the 3D image plane and those shapes along with
        # it. Deferred, because this fires inside a vispy event with napari's GL
        # context current.
        viewer2d.build_layers(
            v, slab, self.session.frame,
            on_slice=self._on_slice,
            volume=self.volume,
            paint=self.paint,
        )
        v.window.activate()

    def _on_slice(self, z) -> None:
        if self.picker is not None:
            self.picker.set_current_slice(z, defer=True)

    # -- the reformat window -----------------------------------------------

    def open_reformat(self, stack, *, sections_in_3d: bool = False,
                      in_world: bool = True) -> None:
        """Show a reformat stack, reusing the window across builds.

        Its own viewer rather than the slice browser's: the arrays are in
        ``(plane index, v, u)``, and every placement rule in `viewer2d` is built around
        raw ``(slice, row, col)``.

        ``in_world=False`` shows the images **without** the 3D overlays. That is the
        case a stack loaded from disk lands in when there is no dataset open, or when
        the one that is open was cut in a different coordinate frame: the pixels are
        self-contained and worth looking at, but the world positions are not meaningful
        here, so drawing plane frames somewhere unrelated would be worse than drawing
        nothing. `reformat_io.check_against` is what decides.
        """
        import napari

        from . import viewer_reformat

        v = self.reformat_viewer
        # A viewer the user closed still exists as an object; detect it and start fresh.
        if v is None or not viewer_alive(v):
            v = napari.Viewer(title=viewer_reformat.TITLE)
            self.reformat_viewer = v
        self.reformat = stack if in_world else None
        if in_world:
            self._show_reformat_in_3d(stack, sections=sections_in_3d)
        else:
            clear = getattr(self.picker, "clear_reformat", None)
            if clear is not None:
                try:
                    clear()
                except Exception:  # noqa: BLE001 - a stale preview must not block this
                    pass
        viewer_reformat.build_layers(v, stack, on_plane=self._on_reformat_plane)
        v.window.activate()

    def _show_reformat_in_3d(self, stack, *, sections: bool = False) -> None:
        """Put the sampling planes into the 3D window beside the tree they came from.

        Two cheap things by default. The **frames** -- one actor, one polyline per plane,
        no images at all -- show where the stack cut and how it is oriented, and are the
        only place a plane collision is visible. The **current section** is the one
        cross-section the reformat window is showing, textured, and it follows the
        napari slider through :meth:`_on_reformat_plane`.

        ``sections`` additionally draws a decimated set of the cross-sections as
        textured quads. Off by default and built only when asked: each one is its own
        actor and its own texture upload, and paying for a couple of dozen of them on
        every build -- for a layer that starts hidden -- is work nobody asked for. The
        moving single section is what makes the two windows readable against each other.
        """
        from . import reformat as reformat_mod
        from .viewer3d import REFORMAT_MAX_TEXTURED

        show = getattr(self.picker, "show_reformat_stack", None)
        if show is None:
            return  # no 3D window in this session (a headless run, or a test harness)
        corners = reformat_mod.plane_corners(stack)
        drawn = []
        if sections:
            picks = reformat_mod.texture_indices(len(corners), REFORMAT_MAX_TEXTURED)
            drawn = [(corners[i], stack.raw[i]) for i in picks]
        middle = len(corners) // 2
        try:
            show(corners, drawn, (corners[middle], stack.raw[middle]))
        except Exception as exc:  # noqa: BLE001 - a preview must not cost the stack
            self.status(f"could not draw the reformat in 3D: {type(exc).__name__}: {exc}")

    def _on_reformat_plane(self, i: int) -> None:
        """Follow the reformat slider in the 3D window.

        Two things move: the axial image plane goes to the raw slice this cross-section
        was cut from, and the current-section quad walks to this plane. Deferred,
        because this fires inside a vispy event with napari's GL context current -- the
        same reason `_on_slice` defers.
        """
        from . import reformat as reformat_mod

        stack = self.reformat
        if self.picker is None or self.session is None or stack is None:
            return
        if not 0 <= i < len(stack.centreline.coords_um):
            return
        z = int(self.session.frame.um_to_raw_index(stack.centreline.coords_um[i])[0][0])
        self.picker.set_current_slice(z, defer=True)

        move = getattr(self.picker, "set_reformat_section", None)
        if move is None:
            return

        def run():
            try:
                corners = reformat_mod.plane_corners(stack, [i])[0]
                move(corners, stack.raw[i])
            except Exception:  # noqa: BLE001 - a preview must not break scrolling
                pass

        _defer(run)

    # -- swapping datasets -------------------------------------------------

    def can_swap(self) -> list[str]:
        """Unsaved work that a dataset change would destroy.

        Not a nicety. `main.interactive`'s `finally` calls losing painted edits
        "unforgivable" and is the only thing protecting them today; a Load button
        that walked past it would be a regression, not a feature.
        """
        blocking = []
        if self.paint is not None:
            edits = self.paint.edits
            if not edits.is_empty and not self.paint.edits_path:
                blocking.append(f"{edits.describe()} with no --edits path to save them to")
        if self.controller is not None:
            history = getattr(self.controller.graph, "history", None)
            if history is not None and getattr(history, "can_undo", False):
                blocking.append("unexported skeleton edits")
        return blocking

    def unload(self) -> None:
        """Tear down everything derived from the current dataset.

        The napari viewer is *closed* rather than cleared: `viewer2d._viewer_state`
        deliberately hangs its per-viewer state off the Qt window, because that is
        exactly the lifetime the state should have. Clearing layers in place would
        leave a builder-shaped hole in it. The next pick recreates the window through
        the path that already exists for a viewer the user closed.
        """
        if self.controller is not None:
            try:
                self.controller.dispose()
            except Exception as exc:  # noqa: BLE001 - teardown must not block a load
                self.status(f"edit controller did not dispose cleanly: {exc}")
            self.controller = None

        if self.paint is not None:
            self.paint.commit()
            saved = self.paint.save()
            if saved is not None:
                self.status(f"[paint] wrote {saved}  ({self.paint.edits.describe()})")
            self.paint = None

        for attr in ("viewer", "reformat_viewer"):
            window = getattr(self, attr)
            if window is not None:
                try:
                    if viewer_alive(window):
                        window.close()
                except Exception:  # noqa: BLE001
                    pass
                setattr(self, attr, None)
        # A reformat is a stack of images cut out of *this* dataset; it means nothing
        # against the next one, and it is tens of megabytes.
        self.reformat = None

        self.builder = None
        self.volume = None
        self.cands = []
        if self.session is not None:
            self.session.close()
            self.session = None
        # 2.34 GB of compressed lattice is only freed when the last reference goes,
        # and the load that follows is about to allocate another one.
        gc.collect()

        if self.picker is not None:
            # `mask=None` matters: the picker holding the previous dataset's
            # resident 2.34 GB would make the collect() above achieve nothing.
            self.picker.set_dataset(graph=None, mesh=None, cands=[], frame=None,
                                    stack=None, labels=None, mask=None)
        self._dataset_changed()

    def load(self, args=None, *, session=None, reset_camera=True) -> bool:
        """Replace the dataset. Returns True if it loaded.

        Constructing a fresh `Session` rather than re-entering `__init__` on the old
        one: 3 s is cheap next to a re-enterable loader that would be a second
        implementation of the only path the tests cover.
        """
        from .main import Session

        if args is not None:
            self.args = args

        self.unload()
        try:
            # Everything, not just the constructor: a lattice whose header parses but
            # whose payload does not, or a `--paint` run with an edits file built for
            # different dimensions, fails *after* `Session.__init__` returns. Letting
            # that escape leaves a half-loaded app behind a button click.
            self.session = session if session is not None else Session(self.args)
            self.builder = self.session.builder()
            self._apply_volume()
            self._apply_paint()
        except Exception as exc:  # noqa: BLE001 - a bad path must not kill the window
            self.status(f"could not load: {type(exc).__name__}: {exc}")
            self.session = None
            self.builder = None
            self.volume = None
            self.paint = None
            self._dataset_changed()
            return False

        if self.picker is not None:
            self.picker.set_dataset(
                graph=self.session.graph,
                mesh=self.session.mesh,
                cands=self.cands,
                frame=self.session.frame,
                stack=self.session.stack,
                labels=self.session.labels,
                mask=self.session.mask,
                materials=self.session.materials,
                seg_box_um=self.args.seg_box_um,
                seg_stride=self.args.seg_stride,
                reset_camera=reset_camera,
            )
        self.status(self.describe())
        self._dataset_changed()
        return True

    def reload_graph(self, path) -> bool:
        """Swap the graph alone -- the common case, and 0.4 s rather than 3 s.

        Stepping through a repair chain changes nothing but the graph. Rebuilding the
        whole session there would drop the decoded-slice caches and the whole-tree
        mask mesh for no reason.
        """
        from .main import graph_paths

        if self.session is None:
            return self.load()
        try:
            self.session.reload_graph(path)
        except Exception as exc:  # noqa: BLE001
            self.status(f"could not read {path}: {type(exc).__name__}: {exc}")
            return False
        # What was asked for, normalised -- `--graph` is a list of paths now, and the
        # Data tab's ';'-separated field arrives here as one string.
        self.args.graph = graph_paths(path)
        self.builder = self.session.builder()
        if self.controller is not None:
            # Its EditableGraph and its 88 s SDF session both belong to the old graph.
            self.controller.dispose()
            self.controller = None
        if self.picker is not None:
            self.picker.set_dataset(graph=self.session.graph, cands=[])
        self.cands = []
        self.status(self.describe())
        self._dataset_changed()
        return True

    def reload_surface(self, path) -> bool:
        if self.session is None:
            return False
        try:
            self.session.reload_surface(path)
        except Exception as exc:  # noqa: BLE001
            self.status(f"could not read {path}: {type(exc).__name__}: {exc}")
            return False
        self.args.surface = str(path)
        self.builder = self.session.builder()
        if self.picker is not None:
            self.picker.set_dataset(mesh=self.session.mesh)
        self._dataset_changed()
        return True

    def set_candidates(self, cands) -> None:
        self.cands = list(cands or [])
        if self.picker is not None:
            self.picker.set_dataset(cands=self.cands)

    def _dataset_changed(self) -> None:
        if self.on_dataset is not None:
            self.on_dataset(self)

    # -- optional features -------------------------------------------------

    def _apply_volume(self) -> None:
        if not getattr(self.args, "volume", False) or self.session is None:
            return
        from .volume import VolumeSource

        # Lazy dask graphs over the whole dataset -- built once, no data touched.
        self.volume = VolumeSource.build(
            self.session.stack, self.session.labels, self.session.frame
        )
        print(self.volume.describe())

    def _apply_paint(self) -> None:
        if not getattr(self.args, "paint", False) or self.session is None:
            return
        from .edit.paint import PaintSession

        args = self.args
        self.paint = PaintSession(self.session.labels, self.session.frame,
                                  size_vox=args.paint_box, edits_path=args.edits)
        mb = args.paint_box ** 3 / 1e6
        print(f"Painting enabled: {args.paint_box}^3 voxel block ({mb:.1f} MB) "
              f"around each pick"
              + (f", edits -> {args.edits}" if args.edits else
                 ", no --edits path (use 'Save edits...' before quitting)"))

    def enable_paint(self) -> bool:
        """Turn painting on for a session that did not start with `--paint`.

        The mask has to be re-read through a `MaskSource` before anything can
        composite corrections onto it, so this reloads. Cheaper than it sounds and
        much safer than retrofitting the wrapper under a live builder.
        """
        if self.session is None:
            self.status("load a dataset first")
            return False
        if self.paint is not None:
            self.status("painting is already on")
            return True
        self.args.paint = True
        return self.load(reset_camera=False)

    def enable_edit(self) -> bool:
        """Prepare the SDF session and attach the edit controller.

        ~88 s the first time, which is why it stays an explicit action rather than
        something a dataset load does on your behalf.
        """
        if self.session is None:
            self.status("load a dataset first")
            return False
        if self.controller is not None:
            self.status("edit mode is already available")
            return True

        # Imported here, not at module scope: the edit tools pull in coronary_sdf,
        # which is not importable in a plain read-only session.
        from .edit.adapter import from_spatial_graph
        from .edit.controller import INSTRUCTIONS as EDIT_KEYS
        from .edit.controller import EditController
        from .edit.sdfpatch import SdfSession

        self.status("Preparing the SDF session (once; about 90 seconds)...")
        triple = from_spatial_graph(self.session.graph)
        sdf = SdfSession(triple, voxel_size_mm=self.args.edit_voxel_mm)
        self.controller = EditController(
            self.picker, triple, session=sdf, min_extent_um=self.args.edit_box_um,
            paint=self.paint, source=self.session.labels, frame=self.session.frame,
        )
        self.controller.attach()
        self.args.edit = True
        print(f"  {sdf.n_capsules:,} capsules, voxel {sdf.voxel_size_mm * 1000:.0f} um, "
              f"prepared in {sdf.prepare_seconds:.1f}s")
        print("Edit mode available. Press 'e' to enable, then:")
        for line in EDIT_KEYS.splitlines():
            print("  " + line)
        print()
        self.status(f"edit mode ready ({sdf.n_capsules:,} capsules)")
        return True

    # -- shutdown ----------------------------------------------------------

    def finish(self) -> None:
        """What `interactive`'s `finally` used to do, verbatim in intent."""
        if self.controller is not None:
            self.controller.detach()
        if self.paint is not None:
            # Painting is manual work; losing it because the window closed would be
            # unforgivable, so commit unconditionally and say where it went.
            self.paint.commit()
            saved = self.paint.save()
            if saved is not None:
                print(f"[paint] wrote {saved}  ({self.paint.edits.describe()})")
            elif not self.paint.edits.is_empty:
                print(f"[paint] WARNING: {self.paint.edits.describe()} were never saved "
                      f"(no --edits path was given)")


def _defer(fn) -> None:
    """Run ``fn`` once the current event has finished, or now if there is no Qt loop.

    Building VTK actors from inside a vispy event -- which is where a napari slider
    callback runs, with napari's GL context current -- is an access violation rather
    than an exception. `viewer3d.set_current_slice` defers for exactly this reason;
    anything else the same callback touches has to as well.

    The live-application check is not belt and braces. ``QTimer.singleShot`` with no
    running ``QApplication`` neither raises nor fires -- it drops the callback on the
    floor -- so without it a headless caller would silently get nothing. And with no Qt
    loop there is no event to defer out of, which is what made deferring necessary.
    """
    try:
        from qtpy.QtCore import QTimer
        from qtpy.QtWidgets import QApplication

        if QApplication.instance() is not None:
            QTimer.singleShot(0, fn)
            return
    except Exception:  # noqa: BLE001 - no Qt binding at all
        pass
    fn()


def viewer_alive(viewer) -> bool:
    try:
        return bool(viewer.window._qt_window.isVisible())
    except Exception:
        return False
