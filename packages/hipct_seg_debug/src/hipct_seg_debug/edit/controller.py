"""Wires the edit tools into the 3D pick window.

Owns the editable graph, the SDF session and the rebuild queue, and translates
between what the viewer reports (a picked vertex id) and what the model needs (a
point id, a segment, a box to rebuild).

Three things about the host window shape this module:

* **A pick already resolves to a graph point.** ``viewer3d`` wires a
  ``vtkPointPicker`` restricted to the centreline cloud, and the polydata it
  renders is built from ``graph.points`` in order, so the pick id *is* the index
  into that array (guarded by ``selftest.test_centreline_point_ids``). Selection
  therefore costs nothing -- there is no hit-testing to write.
* **Nothing may be created from inside a VTK callback.** ``viewer3d.py:833-841``
  documents the crash: building Qt or vispy objects while VTK's GL context is
  current takes the process down with an access violation, not an exception.
  Every key handler here therefore only mutates the model and posts work.
* **Actors are torn down and rebuilt constantly**, so visibility and opacity live
  in the layer registry rather than on the actor -- which is why the surface is
  handed over with ``set_extra_actors`` instead of ``add_mesh``.
"""

from __future__ import annotations

import numpy as np
import pyvista as pv

from .adapter import Triple
from .graphmodel import EditableGraph
from .history import Patch
from .sdfpatch import SdfSession, splice
from .worker import RebuildOutcome, RebuildQueue, qt_poster, session_rebuilder

EDIT_SURFACE_COLOR = "#4fc3f7"
HANDLE_COLOR = "#ffd400"
SELECTED_COLOR = "#ff2d55"

# Keys this module claims. `viewer3d` already uses v n b s c i g, and leaves r
# (VTK reset camera) and q (close) alone; none of those appear here.
KEYS = ("e", "z", "y", "d", "t", "x", "k", "j", "u", "f")

INSTRUCTIONS = (
    "e             toggle edit mode\n"
    "z / y         undo / redo\n"
    "d             delete the picked segment\n"
    "t             delete the picked branch and everything past it\n"
    "x             split the segment at the picked point\n"
    "k / j         widen / narrow the picked segment\n"
    "f             fill the collapsed radii around the pick\n"
    "u             rebuild the surface around the pick"
)


class EditController:
    """Edit the skeleton in the 3D window and watch the lumen surface follow."""

    def __init__(
        self,
        picker,
        triple: Triple,
        *,
        session: SdfSession | None = None,
        min_extent_um: float = 4000.0,
        auto_rebuild: bool = True,
        paint=None,
        source=None,
        frame=None,
    ):
        self.picker = picker
        self.graph = EditableGraph(triple)
        self.session = session if session is not None else SdfSession(self.graph.snapshot())
        self.min_extent_um = float(min_extent_um)
        self.auto_rebuild = bool(auto_rebuild)
        # Voxel painting lives in the *other* window. The controller only needs the
        # store and the geometry, so that a painted correction can be turned into
        # centreline and rebuilt through the same queue as any other edit.
        self.paint = paint
        self.source = source
        self.frame = frame
        if paint is not None:
            paint.on_reskeletonise = self.reskeletonise_box

        self.enabled = False
        self.surface: pv.PolyData | None = None
        self.status = "edit mode off"
        self.on_status: callable | None = None
        self.panel = None
        # Kept as well as the panel so `dispose` can remove the dock itself; a
        # replacement controller would otherwise add a second "edit" dock.
        self.dock = None

        self._queue: RebuildQueue | None = None
        self._surface_actor = None
        self._handle_actor = None
        self._busy = 0

    # ------------------------------------------------------------------ wiring

    def attach(self) -> None:
        """Bind keys, dock the panel and start the worker. Call after ``build()``."""
        plotter = self.picker.plotter
        if plotter is None:
            raise RuntimeError("attach() needs a built plotter; call Picker3D.build() first")

        self._queue = RebuildQueue(
            session_rebuilder(self.session, min_extent_um=self.min_extent_um),
            self._on_rebuilt,
            post=_safe_poster(),
        )

        # pyvista *appends* key handlers rather than replacing them, so clearing
        # first is mandatory even for keys nothing appears to use.
        for key in KEYS:
            plotter.clear_events_for_key(key)
        plotter.add_key_event("e", self.toggle_enabled)
        plotter.add_key_event("z", self.undo)
        plotter.add_key_event("y", self.redo)
        plotter.add_key_event("d", self.delete_picked_segment)
        plotter.add_key_event("t", self.delete_picked_branch)
        plotter.add_key_event("x", self.split_at_pick)
        plotter.add_key_event("k", lambda: self.scale_picked_radius(1.1))
        plotter.add_key_event("j", lambda: self.scale_picked_radius(1 / 1.1))
        plotter.add_key_event("f", self.fill_collapse_at_pick)
        plotter.add_key_event("u", self.rebuild_at_pick)

        self._dock_panel(plotter)
        self._set_status("edit mode off  (press 'e')")

    def detach(self) -> None:
        if self._queue is not None:
            self._queue.shutdown()
            self._queue = None

    def dispose(self) -> None:
        """Undo everything `attach` did, so a new dataset can have its own controller.

        Both halves of the state are bound to one graph: the undo history means
        nothing against a different skeleton, and the `SdfSession` was built from the
        capsules of this one. So a dataset swap disposes and rebuilds rather than
        re-pointing -- which also keeps the 88 s SDF prepare an explicit user action.
        """
        self.detach()
        for key in ("edit_surface", "edit_handles"):
            try:
                self.picker.set_extra_actors(key, None)
            except Exception:  # noqa: BLE001 - teardown must not raise
                pass
        plotter = getattr(self.picker, "plotter", None)
        for actor in (self._surface_actor, self._handle_actor):
            if actor is not None and plotter is not None:
                plotter.remove_actor(actor, render=False)
        self._surface_actor = None
        self._handle_actor = None
        if plotter is not None:
            for key in KEYS:
                plotter.clear_events_for_key(key)
        # The dock, not just the panel: a second controller would otherwise add a
        # second "edit" dock beside the orphaned first one.
        if self.dock is not None:
            self.dock.setParent(None)
            self.dock.deleteLater()
            self.dock = None
        self.panel = None

    def _dock_panel(self, plotter) -> None:
        if not hasattr(plotter, "app_window"):
            return  # a bare pv.Plotter, as used by the test harness
        from qtpy.QtCore import Qt
        from qtpy.QtWidgets import QDockWidget

        from . import controls_edit

        self.panel = controls_edit.build_edit_panel(self)
        self.dock = QDockWidget("edit", plotter.app_window)
        self.dock.setWidget(self.panel)
        plotter.app_window.addDockWidget(Qt.RightDockWidgetArea, self.dock)

    # --------------------------------------------------------------- selection

    def picked_point_id(self) -> int | None:
        """The graph point id under the last pick, or ``None``.

        ``Picker3D`` stores the index into ``graph.points``; the editable model
        keys points by id, and ``point_order`` is the same flat concatenation the
        renderer used, so the two line up by construction.
        """
        index = getattr(self.picker, "_point_i", -1)
        if index is None or index < 0:
            return None
        order = self.graph.point_order()
        if index >= len(order):
            return None
        return order[index]

    def picked_segment_id(self) -> int | None:
        pid = self.picked_point_id()
        if pid is None:
            return None
        return self.graph.segment_of_point().get(pid)

    def _require_pick(self) -> tuple[int, int] | None:
        pid = self.picked_point_id()
        if pid is None:
            self._set_status("nothing picked - double-click a centreline point first")
            return None
        sid = self.graph.segment_of_point().get(pid)
        if sid is None:
            self._set_status("the picked point is not on a segment any more")
            return None
        return pid, sid

    def _guard(self) -> bool:
        if not self.enabled:
            self._set_status("edit mode is off - press 'e' to enable")
            return False
        return True

    # -------------------------------------------------------------- operations

    def toggle_enabled(self) -> None:
        self.enabled = not self.enabled
        if self.enabled and self.surface is None:
            self.rebuild_at_pick()
        self._set_status("edit mode ON" if self.enabled else "edit mode off")
        self._draw_handles()

    def delete_picked_segment(self) -> None:
        if not self._guard():
            return
        picked = self._require_pick()
        if picked is None:
            return
        _, sid = picked
        self._apply(lambda: self.graph.delete_segment(sid), f"deleted segment {sid}")

    def delete_picked_branch(self) -> None:
        """Prune the picked segment and everything downstream of the nearer node."""
        if not self._guard():
            return
        picked = self._require_pick()
        if picked is None:
            return
        pid, sid = picked
        seg = self.graph.segment(sid)
        # Walk away from whichever endpoint the pick is closer to, so clicking
        # near the tip removes the tip rather than the rest of the tree.
        ids = seg["point_ids"]
        from_node = seg["node1"] if ids.index(pid) > len(ids) / 2 else seg["node2"]
        n = len(self.graph.subtree(sid, from_node))
        self._apply(
            lambda: self.graph.delete_subtree(sid, from_node),
            f"pruned {n} segment(s) from segment {sid}",
        )

    def split_at_pick(self) -> None:
        if not self._guard():
            return
        picked = self._require_pick()
        if picked is None:
            return
        pid, sid = picked
        index = self.graph.segment(sid)["point_ids"].index(pid)
        if not 0 < index < len(self.graph.segment(sid)["point_ids"]) - 1:
            self._set_status("pick a point inside the segment, not one of its ends")
            return

        def run():
            nid, _a, _b = self.graph.split_segment(sid, index)
            self._set_status(f"split segment {sid} at a new node {nid}")
            return self.graph.last_patch

        self._apply(run, None)

    def scale_picked_radius(self, factor: float) -> None:
        if not self._guard():
            return
        picked = self._require_pick()
        if picked is None:
            return
        _, sid = picked
        self._apply(
            lambda: self.graph.scale_radii(sid, factor),
            f"segment {sid} radii x{factor:.2f}",
        )

    def set_picked_radius(self, radius_um: float) -> None:
        if not self._guard():
            return
        picked = self._require_pick()
        if picked is None:
            return
        pid, _ = picked
        self._apply(
            lambda: self.graph.set_radius(pid, radius_um),
            f"point {pid} radius = {radius_um:.1f} um",
        )

    def fill_collapse_at_pick(self) -> None:
        """Restore the radii of the collapsed run the pick sits in.

        One keypress rather than marking both ends: the run is grown outwards from
        the picked point until the radius comes back up to the segment's own
        trend, then filled by extrapolating the taper from the healthy tissue
        either side. A pick on healthy tissue does nothing and says so, rather than
        inventing a span around it.
        """
        if not self._guard():
            return
        picked = self._require_pick()
        if picked is None:
            return
        pid, sid = picked
        from . import radius_repair as rr

        index = self.graph.segment(sid)["point_ids"].index(pid)
        span = rr.span_around(self.graph, sid, index)
        if span is None:
            self._set_status(
                "no collapse here - this radius is not below the segment's own trend"
            )
            return

        report = rr.fill_span(self.graph, span)
        if not report.applied:
            self._set_status(f"cannot fill: {report.reason}")
            return
        self._after_edit(
            self.graph.last_patch,
            f"filled seg {sid}[{span.i0}:{span.i1}] "
            f"{report.r_before[0]:.0f}->{report.r_after[0]:.0f} um "
            f"({report.sides}, slope {report.slope_per_mm:+.3f}/mm)",
        )

    def reskeletonise_box(self, box_um, mode: str = "add") -> None:
        """Re-derive the centreline through a painted region and rebuild the surface.

        Driven from the slice browser's paint panel rather than from a key here,
        because that is the window the painting happened in. Safe to call across the
        two windows: both run on the one Qt event loop, a button click is not a
        vispy GL callback, and the rebuild is handed to the worker either way.
        """
        if self.source is None or self.frame is None:
            self._set_status("re-skeletonisation needs the segmentation (--paint)")
            return
        from .reskeletonise import reskeletonise_box

        try:
            report = reskeletonise_box(
                self.graph, self.source, self.frame, box_um, mode=mode
            )
        except Exception as exc:  # noqa: BLE001 - a bad box must not kill the window
            self._set_status(f"re-skeletonise failed: {exc}")
            return
        print("[edit]", report.describe())
        if not report.applied:
            self._set_status(report.describe())
            return
        self._after_edit(report.patch, report.describe())

    def undo(self) -> None:
        patch = self.graph.undo()
        if patch is None:
            self._set_status("nothing to undo")
            return
        self._after_edit(patch, f"undid: {self.graph.history.redo_label}")

    def redo(self) -> None:
        patch = self.graph.redo()
        if patch is None:
            self._set_status("nothing to redo")
            return
        self._after_edit(patch, f"redid: {self.graph.history.undo_label}")

    def _apply(self, run, message: str | None) -> None:
        try:
            patch = run()
        except Exception as exc:  # noqa: BLE001 - a bad edit must not kill the window
            self._set_status(f"edit failed: {exc}")
            return
        self._after_edit(patch, message)

    def _after_edit(self, patch: Patch, message: str | None) -> None:
        self._draw_handles()
        if message:
            self._set_status(message)
        if self.auto_rebuild:
            self.request_rebuild(patch)

    # ----------------------------------------------------------------- rebuild

    def request_rebuild(self, patch: Patch | None = None) -> None:
        """Queue a rebuild of the box `patch` touched (or around the pick)."""
        if self._queue is None:
            return
        if patch is None or patch.aabb is None:
            patch = self._patch_at_pick()
        if patch is None:
            self._set_status("nothing to rebuild - pick a point first")
            return
        self._busy += 1
        self._set_status("rebuilding ...")
        self._queue.request(patch, self.graph.snapshot(), set(self.graph.root_pref))

    def rebuild_at_pick(self) -> None:
        self.request_rebuild(self._patch_at_pick())

    def _patch_at_pick(self) -> Patch | None:
        xyz = getattr(self.picker, "picked", None)
        if xyz is None:
            return None
        centre = np.asarray(xyz, dtype=np.float64).reshape(3)
        half = self.min_extent_um / 2.0
        sid = self.picked_segment_id()
        return Patch(frozenset({sid} if sid is not None else ()),
                     np.array([centre - half, centre + half]))

    def _on_rebuilt(self, outcome: RebuildOutcome) -> None:
        """Back on the UI thread with a finished patch."""
        self._busy = max(0, self._busy - 1)
        if not outcome.ok:
            first = str(outcome.error).splitlines()[:1]
            self._set_status(f"rebuild failed: {first[0] if first else outcome.error}")
            return
        result = outcome.result
        self.surface = splice(self.surface, result) if self.surface is not None else result.surface
        self._draw_surface()
        self._set_status(
            f"rebuilt {result.surface.n_cells:,} triangles in {outcome.seconds:.2f}s"
            + (" (more queued)" if self._busy else "")
        )

    # ---------------------------------------------------------------- drawing

    def _draw_surface(self) -> None:
        plotter = self.picker.plotter
        if plotter is None or self.surface is None or self.surface.n_cells == 0:
            return
        if self._surface_actor is not None:
            plotter.remove_actor(self._surface_actor, reset_camera=False, render=False)
        self._surface_actor = plotter.add_mesh(
            self.surface, color=EDIT_SURFACE_COLOR, opacity=1.0,
            smooth_shading=True, name="edit_surface", reset_camera=False,
        )
        self.picker.set_extra_actors("edit_surface", [self._surface_actor])

    def _draw_handles(self) -> None:
        """Mark the nodes of the picked segment, so an edit's target is visible."""
        plotter = self.picker.plotter
        if plotter is None:
            return
        if self._handle_actor is not None:
            plotter.remove_actor(self._handle_actor, reset_camera=False, render=False)
            self._handle_actor = None

        sid = self.picked_segment_id()
        if not self.enabled or sid is None:
            self.picker.set_extra_actors("edit_handles", None)
            return

        seg = self.graph.segment(sid)
        pts = np.array(
            [self.graph.nodes[n][:3] for n in (seg["node1"], seg["node2"])
             if n in self.graph.nodes],
            dtype=np.float64,
        ).reshape(-1, 3)
        if not len(pts):
            self.picker.set_extra_actors("edit_handles", None)
            return
        radius = max(float(np.mean(self.graph.radii(sid))) * 1.5, 40.0)
        self._handle_actor = plotter.add_mesh(
            pv.PolyData(pts).glyph(geom=pv.Sphere(radius=radius), orient=False, scale=False),
            color=HANDLE_COLOR, name="edit_handles", reset_camera=False,
        )
        self.picker.set_extra_actors("edit_handles", [self._handle_actor])

    # ------------------------------------------------------------------ status

    def _set_status(self, text: str) -> None:
        self.status = text
        if self.on_status is not None:
            self.on_status(text)

    def summary(self) -> str:
        g = self.graph
        return (
            f"{len(g.segments)} segments, {len(g.nodes)} nodes, "
            f"{len(g.components())} component(s), "
            f"{len(g.endpoints())} free ends"
        )

    # ------------------------------------------------------------------ export

    def export(self, output_dir, *, write_graph: bool = True) -> dict:
        """Write the edited graph and a full-quality surface.

        The STL comes from a real ``generate_sdf_surface`` run rather than an
        accumulation of patches: patches are spliced for display and are not
        welded at their seams, which is fine to look at and not fine to mesh.
        """
        from pathlib import Path

        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        written = {}

        if write_graph:
            from .amira_write import write_spatial_graph

            path = out / "edited.am"
            write_spatial_graph(
                self.graph.to_spatial_graph(),
                path,
                parameters_from=self.graph.triple.source,
            )
            written["graph"] = path

        self.session.set_graph(self.graph.snapshot(), self.graph.root_pref)
        surface = self.session.rebuild_full(out)
        if surface is not None:
            written["surface"] = out / "lumen_bspline.stl"
        return written


def _safe_poster():
    """`qt_poster` when Qt is importable, otherwise run the callback in place."""
    try:
        return qt_poster()
    except Exception:  # noqa: BLE001 - headless test harness has no Qt
        return lambda fn: fn()
