"""The Reformat tab: pick a run of vessel, look down it instead of across it.

Nothing here writes anything. The panel assembles a selection of segments, builds a
stack of perpendicular cross-sections through :mod:`~.reformat`, and opens it in its own
napari window. There is no output file and no dataset mutation, so nothing a mis-click
can cost you.

Like the Crop tab, the panel keeps its **own** :class:`~.edit.graphmodel.EditableGraph`
built from the session's spatial graph. `reformat` works in segment ids and needs an
incidence index, which a ``SpatialGraph`` does not have; and it being a copy means the
tree you are looking at cannot be modified by anything in here, including by a bug in it.

Picks arrive as an index into ``graph.points``, which ``point_order`` maps to a point id
and ``segment_of_point`` to a segment -- the same chain `controls_crop` and
`edit/controller.py` use, which holds because ``viewer3d.centreline_polydata`` renders
points in exactly that order.

Selecting a long run one segment at a time is forty picks, and a single miss leaves a
gap that `chain_segments` can only report as two runs. **Trace mode** borrows the Crop
tab's :func:`~.edit.crop.trace_path` -- a Dijkstra over nodes -- so two picks name the
whole vessel between them: a path, therefore connected, therefore exactly the one
continuous run a reformat needs.

**Building reads TIFFs**, which for a long run is seconds to minutes of decode, so it
happens on a worker thread and the napari window is opened from the Qt thread when it
finishes. The "Check" button exists so the geometry -- how much smoothing the run needs,
whether the half-width will be clamped, whether the planes come out disjoint -- can be
had in milliseconds, without committing to the decode.

Qt is imported inside the function, matching every other panel, so this module still
imports in a session with no Qt binding.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from . import reformat as reformat_mod
from . import reformat_io
from .edit import crop as crop_mod

#: Keys this panel binds in the 3D window.
#:
#: Digits, because **every letter is taken**: `viewer3d` owns `v n b s c i g a r q`, the
#: edit controller `e z y d t x k j u f`, the Crop tab `m l o h`, and VTK's own
#: interactor style binds `p` (prop pick), `w` (wireframe), `s` (surface) and `r`
#: (reset). `3` is left alone: it is VTK's stereo toggle, which lives in the interactor
#: style rather than in pyvista's key dictionary, so `clear_events_for_key` cannot take
#: it back, and `5 6 7` belong to the Sections tab. `8` disarms a half-finished trace,
#: which is the one state where the next pick would otherwise mean something the
#: operator has stopped expecting.
KEYS = ("1", "2", "4", "8")

#: ``native`` first: it is the only mode whose pixel grid is pinned to the
#: acquisition, so it is the only one that cannot magnify. The other three derive the
#: pitch from a half-width, so any half-width narrower than ``size_px / 2`` voxels
#: enlarges -- whether that came from a small vessel or from the curvature clamp.
MODES = ("native", "radius", "fixed", "manual")

#: Save layouts, as ``(value, label)``. The npz is one file and the default; the two
#: folder layouts exist because a reformat is often worth taking somewhere else, and a
#: TIFF stack opens in Fiji without a script.
FORMATS = (
    ("npz", ".npz (one file)"),
    ("tiff", "folder: TIFF + JSON"),
    ("npy", "folder: .npy + JSON"),
)

#: Interpolation orders offered, with what each is for. Cubic is the default: measured
#: on a real slice it cuts round-trip RMSE 26% against linear and recovers 73% of the
#: edge energy against 55%, which matters because the default grid magnifies. Quintic
#: adds 2% for three times the cost; nearest and linear are there to compare against.
ORDERS = (
    (0, "nearest (0)"),
    (1, "linear (1)"),
    (3, "cubic (3)"),
    (5, "quintic (5)"),
)


def build_reformat_panel(app):
    """Return the Reformat widget for a `ViewerApp`. Docking is the caller's business."""
    from qtpy.QtWidgets import (
        QCheckBox,
        QComboBox,
        QDoubleSpinBox,
        QFileDialog,
        QHBoxLayout,
        QLabel,
        QLineEdit,
        QListWidget,
        QPushButton,
        QSpinBox,
        QVBoxLayout,
        QWidget,
    )

    box = QWidget()
    lay = QVBoxLayout(box)
    lay.setContentsMargins(8, 8, 8, 8)

    state: dict = {
        "graph": None,       # our private EditableGraph
        "source": None,      # the SpatialGraph it was built from, for identity
        "selected": [],      # ordered segment ids -- pick order breaks chain ties
        "trace_start": None, # the first pick of a trace, waiting for its far end
        "reformat": None,    # the last stack built
        "busy": False,
    }

    banner = QLabel(
        "Cross-sections cut perpendicular to the centreline, stacked along it. "
        "Nothing here writes to the dataset."
    )
    banner.setWordWrap(True)
    banner.setStyleSheet("color: #808090;")
    lay.addWidget(banner)

    # -- selection --------------------------------------------------------
    lay.addWidget(QLabel("selected segments (double-click a centreline point, then '1')"))
    seg_list = QListWidget()
    seg_list.setMaximumHeight(90)
    lay.addWidget(seg_list)

    pick_row = QHBoxLayout()
    add_button = QPushButton("Add pick (1)")
    clear_button = QPushButton("Clear (2)")
    pick_row.addWidget(add_button)
    pick_row.addWidget(clear_button)
    lay.addLayout(pick_row)

    # A mode rather than a third button, so `1` keeps meaning "this pick belongs to the
    # selection" in both: what changes is how much of the vessel one press names.
    trace_row = QHBoxLayout()
    trace_mode = QCheckBox("trace between two picks")
    trace_mode.setToolTip(
        "Add pick (1) sets one end of the run, the next one sets the other, and every "
        "segment on the path between them joins the selection -- in path order, which "
        "is the order the chain walk wants."
    )
    prefer_thick = QCheckBox("prefer thick")
    prefer_thick.setToolTip(
        "Route by length in units of each segment's own radius rather than by length "
        "alone, so a trace goes down the vessel instead of over a thin false bridge. "
        "On a tree the path is unique and this changes nothing."
    )
    cancel_trace_button = QPushButton("Cancel trace (8)")
    trace_row.addWidget(trace_mode)
    trace_row.addWidget(prefer_thick)
    trace_row.addWidget(cancel_trace_button)
    lay.addLayout(trace_row)

    # -- plane geometry ---------------------------------------------------
    mode_row = QHBoxLayout()
    mode_row.addWidget(QLabel("size"))
    mode = QComboBox()
    mode.addItems(MODES)
    mode_row.addWidget(mode, 1)
    mode_row.addWidget(QLabel("px"))
    size_px = QSpinBox()
    size_px.setRange(9, 1025)
    size_px.setSingleStep(2)
    size_px.setValue(reformat_mod.DEFAULT_SIZE_PX)
    mode_row.addWidget(size_px)
    lay.addLayout(mode_row)

    order_row = QHBoxLayout()
    order_row.addWidget(QLabel("interpolation"))
    order = QComboBox()
    for value, label in ORDERS:
        order.addItem(label, value)
    order.setCurrentIndex([v for v, _l in ORDERS].index(reformat_mod.DEFAULT_ORDER))
    order_row.addWidget(order, 1)
    match_button = QPushButton("Match voxel")
    match_button.setToolTip(
        "Size the pixel grid so one output pixel is about one raw voxel, for the "
        "current selection. Above that the section is magnified rather than resolved."
    )
    order_row.addWidget(match_button)
    lay.addLayout(order_row)

    radii_row = QHBoxLayout()
    radii_row.addWidget(QLabel("half-width (radii)"))
    radii_k = QDoubleSpinBox()
    radii_k.setRange(1.0, 40.0)
    radii_k.setSingleStep(0.5)
    radii_k.setValue(reformat_mod.HALF_WIDTH_RADII)
    radii_row.addWidget(radii_k, 1)
    lay.addLayout(radii_row)

    manual_row = QHBoxLayout()
    manual_row.addWidget(QLabel("half um"))
    half_um = QDoubleSpinBox()
    half_um.setRange(1.0, 1e6)
    half_um.setDecimals(0)
    half_um.setValue(1000.0)
    manual_row.addWidget(half_um, 1)
    manual_row.addWidget(QLabel("um/px"))
    px_um = QDoubleSpinBox()
    px_um.setRange(0.01, 1e4)
    px_um.setDecimals(2)
    px_um.setValue(20.0)
    manual_row.addWidget(px_um, 1)
    lay.addLayout(manual_row)

    # -- path -------------------------------------------------------------
    step_row = QHBoxLayout()
    step_row.addWidget(QLabel("step um (0 = one raw voxel)"))
    step_um = QDoubleSpinBox()
    step_um.setRange(0.0, 1e4)
    step_um.setDecimals(2)
    step_um.setValue(0.0)
    step_row.addWidget(step_um, 1)
    lay.addLayout(step_row)

    safety_row = QHBoxLayout()
    safety_row.addWidget(QLabel("curvature safety"))
    safety = QDoubleSpinBox()
    safety.setRange(0.1, 1.0)
    safety.setSingleStep(0.05)
    safety.setValue(reformat_mod.DEFAULT_SAFETY)
    safety_row.addWidget(safety, 1)
    safety_row.addWidget(QLabel("smooth passes"))
    passes = QSpinBox()
    passes.setRange(0, 40)
    passes.setValue(reformat_mod.MAX_SMOOTH_ITERS)
    safety_row.addWidget(passes)
    lay.addLayout(safety_row)

    smooth_row = QHBoxLayout()
    smooth_row.addWidget(QLabel("smoothing window um (0 = auto)"))
    smooth_um = QDoubleSpinBox()
    smooth_um.setRange(0.0, 1e5)
    smooth_um.setDecimals(0)
    smooth_um.setValue(0.0)
    smooth_row.addWidget(smooth_um, 1)
    lay.addLayout(smooth_row)

    with_mask = QCheckBox("segmentation overlay")
    with_mask.setChecked(True)
    lay.addWidget(with_mask)

    # The current cross-section always follows the napari slider into the 3D window --
    # that is one actor, rebuilt as you scroll, and it is what ties the two windows
    # together. This is the *whole decimated stack* drawn at once, which costs a texture
    # upload per plane, so it stays off unless asked for.
    with_sections = QCheckBox("draw the section stack in 3D (slower)")
    with_sections.setChecked(False)
    lay.addWidget(with_sections)

    # -- actions ----------------------------------------------------------
    action_row = QHBoxLayout()
    check_button = QPushButton("Check geometry")
    show_button = QPushButton("Show stack (4)")
    action_row.addWidget(check_button)
    action_row.addWidget(show_button)
    lay.addLayout(action_row)

    # -- save / load ------------------------------------------------------
    # Building is the expensive part -- 37 s for a 47 mm run, almost all of it TIFF
    # decode -- and today it is thrown away when the window closes.
    lay.addWidget(QLabel("save / load a built stack"))
    save_row = QHBoxLayout()
    save_name = QLineEdit()
    save_name.setPlaceholderText("name, e.g. LAD")
    save_row.addWidget(save_name, 1)
    save_format = QComboBox()
    for value, label in FORMATS:
        save_format.addItem(label, value)
    save_row.addWidget(save_format)
    lay.addLayout(save_row)

    save_hint = QLabel("")
    save_hint.setWordWrap(True)
    save_hint.setStyleSheet("color: #808090;")
    lay.addWidget(save_hint)

    io_row = QHBoxLayout()
    save_button = QPushButton("Save stack...")
    load_button = QPushButton("Load stack...")
    io_row.addWidget(save_button)
    io_row.addWidget(load_button)
    lay.addLayout(io_row)

    summary = QLabel("")
    summary.setWordWrap(True)
    summary.setTextInteractionFlags(summary.textInteractionFlags())
    lay.addWidget(summary)
    lay.addStretch(1)

    # ------------------------------------------------------------ helpers

    def say(message: str) -> None:
        summary.setText(message)

    def status(message: str) -> None:
        report = getattr(app, "status", None)
        if report is not None:
            report(message)

    def picked_segment():
        """``(segment id, point id)`` under the last 3D pick, or ``None``.

        The pick is an index into ``graph.points``; ``point_order`` is the same flat
        concatenation the renderer used, so the two line up by construction.
        """
        graph = state["graph"]
        picker = getattr(app, "picker", None)
        index = getattr(picker, "_point_i", -1) if picker is not None else -1
        if graph is None or index is None or index < 0:
            return None
        order = graph.point_order()
        if index >= len(order):
            return None
        pid = order[index]
        sid = graph.segment_of_point().get(pid)
        return None if sid is None else (sid, pid)

    def rebuild_graph() -> None:
        """Take a private copy of the session's graph, dropping any stale selection."""
        from .edit.adapter import from_spatial_graph
        from .edit.graphmodel import EditableGraph

        source = getattr(getattr(app, "session", None), "graph", None)
        state["source"] = source
        state["graph"] = None if source is None else EditableGraph(from_spatial_graph(source))
        # A selection is a set of segment ids in the graph it was made on, and means
        # something else entirely in the next one.
        state["selected"] = []
        state["trace_start"] = None
        state["reformat"] = None
        clear = getattr(getattr(app, "picker", None), "clear_reformat", None)
        if clear is not None:
            try:
                clear()
            except Exception:  # noqa: BLE001 - a dead plotter must not block a load
                pass

    def primary_run() -> set:
        """Segment ids in the run a build would actually sample, or all of them.

        A reformat is one continuous path, so a selection spanning two disconnected
        vessels can only build one of them. Knowing which *before* pressing Show is
        what stops the 3D preview and the segment list from showing one thing while
        the build does another. Pure graph walking, so it costs nothing to ask.
        """
        graph = state["graph"]
        if graph is None or not state["selected"]:
            return set()
        try:
            chains, _notes = reformat_mod.chain_segments(graph, state["selected"])
        except Exception:  # noqa: BLE001 - a preview must never break the panel
            return set(state["selected"])
        return set(chains[0].segment_ids) if chains else set()

    def draw() -> None:
        """Hand the selection preview to the 3D window, if there is one.

        Split into the run that will be sampled and the segments that will not, so the
        two are drawn in different colours rather than looking equally included.
        """
        show = getattr(getattr(app, "picker", None), "show_reformat", None)
        graph = state["graph"]
        if show is None or graph is None:
            return
        run = primary_run()
        kept = [graph.coords(sid) for sid in state["selected"]
                if graph.has_segment(sid) and sid in run]
        lost = [graph.coords(sid) for sid in state["selected"]
                if graph.has_segment(sid) and sid not in run]
        try:
            show(kept, lost)
        except Exception as exc:  # noqa: BLE001 - see the note on slots below
            status(f"  could not draw the reformat preview: {exc}")

    def refresh_lists() -> None:
        seg_list.clear()
        graph = state["graph"]
        run = primary_run()
        for sid in state["selected"]:
            if graph is not None and graph.has_segment(sid):
                length = np.linalg.norm(np.diff(graph.coords(sid), axis=0), axis=1).sum()
                # Marked rather than hidden or reordered: it is still selected, and
                # deselecting something else may well bring it back into the run.
                mark = "" if sid in run else "   [not in the run]"
                seg_list.addItem(f"segment {sid} - {length / 1000:.2f} mm{mark}")
            else:
                seg_list.addItem(f"segment {sid} - gone")

    def options() -> dict:
        chosen = mode.currentText()
        return {
            "mode": chosen,
            "radii_k": float(radii_k.value()),
            "size_px": int(size_px.value()),
            "half_um": float(half_um.value()) if chosen == "manual" else None,
            "px_um": float(px_um.value()) if chosen == "manual" else None,
            "step_um": float(step_um.value()) or None,
            "safety": float(safety.value()),
            "max_smooth_iters": int(passes.value()),
            "smooth_window_um": float(smooth_um.value()) or None,
            "order": int(order.currentData()),
        }

    # Every slot below funnels its failure into the summary label. PyQt5 calls qFatal on
    # an exception that escapes a slot, which aborts the process rather than raising -- a
    # selection that cannot be chained must not take the window down with it.
    #
    # `_arg=None` rather than the obvious `*_args`, and the difference is load-bearing:
    # this wrapper is handed both to Qt signals and to `plotter.add_key_event`, and
    # pyvista rejects any callback with a parameter that has no default. It walks
    # `signature(callback).parameters`, and a `*args` entry always reports no default,
    # so `*_args` is refused even though it is callable with zero arguments -- a
    # TypeError out of `attach_control_dock` that takes the session down before the
    # window opens. One default-valued positional absorbs everything Qt emits here
    # (`clicked(bool)`, `currentTextChanged(str)`) and satisfies pyvista.
    def guarded(fn):
        def run(_arg=None):
            try:
                fn()
            except Exception as exc:  # noqa: BLE001 - see above
                say(f"{type(exc).__name__}: {exc}")
        return run

    # ------------------------------------------------------------ actions

    def do_add() -> None:
        if trace_mode.isChecked():
            do_trace()
            return
        picked = picked_segment()
        if picked is None:
            say("nothing picked - double-click a centreline point in the 3D view")
            return
        sid, _pid = picked
        # A toggle rather than an add: picking the same segment twice is far more likely
        # to be a correction than a request to include it again.
        if sid in state["selected"]:
            state["selected"].remove(sid)
        else:
            state["selected"].append(sid)
        state["reformat"] = None
        refresh_lists()
        draw()
        do_check(quiet=True)

    def do_clear() -> None:
        state["selected"] = []
        state["trace_start"] = None
        state["reformat"] = None
        refresh_lists()
        draw()
        say("selection cleared")

    def do_trace() -> None:
        """Two picks name a whole run: one end, then the other, everything in between.

        The one thing a reformat needs and a hand-made selection cannot promise is that
        the picks are *connected*: `chain_segments` splits an interrupted selection into
        runs and the build then samples only the longest, so a single missed segment
        silently shortens the stack. A traced path is connected by construction.

        Appended in path order rather than merged into a set, because the selection is
        ordered and that order breaks ties in the chain walk -- and path order is
        already the run order. Segments picked before are left where they are: a trace
        extends the selection, it does not replace it.

        The pending start is dropped **before** the trace runs, not after it succeeds.
        A start left behind by a failed trace is the worst state this can be in: the
        next pick silently becomes the far end of a trace the operator has stopped
        expecting.
        """
        picked = picked_segment()
        if picked is None:
            say("nothing picked - double-click a centreline point in the 3D view")
            return
        sid, _pid = picked
        if state["trace_start"] is None:
            state["trace_start"] = sid
            say(f"trace armed at segment {sid}. Pick the far end and press Add pick "
                "again, or Cancel trace.")
            return
        start = state["trace_start"]
        state["trace_start"] = None
        traced = crop_mod.trace_path(
            state["graph"], start, sid, prefer_thick=prefer_thick.isChecked()
        )
        added = [s for s in traced.segments if s not in state["selected"]]
        state["selected"].extend(added)
        state["reformat"] = None
        refresh_lists()
        draw()
        # What was traced is said first and the geometry appended to it, rather than the
        # other way round: a trace is exactly when the geometry is worth knowing, but a
        # raise out of the check must not cost the line saying what was just selected.
        say(f"traced {traced.describe()}; {len(added)} new, "
            f"{len(state['selected'])} segment(s) selected")
        line = summary.text()
        do_check(quiet=True)
        checked = summary.text()
        say(line if checked == line else f"{line}\n{checked}")

    def do_cancel_trace() -> None:
        if state["trace_start"] is None:
            say("no trace in progress")
            return
        state["trace_start"] = None
        say("trace cancelled - the start pick was dropped")

    def geometry():
        """``(chain, centreline, geometry, notes)`` for the current selection.

        Everything except the image sampling, which is milliseconds against the decode.
        Shared by "Check geometry" and "Match voxel" so the second sizes the grid
        against the half-width the build will *actually* use -- including any curvature
        clamp -- rather than against the one the mode asked for. Those can differ by an
        order of magnitude on a tortuous run.
        """
        graph = state["graph"]
        if graph is None or not state["selected"]:
            return None
        opts = options()
        chains, notes = reformat_mod.chain_segments(graph, state["selected"])
        if not chains:
            say("nothing to reformat: no usable segments in the selection")
            return None

        chain = chains[0]
        coords, radii, sids = reformat_mod.chain_arrays(graph, chain.steps)
        if len(coords) < 2:
            say("the selected run is a single point")
            return None

        voxel = _default_step(app)
        step = opts["step_um"] or voxel
        line = reformat_mod.build_centreline(
            coords, radii, sids, step_um=step,
            half_of=_half_of(opts, voxel), safety=opts["safety"],
            smooth_window_um=opts["smooth_window_um"],
            max_smooth_iters=opts["max_smooth_iters"],
        )
        geom = reformat_mod.plane_geometry(
            line, mode=opts["mode"], radii_k=opts["radii_k"], size_px=opts["size_px"],
            half_um=opts["half_um"], px_um=opts["px_um"], safety=opts["safety"],
            min_half_um=voxel, voxel_um=voxel,
        )
        return chain, line, geom, notes

    def do_check(quiet: bool = False) -> None:
        """Report the geometry without reading a single image.

        This is the whole reason the geometry is separable from the sampler: you find
        out that the half-width will be clamped to a third of what you asked for
        *before* committing to a long decode, not after.
        """
        found = geometry()
        if found is None:
            if not quiet and state["graph"] is not None and not state["selected"]:
                say("select at least one segment first")
            return
        chain, line, geom, notes = found
        ok, offenders = reformat_mod.planes_disjoint(
            line.coords_um, line.tangents, line.normals, line.binormals, geom.half_um
        )
        lines = [
            f"{len(chain.steps)} segment(s), {chain.length_um / 1000:.2f} mm",
            line.describe(),
            geom.describe(),
            "planes are disjoint" if ok else f"{len(offenders)} pair(s) of planes overlap",
        ]
        lines += [f"note: {n}" for n in list(notes) + list(line.notes)]
        say("\n".join(lines))

    def do_match_voxel() -> None:
        """Size the pixel grid so one output pixel is about one raw voxel.

        Sized against the half-width the build will actually use, which is **not**
        always the one the mode asked for: the curvature clamp can cut it by an order
        of magnitude on a tortuous run, and because ``size_px`` is fixed that turns
        straight into magnification. Measured on the longest segment of LADAF-2024-28,
        a 982 um requested half-width clamped to 181 um, which at 129 px is 2.8 um/px
        against a 33 um voxel -- 11.65x magnified. Sizing from the *requested* width
        would not have caught that; sizing from the used width does.

        In ``radius`` mode ``um/px`` still varies plane to plane, so one ``size_px``
        can only match at one representative width. The median is used, and the summary
        says so rather than implying an exactness the mode cannot give.
        """
        found = geometry()
        if found is None:
            if state["graph"] is not None and not state["selected"]:
                say("select at least one segment first")
            return
        _chain, _line, geom, _notes = found
        voxel = _default_step(app)

        if geom.mode == "native":
            # The pitch is already one voxel, so there is nothing to match; what is
            # worth setting is the *frame*, to the vessel it is looking at.
            graph = state["graph"]
            radii = np.concatenate([
                graph.radii(sid) for sid in state["selected"] if graph.has_segment(sid)
            ])
            want = 2 * int(round(opt_k(radii_k) * float(np.median(radii)) / voxel)) + 1
            want = max(min(want, size_px.maximum()), size_px.minimum())
            was = int(size_px.value())
            size_px.setValue(want)
            say(f"frame {was} -> {want} px at one {voxel:.1f} um voxel per pixel, "
                f"about {opt_k(radii_k):g} median radii across. "
                f"'native' never magnifies, so this only changes how much you see.")
            return

        if geom.mode == "radius":
            half = float(np.median(geom.half_um))
            note = ("the median half-width; in 'radius' mode the scale varies per "
                    "plane, so this matches at the median only")
        else:
            half = float(np.max(geom.half_um))
            note = "the half-width every plane uses, so the match is exact"
        if geom.mode == "manual":
            px_um.setValue(voxel)

        want = 2 * int(round(half / max(voxel, 1e-9))) + 1
        want = max(min(want, size_px.maximum()), size_px.minimum())
        was = int(size_px.value())
        size_px.setValue(want)
        say(f"grid {was} -> {want} px, about one {voxel:.1f} um voxel per pixel "
            f"({half:,.0f} um half-width), from {note}. "
            f"Was sampling {np.median(geom.oversampling):.1f}x the voxel.")

    def do_show() -> None:
        if state["busy"]:
            say("still building the last one")
            return
        graph = state["graph"]
        session = getattr(app, "session", None)
        if graph is None or session is None:
            say("no dataset loaded - pick a graph in the Data tab")
            return
        if not state["selected"]:
            say("select at least one segment first")
            return

        opts = options()
        opts["step_um"] = opts["step_um"] or _default_step(app)
        labels = session.labels if with_mask.isChecked() else None
        state["busy"] = True
        show_button.setEnabled(False)
        say("building...")

        def work():
            return reformat_mod.build(
                graph, session.frame, session.stack, list(state["selected"]),
                labels=labels, **opts,
            )

        def done(result, error):
            state["busy"] = False
            show_button.setEnabled(True)
            if error is not None:
                say(f"{type(error).__name__}: {error}")
                return
            state["reformat"] = result
            save_button.setEnabled(True)
            refresh_save_hint()
            say(result.describe())
            status(f"[reformat] {result.n_planes} planes")
            _open_viewer(app, result, say, sections_in_3d=with_sections.isChecked())

        _run(work, done)

    def suggested() -> str:
        """The filename the Save dialog will offer, or '' when there is nothing to save."""
        stack = state["reformat"]
        if stack is None:
            return ""
        return reformat_io.suggest_name(stack, save_name.text().strip() or "reformat")

    def refresh_save_hint() -> None:
        """Show the full name before the dialog opens, not after."""
        name = suggested()
        if not name:
            save_hint.setText("build or load a stack first")
            return
        fmt = save_format.currentData()
        save_hint.setText(name + (".npz" if fmt == "npz" else "/  (a folder)"))

    def do_save() -> None:
        stack = state["reformat"]
        if stack is None:
            say("nothing to save - build a stack first, or load one")
            return
        fmt = save_format.currentData()
        name = suggested()
        session = getattr(app, "session", None)

        if fmt == "npz":
            chosen, _filter = QFileDialog.getSaveFileName(
                box, "Save the reformat stack", name + ".npz", "Reformat stack (*.npz)"
            )
        else:
            # A folder format needs a folder, and `getExistingDirectory` cannot offer a
            # name -- so ask where to put it and append the suggested name there.
            parent = QFileDialog.getExistingDirectory(
                box, f"Choose a folder to write '{name}' into"
            )
            chosen = str(Path(parent) / name) if parent else ""
        if not chosen:
            return

        def work():
            return reformat_io.save(
                stack, chosen, fmt=fmt,
                frame=getattr(session, "frame", None),
                graph=state["graph"],
                source=getattr(getattr(session, "graph", None), "path", None),
            )

        def done(result, error):
            state["busy"] = False
            _enable(True)
            if error is not None:
                say(f"{type(error).__name__}: {error}")
                return
            say(f"wrote {result}")
            status(f"[reformat] wrote {result}")

        state["busy"] = True
        _enable(False)
        say("saving...")
        _run(work, done)

    def do_load() -> None:
        chosen, _filter = QFileDialog.getOpenFileName(
            box, "Open a reformat stack", "",
            "Reformat stack (*.npz geometry.json);;All files (*)",
        )
        if not chosen:
            return
        session = getattr(app, "session", None)

        def work():
            loaded = reformat_io.load(chosen)
            return reformat_io.check_against(
                loaded, frame=getattr(session, "frame", None), graph=state["graph"]
            )

        def done(result, error):
            state["busy"] = False
            _enable(True)
            if error is not None:
                say(f"{type(error).__name__}: {error}")
                return
            state["reformat"] = result.reformat
            save_name.setText(Path(chosen).stem.split("__")[0])
            refresh_save_hint()
            say(result.describe())
            status(f"[reformat] loaded {result.path}")
            _open_viewer(app, result.reformat, say,
                         sections_in_3d=with_sections.isChecked(),
                         in_world=result.frame_matches)

        state["busy"] = True
        _enable(False)
        say("loading...")
        _run(work, done)

    def _enable(on: bool) -> None:
        loaded = state["graph"] is not None
        for widget in (add_button, clear_button, check_button, show_button,
                       match_button, cancel_trace_button):
            widget.setEnabled(on and loaded)
        load_button.setEnabled(on)
        save_button.setEnabled(on and state["reformat"] is not None)

    def _run(work, done) -> None:
        """Build off the GUI thread, finish on it.

        A long run is tens of thousands of TIFF decodes; doing that in a slot freezes
        the window for the duration. Falls back to running inline when there is no Qt
        thread to hand, which is what the tests drive.
        """
        try:
            from qtpy.QtCore import QThread
            from .edit.worker import qt_poster
        except Exception:  # noqa: BLE001 - no Qt: run inline
            try:
                done(work(), None)
            except Exception as exc:  # noqa: BLE001
                done(None, exc)
            return

        post = qt_poster()

        class _Job(QThread):
            def run(self):
                try:
                    out, err = work(), None
                except Exception as exc:  # noqa: BLE001 - reported, never raised
                    out, err = None, exc
                post(lambda: done(out, err))

        job = _Job()
        state["job"] = job  # keep it alive; a collected QThread aborts the process
        job.start()

    # ------------------------------------------------------------ wiring

    add_button.clicked.connect(guarded(do_add))
    clear_button.clicked.connect(guarded(do_clear))
    cancel_trace_button.clicked.connect(guarded(do_cancel_trace))
    # Leaving the mode must not leave a start armed behind it: the next plain pick would
    # be read as the far end of a trace that is no longer being made.
    trace_mode.toggled.connect(guarded(lambda: state.update(trace_start=None)))
    check_button.clicked.connect(guarded(do_check))
    save_button.clicked.connect(guarded(do_save))
    load_button.clicked.connect(guarded(do_load))
    save_name.textChanged.connect(guarded(refresh_save_hint))
    save_format.currentIndexChanged.connect(guarded(refresh_save_hint))
    match_button.clicked.connect(guarded(do_match_voxel))
    show_button.clicked.connect(guarded(do_show))
    mode.currentTextChanged.connect(guarded(lambda: _sync_mode(mode, radii_k, half_um, px_um)))

    def refresh() -> None:
        """Re-read the app and update the widgets.

        The graph is rebuilt only when the session's has actually changed: the selection
        is a list of segment ids, and rebuilding on every refresh would throw away the
        operator's picks every time another panel fired.
        """
        try:
            source = getattr(getattr(app, "session", None), "graph", None)
            if source is not state["source"]:
                rebuild_graph()
            loaded = state["graph"] is not None
            for widget in (add_button, clear_button, check_button, show_button,
                       match_button, cancel_trace_button):
                widget.setEnabled(loaded and not state["busy"])
            # Loading needs no dataset at all -- that is the case the feature is for.
            load_button.setEnabled(not state["busy"])
            save_button.setEnabled(state["reformat"] is not None and not state["busy"])
            refresh_save_hint()
            _sync_mode(mode, radii_k, half_um, px_um)
            refresh_lists()
            if not loaded:
                say("no dataset loaded - pick a graph in the Data tab")
            elif not state["selected"]:
                say("double-click a centreline point in the 3D view, then press '1'.")
        except Exception as exc:  # noqa: BLE001 - a panel must not take the window down
            say(f"reformat panel: {type(exc).__name__}: {exc}")

    def bind_keys(plotter) -> None:
        """Bind the 3D shortcuts, if this session has a window to bind them in.

        `add_key_event` appends rather than replaces and cannot be undone, so clear each
        key first and let every handler re-check that there is still a graph to act on.
        """
        if plotter is None or not hasattr(plotter, "add_key_event"):
            return
        for key, action in zip(KEYS, (do_add, do_clear, do_show, do_cancel_trace)):
            plotter.clear_events_for_key(key)
            plotter.add_key_event(key, guarded(action))

    box.refresh = refresh
    box.bind_keys = bind_keys
    box.state = state
    box.widgets = {
        "segments": seg_list, "summary": summary, "mode": mode, "size_px": size_px,
        "radii_k": radii_k, "half_um": half_um, "px_um": px_um, "step_um": step_um,
        "safety": safety, "passes": passes, "smooth_um": smooth_um,
        "with_mask": with_mask, "with_sections": with_sections,
        "order": order, "match": match_button,
        "save_name": save_name, "save_format": save_format, "save_hint": save_hint,
        "save": save_button, "load": load_button,
        "add": add_button, "clear": clear_button,
        "trace_mode": trace_mode, "prefer_thick": prefer_thick,
        "cancel_trace": cancel_trace_button,
        "check": check_button, "show": show_button,
    }
    refresh()
    return box


def opt_k(widget) -> float:
    """The radius multiplier, read from the widget rather than from `options()`.

    `native` mode ignores `radii_k` for the geometry, but "match voxel" still uses it
    to decide how many radii of context the frame should hold.
    """
    return float(widget.value())


def _sync_mode(mode, radii_k, half_um, px_um) -> None:
    """Only the widgets the chosen mode actually reads are live.

    ``native`` reads neither: its half-width comes from the frame size and its pitch
    from the voxel, so both the radius multiplier and the manual pair are inert.
    """
    chosen = mode.currentText()
    manual = chosen == "manual"
    radii_k.setEnabled(chosen in ("radius", "fixed"))
    half_um.setEnabled(manual)
    px_um.setEnabled(manual)


def _half_of(opts, voxel_um=0.0):
    """The half-width the chosen mode would produce, as a function of the radii.

    A callable because the bound has to be re-checked on every smoothing pass against
    the width *that* pass would produce, and resampling changes the radii.
    """
    if opts["mode"] == "native":
        # The frame size decides the width here, not the radii.
        half = (max(int(opts["size_px"]) | 1, 3) // 2) * voxel_um
        return lambda r: np.full(len(r), half)
    if opts["mode"] == "manual" and opts["half_um"]:
        return lambda r: np.full(len(r), float(opts["half_um"]))
    if opts["mode"] == "fixed":
        return lambda r: np.full(len(r), opts["radii_k"] * float(np.max(r)))
    return lambda r: opts["radii_k"] * r


def _default_step(app) -> float:
    """One raw voxel: below that the reformat is inventing detail."""
    frame = getattr(getattr(app, "session", None), "frame", None)
    return float(np.min(frame.raw_voxel)) if frame is not None else 1.0


def _open_viewer(app, result, say, *, sections_in_3d: bool = False,
                 in_world: bool = True) -> None:
    """Show the stack, deferring the window out of whatever callback we are in.

    Creating Qt/napari/vispy objects from inside a VTK key callback is an access
    violation rather than an exception, so when this is reached from a keypress the
    window has to be built after the handler returns.

    ``in_world=False`` shows the images with no 3D overlays -- a loaded stack whose
    coordinate frame is not the one currently open, or none being open at all.
    """
    def run():
        try:
            app.open_reformat(result, sections_in_3d=sections_in_3d, in_world=in_world)
        except Exception as exc:  # noqa: BLE001
            say(f"could not open the reformat window: {type(exc).__name__}: {exc}")

    try:
        from qtpy.QtCore import QTimer

        QTimer.singleShot(0, run)
    except Exception:  # noqa: BLE001 - no Qt binding: the tests call it directly
        run()
