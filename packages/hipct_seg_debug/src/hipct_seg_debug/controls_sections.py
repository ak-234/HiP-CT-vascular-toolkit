"""The Sections tab: look at the planes ``radius-perimeter`` measured a radius in.

The measurement pass reports a radius and a reason code per point. When the radii
come back wrong, neither says *why*, because both are downstream of the one thing
neither records: the square of segmentation the number was taken from -- where it sat,
how big it was, and at what angle it was cut. This panel
re-cuts that square -- through the same tangents and the same
:func:`~.crosssection.stable_transverse_cut` the pass uses -- at sampled points of
the chosen segments, and hands the geometry to the 3D window.

**Sampled, not every point.** A section costs up to three plane samples per candidate
tangent, so cutting a whole tree here would cost what the measurement costs. Every
Nth point of each segment, both ends always included, is enough to see whether the
windows sit on the lumen and whether they contain it -- and the ends are where the
junction mask bites, so they are never the points to drop.

Nothing here writes anything, and nothing it draws is pickable.

Cutting is seconds to a minute, so it runs on a worker thread, exactly as the
Reformat tab's build does. Qt is imported inside the function like every other panel.
"""

from __future__ import annotations

import numpy as np

from .edit import section_frames as sf

#: Keys this panel binds in the 3D window: add the picked segment, clear, cut,
#: cancel a half-finished trace.
#:
#: Digits again, and high ones. Every letter is taken (`viewer3d` owns `v n b s c i g
#: a r q` plus shift-`L`, the edit controller `e z y d t x k j u f`, Crop `m l o h`),
#: Reformat has `1 2 4 8`, and `3` is VTK's stereo toggle, which lives in the
#: interactor style where `clear_events_for_key` cannot reach it. That leaves `9` and
#: `0`, and `tests/test_reformat_panel.py` asserts the whole set stays disjoint --
#: `bind_keys` clears a key before binding it, so a collision is silent theft.
KEYS = ("5", "6", "7", "9")

#: Default points between sampled sections. Twelve is roughly a section every 1-2 mm
#: on LADAF-2024-28's point spacing, which is dense enough to see a window drift off
#: the lumen and coarse enough to cut a whole tree in under a minute.
DEFAULT_STRIDE = 12

#: Default cap on how many sections one run will cut, whatever the stride asks for.
DEFAULT_MAX_FRAMES = 600


def build_sections_panel(app):
    """Return the Sections widget for a `ViewerApp`. Docking is the caller's business."""
    from qtpy.QtWidgets import (
        QCheckBox,
        QHBoxLayout,
        QLabel,
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
        "graph": None,     # our private EditableGraph, as Crop and Reformat keep
        "source": None,    # the SpatialGraph it was built from, for identity
        "selected": [],    # segment ids, or empty for the whole tree
        "survey": None,    # the last survey, kept so the table can be re-printed
        "busy": False,
        "trace_start": None,  # the first pick of a trace, waiting for its far end
    }

    banner = QLabel(
        "Re-cuts the cross-sections `radius-perimeter` measures, at sampled points, "
        "and draws each window, the lumen it measured and how far that lumen's "
        "centroid sits from the centreline point. Nothing here writes anything."
    )
    banner.setWordWrap(True)
    banner.setStyleSheet("color: #808090;")
    lay.addWidget(banner)

    lay.addWidget(QLabel("segments (empty = whole tree; double-click a point, then '5';\n"
                         "with 'trace' on, pick both ends of a vessel)"))
    seg_list = QListWidget()
    seg_list.setMaximumHeight(90)
    lay.addWidget(seg_list)

    pick_row = QHBoxLayout()
    add_button = QPushButton("Add pick (5)")
    clear_button = QPushButton("Clear (6)")
    pick_row.addWidget(add_button)
    pick_row.addWidget(clear_button)
    lay.addLayout(pick_row)

    # -- naming a whole vessel rather than clicking every segment of it --------
    #
    # The same two-pick trace the Crop tab uses, and for the same reason: a main
    # vessel is forty segments and the question asked here -- 'are the sections
    # along this vessel being measured' -- is about the vessel, not about one
    # segment of it. `crop.trace_path` is a Dijkstra over nodes, so the route is a
    # simple path; on a tree it is the only path and no weighting can change it.
    trace_row = QHBoxLayout()
    trace_mode = QCheckBox("trace between two picks")
    trace_mode.setToolTip(
        "Add pick twice: every segment on the path between the two joins the\n"
        "selection, so a whole vessel can be cut in two clicks."
    )
    prefer_thick = QCheckBox("prefer thick")
    prefer_thick.setToolTip(
        "Weight the route by each segment's own radius instead of arclength\n"
        "alone, so a trace goes down the vessel instead of over a thin false\n"
        "bridge. On a tree the path is unique and this changes nothing."
    )
    cancel_trace_button = QPushButton("Cancel trace (9)")
    trace_row.addWidget(trace_mode)
    trace_row.addWidget(prefer_thick)
    trace_row.addWidget(cancel_trace_button)
    lay.addLayout(trace_row)

    # -- sampling ---------------------------------------------------------
    stride_row = QHBoxLayout()
    stride_row.addWidget(QLabel("every N points"))
    stride = QSpinBox()
    stride.setRange(1, 500)
    stride.setValue(DEFAULT_STRIDE)
    stride.setToolTip("Both ends of every segment are cut whatever this says.")
    stride_row.addWidget(stride, 1)
    stride_row.addWidget(QLabel("max sections"))
    max_frames = QSpinBox()
    max_frames.setRange(1, 20000)
    max_frames.setValue(DEFAULT_MAX_FRAMES)
    stride_row.addWidget(max_frames, 1)
    lay.addLayout(stride_row)

    # -- the geometry under suspicion -------------------------------------
    half_row = QHBoxLayout()
    half_row.addWidget(QLabel("max half (vox)"))
    max_half = QSpinBox()
    max_half.setRange(4, 512)
    max_half.setValue(128)
    max_half.setToolTip(
        "Largest half-width the window may grow to, as `radius-perimeter --max-half`.\n"
        "A section still touching the border here is refused and its radius filled in."
    )
    half_row.addWidget(max_half, 1)
    half_row.addWidget(QLabel("start half (0 = from radius)"))
    start_half = QSpinBox()
    start_half.setRange(0, 512)
    start_half.setValue(0)
    start_half.setToolTip(
        "Force the starting window instead of taking 2.5x the stored radius.\n"
        "The direct test of 'the planes are not big enough': raise it and see\n"
        "whether the truncated windows become measured ones."
    )
    half_row.addWidget(start_half, 1)
    lay.addLayout(half_row)

    grow_row = QHBoxLayout()
    grow_row.addWidget(QLabel("grow ceiling (radii, 0 = none)"))
    grow_ceiling = QSpinBox()
    grow_ceiling.setRange(0, 64)
    grow_ceiling.setValue(0)
    grow_ceiling.setToolTip(
        "Ceiling on how far one window may double, in multiples of *that point's*\n"
        "own stored radius -- `radius-perimeter --grow-radii`, the same number.\n\n"
        "Zero is no ceiling, which is what the pass defaults to. So zero is what\n"
        "reproduces the pass, and zero is also what produces a runaway window. This\n"
        "box does not paper over that; the 'h/r' column labels it.\n\n"
        "`cut` doubles the window while the blob touches its border. That is right\n"
        "when the vessel is genuinely bigger than the window, and catastrophic when\n"
        "the plane is not transverse: the section is then a streak *along* the\n"
        "vessel, which touches at any width, so it escalates to 'max half'. Measured\n"
        "here: a 260 um vessel at max half 128 grew to half=88 -- a 2.8 mm window --\n"
        "swallowed the lumen beside it and reported r=1062 um at 3.7:1.\n\n"
        "Four is the usual answer: a vessel that genuinely needs a wide window has a\n"
        "wide radius, so the ceiling does not bind on it. Below about 2.5 nothing can\n"
        "grow at all, because the window already starts at 2.5 radii."
    )
    grow_row.addWidget(grow_ceiling, 1)
    merge_check = QCheckBox("check for merged neighbours")
    merge_check.setChecked(True)
    merge_check.setToolTip(
        "Count, per section, how many *other* segments' centreline samples lie\n"
        "inside this section's own blob -- the same test the pass uses to decide\n"
        "ownership, reported in the 'merge' column as foreign/total.\n\n"
        "This is the one question 'shape' and 'obliq' cannot answer. A collapsed\n"
        "lumen and two vessels measured as one are both elongated at obliquity 1.00,\n"
        "because rotating the plane shortens neither. A foreign centreline inside\n"
        "the blob is what tells them apart.\n\n"
        "Costs one KD-tree over every centreline point of the whole graph, whether\n"
        "one segment is selected or all of them -- a rival is by definition not in\n"
        "the selection. Turn it off on a big tree when the question is about window\n"
        "size rather than ownership."
    )
    grow_row.addWidget(merge_check)
    lay.addLayout(grow_row)

    search_row = QHBoxLayout()
    search_row.addWidget(QLabel("tangent search (deg)"))
    search_deg = QSpinBox()
    search_deg.setRange(0, 45)
    search_deg.setValue(20)
    search_row.addWidget(search_deg, 1)
    branch_aware = QCheckBox("branch-aware (as the pass runs)")
    branch_aware.setChecked(True)
    branch_aware.setToolTip(
        "Off uses a single plain cut on the fitted tangent, with no stability slab -- "
        "the legacy selector, and a useful contrast when the stable cut refuses."
    )
    search_row.addWidget(branch_aware)
    lay.addLayout(search_row)

    trust_fitted = QCheckBox("trust the fitted tangent (no re-cut of flat sections)")
    trust_fitted.setToolTip(
        "The direct test of 'the planes are cut at the wrong axis'.\n\n"
        "The stability slab is stepped along the candidate normal, so on a straight\n"
        "vessel an oblique plane cuts three identical ellipses and scores a perfect\n"
        "1.000 -- tilt is the one error it cannot see. The pass therefore re-cuts any\n"
        "section flatter than 1.10:1 over the search cone and keeps the shortest\n"
        "boundary. Tick this to skip that and take the fitted tangent whenever it was\n"
        "stable, which is what the pass did before: the sections whose radius grows\n"
        "are the oblique ones, and 'obliquity' in the summary is by how much."
    )
    lay.addWidget(trust_fitted)

    with_table = QCheckBox("print the per-section table to the log")
    with_table.setChecked(True)
    lay.addWidget(with_table)

    action_row = QHBoxLayout()
    cut_button = QPushButton("Cut sections (7)")
    hide_button = QPushButton("Hide")
    action_row.addWidget(cut_button)
    action_row.addWidget(hide_button)
    lay.addLayout(action_row)

    summary = QLabel("")
    summary.setWordWrap(True)
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

        The same chain the Crop and Reformat tabs use: the pick is an index into
        ``graph.points``, and ``point_order`` is the flat concatenation the renderer
        drew, so the two line up by construction.
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
        state["graph"] = (
            None if source is None else EditableGraph(from_spatial_graph(source))
        )
        state["selected"] = []
        state["survey"] = None
        clear = getattr(getattr(app, "picker", None), "clear_cross_sections", None)
        if clear is not None:
            try:
                clear()
            except Exception:  # noqa: BLE001 - a dead plotter must not block a load
                pass

    def refresh_lists() -> None:
        seg_list.clear()
        graph = state["graph"]
        for sid in state["selected"]:
            if graph is not None and graph.has_segment(sid):
                n = len(graph.coords(sid))
                seg_list.addItem(f"segment {sid} - {n} points")
            else:
                seg_list.addItem(f"segment {sid} - gone")

    def draw(survey) -> None:
        show = getattr(getattr(app, "picker", None), "show_cross_sections", None)
        if show is None:
            return
        good, bad, contours, offsets, refused = sf.drawables(survey)
        show(corners=good, truncated=bad, contours=contours, offsets=offsets,
             refused=refused)

    # Every slot funnels its failure into the summary label: PyQt5 calls qFatal on an
    # exception that escapes a slot, so a cut that cannot be taken must not take the
    # window down with it. `_arg=None` rather than `*_args` because this wrapper is
    # handed to `plotter.add_key_event` too, and pyvista refuses a callback whose
    # parameters have no defaults.
    def guarded(fn):
        def run(_arg=None):
            try:
                fn()
            except Exception as exc:  # noqa: BLE001 - see above
                say(f"{type(exc).__name__}: {exc}")
        return run

    # ------------------------------------------------------------ actions

    def do_add() -> None:
        picked = picked_segment()
        if picked is None:
            say("nothing picked - double-click a centreline point in the 3D view")
            return
        sid, _pid = picked
        if trace_mode.isChecked():
            do_trace(sid)
            return
        # A toggle, like the other tabs: picking the same segment twice is far more
        # likely to be a correction than a request to include it again.
        if sid in state["selected"]:
            state["selected"].remove(sid)
        else:
            state["selected"].append(sid)
        refresh_lists()
        say(f"{len(state['selected'])} segment(s) selected"
            if state["selected"] else "whole tree")

    def do_trace(sid: int) -> None:
        """Second half of a two-pick trace; the first just remembers where to start.

        A union rather than a replacement, as in Crop: tracing the LAD and then its
        ostium stub is one vessel in two goes, and the second trace must not discard
        the first. Segments already selected are left alone rather than toggled off,
        because a trace that silently removed the overlap with an earlier one would
        be impossible to reason about.
        """
        from .edit import crop as crop_mod

        if state["trace_start"] is None:
            state["trace_start"] = sid
            say(f"trace: start at segment {sid}. Pick the far end and press Add pick "
                "again, or Cancel trace (9).")
            return
        start = state["trace_start"]
        state["trace_start"] = None
        traced = crop_mod.trace_path(
            state["graph"], start, sid, prefer_thick=prefer_thick.isChecked()
        )
        added = [s for s in traced.segments if s not in state["selected"]]
        state["selected"].extend(added)
        refresh_lists()
        say(f"traced {traced.describe()}; {len(added)} new, "
            f"{len(state['selected'])} segment(s) selected")

    def do_cancel_trace() -> None:
        if state["trace_start"] is None:
            say("no trace in progress")
            return
        state["trace_start"] = None
        say("trace cancelled - the start pick was dropped")

    def do_clear() -> None:
        state["selected"] = []
        state["trace_start"] = None
        refresh_lists()
        say("selection cleared - the next cut covers the whole tree")

    def do_hide() -> None:
        clear = getattr(getattr(app, "picker", None), "clear_cross_sections", None)
        if clear is not None:
            clear()
        say("sections hidden - they are still in the log")

    def do_cut() -> None:
        session = getattr(app, "session", None)
        graph = state["graph"]
        if graph is None or session is None or getattr(session, "labels", None) is None:
            say("no dataset loaded - pick a graph and a segmentation in the Data tab")
            return
        if state["busy"]:
            return
        sids = list(state["selected"]) or None
        options = dict(
            stride=int(stride.value()),
            max_frames=int(max_frames.value()),
            max_half=int(max_half.value()),
            initial_half=int(start_half.value()) or None,
            grow_radii=float(grow_ceiling.value()) or None,
            rival_check=bool(merge_check.isChecked()),
            tangent_search_degrees=float(search_deg.value()),
            branch_aware=bool(branch_aware.isChecked()),
            transverse_axis_ratio=(
                float("inf") if trust_fitted.isChecked() else None
            ),
        )

        def work():
            return sf.survey(graph, session.frame, session.labels, sids, **options)

        def done(survey, error) -> None:
            state["busy"] = False
            _enable(True)
            if error is not None:
                say(f"{type(error).__name__}: {error}")
                return
            state["survey"] = survey
            say(survey.describe())
            status(f"[sections] {survey.describe()}")
            if with_table.isChecked():
                for line in survey.table():
                    status("  " + line)
            try:
                draw(survey)
            except Exception as exc:  # noqa: BLE001 - a dead plotter is not a failure
                say(f"{survey.describe()}\n(could not draw: {exc})")

        state["busy"] = True
        _enable(False)
        # The branch index is built before the first progress callback, so a
        # whole-tree cut with the merge check on sits silent for a second or two.
        say("building the branch index, then cutting sections..."
            if merge_check.isChecked() else "cutting sections...")
        _run(work, done)

    def _enable(on: bool) -> None:
        loaded = state["graph"] is not None
        for widget in (add_button, clear_button, cut_button, hide_button,
                       cancel_trace_button):
            widget.setEnabled(on and loaded)

    def _run(work, done) -> None:
        """Cut off the GUI thread, finish on it.

        A few hundred sections is seconds of plane sampling and RLE decode; in a slot
        that is a frozen window. Falls back to running inline when there is no Qt
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
    cancel_trace_button.clicked.connect(guarded(do_cancel_trace))
    # A half-finished trace must not survive the mode being switched off and on.
    trace_mode.toggled.connect(guarded(lambda: state.update(trace_start=None)))
    clear_button.clicked.connect(guarded(do_clear))
    cut_button.clicked.connect(guarded(do_cut))
    hide_button.clicked.connect(guarded(do_hide))

    def refresh() -> None:
        """Re-read the app and update the widgets.

        The graph is rebuilt only when the session's has actually changed, so another
        panel firing does not throw away the operator's selection.
        """
        try:
            source = getattr(getattr(app, "session", None), "graph", None)
            if source is not state["source"]:
                rebuild_graph()
            loaded = state["graph"] is not None
            for widget in (add_button, clear_button, cut_button, hide_button,
                       cancel_trace_button):
                widget.setEnabled(loaded and not state["busy"])
            refresh_lists()
            if not loaded:
                say("no dataset loaded - pick a graph in the Data tab")
            elif state["survey"] is None:
                say("press 'Cut sections' for the whole tree, or add segments first")
        except Exception as exc:  # noqa: BLE001 - a panel must not take the window down
            say(f"sections panel: {type(exc).__name__}: {exc}")

    def bind_keys(plotter) -> None:
        """Bind the 3D shortcuts, if this session has a window to bind them in.

        `add_key_event` appends rather than replaces and cannot be undone, so each key
        is cleared first and every handler re-checks that there is still a graph.
        """
        if plotter is None or not hasattr(plotter, "add_key_event"):
            return
        for key, action in zip(KEYS, (do_add, do_clear, do_cut, do_cancel_trace)):
            plotter.clear_events_for_key(key)
            plotter.add_key_event(key, guarded(action))

    box.refresh = refresh
    box.bind_keys = bind_keys
    box.state = state
    box.widgets = {
        "segments": seg_list, "summary": summary, "stride": stride,
        "max_frames": max_frames, "max_half": max_half, "start_half": start_half,
        "search_deg": search_deg, "branch_aware": branch_aware,
        "trace_mode": trace_mode, "prefer_thick": prefer_thick,
        "cancel_trace": cancel_trace_button,
        "add": add_button, "clear": clear_button,
        "grow_ceiling": grow_ceiling, "merge_check": merge_check,
        "trust_fitted": trust_fitted,
        "with_table": with_table, "cut": cut_button, "hide": hide_button,
    }
    refresh()
    return box


def worst_offsets(survey, limit: int = 10):
    """The frames whose centroid sits furthest from their centreline point.

    Separated from the panel so the ranking -- which is the answer to "does this need
    re-centring" -- can be asked for from a script or a test without Qt.
    """
    frames = [f for f in getattr(survey, "frames", survey) if np.isfinite(f.offset_radii)]
    return sorted(frames, key=lambda f: -f.offset_radii)[:limit]
