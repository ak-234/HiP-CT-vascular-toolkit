"""The Crop tab: name the main vessels, set a rule, see what it would take.

**Nothing here writes to the dataset, and nothing here writes a graph.** Picking
vessels and marking branches records a *rule and a selection* into a crop sidecar;
producing a cropped ``.am`` is a separate ``crop --crop-json ...`` run, which the
`Queue this command` button will start for you. That is the same explicit-apply
division the Reconnect tab uses, and it buys the same two things: a crop can be
designed on a machine that has the graph but not the mask or the surface, and no
amount of clicking in here can cost you a graph you have not saved.

The panel keeps its **own** ``EditableGraph``, built from the session's in-memory
spatial graph. Two reasons, and the second is the load-bearing one:

* `crop` works in segment ids and needs an incidence index, which a ``SpatialGraph``
  does not have;
* it is a copy, so the tree you are looking at cannot be modified by anything this
  panel does -- including by a bug in it.

Picks arrive as an index into ``graph.points``, which ``point_order`` maps to a point
id and ``segment_of_point`` to a segment. That is the same chain `edit/controller.py`
uses, and it holds because ``centreline_polydata`` renders points in exactly that
order (guarded by ``selftest.test_centreline_point_ids``).

There are two ways to name a vessel and the `trace between two picks` box chooses
between them: one pick per segment, or two picks -- an ostium and a far end -- with
`crop.trace_path` taking the path that runs between them. What lands in the state is
the same thing either way, a set of segment ids under a name, so the sidecar, the
ostium and every rule below it cannot tell which was used.

Qt is imported inside the function, matching every other panel, so this module still
imports in a session with no Qt binding.
"""

from __future__ import annotations

from .edit import crop as crop_mod

#: Keys this panel binds in the 3D window. None of them collide with `viewer3d`'s own
#: (`v n b s c i g a r q`), the edit controller's (`e z y d t x k j u f`), or VTK's
#: built-ins (`r` reset, `q` close, `w` wireframe, `p` prop-pick).
KEYS = ("m", "l", "o", "h")


def build_crop_panel(app, runner=None, spec=None):
    """Return the Crop widget for a `ViewerApp`. Docking is the caller's business.

    ``runner`` and ``spec`` are optional so the panel can be built and driven in a
    test with neither a command queue nor an argparse spec behind it.
    """
    from qtpy.QtCore import Qt
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
        "path": None,        # the sidecar path
        "vessels": {},       # name -> set of segment ids
        "colors": {},        # name -> hex
        "drop_segments": [], # hand-marked, this segment only
        "trace_start": None, # the first pick of a trace, waiting for its far end
        "prune_at": [],      # hand-marked, (segment id, from node)
        "plan": None,
    }

    banner = QLabel(
        "Nothing here writes to the dataset. This records a rule and a selection "
        "into a crop sidecar; the cropped .am comes from a separate `crop` run."
    )
    banner.setWordWrap(True)
    banner.setStyleSheet("color: #808090;")
    lay.addWidget(banner)

    # -- sidecar ----------------------------------------------------------
    sidecar_row = QHBoxLayout()
    sidecar_path = QLineEdit()
    sidecar_path.setPlaceholderText("crop sidecar (*.json)")
    open_button = QPushButton("Open...")
    save_button = QPushButton("Save")
    sidecar_row.addWidget(sidecar_path, 1)
    sidecar_row.addWidget(open_button)
    sidecar_row.addWidget(save_button)
    lay.addLayout(sidecar_row)

    # -- main vessels -----------------------------------------------------
    lay.addWidget(QLabel("main vessels"))
    vessel_row = QHBoxLayout()
    vessel_name = QComboBox()
    vessel_name.setEditable(True)
    vessel_name.addItems(crop_mod.PRESET_VESSEL_NAMES)
    swatch = QLabel("    ")
    swatch.setFixedWidth(18)
    add_button = QPushButton("Add pick (m)")
    remove_button = QPushButton("Remove pick")
    vessel_row.addWidget(vessel_name, 1)
    vessel_row.addWidget(swatch)
    vessel_row.addWidget(add_button)
    vessel_row.addWidget(remove_button)
    lay.addLayout(vessel_row)

    vessel_list = QListWidget()
    vessel_list.setMaximumHeight(90)
    lay.addWidget(vessel_list)

    vessel_buttons = QHBoxLayout()
    forget_button = QPushButton("Delete vessel")
    clear_vessels_button = QPushButton("Clear all")
    vessel_buttons.addWidget(forget_button)
    vessel_buttons.addWidget(clear_vessels_button)
    lay.addLayout(vessel_buttons)

    # -- tracing ----------------------------------------------------------
    # A mode rather than a third button, so `m` keeps meaning "the pick belongs to
    # this vessel" in both: what changes is how much of the vessel one press names.
    trace_row = QHBoxLayout()
    trace_mode = QCheckBox("trace between two picks")
    trace_mode.setToolTip(
        "Add pick (m) sets the ostium, the next one sets the far end, and everything "
        "on the path between them joins the vessel."
    )
    prefer_thick = QCheckBox("prefer thick")
    prefer_thick.setToolTip(
        "Route by length in units of each segment's own radius rather than by length "
        "alone, so a trace goes down the vessel instead of over a thin false bridge. "
        "On a tree the path is unique and this changes nothing."
    )
    cancel_trace_button = QPushButton("Cancel trace")
    trace_row.addWidget(trace_mode)
    trace_row.addWidget(prefer_thick)
    trace_row.addWidget(cancel_trace_button)
    lay.addLayout(trace_row)

    # -- rule -------------------------------------------------------------
    lay.addWidget(QLabel("rule"))

    def _threshold(label, widget, *, checked=False):
        row = QHBoxLayout()
        check = QCheckBox(label)
        check.setChecked(checked)
        widget.setEnabled(checked)
        check.toggled.connect(widget.setEnabled)
        row.addWidget(check)
        row.addWidget(widget, 1)
        lay.addLayout(row)
        return check

    strahler = QSpinBox()
    strahler.setRange(1, 12)
    strahler.setValue(2)
    use_strahler = _threshold("min Strahler order", strahler)

    min_radius = QDoubleSpinBox()
    min_radius.setRange(0.0, 100000.0)
    min_radius.setDecimals(1)
    min_radius.setValue(100.0)
    min_radius.setSuffix(" um")
    use_radius = _threshold("min take-off radius", min_radius)

    denominator = QDoubleSpinBox()
    denominator.setRange(1.0, 1000.0)
    denominator.setDecimals(1)
    denominator.setValue(4.0)
    denominator.setPrefix("1 / ")
    use_ratio = _threshold("fraction of the main vessel's ostial radius", denominator)

    unattributed = QCheckBox("also drop subtrees off no named vessel")
    lay.addWidget(unattributed)

    # -- hand marks -------------------------------------------------------
    lay.addWidget(QLabel("marked by hand"))
    manual_list = QListWidget()
    manual_list.setMaximumHeight(70)
    lay.addWidget(manual_list)

    manual_row = QHBoxLayout()
    omit_button = QPushButton("Drop pick (o)")
    prune_button = QPushButton("Prune past pick (h)")
    unmark_button = QPushButton("Unmark")
    manual_row.addWidget(omit_button)
    manual_row.addWidget(prune_button)
    manual_row.addWidget(unmark_button)
    lay.addLayout(manual_row)

    # -- preview and hand-off ---------------------------------------------
    preview_row = QHBoxLayout()
    preview_button = QPushButton("Preview")
    out_path = QLineEdit()
    out_path.setPlaceholderText("cropped graph (*.am)")
    preview_row.addWidget(preview_button)
    preview_row.addWidget(out_path, 1)
    lay.addLayout(preview_row)

    summary = QLabel("no dataset loaded")
    summary.setWordWrap(True)
    summary.setTextInteractionFlags(Qt.TextSelectableByMouse)
    lay.addWidget(summary, 1)

    command_row = QHBoxLayout()
    command_line = QLineEdit()
    command_line.setReadOnly(True)
    queue_button = QPushButton("Queue this command")
    command_row.addWidget(command_line, 1)
    command_row.addWidget(queue_button)
    lay.addLayout(command_row)

    # ------------------------------------------------------------ helpers

    def status(message: str) -> None:
        if getattr(app, "on_status", None):
            app.on_status(message)

    def say(message: str) -> None:
        summary.setText(message)

    def colour_for(name: str) -> str:
        if name not in state["colors"]:
            index = len(state["colors"]) % len(crop_mod.VESSEL_COLORS)
            state["colors"][name] = crop_mod.VESSEL_COLORS[index]
        return state["colors"][name]

    def current_vessel() -> str:
        return vessel_name.currentText().strip()

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

    def rule() -> crop_mod.Rule:
        return crop_mod.Rule(
            min_strahler=strahler.value() if use_strahler.isChecked() else None,
            min_ostium_um=min_radius.value() if use_radius.isChecked() else None,
            ratio=(1.0 / denominator.value()) if use_ratio.isChecked() else None,
            prune_unattributed=unattributed.isChecked(),
        )

    def values() -> dict:
        current = rule()
        return {
            "graph": str(getattr(state["source"], "path", "") or ""),
            "crop_json": sidecar_path.text().strip(),
            "out": out_path.text().strip(),
            "min_strahler": current.min_strahler,
            "min_ostium_um": current.min_ostium_um,
            "ratio": current.ratio,
            "prune_unattributed": current.prune_unattributed,
        }

    def refresh_command() -> None:
        if spec is None:
            command_line.setText("")
            return
        from . import cliform

        command_line.setText(
            cliform.command_line(spec, values(), prog="python -m hipct_seg_debug.edit")
        )

    def refresh_lists() -> None:
        vessel_list.clear()
        ostia = (state["plan"].ostia if state["plan"] else {})
        for name in sorted(state["vessels"]):
            ids = state["vessels"][name]
            ostium = ostia.get(name) or {}
            radius = ostium.get("radius_um")
            detail = f", ostium R = {radius:,.0f} um" if radius else ""
            vessel_list.addItem(f"{name} - {len(ids)} segment(s){detail}")
        manual_list.clear()
        for sid in state["drop_segments"]:
            manual_list.addItem(f"drop segment {sid}")
        for sid, node in state["prune_at"]:
            manual_list.addItem(f"prune past segment {sid} from node {node}")

    def draw() -> None:
        """Hand the preview to the viewer, if there is one to hand it to."""
        show = getattr(getattr(app, "picker", None), "show_crop", None)
        graph = state["graph"]
        if show is None or graph is None:
            return
        plan = state["plan"]
        drop = [graph.coords(sid) for sid in sorted(plan.drop)
                if graph.has_segment(sid)] if plan else []
        vessels = [
            (name, colour_for(name),
             [graph.coords(sid) for sid in sorted(ids) if graph.has_segment(sid)])
            for name, ids in sorted(state["vessels"].items())
        ]
        try:
            show(drop, vessels)
        except Exception as exc:  # noqa: BLE001 - see the note on slots below
            status(f"  could not draw the crop preview: {exc}")

    def rebuild_graph() -> None:
        """Take a private copy of the session's graph, dropping any stale selection."""
        from .edit.adapter import from_spatial_graph
        from .edit.graphmodel import EditableGraph

        source = getattr(getattr(app, "session", None), "graph", None)
        state.update(source=source, plan=None)
        state["graph"] = None if source is None else EditableGraph(from_spatial_graph(source))
        # A selection is a set of segment ids in the graph it was made on, and means
        # something else entirely in the next one.
        state.update(vessels={}, drop_segments=[], prune_at=[], trace_start=None)
        clear = getattr(getattr(app, "picker", None), "clear_crop", None)
        if clear is not None:
            try:
                clear()
            except Exception:  # noqa: BLE001
                pass

    def resolve_into(document) -> None:
        graph = state["graph"]
        resolved = crop_mod.resolve(graph, document)
        state["vessels"] = dict(resolved.vessels)
        state["colors"].update(resolved.colors)
        state["drop_segments"] = list(resolved.drop_segments)
        state["prune_at"] = list(resolved.prune_at)
        state["trace_start"] = None
        for note in resolved.notes:
            status(f"  note: {note}")
        saved = resolved.rule
        for widget, check, value in (
            (strahler, use_strahler, saved.min_strahler),
            (min_radius, use_radius, saved.min_ostium_um),
            (denominator, use_ratio, (1.0 / saved.ratio) if saved.ratio else None),
        ):
            check.blockSignals(True)
            widget.blockSignals(True)
            check.setChecked(value is not None)
            widget.setEnabled(value is not None)
            if value is not None:
                widget.setValue(value)
            check.blockSignals(False)
            widget.blockSignals(False)
        unattributed.blockSignals(True)
        unattributed.setChecked(saved.prune_unattributed)
        unattributed.blockSignals(False)

    # Every slot below funnels its failure into the summary label. PyQt5 calls
    # qFatal on an exception that escapes a slot, which aborts the process rather
    # than raising -- a broken sidecar must not take the window down with it.
    #
    # `_arg=None` rather than the obvious `*_args`, and the difference is load-bearing:
    # this wrapper is handed both to Qt signals and to `plotter.add_key_event`, and
    # pyvista rejects any callback with a parameter that has no default. It walks
    # `signature(callback).parameters`, and a `*args` entry always reports no default,
    # so `*_args` is refused even though it is callable with zero arguments -- a
    # TypeError out of `attach_control_dock` that takes the session down before the
    # window opens. One default-valued positional absorbs everything Qt emits here
    # (`clicked(bool)`, `toggled(bool)`, `valueChanged(int|double)`) and satisfies
    # pyvista. Same shape as `edit/controls_edit._bind`.
    def guarded(fn):
        def run(_arg=None):
            try:
                fn()
            except Exception as exc:  # noqa: BLE001 - see above
                say(f"{type(exc).__name__}: {exc}")
            refresh_command()
        return run

    # ------------------------------------------------------------ actions

    def do_open() -> None:
        path, _filter = QFileDialog.getOpenFileName(
            box, "Open a crop sidecar", sidecar_path.text(), "JSON (*.json)"
        )
        if path:
            load(path)

    def load(path) -> None:
        """Read a sidecar and place it on the loaded graph."""
        if state["graph"] is None:
            say("load a graph before opening a sidecar")
            return
        document = crop_mod.load(path)
        resolve_into(document)
        state["path"] = str(path)
        sidecar_path.setText(str(path))
        refresh_lists()
        say(f"loaded {path}: {len(state['vessels'])} main vessel(s). Press Preview.")

    def do_save() -> None:
        if state["graph"] is None:
            say("nothing to save: no graph loaded")
            return
        path = sidecar_path.text().strip()
        if not path:
            path, _filter = QFileDialog.getSaveFileName(
                box, "Save the crop sidecar", "", "JSON (*.json)"
            )
            if not path:
                return
            sidecar_path.setText(path)
        graph = state["graph"]
        plan = state["plan"] or crop_mod.plan(graph, rule(), vessels=state["vessels"])
        document = crop_mod.document(
            graph, plan, rule(), source=getattr(state["source"], "path", None),
            out=out_path.text().strip() or None, colors=state["colors"],
        )
        crop_mod.carry_manual(document, graph, state["drop_segments"], state["prune_at"])
        state["path"] = str(crop_mod.write(path, document))
        say(f"wrote {state['path']}. Run `crop --crop-json` to write the graph.")
        status(f"[crop] wrote {state['path']}")

    def do_add() -> None:
        if trace_mode.isChecked():
            do_trace()
            return
        picked = picked_segment()
        if picked is None:
            say("nothing picked - double-click a centreline point in the 3D view")
            return
        name = current_vessel()
        if not name:
            say("name the vessel first")
            return
        sid, _pid = picked
        members = state["vessels"].setdefault(name, set())
        # A toggle rather than an add: picking the same segment twice is far more
        # likely to be a correction than a request to add it again.
        members.discard(sid) if sid in members else members.add(sid)
        if not members:
            state["vessels"].pop(name, None)
        colour_for(name)
        state["plan"] = None
        refresh_lists()
        draw()
        say(f"{name}: {len(members)} segment(s)")

    def do_remove() -> None:
        picked = picked_segment()
        if picked is None:
            say("nothing picked")
            return
        sid, _pid = picked
        for name, members in list(state["vessels"].items()):
            members.discard(sid)
            if not members:
                state["vessels"].pop(name, None)
        state["plan"] = None
        refresh_lists()
        draw()

    def do_forget() -> None:
        name = current_vessel()
        if state["vessels"].pop(name, None) is None:
            say(f"no vessel named {name!r}")
            return
        state["plan"] = None
        refresh_lists()
        draw()

    def do_clear_vessels() -> None:
        state["vessels"] = {}
        state["plan"] = None
        refresh_lists()
        draw()

    def do_trace() -> None:
        """Two picks name a whole vessel: the first is the ostium, the second the far end.

        The pending start is dropped **before** the trace runs, not after it succeeds.
        A start left behind by a failed trace is the worst state this can be in: the
        next pick silently becomes the end of a trace the operator has stopped
        expecting, and names a vessel out of the wrong two segments.
        """
        picked = picked_segment()
        if picked is None:
            say("nothing picked - double-click a centreline point in the 3D view")
            return
        name = current_vessel()
        if not name:
            say("name the vessel first")
            return
        sid, _pid = picked
        if state["trace_start"] is None:
            state["trace_start"] = sid
            say(f"{name}: start at segment {sid}. Pick the far end and press Add pick "
                "again, or Cancel trace.")
            return
        start = state["trace_start"]
        state["trace_start"] = None
        traced = crop_mod.trace_path(
            state["graph"], start, sid, prefer_thick=prefer_thick.isChecked()
        )
        members = state["vessels"].setdefault(name, set())
        # A union, not a replacement: tracing the LAD and then its ostium stub is one
        # vessel in two goes, and a second trace must not throw the first one away.
        added = [s for s in traced.segments if s not in members]
        members.update(traced.segments)
        colour_for(name)
        state["plan"] = None
        refresh_lists()
        draw()
        say(f"{name}: traced {traced.describe()}; {len(added)} new, "
            f"{len(members)} segment(s) in all")

    def do_cancel_trace() -> None:
        if state["trace_start"] is None:
            say("no trace in progress")
            return
        state["trace_start"] = None
        say("trace cancelled - the start pick was dropped")

    def do_next_name() -> None:
        vessel_name.setCurrentIndex((vessel_name.currentIndex() + 1)
                                    % max(vessel_name.count(), 1))

    def do_omit() -> None:
        picked = picked_segment()
        if picked is None:
            say("nothing picked")
            return
        sid, _pid = picked
        if sid in state["drop_segments"]:
            state["drop_segments"].remove(sid)
        else:
            state["drop_segments"].append(sid)
        state["plan"] = None
        refresh_lists()
        say(f"{len(state['drop_segments'])} segment(s) marked by hand")

    def do_prune() -> None:
        picked = picked_segment()
        if picked is None:
            say("nothing picked")
            return
        sid, pid = picked
        graph = state["graph"]
        ids = graph.segment(sid)["point_ids"]
        # Clicking near the tip prunes the tip: walk away from the *nearer* node.
        # Same heuristic as `EditController.delete_picked_branch`, without the delete.
        from_node = (graph.segment(sid)["node1"] if ids.index(pid) > len(ids) / 2
                     else graph.segment(sid)["node2"])
        state["prune_at"].append((sid, from_node))
        state["plan"] = None
        refresh_lists()
        say(f"will prune everything past segment {sid}")

    def do_unmark() -> None:
        row = manual_list.currentRow()
        if row < 0:
            return
        if row < len(state["drop_segments"]):
            state["drop_segments"].pop(row)
        else:
            index = row - len(state["drop_segments"])
            if index < len(state["prune_at"]):
                state["prune_at"].pop(index)
        state["plan"] = None
        refresh_lists()

    def do_preview() -> None:
        graph = state["graph"]
        if graph is None:
            say("no dataset loaded")
            return
        state["plan"] = crop_mod.plan(
            graph, rule(), vessels=state["vessels"],
            drop_segments=state["drop_segments"], prune_at=state["prune_at"],
        )
        refresh_lists()
        draw()
        notes = "\n".join(f"note: {n}" for n in state["plan"].notes)
        say(state["plan"].summary() + ("\n" + notes if notes else ""))

    def do_queue() -> None:
        if runner is None or spec is None:
            say("no command queue in this session")
            return
        from . import cliform
        from .runner import Job

        if not sidecar_path.text().strip():
            say("give the sidecar a path first, so the run has the vessels")
            return
        do_save()
        argv = cliform.to_argv(spec, values())
        outputs = tuple(p for p in (out_path.text().strip(),) if p)
        runner.submit(Job(argv=argv, label=argv[0], outputs=outputs))
        say(f"queued: {' '.join(argv)}")

    open_button.clicked.connect(guarded(do_open))
    save_button.clicked.connect(guarded(do_save))
    add_button.clicked.connect(guarded(do_add))
    remove_button.clicked.connect(guarded(do_remove))
    forget_button.clicked.connect(guarded(do_forget))
    clear_vessels_button.clicked.connect(guarded(do_clear_vessels))
    omit_button.clicked.connect(guarded(do_omit))
    prune_button.clicked.connect(guarded(do_prune))
    unmark_button.clicked.connect(guarded(do_unmark))
    preview_button.clicked.connect(guarded(do_preview))
    cancel_trace_button.clicked.connect(guarded(do_cancel_trace))
    # Leaving the mode must not leave a start pick armed behind it.
    trace_mode.toggled.connect(guarded(lambda: state.update(trace_start=None)))
    queue_button.clicked.connect(guarded(do_queue))
    for widget in (strahler, min_radius, denominator):
        widget.valueChanged.connect(guarded(lambda: None))
    for widget in (use_strahler, use_radius, use_ratio, unattributed):
        widget.toggled.connect(guarded(lambda: None))
    vessel_name.currentTextChanged.connect(
        lambda _text: swatch.setStyleSheet(
            f"background: {colour_for(current_vessel())};" if current_vessel() else ""
        )
    )

    # ------------------------------------------------------------ refresh

    def refresh() -> None:
        """Re-read the app and update the widgets.

        The graph is rebuilt only when the session's has actually changed: the
        selection is a set of segment ids, and rebuilding on every refresh would
        throw away an operator's vessel picks every time another panel fired.
        """
        try:
            source = getattr(getattr(app, "session", None), "graph", None)
            if source is not state["source"]:
                rebuild_graph()
            loaded = state["graph"] is not None
            for widget in (add_button, remove_button, forget_button,
                           clear_vessels_button, omit_button, prune_button,
                           unmark_button, preview_button, save_button, queue_button,
                           trace_mode, prefer_thick, cancel_trace_button):
                widget.setEnabled(loaded)
            refresh_lists()
            if not loaded:
                say("no dataset loaded - pick a graph in the Data tab")
            elif state["plan"] is None:
                say(f"{len(state['graph'].segments):,} segment(s) loaded. "
                    "Name the main vessels, set a rule, then press Preview.")
            refresh_command()
        except Exception as exc:  # noqa: BLE001 - a panel must not take the window down
            say(f"crop panel: {type(exc).__name__}: {exc}")

    def bind_keys(plotter) -> None:
        """Bind the 3D shortcuts, if this session has a window to bind them in.

        `add_key_event` appends rather than replaces and cannot be undone, so every
        handler re-checks that there is still a graph to act on.
        """
        if plotter is None or not hasattr(plotter, "add_key_event"):
            return
        for key, action in zip(KEYS, (do_add, do_next_name, do_omit, do_prune)):
            plotter.clear_events_for_key(key)
            plotter.add_key_event(key, guarded(action))

    box.refresh = refresh
    box.load = load
    box.bind_keys = bind_keys
    box.state = state
    box.widgets = {
        "sidecar": sidecar_path, "out": out_path, "vessel_name": vessel_name,
        "vessels": vessel_list, "manual": manual_list, "summary": summary,
        "command": command_line, "strahler": strahler, "use_strahler": use_strahler,
        "min_radius": min_radius, "use_radius": use_radius,
        "denominator": denominator, "use_ratio": use_ratio,
        "unattributed": unattributed, "preview": preview_button,
        "trace_mode": trace_mode, "prefer_thick": prefer_thick,
    }
    refresh()
    return box
