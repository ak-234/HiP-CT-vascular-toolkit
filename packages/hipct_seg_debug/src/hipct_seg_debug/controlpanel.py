"""The control dock: Data, Commands and Workflows tabbed, with the Log beneath them.

The only module that knows the 3D window has a `QMainWindow` behind it. Everything
it assembles -- the four panels, the command runner, the argparse specs -- is
constructible without one, which is what keeps them testable.

One dock rather than four, because `plotter.app_window` already carries `layers` and
(in edit mode) `edit`; a fifth and sixth would leave no render area at all. Qt tabs
this one alongside those, so the whole right-hand side is three tabs deep at most.

**The log is a split pane, not a fourth tab.** It is the only panel that reports on
what the other three are *doing* -- a twelve-minute `skeletonise` streams into it, and
so does the note saying where a finished command wrote its file. As a tab it was hidden
exactly when it mattered: starting a chain from Workflows meant switching away from the
tab that shows its progress, and switching back to watch meant losing the controls. A
vertical splitter shows both at once, and drags shut when the render area is wanted.

It is unconditional, not behind a flag. Building it costs a couple of milliseconds
and imports nothing heavy -- `edit/__main__.build_parser` does its work in argparse
and every handler imports its own dependencies inside its body, so `coronary_sdf`
stays optional. A flag would create a second path, and the one nobody passes is the
one that rots.
"""

from __future__ import annotations

from . import (
    cliform,
    controls_cli,
    controls_crop,
    controls_data,
    controls_log,
    controls_reconnect,
    controls_reformat,
    controls_sections,
    controls_workflow,
)
from .runner import CommandRunner, install_tee


def build_control_panel(app):
    """Return ``(widget, runner)``. Docking is the caller's business.

    Kept separate from docking so the panel can be built and driven in a test
    without a `BackgroundPlotter`, exactly as `controls3d.build_layer_panel` is.
    """
    from qtpy.QtCore import Qt
    from qtpy.QtWidgets import QSplitter, QTabWidget

    from .edit.__main__ import build_parser as edit_parser

    specs = cliform.describe_parser(edit_parser())
    by_name = {spec.name: spec for spec in specs}

    tee, _err = install_tee()
    runner = CommandRunner(post=_qt_poster(), tee=tee)

    tabs = QTabWidget()
    log = controls_log.build_log_panel(runner)
    data = controls_data.build_data_panel(app)
    commands = controls_cli.build_commands_panel(app, runner, specs)
    flows = controls_workflow.build_workflows_panel(app, runner, by_name)
    reconnect = controls_reconnect.build_reconnect_panel(app)
    crop = controls_crop.build_crop_panel(app, runner, by_name.get("crop"))
    reformat = controls_reformat.build_reformat_panel(app)
    sections = controls_sections.build_sections_panel(app)

    tabs.addTab(data, "Data")
    tabs.addTab(commands, "Commands")
    tabs.addTab(flows, "Workflows")
    # Last, because it is the only tab that is useless until a review file exists,
    # and a run has to produce one before there is anything here to do.
    tabs.addTab(reconnect, "Reconnect")
    tabs.addTab(crop, "Crop")
    tabs.addTab(reformat, "Reformat")
    tabs.addTab(sections, "Sections")

    panel = QSplitter(Qt.Vertical)
    panel.addWidget(tabs)
    panel.addWidget(log)
    # Roughly two thirds controls, one third log. `setSizes` wants pixels and the dock
    # has no width yet, so the stretch factors are what actually decide the split; the
    # sizes are the hint for the first show.
    panel.setStretchFactor(0, 2)
    panel.setStretchFactor(1, 1)
    panel.setSizes([420, 210])
    # The log may be dragged shut when the render area is wanted; the controls may not,
    # because a collapsed tab bar leaves no handle to drag back.
    panel.setCollapsible(0, False)
    panel.setCollapsible(1, True)

    def on_queue() -> None:
        commands.refresh_queue()
        flows.refresh()

    def on_started(job) -> None:
        commands.refresh_queue()

    def on_done(result) -> None:
        # The chain first: it decides whether to advance before anything else looks
        # at the outcome.
        flows.on_job_done(result)
        commands.refresh_queue()
        _offer_outputs(app, result, log, data)
        _offer_review(result, log, reconnect, tabs)

    runner.on_queue = on_queue
    runner.on_started = on_started
    runner.on_done = on_done

    app.on_status = log.append
    app.on_dataset = lambda _app: (data.refresh(), commands.refresh(), flows.refresh(),
                                   reconnect.refresh(), crop.refresh(),
                                   reformat.refresh(), sections.refresh())

    panel.refresh = lambda: (data.refresh(), commands.refresh(), flows.refresh(),
                             reconnect.refresh(), crop.refresh(), reformat.refresh(),
                             sections.refresh())
    panel.runner = runner
    panel.panels = {"data": data, "commands": commands, "workflows": flows,
                    "reconnect": reconnect, "crop": crop, "reformat": reformat,
                    "sections": sections, "log": log}
    # The tab widget is no longer the panel itself, so anything that wants to switch
    # tabs -- or count them -- reaches it here rather than through the returned object.
    panel.tabs = tabs
    return panel, runner


def attach_control_dock(app, plotter):
    """Build the panel and dock it into the 3D window, if there is one to dock into.

    A plain ``pv.Plotter`` -- what the test harness substitutes -- has no
    ``app_window``, so this is skipped there, matching `Picker3D._dock_panel`.
    """
    if plotter is None or not hasattr(plotter, "app_window"):
        return None
    from qtpy.QtCore import Qt
    from qtpy.QtWidgets import QDockWidget

    panel, _runner = build_control_panel(app)
    dock = QDockWidget("control", plotter.app_window)
    dock.setWidget(panel)
    plotter.app_window.addDockWidget(Qt.RightDockWidgetArea, dock)
    # Crop, Reformat and Sections are the panels with 3D gestures of their own, and
    # they can only bind them once there is a plotter to bind them in.
    panel.panels["crop"].bind_keys(plotter)
    panel.panels["reformat"].bind_keys(plotter)
    panel.panels["sections"].bind_keys(plotter)
    app.control_panel = panel
    app.control_dock = dock
    return panel


def _offer_outputs(app, result, log, data) -> None:
    """Say what a finished command wrote, and offer to use it.

    Deliberately an offer. Reloading a graph the user is in the middle of inspecting
    -- and silently discarding any skeleton edits with it -- is not something to do
    on their behalf because a background job happened to finish.

    The offer is a *button* rather than a sentence: a fresh ``.am`` is handed to the
    Data tab's "Load result", which fills the graph field and reloads on one click.
    The log lines stay as they are, because they are the record of what a run produced
    and the button only ever holds the most recent one.

    A file already open in the viewer is only ever mentioned, never offered -- that is
    the case where reloading discards edits, so it stays a deliberate keystroke.
    """
    same, fresh = controls_cli.outputs_of(result, app)
    for path in same:
        log.append(f"  note: {path.name} is open in the viewer and was just rewritten. "
                   f"Press 'Reload graph' on the Data tab to pick up the new version.")
    graphs = [p for p in fresh if p.suffix.lower() == ".am"]
    for path in graphs:
        log.append(f"  note: {path} written. Press 'Load result' on the Data tab to "
                   f"look at it.")
    if graphs and hasattr(data, "offer_graph"):
        data.offer_graph(graphs[-1])


def _offer_review(result, log, reconnect, tabs) -> None:
    """Load a freshly written reconnection review into the Reconnect tab.

    Unlike a graph, this one is loaded rather than merely offered. The two cases
    differ in what a mistake costs: reloading a graph silently discards edits in
    progress, whereas a review document is a work list with no in-progress state
    behind it -- and a run that has just produced twenty candidates to adjudicate
    should not also require finding the file it wrote them to.

    Only ever loaded when it holds something to do. A run that resolved everything
    automatically writes an empty review, and switching tabs to show nothing is
    worse than staying put.
    """
    from pathlib import Path

    for path in _output_paths(result):
        if path.suffix.lower() != ".json":
            continue
        try:
            document = controls_reconnect.load_review(path)
        except Exception:  # noqa: BLE001 - some other JSON the run happened to write
            continue
        if not controls_reconnect.pending(document):
            log.append(f"  note: {Path(path).name} has no candidates needing review.")
            continue
        reconnect.load(str(path))
        tabs.setCurrentWidget(reconnect)
        return


def _output_paths(result):
    """Existing files a finished job reported writing.

    Reads ``outputs_written``, the same source ``controls_cli.outputs_of`` uses, so
    a job that wrote nothing offers nothing rather than this panel guessing at
    paths from the command line.
    """
    from pathlib import Path

    out = []
    for path in getattr(result, "outputs_written", ()) or ():
        candidate = Path(path)
        if candidate.exists():
            out.append(candidate)
    return out


def _qt_poster():
    """Hand a callback to the Qt thread, or run it here if Qt is absent.

    Shares `edit.worker.qt_poster` rather than reimplementing it, so the queued-signal
    subtlety documented there is solved in one place. Must be called on the GUI
    thread, which it is -- panels are built from `interactive`.
    """
    try:
        from .edit.worker import qt_poster

        return qt_poster()
    except Exception:  # noqa: BLE001 - no Qt binding: run inline, as the tests do
        return lambda fn: fn()
