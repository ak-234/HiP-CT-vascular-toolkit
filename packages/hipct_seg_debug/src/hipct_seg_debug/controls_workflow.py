"""The Workflows tab: the chains from CLI.md, and the setups that are not chains.

Two lists, kept visibly apart, because they are different things and pretending
otherwise would be the dishonest option. Workflows 7-11 are sequences of commands and
run start to finish on one click. Workflows 1-6 are "open the viewer and look" -- no
runner can do that for you, so those buttons configure the live session and tell you
what to do next.

The chain driver is `workflows.WorkflowRun`; this module is its display.
"""

from __future__ import annotations

import time
from pathlib import Path

from . import workflows as wf


def build_workflows_panel(app, runner, specs):
    """Return the Workflows widget."""
    from qtpy.QtWidgets import (
        QComboBox,
        QGroupBox,
        QHBoxLayout,
        QLabel,
        QLineEdit,
        QPushButton,
        QVBoxLayout,
        QWidget,
    )

    box = QWidget()
    lay = QVBoxLayout(box)
    lay.setContentsMargins(8, 8, 8, 8)

    state = {"run": None}

    # ---- the chains --------------------------------------------------

    chains = QGroupBox("Command chains (workflows 7-11)")
    chain_lay = QVBoxLayout(chains)

    chooser = QComboBox()
    for workflow in wf.WORKFLOWS:
        chooser.addItem(workflow.title, workflow.key)
    chain_lay.addWidget(chooser)

    blurb = QLabel("")
    blurb.setWordWrap(True)
    blurb.setStyleSheet("color: #808090;")
    chain_lay.addWidget(blurb)

    dir_row = QHBoxLayout()
    dir_row.addWidget(QLabel("run dir:"))
    run_dir = QLineEdit()
    run_dir.setToolTip("Where the intermediate files go. One directory per run, so a "
                       "second attempt cannot overwrite the first.")
    dir_row.addWidget(run_dir, 1)
    chain_lay.addLayout(dir_row)

    progress = QLabel("")
    progress.setWordWrap(True)
    chain_lay.addWidget(progress)

    buttons = QHBoxLayout()
    start = QPushButton("Run chain")
    cont = QPushButton("Continue")
    abort = QPushButton("Abort")
    for button in (start, cont, abort):
        buttons.addWidget(button)
    buttons.addStretch(1)
    chain_lay.addLayout(buttons)
    lay.addWidget(chains)

    # ---- the guided setups -------------------------------------------

    setups = QGroupBox("Guided setup (workflows 1-6)")
    setup_lay = QVBoxLayout(setups)
    note = QLabel("These are things you drive yourself. The buttons set the session up "
                  "for them and say what to do next.")
    note.setWordWrap(True)
    note.setStyleSheet("color: #808090;")
    setup_lay.addWidget(note)

    setup_buttons = {}
    for setup in wf.SETUPS:
        button = QPushButton(setup.title)
        button.setToolTip(setup.doc + ("\n\n" + setup.next_steps if setup.next_steps else ""))
        button.clicked.connect(lambda _c=False, s=setup: _do_setup(s))
        setup_lay.addWidget(button)
        setup_buttons[setup.key] = button
    lay.addWidget(setups)
    lay.addStretch(1)

    # -- chain actions ----------------------------------------------------

    def _selected():
        return wf.WORKFLOW_BY_KEY[chooser.currentData()]

    def _default_run_dir() -> str:
        from .main import _default_cache

        cache = getattr(app.session, "cache", None) or _default_cache()
        return str(Path(cache) / "runs" / time.strftime("%Y%m%d-%H%M%S"))

    def on_chooser(*_a) -> None:
        workflow = _selected()
        blurb.setText(workflow.doc)
        progress.setText(f"{len(workflow.steps)} steps")

    def do_start() -> None:
        if app.session is None:
            progress.setText("load a dataset first - the chain needs its paths")
            return
        target = Path(run_dir.text().strip() or _default_run_dir())
        try:
            target.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            progress.setText(f"cannot use {target}: {exc}")
            return
        run_dir.setText(str(target))

        context = wf.default_context(app.args, target)
        run = wf.WorkflowRun(_selected(), context, runner, specs, on_state=_on_state)
        state["run"] = run
        run.start()

    def do_continue() -> None:
        if state["run"] is not None:
            state["run"].continue_()

    def do_abort() -> None:
        runner.stop()
        if state["run"] is not None:
            state["run"].abort("aborted")

    def _on_state(run) -> None:
        progress.setText(run.describe())
        refresh()

    start.clicked.connect(do_start)
    cont.clicked.connect(do_continue)
    abort.clicked.connect(do_abort)
    chooser.currentIndexChanged.connect(on_chooser)

    # -- setup actions ----------------------------------------------------

    def _do_setup(setup) -> None:
        """Run a guided setup.

        Everything is inside the guard because this is a Qt slot, and PyQt5 calls
        ``qFatal`` on an exception that escapes one -- the process aborts rather
        than raising. ``enable_edit`` alone can fail for an entirely ordinary
        reason: a read-only session has no ``coronary_sdf``.
        """
        if "session" in setup.requires and app.session is None:
            app.status("load a dataset first")
            return

        try:
            if setup.action == "note":
                app.status(setup.next_steps or setup.doc)
            elif setup.action == "reload":
                for dest, value in setup.args.items():
                    setattr(app.args, dest, value)
                app.load(reset_camera=False)
            elif setup.action == "enable":
                if setup.key == "edit":
                    app.enable_edit()
                elif setup.key == "paint":
                    app.enable_paint()
            elif setup.action == "run":
                _run_setup(setup)
        except Exception as exc:  # noqa: BLE001 - must not abort the process
            app.status(f"{setup.title}: {type(exc).__name__}: {exc}")
            refresh()
            return

        if setup.next_steps:
            print(setup.next_steps)
        refresh()

    def _run_setup(setup) -> None:
        """The three that already have functions behind them.

        These act on the *live* session rather than on files, so they run here
        rather than through the command runner.
        """
        if setup.key == "validate":
            try:
                app.session.validate(strict=False)
                app.status("validation finished - see the Log tab")
            except Exception as exc:  # noqa: BLE001
                app.status(f"validation raised {type(exc).__name__}: {exc}")
        elif setup.key == "selftest":
            from .selftest import run_selftest

            ok = run_selftest(app.session)
            app.status("all self-tests passed" if ok else "some self-tests failed")
        elif setup.key == "audit":
            cands = app.session.find_candidates()
            # Straight into the 3D view, so 'n' and 'b' walk them immediately. Today
            # a session started with --no-candidates never gets them at all.
            app.set_candidates(cands)
            app.status(f"{len(cands)} candidate sites - walk them with 'n' and 'b'")

    # -- refresh ----------------------------------------------------------

    def refresh() -> None:
        run = state["run"]
        busy = run is not None and run.state == "running"
        start.setEnabled(not busy and app.loaded)
        cont.setEnabled(run is not None and run.state == "paused")
        abort.setEnabled(busy or (run is not None and run.state == "paused"))
        for setup in wf.SETUPS:
            needs = "session" in setup.requires
            setup_buttons[setup.key].setEnabled(app.loaded or not needs)
        if not run_dir.text():
            run_dir.setText(_default_run_dir())

    def on_job_done(result) -> None:
        if state["run"] is not None:
            state["run"].on_job_done(result)

    box.refresh = refresh
    box.on_job_done = on_job_done
    box.current_run = lambda: state["run"]
    on_chooser()
    refresh()
    return box
