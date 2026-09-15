"""Every panel must build, refresh and survive a dataset going away.

These need a QApplication but never a window: `QT_QPA_PLATFORM=offscreen` is set
before qtpy is imported. What they check is the shape a panel has to have -- it is
built from a duck-typed stub, so a panel that reached for something the real object
does not expose fails here rather than at the first click.

The load-bearing case is the last section: every panel has to refresh with
`app.session is None`, because that is the state after a failed load, and a panel
that raised there would take the window down at the exact moment you needed it to
tell you what went wrong.
"""

from __future__ import annotations

import os
import time
from types import SimpleNamespace

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("qtpy")

from hipct_seg_debug import cliform  # noqa: E402
from hipct_seg_debug.edit.__main__ import build_parser as edit_parser  # noqa: E402
from hipct_seg_debug.main import build_parser as viewer_parser  # noqa: E402
from hipct_seg_debug.runner import CommandRunner, Job, JobResult  # noqa: E402

SPECS = cliform.describe_parser(edit_parser())
BY_NAME = {s.name: s for s in SPECS}


@pytest.fixture(scope="module")
def qapp():
    from qtpy.QtWidgets import QApplication

    app = QApplication.instance() or QApplication([])
    yield app


class FakeApp:
    """The `ViewerApp` surface the panels actually touch."""

    def __init__(self, tmp_path, loaded=True):
        self.args = viewer_parser().parse_args([])
        self.args.graph = str(tmp_path / "graph.am")
        self.args.seg = str(tmp_path / "seg.am")
        self.args.raw = str(tmp_path)
        self.args.surface = str(tmp_path / "s.stl")
        self.args.edits = None
        self.session = SimpleNamespace(cache=tmp_path) if loaded else None
        self.picker = None
        self.messages = []
        self.on_status = None
        self.on_dataset = None
        self.loaded_calls = 0

    @property
    def loaded(self):
        return self.session is not None

    def status(self, message):
        self.messages.append(message)

    def describe(self):
        return "fake dataset" if self.loaded else "no dataset loaded"

    def can_swap(self):
        return []

    def load(self, args=None, *, reset_camera=True):
        self.loaded_calls += 1
        return True

    def reload_graph(self, path):
        # Mirrors `ViewerApp.reload_graph`: the skeleton field takes several paths,
        # so what lands in `args.graph` is always a list.
        from hipct_seg_debug.main import graph_paths

        self.args.graph = graph_paths(path)
        return True

    def set_candidates(self, cands):
        self.cands = cands

    def enable_edit(self):
        return True

    def enable_paint(self):
        return True


@pytest.fixture
def runner():
    return CommandRunner(dispatch=lambda argv: 0, post=lambda fn: fn())


# ------------------------------------------------------------------- forms


@pytest.mark.parametrize("name", [s.name for s in SPECS])
def test_every_command_form_builds(qapp, name):
    form = cliform.build_command_form(BY_NAME[name])
    assert form.spec.name == name
    assert set(form.widgets) == set(BY_NAME[name].dests)


def test_a_form_starts_at_the_parsers_defaults(qapp):
    form = cliform.build_command_form(BY_NAME["repair-radius"])
    values = form.values()
    assert values["factor"] == pytest.approx(0.6)
    assert values["source"] == "both"
    assert values["margin"] is None, "an unset spin box must read back as None"


def test_an_untouched_form_produces_the_bare_command(qapp):
    form = cliform.build_command_form(BY_NAME["gaps"])
    form.set_values({"graph": "g.am"})
    assert form.argv() == ["gaps", "g.am"]


def test_setting_a_value_reaches_argv(qapp):
    form = cliform.build_command_form(BY_NAME["skeletonise"])
    form.set_values({"stride": 4, "order": True, "out": "q.am"})
    argv = form.argv()
    assert "--stride=4" in argv and "--order" in argv and "--out=q.am" in argv


def test_show_all_flags_spells_everything_out(qapp):
    form = cliform.build_command_form(BY_NAME["repair-radius"])
    form.set_values({"graph": "g.am"})
    short = form.argv()
    # Find the checkbox by its text rather than by index, so adding a widget above
    # it does not silently make this test check something else.
    from qtpy.QtWidgets import QCheckBox

    show_all = [c for c in form.findChildren(QCheckBox)
                if "every flag" in c.text()][0]
    show_all.setChecked(True)
    assert len(form.argv()) > len(short)
    assert "--factor=0.6" in form.argv()


def test_the_command_line_box_tracks_the_form(qapp):
    form = cliform.build_command_form(BY_NAME["gaps"])
    form.set_values({"graph": "g.am", "out": "o.am"})
    assert "--out=o.am" in form.command_line_box.text()


def test_a_required_field_is_reported_until_filled(qapp):
    form = cliform.build_command_form(BY_NAME["mask-export"])
    assert any("--out" in e for e in form.errors())
    form.set_values({"out": "x.am"})
    assert not any("--out" in e for e in form.errors())


def test_a_browse_callback_writes_into_the_field(qapp):
    form = cliform.build_command_form(
        BY_NAME["gaps"], browse=lambda field: f"/chosen/{field.dest}.am")
    from qtpy.QtWidgets import QPushButton

    button = [b for b in form.findChildren(QPushButton) if b.text() == "..."][0]
    button.click()
    assert "/chosen/" in str(form.values()["graph"])


# ------------------------------------------------------------------ panels


def test_the_log_panel_appends_and_replaces(qapp, runner):
    from hipct_seg_debug.controls_log import build_log_panel

    log = build_log_panel(runner)
    log.append("first", False)
    log.append("  1/10", True)
    log.append("  2/10", True)
    text = log.view.toPlainText()
    assert "first" in text
    assert "1/10" not in text, "a transient line must be overwritten, not appended"
    assert "2/10" in text


def test_the_runner_writes_into_the_log(qapp, runner):
    from hipct_seg_debug.controls_log import build_log_panel

    log = build_log_panel(runner)
    runner.submit(Job(argv=["report", "g.am"], mode="inproc"))
    for _ in range(200):
        if not runner.busy:
            break
        qapp.processEvents()
    assert "report" in log.view.toPlainText()


def test_the_data_panel_shows_the_loaded_paths(qapp, tmp_path):
    from hipct_seg_debug.controls_data import build_data_panel

    app = FakeApp(tmp_path)
    panel = build_data_panel(app)
    assert panel.values()["graph"] == app.args.graph
    assert set(panel.fields) == {"raw", "graph", "seg", "surface", "edits"}


def test_the_data_panel_refuses_a_path_that_does_not_exist(qapp, tmp_path):
    from hipct_seg_debug.controls_data import build_data_panel

    app = FakeApp(tmp_path)
    panel = build_data_panel(app)
    from qtpy.QtWidgets import QPushButton

    load = [b for b in panel.findChildren(QPushButton) if b.text() == "Load all"][0]
    load.click()
    assert app.loaded_calls == 0, "it must not try to load a nonexistent graph"


def test_the_data_panel_loads_when_the_paths_are_real(qapp, tmp_path):
    from hipct_seg_debug.controls_data import build_data_panel

    app = FakeApp(tmp_path)
    for dest in ("graph", "seg"):
        __import__("pathlib").Path(getattr(app.args, dest)).write_text("x")
    app.args.surface = ""
    panel = build_data_panel(app)
    panel.fields["surface"].setCurrentText("")
    from qtpy.QtWidgets import QPushButton

    # The voxel size is never inferred, so a load is refused until it is confirmed.
    panel.voxel.setValue(32.04)
    panel.voxel_confirm.click()
    [b for b in panel.findChildren(QPushButton) if b.text() == "Load all"][0].click()
    assert app.loaded_calls == 1
    assert app.args.voxel_um == 32.04


def test_the_data_panel_refuses_to_load_on_an_unconfirmed_voxel_size(qapp, tmp_path):
    """The suggestion is offered; taking it is a decision the operator makes.

    A folder name or a bounding box is wrong by a couple of percent in a way nothing
    downstream can see, and that factor lands on every radius and length in the
    session -- so the number is confirmed, not defaulted.
    """
    from hipct_seg_debug.controls_data import build_data_panel
    from qtpy.QtWidgets import QPushButton

    app = FakeApp(tmp_path)
    for dest in ("graph", "seg"):
        __import__("pathlib").Path(getattr(app.args, dest)).write_text("x")
    app.args.surface = ""
    app.args.voxel_um = None
    panel = build_data_panel(app)
    panel.voxel.setValue(32.99)  # editing it un-confirms, as typing a value must

    load = [b for b in panel.findChildren(QPushButton) if b.text() == "Load all"][0]
    assert not load.isEnabled()
    load.click()
    assert app.loaded_calls == 0
    assert not panel.voxel_confirmed()

    panel.voxel_confirm.click()
    assert load.isEnabled()
    load.click()
    assert app.loaded_calls == 1


def _load_result(panel):
    """The "Load result" button, found by object name -- its label is not stable."""
    from qtpy.QtWidgets import QPushButton

    return panel.findChild(QPushButton, "load_result")


def test_load_result_is_disabled_until_a_command_writes_a_graph(qapp, tmp_path):
    from hipct_seg_debug.controls_data import build_data_panel

    panel = build_data_panel(FakeApp(tmp_path))
    assert not _load_result(panel).isEnabled()


def test_offering_a_graph_enables_the_button_and_names_it(qapp, tmp_path):
    from hipct_seg_debug.controls_data import build_data_panel

    app = FakeApp(tmp_path)
    panel = build_data_panel(app)
    written = tmp_path / "refined.am"
    written.write_text("x")

    panel.offer_graph(written)

    button = _load_result(panel)
    assert button.isEnabled()
    assert "refined.am" in button.text(), "the button says which file it will open"


def test_load_result_reloads_the_offered_graph(qapp, tmp_path):
    from hipct_seg_debug.controls_data import build_data_panel

    app = FakeApp(tmp_path)
    panel = build_data_panel(app)
    written = tmp_path / "refined.am"
    written.write_text("x")

    panel.offer_graph(written)
    _load_result(panel).click()

    assert app.args.graph == [str(written)], "reload_graph was called with the result"
    assert panel.fields["graph"].currentText() == str(written)


def test_load_result_survives_the_file_disappearing(qapp, tmp_path):
    from hipct_seg_debug.controls_data import build_data_panel

    app = FakeApp(tmp_path)
    panel = build_data_panel(app)
    written = tmp_path / "gone.am"
    written.write_text("x")
    panel.offer_graph(written)
    written.unlink()

    _load_result(panel).click()

    assert app.args.graph != str(written)


def test_offer_outputs_offers_a_fresh_graph_but_not_the_open_one(qapp, tmp_path):
    """Reloading the graph under inspection would discard edits, so it stays manual."""
    from pathlib import Path

    from hipct_seg_debug import controlpanel
    from hipct_seg_debug.controls_data import build_data_panel

    app = FakeApp(tmp_path)
    panel = build_data_panel(app)
    fresh = tmp_path / "new.am"
    fresh.write_text("x")

    lines: list[str] = []
    log = SimpleNamespace(append=lines.append)
    result = SimpleNamespace(outputs_written=(Path(app.args.graph), fresh))

    controlpanel._offer_outputs(app, result, log, panel)

    assert _load_result(panel).isEnabled()
    assert any("Reload graph" in line for line in lines), "the open one is only mentioned"
    assert any("Load result" in line for line in lines)


def test_the_commands_panel_builds_one_form_per_command(qapp, tmp_path, runner):
    from hipct_seg_debug.controls_cli import build_commands_panel

    panel = build_commands_panel(FakeApp(tmp_path), runner, SPECS)
    assert set(panel.forms) == {s.name for s in SPECS}
    # Counted from the parser, not written down: adding a subcommand should make
    # the panel grow, not make this test fail.
    assert panel.chooser.count() == len(SPECS)


def test_use_loaded_paths_fills_the_form(qapp, tmp_path, runner):
    from hipct_seg_debug.controls_cli import build_commands_panel

    app = FakeApp(tmp_path)
    panel = build_commands_panel(app, runner, SPECS)
    panel.chooser.setCurrentText("gaps")
    from qtpy.QtWidgets import QPushButton

    [b for b in panel.findChildren(QPushButton)
     if b.text() == "Use loaded paths"][0].click()
    # A list: every command that processes a skeleton takes several now.
    from hipct_seg_debug.main import graph_paths

    assert panel.forms["gaps"].values()["graph"] == graph_paths(app.args.graph)


def test_use_loaded_paths_does_not_hand_pick_roots_the_segmentation(qapp, tmp_path,
                                                                    runner):
    """`pick-roots` reads the mask only to number the trees, and the roots it
    records are identical without it -- they are recorded by world coordinate. So
    filling it in would turn a three-second command into a full decode plus a
    connected-component labelling, tens of minutes at stride 1, for nothing the
    operator asked for."""
    from hipct_seg_debug.controls_cli import build_commands_panel

    app = FakeApp(tmp_path)
    panel = build_commands_panel(app, runner, SPECS)
    panel.chooser.setCurrentText("pick-roots")
    from qtpy.QtWidgets import QPushButton

    [b for b in panel.findChildren(QPushButton)
     if b.text() == "Use loaded paths"][0].click()

    values = panel.forms["pick-roots"].values()
    assert values["graph"], "the skeleton is still filled in"
    assert not values["seg"], "the segmentation is not"
    # And a command that genuinely needs the mask still gets it.
    panel.chooser.setCurrentText("radius-perimeter")
    [b for b in panel.findChildren(QPushButton)
     if b.text() == "Use loaded paths"][0].click()
    assert panel.forms["radius-perimeter"].values()["seg"] == str(app.args.seg)


def test_running_submits_a_job_with_its_outputs(qapp, tmp_path, runner):
    from hipct_seg_debug.controls_cli import build_commands_panel

    graph = tmp_path / "g.am"
    graph.write_text("x")
    submitted = []
    runner.submit = submitted.append
    panel = build_commands_panel(FakeApp(tmp_path), runner, SPECS)
    panel.chooser.setCurrentText("gaps")
    panel.forms["gaps"].set_values({"graph": str(graph), "out": "o.am"})
    from qtpy.QtWidgets import QPushButton

    [b for b in panel.findChildren(QPushButton) if b.text() == "Run"][0].click()
    assert submitted and submitted[0].outputs == ("o.am",)


def test_run_is_refused_while_an_input_does_not_exist(qapp, tmp_path, runner):
    """Better to grey the button out than to let argparse fail in a subprocess."""
    from hipct_seg_debug.controls_cli import build_commands_panel
    from qtpy.QtWidgets import QPushButton

    panel = build_commands_panel(FakeApp(tmp_path), runner, SPECS)
    panel.chooser.setCurrentText("gaps")
    panel.forms["gaps"].set_values({"graph": str(tmp_path / "missing.am")})
    run = [b for b in panel.findChildren(QPushButton) if b.text() == "Run"][0]
    assert not run.isEnabled()


def test_the_workflows_panel_lists_the_chains_and_setups(qapp, tmp_path, runner):
    from hipct_seg_debug import workflows as wf
    from hipct_seg_debug.controls_workflow import build_workflows_panel

    panel = build_workflows_panel(FakeApp(tmp_path), runner, BY_NAME)
    from qtpy.QtWidgets import QComboBox, QPushButton

    chooser = panel.findChildren(QComboBox)[0]
    assert chooser.count() == len(wf.WORKFLOWS)
    titles = {b.text() for b in panel.findChildren(QPushButton)}
    assert all(s.title in titles for s in wf.SETUPS)


def test_the_whole_dock_assembles(qapp, tmp_path):
    from hipct_seg_debug.controlpanel import build_control_panel

    panel, runner = build_control_panel(FakeApp(tmp_path))
    # Reconnect, Crop and Reformat are last: all three are useless until something else
    # has happened first -- a run that wrote a review file, an operator naming a main
    # vessel, or a segment picked in the 3D window.
    assert [panel.tabs.tabText(i) for i in range(panel.tabs.count())] == [
        "Data", "Commands", "Workflows", "Reconnect", "Crop", "Reformat", "Sections"]
    assert runner is panel.runner
    assert set(panel.panels) == {"data", "commands", "workflows", "reconnect",
                                 "crop", "reformat", "sections", "log"}


def test_the_sections_panel_refuses_to_cut_without_a_dataset(qapp, tmp_path):
    """The empty state is reachable -- a failed load leaves it -- and must not raise."""
    from hipct_seg_debug.controls_sections import build_sections_panel

    app = FakeApp(tmp_path, loaded=False)
    panel = build_sections_panel(app)
    panel.widgets["cut"].click()
    assert not panel.widgets["cut"].isEnabled()
    assert "no dataset" in panel.widgets["summary"].text()


def test_the_sections_panel_cuts_through_the_picker(qapp, tmp_path):
    """A cut with no Qt thread runs inline, and its geometry reaches the 3D window."""
    import numpy as np

    from hipct_seg_debug.controls_sections import build_sections_panel
    from tests.conftest_geometry import SPACING, axis_graph, cylinder, make_frame

    shape = (30, 30, 60)
    frame = make_frame(shape)
    graph = axis_graph(frame, 6, 54, 5 * SPACING, cy=15, cz=15)
    drawn = {}

    app = FakeApp(tmp_path)
    app.session = SimpleNamespace(
        cache=tmp_path, graph=graph.to_spatial_graph(), frame=frame,
        labels=cylinder(shape, 5, 4, 56),
    )
    app.picker = SimpleNamespace(
        _point_i=-1,
        show_cross_sections=lambda **kw: drawn.update(kw),
        clear_cross_sections=lambda: drawn.clear(),
    )

    panel = build_sections_panel(app)
    panel.refresh()
    panel.widgets["stride"].setValue(8)
    panel.widgets["cut"].click()
    # The cut runs on a worker thread and posts its result back, so the loop has to
    # keep turning until it arrives -- a fixed number of `processEvents` calls returns
    # in microseconds and would race the worker every time.
    deadline = time.time() + 60
    while panel.state["busy"] and time.time() < deadline:
        qapp.processEvents()
    assert not panel.state["busy"], "the cut never finished"

    assert drawn.get("corners"), "the measured windows must reach the viewer"
    assert all(np.asarray(q).shape == (4, 3) for q in drawn["corners"])
    # The only test that drives the real `controls_sections.draw`, so it is the one
    # seam where forgetting to forward a drawable array would be silent.
    assert "refused" in drawn
    assert "sections over" in panel.widgets["summary"].text()


def _sections_panel_on_a_chain(tmp_path, n_edges=5):
    """A Sections panel over a straight chain of segments, for the two-pick trace.

    The trace is about naming a *vessel* rather than a segment, so the fixture has to
    be something a path can run along; `axis_graph` is one segment and would prove
    nothing.
    """
    from hipct_seg_debug.controls_sections import build_sections_panel
    from tests.conftest_geometry import SPACING, graph_from

    nodes = [(float(i) * 400.0, 0.0, 0.0) for i in range(n_edges + 1)]
    edges = [(i, i + 1, 6, 5 * SPACING) for i in range(n_edges)]
    graph = graph_from(nodes, edges)

    app = FakeApp(tmp_path)
    app.session = SimpleNamespace(cache=tmp_path, graph=graph.to_spatial_graph(),
                                  frame=None, labels=None)
    order = graph.point_order()
    owner = graph.segment_of_point()

    def pick_segment(sid):
        app.picker._point_i = next(i for i, pid in enumerate(order) if owner[pid] == sid)

    app.picker = SimpleNamespace(_point_i=-1, pick_segment=pick_segment,
                                 show_cross_sections=lambda **kw: None,
                                 clear_cross_sections=lambda: None)
    panel = build_sections_panel(app)
    panel.refresh()
    return panel, app


def test_the_sections_trace_selects_every_segment_between_two_picks(qapp, tmp_path):
    panel, app = _sections_panel_on_a_chain(tmp_path)
    panel.widgets["trace_mode"].setChecked(True)

    app.picker.pick_segment(0)
    panel.widgets["add"].click()
    assert panel.state["trace_start"] == 0, "the first pick only arms the trace"
    assert panel.state["selected"] == [], "and selects nothing on its own"

    app.picker.pick_segment(4)
    panel.widgets["add"].click()
    assert sorted(panel.state["selected"]) == [0, 1, 2, 3, 4]
    assert panel.state["trace_start"] is None
    assert "traced" in panel.widgets["summary"].text()


def test_a_second_sections_trace_adds_rather_than_toggling_the_overlap_off(qapp, tmp_path):
    """Union, as in Crop. A trace that removed its overlap with an earlier one could
    not be reasoned about: the same pick would mean 'add' or 'remove' depending on
    history."""
    panel, app = _sections_panel_on_a_chain(tmp_path)
    panel.widgets["trace_mode"].setChecked(True)
    for a, b in ((0, 3), (2, 4)):
        app.picker.pick_segment(a)
        panel.widgets["add"].click()
        app.picker.pick_segment(b)
        panel.widgets["add"].click()

    assert sorted(panel.state["selected"]) == [0, 1, 2, 3, 4]


def test_cancelling_a_sections_trace_drops_only_the_start_pick(qapp, tmp_path):
    panel, app = _sections_panel_on_a_chain(tmp_path)
    panel.widgets["trace_mode"].setChecked(True)
    app.picker.pick_segment(1)
    panel.widgets["add"].click()
    panel.widgets["cancel_trace"].click()

    assert panel.state["trace_start"] is None
    assert panel.state["selected"] == []
    assert "cancelled" in panel.widgets["summary"].text()

    # and with the mode off, Add pick is the plain toggle it always was
    panel.widgets["trace_mode"].setChecked(False)
    app.picker.pick_segment(2)
    panel.widgets["add"].click()
    assert panel.state["selected"] == [2]
    panel.widgets["add"].click()
    assert panel.state["selected"] == []


def test_the_sections_panel_defaults_reproduce_the_pass(qapp, tmp_path):
    """The panel half of the parity contract, and what would have caught the report.

    `survey`'s defaults are pinned against `measure_radii`'s in
    `test_section_frames.py`. This pins that the panel actually *uses* them: it must
    pass `grow_radii` through (so the box is live) and must NOT pass the two
    stability thresholds at all (so `survey`'s pinned defaults are what run, rather
    than a literal restated here that could drift on its own).
    """
    import hipct_seg_debug.controls_sections as cs_panel
    from hipct_seg_debug.controls_sections import build_sections_panel
    from hipct_seg_debug.edit import radius_perimeter as rp
    from tests.conftest_geometry import SPACING, axis_graph, cylinder, make_frame

    shape = (30, 30, 60)
    frame = make_frame(shape)
    graph = axis_graph(frame, 6, 54, 5 * SPACING, cy=15, cz=15)
    seen = []
    real = cs_panel.sf.survey

    def spy(*a, **kw):
        seen.append(kw)
        return real(*a, **kw)

    app = FakeApp(tmp_path)
    app.session = SimpleNamespace(
        cache=tmp_path, graph=graph.to_spatial_graph(), frame=frame,
        labels=cylinder(shape, 5, 4, 56),
    )
    app.picker = SimpleNamespace(
        _point_i=-1, show_cross_sections=lambda **kw: None,
        clear_cross_sections=lambda: None,
    )

    cs_panel.sf.survey = spy
    try:
        panel = build_sections_panel(app)
        panel.refresh()
        panel.widgets["stride"].setValue(8)

        def cut_and_wait():
            panel.widgets["cut"].click()
            deadline = time.time() + 60
            while panel.state["busy"] and time.time() < deadline:
                qapp.processEvents()
            assert not panel.state["busy"], "the cut never finished"

        cut_and_wait()
        assert seen[-1]["grow_radii"] is None
        assert seen[-1]["grow_radii"] == rp.measure_radii.__kwdefaults__["grow_radii"]
        assert seen[-1]["rival_check"] is True
        assert "stability_variation" not in seen[-1]
        assert "stability_centroid_radii" not in seen[-1]

        panel.widgets["grow_ceiling"].setValue(4)
        panel.widgets["merge_check"].setChecked(False)
        cut_and_wait()
        assert seen[-1]["grow_radii"] == 4.0
        assert seen[-1]["rival_check"] is False
    finally:
        cs_panel.sf.survey = real


def test_the_log_sits_below_the_tabs_not_inside_them(qapp, tmp_path):
    """It reports on what the other panels are doing, so it has to stay visible.

    As a fourth tab, starting a chain from Workflows meant switching away from the
    only view of its progress.
    """
    from qtpy.QtWidgets import QSplitter

    from hipct_seg_debug.controlpanel import build_control_panel

    panel, _runner = build_control_panel(FakeApp(tmp_path))

    assert isinstance(panel, QSplitter)
    assert panel.count() == 2
    assert panel.widget(0) is panel.tabs
    assert panel.widget(1) is panel.panels["log"]
    assert not panel.isCollapsible(0), "the controls must keep a handle to drag back"
    assert panel.isCollapsible(1), "the log may be dragged shut"


def test_the_log_still_receives_runner_output_from_its_new_home(qapp, tmp_path):
    from hipct_seg_debug.controlpanel import build_control_panel

    panel, runner = build_control_panel(FakeApp(tmp_path))
    runner.submit(Job(argv=["report", "g.am"], mode="inproc"))
    for _ in range(200):
        if not runner.busy:
            break
        qapp.processEvents()
    assert "report" in panel.panels["log"].view.toPlainText()


def test_attaching_without_a_window_is_a_no_op(qapp, tmp_path):
    from hipct_seg_debug.controlpanel import attach_control_dock

    assert attach_control_dock(FakeApp(tmp_path), object()) is None
    assert attach_control_dock(FakeApp(tmp_path), None) is None


def test_it_docks_into_a_real_main_window(qapp, tmp_path):
    """`app_window` is a plain QMainWindow, which is testable without VTK.

    A `BackgroundPlotter` needs an OpenGL context and so cannot be built headlessly,
    but the docking itself is ordinary Qt and there is no reason to leave it unrun.
    """
    from qtpy.QtWidgets import QDockWidget, QMainWindow

    from hipct_seg_debug.controlpanel import attach_control_dock

    plotter = SimpleNamespace(app_window=QMainWindow())
    panel = attach_control_dock(FakeApp(tmp_path), plotter)
    docks = plotter.app_window.findChildren(QDockWidget)
    assert [d.windowTitle() for d in docks] == ["control"]
    assert docks[0].widget() is panel


# --------------------------------------------------- the no-dataset state


def test_every_panel_refreshes_with_nothing_loaded(qapp, tmp_path):
    """The state after a failed load. A panel that raises here hides the reason."""
    from hipct_seg_debug.controlpanel import build_control_panel

    panel, _runner = build_control_panel(FakeApp(tmp_path, loaded=False))
    panel.refresh()


def test_the_chain_button_is_disabled_with_nothing_loaded(qapp, tmp_path, runner):
    from hipct_seg_debug.controls_workflow import build_workflows_panel

    panel = build_workflows_panel(FakeApp(tmp_path, loaded=False), runner, BY_NAME)
    from qtpy.QtWidgets import QPushButton

    start = [b for b in panel.findChildren(QPushButton) if b.text() == "Run chain"][0]
    assert not start.isEnabled()


def test_a_finished_job_reports_a_rewritten_open_file(qapp, tmp_path):
    """It must offer to reload, and must not reload on its own."""
    from hipct_seg_debug.controls_cli import outputs_of

    app = FakeApp(tmp_path)
    graph = __import__("pathlib").Path(app.args.graph)
    graph.write_text("x")
    result = JobResult(job=Job(argv=["gaps"]), outputs_written=(graph,))
    same, fresh = outputs_of(result, app)
    assert [p.name for p in same] == [graph.name] and fresh == []


def test_a_finished_job_reports_a_new_file_separately(qapp, tmp_path):
    from hipct_seg_debug.controls_cli import outputs_of

    app = FakeApp(tmp_path)
    other = tmp_path / "step1.am"
    other.write_text("x")
    result = JobResult(job=Job(argv=["gaps"]), outputs_written=(other,))
    same, fresh = outputs_of(result, app)
    assert same == [] and [p.name for p in fresh] == ["step1.am"]


# ------------------------------------------------------------- recent paths


def test_recent_paths_round_trip(tmp_path):
    from hipct_seg_debug.controls_data import load_recent, remember, save_recent

    save_recent(tmp_path, remember({}, "graph", "D:/data/a.am"))
    assert load_recent(tmp_path)["graph"] == ["D:/data/a.am"]


def test_the_newest_path_comes_first_and_does_not_duplicate():
    from hipct_seg_debug.controls_data import remember

    recent = remember(remember(remember({}, "graph", "a"), "graph", "b"), "graph", "a")
    assert recent["graph"] == ["a", "b"]


def test_a_corrupt_recent_file_is_ignored(tmp_path):
    from hipct_seg_debug.controls_data import load_recent, recent_path

    recent_path(tmp_path).write_text("{not json")
    assert load_recent(tmp_path) == {}


def test_saving_to_an_unwritable_place_does_not_raise():
    from hipct_seg_debug.controls_data import save_recent

    save_recent("/nonexistent-dir-xyz", {"graph": ["a"]})


# ------------------------------------------------------ the layer panel

class FakePicker:
    """The `Picker3D` surface `controls3d` touches."""

    def __init__(self, stride=1, stale=False):
        self.seg_stride = stride
        self.labels = object()
        self.strides = []
        self.rebuilds = 0
        self._stale = stale
        self.on_layers_changed = None
        self.color = "radius"
        self.modes = [("radius", "radius (um)"), ("strahler", "Strahler order")]
        self.legend_on = True
        self.figures = []
        self.figure_error = None

    def layer_available(self, key):
        return True

    def layer_visible(self, key):
        return False

    def layer_opacity(self, key):
        return 0.5

    def set_layer_opacity(self, key, value):
        pass

    def set_layer_visible(self, key, on):
        pass

    def set_seg_stride(self, stride):
        self.strides.append(stride)
        self.seg_stride = stride

    def rebuild_segmentation_all(self):
        self.rebuilds += 1

    def segmentation_all_stale(self):
        return self._stale

    def color_modes(self):
        return list(self.modes)

    def color_by(self):
        return self.color

    def set_color_by(self, mode):
        self.color = mode

    def legend_available(self):
        return True

    def legend_visible(self):
        return self.legend_on

    def set_legend_visible(self, on):
        self.legend_on = bool(on)

    def save_figure(self, path):
        if self.figure_error is not None:
            raise self.figure_error
        self.figures.append(path)
        return path


def _layer_panel(picker):
    from hipct_seg_debug.controls3d import build_layer_panel

    return build_layer_panel(picker)


def test_the_legend_checkbox_reaches_the_picker(qapp):
    """The legend is chrome rather than a layer, so it has its own accessors."""
    from qtpy.QtWidgets import QCheckBox

    picker = FakePicker()
    panel = _layer_panel(picker)
    box = [c for c in panel.findChildren(QCheckBox) if c.text() == "legend box"][0]
    assert box.isChecked(), "it starts on, because the picker says it is on"

    box.setChecked(False)
    assert picker.legend_on is False


def test_refresh_writes_the_legend_state_without_calling_back(qapp):
    from qtpy.QtWidgets import QCheckBox

    picker = FakePicker()
    panel = _layer_panel(picker)
    box = [c for c in panel.findChildren(QCheckBox) if c.text() == "legend box"][0]

    picker.legend_on = False
    panel.refresh()
    assert not box.isChecked()
    assert picker.legend_on is False, "refresh must not push the state back"


def test_the_layer_panel_shows_the_current_stride(qapp):
    panel = _layer_panel(FakePicker(stride=4))
    from qtpy.QtWidgets import QSpinBox

    assert panel.findChildren(QSpinBox)[0].value() == 4


def test_changing_the_stride_reaches_the_picker(qapp):
    picker = FakePicker(stride=1)
    panel = _layer_panel(picker)
    from qtpy.QtWidgets import QSpinBox

    panel.findChildren(QSpinBox)[0].setValue(4)
    assert picker.strides == [4]


def test_refresh_writes_the_stride_without_calling_back(qapp):
    """The blockSignals contract: a programmatic write must not look like a user edit."""
    picker = FakePicker(stride=1)
    panel = _layer_panel(picker)
    picker.seg_stride = 8
    panel.refresh()
    from qtpy.QtWidgets import QSpinBox

    assert panel.findChildren(QSpinBox)[0].value() == 8
    assert picker.strides == [], "refresh must not push the value back into the picker"


def test_the_rebuild_button_rebuilds(qapp):
    picker = FakePicker()
    panel = _layer_panel(picker)
    from qtpy.QtWidgets import QPushButton

    [b for b in panel.findChildren(QPushButton) if "rebuild" in b.text()][0].click()
    assert picker.rebuilds == 1


def test_the_rebuild_button_is_marked_when_edits_are_pending(qapp):
    from qtpy.QtWidgets import QPushButton

    plain = _layer_panel(FakePicker(stale=False))
    marked = _layer_panel(FakePicker(stale=True))
    def text(p):
        return [b.text() for b in p.findChildren(QPushButton) if "rebuild" in b.text()][0]
    assert text(plain) == "rebuild"
    assert text(marked).endswith("*")


def test_the_rebuild_button_is_re_enabled_afterwards(qapp):
    """It is disabled during the blocking contour; a stuck button would be worse."""
    picker = FakePicker()
    panel = _layer_panel(picker)
    from qtpy.QtWidgets import QPushButton

    button = [b for b in panel.findChildren(QPushButton) if "rebuild" in b.text()][0]
    button.click()
    assert button.isEnabled()


def test_a_rebuild_that_raises_is_swallowed_not_propagated(qapp, capsys):
    """An exception escaping a Qt slot makes PyQt5 call qFatal and abort the process.

    Not a hypothetical: this test crashed the whole run with exit 127 before the
    handler caught. So a failed rebuild has to be reported, never re-raised.
    """
    class Broken(FakePicker):
        def rebuild_segmentation_all(self):
            raise RuntimeError("vtk said no")

    picker = Broken()
    panel = _layer_panel(picker)
    from qtpy.QtWidgets import QPushButton

    button = [b for b in panel.findChildren(QPushButton) if "rebuild" in b.text()][0]
    button.click()
    assert button.isEnabled(), "a failed rebuild must not leave the button stuck"
    assert "vtk said no" in capsys.readouterr().out


def _colour_combo(panel):
    from qtpy.QtWidgets import QComboBox

    return panel.findChildren(QComboBox)[0]


def test_choosing_a_colour_mode_reaches_the_picker(qapp):
    picker = FakePicker()
    panel = _layer_panel(picker)  # kept alive: Qt deletes the children with it
    combo = _colour_combo(panel)
    keys = [combo.itemData(i) for i in range(combo.count())]

    combo.setCurrentIndex(keys.index("strahler"))
    assert picker.color == "strahler"


def test_refresh_writes_the_colour_mode_without_calling_back(qapp):
    picker = FakePicker()
    panel = _layer_panel(picker)
    picker.color = "strahler"
    panel.refresh()

    combo = _colour_combo(panel)
    assert combo.itemData(combo.currentIndex()) == "strahler"
    assert picker.color == "strahler", "refresh must not push the mode back in"


def test_a_graph_with_one_colour_mode_greys_the_combo_out(qapp):
    """A live box implies there is something else to pick, and there is not."""
    picker = FakePicker()
    picker.modes = [("radius", "radius (um)")]
    panel = _layer_panel(picker)  # kept alive: Qt deletes the children with it
    combo = _colour_combo(panel)

    assert combo.count() == 1
    assert not combo.isEnabled()


def test_the_data_panel_works_from_a_bare_launch(qapp, tmp_path):
    """The state `python -m hipct_seg_debug` with no arguments leaves behind.

    Not the same as `loaded=False` elsewhere in this file: there the paths are still
    on `args` and only the session is gone. Here nothing has ever been named, which is
    what the Data tab has to be usable from -- it is the only way in.
    """
    from hipct_seg_debug.controls_data import build_data_panel

    app = FakeApp(tmp_path, loaded=False)
    for dest in ("raw", "graph", "seg", "surface", "edits"):
        setattr(app.args, dest, None)

    panel = build_data_panel(app)
    assert panel.values() == {"raw": "", "graph": "", "seg": "", "surface": "",
                              "edits": ""}

    # Pressing Load with nothing filled in must complain, not raise or half-load.
    from qtpy.QtWidgets import QPushButton

    load = [b for b in panel.findChildren(QPushButton) if b.text() == "Load all"][0]
    load.click()
    assert app.loaded_calls == 0

    # Filling the three required inputs in makes it go. `--surface` stays empty, which
    # is a warning rather than an error, so this is the minimum a bare launch needs.
    import pathlib as _pathlib

    for dest in ("graph", "seg"):
        path = _pathlib.Path(tmp_path) / f"{dest}.am"
        path.write_text("x")
        panel.fields[dest].setCurrentText(str(path))
    panel.fields["raw"].setCurrentText(str(tmp_path))
    panel.voxel.setValue(32.04)
    panel.voxel_confirm.click()
    load.click()
    assert app.loaded_calls == 1


def _save_button(panel):
    from qtpy.QtWidgets import QPushButton

    return [b for b in panel.findChildren(QPushButton) if "save figure" in b.text()][0]


def _answer_dialog(monkeypatch, chosen):
    """Stand in for the native save dialog, which cannot be driven from a test."""
    from qtpy.QtWidgets import QFileDialog

    monkeypatch.setattr(QFileDialog, "getSaveFileName",
                        staticmethod(lambda *a, **k: (chosen, "")))


def test_the_save_figure_button_reaches_the_picker(qapp, monkeypatch, tmp_path):
    picker = FakePicker()
    panel = _layer_panel(picker)
    out = str(tmp_path / "fig.svg")
    _answer_dialog(monkeypatch, out)

    _save_button(panel).click()
    assert picker.figures == [out]


def test_cancelling_the_save_dialog_writes_nothing(qapp, monkeypatch):
    picker = FakePicker()
    panel = _layer_panel(picker)
    _answer_dialog(monkeypatch, "")  # what Qt returns when the user hits cancel

    _save_button(panel).click()
    assert picker.figures == []


def test_a_save_that_raises_is_swallowed_not_propagated(qapp, monkeypatch, capsys):
    """Same contract as the rebuild button: an exception out of a slot aborts PyQt5."""
    picker = FakePicker()
    picker.figure_error = RuntimeError("gl2ps said no")
    panel = _layer_panel(picker)
    _answer_dialog(monkeypatch, "fig.svg")

    button = _save_button(panel)
    button.click()
    assert button.isEnabled(), "a failed save must not leave the button stuck"
    assert "gl2ps said no" in capsys.readouterr().out
