"""The Crop tab.

The load-bearing claim is in the module docstring of `controls_crop` and is pinned by
`test_the_panel_never_touches_the_session_graph`: designing a crop must not be able to
change the tree you are looking at. Everything else here is the ordinary panel
contract -- it builds with nothing loaded, it refreshes without calling back into what
just moved it, and no slot is allowed to raise, because PyQt5 turns an escaped
exception into a process abort.
"""

from __future__ import annotations

import os
from types import SimpleNamespace

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("qtpy")

from hipct_seg_debug.edit import crop  # noqa: E402

from .test_crop import LAD, ordered_tree  # noqa: E402

#: `graph_from` gives every edge 20 points, laid out edge by edge, so a pick index
#: inside a segment's run resolves to that segment.
POINTS_PER_EDGE = 20


@pytest.fixture(scope="module")
def qapp():
    from qtpy.QtWidgets import QApplication

    app = QApplication.instance() or QApplication([])
    yield app


class FakePicker:
    """The `Picker3D` surface `controls_crop` touches."""

    def __init__(self):
        self._point_i = -1
        self.drawn = []
        self.cleared = 0

    def pick_segment(self, sid: int) -> None:
        """Put the pick in the middle of a segment's point run."""
        self._point_i = sid * POINTS_PER_EDGE + POINTS_PER_EDGE // 2

    def show_crop(self, drop, vessels):
        self.drawn.append((list(drop), list(vessels)))

    def clear_crop(self):
        self.cleared += 1


class FakeApp:
    def __init__(self, graph=None):
        self.session = None if graph is None else SimpleNamespace(graph=graph)
        self.picker = FakePicker()
        self.messages = []
        self.on_status = self.messages.append
        self.on_dataset = None


def _panel(app):
    from hipct_seg_debug.controls_crop import build_crop_panel

    return build_crop_panel(app)


@pytest.fixture
def spatial():
    return ordered_tree().to_spatial_graph()


@pytest.fixture
def app(spatial):
    return FakeApp(spatial)


@pytest.fixture
def panel(qapp, app):
    return _panel(app)


def _text(panel) -> str:
    return panel.widgets["summary"].text()


# ---------------------------------------------------------------- the basics


def test_it_builds_and_refreshes_with_nothing_loaded(qapp):
    box = _panel(FakeApp())
    box.refresh()
    assert "no dataset" in _text(box)
    assert not box.widgets["preview"].isEnabled()


def test_a_loaded_graph_enables_the_controls(panel):
    assert panel.widgets["preview"].isEnabled()
    assert "7 segment(s) loaded" in _text(panel)


def test_the_panel_never_touches_the_session_graph(panel, app, spatial):
    """Designing a crop must not be able to change the tree on screen."""
    before = (spatial.n_edge, spatial.n_point, len(spatial.points))

    app.picker.pick_segment(6)
    panel.widgets["use_radius"].setChecked(True)
    panel.widgets["min_radius"].setValue(500.0)
    panel.widgets["preview"].click()

    assert app.session.graph is spatial
    assert (spatial.n_edge, spatial.n_point, len(spatial.points)) == before
    assert panel.state["graph"].triple is not spatial
    assert panel.state["plan"].drop, "the preview still selected something"


# ---------------------------------------------------------------- picking


def test_adding_a_picked_segment_names_it(panel, app):
    panel.widgets["vessel_name"].setCurrentText("LAD")
    for sid in sorted(LAD):
        app.picker.pick_segment(sid)
        _click_add(panel)

    assert panel.state["vessels"] == {"LAD": set(LAD)}
    assert panel.widgets["vessels"].count() == 1


def test_picking_the_same_segment_twice_takes_it_back_off(panel, app):
    panel.widgets["vessel_name"].setCurrentText("LAD")
    app.picker.pick_segment(1)
    _click_add(panel)
    _click_add(panel)
    assert panel.state["vessels"] == {}


def test_adding_with_nothing_picked_says_so_rather_than_raising(panel, app):
    app.picker._point_i = -1
    panel.widgets["vessel_name"].setCurrentText("LAD")
    _click_add(panel)
    assert "nothing picked" in _text(panel)


def _click_add(panel):
    from qtpy.QtWidgets import QPushButton

    [b for b in panel.findChildren(QPushButton) if b.text().startswith("Add pick")][0].click()


def _click(panel, prefix):
    from qtpy.QtWidgets import QPushButton

    [b for b in panel.findChildren(QPushButton) if b.text().startswith(prefix)][0].click()


def test_a_hand_marked_drop_is_recorded_and_toggles(panel, app):
    app.picker.pick_segment(6)
    _click(panel, "Drop pick")
    assert panel.state["drop_segments"] == [6]
    _click(panel, "Drop pick")
    assert panel.state["drop_segments"] == []


def test_a_hand_marked_prune_walks_away_from_the_nearer_node(panel, app):
    """Clicking near the tip prunes the tip -- the controller's own heuristic."""
    graph = panel.state["graph"]
    app.picker._point_i = 4 * POINTS_PER_EDGE + 1  # segment 4, near its first point
    _click(panel, "Prune past")
    sid, node = panel.state["prune_at"][0]
    assert (sid, node) == (4, graph.segment(4)["node2"])

    panel.state["prune_at"].clear()
    app.picker._point_i = 4 * POINTS_PER_EDGE + POINTS_PER_EDGE - 2  # near the far end
    _click(panel, "Prune past")
    assert panel.state["prune_at"][0] == (4, graph.segment(4)["node1"])


# ---------------------------------------------------------------- tracing


def _trace(panel, app, name, start, end):
    """Name a vessel the two-pick way: mode on, pick the ostium, pick the far end."""
    panel.widgets["trace_mode"].setChecked(True)
    panel.widgets["vessel_name"].setCurrentText(name)
    app.picker.pick_segment(start)
    _click_add(panel)
    app.picker.pick_segment(end)
    _click_add(panel)


def test_two_picks_name_every_segment_between_them(panel, app):
    _trace(panel, app, "LAD", 0, 6)
    assert panel.state["vessels"] == {"LAD": {0, 1, 4, 5, 6}}
    assert "traced 5 segment(s)" in _text(panel)


def test_the_first_pick_only_arms_the_trace(panel, app):
    panel.widgets["trace_mode"].setChecked(True)
    panel.widgets["vessel_name"].setCurrentText("LAD")
    app.picker.pick_segment(0)
    _click_add(panel)

    assert panel.state["vessels"] == {}, "one pick names nothing on its own"
    assert panel.state["trace_start"] == 0
    assert "start at segment 0" in _text(panel)


def test_a_second_trace_adds_to_the_vessel_rather_than_replacing_it(panel, app):
    _trace(panel, app, "LAD", 1, 2)
    _trace(panel, app, "LAD", 3, 3)
    assert panel.state["vessels"] == {"LAD": {1, 2, 3}}


def test_a_trace_draws_the_vessel_it_just_named(panel, app):
    _trace(panel, app, "LAD", 0, 2)
    _drop, vessels = app.picker.drawn[-1]
    assert [name for name, _colour, _lines in vessels] == ["LAD"]
    assert len(vessels[0][2]) == 3


def test_cancelling_forgets_the_start_pick(panel, app):
    panel.widgets["trace_mode"].setChecked(True)
    panel.widgets["vessel_name"].setCurrentText("LAD")
    app.picker.pick_segment(0)
    _click_add(panel)
    _click(panel, "Cancel trace")

    assert panel.state["trace_start"] is None
    app.picker.pick_segment(6)
    _click_add(panel)
    assert panel.state["trace_start"] == 6, "the next pick starts a new trace"
    assert panel.state["vessels"] == {}


def test_leaving_the_mode_drops_a_half_finished_trace(panel, app):
    panel.widgets["trace_mode"].setChecked(True)
    panel.widgets["vessel_name"].setCurrentText("LAD")
    app.picker.pick_segment(0)
    _click_add(panel)
    panel.widgets["trace_mode"].setChecked(False)

    assert panel.state["trace_start"] is None
    app.picker.pick_segment(6)
    _click_add(panel)
    assert panel.state["vessels"] == {"LAD": {6}}, "one pick, one segment, as before"


def test_the_mode_off_still_names_one_segment_at_a_time(panel, app):
    panel.widgets["vessel_name"].setCurrentText("LAD")
    app.picker.pick_segment(0)
    _click_add(panel)
    assert panel.state["vessels"] == {"LAD": {0}}
    assert panel.state["trace_start"] is None


def test_a_trace_that_cannot_be_made_lands_in_the_label_and_disarms(qapp):
    """A start left armed by a failed trace would name a vessel out of the wrong two
    picks the next time the operator clicked."""
    from .conftest_geometry import graph_from

    two_trees = graph_from(
        [(0.0, 0.0, 0.0), (2000.0, 0.0, 0.0), (0.0, 5000.0, 0.0), (2000.0, 5000.0, 0.0)],
        [(0, 1, POINTS_PER_EDGE, 500.0), (2, 3, POINTS_PER_EDGE, 500.0)],
    ).to_spatial_graph()
    app = FakeApp(two_trees)
    box = _panel(app)
    _trace(box, app, "LAD", 0, 1)

    assert "not connected" in _text(box)
    assert box.state["trace_start"] is None
    assert box.state["vessels"] == {}


def test_a_traced_vessel_round_trips_through_the_sidecar(panel, app, tmp_path):
    """A trace is a selection like any other -- the sidecar cannot tell how it was made."""
    _trace(panel, app, "LAD", 0, 6)
    path = tmp_path / "c.json"
    panel.widgets["sidecar"].setText(str(path))
    _click(panel, "Save")

    fresh = _panel(FakeApp(app.session.graph))
    fresh.load(path)
    assert fresh.state["vessels"] == {"LAD": {0, 1, 4, 5, 6}}


# ---------------------------------------------------------------- preview


def test_preview_draws_the_selection_and_the_vessels(panel, app):
    panel.widgets["vessel_name"].setCurrentText("LAD")
    for sid in sorted(LAD):
        app.picker.pick_segment(sid)
        _click_add(panel)
    panel.widgets["use_ratio"].setChecked(True)
    panel.widgets["denominator"].setValue(4.0)
    panel.widgets["preview"].click()

    drop, vessels = app.picker.drawn[-1]
    assert len(drop) == len(panel.state["plan"].drop) > 0
    assert [name for name, _colour, _lines in vessels] == ["LAD"]
    assert vessels[0][1] == crop.VESSEL_COLORS[0]
    assert "would drop" in _text(panel)


def test_a_ratio_with_no_named_vessel_reports_into_the_label(panel):
    """`CropError` is a message, not a traceback and not a dead window."""
    panel.widgets["use_ratio"].setChecked(True)
    panel.widgets["preview"].click()
    assert "named main vessel" in _text(panel)


def test_the_vessel_ostium_shows_in_the_list(panel, app):
    panel.widgets["vessel_name"].setCurrentText("LAD")
    for sid in sorted(LAD):
        app.picker.pick_segment(sid)
        _click_add(panel)
    panel.widgets["preview"].click()

    assert "ostium R = 700 um" in panel.widgets["vessels"].item(0).text()


# ---------------------------------------------------------------- sidecar


def test_the_sidecar_round_trips_through_the_panel(panel, app, tmp_path):
    panel.widgets["vessel_name"].setCurrentText("LAD")
    for sid in sorted(LAD):
        app.picker.pick_segment(sid)
        _click_add(panel)
    app.picker.pick_segment(6)
    _click(panel, "Drop pick")
    panel.widgets["use_strahler"].setChecked(True)
    panel.widgets["strahler"].setValue(3)

    path = tmp_path / "c.json"
    panel.widgets["sidecar"].setText(str(path))
    _click(panel, "Save")
    assert path.exists()

    fresh = _panel(FakeApp(app.session.graph))
    fresh.load(path)
    assert fresh.state["vessels"] == {"LAD": set(LAD)}
    assert fresh.state["drop_segments"] == [6]
    assert fresh.widgets["use_strahler"].isChecked()
    assert fresh.widgets["strahler"].value() == 3


def test_loading_a_sidecar_does_not_call_back_into_the_widgets(panel, app, tmp_path):
    """The blockSignals contract: a programmatic write is not a user edit."""
    path = tmp_path / "c.json"
    panel.widgets["sidecar"].setText(str(path))
    panel.widgets["use_radius"].setChecked(True)
    panel.widgets["min_radius"].setValue(250.0)
    _click(panel, "Save")

    fresh = _panel(FakeApp(app.session.graph))
    fresh.load(path)
    assert fresh.state["plan"] is None, "loading must not silently run a preview"
    assert fresh.widgets["min_radius"].value() == pytest.approx(250.0)


def test_load_refuses_a_foreign_schema(panel, tmp_path):
    """`load` is the programmatic door and raises; the *button* is what swallows it."""
    path = tmp_path / "bad.json"
    path.write_text('{"schema": "somebody.else/1"}', encoding="utf-8")

    with pytest.raises(ValueError, match="expected schema"):
        panel.load(path)


def test_load_refuses_a_sidecar_whose_vessel_no_longer_resolves(panel, tmp_path):
    """A vessel missing its ostium silently re-thresholds everything below it."""
    graph = panel.state["graph"]
    plan = crop.plan(graph, crop.Rule(), vessels={"LAD": set(LAD)})
    document = crop.document(graph, plan, crop.Rule())
    document["vessels"]["LAD"]["seg_keys"][0] = "0" * 16
    path = tmp_path / "c.json"
    crop.write(path, document)

    with pytest.raises(crop.CropError, match="main vessel 'LAD'"):
        panel.load(path)


def test_a_failing_slot_lands_in_the_label_rather_than_aborting(panel, tmp_path):
    """PyQt5 calls qFatal on an exception that escapes a slot, so none may."""
    panel.widgets["sidecar"].setText(str(tmp_path / "nowhere" / "c.json"))
    panel.state["graph"] = None  # make Save fail from inside the slot
    _click(panel, "Save")
    assert "no graph loaded" in _text(panel)


# ---------------------------------------------------------------- hand-off


def test_the_command_line_names_the_sidecar_and_the_output(qapp, app, tmp_path):
    from hipct_seg_debug import cliform
    from hipct_seg_debug.controls_crop import build_crop_panel
    from hipct_seg_debug.edit.__main__ import build_parser

    spec = {s.name: s for s in cliform.describe_parser(build_parser())}["crop"]
    box = build_crop_panel(app, None, spec)
    box.widgets["sidecar"].setText(str(tmp_path / "c.json"))
    box.widgets["out"].setText(str(tmp_path / "cropped.am"))
    box.widgets["use_strahler"].setChecked(True)
    box.widgets["strahler"].setValue(2)

    line = box.widgets["command"].text()
    assert "crop" in line and "--crop-json=" in line
    assert "--min-strahler=2" in line and "--out=" in line


def test_a_dataset_swap_drops_the_selection(panel, app, spatial):
    panel.widgets["vessel_name"].setCurrentText("LAD")
    app.picker.pick_segment(1)
    _click_add(panel)
    assert panel.state["vessels"]

    app.session = SimpleNamespace(graph=ordered_tree().to_spatial_graph())
    panel.refresh()

    assert panel.state["vessels"] == {}, "segment ids mean nothing in the next graph"
    assert app.picker.cleared >= 1


# ------------------------------------------------------------- 3D key binding


def test_bind_keys_binds_callbacks_pyvista_accepts(panel):
    """Bound against a *real* plotter, because that is where the rule lives.

    pyvista validates key callbacks by walking every parameter and rejecting the
    callable if any lacks a default (``render_window_interactor.add_key_event``). A
    ``*args`` parameter always reports no default, so ``def run(*_args)`` is refused
    even though it is callable with zero arguments -- and the refusal is a ``TypeError``
    out of ``attach_control_dock``, which takes the whole session down before the window
    opens.

    A hand-written fake plotter cannot catch that: it records the callback without ever
    introspecting it. This binds the genuine article instead.
    """
    pv = pytest.importorskip("pyvista")
    from hipct_seg_debug.controls_crop import KEYS

    plotter = pv.Plotter(off_screen=True)
    panel.bind_keys(plotter)
    bound = plotter.iren._key_press_event_callbacks
    assert all(bound.get(key) for key in KEYS)


def test_a_bound_key_callback_takes_no_required_arguments(panel):
    """The rule itself, stated once, so a failure says why rather than just that."""
    pv = pytest.importorskip("pyvista")
    from inspect import signature

    from hipct_seg_debug.controls_crop import KEYS

    plotter = pv.Plotter(off_screen=True)
    panel.bind_keys(plotter)
    for key in KEYS:
        for callback in plotter.iren._key_press_event_callbacks[key]:
            required = [
                p.name for p in signature(callback).parameters.values()
                if p.default is p.empty
            ]
            assert not required, f"{key!r} callback has required parameter(s) {required}"


def test_a_bound_key_action_that_raises_does_not_escape(panel, app):
    """`guarded` exists because PyQt5 aborts the process on an escaped exception.

    Changing its signature to satisfy pyvista must not quietly lose that.
    """
    pv = pytest.importorskip("pyvista")
    from hipct_seg_debug.controls_crop import KEYS

    plotter = pv.Plotter(off_screen=True)
    panel.bind_keys(plotter)
    # 'm' adds the pick to the named vessel; with no name set that is an ordinary
    # refusal, so break it properly instead.
    panel.state["graph"] = object()  # every attribute access on this will raise
    plotter.iren._key_press_event_callbacks[KEYS[0]][0]()
    assert _text(panel), "the failure should have landed in the summary label"
