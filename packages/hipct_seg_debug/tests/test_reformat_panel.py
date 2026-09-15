"""The Reformat tab.

The load-bearing claim is the same one the Crop tab makes and for the same reason:
designing a reformat must not be able to change the tree you are looking at. It is
pinned by ``test_the_panel_never_touches_the_session_graph``.

Everything else here is the ordinary panel contract -- it builds with nothing loaded, it
refreshes without calling back into what just moved it, and **no slot is allowed to
raise**, because PyQt5 turns an escaped exception into a process abort rather than into a
traceback.
"""

from __future__ import annotations

import os
from types import SimpleNamespace

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("qtpy")

from .conftest_geometry import graph_from  # noqa: E402
from .test_reformat import FakeStack, unit_frame  # noqa: E402

#: `graph_from` lays points out edge by edge, so a pick index inside a segment's run
#: resolves to that segment.
POINTS_PER_EDGE = 20

NODES = [(20.0, 20.0, 20.0), (60.0, 20.0, 20.0), (100.0, 20.0, 20.0),
         (100.0, 60.0, 20.0)]
EDGES = [(0, 1, POINTS_PER_EDGE, 4.0), (1, 2, POINTS_PER_EDGE, 4.0),
         (2, 3, POINTS_PER_EDGE, 4.0)]


@pytest.fixture(scope="module")
def qapp():
    from qtpy.QtWidgets import QApplication

    app = QApplication.instance() or QApplication([])
    yield app


class FakePicker:
    """The `Picker3D` surface `controls_reformat` touches."""

    def __init__(self):
        self._point_i = -1
        self.drawn = []
        self.dropped = []
        self.cleared = 0
        self.slices = []

    def pick_segment(self, sid: int) -> None:
        """Put the pick in the middle of a segment's point run."""
        self._point_i = sid * POINTS_PER_EDGE + POINTS_PER_EDGE // 2

    def show_reformat(self, polylines, dropped=()):
        self.drawn.append(list(polylines))
        self.dropped.append(list(dropped))

    def clear_reformat(self):
        self.cleared += 1

    def set_current_slice(self, z, defer=False):
        self.slices.append(int(z))


class FakeApp:
    """A `ViewerApp` stand-in with a session the panel can actually build from."""

    def __init__(self, graph=None, *, sampleable=False):
        self.picker = FakePicker()
        self.messages = []
        self.on_status = self.messages.append
        self.on_dataset = None
        self.opened = []
        self.sections_in_3d = []
        self.in_world = []
        self.reformat = None
        if graph is None:
            self.session = None
            return
        shape = (40, 80, 130)
        self.session = SimpleNamespace(
            graph=graph,
            frame=unit_frame(shape),
            stack=FakeStack(np.zeros(shape, dtype=np.uint16)),
            labels=np.ones(shape, dtype=np.uint8) if sampleable else None,
        )

    def status(self, message):
        self.messages.append(message)

    def open_reformat(self, stack, *, sections_in_3d=False, in_world=True):
        self.opened.append(stack)
        self.sections_in_3d.append(bool(sections_in_3d))
        self.in_world.append(bool(in_world))
        self.reformat = stack if in_world else None


def _panel(app):
    from hipct_seg_debug.controls_reformat import build_reformat_panel

    return build_reformat_panel(app)


def _settle(qapp, panel, timeout=60.0):
    """Pump the event loop until the build finishes.

    ``Show`` genuinely is asynchronous -- it hands the decode to a worker thread and
    opens the window from the Qt thread when it lands -- so a test that read the summary
    straight after the click would be reading "building...". Waiting here rather than
    making the panel synchronous under test keeps the tested path the shipped one.
    """
    import time

    end = time.time() + timeout
    while time.time() < end:
        qapp.processEvents()
        if not panel.state["busy"]:
            qapp.processEvents()  # let the queued `done` callback run
            return True
        time.sleep(0.01)
    raise AssertionError(f"build did not finish: {panel.widgets['summary'].text()}")


@pytest.fixture
def spatial():
    return graph_from(NODES, EDGES).to_spatial_graph()


@pytest.fixture
def app(spatial):
    return FakeApp(spatial, sampleable=True)


# --------------------------------------------------------------------------- #


def test_it_builds_with_nothing_loaded(qapp):
    panel = _panel(FakeApp())
    assert "no dataset loaded" in panel.widgets["summary"].text()
    assert not panel.widgets["show"].isEnabled()


def test_the_panel_never_touches_the_session_graph(qapp, app, spatial):
    """The load-bearing claim: the panel works on a copy, always."""
    before = spatial.points.copy()
    panel = _panel(app)
    for sid in (0, 1, 2):
        app.picker.pick_segment(sid)
        panel.widgets["add"].click()
    panel.widgets["check"].click()
    panel.widgets["show"].click()
    _settle(qapp, panel)

    assert panel.state["graph"] is not spatial
    np.testing.assert_array_equal(app.session.graph.points, before)


def test_a_pick_adds_a_segment_and_picking_it_again_removes_it(qapp, app):
    panel = _panel(app)
    app.picker.pick_segment(1)
    panel.widgets["add"].click()
    assert panel.state["selected"] == [1]
    panel.widgets["add"].click()
    assert panel.state["selected"] == []


def test_picks_keep_their_order(qapp, app):
    """Pick order is the tie-breaker when a chain walk is ambiguous, so it is kept."""
    panel = _panel(app)
    for sid in (2, 0, 1):
        app.picker.pick_segment(sid)
        panel.widgets["add"].click()
    assert panel.state["selected"] == [2, 0, 1]


def test_adding_a_pick_previews_the_selection_in_3d(qapp, app):
    panel = _panel(app)
    app.picker.pick_segment(0)
    panel.widgets["add"].click()
    assert app.picker.drawn and len(app.picker.drawn[-1]) == 1


def test_check_reports_the_geometry_without_reading_any_images(qapp, app):
    panel = _panel(app)
    for sid in (0, 1):
        app.picker.pick_segment(sid)
        panel.widgets["add"].click()
    panel.widgets["check"].click()

    text = panel.widgets["summary"].text()
    assert "planes" in text and "mm" in text
    assert app.session.stack.reads == [], "Check must not decode anything"


def test_check_names_the_node_a_branching_selection_splits_at(qapp, spatial):
    """A selection with a branch in it has no total order, so it says so."""
    graph = graph_from(
        NODES + [(100.0, 20.0, 60.0)],
        EDGES + [(2, 4, POINTS_PER_EDGE, 4.0)],
    ).to_spatial_graph()
    app = FakeApp(graph, sampleable=True)
    panel = _panel(app)
    for sid in (1, 2, 3):
        app.picker.pick_segment(sid)
        panel.widgets["add"].click()
    panel.widgets["check"].click()
    assert "node 2" in panel.widgets["summary"].text()


def test_show_builds_a_stack_and_opens_the_window(qapp, app):
    panel = _panel(app)
    for sid in (0, 1):
        app.picker.pick_segment(sid)
        panel.widgets["add"].click()
    panel.widgets["mode"].setCurrentText("fixed")
    panel.widgets["size_px"].setValue(21)
    panel.widgets["show"].click()
    _settle(qapp, panel)

    assert app.opened, panel.widgets["summary"].text()
    stack = app.opened[-1]
    assert stack.raw.shape[1:] == (21, 21)
    assert stack.mask is not None


def test_the_mask_checkbox_is_honoured(qapp, app):
    panel = _panel(app)
    app.picker.pick_segment(0)
    panel.widgets["add"].click()
    panel.widgets["mode"].setCurrentText("fixed")
    panel.widgets["size_px"].setValue(15)
    panel.widgets["with_mask"].setChecked(False)
    panel.widgets["show"].click()
    _settle(qapp, panel)
    assert app.opened[-1].mask is None


def test_manual_mode_enables_only_the_widgets_it_reads(qapp, app):
    panel = _panel(app)
    panel.widgets["mode"].setCurrentText("manual")
    assert panel.widgets["half_um"].isEnabled()
    assert not panel.widgets["radii_k"].isEnabled()
    panel.widgets["mode"].setCurrentText("radius")
    assert panel.widgets["radii_k"].isEnabled()
    assert not panel.widgets["half_um"].isEnabled()


def test_a_new_dataset_drops_the_selection(qapp, app, spatial):
    panel = _panel(app)
    app.picker.pick_segment(0)
    panel.widgets["add"].click()
    assert panel.state["selected"]

    app.session = SimpleNamespace(
        graph=graph_from(NODES, EDGES).to_spatial_graph(),
        frame=app.session.frame, stack=app.session.stack, labels=None,
    )
    panel.refresh()
    # A selection is segment ids in the graph it was made on; it means something else
    # entirely in the next one.
    assert panel.state["selected"] == []
    assert app.picker.cleared >= 1


def test_no_slot_raises_with_nothing_selected(qapp, app):
    """PyQt5 aborts the process on an exception escaping a slot."""
    panel = _panel(app)
    for key in ("add", "clear", "check", "show", "cancel_trace"):
        panel.widgets[key].click()
    assert not app.opened
    assert panel.widgets["summary"].text()


def test_no_slot_raises_with_no_dataset(qapp):
    app = FakeApp()
    panel = _panel(app)
    for key in ("add", "clear", "check", "show", "cancel_trace"):
        panel.widgets[key].click()
    assert panel.widgets["summary"].text()


def test_a_run_that_would_take_too_many_planes_is_refused_with_the_number(qapp, app):
    """A tiny step on a long run is minutes of decode; it says so instead of starting."""
    panel = _panel(app)
    for sid in (0, 1, 2):
        app.picker.pick_segment(sid)
        panel.widgets["add"].click()
    panel.widgets["step_um"].setValue(0.01)  # 120 um of vessel = 12,001 planes
    panel.widgets["show"].click()
    _settle(qapp, panel)
    assert not app.opened
    assert "planes" in panel.widgets["summary"].text()


def test_bind_keys_clears_before_binding(qapp, app):
    """`add_key_event` appends rather than replaces, and cannot be undone."""
    from hipct_seg_debug.controls_reformat import KEYS

    class FakePlotter:
        def __init__(self):
            self.cleared, self.bound = [], []

        def clear_events_for_key(self, key):
            self.cleared.append(key)

        def add_key_event(self, key, _fn):
            self.bound.append(key)

    plotter = FakePlotter()
    _panel(app).bind_keys(plotter)
    assert plotter.cleared == list(KEYS) == plotter.bound


def test_the_keys_do_not_collide_with_anything_else_in_the_window():
    """Every letter is taken, which is why this panel is on digits."""
    from hipct_seg_debug import viewer3d
    from hipct_seg_debug.controls_reformat import KEYS
    from hipct_seg_debug.controls_crop import KEYS as CROP_KEYS
    from hipct_seg_debug.controls_sections import KEYS as SECTION_KEYS
    from hipct_seg_debug.edit.controller import INSTRUCTIONS

    edit_keys = {
        line.strip().split()[0].strip("'\"")
        for line in INSTRUCTIONS.splitlines() if line.strip()
    }
    # The Sections tab is on digits too, and `bind_keys` clears a key before binding
    # it -- so a collision there is silent theft rather than a double binding.
    taken = (set(CROP_KEYS) | set(SECTION_KEYS) | edit_keys
             | set("vnbscigarq") | set("pwrq"))
    assert not (set(KEYS) & taken), f"collides: {set(KEYS) & taken}"
    assert "3" not in KEYS, "3 is VTK's stereo toggle and cannot be cleared"
    assert "reformat_path" in viewer3d.EXTRA_KEYS


def test_bind_keys_binds_callbacks_pyvista_accepts(qapp, app):
    """Bound against a *real* plotter, because that is where the rule lives.

    pyvista validates key callbacks by walking every parameter and rejecting the
    callable if any lacks a default (``render_window_interactor.add_key_event``). A
    ``*args`` parameter always reports no default, so ``def run(*_args)`` is refused
    even though it is callable with zero arguments -- and the refusal is a ``TypeError``
    out of ``attach_control_dock``, which takes the whole session down before the window
    opens.

    ``test_bind_keys_clears_before_binding`` above keeps the fake, because only a fake
    can show that the clear happened *before* the bind. This one proves the callback is
    something pyvista will actually accept.
    """
    pv = pytest.importorskip("pyvista")
    from hipct_seg_debug.controls_reformat import KEYS

    plotter = pv.Plotter(off_screen=True)
    _panel(app).bind_keys(plotter)
    bound = plotter.iren._key_press_event_callbacks
    assert all(bound.get(key) for key in KEYS)


def test_a_bound_key_callback_takes_no_required_arguments(qapp, app):
    """The rule itself, stated once, so a failure says why rather than just that."""
    pv = pytest.importorskip("pyvista")
    from inspect import signature

    from hipct_seg_debug.controls_reformat import KEYS

    plotter = pv.Plotter(off_screen=True)
    _panel(app).bind_keys(plotter)
    for key in KEYS:
        for callback in plotter.iren._key_press_event_callbacks[key]:
            required = [
                p.name for p in signature(callback).parameters.values()
                if p.default is p.empty
            ]
            assert not required, f"{key!r} callback has required parameter(s) {required}"


def test_a_bound_key_action_that_raises_does_not_escape(qapp, app):
    """`guarded` exists because PyQt5 aborts the process on an escaped exception."""
    pv = pytest.importorskip("pyvista")
    from hipct_seg_debug.controls_reformat import KEYS

    plotter = pv.Plotter(off_screen=True)
    panel = _panel(app)
    panel.bind_keys(plotter)
    panel.state["graph"] = object()  # every attribute access on this will raise
    plotter.iren._key_press_event_callbacks[KEYS[0]][0]()
    assert panel.widgets["summary"].text()


def test_the_digit_keys_are_not_already_taken_by_pyvista(qapp):
    """`3` is avoided deliberately; the rest must be free before we clear them."""
    pv = pytest.importorskip("pyvista")
    from hipct_seg_debug.controls_reformat import KEYS

    defaults = set(pv.Plotter(off_screen=True).iren._key_press_event_callbacks)
    assert not (set(KEYS) & defaults), f"pyvista already binds {set(KEYS) & defaults}"


def test_the_3d_section_stack_is_off_unless_asked_for(qapp, app):
    """The current section always follows the slider; the whole stack costs a
    texture upload per plane, so it stays opt-in."""
    panel = _panel(app)
    app.picker.pick_segment(0)
    panel.widgets["add"].click()
    panel.widgets["mode"].setCurrentText("fixed")
    panel.widgets["size_px"].setValue(15)

    assert not panel.widgets["with_sections"].isChecked()
    panel.widgets["show"].click()
    _settle(qapp, panel)
    assert app.sections_in_3d == [False]

    panel.widgets["with_sections"].setChecked(True)
    panel.widgets["show"].click()
    _settle(qapp, panel)
    assert app.sections_in_3d[-1] is True


# ------------------------------------------------- interpolation and sampling density


def test_cubic_is_the_default_and_the_order_reaches_the_build(qapp, app):
    from hipct_seg_debug import reformat as rf

    panel = _panel(app)
    assert panel.widgets["order"].currentData() == rf.DEFAULT_ORDER == 3

    app.picker.pick_segment(0)
    panel.widgets["add"].click()
    panel.widgets["mode"].setCurrentText("fixed")
    panel.widgets["size_px"].setValue(15)
    panel.widgets["order"].setCurrentIndex(
        [v for v, _l in __import__(
            "hipct_seg_debug.controls_reformat", fromlist=["ORDERS"]
        ).ORDERS].index(1)
    )
    panel.widgets["show"].click()
    _settle(qapp, panel)
    assert app.opened[-1].stats.order == 1


def test_match_voxel_sizes_the_grid_to_the_data(qapp, app):
    """And sizes it from the half-width the build will use, not the one asked for.

    The two differ whenever the curvature clamp bites, and because ``size_px`` is
    fixed that difference is pure magnification.
    """
    panel = _panel(app)
    app.picker.pick_segment(0)
    panel.widgets["add"].click()
    panel.widgets["mode"].setCurrentText("fixed")
    panel.widgets["size_px"].setValue(401)   # deliberately far too fine

    panel.widgets["match"].click()
    assert panel.widgets["size_px"].value() < 401
    assert "voxel per pixel" in panel.widgets["summary"].text()

    panel.widgets["show"].click()
    _settle(qapp, panel)
    over = app.opened[-1].geometry.oversampling
    assert over.max() < 1.6, f"still magnifying after match: {over.max():.2f}x"


def test_the_build_reports_oversampling_and_where_the_time_went(qapp, app):
    panel = _panel(app)
    app.picker.pick_segment(0)
    panel.widgets["add"].click()
    panel.widgets["mode"].setCurrentText("fixed")
    panel.widgets["size_px"].setValue(21)
    panel.widgets["show"].click()
    _settle(qapp, panel)

    text = app.opened[-1].describe()
    assert "um voxel" in text
    assert "decode" in text and "interpolate" in text


def test_match_voxel_with_nothing_selected_does_not_raise(qapp, app):
    panel = _panel(app)
    panel.widgets["match"].click()
    assert panel.widgets["summary"].text()


def test_native_is_the_default_mode(qapp, app):
    panel = _panel(app)
    assert panel.widgets["mode"].currentText() == "native"


def test_native_disables_the_widgets_it_does_not_read(qapp, app):
    panel = _panel(app)
    panel.widgets["mode"].setCurrentText("native")
    assert not panel.widgets["radii_k"].isEnabled()
    assert not panel.widgets["half_um"].isEnabled()
    assert not panel.widgets["px_um"].isEnabled()
    assert panel.widgets["size_px"].isEnabled()


def test_a_native_build_comes_back_unmagnified(qapp, app):
    panel = _panel(app)
    app.picker.pick_segment(0)
    panel.widgets["add"].click()
    panel.widgets["mode"].setCurrentText("native")
    panel.widgets["size_px"].setValue(21)
    panel.widgets["show"].click()
    _settle(qapp, panel)

    geom = app.opened[-1].geometry
    assert geom.mode == "native"
    assert np.allclose(geom.oversampling, 1.0)


def test_match_voxel_in_native_mode_sizes_the_frame_not_the_pixel(qapp, app):
    """The pitch is already the voxel, so the only thing left to set is how much you see."""
    panel = _panel(app)
    app.picker.pick_segment(0)
    panel.widgets["add"].click()
    panel.widgets["mode"].setCurrentText("native")
    panel.widgets["size_px"].setValue(401)
    panel.widgets["match"].click()

    assert panel.widgets["size_px"].value() < 401
    assert "never magnifies" in panel.widgets["summary"].text()

    panel.widgets["show"].click()
    _settle(qapp, panel)
    assert np.allclose(app.opened[-1].geometry.oversampling, 1.0)


# ---------------------------------------------- a selection that is not one chain


@pytest.fixture
def split_app():
    """Segments 0-1-2 share nodes; segment 3 is a separate vessel entirely.

    `graph_from` numbers edges in order, so the appended edge is segment **3** -- and
    `FakePicker.pick_segment` indexes into the flat point run, so naming a segment that
    does not exist silently adds nothing rather than failing.
    """
    nodes = list(NODES) + [(20.0, 60.0, 20.0), (60.0, 60.0, 20.0)]
    edges = list(EDGES) + [(4, 5, POINTS_PER_EDGE, 4.0)]
    return FakeApp(graph_from(nodes, edges).to_spatial_graph(), sampleable=True)


def test_a_disconnected_selection_builds_the_longest_run_and_says_so(qapp, split_app):
    """A stack is one continuous path, so the rest cannot be in it."""
    panel = _panel(split_app)
    for sid in (0, 1, 3):
        split_app.picker.pick_segment(sid)
        panel.widgets["add"].click()
    panel.widgets["mode"].setCurrentText("native")
    panel.widgets["size_px"].setValue(15)
    panel.widgets["show"].click()
    _settle(qapp, panel)

    stack = split_app.opened[-1]
    assert sorted(set(stack.centreline.seg_ids.tolist())) == [0, 1]
    text = stack.describe()
    assert "built the longest run only" in text
    assert "[3]" in text and "not sampled" in text


def test_the_3d_preview_separates_the_run_from_what_is_dropped(qapp, split_app):
    """Drawing them alike would show one thing while the build did another."""
    panel = _panel(split_app)
    for sid in (0, 1, 3):
        split_app.picker.pick_segment(sid)
        panel.widgets["add"].click()

    kept, dropped = split_app.picker.drawn[-1], split_app.picker.dropped[-1]
    assert len(kept) == 2, "segments 0 and 1 are the run"
    assert len(dropped) == 1, "segment 3 is not connected to it"


def test_the_segment_list_marks_what_is_not_in_the_run(qapp, split_app):
    panel = _panel(split_app)
    for sid in (0, 1, 3):
        split_app.picker.pick_segment(sid)
        panel.widgets["add"].click()

    rows = [panel.widgets["segments"].item(i).text()
            for i in range(panel.widgets["segments"].count())]
    assert sum("[not in the run]" in r for r in rows) == 1
    assert "[not in the run]" in [r for r in rows if r.startswith("segment 3")][0]


def test_deselecting_the_odd_one_out_brings_the_rest_back_into_one_run(qapp, split_app):
    """The mark is advisory, not a rejection -- the selection is still editable."""
    panel = _panel(split_app)
    for sid in (0, 1, 3):
        split_app.picker.pick_segment(sid)
        panel.widgets["add"].click()
    split_app.picker.pick_segment(3)
    panel.widgets["add"].click()          # toggle it back off

    rows = [panel.widgets["segments"].item(i).text()
            for i in range(panel.widgets["segments"].count())]
    assert not any("[not in the run]" in r for r in rows)
    assert not split_app.picker.dropped[-1]


def test_check_reports_the_split_before_any_decode(qapp, split_app):
    panel = _panel(split_app)
    for sid in (0, 1, 3):
        split_app.picker.pick_segment(sid)
        panel.widgets["add"].click()
    panel.widgets["check"].click()
    assert "separate runs" in panel.widgets["summary"].text()
    assert split_app.session.stack.reads == []


# ------------------------------------------------------------------------ tracing
#
# The answer to the section above: a hand-made selection can be interrupted, a traced
# one cannot. Two picks and a Dijkstra over nodes (`edit.crop.trace_path`) name every
# segment between them, so what lands in the selection is a path -- connected by
# construction, therefore one run.


def _trace(panel, app, start, end):
    """Select the two-pick way: mode on, one end of the run, then the other."""
    panel.widgets["trace_mode"].setChecked(True)
    app.picker.pick_segment(start)
    panel.widgets["add"].click()
    app.picker.pick_segment(end)
    panel.widgets["add"].click()


def test_two_picks_select_every_segment_between_them(qapp, app):
    panel = _panel(app)
    _trace(panel, app, 0, 2)
    assert panel.state["selected"] == [0, 1, 2]
    assert "traced 3 segment(s)" in panel.widgets["summary"].text()


def test_a_traced_selection_is_one_run_by_construction(qapp, app):
    """The whole point: `chain_segments` has nothing left to split."""
    panel = _panel(app)
    _trace(panel, app, 0, 2)
    listed = [panel.widgets["segments"].item(i).text()
              for i in range(panel.widgets["segments"].count())]
    assert len(listed) == 3
    assert not any("not in the run" in text for text in listed)


def test_a_trace_reports_the_geometry_it_just_selected(qapp, app):
    """Milliseconds and no decode, and a trace is exactly when it is worth knowing."""
    panel = _panel(app)
    _trace(panel, app, 0, 2)
    text = panel.widgets["summary"].text()
    assert "traced" in text and "planes" in text
    assert app.session.stack.reads == []


def test_the_first_pick_only_arms_the_trace(qapp, app):
    panel = _panel(app)
    panel.widgets["trace_mode"].setChecked(True)
    app.picker.pick_segment(0)
    panel.widgets["add"].click()

    assert panel.state["selected"] == [], "one pick selects nothing on its own"
    assert panel.state["trace_start"] == 0
    assert "armed at segment 0" in panel.widgets["summary"].text()


def test_a_trace_extends_the_selection_rather_than_replacing_it(qapp, app):
    panel = _panel(app)
    app.picker.pick_segment(0)
    panel.widgets["add"].click()  # picked by hand, before any tracing
    _trace(panel, app, 1, 2)
    assert panel.state["selected"] == [0, 1, 2], "path order, and nothing thrown away"


def test_a_second_trace_repeats_nothing(qapp, app):
    panel = _panel(app)
    _trace(panel, app, 0, 1)
    _trace(panel, app, 1, 2)
    assert panel.state["selected"] == [0, 1, 2]
    assert "1 new" in panel.widgets["summary"].text()


def test_a_trace_previews_the_run_in_3d(qapp, app):
    panel = _panel(app)
    _trace(panel, app, 0, 2)
    assert len(app.picker.drawn[-1]) == 3
    assert app.picker.dropped[-1] == [], "a path has nothing outside the run"


def test_cancelling_forgets_the_start_pick(qapp, app):
    panel = _panel(app)
    panel.widgets["trace_mode"].setChecked(True)
    app.picker.pick_segment(0)
    panel.widgets["add"].click()
    panel.widgets["cancel_trace"].click()

    assert panel.state["trace_start"] is None
    app.picker.pick_segment(2)
    panel.widgets["add"].click()
    assert panel.state["trace_start"] == 2, "the next pick starts a new trace"
    assert panel.state["selected"] == []


def test_leaving_the_mode_drops_a_half_finished_trace(qapp, app):
    panel = _panel(app)
    panel.widgets["trace_mode"].setChecked(True)
    app.picker.pick_segment(0)
    panel.widgets["add"].click()
    panel.widgets["trace_mode"].setChecked(False)

    assert panel.state["trace_start"] is None
    app.picker.pick_segment(2)
    panel.widgets["add"].click()
    assert panel.state["selected"] == [2], "one pick, one segment, as before"


def test_clearing_the_selection_disarms_the_trace(qapp, app):
    """The start was going to extend a selection that no longer exists."""
    panel = _panel(app)
    panel.widgets["trace_mode"].setChecked(True)
    app.picker.pick_segment(0)
    panel.widgets["add"].click()
    panel.widgets["clear"].click()
    assert panel.state["trace_start"] is None


def test_a_new_dataset_disarms_the_trace(qapp, app):
    panel = _panel(app)
    panel.widgets["trace_mode"].setChecked(True)
    app.picker.pick_segment(0)
    panel.widgets["add"].click()

    app.session = SimpleNamespace(
        graph=graph_from(NODES, EDGES).to_spatial_graph(),
        frame=app.session.frame, stack=app.session.stack, labels=None,
    )
    panel.refresh()
    # A segment id means something else in the next graph, and a start pick is one.
    assert panel.state["trace_start"] is None


def test_a_trace_that_cannot_be_made_lands_in_the_label_and_disarms(qapp, split_app):
    """A start left armed by a failed trace would silently become the far end of the
    next one, selecting a run out of the wrong two picks."""
    panel = _panel(split_app)
    _trace(panel, split_app, 0, 3)

    assert "not connected" in panel.widgets["summary"].text()
    assert panel.state["trace_start"] is None
    assert panel.state["selected"] == []


def test_prefer_thick_routes_the_trace_round_a_thin_bridge(qapp):
    r"""Two vessels the segmentation fused where they merely cross: the short way
    between them is the artefact, and length alone will always take it.

        0 --start-- 1 --thin (20 um), 1000 um------- 2 --end-- 4
                     \
                      3 --fat (500 um)-- ... -------
    """
    nodes = [(0.0, 0.0, 0.0), (2000.0, 0.0, 0.0), (3000.0, 0.0, 0.0),
             (2000.0, 3000.0, 0.0), (5000.0, 0.0, 0.0)]
    edges = [(0, 1, POINTS_PER_EDGE, 600.0), (1, 2, POINTS_PER_EDGE, 20.0),
             (1, 3, POINTS_PER_EDGE, 500.0), (3, 2, POINTS_PER_EDGE, 500.0),
             (2, 4, POINTS_PER_EDGE, 600.0)]
    app = FakeApp(graph_from(nodes, edges).to_spatial_graph(), sampleable=True)
    panel = _panel(app)

    _trace(panel, app, 0, 4)
    assert panel.state["selected"] == [0, 1, 4], "by length: over the bridge"

    panel.widgets["clear"].click()
    panel.widgets["prefer_thick"].setChecked(True)
    _trace(panel, app, 0, 4)
    assert panel.state["selected"] == [0, 2, 3, 4], "by thickness: down the vessel"


# ------------------------------------------------------------------ save / load


def _build(qapp, app, panel, *, size_px=15, mode="native"):
    """Build a small stack through the GUI and return it.

    Clears first, because "Add pick" is a *toggle* -- calling this twice in a test
    would otherwise deselect the segment the first call selected and silently build
    nothing.
    """
    panel.widgets["clear"].click()
    app.picker.pick_segment(0)
    panel.widgets["add"].click()
    panel.widgets["mode"].setCurrentText(mode)
    panel.widgets["size_px"].setValue(size_px)
    panel.widgets["show"].click()
    _settle(qapp, panel)
    assert app.opened, panel.widgets["summary"].text()
    return app.opened[-1]


def test_save_is_off_until_there_is_something_to_save(qapp, app):
    panel = _panel(app)
    assert not panel.widgets["save"].isEnabled()
    assert "build or load a stack first" in panel.widgets["save_hint"].text()

    _build(qapp, app, panel)
    assert panel.widgets["save"].isEnabled()


def test_load_is_available_with_no_dataset_at_all(qapp):
    """The case the whole feature exists for."""
    panel = _panel(FakeApp())
    assert panel.widgets["load"].isEnabled()
    assert not panel.widgets["show"].isEnabled()


def test_the_hint_shows_the_filename_before_the_dialog_opens(qapp, app):
    panel = _panel(app)
    _build(qapp, app, panel, size_px=15, mode="native")
    panel.widgets["save_name"].setText("LAD")

    assert panel.widgets["save_hint"].text() == "LAD__native_15px__seg0.npz"
    panel.widgets["save_format"].setCurrentIndex(1)          # folder: TIFF + JSON
    assert panel.widgets["save_hint"].text() == "LAD__native_15px__seg0/  (a folder)"


def test_the_hint_follows_the_frame_size(qapp, app):
    panel = _panel(app)
    _build(qapp, app, panel, size_px=15)
    panel.widgets["save_name"].setText("LAD")
    assert "15px" in panel.widgets["save_hint"].text()

    _build(qapp, app, panel, size_px=21)
    assert "21px" in panel.widgets["save_hint"].text()


@pytest.mark.parametrize("fmt_index", [0, 1, 2])
def test_save_then_load_round_trips_through_the_gui(qapp, app, tmp_path, monkeypatch,
                                                    fmt_index):
    from qtpy.QtWidgets import QFileDialog

    from hipct_seg_debug.controls_reformat import FORMATS

    panel = _panel(app)
    original = _build(qapp, app, panel)
    panel.widgets["save_name"].setText("LAD")
    panel.widgets["save_format"].setCurrentIndex(fmt_index)
    fmt = FORMATS[fmt_index][0]

    target = tmp_path / "LAD__native_15px__seg0"
    monkeypatch.setattr(QFileDialog, "getSaveFileName",
                        staticmethod(lambda *a, **k: (str(target), "")))
    monkeypatch.setattr(QFileDialog, "getExistingDirectory",
                        staticmethod(lambda *a, **k: str(tmp_path)))
    panel.widgets["save"].click()
    _settle(qapp, panel)
    assert "wrote" in panel.widgets["summary"].text(), panel.widgets["summary"].text()

    written = target.with_suffix(".npz") if fmt == "npz" else target
    assert written.exists()

    # ...and back in through Load.
    opened = written if fmt == "npz" else written / "geometry.json"
    monkeypatch.setattr(QFileDialog, "getOpenFileName",
                        staticmethod(lambda *a, **k: (str(opened), "")))
    app.opened.clear()
    panel.widgets["load"].click()
    _settle(qapp, panel)

    assert app.opened, panel.widgets["summary"].text()
    back = app.opened[-1]
    np.testing.assert_array_equal(back.raw, original.raw)
    assert back.describe() == original.describe()


def test_a_loaded_stack_from_another_frame_is_shown_without_the_3d_overlays(
        qapp, app, tmp_path, monkeypatch):
    """The pixels are fine; only their world positions stop meaning anything."""
    from qtpy.QtWidgets import QFileDialog

    from hipct_seg_debug import reformat_io

    panel = _panel(app)
    original = _build(qapp, app, panel)
    path = reformat_io.save(original, tmp_path / "elsewhere", fmt="npz",
                            frame=unit_frame((90, 90, 90)))

    monkeypatch.setattr(QFileDialog, "getOpenFileName",
                        staticmethod(lambda *a, **k: (str(path), "")))
    app.opened.clear()
    panel.widgets["load"].click()
    _settle(qapp, panel)

    assert app.opened
    assert app.in_world[-1] is False
    assert "3D plane overlays are not drawn" in panel.widgets["summary"].text()


def test_loading_names_the_box_from_the_file(qapp, app, tmp_path, monkeypatch):
    from qtpy.QtWidgets import QFileDialog

    from hipct_seg_debug import reformat_io

    panel = _panel(app)
    original = _build(qapp, app, panel)
    path = reformat_io.save(original, tmp_path / "RCA__native_15px__seg0", fmt="npz",
                            frame=app.session.frame)

    monkeypatch.setattr(QFileDialog, "getOpenFileName",
                        staticmethod(lambda *a, **k: (str(path), "")))
    panel.widgets["load"].click()
    _settle(qapp, panel)
    # The base comes back so a re-save suggests the same name rather than "reformat".
    assert panel.widgets["save_name"].text() == "RCA"


def test_a_cancelled_dialog_writes_nothing(qapp, app, tmp_path, monkeypatch):
    from qtpy.QtWidgets import QFileDialog

    panel = _panel(app)
    _build(qapp, app, panel)
    monkeypatch.setattr(QFileDialog, "getSaveFileName", staticmethod(lambda *a, **k: ("", "")))
    monkeypatch.setattr(QFileDialog, "getOpenFileName", staticmethod(lambda *a, **k: ("", "")))
    panel.widgets["save"].click()
    panel.widgets["load"].click()
    assert list(tmp_path.iterdir()) == []
    assert not panel.state["busy"]


def test_loading_an_unreadable_file_reports_rather_than_raises(qapp, app, tmp_path,
                                                               monkeypatch):
    """PyQt5 aborts the process on an exception escaping a slot."""
    from qtpy.QtWidgets import QFileDialog

    junk = tmp_path / "junk.npz"
    junk.write_bytes(b"not an npz")
    monkeypatch.setattr(QFileDialog, "getOpenFileName",
                        staticmethod(lambda *a, **k: (str(junk), "")))
    panel = _panel(app)
    panel.widgets["load"].click()
    _settle(qapp, panel)
    assert "ReformatIOError" in panel.widgets["summary"].text()
    assert not app.opened


def test_save_and_load_do_not_raise_with_nothing_built(qapp, app):
    panel = _panel(app)
    panel.widgets["save"].click()
    assert panel.widgets["summary"].text()
