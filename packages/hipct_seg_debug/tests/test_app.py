"""`ViewerApp` — the state that used to live in `interactive`'s closures.

Driven against stubs, because what is being checked is the ordering of a dataset
swap, not the loading. The order matters more than it looks: the controller has to
dispose before the picker tears its actors down, painting has to be committed before
the mask it was painted onto is dropped, and the veto has to run before any of it.
"""

from __future__ import annotations

import argparse
from types import SimpleNamespace

import pytest

from hipct_seg_debug.app import ViewerApp


class FakePicker:
    def __init__(self):
        self.datasets = []
        self.shapes = []
        self.slices = []
        self.on_dataset_changing = None

    def set_dataset(self, **kw):
        self.datasets.append(kw)

    def set_slice_shapes(self, slab):
        self.shapes.append(slab)

    def set_current_slice(self, z, defer=False):
        self.slices.append(z)


class FakeSession:
    def __init__(self, name="one"):
        self.name = name
        self.graph = SimpleNamespace(n_vertex=3, n_edge=2, n_point=9)
        self.mesh = object()
        self.frame = object()
        self.stack = object()
        self.labels = object()
        self.cache = None
        self.closed = False
        self.reloaded = []
        self.mask = SimpleNamespace(release=lambda: None, peek=lambda: None)
        # A plain binary mask names nothing; a `.Regions.am` would list its trees here.
        self.materials = ()
        # One skeleton loaded; several would arrive merged as one graph of two trees.
        self.graph_paths = ["g.am"]
        self.graph_trees = [[0]]

    def builder(self):
        return SimpleNamespace(name=f"builder-{self.name}")

    def close(self):
        self.closed = True

    def reload_graph(self, path):
        self.reloaded.append(path)
        self.graph = SimpleNamespace(n_vertex=4, n_edge=3, n_point=12)

    def reload_surface(self, path):
        self.mesh = object()

    def validate(self, strict=True):
        return []

    def find_candidates(self):
        return [SimpleNamespace(id=1)]


class FakeController:
    def __init__(self, can_undo=False):
        self.graph = SimpleNamespace(history=SimpleNamespace(can_undo=can_undo))
        self.disposed = False
        self.detached = False

    def dispose(self):
        self.disposed = True

    def detach(self):
        self.detached = True


class FakePaint:
    def __init__(self, empty=True, path=None):
        self.edits = SimpleNamespace(is_empty=empty, describe=lambda: "3 voxels")
        self.edits_path = path
        self.committed = False
        self.saved = False

    def commit(self):
        self.committed = True

    def save(self):
        self.saved = True
        return self.edits_path


def _args(**kw):
    base = dict(graph="g.am", seg="s.am", surface="x.stl", raw="raw", edits=None,
                slab=5, roi=400, probability=False, no_tube=False, no_surface=False,
                volume=False, paint=False, paint_box=192, edit=False,
                seg_box_um=2000.0, seg_stride=4, edit_box_um=4000.0, edit_voxel_mm=None)
    base.update(kw)
    return argparse.Namespace(**base)


def _app(**kw):
    session = kw.pop("session", FakeSession())
    app = ViewerApp(_args(**kw), session=session)
    app.picker = FakePicker()
    return app


# ------------------------------------------------------------------- basics


def test_it_starts_from_the_session_it_was_given():
    app = _app()
    assert app.loaded and app.builder.name == "builder-one"


def test_describe_names_the_graph_file():
    assert "g.am" in _app().describe()


def test_describe_says_so_when_nothing_is_loaded():
    app = ViewerApp(_args(), session=None)
    assert app.describe() == "no dataset loaded"


# --------------------------------------------------------------- the veto


def test_nothing_blocks_a_swap_by_default():
    assert _app().can_swap() == []


def test_unsaved_painting_with_nowhere_to_put_it_blocks():
    """The only thing protecting painted edits today is interactive's `finally`."""
    app = _app()
    app.paint = FakePaint(empty=False, path=None)
    assert app.can_swap() and "3 voxels" in app.can_swap()[0]


def test_painting_with_a_path_does_not_block():
    app = _app()
    app.paint = FakePaint(empty=False, path="edits.npz")
    assert app.can_swap() == []


def test_unexported_skeleton_edits_block():
    app = _app()
    app.controller = FakeController(can_undo=True)
    assert app.can_swap() == ["unexported skeleton edits"]


# -------------------------------------------------------------- unloading


def test_unload_disposes_the_controller_and_commits_the_paint():
    app = _app()
    app.controller = FakeController()
    app.paint = FakePaint(empty=False, path="e.npz")
    session = app.session

    app.unload()
    assert app.controller is None and app.paint is None
    assert session.closed
    assert app.builder is None and app.session is None


def test_unload_puts_the_picker_into_the_empty_state():
    app = _app()
    app.unload()
    assert app.picker.datasets[-1]["graph"] is None


def test_unload_closes_the_napari_viewer_rather_than_clearing_it():
    """viewer2d hangs its per-viewer state off the Qt window deliberately."""
    closed = []
    app = _app()
    app.viewer = SimpleNamespace(
        close=lambda: closed.append(1),
        window=SimpleNamespace(_qt_window=SimpleNamespace(isVisible=lambda: True)),
    )
    app.unload()
    assert closed == [1] and app.viewer is None


def test_a_controller_that_fails_to_dispose_does_not_block_the_load():
    class Broken(FakeController):
        def dispose(self):
            raise RuntimeError("vtk said no")

    app = _app()
    seen = []
    app.on_status = seen.append
    app.controller = Broken()
    app.unload()
    assert app.controller is None
    assert any("did not dispose" in m for m in seen)


# ------------------------------------------------------------------ loading


def test_load_replaces_the_session_and_repopulates_the_picker():
    app = _app()
    second = FakeSession("two")
    assert app.load(session=second)
    assert app.session is second
    assert app.builder.name == "builder-two"
    assert app.picker.datasets[-1]["graph"] is second.graph


def test_a_failed_load_leaves_the_app_empty_rather_than_half_loaded():
    app = _app()
    seen = []
    app.on_status = seen.append

    class Boom(FakeSession):
        def builder(self):
            raise RuntimeError("lattice header is not what it says")

    assert not app.load(session=Boom())
    assert app.session is None and app.builder is None
    assert any("could not load" in m for m in seen)


def test_load_reports_through_the_status_slot():
    app = _app()
    seen = []
    app.on_status = seen.append
    app.load(session=FakeSession("two"))
    assert seen and "g.am" in seen[-1]


def test_the_dataset_callback_fires_on_load_and_unload():
    app = _app()
    ticks = []
    app.on_dataset = lambda _a: ticks.append(1)
    app.load(session=FakeSession("two"))
    app.unload()
    assert len(ticks) >= 2


# ---------------------------------------------------------- graph-only swap


def test_reload_graph_keeps_the_session_and_only_swaps_the_graph():
    """The common case in a repair chain: 0.4 s rather than 3 s."""
    app = _app()
    session = app.session
    assert app.reload_graph("step2.am")
    assert app.session is session, "the lattice and the frame must survive"
    assert app.picker.datasets[-1] == {"graph": session.graph, "cands": []}
    assert app.args.graph == ["step2.am"], "a list now: the field takes several"


def test_reload_graph_drops_the_edit_controller():
    """Its EditableGraph and its 88 s SDF session both belong to the old graph."""
    app = _app()
    controller = FakeController()
    app.controller = controller
    app.reload_graph("step2.am")
    assert controller.disposed and app.controller is None


def test_reload_graph_refreshes_the_builder():
    app = _app()
    first = app.builder
    app.reload_graph("step2.am")
    assert app.builder is not first


def test_reload_graph_reports_a_bad_path_instead_of_raising():
    class Broken(FakeSession):
        def reload_graph(self, path):
            raise ValueError("not an ASCII .am")

    app = _app(session=Broken())
    seen = []
    app.on_status = seen.append
    assert not app.reload_graph("binary.am")
    assert any("not an ASCII" in m for m in seen)


def test_reload_surface_only_touches_the_mesh():
    app = _app()
    app.reload_surface("new.stl")
    assert app.picker.datasets[-1] == {"mesh": app.session.mesh}
    assert app.args.surface == "new.stl"


# ------------------------------------------------------------- candidates


def test_new_candidates_reach_the_3d_view():
    """Today a session started with --no-candidates never gets them at all."""
    app = _app()
    app.set_candidates([SimpleNamespace(id=1), SimpleNamespace(id=2)])
    assert len(app.picker.datasets[-1]["cands"]) == 2


# ----------------------------------------------------------------- picking


def test_open_slices_does_nothing_without_a_dataset():
    app = ViewerApp(_args(), session=None)
    app.picker = FakePicker()
    app.open_slices([0.0, 0.0, 0.0])  # must not raise
    assert app.picker.shapes == []


def test_the_slice_callback_reaches_the_picker_deferred():
    app = _app()
    app._on_slice(2727)
    assert app.picker.slices == [2727]


# ---------------------------------------------------------------- shutdown


def test_finish_commits_painting():
    app = _app()
    app.paint = FakePaint(empty=False, path="e.npz")
    app.controller = FakeController()
    app.finish()
    assert app.paint.committed and app.paint.saved
    assert app.controller.detached


def test_finish_warns_when_there_is_nowhere_to_save(capsys):
    app = _app()
    app.paint = FakePaint(empty=False, path=None)
    app.finish()
    assert "never saved" in capsys.readouterr().out


# --- starting with no dataset at all -----------------------------------------------
#
# `python -m hipct_seg_debug` with no `--raw` / `--graph` / `--seg` opens the window
# empty and waits for the Data tab, rather than refusing to start. Everything below
# checks the routing in `main`, not the loading, so no Qt window is created.


def _bare_args(**over):
    """A namespace with the flags `main` reads, all off unless overridden."""
    base = dict(raw=None, graph=None, seg=None, surface=None, cache=None,
                validate_only=False, selftest=False, no_strict=False,
                no_candidates=False, goto_um=None, goto_candidate=None,
                goto_slice=None, goto_row=None, goto_col=None)
    base.update(over)
    return argparse.Namespace(**base)


def test_a_bare_launch_reaches_the_viewer_with_no_session(monkeypatch):
    from hipct_seg_debug import main as main_mod

    seen = {}

    def fake_interactive(session, cands, args):
        seen["session"] = session
        seen["cands"] = cands
        return 0

    monkeypatch.setattr(main_mod, "interactive", fake_interactive)
    monkeypatch.setattr(main_mod, "build_parser",
                        lambda: SimpleNamespace(parse_args=lambda argv: _bare_args()))

    assert main_mod.main([]) == 0
    assert seen["session"] is None, "the window must open with nothing loaded"
    assert seen["cands"] == [], "no dataset means no candidates to walk"


def test_naming_nothing_is_still_fatal_for_the_modes_that_read_data(monkeypatch):
    import pytest

    from hipct_seg_debug import main as main_mod

    monkeypatch.setattr(main_mod, "interactive",
                        lambda *a, **k: pytest.fail("should not reach the viewer"))
    for flag, value in (("validate_only", True), ("selftest", True),
                        ("goto_slice", 2596), ("goto_candidate", 1)):
        monkeypatch.setattr(
            main_mod, "build_parser",
            lambda v=value, f=flag: SimpleNamespace(
                parse_args=lambda argv: _bare_args(**{f: v})))
        with pytest.raises(SystemExit) as excinfo:
            main_mod.main([])
        assert "needs a dataset" in str(excinfo.value), flag


def test_a_path_that_does_not_load_is_still_fatal(monkeypatch):
    import pytest

    from hipct_seg_debug import main as main_mod

    def explode(args):
        raise main_mod.InputError("the lattice header is not Amira")

    monkeypatch.setattr(main_mod, "Session", explode)
    monkeypatch.setattr(main_mod, "interactive",
                        lambda *a, **k: pytest.fail("should not reach the viewer"))
    monkeypatch.setattr(main_mod, "build_parser",
                        lambda: SimpleNamespace(
                            parse_args=lambda argv: _bare_args(raw="r", graph="g", seg="s")))

    with pytest.raises(SystemExit) as excinfo:
        main_mod.main([])
    assert "not Amira" in str(excinfo.value)


# --- the optional surface ----------------------------------------------------------


def _stub_session_loaders(monkeypatch, tmp_path):
    """Everything `Session.__init__` reads before it reaches the surface.

    Stubbed rather than fixtured because a real session wants a raw stack, an Amira
    graph and a compressed lattice; what is under test is three lines of branching at
    the end of the constructor, not the loaders.
    """
    from hipct_seg_debug import main as main_mod

    stack = SimpleNamespace(n_slices=4, n_rows=8, n_cols=8, dtype="uint16",
                            shape=(4, 8, 8), nominal_voxel_um=1.0)
    graph = SimpleNamespace(n_vertex=3, n_edge=2, n_point=9)
    info = SimpleNamespace(dims=(4, 8, 8), fields={"Labels": 0}, materials=(),
                           regions=())
    # `corrected` and `stl_scale`: the loader asks the frame whether the units it was
    # given disagreed with the file, and takes the surface scale from it rather than
    # from the argument, so a stub frame has to answer both.
    frame_obj = SimpleNamespace(describe=lambda: "frame", seg_dims=(4, 8, 8),
                                corrected=False, stl_scale=1.0)

    monkeypatch.setattr(main_mod.tiffstack, "TiffStack", lambda *a, **k: stack)
    monkeypatch.setattr(main_mod.amira, "read_spatial_graph", lambda *a, **k: graph)
    monkeypatch.setattr(main_mod.amira, "read_lattice_header", lambda *a, **k: info)
    monkeypatch.setattr(main_mod.frame.WorldFrame, "from_inputs",
                        staticmethod(lambda *a, **k: frame_obj))
    monkeypatch.setattr(main_mod.rle, "open_lattice", lambda *a, **k: object())

    import hipct_seg_debug.edit.lattice as lattice_mod
    monkeypatch.setattr(lattice_mod, "MaskVolume", lambda labels: object())

    return _bare_args(raw=str(tmp_path), graph="g.am", seg="s.am",
                      # Stated, because it is no longer inferred: see
                      # `test_a_load_without_a_voxel_size_is_refused`.
                      cache=str(tmp_path / "cache"), pattern="*", voxel_um=1.0,
                      stl_scale=1.0, labels_field="Labels", paint=False, edits=None,
                      probability=False, probability_field="Probability",
                      no_surface=False)


def test_a_session_loads_with_no_surface_named(monkeypatch, tmp_path, capsys):
    """`--surface` is optional, so it can be absent as well as wrong.

    Regression: `Path(None)` raised `TypeError` out of the constructor, which the Data
    tab reported as "could not load" after the raw stack, graph and lattice had all
    already been read.
    """
    from hipct_seg_debug.main import Session

    args = _stub_session_loaders(monkeypatch, tmp_path)
    args.surface = None

    session = Session(args)
    assert session.mesh is None and session.slicer is None
    assert "not given" in capsys.readouterr().out


def test_a_load_without_a_voxel_size_is_refused(monkeypatch, tmp_path):
    """It used to fall back to a folder-name guess, which is wrong silently.

    The number is only ever wrong by a few percent, which is precisely the problem:
    the session stays self-consistent and every length in it carries the error.
    """
    from hipct_seg_debug.main import InputError, Session

    args = _stub_session_loaders(monkeypatch, tmp_path)
    args.voxel_um = None

    with pytest.raises(InputError) as caught:
        Session(args)
    message = str(caught.value)
    assert "--voxel-um" in message
    # The detections are still offered -- as suggestions to recognise, not defaults.
    assert "suggest" in message and "1.0000" in message


def test_a_surface_that_does_not_exist_is_still_only_a_warning(monkeypatch, tmp_path,
                                                               capsys):
    from hipct_seg_debug.main import Session

    args = _stub_session_loaders(monkeypatch, tmp_path)
    args.surface = str(tmp_path / "absent.stl")

    session = Session(args)
    assert session.mesh is None
    assert "not found" in capsys.readouterr().out
