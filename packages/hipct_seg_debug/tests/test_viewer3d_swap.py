"""Loading a second dataset must not leak actors, and must not keep the first one.

These run against `pv.Plotter(off_screen=True)` through `plotter_factory`, so there
is no window, no Qt loop and no dataset -- the same trick `selftest.py:500` already
uses to exercise the layer accessors plotter-less.

The leak test earns its keep because pyvista's cleanup is name-based, and two things
here are not covered by it: candidate actors are named `cand_<kind>`, so a kind the
next dataset lacks is never re-added and therefore never replaced; and `add_legend`
replaces itself only when it is called, which it is not when there are no candidates.
Both leave the previous dataset on screen.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

pv = pytest.importorskip("pyvista")

from hipct_seg_debug.viewer3d import LAYERS, Picker3D  # noqa: E402


class FakeGraph:
    """The handful of SpatialGraph attributes the 3D view actually reads."""

    def __init__(self, n=8, offset=0.0):
        self.n_point = n
        self.points = np.column_stack([
            np.linspace(0, 1000, n) + offset,
            np.zeros(n),
            np.zeros(n),
        ]).astype(float)
        self.thickness = np.full(n, 120.0)
        self.n_edge = 1
        self.n_edge_points = np.array([n])
        self.edge_offsets = np.array([0, n])
        self.connectivity = np.array([[0, 1]])
        self.vertices = self.points[[0, -1]]
        # Enough of the rest of `SpatialGraph` for `adapter.from_spatial_graph`, which
        # the interpolated layer goes through when the file carries no stored flags.
        self.path = None
        self.n_vertex = len(self.vertices)
        self.edge_attrs: dict = {}
        self.vertex_attrs: dict = {}
        self.point_attrs: dict = {}

    def edge_of_point(self):
        return np.zeros(self.n_point, dtype=int)

    def degree(self):
        return np.bincount(self.connectivity.ravel(), minlength=self.n_vertex)


def _candidate(kind, x):
    return SimpleNamespace(kind=kind, xyz=np.array([x, 0.0, 0.0]), detail="", id=0)


def _picker(**kw):
    kw.setdefault("graph", FakeGraph())
    return Picker3D(plotter_factory=lambda title: pv.Plotter(off_screen=True), **kw)


def _counts(p):
    return (
        len(p.renderer.actors),
        len(p.scalar_bars),
        len(getattr(p.renderer, "_labels", {})),
    )


# ------------------------------------------------------------------- the split


def test_build_still_returns_a_populated_plotter():
    picker = _picker()
    p = picker.build()
    assert p is picker.plotter
    assert picker._centreline_actor is not None


def test_the_instructions_actor_is_kept_now():
    """It was discarded, which made it impossible to update after build()."""
    picker = _picker()
    picker.build()
    assert picker._instructions_actor is not None


def test_the_window_survives_a_swap():
    picker = _picker()
    p = picker.build()
    picker.set_dataset(graph=FakeGraph(offset=500.0))
    assert picker.plotter is p


# -------------------------------------------------------------------- leaking


def test_repeated_swaps_do_not_accumulate_actors():
    picker = _picker(cands=[_candidate("murray_deficit", 100.0)])
    p = picker.build()
    baseline = _counts(p)

    for i in range(20):
        picker.set_dataset(graph=FakeGraph(offset=float(i)),
                           cands=[_candidate("murray_deficit", 100.0 + i)])
    assert _counts(p) == baseline


def test_a_candidate_kind_that_disappears_takes_its_actor_with_it():
    """The name-based replacement never fires for a kind the new dataset lacks."""
    picker = _picker(cands=[_candidate("murray_deficit", 100.0),
                            _candidate("premature_end", 200.0)])
    p = picker.build()
    assert len(picker._cand_actors) == 2

    picker.set_dataset(cands=[_candidate("premature_end", 200.0)])
    assert len(picker._cand_actors) == 1
    assert not any("murray_deficit" in name for name in p.renderer.actors)


def test_the_legend_goes_when_the_last_candidate_does():
    """add_legend replaces itself only when called, and it is not called for none."""
    picker = _picker(cands=[_candidate("murray_deficit", 100.0)])
    p = picker.build()
    assert p.renderer.legend is not None

    picker.set_dataset(cands=[])
    assert p.renderer.legend is None


def test_the_legend_can_be_hidden_and_shown_again():
    """Hidden rather than removed: the box outlives the toggle, only its visibility moves."""
    picker = _picker(cands=[_candidate("murray_deficit", 100.0)])
    p = picker.build()
    assert picker.legend_available()
    assert picker.legend_visible()

    picker.set_legend_visible(False)
    assert p.renderer.legend is not None
    assert not p.renderer.legend.GetVisibility()

    picker.set_legend_visible(True)
    assert p.renderer.legend.GetVisibility()


def test_a_hidden_legend_stays_hidden_across_a_swap():
    """The preference belongs to the window; the box is rebuilt per dataset."""
    picker = _picker(cands=[_candidate("murray_deficit", 100.0)])
    p = picker.build()
    picker.set_legend_visible(False)

    picker.set_dataset(graph=FakeGraph(offset=500.0),
                       cands=[_candidate("premature_end", 600.0)])
    assert picker.legend_visible() is False
    assert not p.renderer.legend.GetVisibility()


def test_no_legend_to_toggle_when_nothing_is_labelled():
    picker = _picker(cands=[])
    picker.build()
    assert not picker.legend_available()
    picker.set_legend_visible(False)  # must not raise with no box on screen


SECTION_KEYS = ("section_frames", "section_contours", "section_refused",
                "section_offsets")


def test_the_section_debug_layers_draw_and_clear():
    """The Sections tab's overlay: four layers up, four layers down."""
    picker = _picker()
    p = picker.build()
    quad = np.array([[0, 0, 0], [100, 0, 0], [100, 100, 0], [0, 100, 0]], dtype=float)

    picker.show_cross_sections(
        corners=[quad], truncated=[quad + 200.0],
        contours=[quad[[0, 1, 2, 3, 0]]], offsets=[np.array([[0, 0, 0], [10, 0, 0.0]])],
        refused=[quad[[0, 1, 2, 3, 0]] + 400.0],
    )
    for key in SECTION_KEYS:
        assert picker.layer_actors(key), key
    assert len(picker.layer_actors("section_frames")) == 2, "measured and refused"

    picker.clear_cross_sections()
    for key in SECTION_KEYS:
        assert not picker.layer_actors(key), key
    assert not any("_section" in name for name in p.renderer.actors)


def test_a_refused_boundary_is_not_on_the_measured_lumen_layer():
    """The whole point of the fourth row: one toggle hides the failures, not the
    measurements. Before the split a refused ribbon went on the layer labelled
    "measured lumen", where a slab hundreds of voxels long reads as a cross-section.
    """
    picker = _picker()
    picker.build()
    quad = np.array([[0, 0, 0], [100, 0, 0], [100, 100, 0], [0, 100, 0]], dtype=float)
    ring = quad[[0, 1, 2, 3, 0]]

    picker.show_cross_sections(contours=[ring], refused=[ring + 400.0])

    measured = picker.layer_actors("section_contours")
    refused = picker.layer_actors("section_refused")
    assert len(measured) == 1 and len(refused) == 1
    assert not set(map(id, measured)) & set(map(id, refused))

    picker.set_layer_visible("section_refused", False)
    assert picker.layer_visible("section_contours")
    assert not picker.layer_visible("section_refused")


def test_no_refused_sections_leaves_no_refused_row():
    """An empty PolyData still yields a live actor, so a run where everything was
    measured would otherwise show an available layer controlling nothing, and a legend
    entry reading "refused blob (0)"."""
    picker = _picker()
    picker.build()
    quad = np.array([[0, 0, 0], [100, 0, 0], [100, 100, 0], [0, 100, 0]], dtype=float)

    picker.show_cross_sections(corners=[quad], contours=[quad[[0, 1, 2, 3, 0]]])

    assert picker.layer_actors("section_contours")
    assert not picker.layer_actors("section_refused")
    assert not picker.layer_available("section_refused")


def test_section_actors_go_with_the_dataset_they_were_cut_from():
    picker = _picker()
    p = picker.build()
    quad = np.array([[0, 0, 0], [100, 0, 0], [100, 100, 0], [0, 100, 0]], dtype=float)
    picker.show_cross_sections(corners=[quad])

    picker.set_dataset(graph=FakeGraph(offset=500.0))
    assert not picker.layer_actors("section_frames")
    assert not any("_section" in name for name in p.renderer.actors)


def test_the_scalar_bar_does_not_multiply():
    picker = _picker()
    p = picker.build()
    for _ in range(5):
        picker.set_dataset(graph=FakeGraph())
    assert len(p.scalar_bars) == 1


# ------------------------------------------------------------ the empty state


def test_a_picker_can_be_built_with_no_dataset_at_all():
    picker = _picker(graph=None)
    picker.build()
    assert picker._centreline_actor is None
    assert "no dataset loaded" in picker._status()


def test_every_layer_key_still_resolves_with_no_dataset():
    """The no-dataset extension of selftest's test_layer_registry.

    That state is reachable now, and a panel row that raises rather than greying
    out would take the window down on a failed load.
    """
    picker = _picker(graph=None)
    picker.build()
    for key, _label, _default, _visible in LAYERS:
        assert isinstance(picker.layer_available(key), bool)
        assert isinstance(picker.layer_visible(key), bool)
        assert isinstance(picker.layer_opacity(key), float)
        assert isinstance(picker.layer_actors(key), list)


def test_unloading_is_the_same_path_as_loading_nothing():
    picker = _picker()
    picker.build()
    picker.set_dataset(graph=None, mesh=None, cands=[], labels=None, stack=None)
    assert picker._centreline_actor is None
    assert picker.graph is None
    assert "no dataset loaded" in picker._status()


def test_a_marker_radius_is_still_sane_with_no_graph():
    picker = _picker(graph=None)
    assert picker._marker_radius() > 0


def test_the_centreline_row_greys_out_with_no_dataset():
    picker = _picker(graph=None)
    picker.build()
    assert not picker.layer_available("centreline")
    picker.set_dataset(graph=FakeGraph())
    assert picker.layer_available("centreline")


# --------------------------------------------------------- radius circles


def test_radius_circles_are_available_but_lazy_and_hidden_by_default():
    picker = _picker()
    picker.build()
    assert picker.layer_available("radius_circles")
    assert not picker.layer_visible("radius_circles")
    assert picker._radius_circles_mesh is None
    assert picker._radius_circles_actor is None


def test_enabling_radius_circles_builds_one_cached_non_pickable_actor():
    picker = _picker()
    p = picker.build()
    picker.set_layer_opacity("radius_circles", 0.25)
    picker.set_layer_visible("radius_circles", True)

    mesh = picker._radius_circles_mesh
    actor = picker._radius_circles_actor
    assert mesh is not None and mesh.n_lines == picker.graph.n_point
    assert actor is not None and actor.GetVisibility()
    assert actor.GetProperty().GetOpacity() == pytest.approx(0.25)
    assert actor not in p.pickable_actors
    assert actor.GetMapper().GetScalarRange() == pytest.approx(
        picker._centreline_actor.GetMapper().GetScalarRange()
    )

    picker.set_layer_visible("radius_circles", False)
    assert not actor.GetVisibility()
    picker.set_layer_visible("radius_circles", True)
    assert picker._radius_circles_mesh is mesh
    assert picker._radius_circles_actor is actor


def test_radius_circle_cache_and_actor_are_dropped_on_dataset_swap():
    picker = _picker()
    p = picker.build()
    baseline = _counts(p)

    for i in range(3):
        picker.set_layer_visible("radius_circles", True)
        assert picker._radius_circles_actor is not None
        picker.set_dataset(graph=FakeGraph(offset=float(i)))
        assert picker._radius_circles_actor is None
        assert picker._radius_circles_mesh is None
        assert not picker.layer_visible("radius_circles")
        assert _counts(p) == baseline


# ------------------------------------------------------------- what is kept


def test_remembered_opacity_survives_a_swap():
    """`set_layer_opacity`'s contract is that remembering is the half that matters."""
    picker = _picker()
    picker.build()
    picker.set_layer_opacity("surface", 0.15)
    picker.set_dataset(graph=FakeGraph())
    assert picker.layer_opacity("surface") == pytest.approx(0.15)


def test_the_pick_is_dropped():
    picker = _picker()
    picker.build()
    picker._set_pick([100.0, 0.0, 0.0])
    assert picker.picked is not None
    picker.set_dataset(graph=FakeGraph())
    assert picker.picked is None


def test_the_toggles_are_dropped():
    """A toggle left on rebuilds its overlay from new inputs at the old slice."""
    picker = _picker()
    picker.build()
    picker._plane_on = picker._seg_on = picker._seg_all_on = True
    picker.set_dataset(graph=FakeGraph())
    assert not picker._plane_on and not picker._seg_on and not picker._seg_all_on


def test_the_whole_tree_mesh_is_dropped():
    picker = _picker()
    picker.build()
    picker._seg_all_mesh = object()
    picker.set_dataset(labels=None)
    assert picker._seg_all_mesh is None


def test_derived_state_follows_the_new_graph():
    picker = _picker(graph=FakeGraph(n=8))
    picker.build()
    picker.set_dataset(graph=FakeGraph(n=20))
    assert len(picker._edge_of_point) == 20


def test_the_candidate_positions_follow_the_new_list():
    picker = _picker(cands=[_candidate("premature_end", 1.0)])
    picker.build()
    picker.set_dataset(cands=[_candidate("premature_end", 2.0), _candidate("premature_end", 3.0)])
    assert picker._cand_xyz.shape == (2, 3)


# --------------------------------------------------------- partial updates


def test_an_unnamed_input_is_left_alone():
    """The common swap changes the graph only; the rest must not be discarded."""
    mesh = pv.Sphere()
    labels, stack, frame = object(), object(), object()
    picker = _picker(mesh_um=mesh, labels=labels, stack=stack, frame=frame)
    picker.build()

    picker.set_dataset(graph=FakeGraph(offset=1.0))
    assert picker.mesh is mesh
    assert picker.labels is labels and picker.stack is stack and picker.frame is frame


def test_passing_none_explicitly_does_clear_it():
    picker = _picker(mesh_um=pv.Sphere())
    picker.build()
    picker.set_dataset(mesh=None)
    assert picker.mesh is None and picker._surface_actor is None


def test_the_surface_comes_back_when_a_new_one_is_given():
    picker = _picker(mesh_um=None)
    picker.build()
    assert not picker.layer_available("surface")
    picker.set_dataset(mesh=pv.Sphere())
    assert picker.layer_available("surface")


# ------------------------------------------------------------- notifications


def test_the_owner_of_extra_actors_is_warned_before_teardown():
    """The edit controller holds its own references and must take them back first."""
    picker = _picker()
    p = picker.build()
    picker.set_extra_actors("edit_surface", [p.add_mesh(pv.Sphere())])

    order = []
    picker.on_dataset_changing = lambda: order.append(
        ("changing", bool(picker._extra_actors))
    )
    picker.set_dataset(graph=FakeGraph())
    assert order == [("changing", True)], "it must fire while the actors are still there"
    assert picker._extra_actors == {}


def test_the_layer_panel_is_told_to_refresh():
    picker = _picker()
    picker.build()
    ticks = []
    picker.on_layers_changed = lambda: ticks.append(1)
    picker.set_dataset(graph=FakeGraph())
    assert ticks


# ------------------------------------------------- what Avizo invented, on screen


#: Radii shaped like edge 183 of LADAF-28: a nearly flat ramp over the invented middle,
#: meeting the real vessel with a step of 35% and 53%. The *geometry* stays perfectly
#: straight throughout, anchors included, which is what the artefact actually looks like
#: -- the discontinuity is in the radius alone.
BRIDGE_RADII = np.array(
    [400.0, 396.0, 392.0, 388.0, 250.0, 249.0, 248.0, 380.0, 376.0, 372.0, 368.0, 364.0]
)


class BridgedGraph(FakeGraph):
    """A straight run whose middle carries a stored ``avizo_interpolated`` flag."""

    def __init__(self, n=12, flagged=(4, 5, 6), stored=True):
        super().__init__(n=n)
        if n == len(BRIDGE_RADII):
            self.thickness = BRIDGE_RADII.copy()
        column = np.zeros(n, dtype=np.int64)
        column[list(flagged)] = 1
        if stored:
            self.point_attrs = {"avizo_interpolated": column}


def test_the_interpolated_layer_appears_only_when_something_is_flagged():
    clean = _picker(graph=FakeGraph())
    clean.build()
    assert not clean.layer_available("interpolated")

    picker = _picker(graph=BridgedGraph())
    picker.build()
    assert picker.layer_available("interpolated")
    assert picker.layer_visible("interpolated")


def test_the_centreline_breaks_at_an_invented_span():
    """The real skeleton must stop where the real skeleton stops."""
    from hipct_seg_debug.viewer3d import centreline_polydata

    graph = BridgedGraph(n=12, flagged=(4, 5, 6))
    flagged = np.zeros(12, dtype=bool)
    flagged[[4, 5, 6]] = True

    whole = centreline_polydata(graph)
    broken = centreline_polydata(graph, flagged)

    assert whole.n_lines == 1
    assert broken.n_lines == 2  # points 0-3 and 7-11
    # The pick contract: ids index the graph's own point order, whatever the cells do.
    assert broken.n_points == whole.n_points == graph.n_point
    assert np.allclose(broken.points, graph.points)


def test_the_bridge_layer_reaches_the_vessel_it_joins():
    from hipct_seg_debug.viewer3d import interpolation_polydata

    graph = BridgedGraph(n=12, flagged=(4, 5, 6))
    flagged = np.zeros(12, dtype=bool)
    flagged[[4, 5, 6]] = True

    bridges = interpolation_polydata(graph, flagged)
    assert bridges.n_lines == 1
    # Three flagged points plus one real anchor either side.
    assert bridges.n_points == 5


def test_an_invented_radius_is_kept_out_of_the_colour_range():
    from hipct_seg_debug.viewer3d import _radius_clim

    graph = BridgedGraph(n=12, flagged=(4,))
    graph.thickness = np.full(12, 200.0)
    graph.thickness[4] = 20.0  # a calibration intercept, not a measurement

    flagged = np.zeros(12, dtype=bool)
    flagged[4] = True
    assert _radius_clim(graph) == (20.0, 200.0)
    assert _radius_clim(graph, flagged) == (200.0, 200.0)


def test_radius_circles_skip_invented_points():
    from hipct_seg_debug.viewer3d import radius_circle_polydata

    graph = BridgedGraph(n=12, flagged=(4, 5, 6))
    flagged = np.zeros(12, dtype=bool)
    flagged[[4, 5, 6]] = True

    assert radius_circle_polydata(graph).n_cells == 12
    assert radius_circle_polydata(graph, flagged=flagged).n_cells == 9


def test_the_viewer_detects_when_the_file_carries_no_flags():
    """Display-only, and it says where the answer came from."""
    picker = _picker(graph=BridgedGraph(stored=True))
    picker.build()
    assert "from the file" in picker.interpolation_note()

    # Same graph with the field stripped: the detector finds the same three points.
    picker = _picker(graph=BridgedGraph(stored=False))
    picker.build()
    note = picker.interpolation_note()
    assert "detected here" in note and "flag-interpolation" in note
    assert picker._interpolated_mask().sum() == 3


# ------------------------------------------------------- reconnection review

def _review_record(n_alternatives=1):
    """The subset of a review record that the 3D view reads."""
    return {
        "route": {"path_um": [[10.0 * i, 0.0, 0.0] for i in range(8)]},
        "alternatives": [
            {"path_um": [[10.0 * i, 20.0, 0.0] for i in range(8)]},
            {"path_um": [[10.0 * i, 40.0, 0.0] for i in range(8)]},
        ][:n_alternatives],
    }


def test_a_reviewed_route_is_drawn_with_its_alternatives():
    """Drawing only the winner would hide the question being asked."""
    picker = _picker()
    picker.build()
    picker.show_reconnect_candidate(_review_record(n_alternatives=2))

    names = set(picker._route_actors)
    assert any(n.endswith("-route") for n in names)
    assert sum(1 for n in names if "alternative" in n) == 2


def test_waypoints_are_drawn_only_when_there_are_some():
    picker = _picker()
    picker.build()

    picker.show_reconnect_candidate(_review_record())
    assert not any("waypoints" in n for n in picker._route_actors)

    picker.show_reconnect_candidate(_review_record(),
                                    waypoints=[[10.0, 10.0, 0.0]])
    assert any("waypoints" in n for n in picker._route_actors)


def test_stepping_to_the_next_candidate_replaces_the_previous_route():
    """A leftover alternative reads as a route for the candidate now on screen."""
    picker = _picker()
    p = picker.build()
    picker.show_reconnect_candidate(_review_record(n_alternatives=2))
    with_two = _counts(p)

    picker.show_reconnect_candidate(_review_record(n_alternatives=0))
    assert len(picker._route_actors) == 1
    assert _counts(p)[0] < with_two[0]


def test_review_routes_do_not_survive_a_dataset_swap():
    """They were measured on the old tree; drawn over a new one they mean nothing."""
    picker = _picker()
    p = picker.build()
    baseline = _counts(p)

    picker.show_reconnect_candidate(_review_record(n_alternatives=2),
                                    waypoints=[[1.0, 2.0, 3.0]])
    assert _counts(p)[0] > baseline[0]

    picker.set_dataset(graph=FakeGraph(offset=500.0))
    assert picker._route_actors == {}
    assert _counts(p) == baseline


def test_clearing_is_safe_with_nothing_drawn():
    picker = _picker()
    picker.build()
    picker.clear_reconnect_candidate()
    picker.clear_reconnect_candidate()
    assert picker._route_actors == {}


def test_a_review_route_is_never_pickable():
    """It lies along the centreline, and picking is how a waypoint gets placed."""
    picker = _picker()
    picker.build()
    picker.show_reconnect_candidate(_review_record())
    for actor in picker._route_actors.values():
        assert actor.GetPickable() == 0


# ------------------------------------------------------------- unsampled jumps

class JumpGraph(FakeGraph):
    """A graph whose edge carries one unsampled jump, as the file would store it."""

    def __init__(self, n=8, at=3, stored=True):
        super().__init__(n=n)
        # Put a real gap in the geometry, so the straight line is visibly long.
        self.points[at + 1:, 0] += 5000.0
        self.vertices = self.points[[0, -1]]
        if stored:
            marks = np.zeros(n, dtype=np.int64)
            marks[at] = 1
            self.point_attrs = {"unsampled_jump": marks}


def test_a_jump_breaks_the_centreline_without_dropping_its_anchors():
    """The anchors are real measured centreline; only the span between them is not."""
    from hipct_seg_debug.viewer3d import centreline_polydata

    graph = JumpGraph(n=8, at=3)
    breaks = np.zeros(8, dtype=bool)
    breaks[3] = True

    whole = centreline_polydata(graph)
    cut = centreline_polydata(graph, None, breaks)

    # `n_lines`, not `n_cells`: PolyData adds a vertex cell per point.
    assert whole.n_lines == 1
    assert cut.n_lines == 2, "the polyline was not split at the jump"
    # Every point survives -- the pick contract indexes into this array.
    assert cut.n_points == whole.n_points == 8
    covered = set()
    lines = cut.lines
    i = 0
    while i < len(lines):
        n = int(lines[i])
        covered.update(int(v) for v in lines[i + 1:i + 1 + n])
        i += 1 + n
    assert covered == set(range(8)), "a real anchor was dropped from the drawing"


def test_jump_polydata_draws_one_line_per_jump():
    from hipct_seg_debug.viewer3d import jump_polydata

    graph = JumpGraph(n=8, at=3)
    breaks = np.zeros(8, dtype=bool)
    breaks[[2, 5]] = True
    assert jump_polydata(graph, breaks).n_lines == 2
    assert jump_polydata(graph, np.zeros(8, dtype=bool)).n_lines == 0


def test_the_viewer_reads_stored_jumps_and_draws_them():
    picker = _picker(graph=JumpGraph(stored=True))
    picker.build()
    assert picker._jump_actor is not None
    assert picker._jump_break_mask().sum() == 1
    # One row drives both halves of "what Avizo invented".
    assert picker._jump_actor in picker.layer_actors("interpolated")
    assert picker.layer_available("interpolated")


def test_a_graph_with_no_stored_jumps_draws_none_and_never_detects():
    """Unlike the point signatures this one needs the 2.34 GB mask labelled.

    Opening a file must not quietly spend thirteen seconds decoding it to colour a line,
    so the absence of the field means "not checked" and is left alone.
    """
    picker = _picker(graph=JumpGraph(stored=False))
    picker.build()
    assert picker._jump_actor is None
    assert not picker._jump_break_mask().any()


def test_jump_actors_appear_and_leave_with_their_dataset():
    """Both directions. A jump actor left behind would draw the old tree's invented
    geometry over a new one, at coordinates that mean nothing there."""
    picker = _picker(graph=FakeGraph())
    p = picker.build()
    baseline = _counts(p)
    assert picker._jump_actor is None

    picker.set_dataset(graph=JumpGraph(stored=True))
    assert picker._jump_actor is not None
    assert _counts(p)[0] > baseline[0]

    picker.set_dataset(graph=FakeGraph(offset=500.0))
    assert picker._jump_actor is None
    # Recomputed for the new graph rather than left describing the old one.
    assert not picker._jump_break_mask().any()
    assert _counts(p) == baseline


# ------------------------------------------------------- colouring by Strahler


class OrderedGraph(FakeGraph):
    """Two edges of different Strahler order, under whatever the file called the field."""

    def __init__(self, n=8, field="strahler", orders=(3, 1)):
        super().__init__(n=n)
        half = n // 2
        self.n_edge = 2
        self.n_edge_points = np.array([half, n - half])
        self.edge_offsets = np.array([0, half, n])
        self.connectivity = np.array([[0, 1], [1, 2]])
        self.vertices = self.points[[0, half, -1]]
        self.n_vertex = len(self.vertices)
        self.edge_attrs = {field: np.array(orders)}

    def edge_of_point(self):
        return np.repeat(np.arange(self.n_edge), self.n_edge_points)


@pytest.mark.parametrize("field", ["strahler", "StrahlerOrder"])
def test_the_order_is_found_under_whatever_the_file_called_it(field):
    from hipct_seg_debug.viewer3d import point_strahler

    order = point_strahler(OrderedGraph(n=8, field=field))
    # An edge property, so every point of an edge carries its edge's order.
    assert list(order) == [3, 3, 3, 3, 1, 1, 1, 1]


def test_a_graph_without_orders_offers_no_strahler_mode():
    from hipct_seg_debug.viewer3d import point_strahler

    picker = _picker(graph=FakeGraph())
    picker.build()
    assert point_strahler(picker.graph) is None
    assert [key for key, _label in picker.color_modes()] == ["radius"]
    # Asking anyway must not leave the scene mapped by an array that is not there.
    picker.set_color_by("strahler")
    assert picker.color_by() == "radius"
    assert picker._centreline_actor.GetMapper().array_name == "radius_um"


def test_switching_to_strahler_restyles_both_layers_under_one_bar():
    picker = _picker(graph=OrderedGraph())
    p = picker.build()
    picker.set_layer_visible("radius_circles", True)
    baseline = _counts(p)
    assert [key for key, _label in picker.color_modes()] == ["radius", "strahler"]

    picker.set_color_by("strahler")
    assert picker.color_by() == "strahler"
    for actor in (picker._centreline_actor, picker._radius_circles_actor):
        assert actor.GetMapper().array_name == "strahler"
    # Orders 1 and 3 as three bands, with the outer half-steps that centre them.
    assert picker._centreline_actor.GetMapper().GetScalarRange() == pytest.approx((0.5, 3.5))
    # Bars are keyed by title, so a stale one would sit on screen beside the new one.
    assert list(p.scalar_bars.keys()) == ["Strahler order"]
    assert _counts(p) == baseline

    picker.set_color_by("radius")
    assert picker._centreline_actor.GetMapper().array_name == "radius_um"
    assert list(p.scalar_bars.keys()) == ["radius (um)"]
    assert _counts(p) == baseline


def test_recolouring_keeps_the_pick_contract_and_the_layer_state():
    """Both actors are rebuilt by the switch, so neither may lose what it was."""
    picker = _picker(graph=OrderedGraph())
    p = picker.build()
    picker.set_layer_opacity("radius_circles", 0.25)
    picker.set_layer_visible("radius_circles", True)
    picker.set_layer_visible("centreline", False)

    picker.set_color_by("strahler")

    assert picker._centreline_actor in p.pickable_actors
    assert picker._radius_circles_actor not in p.pickable_actors
    assert not picker.layer_visible("centreline")
    assert picker.layer_visible("radius_circles")
    assert picker._radius_circles_actor.GetVisibility()
    assert picker._radius_circles_actor.GetProperty().GetOpacity() == pytest.approx(0.25)
    # The pick contract: ids still index the graph's own point order.
    assert picker._centreline_actor.mapper.dataset.n_points == picker.graph.n_point


def test_a_swap_to_an_unordered_graph_falls_back_to_radius():
    picker = _picker(graph=OrderedGraph())
    p = picker.build()
    picker.set_color_by("strahler")

    picker.set_dataset(graph=FakeGraph(offset=500.0))
    assert picker.color_by() == "radius"
    assert picker._centreline_actor.GetMapper().array_name == "radius_um"
    assert list(p.scalar_bars.keys()) == ["radius (um)"]


# ---------------------------------------------------------------- crop preview


def _crop_lines(offset=0.0):
    return [np.array([[offset, 0.0, 0.0], [offset + 500.0, 100.0, 0.0]])]


def test_the_crop_layers_are_absent_until_something_registers_them():
    picker = _picker()
    picker.build()
    for key in ("crop_preview", "crop_vessels"):
        assert not picker.layer_available(key)
        assert picker.layer_actors(key) == []


def test_a_crop_preview_registers_both_rows_and_leaves_nothing_behind():
    picker = _picker()
    p = picker.build()
    baseline = _counts(p)

    picker.show_crop(_crop_lines(), [("LAD", "#d62728", _crop_lines(1000.0))])
    assert picker.layer_available("crop_preview")
    assert picker.layer_available("crop_vessels")
    assert _counts(p)[0] > baseline[0]

    picker.clear_crop()
    assert not picker.layer_available("crop_preview")
    assert not picker.layer_available("crop_vessels")
    # `set_extra_actors(key, None)` only unregisters; the actors have to go too, or
    # an emptied preview stays on screen with no panel row left to hide it.
    assert _counts(p) == baseline


def test_redrawing_a_crop_preview_replaces_rather_than_stacks():
    picker = _picker()
    p = picker.build()
    picker.show_crop(_crop_lines(), ())
    once = _counts(p)
    for i in range(3):
        picker.show_crop(_crop_lines(float(i)), ())
    assert _counts(p) == once


def test_crop_actors_are_never_pickable():
    """A tube along the centreline would swallow the picks that build the selection."""
    picker = _picker()
    p = picker.build()
    picker.show_crop(_crop_lines(), [("LAD", "#d62728", _crop_lines(1000.0))])

    for key in ("crop_preview", "crop_vessels"):
        for actor in picker.layer_actors(key):
            assert actor not in p.pickable_actors


def test_the_crop_preview_obeys_its_panel_row():
    picker = _picker()
    picker.build()
    picker.set_layer_opacity("crop_preview", 0.4)
    picker.show_crop(_crop_lines(), ())

    actor = picker.layer_actors("crop_preview")[0]
    assert actor.GetProperty().GetOpacity() == pytest.approx(0.4)
    picker.set_layer_visible("crop_preview", False)
    assert not actor.GetVisibility()


def test_a_dataset_swap_drops_the_crop_preview():
    """A selection is over *this* graph's segments and means nothing on the next."""
    picker = _picker()
    p = picker.build()
    baseline = _counts(p)

    picker.show_crop(_crop_lines(), [("LAD", "#d62728", _crop_lines(1000.0))])
    picker.set_dataset(graph=FakeGraph(offset=500.0))

    assert picker.layer_actors("crop_preview") == []
    assert _counts(p) == baseline


def test_the_window_builds_with_no_dataset_at_all():
    """`python -m hipct_seg_debug` with no inputs opens an empty window.

    Not the same state as a swap: `unload` reaches it from a populated picker, this
    reaches it from the constructor, where every dataset attribute is still at its
    `None` default and no actor has ever been added.
    """
    picker = Picker3D(plotter_factory=lambda title: pv.Plotter(off_screen=True))
    plotter = picker.build()

    assert plotter is not None
    assert picker.graph is None and picker.mesh is None
    # The instructions are drawn from `_build_window`, which does not depend on data,
    # so the window is genuinely usable rather than merely constructed.
    assert picker._instructions_actor is not None

    # And a dataset can then be loaded into it, which is what the Data tab does.
    picker.set_dataset(graph=FakeGraph(), mesh=None, cands=[], frame=None,
                       stack=None, labels=None, mask=None)
    assert picker.graph is not None

