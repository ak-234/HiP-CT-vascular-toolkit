"""The reformat's sampling planes, drawn in the 3D window.

Run against `pv.Plotter(off_screen=True)` through `plotter_factory`, the same trick
`test_viewer3d_swap.py` uses, so there is no window and no Qt loop.

Two things are worth pinning here beyond "an actor appeared".

**The texture orientation.** ``slice_texture_array`` flips its input, because
``pv.Texture`` puts array row 0 at the *high* end of the second texture axis while this
frame puts row 0 at the low end. Getting that wrong mirrors the cross-section about the
vessel axis, which looks entirely plausible and is wrong -- so
``test_the_quad_corners_put_image_row_zero_where_it_was_sampled`` checks the corner
order against where :func:`reformat.plane_points` actually took row 0 from, rather than
against itself.

**Nothing reformat-shaped may be pickable.** A stack of quads lying across the vessel
would swallow every double-click aimed at the centreline behind it, which is the whole
gesture the panel is built on.
"""

from __future__ import annotations

import numpy as np
import pytest

pv = pytest.importorskip("pyvista")

from hipct_seg_debug import reformat as rf  # noqa: E402
from hipct_seg_debug.viewer3d import (  # noqa: E402
    LAYERS,
    REFORMAT_MAX_TEXTURED,
    Picker3D,
)

from .conftest_geometry import graph_from  # noqa: E402
from .test_reformat import FakeStack, tilted_cylinder, unit_frame  # noqa: E402


def _stack(n_points=40, size_px=21):
    """A short reformat through a tilted cylinder, with real sampled images."""
    shape = (70, 70, 70)
    axis = np.array([1.0, 0.6, 0.45])
    direction = axis / np.linalg.norm(axis)
    labels = tilted_cylinder(shape, axis, 7.0)
    volume = (labels * 500 + 100).astype(np.uint16)
    ends = np.array([35.0, 35.0, 35.0]) + np.array([-18.0, 18.0])[:, None] * direction
    graph = graph_from([tuple(ends[0]), tuple(ends[1])], [(0, 1, n_points, 7.0)])
    return rf.build(
        graph, unit_frame(shape), FakeStack(volume), [0], labels=labels,
        mode="fixed", radii_k=3.0, size_px=size_px, step_um=1.0,
    )


def _picker():
    return Picker3D(plotter_factory=lambda title: pv.Plotter(off_screen=True))


def _drawn(stack, limit=REFORMAT_MAX_TEXTURED):
    corners = rf.plane_corners(stack)
    picks = rf.texture_indices(len(corners), limit)
    sections = [(corners[i], stack.raw[i]) for i in picks]
    middle = len(corners) // 2
    return corners, sections, (corners[middle], stack.raw[middle])


# --------------------------------------------------------------------------- #
# Geometry
# --------------------------------------------------------------------------- #


def test_the_quad_corners_put_image_row_zero_where_it_was_sampled():
    """Checked against `plane_points`, not against `plane_corners` restated.

    ``plane_points`` lays the image out with axis 1 (rows) along ``+binormal`` and axis
    2 (columns) along ``+normal``, both running from ``-half`` to ``+half``. So pixel
    ``(0, 0)`` was sampled at ``centre - half*normal - half*binormal``, and that is the
    corner the texture convention sends texture coordinate ``(0, 0)`` to.
    """
    stack = _stack()
    geom, line = stack.geometry, stack.centreline
    i = len(line.coords_um) // 2

    sampled = rf.plane_points(line, geom, i, i + 1)[0]
    corners = rf.plane_corners(stack, [i])[0]

    np.testing.assert_allclose(corners[0], sampled[0, 0], atol=1e-9)   # (0, 0)
    np.testing.assert_allclose(corners[1], sampled[0, -1], atol=1e-9)  # (1, 0)
    np.testing.assert_allclose(corners[2], sampled[-1, -1], atol=1e-9)  # (1, 1)
    np.testing.assert_allclose(corners[3], sampled[-1, 0], atol=1e-9)  # (0, 1)


def test_the_corners_lie_in_the_plane_they_belong_to():
    stack = _stack()
    corners = rf.plane_corners(stack)
    for i, quad in enumerate(corners):
        along = (quad - stack.centreline.coords_um[i]) @ stack.centreline.tangents[i]
        assert np.abs(along).max() < 1e-9


def test_texture_indices_span_the_run_and_respect_the_budget():
    assert rf.texture_indices(5, 24).tolist() == [0, 1, 2, 3, 4]
    picks = rf.texture_indices(2700, 24)
    assert len(picks) <= 24
    assert picks[0] == 0 and picks[-1] == 2699
    assert rf.texture_indices(0, 24).tolist() == []


# --------------------------------------------------------------------------- #
# Drawing
# --------------------------------------------------------------------------- #


def test_showing_a_stack_registers_all_three_layers():
    stack = _stack()
    picker = _picker()
    picker.build()
    picker.show_reformat_stack(*_drawn(stack))

    assert len(picker.layer_actors("reformat_frames")) == 1
    assert len(picker.layer_actors("reformat_planes")) == len(
        rf.texture_indices(stack.n_planes, REFORMAT_MAX_TEXTURED)
    )
    # The current section is the textured quad plus its outline: at a glancing angle
    # the quad is nearly edge-on, which is when you most need to see where it is.
    assert len(picker.layer_actors("reformat_slice")) == 2


def test_a_long_run_does_not_draw_a_texture_per_plane():
    """Each textured quad is its own actor and its own upload."""
    stack = _stack(n_points=40)
    picker = _picker()
    picker.build()
    picker.show_reformat_stack(*_drawn(stack, limit=6))
    assert len(picker.layer_actors("reformat_planes")) <= 6
    # ...while the frames stay one actor however many planes there are.
    assert len(picker.layer_actors("reformat_frames")) == 1


def test_nothing_reformat_shaped_is_pickable():
    """A quad across the vessel would swallow the picks the panel is built on."""
    stack = _stack()
    picker = _picker()
    picker.build()
    picker.show_reformat(  # the selected-run polyline
        [stack.centreline.coords_um]
    )
    picker.show_reformat_stack(*_drawn(stack))

    for key in ("reformat_path", "reformat_frames", "reformat_slice",
                "reformat_planes"):
        for actor in picker.layer_actors(key):
            assert actor.GetPickable() == 0, f"{key} is pickable"


def test_moving_the_current_section_leaves_the_rest_alone():
    """The slider moves one actor; rebuilding the stack at slider rate would crawl."""
    stack = _stack()
    picker = _picker()
    picker.build()
    picker.show_reformat_stack(*_drawn(stack))
    frames = picker.layer_actors("reformat_frames")
    planes = picker.layer_actors("reformat_planes")

    corners = rf.plane_corners(stack, [3])[0]
    picker.set_reformat_section(corners, stack.raw[3])

    assert picker.layer_actors("reformat_frames") == frames
    assert picker.layer_actors("reformat_planes") == planes
    assert len(picker.layer_actors("reformat_slice")) == 2


def test_clear_reformat_takes_every_layer_down():
    stack = _stack()
    picker = _picker()
    picker.build()
    picker.show_reformat([stack.centreline.coords_um])
    picker.show_reformat_stack(*_drawn(stack))
    picker.clear_reformat()

    for key in ("reformat_path", "reformat_frames", "reformat_slice",
                "reformat_planes"):
        assert picker.layer_actors(key) == []


def test_showing_twice_does_not_leak_actors():
    stack = _stack()
    picker = _picker()
    p = picker.build()
    picker.show_reformat_stack(*_drawn(stack))
    before = len(p.renderer.actors)
    picker.show_reformat_stack(*_drawn(stack))
    assert len(p.renderer.actors) == before


def test_a_dataset_swap_drops_the_reformat():
    """Sampled planes are images cut out of the dataset that is going away."""
    stack = _stack()
    picker = _picker()
    picker.build()
    picker.show_reformat_stack(*_drawn(stack))
    picker.set_dataset(graph=None, mesh=None, cands=[], frame=None,
                       stack=None, labels=None, mask=None)
    for key in ("reformat_frames", "reformat_slice", "reformat_planes"):
        assert picker.layer_actors(key) == []


def test_showing_nothing_is_not_an_error():
    picker = _picker()
    picker.build()
    picker.show_reformat_stack(None, (), None)
    assert picker.layer_actors("reformat_frames") == []
    assert picker.layer_actors("reformat_slice") == []


def test_the_new_layers_have_panel_rows():
    keys = {key for key, _label, _opacity, _visible in LAYERS}
    assert {"reformat_frames", "reformat_slice", "reformat_planes"} <= keys
    # The stack of textures is the one thing here that costs anything, so it is the
    # one thing that starts off.
    off = {key for key, _l, _o, visible in LAYERS if not visible}
    assert "reformat_planes" in off
    assert "reformat_frames" not in off


# --------------------------------------------------------------------------- #
# The wiring from `ViewerApp`
# --------------------------------------------------------------------------- #


class RecordingPicker:
    """The `Picker3D` surface `ViewerApp` touches for a reformat."""

    def __init__(self):
        self.stacks = []
        self.sections = []
        self.slices = []

    def show_reformat_stack(self, corners, sections, current):
        self.stacks.append((corners, list(sections), current))

    def set_reformat_section(self, corners, image):
        self.sections.append((corners, image))

    def set_current_slice(self, z, defer=False):
        self.slices.append(int(z))


def _app(stack):
    from types import SimpleNamespace

    from hipct_seg_debug.app import ViewerApp

    app = ViewerApp(SimpleNamespace(), session=None, picker=RecordingPicker())
    app.session = SimpleNamespace(frame=unit_frame((70, 70, 70)))
    app.reformat = stack
    return app


def test_a_finished_build_draws_frames_and_one_section_but_no_texture_stack():
    """The default is cheap: no image is uploaded per plane unless asked for.

    Frames are one actor and carry no images at all; the single current section is the
    one the reformat window is showing and follows its slider. Building a texture for
    every plane on top of that -- for a layer that starts hidden -- is work nobody
    asked for.
    """
    stack = _stack()
    app = _app(stack)
    app._show_reformat_in_3d(stack)

    corners, sections, current = app.picker.stacks[-1]
    assert len(corners) == stack.n_planes
    assert sections == [], "the texture stack must be opt-in"
    assert current is not None


def test_the_texture_stack_is_built_only_when_asked_for():
    stack = _stack()
    app = _app(stack)
    app._show_reformat_in_3d(stack, sections=True)

    _corners, sections, _current = app.picker.stacks[-1]
    assert 0 < len(sections) <= REFORMAT_MAX_TEXTURED


def test_scrolling_the_reformat_moves_both_3d_followers():
    """The axial image plane goes to the raw slice; the section quad walks the vessel."""
    stack = _stack()
    app = _app(stack)
    app._on_reformat_plane(7)

    assert app.picker.slices, "the axial image plane did not follow"
    assert app.picker.sections, "the current section did not follow"
    np.testing.assert_allclose(
        app.picker.sections[-1][0], rf.plane_corners(stack, [7])[0], atol=1e-9
    )


def test_an_out_of_range_plane_index_is_ignored():
    stack = _stack()
    app = _app(stack)
    app._on_reformat_plane(stack.n_planes + 5)
    assert not app.picker.sections


def test_a_picker_without_the_reformat_methods_is_tolerated():
    """The reformat layers are optional; the slice-following contract is not.

    ``set_current_slice`` is the surface `ViewerApp` has always required of a picker
    (`_on_slice` calls it unguarded), so the fake keeps it. What must degrade gracefully
    is the reformat-specific pair, because a picker built before these existed -- or a
    bare test harness -- has neither.
    """
    from types import SimpleNamespace

    from hipct_seg_debug.app import ViewerApp

    stack = _stack()
    picker = SimpleNamespace(set_current_slice=lambda z, defer=False: None)
    app = ViewerApp(SimpleNamespace(), session=None, picker=picker)
    app.session = SimpleNamespace(frame=unit_frame((70, 70, 70)))
    app.reformat = stack
    app._show_reformat_in_3d(stack)  # must not raise
    app._on_reformat_plane(3)
