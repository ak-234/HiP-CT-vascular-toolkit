"""Saving the 3D view as a figure: what is dropped, what is kept, and in what form.

Run against `pv.Plotter(off_screen=True)` through `plotter_factory`, the same trick
`test_viewer3d_swap.py` uses -- there is no window and no Qt loop, and GL2PS writes
from the off-screen render window exactly as it does from a real one.

The point of the feature is a *split*, so the tests are about the split rather than
about the writing: the keybinding block and the pick readout are controls and must
not reach the file, while the legend box and the colour bar are the key to what is
drawn and must. Asserting on the SVG's own text rather than on visibility flags is
deliberate -- a flag flipped at the right moment still proves nothing about what
GL2PS captured, and the colour-bar title is the one string that tells a reader
whether the tree in front of them is banded by Strahler order or ramped by radius.
"""

from __future__ import annotations

import re

import numpy as np
import pytest

pv = pytest.importorskip("pyvista")

from hipct_seg_debug.viewer3d import Picker3D  # noqa: E402

from .test_viewer3d_swap import FakeGraph, _candidate  # noqa: E402


def _picker(strahler=None, cands=(), **kw):
    graph = FakeGraph()
    if strahler is not None:
        graph.edge_attrs = {"strahler": np.array([float(strahler)])}
    return Picker3D(graph=graph, cands=list(cands),
                    plotter_factory=lambda title: pv.Plotter(off_screen=True), **kw)


def _texts(path) -> list[str]:
    """Every vector string in the SVG. The raster half is one embedded image."""
    return re.findall(r"<text[^>]*>([^<]*)<", path.read_text(encoding="utf-8"))


# ------------------------------------------------------- what does not go in


def test_the_keybindings_and_the_pick_readout_are_left_out(tmp_path):
    """The two control overlays: they drive the window, they do not describe it."""
    picker = _picker()
    picker.build()
    picker._set_pick(np.array([500.0, 0.0, 0.0]))

    out = tmp_path / "fig.svg"
    picker.save_figure(out)
    body = out.read_text(encoding="utf-8")

    assert "double-click" not in body, "the keybinding block reached the figure"
    assert "pick (" not in body, "the pick readout reached the figure"


def test_the_control_overlays_come_back_afterwards(tmp_path):
    """Hidden for the write only -- the window is still being used after it."""
    picker = _picker()
    picker.build()
    picker.save_figure(tmp_path / "fig.svg")

    assert picker._instructions_actor.GetVisibility()
    assert picker._label.GetVisibility()


def test_they_are_hidden_before_the_write_rather_than_after_it(tmp_path):
    """The raster half is captured from the render GL2PS drives, so order matters."""
    picker = _picker()
    p = picker.build()
    seen = {}
    real = p.save_graphic

    def spy(filename, *args, **kwargs):
        seen["instructions"] = bool(picker._instructions_actor.GetVisibility())
        seen["status"] = bool(picker._label.GetVisibility())
        return real(filename, *args, **kwargs)

    p.save_graphic = spy
    picker.save_figure(tmp_path / "fig.svg")
    assert seen == {"instructions": False, "status": False}


# ----------------------------------------------------------- what does go in


def test_the_strahler_bar_goes_in_as_vector_text(tmp_path):
    """The whole reason to keep any overlay: which band is which order."""
    picker = _picker(strahler=3)
    picker.build()
    picker.set_color_by("strahler")

    out = tmp_path / "fig.svg"
    picker.save_figure(out)
    texts = _texts(out)

    assert "Strahler order" in texts, "the colour bar's title is missing"
    assert "3" in texts, "the per-order annotation is missing"


def test_the_radius_bar_goes_in_when_that_is_the_mode(tmp_path):
    picker = _picker()
    picker.build()

    out = tmp_path / "fig.svg"
    picker.save_figure(out)
    assert "radius (um)" in _texts(out)


def test_the_legend_box_goes_in(tmp_path):
    picker = _picker(cands=[_candidate("premature_end", 10.0)])
    picker.build()

    out = tmp_path / "fig.svg"
    picker.save_figure(out)
    assert any(t.startswith("premature_end") for t in _texts(out))


def test_a_legend_switched_off_stays_off(tmp_path):
    """The export is of the view you set up, not of a different one."""
    picker = _picker(cands=[_candidate("premature_end", 10.0)])
    picker.build()
    picker.set_legend_visible(False)

    out = tmp_path / "fig.svg"
    picker.save_figure(out)
    texts = _texts(out)
    assert not any(t.startswith("premature_end") for t in texts)
    assert "radius (um)" in texts, "hiding the legend must not take the bar with it"


# ------------------------------------------------------------------ the path


def test_a_path_with_no_suffix_becomes_an_svg(tmp_path):
    picker = _picker()
    picker.build()

    written = picker.save_figure(tmp_path / "fig")
    assert written.endswith("fig.svg")
    assert (tmp_path / "fig.svg").exists()


def test_the_written_path_is_returned(tmp_path):
    picker = _picker()
    picker.build()

    out = tmp_path / "fig.pdf"
    assert picker.save_figure(out) == str(out)
    assert out.exists()


def test_a_raster_suffix_is_refused_by_name(tmp_path):
    """`.png` is the plausible mistake, and GL2PS's own error names no alternative."""
    picker = _picker()
    picker.build()

    with pytest.raises(ValueError, match=r"\.svg"):
        picker.save_figure(tmp_path / "fig.png")


def test_saving_without_a_window_says_so(tmp_path):
    picker = _picker()  # never built

    with pytest.raises(RuntimeError, match="not open"):
        picker.save_figure(tmp_path / "fig.svg")
