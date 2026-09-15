"""Authored bifurcation tapers are recognized before SDF radius preprocessing."""

from __future__ import annotations

import numpy as np
from contextlib import contextmanager
from types import SimpleNamespace

from hipct_seg_debug.edit.adapter import Triple
from hipct_seg_debug.edit.sdfpatch import has_complete_authored_bifurcation_taper


def _tapered_triple():
    nodes = {
        0: (0.0, 0.0, 0.0, 3),
        1: (-100.0, 0.0, 0.0, 1),
        2: (100.0, -50.0, 0.0, 1),
        3: (100.0, 50.0, 0.0, 1),
    }
    points = {}
    segments = []
    pid = 0
    endpoint_ids = []
    for sid, other in enumerate((1, 2, 3)):
        ids = []
        for p in (nodes[0][:3], nodes[other][:3]):
            points[pid] = (*p, 50.0)
            ids.append(pid)
            pid += 1
        endpoint_ids.append(ids[0])
        segments.append({"id": sid, "node1": 0, "node2": other, "point_ids": ids})
    modes = {pid: 0 for pid in points}
    modes[endpoint_ids[0]] = 3
    modes[endpoint_ids[1]] = 4
    modes[endpoint_ids[2]] = 4
    return Triple(
        nodes, points, segments,
        point_attrs={"radius_resolution_mode": modes},
        point_attr_dtypes={"radius_resolution_mode": np.dtype(np.int64)},
    ), endpoint_ids


def test_complete_authored_taper_is_detected():
    triple, _endpoints = _tapered_triple()
    assert has_complete_authored_bifurcation_taper(triple)


def test_partial_taper_does_not_disable_surface_radius_repairs():
    triple, endpoints = _tapered_triple()
    del triple.point_attrs["radius_resolution_mode"][endpoints[-1]]
    assert not has_complete_authored_bifurcation_taper(triple)


def test_audited_input_fallback_still_preserves_the_graph_profile():
    triple, endpoints = _tapered_triple()
    triple.point_attrs["radius_resolution_mode"][endpoints[-1]] = 5
    assert has_complete_authored_bifurcation_taper(triple)


def test_session_uses_non_destructive_surface_profile_for_authored_taper(monkeypatch):
    from hipct_seg_debug.edit import sdfpatch

    triple, _endpoints = _tapered_triple()
    seen = []

    @contextmanager
    def fake_config(profile):
        seen.append(dict(profile))
        yield SimpleNamespace()

    monkeypatch.setattr(sdfpatch, "sdf_config", fake_config)
    monkeypatch.setattr(sdfpatch, "preprocess_graph", lambda value: value.copy())
    monkeypatch.setattr(
        sdfpatch.SdfSession, "_prepare",
        lambda self, clean: SimpleNamespace(seconds=0.0),
    )
    session = sdfpatch.SdfSession.__new__(sdfpatch.SdfSession)
    session.profile = {"DEBUG_VIS": False}
    session.quiet = True
    session.set_graph(triple)

    assert session.authored_bifurcation_taper
    assert seen[-1]["SMOOTH_SEGMENT_RADII"] is False
    assert seen[-1]["PRUNE_BIFURCATION_SHRINK"] is False
    assert seen[-1]["SMOOTH_RADIUS_TRANSITIONS"] is False
    assert seen[-1]["BIF_CARINA_ENABLE"] is False
    assert session.profile == {"DEBUG_VIS": False}, "global/base profile is not mutated"
