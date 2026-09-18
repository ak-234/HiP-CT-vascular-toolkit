import numpy as np
import pytest

from hipct_seg_debug.crosssection import _PlaneSampler, cut, stable_transverse_cut
from hipct_seg_debug.edit.adapter import Triple
from hipct_seg_debug.edit.graphmodel import EditableGraph
from hipct_seg_debug.edit.section_validation import SectionContext, SectionVerdict
from .conftest_geometry import make_frame


def branches(lines, radii):
    points, nodes, segments = {}, {}, []
    for sid, (line, radius) in enumerate(zip(lines, radii)):
        ids = list(range(len(points), len(points)+len(line)))
        points.update({pid: (*p, radius) for pid, p in zip(ids, line)})
        nodes[2*sid], nodes[2*sid+1] = (*line[0], 0), (*line[-1], 0)
        segments.append(dict(id=sid, node1=2*sid, node2=2*sid+1, point_ids=ids))
    return EditableGraph(Triple(nodes, points, segments))


@pytest.mark.parametrize("reverse", [False, True])
def test_parallel_daughter_volume_intersects_without_axis_crossing(reverse):
    shape = (64, 100, 100)
    frame = make_frame(shape)
    z, y, x = np.ogrid[:64, :100, :100]
    labels = (((y-50)**2+(z-30)**2 <= 4**2) |
              (((x-52)**2+(z-30)**2 <= 3**2) & (y >= 50) & (y <= 85))).astype('uint8')
    lines = [np.array([[10, 50, 30], [90, 50, 30]])*10.,
             np.array([[52, 50, 30], [52, 85, 30]])*10.]
    if reverse:
        lines = [line[::-1] for line in lines]
    graph = branches(lines, [40., 30.])
    sampler = _PlaneSampler(labels, frame)
    c = cut(sampler, [50, 50, 30], [1, 0, 0], 42, max_half=45)
    assert c is not None and not c.touches_border
    verdict = SectionContext(graph).validate(0, c, np.array([500., 500., 300.]),
                                             [1, 0, 0], [1, 0, 0], sampler, frame)
    assert not verdict.accepted
    assert verdict.reason == 'neighbouring_lumen_contamination'
    assert verdict.contaminants[0]['tangent_plane_angle_degrees'] == pytest.approx(0.)


def test_parallel_separate_lumen_not_rejected_even_with_overestimated_radius():
    frame = make_frame((64, 100, 100))
    z, y, x = np.ogrid[:64, :100, :100]
    labels = (((y-50)**2+(z-30)**2 <= 4**2) |
              (((x-52)**2+(z-42)**2 <= 2**2) & (y >= 30) & (y <= 85))).astype('uint8')
    graph = branches([np.array([[10, 50, 30], [90, 50, 30]])*10.,
                      np.array([[52, 30, 42], [52, 85, 42]])*10.], [40., 150.])
    sampler = _PlaneSampler(labels, frame)
    c = cut(sampler, [50, 50, 30], [1, 0, 0], 25, max_half=45)
    verdict = SectionContext(graph).validate(0, c, np.array([500., 500., 300.]),
                                             [1, 0, 0], [1, 0, 0], sampler, frame)
    assert verdict.accepted


def test_axial_continuation_has_finite_flat_end_support():
    frame = make_frame((64, 100, 100))
    z, y, x = np.ogrid[:64, :100, :100]
    labels = np.broadcast_to((y-50)**2+(z-30)**2 <= 4**2, (64, 100, 100)).astype('uint8')
    graph = branches([np.array([[10, 50, 30], [60, 50, 30]])*10.,
                      np.array([[60, 50, 30], [90, 50, 30]])*10.], [40., 200.])
    sampler = _PlaneSampler(labels, frame)
    c = cut(sampler, [50, 50, 30], [1, 0, 0], 15, max_half=25)
    verdict = SectionContext(graph).validate(0, c, np.array([500., 500., 300.]),
                                             [1, 0, 0], [1, 0, 0], sampler, frame)
    assert verdict.accepted


def test_every_alternative_plane_passes_validator():
    frame = make_frame((30, 30, 60))
    z, y, x = np.ogrid[:30, :30, :60]
    labels = np.broadcast_to((y-15)**2+(z-15)**2 <= 5**2, (30, 30, 60)).astype('uint8')
    calls, diagnostics = [], {}
    def refuse(c, origin, normal):
        calls.append(normal.copy())
        return SectionVerdict(False, 'neighbouring_lumen_contamination')
    chosen = stable_transverse_cut(_PlaneSampler(labels, frame), [30, 15, 15], [1, 0, 0],
                                  5, spacing_um=10, max_half=20, validator=refuse,
                                  diagnostics=diagnostics)
    assert chosen is None
    assert len(calls) == 9
    assert diagnostics['neighbouring_lumen_contamination'] == 9


@pytest.mark.parametrize('reverse,reorder', [(False, False), (True, False), (True, True)])
def test_owned_slab_reuses_volume_and_preserves_target_lumen(monkeypatch, reverse, reorder):
    from hipct_seg_debug.edit import radius_perimeter as rp
    frame = make_frame((40, 60, 80))
    z, y, x = np.ogrid[:40, :60, :80]
    labels = np.broadcast_to(((y-25)**2+(z-20)**2 <= 5**2) |
                             ((y-33)**2+(z-20)**2 <= 5**2), (40, 60, 80)).astype('uint8')
    lines = [np.array([[5, 25, 20], [75, 25, 20]])*10.,
             np.array([[5, 33, 20], [75, 33, 20]])*10.]
    if reverse:
        lines = [line[::-1] for line in lines]
    if reorder:
        lines = lines[::-1]
    sid = int(reorder)
    graph = branches(lines, [50., 50.])
    sampler = _PlaneSampler(labels, frame)
    original, calls = rp._resolve_owned_slab, []
    def counted(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)
    monkeypatch.setattr(rp, '_resolve_owned_slab', counted)
    chosen = stable_transverse_cut(sampler, [40, 25, 20], [1, 0, 0], 5,
        spacing_um=10, max_half=25, transverse_axis_ratio=np.inf,
        validator=SectionContext(graph).validator(sid, [1, 0, 0], sampler, frame))
    assert chosen is not None and chosen.owned
    assert len(calls) == 1
    assert 60 < chosen.cut.blob4.sum() < 95
    assert chosen.centroid_ratio < .3
    from scipy.ndimage import binary_fill_holes
    np.testing.assert_array_equal(binary_fill_holes(chosen.cut.blob4), chosen.cut.blob4)


def test_merged_incident_branch_is_not_partitioned_into_a_trusted_section(monkeypatch):
    from hipct_seg_debug.edit import radius_perimeter as rp
    from .conftest_geometry import graph_from
    frame = make_frame((40, 60, 80))
    z, y, x = np.ogrid[:40, :60, :80]
    labels = (((y-25)**2+(z-20)**2 <= 5**2) |
              (((x-40)**2+(z-20)**2 <= 4**2) & (y >= 25))).astype('uint8')
    graph = graph_from([(50, 250, 200), (400, 250, 200), (750, 250, 200), (400, 550, 200)],
                       [(0, 1, 12, 50.), (1, 2, 12, 50.), (1, 3, 12, 40.)])
    sampler = _PlaneSampler(labels, frame)
    def forbidden(*args, **kwargs):
        raise AssertionError('watershed must not author a junction boundary')
    monkeypatch.setattr(rp, '_resolve_owned_slab', forbidden)
    chosen = stable_transverse_cut(sampler, [40, 25, 20], [1, 0, 0], 5,
        spacing_um=10, max_half=35,
        validator=SectionContext(graph).validator(0, [1, 0, 0], sampler, frame))
    assert chosen is None
