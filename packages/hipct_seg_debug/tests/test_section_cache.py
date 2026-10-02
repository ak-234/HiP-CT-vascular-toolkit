import copy

import numpy as np

from hipct_seg_debug.edit.section_cache import SectionObservationCache
from .conftest_geometry import graph_from, make_frame


def test_cache_tracks_nearby_tubes_including_incoming_and_large_radius_branches():
    graph = graph_from([(100, 100, 100), (200, 100, 100),
                        (100, 200, 100), (200, 200, 100),
                        (10000, 10000, 100), (10100, 10000, 100)],
                       [(0, 1, 5, 20.), (2, 3, 5, 20.), (4, 5, 5, 20.)])
    cache = SectionObservationCache(10, 10)
    scale = np.full(5, 20.)
    observations = (graph.coords(0).copy(), np.ones(5), [0, 4], [20., 20.])
    cache.refresh(graph)
    assert cache.get(0, scale) is None
    cache.put(0, observations, {'example': 1})
    hit = cache.get(0, scale)
    hit[0][1][:] = 0
    assert np.all(cache.get(0, scale)[0][1] == 1)  # Huber weighting cannot corrupt stored observations.
    graph.set_segment_coords(2, graph.coords(2)+[10, 0, 0])
    cache.refresh(graph)
    assert cache.get(0, scale) is not None
    assert cache.get(0, scale*2) is None
    graph.set_segment_coords(1, graph.coords(1)+[0, 1, 0])
    cache.refresh(graph)
    assert cache.get(0, scale) is None
    cache.put(0, observations, {})
    graph.set_segment_radii(2, np.full(5, 20000.))
    cache.refresh(graph)
    assert cache.get(0, scale) is None
    cache.put(0, observations, {})
    graph.set_segment_radii(2, np.full(5, 20.))
    graph.set_segment_coords(2, graph.coords(0)+[0, 50, 0])
    cache.refresh(graph)
    assert cache.get(0, scale) is None


def test_refinement_cache_matches_disabled_run_and_skips_unchanged_regions(tmp_path, monkeypatch):
    from hipct_seg_debug import amira, rle, rle_write
    from hipct_seg_debug.edit import centreline_refine as cr
    shape = (12, 80, 80)
    frame = make_frame(shape)
    graph = graph_from([(50, 80, 50), (200, 80, 50), (550, 650, 50), (750, 650, 50)],
                       [(0, 1, 20, 20.), (2, 3, 20, 20.)])
    other = copy.deepcopy(graph)
    mask = np.ones(shape, dtype='uint8')
    path = tmp_path/'labels.am'
    rle_write.write_lattice(path, mask, frame.seg_bbox_um)
    header = amira.read_lattice_header(path)
    labels = rle.open_lattice(path, header.fields['Labels'], header.dims, cache_dir=tmp_path/'cache')
    calls = []
    def targets(g, sid, *args, **kwargs):
        calls.append(sid)
        x = g.coords(sid).copy()
        if sid == 0:
            return x, np.zeros(len(x)), [], []
        x[:, 1] += 5*np.sin(np.linspace(0, np.pi, len(x)))
        return x, np.ones(len(x)), list(range(len(x))), [20.]*len(x)
    monkeypatch.setattr(cr, '_targets', targets)
    cached = cr.refine(graph, frame, labels, max_half=4, max_iterations=3, reuse_sections=True)
    cached_calls = calls.count(0)
    calls.clear()
    direct = cr.refine(other, frame, labels, max_half=4, max_iterations=3, reuse_sections=False)
    assert cached_calls == 1 and calls.count(0) == 3
    for sid in graph.segment_ids():
        np.testing.assert_array_equal(graph.coords(sid), other.coords(sid))
    for a, b in zip(cached.history, direct.history):
        assert {k: v for k, v in a.items() if k != 'reused_section_segments'} == {
            k: v for k, v in b.items() if k != 'reused_section_segments'}
    assert cached.history[-1]['reused_section_segments'] >= 1
