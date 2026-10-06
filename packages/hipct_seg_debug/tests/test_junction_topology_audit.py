import copy

import numpy as np
import pytest

from hipct_seg_debug.edit.junction_topology_audit import audit
from .conftest_geometry import graph_from


@pytest.mark.parametrize('reverse', [False, True])
def test_short_unsupported_link_identifies_four_approach_complex_without_editing(reverse):
    g = graph_from([(-100, 0, 0), (0, 0, 0), (20, 0, 0), (120, 0, 0),
                    (0, 100, 0), (20, -100, 0)],
                   [(0, 1, 5, 30), (1, 2, 5, 30), (2, 3, 5, 30),
                    (1, 4, 5, 10), (2, 5, 5, 10)])
    if reverse:
        g.triple.segments.reverse()
        for seg in g.triple.segments:
            seg['node1'], seg['node2'] = seg['node2'], seg['node1']
            seg['point_ids'].reverse()
    before = copy.deepcopy(g)
    geometry = {'segments': {str(s): {'centring_final': {'accepted_sections': 0 if s == 1 else 3}}
                              for s in g.segment_ids()}}
    row = audit(g, geometry)
    assert len(row['complexes']) == 1
    c = row['complexes'][0]
    assert c['nodes'] == [1, 2] and c['internal_segments'] == [1]
    assert [p['segment'] for p in c['external_ports']] == [0, 2, 3, 4]
    assert c['incident_branch_count'] == 4
    assert not row['connectivity_modified']
    for sid in g.segment_ids():
        assert g.segment(sid) == before.segment(sid)
        np.testing.assert_array_equal(g.coords(sid), before.coords(sid))

    # Exclusive support on the connector separates the two junctions for this
    # diagnostic; missing evidence is also not equivalent to a failed audit.
    geometry['segments']['1']['centring_final']['accepted_sections'] = 2
    assert audit(g, geometry)['complexes'] == []
    geometry['segments']['1'] = {}
    assert audit(g, geometry)['complexes'] == []
    assert audit(g, geometry)['short_links'][0]['section_evidence'] == 'not_audited'


@pytest.mark.parametrize('degree', [3, 4, 5, 6])
def test_single_branching_node_retains_incident_branch_count(degree):
    angles = np.arange(degree)*2*np.pi/degree
    nodes = [(0., 0., 0.), *[(100*np.cos(a), 100*np.sin(a), 0.) for a in angles]]
    g = graph_from(nodes, [(0, k, 5, 10.) for k in range(1, degree+1)])
    row = audit(g, {}, sids=g.segment_ids())
    assert row['nodes'][0]['degree'] == degree
    assert row['nodes'][0]['max_endpoint_mismatch_um'] == 0
    if degree >= 4:
        assert row['complexes'][0]['incident_branch_count'] == degree
    else:
        assert row['complexes'] == []
