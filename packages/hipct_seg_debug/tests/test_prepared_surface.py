import numpy as np
import pytest

from hipct_seg_debug.edit.prepared_surface import prepared_field, reconstruct, expected_topology
from .conftest_geometry import graph_from


def star(degree):
    theta = np.arange(degree)*2*np.pi/degree
    direction = np.c_[np.cos(theta), np.sin(theta), .2*(-1.)**np.arange(degree)]
    direction /= np.linalg.norm(direction, axis=1)[:, None]
    graph = graph_from(np.vstack([np.zeros(3), direction*1000]),
                       [(0, i+1, 12, 40.+10*i) for i in range(degree)])
    return graph, direction


@pytest.mark.parametrize('degree', [3, 4, 5, 6])
def test_all_incident_branches_share_one_local_patch_with_distinct_radii(degree):
    graph, direction = star(degree)
    before = dict(graph.points)
    field = prepared_field(graph)
    assert graph.points == before
    assert len(field.junctions) == 1
    assert expected_topology(graph) == (1, 0)
    # Branch surface away from the junction support remains at its own radius.
    for sid in graph.segment_ids():
        normal = np.cross(direction[sid], [0, 0, 1.])
        normal /= np.linalg.norm(normal)
        radius = graph.radii(sid)[0]/1000.
        point = direction[sid]*.65+normal*radius
        values = field.evaluate(np.array([point]))
        assert float(values[0][0]) == pytest.approx(0., abs=1e-9)


def test_actual_prepared_mesh_retains_circular_branch_radius():
    graph = graph_from([(0, 0, 0), (1000, 0, 0)], [(0, 1, 12, 100.)])
    before = dict(graph.points)
    mesh, report = reconstruct(graph, cells_across_diameter=6, maximum_cells=200_000)
    assert mesh.n_cells > 0
    assert graph.points == before
    assert report['geometry_rewritten'] is False and report['radii_rewritten'] is False
    assert report['mesh_validation']['connected_components'] == 1
    assert all(row['missed_rays'] == 0 for row in report['branch_radius_checks'])
    assert all(row['max_absolute_relative_error'] < report['relative_radius_tolerance']
               for row in report['branch_radius_checks'])
    assert report['mesh_validation']['valid']


@pytest.mark.parametrize('degree', [3, 6])
def test_actual_junction_surface_has_no_extra_components_handles_or_intersections(degree):
    graph, _ = star(degree)
    mesh, report = reconstruct(graph, cells_across_diameter=6, maximum_cells=750_000)
    assert mesh.n_cells > 0
    assert report['status'] == 'validated_against_graph'
    assert report['mesh_validation']['self_intersections'] == 0
    assert report['mesh_validation']['genus'] == 0
