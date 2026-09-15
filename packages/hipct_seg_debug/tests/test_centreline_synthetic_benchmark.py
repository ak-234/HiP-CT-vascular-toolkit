import numpy as np
import pytest

from hipct_seg_debug.crosssection import _PlaneSampler
from hipct_seg_debug.edit.centreline_refine import bad_edges
from hipct_seg_debug.edit.centreline_synthetic_benchmark import fixture, score


@pytest.mark.parametrize("kind", ["round", "flat", "curved-flat"])
@pytest.mark.parametrize("step", [1, 3])
def test_phantom_has_contained_offset_curve_and_true_fixed_endpoints(kind, step):
    graph, frame, labels, amplitude = fixture(kind, step)
    x = graph.coords(0)
    assert not bad_edges(x, _PlaneSampler(labels, frame), frame).any()
    true_y = (50+amplitude*np.sin((x[:, 0]/10-5)*np.pi/110))*10
    np.testing.assert_allclose(x[[0, -1], 1], true_y[[0, -1]])
    assert score(x, amplitude)["median_error_vox"] > 2.5
    np.testing.assert_array_equal(graph.radii(0), np.full(len(x), 15.))


def test_score_matches_analytical_straight_offset_and_angle():
    x = np.c_[np.linspace(300, 900, 30), np.full(30, 530.), np.full(30, 280.)]
    result = score(x, 0)
    assert result["median_error_vox"] == pytest.approx(5.)
    assert result["p95_error_vox"] == pytest.approx(5.)
    assert result["median_tangent_deg"] == pytest.approx(0.)
