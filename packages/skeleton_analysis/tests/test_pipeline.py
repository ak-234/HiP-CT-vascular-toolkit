"""End-to-end run_ordering pipeline tests."""

import numpy as np

from skeleton_analysis.io.amira import read_amira
from skeleton_analysis.ordering.pipeline import root_candidates, run_ordering


def test_run_ordering_on_real_file(test_am_path, tmp_path):
    out = tmp_path / "ordered.am"
    # Test.am has a single out-degree-0 root, so root_id can be auto-detected.
    result = run_ordering(test_am_path, out)

    g = result.graph
    assert "strahler" in g.edge_fields
    assert "topo" in g.edge_fields
    assert result.strahler.shape == (g.n_edges,)
    assert result.topo.shape == (g.n_edges,)

    # Leaves have Strahler 1; the root carries the maximum order.
    assert result.strahler.min() >= 1
    assert result.strahler.max() >= 1
    # Generations start at 1 for edges adjacent to the root.
    assert result.topo.min() == 1

    # The written file reloads with the two new EDGE fields intact.
    g2 = read_amira(out)
    np.testing.assert_array_equal(g2.edge_fields["strahler"], result.strahler)
    np.testing.assert_array_equal(g2.edge_fields["topo"], result.topo)


def test_root_candidates(test_am_path):
    cands = root_candidates(test_am_path)
    assert len(cands) >= 1
    assert all(isinstance(c, int) for c in cands)
