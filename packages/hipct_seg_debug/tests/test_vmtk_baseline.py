import numpy as np
import pytest

from hipct_seg_debug.edit.vmtk_baseline import extract


def test_missing_dependency_is_explicit(monkeypatch):
    import sys
    monkeypatch.setitem(sys.modules, 'vmtk', None)
    with pytest.raises(ImportError, match='VMTK comparison unavailable'):
        extract(np.ones((3, 3, 3)), np.ones(3), np.zeros(3), [[0, 0, 0]], [[1, 1, 1]])


def test_vmtk_round_tube_when_installed():
    pytest.importorskip('vmtk')
    from .conftest_geometry import cylinder
    volume = cylinder((21, 21, 61), 4, 3, 58)
    triple, detail = extract(volume, np.ones(3), np.zeros(3), [[2.5, 10, 10]], [[57.5, 10, 10]])
    assert triple.segments
    points = np.asarray(list(triple.points.values()))
    interior = points[(points[:, 0] > 10) & (points[:, 0] < 50)]
    assert len(interior) > 0
    assert np.max(np.linalg.norm(interior[:, 1:3]-10, axis=1)) < 2
    assert 'requires containment/centring review' in detail
