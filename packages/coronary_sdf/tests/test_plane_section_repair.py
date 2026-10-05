from pathlib import Path

import numpy as np
import pytest


def test_reject_elongated_oblique_section():
    import pyvista as pv
    from coronary_sdf.plane_sensitivity import validate_section
    tube = pv.Cylinder(direction=(0, 0, 1), radius=1, height=30, resolution=128).triangulate()
    with pytest.raises(ValueError, match="elongated"):
        validate_section(tube, np.zeros(3), np.array([1.0, 0, 0.1]), 1.1)
    section = validate_section(tube, np.zeros(3), np.array([0, 0, 1.0]), 1.1)
    assert section["section_equivalent_radius_mm"] == pytest.approx(1, abs=0.001)


def test_section_cache_rejects_changed_geometry(tmp_path):
    from coronary_sdf.plane_sensitivity import _cached_plane_geometry_matches, _cfx_session
    path = tmp_path / "sections.csv"
    plane = {"plane_id": "P1", "centre_mm": [0, 0, 0], "normal": [0, 0, 1], "bound_radius_mm": 1.1}
    path.with_suffix(".cse").write_text(_cfx_session(Path("old.res"), path, [plane]))
    assert _cached_plane_geometry_matches(path, [plane])
    assert not _cached_plane_geometry_matches(path, [{**plane, "normal": [0, 1, 0]}])
    assert not _cached_plane_geometry_matches(path, [{**plane, "bound_radius_mm": 2.0}])
