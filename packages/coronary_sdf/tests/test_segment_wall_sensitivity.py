import json
from pathlib import Path

import numpy as np
import pytest

from coronary_sdf.plane_sensitivity import wall_segment_metrics, write_mesh_quality_summary


@pytest.mark.parametrize("coordinate_scale", [1.0, 0.001])
def test_segment_mean_includes_endpoints_without_area_weighting(tmp_path, coordinate_scale):
    graph = {"edges": [
        {"edge_id": 1, "points_mm": [[0, 0, 0], [10, 0, 0]]},
        {"edge_id": 2, "points_mm": [[0, 10, 0], [10, 10, 0]]},
    ]}
    x = np.linspace(0, 10, 10)
    xyz = np.vstack([np.column_stack([x, np.ones(10), np.zeros(10)]),
                     np.column_stack([x, np.full(10, 9), np.zeros(10)])])
    wss = np.r_[np.arange(10.0), np.full(10, 100.0)]
    wss[0] = np.nan
    path = tmp_path / "wall.npz"
    np.savez(path, columns=["X", "Y", "Z", "Wall Shear"],
             values=np.column_stack([xyz * coordinate_scale, wss]),
             surface_control_area=np.arange(1, 21) ** 2)
    planes = [{"plane_id": "P1", "edge_id": 1, "midpoint_radius_mm": 0.1},
              {"plane_id": "P2", "edge_id": 1, "midpoint_radius_mm": 5.0},
              {"plane_id": "P3", "edge_id": 2}]
    result = wall_segment_metrics(path, graph, planes)
    assert result["P1"] == result["P2"]
    assert result["P1"]["wss_mean_pa"] == pytest.approx(5.0)
    assert result["P1"]["wss_max_pa"] == 9.0
    assert result["P1"]["wall_segment_nodes"] == 9
    assert result["P1"]["wall_segment_invalid_wss_nodes"] == 1
    assert result["P3"]["wss_mean_pa"] == 100.0
    # A wall point cloud needs no surface-area field.
    np.savez(path, columns=["X", "Y", "Z", "Wall Shear"],
             values=np.column_stack([xyz * coordinate_scale, wss]))
    assert wall_segment_metrics(path, graph, planes) == result


def test_quality_summary_converts_angles_and_retains_counts(tmp_path):
    mesh = tmp_path / "meshes" / "global_l1"
    mesh.mkdir(parents=True)
    (mesh / "mesh_stats.json").write_text(json.dumps({"total_elements": 100, "core_elements": 60, "boundary_layer_elements": 40}))
    path = write_mesh_quality_summary(tmp_path, tmp_path, [{
        "case_id": "global_l1", "variable": "orthogonality_angle_rad",
        "minimum": 0, "volume_average": np.pi / 4, "maximum": np.pi / 2,
    }])
    import csv
    row = next(csv.DictReader(path.open()))
    assert int(row["total_elements"]) == 100
    assert float(row["orthogonality_angle_degrees_maximum"]) == pytest.approx(90)

