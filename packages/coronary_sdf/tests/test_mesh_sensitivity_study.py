import ast
import json
import math
from pathlib import Path

import numpy as np
import pytest

from coronary_sdf.mesh_sensitivity_study import (
    boundary_layer_floor_evident,
    build_case_plan,
    calibrated_size,
    extract_cfx_seed_contract,
    export_existing_simpleware_quality,
    feasible_targets,
    is_coarse_volume_mesh_failure,
    local_boundary_layer_thickness,
    log_log_size_fit,
    lumped_vertex_areas,
    observed_order_gci,
    parallel_transport_frames,
    parse_cfx_solver_output,
    radius_mesh_size,
    sector_indices,
    select_boundary_layer,
    select_distal_terminal,
    simpleware_width_ratio,
    weighted_quantile,
    wss_band_statistics,
    _write_cfx_boundary_audit_session,
)
from coronary_sdf.flow_fractions import (
    _select_stable_cfx_opening,
    generate_cfx_ccl,
)
from coronary_sdf.plane_sensitivity import (
    adjacent_percent,
    averaged_tangent,
    interpolate_polyline,
    plane_basis,
    point_in_polygon,
    polygon_area_centroid,
    rank_representative_candidates,
    normalized_difference_percent,
    three_grid_gci,
    validate_section,
)


def test_radius_size_and_local_boundary_layer_cap():
    assert radius_mesh_size(0.3, 6) == pytest.approx(0.1)
    assert radius_mesh_size(0.001, 6) == pytest.approx(0.02)
    assert radius_mesh_size(4.0, 6) == pytest.approx(0.4)
    assert local_boundary_layer_thickness(0.5) == pytest.approx(0.075)
    assert local_boundary_layer_thickness(2.0) == pytest.approx(0.2)


def test_plane_midpoint_interpolation_and_symmetric_tangent():
    points = np.array([[0, 0, 0], [1, 0, 0], [3, 0, 0]], float)
    point, radius = interpolate_polyline(points, np.array([1.0, 2.0, 4.0]), 1.5)
    assert np.allclose(point, [1.5, 0, 0])
    assert radius == pytest.approx(2.5)
    assert np.allclose(averaged_tangent(points, 1.5, 2.0), [1, 0, 0])


def test_plane_basis_and_polygon_centroid():
    u, v = plane_basis(np.array([0.0, 0.0, 1.0]))
    assert np.dot(u, v) == pytest.approx(0.0)
    square = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]], float)
    area, centroid = polygon_area_centroid(square)
    assert area == pytest.approx(4.0)
    assert np.allclose(centroid, 0.0)
    assert point_in_polygon(np.array([0.0, 0.0]), square)
    assert not point_in_polygon(np.array([2.0, 0.0]), square)


def test_exact_stl_loop_validation_and_neighbour_rejection():
    pv = pytest.importorskip("pyvista")
    tube = pv.Cylinder(center=(0, 0, 0), direction=(1, 0, 0), radius=1.0, height=10.0, resolution=64, capping=True).triangulate()
    section = validate_section(tube, np.zeros(3), np.array([1.0, 0.0, 0.0]), 1.10)
    assert section["area_mm2"] == pytest.approx(math.pi, rel=0.01)
    assert section["bound_radius_mm"] == pytest.approx(1.1, rel=0.01)
    neighbour = pv.Cylinder(center=(0, 1.3, 0), direction=(1, 0, 0), radius=0.2, height=10.0, resolution=48, capping=True).triangulate()
    with pytest.raises(ValueError, match="second STL branch"):
        validate_section(tube.merge(neighbour), np.zeros(3), np.array([1.0, 0.0, 0.0]), 1.10)


def test_representative_candidate_ranking_and_clearance():
    def edge(edge_id, radius, order, length):
        return {"edge_id": edge_id, "node1": edge_id, "node2": edge_id + 1, "strahler": order,
                "points_mm": [[0, 0, 0], [length, 0, 0]], "radii_mm": [radius, radius], "radius_bin": 1}
    edges = [edge(1, 0.5, 2, 10), edge(2, 0.7, 2, 2), edge(3, 0.6, 2, 12)]
    ranked = rank_representative_candidates(edges, "strahler", 2, 0.58, 2.0)
    assert [item["edge_id"] for item in ranked] == [3, 1]


def test_plane_gci_and_adjacent_percentage():
    counts = [1_500_000, 3_100_000, 7_000_000, 14_800_000]
    values = [12.0, 11.0, 10.5, 10.25]
    result = three_grid_gci(values, counts, 1.25)
    assert result["status"] == "monotonic"
    assert result["observed_order"] > 0
    assert result["gci_fine_percent"] > 0
    assert adjacent_percent(10.0, 10.5) == pytest.approx(100 * 0.5 / 10.5)
    assert normalized_difference_percent(100.0, 110.0, 1000.0) == pytest.approx(1.0)
    pressure_gci = three_grid_gci(values, counts, 1.25, normalization=100.0)
    assert pressure_gci["gci_fine_percent"] < result["gci_fine_percent"]
    exact_second_order = three_grid_gci(
        [0.0, 11.0, 10.25, 10.0625], [1, 1, 8, 64]
    )
    assert exact_second_order["observed_order"] == pytest.approx(2.0)
    oscillatory = three_grid_gci([10.0, 11.0, 10.0, 11.0], counts)
    assert oscillatory["status"].startswith("oscillatory")


def test_simpleware_width_ratio_from_consecutive_growth():
    assert simpleware_width_ratio(4, 1.2) == pytest.approx(1.0 / 1.2**3)
    assert simpleware_width_ratio(6, 1.2) == pytest.approx(1.0 / 1.2**5)
    assert simpleware_width_ratio(8, 1.2) == pytest.approx(1.0 / 1.2**7)
    assert simpleware_width_ratio(4, 1.0) == pytest.approx(1.0)
    with pytest.raises(ValueError):
        simpleware_width_ratio(4, 0.9)


def test_simpleware_console_uses_inherent_clipping_cap_exclusion():
    # The console scripts live in the package, not beside the tests.
    console_path = (
        Path(__file__).resolve().parents[1]
        / "src" / "coronary_sdf" / "simpleware_mesh_sensitivity_console.py"
    )
    tree = ast.parse(console_path.read_text(encoding="utf-8"))
    definitions = {
        node.name: len(node.args.args)
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    calls = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "_validate_inherent_cap_boundary_layer_exclusion"
    ]
    assert definitions["_validate_inherent_cap_boundary_layer_exclusion"] == 4
    assert len(calls) == 1
    assert len(calls[0].args) == definitions[
        "_validate_inherent_cap_boundary_layer_exclusion"
    ]
    method_names = {
        node.func.attr for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }
    assert "IsBoundaryLayerExternalRegionGenerationSet" in method_names
    assert "SetBoundaryLayerRegionByName" not in method_names
    assert "SetUseAdditionalMeshQualityImprovement" in method_names
    assert "SetAdditionalMeshQualityImprovementAllowOffSurface" in method_names
    assert "collect_mesh_quality" in {
        node.func.id for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    source = console_path.read_text(encoding="utf-8")
    assert '"mesh_quality_json"' in source
    assert '"mesh_quality_csv"' in source


def test_element_calibration_and_infeasible_coarse_target():
    assert calibrated_size(0.2, 8_000_000, 1_000_000) == pytest.approx(0.4)
    values = feasible_targets(2_000_000, [1_500_000, 3_200_000, 6_900_000, 15_000_000], 15_000_000)
    assert values[0] == 2_000_000 and values[-1] == 15_000_000
    assert len(values) == 4 and values == sorted(values)
    noisy = [(0.20, 1_800_000), (0.22, 1_900_000)]
    corrected = log_log_size_fit(noisy, 1_500_000)
    assert corrected > noisy[-1][0]
    assert boundary_layer_floor_evident(noisy, 1_500_000)
    assert not boundary_layer_floor_evident(
        [(0.20, 1_800_000), (0.22, 1_600_000)], 1_500_000
    )


def test_coarse_volume_mesh_failure_detection(tmp_path):
    log = tmp_path / "simpleware.log"
    log.write_text(
        "RuntimeError: Volume mesh could not be generated because of an "
        "internal error.\n",
        encoding="utf-8",
    )
    assert is_coarse_volume_mesh_failure(log)
    log.write_text("ordinary license startup failure\n", encoding="utf-8")
    assert not is_coarse_volume_mesh_failure(log)


def test_lumped_area_and_weighted_statistics():
    points = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], float)
    areas = lumped_vertex_areas(points, np.array([[0, 1, 2]]))
    assert areas.sum() == pytest.approx(0.5)
    assert np.allclose(areas, 1 / 6)
    assert weighted_quantile([1, 2, 10], [1, 1, 8], 0.5) > 5
    stats = wss_band_statistics(
        np.arange(1, 9, dtype=float), np.ones(8), np.arange(8)
    )
    assert stats["area_weighted_mean"] == pytest.approx(4.5)
    assert stats["sector_min"] == 1 and stats["sector_max"] == 8


def test_parallel_transport_sector_frame_is_orthonormal():
    points = np.column_stack([np.linspace(0, 2, 8), np.sin(np.linspace(0, 1, 8)), np.zeros(8)])
    tangent, normal, binormal = parallel_transport_frames(points)
    assert np.allclose(np.sum(tangent * normal, axis=1), 0, atol=1e-10)
    radial = np.array([normal[3], binormal[3], -normal[3], -binormal[3]])
    sectors = sector_indices(radial, normal[3], binormal[3])
    assert list(sectors) == [0, 2, 4, 6]


def test_distal_opening_is_longest_graph_geodesic():
    nodes = {
        0: (0, 0, 0, 1), 1: (1000, 0, 0, 3),
        2: (2000, 0, 0, 1), 3: (1000, 3000, 0, 1),
    }
    points = {
        0: (0, 0, 0, 500), 1: (1000, 0, 0, 500),
        2: (1000, 0, 0, 200), 3: (2000, 0, 0, 200),
        4: (1000, 0, 0, 100), 5: (1000, 3000, 0, 100),
    }
    segments = [
        {"id": 10, "node1": 0, "node2": 1, "point_ids": [0, 1]},
        {"id": 11, "node1": 1, "node2": 2, "point_ids": [2, 3]},
        {"id": 12, "node1": 1, "node2": 3, "point_ids": [4, 5]},
    ]
    selected = select_distal_terminal(nodes, points, segments)
    assert selected["inlet_node"] == 0
    assert selected["opening_terminal_id"] == "edge12:node3"


def test_stable_terminal_mapping_survives_boundary_renumbering():
    rows = [
        {"tree_id": (0, 0), "owner_seg_idx": 1, "served_region_ids": [99], "zone_idx": 7, "diam_mm": 1, "region_name": "Outlet_008"},
        {"tree_id": (0, 0), "owner_seg_idx": 0, "served_region_ids": [20], "zone_idx": 2, "diam_mm": 2, "region_name": "Outlet_003"},
    ]
    graph_trees = {0: {"tree_ctx": {"seg_id": np.array([20, 99])}}}
    selected = _select_stable_cfx_opening(rows, graph_trees, "edge20:node5")
    assert selected["region_name"] == "Outlet_003"


def test_opening_fraction_is_not_renormalized(tmp_path: Path):
    inlets = {(0, 0): {"ccl_name": "Inlet_000", "region_name": "Inlet_000"}}
    fixed = [
        {"idx": 1, "tree_id": (0, 0), "fraction": 0.3, "ccl_name": "Outlet_001", "region_name": "Outlet_001"},
        {"idx": 2, "tree_id": (0, 0), "fraction": 0.5, "ccl_name": "Outlet_002", "region_name": "Outlet_002"},
    ]
    opening = {"region_name": "Outlet_003", "opening_ccl_name": "Pressure_Opening"}
    path = tmp_path / "bc.ccl"
    generate_cfx_ccl(inlets, fixed, path, opening=opening, boundary_only=True)
    text = path.read_text()
    assert "QoutletNode1 = 0.300000" in text
    assert "QoutletNode2 = 0.500000" in text
    assert "BOUNDARY: Pressure_Opening" in text
    assert "Relative Pressure = 0 [Pa]" in text
    # A boundary-only remesh CCL still needs the inlet boundary assignment;
    # "boundary-only" means that domain physics and solver controls stay in
    # the immutable seed, not that the inlet is omitted.
    assert "BOUNDARY: Inlet_000" in text


def test_explicit_inlet_flow_materializes_fixed_outlet_flows(tmp_path: Path):
    inlets = {(0, 0): {"ccl_name": "Inlet_000", "region_name": "Inlet_000"}}
    fixed = [
        {"idx": 1, "tree_id": (0, 0), "fraction": 0.3,
         "ccl_name": "Outlet_001", "region_name": "Outlet_001"},
        {"idx": 2, "tree_id": (0, 0), "fraction": 0.5,
         "ccl_name": "Outlet_002", "region_name": "Outlet_002"},
    ]
    path = tmp_path / "bc.ccl"
    generate_cfx_ccl(
        inlets, fixed, path, inlet_mass_flow_kg_s=1.0e-3,
        boundary_only=True,
    )
    text = path.read_text()
    assert "Numeric fixed-outlet mass flows" in text
    assert "MoutletNode1 = 0.0003 [kg s^-1]" in text
    assert "MoutletNode2 = 0.0005 [kg s^-1]" in text
    assert "massFlow()@" not in text


def test_boundary_layer_gate_and_gci():
    r4 = {"wss": 1.02, "pressure_drop": 99.5}
    r6 = {"wss": 1.00, "pressure_drop": 100.0}
    r8 = {"wss": 0.995, "pressure_drop": 100.2}
    assert select_boundary_layer(r4, r6, r8) == 4
    gci = observed_order_gci([1.2, 1.1, 1.05], [1_000_000, 8_000_000, 64_000_000])
    assert gci["status"] == "monotonic" and gci["observed_order"] > 0
    with pytest.raises(RuntimeError):
        select_boundary_layer(r4, r6, {"wss": 1.2, "pressure_drop": 90})


def test_mesh_convergence_precedes_boundary_layer_study():
    cfg = {
        "targets": {"element_counts": [1, 2, 3, 4]},
        "boundary_layer": {
            "mesh_convergence_layers": 5,
            "candidate_layers": [4, 6, 8],
        },
        "adaptive": {
            "min_h_mm": 0.02,
            "max_h_mm": 0.4,
            "maximum_sphere_expansion": 1.35,
        },
    }
    cases = build_case_plan(cfg)
    assert [case["purpose"] for case in cases[:8]] == ["main"] * 8
    assert {case["layers"] for case in cases[:8]} == {5}
    assert [case["layers"] for case in cases[8:]] == [4, 6, 8]
    assert all(
        case["purpose"] == "main"
        for case in build_case_plan(cfg, phase="main")
    )


def test_boundary_layer_thickness_cases_reuse_global_l3_bulk_size_contract():
    cfg = {
        "targets": {"element_counts": [1, 2, 3, 4]},
        "boundary_layer": {
            "mesh_convergence_layers": 5,
            "candidate_layers": [4, 6, 8],
            "maximum_channel_radius_ratio": 0.15,
            "thickness_ratio_candidates": [0.05, 0.10, 0.15],
            "thickness_test_level": "l3",
        },
        "adaptive": {
            "min_h_mm": 0.02,
            "max_h_mm": 0.4,
            "maximum_sphere_expansion": 1.35,
        },
    }
    cases = build_case_plan(cfg, phase="boundary_layer_thickness")
    assert [case["case_id"] for case in cases] == [
        "blthick_global_l3_r050", "blthick_global_l3_r100",
    ]
    assert {case["target_elements"] for case in cases} == {3}
    assert {case["layers"] for case in cases} == {5}
    assert {case["baseline_case_id"] for case in cases} == {"global_l3"}
    assert [case["maximum_channel_radius_ratio"] for case in cases] == [
        0.05, 0.10,
    ]


def test_adaptive_successive_level_seed_uses_realised_previous_mesh(tmp_path, monkeypatch):
    import coronary_sdf.mesh_sensitivity_study as study

    output = tmp_path / "output"
    previous = output / "meshes" / "adaptive_l3"
    previous.mkdir(parents=True)
    (previous / "mesh_stats.json").write_text(json.dumps({
        "total_elements": 6_556_415,
        "n_d": 5.989589074890273,
    }))
    source = tmp_path / "source.sip"
    source.write_bytes(b"source")
    cfg = {
        "paths": {"output_dir": str(output), "source_sip": str(source)},
        "targets": {
            "element_counts": [1_500_000, 3_200_000, 6_900_000, 15_000_000],
            "initial_global_h_mm": 0.18,
            "tolerance_fraction": 0.05,
        },
        "boundary_layer": {},
        "adaptive": {
            "initial_n_d": 6.0,
            "successive_level_initial_n_d_safety_factor": 0.95,
        },
    }
    case = {
        "case_id": "adaptive_l4", "family": "adaptive",
        "purpose": "main", "target_elements": 15_000_000,
    }
    observed = {}

    def fake_trial(_cfg, _case, trial, global_h, n_d, resume):
        observed.update(trial=trial, global_h=global_h, n_d=n_d)
        return {"total_elements": 15_000_000}

    monkeypatch.setattr(study, "_mesh_trial_job", fake_trial)
    monkeypatch.setattr(study.shutil, "copy2", lambda *args, **kwargs: None)
    # Stop after observing the seeded first trial, before promotion expects
    # disposable files from the mocked Simpleware execution.
    with pytest.raises(FileNotFoundError):
        study.calibrate_mesh_case(cfg, case, resume=True)
    expected = 5.989589074890273 * (
        15_000_000 / 6_556_415
    ) ** (1.0 / 3.0) * 0.95
    assert observed["trial"] == 1
    assert observed["n_d"] == pytest.approx(expected)


def test_existing_mesh_quality_stage_uses_saved_sip_without_remeshing(
    tmp_path, monkeypatch
):
    import coronary_sdf.mesh_sensitivity_study as study

    output = tmp_path / "output"
    case_dir = output / "meshes" / "global_l1"
    case_dir.mkdir(parents=True)
    (case_dir / "case.sip").write_bytes(b"saved mesh")
    cfg = {
        "paths": {"output_dir": str(output)},
        "executables": {"simpleware_console": "ConsoleSimpleware.exe"},
    }
    observed = {}

    def fake_stream(command, cwd, log_path):
        observed["command"] = command
        job_path = Path(next(
            arg.split("=", 1)[1] for arg in command
            if arg.startswith("--input-value=")
        ))
        job = json.loads(job_path.read_text())
        Path(job["quality_inspection_json"]).write_text(json.dumps({
            "totals": {"error": 0, "warning": 2, "feature": 0},
            "metrics": [{
                "case_id": "global_l1", "metric_index": 0,
                "metric": "Jacobian", "level": "warning",
                "uses_threshold": True, "threshold": 0.05,
                "less_than_threshold_is_valid": False,
                "problem_count": 2,
            }],
        }))
        Path(job["quality_inspection_csv"]).write_text("metric\nJacobian\n")

    monkeypatch.setattr(study, "_stream_command", fake_stream)
    summary = export_existing_simpleware_quality(cfg, resume=False)
    assert summary.is_file()
    assert json.loads(summary.read_text())["case_count"] == 1
    assert any(
        arg.endswith("simpleware_mesh_quality_inspection_console.py")
        for arg in observed["command"]
        if arg.startswith("--run-script=")
    )
    assert not any("GenerateMesh" in arg for arg in observed["command"])


def test_binary_cfx_seed_contract_and_solver_terminal_state(tmp_path: Path):
    ccl = """
    MATERIAL: Quemada END
    FLOW: Flow Analysis 1
      DOMAIN: Default Domain
        BOUNDARY: Inlet_000
          BOUNDARY CONDITIONS:
            MASS AND MOMENTUM:
              Mass Flow Rate = 0.000897915 [kg s^-1]
            END
          END
        END
        TURBULENCE MODEL: Turbulence
          Option = Laminar
        END
      END
      SOLVER CONTROL:
        CONVERGENCE CONTROL:
          Maximum Number of Iterations = 300
          Minimum Number of Iterations = 150
          Residual Target = 0.000001
          Residual Type = RMS
        END
        CONVERGENCE CONTROL: Fluid Timescale Control
          Timescale Control = Auto Timescale
          Timescale Factor = 1.0
        END
      END
    END
    """
    seed = tmp_path / "seed.cfx"
    seed.write_bytes(ccl.replace(" ", "\x00 ").encode("latin1"))
    contract = extract_cfx_seed_contract(seed)
    assert contract["inlet_mass_flow_kg_s"] == pytest.approx(0.000897915)
    assert contract["material_model"] == "Quemada"
    assert contract["turbulence_option"] == "Laminar"
    assert contract["residual_target"] == pytest.approx(1e-6)
    assert len(contract["physics_control_sha256"]) == 64

    solver = tmp_path / "run.out"
    solver.write_text("Iteration 151\nConvergence criteria satisfied\n")
    solver_state = parse_cfx_solver_output(solver)
    assert {
        key: solver_state[key]
        for key in ("converged", "normal_completion", "last_iteration")
    } == {
        "converged": True, "normal_completion": True, "last_iteration": 151,
    }
    assert solver_state["last_rms_residuals"] == {}


def test_cfd_post_boundary_audit_session_contains_all_boundaries(tmp_path: Path):
    session = tmp_path / "audit.cse"
    _write_cfx_boundary_audit_session(
        session, tmp_path / "audit.csv",
        ["Inlet_000", "Outlet_001", "Pressure_Opening"],
    )
    text = session.read_text()
    assert 'evaluate("massFlow()\\@Inlet_000")' in text
    assert 'evaluate("massFlow()\\@Pressure_Opening")' in text
    assert ">quit" in text
