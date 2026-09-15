from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pyvista as pv
import pytest

from coronary_sdf import config
from coronary_sdf import __main__ as cli_module
from coronary_sdf.amira_lattice import read_amira_lattice
from coronary_sdf.benchmark import CANDIDATES, _capsules_from_graph, run_synthetic
from coronary_sdf.benchmark_metrics import symmetric_surface_metrics
from coronary_sdf.cgal_adapter import (
    cgal_mesh3_available,
    native_field_values,
    native_sizing_values,
)
from coronary_sdf.conflict_audit import audit_capsule_conflicts_against_mask
from coronary_sdf.geometry_constraints import find_capsule_conflicts
from coronary_sdf.implicit_field import (
    GraphImplicitField,
    JunctionBlend,
    round_cone_values,
)
from coronary_sdf.mesh_validation import validate_mesh
from coronary_sdf.profiles import ProfileUnavailable, candidate_cfd_config, resolve_profile
from coronary_sdf.synthetic_cases import synthetic_suite
from coronary_sdf.vtk_htg_mesher import mesh_vtk_hyper_tree_grid

# Sibling test module, not part of the package: under the old flat layout the
# repo root *was* `coronary_sdf`, so this read `coronary_sdf.test_implicit_field`.
# Tests now live outside the package, and pytest puts this directory on sys.path.
from test_implicit_field import _capsules


def test_runtime_configuration_is_nested_and_does_not_mutate_defaults():
    original = config.CENTERLINE_SMOOTHER
    outer = config.SdfConfig(CENTERLINE_SMOOTHER="none")
    inner = outer.with_overrides(CENTERLINE_SMOOTHER="bspline")
    with config.use_config(outer):
        assert config.runtime_config.CENTERLINE_SMOOTHER == "none"
        with config.use_config(inner):
            assert config.runtime_config.CENTERLINE_SMOOTHER == "bspline"
        assert config.runtime_config.CENTERLINE_SMOOTHER == "none"
    assert config.CENTERLINE_SMOOTHER == original


def test_round_cone_matches_dense_union_of_interpolated_balls():
    start = np.asarray([[0.0, 0.0, 0.0]])
    end = np.asarray([[3.0, 0.0, 0.0]])
    r0 = np.asarray([0.8])
    r1 = np.asarray([0.25])
    parameters = np.linspace(0.0, 1.0, 200_001)
    centres = start[0] + parameters[:, None] * (end[0] - start[0])
    radii = r0[0] + parameters * (r1[0] - r0[0])
    probes = np.asarray(
        [[-0.3, 0.2, 0.0], [0.4, 0.9, 0.0], [1.5, 0.5, 0.4], [3.2, 0.0, 0.0]]
    )
    for point in probes:
        exact = round_cone_values(point, start, end, r0, r1)[0][0]
        brute = np.min(np.linalg.norm(centres - point, axis=1) - radii)
        assert abs(exact - brute) < 2e-6


def test_round_cone_bvh_matches_exhaustive_values():
    rng = np.random.default_rng(17)
    starts = rng.normal(size=(40, 3))
    ends = starts + rng.normal(scale=0.8, size=(40, 3))
    lengths = np.linalg.norm(ends - starts, axis=1)
    r0 = rng.uniform(0.05, 0.25, len(starts))
    # Keep the regular round-cone condition for this parity test.
    r1 = np.clip(r0 + rng.uniform(-0.1, 0.1, len(starts)), 0.02, None)
    too_steep = np.abs(r1 - r0) >= 0.9 * lengths
    r1[too_steep] = r0[too_steep]
    capsules = _capsules(starts, ends, r0, r1, np.arange(len(starts)))
    field = GraphImplicitField(capsules, primitive_method="round_cone")
    for point in rng.normal(size=(50, 3)):
        exhaustive = round_cone_values(point, starts, ends, r0, r1)[0].min()
        assert abs(field.sample(point).value - exhaustive) < 1e-12
    points = rng.normal(size=(300, 3))
    values = field.evaluate(points)[0]
    expected = np.asarray(
        [round_cone_values(point, starts, ends, r0, r1)[0].min() for point in points]
    )
    np.testing.assert_allclose(values, expected, atol=1e-12, rtol=1e-12)


@pytest.mark.skipif(not cgal_mesh3_available(), reason="native CGAL extension is not installed")
def test_native_cgal_oracle_matches_shared_field_and_radius_sizing():
    capsules = _capsules(
        np.asarray([[0.0, 0.0, 0.0]]),
        np.asarray([[2.0, 0.0, 0.0]]),
        np.asarray([0.5]),
        np.asarray([0.25]),
        np.asarray([0]),
    )
    field = GraphImplicitField(capsules, primitive_method="round_cone")
    probes = np.asarray([[0.5, 0.2, 0.0], [1.5, 0.6, 0.0]])
    native_values, native_radii = native_field_values(field, probes)
    expected_values, _owners, expected_radii, _gradients = field.evaluate(probes)
    np.testing.assert_allclose(native_values, expected_values, atol=1e-10, rtol=0.0)
    np.testing.assert_allclose(native_radii, expected_radii, atol=1e-10, rtol=0.0)
    sizing = native_sizing_values(field, probes, 10.0)
    np.testing.assert_allclose(sizing, 2.0 * expected_radii / 10.0, atol=1e-10)
    assert np.all(sizing > 0.0)
    assert np.all(sizing <= 0.1)


@pytest.mark.skipif(not cgal_mesh3_available(), reason="native CGAL extension is not installed")
@pytest.mark.parametrize("scale", (0.01, 1.0, 100.0))
def test_native_cgal_oracle_random_adversarial_scale_parity(scale: float):
    rng = np.random.default_rng(4831)
    starts = rng.normal(size=(24, 3)) * scale
    ends = starts + rng.normal(size=(24, 3)) * scale
    lengths = np.linalg.norm(ends - starts, axis=1)
    r0 = rng.uniform(0.0, 0.3, len(starts)) * scale
    r1 = rng.uniform(0.0, 0.3, len(starts)) * scale
    steep = np.abs(r1 - r0) >= 0.95 * lengths
    r1[steep] = r0[steep]
    # Exercise exact zero-radius tips without creating an all-zero field.
    r0[0] = 0.0
    r1[1] = 0.0
    capsules = _capsules(starts, ends, r0, r1, np.arange(len(starts)))
    field = GraphImplicitField(capsules, primitive_method="round_cone")
    probes = np.vstack(
        (
            rng.normal(size=(128, 3)) * scale,
            starts,
            ends,
            starts - (ends - starts),
            ends + (ends - starts),
        )
    )
    native_values, native_radii = native_field_values(field, probes)
    expected_values, _owners, expected_radii, _gradients = field.evaluate(probes)
    np.testing.assert_allclose(native_values, expected_values, atol=1e-10, rtol=0.0)
    np.testing.assert_allclose(native_radii, expected_radii, atol=1e-10, rtol=0.0)
    expected_sizes = 2.0 * expected_radii / 18.0
    np.testing.assert_allclose(
        native_sizing_values(field, probes, 18.0),
        expected_sizes,
        atol=1e-10,
        rtol=0.0,
    )
    assert np.all(expected_sizes > 0.0)


@pytest.mark.skipif(not cgal_mesh3_available(), reason="native CGAL extension is not installed")
def test_native_cgal_junction_support_and_blend_parity():
    capsules = _capsules(
        np.asarray([[-2.0, 0.0, 0.0], [0.0, 0.0, 0.0]]),
        np.asarray([[0.0, 0.0, 0.0], [1.5, 1.5, 0.0]]),
        np.asarray([0.4, 0.4]),
        np.asarray([0.4, 0.3]),
        np.asarray([0, 1]),
        cap_bif_at_start=np.asarray([False, True]),
        cap_bif_at_end=np.asarray([True, False]),
    )
    junction = JunctionBlend(
        node_id=7,
        position=np.zeros(3),
        radius=0.4,
        incident_segments=(0, 1),
        capsule_groups=(np.asarray([0]), np.asarray([1])),
        blend_fraction=0.15,
        support_factor=4.0,
    )
    field = GraphImplicitField(
        capsules, junctions=(junction,), primitive_method="round_cone"
    )
    probes = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [0.2, 0.15, 0.0],
            [0.8, 0.0, 0.0],
            [1.59, 0.0, 0.0],
            [1.61, 0.0, 0.0],
            [-2.5, 0.0, 0.0],
        ]
    )
    native_values, native_radii = native_field_values(field, probes)
    expected_values, _owners, expected_radii, _gradients = field.evaluate(probes)
    np.testing.assert_allclose(native_values, expected_values, atol=1e-10, rtol=0.0)
    np.testing.assert_allclose(native_radii, expected_radii, atol=1e-10, rtol=0.0)


def _literal_rle(data: bytes) -> bytes:
    chunks = []
    for offset in range(0, len(data), 127):
        block = data[offset : offset + 127]
        chunks.append(bytes([128 + len(block)]) + block)
    return b"".join(chunks)


def test_hxbyterle_lattice_roundtrip_to_memmap(tmp_path: Path):
    expected = np.arange(4 * 3 * 2, dtype=np.uint8).reshape(2, 3, 4)
    payload = _literal_rle(expected.tobytes())
    header = (
        b"# Avizo BINARY-LITTLE-ENDIAN 3.0\n\n"
        b"define Lattice 4 3 2\n"
        b"Parameters { BoundingBox 1000 4000 2000 4000 3000 5000, "
        b"CoordType \"uniform\" }\n"
        + f"Lattice {{ byte Labels }} @1(HxByteRLE,{len(payload)})\n\n".encode()
        + b"# Data section follows\n@1\n"
    )
    source = tmp_path / "fixture.am"
    source.write_bytes(header + payload)
    lattice = read_amira_lattice(source, cache_dir=tmp_path / "cache")
    np.testing.assert_array_equal(lattice.slice_z(1), expected[1])
    np.testing.assert_allclose(lattice.header.spacing_mm, [1.0, 1.0, 2.0])
    decoded = lattice.decode_to_memmap(tmp_path / "labels.raw")
    np.testing.assert_array_equal(decoded, expected)


def test_common_mesh_validator_accepts_sphere_and_rejects_open_surface():
    sphere = pv.Sphere(theta_resolution=12, phi_resolution=12).triangulate()
    valid = validate_mesh(sphere, check_self_intersections=True)
    assert valid.valid, valid.errors
    assert valid.genus == 0.0

    plane = pv.Plane(i_resolution=2, j_resolution=2).triangulate()
    invalid = validate_mesh(plane, check_self_intersections=False)
    assert not invalid.valid
    assert invalid.boundary_edges > 0


def test_conflict_classifier_separates_local_bends_from_hairpin_contacts():
    cases = {case.name: case for case in synthetic_suite()}
    for name in ("straight", "torus_kappa_r_0.8"):
        case = cases[name]
        capsules = _capsules_from_graph(case.points, case.segments)
        assert not find_capsule_conflicts(capsules)
    hairpin = cases["hairpin_overlap"]
    conflicts = find_capsule_conflicts(
        _capsules_from_graph(hairpin.points, hairpin.segments)
    )
    assert conflicts
    assert min(item.clearance for item in conflicts) < 0.0


class _FakeLattice:
    def __init__(self, volume: np.ndarray):
        self.volume = volume
        self.nz, self.ny, self.nx = volume.shape
        self.header = SimpleNamespace(
            origin_mm=np.zeros(3, dtype=float),
            spacing_mm=np.ones(3, dtype=float),
        )

    def slice_z(self, z: int) -> np.ndarray:
        return self.volume[z]


def test_mask_conflict_audit_distinguishes_contact_from_radius_overestimate():
    capsules = _capsules(
        np.asarray([[2.0, 4.0, 2.0], [2.0, 6.0, 2.0]]),
        np.asarray([[9.0, 4.0, 2.0], [9.0, 6.0, 2.0]]),
        np.asarray([1.2, 1.2]),
        np.asarray([1.2, 1.2]),
        np.asarray([0, 1]),
    )
    conflicts = find_capsule_conflicts(capsules, maximum_records=1)
    assert conflicts
    separated = np.zeros((5, 12, 12), dtype=np.uint8)
    separated[2, 4, 2:10] = 1
    separated[2, 6, 2:10] = 1
    report = audit_capsule_conflicts_against_mask(
        capsules, conflicts, _FakeLattice(separated)
    )[0]
    assert report.classification == "radius_overestimation"
    assert not report.input_valid

    connected = separated.copy()
    connected[2, 4:7, 2:10] = 1
    report = audit_capsule_conflicts_against_mask(
        capsules, conflicts, _FakeLattice(connected)
    )[0]
    assert report.classification == "mask_confirmed_contact"
    assert report.input_valid


def test_cli_profiles_keep_compatibility_and_gate_unqualified_cfd(tmp_path: Path):
    compat = resolve_profile("compat")
    assert compat.SDF_MESH_METHOD == config.SDF_MESH_METHOD
    assert compat.CENTERLINE_SMOOTHER == config.CENTERLINE_SMOOTHER
    candidate = candidate_cfd_config()
    assert candidate.CENTERLINE_SMOOTHER == "none"
    assert candidate.IMPLICIT_PRIMITIVE_METHOD == "round_cone"
    assert candidate.SDF_MESH_METHOD == "vtk_htg"
    assert not candidate.VTK_HTG_DECOMPOSED_POLYHEDRA
    assert candidate.IMPLICIT_JUNCTION_BLEND_FRACTION == 0.05
    assert candidate.IMPLICIT_JUNCTION_SUPPORT_FACTOR == 2.0
    assert candidate.OUTPUT_VALIDATION_MODE == "error"
    assert resolve_profile("experimental") == candidate
    with pytest.raises(ProfileUnavailable):
        resolve_profile("cfd", qualified_path=tmp_path / "missing.json")
    token = tmp_path / "qualified.json"
    token.write_text(
        json.dumps(
            {
                "qualified": True,
                "winner": "graph_round_cone_vtk_htg",
                "real_cases": ["LADAF_2024_28", "LADAF_2024_56"],
                "config": candidate.to_dict(),
            }
        ),
        encoding="utf-8",
    )
    assert resolve_profile("cfd", qualified_path=token) == candidate


def test_experimental_cli_applies_adaptive_budget_overrides(monkeypatch, tmp_path: Path):
    captured = {}

    def fake_run_pipeline(input_path, output_dir, *, cfg, write_outputs, interactive):
        captured.update(
            input_path=input_path,
            output_dir=output_dir,
            cfg=cfg,
            write_outputs=write_outputs,
            interactive=interactive,
        )

    monkeypatch.setattr(cli_module, "run_pipeline", fake_run_pipeline)
    result = cli_module.main(
        [
            "input.am",
            str(tmp_path / "out"),
            "--profile",
            "experimental",
            "--cells-across-diameter",
            "6",
            "--max-cells",
            "7000000",
            "--validation-mode",
            "warn",
            "--non-interactive",
        ]
    )
    assert result == 0
    assert captured["cfg"].IMPLICIT_CELLS_ACROSS_DIAMETER == 6.0
    assert captured["cfg"].VTK_HTG_MAX_CELLS == 7_000_000
    assert captured["cfg"].OUTPUT_VALIDATION_MODE == "warn"
    assert captured["interactive"] is False


def test_cli_does_not_weaken_qualified_cfd_validation(monkeypatch, tmp_path: Path):
    candidate = candidate_cfd_config()
    token = tmp_path / "qualified.json"
    token.write_text(
        json.dumps(
            {
                "qualified": True,
                "winner": "graph_round_cone_vtk_htg",
                "real_cases": ["LADAF_2024_28", "LADAF_2024_56"],
                "config": candidate.to_dict(),
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(cli_module, "run_pipeline", lambda *args, **kwargs: None)
    with pytest.raises(SystemExit) as exc:
        cli_module.main(
            [
                "input.am",
                str(tmp_path / "out"),
                "--profile",
                "cfd",
                "--qualified-config",
                str(token),
                "--validation-mode",
                "warn",
            ]
        )
    assert exc.value.code == 2


def test_scale_equivariant_dense_candidate_reconstructs_small_scale(tmp_path: Path):
    result = run_synthetic(
        {
            "case": "global_scale_0.01",
            "candidate": "graph_round_cone_dense_mc",
            "preprocessor": "none",
            "scratch": str(tmp_path / "small"),
        }
    )
    assert result["qualified"], result


def test_surface_metrics_are_deterministic():
    first = pv.Sphere(radius=1.0, theta_resolution=17, phi_resolution=13)
    second = pv.Sphere(radius=1.03, theta_resolution=19, phi_resolution=15)
    a = symmetric_surface_metrics(first, second, maximum_samples=53)
    b = symmetric_surface_metrics(first, second, maximum_samples=53)
    assert a == b


def test_vtk_htg_voxel_contour_is_strictly_valid_and_deterministic():
    capsules = _capsules(
        np.asarray([[0.0, 0.0, 0.0]]),
        np.asarray([[4.0, 0.0, 0.0]]),
        np.asarray([0.5]),
        np.asarray([0.5]),
        np.asarray([0]),
    )
    field = GraphImplicitField(capsules, primitive_method="round_cone")
    first, first_stats = mesh_vtk_hyper_tree_grid(
        field, cells_across_diameter=6.0, decomposed_polyhedra=False
    )
    second, second_stats = mesh_vtk_hyper_tree_grid(
        field, cells_across_diameter=6.0, decomposed_polyhedra=False
    )
    report = validate_mesh(first, expected_components=1, check_self_intersections=True)
    assert report.valid, report.to_dict()
    assert report.genus == 0.0
    assert (first.n_points, first.n_cells) == (second.n_points, second.n_cells)
    assert first_stats.field_evaluations == second_stats.field_evaluations
    np.testing.assert_allclose(first.bounds, second.bounds, atol=0.0, rtol=0.0)
    with pytest.raises(MemoryError, match="maximum_cells=1"):
        mesh_vtk_hyper_tree_grid(
            field,
            cells_across_diameter=6.0,
            maximum_cells=1,
            decomposed_polyhedra=False,
        )


@pytest.mark.parametrize("candidate", tuple(CANDIDATES))
def test_each_candidate_runs_production_path_without_writes(
    candidate: str, tmp_path: Path
):
    scratch = tmp_path / candidate
    result = run_synthetic(
        {
            "case": "straight",
            "candidate": candidate,
            "preprocessor": "none",
            "scratch": str(scratch),
        }
    )
    # This is an execution/integration contract. Qualification remains the
    # responsibility of the complete benchmark matrix.
    assert result["validation"]["finite"]
    assert result["mesh_points"] > 0
    assert result["mesh_faces"] > 0
    assert not scratch.exists()
