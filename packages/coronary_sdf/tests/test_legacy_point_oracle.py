"""Parity and safety tests for the legacy field's point-query oracle.

The legacy dense evaluator is the behavioural baseline for the whole package,
and this tree has no version control, so the golden volumes in
``test_data/golden`` are the only reference proving that extracting a
point-queryable kernel out of :func:`sdf_field.evaluate_sdf` is a no-op.
Failures here are findings, not thresholds to relax.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from coronary_sdf import config as config_module
from coronary_sdf import pipeline as pipeline_module
from coronary_sdf.benchmark import candidate_config
from coronary_sdf.implicit_field import build_graph_implicit_field
from coronary_sdf.legacy_field_adapter import LegacyPointField
from coronary_sdf.sdf_field import evaluate_sdf_points
from coronary_sdf.synthetic_cases import synthetic_suite

GOLDEN_DIR = Path(__file__).resolve().parent / "test_data" / "golden"

GOLDEN_FIXTURES = (
    "straight",
    "tapered",
    "junction_degree_3",
    "junction_degree_5",
    "hairpin_overlap",
)


class _Captured(Exception):
    """Abort the pipeline as soon as the dense field has been evaluated."""

    def __init__(self, volume, kwargs):
        super().__init__("captured")
        self.volume = volume
        self.kwargs = kwargs
        self.grid = kwargs["grid"]


def capture_dense_legacy_run(case, **overrides):
    """Run the legacy dense field for ``case`` and capture its inputs/outputs.

    Drives the real pipeline so the 15 topology arrays are assembled exactly as
    production assembles them, rather than reconstructed by the test.
    """

    real_evaluate = pipeline_module.evaluate_sdf

    def capturing(**kwargs):
        raise _Captured(real_evaluate(**kwargs), kwargs)

    cfg = candidate_config(
        "legacy_dense_meshlib", preprocessor="none", resolution=None
    )
    if overrides:
        cfg = cfg.with_overrides(**overrides)
    pipeline_module.evaluate_sdf = capturing
    try:
        with config_module.use_config(cfg):
            pipeline_module._generate_sdf_surface(
                case.nodes,
                case.points,
                case.segments,
                GOLDEN_DIR,
                0,
                write_outputs=False,
                interactive=False,
            )
    except _Captured as captured:
        return captured, cfg
    finally:
        pipeline_module.evaluate_sdf = real_evaluate
    raise AssertionError("the dense legacy field was never evaluated")


def evaluate_dense_legacy_volume(case):
    captured, _cfg = capture_dense_legacy_run(case)
    return np.asarray(captured.volume.sdf), captured.grid


def _kernel_kwargs(captured) -> dict:
    """Topology arguments for the point kernel, minus the grid plumbing."""

    return {
        key: value
        for key, value in captured.kwargs.items()
        if key not in {"grid", "nb_idx"}
    }


@pytest.mark.parametrize("name", GOLDEN_FIXTURES)
def test_legacy_dense_volume_matches_golden(name: str) -> None:
    """The dense legacy volume must stay bitwise identical to its golden.

    Every operation in the narrow-band loop is elementwise per point, and the
    only cross-point steps (the KD-tree query and a per-row ``argmin``) are
    row-independent, so batching cannot legitimately change a value. Any
    difference here is a real behavioural change.
    """

    golden_path = GOLDEN_DIR / f"legacy_sdf_{name}.npz"
    if not golden_path.exists():
        pytest.skip(f"golden volume not captured: {golden_path.name}")
    golden = np.load(golden_path)

    case = {item.name: item for item in synthetic_suite()}[name]
    volume, grid = evaluate_dense_legacy_volume(case)

    assert tuple(int(d) for d in grid.dims) == tuple(
        int(d) for d in golden["dims"]
    ), "grid dimensions drifted from the golden capture"
    assert float(grid.voxel_size) == pytest.approx(
        float(golden["voxel_size"][0]), rel=0.0, abs=0.0
    ), "voxel size drifted from the golden capture"
    np.testing.assert_array_equal(
        volume,
        golden["sdf"],
        err_msg=f"legacy dense field changed for fixture {name!r}",
    )


@pytest.mark.parametrize("name", ["straight", "junction_degree_3", "junction_degree_5"])
def test_point_kernel_matches_the_dense_narrow_band(name: str) -> None:
    """The point form must agree bitwise with the dense path it was cut from.

    Every step in the batch loop is elementwise per point, and the only
    cross-point operations (the KD-tree query and a per-row ``argmin``) are
    row-independent, so a mismatch means a real behavioural divergence rather
    than floating-point reassociation.
    """

    case = {item.name: item for item in synthetic_suite()}[name]
    captured, cfg = capture_dense_legacy_run(case)
    grid = captured.grid
    nb_idx = np.asarray(captured.kwargs["nb_idx"])
    coords = np.column_stack(
        [grid.x[nb_idx[:, 0]], grid.y[nb_idx[:, 1]], grid.z[nb_idx[:, 2]]]
    )

    with config_module.use_config(cfg):
        values = evaluate_sdf_points(
            coords, length_scale=grid.voxel_size, **_kernel_kwargs(captured)
        )

    dense = np.asarray(captured.volume.sdf)[nb_idx[:, 0], nb_idx[:, 1], nb_idx[:, 2]]
    np.testing.assert_array_equal(
        values, dense, err_msg=f"point kernel diverged from the dense path for {name!r}"
    )


def test_point_kernel_is_invariant_to_query_order() -> None:
    """Evaluation must not depend on how the caller batches or orders points."""

    case = {item.name: item for item in synthetic_suite()}["junction_degree_3"]
    captured, cfg = capture_dense_legacy_run(case)
    grid = captured.grid
    nb_idx = np.asarray(captured.kwargs["nb_idx"])
    coords = np.column_stack(
        [grid.x[nb_idx[:, 0]], grid.y[nb_idx[:, 1]], grid.z[nb_idx[:, 2]]]
    )
    permutation = np.random.default_rng(20260818).permutation(len(coords))

    with config_module.use_config(cfg):
        straight = evaluate_sdf_points(
            coords, length_scale=grid.voxel_size, **_kernel_kwargs(captured)
        )
        shuffled = evaluate_sdf_points(
            coords[permutation],
            length_scale=grid.voxel_size,
            batch_size=97,
            **_kernel_kwargs(captured),
        )

    np.testing.assert_array_equal(shuffled, straight[permutation])


def test_legacy_field_zero_set_depends_on_its_length_scale() -> None:
    """Documents that the legacy field is not resolution-independent.

    ``length_scale`` sets the flat-cap soft band, the capsule candidate search
    radius and the carve buffers, so the legacy zero set moves with the
    extractor's grid spacing. This is asserted rather than marked xfail: any
    comparison of the legacy field across two extractors must pin this value,
    and if the coupling is ever removed this test should be revisited
    deliberately.
    """

    case = {item.name: item for item in synthetic_suite()}["junction_degree_5"]
    captured, cfg = capture_dense_legacy_run(case)
    grid = captured.grid
    nb_idx = np.asarray(captured.kwargs["nb_idx"])
    coords = np.column_stack(
        [grid.x[nb_idx[:, 0]], grid.y[nb_idx[:, 1]], grid.z[nb_idx[:, 2]]]
    )

    with config_module.use_config(cfg):
        coarse = evaluate_sdf_points(
            coords, length_scale=grid.voxel_size, **_kernel_kwargs(captured)
        )
        fine = evaluate_sdf_points(
            coords, length_scale=grid.voxel_size / 4.0, **_kernel_kwargs(captured)
        )

    assert not np.array_equal(coarse, fine), (
        "the legacy field no longer depends on its length scale; the ablation's "
        "length-scale pinning may no longer be necessary"
    )


def test_point_kernel_refuses_to_silently_drop_gaussian_smoothing() -> None:
    """Omitting a grid-shaped step would make this a different field."""

    case = {item.name: item for item in synthetic_suite()}["straight"]
    captured, cfg = capture_dense_legacy_run(case)
    smoothed = cfg.with_overrides(SDF_GAUSSIAN_SIGMA_VOXELS=1.0)

    with config_module.use_config(smoothed):
        with pytest.raises(ValueError, match="SDF_GAUSSIAN_SIGMA_VOXELS"):
            evaluate_sdf_points(
                np.zeros((1, 3)), length_scale=0.1, **_kernel_kwargs(captured)
            )


def _legacy_oracle(case, cfg=None):
    """Build a LegacyPointField plus its shared sizing field for ``case``."""

    captured, resolved = capture_dense_legacy_run(case)
    cfg = cfg or resolved
    kwargs = _kernel_kwargs(captured)
    capsules = kwargs.pop("capsules")
    with config_module.use_config(cfg):
        sizing = build_graph_implicit_field(
            capsules,
            case.nodes,
            _node_to_segments(case.segments),
            blend_fraction=cfg.IMPLICIT_JUNCTION_BLEND_FRACTION,
            support_factor=cfg.IMPLICIT_JUNCTION_SUPPORT_FACTOR,
            primitive_method="radial",
            clip_bifurcation_caps=cfg.SDF_FLAT_CAP_BIF,
        )
        oracle = LegacyPointField(
            capsules,
            length_scale=captured.grid.voxel_size,
            sizing_field=sizing,
            **kwargs,
        )
    return oracle, sizing, cfg, captured


def _node_to_segments(segments):
    mapping: dict[int, set[int]] = {}
    for index, segment in enumerate(segments):
        for key in ("node1", "node2"):
            mapping.setdefault(int(segment[key]), set()).add(index)
    return mapping


def test_oracle_shares_bounds_and_sizing_with_the_graph_field() -> None:
    """The hierarchy must be identical so only the field value differs."""

    case = {item.name: item for item in synthetic_suite()}["junction_degree_3"]
    oracle, sizing, _cfg, _captured = _legacy_oracle(case)

    np.testing.assert_array_equal(oracle.bounds_min, sizing.bounds_min)
    np.testing.assert_array_equal(oracle.bounds_max, sizing.bounds_max)
    np.testing.assert_array_equal(oracle.max_radii, sizing.max_radii)

    rng = np.random.default_rng(20260818)
    span = sizing.bounds_max - sizing.bounds_min
    for _ in range(48):
        point = sizing.bounds_min + span * rng.random(3)
        cell = float(10.0 ** rng.uniform(-3, -0.5))
        assert oracle.minimum_relevant_radius(point, cell) == (
            sizing.minimum_relevant_radius(point, cell)
        )


def test_oracle_values_differ_from_the_graph_field_it_borrows_sizing_from() -> None:
    """Sharing geometry must not accidentally share the field itself."""

    case = {item.name: item for item in synthetic_suite()}["junction_degree_3"]
    oracle, sizing, cfg, _captured = _legacy_oracle(case)
    rng = np.random.default_rng(11)
    span = sizing.bounds_max - sizing.bounds_min
    probes = sizing.bounds_min + span * rng.random((256, 3))

    with config_module.use_config(cfg):
        legacy_values, _o, _r, _g = oracle.evaluate(probes)
    graph_values, _o, _r, _g = sizing.evaluate(probes)

    assert not np.allclose(legacy_values, graph_values)


def test_geometric_prune_never_discards_a_cell_containing_the_zero_set() -> None:
    """Hole-prevention proof for the exclusion certificate.

    Every cell the predicate calls 'certainly outside' is densely resampled; all
    samples must be strictly positive. An under-estimate here would silently
    delete surface rather than fail loudly.
    """

    case = {item.name: item for item in synthetic_suite()}["junction_degree_3"]
    oracle, sizing, cfg, _captured = _legacy_oracle(case)
    rng = np.random.default_rng(4242)
    span = sizing.bounds_max - sizing.bounds_min
    half_diagonal = 0.25 * float(np.min(span))
    centres = sizing.bounds_min + span * rng.random((64, 3))

    outside = oracle.certainly_outside_band(centres, half_diagonal)
    assert np.any(outside), "the certificate pruned nothing; the test is vacuous"

    cell = half_diagonal / np.sqrt(3.0)
    offsets = np.stack(
        np.meshgrid(*(np.linspace(-cell, cell, 4),) * 3, indexing="ij"), axis=-1
    ).reshape(-1, 3)
    with config_module.use_config(cfg):
        for centre in centres[outside]:
            values, _o, _r, _g = oracle.evaluate(centre[None, :] + offsets)
            assert np.all(values > 0.0), (
                "a cell certified as outside contains a non-positive value; "
                "the prune certificate is unsound and would create holes"
            )


def test_prune_mode_none_refines_everything() -> None:
    case = {item.name: item for item in synthetic_suite()}["straight"]
    captured, cfg = capture_dense_legacy_run(case)
    kwargs = _kernel_kwargs(captured)
    capsules = kwargs.pop("capsules")
    with config_module.use_config(cfg):
        sizing = build_graph_implicit_field(
            capsules, case.nodes, _node_to_segments(case.segments),
            primitive_method="radial",
        )
        oracle = LegacyPointField(
            capsules,
            length_scale=captured.grid.voxel_size,
            sizing_field=sizing,
            prune_mode="none",
            **kwargs,
        )
    centres = sizing.bounds_min + (sizing.bounds_max - sizing.bounds_min) * np.random.default_rng(3).random((32, 3))
    assert not np.any(oracle.certainly_outside_band(centres, 0.01))


def test_graph_field_has_no_prune_hook_so_its_behaviour_is_unchanged() -> None:
    """The hook must be opt-in; the existing candidate must not shift."""

    case = {item.name: item for item in synthetic_suite()}["straight"]
    _oracle, sizing, _cfg, _captured = _legacy_oracle(case)
    assert not hasattr(sizing, "certainly_outside_band")
