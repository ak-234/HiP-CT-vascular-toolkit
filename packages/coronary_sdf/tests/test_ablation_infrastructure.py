"""Tests for the diagnostic ablation infrastructure.

Covers the per-component completion manifest and the branch-coverage metric.
Kept separate from ``test_refinement_infrastructure.py`` only to stop that file
growing without bound; the style and fixtures are the same.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pyvista as pv
import pytest

from coronary_sdf import benchmark
from coronary_sdf import pipeline as pipeline_module
from coronary_sdf.benchmark import candidate_config
from coronary_sdf.branch_coverage import branch_coverage, segment_centreline_samples
from coronary_sdf.epicardial_annotation import write_amira_xml
from coronary_sdf.pipeline_report import INCOMPLETE_MARKER, MANIFEST_NAME
from coronary_sdf.synthetic_cases import synthetic_suite

TWO_COMPONENT_CASE = "parallel_clearance_+0.2"


def _case(name: str):
    return {item.name: item for item in synthetic_suite()}[name]


def _write_graph(case, directory: Path) -> Path:
    path = directory / "graph.xml"
    write_amira_xml(case.nodes, case.points, case.segments, str(path))
    return path


def _legacy_config(**overrides):
    cfg = candidate_config(
        "legacy_dense_meshlib", preprocessor="none", resolution=None
    )
    return cfg.with_overrides(**overrides) if overrides else cfg


# ── Component manifest ───────────────────────────────────────────────────────


def test_manifest_records_every_component_of_a_complete_run(tmp_path: Path) -> None:
    case = _case(TWO_COMPONENT_CASE)
    graph = _write_graph(case, tmp_path)
    out = tmp_path / "out"

    surfaces, report = pipeline_module.run_pipeline(
        graph,
        out,
        cfg=_legacy_config(),
        write_outputs=True,
        interactive=False,
        return_report=True,
    )

    assert sorted(surfaces) == [0, 1]
    assert report.requested_components == (0, 1)
    assert report.completed_components == (0, 1)
    assert report.failed_components == ()
    assert report.incomplete is False
    assert not (out / INCOMPLETE_MARKER).exists()

    manifest = json.loads((out / MANIFEST_NAME).read_text(encoding="utf-8"))
    assert manifest["incomplete"] is False
    assert [item["status"] for item in manifest["components"]] == ["ok", "ok"]
    # The resolution descriptor must survive, or dense and adaptive runs cannot
    # be placed on a common resolution axis later.
    assert all(item["voxel_size_mm"] > 0 for item in manifest["components"])


def test_component_zero_is_unsuffixed_and_recorded(tmp_path: Path) -> None:
    case = _case(TWO_COMPONENT_CASE)
    graph = _write_graph(case, tmp_path)
    out = tmp_path / "out"

    _surfaces, report = pipeline_module.run_pipeline(
        graph,
        out,
        cfg=_legacy_config(),
        write_outputs=True,
        interactive=False,
        return_report=True,
    )

    by_id = {item.graph_id: item for item in report.components}
    names_0 = {Path(p).name for p in by_id[0].output_paths}
    names_1 = {Path(p).name for p in by_id[1].output_paths}
    assert "lumen_bspline.stl" in names_0
    assert "lumen_bspline_g1.stl" in names_1
    for path in by_id[0].output_paths + by_id[1].output_paths:
        assert Path(path).exists(), f"manifest names a file that was not written: {path}"


@pytest.mark.parametrize("mode", ["continue", "error"])
def test_failed_component_is_recorded_and_directory_marked(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mode: str
) -> None:
    """A partial run must never look like a legitimate single-component run."""

    case = _case(TWO_COMPONENT_CASE)
    graph = _write_graph(case, tmp_path)
    out = tmp_path / f"out_{mode}"
    real = pipeline_module._generate_sdf_surface

    def flaky(nodes, points, segments, output_dir, graph_id=0, **kwargs):
        if graph_id == 1:
            raise RuntimeError("injected component-1 failure")
        return real(nodes, points, segments, output_dir, graph_id, **kwargs)

    monkeypatch.setattr(pipeline_module, "_generate_sdf_surface", flaky)

    if mode == "error":
        with pytest.raises(RuntimeError, match="injected component-1 failure"):
            pipeline_module.run_pipeline(
                graph,
                out,
                cfg=_legacy_config(PIPELINE_COMPONENT_FAILURE="error"),
                write_outputs=True,
                interactive=False,
            )
    else:
        surfaces, report = pipeline_module.run_pipeline(
            graph,
            out,
            cfg=_legacy_config(PIPELINE_COMPONENT_FAILURE="continue"),
            write_outputs=True,
            interactive=False,
            return_report=True,
        )
        # The surviving component must still be returned.
        assert sorted(surfaces) == [0]
        assert report.completed_components == (0,)
        assert report.failed_components == (1,)
        assert report.incomplete is True

    # Both modes must leave an unambiguous directory.
    manifest = json.loads((out / MANIFEST_NAME).read_text(encoding="utf-8"))
    assert manifest["incomplete"] is True
    assert manifest["requested_components"] == [0, 1]
    failed = [item for item in manifest["components"] if item["status"] == "failed"]
    assert [item["graph_id"] for item in failed] == [1]
    assert failed[0]["error_type"] == "RuntimeError"
    marker = out / INCOMPLETE_MARKER
    assert marker.exists()
    assert "missing:   [1]" in marker.read_text(encoding="utf-8")


def test_component_failure_mode_is_validated() -> None:
    with pytest.raises(ValueError, match="PIPELINE_COMPONENT_FAILURE"):
        _legacy_config(PIPELINE_COMPONENT_FAILURE="ignore")


# ── Branch coverage ──────────────────────────────────────────────────────────


def test_centreline_samples_convert_micrometres_to_millimetres() -> None:
    case = _case("straight")
    coords, radii, seg_index, point_ids = segment_centreline_samples(
        case.points, case.segments
    )
    assert len(coords) == len(radii) == len(seg_index) == len(point_ids)
    first = case.points[int(point_ids[0])]
    np.testing.assert_allclose(coords[0], np.asarray(first[:3]) / 1000.0)
    assert radii[0] == pytest.approx(first[3] / 1000.0)


def test_reconstructed_tube_covers_every_segment(tmp_path: Path) -> None:
    case = _case("junction_degree_3")
    graph = _write_graph(case, tmp_path)
    surfaces = pipeline_module.run_pipeline(
        graph, tmp_path / "o", cfg=_legacy_config(), write_outputs=False, interactive=False
    )

    report = branch_coverage(
        list(surfaces.values()), case.points, case.segments, containment=True
    )

    assert report.complete is True
    assert report.missing_segment_indices == ()
    assert report.fraction_beyond_two_radii == 0.0
    # Every sample should sit near its own wall, i.e. around one local radius.
    assert 0.5 < report.median_wall_distance_radii < 1.5
    assert report.component_count == 1
    assert report.largest_component_face_fraction == pytest.approx(1.0)


def test_missing_branch_is_reported_with_its_segment_id() -> None:
    """A surface that does not describe the graph must read as lost, not sparse."""

    case = _case("junction_degree_3")
    elsewhere = pv.Sphere(radius=0.5, center=(500.0, 500.0, 500.0)).triangulate()

    report = branch_coverage([elsewhere], case.points, case.segments)

    assert report.complete is False
    assert report.missing_segment_indices == tuple(range(len(case.segments)))
    assert report.missing_segment_ids == tuple(
        int(s.get("id", i)) for i, s in enumerate(case.segments)
    )
    assert report.fraction_beyond_two_radii == pytest.approx(1.0)


def test_no_surface_reports_total_loss_rather_than_raising() -> None:
    case = _case("straight")
    report = branch_coverage([], case.points, case.segments)
    assert report.complete is False
    assert report.component_count == 0
    assert report.largest_component_face_fraction == 0.0


def test_tiny_component_is_attributed_to_the_nearest_centreline_radius(
    tmp_path: Path,
) -> None:
    """Speckle attribution is how degenerate input radii are distinguished
    from a field or extractor defect."""

    case = _case("junction_degree_3")
    graph = _write_graph(case, tmp_path)
    surfaces = pipeline_module.run_pipeline(
        graph, tmp_path / "o", cfg=_legacy_config(), write_outputs=False, interactive=False
    )
    surface = surfaces[0]

    point_id = list(case.points)[5]
    anchor = np.asarray(case.points[point_id][:3], dtype=float) / 1000.0
    speck = pv.Tetrahedron(
        radius=1e-3, center=tuple(anchor + np.array([0.02, 0.0, 0.0]))
    ).triangulate()

    report = branch_coverage(
        [surface.merge(speck)], case.points, case.segments, tiny_component_faces=64
    )

    assert report.component_count == 2
    assert report.tiny_component_count == 1
    island = report.tiny_components[0]
    assert island.nearest_point_id == point_id
    assert island.nearest_radius_mm == pytest.approx(
        case.points[point_id][3] / 1000.0
    )
    assert island.distance_mm == pytest.approx(0.02, abs=5e-3)
    assert report.largest_component_face_fraction > 0.99


# ── Qualification separated from diagnosis ───────────────────────────────────


class _FakeLattice:
    """Minimal stand-in so the real-run path can be exercised without a mask."""

    def __init__(self) -> None:
        self.header = SimpleNamespace(
            origin_mm=np.zeros(3, dtype=float),
            spacing_mm=np.full(3, 0.066, dtype=float),
        )


def _seed_audit_cache(
    cache_dir: Path, case: dict[str, str], cfg, audits: list[dict]
) -> None:
    """Write an audit cache whose fingerprint matches ``case``."""

    graph_stat = Path(case["graph"]).stat()
    mask_stat = Path(case["segmentation"]).stat()
    payload = {
        "fingerprint": {
            "graph": str(Path(case["graph"]).resolve()),
            "graph_size": graph_stat.st_size,
            "graph_mtime_ns": graph_stat.st_mtime_ns,
            "segmentation": str(Path(case["segmentation"]).resolve()),
            "segmentation_size": mask_stat.st_size,
            "segmentation_mtime_ns": mask_stat.st_mtime_ns,
            "clearance_fraction": cfg.IMPLICIT_GEOMETRY_CLEARANCE_FRACTION,
        },
        "conflict_count": len(audits),
        "audits": audits,
    }
    (cache_dir / f"{case['name']}.conflict-audit.json").write_text(
        json.dumps(payload), encoding="utf-8"
    )


def test_real_run_reports_metrics_despite_an_unqualified_input(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """One unconfirmed site out of many must not suppress the reconstruction.

    It must still block the CFD label, and it must not be able to promote a
    profile — but the fidelity and completeness evidence has to exist.
    """

    case_obj = _case("junction_degree_3")
    graph = _write_graph(case_obj, tmp_path)
    mask = tmp_path / "mask.am"
    mask.write_bytes(b"not a real lattice")
    cache = tmp_path / "cache"
    cache.mkdir()
    scratch = tmp_path / "scratch"
    scratch.mkdir()

    cfg = candidate_config(
        "legacy_dense_meshlib", preprocessor="none", resolution=None
    )
    real_case = {
        "name": "FIXTURE_CASE",
        "graph": str(graph),
        "segmentation": str(mask),
    }
    _seed_audit_cache(
        cache,
        real_case,
        cfg,
        [
            {
                "conflict_index": 0,
                "classification": "mask_confirmed_contact",
                "segment_a": 0,
                "segment_b": 1,
            },
            # The single site that used to abort the whole case.
            {
                "conflict_index": 1,
                "classification": "radius_overestimation",
                "segment_a": 2,
                "segment_b": 1,
                "point_a_mm": [1.0, 2.0, 3.0],
            },
        ],
    )

    monkeypatch.setattr(
        benchmark, "read_amira_lattice", lambda *a, **k: _FakeLattice()
    )
    monkeypatch.setattr(
        benchmark,
        "segmentation_surface_metrics",
        lambda *a, **k: {"hd95_mm": 0.01, "reference_voxel_mm": 0.066},
    )
    monkeypatch.setattr(
        benchmark,
        "cross_section_area_metrics",
        lambda *a, **k: {"median_area_error_fraction": 0.02},
    )

    record = benchmark.run_real(
        {
            "kind": "real",
            "candidate": "legacy_dense_meshlib",
            "case": real_case["name"],
            "real_case": real_case,
            "preprocessor": "none",
            "resolution_factor": None,
            "cells_across_diameter": None,
            "scratch": str(scratch),
            "cache_dir": str(cache),
        }
    )

    # The reconstruction ran and produced evidence.
    assert record["status"] == "ok"
    assert record["mesh_faces"] > 0
    assert "surface_metrics" in record
    assert "coverage_metrics" in record
    assert "component_manifest" in record

    # ...but the input is still disqualified, and so is the run.
    assert record["input_qualified"] is False
    assert record["qualified"] is False
    assert "input_qualified" in record["qualification_blockers"]
    assert record["cfd_gate"]["input_qualified"] is False
    # Every other gate passed, proving the block came from the audit alone.
    assert record["cfd_gate"]["topology_ok"] is True
    assert record["cfd_gate"]["hd95_ok"] is True

    summary = record["conflict_summary"]
    assert summary["invalid_count"] == 1
    assert summary["invalid_classification_counts"] == {"radius_overestimation": 1}
    assert summary["classification_counts"]["mask_confirmed_contact"] == 1
    assert set(summary["invalid_segment_indices"]) == {1, 2}
    # The retired all-or-nothing flag must not linger under its old name.
    assert "input_valid" not in record


def test_conflict_summary_reports_both_index_and_source_segment_id() -> None:
    segments = [
        {"id": 101, "node1": 0, "node2": 1, "point_ids": []},
        {"id": 202, "node1": 1, "node2": 2, "point_ids": []},
        {"id": 303, "node1": 2, "node2": 3, "point_ids": []},
    ]
    audits = [
        {"conflict_index": 0, "classification": "mask_confirmed_contact",
         "segment_a": 0, "segment_b": 1},
        {"conflict_index": 1, "classification": "indeterminate",
         "segment_a": 1, "segment_b": 2},
    ]

    summary = benchmark._summarise_conflict_audit(audits, segments)

    assert summary["input_qualified"] is False
    assert summary["invalid_segment_indices"] == [1, 2]
    # Index and source id are different numbers; conflating them misattributes.
    assert summary["invalid_segment_ids"] == [202, 303]
    assert summary["invalid_graph_components"] == [0]


def test_qualified_input_summary_reports_no_blockers() -> None:
    segments = [{"id": 7, "node1": 0, "node2": 1, "point_ids": []}]
    audits = [
        {"conflict_index": 0, "classification": "mask_confirmed_contact",
         "segment_a": 0, "segment_b": 0}
    ]
    summary = benchmark._summarise_conflict_audit(audits, segments)
    assert summary["input_qualified"] is True
    assert summary["invalid_count"] == 0
    assert summary["input_failure"] is None


def test_spec_id_distinguishes_roi_from_full_graph_runs() -> None:
    """Without this, an ROI run silently overwrites a full-graph run's record."""

    base = {
        "kind": "real",
        "candidate": "graph_round_cone_vtk_htg",
        "case": "LADAF_2024_28",
        "cells_across_diameter": 12.0,
    }
    full = benchmark._spec_id(base)
    roi = benchmark._spec_id({**base, "roi": "deg5", "purpose": "diagnostic"})
    other_roi = benchmark._spec_id({**base, "roi": "trunk", "purpose": "diagnostic"})
    assert len({full, roi, other_roi}) == 3


def test_select_winner_rejects_diagnostic_and_roi_records() -> None:
    diagnostic = {
        "kind": "real",
        "candidate": "graph_round_cone_vtk_htg",
        "case": "LADAF_2024_28",
        "purpose": "diagnostic",
        "status": "ok",
        "qualified": True,  # deliberately inconsistent
    }
    winner, reason = benchmark._select_winner([diagnostic], ["graph_round_cone_vtk_htg"])
    assert winner is None
    assert "inconsistent records" in reason

    roi_only = {**diagnostic, "qualified": False, "roi": "deg5"}
    winner, reason = benchmark._select_winner([roi_only], ["graph_round_cone_vtk_htg"])
    assert winner is None
    assert "diagnostic-only" in reason


def test_coverage_summary_drops_the_per_segment_table() -> None:
    case = _case("straight")
    report = branch_coverage([], case.points, case.segments)
    summary = report.summary()
    assert "per_segment" not in summary
    assert "median_wall_distance_radii" in summary
    assert "per_segment" in report.to_dict()
