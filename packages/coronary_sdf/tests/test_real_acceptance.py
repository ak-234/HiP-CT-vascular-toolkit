"""Explicit, opt-in slow acceptance tests for paired LADAF graph/mask data."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from coronary_sdf.amira_lattice import read_amira_lattice
from coronary_sdf.benchmark import run_real


pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(
        os.environ.get("CORONARY_SDF_RUN_SLOW") != "1",
        reason="set CORONARY_SDF_RUN_SLOW=1 to run real CFD acceptance cases",
    ),
]


def _cases():
    """Case manifest for the opt-in acceptance runs.

    Called at collection time by the parametrize below, so a missing manifest
    must yield no cases rather than raise -- these tests need real paired
    graph/mask data that is not in the repository. Point
    ``CORONARY_SDF_CASES`` at your own manifest; the checked-in
    ``benchmark_cases.example.json`` shows the shape and carries placeholder
    paths only.
    """
    override = os.environ.get("CORONARY_SDF_CASES")
    manifest = (
        Path(override) if override
        else Path(__file__).resolve().parents[1] / "benchmark_cases.example.json"
    )
    if not manifest.is_file():
        return []
    return json.loads(manifest.read_text(encoding="utf-8"))["cases"]


def test_ladaf28_hxbyterle_matches_tiff_slices(tmp_path: Path):
    np = pytest.importorskip("numpy")
    tifffile = pytest.importorskip("tifffile")
    case = _cases()[0]
    tiff_path = Path(case["segmentation"]).with_suffix(".tif")
    lattice = read_amira_lattice(
        case["segmentation"], "Labels", cache_dir=tmp_path / "cache"
    )
    with tifffile.TiffFile(tiff_path) as tif:
        for z in (0, lattice.nz // 2, lattice.nz - 1):
            np.testing.assert_array_equal(lattice.slice_z(z), tif.pages[z].asarray())


@pytest.mark.parametrize("case", _cases(), ids=lambda case: case["name"])
def test_real_case_meets_cfd_acceptance(case, tmp_path: Path):
    result = run_real(
        {
            "kind": "real",
            "candidate": os.environ.get(
                "CORONARY_SDF_ACCEPTANCE_CANDIDATE", "graph_round_cone_adaptive"
            ),
            "case": case["name"],
            "real_case": case,
            "preprocessor": os.environ.get(
                "CORONARY_SDF_ACCEPTANCE_PREPROCESSOR", "none"
            ),
            "resolution_factor": None,
            "cells_across_diameter": float(
                os.environ.get("CORONARY_SDF_ACCEPTANCE_CELLS", "12")
            ),
            "blend_fraction": 0.15,
            "blend_support": 4.0,
            "scratch": str(tmp_path / "scratch"),
            "cache_dir": str(tmp_path / "cache"),
            "diagnostic_dir": str(tmp_path / "diagnostics"),
        }
    )
    assert result["qualified"], result
