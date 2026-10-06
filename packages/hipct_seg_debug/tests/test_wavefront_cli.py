"""``connect --wavefront`` at the command line.

The same file rules as ``--geodesic`` -- paired outputs, dry run by default --
plus the two things that are this connector's own: the refinement runs before
anything is proposed, and the repaired edges carry the wavefront provenance.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from hipct_seg_debug.edit.__main__ import build_parser, cmd_connect
from hipct_seg_debug.edit.adapter import read_triple
from hipct_seg_debug.edit.amira_write import write_spatial_graph
from hipct_seg_debug.edit.graphmodel import EditableGraph
from hipct_seg_debug.edit.reconnect.geodesic import apply as apply_mod
from hipct_seg_debug.edit.reconnect.geodesic import components
from hipct_seg_debug.edit.reconnect.wavefront import agd_available
from hipct_seg_debug.rle_write import write_lattice

from .conftest_geodesic import SHAPE, SPACING, broken_graph, make_frame, ribbon


@pytest.fixture
def dataset(tmp_path):
    """A broken ribbon vessel written out as a real ``.am`` pair."""
    frame = make_frame(SHAPE)
    volume = (ribbon(SHAPE, 6, 2, 5, 28) | ribbon(SHAPE, 6, 2, 30, 55)).astype(np.uint8)
    dims = np.array([SHAPE[2], SHAPE[1], SHAPE[0]])
    bbox = np.empty(6)
    bbox[0::2] = 0.0
    bbox[1::2] = (dims - 1) * SPACING
    seg = tmp_path / "seg.am"
    write_lattice(seg, volume, bbox)
    graph = broken_graph(frame, (5, 28), (30, 55))
    path = tmp_path / "graph.am"
    write_spatial_graph(graph.to_spatial_graph(), path)
    return {"graph": path, "seg": seg, "dir": tmp_path, "volume": volume}


def _args(dataset, **overrides):
    argv = ["connect", str(dataset["graph"]), "--wavefront",
            "--seg", str(dataset["seg"]), "--voxel-um", str(SPACING),
            "--alternatives", "2"]
    for key, value in overrides.items():
        flag = "--" + key.replace("_", "-")
        if value is True:
            argv.append(flag)
        elif value not in (None, False):
            argv.extend([flag, str(value)])
    return build_parser().parse_args(argv)


# ------------------------------------------------------------------ the guards


def test_flags_are_registered_with_their_defaults(dataset):
    args = _args(dataset)
    assert args.wavefront and not args.geodesic and not args.dpc
    assert args.refine_method == "centroid-coherent"
    assert args.engine == "auto" and args.stencil == 26
    assert args.reach_radii is None and args.lookahead_cone_deg is None
    assert not args.explore_open_ends and not args.no_chain_fallback


def test_wavefront_and_geodesic_are_mutually_exclusive(dataset):
    with pytest.raises(SystemExit, match="mutually exclusive"):
        cmd_connect(_args(dataset, geodesic=True))


def test_wavefront_and_dpc_are_mutually_exclusive(dataset):
    with pytest.raises(SystemExit, match="mutually exclusive"):
        cmd_connect(_args(dataset, dpc=True, raw="somewhere"))


def test_writing_the_graph_alone_is_refused(dataset):
    with pytest.raises(SystemExit, match="--out-seg"):
        cmd_connect(_args(dataset, out=str(dataset["dir"] / "out.am")))


def test_writing_the_mask_alone_is_refused(dataset):
    with pytest.raises(SystemExit, match="both"):
        cmd_connect(_args(dataset, out_seg=str(dataset["dir"] / "out-seg.am")))


def test_out_seg_without_a_connector_names_both(dataset):
    argv = ["connect", str(dataset["graph"]), "--seg", str(dataset["seg"]),
            "--out-seg", str(dataset["dir"] / "x.am")]
    with pytest.raises(SystemExit, match="--geodesic or --wavefront"):
        cmd_connect(build_parser().parse_args(argv))


@pytest.mark.skipif(agd_available(), reason="agd happens to be installed")
def test_asking_for_agd_without_it_is_a_clear_refusal(dataset):
    with pytest.raises(SystemExit, match="agd"):
        cmd_connect(_args(dataset, engine="agd"))


# ---------------------------------------------------------------------- runs


def test_dry_run_refines_reports_and_writes_nothing(dataset, capsys):
    assert cmd_connect(_args(dataset, refine_method="laplacian")) == 0
    printed = capsys.readouterr().out
    assert "refining the centreline (laplacian" in printed
    assert "refined with laplacian" in printed
    assert "free end(s) profiled" in printed
    assert "engine lattice" in printed
    assert "dry run" in printed
    assert not list(dataset["dir"].glob("out*.am"))


def test_refinement_can_be_skipped(dataset, capsys):
    assert cmd_connect(_args(dataset, refine_method="none")) == 0
    printed = capsys.readouterr().out
    assert "refinement skipped" in printed
    assert "refining the centreline" not in printed


def test_a_full_run_writes_a_consistent_pair_with_wavefront_provenance(dataset):
    out_graph = dataset["dir"] / "out.am"
    out_seg = dataset["dir"] / "out-seg.am"
    decisions = dataset["dir"] / "decisions.json"
    assert cmd_connect(_args(dataset, refine_method="none", out=str(out_graph),
                             out_seg=str(out_seg), decisions_json=str(decisions))) == 0
    assert out_graph.exists() and out_seg.exists()

    from hipct_seg_debug import amira, rle

    info = amira.read_lattice_header(out_seg)
    lattice = rle.ByteRLELattice(out_seg, info.fields["Labels"], info.dims)
    written = np.stack([lattice.slice_z(k) for k in range(lattice.nz)])
    assert components.from_array(written > 0).n == 1
    assert (written > 0).sum() > (dataset["volume"] > 0).sum()

    graph = EditableGraph(read_triple(out_graph))
    assert len(graph.components()) == 1
    origins = {int(s.get(apply_mod.ORIGIN_FIELD, apply_mod.ORIGINAL)) for s in graph.segments}
    assert apply_mod.ORIGINAL in origins
    assert origins & {apply_mod.WAVEFRONT, apply_mod.RESKELETONISED}
    assert apply_mod.GEODESIC not in origins

    document = json.loads(decisions.read_text(encoding="utf-8"))
    assert document["arguments"]["wavefront"] is True
    record = next(c for c in document["candidates"] if c["decision"]["accepted"])
    assert record["evidence"]["strategy"] == "dual_front"


def test_tunables_reach_the_parameters(dataset, monkeypatch):
    from hipct_seg_debug.edit.reconnect import wavefront

    seen = {}
    real = wavefront.plan

    def spy(*args, **kwargs):
        seen["params"] = kwargs["params"]
        return real(*args, **kwargs)

    monkeypatch.setattr(wavefront, "plan", spy)
    cmd_connect(_args(dataset, refine_method="none", reach_radii=25, keypoint_step=3,
                      lookahead_cone_deg=20, anisotropy_ratio=4, stencil=98,
                      max_unsupported_factor=6, explore_open_ends=True,
                      no_chain_fallback=True, refine_workers=2))
    p = seen["params"]
    assert p.reach_radii == 25 and p.keypoint_step_major == 3
    assert p.lookahead_cone_deg == 20 and p.anisotropy.max_ratio == 4
    assert p.stencil == 98 and p.max_unsupported_factor == 6
    assert p.explore_open_ends and not p.chain_fallback
    assert p.refine_workers == 2 and p.refine_method == "none"


def test_the_shared_unsupported_default_is_the_package_default(dataset, monkeypatch):
    from hipct_seg_debug.edit.reconnect import wavefront

    seen = {}
    monkeypatch.setattr(wavefront, "plan",
                        lambda *a, **k: seen.setdefault("p", k["params"]) and
                        wavefront.route.plan(*a, **k))
    cmd_connect(_args(dataset, refine_method="none"))
    assert seen["p"].max_unsupported_factor == wavefront.route.MAX_UNSUPPORTED_FACTOR
