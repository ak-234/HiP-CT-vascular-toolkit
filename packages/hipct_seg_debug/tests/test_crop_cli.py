"""The `crop` command line and its sidecar.

What is pinned here is about *files*, not about the rules -- those live in
`test_crop.py`. Three properties earn their keep:

* a **dry run still writes the sidecar**. Its product is the record of what would be
  dropped, which is precisely what a dry run is for;
* the sidecar **reproduces its own crop** when it is the only argument given, and a
  flag on the command line overrides it;
* the written graph **keeps the source's ``Parameters`` block**, which holds the
  ``TransformationMatrix``. Lose it and the file opens fine, in the wrong place.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from hipct_seg_debug.amira import read_spatial_graph
from hipct_seg_debug.edit import crop
from hipct_seg_debug.edit.__main__ import build_parser, cmd_crop
from hipct_seg_debug.edit.amira_write import write_spatial_graph

from .test_crop import EDGES, NODES, ordered_tree


def _parameters(path) -> str:
    """The source's ``Parameters { ... }`` block, brace-counted like the writer's."""
    text = path.read_text(encoding="latin-1")
    start = text.find("Parameters {")
    depth, i = 0, start
    while i < len(text):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                break
        i += 1
    return text[start:i + 1]


@pytest.fixture
def graph_file(tmp_path):
    path = tmp_path / "tree.am"
    write_spatial_graph(ordered_tree().to_spatial_graph(), path)
    return path


def _run(argv) -> int:
    return cmd_crop(build_parser().parse_args(["crop", *[str(a) for a in argv]]))


def test_a_dry_run_writes_the_sidecar_but_not_the_graph(graph_file, tmp_path):
    sidecar = tmp_path / "c.json"
    out = tmp_path / "cropped.am"

    assert _run([graph_file, "--min-strahler", 2, "--crop-json", sidecar]) == 0
    assert sidecar.exists(), "a dry run's product is the record of what it would drop"
    assert not out.exists()

    document = json.loads(sidecar.read_text(encoding="utf-8"))
    assert document["schema"] == crop.SCHEMA
    assert document["selection"]["n_dropped"] > 0
    assert document["result"]["out"] is None


def test_ratio_and_ratio_denominator_are_mutually_exclusive(graph_file):
    assert _run([graph_file, "--ratio", 0.2, "--ratio-denominator", 5]) == 2


@pytest.fixture
def sidecar_with_vessel(graph_file, tmp_path):
    """A sidecar naming the LAD, as the Crop tab would leave one behind."""
    from hipct_seg_debug.edit.adapter import read_triple
    from hipct_seg_debug.edit.graphmodel import EditableGraph

    graph = EditableGraph(read_triple(graph_file))
    vessels = {"LAD": {1, 2}}
    plan = crop.plan(graph, crop.Rule(), vessels=vessels)
    path = tmp_path / "c.json"
    crop.write(path, crop.document(graph, plan, crop.Rule(), source=graph_file))
    return path


def test_the_denominator_is_inverted(graph_file, sidecar_with_vessel):
    _run([graph_file, "--ratio-denominator", 4, "--crop-json", sidecar_with_vessel])

    rule = json.loads(sidecar_with_vessel.read_text(encoding="utf-8"))["rule"]
    assert rule["ratio"] == pytest.approx(0.25)
    assert rule["ratio_denominator"] == pytest.approx(4.0)


def test_a_ratio_with_no_named_vessel_exits(graph_file):
    with pytest.raises(SystemExit, match="named main vessel"):
        _run([graph_file, "--ratio", 0.25])


def test_the_named_vessel_comes_back_out_of_the_sidecar(graph_file, sidecar_with_vessel):
    _run([graph_file, "--ratio-denominator", 4, "--crop-json", sidecar_with_vessel])

    document = json.loads(sidecar_with_vessel.read_text(encoding="utf-8"))
    assert set(document["vessels"]) == {"LAD"}
    assert document["vessels"]["LAD"]["ostium"]["radius_um"] == pytest.approx(700.0)
    # The LAD's own 0.25 x 700 = 175 um threshold took the 100 um twig.
    assert [t["rule"] for t in document["selection"]["takeoffs"]] == ["ratio"]


def test_replay_without_a_sidecar_refuses(graph_file, tmp_path):
    with pytest.raises(SystemExit, match="needs an existing"):
        _run([graph_file, "--replay", "--crop-json", tmp_path / "absent.json"])


def test_a_missing_strahler_field_exits_rather_than_dropping_everything(tmp_path):
    from .conftest_geometry import graph_from

    path = tmp_path / "plain.am"
    write_spatial_graph(graph_from(NODES, EDGES).to_spatial_graph(), path)
    with pytest.raises(SystemExit, match="no Strahler order"):
        _run([path, "--min-strahler", 2])


def test_the_sidecar_reproduces_its_own_crop(graph_file, tmp_path):
    sidecar = tmp_path / "c.json"
    _run([graph_file, "--min-strahler", 2, "--crop-json", sidecar])
    first = json.loads(sidecar.read_text(encoding="utf-8"))

    # Only the sidecar this time: the rule comes back out of it.
    _run([graph_file, "--crop-json", sidecar])
    second = json.loads(sidecar.read_text(encoding="utf-8"))

    assert second["rule"]["min_strahler"] == 2
    assert (second["selection"]["dropped_seg_keys"]
            == first["selection"]["dropped_seg_keys"])


def test_a_flag_overrides_the_sidecars_rule(graph_file, tmp_path):
    sidecar = tmp_path / "c.json"
    _run([graph_file, "--min-strahler", 2, "--crop-json", sidecar])
    _run([graph_file, "--min-strahler", 3, "--crop-json", sidecar])

    document = json.loads(sidecar.read_text(encoding="utf-8"))
    assert document["rule"]["min_strahler"] == 3


def test_the_written_graph_keeps_the_parameters_block(graph_file, tmp_path):
    out = tmp_path / "cropped.am"
    _run([graph_file, "--min-strahler", 2, "--no-reorder", "--out", out])

    assert out.exists()
    assert _parameters(out) == _parameters(graph_file)
    assert read_spatial_graph(out).n_edge < read_spatial_graph(graph_file).n_edge


def test_no_reorder_keeps_the_stored_orders(graph_file, tmp_path):
    out = tmp_path / "cropped.am"
    _run([graph_file, "--min-strahler", 2, "--no-reorder", "--out", out])

    written = read_spatial_graph(out)
    # `ordered_tree` puts order 1 on the twig alone, so what survives is 2 and 3 --
    # stale for the cropped tree, which is exactly what --no-reorder asks for.
    assert sorted(set(np.asarray(written.edge_attrs["strahler"]).ravel().tolist())) == [2, 3]
    assert "topo" not in written.edge_attrs


def test_the_default_recomputes_the_orders(graph_file, tmp_path):
    pytest.importorskip("skeleton_analysis")
    out = tmp_path / "cropped.am"
    _run([graph_file, "--min-strahler", 2, "--out", out])

    written = read_spatial_graph(out)
    orders = np.asarray(written.edge_attrs["strahler"]).ravel()
    # Re-derived for the tree that is actually being written, so its leaves are 1 again.
    assert orders.min() == 1
    assert "topo" in written.edge_attrs


def test_the_report_csv_has_one_row_per_takeoff(graph_file, tmp_path):
    report = tmp_path / "c.csv"
    _run([graph_file, "--min-strahler", 2, "--report-csv", report])

    rows = report.read_text(encoding="utf-8").strip().splitlines()
    assert rows[0].split(",") == list(crop.CSV_COLUMNS)
    assert len(rows) > 1
    assert all(row.split(",")[2] == "strahler" for row in rows[1:])


def test_crop_runs_in_process():
    """It reads only the .am, so the GUI streams its log instead of forking."""
    from hipct_seg_debug.runner import INPROC, mode_for

    assert "crop" in INPROC
    assert mode_for(["crop", "graph.am"]) == "inproc"
