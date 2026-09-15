"""Smoke tests for the command-line interface."""

from skeleton_analysis.cli import main
from skeleton_analysis.io.amira import read_amira


def test_cli_info(test_am_path, capsys):
    rc = main(["info", str(test_am_path)])
    assert rc == 0
    out = capsys.readouterr().out
    assert "vertices:  148" in out
    assert "edges:     147" in out


def test_cli_roots(test_am_path, capsys):
    rc = main(["roots", str(test_am_path)])
    assert rc == 0
    assert "21" in capsys.readouterr().out


def test_cli_order(test_am_path, tmp_path):
    out = tmp_path / "ordered.am"
    rc = main(["order", str(test_am_path), str(out)])
    assert rc == 0
    g = read_amira(out)
    assert "strahler" in g.edge_fields
    assert "topo" in g.edge_fields
