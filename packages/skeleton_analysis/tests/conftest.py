"""Shared pytest fixtures and path helpers."""

from pathlib import Path

import numpy as np
import pytest

DATA_DIR = Path(__file__).parent / "data"


@pytest.fixture(scope="session")
def test_am_path() -> Path:
    """Path to the real Avizo spatial graph fixture (Initial_Ordering/Test.am)."""
    return DATA_DIR / "Test.am"


def make_am_text(
    vertex_coords,
    edge_connectivity,
    num_edge_points,
    point_coords,
    thickness,
    *,
    header="# AmiraMesh 3D ASCII 2.0",
) -> str:
    """Build a minimal valid ``.am`` text from arrays (for synthetic tests)."""
    vertex_coords = np.asarray(vertex_coords, dtype=float)
    edge_connectivity = np.asarray(edge_connectivity, dtype=int)
    num_edge_points = np.asarray(num_edge_points, dtype=int)
    point_coords = np.asarray(point_coords, dtype=float)
    thickness = np.asarray(thickness, dtype=float)

    lines = [
        header,
        "",
        "",
        f"define VERTEX {len(vertex_coords)}",
        f"define EDGE {len(edge_connectivity)}",
        f"define POINT {len(point_coords)}",
        "",
        "Parameters {",
        '    ContentType "HxSpatialGraph"',
        "}",
        "",
        "VERTEX { float[3] VertexCoordinates } @1",
        "EDGE { int[2] EdgeConnectivity } @2",
        "EDGE { int NumEdgePoints } @3",
        "POINT { float[3] EdgePointCoordinates } @4",
        "POINT { float thickness } @5",
        "",
        "# Data section follows",
        "@1",
    ]
    lines += [" ".join(f"{v:.6f}" for v in row) for row in vertex_coords]
    lines += ["", "@2"]
    lines += [f"{a} {b}" for a, b in edge_connectivity]
    lines += ["", "@3"]
    lines += [str(int(n)) for n in num_edge_points]
    lines += ["", "@4"]
    lines += [" ".join(f"{v:.6f}" for v in row) for row in point_coords]
    lines += ["", "@5"]
    lines += [f"{v:.6f}" for v in thickness]
    lines += [""]
    return "\n".join(lines) + "\n"


@pytest.fixture
def synthetic_am(tmp_path):
    """Factory that writes a synthetic ``.am`` file and returns its path."""

    def _make(**kwargs):
        text = make_am_text(**kwargs)
        p = tmp_path / "synthetic.am"
        p.write_text(text, encoding="latin-1")
        return p

    return _make
