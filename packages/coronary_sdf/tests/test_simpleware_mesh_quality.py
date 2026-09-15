import csv
import json

from coronary_sdf.simpleware_mesh_quality import collect_mesh_quality, write_mesh_quality_exports


class FakeMeshApi:
    AllVolumeElementTypes = "all-volume"
    Tetrahedron = "tet"
    Hexahedron = "hex"
    QuadraticTetrahedron = "qtet"
    QuadraticHexahedron = "qhex"
    Wedge = "wedge"
    Pyramid = "pyramid"
    Volume = "volume"
    EdgeLengthRatio = "edge-ratio"
    AngularSkew = "angular-skew"
    AspectRatio = "aspect-ratio"
    MinDihedralAngle = "min-dihedral"
    MaxDihedralAngle = "max-dihedral"
    VolumeSkew = "volume-skew"
    ShapeFactor = "shape-factor"
    Jacobian = "jacobian"
    CharacteristicLength = "characteristic-length"
    Count = "count"
    Sum = "sum"
    Mean = "mean"
    Minimum = "minimum"
    Maximum = "maximum"
    PastThresholdCount = "past"
    AllSurfacePrimitiveTypes = "all-surface"
    Triangle = "triangle"
    Quadrilateral = "quad"
    QuadraticTriangle = "qtriangle"
    QuadraticQuadrilateral = "qquad"
    SurfaceArea = "surface-area"
    SurfaceEdgeLengthRatio = "surface-edge-ratio"
    SurfaceDistortion = "surface-distortion"
    SurfaceInOutRatio = "surface-in-out-ratio"
    SurfaceEdgeLength = "surface-edge-length"


class FakeMesh:
    core_counts = {"all-volume": 10, "tet": 9, "pyramid": 1}
    boundary_counts = {"all-volume": 4, "wedge": 4}
    surface_counts = {"all-surface": 20, "triangle": 20}

    def IsVolumeElementDataAvailable(self):
        return True

    def IsBoundaryLayerVolumeElementDataAvailable(self):
        return True

    def IsSurfacePrimitiveDataAvailable(self):
        return True

    def GetVolumeElementCount(self, element_type):
        return self.core_counts.get(element_type, 0)

    def GetBoundaryLayerVolumeElementCount(self, element_type):
        return self.boundary_counts.get(element_type, 0)

    def GetSurfacePrimitiveCount(self, element_type):
        return self.surface_counts.get(element_type, 0)

    @staticmethod
    def _value(element_type, metric, value_type, counts):
        if metric == "angular-skew" and value_type == "maximum":
            raise RuntimeError("unsupported test pair")
        if value_type == "count":
            return counts.get(element_type, 0)
        if value_type == "sum":
            return 12.5
        if value_type == "mean":
            return 0.75
        if value_type == "minimum":
            if metric == "jacobian" and counts is FakeMesh.core_counts:
                return -0.01
            return 0.2
        if value_type == "maximum":
            return 1.25
        if value_type == "past":
            return 2
        raise AssertionError(value_type)

    def GetVolumeElementMetric(self, element_type, metric, value_type):
        return self._value(element_type, metric, value_type, self.core_counts)

    def GetBoundaryLayerVolumeElementMetric(
        self, element_type, metric, value_type
    ):
        return self._value(
            element_type, metric, value_type, self.boundary_counts
        )

    def GetSurfacePrimitiveMetric(self, element_type, metric, value_type):
        return self._value(element_type, metric, value_type, self.surface_counts)

    def GetThresholdValue(self, metric):
        return 0.05

    def GetBoundaryLayerThresholdValue(self, metric):
        return 0.05

    def GetSurfaceThresholdValue(self, metric):
        return 0.05


def test_collects_core_boundary_layer_and_surface_quality(tmp_path):
    payload = collect_mesh_quality(FakeMesh(), FakeMeshApi, "adaptive_l4")

    assert payload["availability"] == {
        "core_volume": True,
        "boundary_layer_volume": True,
        "surface": True,
    }
    assert payload["summary"]["metric_row_count"] == 60
    assert payload["summary"]["minimum_jacobian"] == -0.01
    assert payload["summary"]["negative_jacobian_detected"] is True
    row = next(
        row for row in payload["metrics"]
        if row["domain"] == "core_volume"
        and row["element_type"] == "all"
        and row["metric"] == "volume"
    )
    assert row["element_count"] == 10
    assert row["past_threshold_count"] == 2
    assert row["past_threshold_percent"] == 20.0
    partial = next(
        row for row in payload["metrics"]
        if row["domain"] == "core_volume"
        and row["element_type"] == "all"
        and row["metric"] == "angular_skew"
    )
    assert partial["status"] == "partial"
    assert "unsupported test pair" in partial["error"]

    json_path = tmp_path / "mesh_quality.json"
    csv_path = tmp_path / "mesh_quality.csv"
    write_mesh_quality_exports(payload, json_path, csv_path)
    loaded = json.loads(json_path.read_text(encoding="utf-8"))
    assert loaded["case_id"] == "adaptive_l4"
    with csv_path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 60
    assert {row["domain"] for row in rows} == {
        "core_volume", "boundary_layer_volume", "surface"
    }
