"""Export native Simpleware mesh-quality statistics without importing Simpleware.

The caller supplies ``simpleware.scripting.Mesh`` as ``mesh_api``.  Keeping the
API object injectable makes the collector testable with ordinary CPython while
the real calls are made by ConsoleSimpleware.
"""

import csv
import json
import math
from datetime import datetime, timezone
from pathlib import Path


VOLUME_ELEMENT_TYPES = (
    ("all", "AllVolumeElementTypes"),
    ("tetrahedron", "Tetrahedron"),
    ("hexahedron", "Hexahedron"),
    ("quadratic_tetrahedron", "QuadraticTetrahedron"),
    ("quadratic_hexahedron", "QuadraticHexahedron"),
    ("wedge", "Wedge"),
    ("pyramid", "Pyramid"),
)

VOLUME_METRICS = (
    ("volume", "Volume"),
    ("edge_length_ratio", "EdgeLengthRatio"),
    ("angular_skew", "AngularSkew"),
    ("aspect_ratio", "AspectRatio"),
    ("minimum_dihedral_angle", "MinDihedralAngle"),
    ("maximum_dihedral_angle", "MaxDihedralAngle"),
    ("volume_skew", "VolumeSkew"),
    ("shape_factor", "ShapeFactor"),
    ("jacobian", "Jacobian"),
    ("characteristic_length", "CharacteristicLength"),
)

SURFACE_ELEMENT_TYPES = (
    ("all", "AllSurfacePrimitiveTypes"),
    ("triangle", "Triangle"),
    ("quadrilateral", "Quadrilateral"),
    ("quadratic_triangle", "QuadraticTriangle"),
    ("quadratic_quadrilateral", "QuadraticQuadrilateral"),
)

SURFACE_METRICS = (
    ("surface_area", "SurfaceArea"),
    ("surface_edge_length_ratio", "SurfaceEdgeLengthRatio"),
    ("surface_distortion", "SurfaceDistortion"),
    ("surface_in_out_ratio", "SurfaceInOutRatio"),
    ("surface_edge_length", "SurfaceEdgeLength"),
)

VALUE_TYPES = (
    ("sample_count", "Count"),
    ("sum", "Sum"),
    ("mean", "Mean"),
    ("minimum", "Minimum"),
    ("maximum", "Maximum"),
    ("past_threshold_count", "PastThresholdCount"),
)

CSV_FIELDS = (
    "case_id",
    "domain",
    "element_type",
    "element_count",
    "metric",
    "sample_count",
    "sum",
    "mean",
    "minimum",
    "maximum",
    "threshold",
    "past_threshold_count",
    "past_threshold_percent",
    "status",
    "error",
)


def _finite_number(value):
    """Return a JSON-safe int/float, or ``None`` for non-finite values."""
    if isinstance(value, bool):
        return int(value)
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    if number.is_integer() and abs(number) <= 9007199254740991:
        return int(number)
    return number


def _safe_call(function, *args):
    try:
        return _finite_number(function(*args)), None
    except Exception as exc:  # Simpleware raises RuntimeError for invalid pairs.
        return None, "{}: {}".format(type(exc).__name__, exc)


def _collect_domain(
    mesh,
    mesh_api,
    domain,
    element_types,
    metrics,
    count_method_name,
    metric_method_name,
    threshold_method_name,
):
    rows = []
    count_method = getattr(mesh, count_method_name)
    metric_method = getattr(mesh, metric_method_name)
    threshold_method = getattr(mesh, threshold_method_name)
    for element_name, element_attr in element_types:
        element_value = getattr(mesh_api, element_attr)
        element_count, count_error = _safe_call(count_method, element_value)
        # The combined row and each actually present concrete type are useful;
        # absent concrete types add noise and many invalid metric combinations.
        if element_name != "all" and (element_count is None or element_count <= 0):
            continue
        for metric_name, metric_attr in metrics:
            metric_value = getattr(mesh_api, metric_attr)
            row = {
                "domain": domain,
                "element_type": element_name,
                "element_count": element_count,
                "metric": metric_name,
                "sample_count": None,
                "sum": None,
                "mean": None,
                "minimum": None,
                "maximum": None,
                "threshold": None,
                "past_threshold_count": None,
                "past_threshold_percent": None,
                "status": "ok",
                "error": count_error or "",
            }
            errors = []
            if count_error:
                errors.append(count_error)
            for field_name, value_attr in VALUE_TYPES:
                value, error = _safe_call(
                    metric_method,
                    element_value,
                    metric_value,
                    getattr(mesh_api, value_attr),
                )
                row[field_name] = value
                if error:
                    errors.append("{}={}".format(field_name, error))
            threshold, error = _safe_call(threshold_method, metric_value)
            row["threshold"] = threshold
            if error:
                errors.append("threshold={}".format(error))
            sample_count = row["sample_count"]
            past_count = row["past_threshold_count"]
            if sample_count not in (None, 0) and past_count is not None:
                row["past_threshold_percent"] = 100.0 * past_count / sample_count
            if errors:
                row["status"] = "partial" if any(
                    row[key] is not None
                    for key in ("mean", "minimum", "maximum", "sample_count")
                ) else "unsupported"
                row["error"] = " | ".join(errors)
            rows.append(row)
    return rows


def collect_mesh_quality(mesh, mesh_api, case_id=""):
    """Collect all X-2025.06 quality statistics exposed by ``Mesh``.

    Core volume and boundary-layer volume elements are deliberately separate,
    because Simpleware excludes prism-layer elements from the ordinary volume
    calls.  Surface-face statistics are exported as a third domain.
    """
    availability = {
        "core_volume": bool(mesh.IsVolumeElementDataAvailable()),
        "boundary_layer_volume": bool(
            mesh.IsBoundaryLayerVolumeElementDataAvailable()
        ),
        "surface": bool(mesh.IsSurfacePrimitiveDataAvailable()),
    }
    rows = []
    if availability["core_volume"]:
        rows.extend(_collect_domain(
            mesh, mesh_api, "core_volume", VOLUME_ELEMENT_TYPES,
            VOLUME_METRICS, "GetVolumeElementCount",
            "GetVolumeElementMetric", "GetThresholdValue",
        ))
    if availability["boundary_layer_volume"]:
        rows.extend(_collect_domain(
            mesh, mesh_api, "boundary_layer_volume", VOLUME_ELEMENT_TYPES,
            VOLUME_METRICS, "GetBoundaryLayerVolumeElementCount",
            "GetBoundaryLayerVolumeElementMetric",
            "GetBoundaryLayerThresholdValue",
        ))
    if availability["surface"]:
        rows.extend(_collect_domain(
            mesh, mesh_api, "surface", SURFACE_ELEMENT_TYPES,
            SURFACE_METRICS, "GetSurfacePrimitiveCount",
            "GetSurfacePrimitiveMetric", "GetSurfaceThresholdValue",
        ))
    for row in rows:
        row["case_id"] = str(case_id)

    jacobian_minima = [
        row["minimum"] for row in rows
        if row["domain"] in ("core_volume", "boundary_layer_volume")
        and row["element_type"] == "all"
        and row["metric"] == "jacobian"
        and row["minimum"] is not None
    ]
    negative_jacobian = bool(jacobian_minima and min(jacobian_minima) < 0.0)
    failed = sum(row["status"] != "ok" for row in rows)
    return {
        "schema_version": 1,
        "source": "Simpleware X-2025.06 scripting Mesh statistics API",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "case_id": str(case_id),
        "availability": availability,
        "summary": {
            "metric_row_count": len(rows),
            "partial_or_unsupported_row_count": failed,
            "minimum_jacobian": min(jacobian_minima) if jacobian_minima else None,
            "negative_jacobian_detected": negative_jacobian,
        },
        "notes": [
            "Core-volume statistics exclude boundary-layer elements.",
            "Boundary-layer statistics are reported independently.",
            "past_threshold_count uses the threshold configured by Simpleware.",
            "A partial/unsupported row does not invalidate other reported metrics.",
        ],
        "metrics": rows,
    }


def write_mesh_quality_exports(payload, json_path, csv_path):
    """Write a lossless JSON export and a flat CSV suitable for plotting."""
    json_path = Path(json_path)
    csv_path = Path(csv_path)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(
        json.dumps(payload, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    with csv_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(payload["metrics"])
    return json_path, csv_path
