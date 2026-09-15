"""Headless Simpleware mesh-study job executed by ConsoleSimpleware X-2025.06.

``--input-value`` is a JSON job file.  The job always points at a disposable SIP
copy; the validated source SIP is copied by the outer launcher and is never saved
by this process.
"""

import json
import hashlib
import os
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

from simpleware.scripting import App, Doc, Mesh, Model, RatioSlicing


# Defaults to this file's own package directory, which is correct for a normal
# checkout or install. Set CORONARY_SDF_REPOSITORY_DIR when running from
# Simpleware's Scripting tab, where __file__ does not identify this package.
REPOSITORY_DIR = Path(
    os.environ.get("CORONARY_SDF_REPOSITORY_DIR")
    or Path(__file__).resolve().parent
)
STARTED = time.perf_counter()


def log(message):
    print(
        "[{} +{:8.1f}s] {}".format(
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            time.perf_counter() - STARTED,
            message,
        ),
        flush=True,
    )


def _find_part(document, model, regions):
    surface = regions._choose_surface(document)
    return surface, regions._choose_part(model, surface)


def _file_sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_surface_contract(document, job, regions):
    """Fail before geometry processing if a case SIP contains the wrong STL."""
    contract = job.get("surface_contract")
    if not contract:
        return None
    surface = regions._choose_surface(document)
    actual = {
        "surface_name": str(surface.GetName()),
        "vertices": int(surface.GetVertexCount()),
        "polygons": int(surface.GetPolygonCount()),
    }
    expected = {
        "surface_name": str(contract["surface_name"]),
        "vertices": int(contract["vertices"]),
        "polygons": int(contract["polygons"]),
    }
    if actual != expected:
        raise RuntimeError(
            "Selected Simpleware surface violates the persisted surface "
            "contract: expected {}, found {}".format(expected, actual)
        )
    expected_stl_hash = str(contract.get("candidate_stl_sha256", ""))
    if expected_stl_hash:
        actual_stl_hash = _file_sha256(Path(job["stl_path"]).resolve())
        if actual_stl_hash != expected_stl_hash:
            raise RuntimeError(
                "Meshing STL hash differs from the surface persistence audit"
            )
    log(
        "Persisted surface contract passed: {!r}, {:,} vertices, {:,} "
        "polygons.".format(
            actual["surface_name"], actual["vertices"], actual["polygons"]
        )
    )
    return actual


def _validate_inherent_cap_boundary_layer_exclusion(document, model, part, prefixes):
    """Validate Simpleware's clipping-cap boundary-layer topology.

    SetBoundaryLayerRegionByName only accepts native regions owned by ``part``
    (External, image boundaries, or neighbouring parts).  Finite clipping ROI
    names are deliberately not part regions: their cap faces are created later
    by mesh clipping, so they cannot seed prism layers.  Trying to disable a
    clipping ROI through that API raises "Region does not belong to the model".
    """
    clipping_caps = []
    for roi in document.GetRegionOfInterestVolumes(Doc.Clipping):
        name = str(roi.GetName())
        if any(name.startswith(prefix) for prefix in prefixes):
            clipping_caps.append(name)
    if not clipping_caps:
        raise RuntimeError(
            "No inlet/outlet/opening clipping regions were found for prism exclusion"
        )
    if not bool(model.IsBoundaryLayerExternalRegionGenerationSet(part)):
        raise RuntimeError(
            "Boundary layers are not enabled on the lumen part's External wall region"
        )
    return clipping_caps


def main():
    app = App.GetInstance()
    job_path = Path(str(app.GetInputValue() or "").strip().strip('"')).resolve()
    if not job_path.is_file():
        raise RuntimeError("Mesh-study job JSON not found: {}".format(job_path))
    job = json.loads(job_path.read_text(encoding="utf-8"))
    project_path = Path(job["case_sip"]).resolve()
    if not project_path.is_file():
        raise RuntimeError("Disposable case SIP not found: {}".format(project_path))

    repository = Path(job.get("repository_dir", str(REPOSITORY_DIR))).resolve()
    if str(repository) not in sys.path:
        sys.path.insert(0, str(repository))
    log("Opening disposable case: {}".format(project_path))
    document = app.OpenDocument(str(project_path))
    import simpleware_coronary_regions as regions

    surface_contract = _validate_surface_contract(document, job, regions)

    regions.AMIRA_SPATIAL_GRAPH_PATH = str(Path(job["amira_path"]).resolve())
    regions.STL_SURFACE_PATH = str(Path(job["stl_path"]).resolve())
    regions.BOUNDARY_CENTRELINE_SOURCES = ("amira",)
    regions.REFINEMENT_CENTRELINE_SOURCE = "amira"
    regions.CREATE_INLET_PLANES = True
    regions.CREATE_DISTAL_OPENING = True
    regions.UPDATE_GUI_VISIBILITY = False

    family = str(job["family"]).lower()
    if family == "adaptive":
        regions.CREATE_SMALL_VESSEL_REFINEMENTS = True
        regions.REFINEMENT_SELECTION_MODE = "all"
        regions.REFINEMENT_PRIMITIVE = "sphere"
        regions.REFINEMENT_USE_RADIUS_MESH_SIZE = True
        regions.REFINEMENT_ELEMENTS_ACROSS_DIAMETER = float(job["n_d"])
        regions.REFINEMENT_MIN_MESH_SIZE_MM = float(job["min_h_mm"])
        regions.REFINEMENT_MAX_MESH_SIZE_MM = float(job["max_h_mm"])
        regions.REFINEMENT_SPHERE_MAX_RADIUS_EXPANSION_FACTOR = float(
            job.get("maximum_sphere_expansion", 1.35)
        )
    elif family == "global":
        regions.CREATE_SMALL_VESSEL_REFINEMENTS = False
    else:
        raise ValueError("family must be 'global' or 'adaptive'")

    log("Rebuilding validated contacts and study refinement regions.")
    region_summary = regions.main()
    cropped_graph_path = Path(job["cropped_graph_json"]).resolve()
    cropped_graph_path.parent.mkdir(parents=True, exist_ok=True)
    cropped_graph = (region_summary or {}).get("cropped_amira_graph")
    created_counts = (region_summary or {}).get("created_counts", {})
    if not cropped_graph or not cropped_graph.get("edges"):
        raise RuntimeError("Region automation did not return the STL-cropped Amira graph")
    if created_counts.get("inlets") != 1 or created_counts.get("openings") != 1:
        raise RuntimeError(
            "Every study mesh requires exactly one inlet and one opening; got {}"
            .format(created_counts)
        )
    if created_counts.get("outlets", 0) < 1:
        raise RuntimeError("Study mesh has no fixed-flow outlets")
    if family == "adaptive" and created_counts.get("refinement_volumes", 0) < 1:
        raise RuntimeError("Adaptive case has no radius-aware refinement spheres")
    if family == "adaptive" and created_counts.get("refinement_rejected", 0):
        raise RuntimeError(
            "Adaptive refinement coverage is incomplete: {} of {} covering "
            "sphere(s) were rejected".format(
                created_counts["refinement_rejected"],
                created_counts.get("refinement_candidates", "unknown"),
            )
        )
    if family == "global" and created_counts.get("refinement_volumes", 0) != 0:
        raise RuntimeError("Global case retained scripted refinement volumes")
    cropped_graph_path.write_text(
        json.dumps(cropped_graph, indent=2) + "\n", encoding="utf-8"
    )
    model = document.GetActiveModel()
    if model.GetModelType() != Model.Cfd:
        raise RuntimeError("Active model is not CFD")
    _, part = _find_part(document, model, regions)

    global_h = float(job["global_h_mm"])
    model.SetTargetMinimumEdgeLength(global_h)
    model.SetMaximumEdgeLength(global_h)
    quality = job.get("mesh_quality", {})
    use_additional_improvement = bool(
        quality.get("use_additional_improvement", False)
    )
    model.SetUseAdditionalMeshQualityImprovement(use_additional_improvement)
    if use_additional_improvement:
        model.SetAdditionalMeshQualityImprovementAllowOffSurface(
            bool(quality.get("allow_off_surface", False))
        )
    log(
        "Additional mesh quality improvement: {}{}".format(
            "enabled" if use_additional_improvement else "disabled",
            " (off-surface motion disabled)" if use_additional_improvement else "",
        )
    )
    bl = job["boundary_layer"]
    width_ratio = float(bl["simpleware_width_ratio"])
    if not (0.0 < width_ratio <= 1.0):
        raise ValueError(
            "Simpleware boundary-layer width ratio must be in (0,1]; got {}"
            .format(width_ratio)
        )
    log(
        "Boundary layer: {} layers, consecutive growth {}, Simpleware width "
        "ratio {:.6g}.".format(
            int(bl["layers"]), float(bl["growth_ratio"]), width_ratio
        )
    )
    model.SetBoundaryLayer(
        part,
        float(bl["requested_total_thickness_mm"]),
        float(bl["maximum_channel_radius_ratio"]),
        float(bl.get("minimum_inlet_cell_quality", 0.1)),
        RatioSlicing(int(bl["layers"]), width_ratio),
        bool(bl.get("separate_boundary_layer", False)),
    )
    excluded = _validate_inherent_cap_boundary_layer_exclusion(
        document,
        model,
        part,
        (regions.INLET_PREFIX, regions.OUTLET_PREFIX, regions.OPENING_PREFIX),
    )
    log(
        "Validated {} clipping-generated cap region(s): these are not native "
        "part surfaces and are inherently excluded from prism-layer spawning."
        .format(len(excluded))
    )

    log("Generating mesh.")
    mesh = document.GenerateMesh()
    core_elements = int(mesh.GetVolumeElementCount(Mesh.AllVolumeElementTypes))
    prism_elements = int(
        mesh.GetBoundaryLayerVolumeElementCount(Mesh.AllVolumeElementTypes)
    )
    total_elements = core_elements + prism_elements
    if total_elements <= 0:
        raise RuntimeError("Simpleware returned an empty mesh")

    from simpleware_mesh_quality import (
        collect_mesh_quality,
        write_mesh_quality_exports,
    )
    stats_path = Path(job["mesh_stats_json"]).resolve()
    quality_json_path = Path(job.get(
        "mesh_quality_json", str(stats_path.with_name("mesh_quality.json"))
    )).resolve()
    quality_csv_path = Path(job.get(
        "mesh_quality_csv", str(stats_path.with_name("mesh_quality.csv"))
    )).resolve()
    log("Collecting native Simpleware mesh-quality metrics.")
    quality_payload = collect_mesh_quality(mesh, Mesh, job["case_id"])
    write_mesh_quality_exports(
        quality_payload, quality_json_path, quality_csv_path
    )
    log(
        "Exported {} quality rows; minimum Jacobian={}, negative={}.".format(
            quality_payload["summary"]["metric_row_count"],
            quality_payload["summary"]["minimum_jacobian"],
            quality_payload["summary"]["negative_jacobian_detected"],
        )
    )

    mesh_path = Path(job["fluent_mesh"]).resolve()
    mesh_path.parent.mkdir(parents=True, exist_ok=True)
    log("Exporting Fluent volume mesh: {}".format(mesh_path))
    document.ExportFluentVolume(str(mesh_path), False)
    if not mesh_path.is_file():
        raise RuntimeError("Fluent export was not created: {}".format(mesh_path))

    stats = {
        "case_id": job["case_id"],
        "family": family,
        "target_elements": int(job["target_elements"]),
        "core_elements": core_elements,
        "boundary_layer_elements": prism_elements,
        "total_elements": total_elements,
        "node_count_without_boundary_layer": int(mesh.GetNodeCount()),
        "mesh_seconds": int(mesh.GetTimeToGenerate()),
        "peak_memory_bytes": int(mesh.GetPeakMemoryUsage()),
        "global_h_mm": global_h,
        "n_d": job.get("n_d"),
        "boundary_layer": bl,
        "mesh_quality": {
            "use_additional_improvement": use_additional_improvement,
            "allow_off_surface": bool(quality.get("allow_off_surface", False)),
        },
        "mesh_quality_exports": {
            "json": str(quality_json_path),
            "csv": str(quality_csv_path),
        },
        "mesh_quality_summary": quality_payload["summary"],
        "excluded_boundary_layer_regions": excluded,
        "boundary_layer_cap_policy": "inherent_clipping_cap_exclusion",
        "cropped_graph_json": str(cropped_graph_path),
        "cropped_graph_edge_count": len(cropped_graph["edges"]),
        "created_region_counts": created_counts,
        "surface_contract": surface_contract,
    }
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    stats_path.write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")
    log("Mesh has {:,} elements; saving disposable SIP.".format(total_elements))
    document.Save()
    log("Mesh-study job complete.")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        log("ERROR: {}: {}".format(type(exc).__name__, exc))
        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        raise
