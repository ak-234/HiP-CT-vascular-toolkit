"""Replace one surface in a disposable SIP with an accepted remeshed STL."""

import json
import hashlib
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

from simpleware.scripting import App


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


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _surface_audit(surface):
    return {
        "surface_name": str(surface.GetName()),
        "vertices": int(surface.GetVertexCount()),
        "polygons": int(surface.GetPolygonCount()),
        "has_errors": bool(surface.HasErrors()),
        "has_warnings": bool(surface.HasWarnings()),
        "is_open": bool(surface.IsOpenSurface()),
    }


def main():
    app = App.GetInstance()
    job_path = Path(str(app.GetInputValue() or "").strip().strip('"')).resolve()
    job = json.loads(job_path.read_text(encoding="utf-8"))
    project = Path(job["project_sip"]).resolve()
    source_project = Path(job.get("source_sip", project)).resolve()
    stl = Path(job["candidate_stl"]).resolve()
    report_path = Path(job["result_json"]).resolve()
    if not source_project.is_file() or not stl.is_file():
        raise RuntimeError("Source SIP or candidate STL is missing")
    if source_project == project:
        raise RuntimeError(
            "Surface persistence requires project_sip to differ from source_sip; "
            "the validated source is never overwritten"
        )
    if project.exists():
        raise RuntimeError("Refusing to overwrite an existing output SIP: {}".format(project))

    repository = Path(job["repository_dir"]).resolve()
    if str(repository) not in sys.path:
        sys.path.insert(0, str(repository))
    import simpleware_coronary_regions as regions

    log("Opening immutable source SIP: {}".format(source_project))
    document = app.OpenDocument(str(source_project))
    surface = regions._choose_surface(document)
    old_name = str(surface.GetName())
    old_vertices = int(surface.GetVertexCount())
    old_polygons = int(surface.GetPolygonCount())
    log(
        "Replacing surface {!r} ({:,} vertices, {:,} polygons) with {}."
        .format(old_name, old_vertices, old_polygons, stl)
    )
    # doFixing=False: the candidate has already passed explicit topology,
    # collision, and distance validation.  An implicit repair here would make
    # the persisted Simpleware geometry differ from the measured STL.
    surface.ImportStlFile(str(stl), False, 1.0, False)
    # ImportStlFile adopts the STL filename as the surface name.  Restore the
    # original name so the existing model part remains stable and so a model
    # with multiple parts cannot silently bind to the wrong surface.
    if str(surface.GetName()) != old_name:
        surface.SetName(old_name, False)
    new_vertices = int(surface.GetVertexCount())
    new_polygons = int(surface.GetPolygonCount())
    has_errors = bool(surface.HasErrors())
    has_warnings = bool(surface.HasWarnings())
    is_open = bool(surface.IsOpenSurface())
    log(
        "Imported {:,} vertices and {:,} polygons; errors={}, warnings={}, "
        "open={}.".format(
            new_vertices, new_polygons, has_errors, has_warnings, is_open
        )
    )
    if has_errors or has_warnings or is_open:
        raise RuntimeError(
            "Simpleware rejected the remeshed surface integrity: errors={}, "
            "warnings={}, open={}".format(has_errors, has_warnings, is_open)
        )
    if new_polygons <= 0 or new_polygons >= old_polygons:
        raise RuntimeError(
            "Replacement did not reduce the persisted polygon count ({} -> {})"
            .format(old_polygons, new_polygons)
        )

    modified_after_import = bool(document.IsModified())
    result = {
        "source_sip": str(source_project),
        "project_sip": str(project),
        "candidate_stl": str(stl),
        "candidate_stl_sha256": _sha256(stl),
        "surface_name": old_name,
        "old_vertices": old_vertices,
        "old_polygons": old_polygons,
        "new_vertices": new_vertices,
        "new_polygons": new_polygons,
        "has_errors": has_errors,
        "has_warnings": has_warnings,
        "is_open": is_open,
        "document_modified_after_import": modified_after_import,
        "persisted_reopen_verified": False,
    }
    log(
        "Document modified flag after in-place import: {}. Saving to a "
        "distinct SIP with SaveAs so serialization cannot be skipped."
        .format(modified_after_import)
    )
    project.parent.mkdir(parents=True, exist_ok=True)
    document.SaveAs(str(project))
    document.Close()
    if not project.is_file() or project.stat().st_size <= 0:
        raise RuntimeError("SaveAs did not create the output SIP: {}".format(project))

    log("Reopening the saved SIP for an independent persistence check.")
    reopened = app.OpenDocument(str(project))
    persisted_surface = regions._choose_surface(reopened)
    persisted = _surface_audit(persisted_surface)
    log(
        "Reopened surface {!r}: {:,} vertices, {:,} polygons; errors={}, "
        "warnings={}, open={}.".format(
            persisted["surface_name"], persisted["vertices"],
            persisted["polygons"], persisted["has_errors"],
            persisted["has_warnings"], persisted["is_open"],
        )
    )
    if persisted["surface_name"] != old_name:
        raise RuntimeError("Persisted surface name changed unexpectedly")
    if persisted["vertices"] != new_vertices or persisted["polygons"] != new_polygons:
        raise RuntimeError(
            "Surface replacement was not persisted (expected {:,}/{:,}, "
            "reopened {:,}/{:,} vertices/polygons)".format(
                new_vertices, new_polygons,
                persisted["vertices"], persisted["polygons"],
            )
        )
    if persisted["has_errors"] or persisted["has_warnings"] or persisted["is_open"]:
        raise RuntimeError("Reopened replacement surface failed integrity checks")
    reopened.Close()

    result["persisted_reopen_verified"] = True
    result["persisted_surface"] = persisted
    result["project_sip_sha256"] = _sha256(project)
    result["project_sip_bytes"] = int(project.stat().st_size)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    log("Persisted surface replacement verified: {}".format(report_path))


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise
