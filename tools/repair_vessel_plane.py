"""Repair a reviewed tangent-window outlier and refresh affected CFD sections."""
import csv
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pyvista as pv


def main():
    local = Path(sys.argv[1]).resolve()
    edge_id = int(sys.argv[2])
    window = float(sys.argv[3])
    repo = Path(__file__).resolve().parents[1]
    cfg = json.loads((local / "mesh_sensitivity_study_adaptive_surface_01d88a48e419.json").read_text())
    for key, value in cfg["paths"].items():
        cfg["paths"][key] = value.replace("F:\\coronary_sdf\\", str(local) + "\\")
    output = Path(cfg["paths"]["output_dir"])
    backup = output / "analysis_backups" / (datetime.now().strftime("%Y%m%d_%H%M%S") + "_plane_repair")
    backup.mkdir(parents=True)
    shutil.copy2(local / "plane_sensitivity.py", backup / "plane_sensitivity.py")
    shutil.copy2(repo / "packages/coronary_sdf/src/coronary_sdf/plane_sensitivity.py", local / "plane_sensitivity.py")
    sys.path.insert(0, str(local))
    import plane_sensitivity as ps

    root = output / "plane_sensitivity/all_vessels"
    definitions = root / "geometry/all_vessel_plane_definitions.json"
    shutil.copy2(definitions, backup / definitions.name)
    for relative in ("reports", "plane_sensitivity/reports", "plane_sensitivity/all_vessels/reports"):
        if (output / relative).exists():
            shutil.copytree(output / relative, backup / relative)
    graph = json.loads((output / "meshes/global_l4/cropped_amira_graph.json").read_text())
    payload = json.loads(definitions.read_text())
    old = next(p for p in payload["planes"] if p["edge_id"] == edge_id)
    edge = next(e for e in graph["edges"] if e["edge_id"] == edge_id)
    surface = pv.read(cfg["paths"]["stl"]).triangulate().clean()
    candidate = ps._candidate(edge, window)
    section = ps.validate_section(surface, candidate["midpoint_mm"], candidate["normal"], 1.1)
    opening = json.loads((output / "poi/opening_terminal.json").read_text())
    bins = opening.get("radius_bin_edges_mm") or opening["radius_bins_mm"]
    corrected = {**old, **{k: v for k, v in candidate.items() if k != "edge"}, **section,
                 "radius_bin": ps._radius_bin(section["section_equivalent_radius_mm"], bins),
                 "endpoint_clearance_diameters": candidate["endpoint_clearance_mm"] / (2 * section["section_equivalent_radius_mm"]),
                 "tangent_window_local_diameters": window,
                 "repair_reason": "Local centreline kink produced an elongated oblique slice; widened tangent window at the same arc-length midpoint."}
    assert corrected["endpoint_clearance_diameters"] >= 2
    serial = lambda p: {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in p.items()}
    (backup / "plane_correction.json").write_text(json.dumps({"before": old, "after": serial(corrected)}, indent=2))
    print("Correcting", old["plane_id"], "radius", old["section_equivalent_radius_mm"], "->", corrected["section_equivalent_radius_mm"], flush=True)

    # Preview actual old/new slice loops together with the local wall.
    centre = np.asarray(old["midpoint_mm"])
    region = surface.clip_box(np.ravel(np.column_stack([centre - 3, centre + 3])), invert=False)
    plotter = pv.Plotter(off_screen=True, shape=(1, 2), window_size=(1500, 700))
    for index, plane in enumerate((old, corrected)):
        plotter.subplot(0, index)
        plotter.add_mesh(region, color="lightgray", opacity=0.65)
        plotter.add_mesh(ps._plane_patch(plane, 1), color="red", opacity=0.8)
        plotter.add_mesh(pv.lines_from_points(np.asarray(edge["points_mm"])), color="blue", line_width=2)
        plotter.add_text("Original" if index == 0 else "Corrected: 3-diameter tangent", font_size=14)
        plotter.camera_position = [centre + np.array([5, 7, 4]), centre, [0, 0, 1]]
    preview = repo / "cache/plane_287_correction.png"
    plotter.screenshot(str(preview)); plotter.close()

    new_planes = [corrected if p["edge_id"] == edge_id else p for p in payload["planes"]]
    # Extract only the changed plane. Commit the manifest after all four extracts succeed.
    replacements = []
    for level in ps.LEVELS:
        case = "global_" + level
        res = sorted((output / "cfx" / case).glob(case + "*.res"), key=lambda p: ("continue" in p.stem, p.stat().st_mtime))[-1]
        csv_path = root / "cfx" / case / "all_vessel_velocity_sections.csv"
        rows = list(csv.DictReader(csv_path.open()))
        shutil.copy2(csv_path, backup / (case + "_sections.csv"))
        shutil.copy2(csv_path.with_suffix(".cse"), backup / (case + "_sections.cse"))
        print("Re-extracting", case, old["plane_id"], flush=True)
        replacement = ps.extract_cfx_velocity_planes(cfg["executables"]["cfx_post"], res, [corrected], backup / case / "corrected_section.csv", False)[0]
        print("Area vs STL:", replacement["area_m2"] / (section["area_mm2"] * 1e-6), flush=True)
        replacements.append((csv_path, res, [replacement if r["plane_id"] == old["plane_id"] else r for r in rows]))
    for csv_path, res, rows in replacements:
        ps._write_csv(csv_path, rows)
        csv_path.with_suffix(".cse").write_text(ps._velocity_plane_session(res, csv_path, new_planes), encoding="utf-8")
    payload["planes"] = [serial(p) for p in new_planes]
    definitions.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(ps.run_plane_sensitivity(cfg, resume=True), flush=True)
    print("Backup:", backup, "Preview:", preview, flush=True)


if __name__ == "__main__":
    main()
