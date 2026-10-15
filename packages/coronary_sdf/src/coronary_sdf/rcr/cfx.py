"""Strict explicit-parameter adapter around the pinned SCAPIS CFX exporter."""
from contextlib import redirect_stdout
import hashlib
import json
import math
from pathlib import Path
import shutil

from . import cfx_def_export as legacy
from .ccl_template import generate_windkessel_ccl
from .network import positive
from .prepare import digest, write_json


def checkpoint(path, count, step_index, dt):
    lines = Path(path).read_text().splitlines()
    selected = None
    i = 0
    while i < len(lines):
        header = lines[i].split()
        if len(header) != 3:
            raise ValueError("Invalid restart checkpoint header")
        n, step = int(header[0]), int(header[1])
        interval = float(header[2].replace('D', 'E'))
        if i+n+1 >= len(lines):
            raise ValueError("Incomplete restart checkpoint; select a completed checkpoint file")
        rows = [line.split() for line in lines[i+1:i+n+1]]
        if lines[i+n+1].split() != ["COMMIT", str(step)]:
            raise ValueError("Checkpoint has no matching commit marker")
        if step == step_index:
            if selected is not None or n != count or not math.isclose(interval, dt, rel_tol=1e-6):
                raise ValueError("Duplicate checkpoint or incompatible count/time step")
            if [int(r[0]) for r in rows] != list(range(n)):
                raise ValueError("Checkpoint outlet index mismatch")
            selected = [(float(r[1].replace('D', 'E')), float(r[2].replace('D', 'E'))) for r in rows]
            if not all(math.isfinite(x) for pair in selected for x in pair):
                raise ValueError("Nonfinite checkpoint")
        i += n+2
    if selected is None:
        raise ValueError("Requested completed solver step is absent from checkpoint")
    return selected


def export(results, config, tree, method, output, dry_run=False, compile_sources=False):
    output = Path(output)
    cfg = config.get("cfx", {})
    selected = [r for r in results["results"] if r["tree_id"] == tree and r["method"] == method]
    if len(selected) != 1:
        raise ValueError("Select exactly one calibrated tree and resistance method")
    result = selected[0]
    p = results["physiology"]
    rows = result["outlets"]
    if not math.isclose(math.fsum(r["flow_m3_s"] for r in rows), result["target_flow_m3_s"], rel_tol=1e-7):
        raise ValueError("Outlet mean flows do not conserve the prescribed inlet flow")
    dt = positive(cfg.get("time_step_s", 0.001), "CFX time step")
    mapping = cfg.get("outlet_regions", {}).get(tree, {})
    inlet = cfg.get("inlet_regions", {}).get(tree)
    mesh = Path(cfg["mesh"]) if cfg.get("mesh") else None
    names = [mapping.get(r["outlet_id"], r.get("region_name")) for r in rows]
    if not inlet or any(not n for n in names) or len(set(names+[inlet])) != len(names)+1:
        raise ValueError("Explicit unique mesh region mappings required for inlet and every anatomical outlet")
    for name in names+[inlet]:
        if any(c in name for c in '\n\r#{}=,'):
            raise ValueError("Unsafe/ambiguous CCL region name")
    signature = hashlib.sha256(json.dumps([
        [r["outlet_id"], names[k], r["Rp_Pa_s_per_m3"], r["Rd_Pa_s_per_m3"], r["C_m3_per_Pa"], r["distal_pressure_pa"]]
        for k, r in enumerate(rows)], separators=(",", ":")).encode()).hexdigest()
    restored = None
    offset = int(cfg.get("restart_step", 0))
    if cfg.get("restart"):
        if cfg.get("restart_mapping_sha256") != signature or offset <= 0:
            raise ValueError("Restart requires original mapping hash and exact completed CFD restart_step")
        restored = checkpoint(cfg["restart"], len(rows), offset, dt)
    elif offset:
        raise ValueError("restart_step requires a capacitor checkpoint")
    outs = []
    for k, row in enumerate(rows):
        rp, rd, cap = [positive(row[key], key) for key in ("Rp_Pa_s_per_m3", "Rd_Pa_s_per_m3", "C_m3_per_Pa")]
        if not math.isclose(rp+rd, row["R_T_Pa_s_per_m3"], rel_tol=1e-10):
            raise ValueError("Explicit Rp + Rd must equal RT")
        pc, q = restored[k] if restored else (row["initial_pc_pa"], row["flow_m3_s"])
        outs.append(dict(k=k, ccl_name=row["outlet_id"], region_name=names[k],
            radius_mm=row["radius_m"]*1e3, area_mm2=row["area_m2"]*1e6,
            c=0., R_T=row["R_T_Pa_s_per_m3"], R1=rp, R2=rd, C=cap,
            pc0_pa=pc, pv_pa=row["distal_pressure_pa"], p0_pa=row["initial_outlet_pressure_pa"],
            q_i=q, q_total_rest=result["target_flow_m3_s"], fraction=row["flow_fraction"]))
    source = Path(__file__).parent/"fortran"
    shutil.copytree(source, output/"fortran")
    libraries = Path(cfg.get("user_lib_dir", output/"fortran"))
    pre = legacy.find_cfx_pre(Path(cfg["cfx_pre"]) if cfg.get("cfx_pre") else None)
    compiler = shutil.which("ifort") or shutil.which("ifx")
    source_hashes = {str(f.relative_to(source)): digest(f) for f in source.rglob("*.F")}
    build_error = None
    if compile_sources:
        libraries = output/"fortran"
        try:
            if pre is None:
                raise ValueError("CFX-Pre installation was not found")
            from .build import build_libraries
            build_libraries(libraries, pre, double=cfg.get("double_precision", False))
        except ValueError as exc:
            build_error = str(exc)
    # Old SCAPIS binaries have an incompatible state/argument contract.
    manifest = libraries/"build_manifest.json"
    compiled = False
    if manifest.is_file():
        built = json.loads(manifest.read_text())
        compiled = (built.get("source_sha256") == source_hashes
            and built.get("double_precision") == cfg.get("double_precision", False)
            and built.get("cfx_pre") == str(pre)
            and {Path(path).parts[0] for path in built.get("binary_sha256", {})} == {"windk", "jbpandq", "jbqpast"}
            and bool(built.get("binary_sha256")) and all(
                (libraries/path).is_file() and digest(libraries/path) == expected
                for path, expected in built.get("binary_sha256", {}).items()))
    status = dict(text_generated=False, resolved_text_generated=False, def_generated=False,
                  solver_validated=False, dry_run=dry_run, cfx_pre=str(pre) if pre else None,
                  compiler=compiler, compatible_libraries=compiled, source_sha256=source_hashes,
                  mapping_sha256=signature, mesh_sha256=digest(mesh) if mesh and mesh.is_file() else None,
                  network_sha256=results.get("network_sha256"), pressure_datum="CFX gauge, reference 1 atm")
    status.update(build_error=build_error, solver_available=bool(pre and pre.with_name("cfx5solve.exe").is_file()),
                  licence_status="not_verified")
    write_json(output/"export_status.json", status)
    wave = cfg.get("waveform", {"a": [result["target_flow_m3_s"]], "b": [0.]})
    if len(wave["a"]) != len(wave["b"]) or not math.isclose(wave["a"][0], result["target_flow_m3_s"], rel_tol=1e-8):
        raise ValueError("Waveform arrays must have matching lengths and the prescribed mean flow")
    params = dict(rho=p["density_kg_m3"], delta_t=dt, user_lib_dir=str(libraries),
        p_aortic_mmHg=p["arterial_mmhg"], p_distal_mmHg=p["distal_mmhg"],
        p_init_pa=p["distal_mmhg"]*133.322, period=positive(cfg.get("period_s", 1.), "period"),
        num_cycles=positive(cfg.get("cycles", 5), "cycles"), waveform=(wave["a"], wave["b"]),
        title=f"{tree} {method} explicit coronary RCR", source_file="calibration.json",
        save_last=dt, viscosity="newtonian", mu=p["viscosity_pa_s"], domain="Coronary",
        pulsatile=bool(cfg.get("pulsatile", False)), wall_location=cfg.get("wall_regions", ""),
        max_coeff_loops=cfg.get("max_coeff_loops", 20), min_coeff_loops=2, residual=1e-5,
        step_offset=offset, initial_time_s=offset*dt)
    generate_windkessel_ccl(outs, dict(location=inlet, name=inlet), params, output/"case.ccl")
    write_json(output/"export_outlets.json", [{**row, "cfx_outlet": f"Outlet{k}", "region_name": names[k],
               "initial_pc_pa": outs[k]["pc0_pa"], "initial_flow_m3_s": outs[k]["q_i"],
               "initial_outlet_pressure_pa": outs[k]["pc0_pa"]+outs[k]["R1"]*outs[k]["q_i"]}
              for k, row in enumerate(rows)])
    status["text_generated"] = True
    write_json(output/"export_status.json", status)
    if mesh is None or not mesh.is_file():
        status["blocked_reason"] = "Matching Fluent mesh unavailable; no resolved CCL/session/def generated"
        write_json(output/"export_status.json", status)
        if not dry_run:
            raise ValueError(status["blocked_reason"])
        return status
    table = legacy.read_zone_table(mesh)
    known = {legacy.region_name(name) for _, name in table.values()}
    walls = cfg.get("wall_regions") or ",".join(legacy.zones_of_type(table, legacy.WALL_ZONE_TYPE))
    if not walls or not set(walls.split(",")) <= known or not set(names+[inlet]) <= known:
        raise ValueError("Inlet/outlet/wall regions do not all match the imported mesh")
    if set(walls.split(",")) & set(names+[inlet]):
        raise ValueError("Wall overlaps an inlet or outlet")
    face_names = {legacy.region_name(name) for kind, name in table.values() if kind not in ("fluid", "interior", "solid")}
    if face_names != set(names+[inlet]+walls.split(",")):
        raise ValueError("Mesh has unclaimed boundary regions")
    argv = ["--msh", str(mesh), "--ccl", str(output/"case.ccl"), "--out", str(output/"case.def"),
            "--units", cfg["mesh_units"], "--wall-zone", walls, "--timeout", str(cfg.get("timeout_s", 300))]
    if pre:
        argv += ["--cfx-pre", str(pre)]
    blocked = not compiled or pre is None
    if dry_run or blocked:
        argv.append("--dry-run")
    with (output/"export.log").open("w", encoding="utf-8") as stream, redirect_stdout(stream):
        code = legacy.main(argv)
    status.update(resolved_text_generated=(output/"case_resolved.ccl").is_file(),
                  def_generated=code == 0 and (output/"case.def").is_file())
    if blocked:
        status["blocked_reason"] = "CFX-Pre or source-matched compiled user libraries unavailable"
    write_json(output/"export_status.json", status)
    if code or (blocked and not dry_run):
        raise ValueError(status.get("blocked_reason", "CFX-Pre failed; inspect export log"))
    return status
