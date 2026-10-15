# Adapted from SCAPIS revision 24f36a58b9e7a0adea605a39db6d768cbf600ee4; see THIRD_PARTY.md.
#!/usr/bin/env python3
"""cfx_def_export.py - build the CFX solver input (``.def``) from a pipeline CCL and its mesh.

The last pipeline step. ``windkessel_BC.py`` (and the giessen / mass writers before it) produce
a CCL that describes the physics and every boundary of ONE tree, with each ``Location`` naming a
zone of that tree's Fluent ``.msh``. CFX has no supported text API for binding a mesh to a CCL;
the supported automation is playback of a CFX-Pre session file. This script does exactly what
``coronary_sdf`` does (``mesh_sensitivity_study.prepare_and_run_cfx`` and its
``cfx_remesh_session_template.pre``): it renders a minimal batch session -

    >load mode=new
    >gtmImport filename=<mesh.msh>, type=Fluent, units=mm, genOpt= -n, nameStrategy=Assembly
    >importccl filename=<resolved.ccl>, mode=replace, autoLoadLibrary=none
    >exportccl filename=<stem>_cfxpre.ccl, mode=overwrite      (the CCL CFX-Pre generated)
    >writeCaseFile filename=<stem>.def, operation=write def file
    >quit

- and runs ``cfx5pre -batch`` on it, streaming CFX-Pre's console to stdout so the GUI log shows
it live.

What it writes, all next to ``--out`` (``<stem>`` = the .def stem):

* ``<stem>.def``            the solver input (``cfx5solve -def``); embeds the mesh.
* ``<stem>_resolved.ccl``   the CCL that was imported: the pipeline CCL plus the one thing it
                            cannot know until a mesh is chosen, the DOMAIN ``Location`` (the
                            ``.msh``'s fluid zone). Nothing else is changed.
* ``<stem>_cfxpre.ccl``     CFX-Pre's own ``exportccl`` of the built case - the complete CCL as
                            CFX-Pre resolved it (default wall, mesh names, every default it
                            filled in). This is the file to read when the .def looks wrong.
* ``<stem>.pre``            the session that was played (re-run it by hand in CFX-Pre if needed).
* ``<stem>_cfxpre.log``     CFX-Pre's console output.
* ``<stem>.cfx``            optionally (``--cfx-out``): the CFX-Pre case file too.

MESH NAMES. With ``nameStrategy=Assembly`` CFX-Pre names every region after the Fluent zone with
``-`` replaced by a space (``from-...-plane1-pressure-outlet`` becomes ``from ... plane1 pressure
outlet``), which is the form every pipeline CCL already writes in its ``Location`` lines. Before
CFX-Pre is launched (a minute or more for a 500 MB mesh) each BOUNDARY Location is checked against
the ``(45`` zone table at the end of the ``.msh``; a Location naming no zone stops the run unless
``--allow-unbound`` is given, because CFX-Pre would still write a .def with an unbound boundary
that only fails in the solver.

One tree per run: pass the tree's CCL and its .msh. The GUI runs it once per Windkessel CCL,
pairing each with the ``.msh`` of the same side. ``--dry-run`` writes the resolved CCL and the
session without launching CFX-Pre, which is how the tests exercise it on a machine without CFX.
"""
from __future__ import annotations

import argparse
import glob
import os
import re
import shutil
import subprocess
import time
from pathlib import Path

#: Where the ANSYS installers put CFX-Pre; newest version wins. ``CFX_PRE`` in the environment
#: or ``--cfx-pre`` override the search.
CFX_PRE_GLOBS = (
    r"C:\Program Files\ANSYS Inc\v*\CFX\bin\cfx5pre.exe",
    "/usr/ansys_inc/v*/CFX/bin/cfx5pre",
    "/ansys_inc/v*/CFX/bin/cfx5pre",
)
DEFAULT_UNITS = "mm"
DEFAULT_CFX_VERSION = "25.2"
#: Fluent zone-type names of the mesh interior and the wall in the ``(45`` zone table.
FLUID_ZONE_TYPE = "fluid"
WALL_ZONE_TYPE = "wall"
#: How much of the end of a .msh to read for the zone table before falling back to a full scan.
ZONE_TABLE_TAIL_BYTES = 4 << 20

_ZONE_RE = re.compile(r"^\((?:45|39)\s*\(\s*(\d+)\s+([^\s()]+)\s+([^\s()]+)", re.M)


# -- CFX-Pre ---------------------------------------------------------------------------------


def find_cfx_pre(explicit: str | Path | None = None) -> Path | None:
    """``--cfx-pre`` > ``$CFX_PRE`` > ``cfx5pre`` on PATH > the newest ANSYS install."""
    for cand in (explicit, os.environ.get("CFX_PRE")):
        if cand:
            p = Path(cand)
            return p.resolve() if p.is_file() else None
    on_path = shutil.which("cfx5pre")
    if on_path:
        return Path(on_path)
    hits: list[str] = []
    for pattern in CFX_PRE_GLOBS:
        hits += glob.glob(pattern)
    if not hits:
        return None
    return Path(max(hits, key=lambda h: cfx_version_of(h) or ""))


def cfx_version_of(cfx_pre: str | Path | None) -> str | None:
    """``...\\v252\\CFX\\bin\\cfx5pre.exe`` -> ``25.2`` (the ``CFX Pre Version`` a session names)."""
    m = re.search(r"[\\/]v(\d)(\d)(\d)[\\/]", str(cfx_pre or ""))
    return f"{m.group(1)}{m.group(2)}.{m.group(3)}" if m else None


# -- the .msh zone table ---------------------------------------------------------------------


def read_zone_table(msh: str | Path) -> dict[int, tuple[str, str]]:
    """``zone_id -> (type, name)`` from the ``(45``/``(39`` table of a Fluent ``.msh``.

    The table sits at the very end of the file, after hundreds of MB of faces, so the tail is
    read first; a file whose table is not in the tail (or that has none) is scanned in full.
    """
    msh = Path(msh)
    size = msh.stat().st_size
    with open(msh, "rb") as f:
        f.seek(max(0, size - ZONE_TABLE_TAIL_BYTES))
        tail = f.read().decode("utf-8", errors="replace")
    matches = list(_ZONE_RE.finditer(tail))
    table = {int(m.group(1)): (m.group(2), m.group(3)) for m in matches}
    if table and (size <= ZONE_TABLE_TAIL_BYTES or _tail_holds_whole_table(tail, matches[0].start())):
        return table
    table = {}
    with open(msh, "r", errors="replace") as f:
        for line in f:
            if line.startswith("(45") or line.startswith("(39"):
                m = _ZONE_RE.match(line)
                if m:
                    table[int(m.group(1))] = (m.group(2), m.group(3))
    return table


def _tail_holds_whole_table(tail: str, first_zone_at: int) -> bool:
    """Whether the tail window started before the zone table: some complete line ahead of the
    first zone entry is neither blank nor a ``(0 "..."`` comment (the table is zone lines,
    blank lines and ``(0 " Zone: ..."`` comments only; face data or a section end before it
    proves the window reaches past the table's start)."""
    before = tail[:first_zone_at].splitlines()[1:]         # drop the line the window cut into
    return any(ln.strip() and not ln.lstrip().startswith("(0 ") for ln in before)


def region_name(zone_name: str) -> str:
    """The CFX-Pre region name of a Fluent zone (``nameStrategy=Assembly``): ``-`` -> space."""
    return zone_name.replace("-", " ")


def zones_of_type(table: dict, ztype: str) -> list[str]:
    return [region_name(name) for _, (t, name) in sorted(table.items()) if t == ztype]


def _norm(s: str) -> str:
    return " ".join(re.sub(r"[+\-_]+", " ", (s or "").lower()).split())


# -- the CCL ---------------------------------------------------------------------------------


def _decomment(line: str) -> str:
    """The code part of a CCL line (``#`` starts a comment, as the pipeline writers use it)."""
    return line.split("#", 1)[0].rstrip()


def _logical_lines(text: str) -> list[str]:
    """Lines with CFX-Pre's trailing-backslash continuations joined (an export wraps long values)."""
    out: list[str] = []
    buf: str | None = None
    for line in text.splitlines():
        buf = line if buf is None else buf + line.lstrip()
        if buf.rstrip().endswith("\\"):
            buf = buf.rstrip()[:-1]
            continue
        out.append(buf)
        buf = None
    if buf is not None:
        out.append(buf)
    return out


def resolve_ccl(text: str, fluid_location: str, domain: str | None = None) -> tuple[str, dict]:
    """Give the DOMAIN block a ``Location`` when the CCL has none.

    The pipeline writers leave it out because the fluid zone name is a property of the mesh, not
    of the physics. Inserted right after ``Domain Type``; a DOMAIN that already has a Location is
    left alone (``info["had_location"]``). Returns ``(text, info)``.
    """
    lines = text.splitlines()
    info = {"domain": None, "had_location": False, "inserted": False, "location": fluid_location}
    out: list[str] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        code = _decomment(line).strip()
        m = re.match(r"^DOMAIN:\s*(.+?)\s*$", code)
        if m and (domain is None or m.group(1) == domain) and info["domain"] is None:
            indent = line[:len(line) - len(line.lstrip())]
            info["domain"] = m.group(1)
            out.append(line)
            i += 1
            # the domain's own scalar keys come before its first nested block
            block: list[str] = []
            while i < len(lines):
                c = _decomment(lines[i]).strip()
                if c == "END" or re.match(r"^[A-Z][A-Z0-9 ]*:", c):
                    break
                block.append(lines[i])
                i += 1
            has_loc = any(re.match(r"^\s*Location\s*=", _decomment(b)) for b in block)
            info["had_location"] = has_loc
            if not has_loc:
                new_line = f"{indent}  Location = {fluid_location} # fluid zone of the .msh (cfx_def_export.py)"
                placed = False
                for b in block:
                    out.append(b)
                    if re.match(r"^\s*Domain Type\s*=", _decomment(b)) and not placed:
                        out.append(new_line)
                        placed = True
                if not placed:
                    out.append(new_line)
                info["inserted"] = True
            else:
                out.extend(block)
            continue
        out.append(line)
        i += 1
    return "\n".join(out) + ("\n" if text.endswith("\n") or not text else ""), info


WALL_BOUNDARY_NAME = "Wall"


def resolve_wall(text: str, wall_location: str) -> tuple[str, dict]:
    """Bind the CCL's WALL boundary to the mesh's wall zone, as the template case does.

    ``windkessel_BC.py`` writes a ``BOUNDARY: Wall`` without a ``Location`` (only the mesh knows
    it); that boundary gets ``Location = <wall zone>``. A CCL with no WALL boundary at all (the
    giessen / mass writers) gets the template's ``Wall`` block appended after its last boundary
    - CFX-Pre would otherwise put the zone in an automatic "Default" wall, which is equivalent
    but not what the template holds. A WALL boundary that already has a Location is kept.
    Returns ``(text, info)`` with ``info["action"]`` in ``located`` / ``added`` / ``kept``.
    """
    lines = text.splitlines()
    info = {"action": "kept", "location": wall_location, "name": None}
    out: list[str] = []
    i = 0
    in_domain = False
    last_boundary_end = None       # index in ``out`` after the last top-level BOUNDARY of the domain
    while i < len(lines):
        line = lines[i]
        code = _decomment(line).strip()
        if re.match(r"^DOMAIN:", code):
            in_domain = True
        if in_domain and re.match(r"^BOUNDARY:", code):
            indent = line[:len(line) - len(line.lstrip())]
            name = code.split(":", 1)[1].strip()
            # collect the whole block (nested blocks included)
            block = [line]
            depth = 1
            i += 1
            while i < len(lines) and depth:
                c = _decomment(lines[i]).strip()
                if re.match(r"^[A-Z][A-Z0-9 ]*:", c):
                    depth += 1
                elif c == "END":
                    depth -= 1
                block.append(lines[i])
                i += 1
            is_wall = any(re.match(r"^\s*Boundary Type\s*=\s*WALL\s*$", _decomment(b), re.I) for b in block)
            has_loc = any(re.match(r"^\s*Location\s*=", _decomment(b)) for b in block[1:-1]
                          if not re.match(r"^\s*(BOUNDARY CONDITIONS|[A-Z][A-Z ]*):", _decomment(b).strip()))
            if is_wall and info["action"] == "kept" and not has_loc:
                # insert after "Boundary Type = WALL"
                placed = []
                for b in block:
                    placed.append(b)
                    if re.match(r"^\s*Boundary Type\s*=", _decomment(b)):
                        placed.append(f"{indent}  Location = {wall_location} # wall zone of the .msh (cfx_def_export.py)")
                block = placed
                info["action"], info["name"] = "located", name
            elif is_wall and info["action"] == "kept":
                info["name"] = name
            out.extend(block)
            last_boundary_end = len(out)
            continue
        out.append(line)
        i += 1
    if info["name"] is None and last_boundary_end is not None and wall_location:
        indent = "    "
        block = [f"{indent}BOUNDARY: {WALL_BOUNDARY_NAME}", f"{indent}  Boundary Type = WALL",
                 f"{indent}  Location = {wall_location} # wall zone of the .msh (cfx_def_export.py)",
                 f"{indent}  BOUNDARY CONDITIONS:", f"{indent}    MASS AND MOMENTUM:",
                 f"{indent}      Option = No Slip Wall", f"{indent}    END", f"{indent}  END", f"{indent}END"]
        out[last_boundary_end:last_boundary_end] = block
        info["action"], info["name"] = "added", WALL_BOUNDARY_NAME
    return "\n".join(out) + ("\n" if text.endswith("\n") or not text else ""), info


def boundary_locations(text: str) -> list[tuple[str, str]]:
    """``[(boundary name, Location), ...]`` of every BOUNDARY block of a CCL."""
    found: list[tuple[str, str]] = []
    stack: list[str] = []
    current: str | None = None
    got_loc = False
    for raw in _logical_lines(text):
        code = _decomment(raw).strip()
        if not code:
            continue
        if code == "END":
            if stack:
                popped = stack.pop()
                if popped.startswith("BOUNDARY:"):
                    current, got_loc = None, False
            continue
        if re.match(r"^[A-Z][A-Z0-9 ]*:", code):
            stack.append(code)
            if code.startswith("BOUNDARY:"):
                current, got_loc = code.split(":", 1)[1].strip(), False
            continue
        if current is not None and not got_loc:
            m = re.match(r"^Location\s*=\s*(.+?)\s*$", code)
            if m:
                found.append((current, m.group(1)))
                got_loc = True
    return found


def inlet_settings(text: str) -> list[tuple[str, str, str]]:
    """``[(boundary, MASS AND MOMENTUM option, its value line), ...]`` of every INLET boundary."""
    out: list[tuple[str, str, str]] = []
    stack: list[str] = []
    name: str | None = None
    btype, option, value = "", "", ""
    for raw in _logical_lines(text):
        code = _decomment(raw).strip()
        if not code:
            continue
        if code == "END":
            if stack:
                popped = stack.pop()
                if popped.startswith("BOUNDARY:") and name is not None:
                    if btype == "INLET":
                        out.append((name, option, value))
                    name = None
            continue
        if re.match(r"^[A-Z][A-Z0-9 ]*:", code):
            stack.append(code)
            if code.startswith("BOUNDARY:"):
                name, btype, option, value = code.split(":", 1)[1].strip(), "", "", ""
            continue
        if name is None or "=" not in code:
            continue
        key, _, val = code.partition("=")
        key, val = key.strip(), val.strip()
        if key == "Boundary Type":
            btype = val.upper()
        elif "MASS AND MOMENTUM" in "/".join(stack):
            if key == "Option":
                option = val
            elif key in ("Normal Speed", "Mass Flow Rate", "Bulk Mass Flow Rate"):
                value = f"{key} = {val}"
    return out


def stale_windkessel_ccl(text: str) -> str | None:
    """Why a windkessel CCL predates the template layout (Case_1.cfx), or None.

    The tell-tales: a WK() outlet CCL whose inlet is still a mass-flow BC, or that has no
    ``vconstant`` / ``vin`` expressions - windkessel_BC.py has written those since the template
    layout; a CCL from before then builds a .def without them.
    """
    if "WK(" not in text:
        return None
    has_vconstant = re.search(r"(?m)^\s*vconstant\s*=", text) is not None
    inlets = inlet_settings(text)
    mass_inlets = [n for n, opt, _ in inlets if "Mass Flow" in opt]
    if mass_inlets and not has_vconstant:
        return (f"inlet {', '.join(mass_inlets)} is a '{next(o for _, o, _ in inlets if 'Mass Flow' in o)}' BC and there "
                f"is no vconstant expression: this windkessel CCL was written by windkessel_BC.py from before "
                f"the template layout (Normal Speed = vconstant). Re-run step 5 (Windkessel BCs) to rewrite it, "
                f"then this step.")
    if not has_vconstant:
        return "no vconstant expression: re-run step 5 (Windkessel BCs) so the CCL has the template's inlet."
    return None


def check_locations(text: str, table: dict) -> list[str]:
    """Boundary Locations that name no zone of the mesh, as ``"<boundary>: <location>"``."""
    zones = {_norm(region_name(name)) for _, (_t, name) in table.items()}
    bad = []
    for bname, loc in boundary_locations(text):
        # a comma-separated Location is a list of zones; every part must exist
        parts = [p.strip() for p in loc.split(",") if p.strip()]
        if any(_norm(p) not in zones for p in parts):
            bad.append(f"{bname}: {loc}")
    return bad


# -- the session -----------------------------------------------------------------------------


def _pre_path(p: str | Path) -> str:
    """CFX-Pre session syntax wants forward slashes (it treats ``\\`` as a continuation)."""
    return str(Path(p).resolve()).replace("\\", "/")


def render_session(msh, ccl, out_def, export_ccl=None, cfx_out=None, *, units: str = DEFAULT_UNITS,
                   version: str = DEFAULT_CFX_VERSION) -> str:
    """The batch session: new case, import the mesh, import the CCL, export the CCL CFX-Pre
    resolved, write the .def (and the .cfx if asked)."""
    lines = [
        "# CFX-Pre batch session written by cfx_def_export.py (SCAPIS pipeline)",
        f"# mesh: {msh}", f"# ccl:  {ccl}", "",
        "COMMAND FILE:", f"  CFX Pre Version = {version}", "END", "",
        ">load mode=new", "> update", "",
        f">gtmImport filename={_pre_path(msh)}, type=Fluent, units={units}, genOpt= -n, nameStrategy=Assembly",
        "> update", "",
        f">importccl filename={_pre_path(ccl)}, mode=replace, autoLoadLibrary=none",
        "> update", "",
    ]
    if export_ccl:
        lines += [f">exportccl filename={_pre_path(export_ccl)}, mode=overwrite", "> update", ""]
    if cfx_out:
        # writeCaseFile without an operation is "Save Case As" (a .cfx)
        lines += [f">writeCaseFile filename={_pre_path(cfx_out)}", "> update", ""]
    lines += [f">writeCaseFile filename={_pre_path(out_def)}, operation=write def file", "> update", "",
              ">quit", ""]
    return "\n".join(lines)


def run_cfx_pre(cfx_pre: Path, session: Path, cwd: Path, log_path: Path, timeout: float | None = None) -> int:
    """``cfx5pre -batch <session>``, console streamed to stdout and copied to ``log_path``."""
    cmd = [str(cfx_pre), "-batch", str(session)]
    print(f"[CFX-Pre] $ {' '.join(cmd)}", flush=True)
    started = time.monotonic()
    with open(log_path, "w", encoding="utf-8", errors="replace") as log:
        proc = subprocess.Popen(cmd, cwd=str(cwd), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                stdin=subprocess.DEVNULL, text=True, encoding="utf-8", errors="replace",
                                bufsize=1)
        assert proc.stdout is not None
        try:
            for line in proc.stdout:
                log.write(line)
                print("  | " + line.rstrip("\n"), flush=True)
                if timeout and time.monotonic() - started > timeout:
                    proc.kill()
                    print(f"[CFX-Pre] killed after {timeout:g} s", flush=True)
                    break
        finally:
            proc.stdout.close()
        rc = proc.wait()
    print(f"[CFX-Pre] exit {rc} after {time.monotonic() - started:.0f} s", flush=True)
    return rc


# -- CLI -------------------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Build the CFX solver input (.def) from a pipeline CCL and its Fluent .msh by "
                    "playing a CFX-Pre batch session (one tree per run).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--msh", type=Path, default=None,
                   help="the tree's Fluent mesh (.msh); its zone names are the CCL Locations (required)")
    p.add_argument("--ccl", type=Path, default=None,
                   help="the tree's CCL, normally windkessel_BC.py's *_windkessel_boundary_conditions_<side>.ccl "
                        "(required)")
    p.add_argument("--out", type=Path, default=None,
                   help="the .def to write (default: <ccl dir>/cfx/<ccl stem>.def)")
    p.add_argument("--resolved-ccl", type=Path, default=None,
                   help="where to write the CCL actually imported (default: <out stem>_resolved.ccl)")
    p.add_argument("--export-ccl", type=Path, default=None,
                   help="where CFX-Pre exports the CCL of the built case (default: <out stem>_cfxpre.ccl)")
    p.add_argument("--session-out", type=Path, default=None,
                   help="where to write the CFX-Pre session (default: <out stem>.pre)")
    p.add_argument("--cfx-out", type=Path, default=None,
                   help="also save the CFX-Pre case file (.cfx) here; off by default, it repeats the mesh")
    p.add_argument("--log", type=Path, default=None,
                   help="CFX-Pre console log (default: <out stem>_cfxpre.log)")
    p.add_argument("--cfx-pre", type=Path, default=None,
                   help="cfx5pre executable (default: $CFX_PRE, then PATH, then the newest ANSYS install)")
    p.add_argument("--units", default=DEFAULT_UNITS, help="length unit of the .msh coordinates")
    p.add_argument("--cfx-version", default=None,
                   help="'CFX Pre Version' the session declares (default: from the cfx5pre path, else "
                        f"{DEFAULT_CFX_VERSION})")
    p.add_argument("--fluid-zone", default=None,
                   help="DOMAIN Location to use instead of the .msh's fluid zone")
    p.add_argument("--wall-zone", default=None,
                   help="Location of the Wall boundary instead of the .msh's wall zone(s)")
    p.add_argument("--domain", default=None,
                   help="which DOMAIN block gets the Location (default: the first one)")
    p.add_argument("--allow-unbound", action="store_true",
                   help="launch CFX-Pre even when a boundary Location names no zone of the .msh")
    p.add_argument("--allow-stale", action="store_true",
                   help="build the .def even from a windkessel CCL written before the template layout "
                        "(mass-flow inlet, no vconstant)")
    p.add_argument("--no-export-ccl", action="store_true", help="skip CFX-Pre's exportccl")
    p.add_argument("--timeout", type=float, default=0.0, help="kill CFX-Pre after this many seconds (0 = never)")
    p.add_argument("--dry-run", action="store_true",
                   help="write the resolved CCL and the session only; do not launch CFX-Pre")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.msh or not args.ccl:
        raise SystemExit("[ERROR] --msh and --ccl are both required.")
    msh, ccl = Path(args.msh), Path(args.ccl)
    if not msh.is_file():
        raise SystemExit(f"[ERROR] mesh not found: {msh}")
    if not ccl.is_file():
        raise SystemExit(f"[ERROR] CCL not found: {ccl}")

    out_def = Path(args.out) if args.out else ccl.parent / "cfx" / f"{ccl.stem}.def"
    out_def.parent.mkdir(parents=True, exist_ok=True)
    stem = out_def.with_suffix("")
    resolved = Path(args.resolved_ccl) if args.resolved_ccl else stem.with_name(f"{stem.name}_resolved.ccl")
    export_ccl = None if args.no_export_ccl else (
        Path(args.export_ccl) if args.export_ccl else stem.with_name(f"{stem.name}_cfxpre.ccl"))
    session = Path(args.session_out) if args.session_out else stem.with_suffix(".pre")
    log_path = Path(args.log) if args.log else stem.with_name(f"{stem.name}_cfxpre.log")
    cfx_out = Path(args.cfx_out) if args.cfx_out else None

    print(f"[MSH] reading zone table of {msh.name} ({msh.stat().st_size / 1e6:.0f} MB) ...", flush=True)
    table = read_zone_table(msh)
    fluids = zones_of_type(table, FLUID_ZONE_TYPE)
    walls = zones_of_type(table, WALL_ZONE_TYPE)
    print(f"[MSH] {len(table)} zones: {len(fluids)} fluid, {len(walls)} wall, "
          f"{sum(1 for t, _ in table.values() if 'outlet' in t or 'inlet' in t or 'outflow' in t)} openings")
    if args.fluid_zone:
        fluid_location = args.fluid_zone
    elif fluids:
        fluid_location = ",".join(fluids)
        if len(fluids) > 1:
            print(f"[WARN] {len(fluids)} fluid zones in the mesh; the DOMAIN gets all of them: {fluid_location}")
    else:
        raise SystemExit("[ERROR] no fluid zone in the .msh zone table and no --fluid-zone given; "
                         "the DOMAIN cannot be bound to the mesh.")

    text = ccl.read_text(encoding="utf-8", errors="replace")
    resolved_text, info = resolve_ccl(text, fluid_location, args.domain)
    if info["domain"] is None:
        raise SystemExit(f"[ERROR] no DOMAIN block{' named ' + repr(args.domain) if args.domain else ''} in {ccl}")
    if info["inserted"]:
        print(f"[CCL] DOMAIN '{info['domain']}': Location = {fluid_location}")
    else:
        print(f"[CCL] DOMAIN '{info['domain']}' already has a Location; left as written")
    if args.wall_zone or walls:
        wall_location = args.wall_zone or ",".join(walls)
        resolved_text, wall = resolve_wall(resolved_text, wall_location)
        if wall["action"] == "located":
            print(f"[CCL] BOUNDARY '{wall['name']}': Location = {wall_location}")
        elif wall["action"] == "added":
            print(f"[CCL] added BOUNDARY '{wall['name']}' (no slip) at Location = {wall_location}")
        else:
            print(f"[CCL] BOUNDARY '{wall['name']}' already has a Location; left as written")
    else:
        print("[WARN] no wall zone in the .msh zone table; CFX-Pre will put unclaimed faces in its default wall")

    for name, option, value in inlet_settings(resolved_text):
        print(f"[CCL] inlet '{name}': Option = {option or '?'}{'  (' + value + ')' if value else ''}")
    stale = stale_windkessel_ccl(resolved_text)
    if stale:
        print(f"[{'WARN' if args.allow_stale else 'ERROR'}] {stale}")
        if not args.allow_stale:
            raise SystemExit("[ERROR] stale windkessel CCL; pass --allow-stale to build the .def from it anyway.")
    bad = check_locations(resolved_text, table)
    bound = len(boundary_locations(resolved_text)) - len(bad)
    print(f"[CCL] {bound} boundary Location(s) match a zone of {msh.name}")
    for b in bad:
        print(f"[{'WARN' if args.allow_unbound else 'ERROR'}] Location names no zone of the mesh: {b}")
    if bad and not args.allow_unbound:
        raise SystemExit(f"[ERROR] {len(bad)} boundary Location(s) do not exist in {msh.name}. Is this the "
                         f"CCL of the other tree? Pass --allow-unbound to build the .def anyway.")

    resolved.parent.mkdir(parents=True, exist_ok=True)
    resolved.write_text(resolved_text, encoding="utf-8")
    print(f"[CCL] wrote {resolved}")

    cfx_pre = find_cfx_pre(args.cfx_pre)
    version = args.cfx_version or cfx_version_of(cfx_pre) or DEFAULT_CFX_VERSION
    session.parent.mkdir(parents=True, exist_ok=True)
    session.write_text(render_session(msh, resolved, out_def, export_ccl, cfx_out, units=args.units,
                                      version=version), encoding="utf-8")
    print(f"[PRE] wrote {session} (CFX Pre Version = {version})")

    if args.dry_run:
        print("[PRE] dry run: CFX-Pre not launched")
        return 0
    if cfx_pre is None:
        raise SystemExit("[ERROR] cfx5pre not found. Pass --cfx-pre <path to cfx5pre.exe>, set CFX_PRE, or "
                         "install ANSYS CFX. The session was written; play it in CFX-Pre by hand if needed.")

    for stale in (out_def, export_ccl, cfx_out):
        if stale and stale.exists():
            stale.unlink()
    rc = run_cfx_pre(cfx_pre, session, out_def.parent, log_path, timeout=args.timeout or None)
    ok = True
    if not out_def.is_file():
        print(f"[ERROR] CFX-Pre did not write {out_def} (see {log_path})")
        ok = False
    else:
        print(f"[DEF] wrote {out_def} ({out_def.stat().st_size / 1e6:.1f} MB)")
    if export_ccl:
        if export_ccl.is_file():
            print(f"[CCL] CFX-Pre exported {export_ccl}")
        else:
            print(f"[WARN] CFX-Pre did not export {export_ccl}")
    if cfx_out and cfx_out.is_file():
        print(f"[CFX] saved {cfx_out}")
    if rc and ok:
        print(f"[WARN] cfx5pre exited {rc} but the .def exists; check {log_path}")
    return 0 if ok and rc == 0 else (rc or 1)


if __name__ == "__main__":
    raise SystemExit(main())
