"""Batch-export CFD fields from ANSYS CFX ``.res`` results files.

The Strahler/radius post-processing (:mod:`coronary_sdf.strahler_analysis`) needs
per-node CFD fields sampled on the lumen **wall** (wall shear, pressure) and
through the lumen **volume** (pressure, velocity).  CFX stores these in a
proprietary ``.res`` container, so they are extracted here by driving CFD-Post in
batch mode (``cfx5post -batch``) with a generated session (``.cse``) file.

Two exports are produced per results file:

``<stem>_wall.csv``
    One row per wall node: ``X, Y, Z, Pressure, Wall Shear`` (SI: m, Pa).
``<stem>_volume.csv``
    One row per volume node: ``X, Y, Z, Pressure, Velocity u/v/w, Velocity``.

Both are converted to ``.npz`` on the way out, because the CSVs run to hundreds
of megabytes and are ~40x slower to re-read on every analysis pass.

Usage::

    python -m coronary_sdf.cfx_extract \
        --res <run_001.res> [--res <other.res> ...] \
        --out analysis_out/cfd_extract

Re-running skips any export whose ``.npz`` already exists unless ``--force`` is
given; the CFD-Post pass over a multi-million-node mesh takes minutes.
"""

from __future__ import annotations

import argparse
import hashlib
import math
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

# CFD-Post writes a two-line ``[Name] / <region>`` preamble and a ``[Data]``
# marker before the real header, so the column names are not on line 1.
_DATA_MARKER = "[Data]"

# ``Wall Shear`` is the *magnitude* |tau_w| -- measured over 1.16 M exported wall
# nodes on LADAF-2024-28, exactly 0.0% are negative. The direction lives in the
# components, and without them no directional wall metric exists: OSI, RRT,
# transverse WSS and any reversal fraction all need the vector. They are requested
# here so future exports carry them; `available_variables` drops whatever a given
# .res does not have, so a run without them still exports cleanly, and the
# existing CSVs in `analysis_out/cfd_extract/` predate this and have magnitude
# only. (A directional metric also needs a transient solve -- these are steady
# state -- so this removes one of the two obstacles, not both.)
WALL_VARIABLES = [
    "Pressure", "Wall Shear",
    "Wall Shear X", "Wall Shear Y", "Wall Shear Z",
]
# ``Volume of Finite Volumes`` is the control-volume size at each node. CFX meshes
# these vessels with an inflated boundary layer, so nodes are far denser near the
# wall than in the core; averaging a field over raw node counts would weight the
# slow near-wall fluid several times too heavily. Exporting the volume lets the
# analysis form a proper volume-weighted mean instead.
VOLUME_VARIABLES = [
    "Pressure", "Velocity u", "Velocity v", "Velocity w", "Velocity",
    "Volume of Finite Volumes",
]


def find_cfx_tool(name: str = "cfx5post") -> str:
    """Absolute path to a CFX command-line tool.

    Prefers the matching environment variable (``CFX5POST``/``CFX5EXPORT``) and
    ``PATH``, else falls back to the newest ANSYS install under the standard
    Windows location."""
    env = os.environ.get(name.upper())
    if env and Path(env).exists():
        return env
    found = shutil.which(name) or shutil.which(f"{name}.exe")
    if found:
        return found
    roots = sorted(Path(r"C:\Program Files\ANSYS Inc").glob(f"v*/CFX/bin/{name}.exe"))
    if roots:
        return str(roots[-1])
    raise FileNotFoundError(
        f"{name} not found; set {name.upper()} or add ANSYS CFX bin/ to PATH"
    )


def find_cfx5post() -> str:
    """Backwards-compatible alias for :func:`find_cfx_tool`."""
    return find_cfx_tool("cfx5post")


def probe_regions(res: Path) -> tuple[str, str]:
    """Discover ``(domain, wall_boundary)`` names for one results file.

    The names are **not** fixed across models — a domain re-meshed in CFX-Pre
    becomes ``Default Domain Modified`` and its wall
    ``Default Domain Modified Default`` — so hardcoding them silently exports an
    empty file. ``cfx5export -summary`` lists both, which is cheap because it only
    reads the header.

    The wall is the boundary whose name is the domain plus ``Default`` (the CFX
    convention for "everything not otherwise assigned"); failing that, the first
    boundary that is not an inlet, outlet or opening."""
    out = subprocess.run([find_cfx_tool("cfx5export"), "-summary", str(res)],
                         capture_output=True, text=True).stdout

    domain = "Default Domain"
    m = re.search(r"^\s*\d+\s+domains?:\s*$", out, re.M)
    if m:
        tail = out[m.end():].lstrip("\n")
        first = re.match(r"\s*\d+\s+(.+?)\s*$", tail.splitlines()[0] if tail else "")
        if first:
            domain = first.group(1)

    boundaries: list[str] = []
    b = re.search(r"^\s*\d+\s+boundaries for domain '(.+?)':\s*$", out, re.M)
    if b:
        domain = b.group(1)
        for line in out[b.end():].splitlines():
            if not line.strip():
                if boundaries:
                    break
                continue
            item = re.match(r"\s*\d+\s+(.+?)\s*$", line)
            if not item:
                break
            boundaries.append(item.group(1))

    wall = f"{domain} Default"
    if boundaries and wall not in boundaries:
        skip = ("inlet", "outlet", "opening", "symmetry")
        wall = next((n for n in boundaries
                     if not n.lower().startswith(skip)), boundaries[0])
    return domain, wall


def probe_variables(res: Path) -> set[str]:
    """Names of the solution variables stored in ``res``.

    What a run wrote depends on its *Extra Output Variables List*, so a field
    present in one solve can be absent from the next — the full-skeleton right
    tree carries 10 variables and no ``Wall Shear``, while the left carries 23 and
    does. Asking CFD-Post for a variable that is not there fails the whole export,
    so the request is filtered against this set first."""
    out = subprocess.run([find_cfx_tool("cfx5export"), "-summary", str(res)],
                         capture_output=True, text=True).stdout
    names: set[str] = set()
    m = re.search(r"^\s*\d+\s+variables:\s*$", out, re.M)
    if not m:
        return names
    for line in out[m.end():].splitlines():
        if not line.strip():
            if names:
                break
            continue
        item = re.match(r"\s*\d+\s+(?:\[\d+\]\s*)?(.+?)\s*(?:\[[^\]]*\])?\s*$", line)
        if not item:
            break
        names.add(item.group(1).strip())
    return names


def available_variables(requested: list[str], present: set[str]) -> list[str]:
    """Subset of ``requested`` that ``present`` can supply.

    A vector is listed once (``Velocity``) but CFD-Post will happily export its
    components (``Velocity u``), so a component counts as available whenever its
    parent vector is."""
    if not present:
        return list(requested)          # probe failed; let CFD-Post decide
    out: list[str] = []
    for var in requested:
        base = var.rsplit(" ", 1)[0]
        if var in present or (var[-1:] in "uvw" and base in present):
            out.append(var)
    return out


def _cse_text(
    export_csv: Path, location: str, variables: list[str], res: Path | None = None
) -> str:
    """CFD-Post session that exports ``location`` to ``export_csv``.

    ``Vector Display = Scalar`` makes CFD-Post write the magnitude of a vector
    variable (e.g. ``Velocity``) as a plain column rather than bracketed
    components, which keeps the CSV parseable as a flat table."""
    commands = [
            "COMMAND FILE:",
            "  CFX Post Version = 25.2",
            "END",
    ]
    if res is not None:
        commands.append(">load filename={}".format(res.as_posix()))
    commands.extend(
        [
            "EXPORT:",
            f"  Export File = {export_csv.as_posix()}",
            "  Export Geometry = On",
            "  Export Type = Generic",
            "  Include Header = On",
            f"  Location = {location}",
            f"  Location List = {location}",
            "  Overwrite = On",
            "  Precision = 8",
            '  Separator = ", "',
            "  Spatial Variables = X,Y,Z",
            f"  Variable List = {','.join(variables)}",
            "  Vector Display = Scalar",
            "END",
            ">export",
            "",
        ]
    )
    return "\n".join(commands)


def read_export_csv(path: Path) -> tuple[list[str], np.ndarray]:
    """Parse a CFD-Post generic export into ``(column_names, values)``.

    Column names keep their ``[ unit ]`` suffix stripped; rows that fail to
    parse as floats (CFD-Post writes ``null`` for undefined nodes) are dropped."""
    header: list[str] | None = None
    rows: list[list[float]] = []
    with path.open("r", errors="replace") as fh:
        seen_marker = False
        for line in fh:
            line = line.strip()
            if not line:
                continue
            if not seen_marker:
                seen_marker = line.startswith(_DATA_MARKER)
                continue
            if header is None:
                header = [c.split("[")[0].strip() for c in line.split(",")]
                continue
            try:
                rows.append([float(v) for v in line.split(",")])
            except ValueError:
                continue  # 'null' token or a stray trailing line
    if header is None:
        raise ValueError(f"no [Data] section found in {path}")
    return header, np.asarray(rows, dtype=np.float64)


def _run_export(
    post: str, res: Path, out_csv: Path, location: str, variables: list[str]
) -> None:
    cse = out_csv.with_suffix(".cse")
    # Explicitly load the result inside the session.  In CFX 25.2 the command-
    # line ``-res`` action may not finish before the batch EXPORT is evaluated,
    # leaving an empty state and a zero-byte CSV.
    cse.write_text(_cse_text(out_csv, location, variables, res=res))
    cmd = [post, "-batch", str(cse)]
    print(f"[cfx] {' '.join(cmd)}", flush=True)
    proc = subprocess.run(cmd, capture_output=True, text=True, cwd=str(out_csv.parent))
    if proc.returncode != 0 or not out_csv.exists():
        sys.stderr.write(proc.stdout + "\n" + proc.stderr + "\n")
        raise RuntimeError(f"cfx5post export failed for {res.name} ({location})")


def _short_result_link(res: Path) -> Path:
    """Return a short same-volume link for CFD-Post's legacy path handling."""
    short_dir = Path(__file__).resolve().parent / ".cfx_extract_short"
    short_dir.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha1(str(res.resolve()).encode("utf-8")).hexdigest()[:10]
    short_res = short_dir / f"{res.stem[:16]}_{digest}.res"
    if short_res.exists() and short_res.stat().st_size == res.stat().st_size:
        return short_res
    if short_res.exists():
        short_res.unlink()
    try:
        os.link(res, short_res)
    except OSError:
        shutil.copy2(res, short_res)
    return short_res


def export_res(
    res: Path, out_dir: Path, *, force: bool = False, post: str | None = None,
    fluent_mesh: Path | None = None,
) -> dict[str, Path]:
    """Export wall and volume fields for one ``.res`` file.

    Returns the ``{"wall": .npz, "volume": .npz}`` paths.  The intermediate CSVs
    are kept alongside so the extraction stays auditable by hand."""
    post = post or find_cfx_tool("cfx5post")
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = res.stem
    post_res = _short_result_link(res)
    written: dict[str, Path] = {}

    domain, wall = probe_regions(res)
    present = probe_variables(res)
    print(f"[cfx] {res.name}: domain='{domain}' wall='{wall}'", flush=True)

    for kind, location, requested in (
        ("wall", wall, WALL_VARIABLES),
        ("volume", domain, VOLUME_VARIABLES),
    ):
        npz = out_dir / f"{stem}_{kind}.npz"
        if npz.exists() and not force:
            print(f"[cfx] skip existing {npz.name}", flush=True)
            written[kind] = npz
            continue
        variables = available_variables(requested, present)
        missing = [v for v in requested if v not in variables]
        if missing:
            print(f"[cfx][WARN] {res.name} has no {', '.join(missing)}; "
                  f"exporting {kind} without it", flush=True)
        if not variables:
            print(f"[cfx][WARN] no {kind} variables available; skipping", flush=True)
            continue
        csv_path = out_dir / f"{stem}_{kind}.csv"
        _run_export(post, post_res, csv_path, location, variables)
        cols, vals = read_export_csv(csv_path)
        payload = {"columns": np.array(cols, dtype=object), "values": vals}
        if kind == "wall" and fluent_mesh is not None:
            payload["surface_control_area"] = _wall_surface_control_areas(
                fluent_mesh, cols, vals
            )
        np.savez_compressed(npz, **payload)
        print(f"[cfx] {npz.name}: {vals.shape[0]} nodes, cols={cols}", flush=True)
        written[kind] = npz
    return written


def _wall_surface_control_areas(
    fluent_mesh: Path, columns: list[str], values: np.ndarray
) -> np.ndarray:
    """Map one-third boundary-face areas onto exported CFX wall nodes."""
    try:
        from .flow_fractions import extract_msh_surface
    except ImportError:
        from flow_fractions import extract_msh_surface
    surface, _ = extract_msh_surface(fluent_mesh)
    if surface is None:
        raise RuntimeError("Could not extract Fluent boundary faces for wall weights")
    areas = np.zeros(surface.n_points, dtype=float)
    faces = np.asarray(surface.faces, dtype=np.int64)
    kinds = np.asarray(surface.cell_data["kind_id"])
    points = np.asarray(surface.points)

    # Simpleware's Fluent export is triangular, so process its packed
    # ``[3, i, j, k]`` face array in bounded vectorised batches.  The former
    # per-face Python loop dominated extraction time on the 7--15M-cell study
    # meshes while calculating the same one-third lumped vertex areas.
    triangular = (
        faces.size == 4 * kinds.size
        and (kinds.size == 0 or np.all(faces[0::4] == 3))
    )
    if triangular:
        packed = faces.reshape((-1, 4))
        wall_rows = np.flatnonzero(kinds == 0)
        batch_size = 500_000
        for start in range(0, wall_rows.size, batch_size):
            rows = wall_rows[start:start + batch_size]
            tri = packed[rows, 1:4]
            xyz = points[tri]
            tri_area = 0.5 * np.linalg.norm(
                np.cross(xyz[:, 1] - xyz[:, 0], xyz[:, 2] - xyz[:, 0]),
                axis=1,
            )
            np.add.at(areas, tri.reshape(-1), np.repeat(tri_area / 3.0, 3))
    else:
        cursor = 0
        for cell_index, kind in enumerate(kinds):
            count = int(faces[cursor])
            ids = faces[cursor + 1:cursor + 1 + count]
            cursor += 1 + count
            if int(kind) != 0 or count < 3:
                continue
            root = int(ids[0])
            for j in range(1, count - 1):
                tri = np.array([root, int(ids[j]), int(ids[j + 1])], dtype=int)
                xyz = points[tri]
                area = 0.5 * np.linalg.norm(
                    np.cross(xyz[1] - xyz[0], xyz[2] - xyz[0])
                )
                areas[tri] += area / 3.0
    xyz_columns = [columns.index(name) for name in ("X", "Y", "Z")]
    query = values[:, xyz_columns]
    tree = cKDTree(np.asarray(surface.points))
    options = []
    for scale in (1.0, 1000.0):
        distance, index = tree.query(query * scale)
        options.append((float(np.median(distance)), distance, index, scale))
    median, distance, index, scale = min(options, key=lambda item: item[0])
    positive = areas[areas > 0]
    tolerance = max(1e-4, 0.05 * math.sqrt(float(np.median(positive)))) if len(positive) else 1e-4
    if median > tolerance:
        raise RuntimeError(
            "CFX wall nodes do not align with Fluent wall faces (median {:.6g} mm, scale {})"
            .format(median, scale)
        )
    mapped = areas[index]
    if np.any(mapped <= 0):
        raise RuntimeError("Some CFX wall nodes mapped to cap/non-wall faces")
    return mapped


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--res", action="append", required=True, type=Path,
                    help="CFX results file; repeat for several runs")
    ap.add_argument("--out", type=Path, required=True, help="output directory")
    ap.add_argument("--fluent-mesh", type=Path,
                    help="matching Fluent mesh used to derive lumped wall areas")
    ap.add_argument("--force", action="store_true", help="re-export even if .npz exists")
    args = ap.parse_args(argv)

    post = find_cfx5post()
    print(f"[cfx] using {post}", flush=True)
    for res in args.res:
        if not res.exists():
            raise FileNotFoundError(res)
        export_res(
            res, args.out, force=args.force, post=post,
            fluent_mesh=args.fluent_mesh,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
