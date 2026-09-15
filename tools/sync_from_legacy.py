"""Copy source from the pre-monorepo checkouts into this repository.

The three packages here were assembled from separate checkouts on ``F:\\``.
Those checkouts still exist and may still be where work happens, so this
script carries changes across in that direction only: legacy -> monorepo.
Nothing is ever written back to a legacy checkout.

Two things make a plain copy wrong, and both are handled here.

**Layout.** ``hipct_seg_debug`` and ``skeleton_analysis`` map 1:1 -- their
``src/`` trees are identical on both sides. ``coronary_sdf`` does not: it was a
flat repo whose root *was* the package, so its files are redistributed into
``src/coronary_sdf/``, ``tests/`` and ``research_scripts/``.

**Sanitised files.** Several files were edited during publication to remove
hardcoded dataset paths, fix imports broken by the restructure, or rewrite
provenance notes. The legacy copies still hold the originals, so copying them
over would silently reintroduce paths like ``F:/Edo_latest_spatial_graph/...``
into code that ships in a wheel. Those files are PROTECTED: the script reports
that they differ and shows a diff, but never overwrites them. Resolve those by
hand, or pass ``--force-protected`` once you have checked the diff.

Usage::

    python tools/sync_from_legacy.py --dry-run      # default: show what would change
    python tools/sync_from_legacy.py --apply
    python tools/sync_from_legacy.py --apply --only hipct_seg_debug

After applying, the script re-runs the absolute-path scan that guarded the
initial publication. A leak fails the run loudly rather than reaching a commit.
"""

from __future__ import annotations

import argparse
import difflib
import os
import re
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

# ── Where the legacy checkouts live ──────────────────────────────────────────
# Override any of these with the matching environment variable, so this runs on
# a machine whose drive layout differs.
#
# Each entry lists candidates in preference order; the first that exists wins.
# The variants exist because these folders get renamed as they are retired.
_CANDIDATES = {
    "hipct_seg_debug": ("LEGACY_HIPCT_SEG_DEBUG", [
        r"F:\hipct_seg_debug",
        r"F:\hipct_seg_debug_local",
    ]),
    # Deliberately Skeleton_analysis_python, not Skeleton_analysis-main: the
    # latter is a stale snapshot that also bundles third-party MATLAB code.
    "skeleton_analysis": ("LEGACY_SKELETON_ANALYSIS", [
        r"F:\Skeleton_analysis_python",
        r"F:\Skeleton_analysis_python_local",
    ]),
    "coronary_sdf": ("LEGACY_CORONARY_SDF", [
        r"F:\coronary_sdf",
        r"F:\coronary_sdf_local",
    ]),
}


def _resolve_legacy() -> dict[str, Path]:
    out: dict[str, Path] = {}
    for pkg, (env_var, candidates) in _CANDIDATES.items():
        override = os.environ.get(env_var)
        if override:
            out[pkg] = Path(override)
            continue
        found = next((Path(c) for c in candidates if Path(c).is_dir()), None)
        # Fall back to the first candidate so the error message names a path.
        out[pkg] = found or Path(candidates[0])
    return out


LEGACY = _resolve_legacy()

# ── Files edited during publication; never clobber these ─────────────────────
# This list is not guesswork: the monorepo was assembled from these checkouts
# on 2026-09-15 and nothing else was changed, so every file that differed
# immediately afterwards is one that publication edited. That set is recorded
# here verbatim. Regenerate it with `--print-protected` after a deliberate
# merge, so the list tracks reality instead of drifting from it.
PROTECTED: dict[str, set[str]] = {
    "coronary_sdf": {
        # dataset paths replaced by environment variables
        "src/coronary_sdf/config.py",
        "src/coronary_sdf/flow_fractions.py",
        "src/coronary_sdf/simpleware_coronary_regions.py",
        "src/coronary_sdf/simpleware_coronary_regions_console.py",
        "src/coronary_sdf/simpleware_isolate_contact_console.py",
        "src/coronary_sdf/simpleware_mesh_sensitivity_console.py",
        "src/coronary_sdf/simpleware_centreline_inventory_console.py",
        "src/coronary_sdf/make_heart_overview_gif.py",
        "src/coronary_sdf/make_heart_overview_gif_fullres.py",
        # imports and path resolution fixed for the src/ layout
        "tests/test_adaptive_surface_remesh.py",
        "tests/test_simpleware_coronary_regions.py",
        "tests/test_simpleware_mesh_quality.py",
        "tests/test_real_acceptance.py",
        "tests/test_mesh_sensitivity_study.py",
        # packaging rewritten for src/ layout + metadata
        "pyproject.toml",
        "README.md",
        # F:\ paths rewritten to placeholders
        "HANDOFF_epicardial_radius.md",
        "epicardial_annotation_README.md",
        "plane_sensitivity_README.md",
        "simpleware_coronary_regions_README.md",
        # every one had its hardcoded dataset constants replaced by _paths.*,
        # its sys.path bootstrap removed, and analysis_out/ references updated
        "research_scripts/bifurcation_sweep.py",
        "research_scripts/collapse_conservation.py",
        "research_scripts/junction_damage.py",
        "research_scripts/junction_mask_confirm.py",
        "research_scripts/oblique_synthetic.py",
        "research_scripts/ostium_flare.py",
        "research_scripts/owned_junction_probe.py",
        "research_scripts/perimeter_vs_area_by_order.py",
        "research_scripts/radius_diagnosis.py",
        "research_scripts/reformat_radius.py",
        "research_scripts/segment_diagnosis.py",
        "research_scripts/shrinkage_scale.py",
        "research_scripts/subvoxel_bias.py",
    },
    "hipct_seg_debug": {
        "src/hipct_seg_debug/edit/_deps.py",
        "src/hipct_seg_debug/edit/reconnect/candidates.py",
        "src/hipct_seg_debug/edit/reconnect/endpoints.py",
        "pyproject.toml",
        "README.md",
        "docs/EDITOR.md",
        # was a raw context-compaction transcript; rewritten as a design doc
        "docs/GEODESIC_RECONNECTION.md",
        # analysis_out/subvoxel_bias.py -> research_scripts/subvoxel_bias.py
        "src/hipct_seg_debug/edit/radius_perimeter.py",
        "tests/test_radius_perimeter.py",
    },
    "skeleton_analysis": {
        "Python_port_test.py",
        "README.md",
    },
}

# Never copy these across, whatever the package.
SKIP_DIRS = {
    "__pycache__", ".git", ".pytest_cache", ".ruff_cache", ".mypy_cache",
    ".venv", "venv", "env", ".conda-cfc", "build", "dist", ".eggs",
    ".claude", ".vscode", ".idea", "logs", "cache", "runs", "models",
    "analysis_out", "benchmark_out", "roi_cases", "test_data",
    "dpc_review_regions", "pipeline_test_out", ".simpleware_scripting_api_docs",
}
SKIP_SUFFIXES = {
    ".pyc", ".pyo", ".log", ".am", ".stl", ".tif", ".tiff", ".npz", ".npy",
    ".h5", ".res", ".sip", ".def", ".msh", ".gif", ".pptx", ".chm", ".pdf",
}


@dataclass
class Plan:
    added: list[str] = field(default_factory=list)
    changed: list[str] = field(default_factory=list)
    blocked: list[str] = field(default_factory=list)   # protected and differing
    #: legacy-relative -> repo-relative, for the coronary_sdf remap
    sources: dict[str, Path] = field(default_factory=dict)


def _skip(path: Path, root: Path) -> bool:
    rel = path.relative_to(root)
    if any(part in SKIP_DIRS for part in rel.parts):
        return True
    if path.suffix.lower() in SKIP_SUFFIXES:
        return True
    return any(part.startswith(".pytest_") or part.endswith(".egg-info")
               for part in rel.parts)


def _map_coronary_sdf(rel: Path) -> str | None:
    """Flat legacy layout -> this repo's src/tests/research_scripts split."""
    parts = rel.parts
    if parts[0] == "analysis_out":
        return None  # generated output; its scripts are handled below
    if len(parts) == 1 and rel.suffix == ".py":
        name = rel.name
        if re.match(r"^(test_|_test_|_probe_|_smoke_test)", name):
            return f"tests/{name}"
        return f"src/coronary_sdf/{name}"
    if parts[0] in {"native", "docs"}:
        return str(rel).replace("\\", "/")
    if len(parts) == 1 and rel.suffix in {".md", ".toml", ".yml", ".yaml"}:
        return rel.name
    return None


def _map_src_layout(rel: Path) -> str | None:
    """hipct_seg_debug and skeleton_analysis map 1:1."""
    parts = rel.parts
    if parts[0] in {"src", "tests", "docs"}:
        return str(rel).replace("\\", "/")
    if len(parts) == 1 and rel.suffix in {".md", ".toml", ".yml", ".yaml", ".txt", ".py"}:
        return rel.name
    return None


MAPPERS = {
    "coronary_sdf": _map_coronary_sdf,
    "hipct_seg_debug": _map_src_layout,
    "skeleton_analysis": _map_src_layout,
}


def build_plan(pkg: str) -> Plan:
    legacy = LEGACY[pkg]
    if not legacy.is_dir():
        raise SystemExit(
            f"legacy checkout for {pkg} not found: {legacy}\n"
            f"Set the matching LEGACY_* environment variable if it moved."
        )
    dest_root = REPO / "packages" / pkg
    mapper = MAPPERS[pkg]
    protected = PROTECTED.get(pkg, set())
    plan = Plan()

    # analysis_out/*.py became research_scripts/ -- carried explicitly because
    # the directory itself is in SKIP_DIRS (it holds gigabytes of output).
    extra: list[tuple[Path, str]] = []
    if pkg == "coronary_sdf":
        for p in sorted((legacy / "analysis_out").glob("*.py")):
            extra.append((p, f"research_scripts/{p.name}"))

    candidates: list[tuple[Path, str]] = []
    for path in legacy.rglob("*"):
        if not path.is_file() or _skip(path, legacy):
            continue
        rel_out = mapper(path.relative_to(legacy))
        if rel_out:
            candidates.append((path, rel_out))
    candidates.extend(extra)

    for src, rel_out in candidates:
        dest = dest_root / rel_out
        plan.sources[rel_out] = src
        if not dest.exists():
            plan.added.append(rel_out)
            continue
        if src.read_bytes().replace(b"\r\n", b"\n") == dest.read_bytes().replace(b"\r\n", b"\n"):
            continue
        (plan.blocked if rel_out in protected else plan.changed).append(rel_out)
    return plan


def show_diff(pkg: str, rel: str, src: Path, limit: int = 40) -> None:
    dest = REPO / "packages" / pkg / rel
    a = dest.read_text(encoding="utf-8", errors="replace").splitlines()
    b = src.read_text(encoding="utf-8", errors="replace").splitlines()
    diff = list(difflib.unified_diff(a, b, fromfile=f"monorepo/{rel}",
                                     tofile=f"legacy/{rel}", lineterm=""))
    for line in diff[:limit]:
        print("    " + line)
    if len(diff) > limit:
        print(f"    ... {len(diff) - limit} more diff lines")


LEAK = re.compile(
    r"([A-Za-z]:[\\/]{1,2}(?:Users|Edo|Final|coronary_sdf|hipct_seg_debug|Skeleton))"
    r"|/mnt/|Users[\\/]{1,2}Akash"
)
# Deliberate documentation examples and instructional text, not real paths.
LEAK_OK = re.compile(r"D:[\\/]data|D:\\src|D:\\a b|C:\\data|C:\\path|Program Files|ProgramData")


def scan_for_leaks() -> list[str]:
    """Re-run the guard that gated the initial publication."""
    hits: list[str] = []
    for path in (REPO / "packages").rglob("*"):
        if not path.is_file() or path.suffix not in {".py", ".json", ".toml", ".yml", ".md"}:
            continue
        if any(p in SKIP_DIRS for p in path.relative_to(REPO).parts):
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        for n, line in enumerate(text.splitlines(), 1):
            if LEAK.search(line) and not LEAK_OK.search(line):
                hits.append(f"{path.relative_to(REPO)}:{n}: {line.strip()[:110]}")
    return hits


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--apply", action="store_true",
                    help="actually copy (default is a dry run)")
    ap.add_argument("--dry-run", action="store_true",
                    help="report only; this is the default, the flag is for clarity")
    ap.add_argument("--only", choices=sorted(LEGACY), action="append",
                    help="limit to one package (repeatable)")
    ap.add_argument("--force-protected", action="store_true",
                    help="also overwrite the protected files -- check the diffs first")
    ap.add_argument("--diff", action="store_true",
                    help="show a diff for every blocked file")
    ap.add_argument("--print-protected", action="store_true",
                    help="emit a PROTECTED block covering everything that currently "
                         "differs, to paste back into this file after a merge")
    args = ap.parse_args(argv)
    if args.dry_run and args.apply:
        ap.error("--dry-run and --apply are mutually exclusive")

    packages = args.only or sorted(LEGACY)

    if args.print_protected:
        print("PROTECTED: dict[str, set[str]] = {")
        for pkg in packages:
            plan = build_plan(pkg)
            everything = sorted(set(plan.changed) | set(plan.blocked))
            print(f'    "{pkg}": {{')
            for rel in everything:
                print(f'        "{rel}",')
            print("    },")
        print("}")
        return 0

    total_written = 0
    total_blocked = 0

    for pkg in packages:
        plan = build_plan(pkg)
        n = len(plan.added) + len(plan.changed)
        print(f"\n=== {pkg} ===")
        print(f"  legacy: {LEGACY[pkg]}")
        if not n and not plan.blocked:
            print("  up to date")
            continue

        for rel in plan.added:
            print(f"  + new      {rel}")
        for rel in plan.changed:
            print(f"  ~ changed  {rel}")

        if plan.blocked:
            total_blocked += len(plan.blocked)
            print(f"\n  {len(plan.blocked)} PROTECTED file(s) differ and were NOT copied.")
            print("  These were edited during publication (dataset paths removed,")
            print("  imports fixed, docs rewritten). Merge by hand, or re-run with")
            print("  --force-protected once you have read the diffs.\n")
            for rel in plan.blocked:
                print(f"  ! blocked  {rel}")
                if args.diff:
                    show_diff(pkg, rel, plan.sources[rel])

        if args.apply:
            to_write = list(plan.added) + list(plan.changed)
            if args.force_protected:
                to_write += plan.blocked
            for rel in to_write:
                dest = REPO / "packages" / pkg / rel
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(plan.sources[rel], dest)
            total_written += len(to_write)
            print(f"\n  wrote {len(to_write)} file(s)")

    if not args.apply:
        print("\nDry run. Nothing was written. Re-run with --apply.")
        return 0

    print(f"\nwrote {total_written} file(s) in total")

    leaks = scan_for_leaks()
    if leaks:
        print(f"\nLOCAL PATHS DETECTED in {len(leaks)} line(s) -- fix before committing:")
        for h in leaks[:40]:
            print("  " + h)
        if len(leaks) > 40:
            print(f"  ... {len(leaks) - 40} more")
        return 1

    print("path scan clean")
    if total_blocked:
        print(f"note: {total_blocked} protected file(s) still differ from legacy")
    print("\nNext: review with `git diff`, run the tests, then commit.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
