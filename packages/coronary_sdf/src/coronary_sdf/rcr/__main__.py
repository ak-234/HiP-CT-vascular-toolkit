"""Reproducible coronary RCR case commands."""
import argparse
import csv
import json
from pathlib import Path
import sys
import subprocess

from .prepare import prepare, write_json, digest
from .calibration import calibrate_case


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def main(argv=None):
    parser = argparse.ArgumentParser(prog="coronary-rcr")
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("prepare", "calibrate", "export-cfx", "compare"):
        p = sub.add_parser(name)
        p.add_argument("--config", type=Path, required=True)
        p.add_argument("--out", type=Path, required=True, help="New or empty output directory")
        if name == "prepare":
            p.add_argument("--audit-only", action="store_true")
            p.add_argument("--surfaces", action="store_true")
        elif name == "calibrate":
            p.add_argument("--network", type=Path)
            p.add_argument("--methods", nargs="+", default=["diameter", "structured", "hipct"],
                           choices=["diameter", "structured", "hipct", "legacy-giessen"])
        elif name == "export-cfx":
            p.add_argument("--results", type=Path, required=True)
            p.add_argument("--tree", required=True)
            p.add_argument("--method", required=True)
            p.add_argument("--dry-run", action="store_true")
            p.add_argument("--compile", action="store_true", help="Build exported source copies with cfx5mkext")
        else:
            p.add_argument("--results", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        config = read_json(args.config)
        # Relative data paths are relative to the case file, independent of cwd.
        base = args.config.resolve().parent
        for section, keys in (("geometry", ("graph", "segmentation")),
                              ("cfx", ("mesh", "user_lib_dir", "cfx_pre", "restart")),
                              ("comparison", ("observations",))):
            for key in keys:
                value = config.get(section, {}).get(key)
                if value:
                    config[section][key] = str((base/Path(value)).resolve())
        output = args.out.resolve()
        if output.exists() and any(output.iterdir()):
            raise ValueError("Output directory must be empty; existing results are never overwritten")
        output.mkdir(parents=True, exist_ok=True)
        write_json(output/"case.resolved.json", config)
        revision = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False)
        sources = {p.name: digest(p) for p in Path(__file__).parent.glob("*.py")}
        write_json(output/"run.json", {"command": args.command, "config_sha256": digest(args.config),
                   "git_revision": revision.stdout.strip() if revision.returncode == 0 else None,
                   "source_sha256": sources})
        if args.command == "prepare":
            prepare(config, output, audit_only=args.audit_only, surfaces=args.surfaces)
        elif args.command == "calibrate":
            network_path = args.network or base/Path(config["network"])
            result = calibrate_case(read_json(network_path), config, args.methods)
            result["network_sha256"] = digest(network_path)
            write_json(output/"calibration.json", result)
            rows = [r for tree in result["results"] for r in tree["outlets"]]
            with (output/"outlets.csv").open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
        elif args.command == "export-cfx":
            from .cfx import export
            export(read_json(args.results), config, args.tree, args.method, output, args.dry_run, args.compile)
        else:
            from .compare import compare
            write_json(output/"comparison.json", compare(read_json(args.results), config))
        print(f"{args.command}: completed; {output}")
        return 0
    except (ValueError, KeyError, OSError) as exc:
        print(f"coronary-rcr: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
