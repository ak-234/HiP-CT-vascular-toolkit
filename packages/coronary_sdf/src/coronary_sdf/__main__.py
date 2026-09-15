"""CLI entry: ``python -m coronary_sdf [options] [input output_dir]``."""

from __future__ import annotations

import argparse

from . import config
from .pipeline import run_pipeline
from .profiles import ProfileUnavailable, resolve_profile


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Reconstruct a coronary lumen surface")
    parser.add_argument("input", nargs="?")
    parser.add_argument("output_dir", nargs="?")
    parser.add_argument(
        "--profile", choices=("compat", "experimental", "cfd"), default="compat"
    )
    parser.add_argument(
        "--qualified-config",
        help="qualified CFD profile emitted by the acceptance benchmark",
    )
    parser.add_argument(
        "--cells-across-diameter",
        type=float,
        help="override the adaptive surface resolution (minimum sqrt(3))",
    )
    parser.add_argument(
        "--max-cells",
        type=int,
        help="override the VTK HyperTreeGrid hierarchy cell safety limit",
    )
    parser.add_argument(
        "--validation-mode",
        choices=("off", "warn", "error"),
        help=(
            "override final mesh validation; 'warn' saves an unvalidated mesh "
            "for diagnosis, while 'error' rejects it"
        ),
    )
    parser.add_argument(
        "--component-failure",
        choices=("error", "continue"),
        help=(
            "what a failing connected component does: 'error' aborts the run "
            "(default), 'continue' records it and reconstructs the rest. "
            "component_manifest.json is written either way"
        ),
    )
    parser.add_argument("--non-interactive", action="store_true")
    parser.add_argument("--no-write", action="store_true")
    args = parser.parse_args(argv)
    if (args.input is None) != (args.output_dir is None):
        parser.error("input and output_dir must be supplied together")
    xml = config.INPUT_PATH if args.input is None else args.input
    out = config.OUTPUT_DIR if args.output_dir is None else args.output_dir
    try:
        cfg = resolve_profile(args.profile, qualified_path=args.qualified_config)
    except ProfileUnavailable as exc:
        parser.exit(2, f"error: {exc}\n")
    overrides = {}
    if args.cells_across_diameter is not None:
        if args.cells_across_diameter < 3**0.5:
            parser.error("--cells-across-diameter must be at least sqrt(3)")
        overrides["IMPLICIT_CELLS_ACROSS_DIAMETER"] = args.cells_across_diameter
    if args.max_cells is not None:
        if args.max_cells < 1:
            parser.error("--max-cells must be positive")
        overrides["VTK_HTG_MAX_CELLS"] = args.max_cells
    if args.validation_mode is not None:
        if args.profile == "cfd" and args.validation_mode != "error":
            parser.error("the qualified CFD profile requires --validation-mode error")
        overrides["OUTPUT_VALIDATION_MODE"] = args.validation_mode
    if args.component_failure is not None:
        if args.profile == "cfd" and args.component_failure != "error":
            parser.error(
                "the qualified CFD profile requires --component-failure error; "
                "a partial network is not CFD-qualified"
            )
        overrides["PIPELINE_COMPONENT_FAILURE"] = args.component_failure
    if overrides:
        cfg = cfg.with_overrides(**overrides)
    print("=" * 60)
    print("CORONARY LUMEN -- SDF pipeline")
    print(f"  profile={args.profile}")
    print(f"  field={cfg.SDF_FIELD_METHOD}/{cfg.IMPLICIT_PRIMITIVE_METHOD}")
    print(f"  mesh method={cfg.SDF_MESH_METHOD}")
    if cfg.SDF_MESH_METHOD in {"adaptive", "vtk_htg", "cgal_mesh3"}:
        print(f"  cells across diameter={cfg.IMPLICIT_CELLS_ACROSS_DIAMETER:g}")
    if cfg.SDF_MESH_METHOD == "vtk_htg":
        print(f"  HTG maximum cells={cfg.VTK_HTG_MAX_CELLS:,}")
    if args.profile == "experimental":
        print("  WARNING: experimental graph profile is not qualified for production CFD")
    print(f"  component failure={cfg.PIPELINE_COMPONENT_FAILURE}")
    if cfg.OUTPUT_VALIDATION_MODE != "error":
        print(
            "  WARNING: validation mode is "
            f"{cfg.OUTPUT_VALIDATION_MODE!r}; written meshes are not CFD-qualified"
        )
    print("=" * 60)
    run_pipeline(
        xml,
        out,
        cfg=cfg,
        write_outputs=not args.no_write,
        interactive=False if args.non_interactive else None,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
