"""Named CLI configuration profiles and qualification gating."""

from __future__ import annotations

import json
from pathlib import Path

from .config import SdfConfig, default_config


class ProfileUnavailable(RuntimeError):
    pass


def compatibility_profile() -> SdfConfig:
    return default_config()


def candidate_cfd_config() -> SdfConfig:
    """Unqualified configuration used by the benchmark, never by default CLI."""

    return default_config().with_overrides(
        CENTERLINE_SMOOTHER="none",
        LIMIT_CENTERLINE_CURVATURE=False,
        CENTERLINE_CONSTRAINT_FAILURE="error",
        PRESERVE_INPUT_RADII=True,
        SDF_FIELD_METHOD="graph_implicit",
        IMPLICIT_PRIMITIVE_METHOD="round_cone",
        IMPLICIT_JUNCTION_BLEND_FRACTION=0.05,
        IMPLICIT_JUNCTION_SUPPORT_FACTOR=2.0,
        SDF_MESH_METHOD="vtk_htg",
        OUTPUT_VALIDATION_MODE="error",
        OUTPUT_VALIDATE_SELF_INTERSECTIONS=True,
        MESH_REPAIR=False,
        TAUBIN_ITERS=0,
    )


def experimental_graph_profile() -> SdfConfig:
    """Runnable graph/adaptive candidate without CFD qualification claims.

    Deliberately byte-identical to :func:`candidate_cfd_config`, including the
    fail-fast component policy: its output has to be representative of what the
    CFD candidate would produce. Diagnostic continuation is a property of the
    benchmark harness (``benchmark.candidate_config``), not of this profile.
    """

    return candidate_cfd_config()


def qualified_profile_path() -> Path:
    return Path(__file__).with_name("cfd_qualified_profile.json")


def cfd_profile(path: str | Path | None = None) -> SdfConfig:
    profile_path = qualified_profile_path() if path is None else Path(path)
    if not profile_path.exists():
        raise ProfileUnavailable(
            "CFD profile is unavailable: no candidate has passed every synthetic "
            "and LADAF acceptance gate. Run coronary_sdf.benchmark; do not use "
            "the unqualified adaptive candidate for production CFD."
        )
    payload = json.loads(profile_path.read_text(encoding="utf-8"))
    if payload.get("qualified") is not True:
        raise ProfileUnavailable(f"{profile_path} is not marked qualified")
    if payload.get("diagnostic_only"):
        raise ProfileUnavailable(
            f"{profile_path} was emitted from diagnostic runs and cannot qualify CFD"
        )
    if payload.get("winner") != "graph_round_cone_vtk_htg":
        raise ProfileUnavailable(
            f"{profile_path} winner is not the exact VTK HyperTreeGrid candidate"
        )
    cases = set(payload.get("real_cases", []))
    if cases != {"LADAF_2024_28", "LADAF_2024_56"}:
        raise ProfileUnavailable(f"{profile_path} does not certify both LADAF cases")
    # Every real case that certified the winner must itself have had a clean
    # input audit; an empty list means the benchmark never recorded one.
    audited = set(payload.get("input_qualified_cases", []))
    if audited and not cases <= audited:
        raise ProfileUnavailable(
            f"{profile_path} certifies case(s) whose input audit did not qualify"
        )
    config_values = payload.get("config")
    if not isinstance(config_values, dict):
        raise ProfileUnavailable(f"{profile_path} has no frozen configuration")
    cfg = SdfConfig(**config_values)
    required = candidate_cfd_config()
    for field in (
        "CENTERLINE_SMOOTHER",
        "PRESERVE_INPUT_RADII",
        "SDF_FIELD_METHOD",
        "IMPLICIT_PRIMITIVE_METHOD",
        "IMPLICIT_JUNCTION_BLEND_FRACTION",
        "IMPLICIT_JUNCTION_SUPPORT_FACTOR",
        "SDF_MESH_METHOD",
        "OUTPUT_VALIDATION_MODE",
        "OUTPUT_VALIDATE_SELF_INTERSECTIONS",
        "IMPLICIT_CELLS_ACROSS_DIAMETER",
        "IMPLICIT_ADAPTIVE_MAX_DEPTH",
        "VTK_HTG_PADDING_RADIUS_FACTOR",
        "VTK_HTG_MAX_CELLS",
        "VTK_HTG_DECOMPOSED_POLYHEDRA",
    ):
        if getattr(cfg, field) != getattr(required, field):
            raise ProfileUnavailable(f"{profile_path} violates required CFD setting {field}")
    return cfg


def resolve_profile(name: str, *, qualified_path: str | Path | None = None) -> SdfConfig:
    if name == "compat":
        return compatibility_profile()
    if name == "experimental":
        return experimental_graph_profile()
    if name == "cfd":
        return cfd_profile(qualified_path)
    raise ValueError(f"unknown profile {name!r}")


__all__ = [
    "ProfileUnavailable",
    "candidate_cfd_config",
    "cfd_profile",
    "compatibility_profile",
    "experimental_graph_profile",
    "qualified_profile_path",
    "resolve_profile",
]
