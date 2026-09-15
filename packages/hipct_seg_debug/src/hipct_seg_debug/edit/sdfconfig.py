"""Safely override ``coronary_sdf.config`` for the duration of a call.

``coronary_sdf`` reads its settings from **module globals** at call time. The
``SdfConfig`` dataclass at ``config.py:875`` looks like the configuration API
but nothing in the pipeline reads it -- ``manual_prune.py`` and
``epicardial_annotation.py`` both reconfigure by assigning to
``coronary_sdf.config.X`` directly. That works, but it is process-global state:
an override left behind by one call silently changes the next.

Hence :func:`sdf_config`, which snapshots every setting, applies the overrides,
and restores the snapshot on the way out -- including when the body raises.

The interactive defaults are not optional. ``DEBUG_VIS`` and ``DEBUG_VIS_BLOCK``
both default to ``True`` (``config.py:855-857``), so an unguarded
``generate_sdf_surface`` opens roughly six **blocking** PyVista windows per
call. Inside a viewer that reads as a hang.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Iterator

from ._deps import ensure_coronary_sdf

# Forced off for any pipeline call made from the viewer. Debug windows block the
# Qt event loop; the region VTK export re-derives and re-smooths the whole tree
# for a file nobody asked for.
HEADLESS: dict[str, Any] = {
    "DEBUG_VIS": False,
    "DEBUG_VIS_BLOCK": False,
    "DEBUG_VIS_SAVE_FALLBACK": False,
    "PREVIEW_SDF_BEFORE_MC": False,
    "WRITE_REGION_VTK": False,
}

# Additional overrides for the live patch rebuild, where latency beats polish.
# Both trade *preview* fidelity only -- the exported surface is produced by a
# separate, unmodified full run. `SDF_MAX_CAPSULE_QUERY` is the k of the
# per-voxel KD query and dominates evaluation cost; 16 is the value the legacy
# octree pipeline shipped with.
PREVIEW: dict[str, Any] = {
    **HEADLESS,
    "SDF_MAX_CAPSULE_QUERY": 16,
    "THIN_VESSEL_REFINE": False,
    "PROXIMITY_DIAGNOSTIC_TOP_N": 0,
    "TERMINAL_CLAMP_VERBOSE": False,
    "BLEND_DIAGNOSTIC": False,
}


# Settings `coronary_sdf` *reads* but never *declares*, so they do not appear in
# ``config.py``'s globals and would otherwise be rejected here as typos.
#
# ``CENTERLINE_MAX_DRIFT_RADIUS_FACTOR`` is read at
# ``centerline_optimizer.py:355`` as ``getattr(config, ..., .25)`` and is the only
# knob governing how far the constrained multiscale smoother may move a point --
# its trust region, in local radii. Undeclared, it is effectively hard-coded at
# 0.25 and cannot be swept. Naming it here keeps the guard against mistyped
# settings while making the one parameter that matters tunable.
#
# **Delete an entry once upstream declares it**, or this silently shadows the
# check that would have caught a rename.
UNDECLARED: tuple[str, ...] = ("CENTERLINE_MAX_DRIFT_RADIUS_FACTOR",)


def _settings(config) -> dict[str, Any]:
    """The module's settings: upper-case names that are not imports or types."""
    out = {}
    for name, value in vars(config).items():
        if not name.isupper():
            continue
        if isinstance(value, type) or callable(value):
            continue
        out[name] = value
    return out


@contextmanager
def sdf_config(profile: dict[str, Any] | None = None, **overrides: Any) -> Iterator[Any]:
    """Apply settings to ``coronary_sdf.config``, restoring them afterwards.

    ``profile`` is a base dict (:data:`HEADLESS` or :data:`PREVIEW`); keyword
    arguments layer on top of it. Yields the config module so a caller can read
    values that were not overridden::

        with sdf_config(PREVIEW, BSPLINE_SDF_RESOLUTION=0.05) as cfg:
            surface = rebuild(...)          # cfg.SMIN_PROXIMITY_BLEND_FACTOR etc.

    An unknown setting name is an error rather than a silent no-op: a typo'd
    override that quietly does nothing is the worst possible outcome here.
    """
    ensure_coronary_sdf()
    from coronary_sdf import config

    wanted = {**(profile if profile is not None else HEADLESS), **overrides}
    known = _settings(config)
    unknown = sorted(set(wanted) - set(known) - set(UNDECLARED))
    if unknown:
        raise KeyError(
            f"unknown coronary_sdf.config setting(s): {', '.join(unknown)}"
        )

    # An undeclared name has nothing to restore *to*, so it is removed on the way
    # out rather than set back -- putting a value there permanently would change
    # the default every later `getattr(config, name, fallback)` sees.
    missing = [name for name in wanted if name not in known]
    snapshot = {name: known[name] for name in wanted if name in known}
    try:
        for name, value in wanted.items():
            setattr(config, name, value)
        yield config
    finally:
        for name, value in snapshot.items():
            setattr(config, name, value)
        for name in missing:
            try:
                delattr(config, name)
            except AttributeError:
                pass
