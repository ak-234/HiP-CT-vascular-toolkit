"""Four centreline smoothers behind one signature, so the metric can choose.

Smoothing is the one stage of :mod:`~.skeleton_optimise` that measurement did *not*
vindicate: at stride 4 on LADAF-2024-28 it was the only step to make the super metric
worse (cl-sensitivity 0.951 -> 0.926), and the sweep chose no smoothing at all. Rather
than tune a filter of my own further, this dispatches to the ones ``coronary_sdf``
already has -- and lets :mod:`~.supermetric` decide between them.

===============  ==========================================================
``none``         no-op, so "do not smooth" is a first-class sweep point
``gaussian``     :func:`~.skeleton_optimise.smooth_centreline`, the built-in
``savgol``       ``coronary_sdf`` Gen 1: Savitzky-Golay + curvature post-filter
``bspline``      ``coronary_sdf`` Gen 1: chord-length B-spline + the same filter
``multiscale``   ``coronary_sdf`` Gen 2: constrained multiscale optimiser
===============  ==========================================================

**What "multiscale" means**, since it is not a coarse-to-fine pyramid: a
penalised-spline energy summed over three bandwidths of ``(1, 2, 4)`` *local radii*,
in radius-normalised arc length, solved graph-wide by ADMM against three projections --
a trust region (drift <= 0.25 r), a curvature bound (kappa*r <= 0.95) and tube-tube
clearance. Everything being measured in radii makes it globally scale-equivariant, which
its own test suite pins to 1e-12. The constraints *participate in the fit* rather than
moving points after it, which is the specific criticism ``coronary_sdf_refinement_plan.txt``
levels at Gen 1's fixed 51-sample window: "not invariant to sampling density or vessel
length... segmentwise fitting also cannot prevent one branch moving into another".

**It is nonetheless experimental.** That same plan rates it "Repaired but not qualified
-- it still fails synthetic curvature/contact constraints and the measured LADAF-56
convergence/performance audit, so it remains opt-in", and the ``coronary_sdf`` README
agrees. So ``gaussian`` stays the default here until the sweep says otherwise *on this
data*; qualifying it is exactly what ``optimise-skeleton --sweep "smoother=..."`` is for.

Two behaviours to read carefully rather than mistake for bugs, both Gen 2's:

* **Preserve-only.** Where the input already self-overlaps, it freezes those points and
  their one-radius halo. On a badly overlapping graph it can legitimately return the
  input unchanged, with ``modified_points == 0``.
* **Backtracking.** It ends with a certified line search, accepting the largest fraction
  of its own solution that introduces no new contact. ``backtrack_fraction < 1`` means it
  deliberately kept less than it computed.

Radii are never smoothed. ``coronary_sdf`` has four radius passes and they all stay off
(``PRESERVE_INPUT_RADII``): :mod:`~.radius_perimeter` has just measured every radius from
its own cross-section, and a modelled value has no business overwriting a measured one.
:func:`smooth` asserts the radii come back untouched.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

#: Selectable backends, in increasing order of how much they assume.
SMOOTHERS = ("none", "gaussian", "savgol", "bspline", "multiscale")

#: Overrides applied to ``coronary_sdf.config`` for every Gen 1 / Gen 2 call.
#: Both verbose flags default to True upstream and are extremely chatty per segment.
QUIET = {
    "PRESERVE_INPUT_RADII": True,
    "SMOOTH_DRIFT_VERBOSE": False,
    "CURVATURE_VERBOSE": False,
    "DENSIFY_VERBOSE": False,
}


@dataclass
class SmoothResult:
    """What one smoothing pass did, and whatever the backend reported about it."""

    smoother: str
    n_moved: int = 0
    median_move_um: float = 0.0
    max_move_um: float = 0.0
    seconds: float = 0.0
    detail: dict = field(default_factory=dict)
    notes: list = field(default_factory=list)

    def describe(self) -> str:
        line = (
            f"smooth [{self.smoother}]: {self.n_moved} point(s) moved, median "
            f"{self.median_move_um:.1f} um, max {self.max_move_um:.1f} um "
            f"({self.seconds:.1f}s)"
        )
        for note in self.notes:
            line += f"\n    {note}"
        return line


def _apply_points(graph, new_points: dict) -> tuple[int, float, float]:
    """Write a ``{point id: (x, y, z, r)}`` dict back, segment by segment.

    The backends return a whole points dict keyed by the ids they were given, so the
    write-back goes through :meth:`~.graphmodel.EditableGraph.set_segment_coords` --
    ``move_point`` scans every segment per call, which on 37k points is tens of
    millions of dict lookups.
    """
    moves: list[float] = []
    with graph.batch("smooth centreline"):
        for sid in graph.segment_ids():
            pids = graph.segment(sid)["point_ids"]
            if not pids:
                continue
            old = graph.coords(sid)
            new = np.array(
                [new_points[p][:3] if p in new_points else old[i]
                 for i, p in enumerate(pids)],
                dtype=np.float64,
            ).reshape(-1, 3)
            dist = np.linalg.norm(new - old, axis=1)
            if not (dist > 1e-9).any():
                continue
            graph.set_segment_coords(sid, new)
            moves.extend(dist[dist > 1e-9].tolist())
    if not moves:
        return 0, 0.0, 0.0
    return len(moves), float(np.median(moves)), float(np.max(moves))


def _radii_snapshot(graph) -> dict:
    return {sid: graph.radii(sid).copy() for sid in graph.segment_ids()}


def _check_radii_untouched(graph, before: dict, result: SmoothResult) -> None:
    """A smoother that rewrote a radius has undone :mod:`~.radius_perimeter`.

    Cheap, and worth it: every radius in this pipeline is a measurement from that
    point's own cross-section, and a centreline smoother silently blending them would
    be invisible in the geometry and fatal to the one number this all exists to fix.
    """
    for sid, old in before.items():
        if not graph.has_segment(sid):
            continue
        now = graph.radii(sid)
        if len(now) == len(old) and np.allclose(now, old, rtol=0, atol=1e-9):
            continue
        raise AssertionError(
            f"{result.smoother} changed the radii of segment {sid}; centreline "
            f"smoothing must leave the measured radius alone"
        )


#: A pass that shifts the centreline by less than this fraction of a radius has not
#: smoothed anything, whatever its point count says.
INERT_MOVE_FRAC = 0.01


def _warn_if_inert(graph, result: SmoothResult) -> None:
    """Say so when a backend moved plenty of points by nothing at all.

    ``n_moved`` alone is a trap. Measured on LADAF-2024-28 at stride 1, ``multiscale``
    reported 24,037 of 36,613 points moved -- and moved them a *thousandth of a radius*,
    0.2 um, because its own certified line search had scaled the solution it computed by
    0.0046. The output was geometrically identical to not smoothing at all, while every
    number in the report said the pass had done something. That is how "inert" came to be
    read as "broken smoother"; this note is the fix for the reporting half of it.
    """
    radii = [graph.radii(sid) for sid in graph.segment_ids()]
    radii = [r for r in radii if len(r)]
    if not radii:
        return
    typical = float(np.median(np.concatenate(radii)))
    if typical <= 0 or result.median_move_um >= INERT_MOVE_FRAC * typical:
        return
    line = (
        f"[warning] effectively inert: {result.n_moved} point(s) moved, but the median "
        f"move is {result.median_move_um:.2f} um against a median radius of "
        f"{typical:.0f} um ({100 * result.median_move_um / typical:.2f}% of a radius)"
    )
    frac = result.detail.get("backtrack_fraction")
    if frac is not None and frac < 1.0:
        line += (
            f"; its line search kept {100 * frac:.2f}% of the solution it computed, so "
            f"the constraints rejected it rather than the fit failing"
        )
    result.notes.append(line)


def _coronary_gen1(graph, kind: str, result: SmoothResult) -> dict:
    """Savitzky-Golay or B-spline, then the curvature post-filter, as the pipeline does."""
    from ._deps import ensure_coronary_sdf
    from .sdfconfig import HEADLESS, sdf_config

    ensure_coronary_sdf()
    from coronary_sdf.smoothing import (
        limit_centerline_curvature,
        smooth_segment_centerlines,
    )

    nodes, points, segments = graph.triple.as_args()
    with sdf_config({**HEADLESS, **QUIET}, CENTERLINE_SMOOTHER=kind):
        points, n_smoothed = smooth_segment_centerlines(nodes, points, segments)
        # Gen 1 fits each segment on its own, so curvature is corrected afterwards by
        # moving points that the fit already placed. `pipeline._generate_sdf_surface`
        # runs it in exactly this order and only for these two smoothers.
        points, n_curved = limit_centerline_curvature(nodes, points, segments)
    result.detail = {"segments_smoothed": int(n_smoothed),
                     "segments_curvature_limited": int(n_curved)}
    if n_curved:
        result.notes.append(
            f"{n_curved} segment(s) had a self-intersecting bend straightened "
            f"afterwards (kappa*r > 1)"
        )
    return points


def _coronary_multiscale(graph, result: SmoothResult,
                         drift_radius_factor: float | None = None) -> dict:
    """The Gen 2 constrained multiscale optimiser, with its report kept.

    `drift_radius_factor` is its trust region, in local radii. Upstream never declares
    it -- ``centerline_optimizer.py:355`` reads it as
    ``getattr(config, "CENTERLINE_MAX_DRIFT_RADIUS_FACTOR", .25)`` -- so it is
    effectively hard-coded at 0.25 unless something puts it there, which
    :data:`~.sdfconfig.UNDECLARED` exists to permit. Worth exposing: on LADAF-2024-28
    at stride 4 the reported drift came back with *max and p95 both exactly 0.250*,
    i.e. the cap binds for most of the tree rather than for outliers, so it is the
    parameter most likely to be limiting the result.
    """
    from ._deps import ensure_coronary_sdf
    from .sdfconfig import HEADLESS, sdf_config

    ensure_coronary_sdf()
    from coronary_sdf.centerline_optimizer import (
        smooth_centerlines_constrained_multiscale,
    )

    extra = ({} if drift_radius_factor is None
             else {"CENTERLINE_MAX_DRIFT_RADIUS_FACTOR": float(drift_radius_factor)})
    nodes, points, segments = graph.triple.as_args()
    with sdf_config({**HEADLESS, **QUIET}, **extra):
        # No curvature post-filter: curvature is one of this solver's own constraints,
        # and the pipeline skips `limit_centerline_curvature` in this mode for that
        # reason. Running it here would move points the solver had certified.
        points, report = smooth_centerlines_constrained_multiscale(
            nodes, points, segments
        )

    result.detail = report.to_dict()
    if not report.converged:
        result.notes.append(
            f"[warning] did not converge in {report.iterations} iteration(s)"
        )
    if report.unresolved_constraints:
        result.notes.append(
            f"[warning] {report.unresolved_constraints} constraint(s) left unresolved "
            f"({report.curvature_violations_after} curvature, "
            f"{report.new_branch_conflicts} new branch conflict(s))"
        )
    if report.input_overlaps:
        result.notes.append(
            f"{report.input_overlaps} pre-existing overlap(s): those points and their "
            f"one-radius halo were frozen ({report.frozen_points} frozen in total)"
        )
    if report.backtrack_fraction < 1.0:
        result.notes.append(
            f"certified line search kept {100 * report.backtrack_fraction:.0f}% of the "
            f"solution; the rest would have created new contact"
        )
    result.notes.append(
        f"drift max {report.max_displacement_radius:.3f} r, "
        f"p95 {report.p95_displacement_radius:.3f} r; "
        f"curvature violations {report.curvature_violations_before} -> "
        f"{report.curvature_violations_after}"
    )
    return points


def smooth(name: str, graph, *, window_um: float | None = None,
           max_move_frac: float | None = None,
           drift_radius_factor: float | None = None,
           verbose: bool = False) -> SmoothResult:
    """Smooth `graph`'s centreline with the named backend. Radii are never touched.

    `window_um` and `max_move_frac` apply to ``gaussian`` only, `drift_radius_factor`
    to ``multiscale`` only; ``savgol`` and ``bspline`` take everything from
    ``coronary_sdf.config``, which :func:`~.sdfconfig.sdf_config` sets and restores
    around the call.
    """
    name = (name or "none").lower()
    if name not in SMOOTHERS:
        raise ValueError(f"unknown smoother {name!r}; choose from {SMOOTHERS}")

    result = SmoothResult(smoother=name)
    t0 = time.time()
    if name == "none":
        result.seconds = time.time() - t0
        return result

    before = _radii_snapshot(graph)

    if name == "gaussian":
        from .skeleton_optimise import smooth_centreline

        kw = {}
        if window_um is not None:
            kw["window_um"] = window_um
        if max_move_frac is not None:
            kw["max_move_frac"] = max_move_frac
        move = smooth_centreline(graph, label=f"smooth [{name}]", **kw)
        result.n_moved = move.n_moved
        result.median_move_um = move.median_move_um
        result.max_move_um = move.max_move_um
        result.detail = {"clamped": move.n_clamped,
                         "window_too_small": move.n_window_too_small}
        if move.n_window_too_small:
            result.notes.append(
                f"[warning] {move.n_window_too_small} segment(s) had no neighbouring "
                f"point inside the window, so nothing was smoothed there"
            )
    else:
        points = (
            _coronary_multiscale(graph, result, drift_radius_factor)
            if name == "multiscale"
            else _coronary_gen1(graph, name, result)
        )
        result.n_moved, result.median_move_um, result.max_move_um = _apply_points(
            graph, points
        )
        if result.n_moved == 0:
            result.notes.append(
                "nothing moved -- for 'multiscale' on an overlapping graph this is its "
                "documented preserve-only policy, not a failure"
            )
        else:
            _warn_if_inert(graph, result)

    _check_radii_untouched(graph, before, result)
    result.seconds = time.time() - t0
    if verbose:
        print("   ", result.describe())
    return result
