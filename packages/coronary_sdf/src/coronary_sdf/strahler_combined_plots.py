"""Overlay figures for the left / right / combined comparison.

`strahler_plots` draws one analysis at a time, which answers "what does this tree
look like" but not "does pooling the two change the answer" -- for that the three
have to sit on the same axes. This module reads only the comparison CSVs written
by :mod:`coronary_sdf.strahler_combined`, so every plotted number traces back to
a table.

Two figures, because the two x-axes answer different questions:

``fig_compare_by_strahler``
    grouped bars per Strahler order. Order is a topological position, so this is
    the conventional morphometric view -- but order *n* means a different
    anatomical depth in a 4-order tree than in a 5-order one, which is exactly
    what the figure is there to expose.
``fig_compare_by_radius``
    the same metrics against vessel calibre on shared bins. A radius bin means
    the same physical vessel size in both trees, so this comparison carries no
    assumption about matching topology. It is the one to trust where the two
    disagree.
``fig_cfd_by_radius``
    the haemodynamics against calibre with the pooled curve dropped -- left and
    right as two curves, which is the figure to read when the question is how the
    trees differ rather than what pooling does to them. ``single_panels/`` carries
    one slide-sized version per CFD metric.
``fig_pooling_shift``
    how far the pooled median sits from the n-weighted mean of the parts.

Bars and points show the **median** with the **interquartile range**, matching
`strahler_plots` and for the same reason: the haemodynamic distributions are
right-skewed enough that a symmetric mean +/- SD bar reaches below zero, which is
impossible for a magnitude.

Pressure is absent from the pooled figures: left and right are separate solves,
each with its static pressure referenced to its own outlets at 0 Pa, so a
*combined* pressure curve averages two different zeros. It does appear in
``single_panels/``, where the two trees stay separate curves and nothing is
mixed -- annotated there, because the vertical offset between them is a
difference of reference, not of pressure. The within-tree gradient down the
calibre range is the part that means something.

Usage::

    python -m coronary_sdf.strahler_combined_plots --in analysis_out/combined_lr
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter, ScalarFormatter

from .strahler_plots import COL1, COL2, apply_style, save

# Categorical slots 1-3 of the validated reference palette, assigned in fixed
# order and never cycled: left, right, then the pooled series. The same three
# `strahler_plots` uses, so a panel from either module reads the same way.
LEFT_C, RIGHT_C, COMB_C = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK_SOFT = "#0b0b0b", "#52514e"

SERIES = (("left", "Left tree", LEFT_C),
          ("right", "Right tree", RIGHT_C),
          ("combined", "Combined", COMB_C))

# The two trees without the pooled curve, for the figures whose question is
# "how do the trees differ" rather than "what does pooling do". Same slots in the
# same order, so left is blue and right orange in every figure here.
SERIES_LR = SERIES[:2]

# Panels for the Strahler figure: two morphometric, two haemodynamic. Pressure is
# excluded on purpose (see the module docstring).
PANELS: list[tuple[str, str, bool]] = [
    ("radius_mean_mm", "Vessel radius (mm)", False),
    ("csa_mm2", "Cross-sectional area (mm$^2$)", True),
    ("wss_mean_pa", "Wall shear stress (Pa)", True),
    ("velocity_axial_mean_ms", "Axial velocity (m s$^{-1}$)", True),
]

# Panels for the radius figure. Radius and CSA are dropped here: against a radius
# axis they plot a variable against itself and only redraw the binning. The
# vessel count takes their place, since how many vessels of each calibre each
# tree contributes is what decides how the pooled curve is weighted.
PANELS_RADIUS: list[tuple[str, str, bool]] = [
    ("__count__", "Vessels in bin", False),
    ("wss_mean_pa", "Wall shear stress (Pa)", True),
    ("velocity_axial_mean_ms", "Axial velocity (m s$^{-1}$)", True),
    ("flow_axial_ml_min", "Axial flow estimate (mL min$^{-1}$)", True),
]

# The haemodynamics against calibre, left and right as separate curves and no
# pooled series. Nothing is mixed across the two solves here, so this is the
# figure to read for "how do the trees differ" -- with the caveat that the flow
# split is prescribed, so the gap is anatomy plus the flow each tree was given.
# Metrics whose smallest radius bin is dropped. Mean WSS is not mesh-converged
# there: 8% / 0% / 26% of vessels meet the 5% criterion in the three smallest
# bins against 93-100% above 0.31 mm, and the first bin is the worst of them.
# The point it drew was a discretisation artefact sitting at the eye-catching end
# of the curve, so it is omitted rather than plotted and caveated.
DROP_SMALLEST_BIN: frozenset[str] = frozenset({"wss_mean_pa"})


def _panel_rows(rows: list[dict[str, Any]], xs: list[float], metric: str):
    """``(xs, rows)`` for one panel, with the smallest bin dropped where it must be."""
    if metric in DROP_SMALLEST_BIN and len(rows) > 1:
        return xs[1:], rows[1:]
    return xs, rows


CFD_PANELS: list[tuple[str, str, bool]] = [
    ("wss_mean_pa", "Wall shear stress (Pa)", True),
    ("velocity_mean_ms", "Velocity (m s$^{-1}$)", True),
    ("velocity_axial_mean_ms", "Axial velocity (m s$^{-1}$)", True),
    ("flow_axial_ml_min", "Axial flow estimate (mL min$^{-1}$)", True),
]

# Every CFD metric, for the single-panel set. `logy` is False wherever the
# quantity can be negative (pressure, referenced to each solve's own outlets) or
# is a bounded ratio near 1, both of which a log axis renders unreadable.
CFD_SINGLES: list[tuple[str, str, bool]] = [
    ("wss_mean_pa", "Wall shear stress (Pa)", True),
    ("velocity_mean_ms", "Velocity (m s$^{-1}$)", True),
    ("velocity_axial_mean_ms", "Axial velocity (m s$^{-1}$)", True),
    ("flow_axial_ml_min", "Axial flow estimate (mL min$^{-1}$)", True),
    ("flow_speed_ml_min", "Mean-speed flow estimate (mL min$^{-1}$)", True),
    ("axial_fraction", r"$\langle|v\cdot t|\rangle / \langle|v|\rangle$", False),
    ("flow_coherence", "Flow coherence", False),
    ("pressure_wall_mean_mmhg", "Static pressure, wall (mmHg)", False),
    ("pressure_lumen_mean_mmhg", "Static pressure, lumen (mmHg)", False),
]

# Pressure is referenced to each solve's own outlets, so the two curves sit on
# different zeros. Drawn, because the within-tree gradient is real and worth
# seeing; annotated, so the vertical offset between them is never read as a
# pressure difference.
SEPARATELY_REFERENCED = {"pressure_wall_mean_mmhg", "pressure_lumen_mean_mmhg"}


def read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="") as fh:
        return list(csv.DictReader(fh))


def _f(v: Any) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return math.nan


def _series(rows: list[dict[str, Any]], metric: str, name: str
            ) -> tuple[list[float], list[float], list[float], list[float]]:
    """median, lower error, upper error, n -- clipped so no bar crosses zero.

    The IQR is asymmetric about the median for a skewed distribution, so the two
    error arms are carried separately rather than as one +/- value."""
    med, lo, hi, ns = [], [], [], []
    for r in rows:
        m = _f(r.get(f"{metric}_median_{name}"))
        q1 = _f(r.get(f"{metric}_q25_{name}"))
        q3 = _f(r.get(f"{metric}_q75_{name}"))
        n = _f(r.get(f"{metric}_n_{name}"))
        med.append(m)
        # Fall back to no whisker rather than a wrong one when a quartile is
        # missing (a group of n=1 has no spread to draw).
        lo.append(max(m - q1, 0.0) if math.isfinite(q1) and math.isfinite(m) else 0.0)
        hi.append(max(q3 - m, 0.0) if math.isfinite(q3) and math.isfinite(m) else 0.0)
        ns.append(n)
    return med, lo, hi, ns


def _finite(vals: list[float]) -> bool:
    return any(math.isfinite(v) for v in vals)


def _grouped_points(ax, xs, rows, metric, ylabel, logy):
    """Median dot with an IQR bar, three series offset within each order.

    Dots rather than bars because three of these four metrics span more than a
    decade and are drawn on a log axis: a bar encodes its value by length from
    zero, which a log axis has no room for, so a bar there reads as whatever the
    axis happened to be clipped at. The dot carries the median and the whisker
    the spread, both of which survive the rescaling."""
    n_s = len(SERIES)
    step = 0.72 / n_s
    for k, (name, label, colour) in enumerate(SERIES):
        med, lo, hi, ns = _series(rows, metric, name)
        if not _finite(med):
            continue
        off = (k - (n_s - 1) / 2) * step
        pts = [(i + off, m, l, h) for i, (m, l, h) in enumerate(zip(med, lo, hi))
               if math.isfinite(m)]
        if not pts:
            continue
        px, pm, pl, ph = zip(*pts)
        ax.errorbar(px, pm, yerr=[pl, ph], color=colour, label=label,
                    linestyle="none", marker="o", markersize=5,
                    markeredgecolor="white", markeredgewidth=0.8,
                    elinewidth=1.4, capsize=2.5)
    ax.set_ylabel(ylabel, fontsize=8)
    if logy:
        ax.set_yscale("log")
    ax.set_xticks(list(range(len(xs))))
    ax.set_xticklabels(xs, fontsize=8)
    ax.set_xlim(-0.6, len(xs) - 0.4)


def _log_x_ticks(ax, xs) -> None:
    """Label a fixed readable subset of a log radius axis.

    Matplotlib labels every minor decade step by default, which collides at these
    figure widths -- the same fix `strahler_plots.line_panel` applies, kept
    consistent so a panel from either module carries the same tick set."""
    finite = [x for x in xs if math.isfinite(x) and x > 0]
    if not finite:
        return
    lo, hi = min(finite), max(finite)
    ticks = [t for t in (0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0)
             if lo * 0.9 <= t <= hi * 1.1]
    if len(ticks) >= 2:
        ax.set_xticks(ticks)
        ax.set_xticklabels([f"{t:g}" for t in ticks])
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.xaxis.set_minor_formatter(NullFormatter())


def _lines(ax, xs, rows, metric, ylabel, logy, series=SERIES):
    """One line per series over the shared radius bins, markers on bin centres.

    ``__count__`` plots the vessel count instead of a metric summary; it has no
    spread to draw, so its whiskers are zero."""
    for name, label, colour in series:
        if metric == "__count__":
            med = [_f(r.get(f"n_vessels_{name}")) for r in rows]
            lo = hi = [0.0] * len(med)
        else:
            med, lo, hi, _ns = _series(rows, metric, name)
        pts = [(x, m, l, h) for x, m, l, h in zip(xs, med, lo, hi)
               if math.isfinite(m) and math.isfinite(x)]
        if not pts:
            continue
        px, pm, pl, ph = zip(*pts)
        ax.errorbar(px, pm, yerr=[pl, ph], color=colour, label=label,
                    marker="o", markersize=4.5, linewidth=1.6,
                    markeredgecolor="white", markeredgewidth=0.8,
                    elinewidth=0.9, capsize=2)
    ax.set_ylabel(ylabel, fontsize=8)
    ax.set_xscale("log")
    _log_x_ticks(ax, xs)
    if logy:
        ax.set_yscale("log")


def fig_by_strahler(rows: list[dict[str, Any]], out_dir: Path) -> None:
    apply_style()
    xs = [r["strahler"] for r in rows]
    fig, axes = plt.subplots(2, 2, figsize=(COL2, COL2 * 0.62))
    for ax, (metric, ylabel, logy) in zip(axes.ravel(), PANELS):
        _grouped_points(ax, xs, rows, metric, ylabel, logy)
        ax.set_xlabel("Strahler order", fontsize=8)
    axes.ravel()[0].legend(frameon=False, fontsize=7, loc="upper left")
    fig.suptitle("Left, right and combined by Strahler order "
                 "(median, IQR; root segments excluded)",
                 fontsize=9, color=INK, y=1.00)
    fig.tight_layout()
    save(fig, out_dir, "fig_compare_by_strahler")


def fig_by_radius(rows: list[dict[str, Any]], out_dir: Path) -> None:
    apply_style()
    xs = [_f(r.get("radius_mid_mm_combined")) for r in rows]
    fig, axes = plt.subplots(2, 2, figsize=(COL2, COL2 * 0.62))
    for ax, (metric, ylabel, logy) in zip(axes.ravel(), PANELS_RADIUS):
        _lines(ax, xs, rows, metric, ylabel, logy)
        ax.set_xlabel("Vessel radius (mm)", fontsize=8)
    # Counts fall left to right, so the upper right of that panel is the empty
    # corner; upper left would sit on the pooled curve.
    axes.ravel()[0].legend(frameon=False, fontsize=7, loc="upper right")
    fig.suptitle("Left, right and combined by vessel radius "
                 "(shared bins; median, IQR; root segments excluded)",
                 fontsize=9, color=INK, y=1.00)
    fig.tight_layout()
    save(fig, out_dir, "fig_compare_by_radius")


def fig_cfd_by_radius(rows: list[dict[str, Any]], out_dir: Path) -> None:
    """Haemodynamics against vessel calibre, left and right on the same axes.

    No pooled series: nothing is mixed across the two solves, so every curve is
    one tree's own result. Radius is the x-axis rather than Strahler order
    because a radius bin is the same physical calibre in both trees, whereas an
    order is a position within whichever tree it came from."""
    apply_style()
    xs = [_f(r.get("radius_mid_mm_combined")) for r in rows]
    fig, axes = plt.subplots(2, 2, figsize=(COL2, COL2 * 0.62))
    for ax, (metric, ylabel, logy) in zip(axes.ravel(), CFD_PANELS):
        _lines(ax, *_panel_rows(rows, xs, metric), metric, ylabel, logy, SERIES_LR)
        ax.set_xlabel("Vessel radius (mm)", fontsize=8)
    axes.ravel()[0].legend(frameon=False, fontsize=7, loc="upper right")
    fig.suptitle("Haemodynamics by vessel radius, left vs right (shared bins; "
                 "median, IQR; root segments excluded; smallest bin omitted for "
                 "WSS, not mesh-converged)",
                 fontsize=9, color=INK, y=1.00)
    fig.tight_layout()
    save(fig, out_dir, "fig_cfd_by_radius")


def fig_cfd_singles(rows: list[dict[str, Any]], out_dir: Path) -> None:
    """One single-column panel per CFD metric, left and right overlaid."""
    single_dir = out_dir / "single_panels"
    xs = [_f(r.get("radius_mid_mm_combined")) for r in rows]
    for metric, ylabel, logy in CFD_SINGLES:
        apply_style()
        fig, ax = plt.subplots(figsize=(COL1, 2.7))
        _lines(ax, *_panel_rows(rows, xs, metric), metric, ylabel, logy, SERIES_LR)
        if not ax.lines:  # nothing finite for either tree
            plt.close(fig)
            continue
        ax.set_xlabel("Vessel radius (mm)", fontsize=8)
        ax.legend(frameon=False, fontsize=7)
        if metric in SEPARATELY_REFERENCED:
            # A caveat, not a heading: the house title style is bold, which would
            # give this more weight than the axis label it qualifies.
            ax.set_title("separate solves, each referenced to its own outlets",
                         fontsize=7, color=INK_SOFT, fontweight="normal",
                         loc="left")
        fig.tight_layout()
        save(fig, single_dir, f"fig_radius_{metric}")


def fig_pooling_shift(rows: list[dict[str, Any]], out_dir: Path) -> None:
    """How far the pooled median sits from the n-weighted mean of the parts.

    A pooled median is not the weighted mean of the two per-tree medians, so this
    is not an error: it is the size of the effect that pooling itself introduces,
    which is the whole question. Zero means pooling told you nothing new."""
    apply_style()
    xs = [r["strahler"] for r in rows]
    fig, ax = plt.subplots(figsize=(COL2, COL2 * 0.34))
    metrics = [(m, lbl) for m, lbl, _ in PANELS]
    width = 0.80 / len(metrics)
    # Sequential steps of one hue would say "magnitude"; these four are separate
    # identities, so they take the categorical order, extended by the palette's
    # 4th slot.
    colours = [LEFT_C, RIGHT_C, COMB_C, "#8a63d2"]
    for k, ((metric, label), colour) in enumerate(zip(metrics, colours)):
        vals = [_f(r.get(f"{metric}_pool_delta_pct")) for r in rows]
        if not _finite(vals):
            continue
        off = (k - (len(metrics) - 1) / 2) * width
        ax.bar([i + off for i in range(len(xs))],
               [0.0 if not math.isfinite(v) else v for v in vals],
               width * 0.92, color=colour, label=label,
               linewidth=0.6, edgecolor="white")
    ax.axhline(0, color=INK_SOFT, linewidth=0.8)
    ax.set_ylabel("Pooled median vs n-weighted\nmean of the parts (%)", fontsize=8)
    ax.set_xlabel("Strahler order", fontsize=8)
    ax.set_xticks(range(len(xs)))
    ax.set_xticklabels(xs, fontsize=8)
    ax.legend(frameon=False, fontsize=7, ncol=2)
    fig.tight_layout()
    save(fig, out_dir, "fig_pooling_shift")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in", dest="in_dir", type=Path, required=True,
                    help="the strahler_combined output directory")
    ap.add_argument("--out", type=Path, default=None,
                    help="figure directory (default <in>/comparison/figures)")
    args = ap.parse_args(argv)

    cmp_dir = args.in_dir / "comparison"
    out_dir = args.out or (cmp_dir / "figures")
    out_dir.mkdir(parents=True, exist_ok=True)

    by_strahler = read_csv(cmp_dir / "compare_by_strahler.csv")
    by_radius = read_csv(cmp_dir / "compare_by_radius.csv")

    fig_by_strahler(by_strahler, out_dir)
    fig_by_radius(by_radius, out_dir)
    fig_pooling_shift(by_strahler, out_dir)
    fig_cfd_by_radius(by_radius, out_dir)
    fig_cfd_singles(by_radius, out_dir)
    print(f"[compare] figures in {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
