"""Publication figures for the Strahler / radius analysis.

Reads only the CSVs written by :mod:`coronary_sdf.strahler_analysis`, so a figure
can always be traced back to a table and re-plotted without recomputing anything.

Figures (each written as PNG at 600 dpi, plus vector PDF and SVG):

``fig1_morphometry``      diameter, cross-sectional area, segment count, bifurcations
``fig2_haemodynamics``    wall shear, pressure, velocity, flow rate by order
``fig3_by_radius``        wall shear, pressure, velocity, flow rate by vessel radius
``fig_<metric>``          one single-column panel per metric, for slides

Bars show the **median** with the **interquartile range** across vessel segments
in the order or bin.  The haemodynamic distributions are strongly right-skewed —
their standard deviation exceeds the mean, so a symmetric mean +/- SD bar reaches
below zero, which is impossible for a magnitude — and median/IQR is what the
cardiovascular-mechanics literature reports for them.  ``--summary mean-sd``
restores mean +/- SD (the morphometry convention) and ``mean-sem`` the standard
error.

The CFD behind these figures is a **steady-state** solve, so wall shear is the
steady WSS at the converged operating point, not a cycle-averaged TAWSS; no
oscillatory metric (OSI, RRT) is defined for it.

Usage::

    python -m coronary_sdf.strahler_plots --in analysis_out/ratio_6
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

# Validated categorical slots 1-3 (see the data-viz reference palette); these
# three clear the all-pairs colour-vision floors in both light and dark modes.
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK_SOFT, GRID = "#0b0b0b", "#52514e", "#d8d7d2"

# Journal column widths (inches).
COL1, COL2 = 3.5, 7.2


def apply_style() -> None:
    """Recessive axes, thin marks, no top/right spines — a print-first look."""
    plt.rcParams.update({
        "figure.dpi": 150,
        "savefig.dpi": 600,
        "savefig.bbox": "tight",
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 8,
        "axes.labelsize": 8,
        "axes.titlesize": 9,
        "axes.titleweight": "bold",
        "axes.labelcolor": INK,
        "axes.edgecolor": INK_SOFT,
        "axes.linewidth": 0.6,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": GRID,
        "grid.linewidth": 0.5,
        "grid.alpha": 0.9,
        "xtick.color": INK_SOFT,
        "ytick.color": INK_SOFT,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "legend.frameon": False,
        "legend.fontsize": 7.5,
        "lines.linewidth": 1.4,
        "lines.markersize": 4.5,
    })


def read_csv(path: Path) -> list[dict[str, Any]]:
    """CSV -> list of dicts with numeric strings coerced to float (blank -> NaN)."""
    if not path.exists():
        return []
    out: list[dict[str, Any]] = []
    with path.open(newline="") as fh:
        for row in csv.DictReader(fh):
            rec: dict[str, Any] = {}
            for k, v in row.items():
                if v is None or v == "":
                    rec[k] = math.nan
                    continue
                try:
                    rec[k] = float(v)
                except ValueError:
                    rec[k] = v
            out.append(rec)
    return out


def _col(rows: list[dict[str, Any]], key: str) -> list[float]:
    return [float(r.get(key, math.nan)) for r in rows]


def summarise(
    rows: list[dict[str, Any]], key: str, mode: str = "median-iqr"
) -> tuple[list[float], list[float], list[float]]:
    """``(centre, err_below, err_above)`` for one metric, in the chosen summary.

    ``median-iqr`` is the default because the haemodynamic quantities here are
    strongly right-skewed: their standard deviation exceeds the mean, so a
    symmetric mean +/- SD bar extends below zero, which is impossible for a
    magnitude. Median with the interquartile range is what the cardiovascular
    literature reports for such distributions, and its asymmetry carries the skew
    rather than hiding it. ``mean-sd`` remains available for the morphometry,
    where it is the convention."""
    if mode == "median-iqr":
        centre = _col(rows, f"{key}_median")
        q25, q75 = _col(rows, f"{key}_q25"), _col(rows, f"{key}_q75")
        lo = [c - a if math.isfinite(c) and math.isfinite(a) else 0.0
              for c, a in zip(centre, q25)]
        hi = [b - c if math.isfinite(c) and math.isfinite(b) else 0.0
              for c, b in zip(centre, q75)]
        return centre, lo, hi
    stat = "sem" if mode == "mean-sem" else "sd"
    centre = _col(rows, f"{key}_mean")
    err = _col(rows, f"{key}_{stat}")
    safe = [e if math.isfinite(e) else 0.0 for e in err]
    return centre, safe, list(safe)


def _err_label(mode: str) -> str:
    return {"median-iqr": "median, bars = IQR",
            "mean-sd": "mean $\\pm$ SD",
            "mean-sem": "mean $\\pm$ SEM"}.get(mode, mode)


def _split_err(errs) -> tuple[list[float], list[float]]:
    """Normalise an error spec to ``(below, above)`` lists of finite magnitudes."""
    if (isinstance(errs, tuple) and len(errs) == 2
            and isinstance(errs[0], (list, tuple))):
        lo, hi = errs
    else:
        lo = hi = errs
    clean = lambda a: [abs(e) if isinstance(e, float) and math.isfinite(e) else 0.0
                       for e in a]
    return clean(lo), clean(hi)


def _has_data(values: list[float]) -> bool:
    return any(math.isfinite(v) for v in values)


def _annotate_n(ax, xs, ys, ns, *, dy: float = 0.02) -> None:
    """Write the group size above each mark.

    n varies a lot between Strahler orders (136 vessels at order 1, one at the
    root), and an error bar drawn from one sample means something quite different
    from one drawn from 136 — so the count is part of the figure, not a caption."""
    span = ax.get_ylim()[1] - ax.get_ylim()[0]
    for x, y, n in zip(xs, ys, ns):
        if not math.isfinite(y) or not math.isfinite(n):
            continue
        ax.annotate(f"n={int(n)}", (x, y + dy * span), ha="center", va="bottom",
                    fontsize=6, color=INK_SOFT, clip_on=False)


def bar_panel(ax, xs, means, errs, ns, *, color=BLUE, ylabel="", title="",
              xlabel="Strahler order", annotate=True, log=False) -> bool:
    """Bar + error-bar panel; ``errs`` is either symmetric or an ``(lo, hi)`` pair.

    Returns False (and blanks the axes) when empty."""
    if not _has_data(means):
        ax.text(0.5, 0.5, "no data", transform=ax.transAxes, ha="center",
                va="center", color=INK_SOFT, fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(title, color=INK, loc="left")
        return False
    lo, hi = _split_err(errs)
    ax.bar(xs, means, width=0.62, color=color, edgecolor="white", linewidth=0.8, zorder=2)
    # Counts and totals carry no spread; drawing yerr=0 would stamp a cap on every
    # bar and imply an error bar that collapsed to zero rather than one that does
    # not apply.
    if any(e > 0 for e in lo + hi):
        ax.errorbar(xs, means, yerr=[lo, hi], fmt="none", ecolor=INK_SOFT,
                    elinewidth=0.8, capsize=2.5, capthick=0.8, zorder=3)
    if log:
        ax.set_yscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title, color=INK, loc="left")
    ax.set_xticks(list(xs))
    ax.grid(axis="x", visible=False)
    if annotate and ns is not None:
        top = [m + e for m, e in zip(means, hi)]
        ax.set_ylim(top=max(v for v in top if math.isfinite(v)) * 1.18)
        _annotate_n(ax, xs, top, ns)
    return True


def line_panel(ax, xs, means, errs, *, color=BLUE, ylabel="", title="",
               xlabel="Vessel radius (mm)", logx=True) -> bool:
    """Marker + error-bar panel against a continuous x (radius bins)."""
    if not _has_data(means):
        ax.text(0.5, 0.5, "no data", transform=ax.transAxes, ha="center",
                va="center", color=INK_SOFT, fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(title, color=INK, loc="left")
        return False
    lo, hi = _split_err(errs)
    ax.errorbar(xs, means, yerr=[lo, hi], fmt="o-", color=color,
                ecolor=INK_SOFT, elinewidth=0.8, capsize=2.5, capthick=0.8,
                markeredgecolor="white", markeredgewidth=0.7, zorder=3)
    if logx:
        ax.set_xscale("log")
        # A log axis defaults to labelling every minor decade step, which collides
        # at this figure width. Label a fixed, readable subset spanning the data.
        finite_x = [x for x in xs if math.isfinite(x) and x > 0]
        if finite_x:
            lo, hi = min(finite_x), max(finite_x)
            ticks = [t for t in (0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0)
                     if lo * 0.9 <= t <= hi * 1.1]
            if len(ticks) >= 2:
                ax.set_xticks(ticks)
                ax.set_xticklabels([f"{t:g}" for t in ticks])
        ax.xaxis.set_major_formatter(ScalarFormatter())
        ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title, color=INK, loc="left")
    return True


def save(fig, out_dir: Path, name: str) -> None:
    """Write raster + both vector formats; journals ask for one or the other."""
    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf", "svg"):
        fig.savefig(out_dir / f"{name}.{ext}")
    plt.close(fig)
    print(f"[plots] {name}.png/.pdf/.svg")


# ── figures ───────────────────────────────────────────────────────────────────


def fig_morphometry(anat: list[dict], out_dir: Path, mode: str = "median-iqr") -> None:
    """Diameter, cross-sectional area, segment count and bifurcations per order."""
    xs = _col(anat, "strahler")
    ns = _col(anat, "n_vessels")
    fig, axes = plt.subplots(2, 2, figsize=(COL2, 5.0))

    c, lo, hi = summarise(anat, "diameter_mean_mm", mode)
    bar_panel(axes[0][0], xs, c, (lo, hi), ns,
              ylabel="Vessel diameter (mm)",
              title=f"a  Diameter ({_err_label(mode)})")
    # Summed CSA is a total, not an average, so it carries no error bar.
    bar_panel(axes[0][1], xs, _col(anat, "csa_mm2_sum"),
              [0.0] * len(xs), ns, color=AQUA,
              ylabel="Total CSA (mm$^2$)", title="b  Total cross-sectional area")
    bar_panel(axes[1][0], xs, ns, [0.0] * len(xs), None, color=ORANGE,
              ylabel="Vessel segment count", title="c  Vessel segment count",
              annotate=False, log=True)
    bar_panel(axes[1][1], xs, _col(anat, "n_bifurcations"), [0.0] * len(xs), None,
              color=ORANGE, ylabel="Bifurcations (n)",
              title="d  Bifurcations", annotate=False)
    fig.tight_layout()
    save(fig, out_dir, "fig1_morphometry")


def fig_haemodynamics(cfd: list[dict], out_dir: Path, mode: str = "median-iqr") -> None:
    """Wall shear, pressure, velocity and flow rate per Strahler order."""
    xs = _col(cfd, "strahler")
    fig, axes = plt.subplots(2, 2, figsize=(COL2, 5.0))
    panels = [
        (axes[0][0], "wss_mean_pa", "Wall shear stress (Pa)", "a  Wall shear stress", BLUE),
        (axes[0][1], "pressure_wall_mean_mmhg", "Static pressure (mmHg)",
         "b  Static pressure (wall)", BLUE),
        (axes[1][0], "velocity_mean_ms", "Velocity (m s$^{-1}$)", "c  Velocity", BLUE),
        (axes[1][1], "flow_speed_ml_min", "Mean-speed flow estimate (mL min$^{-1}$)", "d  Flow estimate", BLUE),
    ]
    for ax, key, ylabel, title, color in panels:
        # n here is the number of segments carrying CFD data, which is smaller
        # than the vessel count when part of an order fell outside the mesh.
        ns = _col(cfd, f"{key}_n")
        c, lo, hi = summarise(cfd, key, mode)
        bar_panel(ax, xs, c, (lo, hi), ns, color=color, ylabel=ylabel, title=title)
    fig.tight_layout()
    save(fig, out_dir, "fig2_haemodynamics")


def fig_by_radius(bins: list[dict], out_dir: Path, mode: str = "median-iqr") -> None:
    """The same haemodynamic metrics against vessel radius rather than order."""
    xs = _col(bins, "radius_mid_mm")
    fig, axes = plt.subplots(2, 2, figsize=(COL2, 5.0))
    panels = [
        (axes[0][0], "wss_mean_pa", "Wall shear stress (Pa)", "a  Wall shear stress"),
        (axes[0][1], "pressure_wall_mean_mmhg", "Static pressure (mmHg)",
         "b  Static pressure (wall)"),
        (axes[1][0], "velocity_mean_ms", "Velocity (m s$^{-1}$)", "c  Velocity"),
        (axes[1][1], "flow_speed_ml_min", "Mean-speed flow estimate (mL min$^{-1}$)", "d  Flow estimate"),
    ]
    for ax, key, ylabel, title in panels:
        c, lo, hi = summarise(bins, key, mode)
        line_panel(ax, xs, c, (lo, hi), ylabel=ylabel, title=title)
    fig.tight_layout()
    save(fig, out_dir, "fig3_by_radius")


def fig_distributions(segments: list[dict], out_dir: Path) -> None:
    """Per-order box plots of the raw per-segment values.

    The haemodynamic distributions are strongly right-skewed — a few proximal or
    junction-adjacent segments carry values many times the median — so the
    standard deviation exceeds the mean and a mean+/-SD bar reaches below zero,
    which is impossible for a magnitude.  These box plots show the median, IQR and
    the actual outliers, and are the honest companion to the bar figures."""
    if not segments:
        return
    specs = [
        ("wss_mean_pa", "Wall shear stress (Pa)", "a  Wall shear stress", True),
        ("pressure_wall_mean_mmhg", "Static pressure (mmHg)",
         "b  Static pressure (wall)", False),
        ("velocity_mean_ms", "Velocity (m s$^{-1}$)", "c  Velocity", True),
        ("flow_speed_ml_min", "Mean-speed flow estimate (mL min$^{-1}$)", "d  Flow estimate", True),
    ]
    orders = sorted({int(r["strahler"]) for r in segments
                     if isinstance(r.get("strahler"), float)})
    fig, axes = plt.subplots(2, 2, figsize=(COL2, 5.0))
    for ax, (key, ylabel, title, logy) in zip(axes.ravel(), specs):
        data, labels, positions = [], [], []
        sparse: list[tuple[int, list[float]]] = []
        for pos, o in enumerate(orders, start=1):
            vals = [float(r[key]) for r in segments
                    if int(r["strahler"]) == o
                    and isinstance(r.get(key), float) and math.isfinite(r[key])]
            if logy:
                vals = [v for v in vals if v > 0]
            labels.append(str(o))
            if not vals:
                continue
            # A box needs a distribution. With only a handful of vessels (the root
            # order is a single segment) the quartiles collapse and the patch
            # renders as an invisible zero-height box, so show the values instead.
            if len(vals) < 5:
                sparse.append((pos, vals))
            else:
                data.append(vals)
                positions.append(pos)
        if not data and not sparse:
            ax.text(0.5, 0.5, "no data", transform=ax.transAxes, ha="center",
                    va="center", color=INK_SOFT, fontsize=8)
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_title(title, color=INK, loc="left")
            continue
        if data:
            ax.boxplot(data, positions=positions, widths=0.55, patch_artist=True,
                       showfliers=True,
                       flierprops=dict(marker="o", markersize=2.0,
                                       markerfacecolor=INK_SOFT,
                                       markeredgecolor="none", alpha=0.45),
                       medianprops=dict(color="white", linewidth=1.2),
                       whiskerprops=dict(color=INK_SOFT, linewidth=0.8),
                       capprops=dict(color=INK_SOFT, linewidth=0.8),
                       boxprops=dict(facecolor=BLUE, edgecolor="none"))
        for pos, vals in sparse:
            ax.plot([pos] * len(vals), vals, marker="D", linestyle="none",
                    markersize=4.0, color=BLUE, markeredgecolor="white",
                    markeredgewidth=0.7, zorder=4)
        if logy:
            ax.set_yscale("log")
        ax.set_xticks(range(1, len(labels) + 1))
        ax.set_xticklabels(labels)
        ax.set_xlim(0.4, len(labels) + 0.6)
        ax.set_xlabel("Strahler order")
        ax.set_ylabel(ylabel)
        ax.set_title(title, color=INK, loc="left")
        ax.grid(axis="x", visible=False)
    fig.tight_layout()
    save(fig, out_dir, "fig4_distributions")


#: Compact-table columns rendered as a figure, with the header each gets. Kept
#: separate from `strahler_analysis.TABLE_COLUMNS` so the CSV can carry a column
#: this does not have room for without the figure silently widening.
TABLE_HEADERS: list[tuple[str, str]] = [
    ("strahler", "Order"),
    ("n_vessels", "Vessels"),
    ("n_terminal_branches", "Terminal"),
    ("n_bifurcations", "Bifurcations"),
    ("diameter_median_mm", "Diameter\nmedian (mm)"),
    ("diameter_q25_mm", "Diameter\nQ25 (mm)"),
    ("diameter_q75_mm", "Diameter\nQ75 (mm)"),
    ("csa_total_mm2", "Total CSA\n(mm$^2$)"),
    ("length_total_mm", "Total length\n(mm)"),
    ("volume_total_mm3", "Total volume\n(mm$^3$)"),
]


def fig_table(rows: list[dict], out_dir: Path) -> None:
    """The compact morphometry table as a figure, for a slide.

    Same numbers as ``morphometry_table.csv`` -- rendered rather than recomputed,
    so the figure cannot drift from the file it came from."""
    if not rows:
        return
    counts = {"strahler", "n_vessels", "n_terminal_branches", "n_bifurcations"}

    def cell(key: str, row: dict) -> str:
        raw = row.get(key, "")
        try:
            value = float(raw)
        except (TypeError, ValueError):
            return str(raw)          # the totals row's "total" label
        if not math.isfinite(value):
            return ""
        # `read_csv` floats everything, so counts arrive as 82.0 and would print
        # that way unless they are put back on the integer they never left.
        if key in counts:
            return str(int(round(value)))
        return f"{value:.3f}" if abs(value) < 10 else f"{value:.1f}"

    text = [[cell(k, r) for k, _ in TABLE_HEADERS] for r in rows]
    fig, ax = plt.subplots(figsize=(COL2, 0.36 * len(text) + 1.0))
    ax.axis("off")
    table = ax.table(cellText=text,
                     colLabels=[h for _, h in TABLE_HEADERS],
                     loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(7)
    table.scale(1, 1.5)
    for (r, _c), cell_obj in table.get_celld().items():
        cell_obj.set_edgecolor("#D0D7DE")
        cell_obj.set_linewidth(0.6)
        if r == 0:
            cell_obj.set_text_props(fontweight="bold", color=INK)
        # The last data row is the totals row and reads as a different kind of
        # number; give it the weight to say so.
        elif r == len(text):
            cell_obj.set_text_props(fontweight="bold")
            cell_obj.set_facecolor("#F6F8FA")
    fig.tight_layout()
    save(fig, out_dir, "fig5_morphometry_table")


def fig_singles(anat: list[dict], cfd: list[dict], bins: list[dict],
                out_dir: Path, mode: str = "median-iqr") -> None:
    """One single-column panel per metric — the slide-friendly versions."""
    single_dir = out_dir / "single_panels"
    strahler_specs = [
        (anat, "diameter_mean_mm", "Vessel diameter (mm)",
         "Vessel diameter by Strahler order", BLUE),
        # "CSA per vessel", never "mean CSA": the summed CSA of an order is a
        # different quantity that falls where this one rises, and a panel titled
        # only "CSA" beside `fig1b` reads as a contradiction rather than a
        # distinction.
        (anat, "csa_mm2", "Cross-sectional area per vessel (mm$^2$)",
         "Cross-sectional area per vessel, by Strahler order", AQUA),
        (cfd, "wss_mean_pa", "Wall shear stress (Pa)", "Wall shear stress by Strahler order", BLUE),
        (cfd, "pressure_wall_mean_mmhg", "Static pressure (mmHg)",
         "Static pressure (wall) by Strahler order", BLUE),
        (cfd, "pressure_lumen_mean_mmhg", "Static pressure (mmHg)",
         "Static pressure (lumen) by Strahler order", BLUE),
        (cfd, "velocity_mean_ms", "Velocity (m s$^{-1}$)", "Velocity by Strahler order", BLUE),
        (cfd, "flow_speed_ml_min", "Mean-speed flow estimate (mL min$^{-1}$)", "Mean-speed flow estimate by Strahler order", BLUE),
        # The axial pair. `flow_speed_ml_min` above counts swirl and reversal as
        # forward transport; these project onto the local centreline tangent, so
        # they are the ones to put in front of an audience as flow.
        (cfd, "flow_axial_ml_min", "Axial flow estimate (mL min$^{-1}$)",
         "Axial flow estimate by Strahler order", AQUA),
        (cfd, "velocity_axial_mean_ms", "Axial velocity (m s$^{-1}$)",
         "Axial velocity by Strahler order", AQUA),
        (cfd, "axial_fraction", r"$\langle|v\cdot t|\rangle / \langle|v|\rangle$",
         "Axial fraction by Strahler order", AQUA),
        (cfd, "flow_coherence", "Flow coherence",
         "Flow coherence by Strahler order", AQUA),
    ]
    for rows, key, ylabel, title, color in strahler_specs:
        if not rows:
            continue
        fig, ax = plt.subplots(figsize=(COL1, 2.6))
        c, lo, hi = summarise(rows, key, mode)
        ok = bar_panel(ax, _col(rows, "strahler"), c, (lo, hi),
                       _col(rows, f"{key}_n"),
                       color=color, ylabel=ylabel, title=title)
        fig.tight_layout()
        if ok:
            save(fig, single_dir, f"fig_strahler_{key}")
        else:
            plt.close(fig)

    # Total CSA per order -- the summed quantity from `fig1b`, as its own slide.
    # It is a total, so it carries no error bar, and it is the panel that carries
    # the distal-expansion result.
    if anat:
        fig, ax = plt.subplots(figsize=(COL1, 2.6))
        xs = _col(anat, "strahler")
        ok = bar_panel(ax, xs, _col(anat, "csa_mm2_sum"), [0.0] * len(xs),
                       _col(anat, "n_vessels"), color=AQUA,
                       ylabel="Total CSA (mm$^2$)",
                       title="Total cross-sectional area by Strahler order")
        fig.tight_layout()
        if ok:
            save(fig, single_dir, "fig_strahler_csa_total")
        else:
            plt.close(fig)

    for key, ylabel, title in [
        ("wss_mean_pa", "Wall shear stress (Pa)", "Wall shear stress by radius"),
        ("pressure_wall_mean_mmhg", "Static pressure (mmHg)",
         "Static pressure by radius"),
        ("velocity_mean_ms", "Velocity (m s$^{-1}$)", "Velocity by radius"),
        ("flow_speed_ml_min", "Mean-speed flow estimate (mL min$^{-1}$)", "Mean-speed flow estimate by radius"),
    ]:
        if not bins:
            continue
        fig, ax = plt.subplots(figsize=(COL1, 2.6))
        c, lo, hi = summarise(bins, key, mode)
        ok = line_panel(ax, _col(bins, "radius_mid_mm"), c, (lo, hi),
                        ylabel=ylabel, title=title)
        fig.tight_layout()
        if ok:
            save(fig, single_dir, f"fig_radius_{key}")
        else:
            plt.close(fig)


def report_coverage(val_dir: Path) -> None:
    """Print the surface-coverage split as a build-time diagnostic.

    How much of the graph the surface actually spans is a caveat on the
    validation, not a result, so it is flagged here rather than plotted or
    tabulated: a reader of the figures should not have to interpret coverage, but
    whoever regenerates them should see it. Counts come from
    ``validation_summary.json``, which :mod:`coronary_sdf.surface_validation`
    writes alongside the score tables."""
    path = val_dir / "validation_summary.json"
    if not path.exists():
        return
    try:
        summary = json.loads(path.read_text())
    except (OSError, ValueError):
        return

    inside = summary.get("n_inside")
    missed = summary.get("n_missed")
    cropped = summary.get("n_cropped")
    if inside is None:
        return
    total = inside + missed + cropped
    print(f"[plots][coverage] {inside} inside, {missed} missed, {cropped} cropped "
          f"of {total} stations")
    if total and cropped / total > 0.2:
        print(f"[plots][WARN] {cropped / total * 100:.0f}% of centreline stations "
              "lie outside the reconstruction and are excluded from every score; "
              "the figures describe only the part of the graph the surface spans.")
    if inside + missed and missed / (inside + missed) > 0.1:
        # Plain ASCII: this goes to a Windows console whose code page mangles
        # non-ASCII punctuation.
        print(f"[plots][WARN] {missed / (inside + missed) * 100:.0f}% of in-domain "
              "stations are not enclosed by the surface - check for breaks or gaps.")
    for tree, cov in (summary.get("coverage_per_tree") or {}).items():
        frac = cov.get("cropped_fraction")
        if frac is not None and frac >= 0.999:
            print(f"[plots][coverage] {tree} is entirely outside the surface "
                  "and carries no scores.")


def fig_validation(val_dir: Path, out_dir: Path) -> None:
    """Surface-reconstruction validation against the graph.

    Panel a is the comparison the ask names directly — graph radius against
    reconstructed radius at each Strahler order.  Panel b is the Bland-Altman
    view of the same data, which is what shows whether the error is a constant
    offset or grows with vessel size."""
    agree = [r for r in read_csv(val_dir / "radius_agreement.csv")
             if r.get("strahler") != "all"]
    pts = read_csv(val_dir / "radius_points.csv")
    if not agree:
        return

    xs = [float(r["strahler"]) for r in agree]
    fig, axes = plt.subplots(1, 2, figsize=(COL2, 2.9))

    # a — paired bars. Two series, so a legend is mandatory.
    ax = axes[0]
    width = 0.36
    g_mean = _col(agree, "radius_graph_mean_mm")
    r_mean = _col(agree, "radius_recon_mean_mm")
    # The agreement table stores mean/SD only: radius is near-symmetric
    # within an order, so SD is the appropriate spread here.
    g_err = _col(agree, "radius_graph_sd_mm")
    r_err = _col(agree, "radius_recon_sd_mm")
    left = [x - width / 2 for x in xs]
    right = [x + width / 2 for x in xs]
    ax.bar(left, g_mean, width=width, color=BLUE, edgecolor="white",
           linewidth=0.8, label="Spatial graph", zorder=2)
    ax.bar(right, r_mean, width=width, color=ORANGE, edgecolor="white",
           linewidth=0.8, label="Reconstruction", zorder=2)
    ax.errorbar(left, g_mean, yerr=g_err, fmt="none", ecolor=INK_SOFT,
                elinewidth=0.8, capsize=2.0, capthick=0.8, zorder=3)
    ax.errorbar(right, r_mean, yerr=r_err, fmt="none", ecolor=INK_SOFT,
                elinewidth=0.8, capsize=2.0, capthick=0.8, zorder=3)
    ax.set_xticks(xs)
    ax.set_xlabel("Strahler order")
    ax.set_ylabel("Vessel radius (mm)")
    ax.set_title("a  Radius: graph vs reconstruction", color=INK, loc="left")
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper left")

    # b — Bland-Altman over the individual stations.
    ax = axes[1]
    if pts:
        g = [r["radius_graph_mm"] for r in pts]
        rc = [r["radius_recon_mm"] for r in pts]
        pairs = [(a, b) for a, b in zip(g, rc)
                 if isinstance(a, float) and isinstance(b, float)
                 and math.isfinite(a) and math.isfinite(b)]
        if pairs:
            avg = [(a + b) / 2 for a, b in pairs]
            diff = [b - a for a, b in pairs]
            n = len(diff)
            mean_d = sum(diff) / n
            sd_d = math.sqrt(sum((d - mean_d) ** 2 for d in diff) / max(n - 1, 1))
            ax.scatter(avg, diff, s=3, color=BLUE, alpha=0.25, edgecolors="none",
                       zorder=2)
            for y, ls, lab in ((mean_d, "-", f"bias {mean_d:+.3f} mm"),
                               (mean_d + 1.96 * sd_d, "--", "+1.96 SD"),
                               (mean_d - 1.96 * sd_d, "--", "-1.96 SD")):
                ax.axhline(y, color=ORANGE, linestyle=ls, linewidth=1.0, zorder=3)
                ax.annotate(lab, (ax.get_xlim()[1], y), xytext=(-2, 2),
                            textcoords="offset points", ha="right", va="bottom",
                            fontsize=6, color=INK_SOFT)
    ax.axhline(0, color=INK_SOFT, linewidth=0.6, zorder=1)
    ax.set_xlabel("Mean of graph and reconstruction (mm)")
    ax.set_ylabel("Reconstruction $-$ graph (mm)")
    ax.set_title("b  Bland-Altman", color=INK, loc="left")

    fig.tight_layout()
    save(fig, out_dir, "fig5_surface_validation")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--in", dest="in_dir", type=Path, required=True,
                    help="directory written by strahler_analysis")
    ap.add_argument("--out", type=Path, default=None,
                    help="figure directory (default <in>/figures)")
    ap.add_argument("--summary", choices=["median-iqr", "mean-sd", "mean-sem"],
                    default="median-iqr",
                    help="central value and error bars. Default median with the "
                         "interquartile range, the convention for the skewed "
                         "haemodynamic distributions; mean-sd for morphometry")
    ap.add_argument("--validation", type=Path, default=None,
                    help="directory written by surface_validation; adds fig5")
    args = ap.parse_args(argv)

    out_dir = args.out or args.in_dir / "figures"
    apply_style()

    anat = read_csv(args.in_dir / "by_strahler.csv")
    cfd = read_csv(args.in_dir / "by_strahler_cfd.csv")
    bins = read_csv(args.in_dir / "by_radius_bin.csv")

    if anat:
        fig_morphometry(anat, out_dir, args.summary)
    fig_table(read_csv(args.in_dir / "morphometry_table.csv"), out_dir)
    if cfd:
        fig_haemodynamics(cfd, out_dir, args.summary)
    if cfd and bins:
        fig_by_radius(bins, out_dir, args.summary)
    segments = read_csv(args.in_dir / "segments_cfd.csv")
    fig_distributions(segments, out_dir)
    fig_singles(anat, cfd, bins if cfd else [], out_dir, args.summary)
    if args.validation:
        report_coverage(args.validation)
        fig_validation(args.validation, out_dir)
    print(f"[plots] figures in {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
