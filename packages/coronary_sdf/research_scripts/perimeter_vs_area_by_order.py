"""Is the perimeter radius or the area radius larger, by Strahler order and calibre?

`shrinkage_scale.py` asked whether the perimeter/area gap could stand in for a
global shrinkage factor and answered no: the uplift is a function of section
*shape*, not a scale. It reported that by size decile. This script reports the
same comparison on the two axes the rest of the analysis uses -- **Strahler
order** and **radius bin** -- and writes the per-segment rows out rather than
discarding them, so the tables can be re-cut without re-measuring.

**Both estimators are reported raw and bias-corrected, because which is larger
depends on which pair you mean.**

* *Raw.* `r_perim_raw = P/2pi` and `r_area = sqrt(A/pi)` satisfy
  `r_perim / r_area = sqrt(Q) >= 1` exactly, by the isoperimetric inequality --
  so raw perimeter can never be the smaller of the two for a real shape. Where the
  measured value comes out below 1.0 anyway, that is the estimator's digitisation
  bias, not geometry, and it is the reason the correction exists.
* *Corrected.* `radius_perimeter.correct_perimeter_radius` applies
  `r/1.0548 + 0.5 * spacing`. That is **not a scale**: the half-voxel inset
  dominates on thin vessels and inflates them, while the 1.0548 divisor dominates
  on thick ones and deflates them by ~5%. So a large round vessel can come out
  with the *corrected* perimeter radius below its area radius, and whether it does
  is an empirical question per order and per calibre -- which is what the
  `frac_perim_larger` column answers.

Comparing the corrected pair is the right default: it is the number the pipeline
actually stores, and the raw pair differ partly by an artefact both estimators
would rather not have. The raw columns are kept beside it so the artefact stays
visible instead of being asserted away.

Sections come from `collapse_conservation.sections` -- `reformat` planes at
`mode="native"`, one pixel per segmentation voxel -- so both estimators see the
same digitisation, and a segment needs `MIN_PLANES` usable sections to appear.

Outputs, beside this script:

    perimeter_vs_area_segments.csv    one row per segment -- the audit trail
    perimeter_vs_area_by_strahler.csv
    perimeter_vs_area_by_radius.csv
    perimeter_vs_area.png/.pdf/.svg   the two tables as one figure

Run: python research_scripts/perimeter_vs_area_by_order.py
"""
from __future__ import annotations

import csv
import math
from pathlib import Path

import numpy as np

from collapse_conservation import GRAPH, SEG, open_lattice, sections
from hipct_seg_debug import reformat
from hipct_seg_debug.edit import radius_perimeter as rp_mod
from hipct_seg_debug.edit.adapter import read_triple
from hipct_seg_debug.edit.graphmodel import EditableGraph

OUT_DIR = Path(__file__).resolve().parent

#: Radius bins over the observed calibre range. Logarithmic, and on `r_area`
#: rather than on the perimeter radius or the graph's stored radius: coronary
#: calibres span more than a decade, and the bin must be defined by a quantity
#: that is not itself one of the two being compared on the y-axis.
N_RADIUS_BINS = 8

#: A segment needs this many sections before its median means anything. Same
#: floor `collapse_conservation` uses, applied there inside `sections`.
MIN_SEGMENT_SECTIONS = 8


def measure() -> list[dict]:
    """Per-segment medians of both estimators, raw and corrected."""
    g = EditableGraph(read_triple(GRAPH))
    labels, frame = open_lattice(SEG)
    sp = float(frame.seg_spacing[0])
    sampler = reformat.LabelSampler(labels, frame)
    print(f"graph {GRAPH}\n  {len(g.segments)} segments, voxel {sp:.2f} um")

    rows: list[dict] = []
    for n_done, seg in enumerate(g.segments, 1):
        if n_done % 25 == 0:
            print(f"  {n_done}/{len(g.segments)} segments", flush=True)
        got = sections(sampler, g, seg["id"], sp)
        if got is None:
            continue
        rp_raw, ra, _q_raw = got
        rp_corr = np.array([rp_mod.correct_perimeter_radius(v, sp) for v in rp_raw])
        ok = (rp_corr > 0) & (rp_raw > 0) & (ra > 0)
        if ok.sum() < MIN_SEGMENT_SECTIONS:
            continue
        rp_raw, rp_corr, ra = rp_raw[ok], rp_corr[ok], ra[ok]
        rows.append({
            "seg_id": int(seg["id"]),
            "strahler": int(seg.get("strahler", 0)),
            "n_sections": int(ok.sum()),
            "r_area_um": float(np.median(ra)),
            "r_perim_raw_um": float(np.median(rp_raw)),
            "r_perim_corr_um": float(np.median(rp_corr)),
            # Per-section ratios then medianed, not a ratio of medians: each
            # section is one paired observation of the same lumen.
            "uplift_raw": float(np.median(rp_raw / ra)),
            "uplift_corr": float(np.median(rp_corr / ra)),
            "Q_raw": float(np.median((rp_raw / ra) ** 2)),
            # Fraction of this segment's own sections where the corrected
            # perimeter radius is the larger of the two.
            "frac_sections_perim_larger": float(np.mean(rp_corr > ra)),
        })
    print(f"  {len(rows)} segments with >= {MIN_SEGMENT_SECTIONS} usable sections")
    return rows


def _stats(group: list[dict], key_r: str, key_up: str) -> dict:
    """Summary for one order or bin, on the segments it contains."""
    up = np.array([r[key_up] for r in group], dtype=float)
    ra = np.array([r["r_area_um"] for r in group], dtype=float)
    rp = np.array([r[key_r] for r in group], dtype=float)
    return {
        "n_segments": len(group),
        "r_area_um_median": float(np.median(ra)),
        "r_perim_um_median": float(np.median(rp)),
        "uplift_median": float(np.median(up)),
        "uplift_p25": float(np.percentile(up, 25)),
        "uplift_p75": float(np.percentile(up, 75)),
        # The direct answer to "which is larger here": the share of segments whose
        # median section has the perimeter radius on top.
        "frac_perim_larger": float(np.mean(up > 1.0)),
        "median_diff_um": float(np.median(rp - ra)),
    }


def by_strahler(rows: list[dict], corrected: bool) -> list[dict]:
    key_r = "r_perim_corr_um" if corrected else "r_perim_raw_um"
    key_up = "uplift_corr" if corrected else "uplift_raw"
    out = []
    for order in sorted({r["strahler"] for r in rows}):
        group = [r for r in rows if r["strahler"] == order]
        out.append({"strahler": order, **_stats(group, key_r, key_up)})
    return out


def radius_bin_edges(rows: list[dict]) -> np.ndarray:
    r = np.array([x["r_area_um"] for x in rows], dtype=float)
    r = r[np.isfinite(r) & (r > 0)]
    return np.logspace(math.log10(r.min()), math.log10(r.max() * 1.001),
                       N_RADIUS_BINS + 1)


def by_radius(rows: list[dict], edges: np.ndarray, corrected: bool) -> list[dict]:
    key_r = "r_perim_corr_um" if corrected else "r_perim_raw_um"
    key_up = "uplift_corr" if corrected else "uplift_raw"
    r = np.array([x["r_area_um"] for x in rows], dtype=float)
    idx = np.digitize(r, edges) - 1
    out = []
    for b in range(len(edges) - 1):
        group = [rec for rec, k in zip(rows, idx) if k == b]
        row = {"bin": b,
               "r_lo_um": float(edges[b]),
               "r_hi_um": float(edges[b + 1]),
               "r_mid_um": float(math.sqrt(edges[b] * edges[b + 1]))}
        if not group:
            out.append({**row, "n_segments": 0})
            continue
        out.append({**row, **_stats(group, key_r, key_up)})
    return out


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("")
        return
    cols: list[str] = []
    for r in rows:  # union of keys, first-seen order -- empty bins carry fewer
        for k in r:
            if k not in cols:
                cols.append(k)
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, restval="")
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.6g}" if isinstance(v, float) else v)
                        for k, v in r.items()})
    print(f"  wrote {path.name} ({len(rows)} rows)")


def _print_table(title: str, rows: list[dict], label_key: str,
                 label_fmt) -> None:
    print(f"\n{title}")
    print(f"  {'group':>14}{'n':>5}{'r_area':>9}{'r_perim':>9}"
          f"{'ratio':>8}{'IQR':>16}{'perim>area':>12}{'diff um':>9}")
    for r in rows:
        if not r.get("n_segments"):
            continue
        iqr = f"{r['uplift_p25']:.3f}-{r['uplift_p75']:.3f}"
        print(f"  {label_fmt(r[label_key], r):>14}{r['n_segments']:>5}"
              f"{r['r_area_um_median']:>9.0f}{r['r_perim_um_median']:>9.0f}"
              f"{r['uplift_median']:>8.3f}{iqr:>16}"
              f"{r['frac_perim_larger']:>11.0%}{r['median_diff_um']:>9.1f}")


def main() -> int:
    rows = measure()
    if not rows:
        print("no segments measured")
        return 1
    write_csv(OUT_DIR / "perimeter_vs_area_segments.csv", rows)

    edges = radius_bin_edges(rows)
    tables = {}
    for corrected, tag in ((True, "corrected"), (False, "raw")):
        tables[("strahler", tag)] = by_strahler(rows, corrected)
        tables[("radius", tag)] = by_radius(rows, edges, corrected)

    for tag in ("corrected", "raw"):
        write_csv(OUT_DIR / f"perimeter_vs_area_by_strahler_{tag}.csv",
                  tables[("strahler", tag)])
        write_csv(OUT_DIR / f"perimeter_vs_area_by_radius_{tag}.csv",
                  tables[("radius", tag)])

    up_c = np.array([r["uplift_corr"] for r in rows])
    up_r = np.array([r["uplift_raw"] for r in rows])
    print(f"\nOver all {len(rows)} segments, r_perimeter / r_area:")
    print(f"  raw        p5 {np.percentile(up_r,5):.3f}  "
          f"median {np.median(up_r):.3f}  p95 {np.percentile(up_r,95):.3f}"
          f"   perimeter larger in {np.mean(up_r>1):.0%} of segments")
    print(f"  corrected  p5 {np.percentile(up_c,5):.3f}  "
          f"median {np.median(up_c):.3f}  p95 {np.percentile(up_c,95):.3f}"
          f"   perimeter larger in {np.mean(up_c>1):.0%} of segments")

    for tag in ("corrected", "raw"):
        _print_table(f"By Strahler order ({tag} perimeter radius)",
                     tables[("strahler", tag)], "strahler",
                     lambda v, _r: f"order {v}")
        _print_table(f"By radius bin ({tag} perimeter radius)",
                     tables[("radius", tag)], "bin",
                     lambda v, r: f"{r['r_lo_um']:.0f}-{r['r_hi_um']:.0f}")

    try:
        plot(tables, OUT_DIR)
    except Exception as exc:  # a failed figure must not lose the tables
        print(f"[plot] skipped: {exc}")
    return 0


def plot(tables: dict, out_dir: Path) -> None:
    """Uplift against order and against calibre, corrected and raw."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Categorical slots 1-2 of the validated reference palette, in fixed order.
    CORR, RAW = "#2a78d6", "#eb6834"
    INK_SOFT, GRID = "#52514e", "#d8d7d2"
    plt.rcParams.update({
        "figure.dpi": 150, "savefig.dpi": 600, "savefig.bbox": "tight",
        "font.size": 8, "axes.edgecolor": INK_SOFT, "axes.linewidth": 0.8,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
        "axes.spines.top": False, "axes.spines.right": False,
    })
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9))

    for ax, (scope, xlabel) in zip(axes, (("strahler", "Strahler order"),
                                          ("radius", "Vessel radius, area-equivalent (µm)"))):
        for tag, colour in (("corrected", CORR), ("raw", RAW)):
            rows = [r for r in tables[(scope, tag)] if r.get("n_segments")]
            xs = ([r["strahler"] for r in rows] if scope == "strahler"
                  else [r["r_mid_um"] for r in rows])
            med = [r["uplift_median"] for r in rows]
            lo = [r["uplift_median"] - r["uplift_p25"] for r in rows]
            hi = [r["uplift_p75"] - r["uplift_median"] for r in rows]
            ax.errorbar(xs, med, yerr=[lo, hi], color=colour,
                        label=f"{tag} perimeter", marker="o", markersize=4.5,
                        linewidth=1.6, markeredgecolor="white",
                        markeredgewidth=0.8, elinewidth=0.9, capsize=2)
        # The line that answers the question: above it perimeter is larger.
        ax.axhline(1.0, color=INK_SOFT, linewidth=0.9, linestyle="--", zorder=1)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(r"$r_{\mathrm{perimeter}} / r_{\mathrm{area}}$")
        if scope == "radius":
            ax.set_xscale("log")
        else:
            ax.set_xticks([r["strahler"] for r in tables[("strahler", "corrected")]])
    axes[0].legend(frameon=False, fontsize=7)
    axes[0].annotate("perimeter larger", xy=(0.03, 1.0), xycoords=("axes fraction", "data"),
                     xytext=(0, 3), textcoords="offset points",
                     fontsize=6.5, color=INK_SOFT, va="bottom")
    fig.suptitle("Perimeter vs area radius, median and IQR over segments", fontsize=9)
    fig.tight_layout()
    for ext in ("png", "pdf", "svg"):
        fig.savefig(out_dir / f"perimeter_vs_area.{ext}")
    plt.close(fig)
    print(f"  wrote perimeter_vs_area.png/.pdf/.svg")


if __name__ == "__main__":
    raise SystemExit(main())
