"""Assemble the full metric report: per-edge table, k-means cluster order,
distribution plots, Amira-colourable spatial graphs, and before/after comparison.

Reconstructs the outputs described in the MATLAB readme /
``Graphs_strahler_against_metrics.m``, computing metrics from geometry (see
:mod:`skeleton_analysis.metrics.geometry`) so they respond to the corrections.
Forest-aware: metrics needing a root are computed per connected component.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from skeleton_analysis.io.amira import SpatialGraph, write_amira
from skeleton_analysis.metrics.branching_angles import branching_angles
from skeleton_analysis.metrics.intervessel import intervessel_distance
from skeleton_analysis.metrics.murray import murray_law
from skeleton_analysis.metrics.aggregate import branching_ratio, violin_by_strahler
from skeleton_analysis.metrics import geometry as geom
from skeleton_analysis.ordering.pipeline import order_forest
from skeleton_analysis.utils.split import split_connected_components

_KMEANS_FEATURES = ("topo", "radius", "tortuosity", "branching_angle", "ld_ratio", "intervessel")


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------
def edge_metrics_table(graph: SpatialGraph, roots: Sequence[int]) -> pd.DataFrame:
    """One row per edge with all per-edge metrics (computed, not read from Amira)."""
    edges = np.asarray(graph.edge_connectivity, dtype=np.int64)
    n = len(edges)

    if "strahler" in graph.edge_fields and "topo" in graph.edge_fields:
        strahler = np.asarray(graph.edge_fields["strahler"]).astype(int)
        topo = np.asarray(graph.edge_fields["topo"]).astype(int)
    else:
        strahler, topo, _ = order_forest(edges, roots)

    rstats = geom.radius_stats(graph)
    radius = rstats.avg
    diameter = 2.0 * radius
    length = geom.segment_lengths(graph)
    chord = geom.chord_lengths(graph)
    tort = geom.tortuosity(graph)
    vol = geom.volumes(graph)
    sa = geom.surface_areas(graph)
    with np.errstate(divide="ignore", invalid="ignore"):
        ld = np.where(diameter > 0, length / diameter, np.nan)
    ivd = intervessel_distance(graph)

    # Branching angle + tree id, per connected component.
    ba_edge = np.full(n, np.nan)
    tree = np.full(n, -1, dtype=np.int64)
    for ti, comp in enumerate(split_connected_components(graph)):
        tree[comp.edge_indices] = ti
        rg = next((r for r in roots if r in comp.node_map), None)
        if rg is None:
            continue
        ba_local, _ = branching_angles(comp.graph, root_id=comp.node_map[rg])
        ba_edge[comp.edge_indices] = ba_local

    df = pd.DataFrame(
        {
            "edge": np.arange(n),
            "tree": tree,
            "strahler": strahler,
            "topo": topo,
            "radius": radius,
            "diameter": diameter,
            "length": length,
            "chord": chord,
            "tortuosity": tort,
            "volume": vol,
            "surface_area": sa,
            "ld_ratio": ld,
            "branching_angle": ba_edge,
            "intervessel": ivd,
        }
    )
    # Amira cross-check columns (static; for the 'before' state).
    for src, dst in (("MeanRadius", "amira_MeanRadius"),
                     ("Tortuosity", "amira_Tortuosity"),
                     ("CurvedLength", "amira_CurvedLength")):
        if src in graph.edge_fields:
            df[dst] = np.asarray(graph.edge_fields[src], dtype=float)
    return df


def murray_table(graph: SpatialGraph, roots: Sequence[int]) -> pd.DataFrame:
    """Murray's-law quantities at each branch point (pooled over trees).

    Uses the *computed* mean radius (mean of thickness) so it responds to the
    corrections, rather than Amira's static ``MeanRadius``.
    """
    frames = []
    for comp in split_connected_components(graph):
        rg = next((r for r in roots if r in comp.node_map), None)
        if rg is None:
            continue
        cg = comp.graph
        cg.set_edge_field("MeanRadius", geom.radius_stats(cg).avg)
        df = murray_law(cg, root_id=comp.node_map[rg], radius_field="MeanRadius")
        frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def assign_kmeans(
    table: pd.DataFrame,
    features: Sequence[str] = _KMEANS_FEATURES,
    k: Optional[int] = None,
    k_range: Tuple[int, int] = (2, 8),
    random_state: int = 0,
) -> Tuple[np.ndarray, int]:
    """K-means cluster order over the metric features.

    Standardises features (NaN->0, as in the MATLAB), auto-selects k by silhouette
    when ``k`` is None, and relabels clusters ordered by mean radius so the
    "cluster order" increases with vessel size. Returns ``(labels, k)``.
    Requires the ``[viz]`` extra (scikit-learn).
    """
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    from sklearn.preprocessing import StandardScaler

    X = np.nan_to_num(table[list(features)].to_numpy(dtype=float), nan=0.0)
    Xs = StandardScaler().fit_transform(X)

    if k is None:
        best_k, best_score = k_range[0], -1.0
        for kk in range(k_range[0], k_range[1] + 1):
            if len(Xs) <= kk:
                break
            lab = KMeans(n_clusters=kk, n_init=10, random_state=random_state).fit_predict(Xs)
            if len(np.unique(lab)) < 2:
                continue
            score = silhouette_score(Xs, lab)
            if score > best_score:
                best_score, best_k = score, kk
        k = best_k

    labels = KMeans(n_clusters=k, n_init=10, random_state=random_state).fit_predict(Xs)
    # Relabel clusters in order of increasing mean radius.
    mean_r = pd.Series(table["radius"].to_numpy(dtype=float)).groupby(labels).mean()
    order = list(mean_r.sort_values().index)
    remap = {old: new for new, old in enumerate(order)}
    return np.array([remap[int(l)] for l in labels], dtype=np.int64), int(k)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
_VIOLIN_METRICS = [
    ("radius", "Mean radius (thickness units)"),
    ("tortuosity", "Tortuosity"),
    ("length", "Segment length (um)"),
    ("ld_ratio", "Length / diameter"),
    ("branching_angle", "Branching angle (deg)"),
    ("intervessel", "Intervessel distance (um)"),
    ("volume", "Volume (um^3)"),
]


def plot_report(
    table: pd.DataFrame,
    murray_df: pd.DataFrame,
    out_dir,
    prefix: str = "",
) -> None:
    """Save the full set of distribution/summary plots + the per-edge CSV.

    Requires the ``[viz]`` extra (matplotlib + seaborn).
    """
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(out_dir / f"{prefix}metrics_summary.csv", index=False)

    # Per-Strahler violin plots.
    for col, label in _VIOLIN_METRICS:
        if col not in table or table[col].dropna().empty:
            continue
        fig, ax = plt.subplots(figsize=(6, 4))
        violin_by_strahler(table, col, order_column="strahler", ax=ax)
        ax.set_ylabel(label)
        fig.tight_layout()
        fig.savefig(out_dir / f"{prefix}violin_{col}.png", dpi=120)
        plt.close(fig)

    # Vessel counts + branching ratio (mean±std across trees).
    counts = table.groupby("strahler").size()
    br_rows = []
    for _t, sub in table.groupby("tree"):
        br_rows.append(branching_ratio(sub.groupby("strahler").size()))
    br = pd.DataFrame(br_rows)
    br_mean = br.mean(axis=0) if not br.empty else pd.Series(dtype=float)
    br_std = br.std(axis=0) if not br.empty else pd.Series(dtype=float)

    fig, ax1 = plt.subplots(figsize=(6, 4))
    ax1.bar(counts.index, counts.values, color="0.7", label="vessel count")
    ax1.set_xlabel("Strahler order")
    ax1.set_ylabel("vessel count")
    ax2 = ax1.twinx()
    if not br_mean.empty:
        ax2.errorbar(br_mean.index, br_mean.values, yerr=br_std.values,
                     fmt="bo-", capsize=3, label="branching ratio")
    ax2.axhline(2.0, color="r", ls="--", lw=1)
    ax2.set_ylabel("branching ratio (mean +/- std)")
    fig.tight_layout()
    fig.savefig(out_dir / f"{prefix}strahler_counts_branching_ratio.png", dpi=120)
    plt.close(fig)

    # Counts vs topological generation.
    _bar_counts(table, "topo", "Topological generation",
                out_dir / f"{prefix}topo_counts.png")
    # Counts vs k-means order (if present).
    if "kmeans_cluster" in table:
        _bar_counts(table, "kmeans_cluster", "k-means cluster order",
                    out_dir / f"{prefix}kmeans_counts.png")

    # Murray's law scatter.
    if not murray_df.empty and {"parent_rad_cubed", "sumchild_cubed"}.issubset(murray_df):
        fig, ax = plt.subplots(figsize=(5, 5))
        sc = ax.scatter(murray_df["parent_rad_cubed"], murray_df["sumchild_cubed"],
                        c=murray_df.get("strahler", None), cmap="viridis", s=18)
        lim = [0, float(np.nanmax([murray_df["parent_rad_cubed"].max(),
                                   murray_df["sumchild_cubed"].max()]))]
        ax.plot(lim, lim, "r--", lw=1, label="Murray (y=x)")
        ax.set_xlabel("parent radius^3")
        ax.set_ylabel("sum(child radius^3)")
        ax.set_title("Murray's law")
        if murray_df.get("strahler") is not None:
            fig.colorbar(sc, ax=ax, label="Strahler")
        ax.legend()
        fig.tight_layout()
        fig.savefig(out_dir / f"{prefix}murray.png", dpi=120)
        plt.close(fig)


def _bar_counts(table, col, xlabel, path):
    import matplotlib.pyplot as plt

    counts = table.groupby(col).size()
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(counts.index, counts.values, color="0.6")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("vessel count")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Amira export + state comparison
# ---------------------------------------------------------------------------
_METRIC_EDGE_FIELDS = [
    "strahler", "topo", "kmeans_cluster", "radius", "tortuosity",
    "ld_ratio", "branching_angle", "intervessel", "volume",
]


def write_metric_graph(graph: SpatialGraph, table: pd.DataFrame, out_path) -> None:
    """Write a spatial graph annotated with per-edge metric fields for Amira.

    NaN metrics (e.g. branching angle on non-branch edges) are written as -1 so
    Amira colour-maps don't choke.
    """
    g = graph.copy()
    for col in _METRIC_EDGE_FIELDS:
        if col not in table:
            continue
        vals = table[col].to_numpy()
        if col in ("strahler", "topo", "kmeans_cluster"):
            arr = np.nan_to_num(vals, nan=-1).astype(np.int64)
        else:
            arr = np.where(np.isnan(vals.astype(float)), -1.0, vals.astype(float))
        g.set_edge_field(col, arr)
    write_amira(g, out_path)


_COMPARE_METRICS = ["radius", "volume", "surface_area", "ld_ratio"]


def compare_states(tables: Dict[str, pd.DataFrame], out_dir) -> pd.DataFrame:
    """Overlay per-Strahler means of radius-dependent metrics across states.

    Saves one PNG per metric plus ``state_comparison.csv``. Returns the tidy
    comparison DataFrame (state, strahler, metric, mean, std).
    """
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for metric in _COMPARE_METRICS:
        fig, ax = plt.subplots(figsize=(6, 4))
        for state, tbl in tables.items():
            if metric not in tbl:
                continue
            grp = tbl.groupby("strahler")[metric]
            mean, std = grp.mean(), grp.std()
            ax.errorbar(mean.index, mean.values, yerr=std.values, fmt="o-",
                        capsize=3, label=state)
            for order in mean.index:
                rows.append({"state": state, "strahler": int(order), "metric": metric,
                             "mean": float(mean[order]), "std": float(std.get(order, np.nan))})
        ax.set_xlabel("Strahler order")
        ax.set_ylabel(metric)
        ax.set_title(f"{metric} across correction states")
        ax.legend()
        fig.tight_layout()
        fig.savefig(out_dir / f"compare_{metric}.png", dpi=120)
        plt.close(fig)

    comparison = pd.DataFrame(rows)
    comparison.to_csv(out_dir / "state_comparison.csv", index=False)
    return comparison
