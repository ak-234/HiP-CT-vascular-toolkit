"""Bifurcation carina taper.

Each spline endpoint that terminates at a degree>=3 node has its last
K radii tapered DOWN to a small "carina tip" radius at the bif endpoint.
This converts the capsule's hemispherical end-cap (radius r_local) into
a conical tip (radius -> carina_tip). N cones converging at a shared
bif node smooth-min into a clean Y / T / X surface rather than a
ball-shaped union of N hemispheres.

K per endpoint is the contiguous is_junction-flagged run from the
endpoint, capped by ``BIF_CARINA_TAPER_MAX_PTS`` and floored by
``BIF_CARINA_TAPER_MIN_PTS``. is_junction comes from the existing
patch-84 cross-section overlap detector in
``topology.label_centreline_topology_aware``. Terminal (deg==1)
endpoints are never tapered; the flat-cap pipeline depends on full
endpoint radii there.

Coords are NOT modified — only radii. cs_pos / L / start_radius /
end_radius are kept consistent.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .config import runtime_config as config
from .topology import label_centreline_topology_aware


def taper_bifurcation_carina(
    valid_splines: list[dict[str, Any]],
    nodes: dict[int, tuple],
    points: dict[int, tuple],
    segments: list[dict[str, Any]],
) -> dict[str, Any]:
    """Taper the bif-end radii of each spline at deg>=3 endpoints.

    Mutates each entry of ``valid_splines`` in place: rewrites the last
    K entries of ``sp["radii"]`` (or first K, for start-end taper) with
    a linear interpolation from the interior reference radius down to
    ``carina_tip = max(BIF_CARINA_TIP_RADIUS_FACTOR * r_interior,
    BIF_CARINA_TIP_MIN_MM)`` at the bif endpoint. Updates
    ``sp["start_radius"]`` / ``sp["end_radius"]`` accordingly. ``coords``
    and ``cs_pos`` are untouched.

    Returns a diagnostic report::

        {
            "n_tapered_endpoints": int,
            "n_radii_modified":    int,
            "records":             list[dict],
        }
    """
    report: dict[str, Any] = {
        "n_tapered_endpoints": 0,
        "n_radii_modified": 0,
        "records": [],
    }
    if not valid_splines:
        return report

    node_to_splines: dict[int, list[int]] = {}
    for sp_idx, sp in enumerate(valid_splines):
        for nid in (sp["node1_id"], sp["node2_id"]):
            node_to_splines.setdefault(nid, []).append(sp_idx)

    try:
        import networkx as nx
    except ImportError:
        return report

    g = nx.Graph()
    for nid, nd in nodes.items():
        x, y, z = nd[0], nd[1], nd[2]
        g.add_node(
            nid, pos=np.array([x / 1000.0, y / 1000.0, z / 1000.0], dtype=np.float64)
        )

    for sp_idx, sp in enumerate(valid_splines):
        u, v = sp["node1_id"], sp["node2_id"]
        if u == v:
            continue
        if g.has_edge(u, v):
            continue
        g.add_edge(
            u, v,
            points=np.asarray(sp["coords"], dtype=np.float64),
            radii=np.asarray(sp["radii"], dtype=np.float64),
            sp_idx=sp_idx,
        )

    if g.number_of_edges() == 0:
        return report

    is_junction_flat, _all_pts = label_centreline_topology_aware(g)

    sp_is_junction: dict[int, np.ndarray] = {}
    offset = 0
    for _u, _v, d in g.edges(data=True):
        n_pts = len(d["points"])
        sp_is_junction[int(d["sp_idx"])] = is_junction_flat[offset:offset + n_pts]
        offset += n_pts

    max_pts = max(1, int(config.BIF_CARINA_TAPER_MAX_PTS))
    min_pts = max(1, int(config.BIF_CARINA_TAPER_MIN_PTS))
    tip_factor = float(config.BIF_CARINA_TIP_RADIUS_FACTOR)
    tip_min_mm = float(config.BIF_CARINA_TIP_MIN_MM)

    for sp_idx, sp in enumerate(valid_splines):
        ij = sp_is_junction.get(sp_idx)
        if ij is None:
            continue
        radii = np.asarray(sp["radii"], dtype=np.float64).copy()
        n = len(radii)
        if n != len(ij) or n < min_pts + 2:
            continue

        nid1 = sp["node1_id"]
        nid2 = sp["node2_id"]
        deg1 = len(node_to_splines.get(nid1, []))
        deg2 = len(node_to_splines.get(nid2, []))
        is_bif1 = deg1 >= 3
        is_bif2 = deg2 >= 3
        if not is_bif1 and not is_bif2:
            continue

        # K at start: contiguous is_junction run from coords[0] inward.
        k_start_raw = 0
        if is_bif1:
            while k_start_raw < n and bool(ij[k_start_raw]):
                k_start_raw += 1
        # K at end: contiguous is_junction run from coords[-1] inward.
        k_end_raw = 0
        if is_bif2:
            while k_end_raw < n and bool(ij[n - 1 - k_end_raw]):
                k_end_raw += 1

        k_start = min(k_start_raw, max_pts) if k_start_raw >= min_pts else 0
        k_end = min(k_end_raw, max_pts) if k_end_raw >= min_pts else 0

        # Ensure interior reference points stay available and the two
        # taper regions don't overlap.
        if k_start + k_end >= n - 1:
            shrink = (k_start + k_end) - (n - 2)
            # Trim the larger side preferentially.
            while shrink > 0 and (k_start > 0 or k_end > 0):
                if k_start >= k_end and k_start > 0:
                    k_start -= 1
                elif k_end > 0:
                    k_end -= 1
                else:
                    break
                shrink -= 1
            if k_start < min_pts:
                k_start = 0
            if k_end < min_pts:
                k_end = 0

        if k_start == 0 and k_end == 0:
            continue

        modified_here = 0

        if k_start > 0:
            # Interior reference: one point inside the taper region.
            r_interior = float(radii[k_start])
            r_tip = max(tip_factor * r_interior, tip_min_mm)
            # j=0 -> tip (at coords[0], the bif end);
            # j=k_start-1 -> just inside taper, near r_interior.
            for j in range(k_start):
                t = (k_start - j) / float(k_start)  # 1 at the tip, 1/K at the inner boundary
                radii[j] = (1.0 - t) * r_interior + t * r_tip
            modified_here += k_start
            report["records"].append({
                "seg_id": sp.get("seg_id"),
                "node_id": nid1,
                "end": "start",
                "k_tapered": int(k_start),
                "r_interior": float(r_interior),
                "r_tip": float(r_tip),
                "deg": int(deg1),
            })
            report["n_tapered_endpoints"] += 1

        if k_end > 0:
            r_interior = float(radii[n - 1 - k_end])
            r_tip = max(tip_factor * r_interior, tip_min_mm)
            for j in range(k_end):
                # j=0 -> just inside the taper boundary; j=k_end-1 -> the tip at coords[-1].
                t = (j + 1) / float(k_end)  # 1/K at the inner boundary, 1 at the tip
                radii[n - k_end + j] = (1.0 - t) * r_interior + t * r_tip
            modified_here += k_end
            report["records"].append({
                "seg_id": sp.get("seg_id"),
                "node_id": nid2,
                "end": "end",
                "k_tapered": int(k_end),
                "r_interior": float(r_interior),
                "r_tip": float(r_tip),
                "deg": int(deg2),
            })
            report["n_tapered_endpoints"] += 1

        sp["radii"] = radii
        sp["start_radius"] = float(radii[0])
        sp["end_radius"] = float(radii[-1])
        report["n_radii_modified"] += modified_here

    return report


__all__ = ["taper_bifurcation_carina"]
