"""Segment pruning + radius / length reporting.

Three operations:

- ``segment_mean_radius`` and ``segment_arc_length`` -- shared accessors
  used by the rest of the pipeline.
- ``report_segment_radius_range`` -- per-stage min/median/mean/max radius
  print, ignoring sub-mm bif-to-bif stubs.
- ``prune_short_terminal_nubs`` -- iteratively drops degree-1 segments
  shorter than ``MIN_TERMINAL_LENGTH_MM`` and contracts the new
  degree-2 nodes after each pass.
- ``prune_by_radius`` -- iteratively drops terminal segments whose mean
  radius / arc length is below the configured threshold.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .config import runtime_config as config
from .topology import merge_degree2_segments


def segment_mean_radius(seg: dict[str, Any], points: dict[int, tuple]) -> float:
    """Mean radius of a segment in mm (raw thicknesses are in micrometers)."""
    pids = seg["point_ids"]
    if not pids:
        return 0.0
    return sum(points[p][3] for p in pids) / len(pids) / 1000.0 * config.RADIUS_SCALE


def segment_arc_length(seg: dict[str, Any], points: dict[int, tuple]) -> float:
    """Arc length of a segment in mm."""
    pids = seg["point_ids"]
    if len(pids) < 2:
        return 0.0
    total = 0.0
    px, py, pz = points[pids[0]][0], points[pids[0]][1], points[pids[0]][2]
    for pid in pids[1:]:
        x, y, z = points[pid][0], points[pid][1], points[pid][2]
        dx, dy, dz = x - px, y - py, z - pz
        total += (dx * dx + dy * dy + dz * dz) ** 0.5
        px, py, pz = x, y, z
    return total / 1000.0


def report_segment_radius_range(
    label: str,
    segments: list[dict[str, Any]],
    points: dict[int, tuple],
    stub_max_length_mm: float | None = None,
) -> None:
    """Print min/median/mean/max segment-avg radius, excluding short bif stubs."""
    if not segments:
        print(f"  [{label}] no segments")
        return
    if stub_max_length_mm is None:
        stub_max_length_mm = config.STUB_SEGMENT_MAX_LENGTH_MM

    deg: dict[int, int] = {}
    for s in segments:
        deg[s["node1"]] = deg.get(s["node1"], 0) + 1
        deg[s["node2"]] = deg.get(s["node2"], 0) + 1

    radii_kept: list[float] = []
    n_stub = 0
    n_invalid = 0
    for s in segments:
        r_avg = segment_mean_radius(s, points)
        if r_avg <= 0.0:
            n_invalid += 1
            continue
        if (
            stub_max_length_mm > 0
            and deg.get(s["node1"], 0) >= 2
            and deg.get(s["node2"], 0) >= 2
        ):
            if segment_arc_length(s, points) < stub_max_length_mm:
                n_stub += 1
                continue
        radii_kept.append(r_avg)

    if not radii_kept:
        print(
            f"  [{label}] no non-stub segments with valid radii"
            f" (stubs={n_stub}, invalid={n_invalid})"
        )
        return

    arr = np.asarray(radii_kept)
    stub_note = f" [ignored {n_stub} stubs <{stub_max_length_mm}mm]" if n_stub else ""
    print(
        f"  [{label}] segment avg-radius: "
        f"min={arr.min():.4f}  median={np.median(arr):.4f}  "
        f"mean={arr.mean():.4f}  max={arr.max():.4f}  mm  "
        f"(n={len(arr)}){stub_note}"
    )


def prune_short_terminal_nubs(
    nodes: dict[int, tuple],
    points: dict[int, tuple],
    segments: list[dict[str, Any]],
    min_length_mm: float | None = None,
    max_iters: int | None = None,
) -> tuple[dict[int, tuple], list[dict[str, Any]], int]:
    """Iteratively drop short degree-1 segments + contract newly-exposed deg-2."""
    if min_length_mm is None:
        min_length_mm = config.MIN_TERMINAL_LENGTH_MM
    if max_iters is None:
        max_iters = config.PRUNE_ITER_MAX
    if not segments or min_length_mm <= 0:
        return dict(nodes), list(segments), 0

    cur_segs = list(segments)
    cur_nodes = dict(nodes)
    total = 0

    for _it in range(max_iters):
        deg: dict[int, int] = {}
        for s in cur_segs:
            deg[s["node1"]] = deg.get(s["node1"], 0) + 1
            deg[s["node2"]] = deg.get(s["node2"], 0) + 1

        keep: list[dict[str, Any]] = []
        pruned = 0
        for s in cur_segs:
            d1 = deg.get(s["node1"], 0)
            d2 = deg.get(s["node2"], 0)
            is_terminal = d1 == 1 or d2 == 1
            if is_terminal:
                pids = s["point_ids"]
                if len(pids) >= 2:
                    coords = np.array(
                        [
                            [
                                points[p][0] / 1000.0,
                                points[p][1] / 1000.0,
                                points[p][2] / 1000.0,
                            ]
                            for p in pids
                            if p in points
                        ]
                    )
                    L = (
                        float(np.linalg.norm(np.diff(coords, axis=0), axis=1).sum())
                        if len(coords) >= 2
                        else 0.0
                    )
                else:
                    L = 0.0
                if L < min_length_mm:
                    pruned += 1
                    continue
            keep.append(s)

        if pruned == 0:
            break
        total += pruned
        cur_nodes, cur_segs, _ = merge_degree2_segments(cur_nodes, points, keep)
    return cur_nodes, cur_segs, total


def prune_by_radius(
    nodes: dict[int, tuple],
    points: dict[int, tuple],
    segments: list[dict[str, Any]],
    min_radius: float | None = None,
    min_length: float | None = None,
) -> tuple[dict[int, tuple], dict[int, tuple], list[dict[str, Any]]]:
    """Iteratively prune terminal segments below the radius / length threshold."""
    if min_radius is None:
        min_radius = config.PRUNE_MIN_RADIUS
    if min_length is None:
        min_length = config.MIN_SEGMENT_LENGTH
    if min_radius <= 0 and min_length <= 0:
        return nodes, points, segments

    kept = list(segments)
    total = 0
    while True:
        coord: dict[int, int] = {}
        for s in kept:
            coord[s["node1"]] = coord.get(s["node1"], 0) + 1
            coord[s["node2"]] = coord.get(s["node2"], 0) + 1

        surviving: list[dict[str, Any]] = []
        pruned = 0
        for s in kept:
            is_terminal = coord.get(s["node1"], 0) == 1 or coord.get(s["node2"], 0) == 1
            if is_terminal:
                mean_r = segment_mean_radius(s, points)
                arc_len = segment_arc_length(s, points)
                too_thin = min_radius > 0 and mean_r < min_radius
                too_short = min_length > 0 and arc_len < min_length
                if too_thin or too_short:
                    pruned += 1
                    continue
            surviving.append(s)
        total += pruned
        kept = surviving
        if pruned == 0 or not kept:
            break

    if total > 0:
        reasons = []
        if min_radius > 0:
            reasons.append(f"radius < {min_radius} mm")
        if min_length > 0:
            reasons.append(f"length < {min_length} mm")
        print(
            f"  [PRUNE] Removed {total} terminal segments "
            f"({' or '.join(reasons)}) → {len(kept)} segments remain"
        )

    if not kept:
        print("  [WARN] No segments remain after pruning")
        return nodes, points, []

    keep_nids: set[int] = set()
    keep_pids: set[int] = set()
    for s in kept:
        keep_nids.add(s["node1"])
        keep_nids.add(s["node2"])
        keep_pids.update(s["point_ids"])

    coord_count: dict[int, int] = {nid: 0 for nid in keep_nids}
    for s in kept:
        coord_count[s["node1"]] += 1
        coord_count[s["node2"]] += 1

    filtered_nodes: dict[int, tuple] = {}
    for nid in keep_nids:
        x, y, z, _ = nodes[nid]
        filtered_nodes[nid] = (x, y, z, coord_count[nid])
    filtered_points = {pid: data for pid, data in points.items() if pid in keep_pids}
    return filtered_nodes, filtered_points, kept


__all__ = [
    "segment_mean_radius",
    "segment_arc_length",
    "report_segment_radius_range",
    "prune_short_terminal_nubs",
    "prune_by_radius",
]
