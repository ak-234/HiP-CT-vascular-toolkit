"""The DPC walk.

From *A topology-preserving three-stage framework for fully-connected coronary
artery extraction* (Medical Image Analysis, 2025; arXiv:2504.01597), which
extends the walk introduced in CorSegRec (MICCAI 2023).

Where the geometric proposers interpolate a curve between two endpoints and hope
the image agrees, the DPC walk *reads the image*: it steps voxel by voxel from a
disconnected end toward a candidate target, and at each step scores every
neighbour by three terms whose initials give the method its name.

    D(A_k) = -|| A_k - p_m ||               distance, pulling toward the target
    P(A_k) = centreline probability          max-min normalised over the neighbourhood
    C(A_k) = cos(o_k, o_-1) + cos(o_k, o_-2) agreement with the last two steps

    DPC(A_k) = D(A_k) + w * P_N(A_k) + C(A_k),      w = 5

The highest-scoring neighbour becomes the next centreline point. Neighbours more
than 90 degrees off the recent direction are filtered out first, as are points
already consumed. Per equation 10, C participates while the previous directions
have cosine similarity at most 0.5 and is omitted once they already closely agree.

Three reconnection types are distinguished, as in the paper. Type 1 and 2 both
aim at an endpoint and differ only in permitted reach. **Type 3 -- "branch
occurrence" -- aims at a whole centreline rather than at its end**: D becomes the
negative distance to the *nearest point of* the target polyline, and the
direction filter uses the walk's own history rather than the straight line to the
goal. That is the case where a vessel joins the side of another, and it is why
this module can propose T-junctions that endpoint matching cannot.

Thresholds in the paper are in CTA voxels (< 60, < 80). HiP-CT here is 32.99 um
raw and the segmentation is 2x2x2-binned, so distances are expressed in
micrometres and made relative to the local radius, which transfers across
resolutions.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .candidates import Bridge, endpoint_tangent, resample_by_arclength
from .probability import Roi

# The paper's weight on the probability term (its equation 12).
OMEGA = 5.0
# 5x5x5 neighbourhood, minus the centre. Offsets at Chebyshev distance 2 as well
# as 1, so the walk can cross a one-voxel dropout without stalling.
NEIGHBOURHOOD_RADIUS = 2
# cos(90 deg): neighbours behind the walk are dropped before scoring.
MAX_TURN_COS = 0.0


@dataclass
class DpcParams:
    """Tunables, with the paper's values where it gives one."""

    omega: float = OMEGA
    w_distance: float = 1.0
    w_cosine: float = 1.0
    neighbourhood: int = NEIGHBOURHOOD_RADIUS
    neighbourhood_policy: str = "two-level"  # "two-level" | "full"
    coarse_until_distance: float = 3.0
    max_steps: int = 400
    # Reach, as a multiple of the source radius. The paper's type 1 / type 2
    # split is a hard voxel count; radius-relative transfers between scanners.
    type1_reach: float = 8.0
    type2_reach: float = 15.0
    type3_reach: float = 15.0
    # Acceptance, after the walk has reached its target.
    min_mean_probability: float = 0.15
    max_probability_drop: float = 0.6
    # The mean can hide a trough: a short hop between two vessels that pass close
    # is high at both ends and near zero in the tissue between them, and averages
    # out to something respectable. A real vessel has no such hole, so the
    # *minimum* along the path is checked against the parent vessel too. This is
    # the same property the paper's stationarity test is reaching for, made
    # direct because a stationarity test needs a long series to have power and a
    # bridge is often only a dozen steps.
    min_trough_ratio: float = 0.35
    adf_p_max: float = 0.15
    min_grey_continuity: float = 0.5


@dataclass
class DpcResult:
    path_um: np.ndarray
    probabilities: np.ndarray
    reached: bool
    steps: int
    reason: str = ""
    stats: dict = field(default_factory=dict)
    probability_sequence: np.ndarray = field(default_factory=lambda: np.array([]))
    grayscale_sequence: np.ndarray = field(default_factory=lambda: np.array([]))
    distance_scores: np.ndarray = field(default_factory=lambda: np.array([]))
    probability_scores: np.ndarray = field(default_factory=lambda: np.array([]))
    cosine_scores: np.ndarray = field(default_factory=lambda: np.array([]))


def _offsets(radius: int, level: str = "full") -> np.ndarray:
    r = range(-radius, radius + 1)
    out = np.array([(dz, dy, dx) for dz in r for dy in r for dx in r
                    if (dz, dy, dx) != (0, 0, 0)], dtype=np.float64)
    distance = np.linalg.norm(out, axis=1)
    if level == "fine":
        return out[distance < 2.0]
    if level == "coarse":
        return out[(distance >= 2.0) & (distance <= 3.0)]
    if level != "full":
        raise ValueError(f"unknown DPC neighbourhood level: {level}")
    return out


def walk(
    roi: Roi,
    probability,
    start_um: np.ndarray,
    target_um: np.ndarray,
    *,
    start_direction: np.ndarray | None = None,
    target_is_polyline: bool = False,
    reconnection_type: int | None = None,
    params: DpcParams | None = None,
) -> DpcResult:
    """Walk from `start_um` to `target_um`, one voxel step at a time.

    `target_um` is a single point unless `target_is_polyline`, in which case it is
    an ``(M, 3)`` centreline and the distance term aims at whichever of its points
    is nearest -- the paper's type 3 behaviour.
    """
    p = params or DpcParams()
    if p.neighbourhood_policy not in ("two-level", "full"):
        raise ValueError("neighbourhood_policy must be 'two-level' or 'full'")
    spacing_zyx = np.asarray(roi.spacing_um, dtype=np.float64)[::-1]

    target = np.asarray(target_um, dtype=np.float64).reshape(-1, 3)
    target_idx = roi.to_index(target)
    goal_idx = target_idx if target_is_polyline else target_idx[0]

    current = roi.to_index(np.asarray(start_um).reshape(1, 3))[0]
    history: list[np.ndarray] = []
    if start_direction is not None:
        d = np.asarray(start_direction, dtype=np.float64)[::-1] / np.maximum(spacing_zyx, 1e-9)
        n = np.linalg.norm(d)
        if n > 1e-9:
            history.append(d / n)

    path = [current.copy()]
    probs: list[float] = []
    chosen_d: list[float] = []
    chosen_p: list[float] = []
    chosen_c: list[float] = []
    visited = {tuple(np.round(current).astype(int))}
    # Arriving within one neighbourhood step counts as reaching the target.
    arrive = float(p.neighbourhood) + 0.5

    reason = "ran out of steps"
    reached = False
    for _ in range(p.max_steps):
        remaining = _distance_to_goal(current, goal_idx, target_is_polyline)
        if remaining <= arrive:
            reached = True
            reason = "reached the target"
            break

        if p.neighbourhood_policy == "full":
            offsets = _offsets(p.neighbourhood, "full")
        else:
            level = "coarse" if remaining > p.coarse_until_distance else "fine"
            offsets = _offsets(p.neighbourhood, level)
        candidates = current + offsets
        keep = roi.inside(candidates)
        if not keep.any():
            reason = "walked out of the region of interest"
            break
        candidates = candidates[keep]

        # Direction filter: nothing more than 90 degrees off the recent heading,
        # which is what stops the walk doubling back along the vessel it came from.
        steps = candidates - current
        norms = np.linalg.norm(steps, axis=1)
        good = norms > 1e-9
        candidates, steps, norms = candidates[good], steps[good], norms[good]
        if len(candidates) == 0:
            reason = "no usable neighbour"
            break
        unit = steps / norms[:, None]
        if history:
            cos_recent = unit @ history[-1]
            ahead = cos_recent > MAX_TURN_COS
            candidates, unit = candidates[ahead], unit[ahead]
            if len(candidates) == 0:
                reason = "no neighbour satisfies the direction constraint"
                break
        if target_is_polyline and len(history) >= 2:
            combined = history[-1] + history[-2]
            combined_norm = float(np.linalg.norm(combined))
            if combined_norm > 1e-9:
                ahead = (unit @ (combined / combined_norm)) > MAX_TURN_COS
                candidates, unit = candidates[ahead], unit[ahead]
                if len(candidates) == 0:
                    reason = "no Type 3 neighbour satisfies both history constraints"
                    break

        # Already-consumed points are excluded, so the walk cannot sit still.
        fresh = np.array(
            [tuple(np.round(c).astype(int)) not in visited for c in candidates]
        )
        if fresh.any():
            candidates, unit = candidates[fresh], unit[fresh]
        if len(candidates) == 0:
            reason = "every neighbour already visited"
            break

        # D: negative distance to the goal. The paper only normalises P.
        d_term = -np.array(
            [_distance_to_goal(c, goal_idx, target_is_polyline) for c in candidates]
        )
        # P: centreline probability, max-min normalised over the neighbourhood so
        # the walk still has a gradient where the raw values are all small.
        p_raw = np.asarray(probability(roi.to_world(candidates)), dtype=np.float64)
        # C participates while the recent directions are not already close. This
        # is equation 10's <= 0.5 branch; omitting it on a straight run avoids
        # over-rewarding a direction that already dominates.
        c_term = _cosine_term(unit, history)

        p_term = _minmax(p_raw)
        score = p.w_distance * d_term + p.omega * p_term + p.w_cosine * c_term
        best = int(np.argmax(score))

        step = candidates[best] - current
        norm = float(np.linalg.norm(step))
        if norm > 1e-9:
            history.append(step / norm)
        current = candidates[best]
        visited.add(tuple(np.round(current).astype(int)))
        path.append(current.copy())
        probs.append(float(p_raw[best]))
        chosen_d.append(float(d_term[best]))
        chosen_p.append(float(p_term[best]))
        chosen_c.append(float(c_term[best]))

    path_idx = np.asarray(path)
    return DpcResult(
        path_um=roi.to_world(path_idx),
        probabilities=np.asarray(probs),
        reached=reached,
        steps=len(path) - 1,
        reason=reason,
        distance_scores=np.asarray(chosen_d),
        probability_scores=np.asarray(chosen_p),
        cosine_scores=np.asarray(chosen_c),
        stats={"reconnection_type": reconnection_type},
    )


def _distance_to_goal(point: np.ndarray, goal: np.ndarray, polyline: bool) -> float:
    if not polyline:
        return float(np.linalg.norm(point - goal))
    return float(np.min(np.linalg.norm(goal - point, axis=1)))


def _minmax(values: np.ndarray) -> np.ndarray:
    lo, hi = float(np.min(values)), float(np.max(values))
    if hi - lo < 1e-12:
        return np.zeros_like(values)
    return (values - lo) / (hi - lo)


def _cosine_term(unit_steps: np.ndarray, history: list[np.ndarray]) -> np.ndarray:
    """Equation 10: C is active only while the last two offsets are not close."""
    out = np.zeros(len(unit_steps), dtype=np.float64)
    if len(history) >= 2 and float(history[-1] @ history[-2]) <= 0.5:
        out = unit_steps @ history[-1] + unit_steps @ history[-2]
    return out


def validate(
    result: DpcResult,
    reference_probability: np.ndarray,
    *,
    grey_along_path: np.ndarray | None = None,
    params: DpcParams | None = None,
) -> tuple[bool, str, dict]:
    """Decide whether a completed walk is a real vessel or a shortcut.

    The paper's three checks, in order of how often they fire here:

    1. **Mean probability**, absolute and relative to the known-good centreline
       the walk started from. A path that is plausible in isolation but far worse
       than its own parent vessel is a shortcut through tissue.
    2. **Stationarity (ADF)** of the probability series. A real vessel gives a
       roughly stationary signal along its length; a path that leaves the vessel
       and comes back gives a trending or wandering one.
    3. **Greyscale continuity** at the join, when the raw values are supplied.
    """
    p = params or DpcParams()
    stats: dict = {}
    if not result.reached:
        return False, result.reason, stats
    if len(result.probabilities) < 3:
        return False, "path too short to judge", stats

    mean_p = float(np.mean(result.probabilities))
    ref = float(np.mean(reference_probability)) if len(reference_probability) else 0.0
    stats["mean_probability"] = mean_p
    stats["reference_probability"] = ref
    trough = float(np.min(result.probabilities))
    stats["min_probability"] = trough
    if ref > 1e-9:
        drop = 1.0 - mean_p / ref
        stats["probability_drop"] = drop
        stats["trough_ratio"] = trough / ref

    probability_sequence = (
        result.probability_sequence
        if len(result.probability_sequence) else result.probabilities
    )
    stats["probability_adf_p"] = _adf_pvalue(probability_sequence)
    stats["adf_p"] = stats["probability_adf_p"]  # compatibility with old reports

    grey = result.grayscale_sequence if len(result.grayscale_sequence) else grey_along_path
    if grey is not None and len(grey) >= 3:
        grey = np.asarray(grey, dtype=np.float64)
        stats["grayscale_adf_p"] = _adf_pvalue(grey)
        continuity = _continuity(grey)
        stats["grey_continuity"] = continuity

    hipct_failures = []
    if mean_p < p.min_mean_probability:
        hipct_failures.append(f"mean centreline probability {mean_p:.3f} is too low")
    if stats.get("probability_drop", 0.0) > p.max_probability_drop:
        hipct_failures.append(
            f"probability drops {100*stats['probability_drop']:.0f}% below the parent vessel"
        )
    if stats.get("trough_ratio", 1.0) < p.min_trough_ratio:
        hipct_failures.append(
            f"the path passes through a hole "
            f"({100*stats['trough_ratio']:.0f}% of the parent vessel)"
        )
    if stats.get("grey_continuity", 1.0) < p.min_grey_continuity:
        hipct_failures.append(
            f"greyscale discontinuity at the join ({stats['grey_continuity']:.2f})"
        )

    paper_failures = []
    if stats["probability_adf_p"] is not None and stats["probability_adf_p"] > p.adf_p_max:
        paper_failures.append(
            f"probability series is not stationary (ADF p={stats['probability_adf_p']:.3f})"
        )
    if stats.get("grayscale_adf_p") is not None and stats["grayscale_adf_p"] > p.adf_p_max:
        paper_failures.append(
            f"grayscale series is not stationary (ADF p={stats['grayscale_adf_p']:.3f})"
        )
    stats["paper_validation"] = {
        "accepted": not paper_failures, "failures": paper_failures,
    }
    stats["hipct_safeguards"] = {
        "accepted": not hipct_failures, "failures": hipct_failures,
    }
    if hipct_failures:
        return False, hipct_failures[0], stats
    if paper_failures:
        return False, paper_failures[0], stats

    return True, "accepted", stats


def _adf_pvalue(series: np.ndarray) -> float | None:
    """Augmented Dickey-Fuller p-value, or ``None`` when it cannot be computed.

    Statsmodels' public ``adfuller`` reaches NumPy's SVD. On the supported Windows
    Python 3.9 stack that native call can terminate the interpreter with
    ``0xc06d007f``. The regression is tiny, so solve its normal equations with a
    pivoted scalar Gauss-Jordan implementation and retain Statsmodels' published
    MacKinnon p-value calibration. This is deterministic and avoids the unstable
    DLL path without weakening the test to a different statistic.
    """
    values = np.asarray(series, dtype=np.float64)
    if len(values) < 8 or float(np.std(values)) < 1e-9:
        return None
    try:
        from statsmodels.tsa.adfvalues import mackinnonp

        delta = np.diff(values)
        max_lag = min(int(12.0 * (len(values) / 100.0) ** 0.25),
                      max((len(delta) - 4) // 2, 0))
        best: tuple[float, float] | None = None
        for lag in range(max_lag + 1):
            y = delta[lag:]
            columns = [np.ones(len(y)), values[lag:-1]]
            columns.extend(delta[lag - j:-j] for j in range(1, lag + 1))
            x = np.column_stack(columns)
            width = x.shape[1]
            xtx = np.asarray([
                [sum(float(row[i]) * float(row[j]) for row in x)
                 for j in range(width)]
                for i in range(width)
            ])
            inverse = _gauss_jordan_inverse(xtx)
            if inverse is None:
                continue
            xty = [sum(float(row[i]) * float(value) for row, value in zip(x, y))
                   for i in range(width)]
            beta = np.asarray([
                sum(float(inverse[i, j]) * xty[j] for j in range(width))
                for i in range(width)
            ])
            residual = [float(value) - sum(float(row[j]) * float(beta[j])
                                           for j in range(width))
                        for row, value in zip(x, y)]
            rss = sum(value * value for value in residual)
            dof = len(y) - x.shape[1]
            if dof <= 0 or rss <= 1e-18:
                continue
            variance = rss / dof
            se = float(np.sqrt(max(variance * inverse[1, 1], 0.0)))
            if se <= 1e-15:
                continue
            statistic = float(beta[1] / se)
            aic = len(y) * np.log(rss / len(y)) + 2 * x.shape[1]
            if best is None or aic < best[0]:
                best = (float(aic), statistic)
        if best is None:
            return None
        return float(mackinnonp(best[1], regression="c", N=1))
    except Exception:  # noqa: BLE001 - a missing/unhappy statsmodels must not block
        return None


def _gauss_jordan_inverse(matrix: np.ndarray) -> np.ndarray | None:
    """Inverse of a small dense matrix without entering platform BLAS/LAPACK."""
    a = np.asarray(matrix, dtype=np.float64)
    n = len(a)
    augmented = [list(map(float, a[row])) + [float(row == col) for col in range(n)]
                 for row in range(n)]
    for col in range(n):
        pivot = max(range(col, n), key=lambda row: abs(augmented[row][col]))
        if abs(augmented[pivot][col]) < 1e-12:
            return None
        augmented[col], augmented[pivot] = augmented[pivot], augmented[col]
        scale = augmented[col][col]
        augmented[col] = [value / scale for value in augmented[col]]
        for row in range(n):
            if row == col:
                continue
            factor = augmented[row][col]
            if factor:
                augmented[row] = [left - factor * right
                                  for left, right in zip(augmented[row], augmented[col])]
    return np.asarray([row[n:] for row in augmented], dtype=np.float64)


def _continuity(values: np.ndarray) -> float:
    """1 means no step change; 0 means the largest jump equals the whole range."""
    spread = float(np.max(values) - np.min(values))
    if spread < 1e-9:
        return 1.0
    return float(1.0 - np.max(np.abs(np.diff(values))) / spread)


def refine(
    graph,
    roi: Roi,
    probability,
    bridges,
    *,
    params: DpcParams | None = None,
    keep_rejected: bool = True,
) -> list[Bridge]:
    """Re-walk each geometric proposal through the image, and accept or reject it.

    This is the intended division of labour: the geometric proposers are cheap and
    generate candidate pairs, and the walk -- which has to read the image -- only
    runs on pairs that already passed the cone, radius and tortuosity gates.
    """
    p = params or DpcParams()
    out: list[Bridge] = []

    for bridge in bridges:
        if not bridge.accepted and not keep_rejected:
            continue
        if not bridge.accepted:
            out.append(bridge)
            continue

        source = bridge.source_node
        tangent = endpoint_tangent(graph, source)
        direction = tangent[0] if tangent is not None else None
        start = np.asarray(graph.nodes[source][:3], dtype=np.float64)

        if bridge.kind == "tjunction" and bridge.target_segment is not None:
            target = graph.coords(bridge.target_segment)
            polyline = True
        else:
            target = np.asarray(bridge.coords[-1], dtype=np.float64).reshape(1, 3)
            polyline = False

        reconnect_type = bridge.reconnection_type or (3 if polyline else 1)
        reach_factor = {
            1: p.type1_reach, 2: p.type2_reach, 3: p.type3_reach,
        }[reconnect_type]
        source_radius = float(bridge.metrics.get("r_source", 0.0))
        if source_radius > 0 and bridge.span_um > reach_factor * source_radius:
            out.append(bridge.reject(
                f"DPC: Type {reconnect_type} target exceeds {reach_factor:g} x radius"
            ))
            continue

        result = walk(
            roi, probability, start, target,
            start_direction=direction, target_is_polyline=polyline,
            reconnection_type=reconnect_type, params=p,
        )
        bridge.metrics["dpc_steps"] = result.steps
        bridge.metrics["dpc_reason"] = result.reason
        bridge.metrics["dpc_distance_scores"] = result.distance_scores.tolist()
        bridge.metrics["dpc_probability_scores"] = result.probability_scores.tolist()
        bridge.metrics["dpc_cosine_scores"] = result.cosine_scores.tolist()

        reference = _reference_probability(graph, roi, probability, source)
        probability_sequence, grayscale_sequence = _evidence_sequences(
            graph, bridge, roi, probability, result
        )
        result.probability_sequence = probability_sequence
        result.grayscale_sequence = grayscale_sequence
        ok, reason, stats = validate(
            result, reference, grey_along_path=grayscale_sequence, params=p
        )
        bridge.metrics["dpc_probability_sequence"] = probability_sequence.tolist()
        bridge.metrics["dpc_grayscale_sequence"] = grayscale_sequence.tolist()
        bridge.metrics.update(stats)
        if not ok:
            out.append(bridge.reject(f"DPC: {reason}"))
            continue

        # The walk's own path replaces the interpolated one: it is what the image
        # says the vessel does, rather than what a spline assumed.
        spacing = max(0.9 * bridge.metrics.get("r_source", 1.0), 1.0)
        coords = resample_by_arclength(result.path_um, spacing)
        bridge.coords = coords
        r0 = bridge.metrics.get("r_source", 1.0)
        r1 = bridge.metrics.get("r_target", r0)
        bridge.radii = np.linspace(r0, r1, len(coords))
        bridge.score = float(np.clip(stats.get("mean_probability", 0.0), 0.0, 1.0))
        bridge.reason = "accepted by the DPC walk"
        out.append(bridge)

    return out


def _evidence_sequences(graph, bridge, roi: Roi, probability, result: DpcResult,
                        n: int = 5) -> tuple[np.ndarray, np.ndarray]:
    """Paper-style known tail + walked gap + known target-head evidence."""
    source = _near_node_points(graph, bridge.source_node, n)
    target = _near_target_points(graph, bridge, n)
    walked = np.asarray(result.path_um, dtype=np.float64).reshape(-1, 3)
    blocks = [a for a in (source, walked, target) if len(a)]
    points = np.vstack(blocks) if blocks else np.empty((0, 3))
    if not len(points):
        return np.array([]), np.array([])
    prob = np.asarray(probability(points), dtype=np.float64)
    from .probability import sample_trilinear

    grey = sample_trilinear(roi.volume, roi.to_index(points))
    return prob, grey


def _near_node_points(graph, node: int, n: int) -> np.ndarray:
    segs = graph.node_segments(node)
    if not segs:
        return np.empty((0, 3))
    sid = next(iter(segs))
    coords = np.asarray(graph.coords(sid), dtype=np.float64)
    at_start = graph.segment(sid)["node1"] == node
    near = coords[:n] if at_start else coords[-n:][::-1]
    return near[::-1]  # far -> join


def _near_target_points(graph, bridge, n: int) -> np.ndarray:
    if bridge.target_node is not None:
        near = _near_node_points(graph, bridge.target_node, n)
        return near[::-1]  # join -> far
    if bridge.target_segment is None:
        return np.empty((0, 3))
    coords = np.asarray(graph.coords(bridge.target_segment), dtype=np.float64)
    if not len(coords):
        return coords
    k = int(np.clip(bridge.target_index or 0, 0, len(coords) - 1))
    lo, hi = max(0, k - n // 2), min(len(coords), k + n // 2 + 1)
    return coords[lo:hi]


def _reference_probability(graph, roi: Roi, probability, node: int, n: int = 40
                           ) -> np.ndarray:
    """Probabilities along the vessel the walk is continuing, as its own baseline.

    Comparing against the source vessel rather than a global threshold is what
    makes the test work on both a trunk and a distal twig, whose absolute
    probabilities are nowhere near each other.
    """
    segs = graph.node_segments(node)
    if not segs:
        return np.array([])
    coords = graph.coords(next(iter(segs)))
    if not len(coords):
        return np.array([])
    at_start = graph.segment(next(iter(segs)))["node1"] == node
    near = coords[:n] if at_start else coords[-n:]
    idx = roi.to_index(near)
    near = near[roi.inside(idx)]
    if not len(near):
        return np.array([])
    return np.asarray(probability(near), dtype=np.float64)
