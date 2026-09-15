"""Choose which routes to keep, globally rather than one at a time.

Each candidate is scored on its own evidence, and taking every candidate that
scores well is wrong in three specific ways -- none of which is visible from
inside a single candidate:

**A free end has one continuation.** Two routes arriving at the same endpoint
cannot both be right, and accepting both makes that endpoint a junction the
anatomy does not have. The greedy pass in :mod:`..endpoints` already enforces
this within one proposer, but a geodesic route, a T-junction and a fragment chain
are proposed by different code and only meet here.

**A tree has no cycles.** Two fragments of one vessel can be joined at both ends,
which closes a loop; the coronary tree has none, so the second join is an artefact
even when its evidence is excellent. Union-find over the *current* component
assignment catches it, and the assignment has to be updated as routes are accepted
rather than computed once -- a cycle is usually only closed by the third or fourth
acceptance.

**Two daughters are not one daughter twice.** Two free ends attaching to the same
few voxels of a parent are almost always one branch found twice, so attachments on
one vessel are required to be a vessel-width apart. Genuinely adjacent daughters
survive that; a duplicate does not.

Kruskal's algorithm, in short: sort by confidence, accept what does not conflict.
It is a maximum-confidence spanning forest, and "forest" rather than "tree" is the
important half -- a piece of tree that nothing supports connecting stays
disconnected instead of being attached to whatever was nearest.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

#: Two attachments on one parent vessel must be this many source radii apart.
ATTACHMENT_SEPARATION_RADII = 2.0


@dataclass
class Decision:
    """What was decided about one candidate, and why."""

    candidate: object
    accepted: bool
    status: str  # "accept" | "review" | "reject"
    reason: str
    confidence: float = 0.0
    rank: int = 0
    conflicts: list = field(default_factory=list)

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"<Decision {self.status} conf={self.confidence:.3f} {self.reason}>"


class _Forest:
    """Union-find over graph component ids, updated as routes are accepted."""

    def __init__(self, component_of_node: dict[int, int]):
        self.component_of_node = dict(component_of_node)
        roots = set(self.component_of_node.values())
        self.parent = {r: r for r in roots}

    def _find(self, x: int) -> int:
        if x not in self.parent:
            self.parent[x] = x
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def component(self, node: int):
        # A node the graph does not know -- the sentinel a T-junction attachment
        # or a mask end uses -- gets a namespaced key of its own rather than an
        # integer derived from its id. `-node - 1` collided: node -5 mapped to 4,
        # which is also a perfectly ordinary component index.
        if node in self.component_of_node:
            return self._find(self.component_of_node[node])
        return self._find(("node", int(node)))

    def would_cycle(self, a: int, b: int) -> bool:
        return self.component(a) == self.component(b)

    def join(self, a: int, b: int) -> None:
        ra, rb = self.component(a), self.component(b)
        if ra != rb:
            self.parent[rb] = ra


def component_of_node(graph) -> dict[int, int]:
    """``{node id: component index}``, matching :func:`..endpoints._component_of_node`."""
    out: dict[int, int] = {}
    for i, seg_ids in enumerate(graph.components()):
        for sid in seg_ids:
            seg = graph.segment(sid)
            out[seg["node1"]] = i
            out[seg["node2"]] = i
    return out


def select(graph, scored, *, allow_cycles: bool = False,
           separation_radii: float = ATTACHMENT_SEPARATION_RADII) -> list[Decision]:
    """Pick a maximum-confidence forest from scored candidates.

    `scored` is a sequence of ``(candidate, confidence, status, reason)`` where
    `status` is what the per-candidate evidence concluded -- ``"accept"``,
    ``"review"`` or ``"reject"``. Only ``accept`` competes for the forest;
    ``review`` and ``reject`` pass through unchanged, because a candidate the
    operator has not yet ruled on must not silently consume the endpoint that a
    confidently accepted one needs.

    Returns one :class:`Decision` per input, in the input order, so a caller can
    zip it back against its own candidate list.
    """
    forest = _Forest(component_of_node(graph))
    used_ends: dict[int, int] = {}  # node -> index of the route that claimed it
    # Mask free ends are claimed the same way and for the same reason, but they
    # have no node id to be claimed *by*: their identity is the tip voxel, which
    # is why :class:`..lobes.MaskEnd` carries one.
    used_mask_ends: dict[tuple, int] = {}
    attachments: dict[int, list[tuple[np.ndarray, float]]] = {}
    order = sorted(
        range(len(scored)), key=lambda i: -float(scored[i][1])
    )
    decisions: list[Decision | None] = [None] * len(scored)

    for rank, i in enumerate(order, 1):
        candidate, confidence, status, reason = scored[i]
        if status != "accept":
            decisions[i] = Decision(candidate, False, status, reason,
                                    float(confidence), rank)
            continue

        source = _source_node(candidate)
        target = _target_node(candidate)
        mask_end = _mask_end_key(candidate)
        anchor = _component_anchor(candidate, graph)
        conflicts: list[str] = []

        if source in used_ends:
            conflicts.append(
                f"free end {source} is already continued by a better-scoring route"
            )
        if target is not None and target in used_ends:
            conflicts.append(
                f"free end {target} is already continued by a better-scoring route"
            )
        if mask_end is not None and mask_end in used_mask_ends:
            conflicts.append(
                f"mask free end {mask_end} is already claimed by a better-scoring "
                f"route"
            )
        # Cycle detection runs on the *component anchor*, not on the target end.
        # A T-junction has no target endpoint to consume, but it joins two
        # components just as firmly as an end-to-end route does -- and two
        # fragments that each attach to the other's side close a loop that
        # checking only endpoints would never see.
        if not conflicts and anchor is not None and not allow_cycles:
            if forest.would_cycle(source, anchor):
                conflicts.append(
                    "the two sides are already connected; this route would close a loop"
                )
        if not conflicts:
            clash = _attachment_clash(candidate, attachments, separation_radii)
            if clash:
                conflicts.append(clash)

        if conflicts:
            decisions[i] = Decision(candidate, False, "reject", conflicts[0],
                                    float(confidence), rank, conflicts)
            continue

        used_ends[source] = i
        if mask_end is not None:
            used_mask_ends[mask_end] = i
        if target is not None:
            used_ends[target] = i
        else:
            _record_attachment(candidate, attachments)
        if anchor is not None:
            forest.join(source, anchor)
        decisions[i] = Decision(candidate, True, "accept", reason,
                                float(confidence), rank)

    return [d for d in decisions if d is not None]


def _source_node(candidate) -> int:
    classified = getattr(candidate, "classified", candidate)
    return int(classified.source.node)


def _target_node(candidate):
    """The target free end, or ``None`` for an attachment onto a vessel's side."""
    classified = getattr(candidate, "classified", candidate)
    if classified.target_segment is not None:
        return None
    target = classified.target
    if target is None or target.node < 0:
        return None
    return int(target.node)


def _component_anchor(candidate, graph):
    """A node on the far side of the route, for the cycle test only.

    For an end-to-end join that is the target endpoint. For a T-junction it is
    *any* node of the parent segment, because every node of a segment is in the
    same component and the cycle test only ever asks which component. Returning
    the endpoint would be wrong here and returning ``None`` would be worse -- it
    is what let two mutually-attaching fragments through.

    For a mask end it is the node the lobe hangs off. A lobe is *unskeletonised*,
    not *unattached*: if its lumen runs back into vessel V, then joining it joins
    V's component, and the cycle test has to know that or it will happily close a
    loop through material that has no centreline on it yet. A lobe with nothing
    described within reach genuinely is free-floating, and ``None`` is then the
    right answer rather than a gap in the test.
    """
    classified = getattr(candidate, "classified", candidate)
    mask_end = getattr(classified, "target_mask_end", None)
    if mask_end is not None:
        return None if mask_end.attach_node is None else int(mask_end.attach_node)
    if classified.target_segment is None:
        return _target_node(candidate)
    if graph is None or not graph.has_segment(classified.target_segment):
        return None
    return int(graph.segment(classified.target_segment)["node1"])


def _mask_end_key(candidate):
    """The identity of the mask free end this route claims, if it claims one."""
    classified = getattr(candidate, "classified", candidate)
    mask_end = getattr(classified, "target_mask_end", None)
    return None if mask_end is None else tuple(mask_end.key)


def _attachment_clash(candidate, attachments, separation_radii: float) -> str:
    """Is this T-junction landing on top of one already accepted?"""
    classified = getattr(candidate, "classified", candidate)
    sid = classified.target_segment
    if sid is None:
        return ""
    point = _attachment_point(candidate)
    if point is None:
        return ""
    radius = float(classified.source.radius_um)
    for other, other_radius in attachments.get(int(sid), []):
        separation = float(np.linalg.norm(point - other))
        if separation < separation_radii * max(radius, other_radius):
            return (f"another branch already attaches {separation:.0f} um away on "
                    f"segment {sid}")
    return ""


def _record_attachment(candidate, attachments) -> None:
    classified = getattr(candidate, "classified", candidate)
    sid = classified.target_segment
    point = _attachment_point(candidate)
    if sid is None or point is None:
        return
    attachments.setdefault(int(sid), []).append(
        (point, float(classified.source.radius_um))
    )


def _attachment_point(candidate):
    """Where on the parent vessel this branch would land, in world um."""
    classified = getattr(candidate, "classified", candidate)
    if classified.target is None:
        return None
    return np.asarray(classified.target.point_um, dtype=np.float64)


def summarise(decisions) -> str:
    """What the global pass changed, which is the part per-candidate output hides."""
    counts: dict[str, int] = {}
    for decision in decisions:
        counts[decision.status] = counts.get(decision.status, 0) + 1
    lines = [
        f"{counts.get('accept', 0)} accepted, {counts.get('review', 0)} for review, "
        f"{counts.get('reject', 0)} rejected"
    ]
    for decision in decisions:
        if decision.status == "reject" and decision.conflicts:
            lines.append(f"  conflict: {decision.reason}")
    return "\n".join(lines)
