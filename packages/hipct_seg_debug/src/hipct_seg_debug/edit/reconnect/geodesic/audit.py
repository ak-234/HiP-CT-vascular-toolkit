"""The record of what was decided, and the file an operator decides *with*.

Two documents, because they answer two questions and are read by two different
people at two different times.

**The review file** (``--review-json``) is a work list. It carries only the
candidates that were not resolved automatically, and it carries everything needed
to resolve one: the route, its alternatives, the evidence that made it equivocal,
the component ids on either side, and the reason. It is what the GUI panel loads
and what an operator's waypoints are written back into.

**The decisions file** (``--decisions-json``) is the record. Every candidate,
accepted or not, with its evidence, its provenance and the topology change it
caused. It is what makes a run reproducible and what an audit reads six months
later to ask why a particular vessel is connected.

Both are plain JSON with no numpy in them -- :func:`_plain` is not decoration.
A record that cannot be loaded by anything but this package is not an audit trail.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

SCHEMA = "hipct.geodesic-reconnect/1"


def _plain(value):
    """Recursively convert numpy scalars and arrays into JSON-native types.

    Non-finite floats become ``None``. JSON has no infinity, and `json.dumps` writes
    bare ``NaN`` / ``Infinity`` tokens that are not JSON at all and that a strict
    parser -- anything but Python's own -- refuses to load. ``None`` reads correctly
    as "not measured".

    That check has to sit on **plain** floats, not only on ``np.floating``: an array
    goes through ``tolist()``, which hands back Python floats, so a route with one
    non-finite coordinate in it would otherwise be written as an unloadable document
    by the very path this module exists to make loadable.
    """
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if isinstance(value, np.ndarray):
        return _plain(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        v = float(value)
        return v if np.isfinite(v) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    return value


def _route_record(route, frame, limit: int = 400) -> dict | None:
    """One route: its path in both index and world coordinates, plus its evidence.

    The path is decimated to `limit` points. A voxel-by-voxel route across a
    millimetre gap is a few thousand points, which turns a review file for a few
    dozen candidates into tens of megabytes and helps nobody -- the shape is what
    a reviewer looks at, and it survives decimation.
    """
    if route is None or not len(route.path_zyx):
        return None
    path = np.asarray(route.path_zyx, dtype=np.int64)
    if len(path) > limit:
        keep = np.unique(np.linspace(0, len(path) - 1, limit).astype(int))
        path = path[keep]
    record = {
        "cost": float(route.cost),
        "length_um": float(route.length_um),
        "mean_support": route.mean_support,
        "min_support": route.min_support,
        "n_points": int(len(route.path_zyx)),
        "path_zyx": path.tolist(),
        "expanded": int(route.expanded),
    }
    if frame is not None:
        record["path_um"] = np.asarray(
            frame.seg_to_um(path[:, ::-1]), dtype=np.float64
        ).round(3).tolist()
    return record


def _mask_end_record(mask_end) -> dict | None:
    """The manufactured endpoint a route aimed at, if it aimed at one.

    Carries the evidence that made it a candidate at all -- how much undescribed
    lumen it stands on and how tube-like that lumen is -- because a reviewer looking
    at a repair into material with no centreline is entitled to ask why anyone
    thought there was a vessel there.
    """
    if mask_end is None:
        return None
    return {
        "key": [int(v) for v in mask_end.key],
        "component": int(mask_end.component),
        "radius_um": float(mask_end.radius_um),
        "lobe_voxels": int(mask_end.lobe_voxels),
        "elongation": float(mask_end.elongation),
        "attach_node": (None if mask_end.attach_node is None
                        else int(mask_end.attach_node)),
        "point_um": np.asarray(mask_end.point_um, dtype=float).round(3).tolist(),
        "tangent": np.asarray(mask_end.tangent, dtype=float).round(4).tolist(),
        "note": mask_end.note,
    }


def candidate_record(candidate, frame=None, *, include_alternatives: bool = True) -> dict:
    """Everything known about one candidate, in a form a reviewer can act on."""
    classified = candidate.classified
    source, target = classified.source, classified.target
    record = {
        "kind": candidate.kind,
        "status": candidate.status,
        "reason": candidate.reason,
        "confidence": float(candidate.confidence),
        "source": {
            "node": int(source.node),
            "component": int(source.component),
            "radius_um": float(source.radius_um),
            "point_um": np.asarray(source.point_um, dtype=float).round(3).tolist(),
            "index_zyx": np.asarray(source.index_zyx, dtype=int).tolist(),
            "distance_vox": float(source.distance_vox)
            if np.isfinite(source.distance_vox) else None,
            "ambiguous": bool(source.ambiguous),
            "note": source.note,
        },
        "target": None if target is None else {
            "node": int(target.node),
            "component": int(target.component),
            "radius_um": float(target.radius_um),
            "point_um": np.asarray(target.point_um, dtype=float).round(3).tolist(),
            "index_zyx": np.asarray(target.index_zyx, dtype=int).tolist(),
            "ambiguous": bool(target.ambiguous),
            "note": target.note,
        },
        "target_segment": classified.target_segment,
        "target_index": classified.target_index,
        "mask_end": _mask_end_record(getattr(classified, "target_mask_end", None)),
        "classification_reason": classified.reason,
        "fragments": [
            {"component": int(f.component), "voxels": int(f.voxels),
             "elongation": float(f.elongation), "plausible": bool(f.plausible),
             "reason": f.reason,
             "centroid_um": np.asarray(f.centroid_um, dtype=float).round(3).tolist()}
            for f in classified.fragments
        ],
        "evidence": _plain(candidate.evidence),
        "waypoints_um": _plain(candidate.waypoints),
        "route": _route_record(candidate.route, frame),
    }
    if include_alternatives:
        record["alternatives"] = [
            _route_record(alt, frame) for alt in candidate.alternatives
        ]
    if candidate.completion is not None:
        completion = candidate.completion
        record["completion"] = {
            "voxels": int(len(completion.voxels_zyx)),
            "core_voxels": int(completion.core_voxels),
            "summary": completion.describe(),
            "radii_um": np.asarray(completion.radii_um, dtype=float).round(2).tolist(),
            "areas_um2": np.asarray(completion.areas_um2, dtype=float).round(2).tolist(),
            "metrics": _plain(completion.metrics),
        }
    return _plain(record)


def review_document(plan, frame=None, *, graph=None) -> dict:
    """The work list: every candidate an operator still has to rule on."""
    items = plan.for_review()
    return _plain({
        "schema": SCHEMA,
        "kind": "review",
        "written": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "counts": {
            "for_review": len(items),
            "accepted": len(plan.accepted()),
            "rejected": len(plan.rejected()),
        },
        "stats": plan.stats,
        "graph_components": None if graph is None else len(graph.components()),
        "candidates": [candidate_record(c, frame) for c in items],
    })


def decisions_document(plan, frame=None, *, graph=None, applied=None,
                       arguments=None) -> dict:
    """The record: every candidate, what was decided, and what it changed.

    ``topology`` is the part that cannot be reconstructed from the candidates
    alone -- how many graph components existed before and after, and how many mask
    voxels were added. A repair that reduced the component count by more than the
    number of routes it applied has merged something it did not mean to, and that
    is only visible from these two numbers together.
    """
    by_candidate = {id(a.candidate): a for a in (applied or ())}
    records = []
    for decision in plan.decisions:
        record = candidate_record(decision.candidate, frame)
        record["decision"] = {
            "status": decision.status,
            "accepted": bool(decision.accepted),
            "reason": decision.reason,
            "rank": int(decision.rank),
            "conflicts": list(decision.conflicts),
        }
        result = by_candidate.get(id(decision.candidate))
        if result is not None:
            record["applied"] = {
                "ok": bool(result.ok),
                "reason": result.reason,
                "origin": result.origin,
                "segments": [int(s) for s in result.segments],
                "voxels_added": int(result.voxels_added),
                "planes_touched": [int(p) for p in result.planes_touched],
                "summary": result.describe(),
            }
        records.append(record)

    topology = {
        "graph_components_after": None if graph is None else len(graph.components()),
        "voxels_added": sum(a.voxels_added for a in (applied or ())),
        "segments_added": sum(len(a.segments) for a in (applied or ())),
    }
    return _plain({
        "schema": SCHEMA,
        "kind": "decisions",
        "written": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "arguments": _plain(arguments or {}),
        "stats": plan.stats,
        "topology": topology,
        "candidates": records,
    })


def write(path, document) -> Path:
    """Write a document, creating its directory. Returns the path."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(document, indent=2), encoding="utf-8")
    return destination


def load_decisions(path) -> dict:
    """Read an operator's decisions back, and check it is one of ours.

    The schema check is not ceremony: applying a document written by a different
    version -- or a different tool entirely -- would silently accept or reject
    routes on the strength of fields that mean something else.
    """
    document = json.loads(Path(path).read_text(encoding="utf-8"))
    schema = document.get("schema")
    if schema != SCHEMA:
        raise ValueError(f"{path}: expected schema {SCHEMA!r}, found {schema!r}")
    return document


def apply_decisions(plan, document) -> tuple[list, list]:
    """Match an operator's rulings onto a fresh plan.

    Matched by ``(source node, target node or segment)`` rather than by position
    in the list, because a re-run on an edited graph will not produce the same
    candidates in the same order -- and a decision applied to the wrong candidate
    is worse than one not applied at all.

    Returns ``(approved, unmatched)``: the candidates an operator accepted, and
    the records that no longer correspond to anything, which is itself worth
    reporting because it usually means the graph moved on.
    """
    wanted: dict[tuple, dict] = {}
    for record in document.get("candidates", []):
        ruling = record.get("decision", {}).get("operator")
        if ruling is None:
            continue
        wanted[_key_of_record(record)] = record

    approved: list = []
    for candidate in plan.candidates:
        record = wanted.pop(_key_of_candidate(candidate), None)
        if record is None:
            continue
        ruling = record["decision"]["operator"]
        if ruling.get("accept"):
            waypoints = ruling.get("waypoints_um") or record.get("waypoints_um") or []
            candidate.waypoints = [np.asarray(w, dtype=np.float64) for w in waypoints]
            approved.append(candidate)
        else:
            candidate.status = "reject"
            candidate.reason = ruling.get("reason") or "rejected by the operator"
    return approved, list(wanted.values())


def _key_of_record(record) -> tuple:
    source = record.get("source", {}).get("node")
    target = (record.get("target") or {}).get("node")
    # A T-junction's attachment is an associated *point*, not a node, and is
    # serialised with the sentinel node -1. `Candidate.target_node` reports that
    # same case as None, so the two have to be normalised or no T-junction ruling
    # would ever match -- silently, and looking exactly like a stale document.
    if target is not None and int(target) < 0:
        target = None
    mask_end = record.get("mask_end") or {}
    return (_as_int(source), _as_int(target), _as_int(record.get("target_segment")),
            _mask_end_key(mask_end.get("key")))


def _key_of_candidate(candidate) -> tuple:
    mask_end = getattr(candidate.classified, "target_mask_end", None)
    return (_as_int(candidate.source_node), _as_int(candidate.target_node),
            _as_int(candidate.classified.target_segment),
            None if mask_end is None else _mask_end_key(mask_end.key))


def _mask_end_key(key):
    """A mask end's tip voxel, as a hashable tuple.

    It is part of the identity of the candidate and not an afterthought: two
    mask-end routes from one source both serialise with target node ``-1`` and no
    target segment, so without this they would share a key and an operator's ruling
    on one would silently be applied to the other.
    """
    return None if key is None else tuple(int(v) for v in key)


def _as_int(value):
    return None if value is None else int(value)
