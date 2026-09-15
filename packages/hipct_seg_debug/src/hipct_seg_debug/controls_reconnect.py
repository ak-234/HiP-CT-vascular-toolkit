"""The Reconnect tab: adjudicating the routes the connector would not decide alone.

Everything else in this package produces a file. This one produces a *decision*,
and that is a different design problem: the operator is not being asked to run a
command, they are being asked to look at one break at a time and say whether the
route the search found is the vessel or a shortcut through myocardium.

So the panel is a work list, not a form. It loads a review file written by
``connect --geodesic --review-json``, walks it one candidate at a time, and for
each one shows the three things that actually decide the question:

* **which mask components** the two ends are on, because "these are already one
  component" and "these are two components 400 um apart" are different repairs and
  the difference is invisible in the 3D view;
* **the evidence**, in the units the cost field was calibrated in -- support
  relative to the intact vessel either side, the longest unsupported run, and how
  far behind the runner-up route was;
* **the alternatives**, drawn alongside the winner. A route that looks perfectly
  reasonable on its own very often has a second, equally reasonable one next to
  it, and seeing both is the whole reason ambiguous candidates come here instead
  of being resolved by a threshold.

Waypoints are the one editing gesture. Clicking in the 3D view or on a slice adds
an ordered point the route must pass through, and the route is recomputed through
them -- deterministically, so the same waypoints always give the same answer. That
is what makes a correction reviewable rather than a nudge.

**Nothing here writes to the dataset.** Accept and reject record a ruling into the
decisions document; applying them is a separate ``connect --geodesic`` run with
``--decisions-json``, which is the same explicit-apply division the rest of the
package uses. The panel can therefore be used on a machine that has the review
file but not the 2.34 GB of segmentation behind it.
"""

from __future__ import annotations

import json
from pathlib import Path

#: Evidence rows, as ``(key, label, formatter)``. Ordered by how often the answer
#: turns on them rather than by where they sit in the record.
EVIDENCE_ROWS = (
    ("mean_support", "support vs intact vessel", "{:.2f}"),
    ("min_support", "worst point on the route", "{:.2f}"),
    ("unsupported_um", "longest unsupported run", "{:.0f} um"),
    ("unsupported_allowance_um", "...allowed", "{:.0f} um"),
    ("length_um", "route length", "{:.0f} um"),
    ("span_um", "straight-line gap", "{:.0f} um"),
    ("mask_gap_voxels", "mask gap", "{:.0f} voxel(s)"),
    ("alternative_margin", "next-best route is worse by", "{:.0%}"),
    ("route_cost", "route cost", "{:.2f}"),
)

ACTOR_PREFIX = "reconnect-review"


def load_review(path) -> dict:
    """Read a review or decisions document, refusing one that is not ours."""
    document = json.loads(Path(path).read_text(encoding="utf-8"))
    schema = document.get("schema", "")
    if not schema.startswith("hipct.geodesic-reconnect/"):
        raise ValueError(f"{path}: not a geodesic reconnection document ({schema!r})")
    return document


def final_status(record) -> str:
    """What was actually decided, not what this candidate scored on its own.

    A candidate carries two verdicts and they disagree by design. ``status`` is what its
    own evidence concluded; ``decision.status`` is what the global forest pass concluded
    once it could see the endpoint this candidate was competing for. On the LADAF-28 run
    two routes scored ``accept`` at 0.51 and 0.55 and were then rejected -- one because a
    better route had already claimed its free end, one because it would have closed a
    loop. Showing the first number would tell a reviewer 13 joins were made where 11
    were, which is the one thing this panel exists not to do.
    """
    decided = (record.get("decision") or {}).get("status")
    return decided or record.get("status") or "?"


def decision_note(record) -> str:
    """The selector's reason, when it overruled the candidate's own verdict."""
    decision = record.get("decision") or {}
    if not decision.get("status") or decision["status"] == record.get("status"):
        return ""
    return decision.get("reason") or ""


def describe_candidate(record) -> str:
    """The one-line identity of a candidate, for the work list."""
    source = record.get("source", {})
    target = record.get("target") or {}
    mask_end = record.get("mask_end") or {}
    if record.get("target_segment") is not None:
        where = f"seg {record['target_segment']}"
    elif mask_end:
        # Naming it "node -1" would be true and useless. A mask end has no node --
        # that is the entire reason it exists -- so it is named by the thing it
        # does have, which is a tip voxel.
        where = "mask end " + "/".join(str(v) for v in mask_end.get("key", []))
    else:
        where = f"node {target.get('node', '?')}"
    return (f"[{final_status(record)}] {record.get('kind', '?')}: "
            f"node {source.get('node', '?')} -> {where}"
            f"  (components {source.get('component', '?')} -> "
            f"{target.get('component', '?')}, conf "
            f"{record.get('confidence', 0.0):.2f})")


def evidence_lines(record) -> list[str]:
    """The evidence table for one candidate, skipping what was not measured.

    A missing row means "not measured", which is different from zero and is left
    out rather than printed as 0 -- an unsupported run of 0 um is a strong
    statement about a route and it must not be confusable with never having
    looked.
    """
    evidence = record.get("evidence") or {}
    lines = []
    for key, label, fmt in EVIDENCE_ROWS:
        value = evidence.get(key)
        if value is None:
            continue
        try:
            lines.append(f"{label:<28} {fmt.format(value)}")
        except (TypeError, ValueError):
            lines.append(f"{label:<28} {value}")
    if evidence.get("has_raw") is False:
        lines.append("no raw greyscale: this route is geometry alone")
    if evidence.get("contrast") is False:
        lines.append("no lumen/wall contrast here: the intensity terms said nothing")
    competing = evidence.get("competing_components") or {}
    if competing:
        rivals = ", ".join(f"#{k} ({v} voxels)" for k, v in list(competing.items())[:3])
        lines.append(f"competing components nearby: {rivals}")
    mask_end = record.get("mask_end") or {}
    if mask_end:
        # The reviewer is being asked to approve a repair into material with no
        # centreline on it. What made anyone think a vessel was there belongs on
        # the screen next to the question, not in the audit file only.
        lines.append(
            f"{'target has no centreline':<28} "
            f"{mask_end.get('lobe_voxels', 0)} voxel(s) of undescribed lumen, "
            f"{mask_end.get('elongation', 0.0):.1f}:1"
        )
        attach = mask_end.get("attach_node")
        lines.append(
            f"{'...hanging off':<28} "
            + (f"node {attach}" if attach is not None
               else "nothing described within reach")
        )
    return lines


def route_points(record, which: int = -1):
    """World-micrometre points of the best route, or of alternative `which`.

    ``which`` of -1 means the accepted route; 0, 1, ... index the alternatives.
    Returns ``None`` when that route was not recorded, so a caller can tell "no
    alternative" from "an alternative of zero length".
    """
    route = record.get("route") if which < 0 else \
        (record.get("alternatives") or [None] * (which + 1))[which]
    if not route:
        return None
    points = route.get("path_um")
    return points if points else None


def rule(document, record, *, accept: bool, reason: str = "", waypoints=None) -> dict:
    """Record an operator's ruling into the document, in place.

    Written under ``decision.operator`` rather than overwriting ``decision``: the
    automatic verdict is evidence about the tool and has to survive being
    disagreed with, or a later audit cannot tell a route the operator rescued from
    one the tool accepted on its own.
    """
    for candidate in document.get("candidates", []):
        if candidate is record:
            break
    else:
        document.setdefault("candidates", []).append(record)
    ruling = {"accept": bool(accept)}
    if reason:
        ruling["reason"] = reason
    if waypoints:
        ruling["waypoints_um"] = [list(map(float, w)) for w in waypoints]
    record.setdefault("decision", {})["operator"] = ruling
    record["waypoints_um"] = ruling.get("waypoints_um", record.get("waypoints_um", []))
    return document


def pending(document) -> list:
    """Candidates still awaiting a ruling, in the order they should be shown."""
    out = []
    for record in document.get("candidates", []):
        if (record.get("decision") or {}).get("operator") is not None:
            continue
        if final_status(record) == "review" or record.get("kind") == "unassociated":
            out.append(record)
    # Lowest confidence first: those are the ones where a person adds the most,
    # and a reviewer who runs out of time should have spent it on them.
    return sorted(out, key=lambda r: float(r.get("confidence", 0.0)))


def save(document, path) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(document, indent=2), encoding="utf-8")
    return destination


def summary(document) -> str:
    counts: dict[str, int] = {}
    ruled = 0
    for record in document.get("candidates", []):
        status = final_status(record)
        counts[status] = counts.get(status, 0) + 1
        if (record.get("decision") or {}).get("operator") is not None:
            ruled += 1
    total = sum(counts.values())
    parts = ", ".join(f"{n} {k}" for k, n in sorted(counts.items()))
    return f"{total} candidate(s): {parts}; {ruled} ruled, {len(pending(document))} to go"


# --------------------------------------------------------------------- the widget


def build_reconnect_panel(app):
    """Return the Reconnect widget for a `ViewerApp`.

    Built from the same duck-typed surface as every other panel, so it can be
    constructed and driven in a test without a render window -- which matters more
    here than elsewhere, because the interesting behaviour is the decision
    bookkeeping rather than the drawing.
    """
    from qtpy.QtWidgets import (
        QFileDialog,
        QHBoxLayout,
        QLabel,
        QListWidget,
        QPlainTextEdit,
        QPushButton,
        QVBoxLayout,
        QWidget,
    )

    box = QWidget()
    lay = QVBoxLayout(box)
    lay.setContentsMargins(8, 8, 8, 8)

    state: dict = {"document": None, "path": None, "records": [], "waypoints": []}

    header = QLabel("no review file loaded")
    header.setWordWrap(True)
    lay.addWidget(header)

    row = QHBoxLayout()
    open_button = QPushButton("Open review...")
    save_button = QPushButton("Save decisions")
    save_button.setEnabled(False)
    row.addWidget(open_button)
    row.addWidget(save_button)
    lay.addLayout(row)

    listing = QListWidget()
    lay.addWidget(listing, 1)

    detail = QPlainTextEdit()
    detail.setReadOnly(True)
    lay.addWidget(detail, 1)

    actions = QHBoxLayout()
    accept_button = QPushButton("Accept")
    reject_button = QPushButton("Reject")
    waypoint_button = QPushButton("Add waypoint")
    clear_button = QPushButton("Clear waypoints")
    undo_button = QPushButton("Undo ruling")
    for button in (accept_button, reject_button, waypoint_button, clear_button,
                   undo_button):
        button.setEnabled(False)
        actions.addWidget(button)
    lay.addLayout(actions)

    # -- helpers ----------------------------------------------------------

    def status(message: str) -> None:
        if getattr(app, "on_status", None):
            app.on_status(message)

    def current_pick():
        """The viewer's last picked world point, or ``None`` if there is not one.

        ``Picker3D.picked`` is where both the 3D double-click and the slice pick
        land, so this is one source rather than two. A panel driven without a
        viewer -- a test, or a review session on a machine without the volume --
        simply has no pick, which is a message rather than an error.
        """
        picker = getattr(app, "picker", None)
        pick = getattr(picker, "picked", None) if picker is not None else None
        if pick is None:
            pick = getattr(app, "last_pick_um", None)
        return pick

    def current():
        index = listing.currentRow()
        if 0 <= index < len(state["records"]):
            return state["records"][index]
        return None

    def refresh_list() -> None:
        document = state["document"]
        listing.clear()
        state["records"] = []
        if not document:
            header.setText("no review file loaded")
            return
        for record in document.get("candidates", []):
            state["records"].append(record)
            ruling = (record.get("decision") or {}).get("operator")
            mark = "" if ruling is None else ("[accepted] " if ruling.get("accept")
                                              else "[rejected] ")
            listing.addItem(mark + describe_candidate(record))
        header.setText(summary(document))
        save_button.setEnabled(True)

    def draw(record) -> None:
        """Ask the viewer to show this candidate's routes, if there is a viewer.

        The drawing lives on ``Picker3D`` because that is what owns the actors and
        knows how to drop them on a dataset swap; it is looked up there first and
        on the app second, so a headless review session -- or a test -- simply has
        nothing to draw on.
        """
        show = getattr(getattr(app, "picker", None), "show_reconnect_candidate", None)
        if show is None:
            show = getattr(app, "show_reconnect_candidate", None)
        if show is None:
            return
        try:
            show(record, waypoints=state["waypoints"], actor_prefix=ACTOR_PREFIX)
        except Exception as exc:  # noqa: BLE001 - a drawing failure must not
            status(f"  could not draw the candidate: {exc}")  # block the decision

    def show_current(*, reload_waypoints: bool = False) -> None:
        """Redraw the detail pane for the selected candidate.

        `reload_waypoints` separates two things that look alike and are not:
        *selecting* a candidate adopts whatever waypoints it was saved with, while
        *redrawing* the one already selected must keep the ones just placed.
        Conflating them silently discards every waypoint at the moment it is
        added, which looks exactly like the click not registering.
        """
        record = current()
        enabled = record is not None
        for button in (accept_button, reject_button, waypoint_button, clear_button,
                       undo_button):
            button.setEnabled(enabled)
        if record is None:
            detail.setPlainText("")
            return
        if reload_waypoints:
            state["waypoints"] = list(record.get("waypoints_um") or [])
        lines = [describe_candidate(record), "", record.get("reason", ""), ""]
        overruled = decision_note(record)
        if overruled:
            lines.insert(3, f"OVERRULED by the global pass: {overruled}")
        lines.extend(evidence_lines(record))
        alternatives = record.get("alternatives") or []
        lines.append("")
        lines.append(f"{len(alternatives)} alternative route(s) found")
        if record.get("classification_reason"):
            lines.extend(["", record["classification_reason"]])
        fragments = record.get("fragments") or []
        for fragment in fragments:
            verdict = "plausible" if fragment.get("plausible") else fragment.get("reason")
            lines.append(f"fragment #{fragment.get('component')}: "
                         f"{fragment.get('voxels')} voxels, {verdict}")
        if state["waypoints"]:
            lines.extend(["", f"{len(state['waypoints'])} waypoint(s) placed"])
        detail.setPlainText("\n".join(lines))
        draw(record)

    # -- actions ----------------------------------------------------------

    def on_open() -> None:
        path, _filter = QFileDialog.getOpenFileName(
            box, "Open a reconnection review", "", "JSON (*.json)"
        )
        if not path:
            return
        load(path)

    def load(path) -> None:
        try:
            state["document"] = load_review(path)
        except Exception as exc:  # noqa: BLE001 - a bad file is a message, not a crash
            status(f"  {exc}")
            return
        state["path"] = str(path)
        state["waypoints"] = []
        refresh_list()
        if state["records"]:
            listing.setCurrentRow(0)
        status(f"  loaded {Path(path).name}: {summary(state['document'])}")

    def on_rule(accept: bool) -> None:
        record = current()
        if record is None:
            return
        rule(state["document"], record, accept=accept,
             waypoints=state["waypoints"])
        row_index = listing.currentRow()
        refresh_list()
        listing.setCurrentRow(min(row_index + 1, len(state["records"]) - 1))
        status(f"  {'accepted' if accept else 'rejected'} "
               f"{describe_candidate(record)}")

    def on_undo() -> None:
        record = current()
        if record is None:
            return
        (record.get("decision") or {}).pop("operator", None)
        row_index = listing.currentRow()
        refresh_list()
        listing.setCurrentRow(row_index)
        status("  ruling withdrawn")

    def on_waypoint() -> None:
        """Take the viewer's current pick as the next ordered waypoint.

        Read from the picker rather than from a separate channel, so a waypoint is
        placed by the same gesture that already selects a point in the 3D view or
        on an orthogonal slice -- ``Picker3D.picked`` is the world-micrometre
        result of both.
        """
        pick = current_pick()
        if pick is None:
            status("  pick a point in the 3D view or on a slice first")
            return
        state["waypoints"].append([float(v) for v in pick])
        show_current()

    def on_clear() -> None:
        state["waypoints"] = []
        show_current()

    def on_save() -> None:
        if not state["document"]:
            return
        path = state["path"]
        if not path:
            path, _f = QFileDialog.getSaveFileName(box, "Save decisions", "",
                                                   "JSON (*.json)")
            if not path:
                return
        save(state["document"], path)
        status(f"  wrote {path}. Re-run connect --geodesic --decisions-json to apply "
               f"it.")

    open_button.clicked.connect(on_open)
    save_button.clicked.connect(on_save)
    listing.currentRowChanged.connect(
        lambda _row: show_current(reload_waypoints=True)
    )
    accept_button.clicked.connect(lambda: on_rule(True))
    reject_button.clicked.connect(lambda: on_rule(False))
    waypoint_button.clicked.connect(on_waypoint)
    clear_button.clicked.connect(on_clear)
    undo_button.clicked.connect(on_undo)

    def refresh() -> None:
        """Survive the dataset going away, like every other panel.

        The review document is independent of the loaded dataset -- it is a file of
        decisions, not a view onto the volume -- so a failed load leaves the work
        list exactly where it was rather than clearing it.
        """
        if state["document"]:
            header.setText(summary(state["document"]))

    box.refresh = refresh
    box.load = load
    box.state = state
    box.rule_current = on_rule
    box.add_waypoint = on_waypoint
    box.undo_ruling = on_undo
    box.save_decisions = on_save
    box.widgets = {
        "list": listing, "detail": detail, "header": header,
        "accept": accept_button, "reject": reject_button, "undo": undo_button,
        "waypoint": waypoint_button, "clear": clear_button, "save": save_button,
    }
    return box
