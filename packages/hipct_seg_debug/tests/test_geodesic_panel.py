"""The Reconnect tab: the decision bookkeeping, and surviving a missing dataset.

The panel's job is not drawing -- it is making sure a ruling lands on the
candidate the operator was looking at, survives being written out and read back,
and can be withdrawn. Those are testable without a window, and they are the parts
whose failure would be invisible: a ruling silently attached to the wrong route
looks exactly like a ruling that worked.

The dataset-going-away case is here for the reason the rest of ``test_panels``
has it: that is the state after a failed load, and a panel that raised there would
take the window down at the moment it was needed to explain what went wrong. This
one has a further wrinkle -- a review document is a file of decisions rather than a
view onto the volume, so it must *keep* its work list when the dataset vanishes
rather than clearing it.
"""

from __future__ import annotations

import json
import os
from types import SimpleNamespace

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("qtpy")

from hipct_seg_debug import controls_reconnect  # noqa: E402
from hipct_seg_debug.edit.reconnect.geodesic import audit  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    from qtpy.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


def _record(node, target_node=None, target_segment=None, status="review",
            confidence=0.4, alternatives=1):
    return {
        "kind": "geodesic", "status": status, "reason": "a second route is close",
        "confidence": confidence,
        "source": {"node": node, "component": 1, "radius_um": 20.0,
                   "point_um": [0.0, 0.0, 0.0], "index_zyx": [0, 0, 0],
                   "ambiguous": False, "note": ""},
        "target": None if target_node is None else {
            "node": target_node, "component": 2, "radius_um": 20.0,
            "point_um": [100.0, 0.0, 0.0], "index_zyx": [0, 0, 10],
            "ambiguous": False, "note": ""},
        "target_segment": target_segment, "target_index": None,
        "classification_reason": "mask components 1 and 2 are distinct",
        "fragments": [],
        "evidence": {"mean_support": 0.62, "min_support": 0.21,
                     "unsupported_um": 0.0, "unsupported_allowance_um": 80.0,
                     "length_um": 120.0, "mask_gap_voxels": 3,
                     "alternative_margin": 0.04, "has_raw": True,
                     "contrast": True},
        "waypoints_um": [],
        "route": {"cost": 3.0, "length_um": 120.0, "mean_support": 0.62,
                  "min_support": 0.21, "n_points": 12,
                  "path_zyx": [[0, 0, i] for i in range(12)],
                  "path_um": [[10.0 * i, 0.0, 0.0] for i in range(12)],
                  "expanded": 40},
        "alternatives": [
            {"cost": 3.1, "length_um": 130.0, "mean_support": 0.6,
             "min_support": 0.2, "n_points": 13,
             "path_zyx": [[0, 1, i] for i in range(13)],
             "path_um": [[10.0 * i, 10.0, 0.0] for i in range(13)],
             "expanded": 60}
        ][:alternatives],
        "decision": {"status": status, "accepted": False, "reason": "",
                     "rank": 1, "conflicts": []},
    }


@pytest.fixture
def document():
    return {
        "schema": audit.SCHEMA, "kind": "review", "written": "2026-08-20T00:00:00+00:00",
        "counts": {"for_review": 2, "accepted": 0, "rejected": 0},
        "stats": {}, "graph_components": 3,
        "candidates": [_record(1, target_node=2, confidence=0.30),
                       _record(4, target_segment=7, confidence=0.44)],
    }


@pytest.fixture
def review_file(tmp_path, document):
    path = tmp_path / "review.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    return path


class FakeApp:
    """The surface the panel touches, and nothing else."""

    def __init__(self, loaded=True):
        self.session = SimpleNamespace() if loaded else None
        self.messages = []
        self.on_status = self.messages.append
        self.last_pick_um = None
        self.drawn = []

    @property
    def loaded(self):
        return self.session is not None

    def show_reconnect_candidate(self, record, *, waypoints, actor_prefix):
        self.drawn.append((record, list(waypoints), actor_prefix))


# ----------------------------------------------------------------- pure helpers


def test_a_foreign_document_is_refused(tmp_path):
    path = tmp_path / "other.json"
    path.write_text(json.dumps({"schema": "nope", "candidates": []}), encoding="utf-8")
    with pytest.raises(ValueError, match="not a geodesic reconnection document"):
        controls_reconnect.load_review(path)


def test_evidence_omits_what_was_never_measured():
    """A missing row means "not measured"; zero is a statement and must not blur."""
    record = _record(1, target_node=2)
    record["evidence"].pop("alternative_margin")
    lines = "\n".join(controls_reconnect.evidence_lines(record))
    assert "next-best route" not in lines
    assert "longest unsupported run" in lines  # 0.0 is kept, because 0 means something


def test_evidence_flags_a_raw_less_and_contrast_less_corridor():
    record = _record(1, target_node=2)
    record["evidence"]["has_raw"] = False
    record["evidence"]["contrast"] = False
    record["evidence"]["competing_components"] = {"9": 40}
    lines = "\n".join(controls_reconnect.evidence_lines(record))
    assert "geometry alone" in lines
    assert "no lumen/wall contrast" in lines
    assert "competing components nearby" in lines


def test_pending_orders_the_least_confident_first(document):
    order = [r["confidence"] for r in controls_reconnect.pending(document)]
    assert order == sorted(order)


def test_pending_excludes_what_has_already_been_ruled_on(document):
    record = document["candidates"][0]
    controls_reconnect.rule(document, record, accept=True)
    assert record not in controls_reconnect.pending(document)
    assert len(controls_reconnect.pending(document)) == 1


def test_route_points_distinguish_absent_from_empty():
    record = _record(1, target_node=2, alternatives=0)
    assert controls_reconnect.route_points(record) is not None
    assert controls_reconnect.route_points(record, which=0) is None


def test_a_ruling_does_not_overwrite_the_automatic_verdict(document):
    """The tool's own decision is evidence about the tool and has to survive."""
    record = document["candidates"][0]
    controls_reconnect.rule(document, record, accept=False, reason="not a vessel")
    assert record["decision"]["status"] == "review"      # untouched
    assert record["decision"]["operator"] == {"accept": False, "reason": "not a vessel"}


# --------------------------------------------------------------------- the widget


def test_the_list_shows_what_was_decided_not_what_was_scored():
    """The global pass overrules, and the panel must say so.

    A candidate carries two verdicts: what its own evidence concluded, and what the
    forest pass concluded once it could see the endpoint it was competing for. On the
    LADAF-28 run two routes scored ``accept`` and were then rejected -- one whose free
    end a better route had already claimed, one that would have closed a loop. Reporting
    the first would tell a reviewer more joins were made than were.
    """
    record = _record(1, target_node=2, status="accept", confidence=0.55)
    record["decision"] = {"status": "reject", "accepted": False, "rank": 4,
                          "reason": "would close a loop", "conflicts": ["x"]}

    assert controls_reconnect.final_status(record) == "reject"
    assert "[reject]" in controls_reconnect.describe_candidate(record)
    assert controls_reconnect.decision_note(record) == "would close a loop"

    document = {"schema": audit.SCHEMA, "candidates": [record]}
    assert "1 reject" in controls_reconnect.summary(document)
    assert controls_reconnect.pending(document) == []


def test_an_unoverruled_candidate_carries_no_note():
    record = _record(1, target_node=2, status="review")
    record["decision"]["status"] = "review"
    assert controls_reconnect.decision_note(record) == ""
    assert controls_reconnect.final_status(record) == "review"


def test_the_panel_builds_without_a_review_file(qapp):
    panel = controls_reconnect.build_reconnect_panel(FakeApp())
    assert panel.widgets["header"].text() == "no review file loaded"
    assert not panel.widgets["accept"].isEnabled()
    panel.refresh()  # must not raise


def test_loading_fills_the_work_list_and_selects_the_first(qapp, review_file):
    app = FakeApp()
    panel = controls_reconnect.build_reconnect_panel(app)
    panel.load(str(review_file))

    assert panel.widgets["list"].count() == 2
    assert panel.widgets["list"].currentRow() == 0
    assert panel.widgets["accept"].isEnabled()
    assert "candidate(s)" in panel.widgets["header"].text()
    # The detail pane shows the components and the evidence, not just a name.
    detail = panel.widgets["detail"].toPlainText()
    assert "components 1 -> 2" in detail
    assert "support vs intact vessel" in detail
    assert "1 alternative route(s) found" in detail


def test_selecting_a_candidate_asks_the_viewer_to_draw_it(qapp, review_file):
    app = FakeApp()
    panel = controls_reconnect.build_reconnect_panel(app)
    panel.load(str(review_file))

    assert app.drawn
    record, waypoints, prefix = app.drawn[-1]
    assert record is panel.state["records"][0]
    assert waypoints == []
    assert prefix == controls_reconnect.ACTOR_PREFIX


def test_a_viewer_that_cannot_draw_does_not_block_the_decision(qapp, review_file):
    """A drawing failure is a message, not a lost work list."""

    class Broken(FakeApp):
        def show_reconnect_candidate(self, record, *, waypoints, actor_prefix):
            raise RuntimeError("no render window")

    app = Broken()
    panel = controls_reconnect.build_reconnect_panel(app)
    panel.load(str(review_file))

    assert panel.widgets["list"].count() == 2
    assert any("could not draw" in m for m in app.messages)
    panel.rule_current(True)
    assert panel.state["records"][0]["decision"]["operator"]["accept"] is True


def test_accepting_advances_to_the_next_candidate(qapp, review_file):
    panel = controls_reconnect.build_reconnect_panel(FakeApp())
    panel.load(str(review_file))
    panel.rule_current(True)

    assert panel.widgets["list"].currentRow() == 1
    assert panel.widgets["list"].item(0).text().startswith("[accepted]")


def test_a_ruling_can_be_withdrawn(qapp, review_file):
    panel = controls_reconnect.build_reconnect_panel(FakeApp())
    panel.load(str(review_file))
    panel.rule_current(False)

    assert panel.widgets["list"].item(0).text().startswith("[rejected]")
    panel.widgets["list"].setCurrentRow(0)
    panel.undo_ruling()
    # The row still carries the tool's own verdict in brackets, so the check has to be
    # for the *ruling* markers specifically rather than for any bracket at all.
    assert not panel.widgets["list"].item(0).text().startswith(("[accepted]", "[rejected]"))
    assert "operator" not in panel.state["records"][0]["decision"]


def test_waypoints_come_from_the_viewer_pick_and_are_ordered(qapp, review_file):
    app = FakeApp()
    panel = controls_reconnect.build_reconnect_panel(app)
    panel.load(str(review_file))

    panel.add_waypoint()
    assert any("pick a point" in m for m in app.messages)
    assert panel.state["waypoints"] == []

    app.last_pick_um = (10.0, 20.0, 30.0)
    panel.add_waypoint()
    app.last_pick_um = (40.0, 50.0, 60.0)
    panel.add_waypoint()
    assert panel.state["waypoints"] == [[10.0, 20.0, 30.0], [40.0, 50.0, 60.0]]
    assert "2 waypoint(s) placed" in panel.widgets["detail"].toPlainText()

    panel.rule_current(True)
    ruling = panel.state["records"][0]["decision"]["operator"]
    assert ruling["waypoints_um"] == [[10.0, 20.0, 30.0], [40.0, 50.0, 60.0]]


def test_saved_decisions_are_readable_by_the_cli(qapp, review_file, tmp_path):
    """The round trip that matters: the panel writes what ``connect`` reads."""
    app = FakeApp()
    panel = controls_reconnect.build_reconnect_panel(app)
    panel.load(str(review_file))
    app.last_pick_um = (1.0, 2.0, 3.0)
    panel.add_waypoint()
    panel.rule_current(True)
    panel.save_decisions()

    document = audit.load_decisions(review_file)  # the CLI's own loader, schema-checked
    ruled = [c for c in document["candidates"]
             if (c.get("decision") or {}).get("operator")]
    assert len(ruled) == 1
    assert ruled[0]["decision"]["operator"]["accept"] is True
    assert ruled[0]["decision"]["operator"]["waypoints_um"] == [[1.0, 2.0, 3.0]]


def test_the_work_list_survives_the_dataset_going_away(qapp, review_file):
    app = FakeApp()
    panel = controls_reconnect.build_reconnect_panel(app)
    panel.load(str(review_file))

    app.session = None  # a failed load
    panel.refresh()

    assert panel.widgets["list"].count() == 2
    assert "candidate(s)" in panel.widgets["header"].text()


def test_the_panel_is_docked_into_the_control_tabs(qapp, tmp_path):
    """It has to actually appear, not merely be constructible.

    Built through the real ``build_control_panel`` against the same ``FakeApp``
    the other panel tests use, so this fails if the new tab breaks any of its
    neighbours rather than only when it breaks itself.
    """
    from hipct_seg_debug import controlpanel

    from .test_panels import FakeApp as PanelApp

    panel, _runner = controlpanel.build_control_panel(PanelApp(tmp_path))
    titles = [panel.tabs.tabText(i) for i in range(panel.tabs.count())]
    assert "Reconnect" in titles
    assert "reconnect" in panel.panels
    panel.refresh()  # every tab, including this one, must refresh together
