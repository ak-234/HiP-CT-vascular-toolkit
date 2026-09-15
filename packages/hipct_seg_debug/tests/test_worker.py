"""The rebuild queue must coalesce, serialise, and never swallow an error."""

from __future__ import annotations

import threading
import time

import numpy as np
import pytest

from hipct_seg_debug.edit.history import Patch
from hipct_seg_debug.edit.worker import RebuildOutcome, RebuildQueue, RebuildRequest


def box(lo, hi):
    return Patch(frozenset({0}), np.array([[lo, lo, lo], [hi, hi, hi]], dtype=float))


class Recorder:
    """Collects outcomes, and lets a test block the worker on demand."""

    def __init__(self):
        self.outcomes: list[RebuildOutcome] = []
        self.requests: list[RebuildRequest] = []
        self.lock = threading.Lock()
        self.gate = threading.Event()
        self.gate.set()
        self.entered = threading.Event()

    def rebuild(self, req):
        with self.lock:
            self.requests.append(req)
        self.entered.set()
        self.gate.wait(5.0)
        return f"built {req.snapshot}"

    def done(self, outcome):
        with self.lock:
            self.outcomes.append(outcome)


@pytest.fixture
def rec():
    return Recorder()


def make_queue(rec, **kw):
    kw.setdefault("settle_seconds", 0.0)
    q = RebuildQueue(rec.rebuild, rec.done, **kw)
    yield_q = q
    return yield_q


def test_a_single_request_is_rebuilt_and_delivered(rec):
    q = make_queue(rec)
    try:
        q.request(box(0, 10), "snap-1")
        assert q.wait_idle(5.0)
        time.sleep(0.05)
        assert len(rec.outcomes) == 1
        outcome = rec.outcomes[0]
        assert outcome.ok
        assert outcome.result == "built snap-1"
        assert outcome.seconds >= 0.0
    finally:
        q.shutdown()


def test_requests_arriving_during_a_rebuild_are_coalesced(rec):
    """A drag emits a request per mouse move; they must collapse into one."""
    rec.gate.clear()
    q = make_queue(rec)
    try:
        q.request(box(0, 10), "snap-1")
        assert rec.entered.wait(5.0), "worker never started the first rebuild"

        # Three more arrive while the first is still running.
        q.request(box(20, 30), "snap-2")
        q.request(box(40, 50), "snap-3")
        q.request(box(60, 70), "snap-4")

        rec.gate.set()
        assert q.wait_idle(5.0)
        time.sleep(0.05)

        assert len(rec.requests) == 2, "the three queued requests should have merged into one"
        assert len(rec.outcomes) == 2

        # The merged request carries the newest snapshot...
        second = rec.requests[1]
        assert second.snapshot == "snap-4"
        # ...but a box covering everything the burst touched, so no edit is lost.
        assert second.patch.aabb[0][0] == 20.0
        assert second.patch.aabb[1][0] == 70.0
    finally:
        rec.gate.set()
        q.shutdown()


def test_only_one_rebuild_runs_at_a_time(rec):
    concurrent = []
    active = threading.Lock()
    running = {"n": 0}

    def rebuild(req):
        with active:
            running["n"] += 1
            concurrent.append(running["n"])
        time.sleep(0.02)
        with active:
            running["n"] -= 1
        return req.snapshot

    q = RebuildQueue(rebuild, rec.done, settle_seconds=0.0)
    try:
        for i in range(6):
            q.request(box(i, i + 1), f"snap-{i}")
            time.sleep(0.03)
        assert q.wait_idle(5.0)
        assert max(concurrent) == 1, f"rebuilds overlapped: {concurrent}"
    finally:
        q.shutdown()


def test_an_exception_is_reported_not_swallowed(rec):
    def rebuild(req):
        raise RuntimeError("meshlib fell over")

    q = RebuildQueue(rebuild, rec.done, settle_seconds=0.0)
    try:
        q.request(box(0, 10), "snap-1")
        assert q.wait_idle(5.0)
        time.sleep(0.05)
        assert len(rec.outcomes) == 1
        outcome = rec.outcomes[0]
        assert not outcome.ok
        assert isinstance(outcome.error, RuntimeError)
        assert "meshlib fell over" in outcome.traceback
    finally:
        q.shutdown()


def test_a_failed_rebuild_does_not_stop_the_queue(rec):
    calls = {"n": 0}

    def rebuild(req):
        calls["n"] += 1
        if calls["n"] == 1:
            raise ValueError("first one fails")
        return "ok"

    q = RebuildQueue(rebuild, rec.done, settle_seconds=0.0)
    try:
        q.request(box(0, 1), "a")
        assert q.wait_idle(5.0)
        q.request(box(0, 1), "b")
        assert q.wait_idle(5.0)
        time.sleep(0.05)
        assert [o.ok for o in rec.outcomes] == [False, True]
    finally:
        q.shutdown()


def test_settle_delay_merges_a_fast_burst_into_one_rebuild(rec):
    q = RebuildQueue(rec.rebuild, rec.done, settle_seconds=0.15)
    try:
        for i in range(5):
            q.request(box(i * 10, i * 10 + 5), f"snap-{i}")
            time.sleep(0.01)
        assert q.wait_idle(5.0)
        time.sleep(0.05)
        assert len(rec.requests) == 1, "a burst inside the settle window should rebuild once"
        assert rec.requests[0].patch.aabb[1][0] == 45.0
    finally:
        q.shutdown()


def test_post_is_used_to_hand_results_back(rec):
    """A GUI passes a `post` that hops onto its own thread; we just record it."""
    posted = []

    def post(fn):
        posted.append(threading.current_thread().name)
        fn()

    q = RebuildQueue(rec.rebuild, rec.done, post=post, settle_seconds=0.0)
    try:
        q.request(box(0, 10), "snap-1")
        assert q.wait_idle(5.0)
        time.sleep(0.05)
        assert len(posted) == 1
        assert len(rec.outcomes) == 1
    finally:
        q.shutdown()


def test_shutdown_is_idempotent_and_stops_the_thread(rec):
    q = make_queue(rec)
    q.request(box(0, 10), "snap-1")
    q.wait_idle(5.0)
    q.shutdown()
    q.shutdown()
    assert not q._thread.is_alive()


def test_request_returns_increasing_serials(rec):
    q = make_queue(rec)
    try:
        serials = [q.request(box(0, 1), "s") for _ in range(3)]
        assert serials == [1, 2, 3]
    finally:
        q.shutdown()


def test_merged_request_keeps_the_highest_serial():
    older = RebuildRequest(box(0, 10), "old", None, 1)
    newer = RebuildRequest(box(20, 30), "new", None, 4)
    merged = newer.merged(older)
    assert merged.serial == 4
    assert merged.snapshot == "new"
    assert merged.patch.aabb[0][0] == 0.0 and merged.patch.aabb[1][0] == 30.0


# ------------------------------------------------------- the Qt hand-back


def test_qt_poster_delivers_from_a_worker_thread():
    """The bug this guards is silent: nothing raises, the callback just never runs.

    `qt_poster` used to return `QTimer.singleShot`, which creates its timer in the
    *calling* thread. `RebuildQueue._loop` calls `post` from the worker thread, which
    has no event loop, so every rebuild outcome was dropped on the floor with no
    error anywhere. Verified directly against PyQt5 before the fix.
    """
    import os
    import threading
    import time

    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    pytest.importorskip("qtpy")
    from qtpy.QtWidgets import QApplication

    from hipct_seg_debug.edit.worker import qt_poster

    app = QApplication.instance() or QApplication([])
    post = qt_poster()
    delivered = []

    threading.Thread(target=lambda: post(lambda: delivered.append("here"))).start()

    deadline = time.time() + 5
    while not delivered and time.time() < deadline:
        app.processEvents()
        time.sleep(0.01)
    assert delivered == ["here"]


def test_qt_poster_keeps_its_receiver_alive():
    """A collected QObject takes the connection with it and posts vanish again."""
    pytest.importorskip("qtpy")
    from hipct_seg_debug.edit.worker import qt_poster

    assert getattr(qt_poster(), "_poster", None) is not None
