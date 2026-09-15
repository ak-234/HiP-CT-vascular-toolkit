"""Run rebuilds off the GUI thread, and coalesce bursts of edits.

Everything in the parent package runs on the Qt event loop -- a pick already
costs 4-6 seconds there, and the README's first GUI trap is that building a
napari viewer inside a VTK callback takes the process down. A one-second rebuild
on that thread would freeze both windows on every keystroke.

Two behaviours matter more than raw throughput:

* **one rebuild at a time.** Two ``evaluate_sdf`` calls in parallel would double
  peak memory for a result that is thrown away anyway.
* **newest wins.** Dragging a node emits a rebuild request per mouse move. The
  intermediate ones are already stale by the time they could run, so they are
  merged into one request covering everything the burst touched.

Deliberately free of any Qt import: the queue takes a ``post`` callable that
hands a result back to whichever thread owns the UI. That keeps the logic
testable without a display, and is why the Qt binding at the bottom is three
lines.
"""

from __future__ import annotations

import threading
import time
import traceback
from dataclasses import dataclass
from typing import Any, Callable

from .history import Patch


@dataclass
class RebuildRequest:
    """What to rebuild, and against which snapshot of the graph."""

    patch: Patch
    snapshot: Any  # a Triple, already detached from the live graph
    root_pref: set[int] | None = None
    serial: int = 0

    def merged(self, other: "RebuildRequest") -> "RebuildRequest":
        """Fold an older request into this one, keeping the newer snapshot."""
        return RebuildRequest(
            patch=self.patch.merged(other.patch),
            snapshot=self.snapshot,
            root_pref=self.root_pref,
            serial=max(self.serial, other.serial),
        )


@dataclass
class RebuildOutcome:
    """Delivered back to the UI thread, whether the rebuild worked or not."""

    request: RebuildRequest
    result: Any = None  # a PatchResult on success
    error: BaseException | None = None
    traceback: str = ""
    seconds: float = 0.0

    @property
    def ok(self) -> bool:
        return self.error is None


def _call_directly(fn: Callable[[], None]) -> None:
    """Default `post`: run the callback on the worker thread.

    Correct for scripts and tests. A GUI must override it -- see
    :func:`qt_poster`.
    """
    fn()


class RebuildQueue:
    """Serialises patch rebuilds on a background thread, newest request wins.

    ``request`` is called from the UI thread and returns immediately. Exactly one
    :class:`RebuildOutcome` is delivered through `post` per rebuild that actually
    runs; superseded requests are silently folded into their successor rather
    than producing an outcome of their own.
    """

    def __init__(
        self,
        rebuild: Callable[[RebuildRequest], Any],
        on_done: Callable[[RebuildOutcome], None],
        *,
        post: Callable[[Callable[[], None]], None] = _call_directly,
        settle_seconds: float = 0.12,
        name: str = "sdf-rebuild",
    ):
        self._rebuild = rebuild
        self._on_done = on_done
        self._post = post
        self._settle = settle_seconds

        self._lock = threading.Lock()
        self._wake = threading.Condition(self._lock)
        self._pending: RebuildRequest | None = None
        self._serial = 0
        self._running = False
        self._stop = False
        self._last_request_at = 0.0

        self._thread = threading.Thread(target=self._loop, name=name, daemon=True)
        self._thread.start()

    # ------------------------------------------------------------- UI thread

    def request(self, patch: Patch, snapshot: Any, root_pref: set[int] | None = None) -> int:
        """Queue a rebuild. Returns the serial number it was given."""
        with self._wake:
            self._serial += 1
            self._last_request_at = time.monotonic()
            req = RebuildRequest(patch, snapshot, root_pref, self._serial)
            # Merge rather than replace: a burst of small edits must rebuild the
            # union of what it touched, not just the last one's box.
            self._pending = req if self._pending is None else req.merged(self._pending)
            self._wake.notify()
            return self._serial

    @property
    def busy(self) -> bool:
        with self._lock:
            return self._running or self._pending is not None

    def wait_idle(self, timeout: float = 60.0) -> bool:
        """Block until nothing is queued or running. For tests and shutdown."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if not self.busy:
                return True
            time.sleep(0.01)
        return False

    def shutdown(self, timeout: float = 5.0) -> None:
        with self._wake:
            self._stop = True
            self._pending = None
            self._wake.notify_all()
        self._thread.join(timeout)

    # --------------------------------------------------------- worker thread

    def _loop(self) -> None:
        while True:
            with self._wake:
                while self._pending is None and not self._stop:
                    self._wake.wait(0.25)
                if self._stop:
                    return
                # Let a burst settle before starting, so a drag does not kick off
                # a rebuild it will immediately supersede. Condition.wait returns
                # as soon as a new request notifies it, so this has to loop until
                # the queue has actually been quiet for the whole window rather
                # than simply waiting once.
                while self._settle > 0:
                    quiet_for = time.monotonic() - self._last_request_at
                    if quiet_for >= self._settle:
                        break
                    self._wake.wait(self._settle - quiet_for)
                    if self._stop:
                        return
                req, self._pending = self._pending, None
                if req is None:
                    continue
                self._running = True

            outcome = self._run(req)

            with self._lock:
                self._running = False
            # Deliver even when superseded: the caller asked for this box and a
            # newer request will simply overwrite the display a moment later.
            self._post(lambda o=outcome: self._on_done(o))

    def _run(self, req: RebuildRequest) -> RebuildOutcome:
        t0 = time.time()
        try:
            result = self._rebuild(req)
        except BaseException as exc:  # noqa: BLE001 - reported, never swallowed
            return RebuildOutcome(
                request=req, error=exc, traceback=traceback.format_exc(),
                seconds=time.time() - t0,
            )
        return RebuildOutcome(request=req, result=result, seconds=time.time() - t0)


def qt_poster() -> Callable[[Callable[[], None]], None]:
    """A `post` that hops back onto the Qt event loop. **Call this on the GUI thread.**

    A queued signal, not ``QTimer.singleShot``. The difference is not stylistic:
    ``QTimer.singleShot(0, fn)`` creates the timer in *the thread that calls it*, and
    a worker thread has no event loop to run it, so the callback simply never fires --
    silently, with no error anywhere. ``_loop`` posts from the worker thread, so every
    rebuild outcome was being dropped. Verified directly: a `singleShot` posted from a
    `threading.Thread` never arrives, while the signal below does.

    Emitting a signal from another thread is the supported way across a thread
    boundary; the connection is automatically a queued one, so ``fn`` runs on the
    thread this object was created in. That still satisfies the rule the docstring
    was written for -- nothing Qt or vispy may be built anywhere else, or the process
    dies with an access violation rather than an exception.
    """
    from qtpy.QtCore import QObject, Signal

    class _Poster(QObject):
        fired = Signal(object)

        def __init__(self):
            super().__init__()
            self.fired.connect(lambda fn: fn())

    poster = _Poster()

    def post(fn: Callable[[], None]) -> None:
        poster.fired.emit(fn)

    # The QObject has to outlive the closure: once it is collected the connection
    # goes with it and posts start disappearing again, which is the same failure
    # this function exists to fix.
    post._poster = poster
    return post


def session_rebuilder(session, *, min_extent_um: float = 4000.0):
    """A `rebuild` callable that drives an :class:`~.sdfpatch.SdfSession`.

    Re-prepares the session against the request's snapshot before rebuilding, so
    the patch reflects the graph as it stood when the edit was made rather than
    whatever the user has done since.
    """

    def rebuild(req: RebuildRequest):
        session.set_graph(req.snapshot, req.root_pref)
        return session.rebuild_around(req.patch, min_extent_um=min_extent_um)

    return rebuild
