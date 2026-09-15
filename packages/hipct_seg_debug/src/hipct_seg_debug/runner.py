"""Run CLI commands from the GUI without freezing it, and without lying about it.

Two backends behind one façade, because the nine toolkit commands are not one kind
of thing. `report` is a second of graph arithmetic; `skeletonise` at stride 1 is
twelve minutes of numba. Running both the same way means either paying interpreter
startup on the fast ones or offering a Stop button that does nothing on the slow
ones.

* **In-process** (`report`, `gaps`, `connect`, `repair-radius --source outlier`) --
  a plain thread calling `edit.__main__.main(argv)`. The GUI and the CLI then run
  the same code, so they cannot diverge; stdout is captured by claiming a per-thread
  sink on `sys.stdout`.
* **Subprocess** (everything else) -- `python -u -m hipct_seg_debug.edit ...` under a
  `QProcess`, streamed line by line. Cancellable for real, and a numba segfault takes
  the child down rather than the window.

`threading.Thread` and a `post` callable rather than `QThread`, matching
`edit/worker.py`, whose docstring makes the case: keeping Qt out of the logic is what
lets the queue be tested without a display. The only Qt in this module is inside
`_Subprocess`, imported lazily.

**Two things this deliberately does not pretend to do.** Stop cannot interrupt an
in-process job -- nothing in `find_sites`, `decode_volume` or `fill_spans` has a
cancellation point, and `PyThreadState_SetAsyncExc` is not safe inside numpy's C
code. It clears the queue and marks the result cancelled, and the button says so. And
the stdout capture is Python-level: VTK and numba write to fd 1 directly and keep
going to the terminal. Both are reasons `mode_for` sends anything doubtful to the
subprocess side, where the pipe is real.
"""

from __future__ import annotations

import io
import os
import threading
import time
import traceback
from collections import deque
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

from . import cliform

#: Commands whose work is small enough to run on a thread in this process.
INPROC = ("report", "gaps", "connect", "repair-radius", "crop")


def mode_for(argv: Sequence[str]) -> str:
    """Pick a backend from the whole argv, not just the command name.

    `repair-radius` is only cheap with ``--source outlier``. Its default is ``both``,
    which runs `crosssection.find_sites` over the entire graph -- the same call that
    costs 14.6 s inside `find_candidates`, here without the ROI to bound it.

    `connect` is the same shape of problem in reverse: cheap by default, but
    ``--dpc`` reads a TIFF window and runs a Sato filter per proposal, which is
    minutes and must stay interruptible.
    """
    if not argv:
        return "subprocess"
    if argv[0] == "repair-radius" and not any(a == "--source=outlier" for a in argv):
        return "subprocess"
    if argv[0] == "connect" and "--dpc" in argv:
        return "subprocess"
    return "inproc" if argv[0] in INPROC else "subprocess"


@dataclass(frozen=True)
class Job:
    """One command to run, and what it claims it will write."""

    argv: list[str]
    label: str = ""
    mode: str = ""  # "" picks with mode_for
    outputs: tuple[str, ...] = ()
    tag: Any = None  # the workflow step this belongs to, if any
    executable: str = ""  # optional alternate Python (DF21 uses Python 3.9)

    @property
    def command(self) -> str:
        return self.argv[0] if self.argv else ""

    @property
    def title(self) -> str:
        return self.label or " ".join(self.argv)

    def resolved_mode(self) -> str:
        return "subprocess" if self.executable else (self.mode or mode_for(self.argv))


@dataclass
class JobResult:
    job: Job
    returncode: int = 0
    error: BaseException | None = None
    traceback: str = ""
    seconds: float = 0.0
    cancelled: bool = False
    outputs_written: tuple[Path, ...] = ()

    @property
    def ok(self) -> bool:
        return self.returncode == 0 and self.error is None and not self.cancelled

    def describe(self) -> str:
        if self.cancelled:
            return f"{self.job.title}: stopped after {self.seconds:.1f}s"
        if self.error is not None:
            return f"{self.job.title}: {type(self.error).__name__}: {self.error}"
        if self.returncode:
            return f"{self.job.title}: exit {self.returncode} after {self.seconds:.1f}s"
        wrote = ", ".join(p.name for p in self.outputs_written)
        return (f"{self.job.title}: done in {self.seconds:.1f}s"
                + (f" -> {wrote}" if wrote else ""))


# ------------------------------------------------------------- stdout plumbing


class LineBuffer:
    """Split a stream into log lines, treating ``\\r`` as "replace the last one".

    `cmd_mask_export:494` prints its progress with ``end="\\r"`` every 100 planes.
    Appending those would grow the pane by 13 lines for one export; the log panel
    overwrites instead, which is what the terminal does.
    """

    def __init__(self):
        self._buf = ""

    def feed(self, text: str) -> list[tuple[str, bool]]:
        """Return ``(line, transient)`` pairs. ``transient`` means "will be replaced"."""
        out: list[tuple[str, bool]] = []
        self._buf += text
        while True:
            breaks = [p for p in (self._buf.find("\n"), self._buf.find("\r")) if p >= 0]
            if not breaks:
                break
            i = min(breaks)
            line, sep, self._buf = self._buf[:i], self._buf[i], self._buf[i + 1:]
            if sep == "\r" and self._buf.startswith("\n"):
                # A Windows CRLF is one break, not a transient line plus a real one.
                self._buf = self._buf[1:]
                sep = "\n"
            out.append((line, sep == "\r"))
        return out

    def flush(self) -> list[tuple[str, bool]]:
        """Emit whatever is left when a stream ends without a final newline."""
        rest, self._buf = self._buf, ""
        return [(rest, False)] if rest else []


class TeeStdout(io.TextIOBase):
    """A stdout replacement that routes writes by thread.

    A thread that has claimed a sink writes there; every other thread passes straight
    through to the real stream. `contextlib.redirect_stdout` cannot do this -- it is
    process-global, so a GUI job would swallow anything the Qt thread printed at the
    same moment, and a GUI that eats the output the CLI would have shown is one you
    cannot debug.
    """

    def __init__(self, real):
        self._real = real
        self._sinks: dict[int, Callable[[str], None]] = {}
        self._lock = threading.Lock()

    @contextmanager
    def claim(self, sink: Callable[[str], None]):
        ident = threading.get_ident()
        with self._lock:
            previous = self._sinks.get(ident)
            self._sinks[ident] = sink
        try:
            yield
        finally:
            with self._lock:
                if previous is None:
                    self._sinks.pop(ident, None)
                else:
                    self._sinks[ident] = previous

    def write(self, text: str) -> int:
        with self._lock:
            sink = self._sinks.get(threading.get_ident())
        if sink is None:
            return self._real.write(text)
        sink(text)
        return len(text)

    def flush(self) -> None:
        self._real.flush()

    def isatty(self) -> bool:
        return bool(getattr(self._real, "isatty", lambda: False)())

    @property
    def encoding(self):
        return getattr(self._real, "encoding", "utf-8")

    @property
    def errors(self):
        return getattr(self._real, "errors", "replace")

    def fileno(self):
        # Better to refuse than to hand out fd 1 and have a caller write past the tee.
        raise io.UnsupportedOperation("TeeStdout has no file descriptor")


def install_tee():
    """Replace ``sys.stdout``/``sys.stderr`` with tees, returning them.

    stderr matters as much as stdout: argparse writes "unrecognized arguments" there,
    and a panel that showed nothing when a form produced bad argv would be baffling.
    """
    import sys

    if isinstance(sys.stdout, TeeStdout):
        return sys.stdout, sys.stderr
    sys.stdout = TeeStdout(sys.stdout)
    sys.stderr = TeeStdout(sys.stderr)
    return sys.stdout, sys.stderr


# ------------------------------------------------------------------- the runner


def _call_directly(fn: Callable[[], None]) -> None:
    """Default `post`: run it here. Correct for tests; a GUI must override it."""
    fn()


def _default_dispatch(argv: Sequence[str]) -> int:
    from .edit.__main__ import main as edit_main

    return int(edit_main(list(argv)) or 0)


class CommandRunner:
    """A FIFO of jobs, one at a time, across both backends.

    No coalescing, unlike `edit.worker.RebuildQueue`: newest-wins is right when the
    intermediate results are stale pixels, and wrong here, where every job writes a
    file that the next one reads.

    Callbacks are single slots rather than Qt signals, matching
    `picker.on_layers_changed` and `controller.on_status`, so the whole class imports
    and tests without Qt.
    """

    def __init__(self, dispatch=None, post=None, cwd=None, tee=None):
        self.on_line: Callable[[str, bool], None] | None = None
        self.on_started: Callable[[Job], None] | None = None
        self.on_done: Callable[[JobResult], None] | None = None
        self.on_queue: Callable[[], None] | None = None

        self._dispatch = dispatch or _default_dispatch
        self._post = post or _call_directly
        self._tee = tee
        self.cwd = str(cwd) if cwd else os.getcwd()

        self._queue: deque[Job] = deque()
        self._current: Job | None = None
        self._started_at = 0.0
        self._cancel = False
        self._proc = None
        self._lock = threading.Lock()
        self._pending: list[tuple[str, bool]] = []
        self._draining = False

    # -- state ------------------------------------------------------------

    @property
    def busy(self) -> bool:
        return self._current is not None

    @property
    def current(self) -> Job | None:
        return self._current

    @property
    def queued(self) -> tuple[Job, ...]:
        return tuple(self._queue)

    @property
    def elapsed(self) -> float:
        return time.perf_counter() - self._started_at if self.busy else 0.0

    # -- submission -------------------------------------------------------

    def submit(self, job: Job) -> None:
        self._queue.append(job)
        self._changed()
        self._pump()

    def clear_queue(self) -> int:
        """Drop everything not yet started. Returns how many were dropped."""
        n = len(self._queue)
        self._queue.clear()
        if n:
            self._changed()
        return n

    def stop(self) -> None:
        """Clear the queue, and kill the running job if it is a child process.

        For an in-process job this cannot interrupt the work -- it only stops the
        chain and marks the outcome cancelled. The button's tooltip says so.
        """
        self.clear_queue()
        self._cancel = True
        proc = self._proc
        if proc is not None:
            self.emit_line("(stopping...)", False)
            proc.stop()

    # -- output -----------------------------------------------------------

    def emit_line(self, text: str, transient: bool = False) -> None:
        self._emit([(text, transient)])

    def _emit(self, lines: list[tuple[str, bool]]) -> None:
        """Batch lines onto the UI thread.

        While a drain is already scheduled, further lines just accumulate -- which is
        the coalescing, with no timer to manage.
        """
        if not lines:
            return
        with self._lock:
            self._pending.extend(lines)
            if self._draining:
                return
            self._draining = True
        self._post(self._drain)

    def _drain(self) -> None:
        with self._lock:
            pending, self._pending = self._pending, []
            self._draining = False
        if self.on_line is None:
            return
        for text, transient in pending:
            self.on_line(text, transient)

    def _changed(self) -> None:
        if self.on_queue is not None:
            self.on_queue()

    # -- the pump ---------------------------------------------------------

    def _pump(self) -> None:
        if self._current is not None or not self._queue:
            return
        job = self._queue.popleft()
        self._current = job
        self._cancel = False
        self._started_at = time.perf_counter()
        self._changed()
        if self.on_started is not None:
            self.on_started(job)

        self.emit_line(f"$ {self._echo(job)}", False)
        if job.resolved_mode() == "inproc":
            self._run_inproc(job)
        else:
            self._run_subprocess(job)

    def _echo(self, job: Job) -> str:
        if job.resolved_mode() == "inproc":
            return "python -m hipct_seg_debug.edit " + " ".join(job.argv)
        return " ".join(cliform.subprocess_command(
            job.argv, executable=job.executable or None
        ))

    def _finish(self, result: JobResult) -> None:
        result.seconds = time.perf_counter() - self._started_at
        result.cancelled = result.cancelled or self._cancel
        result.outputs_written = tuple(
            Path(p) for p in result.job.outputs if p and Path(p).exists()
        )
        self._current = None
        self._proc = None
        self._changed()
        self.emit_line(result.describe(), False)
        if self.on_done is not None:
            self.on_done(result)
        self._pump()

    # -- in-process -------------------------------------------------------

    def _run_inproc(self, job: Job) -> None:
        buffer = LineBuffer()

        def sink(text: str) -> None:
            self._emit(buffer.feed(text))

        def body() -> None:
            result = JobResult(job=job)
            try:
                with self._claim(sink):
                    result.returncode = self._dispatch(job.argv)
            except SystemExit as exc:
                # argparse exits 2 on bad argv rather than raising.
                result.returncode = int(exc.code or 0)
            except BaseException as exc:  # noqa: BLE001 - must not kill the window
                result.error = exc
                result.returncode = 1
                result.traceback = traceback.format_exc()
                self._emit([(line, False) for line in result.traceback.splitlines()])
            self._emit(buffer.flush())
            self._post(lambda: self._finish(result))

        threading.Thread(target=body, name=f"cmd-{job.command}", daemon=True).start()

    @contextmanager
    def _claim(self, sink):
        if self._tee is None:
            yield
            return
        with self._tee.claim(sink):
            yield

    # -- subprocess -------------------------------------------------------

    def _run_subprocess(self, job: Job) -> None:
        buffer = LineBuffer()

        def on_text(text: str) -> None:
            self._emit(buffer.feed(text))

        def on_exit(returncode: int, crashed: bool) -> None:
            self._emit(buffer.flush())
            self._finish(JobResult(job=job, returncode=returncode, cancelled=crashed))

        try:
            self._proc = _Subprocess(job, self.cwd, on_text, on_exit)
            self._proc.start()
        except Exception as exc:  # noqa: BLE001 - a missing Qt must not be fatal
            self._finish(JobResult(job=job, error=exc, returncode=1,
                                   traceback=traceback.format_exc()))


class _Subprocess:
    """A `QProcess` wrapper. The only Qt in this module.

    `QProcess` rather than `subprocess` plus a reader thread because
    ``readyReadStandardOutput`` fires *on the GUI thread* with the data already
    buffered -- no extra thread, no hop back, and so no way to break the rule that
    has bitten this codebase three times (`viewer3d.py:1000-1004`).
    """

    def __init__(self, job: Job, cwd: str, on_text, on_exit):
        from qtpy.QtCore import QProcess

        self.job = job
        self._on_text = on_text
        self._on_exit = on_exit
        self._killed = False

        self._proc = QProcess()
        self._proc.setProcessChannelMode(QProcess.MergedChannels)
        self._proc.setWorkingDirectory(cwd)
        self._proc.setProcessEnvironment(_child_environment())
        self._proc.readyReadStandardOutput.connect(self._read)
        self._proc.finished.connect(self._finished)
        self._proc.errorOccurred.connect(self._failed)

    def start(self) -> None:
        command = cliform.subprocess_command(
            self.job.argv, executable=self.job.executable or None
        )
        self._proc.start(command[0], command[1:])

    def _read(self) -> None:
        raw = bytes(self._proc.readAllStandardOutput())
        self._on_text(raw.decode("utf-8", errors="replace"))

    def _finished(self, code, status) -> None:
        from qtpy.QtCore import QProcess

        crashed = self._killed or status == QProcess.CrashExit
        self._on_exit(int(code), crashed)

    def _failed(self, error) -> None:
        from qtpy.QtCore import QProcess

        if error == QProcess.FailedToStart:
            self._on_text(f"could not start {cliform.subprocess_command(self.job.argv)[0]}\n")
            self._on_exit(127, False)

    def stop(self) -> None:
        self._killed = True
        self._proc.terminate()
        if not self._proc.waitForFinished(2000):
            self._proc.kill()


def _child_environment():
    """The child's environment, with the two settings that decide whether it works.

    `PYTHONUNBUFFERED` pairs with the `-u` in `subprocess_command`; `PYTHONIOENCODING`
    stops a `um` or `->` in the output killing the child with `UnicodeEncodeError`
    when the inherited console codepage is cp1252.
    """
    from qtpy.QtCore import QProcessEnvironment

    env = QProcessEnvironment.systemEnvironment()
    env.insert("PYTHONUNBUFFERED", "1")
    env.insert("PYTHONIOENCODING", "utf-8")
    package_parent = str(Path(__file__).resolve().parent.parent)
    inherited = env.value("PYTHONPATH")
    env.insert("PYTHONPATH", package_parent + (os.pathsep + inherited if inherited else ""))
    return env
