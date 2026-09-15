"""The job queue, the line splitter and the stdout tee.

All of it runs synchronously here: `post=_call_directly` collapses the thread hop, so
a queue that would take twelve minutes in the window takes microseconds in a test.
The subprocess backend needs a Qt event loop and is not covered -- its two pure
parts, `subprocess_command` and `LineBuffer`, are tested instead, which is most of
what can go wrong.
"""

from __future__ import annotations

import io
import threading

import pytest

from hipct_seg_debug.runner import (
    INPROC,
    CommandRunner,
    Job,
    JobResult,
    LineBuffer,
    TeeStdout,
    mode_for,
)


def _runner(dispatch, **kw):
    """A runner whose thread hop is collapsed, so submit() runs to completion."""
    calls = []
    runner = CommandRunner(dispatch=dispatch, post=lambda fn: fn(), **kw)
    runner.on_line = lambda text, transient: calls.append(text)
    runner.lines = calls
    return runner


def _wait(runner, timeout=5.0):
    """The in-process backend still uses a real thread; give it a moment."""
    deadline = threading.Event()
    for _ in range(int(timeout / 0.01)):
        if not runner.busy:
            return
        deadline.wait(0.01)
    raise AssertionError("job never finished")


# ------------------------------------------------------------------- mode_for


@pytest.mark.parametrize("command", [c for c in INPROC if c != "repair-radius"])
def test_the_cheap_commands_run_in_process(command):
    assert mode_for([command, "g.am"]) == "inproc"


@pytest.mark.parametrize("command", ["skeletonise", "optimise", "repair-mask",
                                     "mask-export", "surface"])
def test_the_expensive_commands_are_children(command):
    assert mode_for([command, "g.am"]) == "subprocess"


def test_repair_radius_is_only_cheap_with_the_outlier_detector():
    """Its default is `both`, which runs the image detector over the whole graph."""
    assert mode_for(["repair-radius", "g.am"]) == "subprocess"
    assert mode_for(["repair-radius", "g.am", "--source=image"]) == "subprocess"
    assert mode_for(["repair-radius", "g.am", "--source=both"]) == "subprocess"
    assert mode_for(["repair-radius", "g.am", "--source=outlier"]) == "inproc"


def test_connect_is_only_cheap_without_the_dpc_walk():
    """--dpc reads a TIFF window and filters it per proposal: minutes, and Stop
    cannot interrupt an in-process job."""
    assert mode_for(["connect", "g.am"]) == "inproc"
    assert mode_for(["connect", "g.am", "--tjunction"]) == "inproc"
    assert mode_for(["connect", "g.am", "--dpc", "--raw", "d/"]) == "subprocess"


def test_an_empty_argv_does_not_crash():
    assert mode_for([]) == "subprocess"


# ----------------------------------------------------------------- LineBuffer


def test_whole_lines_come_out_permanent():
    assert LineBuffer().feed("one\ntwo\n") == [("one", False), ("two", False)]


def test_a_partial_line_is_held_until_it_ends():
    buf = LineBuffer()
    assert buf.feed("half") == []
    assert buf.feed(" a line\n") == [("half a line", False)]


def test_carriage_return_marks_a_line_transient():
    """`cmd_mask_export:494` prints progress with end='\\r'; those replace, not append."""
    assert LineBuffer().feed("  1/1250\r  2/1250\r") == [("  1/1250", True), ("  2/1250", True)]


def test_crlf_is_one_break_not_two():
    assert LineBuffer().feed("line\r\nnext\r\n") == [("line", False), ("next", False)]


def test_flush_emits_a_trailing_fragment():
    buf = LineBuffer()
    buf.feed("no newline")
    assert buf.flush() == [("no newline", False)]
    assert buf.flush() == []


def test_flush_is_empty_when_everything_was_consumed():
    buf = LineBuffer()
    buf.feed("done\n")
    assert buf.flush() == []


# ------------------------------------------------------------------ TeeStdout


def test_a_claiming_thread_is_captured_and_others_are_not():
    real = io.StringIO()
    tee = TeeStdout(real)
    caught = []
    with tee.claim(caught.append):
        tee.write("captured")
    tee.write("passed through")
    assert caught == ["captured"]
    assert real.getvalue() == "passed through"


def test_another_thread_writes_through_while_one_is_claimed():
    """The GUI thread must keep printing to the terminal while a job is captured."""
    real = io.StringIO()
    tee = TeeStdout(real)
    caught = []
    done = threading.Event()

    def other():
        tee.write("from the other thread")
        done.set()

    with tee.claim(caught.append):
        threading.Thread(target=other).start()
        done.wait(2)
        tee.write("mine")

    assert caught == ["mine"]
    assert real.getvalue() == "from the other thread"


def test_the_claim_is_released_even_when_the_body_raises():
    tee = TeeStdout(io.StringIO())
    with pytest.raises(ValueError):
        with tee.claim(lambda _t: None):
            raise ValueError("boom")
    assert tee._sinks == {}


def test_fileno_refuses_rather_than_handing_out_the_real_one():
    with pytest.raises(io.UnsupportedOperation):
        TeeStdout(io.StringIO()).fileno()


def test_print_reaches_the_sink():
    tee = TeeStdout(io.StringIO())
    caught = []
    with tee.claim(caught.append):
        print("hello", file=tee)
    assert "".join(caught) == "hello\n"


# ---------------------------------------------------------------- the queue


def test_a_job_runs_and_reports_success():
    runner = _runner(lambda argv: 0)
    results = []
    runner.on_done = results.append
    runner.submit(Job(argv=["report", "g.am"], mode="inproc"))
    _wait(runner)
    assert len(results) == 1 and results[0].ok


def test_the_command_is_echoed_before_it_runs():
    runner = _runner(lambda argv: 0)
    runner.submit(Job(argv=["report", "g.am"], mode="inproc"))
    _wait(runner)
    assert runner.lines[0].startswith("$ python -m hipct_seg_debug.edit report")


def test_output_is_captured_and_split(monkeypatch):
    """The tee only sees `print` once it *is* sys.stdout -- that is what install_tee does."""
    import sys

    tee = TeeStdout(io.StringIO())
    monkeypatch.setattr(sys, "stdout", tee)

    def dispatch(argv):
        print("first line")
        print("second line")
        return 0

    runner = _runner(dispatch, tee=tee)
    runner.submit(Job(argv=["report", "g.am"], mode="inproc"))
    _wait(runner)
    assert "first line" in runner.lines and "second line" in runner.lines


def test_a_progress_line_stays_transient_through_the_runner(monkeypatch):
    import sys

    tee = TeeStdout(io.StringIO())
    monkeypatch.setattr(sys, "stdout", tee)
    seen = []

    def dispatch(argv):
        print("  encoding 100/1250 planes", end="\r")
        print("  encoding 200/1250 planes", end="\r")
        return 0

    runner = CommandRunner(dispatch=dispatch, post=lambda fn: fn(), tee=tee)
    runner.on_line = lambda text, transient: seen.append((text.strip(), transient))
    runner.submit(Job(argv=["mask-export"], mode="inproc"))
    _wait(runner)
    assert ("encoding 100/1250 planes", True) in seen


def test_a_nonzero_return_is_not_ok():
    runner = _runner(lambda argv: 3)
    results = []
    runner.on_done = results.append
    runner.submit(Job(argv=["report", "g.am"], mode="inproc"))
    _wait(runner)
    assert results[0].returncode == 3 and not results[0].ok


def test_a_raise_is_reported_rather_than_killing_the_window():
    def dispatch(argv):
        raise RuntimeError("the graph is not ASCII")

    runner = _runner(dispatch)
    results = []
    runner.on_done = results.append
    runner.submit(Job(argv=["report", "g.am"], mode="inproc"))
    _wait(runner)
    assert isinstance(results[0].error, RuntimeError)
    assert not results[0].ok
    assert "RuntimeError" in results[0].traceback
    assert any("RuntimeError" in line for line in runner.lines)


def test_argparse_exiting_becomes_a_return_code():
    """Bad argv makes argparse call sys.exit(2) rather than raising."""
    def dispatch(argv):
        raise SystemExit(2)

    runner = _runner(dispatch)
    results = []
    runner.on_done = results.append
    runner.submit(Job(argv=["nope"], mode="inproc"))
    _wait(runner)
    assert results[0].returncode == 2 and results[0].error is None


def test_jobs_run_one_at_a_time_in_order():
    order = []

    def dispatch(argv):
        order.append(argv[0])
        return 0

    runner = _runner(dispatch)
    for name in ("report", "gaps", "connect"):
        runner.submit(Job(argv=[name, "g.am"], mode="inproc"))
    _wait(runner)
    assert order == ["report", "gaps", "connect"]


def test_a_second_job_waits_rather_than_being_dropped():
    """Unlike RebuildQueue, nothing is coalesced away -- each job writes a file."""
    runner = _runner(lambda argv: 0)
    runner._current = Job(argv=["busy"])  # pretend one is running
    runner.submit(Job(argv=["report", "g.am"], mode="inproc"))
    runner.submit(Job(argv=["gaps", "g.am"], mode="inproc"))
    assert [j.command for j in runner.queued] == ["report", "gaps"]


def test_clear_queue_leaves_the_running_job_alone():
    runner = _runner(lambda argv: 0)
    runner._current = Job(argv=["busy"])
    runner.submit(Job(argv=["report", "g.am"], mode="inproc"))
    assert runner.clear_queue() == 1
    assert runner.queued == () and runner.busy


def test_stop_marks_the_outcome_cancelled_so_a_chain_halts():
    """It cannot interrupt the work; it must still stop the workflow."""
    def dispatch(argv):
        runner.stop()
        return 0

    runner = _runner(lambda argv: dispatch(argv))
    results = []
    runner.on_done = results.append
    runner.submit(Job(argv=["report", "g.am"], mode="inproc"))
    _wait(runner)
    assert results[0].cancelled and not results[0].ok


def test_outputs_are_reported_only_when_they_exist(tmp_path):
    """`repair-radius` returns 0 and writes nothing when it finds no spans."""
    written = tmp_path / "made.am"
    missing = tmp_path / "not_made.am"

    def dispatch(argv):
        written.write_text("x")
        return 0

    runner = _runner(dispatch)
    results = []
    runner.on_done = results.append
    runner.submit(Job(argv=["gaps", "g.am"], mode="inproc",
                      outputs=(str(written), str(missing))))
    _wait(runner)
    assert [p.name for p in results[0].outputs_written] == ["made.am"]


def test_the_queue_callback_fires_on_submit_and_on_finish():
    runner = _runner(lambda argv: 0)
    ticks = []
    runner.on_queue = lambda: ticks.append(1)
    runner.submit(Job(argv=["report", "g.am"], mode="inproc"))
    _wait(runner)
    assert len(ticks) >= 2


def test_on_started_receives_the_job():
    runner = _runner(lambda argv: 0)
    started = []
    runner.on_started = started.append
    job = Job(argv=["report", "g.am"], mode="inproc")
    runner.submit(job)
    _wait(runner)
    assert started == [job]


# -------------------------------------------------------------- descriptions


def test_a_result_describes_what_it_wrote(tmp_path):
    out = tmp_path / "step1.am"
    out.write_text("x")
    result = JobResult(job=Job(argv=["gaps"], label="gaps"), seconds=1.25,
                       outputs_written=(out,))
    assert "gaps: done in 1.2s -> step1.am" == result.describe()


def test_a_cancelled_result_says_so():
    result = JobResult(job=Job(argv=["skeletonise"], label="skeletonise"),
                       cancelled=True, seconds=3.0)
    assert "stopped after 3.0s" in result.describe()


def test_a_job_falls_back_to_its_argv_for_a_title():
    assert Job(argv=["report", "g.am"]).title == "report g.am"
