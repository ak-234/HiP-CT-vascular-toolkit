"""Every step of every chain, checked against argparse.

A typo in a workflow's flag name is otherwise found twenty minutes into a run, by
argparse, in a subprocess. `test_every_step_parses` finds it in a millisecond.

The driver tests use a fake runner rather than a real one: what is being checked is
the advance/stop logic, and in particular that a step which returns 0 without writing
its output stops the chain with an explanation rather than poisoning the next step.
"""

from __future__ import annotations

import pytest

from hipct_seg_debug import cliform, workflows
from hipct_seg_debug.edit.__main__ import build_parser as edit_parser
from hipct_seg_debug.runner import Job, JobResult
from hipct_seg_debug.workflows import Step, Workflow, WorkflowRun, resolve

SPECS = {s.name: s for s in cliform.describe_parser(edit_parser())}
CONTEXT = {
    "run": "R", "graph": "G.am", "seg": "S.am", "edits": "E.npz",
    "raw": "RAW", "cfc_model": "MODEL", "cfc_python": "python39.exe",
}


def _steps():
    for workflow in workflows.WORKFLOWS:
        for i, step in enumerate(workflow.steps):
            yield workflow, i, step


# --------------------------------------------------------------- integrity


def test_every_command_named_by_a_step_exists():
    unknown = [(w.key, s.command) for w, _i, s in _steps()
               if not s.is_pause and s.command not in SPECS]
    assert unknown == []


def test_every_value_names_a_real_flag():
    """The typo check. A dest that argparse does not know would be silently dropped."""
    bad = [
        f"{w.key}[{i}] {s.command}.{dest}"
        for w, i, s in _steps() if not s.is_pause
        for dest in s.values
        if dest not in SPECS[s.command].dests
    ]
    assert bad == []


def test_every_feed_names_a_real_dest_on_both_sides():
    bad = []
    for workflow in workflows.WORKFLOWS:
        for i, step in enumerate(workflow.steps):
            for dest, source in (step.feeds or {}).items():
                if step.is_pause:
                    bad.append(f"{workflow.key}[{i}] a pause step cannot feed")
                    continue
                if dest not in SPECS[step.command].dests:
                    bad.append(f"{workflow.key}[{i}] {step.command}.{dest}")
                previous = workflow.steps[i - 1] if i else None
                if previous is None or previous.is_pause:
                    bad.append(f"{workflow.key}[{i}] feeds from nothing")
                elif source not in SPECS[previous.command].dests:
                    bad.append(f"{workflow.key}[{i}] <- {previous.command}.{source}")
    assert bad == []


def test_every_declared_output_is_a_real_dest():
    bad = [f"{w.key}[{i}] {s.command}.{dest}" for w, i, s in _steps()
           if not s.is_pause for dest in s.outputs
           if dest not in SPECS[s.command].dests]
    assert bad == []


def test_every_step_parses():
    """The end of the line: build each step's argv and hand it to argparse."""
    for workflow in workflows.WORKFLOWS:
        previous = None
        for i, step in enumerate(workflow.steps):
            if step.is_pause:
                continue
            values = resolve(step, CONTEXT, previous)
            spec = SPECS[step.command]
            full = {**cliform.defaults(spec), **values}
            if spec.field("graph") is not None and not full.get("graph"):
                full["graph"] = "G.am"
            argv = cliform.to_argv(spec, full)
            parsed = edit_parser().parse_args(argv)
            assert parsed.command == step.command, f"{workflow.key}[{i}]"
            previous = full


def test_a_pause_step_carries_an_explanation():
    """A chain that stops without saying why is worse than one that does not stop."""
    for workflow in workflows.WORKFLOWS:
        for step in workflow.steps:
            if step.is_pause:
                assert step.note.strip(), workflow.key


def test_workflow_keys_are_unique():
    keys = [w.key for w in workflows.WORKFLOWS]
    assert len(keys) == len(set(keys))


def test_the_full_run_is_composed_from_the_others():
    """Restating them would let workflow 11 drift from workflows 7-9."""
    full = workflows.WORKFLOW_BY_KEY["full-run"].steps
    for key in ("skeletonise-and-score", "repair-graph"):
        part = workflows.WORKFLOW_BY_KEY[key].steps
        # Workflow 7's strided sanity run is dropped from the full run; the rest of
        # each chain must appear verbatim.
        part = part[1:] if key == "skeletonise-and-score" else part
        assert any(full[i:i + len(part)] == part for i in range(len(full))), key


def test_the_full_run_surfaces_the_repaired_graph_not_the_loaded_one():
    """The one step workflow 11 cannot share with workflow 10.

    Building the surface from the graph you loaded would silently discard the whole
    repair chain that just ran, and the STL would look entirely plausible.
    """
    full = workflows.WORKFLOW_BY_KEY["full-run"].steps
    surface = [s for s in full if s.command == "surface"]
    assert len(surface) == 1
    assert surface[0].values["graph"] == "{run}/step3.am"

    repair = workflows.WORKFLOW_BY_KEY["repair-graph"].steps[-1]
    assert repair.command == "repair-radius" and repair.values["out"] == "{run}/step3.am"


def test_the_full_run_has_exactly_one_pause():
    full = workflows.WORKFLOW_BY_KEY["full-run"].steps
    assert sum(1 for s in full if s.is_pause) == 1


def test_the_setups_cover_workflows_one_to_six():
    assert len(workflows.SETUPS) == 7  # 1, 1b, 2, 3, 4, 5, 6
    assert {s.action for s in workflows.SETUPS} <= {"run", "enable", "reload", "note"}


def test_every_setup_has_a_key_and_a_doc():
    for setup in workflows.SETUPS:
        assert setup.key and setup.doc.strip()


def test_setup_args_name_real_viewer_flags():
    from hipct_seg_debug.main import build_parser as viewer_parser

    dests = set(cliform.describe_parser(viewer_parser())[0].dests)
    bad = [f"{s.key}.{d}" for s in workflows.SETUPS for d in s.args if d not in dests]
    assert bad == []


# ---------------------------------------------------------------- resolve


def test_templates_are_filled_from_the_context():
    step = Step("gaps", {"graph": "{graph}", "out": "{run}/step1.am"})
    assert resolve(step, CONTEXT, None) == {"graph": "G.am", "out": "R/step1.am"}


def test_a_feed_takes_the_previous_steps_value():
    step = Step("connect", {"out": "b.am"}, feeds={"graph": "out"})
    assert resolve(step, CONTEXT, {"out": "a.am"})["graph"] == "a.am"


def test_a_feed_with_no_previous_step_is_refused():
    with pytest.raises(ValueError, match="does not exist"):
        resolve(Step("connect", {}, feeds={"graph": "out"}), CONTEXT, None)


def test_non_string_values_pass_through_untouched():
    step = Step("skeletonise", {"stride": 4, "order": True})
    assert resolve(step, CONTEXT, None) == {"stride": 4, "order": True}


# ----------------------------------------------------------------- driver


class FakeRunner:
    """Records jobs instead of running them, so the driver can be stepped by hand."""

    def __init__(self):
        self.jobs = []

    def submit(self, job):
        self.jobs.append(job)


def _run(steps, tmp_path, **kw):
    workflow = Workflow("test", "test", "", tuple(steps))
    runner = FakeRunner()
    run = WorkflowRun(workflow, {**CONTEXT, "run": str(tmp_path)}, runner, SPECS, **kw)
    return run, runner


def _succeed(run, runner, *, write=None):
    """Finish the job the driver last submitted, optionally creating its output."""
    job = runner.jobs[-1]
    for path in write or ():
        path.write_text("x")
    run.on_job_done(JobResult(job=job, returncode=0))


def test_a_chain_advances_through_its_steps(tmp_path):
    a, b = tmp_path / "a.am", tmp_path / "b.am"
    run, runner = _run([
        Step("gaps", {"graph": "{graph}", "out": str(a)}),
        Step("connect", {"out": str(b)}, feeds={"graph": "out"}),
    ], tmp_path)
    run.start()
    assert len(runner.jobs) == 1
    _succeed(run, runner, write=[a])
    assert len(runner.jobs) == 2
    assert "--graph" not in " ".join(runner.jobs[1].argv)  # graph is positional
    assert str(a) in runner.jobs[1].argv
    _succeed(run, runner, write=[b])
    assert run.state == "done"


def test_a_failing_step_stops_the_chain(tmp_path):
    run, runner = _run([
        Step("gaps", {"graph": "{graph}", "out": str(tmp_path / "a.am")}),
        Step("connect", {"out": "b.am"}, feeds={"graph": "out"}),
    ], tmp_path)
    run.start()
    run.on_job_done(JobResult(job=runner.jobs[-1], returncode=1))
    assert run.state == "failed" and len(runner.jobs) == 1
    assert "step 1" in run.message


def test_a_cancelled_step_stops_the_chain(tmp_path):
    run, runner = _run([Step("gaps", {"graph": "{graph}", "out": "a.am"})], tmp_path)
    run.start()
    run.on_job_done(JobResult(job=runner.jobs[-1], returncode=0, cancelled=True))
    assert run.state == "failed" and "stopped" in run.message


def test_a_step_that_writes_nothing_stops_the_chain_with_a_reason(tmp_path):
    """repair-radius returns 0 and writes nothing when it finds no spans."""
    a = tmp_path / "step1.am"
    run, runner = _run([
        Step("gaps", {"graph": "{graph}", "out": str(a)}),
        Step("repair-radius", {"out": str(tmp_path / "step3.am")}, feeds={"graph": "out"}),
        Step("surface", {"out_dir": "s"}, feeds={"graph": "out"}, outputs=()),
    ], tmp_path)
    run.start()
    _succeed(run, runner, write=[a])
    _succeed(run, runner)  # returns 0, writes nothing
    assert run.state == "failed"
    assert "found no collapsed spans" in run.message
    assert str(a) in run.message, "it must name the graph that is still current"
    assert len(runner.jobs) == 2, "surface must not have been submitted"


def test_a_pause_step_waits_and_then_continues(tmp_path):
    a = tmp_path / "a.am"
    run, runner = _run([
        Step("gaps", {"graph": "{graph}", "out": str(a)}),
        Step("", pause=True, note="read the morphometrics"),
        Step("report", {"graph": "{graph}"}, outputs=()),
    ], tmp_path)
    run.start()
    _succeed(run, runner, write=[a])
    assert run.state == "paused" and run.message == "read the morphometrics"
    assert len(runner.jobs) == 1
    run.continue_()
    assert run.state == "running" and len(runner.jobs) == 2


def test_continuing_when_not_paused_does_nothing(tmp_path):
    run, runner = _run([Step("report", {"graph": "{graph}"}, outputs=())], tmp_path)
    run.start()
    run.continue_()
    assert len(runner.jobs) == 1


def test_a_result_from_another_job_is_ignored(tmp_path):
    run, runner = _run([Step("gaps", {"graph": "{graph}", "out": "a.am"})], tmp_path)
    run.start()
    run.on_job_done(JobResult(job=Job(argv=["report"], tag=("someone-else", 0))))
    assert run.state == "running"


def test_the_state_callback_fires_on_every_transition(tmp_path):
    seen = []
    a = tmp_path / "a.am"
    run, runner = _run([Step("gaps", {"graph": "{graph}", "out": str(a)})],
                       tmp_path, on_state=lambda r: seen.append(r.state))
    run.start()
    _succeed(run, runner, write=[a])
    assert seen[-1] == "done"


def test_the_job_carries_its_outputs_so_the_runner_can_check_them(tmp_path):
    out = tmp_path / "a.am"
    run, runner = _run([Step("gaps", {"graph": "{graph}", "out": str(out)})], tmp_path)
    run.start()
    assert runner.jobs[0].outputs == (str(out),)


def test_describe_reports_progress(tmp_path):
    a = tmp_path / "a.am"
    run, runner = _run([
        Step("gaps", {"graph": "{graph}", "out": str(a)}),
        Step("report", {"graph": "{graph}"}, outputs=()),
    ], tmp_path)
    assert run.describe() == "2 steps"
    run.start()
    assert run.describe() == "step 1 of 2"


# ---------------------------------------------------------------- context


def test_the_default_context_comes_from_the_loaded_dataset(tmp_path):
    import argparse

    args = argparse.Namespace(graph="G.am", seg="S.am", edits=None)
    context = workflows.default_context(args, tmp_path)
    assert context["graph"] == "G.am" and context["edits"] == ""
    assert context["run"] == str(tmp_path)
