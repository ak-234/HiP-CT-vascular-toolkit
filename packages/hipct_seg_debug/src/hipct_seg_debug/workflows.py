"""The twelve workflows from CLI.md, as data a button can run.

Five of them are command chains and can genuinely be one click. The other six are
"open the viewer and look", which no runner can perform on your behalf -- so they are
`Setup` entries that configure the live session and tell you what to do next, and
they are labelled as such rather than dressed up as commands.

Two design points carry the weight:

**Steps hand filenames forward by lookup, not by scraping the log.** The driver keeps
each step's resolved values, so `feeds={"graph": "out"}` is a dict access. Parsing
"wrote step1.am" out of stdout would be a second, worse copy of information the
driver already has.

**Every declared output is checked before the next step starts.** `repair-radius`
returns 0 and writes nothing when it finds no collapsed spans (`__main__.py:406-408`),
so a chain that trusted the exit code would feed a file that does not exist to
`surface` and fail three minutes later with "file not found". Stopping at the real
cause, with the real reason, is the difference between a chain you trust and one you
stop using.

Pure data and a driver; no Qt, so `test_workflows.py` can check every step against
argparse without a display.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from . import cliform
from .runner import Job


@dataclass(frozen=True)
class Step:
    """One command in a chain.

    ``values`` may contain ``{run}``, ``{graph}``, ``{seg}`` and ``{edits}``, filled
    from the context when the chain starts. ``feeds`` maps one of *this* step's dests
    to a dest of the *previous* step, which is how step 2 learns what step 1 wrote.
    """

    command: str
    values: dict[str, Any] = field(default_factory=dict)
    feeds: dict[str, str] | None = None
    note: str = ""
    pause: bool = False
    outputs: tuple[str, ...] = ("out",)
    executable: str = ""

    @property
    def is_pause(self) -> bool:
        """A step with no command is a hand-off to the user, not a no-op."""
        return not self.command


@dataclass(frozen=True)
class Workflow:
    key: str
    title: str
    doc: str
    steps: tuple[Step, ...]


@dataclass(frozen=True)
class Setup:
    """A workflow you drive yourself, once the session is configured for it."""

    key: str
    title: str
    doc: str
    action: str  # "run" | "enable" | "reload"
    args: dict[str, Any] = field(default_factory=dict)
    requires: tuple[str, ...] = ()
    next_steps: str = ""


# --------------------------------------------------------------- the chains

_W7 = (
    Step("skeletonise", {"stride": 4, "out": "{run}/quick.am"},
         note="a strided sanity run first - 10 s, and it proves the geometry is right"),
    Step("skeletonise", {"order": True, "out": "{run}/candidate.am"},
         note="the real one: ~12 minutes at stride 1"),
    Step("optimise", {"sensitivity": True}, feeds={"graph": "out"}, outputs=(),
         note="scores it against the Avizo skeleton; writes nothing without --out"),
)

_W8 = (
    Step("report", {"graph": "{graph}"}, outputs=(),
         note="what is wrong before anything changes"),
    # First, because every step after it reads the field and treats an unflagged graph
    # as clean. Flagging changes what `connect` is willing to cut and what
    # `repair-radius` is willing to believe.
    Step("flag-interpolation", {"graph": "{graph}", "show_spans": True,
                                "out": "{run}/step0.am"},
         note="what Avizo interpolated across a gap, rather than measured"),
    Step("gaps", {"out": "{run}/step1.am"}, feeds={"graph": "out"}),
    Step("connect", {"tjunction": True, "show_rejected": True, "out": "{run}/step2.am"},
         feeds={"graph": "out"}),
    Step("repair-radius", {"source": "both", "out": "{run}/step3.am"},
         feeds={"graph": "out"},
         note="writes nothing if it finds no collapsed spans - that is a pass, not a failure"),
)

_W9 = (
    Step("repair-mask", {"min_voxels": 500, "close": 1}, outputs=(),
         note="dry run: read the morphometrics before writing anything"),
    Step("", pause=True,
         note="A closing that mends a real break lowers the component count and leaves\n"
              "Euler alone; one that welds two unrelated vessels lowers both. Read the\n"
              "table above, then continue to write the mask."),
    Step("repair-mask", {"min_voxels": 500, "close": 1, "out": "{run}/mask.tif"}),
    Step("mask-export", {"edits": "{edits}", "out": "{run}/corrected.am"},
         note="the source lattice plus your painted corrections, as a drop-in .am"),
)

_W10 = (
    Step("surface", {"graph": "{graph}", "out_dir": "{run}/surfaces"},
         outputs=("out_dir",),
         note="5-10 minutes. The STL always comes from a full run, never from the\n"
              "preview patches - those are spliced for display and not welded at the seams."),
)

_W8_CFC = (
    Step("report", {"graph": "{graph}"}, outputs=()),
    Step("flag-interpolation", {"graph": "{graph}", "out": "{run}/step0-cfc.am"}),
    Step("gaps", {"graph": "{run}/step0-cfc.am", "out": "{run}/step1-cfc.am"}),
    Step(
        "connect",
        {
            "graph": "{run}/step1-cfc.am", "tjunction": True, "dpc": True,
            "raw": "{raw}", "seg": "{seg}", "cfc_model": "{cfc_model}",
            "show_rejected": True, "out": "{run}/step2-cfc.am",
        },
        executable="{cfc_python}",
        note="runs the persisted DF21 model under the dedicated Python 3.9 interpreter",
    ),
    Step("repair-radius", {"graph": "{run}/step2-cfc.am", "source": "both",
                           "out": "{run}/step3-cfc.am"}),
)

_W12 = (
    Step("skeletonise-all", {"stride": 4, "algorithms": "lee",
                             "out_dir": "{run}/quick"},
         outputs=("out_dir",),
         note="a strided sanity run first - 1 minute, and it proves the geometry is "
              "wired up before anything expensive starts. Its ranking is indicative "
              "only: at 264 um voxels a vessel is 1-2 voxels across"),
    Step("skeletonise-all", {"stride": 1, "algorithms": "lee",
                             "out_dir": "{run}/candidates"},
         outputs=("out_dir",),
         note="the ranking that counts, at the resolution everything below uses. "
              "~25 min: Lee at full resolution, plus the scoring context, which "
              "costs a second skeletonisation of the mask to find its junctions. "
              "Add teasar once kimimaro is installed, and amira to score an "
              "existing Avizo export alongside"),
    # Named by path rather than fed, because `skeletonise-all` declares `out_dir`
    # and `feeds` reads `out`. Feeding would mean running Lee at stride 1 twice.
    Step("pick-roots", {"graph": "{run}/candidates/lee.am",
                        "roots_json": "{run}/roots.json"},
         outputs=("roots_json",),
         note="opens one 3-D window per tree, largest first: click each inlet, q for "
              "the next, x to stop. Recorded by coordinate, so the choice survives "
              "the de-looping and pruning the next step does. --auto records the "
              "automatic roots instead, which is also the headless fallback"),
    Step("optimise-skeleton", {"graph": "{run}/candidates/lee.am",
                               "out": "{run}/refined.am",
                               "roots_json": "{run}/roots.json"},
         note="break the collapse loops, drop leaves shorter than the local vessel "
              "radius, re-centre on the lumen, then smooth. Strahler order is "
              "assigned last, from the chosen roots, because pruning changes what "
              "the tree is"),
    Step("radius-perimeter", {"out": "{run}/radius.am",
                              "roots_json": "{run}/roots.json"},
         feeds={"graph": "out"},
         note="last radius-changing step, after graph/taper repair: every thickness "
              "is replaced from that point's trusted cross-section. The roots decide "
              "parent from daughter at every bifurcation"),
    Step("score", {}, feeds={"graph": "out"}, outputs=(),
         note="the five terms again, to confirm the refinement improved them"),
)


WORKFLOWS: tuple[Workflow, ...] = (
    Workflow(
        "skeletonise-and-score", "7. Derive an independent skeleton and score it",
        "Builds a centreline from the mask with no reference to the Avizo graph, then "
        "scores the two against each other. Expect ~1226 segments against Avizo's 309: "
        "it finds the same vessels and over-branches, which is what the bifurcation "
        "Dice of 0.43 is telling you.",
        _W7,
    ),
    Workflow(
        "repair-graph", "8. Repair a graph, end to end",
        "Fills jumps inside edges, bridges disconnected ends, then restores radii that "
        "the segmentation collapsed. Each step writes a new file, so you can stop and "
        "inspect at any point.",
        _W8,
    ),
    Workflow(
        "repair-graph-cfc", "8C. Repair a graph with CFC-guided DPC",
        "Preserves workflow 8 and replaces only geometric reconnection with the "
        "persisted scan-specific Cascade Forest and sequential Type 1/2/3 DPC walk.",
        _W8_CFC,
    ),
    Workflow(
        "repair-mask", "9. Repair a mask, end to end",
        "Culls debris, closes small breaks, and writes the corrected mask back out as "
        "an Amira lattice. Pauses after the dry run so you can read the morphometrics.",
        _W9,
    ),
    Workflow(
        "surface", "10. Regenerate the surface",
        "One command, 5-10 minutes, producing the STL that goes to CFD.",
        _W10,
    ),
    Workflow(
        "full-run", "11. The full run",
        "Everything, in order: an independent skeleton, the graph repairs, the mask "
        "repairs, a pause for the hand corrections no heuristic should be trusted "
        "with, and finally the surface. Expect the better part of an hour.",
        # Composed from the workflows above rather than restated, so they cannot
        # drift -- with two deliberate differences, both matching CLI.md:
        #  * no strided sanity run (`_W7[1:]`); this is the run you keep;
        #  * the surface is built from the *repaired* graph, not the one you loaded,
        #    which is the whole point of having run the repair chain first. That step
        #    cannot be shared with workflow 10, whose input is whatever you have open.
        _W7[1:] + _W8 + _W9[:1] + (
            Step("", pause=True,
                 note="Now fix by hand what no heuristic should be trusted with.\n"
                      "Turn on painting from the Workflows tab (workflow 6), paint the\n"
                      "collapsed lumen open, Commit, then Re-skeletonise. Continue when done."),
        ) + _W9[2:] + (
            Step("surface", {"graph": "{run}/step3.am", "out_dir": "{run}/surfaces"},
                 outputs=("out_dir",),
                 note="from step3.am - the repaired graph, not the one you loaded"),
        ),
    ),
    Workflow(
        "full-run-cfc", "11C. Full run with CFC-guided DPC",
        "The full pipeline with the persisted CFC model used for graph reconnection.",
        _W7[1:] + _W8_CFC + _W9[:1] + (
            Step("", pause=True,
                 note="Review the CFC reconnections, then perform any necessary hand corrections."),
        ) + _W9[2:] + (
            Step("surface", {"graph": "{run}/step3-cfc.am",
                             "out_dir": "{run}/surfaces-cfc"}, outputs=("out_dir",)),
        ),
    ),
    Workflow(
        "optimised-centreline", "12. Choose a skeletonisation, then measure it",
        "The Walsh-Berg procedure (SKELETONISATION.md): there is no ground-truth "
        "skeleton, so the segmentation is the gold standard and the algorithm is "
        "chosen by how little of it the skeletonisation lost. Runs the algorithms, "
        "scores each on the five-term super metric, then de-loops, prunes the leaves "
        "shorter than the local vessel radius, re-centres on the lumen, and finally "
        "replaces every radius with the perimeter of that point's own cross-section. "
        "That last step is the one that fixes the thickness values: Avizo's come from "
        "a single global fit whose local error runs to a factor of two.",
        _W12,
    ),
)

WORKFLOW_BY_KEY = {w.key: w for w in WORKFLOWS}


# ------------------------------------------------------------ guided setups

SETUPS: tuple[Setup, ...] = (
    Setup(
        "validate", "1. First contact - does this dataset hang together?",
        "Proves the raw stack, graph, lattice and STL share one coordinate frame. "
        "Eight checks, ~15 s. Run it on any new dataset before trusting anything else.",
        action="run", requires=("session",),
        next_steps="If a fatal check fails, the frame is wrong and nothing downstream "
                   "means anything - fix the inputs before going on.",
    ),
    Setup(
        "selftest", "1b. Self-test against the greyscale",
        "21 checks that go past geometry into the image itself: that the mask width "
        "matches 2r, that the vessel wall ring sits just outside the assigned radius, "
        "that the lumen is darker than the tissue around it. ~2 minutes.",
        action="run", requires=("session",),
    ),
    Setup(
        "audit", "2. Audit - what is wrong, and where?",
        "Runs the graph heuristics and the image-based collapse detector, writing "
        "cache/candidates.csv. 249 sites on LADAF-2024-28 in 14.6 s. The results land "
        "in the 3D window, so 'n' and 'b' walk them straight away.",
        action="run", requires=("session",),
        next_steps="Sort the CSV by score, or walk the sites with 'n' / 'b' in the 3D "
                   "window and press 'v' at each to open the slices.",
    ),
    Setup(
        "inspect", "3. Inspect one site, headlessly",
        "The --goto-* flags open one napari window on a single location and block. "
        "In here the equivalent is simply picking the site in 3D and pressing 'v', "
        "which is why this one is a note rather than a button.",
        action="note", requires=(),
        next_steps="Double-click a vessel in the 3D window, then press 'v'.",
    ),
    Setup(
        "volume", "4. Interactive audit with whole-dataset layers",
        "Adds lazy whole-dataset layers so the z slider can leave the slab - worth it "
        "for asking whether a defect continues past the crop. Needs the dataset "
        "reloaded, because the layers are built at load time.",
        action="reload", args={"volume": True}, requires=("session",),
        next_steps="Pick a vessel, press 'v', then drag the z slider past the slab edge.",
    ),
    Setup(
        "edit", "5. Correct the skeleton by hand",
        "Turns on skeleton editing with live SDF regeneration. Each edit rebuilds only "
        "the box it touched, 0.5-0.8 s rather than the 5-10 minutes a full run costs. "
        "Preparing the SDF session takes about 88 s the first time.",
        action="enable", args={"edit": True}, requires=("session", "coronary_sdf"),
    ),
    Setup(
        "paint", "6. Correct the mask by hand",
        "Adds a writable segmentation layer to the slice browser. Paint the collapsed "
        "lumen open, Commit, then Re-skeletonise the painted region: the graph gains a "
        "centreline through what you painted and the surface follows. Corrections go to "
        "the edits file and never into the source lattice.",
        action="enable", args={"paint": True}, requires=("session", "edits"),
        next_steps="Select the 'segmentation (editable)' layer, '2' to paint, '4' to "
                   "erase, then Commit, then Re-skeletonise painted region.",
    ),
)

SETUP_BY_KEY = {s.key: s for s in SETUPS}


# --------------------------------------------------------------- the driver


def resolve(step: Step, context: dict, previous: dict | None) -> dict:
    """Fill a step's values from the context and the previous step's outputs."""
    values = {
        key: value.format(**context) if isinstance(value, str) else value
        for key, value in step.values.items()
    }
    for dest, source in (step.feeds or {}).items():
        if previous is None:
            raise ValueError(f"{step.command}.{dest} feeds from a step that does not exist")
        values[dest] = previous[source]
    return values


def missing_outputs(step: Step, values: dict) -> list[str]:
    """Declared outputs that the step did not actually produce."""
    return [
        str(values[dest]) for dest in step.outputs
        if values.get(dest) and not Path(values[dest]).exists()
    ]


class WorkflowRun:
    """Drive a chain through a `CommandRunner`, stopping at the first real problem.

    A step advances only when the command returned 0 *and* every file it said it
    would write exists. Pause steps stop and wait for `continue_()`.
    """

    def __init__(self, workflow: Workflow, context: dict, runner, specs, on_state=None):
        self.workflow = workflow
        self.context = dict(context)
        self.runner = runner
        self.specs = specs
        self.on_state = on_state
        self.index = 0
        self.previous: dict | None = None
        self.state = "idle"  # idle | running | paused | done | failed
        self.message = ""
        self._results: list = []

    # -- control ----------------------------------------------------------

    def start(self) -> None:
        self.index = 0
        self.previous = None
        self._results.clear()
        self.state = "running"
        self.message = ""
        self._advance()

    def continue_(self) -> None:
        if self.state != "paused":
            return
        self.state = "running"
        self.index += 1
        self._advance()

    def abort(self, message: str = "stopped") -> None:
        self.state = "failed"
        self.message = message
        self._changed()

    # -- the loop ---------------------------------------------------------

    def _advance(self) -> None:
        if self.index >= len(self.workflow.steps):
            self.state = "done"
            self.message = f"{self.workflow.title}: all {self.index} steps finished"
            self._changed()
            return

        step = self.workflow.steps[self.index]
        if step.is_pause:
            self.state = "paused"
            self.message = step.note
            self._changed()
            return

        try:
            values = resolve(step, self.context, self.previous)
        except (KeyError, ValueError) as exc:
            self.abort(f"step {self.index + 1} ({step.command}): {exc}")
            return

        self._pending = (step, values)
        self._changed()
        self.runner.submit(self.job_for(step, values))

    def job_for(self, step: Step, values: dict) -> Job:
        spec = self.specs[step.command]
        full = {**cliform.defaults(spec), **values}
        outputs = tuple(str(full[d]) for d in step.outputs if full.get(d))
        return Job(
            argv=cliform.to_argv(spec, full),
            label=f"{self.workflow.key} step {self.index + 1}: {step.command}",
            outputs=outputs,
            tag=(self.workflow.key, self.index),
            executable=(step.executable.format(**self.context)
                        if step.executable else ""),
        )

    def on_job_done(self, result) -> None:
        """Subscribe this to `runner.on_done`; it ignores jobs that are not ours."""
        if self.state != "running" or result.job.tag != (self.workflow.key, self.index):
            return
        step, values = self._pending
        self._results.append(result)

        if result.cancelled:
            self.abort(f"stopped during step {self.index + 1} ({step.command})")
            return
        if not result.ok:
            self.abort(f"step {self.index + 1} ({step.command}) failed - the chain stops here")
            return

        missing = missing_outputs(step, values)
        if missing:
            self.abort(self._explain_missing(step, missing))
            return

        self.previous = values
        self.index += 1
        self._advance()

    def _explain_missing(self, step: Step, missing: list[str]) -> str:
        """Say why a step that succeeded still cannot be built on.

        The common case is `repair-radius` finding nothing to repair, which is a
        clean pass with no output file. Reporting that as "file not found" three
        steps later is how a chain loses its users.
        """
        latest = self.previous.get("out") if self.previous else None
        tail = f" Your latest graph is still {latest}." if latest else ""
        if step.command == "repair-radius":
            return ("repair-radius found no collapsed spans, so "
                    f"{missing[0]} was not written. Nothing is wrong - there was "
                    f"nothing to fix.{tail}")
        return (f"{step.command} returned 0 but did not write {missing[0]}, "
                f"so the chain stops here.{tail}")

    def _changed(self) -> None:
        if self.on_state is not None:
            self.on_state(self)

    # -- reporting --------------------------------------------------------

    @property
    def total(self) -> int:
        return len(self.workflow.steps)

    def describe(self) -> str:
        if self.state == "idle":
            return f"{self.total} steps"
        if self.state in ("done", "failed"):
            return self.message
        return f"step {min(self.index + 1, self.total)} of {self.total}"


def default_context(session_args, run_dir) -> dict:
    """The substitutions a chain starts with, taken from the loaded dataset."""
    from .main import graph_paths

    return {
        "run": str(run_dir),
        # The *first* skeleton, when several are loaded. Every step in every chain
        # takes exactly one graph, and `{graph}` interpolates into a command line --
        # a list would render as its repr and name no file at all.
        "graph": (graph_paths(getattr(session_args, "graph", None)) or [""])[0],
        "seg": getattr(session_args, "seg", "") or "",
        "edits": getattr(session_args, "edits", "") or "",
        "raw": getattr(session_args, "raw", "") or "",
        "cfc_model": getattr(session_args, "cfc_model", "")
                     or os.environ.get("HIPCT_CFC_MODEL", ""),
        "cfc_python": getattr(session_args, "cfc_python", "")
                      or os.environ.get("HIPCT_CFC_PYTHON", ""),
    }
