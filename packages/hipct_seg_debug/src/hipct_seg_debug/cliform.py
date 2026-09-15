"""Turn an argparse parser into a form, and a form back into argv.

The command line is the only complete description of what this package can do:
two parsers, fifteen commands, 67 flags, each with its own help string and its own
default. Writing a second description of that in Qt would guarantee the two drift --
which is the exact failure `docs/CLI.md` was written to stop, and it would reappear here
one flag at a time.

So nothing is described twice. `describe_parser` walks the parser and every widget,
label, tooltip, range and default comes from the action it was generated from. Add a
flag to `edit/__main__.py` and it shows up in the panel with its help text attached.

The load-bearing rule is in `to_argv`: **a default of None means "omit the flag"**,
not "pass None". Twenty-six of the sixty-two fields on the `edit` parser are that
shape, and the handlers behind them -- `cmd_connect:207-215`, `cmd_repair_radius:382-411`
-- build their keyword dicts by *skipping* the Nones. So `--cone-deg 0.0` is not a
verbose way of saying nothing: it overrides the library's own default with zero. A
form that cannot express "unset" would silently change what half these commands do.

The Qt half is at the bottom and imports qtpy inside the builder, matching
`controls3d.py`. Everything above it is pure, which is what lets the argv round trip
be tested against `parse_args` with no display and no dataset -- see
`tests/test_cliform.py`.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Field kinds. These name what the *widget* has to be, which is a coarser question
# than the action's Python type -- `--source` is a str with choices, and wants a
# combo box rather than the line edit its type alone would suggest.
FLAG = "flag"
INT = "int"
FLOAT = "float"
TEXT = "text"
CHOICE = "choice"

# A spin box has to represent "unset" as a value, because Qt has no null state.
# Nothing in this CLI is legitimately near minus a billion, so the bottom of the
# range is free to mean it -- that is what `setSpecialValueText` is for.
UNSET = -1_000_000_000


@dataclass(frozen=True)
class FieldSpec:
    """One argparse action, in the terms a form needs."""

    dest: str
    flag: str | None  # None for a positional
    kind: str
    default: Any
    required: bool
    choices: tuple[str, ...] | None
    nargs: int
    help: str
    path_role: str | None
    file_filter: str
    repeatable: bool = False

    @property
    def positional(self) -> bool:
        return self.flag is None

    @property
    def label(self) -> str:
        """What to call it on screen: the flag, so the panel teaches the CLI."""
        return self.dest if self.positional else self.flag

    @property
    def optional_input(self) -> bool:
        """A path whose absence is a warning rather than an error.

        `main.py:182-183` continues without a surface, and `--edits` names a file
        that a painting session is about to *create*. Marking either missing-file
        red would be wrong.
        """
        return self.dest in ("surface", "edits")


@dataclass(frozen=True)
class CommandSpec:
    """One subcommand, or the whole parser when it has no subcommands."""

    name: str
    help: str
    fields: tuple[FieldSpec, ...]

    def field(self, dest: str) -> FieldSpec | None:
        for f in self.fields:
            if f.dest == dest:
                return f
        return None

    @property
    def dests(self) -> tuple[str, ...]:
        return tuple(f.dest for f in self.fields)


# ------------------------------------------------------------------ path roles

# Keyed by (command, dest) with a (None, dest) fallback. A heuristic on the name
# would miss `--out-dir` entirely and would stop working silently the first time a
# flag is renamed; this table fails loudly instead, because `test_path_roles_cover_
# every_path_flag` checks both directions.
_ANY = None

PATH_ROLES: dict[tuple[str | None, str], tuple[str, str]] = {
    (_ANY, "report_json"): ("save_file", "JSON report (*.json)"),
    (_ANY, "graph"): ("open_file", "Amira spatial graph (*.am)"),
    (_ANY, "reference"): ("open_file", "Amira spatial graph (*.am)"),
    (_ANY, "seg"): ("open_file", "Amira label lattice (*.am)"),
    (_ANY, "surface"): ("open_file", "Surface mesh (*.stl)"),
    (_ANY, "raw"): ("dir", ""),
    (_ANY, "cfc_model"): ("dir", ""),
    (_ANY, "cfc_python"): ("open_file", "Python executable (python.exe)"),
    (_ANY, "regions"): ("open_file", "DPC region manifest (*.json)"),
    (_ANY, "cache"): ("dir", ""),
    # Loaded if it exists (`main.py:159`) and written on exit (`main.py:415`), so
    # the panel offers both an Open and a Save button for this one.
    (_ANY, "edits"): ("open_file", "Mask edit store (*.npz)"),
    ("surface", "out_dir"): ("save_dir", ""),
    ("skeletonise-all", "out_dir"): ("save_dir", ""),
    ("skeletonise-all", "amira_graph"): ("open_file", "Amira spatial graph (*.am)"),
    ("train-cfc", "out_model"): ("save_dir", ""),
    ("evaluate-dpc", "output"): ("save_file", "JSON report (*.json)"),
    # Read if it exists and rewritten every run, like `edits` -- but there is only one
    # button here, because a crop sidecar names itself rather than the graph it crops.
    ("crop", "crop_json"): ("save_file", "Crop sidecar (*.json)"),
    ("crop", "report_csv"): ("save_file", "CSV (*.csv)"),
    # Same shape as `crop_json`: read if it exists, rewritten by `pick-roots`, and it
    # names itself rather than the graph it roots.
    (_ANY, "roots_json"): ("save_file", "Roots sidecar (*.json)"),
    ("pick-roots", "screenshot"): ("save_file", "PNG image (*.png)"),
    ("export-dpc-regions", "output"): ("save_dir", ""),
    ("repair-mask", "out"): ("save_file", "TIFF stack (*.tif)"),
    # `cmd_mask_export:476` branches on the suffix, so both are real choices.
    ("mask-export", "out"): ("save_file", "Amira lattice (*.am);;TIFF stack (*.tif)"),
    (_ANY, "out"): ("save_file", "Amira spatial graph (*.am)"),
}

#: Roles whose target must already exist for the command to run.
INPUT_ROLES = ("open_file", "dir")


def path_role(command: str, dest: str) -> tuple[str | None, str]:
    """Return ``(role, file_filter)`` for a dest, or ``(None, "")``."""
    hit = PATH_ROLES.get((command, dest)) or PATH_ROLES.get((_ANY, dest))
    return hit if hit else (None, "")


# ------------------------------------------------------------- parser -> specs


def _kind(action) -> str:
    if isinstance(action, (argparse._StoreTrueAction, argparse._StoreFalseAction)):
        return FLAG
    if action.choices is not None:
        return CHOICE
    if action.type is int:
        return INT
    if action.type is float:
        return FLOAT
    return TEXT


def _fields(parser: argparse.ArgumentParser, command: str) -> tuple[FieldSpec, ...]:
    out = []
    for action in parser._actions:
        # `-h` and anything else that opts out of the namespace. Filtering on
        # SUPPRESS rather than on isinstance(_HelpAction) also catches a future
        # `--version`.
        if action.default is argparse.SUPPRESS:
            continue
        if isinstance(action, argparse._SubParsersAction):
            continue
        flag = max(action.option_strings, key=len) if action.option_strings else None
        role, filt = path_role(command, action.dest)
        nargs = action.nargs if isinstance(action.nargs, int) else 1
        # A variadic positional (`nargs="+"`, `pick-roots GRAPH [GRAPH ...]`) is the
        # same shape to a form as an `append` flag: a list of values, of unknown
        # length. Calling it `repeatable` rather than adding a third case keeps
        # `to_argv`, `is_unset` and the widgets on one code path.
        variadic = action.nargs in ("+", "*") and not action.option_strings
        out.append(
            FieldSpec(
                dest=action.dest,
                flag=flag,
                kind=_kind(action),
                default=action.default,
                required=bool(action.required),
                choices=tuple(action.choices) if action.choices else None,
                nargs=max(nargs, 1),
                help=action.help or "",
                path_role=role,
                file_filter=filt,
                repeatable=isinstance(action, argparse._AppendAction) or variadic,
            )
        )
    return tuple(out)


def describe_parser(parser: argparse.ArgumentParser) -> list[CommandSpec]:
    """Describe every subcommand of ``parser``, in declaration order.

    A parser with no subcommands -- the viewer's -- describes as a single unnamed
    command, so callers do not need two code paths.
    """
    subs = [a for a in parser._actions if isinstance(a, argparse._SubParsersAction)]
    if not subs:
        return [CommandSpec(name="", help=parser.description or "", fields=_fields(parser, ""))]

    sub = subs[0]
    helps = {c.dest: c.help for c in sub._choices_actions}
    return [
        CommandSpec(name=name, help=helps.get(name, "") or "", fields=_fields(p, name))
        for name, p in sub.choices.items()
    ]


# ------------------------------------------------------------- specs -> argv


def _text(value: Any, kind: str) -> str:
    """Render one value as a single argv token.

    Floats go through ``repr`` rather than ``str(round(...))`` so 0.6 survives as
    "0.6" instead of arriving as 0.60000000000000009 -- which would then be echoed
    into the copyable command line and look like a bug in the tool.
    """
    if kind == FLOAT:
        return repr(float(value))
    if kind == INT:
        return str(int(value))
    return str(value)


def repeat_values(f: FieldSpec, value) -> list:
    """`value` as the list a repeatable field holds, whatever it arrived as.

    A bare string is one value, not a string to iterate: joining `"g.am"` as if it
    were a sequence produced `"g . a m"`, which read back as four files.
    """
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def repeat_separator(f: FieldSpec) -> str | None:
    """How several values are separated in one line edit; None means whitespace.

    Paths take ``;``. A Windows path routinely contains spaces, and splitting
    ``D:/data dir/a b.am`` on whitespace invents three files that do not exist.
    Everything else -- segment ids, mostly -- keeps the spaces-and-commas it had.
    """
    return ";" if f.path_role else None


def split_repeat(f: FieldSpec, text: str) -> list:
    sep = repeat_separator(f)
    parts = text.split(sep) if sep else text.replace(",", " ").split()
    return [p.strip() for p in parts if p.strip()]


def join_repeat(f: FieldSpec, value) -> str:
    sep = repeat_separator(f)
    return ("; " if sep else " ").join(str(v) for v in repeat_values(f, value))


def is_unset(spec: FieldSpec, value: Any) -> bool:
    """Is this value the form's way of saying "don't pass the flag"?"""
    if value is None:
        return True
    if spec.repeatable:
        return not bool(value)
    if spec.nargs > 1:
        # A multi-value flag is a list or nothing; there is no sentinel to compare.
        return len(value) != spec.nargs
    if spec.kind in (INT, FLOAT) and float(value) == UNSET:
        return True
    if spec.kind == TEXT and value == "":
        return True
    return False


def to_argv(spec: CommandSpec, values: dict, *, force_all: bool = False) -> list[str]:
    """Build the argv that reproduces ``values``.

    Only what differs from the parser's own defaults is emitted, so the command line
    the panel shows is the short one you would actually have typed. ``force_all``
    overrides that for when you want a fully explicit line to paste into a script.

    Single-token ``--flag=value`` throughout: a negative number passed as two tokens
    is ambiguous with a flag, and `--goto-um -1200 …` would be rejected by argparse.
    """
    argv: list[str] = [spec.name] if spec.name else []
    tail: list[str] = []

    for f in spec.fields:
        value = values.get(f.dest, f.default)

        if f.positional:
            if is_unset(f, value):
                continue
            # A variadic positional emits its values as bare tokens, in order, since
            # there is no flag to repeat -- `pick-roots left.am right.am`.
            if f.repeatable:
                argv.extend(_text(v, f.kind) for v in repeat_values(f, value))
            else:
                argv.append(_text(value, f.kind))
            continue

        if f.kind == FLAG:
            if bool(value) != bool(f.default):
                tail.append(f.flag)
            continue

        if f.repeatable:
            if not is_unset(f, value):
                for item in repeat_values(f, value):
                    tail.append(f"{f.flag}={_text(item, f.kind)}")
            continue

        if is_unset(f, value):
            if f.required:
                # Caught by `errors()` before submission; emitting nothing here
                # keeps argparse's own message ("the following arguments are
                # required") as the single source of truth.
                continue
            continue

        if f.nargs > 1:
            tail.append(f.flag)
            tail.extend(_text(v, f.kind) for v in value)
            continue

        if not force_all and not f.required and _same(value, f.default, f.kind):
            continue
        tail.append(f"{f.flag}={_text(value, f.kind)}")

    return argv + tail


def _same(value: Any, default: Any, kind: str) -> bool:
    if default is None:
        return False
    try:
        if kind in (INT, FLOAT):
            return float(value) == float(default)
    except (TypeError, ValueError):
        return False
    return str(value) == str(default)


def from_namespace(spec: CommandSpec, namespace) -> dict:
    """Read a Namespace back into the value dict a form understands."""
    return {f.dest: getattr(namespace, f.dest, f.default) for f in spec.fields}


def defaults(spec: CommandSpec) -> dict:
    return {f.dest: f.default for f in spec.fields}


def errors(spec: CommandSpec, values: dict) -> list[str]:
    """Reasons this form cannot be run yet, in the order a user would fix them."""
    out = []
    for f in spec.fields:
        value = values.get(f.dest, f.default)
        unset = is_unset(f, value)
        if (f.required or f.positional) and unset:
            out.append(f"{f.label} is required")
            continue
        if unset or f.path_role not in INPUT_ROLES or f.optional_input:
            continue
        # A repeatable path field holds several; each is checked on its own, or the
        # whole list stringifies into one name that never exists.
        for one in (repeat_values(f, value) if f.repeatable else [value]):
            path = Path(str(one))
            if not path.exists():
                out.append(f"{f.label}: {path} does not exist")
            elif f.path_role == "dir" and not path.is_dir():
                out.append(f"{f.label}: {path} is not a directory")
    return out


def command_line(spec: CommandSpec, values: dict, *, prog: str, force_all: bool = False) -> str:
    """The command line as you would type it, for the panel's copy box."""
    parts = [prog, *to_argv(spec, values, force_all=force_all)]
    return " ".join(f'"{p}"' if " " in p and not p.startswith("-") else p for p in parts)


def subprocess_command(argv: list[str], *, module: str = "hipct_seg_debug.edit",
                       executable: str | None = None) -> list[str]:
    """The full command for running ``argv`` as a child process.

    ``-u`` is not optional. A child whose stdout is a pipe block-buffers by default,
    so without it a twelve-minute `skeletonise` prints nothing at all until it exits
    -- which defeats the entire point of streaming the log.
    """
    return [executable or sys.executable, "-u", "-m", module, *argv]


# --------------------------------------------------------------------- the form


def build_command_form(spec: CommandSpec, *, on_changed=None, browse=None):
    """Return the form widget for ``spec``. Docking and layout are the caller's.

    The widget carries `values()`, `set_values()`, `argv()`, `errors()` and
    `refresh()`, matching how `controls3d.build_layer_panel` hangs `refresh` on the
    box it returns.

    ``browse`` is injected rather than imported so the panel owns file-dialog policy
    (last directory, parent window) and this function stays a pure mapping from spec
    to widgets.
    """
    from qtpy.QtWidgets import (
        QCheckBox,
        QComboBox,
        QDoubleSpinBox,
        QFormLayout,
        QHBoxLayout,
        QLabel,
        QLineEdit,
        QPushButton,
        QSpinBox,
        QVBoxLayout,
        QWidget,
    )

    box = QWidget()
    outer = QVBoxLayout(box)
    outer.setContentsMargins(6, 6, 6, 6)

    if spec.help:
        blurb = QLabel(spec.help)
        blurb.setWordWrap(True)
        blurb.setStyleSheet("color: #808090;")
        outer.addWidget(blurb)

    form = QFormLayout()
    form.setLabelAlignment(_right_align())
    outer.addLayout(form)
    widgets: dict[str, Any] = {}

    for f in spec.fields:
        widget = _make_widget(f, QCheckBox, QComboBox, QDoubleSpinBox, QLineEdit, QSpinBox)
        widget.setToolTip(_tooltip(f))
        widgets[f.dest] = widget

        if f.path_role:
            row = QWidget()
            lay = QHBoxLayout(row)
            lay.setContentsMargins(0, 0, 0, 0)
            lay.setSpacing(4)
            lay.addWidget(widget, 1)
            pick = QPushButton("...")
            pick.setMaximumWidth(28)
            pick.setToolTip(f"Browse for {f.label}")

            def _browse(_checked=False, f=f, widget=widget):
                if browse is None:
                    return
                chosen = browse(f)
                if chosen:
                    widget.setText(str(chosen))

            pick.clicked.connect(_browse)
            lay.addWidget(pick)
            field_widget = row
        else:
            field_widget = widget

        label = QLabel(f.label + (" *" if f.required or f.positional else ""))
        label.setToolTip(_tooltip(f))
        if f.positional or f.required:
            label.setStyleSheet("font-weight: bold;")
        form.addRow(label, field_widget)

    show_all = QCheckBox("show every flag in the command line")
    show_all.setToolTip(
        "Off, only what differs from the defaults is passed - the short line you "
        "would actually type. On, every flag is spelled out for pasting into a script."
    )
    outer.addWidget(show_all)

    line = QLineEdit()
    line.setReadOnly(True)
    line.setStyleSheet("color: #a0a0b0;")
    outer.addWidget(line)

    def values() -> dict:
        return {f.dest: _read(f, widgets[f.dest]) for f in spec.fields}

    def set_values(new: dict) -> None:
        for f in spec.fields:
            if f.dest not in new:
                continue
            widget = widgets[f.dest]
            widget.blockSignals(True)
            _write(f, widget, new[f.dest])
            widget.blockSignals(False)
        refresh()

    def argv() -> list[str]:
        return to_argv(spec, values(), force_all=show_all.isChecked())

    def current_errors() -> list[str]:
        return errors(spec, values())

    def refresh() -> None:
        line.setText(
            command_line(
                spec,
                values(),
                prog="python -m hipct_seg_debug.edit",
                force_all=show_all.isChecked(),
            )
        )
        # Read off the widget rather than closing over the parameter, so a caller
        # can attach its handler *after* construction -- which it must, when that
        # handler needs the buttons this form is being built for.
        handler = getattr(box, "on_changed", None)
        if handler is not None:
            handler()

    for widget in widgets.values():
        _connect(widget, refresh)
    show_all.toggled.connect(refresh)

    box.values = values
    box.set_values = set_values
    box.argv = argv
    box.errors = current_errors
    box.refresh = refresh
    box.spec = spec
    box.widgets = widgets
    box.command_line_box = line
    box.on_changed = on_changed
    refresh()
    return box


def _right_align():
    from qtpy.QtCore import Qt

    return Qt.AlignRight | Qt.AlignVCenter


def _tooltip(f: FieldSpec) -> str:
    bits = [f.help] if f.help else []
    if f.default is not None and f.kind != FLAG:
        bits.append(f"default: {f.default}")
    elif f.default is None and not f.positional:
        bits.append("unset: the library's own default applies")
    return "\n".join(bits) or f.label


def _step_for(default) -> float:
    """A step that matches the quantity, guessed from its own default.

    A `--murray-percentile` of 0.10 wants 0.01 steps; a `--seg-box-um` of 2000
    wants 100. Both come out of the same expression.
    """
    if default in (None, 0):
        return 1.0
    magnitude = abs(float(default))
    if magnitude < 1:
        return 0.01
    if magnitude < 100:
        return 0.1
    return 10.0


def _make_widget(f, QCheckBox, QComboBox, QDoubleSpinBox, QLineEdit, QSpinBox):
    if f.kind == FLAG:
        widget = QCheckBox()
        widget.setChecked(bool(f.default))
        return widget

    if f.kind == CHOICE:
        widget = QComboBox()
        widget.addItems([str(c) for c in f.choices])
        if f.default is not None:
            widget.setCurrentText(str(f.default))
        return widget

    if f.kind in (INT, FLOAT) and f.nargs == 1 and not f.repeatable:
        widget = QSpinBox() if f.kind == INT else QDoubleSpinBox()
        widget.setRange(UNSET, -UNSET)
        if f.kind == FLOAT:
            widget.setDecimals(4)
            widget.setSingleStep(_step_for(f.default))
        if f.default is None:
            # The bottom of the range *is* "unset", so the leftmost position of the
            # spin box means "let the library decide" rather than "minus a billion".
            widget.setSpecialValueText("(library default)")
            widget.setValue(UNSET)
        else:
            widget.setValue(f.default)
        return widget

    widget = QLineEdit()
    if f.repeatable:
        widget.setText(join_repeat(f, f.default))
        widget.setPlaceholderText(
            "separate several with ';'" if repeat_separator(f)
            else "repeat values separated by spaces"
        )
    elif f.default is not None and f.nargs == 1:
        widget.setText(str(f.default))
    elif f.nargs > 1:
        widget.setPlaceholderText(" ".join(["x", "y", "z"][: f.nargs]))
    else:
        widget.setPlaceholderText("(library default)")
    return widget


def _read(f: FieldSpec, widget):
    name = type(widget).__name__
    if name == "QCheckBox":
        return widget.isChecked()
    if name == "QComboBox":
        return widget.currentText()
    if name in ("QSpinBox", "QDoubleSpinBox"):
        value = widget.value()
        return None if float(value) == UNSET else value
    text = widget.text().strip()
    if not text:
        return None
    if f.repeatable:
        caster = float if f.kind == FLOAT else int if f.kind == INT else str
        try:
            return [caster(p) for p in split_repeat(f, text)]
        except ValueError:
            return None
    if f.nargs > 1:
        parts = text.split()
        caster = float if f.kind == FLOAT else int if f.kind == INT else str
        try:
            return [caster(p) for p in parts] if len(parts) == f.nargs else None
        except ValueError:
            return None
    return text


def _write(f: FieldSpec, widget, value) -> None:
    name = type(widget).__name__
    if name == "QCheckBox":
        widget.setChecked(bool(value))
    elif name == "QComboBox":
        widget.setCurrentText("" if value is None else str(value))
    elif name in ("QSpinBox", "QDoubleSpinBox"):
        widget.setValue(UNSET if value is None else value)
    elif f.repeatable:
        widget.setText(join_repeat(f, value))
    elif f.nargs > 1:
        widget.setText("" if value is None else " ".join(str(v) for v in value))
    else:
        widget.setText("" if value is None else str(value))


def _connect(widget, slot) -> None:
    for signal in ("toggled", "currentTextChanged", "valueChanged", "textChanged"):
        if hasattr(widget, signal):
            getattr(widget, signal).connect(lambda *_a, slot=slot: slot())
            return
