"""The Data tab: which files are loaded, and swapping them for others.

Five inputs, and they are not symmetric -- which is most of what this panel has to
get right. `--raw` is a directory. `--surface` missing is a warning, not an error
(`main.py:182-183`). `--edits` names a file that a painting session may be about to
*create*, so it has an Open button and a Save button. Only the graph can be reloaded
on its own; the raw folder and the lattice both feed `WorldFrame`, so changing either
means rebuilding everything.

Recent paths persist to `<cache>/gui_recent.json`. That is the one exception to this
package's rule against settings files, and it is a narrow one: the rule exists so
derived state cannot go stale, and a list of paths you typed is not derived state.
Losing five long absolute paths on every restart is the single most annoying thing about
a tool like this.
"""

from __future__ import annotations

import json
from pathlib import Path

from . import cliform

#: The five inputs, in load order, as (dest, label, tooltip).
INPUTS = (
    (
        "raw",
        "raw image folder",
        "Directory of TIFF, JPEG 2000, or other supported slices; sets the raw grid.",
    ),
    ("graph", "skeleton (.am)", "ASCII Amira spatial graph, and the only input that "
                                "can be reloaded on its own. Select several - a "
                                "left-tree and a right-tree graph, say - and they are "
                                "held as separate trees in one editable graph."),
    ("seg", "segmentation (.am)", "Amira label lattice. A .Regions.am naming its "
                                  "trees (Left_Tree, Right_Tree) is drawn one "
                                  "surface per material; a binary mask as one. "
                                  "With the raw folder, this defines the frame."),
    ("surface", "surface (.stl)", "Reconstructed lumen surface. Optional - a missing "
                                  "one is a warning, not an error."),
    ("edits", "mask edits (.npz)", "Painted corrections, composited onto the lattice "
                                   "before anything reads it. Created by painting."),
)

MAX_RECENT = 10

#: Inputs holding several paths: ';'-separated in the field, a list in `args`.
#: The skeleton is the one that takes several -- merged into a single graph carrying
#: one `tree` index per source, so every edit tool still has exactly one graph to work
#: on while the trees stay apart. See `main.load_graphs`.
MULTI = ("graph",)


def recent_path(cache_dir) -> Path:
    return Path(cache_dir) / "gui_recent.json"


def load_recent(cache_dir) -> dict:
    """Never raises: a corrupt or absent file just means no history."""
    try:
        with open(recent_path(cache_dir), encoding="utf-8") as fh:
            data = json.load(fh)
        return {k: list(v)[:MAX_RECENT] for k, v in data.items() if isinstance(v, list)}
    except Exception:  # noqa: BLE001
        return {}


def save_recent(cache_dir, recent: dict) -> None:
    try:
        with open(recent_path(cache_dir), "w", encoding="utf-8") as fh:
            json.dump({k: v[:MAX_RECENT] for k, v in recent.items()}, fh, indent=1)
    except Exception:  # noqa: BLE001 - a read-only cache dir must not break the panel
        pass


def remember(recent: dict, dest: str, value: str) -> dict:
    if not value:
        return recent
    entries = [value] + [v for v in recent.get(dest, []) if v != value]
    recent[dest] = entries[:MAX_RECENT]
    return recent


def build_data_panel(app):
    """Return the Data widget for a `ViewerApp`."""
    from qtpy.QtCore import Qt
    from qtpy.QtWidgets import (
        QComboBox,
        QDoubleSpinBox,
        QFileDialog,
        QFormLayout,
        QHBoxLayout,
        QLabel,
        QMessageBox,
        QPushButton,
        QVBoxLayout,
        QWidget,
    )

    box = QWidget()
    lay = QVBoxLayout(box)
    lay.setContentsMargins(8, 8, 8, 8)

    from .main import _default_cache

    cache = getattr(app.session, "cache", None) or _default_cache()
    recent = load_recent(cache)
    fields: dict[str, QComboBox] = {}

    form = QFormLayout()
    lay.addLayout(form)

    for dest, label, tip in INPUTS:
        # Editable combo rather than a line edit: paths here are 90 characters of
        # long dataset paths, which are chosen again and again.
        field = QComboBox()
        field.setEditable(True)
        field.setToolTip(tip)
        field.setMinimumWidth(240)
        field.addItems(recent.get(dest, []))
        field.setCurrentText(str(getattr(app.args, dest, "") or ""))
        fields[dest] = field

        row = QWidget()
        row_lay = QHBoxLayout(row)
        row_lay.setContentsMargins(0, 0, 0, 0)
        row_lay.setSpacing(4)
        row_lay.addWidget(field, 1)

        pick = QPushButton("...")
        pick.setMaximumWidth(28)
        pick.setToolTip(f"Browse for the {label}")

        def _browse(_checked=False, dest=dest, label=label, field=field):
            role, filt = cliform.path_role("", dest)
            start = field.currentText() or ""
            if role == "dir":
                chosen = QFileDialog.getExistingDirectory(box, f"Choose the {label}", start)
            elif dest in MULTI:
                # Several at once, and *appended* to whatever is already listed:
                # adding a right-tree skeleton must not silently drop the left one.
                # Keyed on MULTI rather than on the path role, because `graph` is a
                # single file everywhere else -- the command forms in the Commands
                # tab take one graph, and only this panel merges several.
                picked, _ = QFileDialog.getOpenFileNames(
                    box, f"Choose the {label}", start.split(";")[0].strip(),
                    filt or "All files (*)")
                if picked:
                    have = [p.strip() for p in start.split(";") if p.strip()]
                    field.setCurrentText("; ".join([*have, *picked]))
                return
            else:
                chosen, _ = QFileDialog.getOpenFileName(
                    box, f"Choose the {label}", start, filt or "All files (*)")
            if chosen:
                field.setCurrentText(chosen)

        pick.clicked.connect(_browse)
        row_lay.addWidget(pick)

        name = QLabel(label)
        name.setToolTip(tip)
        form.addRow(name, row)

    # -- the voxel size ---------------------------------------------------
    # Its own row, below the paths and above Load, because it is the one input that
    # is not a path and the one the loader will not guess at. A voxel size taken from
    # a folder name or a bounding box is wrong silently -- it is internally consistent,
    # so every check passes while every radius and length in the session carries the
    # same error. LADAF-2024-28 records 32.99 um for a 32.04 um acquisition.
    voxel_row = QHBoxLayout()
    voxel = QDoubleSpinBox()
    voxel.setDecimals(4)
    voxel.setRange(0.0001, 10000.0)
    voxel.setSingleStep(0.01)
    voxel.setSuffix(" um")
    voxel.setMinimumWidth(120)
    voxel.setToolTip(
        "The acquisition's own raw voxel size. Not inferred: whatever is detected is\n"
        "offered as a suggestion and has to be confirmed, because a rounded value\n"
        "rescales every radius and length in the session by the same error."
    )
    confirmed = {"value": False}
    voxel_note = QLabel("")
    voxel_note.setWordWrap(True)
    voxel_note.setStyleSheet("color: #d0a000;")
    voxel_row.addWidget(QLabel("voxel size"))
    voxel_row.addWidget(voxel)
    voxel_confirm = QPushButton("Use this")
    voxel_confirm.setToolTip("Confirm the voxel size. Loading is refused until you do.")
    voxel_row.addWidget(voxel_confirm)
    voxel_row.addStretch(1)
    lay.addLayout(voxel_row)
    lay.addWidget(voxel_note)

    status = QLabel("")
    status.setWordWrap(True)
    status.setStyleSheet("color: #808090;")
    lay.addWidget(status)

    buttons = QHBoxLayout()
    load = QPushButton("Load all")
    load.setToolTip("Rebuild the whole session from these five paths. About 3 seconds.")
    reload_graph = QPushButton("Reload graph")
    reload_graph.setToolTip(
        "Re-read the skeleton only, keeping the frame, the lattice and the decoded\n"
        "slice caches. About 0.4 s - this is what to press between repair steps.")
    validate = QPushButton("Validate")
    validate.setToolTip("Eight checks that the four inputs share one coordinate frame.")
    # Enabled only once a finished command has written a graph this session. The panel
    # used to say "put it in the graph field and press Reload graph", which was correct
    # and left the user retyping a path the program already knew.
    load_result = QPushButton("Load result")
    # Its label changes to name the file on offer, so it carries a stable object name
    # for anything that needs to find it.
    load_result.setObjectName("load_result")
    load_result.setEnabled(False)
    load_result.setToolTip("No command has written a skeleton yet this session.")
    for button in (load, reload_graph, validate, load_result):
        buttons.addWidget(button)
    lay.addLayout(buttons)

    checks = QLabel("")
    checks.setWordWrap(True)
    checks.setTextFormat(Qt.RichText)
    lay.addWidget(checks)
    lay.addStretch(1)

    # -- actions ----------------------------------------------------------

    def values() -> dict:
        return {dest: fields[dest].currentText().strip() for dest, _l, _t in INPUTS}

    def _as_arg(dest, value):
        """The field's text as `args` wants it: a list for a multi-path input."""
        if dest in MULTI:
            return [p.strip() for p in str(value or "").split(";") if p.strip()]
        return value or None

    def _remember_all() -> None:
        nonlocal recent
        for dest, value in values().items():
            recent = remember(recent, dest, value)
        save_recent(cache, recent)

    def _confirm_swap() -> bool:
        """Refuse to discard unsaved painting or unexported skeleton edits."""
        blocking = app.can_swap()
        if not blocking:
            return True
        answer = QMessageBox.warning(
            box, "Unsaved work",
            "Loading a different dataset will discard:\n\n  "
            + "\n  ".join(blocking)
            + "\n\nThere is nowhere to save them automatically.",
            QMessageBox.Discard | QMessageBox.Cancel, QMessageBox.Cancel)
        return answer == QMessageBox.Discard

    def suggest_voxel() -> float | None:
        """What the inputs imply the voxel size is -- a suggestion, never a decision.

        The segmentation bounding box first, because it is exact arithmetic on a file
        that is already open, and the folder-name guess second. Neither is trusted:
        the point of showing them is that the operator can recognise a wrong one.
        """
        seg = fields["seg"].currentText().strip()
        if seg and Path(seg).exists():
            try:
                from . import amira

                info = amira.read_lattice_header(seg)
                return float(info.spacing[0])
            except Exception:  # noqa: BLE001 - a bad header is the Load button's job
                pass
        raw = fields["raw"].currentText().strip()
        if raw and Path(raw).exists():
            try:
                from .tiffstack import TiffStack

                return TiffStack(raw).nominal_voxel_um
            except Exception:  # noqa: BLE001
                pass
        return None

    def do_suggest_voxel() -> None:
        """Fill the box from the inputs, and mark it as *not* confirmed."""
        value = suggest_voxel()
        if value is None:
            voxel_note.setText("no voxel size could be suggested - type the "
                               "acquisition's own value")
            return
        voxel.blockSignals(True)
        voxel.setValue(float(value))
        voxel.blockSignals(False)
        confirmed["value"] = False
        _refresh_voxel()

    def _refresh_voxel() -> None:
        if confirmed["value"]:
            voxel_note.setText("")
            voxel_note.setStyleSheet("color: #808090;")
        else:
            voxel_note.setText(
                f"{voxel.value():.4f} um is a suggestion from the file names and the "
                "segmentation bounding box, and both can be wrong by a few percent. "
                "Press 'Use this' to confirm the acquisition's own value; every "
                "radius and length is scaled by it."
            )
            voxel_note.setStyleSheet("color: #d0a000;")
        load.setEnabled(confirmed["value"])

    def do_confirm_voxel() -> None:
        confirmed["value"] = True
        app.args.voxel_um = float(voxel.value())
        _refresh_voxel()
        status.setText(f"voxel size set to {voxel.value():.4f} um")

    def do_unconfirm_voxel(_value=None) -> None:
        confirmed["value"] = False
        _refresh_voxel()

    def do_load() -> None:
        chosen = values()
        problems = _problems(chosen)
        if problems:
            status.setText("<br>".join(problems))
            return
        if not confirmed["value"]:
            status.setText("confirm the voxel size first - it is never inferred")
            return
        if not _confirm_swap():
            status.setText("load cancelled")
            return
        for dest, value in chosen.items():
            setattr(app.args, dest, _as_arg(dest, value))
        app.args.voxel_um = float(voxel.value())
        _remember_all()
        status.setText("loading...")
        ok = app.load()
        status.setText(app.describe() if ok else "load failed - see the Log tab")
        refresh()

    def do_reload_graph() -> None:
        chosen = _as_arg("graph", fields["graph"].currentText())
        missing = [p for p in chosen if not Path(p).exists()]
        if not chosen:
            status.setText("no skeleton named")
            return
        if missing:
            status.setText("<br>".join(f"{p} does not exist" for p in missing))
            return
        app.reload_graph(chosen)
        _remember_all()
        status.setText(app.describe())
        refresh()

    offered: dict[str, str] = {}

    def offer_graph(path) -> None:
        """Remember a graph a command just wrote, and enable the button for it.

        An *offer*, never an action. Reloading on a job's behalf would swap the graph
        out from under an inspection and silently discard any skeleton edits with it --
        the reasoning in ``controlpanel._offer_outputs``, which this makes clickable
        rather than automatic.
        """
        offered["graph"] = str(path)
        load_result.setEnabled(True)
        load_result.setText(f"Load {Path(path).name}")
        load_result.setToolTip(f"Reload the viewer on {path}, written by the last command.")

    def do_load_result() -> None:
        path = offered.get("graph", "")
        if not path or not Path(path).exists():
            status.setText(f"{path or 'the result'} is no longer there")
            return
        fields["graph"].setCurrentText(path)
        do_reload_graph()

    def do_validate() -> None:
        if app.session is None:
            checks.setText("nothing loaded")
            return
        try:
            results = app.session.validate(strict=False)
        except Exception as exc:  # noqa: BLE001
            checks.setText(f"validation raised {type(exc).__name__}: {exc}")
            return
        failed = [c for c in results if not c.passed]
        if not failed:
            checks.setText(f"<b>all {len(results)} checks pass</b>")
        else:
            checks.setText(
                f"<b>{len(failed)} of {len(results)} failed</b><br>"
                + "<br>".join(f"{'FATAL' if c.fatal else 'warn'}: {c.name}" for c in failed)
                + "<br><i>the full table is in the Log tab</i>")

    def _problems(chosen) -> list[str]:
        out = []
        for dest, label, _tip in INPUTS:
            value = chosen.get(dest)
            role, _filt = cliform.path_role("", dest)
            optional = dest in ("surface", "edits")
            if not value:
                if optional:
                    continue
                out.append(f"{label} is required")
            elif dest in MULTI:
                for one in _as_arg(dest, value):
                    if not Path(one).exists():
                        out.append(f"{label}: {one} does not exist")
            elif role in cliform.INPUT_ROLES and not optional:
                if not Path(value).exists():
                    out.append(f"{label}: {value} does not exist")
        return out

    load.clicked.connect(do_load)
    voxel_confirm.clicked.connect(do_confirm_voxel)
    # Editing the number un-confirms it: a value typed and left is not a decision, and
    # the whole point of the row is that this one is made deliberately.
    voxel.valueChanged.connect(do_unconfirm_voxel)
    fields["seg"].currentTextChanged.connect(lambda _t: do_suggest_voxel())
    reload_graph.clicked.connect(do_reload_graph)
    validate.clicked.connect(do_validate)
    load_result.clicked.connect(do_load_result)

    def refresh() -> None:
        loaded = app.loaded
        reload_graph.setEnabled(loaded)
        validate.setEnabled(loaded)
        # A session that loaded has a confirmed size behind it, by construction: the
        # loader refuses to invent one.
        stated = getattr(app.args, "voxel_um", None)
        if stated:
            voxel.blockSignals(True)
            voxel.setValue(float(stated))
            voxel.blockSignals(False)
            confirmed["value"] = True
        elif voxel.value() <= 0.0001:
            do_suggest_voxel()
        _refresh_voxel()
        for dest, _l, _t in INPUTS:
            field = fields[dest]
            value = getattr(app.args, dest, "") or ""
            current = "; ".join(str(v) for v in value) if isinstance(value, list)                 else str(value)
            if current and current != field.currentText():
                field.blockSignals(True)
                field.setCurrentText(current)
                field.blockSignals(False)
        if not status.text():
            status.setText(app.describe())

    box.refresh = refresh
    box.fields = fields
    box.values = values
    box.offer_graph = offer_graph
    box.voxel = voxel
    box.voxel_confirm = voxel_confirm
    box.voxel_confirmed = lambda: confirmed["value"]
    refresh()
    return box
