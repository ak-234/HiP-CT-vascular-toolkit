"""The docked edit panel: buttons for every operation, and a status line.

A sibling of ``controls3d``, and deliberately the same shape -- a plain Qt widget
built by a function, with Qt imported inside it so the module still imports in a
session that has no binding. Docking is the caller's business, which keeps the
panel constructible in a test without a ``BackgroundPlotter``.

The keys on the 3D window do the same things, but a key you have to already know
about is not discoverable. Every button here names its shortcut.
"""

from __future__ import annotations

# (attribute on EditController, button label, shortcut hint, tooltip)
ACTIONS = (
    ("delete_picked_segment", "delete segment", "d",
     "Remove the picked segment, its points, and any node it leaves stranded."),
    ("delete_picked_branch", "prune branch", "t",
     "Remove the picked segment and everything downstream of the nearer end."),
    ("split_at_pick", "split here", "x",
     "Turn the picked point into a node, so a vessel can be attached to it."),
    ("fill_collapse_at_pick", "fill collapse", "f",
     "Restore the radii of the collapsed run around the pick by extrapolating the "
     "taper from the healthy vessel either side of it."),
    ("rebuild_at_pick", "rebuild here", "u",
     "Regenerate the lumen surface in a box around the pick."),
)


def build_edit_panel(controller):
    """Return the edit panel widget for `controller`."""
    from qtpy.QtWidgets import (
        QCheckBox,
        QDoubleSpinBox,
        QHBoxLayout,
        QLabel,
        QPushButton,
        QVBoxLayout,
        QWidget,
    )

    box = QWidget()
    outer = QVBoxLayout(box)
    outer.setContentsMargins(8, 8, 8, 8)
    outer.setSpacing(6)

    enable = QCheckBox("edit mode  (e)")
    enable.setToolTip(
        "Edits are refused while this is off, so a stray keypress over the render "
        "window cannot silently change the graph."
    )
    outer.addWidget(enable)

    undo_row = QHBoxLayout()
    undo_btn = QPushButton("undo  (z)")
    redo_btn = QPushButton("redo  (y)")
    undo_row.addWidget(undo_btn)
    undo_row.addWidget(redo_btn)
    outer.addLayout(undo_row)

    for attr, label, key, tip in ACTIONS:
        btn = QPushButton(f"{label}  ({key})")
        btn.setToolTip(tip)
        btn.clicked.connect(_bind(controller, attr))
        outer.addWidget(btn)

    radius_row = QHBoxLayout()
    radius_row.addWidget(QLabel("radius x"))
    factor = QDoubleSpinBox()
    factor.setRange(0.1, 5.0)
    factor.setSingleStep(0.05)
    factor.setValue(1.10)
    factor.setToolTip("Scale every radius along the picked segment. k / j step by 1.1.")
    radius_row.addWidget(factor)
    wider = QPushButton("wider  (k)")
    narrower = QPushButton("narrower  (j)")
    radius_row.addWidget(wider)
    radius_row.addWidget(narrower)
    outer.addLayout(radius_row)

    auto = QCheckBox("rebuild after every edit")
    auto.setChecked(controller.auto_rebuild)
    auto.setToolTip(
        "Off is useful for a run of edits in one region: make them all, then "
        "rebuild once."
    )
    outer.addWidget(auto)

    summary = QLabel(controller.summary())
    summary.setWordWrap(True)
    outer.addWidget(summary)

    status = QLabel(controller.status)
    status.setWordWrap(True)
    status.setStyleSheet("color: #b0b0c0;")
    outer.addWidget(status)
    outer.addStretch(1)

    def refresh(text=None):
        # Signals are blocked while writing, so refreshing the checkbox from a
        # keypress does not loop back into toggle_enabled.
        enable.blockSignals(True)
        enable.setChecked(controller.enabled)
        enable.blockSignals(False)
        undo_btn.setEnabled(controller.graph.history.can_undo)
        redo_btn.setEnabled(controller.graph.history.can_redo)
        summary.setText(controller.summary())
        status.setText(text if text is not None else controller.status)

    enable.toggled.connect(lambda _on: (controller.toggle_enabled(), refresh()))
    undo_btn.clicked.connect(lambda: (controller.undo(), refresh()))
    redo_btn.clicked.connect(lambda: (controller.redo(), refresh()))
    wider.clicked.connect(
        lambda: (controller.scale_picked_radius(factor.value()), refresh())
    )
    narrower.clicked.connect(
        lambda: (controller.scale_picked_radius(1.0 / factor.value()), refresh())
    )
    auto.toggled.connect(lambda on: setattr(controller, "auto_rebuild", bool(on)))

    controller.on_status = refresh
    refresh()
    return box


def _bind(controller, attr):
    """Call `controller.attr()` then refresh, without capturing the loop variable."""

    def run(_checked=False, attr=attr):
        getattr(controller, attr)()
        if controller.on_status is not None:
            controller.on_status(controller.status)

    return run
