"""The log pane: what a command printed, as it prints it.

The only thing here that is not obvious is `\\r` handling. Several commands report
progress by overwriting one line -- `cmd_mask_export:494` prints every 100 planes
with ``end="\\r"`` -- so appending each would add thirteen lines for one export and
bury the result. `runner.LineBuffer` flags those as transient and this replaces the
last line instead, which is what a terminal does.

Qt is imported inside the builder and docking is the caller's business, matching
`controls3d.build_layer_panel`.
"""

from __future__ import annotations

#: Above this, the oldest lines are dropped. A twelve-minute skeletonise prints a
#: line per iteration, and an unbounded QPlainTextEdit will eventually stall the GUI.
MAX_LINES = 5000


def build_log_panel(runner):
    """Return the log widget for ``runner``, wired to its `on_line` slot."""
    from qtpy.QtGui import QFontDatabase, QTextCursor
    from qtpy.QtWidgets import (
        QCheckBox,
        QHBoxLayout,
        QPlainTextEdit,
        QPushButton,
        QVBoxLayout,
        QWidget,
    )

    box = QWidget()
    lay = QVBoxLayout(box)
    lay.setContentsMargins(6, 6, 6, 6)

    view = QPlainTextEdit()
    view.setReadOnly(True)
    view.setMaximumBlockCount(MAX_LINES)
    view.setFont(QFontDatabase.systemFont(QFontDatabase.FixedFont))
    view.setLineWrapMode(QPlainTextEdit.NoWrap)
    lay.addWidget(view, 1)

    row = QHBoxLayout()
    follow = QCheckBox("follow")
    follow.setChecked(True)
    follow.setToolTip("Scroll to the newest line. Turn off to read back "
                      "through the output while a command is still running.")
    row.addWidget(follow)
    row.addStretch(1)

    copy = QPushButton("Copy")
    clear = QPushButton("Clear")
    save = QPushButton("Save...")
    for button in (copy, clear, save):
        row.addWidget(button)
    lay.addLayout(row)

    # True while the last line came from a `\r`, so the next write replaces it.
    state = {"transient": False}

    def append(text: str, transient: bool = False) -> None:
        cursor = view.textCursor()
        cursor.movePosition(QTextCursor.End)
        if state["transient"]:
            # Select the whole last line and overwrite it.
            cursor.select(QTextCursor.LineUnderCursor)
            cursor.removeSelectedText()
            cursor.insertText(text)
        else:
            view.appendPlainText(text)
        state["transient"] = transient
        if follow.isChecked():
            view.verticalScrollBar().setValue(view.verticalScrollBar().maximum())

    def do_copy() -> None:
        from qtpy.QtWidgets import QApplication

        QApplication.clipboard().setText(view.toPlainText())

    def do_clear() -> None:
        view.clear()
        state["transient"] = False

    def do_save() -> None:
        from qtpy.QtWidgets import QFileDialog

        chosen, _ = QFileDialog.getSaveFileName(box, "Save log", "hipct_log.txt",
                                                "Text (*.txt)")
        if chosen:
            with open(chosen, "w", encoding="utf-8") as fh:
                fh.write(view.toPlainText())
            append(f"(log written to {chosen})")

    copy.clicked.connect(do_copy)
    clear.clicked.connect(do_clear)
    save.clicked.connect(do_save)

    runner.on_line = append

    box.append = append
    box.view = view
    return box
