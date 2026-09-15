"""Layer controls for the 3D window: a visibility checkbox and an opacity slider per row.

The same job napari's layer list does for the slice browser. It is a docked Qt widget
rather than pyvista's in-scene slider widgets, which would float over the render, eat
screen space and capture mouse events near a vessel you were trying to pick.

Everything is driven through ``Picker3D``'s layer accessors rather than by poking actors
directly: the image plane and the mask isosurface are *rebuilt* on every pick, so an
opacity written straight to an actor would be discarded the next time you picked, and a
visibility flag set on one would be attached to an actor that no longer exists.

Qt is imported inside the function, matching the lazy import in ``Picker3D._open``, so
``viewer3d`` still imports in a session with no Qt binding.
"""

from __future__ import annotations

from .viewer3d import LAYERS

SLIDER_STEPS = 100


def build_layer_panel(picker):
    """Return the panel widget for ``picker``. Docking it is the caller's business.

    Kept separate from docking so the panel can be built and driven in a test without a
    ``BackgroundPlotter``.
    """
    from qtpy.QtCore import Qt
    from qtpy.QtWidgets import (
        QCheckBox,
        QComboBox,
        QGridLayout,
        QLabel,
        QPushButton,
        QSlider,
        QSpinBox,
        QWidget,
    )

    box = QWidget()
    grid = QGridLayout(box)
    grid.setContentsMargins(8, 8, 8, 8)
    grid.setHorizontalSpacing(8)
    rows = {}

    # Above the layer rows because it is not a layer: it restyles the two that are
    # coloured by a graph scalar -- the centreline and the radius circles -- together.
    color_label = QLabel("colour by")
    color_by = QComboBox()
    color_by.setToolTip(
        "What the centreline and the radius circles are mapped by. "
        "Strahler order needs a graph that carries one."
    )
    grid.addWidget(color_label, 0, 0)
    grid.addWidget(color_by, 0, 1, 1, 2)

    def _color_by_changed(index):
        key = color_by.itemData(index)
        if key is not None:
            picker.set_color_by(key)

    color_by.currentIndexChanged.connect(_color_by_changed)

    # Also above the layer rows, and for the same reason: the legend is chrome drawn
    # over the render rather than a layer of the scene, so it has no opacity to give a
    # slider and nothing about it is rebuilt per pick.
    legend = QCheckBox("legend box")
    legend.setToolTip("The key in the corner of the 3D view.\n"
                      "'L' (shift+l) toggles it there; this is the same state.")
    grid.addWidget(legend, 1, 0, 1, 3)

    def _legend_toggled(state):
        picker.set_legend_visible(bool(state))

    legend.toggled.connect(_legend_toggled)

    # Directly under the legend checkbox because it is the same subject: what the
    # render carries over the geometry. The export keeps the legend and the colour
    # bar and drops the keybinding block and the pick readout, so the checkbox above
    # is part of deciding what the file will contain.
    save = QPushButton("save figure...")
    save.setToolTip("Write the 3D view to SVG (or PDF / EPS / PS / TeX).\n"
                    "The keybindings and the pick readout are left out; the legend\n"
                    "box and the colour bar are kept as they are drawn here.\n"
                    "Geometry lands as an embedded image, the labels as vector text.")
    grid.addWidget(save, 2, 0, 1, 3)

    def _save():
        from qtpy.QtWidgets import QFileDialog

        chosen, _filter = QFileDialog.getSaveFileName(
            box, "Save the 3D view", "hipct_3d.svg",
            "SVG (*.svg);;PDF (*.pdf);;EPS (*.eps);;PostScript (*.ps);;TeX (*.tex)",
        )
        if not chosen:
            return
        # The write blocks the Qt loop, so a second click would otherwise queue and
        # export the same view again the moment the first one returns.
        save.setEnabled(False)
        try:
            print(f"[layers] figure -> {picker.save_figure(chosen)}")
        except Exception as exc:  # noqa: BLE001 - see `_rebuild` below
            print(f"[layers] save figure failed: {type(exc).__name__}: {exc}")
        finally:
            save.setEnabled(True)

    save.clicked.connect(_save)

    for r, (key, label, _default, _visible) in enumerate(LAYERS, start=3):
        check = QCheckBox(label)
        slider = QSlider(Qt.Horizontal)
        slider.setRange(0, SLIDER_STEPS)
        slider.setMinimumWidth(110)
        value = QLabel()
        value.setMinimumWidth(32)
        value.setAlignment(Qt.AlignRight | Qt.AlignVCenter)

        grid.addWidget(check, r, 0)
        grid.addWidget(slider, r, 1)
        grid.addWidget(value, r, 2)
        rows[key] = (check, slider, value)

        # Bind the key per row; a closure over the loop variable would give every row
        # the last key.
        def _toggled(state, key=key):
            picker.set_layer_visible(key, bool(state))

        def _moved(step, key=key, value=value):
            picker.set_layer_opacity(key, step / SLIDER_STEPS)
            value.setText(f"{step / SLIDER_STEPS:.2f}")

        check.toggled.connect(_toggled)
        slider.valueChanged.connect(_moved)

    # The whole-tree isosurface is the one layer with a resolution, and it is cached
    # for the session -- so this row is both "show me more detail" and "pick up what
    # I just painted". Rebuilding is explicit because a full contour per brush stroke
    # would be unusable.
    stride_label = QLabel("whole-tree stride")
    stride = QSpinBox()
    stride.setRange(1, 16)
    stride.setValue(int(getattr(picker, "seg_stride", 1)))
    stride.setToolTip("1 is full resolution (3.75M triangles, ~6s on LADAF-2024-28);\n"
                      "4 is ~0.5s and 213k. Takes effect on rebuild.")
    rebuild = QPushButton("rebuild")
    rebuild.setToolTip("Rebuild the whole-tree isosurface: applies a new stride, and\n"
                       "picks up any mask corrections painted since it was built.")
    grid.addWidget(stride_label, len(LAYERS) + 3, 0)
    grid.addWidget(stride, len(LAYERS) + 3, 1)
    grid.addWidget(rebuild, len(LAYERS) + 3, 2)

    def _stride_changed(value):
        picker.set_seg_stride(int(value))

    def _rebuild():
        # The Qt loop is blocked for the duration, so clicks would otherwise queue
        # and replay a second contour the moment the first returns.
        stride.setEnabled(False)
        rebuild.setEnabled(False)
        try:
            picker.rebuild_segmentation_all()
        except Exception as exc:  # noqa: BLE001 - see below
            # Not merely tidiness. PyQt5 calls qFatal on an exception that escapes a
            # slot, which aborts the process rather than raising -- so a failed
            # rebuild would take the whole window down with no traceback.
            print(f"[layers] rebuild failed: {type(exc).__name__}: {exc}")
        finally:
            refresh()

    stride.valueChanged.connect(_stride_changed)
    rebuild.clicked.connect(_rebuild)

    hint = QLabel("'i' and 'g' toggle the two overlays from the 3D view;\n"
                  "the checkboxes here are the same state. Radius circles\n"
                  "are graph-wide; the three shape rows need 'v'.")
    hint.setStyleSheet("color: #808090;")
    grid.addWidget(hint, len(LAYERS) + 4, 0, 1, 3)
    grid.setRowStretch(len(LAYERS) + 5, 1)

    def refresh():
        """Re-read the picker and update the widgets.

        Signals are blocked while writing: a checkbox set in code would otherwise emit
        ``toggled`` and call straight back into the picker that just moved it.
        """
        legend_available = picker.legend_available()
        legend.setEnabled(legend_available)
        legend.blockSignals(True)
        legend.setChecked(legend_available and picker.legend_visible())
        legend.blockSignals(False)

        for key, (check, slider, value) in rows.items():
            available = picker.layer_available(key)
            opacity = picker.layer_opacity(key)
            for w in (check, slider, value):
                w.setEnabled(available)
            check.blockSignals(True)
            slider.blockSignals(True)
            check.setChecked(available and picker.layer_visible(key))
            slider.setValue(int(round(opacity * SLIDER_STEPS)))
            check.blockSignals(False)
            slider.blockSignals(False)
            value.setText(f"{opacity:.2f}")

        available = picker.layer_available("segmentation_all") or picker.labels is not None
        stride_label.setEnabled(available)
        stride.setEnabled(available)
        rebuild.setEnabled(available)
        stride.blockSignals(True)
        stride.setValue(int(getattr(picker, "seg_stride", 1)))
        stride.blockSignals(False)
        # Rebuilt rather than merely re-selected: which modes a dataset can serve is
        # a property of *its* graph, and a swap can take Strahler away or bring it.
        modes = picker.color_modes()
        current = picker.color_by()
        color_by.blockSignals(True)
        color_by.clear()
        for key, label in modes:
            color_by.addItem(label, key)
        if modes:
            keys = [key for key, _label in modes]
            color_by.setCurrentIndex(keys.index(current) if current in keys else 0)
        color_by.blockSignals(False)
        # One mode is not a choice, and a live combo box implies there is another.
        color_label.setEnabled(len(modes) > 1)
        color_by.setEnabled(len(modes) > 1)

        if getattr(picker, "segmentation_all_stale", lambda: False)():
            rebuild.setText("rebuild *")
        else:
            rebuild.setText("rebuild")

    box.refresh = refresh
    picker.on_layers_changed = refresh
    refresh()
    return box
