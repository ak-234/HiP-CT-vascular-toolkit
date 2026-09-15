"""The brush: a writable segmentation layer in the slice browser.

napari's ``Labels`` layer already *is* a paint tool -- brush, eraser, fill, brush
size, and its own Ctrl+Z. What it cannot do by itself is any of the four things
that make painting this dataset meaningful:

* **be on the right grid.** The mask is 65.98 um and the raw stack is 32.99 um, so
  ``viewer2d`` gathers the segmentation onto the raw pixel grid for display. Painting
  on that grid would need a lossy many-to-one reduction to get back to the mask, so
  the editable layer sits on the **segmentation's own grid** instead, placed by
  :func:`~..volume.seg_window_placement`. One painted pixel is exactly one mask voxel.
* **be finite.** A ``Labels`` layer needs a real array, and the mask is 2.34 GB. The
  layer is therefore a box around the pick -- 192 voxels cubed by default, 12.7 mm
  and 7.1 MB -- which is enough room to follow a vessel without re-picking.
* **outlive the pick.** Every pick rebuilds the slab and overwrites layer data, so
  the edits have to be captured into :class:`~.maskedit.MaskEdits` first.
* **survive undo.** This is the subtle one. ``Labels.events.paint`` fires when you
  paint but **not** when you press Ctrl+Z -- measured, not assumed. Accumulating
  paint events would therefore leave the store holding strokes the user has already
  undone and can no longer see. So the store is never accumulated: :meth:`commit`
  diffs the layer against the pristine window it was opened from, and
  :meth:`~.maskedit.MaskEdits.diff` replaces that window's contribution wholesale.
  Undo restores the array, the next diff sees agreement, the entry vanishes.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .maskedit import MaskEdits, MaskSource

PAINT_LAYER = "segmentation (editable)"
# Deliberately unlike SEG_COLOR and SEG_ALL_COLOR: which mask you are looking at
# matters most at the moment you are about to change one of them.
PAINT_COLOR = "#ffab40"
DEFAULT_BOX_VOX = 192
DEFAULT_BRUSH = 4


@dataclass
class PaintBox:
    """The block of segmentation voxels currently open for painting."""

    origin_kji: tuple  # (k, row, col) segmentation index of corner 0
    shape_kji: tuple
    base: np.ndarray  # pristine window straight from the lattice: the diff baseline

    @property
    def nbytes(self) -> int:
        return int(self.base.nbytes)

    def contains(self, kji) -> bool:
        return all(o <= int(v) < o + n
                   for v, o, n in zip(kji, self.origin_kji, self.shape_kji))

    def bounds_um(self, frame, pad_um: float = 0.0) -> np.ndarray:
        """``(2, 3)`` world box in um, x/y/z, covering the whole block."""
        k0, j0, i0 = self.origin_kji
        nk, nj, ni = self.shape_kji
        lo = frame.seg_to_um([[i0, j0, k0]])[0]
        hi = frame.seg_to_um([[i0 + ni - 1, j0 + nj - 1, k0 + nk - 1]])[0]
        half = np.asarray(frame.seg_spacing, dtype=np.float64) / 2.0 + float(pad_um)
        return np.array([np.minimum(lo, hi) - half, np.maximum(lo, hi) + half])


class PaintSession:
    """Owns the edit store, the open paint box, and the napari layer over it."""

    def __init__(self, source: MaskSource, frame, *, size_vox: int = DEFAULT_BOX_VOX,
                 edits_path=None, splice_mode: str = "add"):
        if not isinstance(source, MaskSource):
            raise TypeError("PaintSession needs a MaskSource, not a bare lattice")
        self.source = source
        self.frame = frame
        self.size_vox = int(size_vox)
        self.edits_path = Path(edits_path) if edits_path else None
        self.splice_mode = splice_mode

        self.box: PaintBox | None = None
        self.layer = None
        self.status = ""
        self.unsaved = False
        self.on_status = None  # set by the panel
        self.on_reskeletonise = None  # set by main.py when --edit is on

    @property
    def edits(self) -> MaskEdits:
        return self.source.edits

    # -------------------------------------------------------------- the box

    def open_box(self, centre_um) -> PaintBox:
        """Decode a fresh block around `centre_um` and make it the paint target.

        Clipped to the lattice rather than zero-padded: a zero pad would let the
        brush deposit edits at indices that are not voxels, which would then be
        silently dropped on export.
        """
        centre = np.asarray(centre_um, dtype=np.float64).reshape(3)
        i, j, k = (int(v) for v in self.frame.um_to_seg_index(centre)[0])
        half = max(self.size_vox // 2, 1)
        src = self.source
        k0, k1 = _clip(k - half, k + half, src.nz)
        j0, j1 = _clip(j - half, j + half, src.ny)
        i0, i1 = _clip(i - half, i + half, src.nx)

        base = src.window(k0, k1, j0, j1, i0, i1, edited=False)
        self.box = PaintBox(origin_kji=(k0, j0, i0),
                            shape_kji=tuple(int(v) for v in base.shape),
                            base=base)
        return self.box

    def current_data(self) -> np.ndarray:
        """The open block with the store composited on: what napari should hold."""
        if self.box is None:
            raise RuntimeError("no paint box is open")
        data = self.box.base.copy()
        self.edits.apply(data, self.box.origin_kji)
        return data

    # ------------------------------------------------------------- the layer

    def attach(self, viewer, centre_um) -> None:
        """(Re)open a box around `centre_um` and put it in front of the user.

        Commits first: this runs on every pick, and the previous box's array is
        about to be replaced.
        """
        import napari

        self.commit()
        self.open_box(centre_um)
        scale, translate = _placement(self.frame, self.box.origin_kji)
        data = self.current_data()

        # Looked up by name rather than trusting the cached handle: a slab whose
        # layer set changed takes `viewer.layers.clear()`, which leaves this
        # session holding a layer that is no longer in any viewer.
        live = [layer for layer in viewer.layers if layer.name == PAINT_LAYER]
        if live:
            # Assign rather than recreate, for the reason `viewer2d._update_layers`
            # documents: tearing a visual down mid-draw kills the process.
            self.layer = live[0]
            self.layer.data = data
            self.layer.scale = scale
            self.layer.translate = translate
            return

        self.layer = viewer.add_labels(
            data,
            name=PAINT_LAYER,
            scale=scale,
            translate=translate,
            opacity=0.45,
            colormap=napari.utils.DirectLabelColormap(
                color_dict={None: "transparent", 1: PAINT_COLOR}
            ),
        )
        self.layer.selected_label = 1
        self.layer.n_edit_dimensions = 2
        self.layer.brush_size = DEFAULT_BRUSH
        self.layer.events.paint.connect(self._on_paint)

    def _on_paint(self, _event=None) -> None:
        self.unsaved = True
        self._notify()

    # ---------------------------------------------------------------- commit

    def commit(self) -> int:
        """Fold what is on screen into the store. Returns the window's voxel count.

        Safe and cheap to call at any time -- it is a comparison of two arrays, a
        few milliseconds over 7 MB -- so it runs on the button, on every pick, and
        on close, rather than trying to track whether it is needed.
        """
        if self.layer is None or self.box is None:
            return 0
        n = self.edits.diff(np.asarray(self.layer.data), self.box.base,
                            self.box.origin_kji)
        if n:
            self.unsaved = True
        self._notify()
        return n

    def revert_box(self) -> int:
        """Throw away every edit inside the open box and restore the lattice values."""
        if self.box is None:
            return 0
        dropped = self.edits.clear_window(self.box.origin_kji, self.box.shape_kji)
        if self.layer is not None:
            self.layer.data = self.box.base.copy()
        self.unsaved = self.unsaved or bool(dropped)
        self._notify()
        return dropped

    def save(self, path=None) -> Path | None:
        self.commit()
        target = Path(path) if path else self.edits_path
        if target is None:
            return None
        self.edits.save(target)
        self.edits_path = target
        self.unsaved = False
        self._notify()
        return target

    # ------------------------------------------------------- re-skeletonise

    def edited_bounds_um(self, pad_um: float = 0.0) -> np.ndarray | None:
        """World box (um) around the *edits*, not around the whole paint box.

        Re-skeletonising 12.7 mm because someone painted 20 voxels would be silly;
        this is what the ROI button actually hands to
        :func:`~.reskeletonise.reskeletonise_box`.
        """
        self.commit()
        bbox = self.edits.bbox_seg()
        if bbox is None:
            return None
        (k0, j0, i0), (k1, j1, i1) = bbox
        lo = self.frame.seg_to_um([[i0, j0, k0]])[0]
        hi = self.frame.seg_to_um([[i1, j1, k1]])[0]
        half = np.asarray(self.frame.seg_spacing, dtype=np.float64) / 2.0 + float(pad_um)
        return np.array([np.minimum(lo, hi) - half, np.maximum(lo, hi) + half])

    # ---------------------------------------------------------------- status

    def describe(self) -> str:
        s = self.edits.stats()
        if self.box is None:
            return "no paint box open"
        nk, nj, ni = self.box.shape_kji
        return (f"box {ni}x{nj}x{nk} voxels ({self.box.nbytes / 1e6:.1f} MB)   "
                f"+{s['added']:,} painted  -{s['removed']:,} erased"
                + ("  [unsaved]" if self.unsaved else ""))

    def _notify(self) -> None:
        self.status = self.describe()
        if self.on_status is not None:
            self.on_status(self.status)


def _clip(lo: int, hi: int, n: int) -> tuple[int, int]:
    lo = max(int(lo), 0)
    hi = min(int(hi), int(n))
    if hi <= lo:  # a pick outside the lattice still has to yield a usable block
        lo, hi = 0, min(1, int(n))
    return lo, hi


def _placement(frame, origin_kji):
    from ..volume import seg_window_placement

    return seg_window_placement(frame, origin_kji)


# --------------------------------------------------------------------- panel


def build_paint_panel(session: PaintSession):
    """The Qt dock for the slice browser.

    Buttons rather than key bindings. napari's ``Labels`` layer already owns
    ``1``-``5``, ``[``, ``]`` and Ctrl+Z/Ctrl+Shift+Z, and a binding that silently
    loses to one of those is worse than no binding at all.
    """
    from qtpy.QtWidgets import (
        QCheckBox, QComboBox, QFileDialog, QHBoxLayout, QLabel, QPushButton,
        QVBoxLayout, QWidget,
    )

    box = QWidget()
    lay = QVBoxLayout(box)

    lay.addWidget(QLabel(
        "<b>Painting the mask</b><br>"
        f"Select <i>{PAINT_LAYER}</i>, then<br>"
        "<b>2</b> paint &nbsp; <b>3</b> fill &nbsp; <b>4</b> erase<br>"
        "<b>[</b> / <b>]</b> brush size &nbsp; <b>Ctrl+Z</b> undo<br>"
        "<i>One pixel here is one mask voxel<br>(65.98 &micro;m), not one raw pixel.</i>"
    ))

    status = QLabel(session.describe())
    status.setWordWrap(True)
    session.on_status = status.setText
    lay.addWidget(status)

    across = QCheckBox("brush across slices (3D)")
    def _across(state):
        if session.layer is not None:
            session.layer.n_edit_dimensions = 3 if state else 2
    across.stateChanged.connect(_across)
    lay.addWidget(across)

    commit = QPushButton("Commit edits")
    commit.clicked.connect(lambda: session.commit())
    lay.addWidget(commit)

    revert = QPushButton("Revert this box")
    revert.clicked.connect(lambda: session.revert_box())
    lay.addWidget(revert)

    row = QHBoxLayout()
    mode = QComboBox()
    mode.addItems(["add", "replace"])
    mode.setCurrentText(session.splice_mode)
    mode.currentTextChanged.connect(lambda t: setattr(session, "splice_mode", t))
    row.addWidget(QLabel("splice:"))
    row.addWidget(mode)
    lay.addLayout(row)

    reskel = QPushButton("Re-skeletonise painted region")
    reskel.setEnabled(session.on_reskeletonise is not None)
    reskel.clicked.connect(lambda: _run_reskeletonise(session))
    lay.addWidget(reskel)
    # The callback is wired after the panel is built when --edit is on, so the
    # button has to learn about it late rather than being fixed at build time.
    box._hipct_reskel_button = reskel

    save = QPushButton("Save edits...")

    def _save():
        target = session.edits_path
        if target is None:
            chosen, _ = QFileDialog.getSaveFileName(
                box, "Save mask edits", "mask_edits.npz", "NumPy archive (*.npz)"
            )
            if not chosen:
                return
            target = chosen
        path = session.save(target)
        print(f"[paint] wrote {path}  ({session.edits.describe()})")

    save.clicked.connect(_save)
    lay.addWidget(save)

    lay.addStretch(1)
    return box


def _run_reskeletonise(session: PaintSession) -> None:
    if session.on_reskeletonise is None:
        print("[paint] re-skeletonisation needs --edit (it rebuilds the surface too)")
        return
    bounds = session.edited_bounds_um()
    if bounds is None:
        print("[paint] nothing painted yet - nothing to re-skeletonise")
        return
    session.on_reskeletonise(bounds, session.splice_mode)


def refresh_panel(panel, session: PaintSession) -> None:
    """Re-enable the ROI button once the controller has registered itself."""
    button = getattr(panel, "_hipct_reskel_button", None)
    if button is not None:
        button.setEnabled(session.on_reskeletonise is not None)
