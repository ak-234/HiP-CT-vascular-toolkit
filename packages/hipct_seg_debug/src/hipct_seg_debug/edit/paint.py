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

Two more things a mask with more than one tree needs:

* **the right label.** Avizo writes Left_Tree and Right_Tree as 1 and 2, and a brush
  fixed at 1 moves every right-coronary correction into the left tree -- invisibly,
  when only label 1 has a colour. The brush takes the majority label of each box, or
  one pinned in the panel; every label present is coloured; and ``preserve_labels``
  keeps one tree's brush off the other's voxels.
* **seeded repairs.** :mod:`.mask_tools` fills a collapsed lumen and splits two fused
  vessels from points placed in the ``paint seeds`` layer. The result is written
  through ``Labels.data_setitem`` inside ``block_history``, so one tool run is one
  Ctrl+Z, and :meth:`PaintSession.commit` picks it up by the same diff as a stroke.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .maskedit import MaskEdits, MaskSource

PAINT_LAYER = "segmentation (editable)"
SEEDS_LAYER = "paint seeds"
RESULT_LAYER = "paint result"
# Deliberately unlike SEG_COLOR and SEG_ALL_COLOR: which mask you are looking at
# matters most at the moment you are about to change one of them. One per label, and
# far apart -- Avizo's own Left_Tree and Right_Tree colours are two blue-purples.
PAINT_COLOR = "#ffab40"
PAINT_COLORS = (PAINT_COLOR, "#69f0ae", "#ff4081", "#40c4ff")
SEED_COLORS = ("#ffab40", "#40c4ff", "#ff4081", "#69f0ae", "#ffff00", "#e040fb")
# The preview of a repair tool: what it wrote, what it refused, where it cut.
RESULT_FILLED, RESULT_REFUSED, RESULT_CUT = 1, 2, 3
RESULT_COLORS = {None: "transparent", RESULT_FILLED: "#00e676",
                 RESULT_REFUSED: "#ff1744", RESULT_CUT: "#ffea00"}
DEFAULT_BOX_VOX = 192
DEFAULT_BRUSH = 4
FILL_PAD = 45  # voxels of context round the seeds: capsule, leak shell, topology margin
SPLIT_PAD = 90  # the split's own margin is at most 40, plus its search for one


def layer_names() -> tuple[str, ...]:
    """Every layer a paint session puts in the viewer.

    ``viewer2d._expected_names`` has to know all of them: a name it does not expect
    sends the next pick down the ``layers.clear()`` path, which kills the process.
    """
    return (PAINT_LAYER, SEEDS_LAYER, RESULT_LAYER)


def label_color_dict(values) -> dict:
    """``{None: transparent, value: colour}`` for every non-zero label in `values`."""
    out = {None: "transparent"}
    for v in sorted({int(v) for v in values if int(v) > 0}):
        out[v] = PAINT_COLORS[(v - 1) % len(PAINT_COLORS)]
    return out


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
    """Owns the edit store, the open paint box, and the napari layers over it."""

    def __init__(self, source: MaskSource, frame, *, size_vox: int = DEFAULT_BOX_VOX,
                 edits_path=None, splice_mode: str = "add", stack=None,
                 materials=(), sampler=None):
        if not isinstance(source, MaskSource):
            raise TypeError("PaintSession needs a MaskSource, not a bare lattice")
        self.source = source
        self.frame = frame
        self.size_vox = int(size_vox)
        self.edits_path = Path(edits_path) if edits_path else None
        self.splice_mode = splice_mode
        self.stack = stack
        self.materials = tuple(materials or ())
        self._sampler = sampler

        self.box: PaintBox | None = None
        self.layer = None
        self.seeds_layer = None
        self.result_layer = None
        self.status = ""
        self.unsaved = False
        #: "auto" -- the majority label of each box -- or a pinned label value.
        self.label_mode: str | int = "auto"
        self.preserve_labels = True
        self.seed_group = 1
        self.last_result = None
        self._setting_label = False
        self.on_status = None  # set by the panel
        self.on_label = None  # set by the panel: the brush label changed under it
        self.on_splice = None  # set by the panel: a split switched the splice mode
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

    # ------------------------------------------------------------ the labels

    @property
    def names(self) -> dict[int, str]:
        """Label value -> material name, for the foreground materials."""
        return {int(m.value): m.name for m in self.materials if not m.is_exterior}

    def label_choices(self) -> list[tuple[str, int]]:
        """``(name, value)`` for every label the brush may write.

        The lattice's own Materials when it declares them; otherwise whatever the
        open box holds, and always at least 1.
        """
        if self.names:
            return [(name, value) for value, name in sorted(self.names.items())]
        values = {1}
        if self.box is not None:
            values |= {int(v) for v in np.unique(self.box.base) if v > 0}
        return [(f"label {v}", v) for v in sorted(values)]

    def default_label(self) -> int:
        """The commonest label in the open box: the tree you are most likely fixing."""
        from .mask_tools import majority_label

        if self.box is None:
            return 1
        return majority_label(self.box.base, self.box.base > 0) or 1

    def current_label(self) -> int:
        return self.default_label() if self.label_mode == "auto" else int(self.label_mode)

    def label_name(self, value: int) -> str:
        return self.names.get(int(value), f"label {int(value)}")

    def set_label(self, mode) -> None:
        """``"auto"`` or a label value; applied to the brush straight away."""
        self.label_mode = "auto" if mode in (None, "auto") else int(mode)
        self._apply_label()
        self._notify()

    def set_preserve(self, on: bool) -> None:
        """Keep the brush (and the eraser) to background and the brush's own label."""
        self.preserve_labels = bool(on)
        if self.layer is not None:
            self.layer.preserve_labels = self.preserve_labels

    def _apply_label(self) -> None:
        if self.layer is None:
            return
        self._setting_label = True
        try:
            self.layer.selected_label = self.current_label()
        finally:
            self._setting_label = False
        if self.on_label is not None:
            self.on_label()

    def _on_selected_label(self, _event=None) -> None:
        # napari's own picker (5 / L) changed the label: that is a choice, so pin it.
        if self._setting_label or self.layer is None:
            return
        self.label_mode = int(self.layer.selected_label)
        if self.on_label is not None:
            self.on_label()
        self._notify()

    def _colormap(self, data):
        import napari

        values = {v for _, v in self.label_choices()} | {int(v) for v in np.unique(data)}
        return napari.utils.DirectLabelColormap(color_dict=label_color_dict(values))

    # ------------------------------------------------------------- the layers

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

        # Looked up by name rather than trusting the cached handles: a slab whose
        # layer set changed takes `viewer.layers.clear()`, which leaves this
        # session holding layers that are no longer in any viewer.
        live = {layer.name: layer for layer in viewer.layers}
        if PAINT_LAYER in live:
            # Assign rather than recreate, for the reason `viewer2d._update_layers`
            # documents: tearing a visual down mid-draw kills the process.
            self.layer = live[PAINT_LAYER]
            self.layer.data = data
            self.layer.scale = scale
            self.layer.translate = translate
            self.layer.colormap = self._colormap(data)
        else:
            self.layer = viewer.add_labels(
                data,
                name=PAINT_LAYER,
                scale=scale,
                translate=translate,
                opacity=0.45,
                colormap=self._colormap(data),
            )
            self.layer.n_edit_dimensions = 2
            self.layer.brush_size = DEFAULT_BRUSH
            self.layer.events.paint.connect(self._on_paint)
            self.layer.events.selected_label.connect(self._on_selected_label)
        self.layer.preserve_labels = self.preserve_labels
        self._apply_label()

        empty = np.zeros(data.shape, dtype=np.uint8)
        if RESULT_LAYER in live:
            self.result_layer = live[RESULT_LAYER]
            self.result_layer.data = empty
            self.result_layer.scale = scale
            self.result_layer.translate = translate
        else:
            self.result_layer = viewer.add_labels(
                empty, name=RESULT_LAYER, scale=scale, translate=translate, opacity=0.6,
                colormap=napari.utils.DirectLabelColormap(color_dict=RESULT_COLORS),
            )
            self.result_layer.editable = False

        # World coordinates, so seeds stay put when the box moves: a run traced
        # across two boxes keeps its first half.
        if SEEDS_LAYER in live:
            self.seeds_layer = live[SEEDS_LAYER]
        else:
            self.seeds_layer = viewer.add_points(
                np.empty((0, data.ndim)), ndim=data.ndim, name=SEEDS_LAYER,
                features={"group": np.empty(0, dtype=int)},
                feature_defaults={"group": self.seed_group},
                face_color="group", face_color_cycle=list(SEED_COLORS), size=4,
            )

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
        if self.result_layer is not None:
            self.result_layer.data = np.zeros(self.box.shape_kji, dtype=np.uint8)
        self.unsaved = self.unsaved or bool(dropped)
        self._notify()
        return dropped

    # ---------------------------------------------------------- repair tools

    @property
    def sampler(self):
        """Raw values on the segmentation grid, or ``None`` without a raw stack."""
        if self._sampler is None and self.stack is not None:
            from .mask_repair import StackSampler

            self._sampler = StackSampler(self.stack, self.frame)
        return self._sampler

    def set_seed_group(self, group: int) -> None:
        """The group the next seed joins: one per vessel for a split."""
        self.seed_group = int(group)
        if self.seeds_layer is not None:
            self.seeds_layer.feature_defaults = {"group": self.seed_group}

    def clear_seeds(self) -> None:
        if self.seeds_layer is not None:
            self.seeds_layer.data = np.empty((0, self.seeds_layer.ndim))

    def seed_groups(self) -> tuple[dict[int, np.ndarray], int]:
        """Seeds as box-local voxel coordinates, by group, in the order placed.

        Returns ``(groups, dropped)``. A tool sees only the open box, so seeds
        outside it are left out and counted rather than clamped onto its face.
        """
        if self.seeds_layer is None or self.layer is None or self.box is None:
            return {}, 0
        pts = np.asarray(self.seeds_layer.data, dtype=np.float64)
        if not len(pts):
            return {}, 0
        groups = np.asarray(self.seeds_layer.features["group"], dtype=int)
        local = np.array([self.layer.world_to_data(p) for p in pts], dtype=np.float64)
        top = np.asarray(self.box.shape_kji, dtype=np.float64) - 0.5
        inside = np.all((local >= -0.5) & (local < top), axis=1)
        out = {int(g): local[inside & (groups == g)] for g in np.unique(groups[inside])}
        return out, int(np.count_nonzero(~inside))

    def _crop(self, pts, pad):
        shape = np.asarray(self.box.shape_kji)
        lo = np.maximum(np.floor(pts.min(axis=0)).astype(int) - pad, 0)
        hi = np.minimum(np.ceil(pts.max(axis=0)).astype(int) + pad + 1, shape)
        return tuple(slice(int(a), int(b)) for a, b in zip(lo, hi)), lo

    def _raw(self, lo, sl):
        sampler = self.sampler
        if sampler is None:
            return None
        origin = np.asarray(self.box.origin_kji) + lo
        return sampler.box(origin, tuple(s.stop - s.start for s in sl))

    def run_fill(self, *, group: int | None = None, **kw):
        """Fill the collapsed lumen along one seed group.

        See :func:`.mask_tools.fill_collapsed` for `kw`. The label is the pinned
        one, or, on "auto", the tree the seeds sit beside -- not the box majority,
        which near a junction of the two trees can be the wrong one.
        """
        from .mask_tools import fill_collapsed

        if self.layer is None or self.box is None:
            return self._report(None, "fill: no paint box open")
        groups, dropped = self.seed_groups()
        g = self.seed_group if group is None else int(group)
        pts = groups.get(g)
        if pts is None or not len(pts):
            return self._report(None, f"fill: no group {g} seeds in this box - add "
                                      f"points to '{SEEDS_LAYER}' along the collapsed run")
        sl, lo = self._crop(pts, FILL_PAD)
        labels = np.asarray(self.layer.data)[sl]
        pinned = None if self.label_mode == "auto" else int(self.label_mode)
        result = fill_collapsed(labels, self._raw(lo, sl), pts - lo, value=pinned,
                                names=self.names, **kw)
        return self._apply_result(result, sl, dropped)

    def run_split(self, **kw):
        """Cut a fused blob between seed groups. See :func:`.mask_tools.split_fused`."""
        from .mask_tools import split_fused

        if self.layer is None or self.box is None:
            return self._report(None, "split: no paint box open")
        groups, dropped = self.seed_groups()
        if len(groups) < 2:
            return self._report(None, "split: needs seeds in two groups, one per vessel "
                                      "- change the group number between them")
        pts = np.vstack(list(groups.values()))
        sl, lo = self._crop(pts, SPLIT_PAD)
        labels = np.asarray(self.layer.data)[sl]
        result = split_fused(labels, self._raw(lo, sl),
                             [g - lo for _, g in sorted(groups.items())],
                             names=self.names, **kw)
        if result.ok and self.splice_mode != "replace":
            # 'add' keeps only the chain through *added* voxels, and a cut adds none.
            self.splice_mode = "replace"
            result.reasons.append("splice set to 'replace': a cut adds no voxels, so "
                                  "'add' would leave the graph as it was")
            if self.on_splice is not None:
                self.on_splice(self.splice_mode)
        return self._apply_result(result, sl, dropped)

    def _apply_result(self, result, sl, dropped):
        """Write a tool's result as one undoable step, and show what it did."""
        idx = np.nonzero(result.write)
        if idx[0].size:
            where = tuple(i + s.start for i, s in zip(idx, sl))
            with self.layer.block_history():
                self.layer.data_setitem(where, int(result.value))
            self.layer.colormap = self._colormap(np.asarray(self.layer.data))
        if dropped:
            result.reasons.append(f"{dropped} seed(s) outside this box were ignored")
        if self.result_layer is not None:
            preview = np.zeros(self.box.shape_kji, dtype=np.uint8)
            view = preview[sl]
            view[result.refused] = RESULT_REFUSED
            view[result.write] = RESULT_CUT if result.kind == "split" else RESULT_FILLED
            self.result_layer.data = preview
        self.commit()
        return self._report(result, "")

    def _report(self, result, message):
        self.last_result = result
        text = result.describe() if result is not None else message
        print(f"[paint] {text}")
        self._notify(text)
        return result

    def relabel_strokes(self) -> dict:
        """Give every painted voxel its tree's label. See :func:`.mask_tools.relabel_added`.

        Reloads the layer afterwards, which also clears napari's undo history for
        it: the store changed under the layer, and the next commit's diff would
        otherwise write the old labels straight back.
        """
        from .mask_tools import relabel_added

        self.commit()
        counts = relabel_added(self.source)
        if self.layer is not None and self.box is not None:
            data = self.current_data()
            self.layer.data = data
            self.layer.colormap = self._colormap(data)
        if any(counts.values()):
            self.unsaved = True
        self._notify(f"relabel: {counts['relabelled']:,} given their tree's label, "
                     f"{counts['restored']:,} painted over the other tree restored, "
                     f"{counts['isolated']:,} with no tree within reach left as they are")
        return counts

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
                f"+{s['added']:,} painted  -{s['removed']:,} erased   "
                f"brush: {self.label_name(self.current_label())}"
                + ("  [unsaved]" if self.unsaved else ""))

    def _notify(self, extra: str = "") -> None:
        self.status = self.describe() + (f"\n{extra}" if extra else "")
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
    from qtpy.QtCore import Qt
    from qtpy.QtGui import QColor, QIcon, QPixmap
    from qtpy.QtWidgets import (
        QApplication, QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog, QGroupBox,
        QHBoxLayout, QLabel, QPushButton, QSpinBox, QVBoxLayout, QWidget,
    )

    box = QWidget()
    lay = QVBoxLayout(box)

    spacing = float(np.asarray(session.frame.seg_spacing, dtype=np.float64)[0])
    lay.addWidget(QLabel(
        "<b>Painting the mask</b><br>"
        f"Select <i>{PAINT_LAYER}</i>, then<br>"
        "<b>2</b> paint &nbsp; <b>3</b> fill &nbsp; <b>4</b> erase<br>"
        "<b>[</b> / <b>]</b> brush size &nbsp; <b>Ctrl+Z</b> undo<br>"
        f"<i>One pixel here is one mask voxel<br>({spacing:.2f} &micro;m), "
        "not one raw pixel.</i>"
    ))

    status = QLabel(session.describe())
    status.setWordWrap(True)
    session.on_status = status.setText
    lay.addWidget(status)

    # ---- the label
    row = QHBoxLayout()
    label_combo = QComboBox()
    label_combo.setToolTip("What the brush and the fill write. 'auto' takes the commonest\n"
                           "label in each new box; napari's own picker (5) pins one.")
    row.addWidget(QLabel("label:"))
    row.addWidget(label_combo, 1)
    lay.addLayout(row)

    def _swatch(value):
        pix = QPixmap(12, 12)
        pix.fill(QColor(label_color_dict([value])[value]))
        return QIcon(pix)

    def _fill_labels():
        label_combo.blockSignals(True)
        label_combo.clear()
        label_combo.addItem(f"auto ({session.label_name(session.default_label())})", "auto")
        for name, value in session.label_choices():
            label_combo.addItem(_swatch(value), f"{name} ({value})", value)
        index = label_combo.findData(session.label_mode)
        label_combo.setCurrentIndex(max(index, 0))
        label_combo.blockSignals(False)

    _fill_labels()
    label_combo.currentIndexChanged.connect(
        lambda i: session.set_label(label_combo.itemData(i)))
    session.on_label = _fill_labels
    box._hipct_fill_labels = _fill_labels

    preserve = QCheckBox("protect other labels")
    preserve.setToolTip("Paint only over background and the brush's own label, and erase\n"
                        "only the brush's own label -- so fixing one tree cannot\n"
                        "overwrite the other.")
    preserve.setChecked(session.preserve_labels)
    preserve.toggled.connect(session.set_preserve)
    lay.addWidget(preserve)

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

    relabel = QPushButton("Relabel painted voxels to their tree")
    relabel.setToolTip("For edits painted before the label picker existed, when every\n"
                       "stroke wrote 1: each painted voxel takes the label of the tree\n"
                       "it touches, and strokes over the other tree are undone.")
    relabel.clicked.connect(lambda: _guarded(session.relabel_strokes))
    lay.addWidget(relabel)

    # ---- seeded repairs
    tools = QGroupBox("Repair tools")
    tl = QVBoxLayout(tools)
    tl.addWidget(QLabel(
        f"Add points to <i>{SEEDS_LAYER}</i> along a vessel.<br>"
        "<b>Fill</b>: one group along the collapsed run.<br>"
        "<b>Split</b>: one group per fused vessel.<br>"
        "<span style='color:#00e676'>green</span> filled, "
        "<span style='color:#ffea00'>yellow</span> cut, "
        "<span style='color:#ff1744'>red</span> refused; Ctrl+Z undoes."
    ))

    row = QHBoxLayout()
    group = QSpinBox()
    group.setRange(1, 9)
    group.setValue(session.seed_group)
    group.setToolTip("The group the next seed joins. A split needs two.")
    group.valueChanged.connect(session.set_seed_group)
    clear = QPushButton("Clear seeds")
    clear.clicked.connect(session.clear_seeds)
    row.addWidget(QLabel("seed group"))
    row.addWidget(group)
    row.addWidget(clear)
    tl.addLayout(row)

    row = QHBoxLayout()
    radius = QDoubleSpinBox()
    radius.setRange(0.0, 60.0)
    radius.setDecimals(1)
    radius.setSpecialValueText("auto")
    radius.setSuffix(" vox")
    radius.setToolTip("Fill capsule radius around the seeds, in mask voxels.\n"
                      "auto = 1.5 x the radius of the mask beside the gap.")
    row.addWidget(QLabel("fill radius"))
    row.addWidget(radius)
    tl.addLayout(row)

    row = QHBoxLayout()
    allow_join = QCheckBox("allow join")
    allow_join.setToolTip("Let a fill reconnect two separate pieces of the same tree --\n"
                          "for a collapse that broke the vessel right through.")
    force = QCheckBox("force split")
    force.setToolTip("Cut even where the boundary looks like a healthy lumen.")
    width = QSpinBox()
    width.setRange(1, 3)
    width.setPrefix("cut ")
    width.setSuffix(" vox")
    row.addWidget(allow_join)
    row.addWidget(force)
    row.addWidget(width)
    tl.addLayout(row)

    row = QHBoxLayout()
    fill_btn = QPushButton("Fill collapsed lumen")
    split_btn = QPushButton("Split fused vessels")
    row.addWidget(fill_btn)
    row.addWidget(split_btn)
    tl.addLayout(row)

    def _busy(fn):
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            _guarded(fn)
        finally:
            QApplication.restoreOverrideCursor()

    fill_btn.clicked.connect(lambda: _busy(lambda: session.run_fill(
        roi_radius=radius.value() or None, allow_join=allow_join.isChecked())))
    split_btn.clicked.connect(lambda: _busy(lambda: session.run_split(
        force=force.isChecked(), cut_width=width.value())))
    lay.addWidget(tools)

    row = QHBoxLayout()
    mode = QComboBox()
    mode.addItems(["add", "replace"])
    mode.setCurrentText(session.splice_mode)
    mode.currentTextChanged.connect(lambda t: setattr(session, "splice_mode", t))
    session.on_splice = mode.setCurrentText
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


def _guarded(fn):
    """Run a panel action; report a failure instead of letting it abort Qt.

    PyQt5 calls qFatal on an exception that escapes a slot, so a tool that trips on
    an odd crop would otherwise take the window -- and the unsaved edits -- with it.
    """
    try:
        return fn()
    except Exception as exc:  # noqa: BLE001 - see the docstring
        print(f"[paint] failed: {type(exc).__name__}: {exc}")
        return None


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
    """Catch the panel up after a pick: the re-skeletonise wiring and the label list."""
    button = getattr(panel, "_hipct_reskel_button", None)
    if button is not None:
        button.setEnabled(session.on_reskeletonise is not None)
    fill_labels = getattr(panel, "_hipct_fill_labels", None)
    if fill_labels is not None:
        fill_labels()
