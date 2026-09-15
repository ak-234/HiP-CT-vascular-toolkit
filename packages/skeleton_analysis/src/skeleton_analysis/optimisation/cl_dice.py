"""Centreline Dice (clDice) family of overlap metrics.

Cleaned, parameterised version of the original ``cl_dice.py``: hard-coded paths
and blocking ``matplotlib`` calls are removed, ``main()`` becomes the callable
:func:`run_cl_sensitivity`, and 3-D skeletonisation uses :func:`skimage.morphology.skeletonize`
(which handles 3-D in modern scikit-image; the deprecated ``skeletonize_3d`` is
used only as a fallback).

Requires the ``[image]`` extra (scikit-image, tifffile). The numeric helpers
(:func:`cl_score`, :func:`dice`) are pure numpy and always importable.
"""

from __future__ import annotations

import glob
import os
from pathlib import Path
from typing import List, Optional

import numpy as np


def _skeletonize(volume: np.ndarray) -> np.ndarray:
    from skimage.morphology import skeletonize  # lazy: [image] extra

    return skeletonize(volume)


def read_image_stack(path, show_slice: Optional[int] = None) -> np.ndarray:
    """Load a multipage 3-D TIFF stack. ``show_slice`` optionally displays one slice."""
    import tifffile  # lazy: [image] extra

    with tifffile.TiffFile(str(path)) as tif:
        data = tif.asarray()
    if show_slice is not None:  # pragma: no cover - interactive
        import matplotlib.pyplot as plt

        plt.imshow(data[show_slice], cmap="gray")
        plt.title(f"Slice {show_slice}")
        plt.axis("off")
        plt.show()
    return data


def cl_score(v: np.ndarray, s: np.ndarray, normalize: float = 255.0) -> float:
    """Skeleton-volume overlap = sum(v*s)/sum(s).

    ``v`` (volume) and ``s`` (skeleton) are divided by ``normalize`` (255 for
    8-bit masks; pass ``1`` for boolean inputs).
    """
    v = np.divide(v, normalize)
    s = np.divide(s, normalize)
    denom = np.sum(s)
    return float(np.sum(v * s) / denom) if denom else float("nan")


def dice(v_p: np.ndarray, v_l: np.ndarray) -> float:
    """Standard Dice coefficient between two boolean volumes."""
    v_p = v_p.astype(bool)
    v_l = v_l.astype(bool)
    denom = np.sum(v_l) + np.sum(v_p)
    return float(2.0 * np.sum(v_l & v_p) / denom) if denom else float("nan")


def cl_dice(v_p: np.ndarray, v_l: np.ndarray, normalize: float = 255.0) -> float:
    """clDice between a predicted volume ``v_p`` and ground truth ``v_l``.

    Skeletons are computed internally with scikit-image.
    """
    tprec = cl_score(v_p, _skeletonize(v_l), normalize)
    tsens = cl_score(v_l, _skeletonize(v_p), normalize)
    denom = tprec + tsens
    return float(2 * tprec * tsens / denom) if denom else float("nan")


def cl_dice_with_skeletons(v_p, s_p, v_l, s_l, normalize: float = 255.0) -> float:
    """clDice when both skeletons are supplied precomputed."""
    tprec = cl_score(v_p, s_l, normalize)
    tsens = cl_score(v_l, s_p, normalize)
    denom = tprec + tsens
    return float(2 * tprec * tsens / denom) if denom else float("nan")


def cl_sensitivity(v_l, s_p, normalize: float = 255.0) -> float:
    """Centreline sensitivity: overlap of a predicted skeleton with the GT volume."""
    return cl_score(v_l, s_p, normalize)


def run_cl_sensitivity(
    ground_truth_path,
    prediction_dir,
    pattern: str = "*converted.tif",
    output_csv: Optional[str] = "cl_sensitivity_result.csv",
    strip_padding: int = 0,
    normalize: float = 255.0,
) -> "pd.DataFrame":  # type: ignore[name-defined]
    """Compute centreline sensitivity for every prediction TIFF in a directory.

    Parameterised replacement for the original ``main()``. ``strip_padding``
    removes an N-voxel border from each prediction (Amira autoskeleton adds 15).
    Returns (and optionally writes) a DataFrame of ``Name, Cl_sense``.
    """
    import pandas as pd

    v_l = read_image_stack(ground_truth_path)
    rows: List[dict] = []
    for file in sorted(glob.glob(os.path.join(str(prediction_dir), pattern))):
        s_p = read_image_stack(file)
        if strip_padding:
            p = strip_padding
            s_p = s_p[p:-p, p:-p, p:-p]
        if v_l.shape != s_p.shape:
            raise ValueError(
                f"Shape mismatch: ground truth {v_l.shape} vs {file} {s_p.shape}"
            )
        rows.append({"Name": os.path.basename(file),
                     "Cl_sense": cl_sensitivity(v_l, s_p, normalize)})

    out = pd.DataFrame(rows)
    if output_csv:
        out.to_csv(Path(prediction_dir) / output_csv, index=False)
    return out
