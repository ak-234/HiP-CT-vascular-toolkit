"""Geometric-mean (Reduced Major Axis / Model-II) regression.

Faithful port of the third-party ``gmregress.m`` and ``gmregresspi.m`` by
A. Trujillo-Ortiz, R. Hernandez-Walls et al. (BSD-2-clause, see
``Other_useful_scripts/license.txt``). Model-II regression is appropriate when
both variables carry error, as with the log(tip-count) vs log(radius) scaling
relationship in :mod:`skeleton_analysis.metrics.exponent`.

References
---------
Ricker, W. E. (1973). Linear regression in fishery research. J. Fish. Res.
Board Can., 30:409-434.
Jolicoeur, P. & Mosimann, J. E. (1968). Biometrie-Praximetrie, 9:121-140.
McArdle, B. (1988). Can. J. Zool. 66:2329-2339.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from scipy import stats


@dataclass
class GMRegressResult:
    """Result of :func:`gmregress`."""

    intercept: float
    slope: float
    ricker_ci: np.ndarray  # 2x2 [[intercept_lo, intercept_hi],[slope_lo, slope_hi]]
    jm_ci: np.ndarray  # 2x2 Jolicoeur-Mosimann / McArdle CI, same layout
    n: int
    r: float  # correlation coefficient

    @property
    def coefficients(self) -> np.ndarray:
        """``[intercept, slope]`` (matches MATLAB ``b``)."""
        return np.array([self.intercept, self.slope])


def _clean(x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    if x.shape != y.shape:
        raise ValueError("x and y must have the same length")
    mask = ~(np.isnan(x) | np.isnan(y))
    return x[mask], y[mask]


def _rma_core(x: np.ndarray, y: np.ndarray):
    """Shared RMA slope/intercept and sums of squares."""
    n = x.size
    mx, my = x.mean(), y.mean()
    dx, dy = x - mx, y - my
    SCX = float(np.sum(dx * dx))
    SCY = float(np.sum(dy * dy))
    SCP = float(np.sum(dx * dy))
    r = SCP / np.sqrt(SCX * SCY)
    s = np.sign(r) if r != 0 else 1.0
    slope = s * np.sqrt(SCY / SCX)
    intercept = my - mx * slope
    return n, mx, my, SCX, SCY, SCP, r, slope, intercept


def gmregress(x, y, alpha: float = 0.05) -> GMRegressResult:
    """Geometric-mean (RMA) regression of ``y`` on ``x``.

    Returns slope, intercept and the Ricker and Jolicoeur-Mosimann/McArdle
    confidence intervals for ``[intercept, slope]``. NaNs are dropped pairwise.
    """
    x, y = _clean(x, y)
    if x.size < 3:
        raise ValueError("gmregress requires at least 3 valid points")
    n, mx, my, SCX, SCY, SCP, r, slope, intercept = _rma_core(x, y)

    # Ricker (1973) CI.
    SCv = SCY - (SCP ** 2) / SCX
    N = SCv / (n - 2)
    sv = np.sqrt(N / SCX)
    t = stats.t.ppf(1 - alpha / 2, n - 2)
    vi, vs = slope - t * sv, slope + t * sv
    ui, us = my - mx * vs, my - mx * vi
    ricker = np.array([sorted([ui, us]), sorted([vi, vs])])

    # Jolicoeur & Mosimann (1968) / McArdle (1988) CI.
    F = stats.f.ppf(1 - alpha, 1, n - 2)
    B = F * (1 - r ** 2) / (n - 2)
    a = np.sqrt(B + 1)
    c = np.sqrt(B)
    qi, qs = slope * (a - c), slope * (a + c)
    pi, ps = my - mx * qs, my - mx * qi
    jm = np.array([sorted([pi, ps]), sorted([qi, qs])])

    return GMRegressResult(
        intercept=float(intercept),
        slope=float(slope),
        ricker_ci=ricker,
        jm_ci=jm,
        n=int(n),
        r=float(r),
    )


def gmregresspi(x, y, xo: float, alpha: float = 0.05):
    """RMA regression plus a prediction interval for a single new ``xo``.

    Port of ``gmregresspi.m``. Returns ``(coefficients, yo, se, pint)`` where
    ``coefficients = [intercept, slope]``, ``yo`` is the predicted value at
    ``xo``, ``se`` its standard error and ``pint = [lo, hi]``.
    """
    x, y = _clean(x, y)
    if x.size < 3:
        raise ValueError("gmregresspi requires at least 3 valid points")
    n, mx, my, SCX, SCY, SCP, r, slope, intercept = _rma_core(x, y)

    ye = intercept + slope * x
    sde = np.sqrt(np.sum(np.abs((y - ye) * (x - ((y - intercept) / slope)))) / (n - 2))
    se = sde * np.sqrt(1 + (1 / n) + (xo - mx) ** 2 / SCX)
    t = stats.t.ppf(1 - alpha / 2, n - 2)
    yo = intercept + slope * xo
    pint = np.array([yo - t * se, yo + t * se])
    return np.array([intercept, slope]), float(yo), float(se), pint
