"""Passive RCR parameters, impedance fitting and a restartable reference integrator."""
from __future__ import annotations

import numpy as np
from scipy.optimize import least_squares

from .network import positive


def impedance(omega, rp, rd, compliance):
    return rp + rd/(1+1j*np.asarray(omega)*rd*compliance)


def parameters(rt, radius_m, area_m2, density, spec, samples=None):
    rt = positive(rt, "total resistance")
    mode = spec["mode"]
    residual = None
    if mode == "scapis":
        c = 13.3/(2*positive(radius_m, "outlet radius")*1000)**0.3
        rp = positive(density, "density")*c/positive(area_m2, "outlet area")
        rd = rt-rp
        compliance = positive(spec.get("tau_total_s", 1.79), "R_T C")/rt
    elif mode == "fixed":
        alpha = float(spec["alpha"])
        if not 0 < alpha < 1:
            raise ValueError("alpha must be strictly between 0 and 1")
        rp, rd = alpha*rt, (1-alpha)*rt
        compliance = positive(spec["tau_distal_s"], "R_d C")/rd
    elif mode == "impedance":
        if samples is None:
            raise ValueError("Impedance fit requires explicit frequency/complex-impedance samples")
        arr = np.asarray(samples, dtype=float)
        if arr.ndim != 2 or arr.shape[1] != 3 or not np.all(np.isfinite(arr)):
            raise ValueError("Impedance rows must be [frequency_hz, real, imaginary]")
        if np.any(arr[:, 0] < 0) or len(np.unique(arr[:, 0])) != len(arr):
            raise ValueError("Impedance frequencies must be unique and nonnegative")
        dc = arr[arr[:, 0] == 0]
        if len(dc) and (not np.isclose(dc[0, 1], rt, rtol=1e-4) or abs(dc[0, 2]) > rt*1e-6):
            raise ValueError("Impedance DC value differs from calibrated R_T")
        arr = arr[arr[:, 0] > 0]
        arr = arr[np.argsort(arr[:, 0])][:int(spec.get("harmonics", 10))]
        if len(arr) < 2:
            raise ValueError("At least two positive-frequency samples are required")
        omega, target = 2*np.pi*arr[:, 0], arr[:, 1]+1j*arr[:, 2]
        # Equal harmonic weights, normalised by DC resistance (not tiny sample magnitudes).
        def objective(x):
            alpha, tau = x[0], np.exp(x[1])
            z = impedance(omega, alpha*rt, (1-alpha)*rt, tau/((1-alpha)*rt))
            e = (z-target)/rt
            return np.r_[e.real, e.imag]
        candidates = [least_squares(objective, [a, np.log(t)],
                      bounds=([1e-8, np.log(1e-6)], [1-1e-8, np.log(1e4)]),
                      gtol=1e-12, ftol=1e-12, xtol=1e-12)
                      for a in (0.05, 0.3, 0.8) for t in (0.05, 1., 10.)]
        fit = min(candidates, key=lambda x: x.cost)
        if not fit.success:
            raise ValueError("RCR impedance optimisation did not converge")
        rp, rd = fit.x[0]*rt, (1-fit.x[0])*rt
        compliance = np.exp(fit.x[1])/rd
        residual = float(np.sqrt(np.mean(np.abs(objective(fit.x))**2)))
    else:
        raise ValueError(f"Unknown RCR parameterisation {mode}")
    for name, value in (("Rp", rp), ("Rd", rd), ("C", compliance)):
        if not np.all(np.isfinite(value)) or np.any(np.asarray(value) <= 0):
            raise ValueError(f"{name} must be finite and positive")
    return dict(R_T_Pa_s_per_m3=rt, Rp_Pa_s_per_m3=float(rp), Rd_Pa_s_per_m3=float(rd),
                C_m3_per_Pa=float(compliance), tau_total_s=float(rt*compliance),
                tau_distal_s=float(rd*compliance), rcr_mode=mode, fit_residual=residual)


def step(pc, q_lag, dt, rp, rd, compliance, pv):
    """Backward Euler capacitor update, frozen lagged Q; pressure is relative to CFX datum."""
    for value, name in ((dt, "dt"), (rp, "Rp"), (rd, "Rd"), (compliance, "C")):
        if not np.all(np.isfinite(value)) or np.any(np.asarray(value) <= 0):
            raise ValueError(f"{name} must be finite and positive")
    pc, q_lag, rp, rd, compliance, pv = np.broadcast_arrays(pc, q_lag, rp, rd, compliance, pv)
    if not all(np.all(np.isfinite(x)) for x in (pc, q_lag, pv)):
        raise ValueError("Non-finite RCR state")
    beta = dt/(rd*compliance)
    updated = (pc+dt*q_lag/compliance+beta*pv)/(1+beta)
    return updated, updated+rp*q_lag


class RCRState:
    """Frozen start-of-step state and idempotent trial/commit semantics.

    JSON checkpoints are portable and include the absolute last committed step.
    Call begin once, trial any number of coefficient loops, then commit once.
    """
    def __init__(self, pc, q, step_index=-1):
        self.pc = np.atleast_1d(np.asarray(pc, float)).copy()
        self.q = np.atleast_1d(np.asarray(q, float)).copy()
        if self.pc.shape != self.q.shape or not np.all(np.isfinite([self.pc, self.q])):
            raise ValueError("Invalid checkpoint state")
        self.step_index = int(step_index)
        self.pending = None

    def begin(self, index, q):
        if index <= self.step_index or self.pending is not None:
            raise ValueError("Step already committed or an uncommitted step is active")
        q = np.atleast_1d(np.asarray(q, float))
        if q.shape != self.q.shape or not np.all(np.isfinite(q)):
            raise ValueError("Invalid outlet flows")
        self.pending = (int(index), q.copy())

    def trial(self, dt, rp, rd, compliance, pv):
        if self.pending is None:
            raise ValueError("No active step")
        self.pc_trial, pressure = step(self.pc, self.pending[1], dt, rp, rd, compliance, pv)
        return pressure.copy()

    def commit(self):
        if self.pending is None or not hasattr(self, "pc_trial"):
            raise ValueError("No evaluated step to commit")
        self.step_index, self.q = self.pending
        self.pc = self.pc_trial.copy()
        self.pending = None
        del self.pc_trial

    def checkpoint(self):
        if self.pending is not None:
            raise ValueError("Checkpoint only at a completed time step")
        return dict(pc=self.pc.tolist(), q=self.q.tolist(), step_index=self.step_index)
