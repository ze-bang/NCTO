"""Light-weight observable extraction helpers for the campaign.

Kept separate from common.py to avoid pulling numpy into the orchestrator
when only filesystem work is needed.  Imported by C7, C8, C9 and the
plotting routines of C1--C6.
"""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import numpy as np


# --------------------------------------------------------------------- M-points
def honeycomb_positions(Lx: int, Ly: int) -> np.ndarray:
    a1 = np.array([1.0, 0.0])
    a2 = np.array([0.5, np.sqrt(3) / 2])
    sub = np.array([[0.0, 0.0], [0.5, 1.0 / (2 * np.sqrt(3))]])
    pos = [n1 * a1 + n2 * a2 + sub[s]
           for n1 in range(Lx) for n2 in range(Ly) for s in range(2)]
    return np.asarray(pos)


M_VECTORS = [
    np.array([np.pi, np.pi / np.sqrt(3)]),
    np.array([0.0,    2 * np.pi / np.sqrt(3)]),
    np.array([-np.pi, np.pi / np.sqrt(3)]),
]


def mpoint_amplitudes(spins: np.ndarray, positions: np.ndarray) -> np.ndarray:
    """Return [|S(M1)|, |S(M2)|, |S(M3)|] for one (n_sites,3) configuration."""
    out = np.zeros(3)
    for i, M in enumerate(M_VECTORS):
        ph = np.exp(1j * positions @ M)
        amp2 = sum(np.abs((ph * spins[:, c]).sum()) ** 2 for c in range(3))
        out[i] = np.sqrt(amp2) / spins.shape[0]
    return out


# --------------------------------------------------------------------- IO
def load_spin_txt(path: Path) -> np.ndarray:
    """Load a spins_*.txt file produced by spin_solver (n_sites x 3)."""
    arr = np.loadtxt(path)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 3)
    return arr[:, -3:]


def read_csv(path: Path) -> List[dict]:
    with Path(path).open() as fh:
        return [{k: _maybe_float(v) for k, v in row.items()}
                for row in csv.DictReader(fh)]


def _maybe_float(v: str):
    try:
        return float(v)
    except Exception:
        return v


# --------------------------------------------------------------------- aggregation
def angle_harmonics(theta_rad: np.ndarray, signal: np.ndarray,
                    orders: Sequence[int] = (0, 2, 4, 6)) -> dict:
    """Project signal(theta) onto cos/sin Fourier modes.  Returns
    {n: (a_n, b_n)} where signal ~ a_0/2 + sum_n a_n cos(n theta) + b_n sin(n theta).
    """
    th = np.asarray(theta_rad)
    s = np.asarray(signal)
    N = len(th)
    out = {}
    for n in orders:
        a = (2.0 / N) * np.sum(s * np.cos(n * th))
        b = (2.0 / N) * np.sum(s * np.sin(n * th))
        out[n] = (a, b)
    return out


def fit_avrami(F: np.ndarray, f_zz: np.ndarray) -> Tuple[float, float, float]:
    """Fit f_zz = f_sat (1 - exp(-(F/F0)**m)).  Returns (f_sat, F0, m)."""
    from scipy.optimize import curve_fit

    def model(F, fsat, F0, m):
        return fsat * (1.0 - np.exp(-(F / F0) ** m))

    p0 = (float(np.max(f_zz)) * 1.05,
          float(F[np.argmax(np.gradient(f_zz))]),
          2.0)
    try:
        popt, _ = curve_fit(model, F, f_zz, p0=p0, maxfev=10000)
    except Exception:
        return p0
    return tuple(popt)
