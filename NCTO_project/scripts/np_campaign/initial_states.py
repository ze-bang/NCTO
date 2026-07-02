"""Initial-state generators for the campaign.

These produce site-resolved (n_sites, 3) spin arrays in the canonical
honeycomb ordering used by spin_solver:
    site_index = 2 * (Ly * n1 + n2) + sublattice  (n1 = 0..Lx-1, n2 = 0..Ly-1)
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from analysis_utils import honeycomb_positions, M_VECTORS


def _save(path: Path, spins: np.ndarray) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(path, spins, fmt="%.10f")
    return path


def triple_q_state(Lx: int, Ly: int, rng=None) -> np.ndarray:
    """Equal-weight superposition of the three M-point single-Q components,
    normalised site-wise so |S_i| = 1."""
    rng = rng or np.random.default_rng(0xD0)
    pos = honeycomb_positions(Lx, Ly)
    # Three orthogonal "spin colors" e_x, e_y, e_z attached to each M.
    cols = [np.array([1.0, 0.0, 0.0]),
            np.array([0.0, 1.0, 0.0]),
            np.array([0.0, 0.0, 1.0])]
    spins = np.zeros((pos.shape[0], 3))
    for M, e in zip(M_VECTORS, cols):
        amp = np.cos(pos @ M)
        spins += np.outer(amp, e)
    norm = np.linalg.norm(spins, axis=1, keepdims=True)
    norm[norm < 1e-9] = 1.0
    return spins / norm


def single_zigzag_state(Lx: int, Ly: int, m_index: int = 0) -> np.ndarray:
    """Collinear z-axis spin texture modulated at the chosen M-vector."""
    pos = honeycomb_positions(Lx, Ly)
    sgn = np.sign(np.cos(pos @ M_VECTORS[m_index]))
    sgn[sgn == 0.0] = 1.0
    spins = np.zeros((pos.shape[0], 3))
    spins[:, 2] = sgn
    return spins


def droplet_state(Lx: int, Ly: int, R: float, *,
                  center=None, m_index: int = 0) -> np.ndarray:
    """3Q background with a circular zigzag inclusion of radius R."""
    pos = honeycomb_positions(Lx, Ly)
    if center is None:
        center = np.array([Lx / 2.0, Ly * math.sqrt(3) / 4.0])
    r = np.linalg.norm(pos - center, axis=1)
    base = triple_q_state(Lx, Ly)
    zz = single_zigzag_state(Lx, Ly, m_index=m_index)
    mask = (r < R)[:, None]
    return np.where(mask, zz, base)


def save_state(path: Path, spins: np.ndarray) -> Path:
    return _save(path, spins)
