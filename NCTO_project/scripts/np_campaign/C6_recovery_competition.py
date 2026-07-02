#!/usr/bin/env python3
"""C6 — recovery competition (real-LLG droplet rewrite).

For mechanisms A (Allen-Cahn curvature) and B (Imry-Ma pinning) we
prepare a 3Q matrix containing a circular zigzag inclusion of radius
R0 and integrate with the in-house stochastic Heun sLLG.  Mechanism C
(diffusion bottleneck) remains analytic because nothing in the spin
code captures a slow ionic/electronic field.

The droplet radius is estimated each frame as
    R(t) = sqrt( N_ZZ(t) / pi )
where N_ZZ counts sites whose local order parameter is closer to a
single-M zigzag basis vector than to the 3Q reference.

References:
  Allen & Cahn, Acta Met. 27, 1085 (1979);
  Bray, Adv. Phys. 43, 357 (1994);
  Schryer & Walker, JAP 45, 5406 (1974);
  Imry & Ma, PRL 35, 1399 (1975).
"""
from __future__ import annotations

import argparse
import math
import subprocess
from pathlib import Path

import numpy as np

from analysis_utils import M_VECTORS, honeycomb_positions
from common import CAMPAIGN_OUT, REPO_ROOT, SPIN_SOLVER, write_csv
from configs_lib import write_langevin_relax_config
from initial_states import droplet_state, save_state

PHASE = CAMPAIGN_OUT / "C6"
PHASE.mkdir(parents=True, exist_ok=True)


def _run(cfg: Path, log: Path) -> int:
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w") as fh:
        proc = subprocess.run([str(SPIN_SOLVER), str(cfg)],
                              cwd=REPO_ROOT, stdout=fh,
                              stderr=subprocess.STDOUT)
    return proc.returncode


def _measure_radius(spin_dir: Path, Lx: int, Ly: int) -> list:
    """Return [(t, R), ...] from spin snapshot files in spin_dir."""
    pos = honeycomb_positions(Lx, Ly)
    out = []
    files = sorted(spin_dir.glob("spin_*.txt")) + sorted(spin_dir.glob("snapshot_*.txt"))
    for f in files:
        try:
            t = float(f.stem.split("_")[-1])
        except ValueError:
            continue
        try:
            S = np.loadtxt(f)
        except Exception:
            continue
        if S.ndim != 2 or S.shape[1] < 3 or S.shape[0] != pos.shape[0]:
            continue
        # Local M-projection: choose dominant M for each site by sublattice mod.
        amp = [np.cos(pos @ M) for M in M_VECTORS]
        proj = np.stack([np.abs(S[:, k] * a) for k, a in enumerate(amp)], axis=1)
        # In 3Q all three are comparable; in ZZ inclusion one M dominates.
        dom = proj.max(axis=1) / (proj.sum(axis=1) + 1e-9)
        n_zz = int((dom > 0.6).sum())
        R = math.sqrt(max(n_zz, 0) / math.pi)
        out.append((t, R))
    out.sort()
    return out


def _run_one(tag: str, *, disorder_strength: float,
             disorder_seed: int, R0: float,
             T: float, lattice, t_max: float) -> Path:
    Lx, Ly, Lz = lattice
    seed = PHASE / "seeds" / f"{tag}.txt"
    save_state(seed, droplet_state(Lx, Ly, R0))
    outdir = PHASE / "runs" / tag
    outdir.mkdir(parents=True, exist_ok=True)
    cfg = PHASE / "configs" / f"{tag}.param"
    write_langevin_relax_config(
        cfg, str(outdir.relative_to(REPO_ROOT)),
        seed_file=str(seed.relative_to(REPO_ROOT)),
        T=T, t_max=t_max, lattice_size=lattice,
        disorder_strength=disorder_strength,
        disorder_seed=disorder_seed,
    )
    _run(cfg, PHASE / "logs" / f"{tag}.log")
    return outdir


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()

    lattice = (16, 16, 1) if args.quick else (24, 24, 1)
    Lx, Ly, _ = lattice
    R0 = 4.0 if args.quick else 6.0
    t_max = 200.0 if args.quick else 800.0
    T_meas = 0.01

    # Clean (mechanism A)
    out_clean = _run_one("clean", disorder_strength=0.0, disorder_seed=0,
                         R0=R0, T=T_meas, lattice=lattice, t_max=t_max)
    write_csv(PHASE / "droplet_radius_clean.csv",
              [{"t": t, "R": R} for t, R in _measure_radius(out_clean, Lx, Ly)])

    # Disordered (mechanism B)
    rows_all = []
    for seed in (1, 2, 3):
        tag = f"dis_seed{seed}"
        out = _run_one(tag, disorder_strength=0.05, disorder_seed=seed,
                       R0=R0, T=T_meas, lattice=lattice, t_max=t_max)
        for t, R in _measure_radius(out, Lx, Ly):
            rows_all.append({"seed": seed, "t": t, "R": R})
    write_csv(PHASE / "droplet_radius_disordered.csv", rows_all)
    print(f"C6 done: clean -> droplet_radius_clean.csv, "
          f"disordered -> droplet_radius_disordered.csv")


if __name__ == "__main__":
    main()
