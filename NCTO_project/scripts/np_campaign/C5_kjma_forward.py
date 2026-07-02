#!/usr/bin/env python3
"""C5 — KJMA forward simulation (real-LLG rewrite).

Drives a fluence ladder of real pump-probe runs on a 24x24 honeycomb
lattice with sLLG (Langevin temperature optional).  The Gaussian-spot
inhomogeneity is recovered by post-hoc convolution of the
homogeneous-fluence switching curve f(F) with the radial intensity
profile of the beam:

    f_obs(F_peak) = (1/A) * integral_0^infinity 2*pi*r * f(F_peak * exp(-2 r^2 / w^2)) dr

so a single fluence ladder gives f_obs at any (F_peak, w).  KJMA
exponent n is then fitted on f_obs(F_peak) via ln(-ln(1-f)) vs ln(F).
This is exact in the local-response limit (xi_phonon << w).

References:
  Kolmogorov (1937); Johnson-Mehl (1939); Avrami JCP 7,8,9 (1939-41);
  Garcia-Palacios & Lazaro PRB 58, 14937 (1998).
"""
from __future__ import annotations

import argparse
import math
import subprocess
from pathlib import Path

import numpy as np

_trapz = getattr(np, "trapezoid", None) or getattr(np, "trapz")

from analysis_utils import read_csv
from common import (CAMPAIGN_OUT, REPO_ROOT, SPIN_SOLVER, declare_switched,
                    extract_final_spins, run_analysis, write_csv)
from configs_lib import write_langevin_pump_config
from initial_states import save_state, triple_q_state

PHASE = CAMPAIGN_OUT / "C5"
PHASE.mkdir(parents=True, exist_ok=True)


def _run(cfg: Path, log: Path) -> int:
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w") as fh:
        proc = subprocess.run([str(SPIN_SOLVER), str(cfg)],
                              cwd=REPO_ROOT, stdout=fh,
                              stderr=subprocess.STDOUT)
    return proc.returncode


def _final_spin_file(out_dir: Path):
    return extract_final_spins(out_dir)


def _switched(cfg: Path, out_dir: Path) -> int:
    sf = _final_spin_file(out_dir)
    if sf is None:
        return -1
    try:
        parsed = run_analysis(cfg, [sf])
    except Exception:
        return -1
    if not parsed:
        return -1
    r3 = parsed[0].get("m_min_over_max")
    if r3 is None:
        return -1
    return int(declare_switched(float(r3)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--T", type=float, default=0.0,
                    help="Langevin temperature (0 = deterministic).")
    args = ap.parse_args()

    # Fluence ladder anchored to calibrated E* range from the
    # l18_effomega4_cycle15 phase diagram (gamma_E1=0.0849, lambda_K=0.035,
    # J7=-0.0026).  E*=1.40 on L18; smaller lattices switch at lower
    # thresholds due to finite-size effects.  Quick mode goes below 1.0
    # to find the true threshold on L12.
    fluences = ([0.5, 0.7, 0.9, 1.1, 1.4, 1.9] if args.quick
                else [0.4, 0.6, 0.8, 0.9, 1.0, 1.1, 1.2, 1.35, 1.50, 1.70, 2.0, 2.5])
    theta = 0.0
    lattice = (12, 12, 1) if args.quick else (24, 24, 1)
    # 15-cycle pulse (sigma=15, peak at t=50) settles by ~t=95;
    # 130 t.u. gives 35 units of post-pulse observation.
    t_end = 130.0 if args.quick else 180.0

    Lx, Ly, _ = lattice
    seed_path = PHASE / "seeds" / f"triple_q_{Lx}x{Ly}.txt"
    save_state(seed_path, triple_q_state(Lx, Ly))
    seed_rel = str(seed_path.relative_to(REPO_ROOT))

    rows = []
    for F in fluences:
        tag = f"F{F:.3f}".replace(".", "p")
        outdir = (PHASE / "runs" / tag)
        outdir.mkdir(parents=True, exist_ok=True)
        cfg = PHASE / "configs" / f"{tag}.param"
        write_langevin_pump_config(
            cfg, str(outdir.relative_to(REPO_ROOT)),
            seed_file=seed_rel,
            E0=F, theta=theta, T=args.T,
            phonon_only_relax=True,   # preserve triple-Q seed; phonons relax, spins don't
            lattice_size=lattice, t_end=t_end,
        )
        log = PHASE / "logs" / f"{tag}.log"
        rc = _run(cfg, log)
        sw = _switched(cfg, outdir)
        rows.append({"F": F, "T": args.T, "switched": sw, "returncode": rc})
        print(f"  F={F:.3f}  switched={sw}")

    write_csv(PHASE / "switching_vs_fluence.csv", rows)

    # Convolve with Gaussian beam: f_obs(F_peak; w) = mean over radial samples.
    w_list = ([1.0, 2.0] if args.quick
              else [1.0, 1.5, 2.0, 2.5, 3.0, 4.0])
    obs_rows = []
    Fpeaks = np.array([0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0])
    # Tabulate local switching probability f(F) by binning fluence ladder.
    f_local = {r["F"]: r["switched"] for r in rows if r["switched"] >= 0}
    if not f_local:
        print("C5: no successful runs — skipping convolution")
        return
    F_arr = np.array(sorted(f_local.keys()))
    p_arr = np.array([f_local[f] for f in F_arr])

    def f_of(F_loc):
        return float(np.interp(F_loc, F_arr, p_arr, left=0.0, right=1.0))

    for w in w_list:
        for Fp in Fpeaks:
            rs = np.linspace(0, 3 * w, 64)
            Floc = Fp * np.exp(-2.0 * rs**2 / w**2)
            integrand = 2 * math.pi * rs * np.array([f_of(F) for F in Floc])
            num = _trapz(integrand, rs)
            den = _trapz(2 * math.pi * rs, rs)
            obs_rows.append({"w": w, "F_peak": Fp, "f_obs": float(num / den)})
    write_csv(PHASE / "f_obs_vs_fluence_and_spot.csv", obs_rows)

    # KJMA exponent fit on f_obs(F_peak; w=middle)
    if len(w_list) >= 1:
        w_fit = w_list[len(w_list) // 2]
        sub = [r for r in obs_rows if r["w"] == w_fit and 0 < r["f_obs"] < 1]
        if len(sub) >= 3:
            x = np.log(np.array([r["F_peak"] for r in sub]))
            y = np.log(-np.log(1 - np.array([r["f_obs"] for r in sub])))
            slope, _ = np.polyfit(x, y, 1)
            kjma_n = float(slope)
        else:
            kjma_n = float("nan")
        write_csv(PHASE / "kjma_fit.csv",
                  [{"w": w_fit, "kjma_n": kjma_n}])
    print(f"C5 done: {len(rows)} LLG runs, {len(obs_rows)} (w,F) combinations")


if __name__ == "__main__":
    main()
