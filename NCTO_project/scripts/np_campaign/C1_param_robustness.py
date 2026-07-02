#!/usr/bin/env python3
"""C1 — microscopic robustness across Hamiltonian and coupling channels.

Sweeps a coarse 4-D grid (J,K,Gamma,Gammap) around the NCTO baseline
for each of the four E2 magneto-elastic channels in turn, scanning a
pump-amplitude ladder at theta=0 and a polarisation grid at the best
amplitude.  Records r3 = m_min/m_max and switching success.

Outputs:
    NCTO_project/np_campaign_out/C1/results.csv
    NCTO_project/np_campaign_out/C1/threshold_map.png

Run modes:
    --quick   : 1-point Hamiltonian, 3 channels, 3 amplitudes, 4 angles.
    --full    : 4x4x4x4 Hamiltonian, 4 channels, 6 amplitudes, 12 angles.
"""
from __future__ import annotations

import argparse
import itertools
from pathlib import Path

from common import (CAMPAIGN_OUT, REPO_ROOT, declare_switched, ensure_3Q_seed,
                    label, render_param, run_analysis, run_spin_solver,
                    write_csv)

PHASE = CAMPAIGN_OUT / "C1"
PHASE.mkdir(parents=True, exist_ok=True)

CHANNELS = ["LAMBDA_K_2", "LAMBDA_GAMMA_2", "LAMBDA_J_2", "LAMBDA_GAMMAP_2"]


def grid(quick: bool):
    if quick:
        J_grid = [-1.0]
        K_grid = [-6.0]
        G_grid = [8.0]
        Gp_grid = [-3.5]
        chans = CHANNELS[:3]
        # Bracket calibrated E*≈1.40 (lambda_K=0.035, J7=-0.0026, gamma_E1=0.0849).
        amps = [1.2, 1.5, 2.0]
        thetas = [0.0, 0.262, 0.524, 1.047]  # 0, 15, 30, 60 deg
    else:
        J_grid = [-1.2, -1.0, -0.8]
        K_grid = [-7.0, -6.0, -5.0]
        G_grid = [7.0, 8.0, 9.0]
        Gp_grid = [-4.0, -3.5, -3.0]
        chans = CHANNELS
        # Calibrated E* range 1.40–2.0 across (lambda_K, J7) grid.
        amps = [1.0, 1.3, 1.5, 1.7, 2.0, 2.5]
        thetas = [i * 0.2618 for i in range(12)]  # 0..165 deg
    return J_grid, K_grid, G_grid, Gp_grid, chans, amps, thetas


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--quick", action="store_true")
    args = p.parse_args()

    J_grid, K_grid, G_grid, Gp_grid, chans, amps, thetas = grid(args.quick)
    rows = []
    cfg_dir = PHASE / "configs"

    for J, K, G, Gp in itertools.product(J_grid, K_grid, G_grid, Gp_grid):
        seed = ensure_3Q_seed(tag=label("seed", J=J, K=K, G=G, Gp=Gp),
                              overrides={"J": J, "K": K, "GAMMA": G, "GAMMAP": Gp})
        for chan in chans:
            for E0 in amps:
                for th in thetas:
                    tag = label("c1", J=J, K=K, G=G, Gp=Gp,
                                ch=CHANNELS.index(chan), E=E0, th=round(th, 3))
                    outdir = PHASE / "runs" / tag
                    cfg = cfg_dir / f"{tag}.param"
                    overrides = {
                        "J": J, "K": K, "GAMMA": G, "GAMMAP": Gp,
                        "E0": E0, "THETA": th,
                        "SEED_FILE": seed.relative_to(REPO_ROOT),
                        "OUTDIR": outdir.relative_to(REPO_ROOT),
                        # zero all channels then turn on the active one
                        "LAMBDA_K_2": 0.0, "LAMBDA_GAMMA_2": 0.0,
                        "LAMBDA_J_2": 0.0, "LAMBDA_GAMMAP_2": 0.0,
                        chan: 0.04,
                    }
                    render_param(cfg, overrides)
                    res = run_spin_solver(cfg)
                    if not res.ok or res.spins_final is None:
                        continue
                    parsed = run_analysis(cfg, [res.spins_final])
                    if not parsed:
                        continue
                    pr = parsed[0]
                    rows.append({
                        "J": J, "K": K, "Gamma": G, "Gammap": Gp,
                        "channel": chan, "E0": E0, "theta": th,
                        "S_M1": pr["S_M1"], "S_M2": pr["S_M2"], "S_M3": pr["S_M3"],
                        "r3": pr["m_min_over_max"],
                        "switched": int(declare_switched(pr["m_min_over_max"])),
                    })

    csv_out = PHASE / "results.csv"
    write_csv(csv_out, rows)
    print(f"C1 done: {len(rows)} runs -> {csv_out}")

    try:
        _plot(csv_out)
    except Exception as e:
        print(f"  plotting skipped: {e}")


def _plot(csv_path: Path):
    import matplotlib.pyplot as plt
    import numpy as np
    from analysis_utils import read_csv
    rows = read_csv(csv_path)
    if not rows:
        return
    fig, axes = plt.subplots(1, len(CHANNELS), figsize=(4 * len(CHANNELS), 3.6),
                             sharey=True)
    if len(CHANNELS) == 1:
        axes = [axes]
    for ax, chan in zip(axes, CHANNELS):
        sub = [r for r in rows if r["channel"] == chan]
        if not sub:
            ax.set_title(f"{chan} (no data)")
            continue
        ths = sorted(set(r["theta"] for r in sub))
        E0s = sorted(set(r["E0"] for r in sub))
        Z = np.full((len(E0s), len(ths)), np.nan)
        for r in sub:
            i = E0s.index(r["E0"]); j = ths.index(r["theta"])
            # accumulate switching rate over (J,K,G,Gp)
            if np.isnan(Z[i, j]):
                Z[i, j] = 0.0
            Z[i, j] += float(r["switched"])
        Z = Z / np.nanmax(Z + 1e-12)
        im = ax.imshow(Z, aspect="auto", origin="lower",
                       extent=[min(ths), max(ths), min(E0s), max(E0s)],
                       cmap="magma", vmin=0, vmax=1)
        ax.set_xlabel(r"$\theta_{\rm pump}$ [rad]")
        ax.set_title(chan)
    axes[0].set_ylabel(r"$E_0$")
    fig.colorbar(im, ax=axes, shrink=0.8, label="switching rate")
    fig.savefig(csv_path.with_suffix("").parent / "threshold_map.png",
                dpi=160, bbox_inches="tight")


if __name__ == "__main__":
    main()
