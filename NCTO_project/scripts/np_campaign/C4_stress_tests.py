#!/usr/bin/env python3
"""C4 — stress tests under realistic imperfections.

Perturbations swept (one at a time, baseline = NCTO):
    T_xi       : Langevin temperature in [0, 0.05, 0.1, 0.2]
    eta_xi     : pump ellipticity ratio  Ey/Ex in [0, 0.1, 0.25, 0.5]
    detune_xi  : Delta omega / omega_E1 in [-0.10, -0.05, 0, 0.05, 0.10]
    strain_xi  : extra static dK in [0, 0.05, 0.10] (in K units)
    disorder_xi: random per-site dK rms in [0, 0.05, 0.10]

For each perturbation strength and each polarisation theta on a coarse
grid, we run a single pump-probe trajectory at the baseline pump
amplitude and record r3, S(M_alpha), and the inferred dominant domain.

Output: NCTO_project/np_campaign_out/C4/stress.csv plus per-axis PNGs.
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

from common import (CAMPAIGN_OUT, REPO_ROOT, declare_switched, ensure_3Q_seed,
                    label, render_param, run_analysis, run_spin_solver,
                    write_csv)

PHASE = CAMPAIGN_OUT / "C4"
PHASE.mkdir(parents=True, exist_ok=True)


def grid(quick: bool):
    thetas = [i * math.pi / 6 for i in range(6)]  # 0..150 deg by 30
    if quick:
        sweeps = {
            "T":        [0.0, 0.05, 0.10],
            "ellip":    [0.0, 0.25],
            "detune":   [-0.05, 0.0, 0.05],
            "strain":   [0.0, 0.10],
            "disorder": [0.0, 0.05],
        }
    else:
        sweeps = {
            "T":        [0.0, 0.025, 0.05, 0.10, 0.20],
            "ellip":    [0.0, 0.1, 0.25, 0.5],
            "detune":   [-0.10, -0.05, 0.0, 0.05, 0.10],
            "strain":   [0.0, 0.025, 0.05, 0.10],
            "disorder": [0.0, 0.025, 0.05, 0.10],
        }
    return thetas, sweeps


def apply_perturbation(axis: str, xi: float, base: dict) -> dict:
    over = dict(base)
    if axis == "T":
        over["T"] = xi
    elif axis == "ellip":
        # Approximate ellipticity by reducing the effective pump amplitude
        # along the orthogonal polarisation direction.  The microscopic
        # solver represents ellipticity through pump_phase + amplitude
        # asymmetry, but the base template only exposes a single E0; we
        # therefore encode ellipticity as an angle-dependent E0 scaling.
        # NOTE: full ellipticity support would require driver-side
        # extension; we mark this axis as approximated here.
        # E0_eff = E0 * sqrt(1 + xi^2) / sqrt(2) for symmetric round trip.
        over["E0"] = float(over.get("E0", 2.0)) * math.sqrt(1.0 + xi * xi) / math.sqrt(2.0)
    elif axis == "detune":
        over["PUMP_FREQ"] = float(over.get("PUMP_FREQ", 4.0)) * (1.0 + xi)
    elif axis == "strain":
        # Static dK enters as additional LAMBDA_K_0 channel offset.
        over["LAMBDA_K_0"] = float(over.get("LAMBDA_K_0", 0.0)) + xi
    elif axis == "disorder":
        # Bond disorder cannot be turned on via the base template;
        # we approximate by inflating gamma_E1 (more diffusive driver),
        # which is a proxy for incoherent broadening.
        over["GAMMA_E1"] = float(over.get("GAMMA_E1", 0.2543)) + xi
    return over


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--quick", action="store_true")
    args = p.parse_args()

    thetas, sweeps = grid(args.quick)
    seed = ensure_3Q_seed(tag="C4_seed")
    base = {
        "SEED_FILE": seed.relative_to(REPO_ROOT),
        "E0": 2.0, "LAMBDA_K_2": 0.04,
    }

    rows = []
    for axis, xis in sweeps.items():
        for xi in xis:
            for th in thetas:
                tag = label("c4", axis=axis,
                            xi=round(xi, 4), th=round(th, 3))
                outdir = PHASE / "runs" / tag
                cfg = PHASE / "configs" / f"{tag}.param"
                overrides = apply_perturbation(axis, xi, base)
                overrides.update({"OUTDIR": outdir.relative_to(REPO_ROOT),
                                  "THETA": th})
                render_param(cfg, overrides)
                res = run_spin_solver(cfg)
                if not res.ok or res.spins_final is None:
                    continue
                parsed = run_analysis(cfg, [res.spins_final])
                if not parsed:
                    continue
                pr = parsed[0]
                rows.append({
                    "axis": axis, "xi": xi, "theta": th,
                    "S_M1": pr["S_M1"], "S_M2": pr["S_M2"], "S_M3": pr["S_M3"],
                    "r3": pr["m_min_over_max"],
                    "dom": pr.get("dominant_domain", ""),
                    "switched": int(declare_switched(pr["m_min_over_max"])),
                })

    write_csv(PHASE / "stress.csv", rows)
    print(f"C4 done: {len(rows)} runs")
    try:
        _plot(rows)
    except Exception as e:
        print(f"  plotting skipped: {e}")


def _plot(rows):
    import matplotlib.pyplot as plt
    from collections import defaultdict
    if not rows:
        return
    axes_to_plot = sorted(set(r["axis"] for r in rows))
    fig, axarr = plt.subplots(1, len(axes_to_plot),
                              figsize=(3.6 * len(axes_to_plot), 3.4),
                              sharey=True)
    if len(axes_to_plot) == 1:
        axarr = [axarr]
    for ax, axis in zip(axarr, axes_to_plot):
        sub = [r for r in rows if r["axis"] == axis]
        xis = sorted(set(r["xi"] for r in sub))
        for xi in xis:
            grp = sorted([r for r in sub if r["xi"] == xi],
                         key=lambda r: r["theta"])
            ax.plot([r["theta"] for r in grp],
                    [r["r3"] for r in grp],
                    "-o", label=f"{axis}={xi}")
        ax.set_xlabel(r"$\theta$ [rad]")
        ax.set_title(axis)
        ax.legend(fontsize=7)
    axarr[0].set_ylabel(r"$r_3 = m_{\min}/m_{\max}$")
    fig.savefig(PHASE / "stress.png", dpi=160, bbox_inches="tight")


if __name__ == "__main__":
    main()
