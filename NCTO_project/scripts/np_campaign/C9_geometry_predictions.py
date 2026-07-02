#!/usr/bin/env python3
"""C9 — geometry-based predictions for the experimental decision tree.

Pure-Python.  Takes the three recovery mechanisms from C6 and tabulates
predicted scalings of tau_off versus:
    pump-spot 1/e^2 radius   w  in [5, 200] um
    film thickness           h  in [10, 1000] nm
    repetition rate          R  in [10, 1e5] Hz
    static symmetry breaking xi in [0, 0.1]

Outputs:
    NCTO_project/np_campaign_out/C9/{predictions.csv, decision_tree.txt,
                                     scaling.png}
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np

from common import CAMPAIGN_OUT, write_csv

PHASE = CAMPAIGN_OUT / "C9"
PHASE.mkdir(parents=True, exist_ok=True)


def tau_A(w, M=1.0, df0=0.02, sigma=0.5):
    # Free Allen–Cahn: R0 ~ w, tau ~ R0 / (M df0) for df0 dominated.
    R0 = w
    return R0 / max(M * df0, 1e-6)


def tau_B(T=0.05, k_shape=2.0, scale=0.5, Gamma0=1.0):
    # Pinned-domain ensemble: tau_off ~ (1/Gamma0) exp(<Delta>/T)
    Delta_mean = k_shape * scale
    return (1.0 / Gamma0) * math.exp(Delta_mean / max(T, 1e-4))


def tau_C(L, D=0.05):
    return L * L / (math.pi * math.pi * D)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--quick", action="store_true")
    args = p.parse_args()

    ws = np.geomspace(5e-6, 2e-4, 5 if args.quick else 11)        # m
    hs = np.geomspace(1e-8, 1e-6, 5 if args.quick else 9)         # m
    Rs = np.geomspace(10, 1e5, 4 if args.quick else 9)            # Hz
    xis = np.linspace(0.0, 0.1, 3 if args.quick else 6)

    rows = []
    for w in ws:
        rows.append({"axis": "spot_radius", "value": float(w),
                     "tau_A": tau_A(w * 1e6),   # convert to um
                     "tau_B": tau_B(),
                     "tau_C": tau_C(w * 1e6)})
    for h in hs:
        # only mech C scales with thickness if interface mode confined.
        rows.append({"axis": "thickness", "value": float(h),
                     "tau_A": tau_A(50.0),
                     "tau_B": tau_B(),
                     "tau_C": tau_C(h * 1e9)})  # in nm
    for R in Rs:
        # rep rate only affects effective T via heat accumulation.
        T_eff = 0.05 * (1.0 + R / 1e4)
        rows.append({"axis": "rep_rate", "value": float(R),
                     "tau_A": tau_A(50.0),
                     "tau_B": tau_B(T=T_eff),
                     "tau_C": tau_C(50.0)})
    for xi in xis:
        # Static symmetry breaking biases df0 -> df0 + xi.
        rows.append({"axis": "static_break", "value": float(xi),
                     "tau_A": tau_A(50.0, df0=0.02 + xi),
                     "tau_B": tau_B(),
                     "tau_C": tau_C(50.0)})
    write_csv(PHASE / "predictions.csv", rows)

    tree = """
NCTO Nature-Physics decision tree for tau_off mechanism
=======================================================

Observation 1: vary pump-spot radius w by factor of 10.
  tau_off scales linearly with w   ->  mechanism A (Allen-Cahn)
  tau_off scales as w^2            ->  mechanism C (diffusion bottleneck)
  tau_off independent of w         ->  mechanism B (pinned ensemble)

Observation 2: vary repetition rate R (fixed energy per pulse).
  tau_off accelerates with R       ->  thermal accumulation (mech B at high T)
  tau_off independent of R         ->  mechanism A or C

Observation 3: apply static in-plane strain xi.
  tau_off shortens linearly in xi  ->  mechanism A (df0 dominated)
  tau_off unchanged                ->  mechanism B or C

Combined: orthogonal scalings of (w, R, xi) uniquely identify the
recovery mechanism without requiring absolute time calibration.
"""
    (PHASE / "decision_tree.txt").write_text(tree.strip() + "\n")

    print("C9 done.")
    try:
        _plot(rows)
    except Exception as e:
        print(f"  plotting skipped: {e}")


def _plot(rows):
    import matplotlib.pyplot as plt
    axes_to_plot = ["spot_radius", "thickness", "rep_rate", "static_break"]
    fig, axarr = plt.subplots(1, 4, figsize=(15, 3.6))
    for ax, axis in zip(axarr, axes_to_plot):
        sub = sorted([r for r in rows if r["axis"] == axis],
                     key=lambda r: r["value"])
        xs = [r["value"] for r in sub]
        for key, marker in [("tau_A", "o"), ("tau_B", "s"), ("tau_C", "^")]:
            ax.loglog(xs, [r[key] for r in sub], marker + "-",
                      label=key.replace("tau_", "mech "))
        ax.set_title(axis); ax.set_xlabel(axis); ax.legend(fontsize=7)
    axarr[0].set_ylabel(r"$\tau_{\rm off}$ (a.u.)")
    fig.tight_layout()
    fig.savefig(PHASE / "scaling.png", dpi=160, bbox_inches="tight")


if __name__ == "__main__":
    main()
