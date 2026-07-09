#!/usr/bin/env python3
"""Polarization switching map r3(theta, E0), clean system, J7 modulation off vs on.

The driven state at intermediate theta settles (converged) into STABLE PARTIAL
nematic configurations (r3 ~ 0.2-0.5), not a clean single-domain zigzag, so a
binary threshold is ambiguous there.  We therefore show the full converged
order parameter r3 over (theta, E0) as a heatmap (low r3 = switched to a single
zigzag variant), with the r3=0.2 switching contour = the critical fluence
boundary E0c(theta).  Two panels: J7 modulation OFF vs ON.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DEFAULT = (ROOT / "NCTO_project" / "tuned_kruger_campaign" / "polarization_fluence_study"
           / "analysis" / "polarization_fluence.csv")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=Path, default=DEFAULT)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    rows = [dict(c=r["coupling"], t=float(r["theta_deg"]), e=float(r["e0"]), r3=float(r["r3"]))
            for r in csv.DictReader(args.csv.open())]
    thetas = sorted({r["t"] for r in rows})
    e0s = sorted({r["e"] for r in rows})

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8), sharey=True)
    couplings = [("j7off", r"J7 modulation OFF ($\lambda_{J7,0}=0$)"),
                 ("j7on", r"J7 modulation ON ($\lambda_{J7,0}=(J7/K)\lambda_{K,2}$)")]
    im = None
    for ax, (coup, title) in zip(axes, couplings):
        Z = np.full((len(e0s), len(thetas)), np.nan)
        for i, e in enumerate(e0s):
            for k, t in enumerate(thetas):
                v = next((r["r3"] for r in rows if r["c"] == coup and r["t"] == t and r["e"] == e), None)
                if v is not None:
                    Z[i, k] = v
        im = ax.pcolormesh(thetas, e0s, Z, cmap="viridis_r", vmin=0, vmax=1, shading="nearest")
        # switching boundary = critical fluence E0c(theta)
        TH, EE = np.meshgrid(thetas, e0s)
        try:
            ax.contour(TH, EE, Z, levels=[0.2], colors="white", linewidths=2.0)
        except Exception:
            pass
        ax.set_xlabel(r"pump polarization $\theta$ (deg)")
        ax.set_title(title, fontsize=10)
    axes[0].set_ylabel(r"pump fluence $E_0$")
    cb = fig.colorbar(im, ax=axes, shrink=0.85)
    cb.set_label(r"converged $r_3$  (low = switched to single zigzag)")
    fig.suptitle(r"Clean-system polarization switching; white line = critical fluence $E_0^c(\theta)$ ($r_3$=0.2)",
                 fontsize=11)
    out = args.out or (args.csv.parent / "polarization_threshold_vs_theta.png")
    fig.savefig(out, dpi=190, bbox_inches="tight")
    print(f"wrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
