#!/usr/bin/env python3
"""Switching-fraction vs fluence, one line per background-disorder strength sigma_K.

Reads a switching cross-check summary JSON (per (E0, sigma_K) cell) and plots the
disorder-rounded switching curves: the driven switched fraction and the mean
quenched ZZ order, each as a family of lines coloured by sigma_K.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm, colors

ROOT = Path(__file__).resolve().parents[2]
DEFAULT = (ROOT / "NCTO_project" / "tuned_kruger_campaign"
           / "pinning_switching_crosscheck_L36" / "analysis"
           / "L36_kred_s0p500_hw1p80_kred_0p8ms_hw1p8_drive_crosscheck_summary.json")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", type=Path, default=DEFAULT)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    rows = json.loads(args.summary.read_text())
    sigmas = sorted({r["sigma_k"] for r in rows})
    norm = colors.Normalize(vmin=min(sigmas), vmax=max(sigmas))
    cmap = cm.viridis

    panels = [("drive_write_fraction", "switched fraction", (-0.03, 1.03)),
              ("mean_quench_zz", r"mean quenched ZZ fraction $\langle m_{ZZ}\rangle$", None)]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6))

    for ax, (key, ylabel, ylim) in zip(axes, panels):
        for sig in sigmas:
            pts = sorted([r for r in rows if r["sigma_k"] == sig], key=lambda r: r["e0"])
            ax.plot([r["e0"] for r in pts], [r[key] for r in pts],
                    "o-", ms=4, lw=1.6, color=cmap(norm(sig)),
                    label=fr"$\sigma_K={sig:.1f}$")
        ax.axvline(16, color="0.6", ls=":", lw=1.0)              # clean-line threshold
        ax.set_xlabel(r"pump fluence $E_0$")
        ax.set_ylabel(ylabel)
        if ylim:
            ax.set_ylim(*ylim)
        ax.grid(alpha=0.25)
    axes[0].legend(title="background disorder", frameon=False, fontsize=9)
    fig.suptitle(f"L36 kred line (hw=1.8) switching vs fluence — disorder rounding\n"
                 f"{args.summary.name}", fontsize=10)
    fig.tight_layout()

    out = args.out or (args.summary.parent / "switching_fraction_vs_fluence.png")
    fig.savefig(out, dpi=190, bbox_inches="tight")
    print(f"wrote {out.relative_to(ROOT) if out.is_relative_to(ROOT) else out}")


if __name__ == "__main__":
    main()
