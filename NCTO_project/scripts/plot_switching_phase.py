#!/usr/bin/env python3
"""Optical switching phase diagram as a critical-fluence map E0c(J7, lambda).

The full (J7, lambda, E0) cube is mostly 'no switch'; the informative object is
the threshold fluence E0c at which the clean all-channel drive flips 3Q -> ZZ.
This collapses the E0 axis into the measured threshold and greys out the region
that never switches within the scanned fluence range.
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
DEFAULT = (ROOT / "NCTO_project" / "tuned_kruger_campaign" / "phase_diagram_allchan"
           / "analysis" / "phase_summary_full.csv")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=Path, default=DEFAULT)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    rows = [dict(j7=float(r["J7"]), lam=float(r["lambda_K2"]),
                 e0=float(r["E0"]), sw=int(r["switched"])) for r in csv.DictReader(args.csv.open())]
    j7s = sorted({r["j7"] for r in rows})
    lams = sorted({r["lam"] for r in rows})
    e0max = max(r["e0"] for r in rows)

    # critical fluence = lowest E0 that switches, else NaN (never switches)
    Ec = np.full((len(lams), len(j7s)), np.nan)
    for i, lam in enumerate(lams):
        for k, j7 in enumerate(j7s):
            sw = [r["e0"] for r in rows if r["j7"] == j7 and r["lam"] == lam and r["sw"]]
            if sw:
                Ec[i, k] = min(sw)

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    cmap = plt.cm.viridis.copy(); cmap.set_bad("0.88")
    im = ax.pcolormesh(j7s, lams, Ec, cmap=cmap, shading="nearest",
                       vmin=np.nanmin(Ec), vmax=e0max)
    cb = fig.colorbar(im, ax=ax)
    cb.set_label(r"critical fluence $E_0^{\,c}$")
    # annotate the no-switch region
    ax.text(0.5, 0.5, "no switching\n(up to $E_0$=%g)" % e0max, transform=ax.transAxes,
            ha="center", va="center", color="0.35", fontsize=11)
    ax.set_xlabel(r"ring exchange $J_7$")
    ax.set_ylabel(r"E1 coupling scale  $\lambda$")
    ax.set_title("Optical switching phase diagram (clean, all-channel drive)")
    fig.tight_layout()
    out = args.out or (args.csv.parent / "switching_threshold_map.png")
    fig.savefig(out, dpi=190, bbox_inches="tight")
    print(f"wrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
