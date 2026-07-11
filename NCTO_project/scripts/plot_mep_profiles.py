#!/usr/bin/env python3
"""MEP energy profiles E(s) per defect half-width -- convergence diagnostic.

One panel per hw from the continuation width sweep's stored bands.  A converged,
physically-relevant path shows: strip minimum at s=0, single interior saddle,
3Q endpoint BELOW the strip (metastable strip decaying into the 3Q ground
state).  Quarter-widths instead make the pinned strip the GROUND state (the
width cutoff includes an unbalanced row of K-enhanced bonds), so their paths
run uphill -- excluded from the lifetime calibration, flagged in red.
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm, colors

ROOT = Path(__file__).resolve().parents[2]
WS = ROOT / "NCTO_project" / "tuned_kruger_campaign" / "kinetic_barrier" / "width_sweep"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=WS / "mep_profiles_per_hw.png")
    args = ap.parse_args()

    items = []
    for d in sorted(glob.glob(str(WS / "hw*"))):
        rep = json.loads((Path(d) / "report.json").read_text())
        dat = np.load(Path(d) / "final_band.npz")
        items.append((rep["half_width"], dat["path_coordinate"],
                      dat["energies"] - dat["energies"][0]))

    norm = colors.Normalize(vmin=min(h for h, *_ in items), vmax=max(h for h, *_ in items))
    cmap = cm.viridis
    ncol = 4
    nrow = (len(items) + ncol - 1) // ncol
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.8 * ncol, 3.0 * nrow), sharex=True)
    axes = np.atleast_2d(axes)
    for ax, (hw, s, e) in zip(axes.ravel(), items):
        metastable = e[-1] < 0.0          # 3Q below strip -> strip decays
        ax.plot(s, e, "o-", ms=3.5, color=cmap(norm(hw)))
        imax = int(np.argmax(e))
        ax.plot(s[imax], e[imax], "r^", ms=8)
        ax.axhline(0, color="0.7", lw=0.7)
        tag = f"ΔG={e.max():.1f} meV" if metastable else "strip = ground state"
        ax.set_title(f"hw={hw:.2f}   {tag}", fontsize=10,
                     color="black" if metastable else "C3")
        ax.grid(alpha=0.25)
    for ax in axes[-1]:
        ax.set_xlabel("path coordinate s (strip → 3Q)")
    for row in axes:
        row[0].set_ylabel("E − E(strip) (meV)")
    for ax in axes.ravel()[len(items):]:
        ax.axis("off")
    fig.suptitle("MEP energy profiles per half-width (continuation GNEB; red ▲ = saddle)",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(args.out, dpi=160, bbox_inches="tight")
    print(f"wrote {args.out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
