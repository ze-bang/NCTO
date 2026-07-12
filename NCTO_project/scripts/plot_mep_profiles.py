#!/usr/bin/env python3
"""MEP energy profiles E(s) overlaid for the defect half-width family.

One curve per hw from the continuation width sweep (strip -> 3Q), coloured by
hw, saddle marked.  Converged decay paths show a single interior maximum with
the 3Q endpoint below the strip; hw=0.5 has no barrier (the strip is not
metastable at that width) and is drawn dashed.
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
    ap.add_argument("--out", type=Path, default=WS / "mep_profiles.png")
    args = ap.parse_args()

    items = []
    for d in sorted(glob.glob(str(WS / "hw*"))):
        rep = json.loads((Path(d) / "report.json").read_text())
        dat = np.load(Path(d) / "final_band.npz")
        items.append((rep["half_width"], dat["path_coordinate"],
                      dat["energies"] - dat["energies"][0]))

    hws = [h for h, *_ in items]
    norm = colors.Normalize(vmin=min(hws), vmax=max(hws))
    cmap = cm.viridis

    fig, ax = plt.subplots(figsize=(7.6, 5.2))
    for hw, s, e in items:
        has_barrier = e.max() > 0.5
        ax.plot(s, e, "o-" if has_barrier else "o--", ms=3.5, lw=1.8,
                color=cmap(norm(hw)),
                label=(fr"hw={hw:.1f}  ($\Delta G$={e.max():.1f} meV)" if has_barrier
                       else fr"hw={hw:.1f}  (strip unstable)"))
        if has_barrier:
            i = int(np.argmax(e))
            ax.plot(s[i], e[i], "^", ms=9, color=cmap(norm(hw)),
                    markeredgecolor="black", markeredgewidth=0.6, zorder=5)
    ax.axhline(0.0, color="0.7", lw=0.8)
    ax.set_xlabel("reaction coordinate  s  (pinned ZZ strip → 3Q)")
    ax.set_ylabel(r"$E - E_{\rm strip}$ (meV)")
    ax.set_title("Minimum-energy paths of strip depinning vs defect half-width\n"
                 "(continuation GNEB, L=36, kred s=0.5; ▲ = saddle)", fontsize=11)
    ax.legend(fontsize=9, frameon=False)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(args.out, dpi=190, bbox_inches="tight")
    print(f"wrote {args.out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
