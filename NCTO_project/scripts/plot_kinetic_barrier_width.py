#!/usr/bin/env python3
"""Kinetic depinning barrier vs defect width, and the hw that pins tau=0.8 ms @ 10 K.

Reads the GNEB finite-size-scan reports (L=36, strength 0.5) and plots:
  (left)  barrier dG(hw)  [deliverable 3]
  (right) Arrhenius lifetime tau(hw)=tau0*exp(dG/kB T) at T=10 K, with the
          0.8 ms target marked -> the pinned hw*  [deliverable 4]
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
FS = ROOT / "NCTO_project" / "tuned_kruger_campaign" / "critical_radius" / "finite_size_scan"

KB = 0.0861733            # meV/K
J = 0.68                  # meV  (tau0 ~ hbar/J)
HBAR = 6.582119e-13       # meV*s
TAU0 = HBAR / J           # ~0.97 ps
T = 10.0
TARGET_TAU = 0.8e-3       # s


def collect(lattice: int, strength: float) -> tuple[np.ndarray, np.ndarray]:
    hw, bar = [], []
    for d in sorted(FS.glob(f"L{lattice}_s{strength:.3f}_hw*".replace(".", "p"))):
        rep = d / "report.json"
        if rep.exists():
            r = json.loads(rep.read_text())
            hw.append(r["half_width"]); bar.append(r["barrier_meV"])
    o = np.argsort(hw)
    return np.array(hw)[o], np.array(bar)[o]


def lifetime(dG_meV: np.ndarray) -> np.ndarray:
    return TAU0 * np.exp(dG_meV / (KB * T))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lattice", type=int, default=36)
    ap.add_argument("--strength", type=float, default=0.5)
    ap.add_argument("--out", type=Path, default=FS / "kinetic_barrier_vs_width.png")
    args = ap.parse_args()

    hw, bar = collect(args.lattice, args.strength)
    tau = lifetime(bar)
    kT = KB * T
    dG_target = kT * math.log(TARGET_TAU / TAU0)
    hw_star = float(np.interp(dG_target, bar, hw)) if bar.min() < dG_target < bar.max() else float("nan")

    fig, (a, b) = plt.subplots(1, 2, figsize=(11.5, 4.6))

    a.plot(hw, bar, "o-", color="C0", ms=6)
    a.axhline(dG_target, color="C3", ls="--", lw=1.2,
              label=fr"$\Delta G^*$={dG_target:.1f} meV (0.8 ms, 10 K)")
    if np.isfinite(hw_star):
        a.axvline(hw_star, color="0.5", ls=":", lw=1.0)
    a.set_xlabel("defect half-width  hw (cells)")
    a.set_ylabel(r"depinning barrier $\Delta G$ (meV)")
    a.set_title(f"(3) GNEB kinetic barrier vs width, L={args.lattice}")
    a.legend(fontsize=9, frameon=False); a.grid(alpha=0.25)

    b.semilogy(hw, tau * 1e3, "o-", color="C0", ms=6)
    b.axhline(0.8, color="C3", ls="--", lw=1.2, label="0.8 ms target")
    b.axhline(0.3, color="C1", ls=":", lw=1.0, label="0.3 ms")
    if np.isfinite(hw_star):
        b.axvline(hw_star, color="0.5", ls=":", lw=1.0)
        b.annotate(fr"hw$^*\approx${hw_star:.2f}", xy=(hw_star, 0.8),
                   xytext=(hw_star + 0.05, 3), fontsize=10, color="C3")
    b.set_xlabel("defect half-width  hw (cells)")
    b.set_ylabel(r"Arrhenius lifetime $\tau$ (ms) @ 10 K")
    b.set_title(r"(4) $\tau=\tau_0 e^{\Delta G/k_BT}$, $\tau_0=\hbar/J$")
    b.legend(fontsize=9, frameon=False); b.grid(alpha=0.25, which="both")

    fig.suptitle(f"Depinning barrier & lifetime vs defect width (kred, s={args.strength}); "
                 fr"$\tau_0=\hbar/J$={TAU0*1e12:.2f} ps", fontsize=10)
    fig.tight_layout()
    fig.savefig(args.out, dpi=190, bbox_inches="tight")
    print(f"hw*(0.8 ms,10K, dG*={dG_target:.1f} meV) = {hw_star:.3f}")
    print(f"wrote {args.out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
