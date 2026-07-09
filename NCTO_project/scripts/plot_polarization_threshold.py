#!/usr/bin/env python3
"""Critical pump fluence E0c vs polarization angle theta, clean system,
with and without the J7 (ring-exchange breathing) modulation.

E0c(theta) = the fluence at which the driven state switches (r3 crosses 0.2),
interpolated from the r3(E0) curve at each theta.  Two lines: J7 modulation
OFF (lambda_J7,0=0) vs ON (lambda_J7,0=(J7/K)lambda_K2).
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
R3_SWITCH = 0.2


def e0c(theta_rows) -> float:
    """Interpolated E0 where r3 first drops through R3_SWITCH (nan if never)."""
    pts = sorted(theta_rows, key=lambda r: r["e0"])
    e = [p["e0"] for p in pts]; r = [p["r3"] for p in pts]
    for i in range(1, len(e)):
        if r[i - 1] >= R3_SWITCH > r[i]:            # crossing from unswitched->switched
            f = (r[i - 1] - R3_SWITCH) / (r[i - 1] - r[i])
            return e[i - 1] + f * (e[i] - e[i - 1])
    return float("nan")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=Path, default=DEFAULT)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    rows = [dict(coupling=r["coupling"], theta=float(r["theta_deg"]),
                 e0=float(r["e0"]), r3=float(r["r3"])) for r in csv.DictReader(args.csv.open())]
    thetas = sorted({r["theta"] for r in rows})

    fig, ax = plt.subplots(figsize=(6.6, 4.8))
    styles = {"j7off": ("o-", "C0", r"J7 modulation OFF ($\lambda_{J7,0}=0$)"),
              "j7on": ("s-", "C3", r"J7 modulation ON ($\lambda_{J7,0}=(J7/K)\lambda_{K,2}$)")}
    for coup, (fmt, color, label) in styles.items():
        ec = [e0c([r for r in rows if r["coupling"] == coup and r["theta"] == t]) for t in thetas]
        ax.plot(thetas, ec, fmt, color=color, ms=5, label=label)
    ax.set_xlabel(r"pump polarization $\theta$ (deg)")
    ax.set_ylabel(r"critical fluence $E_0^{\,c}(\theta)$")
    ax.set_title("Clean-system switching threshold vs polarization")
    ax.legend(fontsize=9, frameon=False); ax.grid(alpha=0.25)
    fig.tight_layout()
    out = args.out or (args.csv.parent / "polarization_threshold_vs_theta.png")
    fig.savefig(out, dpi=190, bbox_inches="tight")
    print(f"wrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
