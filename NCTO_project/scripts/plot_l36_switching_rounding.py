#!/usr/bin/env python3
"""Plot the 36x36 enhanced-|K| centred-disorder switching fraction."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
ANALYSIS = (ROOT / "NCTO_project" / "tuned_kruger_campaign"
            / "pinning_switching_crosscheck_L36" / "analysis")
DEFAULT_SUMMARY = ANALYSIS / "L36_kred_s0p500_hw2p00_highE_drive_crosscheck_summary.json"


def crossing(e0: np.ndarray, y: np.ndarray, level: float) -> float:
    for idx in range(len(e0) - 1):
        left = y[idx] - level
        right = y[idx + 1] - level
        if left == 0:
            return float(e0[idx])
        if left * right <= 0 and y[idx + 1] != y[idx]:
            t = (level - y[idx]) / (y[idx + 1] - y[idx])
            return float(e0[idx] + t * (e0[idx + 1] - e0[idx]))
    return float("nan")


def transition_width(e0: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    low = float(np.nanmin(y))
    high = float(np.nanmax(y))
    if high - low < 0.2:
        return float("nan"), float("nan"), float("nan")
    normalized = (y - low) / (high - low)
    e25 = crossing(e0, normalized, 0.25)
    e75 = crossing(e0, normalized, 0.75)
    return e75 - e25, e25, e75


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--out-dir", type=Path, default=ANALYSIS / "l36_switching_rounding")
    parser.add_argument("--title", default=r"$36\times36$ enhanced-$|K|$ disorder switching")
    parser.add_argument("--protocol-label",
                        default=r"line bonds: $\delta K\sim\mathcal{N}(-0.5|K|,\sigma_K)$, $hw=2$")
    args = parser.parse_args()

    summary = args.summary if args.summary.is_absolute() else ROOT / args.summary
    out_dir = args.out_dir if args.out_dir.is_absolute() else ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = json.loads(summary.read_text())
    grouped: dict[float, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[float(row["sigma_k"])].append(row)

    fig, ax = plt.subplots(figsize=(6.4, 4.5))
    cmap = matplotlib.colormaps["viridis"]
    widths = []
    for idx, sigma in enumerate(sorted(grouped)):
        sigma_rows = sorted(grouped[sigma], key=lambda item: float(item["e0"]))
        e0 = np.array([float(item["e0"]) for item in sigma_rows])
        drive = np.array([float(item["drive_write_fraction"]) for item in sigma_rows])
        excess = np.array([float(item["excess_quench_zz"]) for item in sigma_rows])
        n = np.array([max(1, int(item.get("n", 1))) for item in sigma_rows])
        yerr = np.sqrt(np.clip(drive * (1.0 - drive), 0.0, 1.0) / n)
        color = "k" if sigma == 0.0 else cmap(idx / max(1, len(grouped) - 1))
        label = "uniform line" if sigma == 0.0 else rf"$\sigma_K={sigma:.2g}$ meV"
        ax.errorbar(e0, drive, yerr=yerr, fmt="o-", color=color, lw=1.8, ms=4.8,
                capsize=2.4, elinewidth=0.9, label=label)
        ax.plot(e0, excess, "--", color=color, lw=1.0, alpha=0.45)
        width, e25, e75 = transition_width(e0, drive)
        widths.append({"sigma_k": sigma, "width": width, "e25": e25, "e75": e75,
                   "n_min": int(np.min(n)), "n_max": int(np.max(n))})

    nonzero_n = [row["n_min"] for row in widths if row["sigma_k"] > 0]
    seed_text = (f"nonzero $\\sigma_K$: n={min(nonzero_n)}-{max(nonzero_n)}"
                 if nonzero_n and min(nonzero_n) != max(nonzero_n)
                 else (f"nonzero $\\sigma_K$: n={nonzero_n[0]}" if nonzero_n else "single disorder-free run"))

    ax.axhline(0.5, color="0.55", ls=":", lw=1.0)
    ax.set_title(args.title)
    ax.set_xlabel(r"pump fluence $E_0$")
    ax.set_ylabel("baseline-subtracted switched fraction")
    ax.set_ylim(-0.05, 1.05)
    ax.grid(alpha=0.25, lw=0.5)
    ax.legend(fontsize=7.3, frameon=True, framealpha=0.92, loc="lower right")
    ax.text(0.03, 0.06,
            r"$L=36$, $N=2592$" + "\n"
            + args.protocol_label + "\n"
            + seed_text + "\n"
            + "points: switching fraction; bars: binomial SEM" + "\n"
            + "dashed: excess ZZ",
            transform=ax.transAxes, va="bottom", ha="left", fontsize=7.8,
            bbox=dict(facecolor="white", edgecolor="0.85", alpha=0.92, pad=3.0))
    fig.tight_layout()

    png = out_dir / "l36_switching_rounding.png"
    pdf = out_dir / "l36_switching_rounding.pdf"
    report = out_dir / "l36_switching_rounding_report.json"
    fig.savefig(png, dpi=220)
    fig.savefig(pdf)
    report.write_text(json.dumps({"summary": str(summary.relative_to(ROOT)),
                                  "transition_widths": widths}, indent=2))
    print(f"wrote {png.relative_to(ROOT)}")
    print(f"wrote {pdf.relative_to(ROOT)}")
    print(f"wrote {report.relative_to(ROOT)}")
    for row in widths:
        print(f"sigma_K={row['sigma_k']:.2f} width={row['width']:.3f} "
              f"E25={row['e25']:.3f} E75={row['e75']:.3f}")


if __name__ == "__main__":
    main()