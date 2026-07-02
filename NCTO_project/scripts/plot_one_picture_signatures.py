#!/usr/bin/env python3
"""One-picture summary of the NCTO defect-pinning/switching mechanism."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
MEP = (ROOT / "NCTO_project" / "tuned_kruger_campaign" / "defect_catalogue_L36"
    / "selected_kred_mep")
SWITCH = (ROOT / "NCTO_project" / "tuned_kruger_campaign"
       / "pinning_switching_crosscheck_L36" / "analysis")
OUT = (ROOT / "NCTO_project" / "tuned_kruger_campaign"
       / "one_picture_signatures")
OUT.mkdir(parents=True, exist_ok=True)
DEFAULT_SWITCH_SUMMARY = SWITCH / "L36_kred_s0p500_hw2p00_highE_drive_crosscheck_summary.json"

import sys
sys.path.insert(0, str(ROOT / "util" / "readers_new"))
from reader_strain_lattice import generate_honeycomb_positions  # noqa: E402

KABS = 7.89
KB = 0.0861733
T_LIFE = 10.0
TAU0_PS = 0.6582119569 / KABS
TARGET_MS = 0.3
EA_TARGET = KB * T_LIFE * np.log(TARGET_MS / (TAU0_PS * 1e-9))
LATTICE = 36
N = 2 * LATTICE * LATTICE
NN_CUTOFF = 0.65


def infer_nn_bonds(xy: np.ndarray) -> list[tuple[int, int]]:
    bonds: list[tuple[int, int]] = []
    for i in range(len(xy)):
        delta = xy[i + 1:] - xy[i]
        dist = np.sqrt(np.sum(delta * delta, axis=1))
        for off in np.where(dist < NN_CUTOFF)[0]:
            bonds.append((i, i + 1 + int(off)))
    return bonds


def load_selected_mep() -> dict:
    return json.loads((MEP / "selected_kred_mep_L36_report.json").read_text())


def load_switching(path: Path) -> list[dict]:
    return json.loads(path.read_text())


def geometry_axes(ax) -> None:
    xy = generate_honeycomb_positions(N)[:, :2]
    bonds = infer_nn_bonds(xy)
    bmid = np.array([0.5 * (xy[i] + xy[j]) for i, j in bonds])
    x_center = 0.5 * (float(xy[:, 0].min()) + float(xy[:, 0].max()))
    half_width = 2.0
    selected = {k for k in range(len(bonds)) if abs(bmid[k, 0] - x_center) <= half_width}
    for k, (i, j) in enumerate(bonds):
        xi, yi = xy[i]
        xj, yj = xy[j]
        color = "#2171b5" if k in selected else "0.83"
        lw = 1.65 if k in selected else 0.45
        alpha = 0.95 if k in selected else 0.5
        ax.plot([xi, xj], [yi, yj], color=color, lw=lw, alpha=alpha, solid_capstyle="round")
    ax.axvspan(x_center - half_width, x_center + half_width, color="#2171b5", alpha=0.08)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("(a) hard lifetime defect")
    ax.text(0.02, 0.98,
            r"enhanced-$K$ line: $\delta K=-0.5|K|$" + "\n"
            + r"$36\times36$: 459 NN bonds, $hw=2$",
            transform=ax.transAxes, ha="left", va="top", fontsize=8.3)
    for spine in ax.spines.values():
        spine.set_visible(False)


def mep_axes(ax, report: dict) -> None:
    gneb = report["gneb"]
    x = np.array(gneb["path_coordinate"])
    e = np.array(gneb["relative_energy_meV"])
    imax = int(gneb["saddle_image"])
    ax.plot(x, e, "o-", color="#2b8cbe", lw=1.8, ms=4.0)
    ax.plot(x[imax], e[imax], "*", color="k", ms=12)
    ax.axhline(0.0, color="0.65", lw=0.8)
    ax.axhline(e[-1], color="#238b45", ls="--", lw=1.0)
    ax.annotate(r"$U_{\rm pin}=21.22$ meV", xy=(x[imax], e[imax]),
                xytext=(0.34, 0.82), textcoords="axes fraction",
                arrowprops=dict(arrowstyle="->", lw=0.8), fontsize=8.6)
    ax.annotate(r"relaxed 3Q lower by 24.15 meV", xy=(x[-1], e[-1]),
                xytext=(0.35, 0.12), textcoords="axes fraction",
                arrowprops=dict(arrowstyle="->", lw=0.8), fontsize=8.2)
    ax.set_title(r"(b) $36\times36$ ZZ $\rightarrow$ 3Q MEP")
    ax.set_xlabel("normalized GNEB arc length")
    ax.set_ylabel(r"$E-E_{\rm pinned\ ZZ}$ (meV)")
    ax.grid(alpha=0.25, lw=0.5)


def switching_axes(ax, rows: list[dict], max_sigma: float | None, protocol_label: str) -> list[dict]:
    by_sigma: dict[float, list[dict]] = defaultdict(list)
    for row in rows:
        sigma = float(row["sigma_k"])
        if max_sigma is None or sigma <= max_sigma:
            by_sigma[sigma].append(row)
    cmap = matplotlib.colormaps["viridis"]
    widths = []
    for idx, sigma in enumerate(sorted(by_sigma)):
        rr = sorted(by_sigma[sigma], key=lambda r: float(r["e0"]))
        x = np.array([float(r["e0"]) for r in rr])
        y = np.array([float(r["drive_write_fraction"]) for r in rr])
        dzz = np.array([float(r["excess_quench_zz"]) for r in rr])
        n = np.array([max(1, int(r.get("n", 1))) for r in rr])
        yerr = np.sqrt(np.clip(y * (1.0 - y), 0.0, 1.0) / n)
        color = "k" if sigma == 0.0 else cmap(idx / max(1, len(by_sigma) - 1))
        label = "clean" if sigma == 0.0 else rf"$\sigma_K={sigma:.2g}$ meV"
        ax.errorbar(x, y, yerr=yerr, fmt="o-", color=color, lw=1.7, ms=4.0,
                capsize=2.2, elinewidth=0.85, label=label)
        ax.plot(x, dzz, "--", color=color, lw=1.0, alpha=0.45)
        lo, hi = float(np.nanmin(y)), float(np.nanmax(y))
        width = float("nan")
        if hi - lo > 0.2:
            yn = (y - lo) / (hi - lo)
            e25 = crossing(x, yn, 0.25)
            e75 = crossing(x, yn, 0.75)
            width = e75 - e25
        widths.append({"sigma_k": sigma, "width": width})
    nonzero_n = [int(r.get("n", 1)) for rr in by_sigma.values() for r in rr if float(r["sigma_k"]) > 0]
    seed_text = (f"nonzero $\\sigma_K$: n={min(nonzero_n)}-{max(nonzero_n)}"
                 if nonzero_n and min(nonzero_n) != max(nonzero_n)
                 else (f"nonzero $\\sigma_K$: n={nonzero_n[0]}" if nonzero_n else "single disorder-free run"))
    ax.axhline(0.5, color="0.65", ls=":", lw=1.0)
    ax.set_title("(c) disorder-rounded melting")
    ax.set_xlabel(r"pump fluence $E_0$")
    ax.set_ylabel("baseline-subtracted switched fraction")
    ax.set_ylim(-0.05, 1.05)
    ax.grid(alpha=0.25, lw=0.5)
    ax.legend(fontsize=7.2, frameon=False, loc="lower right")
    ax.text(0.03, 0.97,
            protocol_label + "\n"
            + seed_text + "; error bars: binomial SEM",
            transform=ax.transAxes, ha="left", va="top", fontsize=8.2,
            bbox=dict(facecolor="white", edgecolor="0.88", alpha=0.9, pad=2.0))
    return widths


def crossing(x: np.ndarray, y: np.ndarray, level: float) -> float:
    for i in range(len(x) - 1):
        a = y[i] - level
        b = y[i + 1] - level
        if a == 0:
            return float(x[i])
        if a * b <= 0 and y[i + 1] != y[i]:
            t = (level - y[i]) / (y[i + 1] - y[i])
            return float(x[i] + t * (x[i + 1] - x[i]))
    return float("nan")


def lifetime_axes(ax, barrier: float) -> None:
    energies = np.linspace(8.0, 24.5, 260)
    tau_ms = TAU0_PS * np.exp(energies / (KB * T_LIFE)) * 1e-9
    barrier_tau = TAU0_PS * np.exp(barrier / (KB * T_LIFE)) * 1e-9
    ax.semilogy(energies, tau_ms, color="#756bb1", lw=1.9)
    ax.axhline(TARGET_MS, color="0.25", ls="--", lw=1.0,
               label=rf"0.3 ms target: {EA_TARGET:.1f} meV")
    ax.axvline(EA_TARGET, color="0.25", ls="--", lw=1.0)
    ax.axvline(barrier, color="#d95f0e", lw=1.5,
               label=rf"L36 hard fault: {barrier:.2f} meV")
    ax.plot([barrier], [barrier_tau], "o", color="#d95f0e", ms=6.0)
    ax.set_xlabel(r"2D depinning barrier $U$ (meV)")
    ax.set_ylabel(r"$\tau(10\,\mathrm{K})$ (ms)")
    ax.set_title("(d) quasi-2D lifetime scale")
    ax.grid(alpha=0.25, lw=0.5, which="both")
    ax.legend(fontsize=7.5, frameon=False, loc="upper left")
    ax.text(0.58, 0.18, rf"$\tau\approx{barrier_tau:.1f}$ ms" + "\n"
            + "no layer additivity assumed",
            transform=ax.transAxes, ha="left", va="bottom", fontsize=8.2,
            bbox=dict(facecolor="white", edgecolor="0.88", alpha=0.9, pad=2.0))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--switch-summary", type=Path, default=DEFAULT_SWITCH_SUMMARY)
    parser.add_argument("--out-dir", type=Path, default=OUT)
    parser.add_argument("--max-switch-sigma", type=float, default=0.20)
    parser.add_argument("--protocol-label",
                        default=r"$36\times36$: $\delta K\sim\mathcal{N}(-0.5|K|,\sigma_K)$, $hw=2$")
    args = parser.parse_args()

    switch_summary = args.switch_summary if args.switch_summary.is_absolute() else ROOT / args.switch_summary
    out_dir = args.out_dir if args.out_dir.is_absolute() else ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    mep_report = load_selected_mep()
    switching = load_switching(switch_summary)
    barrier = float(mep_report["gneb"]["barrier_meV"])

    fig = plt.figure(figsize=(11.2, 8.2))
    gs = fig.add_gridspec(2, 2, width_ratios=(1.0, 1.15), height_ratios=(1.0, 1.0),
                          hspace=0.35, wspace=0.32)
    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1])
    ax2 = fig.add_subplot(gs[1, 0])
    ax3 = fig.add_subplot(gs[1, 1])
    geometry_axes(ax0)
    mep_axes(ax1, mep_report)
    widths = switching_axes(ax2, switching, args.max_switch_sigma, args.protocol_label)
    lifetime_axes(ax3, barrier)
    fig.suptitle(r"$36\times36$ quasi-2D defect picture for optically written zigzag in NCTO",
                 y=0.995, fontsize=13.0)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    png = out_dir / "one_picture_experimental_signatures.png"
    pdf = out_dir / "one_picture_experimental_signatures.pdf"
    fig.savefig(png, dpi=220)
    fig.savefig(pdf)
    report = {
        "hard_lifetime_defect": {
            "type": "enhanced Kitaev line/band",
            "delta_K": "-0.5 |K| on selected nearest-neighbor bonds",
            "half_width": 2.0,
            "n_bonds": int(mep_report["geometry"]["n_bonds"]),
        },
        "gneb": mep_report["gneb"],
        "endpoints": mep_report["endpoints"],
        "switching_landscape": {
            "type": "enhanced-|K| line/band with Gaussian selected-bond offsets",
            "mean_delta_K": "-0.5 |K|",
            "half_width": 2.0,
            "summary": str(switch_summary.relative_to(ROOT)),
            "max_sigma_plotted": args.max_switch_sigma,
            "protocol_label": args.protocol_label,
            "transition_widths": widths,
        },
        "lifetime": {
            "quasi_2d_barrier_meV": barrier,
            "target_barrier_meV": EA_TARGET,
            "target_tau_ms": TARGET_MS,
            "tau_ms_at_10K": float(TAU0_PS * np.exp(barrier / (KB * T_LIFE)) * 1e-9),
        },
    }
    (out_dir / "one_picture_experimental_signatures_report.json").write_text(
        json.dumps(report, indent=2))
    print(f"wrote {png.relative_to(ROOT)}")
    print(f"wrote {pdf.relative_to(ROOT)}")
    print(f"wrote {(out_dir / 'one_picture_experimental_signatures_report.json').relative_to(ROOT)}")


if __name__ == "__main__":
    main()