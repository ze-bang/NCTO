#!/usr/bin/env python3
"""B2 -- localized-nucleus GNEB: does the 3Q<->ZZ critical nucleus fit in L?

B1 forces a y-uniform (whole-line) path.  Here we ALLOW the nucleus to be finite
along the defect line: the ZZ endpoint is a patch bounded in BOTH x (|dx|<=hw)
and y (|dy|<=half_len), embedded in 3Q, relaxed with the defect present.  GNEB
to uniform 3Q then finds the true saddle.

We read the saddle image and measure the ZZ-aligned fraction of the defect line:
  saddle_line_frac ~ 1  -> the saddle spans the whole line: the critical nucleus
                          is LARGER than L (box too small; barrier is a slab).
  saddle_line_frac < 1  -> a localized critical nucleus fits; ell* ~ frac * L.
Scanning L (and the seed half_len) shows whether ell* saturates below L.

Reuses the proven GNEB machinery from gneb_finite_size_scan (B1).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import ncto_common as nc
import gneb_finite_size_scan as b1
from gneb_finite_size_scan import (Geom, slerp, rn, defect_file, make_cfg, sa_relax,
                                   mep_barrier, rel)

OUT_ROOT = nc.CAMPAIGN / "critical_radius" / "localized_nucleus"
OUT_ROOT.mkdir(parents=True, exist_ok=True)


def patch_init(g: Geom, width: float, half_len: float, wall: float = 1.2) -> np.ndarray:
    """ZZ patch localized in x (|dx|<=width) AND y (|dy|<=half_len), blended into 3Q."""
    dx = g.X - g.XC
    dx -= g.L * np.round(dx / g.L)
    dy = g.Y - g.YC
    dy -= g.L * np.round(dy / g.L)
    bx = np.clip(0.5 - 0.5 * (np.abs(dx) - width) / wall, 0.0, 1.0)
    by = np.clip(0.5 - 0.5 * (np.abs(dy) - half_len) / wall, 0.0, 1.0)
    blend = bx * by
    out = np.empty_like(g.ZZ)
    for i, val in enumerate(blend):
        out[i] = slerp(g.Q3[i:i + 1], g.ZZ[i:i + 1], float(val))[0]
    return rn(out)


def saddle_extent(g: Geom, work_dir: Path, half_width: float) -> dict:
    """Measure the ZZ-aligned span of the saddle configuration along the line."""
    data = np.load(work_dir / "final_band.npz")
    band, energies = data["band"], data["energies"]
    saddle = band[int(np.argmax(energies))]
    zz = np.einsum("ij,ij->i", saddle, g.ZZ) > 0.8
    band_mask = np.abs(g.X - g.XC) <= half_width + 1.5      # sites on/near the line
    line = band_mask
    line_frac = float(zz[line].mean()) if np.any(line) else 0.0
    # y-span of ZZ sites on the line (PBC-aware via largest gap on the ring)
    ys = np.sort(g.Y[zz & line]) if np.any(zz & line) else np.array([])
    if len(ys) >= 2:
        span = float(g.L)
        gaps = np.diff(np.concatenate([ys, [ys[0] + g.L]]))
        span = float(g.L - gaps.max())      # occupied arc = ring minus largest empty gap
    else:
        span = 0.0
    return {"saddle_line_frac": line_frac, "ell_star_cells": span,
            "n_zz_line": int((zz & line).sum()), "n_line": int(line.sum())}


def run_one(lattice: int, half_len: float, *, strength: float, half_width: float,
            images: int, n_iter: int, dt: float, climb_start: int, plateau_win: int,
            force: bool) -> dict:
    tag = f"L{lattice}_hl{half_len:.1f}_s{strength:.3f}_hw{half_width:.2f}".replace(".", "p")
    work_dir = OUT_ROOT / tag
    work_dir.mkdir(parents=True, exist_ok=True)
    report_path = work_dir / "report.json"
    if report_path.exists() and not force:
        print(f"  L={lattice} hl={half_len}: cached", flush=True)
        return json.loads(report_path.read_text())
    g = Geom(lattice)
    deffile, n_bonds = defect_file(g, work_dir, strength, half_width)
    cfg = make_cfg(g, work_dir, deffile)
    print(f"  L={lattice} hl={half_len}: relax endpoints (N={g.N}) ...", flush=True)
    q3 = sa_relax(g, work_dir, g.Q3, deffile, "r3q", 1200, force)
    q3 = sa_relax(g, work_dir, q3, deffile, "r3q2", 1600, force)
    patch = sa_relax(g, work_dir, patch_init(g, 2.5, half_len), deffile, "rpatch", 1600, force)
    print(f"  L={lattice} hl={half_len}: GNEB ...", flush=True)
    gneb = mep_barrier(g, work_dir, cfg, patch, q3, images=images, n_iter=n_iter,
                       dt=dt, climb_start=climb_start, plateau_win=plateau_win)
    ext = saddle_extent(g, work_dir, half_width)
    report = {"lattice": lattice, "seed_half_len": half_len, "N": g.N,
              "strength": strength, "half_width": half_width,
              "barrier_meV": gneb["barrier_meV"], "saddle_image": gneb["saddle_image"],
              "contained": bool(ext["saddle_line_frac"] < 0.9), **ext}
    report_path.write_text(json.dumps(report, indent=2))
    print(f"  L={lattice} hl={half_len}: barrier={gneb['barrier_meV']:.2f} meV  "
          f"line_frac={ext['saddle_line_frac']:.2f}  ell*={ext['ell_star_cells']:.1f}  "
          f"{'CONTAINED' if report['contained'] else 'SPANS-BOX'}", flush=True)
    return report


def main():
    ap = argparse.ArgumentParser(description="B2 localized-nucleus GNEB")
    ap.add_argument("--lattices", nargs="+", type=int, default=[36, 54])
    ap.add_argument("--half-lens", nargs="+", type=float, default=[6.0, 10.0],
                    help="seed ZZ-patch half-lengths along the line")
    ap.add_argument("--strength", type=float, default=0.5)
    ap.add_argument("--half-width", type=float, default=2.0)
    ap.add_argument("--images", type=int, default=21)
    ap.add_argument("--n-iter", type=int, default=1600)
    ap.add_argument("--dt", type=float, default=0.006)
    ap.add_argument("--climb-start", type=int, default=900)
    ap.add_argument("--plateau-win", type=int, default=120)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    results = []
    for L in sorted(args.lattices):
        for hl in sorted(args.half_lens):
            if hl >= L / 2:
                continue
            print(f"\n--- L={L} half_len={hl} ---", flush=True)
            results.append(run_one(L, hl, strength=args.strength, half_width=args.half_width,
                                   images=args.images, n_iter=args.n_iter, dt=args.dt,
                                   climb_start=args.climb_start, plateau_win=args.plateau_win,
                                   force=args.force))
    summary = OUT_ROOT / "localized_nucleus_summary.json"
    summary.write_text(json.dumps(results, indent=2))

    fig, ax = plt.subplots(figsize=(6, 4.4))
    for L in sorted(set(r["lattice"] for r in results)):
        rs = [r for r in results if r["lattice"] == L]
        ax.plot([r["seed_half_len"] for r in rs], [r["saddle_line_frac"] for r in rs],
                "o-", label=f"L={L}")
    ax.axhline(0.9, color="0.6", ls="--", lw=0.8, label="spans-box threshold")
    ax.set_xlabel("seed ZZ half-length (cells)"); ax.set_ylabel("saddle ZZ line-fraction")
    ax.set_ylim(0, 1.05); ax.set_title("B2: does the critical nucleus fit in L?")
    ax.legend(fontsize=8)
    fig.tight_layout()
    png = OUT_ROOT / "localized_nucleus_linefrac.png"
    fig.savefig(png, dpi=160); plt.close(fig)

    print("\n=== B2 localized-nucleus summary ===")
    print(f"{'L':>4} {'hl':>5} {'barrier':>8} {'line_frac':>10} {'ell*':>6}  verdict")
    for r in results:
        print(f"{r['lattice']:4d} {r['seed_half_len']:5.1f} {r['barrier_meV']:8.2f} "
              f"{r['saddle_line_frac']:10.2f} {r['ell_star_cells']:6.1f}  "
              f"{'contained' if r['contained'] else 'spans-box'}")
    print(f"\nwrote {rel(summary)}\nwrote {rel(png)}")


if __name__ == "__main__":
    main()
