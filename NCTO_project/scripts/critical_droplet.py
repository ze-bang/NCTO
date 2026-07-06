#!/usr/bin/env python3
"""Seed-independent critical-nucleus measurement done the right way.

Instead of picking a box and reading a (finite-size, extensive) GNEB barrier, we
SIZE the critical nucleus first and verify the geometry contains it:

  1. For a chosen J7 (OFF the 3Q/ZZ degeneracy so the bulk driving force Df is
     finite), relax the uniform 3Q and ZZ references -> Df = (E_ZZ - E_3Q)/N.
  2. Build a compact ZZ droplet of radius R in a 3Q background (radial tanh
     interface) and evaluate its energy vs the uniform 3Q:  dE(R).
     dE(R) is seed-independent (energy of a geometric configuration, not a
     dynamical endpoint); in 2D CNT  dE(R) ~ 2*pi*sigma*R - pi*Df_area*R^2.
  3. The MAXIMUM of dE(R) is the critical nucleus:  R* = argmax,  dG* = max.
     If dE(R) only rises up to R_max (limited by L), R* > box  -> geometry does
     NOT contain the critical radius (increase L or move further off degeneracy).
  4. Scan J7 (=> Df) to show R* ~ sigma/Df grows toward the degeneracy, and
     scan L to confirm R*, dG* are converged (box big enough).

This is the geometry-aware replacement for the y-uniform slab GNEB.
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
from ncto_common import ROOT, LatticeContext, sa_relax, energy_of, tile_l18_reference

OUT_ROOT = nc.CAMPAIGN / "critical_radius" / "droplet_energetics"
OUT_ROOT.mkdir(parents=True, exist_ok=True)


def _rel(p: Path) -> Path:
    try:
        return p.relative_to(ROOT)
    except ValueError:
        return p


def rn(s: np.ndarray) -> np.ndarray:
    return s / np.linalg.norm(s, axis=-1, keepdims=True)


def slerp_pair(a: np.ndarray, b: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Per-site slerp between configs a,b with per-site parameter t in [0,1]."""
    dot = np.clip(np.einsum("ij,ij->i", a, b), -1.0, 1.0)
    omega = np.arccos(dot)
    so = np.sin(omega)
    out = np.empty_like(a)
    lin = so < 1e-7
    out[lin] = (1.0 - t[lin])[:, None] * a[lin] + t[lin][:, None] * b[lin]
    nl = ~lin
    o = omega[nl]
    out[nl] = (np.sin((1.0 - t[nl]) * o)[:, None] * a[nl]
               + np.sin(t[nl] * o)[:, None] * b[nl]) / so[nl][:, None]
    return rn(out)


class Ctx:
    def __init__(self, lattice: int):
        self.L = lattice
        c = LatticeContext(lattice)
        self.n = c.n_sites
        self.x = c.x
        self.y = c.y
        self.xc = c.xc
        self.yc = 0.5 * (float(c.y.min()) + float(c.y.max()))
        self.q3_seed = tile_l18_reference(nc.REF_3Q_L18, lattice)
        self.zz_seed = tile_l18_reference(nc.REF_ZZ_L18, lattice)


def droplet(ctx: Ctx, q3: np.ndarray, zz: np.ndarray, R: float, wall: float = 1.2) -> np.ndarray:
    dx = ctx.x - ctx.xc
    dx -= ctx.L * np.round(dx / ctx.L)
    dy = ctx.y - ctx.yc
    dy -= ctx.L * np.round(dy / ctx.L)
    r = np.sqrt(dx * dx + dy * dy)
    blend = np.clip(0.5 - 0.5 * (r - R) / wall, 0.0, 1.0)   # 1 inside (ZZ) -> 0 outside (3Q)
    return slerp_pair(q3, zz, blend)


def run_j7(lattice: int, j7: float, r_values: list[float], force: bool) -> dict:
    tag = f"L{lattice}_J7{int(abs(j7)*1000):04d}".replace(".", "p")
    base = OUT_ROOT / tag
    base.mkdir(parents=True, exist_ok=True)
    ctx = Ctx(lattice)
    # relax uniform references at this J7 (no defect, no drive)
    q3 = sa_relax(lattice=lattice, j7=j7, work_dir=base / "u3q", seed_spins=ctx.q3_seed,
                  n_deterministics=2000, force=force)
    zz = sa_relax(lattice=lattice, j7=j7, work_dir=base / "uzz", seed_spins=ctx.zz_seed,
                  n_deterministics=2000, force=force)
    if q3 is None or zz is None:
        return {}
    e3q = energy_of(lattice=lattice, j7=j7, work_dir=base / "u3q_E", spins=q3)
    ezz = energy_of(lattice=lattice, j7=j7, work_dir=base / "uzz_E", spins=zz)
    if e3q is None or ezz is None:
        return {}
    df_site = (ezz - e3q) / ctx.n            # >0 => ZZ metastable (3Q ground state)

    # Physically relevant nucleus for the lifetime of a pump-created ZZ region:
    # a 3Q droplet (the FAVOURABLE phase) nucleating inside the metastable ZZ
    # background -> dE(R) = 2*pi*sigma*R - pi*Df_area*R^2 turns over at R*.
    curve = []
    for R in r_values:
        d = droplet(ctx, zz, q3, R)          # inside = 3Q, background = ZZ
        eR = energy_of(lattice=lattice, j7=j7, work_dir=base / f"R{R:.1f}".replace(".", "p"), spins=d)
        if eR is None:
            continue
        curve.append((float(R), float(eR - ezz)))   # excess over metastable ZZ
    curve.sort()
    Rs = np.array([c[0] for c in curve])
    dE = np.array([c[1] for c in curve])

    # critical nucleus = peak of dE(R); parabola fit dE = a R - b R^2 near the peak
    imax = int(np.argmax(dE)) if len(dE) else 0
    turned_over = 0 < imax < len(dE) - 1
    Rstar = float(Rs[imax]) if len(dE) else float("nan")
    dGstar = float(dE[imax]) if len(dE) else float("nan")
    a = b = float("nan")
    if len(dE) >= 3:
        b2, b1, b0 = np.polyfit(Rs, dE, 2)   # dE ~ b2 R^2 + b1 R + b0
        a, b = float(b1), float(-b2)
        if b > 0:
            Rstar_fit = a / (2 * b)
            dGstar_fit = a * a / (4 * b)
        else:
            Rstar_fit = dGstar_fit = float("nan")
    else:
        Rstar_fit = dGstar_fit = float("nan")

    report = {
        "lattice": lattice, "J7": j7, "N": ctx.n,
        "E3Q_meV": e3q, "EZZ_meV": ezz, "Df_per_site_meV": df_site,
        "R_values": Rs.tolist(), "dE_meV": dE.tolist(),
        "turned_over_in_box": bool(turned_over),
        "Rstar_argmax": Rstar, "dGstar_argmax_meV": dGstar,
        "Rstar_fit": Rstar_fit, "dGstar_fit_meV": dGstar_fit,
        "line_tension_a_meV_per_cell": a, "areal_Df_b_meV_per_cell2": b,
        "contains_critical_radius": bool(turned_over),
    }
    (base / "report.json").write_text(json.dumps(report, indent=2))
    verdict = "CONTAINS R*" if turned_over else "R* > BOX (too small / too near degeneracy)"
    print(f"  L={lattice} J7={j7:+.2f}: Df/site={df_site:+.4f}  R*~{Rstar:.1f} "
          f"(fit {Rstar_fit:.1f})  dG*~{dGstar:.2f} meV  -> {verdict}", flush=True)
    return report


def main():
    ap = argparse.ArgumentParser(description="Seed-independent critical-droplet energetics")
    ap.add_argument("--lattice", type=int, default=54)
    ap.add_argument("--j7-values", nargs="+", type=float, default=[-0.45, -0.50, -0.55, -0.60, -0.70])
    ap.add_argument("--r-values", nargs="+", type=float,
                    default=[2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16])
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    reports = []
    for j7 in args.j7_values:
        print(f"\n--- L={args.lattice} J7={j7:+.2f} ---", flush=True)
        r = run_j7(args.lattice, j7, args.r_values, args.force)
        if r:
            reports.append(r)
    summary = OUT_ROOT / f"droplet_summary_L{args.lattice}.json"
    summary.write_text(json.dumps(reports, indent=2))

    # dE(R) curves
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(11, 4.4))
    for r in reports:
        ax0.plot(r["R_values"], r["dE_meV"], "o-", label=f"J7={r['J7']:+.2f} (Df={r['Df_per_site_meV']:+.3f})")
        if r["turned_over_in_box"]:
            ax0.plot(r["Rstar_argmax"], r["dGstar_argmax_meV"], "k*", ms=12)
    ax0.axhline(0, color="0.7", lw=0.8)
    ax0.set_xlabel("droplet radius R (cells)"); ax0.set_ylabel(r"$\Delta E(R)$ (meV)")
    ax0.set_title(f"Critical droplet energetics, L={args.lattice}\n(peak=R*; monotone rise=box too small)")
    ax0.legend(fontsize=7)
    # R* vs Df  (CNT: R* ~ sigma/Df diverges toward degeneracy)
    dfs = [r["Df_per_site_meV"] for r in reports if r["turned_over_in_box"]]
    rst = [r["Rstar_argmax"] for r in reports if r["turned_over_in_box"]]
    if dfs:
        ax1.plot(dfs, rst, "o-", color="C3")
    ax1.set_xlabel(r"$\Delta f$ per site (meV)"); ax1.set_ylabel("R* (cells)")
    ax1.set_title("critical radius vs driving force\n(diverges as J7 -> degeneracy)")
    fig.tight_layout()
    png = OUT_ROOT / f"droplet_energetics_L{args.lattice}.png"
    fig.savefig(png, dpi=160); plt.close(fig)

    print("\n=== critical-droplet summary ===")
    print(f"{'J7':>6} {'Df/site':>9} {'R*':>6} {'dG*(meV)':>9}  contains-R*")
    for r in reports:
        print(f"{r['J7']:+6.2f} {r['Df_per_site_meV']:+9.4f} {r['Rstar_argmax']:6.1f} "
              f"{r['dGstar_argmax_meV']:9.2f}  {r['contains_critical_radius']}")
    print(f"\nwrote {_rel(summary)}\nwrote {_rel(png)}")


if __name__ == "__main__":
    main()
