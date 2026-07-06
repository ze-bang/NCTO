#!/usr/bin/env python3
"""Relaxed 3Q|ZZ wall tension -- calibrates the critical-droplet R*.

A droplet's dE(R) uses a fixed tanh wall, which overestimates the interface
energy (sigma_bare) and hence R*.  A FLAT interface has no curvature-driven
runaway, so we can relax it: build a ZZ y-slab in 3Q, relax at (near) the 3Q/ZZ
degeneracy (Df~0, so the interface does not drift), and read the excess energy
over the uniform state = 2*sigma*L_x (two interfaces).

Reports sigma_bare (unrelaxed slab) and sigma_relaxed, and their ratio -- the
factor by which critical_droplet.py's bare R*, dG* should be scaled down.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import ncto_common as nc
from ncto_common import ROOT, LatticeContext, sa_relax, energy_of, tile_l18_reference
from critical_droplet import slerp_pair

OUT = nc.CAMPAIGN / "critical_radius" / "wall_tension"
OUT.mkdir(parents=True, exist_ok=True)


def _rel(p: Path) -> Path:
    try:
        return p.relative_to(ROOT)
    except ValueError:
        return p


def slab_config(ctx, q3, zz, half_slab: float, wall: float = 1.2) -> np.ndarray:
    """ZZ slab |y-yc| < half_slab embedded in 3Q, two flat x-interfaces."""
    dy = ctx.y - 0.5 * (float(ctx.y.min()) + float(ctx.y.max()))
    dy -= ctx.lattice * np.round(dy / ctx.lattice)
    blend = np.clip(0.5 - 0.5 * (np.abs(dy) - half_slab) / wall, 0.0, 1.0)
    return slerp_pair(q3, zz, blend)


def zz_frac_region(ctx, spins, zz, mask):
    aligned = np.einsum("ij,ij->i", spins, zz) > 0.8
    return float(aligned[mask].mean()) if np.any(mask) else 0.0


def main():
    ap = argparse.ArgumentParser(description="relaxed 3Q|ZZ wall tension")
    ap.add_argument("--lattice", type=int, default=54)
    ap.add_argument("--j7", type=float, default=-0.42, help="~degeneracy so the slab does not drift")
    ap.add_argument("--half-slab", type=float, default=None, help="ZZ slab half-height (default L/4)")
    ap.add_argument("--n-det", type=int, default=1200)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    L = args.lattice
    c = LatticeContext(L)
    ctx = type("C", (), {})()
    ctx.lattice, ctx.y = L, c.y
    q3_seed = tile_l18_reference(nc.REF_3Q_L18, L)
    zz_seed = tile_l18_reference(nc.REF_ZZ_L18, L)
    base = OUT / f"L{L}_J7{int(abs(args.j7)*1000):04d}".replace(".", "p")
    base.mkdir(parents=True, exist_ok=True)

    # uniform references at this J7
    q3 = sa_relax(lattice=L, j7=args.j7, work_dir=base / "u3q", seed_spins=q3_seed,
                  n_deterministics=2000, force=args.force)
    zz = sa_relax(lattice=L, j7=args.j7, work_dir=base / "uzz", seed_spins=zz_seed,
                  n_deterministics=2000, force=args.force)
    e3q = energy_of(lattice=L, j7=args.j7, work_dir=base / "u3q_E", spins=q3)
    ezz = energy_of(lattice=L, j7=args.j7, work_dir=base / "uzz_E", spins=zz)
    n = 2 * L * L
    df_site = (ezz - e3q) / n
    e_uniform = 0.5 * (e3q + ezz)     # degenerate reference at ~Df=0

    half_slab = args.half_slab if args.half_slab is not None else L / 4.0
    slab_seed = slab_config(ctx, q3, zz, half_slab)

    # bare (unrelaxed) slab energy
    e_bare = energy_of(lattice=L, j7=args.j7, work_dir=base / "slab_bare_E", spins=slab_seed)
    # relaxed slab
    slab_relaxed = sa_relax(lattice=L, j7=args.j7, work_dir=base / "slab_relax",
                            seed_spins=slab_seed, n_deterministics=args.n_det, force=args.force)
    e_relaxed = energy_of(lattice=L, j7=args.j7, work_dir=base / "slab_relax_E", spins=slab_relaxed)

    # interface length: two interfaces, each spanning the box in x (position units)
    Lx = float(c.x.max() - c.x.min())
    # survival check: did the ZZ slab stay ~half the box?
    dy = c.y - 0.5 * (float(c.y.min()) + float(c.y.max()))
    slab_mask = np.abs(dy) < half_slab
    zzf = zz_frac_region(ctx, slab_relaxed, zz, slab_mask)

    sigma_bare = (e_bare - e_uniform) / (2.0 * Lx)
    sigma_relaxed = (e_relaxed - e_uniform) / (2.0 * Lx)
    ratio = sigma_relaxed / sigma_bare if sigma_bare else float("nan")

    report = {
        "lattice": L, "J7": args.j7, "Df_per_site_meV": df_site,
        "Lx_position_units": Lx, "half_slab": half_slab,
        "E_uniform_meV": e_uniform, "E_slab_bare_meV": e_bare, "E_slab_relaxed_meV": e_relaxed,
        "sigma_bare_meV_per_len": sigma_bare, "sigma_relaxed_meV_per_len": sigma_relaxed,
        "sigma_ratio_relaxed_over_bare": ratio,
        "slab_survived_zzfrac": zzf, "slab_ok": bool(zzf > 0.6),
    }
    (base / "report.json").write_text(json.dumps(report, indent=2))
    OUT.joinpath("wall_tension_summary.json").write_text(json.dumps(report, indent=2))

    print("=== relaxed 3Q|ZZ wall tension ===")
    print(f"  L={L} J7={args.j7:+.2f}  Df/site={df_site:+.4f} meV")
    print(f"  sigma_bare   = {sigma_bare:.3f} meV/len")
    print(f"  sigma_relaxed= {sigma_relaxed:.3f} meV/len   (ratio {ratio:.2f})")
    print(f"  slab survived: zz_frac={zzf:.2f}  ({'OK' if report['slab_ok'] else 'COLLAPSED -> sigma_relaxed unreliable'})")
    print(f"  => scale critical_droplet R*, dG* by ~{ratio:.2f} for the relaxed-wall estimate")
    print(f"wrote {_rel(base / 'report.json')}")


if __name__ == "__main__":
    main()
