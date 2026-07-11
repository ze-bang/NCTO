#!/usr/bin/env python3
"""Kinetic depinning barrier vs defect width -- continuation GNEB sweep.

Computes the 3Q<->ZZ depinning barrier of the enhanced-|K| line defect as a
function of half-width hw at fixed L, sweeping hw in ascending order with
CONTINUATION: each hw warm-starts both its endpoints and its GNEB band from the
previous hw's converged result.  This keeps the whole sweep on one continuous
family of minimum-energy paths, eliminating the basin-hopping scatter that an
independent initialization per hw produces.

Outputs (merged into the kinetic_barrier folder, next to barrier_vs_j7.csv):
  kinetic_barrier/width_sweep/hw*/report.json + final_band.npz
  kinetic_barrier/width_sweep/barrier_vs_hw.csv
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

import ncto_common as nc
from gneb_finite_size_scan import (Geom, defect_file, make_cfg, sa_relax,
                                   strip_init, mep_barrier, rel)

OUT = nc.CAMPAIGN / "kinetic_barrier" / "width_sweep"
OUT.mkdir(parents=True, exist_ok=True)


def run_sweep(args) -> list[dict]:
    g = Geom(args.lattice)
    results = []
    prev_q3 = prev_strip = prev_band = None

    for hw in sorted(args.half_width_values):
        tag = f"hw{hw:.2f}".replace(".", "p")
        work_dir = OUT / tag
        work_dir.mkdir(parents=True, exist_ok=True)
        report_path = work_dir / "report.json"
        band_path = work_dir / "final_band.npz"

        if report_path.exists() and not args.force:
            r = json.loads(report_path.read_text())
            results.append(r)
            if band_path.exists():
                dat = np.load(band_path)
                prev_band = dat["band"]
                prev_strip, prev_q3 = prev_band[0], prev_band[-1]
            print(f"  hw={hw:.2f}: cached  barrier={r['barrier_meV']:.2f} meV", flush=True)
            continue

        deffile, n_bonds = defect_file(g, work_dir, args.strength, hw)
        cfg = make_cfg(g, work_dir, deffile)

        # endpoints: continuation from previous hw where available
        q3_seed = prev_q3 if prev_q3 is not None else g.Q3
        strip_seed = prev_strip if prev_strip is not None else strip_init(g, 2.5)
        q3 = sa_relax(g, work_dir, q3_seed, deffile, "r3q", 1600, args.force)
        strip = sa_relax(g, work_dir, strip_seed, deffile, "rstrip", 1600, args.force)

        print(f"  hw={hw:.2f}: GNEB ({n_bonds} defect bonds"
              f"{', warm-start' if prev_band is not None else ', fresh band'}) ...", flush=True)
        gneb = mep_barrier(g, work_dir, cfg, strip, q3,
                           images=args.images, n_iter=args.n_iter, dt=args.dt,
                           climb_start=args.climb_start, plateau_win=args.plateau_win,
                           init_band=prev_band)
        report = {"lattice": args.lattice, "strength": args.strength, "half_width": hw,
                  "n_defect_bonds": n_bonds, "barrier_meV": gneb["barrier_meV"],
                  "saddle_image": gneb["saddle_image"],
                  "iterations": gneb["iterations_completed"]}
        report_path.write_text(json.dumps(report, indent=2))
        results.append(report)
        print(f"  hw={hw:.2f}: barrier={gneb['barrier_meV']:.2f} meV "
              f"(saddle img {gneb['saddle_image']}, {gneb['iterations_completed']} iters)", flush=True)

        dat = np.load(band_path)
        prev_band = dat["band"]
        prev_strip, prev_q3 = strip, q3
    return results


def main() -> None:
    ap = argparse.ArgumentParser(description="continuation GNEB barrier-vs-width sweep")
    ap.add_argument("--lattice", type=int, default=36)
    ap.add_argument("--strength", type=float, default=0.5)
    ap.add_argument("--half-width-values", nargs="+", type=float,
                    default=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0])
    ap.add_argument("--images", type=int, default=25)
    ap.add_argument("--n-iter", type=int, default=4000)
    ap.add_argument("--dt", type=float, default=0.006)
    ap.add_argument("--climb-start", type=int, default=2400)
    ap.add_argument("--plateau-win", type=int, default=250)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    print(f"continuation width sweep: L={args.lattice} s={args.strength} "
          f"hw={sorted(args.half_width_values)}", flush=True)
    results = run_sweep(args)

    csv_path = OUT / "barrier_vs_hw.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["half_width", "barrier_meV", "dE_3Q_minus_strip_meV",
                                          "n_defect_bonds", "saddle_image", "iterations", "valid"])
        w.writeheader()
        for r in sorted(results, key=lambda r: r["half_width"]):
            tag = f"hw{r['half_width']:.2f}".replace(".", "p")
            e = np.load(OUT / tag / "final_band.npz")["energies"]
            dE = float(e[-1] - e[0])
            # valid = converged decay barrier: interior saddle AND the strip is
            # METASTABLE (3Q endpoint below it). Quarter-widths make the pinned
            # strip the ground state (unbalanced K-enhanced bond row) -> no decay.
            w.writerow(dict(half_width=r["half_width"], barrier_meV=r["barrier_meV"],
                            dE_3Q_minus_strip_meV=round(dE, 3),
                            n_defect_bonds=r["n_defect_bonds"], saddle_image=r["saddle_image"],
                            iterations=r["iterations"],
                            valid=int(0 < r["saddle_image"] < args.images - 1 and dE < 0.0)))
    print(f"\n{'hw':>6} {'bonds':>6} {'barrier(meV)':>13} {'dE(meV)':>9}")
    for r in sorted(results, key=lambda r: r["half_width"]):
        tag = f"hw{r['half_width']:.2f}".replace(".", "p")
        e = np.load(OUT / tag / "final_band.npz")["energies"]
        print(f"{r['half_width']:6.2f} {r['n_defect_bonds']:6d} {r['barrier_meV']:13.2f} {e[-1]-e[0]:9.2f}")
    print(f"wrote {rel(csv_path)}")


if __name__ == "__main__":
    main()
