#!/usr/bin/env python3
"""Convergence check for the width-sweep MEPs: resample each converged band to
more images and re-relax (warm start).  If the barrier is converged, dG must be
stable under refinement and the saddle region smooth instead of cuspy.

Writes width_sweep/hw*/refine<IMAGES>/{report.json, final_band.npz} and a
comparison CSV.  Each hw refines independently (parallelize externally).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import ncto_common as nc
from gneb_finite_size_scan import (Geom, defect_file, make_cfg, slerp,
                                   geodesic_len, mep_barrier, rel)

WS = nc.CAMPAIGN / "kinetic_barrier" / "width_sweep"


def resample_band(band: np.ndarray, m: int) -> np.ndarray:
    """Resample a band to m images, uniformly in geodesic path length."""
    coord = geodesic_len(band)
    total = coord[-1]
    target = np.linspace(0.0, total, m)
    out = [band[0].copy()]
    for v in target[1:-1]:
        si = max(0, min(int(np.searchsorted(coord, v) - 1), len(band) - 2))
        seg = coord[si + 1] - coord[si]
        frac = 0.0 if seg < 1e-12 else (v - coord[si]) / seg
        out.append(slerp(band[si], band[si + 1], float(frac)))
    out.append(band[-1].copy())
    return np.array(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hw", type=float, required=True)
    ap.add_argument("--lattice", type=int, default=36)
    ap.add_argument("--strength", type=float, default=0.5)
    ap.add_argument("--images", type=int, default=41)
    ap.add_argument("--n-iter", type=int, default=3000)
    ap.add_argument("--dt", type=float, default=0.006)
    ap.add_argument("--climb-start", type=int, default=1200)
    ap.add_argument("--plateau-win", type=int, default=250)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    tag = f"hw{args.hw:.2f}".replace(".", "p")
    src = WS / tag
    work = src / f"refine{args.images}"
    work.mkdir(parents=True, exist_ok=True)
    report_path = work / "report.json"
    if report_path.exists() and not args.force:
        print(f"hw={args.hw}: refine cached: {json.loads(report_path.read_text())['barrier_meV']:.2f} meV")
        return

    base = json.loads((src / "report.json").read_text())
    band0 = np.load(src / "final_band.npz")["band"]
    g = Geom(args.lattice)
    deffile, _ = defect_file(g, work, args.strength, args.hw)
    cfg = make_cfg(g, work, deffile)

    band = resample_band(band0, args.images)
    print(f"hw={args.hw}: refining {len(band0)} -> {args.images} images "
          f"(base barrier {base['barrier_meV']:.2f} meV) ...", flush=True)
    gneb = mep_barrier(g, work, cfg, band[0], band[-1],
                       images=args.images, n_iter=args.n_iter, dt=args.dt,
                       climb_start=args.climb_start, plateau_win=args.plateau_win,
                       init_band=band)
    report = {"half_width": args.hw, "images": args.images,
              "barrier_meV": gneb["barrier_meV"], "saddle_image": gneb["saddle_image"],
              "iterations": gneb["iterations_completed"],
              "base_barrier_meV": base["barrier_meV"],
              "delta_vs_base_meV": gneb["barrier_meV"] - base["barrier_meV"]}
    report_path.write_text(json.dumps(report, indent=2))
    print(f"hw={args.hw}: refined barrier={gneb['barrier_meV']:.2f} meV "
          f"(base {base['barrier_meV']:.2f}, delta {report['delta_vs_base_meV']:+.2f})", flush=True)


if __name__ == "__main__":
    main()
