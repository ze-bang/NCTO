#!/usr/bin/env python3
"""C2 — GNEB kinetic-barrier ensemble  (rewrite calling C++ engine).

Replaces the Python SLERP-chain proxy of the previous revision.
This script now calls the in-house geodesic NEB implementation
(Bessarab-Uzdin-Jonsson 2015) exposed via
    simulation_mode = kinetic_barrier
in `runners_strain.cpp`, with FIRE optimisation and climbing image
(Henkelman-Jonsson 2000), and endpoint polishing.

The campaign sweep is over a (|epsilon_Eg|, theta_Eg) grid, which is
the quasi-static Born-Oppenheimer proxy for a driven E1 phonon at
fixed amplitude |Q|^2 = |epsilon_Eg|^2 and polarisation theta_Eg.

Outputs:
    NCTO_project/np_campaign_out/C2/runs/{tag}/...    (GNEB outputs)
    NCTO_project/np_campaign_out/C2/barriers.csv      (Q^2, theta, dE)
"""
from __future__ import annotations

import argparse
import math
import re
import subprocess
from pathlib import Path

from common import CAMPAIGN_OUT, REPO_ROOT, SPIN_SOLVER, write_csv
from configs_lib import write_gneb_strain_config

PHASE = CAMPAIGN_OUT / "C2"
PHASE.mkdir(parents=True, exist_ok=True)


def _ensure_endpoints():
    s3q = REPO_ROOT / "NCTO_project/pump_probe/sa_seeds/very_close/sample_0/spin_strain_config.txt"
    szz = REPO_ROOT / "NCTO_project/pump_probe/pp_runs/very_close_F0p07/sample_0/final_spin_strain.txt"
    if s3q.exists() and szz.exists():
        return s3q, szz
    raise SystemExit(
        "C2: endpoint files not found. Expected:\n"
        f"   {s3q}\n   {szz}\n"
        "Run the existing SA seed pipeline first, or supply --initial / --final.")


def _parse_barrier_log(log_path: Path) -> float:
    if not log_path.exists():
        return math.nan
    text = log_path.read_text()
    pat = re.compile(r"(activation energy|barrier|\bdE\b)[^\d-]*([-+]?\d+\.?\d*(?:[eE][-+]?\d+)?)",
                     re.IGNORECASE)
    best = math.nan
    for m in pat.finditer(text):
        try:
            best = float(m.group(2))
        except Exception:
            pass
    return best


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--quick", action="store_true")
    p.add_argument("--initial")
    p.add_argument("--final")
    args = p.parse_args()

    if args.initial and args.final:
        s3q, szz = Path(args.initial), Path(args.final)
    else:
        s3q, szz = _ensure_endpoints()

    if args.quick:
        eps_mags = [0.0, 0.05, 0.10]
        thetas = [0.0, math.pi / 12]
        n_images, max_iter = 12, 5000
    else:
        eps_mags = [0.0, 0.025, 0.05, 0.075, 0.10, 0.15, 0.20]
        thetas = [0.0, math.pi / 12, math.pi / 6, math.pi / 4,
                  math.pi / 3, 5 * math.pi / 12]
        n_images, max_iter = 32, 50000

    rows = []
    for m in eps_mags:
        for th in thetas:
            tag = f"eps{m:.3f}_th{th:.3f}".replace(".", "p")
            outdir = (PHASE / "runs" / tag).relative_to(REPO_ROOT)
            cfg = PHASE / "configs" / f"{tag}.param"
            write_gneb_strain_config(
                cfg, str(outdir),
                initial_state_file=str(s3q.relative_to(REPO_ROOT)),
                final_state_file=str(szz.relative_to(REPO_ROOT)),
                eps_Eg_magnitude=m, eps_Eg_direction=th,
                n_images=n_images, max_iterations=max_iter,
            )
            log = PHASE / "logs" / f"{tag}.log"
            log.parent.mkdir(parents=True, exist_ok=True)
            with log.open("w") as fh:
                proc = subprocess.run([str(SPIN_SOLVER), str(cfg)],
                                      cwd=REPO_ROOT, stdout=fh,
                                      stderr=subprocess.STDOUT)
            rows.append({
                "eps_Eg": float(m), "theta": float(th),
                "Q2_proxy": float(m * m),
                "barrier": _parse_barrier_log(log),
                "returncode": proc.returncode,
            })

    write_csv(PHASE / "barriers.csv", rows)
    print(f"C2 done: {len(rows)} GNEB runs -> {PHASE/'barriers.csv'}")


if __name__ == "__main__":
    main()
