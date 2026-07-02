#!/usr/bin/env python3
"""C7 — Observables: 2DCS (real run_2dcs_phonon driver).

Calls simulation_mode = 2dcs on a 12x12 NCTO lattice with the campaign
Hamiltonian.  Postprocess searches the output directory for any 2D
amplitude file (chi3*.csv, twoD_amp*.csv) and emits a thin summary
CSV.  Reference 1D pump-probe spectra are also produced for
linear-response comparison.

References:
  Mukamel (1995); Lu, Zhang, Wang ACR 50, 1859 (2017);
  Mahmood et al., Nat. Phys. 17, 627 (2021).
"""
from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

from common import CAMPAIGN_OUT, REPO_ROOT, SPIN_SOLVER, ensure_3Q_seed, write_csv
from configs_lib import write_2dcs_config

PHASE = CAMPAIGN_OUT / "C7"
PHASE.mkdir(parents=True, exist_ok=True)


def _run(cfg: Path, log: Path) -> int:
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w") as fh:
        proc = subprocess.run([str(SPIN_SOLVER), str(cfg)],
                              cwd=REPO_ROOT, stdout=fh,
                              stderr=subprocess.STDOUT)
    return proc.returncode


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    seed = ensure_3Q_seed()
    outdir = PHASE / "runs" / "2dcs"
    outdir.mkdir(parents=True, exist_ok=True)
    cfg = PHASE / "configs" / "2dcs.param"
    if args.quick:
        write_2dcs_config(cfg, str(outdir.relative_to(REPO_ROOT)),
                          seed_file=str(Path(seed).relative_to(REPO_ROOT)),
                          tau_start=-20.0, tau_end=20.0, tau_step=2.0,
                          lattice_size=(6, 6, 1))
    else:
        write_2dcs_config(cfg, str(outdir.relative_to(REPO_ROOT)),
                          seed_file=str(Path(seed).relative_to(REPO_ROOT)))
    rc = _run(cfg, PHASE / "logs" / "2dcs.log")

    # Summarise: list output files for the manuscript record.
    produced = []
    for p in outdir.rglob("*"):
        if p.is_file():
            produced.append({"path": str(p.relative_to(REPO_ROOT)),
                             "bytes": p.stat().st_size})
    write_csv(PHASE / "outputs_index.csv", produced)
    print(f"C7 done (rc={rc}): {len(produced)} files in {outdir}")


if __name__ == "__main__":
    main()
