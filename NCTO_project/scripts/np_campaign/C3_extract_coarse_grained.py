#!/usr/bin/env python3
"""C3 — coarse-grained parameter extraction (PT-based rewrite).

Delta_f0 from PT energy means at the lowest temperature;
sigma from the elongated-slab excess-energy method;
M_wall from C6 droplet decay if available;
threshold from C1 results.

References:
  Hukushima & Nemoto, JPSJ 65, 1604 (1996).
  Ferrenberg & Swendsen, PRL 61, 2635 (1988).
  Schryer & Walker, JAP 45, 5406 (1974).
"""
from __future__ import annotations

import argparse
import statistics
import subprocess
from collections import defaultdict
from pathlib import Path

import numpy as np

from analysis_utils import read_csv
from common import CAMPAIGN_OUT, REPO_ROOT, SPIN_SOLVER, write_csv
from configs_lib import write_pt_strain_config

PHASE = CAMPAIGN_OUT / "C3"
PHASE.mkdir(parents=True, exist_ok=True)


def _run(cfg: Path, log: Path, n_ranks: int = 1) -> int:
    log.parent.mkdir(parents=True, exist_ok=True)
    cmd = [str(SPIN_SOLVER), str(cfg)]
    if n_ranks > 1:
        cmd = ["mpirun", "--oversubscribe", "-n", str(n_ranks)] + cmd
    with log.open("w") as fh:
        proc = subprocess.run(cmd, cwd=REPO_ROOT, stdout=fh,
                              stderr=subprocess.STDOUT)
    return proc.returncode


def _load_pt_energies(out_dir: Path, T_grid: list) -> dict:
    """Read per-rank parallel_tempering_data.h5 and return {T: E_samples}."""
    out_dir = REPO_ROOT / out_dir
    es: dict = {}
    sample = out_dir / "sample_0"
    if not sample.exists():
        return es
    rank_dirs = sorted(sample.glob("rank_*"))
    try:
        import h5py
    except Exception:
        return es
    for k, rd in enumerate(rank_dirs):
        h5 = rd / "parallel_tempering_data.h5"
        if not h5.exists():
            continue
        try:
            with h5py.File(h5, "r") as h:
                E = h["timeseries/energy"][:].astype(float)
        except Exception:
            continue
        T = T_grid[k] if k < len(T_grid) else float("nan")
        es[float(T)] = E
    return es


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--quick", action="store_true")
    p.add_argument("--skip_pt", action="store_true")
    args = p.parse_args()

    if args.quick:
        T_grid = [0.02, 0.05, 0.10]
        n_anneal, n_measure = 5000, 5000
    else:
        T_grid = [0.005, 0.01, 0.02, 0.04, 0.08, 0.15]
        n_anneal, n_measure = 50000, 50000

    out3q = PHASE / "pt_3q"
    outzz = PHASE / "pt_zz"
    outwall = PHASE / "pt_wall"
    for d in (out3q, outzz, outwall):
        d.mkdir(parents=True, exist_ok=True)

    cfg3q = PHASE / "pt_3q.param"
    cfgzz = PHASE / "pt_zz.param"
    cfgwl = PHASE / "pt_wall.param"

    write_pt_strain_config(cfg3q, str(out3q.relative_to(REPO_ROOT)),
                           T_list=T_grid, n_anneal=n_anneal, n_measure=n_measure)
    write_pt_strain_config(cfgzz, str(outzz.relative_to(REPO_ROOT)),
                           T_list=T_grid, n_anneal=n_anneal, n_measure=n_measure)
    write_pt_strain_config(cfgwl, str(outwall.relative_to(REPO_ROOT)),
                           T_list=T_grid, n_anneal=n_anneal, n_measure=n_measure,
                           lattice_size=(12, 4, 1))

    if not args.skip_pt:
        n_ranks = len(T_grid)
        print(f"==> C3 PT (3Q) on {n_ranks} ranks")
        _run(cfg3q, PHASE / "logs" / "pt_3q.log", n_ranks=n_ranks)
        print(f"==> C3 PT (ZZ) on {n_ranks} ranks")
        _run(cfgzz, PHASE / "logs" / "pt_zz.log", n_ranks=n_ranks)
        print(f"==> C3 PT (wall slab) on {n_ranks} ranks")
        _run(cfgwl, PHASE / "logs" / "pt_wall.log", n_ranks=n_ranks)

    es_3q = _load_pt_energies(out3q, T_grid)
    es_zz = _load_pt_energies(outzz, T_grid)
    es_wl = _load_pt_energies(outwall, T_grid)

    df0 = float("nan")
    if es_3q and es_zz:
        T_low = min(es_3q)
        df0 = float(np.mean(es_3q[T_low]) - np.mean(es_zz[T_low]))

    sigma = float("nan")
    if es_wl and es_3q and es_zz:
        T_low = min(es_wl)
        if T_low in es_3q and T_low in es_zz:
            Ewl = float(np.mean(es_wl[T_low]))
            Eref = 0.5 * (float(np.mean(es_3q[T_low])) +
                          float(np.mean(es_zz[T_low])))
            sigma = (Ewl - Eref) / 12.0

    M_wall = float("nan")
    c6 = CAMPAIGN_OUT / "C6" / "droplet_radius_clean.csv"
    if c6.exists():
        rows = read_csv(c6)
        if len(rows) > 5:
            t = np.array([r["t"] for r in rows])
            R = np.array([r["R"] for r in rows])
            mask = R > 2.0
            if mask.sum() > 4 and df0 == df0 and df0 != 0:
                slope = np.polyfit(t[mask], R[mask], 1)[0]
                M_wall = -float(slope) / abs(df0)

    # Threshold samples (from C1)
    c1 = CAMPAIGN_OUT / "C1" / "results.csv"
    thr_samples = []
    if c1.exists():
        rows = read_csv(c1)
        grp = defaultdict(list)
        for r in rows:
            if abs(r.get("theta", 0)) > 1e-3:
                continue
            key = (r.get("J"), r.get("K"), r.get("Gamma"),
                   r.get("Gammap"), r.get("channel"))
            grp[key].append((r.get("E0"), int(r.get("switched", 0))))
        for key, pairs in grp.items():
            pairs.sort(key=lambda x: x[0] if x[0] is not None else 1e9)
            thr = next((E0 for E0, sw in pairs if sw), None)
            thr_samples.append({"J": key[0], "K": key[1], "Gamma": key[2],
                                "Gammap": key[3], "channel": key[4],
                                "E0_threshold": thr if thr is not None else float("nan")})

    thr_vals = [t["E0_threshold"] for t in thr_samples
                if t["E0_threshold"] == t["E0_threshold"]]
    summary = [{
        "Delta_f0_per_site": df0,
        "sigma_wall": sigma,
        "M_wall": M_wall,
        "n_threshold_samples": len(thr_samples),
        "E0_threshold_median": statistics.median(thr_vals) if thr_vals else float("nan"),
    }]
    write_csv(PHASE / "coarse_grained.csv", summary)
    write_csv(PHASE / "threshold_samples.csv", thr_samples)
    print("C3 done:")
    for k, v in summary[0].items():
        print(f"  {k} = {v}")


if __name__ == "__main__":
    main()
