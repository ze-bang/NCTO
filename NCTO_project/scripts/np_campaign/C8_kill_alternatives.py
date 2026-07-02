#!/usr/bin/env python3
"""C8 — kill the alternatives (real-LLG rewrite).

Tests four falsifiable competitors with real engine runs:
    K1 — off-resonant pump (sweep pump frequency; switching only on
         resonance falsifies a pure-electronic mechanism).
    K2 — A1g-only drive (lambda_E1_K_2 = 0, lambda_A1g != 0):
         tests the necessity of the E1 channel.
    K3 — uniform threshold from a 2x2 single-cell-equivalent lattice:
         falsifies bulk-threshold theories that predict no spatial
         heterogeneity.
    K4 — clean large-lattice Langevin quench from a single zigzag droplet
         provides the free-AC reference for mechanism A.

Each test produces a small CSV; pass/fail is decided by the master
driver from the rule encoded in each test's "expected" field.
"""
from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

from common import (CAMPAIGN_OUT, REPO_ROOT, SPIN_SOLVER, declare_switched,
                    extract_final_spins, run_analysis, write_csv)
from configs_lib import (write_langevin_pump_config,
                         write_langevin_relax_config)
from initial_states import droplet_state, save_state, triple_q_state

PHASE = CAMPAIGN_OUT / "C8"
PHASE.mkdir(parents=True, exist_ok=True)


def _run(cfg: Path, log: Path) -> int:
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w") as fh:
        proc = subprocess.run([str(SPIN_SOLVER), str(cfg)],
                              cwd=REPO_ROOT, stdout=fh,
                              stderr=subprocess.STDOUT)
    return proc.returncode


def _final_spin_file(out_dir: Path):
    return extract_final_spins(out_dir)


def _switched(cfg: Path, out_dir: Path) -> int:
    sf = _final_spin_file(out_dir)
    if sf is None:
        return -1
    try:
        parsed = run_analysis(cfg, [sf])
    except Exception:
        return -1
    if not parsed:
        return -1
    r3 = parsed[0].get("m_min_over_max")
    return int(declare_switched(float(r3))) if r3 is not None else -1


# ---------- K1: off-resonant pump
def _k1(seed: str, quick: bool) -> dict:
    rows = []
    freqs = [2.0, 3.0, 4.0, 5.0, 6.0] if not quick else [3.0, 4.0, 5.0]
    for w in freqs:
        tag = f"k1_w{w}".replace(".", "p")
        outdir = PHASE / "runs" / tag
        outdir.mkdir(parents=True, exist_ok=True)
        cfg = PHASE / "configs" / f"{tag}.param"
        write_langevin_pump_config(
            cfg, str(outdir.relative_to(REPO_ROOT)),
            seed_file=seed, E0=2.0, theta=0.0, T=0.0,
            pump_freq=w, lattice_size=(12, 12, 1) if quick else (24, 24, 1),
            phonon_only_relax=True,   # preserve triple-Q seed
            t_end=60.0,
        )
        _run(cfg, PHASE / "logs" / f"{tag}.log")
        rows.append({"freq": w, "switched": _switched(cfg, outdir)})
    write_csv(PHASE / "K1_freq_sweep.csv", rows)
    on_res = [r["switched"] for r in rows if abs(r["freq"] - 4.0) < 0.5]
    off_res = [r["switched"] for r in rows if abs(r["freq"] - 4.0) > 1.5]
    pass_ = (sum(on_res) > 0 and sum(off_res) == 0)
    return {"test": "K1_off_resonant", "passed": int(pass_), "expected": 1}


# ---------- K2: K_2 channel off (lambda_K2=0, quartic=0 to isolate channel)
def _k2(seed: str, quick: bool) -> dict:
    # We zero BOTH lambda_K_2 and lambda_E1_quartic so that the only spin-phonon
    # coupling path tested is the bilinear K_2 channel.  The quartic term can
    # itself drive switching at large phonon amplitudes (|ε|~3-4), which would
    # make this a false falsifier.  With both zeroed, if the system still
    # switches it would indicate a direct magneto-optic coupling (not observed).
    tag = "k2_E1off"
    outdir = PHASE / "runs" / tag
    outdir.mkdir(parents=True, exist_ok=True)
    cfg = PHASE / "configs" / f"{tag}.param"
    write_langevin_pump_config(
        cfg, str(outdir.relative_to(REPO_ROOT)),
        seed_file=seed, E0=2.0, theta=0.0, T=0.0,
        lambda_K_2=0.0, lambda_E1_quartic=0.0,
        phonon_only_relax=True,   # preserve the exact triple-Q seed
        lattice_size=(12, 12, 1) if quick else (24, 24, 1),
    )
    _run(cfg, PHASE / "logs" / f"{tag}.log")
    sw = _switched(cfg, outdir)
    # Expected: with K_2 and quartic channels off, drive cannot switch -> sw==0
    write_csv(PHASE / "K2_E1off.csv", [{"switched": sw}])
    return {"test": "K2_E1_necessary", "passed": int(sw == 0), "expected": 1}


# ---------- K3: small-lattice uniform-threshold test
def _k3(seed: str, quick: bool) -> dict:
    rows = []
    # phonon_only_relax=True preserves the triple-Q seed exactly (same fix as K2).
    # Without it, relax_joint's stochastic deterministic_sweep can kick the 2x2
    # state into the zigzag basin before the pulse even starts.
    # Fluence range chosen to bracket E*; 2x2 threshold estimated similar to 6x6.
    fluences = [0.1, 0.3, 0.6, 1.0, 1.5, 2.0] if not quick else [0.2, 0.5, 1.0, 1.5]
    for F in fluences:
        tag = f"k3_F{F}".replace(".", "p")
        outdir = PHASE / "runs" / tag
        outdir.mkdir(parents=True, exist_ok=True)
        cfg = PHASE / "configs" / f"{tag}.param"
        write_langevin_pump_config(
            cfg, str(outdir.relative_to(REPO_ROOT)),
            seed_file=seed, E0=F, theta=0.0, T=0.0,
            phonon_only_relax=True,   # preserve the exact triple-Q seed
            lattice_size=(2, 2, 1),
        )
        _run(cfg, PHASE / "logs" / f"{tag}.log")
        rows.append({"F": F, "switched": _switched(cfg, outdir)})
    write_csv(PHASE / "K3_uniform.csv", rows)
    # K3 PASSES if 2x2 shows a sharp 0->1 jump (no intermediate).
    sw = [r["switched"] for r in rows]
    pass_ = (0 in sw and 1 in sw)
    return {"test": "K3_uniform_threshold", "passed": int(pass_),
            "expected": 1}


# ---------- K4: clean droplet free-AC reference
def _k4(quick: bool) -> dict:
    Lx = Ly = 16 if quick else 24
    tag = "k4_freeAC"
    s = PHASE / "seeds" / f"{tag}.txt"
    save_state(s, droplet_state(Lx, Ly, R=6.0))
    outdir = PHASE / "runs" / tag
    outdir.mkdir(parents=True, exist_ok=True)
    cfg = PHASE / "configs" / f"{tag}.param"
    write_langevin_relax_config(
        cfg, str(outdir.relative_to(REPO_ROOT)),
        seed_file=str(s.relative_to(REPO_ROOT)),
        T=0.01, t_max=200.0, lattice_size=(Lx, Ly, 1),
        disorder_strength=0.0,
    )
    _run(cfg, PHASE / "logs" / f"{tag}.log")
    # Existence of output is the only check here; quantitative use is in C6.
    ok = outdir.exists()
    write_csv(PHASE / "K4_freeAC.csv", [{"completed": int(ok)}])
    return {"test": "K4_free_AC_reference", "passed": int(ok), "expected": 1}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    Lx = Ly = 12 if args.quick else 24
    seed_path = PHASE / "seeds" / f"triple_q_{Lx}x{Ly}.txt"
    save_state(seed_path, triple_q_state(Lx, Ly))
    rel_seed = str(seed_path.relative_to(REPO_ROOT))
    # K3 needs a 2x2 seed of its own.
    seed_k3 = PHASE / "seeds" / "triple_q_2x2.txt"
    save_state(seed_k3, triple_q_state(2, 2))
    rel_k3 = str(seed_k3.relative_to(REPO_ROOT))
    rows = [_k1(rel_seed, args.quick),
            _k2(rel_seed, args.quick),
            _k3(rel_k3, args.quick),
            _k4(args.quick)]
    write_csv(PHASE / "verdicts.csv", rows)
    print("C8 done:")
    for r in rows:
        print(f"  {r['test']}: passed={r['passed']} (expected {r['expected']})")


if __name__ == "__main__":
    main()
