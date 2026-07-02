"""Shared infrastructure for the NCTO Nature-Physics campaign.

This module is imported by every C*_*.py phase script in this directory.

Responsibilities:
  * Locate the repo root, the spin_solver and analyze_phonon_m_order
    binaries, the base param template, and the output root.
  * Provide a config-templating helper that takes a dict of
    placeholder -> value and writes a new .param file.
  * Provide a single-run launcher that calls spin_solver on a .param,
    captures the log, and returns success / failure.
  * Provide a small helper to run analyze_phonon_m_order on a list of
    spin files and return the parsed CSV rows.

The campaign defaults (NCTO baseline Hamiltonian, phonon, pump
envelope) live here so phase scripts only have to specify what they
sweep.
"""
from __future__ import annotations

import csv
import io
import os
import re
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence

# --------------------------------------------------------------------- paths
REPO_ROOT = Path(__file__).resolve().parents[3]
BUILD = REPO_ROOT / "build"
SPIN_SOLVER = BUILD / "spin_solver"
ANALYZE_BIN = BUILD / "analyze_phonon_m_order"
BASE_TEMPLATE = REPO_ROOT / "NCTO_project/configs/np_campaign_base.param"
CAMPAIGN_OUT = REPO_ROOT / "NCTO_project/np_campaign_out"
CAMPAIGN_OUT.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------- defaults
DEFAULTS: Dict[str, str] = {
    "LX": "6", "LY": "6",
    "OUTDIR": "",                       # required per-run
    "J": "-1.0", "K": "-6.0",
    "GAMMA": "8.0", "GAMMAP": "-3.5",
    # Calibrated from l18_effomega4_cycle15_phase_diagram (May 2025).
    # Most-sensitive switching point: lambda_K=0.035, J7=-0.0026 → E*=1.40.
    "J7": "-0.0026",
    "J2_A": "0.0", "J2_B": "0.0",
    "H_FIELD": "0.0", "ALPHA": "0.05", "T": "0.0",
    "OMEGA_E1": "4.0",
    # gamma_E1 calibrated via effective-frequency matching (not raw Q-factor).
    "GAMMA_E1": "0.0848826363157",
    # All E1 magneto-elastic channels off by default.
    "LAMBDA_J_0": "0.0", "LAMBDA_K_0": "0.0",
    "LAMBDA_GAMMA_0": "0.0", "LAMBDA_GAMMAP_0": "0.0",
    "LAMBDA_J_2": "0.0", "LAMBDA_K_2": "0.035",
    "LAMBDA_GAMMA_2": "0.0", "LAMBDA_GAMMAP_2": "0.0",
    # Pulse envelope calibrated to 15-cycle Gaussian at omega_E1=4.
    # pump_time=50: pulse peak well inside run window.
    # T_END=180: ~85 time units post-pulse for final state to lock in.
    "E0": "2.0", "PUMP_FREQ": "4.0",
    "PUMP_T0": "50.0", "PUMP_SIGMA": "15.0",
    "PUMP_PHASE": "0.0", "THETA": "0.0",
    "T_START": "-4.0", "T_END": "180.0",
    "PHONON_ONLY_RELAX": "false",
    "SEED_FILE": "",                    # required per-run
}


# --------------------------------------------------------------------- templating
def render_param(out_path: Path, overrides: Mapping[str, object],
                 template: Path = BASE_TEMPLATE) -> Path:
    """Write a .param file from BASE_TEMPLATE with placeholders replaced.

    Every placeholder __KEY__ in the template is replaced by either the
    user-supplied value or the DEFAULTS value.  Missing required keys
    (OUTDIR, SEED_FILE) raise ValueError so a phase script fails loudly
    rather than producing a malformed config.
    """
    text = template.read_text()
    merged: Dict[str, str] = {k: str(v) for k, v in DEFAULTS.items()}
    for k, v in overrides.items():
        merged[k] = str(v)
    if not merged["OUTDIR"]:
        raise ValueError("OUTDIR must be provided.")
    if not merged["SEED_FILE"]:
        raise ValueError("SEED_FILE must be provided.")
    for k, v in merged.items():
        text = text.replace(f"__{k}__", v)
    # Only flag tokens that look like real placeholders, not arbitrary
    # words in the template's comment header.
    leftover = re.findall(r"__[A-Z][A-Z0-9_]*__", text)
    if leftover:
        raise RuntimeError(f"Unsubstituted placeholders left in {out_path}: "
                           f"{leftover[:5]}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(text)
    return out_path


# --------------------------------------------------------------------- runners
@dataclass
class RunResult:
    cfg: Path
    outdir: Path
    ok: bool
    log: Path
    spins_final: Optional[Path] = None


def run_spin_solver(cfg: Path, fresh: bool = True) -> RunResult:
    """Execute spin_solver on a config file and return a RunResult."""
    outdir_line = next(
        ln for ln in cfg.read_text().splitlines()
        if ln.strip().startswith("output_dir")
    )
    outdir = REPO_ROOT / outdir_line.split("=", 1)[1].strip()
    if fresh and outdir.exists():
        shutil.rmtree(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    log = outdir / "run.log"
    with log.open("w") as fh:
        proc = subprocess.run(
            [str(SPIN_SOLVER), str(cfg)],
            cwd=REPO_ROOT, stdout=fh, stderr=subprocess.STDOUT,
        )
    spins = outdir / "sample_0" / "spins_final.txt"
    if not spins.exists():
        # pump_probe / MD runs write trajectory.h5 only; extract the
        # last frame to text for the analyzer.
        try:
            spins = extract_final_spins(outdir)
        except NameError:
            spins = None
        if spins is None:
            cand = list(outdir.glob("sample_0/spins_*.txt"))
            cand = [c for c in cand if c.name != "spins_initial.txt"]
            spins = cand[-1] if cand else None
    return RunResult(cfg=cfg, outdir=outdir, ok=(proc.returncode == 0),
                     log=log, spins_final=spins)


def run_analysis(cfg: Path, spin_files: Sequence[Path]) -> List[dict]:
    """Run analyze_phonon_m_order on a batch of spin files and parse CSV."""
    if not spin_files:
        return []
    args = [str(ANALYZE_BIN), str(cfg)] + [str(p) for p in spin_files]
    proc = subprocess.run(args, cwd=REPO_ROOT, capture_output=True, text=True)
    if proc.returncode != 0:
        return []
    # The analyzer prints lattice/Hamiltonian init lines before its CSV
    # output.  Locate the actual header line (starts with 'spin_file,').
    lines = proc.stdout.splitlines()
    start = next((i for i, ln in enumerate(lines)
                  if ln.startswith("spin_file,")), None)
    if start is None:
        return []
    reader = csv.DictReader(io.StringIO("\n".join(lines[start:])))
    rows: List[dict] = []
    for row in reader:
        for key in ("S_M1", "S_M2", "S_M3", "m_3Q",
                    "m_zigzag", "m_min_over_max"):
            if key in row:
                row[key] = float(row[key])
        rows.append(row)
    return rows


# --------------------------------------------------------------------- bookkeeping
def write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    if not rows:
        path.write_text("")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)


def declare_switched(r3: float, threshold: float = 0.2) -> bool:
    return r3 < threshold


def extract_final_spins(out_dir: Path) -> Optional[Path]:
    """Locate the final spin configuration written by spin_solver.

    The solver writes trajectories to `sample_0/trajectory.h5` with the
    spins under `trajectory/spins[T, N, 3]`.  This helper extracts the
    last frame to `sample_0/spins_final.txt` (compatible with
    `analyze_phonon_m_order`) and returns the path.  If a plain text
    `spins_*.txt` already exists it is returned directly.
    """
    sub = out_dir / "sample_0"
    if not sub.exists():
        sub = out_dir
    txts = sorted(sub.glob("spins_*.txt"))
    txts = [t for t in txts if t.name != "spins_initial.txt"]
    if txts:
        return txts[-1]
    h5 = sub / "trajectory.h5"
    if not h5.exists():
        return None
    try:
        import h5py
        import numpy as np
        with h5py.File(h5, "r") as h:
            if "trajectory/spins" not in h:
                return None
            spins = h["trajectory/spins"][-1, :, :]
    except Exception:
        return None
    out = sub / "spins_final.txt"
    np.savetxt(out, spins, fmt="%.10f")
    return out


def label(prefix: str, **kw) -> str:
    """Build a filesystem-safe label from key=value pairs."""
    parts = [prefix] + [f"{k}{v}".replace(".", "p").replace("-", "m")
                        for k, v in kw.items()]
    return "_".join(parts)


# --------------------------------------------------------------------- seeding
SEED_CACHE: Dict[str, Path] = {}


def ensure_3Q_seed(tag: str = "default",
                   overrides: Optional[Mapping[str, object]] = None) -> Path:
    """Run an SA job to produce a 3Q seed and cache the spins file."""
    if tag in SEED_CACHE and SEED_CACHE[tag].exists():
        return SEED_CACHE[tag]
    out = CAMPAIGN_OUT / "seeds" / tag
    out.mkdir(parents=True, exist_ok=True)
    cfg_path = out / "sa.param"
    # SA-specific config: bypass the pump-probe template and write directly.
    over = dict(overrides or {})
    text = f"""system = NCTO
simulation_mode = SA
lattice_size = {over.get('LX', 6)},{over.get('LY', 6)},1
output_dir = {out.relative_to(REPO_ROOT)}
num_trials = 1
J = {over.get('J', -1.0)}
K = {over.get('K', -6.0)}
Gamma = {over.get('GAMMA', 8.0)}
Gammap = {over.get('GAMMAP', -3.5)}
J7 = {over.get('J7', -0.10)}
field_strength = 0.0
field_direction = 0,0,1
alpha_gilbert = 0.0
langevin_temperature = 0.0
omega_E1 = {over.get('OMEGA_E1', 4.0)}
gamma_E1 = {over.get('GAMMA_E1', 0.2543)}
lambda_E1_quartic = 0.0
Z_star = 1.0
lambda_E1_K_2 = 0.0
relax_phonons = true
adiabatic_phonons = false
T_start = 5.0
T_end = 0.005
T_zero = true
annealing_steps = 30000
cooling_rate = 0.95
overrelaxation_rate = 5
n_deterministics = 5000
"""
    cfg_path.write_text(text)
    log = out / "sa.log"
    with log.open("w") as fh:
        subprocess.run([str(SPIN_SOLVER), str(cfg_path)],
                       cwd=REPO_ROOT, stdout=fh, stderr=subprocess.STDOUT)
    seed = out / "sample_0" / "spins_T=0.txt"
    SEED_CACHE[tag] = seed
    return seed


# ---- analytic 3Q seed (for metastable-seed campaigns, e.g. Songvilay) ------
_ANALYTIC_3Q_CACHE: Dict[str, Path] = {}


def ensure_analytic_3Q_seed(Lx: int = 6, Ly: int = 6) -> Path:
    """Write the analytic triple-Q state to a file and return the path.

    Unlike ensure_3Q_seed(), this does NOT run SA.  The analytic
    triple_q_state() is used directly, making it suitable for parameter
    regimes where the 3Q state is metastable (not the classical GS).
    Requires phonon_only_relax=True in pump-probe configs to prevent the
    spin dynamics from relaxing the seed before the pump fires.
    """
    key = f"{Lx}x{Ly}"
    if key in _ANALYTIC_3Q_CACHE and _ANALYTIC_3Q_CACHE[key].exists():
        return _ANALYTIC_3Q_CACHE[key]
    import sys as _sys
    _sys.path.insert(0, str(Path(__file__).parent))
    import initial_states as _is
    out = CAMPAIGN_OUT / "seeds" / f"analytic_3Q_{key}"
    out.mkdir(parents=True, exist_ok=True)
    seed_path = out / "initial_spins.txt"
    spins = _is.triple_q_state(Lx, Ly)
    _is.save_state(seed_path, spins)
    _ANALYTIC_3Q_CACHE[key] = seed_path
    return seed_path
