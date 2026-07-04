#!/bin/bash
# NCTO tuned-Kruger campaign (consolidated driver, L=36).
# Static 3Q/ZZ baseline + Fig 1 (phase cube) + Figs 2c/4 (L36 err10 switching)
# + Fig `pol` (polarization/fluence, J7 phonon off vs on).
#
# Everything runs through ONE driver, run_campaign.py, sharing ncto_common.py
# (single source of the Hamiltonian and the corrected signed-Grueneisen coupling
# lambda_{X,2}=(X/K) lambda_{K,2}).  Fresh-clone friendly and resumable: the
# solver skips existing outputs, so re-running after a walltime kill resumes.
#   Env overrides: WORKERS, RUN_POL (default 1).
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"   # repo root (NCTO/)
cd "$ROOT"
WORKERS="${WORKERS:-$(nproc)}"
export OMP_NUM_THREADS=1                  # one thread per solver process
EXE="$ROOT/build/spin_solver"
CAMP="$ROOT/NCTO_project/scripts/run_campaign.py"
ANA36="$ROOT/NCTO_project/tuned_kruger_campaign/pinning_switching_crosscheck_L36/analysis"
PD_ANA="$ROOT/NCTO_project/tuned_kruger_campaign/phase_diagram_allchan/analysis"
STAT_ANA="$ROOT/NCTO_project/tuned_kruger_campaign/static_phase_diagram/analysis"

echo "==== $(date) START (workers=$WORKERS) repo=$ROOT ===="
[ -x "$EXE" ] || { echo "FATAL: $EXE not found/executable -- build first (build.sbatch)"; exit 1; }
[ -x "$ROOT/build/gneb_field_eval" ] || echo "WARN: build/gneb_field_eval missing -- static-pd energies will be skipped"

# ---------- Step 0: static (no-drive) 3Q-vs-ZZ baseline ----------
echo "==== $(date) STEP 0: static 3Q/ZZ phase diagram (J7 0..-0.7) ===="
python3 "$CAMP" static-pd --workers "$WORKERS"

# ---------- Fig 1: phase-diagram cube (462 x L36) ----------
echo "==== $(date) STEP 1: drive phase diagram (--all-channels) ===="
python3 "$CAMP" drive-pd --all-channels --workers "$WORKERS"

# ---------- Figs 2c + 4: L36 err10 switching (2646 MD + 2646 quench) ----------
echo "==== $(date) STEP 2: L36 err10 switching ===="
python3 "$CAMP" switching --workers "$WORKERS"

# ---------- Fig pol: polarization vs fluence (clean, J7 phonon off vs on) ----------
if [ "${RUN_POL:-1}" = "1" ]; then
  echo "==== $(date) STEP 3: clean polarization/fluence study ===="
  python3 "$CAMP" polarization --workers "$WORKERS"
fi

echo "==== $(date) ALL DATA DONE ===="
echo "Summary outputs:"
echo "  $STAT_ANA/static_phase_diagram.csv"
echo "  $PD_ANA/phase_summary_full.csv"
echo "  $ANA36/L36_kred_s0p500_hw2p00_allJKGG_err10_drive_crosscheck_summary.json"
echo "Run the plotters (cheap; locally is fine) to regenerate the figures -- see README.md."
