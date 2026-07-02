#!/bin/bash
# Corrected-coupling recompute: Fig 1 (phase cube) + Figs 2c/4 (L36 err10).
# Fresh-clone friendly: everything regenerates; the solver skips existing
# outputs, so re-running after a walltime kill simply resumes.
#
# Uniform (signed) Grueneisen coupling lambda_{X,2}=(X/K) lambda_{K,2} is already
# set in the driver scripts.  Env overrides: WORKERS.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"   # repo root
cd "$ROOT"
WORKERS="${WORKERS:-$(nproc)}"
EXE="$ROOT/build/spin_solver"
ANA36="$ROOT/NCTO_project/tuned_kruger_campaign/pinning_switching_crosscheck_L36/analysis"
PD_ANA="$ROOT/NCTO_project/tuned_kruger_campaign/phase_diagram_allchan/analysis"

echo "==== $(date) START (workers=$WORKERS) repo=$ROOT ===="
[ -x "$EXE" ] || { echo "FATAL: $EXE not found/executable -- build first (build.sbatch)"; exit 1; }

# ---------- Fig 1: phase-diagram cube (462 x L18) ----------
echo "==== $(date) STEP 1: phase diagram (--all-channels) ===="
python3 "$ROOT/NCTO_project/scripts/run_tuned_kruger_phase_diagram.py" \
  --all-channels --workers "$WORKERS"
# The cube plotter (run later) expects phase_summary_full.csv:
[ -f "$PD_ANA/phase_summary.csv" ] && cp "$PD_ANA/phase_summary.csv" "$PD_ANA/phase_summary_full.csv"
echo "==== $(date) STEP 1 done ===="

# ---------- Figs 2c + 4: L36 err10 switching (2646 MD + 2646 quench) ----------
echo "==== $(date) STEP 2: L36 err10 switching ===="
python3 "$ROOT/NCTO_project/scripts/cross_validate_pinned_switching_l36.py" \
  --lattice 36 --dtype kred --strength 0.5 --half-width 2.0 \
  --disorder-mode global-zero-k --seeds 25 \
  --sigma-k-values 0.0 0.1 0.2 0.3 0.4 0.5 \
  --e0-values 0 6 8 9 10 10.5 11 11.5 12 12.5 13 13.5 14 15 16 17 18 19 20 22 25 \
  --output-suffix allJKGG_err10 --workers "$WORKERS"
echo "==== $(date) STEP 2 done ===="

# ---------- Polarization vs fluence study (clean, no extended defect) ----------
# J7 ring-exchange phonon coupling OFF vs ON at a sensible strength.  Set
# RUN_POL=0 to skip.
if [ "${RUN_POL:-1}" = "1" ]; then
  echo "==== $(date) STEP 3: clean polarization/fluence study (J7 phonon on vs off) ===="
  python3 "$ROOT/NCTO_project/scripts/run_polarization_fluence_study.py" \
    --lattice 36 --workers "$WORKERS"
  echo "==== $(date) STEP 3 done ===="
fi

echo "==== $(date) ALL DATA DONE ===="
echo "Summary outputs:"
echo "  $PD_ANA/phase_summary.csv"
echo "  $ANA36/L36_kred_s0p500_hw2p00_allJKGG_err10_drive_crosscheck_summary.json"
echo "Run the plotters (cheap; locally is fine) to regenerate the figures -- see README.md."
