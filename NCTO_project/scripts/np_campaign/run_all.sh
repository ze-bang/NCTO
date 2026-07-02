#!/usr/bin/env bash
# Master driver for the Nature-Physics computational campaign.
# Run from the repository root.  Each Python phase respects --quick.
#
#   bash NCTO_project/scripts/np_campaign/run_all.sh quick
#   bash NCTO_project/scripts/np_campaign/run_all.sh full
#
# Critical path:  C1 -> C3 -> C5 -> C7
# Parallel-safe:  C2, C4, C8 can run independently after C1's seed exists.

set -euo pipefail
MODE="${1:-quick}"
case "$MODE" in
  quick) FLAG="--quick" ;;
  full)  FLAG="" ;;
  *)     echo "Usage: $0 {quick|full}" ; exit 1 ;;
esac

cd "$(dirname "$0")/../../.."   # repo root
HERE="NCTO_project/scripts/np_campaign"

run() {
    echo "================================================================"
    echo "==> $1"
    echo "================================================================"
    python3 "$HERE/$1" $FLAG || echo "  $1 returned non-zero; continuing."
}

run C1_param_robustness.py
run C2_barrier_string.py
run C3_extract_coarse_grained.py
run C4_stress_tests.py
run C5_kjma_forward.py
run C6_recovery_competition.py
run C7_observables.py
run C8_kill_alternatives.py
run C9_geometry_predictions.py

echo
echo "==> figures"
(cd "$HERE" && python3 make_figures.py --all) \
    || echo "  make_figures.py returned non-zero; continuing."

echo
echo "All phases done.  Outputs in NCTO_project/np_campaign_out/."
echo "Figures in NCTO_project/np_campaign_out/figs/."
