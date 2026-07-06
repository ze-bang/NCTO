#!/bin/bash
#SBATCH --job-name=ncto_gneb_cr
#SBATCH --account=rrg-ybkim_cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=192
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --output=ncto_gneb_cr_%j.log
# ---------------------------------------------------------------------------
# Critical-radius GNEB tests (B1 finite-size scaling + B2 localized nucleus).
# Each GNEB is serial within a lattice, but the independent (L) and (L,half_len)
# tasks are launched concurrently as background processes on the one fat node.
# Resumable: each task caches its report.json and is skipped if present.
# ---------------------------------------------------------------------------
set -uo pipefail
REPO=/project/6003507/zhouzb79/NCTO
source "$REPO/NCTO_project/cluster_campaign/env_fir.sh"
cd "$REPO"
B1="$REPO/NCTO_project/scripts/gneb_finite_size_scan.py"
B2="$REPO/NCTO_project/scripts/gneb_localized_nucleus.py"

echo "==== $(date) B1 finite-size scan (L=18,36,54) + B2 localized nucleus ===="
# B1: one process per L (independent)
python3 "$B1" --lattices 18 > gneb_b1_L18.log 2>&1 &
python3 "$B1" --lattices 36 > gneb_b1_L36.log 2>&1 &
python3 "$B1" --lattices 54 > gneb_b1_L54.log 2>&1 &
# B2: one process per lattice; rectangular half-len set {6,10,14} so the final
# aggregation is fully cached (hl>=L/2 auto-skipped inside the script).
python3 "$B2" --lattices 36 --half-lens 6 10 14 > gneb_b2_L36.log 2>&1 &
python3 "$B2" --lattices 54 --half-lens 6 10 14 > gneb_b2_L54.log 2>&1 &
wait
echo "==== $(date) all GNEB tasks finished ===="

# rebuild combined summaries+plots from the cached per-task reports (instant)
python3 "$B1" --lattices 18 36 54
python3 "$B2" --lattices 36 54 --half-lens 6 10 14
echo "==== $(date) GNEB CRITICAL-RADIUS DONE ===="
