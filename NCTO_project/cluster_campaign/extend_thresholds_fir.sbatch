#!/bin/bash
#SBATCH --job-name=ncto_extE0
#SBATCH --account=rrg-ybkim_cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=192
#SBATCH --mem=128G
#SBATCH --time=06:00:00
#SBATCH --output=ncto_extE0_%j.log
# ---------------------------------------------------------------------------
# Extend the E0 grids for drive-pd and switching to bracket the (higher) L=36
# switching threshold.  Resumable: existing E0 points are skipped, only the new
# high-E0 runs execute.
# ---------------------------------------------------------------------------
set -uo pipefail
REPO=/project/6003507/zhouzb79/NCTO
source "$REPO/NCTO_project/cluster_campaign/env_fir.sh"
cd "$REPO"
W="${SLURM_CPUS_PER_TASK:-192}"
CAMP="$REPO/NCTO_project/scripts/run_campaign.py"

echo "==== $(date) drive-pd extended E0 ===="
python3 "$CAMP" drive-pd --all-channels --workers "$W" \
  --e0-values 1 2 3 4 5 6 7 8 10 12 16 18 20 22 25 30 35 40

echo "==== $(date) switching extended E0 ===="
python3 "$CAMP" switching --workers "$W" \
  --e0-values 0 6 8 9 10 10.5 11 11.5 12 12.5 13 13.5 14 15 16 17 18 19 20 22 25 30 35 40 50

echo "==== $(date) EXTENDED-E0 DONE ===="
