#!/bin/bash
#SBATCH --job-name=ncto_pol
#SBATCH --account=rrg-ybkim_cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=96
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --output=ncto_pol_%j.log
# Fine polarization/fluence sweep for E0c(theta), clean system, J7 mod off vs on.
set -uo pipefail
REPO=/project/6003507/zhouzb79/NCTO
source "$REPO/NCTO_project/cluster_campaign/env_fir.sh"
cd "$REPO"
python3 NCTO_project/scripts/run_campaign.py polarization --workers "${SLURM_CPUS_PER_TASK:-96}" \
  --theta-deg 0 7.5 15 22.5 30 37.5 45 52.5 60 67.5 75 82.5 90 \
  --e0-values 4 6 8 10 12 14 16 18 20 22 24 26 28 30 34 38
python3 NCTO_project/scripts/plot_polarization_threshold.py
echo "==== $(date) POLARIZATION DONE ===="
